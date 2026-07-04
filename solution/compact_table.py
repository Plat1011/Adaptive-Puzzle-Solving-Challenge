from __future__ import annotations
import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from common import state_key, to_jsonable

KEYS_FILE = 'gt_keys.npy'
PARENTS_FILE = 'gt_parents.npy'
ACTIONS_FILE = 'gt_actions.npy'
DEPTHS_FILE = 'gt_depths.npy'
VOCAB_FILE = 'gt_vocab.json'
NO_PARENT = np.uint64(0xFFFFFFFFFFFFFFFF)


def _rss_mb() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return 0.0


def hash_key(state) -> int:
    k = state_key(state)
    if isinstance(k, bytes):
        b = k
    elif isinstance(k, str):
        b = k.encode('utf-8')
    elif isinstance(k, tuple):
        parts = []
        for item in k:
            if isinstance(item, tuple):
                for x in item:
                    parts.append(x.encode('utf-8') if isinstance(x, str) else (x if isinstance(x, (bytes, bytearray)) else str(x).encode()))
            else:
                parts.append(item.encode('utf-8') if isinstance(item, str) else (item if isinstance(item, (bytes, bytearray)) else str(item).encode()))
        b = b'|'.join(parts)
    else:
        b = str(k).encode('utf-8')
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), 'little')


def build_compact_table(env, deadline, max_states=5000000, max_rss_mb=8192.0, checkpoint_workdir=None, checkpoint_every_sec=300.0, nn_sample_every=0, nn_sample_cap=300000):
    env.reset()
    goal_state = to_jsonable(env.get_state())
    goal_h = hash_key(goal_state)
    table = {goal_h: (int(NO_PARENT), 0, 0)}
    cur_payload = {goal_h: goal_state}
    vocab = {}
    depth = 0
    last_rss_check = 0
    last_checkpoint = time.time()
    nn_X = []
    nn_y = []
    nn_counter = 0
    nn_n_cells = None
    if nn_sample_every > 0:
        try:
            v0 = np.asarray(env.encode_state(goal_state)['content_values'], dtype=np.int64)
            nn_X.append(v0)
            nn_y.append(0)
            nn_n_cells = v0.shape[0]
        except Exception:
            pass
    while cur_payload and time.time() < deadline and len(table) < max_states:
        next_payload = {}
        for h, state in cur_payload.items():
            if time.time() >= deadline or len(table) >= max_states:
                break
            if max_rss_mb and len(table) - last_rss_check >= 50000:
                last_rss_check = len(table)
                if _rss_mb() > max_rss_mb:
                    packed = _pack(table, vocab, depth)
                    if nn_X:
                        return packed + (np.stack(nn_X).astype(np.int64), np.asarray(nn_y, dtype=np.float32))
                    return packed + (None, None)
            if checkpoint_workdir is not None and time.time() - last_checkpoint >= checkpoint_every_sec:
                _save_checkpoint(checkpoint_workdir, table, vocab, depth)
                last_checkpoint = time.time()
            try:
                env.set_state(state)
                acts = env.valid_actions()
            except Exception:
                continue
            for a in acts:
                try:
                    env.set_state(state)
                    env.step(a)
                    ns = to_jsonable(env.get_state())
                except Exception:
                    continue
                try:
                    nh = hash_key(ns)
                except Exception:
                    continue
                if nh in table:
                    continue
                try:
                    edge = env.inverse_action(a)
                except Exception:
                    edge = a
                if not isinstance(edge, str):
                    edge = str(edge)
                if edge not in vocab:
                    if len(vocab) >= 65535:
                        edge = next(iter(vocab))
                    else:
                        vocab[edge] = len(vocab)
                table[nh] = (h, vocab[edge], depth + 1)
                next_payload[nh] = ns
                if nn_sample_every > 0:
                    nn_counter += 1
                    if nn_counter % nn_sample_every == 0 and len(nn_X) < nn_sample_cap:
                        try:
                            vec = np.asarray(env.encode_state(ns)['content_values'], dtype=np.int64)
                            if nn_n_cells is None:
                                nn_n_cells = vec.shape[0]
                            if vec.shape[0] == nn_n_cells:
                                nn_X.append(vec)
                                nn_y.append(depth + 1)
                        except Exception:
                            pass
                if len(table) >= max_states:
                    break
        if not next_payload:
            break
        cur_payload = next_payload
        depth += 1
    packed = _pack(table, vocab, depth)
    if nn_X:
        return packed + (np.stack(nn_X).astype(np.int64), np.asarray(nn_y, dtype=np.float32))
    return packed + (None, None)


def _pack(table, vocab, depth):
    n = len(table)
    keys_raw = np.empty(n, dtype=np.uint64)
    parents_raw = np.empty(n, dtype=np.uint64)
    actions_raw = np.empty(n, dtype=np.uint16)
    depths_raw = np.empty(n, dtype=np.uint16)
    for i, (h, v) in enumerate(table.items()):
        keys_raw[i] = h
        parents_raw[i] = v[0]
        actions_raw[i] = v[1]
        d = v[2]
        depths_raw[i] = d if d < 65535 else 65535
    order = np.argsort(keys_raw, kind='stable')
    return keys_raw[order], parents_raw[order], actions_raw[order], depths_raw[order], [a for a, _ in sorted(vocab.items(), key=lambda kv: kv[1])], depth


def _save_checkpoint(workdir, table, vocab, depth):
    try:
        k, p, a, d, voc, _ = _pack(table, vocab, depth)
        save_compact(workdir, k, p, a, d, voc)
    except Exception:
        pass


def save_compact(workdir, keys, parents, actions, depths, vocab):
    np.save(os.path.join(workdir, KEYS_FILE), keys)
    np.save(os.path.join(workdir, PARENTS_FILE), parents)
    np.save(os.path.join(workdir, ACTIONS_FILE), actions)
    np.save(os.path.join(workdir, DEPTHS_FILE), depths)
    with open(os.path.join(workdir, VOCAB_FILE), 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False)


def load_compact(workdir='.'):
    kpath = os.path.join(workdir, KEYS_FILE)
    if not os.path.exists(kpath):
        return None
    try:
        keys = np.load(kpath, mmap_mode='r')
        parents = np.load(os.path.join(workdir, PARENTS_FILE), mmap_mode='r')
        actions = np.load(os.path.join(workdir, ACTIONS_FILE), mmap_mode='r')
        depths = np.load(os.path.join(workdir, DEPTHS_FILE), mmap_mode='r')
        with open(os.path.join(workdir, VOCAB_FILE), encoding='utf-8') as f:
            vocab = json.load(f)
    except Exception:
        return None
    return (keys, parents, actions, depths, vocab)


def lookup(keys, h):
    idx = int(np.searchsorted(keys, np.uint64(h)))
    if idx < len(keys) and int(keys[idx]) == h:
        return idx
    return -1


def reconstruct(compact, start_hash, max_len=5000):
    keys, parents, actions, depths, vocab = compact
    idx = lookup(keys, start_hash)
    if idx < 0:
        return None
    out = []
    cur = start_hash
    for _ in range(max_len):
        idx = lookup(keys, cur)
        if idx < 0:
            return out
        if int(depths[idx]) == 0:
            return out
        ai = int(actions[idx])
        if 0 <= ai < len(vocab):
            out.append(vocab[ai])
        ph = int(parents[idx])
        if ph == int(NO_PARENT):
            return out
        cur = ph
    return out


def forward_to_compact(env, initial_state, compact, deadline, max_states=80000):
    keys = compact[0]
    start = to_jsonable(initial_state)
    sh = hash_key(start)
    if lookup(keys, sh) >= 0:
        return reconstruct(compact, sh)
    fwd = {sh: (0, '')}
    cur_payload = {sh: start}
    while cur_payload and time.time() < deadline and len(fwd) < max_states:
        next_payload = {}
        for h, state in cur_payload.items():
            if time.time() >= deadline or len(fwd) >= max_states:
                break
            try:
                env.set_state(state)
                acts = env.valid_actions()
            except Exception:
                continue
            for a in acts:
                try:
                    env.set_state(state)
                    env.step(a)
                    ns = to_jsonable(env.get_state())
                except Exception:
                    continue
                try:
                    nh = hash_key(ns)
                except Exception:
                    continue
                if nh in fwd:
                    continue
                fwd[nh] = (h, a)
                if lookup(keys, nh) >= 0:
                    fwd_actions = []
                    cur = nh
                    while True:
                        pk, ac = fwd.get(cur, (0, ''))
                        if cur == sh or not ac:
                            break
                        fwd_actions.append(ac)
                        cur = pk
                    fwd_actions.reverse()
                    back = reconstruct(compact, nh) or []
                    return fwd_actions + back
                next_payload[nh] = ns
                if len(fwd) >= max_states:
                    break
        if not next_payload:
            break
        cur_payload = next_payload
    return None
