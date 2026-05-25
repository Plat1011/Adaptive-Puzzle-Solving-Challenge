from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np
from common import to_jsonable
CONTENT_EMPTY = 0
CONTENT_NUM = 1
CONTENT_COLOR = 2
CONTENT_MASKED = 3
def _l1(a: List[float], b: List[float]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])
def _axis_scales(positions: List[List[float]]) -> List[float]:
    arr = np.asarray(positions, dtype=np.float64)
    scales = [1.0, 1.0, 1.0]
    for ax in range(3):
        vals = np.unique(np.round(arr[:, ax], 6))
        if vals.size < 2:
            scales[ax] = 0.0
            continue
        gaps = np.diff(vals)
        gaps = gaps[gaps > 1e-09]
        if gaps.size == 0:
            scales[ax] = 0.0
            continue
        scales[ax] = 1.0 / float(np.min(gaps))
    return scales
def _target_index_by_num(env) -> Dict[int, int]:
    enc = env.encode_state(env.solved_state())
    out: Dict[int, int] = {}
    for i, (t, v) in enumerate(zip(enc['target_types'], enc['target_values'])):
        if int(t) == CONTENT_NUM:
            out[int(v)] = i
    return out
def _color_target_positions(env) -> Dict[int, List[List[float]]]:
    enc = env.encode_state(env.solved_state())
    out: Dict[int, List[List[float]]] = {}
    for pos, t, v in zip(enc['positions'], enc['target_types'], enc['target_values']):
        if int(t) == CONTENT_COLOR:
            out.setdefault(int(v), []).append([float(pos[0]), float(pos[1]), float(pos[2])])
    return out
def make_heuristic(env):
    num_target_idx = _target_index_by_num(env)
    color_targets = _color_target_positions(env)
    enc_solved = env.encode_state(env.solved_state())
    raw_positions = [[float(p[0]), float(p[1]), float(p[2])] for p in enc_solved['positions']]
    scales = _axis_scales(raw_positions)
    positions = [[p[0] * scales[0], p[1] * scales[1], p[2] * scales[2]] for p in raw_positions]
    def h(state: Any) -> float:
        enc = env.encode_state(state)
        ct = enc['content_types']
        cv = enc['content_values']
        tt = enc['target_types']
        tv = enc['target_values']
        total = 0.0
        for i, (t, v) in enumerate(zip(ct, cv)):
            t = int(t)
            v = int(v)
            if t != CONTENT_NUM:
                continue
            if int(tt[i]) == CONTENT_NUM and int(tv[i]) == v:
                continue
            tgt_idx = num_target_idx.get(v)
            if tgt_idx is None:
                total += 1.0
                continue
            total += _l1(positions[i], positions[tgt_idx])
        color_positions: Dict[int, List[List[float]]] = {}
        for i, (t, v) in enumerate(zip(ct, cv)):
            if int(t) == CONTENT_COLOR:
                color_positions.setdefault(int(v), []).append(positions[i])
        for color, srcs in color_positions.items():
            tgts = color_targets.get(color)
            if not tgts:
                continue
            srcs_remaining = list(srcs)
            tgts_remaining = list(tgts)
            while srcs_remaining and tgts_remaining:
                best_i = 0
                best_j = 0
                best_d = float('inf')
                for ii, s in enumerate(srcs_remaining):
                    for jj, tg in enumerate(tgts_remaining):
                        d = _l1(s, tg)
                        if d < best_d:
                            best_d = d
                            best_i = ii
                            best_j = jj
                total += best_d
                srcs_remaining.pop(best_i)
                tgts_remaining.pop(best_j)
        return float(total)
    return h
def make_batched_heuristic(env):
    h = make_heuristic(env)
    def hb(states: List[Any]) -> np.ndarray:
        return np.fromiter((h(s) for s in states), dtype=np.float32, count=len(states))
    return hb
