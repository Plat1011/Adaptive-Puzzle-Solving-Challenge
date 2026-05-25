from __future__ import annotations
import argparse
import json
import os
import random
import time
import numpy as np
import gym
from detect import probe, is_toggle_only
from gf2_solver import build_action_matrix
from compact_table import build_compact_table, save_compact
from nn_heuristic import train_nn
TIME_LIMIT_DEFAULT = 50 * 60
SAFETY_MARGIN = 90
PROFILE_PATH = 'profile.json'
GF2_PATH = 'gf2.npz'
GOAL_TABLE_MAX_STATES = 8_000_000
GOAL_TABLE_MAX_RSS_MB = 14000.0
CHECKPOINT_EVERY_SEC = 240.0
NN_BUDGET_FRACTION = 0.4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--time_limit', type=int, default=int(os.environ.get('TRAIN_TIME_LIMIT', TIME_LIMIT_DEFAULT)))
    ap.add_argument('--seed', type=int, default=239)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    start = time.time()
    deadline = start + args.time_limit - SAFETY_MARGIN
    env = gym.make_env()
    print(f"env_id={getattr(gym, 'ENV_ID', '?')}")
    print('probing env...')
    try:
        profile = probe(env)
    except Exception as e:
        print(f'probe failed: {repr(e)}')
        profile = {'env_id': getattr(gym, 'ENV_ID', '?'), 'action_types': [], 'branching_mean': 0.0, 'branching_max': 0, 'n_cells': 0, 'involution_actions': False, 'binary_cells': False, 'n_unique_actions_at_start': 0}
    try:
        with open(PROFILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(profile, f, indent=2)
    except Exception as e:
        print(f'profile save failed: {repr(e)}')
    print(f'profile: {json.dumps(profile, indent=2)}')
    if is_toggle_only(profile):
        print('toggle-only puzzle -> building GF(2) action matrix')
        try:
            A, actions = build_action_matrix(env)
            np.savez(GF2_PATH, A=A, actions=np.array(actions))
            print(f'  saved {GF2_PATH}: A={A.shape}, actions={len(actions)}')
        except Exception as e:
            print(f'GF(2) build failed: {repr(e)}')
        return
    bfs_deadline = deadline
    print(f'building compact goal-distance table (until t+{bfs_deadline - start:.0f}s)...')
    t0 = time.time()
    try:
        keys, parents, actions, depths, vocab, max_depth = build_compact_table(
            env, deadline=bfs_deadline, max_states=GOAL_TABLE_MAX_STATES, max_rss_mb=GOAL_TABLE_MAX_RSS_MB,
            checkpoint_workdir='.', checkpoint_every_sec=CHECKPOINT_EVERY_SEC,
        )
        print(f'  table: {len(keys)} states, depth {max_depth}, vocab {len(vocab)}, built in {time.time() - t0:.1f}s')
        save_compact('.', keys, parents, actions, depths, vocab)
        print(f'  saved compact arrays')
    except Exception as e:
        print(f'BFS table build failed: {repr(e)}')
    nn_deadline = deadline
    if time.time() + 10 < nn_deadline:
        print(f'training NN heuristic until t+{nn_deadline - start:.0f}s...')
        try:
            ok = train_nn(env, deadline=nn_deadline, total_samples=300_000, max_walk=120, batch_size=1024, lr=1e-3, emb_dim=16, hidden=256, seed=args.seed)
            print(f'  nn training {"ok" if ok else "skipped"}')
        except Exception as e:
            print(f'NN training failed: {repr(e)}')
    print(f'train.py done in {time.time() - start:.1f}s')


if __name__ == '__main__':
    main()
