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
TIME_LIMIT_DEFAULT = 50 * 60
SAFETY_MARGIN = 90
PROFILE_PATH = 'profile.json'
GF2_PATH = 'gf2.npz'
GOAL_TABLE_MAX_STATES = 12_000_000
GOAL_TABLE_MAX_RSS_MB = 18000.0
CHECKPOINT_EVERY_SEC = 180.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--time_limit', type=int, default=int(os.environ.get('TRAIN_TIME_LIMIT', TIME_LIMIT_DEFAULT)))
    ap.add_argument('--seed', type=int, default=239)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    start = time.time()
    deadline = start + args.time_limit - SAFETY_MARGIN
    env = gym.make_env()
    try:
        profile = probe(env)
    except Exception as e:
        print(f'probe failed: {repr(e)}')
        profile = {'env_id': getattr(gym, 'ENV_ID', '?'), 'action_types': [], 'branching_mean': 0.0, 'branching_max': 0, 'n_cells': 0, 'involution_actions': False, 'binary_cells': False, 'n_unique_actions_at_start': 0}
    try:
        with open(PROFILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(profile, f, indent=2)
    except Exception:
        pass
    if is_toggle_only(profile):
        try:
            A, actions = build_action_matrix(env)
            np.savez(GF2_PATH, A=A, actions=np.array(actions))
        except Exception as e:
            print(f'GF(2) failed: {repr(e)}')
        return
    try:
        keys, parents, actions, depths, vocab, max_depth, _x, _y = build_compact_table(
            env, deadline=deadline, max_states=GOAL_TABLE_MAX_STATES, max_rss_mb=GOAL_TABLE_MAX_RSS_MB,
            checkpoint_workdir='.', checkpoint_every_sec=CHECKPOINT_EVERY_SEC,
        )
        save_compact('.', keys, parents, actions, depths, vocab)
    except Exception as e:
        print(f'BFS failed: {repr(e)}')
    print(f'train done in {time.time() - start:.1f}s')


if __name__ == '__main__':
    main()
