from __future__ import annotations
import argparse
import os
import random
import statistics
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SOLUTION = ROOT / 'solution'
BASELINE = ROOT / 'baseline'
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', default='game_15_2d')
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--scrambles', default='', help='comma-separated scramble lengths to test (default = env defaults)')
    ap.add_argument('--max_states', type=int, default=200000)
    ap.add_argument('--time_limit', type=float, default=5.0, help='seconds per instance')
    args = ap.parse_args()
    os.environ['ENV_ID'] = args.env
    sys.path.insert(0, str(SOLUTION))
    sys.path.insert(0, str(BASELINE))
    import gym
    from bidir_bfs import bidir_bfs
    env = gym.make_env()
    print(f'env_id: {gym.ENV_ID}, n={args.n}, time_limit={args.time_limit}s/inst, max_states={args.max_states}')
    if args.scrambles.strip():
        lengths = [int(x) for x in args.scrambles.split(',')]
    else:
        lengths = list(gym.get_default_scramble_lengths())
    print(f'scrambles to test: {lengths}')
    rng = random.Random(0)
    solved = 0
    move_ratios = []
    fail_lens = []
    for i in range(args.n):
        L = lengths[i % len(lengths)]
        state, gen_actions = env.scramble(length=L, seed=rng.randint(0, 10 ** 9))
        t = time.time()
        sol = bidir_bfs(env, state, deadline=t + args.time_limit, max_states=args.max_states)
        dt = time.time() - t
        if sol is None:
            fail_lens.append(L)
            continue
        env.set_state(state)
        for a in sol:
            if a not in env.valid_actions():
                print(f'  inst {i} (L={L}): invalid action {a}')
                sol = None
                break
            env.step(a)
        if sol is None or not env.is_solved():
            fail_lens.append(L)
            continue
        solved += 1
        move_ratios.append(len(sol) / max(1, len(gen_actions)))
        if i < 5 or i == args.n - 1:
            print(f'  inst {i} (L={L}): solved in {len(sol)} moves, {dt:.2f}s, ratio {move_ratios[-1]:.3f}')
    print(f'\nsolved {solved}/{args.n}')
    if move_ratios:
        print(f'  mean moves/baseline = {statistics.mean(move_ratios):.3f}')
        per_inst_scores = [min(2.0, 1.0 / r) for r in move_ratios]
        print(f'  mean per-instance score = {statistics.mean(per_inst_scores):.3f}')
    if fail_lens:
        print(f'  failed scrambles: {fail_lens}')
if __name__ == '__main__':
    main()
