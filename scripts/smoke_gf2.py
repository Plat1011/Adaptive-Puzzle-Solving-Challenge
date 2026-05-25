from __future__ import annotations
import json
import os
import random
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SOLUTION = ROOT / 'solution'
BASELINE = ROOT / 'baseline'
os.environ.setdefault('ENV_ID', 'toggle_lights')
sys.path.insert(0, str(SOLUTION))
sys.path.insert(0, str(BASELINE))
import gym
from detect import probe, is_toggle_only
from gf2_solver import build_action_matrix, solve_instance, verify_solution
def main() -> None:
    env = gym.make_env()
    print('env_id:', gym.ENV_ID)
    prof = probe(env)
    print('profile:', json.dumps({k: v for k, v in prof.items() if k != 'solved_key'}, indent=2))
    print('toggle-only candidate:', is_toggle_only(prof))
    if not is_toggle_only(prof):
        print('not applicable -> exit')
        return
    t = time.time()
    A, actions = build_action_matrix(env)
    print(f'action matrix: {A.shape} built in {time.time() - t:.2f}s')
    print(f'rank ~ #pivots: {min(A.shape)}, action vectors with all zeros: {int((A.sum(0) == 0).sum())}')
    rng = random.Random(123)
    n_total = 50
    solved = 0
    move_ratios = []
    for i in range(n_total):
        L = rng.choice([15, 20, 25, 30])
        state, gen_actions = env.scramble(length=L, seed=rng.randint(0, 10 ** 9))
        out = solve_instance(env, state, A, actions)
        if out is None:
            continue
        if not verify_solution(env, state, out):
            print(f'  inst {i}: not actually solved!')
            continue
        solved += 1
        ratio = len(out) / max(1, len(gen_actions))
        move_ratios.append(ratio)
    if solved:
        import statistics
        print(f'solved {solved}/{n_total}, mean moves/baseline = {statistics.mean(move_ratios):.3f}, score (clamped 2.0) = {min(2.0, 1 / statistics.mean(move_ratios)):.3f}')
    else:
        print(f'solved {solved}/{n_total}')
if __name__ == '__main__':
    main()
