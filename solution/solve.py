from __future__ import annotations
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, List, Optional
import numpy as np
import gym
import common
from common import state_key, to_jsonable
from bidir_bfs import bidir_bfs
from heuristics import make_batched_heuristic
from compact_table import load_compact, forward_to_compact
from beam import beam_search_to_compact
from nn_heuristic import load_nn_scorer
TIME_LIMIT_DEFAULT = 25 * 60
SAFETY_MARGIN = 30
PROFILE_PATH = 'profile.json'
GF2_PATH = 'gf2.npz'
BBFS_TIME_PER_INSTANCE = 0.6
BBFS_MAX_STATES = 60000
TABLE_FWD_MAX_STATES = 80000


def load_jsonl(path: str) -> List[dict]:
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def load_profile() -> Optional[dict]:
    if not os.path.exists(PROFILE_PATH):
        return None
    try:
        with open(PROFILE_PATH, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def load_gf2():
    if not os.path.exists(GF2_PATH):
        return None
    try:
        npz = np.load(GF2_PATH, allow_pickle=False)
        return (npz['A'], list(npz['actions']))
    except Exception:
        return None


def verify(env, initial_state: Any, actions: List[str]) -> bool:
    try:
        env.set_state(initial_state)
        for a in actions:
            if a not in env.valid_actions():
                return False
            env.step(a)
        return env.is_solved()
    except Exception:
        return False


def solve_one(env, initial_state: Any, solved_key: str, *, gf2, compact, scorer, beam_w: int, deadline: float) -> List[str]:
    if state_key(to_jsonable(initial_state)) == solved_key:
        return []
    if gf2 is not None:
        A, actions_list = gf2
        try:
            from gf2_solver import solve_instance as gf2_solve_instance
            out = gf2_solve_instance(env, initial_state, A, actions_list)
            if out is not None and verify(env, initial_state, out):
                return out
        except Exception:
            pass
    if compact is not None:
        remaining = deadline - time.time()
        if remaining > 0:
            tbl_deadline = time.time() + 0.35 * remaining
            try:
                out = forward_to_compact(env, initial_state, compact, tbl_deadline, TABLE_FWD_MAX_STATES)
                if out is not None and verify(env, initial_state, out):
                    return out
            except Exception:
                pass
    if compact is not None and scorer is not None:
        widths_to_try = [beam_w]
        if beam_w < 96:
            widths_to_try.append(beam_w * 2)
            widths_to_try.append(beam_w * 4)
        else:
            widths_to_try.append(min(beam_w * 2, 512))
        for w in widths_to_try:
            remaining = deadline - time.time()
            if remaining <= 0.05:
                break
            beam_deadline = time.time() + 0.6 * remaining
            try:
                out = beam_search_to_compact(env, initial_state, compact, scorer, beam_deadline, beam_width=w, max_depth=200)
                if out is not None and verify(env, initial_state, out):
                    return out
            except Exception:
                pass
    if compact is None:
        bbfs_deadline = min(deadline, time.time() + BBFS_TIME_PER_INSTANCE)
        try:
            out = bidir_bfs(env, initial_state, deadline=bbfs_deadline, max_states=BBFS_MAX_STATES)
            if out is not None and verify(env, initial_state, out):
                return out
        except Exception:
            pass
    try:
        from astar import solve_astar
        def v_batch(states):
            return scorer(states) if scorer is not None else np.zeros(len(states), dtype=np.float32)
        out = solve_astar(env, initial_state, solved_key, v_batch, deadline, max_nodes=300000, expand_batch=64, h_weight=1.5)
        if out is not None and verify(env, initial_state, out):
            return out
    except Exception:
        pass
    return []


def _run_sequential(instances: List[dict], output_path: str, deadline_abs: float) -> int:
    env = gym.make_env()
    env.reset()
    solved_k = state_key(env.get_state())
    gf2 = load_gf2()
    compact = load_compact('.')
    manhattan = make_batched_heuristic(env)
    nn_score = load_nn_scorer(env)

    def scorer(states):
        m = manhattan(states)
        if nn_score is None:
            return m
        try:
            nn = nn_score(states)
        except Exception:
            return m
        return np.maximum(m, nn)

    profile = load_profile()
    beam_w = 256
    if profile is not None:
        try:
            branching = float(profile.get('branching_mean', 0.0))
            if branching >= 8:
                beam_w = 96
            elif branching >= 5:
                beam_w = 128
            else:
                beam_w = 256
        except Exception:
            beam_w = 256
    print(f'sequential: gf2={gf2 is not None} compact={compact is not None}({len(compact[0]) if compact else 0}) nn={nn_score is not None} beam_w={beam_w}')
    n = len(instances)
    n_solved = 0
    t_start = time.time()
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['instance_id', 'actions'])
        writer.writeheader()
        for i, inst in enumerate(instances):
            iid = inst['instance_id']
            now = time.time()
            if now >= deadline_abs:
                writer.writerow({'instance_id': iid, 'actions': ''})
                continue
            remaining = n - i
            per_instance = max(0.5, (deadline_abs - now) / remaining)
            inst_deadline = now + per_instance
            try:
                out = solve_one(env, inst['state'], solved_k, gf2=gf2, compact=compact, scorer=scorer, beam_w=beam_w, deadline=inst_deadline)
            except Exception as e:
                print(f'  {iid} solve_one error: {repr(e)}')
                out = []
            writer.writerow({'instance_id': iid, 'actions': ' '.join(out)})
            if out:
                n_solved += 1
            if (i + 1) % 50 == 0 or i + 1 == n:
                print(f'  {i + 1}/{n} solved={n_solved} elapsed={time.time() - t_start:.0f}s')
    return n_solved


def _run_parallel(instances: List[dict], output_path: str, deadline_abs: float, n_workers: int) -> Optional[int]:
    tmpdir = tempfile.mkdtemp(prefix='solve_workers_')
    try:
        chunk_paths: List[str] = []
        out_paths: List[str] = []
        for w in range(n_workers):
            chunk = instances[w::n_workers]
            cpath = os.path.join(tmpdir, f'chunk_{w}.jsonl')
            opath = os.path.join(tmpdir, f'out_{w}.csv')
            with open(cpath, 'w', encoding='utf-8') as f:
                for inst in chunk:
                    f.write(json.dumps(inst) + '\n')
            chunk_paths.append(cpath)
            out_paths.append(opath)
        worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'worker.py')
        if not os.path.exists(worker_script):
            print(f'worker.py not found at {worker_script}; falling back to sequential')
            return None
        env_vars = dict(os.environ)
        env_vars['OMP_NUM_THREADS'] = '1'
        env_vars['MKL_NUM_THREADS'] = '1'
        env_vars['OPENBLAS_NUM_THREADS'] = '1'
        procs = []
        for w in range(n_workers):
            p = subprocess.Popen(
                [sys.executable, worker_script, chunk_paths[w], out_paths[w], str(deadline_abs)],
                env=env_vars,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            procs.append(p)
        deadline_with_slack = deadline_abs + 30
        all_done = True
        for p in procs:
            remaining = max(1.0, deadline_with_slack - time.time())
            try:
                p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    p.kill()
                except Exception:
                    pass
                all_done = False
        results: dict = {}
        for opath in out_paths:
            if not os.path.exists(opath):
                continue
            try:
                with open(opath, encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        results[row['instance_id']] = row.get('actions', '') or ''
            except Exception:
                continue
        n_solved = sum(1 for v in results.values() if v)
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['instance_id', 'actions'])
            writer.writeheader()
            for inst in instances:
                iid = inst['instance_id']
                writer.writerow({'instance_id': iid, 'actions': results.get(iid, '')})
        print(f'parallel: {n_solved}/{len(instances)} solved, all_done={all_done}')
        return n_solved
    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='input_states.jsonl')
    ap.add_argument('--output', default='output_actions.csv')
    ap.add_argument('--time_limit', type=int, default=int(os.environ.get('SOLVE_TIME_LIMIT', TIME_LIMIT_DEFAULT)))
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SOLVE_WORKERS', 0)))
    ap.add_argument('--no-parallel', action='store_true')
    args = ap.parse_args()
    start = time.time()
    deadline = start + args.time_limit - SAFETY_MARGIN
    instances = load_jsonl(args.input)
    n = len(instances)
    cpus = os.cpu_count() or 1
    n_workers = args.workers if args.workers > 0 else max(1, min(8, cpus))
    print(f"env_id={getattr(gym, 'ENV_ID', '?')} loaded {n} instances; cpus={cpus} workers={n_workers}")
    use_parallel = (not args.no_parallel) and n_workers > 1 and n >= n_workers
    n_solved_count = None
    if use_parallel:
        try:
            n_solved_count = _run_parallel(instances, args.output, deadline, n_workers)
        except Exception as e:
            print(f'parallel failed: {repr(e)}; falling back to sequential')
            n_solved_count = None
    if n_solved_count is None:
        n_solved_count = _run_sequential(instances, args.output, deadline)
    print(f'final: {n_solved_count}/{n} solved, total {time.time() - start:.1f}s')


if __name__ == '__main__':
    main()
