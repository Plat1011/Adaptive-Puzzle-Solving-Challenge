from __future__ import annotations
import csv
import json
import os
import sys
import time
import numpy as np
import gym
from common import state_key, to_jsonable
from heuristics import make_batched_heuristic
from compact_table import load_compact
from nn_heuristic import load_nn_scorer
from solve import solve_one, load_gf2, load_profile


def main() -> None:
    if len(sys.argv) < 4:
        print('usage: worker.py <chunk.jsonl> <out.csv> <deadline_unix>', file=sys.stderr)
        sys.exit(1)
    chunk_path = sys.argv[1]
    out_path = sys.argv[2]
    deadline_abs = float(sys.argv[3])
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
    with open(chunk_path, encoding='utf-8') as f:
        instances = [json.loads(l) for l in f if l.strip()]
    n = len(instances)
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
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
            except Exception:
                out = []
            writer.writerow({'instance_id': iid, 'actions': ' '.join(out)})


if __name__ == '__main__':
    main()
