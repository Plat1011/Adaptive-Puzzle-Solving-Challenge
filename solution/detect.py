from __future__ import annotations
import random
from typing import Dict, List
import numpy as np
import gym
from common import to_jsonable
def probe(env, n_states: int=16, seed: int=0) -> Dict:
    env.reset()
    rng = random.Random(seed)
    action_types_seen = set()
    branching_factors: List[int] = []
    n_cells = None
    for _ in range(n_states):
        try:
            acts = env.valid_actions()
        except Exception:
            break
        branching_factors.append(len(acts))
        try:
            info = env.encode_actions(acts)
            for t in info.get('action_types', []):
                action_types_seen.add(int(t))
        except Exception:
            pass
        if n_cells is None:
            try:
                enc = env.encode_state()
                n_cells = len(enc.get('content_types', []))
            except Exception:
                pass
        if not acts:
            break
        env.step(rng.choice(acts))
    env.reset()
    sample_acts = env.valid_actions()[:8]
    involution = True
    for a in sample_acts:
        try:
            if env.inverse_action(a) != a:
                involution = False
                break
        except Exception:
            involution = False
            break
    env.reset()
    enc = env.encode_state()
    values = set(enc.get('content_values', []))
    target_values = set(enc.get('target_values', []))
    binary_cells = values.issubset({0, 1}) and target_values.issubset({0, 1})
    return {'env_id': getattr(gym, 'ENV_ID', 'unknown'), 'action_types': sorted(action_types_seen), 'branching_mean': float(np.mean(branching_factors)) if branching_factors else 0.0, 'branching_max': int(max(branching_factors)) if branching_factors else 0, 'n_cells': int(n_cells or 0), 'involution_actions': bool(involution), 'binary_cells': bool(binary_cells), 'n_unique_actions_at_start': len(env.valid_actions())}
def is_toggle_only(profile: Dict) -> bool:
    from gym import ACTION_TOGGLE
    return profile['action_types'] == [int(ACTION_TOGGLE)] and profile['involution_actions'] and profile['binary_cells']
