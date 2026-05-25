import json
import random
from typing import Any, Dict, List, Tuple
import numpy as np
VALUE_VOCAB = 64
CONTENT_TYPES = 4
TOKEN_FEAT_DIM = 3 + CONTENT_TYPES + 1 + CONTENT_TYPES + 1 + 2
DENSE_DIM = 3 + CONTENT_TYPES + CONTENT_TYPES + 2
def to_jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    return x
def _list_to_bytes(x):
    try:
        arr = np.asarray(x)
        if arr.dtype.kind in 'iuf':
            return arr.tobytes()
    except Exception:
        pass
    return json.dumps(to_jsonable(x), sort_keys=True)
def state_key(state):
    if isinstance(state, np.ndarray):
        return state.tobytes()
    if isinstance(state, dict):
        parts = []
        for k in sorted(state.keys()):
            v = state[k]
            if isinstance(v, np.ndarray):
                parts.append((k, v.tobytes()))
            elif isinstance(v, (list, tuple)):
                parts.append((k, _list_to_bytes(v)))
            else:
                parts.append((k, v))
        return tuple(parts)
    if isinstance(state, (list, tuple)):
        return _list_to_bytes(state)
    return json.dumps(to_jsonable(state), sort_keys=True)
def encode_tokens(env, state=None) -> np.ndarray:
    obs = env.encode_state(state)
    pos = np.asarray(obs['positions'], dtype=np.float32)
    ct = np.asarray(obs['content_types'], dtype=np.int64)
    cv = np.asarray(obs['content_values'], dtype=np.int64)
    tt = np.asarray(obs['target_types'], dtype=np.int64)
    tv = np.asarray(obs['target_values'], dtype=np.int64)
    n = len(ct)
    feat = np.zeros((n, TOKEN_FEAT_DIM), dtype=np.float32)
    feat[:, 0:3] = pos
    for t in range(CONTENT_TYPES):
        feat[:, 3 + t] = (ct == t).astype(np.float32)
    feat[:, 3 + CONTENT_TYPES] = np.clip(cv, 0, VALUE_VOCAB - 1).astype(np.float32)
    base = 3 + CONTENT_TYPES + 1
    for t in range(CONTENT_TYPES):
        feat[:, base + t] = (tt == t).astype(np.float32)
    feat[:, base + CONTENT_TYPES] = np.clip(tv, 0, VALUE_VOCAB - 1).astype(np.float32)
    match = ((ct == tt) & (cv == tv)).astype(np.float32)
    feat[:, -2] = match
    feat[:, -1] = 1.0 - match
    return feat
def split_token_features(tokens: np.ndarray) -> Dict[str, np.ndarray]:
    base = 3 + CONTENT_TYPES + 1
    dense = np.concatenate([tokens[:, :3 + CONTENT_TYPES], tokens[:, base:base + CONTENT_TYPES], tokens[:, -2:]], axis=-1).astype(np.float32)
    return {'dense': dense, 'content_value': tokens[:, 3 + CONTENT_TYPES].astype(np.int64), 'target_value': tokens[:, base + CONTENT_TYPES].astype(np.int64)}
def backward_walks(env, num_walks: int, min_len: int, max_len: int, seed: int=0) -> List[Tuple[Any, int]]:
    rng = random.Random(seed)
    pairs: List[Tuple[Any, int]] = []
    for _ in range(num_walks):
        L = rng.randint(min_len, max_len)
        env.reset(seed=rng.randint(0, 10 ** 9))
        for depth in range(1, L + 1):
            a = rng.choice(env.valid_actions())
            env.step(a)
            pairs.append((to_jsonable(env.get_state()), depth))
    return pairs
