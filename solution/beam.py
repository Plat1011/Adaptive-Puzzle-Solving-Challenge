from __future__ import annotations
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from common import state_key, to_jsonable
from compact_table import hash_key, lookup, reconstruct


def beam_search_to_compact(
    env,
    initial_state: Any,
    compact,
    h_fn_batched: Callable[[List[Any]], np.ndarray],
    deadline: float,
    beam_width: int = 256,
    max_depth: int = 200,
) -> Optional[List[str]]:
    if compact is None:
        return None
    keys = compact[0]
    start = to_jsonable(initial_state)
    sh = hash_key(start)
    if lookup(keys, sh) >= 0:
        return reconstruct(compact, sh) or []
    parents: Dict[int, Tuple[int, Optional[str]]] = {sh: (0, None)}
    payload: Dict[int, Any] = {sh: start}
    frontier_keys: List[int] = [sh]
    depth = 0
    while frontier_keys and depth < max_depth and time.time() < deadline:
        cand_keys: List[int] = []
        cand_states: List[Any] = []
        for k in frontier_keys:
            if time.time() >= deadline:
                break
            state = payload[k]
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
                if nh in parents:
                    continue
                parents[nh] = (k, a)
                payload[nh] = ns
                if lookup(keys, nh) >= 0:
                    fwd: List[str] = []
                    cur = nh
                    while True:
                        pk, ac = parents[cur]
                        if ac is None:
                            break
                        fwd.append(ac)
                        cur = pk
                    fwd.reverse()
                    tail = reconstruct(compact, nh) or []
                    return fwd + tail
                cand_keys.append(nh)
                cand_states.append(ns)
        if not cand_keys:
            break
        try:
            scores = h_fn_batched(cand_states)
        except Exception:
            scores = np.zeros(len(cand_keys), dtype=np.float32)
        order = np.argsort(scores)[:beam_width]
        new_frontier = [cand_keys[i] for i in order]
        payload = {k: payload[k] for k in new_frontier}
        frontier_keys = new_frontier
        depth += 1
    return None
