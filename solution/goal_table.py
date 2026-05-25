from __future__ import annotations
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from common import state_key, to_jsonable


def _rss_mb() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return 0.0


def build_goal_table(
    env,
    deadline: float,
    max_states: int = 2_000_000,
    max_rss_mb: float = 4096.0,
    collect_samples: int = 0,
) -> Tuple[Dict[Any, Tuple[Optional[Any], Optional[str], int]], int, Optional[np.ndarray], Optional[np.ndarray]]:
    env.reset()
    goal_state = to_jsonable(env.get_state())
    goal_key = state_key(goal_state)
    table: Dict[Any, Tuple[Optional[Any], Optional[str], int]] = {goal_key: (None, None, 0)}
    cur_payload: Dict[Any, Any] = {goal_key: goal_state}
    depth = 0
    rss_check_every = 50_000
    last_rss_check = 0
    while cur_payload and time.time() < deadline and len(table) < max_states:
        next_payload: Dict[Any, Any] = {}
        for k, state in cur_payload.items():
            if time.time() >= deadline or len(table) >= max_states:
                break
            if max_rss_mb and len(table) - last_rss_check >= rss_check_every:
                last_rss_check = len(table)
                if _rss_mb() > max_rss_mb:
                    return (table, depth, None, None)
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
                    nsk = state_key(ns)
                except Exception:
                    continue
                if nsk in table:
                    continue
                try:
                    edge = env.inverse_action(a)
                except Exception:
                    edge = a
                table[nsk] = (k, edge, depth + 1)
                next_payload[nsk] = ns
                if len(table) >= max_states:
                    break
        if not next_payload:
            break
        cur_payload = next_payload
        depth += 1
    return (table, depth, None, None)


def save_table(table: Dict[Any, Tuple[Optional[Any], Optional[str], int]], path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(table, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_table(path: str) -> Optional[Dict[Any, Tuple[Optional[Any], Optional[str], int]]]:
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def reconstruct_from_table(
    table: Dict[Any, Tuple[Optional[Any], Optional[str], int]],
    start_key: Any,
) -> Optional[list]:
    if start_key not in table:
        return None
    actions = []
    cur = start_key
    while True:
        parent, action, _ = table[cur]
        if action is None:
            break
        actions.append(action)
        cur = parent
        if cur is None:
            break
    return actions


def forward_to_table(
    env,
    initial_state: Any,
    table: Dict[Any, Tuple[Optional[Any], Optional[str], int]],
    deadline: float,
    max_states: int = 50_000,
) -> Optional[list]:
    start = to_jsonable(initial_state)
    sk = state_key(start)
    if sk in table:
        tail = reconstruct_from_table(table, sk)
        return tail if tail is not None else []
    fwd: Dict[Any, Tuple[Optional[Any], Optional[str]]] = {sk: (None, None)}
    cur_payload: Dict[Any, Any] = {sk: start}
    while cur_payload and time.time() < deadline and len(fwd) < max_states:
        next_payload: Dict[Any, Any] = {}
        for k, state in cur_payload.items():
            if time.time() >= deadline or len(fwd) >= max_states:
                break
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
                    nsk = state_key(ns)
                except Exception:
                    continue
                if nsk in fwd:
                    continue
                fwd[nsk] = (k, a)
                if nsk in table:
                    fwd_actions = []
                    cur = nsk
                    while True:
                        pk, ac = fwd.get(cur, (None, None))
                        if pk is None or ac is None:
                            break
                        fwd_actions.append(ac)
                        cur = pk
                    fwd_actions.reverse()
                    back_actions = reconstruct_from_table(table, nsk) or []
                    return fwd_actions + back_actions
                next_payload[nsk] = ns
                if len(fwd) >= max_states:
                    break
        if not next_payload:
            break
        cur_payload = next_payload
    return None
