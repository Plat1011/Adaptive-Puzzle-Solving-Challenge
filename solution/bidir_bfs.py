from __future__ import annotations
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
from common import state_key, to_jsonable
def _expand_level(env, fringe: Dict[str, Tuple[Optional[str], Optional[str]]], new_keys: List[str], other: Dict[str, Tuple[Optional[str], Optional[str]]], forward: bool, max_size: int) -> Tuple[List[str], Optional[str]]:
    next_layer: List[str] = []
    for k in new_keys:
        state = _PAYLOAD[k]
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
            nsk = state_key(ns)
            if nsk in fringe:
                continue
            if forward:
                edge_action = a
            else:
                try:
                    edge_action = env.inverse_action(a)
                except Exception:
                    edge_action = a
            fringe[nsk] = (k, edge_action)
            _PAYLOAD[nsk] = ns
            if nsk in other:
                return (next_layer, nsk)
            next_layer.append(nsk)
            if len(fringe) >= max_size:
                return (next_layer, None)
    return (next_layer, None)
_PAYLOAD: Dict[str, Any] = {}
def bidir_bfs(env, initial_state: Any, deadline: Optional[float]=None, max_states: int=200000) -> Optional[List[str]]:
    _PAYLOAD.clear()
    env.reset()
    goal_state = to_jsonable(env.get_state())
    goal_key = state_key(goal_state)
    start_state = to_jsonable(initial_state)
    start_key = state_key(start_state)
    if start_key == goal_key:
        return []
    fwd: Dict[str, Tuple[Optional[str], Optional[str]]] = {start_key: (None, None)}
    bwd: Dict[str, Tuple[Optional[str], Optional[str]]] = {goal_key: (None, None)}
    _PAYLOAD[start_key] = start_state
    _PAYLOAD[goal_key] = goal_state
    if start_key in bwd:
        return []
    fwd_layer = [start_key]
    bwd_layer = [goal_key]
    while fwd_layer and bwd_layer:
        if deadline is not None and time.time() >= deadline:
            return None
        if len(fwd) + len(bwd) >= max_states:
            return None
        forward = len(fwd_layer) <= len(bwd_layer)
        if forward:
            new_layer, meet = _expand_level(env, fwd, fwd_layer, bwd, True, max_states)
            fwd_layer = new_layer
        else:
            new_layer, meet = _expand_level(env, bwd, bwd_layer, fwd, False, max_states)
            bwd_layer = new_layer
        if meet is not None:
            return _reconstruct(meet, fwd, bwd, env)
    return None
def _reconstruct(meet: str, fwd: Dict[str, Tuple[Optional[str], Optional[str]]], bwd: Dict[str, Tuple[Optional[str], Optional[str]]], env) -> List[str]:
    fwd_actions: List[str] = []
    cur = meet
    while True:
        pk, a = fwd.get(cur, (None, None))
        if pk is None or a is None:
            break
        fwd_actions.append(a)
        cur = pk
    fwd_actions.reverse()
    bwd_actions: List[str] = []
    cur = meet
    while True:
        pk, a = bwd.get(cur, (None, None))
        if pk is None or a is None:
            break
        bwd_actions.append(a)
        cur = pk
    return fwd_actions + bwd_actions
