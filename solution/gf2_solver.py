from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import gym
from common import to_jsonable
def _state_bits(env, state: Optional[Any]=None) -> np.ndarray:
    enc = env.encode_state(state)
    return np.asarray(enc['content_values'], dtype=np.uint8) & 1
def _target_bits(env) -> np.ndarray:
    enc = env.encode_state()
    return np.asarray(enc['target_values'], dtype=np.uint8) & 1
def build_action_matrix(env) -> Tuple[np.ndarray, List[str]]:
    env.reset()
    base = _state_bits(env).copy()
    actions = list(env.valid_actions())
    n_cells = base.shape[0]
    n_actions = len(actions)
    A = np.zeros((n_cells, n_actions), dtype=np.uint8)
    for j, a in enumerate(actions):
        env.reset()
        try:
            env.step(a)
            after = _state_bits(env)
            diff = (after ^ base) & 1
            A[:, j] = diff
        except Exception:
            A[:, j] = 0
    env.reset()
    return (A, actions)
def gf2_solve(A: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    A = (A & 1).astype(np.uint8)
    b = (b & 1).astype(np.uint8)
    n, m = A.shape
    M = np.concatenate([A, b.reshape(-1, 1)], axis=1)
    pivot_col_for_row: List[int] = []
    row = 0
    for col in range(m):
        sel = -1
        for r in range(row, n):
            if M[r, col] == 1:
                sel = r
                break
        if sel == -1:
            continue
        if sel != row:
            M[[row, sel]] = M[[sel, row]]
        for r in range(n):
            if r != row and M[r, col] == 1:
                M[r] ^= M[row]
        pivot_col_for_row.append(col)
        row += 1
        if row == n:
            break
    for r in range(row, n):
        if M[r, -1] == 1:
            return None
    x = np.zeros(m, dtype=np.uint8)
    for r, c in enumerate(pivot_col_for_row):
        x[c] = M[r, -1]
    return x
def solve_instance(env, initial_state, action_matrix: np.ndarray, actions: List[str]) -> Optional[List[str]]:
    env.set_state(initial_state)
    cur = _state_bits(env)
    tgt = _target_bits(env)
    b = (cur ^ tgt) & 1
    x = gf2_solve(action_matrix, b)
    if x is None:
        return None
    chosen = [a for j, a in enumerate(actions) if x[j] == 1]
    return chosen
def verify_solution(env, initial_state, actions: List[str]) -> bool:
    env.set_state(initial_state)
    for a in actions:
        if a not in env.valid_actions():
            return False
        env.step(a)
    return env.is_solved()
