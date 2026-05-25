from __future__ import annotations
import math
import os
import random
import time
from typing import Any, Callable, List, Optional, Tuple
import numpy as np
from common import to_jsonable

MODEL_PATH = 'nn_model.pt'
META_PATH = 'nn_meta.json'


def state_to_vec(env, state) -> np.ndarray:
    enc = env.encode_state(state)
    return np.asarray(enc['content_values'], dtype=np.int64)


def collect_random_walks(env, total_samples: int, max_walk: int, deadline: float, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, int]:
    rng = random.Random(seed)
    samples: List[np.ndarray] = []
    labels: List[int] = []
    n_cells = None
    while len(samples) < total_samples and time.time() < deadline:
        env.reset()
        L = rng.randint(1, max_walk)
        for d in range(1, L + 1):
            try:
                acts = env.valid_actions()
                if not acts:
                    break
                env.step(rng.choice(acts))
                vec = state_to_vec(env, env.get_state())
                if n_cells is None:
                    n_cells = vec.shape[0]
                if vec.shape[0] != n_cells:
                    continue
                samples.append(vec)
                labels.append(d)
                if len(samples) >= total_samples:
                    break
                if time.time() >= deadline:
                    break
            except Exception:
                break
    if not samples:
        return np.zeros((0, 1), dtype=np.int64), np.zeros((0,), dtype=np.float32), 0
    X = np.stack(samples).astype(np.int64)
    y = np.asarray(labels, dtype=np.float32)
    return X, y, X.shape[1]


def _make_model(vocab_size: int, n_cells: int, emb_dim: int, hidden: int):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Heuristic(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(max(2, vocab_size + 1), emb_dim)
            self.fc1 = nn.Linear(emb_dim * n_cells, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.fc3 = nn.Linear(hidden, hidden)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            e = self.emb(x).flatten(1)
            h = F.relu(self.fc1(e))
            h = F.relu(self.fc2(h)) + h
            h = F.relu(self.fc3(h)) + h
            return F.softplus(self.head(h)).squeeze(-1)

    return Heuristic()


def train_nn(env, deadline: float, total_samples: int = 200_000, max_walk: int = 80, batch_size: int = 1024, lr: float = 1e-3, emb_dim: int = 16, hidden: int = 256, seed: int = 0) -> bool:
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except Exception:
        return False
    t0 = time.time()
    data_deadline = t0 + max(15.0, (deadline - t0) * 0.3)
    X, y, n_cells = collect_random_walks(env, total_samples=total_samples, max_walk=max_walk, deadline=data_deadline, seed=seed)
    if n_cells == 0 or X.shape[0] < 100:
        return False
    vocab_size = int(X.max()) + 1 if X.size else 2
    print(f'  collected {X.shape[0]} pairs, n_cells={n_cells}, vocab={vocab_size}, walks until t={time.time() - t0:.1f}s')
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    model = _make_model(vocab_size, n_cells, emb_dim=emb_dim, hidden=hidden)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    n = X.shape[0]
    epoch = 0
    while time.time() < deadline:
        idx = torch.randperm(n)
        epoch_loss = 0.0
        steps = 0
        for s in range(0, n, batch_size):
            if time.time() >= deadline:
                break
            sel = idx[s:s + batch_size]
            if sel.numel() < 2:
                continue
            xb = X_t[sel]
            yb = y_t[sel]
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            steps += 1
        if steps:
            if epoch % 5 == 0 or time.time() >= deadline:
                print(f'  epoch {epoch}: loss={epoch_loss / steps:.4f}')
        epoch += 1
    try:
        sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save({'state_dict': sd, 'vocab_size': vocab_size, 'n_cells': n_cells, 'emb_dim': emb_dim, 'hidden': hidden}, MODEL_PATH)
        with open(META_PATH, 'w', encoding='utf-8') as f:
            import json
            json.dump({'vocab_size': vocab_size, 'n_cells': n_cells, 'emb_dim': emb_dim, 'hidden': hidden}, f)
    except Exception:
        return False
    return True


def load_nn_scorer(env) -> Optional[Callable[[List[Any]], np.ndarray]]:
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        import torch
        ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        vocab_size = int(ckpt.get('vocab_size', 64))
        n_cells = int(ckpt.get('n_cells', 16))
        emb_dim = int(ckpt.get('emb_dim', 16))
        hidden = int(ckpt.get('hidden', 256))
        model = _make_model(vocab_size, n_cells, emb_dim=emb_dim, hidden=hidden)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        torch.set_num_threads(1)
        max_idx = vocab_size - 1

        def score(states: List[Any]) -> np.ndarray:
            vecs = []
            for s in states:
                try:
                    v = state_to_vec(env, s)
                    if v.shape[0] != n_cells:
                        v = np.zeros(n_cells, dtype=np.int64)
                    vecs.append(np.clip(v, 0, max_idx))
                except Exception:
                    vecs.append(np.zeros(n_cells, dtype=np.int64))
            X = np.stack(vecs).astype(np.int64)
            with torch.no_grad():
                t = torch.from_numpy(X)
                pred = model(t).cpu().numpy()
            return pred.astype(np.float32)

        return score
    except Exception:
        return None
