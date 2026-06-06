"""
Backprop baselines on the same numpy substrate as Prizma, for a fair comparison.

  * MLP        -- plain backprop MLP (SGD). The naive sequential baseline; expected to
                  forget catastrophically.
  * EWC        -- MLP + Elastic Weight Consolidation (Kirkpatrick 2017). Uses TASK
                  BOUNDARIES (it must be told when a task ends to snapshot params and the
                  Fisher diagonal). This is the *privileged* upper-bound competitor:
                  Prizma aims to approach it WITHOUT task boundaries.

We implement backprop by hand (no autograd) so the comparison against the local
PC/Prizma learners is on identical numerical footing.
"""

from __future__ import annotations

import numpy as np


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    def __init__(self, sizes, seed=0, act="tanh"):
        self.sizes = sizes
        self.act_name = act
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            self.W.append(rng.normal(0, 1.0 / np.sqrt(fan_in), (sizes[i], sizes[i + 1])).astype(np.float32))
            self.b.append(np.zeros(sizes[i + 1], dtype=np.float32))

    def _act(self, x):
        return np.tanh(x) if self.act_name == "tanh" else np.maximum(0, x)

    def _dact(self, a):
        # derivative as a function of the activation output a
        return (1.0 - a * a) if self.act_name == "tanh" else (a > 0).astype(a.dtype)

    def forward(self, X, cache=False):
        a = X
        acts = [a]
        pre = []
        for i in range(len(self.W) - 1):
            z = a @ self.W[i] + self.b[i]
            a = self._act(z)
            pre.append(z)
            acts.append(a)
        logits = a @ self.W[-1] + self.b[-1]
        if cache:
            return logits, acts
        return logits

    def predict_logits(self, X):
        return self.forward(X)

    def grads(self, X, y):
        """Cross-entropy gradients via manual backprop. Returns (gW, gb, loss)."""
        n = len(X)
        logits, acts = self.forward(X, cache=True)
        probs = _softmax(logits)
        loss = float(-np.log(probs[np.arange(n), y] + 1e-12).mean())
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        delta = probs.copy()
        delta[np.arange(n), y] -= 1.0
        delta /= n
        for i in reversed(range(len(self.W))):
            a_prev = acts[i]
            gW[i] = a_prev.T @ delta
            gb[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * self._dact(acts[i])
        return gW, gb, loss

    def step(self, gW, gb, lr):
        for i in range(len(self.W)):
            self.W[i] -= lr * gW[i]
            self.b[i] -= lr * gb[i]

    def fit_task(self, X, y, epochs=5, batch=128, lr=0.05, ewc=None, rng=None):
        rng = rng or np.random.default_rng(0)
        n = len(X)
        for _ in range(epochs):
            idx = rng.permutation(n)
            for s in range(0, n, batch):
                bi = idx[s:s + batch]
                gW, gb, _ = self.grads(X[bi], y[bi])
                if ewc is not None:
                    ewc.add_penalty_grads(self, gW, gb)
                self.step(gW, gb, lr)


class EWC:
    """Elastic Weight Consolidation. Requires explicit task boundaries."""

    def __init__(self, lam=50.0):
        self.lam = lam
        self.stars = []   # list of (W*, b*) snapshots
        self.fishers = []  # list of (FW, Fb) diagonals

    def add_penalty_grads(self, model, gW, gb):
        for (Ws, bs), (FW, Fb) in zip(self.stars, self.fishers):
            for i in range(len(model.W)):
                gW[i] += self.lam * FW[i] * (model.W[i] - Ws[i])
                gb[i] += self.lam * Fb[i] * (model.b[i] - bs[i])

    def consolidate(self, model, X, y, n_samples=1024, rng=None):
        """Snapshot params + estimate the Fisher diagonal at the task boundary."""
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
        FW = [np.zeros_like(w) for w in model.W]
        Fb = [np.zeros_like(b) for b in model.b]
        for j in idx:
            gW, gb, _ = model.grads(X[j:j + 1], y[j:j + 1])
            for i in range(len(FW)):
                FW[i] += gW[i] ** 2
                Fb[i] += gb[i] ** 2
        FW = [f / len(idx) for f in FW]
        Fb = [f / len(idx) for f in Fb]
        self.stars.append(([w.copy() for w in model.W], [b.copy() for b in model.b]))
        self.fishers.append((FW, Fb))
