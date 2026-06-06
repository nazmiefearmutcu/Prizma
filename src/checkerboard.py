"""
Rotating-Checkerboard continual-learning stream + frozen RBF "HEAD" prior.

This is the committee's pre-validated benchmark. K binary-classification tasks live over
the SAME 2-D input box; task t rotates a checkerboard label field by pi*t/K. Because all
tasks share one input distribution, sequential training maximally interferes -> strong,
clean catastrophic forgetting. Everything is deterministic from integer seeds and pure numpy.

The RBF feature lift is the slow / frozen structured generative prior (the "HEAD"). It is
identical for every learner, so model capacity is matched at the readout and the only thing
that differs between learners is the LEARNING RULE.
"""

from __future__ import annotations

import numpy as np


def checkerboard_task(t, K, n=4000, seed_base=1000):
    """Task t of K: rotate input by pi*t/K, label = (floor(z0)+floor(z1)) mod 2."""
    rng = np.random.default_rng(seed_base + t)
    X = rng.uniform(-2.0, 2.0, size=(n, 2)).astype(np.float32)
    theta = np.pi * t / K
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]], dtype=np.float32)
    Z = X @ R.T
    y = ((np.floor(Z[:, 0]).astype(np.int64) + np.floor(Z[:, 1]).astype(np.int64)) % 2)
    return X, y.astype(np.int64)


class RBFHead:
    """Frozen random RBF feature lift. NEVER updated (the slow generative prior)."""

    def __init__(self, d_feat=120, width=0.32, seed=123):
        rng = np.random.default_rng(seed)
        self.centers = rng.uniform(-2.2, 2.2, size=(d_feat, 2)).astype(np.float32)
        self.width = width
        self.d_feat = d_feat

    def __call__(self, X):
        # ||X - c||^2 via broadcasting: (n, d_feat)
        d2 = ((X[:, None, :] - self.centers[None, :, :]) ** 2).sum(axis=2)
        return np.exp(-d2 / (2.0 * self.width ** 2)).astype(np.float32)


def make_stream(K=3, n_train=4000, n_test=2000):
    """Return list of (Xtr,ytr,Xte,yte) tasks; the continual stream."""
    tasks = []
    for t in range(K):
        Xtr, ytr = checkerboard_task(t, K, n=n_train, seed_base=1000)
        Xte, yte = checkerboard_task(t, K, n=n_test, seed_base=5000)
        tasks.append((Xtr, ytr, Xte, yte))
    return tasks


if __name__ == "__main__":
    # sanity: each task should be learnable (not trivially separable, ~balanced labels)
    head = RBFHead()
    for K in (3,):
        tasks = make_stream(K=K)
        for t, (Xtr, ytr, Xte, yte) in enumerate(tasks):
            print(f"task{t}: train={len(Xtr)} pos_frac={ytr.mean():.3f} "
                  f"feat_dim={head(Xtr[:5]).shape[1]}")
