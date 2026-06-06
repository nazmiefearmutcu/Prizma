"""
Deterministic synthetic continual-learning benchmarks for Prizma.

Everything is offline and seeded. Two task constructions are provided:

  * permuted_tasks  -- analog of Permuted-MNIST. A single base classification
                       problem; task t applies a fixed random *feature permutation*
                       pi_t to the inputs. Output classes are shared. This is the
                       canonical *strong* catastrophic-forgetting setup: the same
                       input coordinates mean something different in each task, so
                       sequential training overwrites prior tasks.

  * split_tasks     -- analog of Split-MNIST. The base classes are partitioned into
                       disjoint groups; task t asks the network to classify only the
                       classes in group t (re-indexed to 0..g-1). Different tasks live
                       in different input regions, so a *router* can tell them apart
                       from input statistics alone -- the regime Prizma's gate exploits.

The base dataset is labelled by a fixed random "teacher" MLP, which guarantees a
non-linear, learnable structure (a linear model cannot solve it) without any download.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Base dataset: teacher-MLP-labelled synthetic classification
# --------------------------------------------------------------------------- #
def _teacher_logits(X, params):
    """Forward pass of a fixed random 2-hidden-layer tanh teacher network."""
    (W1, b1, W2, b2, W3, b3) = params
    h1 = np.tanh(X @ W1 + b1)
    h2 = np.tanh(h1 @ W2 + b2)
    return h2 @ W3 + b3


def make_base_dataset(n_samples=6000, d=20, n_classes=10, hidden=64, seed=0):
    """Return (X, y) for a non-linearly separable synthetic classification task.

    X ~ N(0, I) in R^d, labelled by a fixed random teacher MLP. Returns float32
    inputs in [-1, 1] (squashed) and int labels. Deterministic in `seed`.
    """
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 1.0 / np.sqrt(d), (d, hidden))
    b1 = rng.normal(0, 0.1, hidden)
    W2 = rng.normal(0, 1.0 / np.sqrt(hidden), (hidden, hidden))
    b2 = rng.normal(0, 0.1, hidden)
    W3 = rng.normal(0, 1.0 / np.sqrt(hidden), (hidden, n_classes))
    b3 = rng.normal(0, 0.1, n_classes)
    params = (W1, b1, W2, b2, W3, b3)

    X = rng.normal(0, 1.0, (n_samples, d)).astype(np.float32)
    logits = _teacher_logits(X, params)
    y = np.argmax(logits, axis=1).astype(np.int64)
    # squash inputs to a bounded analog-friendly range
    X = np.tanh(X).astype(np.float32)
    return X, y


def _split_train_test(X, y, test_frac=0.2, seed=0):
    rng = np.random.default_rng(seed + 999)
    n = len(X)
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    te, tr = idx[:n_test], idx[n_test:]
    return (X[tr], y[tr]), (X[te], y[te])


# --------------------------------------------------------------------------- #
# Continual-learning task sequences
# --------------------------------------------------------------------------- #
class Task:
    """A single continual-learning task: train/test splits + metadata."""

    def __init__(self, name, Xtr, ytr, Xte, yte, n_classes):
        self.name = name
        self.Xtr, self.ytr = Xtr, ytr
        self.Xte, self.yte = Xte, yte
        self.n_classes = n_classes

    def __repr__(self):
        return (f"Task({self.name}, train={len(self.Xtr)}, test={len(self.Xte)}, "
                f"classes={self.n_classes})")


def permuted_tasks(n_tasks=5, n_samples=6000, d=20, n_classes=10, seed=0):
    """Permuted-MNIST analog. Each task applies a fixed feature permutation."""
    X, y = make_base_dataset(n_samples=n_samples, d=d, n_classes=n_classes, seed=seed)
    (Xtr, ytr), (Xte, yte) = _split_train_test(X, y, seed=seed)
    rng = np.random.default_rng(seed + 1)
    tasks = []
    for t in range(n_tasks):
        perm = np.arange(d) if t == 0 else rng.permutation(d)
        tasks.append(Task(
            name=f"perm{t}",
            Xtr=Xtr[:, perm].copy(), ytr=ytr.copy(),
            Xte=Xte[:, perm].copy(), yte=yte.copy(),
            n_classes=n_classes,
        ))
    return tasks


def structured_permuted_tasks(n_tasks=5, n_samples=6000, d=24, k_latent=8,
                              n_classes=8, seed=0, noise_std=0.0):
    """Domain-incremental stream that is genuinely INPUT-DISTINGUISHABLE.

    Base features are a fixed linear mixing of latent factors:  v = latent @ A^T,
    latent ~ N(0, I_k), so cov(v) = A A^T != I (correlated features). Labels come from a
    teacher applied to the LATENTS (shared across tasks). Task t permutes the observed
    features by pi_t, so cov(x_t) = P_t (A A^T) P_t^T differs per task -> an autoencoder /
    recognizer CAN tell domains apart from the input alone (unlike permuted iid-Gaussian,
    where permutation leaves the distribution invariant). Naive sequential training still
    forgets: the input->label map differs per permutation and overwrites shared weights.
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 1.0, (d, k_latent)).astype(np.float32)          # fixed mixing
    latent = rng.normal(0, 1.0, (n_samples, k_latent)).astype(np.float32)
    v = latent @ A.T                                                  # correlated features
    v = (v / (v.std(0, keepdims=True) + 1e-6)).astype(np.float32)
    if noise_std > 0:
        # iid (permutation-invariant) noise dilutes the structured signal -> shrinks the
        # recognition margin. A separability knob: noise_std=0 -> cleanly distinguishable;
        # large noise_std -> domains indistinguishable (Prizma should degrade to naive).
        v = (v + noise_std * rng.normal(0, 1, v.shape)).astype(np.float32)
    # teacher labels on the LATENTS (shared rule, domain-invariant)
    Wt1 = rng.normal(0, 1.0 / np.sqrt(k_latent), (k_latent, 32))
    Wt2 = rng.normal(0, 1.0 / np.sqrt(32), (32, n_classes))
    y = np.argmax(np.tanh(latent @ Wt1) @ Wt2, axis=1).astype(np.int64)
    (vtr, ytr), (vte, yte) = _split_train_test(v, y, seed=seed)
    rng2 = np.random.default_rng(seed + 1)
    tasks = []
    for t in range(n_tasks):
        perm = np.arange(d) if t == 0 else rng2.permutation(d)
        tasks.append(Task(
            name=f"sperm{t}",
            Xtr=vtr[:, perm].copy(), ytr=ytr.copy(),
            Xte=vte[:, perm].copy(), yte=yte.copy(),
            n_classes=n_classes,
        ))
    return tasks


def split_tasks(n_tasks=5, classes_per_task=2, n_samples=12000, d=20, seed=0):
    """Split-MNIST analog. Disjoint class groups; labels re-indexed per task."""
    n_classes = n_tasks * classes_per_task
    X, y = make_base_dataset(n_samples=n_samples, d=d, n_classes=n_classes, seed=seed)
    (Xtr, ytr), (Xte, yte) = _split_train_test(X, y, seed=seed)
    tasks = []
    for t in range(n_tasks):
        lo = t * classes_per_task
        hi = lo + classes_per_task
        cls = list(range(lo, hi))
        mtr = np.isin(ytr, cls)
        mte = np.isin(yte, cls)
        tasks.append(Task(
            name=f"split{t}:{cls}",
            Xtr=Xtr[mtr].copy(), ytr=(ytr[mtr] - lo).copy(),
            Xte=Xte[mte].copy(), yte=(yte[mte] - lo).copy(),
            n_classes=classes_per_task,
        ))
    return tasks


if __name__ == "__main__":
    for tasks, label in [(permuted_tasks(), "PERMUTED"), (split_tasks(), "SPLIT")]:
        print(f"\n=== {label} ===")
        for t in tasks:
            print(" ", t)
