"""
Continual-learning metrics (Lopez-Paz & Ranzato 2017 conventions).

The central object is the accuracy matrix R, where

    R[i, j] = test accuracy on task j *after* finishing training on task i.

From R we derive:

  * ACC  -- final average accuracy   = mean_j R[T-1, j]
  * BWT  -- backward transfer        = mean_{j<T-1} ( R[T-1, j] - R[j, j] )
            (negative => forgetting; positive => later learning *helped* earlier tasks)
  * FGT  -- forgetting               = mean_{j<T-1} ( max_i R[i, j] - R[T-1, j] )
            (>=0; 0 means nothing was forgotten)
  * LA   -- learning accuracy        = mean_i R[i, i]   (how well each task was learned when fresh)
"""

from __future__ import annotations

import numpy as np


class AccuracyMatrix:
    def __init__(self, n_tasks: int):
        self.T = n_tasks
        self.R = np.full((n_tasks, n_tasks), np.nan, dtype=np.float64)

    def record(self, after_task: int, on_task: int, acc: float):
        self.R[after_task, on_task] = acc

    def acc(self) -> float:
        """Final average accuracy across all tasks (last row)."""
        return float(np.nanmean(self.R[self.T - 1, :]))

    def bwt(self) -> float:
        if self.T < 2:
            return 0.0
        diffs = [self.R[self.T - 1, j] - self.R[j, j] for j in range(self.T - 1)]
        return float(np.nanmean(diffs))

    def forgetting(self) -> float:
        if self.T < 2:
            return 0.0
        fgs = []
        for j in range(self.T - 1):
            seen = self.R[j:self.T - 1, j]          # accuracies on j from when it was learned onward
            seen = seen[~np.isnan(seen)]
            if len(seen) == 0:
                continue
            fgs.append(float(np.nanmax(seen) - self.R[self.T - 1, j]))
        return float(np.mean(fgs)) if fgs else 0.0

    def learning_acc(self) -> float:
        return float(np.nanmean(np.diag(self.R)))

    def summary(self) -> dict:
        return {
            "ACC": self.acc(),
            "BWT": self.bwt(),
            "FGT": self.forgetting(),
            "LA": self.learning_acc(),
        }

    def pretty(self) -> str:
        lines = ["    " + " ".join(f"T{j:<5}" for j in range(self.T))]
        for i in range(self.T):
            row = " ".join(
                ("  -- " if np.isnan(self.R[i, j]) else f"{self.R[i, j]:.3f}")
                for j in range(self.T)
            )
            lines.append(f"aT{i} {row}")
        s = self.summary()
        lines.append(
            f"  ACC={s['ACC']:.3f}  BWT={s['BWT']:+.3f}  FGT={s['FGT']:.3f}  LA={s['LA']:.3f}"
        )
        return "\n".join(lines)


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    return float((np.argmax(logits, axis=1) == y).mean())


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error -- for the uncertainty / sampling claim (P5)."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0
    n = len(y)
    for b in range(n_bins):
        m = (conf > bins[b]) & (conf <= bins[b + 1])
        if m.sum() == 0:
            continue
        e += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(e)
