"""
Backprop baselines on the same numpy substrate as Prizma, for a fair comparison.

  * MLP        -- plain backprop MLP (SGD). The naive sequential baseline; expected to
                  forget catastrophically.
  * EWC        -- MLP + Elastic Weight Consolidation (Kirkpatrick 2017). Uses TASK
                  BOUNDARIES (it must be told when a task ends to snapshot params and the
                  Fisher diagonal). This is the *privileged* upper-bound competitor:
                  Prizma aims to approach it WITHOUT task boundaries.

  * EWC        -- MLP + Elastic Weight Consolidation (Kirkpatrick 2017). Uses TASK
                  BOUNDARIES (it must be told when a task ends to snapshot params and the
                  Fisher diagonal). This is the *privileged* upper-bound competitor:
                  Prizma aims to approach it WITHOUT task boundaries.
  * OnlineEWC  -- online EWC (Chaudhry et al. 2018; single running Fisher + one anchor,
                  with the online-lambda rescaling of Schwarz et al. 2018). Two modes:
                  boundary=False (default) is fully BOUNDARY-FREE (Fisher EMA per batch);
                  boundary=True mirrors the paper protocol for the apples-to-apples row.
  * MAS        -- Memory Aware Synapses (Aljundi et al. 2018): importance = running EMA of
                  |grad of 0.5*||f(x)||^2|. BOUNDARY-FREE (EMA anchor, no consolidate()).
  * SI         -- Synaptic Intelligence (Zenke et al. 2017): path-integral importance
                  omega = -g.dTheta accumulated online, normalized to Omega at task
                  boundaries. BOUNDARY-USING (honest label), like EWC above.

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

    def l2_grads(self, X):
        """Gradients of L = 0.5 * mean_x ||f(x)||^2 (the output-L2 importance surrogate MAS
        uses when no task gradient is available; Aljundi et al. 2018). Same manual-backprop
        chain as grads(), but the output-error neuron is simply f(x) itself."""
        n = len(X)
        logits, acts = self.forward(X, cache=True)
        delta = logits / n
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        for i in reversed(range(len(self.W))):
            a_prev = acts[i]
            gW[i] = a_prev.T @ delta
            gb[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * self._dact(acts[i])
        return gW, gb

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
                    # optional hook for boundary-free regularizers that need to SEE the batch
                    # (e.g. MAS output-L2 importance). EWC and SI do not define it and are
                    # unaffected. Raw task grads are passed on to add_penalty_grads unchanged.
                    observe = getattr(ewc, "observe", None)
                    if observe is not None:
                        observe(self, X[bi])
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


class OnlineEWC:
    """Online EWC -- ONE running Fisher diagonal and ONE anchor, never a per-task list.

    boundary=False (default, BOUNDARY-FREE): the Fisher is a per-batch sliding-window EMA
      F <- gamma*F + (1-gamma)*g^2 (g = raw task gradient, an unbiased squared-gradient
      proxy of the diagonal Fisher), and the anchor is a slow EMA theta* <- a*theta* +
      (1-a)*theta. No consolidate() call is ever needed or used. The EMA anchor is OUR
      adaptation of the paper's end-of-task snapshot to make the method boundary-free
      (honestly labelled [B->N] in docs/CONTINUAL_CITATION_BAR.md).
    boundary=True (apples-to-apples with EWC): consolidate(model, X, y) estimates the
      Fisher on the finished task exactly like EWC.consolidate, then applies the sliding
      window F <- gamma*F + (1-gamma)*F_t, snapshots theta*, and applies the ONLINE LAMBDA
      UPDATE lam_t = lam / (1 - gamma**t), which rescales the convex-combination window so
      the effective penalty equals lam * (weighted average of past task Fishers) from the
      first task on. (Sliding window: Chaudhry et al. 2018; online-lambda rescaling:
      Schwarz et al. 2018 "Progress & Compress". Our convex-combination normalisation is
      an implementation choice, documented in docs/CONTINUAL_CITATION_BAR.md.)
    """

    def __init__(self, lam=50.0, gamma=0.995, boundary=False, anchor_decay=0.999):
        self.lam = lam
        self.gamma = gamma
        self.boundary = boundary
        self.anchor_decay = anchor_decay
        self.t = 0                          # consolidations seen (boundary mode)
        self.lam_t = lam                    # online lambda (updated in consolidate)
        self.FW = self.Fb = None            # running Fisher diagonal
        self.starW = self.starb = None      # anchor

    def _lazy_init(self, model):
        if self.FW is None:
            self.FW = [np.zeros_like(w) for w in model.W]
            self.Fb = [np.zeros_like(b) for b in model.b]
            self.starW = [w.copy() for w in model.W]
            self.starb = [b.copy() for b in model.b]

    def add_penalty_grads(self, model, gW, gb):
        self._lazy_init(model)
        if not self.boundary:
            # raw task grads are in hand BEFORE the penalty is added in place:
            # sliding-window Fisher EMA + slow-EMA anchor, both at the current theta.
            d = 1.0 - self.gamma
            a = self.anchor_decay
            for i in range(len(gW)):
                self.FW[i] = self.gamma * self.FW[i] + d * gW[i] ** 2
                self.Fb[i] = self.gamma * self.Fb[i] + d * gb[i] ** 2
                self.starW[i] = a * self.starW[i] + (1 - a) * model.W[i]
                self.starb[i] = a * self.starb[i] + (1 - a) * model.b[i]
        for i in range(len(gW)):
            gW[i] += self.lam_t * self.FW[i] * (model.W[i] - self.starW[i])
            gb[i] += self.lam_t * self.Fb[i] * (model.b[i] - self.starb[i])

    def consolidate(self, model, X, y, n_samples=1024, rng=None):
        """Boundary-mode only: estimate Fisher on the finished task, apply the sliding
        window, snapshot the anchor, and apply the online-lambda update. In the default
        boundary-free mode this method is a no-op (the learner never needs it)."""
        if not self.boundary:
            return
        rng = rng or np.random.default_rng(0)
        self._lazy_init(model)
        idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
        FtW = [np.zeros_like(w) for w in model.W]
        Ftb = [np.zeros_like(b) for b in model.b]
        for j in idx:
            gW, gb, _ = model.grads(X[j:j + 1], y[j:j + 1])
            for i in range(len(FtW)):
                FtW[i] += gW[i] ** 2
                Ftb[i] += gb[i] ** 2
        d = 1.0 - self.gamma
        for i in range(len(FtW)):
            self.FW[i] = self.gamma * self.FW[i] + d * FtW[i] / len(idx)
            self.Fb[i] = self.gamma * self.Fb[i] + d * Ftb[i] / len(idx)
            self.starW[i] = model.W[i].copy()
            self.starb[i] = model.b[i].copy()
        self.t += 1
        self.lam_t = self.lam / max(1.0 - self.gamma ** self.t, 1e-8)


class MAS:
    """Memory Aware Synapses (Aljundi et al., NeurIPS 2018), BOUNDARY-FREE online form.

    Importance: omega <- gamma_o*omega + (1-gamma_o)*|d/dtheta 0.5*||f(x)||^2| per batch
    (the paper's unsupervised output-L2 surrogate; the paper accumulates over a task and
    consolidates at the task end -- our per-batch EMA + slow-EMA anchor keeps the stream
    boundary-free, an adaptation labelled [B->N] in docs/CONTINUAL_CITATION_BAR.md).
    Penalty: lam * sum_i omega_i * (theta_i - theta*_i)^2, added to the task gradient.
    """

    def __init__(self, lam=100.0, omega_decay=0.995, anchor_decay=0.999):
        self.lam = lam
        self.omega_decay = omega_decay
        self.anchor_decay = anchor_decay
        self.omegaW = self.omegab = None    # importance diagonal
        self.starW = self.starb = None      # anchor

    def _lazy_init(self, model):
        if self.omegaW is None:
            self.omegaW = [np.zeros_like(w) for w in model.W]
            self.omegab = [np.zeros_like(b) for b in model.b]
            self.starW = [w.copy() for w in model.W]
            self.starb = [b.copy() for b in model.b]

    def observe(self, model, X):
        """Per-batch importance + anchor update (called by MLP.fit_task via the optional
        observe hook; no task boundary involved)."""
        self._lazy_init(model)
        gW, gb = model.l2_grads(X)
        d = 1.0 - self.omega_decay
        a = self.anchor_decay
        for i in range(len(gW)):
            self.omegaW[i] = self.omega_decay * self.omegaW[i] + d * np.abs(gW[i])
            self.omegab[i] = self.omega_decay * self.omegab[i] + d * np.abs(gb[i])
            self.starW[i] = a * self.starW[i] + (1 - a) * model.W[i]
            self.starb[i] = a * self.starb[i] + (1 - a) * model.b[i]

    def add_penalty_grads(self, model, gW, gb):
        self._lazy_init(model)
        for i in range(len(gW)):
            gW[i] += self.lam * self.omegaW[i] * (model.W[i] - self.starW[i])
            gb[i] += self.lam * self.omegab[i] * (model.b[i] - self.starb[i])

    def consolidate(self, model, X=None, y=None, n_samples=None, rng=None):
        """No-op by design: MAS is boundary-free. Present only so a runner that
        unconditionally calls consolidate() (if it exists) stays correct."""
        return


class SI:
    """Synaptic Intelligence (Zenke et al., ICML 2017). BOUNDARY-USING (honest label):
    the running path integral w_i = sum -g_i*dtheta_i is only converted into the kept
    importance Omega_i = max(Omega_i, w_i / (Delta_i^2 + xi)) at a task boundary
    (consolidate), exactly like the EWC baseline shipped above.

    Implementation notes (documented in docs/CONTINUAL_CITATION_BAR.md):
      * assumes the constant-SGD schedule of MLP.fit_task; lr is passed in the
        constructor because the fit_task hook does not carry it. Under plain SGD,
        lr*g_i^2 == -g_i*dtheta_i, so the accumulation below IS the path integral with
        the penalty's own contribution to dtheta ignored (a common practical
        simplification).
      * xi is the damping term of Zenke et al. 2017 (their 1e-3 default practice).
    """

    def __init__(self, lam=1.0, lr=0.1, xi=1e-3):
        self.lam = lam
        self.lr = lr
        self.xi = xi
        self.omW = self.omb = None          # Omega (kept importance)
        self.wW = self.wb = None            # running path integral
        self.dW = self.db = None            # running path length (sum lr*|g|)
        self.starW = self.starb = None      # anchor

    def _lazy_init(self, model):
        if self.omW is None:
            self.omW = [np.zeros_like(w) for w in model.W]
            self.omb = [np.zeros_like(b) for b in model.b]
            self.wW = [np.zeros_like(w) for w in model.W]
            self.wb = [np.zeros_like(b) for b in model.b]
            self.dW = [np.zeros_like(w) for w in model.W]
            self.db = [np.zeros_like(b) for b in model.b]
            self.starW = [w.copy() for w in model.W]
            self.starb = [b.copy() for b in model.b]

    def add_penalty_grads(self, model, gW, gb):
        self._lazy_init(model)
        for i in range(len(gW)):
            # accumulate the path integral on the RAW task grads (before the penalty is
            # added in place); lr*g^2 == -g*dtheta for unpenalized SGD steps.
            self.wW[i] += self.lr * gW[i] ** 2
            self.wb[i] += self.lr * gb[i] ** 2
            self.dW[i] += self.lr * np.abs(gW[i])
            self.db[i] += self.lr * np.abs(gb[i])
            gW[i] += self.lam * self.omW[i] * (model.W[i] - self.starW[i])
            gb[i] += self.lam * self.omb[i] * (model.b[i] - self.starb[i])

    def consolidate(self, model, X=None, y=None, n_samples=None, rng=None):
        """Task boundary: normalise w into Omega, reset the integrators, snapshot anchor."""
        self._lazy_init(model)
        for i in range(len(self.omW)):
            new = self.wW[i] / (self.dW[i] ** 2 + self.xi)
            self.omW[i] = np.maximum(self.omW[i], new)
            new = self.wb[i] / (self.db[i] ** 2 + self.xi)
            self.omb[i] = np.maximum(self.omb[i], new)
            self.wW[i][:] = 0.0
            self.wb[i][:] = 0.0
            self.dW[i][:] = 0.0
            self.db[i][:] = 0.0
            self.starW[i] = model.W[i].copy()
            self.starb[i] = model.b[i].copy()
