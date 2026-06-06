"""
The four continual-learning learners on ONE shared RBF-head readout substrate.

Substrate (identical for all):  h = relu(phi(X) @ W1 + b1) ;  logits = h @ W2 + b2.
Only the LEARNING RULE differs:

  BackpropNet  -- exact backprop (for this 2-layer net, == local PC with W2^T feedback).
  EWC          -- backprop + Fisher penalty; consolidates AT TASK BOUNDARIES (privileged).
  VanillaPC    -- modular substrate, local errors, CONSTANT gate (always plastic). Anchor.
  Prizma        -- modular substrate + surprise-gated metaplasticity (PGM). Three modes:
                    "taskfree"  : consolidation ratchets continuously from running surprise
                                  + usage; NO task-boundary signal anywhere (the honest bar).
                    "boundary"  : consolidation snapshot triggered between tasks (reproduces the
                                  committee prototype; uses a task boundary -> labelled as such).
                    "off"       : gate disabled / always plastic -> the noPGM ablation.

No-weight-transport option: feedback="random" replaces W2^T in the hidden error signal by a
fixed random matrix B (DFA / Feedback-Alignment, Lillicrap/Nokland) to test open-problem P2.
"""

from __future__ import annotations

import numpy as np


def softmax(z):
    z = np.clip(z, -60.0, 60.0)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Net:
    """Shared 2-layer readout over a frozen feature lift."""

    def __init__(self, d_feat, H, K, seed, feedback="exact"):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1.0 / np.sqrt(d_feat), (d_feat, H)).astype(np.float32)
        self.b1 = np.zeros(H, np.float32)
        self.W2 = rng.normal(0, 1.0 / np.sqrt(H), (H, K)).astype(np.float32)
        self.b2 = np.zeros(K, np.float32)
        self.feedback = feedback
        if feedback == "random":
            self.B = rng.normal(0, 1.0 / np.sqrt(K), (K, H)).astype(np.float32)  # delta(n,K) -> (n,H)
        self.H, self.K = H, K

    def forward(self, Phi):
        pre = Phi @ self.W1 + self.b1
        h = np.maximum(0.0, pre)
        logits = h @ self.W2 + self.b2
        return pre, h, logits

    def predict_logits(self, Phi):
        return self.forward(Phi)[2]

    def local_grads(self, Phi, Y, y):
        """Local error-neuron gradients (no global chain rule).
        Returns gW1,gb1,gW2,gb2, plus dh (hidden error) and per-sample loss.
        For a 2-layer net the exact-feedback version equals backprop; the random-feedback
        version is DFA and is genuinely non-backprop.
        """
        n = len(Phi)
        pre, h, logits = self.forward(Phi)
        P = softmax(logits)
        delta = (P - Y) / n                                  # output error neurons (n,K)
        gW2 = h.T @ delta                                    # local: pre-activity (x) error
        gb2 = delta.sum(axis=0)
        if self.feedback == "random":
            dh = (delta @ self.B) * (pre > 0)                # DFA: fixed random feedback
        else:
            dh = (delta @ self.W2.T) * (pre > 0)             # PC feedback through generative W2
        gW1 = Phi.T @ dh
        gb1 = dh.sum(axis=0)
        loss = float(-np.log(P[np.arange(n), y] + 1e-12).mean())
        return gW1, gb1, gW2, gb2, dh, loss


# --------------------------------------------------------------------------- #
class BackpropNet(Net):
    def fit_task(self, Phi, Y, y, epochs, batch, lr, rng, ewc=None):
        n = len(Phi)
        for _ in range(epochs):
            idx = rng.permutation(n)
            for s in range(0, n, batch):
                bi = idx[s:s + batch]
                gW1, gb1, gW2, gb2, _, _ = self.local_grads(Phi[bi], Y[bi], y[bi])
                if ewc is not None:
                    ewc.add(self, gW1, gb1, gW2, gb2)
                self.W1 -= lr * gW1; self.b1 -= lr * gb1
                self.W2 -= lr * gW2; self.b2 -= lr * gb2


class EWC:
    """Elastic Weight Consolidation; uses explicit task boundaries (privileged baseline)."""

    def __init__(self, lam=20.0):
        self.lam = lam
        self.snaps = []   # (W1*,b1*,W2*,b2*)
        self.fish = []    # (F1,Fb1,F2,Fb2)

    def add(self, net, gW1, gb1, gW2, gb2):
        for (W1s, b1s, W2s, b2s), (F1, Fb1, F2, Fb2) in zip(self.snaps, self.fish):
            gW1 += self.lam * F1 * (net.W1 - W1s)
            gb1 += self.lam * Fb1 * (net.b1 - b1s)
            gW2 += self.lam * F2 * (net.W2 - W2s)
            gb2 += self.lam * Fb2 * (net.b2 - b2s)

    def consolidate(self, net, Phi, Y, y, n_samples=512, rng=None):
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(len(Phi), size=min(n_samples, len(Phi)), replace=False)
        F1 = np.zeros_like(net.W1); Fb1 = np.zeros_like(net.b1)
        F2 = np.zeros_like(net.W2); Fb2 = np.zeros_like(net.b2)
        for j in idx:
            gW1, gb1, gW2, gb2, _, _ = net.local_grads(Phi[j:j + 1], Y[j:j + 1], y[j:j + 1])
            F1 += gW1 ** 2; Fb1 += gb1 ** 2; F2 += gW2 ** 2; Fb2 += gb2 ** 2
        m = len(idx)
        self.fish.append((F1 / m, Fb1 / m, F2 / m, Fb2 / m))
        self.snaps.append((net.W1.copy(), net.b1.copy(), net.W2.copy(), net.b2.copy()))


# --------------------------------------------------------------------------- #
class Prizma(Net):
    """Modular surprise-gated metaplastic learner."""

    def __init__(self, d_feat, H, K, seed, M=12, feedback="exact",
                 mode="taskfree", beta=10.0, floor=0.04, ema=0.8, topk=3,
                 eta_c=0.05, omega_max=20.0, omega_consol=5.0, usage_min=0.20,
                 usage_engage=0.05, load_balance=0.2, global_gate=True,
                 global_floor=0.18, solved_floor=0.12, tenure_min=40,
                 lr_floor=0.0, reawaken=True, kappa=0.1):
        super().__init__(d_feat, H, K, seed, feedback=feedback)
        assert H % M == 0, "H must be divisible by M"
        self.M, self.gh = M, H // M
        self.slices = [slice(g * self.gh, (g + 1) * self.gh) for g in range(M)]
        self.mode = mode
        self.beta, self.floor, self.ema, self.topk = beta, floor, ema, topk
        self.eta_c, self.omega_max, self.omega_consol = eta_c, omega_max, omega_consol
        self.usage_min, self.usage_engage = usage_min, usage_engage
        self.load_balance = load_balance
        self.global_gate, self.global_floor = global_gate, global_floor
        self.solved_floor, self.tenure_min = solved_floor, tenure_min
        self.lr_floor, self.reawaken, self.kappa = lr_floor, reawaken, kappa
        # PGM state per group
        self.s_bar = np.zeros(M, np.float32)         # surprise EMA (per-sample scale)
        self.omega = np.zeros(M, np.float32)         # consolidation
        self.usage = np.zeros(M, np.float32)         # running win fraction
        self.tenure = np.zeros(M, np.int64)          # wins accrued since last consolidation phase
        self.win_count = np.zeros(M, np.int64)
        self.gE = 1.0                                # slow EMA of global error (phase detector)
        self.was_solved = False                      # previous solved-state (rising-edge latch)

    def _active_set(self, global_err, consolidated):
        """Build a stable active set of <=topk non-consolidated groups (ART-style):
        keep the committed ones, recruit fresh capacity to fill, and recruit NOTHING when
        the batch is already handled (global_err < global_floor) -> reserves capacity."""
        avail = np.where(~consolidated)[0]
        if global_err < self.global_floor or len(avail) == 0:
            return np.array([], int)
        engaged = avail[self.usage[avail] > self.usage_engage]
        active = list(engaged[np.argsort(-self.usage[engaged])][:self.topk])
        if len(active) < self.topk:
            rest = [g for g in avail[np.argsort(self.usage[avail])] if g not in active]
            active += rest[:self.topk - len(active)]
        return np.array(active, int)

    def train_batch(self, Phi, Y, y, lr):
        gW1, gb1, gW2, gb2, dh, loss = self.local_grads(Phi, Y, y)
        # per-SAMPLE surprise (do NOT divide by n): bid_g = mean |raw error projected to group|
        pre = self.forward(Phi)[0]
        P = softmax(self.forward(Phi)[2])
        raw = (P - Y)
        fb = self.B if self.feedback == "random" else self.W2.T
        dh_raw = (raw @ fb) * (pre > 0)
        bids = np.array([np.abs(dh_raw[:, sl]).mean() for sl in self.slices], np.float32)
        self.s_bar = self.ema * self.s_bar + (1 - self.ema) * bids
        global_err = float(np.abs(raw).mean())

        consolidated = (self.omega >= self.omega_consol) if self.mode != "off" else np.zeros(self.M, bool)

        if self.mode == "off":
            winners = np.arange(self.M)
            gates = np.ones(self.M, np.float32)
        else:
            winners = self._active_set(global_err, consolidated)
            gates = np.zeros(self.M, np.float32)
            if len(winners) > 0:
                gw = 1.0 / (1.0 + np.exp(-self.beta * (self.s_bar[winners] - self.floor)))
                gates[winners] = self.lr_floor + (1 - self.lr_floor) * gw

        for g in winners if self.mode != "off" else range(self.M):
            ge = gates[g]
            if ge <= 1e-6:
                continue
            sl = self.slices[g]
            self.W1[:, sl] -= lr * ge * gW1[:, sl]
            self.b1[sl]    -= lr * ge * gb1[sl]
            self.W2[sl, :] -= lr * ge * gW2[sl, :]
        self.b2 -= lr * float(gates.mean()) * gb2

        won = np.zeros(self.M, np.float32)
        if self.mode != "off" and len(winners) > 0:
            won[winners] = 1.0
            self.win_count[winners] += 1
            self.tenure[winners] += 1
        self.usage = 0.99 * self.usage + 0.01 * won

        # task-free phase detector: when the net's own error falls (current niche SOLVED),
        # on the rising edge into "solved" latch the groups that built tenure on this niche.
        # This is driven purely by internal error dynamics -- NO external task-boundary label.
        if self.mode == "taskfree":
            self.gE = 0.99 * self.gE + 0.01 * global_err
            solved = self.gE < self.solved_floor
            if solved and not self.was_solved:
                latch = (self.tenure > self.tenure_min) & (~consolidated)
                self.omega[latch] = self.omega_consol + 1.0
                self.tenure[:] = 0                       # new phase begins
            self.was_solved = solved
            if self.reawaken and consolidated.all():     # occupied-expert / recruitment fix
                g = int(np.argmax(self.s_bar))
                self.omega[g] = max(0.0, self.omega[g] - self.kappa)
        return loss

    def consolidate_boundary(self):
        """boundary mode: freeze the groups that built tenure on the just-finished task (uses
        an EXTERNAL task boundary -> honestly labelled as boundary-dependent)."""
        if self.mode != "boundary":
            return
        latch = (self.tenure > self.tenure_min)
        self.omega[latch] = self.omega_consol + 1.0
        self.tenure[:] = 0

    def fit_task(self, Phi, Y, y, epochs, batch, lr, rng):
        n = len(Phi)
        for _ in range(epochs):
            idx = rng.permutation(n)
            for s in range(0, n, batch):
                bi = idx[s:s + batch]
                self.train_batch(Phi[bi], Y[bi], y[bi], lr)

    def state(self):
        return {"omega": np.round(self.omega, 2).tolist(),
                "usage": np.round(self.usage, 3).tolist(),
                "consolidated": int((self.omega >= self.omega_consol).sum()),
                "win_count": self.win_count.tolist()}
