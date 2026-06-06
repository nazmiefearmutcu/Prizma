"""
PRISM -- Surprise-Gated Mixture of Predictive-Coding Experts (ART-routing core).

Honest scope (per the design committee): this prototype targets the regime where the headline
claim is *achievable and meaningful* -- a DOMAIN-INCREMENTAL stream whose domains are
input-distinguishable -- and tests whether PRISM can, with NO task labels and NO task
boundaries, (i) discover the domain structure online, (ii) allocate one expert per domain,
and (iii) protect mastered domains. In the fully-ambiguous shared-input regime (same x,
different label per task) we separately PROVE no single-head learner can retain all tasks;
PRISM is not claimed to help there.

Each EXPERT m = predictive-coding auto-encoder (encoder Wenc, decoder Wdec; the recognizer)
            + classifier head (Wcls). All updates are LOCAL: decoder/head use the exact PC /
delta rule (post-error (x) pre-activity); the encoder is trained with FIXED RANDOM FEEDBACK
(Feedback Alignment, Lillicrap/Nokland) so no W^T is ever read (open-problem P2, relaxed).

Routing = ART-style vigilance on the recognizer's reconstruction surprise S_m (label-free,
works at train AND test). An input is "recognized" by the lowest-surprise committed expert if
S_m < vigilance; otherwise it is NOVEL and a fresh expert is recruited. Consolidation (PGM)
freezes an expert once the stream has moved past its domain -- a purely internal, surprise-
driven, task-boundary-free signal.
"""

from __future__ import annotations

import math

import numpy as np


def softmax(z, axis=-1):
    z = np.clip(z, -60.0, 60.0)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


class Expert:
    def __init__(self, d, h, K, seed):
        rng = np.random.default_rng(seed)
        self.Wenc = rng.normal(0, 1.0 / np.sqrt(d), (h, d)).astype(np.float32)
        self.benc = np.zeros(h, np.float32)
        self.Wdec = rng.normal(0, 1.0 / np.sqrt(h), (d, h)).astype(np.float32)
        self.bdec = np.zeros(d, np.float32)
        self.Wcls = rng.normal(0, 1.0 / np.sqrt(h), (K, h)).astype(np.float32)
        self.bcls = np.zeros(K, np.float32)
        self.Bdec = rng.normal(0, 1.0 / np.sqrt(d), (h, d)).astype(np.float32)   # FA feedback
        self.Bcls = rng.normal(0, 1.0 / np.sqrt(K), (h, K)).astype(np.float32)   # FA feedback
        self.committed = False
        self.frozen = False
        self.omega = 0.0
        self.n_seen = 0
        # per-expert PRECISION over its own reconstruction surprise (mu, var EMAs).
        # A batch is "recognized" by this expert iff its recon < mu + z*sigma; otherwise it
        # is NOVEL. This adapts the recognition threshold to each domain's own noise floor.
        self.mu = 1e9            # recon-floor mean (starts huge -> recognizes nothing yet)
        self.var = 1.0
        self.init_recon = None   # recon on the first batch this expert ever saw (for relative commit)

    def encode(self, X):
        return np.tanh(X @ self.Wenc.T + self.benc)

    def recon_error(self, X):
        Z = self.encode(X)
        Xhat = Z @ self.Wdec.T + self.bdec
        return ((X - Xhat) ** 2).mean(axis=1)        # per-sample surprise S_m

    def forward(self, X):
        Z = self.encode(X)
        Xhat = Z @ self.Wdec.T + self.bdec
        EPS = X - Xhat
        logits = Z @ self.Wcls.T + self.bcls
        return Z, EPS, logits


class PRISM:
    def __init__(self, d, h, K, n_experts=8, seed=0,
                 lr=0.05, lr_cls=0.1, lambda_cls=1.0, feedback="random",
                 z_novel=5.0, commit_ratio=0.5, commit_after=256, consolidate=True,
                 route=True, eta_c=0.1, omega_consol=3.0):
        self.d, self.h, self.K, self.M = d, h, K, n_experts
        self.experts = [Expert(d, h, K, seed + 100 * (m + 1)) for m in range(n_experts)]
        self.lr, self.lr_cls, self.lambda_cls = lr, lr_cls, lambda_cls
        self.feedback = feedback
        self.z_novel = z_novel               # novelty z-score on per-expert recon precision
        self.commit_ratio = commit_ratio     # (unused in active-expert scheme; kept for API)
        self.warmup = commit_after           # samples a fresh active expert trains before its
                                             #   recognition is trusted (precision must settle)
        self.consolidate = consolidate
        self.route = route                   # ablation: route=False -> single monolithic expert
        self.eta_c, self.omega_consol = eta_c, omega_consol
        self.active = 0                      # index of the currently-learning expert
        self.route_log = np.zeros(n_experts, np.int64)

    # ----------------------------- routing ------------------------------------ #
    def _recon_matrix(self, X):
        return np.stack([e.recon_error(X) for e in self.experts], axis=1)   # (n, M)

    def route_for_inference(self, X):
        """Label-free routing: each sample goes to the established (trained) expert that best
        recognizes it (lowest reconstruction surprise)."""
        S = self._recon_matrix(X)
        trained = np.array([e.n_seen > 0 for e in self.experts])
        if trained.any():
            Sc = S.copy(); Sc[:, ~trained] = np.inf
            return np.argmin(Sc, axis=1), S
        return np.argmin(S, axis=1), S

    def predict_logits(self, X):
        idx, _ = self.route_for_inference(X)
        out = np.zeros((len(X), self.K), np.float32)
        for m, e in enumerate(self.experts):
            mask = idx == m
            if mask.any():
                out[mask] = e.forward(X[mask])[2]
        return out

    # ----------------------------- learning ----------------------------------- #
    def _train_expert(self, e, X, Y):
        if e.frozen:
            return
        n = len(X)
        if e.init_recon is None:
            e.init_recon = float(e.recon_error(X).mean())
        Z, EPS, logits = e.forward(X)
        P = softmax(logits, axis=1)
        dZ = 1.0 - Z ** 2
        D = (P - Y)
        # decoder: exact local PC rule  dWdec ~ eps (x) z
        e.Wdec += self.lr * (EPS.T @ Z) / n
        e.bdec += self.lr * EPS.mean(0)
        # head: local delta rule  dWcls ~ (p - y) (x) z
        e.Wcls -= self.lr_cls * (D.T @ Z) / n
        e.bcls -= self.lr_cls * D.mean(0)
        # encoder latent error signals. feedback="random" => FIXED RANDOM feedback (DFA, no W^T
        # anywhere). feedback="exact" => use the generative weights' transpose (the PC ideal,
        # which DOES read W^T in the inference/credit path) -- provided only to MEASURE the cost
        # of the no-weight-transport relaxation (open-problem P2).
        if self.feedback == "exact":
            g_rec = (EPS @ e.Wdec) * dZ        # Wdec is (d,h); EPS (n,d) -> (n,h)  == W^T path
            g_cls = (D @ e.Wcls) * dZ          # Wcls is (K,h); D (n,K) -> (n,h)    == W^T path
        else:
            g_rec = (EPS @ e.Bdec.T) * dZ      # fixed random feedback (DFA)
            g_cls = (D @ e.Bcls.T) * dZ
        g_lat = g_rec + self.lambda_cls * g_cls
        e.Wenc += self.lr * (g_lat.T @ X) / n
        e.benc += self.lr * g_lat.mean(0)
        e.n_seen += n
        # update this expert's PRECISION over its own (post-update) reconstruction surprise
        r = float(e.recon_error(X).mean())
        if e.mu > 1e8:
            e.mu, e.var = r, max(1e-4, (0.1 * r) ** 2)
        else:
            d = r - e.mu
            e.mu += 0.05 * d
            e.var = 0.95 * e.var + 0.05 * d * d

    def _recognizes(self, e, X):
        """Precision test: does expert e recognize this batch (recon within z*sigma of floor)?"""
        if e.mu > 1e8:
            return False
        return float(e.recon_error(X).mean()) <= e.mu + self.z_novel * math.sqrt(e.var)

    def train_batch(self, X, Y, y):
        n = len(X)
        if not self.route:
            # ABLATION: no recognition-routing, no phase detection -> a single monolithic local
            # learner trained on every batch. Expected to forget like naive backprop.
            self._train_expert(self.experts[0], X, Y)
            self.route_log[0] += n
            return
        committed = [m for m in range(self.M) if self.experts[m].committed]

        # 1) an OLD domain reappearing -> recognized by a committed (frozen) expert: nothing to
        #    learn (its weights are protected); inference will route there. No update.
        for m in committed:
            if self._recognizes(self.experts[m], X):
                self.route_log[m] += n
                return

        # 2) otherwise the active expert handles it. While the active expert is YOUNG (precision
        #    not yet settled) we always train it. Once mature, we trust its recognition: if it
        #    still recognizes the batch the SAME domain continues -> keep training; if it no
        #    longer recognizes -> the domain CHANGED -> commit+freeze it and advance to a fresh
        #    expert. The phase transition is read off the active expert's OWN precision -- no
        #    external task-boundary label is ever used.
        if self.active >= self.M:
            return
        act = self.experts[self.active]
        young = act.n_seen < self.warmup
        if young or self._recognizes(act, X):
            self._train_expert(act, X, Y)
            self.route_log[self.active] += n
        else:
            act.committed = True
            if self.consolidate:
                act.frozen = True
                act.omega = self.omega_consol + 1.0
            self.active += 1
            if self.active < self.M:
                self._train_expert(self.experts[self.active], X, Y)
                self.route_log[self.active] += n

    def fit_task(self, X, y, epochs=10, batch=128, rng=None):
        rng = rng or np.random.default_rng(0)
        Yall = np.eye(self.K, dtype=np.float32)[y]
        n = len(X)
        for _ in range(epochs):
            idx = rng.permutation(n)
            for s in range(0, n, batch):
                bi = idx[s:s + batch]
                self.train_batch(X[bi], Yall[bi], y[bi])

    @property
    def n_committed(self):
        return sum(e.committed for e in self.experts)

    def state(self):
        return {
            "committed": [int(e.committed) for e in self.experts],
            "frozen": [int(e.frozen) for e in self.experts],
            "n_seen": [int(e.n_seen) for e in self.experts],
            "route_log": self.route_log.tolist(),
            "active": self.active,
        }
