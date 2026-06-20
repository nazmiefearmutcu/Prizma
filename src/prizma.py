"""
Prizma -- Surprise-Gated Mixture of Predictive-Coding Experts (ART-routing core).

Honest scope (per the design committee): this prototype targets the regime where the headline
claim is *achievable and meaningful* -- a DOMAIN-INCREMENTAL stream whose domains are
input-distinguishable -- and tests whether Prizma can, with NO task labels and NO task
boundaries, (i) discover the domain structure online, (ii) allocate one expert per domain,
and (iii) protect mastered domains. In the fully-ambiguous shared-input regime (same x,
different label per task) we separately PROVE no single-head learner can retain all tasks;
Prizma is not claimed to help there.

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
    def __init__(self, d, h, K, seed, feedback="random", lambda_cls=1.0,
                 n_settle_steps=0, eta_settle=0.1, langevin_temp=0.0,
                 weight_bits=None, act_bits=None,
                 noise_in_std=0.0, noise_act_std=0.0, noise_weight_std=0.0):
        rng = np.random.default_rng(seed)
        self.rng = rng
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

        # Simulation & hardware constraints
        self.feedback = feedback
        self.lambda_cls = lambda_cls
        self.n_settle_steps = n_settle_steps
        self.eta_settle = eta_settle
        self.langevin_temp = langevin_temp
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.noise_in_std = noise_in_std
        self.noise_act_std = noise_act_std
        self.noise_weight_std = noise_weight_std

    def _quantize(self, x, bits):
        if bits is None or bits <= 0:
            return x
        # Symmetric quantization
        max_val = np.max(np.abs(x))
        if max_val == 0:
            return x
        min_val = -max_val
        qmin = -(2**(bits - 1) - 1)
        qmax = 2**(bits - 1) - 1
        scale = (max_val - min_val) / (qmax - qmin)
        if scale == 0:
            return x
        q_x = np.round(x / scale)
        q_x = np.clip(q_x, qmin, qmax)
        return q_x * scale

    def _get_weight(self, W, rng=None):
        if self.noise_weight_std > 0:
            r = self.rng if rng is None else rng
            W = W + r.normal(0, self.noise_weight_std, W.shape)
        if self.weight_bits is not None:
            W = self._quantize(W, self.weight_bits)
        return W

    def _get_act(self, A, rng=None):
        if self.noise_act_std > 0:
            r = self.rng if rng is None else rng
            A = A + r.normal(0, self.noise_act_std, A.shape)
        if self.act_bits is not None:
            A = self._quantize(A, self.act_bits)
        return A

    def _get_input(self, X, rng=None):
        if self.noise_in_std > 0:
            r = self.rng if rng is None else rng
            X = X + r.normal(0, self.noise_in_std, X.shape)
        if self.act_bits is not None:
            X = self._quantize(X, self.act_bits)
        return X

    def settle(self, X, Y=None, rng=None):
        r = self.rng if rng is None else rng
        
        Wenc_eff = self._get_weight(self.Wenc, r)
        benc_eff = self._get_weight(self.benc, r)
        Wdec_eff = self._get_weight(self.Wdec, r)
        bdec_eff = self._get_weight(self.bdec, r)
        Wcls_eff = self._get_weight(self.Wcls, r)
        bcls_eff = self._get_weight(self.bcls, r)
        
        X_in = self._get_input(X, r)
        
        # Prior/initial activation Z_prior
        Z_prior = np.tanh(X_in @ Wenc_eff.T + benc_eff)
        Z_prior = self._get_act(Z_prior, r)
        
        Z = Z_prior.copy()
        n_steps = self.n_settle_steps
        if n_steps <= 0:
            return Z
            
        eta = self.eta_settle
        temp = self.langevin_temp
        
        for _ in range(n_steps):
            Xhat = Z @ Wdec_eff.T + bdec_eff
            EPS = X_in - Xhat
            
            if self.feedback == "exact":
                g_rec = EPS @ Wdec_eff
            else:
                Bdec_eff = self._get_weight(self.Bdec, r)
                g_rec = EPS @ Bdec_eff.T
                
            if Y is not None:
                logits = Z @ Wcls_eff.T + bcls_eff
                P = softmax(logits, axis=1)
                D = P - Y
                if self.feedback == "exact":
                    g_cls = D @ Wcls_eff
                else:
                    Bcls_eff = self._get_weight(self.Bcls, r)
                    g_cls = D @ Bcls_eff.T
            else:
                g_cls = 0.0
                
            grad_Z = (Z - Z_prior) - g_rec - self.lambda_cls * g_cls
            
            noise = 0.0
            if temp > 0:
                noise = r.normal(0, np.sqrt(2 * temp * eta), Z.shape)
                
            Z = Z - eta * grad_Z + noise
            Z = np.clip(Z, -1.0, 1.0)
            Z = self._get_act(Z, r)
            
        return Z

    def encode(self, X, Y=None):
        return self.settle(X, Y)

    def recon_error(self, X, Y=None):
        Z = self.encode(X, Y)
        Wdec_eff = self._get_weight(self.Wdec)
        bdec_eff = self._get_weight(self.bdec)
        X_in = self._get_input(X)
        Xhat = Z @ Wdec_eff.T + bdec_eff
        return ((X_in - Xhat) ** 2).mean(axis=1)        # per-sample surprise S_m

    def forward(self, X, Y=None):
        Z = self.encode(X, Y)
        Wdec_eff = self._get_weight(self.Wdec)
        bdec_eff = self._get_weight(self.bdec)
        Wcls_eff = self._get_weight(self.Wcls)
        bcls_eff = self._get_weight(self.bcls)
        X_in = self._get_input(X)
        EPS = X_in - (Z @ Wdec_eff.T + bdec_eff)
        logits = Z @ Wcls_eff.T + bcls_eff
        return Z, EPS, logits


class Prizma:
    def __init__(self, d, h, K, n_experts=8, seed=0,
                 lr=0.05, lr_cls=0.1, lambda_cls=1.0, feedback="random",
                 z_novel=5.0, commit_ratio=0.5, commit_after=256, consolidate=True,
                 route=True, eta_c=0.1, omega_consol=3.0,
                 n_settle_steps=0, eta_settle=0.1, langevin_temp=0.0,
                 weight_bits=None, act_bits=None,
                 noise_in_std=0.0, noise_act_std=0.0, noise_weight_std=0.0):
        self.d, self.h, self.K, self.M = d, h, K, n_experts
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

        # Simulation & hardware constraints
        self.n_settle_steps = n_settle_steps
        self.eta_settle = eta_settle
        self.langevin_temp = langevin_temp
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.noise_in_std = noise_in_std
        self.noise_act_std = noise_act_std
        self.noise_weight_std = noise_weight_std

        self.experts = [
            Expert(d, h, K, seed + 100 * (m + 1),
                   feedback=feedback, lambda_cls=lambda_cls,
                   n_settle_steps=n_settle_steps, eta_settle=eta_settle,
                   langevin_temp=langevin_temp, weight_bits=weight_bits,
                   act_bits=act_bits, noise_in_std=noise_in_std,
                   noise_act_std=noise_act_std, noise_weight_std=noise_weight_std)
            for m in range(n_experts)
        ]

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
            
        Z, EPS, logits = e.forward(X, Y)
        P = softmax(logits, axis=1)
        D = (P - Y)
        
        Wdec_eff = e._get_weight(e.Wdec)
        Wcls_eff = e._get_weight(e.Wcls)
        
        # decoder: exact local PC rule  dWdec ~ eps (x) z
        e.Wdec += self.lr * (EPS.T @ Z) / n
        e.bdec += self.lr * EPS.mean(0)
        # head: local delta rule  dWcls ~ (p - y) (x) z
        e.Wcls -= self.lr_cls * (D.T @ Z) / n
        e.bcls -= self.lr_cls * D.mean(0)
        
        X_in = e._get_input(X)
        
        # encoder latent error signals.
        if e.n_settle_steps > 0:
            Wenc_eff = e._get_weight(e.Wenc)
            benc_eff = e._get_weight(e.benc)
            Z_prior = np.tanh(X_in @ Wenc_eff.T + benc_eff)
            Z_prior = e._get_act(Z_prior)
            g_lat = (Z - Z_prior) * (1.0 - Z_prior ** 2)
        else:
            dZ = 1.0 - Z ** 2
            if self.feedback == "exact":
                g_rec = (EPS @ Wdec_eff) * dZ        # Wdec is (d,h); EPS (n,d) -> (n,h)  == W^T path
                g_cls = (D @ Wcls_eff) * dZ          # Wcls is (K,h); D (n,K) -> (n,h)    == W^T path
            else:
                Bdec_eff = e._get_weight(e.Bdec)
                Bcls_eff = e._get_weight(e.Bcls)
                g_rec = (EPS @ Bdec_eff.T) * dZ      # fixed random feedback (DFA)
                g_cls = (D @ Bcls_eff.T) * dZ
            g_lat = g_rec + self.lambda_cls * g_cls
            
        e.Wenc += self.lr * (g_lat.T @ X_in) / n
        e.benc += self.lr * g_lat.mean(0)
        e.n_seen += n
        
        # update this expert's PRECISION over its own (post-update) reconstruction surprise
        r = float(e.recon_error(X).mean())
        if e.mu > 1e8:
            e.mu, e.var = r, max(1e-4, (0.1 * r) ** 2)
        else:
            d_val = r - e.mu
            e.mu += 0.05 * d_val
            e.var = 0.95 * e.var + 0.05 * d_val * d_val

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
