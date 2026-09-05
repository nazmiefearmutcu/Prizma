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


# ---- Interleaved-router lever constants (PR-2026-09-03-06; frozen a priori) ---- #
# WHY frozen: these are hyperparameters of the FIX MECHANISMS, not tuned knobs. Freezing
# them before any lever run keeps the exploratory selection study a search over the four
# lever strengths only, not over their internals (pre-registration discipline).
HOT_YOUNG_TAU = 200.0      # hot_young decay constant: multiplier is 1+hot*e^-1 at n_seen=200
DV_ALPHA_FAST = 0.2        # novelty-EMA fast component: ~5-batch spike response
DV_ALPHA_SLOW = 0.02       # novelty-EMA slow component: ~50-batch regime window
DV_THETA_LO, DV_THETA_HI = 0.25, 4.0   # clamp for the vigilance scale factor
SAMPLE_NOVEL_FRACTION = 0.5            # sample_top: batch is NOVEL iff novel fraction >= this


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
                 noise_in_std=0.0, noise_act_std=0.0, noise_weight_std=0.0,
                 freeze_min_seen=0, dynamic_vigilance=0.0, hot_young=0.0,
                 route_stat="batch_mean",
                 train_granularity="batch", session_window=0):
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

        # ---- Interleaved-router levers (synthesis C4 + fusion spec 3.3-A/B; PR-06) ---- #
        # All DEFAULT-OFF: at these values every code path below is bit-identical to the
        # shipped behaviour (guarded branches; no rng consumption, no extra float ops on
        # shipped paths). The two documented interleaved failure modes they attack:
        #   (a) mixed batches make batch-mean surprise stationary -> batch-level vigilance
        #       is blind (route_stat attacks the statistic);
        #   (b) round-robin batches freeze immature precision floors that then act as
        #       catch-all recognizers (freeze_min_seen vetoes consolidating those floors;
        #       hot_young makes young experts train hot so floors mature faster).
        # dynamic_vigilance lets the recognition threshold adapt to the stream regime
        # (many NOVEL batches -> loosen; long no-novelty stretches -> tighten) without
        # per-batch whiplash.
        if route_stat not in ("batch_mean", "sample_top"):
            raise ValueError(f"route_stat must be 'batch_mean' or 'sample_top', "
                             f"got {route_stat!r}")
        self.freeze_min_seen = int(freeze_min_seen)   # 0 = shipped: freeze at first transition
        self.dynamic_vigilance = float(dynamic_vigilance)  # 0.0 = shipped: fixed z threshold
        self.hot_young = float(hot_young)             # 0.0 = shipped: constant per-expert lr
        self.route_stat = route_stat                  # "batch_mean" = shipped statistic
        # dynamic-vigilance state: causal EMAs of the per-batch NOVEL label (0 init = the
        # stream is assumed recognized until evidence accumulates; no lookahead).
        self._nov_fast = 0.0
        self._nov_slow = 0.0

        # ---- Training-granularity levers (G1/G2; fusion spec Addendum 2026-09-05) ---- #
        # PR-06's blocker: train_batch applies the whole batch's local delta to the single
        # active expert, so a mixed batch can never be split across experts -- a threshold/
        # statistic repair cannot recover block-stream specialization. These two knobs
        # change WHO trains on WHAT (the decision unit), never the local learning rule
        # (PC/FA/delta stays exactly the shipped one):
        #   G1  train_granularity="sample" -- each sample of the batch is routed
        #       individually against the per-expert calibrated per-sample thresholds
        #       (mu + z*sigma, i.e. the sample_top statistic applied per sample); each
        #       sample trains the expert that recognizes IT; samples recognized by nobody
        #       accrue to a per-batch novel pool and recruitment fires for the batch iff
        #       the novel-pool fraction >= SAMPLE_NOVEL_FRACTION (the existing recruit
        #       machinery: commit+freeze+advance, new active trained on the novel pool).
        #   G2  session_window=W>0 -- the batch is split into consecutive W-sample windows
        #       (last window short when batch % W != 0); each window gets the shipped
        #       BATCH-level decision at window granularity (window-MEAN surprise vs the
        #       same calibrated thresholds; novel window -> recruit, freeze_min_seen veto
        #       still applies). The batch-mean blindness shrinks with W.
        # Both are DEFAULT-OFF, mutually exclusive, and ZERO extra work at defaults
        # (guarded branches; the shipped path is the shipped body verbatim). When either
        # is active the decision unit is smaller than the batch, so the route_stat knob
        # (a choice of BATCH-level statistic) is inert/irrelevant: G1 uses the per-sample
        # thresholds internally and G2 applies the batch_mean statistic per window --
        # route_stat is simply not consulted on those paths.
        if train_granularity not in ("batch", "sample"):
            raise ValueError(f"train_granularity must be 'batch' or 'sample', "
                             f"got {train_granularity!r}")
        self.train_granularity = train_granularity    # "batch" = shipped granularity
        self.session_window = int(session_window)     # 0 = shipped (no windowing)
        if self.session_window < 0:
            raise ValueError(f"session_window must be >= 0, got {session_window!r}")
        if self.train_granularity == "sample" and self.session_window > 0:
            raise ValueError("train_granularity='sample' and session_window>0 are "
                             "mutually exclusive training-granularity levers")

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
    def _hot_mult(self, n_seen):
        """Metaplasticity multiplier for an expert with n_seen samples so far.

        1 + hot_young*exp(-n_seen / HOT_YOUNG_TAU): young experts train hot, decaying to
        1 + hot_young*e^-1 (~1.37x at hot_young=1) at n_seen=200 and ~1 as the expert
        matures. Attaches the fusion-spec 3.3-B per-expert metaplasticity to the expert's
        OWN sample count (a purely local, label-free quantity)."""
        if self.hot_young <= 0:
            return 1.0
        return 1.0 + self.hot_young * math.exp(-n_seen / HOT_YOUNG_TAU)

    def _note_novelty(self, novel):
        """Update the causal novelty-rate EMAs with this batch's NOVEL label.

        Two EMAs: fast (alpha=0.2, ~5 batches) responds to novelty spikes; slow
        (alpha=0.02, ~50 batches) tracks the stream regime. Their MAX is the smoothed
        novelty rate: fast attack, slow release (hysteresis), so the effective vigilance
        adapts to regime changes without batch-to-batch whiplash. No-op when the lever is
        OFF (bit-identity: no state, no arithmetic on the shipped path)."""
        if self.dynamic_vigilance <= 0:
            return
        v = 1.0 if novel else 0.0
        self._nov_fast += DV_ALPHA_FAST * (v - self._nov_fast)
        self._nov_slow += DV_ALPHA_SLOW * (v - self._nov_slow)

    def _theta_scale(self):
        """Vigilance scale s: theta_eff = s * (mu + z*sigma).

        s = clamp(1 + dv*(novelty_rate_smoothed - 0.5), 0.25, 4): at novelty rate 0 the
        threshold TIGHTENS (x 1 - dv/2) -- the regime where a stale catch-all floor
        recognizes everything; at rate 1 it LOOSENS (x 1 + dv/2) -- a genuinely novel
        regime should recruit, not thrash. Clamped to [0.25, 4] x the base threshold."""
        nov = max(self._nov_fast, self._nov_slow)
        return min(DV_THETA_HI, max(DV_THETA_LO,
                                    1.0 + self.dynamic_vigilance * (nov - 0.5)))

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

        # Metaplasticity: the multiplier is read from the PRE-update n_seen (the step the
        # expert is about to take) and scales EVERY local delta step below (decoder, head,
        # encoder). hot_young=0 leaves lr/lr_cls bit-identical (plain attrs, no re-multiply).
        if self.hot_young > 0:
            m_hot = self._hot_mult(e.n_seen)
            lr, lr_cls = self.lr * m_hot, self.lr_cls * m_hot
        else:
            lr, lr_cls = self.lr, self.lr_cls

        Z, EPS, logits = e.forward(X, Y)
        P = softmax(logits, axis=1)
        D = (P - Y)

        Wdec_eff = e._get_weight(e.Wdec)
        Wcls_eff = e._get_weight(e.Wcls)

        # decoder: exact local PC rule  dWdec ~ eps (x) z
        e.Wdec += lr * (EPS.T @ Z) / n
        e.bdec += lr * EPS.mean(0)
        # head: local delta rule  dWcls ~ (p - y) (x) z
        e.Wcls -= lr_cls * (D.T @ Z) / n
        e.bcls -= lr_cls * D.mean(0)

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

        e.Wenc += lr * (g_lat.T @ X_in) / n
        e.benc += lr * g_lat.mean(0)
        e.n_seen += n
        
        # update this expert's PRECISION over its own (post-update) reconstruction surprise
        r = float(e.recon_error(X).mean())
        if e.mu > 1e8:
            e.mu, e.var = r, max(1e-4, (0.1 * r) ** 2)
        else:
            d_val = r - e.mu
            e.mu += 0.05 * d_val
            e.var = 0.95 * e.var + 0.05 * d_val * d_val

    def _threshold(self, e):
        """Expert e's effective recognition threshold: theta_scale*(mu + z*sigma) under
        dynamic_vigilance, else the shipped expression verbatim (identical float ops --
        bit-identity). Shared by _recognizes and the G1 per-sample routing so every
        granularity reads the SAME calibrated per-expert scale."""
        if self.dynamic_vigilance > 0:
            return self._theta_scale() * (e.mu + self.z_novel * math.sqrt(e.var))
        return e.mu + self.z_novel * math.sqrt(e.var)

    def _recognizes(self, e, X):
        """Precision test: does expert e recognize this batch (recon within z*sigma of floor)?

        route_stat='batch_mean' (shipped): the BATCH-MEAN surprise must sit under the
        per-expert threshold. Blind to mixed batches: a 5-domain mixture has a stationary
        mean no matter which domains are in it, so vigilance never fires (failure mode (a)).
        route_stat='sample_top': the SAME calibrated threshold (mu + z*sigma, no new
        calibration) is applied PER SAMPLE, and the batch counts as recognized iff the
        novel fraction is < SAMPLE_NOVEL_FRACTION. The decision stays batch-level (the
        repo's v2 lesson: per-sample DECISIONS thrash); only the statistic changes, and
        the novel fraction is stationary-robust: a 60/40 mixed batch is 60% novel for a
        pure-domain expert even when its mean hides the change. dynamic_vigilance > 0
        scales the threshold by _theta_scale(); at 0 the shipped expression is used
        verbatim (bit-identity). route_stat is only consulted on BATCH-level decisions:
        under train_granularity='sample' or session_window>0 the decision unit itself
        shrinks and this method is either bypassed (G1) or called per window with the
        batch_mean statistic (G2)."""
        if e.mu > 1e8:
            return False
        S = e.recon_error(X)
        thr = self._threshold(e)
        if self.route_stat == "sample_top":
            return float((S > thr).mean()) < SAMPLE_NOVEL_FRACTION
        return float(S.mean()) <= thr

    def train_batch(self, X, Y, y):
        n = len(X)
        if not self.route:
            # ABLATION: no recognition-routing, no phase detection -> a single monolithic local
            # learner trained on every batch. Expected to forget like naive backprop.
            self._train_expert(self.experts[0], X, Y)
            self.route_log[0] += n
            return
        if self.train_granularity == "sample":
            self._train_batch_by_sample(X, Y)          # G1: per-sample routed training
            return
        if self.session_window > 0:                    # G2: window-level decisions
            W = self.session_window
            for s in range(0, n, W):                   # batch % W != 0 -> last short window
                self._train_unit(X[s:s + W], Y[s:s + W], window=True)
            return
        self._train_unit(X, Y)                         # shipped: the whole batch is the unit

    def _recognizes_mean(self, e, X):
        """Window-level recognition (G2, session_window>0 ONLY): the WINDOW-MEAN surprise
        against the same per-expert calibrated threshold (mu + z*sigma, dynamic_vigilance
        scales it identically). route_stat is deliberately NOT consulted here: the
        decision unit is the window, so the batch-level statistic choice does not apply
        (fusion spec Addendum 2026-09-05: under granularity != batch the route_stat knob
        is irrelevant). The statistic changes unit -- batch-mean blindness shrinks with W
        -- but no new calibration is introduced."""
        if e.mu > 1e8:
            return False
        return float(e.recon_error(X).mean()) <= self._threshold(e)

    def _train_unit(self, X, Y, window=False):
        """The shipped decision unit (batch-level routing + the recruit machinery).

        train_granularity="batch"/session_window=0 calls this with the whole batch (the
        body below is the shipped train_batch logic VERBATIM -- bit-identity);
        session_window=W>0 calls it once per consecutive W-sample window with window=True,
        so every decision here (committed scan, active-expert phase check,
        commit/freeze/advance) is taken at window granularity via the window-mean
        statistic (_recognizes_mean) while the learning rule is untouched."""
        n = len(X)
        committed = [m for m in range(self.M) if self.experts[m].committed]

        # 1) an OLD domain reappearing -> recognized by a committed (frozen) expert: nothing to
        #    learn (its weights are protected); inference will route there. No update.
        for m in committed:
            ok = (self._recognizes_mean(self.experts[m], X) if window
                  else self._recognizes(self.experts[m], X))
            if ok:
                self.route_log[m] += n
                self._note_novelty(False)
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
        ok = (self._recognizes_mean(act, X) if window else self._recognizes(act, X))
        if young or ok:
            self._train_expert(act, X, Y)
            self.route_log[self.active] += n
            self._note_novelty(False)
        else:
            act.committed = True
            # Freeze-immaturity veto: consolidation may not freeze an expert whose
            # precision floor saw fewer than freeze_min_seen samples (failure mode (b):
            # an immature floor is loose in ABSOLUTE terms, so once frozen it recognizes
            # every later domain and silently becomes a catch-all that stops all
            # training). 0 = shipped behaviour (always freeze on commit).
            if self.consolidate and act.n_seen >= self.freeze_min_seen:
                act.frozen = True
                act.omega = self.omega_consol + 1.0
            self.active += 1
            if self.active < self.M:
                self._train_expert(self.experts[self.active], X, Y)
                self.route_log[self.active] += n
            self._note_novelty(True)

    def _train_batch_by_sample(self, X, Y):
        """G1 (train_granularity="sample"): per-sample routed training.

        Each SAMPLE is routed individually against the per-expert calibrated thresholds
        (self._threshold, i.e. the sample_top statistic applied per sample -- route_stat
        itself is not consulted on this path). Priority mirrors the shipped unit:
          1. a COMMITTED expert claims every sample it recognizes -> routed there, never
             re-trained (protected old domain; scan in expert-index order, first wins);
          2. of the rest, the ACTIVE expert trains on the samples IT recognizes (one
             _train_expert call on that sub-batch -- the shipped local rule, applied to
             the sub-batch);
          3. samples recognized by NO expert accrue to the per-batch novel pool. If the
             novel-pool fraction >= SAMPLE_NOVEL_FRACTION, recruitment fires for the
             batch via the EXISTING machinery (commit + freeze_min_seen-vetoed freeze +
             advance) and the new active is trained on the novel pool. A never-trained
             active (n_seen == 0) is not committed -- it simply takes the novel pool as
             its first training (its floor then calibrates on it). Sub-threshold novel
             minorities are left untrained for this batch: forcing them onto the active
             expert is exactly the catch-all mechanism PR-06 diagnosed; they can recruit
             later when their fraction grows.
        Design notes: the active's training uses the PRE-update floors for both decisions
        (no within-batch lookahead); the shipped young-expert forced-training rule is
        deliberately NOT carried over (forcing a young expert to absorb whole mixed
        batches poisons its floor with mixture data -- the monolithic mechanism PR-06
        identified); route_log counts TRAINED/claimed samples per expert, so it stays an
        honest training ledger."""
        n = len(X)
        calib = [m for m in range(self.M) if not (self.experts[m].mu > 1e8)]
        free = np.ones(n, dtype=bool)          # not yet claimed by a committed expert
        if calib:
            S = self._recon_matrix(X)          # (n, M) per-sample surprise, computed once
            for m in calib:
                if self.experts[m].committed:
                    take = free & (S[:, m] <= self._threshold(self.experts[m]))
                    free &= ~take
                    if take.any():
                        self.route_log[m] += int(take.sum())
        if self.active >= self.M:
            return                             # pool exhausted: shipped hard-cap (drop)
        a = self.active
        act = self.experts[a]
        if calib and not (act.mu > 1e8):
            take = free & (S[:, a] <= self._threshold(act))
        else:
            take = np.zeros(n, dtype=bool)     # no calibrated floor yet: recognizes nothing
        free &= ~take
        if take.any():
            self._train_expert(act, X[take], Y[take])
            self.route_log[a] += int(take.sum())
        # the remaining samples are the per-batch novel pool
        recruit = bool(free.any()) and float(free.mean()) >= SAMPLE_NOVEL_FRACTION
        if recruit:
            if act.n_seen > 0:                 # existing recruit machinery: commit+advance
                act.committed = True
                if self.consolidate and act.n_seen >= self.freeze_min_seen:
                    act.frozen = True
                    act.omega = self.omega_consol + 1.0
                self.active += 1
                if self.active < self.M:
                    act = self.experts[self.active]
            if act.n_seen == 0 and self.active < self.M:
                self._train_expert(act, X[free], Y[free])
                self.route_log[self.active] += int(free.sum())
            self._note_novelty(True)
        else:
            self._note_novelty(False)

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
