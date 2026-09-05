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

# ---- G3a generative-replay constants (PR-2026-09-03-07; frozen a priori) ---- #
# WHY frozen: hyperparameters of the FIX MECHANISM internals, not tuned knobs (the
# PR-06 discipline). The probe searches over replay strength only via replay_passes;
# these two are fixed before any run.
REPLAY_H_EMA = 0.05        # hidden-activity statistic EMA rate (per real training call;
                           # matches the per-expert precision-floor EMA rate above)
REPLAY_VAR_FLOOR = 1e-4    # diagonal-variance floor -> generation never degenerates to
                           # a delta at the mean (a single-direction pseudo-set)

# ---- G3b probationary-commit constants (PR-2026-09-03-07; frozen a priori) ---- #
# WHY frozen: the patience window is a hyperparameter of the FIX MECHANISM, not a tuned
# knob (the PR-06 discipline): frozen before any probation run.
PROBATION_PATIENCE = 8     # consecutive batches with ZERO recognized free live samples
                           # before a probationary expert's domain is declared passed
                           # (the per-batch surprise-floor exit test, smoothed)


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

        # G3a generative-replay state: a running DIAGONAL-GAUSSIAN fit (mean + per-dim
        # variance EMA) of this expert's hidden activity h over the REAL batches it
        # trained on. Maintained ONLY while the replay lever is ON (replay_passes > 0);
        # None until the first real training call. Additive and cheap: two h-vectors
        # per training call. No raw inputs are ever stored (buffer-free by design).
        self.h_mean = None
        self.h_var = None

        # G3b probationary-commit state: consecutive-miss-batch counter of the expert's
        # surprise-floor EXIT test (a batch with zero recognized free live samples
        # increments it; a batch it trains on resets it). A plain int: zero cost, no
        # rng, no float ops; read only while the probation lever is ON.
        self.prob_since_hit = 0

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
                 train_granularity="batch", session_window=0,
                 replay_passes=0, replay_items=256, probation=False,
                 m_max=0):
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

        # ---- G3a generative replay-before-freeze (DGR-adapted; PR-2026-09-03-07) -- #
        # The granularity probe's measured blocker (results/exploratory/
        # granularity_probe_2026-09-05/): commit-on-recruit grants a specialist ONE
        # unrepeated pass over its fragment, where the block stream grants ~15 epochs --
        # ACQUISITION VOLUME binds. G3a attacks it WITHOUT a replay buffer (the repo's
        # no-buffer claim is load-bearing): at the moment of a PGM freeze, the expert
        # generates replay_items pseudo-inputs x~ = h~ @ Wdec^T + bdec from its OWN
        # decoder, with h~ sampled from a diagonal Gaussian fit (running mean/var EMA)
        # of the hidden activity it saw during its life; it then self-labels x~ with its
        # OWN head (softmax, computed once from the pre-replay weights -- a frozen
        # distillation target) and pseudo-trains for replay_passes passes with the
        # expert's OWN local PC/FA/delta rule. Only THEN does the freeze commit.
        #
        # BORROWED: Deep Generative Replay (Shin et al. 2017) -- consolidation by
        # training on items generated by the system's own generative model, buffer-free.
        # ADAPTED, NOT COPIED: the generator here is the expert's own PC auto-encoder
        # decoder driven by a diagonal-Gaussian latent fit (no GAN, no separate
        # generator network, no generator/solver co-training).
        # NEW: the hidden-statistic EMA inside the local-learning expert; replay
        # embedded in the surprise-driven PGM freeze event; bookkeeping-free
        # pseudo-training (n_seen, precision floors, h-statistics and route_log measure
        # REAL data only, so every ledger stays an honest ledger).
        # HONEST INFORMATION-LIMIT CAVEAT (must travel with any claim): the pseudo-items
        # are approximations sampled from a diagonal (uncorrelated) Gaussian, so this
        # mechanism CANNOT add information beyond what the expert's single real pass
        # contained -- it can only CONSOLIDATE/STRENGTHEN the mapping that pass built,
        # and the probe measures whether that is enough. No raw data is stored.
        #
        # DEFAULT-OFF and bit-identical: at replay_passes=0 the freeze path, the
        # training path and the RNG streams are exactly the shipped ones (guarded
        # branches; no h-statistic work, no rng consumption, no float ops).
        if replay_passes < 0 or int(replay_passes) != replay_passes:
            raise ValueError(f"replay_passes must be a non-negative int, "
                             f"got {replay_passes!r}")
        if replay_items < 1 or int(replay_items) != replay_items:
            raise ValueError(f"replay_items must be a positive int, "
                             f"got {replay_items!r}")
        self.replay_passes = int(replay_passes)       # 0 = shipped: freeze with no replay
        self.replay_items = int(replay_items)         # pseudo-inputs per freeze event
        self.replay_events = []                       # per-freeze diagnostics (G3 only)

        # ---- G3b probationary commit (REAL-data revisit; PR-2026-09-03-07) ----- #
        # The replay probe's measured verdict (results/exploratory/
        # replay_probe_2026-09-05/): generative replay is consolidation WITHOUT new
        # information (pseudo-set recon drops, self-labeled head confidence flat,
        # interleaved ACC +-0.001) and its volume is router-bound (~1e3 generated
        # items vs ~4.5e5 real sample-presentations). The named remaining mechanism
        # class is REAL-data revisit. G3b: a recruited expert stays PROBATIONARY --
        # the recruit event only DEMOTES the previous active instead of committing
        # it -- and it keeps training on every subsequent LIVE sample it recognizes
        # (per-sample routed, under train_granularity="sample") until the stream's
        # surprise floor says its domain passed; only then does it commit/freeze.
        # This converts "one unrepeated pass at recruit" into "continuous live
        # re-processing until domain exit": over an N-epoch interleaved stream the
        # specialist accumulates up to N passes over its OWN fragment -- the block
        # stream's acquisition volume -- from live traffic alone.
        #
        # BUFFER-FREE / SINGLE-PASS-COMPATIBLE (the repo's no-buffer claim is
        # load-bearing): probation stores NOTHING and trains only on live samples at
        # the moment they arrive -- no replay buffer, no pseudo-items, no lookback
        # window. Shorten the stream to a single pass and the mechanism is still
        # exactly defined: it simply has less live volume to re-process (an honest
        # degradation of exposure, never a change of algorithm).
        #
        # SURPRISE-FLOOR EXIT TEST (label-free, causal): on each batch, a
        # probationary NON-ACTIVE expert that recognizes NO free live sample (all of
        # its per-sample surprises above its calibrated threshold mu + z*sigma)
        # increments prob_since_hit; PROBATION_PATIENCE consecutive miss-batches
        # declare its domain passed (hysteresis against batch-level noise, the same
        # rationale as dynamic_vigilance's smoothed rate). A batch it trains on
        # resets the counter. The active expert is exempt: shipped commits it only
        # via the recruit event, and probation defers even that.
        #
        # INTERACTION RULING vs freeze_min_seen: probation IMPLIES the stronger
        # veto. The freeze requires BOTH gates: n_seen >= freeze_min_seen AND the
        # surprise-floor exit test. Which gate wins: NEITHER overrides the other --
        # the freeze fires at the LATER of the two, i.e. the binding gate is
        # whichever is satisfied last. freeze_min_seen=0 (the probe default): the
        # exit test alone decides (freeze exactly at PATIENCE miss-batches). A large
        # floor: the expert stays PROBATIONARY through elapsed patience (no commit
        # either -- an immature expert is never abandoned to catch-all duty) and can
        # resume training if its domain returns; it freezes only once the floor has
        # been crossed by real training AND a fresh PATIENCE miss-streak has since
        # elapsed. The exit-test commit follows the shipped consolidate pairing:
        # consolidate=False commits without freezing.
        #
        # Sample priority on a live batch (deterministic): committed experts claim
        # (shipped, never re-trained), then every uncommitted calibrated expert
        # trains on the free samples IT recognizes in expert-index (= age) order,
        # then the remainder is the per-batch novel pool (recruitment bar unchanged:
        # fraction >= 0.5). A sample is trained by at most ONE expert.
        #
        # DEFAULT-OFF and bit-identical: probation=False adds one guarded branch on
        # the dispatch and one int attribute; every shipped path is the shipped body
        # verbatim, no rng consumption, no extra float ops. Requires
        # train_granularity="sample" (the per-sample decision unit is what makes
        # "every live sample it recognizes updates it" well-defined); anything else
        # raises ValueError.
        if not isinstance(probation, bool):
            raise ValueError(f"probation must be a bool, got {probation!r}")
        if probation and train_granularity != "sample":
            raise ValueError("probation=True requires train_granularity='sample' "
                             "(per-sample routed live re-processing); got "
                             f"train_granularity={train_granularity!r}")
        self.probation = probation                    # False = shipped: commit at recruit

        # ---- Bounded-M economy lever (Policy A recruit-by-eviction; PR-2026-09-03-05) - #
        # docs/EXPERT_ECONOMY.md §3.3 Policy A, VERBATIM: "on a recruit when the pool is
        # at M_max, evict the expert with the lowest lifetime routing share (tie -> most
        # recently recruited), re-use its slot."
        #
        # m_max=0 (default) = OFF = unbounded = the shipped behaviour, bit-identically on
        # every path (the guard never fires; no rng consumption, no float ops). With
        # m_max > 0 the lever CAPS the USABLE experts at m_max while the pool allocation
        # self.M stays unchanged -- slots >= m_max are never used. At a recruit whose
        # advance would move self.active to a slot >= m_max, the caller has already
        # committed the current expert by the usual route (or, under probation, demoted
        # it), then: the COMMITTED expert among slots [0, m_max) with the LOWEST lifetime
        # routing share is evicted, its slot is reset to a FRESH Expert (exactly the
        # constructor args __init__ used for that slot, incl. the seed formula
        # seed + 100*(m+1)), its route_log entry is ZEROED, and the new recruit trains
        # into the reused slot with self.active pointing there (it never reaches m_max).
        #
        # Routing share (documented semantics): share_m = route_log[m] /
        # max(route_log.sum(), 1). The denominator is common to all candidates, so the
        # argmin equals the argmin of raw counts; the normalised form is kept because the
        # spec names a SHARE (raw counts are only proportional to shares when all
        # committed experts saw equal sample totals, which block streams satisfy but
        # interleaved ones do not). Tie-break -> HIGHEST slot index: slots fill
        # left-to-right, so index is the recruitment-recency proxy (the first m_max
        # recruits occupy 0..m_max-1 in order; a re-used slot always holds the newest
        # recruit of the run). The mission's frozen operationalisation is this index
        # proxy; true per-recruit recency ordering is NOT tracked.
        #
        # Honest risk (travels with any ON run): eviction reintroduces forgetting by
        # construction -- the evicted weights are the only record of their domain and
        # there is no replay to recover them. m_max == M is a live cap (it replaces the
        # pool-exhaustion refusal at active == M-1 with an eviction); OFF-identity is
        # therefore stated for m_max = 0 and any m_max > M (both tested). The shipped
        # hard-cap drop path (`if self.active >= self.M: return`) is untouched and still
        # fires whenever the lever is OFF -- that silent-drop behaviour is the documented
        # Policy-B worst case and is pinned by tests/test_mmax_lever.py.
        if m_max < 0 or int(m_max) != m_max:
            raise ValueError(f"m_max must be a non-negative int, got {m_max!r}")
        self.m_max = int(m_max)                       # 0 = shipped: unbounded pool use
        self.eviction_log = []                        # Policy A diagnostics; entries are
                                                      # appended ONLY when the lever fires
        # expert-constructor args for slot resets (exactly what __init__ passes below)
        self._base_seed = seed
        self._expert_kwargs = dict(
            feedback=feedback, lambda_cls=lambda_cls,
            n_settle_steps=n_settle_steps, eta_settle=eta_settle,
            langevin_temp=langevin_temp, weight_bits=weight_bits,
            act_bits=act_bits, noise_in_std=noise_in_std,
            noise_act_std=noise_act_std, noise_weight_std=noise_weight_std)

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
    def _train_expert(self, e, X, Y, _bookkeeping=True):
        """The shipped local learning step (decoder/head delta rule + FA encoder step).

        _bookkeeping=True (every real-data call): also maintains the expert's real-data
        ledgers -- init_recon, n_seen, the precision floor (mu/var) and, when the replay
        lever is ON, the hidden-activity statistics. G3a pseudo-training passes
        _bookkeeping=False so replay moves WEIGHTS ONLY: n_seen, the calibrated floors
        and the h-statistics keep measuring REAL data (the ledgers stay honest, and
        generated approximations can never re-calibrate the router's thresholds)."""
        if e.frozen:
            return
        n = len(X)
        if _bookkeeping and e.init_recon is None:
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

        # G3a hidden-activity statistics: diagonal-Gaussian EMA over the latents the
        # expert actually computed on REAL data, pre-update (what it "saw"). Zero cost
        # when the replay lever is OFF (guarded; shipped path untouched).
        if self.replay_passes > 0 and _bookkeeping:
            self._update_h_stats(e, Z)

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
        if _bookkeeping:
            e.n_seen += n

        # update this expert's PRECISION over its own (post-update) reconstruction surprise
        if _bookkeeping:
            r = float(e.recon_error(X).mean())
            if e.mu > 1e8:
                e.mu, e.var = r, max(1e-4, (0.1 * r) ** 2)
            else:
                d_val = r - e.mu
                e.mu += 0.05 * d_val
                e.var = 0.95 * e.var + 0.05 * d_val * d_val

    def _update_h_stats(self, e, Z):
        """G3a: EMA of a diagonal Gaussian over the expert's hidden activity (real data).

        Per-dimension mean and variance of the batch's latents Z, folded into a running
        estimate at rate REPLAY_H_EMA (first call seeds it). Serves ONLY pseudo-input
        generation at freeze time; it is never read by routing/recognition."""
        m = Z.mean(axis=0)
        v = np.maximum(Z.var(axis=0), REPLAY_VAR_FLOOR)
        if e.h_mean is None:
            e.h_mean = m.astype(np.float32)
            e.h_var = v.astype(np.float32)
        else:
            e.h_mean = ((1.0 - REPLAY_H_EMA) * e.h_mean
                        + REPLAY_H_EMA * m).astype(np.float32)
            e.h_var = ((1.0 - REPLAY_H_EMA) * e.h_var
                       + REPLAY_H_EMA * v).astype(np.float32)

    def _replay_before_freeze(self, e):
        """G3a generative replay BEFORE the PGM freeze commits (default-off).

        Runs INSIDE the freeze branch: an expert that is about to be frozen first
        generates replay_items pseudo-inputs from its OWN decoder (h~ ~ diagonal
        Gaussian of its lifetime hidden activity; x~ = h~ @ Wdec^T + bdec), self-labels
        them ONCE with its own head (frozen distillation target -- a per-pass moving
        target would let the head chase its own updates and anneal its own error to
        zero), then pseudo-trains replay_passes passes with its OWN local rule
        (_train_expert with _bookkeeping=False). Weights move; every real-data ledger
        (n_seen, mu/var floors, h-statistics, route_log) does not.

        Buffer-free by construction (generates from weights, stores nothing). Borrowed:
        Deep Generative Replay (Shin et al. 2017), adapted to the local PC/FA expert
        (no GAN, no co-training). Honest limit: generated items are approximations --
        this consolidates what the single real pass built; it cannot add information.

        No-op when the lever is OFF (replay_passes=0) or the expert has no hidden
        statistics (never trained on real data): zero rng, zero float ops."""
        if self.replay_passes <= 0 or e.h_mean is None:
            return
        r = e.rng
        h_std = np.sqrt(e.h_var)
        Hs = (e.h_mean[None, :]
              + r.normal(0.0, 1.0, (self.replay_items, self.h))
              * h_std[None, :]).astype(np.float32)                    # h~ ~ N(mu_h, diag(var_h))
        Wdec_eff = e._get_weight(e.Wdec)
        bdec_eff = e._get_weight(e.bdec)
        Xt = (Hs @ Wdec_eff.T + bdec_eff).astype(np.float32)          # x~ = h~.Wdec + bdec
        # self-label once with the PRE-replay head; recon/confidence read from the same
        # forward pass (EPS is the pseudo-set's reconstruction error pre-consolidation).
        Zt, EPS_pre, logits = e.forward(Xt)
        P = softmax(logits, axis=1)
        conf_pre = float(P.max(axis=1).mean())
        recon_pre = float((EPS_pre ** 2).mean())
        for _ in range(self.replay_passes):
            self._train_expert(e, Xt, P, _bookkeeping=False)
        # post-freeze surrogate values on the SAME pseudo-set (consolidation read-out)
        _, EPS_post, logits_post = e.forward(Xt)
        P_post = softmax(logits_post, axis=1)
        self.replay_events.append({
            "expert": self.experts.index(e),
            "items": int(self.replay_items),
            "passes": int(self.replay_passes),
            "n_seen_real": int(e.n_seen),
            "recon_pre": recon_pre,
            "recon_post": float((EPS_post ** 2).mean()),
            "conf_pre": conf_pre,
            "conf_post": float(P_post.max(axis=1).mean()),
        })

    # ---------------- bounded-M economy (Policy A; PR-2026-09-03-05) ----------- #
    def _advance_or_evict(self):
        """Advance self.active past a recruit, evicting under the bounded-M cap.

        Called at every recruit site AFTER the current expert has been committed (or,
        under probation, demoted) by that site's own shipped machinery. Shipped advance
        (m_max == 0 or m_max > M): `self.active += 1` verbatim -- bit-identical.
        Lever ON and the advance would land on a slot >= m_max (0 < m_max <= M and
        self.active + 1 >= m_max): Policy A fires -- evict the committed expert in
        [0, m_max) with the lowest lifetime routing share (tie -> highest index),
        reset its slot to a fresh Expert, zero its route_log entry, and point
        self.active at the reused slot (it never reaches m_max, so slots >= m_max stay
        unused for the whole run). If no committed expert exists inside the cap (only
        possible when the recruit DEMOTES instead of committing, i.e. probation), the
        recruit is refused: self.active stays put, mirroring the shipped
        pool-exhausted refusal."""
        if 0 < self.m_max <= self.M and self.active + 1 >= self.m_max:
            victim = self._evict_victim()
            if victim is not None:
                self._evict_slot(victim)
                self.active = victim
            return
        self.active += 1

    def _evict_victim(self):
        """Policy A victim: the committed expert in [0, min(m_max, M)) with the lowest
        lifetime routing share share_m = route_log[m] / max(route_log.sum(), 1)
        (denominator common to all candidates -> same argmin as raw counts); ties break
        to the HIGHEST slot index (most recently recruited under the left-to-right fill
        proxy). Returns None when no committed expert is evictable."""
        cap = min(self.m_max, self.M)
        cands = [m for m in range(cap) if self.experts[m].committed]
        if not cands:
            return None
        tot = max(float(self.route_log.sum()), 1.0)
        return min(cands, key=lambda m: (float(self.route_log[m]) / tot, -m))

    def _evict_slot(self, victim):
        """Policy A eviction: reset slot `victim` to a FRESH Expert built with exactly
        the constructor args __init__ used for that slot (same seed formula
        seed + 100*(slot+1), so the fresh expert is identical to the slot's original
        initialisation), zero its route_log entry (the ledger keeps counting only live
        experts), and record the fire in eviction_log."""
        rec = {
            "victim": int(victim),
            "victim_route_count": int(self.route_log[victim]),
            "victim_share": float(self.route_log[victim])
                            / max(float(self.route_log.sum()), 1.0),
            "route_log_at_fire": self.route_log.tolist(),
            "active_before_reuse": int(self.active),
        }
        self.experts[victim] = Expert(
            self.d, self.h, self.K, self._base_seed + 100 * (victim + 1),
            **self._expert_kwargs)
        self.route_log[victim] = 0
        self.eviction_log.append(rec)

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
            if self.probation:
                self._train_batch_by_sample_probation(X, Y)   # G3b: probationary commits
            else:
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
            # G3a: replay runs INSIDE the freeze branch, before the freeze commits --
            # replay-before-freeze. No freeze (consolidate off / veto fires) -> no
            # replay: it is defined as part of the freeze procedure.
            if self.consolidate and act.n_seen >= self.freeze_min_seen:
                self._replay_before_freeze(act)
                act.frozen = True
                act.omega = self.omega_consol + 1.0
            self._advance_or_evict()                 # bounded-M Policy A fires here
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
                    self._replay_before_freeze(act)   # G3a: replay-before-freeze
                    act.frozen = True
                    act.omega = self.omega_consol + 1.0
                self._advance_or_evict()         # bounded-M Policy A fires here
                if self.active < self.M:
                    act = self.experts[self.active]
            if act.n_seen == 0 and self.active < self.M:
                self._train_expert(act, X[free], Y[free])
                self.route_log[self.active] += int(free.sum())
            self._note_novelty(True)
        else:
            self._note_novelty(False)

    def _train_batch_by_sample_probation(self, X, Y):
        """G3b (probation=True): per-sample routed training with PROBATIONARY commits.

        Same skeleton as _train_batch_by_sample (committed claims -> per-sample
        routing -> novel-pool recruitment), with two mechanism changes:

          1. RECRUIT DEMOTES, NOT COMMITS: when a batch's novel fraction >= 0.5 fires
             recruitment, the previous active is NOT committed/frozen -- it becomes a
             probationary expert and keeps training on every later live sample it
             recognizes. Over an N-epoch stream a specialist therefore accumulates up
             to N passes over its own fragment (REAL-data revisit: live samples only,
             nothing stored, single-pass compatible).
          2. SURPRISE-FLOOR EXIT TEST: after the batch, every probationary NON-ACTIVE
             expert that found no free sample to train on increments
             prob_since_hit; at PROBATION_PATIENCE consecutive miss-batches its
             domain is declared passed and it commits/freezes -- gated by BOTH
             patience AND n_seen >= freeze_min_seen (the stronger veto: the LATER
             gate binds; see the constructor docstring for the full ruling).
             consolidate=False commits without freezing (shipped pairing).

        Priority on a live sample: committed claim > older probationary (expert-index
        order) > active > novel pool. The active expert is exempt from the exit test
        (it is the current recruit-in-training). The pool-exhausted drop path
        (active >= M) returns before the exit test, so dropped batches never advance
        patience. route_log counts trained/claimed samples per expert: the ledger
        stays an honest TRAINING ledger."""
        n = len(X)
        calib = [m for m in range(self.M) if not (self.experts[m].mu > 1e8)]
        free = np.ones(n, dtype=bool)          # not yet claimed/trained this batch
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
        # probationary training: every uncommitted calibrated expert trains on the
        # free samples IT recognizes, oldest first (index order == age order under
        # this machinery; the active is simply the last such expert)
        trained = np.zeros(self.M, dtype=np.int64)
        for m in calib:
            e = self.experts[m]
            if e.committed:
                continue
            take = free & (S[:, m] <= self._threshold(e))
            free &= ~take
            if take.any():
                self._train_expert(e, X[take], Y[take])
                self.route_log[m] += int(take.sum())
                trained[m] = int(take.sum())
        # the remaining samples are the per-batch novel pool
        recruit = bool(free.any()) and float(free.mean()) >= SAMPLE_NOVEL_FRACTION
        if recruit:
            act = self.experts[self.active]
            if act.n_seen > 0:
                # PROBATION: demote, never commit here -- the old active keeps
                # training on later live samples until its exit test fires.
                act.prob_since_hit = 0
                self._advance_or_evict()         # bounded-M Policy A fires here
                                                 # (evictable candidates are COMMITTED
                                                 # experts only; the just-demoted
                                                 # active is not one, and if no
                                                 # committed expert exists inside the
                                                 # cap the recruit is refused)
            if self.active < self.M:
                nact = self.experts[self.active]
                if nact.n_seen == 0:
                    self._train_expert(nact, X[free], Y[free])
                    self.route_log[self.active] += int(free.sum())
            self._note_novelty(True)
        else:
            self._note_novelty(False)
        # surprise-floor exit test for every probationary NON-ACTIVE expert
        for m in range(self.M):
            if m == self.active:
                continue
            e = self.experts[m]
            if e.committed or e.mu > 1e8:
                continue
            if trained[m] > 0:
                e.prob_since_hit = 0
                continue
            e.prob_since_hit += 1
            # BOTH gates required (the LATER one binds); shipped consolidate pairing
            if e.prob_since_hit >= PROBATION_PATIENCE and e.n_seen >= self.freeze_min_seen:
                e.committed = True
                if self.consolidate:
                    self._replay_before_freeze(e)   # G3a composes inside the freeze
                    e.frozen = True
                    e.omega = self.omega_consol + 1.0
                e.prob_since_hit = 0

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
