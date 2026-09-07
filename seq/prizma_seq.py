"""
Prizma-Seq — a Predictive-Coding Gated-DeltaNet sequence model (committee spec PRIZMA_SEQ_SPEC.md).

Mixer = a carried associative workspace state S_t in R^{d_h x d_h} per head, updated by a
precision-gated targeted erase-and-write (the delta rule), which is exactly ONE gradient step on
Prizma's per-token free energy F_t(S)=1/2||v_t - S k_t||^2.  Read = S_{t-1} q_t (recognition-by-
reconstruction, strictly pre-write/causal) + a small exact local window head. FFN is byte-identical
to the Transformer baseline so ONLY the mixer differs.

Honest design note: the write gate beta is INPUT-dependent (sigma(W_beta x_t)), which keeps the
chunk-parallel training form valid. Surprise-proportionality is intrinsic: the delta write is
u_t = beta_t * (v_t - S_{t-1} k_t) = beta_t * epsilon_t — it writes the prediction error itself
(Prizma's dW ~ (Pi*eps) (x) r). An optional surprise-gated variant (two-pass) is exposed for B6.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import RMSNorm, SwiGLU, TFConfig
from .delta import chunked_delta, _surprise_norm_gate, _surprise_norm_update


@dataclass
class PrizmaSeqConfig:
    vocab: int = 64
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 2
    chunk: int = 64
    window: int = 16
    d_ff: int = None
    max_len: int = 1024
    rope: bool = False           # delta keys/queries are POSITION-FREE (DeltaNet/Mamba standard):
                                 #   RoPE makes the recall dot-product distance-dependent and
                                 #   scrambles content-based recall. Position comes from the conv +
                                 #   causal write-order. (RoPE on state keys empirically blocks MQAR.)
    beta_cap: float = 0.99
    gated: bool = False          # data-dependent forget gate alpha (Gated-DeltaNet); off for diagnostics
    learned_pos: bool = False    # add a learned absolute pos-embedding (used for char-LM parity)
    short_conv: int = 4          # short causal depthwise conv before qkv (Mamba/Based/DeltaNet std);
                                 #   lets a token's k/v encode its predecessor -> enables recall. 0=off.
    # --- ablation knobs (B6) ---
    precision_gate: str = "input"   # 'input' (sigma(W_beta x)) | 'uniform' | 'random' (B6 controls)
                                    #   | 'surprise_norm' (PR-2026-09-03-01 arm A4, frozen formula:
                                    #   beta_t = beta_cap*sigmoid(a*(s_t/max(m_t,1e-6) - 1)) computed
                                    #   INSIDE the recurrence from the running state; replaces the
                                    #   learned gate entirely; gain `a` per the pre-reg NO-TUNING
                                    #   rule — selected once on exploratory seed 900, then frozen).
                                    #   TRAINING-PATH NOTE (disclosed, not silent): under
                                    #   'surprise_norm' forward() routes to the EXACT sequential scan
                                    #   (_delta_reference) — the WY/UT chunk-parallel form is invalid
                                    #   because beta_t depends on the running state (R3); training via
                                    #   build_and_train/forward() therefore carries the disclosed 2-5x
                                    #   tax (pre-reg §5). Streaming step() mirrors it exactly (G1
                                    #   guard, state-carried EMA).
    write_mode: str = "delta"       # 'delta' (targeted erase-and-write) | 'additive' (linear-attn)
    use_workspace: bool = True      # False -> no carried state (window head only)
    use_window: bool = True         # False -> no local window head (state only)
    route_readout: bool = True      # False -> read state with a FIXED (input-independent) query
                                    #          (B6 noRouteReadout: kills content-based recall)
    # --- capacity lever: parameter-free quadratic key/query feature map (committee R1, rank #1) - #
    feat_map: str = "none"          # 'none' | 'quad2' | 'quad2_lowrank' | 'rand_linear'.
                                    #   'quad2': expands delta keys/queries from d_h to
                                    #     d_phi = d_h + feat_n2 via FIXED random quadratic monomials
                                    #     -> rectangular state S in R^{d_h x d_phi}, raising
                                    #     associative-recall key-rank toward D=128 at ZERO trainable
                                    #     params (buffers, not Parameters). O(1) inference intact.
                                    #   'quad2_lowrank': leaner variant — projects x (d_h) -> r dims
                                    #     via a FIXED seeded matrix P in R^{d_h x r}, then takes ALL
                                    #     r*(r+1)/2 quadratic monomials in the r-dim projected space,
                                    #     giving d_phi = d_h + r*(r+1)/2. Default r=feat_rank (or 14
                                    #     when feat_rank=0), which yields d_phi=137 (~54% of quad2's
                                    #     256 at d_h=32) while preserving recall capacity (crosstalk
                                    #     gap from quad2 < 0.01). Still ZERO trainable params —
                                    #     P and monomial indices are buffers seeded at 1234.
    feat_n2: int = 96               # number of quadratic monomials (d_phi = d_h + feat_n2); used by
                                    #   'quad2' and 'rand_linear' only (ignored by 'quad2_lowrank').
    feat_rank: int = 0              # low-rank projection dim r for 'quad2_lowrank'. When 0, defaults
                                    #   to 14 -> d_phi = d_h + 14*15//2 = d_h + 105 (≈137 for d_h=32).
                                    #   Effective d_phi = d_h + r*(r+1)//2 (documented in __post_init__).
    out_gate: bool = False      # per-token output gate g=sigma(W_g x); o = o * g before W_o (RWKV-7/GLA)
    state_norm: bool = False    # per-head RMSNorm on the delta-state read o_delta before merge
    banded_window: bool = False # O(T*w) banded sliding-window kernel (exact-equal to _window, default off)
    decoupled_gate: bool = False  # GDN-2: decouple erase gate beta_e (key-side) from write gate beta_w
    n_delta: int = 1              # DeltaProduct: number of sequential delta sub-steps per token (k=1 = today)
    # --- surprise-gated write (Lever A) ---
    surprise_gate: bool = False   # if True, scale write by g_t = 1+tanh(||eps_t||) (default OFF = identical)
    surprise_mode: str = 'norm'   # 'norm' | 'random' | 'constant' — controls for R9 ablation
    surprise_seed: int = 1234     # deterministic seed for the surprise_mode='random' generator (R8/R9):
                                  #   the block re-seeds an OWNED torch.Generator to this value each
                                  #   forward, so the random control gates are reproducible across calls.
                                  #   Ignored by 'norm'/'constant' (which need no generator).
    # --- in-context per-channel learning rate (Lever G, RWKV-7 "Goose" generalized delta) ---
    inctx_lr: bool = False        # if True, replace scalar write gate beta_t with a per-VALUE-channel
                                  #   rate eta_t = beta_cap * sigmoid(W_eta x_t) in R^{d_h}, modulating
                                  #   the delta write per state channel. Default OFF = byte-identical.
    # --- residual + embedding dropout (regularizer-gap lever) ---
    dropout: float = 0.0          # opt-in residual+embedding dropout. Default 0.0 == BYTE-IDENTICAL:
                                  #   nn.Dropout(0.0) is an identity that draws NO rng, so the default
                                  #   forward (and any downstream rng) is unchanged. Closes the
                                  #   architectural regularizer gap vs TFConfig (transformer.py has
                                  #   attention dropout; PrizmaSeqConfig had NONE), so gpu_charlm2.py
                                  #   ran dropout-free to avoid regularizing only the TF (unfair) and
                                  #   leaned on weight_decay alone. Enabling this UNBLOCKS a fair
                                  #   symmetric-dropout char-LM experiment. NOT a BPC claim — capability.
    # --- analog-robustness levers (report 12-H2, "the delta rule is analog-robust") --- #
    # SCOPE LIMITATION (honest, load-bearing): these two levers degrade ONLY the exact O(1)
    # streaming path `step()` — the deployment path where the carried state S is materialized.
    # The chunk-parallel training kernel (`seq/delta.py::chunked_delta`) never materializes
    # per-token states (the WY/UT form recomputes intra-chunk states from chunk aggregates),
    # so a shared per-write quantization would require re-deriving that kernel. The knobs are
    # therefore WIRED TO step() ONLY: train in FP32 (forward() untouched, byte-identical),
    # deploy degraded (step()). This is exactly the inference-time-emulation protocol of
    # commission report 12-H2 ("quantize S to b bits after every write ... reads see the
    # quantized S — write-and-read both degraded"), and it PRESERVES the O(1) step()==forward()
    # guard at defaults. With knobs ON, step() intentionally deviates from forward() (the
    # FP32 reference) — that deviation IS the measured deployment degradation, not a bug.
    state_bits: int = 0           # quantize the carried state S after each write, per head, on the
                                  #   per-head max-abs range (uniform symmetric quantizer, round-
                                  #   to-nearest, straight-through estimator — Bengio et al. 2013).
                                  #   0 = OFF (bit-identical; no extra ops, no rng). In {4,6,8} = the
                                  #   deployment emulation bits (report 12-H2 grid).
    write_noise_std: float = 0.0  # add Gaussian noise ~ N(0, write_noise_std^2) to the write update
                                  #   u_t BEFORE it is added to the state (noisy analog write).
                                  #   0.0 = OFF (bit-identical; draws NO rng).
    write_noise_seed: int = 1234  # reproducibility plumbing (mirrors surprise_seed, R8/R9): each
                                  #   step() owns a fresh torch.Generator seeded with
                                  #   write_noise_seed + pos (pos = token index from the streaming
                                  #   state), so noise is i.i.d. per write AND reproducible for a
                                  #   fixed evaluation order — no hidden cross-call generator state.
    # --- surprise-NORM write gate (PR-2026-09-03-01 arm A4 — REGISTERED-FROZEN pre-registration) ---
    # Inert unless precision_gate == "surprise_norm" (any other gate value never reads them, so the
    # defaults leave every existing mode byte-identical). The pre-registration pins: the gate
    # REPLACES the learned gate (not a multiplier); the signal is per-head; eps_t uses the CURRENT
    # pre-write state; the EMA is causal, reset per sequence in forward() and state-carried in
    # step(); the sigmoid argument is a*(s_t/max(m_t,1e-6) - 1) exactly (not a*log s_tilde, not
    # a*s_tilde - 1); beta_cap multiplies AFTER sigmoid; the normalization guard is max(m_t, 1e-6).
    surprise_gain: float = 1.0    # the gain `a` in beta_t = beta_cap*sigmoid(a*(s_tilde_t - 1)).
                                  #   Supplied per-run from the frozen protocol value: selected ONCE
                                  #   on exploratory seed 900 from {0.5, 1.0, 2.0, 4.0} (highest
                                  #   best_acc on the frozen eval; ties -> smallest a), then frozen
                                  #   for all claim seeds/tasks (NO-TUNING rule, pre-reg §2 A4).
                                  #   Default 1.0 is a protocol-neutral placeholder — never tuned.
    surprise_ema_lambda: float = 0.02  # the EMA rate lam in m_t = (1-lam)*m_{t-1} + lam*s_t.
                                  #   FROZEN A PRIORI at 0.02 (~50-token effective horizon,
                                  #   context-scale for T in {256, 384}); never tuned (pre-reg §2).

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = int(round(8 / 3 * self.d_model / 8) * 8)
        assert self.d_model % self.n_heads == 0
        self.d_h = self.d_model // self.n_heads
        assert self.d_h % 2 == 0, "d_h must be even for RoPE"
        assert self.feat_map in ("none", "quad2", "quad2_lowrank", "rand_linear")
        assert self.precision_gate in ("input", "uniform", "random", "surprise_norm"), \
            f"precision_gate must be 'input', 'uniform', 'random', or 'surprise_norm', " \
            f"got {self.precision_gate!r}"
        # PR-2026-09-03-01 arm A4: the one-novel-lever rule — surprise_norm REPLACES the learned
        # gate and is scoped to the frozen protocol config (pre-reg §2 implementation spec item 2).
        if self.precision_gate == "surprise_norm":
            assert not self.surprise_gate and not self.inctx_lr \
                and not self.decoupled_gate and self.n_delta == 1, \
                ("precision_gate='surprise_norm' (PR-2026-09-03-01 A4) replaces the learned gate "
                 "entirely and is scoped to n_delta==1 with surprise_gate/inctx_lr/decoupled_gate "
                 "all False; got a conflicting lever combination.")
        assert self.surprise_mode in ('norm', 'random', 'constant'), \
            f"surprise_mode must be 'norm', 'random', or 'constant', got {self.surprise_mode!r}"
        # Lever G is scoped to n_delta==1 (per-channel eta is not combined with DeltaProduct).
        assert not (self.inctx_lr and self.n_delta >= 2), \
            "inctx_lr (per-channel eta) is only implemented for n_delta==1, not n_delta>=2."
        # inctx_lr (Lever G) and surprise_gate (Lever A) are the TWO novel-core CANDIDATES for the S3
        # ablation; enabling both is a SILENT FOOTGUN — the delta kernel branch is
        # `if eta is not None: ... elif surprise:`, so eta wins and surprise is silently ignored. Reject
        # the invalid combo so each S3 arm enables exactly one novel-core lever.
        assert not (self.inctx_lr and self.surprise_gate), \
            "inctx_lr and surprise_gate are mutually exclusive novel-core candidates; enable exactly one."
        # d_phi = delta key/query dim after the optional feature map (= d_h when 'none').
        # Analog levers (report 12-H2): validate the degradation knobs (both default OFF).
        assert self.state_bits == 0 or self.state_bits >= 2, \
            f"state_bits must be 0 (off) or >= 2, got {self.state_bits}"
        assert self.write_noise_std >= 0.0, \
            f"write_noise_std must be >= 0.0 (0.0 = off), got {self.write_noise_std}"
        # 'rand_linear' = a FIXED random linear map d_h->d_phi (a CONTROL: it stays in a d_h-rank
        # subspace so it must give NO capacity gain, proving the quad2 MONOMIALS are what help).
        # 'quad2_lowrank': effective r = feat_rank if feat_rank > 0 else 14 (default);
        #   d_phi = d_h + r*(r+1)//2  (all upper-triangular monomial pairs in the projected space).
        if self.feat_map == "quad2_lowrank":
            _r = self.feat_rank if self.feat_rank > 0 else 14
            self._feat_rank_eff = _r
            self.d_phi = self.d_h + _r * (_r + 1) // 2
        elif self.feat_map in ("quad2", "rand_linear"):
            self._feat_rank_eff = 0
            self.d_phi = self.d_h + self.feat_n2
        else:
            self._feat_rank_eff = 0
            self.d_phi = self.d_h


# ----------------------------------- RoPE ------------------------------------------------- #
def _rope_cache(T, d_h, device, dtype, offset=0):
    inv = 1.0 / (10000 ** (torch.arange(0, d_h, 2, device=device, dtype=torch.float32) / d_h))
    pos = torch.arange(offset, offset + T, device=device, dtype=torch.float32)
    ang = torch.outer(pos, inv)                       # [T, d_h/2]
    return torch.cos(ang).to(dtype), torch.sin(ang).to(dtype)


def _apply_rope(x, cos, sin):
    # x: [B,H,T,d_h]; cos/sin: [T,d_h/2]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = rx1
    out[..., 1::2] = rx2
    return out


def _l2(x, eps=1e-6):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _quantize_state_symmetric(x, bits):
    """Uniform SYMMETRIC per-head max-abs quantizer with a straight-through estimator — the
    state-S deployment emulation for the analog-robustness probe (report 12-H2).

    Forward: each head's matrix (the trailing two dims of x[..., d_v, d_k]) is mapped to the
    integer grid {-qmax, ..., 0, ..., +qmax} * s with qmax = 2^(bits-1) - 1 and s = max|x| / qmax
    (round-to-nearest, clamped). The grid is symmetric about 0 and round-to-nearest is monotone
    non-decreasing, so the quantizer is symmetric (q(-x) == -q(x)) and monotone (x1 <= x2 =>
    q(x1) <= q(x2)) — both are unit-tested (tests/test_analog_lever.py).
    Backward: straight-through estimator (identity gradient), the standard quasi-gradient for
    quantization (Bengio, Léonard & Courville 2013, "Estimating or Propagating Gradients Through
    Stochastic Discrete Variables"). step() runs under no_grad so the STE is inert there today,
    but the helper is written STE-correct so future QAT (report 12-H2's optional follow-up) can
    reuse it without a silent wrong-gradient trap.

    Returns a tensor with the quantized VALUES of x and (via the STE residual) gradients of x.
    """
    qmax = 2 ** (bits - 1) - 1
    amax = x.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)   # per-head max-abs range
    s = amax / qmax
    xq = torch.clamp(torch.round(x / s), -qmax, qmax) * s
    return x + (xq - x).detach()      # forward value == xq; backward gradient == identity (STE)


# --------------------------------- the block ---------------------------------------------- #
class PrizmaSeqBlock(nn.Module):
    def __init__(self, cfg: PrizmaSeqConfig):
        super().__init__()
        self.cfg = cfg
        d, H, dh = cfg.d_model, cfg.n_heads, cfg.d_h
        self.H, self.dh = H, dh
        self.norm1 = RMSNorm(d)
        self.kc = cfg.short_conv
        if self.kc > 0:
            self.conv = nn.Conv1d(d, d, self.kc, groups=d, bias=True)   # depthwise causal short conv
        self.W_qkv = nn.Linear(d, 3 * d, bias=False)
        self.W_beta = nn.Linear(d, H, bias=True)
        self.beta_logit = nn.Parameter(torch.zeros(H))      # uniform (input-independent) gate, B6
        self.q_fixed = nn.Parameter(torch.randn(H, dh) * 0.02)   # noRouteReadout fixed query, B6
        self.W_alpha = nn.Linear(d, H, bias=True) if cfg.gated else None
        self.W_o = nn.Linear(d, d, bias=False)
        self.W_g = nn.Linear(d, d, bias=True) if cfg.out_gate else None
        self.state_rms = RMSNorm(dh) if cfg.state_norm else None
        self.W_e = nn.Linear(d, H, bias=True) if cfg.decoupled_gate else None   # erase gate beta_e
        # Lever G: per-VALUE-channel in-context learning rate eta = beta_cap * sigmoid(W_eta x).
        # d = n_heads*d_h -> one rate per head per channel; reshaped to [B,H,T,d_h] in _encode.
        self.W_eta = nn.Linear(d, d, bias=True) if cfg.inctx_lr else None
        # DeltaProduct n_delta>=2: (n_delta-1) additional kv projections + beta heads.
        # Each extra sub-step j>=1 gets its own W_kv_j and W_beta_j (independent params).
        # Total param cost: (n_delta-1) * (2*d*d + H) trainable weights per block.
        # NOTE: the plan explicitly allows repeated projections for simplicity (documented).
        self.W_kv_extra = nn.ModuleList([
            nn.Linear(d, 2 * d, bias=False) for _ in range(cfg.n_delta - 1)
        ]) if cfg.n_delta >= 2 else None
        self.W_beta_extra = nn.ModuleList([
            nn.Linear(d, H, bias=True) for _ in range(cfg.n_delta - 1)
        ]) if cfg.n_delta >= 2 else None
        self.norm2 = RMSNorm(d)
        self.mlp = SwiGLU(TFConfig(d_model=d, d_ff=cfg.d_ff))
        # opt-in residual dropout (default 0.0 == identity, draws no rng -> byte-identical).
        self.drop = nn.Dropout(cfg.dropout)
        self.win_scale = dh ** -0.5
        self.d_phi = cfg.d_phi
        if cfg.feat_map == "quad2":
            # FIXED random quadratic monomials phi(x) = [x ; x[I]*x[J]] -> d_phi. Seeded => a fixed
            # architectural choice (NOT tuned per task; disclosed). Registered as BUFFERS, so
            # param_count is unchanged (byte-identical param-match to the Transformer preserved).
            g = torch.Generator().manual_seed(1234)
            self.register_buffer("feat_I", torch.randint(0, dh, (cfg.feat_n2,), generator=g))
            self.register_buffer("feat_J", torch.randint(0, dh, (cfg.feat_n2,), generator=g))
        elif cfg.feat_map == "quad2_lowrank":
            # Leaner variant (Task 1.D): project x (d_h) -> r dims via FIXED seeded P in R^{d_h x r},
            # then take ALL r*(r+1)/2 upper-triangular monomial pairs in the r-dim space.
            # d_phi = d_h + r*(r+1)//2. At r=14 (default): d_phi = 32+105=137 for d_h=32.
            # P is a buffer (ZERO trainable params), seeded same discipline as quad2 (seed 1234).
            r = cfg._feat_rank_eff
            g = torch.Generator().manual_seed(1234)
            self.register_buffer("feat_P", torch.randn(dh, r, generator=g) * (dh ** -0.5))
            n_pairs = r * (r + 1) // 2
            I_lr = torch.tensor([i for i in range(r) for j in range(i, r)], dtype=torch.long)
            J_lr = torch.tensor([j for i in range(r) for j in range(i, r)], dtype=torch.long)
            self.register_buffer("feat_I_lr", I_lr)
            self.register_buffer("feat_J_lr", J_lr)
        elif cfg.feat_map == "rand_linear":
            # CONTROL (committee): fixed random linear map d_h -> d_phi. rank <= d_h, so it CANNOT
            # raise recall key-rank -> expected NO gain over 'none'. Buffer => param_count unchanged.
            g = torch.Generator().manual_seed(1234)
            self.register_buffer("W_rand", torch.randn(dh, self.d_phi, generator=g) * (dh ** -0.5))

    def _apply_conv(self, x):
        """Causal depthwise short conv + SiLU on the normed input (Mamba/Based-style). x:[B,T,d]."""
        if self.kc == 0:
            return x
        xc = F.pad(x.transpose(1, 2), (self.kc - 1, 0))          # left-pad -> causal
        return F.silu(self.conv(xc).transpose(1, 2))             # [B,T,d]

    def _phi(self, x):
        """Quadratic key/query feature map for the DELTA path only. x:[...,d_h] (already L2-normed)
        -> [...,d_phi]. Identity when feat_map='none'; else _l2([x ; x[I]*x[J]]) over FIXED random
        monomial indices, which escapes the d_h subspace and cuts associative-recall crosstalk
        (D=128: ~0.142 -> ~0.117 for quad2/d_phi=256, key_crosstalk metric). The final _l2 preserves
        ||k||=1, the invariant the delta kernel relies on. The local-window head keeps linear L2 keys.
        'quad2_lowrank': project x -> z=x@P (r-dim), take all r*(r+1)/2 monomials z[I]*z[J],
        concat [x; monomials] -> d_phi = d_h + r*(r+1)/2 (leaner than quad2 at ~half d_phi)."""
        if self.cfg.feat_map == "none":
            return x
        if self.cfg.feat_map == "rand_linear":
            return _l2(x @ self.W_rand)               # control: rank <= d_h -> no capacity gain
        if self.cfg.feat_map == "quad2_lowrank":
            z = x @ self.feat_P                        # [..., r]  (fixed seeded projection)
            two = z[..., self.feat_I_lr] * z[..., self.feat_J_lr]   # [..., n_pairs]
            return _l2(torch.cat([x, two], dim=-1))   # [..., d_phi = d_h + n_pairs]
        # quad2: fixed random monomials from the d_h-dim input
        two = x[..., self.feat_I] * x[..., self.feat_J]
        return _l2(torch.cat([x, two], dim=-1))

    def _encode(self, x, cos, sin):
        B, T, _ = x.shape
        qkv = self.W_qkv(x).view(B, T, self.H, 3, self.dh)
        q, k, v = qkv.unbind(3)                                   # each [B,T,H,dh]
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)   # [B,H,T,dh]
        if self.cfg.rope:
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)
        q, k = _l2(q), _l2(k)
        B, T = x.shape[0], x.shape[1]
        if not self.cfg.route_readout:    # B6: read with a FIXED query -> no content-based recall
            q = _l2(self.q_fixed[None, :, None, :].expand(B, self.H, T, self.dh).contiguous())
        if self.cfg.precision_gate == "uniform":
            beta = torch.sigmoid(self.beta_logit)[None, :, None].expand(B, self.H, T) * self.cfg.beta_cap
        elif self.cfg.precision_gate == "random":   # input-independent random write gate (B6 control)
            beta = torch.rand(B, self.H, T, device=x.device, dtype=x.dtype) * self.cfg.beta_cap
        elif self.cfg.precision_gate == "surprise_norm":
            # PR-2026-09-03-01 arm A4 (pre-reg implementation spec item 2): beta is computed INSIDE
            # the recurrence from the running state (seq/delta.py::_delta_reference erase branch /
            # step()), so _encode returns None — the learned gate is REPLACED, not modulated. The
            # asserts pin the frozen scope (one-novel-lever rule; also enforced in __post_init__).
            assert not self.cfg.surprise_gate and not self.cfg.inctx_lr \
                and not self.cfg.decoupled_gate and self.cfg.n_delta == 1, \
                ("precision_gate='surprise_norm' requires surprise_gate=False, inctx_lr=False, "
                 f"decoupled_gate=False, n_delta==1 (got surprise_gate={self.cfg.surprise_gate}, "
                 f"inctx_lr={self.cfg.inctx_lr}, decoupled_gate={self.cfg.decoupled_gate}, "
                 f"n_delta={self.cfg.n_delta}).")
            beta = None
        else:
            beta = torch.sigmoid(self.W_beta(x)).transpose(1, 2) * self.cfg.beta_cap   # [B,H,T]
        if self.W_alpha is not None:
            alpha = torch.sigmoid(self.W_alpha(x)).transpose(1, 2)                 # [B,H,T]
            alpha = 0.5 + 0.5 * alpha           # keep decay in [0.5,1] (stability)
        else:
            alpha = None
        if self.W_e is not None:
            beta_e = torch.sigmoid(self.W_e(x)).transpose(1, 2) * self.cfg.beta_cap   # [B,H,T]
        else:
            beta_e = None          # => chunked_delta will use beta_e = beta (byte-identical)
        if self.W_eta is not None:
            # Lever G: per-VALUE-channel rate eta = beta_cap * sigmoid(W_eta x). W_eta(x) is [B,T,d];
            # reshape to [B,T,H,d_h] -> [B,H,T,d_h] so eta broadcasts over the value/output channels.
            eta = torch.sigmoid(self.W_eta(x)).view(B, T, self.H, self.dh).transpose(1, 2)
            eta = eta * self.cfg.beta_cap                                              # [B,H,T,d_h]
        else:
            eta = None             # => chunked_delta uses the scalar-beta path (byte-identical)
        return q, k, v, beta, alpha, beta_e, eta

    def _window(self, q, k, v):
        """Exact causal attention restricted to the last `w` tokens (incl. self). Fused SDPA + band
        mask (fast on MPS); scaling 1/sqrt(d_h) matches the streaming step() path."""
        T = q.shape[2]
        w = self.cfg.window
        idx = torch.arange(T, device=q.device)
        band = (idx[None, :] <= idx[:, None]) & (idx[None, :] > idx[:, None] - w)  # [T,T] True=allow
        mask = torch.zeros(T, T, device=q.device, dtype=q.dtype).masked_fill(~band, float("-inf"))
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)            # [B,H,T,dh]

    def _window_banded(self, q, k, v):
        """Sliding-window causal attention in O(T*w) via fixed-size chunks of size w. Each query in
        chunk c attends keys in chunks {c-1, c} masked to [i-w+1, i]. Numerically equals _window."""
        B, H, T, dh = q.shape
        w = self.cfg.window
        outs = []
        for c0 in range(0, T, w):
            c1 = min(c0 + w, T)
            qc = q[:, :, c0:c1]                           # [B,H,Cq,dh]
            k0 = max(0, c0 - w)
            kc = k[:, :, k0:c1]; vc = v[:, :, k0:c1]     # span <= 2w
            qi = torch.arange(c0, c1, device=q.device)[:, None]
            ki = torch.arange(k0, c1, device=q.device)[None, :]
            band = (ki <= qi) & (ki > qi - w)
            mask = torch.zeros(c1 - c0, c1 - k0, device=q.device, dtype=q.dtype).masked_fill(~band, float("-inf"))
            outs.append(F.scaled_dot_product_attention(qc, kc, vc, attn_mask=mask))
        return torch.cat(outs, dim=2)

    def forward(self, h):
        B, T, d = h.shape
        x = self._apply_conv(self.norm1(h))
        cos, sin = _rope_cache(T, self.dh, h.device, h.dtype) if self.cfg.rope else (None, None)
        q, k, v, beta, alpha, beta_e, eta = self._encode(x, cos, sin)
        o = torch.zeros(B, self.H, T, self.dh, device=h.device, dtype=h.dtype)
        if self.cfg.use_workspace:
            # DeltaProduct: for n_delta>=2, build multi-sub-step k,v,beta tensors
            if self.cfg.n_delta >= 2:
                # sub-step 0 uses the main projection; sub-steps 1..n_delta-1 use W_kv_extra
                ks = [k]    # each [B,H,T,d_h]
                vs = [v]
                bs = [beta]
                for i, (wkv, wbeta) in enumerate(zip(self.W_kv_extra, self.W_beta_extra)):
                    kv_i = wkv(x).view(B, T, self.H, 2, self.dh)   # [B,T,H,2,dh]
                    k_i, v_i = kv_i.unbind(3)                       # each [B,T,H,dh]
                    k_i = k_i.transpose(1, 2); v_i = v_i.transpose(1, 2)   # [B,H,T,dh]
                    k_i = _l2(k_i)                                   # unit-norm
                    b_i = torch.sigmoid(wbeta(x)).transpose(1, 2) * self.cfg.beta_cap  # [B,H,T]
                    ks.append(k_i); vs.append(v_i); bs.append(b_i)
                # Stack on a new sub-step axis: [B,H,T,n_delta,d_h]
                k_nd = torch.stack(ks, dim=3)
                v_nd = torch.stack(vs, dim=3)
                b_nd = torch.stack(bs, dim=3)
                # phi applied per sub-step; for simplicity phi(k_sub_0) uses main k (already phi'd below)
                # For k>=2 we apply phi to sub-step 0 key; extra sub-steps use linear keys (d_h, not d_phi)
                # NOTE: n_delta>=2 uses d_h-dimensional state (no feat_map for sub-steps 1+, linear keys)
                # Sub-step 0 key goes through phi; but the state dim must be consistent: we use d_h for all
                # sub-steps when n_delta>=2 (phi expansion is not combined with n_delta in this impl).
                o_delta, _ = chunked_delta(q, k_nd, v_nd, b_nd, alpha,
                                           chunk=self.cfg.chunk, write_mode=self.cfg.write_mode,
                                           beta_e=None, n_delta=self.cfg.n_delta)  # [B,H,T,d_h]
            else:
                # n_delta=1: standard path (byte-identical when surprise_gate=False, with phi and beta_e)
                # Lever G: eta (per-value-channel LR) threads through; eta=None keeps the fast path.
                # Lever A 'random' control needs an explicit torch.Generator (_surprise_gate asserts
                # gen is not None for mode='random'). Own a reproducible one on the COMPUTE device,
                # re-seeded each forward so two passes match byte-for-byte (R8/R9). 'norm'/'constant'
                # need no generator -> surprise_gen stays None and the path is byte-identical to before.
                surprise_gen = None
                if self.cfg.surprise_gate and self.cfg.surprise_mode == 'random':
                    surprise_gen = torch.Generator(device=q.device).manual_seed(self.cfg.surprise_seed)
                # PR-2026-09-03-01 A4: thread the frozen gate tuple (gain, EMA lambda, beta_cap).
                # None on every existing path -> chunked_delta's fast WY/UT path byte-identical.
                # Under 'surprise_norm' chunked_delta routes to the EXACT sequential scan
                # (_delta_reference): beta_t depends on the running state (R3) — the disclosed
                # 2-5x training tax of the pre-registration, documented in PrizmaSeqConfig.
                surprise_norm = None
                if self.cfg.precision_gate == "surprise_norm":
                    surprise_norm = (self.cfg.surprise_gain, self.cfg.surprise_ema_lambda,
                                     self.cfg.beta_cap)
                o_delta, _ = chunked_delta(self._phi(q), self._phi(k), v, beta, alpha,
                                           chunk=self.cfg.chunk, write_mode=self.cfg.write_mode,
                                           beta_e=beta_e,
                                           surprise=self.cfg.surprise_gate,
                                           surprise_mode=self.cfg.surprise_mode,
                                           surprise_gen=surprise_gen,
                                           eta=eta, surprise_norm=surprise_norm)   # [B,H,T,d_h]
            # delta state keyed by phi(q),phi(k) (dim d_phi); values stay d_h -> state [B,H,d_h,d_phi]
            if self.state_rms is not None:
                o_delta = self.state_rms(o_delta)    # per-head RMSNorm over d_h
            o = o + o_delta
        if self.cfg.use_window:
            win_fn = self._window_banded if self.cfg.banded_window else self._window
            o = o + win_fn(q, k, v)                  # window head keeps the LINEAR L2 keys (dim d_h)
        o = o.transpose(1, 2).reshape(B, T, d)                                     # merge heads
        if self.W_g is not None:
            o = o * torch.sigmoid(self.W_g(self.norm1(h)))   # gate on block input (pre-conv normed)
        # opt-in residual dropout on the mixer + MLP outputs (p=0 => identity, no rng -> byte-identical)
        h = h + self.drop(self.W_o(o))
        h = h + self.drop(self.mlp(self.norm2(h)))
        return h

    # ---- analog levers (report 12-H2): noisy write + low-precision carried state ------------
    def _noisy(self, u, pos, sub=0):
        """Add write noise ~ N(0, write_noise_std^2) to the write update u (step() path only).
        Owns a FRESH generator per call seeded write_noise_seed + pos*8 + sub, so the noise is a
        deterministic function of (config, token index, sub-step) — i.i.d. across writes and
        reproducible for a fixed evaluation order (surprise_seed discipline, R8/R9). Guarded: at
        write_noise_std == 0.0 this draws NO rng and returns u untouched (bit-identity)."""
        if self.cfg.write_noise_std <= 0.0:
            return u
        g = torch.Generator(device=u.device).manual_seed(self.cfg.write_noise_seed + pos * 8 + sub)
        return u + self.cfg.write_noise_std * torch.randn(u.shape, generator=g,
                                                          device=u.device, dtype=u.dtype)

    def _quantized(self, S):
        """Quantize the carried state S post-write (step() path only). Guarded: state_bits == 0
        returns S untouched (bit-identity; no extra ops at all on the default path)."""
        if self.cfg.state_bits <= 0:
            return S
        return _quantize_state_symmetric(S, self.cfg.state_bits)

    # ---- O(1)-per-step inference path (for B5 latency / true streaming) ---- #
    @torch.no_grad()
    def step(self, h_t, state):
        """h_t:[B,1,d]; state=(S, ring_k, ring_v, conv_ring, pos, ema_m). Returns o_t, new_state. O(1).
        The 6th slot ema_m is the per-head EMA of the write-error energy for the PR-2026-09-03-01
        A4 surprise-norm gate (init zeros; the first token, pos==0, sets m = s — mirroring
        forward()'s per-sequence reset m_1 = s_1 exactly). All pre-A4 modes carry it untouched
        (no ops, no rng -> bit-identical outputs).
        Analog levers (state_bits / write_noise_std, report 12-H2) degrade THIS path only — at
        their defaults (0 / 0.0) this method is bit-identical to the pre-lever implementation; with
        them ON, step() intentionally deviates from the FP32 forward() (documented scope limit)."""
        B = h_t.shape[0]
        S, rk, rv, cring, pos, ema = state
        m_new = ema   # carried unchanged by every mode except surprise_norm (which updates it)
        xin = self.norm1(h_t)                                    # [B,1,d]
        if self.kc > 0:
            buf = torch.cat([cring, xin], dim=1)                 # [B,kc,d]
            w = self.conv.weight.squeeze(1)                      # [d,kc]
            xc = (buf.transpose(1, 2) * w).sum(-1) + self.conv.bias   # [B,d]
            x = F.silu(xc)[:, None, :]                           # [B,1,d]
            cring = buf[:, 1:, :]                                # keep last kc-1
        else:
            x = xin
        cos, sin = _rope_cache(1, self.dh, h_t.device, h_t.dtype, offset=pos) if self.cfg.rope else (None, None)
        q, k, v, beta, alpha, beta_e, eta = self._encode(x, cos, sin)  # [B,H,1,dh], beta [B,H,1]
        q1, k1, v1 = q[:, :, 0], k[:, :, 0], v[:, :, 0]          # [B,H,dh] (linear L2; window keys)
        # beta is None only under precision_gate='surprise_norm' (A4): the gate is computed inside
        # the recurrence below from eps1 — b1/be1 are then unused (and None).
        b1 = beta[:, :, 0] if beta is not None else None         # [B,H]  write gate beta_w
        a1 = (alpha[:, :, 0] if alpha is not None
              else torch.ones(B, self.H, device=h_t.device, dtype=h_t.dtype))
        # pre-write read (always uses state from end of previous token)
        if self.cfg.n_delta >= 2:
            o_delta = torch.einsum("bhij,bhj->bhi", S, q1)        # [B,H,d_h] (d_h state for n_delta>=2)
        else:
            q1p = self._phi(q)[:, :, 0]                           # [B,H,d_phi] (delta state keys)
            o_delta = torch.einsum("bhij,bhj->bhi", S, q1p)       # pre-write read S_{t-1} phi(q)
        if self.cfg.n_delta >= 2:
            # DeltaProduct: apply n_delta sub-steps sequentially (mirrors the chunked/reference form)
            # Sub-step 0: main kv + alpha decay. Analog levers: noise on the write, quantize the
            # carried state after the token's sub-step loop (same deployment emulation as n_delta==1).
            # FIX (analog-probe finding, 2026-09-03): step() previously applied the DELTA write
            # unconditionally, ignoring write_mode='additive' — an additive-trained model deployed
            # through step() ran a write rule it was never trained with (streaming 0.358 vs
            # forward 0.742 on additive MixedMQAR; no existing test covered additive+step). The
            # additive branches below mirror _delta_reference / chunked_delta exactly; the delta
            # path is byte-identical to the pre-fix code.
            k1_step = k1; v1_step = v1
            if self.cfg.write_mode == "additive":
                u = self._noisy(b1[..., None] * v1_step, pos, sub=0)
            else:
                Sk = torch.einsum("bhij,bhj->bhi", S, k1_step)
                u = self._noisy(b1[..., None] * v1_step - b1[..., None] * (a1[..., None] * Sk),
                                pos, sub=0)
            S = a1[..., None, None] * S + torch.einsum("bhi,bhj->bhij", u, k1_step)
            # Sub-steps 1..(n_delta-1): extra projections, no additional alpha decay
            x1 = x[:, 0, :]                                       # [B,d] (squeeze T=1 dim)
            for j, (wkv, wbeta) in enumerate(zip(self.W_kv_extra, self.W_beta_extra)):
                kv_j = wkv(x1).view(B, self.H, 2, self.dh)        # [B,H,2,dh]
                k_j, v_j = kv_j[:, :, 0], kv_j[:, :, 1]           # [B,H,dh]
                k_j = _l2(k_j)
                b_j = torch.sigmoid(wbeta(x1)).view(B, self.H) * self.cfg.beta_cap  # [B,H]
                if self.cfg.write_mode == "additive":
                    u_j = self._noisy(b_j[..., None] * v_j, pos, sub=j + 1)
                else:
                    Sk_j = torch.einsum("bhij,bhj->bhi", S, k_j)
                    u_j = self._noisy(b_j[..., None] * (v_j - Sk_j), pos, sub=j + 1)
                S = S + torch.einsum("bhi,bhj->bhij", u_j, k_j)
            S = self._quantized(S)
        else:
            k1p = self._phi(k)[:, :, 0]                           # [B,H,d_phi] (delta state keys)
            be1 = beta_e[:, :, 0] if beta_e is not None else b1   # [B,H]  erase gate beta_e
            Sk = torch.einsum("bhij,bhj->bhi", S, k1p)            # [B,H,d_h]
            # Prediction error (free-energy gradient at S_{t-1}): eps = v - alpha*S*k
            eps1 = v1 - a1[..., None] * Sk                        # [B,H,d_h]
            if self.cfg.write_mode == "additive":
                # additive (linear-attn) write: u = beta_w * v, NO erase read-back. Mirrors
                # _delta_reference write_mode='additive' (u = eta*v under Lever G) exactly, so
                # step()==forward() now covers the additive ablation too. FIX: before the
                # analog-probe finding (2026-09-03) this path fell through to the delta write.
                eta1 = eta[:, :, 0] if eta is not None else None
                u = (eta1 * v1) if eta1 is not None else b1[..., None] * v1   # [B,H,d_h]
            elif eta is not None:
                # Lever G: per-VALUE-channel in-context LR replaces the scalar write gate. Mirrors
                # _delta_reference: u = eta_t (elementwise over value channels) * eps_t. Same eta the
                # parallel forward() applies -> step()==forward() (G1 O(1) guard).
                eta1 = eta[:, :, 0]                               # [B,H,d_h]
                u = eta1 * eps1                                   # [B,H,d_h]
            elif self.cfg.surprise_gate:
                # g_t = 1 + tanh(||eps_t||); same formula as _delta_reference / _surprise_gate 'norm'
                # (only 'norm' mode is used in step() — the 'random'/'constant' modes are ablation-only
                # and are not wired through step() since they are not meaningful at inference time).
                # For step==forward parity with surprise_mode='norm', this is exact.
                g1 = (1.0 + torch.tanh(eps1.norm(dim=-1)))[..., None]   # [B,H,1]
                # Apply g to full write vector: u = g * (beta_w*v - beta_e*alpha*Sk)
                u = g1 * (b1[..., None] * v1 - be1[..., None] * (a1[..., None] * Sk))
            elif self.cfg.precision_gate == "surprise_norm":
                # PR-2026-09-03-01 A4 (frozen formula; mirrors _delta_reference's erase branch
                # EXACTLY via the shared helpers, so the G1 step()==forward() guard holds <1e-4):
                #   s_t = ||eps_t||^2 ; m_1 = s_1 (pos==0), else causal EMA with lam=0.02 ;
                #   beta_t = beta_cap * sigmoid(a * (s_t/max(m_t,1e-6) - 1)) — per head;
                #   u_t = beta_t*v_t - beta_t*(alpha_t*S_{t-1}k_t)   (erase gate == write gate).
                # The EMA lives in the streaming state's 6th slot (init zeros; carried below).
                s1 = (eps1 * eps1).sum(dim=-1)                       # [B,H] write-error energy
                m_t = _surprise_norm_update(s1, None if pos == 0 else ema,
                                            self.cfg.surprise_ema_lambda)
                b4 = _surprise_norm_gate(s1, m_t, self.cfg.surprise_gain, self.cfg.beta_cap)
                u = b4[..., None] * v1 - b4[..., None] * (a1[..., None] * Sk)
                m_new = m_t
            else:
                # decoupled: u = beta_w * v  -  beta_e * (alpha * S k)
                u = b1[..., None] * v1 - be1[..., None] * (a1[..., None] * Sk)   # [B,H,d_h]
            # Analog levers (report 12-H2): noise the write, then quantize the carried state
            # post-write so the NEXT token's pre-write read sees the degraded S (write-and-read
            # both degraded). Both calls are identity + rng-free at the defaults (bit-identity).
            u = self._noisy(u, pos, sub=0)
            S = a1[..., None, None] * S + torch.einsum("bhi,bhj->bhij", u, k1p)   # [B,H,d_h,d_phi]
            S = self._quantized(S)
        if self.state_rms is not None:
            o_delta = self.state_rms(o_delta)    # per-head RMSNorm [B,H,d_h], mirrors forward
        # window ring
        rk = torch.cat([rk, k1[:, :, None]], dim=2)[:, :, -self.cfg.window:]
        rv = torch.cat([rv, v1[:, :, None]], dim=2)[:, :, -self.cfg.window:]
        sc = torch.einsum("bhd,bhwd->bhw", q1, rk) * self.win_scale
        aw = torch.softmax(sc, dim=-1)
        o_win = torch.einsum("bhw,bhwd->bhd", aw, rv)
        o = (o_delta + o_win).reshape(B, 1, -1)
        if self.W_g is not None:
            o = o * torch.sigmoid(self.W_g(self.norm1(h_t)))   # gate on block input, mirrors forward
        # mirror forward's residual dropout (at p=0/eval it is identity -> step()==forward() guard holds)
        h = h_t + self.drop(self.W_o(o))
        h = h + self.drop(self.mlp(self.norm2(h)))
        return h, (S, rk, rv, cring, pos + 1, m_new)


class PrizmaSeqLM(nn.Module):
    def __init__(self, cfg: PrizmaSeqConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model) if cfg.learned_pos else None
        # opt-in embedding dropout (default 0.0 == identity, no rng -> byte-identical).
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([PrizmaSeqBlock(cfg) for _ in range(cfg.n_layers)])
        self.nf = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        h = self.tok(idx)
        if self.pos is not None:
            h = h + self.pos(torch.arange(T, device=idx.device))[None]
        h = self.drop(h)   # embedding dropout (p=0 => identity, no rng -> byte-identical)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.nf(h))

    @torch.no_grad()
    def init_state(self, batch, device):
        st = []
        kc1 = max(self.cfg.short_conv - 1, 0)
        # For n_delta>=2, state uses d_h-dim keys (no phi expansion); for n_delta=1 uses d_phi
        state_k_dim = self.cfg.d_h if self.cfg.n_delta >= 2 else self.cfg.d_phi
        for _ in self.blocks:
            S = torch.zeros(batch, self.cfg.n_heads, self.cfg.d_h, state_k_dim, device=device)
            rk = torch.zeros(batch, self.cfg.n_heads, 0, self.cfg.d_h, device=device)
            rv = torch.zeros(batch, self.cfg.n_heads, 0, self.cfg.d_h, device=device)
            cring = torch.zeros(batch, kc1, self.cfg.d_model, device=device)
            # 6th slot: per-head EMA m of the write-error energy (PR-2026-09-03-01 A4). Init zeros;
            # the first streamed token (pos==0) sets m = s (== m_1 = s_1 in the batched forward).
            ema = torch.zeros(batch, self.cfg.n_heads, device=device)
            st.append((S, rk, rv, cring, 0, ema))
        return st

    @torch.no_grad()
    def step(self, tok, state):
        """tok:[B,1] -> (logits[B,1,V], new_state). O(1) in sequence length."""
        h = self.tok(tok)
        if self.pos is not None:
            p = state[0][4] if state else 0
            h = h + self.pos(torch.tensor([p], device=tok.device))[None]
        h = self.drop(h)   # mirror forward's embedding dropout (p=0/eval => identity)
        new = []
        for blk, st in zip(self.blocks, state):
            h, st2 = blk.step(h, st)
            new.append(st2)
        return self.head(self.nf(h)), new


def prizma_seq_factory(d_model=64, n_layers=2, n_heads=2, **kw):
    def f(vocab, max_len):
        return PrizmaSeqLM(PrizmaSeqConfig(vocab=vocab, d_model=d_model, n_layers=n_layers,
                                         n_heads=n_heads, max_len=max_len + 8, **kw))
    return f


if __name__ == "__main__":
    from .common import param_count, get_device
    dev = get_device()
    # O(1) GUARD (committee guardrail): for ALL feat_map settings the streaming step() must equal
    # the parallel forward() to <1e-4, AND param_count must be identical (feature map = 0 params).
    ref_params = None
    for feat in ("none", "quad2", "quad2_lowrank"):
        cfg = PrizmaSeqConfig(vocab=64, d_model=64, n_layers=2, n_heads=2, feat_map=feat)
        m = PrizmaSeqLM(cfg).to(dev)
        x = torch.randint(0, 64, (2, 48), device=dev)
        y = m(x)
        m.train(False)
        st = m.init_state(2, dev)
        outs = []
        for t in range(x.shape[1]):
            lg, st = m.step(x[:, t:t + 1], st)
            outs.append(lg)
        yo = torch.cat(outs, dim=1)
        d = (y - yo).abs().max().item()
        p = param_count(m)
        if ref_params is None:
            ref_params = p
        param_ok = (p == ref_params)
        print(f"[feat_map={feat:<12} d_phi={cfg.d_phi:<4}] forward {tuple(y.shape)} "
              f"params {p} {'(MATCH)' if param_ok else '(MISMATCH!)'} "
              f"step-vs-forward max|d|={d:.2e} {'OK' if d < 1e-4 else 'MISMATCH'}")
    # SURPRISE GATE O(1) guard: surprise_gate=True step()==forward() < 1e-4
    cfg_s = PrizmaSeqConfig(vocab=64, d_model=64, n_layers=2, n_heads=2,
                            feat_map='none', surprise_gate=True, surprise_mode='norm')
    m_s = PrizmaSeqLM(cfg_s).to(dev)
    m_s.train(False)
    torch.manual_seed(7)
    x_s = torch.randint(0, 64, (2, 48), device=dev)
    y_s = m_s(x_s)
    st_s = m_s.init_state(2, dev)
    outs_s = []
    for t in range(x_s.shape[1]):
        lg_s, st_s = m_s.step(x_s[:, t:t + 1], st_s)
        outs_s.append(lg_s)
    yo_s = torch.cat(outs_s, dim=1)
    d_s = (y_s - yo_s).abs().max().item()
    print(f"[surprise_gate=True norm] step-vs-forward max|d|={d_s:.2e} {'OK' if d_s < 1e-4 else 'MISMATCH'}")
    # IN-CONTEXT PER-CHANNEL LR O(1) guard (Lever G): inctx_lr=True step()==forward() < 1e-4
    cfg_g = PrizmaSeqConfig(vocab=64, d_model=64, n_layers=2, n_heads=2,
                            feat_map='quad2', inctx_lr=True)
    m_g = PrizmaSeqLM(cfg_g).to(dev)
    m_g.train(False)
    torch.manual_seed(11)
    x_g = torch.randint(0, 64, (2, 48), device=dev)
    y_g = m_g(x_g)
    st_g = m_g.init_state(2, dev)
    outs_g = []
    for t in range(x_g.shape[1]):
        lg_g, st_g = m_g.step(x_g[:, t:t + 1], st_g)
        outs_g.append(lg_g)
    yo_g = torch.cat(outs_g, dim=1)
    d_g = (y_g - yo_g).abs().max().item()
    print(f"[inctx_lr=True quad2]    step-vs-forward max|d|={d_g:.2e} {'OK' if d_g < 1e-4 else 'MISMATCH'}")
    # SURPRISE-NORM GATE O(1) guard (PR-2026-09-03-01 A4): step()==forward() < 1e-4. The EMA is
    # state-carried in step() (init zeros; first token sets m = s) and reset per sequence in
    # forward() — the two must agree token-by-token, EMA included.
    cfg_a4 = PrizmaSeqConfig(vocab=64, d_model=64, n_layers=2, n_heads=2,
                             feat_map='quad2_lowrank', precision_gate='surprise_norm',
                             surprise_gain=2.0)
    m_a4 = PrizmaSeqLM(cfg_a4).to(dev)
    m_a4.train(False)
    torch.manual_seed(13)
    x_a4 = torch.randint(0, 64, (2, 48), device=dev)
    y_a4 = m_a4(x_a4)
    st_a4 = m_a4.init_state(2, dev)
    outs_a4 = []
    for t in range(x_a4.shape[1]):
        lg_a4, st_a4 = m_a4.step(x_a4[:, t:t + 1], st_a4)
        outs_a4.append(lg_a4)
    yo_a4 = torch.cat(outs_a4, dim=1)
    d_a4 = (y_a4 - yo_a4).abs().max().item()
    print(f"[surprise_norm A4 quad2lr] step-vs-forward max|d|={d_a4:.2e} "
          f"{'OK' if d_a4 < 1e-4 else 'MISMATCH'}")
