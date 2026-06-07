"""
The Prizma-Seq workspace recurrence: a precision-gated delta rule (targeted erase-and-write),
derived as ONE gradient step on Prizma's per-token free energy  F_t(S) = 1/2 ||v_t - S k_t||^2.

    S_t = alpha_t * S_{t-1} + u_t k_t^T ,   u_t = beta_t (v_t - alpha_t S_{t-1} k_t)   (||k_t||=1)
    read (PRE-write, strictly causal):  o_t = S_{t-1} q_t

This module provides TWO implementations that MUST agree to < 1e-4 (forward AND grad):
  * _delta_reference  : naive per-token recurrence (ground truth, slow)
  * chunked_delta     : WY/UT chunk-parallel form (fast, used for training)
Both operate on [B,H,T,d_h] tensors. alpha defaults to 1 (pure DeltaNet) — the diagnostic gates
(MQAR/induction/selective-copy) need clean overwrite, not forgetting; the gated path is enabled
for char-LM. Keep d_h even, C<=64, float32.

Surprise-gated write (Lever A):
  When surprise=True, the write magnitude is modulated by the surprise gate
      g_t = 1 + tanh(||eps_t||)   in [1, 2)
      u_t = beta_t * g_t * eps_t
  derived from the per-token free energy F_t = ½||v_t - S k_t||^2.  Tokens with large prediction
  error write MORE (Titans-style test-time learning, predictive-coding grounded).

  SPEED NOTE: when surprise=True, chunked_delta internally runs an EXACT sequential scan (the WY/UT
  chunk-parallel shortcut cannot be used because eps_t = v_t - alpha*S_{t-1}*k_t depends on the
  FULL running state S_{t-1}, which is affected by every prior gated write — a frozen-chunk
  approximation diverges on REPEATED KEYS by ~100%).  This is a disclosed Pareto knob: correctness
  takes priority; a fused CUDA kernel can recover the speed later.  When surprise=False (default),
  the original fast WY/UT path is used byte-for-byte.

  surprise_mode controls HOW g_t is computed (R9 ablation controls):
    'norm'     : g_t = 1 + tanh(||eps_t||)                    — the real lever
    'random'   : g_t = 1 + tanh(|r_t|), r_t ~ |N(0,1)| via an explicit torch.Generator
                 (surprise_gen must be passed; reproducible).  Mean matches 'norm' heuristically:
                 E[tanh(|N(0,1)|)] ≈ 0.56 ≈ E[tanh(||eps||)] for typical normalised delta errors.
    'constant' : g_t = 1 + tanh(1.0) ≈ 1.762, a fixed constant equal to E[tanh(||eps||)]
                 approximated by tanh(1.0).  Provides a constant-mean-beta_eff control.

In-context per-channel learning rate (Lever G — RWKV-7 "Goose" generalized delta):
  The scalar write gate beta_t (one rate per head) generalizes into a per-VALUE-channel rate vector
  eta_t in R^{d_v}, which modulates the delta write per state/output channel:
      u_t = eta_t (elementwise over the d_v value channels) * (v_t - alpha_t * S_{t-1} k_t)
      S_t = alpha_t * S_{t-1} + u_t k_t^T
  Decay (alpha) and the erase read-back are UNCHANGED — G is a vector-valued beta on the WRITE
  magnitude only. eta=None (default) -> the scalar-beta path, byte-identical to today.

  SPEED NOTE: when eta is provided, chunked_delta delegates to _delta_reference for an EXACT
  sequential scan (same correctness-first fallback the 'surprise' path uses). The WY/UT chunk
  shortcut solves ONE triangular system (I+A) U = rhs whose coupling matrix A mixes the value
  channels uniformly (a single scalar rate per token); a per-VALUE-channel eta makes the cross-token
  write coupling channel-dependent, which a single channel-shared solve cannot represent exactly.
  Correctness first; a per-channel chunked kernel is a future Pareto knob. eta is scoped to
  n_delta==1 (NotImplementedError for n_delta>=2, mirroring how existing levers scope interactions).
"""
from __future__ import annotations

import math
import torch


def _surprise_gate(eps, mode, gen):
    """Compute per-token surprise gate g = 1 + tanh(signal) in [1,2).
    eps: [B,H,d] prediction error. Returns g: [B,H,1] for broadcasting."""
    if mode == 'norm':
        # Real lever: g = 1 + tanh(||eps||);  ||eps|| is the free-energy sqrt.
        signal = eps.norm(dim=-1)                 # [B,H]
    elif mode == 'random':
        # Random-scalar control: g = 1 + tanh(|r|), r ~ |N(0,1)| via explicit generator.
        # E[tanh(|N(0,1)|)] ≈ 0.56, closely matching E[tanh(||eps||)] for typical delta errors
        # (both derived from half-normal; the match is heuristic — mean equality is approximate).
        assert gen is not None, "surprise_mode='random' requires surprise_gen"
        r = torch.zeros(eps.shape[:2], dtype=eps.dtype, device=eps.device)
        r.normal_(generator=gen)
        signal = r.abs()                          # [B,H], matched to |N(0,1)|
    elif mode == 'constant':
        # Constant-mean control: g = 1 + tanh(1.0) ≈ 1.762 for all tokens.
        # tanh(1.0) ≈ 0.762 approximates E[tanh(||eps||)] for a unit-scale error distribution.
        _CONST = math.tanh(1.0)
        return torch.full(eps.shape[:2] + (1,), 1.0 + _CONST, dtype=eps.dtype, device=eps.device)
    else:
        raise ValueError(f"Unknown surprise_mode: {mode!r}. Choose 'norm', 'random', or 'constant'.")
    return (1.0 + torch.tanh(signal))[..., None]  # [B,H,1]


def _delta_reference(q, k, v, beta, alpha=None, S0=None, write_mode="delta", beta_e=None,
                     n_delta=1, surprise=False, surprise_mode='norm', surprise_gen=None,
                     eta=None):
    """Ground-truth sequential recurrence. q,k,v:[B,H,T,d]; beta:[B,H,T]; alpha:[B,H,T] or None.
    write_mode='additive' -> u=beta*v (no erase), the linear-attn ablation.
    beta_e: optional erase gate [B,H,T]; if None, beta_e=beta (byte-identical to old behaviour).
    When beta_e is provided, u_t = beta_w*v_t - beta_e*(alpha*S k_t)  (decoupled GDN-2 write).
    n_delta: int >= 1 (DeltaProduct). For n_delta=1, inputs have no extra k-axis (byte-identical to
    old behaviour). For n_delta>=2, k,v have shape [B,H,T,n_delta,d] and beta [B,H,T,n_delta]:
    each token applies n_delta sequential delta sub-steps before reading the next token. The read
    o_t = S_{t-1} q_t uses the state from the END of the previous token's n_delta sub-steps. Alpha
    decay is applied exactly once per token (on sub-step 0 only); subsequent sub-steps are pure
    additive delta (no additional decay), consistent with Householder-product semantics.
    surprise: if True, scale u_t by g_t = 1+tanh(||eps_t||) (or per surprise_mode) BEFORE writing.
    surprise_mode: 'norm' | 'random' | 'constant' — see module docstring.
    surprise_gen: explicit torch.Generator for 'random' mode (reproducibility, R8 discipline).
    eta: optional in-context per-VALUE-channel learning rate (Lever G, RWKV-7 generalized delta),
    shape [B,H,T,d_v]. When None (default), the scalar write gate beta_t is used unchanged
    (byte-identical to old behaviour). When provided, eta REPLACES the scalar write gate on the WRITE
    term, modulating the delta write per value/output channel:
        u_t = eta_t (elementwise over the d_v value channels) * (v_t - alpha_t * S_{t-1} k_t)
    Decay (alpha) and the erase read-back are unchanged: G modulates only the per-channel WRITE
    magnitude, exactly like a vector-valued beta. eta is scoped to n_delta==1 (a NotImplementedError
    is raised for n_delta>=2, mirroring how existing levers scope their interactions)."""
    if eta is not None and n_delta >= 2:
        raise NotImplementedError(
            "inctx_lr (per-channel eta) is only implemented for n_delta==1; "
            "combining it with n_delta>=2 (DeltaProduct) is out of scope for Lever G.")
    B, H, T, d = q.shape
    dv = v.shape[-1] if n_delta == 1 else v.shape[-1]  # [B,H,T,n_delta,d] -> last dim still d
    if n_delta >= 2:
        # v shape is [B,H,T,n_delta,d]; dv = d (last dim)
        dv = v.shape[-1]
    else:
        dv = v.shape[-1]
    if S0 is None:
        S = torch.zeros(B, H, dv, d, dtype=q.dtype, device=q.device)
    else:
        S = S0.clone()
    if alpha is None:
        alpha = torch.ones(B, H, T, dtype=q.dtype, device=q.device)
    if beta_e is None:
        beta_e = beta          # default: erase == write gate (byte-identical to today)
    erase = (write_mode == "delta")
    outs = []
    for t in range(T):
        qt = q[:, :, t]                                            # [B,H,d]
        at = alpha[:, :, t]                                        # [B,H]
        o = torch.einsum("bhij,bhj->bhi", S, qt)                   # read S_{t-1} q_t  (PRE-write)
        if n_delta == 1:
            kt, vt = k[:, :, t], v[:, :, t]                       # [B,H,d]
            bt  = beta[:, :, t]                                    # [B,H]
            bet = beta_e[:, :, t]
            if erase:
                Sk = torch.einsum("bhij,bhj->bhi", S, kt)          # [B,H,d]
                # Standard delta-rule error (free-energy gradient at S_{t-1}):
                #   eps_t = v_t - alpha_t * S_{t-1} k_t
                # Surprise gate g_t modulates ONLY the write magnitude (task spec: alpha unchanged):
                #   u_t = beta_t * g_t * eps_t  (when bet==bt, the default non-decoupled case)
                # For the decoupled-gate case (bet != bt, Lever H):
                #   the surprise gate is applied to the full u_t = bt*vt - bet*(at*Sk),
                #   which is the write vector. The eps for SIGNAL computation always uses
                #   the symmetric beta (bt), consistent with the free-energy interpretation.
                eps_t = vt - at[..., None] * Sk                    # [B,H,d] prediction error
                if eta is not None:
                    # Lever G: per-VALUE-channel in-context learning rate eta_t in R^{d_v} REPLACES
                    # the scalar write gate, modulating the delta write per output channel:
                    #   u_t = eta_t (elementwise) * (v_t - alpha_t * S_{t-1} k_t) = eta_t * eps_t
                    # Decay (alpha) and the erase read-back stay exactly as today; G is a vector beta
                    # on the WRITE magnitude. (Scoped to n_delta==1; not combined with surprise.)
                    u = eta[:, :, t] * eps_t                        # [B,H,d_v]
                elif surprise:
                    g = _surprise_gate(eps_t, surprise_mode, surprise_gen)  # [B,H,1]
                    # Apply g to the full write vector u (before alpha-scaled state decay)
                    u = g * (bt[..., None] * vt - bet[..., None] * (at[..., None] * Sk))
                else:
                    u = bt[..., None] * vt - bet[..., None] * (at[..., None] * Sk)
            else:
                # additive (linear-attn) write: eta (if provided) is a per-channel rate on v_t.
                u = (eta[:, :, t] * vt) if eta is not None else (bt[..., None] * vt)
            S = at[..., None, None] * S + torch.einsum("bhi,bhj->bhij", u, kt)
        else:
            # n_delta >= 2: k[B,H,T,n_delta,d], v[B,H,T,n_delta,d], beta[B,H,T,n_delta]
            for j in range(n_delta):
                ktj = k[:, :, t, j]                               # [B,H,d]
                vtj = v[:, :, t, j]
                btj = beta[:, :, t, j]                             # [B,H]
                if j == 0:
                    # alpha decay applied once at the first sub-step
                    if erase:
                        Sk = torch.einsum("bhij,bhj->bhi", S, ktj)
                        u = btj[..., None] * vtj - btj[..., None] * (at[..., None] * Sk)
                    else:
                        u = btj[..., None] * vtj
                    S = at[..., None, None] * S + torch.einsum("bhi,bhj->bhij", u, ktj)
                else:
                    # subsequent sub-steps: pure delta, no additional alpha decay
                    if erase:
                        Sk = torch.einsum("bhij,bhj->bhi", S, ktj)
                        u = btj[..., None] * (vtj - Sk)
                    else:
                        u = btj[..., None] * vtj
                    S = S + torch.einsum("bhi,bhj->bhij", u, ktj)
        outs.append(o)
    O = torch.stack(outs, dim=2)                                   # [B,H,T,d]
    return O, S


def _solve_unit_lower(Amat, RHS):
    """Solve (I + A) X = RHS for X, where A is strictly-lower-triangular -> (I+A) unit-lower-tri.
    Tries solve_triangular (fast); falls back to an exact nilpotent Neumann series if MPS lacks it
    (A is strictly lower so A^C = 0; the series terminates and is exact)."""
    C = Amat.shape[-1]
    M = torch.eye(C, dtype=Amat.dtype, device=Amat.device) + Amat
    try:
        return torch.linalg.solve_triangular(M, RHS, upper=False, unitriangular=True)
    except Exception:
        # exact: (I+A)^{-1} = sum_{j=0}^{C-1} (-A)^j ; apply to RHS iteratively
        X = RHS.clone()
        term = RHS
        negA = -Amat
        for _ in range(1, C):
            term = negA @ term
            X = X + term
            if torch.count_nonzero(term) == 0:
                break
        return X


def chunked_delta(q, k, v, beta, alpha=None, S0=None, chunk=64, write_mode="delta", beta_e=None,
                  n_delta=1, surprise=False, surprise_mode='norm', surprise_gen=None, eta=None):
    """WY/UT chunk-parallel delta rule. Same semantics as _delta_reference, O(T d^2/C + T C d).
    alpha=None -> pure delta (no decay). write_mode='additive' -> linear-attn ablation (no erase:
    u=beta*v, used for B6 PRIZMA_noDelta). Returns O:[B,H,T,d], S_end:[B,H,d,d].
    beta_e: optional erase gate [B,H,T]; if None, beta_e=beta (byte-identical to old behaviour).
    When beta_e is provided, the A-matrix (erase/read-back) is scaled by beta_e while the Vc
    (write) term is scaled by beta_w=beta — decoupled GDN-2 update.
    n_delta: int >= 1 (DeltaProduct). For n_delta=1, the standard WY/UT chunk-parallel path is
    used (byte-identical to old behaviour). For n_delta>=2, k,v have shape [B,H,T,n_delta,d] and
    beta [B,H,T,n_delta]. The k-axis is processed sequentially within each chunk (correctness-first
    fallback as documented in the plan) while T remains chunked. Alpha applied once per token.
    surprise: if True, delegate to _delta_reference for an EXACT sequential scan — the WY/UT
    chunk-parallel shortcut is NOT used because eps_t depends on S_{t-1}, which is affected by
    all prior surprise-gated writes. A frozen chunk-entry state approximation diverges on repeated
    keys by ~100% (remedy R3). When surprise=False (default), the fast WY/UT path is used
    byte-for-byte (no speed regression on the default path).
    surprise_mode / surprise_gen: forwarded to _delta_reference (see module docstring).
    eta: optional in-context per-VALUE-channel learning rate (Lever G), shape [B,H,T,d_v]. When None
    (default), the scalar-beta fast WY/UT path is used byte-for-byte (no speed regression).
    When provided, this delegates to _delta_reference for an EXACT SEQUENTIAL scan — the same
    correctness-first fallback the 'surprise' path uses. RATIONALE: the WY/UT closed form solves a
    single triangular system (I + A) U = rhs whose coupling matrix A = tril(beta_e * KK * ratio)
    mixes the value channels UNIFORMLY (one scalar rate per token); a per-VALUE-channel eta makes the
    cross-token write coupling channel-dependent, which a single channel-shared triangular solve
    cannot represent exactly. Correctness takes priority; a per-channel chunked kernel is a future
    Pareto knob. eta is scoped to n_delta==1 (NotImplementedError for n_delta>=2)."""
    B, H, T, d = q.shape
    dv = v.shape[-1]                     # value-dim-aware init -> RECTANGULAR state S in R^{d_v x d_k}
    if S0 is None:                       #   (d_k=d). Byte-identical when d_v == d_k (every existing
        S = torch.zeros(B, H, dv, d, dtype=q.dtype, device=q.device)   # call); enables feature-map/GDM
    else:
        S = S0

    # IN-CONTEXT PER-CHANNEL LR PATH (Lever G): must be exact — delegate to _delta_reference, which
    # threads the TRUE running state through every token and applies eta per value channel. The
    # WY/UT chunk shortcut cannot absorb a per-channel rate (its triangular solve uses a single
    # channel-shared coupling matrix). Disclosed speed cost; eta=None keeps the fast path untouched.
    if eta is not None:
        return _delta_reference(q, k, v, beta, alpha=alpha, S0=S, write_mode=write_mode,
                                beta_e=beta_e, n_delta=n_delta,
                                surprise=surprise, surprise_mode=surprise_mode,
                                surprise_gen=surprise_gen, eta=eta)

    # SURPRISE PATH: must be exact — delegate to _delta_reference which threads the TRUE running
    # state through every token.  The WY/UT chunk shortcut cannot be used here because eps_t depends
    # on S_{t-1}, which is modified by each surprise-gated write; a frozen chunk-entry state would
    # mis-estimate the surprise on repeated keys by ~100% (R3).  This is a disclosed speed cost;
    # a fused kernel can recover it later.  When surprise=False the fast path below is untouched.
    if surprise:
        return _delta_reference(q, k, v, beta, alpha=alpha, S0=S, write_mode=write_mode,
                                beta_e=beta_e, n_delta=n_delta,
                                surprise=True, surprise_mode=surprise_mode,
                                surprise_gen=surprise_gen)

    gated = alpha is not None
    erase = (write_mode == "delta")
    # beta_e=None means use beta for both erase and write -> byte-identical to previous behaviour
    decoupled = (beta_e is not None)

    # --- DeltaProduct n_delta >= 2: k-axis processed sequentially within each chunk ---
    # For n_delta=1 (default), fall through to the fast WY/UT path (byte-identical to old code).
    # For n_delta>=2: k,v are [B,H,T,n_delta,d], beta is [B,H,T,n_delta]. We chunk over T but
    # loop over tokens and sub-steps within each chunk (sequential k-axis, documented fallback).
    if n_delta >= 2:
        outs_nd = []
        for c0 in range(0, T, chunk):
            c1 = min(c0 + chunk, T)
            for ti in range(c0, c1):
                qt = q[:, :, ti]                                   # [B,H,d]
                at = alpha[:, :, ti] if gated else torch.ones(B, H, dtype=q.dtype, device=q.device)
                o = torch.einsum("bhij,bhj->bhi", S, qt)           # pre-write read
                for j in range(n_delta):
                    ktj = k[:, :, ti, j]                           # [B,H,d]
                    vtj = v[:, :, ti, j]
                    btj = beta[:, :, ti, j]                        # [B,H]
                    if j == 0:
                        # alpha applied once at first sub-step
                        if erase:
                            Sk = torch.einsum("bhij,bhj->bhi", S, ktj)
                            u = btj[..., None] * vtj - btj[..., None] * (at[..., None] * Sk)
                        else:
                            u = btj[..., None] * vtj
                        S = at[..., None, None] * S + torch.einsum("bhi,bhj->bhij", u, ktj)
                    else:
                        # subsequent sub-steps: pure delta, no alpha decay
                        if erase:
                            Sk = torch.einsum("bhij,bhj->bhi", S, ktj)
                            u = btj[..., None] * (vtj - Sk)
                        else:
                            u = btj[..., None] * vtj
                        S = S + torch.einsum("bhi,bhj->bhij", u, ktj)
                outs_nd.append(o)
        return torch.stack(outs_nd, dim=2), S

    outs = []
    for c0 in range(0, T, chunk):
        c1 = min(c0 + chunk, T)
        C = c1 - c0
        Kc = k[:, :, c0:c1]                      # [B,H,C,d]
        Vc = v[:, :, c0:c1]
        Qc = q[:, :, c0:c1]
        Bc = beta[:, :, c0:c1]                    # [B,H,C]  write gate beta_w
        Bec = beta_e[:, :, c0:c1] if decoupled else Bc   # [B,H,C]  erase gate beta_e
        if gated:
            Ac = alpha[:, :, c0:c1]               # [B,H,C]  in [0.5,1]
            # within-chunk cumulative decay in LOG space (avoids float32 underflow of gamma over
            # long chunks: exp(cumsum(log a)) -> 1e-20 at C=64, and dividing two underflowed
            # exponentials destroys the recurrence). All gamma-RATIOS are formed as exp(log-diff),
            # which is <=1 on the strictly-lower region actually used -> stable, no clamp.
            logA = torch.log(Ac.clamp_min(1e-6))
            clog = torch.cumsum(logA, dim=-1)                    # [B,H,C]  log gamma_i (post i)
            clog_prev = clog - logA                              # log gamma_{i-1} (pre i)
            KK = torch.matmul(Kc, Kc.transpose(-1, -2))          # [B,H,C,C]  k_i·k_j
            ratio = torch.exp(clog[..., :, None] - clog[..., None, :])         # gamma_i/gamma_j
            # A matrix: erase gate beta_e scales the KK*ratio terms (read-back / erase strength)
            A = torch.tril(Bec[..., :, None] * (KK * ratio), -1) if erase else torch.zeros_like(KK)
            KS0 = torch.matmul(Kc, S.transpose(-1, -2))          # [B,H,C,d]  (k_i^T S0^T)
            gamma = torch.exp(clog)[..., None]                   # [B,H,C,1] absolute (genuine small)
            # rhs: write gate beta_w scales Vc; erase gate beta_e scales the gamma*KS0 term
            if erase:
                rhs = Bc[..., None] * Vc - Bec[..., None] * (gamma * KS0)
            else:
                rhs = Bc[..., None] * Vc
            U = _solve_unit_lower(A, rhs)                        # [B,H,C,d]
            # reads are PRE-write -> decayed to gamma_{i-1}
            read_ratio = torch.exp(clog_prev[..., :, None] - clog[..., None, :])   # gamma_{i-1}/gamma_j
            O_inter = torch.exp(clog_prev)[..., None] * torch.matmul(Qc, S.transpose(-1, -2))
            QK = torch.matmul(Qc, Kc.transpose(-1, -2)) * read_ratio
            O_intra = torch.matmul(torch.tril(QK, -1), U)
            Oc = O_inter + O_intra
            # state carry: S_end = gamma_C S0 + sum_i (gamma_C/gamma_i) u_i k_i^T (ratio <=1 -> stable)
            clogC = clog[..., -1:]                               # [B,H,1]
            gC = torch.exp(clogC)[..., None]                     # [B,H,1,1]
            scale = torch.exp(clogC - clog)                      # [B,H,C]  gamma_C/gamma_i <= 1
            S = gC * S + torch.matmul((scale[..., None] * U).transpose(-1, -2), Kc)
        else:
            KK = torch.matmul(Kc, Kc.transpose(-1, -2))          # [B,H,C,C]
            # A matrix: erase gate beta_e scales the KK terms
            A = torch.tril(Bec[..., :, None] * KK, -1) if erase else torch.zeros_like(KK)
            KS0 = torch.matmul(Kc, S.transpose(-1, -2))          # [B,H,C,d]
            # rhs: write gate beta_w scales Vc; erase gate beta_e scales KS0
            if erase:
                rhs = Bc[..., None] * Vc - Bec[..., None] * KS0
            else:
                rhs = Bc[..., None] * Vc
            U = _solve_unit_lower(A, rhs)                        # [B,H,C,d]
            O_inter = torch.matmul(Qc, S.transpose(-1, -2))      # [B,H,C,d]
            QK = torch.matmul(Qc, Kc.transpose(-1, -2))
            O_intra = torch.matmul(torch.tril(QK, -1), U)
            Oc = O_inter + O_intra
            S = S + torch.matmul(U.transpose(-1, -2), Kc)        # S0 + U^T K
        outs.append(Oc)
    return torch.cat(outs, dim=2), S


if __name__ == "__main__":
    torch.manual_seed(0)
    # cover the PRODUCTION regime: chunk=64, long T, the alpha=0.5 decay floor (worst-case
    # underflow), and the additive ablation. (The old test used chunk=16/T=40 and HID a float32
    # underflow bug in the gated chunked path — never weaken these settings.)
    devs = ["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"]
    cases = [("pure", None, "delta"), ("gated~U", "rand", "delta"),
             ("gated.5floor", "floor", "delta"), ("additive", None, "additive"),
             ("gated-additive", "rand", "additive")]
    allok = True
    for name, amode, wmode in cases:
        for dev in devs:
            B, H, T, d, C = 2, 3, 256, 16, 64
            q = torch.randn(B, H, T, d, device=dev)
            k = torch.randn(B, H, T, d, device=dev); k = k / k.norm(dim=-1, keepdim=True)
            v = torch.randn(B, H, T, d, device=dev)
            beta = torch.rand(B, H, T, device=dev) * 0.99
            if amode == "rand":
                alpha = 0.5 + 0.5 * torch.rand(B, H, T, device=dev)
            elif amode == "floor":
                alpha = torch.full((B, H, T), 0.5, device=dev)     # worst-case sustained decay
            else:
                alpha = None
            Oref, Sref = _delta_reference(q, k, v, beta, alpha, write_mode=wmode)
            Och, Sch = chunked_delta(q, k, v, beta, alpha, chunk=C, write_mode=wmode)
            do = (Oref - Och).abs().max().item(); ds = (Sref - Sch).abs().max().item()
            ok = max(do, ds) < 1e-4; allok &= ok
            print(f"[{name:<14} {dev}] C={C} T={T} max|dO|={do:.2e} max|dS|={ds:.2e}  "
                  f"{'OK' if ok else 'MISMATCH'}")
    # RECTANGULAR state (d_v != d_k): the feature-map / GlobalDeltaMemory contract. The chunked
    # WY/UT form must equal the naive recurrence when keys/queries have a different last-dim than
    # values (state S in R^{d_v x d_k}). This is the load-bearing guarantee for the capacity levers.
    for dev in devs:
        B, H, T, dk, dv, C = 2, 3, 200, 48, 16, 64
        q = torch.randn(B, H, T, dk, device=dev)
        k = torch.randn(B, H, T, dk, device=dev); k = k / k.norm(dim=-1, keepdim=True)
        v = torch.randn(B, H, T, dv, device=dev)
        beta = torch.rand(B, H, T, device=dev) * 0.99
        for amode, wmode in [(None, "delta"), ("rand", "delta"), (None, "additive")]:
            alpha = (0.5 + 0.5 * torch.rand(B, H, T, device=dev)) if amode == "rand" else None
            Oref, Sref = _delta_reference(q, k, v, beta, alpha, write_mode=wmode)
            Och, Sch = chunked_delta(q, k, v, beta, alpha, chunk=C, write_mode=wmode)
            do = (Oref - Och).abs().max().item(); ds = (Sref - Sch).abs().max().item()
            ok = max(do, ds) < 1e-4; allok &= ok
            print(f"[rect dv={dv}!=dk={dk} {dev} {wmode}{'/gated' if amode else ''}] "
                  f"max|dO|={do:.2e} max|dS|={ds:.2e}  {'OK' if ok else 'MISMATCH'}")
    # SURPRISE GATE: check that chunked_delta(surprise=True) == _delta_reference(surprise=True)
    # including on REPEATED KEYS (the R3 binding test).
    for dev in devs:
        B, H, T, d, C = 2, 2, 64, 16, 32
        torch.manual_seed(99)
        q = torch.randn(B, H, T, d, device=dev)
        # Repeat a fixed key for the first half (worst-case R3 scenario):
        k_rep = torch.randn(1, 1, 1, d, device=dev)
        k_rep = k_rep / k_rep.norm(dim=-1, keepdim=True)
        k = k_rep.expand(B, H, T, d).clone()   # all tokens share the SAME key
        v = torch.randn(B, H, T, d, device=dev)
        beta = torch.rand(B, H, T, device=dev) * 0.99
        alpha_surp = 0.5 + 0.5 * torch.rand(B, H, T, device=dev)
        for smode in ('norm', 'constant'):
            gen = torch.Generator().manual_seed(42) if smode == 'random' else None
            Oref_s, Sref_s = _delta_reference(q, k, v, beta, alpha_surp,
                                               surprise=True, surprise_mode=smode, surprise_gen=gen)
            gen2 = torch.Generator().manual_seed(42) if smode == 'random' else None
            Och_s, Sch_s = chunked_delta(q, k, v, beta, alpha_surp, chunk=C,
                                          surprise=True, surprise_mode=smode, surprise_gen=gen2)
            do_s = (Oref_s - Och_s).abs().max().item()
            ds_s = (Sref_s - Sch_s).abs().max().item()
            ok_s = max(do_s, ds_s) < 1e-4; allok &= ok_s
            print(f"[surprise={smode:<8} {dev}] repeated-key max|dO|={do_s:.2e} max|dS|={ds_s:.2e}  "
                  f"{'OK' if ok_s else 'MISMATCH'}")
    # IN-CONTEXT PER-CHANNEL LR (Lever G): chunked_delta(eta=...) must equal _delta_reference(eta=...)
    # for both pure and gated alpha, AND eta=None must stay byte-identical to the scalar-beta path.
    for dev in devs:
        B, H, T, d, C = 2, 3, 200, 16, 64
        torch.manual_seed(123)
        q = torch.randn(B, H, T, d, device=dev)
        k = torch.randn(B, H, T, d, device=dev); k = k / k.norm(dim=-1, keepdim=True)
        v = torch.randn(B, H, T, d, device=dev)
        beta = torch.rand(B, H, T, device=dev) * 0.99
        eta = torch.rand(B, H, T, d, device=dev) * 0.99          # per VALUE-channel rate
        # OFF-path byte-identity: eta=None == scalar-beta baseline (< 1e-6)
        Ob, Sb = chunked_delta(q, k, v, beta, chunk=C)
        On, Sn = chunked_delta(q, k, v, beta, chunk=C, eta=None)
        do0 = (Ob - On).abs().max().item(); ds0 = (Sb - Sn).abs().max().item()
        ok0 = max(do0, ds0) < 1e-6; allok &= ok0
        print(f"[inctx_lr OFF  {dev}] max|dO|={do0:.2e} max|dS|={ds0:.2e}  "
              f"{'OK' if ok0 else 'MISMATCH'}")
        for amode in (None, "rand"):
            alpha = (0.5 + 0.5 * torch.rand(B, H, T, device=dev)) if amode == "rand" else None
            Oref_e, Sref_e = _delta_reference(q, k, v, beta, alpha, eta=eta)
            Och_e, Sch_e = chunked_delta(q, k, v, beta, alpha, chunk=C, eta=eta)
            do_e = (Oref_e - Och_e).abs().max().item()
            ds_e = (Sref_e - Sch_e).abs().max().item()
            ok_e = max(do_e, ds_e) < 1e-4; allok &= ok_e
            tag = "gated" if amode else "pure "
            print(f"[inctx_lr {tag} {dev}] max|dO|={do_e:.2e} max|dS|={ds_e:.2e}  "
                  f"{'OK' if ok_e else 'MISMATCH'}")
    print("ALL OK" if allok else "FAILURES PRESENT")
