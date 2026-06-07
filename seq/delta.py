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
"""
from __future__ import annotations

import torch


def _delta_reference(q, k, v, beta, alpha=None, S0=None, write_mode="delta", beta_e=None,
                     n_delta=1):
    """Ground-truth sequential recurrence. q,k,v:[B,H,T,d]; beta:[B,H,T]; alpha:[B,H,T] or None.
    write_mode='additive' -> u=beta*v (no erase), the linear-attn ablation.
    beta_e: optional erase gate [B,H,T]; if None, beta_e=beta (byte-identical to old behaviour).
    When beta_e is provided, u_t = beta_w*v_t - beta_e*(alpha*S k_t)  (decoupled GDN-2 write).
    n_delta: int >= 1 (DeltaProduct). For n_delta=1, inputs have no extra k-axis (byte-identical to
    old behaviour). For n_delta>=2, k,v have shape [B,H,T,n_delta,d] and beta [B,H,T,n_delta]:
    each token applies n_delta sequential delta sub-steps before reading the next token. The read
    o_t = S_{t-1} q_t uses the state from the END of the previous token's n_delta sub-steps. Alpha
    decay is applied exactly once per token (on sub-step 0 only); subsequent sub-steps are pure
    additive delta (no additional decay), consistent with Householder-product semantics."""
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
                Sk = torch.einsum("bhij,bhj->bhi", S, kt)
                u = bt[..., None] * vt - bet[..., None] * (at[..., None] * Sk)
            else:
                u = bt[..., None] * vt
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
                  n_delta=1):
    """WY/UT chunk-parallel delta rule. Same semantics as _delta_reference, O(T d^2/C + T C d).
    alpha=None -> pure delta (no decay). write_mode='additive' -> linear-attn ablation (no erase:
    u=beta*v, used for B6 PRIZMA_noDelta). Returns O:[B,H,T,d], S_end:[B,H,d,d].
    beta_e: optional erase gate [B,H,T]; if None, beta_e=beta (byte-identical to old behaviour).
    When beta_e is provided, the A-matrix (erase/read-back) is scaled by beta_e while the Vc
    (write) term is scaled by beta_w=beta — decoupled GDN-2 update.
    n_delta: int >= 1 (DeltaProduct). For n_delta=1, the standard WY/UT chunk-parallel path is
    used (byte-identical to old behaviour). For n_delta>=2, k,v have shape [B,H,T,n_delta,d] and
    beta [B,H,T,n_delta]. The k-axis is processed sequentially within each chunk (correctness-first
    fallback as documented in the plan) while T remains chunked. Alpha applied once per token."""
    B, H, T, d = q.shape
    dv = v.shape[-1]                     # value-dim-aware init -> RECTANGULAR state S in R^{d_v x d_k}
    if S0 is None:                       #   (d_k=d). Byte-identical when d_v == d_k (every existing
        S = torch.zeros(B, H, dv, d, dtype=q.dtype, device=q.device)   # call); enables feature-map/GDM
    else:
        S = S0
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
    print("ALL OK" if allok else "FAILURES PRESENT")
