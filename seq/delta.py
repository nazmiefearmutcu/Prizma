"""
The PRISM-Seq workspace recurrence: a precision-gated delta rule (targeted erase-and-write),
derived as ONE gradient step on PRISM's per-token free energy  F_t(S) = 1/2 ||v_t - S k_t||^2.

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


def _delta_reference(q, k, v, beta, alpha=None, S0=None, write_mode="delta"):
    """Ground-truth sequential recurrence. q,k,v:[B,H,T,d]; beta:[B,H,T]; alpha:[B,H,T] or None.
    write_mode='additive' -> u=beta*v (no erase), the linear-attn ablation."""
    B, H, T, d = q.shape
    dv = v.shape[-1]                     # value-dim-aware init: supports a RECTANGULAR state
    if S0 is None:                       #   S in R^{d_v x d_k} (d_k=d), needed by the feature-map
        S = torch.zeros(B, H, dv, d, dtype=q.dtype, device=q.device)   # and GlobalDeltaMemory levers
    else:                                #   (byte-identical when d_v == d_k, i.e. every existing call)
        S = S0.clone()
    if alpha is None:
        alpha = torch.ones(B, H, T, dtype=q.dtype, device=q.device)
    erase = (write_mode == "delta")
    outs = []
    for t in range(T):
        qt, kt, vt = q[:, :, t], k[:, :, t], v[:, :, t]            # [B,H,d]
        bt, at = beta[:, :, t], alpha[:, :, t]                     # [B,H]
        o = torch.einsum("bhij,bhj->bhi", S, qt)                   # read S_{t-1} q_t  (PRE-write)
        if erase:
            Sk = torch.einsum("bhij,bhj->bhi", S, kt)              # S_{t-1} k_t
            u = bt[..., None] * (vt - at[..., None] * Sk)          # [B,H,d]
        else:
            u = bt[..., None] * vt                                # additive (no erase)
        S = at[..., None, None] * S + torch.einsum("bhi,bhj->bhij", u, kt)
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


def chunked_delta(q, k, v, beta, alpha=None, S0=None, chunk=64, write_mode="delta"):
    """WY/UT chunk-parallel delta rule. Same semantics as _delta_reference, O(T d^2/C + T C d).
    alpha=None -> pure delta (no decay). write_mode='additive' -> linear-attn ablation (no erase:
    u=beta*v, used for B6 PRISM_noDelta). Returns O:[B,H,T,d], S_end:[B,H,d,d]."""
    B, H, T, d = q.shape
    dv = v.shape[-1]                     # value-dim-aware init -> RECTANGULAR state S in R^{d_v x d_k}
    if S0 is None:                       #   (d_k=d). Byte-identical when d_v == d_k (every existing
        S = torch.zeros(B, H, dv, d, dtype=q.dtype, device=q.device)   # call); enables feature-map/GDM
    else:
        S = S0
    gated = alpha is not None
    erase = (write_mode == "delta")
    outs = []
    for c0 in range(0, T, chunk):
        c1 = min(c0 + chunk, T)
        C = c1 - c0
        Kc = k[:, :, c0:c1]                      # [B,H,C,d]
        Vc = v[:, :, c0:c1]
        Qc = q[:, :, c0:c1]
        Bc = beta[:, :, c0:c1]                    # [B,H,C]
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
            A = torch.tril(Bc[..., :, None] * (KK * ratio), -1) if erase else torch.zeros_like(KK)
            KS0 = torch.matmul(Kc, S.transpose(-1, -2))          # [B,H,C,d]  (k_i^T S0^T)
            gamma = torch.exp(clog)[..., None]                   # [B,H,C,1] absolute (genuine small)
            rhs = Bc[..., None] * (Vc - gamma * KS0) if erase else Bc[..., None] * Vc
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
            A = torch.tril(Bc[..., :, None] * KK, -1) if erase else torch.zeros_like(KK)
            KS0 = torch.matmul(Kc, S.transpose(-1, -2))          # [B,H,C,d]
            rhs = Bc[..., None] * (Vc - KS0) if erase else Bc[..., None] * Vc
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
