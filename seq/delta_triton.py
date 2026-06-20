"""Task — DRAFT hand-written @triton.jit chunked-delta kernel (deferred A100 verification).

This module is the EXPLICITLY-DEFERRED hand-written Triton kernel that `seq/delta_fused.py`
(Lever F) promised in its docstring ("FUTURE WORK (A100): replace the torch.compile path with a
verified hand-written ``@triton.jit`` WY/UT kernel ... gated against `chunked_delta` to < 1e-4
forward+grad ... before adoption").

STATUS: DRAFT — PENDING A100 NUMERICAL VERIFICATION. NOT YET TRUSTED.
--------------------------------------------------------------------
The ``@triton.jit`` kernel below cannot be executed on this machine (CPU/MPS only, no CUDA, Triton
typically not importable). It is therefore a careful, READABLE draft of the chunked delta-rule core
that we will gate against `seq.delta.chunked_delta` ON the A100 (< 1e-3 forward, ~1e-2 grad) BEFORE
it is ever trusted. NO performance claim is made here; the goal is a correct, verifiable kernel that
can later be benchmarked against `torch.compile(chunked_delta)`.

CORRECTNESS GUARANTEE OFF THE TRITON PATH (by construction)
-----------------------------------------------------------
`triton_chunked_delta(...)` mirrors `fused_chunked_delta`'s public contract: it is EXACTLY
`seq.delta.chunked_delta`'s signature plus a trailing ``backend="auto"`` kwarg, returning the same
`(O, S)` tuple. The hand-written Triton path is taken ONLY when ALL of these hold:

    q.is_cuda  AND  triton is importable  AND  (not surprise)  AND  (eta is None)
    AND  (n_delta == 1)  AND  write_mode == "delta"  AND  beta_e is None
    AND  backend in ("auto", "triton")

— i.e. the SAME simple case that `delta_fused.py` accelerates (no surprise, eta=None, n_delta==1,
the default additive-free delta write, no decoupled erase gate). In EVERY other case — off-CUDA,
Triton missing, surprise set, eta given, n_delta>=2, write_mode!="delta", beta_e given, or
backend=="eager" — this returns the EXACT eager ``chunked_delta(...)``, BYTE-IDENTICALLY. Because
the predicate requires `q.is_cuda` AND an importable Triton, the kernel is NEVER reached locally:
the module imports and every local call uses the exact eager fallback.

This module does NOT modify the published default path: `seq/delta.py` is untouched, and the
off-Triton-path output is byte-identical to `chunked_delta(...)` by construction.

THE KERNEL (one program == one (b, h, chunk) tile)
--------------------------------------------------
For the supported simple case the per-chunk WY/UT recurrence (see `seq.delta.chunked_delta`) is, for
chunk c with tokens i in [c0, c1), size C, state S0 = state at chunk entry (shape [d_v, d_k]):

    gamma_i      = prod_{j<=i} alpha_j           (gamma=1 when alpha is None; pure DeltaNet)
    A_{ij}       = (gamma_i / gamma_j)(k_j . k_i) * beta_i   for j < i   (strictly lower, unit-diag)
    rhs_i        = beta_i * v_i - beta_i * gamma_i * (S0 k_i)
    (I + A) U    = rhs                            (unit-lower-triangular solve -> U_i)
    o_i          = gamma_{i-1} * (S0 q_i) + sum_{j<i} (gamma_{i-1}/gamma_j)(q_i . k_j) U_j   (PRE-write)
    S_end        = gamma_C * S0 + sum_i (gamma_C / gamma_i) U_i k_i^T

The triangular solve is done in-kernel by FORWARD SUBSTITUTION (row i depends only on rows j<i),
which is the part a blind kernel most easily gets wrong — it is written explicitly and sequentially
over the C rows so the A100 gate can check it. All accumulation is in float32.

A100 VERIFICATION MUST CHECK (caveats):
  * block sizes: BLOCK_C (chunk rows) and the d_k / d_v tile dims — this draft assumes ONE chunk
    fits one program and C <= BLOCK_C, d_k <= BLOCK_D, d_v <= BLOCK_DV; larger dims need tiling.
  * masking: the strictly-lower (j<i) mask on A and on the QK read-back, plus the row/col bounds
    masks for the C/d ragged tail chunk (T % chunk != 0).
  * fp32 accumulation: gamma is formed in LOG space (cumsum(log alpha)) to avoid float32 underflow
    over a 64-long chunk, exactly mirroring the eager path; ratios are exp(log-diff) and stay <= 1
    on the causal region. Verify no clamp drift vs eager.
  * gradient support: this DRAFT kernel is FORWARD-ONLY (a raw @triton.jit launch carries no
    grad_fn). The public predicate `_should_use_triton` now GATES on grad: any grad-live call (grad
    enabled AND some input requires grad) routes to the differentiable eager `chunked_delta`, so a
    training step is grad-safe (it uses eager). A NATIVE Triton backward (autograd.Function) is
    future work; until it exists, training always takes the eager path.
  * state carry order and the PRE-write read (gamma_{i-1}, not gamma_i) — the most error-prone
    indexing detail; gate it on repeated-key inputs.

POST-REVIEW HARDENING (2026-06-08, 28-agent adversarial review — see
docs/superpowers/specs/triton_kernel_a100_checklist.md). Locally-safe fixes already APPLIED here:
(1) grad-gate in `_should_use_triton` (above); (2) `clogC = tl.min(...)` not `tl.max(...)` — clog is
non-increasing so gamma_C is the LAST/min row (was a state-carry bug on every gated chunk);
(3) `input_precision='ieee'` on all six `tl.dot` (match eager true-fp32, not A100 TF32); (4) alpha
strides splatted (`*(...)`) into the kernel launch (was a tuple-vs-3-scalars arity bug). STILL
DEFERRED TO THE A100 (compile/validate there — see the checklist doc): non-power-of-2 channel-dim
tiling (BLOCK_DK/BLOCK_DV — the real d_phi=128/137 cannot compile as-is), forward+grad numerical
gates (<1e-3 fwd incl. gated/repeated-key/ragged-tail/S0-carry), bf16 scope, unroll register
pressure. NOT TRUSTED / NOT for production use until that checklist passes.
"""
from __future__ import annotations

import torch

from seq.delta import chunked_delta

# --- Optional Triton import: the module MUST import & run with Triton absent (every call then uses
#     the exact eager fallback). When Triton IS importable AND q.is_cuda, the @triton.jit kernel
#     draft below is eligible — but it is still PENDING A100 verification before being trusted. ----
try:  # pragma: no cover - exercised only on a CUDA+Triton box (the A100), never locally
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # ImportError or any backend init failure -> stay on the eager fallback
    triton = None
    tl = None
    _HAS_TRITON = False


# =================================================================================================
# DRAFT @triton.jit kernel — supported simple case ONLY (no surprise, eta=None, n_delta==1,
# write_mode=="delta", beta_e is None). FORWARD-ONLY. PENDING A100 NUMERICAL VERIFICATION.
# =================================================================================================
_chunked_delta_fwd_kernel = None
_chunked_delta_bwd_kernel = None
TritonChunkedDeltaFunction = None

if _HAS_TRITON:  # pragma: no cover - never compiled/executed on this CPU/MPS machine

    @triton.jit
    def _chunked_delta_fwd_kernel(
        Q_ptr, K_ptr, V_ptr, BETA_ptr, ALPHA_ptr, S0_ptr,
        O_ptr, SEND_ptr,
        # strides (in elements) for the [B, H, T, d] / [B, H, d_v, d_k] layouts
        stride_qb, stride_qh, stride_qt, stride_qd,
        stride_kb, stride_kh, stride_kt, stride_kd,
        stride_vb, stride_vh, stride_vt, stride_vd,
        stride_bb, stride_bh, stride_bt,
        stride_ab, stride_ah, stride_at,
        stride_sb, stride_sh, stride_sv, stride_sk,
        stride_ob, stride_oh, stride_ot, stride_od,
        stride_eb, stride_eh, stride_ev, stride_ek,
        T, C, c0,
        DK: tl.constexpr, DV: tl.constexpr,
        BLOCK_DK: tl.constexpr, BLOCK_DV: tl.constexpr, BLOCK_C: tl.constexpr,
        GATED: tl.constexpr,
    ):
        """One program handles ONE (b, h) and ONE chunk [c0, c0+C). Forward only, fp32 accum."""
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        row = tl.arange(0, BLOCK_C)                      # token index within the chunk [0, BLOCK_C)
        dk = tl.arange(0, BLOCK_DK)                      # key/query channel index
        dv = tl.arange(0, BLOCK_DV)                      # value channel index
        rmask = row < C                                  # ragged-tail row mask

        # --- load chunk tensors: Kc/Qc [C, DK], Vc [C, DV], beta [C] -------------------------------
        q_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
        k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
        v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh
        b_base = BETA_ptr + pid_b * stride_bb + pid_h * stride_bh

        tok = c0 + row
        tok_mask = tok < T
        Qc = tl.load(q_base + tok[:, None] * stride_qt + dk[None, :] * stride_qd,
                     mask=tok_mask[:, None] & (dk[None, :] < DK), other=0.0).to(tl.float32)
        Kc = tl.load(k_base + tok[:, None] * stride_kt + dk[None, :] * stride_kd,
                     mask=tok_mask[:, None] & (dk[None, :] < DK), other=0.0).to(tl.float32)
        Vc = tl.load(v_base + tok[:, None] * stride_vt + dv[None, :] * stride_vd,
                     mask=tok_mask[:, None] & (dv[None, :] < DV), other=0.0).to(tl.float32)
        Bc = tl.load(b_base + tok * stride_bt, mask=tok_mask, other=0.0).to(tl.float32)

        # --- cumulative decay in LOG space (mirror eager path; gamma=1 when not gated) -------------
        if GATED:
            a_base = ALPHA_ptr + pid_b * stride_ab + pid_h * stride_ah
            Ac = tl.load(a_base + tok * stride_at, mask=tok_mask, other=1.0).to(tl.float32)
            logA = tl.log(tl.maximum(Ac, 1e-6))
            clog = tl.cumsum(logA, axis=0)               # log gamma_i (post i)  [C]
            clog_prev = clog - logA                      # log gamma_{i-1} (pre i)
        else:
            clog = tl.zeros([BLOCK_C], tl.float32)       # gamma_i = 1
            clog_prev = tl.zeros([BLOCK_C], tl.float32)

        # --- load chunk-entry state S0 [DV, DK] ----------------------------------------------------
        s_base = S0_ptr + pid_b * stride_sb + pid_h * stride_sh
        S0 = tl.load(s_base + dv[:, None] * stride_sv + dk[None, :] * stride_sk,
                     mask=(dv[:, None] < DV) & (dk[None, :] < DK), other=0.0).to(tl.float32)

        # KK[i,j] = k_i . k_j   [C, C]
        KK = tl.dot(Kc, tl.trans(Kc), input_precision='ieee')
        # ratio[i,j] = gamma_i / gamma_j
        ratio = tl.exp(clog[:, None] - clog[None, :])
        lower = (row[:, None] > row[None, :]) & rmask[:, None] & rmask[None, :]   # strictly lower
        # A[i,j] = beta_i * (gamma_i/gamma_j)(k_j . k_i)   for j<i, else 0
        A = tl.where(lower, Bc[:, None] * (KK * ratio), 0.0)

        # KS0[i, :] = (S0 k_i)  ->  [C, DV]   (k_i^T S0^T)
        KS0 = tl.dot(Kc, tl.trans(S0), input_precision='ieee')   # [C, DV]
        gamma = tl.exp(clog)                             # [C] absolute gamma_i
        # rhs_i = beta_i * v_i - beta_i * gamma_i * (S0 k_i)   [C, DV]
        rhs = Bc[:, None] * Vc - Bc[:, None] * (gamma[:, None] * KS0)

        # --- triangular solve (I + A) U = rhs by FORWARD SUBSTITUTION over rows i ------------------
        U = tl.zeros([BLOCK_C, BLOCK_DV], tl.float32)
        for i in range(0, BLOCK_C):
            row_mask = (row == i).to(tl.float32)
            a_row = tl.sum(A * row_mask[:, None], axis=0)
            contrib = tl.sum(a_row[:, None] * U, axis=0)
            rhs_i = tl.sum(rhs * row_mask[:, None], axis=0)
            u_i = rhs_i - contrib
            U = tl.where((row[:, None] == i) & (i < C), u_i[None, :], U)

        # --- PRE-write reads: o_i = gamma_{i-1} (S0 q_i) + sum_{j<i}(gamma_{i-1}/gamma_j)(q_i.k_j)U_j
        gamma_prev = tl.exp(clog_prev)                   # [C]
        O_inter = gamma_prev[:, None] * tl.dot(Qc, tl.trans(S0), input_precision='ieee')   # [C, DV]
        QK = tl.dot(Qc, tl.trans(Kc), input_precision='ieee')   # [C, C]  q_i . k_j
        read_ratio = tl.exp(clog_prev[:, None] - clog[None, :])     # gamma_{i-1}/gamma_j
        QK = tl.where(lower, QK * read_ratio, 0.0)       # strictly lower
        O_intra = tl.dot(QK, U, input_precision='ieee')          # [C, DV]
        Oc = O_inter + O_intra

        o_base = O_ptr + pid_b * stride_ob + pid_h * stride_oh
        tl.store(o_base + tok[:, None] * stride_ot + dv[None, :] * stride_od, Oc,
                 mask=tok_mask[:, None] & (dv[None, :] < DV))

        # --- state carry: S_end = gamma_C * S0 + sum_i (gamma_C/gamma_i) U_i k_i^T  [DV, DK] -------
        clogC = tl.min(tl.where(rmask, clog, 1e30), axis=0)
        gC = tl.exp(clogC)
        scale = tl.where(rmask, tl.exp(clogC - clog), 0.0)         # gamma_C/gamma_i <= 1   [C]
        Uscaled = scale[:, None] * U                              # [C, DV]
        S_end = gC * S0 + tl.dot(tl.trans(Uscaled), Kc, input_precision='ieee')   # [DV, DK]
        se_base = SEND_ptr + pid_b * stride_eb + pid_h * stride_eh
        tl.store(se_base + dv[:, None] * stride_ev + dk[None, :] * stride_ek, S_end,
                 mask=(dv[:, None] < DV) & (dk[None, :] < DK))

    @triton.jit
    def _chunked_delta_bwd_kernel(
        Q_ptr, K_ptr, V_ptr, BETA_ptr, ALPHA_ptr, S0_ptr,
        DO_ptr, DS_ptr,
        DQ_ptr, DK_ptr, DV_ptr, DBETA_ptr, DALPHA_ptr, DS0_ptr,
        # strides (in elements) for the [B, H, T, d] / [B, H, d_v, d_k] layouts
        stride_qb, stride_qh, stride_qt, stride_qd,
        stride_kb, stride_kh, stride_kt, stride_kd,
        stride_vb, stride_vh, stride_vt, stride_vd,
        stride_bb, stride_bh, stride_bt,
        stride_ab, stride_ah, stride_at,
        stride_sb, stride_sh, stride_sv, stride_sk,
        stride_dob, stride_doh, stride_dot, stride_dod,
        stride_dsb, stride_dsh, stride_dsv, stride_dsk,
        stride_dqb, stride_dqh, stride_dqt, stride_dqd,
        stride_dkb, stride_dkh, stride_dkt, stride_dkd,
        stride_dvb, stride_dvh, stride_dvt, stride_dvd,
        stride_dbb, stride_dbh, stride_dbt,
        stride_dab, stride_dah, stride_dat,
        stride_ds0b, stride_ds0h, stride_ds0v, stride_ds0k,
        T, C, c0,
        DK: tl.constexpr, DV: tl.constexpr,
        BLOCK_DK: tl.constexpr, BLOCK_DV: tl.constexpr, BLOCK_C: tl.constexpr,
        GATED: tl.constexpr,
    ):
        """One program handles ONE (b, h) and ONE chunk [c0, c0+C). Backward gradient updates, fp32 accum."""
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        row = tl.arange(0, BLOCK_C)                      # token index within the chunk [0, BLOCK_C)
        dk = tl.arange(0, BLOCK_DK)                      # key/query channel index
        dv = tl.arange(0, BLOCK_DV)                      # value channel index
        rmask = row < C                                  # ragged-tail row mask

        # --- load chunk tensors and entry state ----------------------------------------------------
        q_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
        k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
        v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh
        b_base = BETA_ptr + pid_b * stride_bb + pid_h * stride_bh
        s_base = S0_ptr + pid_b * stride_sb + pid_h * stride_sh

        tok = c0 + row
        tok_mask = tok < T
        Qc = tl.load(q_base + tok[:, None] * stride_qt + dk[None, :] * stride_qd,
                     mask=tok_mask[:, None] & (dk[None, :] < DK), other=0.0).to(tl.float32)
        Kc = tl.load(k_base + tok[:, None] * stride_kt + dk[None, :] * stride_kd,
                     mask=tok_mask[:, None] & (dk[None, :] < DK), other=0.0).to(tl.float32)
        Vc = tl.load(v_base + tok[:, None] * stride_vt + dv[None, :] * stride_vd,
                     mask=tok_mask[:, None] & (dv[None, :] < DV), other=0.0).to(tl.float32)
        Bc = tl.load(b_base + tok * stride_bt, mask=tok_mask, other=0.0).to(tl.float32)
        S0 = tl.load(s_base + dv[:, None] * stride_sv + dk[None, :] * stride_sk,
                     mask=(dv[:, None] < DV) & (dk[None, :] < DK), other=0.0).to(tl.float32)

        # --- load incoming gradients dO [C, DV] and dS_end [DV, DK] --------------------------------
        do_base = DO_ptr + pid_b * stride_dob + pid_h * stride_doh
        ds_base = DS_ptr + pid_b * stride_dsb + pid_h * stride_dsh
        dOc = tl.load(do_base + tok[:, None] * stride_dot + dv[None, :] * stride_dod,
                      mask=tok_mask[:, None] & (dv[None, :] < DV), other=0.0).to(tl.float32)
        dS_end = tl.load(ds_base + dv[:, None] * stride_dsv + dk[None, :] * stride_dsk,
                         mask=(dv[:, None] < DV) & (dk[None, :] < DK), other=0.0).to(tl.float32)

        # --- cumulative decay in LOG space (mirror eager forward) ----------------------------------
        if GATED:
            a_base = ALPHA_ptr + pid_b * stride_ab + pid_h * stride_ah
            Ac = tl.load(a_base + tok * stride_at, mask=tok_mask, other=1.0).to(tl.float32)
            logA = tl.log(tl.maximum(Ac, 1e-6))
            clog = tl.cumsum(logA, axis=0)               # log gamma_i (post i)  [C]
            clog_prev = clog - logA                      # log gamma_{i-1} (pre i)
        else:
            clog = tl.zeros([BLOCK_C], tl.float32)
            clog_prev = tl.zeros([BLOCK_C], tl.float32)

        KK = tl.dot(Kc, tl.trans(Kc), input_precision='ieee')
        ratio = tl.exp(clog[:, None] - clog[None, :])
        lower = (row[:, None] > row[None, :]) & rmask[:, None] & rmask[None, :]
        A = tl.where(lower, Bc[:, None] * (KK * ratio), 0.0)

        KS0 = tl.dot(Kc, tl.trans(S0), input_precision='ieee')
        gamma = tl.exp(clog)
        rhs = Bc[:, None] * Vc - Bc[:, None] * (gamma[:, None] * KS0)

        # --- forward solve (I + A) U = rhs ---------------------------------------------------------
        U = tl.zeros([BLOCK_C, BLOCK_DV], tl.float32)
        for i in range(0, BLOCK_C):
            row_mask = (row == i).to(tl.float32)
            a_row = tl.sum(A * row_mask[:, None], axis=0)
            contrib = tl.sum(a_row[:, None] * U, axis=0)
            rhs_i = tl.sum(rhs * row_mask[:, None], axis=0)
            u_i = rhs_i - contrib
            U = tl.where((row[:, None] == i) & (i < C), u_i[None, :], U)

        # --- recompute Oc for log-alpha gradients --------------------------------------------------
        gamma_prev = tl.exp(clog_prev)
        O_inter = gamma_prev[:, None] * tl.dot(Qc, tl.trans(S0), input_precision='ieee')
        QK = tl.dot(Qc, tl.trans(Kc), input_precision='ieee')
        read_ratio = tl.exp(clog_prev[:, None] - clog[None, :])
        QK_ratio = tl.where(lower, QK * read_ratio, 0.0)
        O_intra = tl.dot(QK_ratio, U, input_precision='ieee')
        Oc = O_inter + O_intra

        # --- Compute dU_direct [C, DV] -------------------------------------------------------------
        clogC = tl.min(tl.where(rmask, clog, 1e30), axis=0)
        scale_C = tl.where(rmask, tl.exp(clogC - clog), 0.0)
        dSe_K = tl.dot(Kc, tl.trans(dS_end), input_precision='ieee')
        
        read_ratio_bwd = tl.exp(clog_prev[:, None] - clog[None, :])
        M = tl.where(row[:, None] > row[None, :], QK * read_ratio_bwd, 0.0)
        dU_direct_o = tl.dot(tl.trans(M), dOc, input_precision='ieee')
        dU_direct = scale_C[:, None] * dSe_K + dU_direct_o

        # --- backward solve (I + A)^T dU = dU_direct -----------------------------------------------
        dU = tl.zeros([BLOCK_C, BLOCK_DV], tl.float32)
        for i in range(BLOCK_C - 1, -1, -1):
            row_mask = (row == i).to(tl.float32)
            a_col = tl.sum(A * row_mask[None, :], axis=1)
            contrib = tl.sum(a_col[:, None] * dU, axis=0)
            dU_direct_i = tl.sum(dU_direct * row_mask[:, None], axis=0)
            du_i = dU_direct_i - contrib
            dU = tl.where((row[:, None] == i) & (i < C), du_i[None, :], dU)

        # --- gradient with respect to V: dVc [C, DV] -----------------------------------------------
        dVc = Bc[:, None] * dU

        # --- gradient with respect to A: dA [C, C] -------------------------------------------------
        dA = tl.where(lower, -tl.dot(dU, tl.trans(U), input_precision='ieee'), 0.0)

        # --- gradient with respect to beta: dBc [C] ------------------------------------------------
        v_minus_gKS0 = Vc - gamma[:, None] * KS0
        dbeta_rhs = tl.sum(dU * v_minus_gKS0, axis=1)
        A_nobeta = KK * ratio
        dbeta_A = tl.sum(dA * A_nobeta, axis=1)
        dBc = dbeta_rhs + dbeta_A

        # --- gradient with respect to Q: dQc [C, DK] -----------------------------------------------
        dO_S0 = tl.dot(dOc, S0, input_precision='ieee')
        dQ_inter = gamma_prev[:, None] * dO_S0
        dO_U = tl.dot(dOc, tl.trans(U), input_precision='ieee')
        P = tl.where(lower, dO_U * read_ratio, 0.0)
        dQ_intra = tl.dot(P, Kc, input_precision='ieee')
        dQc = dQ_inter + dQ_intra

        # --- gradient with respect to K: dKc [C, DK] -----------------------------------------------
        dU_S0 = tl.dot(dU, S0, input_precision='ieee')
        dK_rhs = - (Bc * gamma)[:, None] * dU_S0
        dA_scaled = dA * Bc[:, None] * ratio
        dA_total = dA_scaled + tl.trans(dA_scaled)
        dK_dA = tl.dot(dA_total, Kc, input_precision='ieee')
        Q_coef = tl.where(row[:, None] > row[None, :], dO_U * read_ratio, 0.0)
        dK_dO = tl.dot(tl.trans(Q_coef), Qc, input_precision='ieee')
        dK_dS = scale_C[:, None] * tl.dot(U, dS_end, input_precision='ieee')
        dKc = dK_rhs + dK_dA + dK_dO + dK_dS

        # --- gradient with respect to entry state S0: dS0 [DV, DK] ---------------------------------
        gC = tl.exp(clogC)
        dS0_first = gC * dS_end
        dS0_second = tl.dot(tl.trans(gamma_prev[:, None] * dOc), Qc, input_precision='ieee')
        dS0_third = tl.dot(tl.trans((Bc * gamma)[:, None] * dU), Kc, input_precision='ieee')
        dS0 = dS0_first + dS0_second - dS0_third

        # --- gradient with respect to alpha: dalpha_val [C] ---------------------------------------
        if GATED:
            dclog_A = tl.sum(dA * A, axis=1) - tl.sum(dA * A, axis=0)
            dclog_rhs = - tl.sum(dU * (Bc * gamma)[:, None] * KS0, axis=1)
            
            dclog_o = tl.sum(dOc * Oc, axis=1)
            dclog_o_next = tl.sum(tl.where(row[:, None] == row[None, :] - 1, dclog_o[None, :], 0.0), axis=1)
            
            M_clog = tl.where(row[:, None] > row[None, :], read_ratio_bwd * QK * dO_U, 0.0)
            dclog_dO = - tl.sum(M_clog, axis=0)
            dclog_dS = - tl.sum(dK_dS * Kc, axis=1)
            
            scale = tl.where(rmask, tl.exp(clogC - clog), 0.0)
            Uscaled = scale[:, None] * U
            S_end = gC * S0 + tl.dot(tl.trans(Uscaled), Kc, input_precision='ieee')
            dS_end_term = tl.sum(dS_end * S_end)
            dclog_C_mask = (row == C - 1).to(tl.float32)
            
            d_clog = dclog_A + dclog_rhs + dclog_o_next + dclog_dO + dclog_dS + dS_end_term * dclog_C_mask
            
            d_log_alpha = tl.sum(tl.where(row[:, None] <= row[None, :], d_clog[None, :], 0.0), axis=1)
            dalpha_val = d_log_alpha / tl.maximum(Ac, 1e-6)
        else:
            dalpha_val = tl.zeros([BLOCK_C], tl.float32)

        # --- Store gradients to pointers ----------------------------------------------------------
        dq_base = DQ_ptr + pid_b * stride_dqb + pid_h * stride_dqh
        dk_base = DK_ptr + pid_b * stride_dkb + pid_h * stride_dkh
        dv_base = DV_ptr + pid_b * stride_dvb + pid_h * stride_dvd
        db_base = DBETA_ptr + pid_b * stride_dbb + pid_h * stride_dbh
        ds0_base = DS0_ptr + pid_b * stride_ds0b + pid_h * stride_ds0h

        tl.store(dq_base + tok[:, None] * stride_dqt + dk[None, :] * stride_dqd, dQc,
                 mask=tok_mask[:, None] & (dk[None, :] < DK))
        tl.store(dk_base + tok[:, None] * stride_dkt + dk[None, :] * stride_dkd, dKc,
                 mask=tok_mask[:, None] & (dk[None, :] < DK))
        tl.store(dv_base + tok[:, None] * stride_dvt + dv[None, :] * stride_dvd, dVc,
                 mask=tok_mask[:, None] & (dv[None, :] < DV))
        tl.store(db_base + tok * stride_dbt, dBc, mask=tok_mask)
        tl.store(ds0_base + dv[:, None] * stride_ds0v + dk[None, :] * stride_ds0k, dS0,
                 mask=(dv[:, None] < DV) & (dk[None, :] < DK))

        if GATED:
            da_base = DALPHA_ptr + pid_b * stride_dab + pid_h * stride_dah
            tl.store(da_base + tok * stride_dat, dalpha_val, mask=tok_mask)

    class TritonChunkedDeltaFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, beta, alpha, S0, chunk):
            B, H, T, dk = q.shape
            dv = v.shape[-1]
            gated = alpha is not None
            num_chunks = (T + chunk - 1) // chunk
            
            S_history = torch.empty(num_chunks, B, H, dv, dk, dtype=q.dtype, device=q.device)
            O = torch.empty(B, H, T, dv, dtype=q.dtype, device=q.device)
            if S0 is None:
                S = torch.zeros(B, H, dv, dk, dtype=q.dtype, device=q.device)
            else:
                S = S0
                
            BLOCK_C = triton.next_power_of_2(chunk)
            BLOCK_DK = triton.next_power_of_2(dk)
            BLOCK_DV = triton.next_power_of_2(dv)
            alpha_arg = alpha if gated else torch.empty(0, device=q.device, dtype=q.dtype)
            
            for idx, c0 in enumerate(range(0, T, chunk)):
                C = min(chunk, T - c0)
                S_in = S.contiguous()
                S_history[idx] = S_in
                S_out = torch.empty_like(S_in)
                grid = (B, H)
                _chunked_delta_fwd_kernel[grid](
                    q, k, v, beta, alpha_arg, S_in, O, S_out,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    beta.stride(0), beta.stride(1), beta.stride(2),
                    *((alpha.stride(0), alpha.stride(1), alpha.stride(2)) if gated else (0, 0, 0)),
                    S_in.stride(0), S_in.stride(1), S_in.stride(2), S_in.stride(3),
                    O.stride(0), O.stride(1), O.stride(2), O.stride(3),
                    S_out.stride(0), S_out.stride(1), S_out.stride(2), S_out.stride(3),
                    T, C, c0,
                    DK=dk, DV=dv, BLOCK_DK=BLOCK_DK, BLOCK_DV=BLOCK_DV, BLOCK_C=BLOCK_C, GATED=gated,
                )
                S = S_out
                
            ctx.save_for_backward(q, k, v, beta, alpha, S_history, S0)
            ctx.chunk = chunk
            ctx.gated = gated
            return O, S

        @staticmethod
        def backward(ctx, grad_O, grad_S):
            q, k, v, beta, alpha, S_history, S0 = ctx.saved_tensors
            chunk = ctx.chunk
            gated = ctx.gated
            B, H, T, dk = q.shape
            dv = v.shape[-1]
            
            dq = torch.zeros_like(q)
            dk_t = torch.zeros_like(k)
            dv_t = torch.zeros_like(v)
            dbeta = torch.zeros_like(beta)
            dalpha = torch.zeros_like(alpha) if gated else None
            
            dS = grad_S.contiguous() if grad_S is not None else torch.zeros(B, H, dv, dk, dtype=q.dtype, device=q.device)
            
            BLOCK_C = triton.next_power_of_2(chunk)
            BLOCK_DK = triton.next_power_of_2(dk)
            BLOCK_DV = triton.next_power_of_2(dv)
            
            alpha_arg = alpha if gated else torch.empty(0, device=q.device, dtype=q.dtype)
            dalpha_arg = dalpha if gated else torch.empty(0, device=q.device, dtype=q.dtype)
            
            num_chunks = (T + chunk - 1) // chunk
            for idx in range(num_chunks - 1, -1, -1):
                c0 = idx * chunk
                C = min(chunk, T - c0)
                S_in = S_history[idx].contiguous()
                dS_in = torch.empty_like(dS)
                grid = (B, H)
                
                _chunked_delta_bwd_kernel[grid](
                    q, k, v, beta, alpha_arg, S_in,
                    grad_O, dS,
                    dq, dk_t, dv_t, dbeta, dalpha_arg, dS_in,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    beta.stride(0), beta.stride(1), beta.stride(2),
                    *((alpha.stride(0), alpha.stride(1), alpha.stride(2)) if gated else (0, 0, 0)),
                    S_in.stride(0), S_in.stride(1), S_in.stride(2), S_in.stride(3),
                    grad_O.stride(0), grad_O.stride(1), grad_O.stride(2), grad_O.stride(3),
                    dS.stride(0), dS.stride(1), dS.stride(2), dS.stride(3),
                    dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
                    dk_t.stride(0), dk_t.stride(1), dk_t.stride(2), dk_t.stride(3),
                    dv_t.stride(0), dv_t.stride(1), dv_t.stride(2), dv_t.stride(3),
                    dbeta.stride(0), dbeta.stride(1), dbeta.stride(2),
                    *((dalpha.stride(0), dalpha.stride(1), dalpha.stride(2)) if gated else (0, 0, 0)),
                    dS_in.stride(0), dS_in.stride(1), dS_in.stride(2), dS_in.stride(3),
                    T, C, c0,
                    dk, dv, BLOCK_DK, BLOCK_DV, BLOCK_C, gated,
                )
                dS = dS_in
                
            dS0 = dS if S0 is not None else None
            return dq, dk_t, dv_t, dbeta, dalpha, dS0, None

    def _triton_chunked_delta(q, k, v, beta, alpha, S0, chunk):
        """Run the DRAFT Triton forward kernel chunk-by-chunk, carrying state S across chunks.
        It uses TritonChunkedDeltaFunction to also support backward gradients."""
        return TritonChunkedDeltaFunction.apply(q, k, v, beta, alpha, S0, chunk)
else:
    def _triton_chunked_delta(q, k, v, beta, alpha, S0, chunk):
        raise NotImplementedError("Triton is not available")


def _should_use_triton(q, k, v, beta, alpha, S0, surprise, eta, n_delta, write_mode, beta_e,
                       backend):
    """Predicate (pure, CPU-testable): take the DRAFT Triton forward path ONLY for the supported
    simple case AND only when NO gradient is required.

    The Triton path is a raw ``@triton.jit`` launch (forward-only, NOT a ``torch.autograd.Function``),
    so its outputs carry no ``grad_fn``. A grad-live call MUST therefore route to the differentiable
    eager ``chunked_delta`` — otherwise backward either hard-fails or silently zeroes the gradients of
    q/k/v/beta/alpha/S0 through the whole delta block. ``grad_live`` is true when grad is enabled AND
    at least one input requires grad; in that case this returns False and the caller falls back to the
    exact eager reference."""
    grad_live = torch.is_grad_enabled() and any(
        (t is not None and t.requires_grad) for t in (q, k, v, beta, alpha, S0))
    return (
        _HAS_TRITON
        and q.is_cuda
        and (not grad_live)
        and (not surprise)
        and (eta is None)
        and (n_delta == 1)
        and (write_mode == "delta")
        and (beta_e is None)
        and backend in ("auto", "triton")
    )


def triton_chunked_delta(q, k, v, beta, alpha=None, S0=None, chunk=64, write_mode="delta",
                         beta_e=None, n_delta=1, surprise=False, surprise_mode='norm',
                         surprise_gen=None, eta=None, backend="auto"):
    """Hand-written Triton chunked-delta (DRAFT, pending A100 verification). EXACTLY
    `seq.delta.chunked_delta`'s signature plus a trailing `backend="auto"` kwarg. Returns the same
    `(O, S)` tuple.

    The DRAFT @triton.jit forward kernel is used ONLY when ALL of these hold:
        q.is_cuda  AND  Triton importable  AND  (not surprise)  AND  (eta is None)
        AND  (n_delta == 1)  AND  write_mode == "delta"  AND  beta_e is None
        AND  backend in ("auto", "triton").
    This is the SAME simple case `seq/delta_fused.py` accelerates. In EVERY other case (CPU/MPS,
    Triton missing, surprise, eta given, n_delta>=2, write_mode!="delta", beta_e given, or
    backend=="eager") this returns the EXACT eager `chunked_delta(...)` — byte-identical to the
    SACRED reference. Because the predicate requires q.is_cuda AND an importable Triton, the kernel
    is NEVER reached on this CPU/MPS machine.

    backend: "auto" (Triton on the cuda fast path) | "triton" (same as auto) | "eager" (force the
             exact eager fallback even on cuda). Default "auto"."""
    if backend not in ("auto", "triton", "eager"):
        raise ValueError(
            f"Unknown backend {backend!r}. Choose 'auto', 'triton', or 'eager'.")
    use_triton = _should_use_triton(q, k, v, beta, alpha, S0, surprise, eta, n_delta, write_mode,
                                    beta_e, backend)
    if use_triton:
        # CUDA fast path: DRAFT hand-written @triton.jit kernel. PENDING A100 verification — gated
        # against chunked_delta by tests/test_delta_triton.py's CUDA-only test before it is trusted.
        return _triton_chunked_delta(q, k, v, beta, alpha, S0, chunk)
    # EXACT eager fallback (CPU/MPS, Triton missing, surprise, eta, n_delta>=2, non-delta write,
    # decoupled erase gate, or backend="eager"). Byte-identical to seq.delta.chunked_delta.
    return chunked_delta(q, k, v, beta, alpha=alpha, S0=S0, chunk=chunk, write_mode=write_mode,
                         beta_e=beta_e, n_delta=n_delta, surprise=surprise,
                         surprise_mode=surprise_mode, surprise_gen=surprise_gen, eta=eta)


if __name__ == "__main__":
    # LOCAL fallback equivalence self-test (mirrors seq/delta_fused.py's __main__):
    # on this Mac the Triton fast path is never taken (q.is_cuda False and/or Triton absent), so
    # triton_chunked_delta returns the EXACT eager chunked_delta — assert they match to < 1e-6.
    torch.manual_seed(0)
    devs = ["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"]
    cases = [("pure", None, "delta"), ("gated~U", "rand", "delta"),
             ("gated.5floor", "floor", "delta"), ("additive", None, "additive")]
    allok = True
    print(f"_HAS_TRITON={_HAS_TRITON}")
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
                alpha = torch.full((B, H, T), 0.5, device=dev)
            else:
                alpha = None
            Ot, St = triton_chunked_delta(q, k, v, beta, alpha, chunk=C, write_mode=wmode)
            Oc, Sc = chunked_delta(q, k, v, beta, alpha, chunk=C, write_mode=wmode)
            do = (Ot - Oc).abs().max().item(); ds = (St - Sc).abs().max().item()
            ok = max(do, ds) < 1e-6; allok &= ok
            print(f"[{name:<14} {dev}] C={C} T={T} max|dO|={do:.2e} max|dS|={ds:.2e}  "
                  f"{'OK' if ok else 'MISMATCH'}")
    print("ALL OK" if allok else "FAILURES PRESENT")
