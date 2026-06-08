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
        DK: tl.constexpr, DV: tl.constexpr, BLOCK_C: tl.constexpr,
        GATED: tl.constexpr,
    ):
        """One program handles ONE (b, h) and ONE chunk [c0, c0+C). Forward only, fp32 accum.

        DRAFT — verify on A100. Assumes the whole chunk fits one program: C <= BLOCK_C,
        d_k == DK <= block, d_v == DV <= block. The triangular solve is an explicit forward
        substitution over the C rows (row i needs rows j<i only)."""
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        row = tl.arange(0, BLOCK_C)                      # token index within the chunk [0, BLOCK_C)
        dk = tl.arange(0, DK)                            # key/query channel index
        dv = tl.arange(0, DV)                            # value channel index
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

        # KK[i,j] = k_i . k_j   [C, C]   (ieee: match eager true-fp32 matmul, not A100 TF32)
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
        #     U_i = rhs_i - sum_{j<i} A[i,j] * U_j   (unit diagonal). Sequential over C rows; the
        #     part a blind kernel most easily gets wrong, so it is written explicitly here.
        U = tl.zeros([BLOCK_C, DV], tl.float32)
        for i in range(0, BLOCK_C):
            # coeff_j = A[i, j] for j<i  ->  contribution sum_j coeff_j * U_j
            a_row = tl.where(row < i, tl.sum(tl.where(row[:, None] == i, A, 0.0), axis=0), 0.0)  # [C]
            contrib = tl.sum(a_row[:, None] * U, axis=0)         # [DV]
            rhs_i = tl.sum(tl.where(row[:, None] == i, rhs, 0.0), axis=0)   # [DV] = rhs row i
            u_i = rhs_i - contrib                                # unit diagonal
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
        # log gamma_C is clog of the LAST valid row. clog = cumsum(log alpha) is NON-INCREASING
        # (alpha<=1), so the last valid row == the MIN over valid rows (matches eager clog[...,-1:]).
        clogC = tl.min(tl.where(rmask, clog, 1e30), axis=0)
        gC = tl.exp(clogC)
        scale = tl.where(rmask, tl.exp(clogC - clog), 0.0)         # gamma_C/gamma_i <= 1   [C]
        Uscaled = scale[:, None] * U                              # [C, DV]
        S_end = gC * S0 + tl.dot(tl.trans(Uscaled), Kc, input_precision='ieee')   # [DV, DK]
        se_base = SEND_ptr + pid_b * stride_eb + pid_h * stride_eh
        tl.store(se_base + dv[:, None] * stride_ev + dk[None, :] * stride_ek, S_end,
                 mask=(dv[:, None] < DV) & (dk[None, :] < DK))


def _triton_chunked_delta(q, k, v, beta, alpha, S0, chunk):
    """Run the DRAFT Triton forward kernel chunk-by-chunk, carrying state S across chunks.

    PENDING A100 VERIFICATION. Only called when the supported-simple-case predicate holds AND
    q.is_cuda AND Triton importable. Returns (O, S_end) matching chunked_delta's shapes/semantics
    for the simple case. Forward only — see module docstring caveats."""
    B, H, T, dk = q.shape
    dv = v.shape[-1]
    gated = alpha is not None

    O = torch.empty(B, H, T, dv, dtype=q.dtype, device=q.device)
    if S0 is None:
        S = torch.zeros(B, H, dv, dk, dtype=q.dtype, device=q.device)
    else:
        S = S0
    # one program per (b, h); BLOCK_C must cover the chunk size (next power of two >= chunk)
    BLOCK_C = triton.next_power_of_2(chunk)
    alpha_arg = alpha if gated else torch.empty(0, device=q.device, dtype=q.dtype)

    for c0 in range(0, T, chunk):
        C = min(chunk, T - c0)
        S_in = S.contiguous()
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
            DK=dk, DV=dv, BLOCK_C=BLOCK_C, GATED=gated,
        )
        S = S_out
    return O, S


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
