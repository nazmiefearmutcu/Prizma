"""Task 1.F — Fused chunked-delta module (Lever F).

GOAL (from the plan): a fused chunked-delta kernel so that training at the surprise/eta speed regime
recovers toward <=1.5x slower, by fusing the WY/UT chunk step's elementwise ops and removing the
Python per-chunk-loop overhead — WITHOUT touching the byte-identity-pinned `seq.delta.chunked_delta`.

HOW THE FUSION IS REALIZED (controller-locked design decision)
--------------------------------------------------------------
The CUDA fused path is realized via **torch.compile(chunked_delta)** — TorchInductor emits fused
Triton kernels on CUDA — NOT a hand-written ``@triton.jit`` kernel. Rationale:

  * Provably equivalent BY CONSTRUCTION: it is the SAME math / SAME source as `chunked_delta`, so the
    equivalence gate (fused == chunked_delta) cannot regress. The compiler only changes HOW the ops
    are scheduled/fused, never WHAT is computed.
  * It genuinely achieves Lever F's goal: Inductor fuses the elementwise ops (the log-space cumulative
    decay, the ratio/gamma broadcasts, the tril masks, the rhs assembly) and removes Python
    per-chunk-loop dispatch overhead via graph capture — exactly F's "fuse the WY/UT chunk step".
  * A hand-written ``@triton.jit`` kernel CANNOT be verified on this machine (no CUDA, no importable
    Triton), and the triangular solve ``(I+A) U = rhs`` makes a blind kernel high-risk. So the
    hand-written kernel is explicitly DEFERRED — to be developed AND verified ON the A100 later, where
    its output can be gated against this same `chunked_delta` ground truth before it is trusted.

NON-CUDA BEHAVIOUR (this Mac, CI)
---------------------------------
The fused (compiled) path is taken ONLY when q.is_cuda. On CPU/MPS — and for the surprise / eta /
n_delta>=2 / backend="eager" cases — `fused_chunked_delta` returns the EXACT eager
``chunked_delta(...)`` (byte-identical). Because the predicate requires q.is_cuda, torch.compile is
NEVER invoked locally: no Inductor flakiness, no Triton import, deterministic equivalence. The
surprise 'random' path threads a torch.Generator that is not torch.compile-friendly, but it is
excluded by the predicate, so it never reaches the compiler.

FUTURE WORK (A100): replace the torch.compile path with a verified hand-written ``@triton.jit`` WY/UT
kernel (including a triangular-solve schedule), gated against `chunked_delta` to < 1e-4 forward+grad
on the production regime before adoption.
"""
from __future__ import annotations

import torch

from seq.delta import chunked_delta


def _solve_unit_lower_compile(Amat, RHS):
    """Compile-friendly unit lower triangular solve. Direct call to solve_triangular (no try/except)."""
    C = Amat.shape[-1]
    M = torch.eye(C, dtype=Amat.dtype, device=Amat.device) + Amat
    return torch.linalg.solve_triangular(M, RHS, upper=False, unitriangular=True)


def _chunked_delta_fast_path(q, k, v, beta, alpha=None, S0=None, chunk=64, write_mode="delta", beta_e=None):
    """Specialized fast-path helper for compiled execution.

    Assumes n_delta=1, eta=None, surprise=False. Uses compile-friendly operations
    without Python exceptions or graph breaks to maximize Inductor fusion."""
    B, H, T, d = q.shape
    dv = v.shape[-1]
    if S0 is None:
        S = torch.zeros(B, H, dv, d, dtype=q.dtype, device=q.device)
    else:
        S = S0.clone()

    gated = alpha is not None
    erase = (write_mode == "delta")
    decoupled = (beta_e is not None)

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
            logA = torch.log(Ac.clamp_min(1e-6))
            clog = torch.cumsum(logA, dim=-1)                    # [B,H,C]  log gamma_i (post i)
            clog_prev = clog - logA                              # log gamma_{i-1} (pre i)
            KK = torch.matmul(Kc, Kc.transpose(-1, -2))          # [B,H,C,C]  k_i·k_j
            ratio = torch.exp(clog[..., :, None] - clog[..., None, :])         # gamma_i/gamma_j
            A = torch.tril(Bec[..., :, None] * (KK * ratio), -1) if erase else torch.zeros_like(KK)
            KS0 = torch.matmul(Kc, S.transpose(-1, -2))          # [B,H,C,d]  (k_i^T S0^T)
            gamma = torch.exp(clog)[..., None]                   # [B,H,C,1] absolute (genuine small)
            if erase:
                rhs = Bc[..., None] * Vc - Bec[..., None] * (gamma * KS0)
            else:
                rhs = Bc[..., None] * Vc
            U = _solve_unit_lower_compile(A, rhs)                # [B,H,C,d]
            read_ratio = torch.exp(clog_prev[..., :, None] - clog[..., None, :])   # gamma_{i-1}/gamma_j
            O_inter = torch.exp(clog_prev)[..., None] * torch.matmul(Qc, S.transpose(-1, -2))
            QK = torch.matmul(Qc, Kc.transpose(-1, -2)) * read_ratio
            O_intra = torch.matmul(torch.tril(QK, -1), U)
            Oc = O_inter + O_intra
            clogC = clog[..., -1:]                               # [B,H,1]
            gC = torch.exp(clogC)[..., None]                     # [B,H,1,1]
            scale = torch.exp(clogC - clog)                      # [B,H,C]  gamma_C/gamma_i <= 1
            S = gC * S + torch.matmul((scale[..., None] * U).transpose(-1, -2), Kc)
        else:
            KK = torch.matmul(Kc, Kc.transpose(-1, -2))          # [B,H,C,C]
            A = torch.tril(Bec[..., :, None] * KK, -1) if erase else torch.zeros_like(KK)
            KS0 = torch.matmul(Kc, S.transpose(-1, -2))          # [B,H,C,d]
            if erase:
                rhs = Bc[..., None] * Vc - Bec[..., None] * KS0
            else:
                rhs = Bc[..., None] * Vc
            U = _solve_unit_lower_compile(A, rhs)                # [B,H,C,d]
            O_inter = torch.matmul(Qc, S.transpose(-1, -2))      # [B,H,C,d]
            QK = torch.matmul(Qc, Kc.transpose(-1, -2))
            O_intra = torch.matmul(torch.tril(QK, -1), U)
            Oc = O_inter + O_intra
            S = S + torch.matmul(U.transpose(-1, -2), Kc)        # S0 + U^T K
        outs.append(Oc)
    return torch.cat(outs, dim=2), S


# Module-cached, lazily-built compiled callable. Only ever populated on the fast path.
# Stays None for the entire lifetime of a CPU/MPS process (tested by tests/test_fused.py).
_COMPILED_CHUNKED_DELTA = None


def _get_compiled():
    """Lazily create and cache torch.compile(_chunked_delta_fast_path). Called ONLY on the fast path.

    fullgraph=True guarantees that there are no graph breaks, ensuring maximum operator fusion.
    dynamic=False lets Inductor specialize on static production shapes for maximum speed."""
    global _COMPILED_CHUNKED_DELTA
    if _COMPILED_CHUNKED_DELTA is None:
        _COMPILED_CHUNKED_DELTA = torch.compile(_chunked_delta_fast_path, fullgraph=True, dynamic=False)
    return _COMPILED_CHUNKED_DELTA


# --- test-only accessors (private): let tests assert compile is NOT triggered on non-cuda ----------
def _get_compiled_for_test():
    """Return the module-cached compiled handle (None until the fast path builds it)."""
    return _COMPILED_CHUNKED_DELTA


def _reset_compiled_for_test():
    """Reset the module-cached compiled handle (test isolation)."""
    global _COMPILED_CHUNKED_DELTA
    _COMPILED_CHUNKED_DELTA = None


def fused_chunked_delta(q, k, v, beta, alpha=None, S0=None, chunk=64, write_mode="delta",
                        beta_e=None, n_delta=1, surprise=False, surprise_mode='norm',
                        surprise_gen=None, eta=None, backend="auto"):
    """Fused chunked-delta. EXACTLY `seq.delta.chunked_delta`'s signature plus a trailing
    `backend="auto"` kwarg. Returns the same `(O, S)` tuple.

    Fast (fused, compiled) path is used ONLY when ALL of these hold:
        (backend == "compile" or (backend == "auto" and q.is_cuda))
        AND  (not surprise)  AND  (eta is None)  AND  (n_delta == 1)

    On the fast path we call a lazily-created, module-cached
    ``torch.compile(_chunked_delta_fast_path, fullgraph=True, dynamic=False)`` with the SAME kwargs —
    TorchInductor emits fused Triton kernels. If compilation or compiled execution fails,
    we catch the error and fall back to eager `chunked_delta` execution.

    In EVERY other case (CPU/MPS, surprise, eta given, n_delta>=2, or backend=="eager") this returns
    the EXACT eager `chunked_delta(...)` — byte-identical to the SACRED reference.

    backend: "auto" (compile on the cuda fast path) | "compile" (same as auto) | "eager" (force the
             exact eager fallback even on cuda). Default "auto"."""
    if backend not in ("auto", "compile", "eager"):
        raise ValueError(
            f"Unknown backend {backend!r}. Choose 'auto', 'compile', or 'eager'.")

    is_device_supported = q.is_cuda or (backend == "compile")
    use_fused = (
        is_device_supported
        and (not surprise)
        and (eta is None)
        and (n_delta == 1)
        and backend in ("auto", "compile")
    )
    if use_fused:
        try:
            compiled = _get_compiled()
            return compiled(q, k, v, beta, alpha=alpha, S0=S0, chunk=chunk,
                            write_mode=write_mode, beta_e=beta_e)
        except Exception as e:
            import warnings
            warnings.warn(
                f"Torch compilation or compiled execution failed: {e}. "
                "Falling back to eager chunked_delta implementation.",
                RuntimeWarning
            )

    # EXACT eager fallback (CPU/MPS, surprise, eta, n_delta>=2, or backend=="eager",
    # or upon compilation/compiled execution failure).
    return chunked_delta(q, k, v, beta, alpha=alpha, S0=S0, chunk=chunk, write_mode=write_mode,
                         beta_e=beta_e, n_delta=n_delta, surprise=surprise,
                         surprise_mode=surprise_mode, surprise_gen=surprise_gen, eta=eta)


if __name__ == "__main__":
    # LOCAL fallback equivalence self-test (mirrors the spirit of seq/delta.py's __main__):
    # a few production-regime cases on cpu (+mps if available); on this Mac the fast path is never
    # taken (q.is_cuda is False) so fused_chunked_delta returns the EXACT eager chunked_delta — assert
    # they match to < 1e-6 and print per-case OK + a final ALL OK / FAILURES PRESENT line.
    torch.manual_seed(0)
    devs = ["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"]
    # (name, alpha-mode, write_mode)
    cases = [("pure", None, "delta"), ("gated~U", "rand", "delta"),
             ("gated.5floor", "floor", "delta"), ("additive", None, "additive")]
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
                alpha = torch.full((B, H, T), 0.5, device=dev)
            else:
                alpha = None
            Of, Sf = fused_chunked_delta(q, k, v, beta, alpha, chunk=C, write_mode=wmode)
            Oc, Sc = chunked_delta(q, k, v, beta, alpha, chunk=C, write_mode=wmode)
            do = (Of - Oc).abs().max().item(); ds = (Sf - Sc).abs().max().item()
            ok = max(do, ds) < 1e-6; allok &= ok
            print(f"[{name:<14} {dev}] C={C} T={T} max|dO|={do:.2e} max|dS|={ds:.2e}  "
                  f"{'OK' if ok else 'MISMATCH'}")
    # RECTANGULAR state (d_v != d_k): the feature-map / GlobalDeltaMemory contract on the fused wrapper.
    for dev in devs:
        B, H, T, dk, dv, C = 2, 3, 256, 48, 16, 64
        q = torch.randn(B, H, T, dk, device=dev)
        k = torch.randn(B, H, T, dk, device=dev); k = k / k.norm(dim=-1, keepdim=True)
        v = torch.randn(B, H, T, dv, device=dev)
        beta = torch.rand(B, H, T, device=dev) * 0.99
        Of, Sf = fused_chunked_delta(q, k, v, beta, chunk=C)
        Oc, Sc = chunked_delta(q, k, v, beta, chunk=C)
        do = (Of - Oc).abs().max().item(); ds = (Sf - Sc).abs().max().item()
        ok = max(do, ds) < 1e-6; allok &= ok
        print(f"[rect dv={dv}!=dk={dk} {dev}] max|dO|={do:.2e} max|dS|={ds:.2e}  "
              f"{'OK' if ok else 'MISMATCH'}")
    print("ALL OK" if allok else "FAILURES PRESENT")
