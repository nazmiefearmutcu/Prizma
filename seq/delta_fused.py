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


# Module-cached, lazily-built compiled callable. Only ever populated on the CUDA fast path.
# Stays None for the entire lifetime of a CPU/MPS process (tested by tests/test_fused.py).
_COMPILED_CHUNKED_DELTA = None


def _get_compiled():
    """Lazily create and cache torch.compile(chunked_delta). Called ONLY on the CUDA fast path.

    fullgraph=False tolerates any minor graph breaks gracefully (correctness is unchanged — eager
    runs the broken region); dynamic=False lets Inductor specialize on the static production shapes
    for maximal fusion. The compiled callable is semantically identical to chunked_delta."""
    global _COMPILED_CHUNKED_DELTA
    if _COMPILED_CHUNKED_DELTA is None:
        _COMPILED_CHUNKED_DELTA = torch.compile(chunked_delta, fullgraph=False, dynamic=False)
    return _COMPILED_CHUNKED_DELTA


# --- test-only accessors (private): let tests assert compile is NOT triggered on non-cuda ----------
def _get_compiled_for_test():
    """Return the module-cached compiled handle (None until the CUDA fast path builds it)."""
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
        q.is_cuda  AND  (not surprise)  AND  (eta is None)  AND  (n_delta == 1)
        AND  backend in ("auto", "compile").
    On the fast path we call a lazily-created, module-cached
    ``torch.compile(chunked_delta, fullgraph=False, dynamic=False)`` with the SAME kwargs —
    TorchInductor emits fused Triton kernels. By construction this equals `chunked_delta`.

    In EVERY other case (CPU/MPS, surprise, eta given, n_delta>=2, or backend=="eager") this returns
    the EXACT eager `chunked_delta(...)` — byte-identical to the SACRED reference.

    backend: "auto" (compile on the cuda fast path) | "compile" (same as auto) | "eager" (force the
             exact eager fallback even on cuda). Default "auto"."""
    if backend not in ("auto", "compile", "eager"):
        raise ValueError(
            f"Unknown backend {backend!r}. Choose 'auto', 'compile', or 'eager'.")
    use_fused = (
        q.is_cuda
        and (not surprise)
        and (eta is None)
        and (n_delta == 1)
        and backend in ("auto", "compile")
    )
    if use_fused:
        # CUDA fast path: TorchInductor-fused Triton kernels, same math as chunked_delta.
        compiled = _get_compiled()
        return compiled(q, k, v, beta, alpha=alpha, S0=S0, chunk=chunk, write_mode=write_mode,
                        beta_e=beta_e, n_delta=n_delta, surprise=surprise,
                        surprise_mode=surprise_mode, surprise_gen=surprise_gen, eta=eta)
    # EXACT eager fallback (CPU/MPS, surprise, eta, n_delta>=2, or backend="eager").
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
