"""Coverage-gap closure for the plain Prizma-Seq delta kernel (campaign 2026-09-11, Lane 2).

The repo's only chunked-delta vs reference parity check for the PLAIN default path lived in the
script-style `__main__` of seq/delta.py (run manually, not collected by pytest). This file pins it:

  1. `chunked_delta == _delta_reference` (< 1e-4) for pure / gated / additive / rectangular at a
     production-ish shape (the WY/UT closed form vs the sequential ground truth);
  2. the C1 solve simplification: `_solve_unit_lower(A, .)` is BIT-IDENTICAL (== 0.0) to the old
     `solve_triangular(torch.eye(C) + A, ..., unitriangular=True)` expression (LAPACK ignores the
     diagonal under unitriangular=True);
  3. the nilpotent-Neumann fallback (used when MPS lacks triangular solve) agrees with the direct
     solve to float32 round-off on the same system.
"""
import torch

from seq.delta import _delta_reference, _solve_unit_lower, chunked_delta


def _mk(T=256, d=16, H=2, B=2, seed=0, dv=None):
    g = torch.Generator().manual_seed(seed)
    dv = d if dv is None else dv
    q = torch.randn(B, H, T, d, generator=g)
    k = torch.randn(B, H, T, d, generator=g)
    k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn(B, H, T, dv, generator=g)
    beta = torch.rand(B, H, T, generator=g) * 0.99
    return q, k, v, beta


def _parity(q, k, v, beta, alpha=None, wmode="delta", C=64):
    Oref, Sref = _delta_reference(q, k, v, beta, alpha=alpha, write_mode=wmode)
    Och, Sch = chunked_delta(q, k, v, beta, alpha=alpha, chunk=C, write_mode=wmode)
    return ((Oref - Och).abs().max().item(), (Sref - Sch).abs().max().item())


def test_default_chunked_matches_reference_pure():
    """Plain default path, alpha=None: chunked WY/UT == sequential reference < 1e-4."""
    q, k, v, beta = _mk(T=256, d=16, H=3, B=2, seed=1)
    dO, dS = _parity(q, k, v, beta, alpha=None)
    print(f"\n[delta pure] max|dO|={dO:.3e} max|dS|={dS:.3e}")
    assert dO < 1e-4, f"pure dO={dO:.3e} >= 1e-4"
    assert dS < 1e-4, f"pure dS={dS:.3e} >= 1e-4"


def test_default_chunked_matches_reference_gated():
    """Plain default path, random alpha in [0.5,1] (the char-LM gated regime): parity < 1e-4."""
    q, k, v, beta = _mk(T=256, d=16, H=3, B=2, seed=2)
    ga = torch.Generator().manual_seed(20)
    alpha = 0.5 + 0.5 * torch.rand(2, 3, 256, generator=ga)
    dO, dS = _parity(q, k, v, beta, alpha=alpha)
    print(f"[delta gated] max|dO|={dO:.3e} max|dS|={dS:.3e}")
    assert dO < 1e-4, f"gated dO={dO:.3e} >= 1e-4"
    assert dS < 1e-4, f"gated dS={dS:.3e} >= 1e-4"


def test_default_chunked_matches_reference_additive():
    """write_mode='additive' (linear-attn ablation): chunked == reference < 1e-4."""
    q, k, v, beta = _mk(T=200, d=16, H=2, B=2, seed=3)
    dO, dS = _parity(q, k, v, beta, alpha=None, wmode="additive")
    print(f"[delta additive] max|dO|={dO:.3e} max|dS|={dS:.3e}")
    assert dO < 1e-4, f"additive dO={dO:.3e} >= 1e-4"
    assert dS < 1e-4, f"additive dS={dS:.3e} >= 1e-4"


def test_default_chunked_matches_reference_rectangular():
    """Rectangular state d_v != d_k (feature-map / GlobalDeltaMemory contract): parity < 1e-4."""
    q, k, v, beta = _mk(T=200, d=24, H=2, B=2, seed=4, dv=12)
    ga = torch.Generator().manual_seed(40)
    alpha = 0.5 + 0.5 * torch.rand(2, 2, 200, generator=ga)
    dO, dS = _parity(q, k, v, beta, alpha=alpha)
    print(f"[delta rect dv=12!=dk=24] max|dO|={dO:.3e} max|dS|={dS:.3e}")
    assert dO < 1e-4, f"rect dO={dO:.3e} >= 1e-4"
    assert dS < 1e-4, f"rect dS={dS:.3e} >= 1e-4"


def test_solve_direct_a_bit_identical_to_eye_plus_add():
    """C1: passing the strictly-lower A DIRECTLY with unitriangular=True must be BIT-IDENTICAL
    (== 0.0) to the old expression torch.eye(C)+A. LAPACK ignores the diagonal under
    unitriangular=True, so the two formulations solve the same system with identical flops."""
    g = torch.Generator().manual_seed(5)
    A = torch.tril(torch.randn(4, 5, 64, 64, generator=g) * 0.1, -1)   # strictly lower
    RHS = torch.randn(4, 5, 64, 16, generator=g)
    X_new = _solve_unit_lower(A, RHS)
    M = torch.eye(64, dtype=A.dtype) + A
    X_old = torch.linalg.solve_triangular(M, RHS, upper=False, unitriangular=True)
    dmax = (X_new - X_old).abs().max().item()
    assert dmax == 0.0, f"direct-A solve not bit-identical: maxdiff={dmax:.3e}"
    assert torch.isfinite(X_new).all()


def test_solve_neumann_fallback_matches_direct_solve(monkeypatch):
    """The nilpotent-Neumann fallback (A strictly lower -> A^C = 0, series terminates exactly) must
    agree with the direct triangular solve to float32 round-off. Force the fallback by making
    solve_triangular raise (the MPS-without-batched-solve scenario)."""
    g = torch.Generator().manual_seed(6)
    A = torch.tril(torch.randn(3, 2, 32, 32, generator=g) * 0.1, -1)
    RHS = torch.randn(3, 2, 32, 8, generator=g)
    X_direct = _solve_unit_lower(A, RHS)

    def boom(*a, **kw):
        raise RuntimeError("solve_triangular unavailable (forced fallback test)")

    monkeypatch.setattr(torch.linalg, "solve_triangular", boom)
    X_fb = _solve_unit_lower(A, RHS)
    dmax = (X_fb - X_direct).abs().max().item()
    assert dmax < 1e-5, f"Neumann fallback diverged from direct solve: maxdiff={dmax:.3e}"
