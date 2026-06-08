"""Tests for the DRAFT hand-written Triton chunked-delta module (seq/delta_triton.py).

`seq/delta_triton.py` is the explicitly-DEFERRED hand-written ``@triton.jit`` kernel that
`seq/delta_fused.py` (Lever F) promised. It is PENDING A100 NUMERICAL VERIFICATION and is NOT yet
trusted: the kernel cannot be executed on this Mac (CPU/MPS only, Triton typically not importable).

The contract of `triton_chunked_delta` is EXACTLY `chunked_delta`'s signature plus a trailing
`backend="auto"` kwarg, returning the same `(O, S)` tuple. The DRAFT Triton path is used ONLY when
ALL hold:  q.is_cuda AND Triton importable AND not surprise AND eta is None AND n_delta == 1 AND
write_mode == "delta" AND beta_e is None AND backend in ("auto","triton").  In every other case the
call returns the EXACT eager `chunked_delta(...)` — byte-identical to the SACRED reference.

This file is collectible+green LOCALLY (cpu/+mps): the Triton fast path requires q.is_cuda AND an
importable Triton, so on this Mac everything returns the eager fallback and the kernel is never
reached. The CUDA equivalence test SKIPS off-GPU and runs on the A100, where it gates the kernel's
forward (and, if grad is wired up, backward) against `chunked_delta` before the draft is trusted.
"""
import torch
import pytest

from seq.delta import chunked_delta
from seq.delta_triton import triton_chunked_delta


# ---------------------------------------------------------------------------------------------
# helpers (mirror tests/test_fused.py)
# ---------------------------------------------------------------------------------------------
def _devices():
    devs = ["cpu"]
    if torch.backends.mps.is_available():
        devs.append("mps")
    return devs


def _mk(B=2, H=3, T=256, dk=16, dv=None, dev="cpu", seed=0):
    """Production-regime tensors on `dev`. dv=None -> square state (dv==dk)."""
    dv = dk if dv is None else dv
    g = torch.Generator(device=dev).manual_seed(seed)
    q = torch.randn(B, H, T, dk, device=dev, generator=g)
    k = torch.randn(B, H, T, dk, device=dev, generator=g)
    k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn(B, H, T, dv, device=dev, generator=g)
    beta = torch.rand(B, H, T, device=dev, generator=g) * 0.99
    return q, k, v, beta


def _maxdiff(a, b):
    return (a - b).abs().max().item()


# ---------------------------------------------------------------------------------------------
# 0) MODULE IMPORTS (always runs): the module must import even with Triton absent.
# ---------------------------------------------------------------------------------------------
def test_module_imports():
    """seq.delta_triton imports and exposes triton_chunked_delta even when Triton is absent."""
    import seq.delta_triton as dt
    assert hasattr(dt, "triton_chunked_delta")
    assert hasattr(dt, "_HAS_TRITON")  # the import guard flag exists


# ---------------------------------------------------------------------------------------------
# 1) FALLBACK BYTE-IDENTITY (runs locally, cpu + mps): supported simple case, chunk=64, T=256.
#    On CPU/MPS the Triton path is NOT taken, so triton_chunked_delta == chunked_delta exactly.
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dev", _devices())
def test_fallback_byte_identity_pure(dev):
    """pure (alpha=None) supported case: triton_chunked_delta == chunked_delta, max abs diff == 0."""
    q, k, v, beta = _mk(dev=dev, seed=1)
    Ot, St = triton_chunked_delta(q, k, v, beta, chunk=64)
    Oc, Sc = chunked_delta(q, k, v, beta, chunk=64)
    assert _maxdiff(Ot, Oc) == 0.0, f"pure dO={_maxdiff(Ot, Oc):.2e}"
    assert _maxdiff(St, Sc) == 0.0, f"pure dS={_maxdiff(St, Sc):.2e}"


@pytest.mark.parametrize("dev", _devices())
def test_fallback_byte_identity_gated(dev):
    """gated (alpha in [0.5,1] random): triton_chunked_delta == chunked_delta, max abs diff == 0."""
    q, k, v, beta = _mk(dev=dev, seed=2)
    g = torch.Generator(device=dev).manual_seed(22)
    alpha = 0.5 + 0.5 * torch.rand(*beta.shape, device=dev, generator=g)
    Ot, St = triton_chunked_delta(q, k, v, beta, alpha=alpha, chunk=64)
    Oc, Sc = chunked_delta(q, k, v, beta, alpha=alpha, chunk=64)
    assert _maxdiff(Ot, Oc) == 0.0, f"gated dO={_maxdiff(Ot, Oc):.2e}"
    assert _maxdiff(St, Sc) == 0.0, f"gated dS={_maxdiff(St, Sc):.2e}"


@pytest.mark.parametrize("dev", _devices())
def test_fallback_byte_identity_gated_half_floor(dev):
    """gated .5-floor (alpha=0.5 const, worst-case decay): byte-identical (max abs diff == 0)."""
    q, k, v, beta = _mk(dev=dev, seed=3)
    alpha = torch.full(beta.shape, 0.5, device=dev)
    Ot, St = triton_chunked_delta(q, k, v, beta, alpha=alpha, chunk=64)
    Oc, Sc = chunked_delta(q, k, v, beta, alpha=alpha, chunk=64)
    assert _maxdiff(Ot, Oc) == 0.0, f"floor dO={_maxdiff(Ot, Oc):.2e}"
    assert _maxdiff(St, Sc) == 0.0, f"floor dS={_maxdiff(St, Sc):.2e}"


@pytest.mark.parametrize("dev", _devices())
def test_fallback_byte_identity_rectangular(dev):
    """RECTANGULAR state (d_v != d_k, dk=48,dv=16): byte-identical (max abs diff == 0)."""
    q, k, v, beta = _mk(dk=48, dv=16, T=256, dev=dev, seed=5)
    Ot, St = triton_chunked_delta(q, k, v, beta, chunk=64)
    Oc, Sc = chunked_delta(q, k, v, beta, chunk=64)
    assert _maxdiff(Ot, Oc) == 0.0, f"rect dO={_maxdiff(Ot, Oc):.2e}"
    assert _maxdiff(St, Sc) == 0.0, f"rect dS={_maxdiff(St, Sc):.2e}"


# ---------------------------------------------------------------------------------------------
# 2) UNSUPPORTED-PATH DELEGATION (runs locally, cpu): surprise / eta / n_delta>=2 / additive /
#    decoupled-erase / backend="eager" all route to the exact eager path == chunked_delta.
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("smode", ["norm", "constant", "random"])
def test_delegation_surprise(smode):
    """surprise=True (all three modes) delegates exactly to chunked_delta (max abs diff == 0).
    'random' threads matched torch.Generator seeds on each side."""
    dev = "cpu"
    q, k, v, beta = _mk(B=2, H=2, T=64, dk=16, dev=dev, seed=7)
    g = torch.Generator(device=dev).manual_seed(11)
    alpha = 0.5 + 0.5 * torch.rand(*beta.shape, device=dev, generator=g)

    gen_t = torch.Generator(device=dev).manual_seed(42) if smode == "random" else None
    gen_c = torch.Generator(device=dev).manual_seed(42) if smode == "random" else None
    Ot, St = triton_chunked_delta(q, k, v, beta, alpha=alpha, chunk=32,
                                  surprise=True, surprise_mode=smode, surprise_gen=gen_t)
    Oc, Sc = chunked_delta(q, k, v, beta, alpha=alpha, chunk=32,
                           surprise=True, surprise_mode=smode, surprise_gen=gen_c)
    assert _maxdiff(Ot, Oc) == 0.0, f"surprise={smode} dO={_maxdiff(Ot, Oc):.2e}"
    assert _maxdiff(St, Sc) == 0.0, f"surprise={smode} dS={_maxdiff(St, Sc):.2e}"


def test_delegation_eta():
    """eta given (per-value-channel rate) delegates exactly to chunked_delta (max abs diff == 0)."""
    dev = "cpu"
    q, k, v, beta = _mk(B=2, H=3, T=200, dk=16, dev=dev, seed=8)
    g = torch.Generator(device=dev).manual_seed(80)
    eta = torch.rand(2, 3, 200, 16, device=dev, generator=g) * 0.99
    Ot, St = triton_chunked_delta(q, k, v, beta, chunk=64, eta=eta)
    Oc, Sc = chunked_delta(q, k, v, beta, chunk=64, eta=eta)
    assert _maxdiff(Ot, Oc) == 0.0, f"eta dO={_maxdiff(Ot, Oc):.2e}"
    assert _maxdiff(St, Sc) == 0.0, f"eta dS={_maxdiff(St, Sc):.2e}"


def test_delegation_n_delta_2():
    """n_delta=2 (DeltaProduct) delegates exactly to chunked_delta (max abs diff == 0)."""
    dev = "cpu"
    B, H, T, d, nd = 2, 2, 128, 16, 2
    g = torch.Generator(device=dev).manual_seed(9)
    q = torch.randn(B, H, T, d, device=dev, generator=g)
    k = torch.randn(B, H, T, nd, d, device=dev, generator=g)
    k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn(B, H, T, nd, d, device=dev, generator=g)
    beta = torch.rand(B, H, T, nd, device=dev, generator=g) * 0.99
    Ot, St = triton_chunked_delta(q, k, v, beta, chunk=64, n_delta=nd)
    Oc, Sc = chunked_delta(q, k, v, beta, chunk=64, n_delta=nd)
    assert _maxdiff(Ot, Oc) == 0.0, f"n_delta=2 dO={_maxdiff(Ot, Oc):.2e}"
    assert _maxdiff(St, Sc) == 0.0, f"n_delta=2 dS={_maxdiff(St, Sc):.2e}"


def test_delegation_additive():
    """write_mode='additive' (linear-attn ablation, NOT the supported simple case) delegates
    exactly to chunked_delta (max abs diff == 0)."""
    dev = "cpu"
    q, k, v, beta = _mk(dev=dev, seed=10)
    Ot, St = triton_chunked_delta(q, k, v, beta, chunk=64, write_mode="additive")
    Oc, Sc = chunked_delta(q, k, v, beta, chunk=64, write_mode="additive")
    assert _maxdiff(Ot, Oc) == 0.0, f"additive dO={_maxdiff(Ot, Oc):.2e}"
    assert _maxdiff(St, Sc) == 0.0, f"additive dS={_maxdiff(St, Sc):.2e}"


def test_delegation_decoupled_beta_e():
    """beta_e given (decoupled GDN-2 erase gate, NOT the supported simple case) delegates exactly
    to chunked_delta (max abs diff == 0)."""
    dev = "cpu"
    q, k, v, beta = _mk(dev=dev, seed=11)
    g = torch.Generator(device=dev).manual_seed(110)
    beta_e = torch.rand(*beta.shape, device=dev, generator=g) * 0.99
    Ot, St = triton_chunked_delta(q, k, v, beta, chunk=64, beta_e=beta_e)
    Oc, Sc = chunked_delta(q, k, v, beta, chunk=64, beta_e=beta_e)
    assert _maxdiff(Ot, Oc) == 0.0, f"beta_e dO={_maxdiff(Ot, Oc):.2e}"
    assert _maxdiff(St, Sc) == 0.0, f"beta_e dS={_maxdiff(St, Sc):.2e}"


# ---------------------------------------------------------------------------------------------
# 3) DISPATCH LOGIC (runs locally): backend="eager" forces fallback; unknown backend raises.
# ---------------------------------------------------------------------------------------------
def test_backend_eager_forces_fallback():
    """backend='eager' must equal chunked_delta exactly (forces the eager fallback)."""
    dev = "cpu"
    q, k, v, beta = _mk(dev=dev, seed=12)
    Ot, St = triton_chunked_delta(q, k, v, beta, chunk=64, backend="eager")
    Oc, Sc = chunked_delta(q, k, v, beta, chunk=64)
    assert _maxdiff(Ot, Oc) == 0.0
    assert _maxdiff(St, Sc) == 0.0


def test_plain_cpu_equals_chunked():
    """A plain cpu call (backend='auto') must equal chunked_delta exactly (Triton path not taken)."""
    dev = "cpu"
    q, k, v, beta = _mk(dev=dev, seed=13)
    Ot, St = triton_chunked_delta(q, k, v, beta, chunk=64)
    Oc, Sc = chunked_delta(q, k, v, beta, chunk=64)
    assert _maxdiff(Ot, Oc) == 0.0
    assert _maxdiff(St, Sc) == 0.0


def test_unknown_backend_raises():
    """An unknown backend value raises ValueError (a caller typo must NOT silently downgrade to
    the eager fallback)."""
    q, k, v, beta = _mk(dev="cpu", seed=15)
    with pytest.raises(ValueError):
        triton_chunked_delta(q, k, v, beta, chunk=64, backend="compile")


# ---------------------------------------------------------------------------------------------
# 4) CUDA-GATED EQUIVALENCE (skips locally, runs on the A100): forward parity < 1e-3
#    (and gradient parity < 1e-2 if/when the DRAFT kernel grows a backward pass).
# ---------------------------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="triton kernel needs CUDA")
@pytest.mark.parametrize("gated,dk,dv", [
    (False, 16, 16),   # pure, square
    (True, 16, 16),    # gated alpha, square
    (False, 48, 16),   # rectangular state (d_v != d_k)
])
def test_cuda_forward_parity(gated, dk, dv):
    """On CUDA: the DRAFT Triton kernel forward ~= chunked_delta (< 1e-3). This is the gate that
    must pass on the A100 before the hand-written kernel is trusted. Covers the supported simple
    case the kernel actually handles: pure/gated, square/rectangular, delta write, no decoupled
    erase gate, no surprise/eta, n_delta==1."""
    dev = "cuda"
    B, H, T = 2, 3, 256
    g = torch.Generator(device=dev).manual_seed(0)
    q = torch.randn(B, H, T, dk, device=dev, generator=g)
    k = torch.randn(B, H, T, dk, device=dev, generator=g)
    k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn(B, H, T, dv, device=dev, generator=g)
    beta = torch.rand(B, H, T, device=dev, generator=g) * 0.99
    alpha = (0.5 + 0.5 * torch.rand(B, H, T, device=dev, generator=g)) if gated else None

    Ot, St = triton_chunked_delta(q, k, v, beta, alpha=alpha, chunk=64)
    Oc, Sc = chunked_delta(q, k, v, beta, alpha=alpha, chunk=64)
    assert _maxdiff(Ot, Oc) < 1e-3, f"fwd dO={_maxdiff(Ot, Oc):.2e}"
    assert _maxdiff(St, Sc) < 1e-3, f"fwd dS={_maxdiff(St, Sc):.2e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="triton kernel needs CUDA")
def test_cuda_repeated_key_forward_parity():
    """On CUDA: forward parity on REPEATED KEYS (the most error-prone state-carry / pre-write
    indexing case) < 1e-3. Gates the gamma_{i-1} pre-write read and the inter-chunk state carry."""
    dev = "cuda"
    B, H, T, d = 2, 2, 192, 16
    g = torch.Generator(device=dev).manual_seed(99)
    q = torch.randn(B, H, T, d, device=dev, generator=g)
    k_rep = torch.randn(1, 1, 1, d, device=dev, generator=g)
    k_rep = k_rep / k_rep.norm(dim=-1, keepdim=True)
    k = k_rep.expand(B, H, T, d).clone()
    v = torch.randn(B, H, T, d, device=dev, generator=g)
    beta = torch.rand(B, H, T, device=dev, generator=g) * 0.99
    alpha = 0.5 + 0.5 * torch.rand(B, H, T, device=dev, generator=g)

    Ot, St = triton_chunked_delta(q, k, v, beta, alpha=alpha, chunk=64)
    Oc, Sc = chunked_delta(q, k, v, beta, alpha=alpha, chunk=64)
    assert _maxdiff(Ot, Oc) < 1e-3, f"repeated-key dO={_maxdiff(Ot, Oc):.2e}"
    assert _maxdiff(St, Sc) < 1e-3, f"repeated-key dS={_maxdiff(St, Sc):.2e}"


# ---------------------------------------------------------------------------------------------
# 5) HARDENING REGRESSION TESTS (post 28-agent adversarial review, all CPU-runnable).
#    See docs/superpowers/specs/triton_kernel_a100_checklist.md. These pin the locally-provable
#    invariants the review surfaced: the grad-gate predicate, the explicit-S0 carry, non-pow2
#    chunks, and the eager-fallback's differentiability (training grad-safety).
# ---------------------------------------------------------------------------------------------
def test_should_use_triton_grad_gate():
    """The Triton path MUST be skipped whenever a gradient is required: the raw @triton.jit launch
    carries no grad_fn, so a grad-live call has to route to the differentiable eager chunked_delta.
    Checked on the pure predicate with _HAS_TRITON forced True and cuda-like mock tensors, so it is
    provable without CUDA (closes review findings grad-1/grad-2)."""
    import seq.delta_triton as dt
    from types import SimpleNamespace

    orig = dt._HAS_TRITON
    dt._HAS_TRITON = True
    try:
        def mk(rg):
            return SimpleNamespace(is_cuda=True, requires_grad=rg)
        base = dict(surprise=False, eta=None, n_delta=1, write_mode="delta", beta_e=None,
                    backend="auto")
        q, k, v, beta = mk(False), mk(False), mk(False), mk(False)
        # cuda-like + Triton present + simple case + NO input needs grad -> Triton eligible
        assert dt._should_use_triton(q, k, v, beta, None, None, **base) is True
        # an input requires grad while grad is globally enabled -> MUST fall back to eager
        qg = mk(True)
        assert dt._should_use_triton(qg, k, v, beta, None, None, **base) is False
        # same grad-requiring input but grad globally DISABLED (inference) -> eligible again
        with torch.no_grad():
            assert dt._should_use_triton(qg, k, v, beta, None, None, **base) is True
        # a grad-requiring alpha/S0 also trips the gate
        assert dt._should_use_triton(q, k, v, beta, mk(True), None, **base) is False
        assert dt._should_use_triton(q, k, v, beta, None, mk(True), **base) is False
    finally:
        dt._HAS_TRITON = orig


@pytest.mark.parametrize("dev", _devices())
def test_fallback_byte_identity_S0(dev):
    """explicit non-None S0 chunk-entry state (pure + gated) carries byte-identically through the
    eager fallback (closes review finding test-1 — with S0=None all three S0-multiply terms vanish,
    so a carry-read regression would otherwise pass unseen)."""
    q, k, v, beta = _mk(dev=dev, seed=17)
    gd = torch.Generator(device=dev).manual_seed(170)
    B, H, T, dk = q.shape
    dv = v.shape[-1]
    S0 = torch.randn(B, H, dv, dk, device=dev, generator=gd) * 0.1
    alphas = [None, 0.5 + 0.5 * torch.rand(*beta.shape, device=dev, generator=gd)]
    for alpha in alphas:
        Ot, St = triton_chunked_delta(q, k, v, beta, alpha=alpha, S0=S0, chunk=64)
        Oc, Sc = chunked_delta(q, k, v, beta, alpha=alpha, S0=S0, chunk=64)
        tag = "pure" if alpha is None else "gated"
        assert _maxdiff(Ot, Oc) == 0.0, f"S0 {tag} dO={_maxdiff(Ot, Oc):.2e}"
        assert _maxdiff(St, Sc) == 0.0, f"S0 {tag} dS={_maxdiff(St, Sc):.2e}"


@pytest.mark.parametrize("chunk", [32, 48, 64, 96])
def test_fallback_byte_identity_chunks(chunk):
    """non-power-of-2 chunk values with T not a multiple of chunk forward+fall-back byte-identically
    (closes review finding test-3 — on the A100 these stress the ragged C<BLOCK_C masking; locally
    they route to the exact eager path). T=200 is ragged for chunk in {32,48,96}."""
    dev = "cpu"
    q, k, v, beta = _mk(T=200, dev=dev, seed=19)
    Ot, St = triton_chunked_delta(q, k, v, beta, chunk=chunk)
    Oc, Sc = chunked_delta(q, k, v, beta, chunk=chunk)
    assert _maxdiff(Ot, Oc) == 0.0, f"chunk={chunk} dO={_maxdiff(Ot, Oc):.2e}"
    assert _maxdiff(St, Sc) == 0.0, f"chunk={chunk} dS={_maxdiff(St, Sc):.2e}"


def test_fallback_is_differentiable_cpu():
    """With requires_grad inputs the (eager-fallback) output is autograd-connected and backward
    populates finite grads — i.e. a training step is grad-safe via the eager path, which is exactly
    what the grad-gate guarantees the Triton path would otherwise break (grad-1)."""
    dev = "cpu"
    q, k, v, beta = _mk(B=1, H=2, T=64, dk=16, dev=dev, seed=21)
    for t in (q, k, v, beta):
        t.requires_grad_(True)
    Ot, _St = triton_chunked_delta(q, k, v, beta, chunk=32)
    assert Ot.grad_fn is not None, "eager-fallback output must stay in the autograd graph"
    Ot.sum().backward()
    for name, t in [("q", q), ("k", k), ("v", v), ("beta", beta)]:
        assert t.grad is not None and torch.isfinite(t.grad).all(), f"{name}.grad missing/non-finite"
