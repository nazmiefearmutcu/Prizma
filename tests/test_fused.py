"""Tests for Task 1.F: the fused chunked-delta module (seq/delta_fused.py).

Lever F realizes the "fused chunked-delta kernel" via torch.compile(chunked_delta) on CUDA
(TorchInductor emits fused Triton kernels), with a pure `return chunked_delta(...)` fallback on
non-CUDA. By construction the fused path is the SAME math/code, so the equivalence gate cannot
regress — a hand-written @triton.jit kernel is explicitly DEFERRED to be developed and verified ON
the A100 later (no CUDA/Triton importable on this Mac).

The contract of `fused_chunked_delta` is EXACTLY `chunked_delta`'s signature plus a trailing
`backend="auto"` kwarg, returning the same `(O, S)` tuple. The compiled (fused) path is used ONLY
when ALL hold:  q.is_cuda AND not surprise AND eta is None AND n_delta == 1 AND backend in
("auto","compile").  In every other case the call returns the EXACT eager `chunked_delta(...)`.

This file is collectible+green LOCALLY (cpu/+mps): the fast path requires q.is_cuda, so on this Mac
everything returns the eager fallback and the compiled handle is never built.  The CUDA equivalence
(forward + backward) test SKIPS off-GPU and runs on the A100.
"""
import torch
import pytest

from seq.delta import chunked_delta
from seq.delta_fused import fused_chunked_delta


# ---------------------------------------------------------------------------------------------
# helpers
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
# 1) FALLBACK BYTE-IDENTITY (runs locally, cpu + mps): production regime chunk=64, T=256
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dev", _devices())
def test_fallback_byte_identity_pure(dev):
    """pure (alpha=None): fused == chunked_delta < 1e-6."""
    q, k, v, beta = _mk(dev=dev, seed=1)
    Of, Sf = fused_chunked_delta(q, k, v, beta, chunk=64)
    Oc, Sc = chunked_delta(q, k, v, beta, chunk=64)
    assert _maxdiff(Of, Oc) < 1e-6, f"pure dO={_maxdiff(Of, Oc):.2e}"
    assert _maxdiff(Sf, Sc) < 1e-6, f"pure dS={_maxdiff(Sf, Sc):.2e}"


@pytest.mark.parametrize("dev", _devices())
def test_fallback_byte_identity_gated(dev):
    """gated (alpha in [0.5,1] random): fused == chunked_delta < 1e-6."""
    q, k, v, beta = _mk(dev=dev, seed=2)
    g = torch.Generator(device=dev).manual_seed(22)
    alpha = 0.5 + 0.5 * torch.rand(*beta.shape, device=dev, generator=g)
    Of, Sf = fused_chunked_delta(q, k, v, beta, alpha=alpha, chunk=64)
    Oc, Sc = chunked_delta(q, k, v, beta, alpha=alpha, chunk=64)
    assert _maxdiff(Of, Oc) < 1e-6, f"gated dO={_maxdiff(Of, Oc):.2e}"
    assert _maxdiff(Sf, Sc) < 1e-6, f"gated dS={_maxdiff(Sf, Sc):.2e}"


@pytest.mark.parametrize("dev", _devices())
def test_fallback_byte_identity_gated_half_floor(dev):
    """gated .5-floor (alpha=0.5 const, worst-case sustained decay): fused == chunked_delta < 1e-6."""
    q, k, v, beta = _mk(dev=dev, seed=3)
    alpha = torch.full(beta.shape, 0.5, device=dev)
    Of, Sf = fused_chunked_delta(q, k, v, beta, alpha=alpha, chunk=64)
    Oc, Sc = chunked_delta(q, k, v, beta, alpha=alpha, chunk=64)
    assert _maxdiff(Of, Oc) < 1e-6, f"floor dO={_maxdiff(Of, Oc):.2e}"
    assert _maxdiff(Sf, Sc) < 1e-6, f"floor dS={_maxdiff(Sf, Sc):.2e}"


@pytest.mark.parametrize("dev", _devices())
def test_fallback_byte_identity_additive(dev):
    """additive (write_mode='additive', linear-attn ablation): fused == chunked_delta < 1e-6."""
    q, k, v, beta = _mk(dev=dev, seed=4)
    Of, Sf = fused_chunked_delta(q, k, v, beta, chunk=64, write_mode="additive")
    Oc, Sc = chunked_delta(q, k, v, beta, chunk=64, write_mode="additive")
    assert _maxdiff(Of, Oc) < 1e-6, f"additive dO={_maxdiff(Of, Oc):.2e}"
    assert _maxdiff(Sf, Sc) < 1e-6, f"additive dS={_maxdiff(Sf, Sc):.2e}"


@pytest.mark.parametrize("dev", _devices())
def test_fallback_byte_identity_rectangular(dev):
    """RECTANGULAR state (d_v != d_k, e.g. dk=48,dv=16): fused == chunked_delta < 1e-6."""
    q, k, v, beta = _mk(dk=48, dv=16, T=256, dev=dev, seed=5)
    Of, Sf = fused_chunked_delta(q, k, v, beta, chunk=64)
    Oc, Sc = chunked_delta(q, k, v, beta, chunk=64)
    assert _maxdiff(Of, Oc) < 1e-6, f"rect dO={_maxdiff(Of, Oc):.2e}"
    assert _maxdiff(Sf, Sc) < 1e-6, f"rect dS={_maxdiff(Sf, Sc):.2e}"


# ---------------------------------------------------------------------------------------------
# 2) UNSUPPORTED-PATH DELEGATION (runs locally, cpu): surprise / eta / n_delta>=2 -> exact fallback
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("smode", ["norm", "constant", "random"])
def test_delegation_surprise(smode):
    """surprise=True (all three modes) delegates exactly to chunked_delta (< 1e-6).
    'random' threads matched torch.Generator seeds on each side."""
    dev = "cpu"
    q, k, v, beta = _mk(B=2, H=2, T=64, dk=16, dev=dev, seed=7)
    g = torch.Generator(device=dev).manual_seed(11)
    alpha = 0.5 + 0.5 * torch.rand(*beta.shape, device=dev, generator=g)

    gen_f = torch.Generator(device=dev).manual_seed(42) if smode == "random" else None
    gen_c = torch.Generator(device=dev).manual_seed(42) if smode == "random" else None
    Of, Sf = fused_chunked_delta(q, k, v, beta, alpha=alpha, chunk=32,
                                 surprise=True, surprise_mode=smode, surprise_gen=gen_f)
    Oc, Sc = chunked_delta(q, k, v, beta, alpha=alpha, chunk=32,
                           surprise=True, surprise_mode=smode, surprise_gen=gen_c)
    assert _maxdiff(Of, Oc) < 1e-6, f"surprise={smode} dO={_maxdiff(Of, Oc):.2e}"
    assert _maxdiff(Sf, Sc) < 1e-6, f"surprise={smode} dS={_maxdiff(Sf, Sc):.2e}"


def test_delegation_eta():
    """eta given (per-value-channel rate) delegates exactly to chunked_delta (< 1e-6)."""
    dev = "cpu"
    q, k, v, beta = _mk(B=2, H=3, T=200, dk=16, dev=dev, seed=8)
    g = torch.Generator(device=dev).manual_seed(80)
    eta = torch.rand(2, 3, 200, 16, device=dev, generator=g) * 0.99
    Of, Sf = fused_chunked_delta(q, k, v, beta, chunk=64, eta=eta)
    Oc, Sc = chunked_delta(q, k, v, beta, chunk=64, eta=eta)
    assert _maxdiff(Of, Oc) < 1e-6, f"eta dO={_maxdiff(Of, Oc):.2e}"
    assert _maxdiff(Sf, Sc) < 1e-6, f"eta dS={_maxdiff(Sf, Sc):.2e}"


def test_delegation_n_delta_2():
    """n_delta=2 (DeltaProduct, k/v shape [B,H,T,2,d], beta [B,H,T,2]) delegates exactly (< 1e-6)."""
    dev = "cpu"
    B, H, T, d, nd = 2, 2, 128, 16, 2
    g = torch.Generator(device=dev).manual_seed(9)
    q = torch.randn(B, H, T, d, device=dev, generator=g)
    k = torch.randn(B, H, T, nd, d, device=dev, generator=g)
    k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn(B, H, T, nd, d, device=dev, generator=g)
    beta = torch.rand(B, H, T, nd, device=dev, generator=g) * 0.99
    Of, Sf = fused_chunked_delta(q, k, v, beta, chunk=64, n_delta=nd)
    Oc, Sc = chunked_delta(q, k, v, beta, chunk=64, n_delta=nd)
    assert _maxdiff(Of, Oc) < 1e-6, f"n_delta=2 dO={_maxdiff(Of, Oc):.2e}"
    assert _maxdiff(Sf, Sc) < 1e-6, f"n_delta=2 dS={_maxdiff(Sf, Sc):.2e}"


# ---------------------------------------------------------------------------------------------
# 3) DISPATCH LOGIC (runs locally): backend="eager" forces fallback; cpu never triggers compile
# ---------------------------------------------------------------------------------------------
def test_backend_eager_forces_fallback():
    """backend='eager' must equal chunked_delta exactly (forces the eager fallback)."""
    dev = "cpu"
    q, k, v, beta = _mk(dev=dev, seed=12)
    Of, Sf = fused_chunked_delta(q, k, v, beta, chunk=64, backend="eager")
    Oc, Sc = chunked_delta(q, k, v, beta, chunk=64)
    assert _maxdiff(Of, Oc) < 1e-6
    assert _maxdiff(Sf, Sc) < 1e-6


def test_plain_cpu_equals_chunked():
    """A plain cpu call (backend='auto') must equal chunked_delta exactly."""
    dev = "cpu"
    q, k, v, beta = _mk(dev=dev, seed=13)
    Of, Sf = fused_chunked_delta(q, k, v, beta, chunk=64)
    Oc, Sc = chunked_delta(q, k, v, beta, chunk=64)
    assert _maxdiff(Of, Oc) < 1e-6
    assert _maxdiff(Sf, Sc) < 1e-6


def test_compile_not_triggered_on_cpu():
    """A cpu call must NOT build the cached torch.compile handle: the private accessor stays None.
    Belt-and-braces: monkeypatch torch.compile to raise and assert the cpu call still succeeds."""
    import seq.delta_fused as df
    # The module-cached compiled handle must be untouched after any non-cuda call.
    df._reset_compiled_for_test()
    q, k, v, beta = _mk(dev="cpu", seed=14)
    fused_chunked_delta(q, k, v, beta, chunk=64)
    assert df._get_compiled_for_test() is None, \
        "torch.compile was built on a non-cuda call (it must only build on the cuda fast path)"

    # Monkeypatch torch.compile to explode; the cpu fast-path predicate (q.is_cuda False) must
    # never reach it, so the call still succeeds and equals chunked_delta.
    orig = torch.compile

    def _boom(*a, **kw):
        raise RuntimeError("torch.compile must NOT be invoked on a non-cuda call")

    torch.compile = _boom
    try:
        Of, Sf = fused_chunked_delta(q, k, v, beta, chunk=64)
    finally:
        torch.compile = orig
    Oc, Sc = chunked_delta(q, k, v, beta, chunk=64)
    assert _maxdiff(Of, Oc) < 1e-6
    assert _maxdiff(Sf, Sc) < 1e-6
    assert df._get_compiled_for_test() is None


# ---------------------------------------------------------------------------------------------
# 4) CUDA-GATED EQUIVALENCE (skips locally, runs on A100): forward + backward parity < 1e-4
# ---------------------------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused CUDA path needs a GPU")
@pytest.mark.parametrize("gated,dk,dv", [
    (False, 16, 16),   # pure, square
    (True, 16, 16),    # gated alpha, square
    (False, 48, 16),   # rectangular state
])
def test_cuda_forward_and_backward_parity(gated, dk, dv):
    """On CUDA: fused == chunked_delta forward (< 1e-4) AND each input grad matches (< 1e-4)."""
    dev = "cuda"
    B, H, T = 2, 3, 256

    def _build(seed=0):
        g = torch.Generator(device=dev).manual_seed(seed)
        q = torch.randn(B, H, T, dk, device=dev, generator=g, requires_grad=True)
        kk = torch.randn(B, H, T, dk, device=dev, generator=g)
        kk = (kk / kk.norm(dim=-1, keepdim=True)).detach().requires_grad_(True)
        v = torch.randn(B, H, T, dv, device=dev, generator=g, requires_grad=True)
        beta = (torch.rand(B, H, T, device=dev, generator=g) * 0.99).detach().requires_grad_(True)
        alpha = (0.5 + 0.5 * torch.rand(B, H, T, device=dev, generator=g)) if gated else None
        return q, kk, v, beta, alpha

    # Two identical input sets (so grads accumulate independently on each path).
    qf, kf, vf, bf, af = _build(seed=0)
    qc, kc, vc, bc, ac = _build(seed=0)

    Of, Sf = fused_chunked_delta(qf, kf, vf, bf, alpha=af, chunk=64)
    Oc, Sc = chunked_delta(qc, kc, vc, bc, alpha=ac, chunk=64)

    assert _maxdiff(Of, Oc) < 1e-4, f"fwd dO={_maxdiff(Of, Oc):.2e}"
    assert _maxdiff(Sf, Sc) < 1e-4, f"fwd dS={_maxdiff(Sf, Sc):.2e}"

    # Backward: fixed weight W on O so the loss is non-trivial across all output channels.
    torch.manual_seed(123)
    W = torch.randn_like(Of)
    (Of * W).sum().backward()
    (Oc * W).sum().backward()

    for name, tf, tc in [("q", qf, qc), ("k", kf, kc), ("v", vf, vc), ("beta", bf, bc)]:
        assert tf.grad is not None and tc.grad is not None, f"{name} grad missing"
        gd = _maxdiff(tf.grad, tc.grad)
        assert gd < 1e-4, f"grad[{name}] mismatch {gd:.2e}"
