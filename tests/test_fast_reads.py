"""Tests for the opt-in `fast_reads` CPU fast path in `chunked_delta` (campaign 2026-09-11, Lane 2).

`fast_reads=True` derives the PRE-write read ratio gamma_{i-1}/gamma_j as (gamma_i/gamma_j)/alpha_i
from the already-materialised ratio matrix instead of a second full [B,H,C,C] sub+exp
(gamma_i = alpha_i * gamma_{i-1} exactly; alpha in [0.5,1] on the production config). In float32 it
drifts from the exp form by ~1e-6..1e-5 — well inside the repo's 1e-4 parity bar, but NOT
byte-identical. Hence the lever is opt-in and this file pins the contract:

  1. DEFAULT PATH: omitting the kwarg and passing `fast_reads=False` are byte-identical (== 0.0) on
     every branch (pure / gated / additive / rectangular) — the new kwarg is genuinely OFF.
  2. FAST vs DEFAULT on the production shape (B=2,H=3,T=256,d=16,C=64, alpha in [0.5,1]):
     max|dO| < 1e-5, max|dS| < 1e-6, gradient parity < 1e-4 (absolute, on a mean-squared loss; the
     scale-free relative drift is also asserted < 1e-4 — measured ~1e-6).
  3. INERT WHERE INAPPLICABLE: with alpha=None (pure) the gated read_ratio branch is never reached,
     so fast_reads=True must be EXACTLY equal to fast_reads=False (== 0.0).
"""
import torch

from seq.delta import chunked_delta


def _mk(B=2, H=3, T=256, d=16, seed=1234, gated=True, dv=None):
    g = torch.Generator().manual_seed(seed)
    dv = d if dv is None else dv
    q = torch.randn(B, H, T, d, generator=g)
    k = torch.randn(B, H, T, d, generator=g)
    k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn(B, H, T, dv, generator=g)
    beta = torch.rand(B, H, T, generator=g) * 0.99
    alpha = (0.5 + 0.5 * torch.rand(B, H, T, generator=g)) if gated else None
    return q, k, v, beta, alpha


def test_fast_reads_forward_parity_production_shape():
    """Production-shape gated parity: fast vs default max|dO| < 1e-5 and max|dS| < 1e-6."""
    B, H, T, d, C = 2, 3, 256, 16, 64
    q, k, v, beta, alpha = _mk(B=B, H=H, T=T, d=d)
    O_def, S_def = chunked_delta(q, k, v, beta, alpha=alpha, chunk=C)
    O_fast, S_fast = chunked_delta(q, k, v, beta, alpha=alpha, chunk=C, fast_reads=True)
    dO = (O_def - O_fast).abs().max().item()
    dS = (S_def - S_fast).abs().max().item()
    print(f"\n[fast_reads fwd] max|dO|={dO:.3e} max|dS|={dS:.3e}")
    assert dO < 1e-5, f"fast_reads dO={dO:.3e} >= 1e-5"
    assert dS < 1e-6, f"fast_reads dS={dS:.3e} >= 1e-6"


def test_fast_reads_grad_parity():
    """Gradient parity (fast vs default) < 1e-4. The loss is deliberately the MEAN squared output
    (not a sum): gradient magnitudes then sit at O(1) and an ABSOLUTE 1e-4 bar is meaningful (with a
    sum-of-squares loss the same kernel-level drift shows up as ~3e-4 absolute purely because the
    grads scale ~1e3 — the repo's own tests/test_inctx_lr.py::_grad_close documents that trap). The
    scale-free RELATIVE drift is asserted too (measured ~1e-6)."""
    B, H, T, d, C = 2, 3, 256, 16, 64
    q, k, v, beta, alpha = _mk(B=B, H=H, T=T, d=d)

    def run(fast):
        qq = q.clone().requires_grad_(True)
        kk = k.clone().requires_grad_(True)
        vv = v.clone().requires_grad_(True)
        bb = beta.clone().requires_grad_(True)
        aa = alpha.clone().requires_grad_(True)
        O, S = chunked_delta(qq, kk, vv, bb, alpha=aa, chunk=C, fast_reads=fast)
        (O.square().mean() + S.square().mean()).backward()
        return O.detach(), S.detach(), qq.grad, kk.grad, vv.grad, bb.grad, aa.grad

    Oa, Sa, gqa, gka, gva, gba, gaa = run(False)
    Ob, Sb, gqb, gkb, gvb, gbb, gab = run(True)
    assert (Oa - Ob).abs().max().item() < 1e-5   # same forward bar as above
    assert (Sa - Sb).abs().max().item() < 1e-6
    for nm, ga, gb in [("q", gqa, gqb), ("k", gka, gkb), ("v", gva, gvb),
                       ("beta", gba, gbb), ("alpha", gaa, gab)]:
        dmax = (ga - gb).abs().max().item()
        scale = max(ga.abs().max().item(), gb.abs().max().item(), 1e-12)
        rel = dmax / scale
        print(f"[fast_reads grad {nm}] abs={dmax:.3e} rel={rel:.3e} scale={scale:.3e}")
        assert dmax < 1e-4, f"fast_reads grad[{nm}] abs={dmax:.3e} >= 1e-4"
        assert rel < 1e-4, f"fast_reads grad[{nm}] rel={rel:.3e} >= 1e-4"


def test_fast_reads_default_off_byte_identity():
    """DEFAULT PATH UNTOUCHED: omitting the kwarg == explicit `fast_reads=False` must be EXACTLY
    0.0 on every branch (pure, gated, additive, gated additive, rectangular d_v != d_k). This pins
    that the new kwarg is genuinely default-OFF — not silently enabling the fast path."""
    cfgs = [
        dict(T=256, d=16, H=3, B=2, seed=11, alpha=False, wmode="delta"),
        dict(T=256, d=16, H=3, B=2, seed=12, alpha=True, wmode="delta"),
        dict(T=200, d=16, H=2, B=2, seed=13, alpha=False, wmode="additive"),
        dict(T=200, d=16, H=2, B=2, seed=14, alpha=True, wmode="additive"),
        dict(T=200, d=24, H=2, B=2, seed=15, alpha=True, wmode="delta", dv=12),  # rectangular
    ]
    for cfg in cfgs:
        q, k, v, beta, alpha = _mk(B=cfg["B"], H=cfg["H"], T=cfg["T"], d=cfg["d"],
                                   seed=cfg["seed"], gated=cfg["alpha"], dv=cfg.get("dv"))
        Oa, Sa = chunked_delta(q, k, v, beta, alpha=alpha, chunk=64, write_mode=cfg["wmode"])
        Ob, Sb = chunked_delta(q, k, v, beta, alpha=alpha, chunk=64, write_mode=cfg["wmode"],
                               fast_reads=False)
        dO = (Oa - Ob).abs().max().item()
        dS = (Sa - Sb).abs().max().item()
        assert dO == 0.0, f"default path NOT byte-identical (O) cfg={cfg}: {dO:.3e}"
        assert dS == 0.0, f"default path NOT byte-identical (S) cfg={cfg}: {dS:.3e}"


def test_fast_reads_inert_without_gate():
    """alpha=None (pure delta): the gated read_ratio branch is never reached, so fast_reads=True is
    a NO-OP and must be EXACTLY equal to fast_reads=False (== 0.0)."""
    for wmode in ("delta", "additive"):
        q, k, v, beta, _ = _mk(T=200, seed=77)
        O0, S0 = chunked_delta(q, k, v, beta, alpha=None, chunk=64, write_mode=wmode,
                               fast_reads=False)
        O1, S1 = chunked_delta(q, k, v, beta, alpha=None, chunk=64, write_mode=wmode,
                               fast_reads=True)
        dO = (O0 - O1).abs().max().item()
        dS = (S0 - S1).abs().max().item()
        assert dO == 0.0, f"fast_reads not inert without gate (O, {wmode}): {dO:.3e}"
        assert dS == 0.0, f"fast_reads not inert without gate (S, {wmode}): {dS:.3e}"
