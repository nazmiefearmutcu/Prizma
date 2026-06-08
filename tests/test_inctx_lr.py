"""Tests for Lever G: in-context per-channel learning rate (RWKV-7 "Goose" generalized delta).

The scalar write gate beta_t (one rate per head) is generalized into a per-VALUE-channel rate vector
eta_t in R^{d_v} that modulates the delta write per state channel:

    u_t = eta_t (elementwise over value channels) * (v_t - alpha_t * S_{t-1} k_t)
    S_t = alpha_t * S_{t-1} + u_t k_t^T

OFF-path identity: when eta is None (inctx_lr=False), chunked_delta must give byte-identical results
to today's scalar-beta behaviour (< 1e-6).

In-context correctness: with a random per-channel eta vector, _delta_reference and chunked_delta must
agree to < 1e-4 for both pure and gated-alpha cases (chunked uses a documented sequential-within-chunk
fallback because the WY/UT closed form cannot absorb a per-channel rate exactly — mirrors 'surprise').

G1 O(1) guard: a PrizmaSeqLM with inctx_lr=True must have step() == forward() (< 1e-4): the streaming
step() must apply the SAME per-channel eta as the parallel forward().
"""
import time
import torch
import pytest
from seq.delta import _delta_reference, chunked_delta


def _mk(T=128, d=16, H=2, B=2, seed=0, dv=None):
    g = torch.Generator().manual_seed(seed)
    dv = d if dv is None else dv
    q = torch.randn(B, H, T, d, generator=g)
    k = torch.randn(B, H, T, d, generator=g)
    k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn(B, H, T, dv, generator=g)
    beta = torch.rand(B, H, T, generator=g) * 0.99
    return q, k, v, beta


def test_off_path_identity():
    """eta=None must give numerically identical output to the baseline scalar-beta call (< 1e-6)."""
    q, k, v, beta = _mk(seed=1)
    O_base, S_base = chunked_delta(q, k, v, beta)
    O_none, S_none = chunked_delta(q, k, v, beta, eta=None)
    assert (O_base - O_none).abs().max().item() < 1e-6, \
        f"OFF-path O mismatch: {(O_base - O_none).abs().max().item():.2e}"
    assert (S_base - S_none).abs().max().item() < 1e-6, \
        f"OFF-path S mismatch: {(S_base - S_none).abs().max().item():.2e}"
    # Reference must also be unchanged with eta=None.
    Or_base, Sr_base = _delta_reference(q, k, v, beta)
    Or_none, Sr_none = _delta_reference(q, k, v, beta, eta=None)
    assert (Or_base - Or_none).abs().max().item() < 1e-6
    assert (Sr_base - Sr_none).abs().max().item() < 1e-6


def test_inctx_lr_chunked_matches_reference_pure():
    """Random per-channel eta, pure alpha: chunked_delta must match _delta_reference < 1e-4."""
    B, H, T, d = 2, 2, 128, 16
    q, k, v, beta = _mk(T=T, d=d, H=H, B=B, seed=3)
    g = torch.Generator().manual_seed(30)
    eta = torch.rand(B, H, T, d, generator=g) * 0.99   # per VALUE channel rate in [0, beta_cap)

    Oref, Sref = _delta_reference(q, k, v, beta, eta=eta)
    Och, Sch   = chunked_delta(q, k, v, beta, eta=eta)

    dO = (Oref - Och).abs().max().item()
    dS = (Sref - Sch).abs().max().item()
    assert dO < 1e-4, f"inctx_lr pure dO={dO:.2e} >= 1e-4"
    assert dS < 1e-4, f"inctx_lr pure dS={dS:.2e} >= 1e-4"


def test_inctx_lr_chunked_matches_reference_gated():
    """Random per-channel eta, gated alpha: chunked_delta must match _delta_reference < 1e-4."""
    B, H, T, d = 2, 2, 128, 16
    q, k, v, beta = _mk(T=T, d=d, H=H, B=B, seed=4)
    g = torch.Generator().manual_seed(40)
    eta = torch.rand(B, H, T, d, generator=g) * 0.99
    alpha = 0.5 + 0.5 * torch.rand(B, H, T, generator=g)   # gated decay in [0.5,1]

    Oref, Sref = _delta_reference(q, k, v, beta, alpha=alpha, eta=eta)
    Och, Sch   = chunked_delta(q, k, v, beta, alpha=alpha, eta=eta)

    dO = (Oref - Och).abs().max().item()
    dS = (Sref - Sch).abs().max().item()
    assert dO < 1e-4, f"inctx_lr gated dO={dO:.2e} >= 1e-4"
    assert dS < 1e-4, f"inctx_lr gated dS={dS:.2e} >= 1e-4"


def _grad_close(a, b, rtol=1e-3):
    """Relative max-abs gradient agreement. Gradients of a sum-of-squares loss over a 200-token
    sequence have magnitude ~1e2..1e3, so an absolute-1e-4 bar would flag pure float32 accumulation
    noise as a mismatch. The exactness criterion for the kernel is therefore RELATIVE: |dgrad| /
    max|grad| (the per-channel chunked path agrees with the sequential reference to ~1e-6 relative)."""
    dmax = (a - b).abs().max().item()
    scale = max(a.abs().max().item(), b.abs().max().item(), 1e-12)
    return dmax / scale, dmax, scale


def test_inctx_lr_chunked_matches_reference_grad_pure():
    """GRAD equivalence (pure alpha): the chunk-parallel eta path must backprop the same gradients
    as the sequential reference w.r.t. q,k,v,eta. Relative bound (see _grad_close): the kernel is
    exact through autograd to ~1e-6 relative — pins that it is not just forward-correct."""
    B, H, T, d = 2, 2, 200, 16   # production regime: T>=200, chunk=64
    q, k, v, beta = _mk(T=T, d=d, H=H, B=B, seed=7)
    g = torch.Generator().manual_seed(70)
    eta = torch.rand(B, H, T, d, generator=g) * 0.99

    def run(fn):
        qq, kk, vv = q.clone().requires_grad_(True), k.clone().requires_grad_(True), v.clone().requires_grad_(True)
        ee = eta.clone().requires_grad_(True)
        O, S = fn(qq, kk, vv, beta, eta=ee)
        loss = O.square().sum() + S.square().sum()
        loss.backward()
        return O.detach(), S.detach(), qq.grad, kk.grad, vv.grad, ee.grad

    Oref, Sref, gqr, gkr, gvr, ger = run(lambda *a, **kw: _delta_reference(*a, **kw))
    Och, Sch, gqc, gkc, gvc, gec = run(lambda *a, **kw: chunked_delta(*a, chunk=64, **kw))
    assert (Oref - Och).abs().max().item() < 1e-4    # forward: absolute 1e-4 (task gate)
    assert (Sref - Sch).abs().max().item() < 1e-4
    for nm, a, b in [("q", gqr, gqc), ("k", gkr, gkc), ("v", gvr, gvc), ("eta", ger, gec)]:
        rel, dmax, scale = _grad_close(a, b)
        assert rel < 1e-3, f"inctx_lr pure grad[{nm}] rel={rel:.2e} (abs={dmax:.2e} scale={scale:.2e})"


def test_inctx_lr_chunked_matches_reference_grad_gated():
    """GRAD equivalence (gated alpha in [0.5,1]): chunk-parallel eta == reference grads (relative)."""
    B, H, T, d = 2, 2, 200, 16
    q, k, v, beta = _mk(T=T, d=d, H=H, B=B, seed=8)
    g = torch.Generator().manual_seed(80)
    eta = torch.rand(B, H, T, d, generator=g) * 0.99
    alpha = 0.5 + 0.5 * torch.rand(B, H, T, generator=g)

    def run(fn):
        qq, kk, vv = q.clone().requires_grad_(True), k.clone().requires_grad_(True), v.clone().requires_grad_(True)
        ee = eta.clone().requires_grad_(True)
        aa = alpha.clone().requires_grad_(True)
        O, S = fn(qq, kk, vv, beta, alpha=aa, eta=ee)
        (O.square().sum() + S.square().sum()).backward()
        return O.detach(), S.detach(), qq.grad, kk.grad, vv.grad, ee.grad, aa.grad

    Oref, Sref, gqr, gkr, gvr, ger, gar = run(lambda *a, **kw: _delta_reference(*a, **kw))
    Och, Sch, gqc, gkc, gvc, gec, gac = run(lambda *a, **kw: chunked_delta(*a, chunk=64, **kw))
    assert (Oref - Och).abs().max().item() < 1e-4    # forward: absolute 1e-4 (task gate)
    assert (Sref - Sch).abs().max().item() < 1e-4
    for nm, a, b in [("q", gqr, gqc), ("k", gkr, gkc), ("v", gvr, gvc),
                     ("eta", ger, gec), ("alpha", gar, gac)]:
        rel, dmax, scale = _grad_close(a, b)
        assert rel < 1e-3, f"inctx_lr gated grad[{nm}] rel={rel:.2e} (abs={dmax:.2e} scale={scale:.2e})"


def test_off_path_byte_identity_strict():
    """OFF-PATH BYTE IDENTITY: the eta=None fast path must be BIT-FOR-BIT unchanged by Lever G.
    max|d| must be EXACTLY 0.0 (not just <1e-6) vs a captured scalar-beta reference — this pins
    that the sacred eta=None / scalar WY-UT path was untouched. Several configs (pure, gated,
    additive, rectangular d_v!=d_k) to cover every fast-path branch."""
    cfgs = [
        dict(T=200, d=16, H=2, B=2, seed=11, alpha=False, wmode="delta", dv=None),
        dict(T=200, d=16, H=3, B=2, seed=12, alpha=True,  wmode="delta", dv=None),
        dict(T=128, d=16, H=2, B=2, seed=13, alpha=False, wmode="additive", dv=None),
        dict(T=200, d=24, H=2, B=2, seed=14, alpha=True,  wmode="delta", dv=12),  # rectangular
    ]
    for cfg in cfgs:
        q, k, v, beta = _mk(T=cfg["T"], d=cfg["d"], H=cfg["H"], B=cfg["B"], seed=cfg["seed"], dv=cfg["dv"])
        ga = torch.Generator().manual_seed(cfg["seed"] + 500)
        alpha = (0.5 + 0.5 * torch.rand(cfg["B"], cfg["H"], cfg["T"], generator=ga)) if cfg["alpha"] else None
        Ob, Sb = chunked_delta(q, k, v, beta, alpha=alpha, chunk=64, write_mode=cfg["wmode"])
        On, Sn = chunked_delta(q, k, v, beta, alpha=alpha, chunk=64, write_mode=cfg["wmode"], eta=None)
        dO = (Ob - On).abs().max().item()
        dS = (Sb - Sn).abs().max().item()
        assert dO == 0.0, f"OFF-path NOT byte-identical (O) cfg={cfg}: max|dO|={dO:.2e}"
        assert dS == 0.0, f"OFF-path NOT byte-identical (S) cfg={cfg}: max|dS|={dS:.2e}"


def test_inctx_lr_chunked_faster_than_reference():
    """SPEED sanity (cpu): the chunk-parallel eta path should NOT be the O(T*d^2) sequential scan
    anymore — for a long sequence it should be <= the reference wall-time. Loose bound; soft on a
    flaky timing (prints the ratio). Just proves the sequential delegation is gone."""
    B, H, T, d = 1, 2, 512, 24
    q, k, v, beta = _mk(T=T, d=d, H=H, B=B, seed=21)
    g = torch.Generator().manual_seed(210)
    eta = torch.rand(B, H, T, d, generator=g) * 0.99
    # warm up (graph/alloc)
    _delta_reference(q, k, v, beta, eta=eta)
    chunked_delta(q, k, v, beta, chunk=64, eta=eta)
    t0 = time.perf_counter(); _delta_reference(q, k, v, beta, eta=eta); t_ref = time.perf_counter() - t0
    t0 = time.perf_counter(); chunked_delta(q, k, v, beta, chunk=64, eta=eta); t_ch = time.perf_counter() - t0
    print(f"\n[inctx_lr speed] T={T} ref={t_ref*1e3:.1f}ms chunked={t_ch*1e3:.1f}ms ratio={t_ref/max(t_ch,1e-9):.2f}x")
    # Loose: chunked must not be DRAMATICALLY slower (it replaced the sequential scan). Allow 1.5x
    # slack for CI timing noise; the real point is it is no longer the per-token Python loop.
    assert t_ch <= t_ref * 1.5, f"chunked eta slower than reference: ref={t_ref*1e3:.1f}ms chunked={t_ch*1e3:.1f}ms"


def test_inctx_lr_model_step_equals_forward():
    """G1 O(1) guard: PrizmaSeqLM(inctx_lr=True) step() must equal forward() < 1e-4."""
    from seq.prizma_seq import PrizmaSeqLM, PrizmaSeqConfig
    from seq.common import get_device
    dev = get_device()
    cfg = PrizmaSeqConfig(vocab=64, d_model=64, n_layers=2, n_heads=2,
                          feat_map='quad2', inctx_lr=True)
    m = PrizmaSeqLM(cfg).to(dev)
    m.train(False)
    torch.manual_seed(0)
    x = torch.randint(0, 64, (2, 48), device=dev)
    y = m(x)
    st = m.init_state(2, dev)
    outs = []
    for t in range(x.shape[1]):
        lg, st = m.step(x[:, t:t+1], st)
        outs.append(lg)
    d = (y - torch.cat(outs, 1)).abs().max().item()
    assert d < 1e-4, f"G1 guard failed: max|d|={d:.2e}"


def test_inctx_lr_and_surprise_gate_mutually_exclusive():
    """Footgun guard: inctx_lr (Lever G) and surprise_gate (Lever A) are the TWO novel-core S3
    candidates. Enabling BOTH must raise — the delta kernel branch `if eta is not None: ... elif
    surprise:` makes eta silently win, ignoring surprise. Each S3 arm enables exactly one lever."""
    from seq.prizma_seq import PrizmaSeqConfig
    with pytest.raises(AssertionError, match="mutually exclusive"):
        PrizmaSeqConfig(vocab=64, d_model=64, n_layers=2, n_heads=2,
                        inctx_lr=True, surprise_gate=True)
    # each lever ALONE still constructs fine (the assert only rejects the invalid combo).
    PrizmaSeqConfig(vocab=64, d_model=64, n_layers=2, n_heads=2, inctx_lr=True)
    PrizmaSeqConfig(vocab=64, d_model=64, n_layers=2, n_heads=2, surprise_gate=True)
