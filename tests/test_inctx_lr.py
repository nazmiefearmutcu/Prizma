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
