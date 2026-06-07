"""Tests for Lever H: decoupled channel-wise erase/write (GDN-2).

OFF-path identity: when beta_e is None (or beta_e==beta), chunked_delta must give byte-identical
results to the old behaviour (< 1e-6).

Decoupled correctness: with a DIFFERENT random beta_e, _delta_reference and chunked_delta must
agree to < 1e-4 for both pure and gated-alpha cases.
"""
import torch
import pytest
from seq.delta import _delta_reference, chunked_delta


def _mk(T=128, d=16, H=2, B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, T, d, generator=g)
    k = torch.randn(B, H, T, d, generator=g)
    k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn(B, H, T, d, generator=g)
    beta = torch.rand(B, H, T, generator=g) * 0.99
    return q, k, v, beta


def test_off_path_identity_no_beta_e():
    """beta_e=None must give numerically identical output to the baseline call (< 1e-6)."""
    q, k, v, beta = _mk(seed=1)
    O_base, S_base = chunked_delta(q, k, v, beta)
    O_none, S_none = chunked_delta(q, k, v, beta, beta_e=None)
    assert (O_base - O_none).abs().max().item() < 1e-6, \
        f"OFF-path O mismatch: {(O_base - O_none).abs().max().item():.2e}"
    assert (S_base - S_none).abs().max().item() < 1e-6, \
        f"OFF-path S mismatch: {(S_base - S_none).abs().max().item():.2e}"


def test_off_path_identity_beta_e_equals_beta():
    """Explicitly passing beta_e=beta must also give < 1e-6 vs baseline."""
    q, k, v, beta = _mk(seed=2)
    O_base, S_base = chunked_delta(q, k, v, beta)
    O_eq, S_eq = chunked_delta(q, k, v, beta, beta_e=beta)
    assert (O_base - O_eq).abs().max().item() < 1e-6, \
        f"OFF-path (be=bw) O mismatch: {(O_base - O_eq).abs().max().item():.2e}"
    assert (S_base - S_eq).abs().max().item() < 1e-6, \
        f"OFF-path (be=bw) S mismatch: {(S_base - S_eq).abs().max().item():.2e}"


def test_decoupled_chunked_matches_reference_pure():
    """With a different random beta_e, chunked_delta must match _delta_reference < 1e-4 (pure alpha)."""
    torch.manual_seed(3)
    B, H, T, d = 2, 2, 128, 16
    q, k, v, beta = _mk(T=T, d=d, H=H, B=B, seed=3)
    beta_e = torch.rand(B, H, T) * 0.99   # DIFFERENT from beta

    Oref, Sref = _delta_reference(q, k, v, beta, beta_e=beta_e)
    Och, Sch   = chunked_delta(q, k, v, beta, beta_e=beta_e)

    dO = (Oref - Och).abs().max().item()
    dS = (Sref - Sch).abs().max().item()
    assert dO < 1e-4, f"Decoupled pure dO={dO:.2e} >= 1e-4"
    assert dS < 1e-4, f"Decoupled pure dS={dS:.2e} >= 1e-4"


def test_decoupled_chunked_matches_reference_gated():
    """With a different random beta_e and gated alpha, chunked_delta must match reference < 1e-4."""
    torch.manual_seed(4)
    B, H, T, d = 2, 2, 128, 16
    q, k, v, beta = _mk(T=T, d=d, H=H, B=B, seed=4)
    beta_e = torch.rand(B, H, T) * 0.99   # DIFFERENT from beta
    alpha  = 0.5 + 0.5 * torch.rand(B, H, T)  # gated

    Oref, Sref = _delta_reference(q, k, v, beta, alpha=alpha, beta_e=beta_e)
    Och, Sch   = chunked_delta(q, k, v, beta, alpha=alpha, beta_e=beta_e)

    dO = (Oref - Och).abs().max().item()
    dS = (Sref - Sch).abs().max().item()
    assert dO < 1e-4, f"Decoupled gated dO={dO:.2e} >= 1e-4"
    assert dS < 1e-4, f"Decoupled gated dS={dS:.2e} >= 1e-4"


def test_decoupled_gate_model_step_equals_forward():
    """G1 O(1) guard: model with decoupled_gate=True must have step()==forward() < 1e-4."""
    from seq.prizma_seq import PrizmaSeqLM, PrizmaSeqConfig
    from seq.common import get_device
    dev = get_device()
    cfg = PrizmaSeqConfig(vocab=64, d_model=64, n_layers=2, n_heads=2,
                          feat_map='quad2', decoupled_gate=True)
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
