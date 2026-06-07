"""Tests for Lever B: higher-order DeltaProduct (n_delta=k).

k=1 must be byte-identical to today's chunked_delta (< 1e-6).
k=2 chunked must match _delta_reference < 1e-4.

NOTE: For n_delta>=2, the chunked form uses a sequential k-axis loop within each T-chunk
(correctness-first fallback as documented). Chunking over T is preserved.
"""
import torch
import pytest
from seq.delta import _delta_reference, chunked_delta


def test_k1_is_identical_to_today():
    """n_delta=1 must give byte-identical output to the old chunked_delta (< 1e-6)."""
    torch.manual_seed(0)
    B, H, T, d = 2, 2, 64, 16
    q = torch.randn(B, H, T, d)
    k = torch.randn(B, H, T, d)
    k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn(B, H, T, d)
    beta = torch.rand(B, H, T) * 0.99

    O0, S0 = chunked_delta(q, k, v, beta)
    O1, S1 = chunked_delta(q, k, v, beta, n_delta=1)

    dO = (O0 - O1).abs().max().item()
    dS = (S0 - S1).abs().max().item()
    assert dO < 1e-6, f"k=1 O mismatch: {dO:.2e}"
    assert dS < 1e-6, f"k=1 S mismatch: {dS:.2e}"


def test_k2_chunked_matches_reference():
    """k=2 chunked_delta must match _delta_reference < 1e-4 (pure alpha)."""
    torch.manual_seed(0)
    B, H, T, d = 2, 2, 96, 16
    # k=2 needs 2 key/val sets stacked on a new axis: shape [B,H,T,2,d]
    q = torch.randn(B, H, T, d)
    k = torch.randn(B, H, T, 2, d)
    k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn(B, H, T, 2, d)
    beta = torch.rand(B, H, T, 2) * 0.99

    Oref, Sref = _delta_reference(q, k, v, beta, n_delta=2)
    Och, Sch   = chunked_delta(q, k, v, beta, n_delta=2)

    dO = (Oref - Och).abs().max().item()
    dS = (Sref - Sch).abs().max().item()
    assert dO < 1e-4, f"k=2 chunked vs reference dO={dO:.2e} >= 1e-4"
    assert dS < 1e-4, f"k=2 chunked vs reference dS={dS:.2e} >= 1e-4"


def test_k2_chunked_matches_reference_gated():
    """k=2 chunked_delta must match _delta_reference < 1e-4 with gated alpha."""
    torch.manual_seed(1)
    B, H, T, d = 2, 2, 96, 16
    q = torch.randn(B, H, T, d)
    k = torch.randn(B, H, T, 2, d)
    k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn(B, H, T, 2, d)
    beta = torch.rand(B, H, T, 2) * 0.99
    alpha = 0.5 + 0.5 * torch.rand(B, H, T)  # gated

    Oref, Sref = _delta_reference(q, k, v, beta, alpha=alpha, n_delta=2)
    Och, Sch   = chunked_delta(q, k, v, beta, alpha=alpha, n_delta=2)

    dO = (Oref - Och).abs().max().item()
    dS = (Sref - Sch).abs().max().item()
    assert dO < 1e-4, f"k=2 gated chunked vs reference dO={dO:.2e} >= 1e-4"
    assert dS < 1e-4, f"k=2 gated chunked vs reference dS={dS:.2e} >= 1e-4"


def test_k1_reference_matches_chunked_is_identical():
    """_delta_reference with n_delta=1 must equal chunked_delta with n_delta=1 < 1e-4
    (standard correctness; ensures the n_delta=1 code path in reference is intact)."""
    torch.manual_seed(2)
    B, H, T, d = 2, 2, 128, 16
    q = torch.randn(B, H, T, d)
    k = torch.randn(B, H, T, d)
    k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn(B, H, T, d)
    beta = torch.rand(B, H, T) * 0.99

    Oref, Sref = _delta_reference(q, k, v, beta, n_delta=1)
    Och, Sch   = chunked_delta(q, k, v, beta, n_delta=1)

    dO = (Oref - Och).abs().max().item()
    dS = (Sref - Sch).abs().max().item()
    assert dO < 1e-4, f"k=1 ref vs chunked dO={dO:.2e}"
    assert dS < 1e-4, f"k=1 ref vs chunked dS={dS:.2e}"
