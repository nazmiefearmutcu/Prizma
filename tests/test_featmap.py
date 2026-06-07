"""Tests for feat_map='quad2_lowrank' (Task 1.D lever D).

(a) quad2_lowrank produces the expected d_phi and ZERO added trainable params.
(b) Forward pass runs; the O(1) step==forward guard passes (< 1e-4).
(c) 'none' and 'quad2' are unchanged by the new code path.
"""
from __future__ import annotations

import torch
import pytest

from seq.prizma_seq import PrizmaSeqLM, PrizmaSeqConfig
from seq.common import param_count, get_device


DEV = get_device()
VOCAB = 64
D_MODEL = 64
N_HEADS = 2
N_LAYERS = 2
# d_h = D_MODEL // N_HEADS = 32

# --------------------------------------------------------------------------- #
# (a) d_phi shape + zero added trainable params
# --------------------------------------------------------------------------- #

def _make(feat_map: str, feat_rank: int = 0, feat_n2: int = 96) -> PrizmaSeqLM:
    cfg = PrizmaSeqConfig(
        vocab=VOCAB, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
        feat_map=feat_map, feat_rank=feat_rank, feat_n2=feat_n2,
    )
    return PrizmaSeqLM(cfg)


def test_lowrank_default_rank_d_phi():
    """With feat_rank=0 (default), effective r=14; d_phi = 32 + 14*15//2 = 32+105 = 137."""
    cfg = PrizmaSeqConfig(d_model=D_MODEL, n_heads=N_HEADS, feat_map="quad2_lowrank", feat_rank=0)
    d_h = cfg.d_h
    r = 14                               # hardcoded default when feat_rank=0
    expected_d_phi = d_h + r * (r + 1) // 2
    assert cfg.d_phi == expected_d_phi, f"d_phi={cfg.d_phi} != {expected_d_phi}"
    assert cfg.d_phi == 137             # concrete check for d_h=32


def test_lowrank_explicit_rank():
    """With feat_rank=11, d_phi = 32 + 11*12//2 = 32+66 = 98."""
    cfg = PrizmaSeqConfig(d_model=D_MODEL, n_heads=N_HEADS, feat_map="quad2_lowrank", feat_rank=11)
    assert cfg.d_phi == 98, f"d_phi={cfg.d_phi}"


def test_lowrank_zero_added_trainable_params():
    """quad2_lowrank must NOT add any trainable parameters vs 'none'."""
    p_none = param_count(_make("none"))
    p_lr = param_count(_make("quad2_lowrank"))
    assert p_none == p_lr, (
        f"quad2_lowrank added {p_lr - p_none} trainable params (expected 0); "
        f"none={p_none}, lowrank={p_lr}"
    )


def test_lowrank_buffers_not_params():
    """feat_P, feat_I_lr, feat_J_lr must be registered buffers, NOT Parameters."""
    m = _make("quad2_lowrank").blocks[0]
    param_names = {n for n, _ in m.named_parameters()}
    for buf_name in ("feat_P", "feat_I_lr", "feat_J_lr"):
        assert buf_name not in param_names, f"{buf_name} is a Parameter, should be a buffer"
        assert hasattr(m, buf_name), f"buffer {buf_name} not found on block"


# --------------------------------------------------------------------------- #
# (b) Forward runs + O(1) step==forward guard < 1e-4
# --------------------------------------------------------------------------- #

def _o1_guard(feat_map: str, feat_rank: int = 0) -> float:
    """Run step()-vs-forward() and return max|delta|."""
    cfg = PrizmaSeqConfig(
        vocab=VOCAB, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
        feat_map=feat_map, feat_rank=feat_rank,
    )
    m = PrizmaSeqLM(cfg).to(DEV)
    x = torch.randint(0, VOCAB, (2, 48), device=DEV)
    with torch.no_grad():
        y_fwd = m(x)
        m.train(False)
        st = m.init_state(2, DEV)
        outs = []
        for t in range(x.shape[1]):
            lg, st = m.step(x[:, t:t + 1], st)
            outs.append(lg)
        y_step = torch.cat(outs, dim=1)
    return (y_fwd - y_step).abs().max().item()


def test_lowrank_forward_runs():
    """Model with quad2_lowrank forward pass produces correct output shape."""
    m = _make("quad2_lowrank").to(DEV)
    x = torch.randint(0, VOCAB, (2, 16), device=DEV)
    with torch.no_grad():
        out = m(x)
    assert out.shape == (2, 16, VOCAB), f"unexpected output shape {out.shape}"


def test_lowrank_o1_guard():
    """step() must match forward() to < 1e-4 for quad2_lowrank (O(1) invariant)."""
    d = _o1_guard("quad2_lowrank")
    assert d < 1e-4, f"O(1) guard FAILED: max|delta|={d:.2e} >= 1e-4"


def test_lowrank_explicit_rank_o1_guard():
    """O(1) guard also holds for an explicitly specified feat_rank (r=11)."""
    d = _o1_guard("quad2_lowrank", feat_rank=11)
    assert d < 1e-4, f"O(1) guard FAILED (feat_rank=11): max|delta|={d:.2e} >= 1e-4"


# --------------------------------------------------------------------------- #
# (c) Regression: 'none' and 'quad2' unchanged
# --------------------------------------------------------------------------- #

def test_none_o1_guard():
    d = _o1_guard("none")
    assert d < 1e-4, f"'none' O(1) guard FAILED: {d:.2e}"


def test_quad2_o1_guard():
    d = _o1_guard("quad2")
    assert d < 1e-4, f"'quad2' O(1) guard FAILED: {d:.2e}"


def test_none_zero_params_unchanged():
    """'none' param count unchanged from before this task."""
    p_none = param_count(_make("none"))
    p_quad2 = param_count(_make("quad2"))
    assert p_none == p_quad2, f"'none' and 'quad2' should have same param count: {p_none} vs {p_quad2}"


def test_quad2_d_phi_unchanged():
    """'quad2' d_phi = d_h + feat_n2 (unchanged behaviour)."""
    cfg = PrizmaSeqConfig(d_model=D_MODEL, n_heads=N_HEADS, feat_map="quad2", feat_n2=96)
    assert cfg.d_phi == cfg.d_h + 96, f"quad2 d_phi={cfg.d_phi}"


def test_none_d_phi_unchanged():
    """'none' d_phi = d_h (unchanged behaviour)."""
    cfg = PrizmaSeqConfig(d_model=D_MODEL, n_heads=N_HEADS, feat_map="none")
    assert cfg.d_phi == cfg.d_h, f"none d_phi={cfg.d_phi}"
