# tests/test_window.py
import torch
from seq.prizma_seq import PrizmaSeqBlock, PrizmaSeqConfig


def test_banded_window_equals_full_window():
    torch.manual_seed(0)
    cfg = PrizmaSeqConfig(vocab=64, d_model=64, n_layers=1, n_heads=2, window=16)
    blk = PrizmaSeqBlock(cfg)
    B, H, T, dh = 2, cfg.n_heads, 200, cfg.d_h
    q = torch.randn(B, H, T, dh)
    k = torch.randn(B, H, T, dh)
    v = torch.randn(B, H, T, dh)
    full = blk._window(q, k, v)
    banded = blk._window_banded(q, k, v)
    assert (full - banded).abs().max().item() < 1e-4
