"""Tests for the tiny-HYBRID baseline arm (Council-3; plan Task 1.Hybrid-baseline).

A Samba / GatedDeltaNet-H-style stack: MOSTLY PrizmaSeqBlocks + ~1 attention Block,
added as a THIRD baseline alongside the pure Transformer and pure Prizma. This is an
adversarial honesty check: if Prizma is NOT Pareto-competitive with this tiny hybrid,
the honest framing is "best pure-O(1) point", not "beats the Transformer".

Covered:
  (a) HybridSeqLM @ d128L4H4 (vocab 64) forward output shape == [B, T, vocab].
  (b) EXACTLY 1 attention Block and (L-1) PrizmaSeqBlocks in the module list, with the
      attention layer at the MIDDLE index (n_layers//2) by default.
  (c) attn_layers override (int and tuple) places attention at the requested indices.
  (d) param-match to a Transformer at the same scale within a named tolerance (< 0.05
      rel), printing the actual spread.
  (e) a tiny FAST smoke: 2 train_model steps on a small MQAR run without error and
      produce a finite loss.
"""
from __future__ import annotations

import torch

from seq.hybrid import HybridSeqLM, hybrid_factory
from seq.transformer import Block, Transformer, TFConfig
from seq.prizma_seq import PrizmaSeqBlock
from seq.common import param_count, TrainConfig, train_model
from seq.tasks import MQAR


VOCAB = 64
D_MODEL = 128
N_LAYERS = 4
N_HEADS = 4
MAX_LEN = 128


# --------------------------------------------------------------------------- #
# (a) forward shape
# --------------------------------------------------------------------------- #
def test_forward_shape():
    m = hybrid_factory(D_MODEL, N_LAYERS, N_HEADS)(VOCAB, MAX_LEN)
    B, T = 2, 48
    logits = m(torch.randint(0, VOCAB, (B, T)))
    assert tuple(logits.shape) == (B, T, VOCAB)


# --------------------------------------------------------------------------- #
# (b) exactly 1 attention Block + (L-1) PrizmaSeqBlocks, attn at MIDDLE by default
# --------------------------------------------------------------------------- #
def test_block_composition_default_middle():
    m = hybrid_factory(D_MODEL, N_LAYERS, N_HEADS)(VOCAB, MAX_LEN)
    blocks = list(m.blocks)
    assert len(blocks) == N_LAYERS
    n_attn = sum(isinstance(b, Block) for b in blocks)
    n_prizma = sum(isinstance(b, PrizmaSeqBlock) for b in blocks)
    assert n_attn == 1, f"expected exactly 1 attention Block, got {n_attn}"
    assert n_prizma == N_LAYERS - 1, f"expected {N_LAYERS - 1} PrizmaSeqBlocks, got {n_prizma}"
    # default attention index is the MIDDLE: n_layers // 2
    mid = N_LAYERS // 2
    assert isinstance(blocks[mid], Block), f"attention Block should be at index {mid}"
    assert m.attn_layers == (mid,)


# --------------------------------------------------------------------------- #
# (c) attn_layers override (int + tuple)
# --------------------------------------------------------------------------- #
def test_attn_layers_override():
    # single int -> placed exactly at that index
    m_int = hybrid_factory(D_MODEL, N_LAYERS, N_HEADS, attn_layers=0)(VOCAB, MAX_LEN)
    assert m_int.attn_layers == (0,)
    assert isinstance(list(m_int.blocks)[0], Block)
    assert sum(isinstance(b, Block) for b in m_int.blocks) == 1

    # tuple of two -> two attention layers, rest Prizma
    m_tup = hybrid_factory(D_MODEL, N_LAYERS, N_HEADS, attn_layers=(0, 3))(VOCAB, MAX_LEN)
    assert m_tup.attn_layers == (0, 3)
    blocks = list(m_tup.blocks)
    assert isinstance(blocks[0], Block) and isinstance(blocks[3], Block)
    assert sum(isinstance(b, Block) for b in blocks) == 2
    assert sum(isinstance(b, PrizmaSeqBlock) for b in blocks) == N_LAYERS - 2

    # n_attn=2 (auto-placement) -> 2 attention layers
    m_n2 = hybrid_factory(D_MODEL, N_LAYERS, N_HEADS, n_attn=2)(VOCAB, MAX_LEN)
    assert sum(isinstance(b, Block) for b in m_n2.blocks) == 2


# --------------------------------------------------------------------------- #
# (d) param-match to TF at the same scale within a named tolerance
# --------------------------------------------------------------------------- #
PARAM_TOL = 0.05   # named tolerance: hybrid must land within 5% of the matched TF baseline

def test_param_match_to_transformer():
    hyb = hybrid_factory(D_MODEL, N_LAYERS, N_HEADS)(VOCAB, MAX_LEN)
    tf = Transformer(TFConfig(vocab=VOCAB, d_model=D_MODEL, n_layers=N_LAYERS,
                              n_heads=N_HEADS, max_len=MAX_LEN + 8, rope=True))
    p_hyb = param_count(hyb)
    p_tf = param_count(tf)
    rel = (p_hyb - p_tf) / p_tf
    print(f"\n  param-match @ d{D_MODEL}L{N_LAYERS}H{N_HEADS} (V={VOCAB}): "
          f"Hybrid {p_hyb:,}p  TF {p_tf:,}p  ({100.0 * rel:+.2f}% vs TF)")
    assert abs(rel) < PARAM_TOL, (
        f"hybrid-vs-TF spread {100.0 * rel:+.2f}% exceeds {100.0 * PARAM_TOL:.0f}% tolerance")


# --------------------------------------------------------------------------- #
# (e) FAST smoke: 2 train_model steps on a small MQAR -> finite loss, no error
# --------------------------------------------------------------------------- #
def test_train_smoke_finite_loss():
    import math
    torch.manual_seed(0)
    # tiny config: tiny model, tiny MQAR, 2 steps, CPU-ok
    m = hybrid_factory(64, 2, 2)(32, 64)
    task = MQAR(vocab=32, num_pairs=4, num_queries=4, gap=0)
    cfg = TrainConfig(steps=2, batch_size=8, eval_every=1, eval_batches=1,
                      min_steps=0, warmup=1, warmup_frac=0.0, log=False)
    device = torch.device("cpu")
    res = train_model(m, task, cfg, device, seed=0)
    assert math.isfinite(res.final_loss), f"non-finite loss: {res.final_loss}"
    assert len(res.history) >= 1
