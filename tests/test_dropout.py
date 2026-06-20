"""Tests for the residual + embedding dropout lever (regularizer-gap closer).

Motivation (honest): gpu_charlm2.py runs dropout-free because transformer.py (TFConfig) HAS
attention dropout but PrizmaSeqConfig had NONE — enabling dropout would have regularized only the
TF (unfair), so the char-LM harness leaned on weight_decay alone. PrizmaSeqConfig.dropout closes
that architectural gap and UNBLOCKS a fair symmetric-dropout experiment. This is a CAPABILITY, not
a BPC claim (whether it improves BPC is a GPU question, not pinned here).

Byte-identity contract (the hard requirement):
  * default cfg (dropout=0.0): nn.Dropout(0.0) is a true no-op that draws NO rng, so TRAIN-mode and
    EVAL-mode forward are bit-for-bit identical (max|d| == 0.0), and the whole default path is
    unchanged (proved additionally by every pre-existing value-pinning test staying green).
  * EVAL is unaffected by the dropout SETTING: a dropout=0.5 model in eval == the same-weights model
    in eval (dropout is identity in eval regardless of p).
  * TRAIN with p>0 is stochastic (some activations zeroed) -> two un-reseeded passes differ; output
    is finite + correct shape; switching to eval() restores determinism.
  * O(1) guard: step()==forward() (< 1e-4) still holds at dropout=0.0 (the residual dropout is an
    identity at p=0/eval, so the streaming-equals-parallel property is unaffected).
"""
import torch

from seq.prizma_seq import PrizmaSeqLM, PrizmaSeqConfig
from seq.common import get_device


def _build(dropout, seed=0, **kw):
    """Build a PrizmaSeqLM with a fixed init seed so two builds share identical weights."""
    torch.manual_seed(seed)
    cfg = PrizmaSeqConfig(vocab=64, d_model=64, n_layers=2, n_heads=2, dropout=dropout, **kw)
    return PrizmaSeqLM(cfg)


def test_dropout_default_byte_identical():
    """dropout=0.0: TRAIN-mode forward == EVAL-mode forward, max|d| EXACTLY 0.0.

    Proves nn.Dropout(0.0) is a true no-op in BOTH modes and draws no rng, so the published default
    path is byte-identical whether the model is in train() or eval()."""
    dev = get_device()
    m = _build(dropout=0.0, seed=0).to(dev)
    torch.manual_seed(1)
    x = torch.randint(0, 64, (2, 32), device=dev)

    m.train(True)
    y_train = m(x)
    m.train(False)
    y_eval = m(x)

    d = (y_train - y_eval).abs().max().item()
    assert d == 0.0, f"dropout=0.0 train!=eval (not a no-op): max|d|={d:.2e}"
    assert y_eval.isfinite().all()


def test_dropout_off_equals_no_dropout_eval():
    """In EVAL, a dropout=0.5 model == the same-seed/same-weights dropout=0.0 model, max|d| == 0.0.

    Dropout is identity at eval regardless of p, so the dropout SETTING does not change eval output.
    Same init seed => identical weights (nn.Dropout has no params, so it does not consume the rng
    stream during init -> weight init is unchanged)."""
    dev = get_device()
    m0 = _build(dropout=0.0, seed=7).to(dev)
    m5 = _build(dropout=0.5, seed=7).to(dev)
    m0.train(False)
    m5.train(False)

    torch.manual_seed(2)
    x = torch.randint(0, 64, (2, 24), device=dev)
    y0 = m0(x)
    y5 = m5(x)

    d = (y0 - y5).abs().max().item()
    assert d == 0.0, f"eval output depends on dropout setting (should not): max|d|={d:.2e}"


def test_dropout_on_trains_stochastic():
    """dropout=0.5 in TRAIN: two un-reseeded forwards differ (activations zeroed), output finite +
    correct shape; switching to eval() makes it deterministic again."""
    dev = get_device()
    m = _build(dropout=0.5, seed=3).to(dev)
    torch.manual_seed(4)
    x = torch.randint(0, 64, (2, 32), device=dev)

    m.train(True)
    y1 = m(x)
    y2 = m(x)            # NO reseed: dropout mask is freshly sampled -> stochastic
    d_train = (y1 - y2).abs().max().item()
    assert d_train > 0.0, "dropout=0.5 in train mode is deterministic (mask not applied?)"
    assert tuple(y1.shape) == (2, 32, 64), f"unexpected logits shape {tuple(y1.shape)}"
    assert y1.isfinite().all() and y2.isfinite().all(), "dropout produced non-finite logits"

    m.train(False)
    ye1 = m(x)
    ye2 = m(x)
    d_eval = (ye1 - ye2).abs().max().item()
    assert d_eval == 0.0, f"eval() not deterministic with dropout=0.5: max|d|={d_eval:.2e}"


def test_dropout_off_step_equals_forward():
    """O(1) guard: dropout=0.0 model has step()==forward() (< 1e-4) — the residual dropout is an
    identity at p=0/eval, so the streaming-equals-parallel property is unaffected by this lever."""
    dev = get_device()
    cfg = PrizmaSeqConfig(vocab=64, d_model=64, n_layers=2, n_heads=2, dropout=0.0)
    m = PrizmaSeqLM(cfg).to(dev)
    m.train(False)
    torch.manual_seed(0)
    x = torch.randint(0, 64, (2, 48), device=dev)
    y = m(x)
    st = m.init_state(2, dev)
    outs = []
    for t in range(x.shape[1]):
        lg, st = m.step(x[:, t:t + 1], st)
        outs.append(lg)
    d = (y - torch.cat(outs, 1)).abs().max().item()
    assert d < 1e-4, f"dropout=0.0 O(1) guard failed: max|d|={d:.2e}"
