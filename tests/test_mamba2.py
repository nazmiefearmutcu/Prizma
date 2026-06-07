"""
Tests for the faithful Mamba-2 (SSD / State-Space Duality) baseline arm (seq/mamba2.py).

Mamba-2 = Dao & Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms
Through Structured State Space Duality" (ICML 2024). The dominant non-attention SOTA
architecture. This is a NON-strawman build: the short depthwise causal conv, the gated output
(z gate), and the D skip are ALL present (omitting Mamba's components silently weakens the
baseline and would corrupt the Pareto comparison). These tests pin the project's non-negotiable
rigor bar:

  1. O(1) GUARD            : streaming step() over each token == forward() < 1e-4
                             (constant-size per-layer state = SSM state Hst + the conv cache,
                             NOT a growing KV cache).
  2. CHUNK == RECURRENT    : the chunk-parallel SSD mixer == the recurrent reference < 1e-4
                             on the production regime (T=256, chunk=64, float32).
  3. PARAM SANITY          : Mamba2LM is within ~10% of a same-d/L/H Transformer (Mamba adds
                             conv + dt + B/C + z-gate params; the exact spread is reported).
  4. REAL-MODEL LEARNING    : Mamba2LM genuinely learns a small deterministic copy toy task
                             (final loss < 0.6 * random-baseline loss) -> not a broken strawman.
  5. param_count + forward-shape smoke.
"""
import math

import pytest
import torch

from seq.mamba2 import Mamba2Config, Mamba2LM, Mamba2Block, mamba2_factory
from seq.transformer import Transformer, TFConfig
from seq.common import param_count, set_seed


def _devices():
    devs = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devs.append(torch.device("mps"))
    return devs


def test_forward_shape_and_param_smoke():
    cfg = Mamba2Config(vocab=64, d_model=64, n_layers=2, n_heads=4)
    m = Mamba2LM(cfg)
    x = torch.randint(0, 64, (3, 17))
    y = m(x)
    assert y.shape == (3, 17, 64), f"expected [3,17,64], got {tuple(y.shape)}"
    p = param_count(m)
    assert p > 0


@pytest.mark.parametrize("dev", _devices())
def test_o1_streaming_guard(dev):
    """The O(1) streaming step() must equal the parallel forward() to < 1e-4 on each device.
    State is a constant-size per-layer (SSM state Hst + conv cache), NOT a growing KV cache."""
    set_seed(0)
    cfg = Mamba2Config(vocab=64, d_model=64, n_layers=2, n_heads=4)
    m = Mamba2LM(cfg).to(dev)
    m.train(False)
    x = torch.randint(0, 64, (2, 48), device=dev)
    with torch.no_grad():
        y = m(x)
        st = m.init_state(2, dev)
        outs = []
        for t in range(x.shape[1]):
            lg, st = m.step(x[:, t:t + 1], st)
            outs.append(lg)
        yo = torch.cat(outs, dim=1)
    d = (y - yo).abs().max().item()
    assert d < 1e-4, f"O(1) streaming step() != forward(): max|d|={d:.2e}"


@pytest.mark.parametrize("dev", _devices())
def test_chunk_equals_recurrent(dev):
    """The chunk-parallel SSD mixer must equal the recurrent reference < 1e-4 on the production
    regime (T=256, chunk=64, float32). This is the load-bearing 'parallel training' correctness
    gate. (Not skipped: the chunk form is shipped as forward().)"""
    set_seed(1)
    cfg = Mamba2Config(vocab=64, d_model=64, n_layers=1, n_heads=4, chunk=64)
    blk = Mamba2Block(cfg).to(dev)
    blk.train(False)
    h = torch.randn(2, 256, 64, device=dev)
    with torch.no_grad():
        o_chunk = blk._mix(blk.norm1(h), use_chunk=True)
        o_recur = blk._mix(blk.norm1(h), use_chunk=False)
    d = (o_chunk - o_recur).abs().max().item()
    assert d < 1e-4, f"chunk-parallel SSD mixer != recurrent reference: max|d|={d:.2e}"


def test_param_sanity_vs_transformer():
    """Mamba2LM within ~10% of a same-d/L/H Transformer at the same vocab. Mamba-2 adds the short
    conv + dt + B/C projections + z output gate over a vanilla TF, so a small spread is expected &
    faithful (the in-proj is value-path only, not 3x like attention's qkv)."""
    V, d, L, H = 256, 128, 3, 4
    tf = Transformer(TFConfig(vocab=V, d_model=d, n_layers=L, n_heads=H, rope=True))
    mamba = Mamba2LM(Mamba2Config(vocab=V, d_model=d, n_layers=L, n_heads=H))
    p_tf = param_count(tf)
    p_m = param_count(mamba)
    spread = (p_m - p_tf) / p_tf
    assert abs(spread) <= 0.10, (
        f"Mamba-2 param spread vs TF too large: TF={p_tf} Mamba2={p_m} "
        f"spread={spread:+.4%} (must be within +-10%)")


def test_real_model_learns_copy_task():
    """A faithful (non-strawman) Mamba-2 must genuinely LEARN. Train Mamba2LM a few hundred steps on
    a tiny deterministic copy task with a fixed seed; assert final loss << random-baseline loss
    (< 0.6 * random). Kept CPU-fast (< ~25s: tiny d/T/steps).

    Task = dense copy-previous: target[t] = input[t-1], supervised at every t>=1. This exercises the
    Mamba-2 recurrence directly (the SSM state must hold the previous token's value and the read must
    recall it), so a model that learns it is doing real state-space recall — not a degenerate fit."""
    set_seed(123)
    dev = torch.device("cpu")
    V, T = 16, 32
    def sample(B):
        seq = torch.randint(1, V, (B, T))
        tgt = torch.zeros(B, T, dtype=torch.long)
        tgt[:, 1:] = seq[:, :-1]                  # predict the previous token
        mask = torch.zeros(B, T)
        mask[:, 1:] = 1.0
        return seq.to(dev), tgt.to(dev), mask.to(dev)

    # random-baseline loss = uniform predictor over the (V-1) informative classes
    random_loss = math.log(V - 1)

    cfg = Mamba2Config(vocab=V, d_model=32, n_layers=2, n_heads=2, max_len=T + 8, chunk=16)
    model = Mamba2LM(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    model.train()
    final_loss = float("nan")
    for step in range(300):
        x, y, m = sample(64)
        logits = model(x)
        V_ = logits.shape[-1]
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, V_), y.reshape(-1), reduction="none")
        mf = m.reshape(-1)
        loss = (ce * mf).sum() / mf.sum().clamp_min(1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        final_loss = float(loss.detach())
    assert final_loss < 0.6 * random_loss, (
        f"Mamba-2 failed to learn the copy toy task: final_loss={final_loss:.4f} "
        f">= 0.6 * random({random_loss:.4f})={0.6 * random_loss:.4f}")


def test_factory_signature():
    """mamba2_factory(d, L, H, **kw) -> (lambda V, T: Mamba2LM) with max_len = T + 8."""
    fac = mamba2_factory(64, 2, 4)
    m = fac(48, 100)
    assert isinstance(m, Mamba2LM)
    assert m.cfg.vocab == 48
    assert m.cfg.max_len == 108
    x = torch.randint(0, 48, (2, 16))
    assert m(x).shape == (2, 16, 48)


def test_make_arm_mamba2_wired():
    """make_arm('mamba2', d, L, H) returns a Mamba2.d{d}L{L}H{H} arm with a (V,T)->Module factory."""
    from seq.gpu_harness import make_arm
    name, fac = make_arm("mamba2", 64, 2, 4)
    assert name.startswith("Mamba2.d64L2H4"), f"unexpected arm name {name!r}"
    m = fac(64, 32)
    assert isinstance(m, Mamba2LM)
    assert m(torch.randint(0, 64, (2, 32))).shape == (2, 32, 64)
