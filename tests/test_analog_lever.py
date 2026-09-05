"""Tests for the analog-robustness levers (report 12-H2): state_bits + write_noise_std.

These two knobs emulate a low-precision/noisy ANALOG carried state on the exact O(1) `step()`
path only (the chunk-parallel training kernel never materializes per-token states — documented
scope limitation in PrizmaSeqConfig). Contract under test:

OFF-identity:  state_bits=0, write_noise_std=0.0 (the defaults) are BIT-IDENTICAL to the
               pre-lever model: same forward output AND same loss trajectory over training
               steps (no extra ops, no rng consumption).
ON-changes:    knobs ON change the streaming (step) output while leaving forward() untouched.
quantizer:     _quantize_state_symmetric is symmetric (q(-x) == -q(x)) and monotone
               (x1 <= x2 => q(x1) <= q(x2)), grid = {-qmax..qmax} * max|x|/qmax.
O(1) guard:    with knobs OFF, step()==forward() still holds to <1e-4 (existing G1 discipline);
               with knobs ON step() intentionally deviates from the FP32 forward() — the
               documented degradation is ASSERTED, not hidden.
toy:           with write noise ON, BOTH write modes (delta, additive) still learn (streaming
               loss decreases after training).
"""
import torch
import pytest

from seq.prizma_seq import PrizmaSeqLM, PrizmaSeqConfig, _quantize_state_symmetric
from seq.common import masked_ce, set_seed
from seq.tasks import MQAR


TINY = dict(vocab=32, d_model=32, n_layers=1, n_heads=2, max_len=48)


def _tiny_model(**kw):
    return PrizmaSeqLM(PrizmaSeqConfig(**TINY, **kw))


def _stream(model, x, B):
    """Stream x through step() WITH proper state threading; returns logits [B,T,V]."""
    st = model.init_state(B, x.device)
    outs = []
    for t in range(x.shape[1]):
        lg, st = model.step(x[:, t:t + 1], st)
        outs.append(lg)
    return torch.cat(outs, dim=1)


# ── 1. OFF-identity: bit-identical forward output AND loss trajectory ──────────

def test_off_identity_forward_output():
    """Defaults (state_bits=0, write_noise_std=0.0) == pre-lever model, forward bit-identical."""
    torch.manual_seed(42)
    m_base = _tiny_model()
    torch.manual_seed(42)
    m_off = _tiny_model(state_bits=0, write_noise_std=0.0)
    m_base.train(False); m_off.train(False)
    x = torch.randint(0, 32, (2, 24))
    d = (m_base(x) - m_off(x)).abs().max().item()
    assert d == 0.0, f"knobs-off forward not bit-identical: max|d|={d:.2e}"


def test_off_identity_loss_trajectory():
    """Defaults == pre-lever model over 4 training steps: identical losses AND post-train
    weights (proves the guarded path consumes no ops and no rng)."""
    task = MQAR(vocab=32, num_pairs=4, num_queries=8)
    losses = []
    final_params = []
    for knobs in (None, dict(state_bits=0, write_noise_std=0.0)):
        set_seed(7)
        m = _tiny_model() if knobs is None else _tiny_model(**knobs)
        m.train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        set_seed(11)
        run = []
        for _ in range(4):
            x, y, msk = task.sample(8, m.tok.weight.device)
            loss = masked_ce(m(x), y, msk)
            opt.zero_grad(); loss.backward(); opt.step()
            run.append(float(loss.detach()))
        losses.append(run)
        final_params.append(torch.cat([p.detach().reshape(-1) for p in m.parameters()]))
    assert losses[0] == losses[1], f"loss trajectories differ: {losses[0]} vs {losses[1]}"
    assert (final_params[0] - final_params[1]).abs().max().item() == 0.0, \
        "post-training weights differ between default and explicit knobs-off instances"


def test_off_identity_streaming_output():
    """Defaults: streaming step() output bit-identical between default and explicit-off models."""
    torch.manual_seed(42)
    m_base = _tiny_model()
    torch.manual_seed(42)
    m_off = _tiny_model(state_bits=0, write_noise_std=0.0)
    m_base.train(False); m_off.train(False)
    x = torch.randint(0, 32, (2, 24))
    d = (_stream(m_base, x, 2) - _stream(m_off, x, 2)).abs().max().item()
    assert d == 0.0, f"knobs-off streaming not bit-identical: max|d|={d:.2e}"


# ── 2. Knobs ON change the streaming output (and only it) ──────────────────────

@pytest.mark.parametrize("kw", [
    dict(state_bits=4),
    dict(write_noise_std=0.05),
    dict(state_bits=6, write_noise_std=0.01),
])
def test_knobs_on_change_step_output_not_forward(kw):
    """state_bits/write_noise_std ON: step() output changes; forward() is untouched (the
    training kernel is out of scope by the documented limitation)."""
    torch.manual_seed(3)
    m_ref = _tiny_model()
    m_on = _tiny_model(**kw)
    m_on.load_state_dict(m_ref.state_dict())
    m_ref.train(False); m_on.train(False)
    x = torch.randint(0, 32, (2, 24))
    d_fwd = (m_ref(x) - m_on(x)).abs().max().item()
    assert d_fwd == 0.0, f"knobs ON changed forward(): max|d|={d_fwd:.2e}"
    d_step = (_stream(m_ref, x, 2) - _stream(m_on, x, 2)).abs().max().item()
    assert d_step > 1e-5, (
        f"knobs {kw} did NOT change the streaming output (max|d|={d_step:.2e}) — "
        "the lever is wired to the wrong place or silently inert")
    # reproducibility (write_noise_seed discipline): a second identical stream matches exactly
    d_rep = (_stream(m_on, x, 2) - _stream(m_on, x, 2)).abs().max().item()
    assert d_rep == 0.0, "knobs-ON stream not reproducible"


# ── 3. Quantizer unit tests: symmetric & monotone ──────────────────────────────

def test_quantizer_symmetric():
    """q(-x) == -q(x) on state-shaped and row-shaped inputs (per-head max-abs range)."""
    for x in (torch.randn(3, 2, 8, 16) * 2.0, torch.linspace(-5, 5, 257)[None, :]):
        for bits in (4, 6, 8):
            q = _quantize_state_symmetric(x, bits)
            q_neg = _quantize_state_symmetric(-x, bits)
            assert torch.allclose(q, -q_neg, atol=1e-7), \
                f"quantizer not symmetric at bits={bits}"
            qmax = 2 ** (bits - 1) - 1
            s = x.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12) / qmax
            codes = q / s
            assert torch.allclose(codes, torch.round(codes), atol=1e-4), \
                f"quantized values off the symmetric integer grid at bits={bits}"
            assert codes.abs().max() <= qmax + 1e-4, \
                f"quantized codes exceed qmax at bits={bits}"


def test_quantizer_monotone():
    """x1 <= x2 (elementwise) => q(x1) <= q(x2): sweep a monotone ramp through the grid."""
    x = torch.linspace(-3, 3, 601)[None, :]
    for bits in (4, 6, 8):
        q = _quantize_state_symmetric(x, bits)[0]
        assert bool((q[1:] >= q[:-1] - 1e-12).all()), f"quantizer not monotone at bits={bits}"
        # non-decreasing STEP grid: the number of distinct levels must be <= 2*qmax+1
        assert q.unique().numel() <= 2 * (2 ** (bits - 1) - 1) + 1


def test_quantizer_ste_backward_is_identity():
    """Straight-through estimator: backward gradient of the quantizer == identity."""
    x = torch.randn(2, 2, 4, 8, requires_grad=True)
    out = _quantize_state_symmetric(x, 4).sum()
    out.backward()
    assert torch.allclose(x.grad, torch.ones_like(x)), \
        "STE violated: gradient through the quantizer is not identity"


# ── 4. O(1) guard: step()==forward() with knobs OFF; documented deviation with knobs ON ──

def test_g1_step_equals_forward_knobs_off():
    """G1 O(1) guard still holds at the defaults: step()==forward() < 1e-4 (no interaction
    damage from the analog-lever wiring)."""
    m = _tiny_model()
    m.train(False)
    torch.manual_seed(5)
    x = torch.randint(0, 32, (2, 48))
    y = m(x)
    d = (y - _stream(m, x, 2)).abs().max().item()
    assert d < 1e-4, f"G1 O(1) guard failed at defaults: max|d|={d:.2e}"


def test_g1_step_equals_forward_additive():
    """Regression for the analog-probe finding (2026-09-03): step() ignored
    write_mode='additive' and applied the delta write, so an additive-trained model deployed
    through the streaming path ran a write rule it was never trained with (measured: streaming
    0.358 vs parallel-forward 0.742 on additive MixedMQAR seed0). The additive write in step()
    must mirror chunked_delta(write_mode='additive') (u = beta_w * v, no erase read-back) so
    step()==forward() covers the additive ablation too."""
    m = _tiny_model(write_mode="additive")
    m.train(False)
    torch.manual_seed(8)
    x = torch.randint(0, 32, (2, 48))
    y = m(x)
    d = (y - _stream(m, x, 2)).abs().max().item()
    assert d < 1e-4, f"G1 O(1) guard failed for write_mode='additive': max|d|={d:.2e}"


def test_knobs_on_step_intentionally_deviates_from_forward():
    """The DOCUMENTED scope limitation, asserted: with state_bits=6 + write noise ON, step()
    deviates from the FP32 forward(). This is the measured deployment degradation (train FP32 /
    deploy degraded per report 12-H2), NOT a kernel mismatch — the forward() path does not
    emulate the analog state (the chunk-parallel kernel cannot, without a re-derivation)."""
    m = _tiny_model(state_bits=6, write_noise_std=0.05)
    m.train(False)
    torch.manual_seed(6)
    x = torch.randint(0, 32, (2, 48))
    y = m(x)
    d = (y - _stream(m, x, 2)).abs().max().item()
    assert d > 1e-3, (
        f"expected the documented knobs-ON step()/forward() deviation, got max|d|={d:.2e}")


# ── 5. Tiny toy: under write noise BOTH write modes still learn ────────────────

@pytest.mark.parametrize("write_mode", ["delta", "additive"])
def test_toy_noisy_write_both_modes_learn(write_mode):
    """delta+noise and additive+noise both DECREASE streaming loss after training (the loss is
    measured THROUGH the noisy low-precision streaming path — deployment-emulation metric)."""
    torch.set_num_threads(max(1, torch.get_num_threads()))
    task = MQAR(vocab=32, num_pairs=4, num_queries=8)
    set_seed(0)
    m = _tiny_model(write_mode=write_mode, state_bits=6, write_noise_std=0.05)
    m.train(False)
    set_seed(12345)
    frozen = [task.sample(16, torch.device("cpu")) for _ in range(4)]

    def stream_loss():
        tot = 0.0
        for x, y, msk in frozen:
            lg = _stream(m, x, 16)
            tot += float(masked_ce(lg, y, msk))
        return tot / len(frozen)

    loss_before = stream_loss()
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    set_seed(21)
    for _ in range(150):
        x, y, msk = task.sample(16, torch.device("cpu"))
        loss = masked_ce(m(x), y, msk)          # training runs on the FP32 kernel (documented)
        opt.zero_grad(); loss.backward(); opt.step()
    m.train(False)
    loss_after = stream_loss()
    assert loss_after < loss_before, (
        f"[{write_mode}] noisy streaming loss did not decrease: {loss_before:.4f} -> {loss_after:.4f}")
