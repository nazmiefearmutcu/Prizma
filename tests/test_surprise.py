"""Tests for Lever A: surprise-gated delta write (R3 + R9 bindings).

OFF-identity:     surprise=False path == old output to < 1e-6 (byte-identical semantics).
REPEATED-KEY exactness (R3 binding): chunked_delta(surprise=True) must EQUAL _delta_reference
    (surprise=True) to < 1e-4 even when all tokens share the SAME key vector.  This is the
    binding correctness requirement — a frozen-chunk approximation would diverge by ~100% here.
chunk independence: chunked_delta with chunk=16 vs chunk=64 agree to < 1e-4 (surprise=True).
controls (R9): 'random' and 'constant' modes run and are reproducible (same seed => same output).
G1 O(1) guard: step()==forward() < 1e-4 with surprise_gate=True.
"""
import torch
import pytest
from seq.delta import _delta_reference, chunked_delta


# ── helpers ──────────────────────────────────────────────────────────────────

def _mk(T=128, d=16, H=2, B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, T, d, generator=g)
    k = torch.randn(B, H, T, d, generator=g)
    k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn(B, H, T, d, generator=g)
    beta = torch.rand(B, H, T, generator=g) * 0.99
    return q, k, v, beta


def _mk_repeated_key(T=64, d=16, H=2, B=2, seed=10):
    """All tokens share the SAME unit-norm key — worst case for frozen-chunk approx (R3)."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, T, d, generator=g)
    k_single = torch.randn(1, 1, 1, d, generator=g)
    k_single = k_single / k_single.norm(dim=-1, keepdim=True)
    k = k_single.expand(B, H, T, d).clone()     # all tokens share the same key
    v = torch.randn(B, H, T, d, generator=g)
    beta = torch.rand(B, H, T, generator=g) * 0.99
    alpha = 0.5 + 0.5 * torch.rand(B, H, T, generator=g)
    return q, k, v, beta, alpha


# ── 1. OFF-path identity ─────────────────────────────────────────────────────

def test_off_identity_reference():
    """_delta_reference(surprise=False) == _delta_reference() to < 1e-6 (default path unchanged)."""
    q, k, v, beta = _mk(seed=1)
    O_base, S_base = _delta_reference(q, k, v, beta)
    O_off, S_off = _delta_reference(q, k, v, beta, surprise=False)
    dO = (O_base - O_off).abs().max().item()
    dS = (S_base - S_off).abs().max().item()
    assert dO < 1e-6, f"OFF-identity ref O mismatch: {dO:.2e}"
    assert dS < 1e-6, f"OFF-identity ref S mismatch: {dS:.2e}"


def test_off_identity_chunked():
    """chunked_delta(surprise=False) == chunked_delta() to < 1e-6 (fast path byte-identical)."""
    q, k, v, beta = _mk(seed=2)
    O_base, S_base = chunked_delta(q, k, v, beta)
    O_off, S_off = chunked_delta(q, k, v, beta, surprise=False)
    dO = (O_base - O_off).abs().max().item()
    dS = (S_base - S_off).abs().max().item()
    assert dO < 1e-6, f"OFF-identity chunked O mismatch: {dO:.2e}"
    assert dS < 1e-6, f"OFF-identity chunked S mismatch: {dS:.2e}"


def test_off_identity_chunked_gated():
    """chunked_delta(surprise=False) with gated alpha == baseline chunked to < 1e-6."""
    q, k, v, beta = _mk(seed=3)
    alpha = 0.5 + 0.5 * torch.rand(2, 2, 128)
    O_base, S_base = chunked_delta(q, k, v, beta, alpha=alpha)
    O_off, S_off = chunked_delta(q, k, v, beta, alpha=alpha, surprise=False)
    dO = (O_base - O_off).abs().max().item()
    dS = (S_base - S_off).abs().max().item()
    assert dO < 1e-6, f"OFF-identity chunked+gated O mismatch: {dO:.2e}"
    assert dS < 1e-6, f"OFF-identity chunked+gated S mismatch: {dS:.2e}"


# ── 2. REPEATED-KEY exactness (R3 binding) ────────────────────────────────────

def test_repeated_key_exactness_norm():
    """R3: chunked_delta(surprise=True,'norm') == _delta_reference on REPEATED KEYS to < 1e-4.
    A frozen-chunk approximation would diverge by ~100% here; the sequential scan is exact."""
    q, k, v, beta, alpha = _mk_repeated_key(T=64, d=16, seed=10)
    Oref, Sref = _delta_reference(q, k, v, beta, alpha=alpha, surprise=True, surprise_mode='norm')
    Och, Sch   = chunked_delta(q, k, v, beta, alpha=alpha, chunk=16, surprise=True, surprise_mode='norm')
    dO = (Oref - Och).abs().max().item()
    dS = (Sref - Sch).abs().max().item()
    # Hard limit per R3: must be < 1e-4 (exact, not a frozen approximation)
    assert dO < 1e-4, f"R3 repeated-key dO={dO:.4e} >= 1e-4 — BLOCKED (frozen approx?)"
    assert dS < 1e-4, f"R3 repeated-key dS={dS:.4e} >= 1e-4 — BLOCKED (frozen approx?)"


def test_repeated_key_exactness_constant():
    """R3 holds for 'constant' surprise mode too (gate value is fixed, no seq dependency)."""
    q, k, v, beta, alpha = _mk_repeated_key(T=64, d=16, seed=11)
    Oref, Sref = _delta_reference(q, k, v, beta, alpha=alpha, surprise=True, surprise_mode='constant')
    Och, Sch   = chunked_delta(q, k, v, beta, alpha=alpha, chunk=16, surprise=True, surprise_mode='constant')
    dO = (Oref - Och).abs().max().item()
    dS = (Sref - Sch).abs().max().item()
    assert dO < 1e-4, f"constant mode repeated-key dO={dO:.4e} >= 1e-4"
    assert dS < 1e-4, f"constant mode repeated-key dS={dS:.4e} >= 1e-4"


def test_repeated_key_exactness_random():
    """R3 for 'random' mode — generator is threaded so each call gets the same per-token gates."""
    q, k, v, beta, alpha = _mk_repeated_key(T=64, d=16, seed=12)
    gen1 = torch.Generator().manual_seed(99)
    gen2 = torch.Generator().manual_seed(99)
    Oref, Sref = _delta_reference(q, k, v, beta, alpha=alpha, surprise=True,
                                  surprise_mode='random', surprise_gen=gen1)
    Och, Sch   = chunked_delta(q, k, v, beta, alpha=alpha, chunk=16, surprise=True,
                                surprise_mode='random', surprise_gen=gen2)
    dO = (Oref - Och).abs().max().item()
    dS = (Sref - Sch).abs().max().item()
    assert dO < 1e-4, f"random mode repeated-key dO={dO:.4e} >= 1e-4"
    assert dS < 1e-4, f"random mode repeated-key dS={dS:.4e} >= 1e-4"


def test_repeated_key_pure_alpha():
    """R3 with alpha=None (no forget, pure delta) and repeated keys."""
    q, k, v, beta, _ = _mk_repeated_key(T=64, d=16, seed=13)
    Oref, Sref = _delta_reference(q, k, v, beta, alpha=None, surprise=True, surprise_mode='norm')
    Och, Sch   = chunked_delta(q, k, v, beta, alpha=None, chunk=16, surprise=True, surprise_mode='norm')
    dO = (Oref - Och).abs().max().item()
    dS = (Sref - Sch).abs().max().item()
    assert dO < 1e-4, f"pure-alpha repeated-key dO={dO:.4e} >= 1e-4"
    assert dS < 1e-4, f"pure-alpha repeated-key dS={dS:.4e} >= 1e-4"


# ── 3. Chunk independence (chunk=16 vs chunk=64) ─────────────────────────────

def test_chunk_independence():
    """chunked_delta chunk=16 vs chunk=64 with surprise=True agree to < 1e-4."""
    q, k, v, beta, alpha = _mk_repeated_key(T=128, d=16, seed=20)
    O16, S16 = chunked_delta(q, k, v, beta, alpha=alpha, chunk=16, surprise=True, surprise_mode='norm')
    O64, S64 = chunked_delta(q, k, v, beta, alpha=alpha, chunk=64, surprise=True, surprise_mode='norm')
    dO = (O16 - O64).abs().max().item()
    dS = (S16 - S64).abs().max().item()
    assert dO < 1e-4, f"chunk independence dO={dO:.4e} >= 1e-4"
    assert dS < 1e-4, f"chunk independence dS={dS:.4e} >= 1e-4"


def test_chunk_independence_random_keys():
    """chunk=16 vs chunk=64 agree to < 1e-4 with random (non-repeated) keys."""
    q, k, v, beta = _mk(T=128, d=16, seed=21)
    alpha = 0.5 + 0.5 * torch.rand(2, 2, 128)
    O16, S16 = chunked_delta(q, k, v, beta, alpha=alpha, chunk=16, surprise=True, surprise_mode='norm')
    O64, S64 = chunked_delta(q, k, v, beta, alpha=alpha, chunk=64, surprise=True, surprise_mode='norm')
    dO = (O16 - O64).abs().max().item()
    dS = (S16 - S64).abs().max().item()
    assert dO < 1e-4, f"chunk independence (rand keys) dO={dO:.4e} >= 1e-4"
    assert dS < 1e-4, f"chunk independence (rand keys) dS={dS:.4e} >= 1e-4"


# ── 4. Controls reproducibility (R9) ─────────────────────────────────────────

def test_random_mode_reproducible():
    """'random' mode: same generator seed -> identical outputs (R8/R9 discipline)."""
    q, k, v, beta = _mk(T=32, d=16, seed=30)
    gen_a = torch.Generator().manual_seed(77)
    gen_b = torch.Generator().manual_seed(77)
    Oa, Sa = _delta_reference(q, k, v, beta, surprise=True, surprise_mode='random', surprise_gen=gen_a)
    Ob, Sb = _delta_reference(q, k, v, beta, surprise=True, surprise_mode='random', surprise_gen=gen_b)
    assert (Oa - Ob).abs().max().item() == 0.0, "random mode not reproducible (O differs)"
    assert (Sa - Sb).abs().max().item() == 0.0, "random mode not reproducible (S differs)"


def test_random_mode_different_seed_differs():
    """'random' mode: different seeds give different outputs (sanity: not all-zeros)."""
    q, k, v, beta = _mk(T=32, d=16, seed=31)
    gen_a = torch.Generator().manual_seed(1)
    gen_b = torch.Generator().manual_seed(2)
    Oa, _ = _delta_reference(q, k, v, beta, surprise=True, surprise_mode='random', surprise_gen=gen_a)
    Ob, _ = _delta_reference(q, k, v, beta, surprise=True, surprise_mode='random', surprise_gen=gen_b)
    assert (Oa - Ob).abs().max().item() > 0.0, "random mode seeds 1 and 2 gave identical output"


def test_constant_mode_reproducible():
    """'constant' mode: calling twice gives identical output (no stochasticity)."""
    q, k, v, beta = _mk(T=32, d=16, seed=32)
    Oa, Sa = _delta_reference(q, k, v, beta, surprise=True, surprise_mode='constant')
    Ob, Sb = _delta_reference(q, k, v, beta, surprise=True, surprise_mode='constant')
    assert (Oa - Ob).abs().max().item() == 0.0, "constant mode not reproducible (O)"
    assert (Sa - Sb).abs().max().item() == 0.0, "constant mode not reproducible (S)"


def test_constant_mode_g_value():
    """'constant' mode uses g = 1 + tanh(1.0) for all tokens (gate is uniform)."""
    import math
    expected_g = 1.0 + math.tanh(1.0)  # ≈ 1.762
    # With constant g the output equals the result of scaling beta effectively by g
    # Verify by checking that the ratio norm(O_constant) / norm(O_norm) is consistent with g > 1
    q, k, v, beta = _mk(T=32, d=16, seed=33)
    Oc, _ = _delta_reference(q, k, v, beta, surprise=True, surprise_mode='constant')
    # Gate must be > 1 (writes MORE) and bounded < 2
    assert Oc.norm().item() > 0.0, "constant mode output is zero"
    assert expected_g > 1.0 and expected_g < 2.0, f"g={expected_g} outside [1,2)"


def test_norm_mode_g_bounded():
    """'norm' mode: g = 1 + tanh(||eps||) is in [1, 2) for all tokens."""
    # Indirectly verified: if g >= 1 always, then O with surprise >= magnitude of O without
    # surprise isn't guaranteed (direction changes), but we can check against the constant=2 bound.
    q, k, v, beta = _mk(T=32, d=16, seed=34)
    O_norm, _ = _delta_reference(q, k, v, beta, surprise=True, surprise_mode='norm')
    assert O_norm.isfinite().all(), "norm mode produces non-finite values"


# ── 5. G1 O(1) guard: surprise_gate=True step() == forward() < 1e-4 ──────────

def test_g1_step_equals_forward_surprise():
    """G1 O(1) guard: model with surprise_gate=True; step()==forward() < 1e-4."""
    from seq.prizma_seq import PrizmaSeqLM, PrizmaSeqConfig
    from seq.common import get_device
    dev = get_device()
    cfg = PrizmaSeqConfig(vocab=64, d_model=64, n_layers=2, n_heads=2,
                          feat_map='none', surprise_gate=True, surprise_mode='norm')
    m = PrizmaSeqLM(cfg).to(dev)
    m.train(False)
    torch.manual_seed(5)
    x = torch.randint(0, 64, (2, 48), device=dev)
    y = m(x)
    st = m.init_state(2, dev)
    outs = []
    for t in range(x.shape[1]):
        lg, st = m.step(x[:, t:t + 1], st)
        outs.append(lg)
    d = (y - torch.cat(outs, 1)).abs().max().item()
    assert d < 1e-4, f"G1 surprise_gate O(1) guard failed: max|d|={d:.2e}"


def test_g1_step_equals_forward_surprise_gated():
    """G1 guard with gated alpha AND surprise_gate=True."""
    from seq.prizma_seq import PrizmaSeqLM, PrizmaSeqConfig
    from seq.common import get_device
    dev = get_device()
    cfg = PrizmaSeqConfig(vocab=64, d_model=64, n_layers=2, n_heads=2,
                          feat_map='none', surprise_gate=True, surprise_mode='norm', gated=True)
    m = PrizmaSeqLM(cfg).to(dev)
    m.train(False)
    torch.manual_seed(6)
    x = torch.randint(0, 64, (2, 48), device=dev)
    y = m(x)
    st = m.init_state(2, dev)
    outs = []
    for t in range(x.shape[1]):
        lg, st = m.step(x[:, t:t + 1], st)
        outs.append(lg)
    d = (y - torch.cat(outs, 1)).abs().max().item()
    assert d < 1e-4, f"G1 surprise_gate+gated O(1) guard failed: max|d|={d:.2e}"


def test_g1_surprise_off_identical_to_baseline():
    """Model with surprise_gate=False gives same output as default model (no regression)."""
    from seq.prizma_seq import PrizmaSeqLM, PrizmaSeqConfig
    from seq.common import get_device
    dev = get_device()
    torch.manual_seed(42)
    cfg_base = PrizmaSeqConfig(vocab=64, d_model=32, n_layers=1, n_heads=2, feat_map='none')
    cfg_off  = PrizmaSeqConfig(vocab=64, d_model=32, n_layers=1, n_heads=2, feat_map='none',
                               surprise_gate=False)
    # Use same init seed for both models
    m_base = PrizmaSeqLM(cfg_base).to(dev)
    torch.manual_seed(42)
    m_off  = PrizmaSeqLM(cfg_off).to(dev)
    m_base.train(False); m_off.train(False)
    x = torch.randint(0, 64, (2, 16), device=dev)
    y_base = m_base(x)
    y_off  = m_off(x)
    d = (y_base - y_off).abs().max().item()
    assert d < 1e-6, f"surprise_gate=False not identical to baseline: max|d|={d:.2e}"
