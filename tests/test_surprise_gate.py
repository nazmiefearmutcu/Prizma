"""Tests for the surprise-NORM write gate (PR-2026-09-03-01 arm A4 — REGISTERED-FROZEN protocol).

These pin the implementation to the frozen pre-registration
(docs/preregistry/2026-09-03-surprise-gating-powered-ablation.md, §2 A4 implementation spec):
the gate  beta_t = beta_cap * sigmoid(a * (s_t/max(m_t,1e-6) - 1))  with s_t = ||eps_t||^2 and a
causal per-(batch,head) EMA (m_1 = s_1, lam = 0.02 frozen a priori), computed INSIDE the recurrence
(gate REPLACES the learned gate; erase gate == write gate; gradients flow through the gate).

Required-by-doc bindings under test:
  (a) OFF-identity: every existing precision_gate/surprise_gate path stays BIT-IDENTICAL after the
      change (loss trajectory + post-train weights, two seeds; the new config fields are inert
      unless precision_gate == 'surprise_norm').
  (b) step()==forward() < 1e-4 for the new mode (the G1 O(1) guard, EMA state-carried vs
      per-sequence reset must agree token-by-token).
  (c) monotonicity: larger s_t at fixed m_t yields larger beta_t; self-calibration s~=m =>
      beta_cap/2; beta_cap applied AFTER sigmoid.
  (d) repeated-key exactness vs _delta_reference (the R3 binding: the WY/UT chunk form is invalid
      here, chunked_delta must route to the exact sequential scan).
Plus the runner's pure layer: the NO-TUNING gain selector (seed 900, tie -> smallest), the frozen
Welch+Holm verdict rule, the t_isf upper-tail MDE checksum, and the smoke/powered ledger refusal.
"""
import torch
import pytest

from seq.delta import _delta_reference, chunked_delta, _surprise_norm_gate, _surprise_norm_update
from seq.prizma_seq import PrizmaSeqLM, PrizmaSeqConfig
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


def _mk_delta(T=64, d=16, H=2, B=2, seed=0, repeated_key=False):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, T, d, generator=g)
    if repeated_key:
        k1 = torch.randn(1, 1, 1, d, generator=g)
        k1 = k1 / k1.norm(dim=-1, keepdim=True)
        k = k1.expand(B, H, T, d).clone()
    else:
        k = torch.randn(B, H, T, d, generator=g)
        k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn(B, H, T, d, generator=g)
    alpha = 0.5 + 0.5 * torch.rand(B, H, T, generator=g)
    return q, k, v, alpha


SN = (2.0, 0.02, 0.99)   # (gain a, EMA lambda, beta_cap) — the protocol's frozen lambda, a probe gain


# ── 1. OFF-identity (doc §2 item 5a): the new fields/knob are inert unless the gate is A4 ──────

@pytest.mark.parametrize("seed", [0, 1])
def test_off_identity_loss_and_weights_two_seeds(seed):
    """A4 OFF (any other precision_gate value, new fields explicitly set) == default model:
    identical loss trajectory AND identical post-training weights (no extra ops, no rng)."""
    task = MQAR(vocab=32, num_pairs=4, num_queries=8)
    losses, final_params = [], []
    for kw in (dict(), dict(precision_gate="input", surprise_gain=4.0, surprise_ema_lambda=0.5)):
        set_seed(seed)
        m = _tiny_model(**kw)
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


def test_off_identity_all_existing_gates_forward_bit_identical():
    """uniform / random / input forwards are bit-identical with the new fields defaulted AND
    explicitly set (the fields must be read ONLY under precision_gate == 'surprise_norm')."""
    for gate in ("input", "uniform"):
        torch.manual_seed(42)
        m_base = _tiny_model(precision_gate=gate)
        torch.manual_seed(42)
        m_off = _tiny_model(precision_gate=gate, surprise_gain=2.0, surprise_ema_lambda=0.02)
        m_base.train(False); m_off.train(False)
        x = torch.randint(0, 32, (2, 24))
        d = (m_base(x) - m_off(x)).abs().max().item()
        assert d == 0.0, f"gate={gate}: new fields perturbed forward (max|d|={d:.2e})"


def test_invalid_gate_name_rejected():
    """The config must REJECT invalid gate names (pre-reg: 'allow surprise_norm in the existing
    assertion' — a typo like 'surpise_norm' can never silently fall back to the input gate)."""
    with pytest.raises(AssertionError):
        PrizmaSeqConfig(precision_gate="surpise_norm")


@pytest.mark.parametrize("bad", [
    dict(inctx_lr=True), dict(surprise_gate=True), dict(decoupled_gate=True), dict(n_delta=2),
])
def test_a4_one_novel_lever_rule_rejects_combos(bad):
    """A4 replaces the learned gate and is scoped to n_delta==1 with surprise_gate/inctx_lr/
    decoupled_gate all False (pre-reg §2 implementation spec item 2 — the one-novel-lever rule)."""
    with pytest.raises(AssertionError):
        PrizmaSeqConfig(precision_gate="surprise_norm", **bad)


def test_a4_additive_write_mode_rejected_at_kernel():
    """The frozen formula includes the erase term beta_t*(alpha_t*S k_t); additive (no erase) is
    OUT of protocol scope and must raise — never degrade silently. Same for n_delta>=2."""
    q, k, v, alpha = _mk_delta(seed=3)
    with pytest.raises(NotImplementedError):
        chunked_delta(q, k, v, None, alpha=alpha, write_mode="additive", surprise_norm=SN)
    k_nd = torch.stack([k, k], dim=3)            # [B,H,T,n_delta,d]
    v_nd = torch.stack([v, v], dim=3)
    beta_nd = torch.rand(2, 2, 64, 2)
    with pytest.raises(NotImplementedError):
        chunked_delta(q, k_nd, v_nd, beta_nd, n_delta=2, surprise_norm=SN)


def test_a4_requires_beta_none():
    """beta is computed INSIDE the recurrence; a caller passing a beta tensor would be silently
    ignored — the kernel refuses instead (assert, per the replace-not-modulate pin)."""
    q, k, v, alpha = _mk_delta(seed=4)
    with pytest.raises(AssertionError):
        _delta_reference(q, k, v, torch.rand(2, 2, 64), alpha=alpha, surprise_norm=SN)


# ── 2. A4 ON changes the writes (sanity) + gradients flow through the gate ─────────────────────

def test_a4_on_changes_forward_output_same_weights():
    """Same weights, gate switched to A4: the forward output MUST change (the lever is not
    silently inert), at both the delta-kernel and full-model level."""
    torch.manual_seed(3)
    m_ref = _tiny_model(feat_map="quad2_lowrank")
    m_a4 = _tiny_model(feat_map="quad2_lowrank", precision_gate="surprise_norm", surprise_gain=2.0)
    m_a4.load_state_dict(m_ref.state_dict())
    m_ref.train(False); m_a4.train(False)
    x = torch.randint(0, 32, (2, 24))
    d = (m_ref(x) - m_a4(x)).abs().max().item()
    assert d > 1e-5, f"A4 did NOT change the forward output (max|d|={d:.2e}) — lever inert?"


def test_a4_gradients_flow_through_the_gate():
    """beta_t is a function of the running state (pre-reg: the mechanism must LEARN to exploit the
    signal, not receive it free) — gradients must reach the state-side inputs through the EMA."""
    q, k, v, alpha = _mk_delta(seed=5)
    q = q.clone().requires_grad_(True)
    v = v.clone().requires_grad_(True)
    O, _ = _delta_reference(q, k, v, None, alpha=alpha, surprise_norm=SN)
    O.sum().backward()
    assert q.grad is not None and float(q.grad.abs().sum()) > 0.0, "no grad through the gate (q)"
    assert v.grad is not None and float(v.grad.abs().sum()) > 0.0, "no grad through the gate (v)"


# ── 3. G1 O(1) guard: step()==forward() < 1e-4 for the new mode (doc §2 item 5b) ───────────────

@pytest.mark.parametrize("feat", ["none", "quad2_lowrank"])
def test_g1_step_equals_forward_a4(feat):
    """The state-carried EMA (init zeros; pos==0 sets m = s) must reproduce forward()'s per-sequence
    reset (m_1 = s_1) token-by-token — including through the phi feature map."""
    cfg = PrizmaSeqConfig(vocab=64, d_model=32, n_layers=2, n_heads=2, feat_map=feat,
                          precision_gate="surprise_norm", surprise_gain=2.0)
    torch.manual_seed(7)
    m = PrizmaSeqLM(cfg)
    m.train(False)
    x = torch.randint(0, 64, (2, 48))
    y = m(x)
    d = (y - _stream(m, x, 2)).abs().max().item()
    assert d < 1e-4, f"[{feat}] G1 guard failed for surprise_norm: max|d|={d:.2e}"


def test_a4_state_tuple_extension_shape_and_carry():
    """The streaming state gains a 6th slot: the per-head EMA m ([B,H], init zeros; first token sets
    m = s). pos stays slot 4 (pre-A4 consumers), and non-A4 modes carry m untouched (all zeros)."""
    torch.manual_seed(9)
    m_a4 = _tiny_model(precision_gate="surprise_norm", surprise_gain=1.0)
    m_in = _tiny_model(precision_gate="input")
    m_a4.train(False); m_in.train(False)
    x = torch.randint(0, 32, (3, 12))
    st = m_a4.init_state(3, x.device)
    assert len(st[0]) == 6 and st[0][5].shape == (3, 2), "A4 state must carry a [B,H] EMA slot"
    assert float(st[0][5].abs().sum()) == 0.0, "EMA must init to zeros"
    assert st[0][4] == 0, "pos must stay slot 4"
    for t in range(x.shape[1]):
        _, st = m_a4.step(x[:, t:t + 1], st)
    assert st[0][4] == x.shape[1], "pos must keep advancing"
    assert float(st[0][5].abs().sum()) > 0.0, "EMA must be updated by A4 writes"
    st_in = m_in.init_state(3, x.device)
    for t in range(x.shape[1]):
        _, st_in = m_in.step(x[:, t:t + 1], st_in)
    assert float(st_in[0][5].abs().sum()) == 0.0, "non-A4 modes must carry the EMA slot untouched"


# ── 4. The frozen formula: monotonicity, self-calibration, EMA (doc §2 item 5c) ────────────────

def test_gate_monotone_in_s_at_fixed_m():
    """Larger s_t at fixed m_t yields larger beta_t (rank-preserving; pre-reg property pin)."""
    m = torch.full((1, 1), 0.7)
    s = torch.linspace(0.0, 3.0, 61).reshape(1, -1).expand(1, -1).contiguous()
    beta = _surprise_norm_gate(s, m.expand_as(s), 2.0, 0.99)
    assert bool((beta[1:] >= beta[:-1] - 1e-7).all()), "beta_t not monotone non-decreasing in s_t"
    assert bool((beta.diff() > 0).all()), "beta_t expected strictly increasing on this ramp"


def test_gate_self_calibration_and_cap_after_sigmoid():
    """s_tilde == 1 => beta == beta_cap/2 exactly (sigma(0)); s_tilde == 2 => beta_cap*sigma(a);
    beta_cap multiplies AFTER sigmoid so beta < beta_cap strictly."""
    m = torch.ones(1, 1)
    b1 = _surprise_norm_gate(torch.ones(1, 1), m, 2.0, 0.99)
    assert abs(float(b1) - 0.99 / 2) < 1e-7, "s~=m must self-calibrate to beta_cap/2"
    b2 = _surprise_norm_gate(torch.full((1, 1), 2.0), m, 2.0, 0.99)
    assert abs(float(b2) - 0.99 * float(torch.sigmoid(torch.tensor(2.0)))) < 1e-6
    # beta_cap multiplies AFTER sigmoid: in the non-saturated regime beta is STRICTLY < beta_cap
    # (fp32 sigmoid saturates to exactly 1.0 at huge arguments, so the strict bound is checked
    # where sigma(a*(s-1)) < 1 in float32, and only <= cap+eps at saturation).
    s = torch.linspace(0.01, 10.0, 200).reshape(1, -1)
    b = _surprise_norm_gate(s, torch.ones_like(s), 0.5, 0.99)
    assert float(b.max()) < 0.99, "beta_cap must apply AFTER sigmoid (strictly < cap in-range)"
    s_big = torch.linspace(0.01, 500.0, 200).reshape(1, -1)
    b_big = _surprise_norm_gate(s_big, torch.ones_like(s_big), 4.0, 0.99)
    assert float(b_big.max()) <= 0.99 + 1e-6, "beta must never exceed beta_cap"


def test_ema_update_frozen_lambda():
    """m_1 = s_1 ; m_t = (1-lam)*m_{t-1} + lam*s_t — the exact causal recursion, differentiable."""
    s1, s2, s3 = torch.tensor(1.0), torch.tensor(3.0), torch.tensor(0.5)
    m1 = _surprise_norm_update(s1, None, 0.02)
    assert float(m1) == 1.0, "first token must set m = s (m_1 = s_1)"
    m2 = _surprise_norm_update(s2, m1, 0.02)
    assert abs(float(m2) - (0.98 * 1.0 + 0.02 * 3.0)) < 1e-7
    m3 = _surprise_norm_update(s3, m2, 0.02)
    assert abs(float(m3) - (0.98 * m2 + 0.02 * 0.5)) < 1e-7
    x = torch.randn(4, requires_grad=True)
    _surprise_norm_update(x * x, x.detach() * 0 + 1.0, 0.02).sum().backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0, "EMA must stay differentiable"


# ── 5. R3 binding: exact sequential scan on repeated keys (doc §2 item 5d) ─────────────────────

def test_repeated_key_exactness_vs_reference():
    """chunked_delta(surprise_norm=...) == _delta_reference(surprise_norm=...) on REPEATED KEYS
    (all tokens share one key): the routing must be the EXACT scan (a frozen-chunk approximation
    would diverge ~100% here, the R3 failure mode)."""
    q, k, v, alpha = _mk_delta(T=64, seed=10, repeated_key=True)
    Oref, Sref = _delta_reference(q, k, v, None, alpha=alpha, surprise_norm=SN)
    Och, Sch = chunked_delta(q, k, v, None, alpha=alpha, chunk=16, surprise_norm=SN)
    dO = (Oref - Och).abs().max().item()
    dS = (Sref - Sch).abs().max().item()
    assert dO < 1e-4, f"R3 repeated-key dO={dO:.4e} >= 1e-4 — BLOCKED (chunk approx?)"
    assert dS < 1e-4, f"R3 repeated-key dS={dS:.4e} >= 1e-4 — BLOCKED (chunk approx?)"


def test_chunk_size_independence():
    """chunk=16 vs chunk=64 agree EXACTLY under A4 (both route to the same sequential scan)."""
    q, k, v, alpha = _mk_delta(T=48, seed=11)
    O1, S1 = chunked_delta(q, k, v, None, alpha=alpha, chunk=16, surprise_norm=SN)
    O2, S2 = chunked_delta(q, k, v, None, alpha=alpha, chunk=64, surprise_norm=SN)
    assert (O1 - O2).abs().max().item() == 0.0 and (S1 - S2).abs().max().item() == 0.0


def test_gated_alpha_a4_runs_and_differs_from_fixed_beta():
    """With gated alpha (the protocol keeps alpha=1, but the kernel must stay correct for alpha<1):
    A4 output differs from a fixed-beta reference — the gate is genuinely state-derived."""
    q, k, v, alpha = _mk_delta(T=48, seed=12)
    O_a4, _ = chunked_delta(q, k, v, None, alpha=alpha, surprise_norm=SN)
    beta_fixed = torch.full((2, 2, 48), 0.495)   # the A4 first-token value (sigma(0)*cap)
    O_fix, _ = _delta_reference(q, k, v, beta_fixed, alpha=alpha)
    assert (O_a4 - O_fix).abs().max().item() > 1e-5, "A4 gate degenerated to a constant"


# ── 6. Runner pure layer: NO-TUNING gain selection (seed 900, tie -> smallest) ─────────────────

def test_gain_constants_frozen():
    import seq.surprise_claim as sc
    assert sc.EXPLORATORY_SEED == 900, "the NO-TUNING exploratory seed is 900, frozen"
    assert sc.GAIN_CANDIDATES == (0.5, 1.0, 2.0, 4.0), "the candidate set is frozen"
    assert sc.CLAIM_SEEDS == (0, 1, 2, 3, 4), "claim seeds are 0-4, n=5"
    assert sc.GAIN_SELECTION_LR == 1e-3 and sc.GAIN_TASK == "MQAR-D64"
    assert sc.STEP_CAP == 40000


def test_select_gain_highest_and_tie_to_smallest():
    from seq.surprise_claim import select_gain
    # unique max -> that gain
    assert select_gain({0.5: 0.8, 1.0: 0.9, 2.0: 0.7, 4.0: 0.6}) == 1.0
    # exact tie at the top -> SMALLEST a (the doc's literal tie-break)
    assert select_gain({0.5: 0.9, 1.0: 0.9, 2.0: 0.7, 4.0: 0.6}) == 0.5
    assert select_gain({0.5: 0.7, 1.0: 0.9, 2.0: 0.9, 4.0: 0.9}) == 1.0
    # all tied -> smallest candidate
    assert select_gain({a: 0.5 for a in (0.5, 1.0, 2.0, 4.0)}) == 0.5
    # the helper never invents a candidate outside the frozen set
    assert select_gain({0.5: 0.1, 1.0: 0.2, 2.0: 0.3, 4.0: 0.4}) == 4.0


# ── 7. Runner pure layer: the frozen §4 verdict (Welch + Holm, reverse guard) ──────────────────

def _accs(sn_by_task, uni_by_task, rnd_by_task):
    d = {}
    for t, (sn, un, rnd) in zip(("MQAR-D64", "MQAR-D128", "SELECTIVE-COPY"),
                                zip(sn_by_task, uni_by_task, rnd_by_task)):
        d[("surprise_norm", t)] = list(sn)
        d[("uniform", t)] = list(un)
        d[("random", t)] = list(rnd)
    return d


def test_verdict_survives_on_two_decisive_tasks():
    from seq.surprise_claim import surprise_verdict
    accs = _accs(
        [[0.95, 0.94, 0.96, 0.95, 0.94], [0.93, 0.94, 0.95, 0.94, 0.93], [0.70, 0.71, 0.70, 0.69, 0.70]],
        [[0.60, 0.61, 0.60, 0.59, 0.60], [0.62, 0.61, 0.62, 0.61, 0.60], [0.69, 0.70, 0.69, 0.70, 0.69]],
        [[0.63, 0.62, 0.63, 0.62, 0.63], [0.60, 0.61, 0.60, 0.61, 0.60], [0.70, 0.69, 0.70, 0.69, 0.70]],
    )
    v = surprise_verdict(accs)
    assert v["n_tasks_won_vs_both_controls"] == 2
    assert v["reverse_guard_ok"] is True
    assert v["survives"] is True, "win vs BOTH controls on 2 tasks + clean guard must SURVIVE"
    assert "SURVIVES" in v["verdict"]


def test_verdict_retired_on_single_task_win():
    from seq.surprise_claim import surprise_verdict
    # wins on ONE task only -> RETIRED even though never significantly worse
    accs = _accs(
        [[0.95, 0.94, 0.96, 0.95, 0.94], [0.70, 0.71, 0.70, 0.69, 0.70], [0.70, 0.71, 0.70, 0.69, 0.70]],
        [[0.60, 0.61, 0.60, 0.59, 0.60], [0.69, 0.70, 0.69, 0.70, 0.69], [0.69, 0.70, 0.69, 0.70, 0.69]],
        [[0.63, 0.62, 0.63, 0.62, 0.63], [0.70, 0.69, 0.70, 0.69, 0.70], [0.70, 0.69, 0.70, 0.69, 0.70]],
    )
    v = surprise_verdict(accs)
    assert v["n_tasks_won_vs_both_controls"] == 1
    assert v["survives"] is False and "RETIRED" in v["verdict"]


def test_verdict_retired_when_significantly_worse_anywhere():
    from seq.surprise_claim import surprise_verdict
    # wins BIG on two tasks but is significantly WORSE on the third -> clause-2 guard -> RETIRED
    accs = _accs(
        [[0.95, 0.94, 0.96, 0.95, 0.94], [0.93, 0.94, 0.95, 0.94, 0.93], [0.30, 0.31, 0.30, 0.29, 0.30]],
        [[0.60, 0.61, 0.60, 0.59, 0.60], [0.62, 0.61, 0.62, 0.61, 0.60], [0.95, 0.94, 0.96, 0.95, 0.94]],
        [[0.63, 0.62, 0.63, 0.62, 0.63], [0.60, 0.61, 0.60, 0.61, 0.60], [0.95, 0.96, 0.95, 0.94, 0.95]],
    )
    v = surprise_verdict(accs)
    assert v["n_tasks_won_vs_both_controls"] == 2
    assert v["reverse_guard_ok"] is False
    assert v["survives"] is False and "RETIRED" in v["verdict"]


def test_mde_checksum_pins_upper_tail_convention():
    """seq.stats.t_isf takes UPPER-TAIL p in (0, 0.5] — the tail-convention bug class PR-03 caught.
    The doc's frozen MDE constants must be reproduced exactly."""
    from seq.surprise_claim import mde_checksum
    ck = mde_checksum()
    assert ck["t_isf(0.05, 8)"]["ok"], ck
    assert ck["t_isf(0.20, 8)"]["ok"], ck
    assert ck["t_isf(0.05/6, 8)"]["ok"], ck


# ── 8. Runner pure layer: ledger separation + refusals (recall_gate pattern) ───────────────────

def test_smoke_and_powered_defaults_distinct(tmp_path, monkeypatch):
    import os
    monkeypatch.setenv("PRIZMA_RESULTS", str(tmp_path))
    from seq import surprise_claim as sc
    smk = sc._default_results_path(smoke=True)
    pw = sc._default_results_path(smoke=False)
    assert os.path.abspath(smk) != os.path.abspath(pw)
    assert smk.endswith("smoke.json") and "surprise_ablation_PR-2026-09-03-01" in pw
    # the LANE-EXPLORATORY gain file is a third location, outside the registry dir
    assert "exploratory" in sc.exploratory_gain_path()


def test_smoke_run_refuses_the_powered_ledger(tmp_path, monkeypatch):
    import os
    monkeypatch.setenv("PRIZMA_RESULTS", str(tmp_path))
    from seq import surprise_claim as sc
    powered = sc._default_results_path(smoke=False)
    with pytest.raises(SystemExit) as ei:
        sc.resolve_results_path(powered, smoke=True)
    assert "refus" in str(ei.value).lower(), "refusal must say why"
    # the deliberate bypass exists and an explicit non-powered --out never trips the guard
    assert sc.resolve_results_path(powered, smoke=True, force_smoke_path=True) == powered
    other = str(tmp_path / "elsewhere.json")
    assert sc.resolve_results_path(other, smoke=True) == other


def test_powered_requires_cuda():
    from seq.surprise_claim import require_cuda
    with pytest.raises(SystemExit) as ei:
        require_cuda(False)
    assert "cuda" in str(ei.value).lower(), "the refusal must name the CUDA requirement"
    require_cuda(True)   # no-op on a CUDA box


def test_cli_rejects_unknown_args_and_requires_one_mode():
    from seq.surprise_claim import _build_parser
    p = _build_parser()
    with pytest.raises(SystemExit) as ei:
        p.parse_args(["--smoke", "--extra"])     # typo'd/unknown flag: never launches anything
    assert ei.value.code != 0
    with pytest.raises(SystemExit):
        p.parse_args(["--smoke", "--powered"])   # mutually exclusive modes
    with pytest.raises(SystemExit):
        p.parse_args([])                          # exactly one mode is required
    assert p.parse_args(["--smoke"]).smoke is True
    assert p.parse_args(["--powered"]).powered is True
