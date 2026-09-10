"""Lane-1 tests — the tissue multi-timescale (fast/slow) cascade (guarded, DEFAULT-OFF).

Campaign 2026-09-11 CONTRACT.md §"Lane 1". Frozen mechanism semantics:
  - per expert head a zero-init W_fast; forward/readout uses W + W_fast;
  - when enabled the expert optimizer receives ONLY the W_fast parameters;
  - after every optimizer step: W <- W + kappa*W_fast; W_fast <- (1-delta)*W_fast;
  - flags --cascade-target {off,tissue} (default off), --cascade-kappa (0.05),
    --cascade-delta (0.10); recorded in ledger meta + config fingerprint;
  - off must be byte-identical to the pre-lever path.

Coverage (acceptance):
  (a) OFF == no-flag exact equality on a tiny fusion training (same params, same bpc, same
      ledger field set — no cascade diagnostics on the OFF path);
  (b) mechanism math: fold equations, forward uses W+F, optimizer receives ONLY W_fast;
  (c) flags thread through all four lane files' entry points (parsers + signatures + the
      fingerprint payloads);
  (d) diagnostics (per-slot fast L2/max) appear only on the ON path; a Policy-A eviction
      re-init inherits the cascade.
"""
import inspect

import pytest
import torch

from seq import fusion_probe as fp
from seq import prizma_lm_claim as plc
from seq import manyblock_claim as mbc
from seq import manyblock_probe as mbp


# ── helpers ─────────────────────────────────────────────────────────────────────────────────

def _tiny_data(V=11, n_segs=4, seed=7):
    g = torch.Generator().manual_seed(seed)
    xs = torch.randint(0, V, (n_segs, fp.SEG), generator=g)
    ys = torch.randint(0, V, (n_segs, fp.SEG), generator=g)
    return xs, ys


def _fresh_ledger():
    return {"recruits": [], "cap_fallback_batches": 0, "cap_fallback_segments": 0,
            "cap_events": []}


def _train_tiny(model, V=11):
    xs, ys = _tiny_data(V)
    led = _fresh_ledger()
    fp.train_stream_fusion(model, xs, ys, 3e-3, "A", led)
    return xs, ys, led


# ── (c) flags: parsers ─────────────────────────────────────────────────────────────────────

def test_cascade_flags_defaults_and_roundtrip_in_all_three_parsers():
    for parser, base in ((plc._build_parser(), ["--powered-cpu"]),
                         (mbc._build_parser(), ["--powered-cpu"]),
                         (mbp._build_parser(), [])):
        a = parser.parse_args(list(base))
        assert a.cascade_target == "off", "default must be off (byte-identical pre-lever)"
        assert a.cascade_kappa == pytest.approx(0.05)
        assert a.cascade_delta == pytest.approx(0.10)
        b = parser.parse_args(list(base) + ["--cascade-target", "tissue",
                                            "--cascade-kappa", "0.2",
                                            "--cascade-delta", "0.3"])
        assert (b.cascade_target, b.cascade_kappa, b.cascade_delta) == ("tissue", 0.2, 0.3)
    with pytest.raises(SystemExit):
        plc._build_parser().parse_args(["--powered-cpu", "--cascade-target", "nonsense"])
    with pytest.raises(SystemExit):
        mbp._build_parser().parse_args(["--cascade-target", "nonsense"])


def test_cascade_alone_changes_no_other_default():
    for parser, base in ((plc._build_parser(), ["--powered-cpu"]),
                         (mbc._build_parser(), ["--powered-cpu"])):
        plain = parser.parse_args(list(base))
        treated = parser.parse_args(list(base) + ["--cascade-target", "tissue",
                                                  "--cascade-kappa", "0.2",
                                                  "--cascade-delta", "0.3"])
        names = set(vars(plain)) - {"cascade_target", "cascade_kappa", "cascade_delta"}
        for key in sorted(names):
            assert getattr(plain, key) == getattr(treated, key), \
                f"--cascade-* alone must not move any other parser default ({key})"


def test_validate_cascade_truth_table():
    assert plc.validate_cascade("off", 0.05, 0.10) == ("off", 0.05, 0.10)
    assert plc.validate_cascade("tissue", 0.0, 1.0) == ("tissue", 0.0, 1.0)
    with pytest.raises(SystemExit, match="cascade-target"):
        plc.validate_cascade("nonsense", 0.05, 0.10)
    with pytest.raises(SystemExit, match="cascade-kappa"):
        plc.validate_cascade("tissue", -0.01, 0.10)
    with pytest.raises(SystemExit, match="cascade-delta"):
        plc.validate_cascade("tissue", 0.05, 1.01)
    # the same guard object is aliased by the sibling runners
    assert mbc.validate_cascade is plc.validate_cascade
    assert mbp.validate_cascade is plc.validate_cascade


def test_cascade_keys_are_in_both_claim_fingerprint_payloads():
    src = inspect.getsource(plc.run)
    assert src.count('"cascade_target": cascade_target') == 2, \
        "the lr-selection AND claim fingerprint payloads must both carry the lever"
    assert src.count('"cascade_kappa": cascade_kappa') == 2
    assert src.count('"cascade_delta": cascade_delta') == 2


def test_fingerprint_sensitivity_to_cascade_values():
    base = {"leg": "claim", "arm": "PRIM-LM", "family": "fusion", "seed": 0, "lr": 3e-3,
            "cascade_target": "off", "cascade_kappa": 0.05, "cascade_delta": 0.10}
    sig_off = plc._fp(dict(base))
    assert plc._fp({**base, "cascade_target": "tissue"}) != sig_off, \
        "a treated rerun must never resume from untreated cells"
    assert plc._fp({**base, "cascade_kappa": 0.2}) != sig_off
    assert plc._fp({**base, "cascade_delta": 0.3}) != sig_off
    mbase = {"leg": "claim", "arm": "EX", "seed": 0, "lr": 3e-3,
             "cascade_target": "tissue", "cascade_kappa": 0.05, "cascade_delta": 0.10}
    assert mbc._fp({**mbase, "cascade_target": "off"}) != mbc._fp(dict(mbase))


def test_flags_thread_signatures_of_all_four_files():
    def has(fn, *names):
        params = inspect.signature(fn).parameters
        return all(n in params for n in names)

    fields = ("cascade_target", "cascade_kappa", "cascade_delta")
    assert has(plc.run, *fields), "prizma_lm_claim.run must thread the flags"
    assert has(plc.run_routed, *fields), "the PRIM/FROZEN/FORCED cell must thread the flags"
    assert has(mbc.run, *fields), "manyblock_claim.run must thread the flags"
    assert has(mbc.run_cell, *fields), "the 5-block cell must thread the flags"
    assert has(mbp.run_seed, *fields), "the probe's seed runner must thread the flags"
    assert has(mbp.main, "argv")
    # source-level: the claim runners build the cascade model + the probe records diagnostics
    assert 'cascade=(cascade_target == "tissue")' in inspect.getsource(plc.run_routed)
    assert 'cascade=(cascade_target == "tissue")' in inspect.getsource(mbc.run_cell)
    assert "cascade_diagnostics" in inspect.getsource(mbc.run_cell)
    assert "cascade_diagnostics" in inspect.getsource(mbp.run_seed)
    assert "expert_cascade_kwargs" in inspect.getsource(plc.route_pr08), \
        "eviction re-init must inherit the cascade"


# ── (b) mechanism math ─────────────────────────────────────────────────────────────────────

def test_off_head_has_no_fast_component_and_forward_is_unchanged():
    V = 5
    a = fp.PCExpertHead(4, 3, V)
    assert a.fast_parameters() is None
    assert not hasattr(a, "Wenc_fast")
    b = fp.PCExpertHead(4, 3, V, cascade=False)
    with torch.no_grad():
        b.load_state_dict(a.state_dict())
    h = torch.randn(7, 4, generator=torch.Generator().manual_seed(1))
    assert torch.equal(a(h), b(h)), "cascade=False forward must be the exact existing path"


def test_cascade_fold_equations_and_forward_uses_w_plus_f():
    V = 5
    head = fp.PCExpertHead(4, 3, V, cascade=True, cascade_kappa=0.2, cascade_delta=0.3)
    with torch.no_grad():
        head.Wenc.weight.copy_(torch.randn(3, 4, generator=torch.Generator().manual_seed(2)))
        head.Wdec.weight.copy_(torch.randn(V, 3, generator=torch.Generator().manual_seed(3)))
        head.Wenc_fast.copy_(torch.randn(3, 4, generator=torch.Generator().manual_seed(4)))
        head.Wdec_fast.copy_(torch.randn(V, 3, generator=torch.Generator().manual_seed(5)))
        we0 = head.Wenc.weight.clone()
        wd0 = head.Wdec.weight.clone()
        fe0 = head.Wenc_fast.clone()
        fd0 = head.Wdec_fast.clone()

    # forward = Wdec(tanh(Wenc)) with W -> W + F on BOTH matrices (the effective weight)
    h = torch.randn(6, 4, generator=torch.Generator().manual_seed(6))
    expect = torch.nn.functional.linear(
        torch.tanh(torch.nn.functional.linear(h, we0 + fe0, head.Wenc.bias)),
        wd0 + fd0, head.Wdec.bias)
    assert torch.equal(head(h), expect), "forward/readout must use W + W_fast"

    # fold: W <- W + kappa*F (with the PRE-decay F); F <- (1-delta)*F
    head.consolidate()
    assert torch.equal(head.Wenc.weight, we0 + 0.2 * fe0)
    assert torch.equal(head.Wdec.weight, wd0 + 0.2 * fd0)
    assert torch.equal(head.Wenc_fast, 0.7 * fe0)
    assert torch.equal(head.Wdec_fast, 0.7 * fd0)
    # the effective weight moves by exactly (kappa - delta)*F when both act on one step
    # (associativity-safe comparison: the two expressions round differently in fp32)
    assert torch.allclose(head.Wenc.weight + head.Wenc_fast,
                          we0 + fe0 + (0.2 - 0.3) * fe0, rtol=1e-6, atol=1e-7)
    # and kappa == delta leaves the effective weight invariant (pure reparameterization of
    # the fold itself; AdamW weight_decay can still move it during the optimizer step)
    head2 = fp.PCExpertHead(4, 3, V, cascade=True, cascade_kappa=0.1, cascade_delta=0.1)
    with torch.no_grad():
        head2.Wenc_fast.copy_(fe0)
    w_before = head2.Wenc.weight.clone()
    head2.consolidate()
    assert torch.allclose(head2.Wenc.weight + head2.Wenc_fast, w_before + fe0,
                          rtol=1e-6, atol=1e-7)


def test_cascade_requires_grad_structure_optimizer_sees_fast_and_biases():
    model = fp.build_model(11, 0, 4, cascade=True, cascade_kappa=0.1, cascade_delta=0.2)
    head = model.experts[0]
    assert head.Wenc.weight.requires_grad is False
    assert head.Wdec.weight.requires_grad is False
    fast = head.fast_parameters()
    # coordinator clarification 2026-09-11: biases are NOT part of the fast/slow split —
    # they train normally in both modes (no bias-training confound in the ON-vs-OFF probe)
    assert [id(p) for p in fast] == [id(head.Wenc_fast), id(head.Wdec_fast),
                                     id(head.Wenc.bias), id(head.Wdec.bias)]
    assert all(p.requires_grad for p in fast)

    bias_e0 = head.Wenc.bias.detach().clone()
    w0 = head.Wenc.weight.detach().clone()
    f0 = head.Wenc_fast.detach().clone()
    assert torch.equal(f0, torch.zeros_like(f0)), "W_fast must be zero-init"

    xs, ys = _tiny_data(11)
    with torch.no_grad():
        model.lm(xs)
    fp._expert_train(model, 0, model._h, ys, [0, 1], 3e-3)

    with torch.no_grad():
        f1 = head.Wenc_fast.clone()
        w1 = head.Wenc.weight.clone()
        # the optimizer moved the fast component (nonzero grad -> nonzero fast)
        assert not torch.equal(f1, f0)
        assert head.Wenc.weight.grad is None, "the slow W must receive no optimizer gradient"
        # the biases ARE in the cascade optimizer: they train (move from zero init), and the
        # consolidation fold leaves them alone (fold touches only the slow weights)
        bias_e1 = head.Wenc.bias.clone()
        assert not torch.equal(bias_e1, bias_e0), "biases train in cascade mode (no confound)"
        # and W moved exactly by the consolidation fold: W1 = W0 + kappa * F_prefold
        f_pre = f1 / (1.0 - 0.2)
        assert torch.allclose(w1, w0 + 0.1 * f_pre, rtol=1e-4, atol=1e-6)


# ── (a) OFF byte-identity on a tiny fusion training ────────────────────────────────────────

def test_off_run_equals_no_flag_run_exactly():
    V, seed, E = 11, 0, 4
    torch.manual_seed(99)
    xs, ys = _tiny_data(V)

    m_default = fp.build_model(V, seed, E)                     # no cascade argument at all
    m_off = fp.build_model(V, seed, E, cascade=False,
                           cascade_kappa=0.05, cascade_delta=0.10)  # explicit off
    sd_default = {k: v.clone() for k, v in m_default.state_dict().items()}
    sd_off = {k: v.clone() for k, v in m_off.state_dict().items()}
    assert set(sd_default) == set(sd_off)
    assert all(torch.equal(sd_default[k], sd_off[k]) for k in sd_default), \
        "OFF construction must be byte-identical to the no-flag construction"

    led1, led2 = _fresh_ledger(), _fresh_ledger()
    fp.train_stream_fusion(m_default, xs, ys, 3e-3, "A", led1)
    fp.train_stream_fusion(m_off, xs, ys, 3e-3, "A", led2)
    s1 = {k: v.clone() for k, v in m_default.state_dict().items()}
    s2 = {k: v.clone() for k, v in m_off.state_dict().items()}
    assert set(s1) == set(s2)
    assert all(torch.equal(s1[k], s2[k]) for k in s1), \
        "a tiny fusion training must be byte-identical between no-flag and explicit off"
    assert fp.eval_bpc_fusion(m_default, xs, ys) == fp.eval_bpc_fusion(m_off, xs, ys)
    assert led1 == led2

    tr = [m_off.n_segments[s] for s in range(E)]
    snap1 = fp.ledger_snapshot(m_default, led1, tr, [0] * E, [0] * E, [0] * E)
    snap2 = fp.ledger_snapshot(m_off, led2, tr, [0] * E, [0] * E, [0] * E)
    assert snap1 == snap2
    assert "cascade" not in snap1, \
        "the OFF ledger field set must be the exact pre-lever shape (no cascade key)"


def test_cascade_on_fires_and_differs_from_off():
    V, seed, E = 11, 0, 4
    xs, ys = _tiny_data(V)
    off_m = fp.build_model(V, seed, E)
    on_m = fp.build_model(V, seed, E, cascade=True)
    sd_off = off_m.state_dict()
    sd_on = on_m.state_dict()
    assert all(torch.equal(sd_off[k], sd_on[k]) for k in sd_off), \
        "the initial slow/backbone weights must be identical; W_fast=0 adds nothing"
    assert set(sd_on) - set(sd_off) == {f"experts.{s}.{n}" for s in range(E)
                                        for n in ("Wenc_fast", "Wdec_fast")}, \
        "the ON state_dict must add exactly the fast components"
    fp.train_stream_fusion(off_m, xs, ys, 3e-3, "A", _fresh_ledger())
    fp.train_stream_fusion(on_m, xs, ys, 3e-3, "A", _fresh_ledger())
    assert not torch.equal(off_m.experts[0].Wdec.weight, on_m.experts[0].Wdec.weight), \
        "the cascade must fire (attached to the guarded branch only, not a silent no-op)"


# ── (d) diagnostics + eviction inheritance ─────────────────────────────────────────────────

def test_diagnostics_only_when_cascade_on():
    V, seed, E = 11, 0, 4
    off = fp.build_model(V, seed, E)
    on = fp.build_model(V, seed, E, cascade=True)
    assert fp.cascade_diagnostics(off) is None
    diag0 = fp.cascade_diagnostics(on)
    assert diag0["target"] == "tissue" and diag0["kappa"] == 0.05 and diag0["delta"] == 0.10
    assert len(diag0["per_slot"]) == E
    assert all(p["fast_l2"] == 0.0 and p["fast_absmax"] == 0.0 for p in diag0["per_slot"]), \
        "zero-init fast components -> zero norms before any optimizer step"

    xs, ys = _tiny_data(V)
    with torch.no_grad():
        on.lm(xs)
    fp._expert_train(on, 0, on._h, ys, [0, 1], 3e-3)
    diag1 = fp.cascade_diagnostics(on)
    assert diag1["per_slot"][0]["fast_l2"] > 0.0
    assert diag1["per_slot"][0]["fast_absmax"] > 0.0
    assert diag1["per_slot"][0]["n_batches"] == 1
    assert diag1["total_fast_l2"] >= diag1["per_slot"][0]["fast_l2"]

    led = _fresh_ledger()
    snap_off = fp.ledger_snapshot(off, led, [0] * E, [0] * E, [0] * E, [0] * E)
    snap_on = fp.ledger_snapshot(on, led, [0] * E, [0] * E, [0] * E, [0] * E)
    assert "cascade" not in snap_off and snap_on["cascade"]["target"] == "tissue"


def test_pr08_ledger_snapshot_carries_cascade_only_when_on():
    V, seed, E = 11, 0, 4
    off = fp.build_model(V, seed, E)
    on = fp.build_model(V, seed, E, cascade=True)
    tr = {"A": [0] * E, "B": [0] * E, "C": [0] * E}
    boundary = {"segments_total": 0, "to_A_expert": 0, "a_expert": 0}
    expected = {"A": 0, "B": 0, "C": 0}
    s_off = plc.ledger_snapshot_pr08(off, plc._fresh_ledger(), tr, {}, expected, boundary, 0)
    s_on = plc.ledger_snapshot_pr08(on, plc._fresh_ledger(), tr, {}, expected, boundary, 0)
    assert "cascade" not in s_off, \
        "the PR-08 OFF ledger shape must stay the exact pre-lever shape"
    assert s_on["cascade"]["target"] == "tissue" and len(s_on["cascade"]["per_slot"]) == E


def test_policy_a_eviction_reinit_inherits_the_cascade():
    V, seed, E = 11, 0, 4
    torch.manual_seed(3)
    xs, ys = _tiny_data(V)
    model = fp.build_model(V, seed, E, cascade=True, cascade_kappa=0.2, cascade_delta=0.3)
    for s in range(E):
        model.committed[s] = True
        model.mu[s] = 1e-6            # near-zero floors -> z >> Z_NOVEL for every segment
        model.var[s] = 1e-8
        model.n_segments[s] = 1000    # mature: the floor-maturity veto never fires
    with torch.no_grad():
        model.lm(xs)
    led = plc._fresh_ledger()
    boundary = {"segments_total": 0, "to_A_expert": 0, "a_expert": 0}
    plc.route_pr08(model, model._h, ys, "C", led, stream_pos=0, boundary=boundary)
    assert len(led["evictions"]) == 1, "the pool-full novelty must evict (Policy A)"
    victim = led["evictions"][0]["victim"]
    new_head = model.experts[victim]
    assert new_head.cascade and new_head.fast_parameters() is not None, \
        "a fresh head must inherit the model's cascade settings"
    assert new_head.cascade_kappa == 0.2 and new_head.cascade_delta == 0.3
    assert torch.equal(new_head.Wenc_fast, torch.zeros_like(new_head.Wenc_fast))
