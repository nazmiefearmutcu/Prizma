"""Tests for the PR-02 registered-grid runner (seq/dfrontier_claim.py — PR-2026-09-03-02).

Covers ONLY what is NEW relative to the sibling runner's suite (tests/test_surprise_gate.py already
pins the shared refusal/checksum patterns for seq.surprise_claim): this file pins
  (a) the new module's smoke/powered ledger separation + refusal and the CUDA guard,
  (b) the CLI arg-guard (unknown/typo'd flag never launches the powered grid),
  (c) the frozen protocol constants (doc §4: rungs, seeds, d_phi, bar, K1 range, solve rule),
  (d) the law math + verdict as PURE functions: sigma2_law vs the committed instrument artifact,
      N*(eps=1)=48.06 at the doc's quad2 sigma2_law, fit-eps formula, K1/K2 ordering, the >=3-of-4
      bar, K3 monotonicity, the eps consistency demotion, and freeze-then-verify resume integrity.
No training, no torch: everything here is importable and runs on the pure layer in <1s.
"""
import json
import os
import pytest

from seq import dfrontier_claim as dc


# ── 1. Ledger separation + refusals (recall_gate pattern, new module) ──────────────────────────

def test_smoke_and_powered_defaults_distinct(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIZMA_RESULTS", str(tmp_path))
    smk = dc._default_results_path(smoke=True)
    pw = dc._default_results_path(smoke=False)
    assert os.path.abspath(smk) != os.path.abspath(pw)
    assert smk.endswith("smoke.json") and "dfrontier_PR-2026-09-03-02" in pw


def test_smoke_run_refuses_the_powered_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIZMA_RESULTS", str(tmp_path))
    powered = dc._default_results_path(smoke=False)
    with pytest.raises(SystemExit) as ei:
        dc.resolve_results_path(powered, smoke=True)
    assert "refus" in str(ei.value).lower(), "refusal must say why"
    # the deliberate bypass exists and an explicit non-powered --out never trips the guard
    assert dc.resolve_results_path(powered, smoke=True, force_smoke_path=True) == powered
    other = str(tmp_path / "elsewhere.json")
    assert dc.resolve_results_path(other, smoke=True) == other


def test_powered_requires_cuda():
    with pytest.raises(SystemExit) as ei:
        dc.require_cuda(False)
    assert "cuda" in str(ei.value).lower(), "the refusal must name the CUDA requirement"
    dc.require_cuda(True)   # no-op on a CUDA box


def test_cli_rejects_unknown_args_and_requires_one_mode():
    p = dc._build_parser()
    with pytest.raises(SystemExit) as ei:
        p.parse_args(["--smoke", "--extra"])     # typo'd/unknown flag: never launches anything
    assert ei.value.code != 0
    with pytest.raises(SystemExit):
        p.parse_args(["--smoke", "--powered"])   # mutually exclusive modes
    with pytest.raises(SystemExit):
        p.parse_args([])                          # exactly one mode is required
    assert p.parse_args(["--smoke"]).smoke is True
    assert p.parse_args(["--powered"]).powered is True


# ── 2. Frozen protocol constants (docs/crosstalk_capacity_law.md §4) ───────────────────────────

def test_protocol_constants_frozen():
    assert dc.REGISTRY_ID == "PR-2026-09-03-02"
    assert dc.PRIMARY_ARM == "quad2"
    assert dc.ARMS == ("quad2", "quad2_lowrank", "none")
    assert dc.D_PHI == {"quad2": 256, "quad2_lowrank": 137, "none": 32}
    assert dc.FIT_RUNGS == (16, 32, 64) and dc.ADJ_RUNGS == (96, 128, 192, 256)
    assert dc.GRID == (16, 32, 64, 96, 128, 192, 256)
    assert dc.CLAIM_SEEDS == (0, 1, 2), "doc §4: seeds {0,1,2}, identical across arms and rungs"
    assert not set(dc.RERUN_SEEDS) & set(dc.CLAIM_SEEDS), "K3 fresh seeds are disjoint from claim seeds"
    assert dc.SOLVE_THRESH == 0.90 and abs(dc.SOLVE_FRAC - 2 / 3) < 1e-12
    assert dc.PASS_MIN_AGREE == 3 and len(dc.ADJ_RUNGS) == 4, "the frozen bar: >= 3 of 4"
    assert dc.K1_LO == 0.25 and dc.K1_HI == 4.0
    assert dc.STEP_CAP == 80000, "identical budgets across rungs: the B1 budget at EVERY rung"
    assert dc.SCALE[0] // dc.SCALE[2] == 32, "d_h = d_model/n_heads = 32 (the instrument geometry)"


def test_feat_kw_produce_the_frozen_d_phi():
    # quad2: 32 + 224 = 256; lowrank: feat_rank=0 -> r=14 -> 32 + 105 = 137; none: 32
    assert dc.ARM_PRIZMA_KW["quad2"] == dict(feat_map="quad2", feat_n2=224)
    assert dc.ARM_PRIZMA_KW["quad2_lowrank"] == dict(feat_map="quad2_lowrank")
    assert dc.ARM_PRIZMA_KW["none"] == dict(feat_map="none")


# ── 3. The law math + instrument artifact agreement (pure) ─────────────────────────────────────

def test_sigma2_law_formula_and_committed_artifact_agreement():
    entry = {"mu_signed": 0.006388, "sigma2_signed_fluct": 0.145774}
    assert dc.sigma2_law_from_entry(entry) == pytest.approx(0.14591, abs=2e-5)
    # the COMMITTED instrument artifact must reproduce the doc §4 frozen constants — the runner
    # refuses to start otherwise, and this test pins that invariant from the test suite side.
    law = dc.load_sigma2_law()   # default path: <repo>/results/feat_map_probe.json
    for arm in dc.ARMS:
        assert law[arm] == pytest.approx(dc.FROZEN_SIGMA2_LAW[arm], abs=dc.SIGMA2_TOLERANCE)


def test_load_sigma2_law_refuses_drift(tmp_path):
    bad = tmp_path / "feat_map_probe.json"
    bad.write_text(json.dumps({"quad2": {"mu_signed": 0.5, "sigma2_signed_fluct": 0.5},
                               "quad2_lowrank": {"mu_signed": 0.016, "sigma2_signed_fluct": 0.157},
                               "none": {"mu_signed": 0.00003, "sigma2_signed_fluct": 0.177}}))
    with pytest.raises(SystemExit) as ei:
        dc.load_sigma2_law(str(bad))
    assert "drift" in str(ei.value).lower(), "instrument drift must STOP, not silently re-fit"


def test_n_star_law_at_doc_numbers():
    # doc §3.4's honest open question: at eps=1 the quad2 numbers give N* ~ 48 << the D=128 PASS.
    # (The doc's illustrative 48.06 uses the signed-fluctuation sigma alone; the §4 law quantity
    # sigma2_law = sqrt(sig_W^2 + mu^2) = 0.14591 gives 47.97 — the two readings agree to 0.2%.)
    assert dc.n_star_of(1.0, 0.14591, 256) == pytest.approx(48.0, abs=0.2)
    # none @ d_phi=32: rank cap binds -> N* = 32 < 96 -> predicted to fail ALL adjudication rungs
    assert dc.n_star_of(1.0, 0.17692, 32) == 32.0
    assert dc.predicted_solves(1.0, 0.17692, 32) == {96: False, 128: False, 192: False, 256: False}


def test_rung_solved_is_the_two_thirds_rule():
    assert dc.rung_solved([0.95, 0.91, 0.50])                    # 2/3 >= 0.9
    assert not dc.rung_solved([0.95, 0.50, 0.50])                # 1/3
    assert dc.rung_solved([0.90, 0.90, 0.50])                    # exactly 0.90 counts (>= thresh)
    assert dc.rung_solved([0.899, 0.99, 0.99])                   # 0.899 misses, the other 2 clear
    assert dc.rung_solved([0.95], n_seeds=1)                     # smoke: ceil(2/3 * 1) = 1 of 1
    assert not dc.rung_solved([0.89], n_seeds=1)


def test_fit_epsilon_exact_formula():
    assert dc.fit_epsilon(64, 0.14591) == pytest.approx(0.14591 * (63 ** 0.5), rel=1e-12)
    assert dc.fit_epsilon(None, 0.14591) is None, "unidentifiable fit -> no eps (K2)"


def test_evaluate_kill_k2_before_k1():
    fit = dc.FIT_RUNGS
    solved_all = {(dc.PRIMARY_ARM, D): True for D in fit}
    solved_none = {(dc.PRIMARY_ARM, D): False for D in fit}
    # eps computable at the ceiling (0.14591*sqrt(63) ~ 1.158, IN range) but K2 fires FIRST
    d_star_all = dc.largest_solved_fit_rung({D: True for D in fit})
    eps_ok = dc.fit_epsilon(d_star_all, 0.14591)
    k = dc.evaluate_kill(solved_all, {dc.PRIMARY_ARM: d_star_all}, {dc.PRIMARY_ARM: eps_ok})
    assert k["outcome"] == "INCOMPLETE_K2" and k["k2_fired"] and not k["k1_fired"]
    k = dc.evaluate_kill(solved_none, {dc.PRIMARY_ARM: None}, {dc.PRIMARY_ARM: None})
    assert k["outcome"] == "INCOMPLETE_K2"
    # K1: eps outside [0.25, 4.0] -> DEMOTED
    k = dc.evaluate_kill({(dc.PRIMARY_ARM, D): D < 32 for D in fit},
                         {dc.PRIMARY_ARM: 16}, {dc.PRIMARY_ARM: 0.14591 * (15 ** 0.5)})
    assert k["outcome"] is None   # 0.14591*sqrt(15) ~ 0.565: alive
    k = dc.evaluate_kill({(dc.PRIMARY_ARM, D): D < 32 for D in fit},
                         {dc.PRIMARY_ARM: 16}, {dc.PRIMARY_ARM: 9.9})
    assert k["outcome"] == "DEMOTED_K1" and k["k1_fired"]
    # split fit tier, eps in range -> alive
    k = dc.evaluate_kill({(dc.PRIMARY_ARM, D): D < 64 for D in fit},
                         {dc.PRIMARY_ARM: 32}, {dc.PRIMARY_ARM: 0.14591 * (31 ** 0.5)})
    assert k["outcome"] is None and not k["k1_fired"] and not k["k2_fired"]


def test_eps_consistency_demote_over_two_times():
    ok = dc.eps_consistency({dc.PRIMARY_ARM: 1.0, "quad2_lowrank": 1.4, "none": 1.1})
    assert ok["demote_to_per_map"] is False
    dem = dc.eps_consistency({dc.PRIMARY_ARM: 1.0, "quad2_lowrank": 2.5, "none": 1.1})
    assert dem["demote_to_per_map"] is True and "DEMOTED" in dem["note"]
    unk = dc.eps_consistency({dc.PRIMARY_ARM: None, "quad2_lowrank": 1.0, "none": 1.0})
    assert unk["demote_to_per_map"] is True   # not evaluable -> honest per-map fallback


# ── 4. K3 monotonicity + the frozen adjudication bar (pure) ────────────────────────────────────

def test_monotone_violations_detects_solved_high_failed_low():
    solved = {("quad2", 16): True, ("quad2", 64): False, ("quad2", 96): False,
              ("quad2", 128): True, ("none", 16): False, ("none", 128): False}
    v = dc.monotone_violations(solved, (16, 64, 96, 128))
    assert ("quad2", 96, 128) in v, "fails 96 but solves 128 -> violation"
    assert ("quad2", 64, 128) in v
    assert all(x[0] != "none" for x in v), "monotone arm must not be flagged"
    assert dc.monotone_violations({("quad2", 16): True, ("quad2", 96): False}, (16, 96)) == []


def test_grid_distance_steps_and_none():
    assert dc.grid_distance(96, 128) == 1
    assert dc.grid_distance(96, 256) == 3
    assert dc.grid_distance(128, 128) == 0
    assert dc.grid_distance(None, 128) is None and dc.grid_distance(96, None) is None
    assert dc.grid_distance(8, 12, grid=(8, 12)) == 1, "measures within the stage's own rung set"


def test_adjudicate_arm_agreement_and_transition():
    pred = {96: True, 128: True, 192: False, 256: False}
    obs_same = {96: True, 128: True, 192: False, 256: False}
    a = dc.adjudicate_arm(obs_same, pred, dc.ADJ_RUNGS)
    assert a["agreement_count"] == 4 and a["pass"] and a["transition_within_plus_minus_1"]
    obs_one_off = {96: True, 128: True, 192: True, 256: False}   # observed transition 1 step later
    a = dc.adjudicate_arm(obs_one_off, pred, dc.ADJ_RUNGS)
    assert a["agreement_count"] == 3 and a["pass"]
    assert a["transition_distance_grid_steps"] == 1 and a["transition_within_plus_minus_1"]
    obs_two_off = {96: False, 128: False, 192: False, 256: False}  # transition 2 steps earlier
    a = dc.adjudicate_arm(obs_two_off, pred, dc.ADJ_RUNGS)
    assert a["agreement_count"] == 2 and not a["pass"] and a["transition_distance_grid_steps"] == 2
    assert not a["transition_within_plus_minus_1"]
    # unidentified arm: no frozen prediction -> reported, never fabricated
    a = dc.adjudicate_arm(obs_same, None, dc.ADJ_RUNGS)
    assert a["identified"] is False and a["pass"] is None


def test_dfrontier_verdict_outcomes():
    frozen = {"predicted_solved": {
        "quad2": {96: True, 128: True, 192: False, 256: False},
        "quad2_lowrank": {96: True, 128: True, 192: False, 256: False},
        "none": {96: False, 128: False, 192: False, 256: False}}}
    obs_pass = {("quad2", D): D in (96, 128) for D in dc.ADJ_RUNGS}
    obs_pass.update({("quad2_lowrank", D): D in (96, 128) for D in dc.ADJ_RUNGS})
    obs_pass.update({("none", D): False for D in dc.ADJ_RUNGS})
    v = dc.dfrontier_verdict(obs_pass, frozen, dc.ADJ_RUNGS)
    assert v["outcome"] == "PASS" and v["per_arm"]["quad2"]["agreement_count"] == 4
    assert v["none_sanity"]["deeper_falsification_flag"] is False

    obs_k4 = dict(obs_pass)
    obs_k4[("quad2", 96)] = False; obs_k4[("quad2", 128)] = False   # agreement 2/4
    v = dc.dfrontier_verdict(obs_k4, frozen, dc.ADJ_RUNGS)
    assert v["outcome"] == "FALSIFIED_K4" and v["per_arm"]["quad2"]["agreement_count"] == 2

    obs_k3 = dict(obs_pass)
    obs_k3[("quad2", 16)] = False   # a fit-rung failure under a solved adjudication rung
    v = dc.dfrontier_verdict(obs_k3, frozen, dc.ADJ_RUNGS, k3_persisted=True)
    assert v["outcome"] == "INCOMPLETE_K3"

    v = dc.dfrontier_verdict(obs_pass, {"predicted_solved": {"quad2": None}}, dc.ADJ_RUNGS)
    assert v["outcome"] == "INCOMPLETE_K2" and v["per_arm"]["quad2"]["identified"] is False

    # the none-arm deeper-falsification flag: none solves a >=96 rung
    obs_deep = dict(obs_pass)
    obs_deep[("none", 96)] = True
    v = dc.dfrontier_verdict(obs_deep, frozen, dc.ADJ_RUNGS)
    assert v["none_sanity"]["deeper_falsification_flag"] is True
    assert v["outcome"] == "PASS", "the sanity flag is recorded but does NOT gate the verdict"


# ── 5. Freeze-then-verify resume integrity (pure ledger fixtures) ──────────────────────────────

def _ledger_with_fit_cells(accs_by_arm):
    """A minimal fake ledger: fit cells keyed exactly as the runner keys them."""
    res = {}
    for arm, per_D in accs_by_arm.items():
        for D, accs in per_D.items():
            for seed, a in zip(dc.CLAIM_SEEDS, accs):
                res[f"fit.{arm}.D{D}.s{seed}"] = {"best": a, "cfgsig": "x"}
    return res


def test_freeze_record_and_verify_roundtrip():
    accs = {arm: {16: [0.95, 0.94, 0.96], 32: [0.95, 0.93, 0.94], 64: [0.30, 0.31, 0.29]}
            for arm in dc.ARMS}
    res = _ledger_with_fit_cells(accs)
    solved = {(arm, D): dc.rung_solved(a) for arm, per_D in accs.items() for D, a in per_D.items()}
    d_star = {arm: dc.largest_solved_fit_rung({D: solved[(arm, D)] for D in dc.FIT_RUNGS})
              for arm in dc.ARMS}
    law = {arm: dc.FROZEN_SIGMA2_LAW[arm] for arm in dc.ARMS}
    eps = {arm: dc.fit_epsilon(d_star[arm], law[arm]) for arm in dc.ARMS}
    kill = dc.evaluate_kill(solved, d_star, eps)
    frozen = dc.freeze_record(
        {(arm, D): accs[arm][D] for arm in dc.ARMS for D in dc.FIT_RUNGS},
        solved, d_star, eps, law, kill, dc.eps_consistency(eps),
        dc.FIT_RUNGS, dc.ADJ_RUNGS, smoke=False)
    assert frozen["d_star_fit"] == {arm: 32 for arm in dc.ARMS}
    assert frozen["predicted_solved"]["quad2"] == {96: False, 128: False, 192: False, 256: False}
    assert frozen["frozen_at"] and "BEFORE any adjudication" in frozen["what"]
    dc.verify_frozen_against_ledger(res, frozen, law, dc.CLAIM_SEEDS)   # must not raise
    # tamper with the frozen eps -> refuse (a freeze belongs to exactly one fit tier)
    bad = json.loads(json.dumps(frozen))
    bad["eps_hat"]["quad2"] = 3.14
    with pytest.raises(SystemExit):
        dc.verify_frozen_against_ledger(res, bad, law, dc.CLAIM_SEEDS)
    # tamper with the ledger's fit cells -> refuse (the freeze no longer matches the ledger):
    # flip D32 below the solve bar on 2 of 3 seeds -> D*_fit drops 32 -> 16 -> eps_hat changes.
    bad_res = _ledger_with_fit_cells(accs)
    bad_res["fit.quad2.D32.s0"]["best"] = 0.10
    bad_res["fit.quad2.D32.s1"]["best"] = 0.10
    with pytest.raises(SystemExit):
        dc.verify_frozen_against_ledger(bad_res, frozen, law, dc.CLAIM_SEEDS)
