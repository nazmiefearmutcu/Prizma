"""Tests for the PR-2026-09-03-08 runner (seq/prizma_lm_claim.py — the Prizma-LM flagship bar).

MINIMAL pure-layer coverage only (the surprise_claim/dfrontier_claim suite pattern): this file
pins
  (a) the slice-pinning helpers EXACTLY (doc §2) incl. the owner-approved C-range repair and
      the A-eval/C-train disjointness invariant that protects bar B1,
  (b) the smoke/powered ledger separation + refusal and the CUDA guard (shared pattern, new
      module),
  (c) the bar math as PURE functions: B1 (one-sample, direction 'below'), B2 (Welch advantage,
      direction 'above'), the Holm family order over B1-B2, the INCONCLUSIVE straddle rule
      (never PASS), the B3 clauses, the B4 report, and the pre-committed §5 branch echoes,
  (d) the budget-projection guard (warn > 2 h) and the frozen protocol constants.
No training, no torch: everything here runs on the pure layer in <1s.
"""
import os
import pytest

from seq import prizma_lm_claim as plc
from seq.stats import t_isf


# ── 1. Slice pinning (doc §2 + the owner-approved C-range repair) ───────────────────────────────

def test_pinned_slices_exact():
    sl = plc.pin_slices()
    assert sl["A_train"] == ("text8", 0, 1_000_000)
    assert sl["A_eval"] == ("text8", 1_000_000, 1_100_000), "A-eval stays PR-07' pinned"
    assert sl["C_retention"] == ("text8", 900_000, 1_000_000), "B2's C-eval slice"
    assert sl["C_train"] == ("text8", 1_100_000, 2_100_000), \
        "owner-approved repair: C skips the A-eval slice, still 1.0M chars"
    assert sl["B_train"] == ("shakespeare", 0.0, 0.9)
    assert sl["B_eval"] == ("shakespeare", 0.9, 1.0), "B-eval = shakespeare last 10%"


def test_repair_invariant_a_eval_disjoint_from_c_train():
    # The doc §2 literal C range would intersect A-eval and corrupt B1; pin_slices must raise
    # rather than let that regression in. Simulate the doc-literal offsets:
    a0, a1 = plc.A_EVAL_START, plc.A_EVAL_END
    c0, c1 = 1_000_000, 2_000_000                     # the doc's literal Block C
    assert max(a0, c0) < min(a1, c1), "sanity: the literal doc range DOES overlap A-eval"
    sl = plc.pin_slices()                              # the runner's pins do not
    _, _, ce = sl["C_train"]
    _, _, ae = sl["A_eval"]
    _, cs, _ = sl["C_train"]
    _, as_, _ = sl["A_eval"]
    assert max(as_, cs) >= min(ae, ce), "pinned C-train must not intersect A-eval"


def test_c_retention_slice_disjoint_from_c_train_and_a_eval():
    sl = plc.pin_slices()
    cr0, cr1 = sl["C_retention"][1], sl["C_retention"][2]
    ct0, ct1 = sl["C_train"][1], sl["C_train"][2]
    ae0, ae1 = sl["A_eval"][1], sl["A_eval"][2]
    assert cr1 <= ct0 and cr1 <= ae0
    # the disclosed prose bug: C-retention IS A-train's tail (both B2 arms saw it in A)
    assert cr0 >= 900_000 and cr1 <= plc.A_CHARS


def test_doc_deviation_notes_are_carried():
    assert "owner-approved" in plc.C_RANGE_REPAIR_NOTE and "addendum" in plc.C_RANGE_REPAIR_NOTE
    assert "A-train's tail" in plc.C_RET_PROSE_NOTE


# ── 2. Ledger separation + refusals (recall_gate pattern) ────────────────────────────────────────

def test_smoke_and_powered_defaults_distinct(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIZMA_RESULTS", str(tmp_path))
    smk = plc._default_results_path(smoke=True)
    pw = plc._default_results_path(smoke=False)
    assert os.path.abspath(smk) != os.path.abspath(pw)
    assert smk.endswith("smoke.json") and "prizma_lm_PR-2026-09-03-08" in pw


def test_smoke_run_refuses_the_powered_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIZMA_RESULTS", str(tmp_path))
    powered = plc._default_results_path(smoke=False)
    with pytest.raises(SystemExit) as ei:
        plc.resolve_results_path(powered, smoke=True)
    assert "refus" in str(ei.value).lower(), "refusal must say why"
    # the deliberate bypass exists and an explicit non-powered --out never trips the guard
    assert plc.resolve_results_path(powered, smoke=True, force_smoke_path=True) == powered
    other = str(tmp_path / "elsewhere.json")
    assert plc.resolve_results_path(other, smoke=True) == other


def test_powered_requires_cuda():
    with pytest.raises(SystemExit) as ei:
        plc.require_cuda(False)
    assert "cuda" in str(ei.value).lower(), "the refusal must name the CUDA requirement"
    plc.require_cuda(True)   # no-op on a CUDA box


def test_cli_rejects_unknown_args_and_requires_one_mode():
    p = plc._build_parser()
    with pytest.raises(SystemExit) as ei:
        p.parse_args(["--smoke", "--extra"])     # typo'd/unknown flag: never launches anything
    assert ei.value.code != 0
    with pytest.raises(SystemExit):
        p.parse_args(["--smoke", "--powered"])   # mutually exclusive modes
    with pytest.raises(SystemExit):
        p.parse_args([])                          # exactly one mode is required
    assert p.parse_args(["--smoke"]).smoke is True
    assert p.parse_args(["--powered"]).powered is True


# ── 3. Frozen protocol constants (pre-reg §2-§4) ────────────────────────────────────────────────

def test_protocol_constants_frozen():
    assert plc.REGISTRY_ID == "PR-2026-09-03-08"
    assert plc.ARMS == ("PRIM-LM", "FROZEN-TRUNK", "SHARED-HEAD", "FROZEN-CHECKPOINT",
                        "FORCED-RECRUIT")
    assert plc.CLAIM_SEEDS == (0, 1, 2, 3, 4)
    assert plc.ARM_FAMILY["PRIM-LM"] == plc.ARM_FAMILY["FORCED-RECRUIT"] == \
        plc.ARM_FAMILY["FROZEN-TRUNK"] == "fusion"
    assert plc.ARM_FAMILY["SHARED-HEAD"] == "shared"
    assert plc.ARM_FAMILY["FROZEN-CHECKPOINT"] == "frozen-checkpoint"
    assert (plc.E_POOL, plc.M_MAX, plc.FREEZE_MIN_SEEN) == (4, 4, 300)
    assert plc.Z_NOVEL == 5.0 and plc.H_SMALL == 64 and plc.H_SHARED == 256
    assert plc.SEG == 256 and plc.BATCH_SEGS == 32 and plc.LR_GRID == (1e-3, 3e-3, 1e-2)
    assert (plc.B1_MARGIN, plc.B2_MARGIN, plc.B3_FRAC_BAR, plc.B3_WINDOW_BATCHES,
            plc.B3_FORCED_COST, plc.ALPHA) == (0.05, 0.10, 0.5, 20, 0.10, 0.05)


# ── 4. Bar math (pure): directions, Holm order, INCONCLUSIVE, branches ──────────────────────────

def _prim(a_preB, a_postC, cret, frac, b_postB, b_postC):
    return {"bpc_A_preB": a_preB, "bpc_A_postC": a_postC, "bpc_Cret_postC": cret,
            "bpc_B_postB": b_postB, "bpc_B_postC": b_postC,
            "boundary_window": {"frac_to_A_expert": frac}}


def _arms(prim_diff, cret_prim, frac, b_postC_prim, b_postC_forced=None):
    # A_preB fixed at 3.0 so each per-seed B1 diff IS the prim_diff element (with spread).
    prim = [_prim(3.0, 3.0 + d, cret_prim + j * 0.001, frac,
                  2.9, b_postC_prim) for j, d in enumerate(prim_diff)]
    frozen_ck = [_prim(3.0, 3.0, 3.30 + j * 0.001, 0.0, 6.0, 6.0) for j in range(5)]
    forced_b = b_postC_forced if b_postC_forced is not None else b_postC_prim - 0.05
    forced = [_prim(3.0, 3.0, cret_prim + 0.15, 0.0, 2.9,
                    forced_b + j * 0.001) for j in range(5)]
    return prim, frozen_ck, forced


def test_b1_pass_b2_pass_b3_pass_claim_pass():
    prim, frozen_ck, forced = _arms(prim_diff=(-0.01, -0.02, 0.0, -0.01, -0.02),
                                    cret_prim=2.80, frac=0.9,
                                    b_postC_prim=4.00, b_postC_forced=3.95)
    v = plc.claim_verdict(prim, frozen_ck, forced)
    assert v["B1"]["status"] == "PASS" and v["B2"]["status"] == "PASS" \
        and v["B3"]["status"] == "PASS"
    assert v["outcome"] == "PASS" and v["branches"] == []
    assert "CLAIM PASS" in v["verdict"] and "B4" in v["verdict"]


def test_b1_direction_below_and_b2_direction_above():
    # B1 ('below'): CI upper must be <= 0.05 for a PASS.
    prim, frozen_ck, forced = _arms(prim_diff=(0.01, 0.0, -0.01, 0.02, 0.0),  # mean ~0.004
                                    cret_prim=2.80, frac=0.9, b_postC_prim=4.0)
    v = plc.claim_verdict(prim, frozen_ck, forced)
    assert v["B1"]["ci"][1] <= 0.05 and v["B1"]["status"] == "PASS"
    # B2 ('above'): PASS needs the advantage CI LOWER >= +0.10.
    prim_hi, frozen_hi, _ = _arms(prim_diff=(-0.01, -0.01, -0.01, -0.01, -0.01),
                                  cret_prim=2.70, frac=0.9, b_postC_prim=4.0)
    v2 = plc.claim_verdict(prim_hi, frozen_hi, forced)
    assert v2["B2"]["delta"] > 0 and v2["B2"]["ci"][0] >= 0.10 and v2["B2"]["status"] == "PASS"


def test_b1_straddle_is_inconclusive_never_pass():
    # mean exactly at the margin, real spread -> the CI straddles 0.05 -> INCONCLUSIVE.
    prim, frozen_ck, forced = _arms(prim_diff=(0.07, 0.03, 0.05, 0.09, 0.01),  # mean 0.05
                                    cret_prim=2.80, frac=0.9, b_postC_prim=4.0)
    v = plc.claim_verdict(prim, frozen_ck, forced)
    assert v["B1"]["mean"] == pytest.approx(0.05)
    assert v["B1"]["ci"][0] < 0.05 < v["B1"]["ci"][1]
    assert v["B1"]["status"] == "INCONCLUSIVE"
    assert v["outcome"] == "INCONCLUSIVE"          # INCONCLUSIVE is never PASS
    assert any("n->10" in b for b in v["branches"]), "the §5 branch must be echoed"


def test_b1_fail_echoes_the_downgrade_branch():
    prim, frozen_ck, forced = _arms(prim_diff=(0.14, 0.10, 0.12, 0.16, 0.08),  # mean 0.12
                                    cret_prim=2.80, frac=0.9, b_postC_prim=4.0)
    v = plc.claim_verdict(prim, frozen_ck, forced)
    assert v["B1"]["ci"][0] > 0.05, "the CI must be entirely on the failing side"
    assert v["B1"]["status"] == "FAIL"
    assert any("single-boundary" in b and "frozen-trunk" in b for b in v["branches"]), \
        "the pre-committed §5 B1-FAIL branch must be echoed"


def test_holm_family_order_over_b1_b2():
    # B1 overwhelming, B2 absent: only B1 survives the Holm family of 2; B2's adjusted p can
    # never drop below its raw p and B2 must not PASS.
    prim, frozen_ck, forced = _arms(prim_diff=(-0.011, -0.009, -0.010, -0.012, -0.008),
                                    cret_prim=3.20, frac=0.9,   # advantage ~0.10 vs 3.30
                                    b_postC_prim=4.0)
    v = plc.claim_verdict(prim, frozen_ck, forced)
    fam = {f["bar"]: f for f in v["holm_family"]}
    assert fam["B1"]["p_raw"] < 0.001 and fam["B1"]["p_holm"] <= fam["B1"]["p_raw"] * 2 + 1e-15
    assert fam["B2"]["p_holm"] >= fam["B2"]["p_raw"]
    assert v["B1"]["status"] == "PASS" and v["B2"]["status"] != "PASS"


def test_b3_clauses_exact_means_no_ci():
    # clause (a): frac < 0.5 -> FAIL even when the forced arm costs nothing.
    prim, frozen_ck, forced = _arms(prim_diff=(-0.01, -0.02, 0.0, -0.01, -0.02),
                                    cret_prim=2.80, frac=0.3,
                                    b_postC_prim=4.00, b_postC_forced=3.95)
    v = plc.claim_verdict(prim, frozen_ck, forced)
    assert v["B3"]["clause_a_ok"] is False and v["B3"]["clause_b_ok"] is True
    assert v["B3"]["status"] == "FAIL" and v["outcome"] == "FAIL"
    assert any("lifelong routing" in b for b in v["branches"]), "the §5 B3 branch must be echoed"
    # clause (b): pattern completion costing PRIM > 0.10 vs the forced counterfactual -> FAIL.
    # (forced BETTER than prim is fine — an idle expert can serve B best, probe-2 measured it.)
    prim2, frozen2, forced2 = _arms(prim_diff=(-0.01, -0.02, 0.0, -0.01, -0.02),
                                    cret_prim=2.80, frac=0.9,
                                    b_postC_prim=4.50, b_postC_forced=4.00)
    v2 = plc.claim_verdict(prim2, frozen2, forced2)
    assert v2["B3"]["clause_a_ok"] is True and v2["B3"]["clause_b_ok"] is False
    assert v2["B3"]["status"] == "FAIL"


def test_b4_reported_not_gated_and_extra_arms_included():
    prim, frozen_ck, forced = _arms(prim_diff=(-0.01, -0.02, 0.0, -0.01, -0.02),
                                    cret_prim=2.80, frac=0.9, b_postC_prim=4.0)
    extra = {"SHARED-HEAD": [_prim(3.0, 3.2, 2.9, 0.0, 2.9, 4.3) for _ in range(5)]}
    v = plc.claim_verdict(prim, frozen_ck, forced, extra_b4=extra)
    assert set(v["B4"]) >= {"PRIM-LM", "FORCED-RECRUIT", "SHARED-HEAD"}
    assert v["B4"]["PRIM-LM"]["per_seed"] == [1.1] * 5
    # B4 never gates: even with degradation present the outcome follows B1-B3 only
    assert v["outcome"] == "PASS"


def test_one_sample_uses_the_shipped_upper_tail_quantiles():
    st = plc._onesample_margin([0.0, 0.02, 0.01, 0.03, 0.04], 0.05)
    tc = t_isf(0.025, 4)
    assert st["ci"][1] == pytest.approx(st["mean"] + tc * st["se"])
    # a mean far BELOW the margin must give a tiny one-sided p (upper-tail convention, flipped
    # onto the advantage scale — this is the sign bug class the sanity run caught)
    assert st["p_raw"] < 0.05


# ── 5. Budget guard ──────────────────────────────────────────────────────────────────────────────

def test_budget_projection_warns_above_two_hours():
    ok = plc.budget_projection(first_cell_s=120, n_cells_total=25, overhead_s=60)
    assert ok["projected_min"] == pytest.approx((120 * 25 + 60) / 60, abs=0.1)
    assert ok["warn"] is False
    slow = plc.budget_projection(first_cell_s=600, n_cells_total=25, overhead_s=60)
    assert slow["projected_min"] > 120 and slow["warn"] is True
    assert "WARNING" in slow["note"]
    none = plc.budget_projection(first_cell_s=0, n_cells_total=25)
    assert none["projected_min"] is None and none["warn"] is False
