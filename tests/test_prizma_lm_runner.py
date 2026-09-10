"""Tests for the PR-2026-09-03-08 runner (seq/prizma_lm_claim.py — the Prizma-LM flagship bar).

MINIMAL pure-layer coverage only (the surprise_claim/dfrontier_claim suite pattern): this file
pins
  (a) the slice-pinning helpers EXACTLY (doc §2) incl. the maintainer-addendum C-range repair and
      the A-eval/C-train disjointness invariant that protects bar B1,
  (b) the smoke/powered ledger separation + refusal and the CUDA guard (shared pattern, new
      module),
  (c) the bar math as PURE functions: B1 (one-sample, direction 'below'), B2 (Welch advantage,
      direction 'above'), the Holm family order over B1-B2, the INCONCLUSIVE straddle rule
      (never PASS), the B3 clauses, the B4 report, and the pre-committed §5 branch echoes,
  (d) the budget-projection guard (warn > 2 h) and the frozen protocol constants,
  (e) the 2026-09-08 pre-GPU-review additions: the --powered-cpu mode's parser contract
      (third exclusive mode; doc addendum 2026-09-08 #2) and the needs_cuda pure guard.
No training, no torch: everything here runs on the pure layer in <1s.
"""
import os
import pytest

from seq import prizma_lm_claim as plc
from seq.stats import t_isf


# ── 1. Slice pinning (doc §2 + the maintainer-addendum C-range repair) ─────────────────────────

def test_pinned_slices_exact():
    sl = plc.pin_slices()
    assert sl["A_train"] == ("text8", 0, 1_000_000)
    assert sl["A_eval"] == ("text8", 1_000_000, 1_100_000), "A-eval stays PR-07' pinned"
    assert sl["C_retention"] == ("text8", 900_000, 1_000_000), "B2's C-eval slice"
    assert sl["C_train"] == ("text8", 1_100_000, 2_100_000), \
        "maintainer-addendum repair: C skips the A-eval slice, still 1.0M chars"
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
    # wording corrected 2026-09-08: the C-range repair is a MAINTAINER addendum (no owner
    # decision occurred) — see the doc's dated addendum and the pre-GPU review (H-1).
    assert "maintainer addendum" in plc.C_RANGE_REPAIR_NOTE
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
    # fixtures model the REAL cell schema: the routing snapshot nests under rec["ledger"]
    # (the first powered-cpu execution caught claim_verdict reading the bare key — a path
    # smoke never reaches)
    return {"bpc_A_preB": a_preB, "bpc_A_postC": a_postC, "bpc_Cret_postC": cret,
            "bpc_B_postB": b_postB, "bpc_B_postC": b_postC,
            "ledger": {"boundary_window": {"frac_to_A_expert": frac}}}


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


# ── 6. Pre-GPU-review additions (2026-09-08): the --powered-cpu mode + needs_cuda ────────────────

def test_powered_cpu_alone_accepted_and_combinations_rejected():
    # The doc section-6 CPU-feasible fallback (doc addendum 2026-09-08 #2) ships as a THIRD,
    # mutually exclusive mode: alone it parses; combined with --smoke or --powered it must be
    # rejected before anything runs (the parser-guard contract).
    p = plc._build_parser()
    args = p.parse_args(["--powered-cpu"])
    assert args.powered_cpu is True and args.smoke is False and args.powered is False
    with pytest.raises(SystemExit):
        p.parse_args(["--powered-cpu", "--smoke"])
    with pytest.raises(SystemExit):
        p.parse_args(["--powered-cpu", "--powered"])
    with pytest.raises(SystemExit):
        p.parse_args(["--powered-cpu", "--smoke", "--powered"])


def test_needs_cuda_only_for_the_powered_mode():
    # The pure mode->CUDA decision (unit-testable without torch): --powered keeps its CUDA
    # refusal byte-identical; --powered-cpu (the doc section-6 fallback) and --smoke are
    # CPU-side by design.
    assert plc.needs_cuda("powered") is True
    assert plc.needs_cuda("powered-cpu") is False
    assert plc.needs_cuda("smoke") is False


# ── 7. FORCED-RECRUIT pool-full contingency (2026-09-08 doc addendum #3; review M-1 fired) ───────

class _StubModel:
    """Just the committed/n_segments surface _forced_placement reads."""

    def __init__(self, committed, n_segments):
        self.committed = committed
        self.n_segments = n_segments


def test_forced_placement_free_slot_is_the_registered_semantics():
    model = _StubModel([True, True, False, False], [500, 300, 0, 0])
    slot, placement = plc._forced_placement(model, E=4, stream_pos=7800)
    assert slot == 2, "the FIRST free slot (left-to-right fill) — byte-identical to the " \
                      "registered free[0] rule"
    assert placement == {"mode": "free_slot"}


def test_forced_placement_pool_full_evicts_lowest_share():
    # e0 carries 60% of the stream, e1 30%, e2 10%, e3 0%+1 -> the lowest share (e3) is the
    # victim, mirroring route_pr08's Policy A exactly.
    model = _StubModel([True, True, True, True], [6000, 3000, 1000, 0])
    slot, placement = plc._forced_placement(model, E=4, stream_pos=10000)
    assert slot == 3
    assert placement["mode"] == "policy_a_eviction"
    assert placement["victim"] == 3
    assert placement["victim_n_segments"] == 0
    assert placement["m_max"] == plc.M_MAX
    assert placement["train_shares_at_fire"]["e0"] == 0.6
    assert placement["at_batch"] == 10000


def test_forced_placement_tie_goes_to_the_highest_slot():
    # Equal shares -> the tie-break picks the HIGHEST slot index (the probe's recency proxy),
    # the same key route_pr08 uses: min by (share, -slot).
    model = _StubModel([True, True, True, True], [2500, 2500, 2500, 2500])
    slot, placement = plc._forced_placement(model, E=4, stream_pos=0)
    assert slot == 3 and placement["victim"] == 3


def test_forced_placement_respects_the_m_max_cap():
    # Committed slots beyond M_MAX are not eviction candidates (cap = min(M_MAX, E) window).
    model = _StubModel([True] * 6, [100, 100, 1, 100, 100, 100])
    slot, placement = plc._forced_placement(model, E=6, stream_pos=0)
    assert slot == 2, "the near-empty slot INSIDE the M_MAX=4 window is the victim; slots " \
                      "4-5 are outside the cap and never selected"


# ── 8. PR-2026-09-03-11 additions (2026-09-09): the guarded lever flags default to None ─────
# ADD-ONLY section: the PR-11 repaired-flagship mode (--trunk-lr-c / --ledger-dir) must leave
# every existing PR-08 parser default untouched. Deeper PR-11 coverage lives in
# tests/test_pr11_repaired_flagship.py.

def test_pr11_flags_default_to_none_so_pr08_defaults_are_unchanged():
    p = plc._build_parser()
    args = p.parse_args(["--powered-cpu"])
    assert args.trunk_lr_c is None, "no --trunk-lr-c => the lever is OFF (byte-identical PR-08)"
    assert args.ledger_dir is None, "no --ledger-dir => the PR-08 default ledger dir (LEDDIR)"


# ── 9. PR-2026-09-03-18 replication mode: --seed-offset ─────────────────────────────────────

def test_seed_offset_parser_defaults_and_fingerprint_sensitivity():
    p = plc._build_parser()
    assert p.parse_args(["--powered-cpu"]).seed_offset == 0
    args = p.parse_args(["--powered-cpu", "--seed-offset", "5"])
    assert args.seed_offset == 5
    payload = {"leg": "claim", "arm": "PRIM-LM", "family": "fusion", "seed": 5, "lr": 3e-3,
               "smoke": False, "vocab": 65, "smoke_segs": None, "seg": 256, "batch_segs": 32,
               "slices": {}, "stream_lengths": {}, "tissue": {}, "bars": {},
               "trunk_lr_c": None, "domain_exclusion": False}
    assert plc._fp({**payload, "seed_offset": 0}) != plc._fp({**payload, "seed_offset": 5}), \
        "fresh-seed replication cells must never resume from original-seed cells"


def test_registered_lrs_canary_constants():
    # the PR-18 LR-selection canary's expected values (kept in sync with the runner)
    import seq.prizma_lm_claim as m
    expected = {"fusion": 3e-3, "shared": 3e-3, "frozen-checkpoint": 1e-2}
    assert m.LR_GRID == (1e-3, 3e-3, 1e-2)
    assert set(m.ARM_FAMILY.values()) == {"fusion", "shared", "frozen-checkpoint"}
    assert m.ARM_FAMILY["PRIM-LM"] == "fusion"
    assert m.ARM_FAMILY["SHARED-HEAD"] == "shared"
    assert m.ARM_FAMILY["FROZEN-CHECKPOINT"] == "frozen-checkpoint"
    assert expected["fusion"] == m.LR_GRID[1] and expected["shared"] == m.LR_GRID[1]
    assert expected["frozen-checkpoint"] == m.LR_GRID[2]
