"""Tests for the PR-2026-09-03-09 runner (seq/floorfreeze_claim.py — the floor-freeze
routing repair, PR-08 B3a's registered successor).

MINIMAL pure-layer coverage (the prizma_lm_runner suite pattern), plus the registered
off-identity pins:
  (a) the inherited slice/protocol constants pinned to the PR-08 values (PR-09 reuses
      plc.pin_slices BY IMPORT and freezes LR at 3e-3 with NO lr-selection leg),
  (b) the smoke/powered ledger separation + refusal and the CLI parser contract,
  (c) the §3 bar math as PURE functions: P1 (exact means), G1 (one-sample 'below'),
      G2 (Welch advantage 'above' vs a fake PR-08 FROZEN-CHECKPOINT list), the Holm family
      order over [G1, G2], the INCONCLUSIVE straddle rule (never PASS), and the §5
      overall-outcome truth table (CLAIMED / NEGATIVE-GUARD / NEGATIVE / INCONCLUSIVE),
  (d) the §4 canary comparator's ignore-set (wall_s + cell-loop wrapper keys) vs science
      fields, with a synthetic mismatch detected,
  (e) OFF-IDENTITY PINS (behavioral, tiny torch tensors — no training): fp._expert_train is
      a no-op on the floors when the `floor_freeze` attribute is absent (and the freeze
      branch pins floors while the expert still trains when present), and plc.route_pr08's
      eviction re-init leaves any freeze set via the guarded discard (a set-safe
      remove-if-present; a dict-style pop(x, None) would TypeError on the registered
      plain-set freeze — disclosed deviation from the task snippet).
Pure tests run without torch in <1s; the two behavioral tests import torch lazily.
"""
import os
import pytest

from seq import floorfreeze_claim as ffc
from seq import prizma_lm_claim as plc


# ── 1. Inherited slice/protocol constants (PR-08 §2 verbatim, reused BY IMPORT) ─────────────────

def test_slices_are_plc_pin_slices_reused_by_import():
    assert ffc.pin_slices is plc.pin_slices, "PR-09 must REUSE PR-08's pinned-slice table"
    sl = ffc.pin_slices()
    assert sl == plc.pin_slices()
    assert sl["A_train"] == ("text8", 0, 1_000_000)
    assert sl["A_eval"] == ("text8", 1_000_000, 1_100_000)
    assert sl["C_retention"] == ("text8", 900_000, 1_000_000)
    assert sl["C_train"] == ("text8", 1_100_000, 2_100_000), "PR-08's C-range repair inherited"
    assert sl["B_train"] == ("shakespeare", 0.0, 0.9)
    assert sl["B_eval"] == ("shakespeare", 0.9, 1.0)


def test_protocol_constants_inherited_and_frozen():
    assert ffc.REGISTRY_ID == "PR-2026-09-03-09"
    assert ffc.LEDDIR == "floorfreeze_PR-2026-09-03-09"
    assert ffc.ARMS == ("OFF", "FROZEN-FLOORS")
    assert ffc.CLAIM_SEEDS == (0, 1, 2, 3, 4)
    assert ffc.LR_FROZEN == 3e-3, "doc §2: LR frozen at 3e-3 for BOTH arms (inherited)"
    # tissue + stream constants are the PR-08 OBJECTS (by import), pinned to PR-08's values
    assert ffc.E_POOL is plc.E_POOL and ffc.E_POOL == 4
    assert ffc.M_MAX is plc.M_MAX and ffc.M_MAX == 4
    assert ffc.FREEZE_MIN_SEEN is plc.FREEZE_MIN_SEEN and ffc.FREEZE_MIN_SEEN == 300
    assert ffc.Z_NOVEL is plc.Z_NOVEL and ffc.Z_NOVEL == 5.0
    assert ffc.H_SMALL is plc.H_SMALL and ffc.H_SMALL == 64
    assert ffc.FLOOR_EMA is plc.FLOOR_EMA and ffc.FLOOR_EMA == 0.05
    assert ffc.SEG is plc.SEG and ffc.SEG == 256
    assert ffc.BATCH_SEGS is plc.BATCH_SEGS and ffc.BATCH_SEGS == 32
    assert ffc.B3_WINDOW_BATCHES is plc.B3_WINDOW_BATCHES and ffc.B3_WINDOW_BATCHES == 20
    # bars (doc §3, exact)
    assert (ffc.P1_FRAC_BAR, ffc.G1_MARGIN, ffc.G2_MARGIN, ffc.ALPHA) == (0.5, 0.05, 0.10, 0.05)
    assert ffc.HOLM_FAMILY == ("G1", "G2"), "P1 is means-based, untested (PR-08 B3 convention)"


# ── 2. Ledger separation + refusals + CLI parser (plc pattern) ──────────────────────────────────

def test_smoke_and_powered_defaults_distinct(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIZMA_RESULTS", str(tmp_path))
    smk = ffc._default_results_path(smoke=True)
    pw = ffc._default_results_path(smoke=False)
    assert os.path.abspath(smk) != os.path.abspath(pw)
    assert smk.endswith("smoke.json") and "floorfreeze_PR-2026-09-03-09" in pw
    assert "floorfreeze_PR-2026-09-03-09" in smk


def test_smoke_run_refuses_the_powered_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIZMA_RESULTS", str(tmp_path))
    powered = ffc._default_results_path(smoke=False)
    with pytest.raises(SystemExit) as ei:
        ffc.resolve_results_path(powered, smoke=True)
    assert "refus" in str(ei.value).lower(), "refusal must say why"
    assert ffc.resolve_results_path(powered, smoke=True, force_smoke_path=True) == powered
    other = str(tmp_path / "elsewhere.json")
    assert ffc.resolve_results_path(other, smoke=True) == other


def test_powered_requires_cuda():
    with pytest.raises(SystemExit) as ei:
        ffc.require_cuda(False)
    assert "cuda" in str(ei.value).lower(), "the refusal must name the CUDA requirement"
    ffc.require_cuda(True)   # no-op on a CUDA box


def test_needs_cuda_is_the_plc_contract_reused():
    assert ffc.needs_cuda is plc.needs_cuda, "doc §6: reuse plc.needs_cuda, do not fork it"
    assert ffc.needs_cuda("powered") is True
    assert ffc.needs_cuda("powered-cpu") is False
    assert ffc.needs_cuda("smoke") is False


def test_cli_rejects_unknown_args_and_requires_one_mode():
    p = ffc._build_parser()
    with pytest.raises(SystemExit) as ei:
        p.parse_args(["--smoke", "--extra"])     # typo'd/unknown flag: never launches anything
    assert ei.value.code != 0
    with pytest.raises(SystemExit):
        p.parse_args(["--smoke", "--powered"])   # mutually exclusive modes
    with pytest.raises(SystemExit):
        p.parse_args([])                          # exactly one mode is required
    assert p.parse_args(["--smoke"]).smoke is True
    assert p.parse_args(["--powered"]).powered is True
    assert p.parse_args(["--smoke", "--force-smoke-path"]).force_smoke_path is True


def test_powered_cpu_alone_accepted_and_combinations_rejected():
    p = ffc._build_parser()
    args = p.parse_args(["--powered-cpu"])
    assert args.powered_cpu is True and args.smoke is False and args.powered is False
    with pytest.raises(SystemExit):
        p.parse_args(["--powered-cpu", "--smoke"])
    with pytest.raises(SystemExit):
        p.parse_args(["--powered-cpu", "--powered"])
    with pytest.raises(SystemExit):
        p.parse_args(["--powered-cpu", "--smoke", "--powered"])


# ── 3. Verdict math (pure): P1 / G1 / G2 / Holm / §5 truth table ────────────────────────────────

def _rec(frac, a_preB, a_postC, cret, b_postB, b_postC, seed=0):
    # models the REAL stored-cell schema: the routing snapshot nests under rec["ledger"]
    return {"config": "PRIM-LM", "E": 4, "seed": seed, "lr": 3e-3,
            "bpc_A_preB": a_preB, "bpc_A_postC": a_postC, "bpc_Cret_postC": cret,
            "bpc_B_postB": b_postB, "bpc_B_postC": b_postC,
            "fgt_A_full": a_preB - a_postC,
            "b_degradation_B_postC": b_postC - b_postB,
            "a_expert": 0,
            "ledger": {"boundary_window": {"frac_to_A_expert": frac},
                       "recruits": [{"slot": 0}], "evictions": [],
                       "vetoed_novel_segments": 0}}


def _arms(*, ff_fracs=(0.55,) * 5, off_fracs=(0.43,) * 5,
          ff_diffs=(-0.01, -0.02, 0.0, -0.01, -0.02),
          ff_cret=(2.70, 2.71, 2.69, 2.72, 2.68),
          ck_cret=(3.30, 3.31, 3.29, 3.32, 3.28)):
    """Default fixture = every bar PASSes (the CLAIMED cell of the truth table)."""
    off = [_rec(f, 3.0, 2.99, 2.80, 2.9, 4.0, seed=s) for s, f in enumerate(off_fracs)]
    ff = [_rec(f, 3.0, 3.0 + d, c, 2.9, 4.0, seed=s)
          for s, (f, d, c) in enumerate(zip(ff_fracs, ff_diffs, ff_cret))]
    ck = [_rec(0.0, 3.0, 3.0, c, 6.0, 6.0, seed=s) for s, c in enumerate(ck_cret)]
    return off, ff, ck


def test_p1_exact_means_pass_and_fail_and_control_recorded():
    off, ff, ck = _arms(ff_fracs=(0.55,) * 5, off_fracs=(0.43,) * 5)
    v = ffc.claim_verdict(off, ff, ck)
    assert v["P1"]["frac_mean"] == pytest.approx(0.55)
    assert v["P1"]["status"] == "PASS"
    assert v["P1"]["off_frac_mean"] == pytest.approx(0.43), "OFF frac is the recorded control"
    assert v["P1"]["frac_per_seed"] == [0.55] * 5 and v["P1"]["off_frac_per_seed"] == [0.43] * 5
    # below the bar -> FAIL (exact means, no CI — the PR-08 B3 form)
    off2, ff2, ck2 = _arms(ff_fracs=(0.45,) * 5)
    v2 = ffc.claim_verdict(off2, ff2, ck2)
    assert v2["P1"]["frac_mean"] == pytest.approx(0.45)
    assert v2["P1"]["status"] == "FAIL"


def test_g1_pass_fail_straddle():
    # PASS: diffs mean -0.012, CI upper well below 0.05.
    off, ff, ck = _arms(ff_diffs=(-0.01, -0.02, 0.0, -0.01, -0.02))
    v = ffc.claim_verdict(off, ff, ck)
    assert v["G1"]["mean"] == pytest.approx(-0.012)
    assert v["G1"]["ci"][1] <= 0.05 and v["G1"]["status"] == "PASS"
    # FAIL: CI entirely ABOVE the margin (mean 0.12).
    off2, ff2, ck2 = _arms(ff_diffs=(0.14, 0.10, 0.12, 0.16, 0.08))
    v2 = ffc.claim_verdict(off2, ff2, ck2)
    assert v2["G1"]["ci"][0] > 0.05 and v2["G1"]["status"] == "FAIL"
    # STRADDLE: mean exactly at the margin with real spread -> INCONCLUSIVE, never PASS.
    off3, ff3, ck3 = _arms(ff_diffs=(0.07, 0.03, 0.05, 0.09, 0.01))
    v3 = ffc.claim_verdict(off3, ff3, ck3)
    assert v3["G1"]["mean"] == pytest.approx(0.05)
    assert v3["G1"]["ci"][0] < 0.05 < v3["G1"]["ci"][1]
    assert v3["G1"]["status"] == "INCONCLUSIVE"


def test_g2_pass_fail_straddle_against_fake_pr08_frozen_ckpt():
    # PASS: advantage ~+0.60 bpc, CI lower >= 0.10 (default fixture).
    off, ff, ck = _arms()
    v = ffc.claim_verdict(off, ff, ck)
    assert v["G2"]["delta"] == pytest.approx(0.60)
    assert v["G2"]["ci"][0] >= 0.10 and v["G2"]["status"] == "PASS"
    # FAIL: negative advantage, CI upper < 0.10.
    off2, ff2, ck2 = _arms(ff_cret=(3.35, 3.36, 3.34, 3.37, 3.33))
    v2 = ffc.claim_verdict(off2, ff2, ck2)
    assert v2["G2"]["delta"] < 0 and v2["G2"]["ci"][1] < 0.10
    assert v2["G2"]["status"] == "FAIL"
    # STRADDLE: advantage mean +0.20 with enough spread that the CI contains 0.10.
    off3, ff3, ck3 = _arms(ff_cret=(3.0, 3.2, 2.8, 3.4, 2.6),
                           ck_cret=(3.2, 3.0, 3.4, 2.8, 3.6))
    v3 = ffc.claim_verdict(off3, ff3, ck3)
    assert v3["G2"]["delta"] == pytest.approx(0.20)
    assert v3["G2"]["ci"][0] < 0.10 < v3["G2"]["ci"][1]
    assert v3["G2"]["status"] == "INCONCLUSIVE"


def test_holm_family_order_over_g1_g2():
    # G1 overwhelming, G2 absent: only G1 survives the Holm family of 2; G2's adjusted p can
    # never drop below its raw p and G2 must not PASS. Family ORDER is [G1, G2].
    off, ff, ck = _arms(ff_diffs=(-0.011, -0.009, -0.010, -0.012, -0.008),
                        ff_cret=(3.28, 3.32, 3.30, 3.26, 3.34),   # advantage ~0 vs ~3.30
                        ck_cret=(3.28, 3.32, 3.30, 3.26, 3.34))
    v = ffc.claim_verdict(off, ff, ck)
    fam = v["holm_family"]
    assert [f["bar"] for f in fam] == ["G1", "G2"]
    assert fam[0]["p_raw"] < 0.001 and fam[0]["p_holm"] <= fam[0]["p_raw"] * 2 + 1e-15
    assert fam[1]["p_holm"] >= fam[1]["p_raw"]
    assert v["G1"]["status"] == "PASS" and v["G2"]["status"] != "PASS"


def test_overall_truth_table_claimed_negative_guard_negative_inconclusive():
    # CLAIMED: P1 PASS + G1 PASS + G2 PASS.
    off, ff, ck = _arms()
    v = ffc.claim_verdict(off, ff, ck)
    assert v["outcome"] == "CLAIMED" and "trunk-floor decoupling" in v["verdict"]
    # NEGATIVE-GUARD: P1 PASS but G1 FAIL (the lever buys routing at the cost of the column).
    off2, ff2, ck2 = _arms(ff_diffs=(0.14, 0.10, 0.12, 0.16, 0.08))
    v2 = ffc.claim_verdict(off2, ff2, ck2)
    assert v2["P1"]["status"] == "PASS" and v2["G1"]["status"] == "FAIL"
    assert v2["outcome"] == "NEGATIVE-GUARD"
    assert "G1" in v2["verdict"] and "Floor scheduling" in v2["verdict"]
    assert any("floor scheduling" in b for b in v2["branches"])
    # NEGATIVE: P1 FAIL (guards intact — floor freeze insufficient).
    off3, ff3, ck3 = _arms(ff_fracs=(0.45,) * 5)
    v3 = ffc.claim_verdict(off3, ff3, ck3)
    assert v3["P1"]["status"] == "FAIL" and v3["outcome"] == "NEGATIVE"
    assert any("trunk-lr scheduling" in b for b in v3["branches"])
    # INCONCLUSIVE: a guard straddles — even with P1 PASS (INCONCLUSIVE is never PASS).
    off4, ff4, ck4 = _arms(ff_diffs=(0.07, 0.03, 0.05, 0.09, 0.01))
    v4 = ffc.claim_verdict(off4, ff4, ck4)
    assert v4["G1"]["status"] == "INCONCLUSIVE" and v4["outcome"] == "INCONCLUSIVE"
    assert any("n=10" in b and "PRE-AUTHORIZED" in b for b in v4["branches"])
    # INCONCLUSIVE via G2 as well.
    off5, ff5, ck5 = _arms(ff_cret=(3.0, 3.2, 2.8, 3.4, 2.6), ck_cret=(3.2, 3.0, 3.4, 2.8, 3.6))
    v5 = ffc.claim_verdict(off5, ff5, ck5)
    assert v5["G2"]["status"] == "INCONCLUSIVE" and v5["outcome"] == "INCONCLUSIVE"


def test_b4_churn_and_per_seed_fracs_reported_not_gated():
    off, ff, ck = _arms()
    v = ffc.claim_verdict(off, ff, ck)
    # B4 trunk drift per arm: bpc_B_postC - bpc_B_postB = 4.0 - 2.9 = 1.1 per seed.
    assert v["B4"]["OFF"]["per_seed"] == [1.1] * 5
    assert v["B4"]["FROZEN-FLOORS"]["per_seed"] == [1.1] * 5
    # churn per arm from the ledger event streams.
    assert v["churn"]["OFF"]["s0"]["recruits"] == 1
    assert v["churn"]["OFF"]["s0"]["evictions"] == 0
    assert v["churn"]["OFF"]["s0"]["vetoed_novel_segments"] == 0
    assert v["churn"]["FROZEN-FLOORS"]["s4"]["recruits"] == 1
    # P1 carries per-seed fracs for BOTH arms.
    assert v["P1"]["frac_per_seed"] == [0.55] * 5 and v["P1"]["off_frac_per_seed"] == [0.43] * 5


# ── 4. The §4 canary comparator: ignore-set vs science fields ───────────────────────────────────

def test_canary_ignore_set_is_exactly_the_registered_one():
    assert ffc.CANARY_IGNORE == frozenset(
        {"wall_s", "arm", "cellkey", "cfgsig", "complete", "forced_placement"})


def test_canary_identical_records_have_no_mismatches():
    off, _, _ = _arms()
    pr08 = dict(off[0])
    pr08.update({"arm": "PRIM-LM", "cellkey": "claim.PRIM-LM.s0",
                 "cfgsig": "deadbeef", "complete": True, "wall_s": 61.0})
    fresh = dict(off[0])
    fresh.update({"arm": "OFF", "cellkey": "claim.OFF.s0",
                  "cfgsig": "cafebabe", "complete": True, "wall_s": 60.0})
    assert ffc.canary_mismatches(fresh, pr08) == []


def test_canary_detects_synthetic_science_mismatch_but_ignores_wrappers():
    off, _, _ = _arms()
    pr08 = dict(off[0])
    pr08.update({"arm": "PRIM-LM", "cellkey": "claim.PRIM-LM.s0", "cfgsig": "x",
                 "complete": True, "wall_s": 61.0})
    fresh = dict(off[0])
    fresh.update({"arm": "OFF", "cellkey": "claim.OFF.s0", "cfgsig": "y",
                  "complete": True, "wall_s": 60.0})
    # a 1-ULP drift in ANY bpc float is a canary failure...
    drifted = dict(fresh)
    drifted["bpc_A_postC"] = fresh["bpc_A_postC"] + 1e-12
    assert "bpc_A_postC" in ffc.canary_mismatches(drifted, pr08)
    # ...including inside the nested routing ledger (the boundary window is P1's quantity).
    drifted2 = dict(fresh)
    drifted2["ledger"] = {"boundary_window": {"frac_to_A_expert": 0.999},
                          "recruits": [], "evictions": [], "vetoed_novel_segments": 0}
    assert "ledger" in ffc.canary_mismatches(drifted2, pr08)
    # science labels (config/E/seed/lr) and a_expert/fgt/b_degradation are compared too.
    drifted3 = dict(fresh)
    drifted3["a_expert"] = 2
    assert "a_expert" in ffc.canary_mismatches(drifted3, pr08)
    # ...while EVERY ignored wrapper/wall field can differ freely.
    wrappers = dict(fresh)
    wrappers.update({"wall_s": 999.0, "arm": "totally-different", "cellkey": "other",
                     "cfgsig": "other", "complete": False,
                     "forced_placement": {"mode": "policy_a_eviction"}})
    assert ffc.canary_mismatches(wrappers, pr08) == []


# ── 5. OFF-IDENTITY PINS (behavioral; tiny torch tensors, no training loop) ─────────────────────

def _tiny_expert(vocab=5):
    import torch
    from torch import nn

    class _E(nn.Module):
        """PCExpertHead-shaped: Wenc (d->h, tanh) + Wdec (h->vocab)."""

        def __init__(self):
            super().__init__()
            self.Wenc = nn.Linear(4, 3)
            self.Wdec = nn.Linear(3, vocab)

        def forward(self, h):
            return self.Wdec(torch.tanh(self.Wenc(h)))

    return _E()


def _fake_model(*, freeze=None, vocab=5, calibrated=False):
    """Minimal fake FusionLM surface: mu/var are plain lists (the guarded branches read them
    as lists); experts are tiny nn.Modules so the real local-optimizer step runs."""
    import torch
    from torch import nn

    class _Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, vocab)

        def forward(self, x):
            return self.lin(x)

    class _LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = _Head()

    class _M:
        def __init__(self):
            self.lm = _LM()
            self.experts = [_tiny_expert(vocab=vocab) for _ in range(4)]
            self.E = 4
            self.committed = [True] * 4
            # calibrated=True -> mid-stream EMA state; else the 1e9 first-calibration sentinel
            self.mu = ([3.0] * 4 if calibrated else [1e9] * 4)
            self.var = ([0.25] * 4 if calibrated else [1.0] * 4)
            self.n_batches = [0] * 4
            self.n_segments = [1000] * 4          # >= FREEZE_MIN_SEEN: no floor-maturity veto
            self.ce_sum = [0.0] * 4
            if freeze is not None:
                self.floor_freeze = freeze        # a plain Python set, as the runner pins it

        def n_committed(self):
            return sum(self.committed)

    return _M()


def _h_y(B=2, T=3, vocab=5, seed=7):
    import torch
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(B, T, 4, generator=g)
    y = torch.randint(0, vocab, (B, T), generator=g)
    return h, y


def test_expert_train_off_identity_no_attribute_floors_still_update():
    """THE off-identity pin: with the `floor_freeze` attribute ABSENT, _expert_train behaves
    exactly as PR-08 — the floor update runs (first-calibration here) and nothing else
    changed (n_batches/n_segments bookkeeping + a real expert optimizer step)."""
    from seq import fusion_probe as fp

    m = _fake_model()                          # NO floor_freeze attribute
    assert not hasattr(m, "floor_freeze")
    h, y = _h_y()
    before_w = m.experts[0].Wdec.weight.detach().clone()
    r = fp._expert_train(m, 0, h, y, [0, 1], 1e-2)
    assert isinstance(r, float)
    assert m.mu[0] < 1e8, "attribute absent -> the first-calibration floor update must run"
    assert m.var[0] == pytest.approx(max(1e-4, (0.1 * m.mu[0]) ** 2))
    assert m.n_batches[0] == 1 and m.n_segments[0] == 1002 and m.ce_sum[0] > 0.0
    assert not m.experts[0].Wdec.weight.detach().equal(before_w), "expert still trains"


def test_expert_train_freeze_pins_floors_but_expert_still_trains():
    """The treatment path: a frozen slot's mu/var are EXACTLY untouched while the expert's
    optimizer step and the n_batches/n_segments/ce_sum bookkeeping still run; a NON-frozen
    slot in the same model keeps updating its floor."""
    from seq import fusion_probe as fp

    m = _fake_model(freeze={0}, calibrated=True)   # mid-stream EMA state, slot 0 frozen
    h, y = _h_y()
    before_w = m.experts[0].Wdec.weight.detach().clone()
    fp._expert_train(m, 0, h, y, [0, 1], 1e-2)
    assert m.mu[0] == 3.0 and m.var[0] == 0.25, "frozen floor must be pinned EXACTLY"
    assert m.n_batches[0] == 1 and m.n_segments[0] == 1002 and m.ce_sum[0] > 0.0
    assert not m.experts[0].Wdec.weight.detach().equal(before_w), \
        "only FLOORS are frozen — the expert predictor/optimizer must still learn"
    fp._expert_train(m, 1, h, y, [0, 1], 1e-2)     # slot 1 is NOT in the freeze set
    assert m.mu[1] != 3.0 and m.n_batches[1] == 1


def test_route_pr08_eviction_pops_the_victim_from_the_freeze_set():
    """Policy-A eviction under a full pool: the re-initialized victim LEAVES the freeze set
    (guarded discard), the other slots stay frozen, and the re-init itself is unchanged."""
    m = _fake_model(freeze={0, 1, 2, 3})
    m.mu, m.var = [0.0] * 4, [1e-4] * 4   # floors that FIRE novelty: z = CE/0.01 >> 5
    h, y = _h_y(B=4)
    ledger = plc._fresh_ledger()
    plc.route_pr08(m, h, y, "C", ledger, stream_pos=100)
    assert len(ledger["evictions"]) == 1, "novelty under a full pool must evict (Policy A)"
    victim = ledger["evictions"][0]["victim"]
    assert victim == 3, "equal shares -> the HIGHEST slot index is the victim"
    assert victim not in m.floor_freeze and m.floor_freeze == {0, 1, 2}
    assert m.mu[victim] == 1e9 and m.var[victim] == 1.0, "re-init unchanged by the lever"
    assert m.n_segments[victim] == 0 and m.n_batches[victim] == 0


def test_route_pr08_discard_of_an_absent_slot_is_a_noop():
    m = _fake_model(freeze={0, 1, 2})                    # victim 3 was never frozen
    m.mu, m.var = [0.0] * 4, [1e-4] * 4                  # floors that FIRE novelty
    h, y = _h_y(B=4)
    ledger = plc._fresh_ledger()
    plc.route_pr08(m, h, y, "C", ledger, stream_pos=100)
    assert m.floor_freeze == {0, 1, 2}, "remove-if-present: an unfrozen victim changes nothing"


def test_route_pr08_off_identity_no_attribute_no_crash_no_attribute_created():
    m = _fake_model()                                    # NO floor_freeze attribute
    m.mu, m.var = [0.0] * 4, [1e-4] * 4                  # floors that FIRE novelty
    h, y = _h_y(B=4)
    ledger = plc._fresh_ledger()
    plc.route_pr08(m, h, y, "C", ledger, stream_pos=100)
    assert len(ledger["evictions"]) == 1
    assert not hasattr(m, "floor_freeze"), "the guarded branch must not conjure the attribute"
