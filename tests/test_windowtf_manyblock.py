"""Tests for the PR-2026-09-03-15 runner (seq/windowtf_manyblock.py — the memory-matched
WINDOW-TF control on the 5-block curriculum).

Pure-layer coverage (the manyblock suite pattern): frozen constants, the ledger separation
+ parser contracts, the P1 verdict math (PASS/FAIL/INCONCLUSIVE by CI position with the
single-primary Holm note), and the reused-baseline dependency (missing PR-14 ledger
refuses before anything runs).
"""
import pytest

from seq import prizma_lm_claim as plc
from seq import manyblock_claim as mbc
from seq import windowtf_manyblock as wtm


def test_protocol_constants_pinned():
    assert wtm.REGISTRY_ID == "PR-2026-09-03-15"
    assert wtm.LEDDIR == "windowtf_manyblock_PR-2026-09-03-15"
    assert wtm.ARMS == ("WINDOW-TF",)
    assert wtm.CLAIM_SEEDS == (0, 1, 2, 3, 4) == plc.CLAIM_SEEDS
    assert wtm.LR_FROZEN == 3e-3
    assert wtm.P1_MARGIN == 0.10 and wtm.ALPHA == 0.05
    assert wtm.PR14_EX_A_RETENTION == pytest.approx(-0.3226)
    assert wtm.BLOCKS == ("A", "B", "C", "D", "E")


def test_aliases_are_true_reuses():
    assert wtm.needs_cuda is plc.needs_cuda
    assert wtm.needs_cuda("powered") is True
    assert wtm.needs_cuda("powered-cpu") is False
    assert wtm.needs_cuda("smoke") is False


def test_smoke_pointed_at_powered_ledger_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIZMA_RESULTS", str(tmp_path))
    powered = tmp_path / "windowtf_manyblock_PR-2026-09-03-15" / "powered.json"
    powered.parent.mkdir(exist_ok=True)
    powered.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        wtm.resolve_results_path(str(powered), smoke=True, force_smoke_path=False)


def test_missing_pr14_ledger_refuses(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIZMA_RESULTS", str(tmp_path))
    (tmp_path / "windowtf_manyblock_PR-2026-09-03-15").mkdir(exist_ok=True)
    with pytest.raises(SystemExit, match="PR-14 powered ledger is MISSING"):
        wtm.run(smoke=True,
                results_path=str(tmp_path / "windowtf_manyblock_PR-2026-09-03-15" / "smoke.json"))


def test_parser_contracts():
    p = wtm._build_parser()
    args = p.parse_args(["--powered-cpu"])
    assert args.powered_cpu is True and args.smoke is False and args.powered is False
    with pytest.raises(SystemExit):
        p.parse_args(["--powered-cpu", "--smoke"])
    with pytest.raises(SystemExit):
        p.parse_args(["--nope"])


def _tf_rec(ret):
    return {"bpc_A_preB": 5.5, "bpc_A_postE": 5.5 + ret,
            "bpc_B_postB": 3.0, "bpc_B_postC": 4.0, "bpc_B_postD": 2.9, "bpc_B_postE": 3.9,
            "bpc_Cret_postC": 2.9, "bpc_Cret_postE": 2.8}


# PR-14 EX reused baseline: A-retention = -0.3226 mean (the registered value)

def test_p1_pass_when_tf_forgets_way_more():
    tf = [_tf_rec(ret=-3.5) for _ in range(5)]     # TF retention -3.5 vs column -0.32
    v = wtm.claim_verdict(tf, [plc and -0.3226 + j * 0.001 for j in range(5)])
    assert v["P1"]["status"] == "PASS" and v["outcome"] == "CLAIMED"
    assert v["P1"]["delta"] == pytest.approx((-0.3226) - (-3.5), abs=0.01),         "delta = mean(column) - mean(TF): POSITIVE when the TF forgets more"


def test_p1_fail_when_tf_holds_comparably():
    tf = [_tf_rec(ret=-0.30) for _ in range(5)]    # TF holds A like the column
    v = wtm.claim_verdict(tf, [-0.3226 + j * 0.001 for j in range(5)])
    assert v["P1"]["status"] == "FAIL" and v["outcome"] == "NEGATIVE"
    assert "does not generalize" in v["verdict"]


def test_p1_inconclusive_straddle():
    tf = [_tf_rec(ret=r) for r in (-0.9, -0.6, -0.4, -0.2, 0.0)]   # mean -0.42:
    # delta = column - tf ~ +0.097 with a wide CI straddling the 0.10 margin -> INCONCLUSIVE
    v = wtm.claim_verdict(tf, [-0.3226 for _ in range(5)])
    assert v["P1"]["status"] == "INCONCLUSIVE"
    assert v["outcome"] == "INCONCLUSIVE"
    assert any("PRE-AUTHORIZED" in b for b in v["branches"])


# ── PR-2026-09-03-17: scheduled-TF attribution (doc §3-§4) ──────────────────────────────────

def test_scheduled_flag_parser_and_fingerprint():
    p = wtm._build_parser()
    args = p.parse_args(["--powered-cpu", "--post-a-lr", "0.0075",
                         "--ledger-dir", "windowtf_sched_PR-2026-09-03-17"])
    assert args.post_a_lr == 0.0075
    assert args.ledger_dir == "windowtf_sched_PR-2026-09-03-17"
    assert p.parse_args(["--powered-cpu"]).post_a_lr is None


def test_attribution_truth_table():
    col = [0.64] * 5                                   # the column's damage (PR-14 EX)
    plain = [1.01] * 5                                 # the plain TF's damage (PR-15)
    sched_better = [0.70] * 5                          # schedule alone mostly closes it
    att = wtm.attribution(sched_better, plain, col)
    assert att["C1_schedule_effect"]["position"] == "ESTABLISHED"
    assert att["attribution"] == "SCHEDULE-CARRIED"

    sched_flat = [0.98] * 5                            # schedule does nothing
    att2 = wtm.attribution(sched_flat, plain, col)
    assert att2["C1_schedule_effect"]["position"] != "ESTABLISHED"
    assert att2["C2_tissue_residual"]["position"] == "ESTABLISHED"
    assert att2["attribution"] == "TISSUE-CARRIED"

    sched_half = [0.82 - j * 0.02 for j in range(5)]   # schedule closes ~half; wide spread
    att3 = wtm.attribution(sched_half, plain, col)
    assert att3["attribution"] in ("COMPOUND", "TISSUE-CARRIED", "SCHEDULE-CARRIED",
                                   "UNRESOLVED at n=5 (doc §4: the n=10 seed extension "
                                   "is pre-authorized)")
    assert att3["C1_schedule_effect"]["ci"][0] < att3["C1_schedule_effect"]["ci"][1],         "a degenerate (zero-width) CI means the fixture has no spread"


def test_attribution_unresolved_when_narrow():
    col = [0.64] * 5
    plain = [1.01] * 5
    sched = [0.85 - j * 0.02 for j in range(5)]        # delta ~ +0.20, tight -> unresolved
    att = wtm.attribution(sched, plain, col)
    assert att["attribution"].startswith("UNRESOLVED at n=5")
