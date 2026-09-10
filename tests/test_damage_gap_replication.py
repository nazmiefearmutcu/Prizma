"""Tests for the PR-2026-09-03-20 runner (seq/boundary_gap_replication.py — the
boundary-damage gap replicated on fresh seeds 5-9; a confirmatory analysis of the PR-19
ledger).

Pure-layer coverage: frozen constants, the ledger separation + parser contracts, the
missing-ledger/cell refusals, and the verdict truth table (PASS at CI lower >= 0.25;
NEGATIVE at delta <= 0; NOT-ESTABLISHED between; the branches).
"""
import pytest

from seq import prizma_lm_claim as plc
from seq import boundary_gap_replication as bgr


def test_protocol_constants_pinned():
    assert bgr.REGISTRY_ID == "PR-2026-09-03-20"
    assert bgr.LEDDIR == "damage_gap_repl_PR-2026-09-03-20"
    assert bgr.CLAIM_SEEDS == (0, 1, 2, 3, 4) == plc.CLAIM_SEEDS
    assert bgr.P1_MARGIN == 0.25 and bgr.ALPHA == 0.05
    assert bgr.PR19_LEDGER == "manyblock_PR-2026-09-03-19"


def test_aliases_and_parser():
    assert bgr.needs_cuda is plc.needs_cuda
    p = bgr._build_parser()
    args = p.parse_args(["--powered-cpu"])
    assert args.powered_cpu is True and args.smoke is False and args.powered is False
    with pytest.raises(SystemExit):
        p.parse_args(["--powered-cpu", "--smoke"])


def test_smoke_pointed_at_powered_ledger_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIZMA_RESULTS", str(tmp_path))
    powered = tmp_path / "damage_gap_repl_PR-2026-09-03-20" / "powered.json"
    powered.parent.mkdir(exist_ok=True)
    powered.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        bgr.resolve_results_path(str(powered), smoke=True, force_smoke_path=False)


def test_missing_pr19_ledger_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIZMA_RESULTS", str(tmp_path))
    with pytest.raises(SystemExit, match="PR-19 powered ledger is MISSING"):
        bgr.run(smoke=True, results_path=str(
            tmp_path / "damage_gap_repl_PR-2026-09-03-20" / "smoke.json"))


def test_damages_helper():
    rec = {"bpc_B_postB": 3.0, "bpc_B_postC": 3.9}
    d = bgr.damages([rec], [dict(rec)])
    assert d["plain"] == pytest.approx([0.9]) and d["ex"] == pytest.approx([0.9])


def test_verdict_truth_table():
    # PASS: PLAIN damages ~0.91, EX ~0.60 (the PR-19 descriptives) -> delta ~0.31
    v = bgr.claim_verdict(plain_damage=[0.915, 0.918, 0.949, 0.916, 0.912],
                          ex_damage=[0.601, 0.601, 0.601, 0.601, 0.601])
    assert v["P1"]["delta"] == pytest.approx(0.321, abs=0.001)
    assert v["P1"]["status"] == "PASS" and v["outcome"] == "CLAIMED"
    assert v["P1"]["plain_damage_mean"] == pytest.approx(0.922, abs=0.01)
    # NEGATIVE: PLAIN damages less or equal
    v2 = bgr.claim_verdict(plain_damage=[0.60] * 5, ex_damage=[0.90] * 5)
    assert v2["P1"]["delta"] == pytest.approx(-0.30)
    assert v2["outcome"] == "NEGATIVE"
    assert "downgraded" in v2["verdict"]
    # NOT-ESTABLISHED: positive but tiny gap, tight -> CI upper < 0.25
    v3 = bgr.claim_verdict(plain_damage=[0.62] * 5, ex_damage=[0.60] * 5)
    assert v3["outcome"] == "NOT-ESTABLISHED"
    assert any("10-14" in b for b in v3["branches"])
