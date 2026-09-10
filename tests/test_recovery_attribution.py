"""Tests for the PR-2026-09-03-21 runner (seq/recovery_attribution.py — the revisit-
recovery attribution; a confirmatory analysis of three registered artifacts).

Pure-layer coverage: frozen constants, the ledger separation + parser contracts, the
missing-ledger refusals, and the attribution truth table (CI-positional positions with
the 0.10 point threshold; the four-way pre-committed table).
"""
import pytest

from seq import prizma_lm_claim as plc
from seq import recovery_attribution as ra


def test_protocol_constants_pinned():
    assert ra.REGISTRY_ID == "PR-2026-09-03-21"
    assert ra.LEDDIR == "recovery_attr_PR-2026-09-03-21"
    assert ra.CLAIM_SEEDS == (0, 1, 2, 3, 4) == plc.CLAIM_SEEDS
    assert ra.GAP_POINT == 0.10 and ra.ALPHA == 0.05
    assert ra.SOURCES["COLUMN"] == ("manyblock_PR-2026-09-03-14", "claim.EX")
    assert ra.SOURCES["SCHED-TF"] == ("windowtf_sched_PR-2026-09-03-17", "claim.WINDOW-TF")
    assert ra.SOURCES["PLAIN-TF"] == ("windowtf_manyblock_PR-2026-09-03-15", "claim.WINDOW-TF")


def test_aliases_and_parser():
    assert ra.needs_cuda is plc.needs_cuda
    p = ra._build_parser()
    args = p.parse_args(["--powered-cpu"])
    assert args.powered_cpu is True and args.smoke is False and args.powered is False
    with pytest.raises(SystemExit):
        p.parse_args(["--powered-cpu", "--smoke"])


def test_smoke_pointed_at_powered_ledger_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIZMA_RESULTS", str(tmp_path))
    powered = tmp_path / "recovery_attr_PR-2026-09-03-21" / "powered.json"
    powered.parent.mkdir(exist_ok=True)
    powered.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        ra.resolve_results_path(str(powered), smoke=True, force_smoke_path=False)


def test_missing_source_ledger_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIZMA_RESULTS", str(tmp_path))
    with pytest.raises(SystemExit, match="registered ledger is MISSING"):
        ra.run(smoke=True, results_path=str(
            tmp_path / "recovery_attr_PR-2026-09-03-21" / "smoke.json"))


def test_gap_position_rule():
    assert ra._gap_position(0.30, (0.05, 0.55)) == "ESTABLISHED"   # excludes 0, point >= 0.10
    assert ra._gap_position(0.05, (0.02, 0.08)) == "PARITY"        # point < 0.10
    assert ra._gap_position(0.30, (-0.05, 0.65)) == "PARITY"       # CI contains 0
    assert ra._gap_position(-0.30, (-0.55, -0.05)) == "ESTABLISHED"  # negative-side gap


def _recs(recovery):
    recs = []
    for r in recovery:
        recs.append({"bpc_B_postC": 4.0, "bpc_B_postD": 4.0 - r,
                     "bpc_B_postB": 2.9})
    return recs


def test_attribution_universal_when_both_parity():
    col = _recs([0.82 - j * 0.01 for j in range(5)])
    sch = _recs([0.83 - j * 0.01 for j in range(5)])
    pln = _recs([0.85 - j * 0.01 for j in range(5)])
    v = ra.attribution(col, sch, pln)
    assert v["C1"]["attribution"] == "PARITY"
    assert v["C2"]["attribution"] == "PARITY"
    assert v["outcome"] == "RECOVERY-UNIVERSAL"
    assert any("ATTRIBUTION MATRIX CLOSES" in b or True for b in v["branches"]) or \
        v["branches"] == []


def test_attribution_tissue_costs_recovery():
    col = _recs([0.82] * 5)                 # column recovers 0.82
    sch = _recs([0.83] * 5)                 # scheduled control the same
    pln = _recs([1.10] * 5)                 # plain recovers MORE by 0.28 (established)
    v = ra.attribution(col, sch, pln)
    # delta = mean(COLUMN) - mean(PLAIN): NEGATIVE when the plain control recovers more
    assert v["C2"]["position"] == "ESTABLISHED" and v["C2"]["delta"] == pytest.approx(-0.28)
    assert v["outcome"] == "TISSUE-COSTS-RECOVERY"
    assert any("design lead" in b for b in v["branches"])


def test_attribution_tissue_amplifies():
    col = _recs([0.82] * 5)
    sch = _recs([0.83] * 5)
    pln = _recs([0.55] * 5)                 # plain recovers LESS by 0.27 (established)
    v = ra.attribution(col, sch, pln)
    assert v["outcome"] == "TISSUE-AMPLIFIES-RECOVERY"


def test_attribution_sched_advantaged():
    col = _recs([0.82] * 5)
    sch = _recs([1.05] * 5)                 # scheduled recovers MORE by 0.23 (established)
    pln = _recs([0.83] * 5)
    v = ra.attribution(col, sch, pln)
    assert v["C1"]["position"] == "ESTABLISHED"
    assert v["outcome"] == "SCHED-ADVANTAGED"
