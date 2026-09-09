"""Tests for the PR-2026-09-03-16 runner (seq/boundary_damage_claim.py — the B-boundary
drift-damage head-to-head; a confirmatory analysis of registered artifacts).

Pure-layer coverage: frozen constants, the ledger separation + parser contracts, the
missing-ledger refusals (both sources), and the verdict truth table (PASS at CI lower
>= 0.25; NOT-ESTABLISHED at CI upper < 0.25; NEGATIVE at delta <= 0; the branches).
"""
import pytest

from seq import prizma_lm_claim as plc
from seq import boundary_damage_claim as bdc


def test_protocol_constants_pinned():
    assert bdc.REGISTRY_ID == "PR-2026-09-03-16"
    assert bdc.LEDDIR == "boundary_damage_PR-2026-09-03-16"
    assert bdc.CLAIM_SEEDS == (0, 1, 2, 3, 4) == plc.CLAIM_SEEDS
    assert bdc.P1_MARGIN == 0.25 and bdc.ALPHA == 0.05
    assert bdc.PR14_EX_LEDGER[0] == "manyblock_PR-2026-09-03-14"
    assert bdc.PR14_EX_LEDGER[1] == "claim.EX"
    assert bdc.PR15_TF_LEDGER[0] == "windowtf_manyblock_PR-2026-09-03-15"
    assert bdc.PR15_TF_LEDGER[1] == "claim.WINDOW-TF"


def test_aliases_and_parser():
    assert bdc.needs_cuda is plc.needs_cuda
    p = bdc._build_parser()
    args = p.parse_args(["--powered-cpu"])
    assert args.powered_cpu is True and args.smoke is False and args.powered is False
    with pytest.raises(SystemExit):
        p.parse_args(["--powered-cpu", "--smoke"])


def test_smoke_pointed_at_powered_ledger_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIZMA_RESULTS", str(tmp_path))
    powered = tmp_path / "boundary_damage_PR-2026-09-03-16" / "powered.json"
    powered.parent.mkdir(exist_ok=True)
    powered.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        bdc.resolve_results_path(str(powered), smoke=True, force_smoke_path=False)


def test_missing_source_ledger_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIZMA_RESULTS", str(tmp_path))
    with pytest.raises(SystemExit, match="registered ledger is MISSING"):
        bdc._damages("manyblock_PR-2026-09-03-14", "claim.EX")


def test_verdict_truth_table():
    # PASS: control damage exceeds column damage by a wide, tight margin
    v = bdc.claim_verdict(col_damage=[0.60] * 5, ctl_damage=[1.15] * 5)
    assert v["P1"]["status"] == "PASS" and v["outcome"] == "CLAIMED"
    assert v["P1"]["delta"] == pytest.approx(0.55)
    # NEGATIVE: the control damages less
    v2 = bdc.claim_verdict(col_damage=[1.15] * 5, ctl_damage=[0.60] * 5)
    assert v2["outcome"] == "NEGATIVE"
    assert any("misleading" in b for b in v2["branches"])
    # NOT-ESTABLISHED: a positive but tiny gap (n=5 tight -> CI upper < 0.25)
    v3 = bdc.claim_verdict(col_damage=[0.60] * 5, ctl_damage=[0.63] * 5)
    assert v3["outcome"] == "NOT-ESTABLISHED"
    assert any("n=10" in b for b in v3["branches"])
