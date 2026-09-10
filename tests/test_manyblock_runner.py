"""Tests for the PR-2026-09-03-14 runner (seq/manyblock_claim.py — many-block accumulation
with a true revisit).

Pure-layer coverage (the trunklr suite pattern): frozen constants, the ledger separation +
parser contracts, the owner-election truth table, the verdict math with the doc addendum
#1 semantics (Holm over [P1, P3] only; G1/G2 CI-positional guards with NO small-p gate),
the overall truth table with the doc §4 branches, and the route_pr08 owner_slot precedence
(behavioral, tiny torch): the folding redirects protected segments to the OWNER slot even
when it differs from the legacy a_expert key.
"""
import pytest

from seq import prizma_lm_claim as plc
from seq import floorfreeze_claim as ffc
from seq import domainexc_claim as dec
from seq import manyblock_claim as mbc


# ── 1. Frozen constants + inherited aliases ─────────────────────────────────────────────────

def test_protocol_constants_pinned():
    assert mbc.REGISTRY_ID == "PR-2026-09-03-14"
    assert mbc.LEDDIR == "manyblock_PR-2026-09-03-14"
    assert mbc.ARMS == ("PLAIN", "EX")
    assert mbc.CLAIM_SEEDS == (0, 1, 2, 3, 4) == plc.CLAIM_SEEDS
    assert mbc.LR_FROZEN == 3e-3 and mbc.TRUNK_LR_SCALE == 0.25
    assert mbc.BACKBONE_LR_POST_A == 7.5e-4
    assert (mbc.P1_MARGIN, mbc.P2_FRAC_BAR, mbc.P3_MARGIN, mbc.G1_ALLOWANCE,
            mbc.G2_MARGIN, mbc.ALPHA) == (0.10, 0.5, 0.05, -0.10, 0.05, 0.05)
    assert mbc.HOLM_FAMILY == ("P1", "P3"), "doc addendum #1: G1/G2 are CI-based guards"
    assert mbc.DOMAIN_OF == {"A": "text8", "B": "shakes", "C": "text8",
                             "D": "shakes", "E": "text8"}


def test_aliases_are_true_reuses():
    assert mbc.needs_cuda is plc.needs_cuda
    assert mbc.canary_mismatches is ffc.canary_mismatches
    assert mbc.needs_cuda("powered") is True
    assert mbc.needs_cuda("powered-cpu") is False


# ── 2. Ledger separation + parser ───────────────────────────────────────────────────────────

def test_smoke_pointed_at_powered_ledger_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIZMA_RESULTS", str(tmp_path))
    powered = tmp_path / "manyblock_PR-2026-09-03-14" / "powered.json"
    powered.parent.mkdir(exist_ok=True)
    powered.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        mbc.resolve_results_path(str(powered), smoke=True, force_smoke_path=False)


def test_parser_contracts():
    p = mbc._build_parser()
    args = p.parse_args(["--powered-cpu"])
    assert args.powered_cpu is True and args.smoke is False and args.powered is False
    with pytest.raises(SystemExit):
        p.parse_args(["--powered-cpu", "--smoke"])
    with pytest.raises(SystemExit):
        p.parse_args(["--nonsense"])


# ── 3. Owner election (pure) ────────────────────────────────────────────────────────────────

def test_owner_election_truth_table():
    committed = [True, True, True, True]
    # shakes history lives on slot 1 (the B-expert) -> the D revisit elects slot 1
    counts = [{"text8": 5000, "shakes": 0}, {"text8": 0, "shakes": 4000},
              {"text8": 0, "shakes": 0}, {"text8": 0, "shakes": 0}]
    assert mbc.elect_owner("shakes", counts, committed) == 1
    assert mbc.elect_owner("text8", counts, committed) == 0
    # no history for the domain -> no owner (first encounter routes freely)
    counts2 = [{"text8": 0, "shakes": 0}] * 4
    assert mbc.elect_owner("shakes", counts2, committed) is None
    # uncommitted slots are never elected
    counts3 = [{"text8": 9000, "shakes": 0}, {"text8": 1, "shakes": 1}]
    assert mbc.elect_owner("text8", counts3, [False, True]) == 1,         "slot 0 is uncommitted (skipped); slot 1 has the only text8 history" 
    # ties keep the LOWEST slot (deterministic)
    counts4 = [{"text8": 100, "shakes": 0}, {"text8": 100, "shakes": 0}]
    assert mbc.elect_owner("text8", counts4, [True, True]) == 0


# ── 4. Verdict math with the doc addendum #1 semantics (pure) ───────────────────────────────

def _rec(b_postB=2.89, b_postC=3.95, b_postD=2.89, a_preB=3.03, a_postE=2.75,
         cret_C=2.85, cret_E=2.74, frac=0.9):
    return {"bpc_A_preB": a_preB, "bpc_A_postB": 3.24, "bpc_A_postE": a_postE,
            "bpc_B_postB": b_postB, "bpc_B_postC": b_postC, "bpc_B_postD": b_postD,
            "bpc_B_postE": 3.9, "bpc_Cret_postC": cret_C, "bpc_Cret_postE": cret_E,
            "boundary_window_D": {"frac_to_owner": frac}}


def _arms(*, b_postD_ex=2.90, b_postD_plain=3.95):
    plain = [_rec(b_postD=b_postD_plain) for _ in range(5)]
    ex = [_rec(b_postD=b_postD_ex, frac=0.9) for _ in range(5)]
    return plain, ex


def test_p1_recovery_bar_pass_and_fail():
    plain, ex = _arms(b_postD_ex=2.90)          # recovery = 3.95 - 2.90 = 1.05 >= 0.10
    v = mbc.claim_verdict(plain, ex)
    assert v["P1"]["status"] == "PASS"
    assert v["P1"]["recovery_mean"] == pytest.approx(1.05, abs=1e-9)
    plain2, ex2 = _arms(b_postD_ex=3.90)        # recovery = 0.05 < 0.10
    v2 = mbc.claim_verdict(plain2, ex2)
    assert v2["P1"]["status"] == "FAIL"
    assert any("abandoned" in b for b in v2["branches"])


def test_p2_and_p3_and_g2():
    plain, ex = _arms()
    v = mbc.claim_verdict(plain, ex)
    assert v["P2"]["status"] == "PASS" and v["P2"]["frac_mean"] == 0.9
    assert v["P3"]["status"] == "PASS"          # 2.75 - 3.03 = -0.28 <= 0.05
    assert v["G2"]["status"] == "PASS"          # 2.74 - 2.85 = -0.11 <= 0.05


def test_g1_is_ci_positional_with_no_small_p_gate():
    # EX BETTER than PLAIN (delta = plain - ex > 0) must be a PASS: an "absence of harm"
    # guard cannot fail because the treatment is good (doc addendum #1).
    plain, ex = _arms(b_postD_ex=2.50, b_postD_plain=3.95)
    v = mbc.claim_verdict(plain, ex)
    assert v["G1"]["status"] == "PASS", "EX better must PASS the absence-of-harm guard"
    # EX worse by >> 0.10 (CI lower > 0.10 with n=5 tight spread) must FAIL
    plain2, ex2 = _arms(b_postD_ex=3.95, b_postD_plain=2.90)
    v2 = mbc.claim_verdict(plain2, ex2)
    assert v2["G1"]["status"] == "FAIL"


def test_overall_truth_table():
    # CLAIMED: everything good
    plain, ex = _arms(b_postD_ex=2.90)
    assert mbc.claim_verdict(plain, ex)["outcome"] == "CLAIMED"
    # P2 FAIL: frac 0.4 while P1 passes
    plain, ex = _arms(b_postD_ex=2.90)
    ex = [dict(r, boundary_window_D={"frac_to_owner": 0.4}) for r in ex]
    v = mbc.claim_verdict(plain, ex)
    assert v["P2"]["status"] == "FAIL" and v["outcome"] == "NEGATIVE"
    assert any("ROUTING LEDGER" in b for b in v["branches"])


# ── 5. Behavioral: route_pr08 honors boundary["owner_slot"] over a_expert ───────────────────

def test_route_pr08_redirect_targets_the_owner_slot():
    # argmin is FORCED onto the PROTECTED slot 1 (its floor mu=0.5 makes it the lowest-CE
    # owner; z ~ 1.5-2.5 < 5 => no novelty, pure argmin path): all segments must FOLD to
    # the owner slot 2 (boundary["owner_slot"] takes precedence over the legacy a_expert).
    import torch
    from seq import fusion_probe as fp

    V = 7
    torch.manual_seed(3)
    xb = torch.randint(0, V, (4, plc.SEG))
    yb = torch.randint(0, V, (4, plc.SEG))
    model = fp.build_model(V, 0, plc.E_POOL)
    mus = {0: 50.0, 1: 0.5, 2: 50.0, 3: 50.0}
    for s in range(plc.E_POOL):
        model.committed[s] = True
        model.mu[s] = mus[s]
        model.var[s] = 1.0
        model.n_segments[s] = 1000
    with torch.no_grad():
        # blow up the NON-owner heads' output weights so argmin lands on protected slot 1
        for s in (0, 2, 3):
            model.experts[s].Wdec.weight *= 50.0
        model.lm(xb)
    model.domain_protect = {1, 3}                 # owner = 2; legacy a_expert key = 0
    led = plc._fresh_ledger()
    boundary = {"a_expert": 0, "owner_slot": 2, "segments_total": 0, "to_A_expert": 0}
    out = plc.route_pr08(model, model._h, yb, "C", led, stream_pos=0, boundary=boundary)
    got = {}
    for s, ids in out:
        got[s] = got.get(s, 0) + len(ids)
    assert 1 not in got, f"protected slot 1 received segments: {got}"
    assert got.get(2, 0) > 0, "redirected segments must land on the OWNER slot (2)"
    assert led["domain_protect_redirects"] == 4,         f"all 4 segments must be counted as redirects: {led['domain_protect_redirects']}"


# ── PR-2026-09-03-19: replication mode (--seed-offset) ──────────────────────────────────────

def test_seed_offset_parser_defaults_and_fingerprint_sensitivity():
    p = mbc._build_parser()
    assert p.parse_args(["--powered-cpu"]).seed_offset == 0
    args = p.parse_args(["--powered-cpu", "--seed-offset", "5",
                         "--ledger-dir", "manyblock_PR-2026-09-03-19"])
    assert args.seed_offset == 5
    assert args.ledger_dir == "manyblock_PR-2026-09-03-19"
    payload = {"leg": "claim", "arm": "EX", "seed": 5, "lr": 3e-3,
               "backbone_lr_post_a": 7.5e-4, "domain_exclusion": True, "smoke": False,
               "vocab": 65, "smoke_segs": None, "seg": 256, "batch_segs": 32,
               "stream_lengths": {}, "blocks": list(mbc.BLOCKS),
               "domain_of": dict(mbc.DOMAIN_OF), "bars": {}}
    assert mbc._fp({**payload, "seed_offset": 0}) != mbc._fp({**payload, "seed_offset": 5}), \
        "fresh-seed replication cells must never resume from original-seed cells"
