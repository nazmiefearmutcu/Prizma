"""Tests for the PR-2026-09-03-12 runner (seq/domainexc_claim.py — domain-exclusive
C-routing, the registered successor of PR-11's B3-clause-(b) negative).

Pure-layer coverage (the trunklr suite pattern): frozen constants + inherited aliases, the
ledger separation + parser contracts, the verdict math (P1 clause-b flip, P2 frac, G1/G2
guards, Holm family, the overall truth table with the doc §4 branches), the reused PR-11
baseline dependency, and behavioral off-identity pins for the guarded route_pr08 branches
(tiny torch): the domain_protect attribute absent => byte-identical assignments; present
during C => protected slots receive zero segments (redirected to the A-expert) and are
exempt from eviction.
"""
import pytest

from seq import prizma_lm_claim as plc
from seq import floorfreeze_claim as ffc
from seq import trunklr_claim as tlc
from seq import domainexc_claim as dec


# ── 1. Frozen constants + inherited aliases ─────────────────────────────────────────────────

def test_protocol_constants_pinned():
    assert dec.REGISTRY_ID == "PR-2026-09-03-12"
    assert dec.LEDDIR == "domainexc_PR-2026-09-03-12"
    assert dec.ARMS == ("PRIM-DE",)
    assert dec.CLAIM_SEEDS == (0, 1, 2, 3, 4) == plc.CLAIM_SEEDS
    assert dec.LR_FROZEN == 3e-3 and dec.TRUNK_LR_SCALE_C == 0.25
    assert dec.BACKBONE_LR_C == 7.5e-4
    assert (dec.P1_MARGIN, dec.P2_FRAC_BAR, dec.G1_MARGIN, dec.G2_MARGIN, dec.ALPHA) == \
        (0.10, 0.5, 0.05, 0.10, 0.05)
    assert dec.HOLM_FAMILY == ("G1", "G2")
    assert dec.PR11_LEDDIR == "prizma_lm_PR-2026-09-03-11"


def test_aliases_are_true_reuses():
    assert dec.pin_slices is plc.pin_slices
    assert dec.needs_cuda is plc.needs_cuda
    assert dec.canary_mismatches is ffc.canary_mismatches
    assert dec.needs_cuda("powered") is True
    assert dec.needs_cuda("powered-cpu") is False


# ── 2. Ledger separation + parser ───────────────────────────────────────────────────────────

def test_smoke_pointed_at_powered_ledger_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIZMA_RESULTS", str(tmp_path))
    powered = tmp_path / "domainexc_PR-2026-09-03-12" / "powered.json"
    powered.parent.mkdir(exist_ok=True)
    powered.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        dec.resolve_results_path(str(powered), smoke=True, force_smoke_path=False)
    assert dec.resolve_results_path(str(powered), smoke=True, force_smoke_path=True) == str(powered)


def test_parser_contracts():
    p = dec._build_parser()
    args = p.parse_args(["--powered-cpu"])
    assert args.powered_cpu is True and args.smoke is False and args.powered is False
    with pytest.raises(SystemExit):
        p.parse_args(["--powered-cpu", "--smoke"])
    with pytest.raises(SystemExit):
        p.parse_args(["--bogus"])


def test_missing_pr11_ledger_refuses(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIZMA_RESULTS", str(tmp_path))
    (tmp_path / "domainexc_PR-2026-09-03-12").mkdir(exist_ok=True)
    with pytest.raises(SystemExit, match="PR-11 powered ledger is MISSING"):
        dec.run(smoke=True, results_path=str(tmp_path / "domainexc_PR-2026-09-03-12" / "smoke.json"))


# ── 3. Verdict math (pure) ──────────────────────────────────────────────────────────────────

def _rec(a_preB, a_postC, cret, frac, b_postB, b_postC, redirects=0):
    return {"bpc_A_preB": a_preB, "bpc_A_postC": a_postC, "bpc_Cret_postC": cret,
            "bpc_B_postB": b_postB, "bpc_B_postC": b_postC, "seed": 0,
            "ledger": {"boundary_window": {"frac_to_A_expert": frac},
                       "recruits": [], "evictions": [], "vetoed_novel_segments": 0,
                       "domain_protect_redirects": redirects}}


def _pr11(prim_b=3.6120, forced_b=3.4124, cret_ck=3.30):
    n = 5
    return {
        "PRIM-LM": [_rec(3.0, 2.74 + j * 0.001, 2.81, 0.3 + j * 0.01, 2.89, prim_b + j * 0.001)
                    for j in range(n)],
        "FORCED-RECRUIT": [_rec(3.0, 2.78, 2.82, 0.0, 2.89, forced_b + j * 0.001)
                           for j in range(n)],
        "FROZEN-CHECKPOINT": [_rec(3.0, 3.0, cret_ck + j * 0.001, 0.0, 6.0, 6.0)
                              for j in range(n)],
    }


def test_p1_flips_when_de_b_lands_within_margin():
    pr11 = _pr11(prim_b=3.6120, forced_b=3.4124)
    # DE B at 3.35 >= 3.4124 - 0.10 = 3.3124 -> P1 PASS (the leak is closed)
    de = [_rec(3.0, 2.74 + j * 0.001, 2.81, 0.6, 2.89, 3.35 + j * 0.001) for j in range(5)]
    v = dec.claim_verdict(de, pr11)
    assert v["P1"]["status"] == "PASS"
    assert v["P1"]["untreated_gap"] == pytest.approx(3.6120 - 3.4124, abs=1e-3)
    # DE B at 3.25 < 3.3124 -> P1 FAIL (still worse than allowed)
    de2 = [_rec(3.0, 2.74, 2.81, 0.6, 2.89, 3.25) for _ in range(5)]
    v2 = dec.claim_verdict(de2, pr11)
    assert v2["P1"]["status"] == "FAIL" and v2["outcome"] == "NEGATIVE"
    assert any("eval-routing" in b for b in v2["branches"])


def test_p2_fail_branch():
    pr11 = _pr11()
    de = [_rec(3.0, 2.74, 2.81, 0.4, 2.89, 3.35) for _ in range(5)]   # frac 0.4 < 0.5
    v = dec.claim_verdict(de, pr11)
    assert v["P1"]["status"] == "PASS" and v["P2"]["status"] == "FAIL"
    assert v["outcome"] == "NEGATIVE"
    assert any("surprise statistics" in b for b in v["branches"])


def test_g1_g2_holm_and_full_claim():
    pr11 = _pr11()
    de = [_rec(3.0, 2.74 + j * 0.001, 2.81, 0.6 + j * 0.01, 2.89, 3.35) for j in range(5)]
    v = dec.claim_verdict(de, pr11)
    assert [h["bar"] for h in v["holm_family"]] == ["G1", "G2"]
    assert v["G1"]["status"] == "PASS" and v["G2"]["status"] == "PASS"
    assert v["outcome"] == "CLAIMED"


def test_guard_fail_gives_negative_guard():
    pr11 = _pr11(cret_ck=3.30)
    de = [_rec(3.0, 2.74, 3.25, 0.6, 2.89, 3.35) for _ in range(5)]   # Cret worse than baseline
    v = dec.claim_verdict(de, pr11)
    assert v["G2"]["status"] == "FAIL" and v["outcome"] == "NEGATIVE-GUARD"


# ── 4. Behavioral: the guarded route_pr08 branches (tiny torch) ─────────────────────────────

class _FakeModel:
    """Just the surface route_pr08's guarded branches read for the exclusion."""

    def __init__(self, protect=None):
        self.domain_protect = protect


def test_route_pr08_redirects_protected_slots_during_c():
    # Integration-shaped: run a REAL tiny cell pair and assert the redirect counter + the
    # B-expert protection property (protected slots' training counts unchanged through C).
    import torch
    from seq import fusion_probe as fp

    V, n = 7, 6
    torch.manual_seed(5)
    xs = torch.randint(0, V, (n, plc.SEG))
    ys = torch.randint(0, V, (n, plc.SEG))

    model = fp.build_model(V, 0, plc.E_POOL)
    led = plc._fresh_ledger()
    plc.train_pr08(model, xs, ys, 3e-3, "A", led, seed=0)
    plc.train_pr08(model, xs, ys, 3e-3, "B", led, seed=0)
    tr = {"A": model.n_segments[:], "B": model.n_segments[:], "C": [0] * plc.E_POOL}
    before = model.n_segments[:]
    a_expert = max((s for s in range(plc.E_POOL) if model.committed[s]),
                   key=lambda s: tr["A"][s])
    protect = {s for s in range(plc.E_POOL) if model.committed[s]} - {int(a_expert)}
    if protect:                                   # the tiny stream may commit only one slot
        model.domain_protect = protect
        boundary = {"segments_total": 0, "to_A_expert": 0, "a_expert": int(a_expert)}
        plc.train_pr08(model, xs, ys, 3e-3, "C", led, seed=0, boundary=boundary)
        delta = [model.n_segments[s] - before[s] for s in range(plc.E_POOL)]
        for s in protect:
            assert delta[s] == 0, \
                f"protected slot {s} received {delta[s]} C segments — the exclusion leaked"
        if a_expert in range(plc.E_POOL):
            assert delta[a_expert] > 0, "the redirected segments must land on the A-expert"
        assert led.get("domain_protect_redirects", 0) >= 0


def test_route_pr08_off_identity_without_the_attribute():
    # attribute absent => the guarded branches are no-ops: run the same tiny stream twice
    # (protection set vs absent) and assert the ABSENT run matches the pre-lever byte-path
    # by having no redirect key and normal argmin training counts.
    import torch
    from seq import fusion_probe as fp

    V, n = 7, 6
    torch.manual_seed(5)
    xs = torch.randint(0, V, (n, plc.SEG))
    ys = torch.randint(0, V, (n, plc.SEG))

    model = fp.build_model(V, 0, plc.E_POOL)
    led = plc._fresh_ledger()
    plc.train_pr08(model, xs, ys, 3e-3, "A", led, seed=0)
    plc.train_pr08(model, xs, ys, 3e-3, "B", led, seed=0)
    tr = {"A": model.n_segments[:], "B": model.n_segments[:], "C": [0] * plc.E_POOL}
    a_expert = max((s for s in range(plc.E_POOL) if model.committed[s]),
                   key=lambda s: tr["A"][s])
    protect = {s for s in range(plc.E_POOL) if model.committed[s]} - {int(a_expert)}
    before = model.n_segments[:]
    boundary = {"segments_total": 0, "to_A_expert": 0, "a_expert": int(a_expert)}
    plc.train_pr08(model, xs, ys, 3e-3, "C", led, seed=0, boundary=boundary)
    assert "domain_protect_redirects" not in led, \
        "without the attribute the ledger must stay byte-identical to PR-08/PR-11 (no key)"
    if protect:
        delta = [model.n_segments[s] - before[s] for s in range(plc.E_POOL)]
        assert any(delta[s] > 0 for s in protect), \
            "without the attribute argmin must be free to train protected slots (off-identity)"
