"""Tests for the PR-2026-09-03-10 runner (seq/trunklr_claim.py — trunk-lr scheduling on the
returning block, the registered successor of PR-09's floor-freeze negative).

Pure-layer coverage first (the floorfreeze suite pattern): the frozen constants, the ledger
separation + refusal, the parser contracts, the CUDA guard alias, the bar math (P1 exact
means; G1 one-sample below; G2 Welch above with the Holm [G1, G2] family), the overall
outcome truth table with the doc §4 pre-committed branches, and the §2 canary comparator
(reused from PR-09's runner — asserted to be the SAME object, not a fork). One behavioral
test (tiny torch, <2s) pins the guarded train_pr08 backbone_lr kwarg: None is byte-identical
to not passing it; the 0.25 dose fires.
"""
import pytest

from seq import prizma_lm_claim as plc
from seq import floorfreeze_claim as ffc
from seq import trunklr_claim as tlc


# ── 1. Frozen constants + inherited aliases ─────────────────────────────────────────────────

def test_protocol_constants_pinned():
    assert tlc.REGISTRY_ID == "PR-2026-09-03-10"
    assert tlc.LEDDIR == "trunklr_PR-2026-09-03-10"
    assert tlc.ARMS == ("OFF", "TRUNK-LR-0.25C")
    assert tlc.CLAIM_SEEDS == (0, 1, 2, 3, 4) == plc.CLAIM_SEEDS
    assert tlc.LR_FROZEN == 3e-3
    assert tlc.TRUNK_LR_SCALE_C == 0.25
    assert tlc.BACKBONE_LR_C == 3e-3 * 0.25 == 7.5e-4
    assert (tlc.P1_FRAC_BAR, tlc.G1_MARGIN, tlc.G2_MARGIN, tlc.ALPHA) == (0.5, 0.05, 0.10, 0.05)
    assert tlc.HOLM_FAMILY == ("G1", "G2")
    assert tlc.PR08_LEDDIR == plc.LEDDIR


def test_slices_and_cuda_guard_are_true_aliases():
    # doc §2 inherits PR-08's stream verbatim and the §2 canary reuses PR-09's comparator:
    # these must be the SAME objects (import reuse, not forks).
    assert tlc.pin_slices is plc.pin_slices
    assert tlc.needs_cuda is plc.needs_cuda
    assert tlc.canary_mismatches is ffc.canary_mismatches
    assert tlc.CANARY_IGNORE == ffc.CANARY_IGNORE
    assert tlc.needs_cuda("powered") is True
    assert tlc.needs_cuda("powered-cpu") is False
    assert tlc.needs_cuda("smoke") is False


def test_pinned_slices_invariants_hold():
    sl = tlc.pin_slices()   # the disjointness assertions run inside (B1 protection inherited)
    assert sl["C_train"] == ("text8", 1_100_000, 2_100_000)
    assert sl["A_eval"] == ("text8", 1_000_000, 1_100_000)


# ── 2. Ledger separation + parser contracts ─────────────────────────────────────────────────

def test_smoke_pointed_at_powered_ledger_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIZMA_RESULTS", str(tmp_path))
    powered = tmp_path / "trunklr_PR-2026-09-03-10" / "powered.json"
    powered.parent.mkdir(exist_ok=True)
    powered.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        tlc.resolve_results_path(str(powered), smoke=True, force_smoke_path=False)
    # the same pointer is allowed only with the deliberate override
    ok = tlc.resolve_results_path(str(powered), smoke=True, force_smoke_path=True)
    assert ok == str(powered)


def test_powered_cpu_alone_accepted_and_combinations_rejected():
    p = tlc._build_parser()
    args = p.parse_args(["--powered-cpu"])
    assert args.powered_cpu is True and args.smoke is False and args.powered is False
    with pytest.raises(SystemExit):
        p.parse_args(["--powered-cpu", "--smoke"])
    with pytest.raises(SystemExit):
        p.parse_args(["--powered-cpu", "--powered"])
    with pytest.raises(SystemExit):
        p.parse_args(["--smoke", "--powered"])


# ── 3. Bar math + overall truth table (pure) ────────────────────────────────────────────────

def _rec(a_preB, a_postC, cret, frac, b_postB, b_postC):
    # fixtures model the REAL cell schema (the routing snapshot nests under rec["ledger"])
    return {"bpc_A_preB": a_preB, "bpc_A_postC": a_postC, "bpc_Cret_postC": cret,
            "bpc_B_postB": b_postB, "bpc_B_postC": b_postC, "seed": 0,
            "ledger": {"boundary_window": {"frac_to_A_expert": frac},
                       "recruits": [], "evictions": [], "vetoed_novel_segments": 0}}


def _arms(frac_tl, *, b_postC_tl=4.0, cret_tl=2.80, n=5):
    off = [_rec(3.0, 2.73 + j * 0.001, 2.81 + j * 0.001, 0.3 + j * 0.01, 2.9, 4.0)
           for j in range(n)]
    tl = [_rec(3.0, 2.73 + j * 0.001, cret_tl + j * 0.001, frac_tl, 2.9, b_postC_tl)
          for j in range(n)]
    frozen_ck = [_rec(3.0, 3.0, 3.30 + j * 0.001, 0.0, 6.0, 6.0) for j in range(n)]
    return off, tl, frozen_ck


def test_p1_pass_and_fail_at_exact_means():
    off, tl, ck = _arms(0.55)
    v = tlc.claim_verdict(off, tl, ck)
    assert v["P1"]["status"] == "PASS" and v["P1"]["frac_mean"] == 0.55
    off2, tl2, ck2 = _arms(0.45)
    v2 = tlc.claim_verdict(off2, tl2, ck2)
    assert v2["P1"]["status"] == "FAIL" and v2["outcome"] == "NEGATIVE"
    assert any("0.25 dose" in b for b in v2["branches"])


def test_g1_g2_and_holm_family():
    off, tl, ck = _arms(0.55)
    v = tlc.claim_verdict(off, tl, ck)
    assert [h["bar"] for h in v["holm_family"]] == ["G1", "G2"]
    assert v["G1"]["status"] == "PASS"      # diffs ~ -0.27, far below 0.05
    assert v["G2"]["status"] == "PASS"      # advantage ~ 0.5, far above 0.10
    assert v["outcome"] == "CLAIMED"


def test_g2_fail_gives_negative_guard():
    # treatment Cret WORSE than the frozen baseline -> G2 FAIL, P1 still PASS
    off, tl, ck = _arms(0.55, cret_tl=3.25)
    v = tlc.claim_verdict(off, tl, ck)
    assert v["P1"]["status"] == "PASS" and v["G2"]["status"] == "FAIL"
    assert v["outcome"] == "NEGATIVE-GUARD"
    assert any("G2" in b for b in v["branches"])


def test_g1_inconclusive_never_passes():
    # craft G1 diffs straddling the 0.05 margin (mean 0.05, wide spread -> CI straddles)
    tl = [_rec(3.0, 3.05 + d, 2.81, 0.6, 2.9, 4.0)
          for d in (-0.08, -0.04, 0.0, 0.04, 0.08)]
    off = [_rec(3.0, 2.73, 2.81, 0.3, 2.9, 4.0)] * 5
    ck = [_rec(3.0, 3.0, 3.30, 0.0, 6.0, 6.0)] * 5
    v = tlc.claim_verdict(off, tl, ck)
    assert v["G1"]["status"] == "INCONCLUSIVE"
    assert v["outcome"] == "INCONCLUSIVE"
    assert any("PRE-AUTHORIZED" in b for b in v["branches"])


# ── 4. §2 canary comparator (reused object) ─────────────────────────────────────────────────

def test_canary_comparator_ignores_wall_and_wrappers_detects_science_drift():
    a = _rec(3.0, 2.73, 2.81, 0.4, 2.9, 4.0)
    b = dict(a)
    assert tlc.canary_mismatches(a, b) == []
    b = dict(a, wall_s=99.9, arm="OFF", cellkey="claim.OFF.s0", cfgsig="x", complete=True,
             forced_placement={"mode": "free_slot"})
    assert tlc.canary_mismatches(a, b) == []          # wrappers + timing ignored
    b = dict(a, bpc_A_postC=2.74)
    assert tlc.canary_mismatches(a, b) == ["bpc_A_postC"]


# ── 5. Behavioral: the guarded train_pr08 backbone_lr kwarg (tiny torch, ~1-2s) ─────────────

def test_train_pr08_backbone_lr_off_identity_and_fire():
    import torch
    from seq import fusion_probe as fp

    V, n = 7, 4
    torch.manual_seed(123)
    xs = torch.randint(0, V, (n, plc.SEG))
    ys = torch.randint(0, V, (n, plc.SEG))

    def one(backbone_lr, sentinel):
        model = fp.build_model(V, 0, plc.E_POOL)
        led = plc._fresh_ledger()
        kw = {} if sentinel else {"backbone_lr": backbone_lr}
        plc.train_pr08(model, xs, ys, 3e-3, "A", led, seed=0, **kw)
        return [p.detach().clone() for p in model.lm.parameters()]

    default_run = one(None, sentinel=True)
    none_run = one(None, sentinel=False)
    scaled_run = one(7.5e-4, sentinel=False)
    assert all(torch.equal(a, b) for a, b in zip(default_run, none_run)), \
        "backbone_lr=None must be byte-identical to not passing it (off-identity)"
    assert not all(torch.equal(a, b) for a, b in zip(default_run, scaled_run)), \
        "the 0.25 backbone dose must fire (backbone weights diverge)"
