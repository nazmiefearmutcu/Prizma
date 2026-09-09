"""Tests for the PR-2026-09-03-11 REPAIRED-FLAGSHIP mode (docs/preregistry/
2026-09-09-repaired-flagship.md) implemented inside the PR-08 runner
(seq/prizma_lm_claim.py + seq/fusion_probe.py's train_stream_shared).

Pure-layer first (the trunklr/floorfreeze suite pattern):
  (a) the parser contract: --trunk-lr-c 0.0075 --ledger-dir X parse; defaults are None;
      --trunk-lr-c alone changes no other default,
  (b) the ledger_dir override of LEDDIR (None => byte-identical default paths),
  (c) the PURE _backbone_lr_for_c truth table (the exact rule run_routed uses — pinned
      load-bearing via a source check that run_routed calls the helper),
  (d) fingerprint sensitivity: trunk_lr_c None vs 7.5e-4 produce DIFFERENT cfgsig values for
      both the lr-selection and claim payload shapes. DISCLOSED APPROACH: plc._fp imports
      seq.gpu_harness which pulls torch — torch IS importable in this venv (CPU wheel; the
      behavioral tests need it anyway), so the real fingerprint function is exercised, and
      additionally the '"trunk_lr_c"' payload key is pinned textually in run()'s source for
      BOTH payload constructions (the fallback the task specifies if _fp were not callable),
  (e) the canary schema design: the lever audit key backbone_lr_c_applied is NOT in
      ffc.CANARY_IGNORE, so a FROZEN cell that accidentally carries it can NEVER pass the
      PR-11 canary (schema-clean enforcement), while a clean frozen cell compares empty,
  (f) the canary gating shape: _pr11_canary is wired into run() behind the lever-ON +
      non-smoke gate.
One behavioral test (tiny torch, <2s) pins train_stream_shared's guarded kwarg:
backbone_lr=None byte-identical to not passing it; a 0.25 dose fires (params diverge).
"""
import inspect

import pytest

from seq import prizma_lm_claim as plc
from seq import floorfreeze_claim as ffc


# ── 1. Parser contract (PR-11 flags; defaults None = byte-identical PR-08) ─────────────────

def test_parser_accepts_pr11_flags():
    p = plc._build_parser()
    a = p.parse_args(["--smoke", "--trunk-lr-c", "0.0075",
                      "--ledger-dir", "prizma_lm_PR-2026-09-03-11"])
    assert a.trunk_lr_c == pytest.approx(0.0075)
    assert a.ledger_dir == "prizma_lm_PR-2026-09-03-11"
    assert a.smoke is True


def test_parser_defaults_are_none():
    p = plc._build_parser()
    a = p.parse_args(["--powered-cpu"])
    assert a.trunk_lr_c is None and a.ledger_dir is None, \
        "no PR-11 flags => byte-identical PR-08 run (lever off, LEDDIR)"


def test_trunk_lr_c_alone_changes_no_other_default():
    p = plc._build_parser()
    plain = p.parse_args(["--smoke"])
    treated = p.parse_args(["--smoke", "--trunk-lr-c", "7.5e-4"])
    for key in ("out", "force_smoke_path", "powered", "powered_cpu"):
        assert getattr(plain, key) == getattr(treated, key), \
            "--trunk-lr-c alone must not move any other parser default"


# ── 2. ledger_dir override (None => LEDDIR, byte-identical) ────────────────────────────────

def test_ledger_dir_overrides_leddir_and_none_is_byte_identical(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIZMA_RESULTS", str(tmp_path))
    for smoke in (False, True):
        dflt = plc._default_results_path(smoke=smoke)
        assert plc._default_results_path(smoke=smoke, ledger_dir=None) == dflt, \
            "ledger_dir=None must give the exact PR-08 default path"
        pr11 = plc._default_results_path(smoke=smoke,
                                         ledger_dir="prizma_lm_PR-2026-09-03-11")
        assert "prizma_lm_PR-2026-09-03-08" in dflt
        assert "prizma_lm_PR-2026-09-03-11" in pr11 and pr11 != dflt
        assert pr11.endswith("powered.json" if not smoke else "smoke.json")
    # and the refusal guard threads ledger_dir (smoke pointed at the PR-11 powered ledger)
    powered11 = plc._default_results_path(smoke=False,
                                          ledger_dir="prizma_lm_PR-2026-09-03-11")
    with pytest.raises(SystemExit):
        plc.resolve_results_path(powered11, smoke=True, ledger_dir="prizma_lm_PR-2026-09-03-11")


# ── 3. _backbone_lr_for_c truth table (PURE; used by run_routed — load-bearing) ────────────

def test_backbone_lr_for_c_truth_table():
    f = plc._backbone_lr_for_c
    assert f(None, False) is None, "lever OFF => None (PR-08-identical call)"
    assert f(None, True) is None
    assert f(7.5e-4, False) == pytest.approx(7.5e-4), "lever ON, backbone trains on C"
    assert f(7.5e-4, True) is None, \
        "FROZEN-TRUNK has no C backbone step: the lever is structurally inapplicable there " \
        "(its C call stays parameter-identical to PR-08 — canary-clean)"


def test_backbone_lr_for_c_is_actually_used_by_run_routed():
    src = inspect.getsource(plc.run_routed)
    assert "_backbone_lr_for_c" in src, \
        "the truth table must be load-bearing: run_routed applies the helper"
    assert "backbone_lr_c_applied" in src, "the applied-lever audit key is recorded in the cell"


def test_run_shared_threads_the_lever_on_the_c_call_only():
    src = inspect.getsource(plc.run_shared)
    assert "backbone_lr_c_applied" in src and "backbone_lr=trunk_lr_c" in src
    # only ONE train_stream_shared call carries the kwarg (C-phase call only)
    assert src.count("backbone_lr=trunk_lr_c") == 1


# ── 4. Fingerprint sensitivity (disclosed: real _fp exercised — torch importable here) ─────

def test_fingerprint_is_sensitive_to_trunk_lr_c():
    claim_base = {"leg": "claim", "arm": "PRIM-LM", "family": "fusion", "seed": 0, "lr": 3e-3}
    lrsel_base = {"leg": "lr-selection", "family": "fusion", "lr_grid": list(plc.LR_GRID)}
    for base in (claim_base, lrsel_base):
        sig_off = plc._fp({**base, "trunk_lr_c": None})
        sig_on = plc._fp({**base, "trunk_lr_c": 7.5e-4})
        sig_nolever = plc._fp(dict(base))          # a pre-PR-11 cell's payload shape
        assert sig_off != sig_on, \
            "a treated rerun must never resume from untreated cells (and vice versa)"
        assert sig_on != sig_nolever and sig_off != sig_nolever, \
            "every cell written by the PR-11-aware code differs from pre-PR-11 stored cfgsigs " \
            "(foreign-fingerprint refusal is loud, never a silent resume)"


def test_trunk_lr_c_key_is_in_both_fingerprint_payload_constructions():
    src = inspect.getsource(plc.run)
    # the lr-selection payload AND the claim payload both carry the key (textual pin —
    # the task's specified fallback, kept alongside the live-sig test above)
    assert src.count('"trunk_lr_c": trunk_lr_c') == 2, \
        "exactly two fingerprint payload constructions gain the lever key"


# ── 5. Canary schema design (audit key NOT in the ignore set) ──────────────────────────────

def _frozen_cell():
    """A PR-08-shaped FROZEN arm cell (no lever audit keys) + the wrapper keys the runner
    adds after the fact (ignored by the comparator)."""
    return {"config": "FROZEN-TRUNK", "E": 4, "seed": 0, "lr": 3e-3,
            "bpc_A_preB": 3.1, "bpc_A_postB": 3.2, "bpc_B_postB": 4.0,
            "bpc_Cret_postB": 3.3, "bpc_A_postC": 3.2, "bpc_B_postC": 5.2,
            "bpc_Cret_postC": 3.3, "fgt_A_full": -0.1, "a_expert": 0,
            "b_degradation_B_postC": 1.2,
            "ledger": {"boundary_window": {"frac_to_A_expert": 0.96}},
            "wall_s": 42.0, "arm": "FROZEN-TRUNK", "cellkey": "claim.FROZEN-TRUNK.s0",
            "cfgsig": "deadbeef00c0ffee", "complete": True}


def test_canary_schema_clean_frozen_cell_compares_empty_vs_pr08():
    pr08 = _frozen_cell()
    fresh = dict(pr08, cfgsig="different")     # PR-11 recomputes cfgsig; it is ignored
    assert ffc.canary_mismatches(fresh, pr08) == [], \
        "a frozen cell with no audit keys is byte-comparable against its PR-08 record"


def test_canary_flags_a_cell_carrying_the_audit_key():
    pr08 = _frozen_cell()
    flagged = dict(pr08, backbone_lr_c_applied=0.0075)
    assert ffc.canary_mismatches(flagged, pr08) == ["backbone_lr_c_applied"], \
        "a frozen arm must never carry the lever audit key"
    assert "backbone_lr_c_applied" not in ffc.CANARY_IGNORE, \
        "the chosen design: the audit key is deliberately NOT in the ignore set, so an " \
        "accidental lever leak onto a frozen cell fails the canary loudly"
    assert "cfgsig" in ffc.CANARY_IGNORE, \
        "cfgsig must stay ignored: PR-11 recomputes it (new fingerprint key) but the frozen " \
        "SCIENCE must still compare clean"


# ── 6. Canary wiring shape (lever ON + claim mode only) ────────────────────────────────────

def test_canary_gating_is_lever_on_and_claim_mode_only():
    src = inspect.getsource(plc.run)
    assert "_pr11_canary" in src
    assert "trunk_lr_c is not None" in src and "not smoke" in src, \
        "the canary must gate ONLY when the lever is ON and never in smoke (a plain PR-08 " \
        "rerun behaves byte-identically to before, including NO canary)"
    for arm in ("FROZEN-TRUNK", "FROZEN-CHECKPOINT"):
        assert arm in src, "both frozen arms serve as canary gates"


def test_pr11_canary_aborts_on_missing_pr08_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIZMA_RESULTS", str(tmp_path))
    res = {"claim.FROZEN-TRUNK.s0": _frozen_cell()}
    with pytest.raises(SystemExit) as ei:
        plc._pr11_canary(res, [0], "FROZEN-TRUNK")
    assert "MISSING" in str(ei.value), "a missing PR-08 ledger is a loud ABORT"


# ── 7. Behavioral: the guarded train_stream_shared kwarg (tiny torch, ~1-2s) ───────────────

def test_train_stream_shared_backbone_lr_off_identity_and_fire():
    import torch
    from seq import fusion_probe as fp

    V, n = 7, 4
    torch.manual_seed(321)
    xs = torch.randint(0, V, (n, plc.SEG))
    ys = torch.randint(0, V, (n, plc.SEG))

    def one(backbone_lr, sentinel):
        model = fp.SharedHeadLM(V, 0, plc.H_SHARED)
        kw = {} if sentinel else {"backbone_lr": backbone_lr}
        fp.train_stream_shared(model, xs, ys, 3e-3, **kw)
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    default_run = one(None, sentinel=True)
    none_run = one(None, sentinel=False)
    scaled_run = one(0.25, sentinel=False)
    assert set(default_run) == set(none_run) == set(scaled_run)
    assert all(torch.equal(default_run[k], none_run[k]) for k in default_run), \
        "backbone_lr=None must be byte-identical to not passing it (off-identity, ALL params)"
    assert not all(torch.equal(default_run[k], scaled_run[k]) for k in default_run), \
        "the dose must fire (backbone — and with it the head — diverges)"
    assert not torch.equal(default_run["lm.head.weight"], scaled_run["lm.head.weight"]), \
        "the backbone divergence propagates through the shared-head objective"
