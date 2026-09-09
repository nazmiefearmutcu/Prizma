"""Tests for the PR-2026-09-03-13 exclusion flagship (seq/prizma_lm_claim.py's guarded
--domain-exclusion mode + the route_pr08 suppression refinement).

Pure-layer contracts (parser, fingerprint sensitivity) plus tiny-torch behavioral pins of
the NEW refinement: with domain-exclusion active, a pool-full recruit whose post-exclusion
eviction pool is only the a_expert is SUPPRESSED (no self-eviction; segments land on the
a_expert via argmin + redirect); without the lever attribute the PR-08 eviction path is
byte-preserved (no suppression key, eviction fires).
"""
import pytest

from seq import prizma_lm_claim as plc


# ── 1. Parser contract ──────────────────────────────────────────────────────────────────────

def test_domain_exclusion_flag_default_off_and_threadable():
    p = plc._build_parser()
    args = p.parse_args(["--powered-cpu"])
    assert args.domain_exclusion is False
    args2 = p.parse_args(["--powered-cpu", "--domain-exclusion",
                          "--trunk-lr-c", "0.0075",
                          "--ledger-dir", "prizma_lm_PR-2026-09-03-13"])
    assert args2.domain_exclusion is True and args2.trunk_lr_c == 0.0075
    assert args2.ledger_dir == "prizma_lm_PR-2026-09-03-13"


def test_domain_exclusion_is_in_the_claim_fingerprint():
    # real _fp (torch available on this box): the lever state must invalidate stale cells
    payload = {"leg": "claim", "arm": "PRIM-LM", "family": "fusion", "seed": 0, "lr": 3e-3,
               "smoke": False, "vocab": 65, "smoke_segs": None, "seg": 256, "batch_segs": 32,
               "slices": {"A_train": ["text8", 0, 100]}, "stream_lengths": {"A": 1},
               "tissue": {}, "bars": {}, "trunk_lr_c": None}
    sig_off = plc._fp({**payload, "domain_exclusion": False})
    sig_on = plc._fp({**payload, "domain_exclusion": True})
    assert sig_off != sig_on


# ── 2. Behavioral: the suppression refinement (tiny torch) ──────────────────────────────────

def _forced_novelty_setup():
    """A model whose 4 slots are all committed with mature, near-zero floors: every segment's
    surprise z is huge, so recruiting_ids = ALL segments and the pool is full — the exact
    PR-12-recycling / PR-13-suppression scenario."""
    import torch
    from seq import fusion_probe as fp

    V = 7
    torch.manual_seed(11)
    xb = torch.randint(0, V, (4, plc.SEG))
    yb = torch.randint(0, V, (4, plc.SEG))
    model = fp.build_model(V, 0, plc.E_POOL)
    for s in range(plc.E_POOL):
        model.committed[s] = True
        model.mu[s] = 1e-6            # near-zero floors => z >> Z_NOVEL for every segment
        model.var[s] = 1e-8           # sd = 1e-4: z = seg_ce/1e-4 ~ 2e4 > 5 (novelty fires)
        model.n_segments[s] = 1000    # mature: the floor-maturity veto never fires
    with torch.no_grad():
        model.lm(xb)                  # the hook fills model._h
    return model, model._h, yb


def test_refinement_suppresses_recruit_and_preserves_a_expert_identity():
    model, h, y = _forced_novelty_setup()
    model.domain_protect = {1, 2, 3}
    led = plc._fresh_ledger()
    boundary = {"segments_total": 0, "to_A_expert": 0, "a_expert": 0}
    out = plc.route_pr08(model, h, y, "C", led, stream_pos=0, boundary=boundary)
    assert led["domain_protect_suppressed_recruits"] == 1, "the suppression must be counted"
    assert len(led["evictions"]) == 0, "the a_expert must NOT be self-evicted"
    assert len(led["recruits"]) == 0, "the suppressed recruit must not be recorded"
    total = sum(len(ids) for _, ids in out)
    assert total == y.shape[0], "every segment must still be trained exactly once"
    slots_used = {s for s, _ in out}
    assert slots_used == {0}, "with all other slots protected, everything lands on the a_expert"


def test_off_identity_without_the_attribute_evicts_normally():
    model, h, y = _forced_novelty_setup()
    assert not hasattr(model, "domain_protect")
    led = plc._fresh_ledger()
    boundary = {"segments_total": 0, "to_A_expert": 0, "a_expert": 0}
    plc.route_pr08(model, h, y, "C", led, stream_pos=0, boundary=boundary)
    assert "domain_protect_suppressed_recruits" not in led, \
        "without the attribute the ledger must stay byte-identical to PR-08 (no key)"
    assert len(led["evictions"]) == 1 and len(led["recruits"]) == 1, \
        "the PR-08 Policy A eviction path must fire unchanged (lowest-share victim)"


def test_refinement_inert_on_earlier_blocks_and_when_lever_absent():
    # corpus != C never suppresses, even with the attribute present
    model, h, y = _forced_novelty_setup()
    model.domain_protect = {1, 2, 3}
    led = plc._fresh_ledger()
    boundary = {"segments_total": 0, "to_A_expert": 0, "a_expert": 0}
    plc.route_pr08(model, h, y, "B", led, stream_pos=0, boundary=boundary)
    assert "domain_protect_suppressed_recruits" not in led
    assert len(led["evictions"]) == 1, "on B the PR-08 eviction path is unchanged"


# ── 3. Dose guard (the 2026-09-09 invalid-dose incident: 0.0075 vs 7.5e-4) ──────────────────

def test_dose_guard_refuses_unregistered_doses():
    assert plc.validate_trunk_lr_c(None) is None            # lever OFF: always valid
    assert plc.validate_trunk_lr_c(plc.REGISTERED_TRUNK_LR_C) is None
    with pytest.raises(SystemExit, match="registered dose"):
        plc.validate_trunk_lr_c(0.0075)                     # the 10x typo that started it
    with pytest.raises(SystemExit):
        plc.validate_trunk_lr_c(1e-3)
