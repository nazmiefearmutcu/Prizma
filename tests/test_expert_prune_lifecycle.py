"""Tests for the opt-in "use-it-or-lose-it" slot prune (docs/EXPERT_ECONOMY.md §3.2).

`Prizma.prune_slots(window)` is a METHOD-ONLY lifecycle operation: no constructor
knob, no automatic cadence, never invoked by training. It releases every expert that
is (a) CONSOLIDATED (frozen) and (b) has ZERO routing in the window, by resetting the
slot to a fresh Expert (the recruit/eviction reset path) so a later recruit re-uses
it.

The shipped `route_log` is a LIFETIME CUMULATIVE ledger (one int per slot, sample
counts) with no per-event history, so the conservative sound mapping is used: only a
lifetime-zero count proves zero routing in ANY finite window; any nonzero count may
include routing inside the caller's window and therefore protects the slot. `window`
is required (None -> ValueError), validated, and recorded for audit; the predicate is
window-independent BY DESIGN (a false prune is impossible by construction). This file
pins that interpretation, exactly.

Coverage (mirrors the lane brief):
  1. release + re-use: a frozen zero-routed slot is reset to a fresh expert and the
     next recruit trains into it (cursor rewired when the cursor is not mid-training);
  2. protection: currently-used (nonzero ledger) and non-frozen experts survive;
  3. guard: if every trained expert is a candidate, the most-recently-used one is
     kept -- at least one trained expert always survives;
  4. freshness: the released slot is byte-identical to the constructor's fresh expert
     for that slot (n_seen=0, mu=1e9, var=1.0, not committed/frozen) and training
     continues normally after the prune (mid-training cursors are never abandoned);
  5. window semantics pinned: required/validated; any recorded count protects at any
     window; zero recorded count releases at any window;
  6. default path: training never invokes the prune (no prune_log appears) and two
     identically-seeded runs stay bit-identical.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import structured_permuted_tasks  # noqa: E402
from src.prizma import Expert, Prizma  # noqa: E402


# ----------------------------------------------------------------------------- #
# Tiny deterministic stream helpers (same style as tests/test_mmax_lever.py)     #
# ----------------------------------------------------------------------------- #
def stub_levels(p):
    """Row-deterministic surprise: an expert's per-sample recon is the row's first
    feature. Levels separated by >= 2x are NOVEL to a calibrated expert (z=5), so
    routing / freeze / cursor behaviour is exactly predictable."""
    for e in p.experts:
        e.recon_error = lambda X: X[:, 0].astype(np.float32)


def level_batch(level, n=4):
    X = np.full((n, 2), level, np.float32)
    X[:, 1] = 0.25
    Y = np.zeros((n, 2), np.float32)
    Y[:, 0] = 1.0
    return X, Y, np.zeros(n, np.int64)


def drive(p, levels, n=4):
    for level in levels:
        p.train_batch(*level_batch(level, n=n))


def make(**kw):
    base = dict(d=2, h=4, K=2, n_experts=4, seed=0, z_novel=5.0)
    base.update(kw)
    return Prizma(**base)


def param_vec(p):
    """Full state trajectory: expert weights + floors + lifecycle flags + route_log
    + cursor. Compared with array_equal for bit-identity."""
    parts = []
    for e in p.experts:
        parts.extend([e.Wenc.ravel(), e.benc, e.Wdec.ravel(), e.bdec,
                      e.Wcls.ravel(), e.bcls, e.Bdec.ravel(), e.Bcls.ravel(),
                      np.array([e.mu, e.var, float(e.n_seen), float(e.committed),
                                float(e.frozen), float(e.omega)])])
    parts.append(p.route_log.astype(np.float64))
    parts.append(np.array([float(p.active)]))
    return np.concatenate(parts)


def assert_fresh_slot(e, ref):
    """The release contract: exactly the constructor's fresh Expert for that slot."""
    assert e.n_seen == 0 and not e.committed and not e.frozen
    assert e.omega == 0.0 and e.mu == 1e9 and e.var == 1.0
    assert e.init_recon is None and e.h_mean is None and e.h_var is None
    assert e.prob_since_hit == 0
    for got, want in ((e.Wenc, ref.Wenc), (e.benc, ref.benc), (e.Wdec, ref.Wdec),
                      (e.bdec, ref.bdec), (e.Wcls, ref.Wcls), (e.bcls, ref.bcls),
                      (e.Bdec, ref.Bdec), (e.Bcls, ref.Bcls)):
        assert np.array_equal(got, want)


# ----------------------------------------------------------------------------- #
# 1. Release + re-use                                                           #
# ----------------------------------------------------------------------------- #
def test_prune_releases_frozen_zero_routed_slot_and_recruit_reuses_it():
    p = make()
    stub_levels(p)
    # White-box lifecycle state (same construction style as test_mmax_lever.py): slot
    # 0 is consolidated but saw ZERO recorded routing in the window; slot 1 is the
    # surviving trained expert; the pool cursor is exhausted (active == M, the
    # shipped state after M recruits -- safe to rewind).
    p.experts[0].committed = True
    p.experts[0].frozen = True
    p.experts[0].n_seen = 12
    p.experts[0].mu, p.experts[0].var = 0.30, 0.01
    p.experts[1].committed = True
    p.experts[1].frozen = True
    p.experts[1].n_seen = 12
    p.route_log[:] = [0, 12, 0, 0]
    p.active = p.M
    survivor = p.experts[1]

    released = p.prune_slots(window=8)

    assert released == [0]
    assert_fresh_slot(p.experts[0], Expert(2, 4, 2, 0 + 100 * (0 + 1)))
    assert p.route_log.tolist() == [0, 12, 0, 0]
    # audit record carries the pre-release state
    rec = p.prune_log[-1]
    assert rec["slot"] == 0 and rec["window"] == 8
    assert rec["route_count_at_release"] == 0 and rec["n_seen_at_release"] == 12
    assert rec["kept_by_guard"] is None and rec["active_before"] == p.M
    # the surviving trained expert is untouched
    assert p.experts[1] is survivor and survivor.frozen and survivor.n_seen == 12
    # cursor rewired -> the NEXT recruit trains in the released slot
    assert p.active == 0
    drive(p, [4.0])
    assert p.experts[0].n_seen == 4 and p.route_log[0] == 4
    assert not p.experts[0].frozen and p.active == 0


# ----------------------------------------------------------------------------- #
# 2. Protection: used (nonzero ledger) and non-frozen experts survive            #
# ----------------------------------------------------------------------------- #
def test_currently_used_and_non_frozen_experts_survive():
    p = make()
    # frozen + USED: the cumulative ledger cannot prove the routing sits outside any
    # window -> the conservative mapping protects the slot
    p.experts[0].committed = True
    p.experts[0].frozen = True
    p.experts[0].n_seen = 9
    p.route_log[0] = 3
    # mid-training cursor (uncommitted, trained): never abandoned, never released
    p.active = 1
    p.experts[1].n_seen = 5
    p.route_log[1] = 5
    before = param_vec(p)

    assert p.prune_slots(window=1) == []
    assert p.prune_slots(window=10 ** 6) == []

    assert p.experts[0].frozen and p.experts[0].n_seen == 9 and p.route_log[0] == 3
    assert p.active == 1 and p.experts[1].n_seen == 5
    # zero-routed but NOT frozen slots (2, 3) are not candidates either
    assert not p.experts[2].frozen and p.experts[2].n_seen == 0
    assert np.array_equal(param_vec(p), before)


# ----------------------------------------------------------------------------- #
# 3. Guard: never prune below one trained expert                                 #
# ----------------------------------------------------------------------------- #
def test_all_prunable_keeps_one_trained_expert():
    p = make()
    for m in (0, 1, 2):
        p.experts[m].committed = True
        p.experts[m].frozen = True
        p.experts[m].n_seen = 9
    p.route_log[:] = 0
    p.active = p.M

    released = p.prune_slots(window=3)

    # all three are candidates; releasing all would leave ZERO trained experts ->
    # the guard keeps the most-recently-used one (highest recorded count; tie ->
    # highest slot index, the same recruitment-recency proxy _evict_victim uses)
    assert sorted(released) == [0, 1]
    assert p.experts[2].frozen and p.experts[2].n_seen == 9
    assert sum(e.n_seen > 0 for e in p.experts) == 1
    assert p.prune_log[-1]["kept_by_guard"] == 2

    # a trained survivor outside the candidate set means no keep is needed
    q = make()
    for m in (0, 1, 2, 3):
        q.experts[m].committed = True
        q.experts[m].frozen = True
        q.experts[m].n_seen = 9
    q.route_log[:] = 0
    q.route_log[3] = 7                  # used -> protected, not a candidate
    q.active = q.M
    rel = q.prune_slots(window=3)
    assert sorted(rel) == [0, 1, 2]
    assert q.experts[3].frozen and q.experts[3].n_seen == 9


# ----------------------------------------------------------------------------- #
# 4. Freshness + training continues after the prune                              #
# ----------------------------------------------------------------------------- #
def test_released_slot_is_fresh_and_training_continues():
    p = make(act_bits=8, commit_after=1)
    stub_levels(p)
    drive(p, [0.5, 1.0])                # organic: slot 0 frozen, slot 1 trained
    assert p.experts[0].frozen and p.route_log[0] == 4 and p.active == 1
    p.route_log[0] = 0                  # ledger shows zero routing in the window
    p.experts[0].mu = 0.30              # consolidated floor: must be reset

    released = p.prune_slots(window=4)

    assert released == [0]
    # cursor was mid-training (slot 1) -> NOT rewound: a recruit in progress is
    # never abandoned by the lifecycle operation
    assert p.active == 1
    # released slot is exactly the constructor's fresh Expert (kwargs propagate)
    assert_fresh_slot(p.experts[0], Expert(2, 4, 2, 100, act_bits=8))
    assert p.route_log[0] == 0 and p.prune_log[-1]["window"] == 4
    stub_levels(p)                      # the fresh Expert is a new object: re-stub it

    # the released slot is real capacity: at the next safe checkpoint the cursor is
    # rewound to it and a subsequent recruit trains there (training still works)
    p.active = p.M
    assert p.prune_slots(window=4) == []
    assert p.active == 0
    drive(p, [4.0])
    assert p.experts[0].n_seen == 4 and p.route_log[0] == 4
    assert p.experts[0].mu == pytest.approx(4.0)   # floor re-calibrated on real data


# ----------------------------------------------------------------------------- #
# 5. Window semantics pinned (conservative cumulative-ledger mapping)            #
# ----------------------------------------------------------------------------- #
def test_window_required_and_conservative_mapping_pinned():
    p = make()
    with pytest.raises(ValueError, match="window"):
        p.prune_slots()
    with pytest.raises(ValueError, match="window"):
        p.prune_slots(window=None)
    for bad in (0, -1, 0.5, "8", True):
        with pytest.raises(ValueError, match="window"):
            p.prune_slots(window=bad)

    # a SINGLE recorded routing protects at ANY window: the cumulative ledger cannot
    # prove that routing happened before the window, so the slot is not released
    u = make()
    u.experts[0].committed = True
    u.experts[0].frozen = True
    u.experts[0].n_seen = 6
    u.route_log[0] = 1
    assert u.prune_slots(window=1) == []
    assert u.prune_slots(window=10 ** 9) == []
    assert u.experts[0].frozen and u.experts[0].n_seen == 6

    # ZERO recorded routing releases at ANY window (zero over the whole recorded
    # history is a superset of any finite window -> sound for every window). Slot 1
    # is a trained survivor, so the one-trained-expert guard does not fire.
    z1 = make()
    z1.experts[0].committed = True
    z1.experts[0].frozen = True
    z1.experts[0].n_seen = 6
    z1.experts[1].committed = True
    z1.experts[1].frozen = True
    z1.experts[1].n_seen = 6
    z1.route_log[:] = 0
    z1.route_log[1] = 5
    z1.active = z1.M
    z2 = make()
    z2.experts[0].committed = True
    z2.experts[0].frozen = True
    z2.experts[0].n_seen = 6
    z2.experts[1].committed = True
    z2.experts[1].frozen = True
    z2.experts[1].n_seen = 6
    z2.route_log[:] = 0
    z2.route_log[1] = 5
    z2.active = z2.M
    assert z1.prune_slots(window=1) == [0]
    assert z2.prune_slots(window=10 ** 9) == [0]
    assert z1.prune_log[-1]["window"] == 1
    assert z2.prune_log[-1]["window"] == 10 ** 9


# ----------------------------------------------------------------------------- #
# 6. Default path: no cadence, no side effects without an explicit call          #
# ----------------------------------------------------------------------------- #
def test_default_path_untouched_without_any_prune_call():
    tasks = structured_permuted_tasks(n_tasks=2, n_samples=384, d=8, k_latent=4,
                                      n_classes=4, seed=0)

    def run():
        p = Prizma(d=8, h=24, K=4, n_experts=5, seed=0, z_novel=3.0, commit_after=128)
        rng = np.random.default_rng(0)
        for t in tasks:
            p.fit_task(t.Xtr, t.ytr, epochs=2, batch=128, rng=rng)
        return p

    p1, p2 = run(), run()
    # seed-deterministic twin: the whole trajectory (weights, floors, flags,
    # route_log, cursor) is bit-identical
    assert np.array_equal(param_vec(p1), param_vec(p2))
    assert p1.state() == p2.state()
    assert p1.route_log.sum() > 0
    # training never invokes the prune: no audit log and no slot was released
    assert not hasattr(p1, "prune_log")
    # inference routing is unchanged across the twins
    i1, S1 = p1.route_for_inference(tasks[0].Xte)
    i2, S2 = p2.route_for_inference(tasks[0].Xte)
    assert np.array_equal(i1, i2) and np.array_equal(S1, S2)
