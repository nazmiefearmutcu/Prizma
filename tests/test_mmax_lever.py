"""Tests for the bounded-M economy lever (Policy A recruit-by-eviction; PR-2026-09-03-05).

ONE Prizma-config knob, DEFAULT-OFF, implementing docs/EXPERT_ECONOMY.md §3.3 Policy A
verbatim: "on a recruit when the pool is at M_max, evict the expert with the lowest
lifetime routing share (tie -> most recently recruited), re-use its slot."
m_max=0 = OFF = unbounded = shipped; m_max > 0 caps the USABLE experts at m_max while
the pool allocation self.M stays unchanged (slots >= m_max are never used).

Contract under test:

  OFF-identity:   m_max=0 (explicit or default) and m_max > M are BIT-IDENTICAL to the
                  shipped behaviour on block streams -- parameter trajectories (all
                  expert weights + mu/var/n_seen), route_log, committed/frozen flags,
                  ACC/FGT -- for ALL THREE recruit sites (shipped batch path, G1
                  per-sample path, G3b probation path). eviction_log stays empty.
  ON-changes:     with m_max below the stream's recruit count the run evicts: the
                  route_log/weight trajectory diverges from the unbounded twin, slots
                  beyond the cap stay zero for the WHOLE block stream, and the reused
                  slot trains the new recruit.
  Shipped cap:    the shipped pool-exhaustion drop path (`if self.active >= self.M:
                  return`) still EXISTS and still FIRES when the lever is OFF -- the
                  silent-drop behaviour (Policy-B worst case, docs §3.3) is pinned,
                  not silently replaced.
  White-box (a):  exactly m_max slots ever hold nonzero route_log (checked after every
                  batch); nonzero entries only in [0, m_max).
  White-box (b):  the victim is the committed expert with the LOWEST lifetime routing
                  share share_m = route_log[m]/max(route_log.sum(), 1).
  White-box (c):  ties break to the HIGHEST slot index (the recruitment-recency proxy:
                  slots fill left-to-right, so most recently recruited).
  White-box (d):  every fire is recorded in eviction_log (victim, its count/share, the
                  route_log at fire time); the reused slot holds a FRESH Expert with
                  exactly the constructor args __init__ uses for that slot (same seed
                  formula seed + 100*(m+1)), and its route_log entry is zeroed.
  Edge ruling:    m_max == M is a live cap (Policy A replaces the exhaustion refusal
                  with an eviction at the pool edge); OFF-identity is therefore stated
                  for m_max = 0 and m_max > M, and both are tested.
  Knob hygiene:   negative / non-integer m_max raise ValueError.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.data import structured_permuted_tasks
from src.metrics import AccuracyMatrix, accuracy
from src.prizma import Expert, Prizma


# ----------------------------------------------------------------------------- #
# Tiny streams + drivers (seconds)                                              #
# ----------------------------------------------------------------------------- #
def stub_levels(p):
    """Give every expert a row-deterministic stub recon_error: the surprise of a row
    is its first feature. Encode each batch's surprise LEVEL in X[:, 0]; routing and
    floor calibration then become exactly predictable (an expert trained at level L
    has threshold 1.5*L at z=5, so levels separated by >= 2x are always NOVEL to it)."""
    for e in p.experts:
        e.recon_error = lambda X: X[:, 0].astype(np.float32)


def level_batch(L, n=4):
    X = np.full((n, 2), L, np.float32)
    X[:, 1] = 0.25
    Y = np.zeros((n, 2), np.float32)
    Y[:, 0] = 1.0
    return X, Y, np.zeros(n, np.int64)


def drive(p, levels, n=4):
    for L in levels:
        p.train_batch(*level_batch(L, n=n))


def make(m_max=None, **kw):
    base = dict(d=2, h=4, K=2, n_experts=6, seed=0, commit_after=1, z_novel=5.0)
    base.update(kw)
    if m_max is not None:
        base["m_max"] = m_max
    return Prizma(**base)


def block_run(p, seed=0, K=3, epochs=3, samples=512):
    """Shipped E-protocol on a tiny block stream -> (AccuracyMatrix, p)."""
    tasks = structured_permuted_tasks(n_tasks=K, n_samples=samples, d=8, k_latent=4,
                                      n_classes=4, seed=seed)
    R = AccuracyMatrix(len(tasks))
    rng = np.random.default_rng(seed)
    for i, t in enumerate(tasks):
        p.fit_task(t.Xtr, t.ytr, epochs=epochs, rng=rng)
        for j, tt in enumerate(tasks):
            R.record(i, j, accuracy(p.predict_logits(tt.Xte), tt.yte))
    return R, p


def param_vec(p):
    """Full parameter/state trajectory: every expert's weights + floors + n_seen,
    then route_log. Compared with array_equal for bit-identity."""
    parts = []
    for e in p.experts:
        parts.extend([e.Wenc.ravel(), e.benc, e.Wdec.ravel(), e.bdec,
                      e.Wcls.ravel(), e.bcls,
                      np.array([e.mu, e.var, float(e.n_seen),
                                float(e.committed), float(e.frozen)])])
    parts.append(p.route_log.astype(np.float64))
    return np.concatenate(parts)


class Recorder(Prizma):
    """Route_log snapshot after every batch (white-box (a): the cap invariant must
    hold THROUGHOUT the stream, not just at the end)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.snaps = []
        self.psnaps = []

    def train_batch(self, X, Y, y):
        super().train_batch(X, Y, y)
        self.snaps.append(self.route_log.copy())
        self.psnaps.append(param_vec(self))


# ----------------------------------------------------------------------------- #
# OFF-identity: bit-identical on all three recruit paths                        #
# ----------------------------------------------------------------------------- #
def test_off_identity_batch_path_bit_identical():
    cfg = dict(d=8, h=24, K=4, n_experts=5, seed=0, z_novel=3.0, commit_after=128)
    r_def, p_def = block_run(Prizma(**cfg))
    r_off, p_off = block_run(Prizma(**cfg, m_max=0))
    r_big, p_big = block_run(Prizma(**cfg, m_max=99))       # m_max > M: inert
    assert p_def.eviction_log == [] and p_off.eviction_log == [] \
        and p_big.eviction_log == []
    assert np.array_equal(param_vec(p_def), param_vec(p_off))
    assert np.array_equal(param_vec(p_def), param_vec(p_big))
    assert r_def.acc() == r_off.acc() == r_big.acc()
    assert r_def.forgetting() == r_off.forgetting() == r_big.forgetting()


def test_off_identity_per_sample_and_probation_paths():
    tasks = structured_permuted_tasks(n_tasks=3, n_samples=512, d=8, k_latent=4,
                                      n_classes=4, seed=1)

    def run(p):
        R = AccuracyMatrix(len(tasks))
        rng = np.random.default_rng(1)
        for i, t in enumerate(tasks):
            p.fit_task(t.Xtr, t.ytr, epochs=3, rng=rng)
            for j, tt in enumerate(tasks):
                R.record(i, j, accuracy(p.predict_logits(tt.Xte), tt.yte))
        return R

    # G1 per-sample recruit path (site 2): default vs m_max=0 vs m_max > M
    g = dict(d=8, h=24, K=4, n_experts=5, seed=2, z_novel=3.0, commit_after=128,
             train_granularity="sample")
    p1, p2, p3 = Prizma(**g), Prizma(**g, m_max=0), Prizma(**g, m_max=99)
    r1, r2, r3 = run(p1), run(p2), run(p3)
    assert p1.eviction_log == [] and p2.eviction_log == [] and p3.eviction_log == []
    assert np.array_equal(param_vec(p1), param_vec(p2))
    assert np.array_equal(param_vec(p1), param_vec(p3))
    assert r1.acc() == r2.acc() == r3.acc()

    # G3b probation recruit path (site 3): default vs m_max=0
    b = dict(g, probation=True)
    q1, q2 = Prizma(**b), Prizma(**b, m_max=0)
    s1, s2 = run(q1), run(q2)
    assert q1.eviction_log == [] and q2.eviction_log == []
    assert np.array_equal(param_vec(q1), param_vec(q2))
    assert s1.acc() == s2.acc() and s1.forgetting() == s2.forgetting()


# ----------------------------------------------------------------------------- #
# Shipped hard-cap drop path still exists and fires when the lever is OFF       #
# ----------------------------------------------------------------------------- #
def test_shipped_hard_cap_drop_path_fires_when_off():
    # M=2, four distinct levels -> the pool exhausts after the 2nd recruit; every
    # later NOVEL batch must be silently dropped (state untouched), exactly the
    # documented Policy-B worst case. Lever explicitly OFF (m_max=0).
    p = make(m_max=0, n_experts=2)
    stub_levels(p)
    drive(p, [0.5, 1.0, 2.0, 4.0])
    assert p.active >= p.M
    before = param_vec(p)
    n_dropped = 0
    for L in (8.0, 16.0, 32.0):
        drive(p, [L])
        if np.array_equal(param_vec(p), before):
            n_dropped += 1
    assert n_dropped == 3, "pool-exhausted batches must leave state untouched (drop)"
    assert p.eviction_log == []          # lever OFF: no eviction may ever fire


# ----------------------------------------------------------------------------- #
# ON-changes: the lever actually binds and diverges from the unbounded twin     #
# ----------------------------------------------------------------------------- #
def test_on_changes_bounded_diverges_and_never_drops():
    levels = [0.5, 1.0, 2.0, 8.0]
    p_off = make(m_max=0, n_experts=6)
    stub_levels(p_off)
    drive(p_off, levels)
    p_on = make(m_max=2, n_experts=6)
    stub_levels(p_on)
    drive(p_on, levels)
    # unbounded: 4 distinct experts recruited into slots 0..3, no eviction
    assert p_off.eviction_log == []
    assert int((p_off.route_log > 0).sum()) == 4
    # bounded: only 2 usable slots, both reused; trajectory diverges
    assert len(p_on.eviction_log) == 2
    assert not np.array_equal(param_vec(p_off), param_vec(p_on))
    assert p_on.active < 2 and (p_on.route_log[2:] == 0).all()
    assert int((p_on.route_log > 0).sum()) == 2
    # and the bounded run handled every batch (Policy A never silently drops): each
    # batch must change state. route_log SUMS are NOT equal across arms -- an eviction
    # zeroes the victim's ledger entry BY DESIGN (the ledger counts live experts'
    # traffic only), which the [4, 4] tail above already shows -- so the no-drop check
    # compares consecutive full states (route_log + parameters) instead.
    rec = Recorder(d=2, h=4, K=2, n_experts=6, seed=0, commit_after=1, z_novel=5.0,
                   m_max=2)
    stub_levels(rec)
    drive(rec, levels)
    prev = None
    for snap, pv in zip(rec.snaps, rec.psnaps):
        assert (snap.tolist(), pv.tobytes()) != prev, \
            "a batch changed nothing -> it was silently dropped"
        prev = (snap.tolist(), pv.tobytes())


# ----------------------------------------------------------------------------- #
# White-box (b): victim = unique lowest lifetime routing share                  #
# ----------------------------------------------------------------------------- #
def test_victim_is_lowest_share_expert():
    # shares at the capped recruit: slot0=8, slot1=8, slot2=4 -> victim MUST be 2
    p = make(m_max=3)
    stub_levels(p)
    drive(p, [0.5, 0.5, 1.0, 1.0, 2.0, 4.0])
    assert [(r["victim"], r["victim_route_count"]) for r in p.eviction_log] == [(2, 4)]
    fire = p.eviction_log[0]
    assert fire["route_log_at_fire"] == [8, 8, 4, 0, 0, 0]
    tot = max(sum(fire["route_log_at_fire"]), 1)
    assert fire["victim_share"] == pytest.approx(4.0 / tot)
    # the reused slot trained exactly the recruit batch (the new recruit's home)
    assert p.experts[2].n_seen == 4 and p.route_log[2] == 4
    assert p.active == 2                 # active points at the reused slot


# ----------------------------------------------------------------------------- #
# White-box (c): ties break to the most recently recruited (highest index)      #
# ----------------------------------------------------------------------------- #
def test_tie_breaks_to_most_recent():
    # every committed expert trained exactly one 4-sample batch -> 4/4/4 tie
    p = make(m_max=3)
    stub_levels(p)
    drive(p, [0.5, 1.0, 2.0, 8.0])
    fire = p.eviction_log[0]
    assert fire["route_log_at_fire"][:3] == [4, 4, 4]
    assert fire["victim"] == 2           # highest index, NOT slot 0
    assert fire["victim_share"] == pytest.approx(4.0 / 12.0)


# ----------------------------------------------------------------------------- #
# White-box (d): fresh-slot reset identity + eviction count ledger              #
# ----------------------------------------------------------------------------- #
def test_evicted_slot_reset_to_fresh_expert_and_counted():
    p = make(m_max=3)
    stub_levels(p)
    drive(p, [0.5, 1.0, 2.0, 8.0, 16.0])     # recruits 4 and 5 both fire
    assert len(p.eviction_log) == 2          # (d) eviction count recorded
    # direct reset check: the fresh expert is identical to what __init__ would have
    # placed in the slot (same seed formula seed + 100*(m+1), same kwargs)
    p2 = make(m_max=3)
    stub_levels(p2)
    p2.experts[1].committed = True
    p2.route_log[1] = 5
    p2._evict_slot(1)
    fresh = Expert(2, 4, 2, 0 + 100 * (1 + 1))
    e = p2.experts[1]
    assert e.n_seen == 0 and not e.committed and not e.frozen and e.mu > 1e8
    for a, b in ((e.Wenc, fresh.Wenc), (e.benc, fresh.benc), (e.Wdec, fresh.Wdec),
                 (e.bdec, fresh.bdec), (e.Wcls, fresh.Wcls), (e.bcls, fresh.bcls),
                 (e.Bdec, fresh.Bdec), (e.Bcls, fresh.Bcls)):
        assert np.array_equal(a, b)
    assert p2.route_log[1] == 0              # ledger entry zeroed
    assert p2.eviction_log[-1]["victim"] == 1

    # kwargs propagate: a non-default constructor arg survives the slot reset
    p3 = make(m_max=3, act_bits=8)
    p3.experts[0].committed = True
    p3._evict_slot(0)
    ref = Expert(2, 4, 2, 0 + 100 * 1, act_bits=8)
    assert p3.experts[0].act_bits == 8
    assert np.array_equal(p3.experts[0].Wenc, ref.Wenc)


# ----------------------------------------------------------------------------- #
# White-box (a): cap invariant over the WHOLE block stream                      #
# ----------------------------------------------------------------------------- #
def test_cap_invariant_throughout_stream():
    p = Recorder(d=2, h=4, K=2, n_experts=8, seed=0, commit_after=1, z_novel=5.0,
                 m_max=3)
    stub_levels(p)
    drive(p, [0.5, 1.0, 2.0, 8.0, 16.0])     # 5 recruits > cap -> 2 evictions
    assert len(p.snaps) == 5
    for s in p.snaps:                        # (a) never more than m_max live slots,
        assert int((s > 0).sum()) <= 3       # never any traffic beyond the cap
        assert (s[3:] == 0).all()
    assert int((p.route_log > 0).sum()) == 3
    assert len(p.eviction_log) == 2
    assert p.active == 2


# ----------------------------------------------------------------------------- #
# Edge ruling: m_max == M is a live cap (replaces the exhaustion refusal)       #
# ----------------------------------------------------------------------------- #
def test_mmax_equal_M_is_a_live_cap():
    # M=3, m_max=3: shipped behaviour would advance active to 3 (== M) and silently
    # drop every later batch; Policy A instead evicts at the pool edge and the 4th
    # domain trains into the reused slot.
    p = make(m_max=3, n_experts=3)
    stub_levels(p)
    drive(p, [0.5, 1.0, 2.0, 8.0])
    assert len(p.eviction_log) == 1 and p.eviction_log[0]["victim"] == 2
    assert p.active < 3 and int((p.route_log > 0).sum()) == 3
    assert p.experts[2].n_seen == 4          # 4th domain WAS trained (not dropped)


# ----------------------------------------------------------------------------- #
# Knob hygiene                                                                  #
# ----------------------------------------------------------------------------- #
def test_validation():
    with pytest.raises(ValueError):
        Prizma(d=2, h=4, K=2, n_experts=2, seed=0, m_max=-1)
    with pytest.raises(ValueError):
        Prizma(d=2, h=4, K=2, n_experts=2, seed=0, m_max=0.5)
