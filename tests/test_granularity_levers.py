"""Tests for the training-granularity levers (G1/G2; fusion spec Addendum 2026-09-05).

Two Prizma-config knobs, BOTH DEFAULT-OFF, attacking the PR-06 blocker (the batch-level
TRAINING GRANULARITY: train_batch applied the whole batch's delta to the single active
expert, so a mixed batch could never be split across experts -- one expert trained on
100% of the interleaved stream, the monolithic ceiling ~0.60):

  train_granularity  -- "batch" (shipped) | "sample" (G1): each sample is routed
                        individually against the per-expert calibrated per-sample
                        thresholds; each sample trains the expert that recognizes IT;
                        the unrecognized remainder forms a per-batch novel pool and
                        recruitment fires iff its fraction >= 0.5 (existing recruit
                        machinery: commit + freeze_min_seen-vetoed freeze + advance,
                        new active trained on the novel pool).
  session_window     -- 0 (shipped) | W > 0 (G2): the batch is split into consecutive
                        W-sample windows (last one short when batch % W != 0); each
                        window takes the shipped batch-level decision at window
                        granularity (window-MEAN surprise vs the same thresholds; novel
                        window -> recruit, veto still applies).

Contract under test:

  OFF-identity:   defaults == explicit-off kwargs, BIT-IDENTICAL (param trajectories,
                  route_log, flags, ACC/FGT) on both a mixed-batch stream and a block
                  stream. The shipped path is the shipped body verbatim.
  ON-changes:     G1 and G2/W=8 each change behaviour on a tiny mixed stream.
  White-box G1:   a constructed mixed batch whose two samples are recognized by
                  different experts is split across TWO experts by ONE train_batch call;
                  a sub-threshold novel minority (< 0.5) is not trained and does not
                  recruit; a committed expert claims recognized samples without any
                  re-training (n_seen and weights untouched).
  White-box G2:   session_window=1 makes every window decision a PER-SAMPLE decision
                  (equivalence asserted against the per-sample threshold test, while the
                  shipped batch-mean decision provably absorbs both); window boundaries
                  handle batch % W != 0 (decision sizes [8, 2] for a 10-batch at W=8).
  Knob hygiene:   route_stat is inert under G1/G2 (documented in the docstrings);
                  invalid combinations raise ValueError.
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
from src.prizma import Prizma, SAMPLE_NOVEL_FRACTION

OFF = dict(train_granularity="batch", session_window=0)


# ----------------------------------------------------------------------------- #
# Tiny stream runners (seconds; the mixed stream is the G1/G2 target regime)     #
# ----------------------------------------------------------------------------- #
def mixed_run(p, epochs=2, seed=0, K=3, d=8, ncls=4, samples=192, batch=64):
    """Tiny mixed-batch stream: samples of all K domains shuffled into ONE stream
    (mirrors expert_economy's mixed construction; batches are mixtures)."""
    tasks = structured_permuted_tasks(n_tasks=K, n_samples=samples, d=d, k_latent=4,
                                      n_classes=ncls, seed=seed)
    Xall = np.vstack([t.Xtr for t in tasks])
    yall = np.concatenate([t.ytr for t in tasks])
    perm = np.random.default_rng(1000 + seed).permutation(len(Xall))
    p.fit_task(Xall[perm], yall[perm], epochs=epochs, batch=batch,
               rng=np.random.default_rng(seed))
    return float(np.mean([accuracy(p.predict_logits(t.Xte), t.yte) for t in tasks]))


def block_run(p, epochs=3, seed=0):
    """Shipped E-protocol on a tiny 2-domain block stream -> AccuracyMatrix."""
    tasks = structured_permuted_tasks(n_tasks=2, n_samples=512, d=8, k_latent=4,
                                      n_classes=4, seed=0)
    R = AccuracyMatrix(len(tasks))
    rng = np.random.default_rng(seed)
    for i, t in enumerate(tasks):
        p.fit_task(t.Xtr, t.ytr, epochs=epochs, rng=rng)
        for j, tt in enumerate(tasks):
            R.record(i, j, accuracy(p.predict_logits(tt.Xte), tt.yte))
    return R


def make(K=3, **kw):
    return Prizma(d=8, h=24, K=4, n_experts=K + 2, seed=0, z_novel=3.0,
                  commit_after=128, **kw)


def param_vec(p):
    parts = []
    for e in p.experts:
        parts.extend([e.Wenc.ravel(), e.benc, e.Wdec.ravel(), e.bdec,
                      e.Wcls.ravel(), e.bcls, np.array([e.mu, e.var])])
    return np.concatenate(parts)


def economy(p):
    return ([int(e.frozen) for e in p.experts],
            [int(e.committed) for e in p.experts], p.route_log.tolist())


THR_SCALE = 0.10 + 5.0 * 0.02      # crafted floor: mu=0.10, sigma=0.02, z=5 -> 0.20


def stub_floor(p, m, per_sample_surprise):
    """Give expert m a calibrated floor (mu=0.10, sigma=0.02 -> threshold 0.20 at z=5)
    and a row-KEYED stub recon_error: each input row maps to its crafted surprise, so
    sub-batch/window calls at any offset see the right values. Pass a scalar for a
    constant surprise, or {(row-tuple): surprise} for per-row values."""
    e = p.experts[m]
    e.mu, e.var = 0.10, 4e-4
    if not isinstance(per_sample_surprise, dict):
        per_sample_surprise = {"*": float(per_sample_surprise)}
    tbl = {tuple(np.round(np.asarray(k, np.float32), 6)): float(v)
           for k, v in per_sample_surprise.items() if k != "*"}

    def _recon(X, Y=None):
        Xa = np.asarray(X, np.float32)
        if "*" in per_sample_surprise:
            return np.full(len(Xa), float(per_sample_surprise["*"]), np.float32)
        return np.array([tbl[tuple(np.round(r, 6))] for r in Xa], np.float32)

    e.recon_error = _recon
    return e


# ----------------------------------------------------------------------------- #
# 1. OFF-identity: bit-identical vs defaults (NON-NEGOTIABLE)                    #
# ----------------------------------------------------------------------------- #
def test_off_identity_mixed_stream_full_trajectory():
    p_ref, p_off = make(), make(**OFF)
    mixed_run(p_ref)
    mixed_run(p_off)
    assert np.array_equal(param_vec(p_ref), param_vec(p_off)), \
        "explicit-off kwargs changed the parameter trajectory on the mixed stream"
    assert economy(p_ref) == economy(p_off)
    assert p_ref.route_log.tolist() == p_off.route_log.tolist()


def test_off_identity_block_stream_acc_fgt_and_params():
    R_ref = block_run(Prizma(d=8, h=24, K=4, n_experts=5, seed=0))
    R_off = block_run(Prizma(d=8, h=24, K=4, n_experts=5, seed=0, **OFF))
    assert np.array_equal(R_ref.R, R_off.R), "explicit-off kwargs changed the ACC matrix"
    assert R_ref.acc() == R_off.acc() and R_ref.forgetting() == R_off.forgetting()


# ----------------------------------------------------------------------------- #
# 2. Each lever ON changes behaviour (the PR-06 blocker regime: mixed batches)   #
# ----------------------------------------------------------------------------- #
def test_base_mixed_stream_is_monolithic():
    """The stream under test exhibits the PR-06 blocker: the shipped batch-level
    granularity trains ONE expert on (nearly) all samples of the mixed stream."""
    p = make()
    mixed_run(p)
    route = p.route_log.tolist()
    tot = sum(route)
    assert max(route) / tot > 0.95, f"expected a monolithic ledger, got {route}"


@pytest.mark.parametrize("kw", [dict(train_granularity="sample"),
                                dict(session_window=8)])
def test_each_granularity_lever_on_changes_behaviour(kw):
    p_ref = make()
    mixed_run(p_ref)
    ref_params = param_vec(p_ref)
    p_on = make(**kw)
    acc = mixed_run(p_on)
    assert not np.array_equal(param_vec(p_on), ref_params), \
        f"lever {kw} ON did not change any parameter -- silently inert"
    # (G1 splits routing end-to-end; W=8 may keep the same monolithic ROUTING on this
    # tiny stream -- its decision-unit change is pinned by the W=1 and boundary tests.)
    assert 0.0 <= acc <= 1.0


# ----------------------------------------------------------------------------- #
# 3. White-box G1: ONE mixed batch split across TWO experts                      #
# ----------------------------------------------------------------------------- #
def _g1_setup():
    """Warm the active expert with one all-novel batch (G1: fraction 1.0 >= 0.5 and
    n_seen == 0 -> the active takes the whole batch as its first training, no commit),
    then craft its floor: threshold 0.20, samples in-floor at 0.04, novel at 0.28."""
    p = Prizma(d=2, h=4, K=2, n_experts=3, seed=0, train_granularity="sample")
    X0 = np.array([[0.0, 0.0], [0.0, 1.0]], np.float32)
    Y0 = np.eye(2, dtype=np.float32)[[0, 1]]
    p.train_batch(X0, Y0, np.array([0, 1]))
    assert p.route_log.tolist() == [2, 0, 0] and p.experts[0].n_seen == 2
    assert not p.experts[0].committed, "first training must not commit the active"
    return p


def test_g1_splits_mixed_batch_across_two_experts():
    p = _g1_setup()
    X1 = np.array([[1.0, 0.0], [1.0, 1.0]], np.float32)
    stub_floor(p, 0, {(1.0, 0.0): 0.04, (1.0, 1.0): 0.28})   # A recognized, B novel
    Y1 = np.eye(2, dtype=np.float32)[[1, 0]]
    p.train_batch(X1, Y1, np.array([1, 0]))
    # A -> the active (e0) trains on it; B -> the novel pool is 1/2 = 0.5 >= 0.5
    # -> recruitment fires: e0 commits+freezes, the new active e1 trains on B.
    assert p.route_log.tolist() == [3, 1, 0]
    assert p.experts[0].committed and p.experts[0].frozen
    assert p.active == 1 and p.experts[1].n_seen == 1 and p.route_log[1] == 1


def test_g1_subthreshold_novel_minority_is_not_trained_and_does_not_recruit():
    p = _g1_setup()
    X1 = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]], np.float32)
    stub_floor(p, 0, {(0.0, 0.0): 0.04, (0.0, 1.0): 0.04, (1.0, 1.0): 0.28})
    Y1 = np.eye(2, dtype=np.float32)[[0, 0, 1]]
    p.train_batch(X1, Y1, np.array([0, 0, 1]))
    assert p.route_log.tolist() == [4, 0, 0], "novel minority must not recruit or train"
    assert not p.experts[0].committed and p.active == 0
    assert p.experts[1].n_seen == 0 and p.experts[1].mu > 1e8


def test_g1_committed_expert_claims_without_retraining():
    p = _g1_setup()
    p.experts[0].committed = True           # force the committed state (protected)
    X1 = np.array([[1.0, 0.0], [1.0, 1.0]], np.float32)
    stub_floor(p, 0, {(1.0, 0.0): 0.04, (1.0, 1.0): 0.28})   # e0 claims sample A only
    Wdec0 = np.array(p.experts[0].Wdec)
    Y1 = np.eye(2, dtype=np.float32)[[1, 0]]
    p.train_batch(X1, Y1, np.array([1, 0]))
    # sample A -> committed e0 claims it with NO training; sample B -> novel pool
    # (1/2 >= 0.5) -> recruitment: the active e0 is already committed, so it advances
    # (no double-commit) and e1 takes the novel sample.
    assert p.experts[0].n_seen == 2, "committed claim must not re-train the expert"
    assert np.array_equal(p.experts[0].Wdec, Wdec0)
    assert p.route_log.tolist() == [3, 1, 0] and p.active == 1


# ----------------------------------------------------------------------------- #
# 4. White-box G2: W=1 == per-sample decisions; window boundaries                #
# ----------------------------------------------------------------------------- #
def test_g2_w1_equals_per_sample_routing_decisions():
    """session_window=1: every window is one sample, so the window decision (window MEAN
    of a single sample) equals the per-sample decision against the SAME threshold --
    asserted directly, while the shipped batch-mean decision provably absorbs both."""
    d, h = 2, 4
    X1 = np.array([[1.0, 0.0], [1.0, 1.0]], np.float32)
    crafted = {(1.0, 0.0): 0.04, (1.0, 1.0): 0.28}      # per-row surprise
    Y1 = np.eye(2, dtype=np.float32)[[1, 0]]
    y1 = np.array([1, 0])
    Z2 = np.zeros((2, d), np.float32)
    Ywarm = np.eye(2, dtype=np.float32)[[0, 1]]

    # reference per-sample decisions against the crafted floor: threshold = mu + z*sigma
    # = 0.10 + 5*0.02 = 0.20; surprises 0.04 / 0.28.
    thr = THR_SCALE
    per_sample = [crafted[(1.0, 0.0)] <= thr, crafted[(1.0, 1.0)] <= thr]
    assert per_sample == [True, False]

    # W=1 twin: warm one 2-sample batch (young -> both windows train), craft, feed.
    p1 = Prizma(d=d, h=h, K=2, n_experts=3, seed=0, commit_after=2, session_window=1)
    p1.train_batch(Z2, Ywarm, np.array([0, 1]))
    assert p1.experts[0].n_seen == 2
    stub_floor(p1, 0, crafted)
    p1.train_batch(X1, Y1, y1)
    # per-sample decisions reproduced exactly: sample 0 -> e0 trains; sample 1 -> NOT
    # recognized by e0 -> commit+freeze+advance -> e1 trains on it.
    assert p1.route_log.tolist() == [3, 1, 0]
    assert p1.experts[0].committed and p1.active == 1

    # shipped twin (W=0): the single batch-MEAN decision (0.16 <= 0.20) absorbs BOTH
    # samples -- the blindness that W=1 removes.
    p0 = Prizma(d=d, h=h, K=2, n_experts=3, seed=0, commit_after=2)
    p0.train_batch(Z2, Ywarm, np.array([0, 1]))
    stub_floor(p0, 0, crafted)
    p0.train_batch(X1, Y1, y1)
    assert p0.route_log.tolist() == [4, 0, 0] and p0.active == 0


def test_g2_window_boundary_batch_not_divisible():
    """batch % W != 0 -> last SHORT window (decisions at sizes [8, 2] for a 10-batch);
    W > n -> a single whole-batch window. A constant-surprise stub makes every window
    decision 'recognized', so the ledger is deterministic."""
    d = 4
    X = np.random.default_rng(0).normal(0, 1, (10, d)).astype(np.float32)
    Y = np.eye(2, dtype=np.float32)[[0] * 10]
    y = np.zeros(10, np.int64)

    p = Prizma(d=d, h=8, K=2, n_experts=2, seed=0, commit_after=5, session_window=8)
    stub_floor(p, 0, 0.04)
    sizes = []
    real = p._recognizes_mean
    p._recognizes_mean = lambda e, X: (sizes.append(len(X)), real(e, X))[1]
    p.train_batch(X, Y, y)
    assert sizes == [8, 2], f"expected window sizes [8, 2], got {sizes}"
    assert p.route_log.tolist() == [10, 0]

    p2 = Prizma(d=d, h=8, K=2, n_experts=2, seed=0, commit_after=5, session_window=16)
    stub_floor(p2, 0, 0.04)
    sizes2 = []
    real2 = p2._recognizes_mean
    p2._recognizes_mean = lambda e, X: (sizes2.append(len(X)), real2(e, X))[1]
    p2.train_batch(X, Y, y)
    assert sizes2 == [10] and p2.route_log.tolist() == [10, 0]


# ----------------------------------------------------------------------------- #
# 5. Knob hygiene: route_stat inert under G1/G2; invalid configs raise           #
# ----------------------------------------------------------------------------- #
@pytest.mark.parametrize("kw", [dict(train_granularity="sample"),
                                dict(session_window=8)])
def test_route_stat_is_inert_under_granularity_levers(kw):
    common = dict(d=8, h=24, K=4, n_experts=5, seed=0, z_novel=3.0, commit_after=128)
    pa, pb = Prizma(**common, **kw), Prizma(**common, route_stat="sample_top", **kw)
    mixed_run(pa)
    mixed_run(pb)
    assert np.array_equal(param_vec(pa), param_vec(pb)), \
        "route_stat must not be consulted when the decision unit is not the batch"
    assert economy(pa) == economy(pb)


def test_invalid_granularity_configs_raise():
    with pytest.raises(ValueError):
        Prizma(d=4, h=4, K=2, n_experts=2, seed=0, train_granularity="window")
    with pytest.raises(ValueError):
        Prizma(d=4, h=4, K=2, n_experts=2, seed=0, session_window=-3)
    with pytest.raises(ValueError):
        Prizma(d=4, h=4, K=2, n_experts=2, seed=0, train_granularity="sample",
               session_window=8)


def test_novel_fraction_constant_is_the_g1_recruitment_bar():
    assert SAMPLE_NOVEL_FRACTION == 0.5
