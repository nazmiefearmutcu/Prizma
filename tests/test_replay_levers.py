"""Tests for the G3a generative replay-before-freeze lever (PR-2026-09-03-07).

ONE Prizma-config knob, DEFAULT-OFF, attacking the granularity probe's measured
blocker (results/exploratory/granularity_probe_2026-09-05/): commit-on-recruit grants
a specialist ONE unrepeated pass over its fragment where the block stream grants ~15
epochs -- ACQUISITION VOLUME binds. G3a (Deep Generative Replay, Shin et al. 2017 --
ADAPTED: generator = the expert's own PC decoder driven by a diagonal-Gaussian fit of
its lifetime hidden activity; no GAN, no buffer): at a PGM freeze the expert generates
replay_items pseudo-inputs x~ = h~ @ Wdec^T + bdec, self-labels them ONCE with its own
pre-replay head, pseudo-trains replay_passes passes with its OWN local rule
(bookkeeping-free), and only THEN freezes.

Contract under test:

  OFF-identity:   defaults == explicit-off kwargs, BIT-IDENTICAL (parameter
                  trajectories, route_log, flags, ACC/FGT) on mixed and block streams.
                  Identity of trajectories also proves zero RNG consumption when off
                  (any draw would shift every later one).
  ON-changes:     at a real freeze event the replayed expert's weights differ from the
                  OFF twin while every OTHER expert is bit-identical (per-expert RNG
                  isolation) and the new active trains on the same batch identically.
  White-box G3a:  exactly replay_passes pseudo-training calls of exactly replay_items
                  rows fire at the freeze; weights move; n_seen / precision floor
                  (mu/var) / h-statistics / route_log measure REAL data only;
                  diagnostics record items/passes + pre/post surrogate values.
  No-op guard:    replay on an expert with no hidden statistics (never trained) is a
                  silent no-op inside the freeze (no events, no rng).
  Composition:    G1 (train_granularity="sample") + replay compose: the G1 recruit
                  freezes with replay before the new active takes the novel pool.
  Knob hygiene:   invalid values raise; the frozen internal constants are pinned.

G3b (probationary commit) is deliberately NOT built in this pass (owner directive:
G3a first, smaller delta), so there is no probation test.
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
from src.prizma import Prizma, REPLAY_H_EMA, REPLAY_VAR_FLOOR

OFF = dict(replay_passes=0, replay_items=256)


# ----------------------------------------------------------------------------- #
# Tiny stream runners (seconds)                                                  #
# ----------------------------------------------------------------------------- #
def mixed_run(p, epochs=2, seed=0, K=3, d=8, ncls=4, samples=192, batch=64):
    """Tiny mixed-batch stream (mirrors the granularity-lever tests' runner)."""
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


def make(**kw):
    return Prizma(d=8, h=24, K=4, n_experts=5, seed=0, z_novel=3.0,
                  commit_after=128, **kw)


def param_vec(p):
    parts = []
    for e in p.experts:
        parts.extend([e.Wenc.ravel(), e.benc, e.Wdec.ravel(), e.bdec,
                      e.Wcls.ravel(), e.bcls, np.array([e.mu, e.var])])
    return np.concatenate(parts)


def stub_floor(p, m, surprise):
    """Expert m: calibrated floor (mu=0.10, sigma=0.02 -> threshold 0.20 at z=5) and a
    constant-surprise stub recon_error (surprise > threshold -> never recognizes)."""
    e = p.experts[m]
    e.mu, e.var = 0.10, 4e-4
    e.recon_error = lambda X, Y=None: np.full(len(X), float(surprise), np.float32)
    return e


def freeze_script(replay_passes, replay_items=16):
    """Deterministic freeze event on the shipped batch path: warm the active expert
    (matures it via commit_after=2), stub its floor, feed a novel batch -> commit +
    freeze (+ replay when ON). Returns (p, weights_before, n_seen_before)."""
    p = Prizma(d=2, h=4, K=2, n_experts=3, seed=0, z_novel=5.0, commit_after=2,
               replay_passes=replay_passes, replay_items=replay_items)
    X0 = np.array([[0.0, 0.0], [0.0, 1.0]], np.float32)
    Y0 = np.eye(2, dtype=np.float32)[[0, 1]]
    p.train_batch(X0, Y0, np.array([0, 1]))
    e0 = p.experts[0]
    assert e0.n_seen == 2
    w_before = e0.Wdec.copy()
    ns_before = e0.n_seen
    stub_floor(p, 0, 0.28)                       # novel batch: 0.28 > threshold 0.20
    X1 = np.array([[1.0, 0.0], [1.0, 1.0]], np.float32)
    Y1 = np.eye(2, dtype=np.float32)[[1, 0]]
    p.train_batch(X1, Y1, np.array([1, 0]))
    assert p.experts[0].frozen and p.active == 1
    return p, w_before, ns_before


# ----------------------------------------------------------------------------- #
# 1. OFF-identity: bit-identical vs defaults (NON-NEGOTIABLE)                    #
# ----------------------------------------------------------------------------- #
def test_off_identity_mixed_stream_full_trajectory():
    p_ref, p_off = make(), make(**OFF)
    mixed_run(p_ref)
    mixed_run(p_off)
    assert np.array_equal(param_vec(p_ref), param_vec(p_off)), \
        "explicit-off kwargs changed the parameter trajectory on the mixed stream"
    assert p_ref.route_log.tolist() == p_off.route_log.tolist()
    assert p_ref.replay_events == [] and p_off.replay_events == []


def test_off_identity_block_stream_acc_fgt_and_params():
    R_ref = block_run(Prizma(d=8, h=24, K=4, n_experts=5, seed=0))
    R_off = block_run(Prizma(d=8, h=24, K=4, n_experts=5, seed=0, **OFF))
    assert np.array_equal(R_ref.R, R_off.R), "explicit-off kwargs changed the ACC matrix"
    assert R_ref.acc() == R_off.acc() and R_ref.forgetting() == R_off.forgetting()


# ----------------------------------------------------------------------------- #
# 2. ON changes behaviour exactly at the freeze event                            #
# ----------------------------------------------------------------------------- #
def test_replay_on_changes_frozen_expert_only():
    p_off, w_off, ns_off = freeze_script(replay_passes=0)
    p_on, w_on, ns_on = freeze_script(replay_passes=3)
    e0_off, e0_on = p_off.experts[0], p_on.experts[0]
    assert p_off.replay_events == [] and len(p_on.replay_events) == 1
    assert not np.array_equal(e0_on.Wdec, w_off), \
        "replay ON did not move the frozen expert's weights -- silently inert"
    assert not np.array_equal(e0_on.Wdec, w_on), "recorded pre-freeze weight is stale"
    # per-expert RNG isolation: every OTHER expert is bit-identical across ON/OFF
    for m in (1, 2):
        for attr in ("Wenc", "benc", "Wdec", "bdec", "Wcls", "bcls"):
            assert np.array_equal(getattr(p_on.experts[m], attr),
                                  getattr(p_off.experts[m], attr))
    assert p_on.route_log.tolist() == p_off.route_log.tolist()


def test_g3a_generates_exactly_items_x_passes_at_freeze():
    p, _, _ = freeze_script(replay_passes=3, replay_items=16)
    ev = p.replay_events[0]
    assert ev["items"] == 16 and ev["passes"] == 3
    assert ev["expert"] == 0 and ev["n_seen_real"] == 2


def test_replay_pseudo_training_calls_are_items_sized_and_bookkeeping_free():
    p = Prizma(d=2, h=4, K=2, n_experts=3, seed=0, z_novel=5.0, commit_after=2,
               replay_passes=3, replay_items=16)
    X0 = np.array([[0.0, 0.0], [0.0, 1.0]], np.float32)
    p.train_batch(X0, np.eye(2, dtype=np.float32)[[0, 1]], np.array([0, 1]))
    stub_floor(p, 0, 0.28)
    calls = []
    real = p._train_expert

    def spy(e, X, Y, _bookkeeping=True):
        calls.append((len(X), _bookkeeping))
        return real(e, X, Y, _bookkeeping)

    p._train_expert = spy
    X1 = np.array([[1.0, 0.0], [1.0, 1.0]], np.float32)
    p.train_batch(X1, np.eye(2, dtype=np.float32)[[1, 0]], np.array([1, 0]))
    replay_calls = [c for c in calls if not c[1]]
    real_calls = [c for c in calls if c[1]]
    assert len(replay_calls) == 3 and all(n == 16 for n, _ in replay_calls), \
        f"expected 3 replay calls of exactly 16 items, got {replay_calls}"
    # the spy went in AFTER the warm batch, so the only real-data call it sees is the
    # NEW active training on the 2-sample novel batch (bookkeeping on)
    assert len(real_calls) == 1 and real_calls[0] == (2, True)


def test_replay_moves_weights_but_not_real_data_ledgers():
    p, _, ns_before = freeze_script(replay_passes=4, replay_items=16)
    e0 = p.experts[0]
    assert e0.n_seen == ns_before, "replay must not inflate the real-sample ledger"
    assert (e0.mu, e0.var) == (0.10, 4e-4), "replay must not re-calibrate the floor"
    assert p.route_log.tolist() == [2, 2, 0], "route_log counts REAL samples only"
    ev = p.replay_events[0]
    assert ev["recon_post"] < ev["recon_pre"], \
        "consolidation should reduce the pseudo-set's reconstruction error"
    assert 0.0 <= ev["conf_pre"] <= 1.0 and 0.0 <= ev["conf_post"] <= 1.0


def test_h_statistics_track_real_data_only_and_are_replay_immune():
    p = Prizma(d=2, h=4, K=2, n_experts=3, seed=0, z_novel=5.0, commit_after=2,
               replay_passes=2, replay_items=8)
    B1 = np.array([[0.0, 0.0], [0.0, 1.0]], np.float32)
    B2 = np.array([[1.0, 0.0], [1.0, 1.0]], np.float32)
    Y = np.eye(2, dtype=np.float32)[[0, 1]]
    # the h-statistic is maintained from the PRE-update latents inside the training
    # step, so the expected values must be computed from weight snapshots
    W0, b0 = p.experts[0].Wenc.copy(), p.experts[0].benc.copy()
    p.train_batch(B1, Y, np.array([0, 1]))
    e0 = p.experts[0]
    Z1 = np.tanh(B1 @ W0.T + b0)
    assert e0.h_mean is not None, "h-statistics must be maintained when replay is ON"
    assert np.allclose(e0.h_mean, Z1.mean(0), atol=1e-6)
    assert np.allclose(e0.h_var, np.maximum(Z1.var(0), REPLAY_VAR_FLOOR), atol=1e-6)
    e0.var = 0.04        # widen the floor: B2 must be RECOGNIZED (otherwise the shipped
                         # recruit path fires and hands B2 to a new expert instead)
    W1, b1 = e0.Wenc.copy(), e0.benc.copy()
    hm_prev, hv_prev = e0.h_mean.copy(), e0.h_var.copy()
    p.train_batch(B2, Y, np.array([0, 1]))
    assert p.route_log.tolist() == [4, 0, 0], "e0 must have trained on B2"
    Z2 = np.tanh(B2 @ W1.T + b1)
    m2 = (1 - REPLAY_H_EMA) * hm_prev + REPLAY_H_EMA * Z2.mean(0)
    v2 = (1 - REPLAY_H_EMA) * hv_prev + REPLAY_H_EMA * np.maximum(Z2.var(0),
                                                                   REPLAY_VAR_FLOOR)
    assert np.allclose(e0.h_mean, m2, atol=1e-6)
    assert np.allclose(e0.h_var, v2, atol=1e-6)
    # replay immunity: after the freeze event the statistics still equal the last
    # REAL-data EMA values (pseudo-training is bookkeeping-free)
    h_mean, h_var = e0.h_mean.copy(), e0.h_var.copy()
    stub_floor(p, 0, 0.28)
    p.train_batch(B1, Y, np.array([0, 1]))       # triggers freeze + replay
    assert len(p.replay_events) == 1
    assert np.array_equal(e0.h_mean, h_mean) and np.array_equal(e0.h_var, h_var)


# ----------------------------------------------------------------------------- #
# 3. No-op guard + knob hygiene                                                  #
# ----------------------------------------------------------------------------- #
def test_replay_is_noop_without_hidden_statistics():
    """A never-trained active (no h-statistics) freezes with replay silently skipped."""
    p = Prizma(d=2, h=4, K=2, n_experts=2, seed=0, z_novel=5.0, commit_after=0,
               replay_passes=3, replay_items=8)
    X = np.array([[0.0, 0.0], [1.0, 1.0]], np.float32)
    p.train_batch(X, np.eye(2, dtype=np.float32)[[0, 1]], np.array([0, 1]))
    assert p.experts[0].frozen and p.experts[0].h_mean is None
    assert p.replay_events == [], "replay must not fire without real-data statistics"


def test_replay_constants_are_pinned():
    assert REPLAY_H_EMA == 0.05
    assert REPLAY_VAR_FLOOR == 1e-4


@pytest.mark.parametrize("kw", [dict(replay_passes=-1), dict(replay_passes=1.5),
                                dict(replay_items=0), dict(replay_items=-4)])
def test_invalid_replay_configs_raise(kw):
    with pytest.raises(ValueError):
        Prizma(d=4, h=4, K=2, n_experts=2, seed=0, **kw)


# ----------------------------------------------------------------------------- #
# 4. Composition: G1 per-sample granularity + replay compose                     #
# ----------------------------------------------------------------------------- #
def test_g1_sample_granularity_composes_with_replay():
    """G1 recruit (novel fraction >= 0.5) freezes the old active WITH replay before the
    new active takes the novel pool -- both levers visible in one train_batch call."""
    p = Prizma(d=2, h=4, K=2, n_experts=3, seed=0, z_novel=5.0,
               train_granularity="sample", replay_passes=2, replay_items=8)
    X0 = np.array([[0.0, 0.0], [0.0, 1.0]], np.float32)
    Y0 = np.eye(2, dtype=np.float32)[[0, 1]]
    p.train_batch(X0, Y0, np.array([0, 1]))      # all-novel -> active takes it whole
    assert p.route_log.tolist() == [2, 0, 0] and not p.experts[0].committed
    e0 = p.experts[0]
    e0.mu, e0.var = 0.10, 4e-4
    tbl = {tuple(np.round(r, 6)): v for r, v in
           zip([X0[0], X0[1], [1.0, 0.0], [1.0, 1.0]], [0.04, 0.04, 0.04, 0.28])}
    e0.recon_error = lambda X, Y=None: np.array(
        [tbl[tuple(np.round(r, 6))] for r in np.asarray(X, np.float32)], np.float32)
    ns_before = e0.n_seen
    X1 = np.array([[1.0, 0.0], [1.0, 1.0]], np.float32)
    Y1 = np.eye(2, dtype=np.float32)[[1, 0]]
    p.train_batch(X1, Y1, np.array([1, 0]))
    # G1 ledger: A recognized -> e0 trains; B novel (fraction 0.5) -> recruit:
    # e0 commits + freezes WITH replay, e1 takes the novel sample.
    assert p.route_log.tolist() == [3, 1, 0]
    assert e0.frozen and p.active == 1
    assert len(p.replay_events) == 1 and p.replay_events[0]["expert"] == 0
    # real ledger: e0 trained on the warm batch (2) + sample A (1); replay adds nothing
    assert e0.n_seen == ns_before + 1, \
        "G1+replay must keep the real-sample ledger honest (only sample A is real)"
