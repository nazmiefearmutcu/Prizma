"""Tests for the interleaved-router levers (PR-2026-09-03-06; synthesis C4, fusion spec 3.3).

Four Prizma-config knobs, ALL DEFAULT-OFF, attacking the two documented interleaved
collapse mechanisms (docs/EXPERT_ECONOMY.md 2.2):

  freeze_min_seen    -- freeze-immaturity veto: PGM consolidation may not freeze an
                        expert with n_seen < floor (round-robin failure mode (b)).
  dynamic_vigilance  -- novelty-rate-modulated vigilance threshold (regime adaptation).
  hot_young          -- per-expert metaplasticity: young experts train hot (fusion 3.3-B).
  route_stat         -- "sample_top": novel-FRACTION statistic instead of batch mean
                        (mixed-batch failure mode (a): the mean is stationary, the
                        per-sample labels are bimodal).

Contract under test:

  OFF-identity:  defaults == explicit-off kwargs, BIT-IDENTICAL: same ACC/FGT on a tiny
                 2-domain block stream AND identical parameter trajectories (weights,
                 mu/var, route_log, frozen flags) -- on BOTH a block stream and a
                 round-robin stream (the latter exercises the veto/novelty bookkeeping).
  ON-changes:    every lever changes observable behaviour on a tiny round-robin stream
                 (the collapse regime: the base run reproduces the immature-freeze
                 catch-all, route_log = [744, 1104, 0, ...], frozen at n_seen=128).
  Unit level:    the veto actually blocks the freeze below the floor; the hot_young
                 multiplier matches 1 + hot*exp(-n/200) and scales every local delta
                 step; route_stat='sample_top' fires the recruit on a constructed 60/40
                 mixed batch where batch_mean stays blind; the dynamic-vigilance EMA and
                 clamp follow the documented formula.
"""
import math
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
from src.prizma import (DV_ALPHA_FAST, DV_ALPHA_SLOW, DV_THETA_HI, DV_THETA_LO,
                        HOT_YOUNG_TAU, SAMPLE_NOVEL_FRACTION, Expert, Prizma)

OFF = dict(freeze_min_seen=0, dynamic_vigilance=0.0, hot_young=0.0,
           route_stat="batch_mean")


# ----------------------------------------------------------------------------- #
# Tiny stream runners (seconds; round-robin mirrors expert_economy's stream)     #
# ----------------------------------------------------------------------------- #
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


def rr_run(p, K=3, d=8, ncls=4, samples=384, batch=64, epochs=2, h=24, seed=0):
    """Tiny round-robin stream: domain-PURE batches, batch order shuffled (the regime
    where the shipped model freezes an immature expert that becomes a catch-all)."""
    tasks = structured_permuted_tasks(n_tasks=K, n_samples=samples, d=d, k_latent=4,
                                      n_classes=ncls, seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        batches = []
        for t in tasks:
            Yall = np.eye(p.K, dtype=np.float32)[t.ytr]
            idx = rng.permutation(len(t.Xtr))
            for s in range(0, len(idx), batch):
                bi = idx[s:s + batch]
                batches.append((t.Xtr[bi], Yall[bi], t.ytr[bi]))
        for b in rng.permutation(len(batches)):
            Xb, Yb, yb = batches[int(b)]
            p.train_batch(Xb, Yb, yb)
    acc = float(np.mean([accuracy(p.predict_logits(t.Xte), t.yte) for t in tasks]))
    return acc


def make_rr(**kw):
    return Prizma(d=8, h=24, K=4, n_experts=5, seed=0, z_novel=3.0,
                  commit_after=128, **kw)


def param_vec(p):
    """Full parameter trajectory: trainable weights + the calibrated floors (mu/var)."""
    parts = []
    for e in p.experts:
        parts.extend([e.Wenc.ravel(), e.benc, e.Wdec.ravel(), e.bdec,
                      e.Wcls.ravel(), e.bcls, np.array([e.mu, e.var])])
    return np.concatenate(parts)


def economy(p):
    return ([int(e.frozen) for e in p.experts],
            [int(e.committed) for e in p.experts], p.route_log.tolist())


# ----------------------------------------------------------------------------- #
# 1. OFF-identity: bit-identical vs a reference instance (NON-NEGOTIABLE)        #
# ----------------------------------------------------------------------------- #
def test_off_identity_block_stream_acc_fgt_and_params():
    R_ref = block_run(Prizma(d=8, h=24, K=4, n_experts=5, seed=0))
    R_off = block_run(Prizma(d=8, h=24, K=4, n_experts=5, seed=0, **OFF))
    assert np.array_equal(R_ref.R, R_off.R), "knobs-off changed the ACC matrix"
    assert R_ref.acc() == R_off.acc() and R_ref.forgetting() == R_off.forgetting()


def test_off_identity_round_robin_full_trajectory():
    """On the round-robin stream the bookkeeping paths (_note_novelty, the veto branch)
    execute; the run must STILL be bit-identical: params, mu/var, route_log, flags."""
    p_ref, p_off = make_rr(), make_rr(**OFF)
    rr_run(p_ref)
    rr_run(p_off)
    assert np.array_equal(param_vec(p_ref), param_vec(p_off)), \
        "knobs-off changed the parameter trajectory on the round-robin stream"
    assert economy(p_ref) == economy(p_off)


# ----------------------------------------------------------------------------- #
# 2. Every lever ON changes behaviour (sanity; on the collapse-regime stream)    #
# ----------------------------------------------------------------------------- #
def test_base_reproduces_immature_freeze_catch_all():
    """The stream under test exhibits the documented round-robin failure: the first
    expert freezes after only its warmup (128 samples) and catch-alls the stream."""
    p = make_rr()
    rr_run(p)
    frozen, committed, route = economy(p)
    assert frozen[0] == 1 and committed[0] == 1
    assert p.experts[0].n_seen == 128                 # froze at the warmup floor: immature
    assert route[0] > 0 and min(route) == 0           # catch-all + dead slots


def test_each_lever_on_changes_behaviour():
    p_ref = make_rr()
    rr_run(p_ref)
    ref_params, (ref_frozen, _, _) = param_vec(p_ref), economy(p_ref)

    # veto: blocks the immature freeze (frozen flag flips; weights may coincide)
    p_v = make_rr(freeze_min_seen=300)
    rr_run(p_v)
    (frozen_v, committed_v, _), params_v = economy(p_v), param_vec(p_v)
    assert frozen_v[0] == 0 and committed_v[0] == 1, "veto must block freeze but not commit"
    assert frozen_v != ref_frozen

    # dv / hot_young / sample_top / settle: decision or gradient changes -> params move
    for name, kw in [("dynamic_vigilance", dict(dynamic_vigilance=0.5)),
                     ("hot_young", dict(hot_young=1.0)),
                     ("route_stat", dict(route_stat="sample_top")),
                     ("n_settle_steps", dict(n_settle_steps=2))]:
        p_on = make_rr(**kw)
        rr_run(p_on)
        assert not np.array_equal(param_vec(p_on), ref_params), \
            f"lever {name} ON did not change any parameter -- silently inert"


def test_route_stat_rejects_unknown_statistic():
    with pytest.raises(ValueError):
        Prizma(d=8, h=8, K=2, n_experts=2, seed=0, route_stat="median")


# ----------------------------------------------------------------------------- #
# 3. Unit: the freeze veto actually blocks the freeze below the floor            #
# ----------------------------------------------------------------------------- #
def _force_transition(freeze_min_seen, batch=128):
    """Train ONE batch (n_seen=batch), force the active expert's floor to fail the next
    batch deterministically (tight mu/var), and let the phase transition fire."""
    p = Prizma(d=8, h=16, K=4, n_experts=3, seed=0, commit_after=batch,
               freeze_min_seen=freeze_min_seen)
    tasks = structured_permuted_tasks(n_tasks=2, n_samples=256, d=8, k_latent=4,
                                      n_classes=4, seed=5)
    Y0 = np.eye(4, dtype=np.float32)[tasks[0].ytr[:batch]]
    p.train_batch(tasks[0].Xtr[:batch], Y0, tasks[0].ytr[:batch])
    assert p.experts[0].n_seen == batch
    # force-fail recognition on the next batch: floor far below any real reconstruction
    p.experts[0].mu, p.experts[0].var = 1e-3, 1e-8
    Y1 = np.eye(4, dtype=np.float32)[tasks[1].ytr[:batch]]
    p.train_batch(tasks[1].Xtr[:batch], Y1, tasks[1].ytr[:batch])
    return p


def test_veto_blocks_freeze_below_floor_but_still_commits():
    p = _force_transition(freeze_min_seen=10_000)
    assert p.experts[0].committed is True, "veto must not block the commit/advance"
    assert p.experts[0].frozen is False, "freeze below the floor must be vetoed"
    assert p.active == 1 and p.route_log[1] > 0, "stream must advance to a fresh expert"


def test_veto_floor_is_inclusive_and_zero_is_shipped():
    p_eq = _force_transition(freeze_min_seen=128)     # n_seen == floor -> freeze allowed
    assert p_eq.experts[0].frozen is True
    p_off = _force_transition(freeze_min_seen=0)      # shipped behaviour
    assert p_off.experts[0].frozen is True and p_off.experts[0].omega > 0


# ----------------------------------------------------------------------------- #
# 4. Unit: hot_young multiplier math                                             #
# ----------------------------------------------------------------------------- #
def test_hot_young_multiplier_formula():
    p = Prizma(d=8, h=8, K=2, n_experts=2, seed=0, hot_young=1.0)
    assert p._hot_mult(0) == pytest.approx(2.0)
    assert p._hot_mult(200) == pytest.approx(1.0 + math.exp(-1.0))     # ~1.368 (frozen tau)
    assert p._hot_mult(2000) == pytest.approx(1.0 + math.exp(-10.0))
    assert HOT_YOUNG_TAU == 200.0
    p_off = Prizma(d=8, h=8, K=2, n_experts=2, seed=0)
    assert p_off._hot_mult(0) == 1.0                                   # exactly inert


def test_hot_young_scales_every_local_delta_step():
    """One batch on two identical experts (same seed): with hot_young=h the decoder,
    head AND encoder deltas are all h-multiplier copies of the shipped deltas."""
    tasks = structured_permuted_tasks(n_tasks=1, n_samples=256, d=8, k_latent=4,
                                      n_classes=4, seed=1)
    t = tasks[0]
    Xb, yb = t.Xtr[:128], t.ytr[:128]
    Yb = np.eye(4, dtype=np.float32)[yb]
    hot = 0.7
    ps = {}
    for name, kw in [("off", {}), ("on", dict(hot_young=hot))]:
        p = Prizma(d=8, h=16, K=4, n_experts=1, seed=0, **kw)
        e = p.experts[0]
        W0 = {k: np.array(getattr(e, k)) for k in ("Wdec", "bdec", "Wcls", "bcls",
                                                   "Wenc", "benc")}
        p._train_expert(e, Xb, Yb)
        ps[name] = (p, {k: np.array(getattr(e, k)) - W0[k] for k in W0})
    # n_seen is read PRE-update: a first batch (n_seen=0) takes the full 1+hot multiplier
    m = 1.0 + hot
    for k in ("Wdec", "bdec", "Wcls", "bcls", "Wenc", "benc"):
        d_off, d_on = ps["off"][1][k], ps["on"][1][k]
        # float32 accumulation: |d_on - m*d_off| is bounded by the rounding of two
        # float32 adds (~2 ulp of the weight magnitude), hence atol=1e-6
        assert np.allclose(d_on, m * d_off, rtol=1e-3, atol=1e-6), \
            f"{k} delta not scaled by the hot multiplier ({m:.4f})"
        assert not np.allclose(d_on, d_off, rtol=1e-3, atol=1e-9), f"{k} delta unchanged"


# ----------------------------------------------------------------------------- #
# 5. Unit: sample_top fires on a constructed 60/40 mixed batch where batch_mean  #
#    stays blind (the stationarity failure mode, constructed directly)           #
# ----------------------------------------------------------------------------- #
def _blind_expert(S):
    """Expert stub whose recon_error returns the crafted per-sample surprise vector S;
    floor mu=0.10, sigma=0.02 -> threshold 0.20 (the expert's real calibrated scale)."""
    e = Expert(4, 8, 2, seed=3)
    e.mu, e.var = 0.10, 4e-4
    S = np.asarray(S, np.float32)
    e.recon_error = lambda X, Y=None: S
    return e


def test_sample_top_recruits_on_60_40_batch_where_batch_mean_blind():
    # 60% novel samples (surprise 0.28 > thr 0.20) + 40% in-floor (0.04):
    # batch MEAN = 0.4*0.04 + 0.6*0.28 = 0.184 <= 0.20 -> batch_mean is blind;
    # novel FRACTION = 0.60 >= 0.5 -> sample_top must reject (recruit).
    S = [0.04] * 40 + [0.28] * 60
    X = np.zeros((100, 4), np.float32)
    e = _blind_expert(S)
    p_mean = Prizma(d=4, h=8, K=2, n_experts=2, seed=0)
    p_top = Prizma(d=4, h=8, K=2, n_experts=2, seed=0, route_stat="sample_top")
    assert float(np.mean(S)) <= e.mu + p_mean.z_novel * math.sqrt(e.var)
    assert p_mean._recognizes(e, X) is True, "batch_mean must stay blind on this batch"
    assert p_top._recognizes(e, X) is False, "sample_top must see the 60% novel majority"
    assert SAMPLE_NOVEL_FRACTION == 0.5


def test_sample_top_fires_recruit_end_to_end_and_batch_mean_does_not():
    """Full train_batch: after the warmup batch, a 60/40 mixed batch recruits a fresh
    expert under sample_top while the batch_mean twin absorbs it (no recruit)."""
    tasks = structured_permuted_tasks(n_tasks=2, n_samples=256, d=4, k_latent=2,
                                      n_classes=4, seed=5)
    batch = 128
    S = np.array([0.04] * 50 + [0.28] * 78, np.float32)      # 60.9% novel; mean 0.1927
    results = {}
    for name, kw in [("batch_mean", {}), ("sample_top", dict(route_stat="sample_top"))]:
        p = Prizma(d=4, h=8, K=4, n_experts=3, seed=0, commit_after=batch, **kw)
        Y0 = np.eye(4, dtype=np.float32)[tasks[0].ytr[:batch]]
        p.train_batch(tasks[0].Xtr[:batch], Y0, tasks[0].ytr[:batch])
        assert p.experts[0].n_seen == batch and p.active == 0
        p.experts[0].mu, p.experts[0].var = 0.10, 4e-4       # threshold 0.20
        p.experts[0].recon_error = lambda X, Y=None: S       # the 60/40 mixed batch
        Y1 = np.eye(4, dtype=np.float32)[tasks[1].ytr[:batch]]
        p.train_batch(tasks[1].Xtr[:batch], Y1, tasks[1].ytr[:batch])
        results[name] = (int(p.active), p.route_log.tolist(),
                         int(p.experts[0].committed))
    assert results["batch_mean"] == (0, [2 * batch, 0, 0], 0), \
        "batch_mean must absorb the mixed batch (the documented blindness)"
    assert results["sample_top"][0] == 1 and results["sample_top"][1][1] == batch, \
        "sample_top must recruit a fresh expert on the same batch"


# ----------------------------------------------------------------------------- #
# 6. Unit: dynamic-vigilance EMA + clamp follow the documented formula           #
# ----------------------------------------------------------------------------- #
def test_dynamic_vigilance_formula_and_clamp():
    p = Prizma(d=4, h=8, K=2, n_experts=2, seed=0, dynamic_vigilance=0.5)
    assert p._nov_fast == 0.0 and p._nov_slow == 0.0
    assert p._theta_scale() == pytest.approx(0.75)            # tightens at rate 0
    p._note_novelty(True)
    assert p._nov_fast == pytest.approx(DV_ALPHA_FAST)        # 0.2
    assert p._nov_slow == pytest.approx(DV_ALPHA_SLOW)        # 0.02
    assert p._theta_scale() == pytest.approx(1.0 + 0.5 * (0.2 - 0.5))   # 0.85
    for _ in range(200):                                      # sustained novelty regime
        p._note_novelty(True)
    assert p._nov_fast == pytest.approx(1.0)
    assert p._theta_scale() == pytest.approx(1.25)            # loosens, inside the clamp
    for _ in range(5000):                                     # slow release (hysteresis)
        p._note_novelty(False)
    assert p._nov_slow < 1e-10 and p._theta_scale() == pytest.approx(0.75)
    # clamp pins: dv=8 drives the unclamped factor to -3 and +5 -> [0.25, 4]
    p8 = Prizma(d=4, h=8, K=2, n_experts=2, seed=0, dynamic_vigilance=8.0)
    assert p8._theta_scale() == DV_THETA_LO
    for _ in range(200):
        p8._note_novelty(True)
    assert p8._theta_scale() == DV_THETA_HI
    # OFF: the bookkeeping is a no-op (never leaves the init state)
    p0 = Prizma(d=4, h=8, K=2, n_experts=2, seed=0)
    for _ in range(50):
        p0._note_novelty(True)
    assert p0._nov_fast == 0.0 and p0._nov_slow == 0.0 and p0._theta_scale() == 1.0
