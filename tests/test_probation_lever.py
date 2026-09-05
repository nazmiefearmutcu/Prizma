"""Tests for the G3b probationary-commit lever (terminal probe; PR-2026-09-03-07).

ONE Prizma-config knob, DEFAULT-OFF, attacking the replay probe's measured verdict
(results/exploratory/replay_probe_2026-09-05/): generative replay consolidates WITHOUT
new information; the named remaining mechanism class is REAL-data revisit. G3b: a
recruited expert stays PROBATIONARY -- it keeps training on every subsequent LIVE sample
it recognizes (per-sample routed, under train_granularity="sample") until the stream's
surprise floor says its domain passed (PROBATION_PATIENCE consecutive miss-batches);
only then commit/freeze. Buffer-free: it stores nothing and trains only on live samples
as they arrive (single-pass compatible by construction). At the recruit event the old
active is only DEMOTED (shipped machinery would commit+freeze it immediately).

Contract under test:

  OFF-identity:   defaults == explicit-off kwargs, BIT-IDENTICAL (parameter
                  trajectories, route_log, flags, ACC/FGT) on mixed and block streams,
                  for the shipped batch granularity AND the G1 per-sample granularity
                  (the config the lever composes with).
  ON-changes:     probation ON does not commit at the recruit event and the demoted
                  expert keeps training on later batches -> weights diverge from the
                  G1-only twin exactly there.
  White-box G3b:  a constructed probationary expert trains on >= 2 DISTINCT later
                  batches while never committed/frozen; its surprise-floor exit test
                  (consecutive miss-batches) freezes it exactly at PROBATION_PATIENCE;
                  the expert stays in the inference router while probationary.
  Gate ruling:    probation implies the STRONGER veto -- freeze needs BOTH
                  n_seen >= freeze_min_seen AND the exit test; neither gate overrides,
                  the LATER one binds (floor=0: exit test alone decides; large floor:
                  the expert stays probationary through elapsed patience and freezes
                  only after the floor is crossed AND a fresh miss-streak elapses).
  Settle probe:   n_settle_steps=4 takes EXACTLY 4 latent delta steps per routed
                  training call (counted via a _get_act probe inside settle), and the
                  deeper branch of _train_expert fires once per call.
  Knob hygiene:   probation requires train_granularity="sample" (batch granularity and
                  session_window>0 raise); non-bool raises; PROBATION_PATIENCE pinned.
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
from src.prizma import Prizma, PROBATION_PATIENCE

OFF = dict(probation=False)


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
    """The G1 base config probation composes with (d=2/h=4 scripted substrate)."""
    return Prizma(d=2, h=4, K=2, n_experts=4, seed=0, z_novel=5.0,
                  train_granularity="sample", **kw)


def make_batch(**kw):
    return Prizma(d=8, h=24, K=4, n_experts=5, seed=0, z_novel=3.0,
                  commit_after=128, **kw)


def make_gr(**kw):
    return Prizma(d=8, h=24, K=4, n_experts=5, seed=0, z_novel=3.0,
                  commit_after=128, train_granularity="sample", **kw)


def param_vec(p):
    parts = []
    for e in p.experts:
        parts.extend([e.Wenc.ravel(), e.benc, e.Wdec.ravel(), e.bdec,
                      e.Wcls.ravel(), e.bcls, np.array([e.mu, e.var])])
    return np.concatenate(parts)


def economy(p):
    return ([int(e.frozen) for e in p.experts],
            [int(e.committed) for e in p.experts], p.route_log.tolist())


THR = 0.10 + 5.0 * 0.02            # crafted floor: mu=0.10, sigma=0.02, z=5 -> 0.20


def stub_floor(p, m, per_sample_surprise):
    """Expert m: calibrated floor (mu=0.10, sigma=0.02 -> threshold 0.20 at z=5) and a
    row-KEYED stub recon_error (scalar -> constant surprise for every row)."""
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


def probation_setup():
    """Warm the active (all-novel batch 0 -> active takes it whole, no commit), craft
    its floor (threshold 0.20; rows of batch 0 in-floor at 0.04, rows of batches 1+ at
    0.28), then feed batch 1 = half recognized / half novel -> the recruit event fires
    and (probation) DEMOTES the old active instead of committing it."""
    p = make(probation=True)
    X0 = np.array([[0.0, 0.0], [0.0, 1.0]], np.float32)
    Y0 = np.eye(2, dtype=np.float32)[[0, 1]]
    p.train_batch(X0, Y0, np.array([0, 1]))
    assert p.route_log.tolist() == [2, 0, 0, 0] and p.experts[0].n_seen == 2
    assert not p.experts[0].committed
    stub_floor(p, 0, {(0.0, 0.0): 0.04, (0.0, 1.0): 0.04,
                      (1.0, 0.0): 0.04, (1.0, 1.0): 0.28})
    X1 = np.array([[1.0, 0.0], [1.0, 1.0]], np.float32)
    Y1 = np.eye(2, dtype=np.float32)[[1, 0]]
    p.train_batch(X1, Y1, np.array([1, 0]))
    # (1,0) recognized -> e0 trains on it; (1,1) novel (fraction 0.5) -> recruit:
    # e0 is DEMOTED (not committed, not frozen), the new active e1 takes the novel pool.
    assert p.route_log.tolist() == [3, 1, 0, 0]
    assert p.active == 1 and p.experts[1].n_seen == 1
    assert not p.experts[0].committed and not p.experts[0].frozen
    assert p.experts[0].prob_since_hit == 0, "demotion must restart the patience counter"
    return p


# ----------------------------------------------------------------------------- #
# 1. OFF-identity: bit-identical vs defaults (NON-NEGOTIABLE)                    #
# ----------------------------------------------------------------------------- #
def test_off_identity_shipped_batch_granularity_mixed_stream():
    p_ref, p_off = make_batch(), make_batch(**OFF)
    mixed_run(p_ref)
    mixed_run(p_off)
    assert np.array_equal(param_vec(p_ref), param_vec(p_off)), \
        "explicit-off kwargs changed the parameter trajectory on the mixed stream"
    assert economy(p_ref) == economy(p_off)


def test_off_identity_g1_granularity_mixed_stream_full_trajectory():
    p_ref, p_off = make_gr(), make_gr(**OFF)
    mixed_run(p_ref)
    mixed_run(p_off)
    assert np.array_equal(param_vec(p_ref), param_vec(p_off)), \
        "explicit-off kwargs changed the parameter trajectory (G1 mixed stream)"
    assert economy(p_ref) == economy(p_off)


def test_off_identity_g1_granularity_block_stream_acc_fgt():
    R_ref = block_run(make_gr())
    R_off = block_run(make_gr(**OFF))
    assert np.array_equal(R_ref.R, R_off.R), "explicit-off kwargs changed the ACC matrix"
    assert R_ref.acc() == R_off.acc() and R_ref.forgetting() == R_off.forgetting()


# ----------------------------------------------------------------------------- #
# 2. ON changes behaviour exactly at the recruit event                           #
# ----------------------------------------------------------------------------- #
def test_probation_defers_commit_at_recruit_and_keeps_training():
    """G1-only twin: the recruit event commits+freezes e0, which then only CLAIMS its
    samples. Probation twin: e0 stays probationary and KEEPS TRAINING on later live
    batches it recognizes -> the weights diverge exactly there."""
    # G1-only reference
    p_ref = make()
    X0 = np.array([[0.0, 0.0], [0.0, 1.0]], np.float32)
    Y0 = np.eye(2, dtype=np.float32)[[0, 1]]
    p_ref.train_batch(X0, Y0, np.array([0, 1]))
    stub_floor(p_ref, 0, {(0.0, 0.0): 0.04, (0.0, 1.0): 0.04,
                          (1.0, 0.0): 0.04, (1.0, 1.0): 0.28})
    X1 = np.array([[1.0, 0.0], [1.0, 1.0]], np.float32)
    Y1 = np.eye(2, dtype=np.float32)[[1, 0]]
    p_ref.train_batch(X1, Y1, np.array([1, 0]))
    assert p_ref.experts[0].committed and p_ref.experts[0].frozen  # shipped commit

    p_on = probation_setup()
    stub_floor(p_ref, 1, 0.04)     # identical active stub for the next batch
    stub_floor(p_on, 1, 0.04)
    Y2 = np.eye(2, dtype=np.float32)[[1, 0]]
    p_ref.train_batch(X1, Y2, np.array([1, 0]))     # committed e0 only CLAIMS (1,0)
    p_on.train_batch(X1, Y2, np.array([1, 0]))      # probationary e0 TRAINS on (1,0)
    assert p_ref.route_log.tolist() == [4, 2, 0, 0]  # claim counts for e0
    assert p_on.route_log.tolist() == [4, 2, 0, 0]   # same ledger, different training
    assert p_ref.experts[0].n_seen == 3 and p_on.experts[0].n_seen == 4, \
        "probationary expert must re-train where the committed twin only claims"
    assert not np.array_equal(p_ref.experts[0].Wdec, p_on.experts[0].Wdec), \
        "probation ON did not change any weight -- silently inert"
    assert not p_on.experts[0].committed and not p_on.experts[0].frozen


# ----------------------------------------------------------------------------- #
# 3. White-box G3b: >= 2 distinct later batches before any freeze can commit;    #
#    surprise-floor exit test freezes exactly at PROBATION_PATIENCE              #
# ----------------------------------------------------------------------------- #
def test_probationary_expert_trains_on_two_distinct_later_batches():
    p = probation_setup()
    e0 = p.experts[0]
    stub_floor(p, 1, 0.04)                       # active recognizes everything left
    Y2 = np.eye(2, dtype=np.float32)[[1, 0]]
    ns_after_recruit = e0.n_seen                 # 3 (warm 2 + recruit-batch take 1)
    # later batch #2: e0 recognized row -> trains
    p.train_batch(np.array([[1.0, 0.0], [1.0, 1.0]], np.float32), Y2, np.array([1, 0]))
    assert e0.n_seen == ns_after_recruit + 1 and not e0.committed and not e0.frozen
    # later batch #3: both rows recognized -> trains (2 samples)
    p.train_batch(np.array([[1.0, 0.0], [1.0, 0.0]], np.float32), Y2, np.array([1, 1]))
    assert e0.n_seen == ns_after_recruit + 3
    assert p.route_log.tolist() == [6, 2, 0, 0]
    assert not e0.committed and not e0.frozen and p.active == 1, \
        "no freeze may commit while the probationary expert keeps hitting"


def test_probationary_expert_stays_in_the_inference_router():
    p = probation_setup()
    idx, _ = p.route_for_inference(np.array([[1.0, 0.0]], np.float32))
    assert idx[0] == 0, "a probationary (trained) expert must remain routable"


def test_exit_test_freezes_exactly_at_probation_patience():
    p = probation_setup()
    e0 = p.experts[0]
    stub_floor(p, 1, 0.04)
    stub_floor(p, 0, 0.28)       # e0's domain passed: recognizes nothing anymore
    Xmiss = np.array([[1.0, 0.0], [1.0, 1.0]], np.float32)
    Ymiss = np.eye(2, dtype=np.float32)[[1, 0]]
    ns = e0.n_seen
    for k in range(1, PROBATION_PATIENCE):
        p.train_batch(Xmiss, Ymiss, np.array([1, 0]))
        assert not e0.frozen, f"freeze fired early at miss-batch {k}"
        assert e0.prob_since_hit == k
    p.train_batch(Xmiss, Ymiss, np.array([1, 0]))     # the PATIENCE-th miss batch
    assert e0.committed and e0.frozen, "exit test must commit+freeze at PATIENCE"
    assert e0.n_seen == ns, "miss batches must not train the expert"
    assert e0.prob_since_hit == 0, "counter resets after the freeze"


def test_hit_resets_the_exit_counter():
    p = probation_setup()
    e0 = p.experts[0]
    stub_floor(p, 1, 0.04)
    stub_floor(p, 0, 0.28)       # e0's domain stops matching: miss streak
    Xmiss = np.array([[1.0, 0.0], [1.0, 1.0]], np.float32)
    Ymiss = np.eye(2, dtype=np.float32)[[1, 0]]
    for _ in range(PROBATION_PATIENCE - 2):
        p.train_batch(Xmiss, Ymiss, np.array([1, 0]))
    assert e0.prob_since_hit == PROBATION_PATIENCE - 2 and not e0.frozen
    stub_floor(p, 0, 0.04)                    # its domain returns -> e0 trains
    p.train_batch(Xmiss, Ymiss, np.array([1, 0]))
    assert e0.prob_since_hit == 0, "a hit batch must reset the exit counter"
    assert e0.n_seen == 5, "the returning-domain batch must train the expert"


# ----------------------------------------------------------------------------- #
# 4. Gate composition: probation implies the STRONGER veto (BOTH gates; the      #
#    LATER one binds)                                                            #
# ----------------------------------------------------------------------------- #
def test_freeze_min_seen_gate_holds_the_expert_probationary():
    """Patience elapsed but n_seen < freeze_min_seen: NO commit, NO freeze -- the
    expert stays probationary and can resume training (the stronger veto)."""
    p = make(probation=True, freeze_min_seen=10_000)
    X0 = np.array([[0.0, 0.0], [0.0, 1.0]], np.float32)
    Y0 = np.eye(2, dtype=np.float32)[[0, 1]]
    p.train_batch(X0, Y0, np.array([0, 1]))
    stub_floor(p, 0, {(0.0, 0.0): 0.04, (0.0, 1.0): 0.04,
                      (1.0, 0.0): 0.04, (1.0, 1.0): 0.28})
    X1 = np.array([[1.0, 0.0], [1.0, 1.0]], np.float32)
    p.train_batch(X1, np.eye(2, dtype=np.float32)[[1, 0]], np.array([1, 0]))
    stub_floor(p, 1, 0.04)
    stub_floor(p, 0, 0.28)
    Xmiss = np.array([[1.0, 0.0], [1.0, 1.0]], np.float32)
    Ymiss = np.eye(2, dtype=np.float32)[[1, 0]]
    for k in range(PROBATION_PATIENCE + 3):
        p.train_batch(Xmiss, Ymiss, np.array([1, 0]))
        assert not p.experts[0].frozen and not p.experts[0].committed
    assert p.experts[0].prob_since_hit >= PROBATION_PATIENCE, \
        "the exit test fired but must not commit while the n_seen floor is unmet"


def test_freeze_fires_at_the_later_of_the_two_gates():
    """n_seen crosses the floor AFTER patience elapsed once, then a FRESH miss-streak
    elapses -> freeze fires only then: the later gate binds, neither overrides."""
    p = make(probation=True, freeze_min_seen=100)
    X0 = np.array([[0.0, 0.0], [0.0, 1.0]], np.float32)
    p.train_batch(X0, np.eye(2, dtype=np.float32)[[0, 1]], np.array([0, 1]))
    stub_floor(p, 0, {(0.0, 0.0): 0.04, (0.0, 1.0): 0.04,
                      (1.0, 0.0): 0.04, (1.0, 1.0): 0.28})
    X1 = np.array([[1.0, 0.0], [1.0, 1.0]], np.float32)
    p.train_batch(X1, np.eye(2, dtype=np.float32)[[1, 0]], np.array([1, 0]))
    stub_floor(p, 1, 0.04)
    stub_floor(p, 0, 0.28)                    # miss phase 1: patience elapses unused
    Xmiss = np.array([[1.0, 0.0], [1.0, 1.0]], np.float32)
    Ymiss = np.eye(2, dtype=np.float32)[[1, 0]]
    for _ in range(PROBATION_PATIENCE + 2):
        p.train_batch(Xmiss, Ymiss, np.array([1, 0]))
    e0 = p.experts[0]
    assert e0.prob_since_hit >= PROBATION_PATIENCE and not e0.frozen
    stub_floor(p, 0, 0.04)                    # domain returns: hits cross the floor
    Xhit = np.array([[1.0, 0.0], [1.0, 0.0]], np.float32)
    for _ in range(60):
        p.train_batch(Xhit, Ymiss, np.array([1, 1]))
        assert not e0.frozen, "hit batches must train, reset patience, and not freeze"
        assert e0.prob_since_hit == 0
    assert e0.n_seen >= 100, "the n_seen floor must have been crossed by real training"
    stub_floor(p, 0, 0.28)                    # miss phase 2: the fresh exit streak
    for k in range(1, PROBATION_PATIENCE):
        p.train_batch(Xmiss, Ymiss, np.array([1, 0]))
        assert not e0.frozen
    p.train_batch(Xmiss, Ymiss, np.array([1, 0]))
    assert e0.committed and e0.frozen, "freeze must fire at the LATER gate's deadline"


def test_consolidate_off_commits_without_freezing_at_the_exit_test():
    p = make(probation=True, consolidate=False)
    X0 = np.array([[0.0, 0.0], [0.0, 1.0]], np.float32)
    p.train_batch(X0, np.eye(2, dtype=np.float32)[[0, 1]], np.array([0, 1]))
    stub_floor(p, 0, {(0.0, 0.0): 0.04, (0.0, 1.0): 0.04,
                      (1.0, 0.0): 0.04, (1.0, 1.0): 0.28})
    X1 = np.array([[1.0, 0.0], [1.0, 1.0]], np.float32)
    p.train_batch(X1, np.eye(2, dtype=np.float32)[[1, 0]], np.array([1, 0]))
    stub_floor(p, 1, 0.04)
    stub_floor(p, 0, 0.28)
    Xmiss = np.array([[1.0, 0.0], [1.0, 1.0]], np.float32)
    Ymiss = np.eye(2, dtype=np.float32)[[1, 0]]
    for _ in range(PROBATION_PATIENCE):
        p.train_batch(Xmiss, Ymiss, np.array([1, 0]))
    assert p.experts[0].committed and not p.experts[0].frozen, \
        "consolidate=False: the exit test commits without freezing (shipped parity)"


def test_exit_test_composes_with_replay_before_freeze():
    """The probation freeze is the shipped freeze procedure: G3a replay runs inside it
    (weights move on generated items; the real-sample ledger does not)."""
    p = make(probation=True, replay_passes=2, replay_items=8)
    X0 = np.array([[0.0, 0.0], [0.0, 1.0]], np.float32)
    p.train_batch(X0, np.eye(2, dtype=np.float32)[[0, 1]], np.array([0, 1]))
    stub_floor(p, 0, {(0.0, 0.0): 0.04, (0.0, 1.0): 0.04,
                      (1.0, 0.0): 0.04, (1.0, 1.0): 0.28})
    X1 = np.array([[1.0, 0.0], [1.0, 1.0]], np.float32)
    p.train_batch(X1, np.eye(2, dtype=np.float32)[[1, 0]], np.array([1, 0]))
    stub_floor(p, 1, 0.04)
    stub_floor(p, 0, 0.28)
    Xmiss = np.array([[1.0, 0.0], [1.0, 1.0]], np.float32)
    Ymiss = np.eye(2, dtype=np.float32)[[1, 0]]
    ns = p.experts[0].n_seen
    for _ in range(PROBATION_PATIENCE):
        p.train_batch(Xmiss, Ymiss, np.array([1, 0]))
    assert p.experts[0].frozen
    assert len(p.replay_events) == 1 and p.replay_events[0]["expert"] == 0
    assert p.replay_events[0]["n_seen_real"] == ns, "real ledger stays honest"


# ----------------------------------------------------------------------------- #
# 5. Settle-depth probe: n_settle_steps=4 == exactly 4 latent delta steps        #
# ----------------------------------------------------------------------------- #
def _count_settle_steps(n_settle_steps, **kw):
    """One routed single-sample training call; count the latent-gradient steps inside
    EVERY settle invocation via a _get_act probe (settle calls _get_act once for
    Z_prior + once per delta step). One bookkeeping training call settles three times
    (init_recon, forward, post-update floor) -- each must take exactly `depth` steps."""
    p = Prizma(d=2, h=4, K=2, n_experts=2, seed=0, z_novel=5.0,
               n_settle_steps=n_settle_steps, eta_settle=0.1, train_granularity="sample",
               **kw)
    e = p.experts[0]
    act_calls = []
    orig_act = e._get_act

    def spy_act(A, rng=None):
        act_calls.append(1)
        return orig_act(A, rng)

    e._get_act = spy_act
    settle_steps = []
    orig_settle = e.settle

    def spy_settle(X, Y=None, rng=None):
        before = len(act_calls)
        out = orig_settle(X, Y, rng)
        settle_steps.append(len(act_calls) - before - 1)   # minus the Z_prior call
        return out

    e.settle = spy_settle
    X = np.array([[0.3, -0.2]], np.float32)
    Y = np.eye(2, dtype=np.float32)[[0]]
    p.train_batch(X, Y, np.array([0]))       # one sample routed to the active
    assert settle_steps and set(settle_steps) == {n_settle_steps}, \
        f"every settle must take exactly {n_settle_steps} delta steps, got {settle_steps}"
    return settle_steps


@pytest.mark.parametrize("depth", [0, 2, 4, 8])
def test_settle_depth_runs_exactly_depth_delta_steps(depth):
    steps = _count_settle_steps(depth)
    assert all(s == depth for s in steps) and len(steps) == 3, \
        f"n_settle_steps={depth}: the 3 settles of a training call (init_recon, " \
        f"forward, post-update floor) must each run exactly {depth} steps, got {steps}"


def test_settle4_counts_four_steps_per_routed_sample_in_g1_stream():
    """End-to-end on a G1 mixed stream with n_settle_steps=4: EVERY settle invocation
    (routing recon, forward, floor bookkeeping -- i.e. every per-exposure processing
    pass) runs exactly 4 latent delta steps."""
    p = Prizma(d=8, h=24, K=4, n_experts=5, seed=0, z_novel=3.0, commit_after=128,
               train_granularity="sample", n_settle_steps=4)
    seen = []
    for e in p.experts:
        orig_settle = e.settle
        act_calls = []
        orig_act = e._get_act

        def spy_act(A, rng=None, _o=orig_act, _c=act_calls):
            _c.append(1)
            return _o(A, rng)

        def spy_settle(X, Y=None, rng=None, _o=orig_settle, _c=act_calls, _s=seen):
            before = len(_c)
            out = _o(X, Y, rng)
            _s.append(len(_c) - before - 1)
            return out

        e._get_act = spy_act
        e.settle = spy_settle
    mixed_run(p, epochs=1)
    assert seen, "the stream must produce settle invocations"
    assert set(seen) == {4}, f"every settle must be exactly 4 steps, got {set(seen)}"


# ----------------------------------------------------------------------------- #
# 6. Knob hygiene                                                                #
# ----------------------------------------------------------------------------- #
def test_probation_requires_sample_granularity():
    with pytest.raises(ValueError):
        Prizma(d=4, h=4, K=2, n_experts=2, seed=0, probation=True)
    with pytest.raises(ValueError):
        Prizma(d=4, h=4, K=2, n_experts=2, seed=0, probation=True, session_window=8)


def test_probation_rejects_non_bool():
    with pytest.raises(ValueError):
        Prizma(d=4, h=4, K=2, n_experts=2, seed=0, train_granularity="sample",
               probation=1)


def test_probation_patience_constant_is_pinned():
    assert PROBATION_PATIENCE == 8


def test_probation_is_inert_until_the_first_recruit_event():
    """Reachability contract on a real stream: with NO recruit event, probation is
    bit-identical to the G1 twin (the lever acts only at/after recruit); divergence
    requires the recruit path. The tiny mixed stream never recruits (verified:
    recruited=1, committed=0), so both configs must match exactly there."""
    p_ref, p_on = make_gr(), make_gr(probation=True)
    acc_ref = mixed_run(p_ref)
    acc_on = mixed_run(p_on)
    assert sum(e.n_seen > 0 for e in p_ref.experts) == 1, \
        "precondition: the tiny stream must not recruit"
    assert np.array_equal(param_vec(p_ref), param_vec(p_on)), \
        "probation ON must be inert on a stream that never recruits"
    assert 0.0 <= acc_ref <= 1.0 and 0.0 <= acc_on <= 1.0
