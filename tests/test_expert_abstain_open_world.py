"""Tests for the opt-in open-world abstain mechanism (docs/EXPERT_ECONOMY.md §3.4).

`Prizma.route_or_novel(X)` turns the documented-but-unimplemented NOVEL rule into
code: today's trained-expert argmin routing (route_for_inference, unchanged) plus a
BATCH-level abstain label from the per-sample minimum z-score over the trained
experts' own precision floors. The mechanism is opt-in via `abstain_z` (default
None); with None every existing path is untouched and the new method refuses.

Coverage:
  1. default-path safety -- abstain_z present/absent changes NOTHING observable on
     route_for_inference (bit-identical idx/S after identical training);
  2. calibration behavior -- mirroring the §2.3 measurement protocol at reduced
     scale on the structured-permuted generator: trained-domain batches yield
     novel=False at z=4 and a never-trained domain's batches yield novel=True
     (deterministic across seeds; margins reported in the lane report);
  3. guards -- abstain_z=None -> ValueError; no trained expert -> novel=True;
     side-effect freedom (repeated calls, no state mutation).
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import structured_permuted_tasks  # noqa: E402
from src.prizma import Prizma  # noqa: E402

Z_ABSTAIN = 4.0   # docs/EXPERT_ECONOMY.md §2.3 calibration point (batch, z=4)

# Reduced-scale mirror of the §2.3 protocol: same generator, same shipped knobs
# (h=48, z_novel=5.0, 15 epochs), fewer domains/samples so the suite stays fast.
CAL = dict(K=3, d=24, ncls=8, h=48, epochs=15, n_samples=3000, z_novel=5.0)


def _param_vec(p):
    parts = []
    for e in p.experts:
        parts.extend([e.Wenc.ravel(), e.benc, e.Wdec.ravel(), e.bdec,
                      e.Wcls.ravel(), e.bcls, np.array([e.mu, e.var])])
    return np.concatenate(parts)


def _fit_calibrated(seed, **overrides):
    """Shipped-protocol block stream on K domains + one never-trained probe domain."""
    cfg = {**CAL, **overrides}
    tasks = structured_permuted_tasks(n_tasks=cfg["K"] + 1, n_samples=cfg["n_samples"],
                                      d=cfg["d"], k_latent=8, n_classes=cfg["ncls"],
                                      seed=seed)
    p = Prizma(d=cfg["d"], h=cfg["h"], K=cfg["ncls"], n_experts=cfg["K"] + 3,
               seed=seed, z_novel=cfg["z_novel"], abstain_z=Z_ABSTAIN)
    rng = np.random.default_rng(seed)
    for t in tasks[:cfg["K"]]:
        p.fit_task(t.Xtr, t.ytr, epochs=cfg["epochs"], rng=rng)
    return p, tasks


def _full_batches(X, size=128):
    """Consecutive non-overlapping full batches (the §2.3 batch composition)."""
    n = (len(X) // size) * size
    return [X[s:s + size] for s in range(0, n, size)]


# ----------------------------------------------------------------------------- #
# 1. Default-path safety (NON-NEGOTIABLE): abstain_z changes nothing today      #
# ----------------------------------------------------------------------------- #
def test_abstain_knob_leaves_route_for_inference_bit_identical():
    tasks = structured_permuted_tasks(n_tasks=3, n_samples=768, d=8, k_latent=4,
                                      n_classes=4, seed=0)

    def build(abstain):
        kw = {} if abstain is None else {"abstain_z": abstain}
        p = Prizma(d=8, h=24, K=4, n_experts=5, seed=0, z_novel=3.0,
                   commit_after=128, **kw)
        rng = np.random.default_rng(0)
        for t in tasks[:2]:
            p.fit_task(t.Xtr, t.ytr, epochs=3, batch=128, rng=rng)
        return p

    p_ref, p_ab = build(None), build(Z_ABSTAIN)
    assert p_ref.abstain_z is None and p_ab.abstain_z == Z_ABSTAIN
    for t in tasks:
        i_ref, S_ref = p_ref.route_for_inference(t.Xte)
        i_ab, S_ab = p_ab.route_for_inference(t.Xte)
        assert np.array_equal(i_ref, i_ab), "abstain_z changed inference routing"
        assert np.array_equal(S_ref, S_ab), "abstain_z changed recon values"
    # training trajectory itself is untouched: identical weights/floors/flags/ledger
    assert np.array_equal(_param_vec(p_ref), _param_vec(p_ab))
    assert p_ref.state() == p_ab.state()


# ----------------------------------------------------------------------------- #
# 2. Calibration behavior (docs/EXPERT_ECONOMY.md §2.3): trained = known,       #
#    never-trained = NOVEL at z=4, with the huge margins the measurement found  #
# ----------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_calibrated_batches_separate_trained_from_never_trained(seed):
    p, tasks = _fit_calibrated(seed)
    # regime guard: the reduced-scale stream still produces one expert per domain
    assert sum(e.n_seen > 0 for e in p.experts) == CAL["K"]

    # trained domains: every full batch is claimed (novel=False) and the returned
    # idx/S are byte-equal to today's route_for_inference
    for t in tasks[:CAL["K"]]:
        for Xb in _full_batches(t.Xte):
            idx, S, novel = p.route_or_novel(Xb)
            assert novel is False, f"seed {seed}: trained batch flagged NOVEL"
            i_ref, S_ref = p.route_for_inference(Xb)
            assert np.array_equal(idx, i_ref) and np.array_equal(S, S_ref)

    # never-trained domain (same generator, new permutation): every batch is NOVEL
    probe = tasks[CAL["K"]]
    probe_batches = _full_batches(probe.Xte)
    assert probe_batches, "probe domain produced no full 128-batch"
    for Xb in probe_batches:
        _, _, novel = p.route_or_novel(Xb)
        assert novel is True, f"seed {seed}: never-trained batch not flagged NOVEL"


# ----------------------------------------------------------------------------- #
# 3. Guards: opt-in requirement, untrained pool, side-effect freedom            #
# ----------------------------------------------------------------------------- #
def test_route_or_novel_requires_opt_in_and_flags_untrained_pool_novel():
    X = np.random.default_rng(0).normal(0, 1.0, (16, 8)).astype(np.float32)

    p_off = Prizma(d=8, h=24, K=4, n_experts=5, seed=0)   # shipped default
    assert p_off.abstain_z is None
    with pytest.raises(ValueError, match="abstain_z"):
        p_off.route_or_novel(X)

    # no trained expert: nothing claims the batch -> novel=True, routing unchanged
    p_fresh = Prizma(d=8, h=24, K=4, n_experts=5, seed=0, abstain_z=Z_ABSTAIN)
    idx, S, novel = p_fresh.route_or_novel(X)
    assert novel is True
    i_ref, S_ref = p_fresh.route_for_inference(X)
    assert np.array_equal(idx, i_ref) and np.array_equal(S, S_ref)


def test_route_or_novel_is_side_effect_free():
    p, tasks = _fit_calibrated(0)
    X = tasks[0].Xte[:128]
    before_state = p.state()
    before_params = _param_vec(p)
    first = p.route_or_novel(X)
    second = p.route_or_novel(X)
    assert first[2] == second[2]
    assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])
    assert p.state() == before_state
    assert np.array_equal(_param_vec(p), before_params)


def test_non_finite_abstain_z_rejected():
    with pytest.raises(ValueError, match="abstain_z"):
        Prizma(d=8, h=24, K=4, n_experts=5, seed=0, abstain_z=float("nan"))
