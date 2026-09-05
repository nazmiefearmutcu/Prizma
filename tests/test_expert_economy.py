"""Tests for the expert-economy analysis tooling (Wave-C C3; report 10 P3).

Scope: these tests pin the BEHAVIOUR OF THE ANALYSIS SCRIPT
(experiments/expert_economy.py) on a tiny CPU stream (seconds) -- they test NO src/
behaviour and modify NO src file. Instrumentation is a thin InstrumentedPrizma
subclass living in the script; src/prizma.py stays untouched.

  (a) the recruit ledger records >= 1 recruit per trained domain;
  (b) the parameter-growth curve matches hand-counts from live Expert array shapes
      (trainable 516 / fixed-FA 288 / total 804 floats per expert at d=8, h=24, K=4);
  (c) the OFFLINE merge simulation finds the expected merge when two experts are
      trained on IDENTICAL inputs (constructed case; Jaccard=1, Welch-indistinguishable,
      post-merge surprise under each vigilance);
  (d) the eviction policy evicts the lowest-lifetime-share expert when M_max is below
      the true domain count, and is INERT (0 fires) when M_max == true domain count.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.data import structured_permuted_tasks
from src.prizma import Expert, Prizma
import expert_economy as ee


# ----------------------------------------------------------------------------- #
# Tiny shared run: 3 distinguishable domains + 1 never-trained probe, seconds    #
# ----------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tiny_run():
    # vigilance tightened to z_novel=3 so per-domain recruitment fires at tiny scale
    # (shipped runs keep z_novel=5.0; see the growth.json config for those)
    return ee.run_block_stream(seed=3, K=3, d=8, ncls=4, h=24, epochs=3,
                               probe_domains=1, z_novel=3.0)


# ----------------------------------------------------------------------------- #
# (a) recruit ledger                                                            #
# ----------------------------------------------------------------------------- #
def test_ledger_records_at_least_one_recruit_per_domain(tiny_run):
    run = tiny_run
    K = run["config"]["K"]
    assert len(run["per_domain"]) == K
    for i, dom in enumerate(run["per_domain"]):
        # by the end of domain i, at least i+1 experts must have been recruited
        assert dom["recruited_cum"] >= i + 1, f"domain {i}: {dom['recruited_cum']}"
    # the ledger's recruit_events and the final n_seen census must agree
    assert run["final"]["recruited"] == len(run["recruit_events"])
    # every freeze event targets an expert that was recruited first
    assert all(f["expert"] < run["final"]["recruited"] for f in run["freeze_events"])


# ----------------------------------------------------------------------------- #
# (b) parameter-growth curve == hand-count from Expert shapes                   #
# ----------------------------------------------------------------------------- #
def test_param_growth_matches_hand_count(tiny_run):
    run = tiny_run
    d, h, K = run["config"]["d"], run["config"]["h"], run["config"]["n_classes"]
    ref = Expert(d, h, K, seed=0)
    trainable, fixed_fa, total = ee.expert_float_counts(ref)
    # hand-count from the class definition (report 10, §1.3 arithmetic)
    assert trainable == h * d + h + d * h + d + K * h + K
    assert fixed_fa == h * d + h * K
    assert (trainable, fixed_fa, total) == (516, 288, 804)  # d=8, h=24, K=4

    for i, dom in enumerate(run["per_domain"]):
        assert dom["params_trainable_cum"] == dom["recruited_cum"] * trainable
        assert dom["params_with_fa_cum"] == dom["recruited_cum"] * total
    fin = run["final"]["params"]
    assert fin["trainable_per_expert"] == trainable
    assert fin["with_fa_recruited"] == run["final"]["recruited"] * total
    assert fin["with_fa_allocated_pool"] == run["config"]["n_experts"] * total


# ----------------------------------------------------------------------------- #
# (c) offline merge simulation on a CONSTRUCTED identical-input case            #
# ----------------------------------------------------------------------------- #
def test_merge_simulation_finds_constructed_merge():
    tasks = structured_permuted_tasks(n_tasks=1, n_samples=400, d=8, k_latent=4,
                                      n_classes=4, seed=0)
    t = tasks[0]
    p = Prizma(d=8, h=12, K=4, n_experts=4, seed=0)
    Yall = np.eye(4, dtype=np.float32)[t.ytr]
    # two experts, SAME seed, trained on the SAME batches with the repo's own local
    # training rule -> identical recognizers over identical inputs
    e_a, e_b = Expert(8, 12, 4, seed=7), Expert(8, 12, 4, seed=7)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(t.Xtr))
    for s in range(0, len(idx), 128):
        bi = idx[s:s + 128]
        for e in (e_a, e_b):
            p._train_expert(e, t.Xtr[bi], Yall[bi])
    e_a.frozen = e_b.frozen = True

    snaps = {0: ee.snapshot_expert(e_a), 1: ee.snapshot_expert(e_b)}
    routed = {0: [(0, i) for i in range(60)], 1: [(0, i) for i in range(60)]}
    res = ee.simulate_merges(snaps, routed, [(0, t.Xte)])

    assert len(res["pairs"]) == 1
    pair = res["pairs"][0]
    assert pair["jaccard"] == 1.0                      # identical routed sets
    assert pair["welch"]["p"] > 0.05                    # surprise indistinguishable
    assert abs(pair["welch"]["cohen_d"]) < 0.5
    assert pair["verdict"] == "MERGEABLE"
    assert res["n_mergeable"] == 1
    # the weight-averaged merged expert reproduces the shared surprise on the set
    m = res["merged_snaps"][(0, 1)]
    assert ee.snap_recon_mean(m, t.Xte[:60]) == pytest.approx(
        ee.snap_recon_mean(snaps[0], t.Xte[:60]), abs=1e-9)


def test_merge_simulation_stays_inert_on_distinguishable_stream(tiny_run):
    # on the input-distinguishable regime routing is pure: no pair may pass the
    # Jaccard gate (this is the measured home-regime result, pinned here)
    run = tiny_run
    assert all(p["verdict"] == "no_overlap_gate"
               for p in run["merge_sim"]["pairs"])
    assert run["merge_sim"]["n_mergeable"] == 0


# ----------------------------------------------------------------------------- #
# (d) eviction policy                                                           #
# ----------------------------------------------------------------------------- #
def test_evict_choice_prefers_lowest_share_most_recent_tiebreak():
    # lowest lifetime routing count wins outright
    assert ee.evict_choice([100, 20, 50], [0, 1, 2]) == 1
    # tie on count -> the MOST RECENTLY recruited expert is evicted
    assert ee.evict_choice([100, 20, 50, 20], [0, 1, 2, 3]) == 3
    # protected experts are never evictable
    assert ee.evict_choice([100, 20, 50], [0, 1, 2], protected={1}) == 2
    # no candidates -> None
    assert ee.evict_choice([10], [0], protected={0}) is None


def test_eviction_replay_fires_exactly_when_m_max_below_domains(tiny_run):
    run = tiny_run
    recruited = run["final"]["recruited"]
    K = run["config"]["K"]
    # cap AT the true domain count: the cap must be inert in the home regime
    at_k = ee.simulate_eviction(run["final"]["route_log"], recruited, m_max=K)
    assert at_k["n_fires"] == 0
    # cap one below: exactly recruited - m_max evictions, victims from the pool
    below = ee.simulate_eviction(run["final"]["route_log"], recruited, m_max=K - 1)
    assert below["n_fires"] == recruited - (K - 1)
    for fire in below["fires"]:
        assert 0 <= fire["victim"] < run["config"]["n_experts"]
        assert fire["victim"] == ee.evict_choice(run["final"]["route_log"],
                                                 [m for m in range(K - 1)])
    # and on THIS tiny distinguishable stream the unbounded run never needed a 4th
    # expert slot: all victims are zero-information ties, exactly as on the E-suite


def test_welch_test_sanity():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 200)
    # identical arrays -> t=0, p=1 regardless of seq.stats availability
    r = ee.welch_test(a, a)
    assert r["t"] == 0.0 and r["p"] == 1.0
    # separated arrays -> |t| large, |cohen_d| > 2
    b = rng.normal(10, 1, 200)
    r2 = ee.welch_test(a, b)
    assert abs(r2["t"]) > 50 and abs(r2["cohen_d"]) > 2
