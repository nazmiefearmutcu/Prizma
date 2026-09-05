"""Tests for the mean/fluctuation crosstalk metrics added to feat_map_probe.py
(Wave-B B2, committee report 11: E = mu*11^T + W split, candidate law
N* = min(1 + eps^2/sigma2^2, d_phi)).

Pure numpy: no torch, no GPU, deterministic (fixed seeds / hand-computed values).
Exact metric definitions (feat_map_probe module docstring):
  mu                  = mean(|c|) over the i<j upper triangle
  mu_signed           = mean(c)          (common mode of the split)
  sigma2              = std(|c|, ddof=1) over the i<j upper triangle
  sigma2_signed_fluct = std(c, ddof=1)   (fluctuation of W)
  eta                 = 1/(sigma2^2 * d_phi)              [inf when sigma2 == 0]
  n_star              = min(1 + 1/sigma2^2, d_phi)        [rank cap when sigma2 == 0]
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import feat_map_probe as fmp


# --------------------------------------------------------------- n_star law ----

def test_n_star_law_hand_values():
    """N* = min(1 + eps^2/sigma2^2, d_phi): SNR term vs rank cap, eps scaling."""
    assert fmp.n_star_law(0.1, 256) == pytest.approx(101.0)   # 1 + 1/0.01 = 101 < 256
    assert fmp.n_star_law(0.1, 50) == 50.0                    # rank cap binds
    assert fmp.n_star_law(0.1, 256, eps=2.0) == 256.0         # 1 + 4/0.01 = 401 -> capped
    assert fmp.n_star_law(0.5, 256) == pytest.approx(5.0)     # 1 + 1/0.25
    assert fmp.n_star_law(0.0, 137) == 137.0                  # degenerate: rank cap


# ------------------------------------------------- hand-computed small matrix ----

def _k3():  # 3 keys in R^2 with controlled overlaps
    s = 1.0 / math.sqrt(2.0)
    return np.array([[1.0, 0.0],
                     [0.0, 1.0],
                     [s, s]])


def test_metrics_controlled_overlap_hand_computed():
    """Keys k1=(1,0), k2=(0,1), k3=(1,1)/sqrt(2):
    |cos| upper triangle = [0, 1/sqrt2, 1/sqrt2] -> mu = 2/(3 sqrt2),
    sigma2 (ddof=1) = 1/sqrt6, eta = 1/(sigma2^2 * d_phi) = 3, N* = min(7, 2) = 2."""
    m = fmp.crosstalk_metrics(_k3())
    s = 1.0 / math.sqrt(2.0)
    assert m["mu"] == pytest.approx(2 * s / 3)
    assert m["mu_signed"] == pytest.approx(2 * s / 3)          # all cosines >= 0 here
    assert m["sigma2"] == pytest.approx(1.0 / math.sqrt(6.0))  # ddof=1 over [0, s, s]
    assert m["sigma2_signed_fluct"] == pytest.approx(1.0 / math.sqrt(6.0))
    assert m["eta"] == pytest.approx(3.0)                      # 1/((1/6)*2)
    assert m["eta_signed_fluct"] == pytest.approx(3.0)
    assert m["n_star"] == pytest.approx(2.0)                   # min(1+6, d_phi=2)
    assert m["n_star_signed_fluct"] == pytest.approx(2.0)
    assert m["offdiag_count"] == 3                             # 3 keys -> 3 upper-tri pairs


def test_metrics_one_negative_cosine_splits_abs_and_signed():
    """k1=(1,0), k2=(-1,0)/2... use k2=(-1,0): |cos|=[1] but signed=[-1] differs from mu."""
    K = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    m = fmp.crosstalk_metrics(K)
    assert m["mu"] == pytest.approx((1.0 + 0.0 + 0.0) / 3.0)
    assert m["mu_signed"] == pytest.approx((-1.0 + 0.0 + 0.0) / 3.0)
    # |cos| vals [1,0,0]: mean 1/3, ddof=1 var = ((2/3)^2 + 2*(1/3)^2)/2 = 1/3 -> std = 1/sqrt(3)
    assert m["sigma2"] == pytest.approx(1.0 / math.sqrt(3.0))
    # signed vals [-1,0,0]: mean -1/3, same spread -> ddof=1 std = 1/sqrt(3)
    assert m["sigma2_signed_fluct"] == pytest.approx(1.0 / math.sqrt(3.0))


# --------------------------------------------------------- boundary cases ----

def test_all_identical_keys_sigma2_zero_rank_cap_and_inf_eta():
    K = np.ones((4, 2))
    m = fmp.crosstalk_metrics(K)
    assert m["mu"] == pytest.approx(1.0)
    assert m["mu_signed"] == pytest.approx(1.0)
    # fp noise keeps sigma2 from being EXACTLY 0 here, but it is ~1e-16: eta explodes to ~1e31
    # and the law degenerates to the rank cap d_phi (the exact-0 branches are covered by
    # test_orthonormal_keys_sigma2_zero / test_single_offdiag_pair_ddof_guard).
    assert m["sigma2"] < 1e-12
    assert m["sigma2_signed_fluct"] < 1e-12
    assert m["eta"] > 1e12
    assert m["eta_signed_fluct"] > 1e12
    assert m["n_star"] == pytest.approx(2.0)        # degenerates to the rank cap d_phi
    assert m["n_star_signed_fluct"] == pytest.approx(2.0)


def test_orthonormal_keys_sigma2_zero():
    K = np.eye(3)
    m = fmp.crosstalk_metrics(K)
    assert m["mu"] == 0.0
    assert m["sigma2"] == 0.0
    assert math.isinf(m["eta"])
    assert m["n_star"] == pytest.approx(3.0)


def test_single_offdiag_pair_ddof_guard():
    """D=2 -> one upper-triangle sample; ddof=1 std is undefined and must fall back to 0.0."""
    K = np.array([[1.0, 0.0], [0.0, 1.0]])
    m = fmp.crosstalk_metrics(K)
    assert m["offdiag_count"] == 1
    assert m["sigma2"] == 0.0 and not math.isnan(m["sigma2"])
    assert m["n_star"] == pytest.approx(2.0)


def test_none_map_dphi_equals_dh():
    """'none' feature map: d_phi == d_h (no expansion) — the boundary case of the law."""
    r = fmp.probe_crosstalk(fmp._phi_none, None, d_h=32, n_trials=4, n_keys=64, seed=0)
    assert r["d_phi"] == 32
    assert r["mu"] > 0
    assert r["n_star"] == pytest.approx(min(1.0 + 1.0 / r["sigma2"] ** 2, 32.0))
    # identity map on random keys: signed fluctuation must sit at the 1/sqrt(d_phi) floor
    assert r["sigma2_signed_fluct"] == pytest.approx(1.0 / math.sqrt(32.0), rel=0.08)
    assert r["mu_signed"] == pytest.approx(0.0, abs=0.01)   # zero common mode by symmetry


# --------------------------------------------------- consistency with legacy ----

def test_mu_matches_legacy_key_crosstalk():
    rng = np.random.default_rng(7)
    K = rng.standard_normal((32, 16))
    m = fmp.crosstalk_metrics(K)
    assert m["mu"] == pytest.approx(fmp._key_crosstalk(K), abs=1e-12)


def test_sigma2_matches_manual_triu_formula():
    rng = np.random.default_rng(11)
    K = fmp._l2(rng.standard_normal((24, 10)))
    sim = K @ K.T
    iu = np.triu_indices(24, k=1)
    vals = np.abs(sim[iu])
    m = fmp.crosstalk_metrics(K)
    assert m["sigma2"] == pytest.approx(float(vals.std(ddof=1)))
    assert m["mu"] == pytest.approx(float(vals.mean()))
    assert m["offdiag_count"] == 24 * 23 // 2


def test_offdiag_upper_deterministic_order():
    sim = np.array([[1.0, 0.5, 0.2],
                    [0.5, 1.0, -0.7],
                    [0.2, -0.7, 1.0]])
    v = fmp.offdiag_upper(sim)
    assert v.tolist() == [0.5, 0.2, -0.7]   # (0,1),(0,2),(1,2): np.triu_indices order
    assert fmp.offdiag_upper(np.abs(sim)).tolist() == [0.5, 0.2, 0.7]


# ----------------------------------------------------- probe end-to-end (tiny) ----

def test_probe_crosstalk_tiny_new_fields_and_legacy_semantics():
    bufs = fmp._make_quad2_buffers(8, 12)
    r = fmp.probe_crosstalk(fmp._phi_quad2, bufs, d_h=8, n_trials=2, n_keys=16, seed=0)
    # legacy keys keep their semantics and ordering
    for k in ("mean", "std", "min", "max", "n_trials", "d_phi"):
        assert k in r
    assert r["n_trials"] == 2 and r["d_phi"] == 8 + 12
    assert r["min"] <= r["mean"] <= r["max"]
    # new law fields present and self-consistent
    for k in ("mu", "mu_signed", "sigma2", "sigma2_signed_fluct",
              "eta", "eta_signed_fluct", "n_star", "n_star_signed_fluct", "offdiag_count"):
        assert k in r
    assert r["offdiag_count"] == 2 * 16 * 15 // 2          # n_trials * n_keys*(n_keys-1)/2
    assert r["mu"] == pytest.approx(r["mean"], abs=1e-12)  # equal-count trials: pooled == mean-of-means
    assert 1.0 <= r["n_star"] <= r["d_phi"]
    assert r["eta"] == pytest.approx(1.0 / (r["sigma2"] ** 2 * r["d_phi"]))
    # pooled values derive from re-normalised keys, so |cos| <= 1 + fp slack
    assert r["mu"] <= 1.0 + 1e-9


def test_phi_maps_reproduce_legacy_buffers():
    """The seeded buffers must stay untouched: quad2 indices from seed 1234, lowrank pairs."""
    I, J = fmp._make_quad2_buffers(32, 224)
    assert I.shape == (224,) and J.shape == (224,)
    assert I.min() >= 0 and I.max() < 32 and J.max() < 32
    P, I_lr, J_lr = fmp._make_lowrank_buffers(32, 14)
    assert P.shape == (32, 14)
    assert I_lr.size == J_lr.size == 14 * 15 // 2
