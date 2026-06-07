"""
Correctness tests for seq/stats.py.

Key objectives:
  R1  t_sf is a REAL Student-t tail (not a normal approximation).
  R2  tost_equivalence uses correct 90% CI half-widths.
  R5  margin_superiority exists and works.
  R7  Identical-model canary: same-distribution arms must NOT be flagged as significant.
  R8  (in test_repro.py) set_seed seeds CUDA + python random.
"""
from __future__ import annotations
import math
import numpy as np
import pytest

# ------------------------------------------------------------------ helpers ----

def _make_arms(mean_a, mean_b, n=10, jitter=0.01):
    jit = np.array([jitter, -jitter] * (n // 2))
    a = list(np.full(n, mean_a) + jit)
    b = list(np.full(n, mean_b) + jit)
    return a, b


# ----------------------------------------------------------------- t_sf -------

def test_t_sf_anchor_df9():
    """t=1.833, df=9 -> one-sided p must be ~0.05 (NOT 0.033 from normal approx)."""
    from seq.stats import t_sf
    p = t_sf(1.833, 9)
    assert 0.045 < p < 0.055, f"Expected ~0.050, got {p:.5f}"


def test_anti_conservative_regression_df9():
    """REGRESSION: the old normal approximation returned p=0.0334 for t=1.833,df=9.
    Assert the corrected value is *not* that falsely-significant value."""
    from seq.stats import t_sf
    p = t_sf(1.833, 9)
    assert p > 0.040, (
        f"ANTI-CONSERVATIVE BUG PRESENT: p={p:.5f} (old normal approx ~0.033); "
        f"correct Student-t value is ~0.050"
    )


def test_t_sf_anchors_multi():
    """Table of known one-sided p values, tolerance 1e-3."""
    from seq.stats import t_sf
    anchors = [
        (1.833,  9, 0.0500),
        (2.262,  9, 0.0250),
        (2.015,  5, 0.0500),
        (1.753, 15, 0.0500),
        (1.697, 30, 0.0500),
        (0.0,    9, 0.5000),
    ]
    for t_val, df, expected in anchors:
        got = t_sf(t_val, df)
        assert abs(got - expected) < 1e-3, (
            f"t_sf({t_val}, {df}): expected {expected:.4f}, got {got:.5f}"
        )


def test_t_sf_vs_scipy():
    """Cross-validate against scipy.stats.t.sf across a grid of df/t values."""
    try:
        from scipy.stats import t as scipy_t
    except ImportError:
        pytest.skip("scipy not available")
    from seq.stats import t_sf
    for df in [5, 9, 15, 30]:
        for t_val in [0.5, 1.0, 1.833, 2.5]:
            ref = scipy_t.sf(t_val, df)
            got = t_sf(t_val, df)
            assert abs(got - ref) < 1e-4, (
                f"t_sf({t_val}, {df}): got {got:.6f}, scipy={ref:.6f}"
            )


def test_t_sf_symmetry():
    """P(T < -t) = P(T > t) by symmetry; t_sf(0,df)=0.5."""
    from seq.stats import t_sf
    for df in [5, 9, 15]:
        assert abs(t_sf(0.0, df) - 0.5) < 1e-10
        # negative t -> p > 0.5
        p_neg = t_sf(-1.833, df)
        p_pos = t_sf(1.833, df)
        assert abs(p_neg - (1.0 - p_pos)) < 1e-6


# ----------------------------------------------------------------- summarize --

def test_summarize_ci95_uses_t():
    """CI half-width must equal t_isf(0.025, n-1)*se (not z*se)."""
    from seq.stats import summarize, t_isf, _welch
    import math, numpy as np
    xs = [0.95, 0.97, 0.99, 0.96, 0.98, 0.94, 0.99, 0.97, 0.96, 0.98]
    s = summarize(xs)
    n = len(xs)
    sd = float(np.std(xs, ddof=1))
    se = sd / math.sqrt(n)
    h_expected = t_isf(0.025, n - 1) * se
    h_got = (s["ci95"][1] - s["ci95"][0]) / 2.0
    assert abs(h_got - h_expected) < 1e-9, f"CI half-width: expected {h_expected:.6f}, got {h_got:.6f}"


def test_summarize_basic():
    xs = [0.95, 0.97, 0.99, 0.96, 0.98, 0.94, 0.99, 0.97, 0.96, 0.98]
    from seq.stats import summarize
    s = summarize(xs, solve_thresh=0.9)
    assert s["n"] == 10
    assert abs(s["median"] - 0.97) < 1e-9
    assert s["solve_rate"] == 1.0
    assert s["ci95"][0] < s["mean"] < s["ci95"][1]


# ----------------------------------------------------------------- superiority_test ---

def test_superiority_detects_real_gap():
    """A genuine 0.10 gap over n=10 must be detected at p<0.05."""
    from seq.stats import superiority_test
    a, b = _make_arms(0.80, 0.70, n=10)
    res = superiority_test(a, b)
    assert res["p_value"] < 0.05 and res["significant"], (
        f"Expected significant, got p={res['p_value']:.4f}"
    )


def test_superiority_identical_model_canary():
    """R7 CANARY: two arms from the SAME distribution must NOT be flagged significant.
    This is the primary anti-conservative regression gate."""
    from seq.stats import superiority_test
    a, b = _make_arms(0.80, 0.80, n=10)   # identical means, identical jitter pattern
    res = superiority_test(a, b)
    assert not res["significant"] and res["p_value"] > 0.05, (
        f"CANARY FAILURE: identical-model arms were flagged significant p={res['p_value']:.4f}"
    )


def test_superiority_uses_t_not_normal():
    """Numerically: for t=1.833 df~9 the normal gives 0.033; t-dist gives ~0.050.
    superiority_test on a pair that produces t≈1.833 must return p≈0.050."""
    from seq.stats import superiority_test, t_sf, _welch
    import numpy as np
    # Construct a pair whose Welch t ≈ 1.833.
    # With n=10 equal-size groups and known SE, set mean diff to t*se.
    # Use deterministic jitter so SE is predictable.
    jit = np.array([0.01, -0.01] * 5)
    base_a = np.full(10, 0.80) + jit
    # sd for each arm = std(jit) = 0.01; se_welch = sqrt(2*0.01^2/10)
    import math
    sd = float(np.std(jit, ddof=1))    # ≈ 0.01026
    se_w = math.sqrt(2 * sd**2 / 10)
    target_diff = 1.833 * se_w
    base_b = np.full(10, 0.80 - target_diff) + jit
    a = list(base_a); b = list(base_b)
    res = superiority_test(a, b)
    # Should be ~0.050, NOT ~0.033
    assert res["p_value"] > 0.040, (
        f"Anti-conservative: p={res['p_value']:.4f} should be ~0.050"
    )


# ----------------------------------------------------------------- margin_superiority ---

def test_margin_superiority_gap_exceeds_margin():
    """a=candidate BPC, b=baseline BPC; 0.05 gap with margin=0.03 -> significant."""
    from seq.stats import margin_superiority
    a, b = _make_arms(0.80, 0.85, n=10)   # b-a = 0.05 > margin=0.03
    res = margin_superiority(a, b, margin=0.03)
    assert res["significant"], (
        f"Expected significant (gap 0.05 > margin 0.03), got p={res['p_value']:.4f}"
    )


def test_margin_superiority_gap_below_margin():
    """0.01 gap with margin=0.03 -> NOT significant."""
    from seq.stats import margin_superiority
    a, b = _make_arms(0.80, 0.81, n=10)   # b-a = 0.01 < margin=0.03
    res = margin_superiority(a, b, margin=0.03)
    assert not res["significant"], (
        f"Expected NOT significant (gap 0.01 < margin 0.03), got p={res['p_value']:.4f}"
    )


# ----------------------------------------------------------------- tost_equivalence ---

def test_tost_equivalence_within_margin():
    from seq.stats import tost_equivalence
    a, b = _make_arms(0.900, 0.901, n=10, jitter=0.002)
    res = tost_equivalence(a, b, margin=0.02)
    assert res["equivalent"], f"Expected equivalent, got {res}"


def test_tost_ci90_halfwidth():
    """CI half-width == t_isf(alpha, df) * se (not the bogus 0.84*t95 factor)."""
    from seq.stats import tost_equivalence, _welch, t_isf
    a, b = _make_arms(0.900, 0.901, n=10, jitter=0.002)
    res = tost_equivalence(a, b, margin=0.02, alpha=0.05)
    _, df, se = _welch(a, b)
    h_expected = t_isf(0.05, df) * se
    h_got = (res["ci90"][1] - res["delta"]) / 1.0   # half-width from upper bound
    assert abs(h_got - h_expected) < 1e-9, (
        f"CI90 half-width: expected {h_expected:.8f}, got {h_got:.8f}"
    )


def test_tost_returns_p_lower_p_upper():
    """tost must return p_lower and p_upper keys."""
    from seq.stats import tost_equivalence
    a, b = _make_arms(0.90, 0.90, n=10, jitter=0.001)
    res = tost_equivalence(a, b, margin=0.02)
    assert "p_lower" in res and "p_upper" in res


# ----------------------------------------------------------------- holm_correction ---

def test_holm_basic():
    """[0.01, 0.04, 0.03] at alpha=0.05 with Holm-Bonferroni."""
    from seq.stats import holm_correction
    pvals = [0.01, 0.04, 0.03]
    results = holm_correction(pvals, alpha=0.05)
    assert len(results) == 3
    # Holm order: sorted p = [0.01, 0.03, 0.04]
    # i=0 (k=3): adj = 3*0.01 = 0.03 < 0.05 -> reject
    # i=1 (k=2): adj = 2*0.03 = 0.06 >= 0.05 -> NOT reject (stop)
    # i=2 (k=1): adj = 1*0.04 = 0.04 — but Holm is sequential so also NOT reject
    rejects = [r["reject"] for r in results]
    assert rejects[0] == True, f"p=0.01 should be rejected in Holm (adj=0.03), got {results[0]}"
    assert rejects[1] == False, f"p=0.04 should NOT be rejected (adj=0.08 or stopped), got {results[1]}"
    assert rejects[2] == False, f"p=0.03 should NOT be rejected (stopped after p=0.01), got {results[2]}"


def test_holm_preserves_original_order():
    """Return list is in the ORIGINAL input order, not sorted."""
    from seq.stats import holm_correction
    pvals = [0.04, 0.01, 0.03]
    results = holm_correction(pvals, alpha=0.05)
    assert results[0]["p"] == 0.04
    assert results[1]["p"] == 0.01
    assert results[2]["p"] == 0.03


def test_holm_all_significant():
    from seq.stats import holm_correction
    results = holm_correction([0.001, 0.002, 0.003], alpha=0.05)
    assert all(r["reject"] for r in results)


def test_holm_none_significant():
    from seq.stats import holm_correction
    results = holm_correction([0.1, 0.2, 0.3], alpha=0.05)
    assert not any(r["reject"] for r in results)
