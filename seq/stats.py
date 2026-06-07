"""Powered statistics for the head-to-head: real Student-t CIs (not normal approximation),
solve-rate, one-sided superiority (Welch t), TOST equivalence, margin superiority, and
Holm-Bonferroni correction.

Key fixes vs. v1:
  R1  _p_one_sided_from_t used normal tail (1-Phi(t)) which is anti-conservative at low df.
      Replaced with a pure-Python regularised incomplete-beta Student-t (no scipy dependency).
  R2  tost_equivalence used a bogus `crit*0.84` factor. Now uses proper t_isf(alpha, df)*se.
  R5  margin_superiority added.
  R7  Identical-model canary: superiority_test now returns correct p ~ 0.50 for same-mean arms.

No scipy dependency assumed (but if scipy is present, cross-validation tests will use it).
"""
from __future__ import annotations
import math
import numpy as np


# ============================================================== Beta / t core ==

def _betacf(a, b, x, itmax=300, eps=1e-14):
    """Lentz's modified continued fraction for the regularised incomplete beta function."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1e-30 if abs(d) < 1e-30 else d
        c = 1.0 + aa / c
        c = 1e-30 if abs(c) < 1e-30 else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1e-30 if abs(d) < 1e-30 else d
        c = 1.0 + aa / c
        c = 1e-30 if abs(c) < 1e-30 else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    """Regularised incomplete beta I_x(a,b) via Numerical Recipes continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    else:
        return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_sf(t, df):
    """One-sided upper-tail survival P(T > t) for Student-t with `df` degrees of freedom.

    Uses the regularised incomplete beta function (Numerical Recipes), so this is exact
    (to floating-point precision), not a normal approximation.

    Convention:
        t >= 0 -> P(T > t) in (0, 0.5]
        t <  0 -> P(T > t) in (0.5, 1)   (by symmetry: P(T>-t) = 1 - P(T>t))
    """
    x = df / (df + t * t)
    ib = _betai(df / 2.0, 0.5, x)   # = P(|T| > |t|) = two-tailed p
    return 0.5 * ib if t >= 0 else 1.0 - 0.5 * ib


def t_isf(p, df):
    """Inverse survival: find t such that P(T > t) = p, for p in (0, 0.5].

    Uses bisection (monotone decreasing in t), 200 iterations -> ~13 significant digits.
    """
    lo, hi = 0.0, 1.0e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_sf(mid, df) > p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ============================================================ Welch helper ====

def _welch(a, b):
    """Return (t, df, se) for Welch's two-sample t-test of (mean(a) - mean(b))."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    se = math.sqrt(va / na + vb / nb) or 1e-12
    t = float((a.mean() - b.mean()) / se)
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1) + 1e-30
    )
    return t, df, se


# ======================================================= Public statistics ====

def summarize(xs, solve_thresh=0.9):
    """Descriptive statistics with a REAL Student-t 95% CI (not z-based).

    Returns: n, mean, median, sd, ci95, min, max, solve_rate.
    """
    a = np.asarray(xs, float)
    n = len(a)
    mean = float(a.mean())
    sd = float(a.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    h = t_isf(0.025, n - 1) * se   # two-sided 95%: p=0.025 in each tail
    return {
        "n": n,
        "mean": mean,
        "median": float(np.median(a)),
        "sd": sd,
        "ci95": (mean - h, mean + h),
        "min": float(a.min()),
        "max": float(a.max()),
        "solve_rate": float((a >= solve_thresh).mean()),
    }


def superiority_test(a, b, alpha=0.05):
    """One-sided Welch t-test for H1: mean(a) > mean(b).

    Uses a real Student-t tail (t_sf), NOT a normal approximation.
    Returns: delta, t, df, p_value, significant.
    """
    t, df, se = _welch(a, b)
    p = t_sf(t, df)
    return {
        "delta": float(np.mean(a) - np.mean(b)),
        "t": t,
        "df": df,
        "p_value": p,
        "significant": p < alpha,
    }


def margin_superiority(a, b, margin, alpha=0.05):
    """One-sided test for H1: (mean(b) - mean(a)) > margin.

    Sign convention (for BPC / lower-is-better metrics):
        a = candidate BPC (lower is better)
        b = baseline BPC
        margin = minimum required advantage (e.g. 0.03)
    Significant when the candidate beats the baseline by AT LEAST `margin`.

    Equivalently tests H0: (mean(b)-mean(a)) <= margin against H1: > margin.
    t = (mean(b) - mean(a) - margin) / se

    Returns: delta (= mean(b)-mean(a)), t, df, p_value, significant, margin.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    _, df, se = _welch(a, b)   # se from _welch; recompute t for the margined hypothesis
    diff = float(b.mean() - a.mean())   # positive = b is larger (worse for BPC)
    t = (diff - margin) / se
    p = t_sf(t, df)
    return {
        "delta": diff,
        "t": t,
        "df": df,
        "p_value": p,
        "significant": p < alpha,
        "margin": margin,
    }


def tost_equivalence(a, b, margin, alpha=0.05):
    """Two one-sided t-tests (TOST) for equivalence of mean(a) and mean(b).

    Equivalent when the (1-2*alpha) CI of (mean(a)-mean(b)) lies within (-margin, +margin).

    The CI uses the correct one-sided critical value t_isf(alpha, df) (not the bogus
    0.84*t95 approximation from v1).

    Returns:
        delta   : mean(a) - mean(b)
        ci90    : (delta - h, delta + h)  where h = t_isf(alpha, df) * se
        margin  : as supplied
        p_lower : P(T > (diff + margin)/se) — tests H0: diff <= -margin
        p_upper : P(T > (margin - diff)/se) — tests H0: diff >= +margin
        equivalent: max(p_lower, p_upper) < alpha
    """
    t, df, se = _welch(a, b)
    diff = float(np.mean(a) - np.mean(b))
    h = t_isf(alpha, df) * se          # one-sided 95% critical value for 90% CI
    p_lower = t_sf((diff + margin) / se, df)
    p_upper = t_sf((margin - diff) / se, df)
    return {
        "delta": diff,
        "ci90": (diff - h, diff + h),
        "margin": margin,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "equivalent": max(p_lower, p_upper) < alpha,
    }


def holm_correction(pvals, alpha=0.05):
    """Holm-Bonferroni correction for multiple comparisons.

    Args:
        pvals : list of raw p-values (any order)
        alpha : family-wise error rate

    Returns:
        List of dicts {p, p_adj, reject} in the ORIGINAL input order.
        Rejection is sequential: once we encounter a non-rejection in ascending p order,
        all subsequent hypotheses are also not rejected.
    """
    n = len(pvals)
    # Tag with original indices and sort ascending by p
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    rejected = [False] * n
    p_adj = [0.0] * n
    stop = False
    for rank, (orig_idx, p) in enumerate(indexed):
        k = n - rank            # number of remaining tests (Holm step)
        adj = min(p * k, 1.0)
        p_adj[orig_idx] = adj
        if not stop and adj < alpha:
            rejected[orig_idx] = True
        else:
            stop = True         # once we fail to reject, all subsequent are also not rejected
    return [
        {"p": pvals[i], "p_adj": p_adj[i], "reject": rejected[i]}
        for i in range(n)
    ]


def solve_rate(xs, thresh=0.9):
    return float((np.asarray(xs, float) >= thresh).mean())
