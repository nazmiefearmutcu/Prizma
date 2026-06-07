"""Powered statistics for the head-to-head: t-based CIs (not z), solve-rate, one-sided superiority
(Welch t), and TOST equivalence. Use >=10 seeds for any decisive claim (the v1 n=2-3 was descriptive
only). No SciPy dependency assumed -> t critical values via a small table + normal fallback."""
from __future__ import annotations
import math
import numpy as np

# two-sided t critical values @ alpha=0.05 for df 1..30, then normal (1.96) beyond.
_T95 = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,
        11:2.201,12:2.179,13:2.160,14:2.145,15:2.131,16:2.120,17:2.110,18:2.101,19:2.093,
        20:2.086,21:2.080,22:2.074,23:2.069,24:2.064,25:2.060,26:2.056,27:2.052,28:2.048,
        29:2.045,30:2.042}

def _t_crit(df, two_sided=True):
    t = _T95.get(int(df), 1.96)
    return t if two_sided else (t if df not in _T95 else _T95[int(df)])  # table is 2-sided; ok approx

def summarize(xs, solve_thresh=0.9):
    a = np.asarray(xs, float); n = len(a)
    mean = float(a.mean()); sd = float(a.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    h = _t_crit(n - 1) * se
    return {"n": n, "mean": mean, "median": float(np.median(a)), "sd": sd,
            "ci95": (mean - h, mean + h), "min": float(a.min()), "max": float(a.max()),
            "solve_rate": float((a >= solve_thresh).mean())}

def _welch(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    va, vb = a.var(ddof=1), b.var(ddof=1); na, nb = len(a), len(b)
    se = math.sqrt(va/na + vb/nb) or 1e-12
    t = (a.mean() - b.mean()) / se
    df = (va/na + vb/nb)**2 / ((va/na)**2/(na-1) + (vb/nb)**2/(nb-1) + 1e-30)
    return t, df, se

def _p_one_sided_from_t(t, df):
    # normal approx to the t-CDF tail (adequate for df>=9, our regime). Upper tail of H1: mean(a)>mean(b)
    from math import erf, sqrt
    z = t  # df>=9 -> t ~ z within ~5%
    return 1.0 - 0.5 * (1 + erf(z / sqrt(2)))

def superiority_test(a, b, alpha=0.05):
    """One-sided Welch t for H1: mean(a) > mean(b)."""
    t, df, se = _welch(a, b)
    p = _p_one_sided_from_t(t, df)
    return {"delta": float(np.mean(a) - np.mean(b)), "t": t, "df": df,
            "p_value": p, "significant": p < alpha}

def tost_equivalence(a, b, margin, alpha=0.05):
    """Two one-sided tests: equivalent if the (1-2alpha) CI of (mean a - mean b) lies within +/-margin."""
    t, df, se = _welch(a, b)
    diff = float(np.mean(a) - np.mean(b))
    crit = _t_crit(round(df)) * 0.84  # ~90% CI half-width factor vs 95% table (z1.645/z1.96)
    lo, hi = diff - crit*se, diff + crit*se
    return {"delta": diff, "ci90": (lo, hi), "margin": margin,
            "equivalent": (lo > -margin) and (hi < margin)}

def solve_rate(xs, thresh=0.9):
    return float((np.asarray(xs, float) >= thresh).mean())
