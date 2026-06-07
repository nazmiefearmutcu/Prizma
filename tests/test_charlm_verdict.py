"""FAST, training-free tests for the powered char-LM v2 verdict helper.

These exercise `gpu_charlm2.charlm_v2_verdict(per_arm_bpcs, margins)` on SYNTHETIC
per-seed TEST-BPC arrays — no model is built and nothing is trained, so the suite is
millisecond-fast and pins ONLY the statistics + sign convention of the verdict.

The headline metric is bits-per-char (BPC): LOWER is better. The verdict must therefore
declare Prizma the winner when its per-seed BPC array sits clearly BELOW the Transformer's.

Cases:
  (a) Prizma clearly lower than TF by > 0.03  -> margin_superiority significant  -> BEATS
  (b) Prizma ~ TF (within the parity margin)   -> TOST equivalent               -> PARITY
  (c) Prizma higher (worse) than TF            -> NOT superiority-significant    -> WORSE
  (d) sign-convention guard: swapping which arm is lower flips BEATS<->WORSE.
  (e) the helper records per-arm mean/median and the vs-Hybrid Pareto check.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpu_charlm2 import charlm_v2_verdict

MARGINS = {"superiority": 0.03, "parity": 0.05}


# ---- tight, low-variance synthetic per-seed BPC arrays (n seeds each) ---------- #
def _arr(center, n=6, jitter=0.004):
    """Deterministic low-variance array centered at `center` (symmetric jitter)."""
    j = np.array([jitter, -jitter] * (n // 2 + 1))[:n]
    return list(np.full(n, float(center)) + j)


# ============================================================ (a) BEATS ========= #
def test_prizma_clearly_lower_beats_tf():
    """Prizma at 1.20 BPC vs TF at 1.30 BPC: a 0.10 advantage > 0.03 margin -> BEATS."""
    per_arm = {"Prizma-v2": _arr(1.20), "TF": _arr(1.30), "Hybrid": _arr(1.25)}
    v = charlm_v2_verdict(per_arm, MARGINS)
    assert v["verdict"] == "BEATS", v
    sup = v["superiority"]
    assert bool(sup["significant"]) is True, sup
    # delta = mean(TF) - mean(Prizma) must be POSITIVE (Prizma lower = better)
    assert sup["delta"] > 0.03, sup
    assert sup["p_value"] < 0.05, sup


# ============================================================ (b) PARITY ======== #
def test_prizma_equal_to_tf_is_parity():
    """Prizma and TF essentially equal -> not superior by 0.03, but TOST-equivalent -> PARITY."""
    per_arm = {"Prizma-v2": _arr(1.250), "TF": _arr(1.252), "Hybrid": _arr(1.251)}
    v = charlm_v2_verdict(per_arm, MARGINS)
    assert bool(v["superiority"]["significant"]) is False, v["superiority"]
    assert bool(v["parity"]["equivalent"]) is True, v["parity"]
    assert v["verdict"] == "PARITY", v


# ============================================================ (c) WORSE ========= #
def test_prizma_higher_is_worse():
    """Prizma at 1.40 vs TF at 1.20: Prizma is WORSE; not superior, not equivalent -> WORSE."""
    per_arm = {"Prizma-v2": _arr(1.40), "TF": _arr(1.20), "Hybrid": _arr(1.25)}
    v = charlm_v2_verdict(per_arm, MARGINS)
    assert bool(v["superiority"]["significant"]) is False, v["superiority"]
    # the candidate is worse, so the superiority delta must be NEGATIVE
    assert v["superiority"]["delta"] < 0, v["superiority"]
    assert bool(v["parity"]["equivalent"]) is False, v["parity"]
    assert v["verdict"] == "WORSE", v


# ============================================================ (d) sign guard ==== #
def test_sign_convention_lower_bpc_wins():
    """Swapping which arm is lower must flip the verdict — proves lower-BPC-wins, not the reverse."""
    low_prizma = {"Prizma-v2": _arr(1.10), "TF": _arr(1.30), "Hybrid": _arr(1.20)}
    low_tf = {"Prizma-v2": _arr(1.30), "TF": _arr(1.10), "Hybrid": _arr(1.20)}
    assert charlm_v2_verdict(low_prizma, MARGINS)["verdict"] == "BEATS"
    assert charlm_v2_verdict(low_tf, MARGINS)["verdict"] == "WORSE"


# ============================================================ (e) bookkeeping === #
def test_records_per_arm_stats_and_hybrid_check():
    """Verdict records per-arm mean/median and a Prizma-vs-Hybrid Pareto comparison."""
    per_arm = {"Prizma-v2": _arr(1.20), "TF": _arr(1.30), "Hybrid": _arr(1.40)}
    v = charlm_v2_verdict(per_arm, MARGINS)
    for arm in ("Prizma-v2", "TF", "Hybrid"):
        assert arm in v["per_arm"], v["per_arm"]
        assert "mean_bpc" in v["per_arm"][arm]
        assert "median_bpc" in v["per_arm"][arm]
        assert abs(v["per_arm"][arm]["mean_bpc"] - np.mean(per_arm[arm])) < 1e-6
    # vs Hybrid: Prizma (1.20) clearly beats Hybrid (1.40) by > 0.03
    assert bool(v["vs_hybrid_superiority"]["significant"]) is True, v["vs_hybrid_superiority"]
    assert v["vs_hybrid_superiority"]["delta"] > 0.03, v["vs_hybrid_superiority"]


def test_hybrid_optional():
    """The helper must not crash when no Hybrid arm is supplied (vs-hybrid keys are None)."""
    per_arm = {"Prizma-v2": _arr(1.20), "TF": _arr(1.30)}
    v = charlm_v2_verdict(per_arm, MARGINS)
    assert v["verdict"] == "BEATS", v
    assert v["vs_hybrid_superiority"] is None
    assert v["vs_hybrid_parity"] is None
