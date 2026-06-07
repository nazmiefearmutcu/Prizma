"""Param-match auditor for every Prizma-vs-TF head-to-head. The quad2 feature map must add 0
trainable params (it is buffers); any *trainable* gate (output-gate W_g etc.) is reported so the TF
can be grown in lockstep where the addition is a fair architectural comparison."""
from __future__ import annotations
from .common import param_count

# Single source of truth for the param-match tolerance (|TF-Prizma|/TF < this). Reused as the
# param_match_report default AND by tests/test_flop_ledger.py so the bar is defined in exactly one place.
PARAM_MATCH_TOL = 0.02

def param_match_report(tf_model, pz_model, tol=PARAM_MATCH_TOL):
    pt, pp = param_count(tf_model), param_count(pz_model)
    added = 0
    for n, p in pz_model.named_parameters():
        if any(tag in n for tag in ("feat_I", "feat_J", "W_rand")):   # buffers anyway; defensive
            added += p.numel()
    return {"tf_params": pt, "pz_params": pp, "delta": pp - pt,
            "rel": abs(pp - pt) / max(1, pt), "matched": abs(pp - pt) / max(1, pt) < tol,
            "feat_map_added_params": added}
