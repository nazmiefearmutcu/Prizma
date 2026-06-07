"""Param-match auditor for every Prizma-vs-TF head-to-head. The quad2 feature map must add 0
trainable params (it is buffers); any *trainable* gate (output-gate W_g etc.) is reported so the TF
can be grown in lockstep where the addition is a fair architectural comparison."""
from __future__ import annotations
from .common import param_count

def param_match_report(tf_model, pz_model, tol=0.02):
    pt, pp = param_count(tf_model), param_count(pz_model)
    added = 0
    for n, p in pz_model.named_parameters():
        if any(tag in n for tag in ("feat_I", "feat_J", "W_rand")):   # buffers anyway; defensive
            added += p.numel()
    return {"tf_params": pt, "pz_params": pp, "delta": pp - pt,
            "rel": abs(pp - pt) / max(1, pt), "matched": abs(pp - pt) / max(1, pt) < tol,
            "feat_map_added_params": added}
