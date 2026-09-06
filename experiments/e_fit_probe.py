"""LANE-EXPLORATORY — tiny-D crosstalk-law epsilon sanity probe (RALPH backlog item 6).

Purpose: BEFORE the GPU-tier registered grid (PR-2026-09-03-02), sanity-check the
capacity law's direction on CPU: find the coarse solve/no-solve transition D* in
{16, 32} (quad2, d_h=32, d_phi=256, n=2 seeds), compute the implied epsilon
eps_fit = sigma2 * sqrt(D*-1) from the probe's measured crosstalk, and compare against
report 11 / docs/crosstalk_capacity_law.md (N*(eps=1) = 48-132 depending on definition;
eps~0.9-1.2 claim). EXPLORATORY: n=2, coarse grid, CPU — motivates, never claims.

Run: python experiments/e_fit_probe.py
Output: results/exploratory/e_fit_probe_2026-09-07/probe.json (crash-safe per-run)
"""
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

import numpy as np
import torch

from seq.common import TrainConfig, build_and_train
from seq.tasks import MixedMQAR
from seq.prizma_seq import prizma_seq_factory
from feat_map_probe import crosstalk_metrics, _phi_quad2, _make_quad2_buffers  # pure functions (numpy)

OUT = os.path.join(_ROOT, "results", "exploratory", "e_fit_probe_2026-09-07")
D_GRID = [16, 32]
SEEDS = [0, 1]
STEPS = 8000
SOLVE = 0.90          # repo-standard solve threshold (MQAR accuracy)
D_H, N2 = 32, 224     # d_phi = 256, matching the committed probe artifact's geometry


def main():
    torch.set_num_threads(int(os.environ.get("PROBE_THREADS", "4")))
    os.makedirs(OUT, exist_ok=True)
    doc = {"meta": {
        "lane": "EXPLORATORY (never a claim)",
        "purpose": "direction sanity for the crosstalk capacity law before the GPU-tier fit",
        "grid": D_GRID, "seeds": SEEDS, "steps": STEPS, "solve_threshold": SOLVE,
        "model": "prizma_seq_factory(d_model=64, n_layers=2, n_heads=2, feat_map='quad2', feat_n2=224)",
        "crosstalk_geometry": {"d_h": D_H, "feat_n2": N2},
    }, "runs": []}
    path = os.path.join(OUT, "probe.json")
    json.dump(doc, open(path, "w", encoding="utf-8"), indent=1)

    rng = np.random.default_rng(7)
    K = rng.standard_normal((4096, D_H))
    K_phi = _phi_quad2(K, _make_quad2_buffers(D_H, N2))
    cm = crosstalk_metrics(K_phi)
    sigma2 = cm["sigma2"]
    doc["crosstalk"] = {k: (v if isinstance(v, (int, float)) else None) for k, v in cm.items()}
    json.dump(doc, open(path, "w", encoding="utf-8"), indent=1, default=str)
    print(f"[efit] crosstalk sigma2(|cos|)={sigma2:.5f} mu={cm['mu']:.5f} "
          f"N*(eps=1)={cm['n_star']:.1f}", flush=True)

    device = "cpu"
    for D in D_GRID:
        accs = []
        for seed in SEEDS:
            t0 = time.time()
            task = MixedMQAR(vocab=256, max_pairs=D, num_queries=64, min_pairs=1)
            fac = prizma_seq_factory(d_model=64, n_layers=2, n_heads=2,
                                     feat_map="quad2", feat_n2=N2)  # f(vocab, max_len) fed by build_and_train
            cfg = TrainConfig(steps=STEPS, min_steps=2000, batch_size=32, eval_every=500,
                              plateau_floor=0.5, log=False)
            r = build_and_train(fac, task, cfg, device, seed=seed,
                                vocab=task.vocab, max_len=task.seq_len)
            acc = float(r.final_acc)
            accs.append(acc)
            rec = {"D": D, "seed": seed, "acc": acc, "wall_s": round(time.time() - t0, 1)}
            doc["runs"].append(rec)
            json.dump(doc, open(path, "w", encoding="utf-8"), indent=1, default=str)
            print(f"[efit] D={D} seed={seed}: acc={acc:.3f} ({rec['wall_s']}s)", flush=True)
        solved = sum(a >= SOLVE for a in accs)
        doc.setdefault("summary", {})[str(D)] = {"solve_seeds": solved, "n": len(accs),
                                                 "mean_acc": float(np.mean(accs))}
        json.dump(doc, open(path, "w", encoding="utf-8"), indent=1, default=str)

    # transition + implied epsilon (coarse, n=2, exploratory)
    summary = doc["summary"]
    d_star = None
    for D in D_GRID:
        if summary[str(D)]["solve_seeds"] < len(SEEDS):
            d_star = D
            break
    if d_star is None:
        d_star = D_GRID[-1] + 1  # solves everywhere on the grid: transition is above it
    eps_implied = sigma2 * (np.sqrt(d_star - 1) if d_star > 1 else 0.0)
    doc["eps_fit"] = {
        "D_star_coarse": d_star,
        "eps_implied_abs": float(eps_implied),
        "note": ("eps = sigma2*sqrt(D*-1) implied by N*(eps)=min(1+eps^2/sigma2^2, d_phi) >= D* "
                 "at the solve/no-solve transition; |cos| definition. Compare vs the law doc's "
                 "N*(eps=1)=132.5 (|cos| def) and the eps~0.9-1.2 claim: if eps_implied is far "
                 "from 1, the law needs the fit epsilon that far from 1 — direction info only."),
    }
    json.dump(doc, open(path, "w", encoding="utf-8"), indent=1, default=str)
    print(f"[efit] D*_coarse={d_star} eps_implied={eps_implied:.3f} -> {path}", flush=True)


if __name__ == "__main__":
    main()
