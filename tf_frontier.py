"""Baseline-strength frontier (fairness precondition).

The Transformer is seed-BIMODAL on MQAR (stability_test: solves 2/5; seed 0 looks collapse-prone),
and a long warmup + cosine-to-floor over a big budget delays the phase transition until the LR has
decayed too far to sharpen it. Before ANY Prizma comparison we must establish: with a transition-
friendly schedule and MULTIPLE seeds, what is the Transformer's reliable solve frontier? A parity
claim is only meaningful where the Transformer is a *strong* baseline (high solve-rate).

This sweeps the matched tiny TF (d=64,L=2,H=2) over a few short-warmup recipes x seeds at the
target D, reporting per-recipe solve-rate(best>0.9) + median + best. If even the best recipe can't
reliably solve a given D at this scale, that D needs a bigger base model (grow BOTH, param-matched).

Run:  python3.13 tf_frontier.py            # default D=64, 3 recipes x 3 seeds, tiny TF
      python3.13 tf_frontier.py 32 64      # custom D list
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np  # noqa: F401
import torch  # noqa: F401

from seq.common import TrainConfig, train_model, param_count, get_device
from seq.tasks import MQAR
from seq.transformer import Transformer, TFConfig

DEV = get_device()
RES = os.path.join(os.path.dirname(__file__), "results")
OUT = os.path.join(RES, "tf_frontier.json")
SEEDS = (0, 1, 2)

# short-warmup, transition-friendly recipes (validate_disjoint solved with a punchy schedule)
RECIPES = {
    "punchy":   dict(lr=1e-3, warmup=200, warmup_frac=0.0, min_lr_frac=0.1),
    "hi-floor": dict(lr=1e-3, warmup=200, warmup_frac=0.0, min_lr_frac=0.3),  # keep LR high post-transition
    "lowlr":    dict(lr=5e-4, warmup=200, warmup_frac=0.0, min_lr_frac=0.1),
}


def cap_for(D):
    return 10000 if D <= 32 else (12000 if D <= 64 else 16000)


def main():
    Ds = [int(a) for a in sys.argv[1:]] or [64]
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    print(f"device={DEV} D={Ds} recipes={list(RECIPES)} seeds={SEEDS} (matched tiny TF d=64,L=2,H=2)",
          flush=True)
    for D in Ds:
        V = max(64, 4 * D)
        tkw = dict(vocab=V, num_pairs=D, num_queries=128, gap=0)
        seq = MQAR(**tkw).seq_len
        cap = cap_for(D)
        key = f"D{D}"
        res.setdefault(key, {})
        print(f"\n==== TF FRONTIER D={D} (V={V}, seq={seq}, cap={cap}) ====", flush=True)
        for rname, rkw in RECIPES.items():
            res[key].setdefault(rname, {})
            for s in SEEDS:
                sk = f"seed{s}"
                if sk in res[key][rname] and "best" in res[key][rname][sk]:
                    continue
                task = MQAR(**tkw)
                m = Transformer(TFConfig(vocab=task.vocab, d_model=64, n_layers=2, n_heads=2,
                                         max_len=task.seq_len + 8, rope=True))
                p = param_count(m)
                cfg = TrainConfig(steps=cap, batch_size=64, log=False, **rkw)
                t0 = time.time()
                r = train_model(m, task, cfg, DEV, seed=s)
                rec = {"best": r.best_acc, "plateau": r.steps_to_plateau, "sec": round(time.time() - t0, 1)}
                res[key][rname][sk] = rec
                json.dump(res, open(OUT, "w"), indent=2)
                print(f"  [D{D}] {rname:<8} seed{s}: best={rec['best']:.3f} plateau@{rec['plateau']} "
                      f"({rec['sec']}s, {p}p)", flush=True)
            bests = [res[key][rname][f"seed{s}"]["best"] for s in SEEDS if f"seed{s}" in res[key][rname]]
            if bests:
                sr = sum(b > 0.9 for b in bests)
                print(f"  => [D{D}] {rname:<8} solve-rate={sr}/{len(bests)} "
                      f"median={sorted(bests)[len(bests)//2]:.3f} best={max(bests):.3f}", flush=True)
    print("\n==== TF FRONTIER SUMMARY (solve-rate by recipe; is the tiny TF strong here?) ====", flush=True)
    for D in Ds:
        for rname in RECIPES:
            arm = res[f"D{D}"].get(rname, {})
            bests = [v["best"] for v in arm.values() if "best" in v]
            if bests:
                print(f"  D={D:<4} {rname:<8} solve {sum(b>0.9 for b in bests)}/{len(bests)} "
                      f"best={max(bests):.3f}", flush=True)
    print(f"\nsaved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
