"""Tune the Transformer to a STABLE, strong config on mixed-D MQAR @ target D (fairness: the
baseline must be given its own best optimizer hyperparameters, per committee two-stage tuning).
The punchy short-warmup schedule destabilizes the TF (loss oscillates 0.8<->4.5, eval flat at
chance) while PRISM is robust. This sweeps gentler (lr, warmup) configs for the TF; the one that
reaches the highest stable eval@target is the TF's fair config for the head-to-head.

Run: python3.13 tf_stabilize.py [target_pairs]   # default 64
"""
from __future__ import annotations

import json
import os
import sys
import time

from seq.common import TrainConfig, train_model, param_count, get_device
from seq.tasks import MixedMQAR
from seq.transformer import Transformer, TFConfig

DEV = get_device()
RES = os.path.join(os.path.dirname(__file__), "results")
OUT = os.path.join(RES, "tf_stabilize.json")

CONFIGS = {                              # gentler schedules to stabilize the fragile TF
    "gen-warm":  dict(lr=1e-3, warmup=2000, warmup_frac=0.0, min_lr_frac=0.1),
    "lowlr-gen": dict(lr=5e-4, warmup=1500, warmup_frac=0.0, min_lr_frac=0.1),
    "vlowlr":    dict(lr=3e-4, warmup=1000, warmup_frac=0.0, min_lr_frac=0.1),
}


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    V = max(64, 4 * target)
    task = MixedMQAR(vocab=V, max_pairs=target, num_queries=128, gap=0, min_pairs=1)
    T = task.seq_len
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    print(f"device={DEV} {task.name} eval@D={target} seq={T} (TF stabilization sweep)", flush=True)
    for cname, ckw in CONFIGS.items():
        if cname in res and "best" in res[cname]:
            print(f"  {cname}: (cached best={res[cname]['best']:.3f})", flush=True)
            continue
        m = Transformer(TFConfig(vocab=V, d_model=64, n_layers=2, n_heads=2, max_len=T + 8, rope=True))
        p = param_count(m)
        cfg = TrainConfig(steps=12000, batch_size=64, log=True, eval_every=2000, **ckw)
        t0 = time.time()
        r = train_model(m, task, cfg, DEV, seed=0)
        res[cname] = {"best": r.best_acc, "plateau": r.steps_to_plateau, "params": p,
                      "sec": round(time.time() - t0, 1), **{k: ckw[k] for k in ckw}}
        json.dump(res, open(OUT, "w"), indent=2)
        print(f">>> TF {cname:<10} best@D{target}={r.best_acc:.3f} plateau@{r.steps_to_plateau} "
              f"({res[cname]['sec']}s, {p}p) -> {'SOLVES' if r.best_acc > 0.9 else ('PARTIAL' if r.best_acc>0.5 else 'no')}",
              flush=True)
    print(f"\nsaved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
