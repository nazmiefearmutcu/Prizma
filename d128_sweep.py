"""The locked target: MQAR D=128 parity, under the SAME fair protocol that gave D=64 parity
(mixed-D training, gen-warm lr=1e-3/warmup=2000, large cap + plateau-stop, eval@fixed D=128).

Runs (order = most informative first; resumable via incremental JSON):
  1. Transformer        -> the strong baseline (does it solve D=128?); needs the largest budget
  2. Prizma-quad2-256    -> lever #1 at key-rank 256 (d_phi=256, +0 params, O(1)); the hero arm
  3. Prizma-none         -> ablation/capacity-ceiling control (D*~32, expected to fail D=128 clearly)

This is the run where the quad2 lever's capacity benefit should become decisive (none's fixed-state
ceiling bites hard at D=128). Seed 0 = directional; the multi-seed run_bar matrix follows.

Run: python3.13 d128_sweep.py
"""
from __future__ import annotations

import json
import os
import time

from seq.common import TrainConfig, train_model, param_count, get_device
from seq.tasks import MixedMQAR
from seq.transformer import Transformer, TFConfig
from seq.prizma_seq import PrizmaSeqLM, PrizmaSeqConfig

DEV = get_device()
RES = os.path.join(os.path.dirname(__file__), "results")
OUT = os.path.join(RES, "d128_sweep.json")
TARGET = 128
V = max(64, 4 * TARGET)            # 512
GENWARM = dict(lr=1e-3, warmup=2000, warmup_frac=0.0, min_lr_frac=0.1)


def tf(T):
    return Transformer(TFConfig(vocab=V, d_model=64, n_layers=2, n_heads=2, max_len=T + 8, rope=True))


def ps_quad256(T):
    return PrizmaSeqLM(PrizmaSeqConfig(vocab=V, d_model=64, n_layers=2, n_heads=2, max_len=T + 8,
                                     feat_map="quad2", feat_n2=224))   # d_phi = 256


def ps_none(T):
    return PrizmaSeqLM(PrizmaSeqConfig(vocab=V, d_model=64, n_layers=2, n_heads=2, max_len=T + 8))


PLAN = [("Transformer", tf, 60000), ("Prizma-quad2-256", ps_quad256, 40000), ("Prizma-none", ps_none, 40000)]


def main():
    task = MixedMQAR(vocab=V, max_pairs=TARGET, num_queries=128, gap=0, min_pairs=1)
    T = task.seq_len
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    print(f"device={DEV} {task.name} eval@D={TARGET} seq={T} protocol=gen-warm", flush=True)
    for name, make, cap in PLAN:
        if name in res and "best" in res[name]:
            print(f"  {name}: (cached best={res[name]['best']:.3f})", flush=True)
            continue
        m = make(T)
        p = param_count(m)
        cfg = TrainConfig(steps=cap, batch_size=64, log=True, eval_every=3000, **GENWARM)
        t0 = time.time()
        r = train_model(m, task, cfg, DEV, seed=0)
        res[name] = {"best": r.best_acc, "plateau": r.steps_to_plateau, "params": p, "cap": cap,
                     "sec": round(time.time() - t0, 1)}
        json.dump(res, open(OUT, "w"), indent=2)
        print(f">>> {name:<16} best@D{TARGET}={r.best_acc:.3f} plateau@{r.steps_to_plateau} "
              f"({res[name]['sec']}s, {p}p)", flush=True)
    print(f"\nsaved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
