"""Decisive capacity validation (committee R1 first_build).

Question: does the parameter-free quadratic feature map (feat_map='quad2', d_phi=128) lift MQAR
associative recall at HIGH D toward the Transformer, while matched PRISM-Seq (d_h=32) hits its
pre-registered ~D*=32 ceiling? Runs the decisive rungs under the FIXED fair protocol (frozen eval
+ per-model plateau early-stop from common.py, best of a small shared LR set). Also doubles as the
fairness check: does the Transformer now reach near-ceiling when trained to its own plateau?

1 seed = directional signal; the full run_bar matrix does >=5 seeds + CIs. Incremental + resumable.

Run:  python3.13 cap_probe.py             # default decisive D in {64,128}
      python3.13 cap_probe.py 16 32 64 128
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
from seq.prism_seq import PRISMSeqLM, PRISMSeqConfig

DEV = get_device()
RES = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RES, exist_ok=True)
OUT = os.path.join(RES, "cap_probe.json")

LRS = (1e-3, 2e-3)
SEED = 0


def tf(V, T):
    return Transformer(TFConfig(vocab=V, d_model=64, n_layers=2, n_heads=2, max_len=T + 8, rope=True))


def ps(V, T):
    return PRISMSeqLM(PRISMSeqConfig(vocab=V, d_model=64, n_layers=2, n_heads=2, max_len=T + 8))


def ps_quad(V, T):
    return PRISMSeqLM(PRISMSeqConfig(vocab=V, d_model=64, n_layers=2, n_heads=2, max_len=T + 8,
                                     feat_map="quad2", feat_n2=96))   # d_phi = 32 + 96 = 128


MODELS = {"Transformer": tf, "PRISM-none": ps, "PRISM-quad2": ps_quad}


def cap_for(D):
    return 12000 if D <= 32 else (16000 if D <= 64 else 20000)


def main():
    Ds = [int(a) for a in sys.argv[1:]] or [64, 128]
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    print(f"device={DEV} D={Ds} LRs={LRS} seed={SEED} (plateau early-stop; frozen eval)", flush=True)
    for D in Ds:
        V = max(64, 4 * D)
        tkw = dict(vocab=V, num_pairs=D, num_queries=128, gap=0)
        seq = MQAR(**tkw).seq_len
        cap = cap_for(D)
        key = f"D{D}"
        res.setdefault(key, {})
        print(f"\n==== CAP D={D} (V={V}, seq={seq}, cap={cap}) ====", flush=True)
        for name, make in MODELS.items():
            res[key].setdefault(name, {})
            for lr in LRS:
                lk = f"{lr:.0e}"
                if lk in res[key][name] and "best" in res[key][name][lk]:
                    print(f"  [D{D}] {name:<12} lr={lk}: (cached best={res[key][name][lk]['best']:.3f})",
                          flush=True)
                    continue
                task = MQAR(**tkw)
                model = make(task.vocab, task.seq_len)
                p = param_count(model)
                cfg = TrainConfig(steps=cap, batch_size=64, lr=lr, log=False)
                t0 = time.time()
                r = train_model(model, task, cfg, DEV, seed=SEED)
                rec = {"best": r.best_acc, "final": r.final_acc, "steps_to_plateau": r.steps_to_plateau,
                       "params": p, "sec": round(time.time() - t0, 1)}
                res[key][name][lk] = rec
                json.dump(res, open(OUT, "w"), indent=2)    # incremental + resumable
                print(f"  [D{D}] {name:<12} lr={lk}: best={rec['best']:.3f} final={rec['final']:.3f} "
                      f"plateau@{rec['steps_to_plateau']} ({rec['sec']}s, {p}p)", flush=True)
    print("\n==== CAP SUMMARY (best over LRs; the central contrast) ====", flush=True)
    for D in Ds:
        cells = []
        for name in MODELS:
            arm = res[f"D{D}"][name]
            b = max((v.get("best", -1) for v in arm.values()), default=-1)
            cells.append(f"{name}={b:.3f}")
        print(f"  D={D:<4} " + "   ".join(cells), flush=True)
    print(f"\nsaved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
