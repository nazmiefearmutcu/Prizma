"""Base-scale frontier: find the SMALLEST Transformer that reliably solves high-D MQAR, so the
D=128-parity comparison happens where the baseline is genuinely strong (fairness precondition).

The matched tiny TF (d=64,L=2) fails MQAR D=64 at chance across 3 recipes x 3 seeds (tf_frontier) ->
D=64/128 are out of range for that size. This scans bigger TF configs (width/depth/head-dim) at the
target D to locate the base scale where the TF solves it; that scale then anchors the param-matched
PRISM-none vs PRISM-quad2 comparison.

Run:  python3.13 scale_frontier.py 64          # default configs at D=64
      python3.13 scale_frontier.py 64 128
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch  # noqa: F401

from seq.common import TrainConfig, train_model, param_count, get_device
from seq.tasks import MQAR
from seq.transformer import Transformer, TFConfig

DEV = get_device()
RES = os.path.join(os.path.dirname(__file__), "results")
OUT = os.path.join(RES, "scale_frontier.json")
SEEDS = (0,)                     # 1 seed for the directional scale scan; multi-seed once a scale wins
PUNCHY = dict(lr=1e-3, warmup=200, warmup_frac=0.0, min_lr_frac=0.1)

CONFIGS = {                      # (d_model, n_layers, n_heads) -> d_h
    "d128L2H4": dict(d_model=128, n_layers=2, n_heads=4),   # wider, d_h=32
    "d128L4H4": dict(d_model=128, n_layers=4, n_heads=4),   # wider + deeper, d_h=32
}


def cap_for(D):
    return 10000 if D <= 32 else (12000 if D <= 64 else 16000)


def main():
    Ds = [int(a) for a in sys.argv[1:]] or [64]
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    print(f"device={DEV} D={Ds} configs={list(CONFIGS)} seeds={SEEDS} recipe=punchy", flush=True)
    for D in Ds:
        V = max(64, 4 * D)
        tkw = dict(vocab=V, num_pairs=D, num_queries=128, gap=0)
        seq = MQAR(**tkw).seq_len
        cap = cap_for(D)
        key = f"D{D}"
        res.setdefault(key, {})
        print(f"\n==== SCALE FRONTIER D={D} (V={V}, seq={seq}, cap={cap}) ====", flush=True)
        for cname, ckw in CONFIGS.items():
            res[key].setdefault(cname, {})
            for s in SEEDS:
                sk = f"seed{s}"
                if sk in res[key][cname] and "best" in res[key][cname][sk]:
                    continue
                task = MQAR(**tkw)
                m = Transformer(TFConfig(vocab=task.vocab, max_len=task.seq_len + 8, rope=True, **ckw))
                p = param_count(m)
                cfg = TrainConfig(steps=cap, batch_size=64, log=False, **PUNCHY)
                t0 = time.time()
                r = train_model(m, task, cfg, DEV, seed=s)
                rec = {"best": r.best_acc, "plateau": r.steps_to_plateau, "params": p,
                       "sec": round(time.time() - t0, 1)}
                res[key][cname][sk] = rec
                json.dump(res, open(OUT, "w"), indent=2)
                print(f"  [D{D}] {cname:<10} seed{s}: best={rec['best']:.3f} plateau@{rec['plateau']} "
                      f"({rec['sec']}s, {p}p)  -> {'SOLVES' if rec['best'] > 0.9 else 'no'}", flush=True)
    print(f"\nsaved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
