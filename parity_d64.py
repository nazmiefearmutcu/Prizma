"""Decisive D=64 head-to-head under the IDENTICAL protocol the Transformer won on (gen-warm:
lr=1e-3, warmup=2000; large cap + plateau-stop), mixed-D training, eval@D=64. TF already = 1.000.
Runs Prizma-quad2 (lever #1) then Prizma-none (ablation control) so the comparison and the causal
ablation share the exact same fair settings. quad2 ran punchy/12k before (0.906, still climbing);
this gives it the same generous budget as the TF.

Run: python3.13 parity_d64.py
"""
from __future__ import annotations

import json
import os
import time

from seq.common import TrainConfig, train_model, param_count, get_device
from seq.tasks import MixedMQAR
from seq.prizma_seq import PrizmaSeqLM, PrizmaSeqConfig

DEV = get_device()
RES = os.path.join(os.path.dirname(__file__), "results")
OUT = os.path.join(RES, "parity_d64.json")
TARGET, CAP = 64, 30000
V = max(64, 4 * TARGET)
GENWARM = dict(lr=1e-3, warmup=2000, warmup_frac=0.0, min_lr_frac=0.1)


def ps_quad(T):
    return PrizmaSeqLM(PrizmaSeqConfig(vocab=V, d_model=64, n_layers=2, n_heads=2, max_len=T + 8,
                                     feat_map="quad2", feat_n2=96))


def ps_none(T):
    return PrizmaSeqLM(PrizmaSeqConfig(vocab=V, d_model=64, n_layers=2, n_heads=2, max_len=T + 8))


def main():
    task = MixedMQAR(vocab=V, max_pairs=TARGET, num_queries=128, gap=0, min_pairs=1)
    T = task.seq_len
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res.setdefault("_baseline", {"Transformer_genwarm": 1.000})
    print(f"device={DEV} {task.name} eval@D={TARGET} seq={T} protocol=gen-warm cap={CAP} (TF=1.000)",
          flush=True)
    for name, make in [("Prizma-quad2", ps_quad), ("Prizma-none", ps_none)]:   # quad2 first (priority)
        if name in res and "best" in res[name]:
            print(f"  {name}: (cached best={res[name]['best']:.3f})", flush=True)
            continue
        m = make(T)
        p = param_count(m)
        cfg = TrainConfig(steps=CAP, batch_size=64, log=True, eval_every=3000, **GENWARM)
        t0 = time.time()
        r = train_model(m, task, cfg, DEV, seed=0)
        res[name] = {"best": r.best_acc, "plateau": r.steps_to_plateau, "params": p,
                     "sec": round(time.time() - t0, 1)}
        json.dump(res, open(OUT, "w"), indent=2)
        print(f">>> {name:<12} best@D{TARGET}={r.best_acc:.3f} plateau@{r.steps_to_plateau} "
              f"({res[name]['sec']}s, {p}p) vs TF=1.000", flush=True)
    print(f"\nsaved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
