"""Honesty/robustness check: confirm MQAR D=128 IS learnable by ATTENTION at sufficient scale, so
the matched-tiny-TF failure (0.016) is correctly framed as an under-capacity-at-small-params result,
not "attention can't do recall" (false) or a task bug. Bigger TF configs, mixed-D, gen-warm.

Run L2 (cheaper) first; if width alone solves D=128, that's the answer. Resumable.
Run: python3.13 tf_d128_scale.py
"""
from __future__ import annotations

import json
import os
import time

from seq.common import TrainConfig, train_model, param_count, get_device
from seq.tasks import MixedMQAR
from seq.transformer import Transformer, TFConfig

DEV = get_device()
RES = os.path.join(os.path.dirname(__file__), "results")
OUT = os.path.join(RES, "tf_d128_scale.json")
TARGET, V = 128, 512
GENWARM = dict(lr=1e-3, warmup=2000, warmup_frac=0.0, min_lr_frac=0.1)
CONFIGS = {                                  # bigger attention models (cheapest first)
    "d128L2H4": dict(d_model=128, n_layers=2, n_heads=4),
    "d128L4H4": dict(d_model=128, n_layers=4, n_heads=4),
}
CAP = 60000


def main():
    task = MixedMQAR(vocab=V, max_pairs=TARGET, num_queries=128, gap=0, min_pairs=1)
    T = task.seq_len
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    print(f"device={DEV} {task.name} eval@D={TARGET} seq={T} gen-warm cap={CAP}", flush=True)
    for cname, ckw in CONFIGS.items():
        if cname in res and "best" in res[cname]:
            print(f"  {cname}: (cached best={res[cname]['best']:.3f})", flush=True)
            continue
        m = Transformer(TFConfig(vocab=V, max_len=T + 8, rope=True, **ckw))
        p = param_count(m)
        cfg = TrainConfig(steps=CAP, batch_size=64, log=True, eval_every=3000, **GENWARM)
        t0 = time.time()
        r = train_model(m, task, cfg, DEV, seed=0)
        res[cname] = {"best": r.best_acc, "plateau": r.steps_to_plateau, "params": p,
                      "sec": round(time.time() - t0, 1)}
        json.dump(res, open(OUT, "w"), indent=2)
        print(f">>> TF {cname:<10} best@D{TARGET}={r.best_acc:.3f} plateau@{r.steps_to_plateau} "
              f"({res[cname]['sec']}s, {p}p) -> {'SOLVES' if r.best_acc > 0.9 else 'no'}", flush=True)
    print(f"\nsaved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
