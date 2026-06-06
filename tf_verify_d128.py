"""Committee R2 rank-1 (the flip test): can the matched tiny Transformer learn MQAR D=128 if given
its BEST shot — a CONSTANT high LR (removing the cosine-decay confound) and a much longer budget,
plus a bigger-TF confirmation arm? If every arm stays dead-flat at chance with NO upward inflection,
the "matched tiny TF does not learn D=128 under any reasonable recipe" finding stands (honestly
framed). If any arm crosses the engagement floor (eval>=0.5), the strong claim is INVALIDATED and
rescoped. Mixed-D training, eval@fixed D=128, seed 0, log every 2000 to catch any knee. Resumable.

Run: python3.13 tf_verify_d128.py
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
OUT = os.path.join(RES, "tf_verify_d128.json")
V = 512


def make(mkey, T):
    cfgs = {"tiny": (64, 2, 2), "d128L2H4": (128, 2, 4)}
    d, L, H = cfgs[mkey]
    return Transformer(TFConfig(vocab=V, d_model=d, n_layers=L, n_heads=H, max_len=T + 8, rope=True))


# (name, model, train-cfg-kwargs, cap). Constant-LR arms set cosine=False -> max high-LR exposure.
ARMS = [
    ("tiny_constLR1e-3", "tiny", dict(lr=1e-3, warmup=2000, warmup_frac=0.0, cosine=False), 120000),
    ("tiny_constLR2e-3", "tiny", dict(lr=2e-3, warmup=2000, warmup_frac=0.0, cosine=False), 120000),
    ("big_d128L2H4",     "d128L2H4", dict(lr=1e-3, warmup=2000, warmup_frac=0.0, min_lr_frac=0.1), 80000),
]


def main():
    task = MixedMQAR(vocab=V, max_pairs=128, num_queries=128, gap=0, min_pairs=1)
    T = task.seq_len
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    print(f"device={DEV} {task.name} eval@D=128 seq={T}  (rank-1 flip test; seed 0)", flush=True)
    for name, mkey, ckw, cap in ARMS:
        if name in res and "best" in res[name]:
            print(f"  {name}: (cached best={res[name]['best']:.3f})", flush=True)
            continue
        m = make(mkey, T)
        p = param_count(m)
        cfg = TrainConfig(steps=cap, batch_size=64, log=True, eval_every=2000, **ckw)
        print(f"\n--- {name} ({mkey}, {p}p, {ckw}, cap={cap}) ---", flush=True)
        t0 = time.time()
        r = train_model(m, task, cfg, DEV, seed=0)
        hist = [(s, round(a, 4)) for (s, _l, a) in r.history]
        res[name] = {"best": r.best_acc, "plateau": r.steps_to_plateau, "params": p, "cap": cap,
                     "crossed_floor": bool(r.best_acc >= 0.5), "acc_curve": hist,
                     "sec": round(time.time() - t0, 1)}
        json.dump(res, open(OUT, "w"), indent=2)
        flip = "CROSSED 0.5 -> INVALIDATES strong claim" if r.best_acc >= 0.5 else "flat (<0.5) -> claim survives this arm"
        print(f">>> {name:<18} best@D128={r.best_acc:.3f} plateau@{r.steps_to_plateau} "
              f"({res[name]['sec']}s) -> {flip}", flush=True)
    print(f"\nsaved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
