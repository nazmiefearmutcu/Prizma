"""De-risk calibration (pre-matrix).

Goal: find the step budget at which the *Transformer* reaches its OWN plateau on the hard MQAR
rungs (rung2, rung3), and confirm each model's best LR, so the fair matrix trains BOTH models to
convergence. This protects the bar's non-strawman rule (#6): a fair comparison requires the
Transformer to be trained to its plateau, not cut off mid-climb by too small a step budget.

Single seed, no early stop (so we see the *whole* accuracy-vs-step curve). Writes
results/calib_budget.json incrementally after every run, and prints a flushed, readable log.

Run:  python3.13 calib_budget.py [rung2|rung3|both]   (default: both, rung2 first)
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np  # noqa: F401  (kept for parity with the rest of the harness)
import torch

from seq.common import TrainConfig, train_model, param_count, get_device
from seq.tasks import MQAR
from seq.transformer import Transformer, TFConfig
from seq.prism_seq import PRISMSeqLM, PRISMSeqConfig

DEV = get_device()
RES = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RES, exist_ok=True)
OUT = os.path.join(RES, "calib_budget.json")


def make_tf(V, T):
    return Transformer(TFConfig(vocab=V, d_model=64, n_layers=2, n_heads=2, max_len=T + 8, rope=True))


def make_ps(V, T):
    return PRISMSeqLM(PRISMSeqConfig(vocab=V, d_model=64, n_layers=2, n_heads=2, max_len=T + 8))


MODELS = {"Transformer": make_tf, "PRISM-Seq": make_ps}

# The two hard rungs from run_bar.B1 (rung1 already ties at 1.000/0.998; not in question here).
RUNGS = {
    "rung2": dict(vocab=96, num_pairs=32, num_queries=160, gap=0),   # seq 224
    "rung3": dict(vocab=192, num_pairs=64, num_queries=256, gap=0),  # seq 384 (the decisive gate)
}
STEPS = 12000           # long budget so the plateau is visible (matrix used 7000-8000)
LRS = (1e-3, 2e-3, 3e-3)
EVAL_EVERY = 1500
SEED = 0


def solve_step(history, thr=0.9):
    """First eval-step whose acc >= thr (None if never)."""
    for (step, _loss, acc) in history:
        if acc >= thr:
            return step
    return None


def run_one(name, make, tkw, steps, lr):
    task = MQAR(**tkw)
    model = make(task.vocab, task.seq_len)
    p = param_count(model)
    cfg = TrainConfig(steps=steps, batch_size=64, lr=lr, eval_every=EVAL_EVERY, log=False,
                      early_stop_acc=2.0)   # 2.0 => early stop never triggers; full curve retained
    t0 = time.time()
    r = train_model(model, task, cfg, DEV, seed=SEED)
    return {
        "lr": lr, "params": p, "best": r.best_acc, "final": r.final_acc,
        "solve_step@0.9": solve_step(r.history, 0.9),
        "history": [(s, round(l, 4), round(a, 4)) for (s, l, a) in r.history],
        "sec": round(time.time() - t0, 1), "seq_len": task.seq_len, "task": task.name,
    }


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    order = ["rung2", "rung3"] if which == "both" else [which]
    results = {}
    if os.path.exists(OUT):
        try:
            results = json.load(open(OUT))
        except Exception:
            results = {}
    print(f"device={DEV} rungs={order} steps={STEPS} lrs={LRS} seed={SEED}", flush=True)
    for rung in order:
        tkw = RUNGS[rung]
        results.setdefault(rung, {})
        print(f"\n==== CALIB {rung}: MQAR{tkw} (seq={MQAR(**tkw).seq_len}) ====", flush=True)
        for name, make in MODELS.items():
            results[rung].setdefault(name, {})
            for lr in LRS:
                try:
                    rec = run_one(name, make, tkw, STEPS, lr)
                except Exception as e:  # keep partials; report which run died
                    rec = {"lr": lr, "error": repr(e)}
                results[rung][name][f"{lr:.0e}"] = rec
                json.dump(results, open(OUT, "w"), indent=2)   # incremental persist
                ss = rec.get("solve_step@0.9")
                print(f"  [{rung}] {name:<12} lr={lr:.0e}: best={rec.get('best', float('nan')):.3f} "
                      f"final={rec.get('final', float('nan')):.3f} solve@0.9={ss} "
                      f"({rec.get('sec', '?')}s, {rec.get('params', '?')}p)", flush=True)
    # readable verdict
    print("\n==== CALIB SUMMARY (does the Transformer reach plateau within budget?) ====", flush=True)
    for rung in order:
        for name in MODELS:
            arm = results[rung][name]
            best_lr = max(arm, key=lambda l: arm[l].get("best", -1))
            b = arm[best_lr]
            print(f"  {rung} {name:<12} best_lr={best_lr} best_acc={b.get('best', float('nan')):.3f} "
                  f"solve@0.9={b.get('solve_step@0.9')}", flush=True)
    print(f"\nsaved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
