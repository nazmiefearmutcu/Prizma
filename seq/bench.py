"""
The head-to-head harness. Defines the standard attention-diagnostic suite with CALIBRATED
training budgets (enough that a proper Transformer masters each task — established empirically),
and runs ANY model factory through it over multiple seeds with identical data/loss/optimiser/budget.

A model_factory is `f(vocab:int, max_len:int) -> nn.Module` mapping inputs[B,T] long ->
logits[B,T,vocab]. Both Transformer and Prizma-Seq are passed as factories so the comparison is
apples-to-apples (same task instance, same TrainConfig, same seeds).
"""
from __future__ import annotations

import json
import numpy as np

from .common import TrainConfig, train_model, param_count, get_device
from . import tasks as T


# (task_factory, TrainConfig) — budgets calibrated so a 2-layer Transformer solves each.
def standard_suite():
    return {
        "induction":  (lambda: T.Induction(vocab=32, seq_len=64),
                        TrainConfig(steps=3000, batch_size=128, lr=1e-3, eval_every=1000, log=False)),
        "selcopy":    (lambda: T.SelectiveCopy(vocab=32, mem_len=64, n_data=16),
                        TrainConfig(steps=3000, batch_size=128, lr=1e-3, eval_every=1000, log=False)),
        "mqar_p8":    (lambda: T.MQAR(vocab=64, num_pairs=8, num_queries=8),
                        TrainConfig(steps=4000, batch_size=128, lr=1e-3, eval_every=1000, log=False)),
        "mqar_p16":   (lambda: T.MQAR(vocab=64, num_pairs=16, num_queries=16),
                        TrainConfig(steps=6000, batch_size=128, lr=1e-3, eval_every=1500, log=False)),
        "mqar_p8_gap": (lambda: T.MQAR(vocab=64, num_pairs=8, num_queries=8, gap=64),
                        TrainConfig(steps=5000, batch_size=128, lr=1e-3, eval_every=1500, log=False)),
    }


def run_model_on_suite(name, model_factory, seeds=(0, 1, 2), device=None, suite=None, log=True):
    device = device or get_device()
    suite = suite or standard_suite()
    out = {}
    for tname, (tfac, cfg) in suite.items():
        accs, secs, params = [], [], None
        for s in seeds:
            task = tfac()
            model = model_factory(task.vocab, task.seq_len)
            params = param_count(model)
            r = train_model(model, task, cfg, device, seed=s)
            accs.append(r.best_acc)
            secs.append(r.seconds)
            if log:
                print(f"  [{name}] {tname:<12} seed{s}: acc={r.best_acc:.3f} "
                      f"params={params} {r.seconds:.0f}s", flush=True)
        out[tname] = {"accs": accs, "mean": float(np.mean(accs)), "std": float(np.std(accs)),
                      "params": params, "sec_mean": float(np.mean(secs))}
        if log:
            print(f"  [{name}] {tname:<12} => {out[tname]['mean']:.3f} ± {out[tname]['std']:.3f} "
                  f"(params {params})", flush=True)
    return out


if __name__ == "__main__":
    import sys
    from .transformer import Transformer, TFConfig

    def tf_factory(vocab, max_len):
        return Transformer(TFConfig(vocab=vocab, d_model=128, n_layers=2, n_heads=4,
                                    max_len=max_len + 4))

    seeds = (0,) if "--quick" in sys.argv else (0, 1, 2)
    dev = get_device()
    print(f"device={dev} seeds={seeds}")
    res = run_model_on_suite("Transformer", tf_factory, seeds=seeds, device=dev)
    print("\n=== Transformer baseline ===")
    print(json.dumps(res, indent=2))
