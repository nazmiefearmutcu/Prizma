"""Give the Transformer enough budget to COMPLETE its (late, slow) MQAR phase transition, so the
D=64 baseline is legitimate. gen-warm (lr=1e-3, warmup=2000) was climbing at step 12k (0.03->0.24);
this extends the cap to 40k with the plateau-stop (stops when it actually converges). Establishes
the fair TF number to compare against Prizma-quad2 (0.906 @ 12k).

Run: python3.13 tf_long.py [target_pairs] [cap]   # default 64, 40000
"""
from __future__ import annotations

import sys
import time

from seq.common import TrainConfig, train_model, param_count, get_device
from seq.tasks import MixedMQAR
from seq.transformer import Transformer, TFConfig

DEV = get_device()


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
    V = max(64, 4 * target)
    task = MixedMQAR(vocab=V, max_pairs=target, num_queries=128, gap=0, min_pairs=1)
    T = task.seq_len
    m = Transformer(TFConfig(vocab=V, d_model=64, n_layers=2, n_heads=2, max_len=T + 8, rope=True))
    p = param_count(m)
    # gen-warm: the config that began transitioning; large cap + plateau-stop ends it at convergence
    cfg = TrainConfig(steps=cap, batch_size=64, lr=1e-3, warmup=2000, warmup_frac=0.0,
                      min_lr_frac=0.1, log=True, eval_every=2000)
    print(f"device={DEV} {task.name} eval@D={target} seq={T} TF gen-warm cap={cap}", flush=True)
    t0 = time.time()
    r = train_model(m, task, cfg, DEV, seed=0)
    print(f">>> TF gen-warm LONG: best@D{target}={r.best_acc:.3f} plateau@{r.steps_to_plateau} "
          f"({time.time() - t0:.0f}s, {p}p) -> {'SOLVES' if r.best_acc > 0.9 else ('PARTIAL' if r.best_acc > 0.5 else 'no')}",
          flush=True)


if __name__ == "__main__":
    main()
