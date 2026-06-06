"""Engineer-a-path test: does MIXED-difficulty MQAR training (variable #pairs/batch, the standard
Zoology distribution) make TARGET-D recall learnable where fixed-D training stalls at chance?

Trains the matched tiny TF, PRISM-none, and PRISM-quad2 on MixedMQAR(pairs 1..target), evaluates at
the FIXED target D, seed 0, punchy short-warmup schedule. If the TF now solves -> the path is found
(fair, applies to both); and we read off whether the zero-param quad2 lever holds PRISM at parity.

Run: python3.13 mixed_test.py [target_pairs]   # default 64
"""
from __future__ import annotations

import sys
import time

from seq.common import TrainConfig, train_model, param_count, get_device
from seq.tasks import MixedMQAR
from seq.transformer import Transformer, TFConfig
from seq.prism_seq import PRISMSeqLM, PRISMSeqConfig

DEV = get_device()
PUNCHY = dict(lr=1e-3, warmup=200, warmup_frac=0.0, min_lr_frac=0.1)


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    V = max(64, 4 * target)
    task = MixedMQAR(vocab=V, max_pairs=target, num_queries=128, gap=0, min_pairs=1)
    T = task.seq_len

    def tf():
        return Transformer(TFConfig(vocab=V, d_model=64, n_layers=2, n_heads=2, max_len=T + 8, rope=True))

    def ps():
        return PRISMSeqLM(PRISMSeqConfig(vocab=V, d_model=64, n_layers=2, n_heads=2, max_len=T + 8))

    def psq():
        return PRISMSeqLM(PRISMSeqConfig(vocab=V, d_model=64, n_layers=2, n_heads=2, max_len=T + 8,
                                         feat_map="quad2", feat_n2=96))

    print(f"device={DEV} {task.name} eval@D={target} seq={T}", flush=True)
    for name, make in [("Transformer", tf), ("PRISM-none", ps), ("PRISM-quad2", psq)]:
        m = make()
        p = param_count(m)
        cfg = TrainConfig(steps=12000, batch_size=64, log=True, eval_every=1000, **PUNCHY)
        t0 = time.time()
        r = train_model(m, task, cfg, DEV, seed=0)
        print(f">>> {name:<12} best@D{target}={r.best_acc:.3f} plateau@{r.steps_to_plateau} "
              f"({time.time() - t0:.0f}s, {p}p)", flush=True)


if __name__ == "__main__":
    main()
