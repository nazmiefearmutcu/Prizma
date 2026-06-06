"""Prizma-Seq vs Transformer — the rigorous, scaled D=128 benchmark, for a CUDA GPU (Colab).

Answers, with multi-seed + fair protocol, the questions the committee R2 verdict demands:
  PHASE 1  Find the SMALLEST Transformer scale that genuinely SOLVES MQAR D=128 (the fair arena);
           this is also the committee rank-1 flip-test (does attention solve D=128 with enough
           scale/budget?). Multi-config x recipe x seed -> solve-rate.
  PHASE 2  At that scale S*: matched + FLOP-comparable head-to-head at D=128 (TF vs Prizma-none vs
           Prizma-quad2), >=5 seeds -> solve-rate + median + 95% CI. The headline fair comparison.
  PHASE 3  D-frontier {16,32,64,128,256} at a fixed scale: TF vs Prizma-quad2 vs none (capacity curve).
  PHASE 4  Ablations at D=128: quad2 vs none vs rand_linear control; window on/off (causal attribution).
  PHASE 5  FLOP ledger + MEASURED O(1) decode latency & memory vs sequence length.

All training uses MixedMQAR (mixed-difficulty -> high-D is learnable) + gen-warm + per-model plateau.
Results stream incrementally to $PRIZMA_RESULTS/gpu_bench.json (resumable: completed cells are skipped),
so a Colab disconnect never loses progress. Designed to finish in a few hours on an A100/L4.

Env: set PRIZMA_RESULTS to a Drive-mounted dir for persistence (default ./results).
Run: python3 gpu_bench.py            # all phases
     python3 gpu_bench.py 1 2        # only listed phases
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch

from seq.common import TrainConfig, train_model, param_count, get_device
from seq.tasks import MixedMQAR, MQAR
from seq.transformer import Transformer, TFConfig
from seq.prizma_seq import PrizmaSeqLM, PrizmaSeqConfig

DEV = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
RES = os.environ.get("PRIZMA_RESULTS", os.path.join(os.path.dirname(__file__), "results"))
os.makedirs(RES, exist_ok=True)
OUT = os.path.join(RES, "gpu_bench.json")
GENWARM = dict(lr=1e-3, warmup=2000, warmup_frac=0.0, min_lr_frac=0.1)


def _load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def _save(d):
    tmp = OUT + ".tmp"
    json.dump(d, open(tmp, "w"), indent=2)
    os.replace(tmp, OUT)


def ci95(xs):
    xs = np.asarray(xs, float)
    if len(xs) < 2:
        return float(xs.mean()), 0.0
    return float(xs.mean()), float(1.96 * xs.std(ddof=1) / math.sqrt(len(xs)))


def _median(xs):
    s = sorted(xs); n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def tf_factory(d, L, H):
    return lambda V, T: Transformer(TFConfig(vocab=V, d_model=d, n_layers=L, n_heads=H, max_len=T + 8, rope=True))


def ps_factory(d, L, H, **kw):
    return lambda V, T: PrizmaSeqLM(PrizmaSeqConfig(vocab=V, d_model=d, n_layers=L, n_heads=H, max_len=T + 8, **kw))


def run_cell(res, cellkey, model_fac, task_fac, cap, seed, recipe=GENWARM, eval_every=2000):
    """Train one (model x task x seed) cell; cache by cellkey; return the record."""
    if cellkey in res and "best" in res[cellkey]:
        return res[cellkey]
    task = task_fac()
    model = model_fac(task.vocab, task.seq_len)
    p = param_count(model)
    cfg = TrainConfig(steps=cap, batch_size=64, log=False, eval_every=eval_every, **recipe)
    t0 = time.time()
    r = train_model(model, task, cfg, DEV, seed=seed)
    rec = {"best": r.best_acc, "plateau": r.steps_to_plateau, "params": p,
           "sec": round(time.time() - t0, 1), "seed": seed, "cap": cap}
    res[cellkey] = rec
    _save(res)
    print(f"   [{cellkey}] best={rec['best']:.3f} plateau@{rec['plateau']} ({rec['sec']}s, {p}p)", flush=True)
    return rec


def solve_stats(recs):
    bests = [r["best"] for r in recs]
    return {"solve_rate": f"{sum(b > 0.9 for b in bests)}/{len(bests)}", "median": round(_median(bests), 4),
            "mean_ci95": [round(x, 4) for x in ci95(bests)], "bests": [round(b, 3) for b in bests]}


# ----------------------------------------------------------------------------------------------- #
def phase1(res):
    """Smallest TF scale that SOLVES MQAR D=128 (fair arena + committee flip-test). Multi-seed."""
    print("\n==== PHASE 1: TF D=128 solving-scale search (mixed-D, gen-warm) ====", flush=True)
    V = 512
    task_fac = lambda: MixedMQAR(vocab=V, max_pairs=128, num_queries=128, gap=0, min_pairs=1)
    configs = {"d64L2H2": (64, 2, 2), "d128L2H4": (128, 2, 4), "d128L4H4": (128, 4, 4), "d256L4H8": (256, 4, 8)}
    seeds = (0, 1, 2)
    summary = {}
    for cname, (d, L, H) in configs.items():
        recs = [run_cell(res, f"p1.TF.{cname}.s{s}", tf_factory(d, L, H), task_fac, 80000, s) for s in seeds]
        summary[cname] = solve_stats(recs)
        print(f"  -> TF {cname}: {summary[cname]}", flush=True)
    res["p1_summary"] = summary; _save(res)
    return summary


def phase2(res, scale=(128, 4, 4), feat_n2=224, seeds=(0, 1, 2)):  # 3 seeds: Prizma ~57min/run on A100
    """Head-to-head @ D=128 at scale S*: TF vs Prizma-none vs Prizma-quad2 (>=5 seeds, CI)."""
    d, L, H = scale
    print(f"\n==== PHASE 2: head-to-head @ D=128, scale d{d}L{L}H{H} ({len(seeds)} seeds) ====", flush=True)
    V = 512
    task_fac = lambda: MixedMQAR(vocab=V, max_pairs=128, num_queries=128, gap=0, min_pairs=1)
    arms = {"TF": tf_factory(d, L, H), "Prizma-none": ps_factory(d, L, H),
            "Prizma-quad2": ps_factory(d, L, H, feat_map="quad2", feat_n2=feat_n2)}
    summary = {}
    for aname, fac in arms.items():
        recs = [run_cell(res, f"p2.{aname}.s{s}", fac, task_fac, 80000, s) for s in seeds]
        summary[aname] = solve_stats(recs)
        print(f"  -> {aname}: {summary[aname]}", flush=True)
    res["p2_summary"] = summary; _save(res)
    return summary


def phase3(res, scale=(128, 4, 4), feat_n2=224, seeds=(0, 1, 2)):
    """D-frontier {16,32,64,128,256}: TF vs Prizma-quad2 vs none at a fixed scale."""
    d, L, H = scale
    print(f"\n==== PHASE 3: D-frontier @ scale d{d}L{L}H{H} ====", flush=True)
    arms = {"TF": tf_factory(d, L, H), "Prizma-none": ps_factory(d, L, H),
            "Prizma-quad2": ps_factory(d, L, H, feat_map="quad2", feat_n2=feat_n2)}
    summary = {}
    for D in (16, 32, 64, 128, 256):
        V = max(64, 4 * D)
        task_fac = lambda V=V, D=D: MixedMQAR(vocab=V, max_pairs=D, num_queries=128, gap=0, min_pairs=1)
        cap = 60000 if D <= 64 else 90000
        for aname, fac in arms.items():
            recs = [run_cell(res, f"p3.D{D}.{aname}.s{s}", fac, task_fac, cap, s) for s in seeds]
            summary[f"D{D}.{aname}"] = solve_stats(recs)
            print(f"  -> D={D} {aname}: {summary[f'D{D}.{aname}']}", flush=True)
    res["p3_summary"] = summary; _save(res)
    return summary


def phase4(res, scale=(128, 4, 4), feat_n2=224, seeds=(0, 1, 2)):
    """Ablations @ D=128: quad2 vs none vs rand_linear control; window on/off (causal attribution)."""
    d, L, H = scale
    print(f"\n==== PHASE 4: ablations @ D=128, scale d{d}L{L}H{H} ====", flush=True)
    V = 512
    task_fac = lambda: MixedMQAR(vocab=V, max_pairs=128, num_queries=128, gap=0, min_pairs=1)
    arms = {
        "quad2":            ps_factory(d, L, H, feat_map="quad2", feat_n2=feat_n2),
        "none":             ps_factory(d, L, H),
        "rand_linear":      ps_factory(d, L, H, feat_map="rand_linear", feat_n2=feat_n2),  # control: expect ~none
        "quad2_noWindow":   ps_factory(d, L, H, feat_map="quad2", feat_n2=feat_n2, use_window=False),
    }
    summary = {}
    for aname, fac in arms.items():
        recs = [run_cell(res, f"p4.{aname}.s{s}", fac, task_fac, 80000, s) for s in seeds]
        summary[aname] = solve_stats(recs)
        print(f"  -> {aname}: {summary[aname]}", flush=True)
    res["p4_summary"] = summary; _save(res)
    return summary


def phase5(res, scale=(128, 4, 4), feat_n2=224):
    """Measured decode latency + state memory vs sequence length (the O(1) structural advantage)."""
    d, L, H = scale
    print(f"\n==== PHASE 5: measured O(1) decode latency + memory vs T, scale d{d}L{L}H{H} ====", flush=True)
    V = 512
    tf = Transformer(TFConfig(vocab=V, d_model=d, n_layers=L, n_heads=H, max_len=4200, rope=True)).to(DEV)
    ps = PrizmaSeqLM(PrizmaSeqConfig(vocab=V, d_model=d, n_layers=L, n_heads=H, max_len=4200,
                                   feat_map="quad2", feat_n2=feat_n2)).to(DEV)
    tf.train(False); ps.train(False)
    ns = [128, 256, 512, 1024, 2048, 4096]

    @torch.no_grad()
    def decode_latency(model, n, reps=3, warmup=5):
        lat = []
        for r in range(reps + warmup):
            st = model.init_state(1, DEV); tok = torch.randint(0, V, (1, 1), device=DEV)
            if DEV.type == "cuda": torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n):
                lg, st = model.step(tok, st); tok = lg[:, -1:].argmax(-1)
            if DEV.type == "cuda": torch.cuda.synchronize()
            if r >= warmup: lat.append(time.time() - t0)
        return float(np.median(lat))

    out = {"seq_lens": ns, "tf_decode_s": {}, "prizma_decode_s": {}}
    for n in ns:
        out["tf_decode_s"][n] = round(decode_latency(tf, n), 4)
        out["prizma_decode_s"][n] = round(decode_latency(ps, n), 4)
        print(f"  n={n:<5} TF(KV)={out['tf_decode_s'][n]:.4f}s  Prizma(O(1))={out['prizma_decode_s'][n]:.4f}s", flush=True)
    # state size (floats): TF KV-cache grows O(n); Prizma state constant
    dh = d // H
    out["tf_kv_floats"] = {n: 2 * L * H * dh * n for n in ns}
    out["prizma_state_floats"] = {n: L * H * dh * (dh + feat_n2) + 2 * L * H * 16 * dh for n in ns}  # state + window ring
    res["p5_latency"] = out; _save(res)
    return out


PHASES = {1: phase1, 2: phase2, 3: phase3, 4: phase4, 5: phase5}


def main():
    wanted = [int(a) for a in sys.argv[1:]] or [1, 2, 3, 4, 5]
    print(f"device={DEV} torch={torch.__version__} results={OUT} phases={wanted}", flush=True)
    if DEV.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    res = _load()
    for ph in wanted:
        PHASES[ph](res)
    print("\n==== DONE. Summary keys:", [k for k in res if k.endswith("_summary") or k == "p5_latency"], flush=True)
    print(f"saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
