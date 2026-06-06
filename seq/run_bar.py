"""
The falsifiable-bar driver: runs PRISM-Seq vs the param-matched Transformer over the agreed
suite and writes results/bar_results.json with per-seed numbers + CIs + param ledger.

Usage:
  python3.13 -m seq.run_bar b1      # MQAR rungs (decisive gate)
  python3.13 -m seq.run_bar b1b     # MQAR capacity sweep (run FIRST; pre-register D*)
  python3.13 -m seq.run_bar b3      # selective copy (+ fixed control)
  python3.13 -m seq.run_bar b5      # inference-cost advantage
  python3.13 -m seq.run_bar b6      # mechanism ablations
  python3.13 -m seq.run_bar all
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import sys

import numpy as np
import torch

from .common import TrainConfig, train_model, param_count, get_device, autoregressive_latency
from .tasks import MQAR, SelectiveCopy, Induction
from .transformer import Transformer, TFConfig
from .prism_seq import PRISMSeqLM, PRISMSeqConfig

RES = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RES, exist_ok=True)


def ci95(xs):
    xs = np.asarray(xs, float)
    if len(xs) < 2:
        return float(xs.mean()), 0.0
    return float(xs.mean()), float(1.96 * xs.std(ddof=1) / math.sqrt(len(xs)))


def tf_factory(d, L, H):
    return lambda V, T: Transformer(TFConfig(vocab=V, d_model=d, n_layers=L, n_heads=H,
                                             max_len=T + 8, rope=True))


def ps_factory(d, L, H, **kw):
    return lambda V, T: PRISMSeqLM(PRISMSeqConfig(vocab=V, d_model=d, n_layers=L, n_heads=H,
                                                  max_len=T + 8, **kw))


def _median(xs):
    s = sorted(xs); n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def run_pair(tag, task_fac, models, cfg, seeds, dev, lr_grid=(1e-3, 2e-3)):
    """models: dict name->factory(V,T). SYMMETRIC per-model LR sweep: every model is run at every
    LR in lr_grid over all seeds; each model's reported result is its OWN best-LR (by median acc),
    so the comparison is best-vs-best (fair) — no architecture is denied an LR the other gets.
    Returns {name:{accs(best-lr), mean, ci95, median, best_lr, sweep, params, sec}}."""
    out = {}
    for name, fac in models.items():
        sweep, params = {}, None
        for lr in lr_grid:
            cfg_m = dataclasses.replace(cfg, lr=lr)
            accs, secs = [], []
            for s in seeds:
                task = task_fac()
                model = fac(task.vocab, task.seq_len)
                params = param_count(model)
                r = train_model(model, task, cfg_m, dev, seed=s)
                accs.append(r.best_acc); secs.append(r.seconds)
            sweep[f"{lr:.0e}"] = {"accs": accs, "median": _median(accs), "sec": float(np.mean(secs))}
            print(f"  [{tag}] {name:<14} lr={lr:.0e}: med={_median(accs):.3f} "
                  f"accs={[round(a,3) for a in accs]} ({params}p)", flush=True)
        best_lr = max(sweep, key=lambda l: sweep[l]["median"])
        accs = sweep[best_lr]["accs"]
        m, c = ci95(accs)
        out[name] = {"accs": accs, "mean": m, "ci95": c, "median": _median(accs),
                     "best_lr": best_lr, "sweep": sweep, "params": params,
                     "sec": sweep[best_lr]["sec"]}
        print(f"  [{tag}] {name:<14} => best_lr={best_lr} median={_median(accs):.3f} "
              f"solved={sum(a>0.9 for a in accs)}/{len(accs)}", flush=True)
    return out


# ------------------------------------- B1: MQAR rungs ------------------------------------- #
def B1(dev, seeds=(0, 1, 2)):
    rungs = [
        ("rung1", dict(vocab=64, num_pairs=16, num_queries=96, gap=0), 6000),    # seq 128
        ("rung2", dict(vocab=96, num_pairs=32, num_queries=160, gap=0), 7000),   # seq 256
        ("rung3", dict(vocab=192, num_pairs=64, num_queries=256, gap=0), 8000),  # seq 384 (the gate)
    ]
    res = {}
    for rname, tkw, steps in rungs:
        task_fac = lambda tkw=tkw: MQAR(**tkw)
        T = MQAR(**tkw).seq_len
        cfg = TrainConfig(steps=steps, batch_size=64, lr=2e-3, eval_every=steps // 4, log=False)
        models = {"Transformer": tf_factory(64, 2, 2), "PRISM-Seq": ps_factory(64, 2, 2)}
        print(f"\n== B1 {rname}: {MQAR(**tkw).name} (seq={T}, steps={steps}) ==", flush=True)
        res[rname] = run_pair(f"B1-{rname}", task_fac, models, cfg, seeds, dev)
    json.dump(res, open(os.path.join(RES, "b1_mqar.json"), "w"), indent=2)
    return res


# ------------------------------- B1b: MQAR capacity sweep --------------------------------- #
def B1b(dev, seeds=(0, 1, 2)):
    res = {"transformer": {}, "prism_by_dh": {}}
    Ds = [8, 16, 32, 64, 128]
    cfg = lambda: TrainConfig(steps=6000, batch_size=64, lr=2e-3, eval_every=1500, log=False)
    for D in Ds:
        V = max(64, 4 * D)                      # disjoint key/value ranges fit comfortably
        task_fac = lambda D=D, V=V: MQAR(vocab=V, num_pairs=D, num_queries=128, gap=0)
        # transformer reference (d=64) — unbounded recall, expected near-ceiling across all D
        r = run_pair(f"B1b-D{D}", task_fac, {"Transformer": tf_factory(64, 2, 2)}, cfg(), seeds, dev)
        res["transformer"][D] = r["Transformer"]
        # PRISM-Seq at two recall ranks: d_h=32 (the param-matched config) and d_h=64 (more state).
        for dh in [32, 64]:
            d = 2 * dh
            r = run_pair(f"B1b-D{D}-dh{dh}", task_fac, {"PRISM-Seq": ps_factory(d, 2, 2)}, cfg(), seeds, dev)
            res["prism_by_dh"].setdefault(dh, {})[D] = r["PRISM-Seq"]
    json.dump(res, open(os.path.join(RES, "b1b_capacity.json"), "w"), indent=2)
    return res


# ----------------------------------- B3: selective copy ----------------------------------- #
def B3(dev, seeds=(0, 1, 2)):
    res = {}
    cfg = TrainConfig(steps=5000, batch_size=64, lr=2e-3, eval_every=1250, log=False)
    for variant, fixed in [("selective", False), ("fixed", True)]:
        task_fac = lambda fixed=fixed: SelectiveCopy(vocab=32, mem_len=64, n_data=16, fixed=fixed)
        models = {"Transformer": tf_factory(64, 2, 2), "PRISM-Seq": ps_factory(64, 2, 2),
                  "PRISM-noGate": ps_factory(64, 2, 2, precision_gate="uniform")}
        print(f"\n== B3 {variant}: {task_fac().name} ==", flush=True)
        res[variant] = run_pair(f"B3-{variant}", task_fac, models, cfg, seeds, dev)
    json.dump(res, open(os.path.join(RES, "b3_selcopy.json"), "w"), indent=2)
    return res


# ------------------------------- B5: inference cost advantage ----------------------------- #
def _measure_decode_memory(model, vocab, n, dev):
    """MEASURED peak decode-state memory (bytes): generate n tokens via the O(1)/KV-cache step path,
    holding state, and read torch.mps allocated bytes (state/cache is largest at the end). Returns
    bytes attributable to the carried state/cache (delta vs the just-initialized baseline)."""
    if dev.type != "mps":
        return None
    import torch
    model.train(False); model.to(dev)
    torch.mps.empty_cache(); torch.mps.synchronize()
    base = torch.mps.current_allocated_memory()
    with torch.no_grad():
        state = model.init_state(1, dev)
        tok = torch.randint(0, vocab, (1, 1), device=dev)
        peak = 0
        for i in range(n):
            logits, state = model.step(tok, state)
            tok = logits[:, -1:].argmax(-1)
            if (i + 1) % max(1, n // 8) == 0:
                torch.mps.synchronize()
                peak = max(peak, torch.mps.current_allocated_memory() - base)
    torch.mps.synchronize()
    peak = max(peak, torch.mps.current_allocated_memory() - base)
    return int(peak)


def B5(dev):
    V, ns = 128, [128, 256, 512, 1024, 2048, 4096]
    d, L, H = 192, 3, 4
    dh = d // H
    tf = Transformer(TFConfig(vocab=V, d_model=d, n_layers=L, n_heads=H, max_len=4200, rope=True))
    ps = PRISMSeqLM(PRISMSeqConfig(vocab=V, d_model=d, n_layers=L, n_heads=H, max_len=4200, gated=True))
    # both via their honest streaming path: Transformer = KV-cache (O(t)/step); PRISM-Seq = O(1)/step
    tf_lat = autoregressive_latency(tf, V, ns, dev, reps=3, step_api=True)
    ps_lat = autoregressive_latency(ps, V, ns, dev, reps=3, step_api=True)
    # MEASURED peak decode-state memory (bytes) via torch.mps (not analytic)
    tf_mem = {n: _measure_decode_memory(tf, V, n, dev) for n in ns}
    ps_mem = {n: _measure_decode_memory(ps, V, n, dev) for n in ns}
    # analytic reference (floats) for cross-check
    tf_mem_an = {n: 2 * L * H * dh * n for n in ns}
    ps_mem_an = {n: L * H * dh * dh + 2 * L * H * 16 * dh for n in ns}   # state + window ring (w=16)
    res = {"seq_lens": ns, "transformer_kvcache_s": tf_lat, "prism_step_s": ps_lat,
           "transformer_state_bytes_measured": tf_mem, "prism_state_bytes_measured": ps_mem,
           "transformer_state_floats_analytic": tf_mem_an, "prism_state_floats_analytic": ps_mem_an,
           "tf_params": param_count(tf), "ps_params": param_count(ps)}
    print("\n== B5 inference: per-call latency + MEASURED decode-state memory (MPS bytes) ==")
    for n in ns:
        print(f"  n={n:<5} TF(KV)={tf_lat[n]:.3f}s mem={tf_mem[n]} | "
              f"PRISM(step)={ps_lat[n]:.3f}s mem={ps_mem[n]}", flush=True)
    json.dump(res, open(os.path.join(RES, "b5_latency.json"), "w"), indent=2, default=float)
    return res


# ----------------------------------- B4: char-LM (BPC) ------------------------------------ #
def B4(dev, seeds=(0, 1)):
    import time
    from .charlm import CharLM, val_bpc
    from .common import set_seed
    import torch.nn as nn
    res = {}
    charlm = CharLM(seq_len=256)
    V = charlm.vocab
    print(f"\n== B4 char-LM: {charlm.name}  (train {len(charlm.train)}, val {len(charlm.val)}, "
          f"test {len(charlm.test)}) ==", flush=True)
    models = {
        "Transformer": lambda: Transformer(TFConfig(vocab=V, d_model=192, n_layers=3, n_heads=4,
                                                    max_len=300, rope=True)),
        "PRISM-Seq": lambda: PRISMSeqLM(PRISMSeqConfig(vocab=V, d_model=192, n_layers=3, n_heads=4,
                                                      max_len=300, gated=True)),
    }
    steps, bs = 4000, 32

    def train_one(fac, lr, s):
        set_seed(s)
        model = fac().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
        best_val, best_test, t0 = 1e9, 1e9, time.time()
        for step in range(steps):
            for g in opt.param_groups:
                g["lr"] = (lr * (step + 1) / 200) if step < 200 else \
                    lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - 200) / (steps - 200))))
            model.train()
            x, y, m = charlm.sample(bs, dev)
            loss = torch.nn.functional.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if (step + 1) % 500 == 0 or step == steps - 1:
                vb = val_bpc(model, charlm, dev, batches=30)
                if vb < best_val:
                    best_val, best_test = vb, val_bpc_on(model, charlm, dev, split="test")
        return best_val, best_test, param_count(model), time.time() - t0

    for name, fac in models.items():
        # symmetric per-model LR sweep on char-LM too (pick best VAL bpc), then report TEST bpc.
        sweep = {}
        for lr in (1e-3, 2e-3):
            vts = [train_one(fac, lr, s) for s in seeds]
            sweep[lr] = vts
            print(f"  [B4] {name:<12} lr={lr:.0e}: val={np.mean([v[0] for v in vts]):.3f} "
                  f"test={np.mean([v[1] for v in vts]):.3f}", flush=True)
        best_lr = min(sweep, key=lambda l: np.mean([v[0] for v in sweep[l]]))
        vts = sweep[best_lr]
        test_bpcs = [v[1] for v in vts]
        m, c = ci95(test_bpcs)
        res[name] = {"test_bpcs": test_bpcs, "mean": m, "ci95": c, "best_lr": f"{best_lr:.0e}",
                     "params": vts[0][2], "corpus": "tiny-shakespeare", "n_seeds": len(seeds)}
        print(f"  [B4] {name:<12} => best_lr={best_lr:.0e} test BPC {m:.3f} ± {c:.3f} "
              f"(shakespeare only; text8 = future)", flush=True)
    json.dump(res, open(os.path.join(RES, "b4_charlm.json"), "w"), indent=2)
    return res


def val_bpc_on(model, charlm, dev, split="test", batches=40, bs=64):
    import math as _m
    import torch.nn.functional as F
    model.train(False)
    data = getattr(charlm, split)
    tot, n = 0.0, 0
    with torch.no_grad():
        for _ in range(batches):
            import numpy as _np
            T = charlm.seq_len
            ix = _np.random.randint(0, len(data) - T - 1, size=bs)
            x = torch.from_numpy(_np.stack([data[i:i + T] for i in ix])).to(dev)
            y = torch.from_numpy(_np.stack([data[i + 1:i + 1 + T] for i in ix])).to(dev)
            ce = F.cross_entropy(model(x).reshape(-1, charlm.vocab), y.reshape(-1), reduction="sum")
            tot += float(ce); n += y.numel()
    return (tot / n) / _m.log(2)


# ------------------------------- B6: mechanism ablations ---------------------------------- #
def B6(dev, seeds=(0, 1, 2)):
    from .baselines_seq import GRULM, LinAttnLM
    res = {}
    # internal ablations (each removes ONE PRISM piece, param-matched within ±5%) + family controls.
    # GRU sized to d=90 to land within ±5% of PRISM full (~102.6K) — NOT the old −47% strawman.
    def common_models():
        return {
            "full":          ps_factory(64, 2, 2),
            "noDelta":       ps_factory(64, 2, 2, write_mode="additive"),       # additive = linear-attn/SSM family (internal, exactly matched)
            "noPrecision":   ps_factory(64, 2, 2, precision_gate="uniform"),
            "randomGate":    ps_factory(64, 2, 2, precision_gate="random"),     # gate control
            "noWorkspace":   ps_factory(64, 2, 2, use_workspace=False),
            "noWindow":      ps_factory(64, 2, 2, use_window=False),
            "noConv":        ps_factory(64, 2, 2, short_conv=0),                # the disputed-claim ablation
            "noRouteReadout": ps_factory(64, 2, 2, route_readout=False),
            "GRU":           lambda V, T: GRULM(V, d_model=90, n_layers=2),     # param-matched family control
            "LinAttn":       lambda V, T: LinAttnLM(V, d_model=64, n_layers=2, n_heads=2),
        }
    mqar = lambda: MQAR(vocab=64, num_pairs=32, num_queries=128, gap=0)   # seq 192
    cfg_m = TrainConfig(steps=6000, batch_size=64, eval_every=2000, log=False)
    print(f"\n== B6 MQAR ablations: {mqar().name} (fixed lr=2e-3; mechanism, not LR, under test) ==", flush=True)
    res["mqar"] = run_pair("B6-mqar", mqar, common_models(), cfg_m, seeds, dev, lr_grid=(2e-3,))
    sel = lambda: SelectiveCopy(vocab=32, mem_len=64, n_data=16, fixed=False)
    cfg_s = TrainConfig(steps=5000, batch_size=64, eval_every=1250, log=False)
    print(f"\n== B6 SelectiveCopy ablations: {sel().name} (fixed lr=2e-3) ==", flush=True)
    res["selcopy"] = run_pair("B6-selcopy", sel, common_models(), cfg_s, seeds, dev, lr_grid=(2e-3,))
    json.dump(res, open(os.path.join(RES, "b6_ablations.json"), "w"), indent=2)
    return res


# ----------------------------------- B2: induction --------------------------------------- #
def B2(dev, seeds=(0, 1, 2)):
    task_fac = lambda: Induction(vocab=40, seq_len=256)
    cfg = TrainConfig(steps=6000, batch_size=128, eval_every=1500, log=False)
    models = {"Transformer": tf_factory(64, 2, 2), "PRISM-Seq": ps_factory(64, 2, 2)}
    print(f"\n== B2 induction: {task_fac().name} ==", flush=True)
    res = run_pair("B2", task_fac, models, cfg, seeds, dev)
    json.dump(res, open(os.path.join(RES, "b2_induction.json"), "w"), indent=2)
    return res


if __name__ == "__main__":
    dev = get_device()
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    nseed = 1 if "--quick" in sys.argv else 3
    for a in sys.argv:
        if a.startswith("--seeds="):
            nseed = int(a.split("=")[1])
    seeds = tuple(range(nseed))
    print(f"device={dev} which={which} seeds={seeds}", flush=True)
    fns = {"b1": B1, "b1b": B1b, "b2": B2, "b3": B3, "b4": B4, "b6": B6, "b5": lambda d, s=None: B5(d)}
    if which == "all":
        # minimal-viable bar (B1+B6+B5+B3+B1b) FIRST, then B2/B4 secondary
        for k in ["b1", "b6", "b5", "b3", "b1b", "b2", "b4"]:
            (B5(dev) if k == "b5" else fns[k](dev, seeds))
    elif which == "b5":
        B5(dev)
    else:
        fns[which](dev, seeds)
