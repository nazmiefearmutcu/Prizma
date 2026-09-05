"""
Analog-robustness probe (report 12-H2, commission item 7.6): "the delta rule is analog-robust".

H2a: the delta write stores the PREDICTION ERROR (self-correcting), so a low-precision carried
state S with noisy writes degrades more gracefully than an additive write (which accumulates
errors). H2b (counter-hypothesis): the correction needs a precise read of S.k, so a quantized S
corrupts the residual and delta degrades at least as fast. Either outcome is informative.

Protocol (train-FP32 / deploy-degraded — the knobs are wired to step() ONLY, see
PrizmaSeqConfig): train ONE clean model per (write_mode x seed), then stream the frozen eval set
through the exact O(1) step() path under every (state_bits x write_noise_std) condition. The
streaming knobs-off cell doubles as the O(1)-sanity cell (it should match the parallel forward
accuracy to ~1e-4-level numerics).

Grid: {delta, additive} x {state_bits: 0,4,6,8} x {write_noise_std: 0.0,0.01,0.05} x 3 seeds.
Task: MixedMQAR (the repo's curriculum MQAR; eval fixed at the target difficulty via eval_sample).
D=64 was PILOTED and dropped for time (measured 254-502 ms/step on this box => the 6-run grid
alone is >1.7 h); the probe runs D=32 instead — disclosed in RESULTS.md.

Raw records: crash-safe per-seed JSON (json -> .tmp -> os.replace) written BEFORE any verdict is
computed (docs/RETENTION.md). LANE-EXPLORATORY (n=3): hypothesis-flagging only, no claim.

Usage:
    python -m seq.analog_probe --pilot    # one delta cell: time + best_acc (grid-sizing gate)
    python -m seq.analog_probe --smoke    # plumbing-only tiny grid -> results/.../_smoke/
    python -m seq.analog_probe            # the full n=3 grid -> results/analog_probe_2026-09-03/
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

RESULTS_DIR = os.path.join("results", "analog_probe_2026-09-03")

# ---- frozen probe config (pilot-selected, then frozen for the whole grid) ----
TASK_KW = dict(vocab=128, max_pairs=32, num_queries=64, gap=0, min_pairs=1)   # MixedMQAR D=32
SCALE_KW = dict(d_model=64, n_layers=2, n_heads=2)                             # d64L2H2 (repo scale)
FEAT_MAP = "none"          # pilot-verified solve; 'quad2_lowrank' doubles step time (502 ms)
LR = 2e-3                  # frozen from the 3-LR pilot (1e-3: 0.745@3k, 2e-3: 0.867@2k/281s,
                           # 3e-3: 0.776@3k; delta seed 0 — selected once, then frozen, NO-TUNING)
STEPS = 3000
BATCH_SIZE = 64
EVAL_BATCHES = 8           # disclosed: 8 frozen eval batches (budget), eval_seed discipline kept
EVAL_SEED = 12345
SEEDS = (0, 1, 2)
STATE_BITS_LIST = (0, 4, 6, 8)
NOISE_LIST = (0.0, 0.01, 0.05)
WRITE_MODES = ("delta", "additive")
THREADS = 8                # measured optimum on this 16-thread box (tiny-model thread thrash)
BUDGET_S = 2 * 3600        # hard CPU budget; at >90 min remaining models drop seed 2 (disclosed)
DEGRADE_AT_S = 90 * 60


def _task():
    from .tasks import MixedMQAR
    return MixedMQAR(**TASK_KW)


def _model_factory(write_mode):
    from .prizma_seq import prizma_seq_factory
    return prizma_seq_factory(write_mode=write_mode, feat_map=FEAT_MAP, **SCALE_KW)


def _save(path, d):
    """Crash-safe json write (json -> .tmp -> os.replace), the RETENTION.md discipline."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, path)


@torch.no_grad()
def _frozen_eval(task, batches, device):
    from .common import set_seed
    set_seed(EVAL_SEED)
    return [task.eval_sample(BATCH_SIZE, device) for _ in range(batches)]


@torch.no_grad()
def _stream_eval(model, frozen, state_bits, noise_std):
    """Stream every frozen batch through the O(1) step() path under the given knobs; returns
    (masked_acc, masked_ce). Knob mutation lives on the shared cfg object; restored by caller.
    Noise draws use per-step OWNED generators (write_noise_seed + pos), so conditions are
    deterministic given the model — no global-RNN dependence, no seeding needed here."""
    from .common import masked_acc, masked_ce
    model.cfg.state_bits = state_bits
    model.cfg.write_noise_std = noise_std
    model.train(False)
    accs, losses = [], []
    for x, y, m in frozen:
        st = model.init_state(x.shape[0], x.device)
        outs = []
        for t in range(x.shape[1]):
            lg, st = model.step(x[:, t:t + 1], st)
            outs.append(lg)
        logits = torch.cat(outs, dim=1)
        accs.append(masked_acc(logits, y, m))
        losses.append(float(masked_ce(logits, y, m)))
    return float(np.mean(accs)), float(np.mean(losses))


def _reset_knobs(model):
    model.cfg.state_bits = 0
    model.cfg.write_noise_std = 0.0


def _eval_all_conditions(model, frozen):
    """Every (state_bits, noise) cell + the parallel-forward reference. Returns list of dicts."""
    from .common import masked_acc
    cells = []
    for b in STATE_BITS_LIST:
        for s in NOISE_LIST:
            acc, ce = _stream_eval(model, frozen, b, s)
            cells.append(dict(state_bits=b, write_noise_std=s, stream_acc=acc, stream_ce=ce))
    _reset_knobs(model)
    model.train(False)
    fwd_acc = float(np.mean([masked_acc(model(x), y, m) for x, y, m in frozen]))
    return cells, fwd_acc


def _train_one(write_mode, seed, steps, device, log=True):
    """build_and_train's exact seeding recipe (set_seed BEFORE construction), keeping the model
    handle for the streaming evaluation."""
    from .common import TrainConfig, train_model, set_seed
    cfg = TrainConfig(steps=steps, batch_size=BATCH_SIZE, lr=LR, eval_every=500,
                      warmup=300, warmup_frac=0.0, min_lr_frac=0.1, log=False)
    task = _task()
    set_seed(seed)
    model = _model_factory(write_mode)(task.vocab, task.seq_len)
    r = train_model(model, task, cfg, device, seed=seed)
    if log:
        print(f"  [{write_mode}] seed{seed}: best_acc={r.best_acc:.3f} "
              f"final={r.final_acc:.3f} {r.seconds:.0f}s params={r.params}", flush=True)
    return r, model


def run(out_dir=RESULTS_DIR, seeds=SEEDS, steps=STEPS, smoke=False):
    torch.set_num_threads(THREADS)
    device = torch.device("cpu")          # CPU-only probe (zero-GPU program; deterministic)
    raw_path = os.path.join(out_dir, "raw_analog_probe.json")
    t0 = time.time()
    degraded = None                        # set to a disclosure string if the budget forces it
    print(f"[analog_probe] device={device} threads={THREADS} out={out_dir}")
    print(f"[analog_probe] task=MixedMQAR(D={TASK_KW['max_pairs']}) scale=d{SCALE_KW['d_model']}L"
          f"{SCALE_KW['n_layers']}H{SCALE_KW['n_heads']} feat_map={FEAT_MAP} lr={LR} steps={steps} "
          f"seeds={list(seeds)} grid={len(WRITE_MODES)}x{len(STATE_BITS_LIST)}x{len(NOISE_LIST)}",
          flush=True)

    records = {"meta": dict(task=TASK_KW, scale=SCALE_KW, feat_map=FEAT_MAP, lr=LR, steps=steps,
                            batch_size=BATCH_SIZE, eval_batches=EVAL_BATCHES,
                            eval_seed=EVAL_SEED, seeds=list(seeds),
                            state_bits=list(STATE_BITS_LIST), noise=list(NOISE_LIST),
                            write_modes=list(WRITE_MODES), threads=THREADS, smoke=smoke,
                            device=str(device), degraded_seeds=None),
               "runs": []}
    if os.path.exists(raw_path):           # resume: keep earlier records (append-only discipline)
        with open(raw_path, encoding="utf-8") as f:
            old = json.load(f)
        records["runs"] = [r for r in old.get("runs", [])]
        print(f"[analog_probe] resuming with {len(records['runs'])} existing runs", flush=True)
    _save(raw_path, records)

    done = {(r["write_mode"], r["seed"]) for r in records["runs"]}
    for mode in WRITE_MODES:
        for seed in seeds:
            if (mode, seed) in done:
                print(f"[analog_probe] skip cached {mode} seed{seed}", flush=True)
                continue
            if time.time() - t0 > DEGRADE_AT_S and seed == 2 and not smoke:
                degraded = ("seeds reduced to (0, 1) for the remaining models: the >90 min "
                            "budget guard fired (hard rule: total <=2 h CPU). Disclosed.")
                print(f"[analog_probe] BUDGET GUARD: {degraded}", flush=True)
                records["meta"]["degraded_seeds"] = degraded
                _save(raw_path, records)
                continue
            r, model = _train_one(mode, seed, steps, device)
            task = _task()
            frozen = _frozen_eval(task, 2 if smoke else EVAL_BATCHES, device)
            cells, fwd_acc = _eval_all_conditions(model, frozen)
            rec = dict(write_mode=mode, seed=seed,
                       train=dict(best_acc=r.best_acc, final_acc=r.final_acc,
                                  final_loss=r.final_loss, seconds=r.seconds,
                                  steps_to_plateau=r.steps_to_plateau, params=r.params,
                                  history=r.history),
                       forward_acc=fwd_acc, cells=cells)
            records["runs"].append(rec)
            _save(raw_path, records)       # raw-first: persisted before any verdict
            print(f"[analog_probe] saved {mode} seed{seed} "
                  f"({len(records['runs'])} runs, {time.time()-t0:.0f}s elapsed)", flush=True)

    summary = _summarize(records, out_dir)
    print(summary)
    return records


def _summarize(records, out_dir):
    """Aggregate + verdict table. Retention = acc_cell / acc_stream(bits=0, noise=0) per seed."""
    runs = records["runs"]
    lines = ["=" * 100, "  ANALOG PROBE — MixedMQAR D=32, d64L2H2, train-FP32/deploy-degraded "
             "(LANE-EXPLORATORY, n=3)", "=" * 100]
    table = {}
    for r in runs:
        clean = next((c["stream_acc"] for c in r["cells"]
                      if c["state_bits"] == 0 and c["write_noise_std"] == 0.0), None)
        for c in r["cells"]:
            key = (c["state_bits"], c["write_noise_std"])
            table.setdefault(key, {}).setdefault(r["write_mode"], []).append(
                (c["stream_acc"], c["stream_acc"] / clean if clean else float("nan")))
    header = f"{'bits':>4} {'noise':>5} | {'delta acc':>18} | {'additive acc':>18} | " \
             f"{'delta ret':>18} | {'additive ret':>18} | {'d-a acc':>7}"
    lines.append(header)
    lines.append("-" * len(header))
    for key in sorted(table):
        b, s = key
        row = f"{b:>4} {s:>5} |"
        accs = {}
        for mode in WRITE_MODES:
            v = table[key].get(mode, [])
            if v:
                a = np.array([x[0] for x in v]); rr = np.array([x[1] for x in v])
                accs[mode] = a.mean()
                row += f" {a.mean():.3f}±{a.std():.3f} (n={len(v)})".rjust(19) + " |"
            else:
                row += " " .rjust(19) + " |"
        for mode in WRITE_MODES:
            v = table[key].get(mode, [])
            if v:
                rr = np.array([x[1] for x in v])
                row += f" {rr.mean():.3f}".rjust(19) + " |"
            else:
                row += " ".rjust(19) + " |"
        if "delta" in accs and "additive" in accs:
            row += f"{accs['delta'] - accs['additive']:>7.3f}"
        lines.append(row)
    lines.append("-" * len(header))
    lines.append("acc = streaming masked accuracy on the frozen eval (mean±std over seeds); "
                 "ret = acc / own knobs-off streaming acc (matched-clean normalization).")
    summary = "\n".join(lines)
    _save(os.path.join(out_dir, "summary.json"),
          {"table": {f"bits{b}_noise{s}": {m: table[(b, s)].get(m) for m in WRITE_MODES}
                     for (b, s) in sorted(table)},
           "meta": records["meta"], "generated_by": "seq/analog_probe.py"})
    return summary


def _pilot(steps=STEPS):
    """Grid-sizing gate: ONE delta cell must train to >0.8 solve in <5 min CPU (task order)."""
    torch.set_num_threads(THREADS)
    device = torch.device("cpu")
    t0 = time.time()
    r, _ = _train_one("delta", 0, steps, device)
    dt = time.time() - t0
    ok = r.best_acc > 0.8 and dt < 300
    print(f"[pilot] delta seed0: best_acc={r.best_acc:.3f} in {dt:.0f}s "
          f"({'PASS' if ok else 'FAIL'}: need >0.8 in <300s)")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()
    if args.pilot:
        _pilot(steps=args.steps or STEPS)
    else:
        out = args.out or (os.path.join(RESULTS_DIR, "_smoke") if args.smoke else RESULTS_DIR)
        seeds = (0,) if args.smoke else SEEDS
        steps = args.steps or (40 if args.smoke else STEPS)
        run(out_dir=out, seeds=seeds, steps=steps, smoke=args.smoke)
