"""
Claim-grade runner for registry id PR-2026-09-03-04 — delta-vs-additive analog robustness of
the carried state (state quantization + write noise), MixedMQAR. Executes the REGISTERED
protocol VERBATIM from results/analog_probe_2026-09-03/RESULTS.md §7 (the binding document;
status IN-WRITE at run time — the INDEX row is frozen by the maintainer from THESE results).

Frozen protocol (§7):
  * Identical model/task/training skeleton to the exploratory §2 (MixedMQAR D=32 vocab=128,
    PrizmaSeqLM d64L2H2 feat_map='none', AdamW 3000 steps batch 64 warmup 300 cosine->0.1x,
    build_and_train seeding), BUT with a per-arm LR sweep over seq.lrsweep.DEFAULT_GRID
    (LR-fairness: no arm is denied an LR another gets), then all claim seeds at the chosen LR.
  * Matched-clean gate (BINDING): an arm enters the comparison only at a (difficulty, lr) where
    its clean streaming accuracy is >= 0.80 on >= 4/5 seeds. If an arm cannot reach the gate at
    D=32, the difficulty drops (D=24, then D=16) IDENTICALLY for both arms until the gate passes
    for both — never per-arm.
  * Conditions: primaries {4-bit, sigma=0} and {FP32, sigma=0.05}; 6/8-bit and sigma=0.01
    secondaries. n = 5 seeds per arm (seeds 0-4), NO substitution. Train-FP32 / deploy-degraded
    through the post-fix step() path only. O(1)-sanity cell (|stream(bits=0,sigma=0) - forward|
    <= 0.005) required PER MODEL, else the run is invalid.
  * Decision rule (frozen): H2a SURVIVES iff, on BOTH primaries, delta retention exceeds
    additive retention by >= +0.05 with one-sided Welch p < 0.05 across the 5v5 seed pairs
    (Holm-corrected over the 2 primaries), AND the O(1)-sanity gate passes for every included
    model. Otherwise H2a is recorded NOT-SUPPORTED at this scale.
  * Retention = acc(condition) / acc(own knobs-off streaming), per seed (matched-clean
    normalization), compared across seed pairs.

Raw-first / crash-safe: every sweep and every claim run is persisted as its OWN JSON file
(json -> .tmp -> os.replace) BEFORE anything else happens; existing files are never overwritten
(resume = re-run the script, completed units are skipped). LANE-CLAIM discipline: no seed
reduction, no bar change, no re-tuning after results are seen.

Usage:
    python -m seq.analog_probe_claim --smoke          # tiny plumbing check -> <out>/_smoke/
    python -m seq.analog_probe_claim                  # the registered run
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from .analog_probe import (_model_factory, _reset_knobs, _save, _stream_eval,
                           BATCH_SIZE, EVAL_BATCHES, EVAL_SEED, FEAT_MAP, SCALE_KW, STEPS,
                           THREADS)
from .lrsweep import DEFAULT_GRID, sweep_lr
from .stats import _welch, holm_correction, superiority_test, t_isf

REGISTRY_ID = "PR-2026-09-03-04"
RESULTS_DIR = os.path.join("results", f"analog_probe_{REGISTRY_ID}")
PROTOCOL_SRC = "results/analog_probe_2026-09-03/RESULTS.md#7"

# ---- frozen claim protocol (§7; identical skeleton constants imported from analog_probe) ----
DIFFICULTIES = (32, 24, 16)          # matched-clean gate drop ladder: D=32 -> 24 -> 16, both arms
SEEDS = (0, 1, 2, 3, 4)              # n=5 claim seeds per arm, NO substitution
SWEEP_SEED = 0                       # stage-1 convention (lrsweep docstring): full grid @ 1 seed
GATE_CLEAN_ACC = 0.80                # matched-clean floor: clean streaming acc >= 0.80 ...
GATE_MIN_SEEDS = 4                   # ... on >= 4/5 seeds
SANITY_TOL = 0.005                   # O(1)-sanity: |stream(bits=0, s=0) - forward| <= 0.005
DELTA_BAR = 0.05                     # retention-points bar on BOTH primaries
ALPHA = 0.05                         # one-sided Welch, Holm over the 2 primaries
ARMS = ("delta", "additive")

CLEAN_CELL = (0, 0.0)                                   # knobs-off: ret=1 + O(1)-sanity cell
PRIMARY_CELLS = ((4, 0.0), (0, 0.05))                   # §7's two registered primaries
SECONDARY_CELLS = ((6, 0.0), (8, 0.0),                  # §7: 6/8-bit and sigma=0.01 secondaries
                   (0, 0.01), (4, 0.01), (6, 0.01), (8, 0.01))
EXTRA_CELLS = ((4, 0.05), (6, 0.05), (8, 0.05))         # exploratory-grid carryover: DESCRIPTIVE
                                                        # only (not registered, not claim-bearing)
CELL_ORDER = (CLEAN_CELL,) + PRIMARY_CELLS + SECONDARY_CELLS + EXTRA_CELLS


def _task(d):
    from .tasks import MixedMQAR
    return MixedMQAR(vocab=128, max_pairs=d, num_queries=64, gap=0, min_pairs=1)


def _zero_arg_fac(mode, task):
    """build_and_train calls model_fac() with NO args; analog_probe's factory needs
    (vocab, max_len) — bind them (same construction, same seeding recipe)."""
    base = _model_factory(mode)
    return lambda: base(task.vocab, task.seq_len)


def _train_cfg(lr, steps=STEPS):
    from .common import TrainConfig
    return TrainConfig(steps=steps, batch_size=BATCH_SIZE, lr=lr, eval_every=500,
                       warmup=300, warmup_frac=0.0, min_lr_frac=0.1, log=False)


# --------------------------------------------------------------------------- persistence ----
def _run_path(out_dir, d, mode, seed):
    return os.path.join(out_dir, "runs", f"run_d{d}_{mode}_seed{seed}.json")


def _sweep_path(out_dir, d, mode):
    return os.path.join(out_dir, f"sweep_d{d}_{mode}.json")


def _load(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


# ------------------------------------------------------------------------------- runners ----
def _ensure_sweep(out_dir, d, mode, device, grid, steps):
    """Stage-1 per-arm LR sweep (full grid @ SWEEP_SEED, best on the frozen eval; the FULL grid
    incl. rejected LRs is recorded for the audit trail — lrsweep guardrail #10)."""
    path = _sweep_path(out_dir, d, mode)
    cached = _load(path)
    if cached is not None:
        print(f"[claim] sweep cached {mode} d{d} -> lr={cached['best_lr']}", flush=True)
        return cached
    task = _task(d)
    t0 = time.time()
    res = sweep_lr(_zero_arg_fac(mode, task), task, _train_cfg(grid[0], steps), device,
                   grid=grid, seed=SWEEP_SEED)
    rec = dict(registry_id=REGISTRY_ID, difficulty=d, write_mode=mode, seed=SWEEP_SEED,
               grid=list(grid), best_lr=res["best_lr"], best_acc=res["best_acc"],
               grid_rows=res["grid"], seconds=time.time() - t0, steps=steps)
    _save(path, rec)                    # raw-first: persisted before any use
    print(f"[claim] sweep {mode} d{d}: best_lr={res['best_lr']} best_acc={res['best_acc']:.3f} "
          f"({rec['seconds']:.0f}s) rows={[(r['lr'], round(r['best_acc'], 3)) for r in res['grid']]}",
          flush=True)
    return rec


@torch.no_grad()
def _eval_cells(model, frozen):
    """All registered cells in CELL_ORDER (primaries FIRST — the partial-order disclosure for a
    crash mid-run), plus the parallel-forward reference for the O(1)-sanity cell."""
    from .common import masked_acc
    cells = []
    for b, s in CELL_ORDER:
        acc, ce = _stream_eval(model, frozen, b, s)
        cells.append(dict(state_bits=b, write_noise_std=s, stream_acc=acc, stream_ce=ce,
                          primary=(b, s) in PRIMARY_CELLS))
    _reset_knobs(model)
    model.train(False)
    fwd_acc = float(np.mean([masked_acc(model(x), y, m) for x, y, m in frozen]))
    return cells, fwd_acc


def _claim_run(out_dir, d, mode, seed, lr, device, steps, eval_batches):
    """One claim model: train-FP32 at the arm's chosen LR, then deploy-degraded streaming eval.
    Raw record saved BEFORE returning (crash-safe, never overwritten)."""
    from .common import set_seed, train_model
    path = _run_path(out_dir, d, mode, seed)
    cached = _load(path)
    if cached is not None:
        print(f"[claim] run cached {mode} d{d} seed{seed} (clean={cached['clean_stream_acc']:.3f})",
              flush=True)
        return cached
    t0 = time.time()
    task = _task(d)
    cfg = _train_cfg(lr, steps)
    set_seed(seed)                                    # build_and_train's exact recipe:
    model = _model_factory(mode)(task.vocab, task.seq_len)   # seed BEFORE construction
    r = train_model(model, task, cfg, device, seed=seed)
    frozen = _frozen_eval(task, eval_batches, device)
    cells, fwd_acc = _eval_cells(model, frozen)
    clean = next(c["stream_acc"] for c in cells
                 if (c["state_bits"], c["write_noise_std"]) == CLEAN_CELL)
    sanity = abs(clean - fwd_acc)
    rec = dict(registry_id=REGISTRY_ID, difficulty=d, write_mode=mode, seed=seed, lr=lr,
               train=dict(best_acc=r.best_acc, final_acc=r.final_acc, final_loss=r.final_loss,
                          seconds=r.seconds, steps_to_plateau=r.steps_to_plateau,
                          params=r.params, history=r.history),
               forward_acc=fwd_acc, clean_stream_acc=clean, sanity_abs_diff=sanity,
               sanity_pass=bool(sanity <= SANITY_TOL),
               cells=cells, wall_seconds=time.time() - t0, steps=steps)
    _save(path, rec)                  # raw-first: persisted before any verdict
    print(f"[claim] run {mode} d{d} seed{seed}: clean={clean:.3f} train_best={r.best_acc:.3f} "
          f"sanity={sanity:.4f} {rec['wall_seconds']:.0f}s", flush=True)
    return rec


def _frozen_eval(task, batches, device):
    from .analog_probe import _frozen_eval as _fe
    return _fe(task, batches, device)


# --------------------------------------------------------------------- gate + statistics ----
def _arm_gate(runs):
    """Matched-clean gate (BINDING): clean streaming acc >= 0.80 on >= 4/5 seeds, AND the
    O(1)-sanity cell passes for every included model."""
    n_pass = sum(1 for r in runs if r["clean_stream_acc"] >= GATE_CLEAN_ACC)
    all_sane = all(r["sanity_pass"] for r in runs)
    return dict(n_seeds=len(runs), n_clean_ge_080=n_pass, all_sane=bool(all_sane),
                passes=bool(len(runs) == len(SEEDS) and n_pass >= GATE_MIN_SEEDS and all_sane),
                clean_accs=[r["clean_stream_acc"] for r in runs],
                forward_accs=[r["forward_acc"] for r in runs],
                sanity_diffs=[r["sanity_abs_diff"] for r in runs])


def _retentions(runs, cell):
    """ret = acc(condition)/acc(own knobs-off streaming), per seed — matched-clean normalization."""
    out = []
    for r in runs:
        acc = next(c["stream_acc"] for c in r["cells"]
                   if (c["state_bits"], c["write_noise_std"]) == cell)
        out.append(acc / r["clean_stream_acc"])
    return out


def _compare(ret_delta, ret_additive):
    """Δret with two-sided 95% Welch CI + one-sided Welch p (H1: delta > additive)."""
    sup = superiority_test(ret_delta, ret_additive)
    _, df, se = _welch(ret_delta, ret_additive)
    d = float(np.mean(ret_delta) - np.mean(ret_additive))
    h = t_isf(0.025, df) * se                       # two-sided 95%: p=0.025 in each tail
    return dict(delta_ret=d, ci95=[d - h, d + h], se=se, welch_t=sup["t"], welch_df=df,
                p_one_sided=sup["p_value"])


def _verdict(runs_by_arm):
    """Frozen decision rule over the 2 primaries (Holm-corrected), secondaries descriptive."""
    per_cell = {}
    for cell in PRIMARY_CELLS + SECONDARY_CELLS + EXTRA_CELLS:
        rd = _retentions(runs_by_arm["delta"], cell)
        ra = _retentions(runs_by_arm["additive"], cell)
        per_cell[f"bits{cell[0]}_noise{cell[1]}"] = dict(
            cell=dict(state_bits=cell[0], write_noise_std=cell[1],
                      primary=cell in PRIMARY_CELLS, secondary=cell in SECONDARY_CELLS),
            delta=dict(rets=rd, mean=float(np.mean(rd)),
                       sd=float(np.std(rd, ddof=1)) if len(rd) > 1 else 0.0),
            additive=dict(rets=ra, mean=float(np.mean(ra)),
                          sd=float(np.std(ra, ddof=1)) if len(ra) > 1 else 0.0),
            **_compare(rd, ra))
    prim = [per_cell[f"bits{b}_noise{s}"] for b, s in PRIMARY_CELLS]
    holm = holm_correction([c["p_one_sided"] for c in prim], alpha=ALPHA)
    for c, h in zip(prim, holm):
        c["p_holm_adj"] = h["p_adj"]
        c["holm_reject"] = h["reject"]
        c["meets_bar"] = bool(c["delta_ret"] >= DELTA_BAR and c["p_one_sided"] < ALPHA
                              and h["reject"])
    sanity_all = all(r["sanity_pass"] for m in ARMS for r in runs_by_arm[m])
    gate_note = None
    survives = bool(all(c["meets_bar"] for c in prim) and sanity_all)
    if not survives:
        failed = [f"bits{b}_noise{s}" for (b, s), c in zip(PRIMARY_CELLS, prim) if not c["meets_bar"]]
        gate_note = dict(failed_primaries=failed, sanity_all_pass=bool(sanity_all))
    return dict(survives=survives, gate_note=gate_note, primaries=prim,
                per_cell=per_cell, sanity_all_pass=bool(sanity_all))


# ---------------------------------------------------------------------------------- main ----
def run(out_dir=RESULTS_DIR, seeds=SEEDS, steps=STEPS, difficulties=DIFFICULTIES, grid=DEFAULT_GRID,
        eval_batches=EVAL_BATCHES, smoke=False):
    torch.set_num_threads(THREADS)
    device = torch.device("cpu")        # zero-GPU program; deterministic CPU
    t0 = time.time()
    runs_dir = os.path.join(out_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    _save(os.path.join(out_dir, "meta.json"), dict(
        registry_id=REGISTRY_ID, protocol=PROTOCOL_SRC, date=time.strftime("%Y-%m-%d %H:%M:%S"),
        task=dict(vocab=128, num_queries=64, gap=0, min_pairs=1, difficulties=list(difficulties)),
        scale=SCALE_KW, feat_map=FEAT_MAP, steps=steps, batch_size=BATCH_SIZE,
        lr_grid=list(grid), sweep_seed=SWEEP_SEED, seeds=list(seeds), eval_batches=eval_batches,
        eval_seed=EVAL_SEED, gate=dict(clean_acc=GATE_CLEAN_ACC, min_seeds=GATE_MIN_SEEDS),
        sanity_tol=SANITY_TOL, delta_bar=DELTA_BAR, alpha=ALPHA, primaries=list(PRIMARY_CELLS),
        secondaries=list(SECONDARY_CELLS), extras_descriptive_only=list(EXTRA_CELLS),
        cell_order=[list(c) for c in CELL_ORDER], threads=THREADS, device=str(device),
        smoke=smoke))
    print(f"[claim] {REGISTRY_ID} device={device} threads={THREADS} out={out_dir}", flush=True)
    print(f"[claim] protocol: per-arm LR sweep {grid} @seed{SWEEP_SEED} -> matched-clean gate "
          f"(>={GATE_CLEAN_ACC} on >={GATE_MIN_SEEDS}/{len(seeds)}) -> n={len(seeds)} seeds/arm; "
          f"primaries {PRIMARY_CELLS}; bar +{DELTA_BAR} one-sided Welch p<{ALPHA} Holm(2)",
          flush=True)

    planned = 2 * len(grid) + 2 * len(seeds)     # per difficulty: 2 sweeps + 2 arms x n seeds
    first_done = False
    decision = None
    gate_history = []
    for d in difficulties:
        print(f"[claim] === difficulty D={d} ===", flush=True)
        sweeps, runs_by_arm = {}, {}
        for mode in ARMS:
            sweeps[mode] = _ensure_sweep(out_dir, d, mode, device, grid, steps)
        if not first_done:
            per_run = sweeps[ARMS[0]]["seconds"] / len(grid)   # first-cell wall time (sweep unit)
            print(f"[claim] FIRST-CELL WALL TIME: {sweeps[ARMS[0]]['seconds']:.0f}s for the first "
                  f"sweep (~{per_run:.0f}s/run) -> projected ~{per_run * planned / 60:.0f} min for "
                  f"the {planned} planned units at D={d} (train only, excl. streaming eval)",
                  flush=True)
            first_done = True
        for mode in ARMS:
            runs_by_arm[mode] = [_claim_run(out_dir, d, mode, s, sweeps[mode]["best_lr"], device,
                                            steps, eval_batches) for s in seeds]
        gates = {m: _arm_gate(runs_by_arm[m]) for m in ARMS}
        for m in ARMS:
            print(f"[claim] GATE {m} d{d}: clean>={GATE_CLEAN_ACC} on {gates[m]['n_clean_ge_080']}"
                  f"/{len(seeds)} seeds, sanity_all={gates[m]['all_sane']} -> "
                  f"{'PASS' if gates[m]['passes'] else 'FAIL'} "
                  f"clean={ [round(c, 3) for c in gates[m]['clean_accs']] }", flush=True)
        both = all(g["passes"] for g in gates.values())
        verdict = _verdict(runs_by_arm) if both else None
        gate_history.append(dict(difficulty=d, gates=gates, passed=both))
        decision = dict(difficulty=d, sweeps={m: sweeps[m] for m in ARMS}, gates=gates,
                        verdict=verdict, gate_passed=both)
        _save(os.path.join(out_dir, f"decision_d{d}.json"),
              dict(registry_id=REGISTRY_ID, **{k: v for k, v in decision.items() if k != "verdict"},
                   verdict=verdict))
        if both:
            break
        print(f"[claim] matched-clean gate FAILED at D={d} for "
              f"{[m for m in ARMS if not gates[m]['passes']]} -> difficulty drops identically "
              f"for BOTH arms (§7 ladder {difficulties})", flush=True)

    if decision is None:
        raise RuntimeError("no difficulty executed")
    if not decision["gate_passed"]:
        final = dict(registry_id=REGISTRY_ID, verdict="NOT-SUPPORTED",
                     reason="matched-clean gate never passed at any registered difficulty "
                            f"{list(difficulties)}; comparison not runnable at the registered "
                            "scale (frozen rule: gate is binding, no per-arm exceptions)",
                     difficulties_tried=list(difficulties),
                     gate_history=gate_history, smoke=smoke)
        _save(os.path.join(out_dir, "verdict.json"), final)
        print(f"[claim] FINAL: gate never passed -> {final['verdict']}", flush=True)
        return final

    v = decision["verdict"]
    table = {}
    for key, c in v["per_cell"].items():
        table[key] = dict(cell=c["cell"],
                          delta=dict(rets=c["delta"]["rets"], mean=c["delta"]["mean"],
                                     sd=c["delta"]["sd"]),
                          additive=dict(rets=c["additive"]["rets"], mean=c["additive"]["mean"],
                                        sd=c["additive"]["sd"]),
                          delta_ret=c["delta_ret"], ci95=c["ci95"],
                          p_one_sided=c.get("p_one_sided"), p_holm_adj=c.get("p_holm_adj"),
                          meets_bar=c.get("meets_bar"))
    primaries = [dict(state_bits=p["cell"]["state_bits"],
                      write_noise_std=p["cell"]["write_noise_std"],
                      delta_ret=p["delta_ret"], ci95=p["ci95"], p_one_sided=p["p_one_sided"],
                      p_holm_adj=p["p_holm_adj"], meets_bar=p["meets_bar"]) for p in v["primaries"]]
    final = dict(registry_id=REGISTRY_ID,
                 verdict=("H2a-SURVIVES" if v["survives"] else "NOT-SUPPORTED"),
                 difficulty=decision["difficulty"], sweeps=decision["sweeps"],
                 gates=decision["gates"], primaries=primaries, table=table,
                 sanity_all_pass=v["sanity_all_pass"],
                 clean_delta=[r["clean_stream_acc"] for r in runs_by_arm["delta"]],
                 clean_additive=[r["clean_stream_acc"] for r in runs_by_arm["additive"]],
                 elapsed_s=time.time() - t0, smoke=smoke)
    _save(os.path.join(out_dir, "verdict.json"), final)
    _save(os.path.join(out_dir, "summary.json"),
          dict(registry_id=REGISTRY_ID, verdict=final["verdict"],
               difficulty=final["difficulty"], table=table, gates=decision["gates"]))
    print(f"[claim] FINAL VERDICT: {final['verdict']} at D={final['difficulty']}", flush=True)
    for p in v["primaries"]:
        print(f"[claim]   primary bits{p['cell']['state_bits']} noise{p['cell']['write_noise_std']}: "
              f"dRet={p['delta_ret']:+.3f} CI[{p['ci95'][0]:+.3f},{p['ci95'][1]:+.3f}] "
              f"p1={p['p_one_sided']:.4f} pHolm={p['p_holm_adj']:.4f} bar={p['meets_bar']}",
              flush=True)
    return final


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--eval-batches", type=int, default=None)
    ap.add_argument("--grid", type=float, nargs="*", default=None)
    ap.add_argument("--difficulties", type=int, nargs="*", default=None)
    args = ap.parse_args()
    out = args.out or (os.path.join(RESULTS_DIR, "_smoke") if args.smoke else RESULTS_DIR)
    kw = dict(out_dir=out)
    if args.smoke:
        kw.update(seeds=(0,), steps=args.steps or 30, grid=(5e-4, 2e-3), eval_batches=2)
    else:
        if args.steps:
            kw["steps"] = args.steps
        if args.seeds:
            kw["seeds"] = tuple(args.seeds)
        if args.eval_batches:
            kw["eval_batches"] = args.eval_batches
        if args.grid:
            kw["grid"] = tuple(args.grid)
        if args.difficulties:
            kw["difficulties"] = tuple(args.difficulties)
    run(**kw)
