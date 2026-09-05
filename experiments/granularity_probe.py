"""
Granularity exploratory selection study -- G1/G2 training-granularity levers
(target registry id PR-2026-09-03-07; fusion spec Addendum 2026-09-05 successor classes).

*** LANE-EXPLORATORY (docs/preregistry/POLICY.md): n=2 seeds (0, 1), toy scale, run to
    SELECT a configuration, never to claim one. The only future-claim vehicle is the
    PR-07 draft (docs/preregistry/2026-09-05-granularity-bar6.md, LANE-CLAIM, fresh
    seeds 20-24 -- disjoint from these). ***

WHAT RUNS
---------
The two exploratory interleaved streams of experiments/interleave_fix_probe.py (which
mirror experiments/expert_economy.py constructions verbatim; same generator, same
permutations, same per-sample exposure as the shipped E1 stream), at the shipped E1
settings (structured_permuted_tasks, K=5, d=24, 8 classes, h=48, M=K+3=8 experts,
epochs=15, batch=128, DFA, consolidate=True, z_novel=5.0), under four configs:

  shipped        -- every lever off (the PR-06 blocker: one expert trains on ~100% of
                    the stream -> monolithic ceiling ~0.60)
  g2_w32         -- session_window=32  (G2: contiguous 32-sample windows routed as a
                    unit, window-mean surprise vs the same calibrated thresholds)
  g2_w8          -- session_window=8   (G2, finer)
  g1_sample      -- train_granularity="sample" (G1: per-sample routed delta steps,
                    per-batch novel pool, recruitment iff novel fraction >= 0.5)

4 configs x 2 streams (mixed, roundrobin) x 2 seeds = 16 runs, plus the E1 block-stream
no-regression guard for the selected config (n=2).

FROZEN SELECTION RULE (written BEFORE running; POLICY.md discipline)
--------------------------------------------------------------------
Let ACC_mean(stream, cfg) = mean over seeds {0,1} of routed test accuracy. Let
MAXFRAC_mixed(cfg) = max over mixed runs and experts of (route_log[m] / sum(route_log))
-- the per-expert TRAINING-fraction ledger (the PR-06 lesson: check the training
ledger, not just the accuracy).

  1. SPECIALIZATION FILTER (hard, per PR-06): a config is eligible only if
     MAXFRAC_mixed(cfg) <= 0.85 (specialization must actually happen; a monolithic
     ledger is the diagnosed failure, whatever the accuracy).
  2. PASS-set := {cfg eligible AND ACC_mean('mixed') >= 0.70 AND
                  ACC_mean('roundrobin') >= 0.70}.
  3. If PASS-set non-empty: the selected config has the FEWEST active knobs (all four
     have <= 1, shipped has 0); ties -> higher mean E1 ACC (computed on demand for the
     tied candidates only, n=2); remaining ties -> the earlier PRECEDENCE entry.
  4. If PASS-set empty: report honestly; the reported best is the config maximizing
     min(ACC_mean over the two interleaved streams) with the same tie-breaks. The
     PR-07 draft fires only if the reported best has min interleaved ACC_mean >= 0.68
     on BOTH streams AND passes the specialization filter (a config that already fails
     the ledger filter at n=2 is not drafted with a specialization bar it cannot meet).
  5. PRECEDENCE (smallest config first): shipped < g2_w32 < g2_w8 < g1_sample
     (0 knobs, then coarser window = fewer decision-unit changes, then finest).

RUNTIME BUDGET: <= ~45 min CPU total; the first cell is timed and the projected total
printed (PR-06's identical-size claim grid ran 60 cells in 75 s, so ~16+2 cells is
minutes; the script aborts before the grid if the projection exceeds the budget).

OUTPUT
------
results/exploratory/granularity_probe_2026-09-05/probe.json (raw per-run records with
route_log ledgers + per-config summary + the applied selection trace) and RESULTS.md
written after the run. Raw artifacts are never deleted (docs/RETENTION.md); per
POLICY.md nothing here may be cited as a claim.

Usage: python experiments/granularity_probe.py [--seeds 0,1]
"""
from __future__ import annotations

import os
import sys
import json
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

import interleave_fix_probe as probe           # frozen stream constructors (reused)

DATE = "2026-09-05"
SEEDS = (0, 1)
BAR = 0.70                 # the BAR-6 bar (exploratory pass level)
DRAFT_FLOOR = 0.68         # min interleaved ACC_mean for the PR-07 draft branch
SPECIALIZATION_MAX = 0.85   # hard ledger filter (PR-06 lesson)
BUDGET_MIN = 45.0          # runtime budget, minutes

# PRECEDENCE order == frozen selection order (smallest config first)
CONFIGS = [
    ("shipped",   dict()),
    ("g2_w32",    dict(session_window=32)),
    ("g2_w8",     dict(session_window=8)),
    ("g1_sample", dict(train_granularity="sample")),
]

OUTDIR = os.path.abspath(os.path.join(_HERE, "..", "results", "exploratory",
                                      f"granularity_probe_{DATE}"))
RAWDIR = os.path.join(OUTDIR, "raw")


def _atomic_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def ledger_fractions(route_log):
    tot = max(sum(route_log), 1)
    return [round(c / tot, 4) for c in route_log]


def run_cell(name, cfg, stream, seed, fn):
    """One (config, stream, seed) run + crash-safe raw persistence."""
    t0 = time.time()
    rec = fn(seed, **cfg)
    dt = round(time.time() - t0, 2)
    if "route_log" in rec:                     # interleaved streams carry the ledger
        rec["fractions"] = ledger_fractions(rec["route_log"])
        rec["max_fraction"] = max(rec["fractions"])
    raw = {"config": name, "stream": stream, "seed": seed, "knobs": cfg,
           "runtime_seconds": dt, "result": rec}
    _atomic_json(os.path.join(RAWDIR, f"{name}__{stream}__seed{seed}.json"), raw)
    extra = (f" maxfrac={rec['max_fraction']:.3f}" if "max_fraction" in rec else "")
    print(f"  [{name} / {stream} / seed {seed}] acc={rec['acc']:.4f}{extra} "
          f"recruited={rec['recruited']} ({dt:.1f}s)", flush=True)
    return rec


def main(seeds=SEEDS):
    t0 = time.time()
    os.makedirs(RAWDIR, exist_ok=True)
    cells = {}
    todo = [(n, c, s, k, f) for n, c in CONFIGS
            for s, f in (("mixed", probe.run_mixed), ("roundrobin", probe.run_roundrobin))
            for k in seeds]
    print(f"[probe] {len(CONFIGS)} configs x 2 streams x {len(seeds)} seeds = "
          f"{len(todo)} runs; BAR={BAR}, draft floor={DRAFT_FLOOR}, "
          f"specialization max-frac<={SPECIALIZATION_MAX}", flush=True)

    # -- budget guard: time the first cell, project the total ------------------- #
    first = todo[0]
    key0 = (first[0], first[2], first[3])
    cells[key0] = run_cell(first[0], first[1], first[2], first[3], first[4])
    per = time.time() - t0
    proj_min = per * len(todo) / 60.0
    print(f"[probe] first run {per:.1f}s -> projected total {proj_min:.1f} min "
          f"(budget {BUDGET_MIN:.0f} min)", flush=True)
    if proj_min > BUDGET_MIN:
        raise SystemExit(f"[probe] ABORT: projected {proj_min:.1f} min exceeds the "
                         f"{BUDGET_MIN:.0f} min budget -- do not run the grid.")
    for (name, cfg, stream, seed, fn) in todo[1:]:
        cells[(name, stream, seed)] = run_cell(name, cfg, stream, seed, fn)

    # -- per-config summary ----------------------------------------------------- #
    summary = {}
    for name, cfg in CONFIGS:
        rec = {"knobs": cfg, "n_active_knobs": int(len(cfg)), "precedence": name}
        for stream in ("mixed", "roundrobin"):
            runs = [cells[(name, stream, k)] for k in seeds]
            rec[stream] = {
                "acc_mean": round(float(np.mean([r["acc"] for r in runs])), 4),
                "acc_per_seed": [round(r["acc"], 4) for r in runs],
                "max_fraction_per_seed": [r["max_fraction"] for r in runs],
                "fractions_per_seed": [r["fractions"] for r in runs],
                "route_log_per_seed": [r["route_log"] for r in runs],
                "recruited_per_seed": [r["recruited"] for r in runs],
            }
        rec["min_interleaved_acc"] = min(rec["mixed"]["acc_mean"],
                                         rec["roundrobin"]["acc_mean"])
        rec["maxfrac_mixed"] = max(rec["mixed"]["max_fraction_per_seed"])
        rec["specialization_ok"] = bool(rec["maxfrac_mixed"] <= SPECIALIZATION_MAX)
        rec["passes_bar"] = bool(rec["mixed"]["acc_mean"] >= BAR
                                 and rec["roundrobin"]["acc_mean"] >= BAR
                                 and rec["specialization_ok"])
        summary[name] = rec

    print("\n[probe] === summary (mixed / roundrobin mean ACC; max train-fraction on "
          "mixed) ===")
    for name, _ in CONFIGS:
        r = summary[name]
        print(f"  {name:<10} mixed={r['mixed']['acc_mean']:.3f} "
              f"rr={r['roundrobin']['acc_mean']:.3f} "
              f"maxfrac_mixed={r['maxfrac_mixed']:.3f} "
              f"{'PASS' if r['passes_bar'] else ''}", flush=True)

    # -- the frozen selection rule, applied mechanically ------------------------ #
    trace = {"bar": BAR, "draft_floor": DRAFT_FLOOR,
             "specialization_max": SPECIALIZATION_MAX}
    pass_set = [n for n, _ in CONFIGS if summary[n]["passes_bar"]]

    def attach_e1(names):
        for name in names:
            if "e1" in summary[name]:
                continue
            runs = [run_cell(name, dict(CONFIGS)[name], "e1", k, probe.run_e1_guard)
                    for k in seeds]
            summary[name]["e1"] = {
                "acc_mean": round(float(np.mean([r["acc"] for r in runs])), 4),
                "fgt_mean": round(float(np.mean([r["fgt"] for r in runs])), 4),
                "acc_per_seed": [round(r["acc"], 4) for r in runs],
                "fgt_per_seed": [round(r["fgt"], 4) for r in runs]}

    if pass_set:
        fewest = min(summary[n]["n_active_knobs"] for n in pass_set)
        tied = [n for n in pass_set if summary[n]["n_active_knobs"] == fewest]
        attach_e1(tied)
        chosen = max(tied, key=lambda n: (summary[n]["e1"]["acc_mean"],
                                          -CONFIGS.index((n, dict(CONFIGS)[n]))))
        trace["rule"] = ("PASS-set non-empty -> fewest active knobs, tie-break higher "
                         "E1 ACC, then precedence order")
        trace["pass_set"] = pass_set
    else:
        best_min = max(summary[n]["min_interleaved_acc"] for n, _ in CONFIGS)
        tied = [n for n, _ in CONFIGS
                if summary[n]["min_interleaved_acc"] == best_min]
        fewest = min(summary[n]["n_active_knobs"] for n in tied)
        tied = [n for n in tied if summary[n]["n_active_knobs"] == fewest]
        attach_e1(tied)
        chosen = max(tied, key=lambda n: (summary[n]["e1"]["acc_mean"],
                                          -CONFIGS.index((n, dict(CONFIGS)[n]))))
        trace["rule"] = ("PASS-set EMPTY -> best-weakest interleaved stream, then "
                         "fewest knobs, then higher E1 ACC, then precedence")
        trace["pass_set"] = []

    attach_e1([chosen])
    ch = summary[chosen]
    trace["chosen"] = chosen
    draft = bool(ch["mixed"]["acc_mean"] >= DRAFT_FLOOR
                 and ch["roundrobin"]["acc_mean"] >= DRAFT_FLOOR
                 and ch["specialization_ok"])
    trace["pr07_draft_branch"] = (
        "DRAFT PR-2026-09-03-07" if draft else
        "NO PR DRAFT (best below the 0.68 floor on some stream, or the specialization "
        "filter fails) -- report honest numbers + ledger + mechanistic read")

    out = {
        "label": "LANE-EXPLORATORY",
        "date": DATE,
        "target_prereg": "docs/preregistry/2026-09-05-granularity-bar6.md "
                         "(PR-2026-09-03-07, IN-WRITE, seeds 20-24 = disjoint)",
        "bar_exploratory": BAR,
        "settings": {"K": 5, "d": 24, "n_classes": 8, "h": 48, "epochs": 15,
                     "batch": 128, "n_experts": 8, "z_novel": 5.0,
                     "seeds": list(seeds),
                     "stream_note": ("mixed/roundrobin constructors reused verbatim "
                                     "from experiments/interleave_fix_probe.py; "
                                     "E1 guard = shipped block benchmark")},
        "selection_rule_frozen": (
            "Specialization filter (hard, PR-06 lesson): eligible iff max per-expert "
            "train-fraction over mixed runs <= 0.85. PASS-set = eligible AND mixed >= "
            "0.70 AND roundrobin >= 0.70 at n=2; choose fewest active knobs, tie-break "
            "higher E1 ACC (on demand for tied candidates), then precedence "
            "[shipped < g2_w32 < g2_w8 < g1_sample]. If empty: best-weakest min "
            "interleaved ACC with the same tie-breaks; PR-07 drafted only if the "
            "reported best is >= 0.68 on BOTH streams AND passes the ledger filter."),
        "selection_trace": trace,
        "configs": summary,
        "chosen": chosen,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    outp = os.path.join(OUTDIR, "probe.json")
    _atomic_json(outp, out)

    print(f"\n[probe] CHOSEN: {chosen}")
    print(f"[probe]   mixed={ch['mixed']['acc_mean']:.3f} "
          f"rr={ch['roundrobin']['acc_mean']:.3f} "
          f"maxfrac_mixed={ch['maxfrac_mixed']:.3f}")
    print(f"[probe]   E1 guard: ACC={ch['e1']['acc_mean']:.3f} "
          f"FGT={ch['e1']['fgt_mean']:.3f} (n={len(seeds)})")
    print(f"[probe]   BRANCH: {trace['pr07_draft_branch']}")
    print(f"[probe] saved -> {os.path.normpath(outp)}  ({time.time() - t0:.1f}s total)")
    return out


if __name__ == "__main__":
    seed_args = [a for a in sys.argv[1:] if a.startswith("--seeds=")]
    seeds = tuple(int(x) for x in seed_args[0].split("=")[1].split(",")) \
        if seed_args else SEEDS
    main(seeds=seeds)
