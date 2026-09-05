"""
BAR-6 exploratory selection study -- interleaved-router fix levers (PR-2026-09-03-06).

*** LANE-EXPLORATORY (docs/preregistry/POLICY.md): n=2 seeds (0, 1), toy scale, run to
    SELECT a configuration, never to claim one. The only future-claim vehicle is the
    PR-06 draft (docs/preregistry/2026-09-03-interleaved-router-bar6.md, LANE-CLAIM,
    seeds 10-14 -- disjoint from these). ***

WHAT RUNS
---------
The two exploratory interleaved streams of expert_economy.py (same generator, same
permutations, same per-sample exposure as the shipped E1 stream; only presentation
changes), at the shipped E1 settings (structured_permuted_tasks, K=5, d=24, 8 classes,
h=48, M=K+3=8 experts, epochs=15, DFA, consolidate=True, z_novel=5.0):

  mixed       -- run_interleaved_stream semantics: samples of all K domains shuffled
                 into one stream (batch-mean surprise stationary -> vigilance blind).
  roundrobin  -- run_roundrobin_stream semantics: domain-pure batches, batch order
                 shuffled (immature floors freeze -> catch-all recognizers).

Grid (full factorial, 2^5 = 32 configs x 2 streams x 2 seeds = 128 runs; feasible on
CPU in minutes -- expert_economy.py ran the same-size streams at ~1 s/run):

  {shipped} x freeze_min_seen {0, 300} x dynamic_vigilance {0, 0.5}
            x hot_young {0, 1.0} x route_stat {batch_mean, sample_top}
            x n_settle_steps {0, 2}        (the shipped knob; swept, not modified)

FROZEN SELECTION RULE (written BEFORE running; POLICY.md discipline)
--------------------------------------------------------------------
Let ACC_mean(stream, cfg) be the mean over seeds 0,1 of the routed test accuracy
(mean over the K domains of per-sample routed-head accuracy, as in run_continual).

  1. PASS-set := {cfg : ACC_mean('mixed', cfg) >= 0.70 AND ACC_mean('roundrobin', cfg) >= 0.70}.
  2. If PASS-set is non-empty: choose the cfg with the FEWEST active levers (a lever is
     active iff its value differs from shipped: freeze_min_seen!=0, dynamic_vigilance!=0,
     hot_young!=0, route_stat!='batch_mean', n_settle_steps!=0). Ties: higher mean ACC on
     the shipped E1 BLOCK stream (no-regression direction; evaluated on demand for the
     tied candidates only, n=2, same seeds). Remaining ties: lexicographic.
  3. If PASS-set is EMPTY: choose the cfg maximizing min(ACC_mean over the two
     interleaved streams) -- the best-weakest choice -- with the same tie-breaks, and
     PR-06 is DRAFTED WITH THE HONEST 0.70 BAR ANYWAY: the claim run decides; this
     exploration only picks the arm.
  4. The chosen cfg (or the no-pass winner) is then run on the shipped E1 block stream
     (n=2) as the no-regression guard row.

OUTPUT
------
results/exploratory/interleave_fix_probe_2026-09-05/probe.json (raw per-run records +
per-config summary + the applied selection trace). Per docs/RETENTION.md raw artifacts
are never deleted; per POLICY.md nothing here may be cited as a claim.

Usage: python experiments/interleave_fix_probe.py [--seeds 0,1]
"""
from __future__ import annotations

import os
import sys
import json
import time
import itertools

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

from data import structured_permuted_tasks
from metrics import AccuracyMatrix, accuracy
from prizma import Prizma
from expert_economy import InstrumentedPrizma, eval_routing

DATE = "2026-09-05"
SEEDS = (0, 1)
K, D, NCLS, H, EPOCHS, BATCH = 5, 24, 8, 48, 15, 128
BAR = 0.70

# The grid. `shipped` is the all-off row (present by construction: all values at off).
GRID = dict(
    freeze_min_seen=[0, 300],
    dynamic_vigilance=[0.0, 0.5],
    hot_young=[0.0, 1.0],
    route_stat=["batch_mean", "sample_top"],
    n_settle_steps=[0, 2],
)


def make_prizma(seed, **cfg):
    """InstrumentedPrizma at the shipped E1 settings + the lever config."""
    return InstrumentedPrizma(d=D, h=H, K=NCLS, n_experts=K + 3, seed=seed,
                              consolidate=True, feedback="random", z_novel=5.0,
                              **cfg)


def routed_acc(p, tasks):
    """Mean over domains of per-sample routed-head accuracy (predict_logits routes
    per sample to the lowest-surprise trained expert -- the shipped inference path)."""
    return float(np.mean([accuracy(p.predict_logits(t.Xte), t.yte) for t in tasks]))


# --------------------------------------------------------------------------- #
# The two interleaved streams (mirror expert_economy.py constructions exactly) #
# --------------------------------------------------------------------------- #
def run_mixed(seed, **cfg):
    tasks = structured_permuted_tasks(n_tasks=K, d=D, n_classes=NCLS, seed=seed)
    Xall = np.vstack([t.Xtr for t in tasks])
    yall = np.concatenate([t.ytr for t in tasks])
    perm = np.random.default_rng(1000 + seed).permutation(len(Xall))
    p = make_prizma(seed, **cfg)
    p._task_label = "interleaved(EXPLORATORY)"
    rng = np.random.default_rng(seed)
    p.fit_task(Xall[perm], yall[perm], epochs=EPOCHS, rng=rng)
    acc = routed_acc(p, tasks)
    ev = eval_routing(p, [(j, t.Xte) for j, t in enumerate(tasks)])
    return {
        "acc": acc,
        "recruited": int(sum(e.n_seen > 0 for e in p.experts)),
        "route_log": p.route_log.tolist(),
        "dropped_batches": p.ledger["n_dropped_batches"],
        "purity": [round(d["purity"], 4) for d in ev["domains"]],
    }


def run_roundrobin(seed, **cfg):
    tasks = structured_permuted_tasks(n_tasks=K, d=D, n_classes=NCLS, seed=seed)
    p = make_prizma(seed, **cfg)
    p._task_label = "roundrobin(EXPLORATORY)"
    rng = np.random.default_rng(seed)
    for _ in range(EPOCHS):
        batches = []
        for t in tasks:
            Yall = np.eye(p.K, dtype=np.float32)[t.ytr]
            idx = rng.permutation(len(t.Xtr))
            for s in range(0, len(idx), BATCH):
                bi = idx[s:s + BATCH]
                batches.append((t.Xtr[bi], Yall[bi], t.ytr[bi]))
        for b in rng.permutation(len(batches)):
            Xb, Yb, yb = batches[int(b)]
            p.train_batch(Xb, Yb, yb)
    acc = routed_acc(p, tasks)
    ev = eval_routing(p, [(j, t.Xte) for j, t in enumerate(tasks)])
    return {
        "acc": acc,
        "recruited": int(sum(e.n_seen > 0 for e in p.experts)),
        "route_log": p.route_log.tolist(),
        "dropped_batches": p.ledger["n_dropped_batches"],
        "purity": [round(d["purity"], 4) for d in ev["domains"]],
    }


def run_e1_guard(seed, **cfg):
    """Shipped E1 block stream (the shipped benchmark; the no-regression guard)."""
    tasks = structured_permuted_tasks(n_tasks=K, d=D, n_classes=NCLS, seed=seed)
    p = make_prizma(seed, **cfg)
    rng = np.random.default_rng(seed)
    R = AccuracyMatrix(len(tasks))
    for i, t in enumerate(tasks):
        p.fit_task(t.Xtr, t.ytr, epochs=EPOCHS, rng=rng)
        for j, tt in enumerate(tasks):
            R.record(i, j, accuracy(p.predict_logits(tt.Xte), tt.yte))
    return {"acc": R.acc(), "fgt": R.forgetting(),
            "recruited": int(sum(e.n_seen > 0 for e in p.experts))}


# --------------------------------------------------------------------------- #
# Selection machinery (the frozen rule, mechanically applied)                  #
# --------------------------------------------------------------------------- #
def n_active_levers(cfg):
    return int(sum((cfg["freeze_min_seen"] != 0, cfg["dynamic_vigilance"] != 0.0,
                    cfg["hot_young"] != 0.0, cfg["route_stat"] != "batch_mean",
                    cfg["n_settle_steps"] != 0)))


def cfg_label(cfg):
    if n_active_levers(cfg) == 0:
        return "shipped"
    active = []
    if cfg["freeze_min_seen"]:
        active.append(f"veto{cfg['freeze_min_seen']}")
    if cfg["dynamic_vigilance"]:
        active.append(f"dv{cfg['dynamic_vigilance']}")
    if cfg["hot_young"]:
        active.append(f"hot{cfg['hot_young']}")
    if cfg["route_stat"] != "batch_mean":
        active.append(cfg["route_stat"])
    if cfg["n_settle_steps"]:
        active.append(f"settle{cfg['n_settle_steps']}")
    return "+".join(active)


def main(seeds=SEEDS):
    t0 = time.time()
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    print(f"[probe] {len(combos)} configs x {len(seeds)} seeds x 2 interleaved streams"
          f" = {len(combos) * len(seeds) * 2} runs (+ E1 guards); BAR={BAR}")

    runs, table = [], []
    for i, cfg in enumerate(combos):
        rec = {"config": cfg, "label": cfg_label(cfg),
               "n_active_levers": n_active_levers(cfg)}
        for stream, fn in (("mixed", run_mixed), ("roundrobin", run_roundrobin)):
            per_seed = [fn(s, **cfg) for s in seeds]
            rec[stream] = {"acc_mean": float(np.mean([r["acc"] for r in per_seed])),
                           "acc_per_seed": [round(r["acc"], 4) for r in per_seed],
                           "runs": per_seed}
        rec["min_interleaved_acc"] = min(rec["mixed"]["acc_mean"],
                                         rec["roundrobin"]["acc_mean"])
        rec["passes_bar"] = (rec["mixed"]["acc_mean"] >= BAR
                             and rec["roundrobin"]["acc_mean"] >= BAR)
        runs.append(rec)
        table.append(rec)
        print(f"[{i + 1:02d}/{len(combos)}] {rec['label']:<34} "
              f"mixed={rec['mixed']['acc_mean']:.3f} "
              f"rr={rec['roundrobin']['acc_mean']:.3f} "
              f"{'PASS' if rec['passes_bar'] else ''}", flush=True)

    pass_set = [r for r in runs if r["passes_bar"]]
    trace = {"bar": BAR, "n_pass": len(pass_set)}

    def e1_acc_for(recs):
        """Tie-break statistic: mean E1 ACC over seeds, computed on demand."""
        if "e1_acc_mean" not in recs[0]:
            for r in recs:
                per_seed = [run_e1_guard(s, **r["config"]) for s in seeds]
                r["e1_acc_mean"] = float(np.mean([x["acc"] for x in per_seed]))
                r["e1_fgt_mean"] = float(np.mean([x["fgt"] for x in per_seed]))
                r["e1_runs"] = per_seed
        return recs

    if pass_set:
        fewest = min(r["n_active_levers"] for r in pass_set)
        tied = e1_acc_for([r for r in pass_set if r["n_active_levers"] == fewest])
        chosen = max(tied, key=lambda r: (r["e1_acc_mean"], ))
        trace["rule"] = ("PASS-set non-empty -> fewest active levers, tie-break higher "
                         "E1 ACC")
    else:
        best_min = max(r["min_interleaved_acc"] for r in runs)
        tied = e1_acc_for([r for r in runs if r["min_interleaved_acc"] == best_min])
        fewest = min(r["n_active_levers"] for r in tied)
        tied = [r for r in tied if r["n_active_levers"] == fewest]
        chosen = max(tied, key=lambda r: (r["e1_acc_mean"], ))
        trace["rule"] = ("PASS-set EMPTY -> best-weakest interleaved stream, then fewest "
                         "levers, then higher E1 ACC; PR-06 keeps the honest 0.70 bar")

    # no-regression guard at the chosen config (already computed for tie-breaks; ensure)
    e1_acc_for([chosen])
    trace["chosen_label"] = chosen["label"]
    trace["chosen_config"] = chosen["config"]

    out = {
        "label": "LANE-EXPLORATORY",
        "date": DATE,
        "prereg_draft": "docs/preregistry/2026-09-03-interleaved-router-bar6.md"
                        " (PR-2026-09-03-06, IN-WRITE, seeds 10-14 = disjoint)",
        "bar_exploratory": BAR,
        "grid": {k: [str(x) for x in v] for k, v in GRID.items()},
        "settings": {"K": K, "d": D, "n_classes": NCLS, "h": H, "epochs": EPOCHS,
                     "batch": BATCH, "n_experts": K + 3, "z_novel": 5.0,
                     "seeds": list(seeds),
                     "stream_note": ("both interleaved streams mirror"
                                     " experiments/expert_economy.py constructions;"
                                     " E1 guard = shipped block benchmark")},
        "selection_rule_frozen": (
            "PASS-set = mixed>=0.70 AND roundrobin>=0.70 at n=2; choose fewest active "
            "levers, tie-break higher E1 ACC (computed on demand for tied candidates), "
            "then lexicographic; if empty, maximize min interleaved ACC with the same "
            "tie-breaks and draft PR-06 with the honest bar anyway."),
        "selection_trace": trace,
        "configs": runs,
        "chosen": chosen,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    outdir = os.path.join(_HERE, "..", "results", "exploratory",
                          f"interleave_fix_probe_{DATE}")
    os.makedirs(outdir, exist_ok=True)
    outp = os.path.join(outdir, "probe.json")
    with open(outp, "w") as f:
        json.dump(out, f, indent=1)

    print("\n[probe] === summary (sorted by min interleaved ACC) ===")
    for r in sorted(runs, key=lambda r: -r["min_interleaved_acc"])[:12]:
        print(f"  {r['label']:<34} mixed={r['mixed']['acc_mean']:.3f} "
              f"rr={r['roundrobin']['acc_mean']:.3f} levers={r['n_active_levers']} "
              f"E1={r.get('e1_acc_mean', float('nan')):.3f}")
    print(f"[probe] CHOSEN: {chosen['label']}  cfg={chosen['config']}")
    print(f"[probe]   E1 guard: ACC={chosen['e1_acc_mean']:.3f} "
          f"FGT={chosen['e1_fgt_mean']:.3f} (n={len(seeds)})")
    print(f"[probe] saved -> {os.path.normpath(outp)}  "
          f"({time.time() - t0:.1f}s total)")
    return out


if __name__ == "__main__":
    seed_args = [a for a in sys.argv[1:] if a.startswith("--seeds=")]
    seeds = tuple(int(x) for x in seed_args[0].split("=")[1].split(",")) \
        if seed_args else SEEDS
    main(seeds=seeds)
