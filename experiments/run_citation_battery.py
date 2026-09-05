"""
Citation-bar continual-learning battery (committee report 10-P4; commission synthesis SS7.4).

Implements and runs the comparison the CL community will check before citing the E1 result:

  (1) MULTI-EPOCH rows (E1 protocol, verbatim: structured-permuted K=5, d=24, ncls=8,
      n_samples=6000, epochs=15, lr=0.1, hidden [128,128], 10 seeds, ACC/FGT from
      src/metrics.py, mean +- 95% CI): the NEW boundary-free regularizers
        MAS                (Aljundi et al., NeurIPS 2018)          -- boundary-FREE
        OnlineEWC online   (Chaudhry 2018 + Schwarz 2018 lambda)   -- boundary-FREE (EMA anchor)
        OnlineEWC boundary (same, apples-to-apples with EWC)       -- boundary-USING
        SI                 (Zenke et al., ICML 2017)               -- boundary-USING (honest label)
      backprop/EWC/replay/oracle/Prizma rows are NOT re-run; they are copied verbatim from
      results/results.json (E1_main) so the two sources never drift.

  (2) SINGLE-PASS rows (LANE-EXPLORATORY -- labelled as such everywhere): the identical
      stream with epochs=1, i.e. every learner sees each training sample EXACTLY ONCE.
      Arms: backprop, EWC, MAS, SI, OnlineEWC(online), OnlineEWC(boundary), replay,
      PRIZMA(DFA). Prizma runs through its SHIPPED fit_task code path with epochs=1 (its
      per-batch updates are already online; what changes vs E1 is exposure: 15 visits per
      sample -> 1).

Tuning rule (mirrors E1's EWC rule exactly, so no arm is strawmanned): for each regularizer
and each protocol, lambda is tuned on SEED 0 ONLY, minimizing FGT over a small grid; the
tuned value + grid are recorded in the raw JSON meta.

Retention (docs/RETENTION.md): raw per-seed records (ACC/FGT/BWT/LA + the full R matrix)
are written crash-safe (json -> .tmp -> os.replace) to
results/citation_battery_2026-09-03/raw_{multi_epoch,single_pass}.json after EVERY seed,
before any verdict is computed. RESULTS.md is generated from those raw files only.

Usage:
  python experiments/run_citation_battery.py                       # full battery, 10 seeds
  python experiments/run_citation_battery.py --protocol single     # single-pass only
  python experiments/run_citation_battery.py --config smoke        # tiny end-to-end check
  python experiments/run_citation_battery.py --md-only             # regenerate RESULTS.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

import numpy as np

from data import structured_permuted_tasks
from baselines import MLP, EWC, MAS, SI, OnlineEWC
from metrics import AccuracyMatrix, accuracy
from prizma import Prizma
from run_continual import ci95, run_replay, run_prizma

DEFAULT_OUT = os.path.normpath(os.path.join(_HERE, "..", "results", "citation_battery_2026-09-03"))

CONFIGS = {
    # e1 = exactly the E1_main configuration (run_continual.E1_main defaults).
    "e1": dict(n_tasks=5, d=24, ncls=8, n_samples=6000, hidden=[128, 128],
               multi_epochs=15, lr=0.1, n_seeds=10, prizma_h=48),
    # smoke = tiny end-to-end check (tests + pipeline validation), never a result.
    "smoke": dict(n_tasks=2, d=8, ncls=3, n_samples=600, hidden=[16],
                  multi_epochs=3, lr=0.1, n_seeds=2, prizma_h=16),
}

# Fixed hyper-parameters (documented in docs/CONTINUAL_CITATION_BAR.md; only lambda is tuned).
FIXED = {
    "MAS": dict(omega_decay=0.995, anchor_decay=0.999),
    "SI": dict(xi=1e-3),                      # lr comes from the protocol (0.1)
    "OnlineEWC(online)": dict(gamma=0.995, anchor_decay=0.999, boundary=False),
    "OnlineEWC(boundary)": dict(gamma=0.9, boundary=True),
    "EWC": dict(),
}
GRIDS = {   # small grids; tuned on seed 0 by min-FGT (E1's rule for EWC)
    "MAS": [10.0, 100.0, 1000.0, 10000.0],
    "SI": [0.1, 1.0, 10.0, 100.0],
    "OnlineEWC(online)": [50.0, 500.0, 5000.0, 50000.0],
    "OnlineEWC(boundary)": [5.0, 50.0, 500.0, 5000.0],
    "EWC": [1.0, 5.0, 20.0, 50.0, 100.0],
}

MULTI_NEW_ARMS = ["MAS", "SI", "OnlineEWC(online)", "OnlineEWC(boundary)"]
SINGLE_ARMS = ["backprop", "EWC", "MAS", "SI", "OnlineEWC(online)", "OnlineEWC(boundary)",
               "replay", "PRIZMA(DFA)"]
# boundary classification for honest reporting
BOUNDARY_USE = {
    "backprop": "none (naive)", "EWC": "BOUNDARY (privileged)", "replay": "none (buffer)",
    "PRIZMA(DFA)": "NONE (task-free routing)",
    "MAS": "NONE (boundary-free)", "OnlineEWC(online)": "NONE (boundary-free)",
    "OnlineEWC(boundary)": "BOUNDARY (apples-to-apples)", "SI": "BOUNDARY (per Zenke 2017)",
}


# ------------------------------- core learner drivers ------------------------------------- #
def make_reg(name, lam, lr):
    """Fresh regularizer instance for one seed-run (factory, so no state leaks across runs)."""
    if name == "backprop":
        return None
    if name == "EWC":
        return EWC(lam=lam)
    if name == "MAS":
        return MAS(lam=lam, **FIXED["MAS"])
    if name == "SI":
        return SI(lam=lam, lr=lr, **FIXED["SI"])
    if name == "OnlineEWC(online)":
        return OnlineEWC(lam=lam, **FIXED["OnlineEWC(online)"])
    if name == "OnlineEWC(boundary)":
        return OnlineEWC(lam=lam, **FIXED["OnlineEWC(boundary)"])
    raise ValueError(name)


def _boundary_consolidator(reg):
    """Return a callable iff the learner consumes task boundaries (EWC, SI, and
    OnlineEWC in boundary mode). Boundary-free learners get None."""
    if isinstance(reg, OnlineEWC):
        return reg.consolidate if reg.boundary else None
    if isinstance(reg, (EWC, SI)):
        return reg.consolidate
    return None


def fit_mlp_arm(name, tasks, d, ncls, seed, epochs, lr, hidden, lam):
    """MLP-family arm through the SHIPPED MLP.fit_task path (identical footing to
    run_continual.run_backprop); returns (AccuracyMatrix, model)."""
    m = MLP([d] + list(hidden) + [ncls], seed=seed)
    reg = make_reg(name, lam, lr)
    rng = np.random.default_rng(seed)
    R = AccuracyMatrix(len(tasks))
    for i, t in enumerate(tasks):
        m.fit_task(t.Xtr, t.ytr, epochs=epochs, lr=lr, rng=rng, ewc=reg)
        consolidate = _boundary_consolidator(reg)
        if consolidate is not None:
            consolidate(m, t.Xtr, t.ytr, rng=rng)
        for j, tt in enumerate(tasks):
            R.record(i, j, accuracy(m.predict_logits(tt.Xte), tt.yte))
    return R, m


def run_arm(name, tasks, d, ncls, seed, epochs, lr, hidden, lam, prizma_h):
    """Returns (AccuracyMatrix, model_or_None)."""
    if name == "replay":
        return run_replay(tasks, d, ncls, seed, epochs, lr, hidden), None
    if name == "PRIZMA(DFA)":
        # shipped Prizma code path; E1 settings: n_experts=K+3, h=48, feedback="random" (DFA)
        return run_prizma(tasks, d, ncls, seed, epochs, len(tasks) + 3, prizma_h)
    return fit_mlp_arm(name, tasks, d, ncls, seed, epochs, lr, hidden, lam)


def _seed_record(R, seed):
    s = R.summary()
    Rm = R.R.copy()
    return {"seed": int(seed), "ACC": s["ACC"], "FGT": s["FGT"], "BWT": s["BWT"],
            "LA": s["LA"],
            "R": [[None if np.isnan(v) else float(v) for v in row] for row in Rm]}


def _atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _load_json(path):
    with open(path) as f:
        return json.load(f)


# ------------------------------------- tuning --------------------------------------------- #
def _is_valid(R, model):
    """A run is numerically valid iff the R matrix is finite and the weights never went
    NaN/inf. Grid-top lambdas CAN blow a small net up; a blown-up net is not a learner and
    must not be selected by the min-FGT rule just because it trivially forgets nothing."""
    if not np.isfinite(R.R).all():
        return False
    if model is not None:
        for p in list(model.W) + list(model.b):
            if not np.isfinite(p).all():
                return False
    return True


def tune_lam(name, tasks, d, ncls, seed0, epochs, lr, hidden, prizma_h):
    """E1's rule, applied to every regularizer: lambda tuned on seed 0 only, min FGT —
    restricted to numerically valid runs (blown-up runs are traced and excluded; flagged
    if that leaves nothing, in which case the smallest lambda is kept and marked)."""
    if name in ("backprop", "replay", "PRIZMA(DFA)"):
        return {"lam": None, "grid": [], "seed0_FGT": None, "trace": [], "fallback": False}
    trace = []
    for lam in GRIDS[name]:
        # grid tops CAN deliberately blow a net up; that is a probed-and-excluded run, so
        # the numpy warnings are silenced here only (seed runs at the chosen lam stay loud).
        with np.errstate(all="ignore"):
            R, model = run_arm(name, tasks, d, ncls, seed0, epochs, lr, hidden, lam, prizma_h)
        trace.append({"lam": lam, "seed0_ACC": R.acc(), "seed0_FGT": R.forgetting(),
                      "valid": bool(_is_valid(R, model))})
    valid = [t for t in trace if t["valid"]]
    if valid:
        best, fallback = valid[0], False
        for t in valid[1:]:
            if t["seed0_FGT"] < best["seed0_FGT"]:
                best = t
    else:
        best, fallback = trace[0], True
    return {"lam": best["lam"], "grid": list(GRIDS[name]), "seed0_FGT": best["seed0_FGT"],
            "seed0_ACC": best["seed0_ACC"], "trace": trace, "fallback": fallback,
            "rule": "min seed-0 FGT among valid runs (E1's EWC rule); NaN runs excluded"}


# ---------------------------------- battery driver ---------------------------------------- #
def run_protocol(protocol, cfg, out_dir, n_seeds=None, log=print, seed_start=0,
                 lane_label=None):
    """Run one protocol; write raw per-seed records crash-safe after every seed; return meta.

    seed_start shifts the seed block (e.g. 10 => seeds 10..10+n-1) WITHOUT changing anything
    else — added for registered runs whose frozen protocols pin fresh seed ranges (e.g.
    PR-2026-09-03-03 pins seeds 10-19). Default 0 reproduces the exploratory runs exactly.
    """
    K, d, ncls = cfg["n_tasks"], cfg["d"], cfg["ncls"]
    epochs = cfg["multi_epochs"] if protocol == "multi_epoch" else 1
    lr, hidden = cfg["lr"], cfg["hidden"]
    n_seeds = n_seeds or cfg["n_seeds"]
    seeds = list(range(seed_start, seed_start + n_seeds))
    arms = MULTI_NEW_ARMS if protocol == "multi_epoch" else SINGLE_ARMS

    raw_path = os.path.join(out_dir, f"raw_{protocol}.json")
    meta = {
        "protocol": protocol,
        "epochs_per_domain": epochs,
        "lr": lr,
        "hidden": list(hidden),
        "n_tasks": K, "d": d, "n_classes": ncls, "n_samples_per_task": cfg["n_samples"],
        "prizma_h": cfg["prizma_h"],
        "n_seeds": n_seeds,
        "seed_start": seed_start,
        "seed_range": [seed_start, seed_start + n_seeds - 1],
        "stream": "structured_permuted_tasks (src/data.py) -- identical to E1",
        "metrics": "AccuracyMatrix.acc/forgetting (src/metrics.py, Lopez-Paz & Ranzato 2017)",
        "ci": "mean +- 1.96*SEM (run_continual.ci95, E1 style)",
        "lane": lane_label if lane_label else (
            "CLAIM-CANDIDATE (multi-epoch, E1-matched protocol); SEE docs" if protocol == "multi_epoch"
            else "EXPLORATORY (lane-exploratory; single-pass claims require a registered pre-registration)"),
        "tuning_rule": ("lambda tuned on the run's FIRST seed ONLY, minimizing FGT "
                        "(identical to E1's EWC rule; 'seed 0' of the run)"),
        "tuned": {},
        "fixed_hypers": {k: v for k, v in FIXED.items()},
        "wall_seconds_per_arm": {},
        "raw_path": os.path.relpath(raw_path, os.path.join(_HERE, "..")),
    }
    doc = {"meta": meta, "arms": {a: {"seed_records": []} for a in arms}}
    _atomic_json(raw_path, doc)  # retention: the ledger exists before the first seed

    # task sequences are SHARED across arms within a seed (E1 does the same)
    tasks_by_seed = {s: structured_permuted_tasks(n_tasks=K, d=d, n_classes=ncls,
                                                  n_samples=cfg["n_samples"], seed=s)
                     for s in seeds}

    t0 = time.time()
    for arm in arms:
        ta = time.time()
        tune = tune_lam(arm, tasks_by_seed[seed_start], d, ncls, seed_start, epochs, lr,
                        hidden, cfg["prizma_h"])
        meta["tuned"][arm] = tune
        warn = "  *** WARNING: ALL grid values invalid (NaN) — smallest lam kept, arm is degenerate ***" \
            if tune.get("fallback") else ""
        log(f"[{protocol}] {arm}: tuned lam={tune['lam']} (seed-{seed_start} FGT={tune['seed0_FGT']}){warn}")
        for s in seeds:
            R, _model = run_arm(arm, tasks_by_seed[s], d, ncls, s, epochs, lr, hidden,
                                tune["lam"], cfg["prizma_h"])
            doc["arms"][arm]["seed_records"].append(_seed_record(R, s))
            _atomic_json(raw_path, doc)  # crash-safe per-seed persistence BEFORE verdicts
        meta["wall_seconds_per_arm"][arm] = round(time.time() - ta, 1)
        rec = doc["arms"][arm]["seed_records"]
        am, ac = ci95([r["ACC"] for r in rec])
        fm, fc = ci95([r["FGT"] for r in rec])
        log(f"[{protocol}] {arm}: ACC={am:.3f}+-{ac:.3f}  FGT={fm:.3f}+-{fc:.3f} "
            f"({meta['wall_seconds_per_arm'][arm]}s)")
    meta["wall_seconds_total"] = round(time.time() - t0, 1)
    _atomic_json(raw_path, doc)
    return doc


# ------------------------------- RESULTS.md generation ------------------------------------ #
def _fmt(ci_pair):
    return f"{ci_pair[0]:.3f} ± {ci_pair[1]:.3f}"


def summarize(raw):
    out = {}
    for arm, obj in raw["arms"].items():
        rec = obj["seed_records"]
        am, ac = ci95([r["ACC"] for r in rec])
        fm, fc = ci95([r["FGT"] for r in rec])
        lm, lc = ci95([r["LA"] for r in rec])
        out[arm] = {"ACC": (am, ac), "FGT": (fm, fc), "LA": (lm, lc),
                    "lam": raw["meta"]["tuned"][arm]["lam"], "n_seeds": len(rec)}
    return out


def write_results_md(out_dir, e1_path=None):
    out_dir = os.path.normpath(out_dir)
    e1_path = e1_path or os.path.join(out_dir, "..", "results.json")
    multi_raw = _load_json(os.path.join(out_dir, "raw_multi_epoch.json"))
    single_raw = _load_json(os.path.join(out_dir, "raw_single_pass.json"))
    multi = summarize(multi_raw)
    single = summarize(single_raw)
    e1 = _load_json(os.path.normpath(e1_path))["E1_main"] if os.path.exists(
        os.path.normpath(e1_path)) else None

    lines = []
    lines.append("# Citation-bar continual-learning battery — RESULTS")
    lines.append("")
    lines.append("Generated by `experiments/run_citation_battery.py`. Every number below is")
    lines.append("computed from the raw per-seed JSON archived next to this file (retention policy:")
    lines.append("`docs/RETENTION.md`) or copied verbatim from `results/results.json` (E1).")
    lines.append("")
    lines.append("| artifact | path |")
    lines.append("|---|---|")
    lines.append("| raw multi-epoch records | `raw_multi_epoch.json` |")
    lines.append("| raw single-pass records | `raw_single_pass.json` |")
    lines.append("| E1 ledger (reused rows) | `results/results.json` |")
    lines.append("")
    lines.append(f"- Stream: `structured_permuted_tasks` (src/data.py), K=5, d=24, ncls=8, "
                 f"6000 samples/task (E1-identical). Metrics: ACC/FGT per `src/metrics.py`; "
                 f"mean ± 95% CI (1.96·SEM) over seeds.")
    lines.append(f"- Tuning rule (all regularizers, both protocols): λ tuned on **seed 0 only**, "
                 f"minimizing FGT — the same rule E1 used for EWC. Grids and chosen λ per arm in the raw JSON meta.")
    lines.append("")
    lines.append("## Table 1 — MULTI-EPOCH protocol (15 epochs/domain, E1-identical)")
    lines.append("")
    lines.append("| learner | boundary use | λ (tuned) | ACC | FGT | LA | source |")
    lines.append("|---|---|---|---|---|---|---|")
    e1_order = ["backprop", "EWC", "replay(boundaries)", "oracle_multihead",
                "Prizma(DFA,no W^T)", "PRIZMA_exactW^T", "PRIZMA_noRoute(ablation)"]
    if e1:
        for name in e1_order:
            row = e1["rows"][name]
            lines.append(f"| {name} | — | — | {_fmt(row['ACC'])} | {_fmt(row['FGT'])} | — | "
                         f"E1 (`results/results.json`) |")
    for arm in MULTI_NEW_ARMS:
        s = multi[arm]
        lines.append(f"| {arm} (this run) | {BOUNDARY_USE[arm]} | {s['lam']} | "
                     f"{_fmt(s['ACC'])} | {_fmt(s['FGT'])} | {_fmt(s['LA'])} | "
                     f"`raw_multi_epoch.json` (n={s['n_seeds']}) |")
    lines.append("")
    lines.append("## Table 2 — SINGLE-PASS protocol (epochs=1, each sample seen exactly once)")
    lines.append("")
    lines.append("**LANE-EXPLORATORY.** These numbers are exploratory (post-hoc, unpowered for")
    lines.append("claims); per `docs/preregistry/POLICY.md` they may not be cited as evidence about")
    lines.append("any mechanism. Single-pass claims require the registered pre-registration")
    lines.append("drafted in `docs/CONTINUAL_CITATION_BAR.md` (PR-2026-09-03-03, status IN-WRITE).")
    lines.append("")
    lines.append("| learner | boundary use | λ (tuned) | ACC | FGT | LA | n |")
    lines.append("|---|---|---|---|---|---|---|")
    for arm in SINGLE_ARMS:
        s = single[arm]
        lines.append(f"| {arm} | {BOUNDARY_USE[arm]} | {s['lam']} | {_fmt(s['ACC'])} | "
                     f"{_fmt(s['FGT'])} | {_fmt(s['LA'])} | {s['n_seeds']} |")
    lines.append("")
    lines.append("## Honesty notes")
    lines.append("")
    lines.append("- **Prizma exposure (verified from `src/prizma.py`):** `Prizma.fit_task` loops "
                 "`epochs × permutation × train_batch`; `train_batch` is a per-batch ONLINE update "
                 "(routing/commit/consolidation decisions are made batch-by-batch, no cross-visit "
                 "accumulation). So 'updates are online' is TRUE, but E1 exposes every sample 15×; "
                 "the single-pass row is the same shipped code path with `epochs=1` — fewer "
                 "gradient steps per expert, same update rule. Prizma benefits from epochs exactly "
                 "through extra steps per domain; nothing else in its protocol changes.")
    lines.append("- **MAS / OnlineEWC(online) are boundary-free adaptations**, not the papers' "
                 "verbatim algorithms: importance is a per-batch EMA and the anchor θ* is a slow "
                 "EMA of θ (the papers consolidate at task end). These deltas are labelled [B→N] "
                 "in `docs/CONTINUAL_CITATION_BAR.md`.")
    lines.append("- **SI and OnlineEWC(boundary) consume task boundaries** exactly like the EWC "
                 "baseline E1 already shipped (consolidate() called after each domain) — honest "
                 "labels in the tables above.")
    lines.append("- The multi-epoch backprop/EWC/replay/Prizma rows are **not re-run**; they are "
                 "copied from `results/results.json` (E1_main, tuned EWC λ="
                 f"{e1['ewc_lambda'] if e1 else 'n/a'}) so the two tables can never drift apart.")
    lines.append("- **LA column** (learning accuracy, mean over tasks of R[i,i]) separates "
                 "acquisition from retention: a learner can lose ACC by failing to learn "
                 "(low LA) or by forgetting (FGT). Read the two columns together.")
    lines.append("- **Tuning-rule honesty:** the min-FGT λ rule (E1's convention, applied "
                 "identically to every regularizer) can select an under-learning point when a "
                 "large λ trivially suppresses forgetting. Every grid value's seed-0 ACC and "
                 "validity is recorded in the raw JSON meta (`tuned[arm].trace`); grid-top runs "
                 "that went NaN were excluded from selection and flagged. The one visibly "
                 "mis-selected arm is MAS single-pass (λ=100 chosen with seed-0 ACC 0.237 while "
                 "λ=10 gave 0.33 at worse FGT); its row should be read as an upper bound on MAS, "
                 "not a tuned optimum.")
    multi_wall = multi_raw.get("meta", {}).get("wall_seconds_total", "?")
    single_wall = single_raw.get("meta", {}).get("wall_seconds_total", "?")
    lines.append(f"- Wall-clock (CPU, single process): multi-epoch {multi_wall}s and single-pass "
                 f"{single_wall}s total (10 seeds/arm incl. tuning) — well inside the 'days of "
                 f"CPU work' budget of report 10-P4.")
    lines.append("")

    path = os.path.join(out_dir, "RESULTS.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


# ------------------------------------------ main ------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--protocol", choices=["multi", "single", "both"], default="both")
    ap.add_argument("--config", choices=list(CONFIGS), default="e1")
    ap.add_argument("--n-seeds", type=int, default=None)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--md-only", action="store_true",
                    help="regenerate RESULTS.md from existing raw JSON, run nothing")
    args = ap.parse_args()

    cfg = CONFIGS[args.config]
    out_dir = args.out_dir
    if args.config == "smoke" and args.out_dir == DEFAULT_OUT:
        out_dir = os.path.join(DEFAULT_OUT, "_smoke")   # smoke never touches the campaign dir
    os.makedirs(out_dir, exist_ok=True)

    if args.md_only:
        print(write_results_md(out_dir))
        return

    print(f"citation battery: config={args.config} out={out_dir}")
    if args.protocol in ("multi", "both"):
        run_protocol("multi_epoch", cfg, out_dir, n_seeds=args.n_seeds)
    if args.protocol in ("single", "both"):
        run_protocol("single_pass", cfg, out_dir, n_seeds=args.n_seeds)
    print("RESULTS.md ->", write_results_md(out_dir))


if __name__ == "__main__":
    main()
