"""Replay-before-freeze exploratory selection study -- G3a generative replay
(target registry id PR-2026-09-03-07; fusion spec Addendum 2 successor class).

*** LANE-EXPLORATORY (docs/preregistry/POLICY.md): n=2 seeds (0, 1), toy scale, run to
    SELECT a configuration, never to claim one. The only future-claim vehicle is the
    PR-07 draft (docs/preregistry/2026-09-05-replay-bar6.md, LANE-CLAIM, fresh
    seeds 20-24 -- disjoint from these). ***

WHAT RUNS
---------
The two exploratory interleaved streams of experiments/interleave_fix_probe.py (the
same constructors the granularity probe reused: same generator, same permutations,
same per-sample exposure), at the shipped E1 settings (structured_permuted_tasks,
K=5, d=24, 8 classes, h=48, M=K+3=8 experts, epochs=15, batch=128, DFA,
consolidate=True, z_novel=5.0), under three configs:

  shipped     -- every lever off (granularity-probe baseline: monolithic ledger,
                 min interleaved ACC 0.570)
  g3a         -- replay_passes=15, replay_items=256 (G3a generative replay at the
                 shipped batch granularity): at every PGM freeze the frozen expert
                 generates 256 pseudo-inputs from its own decoder (h~ ~ diagonal
                 Gaussian EMA of its lifetime hidden activity; x~ = h~ Wdec + bdec),
                 self-labels them once with its own head, pseudo-trains 15 passes
                 with its own local rule, then freezes. Buffer-free (DGR, Shin et
                 al. 2017, adapted -- no raw data stored, no GAN).
  best_prior  -- train_granularity="sample" (G1, the granularity probe's
                 specialization fix: max train-fraction 1.000 -> 0.706)
                 + route_stat="sample_top" (the PR-06 statistic; INERT under G1 --
                 documented + tested -- kept because it is the frozen prior stack)
                 + the same G3a replay. The candidate that attacks BOTH measured
                 causes: G1 gives domain specialists, G3a gives the repeat passes.

3 configs x 2 streams (mixed, roundrobin) x 2 seeds = 12 runs, plus the E1
block-stream guard (run on demand for candidates/best per the rule below, n=2).

FROZEN SELECTION RULE (written BEFORE running; POLICY.md discipline)
--------------------------------------------------------------------
Let ACC_mean(stream, cfg) = mean over seeds {0,1} of routed test accuracy. Let
MAXFRAC_mixed(cfg) = max over mixed runs and experts of route_log[m]/sum(route_log)
(the per-expert TRAINING-fraction ledger; PR-06 lesson: check the ledger).

  1. SPECIALIZATION FILTER (hard): a config is eligible iff
     MAXFRAC_mixed(cfg) <= 0.85.
  2. PASS-set := {cfg eligible AND ACC_mean('mixed') >= 0.70 AND
                  ACC_mean('roundrobin') >= 0.70 AND E1_guard(cfg)}, where the E1
     guard is |ACC_mean(E1) - 0.834| <= 0.02 AND FGT_mean(E1) <= 0.02 (the shipped
     block headline; RISK FLAG, recorded honestly: at n=2 seed noise alone moved the
     shipped E1 guard to 0.813 in the granularity probe, i.e. the tolerance band is
     tight at this n).
  3. If PASS-set non-empty: the selected config has the FEWEST active knobs (a knob
     is active iff its value differs from shipped); ties -> precedence order.
  4. If PASS-set empty: the reported best is the ELIGIBLE config maximizing
     min(ACC_mean over the two interleaved streams); ties -> fewest knobs, then
     precedence. Its E1 guard is run and recorded.
  5. BRANCH: draft PR-2026-09-03-07 (docs/preregistry/2026-09-05-replay-bar6.md)
     iff the reported best has ACC_mean >= 0.68 on BOTH streams AND passes the
     specialization filter. Otherwise NO PR DRAFT -- the honest mechanistic
     conclusion is written instead (the pre-committed failure branch).
  6. PRECEDENCE: shipped < g3a < best_prior.

REPLAY STRENGTH FROZEN A PRIORI: replay_passes=15 matches the block stream's 15
epochs -- the acquisition-volume hypothesis gives the recruit the same NUMBER of
passes the block stream grants; replay_items=256 is one batch-equivalent. Internal
constants (hidden EMA rate 0.05, variance floor 1e-4) were frozen in src/prizma.py
before any run. The honest information-limit caveat travels with every arm: generated
items are diagonal-Gaussian approximations -- replay CANNOT add information beyond
the single real pass; it can only consolidate it. This probe measures whether that
is enough.

RUNTIME BUDGET: <= ~45 min CPU; the first cell is timed and the projected total
printed (the granularity probe ran the same-size grid in 19.5 s; replay adds only
the freeze events' pseudo-training); the script aborts before the grid if the
projection exceeds the budget.

OUTPUT
------
results/exploratory/replay_probe_2026-09-05/probe.json (raw per-run records with
route_log + n_seen ledgers + G3 replay diagnostics + the applied selection trace).
Raw artifacts are never deleted (docs/RETENTION.md); per POLICY.md nothing here may
be cited as a claim.

Usage: python experiments/replay_probe.py [--seeds 0,1]
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

import interleave_fix_probe as probe    # frozen stream constructors + E1 guard (reused)

DATE = "2026-09-05"
SEEDS = (0, 1)
BAR = 0.70                  # the BAR-6 bar (exploratory pass level)
DRAFT_FLOOR = 0.68          # min interleaved ACC_mean for the PR-07 draft branch
SPECIALIZATION_MAX = 0.85    # hard ledger filter (PR-06 lesson)
E1_TARGET_ACC = 0.834       # shipped E1 headline (the no-regression anchor)
E1_ACC_TOL = 0.02
E1_MAX_FGT = 0.02
BUDGET_MIN = 45.0           # runtime budget, minutes

# PRECEDENCE order == frozen selection order (smallest config first)
CONFIGS = [
    ("shipped",    dict()),
    ("g3a",        dict(replay_passes=15, replay_items=256)),
    ("best_prior", dict(train_granularity="sample", route_stat="sample_top",
                        replay_passes=15, replay_items=256)),
]

OUTDIR = os.path.abspath(os.path.join(_HERE, "..", "results", "exploratory",
                                      f"replay_probe_{DATE}"))
RAWDIR = os.path.join(OUTDIR, "raw")


def _atomic_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def ledger_fractions(counts):
    tot = max(sum(counts), 1)
    return [round(c / tot, 4) for c in counts]


def replay_diag(p):
    """G3 replay diagnostics: items generated + pre/post-freeze surrogate values
    (recon MSE and head confidence on the pseudo-set), aggregated over freeze events."""
    evs = p.replay_events
    if not evs:
        return {"n_events": 0, "items_total": 0}
    return {
        "n_events": len(evs),
        "items_total": int(sum(e["items"] for e in evs)),
        "passes_per_event": int(evs[0]["passes"]),
        "recon_pre_mean": round(float(np.mean([e["recon_pre"] for e in evs])), 5),
        "recon_post_mean": round(float(np.mean([e["recon_post"] for e in evs])), 5),
        "conf_pre_mean": round(float(np.mean([e["conf_pre"] for e in evs])), 5),
        "conf_post_mean": round(float(np.mean([e["conf_post"] for e in evs])), 5),
        "events": [{k: (round(v, 5) if isinstance(v, float) else v)
                    for k, v in e.items()} for e in evs],
    }


def run_cell_g3(name, cfg, stream, seed, stream_kind):
    """One (config, interleaved-stream, seed) run + crash-safe raw persistence.

    The stream construction mirrors experiments/interleave_fix_probe.run_mixed /
    .run_roundrobin VERBATIM (same generators, permutations, exposure); the only
    additions are the G3 replay diagnostics and the n_seen trained-volume ledger,
    which the reused runner does not expose (it does not return the model)."""
    t0 = time.time()
    tasks = probe.structured_permuted_tasks(n_tasks=probe.K, d=probe.D,
                                            n_classes=probe.NCLS, seed=seed)
    if stream_kind == "mixed":
        Xall = np.vstack([t.Xtr for t in tasks])
        yall = np.concatenate([t.ytr for t in tasks])
        perm = np.random.default_rng(1000 + seed).permutation(len(Xall))
        p = probe.make_prizma(seed, **cfg)
        p._task_label = "interleaved(EXPLORATORY)"
        rng = np.random.default_rng(seed)
        p.fit_task(Xall[perm], yall[perm], epochs=probe.EPOCHS, rng=rng)
    else:  # roundrobin: domain-pure batches, globally shuffled order
        p = probe.make_prizma(seed, **cfg)
        p._task_label = "roundrobin(EXPLORATORY)"
        rng = np.random.default_rng(seed)
        for _ in range(probe.EPOCHS):
            batches = []
            for t in tasks:
                Yall = np.eye(p.K, dtype=np.float32)[t.ytr]
                idx = rng.permutation(len(t.Xtr))
                for s in range(0, len(idx), probe.BATCH):
                    bi = idx[s:s + probe.BATCH]
                    batches.append((t.Xtr[bi], Yall[bi], t.ytr[bi]))
            for b in rng.permutation(len(batches)):
                Xb, Yb, yb = batches[int(b)]
                p.train_batch(Xb, Yb, yb)
    acc = probe.routed_acc(p, tasks)
    ev = probe.eval_routing(p, [(j, t.Xte) for j, t in enumerate(tasks)])
    n_seen = [int(e.n_seen) for e in p.experts]
    rec = {
        "acc": acc,
        "recruited": int(sum(e.n_seen > 0 for e in p.experts)),
        "route_log": p.route_log.tolist(),
        "fractions": ledger_fractions(p.route_log.tolist()),
        "max_fraction": max(ledger_fractions(p.route_log.tolist())),
        "n_seen": n_seen,
        "trained_fractions": ledger_fractions(n_seen),
        "max_trained_fraction": max(ledger_fractions(n_seen)),
        "dropped_batches": p.ledger["n_dropped_batches"],
        "purity": [round(d["purity"], 4) for d in ev["domains"]],
        "replay": replay_diag(p),
    }
    dt = round(time.time() - t0, 2)
    raw = {"config": name, "stream": stream, "seed": seed, "knobs": cfg,
           "runtime_seconds": dt, "result": rec}
    _atomic_json(os.path.join(RAWDIR, f"{name}__{stream}__seed{seed}.json"), raw)
    rinfo = (f" replay_ev={rec['replay']['n_events']}"
             f"/{rec['replay']['items_total']}it"
             f" dRecon={rec['replay'].get('recon_pre_mean', 0) - rec['replay'].get('recon_post_mean', 0):+.4f}"
             if rec["replay"]["n_events"] else "")
    print(f"  [{name} / {stream} / seed {seed}] acc={rec['acc']:.4f} "
          f"maxfrac={rec['max_fraction']:.3f} recruited={rec['recruited']}{rinfo} "
          f"({dt:.1f}s)", flush=True)
    return rec


def run_e1_guard(name, cfg, seed):
    """Shipped E1 block stream via the REUSED guard in interleave_fix_probe."""
    t0 = time.time()
    rec = probe.run_e1_guard(seed, **cfg)
    dt = round(time.time() - t0, 2)
    raw = {"config": name, "stream": "e1", "seed": seed, "knobs": cfg,
           "runtime_seconds": dt, "result": rec}
    _atomic_json(os.path.join(RAWDIR, f"{name}__e1__seed{seed}.json"), raw)
    print(f"  [{name} / e1 / seed {seed}] acc={rec['acc']:.4f} fgt={rec['fgt']:.4f} "
          f"({dt:.1f}s)", flush=True)
    return rec


def main(seeds=SEEDS):
    t0 = time.time()
    os.makedirs(RAWDIR, exist_ok=True)
    cells = {}
    todo = [(n, c, s, k) for n, c in CONFIGS
            for s, kind in (("mixed", "mixed"), ("roundrobin", "roundrobin"))
            for k in seeds]
    print(f"[probe] {len(CONFIGS)} configs x 2 streams x {len(seeds)} seeds = "
          f"{len(todo)} runs (+ E1 guards on demand); BAR={BAR}, draft floor="
          f"{DRAFT_FLOOR}, specialization max-frac<={SPECIALIZATION_MAX}, E1 guard "
          f"|acc-{E1_TARGET_ACC}|<={E1_ACC_TOL} & fgt<={E1_MAX_FGT}", flush=True)

    # -- budget guard: time the first cell, project the total ------------------- #
    n0, c0, s0, k0 = todo[0]
    cells[(n0, s0, k0)] = run_cell_g3(n0, c0, s0, k0, "mixed")
    per = time.time() - t0
    proj_min = per * len(todo) / 60.0
    print(f"[probe] first run {per:.1f}s -> projected grid total {proj_min:.1f} min "
          f"(budget {BUDGET_MIN:.0f} min)", flush=True)
    if proj_min > BUDGET_MIN:
        raise SystemExit(f"[probe] ABORT: projected {proj_min:.1f} min exceeds the "
                         f"{BUDGET_MIN:.0f} min budget -- do not run the grid.")
    for (name, cfg, stream, seed) in todo[1:]:
        kind = "roundrobin" if stream == "roundrobin" else "mixed"
        cells[(name, stream, seed)] = run_cell_g3(name, cfg, stream, seed, kind)

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
                "max_trained_fraction_per_seed":
                    [r["max_trained_fraction"] for r in runs],
                "route_log_per_seed": [r["route_log"] for r in runs],
                "recruited_per_seed": [r["recruited"] for r in runs],
                "replay_diag_per_seed": [r["replay"] for r in runs],
            }
        rec["min_interleaved_acc"] = min(rec["mixed"]["acc_mean"],
                                         rec["roundrobin"]["acc_mean"])
        rec["maxfrac_mixed"] = max(rec["mixed"]["max_fraction_per_seed"])
        rec["specialization_ok"] = bool(rec["maxfrac_mixed"] <= SPECIALIZATION_MAX)
        rec["eligible"] = rec["specialization_ok"]
        rec["passes_bar"] = bool(rec["mixed"]["acc_mean"] >= BAR
                                 and rec["roundrobin"]["acc_mean"] >= BAR
                                 and rec["eligible"])
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
             "specialization_max": SPECIALIZATION_MAX,
             "e1_guard": {"target_acc": E1_TARGET_ACC, "acc_tol": E1_ACC_TOL,
                          "max_fgt": E1_MAX_FGT}}

    def attach_e1(names):
        for name in names:
            if "e1" in summary[name]:
                continue
            runs = [run_e1_guard(name, dict(CONFIGS)[name], k) for k in seeds]
            acc_m = float(np.mean([r["acc"] for r in runs]))
            fgt_m = float(np.mean([r["fgt"] for r in runs]))
            summary[name]["e1"] = {
                "acc_mean": round(acc_m, 4),
                "fgt_mean": round(fgt_m, 4),
                "acc_per_seed": [round(r["acc"], 4) for r in runs],
                "fgt_per_seed": [round(r["fgt"], 4) for r in runs],
                "guard_ok": bool(abs(acc_m - E1_TARGET_ACC) <= E1_ACC_TOL
                                 and fgt_m <= E1_MAX_FGT)}

    def pick(tied):
        fewest = min(summary[n]["n_active_knobs"] for n in tied)
        tied = [n for n in tied if summary[n]["n_active_knobs"] == fewest]
        return min(tied, key=lambda n: CONFIGS.index((n, dict(CONFIGS)[n])))

    # PASS-set needs the E1 guard on its members (evaluated on demand)
    pass_set = [n for n, _ in CONFIGS if summary[n]["passes_bar"]]
    if pass_set:
        attach_e1(pass_set)
        pass_set = [n for n in pass_set if summary[n]["e1"]["guard_ok"]]
    if pass_set:
        chosen = pick(pass_set)
        trace["rule"] = ("PASS-set non-empty (bar + specialization + E1 guard) -> "
                         "fewest active knobs, then precedence [shipped < g3a < "
                         "best_prior]")
        trace["pass_set"] = pass_set
    else:
        elig = [n for n, _ in CONFIGS if summary[n]["eligible"]]
        best_min = max(summary[n]["min_interleaved_acc"] for n in elig)
        tied = [n for n in elig if summary[n]["min_interleaved_acc"] == best_min]
        chosen = pick(tied)
        attach_e1([chosen])
        trace["rule"] = ("PASS-set EMPTY -> best-weakest min interleaved ACC among "
                         "ELIGIBLE configs, then fewest knobs, then precedence; E1 "
                         "guard recorded for the reported best")
        trace["pass_set"] = []

    ch = summary[chosen]
    trace["chosen"] = chosen
    draft = bool(ch["mixed"]["acc_mean"] >= DRAFT_FLOOR
                 and ch["roundrobin"]["acc_mean"] >= DRAFT_FLOOR
                 and ch["specialization_ok"])
    trace["pr07_draft_branch"] = (
        "DRAFT PR-2026-09-03-07" if draft else
        "NO PR DRAFT (best below the 0.68 floor on some stream, or the specialization "
        "filter fails) -- write the honest mechanistic conclusion")

    out = {
        "label": "LANE-EXPLORATORY",
        "date": DATE,
        "target_prereg": "docs/preregistry/2026-09-05-replay-bar6.md "
                         "(PR-2026-09-03-07, IN-WRITE, seeds 20-24 = disjoint)",
        "bar_exploratory": BAR,
        "settings": {"K": probe.K, "d": probe.D, "n_classes": probe.NCLS,
                     "h": probe.H, "epochs": probe.EPOCHS, "batch": probe.BATCH,
                     "n_experts": probe.K + 3, "z_novel": 5.0,
                     "replay": {"replay_passes": 15, "replay_items": 256,
                                "hidden_EMA_rate": 0.05, "var_floor": 1e-4},
                     "seeds": list(seeds),
                     "stream_note": ("mixed/roundrobin constructors reused verbatim "
                                     "from experiments/interleave_fix_probe.py; "
                                     "E1 guard = shipped block benchmark")},
        "selection_rule_frozen": (
            "Eligibility (hard): max per-expert train-fraction over mixed runs <= "
            "0.85. PASS-set = eligible AND mixed >= 0.70 AND roundrobin >= 0.70 AND "
            "E1 guard (|ACC-0.834| <= 0.02, FGT <= 0.02); choose fewest active knobs, "
            "tie-break precedence [shipped < g3a < best_prior]. If empty: best-weakest "
            "min interleaved ACC among eligible, same tie-breaks; E1 guard recorded. "
            "PR-2026-09-03-07 drafted iff the reported best is >= 0.68 on BOTH "
            "streams AND passes the ledger filter; otherwise the honest mechanistic "
            "conclusion is written (pre-committed failure branch)."),
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
    if "e1" in ch:
        print(f"[probe]   E1 guard: ACC={ch['e1']['acc_mean']:.3f} "
              f"FGT={ch['e1']['fgt_mean']:.3f} "
              f"({'OK' if ch['e1']['guard_ok'] else 'RISK FLAG'}) (n={len(seeds)})")
    print(f"[probe]   BRANCH: {trace['pr07_draft_branch']}")
    print(f"[probe] saved -> {os.path.normpath(outp)}  ({time.time() - t0:.1f}s total)")
    return out


if __name__ == "__main__":
    seed_args = [a for a in sys.argv[1:] if a.startswith("--seeds=")]
    seeds = tuple(int(x) for x in seed_args[0].split("=")[1].split(",")) \
        if seed_args else SEEDS
    main(seeds=seeds)
