"""TERMINAL probe -- G3b probationary commit (REAL-data revisit) + settle-depth arm
(target registry id PR-2026-09-03-07; terminal probe of the zero-GPU program).

*** LANE-EXPLORATORY (docs/preregistry/POLICY.md): n=2 seeds (0, 1), toy scale, run to
    SELECT a configuration, never to claim one. The only future-claim vehicle is the
    terminal PR-07 draft (docs/preregistry/2026-09-05-terminal-bar6.md, LANE-CLAIM,
    fresh seeds 20-24 -- disjoint from these). TERMINAL: whichever way this lands, the
    BAR-6 mechanism ladder ends here -- either a draft or the honest terminal verdict.

MECHANISM LADDER (measured, in order)
-------------------------------------
thresholds +0.06 (PR-06) -> G1 granularity fixes specialization (max train-fraction
1.000 -> 0.706) but ACC stays ~0.56 (granularity probe) -> G3a generative replay =
consolidation WITHOUT new information (recon drops on the pseudo-set, self-labeled
confidence flat, ACC +-0.001; volume router-bound ~1e3 items vs ~4.5e5 real
presentations) -- replay_probe_2026-09-05. The named remaining class: REAL-data revisit.

WHAT RUNS (the terminal candidate mechanisms)
---------------------------------------------
  G3b  probation=True (NEW lever, default-off, train_granularity="sample" only): a
       recruited expert stays PROBATIONARY -- it keeps training on every subsequent
       LIVE sample it recognizes (per-sample routed) until the stream's surprise floor
       says its domain passed (PROBATION_PATIENCE=8 consecutive batches with zero
       recognized free samples); only then commit/freeze. NO stored data -- it trains
       only on live samples as they arrive (buffer-free, single-pass compatible).
       Converts "one unrepeated pass at recruit" into "continuous live re-processing
       until domain exit". Freeze needs BOTH n_seen >= freeze_min_seen AND the exit
       test (probation implies the stronger veto; the LATER gate binds).
  Settle-depth arm (existing knob, never swept beyond {0,2}): n_settle_steps in {4, 8}
       -- deeper iterative processing per exposure (cortical FF/FB settling; each
       settle = one more gradient step on the same per-sample free energy). K settles
       on G1-routed samples = Kx acquisition steps without any revisit.

Configs (precedence: smallest first):
  best_prior         = train_granularity="sample" + route_stat="sample_top" (the
                       frozen prior stack; route_stat INERT under G1 -- documented +
                       tested -- kept because it is the prior stack)
  +probation         = best_prior + probation=True
  +settle4           = best_prior + n_settle_steps=4
  +settle8           = best_prior + n_settle_steps=8
  +probation+settle4 = best_prior + probation=True + n_settle_steps=4

5 configs x 2 streams (mixed, roundrobin) x 2 seeds = 20 runs, plus the E1 block-stream
guard (run on demand per the frozen rule below, n=2).

FROZEN SELECTION RULE (written BEFORE running; POLICY.md discipline)
--------------------------------------------------------------------
Let ACC_mean(stream, cfg) = mean over seeds {0,1} of routed test accuracy. Let
MAXFRAC_mixed(cfg) = max over mixed runs and experts of route_log[m]/sum(route_log)
(the per-expert TRAINING-fraction ledger; PR-06 lesson: check the ledger).

  1. SPECIALIZATION FILTER (hard): eligible iff MAXFRAC_mixed(cfg) <= 0.85.
  2. PASS-set := {cfg eligible AND ACC_mean('mixed') >= 0.70 AND
                  ACC_mean('roundrobin') >= 0.70 AND E1_guard(cfg)} where the E1 guard
     is |ACC_mean(E1) - 0.834| <= 0.02 (hard side: ACC >= 0.814) AND FGT_mean <= 0.02
     (the shipped block headline; RISK FLAG recorded honestly at n=2 in both prior
     probes: seed noise alone moved shipped E1 to 0.813 -- inside the band -- and
     best_prior's G1+G3a E1 to 0.666 -- far outside).
  3. If PASS-set non-empty: chosen = FEWEST active knobs; ties -> precedence order.
  4. If PASS-set empty: chosen = the ELIGIBLE config maximizing min(ACC_mean over the
     two interleaved streams); ties -> fewest knobs, then precedence. Its E1 guard is
     run and recorded. (If nothing is even eligible, the same rule over all configs,
     recorded honestly.)
  5. BRANCH: draft docs/preregistry/2026-09-05-terminal-bar6.md (PR-2026-09-03-07,
     LANE-CLAIM) iff the chosen config has ACC_mean >= 0.68 on BOTH streams AND passes
     the specialization filter AND its E1 guard holds. Otherwise NO PR DRAFT -- the
     honest TERMINAL mechanistic verdict is written instead (pre-committed failure
     branch): does live re-processing / deeper per-exposure compute close the
     acquisition-volume gap, or is the interleaved-single-pass regime information-
     bounded for this architecture at toy scale?
  6. PRECEDENCE: best_prior < probation < settle4 < settle8 < probation_settle4.

PROBATION PATTERN FROZEN A PRIORI: PROBATION_PATIENCE = 8 miss-batches (frozen in
src/prizma.py before any probation run). Settle depths {4, 8} frozen here. No other
probation/settle strength will be run in this probe.

RUNTIME BUDGET: <= ~60 min CPU; the first cell is timed and the projected total
printed; the script aborts before the grid if the projection exceeds the budget.

OUTPUT
------
results/exploratory/terminal_probe_2026-09-05/probe.json (raw per-run records with
route_log/n_seen ledgers + end-state expert flags + freeze-event counts + the applied
selection trace). Raw artifacts are never deleted (docs/RETENTION.md); per POLICY.md
nothing here may be cited as a claim.

Usage: python experiments/terminal_probe.py [--seeds 0,1]
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
DRAFT_FLOOR = 0.68          # min interleaved ACC_mean for the terminal draft branch
SPECIALIZATION_MAX = 0.85    # hard ledger filter (PR-06 lesson)
E1_TARGET_ACC = 0.834       # shipped E1 headline (the no-regression anchor)
E1_ACC_TOL = 0.02           # |acc - 0.834| <= 0.02  (hard side: acc >= 0.814)
E1_MAX_FGT = 0.02
BUDGET_MIN = 60.0           # runtime budget, minutes

# PRECEDENCE order == frozen selection order (smallest config first)
CONFIGS = [
    ("best_prior",        dict(train_granularity="sample", route_stat="sample_top")),
    ("probation",         dict(train_granularity="sample", route_stat="sample_top",
                              probation=True)),
    ("settle4",           dict(train_granularity="sample", route_stat="sample_top",
                               n_settle_steps=4)),
    ("settle8",           dict(train_granularity="sample", route_stat="sample_top",
                               n_settle_steps=8)),
    ("probation_settle4", dict(train_granularity="sample", route_stat="sample_top",
                               probation=True, n_settle_steps=4)),
]

OUTDIR = os.path.abspath(os.path.join(_HERE, "..", "results", "exploratory",
                                      f"terminal_probe_{DATE}"))
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


def run_cell_g3b(name, cfg, stream, seed, stream_kind):
    """One (config, interleaved-stream, seed) run + crash-safe raw persistence.

    The stream construction mirrors experiments/interleave_fix_probe.run_mixed /
    .run_roundrobin VERBATIM (same generators, permutations, exposure); additions are
    the end-state expert flags (probationary/committed/frozen), the freeze-event count
    and the n_seen trained-volume ledger, which the reused runner does not expose."""
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
    probationary = [int((not e.committed) and e.n_seen > 0) for e in p.experts]
    rec = {
        "acc": acc,
        "recruited": int(sum(e.n_seen > 0 for e in p.experts)),
        "route_log": p.route_log.tolist(),
        "fractions": ledger_fractions(p.route_log.tolist()),
        "max_fraction": max(ledger_fractions(p.route_log.tolist())),
        "n_seen": n_seen,
        "trained_fractions": ledger_fractions(n_seen),
        "max_trained_fraction": max(ledger_fractions(n_seen)),
        "committed_end": [int(e.committed) for e in p.experts],
        "frozen_end": [int(e.frozen) for e in p.experts],
        "probationary_end": probationary,
        "freeze_events": len(p.ledger["freeze_events"]),
        "dropped_batches": p.ledger["n_dropped_batches"],
        "purity": [round(d["purity"], 4) for d in ev["domains"]],
    }
    dt = round(time.time() - t0, 2)
    raw = {"config": name, "stream": stream, "seed": seed, "knobs": cfg,
           "runtime_seconds": dt, "result": rec}
    _atomic_json(os.path.join(RAWDIR, f"{name}__{stream}__seed{seed}.json"), raw)
    pinfo = (f" probationary_end={sum(probationary)}" if cfg.get("probation") else "")
    print(f"  [{name} / {stream} / seed {seed}] acc={rec['acc']:.4f} "
          f"maxfrac={rec['max_fraction']:.3f} recruited={rec['recruited']}{pinfo} "
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
    print(f"[probe] TERMINAL: {len(CONFIGS)} configs x 2 streams x {len(seeds)} seeds "
          f"= {len(todo)} runs (+ E1 guards on demand); BAR={BAR}, draft floor="
          f"{DRAFT_FLOOR}, specialization max-frac<={SPECIALIZATION_MAX}, E1 guard "
          f"|acc-{E1_TARGET_ACC}|<={E1_ACC_TOL} & fgt<={E1_MAX_FGT}", flush=True)

    # -- budget guard: time the first cell, project the total ------------------- #
    n0, c0, s0, k0 = todo[0]
    cells[(n0, s0, k0)] = run_cell_g3b(n0, c0, s0, k0, "mixed")
    per = time.time() - t0
    proj_min = per * len(todo) / 60.0
    print(f"[probe] first run {per:.1f}s -> projected grid total {proj_min:.1f} min "
          f"(budget {BUDGET_MIN:.0f} min)", flush=True)
    if proj_min > BUDGET_MIN:
        raise SystemExit(f"[probe] ABORT: projected {proj_min:.1f} min exceeds the "
                         f"{BUDGET_MIN:.0f} min budget -- do not run the grid.")
    for (name, cfg, stream, seed) in todo[1:]:
        kind = "roundrobin" if stream == "roundrobin" else "mixed"
        cells[(name, stream, seed)] = run_cell_g3b(name, cfg, stream, seed, kind)

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
                "n_seen_per_seed": [r["n_seen"] for r in runs],
                "probationary_end_per_seed": [r["probationary_end"] for r in runs],
                "committed_end_per_seed": [r["committed_end"] for r in runs],
                "frozen_end_per_seed": [r["frozen_end"] for r in runs],
                "freeze_events_per_seed": [r["freeze_events"] for r in runs],
                "recruited_per_seed": [r["recruited"] for r in runs],
                "purity_per_seed": [r["purity"] for r in runs],
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
        print(f"  {name:<18} mixed={r['mixed']['acc_mean']:.3f} "
              f"rr={r['roundrobin']['acc_mean']:.3f} "
              f"maxfrac_mixed={r['maxfrac_mixed']:.3f} "
              f"{'PASS' if r['passes_bar'] else ''}", flush=True)

    # -- the frozen selection rule, applied mechanically ------------------------ #
    trace = {"bar": BAR, "draft_floor": DRAFT_FLOOR,
             "specialization_max": SPECIALIZATION_MAX,
             "e1_guard": {"target_acc": E1_TARGET_ACC, "acc_tol": E1_ACC_TOL,
                          "min_acc": E1_TARGET_ACC - E1_ACC_TOL,
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
                         "fewest active knobs, then precedence [best_prior < "
                         "probation < settle4 < settle8 < probation_settle4]")
        trace["pass_set"] = pass_set
    else:
        pool = [n for n, _ in CONFIGS if summary[n]["eligible"]] or \
               [n for n, _ in CONFIGS]
        best_min = max(summary[n]["min_interleaved_acc"] for n in pool)
        tied = [n for n in pool if summary[n]["min_interleaved_acc"] == best_min]
        chosen = pick(tied)
        attach_e1([chosen])
        trace["rule"] = ("PASS-set EMPTY -> best-weakest min interleaved ACC among "
                         "ELIGIBLE configs (all configs if none eligible), then fewest "
                         "knobs, then precedence; E1 guard recorded for the reported "
                         "best")
        trace["pass_set"] = []

    ch = summary[chosen]
    trace["chosen"] = chosen
    e1_ok = bool(ch.get("e1", {}).get("guard_ok", False))
    if "e1" not in ch:                      # defensive: chosen must carry an E1 row
        attach_e1([chosen])
        e1_ok = ch["e1"]["guard_ok"]
    draft = bool(ch["mixed"]["acc_mean"] >= DRAFT_FLOOR
                 and ch["roundrobin"]["acc_mean"] >= DRAFT_FLOOR
                 and ch["specialization_ok"] and e1_ok)
    trace["e1_guard_ok"] = e1_ok
    trace["terminal_draft_branch"] = (
        "DRAFT PR-2026-09-03-07 (docs/preregistry/2026-09-05-terminal-bar6.md)" if draft
        else "NO PR DRAFT (terminal): write the honest mechanistic verdict -- the "
             "interleaved regime is information-bounded for this architecture at toy "
             "scale at the measured exposures")

    # winner-minus arms for the draft (mechanical dict-subtraction, honest notes)
    if draft:
        cfg_win = dict(CONFIGS)[chosen]
        arms = {"shipped": {}, "winner": cfg_win}
        notes = []
        for lever, off, arm_name in (("probation", False, "probation"),
                                     ("n_settle_steps", 0, "settle")):
            minus = dict(cfg_win)
            minus[lever] = off
            match = [n for n, c in CONFIGS if c == minus]
            arms[f"winner-minus-{arm_name}"] = minus
            if match:
                notes.append(f"winner-minus-{arm_name} == probe arm '{match[0]}'")
            elif not minus:
                notes.append(f"winner-minus-{arm_name} == the shipped arm")
            else:
                notes.append(f"winner-minus-{arm_name} is a NEW config "
                             f"(not in the probe grid)")
        trace["draft_arms"] = arms
        trace["draft_arm_notes"] = notes

    out = {
        "label": "LANE-EXPLORATORY",
        "terminal": True,
        "date": DATE,
        "target_prereg": "docs/preregistry/2026-09-05-terminal-bar6.md "
                         "(PR-2026-09-03-07, IN-WRITE, seeds 20-24 = disjoint)",
        "bar_exploratory": BAR,
        "settings": {"K": probe.K, "d": probe.D, "n_classes": probe.NCLS,
                     "h": probe.H, "epochs": probe.EPOCHS, "batch": probe.BATCH,
                     "n_experts": probe.K + 3, "z_novel": 5.0,
                     "probation": {"PROBATION_PATIENCE": 8,
                                   "requires": "train_granularity='sample'"},
                     "settle": {"depths": [4, 8],
                                "note": "existing shipped knob, first sweep past 2"},
                     "seeds": list(seeds),
                     "stream_note": ("mixed/roundrobin constructors reused verbatim "
                                     "from experiments/interleave_fix_probe.py; "
                                     "E1 guard = shipped block benchmark")},
        "selection_rule_frozen": (
            "Eligibility (hard): max per-expert train-fraction over mixed runs <= "
            "0.85. PASS-set = eligible AND mixed >= 0.70 AND roundrobin >= 0.70 AND "
            "E1 guard (|ACC-0.834| <= 0.02 i.e. ACC >= 0.814, FGT <= 0.02); choose "
            "fewest active knobs, tie-break precedence [best_prior < probation < "
            "settle4 < settle8 < probation_settle4]. If empty: best-weakest min "
            "interleaved ACC among eligible, same tie-breaks; E1 guard recorded. "
            "Terminal PR-2026-09-03-07 drafted iff the reported best is >= 0.68 on "
            "BOTH streams AND passes the ledger filter AND its E1 guard holds; "
            "otherwise the honest TERMINAL mechanistic verdict is written "
            "(pre-committed failure branch)."),
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
    print(f"[probe]   BRANCH: {trace['terminal_draft_branch']}")
    print(f"[probe] saved -> {os.path.normpath(outp)}  ({time.time() - t0:.1f}s total)")
    return out


if __name__ == "__main__":
    seed_args = [a for a in sys.argv[1:] if a.startswith("--seeds=")]
    seeds = tuple(int(x) for x in seed_args[0].split("=")[1].split(",")) \
        if seed_args else SEEDS
    main(seeds=seeds)
