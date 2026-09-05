"""
BAR-6 CLAIM RUN -- PR-2026-09-03-06 (docs/preregistry/2026-09-03-interleaved-router-bar6.md).

*** LANE-CLAIM. Executes the frozen protocol VERBATIM: arms, streams, seeds (10-14),
bars, and decision rule are all fixed by the registered doc -- no knob, seed, or bar
changes. Runs the stream constructors from experiments/interleave_fix_probe.py
(mirror of experiments/expert_economy.py) UNMODIFIED; this file adds only the arm
grid, crash-safe raw persistence, and the doc's decision rule. ***

ARMS (doc §5)
  shipped                     all levers off (sanity: reproduce shipped E1 row)
  plus_fixes                  hot_young=1.0 + route_stat="sample_top"   (frozen §4)
  plus_fixes_minus_route_stat route_stat="batch_mean" (isolates hot_young)
  plus_fixes_minus_hot_young  hot_young=0.0           (isolates route_stat)

STREAMS: mixed, roundrobin (interleaved) + E1 block stream (no-regression guard).
SEEDS: 10-14 (n=5, frozen; exploration used 0-1, disjoint).

BARS (doc §6, exact):
  1. +fixes E1: mean ACC within +-0.02 of 0.834 AND mean FGT <= 0.02   (gate)
  2. +fixes mean ACC >= 0.70 on BOTH interleaved streams (point estimate)  (primary)
  3. each active lever causally necessary (minus-arm fails bar 2 on some stream OR
     costs >= 0.02 mean interleaved ACC on some stream) OR config simplified.

RETENTION (docs/RETENTION.md): raw-first, crash-safe. Every completed (arm, stream,
seed) atomically persists raw/<arm>__<stream>__seed<k>.json (json -> .tmp ->
os.replace) BEFORE any verdict is computed; cells already on disk are never re-run
and never overwritten (resume after crash). The consolidated ledger is streamed to
the doc's pre-registered primary artifact path
results/interleave_fix_PR-2026-09-03-06/raw.json (mirrored at
results/interleave_bar6_2026-09-05/raw.json) after every cell.

Usage: python experiments/interleave_bar6_claim.py
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
for _p in ("..",):
    _ap = os.path.abspath(os.path.join(_HERE, _p))
    if _ap not in sys.path:
        sys.path.insert(0, _ap)

import numpy as np

import interleave_fix_probe as probe           # frozen stream constructors (reused)
from seq.stats import summarize, t_isf, _welch  # seq/stats.py machinery

DATE = "2026-09-05"
SEEDS = (10, 11, 12, 13, 14)                   # doc §5: frozen, no additions
BAR = 0.70                                     # doc §6 bar 2
E1_HEADLINE = 0.834                            # shipped 10-seed E1 mean ACC
E1_TOL = 0.02
FGT_MAX = 0.02
ATTR_COST = 0.02                               # doc §6 bar 3 attribution cost

OUTDIR = os.path.abspath(os.path.join(_HERE, "..", "results", f"interleave_bar6_{DATE}"))
RAWDIR = os.path.join(OUTDIR, "raw")
DOC_RAW = os.path.abspath(os.path.join(_HERE, "..", "results",
                                       "interleave_fix_PR-2026-09-03-06", "raw.json"))

# doc §5 arms, verbatim
ARMS = {
    "shipped": dict(freeze_min_seen=0, dynamic_vigilance=0.0, hot_young=0.0,
                    route_stat="batch_mean", n_settle_steps=0),
    "plus_fixes": dict(freeze_min_seen=0, dynamic_vigilance=0.0, hot_young=1.0,
                       route_stat="sample_top", n_settle_steps=0),
    "plus_fixes_minus_route_stat": dict(freeze_min_seen=0, dynamic_vigilance=0.0,
                                        hot_young=1.0, route_stat="batch_mean",
                                        n_settle_steps=0),
    "plus_fixes_minus_hot_young": dict(freeze_min_seen=0, dynamic_vigilance=0.0,
                                       hot_young=0.0, route_stat="sample_top",
                                       n_settle_steps=0),
}
STREAMS = {"mixed": probe.run_mixed, "roundrobin": probe.run_roundrobin,
           "e1": probe.run_e1_guard}


def _atomic_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def run_cell(arm, stream, seed, fn):
    """One (arm, stream, seed) run + crash-safe raw persistence. Returns record."""
    t0 = time.time()
    rec = fn(seed, **ARMS[arm])
    dt = round(time.time() - t0, 2)
    raw = {"arm": arm, "stream": stream, "seed": seed, "config": ARMS[arm],
           "runtime_seconds": dt, "result": rec}
    _atomic_json(os.path.join(RAWDIR, f"{arm}__{stream}__seed{seed}.json"), raw)
    print(f"  [{arm} / {stream} / seed {seed}] "
          f"acc={rec['acc']:.4f}" + (f" fgt={rec['fgt']:.4f}" if "fgt" in rec else "")
          + f" ({dt:.1f}s)", flush=True)
    return rec


def load_or_run():
    """Crash-safe resume: cells on disk are trusted and never re-run/overwritten."""
    cells = {}
    for arm in ARMS:
        for stream, fn in STREAMS.items():
            for seed in SEEDS:
                cells[(arm, stream, seed)] = None
    # adopt any completed cells
    for (arm, stream, seed) in list(cells):
        path = os.path.join(RAWDIR, f"{arm}__{stream}__seed{seed}.json")
        if os.path.exists(path):
            with open(path) as f:
                cells[(arm, stream, seed)] = json.load(f)["result"]
    todo = [k for k, v in cells.items() if v is None]
    total = len(todo)
    print(f"[claim] {len(cells) - total}/{len(cells)} cells already on disk; "
          f"{total} to run (seeds {list(SEEDS)})", flush=True)
    t_first = time.time()
    for i, (arm, stream, seed) in enumerate(todo):
        cells[(arm, stream, seed)] = run_cell(arm, stream, seed, STREAMS[stream])
        if i == 0:
            per = time.time() - t_first
            print(f"[claim] first run {per:.1f}s -> ETA {per * (total - 1) / 60:.1f} min "
                  f"for remaining {total - 1} (budget 60 min)", flush=True)
        # stream the consolidated ledger after every cell (crash-safe, never lost)
        ledger = {
            "label": "LANE-CLAIM", "prereg": "PR-2026-09-03-06",
            "doc": "docs/preregistry/2026-09-03-interleaved-router-bar6.md",
            "date": DATE, "seeds": list(SEEDS), "arms": ARMS,
            "raw_dir": os.path.relpath(RAWDIR, os.path.join(_HERE, "..")),
            "records": [
                {"arm": a, "stream": s, "seed": k, "config": ARMS[a], "result": v}
                for (a, s, k), v in sorted(cells.items()) if v is not None
            ],
        }
        _atomic_json(DOC_RAW, ledger)                       # doc §"Primary artifact"
        _atomic_json(os.path.join(OUTDIR, "raw.json"), ledger)  # mirrored copy
    return cells


def verdict(cells):
    """The doc §6 bars + §7 decision rule, computed mechanically."""
    V = {"bar_070": BAR, "e1_headline": E1_HEADLINE, "e1_tol": E1_TOL, "fgt_max": FGT_MAX}

    def accs(arm, stream):
        return [cells[(arm, stream, s)]["acc"] for s in SEEDS]

    # per-arm x stream stats (mean, sd, Welch/Student-t 95% CI over seeds)
    table = {}
    for arm in ARMS:
        for stream in STREAMS:
            a = accs(arm, stream)
            st = summarize(a)
            row = {"n": st["n"], "mean": round(st["mean"], 4), "sd": round(st["sd"], 4),
                   "ci95": [round(x, 4) for x in st["ci95"]],
                   "per_seed": [round(x, 4) for x in a]}
            if stream == "e1":
                fg = [cells[(arm, "e1", s)]["fgt"] for s in SEEDS]
                fst = summarize(fg)
                row["fgt_mean"] = round(fst["mean"], 4)
                row["fgt_per_seed"] = [round(x, 4) for x in fg]
            table[f"{arm}|{stream}"] = row
    V["table"] = table

    # ---- bar 1 (gate): +fixes E1 ----
    g = table["plus_fixes|e1"]
    acc_ok = (E1_HEADLINE - E1_TOL) <= g["mean"] <= (E1_HEADLINE + E1_TOL)
    fgt_ok = g["fgt_mean"] <= FGT_MAX
    V["bar1_gate"] = {"arm": "plus_fixes", "stream": "e1",
                      "acc_mean": g["mean"], "acc_window": [E1_HEADLINE - E1_TOL,
                                                           E1_HEADLINE + E1_TOL],
                      "acc_ok": bool(acc_ok), "fgt_mean": g["fgt_mean"],
                      "fgt_ok": bool(fgt_ok), "pass": bool(acc_ok and fgt_ok)}
    # shipped-arm E1 sanity row (doc §5: reproduce the shipped row within seed noise)
    sh = table["shipped|e1"]
    V["shipped_e1_sanity"] = {"acc_mean": sh["mean"], "ci95": sh["ci95"],
                              "headline": E1_HEADLINE,
                              "within_seed_noise": bool(
                                  abs(sh["mean"] - E1_HEADLINE) <= 3 * (sh["sd"] or 0.02))}

    # ---- bar 2 (primary): +fixes both interleaved streams >= 0.70 ----
    b2 = {}
    for stream in ("mixed", "roundrobin"):
        r = table[f"plus_fixes|{stream}"]
        b2[stream] = {"mean": r["mean"], "ci95": r["ci95"], "pass": bool(r["mean"] >= BAR)}
    b2["pass"] = bool(all(b2[s]["pass"] for s in ("mixed", "roundrobin")))
    V["bar2_recovery"] = b2

    # ---- bar 3 (attribution): necessity of route_stat and hot_young ----
    attr = {}
    for lever, minus_arm in (("route_stat", "plus_fixes_minus_route_stat"),
                             ("hot_young", "plus_fixes_minus_hot_young")):
        per_stream = {}
        for stream in ("mixed", "roundrobin"):
            pf = table[f"plus_fixes|{stream}"]["mean"]
            mn = table[f"{minus_arm}|{stream}"]["mean"]
            wa, wb = accs("plus_fixes", stream), accs(minus_arm, stream)
            _, df, se = _welch(np.asarray(wa), np.asarray(wb))
            h = t_isf(0.025, df) * se
            per_stream[stream] = {
                "plus_fixes_mean": pf, "minus_mean": mn,
                "delta_plus_minus": round(pf - mn, 4),
                "welch_ci95_delta": [round(pf - mn - h, 4), round(pf - mn + h, 4)],
                "minus_fails_bar2": bool(mn < BAR),
                "costs_ge_002": bool(pf - mn >= ATTR_COST)}
        nec = any(per_stream[s]["minus_fails_bar2"] or per_stream[s]["costs_ge_002"]
                  for s in ("mixed", "roundrobin"))
        attr[lever] = {"minus_arm": minus_arm, "per_stream": per_stream,
                       "causally_necessary": bool(nec)}
    V["bar3_attribution"] = attr

    # ---- doc §7 decision rule ----
    if not V["bar1_gate"]["pass"]:
        verdict_ = ("INSUFFICIENT (FAIL of bar 1: gate) -- fix rejected as "
                    "shipped-regressing; levers stay default-off; README/design docs "
                    "must not present them as fixes. BAR-6 ANALYZED->NEGATIVE; "
                    "PR-LM-1 BLOCKED on the router.")
    elif not b2["pass"]:
        verdict_ = ("INSUFFICIENT (FAIL of bar 2 at the frozen config) -- BAR-6 "
                    "recorded ANALYZED->NEGATIVE; PR-LM-1 BLOCKED on the router; the "
                    "n=5 failure numbers below are the documented blocker.")
    elif all(attr[l]["causally_necessary"] for l in ("route_stat", "hot_young")):
        verdict_ = ("PASS (bars 1+2+3) -- interleaved-router fix declared sufficient "
                    "at toy scale; BAR-6 ANALYZED->CLAIMED.")
    else:
        # simplification branch: config may be simplified to the levers that pass
        keep = [l for l in ("route_stat", "hot_young") if attr[l]["causally_necessary"]]
        verdict_ = (f"BAR 3 SIMPLIFICATION: not all levers necessary "
                    f"(necessary: {keep or 'none'}); the simplified config must be the "
                    f"claim config if it passes bars 1+2 at n=5 in THIS campaign, else "
                    f"a follow-up n=5 run is required before any claim.")
    V["verdict"] = verdict_

    # ---- blocker ledger: per-arm route_log / training-fraction (interleaved) ----
    ledger = {}
    for arm in ARMS:
        ledger[arm] = {}
        for stream in ("mixed", "roundrobin"):
            recs = [cells[(arm, stream, s)] for s in SEEDS]
            tot = sum(sum(r["route_log"]) for r in recs)
            fracs = np.mean([[c / max(sum(r["route_log"]), 1) for c in r["route_log"]]
                             for r in recs], axis=0)
            top = float(max(fracs))
            ledger[arm][stream] = {
                "route_log_per_seed": [r["route_log"] for r in recs],
                "mean_train_fraction_per_expert": [round(float(x), 4) for x in fracs],
                "top_expert_fraction_mean": round(top, 4),
                "experts_with_any_training_mean": float(np.mean(
                    [sum(1 for c in r["route_log"] if c > 0) for r in recs])),
                "recruited_per_seed": [r["recruited"] for r in recs],
                "dropped_batches_per_seed": [r.get("dropped_batches") for r in recs],
                "purity_per_seed": [r.get("purity") for r in recs]}
    V["blocker_ledger"] = ledger
    V["runtime"] = {"seeds": list(SEEDS), "arms": list(ARMS), "streams": list(STREAMS)}
    return V


def main():
    t0 = time.time()
    os.makedirs(RAWDIR, exist_ok=True)
    print(f"[claim] PR-2026-09-03-06 BAR-6 CLAIM RUN -- seeds {list(SEEDS)}, "
          f"arms={list(ARMS)}, streams={list(STREAMS)}", flush=True)
    cells = load_or_run()
    V = verdict(cells)
    V["wall_seconds"] = round(time.time() - t0, 1)
    _atomic_json(os.path.join(OUTDIR, "verdict.json"), V)
    print("\n[claim] === per-arm x stream mean ACC (n=5, seeds 10-14) ===")
    for arm in ARMS:
        row = []
        for stream in STREAMS:
            r = V["table"][f"{arm}|{stream}"]
            row.append(f"{stream}={r['mean']:.3f}±{r['sd']:.3f}")
        print(f"  {arm:<30} " + "  ".join(row))
    print(f"[claim] bar1 gate: {V['bar1_gate']}")
    print(f"[claim] bar2: {V['bar2_recovery']}")
    print(f"[claim] bar3: route_stat nec={V['bar3_attribution']['route_stat']['causally_necessary']}"
          f"  hot_young nec={V['bar3_attribution']['hot_young']['causally_necessary']}")
    print(f"[claim] VERDICT: {V['verdict']}")
    print(f"[claim] raw dir: {RAWDIR}")
    print(f"[claim] primary artifact: {DOC_RAW}")
    print(f"[claim] done in {V['wall_seconds']}s")


if __name__ == "__main__":
    main()
