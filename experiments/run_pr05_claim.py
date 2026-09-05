"""PR-2026-09-03-05 claim run — bounded-M economy lever, Policy A (docs/EXPERT_ECONOMY.md §3.3).

Registered protocol (INDEX.md row PR-2026-09-03-05, status REGISTERED; authorized),
executed VERBATIM: on the shipped E1 stream (src.data.structured_permuted_tasks,
K=5 domains, d=24, 8 classes, n_samples=6000, h=48, M_unbounded=K+3=8 experts,
epochs=15, DFA feedback, consolidate=True, z_novel=5.0 — identical to
run_continual.E1_main's Prizma leg), arms UNBOUNDED (shipped, m_max=0) vs BOUNDED
(Policy A recruit-by-eviction at M_max=K=5), seeds 3-12 (n=10 fresh; the 0-2
exploratory seeds are untouched), 95% CIs from run_continual.ci95.

Bar (FROZEN, verbatim): PASS iff |ACC_bounded - ACC_unbounded| <= 0.02 AND
|FGT_bounded - FGT_unbounded| <= 0.02 (seed-summary means; the per-seed max |delta|
is reported honestly alongside). Falsified if either delta exceeds 0.02.

Reporting duty (docs/EXPERT_ECONOMY.md §3.3): ACC-vs-M_max (the arms table), eviction
fire counts and victims, per-expert train-fraction ledger.

Retention (docs/RETENTION.md): raw per-seed records are streamed to
results/expert_economy_PR-2026-09-03-05/raw_<arm>_seed_<s>.json the moment each run
finishes (json -> .tmp -> os.replace, crash-safe); a re-entry resumes from the raw
files on disk; raw first, verdict second.

Wall-time protocol: the first run doubles as the estimate probe. A single E1-stream
run measured ~1.5 s CPU (see RESULTS.md); the pre-declared shrink rule (>=10 seeds in
the registered protocol; shrink to 6 per arm ONLY with a prominent disclosure) fires
iff the probe exceeds ~180 s.

Run:  python experiments/run_pr05_claim.py [--force]
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np  # noqa: E402

from data import structured_permuted_tasks  # noqa: E402
from prizma import Prizma  # noqa: E402
from metrics import AccuracyMatrix, accuracy  # noqa: E402
from run_continual import ci95  # noqa: E402  (the repo's own CI convention)

OUT_DIR = os.path.join(_ROOT, "results", "expert_economy_PR-2026-09-03-05")
LANE = ("CLAIM PR-2026-09-03-05 (registered; protocol = docs/EXPERT_ECONOMY.md §5 "
        "operationalisation + §3.3 Policy A, executed verbatim)")
K, D, NCLS, H, EPOCHS, M_UNBOUNDED = 5, 24, 8, 48, 15, 8
M_MAX = K                       # bounded arm: M_max = the stream's true domain count
SEED_START, N_SEEDS = 3, 10     # fresh seeds 3..12; exploratory 0-2 untouched
BAR = 0.02                      # frozen |delta| threshold for BOTH ACC and FGT
PROBE_LIMIT_S = 180.0           # pre-declared: a single run above this shrinks n
ARMS = ("unbounded", "bounded")


def raw_path(arm, seed):
    return os.path.join(OUT_DIR, f"raw_{arm}_seed_{seed:03d}.json")


def save_json(path, obj):
    """Crash-safe write per docs/RETENTION.md: json -> .tmp -> os.replace."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def run_one(seed, arm):
    """One E1-stream Prizma run at the shipped settings; returns the raw record."""
    t0 = time.time()
    tasks = structured_permuted_tasks(n_tasks=K, d=D, n_classes=NCLS, seed=seed)
    kw = dict(d=D, h=H, K=NCLS, n_experts=M_UNBOUNDED, seed=seed,
              consolidate=True, feedback="random", z_novel=5.0)
    if arm == "bounded":
        kw["m_max"] = M_MAX                      # Policy A recruit-by-eviction
    p = Prizma(**kw)
    rng = np.random.default_rng(seed)            # E1 semantics: one rng across tasks
    R = AccuracyMatrix(len(tasks))
    for i, t in enumerate(tasks):
        p.fit_task(t.Xtr, t.ytr, epochs=EPOCHS, rng=rng)
        for j, tt in enumerate(tasks):
            R.record(i, j, accuracy(p.predict_logits(tt.Xte), tt.yte))
    tot = max(int(p.route_log.sum()), 1)
    return {
        "seed": seed,
        "arm": arm,
        "config": {"K": K, "d": D, "n_classes": NCLS, "h": H, "epochs": EPOCHS,
                   "n_experts": M_UNBOUNDED, "m_max": kw.get("m_max", 0),
                   "n_samples": 6000, "feedback": "random", "consolidate": True,
                   "z_novel": 5.0, "rng": "one default_rng(seed) across all tasks"},
        "ACC": R.acc(),
        "FGT": R.forgetting(),
        "acc_matrix": R.R.tolist(),
        "route_log": p.route_log.tolist(),
        "route_fraction": [round(float(c) / tot, 6) for c in p.route_log],
        "n_seen": [int(e.n_seen) for e in p.experts],
        "committed": [int(e.committed) for e in p.experts],
        "frozen": [int(e.frozen) for e in p.experts],
        "active": int(p.active),
        "eviction_log": p.eviction_log,
        "eviction_fires": len(p.eviction_log),
        "eviction_victims": [r["victim"] for r in p.eviction_log],
        "wall_seconds": round(time.time() - t0, 2),
        "lane": LANE,
    }


def load_all(seeds):
    raw = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for s in seeds:
            with open(raw_path(arm, s), encoding="utf-8") as f:
                raw[arm].append(json.load(f))
        raw[arm].sort(key=lambda r: r["seed"])
    return raw


def main():
    global N_SEEDS
    force = "--force" in sys.argv
    os.makedirs(OUT_DIR, exist_ok=True)
    seeds = list(range(SEED_START, SEED_START + N_SEEDS))

    # -- wall-time probe = the first real run (nothing wasted). The registered
    #    protocol is >=10 seeds; only if ONE run exceeds ~3 min do we shrink to 6
    #    per arm, with a prominent disclosure written into RESULTS.md.
    shrink_disclosure = None
    if force or not os.path.exists(raw_path("unbounded", seeds[0])):
        rec = run_one(seeds[0], "unbounded")
        save_json(raw_path("unbounded", seeds[0]), rec)
        print(f"[pr05] probe run seed {seeds[0]} unbounded: {rec['wall_seconds']}s  "
              f"ACC={rec['ACC']:.4f} FGT={rec['FGT']:.4f}")
        if rec["wall_seconds"] > PROBE_LIMIT_S:
            seeds = seeds[:6]
            N_SEEDS = 6
            shrink_disclosure = (f"PROTOCOL SHRINK DISCLOSURE: one E1-stream run took "
                                 f"{rec['wall_seconds']}s > {PROBE_LIMIT_S}s, so the "
                                 f"pre-declared shrink rule fired: n reduced 10 -> 6 "
                                 f"seeds per arm (seeds {seeds[0]}-{seeds[-1]}). The "
                                 f"registered protocol says >=10; this run is "
                                 f"under-powered relative to it.")

    for s in seeds:
        for arm in ARMS:
            if not force and os.path.exists(raw_path(arm, s)):
                continue
            rec = run_one(s, arm)
            save_json(raw_path(arm, s), rec)     # raw first, verdict second
            print(f"[pr05] seed {s:>2} {arm:<9} ACC={rec['ACC']:.4f} "
                  f"FGT={rec['FGT']:.4f} evictions={rec['eviction_fires']} "
                  f"({rec['wall_seconds']}s)")

    raw = load_all(seeds)
    n = len(seeds)
    acc = {a: [r["ACC"] for r in raw[a]] for a in ARMS}
    fgt = {a: [r["FGT"] for r in raw[a]] for a in ARMS}
    d_acc = [b - u for b, u in zip(acc["bounded"], acc["unbounded"])]
    d_fgt = [b - u for b, u in zip(fgt["bounded"], fgt["unbounded"])]
    acc_u_m, acc_u_c = ci95(acc["unbounded"])
    acc_b_m, acc_b_c = ci95(acc["bounded"])
    fgt_u_m, fgt_u_c = ci95(fgt["unbounded"])
    fgt_b_m, fgt_b_c = ci95(fgt["bounded"])
    d_acc_mean = acc_b_m - acc_u_m
    d_fgt_mean = fgt_b_m - fgt_u_m
    max_d_acc, max_d_fgt = max(abs(x) for x in d_acc), max(abs(x) for x in d_fgt)

    # the frozen bar (VERBATIM): seed-summary means, both metrics, threshold 0.02
    passed = abs(d_acc_mean) <= BAR and abs(d_fgt_mean) <= BAR
    verdict = "PASS" if passed else "FAIL"

    ev_fires = sum(r["eviction_fires"] for r in raw["bounded"])
    ev_victims = [v for r in raw["bounded"] for v in r["eviction_victims"]]
    identical = all(da == 0.0 and df == 0.0
                    for da, df in zip(d_acc, d_fgt))
    n_experts_b = [int((np.array(r["route_log"]) > 0).sum()) for r in raw["bounded"]]
    n_experts_u = [int((np.array(r["route_log"]) > 0).sum()) for r in raw["unbounded"]]
    frac_b = np.mean([r["route_fraction"] for r in raw["bounded"]], axis=0).tolist()
    frac_u = np.mean([r["route_fraction"] for r in raw["unbounded"]], axis=0).tolist()

    # aggregate raw ledger (the per-seed files stay on disk as the primary raws)
    save_json(os.path.join(OUT_DIR, "raw_all_seeds.json"),
              {"lane": LANE, "bar_verbatim":
               "PASS iff |ACC_bounded - ACC_unbounded| <= 0.02 AND "
               "|FGT_bounded - FGT_unbounded| <= 0.02 (seed-summary means)",
               "seeds": seeds, "arms": raw})

    lines = [
        "# PR-2026-09-03-05 — bounded-M economy lever (Policy A) — RESULTS (registered run)",
        "",
        f"- Protocol: docs/EXPERT_ECONOMY.md §3.3 Policy A + §5 operationalisation, executed",
        f"  VERBATIM on the shipped E1 stream (structured_permuted_tasks, K=5, d=24, 8 classes,",
        f"  n_samples=6000, h=48, M_unbounded=K+3=8, epochs=15, DFA, consolidate=True, z_novel=5.0).",
        f"- Arms: UNBOUNDED = shipped (m_max=0, pool M=8); BOUNDED = Policy A recruit-by-eviction",
        f"  at M_max=K=5 (pool allocation stays 8; slots >= 5 never used).",
        f"- Seeds {seeds[0]}-{seeds[-1]} (n={n}, fresh; the exploratory seeds 0-2 untouched).",
        f"- Wall-time estimate measured before the campaign: first E1-stream run = "
        f"{raw['unbounded'][0]['wall_seconds']}s; total campaign ≈ "
        f"{sum(r['wall_seconds'] for a in ARMS for r in raw[a]):.0f}s.",
    ]
    if shrink_disclosure:
        lines.append(f"- **{shrink_disclosure}**")
    lines += [
        f"- Eviction policy under test: at a capped recruit, evict the committed expert in",
        f"  [0, M_max) with the lowest lifetime routing share route_log[m]/max(sum,1),",
        f"  tie -> highest slot index (most recently recruited); reuse its slot.",
        "",
        "## Arms table (ACC-vs-M_max reporting duty; mean ± 95% CI, run_continual.ci95)",
        "",
        "| arm | M_max | ACC | FGT |",
        "|---|---|---|---|",
        f"| UNBOUNDED (shipped) | — (pool 8, {sum(n_experts_u) / n:.1f} experts used mean) "
        f"| {acc_u_m:.4f} ± {acc_u_c:.4f} | {fgt_u_m:.4f} ± {fgt_u_c:.4f} |",
        f"| BOUNDED (Policy A) | {M_MAX} ({sum(n_experts_b) / n:.1f} experts used mean) "
        f"| {acc_b_m:.4f} ± {acc_b_c:.4f} | {fgt_b_m:.4f} ± {fgt_b_c:.4f} |",
        "",
        "## Verdict arithmetic",
        "",
        f"- ΔACC = {d_acc_mean:+.6f} (bar: |ΔACC| <= {BAR})",
        f"- ΔFGT = {d_fgt_mean:+.6f} (bar: |ΔFGT| <= {BAR})",
        f"- Per-seed max |ΔACC| = {max_d_acc:.6f}; per-seed max |ΔFGT| = {max_d_fgt:.6f} "
        f"(honest per-seed spread; the bar compares seed-summary means).",
        f"- Eviction fires (bounded arm, all seeds): **{ev_fires}**"
        + (f"; victims: {ev_victims}" if ev_fires else
           " (the §2.1 measured expectation — the cap is inert at home)").rstrip(".") + ".",
        f"- Per-expert train-fraction ledger (mean over seeds, slots 0-7):",
        f"    - UNBOUNDED: {[round(x, 4) for x in frac_u]}",
        f"    - BOUNDED:   {[round(x, 4) for x in frac_b]}",
        "",
        "## Per-seed records",
        "",
        "| seed | ACC_unb | ACC_bnd | ΔACC | FGT_unb | FGT_bnd | ΔFGT | evictions (victims) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    ev_by_seed = {r["seed"]: (r["eviction_fires"], r["eviction_victims"])
                  for r in raw["bounded"]}
    for i, s in enumerate(seeds):
        ef, ev = ev_by_seed[s]
        lines.append(
            f"| {s} | {acc['unbounded'][i]:.4f} | {acc['bounded'][i]:.4f} "
            f"| {d_acc[i]:+.4f} | {fgt['unbounded'][i]:.4f} | {fgt['bounded'][i]:.4f} "
            f"| {d_fgt[i]:+.4f} | {ef} ({ev}) |")
    if passed:
        verdict_note = ("Both deltas are within the frozen ±0.02 band: on the shipped "
                        "E1 stream, bounded-M_max=K recruit-by-eviction is "
                        "indistinguishable from unbounded M in ACC and FGT.")
    else:
        verdict_note = ("At least one delta exceeds the frozen ±0.02 band: the "
                        "bounded-M claim is FALSIFIED as specified.")
    if identical:
        verdict_note += (
            f" Mechanistic account: {ev_fires} evictions fired, so the bounded arm "
            f"never left the shipped code path on this stream (Policy A guards only "
            f"fire at a capped recruit); per-seed ACC/FGT are exactly equal to the "
            f"unbounded arm (bit-identical trajectories) — the measured realization "
            f"of the §2.1 'cap is inert at home' expectation. The lever binds only "
            f"in overlap/open-world regimes, where its forgetting cost must be "
            f"measured, not assumed away.")
    else:
        verdict_note += (" The arms diverged (evictions fired and/or cap effects); "
                         "see the per-seed table for where.")
    lines += [
        "",
        f"## VERDICT: **{verdict}**",
        "",
        verdict_note,
        "",
        "Raw per-seed records (docs/RETENTION.md: raw first, verdict second): "
        "`raw_<arm>_seed_<sss>.json` in this directory (crash-safe per-seed writes, "
        "resumable), aggregated in `raw_all_seeds.json`.",
    ]
    out_md = os.path.join(OUT_DIR, "RESULTS.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[pr05] VERDICT: {verdict}  (ΔACC={d_acc_mean:+.4f}, ΔFGT={d_fgt_mean:+.4f}, "
          f"evictions={ev_fires}, max|ΔACC|={max_d_acc:.4f}, max|ΔFGT|={max_d_fgt:.4f})")
    print(f"[pr05] wrote {os.path.relpath(out_md, _ROOT)}")


if __name__ == "__main__":
    main()
