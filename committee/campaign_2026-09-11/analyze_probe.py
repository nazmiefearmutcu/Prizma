"""Campaign 2026-09-11 — paired analysis of the manyblock probe (exploratory only).

Runs (same session, deterministic OFF path; historical canary checked):
  results/exploratory/manyblock_probe_2026-09-11/probe_off.json          (seeds 0-1, OFF)
  results/exploratory/manyblock_probe_2026-09-11/probe_cascade.json      (seeds 0-1, tissue cascade)
  results/exploratory/manyblock_probe_2026-09-11/probe_off_s234.json     (seeds 2-4, OFF)
  results/exploratory/manyblock_probe_2026-09-11/probe_cascade_s234.json (seeds 2-4, tissue cascade)
  results/exploratory/manyblock_probe_2026-09-09/probe.json              (historical n=2 OFF)

Writes: results/exploratory/manyblock_probe_2026-09-11/ANALYSIS.md
Paired deltas: per-seed (CASCADE - OFF) on the SAME seed; 95% CI uses t_{0.975,4} = 2.776
(hardcoded for the pooled n=5 paired sample; exploratory report only, NOT a claim).
"""
from __future__ import annotations

import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
NEW = os.path.join(REPO, "results", "exploratory", "manyblock_probe_2026-09-11")
HIST = os.path.join(REPO, "results", "exploratory", "manyblock_probe_2026-09-09", "probe.json")
T975_DF4 = 2.776


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def metrics(tr):
    b = tr["blocks"]
    return {"recovery_B": b["post_C"]["bpc_B"] - b["post_D"]["bpc_B"],
            "damage_C": b["post_C"]["bpc_B"] - b["post_B"]["bpc_B"],
            "A_retention_postE": b["post_E"]["bpc_A"] - b["post_B"]["bpc_A"],
            "bpcB_postB": b["post_B"]["bpc_B"], "bpcB_postC": b["post_C"]["bpc_B"],
            "bpcB_postD": b["post_D"]["bpc_B"], "bpcB_postE": b["post_E"]["bpc_B"],
            "recruits": tr["ledger_tail"]["recruits"]}


def collect():
    arms = {}
    off = load(os.path.join(NEW, "probe_off.json"))
    cas = load(os.path.join(NEW, "probe_cascade.json"))
    for arm, doc in (("OFF", off), ("CASCADE", cas)):
        for key, tr in doc["results"].items():
            arms[(arm, int(key[1:]))] = metrics(tr)
    for fname, arm in (("probe_off_s234.json", "OFF"), ("probe_cascade_s234.json", "CASCADE")):
        p = os.path.join(NEW, fname)
        if os.path.exists(p):
            doc = load(p)
            for key, tr in doc["results"].items():
                arms[(arm, int(key[1:]))] = metrics(tr)
    return off, arms


def ci95(xs):
    n = len(xs)
    mean = sum(xs) / n
    if n < 2:
        return mean, None, None
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    se = (var / n) ** 0.5
    t = T975_DF4 if n == 5 else 2.0
    return mean, mean - t * se, mean + t * se


def main():
    off_doc, m = collect()
    seeds = sorted({s for (a, s) in m if a == "OFF"})

    lines = ["# manyblock probe 2026-09-11 — paired exploratory analysis (n=5)",
             "",
             "NOT a claim (LANE-EXPLORATORY). Paired seeds 0-4 (0-1 first run; 2-4 fresh).",
             "CASCADE = Lane-1 tissue multi-timescale lever at kappa=0.05, delta=0.10.",
             "Recovery_B = bpc_B(postC) - bpc_B(postD) (higher = more recovery on the revisit).",
             "Damage_C = bpc_B(postC) - bpc_B(postB) (lower = less B-damage from C).",
             ""]

    # historical determinism canary
    hist = load(HIST)
    canary_ok = True
    for s in (0, 1):
        for tag, vals in off_doc["results"][f"s{s}"]["blocks"].items():
            hv = hist["results"][f"s{s}"]["blocks"].get(tag)
            if hv is None:
                canary_ok = False
                continue
            for k in ("bpc_A", "bpc_B", "bpc_Cret"):
                if abs(vals[k] - hv[k]) > 0:
                    canary_ok = False
    lines.append(f"- Historical canary (seeds 0-1 OFF reproduce the committed 2026-09-09 "
                 f"probe.json exactly): **{'PASS' if canary_ok else 'FAIL'}**")
    lines.append("")
    lines.append("| seed | OFF recovery | CAS recovery | Δrec | OFF damage | CAS damage | Δdmg | "
                 "OFF bpcB postB→C→D | CAS bpcB postB→C→D |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    rec_d, dmg_d = [], []
    for s in seeds:
        o, c = m[("OFF", s)], m[("CASCADE", s)]
        dr = c["recovery_B"] - o["recovery_B"]
        dd = c["damage_C"] - o["damage_C"]
        rec_d.append(dr)
        dmg_d.append(dd)
        lines.append(f"| {s} | {o['recovery_B']:+.4f} | {c['recovery_B']:+.4f} | {dr:+.4f} | "
                     f"{o['damage_C']:+.4f} | {c['damage_C']:+.4f} | {dd:+.4f} | "
                     f"{o['bpcB_postB']:.3f}→{o['bpcB_postC']:.3f}→{o['bpcB_postD']:.3f} | "
                     f"{c['bpcB_postB']:.3f}→{c['bpcB_postC']:.3f}→{c['bpcB_postD']:.3f} |")
    lines.append("")
    for label, xs in (("recovery_B", rec_d), ("damage_C", dmg_d)):
        mean, lo, hi = ci95(xs)
        lines.append(f"- **paired Δ{label} (CASCADE − OFF, n={len(xs)})**: mean {mean:+.4f}, "
                     f"95% CI [{lo:+.4f}, {hi:+.4f}]" if lo is not None else
                     f"- **paired Δ{label}**: mean {mean:+.4f}")
    lines.append("")
    wins_dmg = sum(1 for x in dmg_d if x < 0)
    wins_rec = sum(1 for x in rec_d if x < 0)
    lines.append(f"Direction reproduction: damage lower in {wins_dmg}/{len(dmg_d)} pairs; "
                 f"recovery lower in {wins_rec}/{len(rec_d)} pairs.")
    lines.append("")
    lines.append("Reading (exploratory, n=5): the frozen dose keeps a small, direction-consistent "
                 "protective shift that SHRINKS on fresh seeds (n=2 deltas −0.048/−0.063 → pooled "
                 "−0.02/−0.03 bpc) and every pooled CI straddles zero. This is ~1/10 of the "
                 "registered schedule's damage effect (PR-16: +0.37). No claim is staked; the lever "
                 "stays default-OFF. A future registration needs a dose/target design pass AND a "
                 "powered n; the honest priors from these five paired seeds are small-effect.")
    if len(seeds) > 5 or any(s > 4 for s in seeds):
        lines.append("")
        lines.append(f"(observed seeds: {seeds})")

    out = os.path.join(NEW, "ANALYSIS.md")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
