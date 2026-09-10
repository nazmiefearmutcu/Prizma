"""Campaign 2026-09-11 — paired analysis of the manyblock probe (exploratory only).

Reads:
  results/exploratory/manyblock_probe_2026-09-11/probe_off.json       (this session, OFF baseline)
  results/exploratory/manyblock_probe_2026-09-11/probe_cascade.json   (this session, tissue cascade)
  results/exploratory/manyblock_probe_2026-09-09/probe.json           (historical n=2 OFF)
Writes:
  results/exploratory/manyblock_probe_2026-09-11/ANALYSIS.md (+ prints)
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


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def cell(tr, tag):
    b = tr["blocks"][tag]
    return b["bpc_A"], b["bpc_B"]


def metrics(tr):
    b = tr["blocks"]
    rec = {s: b["post_C"][s] - b["post_D"][s] for s in ("bpc_A", "bpc_B")}
    dmg = b["post_C"]["bpc_B"] - b["post_B"]["bpc_B"]
    ret = b["post_E"]["bpc_A"] - b["post_B"]["bpc_A"]
    return {"recovery_B": rec["bpc_B"], "damage_C": dmg, "A_retention_postE": ret,
            "bpcB_postB": b["post_B"]["bpc_B"], "bpcB_postC": b["post_C"]["bpc_B"],
            "bpcB_postD": b["post_D"]["bpc_B"], "bpcB_postE": b["post_E"]["bpc_B"],
            "bpcA_postE": b["post_E"]["bpc_A"], "cre_t_postE": b["post_E"]["bpc_Cret"],
            "recruits": tr["ledger_tail"]["recruits"], "evictions": tr["ledger_tail"]["evictions"]}


def fmt(v):
    return f"{v:+.4f}" if isinstance(v, float) else str(v)


def main():
    off = load(os.path.join(NEW, "probe_off.json"))
    cas = load(os.path.join(NEW, "probe_cascade.json"))
    hist = load(HIST)

    lines = ["# manyblock probe 2026-09-11 — paired exploratory analysis",
             "",
             "NOT a claim (LANE-EXPLORATORY). n=2 seeds 0-1. OFF = baseline probe; CASCADE = the",
             "Lane-1 tissue multi-timescale lever at kappa=0.05, delta=0.10. Recovery_B =",
             "bpc_B(postC) - bpc_B(postD) (higher = more recovery on the literal revisit).",
             "Damage_C = bpc_B(postC) - bpc_B(postB) (lower = less B-damage from C).",
             ""]

    # historical determinism canary (OFF path must reproduce the committed 2026-09-09 run)
    canary_ok = True
    for s in ("s0", "s1"):
        for tag, vals in off["results"][s]["blocks"].items():
            hv = hist["results"][s]["blocks"].get(tag)
            if hv is None:
                canary_ok = False
                continue
            for k in ("bpc_A", "bpc_B", "bpc_Cret"):
                if abs(vals[k] - hv[k]) > 0:
                    canary_ok = False
    lines.append(f"- Historical canary (OFF reproduces 2026-09-09 probe.json exactly): "
                 f"**{'PASS' if canary_ok else 'FAIL'}**")
    lines.append("")

    lines.append("| seed | arm | recovery_B | damage_C | A_ret(postE-postB) | bpcB postB→C→D→E | recruits |")
    lines.append("|---|---|---|---|---|---|---|")
    summary = {}
    for arm, doc in (("OFF", off), ("CASCADE", cas)):
        for s in ("s0", "s1"):
            m = metrics(doc["results"][s])
            lines.append(f"| {s} | {arm} | {m['recovery_B']:+.4f} | {m['damage_C']:+.4f} | "
                         f"{m['A_retention_postE']:+.4f} | {m['bpcB_postB']:.3f}→"
                         f"{m['bpcB_postC']:.3f}→{m['bpcB_postD']:.3f}→{m['bpcB_postE']:.3f} | "
                         f"{m['recruits']} |")
            summary.setdefault(arm, []).append(m)

    lines.append("")
    for arm in ("OFF", "CASCADE"):
        ms = summary[arm]
        rec = sum(m["recovery_B"] for m in ms) / len(ms)
        dmg = sum(m["damage_C"] for m in ms) / len(ms)
        lines.append(f"- **{arm}** mean recovery_B {rec:+.4f}, mean damage_C {dmg:+.4f}")
    dro = sum(m["recovery_B"] for m in summary["CASCADE"]) / 2 - \
        sum(m["recovery_B"] for m in summary["OFF"]) / 2
    ddm = sum(m["damage_C"] for m in summary["CASCADE"]) / 2 - \
        sum(m["damage_C"] for m in summary["OFF"]) / 2
    lines.append("")
    lines.append(f"**Paired deltas (CASCADE - OFF, mean of seeds): recovery {dro:+.4f} bpc, "
                 f"damage {ddm:+.4f} bpc.**")
    lines.append("")
    lines.append("Reading (exploratory, n=2): the frozen dose shifts BOTH metrics slightly in the "
                 "protective direction (a little less damage) at a small cost in revisit recovery — "
                 "the same trade direction as the registered lr schedule (PR-21), far smaller in "
                 "magnitude and within seed/run noise at this n. No claim is staked; the lever stays "
                 "default-OFF. A future registration would need a dose/target design pass first.")

    out = os.path.join(NEW, "ANALYSIS.md")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
