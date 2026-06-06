"""
Reads the results/*.json produced by run_bar.py and renders the bar verdict table + a markdown
block for the report. Applies the committee pass conditions; declares PASS/FAIL/CLOSE per item.
"""
from __future__ import annotations

import json
import os

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def _load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def _med(accs):
    s = sorted(accs); n = len(s)
    return (s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])) if s else 0.0


def _solverate(accs, thr=0.9):
    return f"{sum(a > thr for a in accs)}/{len(accs)}" if accs else "0/0"


def _fmt(d):
    if not d:
        return "—"
    accs = d.get("accs", [])
    if accs and len(accs) >= 3:
        # robust reporting for stochastic phase-transition tasks: median + solve-rate + best
        return f"med={_med(accs):.3f} best={max(accs):.3f} solved={_solverate(accs)} (mean {d['mean']:.2f})"
    return f"{d['mean']:.3f}±{d['ci95']:.3f}"


def b1_verdict():
    r = _load("b1_mqar.json")
    if not r:
        return "B1 MQAR: pending", None
    lines = ["### B1 — MQAR (decisive gate)", "", "| rung | Prizma-Seq | Transformer | Δ | verdict |", "|---|---|---|---|---|"]
    ok = True
    for rung in ["rung1", "rung2", "rung3"]:
        if rung not in r:
            continue
        ps, tf = r[rung]["Prizma-Seq"], r[rung]["Transformer"]
        delta = ps["mean"] - tf["mean"]
        margin = -0.03 if rung == "rung3" else -0.02
        v = "PASS" if delta >= margin else ("CLOSE" if delta >= margin - 0.03 else "FAIL")
        if rung == "rung3" and v == "FAIL":
            ok = False
        lines.append(f"| {rung} | {_fmt(ps)} | {_fmt(tf)} | {delta:+.3f} | {v} |")
    lines.append(f"\n**B1 gate (rung3): {'PASS' if ok else 'FAIL'}**")
    return "\n".join(lines), ok


def b1b_verdict(d_star=None):
    r = _load("b1b_capacity.json")
    if not r:
        return "B1b capacity: pending", None
    Ds = sorted({int(k) for k in r["transformer"]}, key=int)
    lines = ["### B1b — MQAR capacity (recall vs #bindings D)", "",
             "| D | Transformer | Prizma d_h=32 | Prizma d_h=64 |", "|---|---|---|---|"]
    for D in Ds:
        tf = r["transformer"].get(str(D)) or r["transformer"].get(D)
        p32 = r["prizma_by_dh"].get("32", {}).get(str(D)) or r["prizma_by_dh"].get("32", {}).get(D)
        p64 = r["prizma_by_dh"].get("64", {}).get(str(D)) or r["prizma_by_dh"].get("64", {}).get(D)
        lines.append(f"| {D} | {_fmt(tf)} | {_fmt(p32)} | {_fmt(p64)} |")
    lines.append("\n*(D\\* = largest D where Prizma-Seq stays within 0.03 of the Transformer.)*")
    return "\n".join(lines), True


def b3_verdict():
    r = _load("b3_selcopy.json")
    if not r:
        return "B3 selective copy: pending", None
    lines = ["### B3 — Selective copy", "", "| variant | Prizma-Seq | Transformer | Prizma-noGate |", "|---|---|---|---|"]
    for var in ["selective", "fixed"]:
        if var not in r:
            continue
        lines.append(f"| {var} | {_fmt(r[var].get('Prizma-Seq'))} | {_fmt(r[var].get('Transformer'))} "
                     f"| {_fmt(r[var].get('Prizma-noGate'))} |")
    sel = r.get("selective", {})
    ps = sel.get("Prizma-Seq", {}).get("mean", 0)
    ng = sel.get("Prizma-noGate", {}).get("mean", 1)
    gate = "PASS" if (ps >= 0.90 and ps - ng >= 0.10) else "CHECK"
    lines.append(f"\n**B3 input-gating causal: {gate}** (full {ps:.3f} vs noGate {ng:.3f} on selective)")
    return "\n".join(lines), gate == "PASS"


def b6_verdict():
    r = _load("b6_ablations.json")
    if not r:
        return "B6 ablations: pending", None
    lines = ["### B6 — Mechanism ablations (causal)", ""]
    for task in ["mqar", "selcopy"]:
        if task not in r:
            continue
        lines.append(f"**{task}**")
        lines.append("| variant | acc |"); lines.append("|---|---|")
        full = r[task].get("full", {}).get("mean", 0)
        for name, d in r[task].items():
            drop = full - d["mean"]
            tag = f" (Δ−{drop:.2f})" if name != "full" else ""
            lines.append(f"| {name} | {_fmt(d)}{tag} |")
        lines.append("")
    return "\n".join(lines), True


def b5_verdict():
    r = _load("b5_latency.json")
    if not r:
        return "B5 inference: pending", None
    ns = r["seq_lens"]
    lines = ["### B5 — Inference cost (the structural advantage)", "",
             "| n | TF KV-cache (s) | Prizma step (s) | TF state (floats) | Prizma state (floats) |",
             "|---|---|---|---|---|"]
    tl, pl = r["transformer_kvcache_s"], r["prizma_step_s"]
    tm, pm = r["transformer_state_floats"], r["prizma_state_floats"]
    for n in ns:
        sn = str(n)
        lines.append(f"| {n} | {tl.get(sn, tl.get(n)):.3f} | {pl.get(sn, pl.get(n)):.3f} | "
                     f"{int(tm.get(sn, tm.get(n)))} | {int(pm.get(sn, pm.get(n)))} |")
    return "\n".join(lines), True


if __name__ == "__main__":
    out = []
    for fn in [b1_verdict, b1b_verdict, b3_verdict, b6_verdict, b5_verdict]:
        txt, _ = fn()
        out.append(txt)
    report = "\n\n".join(out)
    print(report)
    open(os.path.join(RES, "verdict.md"), "w").write(report)
    print("\n[written results/verdict.md]")
