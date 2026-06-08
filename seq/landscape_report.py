"""TORCH-FREE renderer: the landscape verdict JSONs -> a Council-3 Pareto MARKDOWN report.

WHAT THIS IS (and is NOT).
  seq/landscape.py's run_landscape writes results/gpu_landscape.json (the RECALL leg, accuracy,
  higher-is-better) and run_landscape_charlm writes results/gpu_landscape_charlm.json (the CHAR-LM leg,
  BPC, LOWER-is-better). Each stores, under res["landscape_report"] / res["landscape_charlm_report"], a
  dict of {tasks, negative_control, meta, smoke, ...}; every task block carries the powered Pareto table
  + Holm-corrected pairwise verdicts that seq.landscape.landscape_verdict produced.

  This module is a PURE RENDERER. It READS those JSONs and emits a markdown string. It does NO torch
  import and NO training — so it stays importable + runnable in milliseconds, on any machine, exactly
  like landscape.py's pure verdict layer. It renders ONLY what the JSON contains: no fabricated numbers,
  no spin, and — crucially — NO per-FLOP / efficiency claims, because this layer only has accuracy/BPC,
  not FLOPs. Per-FLOP "dramatic" framing stays a CONDITIONAL caveat carried in the scope rider, never an
  assertion this renderer makes on its own.

  render_report(landscape_json, charlm_json) -> markdown str. A missing JSON => that leg is OMITTED with
  an explicit note (never a crash). main() is the argparse CLI (--recall-json / --charlm-json / --out),
  and an unknown flag exits non-zero (arg-guard), mirroring seq/landscape.py's main.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# The verbatim SCOPE RIDER. Emitted as-is so the honest-scope framing travels WITH every rendered report
# (this is the one place a per-FLOP claim is even mentioned, and only as a CONDITIONAL caveat).
SCOPE_RIDER = ("Scope: ≤2M params (+1 confirmation 10–50M); char-LM + diagnostics; "
               "NOT a frontier/MMLU/long-context claim; per-FLOP 'dramatic' stays conditional unless "
               "all axes hold + powered.")

# The loud smoke banner (rendered when a leg's meta marks the run as a plumbing-only smoke).
SMOKE_BANNER = "⚠️ SMOKE — plumbing only, NOT a scientific result"

DEFAULT_RECALL_JSON = os.path.join(os.path.dirname(__file__), "..", "results", "gpu_landscape.json")
DEFAULT_CHARLM_JSON = os.path.join(os.path.dirname(__file__), "..", "results", "gpu_landscape_charlm.json")

_EMDASH = "—"
_VERDICT_LABELS = ("BEATS", "PARITY", "WORSE", "INCONCLUSIVE")


# ------------------------------------------------------------------ helpers --
def _kind(name):
    """The short arm kind/label for a full arm name (e.g. 'TF.d48L1H2' -> 'TF', 'Prizma.dX_feat..' ->
    'Prizma'). The persisted names are '<Kind>.<scale...>', so the prefix before the first '.' is the
    display label (matches landscape.py's own `name.split('.')[0]` convention)."""
    if not isinstance(name, str):
        return str(name)
    return name.split(".")[0]


def _fmt_num(x, nd=3):
    """Render a number to nd decimals, or the em-dash if it is None/non-numeric (no invented values)."""
    if x is None:
        return _EMDASH
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return _EMDASH


def _fmt_params(p):
    """Render a param count with thousands separators, or em-dash if absent (spreads disclosed, never
    hidden; a missing count is shown as '—', not faked)."""
    if p is None:
        return _EMDASH
    try:
        return f"{int(p):,}"
    except (TypeError, ValueError):
        return _EMDASH


def _fmt_mean_ci(row):
    """Render 'mean ± CI95' for a pareto row, tolerating a missing/short ci95 (-> just the mean, or '—').
    The CI is rendered as the half-width-style 'mean (lo, hi)' so the disclosed interval is explicit."""
    mean = row.get("mean")
    ci = row.get("ci95")
    if mean is None:
        return _EMDASH
    m = _fmt_num(mean)
    if isinstance(ci, (list, tuple)) and len(ci) == 2 and ci[0] is not None and ci[1] is not None:
        return f"{m} (CI95 {_fmt_num(ci[0])}, {_fmt_num(ci[1])})"
    return m


def _metric_label(lower_is_better):
    """The metric name + direction marker for a leg's column header."""
    if lower_is_better:
        return "BPC (lower better)"
    return "accuracy"


def _load_leg(path, report_key):
    """Load one leg's report block from its JSON file. Returns (block_or_None, note).
    note is None on success, else a human note explaining why the leg is omitted (missing file, bad
    JSON, or the expected report key absent) — rendered honestly, never silently dropped."""
    if not path or not os.path.exists(path):
        return None, f"results JSON not found at `{path}` — section omitted."
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return None, f"could not read `{path}` ({type(e).__name__}) — section omitted."
    block = data.get(report_key)
    if not isinstance(block, dict):
        return None, f"`{path}` has no `{report_key}` block — section omitted."
    return block, None


# ------------------------------------------------------------------ rendering --
def _sorted_rows(rows, lower_is_better):
    """Return the pareto rows ranked BEST-FIRST by mean (descending for accuracy, ascending for a
    lower-is-better loss/BPC). landscape_verdict already persists them sorted, but a hand-built or older
    JSON might not be, so the renderer enforces best-first itself (stable; rows with no mean sink last).
    No numbers are changed — only the row ORDER — so this never invents or alters a value."""
    def key(r):
        m = r.get("mean")
        # rows missing a mean sort to the end regardless of direction.
        if m is None:
            return (1, 0.0)
        return (0, float(m) if lower_is_better else -float(m))
    return sorted(rows, key=key)


def _render_pareto_table(task_block):
    """Render ONE task's Pareto table as a markdown table (rows ranked best-first). Columns:
        Arm | Params | mean ± CI95 | solve_rate | Verdict-vs-Prizma
    The candidate (Prizma) row is marked '**Prizma** (cand)' and carries NO self-verdict ('—').
    Returns (markdown_lines, verdict_counts) where verdict_counts tallies the pairwise labels."""
    pairwise = task_block.get("pairwise") or {}
    cand_key = task_block.get("cand_key")
    lower = bool(task_block.get("lower_is_better", False))
    rows = _sorted_rows(task_block.get("pareto_table") or [], lower)
    metric = _metric_label(lower)

    out = []
    out.append(f"| Arm | Params | mean ± CI95 ({metric}) | solve_rate | Verdict vs Prizma |")
    out.append("| --- | --- | --- | --- | --- |")

    counts = {k: 0 for k in _VERDICT_LABELS}
    for row in rows:
        name = row.get("name")
        is_cand = (name == cand_key)
        label = _kind(name)
        if is_cand:
            arm_cell = f"**{label}** (cand)"
            verdict_cell = _EMDASH                      # the candidate has no self-verdict
        else:
            arm_cell = label
            pv = pairwise.get(name) or {}
            verdict = pv.get("verdict")
            if verdict in counts:
                counts[verdict] += 1
            verdict_cell = verdict if verdict else _EMDASH
        params_cell = _fmt_params(row.get("params"))
        meanci_cell = _fmt_mean_ci(row)
        solve_cell = _fmt_num(row.get("solve_rate"), nd=2)
        out.append(f"| {arm_cell} | {params_cell} | {meanci_cell} | {solve_cell} | {verdict_cell} |")
    return out, counts


def _render_negctrl(neg):
    """Render the negative-control PASS/FAIL line honestly from the negative_control block (two
    byte-identical arms must NOT differ; pass=True => PASS)."""
    if not isinstance(neg, dict):
        return "_Negative control: not recorded._"
    passed = neg.get("pass")
    p = neg.get("p_value")
    sig = neg.get("significant")
    verdict = "PASS" if passed else "FAIL"
    pstr = _fmt_num(p) if p is not None else _EMDASH
    return (f"**Negative control** (two byte-identical arms must NOT differ): "
            f"p={pstr}, significant={bool(sig)} → **{verdict}**")


def _render_leg(title, block, *, default_lower):
    """Render one leg (recall or char-LM): title, smoke banner (if smoke), per-task tables, and the
    negative-control line. Returns (markdown_lines, combined_counts) where combined_counts tallies the
    pairwise verdicts across ALL tasks in this leg (fuel for the combined Pareto picture)."""
    lines = [f"## {title}"]
    meta = block.get("meta") or {}
    scale = meta.get("scale")
    seeds = meta.get("seeds")
    margin = meta.get("margin")
    bits = []
    if scale:
        bits.append(f"scale `{scale}`")
    if seeds is not None:
        bits.append(f"{len(seeds) if isinstance(seeds, (list, tuple)) else seeds} seeds")
    if margin is not None:
        bits.append(f"margin {_fmt_num(margin)}")
    if meta.get("corpus"):
        bits.append(f"corpus `{meta['corpus']}`")
    if meta.get("random_baseline_bpc") is not None:
        bits.append(f"random-BPC {_fmt_num(meta['random_baseline_bpc'])}")
    if bits:
        lines.append("_" + ", ".join(bits) + "._")

    if block.get("smoke"):
        lines.append("")
        lines.append(f"> {SMOKE_BANNER}")

    leg_counts = {k: 0 for k in _VERDICT_LABELS}
    tasks = block.get("tasks") or {}
    if not tasks:
        lines.append("")
        lines.append("_No tasks recorded in this leg._")
    for task_name, task_block in tasks.items():
        lower = bool(task_block.get("lower_is_better", default_lower))
        lines.append("")
        lines.append(f"### Task: {task_name} — metric: {_metric_label(lower)}")
        tbl, counts = _render_pareto_table(task_block)
        lines.extend(tbl)
        for k in leg_counts:
            leg_counts[k] += counts[k]
        # disclose any skipped/unrunnable arms honestly.
        unrunnable = task_block.get("unrunnable") or {}
        if unrunnable:
            for kind, info in unrunnable.items():
                reason = info.get("reason") if isinstance(info, dict) else info
                lines.append(f"- _skipped `{kind}`: {reason}_")

    lines.append("")
    lines.append(_render_negctrl(block.get("negative_control")))
    return lines, leg_counts


def _render_pareto_picture(per_leg_counts):
    """The combined 'Pareto picture': across all legs, count Prizma's BEATS/PARITY/WORSE/INCONCLUSIVE vs
    each baseline and state plainly whether Prizma is Pareto-competitive. NO per-FLOP / efficiency claim
    (this renderer only has accuracy/BPC): efficiency framing stays in the conditional scope rider."""
    total = {k: 0 for k in _VERDICT_LABELS}
    for counts in per_leg_counts:
        for k in _VERDICT_LABELS:
            total[k] += counts.get(k, 0)
    n = sum(total.values())

    lines = ["## Pareto picture (across all legs)"]
    lines.append(f"- BEATS: {total['BEATS']}")
    lines.append(f"- PARITY: {total['PARITY']}")
    lines.append(f"- WORSE: {total['WORSE']}")
    lines.append(f"- INCONCLUSIVE: {total['INCONCLUSIVE']}")

    if n == 0:
        verdict = ("No pairwise comparisons were rendered (no leg data), so no Pareto-competitiveness "
                   "statement can be made.")
    elif total["WORSE"] == 0 and (total["BEATS"] + total["PARITY"]) > 0:
        verdict = ("On the rendered accuracy/BPC axes Prizma is at-or-above every baseline it was tested "
                   "against (no WORSE), i.e. Pareto-competitive on these axes. "
                   f"({total['INCONCLUSIVE']} comparison(s) remain under-powered/inconclusive.)")
    elif total["WORSE"] > 0:
        verdict = (f"Prizma is WORSE than a baseline on {total['WORSE']} comparison(s); it is NOT "
                   "uniformly Pareto-competitive on the rendered axes — see the per-task tables.")
    else:
        verdict = ("All rendered comparisons are inconclusive/under-powered; no Pareto-competitiveness "
                   "claim is supported.")
    lines.append("")
    lines.append(verdict)
    lines.append("")
    lines.append("_This summary uses ONLY accuracy/BPC. No per-FLOP or efficiency claim is made here; "
                 "per-FLOP framing stays conditional per the scope rider above._")
    return lines


def render_report(landscape_json=None, charlm_json=None):
    """Load whichever leg JSONs exist and return a Council-3 Pareto markdown report.

    Args:
      landscape_json : path to the RECALL leg JSON (default results/gpu_landscape.json). Missing -> the
                       recall section is omitted with a note.
      charlm_json    : path to the CHAR-LM leg JSON (default results/gpu_landscape_charlm.json). Missing
                       -> the char-LM section is omitted with a note.

    Returns: a markdown string. Never raises on missing/partial JSON (renders a note instead). Renders
    ONLY what the JSON contains (no invented numbers; per-FLOP claims out)."""
    recall_path = landscape_json if landscape_json is not None else DEFAULT_RECALL_JSON
    charlm_path = charlm_json if charlm_json is not None else DEFAULT_CHARLM_JSON

    recall_block, recall_note = _load_leg(recall_path, "landscape_report")
    charlm_block, charlm_note = _load_leg(charlm_path, "landscape_charlm_report")

    lines = []
    lines.append("# Prizma-Seq v2 — SOTA Landscape Council-3 Pareto Report")
    lines.append("")
    lines.append(f"> {SCOPE_RIDER}")
    lines.append("")
    lines.append("Head-to-head: TF vs **Prizma** (cand) vs GLA vs Mamba-2, at matched (d, L, H) scale. "
                 "Verdicts are powered (Student-t CIs / TOST) and Holm-corrected per pairwise family. "
                 "Rendered from the persisted verdict JSONs — no training, no torch, no invented "
                 "numbers.")

    per_leg_counts = []

    # ---- RECALL leg ----
    lines.append("")
    if recall_block is not None:
        leg_lines, counts = _render_leg("Recall leg (diagnostics — accuracy, higher better)",
                                        recall_block, default_lower=False)
        lines.extend(leg_lines)
        per_leg_counts.append(counts)
    else:
        lines.append("## Recall leg (diagnostics)")
        lines.append(f"_Omitted: {recall_note}_")

    # ---- CHAR-LM leg ----
    lines.append("")
    if charlm_block is not None:
        leg_lines, counts = _render_leg("Char-LM leg (language modeling — BPC, lower better)",
                                        charlm_block, default_lower=True)
        lines.extend(leg_lines)
        per_leg_counts.append(counts)
    else:
        lines.append("## Char-LM leg (language modeling)")
        lines.append(f"_Omitted: {charlm_note}_")

    # ---- combined Pareto picture ----
    lines.append("")
    lines.extend(_render_pareto_picture(per_leg_counts))

    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------- main --
def _build_parser():
    """Argparse parser for the renderer CLI. Recognizes ONLY --recall-json / --charlm-json / --out;
    argparse rejects any unknown flag with a usage message + non-zero exit (arg-guard). Kept as its own
    helper so the guard is unit-testable."""
    p = argparse.ArgumentParser(
        prog="landscape_report",
        description="TORCH-FREE renderer: turn the landscape verdict JSONs (recall + char-LM) into a "
                    "Council-3 Pareto markdown report. Reads only; never trains. An unknown flag is "
                    "rejected (non-zero exit).")
    p.add_argument("--recall-json", default=DEFAULT_RECALL_JSON,
                   help="path to the recall-leg results JSON (default results/gpu_landscape.json)")
    p.add_argument("--charlm-json", default=DEFAULT_CHARLM_JSON,
                   help="path to the char-LM-leg results JSON (default results/gpu_landscape_charlm.json)")
    p.add_argument("--out", default=None,
                   help="write the markdown report to PATH (default: stdout)")
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(argv)   # argparse SystemExits non-zero on an unknown flag
    md = render_report(landscape_json=args.recall_json, charlm_json=args.charlm_json)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md)
    else:
        sys.stdout.write(md)


if __name__ == "__main__":
    main()
