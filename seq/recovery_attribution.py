"""
PR-2026-09-03-21 CLAIM RUNNER — the revisit-recovery attribution (closes the attribution
matrix; a CONFIRMATORY ANALYSIS of three registered artifacts; NO training). Executes the
FROZEN pre-registration `docs/preregistry/2026-09-11-recovery-attribution.md` VERBATIM.

  python seq/recovery_attribution.py --powered-cpu   # the analysis (CPU-native, seconds)
  python seq/recovery_attribution.py --smoke         # identical; separate ledger (parity)

RECOVERIES per seed s in {0..4}: bpc_B(post_C) - bpc_B(post_D), read from THREE registered
ledgers:
  COLUMN:   results/manyblock_PR-2026-09-03-14/powered.json   claim.EX.s{s}
  SCHED-TF: results/windowtf_sched_PR-2026-09-03-17/powered.json claim.WINDOW-TF.s{s}
  PLAIN-TF: results/windowtf_manyblock_PR-2026-09-03-15/powered.json claim.WINDOW-TF.s{s}

ATTRIBUTION (doc §3, CI-positional, NO small-p gate):
  C1 = mean(SCHED-TF recovery) - mean(COLUMN recovery)
  C2 = mean(PLAIN-TF recovery) - mean(COLUMN recovery)
  A CI "establishes a gap" iff it excludes 0 AND |point| >= 0.10.
  C1 established, C2 not  -> SCHED-ADVANTAGED (the schedule also carries recovery)
  C2 PLAIN-side established (PLAIN > COLUMN by >= 0.10) -> TISSUE-COSTS-RECOVERY
  C2 COLUMN-side established (COLUMN > PLAIN by >= 0.10) -> TISSUE-AMPLIFIES-RECOVERY
  both parities -> RECOVERY-UNIVERSAL (recovery is a property of re-encounter training)
Descriptive: all three recovery distributions + the absolute post_D BPCs (quality,
reported not gated).
"""
from __future__ import annotations

import argparse
import os
import sys

try:
    from .stats import t_sf  # noqa: F401  (parity)
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from seq.stats import t_sf

import seq.prizma_lm_claim as plc


REGISTRY_ID = "PR-2026-09-03-21"
LEDDIR = "recovery_attr_PR-2026-09-03-21"
SMOKE_BASENAME = "smoke.json"
POWERED_BASENAME = "powered.json"

CLAIM_SEEDS = plc.CLAIM_SEEDS
GAP_POINT = 0.10
ALPHA = plc.ALPHA

SOURCES = {
    "COLUMN": ("manyblock_PR-2026-09-03-14", "claim.EX"),
    "SCHED-TF": ("windowtf_sched_PR-2026-09-03-17", "claim.WINDOW-TF"),
    "PLAIN-TF": ("windowtf_manyblock_PR-2026-09-03-15", "claim.WINDOW-TF"),
}


def _results_root() -> str:
    root = os.environ.get("PRIZMA_RESULTS", os.path.join(os.path.dirname(__file__), "..", "results"))
    return os.path.abspath(root)


def _default_results_path(smoke: bool) -> str:
    return os.path.join(_results_root(), LEDDIR, SMOKE_BASENAME if smoke else POWERED_BASENAME)


def _ledger_path(subdir: str) -> str:
    return os.path.join(_results_root(), subdir, POWERED_BASENAME)


def resolve_results_path(explicit=None, *, smoke: bool = False, force_smoke_path: bool = False) -> str:
    path = explicit if explicit else _default_results_path(smoke)
    if smoke and not force_smoke_path:
        powered = _default_results_path(smoke=False)
        if os.path.normcase(os.path.abspath(path)) == os.path.normcase(os.path.abspath(powered)):
            raise SystemExit(
                f"refusing: a --smoke run was pointed at the powered ledger ({powered}). Pass "
                f"--out <path> or --force-smoke-path (ledger separation, pattern parity).")
    return path


needs_cuda = plc.needs_cuda   # pattern parity: the analysis is CPU-native


# ==================================================================== PURE: recoveries + attribution
def recoveries(recs):
    """Per-seed revisit recovery: bpc_B(post_C) - bpc_B(post_D). Positive = recovered."""
    return [r["bpc_B_postC"] - r["bpc_B_postD"] for r in recs]


def _gap_position(delta, ci):
    """CI-positional attribution for one named delta: ESTABLISHED iff the CI excludes 0
    AND |point| >= GAP_POINT; else parity-at-this-resolution."""
    excludes_zero = (ci[0] > 0) or (ci[1] < 0)
    if excludes_zero and abs(delta) >= GAP_POINT:
        return "ESTABLISHED"
    return "PARITY"


def attribution(col_rec, sched_rec, plain_rec, *, alpha=ALPHA):
    """Frozen §3 attribution. Each arg = per-seed recovery list (see recoveries())."""
    col, sch, pln = recoveries(col_rec), recoveries(sched_rec), recoveries(plain_rec)
    c1 = plc._welch_margin(sch, col, 0.0)
    c1 = {"delta": c1["delta"], "ci": c1["ci"],
          "position": _gap_position(c1["delta"], c1["ci"]),
          "note": "delta = mean(SCHED-TF recovery) - mean(COLUMN recovery)"}
    c2 = plc._welch_margin(pln, col, 0.0)
    c2 = {"delta": c2["delta"], "ci": c2["ci"],
          "position": _gap_position(c2["delta"], c2["ci"]),
          "note": "delta = mean(PLAIN-TF recovery) - mean(COLUMN recovery)"}

    def _side(delta):
        return "higher" if delta > 0 else "lower"

    if c1["position"] == "ESTABLISHED":
        c1["attribution"] = f"SCHED-ADVANTAGED ({_side(c1['delta'])} recovery)"
    else:
        c1["attribution"] = "PARITY"
    if c2["position"] == "ESTABLISHED":
        c2["attribution"] = (f"PLAIN-ADVANTAGED ({_side(c2['delta'])} recovery) => "
                             "TISSUE-COSTS-RECOVERY" if c2["delta"] < 0 else
                             f"COLUMN-ADVANTAGED ({_side(c2['delta'])} recovery) => "
                             "TISSUE-AMPLIFIES-RECOVERY")
    else:
        c2["attribution"] = "PARITY"

    if c1["attribution"] == "PARITY" and c2["attribution"] == "PARITY":
        outcome = "RECOVERY-UNIVERSAL"
        verdict_text = ("RECOVERY-UNIVERSAL — every form (column, scheduled control, plain "
                        "control) recovers the re-encountered domain at indistinguishable "
                        "magnitude: recovery is a property of re-encounter training itself. "
                        "THE ATTRIBUTION MATRIX CLOSES: damage = schedule-carried; recovery = "
                        "universal; routing ledger + purity = tissue-only (doc §4).")
    elif c1["attribution"].startswith("SCHED-ADVANTAGED"):
        outcome = "SCHED-ADVANTAGED"
        verdict_text = ("SCHED-ADVANTAGED — the schedule also carries the revisit recovery "
                        "(doc §4).")
    elif c2["attribution"].startswith("PLAIN-ADVANTAGED"):
        outcome = "TISSUE-COSTS-RECOVERY"
        verdict_text = ("TISSUE-COSTS-RECOVERY — the plain control recovers MORE than the "
                        "column: the ledger machinery's honesty price, recorded as a design "
                        "lead (doc §4).")
    else:
        outcome = "TISSUE-AMPLIFIES-RECOVERY"
        verdict_text = ("TISSUE-AMPLIFIES-RECOVERY — the column recovers MORE than the plain "
                        "control: the tissue amplifies re-encounter recovery (doc §4).")
    branches = []
    if outcome == "TISSUE-COSTS-RECOVERY":
        branches.append("§4: the ledger-vs-recovery trade-off becomes a design lead — a new "
                        "registration would test ledger machinery that does not tax recovery.")
    return {"outcome": outcome, "verdict": verdict_text, "branches": branches,
            "C1": c1, "C2": c2,
            "recovery_per_seed": {"COLUMN": col, "SCHED-TF": sch, "PLAIN-TF": pln},
            "post_D_absolute": {"COLUMN": [r["bpc_B_postD"] for r in col_rec],
                                "SCHED-TF": [r["bpc_B_postD"] for r in sched_rec],
                                "PLAIN-TF": [r["bpc_B_postD"] for r in plain_rec]},
            "alpha": alpha}


# ================================================================================= runner ========
def run(*, smoke: bool, results_path=None, force_smoke_path: bool = False,
        powered_cpu: bool = False):
    del powered_cpu  # pattern parity: the analysis is CPU-native
    path = resolve_results_path(results_path, smoke=smoke, force_smoke_path=force_smoke_path)

    from seq.gpu_harness import load_results, _save
    from seq.recall_gate import archive_run

    recs = {}
    for name, (subdir, prefix) in SOURCES.items():
        p = _ledger_path(subdir)
        if not os.path.isfile(p):
            raise SystemExit(
                f"refusing: the registered ledger is MISSING at {p} — PR-21 compares "
                "REGISTERED artifacts only (doc §2).")
        led = load_results(p)
        for s in CLAIM_SEEDS:
            r = led.get(f"{prefix}.s{s}")
            if not isinstance(r, dict):
                raise SystemExit(
                    f"refusing: {p} lacks {prefix}.s{s} — nothing is substituted.")
            recs.setdefault(name, []).append(r)

    col_rec = recs["COLUMN"]
    sched_rec = recs["SCHED-TF"]
    plain_rec = recs["PLAIN-TF"]

    res = load_results(path)
    res["meta"] = {
        "registry": REGISTRY_ID, "smoke": bool(smoke),
        "lane": "CLAIM (frozen pre-registration; confirmatory ANALYSIS)",
        "doc": "docs/preregistry/2026-09-11-recovery-attribution.md",
        "sources": {name: _ledger_path(sub) for name, (sub, _) in SOURCES.items()},
        "recovery_def": "bpc_B(post_C) - bpc_B(post_D) per seed (the revisit improvement)",
        "thresholds": {"gap_point": GAP_POINT, "ci_rule": "CI excludes 0 AND |point| >= 0.10",
                       "alpha": ALPHA},
        "note": ("all three arms share the stream and the eval slices; COLUMN/SCHED-TF/PLAIN"
                 "-TF differ only in the registered components (doc §2)"),
    }
    _save(res, path)
    print(f"[pr21] sources loaded: {list(SOURCES)}", flush=True)

    verdict = attribution(col_rec, sched_rec, plain_rec, alpha=ALPHA)

    raw_archive = archive_run(res, label=f"recovery-attr-{REGISTRY_ID}")
    res.setdefault("meta", {})["raw_archive"] = raw_archive

    report = {"registry": REGISTRY_ID, "smoke": bool(smoke),
              "raw_archive": raw_archive, "verdict": verdict,
              "powered_cpu_note": "analysis is CPU-native"}
    res["verdict"] = verdict
    res["report"] = report
    _save(res, path)

    print("\n" + "=" * 78, flush=True)
    print(f"  {REGISTRY_ID} — REVISIT-RECOVERY ATTRIBUTION (confirmatory analysis)",
          flush=True)
    print("=" * 78, flush=True)
    for name in ("COLUMN", "SCHED-TF", "PLAIN-TF"):
        r = verdict["recovery_per_seed"][name]
        print(f"    {name} recovery/seed: {[round(x, 3) for x in r]} "
              f"mean={sum(r)/len(r):.4f}", flush=True)
    print(f"    C1 (SCHED - COLUMN): delta={verdict['C1']['delta']:.4f} "
          f"CI=[{verdict['C1']['ci'][0]:.4f},{verdict['C1']['ci'][1]:.4f}] "
          f"-> {verdict['C1']['position']}", flush=True)
    print(f"    C2 (PLAIN - COLUMN): delta={verdict['C2']['delta']:.4f} "
          f"CI=[{verdict['C2']['ci'][0]:.4f},{verdict['C2']['ci'][1]:.4f}] "
          f"-> {verdict['C2']['position']}", flush=True)
    print(f"  OUTCOME: {verdict['outcome']}", flush=True)
    print(f"  {verdict['verdict']}", flush=True)
    for br in verdict["branches"]:
        print(f"  branch: {br}", flush=True)
    print(f"  ledger: {path}", flush=True)
    print(f"  raw archive: {raw_archive}", flush=True)
    print("=" * 78, flush=True)
    return report


def _build_parser():
    p = argparse.ArgumentParser(
        prog="recovery_attribution",
        description="PR-2026-09-03-21 claim runner (the revisit-recovery attribution; a "
                    "confirmatory analysis of registered artifacts — no training).")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true",
                      help="same analysis; separate smoke ledger (pattern parity)")
    mode.add_argument("--powered", action="store_true",
                      help="the registered analysis (CPU-native; no CUDA requirement)")
    mode.add_argument("--powered-cpu", action="store_true",
                      help="the registered analysis (identical; pattern parity)")
    p.add_argument("--out", default=None,
                   help="explicit results JSON path (overrides the default)")
    p.add_argument("--force-smoke-path", action="store_true",
                   help="let a --smoke run write the powered ledger it was pointed at "
                        "(default: REFUSED — separate ledgers)")
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(argv)
    run(smoke=args.smoke, results_path=args.out, force_smoke_path=args.force_smoke_path,
        powered_cpu=args.powered_cpu)


if __name__ == "__main__":
    main()
