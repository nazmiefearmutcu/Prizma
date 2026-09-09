"""
PR-2026-09-03-16 CLAIM RUNNER — the B-boundary drift-damage head-to-head (a CONFIRMATORY
ANALYSIS of two registered artifacts; NO training). Executes the FROZEN pre-registration
`docs/preregistry/2026-09-10-boundary-damage-headtohead.md` VERBATIM.

  python seq/boundary_damage_claim.py --powered-cpu   # the analysis (CPU-native, seconds)
  python seq/boundary_damage_claim.py --smoke         # identical here (analysis-only)

DAMAGES (both from registered ledgers, read-only):
  COLUMN: results/manyblock_PR-2026-09-03-14/powered.json claim.EX.s{s}
          bpc_B_postC - bpc_B_postB
  CONTROL: results/windowtf_manyblock_PR-2026-09-03-15/powered.json claim.WINDOW-TF.s{s}
          bpc_B_postC - bpc_B_postB
DELTA = mean(CONTROL damage) - mean(COLUMN damage); positive = the column damages less.
BAR (doc §3): PASS iff Welch CI lower >= 0.25; CI upper < 0.25 => NOT-ESTABLISHED
(n=10 seed extension PRE-AUTHORIZED); delta <= 0 => NEGATIVE; straddle =>
INCONCLUSIVE-NARROW (= NOT-ESTABLISHED per doc §4). Descriptive: per-seed damages both
sides + the revisit-recovery head-to-head.
"""
from __future__ import annotations

import argparse
import os
import sys

try:
    from .stats import t_sf
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from seq.stats import t_sf

import seq.prizma_lm_claim as plc
import seq.floorfreeze_claim as ffc  # noqa: F401  (pattern parity)


REGISTRY_ID = "PR-2026-09-03-16"
LEDDIR = "boundary_damage_PR-2026-09-03-16"
SMOKE_BASENAME = "smoke.json"
POWERED_BASENAME = "powered.json"

CLAIM_SEEDS = plc.CLAIM_SEEDS
P1_MARGIN = 0.25
ALPHA = plc.ALPHA

PR14_EX_LEDGER = ("manyblock_PR-2026-09-03-14", "claim.EX")
PR15_TF_LEDGER = ("windowtf_manyblock_PR-2026-09-03-15", "claim.WINDOW-TF")


def _results_root() -> str:
    root = os.environ.get("PRIZMA_RESULTS", os.path.join(os.path.dirname(__file__), "..", "results"))
    return os.path.abspath(root)


def _default_results_path(smoke: bool) -> str:
    # one ledger; the analysis is deterministic, so smoke and powered carry the same name
    # in their own subdirs per the pattern (the smoke/powered separation is kept for parity)
    return os.path.join(_results_root(), LEDDIR, SMOKE_BASENAME if smoke else POWERED_BASENAME)


def resolve_results_path(explicit=None, *, smoke: bool = False, force_smoke_path: bool = False) -> str:
    path = explicit if explicit else _default_results_path(smoke)
    if smoke and not force_smoke_path:
        powered = _default_results_path(smoke=False)
        if os.path.normcase(os.path.abspath(path)) == os.path.normcase(os.path.abspath(powered)):
            raise SystemExit(
                f"refusing: a --smoke run was pointed at the powered ledger ({powered}). Pass "
                f"--out <path> or --force-smoke-path (ledger separation, pattern parity).")
    return path


needs_cuda = plc.needs_cuda   # pattern parity: the analysis is CPU-native; modes accepted


# ==================================================================== PURE: verdict (§3) =========
def claim_verdict(col_damage, ctl_damage, *, alpha=ALPHA):
    """Frozen §3 verdict. col/ctl = per-seed damage lists (post_C - post_B)."""
    p1 = plc._welch_margin(col_damage, ctl_damage, P1_MARGIN)
    # delta = mean(ctl) - mean(col): positive = the column damages less.
    # PASS iff CI lower >= 0.25; the Welch p is one-sided for delta > 0.25 (the claim's
    # direction) — small p supports the claim; the CI position is the gate (doc §3).
    p1["damage_col_per_seed"] = list(col_damage)
    p1["damage_ctl_per_seed"] = list(ctl_damage)
    p1["col_damage_mean"] = sum(col_damage) / len(col_damage)
    p1["ctl_damage_mean"] = sum(ctl_damage) / len(ctl_damage)
    p1["note"] = ("delta = mean(CONTROL damage) - mean(COLUMN damage); positive = the column "
                  "damages less. PASS iff CI lower >= 0.25 (doc §3).")
    if p1["ci"][0] >= P1_MARGIN:
        p1["status"] = "PASS"
        outcome = "CLAIMED"
        verdict_text = ("CLAIMED — the tissue halves B-boundary drift damage: the control's "
                        "damage exceeds the column's by a CI-established margin above 0.25 "
                        "bpc (doc §4).")
    elif p1["delta"] <= 0:
        p1["status"] = "FAIL"
        outcome = "NEGATIVE"
        verdict_text = ("NEGATIVE — the control damages less or equal: the descriptives were "
                        "misleading (doc §4).")
    elif p1["ci"][1] < P1_MARGIN:
        p1["status"] = "NOT-ESTABLISHED"
        outcome = "NOT-ESTABLISHED"
        verdict_text = ("NOT-ESTABLISHED — the point gap stands but n=5 cannot establish half "
                        "of it (CI upper < 0.25); the pre-authorized n=10 seed extension is "
                        "the next iteration's first action (doc §4).")
    else:
        p1["status"] = "INCONCLUSIVE-NARROW"
        outcome = "NOT-ESTABLISHED"
        verdict_text = ("INCONCLUSIVE-NARROW — the CI straddles the 0.25 margin; same "
                        "pre-authorized n=10 path (doc §4).")
    branches = []
    if outcome == "NEGATIVE":
        branches.append("§4 (delta <= 0): the control damages less or equal — the "
                        "descriptives were misleading; no extension on this mechanism.")
    elif outcome != "CLAIMED":
        branches.append("§4: the n=10 seed extension (seeds 5-9, NEW training via the PR-14/"
                        "PR-15 runners with a registered seed extension) is the next "
                        "iteration's first action.")
    return {"outcome": outcome, "verdict": verdict_text, "branches": branches,
            "P1": p1, "alpha": alpha}


# ================================================================================= runner ========
def _damages(ledger_subdir, cell_prefix):
    """Per-seed B-boundary drift damage from a registered ledger (read-only)."""
    from seq.gpu_harness import load_results
    p = os.path.join(_results_root(), ledger_subdir, "powered.json")
    if not os.path.isfile(p):
        raise SystemExit(
            f"refusing: the registered ledger is MISSING at {p} — PR-16 compares "
            "REGISTERED artifacts only (doc §2).")
    led = load_results(p)
    out = []
    for s in CLAIM_SEEDS:
        rec = led.get(f"{cell_prefix}.s{s}")
        if not isinstance(rec, dict):
            raise SystemExit(
                f"refusing: {p} lacks {cell_prefix}.s{s} — PR-16 compares the registered "
                "n=5 pairs; nothing is substituted.")
        out.append(rec["bpc_B_postC"] - rec["bpc_B_postB"])
    return out, p


def run(*, smoke: bool, results_path=None, force_smoke_path: bool = False,
        powered_cpu: bool = False):
    del powered_cpu  # pattern parity: the analysis is CPU-native
    path = resolve_results_path(results_path, smoke=smoke, force_smoke_path=force_smoke_path)

    from seq.gpu_harness import load_results, _save
    from seq.recall_gate import archive_run

    col_damage, col_path = _damages(*PR14_EX_LEDGER)
    ctl_damage, ctl_path = _damages(*PR15_TF_LEDGER)

    res = load_results(path)
    res["meta"] = {
        "registry": REGISTRY_ID, "smoke": bool(smoke),
        "lane": "CLAIM (frozen pre-registration; confirmatory ANALYSIS)",
        "doc": "docs/preregistry/2026-09-10-boundary-damage-headtohead.md",
        "sources": {"column": col_path, "control": ctl_path,
                    "cells": "claim.EX.s0..s4 / claim.WINDOW-TF.s0..s4 (reused, disclosed)"},
        "damage_def": "bpc_B_postC - bpc_B_postB per seed (the degradation from block C)",
        "bars": {"P1_margin": P1_MARGIN, "alpha": ALPHA,
                 "t_isf_convention": "UPPER-TAIL p in (0, 0.5] (the PR-03 lesson)"},
        "post_hoc_disclosure": ("the ~0.55 point gap was descriptively visible in PR-15's "
                                "console/INDEX before this freeze; the bar is HALF that point "
                                "estimate (doc §2)"),
    }
    _save(res, path)
    print(f"[pr16] sources: column={col_path}; control={ctl_path}", flush=True)

    verdict = claim_verdict(col_damage, ctl_damage, alpha=ALPHA)

    # ---------------- RETENTION: archive BEFORE the ledgered verdict --------------------------
    raw_archive = archive_run(res, label=f"boundary-damage-{REGISTRY_ID}")
    res.setdefault("meta", {})["raw_archive"] = raw_archive

    report = {"registry": REGISTRY_ID, "smoke": bool(smoke),
              "raw_archive": raw_archive, "verdict": verdict,
              "powered_cpu_note": "analysis is CPU-native"}
    res["verdict"] = verdict
    res["report"] = report
    _save(res, path)

    print("\n" + "=" * 78, flush=True)
    print(f"  {REGISTRY_ID} — B-BOUNDARY DRIFT-DAMAGE HEAD-TO-HEAD (confirmatory analysis)",
          flush=True)
    print("=" * 78, flush=True)
    p1 = verdict["P1"]
    print(f"    column damage/seed: {p1['damage_col_per_seed']}", flush=True)
    print(f"    control damage/seed: {p1['damage_ctl_per_seed']}", flush=True)
    print(f"    P1: delta mean={p1['delta']:.4f} CI=[{p1['ci'][0]:.4f},{p1['ci'][1]:.4f}] "
          f"(bar: CI lower >= {p1['margin']}) p_raw={p1['p_raw']:.4f} -> {p1['status']}",
          flush=True)
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
        prog="boundary_damage_claim",
        description="PR-2026-09-03-16 claim runner (the B-boundary drift-damage head-to-head; "
                    "a confirmatory analysis of registered artifacts — no training).")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true",
                      help="same analysis; separate smoke ledger (pattern parity)")
    mode.add_argument("--powered", action="store_true",
                      help="the registered analysis (CPU-native; no CUDA requirement)")
    mode.add_argument("--powered-cpu", action="store_true",
                      help="the registered analysis (identical; pattern parity with the "
                           "training runners)")
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
