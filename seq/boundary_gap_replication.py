"""
PR-2026-09-03-20 CLAIM RUNNER — the boundary-damage gap REPLICATED on fresh seeds (a
CONFIRMATORY ANALYSIS of the PR-19 ledger; NO training). Executes the FROZEN
pre-registration `docs/preregistry/2026-09-10-damage-gap-replication.md` VERBATIM.

  python seq/boundary_gap_replication.py --powered-cpu   # the analysis (CPU-native, seconds)
  python seq/boundary_gap_replication.py --smoke         # identical; separate ledger (parity)

DAMAGES (both from the PR-19 registered ledger, read-only — fresh seeds 5-9):
  EX damage:    claim.EX.s{s}.bpc_B_postC - claim.EX.s{s}.bpc_B_postB
  PLAIN damage: claim.PLAIN.s{s}.bpc_B_postC - claim.PLAIN.s{s}.bpc_B_postB
DELTA = mean(PLAIN damage) - mean(EX damage); positive = the exclusion damages less.
BAR (doc §3): PASS iff Welch CI lower >= 0.25 (PR-16's own bar); delta <= 0 => NEGATIVE;
otherwise => NOT-ESTABLISHED (the seeds 10-14 extension is PRE-AUTHORIZED per doc §4).
"""
from __future__ import annotations

import argparse
import os
import sys

try:
    from .stats import t_sf  # noqa: F401  (parity with the analysis runners)
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from seq.stats import t_sf

import seq.prizma_lm_claim as plc
import seq.boundary_damage_claim as bdc   # noqa: F401  (pattern parity)


REGISTRY_ID = "PR-2026-09-03-20"
LEDDIR = "damage_gap_repl_PR-2026-09-03-20"
SMOKE_BASENAME = "smoke.json"
POWERED_BASENAME = "powered.json"

CLAIM_SEEDS = plc.CLAIM_SEEDS
PR20_SEEDS = (5, 6, 7, 8, 9)              # the PR-19 replication's FRESH seeds (its ledger
                                          # carries claim.*.s5..s9 — not s0..s4)
P1_MARGIN = 0.25
ALPHA = plc.ALPHA

PR19_LEDGER = "manyblock_PR-2026-09-03-19"


def _results_root() -> str:
    root = os.environ.get("PRIZMA_RESULTS", os.path.join(os.path.dirname(__file__), "..", "results"))
    return os.path.abspath(root)


def _default_results_path(smoke: bool) -> str:
    return os.path.join(_results_root(), LEDDIR, SMOKE_BASENAME if smoke else POWERED_BASENAME)


def _pr19_powered_path() -> str:
    return os.path.join(_results_root(), PR19_LEDGER, POWERED_BASENAME)


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


# ==================================================================== PURE: damages + verdict ====
def damages(plain_rec, ex_rec):
    """Per-seed B-boundary drift damage (bpc_B postC - postB) for one PR-19 arm pair."""
    return {"plain": [r["bpc_B_postC"] - r["bpc_B_postB"] for r in plain_rec],
            "ex": [r["bpc_B_postC"] - r["bpc_B_postB"] for r in ex_rec]}


def claim_verdict(plain_damage, ex_damage, *, alpha=ALPHA):
    """Frozen §3 verdict: delta = mean(PLAIN damage) - mean(EX damage); PASS iff Welch
    CI lower >= 0.25; delta <= 0 => NEGATIVE; otherwise NOT-ESTABLISHED (n=10 extension
    pre-authorized)."""
    p1 = plc._welch_margin(ex_damage, plain_damage, P1_MARGIN)
    # delta = mean(b=PLAIN) - mean(a=EX): positive = the exclusion damages less.
    p1["damage_plain_per_seed"] = list(plain_damage)
    p1["damage_ex_per_seed"] = list(ex_damage)
    p1["plain_damage_mean"] = sum(plain_damage) / len(plain_damage)
    p1["ex_damage_mean"] = sum(ex_damage) / len(ex_damage)
    p1["note"] = ("delta = mean(PLAIN damage) - mean(EX damage); positive = the exclusion "
                  "damages less. PASS iff CI lower >= 0.25 (PR-16's own bar, doc §3).")
    if p1["ci"][0] >= P1_MARGIN:
        p1["status"] = "PASS"
        outcome = "CLAIMED"
        verdict_text = ("CLAIMED — the boundary-damage gap REPLICATES on fresh seeds: the "
                        "exclusion's protection is CI-established at the 0.25 margin on "
                        "seeds 5-9 (doc §4). The replication set is COMPLETE: flagship "
                        "(PR-18), accumulation (PR-19), damage gap (this registration).")
    elif p1["delta"] <= 0:
        p1["status"] = "FAIL"
        outcome = "NEGATIVE"
        verdict_text = ("NEGATIVE — the control-arm damages less or equal: the PR-16 gap "
                        "was a small-sample artifact (doc §4); PR-16's interpretation is "
                        "downgraded.")
    else:
        p1["status"] = "NOT-ESTABLISHED"
        outcome = "NOT-ESTABLISHED"
        verdict_text = ("NOT-ESTABLISHED — the point gap is positive but n=5 cannot "
                        "establish the 0.25 margin (doc §4); the seeds 10-14 extension is "
                        "the next iteration's first action (pre-authorized).")
    branches = []
    if outcome != "CLAIMED":
        branches.append("§4: the seeds 10-14 extension (PLAIN + EX at offset 10, n=10 "
                        "fresh cumulative) is PRE-AUTHORIZED — the next iteration's first "
                        "action.")
    return {"outcome": outcome, "verdict": verdict_text, "branches": branches,
            "P1": p1, "alpha": alpha}


# ================================================================================= runner ========
def run(*, smoke: bool, results_path=None, force_smoke_path: bool = False,
        powered_cpu: bool = False):
    del powered_cpu  # pattern parity: the analysis is CPU-native
    path = resolve_results_path(results_path, smoke=smoke, force_smoke_path=force_smoke_path)

    from seq.gpu_harness import load_results, _save
    from seq.recall_gate import archive_run

    pr19_path = _pr19_powered_path()
    if not os.path.isfile(pr19_path):
        raise SystemExit(
            f"refusing: the PR-19 powered ledger is MISSING at {pr19_path} — PR-20 reads "
            "the fresh-seed damage pairs from it (doc §2). Run this where the PR-19 ledger "
            "exists.")
    pr19 = load_results(pr19_path)
    plain_rec = []
    ex_rec = []
    for s in PR20_SEEDS:
        for cellkey, bucket in ((f"claim.PLAIN.s{s}", plain_rec),
                                (f"claim.EX.s{s}", ex_rec)):
            r = pr19.get(cellkey)
            if not isinstance(r, dict):
                raise SystemExit(
                    f"refusing: {pr19_path} lacks {cellkey} — PR-20 compares the registered "
                    "n=5 fresh pairs; nothing is substituted.")
            bucket.append(r)
    dmg = damages(plain_rec, ex_rec)

    res = load_results(path)
    res["meta"] = {
        "registry": REGISTRY_ID, "smoke": bool(smoke),
        "lane": "CLAIM (frozen pre-registration; confirmatory ANALYSIS)",
        "doc": "docs/preregistry/2026-09-10-damage-gap-replication.md",
        "source": {"ledger": pr19_path,
                   "cells": "claim.PLAIN.s5..s9 / claim.EX.s5..s9 (reused, disclosed)",
                   "note": ("both arms share A/B training per seed — the PR-19 cross-arm "
                            "canary proved it; the postB baselines are therefore "
                            "bit-identical across arms")},
        "damage_def": "bpc_B_postC - bpc_B_postB per seed (the degradation from block C)",
        "bars": {"P1_margin": P1_MARGIN, "alpha": ALPHA,
                 "t_isf_convention": "UPPER-TAIL p in (0, 0.5] (the PR-03 lesson)"},
        "post_hoc_disclosure": ("the gap was descriptively visible in the PR-19 console "
                                "(~+0.31) before this freeze; the bar is PR-16's own 0.25"),
    }
    _save(res, path)
    print(f"[pr20] source: {pr19_path}", flush=True)

    verdict = claim_verdict(dmg["plain"], dmg["ex"], alpha=ALPHA)

    raw_archive = archive_run(res, label=f"damage-gap-repl-{REGISTRY_ID}")
    res.setdefault("meta", {})["raw_archive"] = raw_archive

    report = {"registry": REGISTRY_ID, "smoke": bool(smoke),
              "raw_archive": raw_archive, "verdict": verdict,
              "powered_cpu_note": "analysis is CPU-native"}
    res["verdict"] = verdict
    res["report"] = report
    _save(res, path)

    print("\n" + "=" * 78, flush=True)
    print(f"  {REGISTRY_ID} — DAMAGE-GAP REPLICATION (confirmatory analysis)", flush=True)
    print("=" * 78, flush=True)
    p1 = verdict["P1"]
    print(f"    PLAIN damage/seed: {p1['damage_plain_per_seed']}", flush=True)
    print(f"    EX damage/seed:    {p1['damage_ex_per_seed']}", flush=True)
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
        prog="boundary_gap_replication",
        description="PR-2026-09-03-20 claim runner (the boundary-damage gap replicated on "
                    "fresh seeds; a confirmatory analysis of the PR-19 ledger — no training).")
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
