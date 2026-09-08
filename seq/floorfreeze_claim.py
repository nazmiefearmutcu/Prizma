"""
PR-2026-09-03-09 CLAIM RUNNER — the floor-freeze routing repair (the registered successor of
PR-08's B3a negative). Executes the FROZEN pre-registration
`docs/preregistry/2026-09-09-floorfreeze-routing-repair.md` VERBATIM. One script, three modes
(the prizma_lm_claim pattern, mirrored):

  python seq/floorfreeze_claim.py --smoke        # tiny CPU plumbing run -> the SMOKE ledger
  python seq/floorfreeze_claim.py --powered      # the claim campaign (refuses without CUDA)
  python seq/floorfreeze_claim.py --powered-cpu  # the doc section-6 CPU-feasible fallback

WHAT IS FROZEN (pre-reg §2-§5 — inherited from PR-08's registered text, reused BY IMPORT from
seq/prizma_lm_claim.py, NOT re-declared): stream, slices, corpora, tissue constants (E=4,
M_max=4, z=5, freeze_min_seen=300, H=64, FLOOR_EMA=0.05), SEG/BATCH_SEGS, seeds 0-4, PR-08's
routing semantics (floor-maturity veto + Policy A eviction). LR is FROZEN at 3e-3 for both
arms (inherited from PR-08's registered lr_selection.fusion — doc §2: NO re-selection, NO
lr-selection leg in this runner; comparability with PR-08 is the point).

Arms (2, exclusive):
  OFF            — PR-08 PRIM-LM verbatim (plc.run cell body reused structurally; never sets
                   the freeze attribute).
  FROZEN-FLOORS  — identical cell; after block-A training completes the model gains
                   `model.floor_freeze = {slots committed at A end}` (a plain Python set of
                   int slot indices). From B onward the guarded branch in
                   seq/fusion_probe.py `_expert_train` skips those slots' precision-floor
                   updates (mu/var pinned at their A-end values); a Policy-A re-initialized
                   slot LEAVES the set via the guarded discard in seq/prizma_lm_claim.py
                   `route_pr08`. Expert predictors/optimizers are NOT frozen — only floors.
The freeze is a guarded default-off lever: with the attribute absent both touched files are
semantically byte-identical to PR-08 (off-identity pinned behaviorally by tests).

BARS (doc §3, exact; n=5 seeds 0-4):
  P1 (primary, repair): mean over FROZEN-FLOORS seeds of ledger.boundary_window.
      frac_to_A_expert >= 0.5 — EXACT MEANS, no CI (the PR-08 B3 form). The OFF arm's frac
      mean is the recorded control (PR-08 measured 0.428).
  G1 (retention guard): BPC(A-eval, post-C) - BPC(A-eval, pre-B) <= 0.05 for FROZEN-FLOORS
      (one-sample, CI upper <= 0.05 AND Holm p < 0.05; CI lower > 0.05 -> FAIL; straddle ->
      INCONCLUSIVE).
  G2 (adaptation guard): BPC(C-ret, post-C) advantage of FROZEN-FLOORS vs PR-08's
      FROZEN-CHECKPOINT >= 0.10 (Welch, CI lower >= 0.10; CI upper < 0.10 -> FAIL;
      straddle -> INCONCLUSIVE). The FROZEN-CHECKPOINT baseline cells are REUSED from the
      PR-08 powered ledger (disclosed: that arm has no tissue, is lever-independent, same
      seeds and frozen protocol).
  Holm family = [G1, G2] (P1 is means-based, untested — the PR-08 B3 convention).
  Reported, never gated: B4 trunk drift per arm; routing-ledger churn (recruits/evictions/
  vetoed) per arm; per-seed fracs for both arms.

Overall (doc §5, pre-committed): CLAIMED iff P1 PASS and G1 PASS and G2 PASS; NEGATIVE-GUARD
if P1 PASS and any guard FAIL; NEGATIVE if P1 FAIL; any guard INCONCLUSIVE -> INCONCLUSIVE
(seed extension to n=10, seeds 5-9, PRE-AUTHORIZED by the registration).

CANARY (doc §4, fail-loud, POWERED MODE ONLY): after the OFF arm's 5 cells, the PR-08
powered ledger (results/prizma_lm_PR-2026-09-03-08/powered.json, read-only) is loaded and
every science field of OFF.s{seed} must equal PR-08's claim.PRIM-LM.s{seed} EXACTLY
(bpc_* floats, the ledger dict, a_expert, fgt_A_full, b_degradation_B_postC; wall_s and the
cell-loop wrapper keys arm/cellkey/cfgsig/complete/forced_placement are ignored). Any
mismatch — or a missing PR-08 ledger — is a SystemExit ABORT before the treatment arm runs.
The FROZEN-FLOORS A-phase must match the OFF arm's A-phase bit-for-bit (the freeze activates
only after A) — asserted in smoke via equal bpc_A_preB.

LEDGER SEPARATION + RETENTION (prizma_lm_claim pattern, mirrored):
  smoke   -> results/floorfreeze_PR-2026-09-03-09/smoke.json
  powered -> results/floorfreeze_PR-2026-09-03-09/powered.json
  A --smoke run pointed at the powered ledger is REFUSED unless --force-smoke-path. Raw
  records stream crash-safe after every cell AND are archived verbatim
  (seq.recall_gate.archive_run) BEFORE any verdict; the verdict references the archive path.
  Resume is keyed on (cellkey, config-fingerprint) with the arm's floor_freeze state IN the
  fingerprint; a cell present at a FOREIGN fingerprint is a hard refusal.
"""
from __future__ import annotations

import argparse
import os
import sys

# Light imports only at module top: the pure layer (paths/refusal, bar math, verdict, canary
# comparator) must be importable + unit-testable without torch or training (surprise_claim
# discipline). PR-08's plumbing is reused BY IMPORT, not copy-pasted.
try:
    from .stats import holm_correction
except ImportError:                                   # run as a bare script: bootstrap sys.path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from seq.stats import holm_correction

import seq.prizma_lm_claim as plc                     # PR-08 protocol pieces, reused VERBATIM


# ================================================================== frozen protocol constants ====
REGISTRY_ID = "PR-2026-09-03-09"
LEDDIR = "floorfreeze_PR-2026-09-03-09"         # under $PRIZMA_RESULTS (default ./results)
SMOKE_BASENAME = "smoke.json"
POWERED_BASENAME = "powered.json"

ARMS = ("OFF", "FROZEN-FLOORS")                 # control first — the canary runs between arms
CLAIM_SEEDS = plc.CLAIM_SEEDS                   # (0, 1, 2, 3, 4) — inherited, never substituted

# ---- protocol constants inherited from PR-08 BY IMPORT (doc §2: inherits PR-08 verbatim) ----
LR_FROZEN = 3e-3                        # doc §2: inherited from PR-08's lr_selection.fusion —
                                        # NO lr-selection leg exists in this runner
SEG = plc.SEG                           # 256 chars per segment
BATCH_SEGS = plc.BATCH_SEGS             # 32 segments per training step
E_POOL = plc.E_POOL                     # 4
M_MAX = plc.M_MAX                       # 4
FREEZE_MIN_SEEN = plc.FREEZE_MIN_SEEN   # 300 (floor-maturity veto, unchanged)
Z_NOVEL = plc.Z_NOVEL                   # 5.0
H_SMALL = plc.H_SMALL                   # 64
FLOOR_EMA = plc.FLOOR_EMA               # 0.05
FRESH_HEAD_SEED_BASE = plc.FRESH_HEAD_SEED_BASE   # 20260908 pinned eviction re-init formula
B3_WINDOW_BATCHES = plc.B3_WINDOW_BATCHES         # 20 — the boundary window (P1's quantity)

# ---- bars (doc §3, exact) ----
P1_FRAC_BAR = 0.5                       # mean frac_to_A_expert >= 0.5 (exact means, no CI)
G1_MARGIN = 0.05                        # BPC(A, post-C) - BPC(A, pre-B) <= 0.05 (CI upper)
G2_MARGIN = 0.10                        # C-ret advantage vs FROZEN-CHECKPOINT >= 0.10 (Welch)
ALPHA = plc.ALPHA                       # 0.05
HOLM_FAMILY = ("G1", "G2")              # P1 is means-based, untested (PR-08 B3 convention)

PR08_LEDDIR = plc.LEDDIR                # "prizma_lm_PR-2026-09-03-08" — the canary/G2 source
POWERED_CPU_FALLBACK_NOTE = (
    "POWERED VIA THE DOC SECTION-6 CPU-FEASIBLE FALLBACK (PR-09 doc §6; precedent = PR-08 "
    "doc addendum 2026-09-08 #2, point 1): protocol-identical to the A100 tier; --powered "
    "(CUDA-refusing) remains available for the GPU session.")


# ==================================================================== paths + BAR-0 refusal ======
def _results_root() -> str:
    root = os.environ.get("PRIZMA_RESULTS", os.path.join(os.path.dirname(__file__), "..", "results"))
    return os.path.abspath(root)


def _default_results_path(smoke: bool) -> str:
    return os.path.join(_results_root(), LEDDIR, SMOKE_BASENAME if smoke else POWERED_BASENAME)


def resolve_results_path(explicit=None, *, smoke: bool = False, force_smoke_path: bool = False) -> str:
    """Resolve the results path and enforce the smoke/powered file separation (the plc
    refusal pattern, pointed at the PR-09 led dir): a --smoke run pointed at the POWERED
    ledger is refused (SystemExit) unless force_smoke_path."""
    path = explicit if explicit else _default_results_path(smoke)
    if smoke and not force_smoke_path:
        powered = _default_results_path(smoke=False)
        if os.path.normcase(os.path.abspath(path)) == os.path.normcase(os.path.abspath(powered)):
            raise SystemExit(
                f"refusing: a --smoke run was pointed at the powered ledger ({powered}). Smoke and "
                f"campaign results use separate files by default (PR-2026-09-03-09 §6 ledger "
                f"separation; see results/campaign_2026-06-08/CONTAMINATION.md). Pass --out <path> "
                f"to write the smoke elsewhere, or --force-smoke-path to override deliberately.")
    return path


def require_cuda(has_cuda: bool) -> None:
    """Pure guard for the --powered mode (unit-testable without a CUDA box): the PR-09 claim
    campaign is the GPU session's job; a CPU 'powered' run would be neither powered nor the
    pre-registered environment. --powered-cpu (doc §6 fallback) skips this guard by design."""
    if not has_cuda:
        raise SystemExit(
            "refusing: --powered executes the PR-2026-09-03-09 claim campaign and requires a CUDA "
            "device. No CUDA device is visible here. Use --smoke for the CPU plumbing run, or "
            "--powered-cpu for the doc section-6 CPU-feasible fallback.")


# REUSED from plc by import (doc §6 contract): only the --powered mode demands CUDA;
# --powered-cpu (the doc section-6 fallback) and --smoke run CPU-side by design. TRUE alias,
# not a fork (pinned by tests).
needs_cuda = plc.needs_cuda


# ==================================================================== PURE: slices (inherited) ===
# PR-09 inherits PR-08's §2 stream verbatim, so the pinned-slice table IS plc.pin_slices
# (reused BY IMPORT — including its disjointness assertions, which pin the PR-08 C-range
# repair against regressions here too).
pin_slices = plc.pin_slices


# ==================================================================== PURE: canary comparator ====
# The doc §4 canary: OFF.s{seed} must equal PR-08's claim.PRIM-LM.s{seed} EXACTLY on every
# science field. Ignored: wall_s (timing) + the cell-loop wrapper keys the PR-08 runner adds
# AFTER run_routed returns (arm/cellkey/cfgsig/complete) or the FORCED-RECRUIT-only placement
# record (forced_placement — never present on PRIM-LM cells, listed for completeness).
CANARY_IGNORE = frozenset({"wall_s", "arm", "cellkey", "cfgsig", "complete", "forced_placement"})


def canary_mismatches(off_rec, pr08_rec):
    """Pure §4 comparator: the list of science fields whose values differ between the fresh
    OFF cell and the PR-08 PRIM-LM record (exact == equality on floats and the nested ledger
    dict). Empty list = the canary passes for this cell."""
    science = sorted(k for k in set(off_rec) | set(pr08_rec) if k not in CANARY_IGNORE)
    return [k for k in science if off_rec.get(k) != pr08_rec.get(k)]


# ==================================================================== PURE: verdict (§3+§5) ======
def claim_verdict(off, ff, frozen_ck, *, alpha=ALPHA):
    """The frozen §3/§5 verdict as a PURE function over per-seed record dicts (unit-tested
    without torch). Args: lists of stored cell records — off = the OFF control seeds, ff =
    the FROZEN-FLOORS seeds (both from THIS ledger), frozen_ck = PR-08's FROZEN-CHECKPOINT
    cells (REUSED from the PR-08 powered ledger; only bpc_Cret_postC is read from them).

    P1: exact means, no CI (the PR-08 B3 form): mean(ff boundary_window.frac_to_A_expert)
        >= 0.5 -> PASS else FAIL; the OFF frac mean is recorded as the control.
    G1: one-sample, diffs = bpc_A_postC - bpc_A_preB per ff seed; PASS iff CI upper <= 0.05
        AND Holm p < alpha; CI lower > 0.05 -> FAIL; straddle -> INCONCLUSIVE.
    G2: Welch advantage mean(frozen_ck bpc_Cret_postC) - mean(ff bpc_Cret_postC) >= 0.10;
        PASS iff CI lower >= 0.10 AND Holm p < alpha; CI upper < 0.10 -> FAIL; straddle ->
        INCONCLUSIVE.
    Holm family = [G1, G2] (P1 is means-based, untested).
    Overall: any guard INCONCLUSIVE -> INCONCLUSIVE; P1 PASS + G1 + G2 PASS -> CLAIMED;
    P1 PASS + any guard FAIL -> NEGATIVE-GUARD; P1 FAIL -> NEGATIVE.
    """
    # ---- P1 (primary, repair): exact means over the boundary-window fracs ----
    ff_fracs = [float(r["ledger"]["boundary_window"]["frac_to_A_expert"]) for r in ff]
    off_fracs = [float(r["ledger"]["boundary_window"]["frac_to_A_expert"]) for r in off]
    ff_frac_mean = sum(ff_fracs) / len(ff_fracs)
    off_frac_mean = sum(off_fracs) / len(off_fracs)
    p1_ok = ff_frac_mean >= P1_FRAC_BAR
    p1 = {"frac_per_seed": ff_fracs, "frac_mean": ff_frac_mean, "frac_bar": P1_FRAC_BAR,
          "status": "PASS" if p1_ok else "FAIL",
          "off_frac_per_seed": off_fracs, "off_frac_mean": off_frac_mean,
          "off_note": "OFF is the recorded control (PR-08 PRIM-LM measured 0.428 at n=5)"}

    # ---- G1 (retention guard): one-sample, direction 'below' ----
    g1 = plc._onesample_margin([r["bpc_A_postC"] - r["bpc_A_preB"] for r in ff], G1_MARGIN)

    # ---- G2 (adaptation guard): Welch advantage, direction 'above'; the candidate is
    # FROZEN-FLOORS and the baseline PR-08's FROZEN-CHECKPOINT (lower bpc is better, so the
    # advantage = mean(baseline) - mean(candidate), exactly plc._welch_margin's delta). ----
    g2 = plc._welch_margin([r["bpc_Cret_postC"] for r in ff],
                           [r["bpc_Cret_postC"] for r in frozen_ck], G2_MARGIN)
    g2["baseline_note"] = ("baseline = PR-08 FROZEN-CHECKPOINT cells REUSED from "
                           f"results/{PR08_LEDDIR}/powered.json (no tissue, lever-independent, "
                           "same seeds + frozen protocol — disclosed, doc §3)")

    holm = holm_correction([g1["p_raw"], g2["p_raw"]], alpha=alpha)
    g1["p_holm"] = holm[0]["p_adj"]
    g2["p_holm"] = holm[1]["p_adj"]
    g1["status"] = plc._bar_status(g1["ci"], G1_MARGIN, g1["p_holm"], "below", alpha)
    g2["status"] = plc._bar_status(g2["ci"], G2_MARGIN, g2["p_holm"], "above", alpha)

    # ---- reported, never gated: B4 trunk drift + routing-ledger churn per arm ----
    def _b4(recs):
        d = [r["bpc_B_postC"] - r["bpc_B_postB"] for r in recs]
        return {"per_seed": d, "mean": sum(d) / len(d)}

    def _churn(recs):
        out = {}
        for r in recs:
            led = r["ledger"]
            key = f"s{r['seed']}"
            out[key] = {"recruits": len(led["recruits"]),
                        "evictions": len(led["evictions"]),
                        "vetoed_novel_segments": led["vetoed_novel_segments"]}
        return out

    b4 = {"OFF": _b4(off), "FROZEN-FLOORS": _b4(ff)}
    churn = {"OFF": _churn(off), "FROZEN-FLOORS": _churn(ff)}

    # ---- overall verdict + pre-committed §5 branches, echoed verbatim ----
    branches = []
    guard_statuses = [g1["status"], g2["status"]]
    if "INCONCLUSIVE" in guard_statuses:
        branches.append("§5 (any guard CI straddles its margin): INCONCLUSIVE for that guard; "
                        "seed extension to n=10 (seeds 5-9) is PRE-AUTHORIZED by this "
                        "registration, executed time permitting (the 03:55 wind-down rule "
                        "always wins). INCONCLUSIVE is never PASS (PR-03 mirror).")
    if "INCONCLUSIVE" in guard_statuses:
        outcome = "INCONCLUSIVE"
    elif p1_ok and all(s == "PASS" for s in guard_statuses):
        outcome = "CLAIMED"
    elif p1_ok:
        outcome = "NEGATIVE-GUARD"
    else:
        outcome = "NEGATIVE"

    if outcome == "CLAIMED":
        verdict_text = ("CLAIMED — P1 PASS with G1 (retention-A) and G2 (adaptation-C) intact: "
                        "trunk-floor decoupling confirmed as the B3a mechanism; PR-LM-1 "
                        "designs inherit floor freezing on returning blocks (doc §5).")
    elif outcome == "NEGATIVE-GUARD":
        broke = [name for name, st in zip(("G1", "G2"), guard_statuses) if st == "FAIL"]
        verdict_text = ("NEGATIVE (with lead) — the lever buys routing at the cost of the "
                        "column: P1 PASS but " + " and ".join(broke) + " FAIL. Floor "
                        "scheduling (e.g. freeze only on C) becomes the next candidate (doc §5).")
        branches.append("§5 (P1 PASS + guard FAIL): record which guard broke — "
                        f"{', '.join(broke)}; floor scheduling (e.g. freeze only on C) becomes "
                        "the next candidate.")
    elif outcome == "NEGATIVE":
        verdict_text = ("NEGATIVE — P1 FAIL: floor freeze insufficient to repair the "
                        "returning-domain boundary routing. The next ladder candidate "
                        "(trunk-lr scheduling on returning blocks) gets its own prereg; no "
                        "re-tuning of this lever without a new registration (doc §5).")
        branches.append("§5 (P1 FAIL): floor freeze insufficient; the next ladder candidate "
                        "(trunk-lr scheduling on returning blocks) gets its own prereg. No "
                        "re-tuning of this lever without a new registration.")
    else:
        verdict_text = ("INCONCLUSIVE — a guard CI straddles its margin (PR-03 mirror: never "
                        "PASS). Pre-committed §5 branch applies (see 'branches').")

    return {"outcome": outcome, "verdict": verdict_text, "branches": branches,
            "P1": p1, "G1": g1, "G2": g2, "B4": b4, "churn": churn,
            "holm_family": [{"bar": "G1", "p_raw": g1["p_raw"], "p_holm": g1["p_holm"],
                             "status": g1["status"]},
                            {"bar": "G2", "p_raw": g2["p_raw"], "p_holm": g2["p_holm"],
                             "status": g2["status"]}],
            "alpha": alpha}


# ================================================================================= runner ========
def run_cell(vocab_size, seed, data, lr, *, freeze_floors):
    """PR-09 cell — plc.run_routed's structure VERBATIM (same phase order, same eval order,
    same bpc keys, same ledger snapshot, same boundary window, same B4 quantity) plus the
    floor-freeze wiring:
      OFF           (freeze_floors=False): identical to plc.run_routed's PRIM-LM path — the
                    freeze attribute is NEVER set (off-identity; the powered canary proves it
                    against PR-08's stored records).
      FROZEN-FLOORS (freeze_floors=True): AFTER block-A training + the bpc_A_preB eval (the
                    A-phase is bit-identical to OFF — the freeze activates only after A), the
                    committed slots' set is pinned onto the model and their (mu, var) values
                    are recorded for audit; during B/C the guarded branches do the rest (no
                    per-batch runner code)."""
    import time

    import torch
    from seq import fusion_probe as fp

    t0 = time.time()
    Ax, Ay, Bx, By, Cx, Cy, Aex, Aey, Bex, Bey, Crx, Cry = data
    E = E_POOL
    model = fp.build_model(vocab_size, seed, E)
    ledger = plc._fresh_ledger()
    tr = {"A": [0] * E, "B": [0] * E, "C": [0] * E}
    rec = {"config": ("PRIM-LM+floor-freeze" if freeze_floors else "PRIM-LM"),
           "E": E, "seed": seed, "lr": lr}

    # ---- block A (backbone trains; the first slot recruits into the empty pool) ----
    plc.train_pr08(model, Ax, Ay, lr, "A", ledger, seed=seed)
    tr["A"] = model.n_segments[:]
    rec["bpc_A_preB"] = fp.eval_bpc_fusion(model, Aex, Aey)

    # ---- PR-09 treatment wiring (doc §2): pin the floors AFTER block A ----
    if freeze_floors:
        frozen_slots = {s for s in range(E) if model.committed[s]}
        model.floor_freeze = frozen_slots
        rec["floor_freeze"] = {
            "slots": sorted(frozen_slots),
            "pinned_mu": {str(s): float(model.mu[s]) for s in frozen_slots},
            "pinned_var": {str(s): float(model.var[s]) for s in frozen_slots},
            "note": ("mu/var PINNED at block-A end; _expert_train (seq/fusion_probe.py) skips "
                     "floor updates for these slots from B onward; a Policy-A re-initialized "
                     "slot leaves the set (guarded discard in seq/prizma_lm_claim.py "
                     "route_pr08). Expert predictors/optimizers are NOT frozen."),
        }

    # ---- block B (drift, strong) ----
    before = model.n_segments[:]
    train_frozen = False                      # PR-09 arms are fusion-family PRIM cells only
    plc.train_pr08(model, Bx, By, lr, "B", ledger, seed=seed, frozen=train_frozen)
    tr["B"] = [model.n_segments[s] - before[s] for s in range(E)]
    rec["bpc_A_postB"] = fp.eval_bpc_fusion(model, Aex, Aey)      # descriptive
    rec["bpc_B_postB"] = fp.eval_bpc_fusion(model, Bex, Bey)
    rec["bpc_Cret_postB"] = fp.eval_bpc_fusion(model, Crx, Cry)   # descriptive

    # ---- block C (drift, mild + RETURNING domain) ----
    a_expert = max((s for s in range(E) if model.committed[s]), key=lambda s: tr["A"][s])
    rec["a_expert"] = int(a_expert)
    boundary = {"segments_total": 0, "to_A_expert": 0, "a_expert": int(a_expert)}
    before = model.n_segments[:]
    plc.train_pr08(model, Cx, Cy, lr, "C", ledger, seed=seed, boundary=boundary)
    tr["C"] = [model.n_segments[s] - before[s] for s in range(E)]

    ev_A, ev_B, ev_Cr = [0] * E, [0] * E, [0] * E
    rec["bpc_A_postC"] = fp.eval_bpc_fusion(model, Aex, Aey, routes=ev_A)
    rec["bpc_B_postC"] = fp.eval_bpc_fusion(model, Bex, Bey, routes=ev_B)
    rec["bpc_Cret_postC"] = fp.eval_bpc_fusion(model, Crx, Cry, routes=ev_Cr)
    rec["fgt_A_full"] = rec["bpc_A_preB"] - rec["bpc_A_postC"]          # descriptive
    rec["b_degradation_B_postC"] = rec["bpc_B_postC"] - rec["bpc_B_postB"]  # B4 quantity
    rec["ledger"] = plc.ledger_snapshot_pr08(model, ledger, tr,
                                             {"A_eval": ev_A, "B_eval": ev_B,
                                              "C_retention": ev_Cr},
                                             {"A": int(Ax.shape[0]), "B": int(Bx.shape[0]),
                                              "C": int(Cx.shape[0])},
                                             boundary, int(a_expert))
    if freeze_floors:
        # end-of-run freeze audit: which slots are STILL frozen (Policy-A re-initialized
        # slots left the set via the guarded discard)
        rec["floor_freeze"]["slots_at_end"] = sorted(model.floor_freeze)
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def _fp(payload: dict) -> str:
    from seq.gpu_harness import config_fingerprint
    return config_fingerprint({"registry": REGISTRY_ID, **payload})


def _pr08_powered_path() -> str:
    return os.path.join(_results_root(), PR08_LEDDIR, POWERED_BASENAME)


def run_canary(res, seeds):
    """Doc §4 fail-loud canary (POWERED MODE ONLY): every science field of OFF.s{seed} must
    equal PR-08's claim.PRIM-LM.s{seed} EXACTLY. A missing PR-08 ledger or any mismatch is a
    SystemExit ABORT before the treatment arm's cells run. Returns the canary record stored
    in the ledger."""
    from seq.gpu_harness import load_results

    pr08_path = _pr08_powered_path()
    if not os.path.isfile(pr08_path):
        raise SystemExit(
            "CANARY ABORT (PR-2026-09-03-09 §4): the PR-08 powered ledger is MISSING at "
            f"{pr08_path} — the OFF control cannot be verified against "
            "claim.PRIM-LM.s{0..4}. The registered protocol aborts before the treatment arm "
            "runs; run this where the PR-08 ledger exists.")
    pr08 = load_results(pr08_path)
    checked = []
    for seed in seeds:
        cellkey = f"claim.OFF.s{seed}"
        refkey = f"claim.PRIM-LM.s{seed}"
        off = res.get(cellkey)
        ref = pr08.get(refkey)
        if not isinstance(off, dict) or not isinstance(ref, dict):
            raise SystemExit(
                f"CANARY ABORT (PR-2026-09-03-09 §4): missing cell for comparison "
                f"({cellkey} or {refkey} absent) — refusing to proceed to the treatment arm.")
        mm = canary_mismatches(off, ref)
        if mm:
            raise SystemExit(
                f"CANARY ABORT (PR-2026-09-03-09 §4): OFF {cellkey} does NOT reproduce PR-08 "
                f"{refkey} bit-identically — mismatched science fields: {mm}. Investigate "
                f"before the treatment arm runs (this also re-proves cross-process "
                f"determinism end-to-end). Ledger: {pr08_path}")
        checked.append(cellkey)
    rec = {"ok": True, "pr08_ledger": pr08_path, "cells_checked": checked,
           "ignored_fields": sorted(CANARY_IGNORE),
           "note": ("every science field of OFF == PR-08 claim.PRIM-LM EXACTLY "
                    "(bpc_*, ledger dict, a_expert, fgt_A_full, b_degradation_B_postC)")}
    print(f"[pr09] CANARY PASS: {len(checked)} OFF cells bit-identical to PR-08 "
          f"claim.PRIM-LM ({pr08_path})", flush=True)
    return rec, pr08


def _smoke_checks(res):
    """Smoke-only fail-loud plumbing assertions (doc §4 second canary + the treatment
    wiring): (i) the A-phase is bit-identical between arms (equal bpc_A_preB); (ii) the
    frozen-slots audit is present, non-empty and actually calibrated (pinned mu < 1e8);
    (iii) frozen slots that survived to the end show their A-end PINNED floors in the final
    ledger (the freeze really skipped the floor updates)."""
    off = res["claim.OFF.s0"]
    ff = res["claim.FROZEN-FLOORS.s0"]
    import math
    if off["bpc_A_preB"] != ff["bpc_A_preB"]:
        raise AssertionError(
            f"A-PHASE IDENTITY BROKEN: OFF bpc_A_preB={off['bpc_A_preB']!r} != "
            f"FROZEN-FLOORS bpc_A_preB={ff['bpc_A_preB']!r} — the freeze must activate "
            "only AFTER block A.")
    audit = ff.get("floor_freeze")
    assert isinstance(audit, dict) and audit.get("slots"), \
        "FROZEN-FLOORS smoke cell carries no frozen-slots audit"
    assert all(m < 1e8 for m in audit["pinned_mu"].values()), \
        "pinned mu values must be CALIBRATED (post-A), not the 1e9 init sentinel"
    still = [s for s in audit.get("slots_at_end", [])]
    per_expert = {p["slot"]: p for p in ff["ledger"]["per_expert"] if p.get("committed")}
    pinned_ok = []
    for s in still:
        p = per_expert.get(s)
        if p is None:
            continue
        want_sigma = round(math.sqrt(audit["pinned_var"][str(s)]), 4)
        assert p["mu"] == round(audit["pinned_mu"][str(s)], 4) and p["sigma"] == want_sigma, \
            f"slot {s} was frozen yet its floor moved (mu {p['mu']} vs pinned " \
            f"{round(audit['pinned_mu'][str(s)], 4)}) — the guarded branch did not hold"
        pinned_ok.append(s)
    return {"a_phase_bit_identity": True,
            "bpc_A_preB": off["bpc_A_preB"],
            "frozen_slots_at_A_end": audit["slots"],
            "frozen_slots_at_end": audit.get("slots_at_end"),
            "pinned_floors_verified_through_B_and_C": pinned_ok,
            "note": ("equal bpc_A_preB across arms = A-phase bit-identity; pinned floors of "
                     "still-frozen slots unchanged through B/C training")}


def run(*, smoke: bool, results_path=None, force_smoke_path: bool = False,
        powered_cpu: bool = False):
    """Execute the PR-09 protocol: --smoke (CPU plumbing), --powered (the GPU claim
    campaign), or --powered-cpu (the doc section-6 CPU-feasible fallback: identical to
    --powered except the CUDA guard is skipped and the ledger meta records powered_cpu +
    the fallback note)."""
    # BAR-0 FIRST: resolve + guard the results path before any heavy import or write.
    path = resolve_results_path(results_path, smoke=smoke, force_smoke_path=force_smoke_path)

    import time

    import torch

    from seq import blockdrift_claim as claim          # PR-07' protocol pieces, imported VERBATIM
    from seq import fusion_probe as fp                 # the fused column, imported VERBATIM
    from seq.gpu_harness import load_results, _save
    from seq.recall_gate import archive_run

    torch.set_num_threads(int(os.environ.get("BENCH_THREADS", "8")))

    if smoke:
        smoke_segs = 64                        # per training block (plc smoke precedent;
                                               # held-out eval slices stay FULL)
        seeds = (0,)
    else:
        if needs_cuda("powered-cpu" if powered_cpu else "powered"):
            require_cuda(torch.cuda.is_available())  # refuse BEFORE anything else on a CPU-only box
        # (--powered-cpu skips that guard by design: the doc section-6 CPU-feasible fallback,
        # PR-09 doc §6, precedent PR-08 addendum 2026-09-08 #2)
        smoke_segs = None
        seeds = CLAIM_SEEDS

    # ---------------- data: the 3-block stream + the pinned eval slices (plc §2, inherited) ---
    slices = pin_slices()
    A_all, B_all = claim.fetch_corpora()       # archived corpora; download failure = ABORT
    A_train = A_all[slices["A_train"][1]:slices["A_train"][2]]
    A_eval = A_all[slices["A_eval"][1]:slices["A_eval"][2]]
    C_ret = A_all[slices["C_retention"][1]:slices["C_retention"][2]]
    C_train = A_all[slices["C_train"][1]:slices["C_train"][2]]
    B_train = B_all[: int(len(B_all) * slices["B_train"][2])]
    B_eval = B_all[int(len(B_all) * slices["B_eval"][1]):]
    chars = sorted(set(A_train) | set(A_eval) | set(C_ret) | set(C_train) | set(B_all))
    vocab = {c: i for i, c in enumerate(chars)}
    V = len(vocab)

    def _segs(text):
        return claim.make_segments(text, vocab)

    Ax, Ay = _segs(A_train)
    Bx, By = _segs(B_train)
    Cx, Cy = _segs(C_train)
    Aex, Aey = _segs(A_eval)
    Bex, Bey = _segs(B_eval)
    Crx, Cry = _segs(C_ret)
    if smoke_segs:
        Ax, Ay = Ax[:smoke_segs], Ay[:smoke_segs]
        Bx, By = Bx[:smoke_segs], By[:smoke_segs]
        Cx, Cy = Cx[:smoke_segs], Cy[:smoke_segs]
    data = (Ax, Ay, Bx, By, Cx, Cy, Aex, Aey, Bex, Bey, Crx, Cry)

    # ---------------- ledger + meta (NO lr-selection leg: LR is frozen by inheritance) --------
    res = load_results(path)
    bb_params = sum(p.numel() for p in fp._PlainWrap(V, 0).lm.parameters())
    per_expert = sum(p.numel() for p in fp.PCExpertHead(64, H_SMALL, V).parameters())
    res["meta"] = {
        "registry": REGISTRY_ID, "smoke": bool(smoke),
        "lane": "CLAIM (frozen pre-registration)" if not smoke else "SMOKE (plumbing only)",
        "doc": "docs/preregistry/2026-09-09-floorfreeze-routing-repair.md",
        "arms": list(ARMS), "claim_seeds": list(seeds),
        "seg": SEG, "batch_segs": BATCH_SEGS,
        "lr_rule": ("FROZEN 3e-3 for BOTH arms — inherited from PR-08's registered "
                    "lr_selection.fusion (doc §2: no re-selection; comparability with PR-08 "
                    "is the point). This runner has NO lr-selection leg."),
        "lr_inheritance": {"lr": LR_FROZEN,
                           "source_registry": plc.REGISTRY_ID,
                           "source_ledger": f"results/{PR08_LEDDIR}/powered.json "
                                            "(lr_selection.fusion)"},
        "tissue": {"E_pool": E_POOL, "m_max": M_MAX, "freeze_min_seen": FREEZE_MIN_SEEN,
                   "z_novel": Z_NOVEL, "h_small": H_SMALL, "floor_ema": FLOOR_EMA,
                   "fresh_head_seed_formula": f"{FRESH_HEAD_SEED_BASE} + 17*slot + n_evictions",
                   "floor_freeze": "the PR-09 lever (binary, default-OFF): a plain Python set "
                                   "of committed-at-A-end slot indices pinned on the model "
                                   "after block A; guarded branches in seq/fusion_probe.py "
                                   "_expert_train (skip floor updates) + seq/prizma_lm_claim.py "
                                   "route_pr08 (re-initialized slot leaves the set)",
                   "params_backbone": bb_params, "params_per_expert": per_expert,
                   "params_pool_e4": per_expert * E_POOL},
        "pinned_slices": {k: list(v) for k, v in slices.items()},
        "stream_lengths": {"A_train": len(A_train), "B_train": len(B_train),
                           "C_train": len(C_train), "A_eval": len(A_eval),
                           "B_eval": len(B_eval), "C_retention": len(C_ret)},
        "vocab": V, "threads": torch.get_num_threads(),
        "c_range_repair": plc.C_RANGE_REPAIR_NOTE,      # inherited PR-08 disclosures (§2)
        "c_retention_prose_note": plc.C_RET_PROSE_NOTE,
        "bars": {"P1_frac_bar": P1_FRAC_BAR, "G1_margin": G1_MARGIN, "G2_margin": G2_MARGIN,
                 "alpha": ALPHA, "holm_family": list(HOLM_FAMILY),
                 "p1_form": "exact means, no CI (the PR-08 B3 form); OFF frac is the control",
                 "t_isf_convention": "UPPER-TAIL p in (0, 0.5] (the PR-03 lesson)"},
        "boundary_note": ("A-expert := the committed slot with max train_A at C start; frac = "
                          "TRAINING-ledger fraction of C segments in the first "
                          f"{B3_WINDOW_BATCHES} C-batches assigned to it (P1's quantity, "
                          "PR-08 B3a's form)"),
        "canary": ("doc §4 (powered only): OFF.s{seed} must equal PR-08 claim.PRIM-LM.s{seed} "
                   "EXACTLY on every science field; mismatch or missing PR-08 ledger = ABORT "
                   "before the treatment arm. FROZEN-FLOORS A-phase bit-identity is asserted "
                   "in smoke via equal bpc_A_preB."),
    }
    if powered_cpu:
        res["meta"]["powered_cpu"] = True
        res["meta"]["compute_fallback_note"] = POWERED_CPU_FALLBACK_NOTE
    _save(res, path)
    print(f"[pr09] corpora: A={Ax.shape[0]} B={Bx.shape[0]} C={Cx.shape[0]} segs; "
          f"eval A={Aex.shape[0]} B={Bex.shape[0]} Cret={Crx.shape[0]}; vocab={V}; "
          f"smoke={smoke}; lr={LR_FROZEN} (frozen, inherited); results={path}", flush=True)

    # ---------------- the 2 arms x seeds (OFF first; the canary runs between arms) -------------
    n_cells_total = len(ARMS) * len(seeds)
    first_cell_s = None
    for arm in ARMS:
        freeze_floors = (arm == "FROZEN-FLOORS")
        for seed in seeds:
            cellkey = f"claim.{arm}.s{seed}"
            cfgsig = _fp({"leg": "claim", "arm": arm, "seed": seed, "lr": LR_FROZEN,
                          "smoke": bool(smoke), "vocab": V, "smoke_segs": smoke_segs,
                          "seg": SEG, "batch_segs": BATCH_SEGS,
                          "slices": res["meta"]["pinned_slices"],
                          "stream_lengths": res["meta"]["stream_lengths"],
                          "tissue": dict(res["meta"]["tissue"],
                                         floor_freeze=freeze_floors),  # the lever is IN the
                                                                        # fingerprint
                          "bars": res["meta"]["bars"]})
            prior = res.get(cellkey)
            if isinstance(prior, dict) and prior.get("cfgsig") == cfgsig and prior.get("complete"):
                print(f"[pr09] {arm} seed {seed}: resumed from ledger "
                      f"(bpc_A_postC={prior['bpc_A_postC']:.3f})", flush=True)
                continue
            if isinstance(prior, dict):
                raise SystemExit(f"cell {cellkey} exists at a foreign config fingerprint "
                                 f"({prior.get('cfgsig')} != {cfgsig}) — refusing to resume")
            t0 = time.time()
            rec = run_cell(V, seed, data, LR_FROZEN, freeze_floors=freeze_floors)
            rec.update({"arm": arm, "seed": seed, "lr": LR_FROZEN, "cellkey": cellkey,
                        "cfgsig": cfgsig, "complete": True,
                        "wall_s": round(time.time() - t0, 1)})
            res[cellkey] = rec
            _save(res, path)                    # crash-safe after every cell
            if first_cell_s is None:
                first_cell_s = rec["wall_s"]
                proj = plc.budget_projection(first_cell_s, n_cells_total)
                res.setdefault("meta", {})["budget_projection"] = proj
                _save(res, path)
                print(f"[pr09] budget projection from first cell: {proj['projected_min']} min "
                      f"for {n_cells_total} cells"
                      f"{'  *** ' + proj['note'] if proj['warn'] else ''}", flush=True)
            ff_note = ""
            if freeze_floors:
                ff_note = (f" frozen_slots={rec['floor_freeze']['slots']}"
                           f"->at_end={rec['floor_freeze']['slots_at_end']}")
            print(f"[pr09] {arm} seed {seed}: A_preB={rec['bpc_A_preB']:.3f} "
                  f"A_postC={rec['bpc_A_postC']:.3f} B_postB={rec['bpc_B_postB']:.3f} "
                  f"B_postC={rec['bpc_B_postC']:.3f} Cret_postC={rec['bpc_Cret_postC']:.3f} "
                  f"frac_C(1-20)to_A={rec['ledger']['boundary_window']['frac_to_A_expert']}"
                  f"{ff_note} wall={rec['wall_s']}s", flush=True)
        if not smoke and arm == "OFF":
            # ---------------- doc §4 fail-loud canary: BEFORE the treatment arm ---------------
            canary_rec, pr08 = run_canary(res, seeds)
            res["canary"] = canary_rec
            _save(res, path)

    # ---------------- smoke-only plumbing assertions (fail-loud) -------------------------------
    smoke_checks = None
    if smoke:
        smoke_checks = _smoke_checks(res)
        res["smoke_checks"] = smoke_checks
        _save(res, path)

    # ---------------- RETENTION (docs/RETENTION.md): archive BEFORE any verdict ----------------
    raw_archive = archive_run(res, label=f"floorfreeze-{REGISTRY_ID}")
    res.setdefault("meta", {})["raw_archive"] = raw_archive

    # ---------------- the frozen §3/§5 verdict (claim mode) / plumbing report (smoke) ----------
    report = {"registry": REGISTRY_ID, "smoke": bool(smoke), "arms": list(ARMS),
              "claim_seeds": list(seeds), "lr": LR_FROZEN,
              "lr_inheritance": res["meta"]["lr_inheritance"],
              "pinned_slices": res["meta"]["pinned_slices"],
              "raw_archive": raw_archive,
              "cells": {k: res[k] for k in sorted(res) if k.startswith("claim.")}}
    if powered_cpu:
        report["powered_cpu"] = True
    if smoke:
        report["smoke_checks"] = smoke_checks
    else:
        off = [res[f"claim.OFF.s{s}"] for s in seeds]
        ff = [res[f"claim.FROZEN-FLOORS.s{s}"] for s in seeds]
        frozen_ck = [pr08[f"claim.FROZEN-CHECKPOINT.s{s}"] for s in seeds]   # REUSED from PR-08
        verdict = claim_verdict(off, ff, frozen_ck, alpha=ALPHA)
        report["verdict"] = verdict
        report["canary"] = res.get("canary")
        res["verdict"] = verdict
    res["report"] = report
    _save(res, path)

    _print_report(report, path, smoke=smoke)
    return report


# ------------------------------------------------------------------ report + CLI ----------------
def _print_report(report, path, *, smoke):
    print("\n" + "=" * 78, flush=True)
    print(f"  {REGISTRY_ID} — {'SMOKE (PLUMBING-ONLY)' if smoke else 'POWERED CLAIM CAMPAIGN'}",
          flush=True)
    print("=" * 78, flush=True)
    for key, c in report["cells"].items():
        arm = c.get("arm", c.get("config"))
        print(f"    {key:<34} lr={c['lr']:.0e} A_preB={c['bpc_A_preB']:.3f} "
              f"A_postC={c['bpc_A_postC']:.3f} B_postB={c['bpc_B_postB']:.3f} "
              f"B_postC={c['bpc_B_postC']:.3f} Cret={c['bpc_Cret_postC']:.3f} "
              f"wall={c['wall_s']}s", flush=True)
        led = c.get("ledger")
        if led:
            bw = led["boundary_window"]
            print(f"      ledger: committed={led['n_committed']} a_expert={led['a_expert']} "
                  f"recruits={len(led['recruits'])} evictions={len(led['evictions'])} "
                  f"vetoed_segs={led['vetoed_novel_segments']} "
                  f"frac_C(1-20)to_A={bw['frac_to_A_expert']}", flush=True)
        if c.get("floor_freeze"):
            ffa = c["floor_freeze"]
            print(f"      floor_freeze: slots@A_end={ffa['slots']} "
                  f"slots@end={ffa.get('slots_at_end')} "
                  f"pinned_mu={ {k: round(v, 3) for k, v in ffa['pinned_mu'].items()} }",
                  flush=True)
    if smoke:
        sc = report.get("smoke_checks") or {}
        print(f"    [SMOKE CHECK] A-phase bit-identity (equal bpc_A_preB): "
              f"{sc.get('a_phase_bit_identity')} (bpc_A_preB={sc.get('bpc_A_preB'):.6f})",
              flush=True)
        print(f"    [SMOKE CHECK] frozen slots @A_end={sc.get('frozen_slots_at_A_end')} "
              f"@end={sc.get('frozen_slots_at_end')}; pinned floors verified through B/C: "
              f"{sc.get('pinned_floors_verified_through_B_and_C')}", flush=True)
        print("  [SMOKE] numbers are plumbing-only and MEANINGLESS — do NOT cite.", flush=True)
    else:
        v = report["verdict"]
        p1 = v["P1"]
        print(f"    P1: ff_frac_mean={p1['frac_mean']:.3f} (bar >= {p1['frac_bar']}) "
              f"[off control mean={p1['off_frac_mean']:.3f}] -> {p1['status']}", flush=True)
        for bar in ("G1", "G2"):
            b = v[bar]
            center = b["mean"] if bar == "G1" else b["delta"]
            print(f"    {bar}: mean={center:.4f} CI=[{b['ci'][0]:.4f}, {b['ci'][1]:.4f}] "
                  f"p_raw={b['p_raw']:.4f} p_holm={b['p_holm']:.4f} -> {b['status']}", flush=True)
        for arm, d in v["B4"].items():
            print(f"    B4 {arm}: B-eval degradation post-C mean {d['mean']:+.4f} "
                  f"(reported, not gated)", flush=True)
        for arm, ch in v["churn"].items():
            tot_r = sum(c["recruits"] for c in ch.values())
            tot_e = sum(c["evictions"] for c in ch.values())
            tot_v = sum(c["vetoed_novel_segments"] for c in ch.values())
            print(f"    churn {arm}: recruits={tot_r} evictions={tot_e} "
                  f"vetoed_segs={tot_v} (per-seed: {ch})", flush=True)
        print(f"    per-seed fracs: OFF={v['P1']['off_frac_per_seed']} "
              f"FROZEN-FLOORS={v['P1']['frac_per_seed']}", flush=True)
        print(f"  OUTCOME: {v['outcome']}", flush=True)
        print(f"  {v['verdict']}", flush=True)
        for br in v["branches"]:
            print(f"  branch: {br}", flush=True)
    print(f"  ledger: {path}", flush=True)
    print(f"  raw archive: {report.get('raw_archive')}", flush=True)
    print("=" * 78, flush=True)


def _build_parser():
    """Argparse guard (plc pattern): ONLY --smoke / --powered / --powered-cpu / --out /
    --force-smoke-path; an unknown or typo'd flag exits non-zero BEFORE anything runs.
    Exactly one mode is required."""
    p = argparse.ArgumentParser(
        prog="floorfreeze_claim",
        description="PR-2026-09-03-09 claim runner (the floor-freeze routing repair). --smoke "
                    "= tiny CPU plumbing run; --powered = the frozen claim campaign (requires "
                    "CUDA); --powered-cpu = the doc section-6 CPU-feasible fallback "
                    "(protocol-identical claim campaign on an explicitly authorized CPU box). "
                    "An unknown flag is rejected without launching anything.")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="tiny plumbing-only run (CPU, minutes)")
    mode.add_argument("--powered", action="store_true",
                      help="the claim campaign (refuses to start without a CUDA device)")
    mode.add_argument("--powered-cpu", action="store_true",
                      help="the claim campaign via the doc section-6 CPU-feasible fallback "
                           "(identical protocol and ledger; runs WITHOUT a CUDA device; "
                           "precedent PR-08 addendum 2026-09-08 #2)")
    p.add_argument("--out", default=None,
                   help="explicit results JSON path (overrides the default)")
    p.add_argument("--force-smoke-path", action="store_true",
                   help="let a --smoke run write the powered ledger it was pointed at "
                        "(default: REFUSED — separate ledgers)")
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(argv)   # SystemExit non-zero on unknown args: nothing runs
    run(smoke=args.smoke, results_path=args.out, force_smoke_path=args.force_smoke_path,
        powered_cpu=args.powered_cpu)


if __name__ == "__main__":
    main()
