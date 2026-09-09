"""
PR-2026-09-03-10 CLAIM RUNNER — trunk-lr scheduling on the returning block (the registered
successor of PR-09's floor-freeze negative). Executes the FROZEN pre-registration
`docs/preregistry/2026-09-09-trunklr-routing-repair.md` VERBATIM. One script, three modes
(the prizma_lm_claim pattern, mirrored):

  python seq/trunklr_claim.py --smoke        # tiny CPU plumbing run -> the SMOKE ledger
  python seq/trunklr_claim.py --powered      # the claim campaign (refuses without CUDA)
  python seq/trunklr_claim.py --powered-cpu  # the doc §5 CPU-feasible fallback

WHAT IS FROZEN (pre-reg §2 — inherited from PR-08's registered text, reused BY IMPORT from
seq/prizma_lm_claim.py, NOT re-declared): stream, slices, corpora, tissue constants, SEG/
BATCH_SEGS, seeds 0-4, PR-08 routing semantics. LR: 3e-3 FROZEN for blocks A and B in BOTH
arms (PR-08 lr_selection.fusion inheritance); the treatment arm scales the BACKBONE lr to
7.5e-4 (= 3e-3 x 0.25, frozen a priori — NOT tuned on data) during block C ONLY. Tissue
per-expert local optimizers keep 3e-3 everywhere; floors stay LIVE (PR-09 settled that axis).

Arms (2, exclusive):
  OFF            — PR-08 PRIM-LM verbatim (never passes backbone_lr; off-identity proven by
                   the powered canary against PR-08's stored PRIM cells).
  TRUNK-LR-0.25C — identical cell; the block-C training call passes the guarded
                   `backbone_lr=7.5e-4` kwarg of seq/prizma_lm_claim.py train_pr08 (the
                   backbone AdamW alone is scaled; the tissue _expert_train keeps `lr`).
The lever is guarded default-off: backbone_lr=None is the byte-identical PR-08 path.

BARS (doc §3, exact; n=5 seeds 0-4):
  P1 (primary, repair): mean over TRUNK-LR-0.25C seeds of ledger.boundary_window.
      frac_to_A_expert >= 0.5 — EXACT MEANS, no CI (the PR-08 B3 form). OFF's frac mean
      is the recorded control (PR-08 measured 0.428; PR-09 re-measured 0.428).
  G1 (retention guard): BPC(A-eval, post-C) - BPC(A-eval, pre-B) <= 0.05 for TRUNK-LR-0.25C
      (one-sample, CI upper <= 0.05 AND Holm p < 0.05; CI lower > 0.05 -> FAIL; straddle ->
      INCONCLUSIVE).
  G2 (adaptation guard): BPC(C-ret, post-C) advantage of TRUNK-LR-0.25C vs PR-08's
      FROZEN-CHECKPOINT >= 0.10 (Welch, CI lower >= 0.10; straddle -> INCONCLUSIVE). The
      FROZEN-CHECKPOINT baseline cells are REUSED from the PR-08 powered ledger (disclosed:
      no tissue, lever-independent, same seeds and frozen protocol).
  Holm family = [G1, G2]. Reported, never gated: B4 trunk drift per arm; churn per arm;
  per-seed fracs for both arms.

Overall (doc §4, pre-committed): CLAIMED iff P1 PASS and G1 PASS and G2 PASS;
NEGATIVE-GUARD if P1 PASS and any guard FAIL; NEGATIVE if P1 FAIL; any guard INCONCLUSIVE
-> INCONCLUSIVE (seed extension to n=10, seeds 5-9, PRE-AUTHORIZED by the registration).

CANARY (doc §2, fail-loud, POWERED MODE ONLY): after the OFF arm's 5 cells, every science
field of OFF.s{seed} must equal PR-08's claim.PRIM-LM.s{seed} EXACTLY (comparator + ignore
set REUSED from seq/floorfreeze_claim.py). Missing PR-08 ledger or any mismatch = ABORT
before the treatment arm. The treatment arm's A and B phases must match OFF bit-for-bit
(divergence begins at C only) — asserted in smoke via equal bpc_A_preB AND bpc_B_postB,
plus a lever-fires check (bpc_B_postC must DIFFER between arms).
"""
from __future__ import annotations

import argparse
import os
import sys

# Light imports only at module top: the pure layer stays importable + unit-testable without
# torch (surprise_claim discipline). PR-08 plumbing and the PR-09 canary comparator are
# reused BY IMPORT, not copy-pasted.
try:
    from .stats import holm_correction
except ImportError:                                   # run as a bare script: bootstrap sys.path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from seq.stats import holm_correction

import seq.prizma_lm_claim as plc                     # PR-08 protocol pieces, reused VERBATIM
import seq.floorfreeze_claim as ffc                   # canary comparator + aliases, reused


# ================================================================== frozen protocol constants ====
REGISTRY_ID = "PR-2026-09-03-10"
LEDDIR = "trunklr_PR-2026-09-03-10"             # under $PRIZMA_RESULTS (default ./results)
SMOKE_BASENAME = "smoke.json"
POWERED_BASENAME = "powered.json"

ARMS = ("OFF", "TRUNK-LR-0.25C")                # control first — the canary runs between arms
CLAIM_SEEDS = plc.CLAIM_SEEDS                   # (0, 1, 2, 3, 4) — inherited, never substituted

# ---- protocol constants inherited from PR-08 BY IMPORT (doc §2) ----
LR_FROZEN = ffc.LR_FROZEN               # 3e-3 — blocks A and B, both arms (PR-08 inheritance)
TRUNK_LR_SCALE_C = 0.25                 # doc §2: frozen a priori, NOT tuned on data
BACKBONE_LR_C = LR_FROZEN * TRUNK_LR_SCALE_C      # 7.5e-4 — block C backbone only, treatment
SEG = plc.SEG
BATCH_SEGS = plc.BATCH_SEGS
E_POOL = plc.E_POOL
M_MAX = plc.M_MAX
FREEZE_MIN_SEEN = plc.FREEZE_MIN_SEEN
Z_NOVEL = plc.Z_NOVEL
H_SMALL = plc.H_SMALL
FLOOR_EMA = plc.FLOOR_EMA
FRESH_HEAD_SEED_BASE = plc.FRESH_HEAD_SEED_BASE
B3_WINDOW_BATCHES = plc.B3_WINDOW_BATCHES

# ---- bars (doc §3, exact — the PR-09 forms, same numbers) ----
P1_FRAC_BAR = 0.5
G1_MARGIN = 0.05
G2_MARGIN = 0.10
ALPHA = plc.ALPHA
HOLM_FAMILY = ("G1", "G2")

PR08_LEDDIR = plc.LEDDIR                # "prizma_lm_PR-2026-09-03-08" — the canary/G2 source
POWERED_CPU_FALLBACK_NOTE = (
    "POWERED VIA THE DOC SECTION-5 CPU-FEASIBLE FALLBACK (PR-10 doc §5; precedent = PR-08 "
    "doc addendum 2026-09-08 #2 + PR-09/PR-10 powered-cpu executions): protocol-identical "
    "to the A100 tier; --powered (CUDA-refusing) remains available for the GPU session.")


# ==================================================================== paths + BAR-0 refusal ======
def _results_root() -> str:
    root = os.environ.get("PRIZMA_RESULTS", os.path.join(os.path.dirname(__file__), "..", "results"))
    return os.path.abspath(root)


def _default_results_path(smoke: bool) -> str:
    return os.path.join(_results_root(), LEDDIR, SMOKE_BASENAME if smoke else POWERED_BASENAME)


def resolve_results_path(explicit=None, *, smoke: bool = False, force_smoke_path: bool = False) -> str:
    """Resolve the results path and enforce the smoke/powered file separation (the plc
    refusal pattern, pointed at the PR-10 led dir)."""
    path = explicit if explicit else _default_results_path(smoke)
    if smoke and not force_smoke_path:
        powered = _default_results_path(smoke=False)
        if os.path.normcase(os.path.abspath(path)) == os.path.normcase(os.path.abspath(powered)):
            raise SystemExit(
                f"refusing: a --smoke run was pointed at the powered ledger ({powered}). Smoke and "
                f"campaign results use separate files by default (PR-2026-09-03-10 §5 ledger "
                f"separation; see results/campaign_2026-06-08/CONTAMINATION.md). Pass --out <path> "
                f"to write the smoke elsewhere, or --force-smoke-path to override deliberately.")
    return path


def require_cuda(has_cuda: bool) -> None:
    """Pure guard for the --powered mode (unit-testable without a CUDA box). --powered-cpu
    (the registered CPU fallback) skips this guard by design."""
    if not has_cuda:
        raise SystemExit(
            "refusing: --powered executes the PR-2026-09-03-10 claim campaign and requires a "
            "CUDA device. No CUDA device is visible here. Use --smoke for the CPU plumbing "
            "run, or --powered-cpu for the registered CPU fallback.")


# REUSED from plc by import: only --powered demands CUDA. TRUE alias, not a fork (pinned
# by tests against ffc's alias — same plc object).
needs_cuda = plc.needs_cuda

# PR-10 inherits PR-08's §2 stream verbatim -> the pinned-slice table IS plc.pin_slices.
pin_slices = plc.pin_slices

# The §2 canary comparator + ignore set are REUSED from PR-09's runner (identical contract:
# OFF vs PR-08 claim.PRIM-LM bit-identity on every science field).
canary_mismatches = ffc.canary_mismatches
CANARY_IGNORE = ffc.CANARY_IGNORE


# ==================================================================== PURE: verdict (§3+§4) ======
def claim_verdict(off, tl, frozen_ck, *, alpha=ALPHA):
    """The frozen §3/§4 verdict as a PURE function over per-seed record dicts (unit-tested
    without torch). Args: off = OFF control seeds (THIS ledger), tl = TRUNK-LR-0.25C seeds
    (THIS ledger), frozen_ck = PR-08's FROZEN-CHECKPOINT cells (REUSED; only bpc_Cret_postC
    is read).

    P1: exact means, no CI: mean(tl boundary_window.frac_to_A_expert) >= 0.5 -> PASS else
        FAIL; the OFF frac mean is recorded as the control.
    G1: one-sample, diffs = bpc_A_postC - bpc_A_preB per tl seed; PASS iff CI upper <= 0.05
        AND Holm p < alpha; CI lower > 0.05 -> FAIL; straddle -> INCONCLUSIVE.
    G2: Welch advantage mean(frozen_ck bpc_Cret_postC) - mean(tl bpc_Cret_postC) >= 0.10;
        PASS iff CI lower >= 0.10 AND Holm p < alpha; CI upper < 0.10 -> FAIL; straddle ->
        INCONCLUSIVE.
    Holm family = [G1, G2]. Overall: any guard INCONCLUSIVE -> INCONCLUSIVE; P1 PASS + G1 +
    G2 PASS -> CLAIMED; P1 PASS + any guard FAIL -> NEGATIVE-GUARD; P1 FAIL -> NEGATIVE.
    """
    tl_fracs = [float(r["ledger"]["boundary_window"]["frac_to_A_expert"]) for r in tl]
    off_fracs = [float(r["ledger"]["boundary_window"]["frac_to_A_expert"]) for r in off]
    tl_frac_mean = sum(tl_fracs) / len(tl_fracs)
    off_frac_mean = sum(off_fracs) / len(off_fracs)
    p1_ok = tl_frac_mean >= P1_FRAC_BAR
    p1 = {"frac_per_seed": tl_fracs, "frac_mean": tl_frac_mean, "frac_bar": P1_FRAC_BAR,
          "status": "PASS" if p1_ok else "FAIL",
          "off_frac_per_seed": off_fracs, "off_frac_mean": off_frac_mean,
          "off_note": "OFF is the recorded control (PR-08 PRIM-LM measured 0.428 at n=5)"}

    g1 = plc._onesample_margin([r["bpc_A_postC"] - r["bpc_A_preB"] for r in tl], G1_MARGIN)
    g2 = plc._welch_margin([r["bpc_Cret_postC"] for r in tl],
                           [r["bpc_Cret_postC"] for r in frozen_ck], G2_MARGIN)
    g2["baseline_note"] = ("baseline = PR-08 FROZEN-CHECKPOINT cells REUSED from "
                           f"results/{PR08_LEDDIR}/powered.json (no tissue, lever-independent, "
                           "same seeds + frozen protocol — disclosed, doc §3)")

    holm = holm_correction([g1["p_raw"], g2["p_raw"]], alpha=alpha)
    g1["p_holm"] = holm[0]["p_adj"]
    g2["p_holm"] = holm[1]["p_adj"]
    g1["status"] = plc._bar_status(g1["ci"], G1_MARGIN, g1["p_holm"], "below", alpha)
    g2["status"] = plc._bar_status(g2["ci"], G2_MARGIN, g2["p_holm"], "above", alpha)

    def _b4(recs):
        d = [r["bpc_B_postC"] - r["bpc_B_postB"] for r in recs]
        return {"per_seed": d, "mean": sum(d) / len(d)}

    def _churn(recs):
        out = {}
        for r in recs:
            led = r["ledger"]
            out[f"s{r['seed']}"] = {"recruits": len(led["recruits"]),
                                    "evictions": len(led["evictions"]),
                                    "vetoed_novel_segments": led["vetoed_novel_segments"]}
        return out

    b4 = {"OFF": _b4(off), "TRUNK-LR-0.25C": _b4(tl)}
    churn = {"OFF": _churn(off), "TRUNK-LR-0.25C": _churn(tl)}

    branches = []
    guard_statuses = [g1["status"], g2["status"]]
    if "INCONCLUSIVE" in guard_statuses:
        outcome = "INCONCLUSIVE"
        branches.append("§4 (any guard CI straddles its margin): INCONCLUSIVE for that guard; "
                        "seed extension to n=10 (seeds 5-9) is PRE-AUTHORIZED by this "
                        "registration, executed time permitting (the 03:55 wind-down rule "
                        "always wins). INCONCLUSIVE is never PASS (PR-03 mirror).")
    elif p1_ok and all(s == "PASS" for s in guard_statuses):
        outcome = "CLAIMED"
    elif p1_ok:
        outcome = "NEGATIVE-GUARD"
    else:
        outcome = "NEGATIVE"

    if outcome == "CLAIMED":
        verdict_text = ("CLAIMED — P1 PASS with G1 (retention-A) and G2 (adaptation-C) intact: "
                        "the routing scatter is monotone in trunk plasticity and separable from "
                        "adaptation; PR-LM-1 designs inherit C-block trunk-lr scaling (doc §4).")
    elif outcome == "NEGATIVE-GUARD":
        broke = [name for name, st in zip(("G1", "G2"), guard_statuses) if st == "FAIL"]
        verdict_text = ("NEGATIVE (with lead) — P1 PASS but " + " and ".join(broke) +
                        " FAIL: the lever buys routing at the cost of the column. A dose "
                        "ladder (0.5 / 0.1) becomes a NEW prereg with the guard tradeoff as "
                        "its object (doc §4).")
        branches.append("§4 (P1 PASS + guard FAIL): record which guard broke — "
                        f"{', '.join(broke)}; dose ladder (0.5 / 0.1) as a NEW prereg.")
    elif outcome == "NEGATIVE":
        verdict_text = ("NEGATIVE — P1 FAIL at the 0.25 dose: trunk-lr scheduling insufficient "
                        "to repair the returning-boundary routing. Next candidates, each its "
                        "own prereg: (a) C-block trunk lr = 0 on the shared predictor with a "
                        "separate C-adapter (capacity control required), (b) floor "
                        "re-anchoring combined with a slow trunk (doc §4).")
        branches.append("§4 (P1 FAIL): 0.25 dose insufficient; next candidates (a) C-adapter "
                        "with capacity control, (b) floor re-anchoring + slow trunk — each "
                        "its own prereg. No dose re-tuning without a new registration.")
    else:
        verdict_text = ("INCONCLUSIVE — a guard CI straddles its margin (PR-03 mirror: never "
                        "PASS). Pre-committed §4 branch applies (see 'branches').")

    return {"outcome": outcome, "verdict": verdict_text, "branches": branches,
            "P1": p1, "G1": g1, "G2": g2, "B4": b4, "churn": churn,
            "holm_family": [{"bar": "G1", "p_raw": g1["p_raw"], "p_holm": g1["p_holm"],
                             "status": g1["status"]},
                            {"bar": "G2", "p_raw": g2["p_raw"], "p_holm": g2["p_holm"],
                             "status": g2["status"]}],
            "alpha": alpha}


# ================================================================================= runner ========
def run_cell(vocab_size, seed, data, lr, *, trunk_lr_c):
    """PR-10 cell — plc.run_routed's structure VERBATIM (same phase order, same eval order,
    same bpc keys, same ledger snapshot, same boundary window, same B4 quantity) plus the
    trunk-lr wiring:
      OFF            (trunk_lr_c=None): identical to plc.run_routed's PRIM-LM path — the
                     guarded kwarg is NEVER passed (off-identity; the powered canary proves
                     it against PR-08's stored records).
      TRUNK-LR-0.25C (trunk_lr_c=BACKBONE_LR_C): the block-C training call alone passes
                     `backbone_lr=7.5e-4`; blocks A and B use the frozen 3e-3 in both arms,
                     so A/B phases are bit-identical across arms (divergence begins at C)."""
    import time

    import torch
    from seq import fusion_probe as fp

    t0 = time.time()
    Ax, Ay, Bx, By, Cx, Cy, Aex, Aey, Bex, Bey, Crx, Cry = data
    E = E_POOL
    model = fp.build_model(vocab_size, seed, E)
    ledger = plc._fresh_ledger()
    tr = {"A": [0] * E, "B": [0] * E, "C": [0] * E}
    rec = {"config": ("PRIM-LM+trunk-lr-0.25C" if trunk_lr_c is not None else "PRIM-LM"),
           "E": E, "seed": seed, "lr": lr}

    # ---- block A (backbone trains; the first slot recruits into the empty pool) ----
    plc.train_pr08(model, Ax, Ay, lr, "A", ledger, seed=seed)
    tr["A"] = model.n_segments[:]
    rec["bpc_A_preB"] = fp.eval_bpc_fusion(model, Aex, Aey)

    # ---- block B (drift, strong) — identical across arms (divergence begins at C) ----
    before = model.n_segments[:]
    plc.train_pr08(model, Bx, By, lr, "B", ledger, seed=seed)
    tr["B"] = [model.n_segments[s] - before[s] for s in range(E)]
    rec["bpc_A_postB"] = fp.eval_bpc_fusion(model, Aex, Aey)      # descriptive
    rec["bpc_B_postB"] = fp.eval_bpc_fusion(model, Bex, Bey)
    rec["bpc_Cret_postB"] = fp.eval_bpc_fusion(model, Crx, Cry)   # descriptive

    # ---- block C (drift, mild + RETURNING domain; the ONLY treated phase) ----
    a_expert = max((s for s in range(E) if model.committed[s]), key=lambda s: tr["A"][s])
    rec["a_expert"] = int(a_expert)
    boundary = {"segments_total": 0, "to_A_expert": 0, "a_expert": int(a_expert)}
    before = model.n_segments[:]
    plc.train_pr08(model, Cx, Cy, lr, "C", ledger, seed=seed, boundary=boundary,
                   backbone_lr=trunk_lr_c)
    tr["C"] = [model.n_segments[s] - before[s] for s in range(E)]
    if trunk_lr_c is not None:
        rec["backbone_lr_c"] = {
            "lr": float(trunk_lr_c), "scale": TRUNK_LR_SCALE_C, "blocks": ["C"],
            "tissue_lr": lr, "floors": "live (PR-09 settled that axis)",
            "note": ("backbone AdamW alone scaled on block C via the guarded train_pr08 "
                     "backbone_lr kwarg; tissue per-expert optimizers keep the frozen lr"),
        }

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
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def _fp(payload: dict) -> str:
    from seq.gpu_harness import config_fingerprint
    return config_fingerprint({"registry": REGISTRY_ID, **payload})


def _pr08_powered_path() -> str:
    return os.path.join(_results_root(), PR08_LEDDIR, POWERED_BASENAME)


def run_canary(res, seeds):
    """Doc §2 fail-loud canary (POWERED MODE ONLY): every science field of OFF.s{seed} must
    equal PR-08's claim.PRIM-LM.s{seed} EXACTLY (comparator reused from PR-09's runner).
    A missing PR-08 ledger or any mismatch is a SystemExit ABORT before the treatment arm."""
    from seq.gpu_harness import load_results

    pr08_path = _pr08_powered_path()
    if not os.path.isfile(pr08_path):
        raise SystemExit(
            f"CANARY ABORT (PR-2026-09-03-10 §2): the PR-08 powered ledger is MISSING at "
            f"{pr08_path} — the OFF control cannot be verified against claim.PRIM-LM.s{{0..4}}. "
            "The registered protocol aborts before the treatment arm runs; run this where the "
            "PR-08 ledger exists.")
    pr08 = load_results(pr08_path)
    checked = []
    for seed in seeds:
        cellkey = f"claim.OFF.s{seed}"
        refkey = f"claim.PRIM-LM.s{seed}"
        off = res.get(cellkey)
        ref = pr08.get(refkey)
        if not isinstance(off, dict) or not isinstance(ref, dict):
            raise SystemExit(
                f"CANARY ABORT (PR-2026-09-03-10 §2): missing cell for comparison "
                f"({cellkey} or {refkey} absent) — refusing to proceed to the treatment arm.")
        mm = canary_mismatches(off, ref)
        if mm:
            raise SystemExit(
                f"CANARY ABORT (PR-2026-09-03-10 §2): OFF {cellkey} does NOT reproduce PR-08 "
                f"{refkey} bit-identically — mismatched science fields: {mm}. Investigate "
                f"before the treatment arm runs (this also re-proves cross-process "
                f"determinism end-to-end). Ledger: {pr08_path}")
        checked.append(cellkey)
    rec = {"ok": True, "pr08_ledger": pr08_path, "cells_checked": checked,
           "ignored_fields": sorted(CANARY_IGNORE),
           "note": ("every science field of OFF == PR-08 claim.PRIM-LM EXACTLY "
                    "(bpc_*, ledger dict, a_expert, fgt_A_full, b_degradation_B_postC)")}
    print(f"[pr10] CANARY PASS: {len(checked)} OFF cells bit-identical to PR-08 "
          f"claim.PRIM-LM ({pr08_path})", flush=True)
    return rec, pr08


def _smoke_checks(res):
    """Smoke-only fail-loud plumbing assertions: (i) the A phase is bit-identical between
    arms (equal bpc_A_preB); (ii) the B phase is bit-identical between arms (equal
    bpc_B_postB — divergence must begin at C only); (iii) the lever actually FIRES
    (bpc_B_postC differs between arms — the C-phase backbone lr had an effect); (iv) the
    treatment cell carries the backbone_lr_c audit with the exact frozen dose."""
    off = res["claim.OFF.s0"]
    tl = res["claim.TRUNK-LR-0.25C.s0"]
    if off["bpc_A_preB"] != tl["bpc_A_preB"]:
        raise AssertionError(
            f"A-PHASE IDENTITY BROKEN: OFF bpc_A_preB={off['bpc_A_preB']!r} != "
            f"TRUNK-LR-0.25C bpc_A_preB={tl['bpc_A_preB']!r} — divergence must begin at C.")
    if off["bpc_B_postB"] != tl["bpc_B_postB"]:
        raise AssertionError(
            f"B-PHASE IDENTITY BROKEN: OFF bpc_B_postB={off['bpc_B_postB']!r} != "
            f"TRUNK-LR-0.25C bpc_B_postB={tl['bpc_B_postB']!r} — divergence must begin at C.")
    if off["bpc_B_postC"] == tl["bpc_B_postC"]:
        raise AssertionError(
            "LEVER DID NOT FIRE: OFF and TRUNK-LR-0.25C bpc_B_postC are identical — the "
            "C-phase backbone_lr override had no effect; investigate before any campaign.")
    audit = tl.get("backbone_lr_c")
    assert isinstance(audit, dict) and audit["lr"] == BACKBONE_LR_C \
        and audit["scale"] == TRUNK_LR_SCALE_C and audit["blocks"] == ["C"], \
        "treatment cell must carry the exact frozen backbone_lr_c audit"
    return {"a_phase_bit_identity": True, "b_phase_bit_identity": True,
            "bpc_A_preB": off["bpc_A_preB"], "bpc_B_postB": off["bpc_B_postB"],
            "lever_fires": True, "backbone_lr_c_audit": audit,
            "note": ("equal A/B phases across arms = divergence begins at C only; differing "
                     "bpc_B_postC = the lever fires")}


def run(*, smoke: bool, results_path=None, force_smoke_path: bool = False,
        powered_cpu: bool = False):
    """Execute the PR-10 protocol: --smoke (CPU plumbing), --powered (the GPU claim
    campaign), or --powered-cpu (the registered CPU fallback: identical to --powered except
    the CUDA guard is skipped and the ledger meta records powered_cpu + the fallback note)."""
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
        # (--powered-cpu skips that guard by design: the registered CPU fallback, doc §5,
        # precedents PR-08 addendum 2026-09-08 #2 + the PR-09 execution)
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
        "doc": "docs/preregistry/2026-09-09-trunklr-routing-repair.md",
        "arms": list(ARMS), "claim_seeds": list(seeds),
        "seg": SEG, "batch_segs": BATCH_SEGS,
        "lr_rule": ("FROZEN 3e-3 for blocks A and B in BOTH arms (PR-08 lr_selection.fusion "
                    "inheritance); the treatment arm scales the BACKBONE lr to "
                    f"{BACKBONE_LR_C} (x{TRUNK_LR_SCALE_C}, frozen a priori) on block C only. "
                    "This runner has NO lr-selection leg."),
        "lr_inheritance": {"lr": LR_FROZEN, "backbone_lr_c": BACKBONE_LR_C,
                           "scale": TRUNK_LR_SCALE_C,
                           "source_registry": plc.REGISTRY_ID,
                           "source_ledger": f"results/{PR08_LEDDIR}/powered.json "
                                            "(lr_selection.fusion)"},
        "tissue": {"E_pool": E_POOL, "m_max": M_MAX, "freeze_min_seen": FREEZE_MIN_SEEN,
                   "z_novel": Z_NOVEL, "h_small": H_SMALL, "floor_ema": FLOOR_EMA,
                   "fresh_head_seed_formula": f"{FRESH_HEAD_SEED_BASE} + 17*slot + n_evictions",
                   "floors": "LIVE in both arms (PR-09 settled the floor axis: pinning does "
                             "not move boundary routing)",
                   "trunk_lr_lever": ("the PR-10 lever (guarded, default-OFF): the block-C "
                                      "train_pr08 call passes backbone_lr=7.5e-4; the backbone "
                                      "AdamW alone is scaled, tissue optimizers keep 3e-3"),
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
        "canary": ("doc §2 (powered only): OFF.s{seed} must equal PR-08 claim.PRIM-LM.s{seed} "
                   "EXACTLY on every science field; mismatch or missing PR-08 ledger = ABORT "
                   "before the treatment arm. A/B-phase bit-identity across arms + the "
                   "lever-fires check are asserted in smoke."),
    }
    if powered_cpu:
        res["meta"]["powered_cpu"] = True
        res["meta"]["compute_fallback_note"] = POWERED_CPU_FALLBACK_NOTE
    _save(res, path)
    print(f"[pr10] corpora: A={Ax.shape[0]} B={Bx.shape[0]} C={Cx.shape[0]} segs; "
          f"eval A={Aex.shape[0]} B={Bex.shape[0]} Cret={Crx.shape[0]}; vocab={V}; "
          f"smoke={smoke}; lr={LR_FROZEN} (frozen, inherited); backbone_lr_C={BACKBONE_LR_C} "
          f"(treatment, C only); results={path}", flush=True)

    # ---------------- the 2 arms x seeds (OFF first; the canary runs between arms) -------------
    n_cells_total = len(ARMS) * len(seeds)
    first_cell_s = None
    pr08 = None
    for arm in ARMS:
        trunk_lr_c = BACKBONE_LR_C if arm == "TRUNK-LR-0.25C" else None
        for seed in seeds:
            cellkey = f"claim.{arm}.s{seed}"
            cfgsig = _fp({"leg": "claim", "arm": arm, "seed": seed, "lr": LR_FROZEN,
                          "backbone_lr_c": trunk_lr_c,
                          "smoke": bool(smoke), "vocab": V, "smoke_segs": smoke_segs,
                          "seg": SEG, "batch_segs": BATCH_SEGS,
                          "slices": res["meta"]["pinned_slices"],
                          "stream_lengths": res["meta"]["stream_lengths"],
                          "tissue": dict(res["meta"]["tissue"]),       # constants only — the
                                                                       # C-lr is its own key
                          "bars": res["meta"]["bars"]})
            prior = res.get(cellkey)
            if isinstance(prior, dict) and prior.get("cfgsig") == cfgsig and prior.get("complete"):
                print(f"[pr10] {arm} seed {seed}: resumed from ledger "
                      f"(bpc_A_postC={prior['bpc_A_postC']:.3f})", flush=True)
                continue
            if isinstance(prior, dict):
                raise SystemExit(f"cell {cellkey} exists at a foreign config fingerprint "
                                 f"({prior.get('cfgsig')} != {cfgsig}) — refusing to resume")
            t0 = time.time()
            rec = run_cell(V, seed, data, LR_FROZEN, trunk_lr_c=trunk_lr_c)
            rec.update({"arm": arm, "seed": seed, "lr": LR_FROZEN,
                        "backbone_lr_c_applied": trunk_lr_c, "cellkey": cellkey,
                        "cfgsig": cfgsig, "complete": True,
                        "wall_s": round(time.time() - t0, 1)})
            res[cellkey] = rec
            _save(res, path)                    # crash-safe after every cell
            if first_cell_s is None:
                first_cell_s = rec["wall_s"]
                proj = plc.budget_projection(first_cell_s, n_cells_total)
                res.setdefault("meta", {})["budget_projection"] = proj
                _save(res, path)
                print(f"[pr10] budget projection from first cell: {proj['projected_min']} min "
                      f"for {n_cells_total} cells"
                      f"{'  *** ' + proj['note'] if proj['warn'] else ''}", flush=True)
            tl_note = ""
            if trunk_lr_c is not None:
                tl_note = f" backbone_lr_C={trunk_lr_c}"
            print(f"[pr10] {arm} seed {seed}: A_preB={rec['bpc_A_preB']:.3f} "
                  f"A_postC={rec['bpc_A_postC']:.3f} B_postB={rec['bpc_B_postB']:.3f} "
                  f"B_postC={rec['bpc_B_postC']:.3f} Cret_postC={rec['bpc_Cret_postC']:.3f} "
                  f"frac_C(1-20)to_A={rec['ledger']['boundary_window']['frac_to_A_expert']}"
                  f"{tl_note} wall={rec['wall_s']}s", flush=True)
        if not smoke and arm == "OFF":
            # ---------------- doc §2 fail-loud canary: BEFORE the treatment arm ---------------
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
    raw_archive = archive_run(res, label=f"trunklr-{REGISTRY_ID}")
    res.setdefault("meta", {})["raw_archive"] = raw_archive

    # ---------------- the frozen §3/§4 verdict (claim mode) / plumbing report (smoke) ----------
    report = {"registry": REGISTRY_ID, "smoke": bool(smoke), "arms": list(ARMS),
              "claim_seeds": list(seeds), "lr": LR_FROZEN,
              "backbone_lr_c": BACKBONE_LR_C, "scale": TRUNK_LR_SCALE_C,
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
        tl = [res[f"claim.TRUNK-LR-0.25C.s{s}"] for s in seeds]
        frozen_ck = [pr08[f"claim.FROZEN-CHECKPOINT.s{s}"] for s in seeds]   # REUSED from PR-08
        verdict = claim_verdict(off, tl, frozen_ck, alpha=ALPHA)
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
        if c.get("backbone_lr_c") and isinstance(c["backbone_lr_c"], dict):
            a = c["backbone_lr_c"]
            print(f"      backbone_lr_c: lr={a['lr']} scale={a['scale']} blocks={a['blocks']}",
                  flush=True)
    if smoke:
        sc = report.get("smoke_checks") or {}
        print(f"    [SMOKE CHECK] A/B-phase bit-identity: A={sc.get('a_phase_bit_identity')} "
              f"B={sc.get('b_phase_bit_identity')} (bpc_A_preB={sc.get('bpc_A_preB'):.6f}, "
              f"bpc_B_postB={sc.get('bpc_B_postB'):.6f})", flush=True)
        print(f"    [SMOKE CHECK] lever fires (B_postC differs across arms): "
              f"{sc.get('lever_fires')}", flush=True)
        print("  [SMOKE] numbers are plumbing-only and MEANINGLESS — do NOT cite.", flush=True)
    else:
        v = report["verdict"]
        p1 = v["P1"]
        print(f"    P1: tl_frac_mean={p1['frac_mean']:.3f} (bar >= {p1['frac_bar']}) "
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
              f"TRUNK-LR-0.25C={v['P1']['frac_per_seed']}", flush=True)
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
        prog="trunklr_claim",
        description="PR-2026-09-03-10 claim runner (trunk-lr scheduling on the returning "
                    "block). --smoke = tiny CPU plumbing run; --powered = the frozen claim "
                    "campaign (requires CUDA); --powered-cpu = the registered CPU fallback "
                    "(protocol-identical claim campaign on an explicitly authorized CPU box). "
                    "An unknown flag is rejected without launching anything.")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="tiny plumbing-only run (CPU, minutes)")
    mode.add_argument("--powered", action="store_true",
                      help="the claim campaign (refuses to start without a CUDA device)")
    mode.add_argument("--powered-cpu", action="store_true",
                      help="the claim campaign via the registered CPU fallback (identical "
                           "protocol and ledger; runs WITHOUT a CUDA device; precedents PR-08 "
                           "addendum 2026-09-08 #2 + the PR-09 execution)")
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
