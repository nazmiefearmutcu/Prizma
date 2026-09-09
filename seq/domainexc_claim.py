"""
PR-2026-09-03-12 CLAIM RUNNER — domain-exclusive C-routing (the registered successor of
PR-11's B3-clause-(b) negative). Executes the FROZEN pre-registration
`docs/preregistry/2026-09-09-domain-exclusive-c.md` VERBATIM. One script, three modes
(the prizma_lm_claim pattern, mirrored):

  python seq/domainexc_claim.py --smoke        # tiny CPU plumbing run -> the SMOKE ledger
  python seq/domainexc_claim.py --powered      # the claim campaign (refuses without CUDA)
  python seq/domainexc_claim.py --powered-cpu  # the registered CPU fallback

WHAT IS FROZEN (pre-reg §2 — inherited from PR-08/PR-11's registered text, reused BY IMPORT):
stream, slices, corpora, tissue constants, routing semantics (veto + Policy A + forced
placement + the PR-10 trunk-lr x0.25 C lever), seeds 0-4, LR rules. THE ONE LEVER: during
block C, protected slots (committed at C start MINUS the a_expert, pinned on the model as
`domain_protect`) receive NO training — their would-be segments REDIRECT to the A-expert
(counted in the ledger; the boundary window counts the post-redirect trained fraction);
protected slots are exempt from Policy A eviction during C. Attribute absent => the
byte-identical PR-08/PR-11 paths.

Arm (1, treated):
  PRIM-DE — PR-11's PRIM-LM cell + the domain-exclusion lever. BEFORE the C phase the
            runner pins model.domain_protect = committed slots - {a_expert} and records the
            audit. A/B phases are bit-identical to PR-11's PRIM cells (the lever activates
            only at C) — asserted per cell, fail-loud (doc §2 canary).

REUSED BASELINES (doc §2, disclosed; read-only from
results/prizma_lm_PR-2026-09-03-11/powered.json): claim.PRIM-LM (the argmin-C control),
claim.FORCED-RECRUIT (the clause-(b) baseline), claim.FROZEN-CHECKPOINT (the G2 floor;
bit-identical to PR-08's by the PR-11 canary).

BARS (doc §3, exact; n=5 seeds 0-4):
  P1 (primary, the clause-(b) flip): mean(PRIM-DE bpc_B_postC) >= mean(FORCED-11
      bpc_B_postC) - 0.10 — EXACT MEANS (the PR-08 B3b form). PR-11's PRIM-LM B mean
      (3.6120) is recorded as the untreated control.
  P2 (the repair must hold): mean frac_to_A_expert >= 0.5 (PR-11 measured 0.575 untreated).
  G1 (retention guard): one-sample, CI upper <= 0.05 AND Holm p < alpha; straddle ->
      INCONCLUSIVE.
  G2 (adaptation guard): Welch advantage vs FROZEN-CHECKPOINT-11 >= 0.10, CI lower >= 0.10
      AND Holm p < alpha; straddle -> INCONCLUSIVE.
  Holm family = [G1, G2]. Reported, never gated: B4 per arm; churn incl. the
  domain_protect_redirects counter; per-seed fracs.

Overall (doc §4, pre-committed): CLAIMED iff P1 + P2 + G1 + G2 all PASS; NEGATIVE per the
doc's branches when P1 or P2 fails; NEGATIVE-GUARD when only guards fail; any guard
INCONCLUSIVE -> INCONCLUSIVE (n=10, seeds 5-9, PRE-AUTHORIZED).

LEDGER SEPARATION + RETENTION (plc pattern, mirrored): smoke -> smoke.json, powered ->
powered.json under $PRIZMA_RESULTS/domainexc_PR-2026-09-03-12; archive BEFORE any verdict;
per-cell fingerprints include the lever; foreign-fingerprint resume = hard refusal.
"""
from __future__ import annotations

import argparse
import os
import sys

try:
    from .stats import holm_correction
except ImportError:                                   # run as a bare script: bootstrap sys.path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from seq.stats import holm_correction

import seq.prizma_lm_claim as plc                     # PR-08 protocol pieces, reused VERBATIM
import seq.floorfreeze_claim as ffc                   # canary comparator, reused


# ================================================================== frozen protocol constants ====
REGISTRY_ID = "PR-2026-09-03-12"
LEDDIR = "domainexc_PR-2026-09-03-12"
SMOKE_BASENAME = "smoke.json"
POWERED_BASENAME = "powered.json"

ARMS = ("PRIM-DE",)                             # single treated arm; baselines are REUSED
CLAIM_SEEDS = plc.CLAIM_SEEDS                   # (0, 1, 2, 3, 4) — inherited, never substituted

LR_FROZEN = ffc.LR_FROZEN                       # 3e-3 — blocks A and B (PR-08 inheritance)
TRUNK_LR_SCALE_C = 0.25                         # PR-10's registered dose — carried UNCHANGED
BACKBONE_LR_C = LR_FROZEN * TRUNK_LR_SCALE_C    # 7.5e-4 — block C backbone (PR-11 carried)
H_SMALL = plc.H_SMALL                           # 64 — per-expert predictor width (audit use)
E_POOL = plc.E_POOL                             # 4
M_MAX = plc.M_MAX                               # 4
FREEZE_MIN_SEEN = plc.FREEZE_MIN_SEEN           # 300
Z_NOVEL = plc.Z_NOVEL                           # 5.0
FRESH_HEAD_SEED_BASE = plc.FRESH_HEAD_SEED_BASE # pinned eviction re-init formula base
B3_WINDOW_BATCHES = plc.B3_WINDOW_BATCHES       # 20 — the boundary window (P2's quantity)
P1_MARGIN = 0.10                        # clause-(b) allowance: primDE_B >= FORCED_B - 0.10
P2_FRAC_BAR = 0.5                       # clause (a) must keep holding
G1_MARGIN = 0.05
G2_MARGIN = 0.10
ALPHA = plc.ALPHA
HOLM_FAMILY = ("G1", "G2")

PR11_LEDDIR = "prizma_lm_PR-2026-09-03-11"      # the reused-baselines source (read-only)
POWERED_CPU_FALLBACK_NOTE = (
    "POWERED VIA THE REGISTERED CPU FALLBACK (PR-12 doc §5; precedent = PR-08 addendum "
    "2026-09-08 #2 + the PR-09/PR-10/PR-11 powered-cpu executions): protocol-identical to "
    "the A100 tier; --powered (CUDA-refusing) remains available for the GPU session.")


# ==================================================================== paths + BAR-0 refusal ======
def _results_root() -> str:
    root = os.environ.get("PRIZMA_RESULTS", os.path.join(os.path.dirname(__file__), "..", "results"))
    return os.path.abspath(root)


def _default_results_path(smoke: bool) -> str:
    return os.path.join(_results_root(), LEDDIR, SMOKE_BASENAME if smoke else POWERED_BASENAME)


def _pr11_powered_path() -> str:
    return os.path.join(_results_root(), PR11_LEDDIR, POWERED_BASENAME)


def resolve_results_path(explicit=None, *, smoke: bool = False, force_smoke_path: bool = False) -> str:
    """Resolve the results path and enforce the smoke/powered file separation (the plc
    refusal pattern, pointed at the PR-12 led dir)."""
    path = explicit if explicit else _default_results_path(smoke)
    if smoke and not force_smoke_path:
        powered = _default_results_path(smoke=False)
        if os.path.normcase(os.path.abspath(path)) == os.path.normcase(os.path.abspath(powered)):
            raise SystemExit(
                f"refusing: a --smoke run was pointed at the powered ledger ({powered}). Smoke and "
                f"campaign results use separate files by default (PR-2026-09-03-12 §5 ledger "
                f"separation). Pass --out <path> to write the smoke elsewhere, or "
                f"--force-smoke-path to override deliberately.")
    return path


def require_cuda(has_cuda: bool) -> None:
    """Pure guard for the --powered mode (unit-testable without a CUDA box). --powered-cpu
    (the registered CPU fallback) skips this guard by design."""
    if not has_cuda:
        raise SystemExit(
            "refusing: --powered executes the PR-2026-09-03-12 claim campaign and requires a "
            "CUDA device. No CUDA device is visible here. Use --smoke for the CPU plumbing "
            "run, or --powered-cpu for the registered CPU fallback.")


needs_cuda = plc.needs_cuda                       # TRUE alias (pinned by tests)
pin_slices = plc.pin_slices                       # PR-08's §2 stream inherited verbatim

canary_mismatches = ffc.canary_mismatches         # reused comparator (same object, pinned)


# ==================================================================== PURE: verdict (§3+§4) ======
def claim_verdict(prim_de, pr11, *, alpha=ALPHA):
    """The frozen §3/§4 verdict as a PURE function. prim_de = the PRIM-DE records from THIS
    ledger; pr11 = dict of REUSED PR-11 records keyed 'claim.PRIM-LM', 'claim.FORCED-RECRUIT',
    'claim.FROZEN-CHECKPOINT' (lists over seeds; only bpc_* keys are read).

    P1: exact means — mean(primDE bpc_B_postC) >= mean(FORCED-11 bpc_B_postC) - 0.10.
    P2: exact means — mean frac_to_A_expert >= 0.5.
    G1: one-sample retention diffs, CI upper <= 0.05 AND Holm.
    G2: Welch advantage vs FROZEN-CHECKPOINT-11 >= 0.10, CI lower >= 0.10 AND Holm.
    Holm = [G1, G2]. Overall: guards INCONCLUSIVE -> INCONCLUSIVE; P1+P2+G1+G2 PASS ->
    CLAIMED; P1/P2 FAIL -> NEGATIVE (branch echoed per which); guards FAIL -> NEGATIVE-GUARD.
    """
    de_b = [r["bpc_B_postC"] for r in prim_de]
    forced_b = [r["bpc_B_postC"] for r in pr11["FORCED-RECRUIT"]]
    prim11_b = [r["bpc_B_postC"] for r in pr11["PRIM-LM"]]
    de_b_mean = sum(de_b) / len(de_b)
    forced_b_mean = sum(forced_b) / len(forced_b)
    prim11_b_mean = sum(prim11_b) / len(prim11_b)
    p1_ok = de_b_mean >= forced_b_mean - P1_MARGIN
    p1 = {"de_b_per_seed": de_b, "de_b_mean": de_b_mean,
          "forced_b_mean": forced_b_mean, "margin": P1_MARGIN,
          "status": "PASS" if p1_ok else "FAIL",
          "prim11_b_mean": prim11_b_mean,
          "untreated_gap": prim11_b_mean - forced_b_mean,
          "treated_gap": de_b_mean - forced_b_mean,
          "control_note": ("untreated control = PR-11 claim.PRIM-LM (argmin-C), B mean "
                           "3.6120 at n=5")}

    de_fracs = [float(r["ledger"]["boundary_window"]["frac_to_A_expert"]) for r in prim_de]
    de_frac_mean = sum(de_fracs) / len(de_fracs)
    p2 = {"frac_per_seed": de_fracs, "frac_mean": de_frac_mean, "frac_bar": P2_FRAC_BAR,
          "status": "PASS" if de_frac_mean >= P2_FRAC_BAR else "FAIL",
          "pr11_untreated_frac_mean": 0.575}

    g1 = plc._onesample_margin([r["bpc_A_postC"] - r["bpc_A_preB"] for r in prim_de], G1_MARGIN)
    g2 = plc._welch_margin([r["bpc_Cret_postC"] for r in prim_de],
                           [r["bpc_Cret_postC"] for r in pr11["FROZEN-CHECKPOINT"]], G2_MARGIN)
    g2["baseline_note"] = ("baseline = PR-11 claim.FROZEN-CHECKPOINT cells REUSED "
                           "(bit-identical to PR-08's by the PR-11 canary)")

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
                                    "vetoed_novel_segments": led["vetoed_novel_segments"],
                                    "domain_protect_redirects":
                                        led.get("domain_protect_redirects", 0)}
        return out

    b4 = {"PRIM-DE": _b4(prim_de), "PRIM-LM (PR-11 untreated)": _b4(pr11["PRIM-LM"]),
          "FORCED-RECRUIT (PR-11)": _b4(pr11["FORCED-RECRUIT"])}
    churn = {"PRIM-DE": _churn(prim_de)}

    branches = []
    guard_statuses = [g1["status"], g2["status"]]
    if "INCONCLUSIVE" in guard_statuses:
        outcome = "INCONCLUSIVE"
        branches.append("§4 (guard CI straddles): INCONCLUSIVE; n=10 (seeds 5-9) "
                        "PRE-AUTHORIZED (03:55 rule always wins). Never PASS (PR-03 mirror).")
    elif p1_ok and de_frac_mean >= P2_FRAC_BAR and all(s == "PASS" for s in guard_statuses):
        outcome = "CLAIMED"
    elif not p1_ok:
        outcome = "NEGATIVE"
        branches.append("§4 (P1 FAIL): the contamination hypothesis is insufficient; the next "
                        "lead is eval-routing (where B segments land at eval time), own prereg.")
    elif de_frac_mean < P2_FRAC_BAR:
        outcome = "NEGATIVE"
        branches.append("§4 (P2 FAIL): exclusion breaks the boundary routing it was meant to "
                        "protect; investigate the redirect's surprise statistics before any "
                        "new registration.")
    else:
        outcome = "NEGATIVE-GUARD"
        broke = [name for name, st in zip(("G1", "G2"), guard_statuses) if st == "FAIL"]
        branches.append(f"§4 (guard FAIL): {' and '.join(broke)} broke — record and stop.")

    if outcome == "CLAIMED":
        verdict_text = ("CLAIMED — domain-exclusion closes the clause-(b) value gap (P1) while "
                        "the repaired boundary routing holds (P2) and retention/adaptation "
                        "stay intact (G1/G2): the argmin C-leak was the remaining defect. S2c "
                        "= the full 5-arm flagship with the exclusion lever (own prereg).")
    elif outcome == "NEGATIVE-GUARD":
        verdict_text = ("NEGATIVE (with lead) — P1/P2 hold but a guard failed; see branches.")
    elif outcome == "NEGATIVE":
        verdict_text = ("NEGATIVE — " + ("P1 FAIL: exclusion did not close the value gap; the "
                        "contamination hypothesis is insufficient." if not p1_ok else
                        "P2 FAIL: exclusion broke the boundary routing.") + " See branches.")
    else:
        verdict_text = ("INCONCLUSIVE — a guard CI straddles its margin (never PASS); the "
                        "pre-authorized n=10 extension applies (see 'branches').")

    return {"outcome": outcome, "verdict": verdict_text, "branches": branches,
            "P1": p1, "P2": p2, "G1": g1, "G2": g2, "B4": b4, "churn": churn,
            "holm_family": [{"bar": "G1", "p_raw": g1["p_raw"], "p_holm": g1["p_holm"],
                             "status": g1["status"]},
                            {"bar": "G2", "p_raw": g2["p_raw"], "p_holm": g2["p_holm"],
                             "status": g2["status"]}],
            "alpha": alpha}


# ================================================================================= runner ========
def run_cell(vocab_size, seed, data, lr):
    """PR-12 cell — plc.run_routed's structure VERBATIM with the domain-exclusion wiring:
    right BEFORE the C phase, model.domain_protect = committed slots - {a_expert} (the
    guarded branches in route_pr08 do the rest); the PR-10/PR-11 trunk-lr C dose
    (backbone_lr=7.5e-4) is carried UNCHANGED. A/B phases are bit-identical to PR-11's
    PRIM cells (the lever activates only at C) — the caller asserts this per cell."""
    import time

    import torch
    from seq import fusion_probe as fp

    t0 = time.time()
    Ax, Ay, Bx, By, Cx, Cy, Aex, Aey, Bex, Bey, Crx, Cry = data
    E = E_POOL
    model = fp.build_model(vocab_size, seed, E)
    ledger = plc._fresh_ledger()
    tr = {"A": [0] * E, "B": [0] * E, "C": [0] * E}
    rec = {"config": "PRIM-DE", "E": E, "seed": seed, "lr": lr}

    # ---- block A (identical to PR-11 PRIM: no lever) ----
    plc.train_pr08(model, Ax, Ay, lr, "A", ledger, seed=seed)
    tr["A"] = model.n_segments[:]
    rec["bpc_A_preB"] = fp.eval_bpc_fusion(model, Aex, Aey)

    # ---- block B (identical to PR-11 PRIM: no lever yet) ----
    before = model.n_segments[:]
    plc.train_pr08(model, Bx, By, lr, "B", ledger, seed=seed)
    tr["B"] = [model.n_segments[s] - before[s] for s in range(E)]
    rec["bpc_A_postB"] = fp.eval_bpc_fusion(model, Aex, Aey)      # descriptive
    rec["bpc_B_postB"] = fp.eval_bpc_fusion(model, Bex, Bey)
    rec["bpc_Cret_postB"] = fp.eval_bpc_fusion(model, Crx, Cry)   # descriptive

    # ---- block C (drift, mild + RETURNING domain; the treated phase) ----
    a_expert = max((s for s in range(E) if model.committed[s]), key=lambda s: tr["A"][s])
    rec["a_expert"] = int(a_expert)
    model.domain_protect = {s for s in range(E) if model.committed[s]} - {int(a_expert)}
    rec["domain_protect"] = {
        "slots": sorted(model.domain_protect), "a_expert": int(a_expert),
        "note": ("committed-at-C-start slots other than the A-expert receive NO C training; "
                 "their would-be segments redirect to the A-expert and they are exempt from "
                 "Policy A eviction during C (guarded branches in seq/prizma_lm_claim.py "
                 "route_pr08); trunk-lr C dose 7.5e-4 carried from PR-10/PR-11"),
    }
    boundary = {"segments_total": 0, "to_A_expert": 0, "a_expert": int(a_expert)}
    before = model.n_segments[:]
    plc.train_pr08(model, Cx, Cy, lr, "C", ledger, seed=seed, boundary=boundary,
                   backbone_lr=BACKBONE_LR_C)
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
    rec["domain_protect"]["redirects"] = ledger.get("domain_protect_redirects", 0)
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def _fp(payload: dict) -> str:
    from seq.gpu_harness import config_fingerprint
    return config_fingerprint({"registry": REGISTRY_ID, **payload})


def run_canary(rec, seed, pr11):
    """Doc §2 per-cell fail-loud canary: bpc_A_preB AND bpc_B_postB must equal PR-11's
    claim.PRIM-LM.s{seed} EXACTLY (the lever activates only at C)."""
    ref = pr11.get(f"claim.PRIM-LM.s{seed}")
    if not isinstance(ref, dict):
        raise SystemExit(
            f"CANARY ABORT (PR-2026-09-03-12 §2): PR-11 ledger lacks claim.PRIM-LM.s{seed}.")
    for k in ("bpc_A_preB", "bpc_B_postB"):
        if rec[k] != ref[k]:
            raise SystemExit(
                f"CANARY ABORT (PR-2026-09-03-12 §2): PRIM-DE s{seed} {k}={rec[k]!r} != "
                f"PR-11 PRIM {ref[k]!r} — the A/B phases must be bit-identical (the lever "
                f"activates only at C).")
    return {"ok": True, "seed": seed, "checked": ["bpc_A_preB", "bpc_B_postB"]}


def run(*, smoke: bool, results_path=None, force_smoke_path: bool = False,
        powered_cpu: bool = False):
    """Execute the PR-12 protocol: --smoke (CPU plumbing), --powered (the GPU claim
    campaign), or --powered-cpu (the registered CPU fallback)."""
    path = resolve_results_path(results_path, smoke=smoke, force_smoke_path=force_smoke_path)

    import time

    import torch

    from seq import blockdrift_claim as claim          # PR-07' protocol pieces, imported VERBATIM
    from seq import fusion_probe as fp                 # the fused column, imported VERBATIM
    from seq.gpu_harness import load_results, _save
    from seq.recall_gate import archive_run

    torch.set_num_threads(int(os.environ.get("BENCH_THREADS", "8")))

    if smoke:
        smoke_segs = 64
        seeds = (0,)
    else:
        if needs_cuda("powered-cpu" if powered_cpu else "powered"):
            require_cuda(torch.cuda.is_available())
        smoke_segs = None
        seeds = CLAIM_SEEDS

    # ---------------- data: the 3-block stream + the pinned eval slices (inherited) ----------
    slices = pin_slices()
    A_all, B_all = claim.fetch_corpora()
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

    # ---------------- the reused PR-11 baselines (fail-loud if missing) -----------------------
    pr11_path = _pr11_powered_path()
    if not os.path.isfile(pr11_path):
        raise SystemExit(
            f"refusing: the PR-11 powered ledger is MISSING at {pr11_path} — PR-12's baselines "
            "(PRIM argmin control / FORCED clause-b baseline / FROZEN-CHECKPOINT G2 floor) and "
            "the per-cell A/B canary all come from it (doc §2). Run this where the PR-11 "
            "ledger exists.")
    pr11 = load_results(pr11_path)

    # ---------------- ledger + meta (NO lr-selection leg: LR frozen by inheritance) -----------
    res = load_results(path)
    bb_params = sum(p.numel() for p in fp._PlainWrap(V, 0).lm.parameters())
    per_expert = sum(p.numel() for p in fp.PCExpertHead(64, H_SMALL, V).parameters())
    res["meta"] = {
        "registry": REGISTRY_ID, "smoke": bool(smoke),
        "lane": "CLAIM (frozen pre-registration)" if not smoke else "SMOKE (plumbing only)",
        "doc": "docs/preregistry/2026-09-09-domain-exclusive-c.md",
        "arms": list(ARMS), "claim_seeds": list(seeds),
        "seg": plc.SEG, "batch_segs": plc.BATCH_SEGS,
        "lr_rule": ("FROZEN by inheritance: 3e-3 A/B both arms; backbone C dose 7.5e-4 "
                    "(PR-10/PR-11 carried). NO lr-selection leg."),
        "lever": ("domain-exclusion: protected slots (committed at C start minus the a_expert) "
                  "receive no C training — redirected to the a_expert; exempt from Policy A "
                  "eviction during C; guarded default-off branches in route_pr08"),
        "reused_baselines": {"ledger": pr11_path,
                             "cells": ["claim.PRIM-LM", "claim.FORCED-RECRUIT",
                                       "claim.FROZEN-CHECKPOINT"],
                             "note": ("same seeds, same lever, same frozen protocol; "
                                      "read-only; disclosed in doc §2")},
        "pinned_slices": {k: list(v) for k, v in slices.items()},
        "stream_lengths": {"A_train": len(A_train), "B_train": len(B_train),
                           "C_train": len(C_train), "A_eval": len(A_eval),
                           "B_eval": len(B_eval), "C_retention": len(C_ret)},
        "vocab": V, "threads": torch.get_num_threads(),
        "c_range_repair": plc.C_RANGE_REPAIR_NOTE,
        "c_retention_prose_note": plc.C_RET_PROSE_NOTE,
        "bars": {"P1_margin": P1_MARGIN, "P2_frac_bar": P2_FRAC_BAR, "G1_margin": G1_MARGIN,
                 "G2_margin": G2_MARGIN, "alpha": ALPHA, "holm_family": list(HOLM_FAMILY),
                 "p1_p2_form": "exact means, no CI (the PR-08 B3 form)",
                 "t_isf_convention": "UPPER-TAIL p in (0, 0.5] (the PR-03 lesson)"},
        "boundary_note": ("A-expert := the committed slot with max train_A at C start; frac = "
                          "TRAINING-ledger fraction of C segments in the first 20 C-batches "
                          "assigned to it, counting POST-REDIRECT routing"),
        "canary": ("doc §2, per cell: bpc_A_preB and bpc_B_postB must equal PR-11's "
                   "claim.PRIM-LM.s{seed} EXACTLY — the lever activates only at C."),
    }
    if powered_cpu:
        res["meta"]["powered_cpu"] = True
        res["meta"]["compute_fallback_note"] = POWERED_CPU_FALLBACK_NOTE
    _save(res, path)
    print(f"[pr12] corpora: A={Ax.shape[0]} B={Bx.shape[0]} C={Cx.shape[0]} segs; "
          f"eval A={Aex.shape[0]} B={Bex.shape[0]} Cret={Crx.shape[0]}; vocab={V}; "
          f"smoke={smoke}; lr={LR_FROZEN} + backbone_C={BACKBONE_LR_C} (carried); "
          f"baselines={pr11_path}", flush=True)

    # ---------------- the treated arm x seeds (per-cell canary) -------------------------------
    n_cells_total = len(ARMS) * len(seeds)
    first_cell_s = None
    canaries = []
    for arm in ARMS:
        for seed in seeds:
            cellkey = f"claim.{arm}.s{seed}"
            cfgsig = _fp({"leg": "claim", "arm": arm, "seed": seed, "lr": LR_FROZEN,
                          "backbone_lr_c": BACKBONE_LR_C, "domain_exclusion": True,
                          "smoke": bool(smoke), "vocab": V, "smoke_segs": smoke_segs,
                          "seg": plc.SEG, "batch_segs": plc.BATCH_SEGS,
                          "slices": res["meta"]["pinned_slices"],
                          "stream_lengths": res["meta"]["stream_lengths"],
                          "bars": res["meta"]["bars"]})
            prior = res.get(cellkey)
            if isinstance(prior, dict) and prior.get("cfgsig") == cfgsig and prior.get("complete"):
                print(f"[pr12] {arm} seed {seed}: resumed from ledger "
                      f"(bpc_B_postC={prior['bpc_B_postC']:.3f})", flush=True)
                continue
            if isinstance(prior, dict):
                raise SystemExit(f"cell {cellkey} exists at a foreign config fingerprint "
                                 f"({prior.get('cfgsig')} != {cfgsig}) — refusing to resume")
            t0 = time.time()
            rec = run_cell(V, seed, data, LR_FROZEN)
            if not smoke:
                # claim-mode-only (doc addendum 2026-09-09 #1): the PR-11 ledger is
                # POWERED-scale; smoke cells live at the 64-seg scale, so the A/B
                # bit-identity comparison is meaningful only in claim mode.
                canaries.append(run_canary(rec, seed, pr11))
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
                print(f"[pr12] budget projection from first cell: {proj['projected_min']} min "
                      f"for {n_cells_total} cells"
                      f"{'  *** ' + proj['note'] if proj['warn'] else ''}", flush=True)
            print(f"[pr12] {arm} seed {seed}: A_preB={rec['bpc_A_preB']:.3f} "
                  f"A_postC={rec['bpc_A_postC']:.3f} B_postB={rec['bpc_B_postB']:.3f} "
                  f"B_postC={rec['bpc_B_postC']:.3f} Cret_postC={rec['bpc_Cret_postC']:.3f} "
                  f"frac_C(1-20)to_A={rec['ledger']['boundary_window']['frac_to_A_expert']} "
                  f"protect={rec['domain_protect']['slots']} "
                  f"redirects={rec['domain_protect']['redirects']} wall={rec['wall_s']}s",
                  flush=True)

    # ---------------- RETENTION: archive BEFORE any verdict -----------------------------------
    raw_archive = archive_run(res, label=f"domainexc-{REGISTRY_ID}")
    res.setdefault("meta", {})["raw_archive"] = raw_archive

    # ---------------- the frozen §3/§4 verdict (claim mode) / plumbing report (smoke) ---------
    report = {"registry": REGISTRY_ID, "smoke": bool(smoke), "arms": list(ARMS),
              "claim_seeds": list(seeds), "lr": LR_FROZEN, "backbone_lr_c": BACKBONE_LR_C,
              "reused_baselines": res["meta"]["reused_baselines"],
              "pinned_slices": res["meta"]["pinned_slices"],
              "raw_archive": raw_archive,
              "cells": {k: res[k] for k in sorted(res) if k.startswith("claim.")}}
    if powered_cpu:
        report["powered_cpu"] = True
    if not smoke:
        prim_de = [res[f"claim.PRIM-DE.s{s}"] for s in seeds]
        pr11_b = {arm: [pr11[f"claim.{arm}.s{s}"] for s in seeds]
                  for arm in ("PRIM-LM", "FORCED-RECRUIT", "FROZEN-CHECKPOINT")}
        verdict = claim_verdict(prim_de, pr11_b, alpha=ALPHA)
        report["verdict"] = verdict
        report["canaries"] = canaries
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
        print(f"    {key:<34} lr={c['lr']:.0e} A_preB={c['bpc_A_preB']:.3f} "
              f"A_postC={c['bpc_A_postC']:.3f} B_postB={c['bpc_B_postB']:.3f} "
              f"B_postC={c['bpc_B_postC']:.3f} Cret={c['bpc_Cret_postC']:.3f} "
              f"wall={c['wall_s']}s", flush=True)
        led = c.get("ledger")
        if led:
            bw = led["boundary_window"]
            print(f"      ledger: committed={led['n_committed']} a_expert={led['a_expert']} "
                  f"recruits={len(led['recruits'])} evictions={len(led['evictions'])} "
                  f"redirects={c['domain_protect']['redirects']} "
                  f"frac_C(1-20)to_A={bw['frac_to_A_expert']}", flush=True)
    if not smoke:
        v = report["verdict"]
        p1 = v["P1"]
        print(f"    P1: de_b_mean={p1['de_b_mean']:.4f} vs FORCED_B-0.10="
              f"{p1['forced_b_mean'] - p1['margin']:.4f} "
              f"[untreated PRIM_B={p1['prim11_b_mean']:.4f}] -> {p1['status']}", flush=True)
        p2 = v["P2"]
        print(f"    P2: frac_mean={p2['frac_mean']:.3f} (bar >= {p2['frac_bar']}) "
              f"[PR-11 untreated 0.575] -> {p2['status']}", flush=True)
        for bar in ("G1", "G2"):
            b = v[bar]
            center = b["mean"] if bar == "G1" else b["delta"]
            print(f"    {bar}: mean={center:.4f} CI=[{b['ci'][0]:.4f}, {b['ci'][1]:.4f}] "
                  f"p_raw={b['p_raw']:.4f} p_holm={b['p_holm']:.4f} -> {b['status']}", flush=True)
        for arm, d in v["B4"].items():
            print(f"    B4 {arm}: B-eval degradation post-C mean {d['mean']:+.4f} "
                  f"(reported, not gated)", flush=True)
        for arm, ch in v["churn"].items():
            print(f"    churn {arm}: {ch}", flush=True)
        print(f"  OUTCOME: {v['outcome']}", flush=True)
        print(f"  {v['verdict']}", flush=True)
        for br in v["branches"]:
            print(f"  branch: {br}", flush=True)
    else:
        print("  [SMOKE] numbers are plumbing-only and MEANINGLESS — do NOT cite.", flush=True)
    print(f"  ledger: {path}", flush=True)
    print(f"  raw archive: {report.get('raw_archive')}", flush=True)
    print("=" * 78, flush=True)


def _build_parser():
    """Argparse guard (plc pattern): ONLY --smoke / --powered / --powered-cpu / --out /
    --force-smoke-path; an unknown or typo'd flag exits non-zero BEFORE anything runs."""
    p = argparse.ArgumentParser(
        prog="domainexc_claim",
        description="PR-2026-09-03-12 claim runner (domain-exclusive C-routing). --smoke = "
                    "tiny CPU plumbing run; --powered = the frozen claim campaign (requires "
                    "CUDA); --powered-cpu = the registered CPU fallback. An unknown flag is "
                    "rejected without launching anything.")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="tiny plumbing-only run (CPU, minutes)")
    mode.add_argument("--powered", action="store_true",
                      help="the claim campaign (refuses to start without a CUDA device)")
    mode.add_argument("--powered-cpu", action="store_true",
                      help="the claim campaign via the registered CPU fallback (identical "
                           "protocol and ledger; runs WITHOUT a CUDA device)")
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
