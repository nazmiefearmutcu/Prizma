"""
PR-2026-09-03-08 CLAIM RUNNER — the Prizma-LM flagship bar (many-block continual char-LM,
fused column). Executes the FROZEN pre-registration
`docs/preregistry/2026-09-08-prizma-lm-flagship.md` VERBATIM when a GPU session runs it.
One script, three modes (the surprise_claim pattern, mirrored):

  python seq/prizma_lm_claim.py --smoke      # tiny CPU plumbing run -> the SMOKE ledger
  python seq/prizma_lm_claim.py --powered    # the A100 claim campaign -> the POWERED ledger
                                             # (refuses to start without a CUDA device)
  python seq/prizma_lm_claim.py --powered-cpu
                                             # the doc section-6 CPU-feasible fallback -> the
                                             # POWERED ledger on the local CPU box (protocol-
                                             # identical; doc addendum 2026-09-08 #2, point 1)

WHAT IS FROZEN (pre-reg §2-§5 — the constants below ARE that text, not choices):
  stream = 3 blocks, one pass, char-level: A = text8[0, 1.0M) -> B = tiny-shakespeare (full;
           eval = last 10%) -> C = text8 (1.0M chars). Held-out slices: A-eval = text8[1.0M,
           1.1M) (PR-07' pinned); C-retention ("C-eval" of bar B2) = text8[0.9M, 1.0M).
  arms   = PRIM-LM (backbone-train fusion, E=4, M_max=4 + Policy A eviction, floor-maturity
           veto freeze_min_seen=300, trunk trained on all segments) / FROZEN-TRUNK (backbone
           frozen after A) / SHARED-HEAD (equal extra params, one head, no routing) /
           FROZEN-CHECKPOINT (PR-07' adaptation floor) / FORCED-RECRUIT (C routing replaced
           by a forced fresh recruit).
  seeds  = claim (0,1,2,3,4), n=5 per arm, never substituted. LR = one per arm-family
           (fusion: PRIM-LM+FORCED-RECRUIT+FROZEN-TRUNK; shared; frozen-checkpoint), chosen
           on seed 0's first-200-A-segment loss over grid {1e-3, 3e-3, 1e-2} (PR-07' addendum
           rule; the FROZEN-TRUNK reuse of the fusion-family LR is the disclosed probe-2 P1
           convention — an A-phase selection cannot distinguish a frozen-B arm).
  bars   = B1 retention-A: per-seed d = BPC(A-eval, post-C) - BPC(A-eval, pre-B); one-sample
           95% CI upper <= 0.05. B2 adaptation-C: BPC_PRIM(C-retention, post-C) <=
           BPC_FROZEN-CHECKPOINT(same) - 0.10; Welch. B1+B2 raw one-sided p-values (UPPER-TAIL
           t_isf convention, the PR-03 lesson) form the Holm family of 2. B3 pattern completion
           (ledger primary): mean over PRIM-LM seeds of the fraction of C segments TRAINED by
           the A-recruited expert within the first 20 C-batches >= 0.5, AND
           mean(BPC_FORCED(B-eval, post-C)) >= mean(BPC_PRIM(B-eval, post-C)) - 0.10.
           B4 trunk-drift accounting: B-eval degradation post-C per arm, REPORTED not gated.
           INCONCLUSIVE mirror (PR-03): a straddling CI is INCONCLUSIVE, never PASS.
  branches = B1 FAIL -> PR-LM-1 wording downgrades to single-boundary (PR-07' scope), the
           trunk-drift measurement becomes the headline negative, FROZEN-TRUNK is the designed
           rescue path; B3 ledger FAIL -> the "lifelong routing" wording is retired regardless
           of accuracy; any INCONCLUSIVE -> maintainer addendum decides n->10 BEFORE unblinding,
           or the leg is honestly abandoned. All echoed verbatim in the verdict dict.

TWO DISCLOSED DOC-VS-CODE NOTES (no silent deviation — both recorded in the ledger meta):
  (1) C-RANGE REPAIR (maintainer addendum, 2026-09-08): pre-reg §2 pins Block C =
      text8[1.0M, 2.0M) AND A-eval = text8[1.0M, 1.1M). Taken literally, C-training would
      train on exactly the A-eval chars and bar B1 would measure TRAINING, not retention.
      The doc's own cited provenance (fusion_probe2 P3) skipped [1.0M, 1.1M) for exactly this
      reason. Per the maintainer's pre-run addendum the runner pins C = text8[1.1M, 2.1M) (1.0M chars, the
      doc's size, A-eval protected). This needs a dated maintainer addendum to formalize.
  (2) C-RETENTION SLICE PROSE BUG: §2 calls text8[0.9M, 1.0M) "held-out ... disjoint from
      A-train's tail", but text8[0.9M, 1.0M) IS A-train's tail ([0.9M, 1.0M) ⊂ [0, 1.0M) =
      A-train). The slice VALUE is unambiguous in the doc and the commission instructions and
      is implemented as pinned; the honesty consequence is disclosed: both B2 arms saw this
      slice in A-training equally, so the B2 comparison remains apples-to-apples (it measures
      C-training's in-domain benefit over the frozen floor, not unseen-text retention).

DISCLOSURE (review M-1; addendum #2 pre-disclosed the crash, addendum #3 defines the
  contingency that FIRED on the first powered-cpu execution): if the pool has no free slot
  at C start, the forced fresh recruit is placed by Policy A eviction (PRIM's own registered
  victim rule + pinned fresh-head formula); with a free slot the behavior is byte-identical
  to the registered semantics (see _forced_placement and the forced_placement provenance key).

TISSUE PROVENANCE: the fused column is REUSED verbatim from seq/fusion_probe.py + its probe-2
extensions (PCExpertHead, FusionLM/build_model, _segment_surprise, _expert_train local
optimizers on detached activations, SharedHeadLM capacity control, eval_bpc_fusion/
eval_bpc_shared). The two PR-08-only mechanisms are implemented in this file: the
floor-maturity veto (a novel segment whose argmin expert has n_segments < freeze_min_seen
does NOT trigger recruitment — the probe-2 cascade lesson, "an expert below a minimum
n_segments must not trigger recruits"; src/prizma.py's own freeze-gate has no LM-tissue analog
since the tissue has no expert-freezing, disclosed) and Policy A eviction at the M_max cap
(lowest lifetime routing share evicted, tie -> highest slot index; fresh head re-seeded by the
PINNED formula 20260908 + 17*slot + n_evictions).

LEDGER SEPARATION + RETENTION (surprise_claim pattern, mirrored):
  smoke   -> results/prizma_lm_PR-2026-09-03-08/smoke.json
  powered -> results/prizma_lm_PR-2026-09-03-08/powered.json
  A --smoke run pointed at the powered ledger is REFUSED unless --force-smoke-path. Raw records
  stream crash-safe after every cell AND are archived verbatim (seq.recall_gate.archive_run)
  BEFORE any verdict; the verdict references the archive path. Resume is keyed on
  (cellkey, config-fingerprint); a cell present at a FOREIGN fingerprint is a hard refusal.
  BUDGET GUARD: powered mode projects total wall from the first completed claim cell and WARNS
  when the projection exceeds 2 h (A100-equivalent; the projection is recorded in the ledger).

PR-2026-09-03-11 REPAIRED-FLAGSHIP MODE (guarded, DEFAULT-OFF; frozen protocol
  docs/preregistry/2026-09-09-repaired-flagship.md): the PR-08 protocol VERBATIM + ONE lever —
  the block-C BACKBONE lr is 7.5e-4 (= 3e-3 x 0.25, the PR-10 frozen dose) in every arm that
  trains a backbone on C (PRIM-LM and FORCED-RECRUIT via train_pr08's guarded backbone_lr;
  SHARED-HEAD via train_stream_shared's guarded kwarg — the capacity control sees the same
  schedule). FROZEN-TRUNK (backbone frozen after A) and FROZEN-CHECKPOINT (no B/C training)
  have no C backbone step: their training calls stay parameter-identical to PR-08 and their
  cells carry NO lever audit key, so they are byte-comparable canaries. FAIL-LOUD CANARIES
  (claim mode + lever ON only): after the FROZEN-TRUNK arm's 5 cells and again after the
  FROZEN-CHECKPOINT arm's 5 cells, every science field of claim.<ARM>.s{seed} must equal the
  PR-08 powered ledger's same-arm cells EXACTLY (ffc.canary_mismatches, ffc.CANARY_IGNORE;
  missing PR-08 ledger or any mismatch = SystemExit ABORT before any further arm runs).
  HONEST ORDERING NOTE: ARMS runs PRIM-LM BEFORE the first canary gate, so if a canary fails,
  the already-run PRIM cells are quarantine-suspect — the run aborts loudly either way, and
  the FROZEN canaries still prove the lever's blast radius end-to-end. Wiring:
  run(..., trunk_lr_c=None, ledger_dir=None, provenance=None) + CLI --trunk-lr-c /
  --ledger-dir. With trunk_lr_c None the behavior is byte-identical to before, including NO
  canary. The lever value is IN every cell + lr-selection fingerprint (a treated rerun never
  resumes from untreated cells); ledger_dir points the run at its OWN ledger subdir (default
  None = LEDDIR, byte-identical; the PR-11 run uses prizma_lm_PR-2026-09-03-11 so the PR-08
  ledger is never touched — it is read by the canaries, never written); provenance (when
  given) is recorded as meta["pr11_repair"].
"""
from __future__ import annotations

import argparse
import os
import sys

# Light imports only at module top: the pure layer (paths/refusal, slice pinning, bar math,
# verdict, budget guard) must be importable + unit-testable without torch or training
# (surprise_claim discipline). seq.stats pulls numpy only.
try:
    from .stats import t_isf, t_sf, holm_correction
except ImportError:                                   # run as a bare script: bootstrap sys.path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from seq.stats import t_isf, t_sf, holm_correction


# ================================================================== frozen protocol constants ====
REGISTRY_ID = "PR-2026-09-03-08"
LEDDIR = "prizma_lm_PR-2026-09-03-08"           # under $PRIZMA_RESULTS (default ./results)
SMOKE_BASENAME = "smoke.json"
POWERED_BASENAME = "powered.json"

ARMS = ("PRIM-LM", "FROZEN-TRUNK", "SHARED-HEAD", "FROZEN-CHECKPOINT", "FORCED-RECRUIT")
CLAIM_SEEDS = (0, 1, 2, 3, 4)                   # n=5 per arm — frozen, never substituted

# ---- stream pinning (doc §2 + the two disclosed notes in the module docstring) ----
A_CHARS = 1_000_000                    # Block A: text8[0, 1.0M)
A_EVAL_START, A_EVAL_END = 1_000_000, 1_100_000   # A-eval: text8[1.0M, 1.1M) (PR-07' pinned)
C_OFFSET = 1_100_000                   # REPAIR (maintainer addendum): doc said 1.0M; see docstring
C_CHARS = 1_000_000                    # Block C: text8[1.1M, 2.1M) — 1.0M chars as §2 specifies
C_RET_START, C_RET_END = 900_000, 1_000_000       # C-retention (B2's "C-eval"): text8[0.9M, 1.0M)
B_EVAL_FRAC = 0.10                     # B-eval = tiny-shakespeare last 10%

SEG = 256                              # chars per segment (PR-07' verbatim)
BATCH_SEGS = 32                        # segments per training step (PR-07' verbatim)
LR_GRID = (1e-3, 3e-3, 1e-2)           # PR-07' LR grid
LR_SELECT_SEGS = 200                   # PR-07' addendum: seed-0 first-200-A-segment loss

# ---- fused-column pinning (fusion_probe + probe-2 + pre-reg §3) ----
E_POOL = 4                             # vigilance-expert pool (doc §3 arm 1)
M_MAX = 4                              # bounded pool; Policy A eviction at the cap (PR-05 port)
FREEZE_MIN_SEEN = 300                  # floor-maturity veto (doc §3; probe-2 cascade lesson)
Z_NOVEL = 5.0                          # shipped src/prizma.py z_novel default (probe-pinned)
H_SMALL = 64                           # per-expert predictor width (probe-pinned)
H_SHARED = 256                         # SHARED-HEAD width ~ the E=4 pool budget (probe-2 P2)
FLOOR_EMA = 0.05                       # precision-floor EMA rate (src/prizma.py mirror)
FRESH_HEAD_SEED_BASE = 20260908        # pinned eviction re-init seed formula base

ARM_FAMILY = {                         # one LR per arm-family (PR-07' addendum rule)
    "PRIM-LM": "fusion",
    "FORCED-RECRUIT": "fusion",        # probe-2 P3: identical A/B trajectory, forced-C only
    "FROZEN-TRUNK": "fusion",          # probe-2 P1 disclosure (A-phase selection is blind to
                                       # the frozen-B phase) — carried, disclosed
    "SHARED-HEAD": "shared",
    "FROZEN-CHECKPOINT": "frozen-checkpoint",
}

# ---- bars (doc §4, exact) ----
B1_MARGIN = 0.05            # BPC(A-eval, post-C) - BPC(A-eval, pre-B) <= 0.05 (CI upper)
B2_MARGIN = 0.10            # BPC_PRIM(C-ret, post-C) <= BPC_FROZEN-C(C-ret) - 0.10 (Welch)
B3_FRAC_BAR = 0.5           # fraction of C segments (first 20 C-batches) to the A-expert
B3_WINDOW_BATCHES = 20      # "within the first 20 C-batches"
B3_FORCED_COST = 0.10       # forced arm's B-eval post-C must not be >0.10 worse than PRIM's
ALPHA = 0.05
BUDGET_WARN_MIN = 120.0     # powered wall-time projection warning threshold (2 h, A100-equiv)

C_RANGE_REPAIR_NOTE = (
    "DOC-VS-CODE REPAIR (maintainer addendum 2026-09-08 — see the doc's dated addendum section): "
    "pre-reg section 2 pins Block C = text8[1.0M, 2.0M) while A-eval = text8[1.0M, 1.1M) — "
    "literal C training would consume the A-eval slice and bar B1 would measure training, not "
    "retention. The doc's own cited provenance (fusion_probe2 P3) skipped [1.0M, 1.1M) for "
    "exactly this reason. Pinned here: C = text8[1.1M, 2.1M) (1.0M chars, the doc's block size, "
    "A-eval clean).")
C_RET_PROSE_NOTE = (
    "DOC PROSE BUG (disclosed, slice implemented as pinned): section 2 calls text8[0.9M, 1.0M) "
    "'held-out ... disjoint from A-train's tail', but that slice IS A-train's tail "
    "([0.9M, 1.0M) is inside [0, 1.0M)). Both B2 arms saw it equally in A-training, so B2 stays "
    "a fair adaptation comparison (C-training's in-domain benefit over the frozen floor).")
POWERED_CPU_FALLBACK_NOTE = (
    "POWERED VIA THE DOC SECTION-6 CPU-FEASIBLE FALLBACK (maintainer addendum 2026-09-08 #2, "
    "point 1): protocol-identical to the A100 tier; --powered (CUDA-refusing) remains "
    "available for the GPU session.")


# ==================================================================== paths + BAR-0 refusal ======
def _results_root() -> str:
    root = os.environ.get("PRIZMA_RESULTS", os.path.join(os.path.dirname(__file__), "..", "results"))
    return os.path.abspath(root)


def _default_results_path(smoke: bool, ledger_dir=None) -> str:
    # PR-2026-09-03-11 (guarded): ledger_dir overrides LEDDIR; None => LEDDIR (byte-identical).
    return os.path.join(_results_root(), LEDDIR if ledger_dir is None else ledger_dir,
                        SMOKE_BASENAME if smoke else POWERED_BASENAME)


def resolve_results_path(explicit=None, *, smoke: bool = False, force_smoke_path: bool = False,
                         ledger_dir=None) -> str:
    """Resolve the results path and enforce the smoke/powered file separation (the recall_gate
    refusal pattern): a --smoke run pointed at the POWERED ledger is refused (SystemExit) unless
    force_smoke_path. Smoke numbers are plumbing-only; a smoke entry inside the claim ledger is
    exactly the 2026-06-08 contamination mechanism. PR-2026-09-03-11 (guarded): ledger_dir
    overrides LEDDIR in both the resolved default and the powered-ledger refusal reference;
    None => byte-identical to before."""
    path = explicit if explicit else _default_results_path(smoke, ledger_dir)
    if smoke and not force_smoke_path:
        powered = _default_results_path(smoke=False, ledger_dir=ledger_dir)
        if os.path.normcase(os.path.abspath(path)) == os.path.normcase(os.path.abspath(powered)):
            raise SystemExit(
                f"refusing: a --smoke run was pointed at the powered ledger ({powered}). Smoke and "
                f"campaign results use separate files by default (PR-2026-09-03-08 §6 ledger "
                f"separation; see results/campaign_2026-06-08/CONTAMINATION.md). Pass --out <path> "
                f"to write the smoke elsewhere, or --force-smoke-path to override deliberately.")
    return path


def require_cuda(has_cuda: bool) -> None:
    """Pure guard for the --powered mode (unit-testable without a CUDA box): the claim campaign
    is the A100 Colab session's job; a CPU 'powered' run would be neither powered nor the
    pre-registered environment."""
    if not has_cuda:
        raise SystemExit(
            "refusing: --powered executes the PR-2026-09-03-08 claim campaign and requires a CUDA "
            "device (the A100 Colab session). No CUDA device is visible here. "
            "Use --smoke for the CPU plumbing run.")


def needs_cuda(mode: str) -> bool:
    """Pure mode->CUDA-requirement guard (unit-testable without torch): only the --powered A100
    claim campaign demands a CUDA device. --powered-cpu (the doc section-6 CPU-feasible
    fallback, doc addendum 2026-09-08 #2) and --smoke run CPU-side by design."""
    return mode == "powered"


# ================================================================ PURE: slice pinning ===========
def pin_slices() -> dict:
    """The §2 held-out slice table as a PURE pinned structure (unit-tested). Values are
    (corpus, start, end) with end-exclusive char offsets; B slices are fractions of the
    shakespeare corpus length. Asserts the disjointness invariants the bars depend on, so the
    C-range repair cannot silently regress."""
    sl = {
        "A_train":     ("text8", 0, A_CHARS),
        "A_eval":      ("text8", A_EVAL_START, A_EVAL_END),
        "C_retention": ("text8", C_RET_START, C_RET_END),
        "C_train":     ("text8", C_OFFSET, C_OFFSET + C_CHARS),
        "B_train":     ("shakespeare", 0.0, 1.0 - B_EVAL_FRAC),
        "B_eval":      ("shakespeare", 1.0 - B_EVAL_FRAC, 1.0),
    }
    # B1's premise: A-eval must NOT intersect C-train (the doc §2 slip this runner repairs).
    if max(A_EVAL_START, C_OFFSET) < min(A_EVAL_END, C_OFFSET + C_CHARS):
        raise AssertionError("A-eval intersects C-train — B1 would be corrupted; refusing")
    # C-retention must not intersect C-train or A-eval either.
    if max(C_RET_START, C_OFFSET) < min(C_RET_END, C_OFFSET + C_CHARS):
        raise AssertionError("C-retention intersects C-train; refusing")
    if max(C_RET_START, A_EVAL_START) < min(C_RET_END, A_EVAL_END):
        raise AssertionError("C-retention intersects A-eval; refusing")
    return sl


# ================================================================ PURE: bar math (§4) ===========
def _onesample_margin(diffs, margin):
    """One-sample t statistics for H1: mean(diffs) <= margin. The one-sided p uses the shipped
    UPPER-TAIL convention by flipping the hypothesis onto the advantage scale:
    p = P(T > (margin - mean)/se), small <=> mean significantly BELOW the margin.
    Returns mean, se, ci (two-sided 95%), p_raw, n."""
    n = len(diffs)
    assert n >= 2, "the bars require n >= 2 seeds (smoke never reaches the verdict)"
    mean = sum(diffs) / n
    var = sum((x - mean) ** 2 for x in diffs) / (n - 1)
    se = max((var / n) ** 0.5, 1e-12)
    tc = t_isf(0.025, n - 1)
    ci = (mean - tc * se, mean + tc * se)
    p_raw = t_sf((margin - mean) / se, n - 1)
    return {"n": n, "mean": mean, "se": se, "ci": ci, "p_raw": p_raw, "margin": margin}


def _welch_margin(a, b, margin):
    """Welch statistics for H1: mean(b) - mean(a) > margin — for BPC (lower is better), a is the
    candidate and b the baseline; margin is the required advantage. UPPER-TAIL p convention.
    Returns delta (= mean(b)-mean(a)), se, df, ci (two-sided 95% of delta), p_raw."""
    na, nb = len(a), len(b)
    assert na >= 2 and nb >= 2, "the bars require n >= 2 seeds per arm"
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    sea, seb = va / na, vb / nb
    se = max((sea + seb) ** 0.5, 1e-12)
    df = (sea + seb) ** 2 / (sea ** 2 / (na - 1) + seb ** 2 / (nb - 1) + 1e-30)
    delta = mb - ma                        # positive = candidate better by `delta` bpc
    tc = t_isf(0.025, df)
    ci = (delta - tc * se, delta + tc * se)
    p_raw = t_sf((delta - margin) / se, df)
    return {"n_a": na, "n_b": nb, "delta": delta, "se": se, "df": df, "ci": ci,
            "p_raw": p_raw, "margin": margin}


def _bar_status(ci, margin, p_holm, direction, alpha=ALPHA):
    """The PR-03-mirroring bar rule with the Holm gate: PASS iff the 95% CI is entirely on the
    passing side of the margin AND the Holm-adjusted p < alpha; FAIL iff the CI is entirely on
    the failing side; everything else (a straddling CI) is INCONCLUSIVE, never PASS.
    direction='below': the bar form is metric <= margin (B1: PASS iff ci_upper <= margin).
    direction='above': the bar form is advantage >= margin (B2: PASS iff ci_lower >= margin)."""
    if direction == "below":
        ci_pass, ci_fail = ci[1] <= margin, ci[0] > margin
    elif direction == "above":
        ci_pass, ci_fail = ci[0] >= margin, ci[1] < margin
    else:
        raise ValueError(f"unknown direction {direction!r}")
    if ci_pass and p_holm < alpha:
        return "PASS"
    if ci_fail:
        return "FAIL"
    return "INCONCLUSIVE"


def claim_verdict(prim, frozen_ck, forced, *, extra_b4=None, alpha=ALPHA):
    """The frozen §4 verdict as a PURE function over per-seed record dicts (unit-tested without
    torch). Args: lists of records with the pinned bpc keys — prim = PRIM-LM seeds,
    frozen_ck = FROZEN-CHECKPOINT seeds, forced = FORCED-RECRUIT seeds; extra_b4 (optional)
    maps arm name -> records consulted ONLY for the B4 report.

    B1: d_i = bpc_A_postC - bpc_A_preB (PRIM), one-sample CI upper <= 0.05.
    B2: bpc_Cret_postC PRIM vs FROZEN-CHECKPOINT, Welch, advantage >= 0.10.
        B1+B2 raw upper-tail p-values -> Holm family of 2 (alpha=0.05).
    B3: (a) mean over PRIM seeds of boundary_window.frac_to_A_expert >= 0.5;
        (b) mean(forced bpc_B_postC) >= mean(prim bpc_B_postC) - 0.10. (No CI in the doc —
        the bar is exact means; per-seed values are recorded either way.)
    B4: B-eval degradation post-C per arm — reported, never gated (doc §4).
    """
    b1 = _onesample_margin([r["bpc_A_postC"] - r["bpc_A_preB"] for r in prim], B1_MARGIN)
    b2 = _welch_margin([r["bpc_Cret_postC"] for r in prim],
                       [r["bpc_Cret_postC"] for r in frozen_ck], B2_MARGIN)
    holm = holm_correction([b1["p_raw"], b2["p_raw"]], alpha=alpha)
    b1["p_holm"] = holm[0]["p_adj"]
    b2["p_holm"] = holm[1]["p_adj"]
    b1["status"] = _bar_status(b1["ci"], B1_MARGIN, b1["p_holm"], "below", alpha)
    b2["status"] = _bar_status(b2["ci"], B2_MARGIN, b2["p_holm"], "above", alpha)

    # Real cell schema: the routing snapshot lives under rec["ledger"] (the smoke report reads
    # the same path). The first powered-cpu execution crashed here because the verdict read the
    # bare key — a path that never executes in smoke (smoke never reaches the verdict).
    fracs = [float(r["ledger"]["boundary_window"]["frac_to_A_expert"]) for r in prim]
    frac_mean = sum(fracs) / len(fracs)
    prim_b = [r["bpc_B_postC"] for r in prim]
    forced_b = [r["bpc_B_postC"] for r in forced]
    prim_b_mean = sum(prim_b) / len(prim_b)
    forced_b_mean = sum(forced_b) / len(forced_b)
    b3a_ok = frac_mean >= B3_FRAC_BAR
    b3b_ok = forced_b_mean >= prim_b_mean - B3_FORCED_COST
    b3 = {"frac_per_seed": fracs, "frac_mean": frac_mean, "frac_bar": B3_FRAC_BAR,
          "clause_a_ok": b3a_ok,
          "forced_b_per_seed": forced_b, "forced_b_mean": forced_b_mean,
          "prim_b_mean": prim_b_mean,
          "forced_cost": forced_b_mean - prim_b_mean, "forced_cost_bar": B3_FORCED_COST,
          "clause_b_ok": b3b_ok,
          "status": "PASS" if (b3a_ok and b3b_ok) else "FAIL"}

    # B4 trunk-drift accounting: reported per arm, never gated (doc §4).
    def _b4(recs):
        d = [r["bpc_B_postC"] - r["bpc_B_postB"] for r in recs]
        return {"per_seed": d, "mean": sum(d) / len(d)}
    b4 = {"PRIM-LM": _b4(prim), "FORCED-RECRUIT": _b4(forced)}
    if extra_b4:
        for arm, recs in extra_b4.items():
            b4[arm] = _b4(recs)

    # ---- overall verdict + pre-committed §5 branches, echoed verbatim ----
    branches = []
    statuses = [b1["status"], b2["status"], b3["status"]]
    if "INCONCLUSIVE" in statuses:
        branches.append("§5 (any INCONCLUSIVE): maintainer addendum decides seed extension "
                        "(n->10) BEFORE unblinding per-arm numbers, or the leg is honestly "
                        "abandoned. INCONCLUSIVE is never PASS (PR-03 mirror).")
    if b1["status"] == "FAIL":
        branches.append("§5 (B1 FAIL): tissue does not protect retention at many-block scale "
                        "=> PR-LM-1 wording downgrades to single-boundary (PR-07' scope); the "
                        "trunk-drift measurement becomes the headline negative; the frozen-trunk "
                        "ablation (B2 arm) is the designed rescue path, registrable separately.")
    if b3["status"] == "FAIL":
        branches.append("§5 (B3 ledger FAIL): the 'lifelong routing' wording is retired "
                        "regardless of accuracy (accuracy without routing is not the claim).")
    if "INCONCLUSIVE" in statuses:
        outcome = "INCONCLUSIVE"
    elif all(s == "PASS" for s in statuses):
        outcome = "PASS"
    else:
        outcome = "FAIL"
    if outcome == "PASS":
        verdict_text = ("CLAIM PASS — the fused column meets B1 (retention-A), B2 (adaptation-C) "
                        "and B3 (routing ledger + forced-recruit cost) at n=5; B4 trunk-drift "
                        "accounting is reported per arm (reported, not gated).")
    elif outcome == "FAIL":
        verdict_text = ("FAIL — " + "; ".join(
            f"{name} {st}" for name, st in zip(("B1", "B2", "B3"), statuses) if st == "FAIL") +
            ". Pre-committed §5 failure branches are echoed in 'branches'.")
    else:
        verdict_text = ("INCONCLUSIVE — a bar CI straddles its margin (PR-03 mirror: never "
                        "PASS). Pre-committed §5 branch applies (see 'branches').")
    return {"outcome": outcome, "verdict": verdict_text, "branches": branches,
            "B1": b1, "B2": b2, "B3": b3, "B4": b4,
            "holm_family": [{"bar": "B1", "p_raw": b1["p_raw"], "p_holm": b1["p_holm"],
                             "status": b1["status"]},
                            {"bar": "B2", "p_raw": b2["p_raw"], "p_holm": b2["p_holm"],
                             "status": b2["status"]}],
            "alpha": alpha}


# ================================================================ PURE: budget guard ============
def budget_projection(first_cell_s, n_cells_total, overhead_s=0.0, warn_min=BUDGET_WARN_MIN):
    """Powered-mode fixed budget guard: project total wall from the FIRST completed claim cell
    (linear in the cells) and WARN when the projection exceeds 2 h (A100-equivalent).
    Pure; the projection is recorded in the ledger either way."""
    if not first_cell_s or first_cell_s <= 0 or n_cells_total <= 0:
        return {"projected_min": None, "warn": False,
                "note": "no completed cell yet — no projection"}
    projected_min = (first_cell_s * n_cells_total + overhead_s) / 60.0
    return {"projected_min": round(projected_min, 1), "warn": bool(projected_min > warn_min),
            "warn_threshold_min": warn_min,
            "note": ("WARNING: projected wall exceeds %.0f min — check the ledger before "
                     "continuing (projection from the first cell only; A100-equivalent)."
                     % warn_min) if projected_min > warn_min else "within budget"}


# ================================================================================= runner =======
def _fresh_ledger():
    """PR-08 routing ledger: the probe fields plus the two registered event streams."""
    return {"recruits": [], "evictions": [], "veto_events": [], "vetoed_novel_segments": 0,
            "cap_fallback_batches": 0, "cap_fallback_segments": 0, "cap_events": []}


def _count_boundary(boundary, slot, n_segs, stream_pos):
    """B3a accounting: within the first B3_WINDOW_BATCHES C-batches, how many TRAINED segments
    went to which slot (the A-expert fraction is computed by the caller's ledger snapshot)."""
    if boundary is None:
        return
    if stream_pos // BATCH_SEGS + 1 > B3_WINDOW_BATCHES:
        return
    boundary["segments_total"] += int(n_segs)
    if slot == boundary.get("a_expert"):
        boundary["to_A_expert"] += int(n_segs)


def route_pr08(model, h, y, corpus, ledger, stream_pos, *, boundary=None, force_slot=None):
    """Segment-level vigilance routing with the two registered PR-08 mechanisms.

    Base semantics are seq.fusion_probe.route_batch VERBATIM (per-segment z-test against the
    owning expert's floor, batch-level recruit granularity, the fresh slot owns the batch's
    novel pool EXCLUSIVELY — the PR-06 ledger lesson). PR-08 adds:
      (1) FLOOR-MATURITY VETO (freeze_min_seen=300; the probe-2 cascade lesson): a novel
          segment whose ARGMIN expert has n_segments < FREEZE_MIN_SEEN does NOT trigger
          recruitment (its floor is immature; single-segment novelty under it is noise). The
          segment still routes by argmin. Veto events are counted and recorded.
      (2) POLICY A EVICTION (M_max=4; PR-05 port): a recruit when every slot is committed
          evicts the committed slot with the LOWEST lifetime routing share (n_segments; tie ->
          HIGHEST slot index, the probe's recency proxy), resets it to a fresh PCExpertHead
          (pinned seed formula) and recruits into it. The probe's argmin-fallback-at-cap is
          therefore structurally dead and its counter stays at 0 (recorded for ledger parity).
      `force_slot` (FORCED-RECRUIT, C only) is fusion_probe's probe-2 override verbatim: the
      first C batch commits the forced slot, which then owns ALL C segments exclusively.
    """
    from seq import fusion_probe as fp
    import torch

    B = y.shape[0]
    if force_slot is not None and corpus == "C":
        if not model.committed[force_slot]:
            model.committed[force_slot] = True
            ledger["recruits"].append({"slot": force_slot, "at_batch": stream_pos,
                                       "corpus": corpus, "n_novel_segs": int(B),
                                       "trigger_z_max": None, "reason": "forced_novel_C"})
        _count_boundary(boundary, force_slot, B, stream_pos)
        return [(force_slot, list(range(B)))]

    if model.n_committed() == 0:
        slot = next(s for s in range(model.E) if not model.committed[s])
        model.committed[slot] = True
        ledger["recruits"].append({"slot": slot, "at_batch": stream_pos, "corpus": corpus,
                                   "n_novel_segs": int(B), "trigger_z": None,
                                   "reason": "empty_pool"})
        _count_boundary(boundary, slot, B, stream_pos)
        return [(slot, list(range(B)))]

    S, slots = fp._segment_surprise(model, h, y)
    min_idx = S.argmin(dim=0)                                   # [B]
    seg_ce = S[min_idx, torch.arange(B)]
    mu = torch.tensor([model.mu[s] for s in slots])
    sd = torch.tensor([max(model.var[s], 1e-12) ** 0.5 for s in slots])
    z = (seg_ce - mu[min_idx]) / sd[min_idx]
    novel = z > Z_NOVEL
    novel_ids = sorted(int(i) for i in torch.nonzero(novel).reshape(-1))
    # (1) floor-maturity veto: novelty under an immature argmin expert never recruits.
    recruiting_ids, vetoed_ids = [], []
    for i in novel_ids:
        owner = slots[int(min_idx[i])]
        (recruiting_ids if model.n_segments[owner] >= FREEZE_MIN_SEEN else vetoed_ids).append(i)
    if vetoed_ids:
        ledger["vetoed_novel_segments"] += len(vetoed_ids)
        zs = [round(float(z[i]), 3) for i in vetoed_ids[:8]]
        ledger["veto_events"].append({"at_batch": stream_pos, "corpus": corpus,
                                      "n_vetoed_segs": len(vetoed_ids),
                                      "owner_slots": sorted({slots[int(min_idx[i])]
                                                             for i in vetoed_ids}),
                                      "z_first8": zs,
                                      "freeze_min_seen": FREEZE_MIN_SEEN})
    free = [s for s in range(model.E) if not model.committed[s]]

    if recruiting_ids:
        # (2) recruit — from a free slot, or by Policy A eviction at the M_max cap.
        if free:
            slot, reason = free[0], "novel"          # left-to-right fill = recruit recency
        else:
            cap = min(M_MAX, model.E)
            cands = [s for s in range(cap) if model.committed[s]]
            tot = max(float(sum(model.n_segments[s] for s in cands)), 1.0)
            victim = min(cands, key=lambda s: (model.n_segments[s] / tot, -s))
            slot, reason = victim, "novel_policy_a_eviction"
            shares = {f"e{s}": round(model.n_segments[s] / tot, 4) for s in cands}
            victim_nsegs = model.n_segments[victim]
            vocab_size = model.experts[victim].Wdec.out_features
            torch.manual_seed(FRESH_HEAD_SEED_BASE + 17 * victim + len(ledger["evictions"]))
            model.experts[victim] = fp.PCExpertHead(64, H_SMALL, vocab_size)
            model.mu[victim], model.var[victim] = 1e9, 1.0
            # PR-2026-09-03-09 floor-freeze lever (guarded, DEFAULT-OFF; frozen protocol
            # docs/preregistry/2026-09-09-floorfreeze-routing-repair.md §2): a re-initialized
            # slot LEAVES any floor-freeze set — it must recalibrate freely on its new domain.
            # discard = remove-if-present (the freeze is a plain Python set of slot ints).
            # Attribute absent -> fz is None -> no-op (off-identity).
            fz = getattr(model, "floor_freeze", None)
            if fz is not None:
                fz.discard(int(victim))
            model.n_batches[victim] = 0
            model.n_segments[victim] = 0
            model.ce_sum[victim] = 0.0
            ledger["evictions"].append({
                "victim": int(victim), "at_batch": stream_pos, "corpus": corpus,
                "victim_n_segments": victim_nsegs,
                "train_shares_at_fire": shares,
                "m_max": M_MAX,
                "policy": "A (lowest lifetime routing share; tie -> highest slot)",
                "fresh_head_seed": FRESH_HEAD_SEED_BASE + 17 * victim
                                   + len(ledger["evictions"]) - 1})
        model.committed[slot] = True
        zs = [round(float(z[i]), 3) for i in recruiting_ids[:8]]
        ledger["recruits"].append({"slot": slot, "at_batch": stream_pos, "corpus": corpus,
                                   "n_novel_segs": len(recruiting_ids),
                                   "trigger_z_max": max(zs) if zs else None, "reason": reason})
        assign = {slot: list(recruiting_ids)}
        for i in range(B):
            if i not in recruiting_ids:
                assign.setdefault(slots[int(min_idx[i])], []).append(i)
    else:
        # no RECRUITING novelty (none at all, or vetoed-only): argmin routing for every
        # segment. (The probe's argmin-fallback-at-cap is structurally dead under Policy A;
        # its counters stay 0 — recorded for ledger parity, not silently dropped.)
        assign = {}
        for i in range(B):
            assign.setdefault(slots[int(min_idx[i])], []).append(i)
    out = []
    for s, ids in assign.items():
        ids = sorted(ids)
        _count_boundary(boundary, s, len(ids), stream_pos)
        out.append((s, ids))
    return out


def train_pr08(model, xs, ys, lr, corpus, ledger, *, seed=None, frozen=False,
               boundary=None, force_slot=None, backbone_lr=None):
    """One online pass over in-order segment batches. Backbone step FIRST on the whole batch
    (shared LM loss, all segments — fusion_probe.train_stream_fusion verbatim); expert steps
    SECOND on their routed segments using the same (pre-update) forward — a single forward per
    batch, cost-honest. `frozen=True` (FROZEN-TRUNK after A) is
    fusion_probe.train_stream_fusion_frozen verbatim: backbone forward under no_grad, NO
    backbone optimizer step; only the routed tissue learns."""
    from seq import fusion_probe as fp
    import torch
    import torch.nn.functional as F

    if seed is not None:
        torch.manual_seed(seed)                    # claim.train_stream's seed convention
    # PR-2026-09-03-10 trunk-lr lever (guarded, DEFAULT-OFF; frozen protocol
    # docs/preregistry/2026-09-09-trunklr-routing-repair.md §2): backbone_lr overrides the
    # BACKBONE optimizer's lr only (tissue _expert_train keeps `lr`); None -> the exact
    # PR-08 value (off-identity pinned behaviorally by tests).
    opt_bb = torch.optim.AdamW(model.lm.parameters(),
                               lr=lr if backbone_lr is None else backbone_lr)
    model.train()
    n = xs.shape[0]
    for i in range(0, n, BATCH_SEGS):
        xb, yb = xs[i: i + BATCH_SEGS], ys[i: i + BATCH_SEGS]
        if frozen:
            with torch.no_grad():
                model.lm(xb)                       # hook captures hidden; no backbone update
        else:
            opt_bb.zero_grad()
            logits = model.lm(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                                   yb.reshape(-1))
            loss.backward()
            opt_bb.step()
        h = model._h
        for slot, ids in route_pr08(model, h, yb, corpus, ledger, i, boundary=boundary,
                                    force_slot=force_slot):
            fp._expert_train(model, slot, h, yb, ids, lr)


def ledger_snapshot_pr08(model, ledger, tr, ev_routes, expected, boundary, a_expert):
    """Per-block routing ledger (fusion_probe.ledger_snapshot3 shape) + the PR-08 event
    streams (evictions, vetoes) + the B3 boundary window."""
    per_expert = []
    for s in range(model.E):
        if not model.committed[s]:
            per_expert.append({"slot": s, "committed": False})
            continue
        tot = tr["A"][s] + tr["B"][s] + tr["C"][s]
        per_expert.append({
            "slot": s, "committed": True,
            "train_A": tr["A"][s], "train_B": tr["B"][s], "train_C": tr["C"][s],
            "train_total": tot,
            "mu": round(model.mu[s], 4), "sigma": round(max(model.var[s], 0.0) ** 0.5, 4),
            "n_batches": model.n_batches[s], "n_segments": model.n_segments[s],
            "is_A_expert": bool(s == a_expert),
        })
    block_majority = {}
    balance = {}
    for blk in ("A", "B", "C"):
        nseg = sum(tr[blk])
        block_majority[blk] = (round(max(tr[blk]) / nseg, 4) if nseg else None)
        balance[blk] = {"sum": nseg, "expected": expected[blk], "ok": nseg == expected[blk]}
    frac = (boundary["to_A_expert"] / boundary["segments_total"]) \
        if boundary["segments_total"] else None
    return {"n_committed": model.n_committed(),
            "a_expert": a_expert,
            "recruits": ledger["recruits"],
            "evictions": ledger["evictions"],
            "veto_events": ledger["veto_events"][:50],
            "vetoed_novel_segments": ledger["vetoed_novel_segments"],
            "cap_fallback_batches": ledger["cap_fallback_batches"],
            "cap_fallback_segments": ledger["cap_fallback_segments"],
            "cap_events": ledger["cap_events"][:50],
            "per_expert": per_expert,
            "block_majority_share": block_majority,
            "train_sums_balance": balance,
            "eval_routes": ev_routes,
            "boundary_window": {
                "c_batches_1_to": B3_WINDOW_BATCHES,
                "segments_total": boundary["segments_total"],
                "to_A_expert": boundary["to_A_expert"],
                "frac_to_A_expert": round(frac, 4) if frac is not None else None,
                "recruit_events_c_batches_1_to_20": [
                    {"slot": r["slot"], "c_batch_1based": r["at_batch"] // BATCH_SEGS + 1,
                     "reason": r["reason"]}
                    for r in ledger["recruits"] if r.get("corpus") == "C"
                    and r["at_batch"] // BATCH_SEGS < B3_WINDOW_BATCHES],
            }}


def _forced_placement(model, *, E, stream_pos):
    """Registered forced-recruit slot placement (doc addendum #3, 2026-09-08). A free slot when
    one exists (the registered semantics, byte-identical); otherwise the PRIM arm's own Policy A
    eviction (lowest lifetime routing share, tie -> HIGHEST slot) makes room — the
    previously-fatal pool-full path (review M-1, fired on the first powered-cpu execution at
    FORCED seed 2). Pure decision over the model's committed/n_segments state; the caller
    applies the torch-side re-init from the returned record. Returns (slot, placement_record)."""
    free = [s for s in range(E) if not model.committed[s]]
    if free:
        return free[0], {"mode": "free_slot"}
    cap = min(M_MAX, E)
    cands = [s for s in range(cap) if model.committed[s]]
    tot = max(float(sum(model.n_segments[s] for s in cands)), 1.0)
    victim = min(cands, key=lambda s: (model.n_segments[s] / tot, -s))
    return victim, {"mode": "policy_a_eviction", "victim": int(victim),
                    "victim_n_segments": int(model.n_segments[victim]),
                    "train_shares_at_fire": {f"e{s}": round(model.n_segments[s] / tot, 4)
                                             for s in cands},
                    "m_max": M_MAX,
                    "policy": "A (lowest lifetime routing share; tie -> highest slot)",
                    "at_batch": int(stream_pos)}


def _backbone_lr_for_c(trunk_lr_c, frozen_after_A):
    """PR-2026-09-03-11 guarded lever (PURE truth table, USED by run_routed): the block-C
    backbone-lr override for a routed arm. Returns None when the lever is OFF (trunk_lr_c is
    None) or structurally inapplicable (frozen_after_A: FROZEN-TRUNK has no C backbone step) —
    in both cases the train_pr08 C call stays parameter-identical to PR-08 and the cell
    carries NO lever audit key, keeping FROZEN-TRUNK cells byte-comparable canaries."""
    if trunk_lr_c is None or frozen_after_A:
        return None
    return trunk_lr_c


def run_routed(vocab_size, seed, data, lr, *, frozen_after_A=False, forced_c=False,
               trunk_lr_c=None):
    """PRIM-LM / FROZEN-TRUNK / FORCED-RECRUIT cell. A and B phases are identical across
    PRIM-LM and FORCED-RECRUIT (FROZEN-TRUNK shares phase A only; its B-phase backbone is
    frozen by design) (same seed stream -> identical init and trajectory — the probe-2
    determinism property); FROZEN-TRUNK freezes the backbone after A (parameter-identity
    proven); FORCED-RECRUIT replaces C routing with a forced fresh recruit (probe-2
    override).
    trunk_lr_c (PR-2026-09-03-11, guarded DEFAULT-OFF): the block-C backbone-lr lever,
    applied via _backbone_lr_for_c ONLY when trunk_lr_c is not None AND this arm trains a
    backbone on C (not frozen_after_A) — FROZEN-TRUNK's C call stays parameter-identical to
    PR-08 and carries no audit key. When the lever is applied the cell records
    rec["backbone_lr_c_applied"] = trunk_lr_c."""
    import time
    import torch
    from seq import fusion_probe as fp

    t0 = time.time()
    Ax, Ay, Bx, By, Cx, Cy, Aex, Aey, Bex, Bey, Crx, Cry = data
    E = E_POOL
    model = fp.build_model(vocab_size, seed, E)
    ledger = _fresh_ledger()
    tr = {"A": [0] * E, "B": [0] * E, "C": [0] * E}
    rec = {"config": ("FROZEN-TRUNK" if frozen_after_A else
                      "FORCED-RECRUIT" if forced_c else "PRIM-LM"),
           "E": E, "seed": seed, "lr": lr}

    # ---- block A (backbone trains; the first slot recruits into the empty pool) ----
    train_pr08(model, Ax, Ay, lr, "A", ledger, seed=seed)
    tr["A"] = model.n_segments[:]
    rec["bpc_A_preB"] = fp.eval_bpc_fusion(model, Aex, Aey)

    # ---- block B (drift, strong) ----
    before = model.n_segments[:]
    bb = None
    if frozen_after_A:
        bb = [p.detach().clone() for p in model.lm.parameters()]
    train_pr08(model, Bx, By, lr, "B", ledger, seed=seed, frozen=frozen_after_A)
    if bb is not None:
        rec["backbone_frozen_check"] = all(
            torch.equal(x, y) for x, y in zip(bb, [p.detach().clone()
                                                   for p in model.lm.parameters()]))
    tr["B"] = [model.n_segments[s] - before[s] for s in range(E)]
    rec["bpc_A_postB"] = fp.eval_bpc_fusion(model, Aex, Aey)      # descriptive
    rec["bpc_B_postB"] = fp.eval_bpc_fusion(model, Bex, Bey)
    rec["bpc_Cret_postB"] = fp.eval_bpc_fusion(model, Crx, Cry)   # descriptive

    # ---- block C (drift, mild + RETURNING domain) ----
    a_expert = max((s for s in range(E) if model.committed[s]), key=lambda s: tr["A"][s])
    rec["a_expert"] = int(a_expert)
    force_slot = None
    if forced_c:
        force_slot, placement = _forced_placement(model, E=E,
                                                  stream_pos=Ax.shape[0] + Bx.shape[0])
        rec["forced_slot"] = int(force_slot)
        rec["forced_placement"] = placement
        if placement["mode"] == "policy_a_eviction":
            # The pool-full contingency (doc addendum #3): re-init the victim exactly like
            # route_pr08's Policy A eviction (pinned fresh-head seed formula + ledger record),
            # then hand the slot to the forced-C path below.
            victim = force_slot
            vocab_size_victim = model.experts[victim].Wdec.out_features
            torch.manual_seed(FRESH_HEAD_SEED_BASE + 17 * victim + len(ledger["evictions"]))
            model.experts[victim] = fp.PCExpertHead(64, H_SMALL, vocab_size_victim)
            model.mu[victim], model.var[victim] = 1e9, 1.0
            model.n_batches[victim] = 0
            model.n_segments[victim] = 0
            model.ce_sum[victim] = 0.0
            ledger["evictions"].append({
                "victim": int(victim), "at_batch": int(Ax.shape[0] + Bx.shape[0]),
                "corpus": "C", "victim_n_segments": placement["victim_n_segments"],
                "train_shares_at_fire": placement["train_shares_at_fire"],
                "m_max": M_MAX,
                "policy": "A (lowest lifetime routing share; tie -> highest slot)",
                "fresh_head_seed": FRESH_HEAD_SEED_BASE + 17 * victim
                                   + len(ledger["evictions"]) - 1,
                "reason": "forced_recruit_pool_full (doc addendum #3)"})
    boundary = {"segments_total": 0, "to_A_expert": 0, "a_expert": int(a_expert)}
    before = model.n_segments[:]
    # PR-2026-09-03-11 guarded lever: the C call passes backbone_lr ONLY when the lever is ON
    # and this arm trains a backbone on C (empty kwargs = parameter-identical to PR-08, so
    # FROZEN-TRUNK cells remain byte-comparable canaries).
    bb_lr_c = _backbone_lr_for_c(trunk_lr_c, frozen_after_A)
    c_kwargs = {}
    if bb_lr_c is not None:
        rec["backbone_lr_c_applied"] = trunk_lr_c
        c_kwargs["backbone_lr"] = bb_lr_c
    train_pr08(model, Cx, Cy, lr, "C", ledger, seed=seed, boundary=boundary,
               force_slot=force_slot, **c_kwargs)
    tr["C"] = [model.n_segments[s] - before[s] for s in range(E)]

    ev_A, ev_B, ev_Cr = [0] * E, [0] * E, [0] * E
    rec["bpc_A_postC"] = fp.eval_bpc_fusion(model, Aex, Aey, routes=ev_A)
    rec["bpc_B_postC"] = fp.eval_bpc_fusion(model, Bex, Bey, routes=ev_B)
    rec["bpc_Cret_postC"] = fp.eval_bpc_fusion(model, Crx, Cry, routes=ev_Cr)
    rec["fgt_A_full"] = rec["bpc_A_preB"] - rec["bpc_A_postC"]          # descriptive
    rec["b_degradation_B_postC"] = rec["bpc_B_postC"] - rec["bpc_B_postB"]  # B4 quantity
    rec["ledger"] = ledger_snapshot_pr08(model, ledger, tr,
                                         {"A_eval": ev_A, "B_eval": ev_B,
                                          "C_retention": ev_Cr},
                                         {"A": int(Ax.shape[0]), "B": int(Bx.shape[0]),
                                          "C": int(Cx.shape[0])},
                                         boundary, int(a_expert))
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def run_shared(vocab_size, seed, data, lr, *, trunk_lr_c=None):
    """SHARED-HEAD capacity control (probe-2 P2 verbatim): one head, no routing, the exact
    _expert_train learning rule on every segment; backbone objective unchanged.
    trunk_lr_c (PR-2026-09-03-11, guarded DEFAULT-OFF): threaded to train_stream_shared's
    guarded kwarg on the C-phase call ONLY (the capacity control must see the same C
    backbone schedule to stay honest — frozen protocol §2); when OFF the call is
    parameter-identical to PR-08. When applied the cell records
    rec["backbone_lr_c_applied"] = trunk_lr_c."""
    import time
    from seq import fusion_probe as fp

    t0 = time.time()
    Ax, Ay, Bx, By, Cx, Cy, Aex, Aey, Bex, Bey, Crx, Cry = data
    model = fp.SharedHeadLM(vocab_size, seed, H_SHARED)
    rec = {"config": "SHARED-HEAD", "h_small": H_SHARED, "seed": seed, "lr": lr,
           "routing": "none (single shared head, trained on ALL segments)"}
    fp.train_stream_shared(model, Ax, Ay, lr)
    rec["bpc_A_preB"] = fp.eval_bpc_shared(model, Aex, Aey)
    fp.train_stream_shared(model, Bx, By, lr)
    rec["bpc_A_postB"] = fp.eval_bpc_shared(model, Aex, Aey)
    rec["bpc_B_postB"] = fp.eval_bpc_shared(model, Bex, Bey)
    rec["bpc_Cret_postB"] = fp.eval_bpc_shared(model, Crx, Cry)
    # PR-2026-09-03-11 guarded lever (C-phase call only; OFF = parameter-identical to PR-08).
    if trunk_lr_c is not None:
        rec["backbone_lr_c_applied"] = trunk_lr_c
        fp.train_stream_shared(model, Cx, Cy, lr, backbone_lr=trunk_lr_c)
    else:
        fp.train_stream_shared(model, Cx, Cy, lr)
    rec["bpc_A_postC"] = fp.eval_bpc_shared(model, Aex, Aey)
    rec["bpc_B_postC"] = fp.eval_bpc_shared(model, Bex, Bey)
    rec["bpc_Cret_postC"] = fp.eval_bpc_shared(model, Crx, Cry)
    rec["fgt_A_full"] = rec["bpc_A_preB"] - rec["bpc_A_postC"]
    rec["b_degradation_B_postC"] = rec["bpc_B_postC"] - rec["bpc_B_postB"]
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def run_frozen_checkpoint(vocab_size, seed, data, lr):
    """FROZEN-CHECKPOINT control (PR-07' FROZEN analog): train A, then freeze; the B2
    adaptation floor. postB/postC keys alias the same post-A evals (weights unchanged after A
    by construction; evaluated once, honestly aliased)."""
    import time
    from seq import blockdrift_claim as claim

    t0 = time.time()
    Ax, Ay, Bx, By, Cx, Cy, Aex, Aey, Bex, Bey, Crx, Cry = data
    model, _ = claim.build_model(vocab_size, seed)
    rec = {"config": "FROZEN-CHECKPOINT", "seed": seed, "lr": lr,
           "routing": "none (plain backbone, frozen after A)"}
    claim.train_stream(model, Ax, Ay, lr, seed)
    a = claim.eval_bpc(model, Aex, Aey)
    b = claim.eval_bpc(model, Bex, Bey)
    cr = claim.eval_bpc(model, Crx, Cry)
    rec.update({"bpc_A_preB": a, "bpc_A_postB": a, "bpc_A_postC": a,
                "bpc_B_postB": b, "bpc_B_postC": b,
                "bpc_Cret_postB": cr, "bpc_Cret_postC": cr,
                "frozen_note": "weights unchanged after A; postB==postC by construction "
                               "(single evaluation, honestly aliased)",
                "fgt_A_full": 0.0, "b_degradation_B_postC": 0.0})
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


# ------------------------------------------------------------------ runner core -----------------
def _fp(payload: dict) -> str:
    from seq.gpu_harness import config_fingerprint
    return config_fingerprint({"registry": REGISTRY_ID, **payload})


def _pr11_powered_path() -> str:
    """The PR-08 powered ledger — the canary's READ-ONLY reference. Always LEDDIR (never
    ledger_dir): a PR-11 run pointed at its own ledger subdir must still verify against the
    untouched PR-08 ledger."""
    return os.path.join(_results_root(), LEDDIR, POWERED_BASENAME)


def _pr11_canary(res, seeds, arm):
    """PR-2026-09-03-11 fail-loud canary (claim mode + lever ON ONLY): every science field of
    the just-finished FROZEN arm's cells (FROZEN-TRUNK / FROZEN-CHECKPOINT — no C backbone
    step, so the lever is structurally inapplicable there) must equal the PR-08 powered
    ledger's claim.<ARM>.s{seed} EXACTLY. Comparator + ignore set REUSED from
    seq/floorfreeze_claim.py (the PR-09/PR-10 canary style; seq/trunklr_claim.py aliases the
    same objects). A missing PR-08 ledger or any mismatch is a SystemExit ABORT before any
    further arm runs. Lazy ffc import: ffc imports THIS module at its own module level, so it
    must never be imported at ours."""
    from seq.gpu_harness import load_results
    import seq.floorfreeze_claim as ffc

    pr08_path = _pr11_powered_path()
    if not os.path.isfile(pr08_path):
        raise SystemExit(
            "CANARY ABORT (PR-2026-09-03-11): the PR-08 powered ledger is MISSING at "
            f"{pr08_path} — the frozen control arm {arm} cannot be verified against "
            f"claim.{arm}.s{{0..4}}. The registered protocol aborts before any further arm "
            "runs; run this where the PR-08 ledger exists.")
    pr08 = load_results(pr08_path)
    checked = []
    for seed in seeds:
        cellkey = f"claim.{arm}.s{seed}"
        fresh = res.get(cellkey)
        ref = pr08.get(cellkey)
        if not isinstance(fresh, dict) or not isinstance(ref, dict):
            raise SystemExit(
                f"CANARY ABORT (PR-2026-09-03-11): missing cell for comparison ({cellkey} "
                "absent here or in the PR-08 ledger) — refusing to proceed to any further arm.")
        mm = ffc.canary_mismatches(fresh, ref)
        if mm:
            raise SystemExit(
                f"CANARY ABORT (PR-2026-09-03-11): {cellkey} does NOT reproduce the PR-08 "
                f"record bit-identically — mismatched science fields: {mm}. A frozen arm has "
                "no C backbone step, so the trunk-lr lever must not touch it; arms already run "
                "before this gate (PRIM-LM precedes it in ARMS order) are quarantine-suspect. "
                f"Ledger: {pr08_path}")
        checked.append(cellkey)
    rec = {"ok": True, "arm": arm, "pr08_ledger": pr08_path, "cells_checked": checked,
           "ignored_fields": sorted(ffc.CANARY_IGNORE),
           "note": (f"every science field of {arm} == PR-08 claim.{arm} EXACTLY "
                    "(no C backbone step -> the lever is structurally inapplicable here)")}
    print(f"[pr11] CANARY PASS ({arm}): {len(checked)} cells bit-identical to the PR-08 "
          f"powered ledger ({pr08_path})", flush=True)
    return rec


def run(*, smoke: bool, results_path=None, force_smoke_path: bool = False,
        powered_cpu: bool = False, trunk_lr_c=None, ledger_dir=None, provenance=None):
    """Execute the protocol: --smoke (CPU plumbing), --powered (the A100 claim campaign), or
    --powered-cpu (the doc section-6 CPU-feasible fallback: identical to --powered except the
    CUDA guard is skipped and the ledger meta records powered_cpu + the fallback note).
    PR-2026-09-03-11 (guarded, DEFAULT-OFF): trunk_lr_c = the block-C backbone-lr lever for
    the arms that train a backbone on C (None = byte-identical PR-08 behavior, including NO
    canary); ledger_dir redirects the ledger subdir (None = LEDDIR, byte-identical);
    provenance (when given) is recorded as meta["pr11_repair"]. The lever value is part of
    every cell + lr-selection fingerprint."""
    # BAR-0 FIRST: resolve + guard the results path before any heavy import or write.
    path = resolve_results_path(results_path, smoke=smoke, force_smoke_path=force_smoke_path,
                                ledger_dir=ledger_dir)

    import time

    import torch

    from seq import blockdrift_claim as claim          # PR-07' protocol pieces, imported VERBATIM
    from seq import fusion_probe as fp                 # the fused column, imported VERBATIM
    from seq.gpu_harness import load_results, _save
    from seq.recall_gate import archive_run

    torch.set_num_threads(int(os.environ.get("BENCH_THREADS", "8")))

    if smoke:
        smoke_segs = 64                        # per training block (probe-2 smoke precedent;
                                               # held-out eval slices stay FULL)
        seeds = (0,)
    else:
        if needs_cuda("powered-cpu" if powered_cpu else "powered"):
            require_cuda(torch.cuda.is_available())  # refuse BEFORE anything else on a CPU-only box
        # (--powered-cpu skips that guard by design: the doc section-6 CPU-feasible fallback,
        # doc addendum 2026-09-08 #2 — protocol-identical to powered in every other respect)
        smoke_segs = None
        seeds = CLAIM_SEEDS

    # ---------------- data: the 3-block stream + the pinned eval slices ------------------------
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

    # ---------------- ledger + meta -----------------------------------------------------------
    res = load_results(path)
    bb_params = sum(p.numel() for p in fp._PlainWrap(V, 0).lm.parameters())
    per_expert = sum(p.numel() for p in fp.PCExpertHead(64, H_SMALL, V).parameters())
    shared_params = sum(p.numel() for p in fp.PCExpertHead(64, H_SHARED, V).parameters())
    res["meta"] = {
        "registry": REGISTRY_ID, "smoke": bool(smoke),
        "lane": "CLAIM (frozen pre-registration)" if not smoke else "SMOKE (plumbing only)",
        "doc": "docs/preregistry/2026-09-08-prizma-lm-flagship.md",
        "arms": list(ARMS), "arm_family": dict(ARM_FAMILY), "claim_seeds": list(seeds),
        "seg": SEG, "batch_segs": BATCH_SEGS, "lr_grid": list(LR_GRID),
        "lr_select_segs": LR_SELECT_SEGS,
        "lr_rule": "one per arm-family, seed-0 first-200-A-segment loss, then frozen (PR-07' "
                   "addendum); smoke fixes 3e-3 (probe-2 precedent)",
        "tissue": {"E_pool": E_POOL, "m_max": M_MAX, "freeze_min_seen": FREEZE_MIN_SEEN,
                   "z_novel": Z_NOVEL, "h_small": H_SMALL, "h_shared": H_SHARED,
                   "floor_ema": FLOOR_EMA,
                   "fresh_head_seed_formula": f"{FRESH_HEAD_SEED_BASE} + 17*slot + n_evictions",
                   "params_backbone": bb_params, "params_per_expert": per_expert,
                   "params_pool_e4": per_expert * E_POOL, "params_shared_h256": shared_params,
                   "shared_parity_note": f"shared head {shared_params} vs pool "
                                         f"{per_expert * E_POOL} params (probe-2 P2 budget)"},
        "pinned_slices": {k: list(v) for k, v in slices.items()},
        "stream_lengths": {"A_train": len(A_train), "B_train": len(B_train),
                           "C_train": len(C_train), "A_eval": len(A_eval),
                           "B_eval": len(B_eval), "C_retention": len(C_ret)},
        "vocab": V, "threads": torch.get_num_threads(),
        "c_range_repair": C_RANGE_REPAIR_NOTE,
        "c_retention_prose_note": C_RET_PROSE_NOTE,
        "bars": {"B1_margin": B1_MARGIN, "B2_margin": B2_MARGIN, "B3_frac": B3_FRAC_BAR,
                 "B3_window_batches": B3_WINDOW_BATCHES, "B3_forced_cost": B3_FORCED_COST,
                 "alpha": ALPHA, "holm_family": ["B1", "B2"],
                 "t_isf_convention": "UPPER-TAIL p in (0, 0.5] (the PR-03 lesson)"},
        "boundary_note": "A-expert := the committed slot with max train_A at C start (B3's "
                         "'A-recruited expert'); frac = TRAINING-ledger fraction of C segments "
                         "in the first 20 C-batches assigned to it",
    }
    if powered_cpu:
        res["meta"]["powered_cpu"] = True
        res["meta"]["compute_fallback_note"] = POWERED_CPU_FALLBACK_NOTE
    if provenance is not None:
        res["meta"]["pr11_repair"] = provenance   # PR-2026-09-03-11 provenance (orchestrator)
    _save(res, path)
    print(f"[pr08] corpora: A={Ax.shape[0]} B={Bx.shape[0]} C={Cx.shape[0]} segs; "
          f"eval A={Aex.shape[0]} B={Bex.shape[0]} Cret={Crx.shape[0]}; vocab={V}; "
          f"smoke={smoke}; results={path}", flush=True)

    # ---------------- LR selection (one per arm-family; PR-07' addendum rule) ------------------
    lrs, lr_secs = {}, 0.0
    xs200, ys200 = Ax[:LR_SELECT_SEGS], Ay[:LR_SELECT_SEGS]
    if smoke:
        for family in ("fusion", "shared", "frozen-checkpoint"):
            lrs[family] = 3e-3               # probe-2 smoke precedent: selection is plumbing
        res["lr_selection"] = {"smoke_note": "LR selection skipped (fixed 3e-3, probe-2 smoke "
                                             "precedent); the numbers are plumbing-only"}
        _save(res, path)
    else:
        def _lr_cell(family, build_and_score):
            nonlocal lr_secs
            cellkey = f"lr_selection.{family}"
            cfgsig = _fp({"leg": "lr-selection", "family": family, "lr_grid": list(LR_GRID),
                          "select_segs": int(xs200.shape[0]), "seed": 0, "vocab": V,
                          "smoke": bool(smoke), "tissue": res["meta"]["tissue"],
                          "slices": res["meta"]["pinned_slices"],
                          "trunk_lr_c": trunk_lr_c})   # PR-11: the lever is IN the fingerprint
                                                       # (a treated rerun never resumes from
                                                       # untreated lr-selection cells)
            rec = res.get(cellkey)
            if isinstance(rec, dict) and rec.get("cfgsig") == cfgsig and rec.get("complete"):
                lrs[family] = rec["lr"]
                return
            if isinstance(rec, dict):
                raise SystemExit(f"cell {cellkey} exists at a foreign config fingerprint "
                                 f"({rec.get('cfgsig')} != {cfgsig}) — refusing to resume")
            t0 = time.time()
            grid = build_and_score()
            lr = min(grid, key=grid.get)
            lr_secs += time.time() - t0
            res[cellkey] = {"family": family, "lr": lr,
                            "grid": {str(k): v for k, v in grid.items()},
                            "seconds": round(time.time() - t0, 1), "cfgsig": cfgsig,
                            "complete": True}
            _save(res, path)
            lrs[family] = lr
            print(f"[pr08] lr-select {family}: lr={lr} grid={grid}", flush=True)

        def _sel_fusion():
            grid = {}
            for lr in LR_GRID:
                m = fp.build_model(V, 0, E_POOL)
                led = _fresh_ledger()
                train_pr08(m, xs200, ys200, lr, "A", led, seed=0)
                grid[lr] = fp.eval_bpc_fusion(m, xs200, ys200)
                print(f"[pr08]   fusion lr={lr} A-loss={grid[lr]:.4f}", flush=True)
            return grid

        def _sel_shared():
            grid = {}
            for lr in LR_GRID:
                m = fp.SharedHeadLM(V, 0, H_SHARED)
                fp.train_stream_shared(m, xs200, ys200, lr)
                grid[lr] = fp.eval_bpc_shared(m, xs200, ys200)
                print(f"[pr08]   shared lr={lr} A-loss={grid[lr]:.4f}", flush=True)
            return grid

        def _sel_frozenck():
            grid = {}
            for lr in LR_GRID:
                m, _ = claim.build_model(V, 0)
                claim.train_stream(m, xs200, ys200, lr, 0)
                grid[lr] = claim.eval_bpc(m, xs200, ys200)
                print(f"[pr08]   frozen-checkpoint lr={lr} A-loss={grid[lr]:.4f}", flush=True)
            return grid

        _lr_cell("fusion", _sel_fusion)
        _lr_cell("shared", _sel_shared)
        _lr_cell("frozen-checkpoint", _sel_frozenck)

    # ---------------- the 5 arms x seeds --------------------------------------------------------
    n_cells_total = len(ARMS) * len(seeds)
    first_cell_s = None
    for arm in ARMS:
        lr = lrs[ARM_FAMILY[arm]]
        for seed in seeds:
            cellkey = f"claim.{arm}.s{seed}"
            cfgsig = _fp({"leg": "claim", "arm": arm, "family": ARM_FAMILY[arm], "seed": seed,
                          "lr": lr, "smoke": bool(smoke), "vocab": V,
                          "smoke_segs": smoke_segs, "seg": SEG, "batch_segs": BATCH_SEGS,
                          "slices": res["meta"]["pinned_slices"],
                          "stream_lengths": res["meta"]["stream_lengths"],
                          "tissue": res["meta"]["tissue"],
                          "bars": res["meta"]["bars"],   # review M-2: the bars constants (B3's
                                                         # window applies at train time) must
                                                         # invalidate stale resume cells
                          "trunk_lr_c": trunk_lr_c})     # PR-11: the lever is IN the fingerprint
                                                         # (a treated rerun never resumes from
                                                         # untreated cells)
            prior = res.get(cellkey)
            if isinstance(prior, dict) and prior.get("cfgsig") == cfgsig and prior.get("complete"):
                print(f"[pr08] {arm} seed {seed}: resumed from ledger "
                      f"(bpc_A_postC={prior['bpc_A_postC']:.3f})", flush=True)
                continue
            if isinstance(prior, dict):
                raise SystemExit(f"cell {cellkey} exists at a foreign config fingerprint "
                                 f"({prior.get('cfgsig')} != {cfgsig}) — refusing to resume")
            t0 = time.time()
            if arm == "FROZEN-CHECKPOINT":
                rec = run_frozen_checkpoint(V, seed, data, lr)
            elif arm == "SHARED-HEAD":
                rec = run_shared(V, seed, data, lr, trunk_lr_c=trunk_lr_c)
            else:
                # PRIM-LM / FROZEN-TRUNK / FORCED-RECRUIT. trunk_lr_c is threaded for all
                # three; _backbone_lr_for_c nulls it for FROZEN-TRUNK (no C backbone step —
                # its call stays parameter-identical to PR-08, no audit key: canary-clean).
                rec = run_routed(V, seed, data, lr,
                                 frozen_after_A=(arm == "FROZEN-TRUNK"),
                                 forced_c=(arm == "FORCED-RECRUIT"),
                                 trunk_lr_c=trunk_lr_c)
            rec.update({"arm": arm, "seed": seed, "lr": lr, "cellkey": cellkey,
                        "cfgsig": cfgsig, "complete": True,
                        "wall_s": round(time.time() - t0, 1)})
            res[cellkey] = rec
            _save(res, path)                    # crash-safe after every cell
            if first_cell_s is None:
                first_cell_s = rec["wall_s"]
                proj = budget_projection(first_cell_s, n_cells_total, overhead_s=lr_secs)
                res.setdefault("meta", {})["budget_projection"] = proj
                _save(res, path)
                print(f"[pr08] budget projection from first cell: {proj['projected_min']} min "
                      f"for {n_cells_total} cells"
                      f"{'  *** ' + proj['note'] if proj['warn'] else ''}", flush=True)
            print(f"[pr08] {arm} seed {seed}: A_preB={rec['bpc_A_preB']:.3f} "
                  f"A_postC={rec['bpc_A_postC']:.3f} B_postB={rec['bpc_B_postB']:.3f} "
                  f"B_postC={rec['bpc_B_postC']:.3f} Cret_postC={rec['bpc_Cret_postC']:.3f} "
                  f"wall={rec['wall_s']}s", flush=True)
        if not smoke and trunk_lr_c is not None and arm in ("FROZEN-TRUNK", "FROZEN-CHECKPOINT"):
            # PR-2026-09-03-11 fail-loud canary: a frozen arm has NO C backbone step, so the
            # lever cannot touch it — bit-identity vs the (read-only) PR-08 powered ledger
            # gates every arm that follows. HONEST ORDERING NOTE: ARMS runs PRIM-LM before the
            # FIRST gate, so if a canary fails, the already-run PRIM cells are
            # quarantine-suspect (the run aborts loudly either way).
            res[f"canary.{arm}"] = _pr11_canary(res, seeds, arm)
            _save(res, path)

    # ---------------- RETENTION (docs/RETENTION.md): archive BEFORE any verdict ----------------
    raw_archive = archive_run(res, label=f"prizma-lm-{REGISTRY_ID}")
    res.setdefault("meta", {})["raw_archive"] = raw_archive

    # ---------------- the frozen §4 verdict (claim mode) / plumbing report (smoke) -------------
    report = {"registry": REGISTRY_ID, "smoke": bool(smoke), "arms": list(ARMS),
              "claim_seeds": list(seeds), "lrs": dict(lrs),
              "pinned_slices": res["meta"]["pinned_slices"],
              "c_range_repair": C_RANGE_REPAIR_NOTE,
              "c_retention_prose_note": C_RET_PROSE_NOTE, "raw_archive": raw_archive,
              "cells": {k: res[k] for k in sorted(res) if k.startswith("claim.")}}
    if powered_cpu:
        report["powered_cpu"] = True
    if trunk_lr_c is not None:
        report["trunk_lr_c"] = trunk_lr_c         # PR-2026-09-03-11 lever (recorded when set)
    if not smoke:
        prim = [res[f"claim.PRIM-LM.s{s}"] for s in seeds]
        frozen_ck = [res[f"claim.FROZEN-CHECKPOINT.s{s}"] for s in seeds]
        forced = [res[f"claim.FORCED-RECRUIT.s{s}"] for s in seeds]
        extra_b4 = {arm: [res[f"claim.{arm}.s{s}"] for s in seeds]
                    for arm in ("FROZEN-TRUNK", "SHARED-HEAD")}
        verdict = claim_verdict(prim, frozen_ck, forced, extra_b4=extra_b4, alpha=ALPHA)
        report["verdict"] = verdict
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
    if not smoke:
        v = report["verdict"]
        for bar in ("B1", "B2"):
            b = v[bar]
            center = b["mean"] if bar == "B1" else b["delta"]
            print(f"    {bar}: mean={center:.4f} CI=[{b['ci'][0]:.4f}, {b['ci'][1]:.4f}] "
                  f"p_raw={b['p_raw']:.4f} p_holm={b['p_holm']:.4f} -> {b['status']}", flush=True)
        b3 = v["B3"]
        print(f"    B3: frac_mean={b3['frac_mean']:.3f} (bar >= {b3['frac_bar']}) "
              f"forced_cost={b3['forced_cost']:+.4f} (bar <= {b3['forced_cost_bar']}) "
              f"-> {b3['status']}", flush=True)
        for arm, d in v["B4"].items():
            print(f"    B4 {arm}: B-eval degradation post-C mean {d['mean']:+.4f} "
                  f"(reported, not gated)", flush=True)
        print(f"  OUTCOME: {v['outcome']}", flush=True)
        print(f"  {v['verdict']}", flush=True)
        for br in v["branches"]:
            print(f"  branch: {br}", flush=True)
    print(f"  ledger: {path}", flush=True)
    print(f"  raw archive: {report.get('raw_archive')}", flush=True)
    if smoke:
        print("  [SMOKE] numbers are plumbing-only and MEANINGLESS — do NOT cite.", flush=True)
        print(f"  [SMOKE] C-range repair pinned: C = text8[{C_OFFSET}, {C_OFFSET + C_CHARS}) "
              f"(the doc's literal range would corrupt B1).", flush=True)
    print("=" * 78, flush=True)


def _build_parser():
    """Argparse guard (recall_gate pattern): ONLY --smoke / --powered / --out /
    --force-smoke-path; an unknown or typo'd flag exits non-zero BEFORE anything runs. Exactly
    one mode is required."""
    p = argparse.ArgumentParser(
        prog="prizma_lm_claim",
        description="PR-2026-09-03-08 claim runner (the Prizma-LM flagship bar). --smoke = tiny "
                    "CPU plumbing run; --powered = the frozen A100 claim campaign (requires "
                    "CUDA); --powered-cpu = the doc section-6 CPU-feasible fallback "
                    "(protocol-identical claim campaign on an explicitly authorized CPU box). "
                    "An unknown flag is rejected without launching anything.")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="tiny plumbing-only run (CPU, minutes)")
    mode.add_argument("--powered", action="store_true",
                      help="the claim campaign (refuses to start without a CUDA device)")
    mode.add_argument("--powered-cpu", action="store_true",
                      help="the claim campaign via the doc section-6 CPU-feasible fallback "
                           "(identical protocol and ledger; runs WITHOUT a CUDA device; doc "
                           "addendum 2026-09-08 #2)")
    p.add_argument("--out", default=None,
                   help="explicit results JSON path (overrides the default)")
    p.add_argument("--force-smoke-path", action="store_true",
                   help="let a --smoke run write the powered ledger it was pointed at "
                        "(default: REFUSED — separate ledgers)")
    p.add_argument("--trunk-lr-c", type=float, default=None,
                   help="PR-2026-09-03-11 guarded lever: the block-C BACKBONE lr in the arms "
                        "that train a backbone on C (the registered repaired-flagship dose is "
                        "7.5e-4 = 3e-3 x 0.25). Default None = byte-identical PR-08 behavior "
                        "(no audit keys, no canaries).")
    p.add_argument("--ledger-dir", default=None,
                   help="ledger subdirectory under $PRIZMA_RESULTS (default None = the PR-08 "
                        "default dir; the PR-11 run uses prizma_lm_PR-2026-09-03-11 so the "
                        "PR-08 ledger is never written)")
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(argv)   # SystemExit non-zero on unknown args: nothing runs
    run(smoke=args.smoke, results_path=args.out, force_smoke_path=args.force_smoke_path,
        powered_cpu=args.powered_cpu, trunk_lr_c=args.trunk_lr_c, ledger_dir=args.ledger_dir)


if __name__ == "__main__":
    main()
