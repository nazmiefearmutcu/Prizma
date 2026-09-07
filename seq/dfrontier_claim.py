"""
PR-2026-09-03-02 CLAIM RUNNER — the registered D-frontier grid (crosstalk capacity law).

Executes the FROZEN pre-registration `docs/crosstalk_capacity_law.md` (§4 definitions + §5
fit -> FREEZE -> adjudicate protocol + kill conditions K1-K4) VERBATIM when a GPU session runs it.
One script, two modes:

  python seq/dfrontier_claim.py --smoke      # tiny CPU plumbing run -> the SMOKE ledger (~1 min)
  python seq/dfrontier_claim.py --powered    # the A100 registered grid -> the POWERED ledger
                                             # (refuses to start without a CUDA device)

WHAT IS FROZEN (capacity-law doc §4-§5 — the constants below ARE that text, not choices):
  arms  = primary `quad2` @ d_phi=256 (feat_n2=224); secondary (non-binding) `quad2_lowrank`
          @ d_phi=137 (feat_rank=0 -> r=14); sanity arm `none` @ d_phi=32. Buffers are seeded
          inside PrizmaSeqConfig exactly as the probe (seed 1234), so run keys and probe keys are
          the same map.
  scale = d64 L2 H2. NOT a free choice: the frozen d_phi values (256/137/32) are d_h + map width
          with d_h = d_model/n_heads = 32, which pins d_model=64, n_heads=2; n_layers=2 is the B1
          family's smallest config (gpu_bench phase-1 "d64L2H2"). The feature maps add ZERO
          trainable parameters, so the doc's "matched-params config" holds by construction
          (asserted at runtime); the doc's "shared-lr policy" is the B1 family's fixed GENWARM
          lr=1e-3 (NO per-arm sweep — a sweep would be a protocol change).
  tasks = the P3 D-frontier family on the existing MQAR harness:
          MixedMQAR(vocab=max(64, 4*D), max_pairs=D, num_queries=128, gap=0, min_pairs=1)
          (dense queries, identical query count across rungs).
  grid  = fit rungs D in {16, 32, 64}; adjudication rungs D in {96, 128, 192, 256}
          (union grid [16, 32, 64, 96, 128, 192, 256]; one grid step = adjacent entries).
  seeds = {0, 1, 2} per arm per rung, identical across arms and rungs, never substituted.
          K3's fresh-seed re-run uses {10, 11, 12} (pre-pinned here, disjoint from claim seeds).
  solve = rung SOLVED iff >= 2/3 seeds reach eval accuracy >= 0.90 (repo precedent: B1
          "3/3, 0.997"; best_acc on the frozen eval, the gpu_bench precedent).
  cap   = 80000 steps at EVERY rung, batch 64, eval_every 2000, GENWARM, plateau early-stop
          (TrainConfig defaults). The doc §4 pins "identical budgets across rungs"; the B1 budget
          (gpu_bench phase-1/2 cap) is 80000. DISCLOSED DEPARTURE FROM gpu_bench PHASE-3 (not from
          the doc): phase3 used 60000 (D<=64) / 90000 (D>64) — that split violates the doc's
          identical-budgets constraint, so the doc wins and one budget is used everywhere.
  sigma2_law per arm = sqrt(sigma2_signed_fluct^2 + mu_signed^2) from the COMMITTED instrument
          artifact results/feat_map_probe.json; the runner refuses to start if the artifact's
          values drift from the doc §4 frozen constants {none 0.17692, quad2 0.14591,
          quad2_lowrank 0.15829} (instrument drift = STOP, never a silent re-fit).

THE PROTOCOL (doc §5, verbatim mechanics):
  Step 1  fit eps ONCE at the fit tier: per arm D*_fit = largest fit rung SOLVED;
          eps_hat(arm) = sqrt((D*_fit - 1) * sigma2_law(arm)^2).
  Step 2  FREEZE predictions BEFORE any adjudication rung runs. The doc's freeze is "a follow-up
          commit to this file"; that file is a binding READ-ONLY protocol document, so the freeze
          is implemented as the LEDGER RECORD written here: res["frozen_predictions"] (eps_hat
          per arm, N*(arm) = min(1 + eps_hat^2/sigma2_law^2, d_phi), the per-rung predicted
          solve/not-solve for the adjudication rungs, the fit cells behind them, a timestamp and
          the kill-condition evaluation) is written and saved BEFORE the first adjudication cell.
          A resumed run REUSES the frozen record and REFUSES (SystemExit) if the ledger's fit
          cells no longer reproduce it. The maintainer can commit the doc pointer to this ledger
          record; the runner's own invariant (no adjudication cell without a frozen record) makes
          the doc's "adjudication before freeze = protocol violation" mechanically impossible.
  Step 3  adjudicate: 3 arms x 4 rungs x 3 seeds = 36 runs on the same harness.
  PASS bar (primary quad2, exact): observed per-rung solve matches the frozen prediction on
          >= 3 of 4 adjudication rungs (equivalent formulation, reported: the observed
          solve-transition within +/-1 grid step of the predicted one). Secondary arms are scored
          under the same bar but do NOT gate the verdict. `none` doubles as the doc's sanity
          check (rank cap 32 < 96 -> predicted to fail ALL four rungs; any `none` solve at a
          rung >= 96 is flagged as falsifying something deeper than this law).
  KILL CONDITIONS (doc §5, any one fires):
          K1  eps_hat(primary) outside [0.25, 4.0] -> DEMOTED, no adjudication run booked.
          K2  primary solves ALL fit rungs or fails ALL -> unidentifiable fit -> INCOMPLETE
              (never counted as PASS), no adjudication.
          K3  non-monotone frontier (arm solves D_k but fails D_j < D_k) -> ONE re-run of the
              failed lower rung with the fresh seeds {10,11,12}; persists -> INCOMPLETE +
              investigation. In --powered a K1/K2 outcome STOPS before adjudication; in --smoke
              K1/K2 are evaluated + recorded but do NOT stop the plumbing run (disclosed; smoke
              numbers are meaningless, and the point is to exercise freeze -> adjudicate ->
              verdict end-to-end).
          K4  primary-arm agreement < 3 of 4 -> the law is FALSIFIED as the predictive account.
  Also recorded: the doc's eps consistency check |eps_hat(quad2)/eps_hat(lowrank) - 1| (per-arm
  eps_hat differing by > 2x demotes the "single task-level constant" reading to per-map, reported
  honestly — non-gating), and a per-fit-rung implied-eps table (eps_hat is identified only at the
  transition; the report shows its rung-dependence — report-only, no extra runs).
  Statistics note: this protocol's decision rule is a pre-registered AGREEMENT COUNT, not a
  Welch test, so seq.stats.t_isf is not consulted here; its upper-tail p in (0, 0.5] convention
  is pinned by tests/test_surprise_gate.py::test_mde_checksum_pins_upper_tail_convention.

LEDGER SEPARATION + RETENTION (the surprise_claim/recall_gate pattern, mirrored):
  smoke   -> results/dfrontier_PR-2026-09-03-02/smoke.json
  powered -> results/dfrontier_PR-2026-09-03-02/powered.json
  A --smoke run pointed at the powered ledger is REFUSED (_resolve_results_path) unless
  --force-smoke-path. Every training cell streams crash-safe (json -> .tmp -> os.replace) with
  (cellkey, config-fingerprint) resume keyed via seq.gpu_harness; mixed-configuration cells are
  refused at aggregation time. Raw records are archived verbatim
  (seq.recall_gate.archive_run) BEFORE any verdict; the verdict references the archive path.

COST HONESTY (doc §5): 63 runs total (27 fit + 36 adjudication) ~ 12-18 A100-h at the B1 budget;
the fit stage alone (~27 runs) is the cheap decisive filter — K1/K2 can kill the law for ~3-5
GPU-h without touching the adjudication rungs.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

# Light imports only at module top: the pure layer (paths/refusal, law math, kill conditions,
# verdict) must be importable + unit-testable without torch or training (recall_gate discipline).
try:
    from .stats import solve_rate  # noqa: F401  (re-exported for symmetry with the sibling runner)
except ImportError:                                   # run as a bare script: bootstrap sys.path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from seq.stats import solve_rate  # noqa: F401


# ================================================================== frozen protocol constants ====
REGISTRY_ID = "PR-2026-09-03-02"
LEDDIR = "dfrontier_PR-2026-09-03-02"   # under $PRIZMA_RESULTS (default ./results)
SMOKE_BASENAME = "smoke.json"
POWERED_BASENAME = "powered.json"

PRIMARY_ARM = "quad2"
ARMS = ("quad2", "quad2_lowrank", "none")
D_PHI = {"quad2": 256, "quad2_lowrank": 137, "none": 32}     # doc §4 rank caps (frozen)
ARM_PRIZMA_KW = {
    "quad2": dict(feat_map="quad2", feat_n2=224),            # d_phi = 32 + 224 = 256
    "quad2_lowrank": dict(feat_map="quad2_lowrank"),         # feat_rank=0 -> r=14 -> d_phi=137
    "none": dict(feat_map="none"),                           # d_phi = d_h = 32
}
SCALE = (64, 2, 2)   # (d_model, n_layers, n_heads) -> d_h = 32: pinned by the frozen d_phi values

FIT_RUNGS = (16, 32, 64)
ADJ_RUNGS = (96, 128, 192, 256)
GRID = FIT_RUNGS + ADJ_RUNGS                    # one grid step = adjacent entries

CLAIM_SEEDS = (0, 1, 2)     # doc §4: "Seeds {0,1,2} minimum, identical across arms and rungs"
RERUN_SEEDS = (10, 11, 12)  # K3's ONE fresh-seed re-run, pre-pinned, disjoint from claim seeds

STEP_CAP = 80000            # the B1 budget (gpu_bench phase-1/2), IDENTICAL across rungs (doc §4)
BATCH_SIZE = 64
EVAL_EVERY = 2000
NUM_QUERIES = 128           # dense queries, identical across rungs (the B1/P3 task family)
SOLVE_THRESH = 0.90
SOLVE_FRAC = 2.0 / 3.0      # rung SOLVED iff >= 2/3 of the rung's seeds reach SOLVE_THRESH

K1_LO, K1_HI = 0.25, 4.0    # K1: eps_hat(primary) outside this range -> DEMOTED
PASS_MIN_AGREE = 3          # of the 4 adjudication rungs (primary arm)
EPS_CONSISTENCY_DEMOTE_RATIO = 2.0   # per-arm eps_hat differing by > 2x -> demote to per-map

# doc §4 frozen sigma2_law constants (the runner verifies the committed artifact against these)
FROZEN_SIGMA2_LAW = {"none": 0.17692, "quad2": 0.14591, "quad2_lowrank": 0.15829}
SIGMA2_TOLERANCE = 2e-4     # the doc constants are 5-dp roundings of the artifact values

# smoke shrinkage (shapes/steps/seeds shrink; the CRITERION FORMULA does not — need = ceil(2/3 n))
SMOKE_FIT_RUNGS = (8,)
SMOKE_ADJ_RUNGS = (12,)
SMOKE_STEP_CAP = 500
SMOKE_BATCH_SIZE = 16
SMOKE_EVAL_EVERY = 100
SMOKE_SEEDS = (0,)
SMOKE_NUM_QUERIES = 16
SMOKE_RECIPE = dict(warmup=200, warmup_frac=0.0, min_lr_frac=0.1)
GENWARM_RECIPE = dict(lr=1e-3, warmup=2000, warmup_frac=0.0, min_lr_frac=0.1)  # B1 shared-lr


# ================================================== pure: instrument artifact (sigma2_law) ======
def probe_artifact_path() -> str:
    """The COMMITTED instrument artifact (feat_map_probe.py's output), located from the repo root —
    NOT under $PRIZMA_RESULTS: it is a frozen input to the protocol, like the protocol doc."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                        "feat_map_probe.json")


def sigma2_law_from_entry(entry: dict) -> float:
    """The doc §4 law quantity: sigma2_law = sqrt(sigma2_signed_fluct^2 + mu_signed^2) — the SNR
    second moment E[c^2]^1/2. (The |cos|-std `sigma2` is a reported instrument metric, NOT this.)"""
    return math.sqrt(float(entry["sigma2_signed_fluct"]) ** 2 + float(entry["mu_signed"]) ** 2)


def load_sigma2_law(probe_path=None, *, frozen=None, tol=SIGMA2_TOLERANCE) -> dict:
    """Load per-arm sigma2_law from the committed artifact and REFUSE (SystemExit) if it drifts
    from the doc §4 frozen constants. A drifting instrument invalidates every eps fit — stopping
    is the only honest move (no silent re-fit, no silent proceed)."""
    frozen = FROZEN_SIGMA2_LAW if frozen is None else frozen
    path = probe_path if probe_path is not None else probe_artifact_path()
    if not os.path.exists(path):
        raise SystemExit(f"refusing: the committed instrument artifact {path} is missing — the law's "
                         f"sigma2_law inputs are frozen inputs to this protocol (regenerate it with "
                         f"`python feat_map_probe.py` and confirm it reproduces the doc §4 values).")
    import json
    with open(path, "r") as f:
        probe = json.load(f)
    out = {}
    for arm in ARMS:
        if arm not in probe:
            raise SystemExit(f"refusing: instrument artifact {path} has no entry for arm {arm!r}")
        val = sigma2_law_from_entry(probe[arm])
        if abs(val - frozen[arm]) > tol:
            raise SystemExit(
                f"refusing: sigma2_law({arm}) = {val:.6f} in {path} drifted from the doc §4 frozen "
                f"constant {frozen[arm]} (tolerance {tol}). The instrument moved — the frozen eps "
                f"fit is invalid. STOP: re-freeze the protocol constants first; do not proceed.")
        out[arm] = val
    return out


# ==================================================================== pure: the law math ========
def _import_feat_map_probe():
    """feat_map_probe.py lives at the repo ROOT (not in the seq/ package): importable only with the
    repo root on sys.path (the experiments/e_fit_probe.py pattern)."""
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    import feat_map_probe
    return feat_map_probe


def n_star_of(eps: float, sigma2_law: float, d_phi: int) -> float:
    """N*(eps; phi) = min(1 + eps^2/sigma2_law^2, d_phi) — via feat_map_probe.n_star_law so the
    runner and the instrument share ONE implementation of the candidate law."""
    return float(_import_feat_map_probe().n_star_law(sigma2_law, d_phi, eps=float(eps)))


def rung_solved(accs, thresh=SOLVE_THRESH, n_seeds=None) -> bool:
    """The frozen solve criterion: rung SOLVED iff >= 2/3 of the rung's seeds reach eval accuracy
    >= 0.90. n_seeds defaults to len(accs); the 2/3 FORMULA is the protocol (3 seeds -> 2)."""
    accs = list(accs)
    n = len(accs) if n_seeds is None else n_seeds
    need = max(1, math.ceil(SOLVE_FRAC * n))
    return sum(1 for a in accs if a >= thresh) >= need


def largest_solved_fit_rung(solved_by_rung: dict, rungs=FIT_RUNGS):
    """D*_fit = largest fit rung SOLVED (None if the arm fails the whole fit tier)."""
    solved = [D for D in rungs if solved_by_rung.get(D)]
    return max(solved) if solved else None


def fit_epsilon(d_star, sigma2_law: float):
    """eps_hat = sqrt((D*_fit - 1) * sigma2_law^2) — the doc §5 Step-1 fit (exact formula).
    None when D*_fit is None (unidentifiable fit -> K2)."""
    if d_star is None:
        return None
    return float(sigma2_law) * math.sqrt(d_star - 1)


def predicted_solves(eps: float, sigma2_law: float, d_phi: int, rungs=ADJ_RUNGS) -> dict:
    """The frozen per-rung prediction: rung D predicted SOLVED iff D <= N*(eps; phi)."""
    ns = n_star_of(eps, sigma2_law, d_phi)
    return {int(D): bool(D <= ns) for D in rungs}


def evaluate_kill(solved_fit: dict, d_star: dict, eps_hat: dict, fit_rungs=FIT_RUNGS,
                  *, primary=PRIMARY_ARM) -> dict:
    """K1/K2 (doc §5). solved_fit: {(arm, D) -> bool}. K2 (unidentifiable fit) is checked before
    K1 (range). Outcome ordering matches the doc: K2 -> INCOMPLETE (never a PASS); K1 -> DEMOTED
    with no adjudication booked."""
    solvable = [bool(solved_fit.get((primary, D))) for D in fit_rungs]
    k2 = all(solvable) or not any(solvable)
    if k2:
        which = "solves ALL of" if all(solvable) else "fails ALL of"
        return {"k2_fired": True, "k1_fired": False, "outcome": "INCOMPLETE_K2",
                "reason": (f"K2 (unidentifiable fit): the primary arm {which} the fit rungs "
                           f"{list(fit_rungs)} — eps_hat is not identified. Outcome INCOMPLETE "
                           f"(doc §5: never counted as PASS); no adjudication run is booked.")}
    eps = eps_hat.get(primary)
    k1 = eps is not None and not (K1_LO <= eps <= K1_HI)
    if k1:
        return {"k2_fired": False, "k1_fired": True, "outcome": "DEMOTED_K1",
                "reason": (f"K1: eps_hat(primary) = {eps:.4f} outside the pre-registered O(1) range "
                           f"[{K1_LO}, {K1_HI}] — the law has no predictive content beyond its fit "
                           f"rung. Law DEMOTED; no adjudication run is booked.")}
    return {"k2_fired": False, "k1_fired": False, "outcome": None, "reason": "fit tier alive"}


def eps_consistency(eps_hat: dict, *, a=PRIMARY_ARM, b="quad2_lowrank") -> dict:
    """The doc §5 consistency check: |eps_hat(a)/eps_hat(b) - 1|; per-arm eps_hat differing by
    > 2x (max/min) demotes the 'single task-level constant' reading to per-map (recorded
    honestly, non-gating — predictions are per-arm either way)."""
    ea, eb = eps_hat.get(a), eps_hat.get(b)
    if ea is None or eb is None or eb == 0 or ea == 0:
        return {"ratio": None, "abs_gap": None, "demote_to_per_map": True,
                "note": "eps_hat unidentifiable (or zero) for an arm — 'single constant' reading "
                        "not evaluable; treated as per-map"}
    ratio = abs(ea) / abs(eb)
    spread = max(abs(ea), abs(eb)) / min(abs(ea), abs(eb))
    demote = spread > EPS_CONSISTENCY_DEMOTE_RATIO
    return {"ratio": ratio, "abs_gap": abs(ratio - 1.0), "demote_to_per_map": bool(demote),
            "note": (f"per-arm eps_hat differ by {spread:.2f}x (> {EPS_CONSISTENCY_DEMOTE_RATIO}x): "
                     f"'single task-level tolerance' DEMOTED to per-map (doc §5)" if demote else
                     "per-arm eps_hat within 2x — consistent with a single O(1) constant (doc §5 reading)")}


def implied_eps_by_fit_rung(fit_rungs=FIT_RUNGS, sigma2_law=None, arm=PRIMARY_ARM) -> list:
    """Report-only 'eps-invariance across fit rungs' table: eps implied if EACH fit rung were the
    transition (eps(D) = sigma2_law*sqrt(D-1)). eps_hat is identified ONLY at D*_fit; this makes
    the fit's rung-dependence visible instead of hidden (raw-first retention discipline)."""
    if sigma2_law is None:
        return []
    return [{"D": int(D), "eps_if_transition": float(sigma2_law) * math.sqrt(D - 1)}
            for D in fit_rungs]


# ======================================================== pure: frontier + adjudication ========
def monotone_violations(solved_by_arm_rung: dict, rungs=GRID) -> list:
    """K3 detector: [(arm, D_failed, D_solved)] for every arm that SOLVES a higher rung D_solved
    but FAILS a lower rung D_failed (the law presumes a monotone solve frontier in D)."""
    out = []
    for arm in ARMS:
        for d_hi in rungs:
            if not solved_by_arm_rung.get((arm, d_hi)):
                continue
            for d_lo in rungs:
                if d_lo < d_hi and not solved_by_arm_rung.get((arm, d_lo)):
                    out.append((arm, d_lo, d_hi))
    return out


def transition_rung(solved_by_rung: dict, rungs) -> int | None:
    """Smallest rung NOT solved (None if every rung in `rungs` is solved)."""
    for D in rungs:
        if not solved_by_rung.get(D):
            return int(D)
    return None


def grid_distance(d_a, d_b, grid=ADJ_RUNGS):
    """Distance in grid steps (adjacent-entry = 1) between two rungs; None when either side has no
    transition (all-solved) — reported as 'n/a' rather than a fake number. Both transitions are
    members of the adjudication rung set, so that set is the measuring stick."""
    if d_a is None or d_b is None:
        return None
    return abs(list(grid).index(int(d_a)) - list(grid).index(int(d_b)))


def adjudicate_arm(observed_by_rung: dict, predicted_by_rung, rungs) -> dict:
    """Per-arm adjudication: agreement count vs the FROZEN per-rung prediction, the +/-1-grid-step
    transition report (the doc's equivalent formulation), and the pass flag at the frozen bar."""
    if predicted_by_rung is None:
        return {"identified": False, "agreement": None, "pass": None,
                "note": "eps_hat unidentifiable for this arm — no frozen prediction to test"}
    obs = {int(D): bool(observed_by_rung.get(D)) for D in rungs}
    agree = [int(D) for D in rungs if obs[int(D)] == bool(predicted_by_rung.get(int(D)))]
    bar = PASS_MIN_AGREE if list(rungs) == list(ADJ_RUNGS) else len(rungs)  # smoke: all rungs must agree
    t_pred = transition_rung(predicted_by_rung, rungs)
    t_obs = transition_rung(obs, rungs)
    dist = grid_distance(t_obs, t_pred, rungs)
    return {
        "identified": True,
        "observed": obs,
        "predicted": {int(D): bool(predicted_by_rung.get(int(D))) for D in rungs},
        "agreement_count": len(agree),
        "agreement_rungs": agree,
        "bar": bar,
        "pass": len(agree) >= bar,
        "predicted_transition": t_pred,
        "observed_transition": t_obs,
        "transition_distance_grid_steps": dist,
        "transition_within_plus_minus_1": (dist is not None and dist <= 1) or (t_pred is None and t_obs is None),
    }


def dfrontier_verdict(observed_solved: dict, frozen: dict, adj_rungs=ADJ_RUNGS, *,
                      k3_persisted: bool = False, primary=PRIMARY_ARM, smoke: bool = False) -> dict:
    """The frozen PASS bar (doc §5): primary-arm per-rung agreement >= 3 of 4 adjudication rungs.
    K3-persisted -> INCOMPLETE (integrity failure) regardless of agreement. Secondary arms are
    scored under the same bar but do NOT gate; `none` is the doc's sanity arm (predicted to fail
    every rung >= 96 by its rank cap; any such solve falsifies something deeper — flagged,
    non-gating). Order: K3 > primary bar (K4)."""
    per_arm = {arm: adjudicate_arm({D: observed_solved.get((arm, D)) for D in adj_rungs},
                                   (frozen.get("predicted_solved", {}).get(arm) if frozen else None),
                                   adj_rungs)
               for arm in ARMS}
    p = per_arm[primary]

    none_rungs = [int(D) for D in adj_rungs if int(D) >= 96]
    none_solves = [int(D) for D in none_rungs if observed_solved.get(("none", D))]
    sanity = {"applicable": bool(none_rungs), "none_solved_rungs_ge_96": none_solves,
              "deeper_falsification_flag": bool(none_solves),
              "note": ("none solved a rung >= 96 despite its rank cap 32 — falsifies something "
                       "deeper than this law (doc §5); flagged, non-gating" if none_solves else
                       "none failed every >=96 rung (or none applicable in smoke): sanity clean")}

    if k3_persisted:
        outcome = "INCOMPLETE_K3"
        verdict = ("INCOMPLETE — non-monotone frontier persisted after the fresh-seed re-run "
                   "(K3): protocol integrity failure + investigation before any reading "
                   "(the law presumes a monotone solve frontier in D). Never counted as PASS.")
    elif not p.get("identified"):
        outcome = "INCOMPLETE_K2"
        verdict = ("INCOMPLETE — the primary arm's eps_hat was unidentifiable at the fit tier "
                   "(K2); there is no frozen prediction to adjudicate. Never counted as PASS.")
    elif p["pass"]:
        outcome = "PASS"
        verdict = (f"PASS — the law's per-rung solve prediction agrees with observation on "
                   f"{p['agreement_count']} of {len(list(adj_rungs))} adjudication rungs "
                   f"(>= {PASS_MIN_AGREE} of 4 frozen bar, primary {primary}).")
    else:
        outcome = "FALSIFIED_K4"
        verdict = (f"FALSIFIED (K4) — primary-arm agreement {p['agreement_count']} of "
                   f"{len(list(adj_rungs))} < {PASS_MIN_AGREE}: the law is falsified as the "
                   f"predictive account of the D-frontier. Flags A/B survive independently; the "
                   f"repo then states honestly that it has no predictive capacity account.")
    return {"outcome": outcome, "verdict": verdict, "smoke": bool(smoke), "primary": primary,
            "per_arm": per_arm, "none_sanity": sanity}


# ==================================================================== paths + BAR-0 refusal ======
def _results_root() -> str:
    root = os.environ.get("PRIZMA_RESULTS", os.path.join(os.path.dirname(__file__), "..", "results"))
    return os.path.abspath(root)


def _default_results_path(smoke: bool) -> str:
    return os.path.join(_results_root(), LEDDIR, SMOKE_BASENAME if smoke else POWERED_BASENAME)


def resolve_results_path(explicit=None, *, smoke: bool = False, force_smoke_path: bool = False) -> str:
    """Resolve the results path and enforce the smoke/powered file separation (the recall_gate
    refusal pattern): a --smoke run pointed at the POWERED ledger is refused (SystemExit) unless
    force_smoke_path. Smoke numbers are plumbing-only; a smoke cell inside the claim ledger is
    exactly the 2026-06-08 contamination mechanism."""
    path = explicit if explicit else _default_results_path(smoke)
    if smoke and not force_smoke_path:
        powered = _default_results_path(smoke=False)
        if os.path.normcase(os.path.abspath(path)) == os.path.normcase(os.path.abspath(powered)):
            raise SystemExit(
                f"refusing: a --smoke run was pointed at the powered ledger ({powered}). Smoke and "
                f"campaign results use separate files by default (PR-2026-09-03-02; see "
                f"results/campaign_2026-06-08/CONTAMINATION.md). Pass --out <path> to write the smoke "
                f"elsewhere, or --force-smoke-path to override this guard deliberately.")
    return path


def require_cuda(has_cuda: bool) -> None:
    """Pure guard for the --powered mode (unit-testable without a CUDA box): the registered grid is
    the A100 Colab session's job; a CPU 'powered' run would be neither powered nor the
    pre-registered environment (and would silently redefine the B1 budget's wall-clock)."""
    if not has_cuda:
        raise SystemExit(
            "refusing: --powered executes the PR-2026-09-03-02 registered D-frontier grid and "
            "requires a CUDA device (the A100 Colab session). No CUDA device is visible here. "
            "Use --smoke for the CPU plumbing run.")


# ============================================================================ arm helpers =======
def _arm_factory(arm: str):
    """(vocab, max_len) -> PrizmaSeqLM via seq.prizma_seq.prizma_seq_factory: the factory takes
    (vocab, max_len) from build_and_train's **fac_kw — passed explicitly by seq.gpu_harness.run_cell
    from the task's vocab/seq_len, exactly the experiments/e_fit_probe.py wiring."""
    from seq.prizma_seq import prizma_seq_factory
    return prizma_seq_factory(d_model=SCALE[0], n_layers=SCALE[1], n_heads=SCALE[2],
                              **ARM_PRIZMA_KW[arm])


def _task_fac(D: int, *, smoke: bool):
    """Zero-arg task factory: the P3 D-frontier task family (dense queries). Smoke shrinks the
    query count, NOT the family (smoke vocab still max(64, 4*D) — for D in {8,12} that is 64)."""
    from seq.tasks import MixedMQAR
    q = SMOKE_NUM_QUERIES if smoke else NUM_QUERIES
    V = max(64, 4 * D)
    return lambda: MixedMQAR(vocab=V, max_pairs=D, num_queries=q, gap=0, min_pairs=1)


def assert_arm_geometry() -> dict:
    """Runtime integrity: every arm's expanded key dim equals its frozen d_phi, and all arms have
    IDENTICAL trainable parameter counts (the doc's matched-params config; feature maps add zero
    params — a violation here means the arms are not the protocol's arms)."""
    from seq.prizma_seq import PrizmaSeqConfig, PrizmaSeqLM
    from seq.common import param_count
    counts = {}
    for arm in ARMS:
        cfg = PrizmaSeqConfig(vocab=64, d_model=SCALE[0], n_layers=SCALE[1], n_heads=SCALE[2],
                              max_len=64, **ARM_PRIZMA_KW[arm])
        assert cfg.d_phi == D_PHI[arm], \
            f"{arm}: model d_phi {cfg.d_phi} != frozen {D_PHI[arm]} — arm geometry broken"
        counts[arm] = param_count(PrizmaSeqLM(cfg))
    assert len(set(counts.values())) == 1, f"matched-params violated across arms: {counts}"
    return counts


def _fingerprint(payload: dict) -> str:
    from seq.gpu_harness import config_fingerprint
    return config_fingerprint({"registry": REGISTRY_ID, **payload})


def _cell_cfgsig(leg: str, arm: str, D: int, seed: int, *, cap, batch_size, eval_every, recipe,
                 num_queries, grid_tag: str) -> str:
    return _fingerprint({"leg": leg, "arm": arm, "D": int(D), "seed": int(seed),
                         "scale": list(SCALE), "d_phi": D_PHI[arm],
                         "feat_kw": ARM_PRIZMA_KW[arm], "cap": cap, "batch": batch_size,
                         "eval_every": eval_every, "recipe": recipe, "num_queries": num_queries,
                         "solve": [SOLVE_THRESH, SOLVE_FRAC], "grid_tag": grid_tag})


def _train_stage(res, leg: str, rungs, seeds, cfg, device, *, cap, batch_size, eval_every,
                 recipe, num_queries, grid_tag, out_path):
    """One stage (fit / adj / k3rerun) of the grid: arms x rungs x seeds via seq.gpu_harness.run_cell
    (seed-pinned init via build_and_train, crash-safe ledger, fingerprint resume). Returns
    {(arm, D) -> [per-seed best_acc]} and asserts, per (leg, D), that no cell resumed at a foreign
    fingerprint and that all arms train the same parameter count (matched-params)."""
    from seq.gpu_harness import run_cell
    accs, params = {}, {}
    for arm in ARMS:
        fac = _arm_factory(arm)
        for D in rungs:
            cell_accs = []
            for seed in seeds:
                cellkey = f"{leg}.{arm}.D{D}.s{seed}"
                cfgsig = _cell_cfgsig(leg, arm, D, seed, cap=cap, batch_size=batch_size,
                                      eval_every=eval_every, recipe=recipe,
                                      num_queries=num_queries, grid_tag=grid_tag)
                rec = run_cell(res, cellkey, fac, _task_fac(D, smoke=(grid_tag == "smoke")),
                               cfg, device, seed=seed, out_path=out_path, cfgsig=cfgsig)
                if rec.get("cfgsig") != cfgsig:
                    raise AssertionError(
                        f"cell {cellkey} resumed at a different config ({rec.get('cfgsig')} != "
                        f"{cfgsig}) — refusing to aggregate mixed configurations")
                cell_accs.append(rec["best"])
                params[(arm, D)] = rec["params"]
            accs[(arm, D)] = cell_accs
            rates = [f"{a:.3f}" for a in cell_accs]
            print(f"   [{leg}] {arm:<15} D={D:<4} accs={rates} "
                  f"solved={rung_solved(cell_accs, n_seeds=len(seeds))}", flush=True)
    for D in rungs:
        counts = {arm: params[(arm, D)] for arm in ARMS}
        assert len(set(counts.values())) == 1, \
            f"{leg} D={D}: arms disagree on parameter count {counts} — not matched-params"
    return accs


# ================================================================== freeze + verify (pure-ish) ==
def freeze_record(fit_accs: dict, solved_fit: dict, d_star: dict, eps_hat: dict, sigma2_law: dict,
                  kill: dict, consistency: dict, fit_rungs, adj_rungs, *, smoke: bool) -> dict:
    """The Step-2 FREEZE (doc §5): eps_hat per arm, N* per arm, the per-rung predicted solve for
    the ADJUDICATION rungs, and everything the prediction was derived from. Written into the
    ledger BEFORE any adjudication cell; a resumed run reuses it and never re-fits."""
    predictions, n_star = {}, {}
    for arm in ARMS:
        e = eps_hat.get(arm)
        if e is None:
            predictions[arm] = None
            n_star[arm] = None
            continue
        n_star[arm] = n_star_of(e, sigma2_law[arm], D_PHI[arm])
        predictions[arm] = predicted_solves(e, sigma2_law[arm], D_PHI[arm], adj_rungs)
    return {
        "registry": REGISTRY_ID,
        "what": "Step-2 FREEZE (docs/crosstalk_capacity_law.md §5): frozen BEFORE any adjudication "
                "rung ran. The doc's 'follow-up commit to this file' is implemented as this ledger "
                "record (the doc is a binding READ-ONLY protocol document); the runner refuses to "
                "launch adjudication cells without it and refuses to re-fit on resume.",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "smoke": bool(smoke),
        "fit_rungs": list(fit_rungs), "adj_rungs": list(adj_rungs),
        "sigma2_law": {arm: sigma2_law[arm] for arm in ARMS},
        "fit_accs": {f"{arm}.D{D}": fit_accs[(arm, D)] for arm in ARMS for D in fit_rungs},
        "solved_fit": {f"{arm}.D{D}": bool(solved_fit[(arm, D)]) for arm in ARMS for D in fit_rungs},
        "d_star_fit": {arm: d_star.get(arm) for arm in ARMS},
        "eps_hat": {arm: eps_hat.get(arm) for arm in ARMS},
        "n_star": n_star,
        "predicted_solved": predictions,
        "kill_evaluation": kill,
        "eps_consistency": consistency,
        "implied_eps_by_fit_rung": implied_eps_by_fit_rung(fit_rungs, sigma2_law.get(PRIMARY_ARM),
                                                           PRIMARY_ARM),
    }


def verify_frozen_against_ledger(res, frozen: dict, sigma2_law: dict, seeds, fit_rungs=None) -> None:
    """Resume integrity: the frozen predictions must still be derivable from the ledger's fit
    cells at the SAME protocol. Any mismatch -> SystemExit (a frozen record belongs to exactly one
    configuration; silently re-fitting or silently proceeding are both violations)."""
    fit_rungs = tuple(frozen.get("fit_rungs", [])) if fit_rungs is None else tuple(fit_rungs)
    if tuple(frozen.get("adj_rungs", [])) != tuple(ADJ_RUNGS) and not frozen.get("smoke"):
        raise SystemExit("refusing: frozen predictions target a different adjudication grid — "
                         "the protocol changed after the freeze; open a new pre-registration.")
    for arm in ARMS:
        accs = []
        for D in fit_rungs:
            for seed in seeds:
                rec = res.get(f"fit.{arm}.D{D}.s{seed}")
                if not isinstance(rec, dict) or "best" not in rec:
                    raise SystemExit(f"refusing: frozen predictions reference fit cell "
                                     f"fit.{arm}.D{D}.s{seed} which is not in this ledger.")
                accs.append(rec["best"])
    # recompute eps_hat per arm from the ledger's fit cells and require EXACT equality with the
    # frozen values (same records, same arithmetic -> bit-identical; anything else = tampering)
    for arm in ARMS:
        solved = {D: rung_solved([res[f"fit.{arm}.D{D}.s{s}"]["best"] for s in seeds], n_seeds=len(seeds))
                  for D in fit_rungs}
        ds = largest_solved_fit_rung(solved, fit_rungs)
        e = fit_epsilon(ds, sigma2_law[arm])
        frozen_e = frozen.get("eps_hat", {}).get(arm)
        if (e is None) != (frozen_e is None) or (e is not None and e != frozen_e):
            raise SystemExit(
                f"refusing: ledger fit cells give eps_hat({arm}) = {e} but the frozen record says "
                f"{frozen_e} — the freeze does not match this ledger's fit tier. Investigate; do "
                f"not adjudicate against a stale freeze.")
    if any(abs(sigma2_law[arm] - frozen.get("sigma2_law", {}).get(arm, -1.0)) > 1e-9 for arm in ARMS):
        raise SystemExit("refusing: frozen record was built at different sigma2_law values "
                         "(instrument drift) — do not adjudicate against a stale freeze.")


# ================================================================================= runner =======
def run(*, smoke: bool, results_path=None, force_smoke_path: bool = False):
    """Execute the protocol: --smoke (CPU plumbing) or --powered (the A100 registered grid)."""
    # BAR-0 FIRST: resolve + guard the results path before any heavy import or write.
    path = resolve_results_path(results_path, smoke=smoke, force_smoke_path=force_smoke_path)

    import torch
    from seq.gpu_harness import make_cfg, load_results, _save, get_device
    from seq.recall_gate import archive_run

    # The law's frozen inputs: per-arm sigma2_law from the committed instrument artifact,
    # verified against the doc §4 constants (refuses on drift, BEFORE anything runs).
    sigma2_law = load_sigma2_law()

    if smoke:
        device = torch.device("cpu")            # smoke stays deterministic + CPU-runnable
        cap, batch_size, eval_every = SMOKE_STEP_CAP, SMOKE_BATCH_SIZE, SMOKE_EVAL_EVERY
        seeds, fit_rungs, adj_rungs = SMOKE_SEEDS, SMOKE_FIT_RUNGS, SMOKE_ADJ_RUNGS
        recipe, num_queries, rerun_seeds = SMOKE_RECIPE, SMOKE_NUM_QUERIES, (10,)
        grid_tag = "smoke"
        print("=" * 78, flush=True)
        print("  *** SMOKE MODE: PLUMBING-ONLY (PR-2026-09-03-02) ***", flush=True)
        print("  Proves fit -> FREEZE -> adjudicate -> verdict + ledgers wire end-to-end on CPU.",
              flush=True)
        print("  K1/K2 are evaluated + recorded but do NOT stop a smoke run (disclosed; the", flush=True)
        print("  powered run hard-stops per doc §5). The numbers are MEANINGLESS — do NOT cite.", flush=True)
        print("=" * 78, flush=True)
    else:
        require_cuda(torch.cuda.is_available())  # refuse BEFORE anything else on a CPU-only box
        device = get_device()
        cap, batch_size, eval_every = STEP_CAP, BATCH_SIZE, EVAL_EVERY
        seeds, fit_rungs, adj_rungs = CLAIM_SEEDS, FIT_RUNGS, ADJ_RUNGS
        recipe, num_queries, rerun_seeds = GENWARM_RECIPE, NUM_QUERIES, RERUN_SEEDS
        grid_tag = "powered"

    print(f"device={device} registry={REGISTRY_ID} smoke={smoke} results={path}", flush=True)
    print(f"sigma2_law (frozen instrument): " +
          ", ".join(f"{a}={sigma2_law[a]:.5f}" for a in ARMS), flush=True)
    res = load_results(path)
    res["meta"] = {"registry": REGISTRY_ID, "smoke": bool(smoke), "scale": list(SCALE),
                   "arms": list(ARMS), "d_phi": dict(D_PHI), "primary": PRIMARY_ARM,
                   "fit_rungs": list(fit_rungs), "adj_rungs": list(adj_rungs),
                   "grid": list(fit_rungs) + list(adj_rungs), "claim_seeds": list(seeds),
                   "cap": cap, "batch_size": batch_size, "eval_every": eval_every,
                   "recipe": recipe, "num_queries": num_queries,
                   "solve": [SOLVE_THRESH, SOLVE_FRAC],
                   "sigma2_law": sigma2_law,
                   "lane": "CLAIM (frozen pre-registration docs/crosstalk_capacity_law.md §4-§5)"
                           if not smoke else "SMOKE (plumbing only)"}
    _save(res, path)

    param_counts = assert_arm_geometry()
    res["meta"]["matched_params"] = param_counts
    print(f"matched-params check (trainable params, identical across arms): {param_counts}", flush=True)

    cfg = make_cfg(cap, batch_size=batch_size, eval_every=eval_every, log=False, **recipe)

    # ---- Step 1: fit eps ONCE at the fit tier (3 arms x |fit_rungs| x |seeds| runs) --------------
    print(f"\n-- FIT TIER: arms x D in {list(fit_rungs)} x seeds {list(seeds)} --", flush=True)
    fit_accs = _train_stage(res, "fit", fit_rungs, seeds, cfg, device, cap=cap,
                            batch_size=batch_size, eval_every=eval_every, recipe=recipe,
                            num_queries=num_queries, grid_tag=grid_tag, out_path=path)
    solved_fit = {(arm, D): rung_solved(v, n_seeds=len(seeds)) for (arm, D), v in fit_accs.items()}
    d_star = {arm: largest_solved_fit_rung({D: solved_fit[(arm, D)] for D in fit_rungs}, fit_rungs)
              for arm in ARMS}
    eps_hat = {arm: fit_epsilon(d_star[arm], sigma2_law[arm]) for arm in ARMS}
    kill = evaluate_kill(solved_fit, d_star, eps_hat, fit_rungs)
    consistency = eps_consistency(eps_hat)
    for arm in ARMS:
        print(f"   [{arm}] D*_fit={d_star[arm]}  eps_hat={eps_hat[arm]}", flush=True)
    print(f"   kill evaluation: {kill['outcome']} — {kill['reason']}", flush=True)
    print(f"   eps consistency: {consistency}", flush=True)

    # ---- Step 2: FREEZE predictions BEFORE any adjudication cell (reuse on resume; never re-fit) -
    frozen = res.get("frozen_predictions")
    if frozen is None:
        frozen = freeze_record(fit_accs, solved_fit, d_star, eps_hat, sigma2_law, kill,
                               consistency, fit_rungs, adj_rungs, smoke=smoke)
        res["frozen_predictions"] = frozen
        _save(res, path)
        print("\n[FREEZE] predictions written to the ledger BEFORE any adjudication cell "
              "(doc §5 Step 2; maintained as the ledger record — see module docstring).", flush=True)
    else:
        verify_frozen_against_ledger(res, frozen, sigma2_law, seeds, fit_rungs)
        print("\n[FREEZE] reusing the ledger's existing frozen predictions (resume; re-fit "
              "refused by protocol).", flush=True)

    # ---- kill conditions K1/K2: hard-stop in --powered (doc §5); recorded-only in --smoke --------
    if kill["outcome"] is not None and not smoke:
        raw_archive = archive_run(res, label=f"dfrontier-{REGISTRY_ID}")
        res.setdefault("meta", {})["raw_archive"] = raw_archive
        res["verdict"] = {"outcome": kill["outcome"], "verdict": kill["reason"],
                          "smoke": False, "raw_archive": raw_archive}
        report = {"registry": REGISTRY_ID, "smoke": False, "grid": list(fit_rungs) + list(adj_rungs),
                  "sigma2_law": sigma2_law, "d_star_fit": d_star, "eps_hat": eps_hat,
                  "kill_evaluation": kill, "eps_consistency": consistency,
                  "frozen_predictions": frozen, "raw_archive": raw_archive,
                  "verdict": res["verdict"]}
        res["report"] = report
        _save(res, path)
        print("\n" + "=" * 78, flush=True)
        print(f"  {REGISTRY_ID} — STOPPED AT THE FIT TIER: {kill['outcome']}", flush=True)
        print(f"  {kill['reason']}", flush=True)
        print(f"  ledger: {path}\n  raw archive: {raw_archive}", flush=True)
        print("=" * 78, flush=True)
        return report

    # ---- Step 3: adjudicate (3 arms x |adj_rungs| x |seeds| runs; ~12-18 A100-h powered) ---------
    print(f"\n-- ADJUDICATION TIER: arms x D in {list(adj_rungs)} x seeds {list(seeds)} --", flush=True)
    adj_accs = _train_stage(res, "adj", adj_rungs, seeds, cfg, device, cap=cap,
                            batch_size=batch_size, eval_every=eval_every, recipe=recipe,
                            num_queries=num_queries, grid_tag=grid_tag, out_path=path)
    observed_solved = dict(solved_fit)
    observed_solved.update({k: rung_solved(v, n_seeds=len(seeds)) for k, v in adj_accs.items()})

    # ---- K3: monotone frontier over every observed rung; ONE fresh-seed re-run, then INCOMPLETE --
    k3_persisted = False
    all_rungs = tuple(fit_rungs) + tuple(adj_rungs)
    viol = monotone_violations(observed_solved, all_rungs)
    if viol:
        print(f"\n[K3] non-monotone frontier detected: {viol} — one fresh-seed re-run per failed "
              f"lower rung (seeds {list(rerun_seeds)})", flush=True)
        rerun_accs = _train_stage(res, "k3rerun", sorted({d_lo for _, d_lo, _ in viol}), rerun_seeds,
                                  cfg, device, cap=cap, batch_size=batch_size,
                                  eval_every=eval_every, recipe=recipe, num_queries=num_queries,
                                  grid_tag=grid_tag + "-k3rerun", out_path=path)
        res["k3_rerun_accs"] = {f"{arm}.D{D}": v for (arm, D), v in rerun_accs.items()}
        res["k3_violations_first_pass"] = [list(v) for v in viol]
        for (arm, d_lo, _d_hi) in viol:
            observed_solved[(arm, d_lo)] = rung_solved(rerun_accs[(arm, d_lo)],
                                                       n_seeds=len(rerun_seeds))
        viol2 = monotone_violations(observed_solved, all_rungs)
        k3_persisted = bool(viol2)
        print(f"[K3] after re-run: violations={viol2} -> "
              f"{'INCOMPLETE (persists)' if k3_persisted else 'resolved — adjudication proceeds'}",
              flush=True)

    # ---- RETENTION (docs/RETENTION.md): archive raw records BEFORE any verdict; verdict cites it -
    raw_archive = archive_run(res, label=f"dfrontier-{REGISTRY_ID}")
    res.setdefault("meta", {})["raw_archive"] = raw_archive

    # ---- the frozen verdict (primary bar >= 3 of 4; K4 / K3 wordings verbatim) --------------------
    verdict = dfrontier_verdict(observed_solved, frozen, adj_rungs, k3_persisted=k3_persisted,
                                smoke=smoke)
    res["verdict"] = verdict
    report = {"registry": REGISTRY_ID, "smoke": bool(smoke), "scale": list(SCALE),
              "grid": list(fit_rungs) + list(adj_rungs), "claim_seeds": list(seeds),
              "cap": cap, "sigma2_law": sigma2_law,
              "matched_params": param_counts, "d_star_fit": d_star, "eps_hat": eps_hat,
              "kill_evaluation": kill, "eps_consistency": consistency,
              "frozen_predictions": frozen,
              "observed_solved": {f"{arm}.D{D}": bool(v) for (arm, D), v in observed_solved.items()},
              "k3_persisted": k3_persisted, "raw_archive": raw_archive, "verdict": verdict}
    res["report"] = report
    _save(res, path)

    _print_report(report, path, smoke=smoke)
    return report


def _print_report(report, path, *, smoke):
    print("\n" + "=" * 78, flush=True)
    print(f"  {REGISTRY_ID} — {'SMOKE (PLUMBING-ONLY)' if smoke else 'POWERED REGISTERED GRID'}",
          flush=True)
    print("=" * 78, flush=True)
    for arm, e in report["eps_hat"].items():
        ns = report["frozen_predictions"].get("n_star", {}).get(arm)
        print(f"    {arm:<15} sigma2_law={report['sigma2_law'][arm]:.5f}  D*_fit="
              f"{report['d_star_fit'][arm]}  eps_hat={e}  N*={ns}", flush=True)
    pred = report["frozen_predictions"].get("predicted_solved", {})
    for arm in ARMS:
        p = pred.get(arm)
        if p:
            print(f"    frozen prediction [{arm}]: " +
                  ", ".join(f"D{D}={'solve' if s else 'FAIL'}" for D, s in p.items()), flush=True)
    for arm in ARMS:
        obs = {k: v for k, v in report["observed_solved"].items() if k.startswith(f"{arm}.")}
        print(f"    observed [{arm}]: " +
              ", ".join(f"{k}={'SOLVE' if v else 'fail'}" for k, v in sorted(obs.items())), flush=True)
    v = report["verdict"]
    for arm, a in v["per_arm"].items():
        if a.get("identified"):
            print(f"    adjudication [{arm}]: agree {a['agreement_count']}/{a['bar']} "
                  f"(pass={a['pass']}, transition distance "
                  f"{a['transition_distance_grid_steps']} grid steps)", flush=True)
        else:
            print(f"    adjudication [{arm}]: UNIDENTIFIED fit — no frozen prediction to test",
                  flush=True)
    print(f"  none-sanity: {v['none_sanity']['note']}", flush=True)
    print(f"  VERDICT [{v['outcome']}]: {v['verdict']}", flush=True)
    print(f"  ledger: {path}", flush=True)
    print(f"  raw archive: {report.get('raw_archive', 'n/a (in meta)')}", flush=True)
    if smoke:
        print("  [SMOKE] numbers are plumbing-only and MEANINGLESS — do NOT cite.", flush=True)
    print("=" * 78, flush=True)


# ==================================================================================== CLI =======
def _build_parser():
    """Argparse guard (recall_gate/surprise_claim pattern): ONLY --smoke / --powered / --out /
    --force-smoke-path; an unknown or typo'd flag exits non-zero BEFORE anything runs — it can
    never silently launch the multi-hour powered grid. Exactly one mode is required."""
    p = argparse.ArgumentParser(
        prog="dfrontier_claim",
        description="PR-2026-09-03-02 claim runner (registered D-frontier grid, crosstalk capacity "
                    "law). --smoke = tiny CPU plumbing run; --powered = the frozen A100 grid "
                    "(requires CUDA). An unknown flag is rejected without launching anything.")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="tiny plumbing-only run (CPU, minutes)")
    mode.add_argument("--powered", action="store_true",
                      help="the registered grid (refuses to start without a CUDA device)")
    p.add_argument("--out", default=None, help="explicit results JSON path (overrides the default)")
    p.add_argument("--force-smoke-path", action="store_true",
                   help="let a --smoke run write the powered ledger it was pointed at "
                        "(default: REFUSED — separate ledgers, BAR-0)")
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(argv)   # SystemExit non-zero on unknown args: nothing runs
    run(smoke=args.smoke, results_path=args.out, force_smoke_path=args.force_smoke_path)


if __name__ == "__main__":
    main()
