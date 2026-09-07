"""
PR-2026-09-03-01 CLAIM RUNNER — powered surprise-gating ablation (retire-if-negative).

Executes the FROZEN pre-registration
`docs/preregistry/2026-09-03-surprise-gating-powered-ablation.md` VERBATIM when a GPU session runs
it. One script, two modes:

  python seq/surprise_claim.py --smoke      # tiny CPU plumbing run -> the SMOKE ledger (~5-10 min)
  python seq/surprise_claim.py --powered    # the A100 claim campaign -> the POWERED ledger
                                            # (refuses to start without a CUDA device)

The pre-registration's named entry point `gpu_surprise_ablation.py` (repo root) is a thin alias of
this module, so the doc's pre-flight command works verbatim.

WHAT IS FROZEN (pre-reg §2-§4 — the constants below ARE that text, not choices):
  arms  = {input, uniform, random, surprise_norm} at d64 L2 H2, shared config feat_map="quad2_lowrank"
          (feat_rank=0 -> r=14, d_phi=137), gated=False, decoupled_gate=False, inctx_lr=False,
          n_delta=1, surprise_gate=False, write_mode="delta", use_workspace=True, use_window=True,
          route_readout=True, short_conv=4, window=16, chunk=64, rope=False, learned_pos=False,
          out_gate=False, state_norm=False, banded_window=False, dropout=0.0, beta_cap=0.99.
  tasks = MQAR-D64 = MixedMQAR(V=256, max_pairs=64, num_queries=128, gap=0, min_pairs=1)
          MQAR-D128 = MixedMQAR(V=512, max_pairs=128, num_queries=128, gap=0, min_pairs=1)
          SELECTIVE-COPY = SelectiveCopy(V=32, mem_len=64, n_data=16, fixed=False)
  seeds = claim (0,1,2,3,4) — n=5 per arm per task, never added/dropped/substituted; LR-sweep
          seed 0 (repo convention); A4 gain-`a` exploratory seed 900 (never a claim seed).
  LR    = per (arm x task) stage-1 sweep over seq.lrsweep.DEFAULT_GRID (rejected LRs recorded = the
          LR-fairness audit), stage-2 retrains every claim seed fresh at the chosen LR
          (seq.gpu_harness.sweep_then_seeds).
  cap   = 40000 steps, batch 64, eval_every 2000, GENWARM recipe (warmup 2000, warmup_frac 0,
          min_lr_frac 0.1), plateau early-stop with the engagement floor (TrainConfig defaults).
  A4 gain `a` (NO-TUNING rule): selected ONCE on exploratory seed 900, task MQAR-D64 only, recipe
          lr = 1e-3 (the GENWARM default), candidates {0.5, 1.0, 2.0, 4.0}; highest best_acc on the
          frozen eval; ties -> smallest a; then frozen for ALL claim seeds and ALL tasks. The four
          exploratory runs are LANE-EXPLORATORY (never a claim, never cited) and go to
          results/exploratory/surprise_gain_selection.json. In --smoke the selection is plumbing and
          stays INSIDE the smoke ledger (the exploratory file is never touched by smoke).
  stats = per task: P1 = superiority_test(acc[surprise_norm, t], acc[uniform, t]); P2 = vs random.
          The 6 raw one-sided Welch p-values are Holm-Bonferroni-corrected at alpha=0.05
          (seq.stats.holm_correction). Decision rule (verbatim §4): SURVIVES iff surprise_norm wins
          vs BOTH controls on >= 2 of the 3 tasks (win = Holm-adjusted p < 0.05 AND point estimate
          mean > ) AND is never RAW one-sided-significantly WORSE than either control on any task
          (the doc's reverse guard; on a won task the guard is mathematically implied by the win,
          so implementing it on every task x control is exactly equivalent to the doc's rule and
          matches its own gloss "never significantly worse anywhere"). Otherwise RETIRED.
          seq.stats.t_isf consumes UPPER-TAIL p in (0, 0.5] — the tail-convention bug PR-03 caught
          is pinned by a checksum against the doc's MDE constants (test + mde_checksum()).
  canary= seq.gpu_harness.negative_control (two byte-identical arms, MQAR-D64, claim seeds 0-1)
          must show NO significant difference; a FAIL invalidates the campaign -> recorded, verdict
          INCONCLUSIVE, no claim (doc §3), and the runner stops before burning further claim budget.

LEDGER SEPARATION + RETENTION (the doc's own mandate; the recall_gate pattern, mirrored):
  smoke   -> results/surprise_ablation_PR-2026-09-03-01/smoke.json
  powered -> results/surprise_ablation_PR-2026-09-03-01/powered.json
  A --smoke run pointed at the powered ledger is REFUSED (_resolve_results_path) unless
  --force-smoke-path. Raw records stream crash-safe after every cell AND are archived verbatim
  (seq.recall_gate.archive_run) BEFORE any verdict; the verdict references the archive path.
  Resume is keyed on (cellkey, config-fingerprint) via seq.gpu_harness; aggregation refuses
  mixed-configuration cells (asserted at verdict time).
  OPERATIONAL NOTE (disclosed, the one deliberate departure): pre-reg §2 item 6 names
  `results/surprise_ablation_smoke.json` / `results/surprise_ablation_powered.json`; the owner's
  commission for this implementation pinned the registry-id directory used here. The
  separation+refusal mandate is implemented verbatim; only the directory differs (LEDDIR below).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Light imports only at module top: the pure layer (paths/refusal, gain selection, verdict, MDE
# checksum) must be importable + unit-testable without torch or training (recall_gate discipline).
try:
    from .stats import superiority_test, holm_correction, t_isf
except ImportError:                                   # run as a bare script: bootstrap sys.path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from seq.stats import superiority_test, holm_correction, t_isf


# ================================================================== frozen protocol constants ====
REGISTRY_ID = "PR-2026-09-03-01"
LEDDIR = "surprise_ablation_PR-2026-09-03-01"   # under $PRIZMA_RESULTS (default ./results)
SMOKE_BASENAME = "smoke.json"
POWERED_BASENAME = "powered.json"

ARMS = ("input", "uniform", "random", "surprise_norm")
TASKS = ("MQAR-D64", "MQAR-D128", "SELECTIVE-COPY")
CONTROLS = ("uniform", "random")          # the decision rule consults ONLY these (owner decision 4)
SCALE = (64, 2, 2)                        # d64 L2 H2 (~130K params), the B6/P2eff scale

CLAIM_SEEDS = (0, 1, 2, 3, 4)             # n=5 per arm per task — frozen, never substituted
EXPLORATORY_SEED = 900                    # the ONE gain-selection seed; never a claim seed
GAIN_CANDIDATES = (0.5, 1.0, 2.0, 4.0)    # frozen candidate set (NO-TUNING rule)
GAIN_SELECTION_LR = 1e-3                  # the GENWARM default, frozen
GAIN_TASK = "MQAR-D64"                    # exploratory gain selection runs on THIS task only

STEP_CAP = 40000
BATCH_SIZE = 64
EVAL_EVERY = 2000
SOLVE_THRESH = 0.9                        # the repo's SOLVE_THRESH (best_acc >= 0.9)
ALPHA = 0.05

# Shared arm config — every value pinned by pre-reg §2 ("identical across arms"). Arm-specific knob
# is ONLY precision_gate (+ the frozen surprise_gain for the A4 arm).
SHARED_PRIZMA_KW = dict(
    feat_map="quad2_lowrank",   # feat_rank=0 -> r=14, d_phi=137, 0 trainable params
    gated=False,                # alpha = 1 -> eps_t = v_t - S_{t-1} k_t exactly (diagnostic convention)
    decoupled_gate=False,       # erase gate = write gate
    inctx_lr=False,
    n_delta=1,
    surprise_gate=False,        # the Lever-A multiplier machinery is NOT part of A4
    write_mode="delta",
    use_workspace=True, use_window=True, route_readout=True,
    short_conv=4, window=16, chunk=64,
    rope=False, learned_pos=False, out_gate=False, state_norm=False,
    banded_window=False, dropout=0.0,
)


def arm_prizma_kw(arm: str, frozen_gain: float | None = None) -> dict:
    """The PrizmaSeqConfig kwargs for one arm: SHARED_PRIZMA_KW + precision_gate (+ the frozen A4
    gain, which changes what the arm's numbers MEAN and therefore enters the config fingerprint)."""
    kw = dict(SHARED_PRIZMA_KW)
    kw["precision_gate"] = arm
    if arm == "surprise_norm":
        assert frozen_gain is not None, \
            "the A4 arm requires the frozen gain from the exploratory NO-TUNING selection"
        kw["surprise_gain"] = float(frozen_gain)
        kw["surprise_ema_lambda"] = 0.02     # frozen a priori (pre-reg §2) — explicit, auditable
    return kw


# ==================================================================== paths + BAR-0 refusal ======
def _results_root() -> str:
    root = os.environ.get("PRIZMA_RESULTS", os.path.join(os.path.dirname(__file__), "..", "results"))
    return os.path.abspath(root)


def _default_results_path(smoke: bool) -> str:
    return os.path.join(_results_root(), LEDDIR, SMOKE_BASENAME if smoke else POWERED_BASENAME)


def exploratory_gain_path() -> str:
    """The LANE-EXPLORATORY gain-selection ledger (pre-reg §2: results/exploratory/
    surprise_gain_selection.json). NEVER written by a --smoke run."""
    return os.path.join(_results_root(), "exploratory", "surprise_gain_selection.json")


def resolve_results_path(explicit=None, *, smoke: bool = False, force_smoke_path: bool = False) -> str:
    """Resolve the results path and enforce the smoke/powered file separation (the doc's own
    mandate, implemented with the recall_gate refusal pattern): a --smoke run pointed at the
    POWERED ledger is refused (SystemExit) unless force_smoke_path. Refusal is the safer default:
    smoke numbers are plumbing-only and a smoke entry inside the claim ledger is exactly the
    2026-06-08 contamination mechanism (results/campaign_2026-06-08/CONTAMINATION.md)."""
    path = explicit if explicit else _default_results_path(smoke)
    if smoke and not force_smoke_path:
        powered = _default_results_path(smoke=False)
        if os.path.normcase(os.path.abspath(path)) == os.path.normcase(os.path.abspath(powered)):
            raise SystemExit(
                f"refusing: a --smoke run was pointed at the powered ledger ({powered}). Smoke and "
                f"campaign results use separate files by default (PR-2026-09-03-01 §2 item 6; see "
                f"results/campaign_2026-06-08/CONTAMINATION.md). Pass --out <path> to write the smoke "
                f"elsewhere, or --force-smoke-path to override this guard deliberately.")
    return path


def require_cuda(has_cuda: bool) -> None:
    """Pure guard for the --powered mode (unit-testable without a CUDA box): the claim campaign is
    the A100 Colab session's job; this machine has no CUDA, and a CPU 'powered' run would be neither
    powered nor the pre-registered environment."""
    if not has_cuda:
        raise SystemExit(
            "refusing: --powered executes the PR-2026-09-03-01 claim campaign and requires a CUDA "
            "device (the A100 Colab session). No CUDA device is visible here. "
            "Use --smoke for the CPU plumbing run.")


# ================================================================ PURE: gain selection (A4) =====
def select_gain(gain_to_best_acc: dict, candidates=GAIN_CANDIDATES) -> float:
    """The NO-TUNING rule as a PURE function (no training): pick the candidate with the highest
    best_acc on the frozen eval; TIES -> the smallest `a`. Exact float equality defines a tie (the
    doc's literal rule — a tolerance would be an extra knob). Iterating candidates in ASCENDING
    order with a strict `>` keeps the smallest `a` on exact ties."""
    best = None
    for a in sorted(candidates):
        acc = gain_to_best_acc[a]
        if best is None or acc > best[1]:
            best = (a, acc)
    return best[0]


# ============================================== PURE: the frozen §4 verdict (Welch + Holm) ======
def surprise_verdict(accs: dict, *, alpha: float = ALPHA) -> dict:
    """The frozen decision rule (pre-reg §4) as a PURE function — unit-testable without training.

    Args:
      accs: {(arm, task) -> [per-seed best_acc, ...]} for arms 'surprise_norm', 'uniform',
            'random' over the three frozen tasks (5 claim seeds each).
    Returns the full audit dict: per-comparison raw tests, the Holm family, per-task wins, the
    reverse-guard table, and survives/verdict wording.
    """
    family, raw = [], {}
    for t in TASKS:
        for ctrl in CONTROLS:
            st = superiority_test(accs[("surprise_norm", t)], accs[(ctrl, t)])
            raw[(t, ctrl)] = st
            family.append(st["p_value"])
    assert len(family) == 6, "the §4 family is exactly the 6 primaries {P1_t, P2_t}"
    holm = holm_correction(family, alpha=alpha)

    wins = {}
    for i, (t, ctrl) in enumerate([(t, c) for t in TASKS for c in CONTROLS]):
        h, st = holm[i], raw[(t, ctrl)]
        sn_mean = float(sum(accs[("surprise_norm", t)]) / len(accs[("surprise_norm", t)]))
        c_mean = float(sum(accs[(ctrl, t)]) / len(accs[(ctrl, t)]))
        # "win" ≡ Holm-adjusted p < 0.05 AND point estimate mean(surprise_norm) > mean(control)
        wins[(t, ctrl)] = bool(h["p_adj"] < alpha) and (sn_mean > c_mean)

    # Reverse guard (§4 rule 2): surprise_norm is never RAW one-sided-significantly WORSE than
    # either control on any task. On a won task the guard is implied by the win (a significant
    # forward superiority forces the reverse p > 0.5), so testing every task x control is exactly
    # the doc's rule incl. its "or, if it wins on all three, on every task" clause.
    guard, reverse = True, {}
    for t in TASKS:
        for ctrl in CONTROLS:
            rev = superiority_test(accs[(ctrl, t)], accs[("surprise_norm", t)])
            reverse[(t, ctrl)] = rev
            guard = guard and not bool(rev["significant"])

    n_both = sum(1 for t in TASKS if wins[(t, "uniform")] and wins[(t, "random")])
    # Rule 1 reading (documented): a task is WON when surprise_norm beats BOTH controls; the
    # mechanism survives iff >= 2 of the 3 tasks are won (plus the reverse guard).
    survives = bool(n_both >= 2 and guard)
    if survives:
        verdict = ("SURVIVES — surprise_norm beat BOTH the constant and random controls on >= 2 of "
                   "3 tasks (Holm-corrected Welch, one-sided) and is never significantly worse "
                   "anywhere; claim scoped to §4 of the pre-registration.")
    else:
        verdict = ("RETIRED — the surprise-norm gate did NOT meet the frozen survival bar "
                   "(pre-reg §4 / owner decision 4): retired from all README novelty claims; the "
                   "§6 retirement patches apply verbatim.")
    return {
        "family": [{"task": t, "control": c, "p_value": raw[(t, c)]["p_value"],
                    "holm_p_adj": holm[i]["p_adj"], "holm_reject": bool(holm[i]["reject"]),
                    "delta": raw[(t, c)]["delta"], "win": wins[(t, c)]}
                   for i, (t, c) in enumerate([(t, c) for t in TASKS for c in CONTROLS])],
        "reverse_guard": [{"task": t, "control": c, "p_value": reverse[(t, c)]["p_value"],
                           "significantly_worse": bool(reverse[(t, c)]["significant"])}
                          for t in TASKS for c in CONTROLS],
        "n_tasks_won_vs_both_controls": n_both,
        "reverse_guard_ok": bool(guard),
        "survives": survives,
        "verdict": verdict,
        "alpha": alpha,
    }


def mde_checksum() -> dict:
    """Recompute the doc's MDE t-quantiles with seq.stats.t_isf (UPPER-TAIL p in (0, 0.5]) and
    compare against the frozen constants (§4 MDE table). Guards the t_isf tail-convention bug class
    that PR-03 caught before accepting a verdict. Returns {name: {computed, frozen, ok}}."""
    frozen = {"t_isf(0.05, 8)": (t_isf(0.05, 8), 1.8595),
              "t_isf(0.20, 8)": (t_isf(0.20, 8), 0.8889),
              "t_isf(0.05/6, 8)": (t_isf(0.05 / 6, 8), 3.0158)}
    out = {}
    for name, (computed, expect) in frozen.items():
        out[name] = {"computed": computed, "frozen": expect, "ok": abs(computed - expect) < 5e-4}
    out["sqrt(2/5)"] = {"computed": (2 / 5) ** 0.5, "frozen": 0.6325, "ok": True}
    return out


# ================================================================================= runner =======
def _task_fac(name: str, smoke: bool = False):
    """Zero-arg task factory per the frozen §3 table (smoke shrinks the shapes, NOT the protocol)."""
    from seq.tasks import MixedMQAR, SelectiveCopy
    if name == "MQAR-D64":
        if smoke:
            return lambda: MixedMQAR(vocab=64, max_pairs=16, num_queries=16, gap=0, min_pairs=1)
        return lambda: MixedMQAR(vocab=256, max_pairs=64, num_queries=128, gap=0, min_pairs=1)
    if name == "MQAR-D128":
        if smoke:
            return lambda: MixedMQAR(vocab=64, max_pairs=16, num_queries=16, gap=0, min_pairs=1)
        return lambda: MixedMQAR(vocab=512, max_pairs=128, num_queries=128, gap=0, min_pairs=1)
    if name == "SELECTIVE-COPY":
        if smoke:
            return lambda: SelectiveCopy(vocab=32, mem_len=16, n_data=8, fixed=False)
        return lambda: SelectiveCopy(vocab=32, mem_len=64, n_data=16, fixed=False)
    raise KeyError(f"unknown task {name!r}; frozen tasks are {TASKS}")


def _arm_factory(arm: str, frozen_gain=None):
    """(name, (V,T)->Module) via seq.gpu_harness.make_arm — the exact arm the campaign scores."""
    from seq.gpu_harness import make_arm
    return make_arm("prizma", SCALE[0], SCALE[1], SCALE[2],
                    **arm_prizma_kw(arm, frozen_gain=frozen_gain))


def _fingerprint(payload: dict) -> str:
    from seq.gpu_harness import config_fingerprint
    return config_fingerprint({"registry": REGISTRY_ID, **payload})


def _train_gain_selection(gain_res, gain_path, task_fac, base_cfg, device, *, cap, batch_size,
                          eval_every, smoke, grid_tag):
    """The four LANE-EXPLORATORY runs behind the NO-TUNING rule: seed 900, MQAR-D64 only, lr=1e-3,
    candidates {0.5, 1.0, 2.0, 4.0}; returns ({a: best_acc}, frozen_a, meta). Crash-safe +
    fingerprint-guarded resume (each candidate is a run_cell under its own cfgsig).

    In --smoke: selection is PLUMBING — results stay inside the SMOKE ledger, the exploratory file
    is never touched, and the tiny smoke task/cap make the numbers meaningless by construction.
    In --powered: `gain_path` is results/exploratory/surprise_gain_selection.json (LANE-EXPLORATORY,
    never a claim) and the per-run wall clock is recorded (pre-reg §5: the measured A4 cost is
    written into the ledger before claim seeds start)."""
    from seq.gpu_harness import run_cell
    facs = {a: _arm_factory("surprise_norm", frozen_gain=a)[1] for a in GAIN_CANDIDATES}
    gain_to_best, secs = {}, {}
    for a in GAIN_CANDIDATES:
        cellkey = f"gainselection.a{a}"
        cfgsig = _fingerprint({"leg": "gain-selection", "gain": a, "task": GAIN_TASK,
                               "cap": cap, "lr": GAIN_SELECTION_LR, "seed": EXPLORATORY_SEED,
                               "scale": list(SCALE), "batch": batch_size,
                               "prizma_kw": arm_prizma_kw("surprise_norm", frozen_gain=a),
                               "grid_tag": grid_tag})
        t0 = time.time()
        rec = run_cell(gain_res, cellkey, facs[a], task_fac, base_cfg, device,
                       seed=EXPLORATORY_SEED, out_path=gain_path, cfgsig=cfgsig)
        secs[a] = round(time.time() - t0, 1)
        gain_to_best[a] = rec["best"]
        if rec.get("cfgsig") != cfgsig:
            raise AssertionError(f"gain-selection cell {cellkey} resumed at a different config "
                                 f"({rec.get('cfgsig')} != {cfgsig}) — refusing to aggregate")
    frozen_a = select_gain(gain_to_best)
    meta = {
        "registry": REGISTRY_ID,
        "lane": "LANE-EXPLORATORY — never a claim, never cited as evidence about the mechanism "
                "(pre-reg §2 NO-TUNING rule; POLICY.md)",
        "seed": EXPLORATORY_SEED, "task": GAIN_TASK, "lr": GAIN_SELECTION_LR,
        "candidates": list(GAIN_CANDIDATES),
        "gain_to_best_acc": {str(a): b for a, b in gain_to_best.items()},
        "frozen_gain": frozen_a,
        "rule": "highest best_acc on the frozen eval; ties -> smallest a (exact float equality)",
        "per_run_seconds": secs,
    }
    return gain_to_best, frozen_a, meta


def run(*, smoke: bool, results_path=None, force_smoke_path: bool = False):
    """Execute the protocol: --smoke (CPU plumbing) or --powered (the A100 claim campaign)."""
    # BAR-0 FIRST: resolve + guard the results path before any heavy import or write.
    path = resolve_results_path(results_path, smoke=smoke, force_smoke_path=force_smoke_path)

    import torch
    from seq.gpu_harness import (make_cfg, sweep_then_seeds, powered_summary, load_results, _save,
                                 get_device, negative_control)
    from seq.lrsweep import DEFAULT_GRID
    from seq.recall_gate import archive_run

    if smoke:
        device = torch.device("cpu")            # smoke stays deterministic + CPU-runnable
        cap, batch_size, eval_every = 500, 16, 250
        seeds, grid = (0,), (1e-3,)
        recipe = dict(warmup=200, warmup_frac=0.0, min_lr_frac=0.1)
        run_tasks = ("MQAR-D64", "SELECTIVE-COPY")   # both task constructors exercised, tiny
        print("=" * 78, flush=True)
        print("  *** SMOKE MODE: PLUMBING-ONLY (PR-2026-09-03-01) ***", flush=True)
        print("  Proves the four arms + gain selection + ledgers wire end-to-end on CPU.", flush=True)
        print("  The numbers are MEANINGLESS (tiny scale/steps) — do NOT cite them.", flush=True)
        print("=" * 78, flush=True)
    else:
        require_cuda(torch.cuda.is_available())  # refuse BEFORE anything else on a CPU-only box
        device = get_device()
        cap, batch_size, eval_every = STEP_CAP, BATCH_SIZE, EVAL_EVERY
        seeds, grid = CLAIM_SEEDS, DEFAULT_GRID
        recipe = dict(warmup=2000, warmup_frac=0.0, min_lr_frac=0.1)   # GENWARM, frozen
        run_tasks = TASKS

    print(f"device={device} registry={REGISTRY_ID} smoke={smoke} results={path}", flush=True)
    res = load_results(path)
    res["meta"] = {"registry": REGISTRY_ID, "smoke": bool(smoke), "scale": list(SCALE),
                   "arms": list(ARMS), "tasks": list(run_tasks), "claim_seeds": list(seeds),
                   "cap": cap, "batch_size": batch_size, "eval_every": eval_every,
                   "grid": list(grid), "recipe": recipe,
                   "lane": "CLAIM (frozen pre-registration)" if not smoke else "SMOKE (plumbing only)"}
    _save(res, path)

    # ---- A4 NO-TUNING gain selection (exploratory seed 900; BEFORE any claim seed, pre-reg §5) ----
    gain_task_fac = _task_fac(GAIN_TASK, smoke=smoke)
    gain_base_cfg = make_cfg(cap, batch_size=batch_size, eval_every=eval_every, log=False,
                             lr=GAIN_SELECTION_LR, **recipe)
    if smoke:
        gain_path = path                        # stay INSIDE the smoke ledger
        gain_res = res
    else:
        gain_path = exploratory_gain_path()     # LANE-EXPLORATORY file (never touched by smoke)
        gain_res = load_results(gain_path)
    _, frozen_a, gain_meta = _train_gain_selection(gain_res, gain_path, gain_task_fac,
                                                   gain_base_cfg, device, cap=cap,
                                                   batch_size=batch_size, eval_every=eval_every,
                                                   smoke=smoke, grid_tag="smoke" if smoke else "claim")
    if not smoke:
        _save(gain_res, gain_path)
        print(f"[gain-selection] frozen a = {frozen_a} (exploratory ledger: {gain_path})", flush=True)
    res["surprise_gain_selection"] = gain_meta
    res["meta"]["frozen_surprise_gain"] = frozen_a
    _save(res, path)

    # ---- arms x tasks: per (arm x task) LR sweep (seed 0) then the claim seeds at the chosen LR ----
    base_cfg = make_cfg(cap, batch_size=batch_size, eval_every=eval_every, log=False, **recipe)
    cells, accs = {}, {}
    for task_name in run_tasks:
        task_fac = _task_fac(task_name, smoke=smoke)
        for arm in ARMS:
            name, fac = _arm_factory(arm, frozen_gain=frozen_a)
            payload = {"task": task_name, "arm": arm, "scale": list(SCALE), "cap": cap,
                       "batch": batch_size, "recipe": recipe, "grid": list(grid),
                       "prizma_kw": arm_prizma_kw(arm, frozen_gain=frozen_a),
                       "claim_seeds": list(seeds)}
            cfgsig = _fingerprint(payload)
            print(f"\n-- [{task_name}] arm '{arm}' [{name}] : LR sweep (seed 0) then seeds "
                  f"{list(seeds)} --", flush=True)
            r = sweep_then_seeds(res, f"{task_name}.{arm}", fac, task_fac, base_cfg, device,
                                 seeds, grid=grid, out_path=path, cfgsig=cfgsig)
            # mixed-configuration aggregation raises (pre-reg §3 resume/integrity)
            bad = [rec.get("cfgsig") for rec in r["per_seed"] if rec.get("cfgsig") != cfgsig]
            assert not bad, (f"{task_name}.{arm}: seeds carry a foreign config fingerprint {bad} "
                             f"(this run is {cfgsig}) — refusing to aggregate mixed configurations")
            assert len({rec["params"] for rec in r["per_seed"]}) == 1, \
                f"{task_name}.{arm}: seeds disagree on parameter count — not the same model"
            summ = powered_summary(r["accs"], solve_thresh=SOLVE_THRESH)
            cells[f"{task_name}.{arm}"] = {"name": name, "cfgsig": cfgsig, "best_lr": r["best_lr"],
                                           "lr_grid": r["lr_grid"], "accs": r["accs"],
                                           "params": r["params"], "summary": summ}
            accs[(arm, task_name)] = list(r["accs"])
            ci = summ["ci95"]
            print(f"   best_lr={r['best_lr']:.1e}  mean={summ['mean']:.3f} "
                  f"CI95=[{ci[0]:.3f},{ci[1]:.3f}]  solve={summ['solve_rate']:.2f} "
                  f"accs={[round(x, 3) for x in r['accs']]}", flush=True)

    # ---- integrity canary FIRST-class: two byte-identical arms must NOT differ (powered only; the
    # smoke's n=1 seeds cannot feed a Welch test, so the canary is a claim-mode obligation). -------
    nc = None
    if not smoke:
        print("\n-- integrity canary: two byte-identical arms (MQAR-D64, seeds 0-1) must NOT "
              "differ --", flush=True)
        nc = negative_control(res, SCALE, _task_fac("MQAR-D64", smoke=False), base_cfg, device,
                              seeds=(0, 1), out_path=path, grid=grid)
        print(f"   p={nc['p_value']:.3f}  significant={nc['significant']}  PASS={nc['pass']}",
              flush=True)
        if not nc["pass"]:
            res["negative_control"] = nc
            res["verdict"] = {"survives": None, "verdict": "INCONCLUSIVE",
                              "reason": "integrity canary FAILED — campaign numbers invalidated; "
                                        "find the harness bug and re-run (pre-reg §3). No claim."}
            _save(res, path)
            raise SystemExit("CANARY FAIL: identical-arm control differs significantly — campaign "
                             "INCONCLUSIVE per PR-2026-09-03-01 §3. No claim until re-run.")

    # ---- RETENTION (docs/RETENTION.md): archive raw records BEFORE any verdict; verdict cites it --
    raw_archive = archive_run(res, label=f"surprise-ablation-{REGISTRY_ID}")
    res.setdefault("meta", {})["raw_archive"] = raw_archive

    # ---- the frozen §4 verdict (claim mode) / plumbing report (smoke mode) ------------------------
    report = {"registry": REGISTRY_ID, "smoke": bool(smoke), "scale": list(SCALE),
              "arms": list(ARMS), "tasks": list(run_tasks), "claim_seeds": list(seeds),
              "cap": cap, "grid": list(grid), "mde_checksum": mde_checksum(),
              "surprise_gain_selection": gain_meta, "cells": cells, "raw_archive": raw_archive}
    if nc is not None:
        report["negative_control"] = nc
    if not smoke:
        missing = [k for k in [(a, t) for t in run_tasks for a in ("surprise_norm", "uniform", "random")]
                   if k not in accs]
        assert not missing, f"verdict requires all three scored arms per task; missing {missing}"
        verdict = surprise_verdict(accs, alpha=ALPHA)
        report["verdict"] = verdict
        res["verdict"] = verdict
    res["report"] = report
    _save(res, path)

    _print_report(report, path, smoke=smoke)
    return report


def _print_report(report, path, *, smoke):
    print("\n" + "=" * 78, flush=True)
    print(f"  {REGISTRY_ID} — {'SMOKE (PLUMBING-ONLY)' if smoke else 'POWERED CLAIM CAMPAIGN'}",
          flush=True)
    print("=" * 78, flush=True)
    gm = report["surprise_gain_selection"]
    print(f"  frozen gain a = {gm['frozen_gain']}  (seed {gm['seed']}, task {gm['task']}, "
          f"accs={gm['gain_to_best_acc']})", flush=True)
    for key, cell in report["cells"].items():
        s = cell["summary"]
        print(f"    {key:<28} lr={cell['best_lr']:.1e} mean={s['mean']:.3f} "
              f"sd={s['sd']:.3f} solve={s['solve_rate']:.2f} {cell['params']}p", flush=True)
    if not smoke:
        for fam in report["verdict"]["family"]:
            print(f"    P[{fam['task']} vs {fam['control']}]  p={fam['p_value']:.4f}  "
                  f"holm={fam['holm_p_adj']:.4f}  win={fam['win']}", flush=True)
        print(f"  tasks won vs BOTH controls: "
              f"{report['verdict']['n_tasks_won_vs_both_controls']}/3   "
              f"reverse-guard ok: {report['verdict']['reverse_guard_ok']}", flush=True)
        print(f"  VERDICT: {report['verdict']['verdict']}", flush=True)
    print(f"  ledger: {path}", flush=True)
    print(f"  raw archive: {report.get('raw_archive', 'n/a (in meta)')}", flush=True)
    if smoke:
        print("  [SMOKE] numbers are plumbing-only and MEANINGLESS — do NOT cite.", flush=True)
    print("=" * 78, flush=True)


# ==================================================================================== CLI =======
def _build_parser():
    """Argparse guard (recall_gate pattern): ONLY --smoke / --powered / --out / --force-smoke-path;
    an unknown or typo'd flag exits non-zero BEFORE anything runs — it can never silently launch
    the multi-hour powered campaign. Exactly one mode is required."""
    p = argparse.ArgumentParser(
        prog="surprise_claim",
        description="PR-2026-09-03-01 claim runner (powered surprise-gating ablation). --smoke = "
                    "tiny CPU plumbing run; --powered = the frozen A100 claim campaign (requires "
                    "CUDA). An unknown flag is rejected without launching anything.")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="tiny plumbing-only run (CPU, minutes)")
    mode.add_argument("--powered", action="store_true",
                      help="the claim campaign (refuses to start without a CUDA device)")
    p.add_argument("--out", default=None, help="explicit results JSON path (overrides the default)")
    p.add_argument("--force-smoke-path", action="store_true",
                   help="let a --smoke run write the powered ledger it was pointed at "
                        "(default: REFUSED — separate ledgers, pre-reg §2 item 6)")
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(argv)   # SystemExit non-zero on unknown args: nothing runs
    run(smoke=args.smoke, results_path=args.out, force_smoke_path=args.force_smoke_path)


if __name__ == "__main__":
    main()
