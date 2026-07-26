"""SOTA-LANDSCAPE powered head-to-head runner for Prizma-Seq v2 (Council-3 deliverable).

THE QUESTION THIS RUNNER ANSWERS.
  A faithful Transformer, GLA (seq/gla.py), Mamba-2 (seq/mamba2.py) and Prizma-v2 now all exist as
  make_arm kinds ('tf','prizma','gla','mamba2'), but nothing compares them HEAD-TO-HEAD. This is the
  Council-3 deliverable: "is Prizma Pareto-competitive across the SOTA landscape at matched scale?",
  answered with POWERED statistics on the diagnostic recall tasks (MQAR-hard / induction /
  selective-copy — the SAME task factories seq/recall_gate.py uses, so the numbers are comparable).

  landscape.py is recall_gate.py GENERALIZED from the TF-vs-Prizma 2-arm gate to the 4-arm SOTA
  landscape. It REUSES the campaign primitives — it does NOT reinvent training, stats, or the LR
  sweep (the mixed-length recall task wrapper below mirrors recall_gate.py's composition):
    seq.gpu_harness : make_arm, make_cfg, sweep_then_seeds, h2h, holm_family,
                      negative_control, load_results, _save, get_device
    seq.stats       : summarize, solve_rate, superiority_test, margin_superiority, tost_equivalence
    seq.tasks       : MixedMQAR, SelectiveCopy, Induction (the recall diagnostics)

TWO LAYERS (cleanly separated so the verdict is UNIT-TESTABLE WITHOUT TRAINING):

  (A) PURE VERDICT LAYER  landscape_verdict(arm_accs, *, cand_key, margin, lower_is_better, params):
        Consumes per-arm per-seed accuracy arrays (no torch, no training) and returns:
          * pareto_table : every arm ranked best-first (by mean; ascending if lower_is_better) with
                           its POWERED summary (real Student-t CI95 via seq.stats.summarize),
                           solve-rate, per-seed accs and disclosed param count,
          * pairwise     : Prizma-vs-EACH-baseline {BEATS / PARITY / WORSE / INCONCLUSIVE} via h2h
                           (margin_superiority + TOST + reverse check), comparison family Holm-corrected,
          * the lower-is-LOSS sign handled correctly (BEATS/WORSE flip when lower_is_better=True).

  (B) TRAINING RUNNER  run_landscape(...):  [the RECALL leg]
        Per diagnostic task: builds the FOUR arms via make_arm at a matched (d,L,H) scale, pre-flight
        runnable-checks each (a broken arm SKIPs with its reason, never crashes the run — mirrors
        gpu_ablation._arm_runnable), then per arm runs sweep_then_seeds (stage-1 LR sweep @1 seed over
        the grid + stage-2 N seeds, SEED-PINNED via build_and_train) and powered_summary. It then
        computes the powered Pareto table + Holm-corrected pairwise verdicts, runs an identical-model
        NEGATIVE CONTROL (integrity canary), and streams everything crash-safe (json -> .tmp ->
        os.replace, resumable by cellkey) to results/gpu_landscape.json.

  (C) CHAR-LM RUNNER  run_landscape_charlm(...):  [the LANGUAGE-MODELING leg]
        The SAME 4 arms (TF / Prizma / GLA / Mamba-2) at matched (d,L,H), but on the char-LM
        bits-per-char axis (BPC = held-out next-char CE (nats)/ln2, LOWER is better). It REUSES
        gpu_charlm2.py's corpus loader (load_corpus / CharData) + char-LM training cell (train_charlm:
        the BPC metric, val-selected early-stop) and the SAME pure verdict layer (A) with
        lower_is_better=True. The Prizma arm carries PRIZMA_CHARLM_KNOBS — the GATE SUPERSET (forget +
        output gates ON), DISTINCT from the recall arm's PRIZMA_V2_KNOBS (gates OFF for clean
        overwrite): char-LM rewards forgetting, recall does not (see seq/delta.py). Same pre-flight ->
        per-arm LR sweep -> seed-pinned multi-seed -> powered Pareto-BPC + Holm pairwise -> identical
        -model negative control -> crash-safe resumable JSON, streamed to results/gpu_landscape_charlm.json
        (a SEPARATE file from the recall leg's gpu_landscape.json).

INTEGRITY.
  * Every number is POWERED (real Student-t CIs / TOST), SEED-PINNED (build_and_train), and
    LR-SWEPT per arm (no magic single-LR). No fabricated metrics.
  * A NEGATIVE CONTROL (two byte-identical Prizma arms with seed-shifted seeds) MUST NOT differ
    significantly — reused from gpu_harness.negative_control.
  * Param spreads are DISCLOSED per arm (GLA / Mamba-2 / Prizma differ from TF by a few % at matched
    d/L/H), recorded in each table row's `params`, never hidden.
  * The --smoke path uses a TINY config (CPU-fast, plumbing-only) and prints a loud DISCLAIMER that
    smoke numbers are NOT a scientific landscape result.

CLI SAFETY. main() parses with argparse: it recognizes --smoke, --full and --charlm and NOTHING else.
An unknown/typo'd flag prints usage and exits NON-ZERO — it NEVER silently launches a multi-hour full
landscape (which would also overwrite the committed results JSON). The FULL landscape runs ONLY on an
explicit no-arg invocation or --full. --charlm selects the LANGUAGE-MODELING leg and COMPOSES with
--smoke (--smoke --charlm = the tiny char-LM smoke); the default (no --charlm) RECALL path is unchanged.

Run (use the module form; the deferred relative imports need the package context):
  python3.13 -m seq.landscape --smoke              # tiny RECALL plumbing smoke -> results/gpu_landscape.json
  python3.13 -m seq.landscape --full               # FULL recall landscape (needs a GPU + budget)
  python3.13 -m seq.landscape                       # no-arg ALSO runs the FULL recall landscape
  python3.13 -m seq.landscape --smoke --charlm     # tiny CHAR-LM smoke -> results/gpu_landscape_charlm.json
  python3.13 -m seq.landscape --charlm             # FULL char-LM landscape (text8, >=10-seed; GPU + budget)
"""
from __future__ import annotations

import argparse
import os
import sys

# Dual-invocation import (works as `python3.13 -m seq.landscape` AND `python3.13 seq/landscape.py`):
# in the bare-script case __package__ is empty and the relative import fails, so we put the repo root
# on sys.path and retry with absolute imports (the deferred torch/harness imports use the same idiom).
# NOTE: only the torch-FREE stats helpers are imported here, so the pure verdict layer below stays
# importable + unit-testable without torch.
try:
    from .stats import summarize, solve_rate, superiority_test, margin_superiority, tost_equivalence
except ImportError:                                  # run as a bare script: bootstrap sys.path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from seq.stats import summarize, solve_rate, superiority_test, margin_superiority, tost_equivalence


# The Prizma arm config for the RECALL diagnostics. The landscape runs seq/recall_gate.py's tasks
# (MQAR-hard / induction / selective-copy), so the Prizma arm MUST match recall_gate.py's REAL-run
# Prizma (seq/recall_gate.py:457) to stay comparable: feat_map='quad2_lowrank' (the v2 lean
# 0-trainable-param recall lever) with the forget/output GATES OFF — the diagnostic recall tasks need a
# CLEAN OVERWRITE, not forgetting (see seq/delta.py's module docstring). The char-LM-tuned gate superset
# (out_gate/state_norm/decoupled_gate/gated, used by gpu_charlm2.py) belongs to a SEPARATE char-LM
# landscape leg, NOT here — using it on recall tasks would be a different model than the campaign reports.
PRIZMA_V2_KNOBS = dict(feat_map="quad2_lowrank")


# The Prizma arm config for the CHAR-LM landscape leg (run_landscape_charlm below). Unlike the recall
# arm (PRIZMA_V2_KNOBS, gates OFF), the char-LM arm turns ON the FULL gate SUPERSET — exactly the v2
# Prizma config gpu_charlm2.py's --v2 arm uses (gpu_charlm2.V2_GATE_KW = out_gate/state_norm/
# decoupled_gate/gated) on top of the shared feat_map='quad2_lowrank' lean recall lever. WHY THE
# DIFFERENCE: per seq/delta.py's module docstring, the diagnostic recall tasks (MQAR/induction/
# selective-copy) need a CLEAN OVERWRITE — "the diagnostic gates ... need clean overwrite, not
# forgetting; the gated path is enabled for char-LM." Char-LM is the opposite regime: natural language
# REWARDS forgetting + an output gate (stale context must decay, the read must be gated), so the gate
# superset HELPS BPC. Using the recall (gates-off) config on char-LM — or the char-LM (gates-on) config
# on recall — would each be a DIFFERENT model than the campaign reports, so the two are pinned distinct.
# feat_map is shared verbatim with gpu_charlm2's char-LM Prizma so the landscape char-LM Prizma == the
# campaign char-LM Prizma. (Gate values mirrored from gpu_charlm2.V2_GATE_KW — single source of truth.)
PRIZMA_CHARLM_KNOBS = dict(out_gate=True, state_norm=True, decoupled_gate=True, gated=True,
                           feat_map="quad2_lowrank")


# ============================================================ PURE VERDICT LAYER ==
# (no torch import here on purpose: this layer must be importable + testable without training)

def _pair_verdict(cand, base, *, margin, lower_is_better, holm_reject):
    """Prizma-vs-one-baseline BEATS / PARITY / WORSE / INCONCLUSIVE, Holm-aware, sign-correct.

    Mirrors seq.gpu_harness.h2h's four-way semantics (WIN / EQUIVALENT / INCONCLUSIVE, plus an explicit
    LOSE) — crucially it does NOT collapse a noisy tie into a loss:

      BEATS        Prizma is the PROVEN winner: the Holm-corrected one-sided superiority of cand>base is
                   rejected (higher-is-acc: superiority_test; lower-is-LOSS: margin_superiority).
      PARITY       TOST-equivalent within +/-margin (a statistically demonstrated tie).
      WORSE        the BASELINE is the proven winner: the REVERSE one-sided test (base>cand) is
                   significant. This reverse check is raw/uncorrected + directional (the Holm family
                   covered only cand>base); it exists solely to separate a real deficit from noise.
      INCONCLUSIVE neither side is proven and it is not equivalent — an UNDER-POWERED call. A comparison
                   where Prizma even LEADS on the mean but isn't significant lands HERE, never WORSE
                   (the bug this fixes: a statistical tie must not read as 'Prizma loses').

    `holm_reject` is the FAMILY-WISE corrected decision for THIS comparison (threaded in by the caller
    from holm_family over the whole pairwise family), so a BEATS label always reflects the corrected
    decision, never a raw single-test flag.
    """
    tost = tost_equivalence(cand, base, margin)
    if lower_is_better:
        test = margin_superiority(cand, base, margin)   # delta = mean(base)-mean(cand); + = cand better
        delta = test["delta"]
    else:
        test = superiority_test(cand, base)             # delta = mean(cand)-mean(base); + = cand better
        delta = test["delta"]

    rev = None
    if bool(holm_reject):
        verdict = "BEATS"
    elif tost["equivalent"]:
        verdict = "PARITY"
    else:
        # Not proven ahead (Holm) and not equivalent. Distinguish a genuine DEFICIT (baseline proven
        # better -> WORSE) from an under-powered tie (-> INCONCLUSIVE) via the REVERSE one-sided test.
        if lower_is_better:
            rev = margin_superiority(base, cand, margin)   # base's loss lower by >= margin => base better
        else:
            rev = superiority_test(base, cand)             # base acc higher => base better
        verdict = "WORSE" if bool(rev["significant"]) else "INCONCLUSIVE"

    return {
        "verdict": verdict,
        "delta": float(delta),                 # signed so + always means Prizma better (both metrics)
        "test": test,
        "tost": tost,
        "reverse_test": rev,                   # the base>cand check (only computed for WORSE/INCONCLUSIVE)
        "equivalent": bool(tost["equivalent"]),
        "raw_p": float(test["p_value"]),
        "margin": margin,
        "lower_is_better": bool(lower_is_better),
    }


def landscape_verdict(arm_accs, *, cand_key="Prizma", margin=0.05, lower_is_better=False,
                      params=None, solve_thresh=0.9):
    """Deterministic Pareto table + pairwise verdicts from per-arm per-seed accuracy arrays.

    Args:
      arm_accs      : {arm_name -> list[float]} per-seed scores for ONE task. Must contain cand_key.
      cand_key      : the candidate arm name (Prizma) — compared against every other arm.
      margin        : the superiority "win" bar AND the TOST equivalence band (score units).
      lower_is_better: False (accuracy, higher=better, default) or True (loss/BPC, lower=better).
      params        : optional {arm_name -> int} param count, disclosed per table row (spreads are
                      shown, not hidden). Missing arms record params=None.
      solve_thresh  : per-arm solve-rate threshold — fraction of seeds with score >= thresh, or <= thresh
                      when lower_is_better (so a BPC solve-rate counts the seeds BELOW the bar).

    Returns:
      {pareto_table, pairwise, pair_order, cand_key, margin, lower_is_better, solve_thresh}
        pareto_table : [ {name, params, mean, median, ci95, sd, solve_rate, accs} ] ranked best-first
                       (descending mean if higher-is-acc, ascending if lower_is_better).
        pairwise     : {baseline_name -> _pair_verdict(...)+holm_p_adj+holm_reject} for every NON-cand
                       arm, Holm-corrected over the whole pairwise family.
        pair_order   : the baseline-name ordering the Holm family was computed in (audit trail).
    """
    from .gpu_harness import h2h, holm_family   # deferred: keeps the pure layer torch-free at import

    if cand_key not in arm_accs:
        raise KeyError(f"landscape_verdict: cand_key {cand_key!r} not in arms {list(arm_accs)}")
    params = params or {}

    # ---- POWERED per-arm summary + ranked Pareto table ----------------------------------------- #
    rows = []
    for name, xs in arm_accs.items():
        s = summarize(xs, solve_thresh=solve_thresh)
        rows.append({
            "name": name,
            "params": params.get(name),
            "mean": s["mean"],
            "median": s["median"],
            "ci95": s["ci95"],
            "sd": s["sd"],
            # solve-rate is SIGN-AWARE: higher-is-acc counts seeds >= thresh; lower-is-LOSS (BPC) counts
            # seeds <= thresh (seq.stats.solve_rate is >=-only, so flip it for the lower_is_better leg).
            "solve_rate": ((sum(1 for x in xs if float(x) <= solve_thresh) / len(xs))
                           if lower_is_better else solve_rate(xs, thresh=solve_thresh)),
            "accs": [float(x) for x in xs],
        })
    # rank: higher mean is better (acc) -> descending; lower is better (loss) -> ascending.
    rows.sort(key=lambda r: r["mean"], reverse=not lower_is_better)

    # ---- Prizma-vs-each-baseline, Holm-corrected over the WHOLE pairwise family ----------------- #
    base_names = [n for n in arm_accs if n != cand_key]
    cand = arm_accs[cand_key]

    # 1) raw one-sided p per baseline (the p-value source for the family correction). For lower-is-loss
    #    we use margin_superiority's p (cand beats base by >= margin); for higher-is-acc the plain
    #    superiority p (cand > base). This is the SAME p that drives the BEATS decision in _pair_verdict.
    raw_pvals = []
    for b in base_names:
        if lower_is_better:
            raw_pvals.append(margin_superiority(cand, arm_accs[b], margin)["p_value"])
        else:
            raw_pvals.append(superiority_test(cand, arm_accs[b])["p_value"])
    holm = holm_family(raw_pvals) if raw_pvals else []
    holm_by_base = dict(zip(base_names, holm))

    pairwise = {}
    for b in base_names:
        hr = holm_by_base[b]
        pv = _pair_verdict(cand, arm_accs[b], margin=margin, lower_is_better=lower_is_better,
                           holm_reject=bool(hr["reject"]))
        pv["holm_p_adj"] = float(hr["p_adj"])
        pv["holm_reject"] = bool(hr["reject"])
        # also surface the gpu_harness.h2h verdict object verbatim (the campaign's WIN/EQUIVALENT
        # string) so the persisted record ties back to the shared primitive, not a private re-impl.
        pv["h2h"] = h2h(cand, arm_accs[b], margin=margin, lower_is_better=lower_is_better,
                        holm_reject=bool(hr["reject"]))
        pairwise[b] = pv

    return {
        "pareto_table": rows,
        "pairwise": pairwise,
        "pair_order": base_names,
        "cand_key": cand_key,
        "margin": margin,
        "lower_is_better": bool(lower_is_better),
        "solve_thresh": solve_thresh,
    }


# ============================================================ TRAINING RUNNER LAYER ==
# Heavy imports (torch, harness, tasks) are deferred into the runner so the PURE layer above stays
# importable + unit-testable in milliseconds without pulling in torch.

def _results_path(explicit=None, *, fname="gpu_landscape.json"):
    if explicit:
        return explicit
    res_dir = os.environ.get("PRIZMA_RESULTS", os.path.join(os.path.dirname(__file__), "..", "results"))
    res_dir = os.path.abspath(res_dir)
    os.makedirs(res_dir, exist_ok=True)
    return os.path.join(res_dir, fname)


def _synthetic_charlm_text(n_chars, seed=0):
    """A deterministic synthetic char corpus for the PLUMBING-ONLY char-LM smoke.

    Used only when tiny-shakespeare (which this repo does not commit -- see .gitignore) is absent, so
    the smoke stays runnable on a fresh clone and in network-free CI. It is a fixed-seed 2nd-order
    Markov-ish stream over a small alphabet: learnable enough that BPC drops below the random-baseline
    (which is what the smoke's plumbing assertions need) and reproducible across runs (which is what
    the resume assertion needs). It is NOT a language corpus and no reported result uses it.
    """
    import random

    rng = random.Random(seed)
    alphabet = "abcdefghijklmnopqrstuvwxyz .,\n"
    words = ["".join(rng.choice(alphabet[:26]) for _ in range(rng.randint(2, 8))) for _ in range(64)]
    out = []
    total = 0
    while total < n_chars:
        w = words[rng.randrange(len(words))]
        sep = rng.choice([" ", " ", " ", ", ", ".\n"])
        out.append(w)
        out.append(sep)
        total += len(w) + len(sep)
    return "".join(out)[:n_chars]


def _make_mixed_induction(vocab, lens):
    """Mixed-length INDUCTION wrapper — the SAME composition seq/recall_gate.py uses (COMPOSES
    seq.tasks.Induction; trains over a set of lengths, evals frozen at the longest/hardest prefix)."""
    from .tasks import Induction

    class _MixedInduction:
        def __init__(self):
            self.vocab = vocab
            self.lens = tuple(sorted(lens))
            self._tasks = {L: Induction(vocab=vocab, seq_len=L) for L in self.lens}
            self.seq_len = max(t.seq_len for t in self._tasks.values())
            self.name = f"MixedInduction(V={vocab},lens={self.lens})"

        def sample(self, B, device):
            import torch
            L = int(self.lens[int(torch.randint(0, len(self.lens), (1,)).item())])
            return self._tasks[L].sample(B, device)

        def eval_sample(self, B, device):
            return self._tasks[self.lens[-1]].sample(B, device)

    return _MixedInduction()


def _arm_runnable(fac, task_fac, device):
    """Cheap pre-flight (mirrors gpu_ablation._arm_runnable): build the arm at the task's (V,T) and run
    ONE tiny forward. Returns (ok, reason). Catches a broken arm BEFORE a costly LR sweep is burned, so
    it SKIPs honestly (its exact error recorded) instead of crashing the whole landscape run."""
    import torch
    try:
        task = task_fac()
        model = fac(task.vocab, task.seq_len).to(device)
        x = torch.randint(0, task.vocab, (2, min(task.seq_len, 16)), device=device)
        with torch.no_grad():
            out = model(x)
        assert out.shape[-1] == task.vocab
        return True, None
    except Exception as e:            # surfaced honestly, not swallowed
        return False, f"{type(e).__name__}: {e}"


# The 4-arm SOTA landscape registry at a matched (d,L,H) scale. 'prizma' carries the tuned v2 knobs so
# the landscape Prizma == the campaign Prizma; the SOTA baselines (tf/gla/mamba2) take no extra knobs.
ARM_KINDS = ("tf", "prizma", "gla", "mamba2")


def _arms(scale):
    """Build the 4 landscape arms via make_arm at scale=(d,L,H). Returns {kind -> (name, factory)}."""
    from .gpu_harness import make_arm
    d, L, H = scale
    out = {}
    for kind in ARM_KINDS:
        knobs = PRIZMA_V2_KNOBS if kind == "prizma" else {}
        out[kind] = make_arm(kind, d, L, H, **knobs)
    return out


def _run_task(res, results_path, task_name, task_fac, *, scale, device, seeds, grid, cap, eval_every,
              margin, solve_thresh, batch_size=64):
    """Train all 4 arms on ONE task (pre-flight -> per-arm LR sweep -> multi-seed -> powered summary),
    then compute the powered Pareto table + Holm-corrected pairwise verdicts. Crash-safe + resumable
    (sweep_then_seeds caches by cellkey; a disconnect resumes exactly where it stopped)."""
    from .gpu_harness import config_fingerprint, make_cfg, sweep_then_seeds, _save

    print(f"\n==== TASK: {task_name} @ d{scale[0]}L{scale[1]}H{scale[2]} ({len(seeds)} seeds) ====",
          flush=True)
    arms = _arms(scale)
    base_cfg = make_cfg(cap, batch_size=batch_size, eval_every=eval_every, log=False)
    # Guard the resume cache on the configuration, not just the cellkey: --smoke and the real run share
    # a default results path, and a key-only skip lets the small one poison the big one (see
    # gpu_harness.config_fingerprint).
    cfgsig = config_fingerprint({"task": task_name, "scale": list(scale), "cap": cap,
                                 "batch_size": batch_size, "eval_every": eval_every,
                                 "grid": list(grid)})

    arm_accs, params, arms_present, unrunnable = {}, {}, {}, {}
    for kind, (name, fac) in arms.items():
        ok, reason = _arm_runnable(fac, task_fac, device)
        if not ok:
            print(f"  -- arm '{kind}' [{name}] UNRUNNABLE -> SKIP : {reason}", flush=True)
            arms_present[kind] = {"name": name, "status": "skipped", "reason": reason}
            unrunnable[kind] = {"name": name, "reason": reason}
            continue
        print(f"  -- arm '{kind}' [{name}] : LR sweep (@seed {seeds[0]}) then {len(seeds)} seeds --",
              flush=True)
        r = sweep_then_seeds(res, f"{task_name}.{kind}", fac, task_fac, base_cfg, device, seeds,
                             grid=grid, out_path=results_path, cfgsig=cfgsig)
        arm_accs[name] = r["accs"]
        params[name] = r["params"]
        arms_present[kind] = {"name": name, "status": "ok", "best_lr": r["best_lr"],
                              "lr_grid": r["lr_grid"]}
        summ = summarize(r["accs"], solve_thresh)
        print(f"     best_lr={r['best_lr']:.1e} solve={solve_rate(r['accs'], solve_thresh):.2f} "
              f"median={summ['median']:.3f} mean={summ['mean']:.3f} "
              f"CI95=[{summ['ci95'][0]:.3f},{summ['ci95'][1]:.3f}] {r['params']:,}p "
              f"accs={[round(a, 3) for a in r['accs']]}", flush=True)

    # the candidate must be runnable to compute the landscape verdict.
    prizma_name = arms["prizma"][0]
    assert prizma_name in arm_accs, (
        f"Prizma arm must be runnable (it is the head-to-head candidate); "
        f"unrunnable={unrunnable.get('prizma')}")

    verdict = landscape_verdict(arm_accs, cand_key=prizma_name, margin=margin,
                                lower_is_better=False, params=params, solve_thresh=solve_thresh)
    block = dict(verdict)
    block["arms_present"] = arms_present
    block["unrunnable"] = unrunnable
    res.setdefault("landscape_report", {}).setdefault("tasks", {})[task_name] = block
    _save(res, results_path)

    print(f"  -> {task_name}: Pareto rank = "
          f"{[r['name'].split('.')[0] for r in block['pareto_table']]}", flush=True)
    for b, pv in block["pairwise"].items():
        print(f"     Prizma vs {b.split('.')[0]:<8} {pv['verdict']:<7} "
              f"(delta={pv['delta']:+.4f}, holm_p_adj={pv['holm_p_adj']:.3f})", flush=True)
    return block


def run_landscape(scale=(128, 4, 4), seeds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), smoke=False,
                  results_path=None, grid=None, margin=0.05, solve_thresh=0.9):
    """Run the 4-arm SOTA landscape head-to-head (TF vs Prizma-v2 vs GLA vs Mamba-2) on the diagnostic
    recall tasks, compute the powered Pareto table + Holm-corrected pairwise verdicts per task, run an
    identical-model negative control, and stream everything crash-safe to results/gpu_landscape.json
    (resumable by cellkey).

    Args:
      scale        : (d_model, n_layers, n_heads) for ALL arms (the matched arena).
      seeds        : per-arm seeds for stage-2 (default 10 = the real powered run; smoke overrides).
      smoke        : True -> TINY config (CPU-fast, plumbing-only; a loud DISCLAIMER is printed). One
                     task, 2 seeds, tiny d/L/H, short cap, single-LR grid.
      results_path : explicit results JSON path (default $PRIZMA_RESULTS/gpu_landscape.json).
      grid         : LR sweep grid (default seq.lrsweep.DEFAULT_GRID; smoke uses a short grid).
      margin       : the superiority win bar + TOST equivalence band (accuracy units).
      solve_thresh : per-arm solve-rate threshold.

    Returns the landscape_report dict (tasks, negative_control, meta).
    """
    import torch
    from .gpu_harness import load_results, negative_control, make_cfg, get_device, _save
    from .lrsweep import DEFAULT_GRID
    from .tasks import MixedMQAR, SelectiveCopy

    # --------- config: smoke (plumbing only) vs full (the real landscape) ---------
    if smoke:
        device = torch.device("cpu")              # CPU keeps the smoke deterministic + MPS-gap-free
        scale = (48, 1, 2)
        seeds = (0, 1)
        cap = 300                                 # short: validates plumbing, not convergence
        eval_every = 150
        batch_size = 16
        # ONE LR on the grid keeps the smoke cheap: with a single LR the per-arm stage-1 sweep is a
        # single train at seed[0] (reused as that seed's stage-2 cell), so each arm is just 2 trains.
        # Four arms + the 2-arm negative control => ~12 tiny trains, CPU-fast (<~60s) and resumable.
        grid = grid or (2e-3,)
        mqar_vocab, mqar_pairs = 48, 6            # ONE task (MQAR-hard, tiny D=6 rung)
        tasks = {
            "MQAR-HARD": lambda: MixedMQAR(vocab=mqar_vocab, max_pairs=mqar_pairs,
                                           num_queries=min(128, 2 * mqar_pairs), gap=0, min_pairs=1),
        }
        margin, solve_thresh = 0.05, 0.5          # plumbing: a low solve bar so the smoke is shaped
        print("=" * 78, flush=True)
        print("  *** SMOKE MODE: PLUMBING-ONLY ***", flush=True)
        print("  These numbers validate that the landscape runner wires together (pre-flight ->", flush=True)
        print("  per-arm LR sweep -> seed-pinned multi-seed -> powered Pareto + Holm pairwise ->", flush=True)
        print("  negative control -> crash-safe JSON). They are NOT a scientific landscape result.", flush=True)
        print("  A real verdict requires the A100 >=10-seed run at the matched scale. DO NOT cite.", flush=True)
        print("=" * 78, flush=True)
    else:
        device = get_device()
        cap = 80000
        eval_every = 2000
        batch_size = 64
        grid = grid or DEFAULT_GRID
        # the diagnostic recall tasks — the SAME factories seq/recall_gate.py uses (so the landscape
        # is directly comparable to the recall gate). MQAR-hard D=128 / mixed-length induction /
        # selective-copy.
        tasks = {
            "MQAR-HARD": lambda: MixedMQAR(vocab=512, max_pairs=128, num_queries=128, gap=0, min_pairs=1),
            "INDUCTION": lambda: _make_mixed_induction(32, (64, 128, 256)),
            "SELECTIVE-COPY": lambda: SelectiveCopy(vocab=32, mem_len=64, n_data=16, fixed=False),
        }
        margin, solve_thresh = 0.05, 0.9

    results_path = _results_path(results_path)
    print(f"device={device} results={results_path} scale=d{scale[0]}L{scale[1]}H{scale[2]} "
          f"seeds={list(seeds)} arms={list(ARM_KINDS)} prizma_knobs={PRIZMA_V2_KNOBS}", flush=True)

    res = load_results(results_path)
    res.setdefault("landscape_report", {})
    res["landscape_report"]["meta"] = {
        "scale": f"d{scale[0]}L{scale[1]}H{scale[2]}", "seeds": list(seeds),
        "arms": list(ARM_KINDS), "prizma_knobs": dict(PRIZMA_V2_KNOBS),
        "margin": margin, "solve_thresh": solve_thresh, "grid": list(grid), "cap": cap,
    }
    res["landscape_report"]["smoke"] = bool(smoke)
    _save(res, results_path)

    # ---- per-task head-to-head ------------------------------------------------------------------ #
    for task_name, task_fac in tasks.items():
        _run_task(res, results_path, task_name, task_fac, scale=scale, device=device, seeds=seeds,
                  grid=grid, cap=cap, eval_every=eval_every, margin=margin, solve_thresh=solve_thresh,
                  batch_size=batch_size)

    # ---- NEGATIVE CONTROL (integrity canary): two byte-identical Prizma arms must NOT differ ----- #
    # Reused verbatim from gpu_harness.negative_control (same arch, seed-shifted seeds -> a genuine
    # canary, not a tautology). Run on the first task so it shares the same arena.
    print("\n-- negative control: two byte-identical Prizma arms must NOT differ significantly --",
          flush=True)
    nc_task_fac = next(iter(tasks.values()))
    base_cfg = make_cfg(cap, batch_size=batch_size, eval_every=eval_every, log=False)
    nc = negative_control(res, scale, nc_task_fac, base_cfg, device, seeds, out_path=results_path,
                          grid=grid)
    res["landscape_report"]["negative_control"] = nc
    _save(res, results_path)
    print(f"   p={nc['p_value']:.3f}  significant={nc['significant']}  PASS={nc['pass']}", flush=True)

    report = res["landscape_report"]
    _print_summary(report, smoke)
    return report


# ============================================================ CHAR-LM LANDSCAPE LEG ==
# The LANGUAGE-MODELING axis of the SAME 4 arms (TF / Prizma / GLA / Mamba-2): char-LM bits-per-char
# (BPC), LOWER is better. This REUSES gpu_charlm2.py's corpus loader + char-LM training cell (so the
# landscape char-LM Prizma == the campaign char-LM Prizma) and the EXISTING pure verdict layer
# (landscape_verdict supports lower_is_better=True). It does NOT touch the recall path above.
#
# Why a thin BPC sweep+seeds here instead of gpu_harness.sweep_then_seeds: the recall harness scores
# token ACCURACY (build_and_train -> masked_acc, higher-is-better) and writes to gpu_charlm2.json via
# gpu_charlm2's module globals. Char-LM scores BPC (lower-is-better) and must stream to THIS leg's own
# results/gpu_landscape_charlm.json. So the cell loop below delegates the TRAINING to the audited
# gpu_charlm2.train_charlm (the BPC metric + val-selected early-stop) but owns the crash-safe cache +
# device, mirroring sweep_then_seeds's stage-1-sweep-then-stage-2-seeds structure exactly.

def _charlm_arms(scale, T):
    """Build the 4 char-LM landscape arms as SINGLE-ARG (V) factories at scale=(d,L,H), context T.

    make_arm yields (name, (V,T)->module) arms; char-LM's run-cell takes a single-arg (V) factory
    (gpu_charlm2 bakes T into ctx), so we bake T in here. Prizma carries the char-LM gate superset +
    learned_pos (its delta path is position-free, so char-LM parity uses a learned position embedding —
    the same convention as gpu_charlm2.ps_factory). Returns {kind -> (name, (V)->module)}."""
    from .gpu_harness import make_arm
    d, L, H = scale
    out = {}
    for kind in ARM_KINDS:
        if kind == "prizma":
            knobs = dict(PRIZMA_CHARLM_KNOBS)
            knobs.setdefault("learned_pos", True)   # char-LM parity: learned absolute positions
        else:
            knobs = {}
        name, fac2 = make_arm(kind, d, L, H, **knobs)   # fac2(V, T) -> module
        out[kind] = (name, (lambda f: (lambda V: f(V, T)))(fac2))   # bake T -> single-arg V factory
    return out


def _charlm_arm_runnable(fac, vocab, T, device):
    """Cheap pre-flight for a char-LM arm (mirrors _arm_runnable): build at the corpus (V) and run ONE
    tiny forward. fac is the single-arg (V) factory. Returns (ok, reason)."""
    import torch
    try:
        model = fac(vocab).to(device)
        x = torch.randint(0, vocab, (2, min(T, 16)), device=device)
        with torch.no_grad():
            out = model(x)
        assert out.shape[-1] == vocab
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _charlm_cell(res, key, build_model, data, hp, device, seed, lr, results_path, log=False):
    """Train ONE (arm x seed) char-LM cell at `lr`, SEED-PINNED, cached + crash-safe to results_path.
    Delegates the actual training to the audited gpu_charlm2.train_charlm (BPC metric, val-selected
    early-stop) but owns the cache key + device + ledger so the char-LM leg streams to its OWN file."""
    import time
    import gpu_charlm2 as gc2
    from .common import param_count, set_seed
    from .gpu_harness import _save

    if key in res and "best_bpc" in res[key]:
        return res[key]
    set_seed(seed)                                  # deterministic model construction (init RNG)
    model = build_model(data.vocab)
    p = param_count(model)
    t0 = time.time()
    r = gc2.train_charlm(model, data, device, lr=lr, seed=seed,
                         steps=hp["steps"], batch_size=hp["batch_size"], warmup=hp["warmup"],
                         grad_clip=hp["grad_clip"], weight_decay=hp["weight_decay"], betas=hp["betas"],
                         eval_every=hp["eval_every"], eval_batches=hp["eval_batches"], log=log)
    rec = {"best_bpc": round(r["best_bpc"], 5), "best_step": r["best_step"],
           "final_bpc": round(r["final_bpc"], 5), "min_test_bpc": r["min_test_bpc"],
           "early_stop": r["early_stop"], "params": p, "lr": lr, "seed": seed,
           "steps": hp["steps"], "sec": round(time.time() - t0, 1)}
    res[key] = rec
    _save(res, results_path)
    return rec


def _charlm_sweep_then_seeds(res, prefix, build_model, data, hp, device, seeds, grid, results_path,
                             log=False):
    """Per-arm BPC LR sweep @seeds[0] (stage-1), then all seeds at the winning LR (stage-2). Crash-safe
    + resumable (each cell cached by key). Returns {best_lr, lr_grid, accs (per-seed BPC), params}.
    Mirrors gpu_harness.sweep_then_seeds, but the metric is BPC (lower-is-better) so the LR winner is
    the MIN best_bpc on the grid (not the max accuracy)."""
    lr_seed = seeds[0]
    grid_rows, grid_bpc = [], {}
    for lr in grid:
        rec = _charlm_cell(res, f"{prefix}.lrsel.lr{lr:.0e}.s{lr_seed}", build_model, data, hp,
                           device, lr_seed, lr, results_path, log=log)
        grid_bpc[lr] = rec["best_bpc"]
        grid_rows.append({"lr": lr, "best_bpc": rec["best_bpc"], "best_step": rec["best_step"]})
    best_lr = min(grid_bpc, key=grid_bpc.get)        # lower BPC is better
    per_seed = []
    for s in seeds:
        if s == lr_seed:                              # reuse the sweep cell that already ran @best_lr
            per_seed.append(res[f"{prefix}.lrsel.lr{best_lr:.0e}.s{lr_seed}"])
        else:
            per_seed.append(_charlm_cell(res, f"{prefix}.final.lr{best_lr:.0e}.s{s}", build_model, data,
                                         hp, device, s, best_lr, results_path, log=log))
    return {"best_lr": best_lr, "lr_grid": grid_rows,
            "accs": [rec["best_bpc"] for rec in per_seed], "params": per_seed[0]["params"]}


def _charlm_negative_control(res, scale, T, data, hp, device, seeds, grid, results_path,
                             seed_offset=None):
    """INTEGRITY CANARY for the char-LM leg: two BYTE-IDENTICAL Prizma char-LM arms (same gate-superset
    config) trained on DIFFERENT seeds must NOT differ significantly in BPC. Mirrors
    gpu_harness.negative_control (same arch, seed-shifted seeds => a genuine canary, not a tautology),
    but on the BPC metric via tost/superiority over the per-seed BPC arrays."""
    from .gpu_harness import NEGCTRL_SEED_OFFSET, make_arm
    seed_offset = NEGCTRL_SEED_OFFSET if seed_offset is None else seed_offset
    d, L, H = scale
    knobs = dict(PRIZMA_CHARLM_KNOBS)
    knobs.setdefault("learned_pos", True)
    _, fac2 = make_arm("prizma", d, L, H, **knobs)
    build_model = lambda V: fac2(V, T)
    seeds_a = tuple(seeds)
    seeds_b = tuple(s + seed_offset for s in seeds)   # SAME arch, DIFFERENT seeds (the real canary)
    ra = _charlm_sweep_then_seeds(res, "charlm.negctrl.A", build_model, data, hp, device, seeds_a,
                                  grid, results_path)
    rb = _charlm_sweep_then_seeds(res, "charlm.negctrl.B", build_model, data, hp, device, seeds_b,
                                  grid, results_path)
    st = superiority_test(ra["accs"], rb["accs"])
    return {"p_value": float(st["p_value"]), "significant": bool(st["significant"]),
            "pass": (not bool(st["significant"])), "delta": float(st["delta"]),
            "accs_a": [float(x) for x in ra["accs"]], "accs_b": [float(x) for x in rb["accs"]],
            "seeds_a": list(seeds_a), "seeds_b": list(seeds_b)}


def run_landscape_charlm(scale=(256, 4, 4), seeds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), smoke=False,
                         results_path=None, lr_grid=None, margin=0.05,
                         solve_thresh=0.9, log=False):
    """Run the 4-arm char-LM BPC head-to-head (TF vs Prizma-charLM vs GLA vs Mamba-2) on the char-LM
    corpus, compute the powered Pareto-BPC table + Holm-corrected pairwise verdicts (lower_is_better),
    run an identical-model negative control, and stream everything crash-safe to
    results/gpu_landscape_charlm.json (resumable by cellkey).

    This is the LANGUAGE-MODELING leg of the landscape (the recall leg lives in run_landscape). It
    REUSES gpu_charlm2.py's corpus loader (load_corpus) + char-LM training cell (train_charlm); the
    metric is BPC = held-out next-char CE (nats)/ln(2), LOWER is better. The Prizma arm carries
    PRIZMA_CHARLM_KNOBS (the gate superset — char-LM benefits from forget/output gates, unlike recall).

    Args:
      scale        : (d_model, n_layers, n_heads) for ALL arms (the matched arena; smoke overrides).
      seeds        : per-arm seeds (default 10 = the real powered run; smoke overrides to 2).
      smoke        : True -> TINY CPU-fast config on a small tiny-shakespeare slice, or on a
                     deterministic synthetic corpus if tiny-shakespeare is not on disk (a loud
                     DISCLAIMER is printed either way, and the corpus actually used is recorded in
                     the JSON); 4 arms + the 2-arm negative control, 2 seeds, 1-LR grid, short cap.
      results_path : explicit results JSON path (default $PRIZMA_RESULTS/gpu_landscape_charlm.json).
      lr_grid      : LR sweep grid (default seq.lrsweep.DEFAULT_GRID; smoke uses a single-LR grid).
      margin       : the BPC superiority win bar AND the TOST equivalence band (BPC units) — the verdict
                     reuses the single `margin` for both (the h2h convention), so there is no separate band.
      solve_thresh : per-arm solve-rate threshold (BPC <= thresh counts as solved; lower-is-better).

    Returns the landscape_charlm_report dict (tasks, negative_control, meta).
    """
    import torch
    import gpu_charlm2 as gc2
    from .gpu_harness import load_results, _save
    from .lrsweep import DEFAULT_GRID

    # --------- config: smoke (plumbing only) vs full (the real char-LM landscape) ---------
    if smoke:
        device = torch.device("cpu")              # CPU keeps the smoke deterministic + MPS-gap-free
        scale = (48, 1, 2)
        seeds = (0, 1)
        T = 64                                    # tiny context
        # tiny tiny-shakespeare slice (~1.1M chars when present locally): take a small contiguous
        # block so CharData's contiguous 90/10 split + frozen eval are both well-formed but CPU-fast.
        # If the corpus is not on disk the loader below substitutes a synthetic one -- see there.
        corpus = "shakespeare"
        hp = dict(steps=120, batch_size=16, warmup=20, grad_clip=1.0, weight_decay=0.1,
                  betas=(0.9, 0.95), eval_every=60, eval_batches=6)
        lr_grid = lr_grid or (2e-3,)              # ONE LR -> each arm is just 2 tiny trains
        slice_chars = 40_000                      # small slice (90/10 -> ~36k train / 4k test)
        margin, solve_thresh = 0.05, 3.0   # low BPC bar so the smoke is shaped
        print("=" * 78, flush=True)
        print("  *** CHAR-LM SMOKE MODE: PLUMBING-ONLY ***", flush=True)
        print("  These BPC numbers validate that the char-LM landscape leg wires together (pre-flight", flush=True)
        print("  -> per-arm LR sweep -> seed-pinned multi-seed -> powered Pareto-BPC + Holm pairwise", flush=True)
        print("  (lower-is-better) -> negative control -> crash-safe JSON). They are NOT a scientific", flush=True)
        print("  result. A real verdict requires the A100 >=10-seed run at scale. DO NOT cite.", flush=True)
        print("=" * 78, flush=True)
    else:
        device = gc2.DEV
        T = 256
        corpus = "text8"
        hp = dict(steps=20000, batch_size=48, warmup=1000, grad_clip=1.0, weight_decay=0.1,
                  betas=(0.9, 0.95), eval_every=250, eval_batches=40)
        lr_grid = lr_grid or DEFAULT_GRID
        slice_chars = None                        # text8 path uses gpu_charlm2's explicit subset sizing
        margin = margin if margin is not None else 0.05

    results_path = _results_path(results_path, fname="gpu_landscape_charlm.json")

    # ---- corpus (reuse gpu_charlm2.load_corpus / CharData; the metric is BPC = CE/ln2) ----------- #
    if corpus == "shakespeare":
        # Use a small contiguous slice of tiny-shakespeare so the smoke is network-free + CPU-fast.
        # CharData does a CONTIGUOUS 90/10 train/test split (no n-gram leak); no val split, so
        # train_charlm falls back to min-over-test (documented in gpu_charlm2).
        #
        # tiny-shakespeare is NOT committed to this repo (see .gitignore: it is fetched on demand from
        # the karpathy URL by gpu_charlm*.py). When it is absent -- a fresh clone, or CI -- the smoke
        # falls back to a DETERMINISTIC SYNTHETIC char corpus rather than failing. That is sound here
        # and only here: this branch is PLUMBING-ONLY (see the disclaimer above), so the identity of
        # the corpus is irrelevant to what it checks. Which corpus was actually used is recorded in
        # the results JSON as `src` + `corpus`, so a synthetic-fallback smoke can never be mistaken
        # for a shakespeare one. No scientific (non-smoke) path ever takes this fallback.
        if os.path.exists(gc2._SHAKES):
            with open(gc2._SHAKES, "r", encoding="utf-8") as f:
                text = f.read()
            if slice_chars:
                text = text[:slice_chars]
            data = gc2.CharData(text, T, "shakespeare")
            src = "local-slice"
        else:
            text = _synthetic_charlm_text(slice_chars or 40_000)
            corpus = "synthetic-smoke"
            data = gc2.CharData(text, T, corpus)
            src = "synthetic-fallback (tiny-shakespeare absent; smoke is plumbing-only)"
            print(f"  [corpus] {gc2._SHAKES} absent -> {src}", flush=True)
    else:
        data, src = gc2.load_corpus(corpus, T)
    print(f"device={device} results={results_path} scale=d{scale[0]}L{scale[1]}H{scale[2]} "
          f"corpus={data.name} (src={src}, rand-BPC={data.rand_bpc:.3f}) seeds={list(seeds)} "
          f"arms={list(ARM_KINDS)} prizma_knobs={PRIZMA_CHARLM_KNOBS}", flush=True)

    res = load_results(results_path)
    res.setdefault("landscape_charlm_report", {})
    rep = res["landscape_charlm_report"]
    rep["meta"] = {
        "scale": f"d{scale[0]}L{scale[1]}H{scale[2]}", "seeds": list(seeds), "arms": list(ARM_KINDS),
        "prizma_knobs": dict(PRIZMA_CHARLM_KNOBS), "corpus": corpus, "ctx": T,
        "margin": margin, "solve_thresh": solve_thresh,
        "lr_grid": list(lr_grid), "hp": dict(hp), "device": device.type,
        "random_baseline_bpc": round(data.rand_bpc, 4),
    }
    rep["smoke"] = bool(smoke)
    rep["lower_is_better"] = True
    rep["metric"] = "bits_per_char"
    rep["prizma_knobs"] = dict(PRIZMA_CHARLM_KNOBS)
    rep["smoke_disclaimer"] = (
        "PLUMBING-ONLY: char-LM smoke BPC + verdict are from a tiny CPU config (few steps/seeds, "
        "tiny corpus slice) and are NOT a scientific result." if smoke else None)
    _save(res, results_path)

    # ---- the single char-LM TASK: 4-arm BPC head-to-head ---------------------------------------- #
    task_name = f"CHARLM-{corpus.upper()}"
    print(f"\n==== TASK: {task_name} @ d{scale[0]}L{scale[1]}H{scale[2]} ({len(seeds)} seeds) ====",
          flush=True)
    arms = _charlm_arms(scale, T)
    arm_accs, params, arms_present, unrunnable = {}, {}, {}, {}
    for kind, (name, fac) in arms.items():
        ok, reason = _charlm_arm_runnable(fac, data.vocab, T, device)
        if not ok:
            print(f"  -- arm '{kind}' [{name}] UNRUNNABLE -> SKIP : {reason}", flush=True)
            arms_present[kind] = {"name": name, "status": "skipped", "reason": reason}
            unrunnable[kind] = {"name": name, "reason": reason}
            continue
        print(f"  -- arm '{kind}' [{name}] : LR sweep (@seed {seeds[0]}) then {len(seeds)} seeds --",
              flush=True)
        r = _charlm_sweep_then_seeds(res, f"{task_name}.{kind}", fac, data, hp, device, seeds, lr_grid,
                                     results_path, log=log)
        arm_accs[name] = r["accs"]
        params[name] = r["params"]
        arms_present[kind] = {"name": name, "status": "ok", "best_lr": r["best_lr"],
                              "lr_grid": r["lr_grid"]}
        summ = summarize(r["accs"], solve_thresh)
        print(f"     best_lr={r['best_lr']:.1e} median_bpc={summ['median']:.3f} "
              f"mean_bpc={summ['mean']:.3f} CI95=[{summ['ci95'][0]:.3f},{summ['ci95'][1]:.3f}] "
              f"{r['params']:,}p accs={[round(a, 3) for a in r['accs']]}", flush=True)

    prizma_name = arms["prizma"][0]
    assert prizma_name in arm_accs, (
        f"Prizma char-LM arm must be runnable (it is the head-to-head candidate); "
        f"unrunnable={unrunnable.get('prizma')}")

    # ---- powered Pareto-BPC table + Holm-corrected pairwise verdicts (LOWER is better) ----------- #
    verdict = landscape_verdict(arm_accs, cand_key=prizma_name, margin=margin,
                                lower_is_better=True, params=params, solve_thresh=solve_thresh)
    block = dict(verdict)
    block["arms_present"] = arms_present
    block["unrunnable"] = unrunnable
    rep.setdefault("tasks", {})[task_name] = block
    _save(res, results_path)
    print(f"  -> {task_name}: Pareto-BPC rank (best=lowest) = "
          f"{[r['name'].split('.')[0] for r in block['pareto_table']]}", flush=True)
    for b, pv in block["pairwise"].items():
        print(f"     Prizma vs {b.split('.')[0]:<8} {pv['verdict']:<7} "
              f"(delta={pv['delta']:+.4f}, holm_p_adj={pv['holm_p_adj']:.3f})", flush=True)

    # ---- NEGATIVE CONTROL (integrity canary): two byte-identical Prizma arms must NOT differ ------ #
    print("\n-- negative control: two byte-identical Prizma char-LM arms must NOT differ in BPC --",
          flush=True)
    nc = _charlm_negative_control(res, scale, T, data, hp, device, seeds, lr_grid, results_path)
    rep["negative_control"] = nc
    _save(res, results_path)
    print(f"   p={nc['p_value']:.3f}  significant={nc['significant']}  PASS={nc['pass']}", flush=True)

    _print_charlm_summary(rep, smoke)
    return rep


def _print_charlm_summary(report, smoke):
    print("\n" + "=" * 78, flush=True)
    print("  CHAR-LM LANDSCAPE HEAD-TO-HEAD (BPC, lower-is-better) — " + (
          "SMOKE (PLUMBING-ONLY, NOT A RESULT)" if smoke else "POWERED RESULTS"), flush=True)
    print("=" * 78, flush=True)
    meta = report.get("meta", {})
    print(f"  scale={meta.get('scale')}  corpus={meta.get('corpus')}  seeds={meta.get('seeds')}  "
          f"rand-BPC={meta.get('random_baseline_bpc')}  margin={meta.get('margin')}", flush=True)
    for task_name, block in report.get("tasks", {}).items():
        print(f"\n  [{task_name}]  Pareto-BPC rank (best=lowest):", flush=True)
        for i, row in enumerate(block["pareto_table"], 1):
            ci = row["ci95"]
            pstr = f"{row['params']:,}p" if row["params"] is not None else "?p"
            print(f"    {i}. {row['name'].split('.')[0]:<8} mean_bpc={row['mean']:.3f} "
                  f"median_bpc={row['median']:.3f} CI95=[{ci[0]:.3f},{ci[1]:.3f}]  {pstr}", flush=True)
        if block.get("unrunnable"):
            for kind, info in block["unrunnable"].items():
                print(f"    (skipped {kind}: {info['reason']})", flush=True)
        print(f"    Prizma vs baselines (lower BPC = Prizma better):", flush=True)
        for b, pv in block["pairwise"].items():
            print(f"      vs {b.split('.')[0]:<8} {pv['verdict']:<7} "
                  f"(delta={pv['delta']:+.4f}, holm_p_adj={pv['holm_p_adj']:.3f}, "
                  f"equivalent={pv['equivalent']})", flush=True)
    nc = report.get("negative_control")
    if nc:
        print(f"\n  NEGATIVE CONTROL (identical models must NOT differ): "
              f"p={nc['p_value']:.3f} significant={nc['significant']} "
              f"=> {'PASS' if nc['pass'] else 'FAIL'}", flush=True)
    if smoke:
        print("\n  [SMOKE] The BPC numbers above are PLUMBING-ONLY (tiny model, few steps/seeds).",
              flush=True)
        print("  [SMOKE] They prove the char-LM pipeline runs end-to-end incl. the negative control.",
              flush=True)
        print("  [SMOKE] They are NOT a scientific result — do NOT cite.", flush=True)
    print("=" * 78, flush=True)


# ------------------------------------------------------------------ presentation --
def _print_summary(report, smoke):
    print("\n" + "=" * 78, flush=True)
    print("  SOTA-LANDSCAPE HEAD-TO-HEAD — " + ("SMOKE (PLUMBING-ONLY, NOT A RESULT)" if smoke
          else "POWERED RESULTS"), flush=True)
    print("=" * 78, flush=True)
    meta = report.get("meta", {})
    print(f"  scale={meta.get('scale')}  seeds={meta.get('seeds')}  arms={meta.get('arms')}  "
          f"margin={meta.get('margin')}", flush=True)
    for task_name, block in report.get("tasks", {}).items():
        print(f"\n  [{task_name}]  Pareto rank (best-first):", flush=True)
        for i, row in enumerate(block["pareto_table"], 1):
            ci = row["ci95"]
            pstr = f"{row['params']:,}p" if row["params"] is not None else "?p"
            print(f"    {i}. {row['name'].split('.')[0]:<8} mean={row['mean']:.3f} "
                  f"median={row['median']:.3f} CI95=[{ci[0]:.3f},{ci[1]:.3f}] "
                  f"solve={row['solve_rate']:.2f}  {pstr}", flush=True)
        if block.get("unrunnable"):
            for kind, info in block["unrunnable"].items():
                print(f"    (skipped {kind}: {info['reason']})", flush=True)
        print(f"    Prizma vs baselines:", flush=True)
        for b, pv in block["pairwise"].items():
            print(f"      vs {b.split('.')[0]:<8} {pv['verdict']:<7} "
                  f"(delta={pv['delta']:+.4f}, holm_p_adj={pv['holm_p_adj']:.3f}, "
                  f"equivalent={pv['equivalent']})", flush=True)
    nc = report.get("negative_control")
    if nc:
        print(f"\n  NEGATIVE CONTROL (identical models must NOT differ): "
              f"p={nc['p_value']:.3f} significant={nc['significant']} "
              f"=> {'PASS' if nc['pass'] else 'FAIL'}", flush=True)
    if smoke:
        print("\n  [SMOKE] The numbers above are PLUMBING-ONLY (tiny model, few steps/seeds).", flush=True)
        print("  [SMOKE] They prove the pipeline runs end-to-end incl. the negative control.", flush=True)
        print("  [SMOKE] They are NOT a scientific result — do NOT cite.", flush=True)
    print("=" * 78, flush=True)


# ------------------------------------------------------------------------- main --
def _build_parser():
    """Argparse parser for the landscape CLI. Recognizes ONLY --smoke and --full; argparse rejects any
    other (unknown/typo'd) flag with a usage message + non-zero exit, so an unintended arg can NEVER
    silently launch the multi-hour FULL landscape (which would overwrite results/gpu_landscape.json).
    Kept as its own helper so the arg-guard is unit-testable WITHOUT any training."""
    p = argparse.ArgumentParser(
        prog="landscape",
        description="SOTA-LANDSCAPE powered head-to-head (TF vs Prizma-v2 vs GLA vs Mamba-2). With no "
                    "flags (or --full) runs the FULL multi-hour RECALL landscape; --smoke runs the tiny "
                    "plumbing-only smoke; --charlm switches to the char-LM BPC leg (composes with "
                    "--smoke). An unknown flag is rejected (non-zero exit) and does NOT launch a full "
                    "landscape.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true",
                      help="tiny plumbing-only smoke (CPU/MPS, minutes)")
    mode.add_argument("--full", action="store_true",
                      help="explicit FULL landscape (same as no-arg; needs a GPU + budget)")
    # The char-LM LEG selector. COMPOSABLE with --smoke (--smoke --charlm = tiny char-LM smoke) and
    # orthogonal to the --smoke/--full mode group, so the default (no-arg) RECALL path is unchanged.
    p.add_argument("--charlm", action="store_true",
                   help="run the char-LM BPC leg (4-arm TF/Prizma/GLA/Mamba-2 head-to-head on "
                        "bits-per-char, lower-is-better) instead of the recall leg; composes with "
                        "--smoke for a tiny char-LM smoke")
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # argparse SystemExits non-zero on an unknown arg (printing usage) BEFORE we ever launch a run.
    args = _build_parser().parse_args(argv)
    if args.charlm:
        run_landscape_charlm(smoke=args.smoke)   # the LANGUAGE-MODELING (BPC) leg
    else:
        run_landscape(smoke=args.smoke)          # the default RECALL leg (unchanged)


if __name__ == "__main__":
    main()
