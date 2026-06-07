"""Shared, correct, v2-ready campaign primitives for the Prizma-Seq A100 study.

This module REPLACES the three reproducibility/statistics defects baked into gpu_bench.py that make
its A100 numbers un-citable for v2. It does NOT modify gpu_bench.py — it composes the already-correct
primitives in seq/ into a campaign harness the v2 ablations build on:

  D1  SEED-PINNED INIT.  gpu_bench.run_cell builds the model THEN trains, so per-seed init is not
      seed-pinned (the documented v1 repro defect). Here run_cell drives seq.common.build_and_train,
      which calls set_seed(seed) BEFORE constructing the model -> init + data are both seed-pinned.
  D2  POWERED STATISTICS.  gpu_bench.ci95 uses a normal 1.96 multiplier (anti-conservative at low df).
      powered_summary / h2h here use the REAL Student-t machinery in seq.stats (summarize,
      superiority_test, margin_superiority, tost_equivalence, holm_correction, solve_rate).
  D3  PER-MODEL LR SWEEP.  gpu_bench uses one fixed GENWARM lr. sweep_then_seeds runs seq.lrsweep
      stage-1 (full grid @1 seed, recording the rejected LRs = the LR-fairness audit) per arm, then
      runs the multi-seed stage-2 at the chosen LR.

The atomic crash-safe JSON idiom (json -> .tmp -> os.replace) from gpu_bench._save is reused verbatim
so a Colab disconnect never loses progress and a half-written file is never observed.

A "model factory" here has the (lambda V, T: nn.Module) signature used everywhere in this repo
(tf_factory / ps_factory / hybrid_factory), so make_arm's factories drop straight into run_cell.
A "task factory" is a zero-arg callable returning a freshly-built task (so each cell gets its own
RNG-clean task instance, exactly like gpu_bench's task_fac).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import replace

from .common import TrainConfig, build_and_train, param_count, get_device  # noqa: F401 (re-export)
from .lrsweep import sweep_lr, DEFAULT_GRID
from .stats import (
    summarize,
    solve_rate,
    superiority_test,
    margin_superiority,
    tost_equivalence,
    holm_correction,
)
from .transformer import Transformer, TFConfig
from .prizma_seq import PrizmaSeqLM, PrizmaSeqConfig
from .hybrid import hybrid_factory


# =========================================================== recipe constants ====
# GENWARM-style default recipe (mirrors gpu_bench.GENWARM): a single generous absolute warmup with
# cosine decay, IDENTICAL across every arm so the comparison stays fair. Named constants so the
# campaign + tests reference one source of truth instead of magic numbers.
GENWARM_LR = 1e-3
GENWARM_WARMUP = 2000
GENWARM_WARMUP_FRAC = 0.0
GENWARM_MIN_LR_FRAC = 0.1
GENWARM = dict(lr=GENWARM_LR, warmup=GENWARM_WARMUP,
               warmup_frac=GENWARM_WARMUP_FRAC, min_lr_frac=GENWARM_MIN_LR_FRAC)


def make_cfg(cap, *, batch_size=64, eval_every=2000, log=False, **recipe):
    """TrainConfig builder for the campaign: applies the GENWARM recipe by default, with `recipe`
    overrides (e.g. lr from the per-arm sweep). `cap` = step budget. Mirrors the gpu_bench cell cfg
    so cells are directly comparable, but exposed so the ablation + tests build configs uniformly."""
    merged = dict(GENWARM)
    merged.update(recipe)
    return TrainConfig(steps=cap, batch_size=batch_size, log=log,
                       eval_every=eval_every, **merged)


# ================================================================ crash-safe IO ==
def _json_default(o):
    """JSON fallback for the numpy scalars (np.bool_/np.float64/np.int64) that seq.stats returns —
    superiority_test/tost/summarize produce numpy types, and the crash-safe ledger must persist them.
    Coerce to native Python via .item() (covers all numpy 0-d scalar types)."""
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _save(d, out_path):
    """Atomic crash-safe write (json -> .tmp -> os.replace), reused from gpu_bench._save. os.replace
    is atomic on the same filesystem, so a reader never observes a half-written results file and a
    Colab disconnect mid-write cannot corrupt the ledger. `default=_json_default` makes the numpy
    scalars in the powered-stats payload serializable (the payload this harness exists to persist)."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2, default=_json_default)
    os.replace(tmp, out_path)


def load_results(out_path):
    """Load the resumable ledger (empty dict if absent)."""
    return json.load(open(out_path)) if os.path.exists(out_path) else {}


# ===================================================================== run_cell ==
def run_cell(res, cellkey, model_fac, task_fac, cfg: TrainConfig, device, seed, out_path):
    """Train ONE (model x task x seed) cell, SEED-PINNED, cached + crash-safe.

    Args:
      res      : the in-memory results dict (the resumable ledger).
      cellkey  : unique string key for this cell; if already present (with a 'best'), SKIP + return it.
      model_fac: (lambda V, T: nn.Module) — the arm factory (from make_arm / tf_factory / ...).
      task_fac : zero-arg callable -> a fresh task instance (carries .vocab and .seq_len).
      cfg      : TrainConfig (the lr/cap/recipe for this cell).
      device   : torch.device.
      seed     : per-seed integer; set BEFORE the model is built (via build_and_train) so INIT is pinned.
      out_path : JSON path for the crash-safe ledger.

    Returns the cell record: {best, plateau, params, sec, seed, lr, cap}.
    """
    if cellkey in res and "best" in res[cellkey]:
        return res[cellkey]
    task = task_fac()
    # Capture the (V,T) the arm needs, then hand build_and_train a ZERO-ARG factory so it can
    # set_seed(seed) BEFORE constructing the model (the seed-pinned-init fix). This is the crux of D1.
    V, T = task.vocab, task.seq_len
    zero_arg_fac = lambda: model_fac(V, T)
    t0 = time.time()
    r = build_and_train(zero_arg_fac, task, cfg, device, seed=seed)
    rec = {
        "best": r.best_acc,
        "plateau": r.steps_to_plateau,
        "params": r.params,
        "sec": round(time.time() - t0, 1),
        "seed": seed,
        "lr": cfg.lr,
        "cap": cfg.steps,
    }
    res[cellkey] = rec
    _save(res, out_path)
    return rec


# ============================================================ sweep_then_seeds ==
def sweep_then_seeds(res, prefix, model_fac, task_fac, base_cfg: TrainConfig, device,
                     seeds, grid=DEFAULT_GRID, out_path=None):
    """Per-arm LR FAIRNESS: stage-1 sweep_lr @1 seed over `grid` (recording the FULL grid incl.
    rejected LRs = the audit trail), then stage-2 run_cell for each seed at the chosen best_lr.

    Everything is cached + crash-safe: the sweep result is stored under f'{prefix}.sweep' and each
    seed cell under f'{prefix}.s{seed}', so a disconnect resumes exactly where it stopped.

    Returns: {best_lr, lr_grid (list of {lr,best_acc,steps_to_plateau}), per_seed: [records], accs}.
    """
    sweep_key = f"{prefix}.sweep"
    if sweep_key in res and "best_lr" in res[sweep_key]:
        sweep = res[sweep_key]
    else:
        task = task_fac()
        V, T = task.vocab, task.seq_len
        zero_arg_fac = lambda: model_fac(V, T)
        # sweep_lr internally uses build_and_train (seed-pinned) for every LR on the grid.
        sweep = sweep_lr(zero_arg_fac, task, base_cfg, device, grid=grid, seed=seeds[0])
        res[sweep_key] = sweep
        if out_path is not None:
            _save(res, out_path)

    best_lr = sweep["best_lr"]
    seed_cfg = replace(base_cfg, lr=best_lr)
    per_seed = [
        run_cell(res, f"{prefix}.s{s}", model_fac, task_fac, seed_cfg, device, seed=s, out_path=out_path)
        for s in seeds
    ]
    return {
        "best_lr": best_lr,
        "lr_grid": sweep["grid"],
        "per_seed": per_seed,
        "accs": [rec["best"] for rec in per_seed],
    }


# ============================================================== powered_summary ==
def powered_summary(accs, solve_thresh=0.9):
    """seq.stats.summarize (REAL Student-t CI) merged with solve_rate. summarize already computes a
    solve_rate at the same threshold; we re-assert it explicitly so the field is unambiguous."""
    s = dict(summarize(accs, solve_thresh=solve_thresh))
    s["solve_rate"] = solve_rate(accs, thresh=solve_thresh)
    return s


# ===================================================================== h2h ======
def h2h(cand_accs, base_accs, *, margin, lower_is_better=False):
    """Powered head-to-head of a candidate arm vs a baseline arm.

    Always reports BOTH a superiority test AND a TOST equivalence test, plus a plain verdict string:

      higher-is-better (accuracy, default):
        superiority_test(cand, base)  -> H1: mean(cand) > mean(base)
      lower-is-better (BPC):
        margin_superiority(cand, base, margin) -> H1: baseline worse than candidate by >= margin
        (a=candidate BPC, b=baseline BPC; significant when candidate beats baseline by >= margin)

    `margin` doubles as the TOST equivalence band: equivalent when |mean(cand)-mean(base)| < margin.

    Returns a dict with: superiority OR margin_superiority, tost, delta, lower_is_better, verdict.
    """
    tost = tost_equivalence(cand_accs, base_accs, margin)
    out = {"lower_is_better": lower_is_better, "tost": tost, "margin": margin}

    if lower_is_better:
        ms = margin_superiority(cand_accs, base_accs, margin)
        out["margin_superiority"] = ms
        win = ms["significant"]
        out["delta"] = ms["delta"]                 # = mean(base) - mean(cand) (positive = cand better)
    else:
        sup = superiority_test(cand_accs, base_accs)
        out["superiority"] = sup
        win = sup["significant"]
        out["delta"] = sup["delta"]                # = mean(cand) - mean(base)

    if win:
        out["verdict"] = f"WIN (margin>={margin})"
    elif tost["equivalent"]:
        out["verdict"] = f"EQUIVALENT (within +/-{margin})"
    else:
        out["verdict"] = "INCONCLUSIVE"
    return out


def holm_family(pvals, alpha=0.05):
    """Holm-Bonferroni wrapper over a family of comparison p-values (exposes seq.stats.holm_correction).
    Returns the list of {p, p_adj, reject} in input order."""
    return holm_correction(pvals, alpha=alpha)


# =============================================================== ARM SPEC =======
def make_arm(kind, d, L, H, **knobs):
    """Declarative arm spec -> (name, factory).

    kind in {'tf','prizma','hybrid'}.
    factory has the (lambda V, T: nn.Module) signature so it drops into run_cell / sweep_then_seeds.
    For 'prizma' and 'hybrid', `knobs` forward the v2 PrizmaSeqConfig levers verbatim:
      out_gate, state_norm, decoupled_gate, surprise_gate, surprise_mode, n_delta,
      feat_map, feat_n2, feat_rank, inctx_lr, gated  (and any other PrizmaSeqConfig field).
    'hybrid' also accepts n_attn / attn_layers / tf_rope (consumed by hybrid_factory) — the rest
    forward to the per-layer PrizmaSeqConfig.

    The name encodes scale + the non-default knobs so cellkeys are self-describing in the ledger.
    """
    scale = f"d{d}L{L}H{H}"

    def _knob_tag():
        if not knobs:
            return ""
        return "_" + "_".join(f"{k}={v}" for k, v in sorted(knobs.items()))

    if kind == "tf":
        if knobs:
            raise ValueError(f"'tf' arm takes no Prizma knobs, got {knobs}")
        name = f"TF.{scale}"
        fac = lambda V, T: Transformer(
            TFConfig(vocab=V, d_model=d, n_layers=L, n_heads=H, max_len=T + 8, rope=True))
        return name, fac

    if kind == "prizma":
        name = f"Prizma.{scale}{_knob_tag()}"
        fac = lambda V, T: PrizmaSeqLM(
            PrizmaSeqConfig(vocab=V, d_model=d, n_layers=L, n_heads=H, max_len=T + 8, **knobs))
        return name, fac

    if kind == "hybrid":
        name = f"Hybrid.{scale}{_knob_tag()}"
        fac = hybrid_factory(d, L, H, **knobs)   # returns lambda V, T: HybridSeqLM(...)
        return name, fac

    raise ValueError(f"unknown arm kind {kind!r} (expected 'tf'|'prizma'|'hybrid')")


# ============================================================ negative_control ==
def negative_control(res, scale, task_fac, base_cfg: TrainConfig, device, seeds, out_path):
    """The INTEGRITY CANARY. Build TWO byte-IDENTICAL Prizma arms (same config, only the cellkey
    differs), run sweep_then_seeds for each, and assert via superiority_test that they are NOT
    significantly different (p should sit near ~0.5, definitely not < 0.05).

    If two identical models show a 'significant win', the pipeline (stats, seeding, or eval) is
    broken — so this MUST be present and pass before any real 'win' is trusted.

    Returns {p_value, significant, pass, accs_a, accs_b, delta}.
    """
    d, L, H = scale
    _, fac = make_arm("prizma", d, L, H)          # baseline Prizma, no knobs — identical config
    ra = sweep_then_seeds(res, "negctrl.A", fac, task_fac, base_cfg, device, seeds, out_path=out_path)
    rb = sweep_then_seeds(res, "negctrl.B", fac, task_fac, base_cfg, device, seeds, out_path=out_path)
    st = superiority_test(ra["accs"], rb["accs"])
    return {
        "p_value": st["p_value"],
        "significant": st["significant"],
        "pass": not st["significant"],
        "delta": st["delta"],
        "accs_a": ra["accs"],
        "accs_b": rb["accs"],
    }
