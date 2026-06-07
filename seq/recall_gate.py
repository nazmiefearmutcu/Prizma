"""
RECALL TOST-PARITY GATE — the pass/fail gate for the word "dominant" (Council-3; plan
"Task 1.Recall-gate"; council record committee/round0_v2_synthesis.md item 9a).

THE QUESTION THIS GATE ANSWERS.
  Can Prizma reach TUNED-Transformer PARITY on the RECALL legs (MQAR-hard, induction, selective-copy)?
  If YES on every leg -> the honest claim may use the word "dominant" (Pareto-dominant: O(1) inference
  AND no recall loss). If NO on any leg -> the honest claim DOWNGRADES to "Pareto-competitive".
  This module is the deterministic referee that turns multi-seed accuracy arrays into that verdict.

TWO LAYERS (cleanly separated so the verdict is UNIT-TESTABLE WITHOUT TRAINING):

  (A) PURE VERDICT LOGIC  (no torch, no training, deterministic):
        recall_gate_verdict(arm_accs, *, tf_key='TF', cand_key, tost_margin, solve_thresh, flip_solved)
        combine_gate(legs)
      These consume per-seed best-accuracy arrays and emit the pass/fail dicts. They use the POWERED
      statistics in seq.stats (real Student-t CIs + TOST + Welch superiority), NEVER a normal-approx CI.

  (B) TRAINING RUNNER  run_recall_gate(...):
      Trains the three arms (TF / Prizma / Hybrid) on each recall leg via the SEED-PINNED entry point
      build_and_train, with a per-model stage-1 LR sweep (sweep_lr, records rejected LRs = the
      LR-fairness audit), then stage-2 at the chosen LR for the requested seeds. On the MQAR-hard rung
      it also runs the FLIP-TEST (a deliberately BIGGER TF) so a tiny-TF failure is attributable to
      capacity, not "attention can't". Everything streams crash-safe (json -> .tmp -> os.replace,
      resumable by cellkey) to results/recall_gate.json, mirroring gpu_bench.run_cell.

INTEGRITY.
  No fabricated metrics. The --smoke path uses a TINY config that runs in minutes on CPU/MPS purely to
  validate the PLUMBING; it prints a loud DISCLAIMER that smoke numbers are NOT a scientific parity
  result (that needs the A100 >=10-seed run). All significance flows through seq.stats (powered), and
  every trained number is seed-pinned via build_and_train -> reproducible.

PARITY DEFINITION (the council bar).
  Prizma reaches PARITY with TF on a leg iff the candidate is TOST-EQUIVALENT to TF within `tost_margin`
  (two one-sided t-tests, the (1-2*alpha) CI of mean(cand)-mean(tf) lies inside (-margin,+margin)).
  A clean PASS additionally requires flip_solved=True on the hard rung. If equivalence fails but Prizma
  is statistically NOT-WORSE than TF by the margin (margin_superiority on the not-worse direction), the
  verdict records `not_worse=True` for the audit, but parity (hence leg_pass) still requires TOST
  equivalence — equivalence is the strict council bar.

Run:
  python3.13 seq/recall_gate.py --smoke        # tiny plumbing smoke (CPU/MPS, minutes) -> results/recall_gate.json
  python3.13 -m seq.recall_gate --smoke        # same, as a module
  python3.13 seq/recall_gate.py                 # FULL gate (needs a GPU + budget; legs at the real scale)
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

# Dual-invocation import: works BOTH as a package module (`python3.13 -m seq.recall_gate`) AND as a
# bare script (`python3.13 seq/recall_gate.py`). In the bare-script case __package__ is empty and the
# relative import fails, so we put the repo root on sys.path and retry with absolute imports. The
# deferred imports inside the runner (torch, models, tasks) use the same try/except pattern.
try:
    from .stats import summarize, solve_rate, tost_equivalence, superiority_test, margin_superiority
except ImportError:                                   # run as a bare script: bootstrap sys.path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from seq.stats import summarize, solve_rate, tost_equivalence, superiority_test, margin_superiority


# ============================================================ PURE VERDICT LAYER ==
# (no torch import here on purpose: this layer must be importable + testable without training)

def recall_gate_verdict(arm_accs: dict, *, tf_key: str = "TF", cand_key: str,
                        tost_margin: float, solve_thresh: float,
                        flip_solved):
    """Deterministic per-leg verdict from per-seed best-accuracy arrays.

    Args:
      arm_accs    : {arm_name -> list[float]} per-seed best-accuracies for ONE leg. Must contain
                    `tf_key` and `cand_key`; extra arms (e.g. 'Hybrid') are summarized for the audit.
      tf_key      : the tuned-Transformer arm name (the reference).
      cand_key    : the candidate arm name (Prizma).
      tost_margin : the equivalence margin (the council parity bar).
      solve_thresh: per-arm solve-rate threshold (fraction of seeds with best_acc >= thresh).
      flip_solved : did a BIGGER TF solve this hard rung? True -> a tiny-TF failure is attributable to
                    capacity (clean gate). False/None -> no bigger-TF evidence -> leg is INCONCLUSIVE
                    (cannot cleanly attribute a TF failure, so cannot cleanly pass).

    Returns:
      {leg_pass, parity, equivalent, not_worse, flip_solved, per_arm, delta, ci90, reason}
        leg_pass   : bool — a CLEAN pass (parity AND flip_solved is True)
        parity     : bool — TOST-equivalent to TF within tost_margin (the strict bar)
        equivalent : bool — same as parity (TOST equivalent) — surfaced explicitly for the test/audit
        not_worse  : bool — Prizma is statistically not-worse than TF by `tost_margin` (audit only)
        flip_solved: the (normalized) flip flag — True only if explicitly True, else False
        per_arm    : {arm -> summarize(...)+'solve_rate_thresh'} for every arm
        delta, ci90: the TOST delta (mean(cand)-mean(tf)) + its (1-2*alpha) CI (audit trail)
        reason     : a human-readable one-liner explaining the verdict
    """
    if tf_key not in arm_accs:
        raise KeyError(f"recall_gate_verdict: tf_key {tf_key!r} not in arms {list(arm_accs)}")
    if cand_key not in arm_accs:
        raise KeyError(f"recall_gate_verdict: cand_key {cand_key!r} not in arms {list(arm_accs)}")

    tf_accs = list(arm_accs[tf_key])
    cand_accs = list(arm_accs[cand_key])

    # POWERED per-arm summaries (real Student-t CI + solve-rate) for EVERY arm in the leg.
    per_arm = {}
    for name, xs in arm_accs.items():
        s = summarize(xs, solve_thresh)
        s["solve_rate_thresh"] = solve_rate(xs, solve_thresh)   # explicit at the gate's threshold
        per_arm[name] = s

    # TOST equivalence: is the candidate statistically EQUIVALENT to TF within the margin?
    tost = tost_equivalence(cand_accs, tf_accs, tost_margin)     # delta = mean(cand)-mean(tf)
    equivalent = bool(tost["equivalent"])

    # NOT-WORSE direction (audit only): test H1 that TF does NOT beat Prizma by more than the margin.
    #   margin_superiority(a=tf, b=cand, margin) tests (mean(cand)-mean(tf)) > margin, i.e. Prizma
    #   beats TF by the margin. For "not worse" we want the reverse guard: Prizma is no worse than
    #   TF by `tost_margin`. That is exactly the lower TOST one-sided test (p_lower < alpha), which
    #   tests H0: diff <= -margin. We report it for the audit trail.
    not_worse = bool(tost["p_lower"] < 0.05)

    # Superiority (descriptive, audit): is Prizma strictly better than TF? (not required for parity)
    sup = superiority_test(cand_accs, tf_accs)

    parity = equivalent
    flip_ok = (flip_solved is True)

    # ---- assemble the verdict ----
    if not flip_ok:
        leg_pass = False
        reason = (
            f"INCONCLUSIVE: flip-test did NOT establish that a bigger TF solves this hard rung "
            f"(flip_solved={flip_solved!r}); a TF/candidate failure here cannot be cleanly attributed "
            f"to capacity vs. 'attention can't', so the leg cannot cleanly pass. "
            f"[TOST equivalent={equivalent}, delta={tost['delta']:+.4f} within +-{tost_margin}]"
        )
    elif parity:
        leg_pass = True
        reason = (
            f"PASS: Prizma is TOST-equivalent to {tf_key} within +-{tost_margin} "
            f"(delta={tost['delta']:+.4f}, ci90={_round_pair(tost['ci90'])}); flip-test clean."
        )
    else:
        leg_pass = False
        reason = (
            f"FAIL: Prizma is NOT TOST-equivalent to {tf_key} within +-{tost_margin} "
            f"(delta={tost['delta']:+.4f}, ci90={_round_pair(tost['ci90'])}; "
            f"not_worse={not_worse}); flip-test clean but parity bar not met."
        )

    return {
        "leg_pass": bool(leg_pass),
        "parity": bool(parity),
        "equivalent": equivalent,
        "not_worse": not_worse,
        "superior": bool(sup["significant"]),
        "flip_solved": flip_ok,
        "flip_solved_raw": flip_solved,
        "per_arm": per_arm,
        "delta": tost["delta"],
        "ci90": tost["ci90"],
        "tost_margin": tost_margin,
        "tost_p_lower": tost["p_lower"],
        "tost_p_upper": tost["p_upper"],
        "tf_key": tf_key,
        "cand_key": cand_key,
        "reason": reason,
    }


def combine_gate(legs: dict):
    """Top-level gate: ALL legs must cleanly pass for the word 'dominant'.

    Args:
      legs : {leg_name -> verdict dict from recall_gate_verdict}
    Returns:
      {gate_pass, per_leg, downgrade_word}
        gate_pass     : bool — True iff EVERY leg has leg_pass True
        per_leg       : {leg_name -> {leg_pass, parity, equivalent, flip_solved, reason}}
        downgrade_word: 'dominant' if gate_pass else 'competitive'
    """
    per_leg = {}
    for name, v in legs.items():
        per_leg[name] = {
            "leg_pass": bool(v.get("leg_pass", False)),
            "parity": bool(v.get("parity", False)),
            "equivalent": bool(v.get("equivalent", False)),
            "flip_solved": bool(v.get("flip_solved", False)),
            "reason": v.get("reason", ""),
        }
    gate_pass = bool(legs) and all(v.get("leg_pass", False) for v in legs.values())
    return {
        "gate_pass": gate_pass,
        "per_leg": per_leg,
        "downgrade_word": "dominant" if gate_pass else "competitive",
    }


def _round_pair(p, nd=4):
    try:
        return (round(float(p[0]), nd), round(float(p[1]), nd))
    except Exception:
        return p


# ============================================================ TRAINING RUNNER LAYER ==
# Heavy imports (torch, models, tasks) are deferred into the runner so the PURE layer above stays
# importable + unit-testable in milliseconds without pulling in torch.

# --- result IO (atomic json -> .tmp -> os.replace; resumable by cellkey; mirrors gpu_bench) ----- #
def _results_path(explicit=None):
    if explicit:
        return explicit
    res_dir = os.environ.get("PRIZMA_RESULTS", os.path.join(os.path.dirname(__file__), "..", "results"))
    res_dir = os.path.abspath(res_dir)
    os.makedirs(res_dir, exist_ok=True)
    return os.path.join(res_dir, "recall_gate.json")


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def _save(path, d):
    tmp = path + ".tmp"
    json.dump(d, open(tmp, "w"), indent=2)
    os.replace(tmp, path)   # atomic: a crash mid-write never corrupts the ledger


# --- mixed-length INDUCTION wrapper (mirrors gpu_diag._MixedInduction; COMPOSES seq.tasks) ------- #
def _make_mixed_induction(vocab, lens):
    from seq.tasks import Induction

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

        def eval_sample(self, B, device):     # frozen eval fixed at the longest (hardest) prefix
            return self._tasks[self.lens[-1]].sample(B, device)

    return _MixedInduction()


# --- arm factories (TF / Prizma / Hybrid), each a lambda(V, T) -> nn.Module (repo convention) ----- #
# NOTE: build_and_train / sweep_lr call `model_fac(**fac_kw)` with NO positional args, so a (V,T)
# factory must be BOUND to a concrete vocab/seq_len first (see `_bind_factory`).
def _tf_factory(d, L, H):
    from seq.transformer import Transformer, TFConfig
    return lambda V, T: Transformer(TFConfig(vocab=V, d_model=d, n_layers=L, n_heads=H,
                                             max_len=T + 8, rope=True))


def _prizma_factory(d, L, H, **prizma_kw):
    from seq.prizma_seq import PrizmaSeqLM, PrizmaSeqConfig
    return lambda V, T: PrizmaSeqLM(PrizmaSeqConfig(vocab=V, d_model=d, n_layers=L, n_heads=H,
                                                    max_len=T + 8, **prizma_kw))


def _hybrid_factory(d, L, H, n_attn=1, **prizma_kw):
    from seq.hybrid import hybrid_factory
    return hybrid_factory(d, L, H, n_attn=n_attn, **prizma_kw)


def _arms_for(scale, prizma_kw, hybrid_n_attn=1):
    d, L, H = scale
    return {
        "TF": _tf_factory(d, L, H),
        "Prizma": _prizma_factory(d, L, H, **prizma_kw),
        "Hybrid": _hybrid_factory(d, L, H, n_attn=hybrid_n_attn, **prizma_kw),
    }


def _bind_factory(vt_factory, vocab, seq_len):
    """Bind a (V, T) -> Module factory to a concrete vocab/seq_len, yielding a ZERO-arg factory that
    build_and_train / sweep_lr can call via `model_fac()`. seq_len is passed UNADJUSTED — the
    underlying factories add their own +8 max_len headroom (matches the repo convention)."""
    return lambda: vt_factory(vocab, seq_len)


# --- per-(arm x leg) training: stage-1 LR sweep + stage-2 multi-seed (seed-pinned) --------------- #
def _train_arm(res, results_path, leg, arm, model_fac, task_fac, *,
               device, cap, seeds, lr_grid, recipe, eval_every, batch_size):
    """Stage-1: sweep_lr (records rejected LRs). Stage-2: build_and_train at the chosen LR for each
    seed. Crash-safe + resumable by cellkey. Returns the per-arm record incl. per-seed best_accs."""
    from dataclasses import replace as _dc_replace
    from seq.common import TrainConfig, build_and_train
    from seq.lrsweep import sweep_lr

    armkey = f"{leg}.{arm}"
    res.setdefault("cells", {})
    cell = res["cells"].get(armkey, {})

    base_cfg = TrainConfig(steps=cap, batch_size=batch_size, log=False, eval_every=eval_every, **recipe)

    # Bind the (V,T) factory to this leg's concrete vocab/seq_len -> a ZERO-arg factory that
    # build_and_train()/sweep_lr() can call as model_fac() (they pass no positional args).
    _probe = task_fac()
    bound_fac = _bind_factory(model_fac, _probe.vocab, _probe.seq_len)

    # ---- stage-1: LR sweep (1 seed), records the FULL grid incl. rejected LRs (LR-fairness audit) ----
    if "sweep" not in cell:
        task = task_fac()
        sw = sweep_lr(bound_fac, task, base_cfg, device, grid=lr_grid, seed=seeds[0])
        cell["sweep"] = sw
        res["cells"][armkey] = cell
        _save(results_path, res)
        print(f"   [{armkey}] LR-sweep best_lr={sw['best_lr']:.2e} best_acc={sw['best_acc']:.3f} "
              f"grid={[(g['lr'], round(g['best_acc'],3)) for g in sw['grid']]}", flush=True)
    best_lr = cell["sweep"]["best_lr"]

    # ---- stage-2: multi-seed at the chosen LR (seed-pinned via build_and_train) ----
    cell.setdefault("seeds", {})
    cfg = _dc_replace(base_cfg, lr=best_lr)
    for s in seeds:
        sk = str(s)
        if sk in cell["seeds"] and "best" in cell["seeds"][sk]:
            continue
        task = task_fac()
        t0 = time.time()
        r = build_and_train(bound_fac, task, cfg, device, seed=s)
        cell["seeds"][sk] = {"best": r.best_acc, "plateau": r.steps_to_plateau,
                             "params": r.params, "sec": round(time.time() - t0, 1), "lr": best_lr}
        res["cells"][armkey] = cell
        _save(results_path, res)
        print(f"   [{armkey} s{s}] best={r.best_acc:.3f} plateau@{r.steps_to_plateau} "
              f"({cell['seeds'][sk]['sec']}s, {r.params}p)", flush=True)

    best_accs = [cell["seeds"][str(s)]["best"] for s in seeds]
    params = cell["seeds"][str(seeds[0])]["params"]
    cell["best_accs"] = best_accs
    cell["best_lr"] = best_lr
    cell["params"] = params
    res["cells"][armkey] = cell
    _save(results_path, res)
    return cell


def _run_leg(res, results_path, leg, task_fac, *, scale, prizma_kw, device, cap, seeds,
             lr_grid, recipe, eval_every, batch_size, tost_margin, solve_thresh,
             flip_solved=None, hybrid_n_attn=1):
    """Train all three arms on a leg, then compute the powered verdict. flip_solved is supplied by the
    caller (None for legs without a flip-test; the MQAR-hard leg passes the bigger-TF result)."""
    print(f"\n==== LEG: {leg} @ d{scale[0]}L{scale[1]}H{scale[2]} ({len(seeds)} seeds) ====", flush=True)
    arms = _arms_for(scale, prizma_kw, hybrid_n_attn=hybrid_n_attn)
    arm_accs = {}
    for aname, fac in arms.items():
        cell = _train_arm(res, results_path, leg, aname, fac, task_fac, device=device, cap=cap,
                          seeds=seeds, lr_grid=lr_grid, recipe=recipe, eval_every=eval_every,
                          batch_size=batch_size)
        arm_accs[aname] = cell["best_accs"]

    verdict = recall_gate_verdict(arm_accs, tf_key="TF", cand_key="Prizma",
                                  tost_margin=tost_margin, solve_thresh=solve_thresh,
                                  flip_solved=flip_solved)
    res.setdefault("legs", {})[leg] = verdict
    _save(results_path, res)
    print(f"  -> {leg}: leg_pass={verdict['leg_pass']} parity={verdict['parity']} "
          f"equivalent={verdict['equivalent']} flip_solved={verdict['flip_solved']}", flush=True)
    print(f"     {verdict['reason']}", flush=True)
    return verdict


def _flip_test(res, results_path, scale, task_fac, *, device, cap, seeds, lr_grid, recipe,
               eval_every, batch_size, solve_thresh):
    """FLIP-TEST on the MQAR-hard rung: train a deliberately BIGGER TF (one scale up) and record
    whether it SOLVES (best_acc >= solve_thresh on >=1 seed). flip_solved=True means a tiny-TF failure
    on this rung is attributable to capacity, not 'attention can't' -> a clean gate."""
    d, L, H = scale
    big = (d * 2, L, H * 2)     # one scale up (wider model, more heads)
    print(f"\n---- FLIP-TEST: bigger TF d{big[0]}L{big[1]}H{big[2]} on MQAR-hard ----", flush=True)
    cell = _train_arm(res, results_path, "MQAR-HARD-FLIP", "TF-big", _tf_factory(*big), task_fac,
                      device=device, cap=cap, seeds=seeds, lr_grid=lr_grid, recipe=recipe,
                      eval_every=eval_every, batch_size=batch_size)
    bests = cell["best_accs"]
    flip_solved = bool(any(b >= solve_thresh for b in bests))
    res.setdefault("flip_test", {})["MQAR-HARD"] = {
        "big_scale": f"d{big[0]}L{big[1]}H{big[2]}", "bests": [round(b, 4) for b in bests],
        "solve_thresh": solve_thresh, "flip_solved": flip_solved,
    }
    _save(results_path, res)
    print(f"  -> flip_solved={flip_solved} (bigger-TF bests={[round(b,3) for b in bests]} "
          f"vs thresh {solve_thresh})", flush=True)
    return flip_solved


# ----------------------------------------------------------------------------------------------- #
def run_recall_gate(scale=(128, 2, 4), seeds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), smoke=False,
                    prizma_kw=None, results_path=None, lr_grid=None, hybrid_n_attn=1):
    """Train the three arms on the recall legs (MQAR-hard, induction, selective-copy), run the
    flip-test on MQAR-hard, compute the powered verdict per leg + the combined gate, and stream
    everything crash-safe to results/recall_gate.json (resumable by cellkey).

    Args:
      scale       : (d_model, n_layers, n_heads) for all arms (the param-matched arena).
      seeds       : per-arm seeds for stage-2 (default 10 — the real powered run; smoke overrides).
      smoke       : True -> TINY config (CPU/MPS, minutes) that validates PLUMBING ONLY (not a parity
                    result — a loud DISCLAIMER is printed). Wires argv '--smoke' to this.
      prizma_kw   : kwargs for PrizmaSeqConfig (the recall-capacity lever). Default = the v2 lean
                    'quad2_lowrank' (d_phi=137); pass feat_map='quad2', feat_n2=256 for the heavy arm.
      results_path: explicit results JSON path (default $PRIZMA_RESULTS/recall_gate.json or ./results).
      lr_grid     : LR sweep grid (default seq.lrsweep.DEFAULT_GRID; smoke uses a short grid).
      hybrid_n_attn: number of attention layers in the Hybrid arm (default 1 = tiny hybrid).

    Returns the combine_gate(...) dict (gate_pass, per_leg, downgrade_word).
    """
    import torch
    from seq.common import get_device
    from seq.tasks import MixedMQAR, SelectiveCopy
    from seq.lrsweep import DEFAULT_GRID

    # --------- config: smoke (plumbing only) vs full (the real gate) ---------
    if smoke:
        # Smoke uses the LEAN feat_map='none' Prizma unless overridden: it exercises the EXACT same
        # pipeline (sweep -> build_and_train -> verdict -> gate -> JSON) but with a much cheaper forward,
        # so the plumbing check finishes fast on CPU. The real run uses 'quad2_lowrank' (the v2 lever).
        if prizma_kw is None:
            prizma_kw = dict(feat_map="none")
        device = torch.device("cpu")                    # CPU keeps the smoke deterministic + MPS-gap-free
        scale = (64, 2, 2)
        seeds = (0, 1)
        cap = 800                                       # short: smoke validates plumbing, not convergence
        eval_every = 400                                # 2 evals over the cap -> best_acc is recorded
        batch_size = 16
        lr_grid = lr_grid or (1e-3, 2e-3)
        recipe = dict(warmup=120, warmup_frac=0.0, min_lr_frac=0.1, min_steps=0)
        mqar_pairs = 8                                  # D=8 hard rung (tiny)
        mqar_vocab = 64
        ind_lens = (24,)
        sc_vocab, sc_mem, sc_k = 32, 24, 6
        tost_margin = 0.05
        solve_thresh = 0.5                              # plumbing: a low bar so the smoke is meaningful-shaped
        print("=" * 78, flush=True)
        print("  *** SMOKE MODE: PLUMBING-ONLY ***", flush=True)
        print("  These numbers validate that the runner wires together (LR sweep -> seed-pinned", flush=True)
        print("  train -> powered verdict -> combined gate -> crash-safe JSON). They are NOT a", flush=True)
        print("  scientific parity result. A real verdict requires the A100 >=10-seed run at the", flush=True)
        print("  param-matched scale. DO NOT cite smoke numbers as evidence of parity.", flush=True)
        print("=" * 78, flush=True)
    else:
        if prizma_kw is None:
            prizma_kw = dict(feat_map="quad2_lowrank")   # the v2 lean lever (d_phi=137, 0 trainable params)
        device = get_device()
        cap = 80000
        eval_every = 2000
        batch_size = 64
        lr_grid = lr_grid or DEFAULT_GRID
        recipe = dict(lr=1e-3, warmup=2000, warmup_frac=0.0, min_lr_frac=0.1)  # gen-warm (gpu_bench)
        mqar_pairs = 128                                # the hard rung (D=128)
        mqar_vocab = 512
        ind_lens = (64, 128, 256)
        sc_vocab, sc_mem, sc_k = 32, 64, 16
        tost_margin = 0.05
        solve_thresh = 0.9

    results_path = _results_path(results_path)
    print(f"device={device} results={results_path} scale=d{scale[0]}L{scale[1]}H{scale[2]} "
          f"seeds={list(seeds)} prizma_kw={prizma_kw}", flush=True)
    res = _load(results_path)
    res["meta"] = {"scale": f"d{scale[0]}L{scale[1]}H{scale[2]}", "seeds": list(seeds),
                   "prizma_kw": {k: v for k, v in prizma_kw.items()},
                   "tost_margin": tost_margin, "solve_thresh": solve_thresh, "smoke": bool(smoke),
                   "lr_grid": list(lr_grid)}
    _save(results_path, res)

    # ---- leg task factories ----
    mqar_fac = lambda: MixedMQAR(vocab=mqar_vocab, max_pairs=mqar_pairs, num_queries=min(128, 2 * mqar_pairs),
                                 gap=0, min_pairs=1)
    ind_fac = lambda: _make_mixed_induction(32, ind_lens)
    sc_fac = lambda: SelectiveCopy(vocab=sc_vocab, mem_len=sc_mem, n_data=sc_k, fixed=False)

    common = dict(scale=scale, prizma_kw=prizma_kw, device=device, cap=cap, seeds=seeds,
                  lr_grid=lr_grid, recipe=recipe, eval_every=eval_every, batch_size=batch_size,
                  tost_margin=tost_margin, solve_thresh=solve_thresh, hybrid_n_attn=hybrid_n_attn)

    # ---- FLIP-TEST first (its result feeds the MQAR-hard verdict) ----
    flip_solved = _flip_test(res, results_path, scale, mqar_fac, device=device, cap=cap, seeds=seeds,
                             lr_grid=lr_grid, recipe=recipe, eval_every=eval_every,
                             batch_size=batch_size, solve_thresh=solve_thresh)

    legs = {}
    legs["MQAR-HARD"] = _run_leg(res, results_path, "MQAR-HARD", mqar_fac, flip_solved=flip_solved, **common)
    legs["INDUCTION"] = _run_leg(res, results_path, "INDUCTION", ind_fac, flip_solved=True, **common)
    legs["SELECTIVE-COPY"] = _run_leg(res, results_path, "SELECTIVE-COPY", sc_fac, flip_solved=True, **common)

    gate = combine_gate(legs)
    res["gate"] = gate
    _save(results_path, res)

    # ---- final summary ----
    print("\n" + "=" * 78, flush=True)
    print("  RECALL TOST-PARITY GATE — SUMMARY", flush=True)
    print("=" * 78, flush=True)
    for leg, v in gate["per_leg"].items():
        status = "PASS" if v["leg_pass"] else ("INCONCLUSIVE" if not v["flip_solved"] else "NOT MET")
        print(f"  {leg:<16} {status:<13} parity={v['parity']} equivalent={v['equivalent']} "
              f"flip_solved={v['flip_solved']}", flush=True)
    print("-" * 78, flush=True)
    verdict_line = "PASS — claim may use the word 'DOMINANT'" if gate["gate_pass"] \
        else "NOT MET — honest claim DOWNGRADES to 'COMPETITIVE'"
    print(f"  GATE: {verdict_line}", flush=True)
    print(f"  downgrade_word = {gate['downgrade_word']!r}", flush=True)
    if smoke:
        print("\n  (SMOKE: plumbing-only — the verdict above is NOT a scientific parity result.)", flush=True)
    print(f"  saved -> {results_path}", flush=True)
    return gate


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    smoke = "--smoke" in argv
    run_recall_gate(smoke=smoke)


if __name__ == "__main__":
    main()
