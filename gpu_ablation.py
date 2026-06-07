"""S3 NOVEL-CORE ablation for Prizma-Seq v2, built on the seed-pinned + powered + LR-swept +
crash-safe harness (seq.gpu_harness). Answers the v2 novel-core science question on the hardest
discriminator (MQAR-hard, MixedMQAR D=128 — recall is where surprise-TARGETING should matter):

  Lever A (surprise-gated write).  Does surprise-TARGETING beat BOTH of its controls?
    - 'surprise_norm'     : write scaled by g_t = f(||eps_t||)         (the real lever)
    - 'surprise_random'   : write scaled by a random gate of matched magnitude   (control)
    - 'surprise_constant' : write scaled by a constant gate of matched magnitude (control)
    The lever EARNS the novel-core slot only if surprise_norm beats BOTH controls, not just baseline.

  Lever G (in-context per-channel learning rate, RWKV-7 "Goose" generalized delta).
    - 'inctx_lr'          : per-VALUE-channel rate eta_t replaces the scalar write gate beta_t.
    Question: does inctx_lr beat the plain baseline?

  'baseline' : Prizma feat_map='none', no levers — the param/behaviour reference for both questions.

PARAM NOTE. Trainable-gate arms (surprise_*, inctx_lr add W_g / W_eta) add a small number of params
over baseline; the matched-param REFERENCE for the whole family is the TF arm at this scale (the TF
arm is not run here — this is an intra-Prizma novel-core ablation; the full TF head-to-head lives in
gpu_bench phase 2). All Prizma arms share the SAME d/L/H scale, so they grow in lockstep and the
inter-arm param spread is only the gate projections (disclosed in the per-arm 'params' field).

PROTOCOL (every arm, no exceptions):
  1. sweep_then_seeds: per-arm stage-1 LR sweep @1 seed (records rejected LRs = LR-fairness audit),
     then stage-2 multi-seed at the chosen LR. SEED-PINNED init via build_and_train.
  2. powered_summary: solve-rate + median + REAL Student-t 95% CI.
  3. h2h vs baseline: superiority + TOST, then Holm-correct the whole comparison FAMILY.
  4. negative_control: two byte-identical Prizma arms must NOT differ significantly (integrity canary).
  5. Stream everything crash-safe to results/gpu_ablation.json (resumable).

Council-1 signs off the CAUSAL claim later; this script only produces the powered numbers + verdicts.

Run:
  python3.13 gpu_ablation.py            # full S3 ablation (A100-scale)
  python3.13 gpu_ablation.py --smoke    # TINY plumbing-only run (CPU/MPS, a few minutes)
"""
from __future__ import annotations

import os
import sys

import torch

from seq.gpu_harness import (
    make_arm, make_cfg, sweep_then_seeds, powered_summary, h2h,
    holm_family, negative_control, load_results, _save, get_device,
)
from seq.tasks import MixedMQAR


RES_DIR = os.environ.get("PRIZMA_RESULTS", os.path.join(os.path.dirname(__file__), "results"))
OUT = os.path.join(RES_DIR, "gpu_ablation.json")

# Margin for both the superiority "win" bar and the TOST equivalence band (accuracy units).
MARGIN = 0.05
SOLVE_THRESH = 0.9


# ----------------------------------------------------------------- arm registry --
def _arms(scale):
    """The S3 novel-core arms at `scale=(d,L,H)`. 'baseline' first so it's the h2h reference."""
    d, L, H = scale
    specs = [
        ("baseline",          dict()),                                          # Prizma none/plain
        ("surprise_norm",     dict(surprise_gate=True, surprise_mode="norm")),      # Lever A (real)
        ("surprise_random",   dict(surprise_gate=True, surprise_mode="random")),    # Lever A control
        ("surprise_constant", dict(surprise_gate=True, surprise_mode="constant")),  # Lever A control
        ("inctx_lr",          dict(inctx_lr=True)),                                 # Lever G
    ]
    out = {}
    for tag, knobs in specs:
        name, fac = make_arm("prizma", d, L, H, **knobs)
        out[tag] = (name, fac)
    return out


def _arm_runnable(fac, task_fac, device):
    """Cheap pre-flight: build the arm at the task's (V,T) and run ONE tiny forward. Returns
    (ok, reason). This catches model-side wiring gaps (e.g. surprise_mode='random' currently lacks a
    threaded RNG in PrizmaSeqLM.forward) BEFORE we burn a costly LR sweep on an arm that cannot run.
    Honest by construction: an unrunnable control is recorded with its exact error, never faked."""
    try:
        task = task_fac()
        model = fac(task.vocab, task.seq_len).to(device)
        x = torch.randint(0, task.vocab, (2, min(task.seq_len, 16)), device=device)
        with torch.no_grad():
            out = model(x)
        assert out.shape[-1] == task.vocab
        return True, None
    except Exception as e:           # surfaced honestly, not swallowed
        return False, f"{type(e).__name__}: {e}"


# ----------------------------------------------------------------------- runner --
def run_ablation(*, scale, task_fac, device, seeds, grid, cap, eval_every, smoke):
    res = load_results(OUT)
    arms = _arms(scale)

    base_cfg = make_cfg(cap, eval_every=eval_every, log=False)

    # ---- per-arm: pre-flight -> per-arm LR sweep -> multi-seed -> powered summary ---------------- #
    arm_results = {}
    unrunnable = {}
    for tag, (name, fac) in arms.items():
        ok, reason = _arm_runnable(fac, task_fac, device)
        if not ok:
            print(f"\n-- arm '{tag}' [{name}] : UNRUNNABLE in this model build -> SKIP --", flush=True)
            print(f"   reason: {reason}", flush=True)
            unrunnable[tag] = {"name": name, "reason": reason}
            continue
        print(f"\n-- arm '{tag}' [{name}] : LR sweep (@seed {seeds[0]}) then {len(seeds)} seeds --",
              flush=True)
        r = sweep_then_seeds(res, f"s3.{tag}", fac, task_fac, base_cfg, device, seeds,
                             grid=grid, out_path=OUT)
        summ = powered_summary(r["accs"], solve_thresh=SOLVE_THRESH)
        arm_results[tag] = {"name": name, "best_lr": r["best_lr"], "lr_grid": r["lr_grid"],
                            "accs": r["accs"], "summary": summ,
                            "params": res[f"s3.{tag}.s{seeds[0]}"]["params"]}
        ci = summ["ci95"]
        print(f"   best_lr={r['best_lr']:.1e}  solve={summ['solve_rate']:.2f}  "
              f"median={summ['median']:.3f}  mean={summ['mean']:.3f} "
              f"CI95=[{ci[0]:.3f},{ci[1]:.3f}]  accs={[round(a,3) for a in r['accs']]}", flush=True)

    assert "baseline" in arm_results, "baseline arm must be runnable (it is the h2h reference)"
    base_accs = arm_results["baseline"]["accs"]

    # ---- h2h vs baseline (each RUNNABLE candidate arm) ------------------------------------------ #
    cand_tags = [t for t in arms if t != "baseline" and t in arm_results]
    h2h_results = {}
    for tag in cand_tags:
        h2h_results[tag] = h2h(arm_results[tag]["accs"], base_accs, margin=MARGIN)

    # ---- Holm-correct the whole comparison FAMILY ----------------------------------------------- #
    # p-value source per arm: superiority p (higher-is-better accuracy).
    fam_pvals = [h2h_results[tag]["superiority"]["p_value"] for tag in cand_tags]
    holm = holm_family(fam_pvals) if fam_pvals else []
    for tag, hr in zip(cand_tags, holm):
        h2h_results[tag]["holm_p_adj"] = hr["p_adj"]
        h2h_results[tag]["holm_reject"] = bool(hr["reject"])

    # ---- scientific verdicts -------------------------------------------------------------------- #
    # surprise-TARGETING wins ONLY if surprise_norm beats BOTH controls (and baseline, Holm-adjusted).
    # Honest about missing controls: a control that is unrunnable in this model build cannot be beaten,
    # so the "beats BOTH controls" claim is recorded as INCONCLUSIVE (not silently True).
    surprise_verdict = {"vs_baseline_holm_reject": False, "vs_random": None, "vs_constant": None,
                        "beats_both_controls": False, "earns_novel_core_slot": False,
                        "controls_available": {"random": "surprise_random" in arm_results,
                                               "constant": "surprise_constant" in arm_results},
                        "note": None}
    if "surprise_norm" in arm_results:
        sn = arm_results["surprise_norm"]["accs"]
        have_random = "surprise_random" in arm_results
        have_const = "surprise_constant" in arm_results
        vs_random = h2h(sn, arm_results["surprise_random"]["accs"], margin=MARGIN) if have_random else None
        vs_const = h2h(sn, arm_results["surprise_constant"]["accs"], margin=MARGIN) if have_const else None
        surprise_verdict["vs_random"] = vs_random["verdict"] if vs_random else "UNAVAILABLE (control unrunnable)"
        surprise_verdict["vs_constant"] = vs_const["verdict"] if vs_const else "UNAVAILABLE (control unrunnable)"
        surprise_verdict["vs_baseline_holm_reject"] = h2h_results["surprise_norm"]["holm_reject"]
        if have_random and have_const:
            beats = bool(vs_random["superiority"]["significant"] and vs_const["superiority"]["significant"])
            surprise_verdict["beats_both_controls"] = beats
            surprise_verdict["earns_novel_core_slot"] = bool(
                beats and surprise_verdict["vs_baseline_holm_reject"])
        else:
            missing = [c for c, ok in surprise_verdict["controls_available"].items() if not ok]
            surprise_verdict["note"] = (
                f"INCONCLUSIVE: control(s) {missing} unrunnable in this model build; "
                f"cannot claim surprise-targeting beats BOTH controls.")
    else:
        surprise_verdict["note"] = "surprise_norm arm unrunnable in this model build."

    inctx_verdict = {
        "vs_baseline": h2h_results["inctx_lr"]["verdict"] if "inctx_lr" in h2h_results else "UNAVAILABLE",
        "vs_baseline_holm_reject": h2h_results["inctx_lr"]["holm_reject"] if "inctx_lr" in h2h_results else False,
    }

    # ---- negative control (integrity canary) ---------------------------------------------------- #
    print("\n-- negative control: two byte-identical Prizma arms must NOT differ --", flush=True)
    nc = negative_control(res, scale, task_fac, base_cfg, device, seeds, out_path=OUT)
    print(f"   p={nc['p_value']:.3f}  significant={nc['significant']}  PASS={nc['pass']}", flush=True)

    # ---- persist the full report ---------------------------------------------------------------- #
    report = {
        "smoke": smoke,
        "scale": list(scale),
        "seeds": list(seeds),
        "grid": list(grid),
        "cap": cap,
        "margin": MARGIN,
        "solve_thresh": SOLVE_THRESH,
        "arms": {t: {"name": ar["name"], "best_lr": ar["best_lr"], "params": ar["params"],
                     "summary": ar["summary"], "accs": ar["accs"], "lr_grid": ar["lr_grid"]}
                 for t, ar in arm_results.items()},
        "unrunnable_arms": unrunnable,
        "h2h_vs_baseline": h2h_results,
        "surprise_verdict": surprise_verdict,
        "inctx_lr_verdict": inctx_verdict,
        "negative_control": nc,
    }
    res["s3_report"] = report
    _save(res, OUT)
    return report


# ------------------------------------------------------------------ presentation --
def _print_summary(report):
    smoke = report["smoke"]
    print("\n" + "=" * 78, flush=True)
    if smoke:
        print("  S3 NOVEL-CORE ABLATION — SMOKE (PLUMBING-ONLY, NOT A RESULT)", flush=True)
    else:
        print("  S3 NOVEL-CORE ABLATION — POWERED RESULTS", flush=True)
    print("=" * 78, flush=True)
    print(f"  scale=d{report['scale'][0]}L{report['scale'][1]}H{report['scale'][2]}  "
          f"seeds={report['seeds']}  cap={report['cap']}  margin={report['margin']}", flush=True)

    print("\n  per-arm (solve-rate / median / mean / CI95 / params):", flush=True)
    for tag, ar in report["arms"].items():
        s = ar["summary"]
        ci = s["ci95"]
        print(f"    {tag:<18} solve={s['solve_rate']:.2f}  median={s['median']:.3f}  "
              f"mean={s['mean']:.3f}  CI95=[{ci[0]:.3f},{ci[1]:.3f}]  "
              f"lr={ar['best_lr']:.1e}  {ar['params']:,}p", flush=True)

    if report.get("unrunnable_arms"):
        print("\n  UNRUNNABLE arms (recorded honestly, NOT faked):", flush=True)
        for tag, info in report["unrunnable_arms"].items():
            print(f"    {tag:<18} {info['reason']}", flush=True)

    print("\n  h2h vs baseline (superiority p / Holm-adj / verdict):", flush=True)
    for tag, hr in report["h2h_vs_baseline"].items():
        print(f"    {tag:<18} p={hr['superiority']['p_value']:.3f}  "
              f"holm_p_adj={hr['holm_p_adj']:.3f}  reject={hr['holm_reject']}  "
              f"-> {hr['verdict']}", flush=True)

    sv = report["surprise_verdict"]
    print("\n  SURPRISE-TARGETING (Lever A) — must beat BOTH controls AND baseline (Holm):", flush=True)
    print(f"    vs baseline (Holm reject): {sv['vs_baseline_holm_reject']}", flush=True)
    print(f"    vs random control:         {sv['vs_random']}", flush=True)
    print(f"    vs constant control:       {sv['vs_constant']}", flush=True)
    print(f"    beats BOTH controls:       {sv['beats_both_controls']}", flush=True)
    print(f"    => EARNS novel-core slot:  {sv['earns_novel_core_slot']}", flush=True)
    if sv.get("note"):
        print(f"    NOTE: {sv['note']}", flush=True)

    iv = report["inctx_lr_verdict"]
    print("\n  IN-CONTEXT LR (Lever G) vs baseline:", flush=True)
    print(f"    verdict: {iv['vs_baseline']}   (Holm reject: {iv['vs_baseline_holm_reject']})", flush=True)

    nc = report["negative_control"]
    print("\n  NEGATIVE CONTROL (integrity canary — identical models must NOT differ):", flush=True)
    print(f"    p={nc['p_value']:.3f}  significant={nc['significant']}  "
          f"=> {'PASS' if nc['pass'] else 'FAIL'}", flush=True)

    if smoke:
        print("\n  [SMOKE] The numbers above are PLUMBING-ONLY (tiny model, few steps/seeds).", flush=True)
        print("  [SMOKE] They prove the pipeline runs end-to-end incl. the negative control.", flush=True)
        print("  [SMOKE] They are NOT a scientific result — do NOT cite.", flush=True)
    print("=" * 78, flush=True)


# ------------------------------------------------------------------------- main --
def main():
    smoke = "--smoke" in sys.argv[1:]
    os.makedirs(RES_DIR, exist_ok=True)
    device = get_device()
    print(f"device={device} torch={torch.__version__} results={OUT} smoke={smoke}", flush=True)

    if smoke:
        # TINY plumbing-only config: small model, tiny MQAR, 2 seeds, 2-LR grid, few steps.
        scale = (64, 2, 2)
        V = 32
        task_fac = lambda: MixedMQAR(vocab=V, max_pairs=8, num_queries=16, gap=0, min_pairs=1)
        seeds = (0, 1)
        grid = (1e-3, 2e-3)
        cap = 400
        eval_every = 200
    else:
        # A100-scale S3 novel-core ablation on MQAR-hard (D=128).
        scale = (128, 4, 4)
        V = 512
        task_fac = lambda: MixedMQAR(vocab=V, max_pairs=128, num_queries=128, gap=0, min_pairs=1)
        seeds = (0, 1, 2, 3, 4)
        from seq.lrsweep import DEFAULT_GRID
        grid = DEFAULT_GRID
        cap = 80000
        eval_every = 2000

    report = run_ablation(scale=scale, task_fac=task_fac, device=device, seeds=seeds,
                          grid=grid, cap=cap, eval_every=eval_every, smoke=smoke)
    _print_summary(report)


if __name__ == "__main__":
    main()
