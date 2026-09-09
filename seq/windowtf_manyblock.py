"""
PR-2026-09-03-15 CLAIM RUNNER — the memory-matched WINDOW-TF control on the 5-block
curriculum (the first head-to-head retention comparison). Executes the FROZEN
pre-registration `docs/preregistry/2026-09-10-windowtf-manyblock-control.md` VERBATIM.

  python seq/windowtf_manyblock.py --smoke        # tiny CPU plumbing run -> the SMOKE ledger
  python seq/windowtf_manyblock.py --powered      # the claim campaign (refuses without CUDA)
  python seq/windowtf_manyblock.py --powered-cpu  # the registered CPU fallback

STREAM: identical to PR-14 (A = text8[0,1M) -> B = shakes[0,90%) -> C = text8[1.1M,2.1M)
-> D = shakes[0,90%) REVISITED -> E = text8[2.1M,3.1M); eval slices A-eval/B-eval/Cret).
CONTROL: seq/transformer.py TFConfig(vocab, d_model=64, n_layers=2, n_heads=4,
max_len=SEG=256) + train_stream/eval_bpc — the PR-07' descriptive WINDOW-TF path VERBATIM
(sliding-window teacher forcing per segment batch; no boundaries given; NO lr schedule —
the control's plain learning rule IS the comparison, disclosed in the doc §2).

BAR (doc §3): P1 — delta = mean(TF bpc_A(post_E) - bpc_A(preB)) minus the repaired
column's A-retention (-0.3226, PR-14 EX cells REUSED read-only from
results/manyblock_PR-2026-09-03-14/powered.json); Welch CI lower >= 0.10. Holm family
= [P1]. Descriptive: the TF's full per-block A/B/Cret trajectories. Straddling CI ->
INCONCLUSIVE (n=10, seeds 5-9, PRE-AUTHORIZED).
"""
from __future__ import annotations

import argparse
import os
import sys

try:
    from .stats import t_sf
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from seq.stats import t_sf

import seq.prizma_lm_claim as plc


REGISTRY_ID = "PR-2026-09-03-15"
LEDDIR = "windowtf_manyblock_PR-2026-09-03-15"
SMOKE_BASENAME = "smoke.json"
POWERED_BASENAME = "powered.json"

ARMS = ("WINDOW-TF",)
CLAIM_SEEDS = plc.CLAIM_SEEDS
LR_FROZEN = 3e-3                          # the PR-07' WINDOW-TF lr (carried VERBATIM)
P1_MARGIN = 0.10
PR14_EX_A_RETENTION = -0.3226             # the reused baseline's registered A-retention mean
ALPHA = plc.ALPHA
BLOCKS = ("A", "B", "C", "D", "E")
POWERED_CPU_FALLBACK_NOTE = (
    "POWERED VIA THE REGISTERED CPU FALLBACK (PR-15 doc §5; precedent = PR-08 addendum "
    "2026-09-08 #2 + the PR-09..PR-14 powered-cpu executions).")


def _results_root() -> str:
    root = os.environ.get("PRIZMA_RESULTS", os.path.join(os.path.dirname(__file__), "..", "results"))
    return os.path.abspath(root)


def _default_results_path(smoke: bool) -> str:
    return os.path.join(_results_root(), LEDDIR, SMOKE_BASENAME if smoke else POWERED_BASENAME)


def _pr14_powered_path() -> str:
    return os.path.join(_results_root(), "manyblock_PR-2026-09-03-14", POWERED_BASENAME)


def resolve_results_path(explicit=None, *, smoke: bool = False, force_smoke_path: bool = False) -> str:
    path = explicit if explicit else _default_results_path(smoke)
    if smoke and not force_smoke_path:
        powered = _default_results_path(smoke=False)
        if os.path.normcase(os.path.abspath(path)) == os.path.normcase(os.path.abspath(powered)):
            raise SystemExit(
                f"refusing: a --smoke run was pointed at the powered ledger ({powered}). Pass "
                f"--out <path> or --force-smoke-path (PR-2026-09-03-15 §5 ledger separation).")
    return path


def require_cuda(has_cuda: bool) -> None:
    if not has_cuda:
        raise SystemExit(
            "refusing: --powered executes the PR-2026-09-03-15 claim campaign and requires a "
            "CUDA device. Use --smoke, or --powered-cpu for the registered CPU fallback.")


needs_cuda = plc.needs_cuda


# ==================================================================== PURE: verdict (§3) =========
def claim_verdict(tf, pr14_ex_retentions, *, alpha=ALPHA):
    """Frozen §3 verdict. tf = per-seed WINDOW-TF records; pr14_ex_retentions = the reused
    PR-14 EX per-seed A-retention values (bpc_A(post_E) - bpc_A(preB))."""
    tf_ret = [r["bpc_A_postE"] - r["bpc_A_preB"] for r in tf]
    # delta = mean(TF retention) - mean(column retention); positive = TF forgets MORE
    p1 = plc._welch_margin(tf_ret, list(pr14_ex_retentions), P1_MARGIN)
    p1["tf_retention_per_seed"] = tf_ret
    p1["column_retention_mean"] = (sum(pr14_ex_retentions) / len(pr14_ex_retentions)
                                   if pr14_ex_retentions else None)
    p1["tf_retention_mean"] = sum(tf_ret) / len(tf_ret)
    p1["note"] = ("delta = mean(TF A-retention) - mean(column A-retention); positive = the "
                  "TF forgets MORE. PASS iff CI lower >= 0.10 (doc §3).")
    # CI lower of delta (two-sided 95%): the PASS/FAIL/straddle rule uses the CI position.
    if p1["ci"][0] >= P1_MARGIN and p1["p_raw"] < alpha:
        p1["status"] = "PASS"
    elif p1["ci"][1] < P1_MARGIN:
        p1["status"] = "FAIL"
    else:
        p1["status"] = "INCONCLUSIVE"
    if p1["status"] == "PASS":
        outcome = "CLAIMED"
        verdict_text = ("CLAIMED — the sliding-window control cannot hold domain A across the "
                        "5-block curriculum: its retention is worse than the repaired column's "
                        "by a CI-established margin above 0.10 bpc (doc §4).")
    elif p1["status"] == "FAIL":
        outcome = "NEGATIVE"
        verdict_text = ("NEGATIVE — the WINDOW-TF holds A comparably or better; the repaired "
                        "column's retention advantage does not generalize to the 5-block "
                        "curriculum (doc §4).")
    else:
        outcome = "INCONCLUSIVE"
        verdict_text = ("INCONCLUSIVE — the retention gap CI straddles the 0.10 margin; the "
                        "pre-authorized n=10 extension applies (doc §4).")
    branches = []
    if outcome == "INCONCLUSIVE":
        branches.append("§4 (INCONCLUSIVE): n=10 (seeds 5-9) PRE-AUTHORIZED; never PASS.")
    return {"outcome": outcome, "verdict": verdict_text, "branches": branches,
            "P1": p1, "alpha": alpha}


# ================================================================================= runner ========
def run_cell(seed, blocks_data, evals, lr, *, post_a_lr=None):
    """PR-15/PR-17 cell: the WINDOW-TF trains the 5 blocks sequentially (sliding-window,
    the PR-07' path verbatim); A/B/Cret evals after every block.
    post_a_lr (PR-2026-09-03-17, guarded DEFAULT-OFF): when set, the POST-A blocks train
    with this backbone lr (the L1 dose) while block A keeps `lr` — the scheduled control.
    None => every block trains at `lr` (byte-identical PR-15 behavior)."""
    import time

    import torch
    from seq import blockdrift_claim as claim

    t0 = time.time()
    model, _ = claim.build_model(vocab_size := evals["vocab"], seed)
    rec = {"config": ("WINDOW-TF+scheduled" if post_a_lr is not None else "WINDOW-TF"),
           "seed": seed, "lr": lr}

    def eval_all():
        return {"bpc_A": claim.eval_bpc(model, evals["A"][0], evals["A"][1]),
                "bpc_B": claim.eval_bpc(model, evals["B"][0], evals["B"][1]),
                "bpc_Cret": claim.eval_bpc(model, evals["Cret"][0], evals["Cret"][1])}

    for tag in BLOCKS:
        bx, by = blocks_data[tag]
        block_lr = lr if (post_a_lr is None or tag == "A") else post_a_lr
        claim.train_stream(model, bx, by, block_lr, seed)
        ev = eval_all()
        rec[f"traj_{tag}"] = ev
        rec[f"block_{tag}"] = {"wall_note": "sequential training, no schedule (control)"}

    rec["bpc_A_preB"] = rec["traj_A"]["bpc_A"]
    rec["bpc_A_postE"] = rec["traj_E"]["bpc_A"]
    rec["bpc_A_retention"] = rec["bpc_A_postE"] - rec["bpc_A_preB"]
    rec["bpc_B_postB"] = rec["traj_B"]["bpc_B"]
    rec["bpc_B_postC"] = rec["traj_C"]["bpc_B"]
    rec["bpc_B_postD"] = rec["traj_D"]["bpc_B"]
    rec["bpc_B_postE"] = rec["traj_E"]["bpc_B"]
    rec["b_degradation_B_postC"] = rec["bpc_B_postC"] - rec["bpc_B_postB"]
    rec["b_recovery_postD"] = rec["bpc_B_postC"] - rec["bpc_B_postD"]   # descriptive
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def attribution(sched_damage, plain_damage, col_damage, *, alpha=ALPHA):
    """PURE PR-2026-09-03-17 attribution (doc §3-§4): C1 = mean(plain) - mean(sched) — the
    schedule effect on the control; C2 = mean(sched) - mean(col) — the tissue's residual.
    Both CIs are two-sided 95% Welch on the named deltas; the attributions are
    CI-positional (established iff CI lower >= 0.25; ~0 iff CI upper < 0.25; straddle =
    unresolved). Returns the pre-committed attribution pair."""
    # delta directions per the doc: C1 = mean(PLAIN) - mean(SCHED); C2 = mean(SCHED) - mean(COL)
    c1 = plc._welch_margin(sched_damage, plain_damage, 0.25)
    c2 = plc._welch_margin(col_damage, sched_damage, 0.25)

    def _pos(ci):
        if ci[0] >= 0.25:
            return "ESTABLISHED"
        if ci[1] < 0.25:
            return "~0 (CI upper < 0.25)"
        return "UNRESOLVED (CI straddles 0.25)"

    c1_pos, c2_pos = _pos(c1["ci"]), _pos(c2["ci"])
    if c1_pos == "ESTABLISHED" and c2_pos != "ESTABLISHED":
        attrib = "SCHEDULE-CARRIED"
    elif c1_pos != "ESTABLISHED" and c2_pos == "ESTABLISHED":
        attrib = "TISSUE-CARRIED"
    elif c1_pos == "ESTABLISHED" and c2_pos == "ESTABLISHED":
        attrib = "COMPOUND"
    else:
        attrib = "UNRESOLVED at n=5 (doc §4: the n=10 seed extension is pre-authorized)"
    return {"C1_schedule_effect": {"delta_mean": c1["delta"], "ci": c1["ci"],
                                   "position": c1_pos,
                                   "note": "mean(PLAIN-TF damage) - mean(SCHED-TF damage)"},
            "C2_tissue_residual": {"delta_mean": c2["delta"], "ci": c2["ci"],
                                   "position": c2_pos,
                                   "note": "mean(SCHED-TF damage) - mean(COLUMN damage)"},
            "attribution": attrib}


def run(*, smoke: bool, results_path=None, force_smoke_path: bool = False,
        powered_cpu: bool = False, post_a_lr=None, ledger_dir=None):
    path = os.path.join(_results_root(), ledger_dir,
                        SMOKE_BASENAME if smoke else POWERED_BASENAME)         if ledger_dir else resolve_results_path(results_path, smoke=smoke,
                                                force_smoke_path=force_smoke_path)

    import torch

    from seq import blockdrift_claim as claim
    from seq.gpu_harness import load_results, _save
    from seq.recall_gate import archive_run

    torch.set_num_threads(int(os.environ.get("BENCH_THREADS", "8")))

    if smoke:
        smoke_chars = 64 * plc.SEG
        seeds = (0,)
    else:
        if needs_cuda("powered-cpu" if powered_cpu else "powered"):
            require_cuda(torch.cuda.is_available())
        smoke_chars = None
        seeds = CLAIM_SEEDS

    # ---------------- the 5-block stream (PR-14's exact slices) -------------------------------
    A_all, B_all = claim.fetch_corpora()
    shakes_train_end = int(len(B_all) * 0.9)
    text_blocks = {"A": A_all[0:1_000_000], "C": A_all[1_100_000:2_100_000],
                   "E": A_all[2_100_000:3_100_000]}
    shake_blocks = {"B": B_all[:shakes_train_end], "D": B_all[:shakes_train_end]}
    all_text = "".join([text_blocks["A"], shake_blocks["B"], text_blocks["C"],
                        text_blocks["E"], A_all[1_000_000:1_100_000],
                        B_all[shakes_train_end:], A_all[900_000:1_000_000]])
    chars = sorted(set(all_text))
    vocab = {c: i for i, c in enumerate(chars)}
    V = len(vocab)

    def segs(t):
        return claim.make_segments(t, vocab)

    blocks_data = {}
    for tag in BLOCKS:
        text = text_blocks[tag] if tag in text_blocks else shake_blocks[tag]
        bx, by = segs(text)
        if smoke_chars:
            bx, by = bx[:smoke_chars // plc.SEG], by[:smoke_chars // plc.SEG]
        blocks_data[tag] = (bx, by)
    evals = {"vocab": V,
             "A": segs(A_all[1_000_000:1_100_000]),
             "B": segs(B_all[shakes_train_end:]),
             "Cret": segs(A_all[900_000:1_000_000])}

    # ---------------- the reused PR-14 baseline (fail-loud if missing) ------------------------
    pr14_path = _pr14_powered_path()
    if not os.path.isfile(pr14_path):
        raise SystemExit(
            f"refusing: the PR-14 powered ledger is MISSING at {pr14_path} — P1's baseline "
            "(the repaired column's A-retention) is REUSED from it (doc §2). Run this where "
            "the PR-14 ledger exists.")
    pr14 = load_results(pr14_path)
    pr14_ex_ret = [pr14[f"claim.EX.s{s}"]["bpc_A_postE"] - pr14[f"claim.EX.s{s}"]["bpc_A_preB"]
                   for s in CLAIM_SEEDS]

    res = load_results(path)
    res["meta"] = {
        "registry": REGISTRY_ID, "smoke": bool(smoke),
        "lane": "CLAIM (frozen pre-registration)" if not smoke else "SMOKE (plumbing only)",
        "doc": "docs/preregistry/2026-09-10-windowtf-manyblock-control.md",
        "arms": list(ARMS), "claim_seeds": list(seeds),
        "blocks": list(BLOCKS),
        "control": {"config": "TFConfig(vocab, d64, 2L, H4, window=SEG=256) — the PR-07' "
                              "WINDOW-TF path verbatim",
                    "lr": LR_FROZEN,
                    "post_a_lr": post_a_lr,
                    "post_a_lr_note": (None if post_a_lr is None else
                                       "PR-2026-09-03-17 scheduled control: post-A blocks "
                                       "train at this backbone lr (the L1 dose); block A "
                                       "keeps the base lr"),
                    "note": ("no lr schedule, no tissue — the control's plain learning rule "
                             "IS the comparison (doc §2, disclosed)")},
        "stream_lengths": {t: len(text_blocks.get(t, shake_blocks.get(t, "")))
                           for t in BLOCKS},
        "vocab": V, "threads": torch.get_num_threads(),
        "bars": {"P1_margin": P1_MARGIN, "alpha": ALPHA,
                 "baseline": {"source": pr14_path,
                              "cells": "claim.EX.s0..s4 (reused, disclosed)",
                              "a_retention_mean": PR14_EX_A_RETENTION}},
        "t_isf_convention": "UPPER-TAIL p in (0, 0.5] (the PR-03 lesson)",
    }
    if powered_cpu:
        res["meta"]["powered_cpu"] = True
        res["meta"]["compute_fallback_note"] = POWERED_CPU_FALLBACK_NOTE
    _save(res, path)
    print(f"[pr15] stream: A/B/C/D/E segs="
          f"{[blocks_data[t][0].shape[0] for t in BLOCKS]}; vocab={V}; smoke={smoke}; "
          f"lr={LR_FROZEN} (PR-07' WINDOW-TF); baseline={pr14_path}", flush=True)

    # ---------------- the control arm x seeds --------------------------------------------------
    for arm in ARMS:
        for seed in seeds:
            cellkey = f"claim.{arm}.s{seed}"
            from seq.gpu_harness import config_fingerprint
            cfgsig = config_fingerprint({
                "registry": REGISTRY_ID, "leg": "claim", "arm": arm, "seed": seed,
                "lr": LR_FROZEN, "smoke": bool(smoke), "vocab": V,
                "tf_config": {"d_model": 64, "n_layers": 2, "n_heads": 4,
                              "max_len": plc.SEG},
                "post_a_lr": post_a_lr,   # PR-17: the schedule is IN the fingerprint
                "stream_lengths": res["meta"]["stream_lengths"]})
            prior = res.get(cellkey)
            if isinstance(prior, dict) and prior.get("cfgsig") == cfgsig and prior.get("complete"):
                print(f"[pr15] {arm} seed {seed}: resumed from ledger "
                      f"(bpc_A_retention={prior['bpc_A_retention']:.3f})", flush=True)
                continue
            if isinstance(prior, dict):
                raise SystemExit(f"cell {cellkey} exists at a foreign config fingerprint "
                                 f"({prior.get('cfgsig')} != {cfgsig}) — refusing to resume")
            t0 = __import__("time").time()
            rec = run_cell(seed, blocks_data, evals, LR_FROZEN, post_a_lr=post_a_lr)
            rec.update({"arm": arm, "seed": seed, "lr": LR_FROZEN, "cellkey": cellkey,
                        "cfgsig": cfgsig, "complete": True,
                        "wall_s": round(__import__("time").time() - t0, 1)})
            res[cellkey] = rec
            _save(res, path)
            print(f"[pr15] {arm} seed {seed}: A: preB={rec['bpc_A_preB']:.3f} "
                  f"postE={rec['bpc_A_postE']:.3f} retention={rec['bpc_A_retention']:.3f} | "
                  f"B: postB={rec['bpc_B_postB']:.3f} postC={rec['bpc_B_postC']:.3f} "
                  f"postD={rec['bpc_B_postD']:.3f} postE={rec['bpc_B_postE']:.3f} "
                  f"wall={rec['wall_s']}s", flush=True)

    # ---------------- RETENTION: archive BEFORE any verdict -----------------------------------
    raw_archive = archive_run(res, label=f"windowtf-{REGISTRY_ID}")
    res.setdefault("meta", {})["raw_archive"] = raw_archive

    # ---------------- the frozen §3 verdict (claim mode) --------------------------------------
    report = {"registry": REGISTRY_ID, "smoke": bool(smoke), "arms": list(ARMS),
              "claim_seeds": list(seeds), "lr": LR_FROZEN,
              "raw_archive": raw_archive,
              "cells": {k: res[k] for k in sorted(res) if k.startswith("claim.")}}
    if powered_cpu:
        report["powered_cpu"] = True
    if post_a_lr is not None:
        report["post_a_lr"] = post_a_lr          # PR-2026-09-03-17 (recorded when set)
    if not smoke:
        tf = [res[f"claim.WINDOW-TF.s{s}"] for s in CLAIM_SEEDS]
        verdict = claim_verdict(tf, pr14_ex_ret, alpha=ALPHA)
        report["verdict"] = verdict
        res["verdict"] = verdict
        if post_a_lr is not None:
            # PR-2026-09-03-17 attribution: SCHED vs the plain TF (PR-15 ledger) and vs
            # the column (PR-14 EX ledger, already loaded above for the baseline reuse).
            pr15_path = os.path.join(_results_root(), "windowtf_manyblock_PR-2026-09-03-15",
                                     POWERED_BASENAME)
            if not os.path.isfile(pr15_path):
                raise SystemExit(
                    "refusing: the PR-15 powered ledger is MISSING at "
                    f"{pr15_path} — C1's plain-TF damage side comes from it (doc §2).")
            pr15 = load_results(pr15_path)
            sched_damage = [res[f"claim.WINDOW-TF.s{s}"]["b_degradation_B_postC"]
                            for s in CLAIM_SEEDS]
            plain_damage = [pr15[f"claim.WINDOW-TF.s{s}"]["b_degradation_B_postC"]
                            for s in CLAIM_SEEDS]
            col_damage = [pr14[f"claim.EX.s{s}"]["b_degradation_B_postC"]
                          for s in CLAIM_SEEDS]
            att = attribution(sched_damage, plain_damage, col_damage)
            report["attribution"] = att
            res["attribution"] = att
            print(f"[pr15->pr17] C1 schedule effect: {att['C1_schedule_effect']}", flush=True)
            print(f"[pr15->pr17] C2 tissue residual: {att['C2_tissue_residual']}", flush=True)
            print(f"[pr17] ATTRIBUTION: {att['attribution']}", flush=True)
    res["report"] = report
    _save(res, path)
    _print_report(report, path, smoke=smoke)
    return report


def _print_report(report, path, *, smoke):
    print("\n" + "=" * 78, flush=True)
    print(f"  {REGISTRY_ID} — {'SMOKE (PLUMBING-ONLY)' if smoke else 'POWERED CLAIM CAMPAIGN'}",
          flush=True)
    print("=" * 78, flush=True)
    for key, c in report["cells"].items():
        print(f"    {key:<30} A: preB={c['bpc_A_preB']:.3f} postE={c['bpc_A_postE']:.3f} "
              f"retention={c['bpc_A_retention']:.3f} | B: postB={c['bpc_B_postB']:.3f} "
              f"postC={c['bpc_B_postC']:.3f} postD={c['bpc_B_postD']:.3f} "
              f"postE={c['bpc_B_postE']:.3f} wall={c['wall_s']}s", flush=True)
    if not smoke:
        v = report["verdict"]
        p1 = v["P1"]
        print(f"    P1: delta(TF-column retention) mean={p1['delta']:.4f} "
              f"CI=[{p1['ci'][0]:.4f},{p1['ci'][1]:.4f}] (bar: CI lower >= {p1['margin']}) "
              f"p_raw={p1['p_raw']:.4f} -> {p1['status']}", flush=True)
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
    p = argparse.ArgumentParser(
        prog="windowtf_manyblock",
        description="PR-2026-09-03-15 claim runner (the WINDOW-TF control on the 5-block "
                    "curriculum). --smoke = tiny CPU plumbing run; --powered = the frozen "
                    "claim campaign (requires CUDA); --powered-cpu = the registered CPU "
                    "fallback. An unknown flag is rejected without launching anything.")
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
    p.add_argument("--post-a-lr", type=float, default=None,
                   help="PR-2026-09-03-17 guarded lever: the POST-A blocks' backbone lr "
                        "(the scheduled control uses 7.5e-4). Default None = byte-identical "
                        "PR-15 behavior.")
    p.add_argument("--ledger-dir", default=None,
                   help="ledger subdirectory under $PRIZMA_RESULTS (default None = the "
                        "PR-15 dir; the PR-17 run uses windowtf_sched_PR-2026-09-03-17 so "
                        "the PR-15 ledger is never written)")
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(argv)
    run(smoke=args.smoke, results_path=args.out, force_smoke_path=args.force_smoke_path,
        powered_cpu=args.powered_cpu, post_a_lr=args.post_a_lr, ledger_dir=args.ledger_dir)


if __name__ == "__main__":
    main()
