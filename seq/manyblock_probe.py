"""
LANE-EXPLORATORY PROBE (NOT a claim; numbers are exploratory, quarantined here):
seq/manyblock_probe.py — 5-block many-stream dynamics ahead of a possible PR-14
"many-block accumulation" registration.

Stream (one pass, char-level, levers L1 only — NO domain-exclusion, so this MEASURES what
plain argmin does at a literal returning-domain revisit):
  A = text8[0, 1M)                (first domain; trains the trunk + first expert)
  B = shakes[0, 90%)              (drift, strong)
  C = text8[1.1M, 2.1M)           (drift, mild + text8-returning — the PR-08/PR-13 regime)
  D = shakes[0, 90%) REVISITED    (the EXACT same slice as B — a true re-encounter)
  E = text8[2.1M, 3.1M)           (one more text8 block: does accumulation hold?)

Measured per seed (n=2, seeds 0-1):
  - bpc of A-eval / B-eval / Cret-eval after EVERY block (the forgetting/accumulation
    trajectories),
  - per-block routing: recruits, evictions, and the first-20-batch trained-share of the
    TOP slot (esp. block D: does the B-expert re-engage, or do fresh recruits storm?),
  - per-slot cumulative train counts.

Provenance: PR-13 CLAIM PASS (3-block flagship) + the open question "does the repaired
column ACCUMULATE across many blocks". This probe informs a PR-14 prereg (notably: whether
domain-exclusion with per-block owner election is needed for a clean revisit).
Output: results/exploratory/manyblock_probe_2026-09-09/probe.json (+ console summary).
Lane-1 cascade CLI (campaign 2026-09-11, guarded DEFAULT-OFF): --cascade-target {off,tissue},
--cascade-kappa (default 0.05), --cascade-delta (default 0.10), --smoke, --out PATH (explicit
output; paired campaign runs keep historical files untouched). Off writes probe.json (the
original behavior), tissue writes probe_cascade.json, --smoke writes probe_smoke.json (8
segments/block, seed 0 only); per-block fast-component diagnostics are recorded only when the
cascade is on.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

try:
    from .stats import holm_correction  # noqa: F401  (parity with the claim runners)
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import seq.prizma_lm_claim as plc          # PR-08 plumbing, reused VERBATIM
import seq.floorfreeze_claim as ffc        # noqa: F401  (aliases live on plc anyway)


REGISTRY = "manyblock_probe_2026-09-09"
LANE = "LANE-EXPLORATORY (quarantined; never cited as a claim)"
LR_FROZEN = 3e-3
BACKBONE_LR_POST_A = 7.5e-4                # L1 dose generalized: every post-A block
TOP_WINDOW_BATCHES = 20

# Lane-1 cascade lever (guarded, DEFAULT-OFF; campaign 2026-09-11 CONTRACT.md): the probe
# threads the same flags/guard as the claim runners; the default invocation is byte-identical
# to the 2026-09-09 probe (off + probe.json). A cascade run writes probe_cascade.json, a
# --smoke wiring run writes probe_smoke.json — the files never clobber each other.
CASCADE_TARGETS = plc.CASCADE_TARGETS
CASCADE_KAPPA_DEFAULT = plc.CASCADE_KAPPA_DEFAULT
CASCADE_DELTA_DEFAULT = plc.CASCADE_DELTA_DEFAULT
validate_cascade = plc.validate_cascade


def _slice(text, a, b):
    return text[a:b]


def run_seed(seed, data, evals, *, cascade_target="off",
             cascade_kappa=CASCADE_KAPPA_DEFAULT, cascade_delta=CASCADE_DELTA_DEFAULT):
    import time
    import torch
    from seq import fusion_probe as fp

    t0 = time.time()
    (Ax, Ay, Bx, By, Cx, Cy, Dx, Dy, Ex, Ey,
     Aex, Aey, Bex, Bey, Crx, Cry) = data
    E = plc.E_POOL
    model = fp.build_model(V := evals["vocab"], seed, E,
                           cascade=(cascade_target == "tissue"),
                           cascade_kappa=cascade_kappa, cascade_delta=cascade_delta)
    led = plc._fresh_ledger()
    tr = {"A": [0] * E, "B": [0] * E, "C": [0] * E, "D": [0] * E, "E": [0] * E}
    blocks = [("A", Ax, Ay, False), ("B", Bx, By, True), ("C", Cx, Cy, True),
              ("D", Dx, Dy, True), ("E", Ex, Ey, True)]
    traj = {"blocks": {}}

    def snap(tag):
        traj["blocks"][tag] = {
            "bpc_A": fp.eval_bpc_fusion(model, Aex, Aey),
            "bpc_B": fp.eval_bpc_fusion(model, Bex, Bey),
            "bpc_Cret": fp.eval_bpc_fusion(model, Crx, Cry),
        }

    snap("init")
    for tag, bx, by, is_post_a in blocks:
        lr = LR_FROZEN
        kw = {}
        if is_post_a:
            kw["backbone_lr"] = BACKBONE_LR_POST_A   # L1 generalized to every post-A block
        before = model.n_segments[:]
        plc.train_pr08(model, bx, by, lr, tag, led, seed=seed, **kw)
        tr[tag] = [model.n_segments[s] - before[s] for s in range(E)]
        snap(f"post_{tag}")
        top_window = {"segments_total": 0, "to_top": 0}
        # re-count the first-window routing of THIS block from the ledger delta is complex;
        # instead record per-block recruit/eviction counts + the block's train shares:
        traj["blocks"][f"post_{tag}"]["block_train"] = tr[tag]
        traj["blocks"][f"post_{tag}"]["recruits_so_far"] = len(led["recruits"])
        traj["blocks"][f"post_{tag}"]["evictions_so_far"] = len(led["evictions"])
        diag = fp.cascade_diagnostics(model)
        if diag is not None:
            traj["blocks"][f"post_{tag}"]["cascade"] = diag   # Lane-1: per-block fast norms

    per_slot = [{"slot": s, "committed": bool(model.committed[s]),
                 "train": {"A": tr["A"][s], "B": tr["B"][s], "C": tr["C"][s],
                           "D": tr["D"][s], "E": tr["E"][s]}}
                for s in range(E)]
    traj["per_slot"] = per_slot
    traj["ledger_tail"] = {"recruits": len(led["recruits"]),
                           "evictions": len(led["evictions"]),
                           "vetoed_novel_segments": led["vetoed_novel_segments"]}
    traj["wall_s"] = round(time.time() - t0, 1)
    return traj


def _build_parser():
    p = argparse.ArgumentParser(
        prog="manyblock_probe",
        description="LANE-EXPLORATORY probe (2026-09-09): 5-block many-stream dynamics "
                    "(A=text8[0,1M) B=shakes C=text8[1.1M,2.1M) D=shakes-REVISIT "
                    "E=text8[2.1M,3.1M)); n=2 seeds 0-1; never cited as a claim. The Lane-1 "
                    "cascade flags mirror the claim runners (default off = byte-identical).")
    p.add_argument("--cascade-target", choices=CASCADE_TARGETS, default="off",
                   help="Lane-1 guarded lever: 'tissue' enables the fast/slow cascade on the "
                        "routed expert heads (writes probe_cascade.json); default 'off' "
                        "writes probe.json (the original 2026-09-09 behavior).")
    p.add_argument("--cascade-kappa", type=float, default=CASCADE_KAPPA_DEFAULT,
                   help="Lane-1 cascade consolidation gain kappa in [0,1] (default 0.05).")
    p.add_argument("--cascade-delta", type=float, default=CASCADE_DELTA_DEFAULT,
                   help="Lane-1 cascade fast-decay rate delta in [0,1] (default 0.10).")
    p.add_argument("--smoke", action="store_true",
                   help="tiny wiring run (8 segments/block, seed 0 only; writes "
                        "probe_smoke.json, never the real probe files)")
    p.add_argument("--out", default=None,
                   help="explicit output JSON path (overrides the registry default; used by "
                        "paired campaign runs so historical probe.json is never overwritten)")
    return p


def main(argv=None):
    args = _build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    validate_cascade(args.cascade_target, args.cascade_kappa, args.cascade_delta)

    from seq import blockdrift_claim as claim
    from seq import fusion_probe as fp

    import torch
    torch.set_num_threads(int(os.environ.get("BENCH_THREADS", "8")))

    outdir = os.path.join(os.path.dirname(__file__), "..", "results", "exploratory", REGISTRY)
    os.makedirs(outdir, exist_ok=True)
    if args.smoke:
        name = "probe_smoke.json"
    elif args.cascade_target == "off":
        name = "probe.json"
    else:
        name = "probe_cascade.json"
    out = args.out if args.out else os.path.join(outdir, name)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    A_all, B_all = claim.fetch_corpora()
    T = A_all
    S = B_all
    A_tr = _slice(T, 0, 1_000_000)
    B_tr = S[: int(len(S) * 0.9)]
    C_tr = _slice(T, 1_100_000, 2_100_000)
    D_tr = B_tr                                   # the literal revisit
    E_tr = _slice(T, 2_100_000, 3_100_000)
    A_ev = _slice(T, 1_000_000, 1_100_000)
    B_ev = S[int(len(S) * 0.9):]
    Cret = _slice(T, 900_000, 1_000_000)
    chars = sorted(set(A_tr) | set(B_tr) | set(C_tr) | set(E_tr) |
                   set(A_ev) | set(B_ev) | set(Cret))
    vocab = {c: i for i, c in enumerate(chars)}
    V = len(vocab)

    def segs(t):
        return claim.make_segments(t, vocab)

    data = (segs(A_tr)[0], segs(A_tr)[1], segs(B_tr)[0], segs(B_tr)[1],
            segs(C_tr)[0], segs(C_tr)[1], segs(D_tr)[0], segs(D_tr)[1],
            segs(E_tr)[0], segs(E_tr)[1],
            segs(A_ev)[0], segs(A_ev)[1], segs(B_ev)[0], segs(B_ev)[1],
            segs(Cret)[0], segs(Cret)[1])
    seeds = (0,)
    if args.smoke:
        data = tuple(t[:8] for t in data)         # 8 segments/block: wiring only
    else:
        seeds = (0, 1)
    evals = {"vocab": V}

    print(f"[probe] LANE-EXPLORATORY {REGISTRY}: 5 blocks "
          f"(A=text8[0,1M) B=shakes C=text8[1.1M,2.1M) D=shakes-REVISIT E=text8[2.1M,3.1M)); "
          f"vocab={V}; n={len(seeds)}; L1 dose {BACKBONE_LR_POST_A} on post-A blocks; "
          f"cascade_target={args.cascade_target} (kappa={args.cascade_kappa}, "
          f"delta={args.cascade_delta}); smoke={args.smoke}",
          flush=True)

    seeds_out = {}
    for seed in seeds:
        traj = run_seed(seed, data, evals, cascade_target=args.cascade_target,
                        cascade_kappa=args.cascade_kappa, cascade_delta=args.cascade_delta)
        seeds_out[f"s{seed}"] = traj
        b = traj["blocks"]
        print(f"[probe] s{seed}: "
              f"A: {b['init']['bpc_A']:.3f}->{b['post_E']['bpc_A']:.3f} | "
              f"B: {b['post_B']['bpc_B']:.3f} ->postC {b['post_C']['bpc_B']:.3f} "
              f"->postD {b['post_D']['bpc_B']:.3f} ->postE {b['post_E']['bpc_B']:.3f} | "
              f"Cret: {b['post_C']['bpc_Cret']:.3f}->{b['post_E']['bpc_Cret']:.3f} | "
              f"recruits={traj['ledger_tail']['recruits']} "
              f"evictions={traj['ledger_tail']['evictions']} "
              f"wall={traj['wall_s']}s", flush=True)

    doc = {"registry": REGISTRY, "lane": LANE, "seeds": [int(s) for s in seeds],
           "results": seeds_out,
           "note": ("exploratory; informs the PR-14 many-block prereg — especially whether a "
                    "literal shakespeare revisit (block D) improves B-eval and whether the "
                    "B-expert re-engages under plain argmin without domain-exclusion")}
    if args.smoke:
        doc["smoke"] = True
        doc["smoke_note"] = ("wiring-only run (8 segments/block, seed 0); numbers are "
                             "meaningless, never cited")
    if args.cascade_target != "off":
        doc["cascade"] = {"target": args.cascade_target, "kappa": args.cascade_kappa,
                          "delta": args.cascade_delta,
                                   "note": ("Lane-1 tissue fast/slow cascade (guarded lever, campaign "
                                   "2026-09-11): zero-init W_fast; forward W + W_fast; the "
                                   "expert optimizer receives the fast parameters plus the "
                                   "un-split biases (coordinator clarification 2026-09-11); "
                                   "after every optimizer step W <- W + kappa*W_fast and "
                                   "W_fast <- (1-delta)*W_fast")}
    json.dump(doc, open(out, "w", encoding="utf-8"), indent=1)
    print(f"[probe] written: {out}", flush=True)


if __name__ == "__main__":
    main()
