"""
PR-2026-09-03-14 CLAIM RUNNER — many-block accumulation with a true revisit (5 blocks,
3 drift boundaries). Executes the FROZEN pre-registration
`docs/preregistry/2026-09-09-manyblock-accumulation.md` VERBATIM. One script, three modes
(the prizma_lm_claim pattern, mirrored):

  python seq/manyblock_claim.py --smoke        # tiny CPU plumbing run -> the SMOKE ledger
  python seq/manyblock_claim.py --powered      # the claim campaign (refuses without CUDA)
  python seq/manyblock_claim.py --powered-cpu  # the registered CPU fallback

STREAM (one pass, char-level, ~5.1M chars):
  A = text8[0, 1M)          B = shakes[0, 90%)      C = text8[1.1M, 2.1M)
  D = shakes[0, 90%) REVISITED (the EXACT B slice)  E = text8[2.1M, 3.1M)

LEVERS: L1 — backbone lr 7.5e-4 on EVERY post-A block (generalized from PR-10/PR-11's
C-only dose). L2 (EX arm only) — domain-exclusion with PER-BLOCK DOMAIN-OWNER election:
at each post-A block start the owner slot is the committed slot with the maximum
cumulative trained-segment count in the block's DOMAIN (text8 = A+C+E shares; shakespeare
= B+D shares); protected slots = committed MINUS the owner; the guarded route_pr08
branches redirect protected would-be segments to `boundary["owner_slot"]` and suppress
recruits rather than self-evict the owner (the PR-13 refinement, carried). Attribute
absent (PLAIN arm) => the byte-identical PR-08/PR-11 path.

ARMS (2): PLAIN (L1 only, plain argmin — the probe's regime) and EX (L1 + L2).
n=5 seeds 0-4. CANARY: the A and B phases are bit-identical ACROSS arms (bpc_A_preB,
bpc_A_postB, bpc_B_postB — divergence begins at C, the first protected block); asserted
per cell in claim mode (fail-loud ABORT) and in smoke.

BARS (doc §3, exact):
  P1 (revisit recovery, EX): diffs = bpc_B(post_C) - bpc_B(post_D); mean >= 0.10 with
      CI lower >= 0.10 (Holm family member).
  P2 (re-engagement ledger, EX): frac of D's first-20-batch trained segments routed to
      the B-expert (the shakespeare owner) >= 0.5 — exact means (PR-08 B3a form).
  P3 (accumulation, EX): diffs = bpc_A(post_E) - bpc_A(post_A); CI upper <= 0.05
      (Holm family member).
  G1 (exclusion must not cost the revisit): Welch delta = mean(PLAIN bpc_B post_D) -
      mean(EX bpc_B post_D) vs margin -0.10, direction 'above': PASS iff CI lower >= -0.10
      AND Holm; FAIL iff CI upper < -0.10; straddle -> INCONCLUSIVE (Holm family member).
  G2 (text8 continuity, EX): diffs = bpc_Cret(post_E) - bpc_Cret(post_C); CI upper <= 0.05
      (Holm family member).
  Holm family = [P1, P3, G1, G2] (P2 is means-based, untested). Reported, never gated:
  both arms' full per-block trajectories, per-block churn + redirect/suppression counters.

Overall (doc §4): any INCONCLUSIVE -> INCONCLUSIVE (n=10, seeds 5-9, PRE-AUTHORIZED);
P1+P2+P3+G1+G2 all PASS -> CLAIMED; P1 FAIL -> NEGATIVE (no recovery); P2 FAIL ->
NEGATIVE (re-encounter real, ledger not); P3 FAIL -> NEGATIVE-GUARD; only guards FAIL ->
NEGATIVE-GUARD.
"""
from __future__ import annotations

import argparse
import os
import sys

try:
    from .stats import holm_correction
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from seq.stats import holm_correction

import seq.prizma_lm_claim as plc
import seq.floorfreeze_claim as ffc


REGISTRY_ID = "PR-2026-09-03-14"
LEDDIR = "manyblock_PR-2026-09-03-14"
SMOKE_BASENAME = "smoke.json"
POWERED_BASENAME = "powered.json"

ARMS = ("PLAIN", "EX")
CLAIM_SEEDS = plc.CLAIM_SEEDS                   # (0, 1, 2, 3, 4)

LR_FROZEN = 3e-3
TRUNK_LR_SCALE = 0.25
BACKBONE_LR_POST_A = LR_FROZEN * TRUNK_LR_SCALE  # 7.5e-4 — L1, every post-A block
P1_MARGIN = 0.10
P2_FRAC_BAR = 0.5
P3_MARGIN = 0.05
G1_ALLOWANCE = -0.10
G2_MARGIN = 0.05
ALPHA = plc.ALPHA
HOLM_FAMILY = ("P1", "P3")   # doc addendum #1: G1/G2 are CI-based guards (an
                             # "absence of harm" bar cannot carry a small-p gate)
BLOCKS = ("A", "B", "C", "D", "E")
DOMAIN_OF = {"A": "text8", "B": "shakes", "C": "text8", "D": "shakes", "E": "text8"}
TOP_WINDOW_BATCHES = plc.B3_WINDOW_BATCHES       # 20 — the re-engagement window (P2)

PR11_LEDDIR = "prizma_lm_PR-2026-09-03-11"
POWERED_CPU_FALLBACK_NOTE = (
    "POWERED VIA THE REGISTERED CPU FALLBACK (PR-14 doc §5; precedent = PR-08 addendum "
    "2026-09-08 #2 + the PR-09..PR-13 powered-cpu executions).")


def _results_root() -> str:
    root = os.environ.get("PRIZMA_RESULTS", os.path.join(os.path.dirname(__file__), "..", "results"))
    return os.path.abspath(root)


def _default_results_path(smoke: bool) -> str:
    return os.path.join(_results_root(), LEDDIR, SMOKE_BASENAME if smoke else POWERED_BASENAME)


def resolve_results_path(explicit=None, *, smoke: bool = False, force_smoke_path: bool = False) -> str:
    path = explicit if explicit else _default_results_path(smoke)
    if smoke and not force_smoke_path:
        powered = _default_results_path(smoke=False)
        if os.path.normcase(os.path.abspath(path)) == os.path.normcase(os.path.abspath(powered)):
            raise SystemExit(
                f"refusing: a --smoke run was pointed at the powered ledger ({powered}). Pass "
                f"--out <path> or --force-smoke-path (PR-2026-09-03-14 §5 ledger separation).")
    return path


def require_cuda(has_cuda: bool) -> None:
    if not has_cuda:
        raise SystemExit(
            "refusing: --powered executes the PR-2026-09-03-14 claim campaign and requires a "
            "CUDA device. Use --smoke, or --powered-cpu for the registered CPU fallback.")


needs_cuda = plc.needs_cuda
canary_mismatches = ffc.canary_mismatches   # reused comparator (same object, pinned)


# ==================================================================== PURE: owner election ======
def elect_owner(domain, dom_counts, committed):
    """PURE per-block domain-owner election (doc §2): the COMMITTED slot with the maximum
    cumulative trained-segment count in `domain`; None when no committed slot has history
    (first encounter — first encounters route/recruit freely). Ties keep the LOWEST slot
    (deterministic). dom_counts: list of per-slot {'text8': int, 'shakes': int}."""
    best, best_n = None, 0
    for s in range(len(dom_counts)):
        if not committed[s]:
            continue
        n = dom_counts[s].get(domain, 0)
        if n > best_n:
            best, best_n = s, n
    return best


# ==================================================================== PURE: verdict (§3+§4) ======
def claim_verdict(plain, ex, *, alpha=ALPHA):
    """Frozen §3/§4 verdict. plain/ex = per-seed record lists (THIS ledger)."""
    # P1 flipped onto the degradation scale: recovery >= 0.10 <=> (postD - postC) <= -0.10
    p1 = plc._onesample_margin([r["bpc_B_postD"] - r["bpc_B_postC"] for r in ex], -P1_MARGIN)
    p1["recovery_per_seed"] = [r["bpc_B_postC"] - r["bpc_B_postD"] for r in ex]
    p1["recovery_mean"] = -p1["mean"]
    de_fracs = [float(r["boundary_window_D"]["frac_to_owner"]) for r in ex]
    p2 = {"frac_per_seed": de_fracs,
          "frac_mean": sum(de_fracs) / len(de_fracs), "frac_bar": P2_FRAC_BAR,
          "status": "PASS" if sum(de_fracs) / len(de_fracs) >= P2_FRAC_BAR else "FAIL",
          "note": "D's first-20-batch trained fraction to the shakespeare owner (B-expert)"}
    p3 = plc._onesample_margin([r["bpc_A_postE"] - r["bpc_A_preB"] for r in ex], P3_MARGIN)
    # G1/G2: CI-BASED GUARDS (doc addendum #1) — "absence of harm" bars cannot carry a
    # small-p gate. delta for G1 = mean(PLAIN post_D) - mean(EX post_D): PASS iff
    # CI upper <= 0.10, FAIL iff CI lower > 0.10, else INCONCLUSIVE.
    # harm := bpc(EX post_D) - bpc(PLAIN post_D) (positive = EX worse); CI-positional guard:
    # PASS iff harm CI upper <= 0.10, FAIL iff harm CI lower > 0.10 (doc addendum #1).
    g1 = plc._welch_margin([r["bpc_B_postD"] for r in plain],
                           [r["bpc_B_postD"] for r in ex], 0.10)
    g1["harm_mean"] = g1["delta"]
    g1["note"] = ("harm = mean(EX post_D) - mean(PLAIN post_D); CI-based guard: PASS iff "
                  "harm CI upper <= 0.10, FAIL iff harm CI lower > 0.10 (doc addendum #1)")
    g2 = plc._onesample_margin([r["bpc_Cret_postE"] - r["bpc_Cret_postC"] for r in ex], G2_MARGIN)

    holm = holm_correction([p1["p_raw"], p3["p_raw"]], alpha=alpha)
    bars = {"P1": p1, "P3": p3, "G1": g1, "G2": g2}
    for name, h in zip(("P1", "P3"), holm):
        bars[name]["p_holm"] = h["p_adj"]

    def _below(res, margin, p_holm):
        if res["ci"][1] <= margin and p_holm < alpha:
            return "PASS"
        if res["ci"][0] > margin:
            return "FAIL"
        return "INCONCLUSIVE"

    p1["status"] = _below(p1, -P1_MARGIN, p1["p_holm"])
    p3["status"] = _below(p3, P3_MARGIN, p3["p_holm"])
    g2["status"] = plc._bar_status(g2["ci"], G2_MARGIN, 0.0, "below", alpha)
    g1["status"] = plc._bar_status(g1["ci"], 0.10, 0.0, "below", alpha)

    branches, guard_statuses = [], [g1["status"], g2["status"]]
    if "INCONCLUSIVE" in guard_statuses:
        outcome = "INCONCLUSIVE"
        branches.append("§4 (guard CI straddles): n=10 (seeds 5-9) PRE-AUTHORIZED; never "
                        "PASS (PR-03 mirror).")
    else:
        p1_ok = p1["status"] == "PASS"
        claims_ok = p1_ok and p2["status"] == "PASS" and p3["status"] == "PASS"
        if claims_ok and all(st == "PASS" for st in guard_statuses):
            outcome = "CLAIMED"
        elif not p1_ok:
            outcome = "NEGATIVE"
            branches.append("§4 (P1 FAIL): no revisit recovery at claim grade; the "
                            "many-block design is abandoned for this mechanism class.")
        elif p2["status"] == "FAIL":
            outcome = "NEGATIVE"
            branches.append("§4 (P2 FAIL): the re-encounter is real but the ROUTING LEDGER "
                            "does not re-engage the B-expert; 'lifelong routing' stays "
                            "wording-limited.")
        elif p3["status"] == "FAIL":
            outcome = "NEGATIVE-GUARD"
            branches.append("§4 (P3 FAIL): accumulation does not hold at 5 blocks.")
        else:
            outcome = "NEGATIVE-GUARD"
            broke = [n for n, st in zip(("G1", "G2"), guard_statuses) if st == "FAIL"]
            branches.append(f"§4 (guard FAIL): {' and '.join(broke)} — record and stop.")
    if outcome == "CLAIMED":
        verdict_text = ("CLAIMED — the column accumulates (P3), recovers the re-encountered "
                        "domain (P1), keeps the re-encounter ledger honest (P2), and the "
                        "exclusion neither costs the revisit (G1) nor text8 continuity (G2): "
                        "the lifelong claim rests on 5 blocks (doc §4).")
    elif outcome == "NEGATIVE":
        verdict_text = "NEGATIVE — see branches."
    elif outcome == "NEGATIVE-GUARD":
        verdict_text = "NEGATIVE-GUARD — a guard failed; see branches."
    else:
        verdict_text = ("INCONCLUSIVE — a guard CI straddles its margin (never PASS); the "
                        "pre-authorized n=10 extension applies (see 'branches').")
    plain_b = [r["bpc_B_postD"] for r in plain]
    return {"outcome": outcome, "verdict": verdict_text, "branches": branches,
            "P1": p1, "P2": p2, "P3": p3, "G1": g1, "G2": g2,
            "plain_b_postD_per_seed": plain_b,
            "holm_family": [{"bar": n, "p_raw": bars[n]["p_raw"], "p_holm": bars[n]["p_holm"],
                             "status": bars[n]["status"]} for n in HOLM_FAMILY],
            "ci_guards": {"G1": {"harm_mean": g1["harm_mean"], "ci": g1["ci"],
                                 "status": g1["status"]},
                          "G2": {"ci": g2["ci"], "status": g2["status"]}},
            "alpha": alpha}


# ================================================================================= runner ========
def run_cell(vocab_size, seed, blocks_data, evals, lr, *, exclusion):
    """PR-14 cell — the 5-block stream. EX pins model.domain_protect at every post-A block
    start (owner elected from the cumulative domain ledger); PLAIN never sets it. The
    A/B phases are bit-identical ACROSS arms (both arms route B freely — a first encounter
    elects no owner; divergence begins at C)."""
    import time

    import torch
    from seq import fusion_probe as fp

    t0 = time.time()
    E = plc.E_POOL
    model = fp.build_model(vocab_size, seed, E)
    ledger = plc._fresh_ledger()
    dom_counts = [{"text8": 0, "shakes": 0} for _ in range(E)]
    rec = {"config": ("PRIM-EX" if exclusion else "PRIM-PLAIN"),
           "E": E, "seed": seed, "lr": lr}

    Aex, Aey = evals["A"]
    Bex, Bey = evals["B"]
    Crx, Cry = evals["Cret"]

    def eval_all():
        return {"bpc_A": fp.eval_bpc_fusion(model, Aex, Aey),
                "bpc_B": fp.eval_bpc_fusion(model, Bex, Bey),
                "bpc_Cret": fp.eval_bpc_fusion(model, Crx, Cry)}

    for tag in BLOCKS:
        domain = DOMAIN_OF[tag]
        is_post_a = tag != "A"
        bx, by = blocks_data[tag]
        owner = None
        if is_post_a and exclusion:
            owner = elect_owner(domain, dom_counts, model.committed)
            if owner is not None:
                protect = {s for s in range(E) if model.committed[s]} - {int(owner)}
                if protect:
                    model.domain_protect = protect
        boundary = {"a_expert": int(owner) if owner is not None else None,
                    "owner_slot": int(owner) if owner is not None else None,
                    "segments_total": 0, "to_A_expert": 0}
        before = model.n_segments[:]
        kw = {"backbone_lr": BACKBONE_LR_POST_A} if is_post_a else {}
        plc.train_pr08(model, bx, by, lr, tag, ledger, seed=seed, boundary=boundary, **kw)
        block_tr = [model.n_segments[s] - before[s] for s in range(E)]
        for s in range(E):
            dom_counts[s][domain] += block_tr[s]
        ev = eval_all()
        rec[f"traj_{tag}"] = ev
        rec[f"block_{tag}"] = {
            "domain": domain, "owner": (int(owner) if owner is not None else None),
            "train": block_tr,
            "frac_to_owner_first20": (boundary["to_A_expert"] / boundary["segments_total"]
                                      if boundary["segments_total"] else None),
            "recruits_so_far": len(ledger["recruits"]),
            "evictions_so_far": len(ledger["evictions"]),
            "suppressed_so_far": ledger.get("domain_protect_suppressed_recruits", 0),
            "redirects_so_far": ledger.get("domain_protect_redirects", 0),
        }
        if hasattr(model, "domain_protect"):
            rec[f"block_{tag}"]["protected"] = sorted(model.domain_protect)
            del model.domain_protect

    # claim-facing flat keys (the verdict reads these)
    rec["bpc_A_preB"] = rec["traj_A"]["bpc_A"]           # post-A (= "bpc_A(post_A)")
    rec["bpc_A_postB"] = rec["traj_B"]["bpc_A"]
    rec["bpc_B_postB"] = rec["traj_B"]["bpc_B"]
    rec["bpc_B_postC"] = rec["traj_C"]["bpc_B"]
    rec["bpc_B_postD"] = rec["traj_D"]["bpc_B"]
    rec["bpc_B_postE"] = rec["traj_E"]["bpc_B"]
    rec["bpc_A_postE"] = rec["traj_E"]["bpc_A"]
    rec["bpc_Cret_postC"] = rec["traj_C"]["bpc_Cret"]
    rec["bpc_Cret_postE"] = rec["traj_E"]["bpc_Cret"]
    rec["boundary_window_D"] = {"frac_to_owner": rec["block_D"]["frac_to_owner_first20"]}
    rec["b_degradation_B_postC"] = rec["bpc_B_postC"] - rec["bpc_B_postB"]   # B4-style
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def _fp(payload: dict) -> str:
    from seq.gpu_harness import config_fingerprint
    return config_fingerprint({"registry": REGISTRY_ID, **payload})


def run_canary(plain_rec, ex_rec, seed):
    """Doc §2 per-cell cross-arm canary: the A and B phases must be BIT-IDENTICAL between
    PLAIN and EX (divergence begins at C, the first protected block)."""
    for k in ("bpc_A_preB", "bpc_A_postB", "bpc_B_postB"):
        if plain_rec[k] != ex_rec[k]:
            raise SystemExit(
                f"CANARY ABORT (PR-2026-09-03-14 §2): seed {seed} {k} "
                f"PLAIN={plain_rec[k]!r} != EX={ex_rec[k]!r} — the arms must share the A/B "
                f"phases bit-for-bit (divergence begins at C).")
    return {"ok": True, "seed": seed, "checked": ["bpc_A_preB", "bpc_A_postB", "bpc_B_postB"]}


def run(*, smoke: bool, results_path=None, force_smoke_path: bool = False,
        powered_cpu: bool = False, seed_offset=0, ledger_dir=None):
    path = os.path.join(_results_root(), ledger_dir,
                        SMOKE_BASENAME if smoke else POWERED_BASENAME)         if ledger_dir else resolve_results_path(results_path, smoke=smoke,
                                                force_smoke_path=force_smoke_path)

    import time

    import torch

    from seq import blockdrift_claim as claim
    from seq import fusion_probe as fp
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
        seeds = tuple(s + seed_offset for s in CLAIM_SEEDS)
        # PR-2026-09-03-19 replication mode: seed_offset shifts the FRESH seed set (5-9 for
        # offset 5). Fingerprints carry the offset (fresh-seed cells never resume from
        # original-seed cells).

    # ---------------- the 5-block stream (doc §2, exact slices) -------------------------------
    A_all, B_all = claim.fetch_corpora()
    shakes_train_end = int(len(B_all) * 0.9)
    text_blocks = {"A": A_all[0:1_000_000], "C": A_all[1_100_000:2_100_000],
                   "E": A_all[2_100_000:3_100_000]}
    shake_blocks = {"B": B_all[:shakes_train_end], "D": B_all[:shakes_train_end]}
    # (build the vocab over ALL blocks + evals, then segment)
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
    evals = {"A": segs(A_all[1_000_000:1_100_000]),
             "B": segs(B_all[shakes_train_end:]),
             "Cret": segs(A_all[900_000:1_000_000])}

    # ---------------- ledger + meta (NO lr-selection leg: LR frozen by inheritance) -----------
    res = load_results(path)
    res["meta"] = {
        "registry": REGISTRY_ID, "smoke": bool(smoke),
        "lane": "CLAIM (frozen pre-registration)" if not smoke else "SMOKE (plumbing only)",
        "doc": "docs/preregistry/2026-09-09-manyblock-accumulation.md",
        "arms": list(ARMS), "claim_seeds": list(seeds),
        "blocks": list(BLOCKS), "domain_of": dict(DOMAIN_OF),
        "seg": plc.SEG, "batch_segs": plc.BATCH_SEGS,
        "lr_rule": ("FROZEN by inheritance: 3e-3 on block A; backbone 7.5e-4 (x0.25, PR-10 "
                    "dose) on EVERY post-A block; tissue optimizers 3e-3 everywhere."),
        "lever": ("L2 generalized: per-block domain-OWNER election (max cumulative domain-"
                  "trained count among committed slots; no history => no owner, free "
                  "routing); protected slots = committed MINUS owner; guarded branches in "
                  "route_pr08 (owner_slot precedence + redirect + suppression refinement)"),
        "stream_lengths": {t: len(text_blocks.get(t, shake_blocks.get(t, "")))
                           for t in BLOCKS},
        "eval_slices": {"A_eval": "text8[1.0M,1.1M)", "B_eval": "shakes[90%,100%]",
                        "Cret": "text8[0.9M,1.0M)"},
        "vocab": V, "threads": torch.get_num_threads(),
        "bars": {"P1_margin": P1_MARGIN, "P2_frac_bar": P2_FRAC_BAR, "P3_margin": P3_MARGIN,
                 "G1_allowance": G1_ALLOWANCE, "G2_margin": G2_MARGIN, "alpha": ALPHA,
                 "holm_family": list(HOLM_FAMILY),
                 "t_isf_convention": "UPPER-TAIL p in (0, 0.5] (the PR-03 lesson)"},
        "canary": ("doc §2, per cell (claim mode): the A/B phases must be bit-identical "
                   "across arms (divergence begins at C)."),
    }
    if powered_cpu:
        res["meta"]["powered_cpu"] = True
        res["meta"]["compute_fallback_note"] = POWERED_CPU_FALLBACK_NOTE
    if seed_offset:
        res["meta"]["seed_offset"] = seed_offset
        res["meta"]["replication_note"] = (
            f"PR-2026-09-03-19: fresh seeds {seeds[0]}..{seeds[-1]} (offset {seed_offset}); "
            "BOTH arms re-run for the cross-arm canary and the fresh G1 comparison")
    _save(res, path)
    print(f"[pr14] stream: A/B/C/D/E segs="
          f"{[blocks_data[t][0].shape[0] for t in BLOCKS]}; vocab={V}; smoke={smoke}; "
          f"lr={LR_FROZEN} + backbone_postA={BACKBONE_LR_POST_A}; results={path}", flush=True)

    # ---------------- the 2 arms x seeds (PLAIN first; per-cell cross-arm canary) --------------
    first_cell_s = None
    canaries = []
    for arm in ARMS:
        exclusion = (arm == "EX")
        for seed in seeds:
            cellkey = f"claim.{arm}.s{seed}"
            cfgsig = _fp({"leg": "claim", "arm": arm, "seed": seed, "lr": LR_FROZEN,
                          "backbone_lr_post_a": BACKBONE_LR_POST_A,
                          "domain_exclusion": exclusion,
                          "smoke": bool(smoke), "vocab": V,
                          "seg": plc.SEG, "batch_segs": plc.BATCH_SEGS,
                          "stream_lengths": res["meta"]["stream_lengths"],
                          "blocks": list(BLOCKS), "domain_of": dict(DOMAIN_OF),
                          "bars": res["meta"]["bars"],
                          "seed_offset": seed_offset})   # PR-19: replication discipline
            prior = res.get(cellkey)
            if isinstance(prior, dict) and prior.get("cfgsig") == cfgsig and prior.get("complete"):
                print(f"[pr14] {arm} seed {seed}: resumed from ledger "
                      f"(bpc_B_postD={prior['bpc_B_postD']:.3f})", flush=True)
                continue
            if isinstance(prior, dict):
                raise SystemExit(f"cell {cellkey} exists at a foreign config fingerprint "
                                 f"({prior.get('cfgsig')} != {cfgsig}) — refusing to resume")
            t0 = time.time()
            rec = run_cell(V, seed, blocks_data, evals, LR_FROZEN, exclusion=exclusion)
            if arm == "PLAIN":
                res[cellkey] = {**rec, "arm": arm, "seed": seed, "lr": LR_FROZEN,
                                "backbone_lr_post_a": BACKBONE_LR_POST_A,
                                "domain_exclusion": False, "cellkey": cellkey,
                                "cfgsig": cfgsig, "complete": True,
                                "wall_s": round(time.time() - t0, 1)}
                _save(res, path)
                print(f"[pr14] {arm} seed {seed}: B: postB={rec['bpc_B_postB']:.3f} "
                      f"postC={rec['bpc_B_postC']:.3f} postD={rec['bpc_B_postD']:.3f} "
                      f"postE={rec['bpc_B_postE']:.3f} | A_postE={rec['bpc_A_postE']:.3f} "
                      f"wall={rec['wall_s']}s", flush=True)
                continue
            # EX: run the canary against the JUST-COMPLETED PLAIN cell BEFORE storing
            canaries.append(run_canary(res[f"claim.PLAIN.s{seed}"], rec, seed))
            rec.update({"arm": arm, "seed": seed, "lr": LR_FROZEN,
                        "backbone_lr_post_a": BACKBONE_LR_POST_A,
                        "domain_exclusion": True, "cellkey": cellkey,
                        "cfgsig": cfgsig, "complete": True,
                        "wall_s": round(time.time() - t0, 1)})
            res[cellkey] = rec
            _save(res, path)                    # crash-safe after every cell
            if first_cell_s is None:
                first_cell_s = rec["wall_s"] + res["claim.PLAIN.s0"]["wall_s"]
                proj = plc.budget_projection(first_cell_s, n_cells_total=len(ARMS) * len(seeds))
                res.setdefault("meta", {})["budget_projection"] = proj
                _save(res, path)
            bd = rec["boundary_window_D"]["frac_to_owner"]
            print(f"[pr14] {arm} seed {seed}: A_postE={rec['bpc_A_postE']:.3f} "
                  f"B: postB={rec['bpc_B_postB']:.3f} postC={rec['bpc_B_postC']:.3f} "
                  f"postD={rec['bpc_B_postD']:.3f} postE={rec['bpc_B_postE']:.3f} "
                  f"D_frac_to_owner={bd} "
                  f"wall={rec['wall_s']}s", flush=True)

    # ---------------- RETENTION: archive BEFORE any verdict -----------------------------------
    raw_archive = archive_run(res, label=f"manyblock-{REGISTRY_ID}")
    res.setdefault("meta", {})["raw_archive"] = raw_archive

    # ---------------- the frozen §3/§4 verdict (claim mode) -----------------------------------
    report = {"registry": REGISTRY_ID, "smoke": bool(smoke), "arms": list(ARMS),
              "claim_seeds": list(seeds), "lr": LR_FROZEN,
              "backbone_lr_post_a": BACKBONE_LR_POST_A,
              "raw_archive": raw_archive, "canaries": canaries,
              "cells": {k: res[k] for k in sorted(res) if k.startswith("claim.")}}
    if powered_cpu:
        report["powered_cpu"] = True
    if seed_offset:
        report["seed_offset"] = seed_offset         # PR-2026-09-03-19 (recorded when set)
    if not smoke:
        plain = [res[f"claim.PLAIN.s{s}"] for s in seeds]
        ex = [res[f"claim.EX.s{s}"] for s in seeds]
        verdict = claim_verdict(plain, ex, alpha=ALPHA)
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
    for key, c in report["cells"].items():
        print(f"    {key:<30} B: postB={c['bpc_B_postB']:.3f} postC={c['bpc_B_postC']:.3f} "
              f"postD={c['bpc_B_postD']:.3f} postE={c['bpc_B_postE']:.3f} | "
              f"A_postE={c['bpc_A_postE']:.3f} Cret_postE={c['bpc_Cret_postE']:.3f} "
              f"wall={c['wall_s']}s", flush=True)
        d = c.get("block_D", {})
        print(f"      D: owner={d.get('owner')} frac_to_owner_first20="
              f"{d.get('frac_to_owner_first20')} suppressed={d.get('suppressed_so_far')} "
              f"redirects={d.get('redirects_so_far')}", flush=True)
    if not smoke:
        v = report["verdict"]
        p1 = v["P1"]
        print(f"    P1: recovery mean={p1['recovery_mean']:.4f} "
              f"(flipped-scale mean={p1['mean']:.4f}, CI=[{p1['ci'][0]:.4f},{p1['ci'][1]:.4f}], "
              f"bar flipped <= {p1['margin']}) p_holm={p1['p_holm']:.4f} -> {p1['status']}",
              flush=True)
        p2 = v["P2"]
        print(f"    P2: frac_mean={p2['frac_mean']:.3f} (bar >= {p2['frac_bar']}) "
              f"-> {p2['status']}", flush=True)
        p3 = v["P3"]
        print(f"    P3: A-retention mean={p3['mean']:.4f} CI=[{p3['ci'][0]:.4f},{p3['ci'][1]:.4f}] "
              f"(bar <= {p3['margin']}) p_holm={p3['p_holm']:.4f} -> {p3['status']}", flush=True)
        g1 = v["G1"]
        print(f"    G1: harm mean={g1['harm_mean']:.4f} CI=[{g1['ci'][0]:.4f},"
              f"{g1['ci'][1]:.4f}] (allowance 0.10; CI-positional, no p-gate — doc "
              f"addendum #1) -> {g1['status']}", flush=True)
        g2 = v["G2"]
        print(f"    G2: Cret drift mean={g2['mean']:.4f} CI=[{g2['ci'][0]:.4f},{g2['ci'][1]:.4f}] "
              f"(bar <= {g2['margin']}; CI-positional, no p-gate — doc addendum #1) "
              f"-> {g2['status']}", flush=True)
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
        prog="manyblock_claim",
        description="PR-2026-09-03-14 claim runner (many-block accumulation with a true "
                    "revisit). --smoke = tiny CPU plumbing run; --powered = the frozen claim "
                    "campaign (requires CUDA); --powered-cpu = the registered CPU fallback. "
                    "An unknown flag is rejected without launching anything.")
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
    p.add_argument("--seed-offset", type=int, default=0,
                   help="PR-2026-09-03-19: shift the claim seeds by this offset (5 => "
                        "fresh seeds 5-9 for a replication run; BOTH arms re-run). Default "
                        "0 = byte-identical behavior.")
    p.add_argument("--ledger-dir", default=None,
                   help="ledger subdirectory under $PRIZMA_RESULTS (default None = the "
                        "PR-14 dir; the PR-19 run uses manyblock_PR-2026-09-03-19 so the "
                        "PR-14 ledger is never written)")
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(argv)
    run(smoke=args.smoke, results_path=args.out, force_smoke_path=args.force_smoke_path,
        powered_cpu=args.powered_cpu, seed_offset=args.seed_offset,
        ledger_dir=args.ledger_dir)


if __name__ == "__main__":
    main()
