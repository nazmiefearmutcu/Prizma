"""
Expert economy analysis -- Wave-C C3 (commission report 10, P3 + synthesis C4 caveat).

*** EXPLORATORY (LANE-EXPLORATORY): analysis + design support only. Nothing here is a
    claim; the only future-claim vehicle is the pre-registration DRAFT at the bottom of
    docs/EXPERT_ECONOMY.md (PR-2026-09-03-05, status IN-WRITE). ***

What this script does (READ-ONLY w.r.t. src/ -- all instrumentation lives here, via a
thin subclass of Prizma; src/prizma.py and experiments/run_continual.py are untouched):

  1. GROWTH QUANTIFICATION. Runs the repo's OWN shipped E-suite stream
     (src.data.structured_permuted_tasks) at the shipped E1 settings
     (d=24, h=48, K=8 classes, n_experts=K+3=8, epochs=15, DFA feedback,
     consolidate=True, z_novel=5.0 -- identical to run_continual.E1_main) and records a
     per-domain expert ledger: recruits, freezes, route_log, parameter counts, the
     inference routing matrix, and the surprise distribution at each recruit event.
     Also runs an EXPLORATORY interleaved variant (same generator, same permutations,
     same per-sample exposure; presentation order shuffled across domains instead of
     contiguous blocks). The interleaved stream is NEW and exploratory -- it is NOT a
     shipped benchmark; it exists because synthesis C4 flags interleaved collapse as an
     open caveat.

  2. OFFLINE ECONOMY SIMULATIONS, all pure functions of the recorded ledger:
       - merge: vigilance-compatible merge (Jaccard of routed test-sample sets > tau,
         Welch-indistinguishable surprise, post-merge surprise under each vigilance).
         Welch p-values come from seq/stats.py when importable (analysis -> seq import
         only; src never imports seq); otherwise effect sizes are reported.
       - prune: zero-routing experts over the stream window (use-it-or-lose-it).
       - evict: recruit-by-eviction policy under a hard M_max (replay of the recorded
         recruit sequence; lowest lifetime routing share is evicted first).

  3. OPEN-WORLD ABSTAIN PROBE (spec support for the NOVEL output, not an
     implementation in src): a held-out 6th domain (same generator, never trained on)
     is routed at test time; we report the min-surprise z-score distribution and what
     fraction of known vs novel inputs a vigilance rule "NOVEL iff min_z > z_abstain"
     would flag. Current code force-routes every input (argmin); this quantifies that gap.

Output: results/expert_economy_2026-09-03/growth.json

Usage: python experiments/expert_economy.py [--seeds 0,1,2] [--quick]
"""
from __future__ import annotations

import os
import sys
import json
import math
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import numpy as np

from data import structured_permuted_tasks
from prizma import Prizma

# Analysis-side import of seq/stats.py for Welch p-values. Direction check: src/ never
# imports seq/ (grep-verified 2026-09-03); importing seq from an experiments analysis
# script is one-directional and touches neither package. Fallback: effect sizes only.
_REPO = os.path.join(_HERE, "..")
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
try:
    from seq.stats import t_sf as _t_sf
    HAVE_SEQ_STATS = True
except Exception:
    HAVE_SEQ_STATS = False

DATE = "2026-09-03"
LABEL = "EXPLORATORY"


# ============================================================================ #
# Welch test (local statistic; p-values via seq.stats.t_sf when available)      #
# ============================================================================ #
def welch_test(a, b):
    """Welch t, Welch-Satterthwaite df, Cohen's d, two-sided p (if t_sf available).

    Returns dict; p is None when seq.stats is not importable (effect sizes still valid).
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {"t": None, "df": None, "cohen_d": None, "p": None}
    va, vb = a.var(ddof=1), b.var(ddof=1)
    va_na, vb_nb = va / na, vb / nb
    se = math.sqrt(va_na + vb_nb)
    if se == 0.0:
        return {"t": 0.0, "df": float(na + nb - 2), "cohen_d": 0.0, "p": 1.0}
    t = (a.mean() - b.mean()) / se
    df = (va_na + vb_nb) ** 2 / ((va_na ** 2) / (na - 1) + (vb_nb ** 2) / (nb - 1))
    pooled_sd = math.sqrt((va + vb) / 2.0)
    d = (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0.0 else 0.0
    p = 2.0 * _t_sf(abs(t), df) if HAVE_SEQ_STATS else None
    return {"t": float(t), "df": float(df), "cohen_d": float(d), "p": p}


# ============================================================================ #
# Parameter accounting straight from live Expert array shapes                  #
# ============================================================================ #
def expert_float_counts(expert):
    """(trainable, fixed_fa, total) floats per expert, counted from the actual arrays.

    Trainable: Wenc(h*d) + benc(h) + Wdec(d*h) + bdec(d) + Wcls(K*h) + bcls(K).
    Fixed FA feedback (never trained): Bdec(h*d) + Bcls(h*K).
    """
    trainable = (expert.Wenc.size + expert.benc.size +
                 expert.Wdec.size + expert.bdec.size +
                 expert.Wcls.size + expert.bcls.size)
    fixed = expert.Bdec.size + expert.Bcls.size
    return int(trainable), int(fixed), int(trainable + fixed)


def mlp_param_count(sizes):
    """Backprop MLP baseline, same closed form as run_continual.param_count_mlp."""
    return int(sum(sizes[i] * sizes[i + 1] + sizes[i + 1]
                   for i in range(len(sizes) - 1)))


# ============================================================================ #
# Instrumented Prizma: thin subclass, zero src/ changes                        #
# ============================================================================ #
class InstrumentedPrizma(Prizma):
    """Prizma + a recruit/freeze/drop ledger. Inherits ALL learning behaviour.

    train_batch is wrapped: state is snapshotted before/after the parent call and the
    diff IS the ledger (recruit = an expert's n_seen goes 0 -> >0; freeze = a frozen
    flag flips; drop = the parent returned with neither route_log nor weights touched,
    which is what the current `if self.active >= self.M: return` hard-cap does).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._task_label = "?"
        self.ledger = {
            "recruit_events": [],
            "freeze_events": [],
            "n_dropped_batches": 0,
            "n_dropped_samples": 0,
            "vigilance_tests": 0,          # _recognizes calls (batch level)
            "vigilance_recognized": 0,
            "per_expert_vigilance": [[0, 0] for _ in range(self.M)],  # [tests, passed]
        }

    # -- vigilance test counting (committed-expert scan AND active expert's own
    #    phase check; counts only -- recruit events carry the detail) -- #
    def _recognizes(self, e, X):
        m = self.experts.index(e)
        ok = super()._recognizes(e, X)
        self.ledger["vigilance_tests"] += 1
        self.ledger["per_expert_vigilance"][m][0] += 1
        if ok:
            self.ledger["vigilance_recognized"] += 1
            self.ledger["per_expert_vigilance"][m][1] += 1
        return ok

    # -- state diff around the parent call -- #
    def train_batch(self, X, Y, y):
        if not self.route:      # ablation mode has no economy; delegate untouched
            return super().train_batch(X, Y, y)
        n_seen0 = np.array([e.n_seen for e in self.experts])
        frozen0 = np.array([e.frozen for e in self.experts])
        rlog0 = self.route_log.copy()

        super().train_batch(X, Y, y)

        n_seen1 = np.array([e.n_seen for e in self.experts])
        frozen1 = np.array([e.frozen for e in self.experts])
        rlog1 = self.route_log.copy()

        recruited = np.where((n_seen0 == 0) & (n_seen1 > 0))[0]
        froze = np.where(~frozen0 & frozen1)[0]
        touched = bool((rlog1 > rlog0).any() or (n_seen1 > n_seen0).any())

        for m in froze:
            self.ledger["freeze_events"].append(
                {"task": self._task_label, "expert": int(m),
                 "n_seen": int(n_seen1[m])})
        for m in recruited:
            # Surprise of the triggering batch on every expert that has a precision
            # floor. Committed/just-frozen experts are unchanged by this batch, so the
            # post-call measurement is faithful to the decision that recruited m.
            rec = []
            for j, e in enumerate(self.experts):
                if e.mu < 1e8:
                    r = float(e.recon_error(X).mean())
                    sd = math.sqrt(max(e.var, 1e-12))
                    rec.append({"expert": int(j), "recon_mean": r,
                                "mu": float(e.mu), "sigma": sd,
                                "z": (r - float(e.mu)) / sd})
            self.ledger["recruit_events"].append(
                {"task": self._task_label, "new_expert": int(m),
                 "batch_size": int(len(X)),
                 "n_recruited_after": int((n_seen1 > 0).sum()),
                 "surprise_on_batch": rec})
        if not touched and len(X) > 0:
            # current hard-cap behaviour: pool exhausted -> batch silently dropped
            self.ledger["n_dropped_batches"] += 1
            self.ledger["n_dropped_samples"] += int(len(X))


# ============================================================================ #
# Snapshot helpers (plain dicts; the economy sims run from these)              #
# ============================================================================ #
def snapshot_expert(e):
    return {"Wenc": e.Wenc.astype(np.float64).tolist(),
            "benc": e.benc.astype(np.float64).tolist(),
            "Wdec": e.Wdec.astype(np.float64).tolist(),
            "bdec": e.bdec.astype(np.float64).tolist(),
            "Wcls": e.Wcls.astype(np.float64).tolist(),
            "bcls": e.bcls.astype(np.float64).tolist(),
            "mu": float(e.mu), "var": float(e.var),
            "committed": bool(e.committed), "frozen": bool(e.frozen),
            "n_seen": int(e.n_seen)}


def snap_recon(snap, X):
    """Reconstruction surprise of a snapshot expert. Mirrors Expert.recon_error on the
    shipped E-suite substrate (n_settle_steps=0, no noise/quantization): encode is a
    single tanh layer. Documented assumption; exact for every run in this file."""
    Wenc = np.asarray(snap["Wenc"], np.float64)
    benc = np.asarray(snap["benc"], np.float64)
    Wdec = np.asarray(snap["Wdec"], np.float64)
    bdec = np.asarray(snap["bdec"], np.float64)
    X = np.asarray(X, np.float64)
    Z = np.tanh(X @ Wenc.T + benc)
    Xhat = Z @ Wdec.T + bdec
    return ((X - Xhat) ** 2).mean(axis=1)


def snap_recon_mean(snap, X):
    return float(snap_recon(snap, X).mean())


def vigilance_theta(snap, z_novel):
    return snap["mu"] + z_novel * math.sqrt(max(snap["var"], 1e-12))


# ============================================================================ #
# Evaluation: routing matrix, routed sets, min-surprise z (abstain probe)      #
# ============================================================================ #
def eval_routing(p, domains_X):
    """Route each domain's test set; return counts, purity, routed sample sets, and the
    per-sample min-surprise z-score over trained experts (the NOVEL-rule statistic)."""
    trained = np.array([e.n_seen > 0 for e in p.experts], bool)
    mu = np.array([e.mu if e.mu < 1e8 else np.nan for e in p.experts], float)
    sd = np.array([math.sqrt(max(e.var, 1e-12)) if e.mu < 1e8 else np.nan
                   for e in p.experts], float)
    out = {"domains": [], "routed_sets": {m: [] for m in range(p.M)}}
    for dom, X in domains_X:
        idx, S = p.route_for_inference(X)
        counts = np.bincount(idx, minlength=p.M)
        purity = float(counts.max() / len(X)) if len(X) else 0.0
        # z-scores vs each trained expert's own precision floor
        Z = (S - mu[None, :]) / sd[None, :]
        Zt = Z[:, trained]
        min_z = Zt.min(axis=1) if Zt.shape[1] else np.full(len(X), np.inf)
        # batch-aggregated abstain statistic (repo lesson from v2: vigilance decisions
        # are BATCH-level; per-sample thresholds thrash). Consecutive non-overlapping
        # batches of 128; a batch is NOVEL iff its mean min_z exceeds z.
        bn = len(X) // 128
        batch_novel = {}
        for z in (2.0, 3.0, 4.0):
            if bn:
                bm = min_z[:bn * 128].reshape(bn, 128).mean(axis=1)
                batch_novel[str(z)] = float((bm > z).mean())
            else:
                batch_novel[str(z)] = None
        out["domains"].append({
            "domain": int(dom), "n": int(len(X)),
            "counts": counts.tolist(),
            "purity": purity,
            "min_z_mean": float(min_z.mean()),
            "min_z_p95": float(np.percentile(min_z, 95)),
            "novel_rate": {str(z): float((min_z > z).mean()) for z in (2.0, 3.0, 4.0)},
            "batch_novel_rate": batch_novel,
        })
        for m in range(p.M):
            if counts[m]:
                out["routed_sets"][int(m)].extend(
                    [(int(dom), int(i)) for i in np.where(idx == m)[0]])
    return out


# ============================================================================ #
# Merge simulation (offline, pure function of ledger snapshots)                #
# ============================================================================ #
def jaccard(set_a, set_b):
    ua, ub = set(set_a), set(set_b)
    union = ua | ub
    return len(ua & ub) / len(union) if union else 0.0


def _merged_snapshot(s1, s2):
    """Weight-average of two snapshot experts (trainable core; mu/var averaged).
    Fixed FA matrices are expert-specific random bases -- report 10 flags that merging
    encoders is only principled when latent bases align, so the merged snap drops FA
    (it is never used by snap_recon on the settle-0 substrate anyway)."""
    merged = {}
    for k in ("Wenc", "benc", "Wdec", "bdec", "Wcls", "bcls"):
        merged[k] = ((np.asarray(s1[k], np.float64)
                      + np.asarray(s2[k], np.float64)) / 2.0).tolist()
    merged["mu"] = (s1["mu"] + s2["mu"]) / 2.0
    merged["var"] = (s1["var"] + s2["var"]) / 2.0
    merged["committed"] = True
    merged["frozen"] = True
    merged["n_seen"] = s1["n_seen"] + s2["n_seen"]
    return merged


def simulate_merges(snaps, routed_sets, domains_X, tau=0.5, z_novel=5.0,
                    welch_alpha=0.05, max_cohen_d=0.5):
    """Vigilance-compatible merge criterion, evaluated OFFLINE from recorded ledger data.

    Stage 1 (gate):   Jaccard overlap of routed sample sets > tau.
    Stage 2 (stats):  surprise distributions on the UNION set statistically
                      indistinguishable (Welch p > alpha AND |cohen_d| < max_cohen_d).
    Stage 3 (check):  weight-averaged merged expert's mean surprise on EACH input set
                      stays under THAT set's own vigilance theta_i = mu_i + z*sigma_i.
    Only stage-3 survivors are mergeable. Returns per-pair verdicts + merged snaps.
    """
    X_by_dom = {int(dom): np.asarray(X, np.float32) for dom, X in domains_X}
    pool = [m for m, s in snaps.items() if s["frozen"] and s["n_seen"] > 0]
    pairs = []
    merged_snaps = {}
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            m1, m2 = pool[i], pool[j]
            s1, s2 = snaps[m1], snaps[m2]
            r1, r2 = routed_sets.get(m1, []), routed_sets.get(m2, [])
            jac = jaccard(r1, r2)
            rec = {"pair": [int(m1), int(m2)],
                   "n1": len(r1), "n2": len(r2), "jaccard": float(jac)}
            if jac <= tau:
                rec["verdict"] = "no_overlap_gate"
                pairs.append(rec)
                continue
            # union-set surprise distributions
            union = sorted(set(r1) | set(r2))
            Xu = np.vstack([X_by_dom[d][[k]] for (d, k) in union])
            S1 = snap_recon(s1, Xu)
            S2 = snap_recon(s2, Xu)
            wt = welch_test(S1, S2)
            rec["welch"] = wt
            if wt["p"] is None:
                rec["verdict"] = "welch_p_unavailable_effect_sizes_only"
                pairs.append(rec)
                continue
            indist = wt["p"] > welch_alpha and abs(wt["cohen_d"]) < max_cohen_d
            if not indist:
                rec["verdict"] = "surprise_distinguishable"
                pairs.append(rec)
                continue
            merged = _merged_snapshot(s1, s2)
            th1, th2 = vigilance_theta(s1, z_novel), vigilance_theta(s2, z_novel)
            X1 = np.vstack([X_by_dom[d][[k]] for (d, k) in sorted(set(r1))])
            X2 = np.vstack([X_by_dom[d][[k]] for (d, k) in sorted(set(r2))])
            sm1 = snap_recon_mean(merged, X1)
            sm2 = snap_recon_mean(merged, X2)
            rec["theta"] = [th1, th2]
            rec["merged_surprise"] = [sm1, sm2]
            ok = sm1 <= th1 and sm2 <= th2
            rec["verdict"] = "MERGEABLE" if ok else "post_merge_vigilance_violated"
            if ok:
                merged_snaps[(m1, m2)] = merged
            pairs.append(rec)
    return {"tau": tau, "z_novel": z_novel, "pairs": pairs,
            "n_mergeable": len(merged_snaps), "merged_snaps": merged_snaps,
            "frozen_pool": [int(m) for m in pool]}


# ============================================================================ #
# Prune + eviction policies (unit-testable pure functions)                     #
# ============================================================================ #
def prune_candidates(route_counts, min_share=0):
    """Use-it-or-lose-it: experts with <= min_share routed samples over the window.
    Changeux & Danchin 1976 (selective stabilisation); Chechik et al. 1998 (pruning)."""
    return [int(m) for m, c in enumerate(route_counts) if c <= min_share]


def evict_choice(route_counts, recruited_order, protected=None):
    """Recruit-by-eviction policy: index of the expert to evict when the pool is full.

    Lowest lifetime routing count wins; ties -> the MOST RECENTLY recruited expert
    (least consolidated, least time to serve other domains). `protected` are indices
    never evictable (e.g. the incoming active expert)."""
    protected = set(protected or ())
    order = {m: r for r, m in enumerate(recruited_order)}
    cands = [m for m in order if m not in protected]
    if not cands:
        return None
    return int(min(cands, key=lambda m: (route_counts[m], -order[m])))


def simulate_eviction(route_log_at_end, recruited_total, m_max, protected=None):
    """Replay the recorded recruit sequence under a hard cap M_max.

    Recruit #r fires eviction iff r > M_max; the victim is the lowest-lifetime-share
    expert among those recruited so far. Uses the FINAL route_log as the lifetime-share
    proxy (block streams: shares are dominated by the expert's own domain block, so
    this is a faithful replay of what a streaming counter would see). Evicted slots are
    re-used, so the pool never exceeds M_max."""
    fires = []
    recruited_so_far = []
    for r in range(1, recruited_total + 1):
        if r > m_max:
            victim = evict_choice(route_log_at_end, recruited_so_far, protected)
            fires.append({
                "recruit_no": r,
                "victim": victim,
                "victim_route_count": (int(route_log_at_end[victim])
                                       if victim is not None else None),
                "total_routed": int(sum(route_log_at_end)),
            })
            if victim is not None and victim not in recruited_so_far:
                recruited_so_far.append(victim)   # slot re-used by the new recruit
        else:
            # shipped pool: experts fill left-to-right, r-th recruit takes index r-1
            recruited_so_far.append(r - 1)
    return {"m_max": int(m_max), "n_recruits": int(recruited_total),
            "n_fires": len(fires), "fires": fires}


# ============================================================================ #
# Stream runners                                                               #
# ============================================================================ #
def run_block_stream(seed, K=5, d=24, ncls=8, h=48, epochs=15, probe_domains=1,
                     save_experts=False, z_novel=5.0):
    """E1-style stream at shipped settings + per-domain expert ledger.

    Stream: structured_permuted_tasks(n_tasks=K+probe, d=24, n_classes=8, seed=seed) --
    the exact constructor run_continual.make_sperm uses. The LAST `probe_domains` tasks
    are NEVER trained: they are the open-world probe for the abstain statistic.
    z_novel defaults to the shipped 5.0 (tests may tighten it for tiny streams)."""
    tasks = structured_permuted_tasks(n_tasks=K + probe_domains, d=d,
                                      n_classes=ncls, seed=seed)
    train_tasks = tasks[:K]
    p = InstrumentedPrizma(d=d, h=h, K=ncls, n_experts=K + 3, seed=seed,
                           consolidate=True, feedback="random", z_novel=z_novel)
    tr_per, fa_per, tot_per = expert_float_counts(p.experts[0])
    rng = np.random.default_rng(seed)          # E1 semantics: one rng across all tasks
    per_domain = []
    for i, t in enumerate(train_tasks):
        p._task_label = f"sperm{i}"
        p.fit_task(t.Xtr, t.ytr, epochs=epochs, rng=rng)
        rec = (p.ledger["recruit_events"][-1]["n_recruited_after"]
               if p.ledger["recruit_events"] else 0)
        frozen = int(sum(e.frozen for e in p.experts))
        # routing matrix over all domains seen so far
        ev = eval_routing(p, [(j, tt.Xte) for j, tt in enumerate(train_tasks[:i + 1])])
        per_domain.append({
            "domain": i,
            "recruited_cum": int(rec),
            "frozen_cum": frozen,
            "route_log": p.route_log.tolist(),
            "params_trainable_cum": int(rec * tr_per),
            "params_with_fa_cum": int(rec * tot_per),
            "routing": ev["domains"],
        })
    # ---------------- final evaluation ---------------- #
    recruited = int(sum(e.n_seen > 0 for e in p.experts))
    ev = eval_routing(p, [(j, tt.Xte) for j, tt in enumerate(train_tasks)])
    final = {
        "recruited": recruited,
        "committed": int(sum(e.committed for e in p.experts)),
        "frozen": int(sum(e.frozen for e in p.experts)),
        "route_log": p.route_log.tolist(),
        "routing": ev["domains"],
        "routed_sets": {k: v for k, v in ev["routed_sets"].items() if v},
        "params": {
            "trainable_per_expert": tr_per, "fixed_fa_per_expert": fa_per,
            "total_per_expert": tot_per,
            "trainable_recruited": int(recruited * tr_per),
            "with_fa_recruited": int(recruited * tot_per),
            "with_fa_allocated_pool": int(p.M * tot_per),
        },
    }
    # open-world probe (never-trained domain(s))
    probe = None
    if probe_domains:
        pev = eval_routing(p, [(K + b, tasks[K + b].Xte) for b in range(probe_domains)])
        probe = {"probe_domains": [int(K + b) for b in range(probe_domains)],
                 "never_trained": True, "domains": pev["domains"]}
    # ---------------- economy simulations (offline, from ledger) ---------------- #
    snaps = {m: snapshot_expert(e) for m, e in enumerate(p.experts)}
    merge_sim = simulate_merges(snaps, ev["routed_sets"],
                                [(j, tt.Xte) for j, tt in enumerate(train_tasks)])
    prune = prune_candidates(p.route_log)
    evk = simulate_eviction(p.route_log, recruited, m_max=K)
    evk1 = simulate_eviction(p.route_log, recruited, m_max=max(1, K - 1))
    return {
        "seed": seed,
        "stream": "structured_permuted E1-style (shipped constructor)",
        "config": {"K": K, "d": d, "n_classes": ncls, "h": h, "epochs": epochs,
                   "n_experts": K + 3, "z_novel": z_novel, "feedback": "random",
                   "consolidate": True},
        "per_domain": per_domain,
        "recruit_events": p.ledger["recruit_events"],
        "freeze_events": p.ledger["freeze_events"],
        "dropped_batches": p.ledger["n_dropped_batches"],
        "dropped_samples": p.ledger["n_dropped_samples"],
        "vigilance": {"tests": p.ledger["vigilance_tests"],
                      "recognized": p.ledger["vigilance_recognized"],
                      "per_expert": p.ledger["per_expert_vigilance"]},
        "final": final,
        "abstain_probe": probe,
        "merge_sim": {k: v for k, v in merge_sim.items() if k != "merged_snaps"},
        "prune_candidates_zero_route": prune,
        "eviction_sim": {"m_max_equals_K": evk, "m_max_minus1": evk1},
        "expert_snaps": (snaps if save_experts else None),
    }


def run_interleaved_stream(seed, K=5, d=24, ncls=8, h=48, epochs=15):
    """EXPLORATORY interleaved variant -- NEW stream, NOT shipped.

    Same generator, same permutations, same base data and labels (the teacher labels
    are shared across permutations, so the classification target stays well-defined),
    same total per-sample exposure (epochs over the concatenation). Only the ORDER
    changes: samples from all K domains are shuffled into one stream. This is the
    presentation-mode probe for the synthesis-C4 interleaved-collapse caveat; it is
    labelled exploratory everywhere it appears."""
    tasks = structured_permuted_tasks(n_tasks=K, d=d, n_classes=ncls, seed=seed)
    Xall = np.vstack([t.Xtr for t in tasks])
    yall = np.concatenate([t.ytr for t in tasks])
    perm = np.random.default_rng(1000 + seed).permutation(len(Xall))
    p = InstrumentedPrizma(d=d, h=h, K=ncls, n_experts=K + 3, seed=seed,
                           consolidate=True, feedback="random", z_novel=5.0)
    tr_per, fa_per, tot_per = expert_float_counts(p.experts[0])
    p._task_label = "interleaved(EXPLORATORY)"
    rng = np.random.default_rng(seed)
    p.fit_task(Xall[perm], yall[perm], epochs=epochs, rng=rng)
    ev = eval_routing(p, [(j, tt.Xte) for j, tt in enumerate(tasks)])
    snaps = {m: snapshot_expert(e) for m, e in enumerate(p.experts)}
    merge_sim = simulate_merges(snaps, ev["routed_sets"],
                                [(j, tt.Xte) for j, tt in enumerate(tasks)])
    recruited = int(sum(e.n_seen > 0 for e in p.experts))
    return {
        "seed": seed,
        "stream": "interleaved EXPLORATORY (new, not shipped)",
        "config": {"K": K, "d": d, "n_classes": ncls, "h": h, "epochs": epochs,
                   "n_experts": K + 3,
                   "note": "same generator/exposure, shuffled presentation order"},
        "recruited": recruited,
        "frozen": int(sum(e.frozen for e in p.experts)),
        "route_log": p.route_log.tolist(),
        "dropped_batches": p.ledger["n_dropped_batches"],
        "dropped_samples": p.ledger["n_dropped_samples"],
        "recruit_events": [{"new_expert": r["new_expert"],
                            "n_recruited_after": r["n_recruited_after"]}
                           for r in p.ledger["recruit_events"]],
        "routing": ev["domains"],
        "params": {"trainable_recruited": recruited * tr_per,
                   "with_fa_recruited": recruited * tot_per,
                   "with_fa_allocated_pool": int((K + 3) * tot_per)},
        "pool_exhausted": bool(p.ledger["n_dropped_batches"] > 0),
        "merge_sim": {k: v for k, v in merge_sim.items() if k != "merged_snaps"},
    }


def run_roundrobin_stream(seed, K=5, d=24, ncls=8, h=48, epochs=15, batch=128):
    """EXPLORATORY round-robin interleaved variant -- NEW stream, NOT shipped.

    Same generator/permutations/labels/exposure as the block stream, but the BATCH
    ORDER is shuffled across domains while every batch stays DOMAIN-PURE (128 samples
    from one domain). This is the interleaving mode vigilance CAN see: every domain
    switch is a batch-level novelty event, so recruit/commit machinery is exercised
    K-times more often than in contiguous blocks. Probes pool pressure (burn) under
    the shipped M=K+3 pool. The mixed-batch variant (run_interleaved_stream) is the
    opposite pole: every batch is a mixture, so batch-mean surprise is stationary and
    vigilance never fires. Both are labelled exploratory everywhere."""
    tasks = structured_permuted_tasks(n_tasks=K, d=d, n_classes=ncls, seed=seed)
    p = InstrumentedPrizma(d=d, h=h, K=ncls, n_experts=K + 3, seed=seed,
                           consolidate=True, feedback="random", z_novel=5.0)
    tr_per, fa_per, tot_per = expert_float_counts(p.experts[0])
    p._task_label = "roundrobin(EXPLORATORY)"
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        batches = []
        for di, t in enumerate(tasks):
            Yall = np.eye(p.K, dtype=np.float32)[t.ytr]
            idx = rng.permutation(len(t.Xtr))
            for s in range(0, len(idx), batch):
                bi = idx[s:s + batch]
                batches.append((di, t.Xtr[bi], Yall[bi], t.ytr[bi]))
        order = rng.permutation(len(batches))
        for b in order:
            di, Xb, Yb, yb = batches[b]
            p._task_label = f"roundrobin_dom{di}(EXPLORATORY)"
            p.train_batch(Xb, Yb, yb)
    ev = eval_routing(p, [(j, tt.Xte) for j, tt in enumerate(tasks)])
    recruited = int(sum(e.n_seen > 0 for e in p.experts))
    return {
        "seed": seed,
        "stream": "round-robin interleaved EXPLORATORY (new, not shipped)",
        "config": {"K": K, "d": d, "n_classes": ncls, "h": h, "epochs": epochs,
                   "n_experts": K + 3,
                   "note": "domain-pure batches, batch order shuffled across domains"},
        "recruited": recruited,
        "frozen": int(sum(e.frozen for e in p.experts)),
        "route_log": p.route_log.tolist(),
        "dropped_batches": p.ledger["n_dropped_batches"],
        "dropped_samples": p.ledger["n_dropped_samples"],
        "n_recruit_events": len(p.ledger["recruit_events"]),
        "recruits_per_expert": np.bincount(
            [r["new_expert"] for r in p.ledger["recruit_events"]],
            minlength=p.M).tolist(),
        "routing": ev["domains"],
        "params": {"trainable_recruited": recruited * tr_per,
                   "with_fa_recruited": recruited * tot_per,
                   "with_fa_allocated_pool": int((K + 3) * tot_per)},
        "pool_exhausted": bool(p.ledger["n_dropped_batches"] > 0),
    }


# ============================================================================ #
# Main: growth quantification over seeds, JSON out                              #
# ============================================================================ #
def main(seeds=(0, 1, 2), K=5, quick=False):
    t0 = time.time()
    epochs = 5 if quick else 15
    block_runs, inter_runs = [], []
    for s in seeds:
        block_runs.append(run_block_stream(s, K=K, epochs=epochs,
                                           save_experts=(s == seeds[0])))
        b = block_runs[-1]
        print(f"[block  seed {s}] recruited={b['final']['recruited']} "
              f"frozen={b['final']['frozen']} "
              f"purity={[round(r['purity'], 3) for r in b['final']['routing']]} "
              f"evict_fires(M=K)={b['eviction_sim']['m_max_equals_K']['n_fires']}")
    for s in seeds:
        inter_runs.append(run_interleaved_stream(s, K=K, epochs=epochs))
        r = inter_runs[-1]
        print(f"[interl seed {s}] recruited={r['recruited']} "
              f"dropped_batches={r['dropped_batches']} "
              f"purity={[round(x['purity'], 3) for x in r['routing']]}")
    rr_runs = []
    for s in seeds:
        rr_runs.append(run_roundrobin_stream(s, K=K, epochs=epochs))
        r = rr_runs[-1]
        print(f"[roundr seed {s}] recruited={r['recruited']} "
              f"recruit_events={r['n_recruit_events']} "
              f"dropped_batches={r['dropped_batches']} "
              f"purity={[round(x['purity'], 3) for x in r['routing']]}")

    # -------- headline aggregates (means over seeds) -------- #
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else None

    tr_per = block_runs[0]["final"]["params"]["trainable_per_expert"]
    fa_per = block_runs[0]["final"]["params"]["fixed_fa_per_expert"]
    tot_per = block_runs[0]["final"]["params"]["total_per_expert"]
    mlp = mlp_param_count([24, 128, 128, 8])       # run_continual.E1 backprop baseline
    rec = [r["final"]["recruited"] for r in block_runs]
    summary = {
        "floats_per_expert": {"trainable": tr_per, "fixed_fa": fa_per,
                              "total": tot_per, "commission_quoted": 4304,
                              "verified_match": (tr_per + fa_per) == 4304},
        "baseline_mlp_params_fixed": mlp,
        "recruited_at_stream_end_mean": mean(rec),
        "params_with_fa_recruited_mean": mean([r["final"]["params"]["with_fa_recruited"]
                                               for r in block_runs]),
        "ratio_recruited_with_fa_vs_mlp_mean":
            mean([r["final"]["params"]["with_fa_recruited"] / mlp for r in block_runs]),
        "ratio_recruited_trainable_vs_mlp_mean":
            mean([r["final"]["params"]["trainable_recruited"] / mlp
                  for r in block_runs]),
        "ratio_allocated_pool_with_fa_vs_mlp": (K + 3) * tot_per / mlp,
        "eviction_fires_m_max_eq_K_total":
            int(sum(r["eviction_sim"]["m_max_equals_K"]["n_fires"] for r in block_runs)),
        "eviction_fires_m_max_eq_K_minus1_total":
            int(sum(r["eviction_sim"]["m_max_minus1"]["n_fires"] for r in block_runs)),
        "mergeable_pairs_found_total":
            int(sum(r["merge_sim"]["n_mergeable"] for r in block_runs)),
        "block_growth_curve_mean": [
            {"domain": i,
             "recruited_cum_mean": mean([r["per_domain"][i]["recruited_cum"]
                                         for r in block_runs]),
             "params_with_fa_cum_mean": mean([r["per_domain"][i]["params_with_fa_cum"]
                                              for r in block_runs])}
            for i in range(K)],
    }
    out = {
        "label": LABEL,
        "date": DATE,
        "provenance": {
            "commission":
                "committee/brainstorm_2026-09-03/10_continual_lifelong_learning.md P3"
                " + 00_COMMISSION_SYNTHESIS.md C4 caveat",
            "stream": "src.data.structured_permuted_tasks (the shipped E-suite"
                      " constructor; identical to experiments/run_continual.make_sperm)",
            "note": "interleaved variant is a NEW exploratory stream (same generator,"
                    " shuffled presentation order) -- NOT a shipped benchmark",
            "instrumentation": "InstrumentedPrizma subclass in THIS script;"
                               " src/prizma.py and experiments/run_continual.py untouched",
            "welch_p_from": ("seq.stats.t_sf" if HAVE_SEQ_STATS
                             else "unavailable -> effect sizes only"),
        },
        "config": {"K": K, "d": 24, "n_classes": 8, "h": 48, "epochs": epochs,
                   "n_experts": K + 3, "seeds": list(seeds)},
        "summary": summary,
        "runs": {"block_E1style": block_runs,
                 "interleaved_exploratory": inter_runs,
                 "roundrobin_exploratory": rr_runs},
        "runtime_seconds": round(time.time() - t0, 1),
    }
    outdir = os.path.join(_HERE, "..", "results", f"expert_economy_{DATE}")
    os.makedirs(outdir, exist_ok=True)
    outp = os.path.join(outdir, "growth.json")
    with open(outp, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nSaved -> {os.path.normpath(outp)}  ({time.time() - t0:.1f}s total)")
    print(f"[headline] floats/expert: trainable={tr_per} + fixedFA={fa_per} = {tot_per}"
          f" (commission quoted 4304 -> match:"
          f" {summary['floats_per_expert']['verified_match']})")
    print(f"[headline] MLP baseline={mlp}; recruited-with-FA/MLP="
          f"{summary['ratio_recruited_with_fa_vs_mlp_mean']:.3f}; eviction fires on"
          f" E1 (M_max=K)={summary['eviction_fires_m_max_eq_K_total']}")
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    quick = "--quick" in args
    seed_args = [a for a in args if a.startswith("--seeds=")]
    seeds = tuple(int(x) for x in seed_args[0].split("=")[1].split(",")) \
        if seed_args else (0, 1, 2)
    main(seeds=seeds, quick=quick)
