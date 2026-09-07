"""Fusion probe — Toy Prizma-LM vigilance-routed expert head on the PR-07' block-drift stream.

LANE-EXPLORATORY (docs/preregistry/POLICY.md): design-informing ONLY, never a claim.
It motivates the GPU-tier PR-LM-1 architecture decision (fusion spec §3.1-3.2,
docs/superpowers/specs/2026-09-03-prizma-lm-fusion-design.md — the "unified column"
idea at the smallest honest scale). Zero changes to any existing module.

Question: on the SAME stream where PR-07' passed (text8[0:1M) -> tiny-shakespeare,
one pass, char-level), does bolting the CL thread's vigilance-routed expert tissue
onto the mixer backbone change retention/adaptation vs the plain mixer?

Design (frozen a priori, before any run):
  Backbone  Prizma-SeqLM, PR-07' pinned config (2 layers, d_model=64, H=4, chunk=64,
            window=16, max_len=256). PREFERRED VARIANT (disclosed): the backbone TRAINS
            on ALL segments with the shared LM loss (verbatim PR-07' train_stream
            objective). The frozen-backbone variant is reported as a design note in
            RESULTS.md, not run.
  Tissue    E PC-expert heads (E in {4, 8}), the src/prizma.py Expert idea ported to
            torch as a small per-expert PREDICTOR: Wenc_e: d_model -> h_small (tanh,
            h_small = 64, frozen a priori), Wdec_e: h_small -> vocab. No classifier
            head, no FA feedback (the LM analog of the Expert's reconstruction
            surprise is the segment-mean CE of the model served by that expert).
            Final logits = base lm_head logits + routed expert logits (the expert is
            a RESIDUAL CORRECTOR on top of detached base logits).
  Routing   Per SEGMENT, not per token (the block-drift lesson: segment-level routing
            avoids the interleaved partition-bound). surprise_s = mean CE of the
            combined prediction (base + expert) over segment s's tokens under its
            owning expert; calibrated per-expert with mu/sigma EMAs over the segments
            the expert TRAINS on (rate 0.05, first update seeds mu=r,
            var=max(1e-4, (0.1 r)^2) — mirrored verbatim from src/prizma.py
            _train_expert's precision-floor bookkeeping). Recognized iff
            z = (S - mu)/sigma <= z_novel = 5.0 (the shipped default).
  Recruit   Novelty recruits the next pool slot (left-to-right fill = recruit recency,
            mirroring src/prizma.py). m_max = E: once at cap, a novelty event falls
            back to ARGMIN-SURPRISE routing (this probe's recruit-at-cap
            operationalisation). Relationship to the registered semantics: PR-05's
            m_max Policy A is recruit-by-eviction (evict lowest lifetime routing
            share, tie -> most recent); this probe implements the milder fallback
            (no eviction mid-stream). With 2 corpus blocks and E >= 4 the cap is
            expected INERT (0 fires), mirroring PR-05's 0-evictions-at-home
            measurement; cap fallbacks are counted in the ledger either way.
  Learning  Local: each expert owns a private AdamW over its own two matrices and
            trains ONLY on its routed segments. Routing by surprise is
            non-differentiable and that is the point: NO gradient crosses the router.
            Expert loss uses DETACHED base logits and DETACHED backbone hidden — no
            shared backprop into the backbone; the backbone's only signal is the
            shared LM loss on all segments.
  Eval      bpc over combined logits; each eval segment served by the argmin-surprise
            committed expert (route_for_inference analog; no recruitment at eval).

Protocol (PR-07' verbatim, imported from seq.blockdrift_claim): corpora loading
(archived at results/blockdrift_PR-2026-09-03-07/corpora/), make_segments,
train_stream, eval_bpc, SEG=256, BATCH_SEGS=32, LR grid {1e-3, 3e-3, 1e-2}.
LRs: one per arm-family, chosen on seed 0's first-200-A-segment loss, then frozen
(PR-07' addendum rule); in fusion cells the same lr drives backbone and experts.
Cells: {plain (re-run for within-session comparability), fusion E=4, fusion E=8}
x seeds {0, 1}. torch.set_num_threads(8). Crash-safe per-run JSON (tmp + os.replace)
-> results/exploratory/fusion_probe_2026-09-07/raw.json.

Smoke wiring check (results discarded, separate file):
  FUSION_PROBE_SMOKE=1 python seq/fusion_probe.py   (64+64 segments, seed 0 only,
  lr fixed 3e-3, writes raw_smoke.json — never raw.json).

Run:  python seq/fusion_probe.py
"""
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)  # seq is imported AS A PACKAGE (its modules use relative imports)

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import seq.blockdrift_claim as claim          # PR-07' protocol pieces, imported VERBATIM
from seq.prizma_seq import PrizmaSeqConfig, PrizmaSeqLM

OUT = os.path.join(_ROOT, "results", "exploratory", "fusion_probe_2026-09-07")
SMOKE = bool(os.environ.get("FUSION_PROBE_SMOKE"))

SEG = claim.SEG                    # 256 chars per segment
BATCH_SEGS = claim.BATCH_SEGS      # 32 segments per training step
LR_GRID = claim.LR_GRID            # [1e-3, 3e-3, 1e-2]
SEEDS = [0, 1]
CONFIGS = [("plain", 0), ("fusion_e4", 4), ("fusion_e8", 8)]
Z_NOVEL = 5.0                      # shipped src/prizma.py z_novel default, frozen a priori
H_SMALL = 64                       # per-expert predictor width, frozen a priori
LR_SELECT_SEGS = 200               # PR-07' addendum: seed-0 A-segment loss, first 200 segs
FLOOR_EMA = 0.05                   # precision-floor EMA rate, mirrored from src/prizma.py


# ----------------------------- fusion tissue -------------------------------- #

class PCExpertHead(nn.Module):
    """Torch port of the src/prizma.py Expert as an LM head tissue: Wenc (d->h, tanh)
    + Wdec (h->vocab). Init mirrors Expert (normal, std=1/sqrt(fan_in); zero bias).
    No classifier head, no FA feedback — see module docstring."""

    def __init__(self, d_model, h_small, vocab):
        super().__init__()
        self.Wenc = nn.Linear(d_model, h_small)
        self.Wdec = nn.Linear(h_small, vocab)
        nn.init.normal_(self.Wenc.weight, std=d_model ** -0.5)
        nn.init.zeros_(self.Wenc.bias)
        nn.init.normal_(self.Wdec.weight, std=h_small ** -0.5)
        nn.init.zeros_(self.Wdec.bias)

    def forward(self, h):
        return self.Wdec(torch.tanh(self.Wenc(h)))


class FusionLM(nn.Module):
    """Prizma-SeqLM backbone (PR-07' pinned config) + a PRE-ALLOCATED pool of E expert
    heads. `committed[s]` (plain python, outside the autograd graph) mirrors
    src/prizma.py's committed flag: only committed slots route and train."""

    def __init__(self, vocab, seed, E):
        super().__init__()
        torch.manual_seed(seed)   # same seed stream as claim.build_model -> identical backbone init
        cfg = PrizmaSeqConfig(vocab=vocab, d_model=64, n_layers=2, n_heads=4, chunk=64,
                              window=16, max_len=SEG)
        self.lm = PrizmaSeqLM(cfg)
        self.experts = nn.ModuleList([PCExpertHead(64, H_SMALL, vocab) for _ in range(E)])
        self.E = E
        self.committed = [False] * E
        # per-slot precision floors over the expert's OWN stream segments (mu/sigma EMAs,
        # mirrored from src/prizma.py _train_expert; mu starts huge -> recognizes nothing)
        self.mu = [1e9] * E
        self.var = [1.0] * E
        self.n_batches = [0] * E
        self.n_segments = [0] * E
        # per-slot surprise statistics for the ledger
        self.ce_sum = [0.0] * E
        self._register_hidden_hook()

    def _register_hidden_hook(self):
        # capture the lm_head input (= final RMSNorm output) — the hidden the tissue reads
        self._h = None
        self.lm.head.register_forward_hook(lambda m, i, o: setattr(self, "_h", i[0]))

    def forward(self, idx):
        return self.lm(idx)      # self._h now holds the normalized hidden [B,T,d]

    def n_committed(self):
        return sum(self.committed)


# ----------------------------- routing -------------------------------------- #

@torch.no_grad()
def _segment_surprise(model, h, y):
    """Per-segment mean CE of the combined prediction (base + expert) under EVERY
    committed expert. Returns (S [n_committed, B], slot_ids [n_committed])."""
    B, T = y.shape
    flat_h = h.detach().reshape(B * T, -1)
    flat_y = y.detach().reshape(B * T)
    base = model.lm.head(h.detach()).reshape(B * T, -1)   # detached base logits
    slots = [s for s in range(model.E) if model.committed[s]]
    S = []
    for s in slots:
        comb = base + model.experts[s](flat_h)
        ce = F.cross_entropy(comb, flat_y, reduction="none").view(B, T).mean(1)
        S.append(ce)
    return torch.stack(S) if S else torch.zeros(0, B), slots


@torch.no_grad()
def route_batch(model, h, y, corpus, ledger, stream_pos):
    """Segment-level vigilance routing for one training batch.

    Returns assignments (list of (slot, [seg indices])). Recruits the next slot for a
    batch's novel pool (batch-level recruit granularity, mirroring src/prizma.py's
    batch-level recruitment; routing DECISIONS stay per-segment — the G1 lesson).
    At cap (m_max=E): argmin-surprise fallback, counted in the ledger."""
    B = y.shape[0]
    if model.n_committed() == 0:
        slot = next(s for s in range(model.E) if not model.committed[s])
        model.committed[slot] = True
        ledger["recruits"].append({"slot": slot, "at_batch": stream_pos, "corpus": corpus,
                                   "n_novel_segs": int(B), "trigger_z": None,
                                   "reason": "empty_pool"})
        return [(slot, list(range(B)))]

    S, slots = _segment_surprise(model, h, y)
    min_idx = S.argmin(dim=0)                      # [B]
    seg_ce = S[min_idx, torch.arange(B)]           # per-segment min surprise
    mu = torch.tensor([model.mu[s] for s in slots])
    sd = torch.tensor([max(model.var[s], 1e-12) ** 0.5 for s in slots])
    z = (seg_ce - mu[min_idx]) / sd[min_idx]
    novel = z > Z_NOVEL                            # per-segment vigilance test
    novel_ids = set(int(i) for i in torch.nonzero(novel).reshape(-1))
    free = [s for s in range(model.E) if not model.committed[s]]

    if novel_ids and free:
        # RECRUIT: the fresh slot owns the batch's novel pool EXCLUSIVELY (the argmin
        # expert does not also train on them — training-ledger exclusivity, the
        # PR-06 lesson: check the ledger, not just the accuracy).
        slot = free[0]                              # left-to-right fill = recruit recency
        model.committed[slot] = True
        zs = [round(float(v), 3) for v in z[novel][:8]]
        ledger["recruits"].append({"slot": slot, "at_batch": stream_pos,
                                   "corpus": corpus, "n_novel_segs": len(novel_ids),
                                   "trigger_z_max": max(zs) if zs else None,
                                   "reason": "novel"})
        assign = {slot: []}
        for i in range(B):
            if i not in novel_ids:
                assign.setdefault(slots[int(min_idx[i])], []).append(i)
        assign[slot] = sorted(novel_ids)
    else:
        # no novelty, or AT CAP: argmin-surprise routing for every segment (the
        # m_max=E fallback; counted in the ledger)
        if novel_ids:
            ledger["cap_fallback_batches"] += 1
            ledger["cap_fallback_segments"] += len(novel_ids)
            ledger["cap_events"].append({"at_batch": stream_pos, "corpus": corpus,
                                         "n_novel_segs": len(novel_ids),
                                         "note": "argmin-surprise fallback at m_max=E"})
        assign = {}
        for i in range(B):
            assign.setdefault(slots[int(min_idx[i])], []).append(i)
    return [(s, ids) for s, ids in assign.items() if ids]


def _expert_train(model, slot, h, y, ids, lr):
    """LOCAL update: expert `slot` trains only on its routed segments; loss uses
    DETACHED base logits + DETACHED hidden (no gradient reaches the backbone or any
    other expert). Then the post-update precision-floor EMA update, mirrored from
    src/prizma.py _train_expert bookkeeping (post-update surprise, batch-level)."""
    B, T, _ = h.shape
    idx = torch.tensor(sorted(ids), dtype=torch.long)
    hs = h.detach().reshape(B * T, -1)[(idx[:, None] * T + torch.arange(T)[None, :]).reshape(-1)]
    ys = y[idx].reshape(-1)
    base = model.lm.head(h.detach()).reshape(B * T, -1)[(idx[:, None] * T + torch.arange(T)[None, :]).reshape(-1)]
    opt = torch.optim.AdamW(model.experts[slot].parameters(), lr=lr)
    opt.zero_grad()
    comb = base + model.experts[slot](hs)
    loss = F.cross_entropy(comb, ys)
    loss.backward()
    opt.step()
    model.n_batches[slot] += 1
    model.n_segments[slot] += len(ids)
    # post-update floor update on the SAME segments (bookkeeping, no grad)
    with torch.no_grad():
        r = float(F.cross_entropy(base + model.experts[slot](hs), ys, reduction="mean").detach())
    model.ce_sum[slot] += r * len(ids)
    if model.mu[slot] > 1e8:                       # first calibration seeds the floor
        model.mu[slot] = r
        model.var[slot] = max(1e-4, (0.1 * r) ** 2)
    else:                                          # EMA, mirrored verbatim rates
        d = r - model.mu[slot]
        model.mu[slot] += FLOOR_EMA * d
        model.var[slot] = (1 - FLOOR_EMA) * model.var[slot] + FLOOR_EMA * d * d
    return float(loss.detach())


# ----------------------------- train / eval --------------------------------- #

def train_stream_fusion(model, xs, ys, lr, corpus, ledger):
    """One online pass over in-order segment batches (PR-07' batching verbatim).
    Backbone step FIRST on the whole batch (shared LM loss, all segments); expert
    steps SECOND on their routed segments using the same (pre-update) forward —
    a single forward per batch, cost-honest."""
    opt_bb = torch.optim.AdamW(model.lm.parameters(), lr=lr)
    model.train()
    n = xs.shape[0]
    for i in range(0, n, BATCH_SEGS):
        xb, yb = xs[i: i + BATCH_SEGS], ys[i: i + BATCH_SEGS]
        opt_bb.zero_grad()
        logits = model.lm(xb)                       # hook captures hidden
        h = model._h
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), yb.reshape(-1))
        loss.backward()
        opt_bb.step()
        for slot, ids in route_batch(model, h, yb, corpus, ledger, i):
            _expert_train(model, slot, h, yb, ids, lr)


@torch.no_grad()
def eval_bpc_fusion(model, xs, ys, routes=None):
    """PR-07' eval_bpc math over COMBINED logits; each eval segment is served by its
    argmin-surprise committed expert (route_for_inference analog, no recruitment).
    `routes` (optional list) accumulates per-expert eval assignment counts."""
    model.eval()
    tot_nll, tot_tok = 0.0, 0
    for i in range(0, xs.shape[0], BATCH_SEGS):
        xb, yb = xs[i: i + BATCH_SEGS], ys[i: i + BATCH_SEGS]
        logits, h = model.lm(xb), model._h
        B, T = yb.shape
        S, slots = _segment_surprise(model, h, yb)
        if S.numel() == 0:
            comb = logits
        else:
            min_idx = S.argmin(dim=0)
            if routes is not None:
                for s_i, s in enumerate(slots):
                    routes[s] += int((min_idx == s_i).sum())
            comb = logits.clone()
            for s_i, s in enumerate(slots):
                mask = min_idx == s_i
                if bool(mask.any()):
                    hs = h[mask].reshape(-1, h.shape[-1])
                    comb[mask] = comb[mask] + model.experts[s](hs).view(mask.sum(), T, -1)
        nll = F.cross_entropy(comb.reshape(-1, comb.shape[-1]).float(), yb.reshape(-1),
                              reduction="sum")
        tot_nll += float(nll)
        tot_tok += yb.numel()
    return (tot_nll / tot_tok) / np.log(2)


def build_model(vocab, seed, E):
    return FusionLM(vocab, seed, E) if E > 0 else _PlainWrap(vocab, seed)


class _PlainWrap(nn.Module):
    """The plain mixer: the PR-07' model verbatim (same seed stream -> same init)."""

    def __init__(self, vocab, seed):
        super().__init__()
        torch.manual_seed(seed)
        cfg = PrizmaSeqConfig(vocab=vocab, d_model=64, n_layers=2, n_heads=4, chunk=64,
                              window=16, max_len=SEG)
        self.lm = PrizmaSeqLM(cfg)


# ----------------------------- protocol ------------------------------------- #

def save(doc):
    p = os.path.join(OUT, "raw_smoke.json" if SMOKE else "raw.json")
    tmp = p + ".tmp"
    json.dump(doc, open(tmp, "w", encoding="utf-8"), indent=1, default=str)
    os.replace(tmp, p)


def ledger_snapshot(model, ledger, tr_A, tr_B, ev_A, ev_B):
    """Routing ledger + corpus-purity statistics (the specialization check)."""
    per_expert = []
    for s in range(model.E):
        if not model.committed[s]:
            per_expert.append({"slot": s, "committed": False})
            continue
        n = tr_A[s] + tr_B[s]
        per_expert.append({
            "slot": s, "committed": True,
            "train_A": tr_A[s], "train_B": tr_B[s], "train_total": n,
            "expert_majority_corpus_share": (max(tr_A[s], tr_B[s]) / n) if n else None,
            "mu": round(model.mu[s], 4), "sigma": round(max(model.var[s], 0.0) ** 0.5, 4),
            "mean_calib_ce": round(model.ce_sum[s] / model.n_segments[s], 4) if model.n_segments[s] else None,
            "n_batches": model.n_batches[s], "n_segments": model.n_segments[s],
            "eval_A": ev_A[s], "eval_B": ev_B[s],
        })
    nA, nB = sum(tr_A), sum(tr_B)
    purity = {
        "A_majority_expert_share": (max(tr_A) / nA) if nA else None,
        "B_majority_expert_share": (max(tr_B) / nB) if nB else None,
        "mean_expert_majority_corpus_share": float(np.mean([
            p["expert_majority_corpus_share"] for p in per_expert
            if p.get("committed") and p["train_total"] > 0])) if any(
            p.get("committed") and p["train_total"] > 0 for p in per_expert) else None,
    }
    return {"n_committed": model.n_committed(), "recruits": ledger["recruits"],
            "cap_fallback_batches": ledger["cap_fallback_batches"],
            "cap_fallback_segments": ledger["cap_fallback_segments"],
            "cap_events": ledger["cap_events"][:20],
            "per_expert": per_expert, "purity": purity}


def run_cell(config, E, seed, vocab, data, lr, wall_first):
    t0 = time.time()
    model = build_model(vocab, seed, E)
    Ax, Ay, Bx, By, Aex, Aey, Bex, Bey = data
    rec = {"config": config, "E": E, "seed": seed, "lr": lr}
    if E == 0:
        claim.train_stream(model.lm, Ax, Ay, lr, seed)
        rec["bpc_A_pre"] = claim.eval_bpc(model.lm, Aex, Aey)
        claim.train_stream(model.lm, Bx, By, lr, seed)
        rec["bpc_A_post"] = claim.eval_bpc(model.lm, Aex, Aey)
        rec["bpc_B"] = claim.eval_bpc(model.lm, Bex, Bey)
    else:
        ledger = {"recruits": [], "cap_fallback_batches": 0, "cap_fallback_segments": 0,
                  "cap_events": []}
        tr_A = [0] * E; tr_B = [0] * E
        train_stream_fusion(model, Ax, Ay, lr, "A", ledger)
        for s in range(E):
            if model.committed[s]:
                tr_A[s] = model.n_segments[s]
        rec["bpc_A_pre"] = eval_bpc_fusion(model, Aex, Aey)
        before = [model.n_segments[s] for s in range(E)]
        train_stream_fusion(model, Bx, By, lr, "B", ledger)
        for s in range(E):
            if model.committed[s]:
                tr_B[s] = model.n_segments[s] - before[s]
        rec["bpc_A_post"] = eval_bpc_fusion(model, Aex, Aey)
        ev_B = [0] * E
        rec["bpc_B"] = eval_bpc_fusion(model, Bex, Bey, routes=ev_B)
        # eval-time routing ledger on the final (post-B) model — the state that
        # retention bpc_A_post is measured on
        ev_A = [0] * E
        eval_bpc_fusion(model, Aex, Aey, routes=ev_A)
        rec["ledger"] = ledger_snapshot(model, ledger, tr_A, tr_B, ev_A, ev_B)
    rec["fgt_A"] = rec["bpc_A_pre"] - rec["bpc_A_post"]
    rec["wall_s"] = round(time.time() - t0, 1)
    if wall_first is not None:
        rec["first_cell_note"] = wall_first
    return rec


def select_lr(config, E, vocab, Ax, Ay, seed0):
    """PR-07' addendum rule: one LR per arm-family, chosen on seed 0's first-200
    A-segment loss, then frozen. Fusion cells score the COMBINED model bpc."""
    grid = {}
    xs, ys = Ax[:LR_SELECT_SEGS], Ay[:LR_SELECT_SEGS]
    for lr in LR_GRID:
        m = build_model(vocab, seed0, E)
        if E == 0:
            claim.train_stream(m.lm, xs, ys, lr, seed0)
            grid[lr] = claim.eval_bpc(m.lm, xs, ys)
        else:
            led = {"recruits": [], "cap_fallback_batches": 0, "cap_fallback_segments": 0,
                   "cap_events": []}
            train_stream_fusion(m, xs, ys, lr, "A", led)
            grid[lr] = eval_bpc_fusion(m, xs, ys)
        print(f"[fusion] lr-select {config}: lr={lr} A-loss={grid[lr]:.4f}", flush=True)
    best = min(grid, key=grid.get)
    return best, grid


def main():
    torch.set_num_threads(int(os.environ.get("BENCH_THREADS", "8")))
    os.makedirs(OUT, exist_ok=True)
    A_all, B_all = claim.fetch_corpora()
    A_train, A_eval = A_all[:claim.A_CHARS], A_all[claim.A_CHARS:claim.A_CHARS + claim.A_EVAL_CHARS]
    B_train, B_eval = B_all[: int(len(B_all) * 0.9)], B_all[int(len(B_all) * 0.9):]
    chars = sorted(set(A_all[:claim.A_CHARS + claim.A_EVAL_CHARS]) | set(B_all))
    vocab = {c: i for i, c in enumerate(chars)}
    V = len(vocab)
    Ax, Ay = claim.make_segments(A_train, vocab)
    Bx, By = claim.make_segments(B_train, vocab)
    Aex, Aey = claim.make_segments(A_eval, vocab)
    Bex, Bey = claim.make_segments(B_eval, vocab)
    if SMOKE:
        Ax, Ay, Bx, By = Ax[:64], Ay[:64], Bx[:64], By[:64]
    data = (Ax, Ay, Bx, By, Aex, Aey, Bex, Bey)

    plain_params = sum(p.numel() for p in _PlainWrap(V, 0).lm.parameters())
    expert_params = sum(p.numel() for p in PCExpertHead(64, H_SMALL, V).parameters())
    doc = {"meta": {
        "protocol": "fusion_probe (LANE-EXPLORATORY — design-informing, never a claim)",
        "question": "does vigilance-routed expert tissue change retention/adaptation vs "
                    "the plain mixer on the PR-07' block-drift stream?",
        "stream": "text8[0:1.0M) -> tiny-shakespeare, one pass, char-level (PR-07' verbatim)",
        "backbone": "PrizmaSeqLM 2-layer d_model=64 H=4 chunk=64 window=16 (PR-07' pinned)",
        "variant": "PREFERRED: backbone TRAINS on all segments (shared LM loss); expert "
                   "heads local (private AdamW, detached base+hidden, no router gradient)",
        "routing": f"per-segment vigilance: z=(segCE-mu)/sigma <= {Z_NOVEL} (shipped "
                   f"z_novel); mu/sigma EMAs rate {FLOOR_EMA} over OWN trained segments "
                   "(src/prizma.py mirror); recruit next slot; at cap m_max=E -> "
                   "argmin-surprise fallback (PR-05 Policy A relationship documented)",
        "expert_head": f"PCExpertHead Wenc 64->{H_SMALL} tanh, Wdec {H_SMALL}->{V}; "
                       "residual corrector on detached base logits",
        "smoke": SMOKE, "vocab": V, "seg": SEG, "batch_segs": BATCH_SEGS, "seeds": SEEDS,
        "segments": {"A_train": int(Ax.shape[0]), "B_train": int(Bx.shape[0]),
                     "A_eval": int(Aex.shape[0]), "B_eval": int(Bex.shape[0])},
        "params": {"backbone": plain_params, "per_expert": expert_params,
                   "pool_e4": 4 * expert_params, "pool_e8": 8 * expert_params},
        "threads": torch.get_num_threads(),
        "lrs_rule": "per arm-family, seed-0 first-200-A-segment loss, then frozen",
    }, "lr_selection": {}, "cells": []}
    save(doc)
    print(f"[fusion] corpora loaded: A={Ax.shape[0]} B={Bx.shape[0]} evalA={Aex.shape[0]} "
          f"evalB={Bex.shape[0]} vocab={V} smoke={SMOKE}", flush=True)

    lrs = {}
    for config, E in CONFIGS:
        if SMOKE:
            lrs[config] = (3e-3, {})
        else:
            lrs[config] = select_lr(config, E, V, Ax, Ay, 0)
        doc["lr_selection"][config] = {"lr": lrs[config][0], "grid": lrs[config][1]}
        save(doc)

    wall_first = None
    for config, E in CONFIGS:
        for seed in SEEDS:
            note = None
            rec = run_cell(config, E, seed, V, data, lrs[config][0], wall_first)
            if wall_first is None:
                wall_first = (f"first cell ({config} seed {seed}) wall {rec['wall_s']}s — "
                              f"budget extrapolates to "
                              f"{round(rec['wall_s'] * 6 / 60, 1)} min for all 6 cells")
                rec["first_cell_note"] = wall_first
            doc["cells"].append(rec)
            save(doc)
            print(f"[fusion] {config} seed {seed}: A_pre={rec['bpc_A_pre']:.3f} "
                  f"A_post={rec['bpc_A_post']:.3f} FGT={rec['fgt_A']:+.3f} "
                  f"B={rec['bpc_B']:.3f} wall={rec['wall_s']}s", flush=True)

    # ---- summary (means over seeds; n=2 -> descriptive, no claim) ----
    summary = {}
    for config, E in CONFIGS:
        cells = [c for c in doc["cells"] if c["config"] == config]
        summary[config] = {
            "E": E, "n_seeds": len(cells),
            "bpc_A_pre": round(float(np.mean([c["bpc_A_pre"] for c in cells])), 4),
            "bpc_A_post": round(float(np.mean([c["bpc_A_post"] for c in cells])), 4),
            "fgt_A": round(float(np.mean([c["fgt_A"] for c in cells])), 4),
            "bpc_B": round(float(np.mean([c["bpc_B"] for c in cells])), 4),
            "wall_s": round(float(np.sum([c["wall_s"] for c in cells])), 1),
        }
    doc["summary"] = summary
    doc["reference_pr07"] = {"stream_bpc_B_2.954": 2.954, "stream_fgt_A": -0.189,
                             "note": "plain-mixer n=5 registered reference "
                                     "(results/blockdrift_PR-2026-09-03-07/RESULTS.md)"}
    save(doc)
    print("\n[fusion] SUMMARY (n=2 seeds, descriptive):", flush=True)
    for config, _ in CONFIGS:
        s = summary[config]
        print(f"  {config}: A_pre={s['bpc_A_pre']} A_post={s['bpc_A_post']} "
              f"FGT_A={s['fgt_A']} B={s['bpc_B']}", flush=True)
    print(f"[fusion] wrote {os.path.join(OUT, 'raw_smoke.json' if SMOKE else 'raw.json')}",
          flush=True)


if __name__ == "__main__":
    main()
