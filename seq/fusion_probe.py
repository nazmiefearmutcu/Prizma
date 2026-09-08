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
    # PR-2026-09-03-09 floor-freeze lever (guarded, DEFAULT-OFF; frozen protocol
    # docs/preregistry/2026-09-09-floorfreeze-routing-repair.md §2): when the model carries a
    # `floor_freeze` set of slot indices, precision-floor updates for THOSE slots are SKIPPED
    # (mu/var stay pinned at their block-A calibration; the expert optimizer step and the
    # n_batches/n_segments/ce_sum bookkeeping above are NOT frozen). Attribute absent ->
    # freeze is None -> the exact PR-08 operations below, unchanged in order (off-identity
    # pinned behaviorally by tests/test_floorfreeze_runner.py).
    freeze = getattr(model, "floor_freeze", None)
    if not (freeze is not None and slot in freeze):
        if model.mu[slot] > 1e8:                   # first calibration seeds the floor
            model.mu[slot] = r
            model.var[slot] = max(1e-4, (0.1 * r) ** 2)
        else:                                      # EMA, mirrored verbatim rates
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


# ============================================================================
# probe2 — completion probes for the PR-LM-1 design (2026-09-08 session)
# ----------------------------------------------------------------------------
# LANE-EXPLORATORY add-on (docs/preregistry/POLICY.md: design-informing, never a
# claim). Everything below is ADDITIVE: the original probe's functions, defaults
# and CLI behavior above are untouched — running `python seq/fusion_probe.py`
# with no argument reproduces the 2026-09-07 probe exactly. New modes are
# selected with the first CLI argument:
#   python seq/fusion_probe.py frozen    # P1: frozen-backbone variant (2-block)
#   python seq/fusion_probe.py shared    # P2: shared-extra-head capacity control
#   python seq/fusion_probe.py return    # P3: returning-domain A->B->C probe
#   python seq/fusion_probe.py all2      # all three, in that order
# FUSION_PROBE_SMOKE=1 shrinks any mode to a wiring check (own *_smoke.json file,
# never the real raw files). Output: results/exploratory/fusion_probe2_2026-09-08/
# (crash-safe per-cell JSON, tmp + os.replace).
#
# P1 FROZEN-BACKBONE: backbone weights FROZEN after corpus A; only the routed
#     expert tissue learns through B. Question: does the expert tissue alone
#     carry adaptation when the shared trunk cannot move? (trunk = general,
#     experts = domain tissue — the fusion design's cleanest division of labor.)
# P2 SHARED-EXTRA-HEAD control: the capacity confound the 2026-09-07 probe
#     flagged. ONE shared head (no routing, no recruitment) trained on ALL
#     segments with the exact _expert_train learning rule. Two widths: h=64
#     (same size as ONE expert head) and h=256 (~ the fusion E=4 pool budget:
#     33,345 vs 33,540 params). If shared_h256 ~ fusion E=4 on bpc_B/FGT, the
#     fusion gain was capacity, not routing.
# P3 RETURNING-DOMAIN: three-block stream text8[0:1M) -> shakespeare ->
#     text8[1.1M:2.0M) (C = fresh text8, text-disjoint from A; the slice
#     text8[1.0M:1.1M) is SKIPPED because it is A_eval — training on it would
#     corrupt the retention probe; C_eval = text8[2.0M:2.1M)). Measures: does
#     the router RE-ROUTE C segments to the existing text8 expert (pattern
#     completion) or RECRUIT a fresh one (treats it as novel); C bpc under
#     re-routed vs forced-fresh handling (two arms); A/C retention after the
#     full stream; the routing ledger per block. Direct CPU precursor of
#     PR-LM-1's many-block stream design requirement.
# ============================================================================

OUT2 = os.path.join(_ROOT, "results", "exploratory", "fusion_probe2_2026-09-08")
C_OFFSET = 1_100_000        # C starts AFTER the A_eval slice text8[1.0M:1.1M)
C_TRAIN_CHARS = 900_000
C_EVAL_CHARS = 100_000
SHARED_WIDTHS = {"shared_h64": 64, "shared_h256": 256}
CKPT_FRACS = (0.25, 0.50, 0.75)   # C-training checkpoints (fractions of C_train)


def _fresh_ledger():
    return {"recruits": [], "cap_fallback_batches": 0, "cap_fallback_segments": 0,
            "cap_events": []}


def save2(doc, name):
    p = os.path.join(OUT2, (name + "_smoke.json") if SMOKE else (name + ".json"))
    tmp = p + ".tmp"
    json.dump(doc, open(tmp, "w", encoding="utf-8"), indent=1, default=str)
    os.replace(tmp, p)


def _bb_snapshot(model):
    return [p.detach().clone() for p in model.lm.parameters()]


def _bb_equal(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))


# ----------------------------- P1: frozen backbone --------------------------- #

def train_stream_fusion_frozen(model, xs, ys, lr, corpus, ledger):
    """FROZEN-BACKBONE phase: the backbone receives NO update (forward under
    no_grad, no optimizer step); ONLY the routed expert tissue learns. Everything
    else (segment-level vigilance routing, per-batch private AdamW local expert
    steps, floor EMAs) mirrors train_stream_fusion verbatim."""
    model.train()
    n = xs.shape[0]
    for i in range(0, n, BATCH_SEGS):
        xb, yb = xs[i: i + BATCH_SEGS], ys[i: i + BATCH_SEGS]
        with torch.no_grad():
            model.lm(xb)                            # hook captures hidden
        h = model._h
        for slot, ids in route_batch(model, h, yb, corpus, ledger, i):
            _expert_train(model, slot, h, yb, ids, lr)


def run_cell_frozen(seed, vocab, data, lr):
    """P1 cell: fusion E=4 through A (backbone TRAINS, verbatim), then B with the
    backbone FROZEN. Backbone-freeze is proven by a parameter-identity check."""
    t0 = time.time()
    model = build_model(len(vocab), seed, 4)
    Ax, Ay, Bx, By, Aex, Aey, Bex, Bey = data
    E = 4
    ledger = _fresh_ledger()
    tr_A = [0] * E; tr_B = [0] * E
    rec = {"config": "frozen_e4", "E": E, "seed": seed, "lr": lr}
    train_stream_fusion(model, Ax, Ay, lr, "A", ledger)     # phase A verbatim
    tr_A = model.n_segments[:]
    rec["bpc_A_pre"] = eval_bpc_fusion(model, Aex, Aey)
    rec["bpc_A_pre_base"] = claim.eval_bpc(model.lm, Aex, Aey)   # base-only reference
    bb = _bb_snapshot(model)
    train_stream_fusion_frozen(model, Bx, By, lr, "B", ledger)
    rec["backbone_frozen_check"] = bool(_bb_equal(bb, _bb_snapshot(model)))
    tr_B = [model.n_segments[s] - tr_A[s] for s in range(E)]
    rec["bpc_A_post"] = eval_bpc_fusion(model, Aex, Aey)
    ev_B = [0] * E
    rec["bpc_B"] = eval_bpc_fusion(model, Bex, Bey, routes=ev_B)
    rec["bpc_B_base"] = claim.eval_bpc(model.lm, Bex, Bey)       # FROZEN-arm analog
    ev_A = [0] * E
    eval_bpc_fusion(model, Aex, Aey, routes=ev_A)
    rec["ledger"] = ledger_snapshot(model, ledger, tr_A, tr_B, ev_A, ev_B)
    rec["fgt_A"] = rec["bpc_A_pre"] - rec["bpc_A_post"]
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def run_frozen():
    torch.set_num_threads(int(os.environ.get("BENCH_THREADS", "8")))
    os.makedirs(OUT2, exist_ok=True)
    vocab, data = _load_2block_data()
    if SMOKE:
        Ax, Ay, Bx, By, Aex, Aey, Bex, Bey = data
        data = (Ax[:64], Ay[:64], Bx[:64], By[:64], Aex, Aey, Bex, Bey)
    doc = {"meta": {
        "probe": "P1 FROZEN-BACKBONE (LANE-EXPLORATORY — design-informing, never a claim)",
        "question": "does the expert tissue ALONE carry adaptation when the shared "
                    "trunk cannot move? (backbone frozen after corpus A)",
        "stream": "text8[0:1M) -> tiny-shakespeare, one pass (PR-07' verbatim)",
        "phase_A": "verbatim train_stream_fusion: backbone + e0 train (same A-phase as "
                   "the 2026-09-07 fusion probe)",
        "phase_B": "train_stream_fusion_frozen: backbone forward under no_grad, NO "
                   "backbone optimizer step; expert routing + local expert steps unchanged",
        "lr_note": "fusion-e4 family LR re-selected in-session by the frozen PR-07' rule "
                   "(seed 0, first-200-A-segment loss); the same LR drives phase-A and "
                   "the phase-B expert steps (A-phase selection cannot distinguish a "
                   "frozen-B arm; disclosed)",
        "references": {"pr07_FROZEN_bpc_B_mean_n5": 6.132,
                       "fusion_e4_backbone_trained_2026-09-07": {"bpc_B": 2.8937,
                                                                 "fgt_A": -0.1934}},
        "smoke": SMOKE, "seeds": [0] if SMOKE else SEEDS,
        "threads": torch.get_num_threads(),
    }, "lr_selection": {}, "cells": []}
    save2(doc, "frozen")
    if SMOKE:
        lr = 3e-3
    else:
        lr, grid = select_lr("fusion_e4", 4, len(vocab), data[0], data[1], 0)
        doc["lr_selection"]["frozen_e4"] = {"lr": lr, "grid": grid}
        save2(doc, "frozen")
    seeds = [0] if SMOKE else SEEDS
    wall_first = None
    for seed in seeds:
        rec = run_cell_frozen(seed, vocab, data, lr)
        if wall_first is None:
            wall_first = (f"first cell wall {rec['wall_s']}s — budget extrapolates to "
                          f"{round(rec['wall_s'] * len(seeds) / 60, 1)} min for this probe")
            rec["first_cell_note"] = wall_first
        doc["cells"].append(rec)
        save2(doc, "frozen")
        print(f"[p1-frozen] seed {seed}: A_pre={rec['bpc_A_pre']:.3f} "
              f"A_post={rec['bpc_A_post']:.3f} FGT={rec['fgt_A']:+.3f} "
              f"B={rec['bpc_B']:.3f} (base-only {rec['bpc_B_base']:.3f}) "
              f"frozen_ok={rec['backbone_frozen_check']} wall={rec['wall_s']}s", flush=True)
    cells = doc["cells"]
    doc["summary"] = {
        "n_seeds": len(cells),
        "bpc_A_pre": round(float(np.mean([c["bpc_A_pre"] for c in cells])), 4),
        "bpc_A_post": round(float(np.mean([c["bpc_A_post"] for c in cells])), 4),
        "fgt_A": round(float(np.mean([c["fgt_A"] for c in cells])), 4),
        "bpc_B_combined": round(float(np.mean([c["bpc_B"] for c in cells])), 4),
        "bpc_B_base_only": round(float(np.mean([c["bpc_B_base"] for c in cells])), 4),
        "backbone_frozen_all_seeds": all(c["backbone_frozen_check"] for c in cells),
        "wall_s": round(float(np.sum([c["wall_s"] for c in cells])), 1),
    }
    save2(doc, "frozen")
    print(f"[p1-frozen] summary: {doc['summary']}", flush=True)


# ----------------------------- P2: shared extra head ------------------------- #

class SharedHeadLM(nn.Module):
    """Plain PR-07' backbone + ONE shared extra head: the capacity control. No
    routing, no recruitment, no vigilance — the head trains on ALL segments with
    the exact _expert_train learning rule (detached base logits + detached hidden,
    private per-batch AdamW). Same seed stream as build_model -> identical
    backbone init."""

    def __init__(self, vocab, seed, h_small):
        super().__init__()
        torch.manual_seed(seed)
        cfg = PrizmaSeqConfig(vocab=vocab, d_model=64, n_layers=2, n_heads=4, chunk=64,
                              window=16, max_len=SEG)
        self.lm = PrizmaSeqLM(cfg)
        self.head_x = PCExpertHead(64, h_small, vocab)
        self._h = None
        self.lm.head.register_forward_hook(lambda m, i, o: setattr(self, "_h", i[0]))

    def forward(self, idx):
        return self.lm(idx)


def train_stream_shared(model, xs, ys, lr):
    """Backbone steps FIRST on the base LM loss (objective unchanged vs plain and
    fusion); the shared head then takes ONE local AdamW step on the WHOLE batch —
    the exact _expert_train math (post-update base logits recomputed on detached
    pre-step hidden, mirrored verbatim), minus routing."""
    opt_bb = torch.optim.AdamW(model.lm.parameters(), lr=lr)
    model.train()
    n = xs.shape[0]
    for i in range(0, n, BATCH_SEGS):
        xb, yb = xs[i: i + BATCH_SEGS], ys[i: i + BATCH_SEGS]
        opt_bb.zero_grad()
        logits = model.lm(xb)
        h = model._h
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), yb.reshape(-1))
        loss.backward()
        opt_bb.step()
        B, T, _ = h.shape
        hs = h.detach().reshape(B * T, -1)
        ys_flat = yb.reshape(-1)
        base = model.lm.head(h.detach()).reshape(B * T, -1)
        opt_h = torch.optim.AdamW(model.head_x.parameters(), lr=lr)
        opt_h.zero_grad()
        comb = base + model.head_x(hs)
        F.cross_entropy(comb, ys_flat).backward()
        opt_h.step()


@torch.no_grad()
def eval_bpc_shared(model, xs, ys):
    """bpc over COMBINED logits (base + shared head), PR-07' eval math."""
    model.eval()
    tot_nll, tot_tok = 0.0, 0
    for i in range(0, xs.shape[0], BATCH_SEGS):
        xb, yb = xs[i: i + BATCH_SEGS], ys[i: i + BATCH_SEGS]
        logits, h = model.lm(xb), model._h
        comb = logits + model.head_x(h)
        nll = F.cross_entropy(comb.reshape(-1, comb.shape[-1]).float(), yb.reshape(-1),
                              reduction="sum")
        tot_nll += float(nll)
        tot_tok += yb.numel()
    return (tot_nll / tot_tok) / np.log(2)


def select_lr_shared(h_small, vocab, Ax, Ay, seed0):
    """PR-07' addendum rule applied to a shared-head family: seed 0, first-200
    A-segment loss, grid LR_GRID, then frozen."""
    grid = {}
    xs, ys = Ax[:LR_SELECT_SEGS], Ay[:LR_SELECT_SEGS]
    for lr in LR_GRID:
        m = SharedHeadLM(len(vocab), seed0, h_small)
        train_stream_shared(m, xs, ys, lr)
        grid[lr] = eval_bpc_shared(m, xs, ys)
        print(f"[p2-shared] lr-select h={h_small}: lr={lr} A-loss={grid[lr]:.4f}", flush=True)
    best = min(grid, key=grid.get)
    return best, grid


def run_cell_shared(config, h_small, seed, vocab, data, lr):
    t0 = time.time()
    model = SharedHeadLM(len(vocab), seed, h_small)
    Ax, Ay, Bx, By, Aex, Aey, Bex, Bey = data
    rec = {"config": config, "h_small": h_small, "seed": seed, "lr": lr,
           "routing": "none (single shared head, trained on ALL segments)"}
    train_stream_shared(model, Ax, Ay, lr)
    rec["bpc_A_pre"] = eval_bpc_shared(model, Aex, Aey)
    train_stream_shared(model, Bx, By, lr)
    rec["bpc_A_post"] = eval_bpc_shared(model, Aex, Aey)
    rec["bpc_B"] = eval_bpc_shared(model, Bex, Bey)
    rec["fgt_A"] = rec["bpc_A_pre"] - rec["bpc_A_post"]
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def run_shared():
    torch.set_num_threads(int(os.environ.get("BENCH_THREADS", "8")))
    os.makedirs(OUT2, exist_ok=True)
    vocab, data = _load_2block_data()
    if SMOKE:
        Ax, Ay, Bx, By, Aex, Aey, Bex, Bey = data
        data = (Ax[:64], Ay[:64], Bx[:64], By[:64], Aex, Aey, Bex, Bey)
    head_params = {name: sum(p.numel() for p in PCExpertHead(64, h, len(vocab)).parameters())
                   for name, h in SHARED_WIDTHS.items()}
    doc = {"meta": {
        "probe": "P2 SHARED-EXTRA-HEAD capacity control (LANE-EXPLORATORY — design-"
                 "informing, never a claim)",
        "question": "was the 2026-09-07 fusion bpc gain CAPACITY (extra params) or "
                    "ROUTING? Same extra parameter budget as fusion E=4, but ONE shared "
                    "head (no routing, no recruitment) trained on all segments.",
        "arms": {"plain": "re-run for within-session comparability (verbatim run_cell)",
                 "fusion_e4": "re-run for within-session comparability (verbatim run_cell)",
                 "shared_h64": "one head, same size as ONE expert head (8,385 params)",
                 "shared_h256": "one wide head ~ fusion E=4 pool budget (33,345 vs "
                                "4x8,385=33,540 params)"},
        "learning_rule": "shared head = exact _expert_train rule (detached base+hidden, "
                         "private per-batch AdamW) on EVERY segment; backbone objective "
                         "unchanged (base LM loss); eval over combined logits",
        "budget_note": "at home only 2 of the E=4 experts ever commit, so the TRAINED "
                       "extra-parameter budget of fusion E=4 is 2x8,385=16,770 while the "
                       "ALLOCATED pool budget is 33,540; both controls bracket this",
        "smoke": SMOKE, "seeds": [0] if SMOKE else SEEDS,
        "threads": torch.get_num_threads(),
        "head_params": head_params,
    }, "lr_selection": {}, "cells": []}
    save2(doc, "shared")
    lrs = {}
    if SMOKE:
        lrs = {"plain": 3e-3, "fusion_e4": 3e-3, "shared_h64": 3e-3, "shared_h256": 3e-3}
    else:
        for config, E in [("plain", 0), ("fusion_e4", 4)]:
            lr, grid = select_lr(config, E, len(vocab), data[0], data[1], 0)
            lrs[config] = lr
            doc["lr_selection"][config] = {"lr": lr, "grid": grid}
        for config, h in SHARED_WIDTHS.items():
            lr, grid = select_lr_shared(h, vocab, data[0], data[1], 0)
            lrs[config] = lr
            doc["lr_selection"][config] = {"lr": lr, "grid": grid}
        save2(doc, "shared")

    def cells_for(config):
        return [c for c in doc["cells"] if c["config"] == config]

    wall_first = None
    for config in ["plain", "fusion_e4", "shared_h64", "shared_h256"]:
        for seed in ([0] if SMOKE else SEEDS):
            if config in SHARED_WIDTHS:
                rec = run_cell_shared(config, SHARED_WIDTHS[config], seed, vocab, data,
                                      lrs[config])
            elif config == "fusion_e4":
                rec = run_cell("fusion_e4", 4, seed, len(vocab), data, lrs[config], None)
            else:
                rec = run_cell("plain", 0, seed, len(vocab), data, lrs[config], None)
            if wall_first is None:
                wall_first = (f"first cell wall {rec['wall_s']}s; 8 cells + lr-selects "
                              f"budgeted separately per family")
                rec["first_cell_note"] = wall_first
            doc["cells"].append(rec)
            save2(doc, "shared")
            print(f"[p2-shared] {config} seed {seed}: A_pre={rec['bpc_A_pre']:.3f} "
                  f"A_post={rec['bpc_A_post']:.3f} FGT={rec['fgt_A']:+.3f} "
                  f"B={rec['bpc_B']:.3f} wall={rec['wall_s']}s", flush=True)
    doc["summary"] = {}
    for config in ["plain", "fusion_e4", "shared_h64", "shared_h256"]:
        cells = cells_for(config)
        doc["summary"][config] = {
            "n_seeds": len(cells),
            "bpc_A_pre": round(float(np.mean([c["bpc_A_pre"] for c in cells])), 4),
            "bpc_A_post": round(float(np.mean([c["bpc_A_post"] for c in cells])), 4),
            "fgt_A": round(float(np.mean([c["fgt_A"] for c in cells])), 4),
            "bpc_B": round(float(np.mean([c["bpc_B"] for c in cells])), 4),
            "wall_s": round(float(np.sum([c["wall_s"] for c in cells])), 1),
        }
    save2(doc, "shared")
    print(f"[p2-shared] summary: {json.dumps(doc['summary'])}", flush=True)


# ----------------------------- P3: returning domain -------------------------- #

def route_batch_return(model, h, y, corpus, ledger, stream_pos, force_slot=None):
    """P3 routing: verbatim route_batch semantics, plus an optional forced-recruit
    override for corpus 'C' — the treat-as-novel counterfactual arm. When forced,
    C segments NEVER route to pre-C experts: the first C batch commits the forced
    slot, which then owns ALL C segments (exclusive ownership) — exactly the
    training-ledger shape vigilance-recruitment would have produced had the C
    boundary fired. Routing only: the backbone objective is identical across arms."""
    if force_slot is not None and corpus == "C":
        B = y.shape[0]
        if not model.committed[force_slot]:
            model.committed[force_slot] = True
            ledger["recruits"].append({"slot": force_slot, "at_batch": stream_pos,
                                       "corpus": corpus, "n_novel_segs": int(B),
                                       "trigger_z_max": None, "reason": "forced_novel_C"})
        return [(force_slot, list(range(B)))]
    return route_batch(model, h, y, corpus, ledger, stream_pos)


def train_stream_C(model, xs, ys, lr, ledger, force_slot=None, ckpt_fn=None,
                   ckpt_marks=()):
    """Block-C training with train_stream_fusion semantics verbatim (backbone step
    first on the whole batch, then routed local expert steps), GLOBAL batch
    positions in the ledger, the optional forced-recruit override, and eval
    checkpoints after the given global segment marks."""
    opt_bb = torch.optim.AdamW(model.lm.parameters(), lr=lr)
    model.train()
    n = xs.shape[0]
    marks = sorted(ckpt_marks)
    for i in range(0, n, BATCH_SEGS):
        xb, yb = xs[i: i + BATCH_SEGS], ys[i: i + BATCH_SEGS]
        opt_bb.zero_grad()
        logits = model.lm(xb)
        h = model._h
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), yb.reshape(-1))
        loss.backward()
        opt_bb.step()
        for slot, ids in route_batch_return(model, h, yb, "C", ledger, i,
                                            force_slot=force_slot):
            _expert_train(model, slot, h, yb, ids, lr)
        while marks and i + xb.shape[0] >= marks[0]:
            marks.pop(0)
            if ckpt_fn is not None:
                ckpt_fn(i + xb.shape[0])


def train_stream_plain_C(model, xs, ys, lr, seed, ckpt_fn=None, ckpt_marks=()):
    """claim.train_stream verbatim (one AdamW for the whole block) + checkpoints —
    the plain 3-block reference arm's C phase."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    n = xs.shape[0]
    marks = sorted(ckpt_marks)
    for i in range(0, n, BATCH_SEGS):
        xb, yb = xs[i: i + BATCH_SEGS], ys[i: i + BATCH_SEGS]
        opt.zero_grad()
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), yb.reshape(-1))
        loss.backward()
        opt.step()
        while marks and i + xb.shape[0] >= marks[0]:
            marks.pop(0)
            if ckpt_fn is not None:
                ckpt_fn(i + xb.shape[0])


def ledger_snapshot3(model, ledger, tr, ev_routes, expected):
    """Per-block routing ledger (the P3 requirement: ledger per block per config)."""
    per_expert = []
    for s in range(model.E):
        if not model.committed[s]:
            per_expert.append({"slot": s, "committed": False})
            continue
        tot = tr["A"][s] + tr["B"][s] + tr["C"][s]
        per_expert.append({
            "slot": s, "committed": True,
            "train_A": tr["A"][s], "train_B": tr["B"][s], "train_C": tr["C"][s],
            "train_total": tot,
            "mu": round(model.mu[s], 4), "sigma": round(max(model.var[s], 0.0) ** 0.5, 4),
            "n_batches": model.n_batches[s], "n_segments": model.n_segments[s],
        })
    block_majority = {}
    balance = {}
    for blk in ("A", "B", "C"):
        n = sum(tr[blk])
        block_majority[blk] = (round(max(tr[blk]) / n, 4) if n else None)
        balance[blk] = {"sum": n, "expected": expected[blk], "ok": n == expected[blk]}
    return {"n_committed": model.n_committed(), "recruits": ledger["recruits"],
            "cap_fallback_batches": ledger["cap_fallback_batches"],
            "cap_fallback_segments": ledger["cap_fallback_segments"],
            "cap_events": ledger["cap_events"][:50],
            "per_expert": per_expert,
            "block_majority_share": block_majority,
            "train_sums_balance": balance,
            "eval_routes": ev_routes}


def _c_batch0_surprise(model, Cx, Cy):
    """Diagnostic: per-expert surprise of the FIRST C batch under every committed
    expert, with the vigilance z each expert would assign — mechanism color for the
    re-route-vs-recruit verdict, recorded regardless of outcome."""
    with torch.no_grad():
        xb, yb = Cx[:BATCH_SEGS], Cy[:BATCH_SEGS]
        model.eval()
        _ = model.lm(xb)
        h = model._h
        S, slots = _segment_surprise(model, h, yb)
        out = {}
        for i, s in enumerate(slots):
            mu, sd = model.mu[s], max(model.var[s], 1e-12) ** 0.5
            z = (S[i] - mu) / sd
            out[f"e{s}"] = {"mean_ce": round(float(S[i].mean()), 4),
                            "mu_floor": round(mu, 4), "sigma_floor": round(sd, 4),
                            "z_mean": round(float(z.mean()), 2),
                            "z_max": round(float(z.max()), 2),
                            "frac_novel_z_gt5": round(float((z > Z_NOVEL).float().mean()), 4)}
        model.train()
    return out


def _make_ckpt_fn(model, Cex, Cey, Aex, Aey, E, store, routed):
    """Checkpoint closure: records the re-mastery curve (C-eval bpc, A-eval bpc and
    — for routed models — the expert serving each C_eval segment) at each mark,
    then restores train mode."""
    def ckpt(done):
        routes = [0] * E if routed else None
        if routed:
            bpc_C = eval_bpc_fusion(model, Cex, Cey, routes=routes)
            bpc_A = eval_bpc_fusion(model, Aex, Aey)
            model.train()
        else:
            bpc_C = claim.eval_bpc(model.lm, Cex, Cey)
            bpc_A = claim.eval_bpc(model.lm, Aex, Aey)
            model.lm.train()
        store.append({"after_C_segments": done, "bpc_C": bpc_C, "bpc_A": bpc_A,
                      **({"routes_C_eval": routes} if routed else {})})
    return ckpt


def run_cell_return(arm, seed, vocab, data, lr):
    """P3 cell. arm: 'return_vigilant' (natural vigilance on C) or
    'return_forced' (C forced to a fresh slot) or 'plain3' (no tissue)."""
    t0 = time.time()
    Ax, Ay, Bx, By, Cx, Cy, Aex, Aey, Bex, Bey, Cex, Cey = data
    rec = {"config": arm, "seed": seed, "lr": lr}
    n_C = int(Cx.shape[0])
    marks = [int(n_C * f) for f in CKPT_FRACS]
    if arm == "plain3":
        model = build_model(len(vocab), seed, 0)
        claim.train_stream(model.lm, Ax, Ay, lr, seed)
        rec["bpc_A_pre"] = claim.eval_bpc(model.lm, Aex, Aey)
        claim.train_stream(model.lm, Bx, By, lr, seed)
        rec["bpc_A_postB"] = claim.eval_bpc(model.lm, Aex, Aey)
        rec["bpc_B_postB"] = claim.eval_bpc(model.lm, Bex, Bey)
        rec["bpc_C_preC"] = claim.eval_bpc(model.lm, Cex, Cey)
        ckpts = []
        train_stream_plain_C(model.lm, Cx, Cy, lr, seed,
                             ckpt_fn=_make_ckpt_fn(model, Cex, Cey, Aex, Aey, 0,
                                                   ckpts, routed=False),
                             ckpt_marks=marks)
        rec["ckpts"] = ckpts
        rec["bpc_A_final"] = claim.eval_bpc(model.lm, Aex, Aey)
        rec["bpc_B_final"] = claim.eval_bpc(model.lm, Bex, Bey)
        rec["bpc_C_final"] = claim.eval_bpc(model.lm, Cex, Cey)
        rec["routing"] = "none (plain mixer)"
    else:
        force_slot = 2 if arm == "return_forced" else None
        model = build_model(len(vocab), seed, 4)
        E = 4
        ledger = _fresh_ledger()
        tr = {"A": [0] * E, "B": [0] * E, "C": [0] * E}
        train_stream_fusion(model, Ax, Ay, lr, "A", ledger)
        tr["A"] = model.n_segments[:]
        rec["bpc_A_pre"] = eval_bpc_fusion(model, Aex, Aey)
        train_stream_fusion(model, Bx, By, lr, "B", ledger)
        tr["B"] = [model.n_segments[s] - tr["A"][s] for s in range(E)]
        rec["bpc_A_postB"] = eval_bpc_fusion(model, Aex, Aey)
        rec["bpc_B_postB"] = eval_bpc_fusion(model, Bex, Bey)
        rec["bpc_C_preC"] = eval_bpc_fusion(model, Cex, Cey)
        rec["c_batch0_surprise"] = _c_batch0_surprise(model, Cx, Cy)
        ckpts = []
        train_stream_C(model, Cx, Cy, lr, ledger, force_slot=force_slot,
                       ckpt_fn=_make_ckpt_fn(model, Cex, Cey, Aex, Aey, E,
                                             ckpts, routed=True),
                       ckpt_marks=marks)
        tr["C"] = [model.n_segments[s] - tr["A"][s] - tr["B"][s] for s in range(E)]
        rec["ckpts"] = ckpts
        ev_A, ev_B, ev_C = [0] * E, [0] * E, [0] * E
        rec["bpc_A_final"] = eval_bpc_fusion(model, Aex, Aey, routes=ev_A)
        rec["bpc_B_final"] = eval_bpc_fusion(model, Bex, Bey, routes=ev_B)
        rec["bpc_C_final"] = eval_bpc_fusion(model, Cex, Cey, routes=ev_C)
        expected = {"A": int(Ax.shape[0]), "B": int(Bx.shape[0]), "C": n_C}
        rec["ledger"] = ledger_snapshot3(model, ledger, tr,
                                        {"A_eval": ev_A, "B_eval": ev_B, "C_eval": ev_C},
                                        expected)
        rec["c_outcome"] = _c_outcome(rec)
    rec["fgt_A_full"] = rec["bpc_A_pre"] - rec["bpc_A_final"]
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def _c_outcome(rec):
    """Classify the C-boundary behavior from the ledger (vigilant arm): 'reroute'
    (C segments trained by the PRE-C text8 expert), 'recruit' (a fresh slot owns
    C), or 'mixed'."""
    led = rec.get("ledger", {})
    pre_C = [p["slot"] for p in led.get("per_expert", [])
             if p.get("committed") and (p.get("train_A", 0) + p.get("train_B", 0)) > 0]
    recruits_C = [r for r in led.get("recruits", []) if r.get("corpus") == "C"]
    c_tr = {p["slot"]: p.get("train_C", 0) for p in led.get("per_expert", [])
            if p.get("committed")}
    c_on_pre = sum(c_tr.get(s, 0) for s in pre_C)
    c_total = sum(c_tr.values())
    if c_total == 0:
        return {"class": "none", "c_on_preC_expert": 0, "c_total": 0,
                "recruits_C": len(recruits_C)}
    frac_pre = c_on_pre / c_total
    cls = "reroute" if (frac_pre > 0.5 and not recruits_C) else (
        "recruit" if recruits_C and frac_pre < 0.5 else "mixed")
    return {"class": cls, "c_on_preC_expert": c_on_pre, "c_total": c_total,
            "frac_preC": round(frac_pre, 4), "recruits_C": [r["slot"] for r in recruits_C]}


def _load_return_data():
    A_all, B_all = claim.fetch_corpora()
    A_train = A_all[:claim.A_CHARS]
    A_eval = A_all[claim.A_CHARS:claim.A_CHARS + claim.A_EVAL_CHARS]
    B_train = B_all[: int(len(B_all) * 0.9)]
    B_eval = B_all[int(len(B_all) * 0.9):]
    C_raw = A_all[C_OFFSET: C_OFFSET + C_TRAIN_CHARS + C_EVAL_CHARS]
    C_train, C_eval = C_raw[:C_TRAIN_CHARS], C_raw[C_TRAIN_CHARS:]
    chars = sorted(set(A_train) | set(A_eval) | set(B_all) | set(C_raw))
    vocab = {c: i for i, c in enumerate(chars)}
    Ax, Ay = claim.make_segments(A_train, vocab)
    Bx, By = claim.make_segments(B_train, vocab)
    Cx, Cy = claim.make_segments(C_train, vocab)
    Aex, Aey = claim.make_segments(A_eval, vocab)
    Bex, Bey = claim.make_segments(B_eval, vocab)
    Cex, Cey = claim.make_segments(C_eval, vocab)
    return vocab, (Ax, Ay, Bx, By, Cx, Cy, Aex, Aey, Bex, Bey, Cex, Cey)


def run_return():
    torch.set_num_threads(int(os.environ.get("BENCH_THREADS", "8")))
    os.makedirs(OUT2, exist_ok=True)
    vocab, data = _load_return_data()
    if SMOKE:
        Ax, Ay, Bx, By, Cx, Cy, Aex, Aey, Bex, Bey, Cex, Cey = data
        data = (Ax[:64], Ay[:64], Bx[:64], By[:64], Cx[:64], Cy[:64],
                Aex, Aey, Bex, Bey, Cex, Cey)
    doc = {"meta": {
        "probe": "P3 RETURNING-DOMAIN (LANE-EXPLORATORY — design-informing, never a claim)",
        "question": "when the stream RETURNS to a seen domain (fresh text8 after "
                    "shakespeare), does the router RE-ROUTE C segments to the EXISTING "
                    "text8 expert (pattern completion — the lifelong win) or RECRUIT a "
                    "fresh one (treats it as novel)? And at what bpc cost either way?",
        "stream": "text8[0:1M) -> tiny-shakespeare -> text8[1.1M:2.0M), one pass; "
                  "C_eval = text8[2.0M:2.1M)",
        "C_disclosure": "C starts at 1.1M, NOT 1.0M: text8[1.0M:1.1M) is A_eval in the "
                        "PR-07'/fusion protocol — training on it would corrupt the "
                        "A-retention probe. C is therefore fully text-disjoint from "
                        "A_train AND A_eval, same domain (Wikipedia text8).",
        "arms": {"return_vigilant": "fusion E=4, natural per-segment vigilance on C "
                                    "(verbatim route_batch semantics)",
                 "return_forced": "fusion E=4, C-batch-0 force-recruits slot 2 which then "
                                  "owns ALL C segments exclusively (the treat-as-novel "
                                  "counterfactual; routing-only override — the backbone "
                                  "objective is identical across arms)",
                 "plain3": "plain mixer, no tissue (reference)"},
        "checkpoints": f"C-eval + A-eval bpc at {CKPT_FRACS} of C_train + final — the "
                       "re-mastery curve",
        "smoke": SMOKE, "seeds": [0] if SMOKE else SEEDS,
        "vocab": len(vocab), "threads": torch.get_num_threads(),
        "segments": {"A_train": int(data[0].shape[0]), "B_train": int(data[2].shape[0]),
                     "C_train": int(data[4].shape[0]), "A_eval": int(data[6].shape[0]),
                     "B_eval": int(data[8].shape[0]), "C_eval": int(data[10].shape[0])},
    }, "lr_selection": {}, "cells": []}
    save2(doc, "return")
    lrs = {}
    if SMOKE:
        lrs = {"return_vigilant": 3e-3, "return_forced": 3e-3, "plain3": 3e-3}
    else:
        lr, grid = select_lr("fusion_e4", 4, len(vocab), data[0], data[1], 0)
        lrs["return_vigilant"] = lrs["return_forced"] = lr
        doc["lr_selection"]["fusion_e4_family"] = {"lr": lr, "grid": grid}
        lr, grid = select_lr("plain", 0, len(vocab), data[0], data[1], 0)
        lrs["plain3"] = lr
        doc["lr_selection"]["plain"] = {"lr": lr, "grid": grid}
        save2(doc, "return")
    wall_first = None
    for arm in ["return_vigilant", "return_forced", "plain3"]:
        for seed in ([0] if SMOKE else SEEDS):
            rec = run_cell_return(arm, seed, vocab, data, lrs[arm])
            if wall_first is None:
                wall_first = (f"first cell wall {rec['wall_s']}s — budget extrapolates to "
                              f"{round(rec['wall_s'] * 6 / 60, 1)} min for this probe")
                rec["first_cell_note"] = wall_first
            doc["cells"].append(rec)
            save2(doc, "return")
            c_out = rec.get("c_outcome", {})
            print(f"[p3-return] {arm} seed {seed}: A_pre={rec['bpc_A_pre']:.3f} "
                  f"A_final={rec['bpc_A_final']:.3f} B_final={rec['bpc_B_final']:.3f} "
                  f"C_final={rec['bpc_C_final']:.3f} C-class={c_out.get('class')} "
                  f"wall={rec['wall_s']}s", flush=True)
    doc["summary"] = {}
    for arm in ["return_vigilant", "return_forced", "plain3"]:
        cells = [c for c in doc["cells"] if c["config"] == arm]
        doc["summary"][arm] = {
            "n_seeds": len(cells),
            "bpc_A_pre": round(float(np.mean([c["bpc_A_pre"] for c in cells])), 4),
            "bpc_A_postB": round(float(np.mean([c["bpc_A_postB"] for c in cells])), 4),
            "bpc_A_final": round(float(np.mean([c["bpc_A_final"] for c in cells])), 4),
            "bpc_B_postB": round(float(np.mean([c["bpc_B_postB"] for c in cells])), 4),
            "bpc_B_final": round(float(np.mean([c["bpc_B_final"] for c in cells])), 4),
            "bpc_C_preC": round(float(np.mean([c["bpc_C_preC"] for c in cells])), 4),
            "bpc_C_final": round(float(np.mean([c["bpc_C_final"] for c in cells])), 4),
            "c_classes": [c.get("c_outcome", {}).get("class") for c in cells],
            "wall_s": round(float(np.sum([c["wall_s"] for c in cells])), 1),
        }
    save2(doc, "return")
    print(f"[p3-return] summary: {json.dumps(doc['summary'])}", flush=True)


def _load_2block_data():
    """Identical construction to the original main() — bit-compatible with the
    2026-09-07 run (same vocab, same segment tensors)."""
    A_all, B_all = claim.fetch_corpora()
    A_train = A_all[:claim.A_CHARS]
    A_eval = A_all[claim.A_CHARS:claim.A_CHARS + claim.A_EVAL_CHARS]
    B_train = B_all[: int(len(B_all) * 0.9)]
    B_eval = B_all[int(len(B_all) * 0.9):]
    chars = sorted(set(A_all[:claim.A_CHARS + claim.A_EVAL_CHARS]) | set(B_all))
    vocab = {c: i for i, c in enumerate(chars)}
    Ax, Ay = claim.make_segments(A_train, vocab)
    Bx, By = claim.make_segments(B_train, vocab)
    Aex, Aey = claim.make_segments(A_eval, vocab)
    Bex, Bey = claim.make_segments(B_eval, vocab)
    return vocab, (Ax, Ay, Bx, By, Aex, Aey, Bex, Bey)


def main2(mode):
    os.makedirs(OUT2, exist_ok=True)
    if mode == "frozen":
        run_frozen()
    elif mode == "shared":
        run_shared()
    elif mode == "return":
        run_return()
    elif mode == "all2":
        run_frozen()
        run_shared()
        run_return()
    else:
        raise SystemExit(f"unknown mode {mode!r}; expected frozen|shared|return|all2")
    print(f"[probe2] wrote JSONs to {OUT2}", flush=True)


if __name__ == "__main__":
    _MODE = sys.argv[1] if len(sys.argv) > 1 else None
    if _MODE in {"frozen", "shared", "return", "all2"}:
        main2(_MODE)
    else:
        main()
