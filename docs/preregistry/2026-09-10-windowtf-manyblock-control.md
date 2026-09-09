# PR-2026-09-03-15 — the memory-matched WINDOW-TF control on the 5-block curriculum

**Lane:** CLAIM · **Status:** REGISTERED 2026-09-10 00:1x (frozen before any run) · **Written:** 2026-09-10
**Provenance:** PR-13 CLAIM PASS (the 3-block repaired flagship, all bars) → PR-14 CLAIMED
(5-block accumulation with a true revisit: recovery +0.82, A-retention −0.32, re-engagement
0.971). The repaired column's lifelong behavior is now measured — but always against its
OWN arms. This registration runs the standing **memory-matched sliding-window transformer
control** (the PR-07′ WINDOW-TF config, carried VERBATIM: 2L, d=64, H=4, window = SEG) on
the IDENTICAL 5-block curriculum, scored against PR-14's EX cells (reused, disclosed: same
seeds, same stream, same eval slices).

## 1. Claim under test

On the same 5-block curriculum, the sliding-window transformer — with no persistent-state
machinery beyond its weights — CANNOT hold domain A across the stream: its A-retention
(post_E − post_A) is worse than the repaired column's by more than 0.10 bpc (Welch,
CI lower ≥ 0.10). This is the first head-to-head retention comparison of the ladder: the
control degrades because nothing protects it; the repaired column holds because the
tissue + trunk-lr schedule do.

## 2. Protocol

Stream, slices, corpora, seeds 0–4, LR rules: EXACTLY docs/preregistry/2026-09-09-
manyblock-accumulation.md (PR-14). WINDOW-TF config: `seq/transformer.py` TFConfig(vocab,
d_model=64, n_layers=2, n_heads=4, max_len=SEG=256) + `train_stream`/`eval_bpc` — the
PR-07′ descriptive control path VERBATIM (sliding-window teacher forcing per segment
batch; no boundaries given). The TF trains on all five blocks sequentially with the SAME
per-block order; NO backbone-lr scheduling (the control has no tissue and no registered
schedule — its learning rule is the plain window LM's, disclosed: this is precisely the
comparison — the column's schedule is part of its mechanism).

## 3. Bars (exact; n=5 seeds 0–4; Welch; t_isf UPPER-TAIL)

- **P1 (the retention gap, primary):** delta = mean(bpc_A(post_E) − bpc_A(preB)) of
  WINDOW-TF minus the repaired column's (−0.3226, PR-14 EX cells REUSED from
  results/manyblock_PR-2026-09-03-14/powered.json, disclosed); Welch CI lower ≥ 0.10.
  Holm family = [P1] (single primary).
- **Descriptive, never gated:** the TF's full per-block trajectories (A/B/Cret after each
  block); its B seesaw (post_B → post_C → post_D → post_E); per-block loss curves.
- INCONCLUSIVE rule: straddling CI ⇒ INCONCLUSIVE, n=10 (seeds 5–9) PRE-AUTHORIZED.

## 4. Pre-committed branches

- P1 PASS ⇒ **CLAIMED** — the control cannot hold A; the retention gap is established on
  the many-block curriculum; further controls (longer-window TF, replaying TF) become new
  registrations if the ladder needs them.
- P1 FAIL (the TF holds A comparably or better) ⇒ **NEGATIVE** — the repaired column's
  retention advantage does not generalize to the 5-block curriculum; the comparison record
  stays honest and the next lever (what the TF has that the column lacks) becomes the lead.

## 5. Budget & runner

5 cells ≈ 15–30 min CPU. Runner: NEW seq/windowtf_manyblock.py mirroring
seq/manyblock_claim.py's stream plumbing (single arm; reused PR-14 EX baseline;
per-cell fingerprints incl. the TF config; smoke/powered/--powered-cpu;
archive-before-verdict). Smoke on CPU first.

## 6. Self-audit

The control's config is the repo's own pinned PR-07′ WINDOW-TF (not a strawman: it is the
same memory-matched convention the earlier reports used). The baseline reuse is the exact
cells the claim compares against, same seeds/protocol. One bar, one direction, no tuning
surface. What would change our mind: only dated addenda.
