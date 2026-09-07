# PR-2026-09-03-08 — Prizma-LM flagship bar (many-block continual char-LM, fused column)

**Lane:** CLAIM · **Status:** REGISTERED 2026-09-08 (frozen before any run; execution awaits
the owner's GPU session — the notebook gains a stage, see §7) · **Written:** 2026-09-08
**Provenance:** fusion design spec §4 (as amended by Addenda 1–3) + the exploratory chain it
was built on: PR-07′ (block-drift PASS), fusion_probe (2026-09-07: specialization purity
0.997–1.0 in the block regime), fusion_probe2 (2026-09-08: capacity confound REFUTED —
shared head *worse* at equal budget; pattern completion fires at a returning-domain
boundary 2/2 seeds; TRUNK DRIFT is the dominant many-block forgetting term).

## 1. Claim under test

A fused column — Prizma-Seq predictive-coding delta-state backbone + vigilance-routed,
locally-optimized expert tissue — learns a MANY-block drifting char stream (≥3 corpora,
2 drift boundaries) with (i) near-zero forgetting of earlier blocks, (ii) full adaptation
to each new block, (iii) ROUTING (not just accuracy) demonstrably carrying the lifelong
behavior: boundary-recruit timing, returning-domain pattern completion, and per-block
expert ledgers. Frozen checkpoints and capacity-matched shared heads cannot follow.

## 2. Streams (exact)

Three blocks, one pass, concatenated, no labels/boundaries to the model:
- **Block A:** text8 chars [0, 1,000,000) — *retention target 1*
- **Block B:** tiny-shakespeare full — *drift (strong)*
- **Block C:** text8 chars [1,000,000, 2,000,000) — *drift (mild) + RETURNING domain*
  (disjoint char range from A, same generator/domain)
- Held-out evals: A-eval text8 [1.9M, 2.0M) is NOT used (in-block to C); A-eval stays
  text8 [1.0M, 1.1M) as in PR-07′; B-eval = shakespeare last 10%; C-eval = shakespeare is
  not re-evaluated for C; C retention = text8 [0.9M, 1.0M) held-out slice (in-domain,
  disjoint from A-train's tail) — PINNED exactly in the runner.
- Corpora already archived (sha256 committed, PR-07′). Download failure = ABORT (same rule).

## 3. Arms (exact)

1. **PRIM-LM (primary)** — backbone-train fusion: PrizmaSeqLM (2L, d=64, H=4 — the pinned
   PR-07′ config) whose LM head is augmented by the E=4 vigilance-expert tissue with
   segment-level routing, per-expert local optimizers on detached activations, bounded pool
   M_max=4, recruit-by-eviction (PR-05 Policy A), floor-maturity veto (freeze_min_seen=300;
   the probe-2 cascade lesson), trunk trained on all segments (shared LM loss).
2. **FROZEN-TRUNK ablation** — backbone frozen after A; only the tissue learns. The
   retention-corner (probe-2: exact retention, ~74% adaptation closure).
3. **SHARED-HEAD control** — same extra parameter budget as PRIM-LM, one head, no routing
   (probe-2: this arm LOSES to fusion at equal budget — the capacity control).
4. **FROZEN-CHECKPOINT control** — as PR-07′ (adaptation floor).
5. **FORCED-RECRUIT ablation** — PRIM-LM with routing replaced at C by forced-fresh recruit
   (kills pattern completion; isolates its value — the probe-2 forced arm costs ~nothing on
   C bpc, so the claim stakes on LEDGERS and RETENTION, not C accuracy).

## 4. Bars (exact; n=5 seeds 0–4; Welch; t_isf UPPER-TAIL convention)

- **B1 Retention-A (after C):** BPC(A-eval, post-C) − BPC(A-eval, pre-B) ≤ 0.05 (one-sample
  CI upper ≤ 0.05) for PRIM-LM.
- **B2 Adaptation-C:** BPC_PRIM(C-eval := text8 [0.9M,1.0M) slice, post-C) ≤
  BPC_FROZEN-C(same) − 0.10 (two-sample Welch; Holm over B1–B2).
- **B3 Pattern completion (ledger primary):** fraction of C segments routed to the
  A-recruited expert within the first 20 C-batches ≥ 0.5 (2/2 probe seeds already show
  zero-recruit at the boundary), AND forced-recruit ablation's BPC(B-eval, post-C) ≥
  PRIM-LM's BPC(B-eval, post-C) − 0.10 (pattern completion must not cost B — probe-2
  measured ≈0.011 cost).
- **B4 Trunk-drift accounting (the probe-2 lesson, reported not gated):** B-eval degradation
  post-C per arm — the registration's purpose is to MEASURE whether the tissue bounds trunk
  drift, not to assume it.
- WINDOW-TF / FROZEN-CHECKPOINT descriptive per PR-07′ conventions. INCONCLUSIVE rules
  mirror PR-03 (straddling CI ⇒ INCONCLUSIVE, never PASS).

## 5. Pre-committed failure handling

- B1 FAIL ⇒ tissue does not protect retention at many-block scale ⇒ PR-LM-1 wording
  downgrades to single-boundary (PR-07′ scope) and the trunk-drift measurement becomes the
  headline negative; frozen-trunk ablation (B2 arm) is the designed rescue path, registrable
  separately.
- B3 ledger FAIL ⇒ "lifelong routing" wording is retired regardless of accuracy (accuracy
  without routing is not the claim — probe-2's forced-arm cost ≈0 makes this a real risk).
- Any bar INCONCLUSIVE ⇒ maintainer addendum decides seed extension (n→10) BEFORE unblinding
  per-arm numbers, or the leg is honestly abandoned.

## 6. Budget & runner

Estimate from PR-07′ (~42 s/stream-seed) × 2.1→3.1M chars × 5 arms × 5 seeds ≈ **45–90 min
A100**, CPU-feasible fallback ≈ 6–8 h (acceptable — PR-07′ ran 9 min CPU vs 15× estimate).
Runner: NEW seq/prizma_lm_claim.py mirroring surprise_claim.py (smoke/powered separation,
CUDA refusal on --powered, fingerprint resume, archive-before-verdict, VERBATIM bars above,
t_isf upper-tail). To be implemented (CPU) before the GPU session; smoke on CPU first.

## 7. Self-audit

Every knob pinned (config repr stored per record); corpora pinned by archived sha256;
arms exclusive; bars exact; statistics fully specified with the upper-tail convention;
exploratory provenance chain (PR-07′ → fusion probes) committed before this freeze; what
would change our mind: only dated addenda. The registration's one bet: that probe-scale
routing behavior (purity, boundary-recruit, returning-domain completion) survives the
+many-block step — B3 is designed to catch it if it doesn't.

---

## Addendum 2026-09-08 (pre-run, maintainer) — C-block slice corrected; two disclosures

1. **C-block range corrected to text8 [1,100,000, 2,100,000).** The literal §2 range
   [1.0M, 2.0M) trains on the A-eval slice [1.0M, 1.1M), which would corrupt bar B1
   (retention evaluated on trained chars). Caught by the runner implementer BEFORE any
   run; the corrected range preserves the returning-domain design (disjoint from A-train
   [0,1.0M) and from A-eval, same domain as A). This is a maintainer correction of a spec
   bug, not an experimental change.
2. **Disclosed:** the C-retention slice [0.9M, 1.0M) overlaps A-train's tail (the doc
   called it "disjoint" — it is not). Implemented as pinned; B2 remains fair because both
   compared arms saw it in A. 
3. **Disclosed:** freeze_min_seen is ported as the recruit-side veto (probe-2 cascade
   lesson); src/prizma.py's consolidation freeze-gate has no tissue analog in this runner.
