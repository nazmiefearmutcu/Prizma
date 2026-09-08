# PR-2026-09-03-07 — Block-drift BAR-6′ (continual char-LM, sequential corpora)

**Lane:** CLAIM · **Status:** REGISTERED 2026-09-06 · **Written:** 2026-09-06 (frozen before
any run) · **Provenance:** fusion design spec Addendum 3 (BAR-6 gate re-scope,
docs/superpowers/specs/2026-09-03-prizma-lm-fusion-design.md) — the evidence-driven
replacement for the interleaved BAR-6 (PR-2026-09-03-06, NEGATIVE terminal).

## 1. Claim under test

Prizma-Seq's O(1) carried state provides CONTINUAL adaptation on a block-drift token stream:
after streaming corpus B, the model retains corpus A (little forgetting) AND has learned B
(adaptation) — with no task labels, no boundaries given, no replay buffer — where a frozen
checkpoint structurally cannot adapt and a state-reset ablation structurally cannot retain.

## 2. Streams (exact, no substitutions)

- **Corpus A (train + retention eval):** text8, chars [0, 1,000,000) for streaming; A-eval =
  chars [1,000,000, 1,100,000) (disjoint). Canonical source: https://mattmahoney.net/dc/text8.zip
- **Corpus B (drift + adaptation eval):** tiny-shakespeare (~1.1 MB). Canonical source:
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
- Both raw texts (and their sha256) are archived under
  `results/blockdrift_PR-2026-09-03-07/corpora/` BEFORE training (retention policy). If a
  canonical download fails, the run ABORTS — corpora are never silently substituted.
- Stream: A then B, concatenated, char-level, no boundary signal given to the model
  (the boundary exists only in the evaluation's accounting).

## 3. Arms (exact)

1. **STREAM** — Prizma-Seq (PrizmaSeqConfig: 2 layers, d_model=64, H=4, chunk=64, window=16;
   the config object is pinned in the runner and its repr is stored in every raw record)
   trained online over the full A+B stream, then evaluated on A-eval and B-eval.
2. **FROZEN control** — same model trained on A only, weights frozen after A, evaluated on
   B-eval (upper-bounds retention, lower-bounds adaptation).
3. **RESET ablation** — STREAM but the carried state S is zeroed at the A→B boundary
   (proves retention is state-carried; its A-eval drop is the causal contrast).
4. **WINDOW-TF control (descriptive, never a claim)** — memory-matched sliding-window
   Transformer (window byte-budget = Prizma's measured state+optimizer-free inference bytes;
   budget arithmetic disclosed in RESULTS). Reported, not gated (fusion spec §4's honest
   default: if it matches STREAM, the claim downgrades to "competitive at constant memory").

Hyper-parameters: one LR per arm chosen by the repo's standard first-seed plateau rule on a
3-point grid {1e-3, 3e-3, 1e-2} evaluated on seed 0's A-segment loss ONLY, then frozen for
all seeds/arms (per-arm differences disclosed). No per-corpus LR changes.

## 4. Bars (exact; n=5 seeds 0–4; one-sample/two-sample Welch t at α=0.05; t_isf takes upper-tail p)

- **Retention:** FGT_A = BPC_STREAM(A-eval, pre-B) − BPC_STREAM(A-eval, post-B) ≤ 0.05
  (mean over seeds; one-sample t CI upper bound ≤ 0.05).
- **Adaptation:** BPC_STREAM(B-eval) ≤ BPC_FROZEN(B-eval) − 0.10 (two-sample Welch, Holm
  over the two primaries).
- **Causal ablation:** FGT_A(RESET) − FGT_A(STREAM) ≥ 0.15 (retention is carried by the state).
- INCONCLUSIVE handling mirrors PR-03: straddling CI ⇒ INCONCLUSIVE (never PASS).

## 5. Pre-committed failure handling

- Retention bar FAIL ⇒ the "state provides block-drift continual retention" sentence is not
  made at any scale; PR-LM-1 waits on a state-design fix (multi-timescale decay / phase axis
  candidates from report 09) as a NEW pre-registration.
- Adaptation bar FAIL ⇒ investigate before any re-registration (learning-rate starvation on
  drift is the known suspect); no silent knob changes.
- WINDOW-TF matches STREAM ⇒ public wording is "competitive at constant memory", never
  "continual advantage" (fusion spec §4's honest default).

## 6. Budget & execution notes

4 arms × 5 seeds × ~3.1M chars at d=64 CPU: estimated ~2–3 h on the Ryzen 7 5750G
(BENCH_THREADS=8; measure the first cell and disclose the projection in RESULTS.md).
Runner: NEW seq/blockdrift_claim.py (reuses seq/prizma_seq.py + seq/charlm.py data
conventions; crash-safe per-seed raw JSON per docs/RETENTION.md; verdict references artifact
paths). Scope limitation carried from the analog lever: knobs are exact-step-path only;
this protocol uses default-off knobs, so no limitation bites.

## 7. Self-audit

- No placeholders; every knob pinned; corpus identity pinned by source URL + archived
  sha256; arms mutually exclusive; bars are exact numbers; statistics fully specified;
  t_isf upper-tail convention noted (the PR-03 incident's lesson).
- What would change our mind: only dated addenda (POLICY.md); the exploration that motivated
  this gate (E1 block-stream guards, 0.834/FGT 0.000 across all probes) used synthetic
  streams — real-corpus drift is exactly what this run tests.

## Addendum 2026-09-07 (pre-run clarification, BEFORE any results)

1. **RESET arm re-scoped to an implementation-integrity canary.** Segment training uses the
   chunk-parallel `forward` path, whose carried state spans one segment, not the whole
   stream; cross-stream memory at this scale is therefore WEIGHT-carried, and a
   state-zeroing-at-the-boundary arm is vacuously identical to STREAM. The arm still runs
   (seed 0 only) and must reproduce STREAM bit-identically; the §4 causal-ablation bar
   (reset ΔFGT ≥ 0.15) is WITHDRAWN as vacuous — it is replaced by the canary equality
   assertion (any deviation = broken runner, run invalid).
2. **LR rule interpretation.** "One LR per arm ... frozen for all seeds/arms" is read as:
   one LR per ARM FAMILY {stream-family (STREAM+RESET), frozen, window-tf}, each chosen on
   seed 0's A-segment loss from the 3-point grid, then frozen across that family's seeds.
   Chosen LRs are disclosed in every raw record.
3. **Partial order.** Primaries first (STREAM, FROZEN, all seeds), then RESET canary, then
   WINDOW-TF (descriptive; may be cut by the 03:55 clock rule with disclosure — it gates
   nothing).

## Addendum 2026-09-08 (post-run disclosure) — Holm correction documented but not implemented in the runner

The §4 statistics specified Holm over the two primaries; the runner computed the two
primaries with plain Welch comparisons and no multiplicity adjustment. Pre-registered
disclosure: with the observed margins (retention CI upper −0.119 vs bar ≤ 0.05; adaptation
diff −3.18, CI [−3.420, −2.936] vs bar ≤ −0.10) Holm adjustment is arithmetically incapable
of flipping either verdict (each primary clears its bar at p ≪ 0.025). The CLAIMED status
stands; the runner's verdict text was corrected to describe this honestly. Recorded by the
2026-09-08 pre-GPU review (committee/review_2026-09-08/PRE_GPU_REVIEW.md, H-2).

## Addendum 2026-09-08 (post-hoc, maintainer) — registered Holm implemented post-hoc; disclosure (review H-2)

1. **Disclosure.** The §4 statistics registered "Holm over the two primaries", but the
   shipped runner (seq/blockdrift_claim.py) implemented only the CI rules and never called
   Holm. The CI rules are themselves conservative (they are the same one-sided tests
   displayed as 95% CIs), but the registered statistic was not the computed statistic.
   Disclosed by the 2026-09-08 pre-GPU review (H-2); this addendum closes the finding
   post-hoc, from the raw records, with no re-run.
2. **Recomputation** (committee/review_2026-09-08/h2_holm_recompute.py, output
   h2_holm_recompute_OUTPUT.txt; loads results/blockdrift_PR-2026-09-03-07/raw.json and uses
   the seq.stats t_sf upper-tail convention; every intermediate value printed): Bar 1
   (FGT_A, one-sample, n=5, df=4, mean −0.189010, se 0.025374): p_raw = 3.540489e-04.
   Bar 2 (adaptation, Welch, delta = mean(FROZEN) − mean(STREAM) = +3.177780 vs required
   +0.10, se 0.097402, df 5.660): p_raw = 6.944012e-08. Holm(2) step-down at alpha = 0.05:
   sorted ascending [6.944012e-08, 3.540489e-04], multipliers 2 then 1 → Holm-adjusted p =
   1.388802e-07 (bar 2) and 3.540489e-04 (bar 1); both hypotheses rejected. Both RAW
   p-values are far below the binding first threshold alpha/2 = 0.025, so the
   registered-but-unimplemented Holm is a no-op that cannot flip the CLAIMED outcome.
3. **Artifacts.** Script + full console output: committee/review_2026-09-08/h2_holm_recompute.py
   and committee/review_2026-09-08/h2_holm_recompute_OUTPUT.txt (read-only recompute; the
   recomputed mean and CIs reproduce the ledger's stored fgt_mean / fgt_ci_hi / adapt_ci
   exactly, difference 0.00e+00).
4. **No side effects.** No re-run was performed; nothing under results/ was modified or
   created by this closure.
