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
