# PR-2026-09-03-11 — the REPAIRED flagship bar (PR-08 protocol + the registered C-block trunk-lr lever)

**Lane:** CLAIM · **Status:** REGISTERED 2026-09-09 19:0x (frozen before any run) · **Written:** 2026-09-09
**Provenance:** PR-08 (B1/B2 PASS, B4 tissue-bounds-drift measured, B3 ledger FAIL frac
0.428 → 'lifelong routing' wording retired per pre-commitment) → PR-09 (floors exonerated:
pinning left routing near-identical) → PR-10 **CLAIMED** (backbone lr ×0.25 on block C only:
frac mean 0.575 ≥ 0.5 with G1 retention −0.282 and G2 adaptation +0.308 intact, Holm
p 0.0205; B4 trunk drift +1.202 → +0.721, best of any arm). **Hypothesis carried into this
registration:** with the registered lever wired into the flagship's C phases, the B3 ledger
clears its ≥0.5 bar at n=5 while B1/B2 hold — reinstating the lifelong-routing claim WITH
evidence.

## 1. Claim under test

The PR-08 flagship protocol, executed VERBATIM except for the single registered lever
(backbone lr ×0.25 on block C in every arm that trains a backbone on C), now clears ALL
THREE gated bars: B1 retention-A ≤ 0.05, B2 adaptation-C ≥ 0.10, B3 ledger-primary pattern
completion (frac ≥ 0.5 within the first 20 C-batches AND forced-recruit cost ≤ 0.10), with
B4 trunk-drift accounting reported.

## 2. Protocol (PR-08 verbatim + the one lever)

Stream, slices, corpora (archived sha256), tissue constants, routing semantics
(floor-maturity veto + Policy A eviction + the PR-08 addendum #3 forced-placement
contingency), seeds 0–4, LR 3e-3 for blocks A/B, arm families, and the frozen bars §4 of
docs/preregistry/2026-09-08-prizma-lm-flagship.md (as amended by addenda #1–#3) are
inherited VERBATIM. THE ONE CHANGE (evidence-registered by PR-10): in the block-C phase,
the BACKBONE optimizer's lr is 7.5e-4 (3e-3 × 0.25, the PR-10 frozen dose) in
- **PRIM-LM** and **FORCED-RECRUIT** (both train a backbone on C via train_pr08),
- **SHARED-HEAD** (its shared head trains on C via train_stream_shared — the same guarded
  kwarg pattern; the capacity control must see the same schedule to stay honest).
FROZEN-TRUNK (backbone frozen after A) and FROZEN-CHECKPOINT (no B/C backbone training)
have no C backbone step — the lever is structurally inapplicable there (recorded as such).
Tissue per-expert optimizers keep 3e-3 everywhere; floors stay LIVE.

## 3. Arms (5, as PR-08 §3) + canaries

PRIM-LM / FROZEN-TRUNK / SHARED-HEAD / FROZEN-CHECKPOINT / FORCED-RECRUIT — identical to
PR-08 including the forced-placement contingency. Fail-loud canaries (powered mode):
(1) the OFF-flagship reproduction is NOT re-run — instead, FROZEN-CHECKPOINT and
FROZEN-TRUNK cells carry the canary load: FROZEN-CHECKPOINT cells (no C backbone training,
lever-inapplicable) must be BIT-IDENTICAL to PR-08's claim.FROZEN-CHECKPOINT cells;
FROZEN-TRUNK cells (backbone frozen on C; tissue trains identically; no backbone step on C)
must likewise be BIT-IDENTICAL to PR-08's claim.FROZEN-TRUNK cells. Mismatch or a missing
PR-08 ledger = ABORT before any treated arm (this proves, end-to-end, that the lever
touches ONLY the C backbone step). (2) In smoke: the PRIM A/B phases are bit-identical to
PR-08's PRIM records' bpc_A_preB / bpc_B_postB, and the lever fires (bpc_B_postC differs
from PR-08's PRIM cell).

## 4. Bars (PR-08 §4 verbatim; n=5 seeds 0–4; Welch; t_isf UPPER-TAIL)

- **B1 Retention-A:** BPC(A-eval, post-C) − BPC(A-eval, pre-B) ≤ 0.05 (one-sample CI
  upper ≤ 0.05) for PRIM-LM.
- **B2 Adaptation-C:** BPC_PRIM(C-ret, post-C) ≤ BPC_FROZEN-CHECKPOINT(same) − 0.10 (Welch;
  Holm over B1–B2).
- **B3 Pattern completion (ledger primary):** frac of C segments (first 20 C-batches)
  routed to the A-recruited expert ≥ 0.5 for PRIM-LM, AND forced-recruit ablation's
  BPC(B-eval, post-C) ≥ PRIM-LM's − 0.10.
- **B4 Trunk-drift accounting:** reported per arm, never gated.
- WINDOW-TF/FROZEN-CHECKPOINT descriptive per PR-07′ conventions; straddling CI ⇒
  INCONCLUSIVE, never PASS.

## 5. Pre-committed branches

- ALL THREE bars PASS ⇒ **CLAIMED** — the repaired flagship stands; the 'lifelong routing'
  wording is REINSTATED with evidence (superseding the PR-08 B3 retirement, which stays on
  the record); S2 complete, the ladder advances to S3 (GPU scale tier) with PR-LM-1
  designs inheriting the C-block trunk-lr schedule.
- B3 PASS + B1 or B2 FAIL ⇒ **NEGATIVE-GUARD** — routing at the cost of the column;
  dose ladder (0.5/0.1) as a NEW prereg with the tradeoff as its object.
- B3 FAIL ⇒ **NEGATIVE** — the 0.25 dose does not transfer from the 2-arm probe to the
  5-arm flagship; next candidates: dose ladder, or recruiting-block scheduling (B-block lr),
  each its own prereg.
- Any bar INCONCLUSIVE ⇒ seed extension to n=10 (seeds 5–9) PRE-AUTHORIZED (03:55 rule
  always wins).

## 6. Budget & runner

25 cells ≈ 30–35 min on the local CPU box. Runner: prizma_lm_claim.py gains a guarded
`trunk_lr_c=None` mode (CLI flag --trunk-lr-c; default None = byte-identical PR-08 paths)
writing to its OWN ledger dir `prizma_lm_PR-2026-09-03-11` (the PR-08 ledger is never
touched); per-cell fingerprints include the lever; smoke/powered/--powered-cpu separation,
archive-before-verdict, VERBATIM bars. Smoke on CPU first.

## 7. Self-audit

One lever, one dose, inherited from a CLAIMED registration (PR-10); the control arms that
can carry bit-identity canaries (FROZEN-CHECKPOINT, FROZEN-TRUNK) make the lever's blast
radius mechanically provable; bars are PR-08's own frozen text; what would change our
mind: only dated addenda. The registration's one bet: that PR-10's 2-arm repair survives
the full 5-arm flagship context (its SHARED-HEAD control now also carries the schedule, so
the capacity control stays matched).
