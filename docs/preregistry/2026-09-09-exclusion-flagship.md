# PR-2026-09-03-13 — the EXCLUSION flagship (S2c: PR-08 protocol + both registered repair levers)

**Lane:** CLAIM · **Status:** REGISTERED 2026-09-09 20:3x (frozen before any run) · **Written:** 2026-09-09
**Provenance:** PR-08 (B1/B2 PASS, B3 ledger FAIL) → PR-10 **CLAIMED** (trunk-lr ×0.25 on C
repairs boundary routing: frac 0.575, guards intact) → PR-11 (clause (a) transferred to the
flagship at 0.575; clause (b) FAIL forced_cost −0.1995) → PR-12 **CLAIMED** (domain-exclusion
eliminates the B-expert contamination: PRIM-DE B 3.4124 == FORCED 3.4124, frac 0.976, guards
intact; disclosed side effect: in seed 2 the eviction-exemption collapsed the candidate pool
to the a_expert slot, recycled 22×). **The two registered repair levers together are
hypothesized to clear ALL THREE gated flagship bars at n=5.**

## 1. Claim under test

The PR-08 flagship protocol executed VERBATIM except for TWO registered levers — (L1) the
PR-10/PR-11 backbone-lr ×0.25 on block C, and (L2) the PR-12 domain-exclusion with a NEW
eviction refinement — clears B1 (retention-A ≤ 0.05), B2 (adaptation-C ≥ 0.10) and B3
(pattern completion: frac ≥ 0.5 AND forced-cost ≥ −0.10) at n=5, reinstating the
'lifelong routing' claim WITH evidence.

## 2. Protocol (PR-08 verbatim + the two levers + one refinement)

Stream, slices, corpora, tissue constants, routing semantics (veto + Policy A + forced
placement contingency), seeds 0–4, LR 3e-3 A/B, arm families, bars §4: EXACTLY
docs/preregistry/2026-09-08-prizma-lm-flagship.md as amended (addenda #1–#3).

- **L1 (PR-10/PR-11, carried):** backbone lr 7.5e-4 on block C in every arm that trains a
  backbone on C (PRIM-LM, FORCED-RECRUIT, SHARED-HEAD).
- **L2 (PR-12, carried, PRIM-LM only):** at C start, `domain_protect` = committed slots
  MINUS the a_expert; protected slots receive no C training (segments redirect to the
  a_expert, post-redirect routing counted) and are exempt from Policy A eviction during C.
  FORCED-RECRUIT is structurally exclusive (no machinery needed); SHARED-HEAD has no
  routing; **FROZEN-TRUNK and FROZEN-CHECKPOINT are UNTOUCHED by L2** (their cells remain
  byte-comparable canaries; FROZEN-TRUNK's tissue keeps PR-08 argmin routing — disclosed,
  its B4 role is descriptive).
- **L2 refinement (NEW, this registration, from PR-12's disclosed side effect):** during C,
  if the post-exclusion eviction candidate pool is empty OR contains only the a_expert, the
  recruit is SUPPRESSED — the novelty segments route by argmin and the redirect folding
  sends them to the a_expert; the A-expert slot identity is thereby preserved (no
  self-eviction recycling). The suppression event is counted in the ledger
  (`domain_protect_suppressed_recruits`). Off-identity: with the lever attributes absent
  every touched path is byte-identical to PR-08/PR-11.

## 3. Arms (5, as PR-08 §3) + canaries

PRIM-LM (L1+L2+refinement) / FROZEN-TRUNK (untouched) / SHARED-HEAD (L1) /
FROZEN-CHECKPOINT (untouched) / FORCED-RECRUIT (L1; forced semantics already exclusive).
**Fail-loud canaries (powered mode, after each canary arm completes):** FROZEN-TRUNK's 5
cells and FROZEN-CHECKPOINT's 5 cells must each be BIT-IDENTICAL to PR-08's
claim.FROZEN-TRUNK / claim.FROZEN-CHECKPOINT cells (mismatch or missing PR-08 ledger =
ABORT before further arms). In smoke: the PRIM A/B phases bit-identical to PR-08's PRIM
records' bpc_A_preB / bpc_B_postB, the lever audits present, and the levers fire
(bpc_B_postC differs from PR-08's PRIM cell).

## 4. Bars (PR-08 §4 verbatim; n=5 seeds 0–4; Welch; t_isf UPPER-TAIL)

- **B1 Retention-A:** BPC(A-eval, post-C) − BPC(A-eval, pre-B) ≤ 0.05 (one-sample CI
  upper ≤ 0.05) for PRIM-LM.
- **B2 Adaptation-C:** BPC_PRIM(C-ret, post-C) ≤ BPC_FROZEN-CHECKPOINT(same) − 0.10
  (Welch; Holm over B1–B2).
- **B3 Pattern completion (ledger primary):** frac of C segments (first 20 C-batches,
  post-redirect trained fraction) to the A-recruited expert ≥ 0.5, AND
  forced-recruit ablation's BPC(B-eval, post-C) ≥ PRIM-LM's − 0.10.
- **B4 Trunk-drift accounting:** reported per arm, never gated.
- Straddling CI ⇒ INCONCLUSIVE, never PASS.

## 5. Pre-committed branches

- B1 + B2 + B3 ALL PASS ⇒ **CLAIMED** — the repaired flagship stands; the 'lifelong
  routing' wording is REINSTATED WITH EVIDENCE (superseding the PR-08 B3 retirement and
  the PR-11 confirmation of it, both kept on record); S2 complete; the ladder advances to
  S3 (GPU scale tier) with PR-LM-1 inheriting L1 + L2 + the refinement.
- B3 clause (a) PASS + clause (b) FAIL ⇒ **NEGATIVE** — the refinement did not preserve
  pattern-completion value; eval-routing becomes the lead (own prereg).
- B3 clause (a) FAIL ⇒ **NEGATIVE** — exclusion + refinement broke the boundary routing;
  investigate before any new registration.
- B1 or B2 FAIL ⇒ **NEGATIVE-GUARD** (record which); any bar INCONCLUSIVE ⇒ n=10
  (seeds 5–9) PRE-AUTHORIZED (03:55 rule wins).

## 6. Budget & runner

25 cells ≈ 30–35 min CPU. Runner: prizma_lm_claim.py's guarded PR-11 mode extended with
`--domain-exclusion` (sets the L2 wiring + the refinement in run_routed's C phase);
ledger dir `prizma_lm_PR-2026-09-03-13`; fingerprints include both lever states; smoke/
powered/--powered-cpu separation; archive-before-verdict. Smoke on CPU first.

## 7. Self-audit

No new tuning surface: L1's dose is PR-10's frozen 0.25; L2 is PR-12's registered
exclusion; the refinement is a deterministic fail-safe (suppress rather than self-evict),
counted in the ledger. The canary arms prove the levers' blast radius mechanically. Bars
are PR-08's own frozen text. What would change our mind: only dated addenda. The one bet:
that PR-10's and PR-12's independently CLAIMED repairs compose — that fixing the trunk
schedule AND the expert-contamination jointly leaves no third defect at this scale.
