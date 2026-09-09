# PR-2026-09-03-12 — domain-exclusive C-routing (the PR-11 B3-clause-(b) negative's registered successor)

**Lane:** CLAIM · **Status:** REGISTERED 2026-09-09 20:1x (frozen before any run) · **Written:** 2026-09-09
**Provenance:** PR-10 CLAIMED (trunk-lr ×0.25 on C repairs boundary routing) → PR-11 FAIL on
B3 clause (b) WITH clause (a) REPAIRED (frac 0.575 ≥ 0.5): forced_cost −0.1995 — killing
pattern completion IMPROVES B-eval. Measured mechanism: in PRIM, the non-A C segments
(42.5%) route by argmin into B-committed experts and contaminate them toward C; the FORCED
arm sends all C to a fresh slot and its B-experts stay B-calibrated. (Sign-convention
correction of the PR-08/PR-09 wrap-up prose recorded 2026-09-09; outcomes unaffected.)

## 1. Claim under test

Protecting committed-at-C-start slots other than the A-expert from ALL C-training (their
would-be segments are REDIRECTED to the A-expert; they are exempt from Policy A eviction
during C) closes the clause-(b) gap — PRIM-DE's B-eval lands within 0.10 of FORCED's —
while the repaired boundary routing HOLDS (frac ≥ 0.5) and the B1/B2 guards stay intact.
I.e.: the lifelong-routing claim fails today not because routing lacks value but because
argmin leaks C into B-experts; plug the leak and the ledger clears.

## 2. Protocol (PR-08/PR-11 verbatim + the one lever)

Stream, slices, corpora, tissue constants, routing semantics (veto + Policy A + the
trunk-lr ×0.25 C lever from PR-10/PR-11), seeds 0–4, LR rules — all EXACTLY
docs/preregistry/2026-09-09-repaired-flagship.md (PR-11). THE ONE CHANGE: during block C,
a guarded domain-exclusion lever (model attribute `domain_protect`, a plain Python set
pinned at C start = committed slots MINUS the a_expert) makes route_pr08
- REDIRECT every would-be assignment into a protected slot to the a_expert (the boundary
  window counts the post-redirect trained fraction), and
- EXEMPT protected slots from the Policy A eviction candidate pool
  (recruit assignments are unaffected: they target fresh/evicted slots, never protected
  ones; the forced-path early-return is untouched — forced semantics are already
  domain-exclusive).
Attribute absent ⇒ byte-identical PR-08/PR-11 paths (off-identity pinned by tests).

**Arm (1, treated):** PRIM-DE = PR-11's PRIM-LM cell + the lever above.
**Reused baselines (disclosed; same seeds, same lever, same frozen protocol, from
results/prizma_lm_PR-2026-09-03-11/powered.json, read-only):** PRIM-LM cells (the
argmin-C control), FORCED-RECRUIT cells (the clause-(b) baseline — forced semantics are
already exclusive), FROZEN-CHECKPOINT cells (the G2 adaptation floor; bit-identical to
PR-08's by the PR-11 canary).

**Fail-loud canary (every treated cell, claim and smoke mode):** bpc_A_preB AND
bpc_B_postB must equal PR-11's claim.PRIM-LM.s{seed} EXACTLY — the lever activates only at
C; any earlier divergence = ABORT (re-proves cross-process determinism and the lever's
blast radius per cell).

## 3. Bars (exact; n=5 seeds 0–4; Welch; t_isf UPPER-TAIL)

- **P1 (primary, the clause-(b) flip):** mean(PRIM-DE bpc_B_postC) ≥ mean(FORCED-11
  bpc_B_postC) − 0.10 — EXACT MEANS (the PR-08 B3b form). The PR-11 PRIM-LM B mean is
  recorded as the untreated control (measured 3.6120).
- **P2 (the repair must hold):** mean frac_to_A_expert (first 20 C-batches) ≥ 0.5
  (PR-11 measured 0.575 untreated; exclusion should only raise it).
- **G1 (retention guard):** BPC(A-eval, post-C) − BPC(A-eval, pre-B) ≤ 0.05, CI upper
  ≤ 0.05 (one-sample).
- **G2 (adaptation guard):** BPC(C-ret, post-C) advantage vs FROZEN-CHECKPOINT-11 ≥ 0.10
  (Welch, CI lower ≥ 0.10).
- Holm family = [G1, G2]; P1/P2 are means-based, untested. Reported, never gated: B4 per
  arm; churn incl. the redirect counter; per-seed fracs.

## 4. Pre-committed branches

- P1 + P2 + G1 + G2 all PASS ⇒ **CLAIMED** — domain-exclusion closes the value gap;
  S2c = the full 5-arm flagship with the exclusion lever (own prereg) before the wording
  question is revisited.
- P1 FAIL ⇒ **NEGATIVE** — the contamination hypothesis is insufficient; the next lead
  becomes eval-routing (where B segments land at eval time), own prereg.
- P2 FAIL ⇒ **NEGATIVE** — exclusion breaks the boundary routing it was meant to protect;
  investigate the redirect's surprise statistics before any new registration.
- Any guard FAIL ⇒ **NEGATIVE-GUARD** (record which); guard CI straddling ⇒ INCONCLUSIVE,
  n=10 (seeds 5–9) PRE-AUTHORIZED (03:55 rule wins).

## 5. Budget & runner

5 cells ≈ 6–8 min on the local CPU box. Runner: NEW seq/domainexc_claim.py mirroring
seq/trunklr_claim.py (single treated arm; reused baselines from the PR-11 ledger;
per-cell fingerprints include the lever; smoke/powered/--powered-cpu separation;
archive-before-verdict; A/B-phase canary per cell). Smoke on CPU first.

## 6. Self-audit

One lever, binary, pinned from measured provenance (PR-11's contamination finding); the
baselines are the SAME runs the claim compares against — no new control degrees of freedom;
the per-cell A/B canary proves the lever starts at C; what would change our mind: only
dated addenda. The registration's one bet: that B-eval damage flows through expert-head
contamination (training-time), not through eval-time routing of B segments.
