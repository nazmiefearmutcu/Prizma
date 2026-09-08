# PR-2026-09-03-10 — trunk-lr scheduling on the returning block (the PR-09 negative's registered successor)

**Lane:** CLAIM · **Status:** REGISTERED 2026-09-09 01:2x (frozen before any run) · **Written:** 2026-09-09
**Provenance:** PR-08 (B3a FAIL frac 0.428; FROZEN-TRUNK routing domain-clean 0.96-0.99) →
PR-09 (floor freeze NEGATIVE: frac mean 0.405 vs 0.428 OFF, per-seed near-identical to OFF —
the boundary scatter is INSENSITIVE to floor pinning; guards held; churn fell 179→104) →
narrowed hypothesis: **the routing scatter is driven by the trunk's SHARED-WEIGHT drift
itself, not by floor re-calibration.** Dose-response prediction: reducing trunk plasticity
on the returning block should move frac toward the FROZEN-TRUNK regime (0.96-0.99)
proportionally, while keeping enough adaptation to clear the G2 bar.

## 1. Claim under test

Scaling the BACKBONE learning rate by 0.25 during block C only (tissue expert optimizers
UNCHANGED, floors live as PR-08) raises the returning-boundary routing fraction to >= 0.5
while retention (G1) and adaptation (G2) guards hold — i.e. the routing scatter is
monotone in trunk plasticity and separable from adaptation.

## 2. Protocol (inherits PR-08 verbatim unless stated)

Stream, slices, corpora, tissue constants, routing semantics, seeds 0-4: EXACTLY
docs/preregistry/2026-09-08-prizma-lm-flagship.md as amended. LR: 3e-3 frozen (PR-08
inheritance) for blocks A and B in BOTH arms; treatment arm uses backbone_lr_C = 3e-3 * 0.25
= 7.5e-4 during block C ONLY (backbone optimizer steps on C; tissue per-expert local
optimizers keep lr 3e-3; floor EMAs live/unfrozen — PR-09 settled that axis). The OFF arm
is PR-08 PRIM-LM verbatim and DOUBLES as the registered canary: its 5 cells must reproduce
PR-08 claim.PRIM-LM bit-identically (fail-loud ABORT otherwise).

**Arms (2, exclusive):** (1) OFF control; (2) TRUNK-LR-0.25C.

## 3. Bars (exact; n=5; Welch; t_isf UPPER-TAIL)

- **P1 (primary):** mean frac_to_A_expert (first 20 C-batches) for TRUNK-LR-0.25C >= 0.5
  (exact means). OFF's frac mean (0.428, re-measured) is the recorded control.
- **G1 (retention guard):** BPC(A-eval, post-C) - BPC(A-eval, pre-B) <= 0.05, CI upper
  <= 0.05 (one-sample), for TRUNK-LR-0.25C.
- **G2 (adaptation guard):** BPC(C-ret, post-C) advantage vs PR-08 FROZEN-CHECKPOINT cells
  (reused, lever-independent, disclosed) >= 0.10, CI lower >= 0.10 (Welch).
- **Reported:** B4 trunk drift per arm; churn per arm; per-seed fracs; the C-phase
  backbone-loss curve (descriptive dose evidence).

## 4. Pre-committed branches

- P1 PASS + G1 + G2 PASS => **CLAIMED** — trunk-plasticity scheduling repairs routing;
  PR-LM-1 designs inherit C-block trunk-lr scaling.
- P1 PASS + guard FAIL => **NEGATIVE (with lead)** — record which guard; a dose ladder
  (0.5 / 0.1) becomes a NEW prereg with the guard tradeoff as its object.
- P1 FAIL => **NEGATIVE** — plasticity dose (at 0.25) insufficient; next candidates in
  order: (a) C-block trunk lr = 0 for the shared predictor with a separate C-adapter
  (registerable only with a capacity control), (b) floor re-anchoring combined with slow
  trunk — each its own prereg. No dose re-tuning without a new registration.
- Guard CI straddling => INCONCLUSIVE; n=10 (seeds 5-9) PRE-AUTHORIZED, 03:55 rule wins.

## 5. Budget & runner

Extend seq/floorfreeze_claim.py's infrastructure in a NEW seq/trunklr_claim.py (mirror
it; arms OFF / TRUNK-LR-0.25C; per-cell fingerprints include the arm's C-lr; canary vs
PR-08 PRIM cells fail-loud; smoke/powered/--powered-cpu; archive-before-verdict).
10 cells ~ 15-20 min CPU. Smoke first.

## 6. Self-audit

One new knob (backbone C-lr scale = 0.25, frozen a priori — NOT tuned on data); OFF canary
pins reproducibility; FROZEN-CHECKPOINT reuse disclosed; P1/G1/G2 separate plasticity from
adaptation exactly as PR-09 separated floors. What would change our mind: only dated
addenda. The registration's bet: plasticity dose-response. If 0.25 fails AND churn/B4 show
no movement toward FROZEN-TRUNK behavior, the shared-predictor hypothesis itself weakens
and the successor must add capacity (adapter) rather than schedule lr.
