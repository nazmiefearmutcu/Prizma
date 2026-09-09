# PR-2026-09-03-16 — the B-boundary drift-damage head-to-head (confirmatory analysis of registered artifacts)

**Lane:** CLAIM · **Status:** REGISTERED 2026-09-10 01:1x (frozen before any analysis run) · **Written:** 2026-09-10
**Provenance:** PR-15 NEGATIVE (the A-retention bar was non-discriminating: the curriculum
re-exposes text8 at C and E) — but its ledgers measured the DISCRIMINATING metric
descriptively: **B-boundary drift damage** (bpc_B(post_C) − bpc_B(post_B), the degradation
incurred by training on block C) is ~+1.15 bpc for the WINDOW-TF control vs ~+0.60 bpc for
the repaired column (PR-14 EX). This registration freezes the head-to-head ANALYSIS of
those two registered artifacts BEFORE the formal test is computed.

## 1. Claim under test

The repaired column's B-boundary drift damage is smaller than the memory-matched
WINDOW-TF control's by a CI-established margin of at least 0.25 bpc (half the descriptively
measured ~0.55 gap) — i.e. the tissue HALVES boundary damage, at claim grade.

## 2. Protocol (analysis of registered artifacts — no new training)

Damage definitions (both from registered ledgers, read-only):
- COLUMN damage per seed s: `claim.EX.s{s}.bpc_B_postC − claim.EX.s{s}.bpc_B_postB`
  (results/manyblock_PR-2026-09-03-14/powered.json).
- CONTROL damage per seed s: `claim.WINDOW-TF.s{s}.bpc_B_postC − claim.WINDOW-TF.s{s}.bpc_B_postB`
  (results/windowtf_manyblock_PR-2026-09-03-15/powered.json).
Seeds 0–4 both sides (the registered n=5 pairs). Delta = mean(CONTROL damage) −
mean(COLUMN damage); positive = the column damages less. Welch two-sample, t_isf
UPPER-TAIL convention.

**Disclosed post-hoc status:** the ~0.55 point gap was descriptively visible in PR-15's
console/INDEX before this freeze; the bar below is fixed at HALF that point estimate so
the formal test is not anchored to the full observed gap. No new data is collected; if
the n=5 analysis is underpowered, the n=10 extension (seeds 5–9, NEW training by the
PR-14/PR-15 runners with a registered seed extension) is PRE-AUTHORIZED as the next
iteration.

## 3. Bar (exact; Welch; t_isf UPPER-TAIL)

- **P1 (primary, single):** delta = mean(CONTROL damage) − mean(COLUMN damage);
  **PASS iff Welch CI lower ≥ 0.25**; FAIL-side: if CI upper < 0.25 ⇒ NOT-ESTABLISHED
  (the n=10 pre-authorization fires); straddle ⇒ INCONCLUSIVE-NARROW (also n=10).
  Descriptive: per-seed damages both sides; the revisit-recovery head-to-head
  (CONTROL b_recovery_postD vs COLUMN's PR-14 recovery) reported, never gated.

## 4. Pre-committed branches

- P1 PASS ⇒ **CLAIMED** — the tissue halves boundary damage at claim grade; the
  comparison ledger's first established gap.
- CI upper < 0.25 ⇒ **NOT-ESTABLISHED** — the point gap stands but n=5 cannot establish
  half-of-it; the n=10 seed extension is the next iteration's first action.
- Delta ≤ 0 (the control damages less) ⇒ **NEGATIVE** — the descriptives were misleading.
- INCONCLUSIVE-NARROW ⇒ same as NOT-ESTABLISHED (n=10).

## 5. Runner

NEW seq/boundary_damage_claim.py: reads the two ledgers read-only, recomputes the damages,
computes the Welch verdict, writes its OWN ledger + archive (no training; runs in
seconds; --powered-cpu semantics trivially satisfied — the analysis is CPU-native; the
powered/CUDA refusal does not apply but the mode flags are accepted for pattern parity).

## 6. Self-audit

Two registered artifacts, one pre-frozen bar at half the observed effect, one Welch test.
The post-hoc visibility of the gap is disclosed; the halving bar is the honest response.
What would change our mind: only dated addenda.
