# PR-2026-09-03-21 — revisit-recovery attribution (confirmatory analysis; closes the attribution matrix)

**Lane:** CLAIM (attribution analysis) · **Status:** REGISTERED 2026-09-11 00:1x (frozen
before any analysis run) · **Written:** 2026-09-11
**Provenance:** PR-17 attributed the boundary-DAMAGE protection to the lr schedule
(SCHEDULE-CARRIED; the tissue's residual ~0). The attribution matrix has one cell left
open: **the revisit RECOVERY** (bpc_B(post_C) − bpc_B(post_D), the +bpc improvement when
the stream re-enters shakespeare). Descriptives already on record: COLUMN recovery
+0.82 (PR-14), SCHED-TF +0.83 (PR-17 console), PLAIN-TF ~+0.90-1.08 (the PR-14 probe +
PR-19's fresh PLAIN). This registration freezes the three-way recovery comparison BEFORE
it is formally computed.

## 1. Claim under test (attribution)

Which component, if any, carries the revisit recovery: the lr schedule, the tissue, or is
recovery **universal** (a property of re-encounter training itself, present in every
form)? No pass/fail polarity: the attribution pair IS the result.

## 2. Protocol (analysis of registered artifacts — no new training)

Recovery per seed s ∈ {0..4}: `bpc_B(post_C) − bpc_B(post_D)`, from three registered
ledgers (read-only):
- COLUMN: results/manyblock_PR-2026-09-03-14/powered.json claim.EX.s{s}
- SCHED-TF: results/windowtf_sched_PR-2026-09-03-17/powered.json claim.WINDOW-TF.s{s}
- PLAIN-TF: results/windowtf_manyblock_PR-2026-09-03-15/powered.json claim.WINDOW-TF.s{s}
(The PR-17 scheduled ledger's cells carry the schedule; all three share the stream and
the eval slices.)

## 3. Comparisons (exact; n=5 v n=5 Welch, two-sided 95% CIs; NO small-p gate)

- **C1:** delta = mean(SCHED-TF recovery) − mean(COLUMN recovery). "SCHED-ADVANTAGED"
  iff CI excludes 0 AND |point| ≥ 0.10; "COLUMN-ADVANTAGED" iff CI excludes 0 on the
  other side AND |point| ≥ 0.10; else parity-at-this-resolution.
- **C2:** delta = mean(PLAIN-TF recovery) − mean(COLUMN recovery), same rule.
- **Attribution table (pre-committed):**
  - C1 parity AND C2 parity ⇒ **RECOVERY-UNIVERSAL** — every form recovers the
    re-encountered domain; recovery is a property of re-encounter training itself.
  - C1 SCHED-ADVANTAGED ⇒ the schedule also carries recovery.
  - C2 PLAIN-ADVANTAGED (with C1 parity) ⇒ the TISSUE COSTS recovery magnitude — the
    ledger's honesty price, recorded.
  - C2 COLUMN-ADVANTAGED ⇒ the tissue amplifies recovery beyond plain re-training.
- Descriptive: all three recovery distributions per seed; also the absolute post_D BPCs
  (who lands lowest after the revisit — the quality question, reported not gated).

## 4. Pre-committed branches

- RECOVERY-UNIVERSAL ⇒ the attribution matrix CLOSES: damage = schedule-carried;
  recovery = universal; routing ledger + purity = tissue-only. The comparison record is
  complete; S3 remains the ladder's next rung.
- Any ADVANTAGED cell ⇒ the named component owns recovery; a tissue-costs-recovery
  result additionally becomes a design lead (the ledger machinery vs recovery trade-off).
- No extension pre-authorized (n=5v5v5 with the recorded spreads resolves or it does
  not; a follow-up would be a NEW registration).

## 5. Runner

NEW seq/recovery_attribution.py: reads the three ledgers read-only, computes the three
recoveries + the two CIs, writes its OWN ledger + archive (analysis-only, seconds;
mirrors seq/boundary_gap_replication.py).

## 6. Self-audit

All cells are registered artifacts; the thresholds (0.10, CI-excludes-0) are frozen
before computation; the descriptive post_D quality is reported but cannot flip the
attribution. What would change our mind: only dated addenda.
