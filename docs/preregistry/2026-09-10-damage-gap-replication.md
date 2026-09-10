# PR-2026-09-03-20 — the boundary-damage gap REPLICATED on fresh seeds (confirmatory analysis of the PR-19 ledger)

**Lane:** CLAIM (replication) · **Status:** REGISTERED 2026-09-10 22:1x (frozen before any
analysis run) · **Written:** 2026-09-10
**Provenance:** PR-16 CLAIMED (the boundary-damage gap at seeds 0-4: delta +0.3685, CI
[0.27, 0.47]) → PR-19 CLAIMED (the accumulation claim replicated on fresh seeds 5-9, both
arms re-run). The PR-19 ledger therefore CONTAINS the fresh-seed damage pairs this
registration needs: PLAIN cells (argmin, L1) vs EX cells (L1 + L2) at seeds 5-9. This
completes the replication set: after PR-18 (flagship) and PR-19 (accumulation), the
damage-protection gap is the last established claim without a fresh-seed replication.

## 1. Claim under test

PR-16's verdict REPLICATES on fresh seeds: the EX column's B-boundary damage is smaller
than the fresh PLAIN arm's by a CI-established margin ≥ 0.25 bpc.

## 2. Protocol (analysis of registered artifacts — no new training)

Damage per seed s ∈ {5..9}: `claim.PLAIN.s{s}.b_degradation_B_postC −
claim.EX.s{s}.b_degradation_B_postC`... concretely both sides read from
results/manyblock_PR-2026-09-03-19/powered.json (read-only):
- EX damage: `claim.EX.s{s}.bpc_B_postC − claim.EX.s{s}.bpc_B_postB`
- PLAIN damage: `claim.PLAIN.s{s}.bpc_B_postC − claim.PLAIN.s{s}.bpc_B_postB`
Delta = mean(PLAIN damage) − mean(EX damage); positive = the exclusion damages less.
Welch two-sample, t_isf UPPER-TAIL. (Both arms share A/B training per seed, so the
postB baselines are bit-identical across arms — the PR-19 canary proved it.)

## 3. Bar (exact; Welch; t_isf UPPER-TAIL)

- **P1 (primary, single):** delta; **PASS iff Welch CI lower ≥ 0.25** (PR-16's bar —
  pre-fixed, the same halving bar); FAIL-side: delta ≤ 0 ⇒ **NEGATIVE**; otherwise
  (positive point, CI upper < 0.25 or straddle) ⇒ **NOT-ESTABLISHED at n=5**.
- Descriptive: per-seed damages both arms; the comparison vs PR-16's seeds-0-4 delta
  (+0.3685) reported.

## 4. Pre-committed branches

- P1 PASS ⇒ **CLAIMED (REPLICATED)** — every established claim of the campaign now has a
  fresh-seed replication; the replication set is COMPLETE.
- NOT-ESTABLISHED ⇒ the seeds 10-14 extension (PLAIN + EX at offset 10, both arms, n=10
  fresh cumulative) is PRE-AUTHORIZED as the next iteration's first action.
- delta ≤ 0 ⇒ **NEGATIVE** — the gap was a small-sample artifact; PR-16's interpretation
  is downgraded.

## 5. Runner

NEW seq/boundary_gap_replication.py: reads the PR-19 ledger read-only, computes the Welch
verdict, writes its OWN ledger + archive (analysis-only, seconds). The damages helper is
REUSED conceptually from seq/boundary_damage_claim.py (different ledgers/keys, so a local
implementation with the same shape; the shared comparator alias is kept for parity).

## 6. Self-audit

Zero new data, zero new degrees of freedom: the bar is PR-16's own bar; the cells are
PR-19's registered artifacts; the lever states are in both ledgers' fingerprints. What
would change our mind: only dated addenda.
