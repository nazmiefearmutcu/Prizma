# PR-2026-09-03-14 — many-block accumulation with a true revisit (5 blocks, 3 drift boundaries)

**Lane:** CLAIM · **Status:** REGISTERED 2026-09-09 23:1x (frozen before any run) · **Written:** 2026-09-09
**Provenance:** PR-13 CLAIM PASS (the 3-block repaired flagship) + the manyblock probe
(results/exploratory/manyblock_probe_2026-09-09/, LANE-EXPLORATORY, n=2): under plain
argmin the literal shakespeare REVISIT recovered B-eval by ~1.08 bpc (3.96 -> 2.89, below
even the post-B level 3.04), A-retention held through 5 blocks (2.71 post-E), text8
accumulated (Cret 2.74). The lifelong question this registration answers: **does the
repaired column ACCUMULATE — retain its first domain, recover a re-encountered domain,
and keep the routing ledger honest — across a 5-block stream, and does domain-exclusive
routing (L2 generalized with per-block owner election) match plain argmin's recovery
while preventing cross-domain leakage?**

## 1. Claim under test

Across the 5-block stream (A=text8[0,1M) -> B=shakes[0,90%) -> C=text8[1.1M,2.1M) ->
D=shakes[0,90%) REVISITED -> E=text8[2.1M,3.1M)), the fused column with L1 (backbone
x0.25 on every post-A block) and — in the EX arm — L2 generalized (per-block domain-owner
election + protection + the PR-13 suppression refinement) (i) preserves domain A
(retention), (ii) RECOVERS the re-encountered B domain by >= 0.10 bpc, (iii) routes the
revisit's early segments back to the B-expert (re-engagement ledger), while matching the
plain-argmin arm's recovery (the exclusion must not cost the revisit).

## 2. Protocol

Corpora: the archived text8/shakespeare files (committed sha256; download failure =
ABORT). Levers: L1 dose 7.5e-4 on blocks B, C, D, E (generalized from PR-10/PR-11's C-only
dose — every post-A block is a returning-or-new drift block); L2 refinement semantics
exactly as PR-13 (suppress rather than self-evict; counted). **Per-block owner election
(NEW, this registration):** at each post-A block start, the block's OWNER slot is the
committed slot with the maximum cumulative trained-segment count in the block's DOMAIN
(text8 = A+C(+E) shares; shakespeare = B(+D) shares); a domain with no history elects no
owner (first encounters route/recruit freely, PR-08 semantics). Protected slots =
committed MINUS the owner. The guarded route_pr08 branches read `boundary["owner_slot"]`
(falling back to `a_expert` when absent — PR-13 behavior unchanged). Suppression + the
redirect counter are recorded per block.

**Arms (2, exclusive):**
1. **PLAIN** — L1 only, plain argmin (the probe's regime, now a registered arm).
2. **EX** — L1 + L2 generalized (owner election + protection + refinement).
n=5 seeds 0-4, both arms. **Canary:** the A and B phases are bit-identical ACROSS arms
(bpc_A_preB, bpc_A_postB, bpc_B_postB equal) — divergence begins at C (the first protected
block); asserted in smoke via the PR-11 ledger? NO — the 5-block stream never ran as a
claim; smoke asserts cross-arm A/B identity internally (fail-loud), and claim mode asserts
it per cell (abort on drift).

## 3. Bars (exact; n=5 seeds 0-4; Welch; t_isf UPPER-TAIL)

- **P1 (revisit recovery, EX arm):** per-seed diffs d_i = bpc_B(post_C) − bpc_B(post_D);
  one-sample mean >= 0.10 with CI lower >= 0.10 (Holm family member).
- **P2 (re-engagement ledger, EX arm):** fraction of D's first-20-batch trained segments
  routed to the B-expert (the shakespeare owner) >= 0.5 — exact means (the PR-08 B3a form).
- **P3 (accumulation, EX arm):** bpc_A(post_E) − bpc_A(post_A) <= 0.05 with CI upper
  <= 0.05 (Holm family member).
- **G1 (exclusion must not cost the revisit):** Welch — mean(bpc_B(post_D)) of PLAIN minus
  EX >= −0.10 i.e. EX is at most 0.10 WORSE than PLAIN; CI-based: FAIL iff EX is worse by
  CI-established margin > 0.10 (CI lower of (PLAIN − EX) > 0.10 -> FAIL; CI upper < 0.10
  -> PASS; straddle -> INCONCLUSIVE) (Holm family member).
- **G2 (text8 continuity, EX arm):** bpc_Cret(post_E) − bpc_Cret(post_C) <= 0.05 with CI
  upper <= 0.05 (Holm family member).
- Reported, never gated: both arms' full B/A/Cret trajectories per block; churn +
  redirect/suppression counters per block; the PLAIN arm's revisit recovery (descriptive,
  the probe predicted ~1.08).

Holm family = [P1, P3, G1, G2] (P2 is means-based, untested). Overall: any INCONCLUSIVE ->
INCONCLUSIVE (n=10, seeds 5-9, PRE-AUTHORIZED); P1+P2+P3+G1+G2 all PASS -> CLAIMED;
P1/P2/P3 FAIL -> NEGATIVE (echo which); only guards FAIL -> NEGATIVE-GUARD.

## 4. Pre-committed branches

- ALL PASS ⇒ **CLAIMED** — the column accumulates and recovers at many-block scale with a
  clean re-encounter ledger; the 'lifelong' claim now rests on 5 blocks; PR-LM-1 inherits
  the generalized owner election.
- P2 FAIL ⇒ **NEGATIVE** — the re-encounter is real (P1) but the ROUTING LEDGER does not
  re-engage the B-expert; 'lifelong routing' stays wording-limited to single-boundary.
- P1 FAIL ⇒ **NEGATIVE** — no recovery at claim grade (probe was a fluke or n=2 flattered
  it); the many-block design is abandoned for this mechanism class.
- G1 FAIL ⇒ **NEGATIVE-GUARD** — exclusion COSTS the revisit; plain argmin remains the
  deployment form; exclusion retires to the PR-11 scope.
- P3/G2 FAIL ⇒ **NEGATIVE-GUARD** — accumulation does not hold at 5 blocks.

## 5. Budget & runner

10 cells ≈ 22-28 min CPU (5 blocks ≈ 5.1M chars ≈ 20k segments/cell). Runner: NEW
seq/manyblock_claim.py mirroring seq/trunklr_claim.py (2 arms, the generalized owner
election inside route_pr08's guarded branches via boundary["owner_slot"], per-cell
fingerprints incl. arm + lever states, cross-arm A/B canary, smoke/powered/--powered-cpu,
archive-before-verdict, VERBATIM bars). Smoke on CPU first.

## 6. Self-audit

Every bar's margin is anchored to a MEASURED exploratory value (recovery ~1.08 vs bar
0.10; retention ~ -3.3 vs bar 0.05; second-visit 2.89 < first 3.04). The owner election is
deterministic from the training ledger (no tuning surface). The PLAIN arm is registered —
not reused — so the exclusion-cost guard G1 is a real comparison. What would change our
mind: only dated addenda. The one bet: that the probe's revisit dynamics survive
registration-grade n=5 and that exclusion does not interfere with re-encounter recovery.
