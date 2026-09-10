# PR-2026-09-03-19 — many-block accumulation REPLICATED on fresh seeds (5–9; 10–14 pre-authorized)

**Lane:** CLAIM (replication) · **Status:** REGISTERED 2026-09-10 21:2x (frozen before any
run) · **Written:** 2026-09-10
**Provenance:** PR-14 CLAIMED — ALL FIVE BARS (the 5-block stream with a literal
shakespeare revisit: recovery +0.82, re-engagement 0.971, A-retention −0.32, harm −0.008,
text8 −0.12). PR-18 just replicated the 3-block flagship on fresh seeds (n=10 fresh,
all bars). This registration applies the same hardening to the accumulation claim: **a
straight replication on FRESH seeds 5–9.**

## 1. Claim under test

PR-14's verdict REPLICATES: on fresh seeds 5–9, the EX arm again (i) recovers the
re-encountered domain by ≥ 0.10 bpc (P1), (ii) re-engages the B-expert in the revisit
window at frac ≥ 0.5 (P2), (iii) holds domain A through 5 blocks (P3), while the guards
hold (G1 no revisit-cost vs PLAIN-fresh, G2 text8 continuity).

## 2. Protocol

EXACTLY docs/preregistry/2026-09-09-manyblock-accumulation.md (PR-14) with the ONLY
change being the seed set: **seeds 5–9** via the runner's registered `--seed-offset 5`.
BOTH arms re-run (PLAIN + EX — the cross-arm A/B canary needs both, and G1 compares
fresh-PLAIN vs fresh-EX). LR: frozen by inheritance (3e-3; backbone 7.5e-4 post-A) —
no selection leg, no canary needed there. Ledger: `manyblock_PR-2026-09-03-19`
(PR-14's ledger untouched).

## 3. Bars (PR-14 §3 VERBATIM; n=5 FRESH seeds 5–9)

- **P1:** revisit recovery (bpc_B(post_C) − bpc_B(post_D)) mean ≥ 0.10, CI lower ≥ 0.10.
- **P2:** D-window frac to the shakespeare owner ≥ 0.5 (exact means).
- **P3:** bpc_A(post_E) − bpc_A(post_A) CI upper ≤ 0.05.
- **G1:** EX at most 0.10 worse than fresh-PLAIN on bpc_B(post_D) (CI-positional).
- **G2:** text8 continuity CI upper ≤ 0.05.
- Holm = [P1, P3]; P2 means-based; G1/G2 CI-positional guards (the PR-14 addendum #1
  semantics). Descriptive: per-seed trajectories alongside PR-14's seeds 0–4 (the pooled
  n=10 fresh picture REPORTED, never gated — the claim is the fresh-seed replication).

## 4. Pre-committed branches

- ALL PASS ⇒ **CLAIMED (REPLICATED)** — the accumulation claim stands on two independent
  seed samples (n=10 cumulative fresh).
- P1 or P2 FAIL on fresh seeds ⇒ **REPLICATION FAILED** — PR-14's verdict is DOWNGRADED
  (INDEX records it; the revisit claim narrows to the original seeds).
- P3 or G2 FAIL ⇒ **REPLICATION FAILED (GUARD)** — accumulation does not survive fresh
  seeds.
- G1 FAIL ⇒ **REPLICATION FAILED (GUARD)** — exclusion costs the revisit on fresh seeds.
- Any INCONCLUSIVE ⇒ seeds 10–14 PRE-AUTHORIZED (the §5 pattern).

## 5. Budget & runner

10 cells ≈ 22–28 min CPU. Runner: seq/manyblock_claim.py gains the guarded
`--seed-offset` flag (default 0 = byte-identical; fingerprints include it; own ledger via
--ledger-dir). Smoke on CPU first.

## 6. Self-audit

Mirrors PR-18's replication design (which PASSED cleanly) on the accumulation claim:
fresh seeds, frozen text, both-directions pre-commitment. What would change our mind:
only dated addenda.
