# PR-2026-09-03-18 — the repaired flagship REPLICATED on fresh seeds (5-block... 3-block, seeds 5–9)

**Lane:** CLAIM (replication) · **Status:** REGISTERED 2026-09-10 20:1x (frozen before any
run) · **Written:** 2026-09-10
**Provenance:** PR-13 CLAIM PASS (the repaired flagship, all three bars, seeds 0–4) →
PR-14 CLAIMED (5-block accumulation) → PR-15/16/17 (the control comparisons + the
schedule attribution). The flagship claim's one remaining soft spot is its sample size:
n=5. This registration is a **straight replication on FRESH seeds 5–9** — same protocol,
same levers, never-seen seeds, all bars re-adjudicated.

## 1. Claim under test

PR-13's verdict REPLICATES: on fresh seeds 5–9, the repaired flagship (L1 trunk-lr dose +
L2 domain-exclusion + the refinement) again clears B1 (retention-A ≤ 0.05), B2
(adaptation-C ≥ 0.10), and B3 (routing ledger: frac ≥ 0.5 AND forced cost ≤ 0.10).

## 2. Protocol

EXACTLY docs/preregistry/2026-09-03-13 (i.e. PR-08 as amended + L1 + L2 + the
refinement), with the ONLY change being the seed set: **seeds 5–9** (the runner's
registered `--seed-offset 5`). LR selection re-runs per the PR-07′ addendum rule
(seed-0-first-200-A-segments) and **must reproduce the registered lrs (fusion 3e-3,
shared 3e-3, frozen-checkpoint 1e-2) exactly** — a fail-loud canary (any drift = ABORT:
the selection rule is seed-0-based and seed-independent by construction). Ledger:
`prizma_lm_PR-2026-09-03-18` (PR-13's ledger untouched). Cross-arm structural canaries:
the A/B phases bit-identical between PRIM-LM and FORCED-RECRUIT per seed (the PR-08
determinism property), and FROZEN-CHECKPOINT honest aliasing. FORCED-RECRUIT pool-full
contingency + refinement carried as registered.

## 3. Bars (PR-08 §4 VERBATIM; n=5 FRESH seeds 5–9; Welch; t_isf UPPER-TAIL)

- **B1:** BPC(A-eval, post-C) − BPC(A-eval, pre-B) ≤ 0.05 (one-sample CI upper).
- **B2:** BPC_PRIM(C-ret, post-C) ≤ BPC_FROZEN-CHECKPOINT(same) − 0.10 (Welch; Holm
  over B1–B2 with B3 means-based).
- **B3:** frac to the A-recruited expert (first 20 C-batches) ≥ 0.5 AND forced-recruit
  cost ≤ 0.10.
- **B4:** reported per arm.
- Descriptive: the per-seed values alongside PR-13's seeds 0–4 (the pooled n=10 picture
  is REPORTED, never gated — the claim is the fresh-seed replication).

## 4. Pre-committed branches

- ALL THREE bars PASS ⇒ **CLAIMED (REPLICATED)** — the flagship claim now stands on two
  independent seed samples (n=10 cumulative); PR-LM-1 inherits everything.
- ANY bar FAIL on fresh seeds ⇒ **REPLICATION FAILED** — the original verdict is
  DOWNGRADED (the INDEX records the replication failure; the wording question reopens);
  this is the pre-committed cost of claiming at n=5.
- ANY bar INCONCLUSIVE ⇒ n→15 (seeds 10–14) PRE-AUTHORIZED; 03:55 rule wins.
- LR-selection canary drift ⇒ ABORT before any cell (investigate determinism first).

## 5. Budget & runner

25 cells ≈ 30–35 min CPU. Runner: prizma_lm_claim.py gains the guarded `--seed-offset`
flag (default 0 = byte-identical existing behavior; fingerprints include it; the fresh
ledger is separate via --ledger-dir). Smoke on CPU first.

## 6. Self-audit

The replication is the strongest post-claim hardening available on CPU: fresh seeds, same
frozen text, verdict pre-committed in BOTH directions (a fresh-seed failure DOWNGRADES the
claim — recorded before unblinding). What would change our mind: only dated addenda.

---

## Addendum 2026-09-10 #1 (pre-adjudication, maintainer) — canary seed scope

The PR-08 bit-identity canaries reference seeds 0-4 only. On this registration's FRESH
seeds (5-9) there is no PR-08 counterpart, so the FROZEN-arm canary SKIPS those seeds
(recorded per cell as cells_skipped_fresh_seed) instead of aborting; a PRESENT
reference with any mismatch still aborts loudly. On fresh seeds the frozen arms'
honesty is carried by the in-run structural checks (FROZEN-TRUNK's
backbone_frozen_check parameter identity; FROZEN-CHECKPOINT's zero-delta construction)
plus the LR-selection canary (which PASSED: 3e-3/3e-3/1e-2 reproduced exactly).
