# PR-2026-09-03-09 — floor-freeze routing repair (the PR-08 B3a negative's registered successor)

**Lane:** CLAIM · **Status:** REGISTERED 2026-09-09 00:1x (frozen before any run) · **Written:** 2026-09-09
**Provenance:** PR-08's registered verdict (FAIL on B3 clause (a): boundary-window frac to the
A-expert mean 0.428 < 0.5, per-seed 0.325/0.470/0.294/0.291/0.761) + its B4 accounting
(tissue bounds trunk drift: PRIM +1.20 < FROZEN-TRUNK +1.87 < SHARED-HEAD +2.78) + the
measured mechanism signature: FROZEN-TRUNK's boundary routing stays domain-clean
(frac 0.96–0.99) while PRIM's trained trunk scatters it → hypothesis: **trunk training
re-calibrates the tissue's precision floors during B/C, so the returning-domain boundary
loses its novelty signal and routing scatters.**

## 1. Claim under test

Pinning the tissue's precision floors (per-expert μ/σ² EMAs) to their block-A calibration
restores clean boundary routing at the returning domain (B3a repaired to ≥ 0.5) WITHOUT
sacrificing what the column already delivers: negative forgetting of A (B1 guard) and
C-adaptation (B2 guard).

## 2. Protocol (inherits PR-08 verbatim)

Stream, slices, corpora (archived sha256), tissue constants (E=4, M_max=4, z=5,
freeze_min_seen=300, H=64, FLOOR_EMA=0.05), SEG/BATCH_SEGS, seeds 0–4, and the PR-08
routing semantics (floor-maturity veto + Policy A eviction) are EXACTLY
docs/preregistry/2026-09-08-prizma-lm-flagship.md as amended (addenda #1–#3). LR is
FROZEN at 3e-3 for both arms (inherited from PR-08's registered fusion-family selection —
no re-selection; comparability with PR-08 is the point).

**Arms (2, exclusive):**
1. **OFF control** — PR-08 PRIM-LM verbatim (no freeze machinery active).
2. **FROZEN-FLOORS (treatment)** — at block-A end, the μ/var entries of all committed
   slots are PINNED: from B onward, `_expert_train`'s floor update is skipped for those
   slots. A slot re-initialized by Policy A eviction LEAVES the frozen set (recalibrates
   freely on its new domain); post-A fresh recruits are never in the set. Expert
   predictors/optimizers are NOT frozen — only floor statistics.

The freeze is implemented as a guarded default-off branch (`getattr(model, "floor_freeze",
None)`) in seq/fusion_probe.py `_expert_train` + a pop in seq/prizma_lm_claim.py
`route_pr08`'s eviction re-init; with the attribute absent both files are semantically
byte-identical to PR-08 (off-identity pinned by tests).

## 3. Bars (exact; n=5 seeds 0–4; Welch; t_isf UPPER-TAIL convention)

- **P1 (primary, repair):** mean over 5 seeds of `boundary_window.frac_to_A_expert`
  (first 20 C-batches, TRAINED segments) for FROZEN-FLOORS **≥ 0.5** (exact means, the
  PR-08 B3 form). The OFF arm's frac is the recorded control (PR-08 measured 0.428).
- **G1 (retention guard, must hold):** BPC(A-eval, post-C) − BPC(A-eval, pre-B) ≤ 0.05
  for FROZEN-FLOORS (one-sample, CI upper ≤ 0.05).
- **G2 (adaptation guard, must hold):** BPC(C-ret, post-C) advantage of FROZEN-FLOORS vs
  FROZEN-CHECKPOINT ≥ 0.10 (Welch, CI lower ≥ 0.10). The FROZEN-CHECKPOINT baseline cells
  are REUSED from the PR-08 powered ledger (disclosed: that arm has no tissue, is
  lever-independent, same seeds and frozen protocol).
- **Reported, never gated:** B4 trunk drift per arm; routing-ledger churn (recruits/
  evictions/vetoed) per arm; per-seed fracs; OFF-arm churn vs PR-08.

## 4. Canaries (fail-loud)

- The OFF control cells must reproduce PR-08's `claim.PRIM-LM.s{0..4}` records BIT-IDENTICALLY
  (same fingerprints; any mismatch = ABORT before the treatment arm and investigate — this
  also re-proves cross-process determinism end-to-end).
- The FROZEN-FLOORS A-phase (pre-freeze) must match the OFF arm's A-phase bit-for-bit
  (the freeze activates only after A).

## 5. Pre-committed branches

- P1 PASS + G1 + G2 PASS ⇒ **CLAIMED** — trunk-floor decoupling confirmed as the B3a
  mechanism; PR-LM-1 designs inherit floor freezing on returning blocks.
- P1 PASS + any guard FAIL ⇒ **NEGATIVE (with lead)** — the lever buys routing at the
  cost of the column; record which guard broke; floor scheduling (e.g. freeze only on C)
  becomes the next candidate.
- P1 FAIL ⇒ **NEGATIVE** — floor freeze insufficient; the next ladder candidate (trunk-lr
  scheduling on returning blocks) gets its own prereg. No re-tuning of this lever without
  a new registration.
- Any guard CI straddling its margin ⇒ INCONCLUSIVE for that guard; seed extension to
  n=10 (seeds 5–9) is PRE-AUTHORIZED by this registration, executed time permitting (the
  03:55 wind-down rule always wins).

## 6. Budget & runner

10 cells ≈ 13–15 min on the local CPU box (PR-08 measured 61–84 s/cell). Runner: NEW
seq/floorfreeze_claim.py mirroring prizma_lm_claim.py (smoke/powered separation, --powered
CUDA refusal + --powered-cpu fallback via the same needs_cuda contract, per-cell
fingerprints INCLUDING the arm's freeze state, crash-safe _save per cell,
archive-before-verdict, VERBATIM bars above). Smoke on CPU first.

## 7. Self-audit

Every knob inherited from PR-08's frozen text (no new tuning surface except the freeze
itself, which is binary); lr frozen by inheritance; OFF canary pins reproducibility;
FROZEN-CHECKPOINT reuse is lever-independent and disclosed; what would change our mind:
only dated addenda. The registration's one bet: that the FROZEN-TRUNK-vs-PRIM routing
signature is caused by floor re-calibration (not by trunk-weight drift in the shared
predictor path) — G1/G2/P1 together separate these: if floors are the cause, P1 passes
with guards intact; if trunk drift itself is the cause, freezing floors cannot restore
routing and P1 fails.
