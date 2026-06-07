# Council 1 — The General Council (the GATE)

**Role:** the cross-disciplinary review gate. Every major decision (architecture change, claim,
experiment design, statistical method) is presented to this council. **On rejection, the rejection
cause must be fixed before proceeding** — a reject is binding, not advisory.

## Members (lenses)
1. **ML-systems engineer** — kernels, memory, throughput, feasibility on the actual hardware.
2. **Statistics / experimental-design referee** — power, seeds, CIs, TOST/superiority, multiple
   comparisons, pre-registration.
3. **Optimization theorist** — LR/warmup fairness, μP/coordinate-checks, convergence vs capacity.
4. **Information theorist** — capacity/recall arguments, what a mechanism can vs cannot represent.
5. **Reproducibility / research-integrity referee** — seed-pinning, RNG hygiene, leakage, overclaiming.
6. **Adversarial skeptic** — "what is the single most likely way this produces a misleading win?"

## Protocol
`decide → present → approve / reject / needs-experiments → (if not approve) remedy → re-present`.
Output every round to `committee/verdicts/r{N}_general_{topic}.{md,json}`. The JSON follows
`committee/verdicts/_TEMPLATE.json`. A decision is **cleared** only on an explicit `approve` with the
scope of approval named.

## Standing honesty guardrails (binding on every reported number)
Inherits `committee/round1_synthesis.md` §D: param-match every head-to-head; measured FLOP+throughput
ledger; O(1) `step()==forward()<1e-4` guard before any accuracy number; causal ablation with controls;
no best-of-noisy-curve (frozen eval); report failure modes + solve-rate + median + CI over ≥ the
pre-registered seed count; pre-register D\*/margins; both sides of every contrast; seed-frozen
architecture choices; crash-safe audit trail.

## Round 0 verdict (2026-06-07)
**CONDITIONAL-REJECT / needs-experiments** on the v2 plan artifacts (not the direction). Binding
remedies R1–R10 (`committee/verdicts/r0_general_plan-review.md`): real Student-t p-values (R1),
correct TOST (R2), lever-A exactness on repeated keys (R3), d_φ reconciliation (R4),
margin-superiority test (R5), per-combination LR fairness (R6), identical-model negative control +
Holm (R7), RNG-generator hygiene + CUDA repro (R8), lever-A/D second controls (R9), §3 targets marked
conditional on their enabling levers (R10). **Phase 1 accuracy runs are blocked until R1/R2/R5/R7
land in `seq/stats.py`; lever A is blocked until R3+R9; no FLOP claim until R4.**
