# Prizma-Seq v2 — Council Round 0 Synthesis (durable decision record)

> 2026-06-07. Produced after the three councils reviewed the v2 spec + plan in parallel. This is the
> authoritative record of what was decided and how the plan-of-record changes. Sources:
> `committee/verdicts/r0_general_plan-review.{md,json}`, `committee/quant/round1_levers.md`,
> `committee/standards/round1_landscape.md`.

## Verdicts
- **Council 1 (General, the gate): CONDITIONAL-REJECT / needs-experiments.** The direction is sound;
  the *artifacts* ship anti-conservative statistics and a lever-A correctness hole. Binding remedies
  R1–R10. Phase-1 accuracy runs blocked until R1/R2/R5/R7 land in `seq/stats.py`; lever A blocked
  until R3+R9; no FLOP claim until R4.
- **Council 2 (Quant): win-triad C+D+E**, plus three new high-EV levers (H/G/I). Sequence
  F → C+D+E → H → B → {A or G}. D must be MQAR-gated first.
- **Council 3 (Standards): bar raised.** Recall becomes a hard TOST-parity gate; per-FLOP ≤1.0× is
  unmet so "dramatic" is conditional; scope rider mandatory; add a tiny-hybrid baseline arm.

## Binding changes to the plan-of-record
1. **Statistics (R1,R2,R5,R7) — DONE/IN-FLIGHT** in `seq/stats.py`: real Student-t survival
   (incomplete-beta), correct TOST (two one-sided t at `t_{0.95,df}`), `margin_superiority` (BPC
   lower-is-better, ≥0.03), identical-model negative-control canary, Holm correction. No significance
   claim may be quoted until these are merged + validated.
2. **Lever A (R3,R9) — re-scoped:** the chunked surprise gate breaks on repeated keys (the recall
   signal). A's accuracy runs require EITHER an exact two-pass chunked form OR inference-only OR
   chunk=1, plus a **repeated-key equivalence test at <1e-4** (no 5e-3 random-data tolerance), plus
   two controls: a **random-scalar** gate and a **constant-mean β_eff** gate (separates "surprise-
   targeted" from "more average write"). Reframe A as capacity-*reallocation*, not capacity-add.
3. **Lever H (NEW, promoted ahead of A):** decoupled channel-wise erase/write (Gated-DeltaNet-2) —
   split β into a key-side erase gate + a value-side write gate. Param-matched, ~5% train cost,
   strong recall+LM gain. Highest-confidence "free" win; becomes a Phase-1 task.
4. **Lever G (candidate novel core):** RWKV-7 in-context per-channel learning rate — generalizes the
   scalar β and may subsume both gating (C) and surprise (A). Carried as the A-alternative for the
   "novel core" slot; the one that wins the ablation is kept.
5. **Lever I (free):** stochastic training window size — bolt onto E at zero extra cost.
6. **Lever D (R9 + caveat):** gate is the **end-to-end ≥10-seed MQAR-D128 recall** at the reduced
   d_φ, run BEFORE any LM run — not just the crosstalk probe. Back off d_φ if the solve point regresses.
7. **d_φ reconciliation (R4) — RESOLVED.** code default `feat_n2=96` → d_φ=128; the REPORT's headline
   FLOP/recall numbers used d_φ=256 (= the v1 full-quad2 reference, `feat_n2=224`); v2 lean target =
   d_φ=137 (`quad2_lowrank`, r=14). Per-config FLOP ledger **re-emitted** for all four configs
   (none/32, code/128, v1/256, lean/137) at the headline scale d128L4H4, each FLOP number pinned to its
   exact `(feat_map, feat_n2/feat_rank → d_φ)` config AND tied to a REAL param-matched module pair
   (param-match holds for all four, +0.6% vs TF). See `flop_ledger.py::emit_per_config_ledger`,
   `results/flop_ledger_v2.json` / `results/flop_ledger_v2.txt`, the REPORT "d_φ reconciliation (R4)"
   note, and `tests/test_flop_ledger.py`. As-coded / ideal ratios vs param-matched TF: none 1.36×/1.00×,
   code-d128 1.70×/1.34×, v1-d256 2.14×/1.78×, lean-d137 1.73×/1.37×. **The canonical v2 d_φ is NOT
   chosen here** — it remains LOCKED by the pending A100 ≥10-seed MQAR-D128 solve-rate gate (Task 1.D,
   item 6 / Lever D).
8. **LR fairness (R6):** Stage-1 LR uses ≥3 seeds for the bimodal legs (select on solve-rate→median,
   not best_acc@1seed) and is re-swept **per lever combination**; drop the "μP-aware" label unless a
   coordinate-check passes.
9. **§3 bar (R10 + Council-3):** mark latency / abs-length-extrap / per-FLOP targets **conditional on
   their enabling levers landing**; abs-length-extrap is a Pareto knob (no mechanism targets it
   directly). ADD: (a) a **recall TOST-parity hard gate**; (b) a **matched tiny-hybrid baseline arm**;
   (c) a **mandatory scope rider** on every claim. "Dramatic" requires char-LM(iso-FLOP) + all-n
   latency + O(1) memory + recall-parity holding **simultaneously and powered**; drop one → downgrade
   to "Pareto-efficient / -competitive in the tested regime".

## Revised Phase-1 sequence (plan-of-record)
F (train-speed enabler, parallelizable) → **C + E** (approved now, post-R1/R2/R8) → **H** → **D**
(MQAR-gated) → **B** → **{A vs G}** (novel-core ablation, post-R3/R9). Recall-gate + tiny-hybrid arm
run alongside the headline matrix. Each lever: G1 O(1) guard + G2 chunked==reference, then ≥10-seed
A100 validation, then Council-1 sign-off on its causal-attribution claim.
