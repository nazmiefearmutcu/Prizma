# Prizma-Seq v2 — Pareto-Dominance Design Spec

> **Date:** 2026-06-07 · **Status:** approved (owner) → writing-plans next
> **Repo:** `/Volumes/disk 2/Desktop_Migrate_2026-05-28/Projeler/proje/PRISM`
> **Predecessor record:** `docs/PRIZMA_SEQ_REPORT.md`, `committee/round1_synthesis.md`, `docs/TRANSFORMER_ALTERNATIVE_BRIEF.md`

## 0. Mission (owner, locked)
Take Prizma-Seq from "razor-thin / within-margin candidate" to a **dramatically Pareto-dominant**
non-Transformer sequence architecture: at small scale and **parameter/FLOP-matched against a tuned
Transformer**, it must **win on every axis** (real-task LM + latency + constant memory + parameter
efficiency + length-extrapolation) — *and* the win must come from **genuinely novel architecture**,
not a strawman baseline. Better-than-Transformer, **not a Transformer**.

**Owner decisions (2026-06-07):** (1) headline = **Pareto-dominance** (beat TF on every axis);
(2) architecture appetite = **both** (widen margins on the proven quad2-DeltaNet synthesis **and**
introduce new primitives); (3) scale = **stabilize-then-scale** (lock the proof at ≤2M params with
≥10 seeds + powered equivalence/superiority tests, then **one** scale-up confirmation at ~10–50M).

**Non-negotiables (inherited, binding):** no faked metrics; every number from a reproducible script
with seeds + CIs; param/FLOP-matched non-strawman TF baseline; preserve & strengthen the measured
**O(1)/constant-memory** inference differentiator; `step()==forward()` < 1e-4 guard is a hard blocker
before any accuracy number for every mechanism; keep the honest borrowed-vs-new ledger and explicit
failure modes.

## 1. Current state (the razor-thin frontier we are blowing open)
Prizma-Seq today = a Gated-DeltaNet-family mixer (short causal conv → QKV → L2 delta state via the
chunked WY/UT parallel form + a local window head → SwiGLU FFN), whose novel lever is the
**parameter-free quadratic feature map `quad2`** (rectangular state `S ∈ R^{d_h×d_φ}`, fixed seeded
monomial buffers → 0 added params). It clears the §4 bar but **barely**:

| Leg | Current verdict | Why it's thin |
|---|---|---|
| MQAR D=128 | PASS — parity @860K; solves @130K where TF needs ≥461K → ≥3.5× param-eff | coarse 4-point grid |
| Induction / Selective-copy | PASS | **non-discriminating** (Prizma-`none` also solves) |
| **Char-LM (text8)** | PASS within margin — **LOSES** 1.7496 vs 1.7254 BPC (−0.024) | the one real task, and it's behind |
| Inference latency | PASS — crossover only at **n≥32k**, ~1.3–1.5× *slower* below | window head is a full T² SDPA (17.5% FLOPs) |
| FLOP | **2.14× TF FLOPs/token** — **no per-FLOP claim** | quad2 d_φ=256 → 5.5× delta-state FLOPs; FLOP-matched arms optimization-confounded |
| Train speed | ~5× slower/step | sequential-across-chunks delta |
| Memory | constant 17.9 MB ∀n (28–455× less) | already a decisive win — keep it |
| Length-extrap | relative win, absolute only ~0.40 @8× | abs accuracy still poor |

Known integrity debts to repay: init is **not seed-pinned** (`set_seed` after `model_fac` in
`run_cell`); MQAR/induction/selcopy/causal legs ran at a **single shared lr=1e-3** (not per-model
swept); seeds underpowered (n=2–3); FLOP-matched TF arms confounded by missing per-width LR sweep.

## 2. Architecture v2 — the levers (Approach 1: "Precision-Gated Test-Time Memory")
Lead with the genuinely-Prizma novelty; layer the LM and efficiency levers on top. Every lever is
**default-off / config-gated**, has a `step()==forward()` O(1) guard, and a causal ablation.

| # | Lever | Mechanism | Primary win | Risk |
|---|---|---|---|---|
| **A** | **Surprise-gated write/forget** (PC free-energy made causal — the novel core) | use the surprise *magnitude* `‖ε_t‖` (= free-energy gradient norm), not just the input gate: `β_eff = β_t·f(‖ε_t‖)`, and let surprise modulate the forget gate α (release stale memory on a large surprise). Test-time memory in the Titans sense, but **derived** from Prizma's free energy. | novel differentiator; LM + long-context recall | med — ε_t depends on S_{t-1}; must preserve a valid chunk-parallel form (within-chunk approximation + exact `step()`), guarded |
| **B** | **Higher-order DeltaProduct** | k delta steps / token (Householder product state transition), k=2 default | discriminating power (induction/selcopy become discriminating) + LM + state-tracking | med — k× delta FLOP (offset by E) |
| **C** | **Output-gating + per-head state RMSNorm** | per-token output gate `g_t=σ(W_g x_t)`, `o ⊙ g`; GroupNorm/RMSNorm on the state read (RWKV-7/GLA standard) | flips char-LM −0.024 → decisive win | low |
| **D** | **Leaner / structured feature map** | replace fixed random quad monomials @d_φ=256 with a structured/low-rank-then-quadratic map hitting the same D=128 recall at d_φ≈96–128 | FLOP 2.14× → ≤1.0× | med — must keep recall + O(1) |
| **E** | **Banded-window kernel** | replace the full-T² SDPA window with a true sliding-window / block-sparse (FlexAttention) kernel | 17.5%→~0.9% FLOP; latency crossover 32k→~2k | low |
| **F** | **Fused chunked-delta kernel** | A100/Triton (or FLA-style) fused chunked delta to kill the Python per-chunk loop + many small matmuls | train 5× slower → ≤1.5× | med-high |

**Borrowed-vs-new discipline:** A is new (PC-derived surprise-modulated learning rate, tested
causally); B/C/D/E/F are borrowed-and-tuned (DeltaProduct, output-gating/state-norm, Based/Hedgehog
feature maps, sliding-window attention, FLA kernels) — all logged in the ledger; novelty is honestly
scoped to A + the rectangular-delta-state framing.

## 3. The pre-registered "DRAMATIC Pareto-Dominance" bar
Replaces the old "within-margin / parity" bar. Council 1 (honesty) and Council 3 (standards) refine
and **raise** these before each multi-seed run; numbers below are the floor.

| Leg | Now | DRAMATIC target (pre-registered) |
|---|---|---|
| Char-LM (text8, then enwik8) | −0.024 (loses) | **beats matched TF by ≥0.03 BPC**, ≥10 seeds, TOST-superiority p<0.05 |
| Param-efficiency (MQAR D=128) | ≥3.5× (coarse) | **≥4× on a dense grid**, crossover pinned |
| Latency | crossover n≥32k | **faster than TF at all n ≳ 2k** (measured, banded window) |
| FLOP | 2.14× (no claim) | **≤1.0× (per-FLOP parity or win)**, per-width-LR-swept clean baseline |
| Memory | 28–455× less | keep — constant state ∀n |
| Length-extrapolation | rel-win, abs 0.40 @8× | **abs ≥0.70 @4×, ≥0.50 @8×** |
| Train speed | ~5× slower | **≤1.5× slower** (fused kernel) or honestly disclosed |
| Ablations | quad2 ≫ none | **each new primitive causally attributed, ≥10 seeds** |

A leg only counts as "dramatically won" when it clears the target **with powered statistics**
(≥10 seeds, TOST/superiority, solve-rate + median + CI), seed-pinned init, and a fair per-model LR.

## 4. Honest-science scaffolding (strengthened)
- **Reproducibility fix:** seed **before** `model_fac` in every harness path (B1–B3/B6, not just B4).
- **Per-model & per-width LR sweep** (μP-aware) so FLOP-matched arms stop being optimization-confounded
  → a clean per-FLOP claim becomes possible.
- **Powered stats:** ≥10 seeds on headline legs; TOST equivalence + one-sided superiority; report
  solve-rate(>0.9) + median + 95% CI (t, not z); never best-of-noisy-curve (frozen eval set).
- **Param/FLOP ledger** auto-emitted per run; any new gate (C's W_g, etc.) accounted, TF grown in
  lockstep where the addition is a fair architectural comparison.
- **O(1) guard** (`step()==forward()`<1e-4) is a hard blocker per mechanism, including A's
  surprise gate and B's DeltaProduct.
- **Crash-safe** atomic per-experiment JSON checkpoints incl. full LR sweep + rejected LRs + steps-to-plateau.

## 5. The three councils (parallel subagent groups)
| Council | Role | Members (subagent roles) | Loop |
|---|---|---|---|
| **1 — General Council** (the gate) | every major decision is presented; **on rejection, fix the rejection cause before proceeding** | ML-systems engineer · statistics / experimental-design referee · optimization theorist · information theorist · reproducibility/integrity referee · adversarial skeptic | decide → present → approve/reject → (if reject) remedy → re-present |
| **2 — Quant Team** (alpha hunt) | research *how to dramatically surpass* the Transformer; propose aggressive architecture bets, rank by expected-margin-gain per honesty-risk, feed the build queue | quant researcher · architecture designer · kernel/perf specialist · literature scout · risk assessor | loop-until-dry ideation, adversarially filtered |
| **3 — Standards / Landscape Council** (high bar) | "Has the Transformer actually been surpassed? By how much? What is the industry direction?" — position vs real SOTA (Mamba-2, Gated DeltaNet, RWKV-7, Titans…), set the "dramatically surpassed" standard, block strawman victories | landscape analyst · benchmark authority · senior skeptic jury | judgment + standard-raise at each phase boundary |

**Flow:** Quant Team proposes levers → General Council gates each design decision → parallel
subagents build (subagent-driven-development) → Colab A100 validates → Standards Council judges the
result against the real landscape → rejections feed back → **ralph-loop** sustains until the bar is
cleared (no self-written completion phrase; only the owner stops it).

## 6. Never-idle parallelism
While any A100 run is in flight, parallel subagents must always be doing the next useful thing:
building the next experiment's infra, implementing the next lever, writing analysis harnesses,
deepening ablations, drafting the report, running council rounds. No blocking on a single GPU job.
2+ independent tasks ⇒ a single message with multiple parallel `Agent` calls.

## 7. Compute (4× A100 via Colab + normal Chrome)
- Normal Chrome (logged-in account) drives Colab; up to **4 parallel A100 sessions**, each owning a
  workstream (e.g. S1 char-LM/output-gating · S2 surprise-gated MQAR · S3 DeltaProduct+feature-map ·
  S4 latency/FLOP/length-extrap).
- **Calibrate-then-matrix**: short per-model LR/warmup calibration, then the powered matrices.
- Crash-safe atomic JSON checkpoints; results streamed back to `results/`.
- Local Mac (MPS) is for kernel self-tests, the O(1) guard, smoke tests, and infra — not headline numbers.

## 8. Phasing
- **Phase 0 — Foundation & councils:** charter the 3 councils; repo hygiene (working-dir/migrate
  path, seed-pinning, repro); powered-stats + per-width-LR harness; lock the v2 baseline. (Local/cheap.)
- **Phase 1 — Lever R&D (≤2M, multi-seed):** implement & ablate A–F, each gated by Council 1,
  validated on A100. Target: each razor-thin leg flips to a decisive, powered win.
- **Phase 2 — Consolidate the Pareto-dominant model:** combine winning levers; full §4 bar at
  ≥10 seeds + TOST; clean per-FLOP claim; banded-window latency crossover pushed down.
- **Phase 3 — One scale-up confirmation (~10–50M, fuller corpus):** confirm the win holds at scale,
  matched tuned TF; Council 3 judges vs SOTA landscape.
- **Phase 4 — Report + adversarial referee gate:** update the living report; final integrity audit.

## 9. Risks & mitigations
- **Surprise-gate breaks chunk-parallelism** → keep an exact sequential `step()` reference + a guarded
  within-chunk approximation; if the approximation diverges >1e-4, fall back to a two-pass form or
  scope A to inference-time only.
- **DeltaProduct FLOP cost eats the per-FLOP win** → offset with E (banded window) + D (leaner map);
  keep k=2; treat k as a Pareto knob.
- **Char-LM win doesn't materialize at ≤2M** → escalate to the scale-up earlier for the LM leg only,
  Council-3-gated; never claim a win not measured.
- **Optimization-confounding returns** → per-width LR re-sweep is mandatory before any per-FLOP claim;
  excluded arms stay excluded.
- **Overclaiming pressure** (chasing "dramatic") → Council 1 integrity referee + Council 3 standards
  bar are veto gates; the bar in §3 requires powered stats, not point estimates.

## 10. Success definition
Prizma-Seq v2 is done when: the §3 bar is cleared on ≥3 axes decisively (char-LM win + a measured
all-n latency win + per-FLOP parity/win) **with powered statistics and seed-pinned reproducibility**,
the constant-memory and param-efficiency edges are retained, the novel surprise-gate is causally
attributed, **and all three councils sign off** that this is a credible, non-strawman, dramatically
Pareto-dominant non-Transformer architecture in the tested regime — with scale stated honestly.
