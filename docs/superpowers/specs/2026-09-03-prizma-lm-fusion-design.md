# Prizma-LM — Thread-Fusion Design (Owner-Locked Flagship)

**Date:** 2026-09-03 (written during the zero-GPU month; execution begins when GPU budget returns)
**Status:** DESIGN — approved direction (owner decision 2026-09-03: identity = lifelong LM via
thread fusion). This document is the brainstorming-skill architectural output; the
pre-registrations it spawns are separate registry entries.
**Feeds on:** committee/brainstorm_2026-09-03/00_COMMISSION_SYNTHESIS.md (14 reports),
results/citation_battery_2026-09-03/, results/expert_economy_2026-09-03/,
results/analog_probe_2026-09-03/, docs/preregistry/ (PR-01…05).

---

## 1. Goal and non-goals

**Goal.** One model that keeps learning from an unlabeled, boundary-free token stream —
Prizma-Seq's predictive-coding delta-state mixer as the backbone, Prizma-CL's vigilance-routed
predictive-coding experts as the adaptive tissue — demonstrated on the smallest honest
lifelong-LM bar the repo can pre-register.

**Why this is the flagship (commission consensus + owner decision).** Transformers freeze at
deployment; their KV-cache never learns. Continual adaptation is the one capability where the
incumbent architecture *structurally* does not compete, and it is where Prizma's two threads
are complementary by construction: the carried state S is a fast, labile store (hippocampal
analogy), expert weights are slow, consolidating tissue (cortical analogy), and the vigilance
router is the novelty signal that decides which tissue forms.

**Non-goals (this design cycle).** No large-scale LM parity claim; no "attention replacement"
narrative (retired — GDN is production in Qwen3-Next hybrids); no ambiguity-regime promises
(proven impossible for single-head learners; the repo's own honesty line); no claim until a
pre-registered bar passes. The words "rival / inevitable" stay banned until BAR-0..6 pass
(owner decision).

## 2. What this month's zero-GPU work already established (design inputs)

1. **E1 headline survived the citation-bar audit** (citation_battery, multi-epoch, 10 seeds):
   boundary-free regularizers do not close the gap — MAS 0.415±0.019, online-EWC 0.429±0.020
   vs Prizma(DFA) 0.834±0.015 / FGT 0.000. CIs disjoint by ~0.39. The routing mechanism's
   advantage is real in the regime it claims.
2. **The single-pass crack is acquisition, not retention** (citation_battery, exploratory):
   everything compresses to 0.41–0.55 ACC in single-pass; Prizma's FGT (0.185) already matches
   replay's (0.199) — the deficit is *learning speed* (LA 0.627 vs replay 0.708). The fusion's
   #1 mechanism problem is **faster acquisition under one exposure**, not less forgetting.
3. **Interleaved streams defeat recruitment by two mechanisms** (expert_economy, exploratory):
   (a) mixed batches make batch-mean surprise stationary → vigilance is blind; (b) round-robin
   pure batches freeze immature experts that become catch-all recognizers. Any lifelong-LM
   routing needs a **freeze-immaturity veto** and a **stationarity-robust surprise statistic**.
4. **Economy is quantified**: 4,304 floats/domain (2,768 trainable + 1,536 FA fixed) at
   shipped shapes; growth is 1 recruit/domain, and the recruited pool at E1's K=5 is ~1.04× the
   backprop baseline. Eviction with M_max=K fires **zero** times on the home stream — the cap
   is inert where the model is honest and protective where the stream drifts forever.
5. **Delta writes are analog-robust (directional, exploratory)**: at 4-bit state, delta
   retention 0.724 vs additive 0.621; at write-noise σ=0.05, 0.974 vs 0.888 — plus a real bug
   fix (step() ignored additive mode; 38-pt deployment gap closed). Deployment tailwind for
   always-on edge, pending PR-04.
6. **Theory floor fixed**: the Gershgorin bound never explained D=128; the crosstalk-spectrum
   law (PR-02) will predict the D-frontier before the GPU runs it — the fusion's capacity
   planning (how many experts × what d_φ) should use N* = min(1+ε²/σ₂², d_φ) once fitted.

## 3. Architecture

### 3.1 The column (unit of the fused model)

One **column** = the Prizma-CL expert, extended with a share of the sequence state:

- Recognizer: predictive-coding autoencoder (Wenc, Wdec) with fixed random feedback (FA) —
  unchanged from src/prizma.py; routes by reconstruction surprise with per-expert precision
  (μ, σ EMA). All-local learning; no weight transport.
- Head: classifier or (at LM scale) the token-prediction projection served by this column.
- **State share (new):** a per-column slice of the Prizma-Seq mixer's carried state S —
  writes to a column's slice are gated by that column's current precision (the column that
  recognizes the context owns the fast memory for it). Read path is unchanged (pre-write,
  content-addressed). This is the 01-P5 "unified column" mapping: state = labile store,
  weights = stable tissue, vigilance = allocation.

### 3.2 The backbone

Unchanged Prizma-Seq mixer (chunked delta kernel, quad2 feature map, short conv + local
window head) with ONE structural addition: the FFN block becomes a **vigilance-routed expert
mixture** (06-P2). Two properties make this mixture unlike MoE-as-shipped:

- **No auxiliary load-balancing loss.** Balance is emergent: recruitment only fires on
  unrecognized input, and each expert's vigilance θ_m adapts to its own domain noise floor
  (the shipped ART mechanism). Transformers need aux losses because their experts have no
  novelty semantics; Prizma's do.
- **Unbounded → bounded growth.** M_max with recruit-by-eviction (expert_economy P3 spec),
  plus merge of vigilance-compatible frozen experts. Inert on distinguishable streams
  (measured: 0 evictions), protective on open-ended drift.

Token-level routing uses the **write-error energy** ‖ε_t‖² already computed inside the mixer's
delta step — no extra forward pass. Routing is pointwise → kernel-feasibility class T0/T1 in
report 09's grammar (no chunk-parallel violation).

### 3.3 Acquisition fix (the single-pass crack)

Prizma under one exposure under-learns (LA 0.627). Three candidate mechanisms, pre-registered
separately, cheapest first — all CPU/AAPU-testable at toy scale:

- **A. Settling steps (already shipped as a knob, `n_settle_steps`)** — iterate the
  recognition loop before the write; cortical FF/FB settling (01-P3). K settles = K gradient
  steps on the same per-token free energy. Zero new theory; measure whether K=2–4 closes LA.
- **B. Per-column learning-rate metaplasticity** (04's ω): young experts train hot, old experts
  cold — attacks the immature-floor problem at the same time (freeze-immaturity veto: an
  expert with n_seen below a floor cannot be frozen nor become a router catch-all).
- **C. Surprise-normalized writes** — RESULT-GATED on PR-01: if the powered ablation retires
  the surprise gate, this line is dead and B carries the weight.

### 3.4 Consolidation (the CLS bridge — 02-P1)

When a column's surprise-floor EMA indicates its domain has passed (the shipped freeze
signal), run **replay-before-freeze**: generate pseudo-items from the column's carried state
slice, train the slow weights on them, then decay the slice. Claim unlocked at LM scale:
constant memory *and* retained knowledge — the KV-cache cannot follow (its content is the
memory; Prizma's memory migrates from state to weights). Pre-registered retention bar below.

## 4. The smallest falsifiable demonstration (PR-LM-1, draft pre-registration)

**Task.** Continual char-LM over two drifting corpora, no task labels, no boundaries, no
replay buffer: corpus A → corpus B (start: text8 → held-out enwik8 slice; B4 lesson: retain
ALL raw artifacts, both corpora committed in results/). Model sees the concatenated stream;
Prizma's router must discover the shift itself.

**Arms.** (1) Prizma-LM as designed; (2) frozen-checkpoint control (train A, freeze, eval B —
upper-bounds retention, lower-bounds adaptation); (3) memory-matched sliding-window Transformer
(replay window = Prizma's total recruited state+expert bytes — the honest control the
commission demanded for every "beats the cache" claim); (4) Prizma-LM minus routing
(ablation — the causal lever, mirroring E1's noRoute).

**Bars (exact).** n≥5 seeds; on corpus A re-eval after corpus B: Prizma-LM FGT ≤ 0.05 BPC
AND within +0.10 BPC of its own pre-shift eval; arm (3) may use its window but must stay
within the same memory budget; Prizma-LM beats arm (2) on corpus B by ≥0.10 BPC (adaptation
happens) with Welch CI; ablation (4) must show a routing-attributable retention gap (mirrors
E1). Honest default: if arm (3) matches Prizma-LM at matched memory, the claim downgrades to
"competitive at constant memory" — pre-committed.

**Budget estimate.** ~80–120 A100-h (2-layer d=128–256 char-LM × 4 arms × 5 seeds), only
after the GPU-return prerequisite ladder (§6).

## 5. Risks and mitigations (commission red-team, surviving set)

- **Interleaved collapse** (measured ACC ~0.58 shipped; recruitment-blindness measured this
  month) → §3.3-B mechanisms are PREREQUISITES, not options; bar: the fixed router must clear
  BAR-6 (interleaved ACC ≥0.70) at toy scale before PR-LM-1 spends GPU.
- **FLOPs** (experts widen the model; the 5× step gap is unhealed) → sparse top-k routing
  (06-P1's f≈6–12% arithmetic → ~1.0× TF FLOPs) is a design requirement, not an option;
  routed experts activate 1–2 columns/token.
- **Capacity interplay** → d_φ and column count planned against the fitted N* law (PR-02);
  if the law is refuted at its D-frontier, fall back to empirical capacity probing and say so.
- **Brand risk** ("active inference LM" is an overclaim today — report 05 scorecard) → public
  wording: "free-energy-derived, locally-learnable, continual" — precision/FE terminology only
  where the math is literal.
- **Analog/edge** is a downstream narrative fed by PR-04, not part of PR-LM-1.

## 6. Execution order when GPU budget returns (locked by owner decisions)

1. Tier-0 repairs (commission C6, ≈63 A100-h): clean recall gate, init-fix reruns, B4 closure,
   first GLA/Mamba-2 landscape run.
2. PR-01 surprise ablation (frozen protocol, ~10–15 A100-h) — gates §3.3-C.
3. Kernel decision session (report 08-P1: Triton backward verify-or-port; bar ≤1.5× TF step).
4. BAR-6 interleaved-router check at toy scale (CPU/APU-feasible) with §3.3 mechanisms.
5. PR-LM-1 continual char-LM (§4) — the flagship bar.
6. PR-02 D-frontier adjudication, PR-04 analog rerun, PR-03 citation bar, PR-05 bounded-M —
   parallelizable small runs.

## 7. What this design deliberately does NOT promise

No LM-scale parity, no ambiguity-regime gains, no attention-replacement headline, no claims
before their pre-registrations pass, no device-validated energy numbers (arithmetic bands
only, per report 12-H5), and no surprise-gated-write mechanism unless PR-01 survives.

---

## Addendum 2026-09-05 — PR-06 outcome: PR-LM-1 BLOCKED on the router (granularity, not thresholds)

PR-2026-09-03-06 executed its frozen protocol verbatim and fired the pre-committed
INSUFFICIENT branch: bar 1 (E1 no-regression) PASSED (0.842 / FGT 0.000), bar 2 FAILED on
both interleaved streams (+fixes arm 0.597/0.592, CI uppers 0.646/0.643, both short of 0.70).
Attribution: route_stat=sample_top carries the entire effect (+0.058 round-robin);
hot_young ≈ 0; dynamic_vigilance harms standalone; the freeze veto is ACC-neutral.
Mechanistic ledger (n=5): the +fixes arm routes ONE expert onto 100% of the stream — the
model lands exactly on the monolithic-learner ceiling (~0.60). Conclusion: §3.3's lever set
(veto, dynamic vigilance, metaplasticity, settling, route statistic) repairs threshold
sensitivity but cannot recover block-stream specialization, because `train_batch` applies the
whole batch's delta to the single active expert. The binding constraint is **training
granularity**, not routing sensitivity.

Consequences for this spec:
- §6 step 4 (BAR-6 check) is done — outcome NEGATIVE; PR-LM-1 stays blocked until the router
  gains per-sample / expert-stream training granularity.
- §3.3 candidate set is superseded by three successor mechanism classes, each requiring its
  own pre-registration: (G1) per-sample routed delta steps (train each sample on the expert
  that recognizes IT, not the batch's majority); (G2) recruitment-session windows (contiguous
  sample runs routed as a unit before any consolidation); (G3) replay-before-freeze at CL
  scale (02-P1, pulled forward from "flagship body" to "router prerequisite").
- Exploratory lane before any new registration: an n=2 granularity probe (G1 vs G2 toy
  variants) to freeze PR-07's exact mechanism — the PR-06 lesson: freeze the STATISTIC, and
  check the training ledger, not just the accuracy.

No LM-scale parity, no ambiguity-regime gains, no attention-replacement headline, no claims
before their pre-registrations pass, no device-validated energy numbers (arithmetic bands
only, per report 12-H5), and no surprise-gated-write mechanism unless PR-01 survives.
