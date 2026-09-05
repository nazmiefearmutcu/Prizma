# Brainstorm — Sequence-Model Competitive Landscape lens: where Prizma-Seq must win to be a credible, inevitable rival

**Commission:** Prizma → (1) inevitable Transformer rival + (2) maximally brain-like
**Author:** Sequence-Model Competitive Landscape Analyst (attention alternatives: Gated DeltaNet,
Mamba/Mamba-2, GLA, RWKV-7, DeltaNet/DeltaProduct, TTT, Titans, Based, Hedgehog, RetNet,
hybrid attention+SSM stacks; positioning strategy for architecture papers)
**Date:** 2026-09-03
**Classification (per brainstorming skill):** **Architectural-class research-direction brainstorm.**
HARD-GATE honored: **no code, no implementation** — this is a strategy/ideas document only. The
skill's interactive clarifying questions cannot reach the owner from inside a committee, so all
direction-setting questions are collected in **QUESTIONS FOR THE OWNER** at the end.

Sources read before writing: `README.md` (honest-scope + quarantine sections),
`docs/PRIZMA_SEQ_REPORT.md`, `docs/HANDOFF.md`, `seq/prizma_seq.py` (config/levers),
`seq/landscape.py` (module docstring + arm configs), `seq/benchmark_results.md`. Web verification
(2025-2026): Gated DeltaNet in production hybrids — [Qwen3-Next blog](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list),
[vLLM on Qwen3-Next](https://vllm.ai/blog/2025-09-11-qwen3-next), [Raschka's hybrid-attention gallery](https://sebastianraschka.com/llm-architecture-gallery/hybrid-attention/)
(48 layers = 12× a 3:1 GDN:gated-attention pattern in Qwen3-Next-80B-A3B; Qwen3.5 and
Qwen3-Coder-Next continue the lineage; Moonshot's Kimi Linear swaps in Kimi Delta Attention).
Claims from my training knowledge rather than a live source are marked *(knowledge)* — no numbers
are invented; qualitative claims only.

Borrowed-vs-new discipline: every proposal marks **[B]** borrowed / **[N]** new / **[B→N]**
borrowed-component-with-new-framing.

---

## 0. Executive frame: the field moved. "Replace attention" is over; a new slot opened.

The single most important landscape fact for this commission:

**The delta-rule linear-attention family won — and in winning became a commodity.** Gated DeltaNet
is not a 2024 paper anymore; it is the linear-attention layer inside Qwen3-Next (80B-A3B, 3:1
hybrid), Qwen3.5, Qwen3-Coder-Next, and (as the closely-related KDA variant) Moonshot's Kimi
Linear *(knowledge + sources above)*. Jamba, Samba, Falcon-H1, Granite 4 (Mamba-2 hybrids)
*(knowledge)* fill the same slot. The community consensus answer of 2025-2026 is **"mostly-delta /
mostly-SSM stack + a small fraction of full-attention layers"** — the pure-Transformer monopoly is
already broken in production.

Three consequences for Prizma:

1. **"Can a linear mixer stand in for attention?" is no longer a paper.** It is a solved,
   industrialized question with a library (FLA *(knowledge)*) full of variants. Prizma's §4
   diagnostics PASS (MQAR param-efficiency, induction, selective-copy, constant memory) are
   **table stakes** — necessary to enter the arena, worth zero novelty credit. The repo already
   says this honestly ("the DeltaNet/Mamba family ARE the field's accepted efficient
   attention-replacements"); the strategy must take the further step.
2. **A new, enormous target exists that did not exist in 2024: the deployed hybrid slot.** The
   world's most-produced new architecture has a *GDN-shaped socket* at a 3:1 ratio. Prizma-Seq is
   GDN-shaped by construction (byte-identical FFN/norms/embeddings; chunk-parallel delta kernel;
   O(1) `step()==forward` guard green). "Replace attention" is a war against a finished war;
   **"be the drop-in upgrade for the hybrid slot"** is the live one.
3. **What remains genuinely contested (the vacant thrones):**
   - **Pure-linear recall at scale** (removing even the 1:4 full-attention layers) — unsolved;
     MQAR saturates in diagnostics but recall-intensive downstream tasks degrade without full
     attention layers *(knowledge — hybrid papers' own ablations)*.
   - **Theory/unification.** The family is accreting variants (GDN scalar gates, RWKV-7
     vector-valued in-context rates, KDA element-wise gates, DeltaProduct k-steps, Titans
     momentum+surprise memory) with no accepted account of *why* they work or how they relate.
     The DeltaProduct line *(knowledge)* takes a step here; **nobody owns the free-energy /
     optimizer view** — and that is exactly Prizma's identity.
   - **Training cost.** Delta kernels are still slower than attention's SDPA stack in many
     regimes; a credible kernel story is a differentiator, not a footnote.
   - **Local/online learning.** Every member of the family still trains by BPTT-through-chunks.
     A delta-state model with a *local-learning* (BPTT-free or reduced) training mode would be
     unique in the landscape — and it is Prizma's birthright from the other thread.
   - **Brain grounding.** The entire production family is neuro-vibes-free. The fast-weights
     lineage (Ba; Schlag et al. fast-weight programmers; Schmidhuber 1987 delta rule; Hinton's
     40-year-old conjecture *(knowledge)*) is folklore, not a falsifiable program. **The slot
     "the brain-like member of the delta-rule family with LM parity" is completely vacant.**

**The wedge, stated in one sentence:** Prizma cannot and should not try to be a better-commodity
GDN; it can be the **free-energy-derived, parameter-free-capacity, locally-learnable member of the
family that won** — with the predictive-coding identity as its theory, quad2 as its capacity lever,
and brain-alignment as its falsifiable second axis.

---

## 1. Where exactly must Prizma win? (two bars, explicitly defined)

### Bar A — "Credible rival" (the community's minimum, currently unmet)
To be *cited* as a rival rather than a curiosity, Prizma must, at **n≥10 powered seeds**, Holm-
corrected, LR-swept:
1. Show **TOST parity** (not just win-on-point-estimates) with GDN-family and Mamba-2 arms on the
   standard diagnostics — the quarantined recall gate must be re-run clean and this time *pass*
   (or honestly fail and the claim rescoped). Today this leg is a NON-result.
2. Close **B4 properly**: both corpora, n≥5 (the pre-registration), with the init-seed bug fixed.
   A "PARTIAL with a failed corpus" on the *simplest* LM axis is the single most quotable negative.
3. Beat or match **GLA and Mamba-2** (not only the TF) — the baseline the field actually compares
   against. The 4-arm harness exists and has never been run; **this is the repo's own credibility
   debt**.
4. Publish a **FLOP/throughput ledger vs an optimized kernel baseline** (even if unfavorable). The
   current 2.14× as-coded FLOP ratio and ~5×/step training disadvantage are honest but fatal if
   never attacked; the banded-window ideal (1.78×) shows the headroom.

### Bar B — "Inevitable rival" (what makes the community *adopt* rather than cite)
1. **Hybrid drop-in win:** replace the GDN arm inside a Samba/Qwen3-Next-style 3:1 stack at
   50M→150M params and beat matched GDN-hybrid and Mamba-2-hybrid on pretraining loss, a few-shot
   battery, and long-context recall — at comparable kernel speed. This targets the socket the
   industry actually buys. (`seq/hybrid.py` exists; `seq/scaling_analysis.py` has the 50M/100M
   param tables.)
2. **Own the unification:** one objective (per-token free energy F_t(S) = ½‖v_t − S k_t‖²), one
   update (one gradient step), with GDN / RWKV-7 / KDA / DeltaProduct / Titans recovered as
   (preconditioner × gate granularity × inner-step count × momentum) choices — plus a *measurement*
   that tests the identity's non-trivial content (see P5). Nobody else has an identity that
   predicts which lever helps which task.
3. **Own the brain axis with pre-registered metrics:** local-learning parity (quantified tax),
   perplexity-matched neural-alignment evaluation, and a two-timescale (CLS) integration with the
   original Prizma continual thread. Falsifiable or silent — brain-vibes without metrics is the
   fastest route to crank status in this subfield.
4. **A capacity law:** the Gershgorin crosstalk bound in `docs/quad2_theoretical_convergence.md`
   (N < 1 + 1/cross(φ); quad2 N<14 vs linear N<8) is a seed of a *feature-map capacity law*. A law
   that predicts MQAR D* from a feature map's crosstalk before running would be the field's first
   predictive recall-capacity rule — genuinely new and cheap to validate.

**Where Prizma must NOT fight:** raw MQAR victory laps (diagnostics are saturated and gameable —
the community knows it), "attention fails" framing (P1 already closed that honestly — attention
solves D=128 given capacity; the differentiators are efficiency and grounding), and per-FLOP
dramatics until a real kernel exists.

---

## 2. The decisive experiment ladder (what the community will actually demand)

Each rung has a pre-registered gate; a rung that fails rescopes everything above it. Arms are
always **param-matched, FLOP-disclosed, LR-swept per arm, seed-pinned, Holm-corrected** (the house
protocol).

**R0 — Integrity rung (do first; ~28–60 A100-hours, harness exists):**
- Re-run the quarantined n=10 recall gate on the fixed (seed, config-fingerprint) resume path.
- Fix the init-before-`set_seed` pinning bug and re-run B1–B3/B6 as disclosed as open work.
- Close B4: text8 + tiny-shakespeare (diagnose the −0.09 overfitting recipe failure first;
  dropout lever now exists), n≥5 per corpus.
- **Gate:** TOST parity vs TF on ≥2 of 3 recall legs at ±0.05; B4 PASS as pre-registered. If the
  clean rerun fails parity, the entire "credible rival" program downgrades to "param-efficiency
  niche" — and the committee must say so.

**R1 — Landscape rung (~60–120 A100-hours; `seq/landscape.py` exists, never run):**
- The built 4 arms (TF / Prizma / GLA / Mamba-2) + **two nearly-free arms from existing levers**:
  Prizma-`inctx_lr` (an RWKV-7-style per-channel rate) and Prizma-`n_delta≥2` (DeltaProduct-style
  multi-step inner loop). This turns the 4-arm landscape into a 6-arm one and makes Prizma's code
  the *substrate* on which the family's axes are compared — a positioning coup that costs one
  config each.
- Recall diagnostics + `--charlm` leg (2 corpora, n≥10) both exist in the harness.
- **Gate:** Prizma Pareto-competitive (no WORSE verdict, ≥1 BEATS) vs GLA and Mamba-2 at matched
  params.

**R2 — Feature-map shootout rung (~40 A100-hours; partially exists):**
- quad2 vs quad2_lowrank vs learned-projection φ (DeltaNet-style) vs Hedgehog-style exp map vs
  Based-style Taylor map vs rand_linear control, all at matched params AND matched FLOPs, on the
  MQAR D-frontier (16→256, the deferred P3) + char-LM.
- Also validate the crosstalk→D* capacity law's predictions blind (predict D* from measured
  crosstalk per map, then test).
- **Gate:** param-free quad2 within margin of the best *learned* map at matched FLOPs → "capacity
  for free" is earned; else honest downgrade to the low-rank dial + the law.

**R3 — Small-LM pretraining rung (~150–400 A100-hours; apparatus exists, never trained):**
- 50M and 150M param models, 5–10B tokens FineWeb-Edu subset *(knowledge; Pythia-70M/160M-style
  regime)*. Arms: TF, GDN (FLA), Mamba-2, **Prizma pure**, **Prizma-hybrid 3:1** (the field's
  deployment shape). Pre-register: loss-parity within a stated tolerance at matched params;
  FLOP ledger per arm; tokenizer/eval fixed.
- Downstream few-shot via `seq/downstream.py` (HellaSwag/ARC-easy/MMLU-mini/GSM8k-closed-form)
  + long-context RULER/NIAH at 32k–128k + measured decode throughput vs KV-cache.
- **Gate:** Prizma-hybrid ≤ GDN-hybrid loss + beats it on ≥1 downstream axis, no axis lost by
  more than the pre-registered margin. This is the make-or-break rung for "inevitable".

**R4 — Kernel rung (parallel, engineering):**
- Triton chunked-delta kernel at FLA-level throughput (the deferred `delta_triton.py` path,
  developed+verified ON the GPU, never shipped blind — the repo's own rule) + banded window head.
- **Gate:** ≤1.15× attention-stack step time at 150M, training no longer the confound. Without
  this rung, R3 results are discountable ("slower model loses on loss-per-compute").

**R5 — Moonshot rung (see §6).**

Compute honesty: R0–R2 are squarely in the owner's demonstrated Colab-credit regime. R3 at 150M ×
6 arms × 5–10B tokens is 10–30× larger than anything the repo has run; it needs a budget decision
(Question Q1) or a collaborator with a cluster.

---

## 3. quad2 vs Based/Hedgehog: is "the framing is the novelty" enough?

The repo's honest ledger says quad2's kernel is borrowed (Based/Hedgehog family) and the novelty
is the rectangular-delta-state framing. My landscape verdict: **framing alone is not enough for a
paper, but framing + three concrete upgrades is enough for a section that no competitor has.**

1. **The zero-params claim is real and unowned.** Learned feature maps (DeltaNet projections) and
   closed-form maps (Hedgehog exp, Based Taylor) all *either* add parameters *or* are not framed as
   a capacity dial. quad2 = seeded monomial **buffers**: capacity up, params +0, O(1) intact, with
   a low-rank variant (d_φ=137) as a FLOP dial. "Capacity-per-parameter as a free axis" is a Pareto
   point nobody else states. R2 must earn it (matched-FLOP comparison, not just matched-param).
2. **The capacity law is the upgrade.** If crosstalk (measured, 0.076 for quad2) predicts the MQAR
   D* across {none, rand_linear, quad2_lowrank, quad2, learned-φ} blind, the framing upgrades from
   "novel packaging" to **"the first predictive recall-capacity rule for feature-mapped delta
   states"** — and it rhymes with 60 years of associative-memory capacity theory (Hopfield/Willshaw
   *(knowledge)*), which doubles as the hippocampus-CA3 brain link (see 07-adjacent neuro report).
3. **What would upgrade the kernel itself (optional, ranked):** (a) learned-φ-in-addition arm so
   quad2 can *win* rather than tie (if the fixed map matches learned maps, parsimony wins; if it
   loses, the law explains the gap); (b) a higher-order or data-adaptive monomial selection rule
   (new kernel territory — only if R2 shows the fixed monomials are the bottleneck); (c) port quad2
   as an FLA-library feature map so *other* groups can adopt the lever — adoption, not novelty, is
   what makes a lever "real" to this community.

If R2 shows quad2 merely ties learned maps, the honest claim becomes "param-free at parity" — a
weaker but real efficiency claim. If it loses at matched FLOPs, kill the lever claim and keep the
law. Pre-register both branches now.

---

## 4. The narrative pivot: from "replace attention" to what, exactly?

Recommended repositioning, in one paragraph:

> *Prizma-Seq is not an attempt to replace attention. It is the **free-energy member of the
> delta-rule family that production has already adopted**: its state update is exactly one gradient
> step on a per-token free energy (predictive coding); its capacity lever is parameter-free and
> governed by a measurable crosstalk law; it drops into the 3:1 hybrid stack the industry deploys;
> and — uniquely in the family — its learning rule is local, opening a BPTT-free training mode and
> a falsifiable brain-alignment program.*

Why this pivot is forced by the landscape: (a) "replace attention" is unwinnable and the field
stopped contesting it (attention layers persist in every deployed hybrid, at 1:4); (b) every
**engineering** claim in the current narrative (O(1) memory, long-context decode, diagnostics) is
commodity; (c) the only claims in the repo that are *not* commodity — the free-energy derivation,
param-free capacity, the local-learning lineage, the continual thread — are exactly the ones the
pivot foregrounds; (d) the pivot *keeps* the repo's honesty culture intact: every pivot claim maps
to a pre-registered rung above.

What the pivot does NOT license: dropping the engineering bar. The brain/local-learning story is
only credible **on top of** demonstrated Pareto competitiveness (Bar A). "Brain-like but slower and
worse" is a workshop poster; "brain-like AND matches the family at parity with a law and a drop-in
win" is an architecture paper with a brand.

Brand note: the two threads (continual local learner + sequence mixer) stop being separate projects
under this pivot and become **one complementary-learning-systems program** — the carried state is
the fast store, the (locally-trained) weights are the slow store. That is the "inevitability" story
at the program level: not one paper, but the only research program in the landscape that owns
theory + engineering + brain grounding simultaneously.

---

## 5. Proposals (ranked; B = borrowed, N = new, B→N = borrowed component, new framing)

### P1. Clean-Slate Powered Landscape — run the debt before writing a word
**WHAT:** R0+R1 executed exactly as specified in §2: clean quarantined-gate rerun, init-pinning
fix, B4 closed on both corpora, then the built 4-arm `landscape.py` (recall + `--charlm`) plus the
two free arms (`inctx_lr`, `n_delta=2`).
**WHY:** Every other proposal borrows this repo's credibility; the quarantine and B4 deviations are
the first thing any referee finds (the repo itself documents them). The harness is *already
written and tested* — the marginal cost is GPU-hours, not engineering. Also settles, at power,
whether the delta family arms (GLA/Mamba-2) beat Prizma at matched params — the fact the whole
program currently rests on not knowing.
**HOW:** Colab A100 campaign in resumable cells (the infrastructure pattern exists in
`PRIZMA_D128_GPU.ipynb`); pre-register gates (R0/R1 above) in a committed STAGES file before the
first run; stream crash-safe JSONs (already the harness behavior).
**FALSIFIABLE TEST:** The gates themselves — TOST parity ±0.05 on ≥2 recall legs; B4 PASS as
pre-registered; no WORSE Holm verdict vs GLA/Mamba-2. Failure ⇒ public rescoping to
"param-efficiency + memory niche".
**RISKS:** Clean numbers may be worse than the quarantined ones (the direction was conservative,
but seed noise is real); tiny-shakespeare may fail again even with the dropout lever ⇒ B4 stays
PARTIAL and the paper's LM leg rests on one corpus + diagnostics.
**IMPACT 9 / FEASIBILITY 9** (harness exists; ~60–120 A100-h; highest leverage per hour in the
repo). **[B→N]** — borrowed harness, new powered numbers and verdicts.

### P2. Feature-Map Shootout + the crosstalk capacity law
**WHAT:** R2: {none, rand_linear, quad2_lowrank, quad2, learned-φ, Hedgehog-exp, Based-Taylor} at
matched params *and* matched FLOPs on the MQAR D-frontier + char-LM; blind-test the
crosstalk→D* prediction per map; publish the law with its failures.
**WHY:** Converts quad2 from "borrowed kernel, new framing" into either (a) an owned Pareto point
("capacity for free at matched FLOPs") or (b) a predictive law — and possibly both. The law is the
only item in the repo that could become *other groups' tool*, which is how architecture ideas
become inevitable (adoption). Rhymes with classical associative-memory capacity theory → the
brain bridge is a corollary, not a vibe.
**HOW:** `flop_ledger.py` already computes per-config FLOPs; `seq/tasks.py` has the D-frontier
machinery (the deferred P3); crosstalk measurable analytically per map. Pre-register both outcome
branches (§3) before running.
**FALSIFIABLE TEST:** quad2 ≥ learned-φ − margin at matched FLOPs on the D-frontier; measured
crosstalk predicts empirical D* within a pre-registered band across all maps. Either branch's
failure is a published negative.
**RISKS:** Learned φ may simply win at matched FLOPs (killing the lever claim); the law may fit
quad2 but not the learned maps (scope it honestly); diagnostics saturation means the D-frontier
must go to D=256+ to discriminate.
**IMPACT 8 / FEASIBILITY 8.** **[B→N]** — maps borrowed (Based/Hedgehog/DeltaNet), the
param-free-buffers instantiation, the law, and the blind-test protocol are new.

### P3. The Free-Energy Unification — one objective for the delta-rule family, with a measurement
**WHAT:** A theory+analysis paper (or the theory section of the moonshot): derive GDN (scalar
gate), RWKV-7 (vector per-channel rate = `inctx_lr`), KDA (element-wise gates) *(knowledge)*,
DeltaProduct (k steps = `n_delta`), Titans (momentum + adaptive rate) *(knowledge)* as
(preconditioner × gate granularity × inner steps × momentum) choices on one-step free-energy
descent on F_t(S); then **measure** whether trained models' gates implement precision: correlate
learned β/α/η with token surprisal in R1's trained char-LMs and any R3 model.
**WHY:** The family's unification slot is vacant from the optimizer view; the DeltaProduct line
approaches it algebraically *(knowledge)* but nobody grounds the levers in an objective with a
falsifiable precision-prediction. This is also the only way to rescue the surprise/precision story
from its current honest low point (the only evidence points the wrong way) — either the gates
track surprisal (the PC identity has non-trivial content) or they don't (the identity is algebra;
say so and stop charging novelty for it).
**HOW:** Zero training cost if built on P1/R1 artifacts (gates + surprisal extractable from
checkpoints). The theory must be written against the actual family papers, distinguishing what
DeltaProduct already unified — novelty = the free-energy/optimizer grounding + the measurement.
**FALSIFIABLE TEST:** Gate-surprisal correlation significantly positive and *causally load-bearing*
(permute/downweight the gate at eval → recall degradation tracks correlation strength); families
predicted to help recall (higher effective rank preconditioners) actually help.
**RISKS:** Gates may decorrelate from surprisal once trained (the smoke ablation's direction
suggests caution); the unification may overlap DeltaProduct's algebra; theory sections in
architecture papers attract the harshest referees.
**IMPACT 8 / FEASIBILITY 7.** **[B→N]** — family members borrowed (they are other people's
models); the objective-grounding, the axes, and the precision measurement are new.

### P4. Kernel Rung — FLA-grade Triton delta + banded window
**WHAT:** R4: hand-written Triton chunked-delta WY/UT kernel + banded window head, developed and
verified on-GPU (never shipped blind — house rule), target ≤1.15× attention-stack step time at
150M; upstream to the FLA library if quality allows.
**WHY:** The 5×/step training tax and 2.14× FLOP ratio make every other number discountable; in
2025-2026 an architecture paper without a kernel story is dead on arrival at the production-hybrid
audience. FLA upstreaming is also the cheapest possible adoption channel — the wedge in §0's
sense.
**HOW:** `delta_fused.py` (torch.compile) exists as the bridge; `delta_triton.py` is deferred
scaffolding; verify by equivalence tests (chunk==recurrent, step==forward) at every tile-size
change.
**FALSIFIABLE TEST:** Throughput table vs eager and vs FLA's GDN kernel at fixed shapes; bit-level
equivalence guards; R3 rerun with the kernel to show conclusions unchanged.
**RISKS:** Heavy engineering with no scientific novelty credit; Triton-on-nonstandard-hardware
pain; may land *after* the field's kernels improve again (moving target).
**IMPACT 8 (enabler) / FEASIBILITY 5.** **[B]** (kernel family standard) with N tile-level work.

### P5. Hybrid Drop-In — beat GDN inside the deployment shape
**WHAT:** R3's headline arm: a Samba/Qwen3-Next-style 3:1 Prizma-hybrid vs GDN-hybrid and
Mamba-2-hybrid at 50M and 150M, same data/eval; the claim tested is "Prizma upgrades the deployed
slot", not "Prizma replaces attention".
**WHY:** It targets the one socket the industry demonstrably buys (§0); hybrid arms also *rescue
pure-Prizma's known weaknesses* (recall-intensive tasks) the way every production system already
does — testing Prizma where it would actually run. A win here is the single most legible result
possible for "inevitable".
**HOW:** `seq/hybrid.py` exists (Samba-style); pre-register the loss/downstream/long-context gates
of §2-R3; include the pure arms so the hybrid's value-add is attributable; keep the FLOP ledger
honest per arm.
**FALSIFIABLE TEST:** Pre-registered: hybrid-Prizma ≤ hybrid-GDN loss at matched params (stated
tolerance), ≥1 downstream win, no axis lost beyond margin, decode-throughput advantage retained.
Failure ⇒ "not a drop-in upgrade" is published and the pure-model story carries on alone.
**RISKS:** Compute (largest item in the program); GDN-hybrid is a strong, well-tuned adversary;
Prizma's training-tax may make the hybrid lose on wall-clock even if it wins on loss (report both;
that's why P4 exists).
**IMPACT 9 / FEASIBILITY 5.** **[B→N]** — hybrid pattern borrowed (Samba/Qwen3-Next shape);
Prizma-in-the-slot comparison is new.

### P6. Small-LM Pretraining Ladder — the parity paper
**WHAT:** R3 run to completion (both hybrid and pure arms, 50M→150M, 5–10B tokens) as the field's
required parity evidence, downstream few-shot + long-context included.
**WHY:** At ≤1.4M params the repo's claims stop at "candidate"; the community's bar for
"credible rival" starts around the Pythia-70M/160M regime with pretraining loss + few-shot. This
rung is where the current apparatus (`downstream.py`, `scaling_analysis.py` param tables) finally
produces results instead of apparatus.
**HOW:** Only after P1 (clean diagnostics) and ideally after P4 (kernel). Pre-register everything;
commit raw JSONs per house rules; use one public data order; publish the failures too.
**FALSIFIABLE TEST:** The pre-registered parity band per size and per benchmark; the honest
alternative outcome is a measured scaling gap (which itself is a publishable, citable negative —
cf. how the field treats small-scale Mamba-2-vs-TF gaps *(knowledge)*).
**RISKS:** Biggest compute item; risk of an unambiguous loss at 150M if the kernel/runge lags;
single-owner compute fragility (Colab disconnects already truncated B4 once).
**IMPACT 10 (if run) / FEASIBILITY 4 (compute-bound).** **[B]** protocols, **[N]** Prizma results.

### P7. Local-Learning Leg — the BPTT-free (or reduced) delta model
**WHAT:** Train Prizma-Seq with the state update kept local (it already is) and the projections
trained by a local or surrogate rule (DFA — the other thread's proven machinery — or eprop-style
eligibility traces *(knowledge)*), quantifying the tax vs full BPTT at matched budgets.
**WHY:** Uniquely Prizma in the entire landscape: GDN/RWKV-7/KDA/Titans all train by backprop
through chunks. A measured "what does dropping BPTT cost on the standard suite" result (a) merges
the two Prizma threads into one program, (b) is the neuromorphic hardware story, (c) is the
falsifiable core of the brain claim. Even a large measured tax is publishable as the first
quantification for this family.
**HOW:** The delta write's locality is already argued in `seq/prizma_seq.py`'s docstring; DFA
baselines and the honesty apparatus exist in `src/` (thread 1). Scope: diagnostics + char-LM
first; no downstream claims without parity of protocol.
**FALSIFIABLE TEST:** Pre-registered tax bands (e.g., MQAR solve intact; BPC within +X of BPTT);
ablation isolating which credits (state-vs-projections) matter.
**RISKS:** The tax may be catastrophic for LM (plausible — deep credit assignment is real);
reviewers may see it as a toy mode unless the tax table is exhaustive and honest.
**IMPACT 8 / FEASIBILITY 5.** **[B→N]** — DFA/eprop borrowed; application to the delta-mixer
family is new.

### P8. Brain-Alignment Axis with pre-registered metrics
**WHAT:** Perplexity-matched comparison of Prizma vs TF vs GDN on neural-alignment benchmarks
(Story/audiobook fMRI+EEG next-word prediction style *(knowledge: Schrimpf et al. lineage)*), plus
representational tests of the state (does the carried state decode earlier-trial content the way
hippocampal replay/CA3 models do — a synthetic probe first).
**WHY:** The vacant throne (§0): the production family has zero brain grounding; Prizma's PC
identity predicts *testable* differences (error-driven state updates ⇒ stronger late-surprise
encoding). Aligning with the neuroscience-lens committee members' proposals, this gives the
"inevitable + brain-like" double goal its only measurable form — and protects the brand from
unfalsifiable creep.
**HOW:** Public benchmarks exist *(knowledge)*; match models on BPC/loss so alignment differences
are attributable to architecture, not fit; pre-register the alignment metrics and the direction of
predicted differences; synthetic state-decoding probe needs no human data (cheap first step).
**FALSIFIABLE TEST:** Pre-registered alignment delta bands at matched LM quality; state-decode
curves vs hippocampal-model predictions. A null is published as a null.
**RISKS:** Cross-disciplinary referee risk; benchmark pipelines are an integration burden;
alignment literature has its own replication disputes — choose benchmarks with documented
reliability.
**IMPACT 7 / FEASIBILITY 4.** **[N]** for this family; benchmarks borrowed.

### P9. Surprise-Gate Truth Commission — close the repo's own negative at power
**WHAT:** The powered version of the one experiment whose point estimates run *against* the
mechanism (surprise_norm < constant < random at n=2 smoke): full n≥10 ablation, plus a Titans-style
momentum variant to test whether the *signal* was weak or the *use* of it was wrong.
**WHY:** The ledger currently says "asserted, not demonstrated; evidence points the wrong way." No
narrative (§4) survives a known-unrepaired negative on its central mechanism claim. Titans'
surprise-based memory *(knowledge)* makes the direction respectable — the commission owes itself
the powered answer either way.
**HOW:** Part of P1's campaign (the ablation harness exists, `results/gpu_ablation.json` defines
the arms); add the momentum arm; pre-register that a confirmed negative demotes the PC mechanism
from "novelty" to "derivation" in the ledger.
**FALSIFIABLE TEST:** It is one.
**RISKS:** Confirms the negative (acceptable — the program keeps the derivation and drops the
mechanism claim; honesty is the brand).
**IMPACT 6 / FEASIBILITY 8.** **[N]** protocol; **[B]** Titans variant.

### P10. Narrative + Ledger Rewrite — make the pivot official
**WHAT:** Rewrite README/report claim hierarchy per §4: lead with the free-energy delta-rule
identity + param-free capacity + hybrid drop-in + local-learning program; demote "replace
attention" to historical context; fold the borrowed-vs-new ledger to include the 2025-2026 family
(GDN-in-Qwen3-Next, KDA, RWKV-7, DeltaProduct, Titans) so the repo's positioning is auditable
against the live landscape.
**WHY:** Positioning is read before results; the current framing undersells the only
non-commodity assets and overexposes the commodity ones. The ledger update is the
pre-registration *of the strategy itself* — future referees can check the pivot against the
landscape as it stood.
**HOW:** Docs-only; zero GPU.
**FALSIFIABLE TEST:** Every new headline claim maps to a rung gate in §2 (auditable mapping
table).
**RISKS:** None beyond churn; coordinate with other committee lenses (esp. free-energy and
neuro reports) for one consistent story.
**IMPACT 7 / FEASIBILITY 9.** **[N]** framing.

**Sequencing (strict):** P1 → P9 → P2 → P10 (first pass) → P3 (on P1 artifacts) → P4 ∥ P5/P6 →
P7 → P8. P1 and P9 unblock everything; P5/P6 are the moonshot's load-bearing rungs and must not
start before P1's gates pass.

---

## 6. Minimum publishable unit vs moonshot

**MPU (achievable with existing code + ~100–200 A100-hours):** *"Prizma-Seq: a free-energy-derived
gated-delta network with parameter-free quadratic capacity — a powered landscape study."*
Contents: clean R0+R1 (parity vs TF, head-to-head vs GLA/Mamba-2, B4 closed, quarantine repaired),
P2 shootout + capacity-law validation, P9 surprise-gate verdict at power. Every element uses
committed, tested harnesses. Publishable honestly even if the verdicts are mixed — that is the
house style, and the field respects negatives from a repo that audits itself.

**Moonshot (the "inevitable" paper + program):** *"The Free-Energy Sequence Model."* One objective
(P3), one kernel (P4), one drop-in win in the deployment shape (P5 at 150M), one scaling ladder
(P6), one local-learning tax table (P7), one brain-alignment axis (P8), one capacity law (P2) —
positioned as the canonical form of the family production already adopted, and the only
brain-grounded member of it. This is an oral-track-shaped claim and needs the owner's compute
answers (Q1) plus at least one kernel-engineering collaboration. Fallback at every rung is the
previous rung's honest paper; nothing is all-or-nothing.

---

## 7. Borrowed-vs-new ledger updates this analysis proposes

| Item | Status change | Rationale |
|---|---|---|
| "Efficient attention replacement" headline | **retire** | commodity since the 2025 production hybrids |
| Hybrid 3:1 drop-in testing | new **[B]** (Samba/Qwen3-Next pattern) | the deployment shape is the community's, the test is ours |
| Free-energy identity | stays **[N] but gated on P3/P9** | currently a derivation + a suspected-false mechanism claim |
| quad2 | stays **[B→N]**; N upgrades only if P2 earns the law or the matched-FLOP win | framing alone is not enough |
| `inctx_lr` / `n_delta` arms | **[B]** (RWKV-7 / DeltaProduct) used as comparison axes | honesty requires naming them as reimplementations in the landscape |
| Local-learning (DFA/eprop) mode | **[B→N]** | DFA is thread-1's own machinery; application to the family is new |
| Brain-alignment claims | **[N], permitted only with pre-registered metrics (P8)** | otherwise unfalsifiable creep |

---

## 8. QUESTIONS FOR THE OWNER

1. **GPU budget envelope** (A100-hours total and per month, Colab credits vs any cluster access)?
   This decides whether the program is: MPU only (P1+P2+P9, ~150 h), adds P5@50M (~+150 h), or the
   full R3 ladder at 150M (~+400 h).
2. **Positioning pivot approval:** retire "replace attention" as the headline; adopt "the
   free-energy, param-free-capacity, locally-learnable member of the family production already
   adopted" (§4). Yes/no/amend?
3. **Kernel appetite:** is hands-on Triton engineering (P4, the single biggest non-scientific
   investment, possibly needing a collaborator) in scope, or do we cap at torch.compile and accept
   the training-tax discount on all wall-clock claims?
4. **Thread merger:** do you want the two Prizma threads formally merged into one
   complementary-learning-systems program (shared papers, shared name), or kept as separate brands
   with P7 as the only bridge?
5. **Brain-axis scope:** invest in the neural-alignment benchmark integration (P8 — needs
   cross-disciplinary care, maybe an external collaborator), or restrict brain claims to the
   local-learning tax table (P7) for now?
6. **Venue/timeline:** is there a target venue+deadline that should order P1–P9 (e.g., a workshop
   first for the MPU, main-track for the moonshot), or is the science sequence free to dictate?
7. **Adoption posture:** may we upstream quad2 and the delta kernel to the FLA library (and say
   so) as part of the "inevitability via adoption" strategy, given it means our levers run inside
   other people's frameworks?
