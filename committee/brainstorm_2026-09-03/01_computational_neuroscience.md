# Brainstorm — Computational Neuroscience lens: canonical cortical microcircuits & hierarchical predictive processing

**Commission:** Prizma → inevitable Transformer rival + maximally brain-like
**Author:** Computational Neuroscientist (Douglas/Martin canonical circuit, hierarchical
predictive processing, cortical column repetition, FF/FB loops, dendritic computation)
**Date:** 2026-09-03
**Classification (per brainstorming skill):** this is an **Architectural-class research-direction
brainstorm**. HARD-GATE honored: **no code, no implementation** — this report is a design/ideas
document only. Because the skill's "ask clarifying questions" step cannot reach the human owner
from inside a committee, every genuinely open, direction-setting question is collected in the
final section **QUESTIONS FOR THE OWNER**.

Sources read before writing: `README.md`, `docs/Prizma.md`, `docs/PRIZMA_SEQ_REPORT.md`,
`committee/round1_synthesis.md`, skim of `seq/prizma_seq.py` and `src/prizma.py`.

---

## 0. Framing: what the brain has that the Transformer structurally lacks

A Transformer layer is a *single pass* of feedforward message passing: activation in →
activation out, no explicit prediction, no explicit error, no feedback, no settling, no
compartmentalization. Cortex, by contrast, has (a) **explicit prediction and error
populations** (Rao & Ballard 1999; Keller & Mrsic-Flogel 2018), (b) a **reciprocal FF/FB
loop that iterates** — perception involves ~10 cortico-cortical cycles at ~10 ms each
(Lamme & Roelfsema 2000), not one forward sweep, (c) **two-compartment neurons** where
feedback lands on apical dendrites and acts as a *teaching signal distinct from the
somatic drive* (Sacramento et al. 2018; Larkum 2013), (d) a **canonical microcircuit**
with a large, approximately equal-strength inhibitory pool performing divisive normalization
(Douglas & Martin 2004; Carandini & Heeger 2012), (e) **columnar repetition** — the same
canonical circuit tiled ~10^5 times with only connectivity and neuromodulatory context
differing, and (f) **neuromodulatory precision control** distributing global surprise
across timescales (Yu & Dayan 2005; Moran et al. 2013).

The striking fact about Prizma, read from the repo: **it already possesses skeleton
versions of (a), (d), (e) and (f)**. `docs/Prizma.md` literally calls the architecture a
"cortical workspace network" with modules-as-areas, error neurons, precision gates, and
ART-style vigilance. `seq/prizma_seq.py`'s delta write is one step of free-energy descent,
i.e. a biological learning rule, and the read is recognition-by-reconstruction. What is
missing, and what this report proposes, is the **hierarchical and dendritic** machinery:
cortex's FF/FB loop, its compartmental teaching, and its laminar organization. These are
exactly the ingredients that could make Prizma *inevitable* rather than merely a DeltaNet
variant, because they change what a layer does (settled inference on an explicit generative
model), not just how fast it runs.

Borrowed-vs-new discipline: each proposal marks **[B]** borrowed (with citations) or
**[N]** new synthesis. Rankings at the end.

---

## 1. Proposal inventory (ranked by expected impact)

### P1 — Stacked hierarchical free energy: each layer predicts the layer below; errors, not activations, cross the boundary
**[B]** mechanism, **[N]** integration into the delta-state mixer.

- **WHAT.** Today Prizma-Seq layers exchange raw activations exactly like a Transformer.
  Replace this with a two-message interface per boundary. Layer ℓ carries its own
  generative side-weights `G_ℓ` producing a top-down prediction `μ_ℓ = G_ℓ f(z_{ℓ+1})`;
  layer ℓ computes its bottom-up **error** `ε_ℓ = f(z_ℓ) − μ_ℓ` and sends *ε* upward
  (and to the mixer). The mixer's write becomes precision-weighted by the *layer-local*
  error energy: `β_t^{(ℓ)} = σ(W_β x_t) · π(E_t^{(ℓ)})` with `E_t^{(ℓ)} = ½‖ε_ℓ‖²`.
  This is exactly the Rao–Ballard hierarchical PC graph (Rao & Ballard 1999; Friston 2008;
  Millidge, Tschantz & Buckley 2021 review) grafted onto the existing per-head state.
- **WHY (brain).** This is the literal canonical connectivity of sensory cortex: L4 → L2/3
  feedforward error, L5/6 deep pyramidal cells emit top-down predictions; each area predicts
  the one below it (Felleman & Van Essen 1991 hierarchy; Bastos et al. 2012 — the canonical
  microcircuit model explicitly maps L2/3 = error units, deep PT-IT = prediction units).
- **WHY (beat transformers).** Transformers pay full d_model² bandwidth to carry
  *uncompressed activations* between layers. A precision-gated error signal is (i) sparse
  where the model already predicts well (most tokens at depth are predicted — error
  sparsity grows with depth, enabling error-skipping/early-exit per token, which attention
  cannot do), (ii) it gives every layer an *online credit signal* for its own state quality
  — currently Prizma's delta write is only driven by the *token-level* free energy
  `F_t(S)=½‖v_t−Sk_t‖²`, blind to whether the layer's output was actually predicted by the
  layer above.
- **HOW.** Add `G_ℓ` (a d_model→d_model linear, or tied-lowrank to keep param honesty) to
  the layer in `seq/prizma_seq.py`; compute `ε_ℓ` in the residual stream; add a
  `hier_pc: bool` config knob (default OFF = byte-identical, per repo culture); ablate as
  B7. Cost accounting must grow the TF baseline in lockstep (guardrail #1 of round1_synthesis).
- **FALSIFIABLE TEST (pre-registered bar).** Char-LM text8 at d128L4H4-matched params:
  `BPC_hierPC ≤ BPC_base − 0.03` (Prizma vs Prizma, n≥5 seeds, Welch test) AND ≥3 of 4
  diagnostic legs unchanged (induction, selective-copy, MQAR, memory). If error-gating
  degrades MQAR (recall needs full-state writes, not error-gated ones), rescope to
  "hier-PC helps only on natural LM, not synthetic recall" — a real, publishable dissociation.
- **RISKS.** Extra params break the beautiful "byte-identical FFN" story; error signals can
  be noisy early in training (precision Π must anneal); chunk-parallel training of G_ℓ is
  easy but the precision gating may need a two-pass (sees the whole chunk first).
- **IMPACT 9 / FEASIBILITY 6.**

### P2 — Apical compartment: a separate feedback input that modulates, not drives, the somatic rate (dendritic error PC)
**[B]** core mechanism (Sacramento et al. 2018; Larkum 2013; Urbanczik & Senn 2014),
**[N]** as a *reusable unit primitive* across both Prizma threads.

- **WHAT.** Give every unit in a Prizma-CL expert (and optionally every channel of the
  Prizma-Seq FFN) **two compartments**: somatic `z_basal = f(W x + b)` and apical
  `z_apical = σ(B ε_{ℓ+1} + I_v)` where `ε_{ℓ+1}` is the error from the layer above (or the
  DFA feedback `B` already present in `src/prizma.py`) and `I_v` is a voltage threshold.
  Output is `f(z_basal + κ·relu(z_apical − θ_ap))`: the apical input **gates plasticity
  windows and increases gain only when it is depolarized** — burst mode. The learning rule
  becomes: update W only when the soma spikes AND the apical dendrite carries a top-down
  error (the classic dendritic coincidence detector; Larkum's "backpropagating-AP + apical
  EPSP" gate).
- **WHY (brain).** This is the anatomical fact of layer-5 pyramidal cells: basal ← feedforward,
  apical ← feedback; coincidence of the two triggers calcium bursts (Larkum, Zhu & Sakmann
  1999). Sacramento et al. showed this circuit *is* an exact implementation of predictive-
  coding gradient descent without weight transport — which directly attacks Prizma's open
  problem **P2** with a mechanism biologically more principled than DFA.
- **WHY (beat transformers).** DFA gives random-but-fixed feedback; the dendritic scheme
  *learns* the feedback locally (`B` trained by the same local delta rule, per
  `docs/Prizma.md` §2.5's `dQ_m/dt`). Repo evidence says DFA ≥ exact W^T in the current
  regime (E4: 0.834 vs 0.708) — but that regime is shallow. A learned-apical variant is
  the natural next rung for scaling the backprop-free claim (P1 open problem).
- **HOW.** In `src/prizma.py::Expert`, replace the single-step DFA feedback with a
  learned `B` + apical gate (`apical: 'fixed'|'learned'|'off'`); measure on the E1 suite.
  Optional Seq analog: apical input to the FFN channels from the residual-stream error of P1.
- **FALSIFIABLE TEST.** E1 structured-permuted, ≥10 seeds: `ACC_apical-learned ≥ 0.85·ACC_oracle`
  maintained AND on a *harder* interleaved stream (where Prizma currently collapses to ~0.58)
  the apical-feedback variant must beat `ACC_naive` by ≥0.05. If it only matches DFA on E1,
  the claim fails (honest scope: it wins only where P2-hard feedback matters).
- **RISKS.** Two-compartment units add state and hyperparameters (κ, θ_ap, τ_ap); the
  interleaved regime may remain fundamentally hard (docs §8 says temporal contiguity is
  exploited) — the test above may be unreachable, which itself is informative.
- **IMPACT 7 / FEASIBILITY 7.**

### P3 — Iterative settling: replace one-pass forward with a small fixed number of free-energy descent cycles (the FF/FB loop as computation)
**[B]** mechanism (PC inference = iterative settling; Bogacz 2017; cf. Deep Equilibrium
Models, Bai et al. 2019), **[N]** binding it to the *pre-write read* so the state S also
settles.

- **WHAT.** Cortex perceives in ~10 gamma cycles; a Transformer sees each token exactly once.
  Let Prizma-Seq iterate K∈{2,3,4} inner steps per token at *inference*: read `o = S q`,
  refine q via the window head + FFN, re-read. Mathematically: `q^{(k+1)} = q^{(k)} − τ∇_q
  ½‖o^{(k)} − S q^{(k)}‖²` — gradient descent on the *same* per-token free energy the write
  already uses (the architecture becomes symmetric: write = one step on F(S), read =
  K steps on F(q)). This is recognition-by-reconstruction *iterated*, which is literally
  what cortical FF/FB does (Rao & Ballard's E-step; Friston's perception-as-inference).
- **WHY (beat transformers).** It buys accuracy with **inference compute, not parameters or
  context length** — an axis Transformers cannot touch at O(1) memory (a TF iterating on
  itself needs a KV cache that stays). This is the "adaptive-compute at O(1) state" wedge:
  Prizma could match a bigger TF on hard tokens by spending 3 reads instead of 3 layers.
- **HOW.** `step()`/`forward()` in `prizma_seq.py`: wrap the read in a K-iteration loop;
  new knob `settle_k: int = 1` (1 = byte-identical); guardrail #4 requires a new
  `step == forward` equivalence check with settle_k>1 (both must loop identically).
  Sweep K on MQAR D=128 + text8 against a depth-matched TF.
- **FALSIFIABLE TEST.** Pre-register: `acc(K=3) − acc(K=1) ≥ 0.02` on MQAR D=128 at
  fixed params, AND `BPC(K=3) ≤ BPC(K=1) − 0.02` on text8, with the K=3 latency cost
  ≤ 2.5× per token (measured, not analytic). If accuracy is flat in K, the settling
  hypothesis is falsified for this state form — report and move on.
- **RISKS.** Sequential delta is already the training bottleneck; iterating reads may not
  train (train with K=1, settle only at eval — like acting-then-thinking; must disclose).
  Non-convergence for large K (the Jacobian bound in `docs/quad2_theoretical_convergence.md`
  suggests decay-gated stability may carry over, but that is a conjecture to test).
- **IMPACT 8 / FEASIBILITY 8.**

### P4 — Canonical microcircuit block: explicit E/I populations with divisive normalization inside every layer
**[B]** Douglas & Martin 2004; Carandini & Heeger 2012 (normalization), Rubin et al. 2017
review; **[N]** parameterization choice (low-rank shared inhibitory pool per head).

- **WHAT.** The canonical circuit's core quantitative fact: inhibitory cells form a
  strongly recurrent pool that *divisively normalizes* the excitatory drive, providing
  (i) gain control, (ii) stability, (iii) automatic proportional coding. Implement the
  layer computation as `y = W_e x ⊘ (κ + ‖W_i x‖²)`, with `W_i` a *low-rank* (rank 8–16,
  per-head) projection — not a free full matrix — and κ a learned threshold. This is
  pLSTM/normalized-attention adjacent but motivated and structured differently: the
  inhibitor pool is shared across the head, giving a small param cost and a stabilizing
  spectral effect on the delta state (‖S‖ bounded by normalization → helps the capacity-
  vs-stability tradeoff that Gershgorin analysis in the repo quantifies).
- **WHY (brain).** This *is* the Douglas–Martin canonical microcircuit: 80/20 E/I ratio,
  strong thalamic drive to both, inhibition dominating the population response; the
  normalization family is the best-quantitative model of cortical gain control.
- **WHY (beat transformers).** Transformers contain only softmax (a global normalizer in
  attention) and RMSNorm (per-token); neither normalizes *over the state*, which is where
  Prizma's failures live (state norm growth over long streams degrades delta writes; the
  char-LM tiny-shakespeare failure with an overfitting recipe smells like exactly this).
- **HOW.** Knob `ei_pool: {off, lowrank-r}` in `prizma_seq.py`; ablate on text8 (the
  failing regime) and lengen (norm drift is a long-context disease).
- **FALSIFIABLE TEST.** Pre-register: tiny-shakespeare BPC gap to TF shrinks from −0.09 to
  ≥ −0.03 at matched params, n≥5; state-norm growth over 65k tokens reduced ≥5× (measured
  diagnostic). If shakespeare does not improve, the "overfitting was state-drift" hypothesis
  is dead and the char-LM failure gets its honest diagnosis anyway.
- **RISKS.** Divisive normalization can kill recall sharpness (MQAR needs sharp readouts);
  keep a per-head bypass. Risk of "yet another norm layer" reviewer fatigue — the defense is
  the microcircuit rationale + causal ablation.
- **IMPACT 7 / FEASIBILITY 8.**

### P5 — Cortical-column unification: Prizma-CL experts and Prizma-Seq heads are the same canonical column; one codebase, two regimes
**[N]** synthesis; columnar tiling borrowed as *principle* (Mountcastle 1957; Douglas & Martin).

- **WHAT.** Cortex does not have "an RNN part" and "a continual-learning part" — it has one
  canonical column repeated, with different input statistics selecting different dynamics.
  Concretely: define a single **Column** module = PC autoencoder (encoder/decoder, local
  delta rule, vigilance μ/σ precision pair from `src/prizma.py`) + a small carried
  delta-state S (from `seq/prizma_seq.py`). A *layer* of the sequence model is a bank of
  Columns; routing (vigilance on reconstruction surprise) selects which column writes to
  the workspace. Prizma-CL is then "columns over domains", Prizma-Seq is "columns over
  time" — and the union is **columns over both**, i.e. continual sequence modeling with
  expert-level memory isolation, which no Transformer does and which directly attacks the
  biggest unsolved ML problem (catastrophic forgetting in LMs).
- **WHY (brain).** Columnar repetition + area-level specialization is Mountcastle's
  organizing principle; vigilance/consolidation maps to hippocampo-cortical gating
  (cf. Kumaran, Hassabis & McClelland 2016 — complementary learning systems).
- **WHY (beat transformers).** The union product — an LM whose experts recruit/freeze
  online (zero forgetting over *domains of text*, languages, users) with O(1) per-expert
  state — is a capability Transformer cannot emulate without MoE-with-boundaries hacks.
  This is the single most "inevitability"-shaped idea in this report.
- **HOW.** No code yet — first a design note unifying the two configs and the shared
  `Precision`/`DeltaWrite` abstractions; then the smallest union experiment: char-LM on
  an interleaved-corpus continual stream (text8 → shakespeare → text8 again), measuring
  forgetting of corpus 1 without replay.
- **FALSIFIABLE TEST.** Pre-register: after sequential training on text8→shakespeare→re-test
  text8, `BPC_text8(after) ≤ BPC_text8(before-shakespeare) + 0.05` with zero replay, at
  matched total params vs a monolithic Prizma-Seq whose BPC degrades by ≥0.15. If vigilance
  routing on *token* surprise (vs the batch-level novelty CL uses) fails to isolate domains,
  report the minimum domain-size at which it works.
- **RISKS.** The interleaved-regime collapse documented in `docs/Prizma.md` §8 says this is
  hard in exactly the regime text streams live in; per-token (not per-domain) routing is a
  real research risk. This is the highest-variance proposal here.
- **IMPACT 10 / FEASIBILITY 4.**

### P6 — Successor-representation state: interpret and reshape S as a predictive state of *features*, not a key-value store
**[B]** Dayan 1993 (successor representation); Stachenfeld et al. 2017 (hippocampus as SR);
**[N]** as the semantics of the delta state.

- **WHAT.** Currently `S` is justified as associative memory (MQAR). An alternative
  neuroscience-anchored semantics: `S_t` is a **successor feature** estimate — the
  discounted predictive representation of future φ-features. The delta write with decay
  `α_t` is already the SR/TD update form: `S_t = S_{t-1} + β_t (φ_t + γ φ_{t+1}·? − S_{t-1}k_t)k_tᵀ`
  — add a single learnable per-head γ (discount) and the state becomes a predictive map.
  The read `S q` then answers "what features tend to follow context q?" — in-context
  *prediction* rather than pure lookup.
- **WHY (brain).** Hippocampal CA1 representations are best explained as SRs
  (Stachenfeld, Botvinick & Gershman 2017); the delta rule family *is* TD(0).
- **WHY (beat transformers).** The KV cache stores *what happened*; an SR state stores
  *what will happen*, which extrapolates: length-generalization (currently Prizma's
  absolute 0.40 @8×) is exactly where predictive-state semantics should beat a cache.
- **HOW.** Knob `sr_gamma: float = 0` (0 = byte-identical off); lengen leg B5/frontier is
  the natural bench. Pre-register before running (repo rule).
- **FALSIFIABLE TEST.** Length-extrapolation: retention @8× train length improves from
  ~0.40 to ≥0.55 with sr_gamma>0 vs sr_gamma=0, same seeds, matched params. If flat → SR
  semantics adds nothing at this state rank; report.
- **RISKS.** γ interacts with the gated decay α_t (double discounting); theory is thin.
- **IMPACT 6 / FEASIBILITY 7.**

### P7 — Neuromodulatory precision hierarchy: one global scalar distributes per-layer plasticity/attention gains (the repo's NM, extended to depth)
**[B]** Yu & Dayan 2005 (ACh = expected uncertainty, NE = unexpected); Moran et al. 2013
(PT hierarchical precision); **[N]** its use as the *depth-wise* learning-rate profile.

- **WHAT.** `docs/Prizma.md` already has NM (global outcome-error scalar) and per-module
  metaplasticity. Extend: a single global surprise scalar `E_t` (from the deepest layer's
  free energy) modulates layerwise plasticity **exponentially with depth**: shallow layers
  learn on *expected* error (high precision, low β), deep layers on *unexpected* surprise
  (low precision, high β). This is the brain's laminar gradient (deep layers = slower,
  more abstract; L2/3 = fast, plastic) and prevents the classic failure where late layers
  memorize and early features drift.
- **WHY (beat transformers).** Gives continual-LM training a principled depth-wise LR
  schedule *derived from the model's own surprise*, useful in both threads; in CL it
  directly complements ω-consolidation with a depth dimension.
- **HOW.** In `src/prizma.py`'s PGM: `β_m → β_{m,ℓ} = β_m · exp(−λℓ·Ẽ_t)`, one new
  hyperparameter λ. Test on E1 and on a depth-2/3 expert variant.
- **FALSIFIABLE TEST.** E1: ACC non-inferior (≥0.83) AND on the noise-sweep E2 the
  depth-modulated variant holds ACC ≥0.6 at noise 0.9 (vs 0.43 baseline) — i.e. the
  precision gradient buys robustness specifically in the hard-noise regime.
- **RISKS.** Two knobs (λ per depth) invite tuning smells; keep one global λ.
- **IMPACT 5 / FEASIBILITY 8.**

### P8 — Theta/gamma two-timescale workspace: slow `a` (gamma-bound) rides on fast per-token state, with phase-coded writes
**[B]** Lisman & Jensen 2013 (theta-gamma coding), Buzsáki (timescales); **[N]** coupling.

- **WHAT.** Cortex multiplexes a fast token stream (gamma) onto a slow working-memory
  carrier (theta). In Prizma terms: the per-head S updates per token (fast), while a
  small "theta frame" vector θ_t updates only every W tokens (a chunk-level workspace
  with its own vigilance test). Writes to S are phase-gated within the chunk: tokens in
  the "encode" phase write; tokens in the "retrieve" phase read with amplified gain —
  a *temporal* separation of read/write that today's single-pass mixer interleaves
  arbitrarily. This is a microcircuit-consistent reason why the existing local-window
  attention head (window=16 ≈ one theta cycle) exists in the architecture.
- **WHY (beat transformers).** Phase-gated read/write could cut write FLOPs (Prizma's
  disclosed 2.14× FLOP deficit vs TF) by writing only on encode phases — a compute
  advantage *derived from* the neuroscience, and directly aimed at the biggest honest
  weakness in the ledger.
- **HOW.** Knob `theta_phase: off|periodic`; FLOP ledger re-run; MQAR + selective-copy
  (write-heavy) and induction (read-heavy) legs expected to dissociate cleanly — that
  dissociation IS the test.
- **FALSIFIABLE TEST.** Pre-register: FLOPs/token ≤ 1.4× TF (from 2.14×) with all
  diagnostic legs PASS unchanged. If MQAR degrades >0.01, phase gating loses; report.
- **RISKS.** Feels like chunked-parallel training's existing chunk structure — risk of
  reinventing `chunk=64` under a neuroscience name. Must show a *measured* FLOP change,
  not a relabeling.
- **IMPACT 6 / FEASIBILITY 5.**

---

## 1b. Explicit microcircuit mapping table (what Prizma's parts *are* in cortex terms)

| Prizma component | Cortical counterpart | Fidelity | Gap |
|---|---|---|---|
| Per-head state `S` (seq) | Hippocampal/dentate associative state; CA3 autoencoder | medium | no SR semantics (P6), no decay hierarchy |
| Delta write (1 grad step on F) | One-shot delta-rule plasticity / eLTP | high (this is genuinely biological) | write gate is input-dep, not surprise-dep at scale (surprise_gate exists, ablation inconclusive) |
| Read `S q` (pre-write) | Recognition-by-reconstruction, fast feedforward sweep | high | no iterative settling (P3) |
| Prizma-CL experts | Cortical areas / columns | high | no laminar structure, no FF/FB between experts (P1, P2) |
| Vigilance (μ/σ on recon) | Novelty detection; dentate pattern separation; dendritic mismatch | high | batch-level, not online-dendritic |
| PGM ω consolidation | Synaptic consolidation, systems consolidation | high (Bayesian-synapse grounded) | no depth profile (P7) |
| FA/DFA feedback | Feedback alignment — biologically plausible *shadow* of feedback | low-medium | real cortex has learned apical feedback (P2) |
| Workspace `a` + gate | Thalamus + PFC, PBWM | medium | not temporally structured (P8) |
| `quad2` monomials | Dendritic nonlinearities (quadratic coincidence detection) | medium — an honest *analogy*, not a claim | unexplored as dendritic feature binding |
| Layer-to-layer messages | Raw activations (same as TF!) | **none — this is the biggest gap** | P1 is the fix |
| Inhibitory interneurons | Absent (only RMSNorm) | **none** | P4 is the fix |

The two "none" rows are the strategic conclusion of this report: **Prizma's units and rules
are already more brain-like than a Transformer's; its *inter-unit structure* is not.**
Hierarchy (P1) and inhibition (P4) are the missing canonical-circuit organs, and both
happen to be mechanisms Transformers lack — so brain-likeness and rivalry point the same way.

---

## 2. Ranking summary

| # | Proposal | Borrowed/New | Impact | Feasibility | One-line bet |
|---|---|---|---|---|---|
| P1 | Hierarchical FF/FB error messaging between layers | B(Rao-Ballard)/N-integration | 9 | 6 | Errors across boundaries are the organ Transformers don't have |
| P3 | Iterative settling (K-step read) | B(Bogacz)/N-binding | 8 | 8 | Accuracy from inference compute at O(1) state |
| P5 | Column unification → continual LM | N / B(Mountcastle, CLS) | 10 | 4 | The "inevitability" product: an LM that never forgets a domain |
| P2 | Apical compartment (dendritic error PC) | B(Sacramento, Larkum) | 7 | 7 | Learned feedback beats DFA exactly where scaling bites |
| P4 | Canonical E/I pool, divisive normalization of state | B(Douglas-Martin, Carandini) | 7 | 8 | State-norm drift is the char-LM failure; inhibition is its cure |
| P6 | Successor-representation state semantics | B(Stachenfeld)/N | 6 | 7 | Predictive state beats a cache on length extrapolation |
| P8 | Theta-gamma phase read/write separation | B(Lisman-Jensen)/N | 6 | 5 | A neuroscience-derived FLOP cut on the 2.14× deficit |
| P7 | Depth-wise neuromodulatory precision gradient | B(Yu-Dayan)/N | 5 | 8 | Surprise-derived depth LR schedule for CL robustness |

Recommended sequencing (honesty-preserving): **P3 and P4 first** (cheapest, both target
known failure modes, both default-off byte-identical knobs); **P1** as the flagship
architectural bet once P4 stabilizes text8; **P2** inside Prizma-CL in parallel (independent
codebase); **P5** as the strategic union project requiring a design note before any code.
Each keeps the repo's guardrails: default-off = byte-identical, param/FLOP-matched TF growth
in lockstep, pre-registered bars, ≥5 seeds, measured ledgers, causal ablations.

---

## 3. QUESTIONS FOR THE OWNER

1. **Strategic identity:** is Prizma's *headline identity* (a) the O(1)-state efficient
   attention replacement (Prizma-Seq scaled up), (b) the backprop-free continual learner
   (Prizma-CL scaled up), or (c) the unified cortical column (P5)? Resource allocation
   between the threads depends entirely on this.
2. **Compute budget:** P1/P5 need multi-A100 runs (the quarantined recall gate alone was
   ~28 A100-hours). What is the realistic compute envelope for the next 3 months?
3. **Brain-likeness vs rivalry priority:** when they conflict (e.g. P4's E/I pool helps
   scaling but weakens the "every unit is a pyramidal cell" story), which wins?
4. **P5 union risk appetite:** interleaved-stream collapse is documented. Are you willing
   to pre-register a possibly-failing bar for the continual-LM union, knowing a clean
   failure is still publishable in the repo's culture?
5. **Backprop-free frontier:** is backprop-free *sequence* LM training a goal (P2/P7
   extended to seq), or does seq stay backprop-trained while CL carries the locality flag?
6. **Surprise-gating status:** the R9 ablation point estimates ran *against* the
   surprise mechanism. Is closing that question (powered surprise_gate ablation) a
   priority before layering more precision machinery (P1, P7) on top of it?
7. **Naming the biology:** the repo is admirably literal about borrowed vs new. Should
   the neuroscience analogies (column, workspace, vigilance) stay as documentation-level
   framing only, or become pre-registered *neural-prediction* targets (e.g. testing
   emergent error-unit structure), which is a heavier but more distinctive claim?
