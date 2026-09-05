# 05 — Predictive Processing / Free-Energy Principle lens
## Committee brainstorm 2026-09-03 — "Prizma as inevitable rival + maximally brain-like"

**Author role:** Predictive Processing / FEP Theorist (Friston free energy, active inference,
hierarchical generative models, precision engineering, expected free energy).
**Classification:** Architectural-class research-direction brainstorm. **Report only — no code.**
**Repo state read:** README.md, docs/Prizma.md, docs/PRIZMA_SEQ_REPORT.md, seq/prizma_seq.py,
(src/prizma.py via its full writeup). Committee standards honored: pre-registered falsifiable
bars, param/FLOP-matched baselines, honest limits, borrowed-vs-new ledger.

---

## 0. Where Prizma genuinely already sits on the FEP map (an honest audit)

Most "predictive coding" branding in ML is decoration. Prizma's is not — but it is also not yet
FEP. The honest audit:

**Already real FEP:**
- Prizma-Seq's delta write `u_t = β_t(v_t − S_{t−1}k_t)` is **exactly one gradient step on a
  per-token variational free energy** `F_t(S) = ½‖v_t − S k_t‖²` (prizma_seq.py docstring; the
  `step()` code confirms: `eps1 = v1 − a1·Sk; u = β·eps1`). This is the descent-on-generative-
  -weights half of FEP, realized as a *state update* rather than a weight update. Transformers
  have no such identity at all — attention writes by score-weighted copying, not error minimization.
- Prizma-CL's expert routing bids are **precision-weighted surprise** `b_m = ½ε_mᵀΠ_mε_m` with
  per-expert `(μ,σ)` EMAs — a variational free-energy proxy used for **Bayesian model selection**
  (which expert explains this input best). The sign-tension resolution (one surprise energy, two
  opposite readouts: precision ↑ for attention, plasticity ↓ for consolidation) is a real
  contribution, not a citation.
- The master functional `F = Σ precision-weighted errors + complexity prior` in docs/Prizma.md §2.1
  is the genuine F=accuracy+complexity decomposition.

**Not yet FEP (the gaps this report attacks):**
1. **Depth.** The generative model is ONE level: `v_t ≈ S k_t`. The cortex is a deep hierarchy of
   predictions; FEP's explanatory power comes from *cross-level* precision and error propagation.
   Prizma-Seq stacks L blocks but each block's delta write is a flat, level-free association —
   no block predicts another block's states.
2. **Precision is a gate, not an object.** `β_t = σ(W_β x_t)` is a learned scalar squashing. Π in
   Prizma-CL is a per-expert scalar EMA. Neither is a *structured, inferred precision* over state
   channels, heads, timescales, or tokens — the thing FEP says the brain actually estimates
   (attention AS precision optimization).
3. **No action/expected free energy.** FEP's second half is active inference: acting to change
   sensory sampling so as to minimize *expected* free energy (epistemic + pragmatic value).
   Prizma's only "action" is the write, and its gate is heuristic (`σ(W_β x)`), not derived from
   expected free energy. The repo's own ablation admits the surprise-gate evidence "points the
   wrong way" (gpu_ablation.json, INCONCLUSIVE, constant>real signal at n=2). A heuristic gate
   that fails its own ablation is exactly where a principled G(x_t, S) derivation should go.
4. **Inference stops at training.** A transformer freezes at deployment. FEP says an organism is
   *perpetually* inferring — test-time state adaptation is Prizma's structural right (it carries
   state at O(1)) and is currently unused at decode.

**The strategic thesis of this report:** the Transformer's deep weakness, from the PP lens, is
that it is a **pure pattern-completion machine with no generative model of its own state** — no
precision, no expected surprise, no continual inference, O(n) memory of its own sensory history.
Every gap above is simultaneously a *brain-likeness* win and a *capability* moat. Prizma should
not out-transformer the Transformer; it should be the first LM whose *inference loop itself*
minimizes free energy at train AND test time.

---

## 1. Proposals (ranked)

Scoring: IMPACT 1–10 on the twin goals (rival + brain-like); FEASIBILITY 1–10 within the repo's
honest, small-scale, pre-registered culture. [B]=borrowed with citation, [N]=new, [B/N]=borrowed
mechanism, new synthesis.

---

### P1. Deep predictive hierarchy: cross-layer error propagation (levels, not just layers)
**WHAT.** Today each block computes its *own* local delta free energy F_t^(l)(S^(l)). Restructure
the stack so layer l's *output state read* is the top-down prediction for layer l−1's error
units: ε^(l) = read(S^(l)) − g(S^(l+1)), each layer minimizing its precision-weighted error
against the layer above, with a small top-level prior over the workspace state. Concretely: keep
the delta state per layer, but add (a) an explicit error carrier between blocks (replace the
plain residual h + f(h) with a prediction-error unit whose norm is tracked and precision-gated),
(b) a hierarchical timing asymmetry — lower layers do N≥1 fast local delta steps per token,
upper layers 1 slow step (cortical timescale separation), amortized identically under the O(1)
streaming regime.
**WHY.** This is the single biggest move from "PC-branded" to "PC architecture." It is also a
concrete rival differentiator: in a transformer, gradients are the *only* vertical information
channel, and only during training; in hierarchical PC, prediction and error flow vertically at
*inference time in the forward pass*, which is exactly how the brain does vision/language
(Rao & Ballard 1999; Friston 2008; Bogacz 2017 [B]; depth-wise PC nets: Salvatori et al. 2022,
PCConv/transformer work [B]). Predictable structure gets absorbed at low levels with low
precision; surprises propagate up — giving Prizma structured, layer-localized OOD/surprise
readouts a transformer cannot produce without extra machinery.
**HOW.** Stage 1 (cheap): instrument — per-layer ε-norm trajectories on char-LM and MQAR; verify
that layers do develop the PP signature (low-level errors → low on predictable corpora, spikes at
boundaries/rare tokens). Stage 2: add top-down prediction taps between two adjacent layers only;
ablate. Stage 3: full N-fast/1-slow asymmetry. Param-matched vs the existing Prizma-Seq and TF
arms at every stage.
**FALSIFIABLE TEST (pre-register).** H1: two-layer coupled-error Prizma ≥ parity with the current
stack on char-LM text8 (Δ BPC ≤ +0.02) at matched params. H2 (the PP signature): per-layer
error norms on held-out text show monotone depth-wise compression (lower-layer error energy
decreases faster than upper) — quantified as the slope of log ε-l vs depth; a transformer with
identical probes shows no such structured slope. H3: domain-shift token range (swap corpus at
test) produces error spikes localized to LOW layers first, upper layers later — a temporal
hierarchy signature. If H1 fails parity or H2 shows no structure, the hierarchy is dead weight;
report and stop.
**RISKS.** Inter-layer coupling can destabilize chunk-parallel training (the delta kernel's
parallel form assumes per-layer independence — coupling may force sequential or nested-parallel
schemes, worsening the existing 5× training-speed tax). Risk of the hierarchy collapsing to the
residual stream (layers ignore top-down taps) — that is what H2 catches. Adds params for
prediction taps (must be counted; keep taps low-rank).
**IMPACT 9 / FEASIBILITY 5.** [B: hierarchical PC — Rao&Ballard, Friston, Bogacz, Salvatori;
N: error-carried delta states + O(1) hierarchical streaming — new synthesis]

---

### P2. Precision as a first-class inferred object (from scalar gate to structured Π)
**WHAT.** Replace `β_t = σ(W_β x_t)` (and the CL scalar Π) with an actual inferred precision
matrix over the delta-state value channels: Π_t = diag(π_t) with π_t = σ(W_π x_t) learned as the
*inverse-variance of the one-step-ahead prediction error*, trained by minimizing the proper
Gaussian free energy F_t = ½ ε_tᵀ Π_t ε_t − ½ log det Π_t. The optimal solution of this objective
is π_t = 1/E[ε²] — the gate becomes the inverse error variance *by derivation*, not by
parameterization. Extend across heads (per-head precision), channels (Lever G's per-value η is
already half of this — W_eta is the natural substrate), and timescales (EMA-tracked per-channel
error variance at 2–3 rates, à la Prizma-CL's (μ,σ) but per-channel and inside Seq).
**WHY.** This is the FEP core claim made operational: **attention IS precision optimization**
(Feldman & Friston 2010 [B]). It converts Prizma's weakest empirical point — the surprise-gate
ablation where the "real signal" underperformed constant/random controls (n=2, INCONCLUSIVE, but
pointing wrong) — into a *derived* gate with a closed-form optimum, plus a self-auditing
signature: after training, π_t should correlate ≈ −corr with realized squared error. A scalar
gate has no such measurable optimality condition; this one does, which is exactly the repo's
kind of falsifiable claim. Also unifies the two threads: Seq's η and CL's (μ,σ) become two
readouts of one precision object — the "same signal, two signs" synthesis, now within one model.
**HOW.** Stage 1: replace β with ½‖ε‖²·π-derived scalar on MQAR + char-LM, uniform-precision and
random-precision controls (reusing B6 discipline). Stage 2: per-channel diag(π) via W_eta
upgrade; compare against Lever-G-as-is. Stage 3: multi-timescale per-channel variance (fast EMA +
slow EMA) — fast drives the write gate, slow drives a consolidation-style freeze of high-precision
channels (metaplasticity inside the state, no weights touched).
**FALSIFIABLE TEST.** Pre-register: (a) optimality audit — trained model's −log π_t vs realized
log ε² Spearman ρ ≥ 0.5 on held-out text (a random/constant gate gives ρ≈0 by construction);
(b) powered (n≥5) MQAR D=128 + text8: diag-precision arm ≥ baseline with non-overlapping CI, or
the honest verdict "precision signal adds nothing at this scale" — *finally powering the
surprise/precision ablation the repo itself says is owed*. (c) Zero-forgetting style test: freeze
the top-decile-precision channels at test-time domain shift and show retention improves vs
freezing random channels.
**RISKS.** −½log det Π with diag(π) is safe, but precision can run to saturation (all-π→∞ on
deterministic data; clamp + β_cap discipline as today). Correlation-audit can pass while task
performance doesn't move (precision real but useless) — that's a partial result, report as such.
The existing n=2 evidence pointing the wrong way is a real prior against the mechanism; this
proposal's value is precisely that it converts that embarrassment into a powered, derived test.
**IMPACT 8 / FEASIBILITY 7.** [B: Gaussian free energy, precision-weighting — Friston 2005;
Feldman&Friston 2010; N: derived-precision delta write unifying Seq's η and CL's (μ,σ)]

---

### P3. Expected free energy for the write: epistemic foraging as a principled write gate
**WHAT.** The current write gate asks "how surprising is this token?" — a *past-directed*,
heuristic quantity. Replace it with a one-step **expected free energy** of writing vs not
writing: G(write) = E[ε²_{t+1} | write, S'] − E[ε²_{t+1} | no-write] + γ·(info gain about the
state), computable in closed form for the linear-Gaussian association because the delta model
has an analytic posterior over what state S' would predict. Tokens get written when writing
*expectedly* reduces future free energy — not when they were surprising. This subsumes the
epistemic/pragmatic split: epistemic value = reduction in uncertainty about the state (novel
key with unexplored binding slots); pragmatic value = reduction in expected retrieval error
(content that will be queried).
**WHY.** This is the repo's honest gap: the surprise gate is asserted, untested, and the only
evidence points against it. Surprise-gating writes *everything surprising*, including noise and
aliasing collisions that destroy associative memory — expected-FE writes only what *pays*.
Mechanistically it is the active-inference move (Friston et al. 2015, 2017 expected free
energy; Schwartenbeck et al. on epistemic value [B]) applied to state action rather than motor
action — the closest thing in the literature is work on data-dependent memory writes in
DeltaNet/Mamba family, which are all learned heuristics [B-family], none derived from G.
**HOW.** Derive G in closed form under the L2-key linear-Gaussian assumption (the state's
prediction is a linear-Gaussian map; one-step expected error under write/no-write is analytic —
feasible as a report-level derivation first, then a two-pass streaming variant reusing the
existing surprise_gate two-pass scaffolding). Ablate: G-gate vs surprise-gate vs learned-σ gate
vs constant/random controls (the existing B6 control discipline slots in directly).
**FALSIFIABLE TEST.** Pre-register: on selective-copy and MQAR at aliasing-heavy settings
(D > d_h·H, where collisions matter), G-gate > surprise-gate > baseline with ≥5 seeds, non-
overlapping CIs; and per-token write decisions correlate with ground-truth "useful binding"
tokens (measurable exactly on synthetic tasks — MQAR knows which kv pairs matter). If G ≈
surprise-gate ≈ baseline, the honest headline is "write gating does not need active inference
at this scale" — still a publishable negative that kills the overclaim.
**RISKS.** Two-pass inference breaks the clean O(1)/step story (mitigation: one-pass G with
EMA-estimated expectations; the repo already ships a two-pass surprise variant, so precedent
exists). Derivation risk: the closed form needs noise-level assumptions the state doesn't
actually satisfy (quad2 keys are not Gaussian) — validate the approximation empirically first.
Compute: G costs one extra state read per token.
**IMPACT 8 / FEASIBILITY 5.** [B: expected free energy, epistemic value — Friston/Schwartenbeck;
N: G-gated *state writes* in a sequence LM — new]

---

### P4. Test-time perpetual inference: the online-adaptation bar (Prizma's killer differentiator)
**WHAT.** A transformer at decode is frozen: identical function forever. Prizma's carried state
+ delta write IS a continual inference machine; at decode the state keeps minimizing F_t on the
*deployment* distribution for free. Formalize this as a named capability axis with its own
pre-registered bar: **O(1)-memory online adaptation** — evaluate BPC on a corpus whose
distribution drifts mid-stream (e.g., text8 → shakespeare → code, or synthetic Markov chain
with a transition-matrix switch) with NO gradient step on weights, Prizma's state adapting
naturally, vs (a) frozen transformer, (b) frozen transformer + sliding window re-read of
equivalent memory footprint (the fair control: give the TF the same constant KB of context).
**WHY.** This is the one axis where the architecture *structurally* wins and the transformer
structurally cannot follow without O(n) memory: perpetual inference is what FEP says minds do
(Friston: organisms resist surprisal continuously [B]) and what nobody's flagship LM does at
constant memory. It is also where Prizma-CL's zero-forgetting evidence already points — the CL
thread proved task-free adaptation of modules; Seq can inherit it at the state level. The repo
culture rewards this: a pre-registered adaptation bar with param- AND memory-matched controls is
unimpeachable in a way "we're more brain-like" hand-waving is not.
**HOW.** Benchmark builder: distribution-switch streaming eval; metrics = pre-switch plateau
BPC, post-switch half-life of adaptation (tokens to recover ½ the gap), asymptotic post-switch
BPC. Controls: frozen TF @ same param, TF + w-token sliding window with w sized to match Prizma's
state bytes (the honest "you could just cache recent context" control), linear-attention and
DeltaNet arms (adaptation is a *family* property — Prizma must beat its own family to claim the
FEP framing adds anything, else the honest claim is "delta-family adaptation, free-energy-derived").
**FALSIFIABLE TEST.** Pre-register: Prizma post-switch half-life ≤ 0.5× the memory-matched
window-TF control's, and asymptotic gap ≥ 0.1 BPC in Prizma's favor, ≥3 switches, ≥3 seeds. If
the sliding-window TF matches adaptation, the claim downgrades to "same adaptation, less memory"
— still a win, honestly scoped.
**RISKS.** State contamination: after a switch, old associations may interfere (the CL thread's
interleaving-collapse warning applies at state level — frozen weights but plastic state can
catastrophically overwrite). Mitigation exists in-house: precision-gated writes should *rate-limit*
overwrites (connects to P2). Drift benchmark construction is easy to get wrong (token overlap
leakage) — use disjoint vocab segments.
**IMPACT 9 / FEASIBILITY 8.** [B: perpetual inference/homeostasis — Friston; continual state
adaptation literature; N: pre-registered O(1)-memory adaptation bar + free-energy-derived
rate-limiting — new framing, in-house mechanism]

---

### P5. Variational coupling of the threads: routing as Bayesian model reduction
**WHAT.** Prizma-CL routes by reconstruction surprise with per-expert (μ,σ). Recast the entire
expert pool as a *hypothesis space of generative models* m, and routing as **Bayesian model
reduction** (Friston & Hobert 2011; Friston et al. on structure learning [B]): maintain an
approximate posterior over experts p(m|x_{1:t}) ∝ p(x_{1:t}|m)p(m), updated online by each
expert's accumulated free energy, with the write/learn gain for expert m set by its posterior
mass. The (μ,σ) vigilance test becomes a likelihood-ratio test under this posterior — the
existing mechanism, re-derived rather than re-invented, with the addition of a proper prior
and posterior mass as the freeze criterion (freeze when p(m|x) > 1−δ, reawaken on posterior
collapse — the "reawakening" rule in Prizma.md §3 gets a probabilistic form).
**WHY.** It upgrades "recognition-by-reconstruction routing" (engineering) into "posterior over
generative models" (identity), yields calibration claims the current (μ,σ) heuristic can't make
(posterior mass should be calibrated; testable), and imports structure-learning results: the
posterior collapses automatically when a domain is mastered and re-expands on novelty — the
phase detector becomes a theorem-shaped object instead of a threshold heuristic.
**HOW.** Replace θ_m threshold logic with an online log-evidence accumulator per expert (cost:
one EMA per expert — the (μ,σ) EMAs already exist, they just get a probabilistic reading);
re-run E1–E5 exactly; add calibration plots (reliability of posterior mass vs actual domain).
**FALSIFIABLE TEST.** (a) E1 headline must be preserved (no regression vs 0.834/FGT=0 — a hard
gate: if the probabilistic form loses, keep the heuristic and record that). (b) NEW claim:
posterior-mass calibration — 95%-mass experts are right ≥95% of the time (reliability diagram);
the (μ,σ) heuristic makes no such prediction. (c) Interleaved-stream failure (the honest §8
limit) should *soften*: a posterior (unlike a threshold) can split mass across interleaved
domains; pre-register any improvement ≥ +0.05 ACC on the shuffled stream or report no change.
**RISKS.** Bayesian dressing of the same heuristic — the repo's anti-buzzword culture demands
the calibration test (b) as the real deliverable, not the vocabulary. Risk the accumulator is
just the EMA renamed — then say so and keep the honest ledger row "recast, not improved."
**IMPACT 6 / FEASIBILITY 8.** [B: Bayesian model reduction — Friston&Hobert 2011; N: online
evidence-accumulated expert routing with a calibration claim]

---

### P6. Langevin/temperature state: calibrated uncertainty as an output, not a byproduct
**WHAT.** Prizma.md P5 notes fixed-T Langevin breaks calibration. Make sampling temperature a
*precision readout*: T_t ∝ 1/π̂_t (per-token, from P2's estimated precision), so the LM samples
boldly where its state is confident and hedges where error variance is high — and, on ambiguous
inputs, the read itself carries uncertainty (the read o_t = S q_t gets an error bar from the
state's local prediction residuals). Deliverable: an LM that outputs calibrated next-token
entropy that tracks its own state-level residual error.
**WHY.** Transformers' calibration is post-hoc (temperature-scaled on a val set); a PP machine's
uncertainty is *native* — precision is computed for free energy anyway and the sampling rule
falls out. Brain-likeness: cortical variability tracks precision (Orbán et al. 2016, probabilistic
PP sampling [B]). Rival-value: calibrated streaming uncertainty is a product-capability (abstain,
handoff, OOD detection) at zero extra parameters.
**HOW.** Use P2's per-channel π; set decode temperature per step; measure ECE + AUROC of
state-residual-based OOD detection vs a temperature-scaled TF baseline on a drift benchmark
(reuse P4's benchmark).
**FALSIFIABLE TEST.** Pre-register: ECE ≤ 0.5× temperature-scaled-TF ECE on the drift benchmark,
and OOD AUROC ≥ 0.85 on corpus switches, without any post-hoc calibration on the eval
distribution. If post-hoc TF calibration matches Prizma's native calibration, the claim narrows
to "calibration without a calibration set" — say exactly that.
**RISKS.** Precision estimates early in a stream are garbage → warmup discipline needed.
Sampling-T coupling can hurt greedy-Decode BPC (keep as an opt-in decoding axis, gate off by
default — byte-identical discipline as usual).
**IMPACT 6 / FEASIBILITY 6.** [B: PP sampling, Orbán/Friston; N: precision-native decode
temperature with a no-post-hoc-calibration bar]

---

### P7. Minimum-description-length identity: "Prizma = online compression" as the rival thesis
**WHAT.** A positioning proposal, not a mechanism. Frame the entire architecture claim in the one
sentence FEP licenses: *a transformer memorizes a corpus into weights once; Prizma is an online
compressor — total description length = weights + accumulated state free energy — that never
stops compressing.* Then every existing result re-reads as a compression claim: param-efficiency
(≥3.5× on MQAR) = better weight-DL; O(1) state = bounded state-DL; continual learning = online
coding. The identity claim to pre-register: **on any corpus stream, Prizma's total code length
(weight bits + per-token −log p) is competitive with a matched transformer's at equal weights
and strictly better as stream length → ∞ under distribution shift** (the transformer pays the
same −log p forever, frozen; Prizma's state keeps improving it).
**WHY.** Free energy's deepest identity is with compression (−log p = code length; Friston's
"self-evidencing"; Solomonoff/MDL lineage [B]). It gives Prizma a rival thesis that is *aging-
proof*: transformers are batch compressors, Prizma is a stream compressor, and all intelligence-
as-compression arguments (Hutter, Solomonoff) sit on Prizma's side of the line. It is also the
honest answer to "is Prizma = active inference LM?" — no overclaim needed: the *state dynamics*
are literally free-energy descent; only the *action* half is heuristic (P3's job).
**HOW.** No code needed initially: write the identity doc (what is and is not claimed), then one
pre-registered streaming-code-length experiment piggybacking P4's drift benchmark (report
cumulative NLL per token, pre/post switch, vs matched TF and window-TF).
**FALSIFIABLE TEST.** Cumulative code length on a 3-corpus stream: Prizma < frozen-TF by a
pre-registered margin (≥0.05 bits/token averaged post-switches), with the window-TF control
bracketing the memory-fair comparison.
**RISKS.** Rhetoric risk — the repo punishes framing that outruns results; ship the framing doc
and the experiment *together*, never the doc alone. If streaming NLL doesn't beat frozen-TF, the
thesis dies quietly (that is fine; it cost one benchmark).
**IMPACT 7 (as framing + bar) / FEASIBILITY 8.** [B: MDL/Solomonoff, free-energy=compression
identity; N: streaming-code-length bar for LMs — new framing]

---

### P8. Sleep-phase consolidation: offline replay of the state's own generative model
**WHAT.** Brains consolidate offline (hippocampal replay, synaptic homeostasis — Tononi, Wilson &
McNaughton [B]). Give Prizma-Seq an optional "sleep" phase: between streams, re-descend the
*state* (not weights) on its own generated/reweighted token history — regenerate from the state
read, re-write with a low precision floor — distilling the state toward a fixed point that
preserves high-precision bindings and erodes low-precision ones (precision-weighted shrinkage:
exact FEP homeostasis, "minimize expected free energy under the null action"). At O(state) cost,
no labels, no corpus access.
**WHY.** Attacks the state-contamination risk in P4 (drift leaves stale bindings that interfere);
directly brain-like (sleep's function is exactly precision-weighted memory triage); and creates
a second-timescale object the transformer family simply lacks. Also links to Prizma-CL's
consolidation (ω_m) — the same one-signal-two-sign synthesis, now at state level in Seq.
**HOW.** Simulate: after a drift switch in P4's benchmark, run K sleep steps (self-generated
tokens from greedy decode, re-written with β scaled by current precision); measure post-switch
half-life with vs without sleep. Cost must be counted in the honest ledger (sleep FLOPs per
switch).
**FALSIFIABLE TEST.** Pre-register: sleep reduces old-domain BPC degradation after a switch by
≥30% at ≤10% extra FLOPs, with a no-sleep control and a random-replay-shrinkage control
(sleep must beat precision-blind shrinkage, else it's just forgetting).
**RISKS.** Self-generated replay can amplify model pathologies (mode collapse on its own samples).
Cheap to test, cheap to kill — a good falsifiable shape.
**IMPACT 5 / FEASIBILITY 6.** [B: replay/homeostasis — Tononi, Wilson&McNaughton; N: precision-
weighted state-distillation sleep for associative LM state]

---

### P9. The identity audit: is "Prizma = active inference LM" defensible?
**WHAT.** A verdict, offered to the committee as the final section's input. Scorecard of the FEP
components: generative model (YES — v≈Sk, explicit), free energy (YES — exact at the write),
precision (GESTATING — gate, not object; P2), hierarchical depth (NO — P1), action/expected FE
(NO — gate heuristic; P3), perpetual inference (STRUCTURAL YES, unused — P4), Bayesian model
selection (YES in CL — P5 deepens it). **Verdict: today Prizma is a *variational free-energy
state machine*, not yet an active-inference system; that is already strictly more FEP than any
mainstream LM and the claim is defensible ONLY in that exact wording.** "Active inference LM"
becomes defensible the day P3 (expected FE gate) or P4 (perpetual inference with a pre-registered
bar) lands. Recommended public phrasing until then: "an LM whose memory updates are exact
free-energy descent." The repo's culture makes this discipline load-bearing: one overclaim costs
the credibility that the honest-verdict brand has banked.
**IMPACT — / FEASIBILITY — (positioning).** [N: the scorecard itself]

---

## 2. Ranked summary

| Rank | Proposal | Impact | Feasibility | One-line |
|---|---|---|---|---|
| 1 | P4 Perpetual-inference adaptation bar | 9 | 8 | O(1) test-time adaptation — the structural moat, benchmarkable now |
| 2 | P2 Structured precision object | 8 | 7 | Derived gate; powers the owed surprise ablation honestly |
| 3 | P1 Deep predictive hierarchy | 9 | 5 | One-level → cortical hierarchy; biggest brain-likeness move, hardest |
| 4 | P3 Expected-FE write gate | 8 | 5 | Heuristic gate → principled epistemic foraging |
| 5 | P7 MDL/streaming-compression identity | 7 | 8 | The rival thesis that is aging-proof |
| 6 | P5 Bayesian model-reduction routing | 6 | 8 | CL routing re-derived; buys a calibration claim |
| 7 | P6 Precision-native calibration | 6 | 6 | Native uncertainty as a product capability |
| 8 | P8 Sleep-phase state consolidation | 5 | 6 | Second-timescale state triage; protects P4 |
| 9 | P9 Identity scorecard | — | — | Say "free-energy state machine" today; "active inference" only after P3/P4 |

Sequencing note: P4 + P2 are near-term and mutually reinforcing (precision rate-limits drift
overwrites); P1 and P3 are the deep bets; P7 is the framing that packages whatever wins.

## 3. What would make Prizma *inevitable* (this lens's answer)

A transformer is a brilliant solution to a problem FEP says no brain has: batch-optimized,
frozen, precision-blind pattern completion with unbounded sensory memory. The inevitable rival
is not "attention but cheaper" — the linear-attention family is already that, and Prizma-Seq
honestly sits in it (the repo says so itself). Inevitability comes from owning a capability axis
the incumbent *cannot reach without abandoning its core*: (1) O(1)-memory *perpetual inference*
with a pre-registered bar (P4); (2) *derived* precision — gates with optimality conditions you
can audit, not squashing functions (P2); (3) a compression identity that improves with stream
length instead of degrading (P7). Brain-likeness is not the marketing on top; it is the
mechanism source: hierarchy (P1), expected free energy (P3), sleep (P8) are cortical
computations first and capabilities second. The borrowed-vs-new ledger line this committee
should be able to write in three years: *the delta write is borrowed; the free-energy
**machine** around it — precision, hierarchy, action, sleep — is ours, each with its own
falsifiable bar.*

---

## QUESTIONS FOR THE OWNER

1. **Compute envelope:** P4/P2 need only CPU/A100-small runs, but P1 (hierarchy) and P3 (expected
   FE) need kernel rework (sequential coupling breaks chunk-parallel). Is a Triton-kernel budget
   realistic this cycle, or should proposals be ranked kernel-free first?
2. **Claim wording:** do you accept locking public phrasing to "free-energy state machine" until
   P3 or P4 passes its bar (P9), or do you want to keep "predictive-coding" unqualified?
3. **Scope priority:** P4's adaptation bar creates a NEW benchmark (drift streams). Build it
   in-repo (honest but more surface area) or piggyback on existing corpora switches only?
4. **CL↔Seq unification:** should P5's Bayesian-routing recast be validated on the CL thread
   first (cheap, E1–E5 rerun) before any Seq-side coupling, per the repo's one-change-at-a-time
   discipline?
