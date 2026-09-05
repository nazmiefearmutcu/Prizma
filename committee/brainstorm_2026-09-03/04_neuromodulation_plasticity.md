# Committee Report 04 — Neuromodulation & Synaptic Plasticity

**Author role:** Neuromodulation & Synaptic Plasticity Expert
(dopamine RPE; ACh/vigilance & ART; NE/uncertainty gating; STDP, eligibility traces,
three-factor rules, metaplasticity, BCM, Fusi/Benna consolidation cascades)
**Date:** 2026-09-03 · **Scope:** Architectural-class research-direction brainstorm.
**HARD-GATE honored:** report only — no code was written or changed in the repo.

---

## 0. Framing: why the learning rule is Prizma's decisive battleground

A Transformer's learning rule is *frozen at birth*: SGD/backprop through a frozen computation
graph, one global learning rate (at best per-parameter Adam moments), one timescale. Nothing in
the architecture *is* a learning rule; plasticity is imposed from outside the substrate. The brain
is the opposite: its substrates **are** plastic media whose write rules are gated, timescaled, and
metaplastic *by construction* — dopamine, acetylcholine, noradrenaline and serotonin are
architectural signals, not optimizer hyperparameters.

Prizma already has the substrate property in both threads:

- **Prizma-Seq:** the delta write `u_t = β_t·(v_t − S_{t−1}k_t)` is a *two-factor local rule*
  (prediction error × precision gate) that is simultaneously the forward computation. The state
  `S` is not a cache — it is a plastic synapse matrix updated online, every token, for free.
- **Prizma-CL:** `dW ~ (Π·ε)(x)·r` with an eligibility trace `Tr`, a metaplastic `ω_m`,
  an ART-vigilance router, and a documented "one surprise energy `E_m`, two opposite readouts"
  sign-tension resolution (docs/Prizma.md §2.3) — which is *literally* an ACh/DA dissociation
  (ACh = attention/precision readout, DA-plasticity-window readout), arrived at independently.

**The critical angle:** a Transformer *structurally cannot* claim (a) a write rule that is also
the inference dynamic, (b) neuromodulator-style global scalars that broadcast without
weight-transport, (c) synapse-level multi-timescale memory without a KV cache that grows with n.
Everything below is aimed at making that claim *measured*, not rhetorical — under the repo's
culture of pre-registered falsifiable bars and param/FLOP-matched baselines.

**Borrowed-vs-new discipline:** each proposal marks citations. The repo's honest-ledger rule
applies: borrowed mechanisms stay labeled borrowed; novelty must live in the synthesis and be
causally ablated.

---

## 1. The neuromodulator → Prizma-signal mapping (the shared dictionary)

| Neuromodulator | Canonical role | Prizma-Seq signal | Prizma-CL signal | Status today |
|---|---|---|---|---|
| **Dopamine (RPE)** | phasic δ = outcome − prediction gates plasticity | the delta write *is* an RPE: `ε_t = v_t − S_{t−1}k_t` | `b_m = π‖ε_m‖²` surprise energy | intrinsic; **unexploited as a *temporal-difference* signal across tokens/batches** |
| **Acetylcholine (ACh)** | vigilance, cortical state, sharpens PT error & plasticity on novelty (ART: mismatch → θ rises → recode; Yu & Dayan's "expected uncertainty") | none | vigilance θ on recon surprise; precision `Π_m` readout | present but **static / per-expert-adaptive only** |
| **Noradrenaline (NE)** | phasic burst on uncertainty/OOD → global gating reset ("network reset", Aston-Jones; uncertainty-scaled gain, Ullsperger) | write gate `β_t = σ(W_β x_t)` — *learned, but not uncertainty-derived* | route/novelty detector | `surprise_gate` lever exists; its own ablation is INCONCLUSIVE (results/gpu_ablation.json, n=2 smoke) |
| **Serotonin (5-HT)** | timescale of plasticity, behavioral inhibition, timescale re-scaling | state decay gate `α_t` (gated) | `ω_m` consolidation rate | fixed constants, not modulated |
| **Metaplasticity / BCM** | sliding modification threshold θ_m; synapses' "plasticity of plasticity" | none | `α = α₀/(1+ω_m)` per expert | per-expert only; **not per-head, not per-synapse, not in Seq** |
| **Consolidation cascade (Fusi/Benna)** | cascade of states with geometric timescales → power-law memory | `S` has a single (optionally gated) timescale | freeze/`ω` binary consolidation | absent |

The gaps in the right-hand column are the proposal space.

---

## 2. Ranked proposals (WHAT / WHY / HOW / FALSIFIABLE TEST / RISKS / SCORES)

### P1. Dynamic Modulatory Vigilance (ACh as a *state*, not a threshold) — RANK 1

- **WHAT:** Replace Prizma-CL's (effectively fixed / per-expert-EMA) vigilance with a *globally
  modulatory, dynamic* θ(t): a slow integrator of recent global novelty rate
  (fraction of inputs rejected by all committed experts), bounded in [θ_min, θ_max], with
  hysteresis. High novelty pressure *lowers* θ (recruit aggressively); sustained recognition
  *raises* θ (require tighter match before committing a new expert). Per-expert vigilance then
  becomes θ(t) × expert-specific margin, modulated, not stored.
- **WHY:** ART's core claim (Carpenter & Grossberg 1987) is precisely that vigilance is a
  *modulatory parameter under arousal control*, not a constant — "vigilance can rise when the
  world proves unfamiliar." Prizma-CL already borrows ART routing but keeps the vigilance
  quasi-static; the docs admit the fixed-vigilance mistake caused the E2 noise=0.3 collapse and
  was patched with the per-expert (μ,σ) phase detector. The dynamic-θ formulation *subsumes*
  that patch and directly attacks the biggest open limit: the **interleaved-stream collapse
  (ACC ~0.58)**, which happens because a single static vigilance regime must serve both
  contiguous-block and mixed streams.
- **HOW (mechanism, no code here):** θ_{t+1} = clip(θ_t + η_θ·(ρ_t − ρ*), θ_min, θ_max), where
  ρ_t = EMA of novelty fraction and ρ* a target novelty rate (~1/K for K expected domains — but
  ρ* itself can be self-estimated from the novelty-rate autocorrelation). The existing
  precision-phase detector stays as the *local* signal; θ(t) is the *global* modulator — exactly
  the ACh "arousal × local mismatch" division of labor in the ART literature.
- **FALSIFIABLE TEST (pre-register):** Structured-permuted, K=5, interleaved/shuffled stream,
  ≥10 seeds. **Bar:** dynamic-θ Prizma ACC ≥ 0.70 (vs ~0.58 static) with FGT ≤ 0.05, CI-separated
  from the static-vigilance arm, expert count within 2×K. Also re-run E2 noise sweep: bar = no
  sharp collapse at any noise level (ACC drop between adjacent noise levels ≤ 0.15). Failure =
  θ oscillates or routes multi-expert-per-domain: report honestly.
- **RISKS:** θ dynamics add one hyperparameter family (η_θ, bounds); oscillation under
  adversarial novelty-rate injection; may improve interleaved ACC while *degrading* the
  contiguous-block headline. Mitigation: hysteresis + report both regimes always.
- **Impact 9 · Feasibility 9.** Borrowed: ART vigilance dynamics (Carpenter & Grossberg),
  Yu & Dayan 2005 (ACh = expected uncertainty). New: the mapping onto reconstruction-surprise
  routing with a novelty-rate integrator, and the claim that it removes the block-contiguity
  assumption.

### P2. Eligibility-Traced Three-Factor Sequence Learning (decouple write from update) — RANK 2

- **WHAT:** Give Prizma-Seq a per-head **eligibility-trace state** `E_t` alongside `S_t`:
  `E_t = λ·E_{t−1} + φ(k_t)⊗v_t` (or trace the error product). The state update becomes
  **three-factor**: `S_t = α_t·S_{t−1} + β_t·ε_t⊗φ(k_t) + γ·M·E_{t−1}`, where `M` is a *delayed,
  broadcast scalar* (see P4) arriving many tokens after the writes it credits. The write and the
  credit are decoupled in time by the trace horizon τ_e.
- **WHY:** This is the single strongest "transformers structurally cannot" claim available.
  A Transformer assigns credit only by BPTT through its frozen graph — credit assignment and
  computation are the same mechanism, and both vanish at inference (the KV cache never learns).
  Prizma-Seq's `S` is a live synapse; adding traces converts the *delta write* (two-factor,
  Hebbian-at-inference) into a genuine **three-factor rule** (Frémaux & Gerstner 2016:
  pre ⊗ post ⊗ neuromodulator) that can receive delayed feedback **without backprop**. The
  Prizma-CL doc already defines `Tr` and never wired it into the sequence thread — this is the
  missing bridge, and it is the only route to a *backprop-free sequence trainer* claim, which is
  currently listed as an open frontier nobody in the DeltaNet family can touch.
- **HOW:** Traces cost one extra d_h×d_φ state per head (or a low-rank factorization of it to
  keep the O(1) budget honest — flag the param/FLOP ledger impact explicitly; a matched-TF
  comparison must charge the trace). The delayed modulator `M` can be (a) a downstream
  reconstruction/RPE on a later token, (b) a chunk-level free-energy delta (see P6), or (c) an
  externally delivered reward in a bandit-style probe task.
- **FALSIFIABLE TEST (pre-register):** (i) Credit-assignment probe: a delayed-consequence MQAR
  variant where the informative write precedes the supervisory signal by T_del ∈ {0, 32, 128}
  tokens. **Bar:** trace-Prizma solves at T_del=128 where trace-ablated Prizma and a
  param-matched TF trained with truncated BPTT window < T_del both fail. (ii) Backprop-free arm:
  train the full stack with the three-factor rule only (no autograd on the mixer) and report the
  gap vs BPTT on B1–B4. Honest bound: if the gap is large, the claim shrinks to the probe result.
- **RISKS:** trace state doubles mixer memory (ledger must disclose); τ_e tradeoff (long traces =
  noise accumulation, the classic animal-level problem); could degrade standard benchmarks if
  γ>0 always-on (gate it: γ=0 must be byte-identical, per repo culture).
- **Impact 10 · Feasibility 5.** Borrowed: eligibility traces & three-factor rules (Gerstner &
  Kistler; Frémaux & Gerstner 2016; Izhikevich 2007 DA-credit). New: carrying a trace *inside an
  O(1)-state attention-replacement mixer* and using it for delayed, transport-free credit.

### P3. Benna–Fusi Consolidation Cascade for the State S (multi-timescale memory, zero extra params budget) — RANK 3

- **WHAT:** Replace the single `S` per head with a small cascade `S^{(1)}…S^{(K)}` (K=3–5):
  writes enter `S^{(1)}` only; a *consolidation flux* (analog of cascade down-transfer, Benna &
  Fusi 2016; Fusi et al. 2005) slowly copies `S^{(k)} → S^{(k+1)}` with geometrically increasing
  timescales; reads combine all compartments (read weights fixed, e.g. equal or 2^{−k}).
- **WHY:** Two payoff axes. (1) *Architecture:* power-law forgetting from discrete exponential
  compartments is the brain's trick for spanning milliseconds-to-lifetime with synapse-level
  variables — it would directly attack the length-extrapolation weakness (Prizma's absolute
  retention ~0.40 @8×) and the char-LM gap by keeping slow task-level structure while the fast
  compartment stays plastic for local context. (2) *Continual:* in Prizma-CL the same cascade
  replaces the binary freeze (ω → consol threshold) with graded consolidation — "the expert
  doesn't freeze, its fast compartment does," which is biologically truer (no brain circuit
  hard-freezes) and removes the fragile phase-detector threshold.
- **HOW:** K compartments = K× mixer memory unless compartment k+1 is low-rank (recommended:
  full-rank fast, rank-1 or rank-2 slow compartments — the slow tail carries summary structure,
  and low rank is exactly what a consolidated gist should be). Charge honestly in the FLOP ledger
  (flop_ledger.py already parameterizes this kind of accounting).
- **FALSIFIABLE TEST (pre-register):** (i) Length-extrapolation leg: retention at 8× train
  length. **Bar:** cascade ≥ 0.40→≥0.55 absolute with param-matched single-S control, ≥5 seeds.
  (ii) Forgetting-curve shape: measure recall accuracy vs intervening-distractor length on MQAR;
  **bar:** the accuracy-vs-log-delay curve is better fit by a power law (cascade) than exponential
  (single-S), AIC-separated, per Benna-Fusi's signature prediction. (iii) Prizma-CL: graded
  cascade vs hard-freeze on the E2 noise sweep — bar: no degradation, ≥ parity on E1.
- **RISKS:** FLOP/memory overhead is the honest cost (K compartments); read-combination weights
  could become a tuned hyperparameter (keep fixed/random, per repo's buffer-culture like quad2);
  consolidation flux may smear fast context needed for induction-head behavior — B2 must not
  regress; add B2 to the gate.
- **Impact 8 · Feasibility 7.** Borrowed: Benna & Fusi 2016 / Fusi, Drew & Abbott 2005.
  New: cascade inside a delta-state attention-replacement head; graded (non-binary) expert
  consolidation in Prizma-CL.

### P4. NE: Uncertainty-Derived (not learned-static) Write Gating — RANK 4

- **WHAT:** Re-derive the write gate `β_t` from an *epistemic uncertainty* signal rather than
  (only) `σ(W_β x_t)`: cheap options in strict locality order — (a) state-norm surprise:
  ‖S_{t−1}φ(k_t)‖ vs its own EMA (recognition strength, i.e. NE tonic level); (b) read-variance:
  variance of `S_{t−1}q` under the Langevin noise the hardware doc already treats as a sampler
  (uncertainty = posterior spread, Yu & Dayan's NE = *unexpected* uncertainty); (c) the existing
  `surprise_gate` tanh(‖ε‖) as the phasic component. Compose: tonic gain × phasic burst
  (Aston-Jones: tonic mode = gain control, phasic burst = engage-and-reset).
- **WHY:** The repo's own honest ledger notes the surprise-gating ablation is **INCONCLUSIVE
  (n=2 smoke, point estimates against the mechanism)**. This is the right response: not to
  quietly drop the neuromodulation story, but to *power the experiment and diversify the signal*.
  A learned σ(W_β x) gate cannot be claimed as "brain-like" — it is just another linear layer;
  only a *state-derived* gate makes the NE claim structural. If state-derived gating matches the
  learned gate, Prizma deletes parameters and gains the biological story for free.
- **HOW:** Keep `precision_gate='input'` as one arm; add `'uncert_ema'` and
  `'uncert_langevin'` arms; all gated-off defaults byte-identical (existing knob culture).
- **FALSIFIABLE TEST (pre-register):** ≥10 seeds, MQAR D=128 + selective-copy + char-LM(text8).
  **Bar:** an uncertainty-derived gate matches the learned gate within +0.05 BPC / 1% recall
  (CI), AND beats `uniform` and `random` controls CI-separated (the real falsification risk —
  this is exactly what R8/R9 was built to test and underpowered). Second, sharper bar: on
  *noisy/OOD-injected* streams the uncertainty gate outperforms the learned gate (NE should win
  precisely under unexpected uncertainty — specify the injection protocol in advance).
- **RISKS:** the current adverse point estimates may be real (norm-based gating may double-count
  what β already encodes); Langevin-variance is unavailable when noise is off. Honest fallback:
  if all state-derived gates fail the powered test, *write the negative result into the ledger*
  and demote NE from "mechanism" to "analogy" — that too is a repo win.
- **Impact 7 · Feasibility 8.** Borrowed: Yu & Dayan 2005; Aston-Jones & Cohen 2005;
  uncertainty-gated plasticity (Lim, Wichert et al.). New: state-intrinsic uncertainty readouts
  (EMA-norm, noise-variance) as gate *sources* in a delta-state mixer.

### P5. BCM + Synaptic Metaplasticity at Head Level (plasticity of plasticity, in both threads) — RANK 5

- **WHAT:** Promote `ω_m` from expert-level to (a) per-head in Prizma-CL (heads within an expert
  consolidate at different rates — some features of a domain are stable, others keep moving) and
  (b) into Prizma-Seq as a per-head sliding modification threshold θ_h (BCM, Bienenstock–Cooper–
  Munro 1982): each head's effective write magnitude is normalized by its own recent activity,
  so heads that "fire too much" (rich-get-richer, the MoE failure mode in the iteration log v1)
  self-throttle, and heads that fire too little get a plasticity window before dying.
- **WHY:** The iteration log documents exactly the pathology BCM was invented for: v1 soft-MoE
  collapsed to uniform; dead-expert pressure needed a hand-built load-balancer (`θ_m ← θ_m +
  η_b(usage − target)`). BCM's sliding threshold *is* that fix, derived rather than bolted on,
  and extends it to any future head-level plasticity. ATransformer has nothing at this level:
  its "plasticity" (optimizer state) is per-parameter but *input-experience-blind in the
  biological sense* — it never re-tunes its own learnability from its own firing statistics at
  inference.
- **HOW:** per-head EMA of write energy; effective β scaled by BCM curve φ(EMA/θ_h) with θ_h
  sliding. Same machinery Prizma-CL already runs at expert granularity — this is mostly a
  *granularity* promotion, which is why feasibility is high.
- **FALSIFIABLE TEST (pre-register):** (i) Seq: arm = BCM-normalized β on char-LM; **bar:**
  within +0.05 BPC of default AND improved robustness of early training (fewer diverged seeds —
  pre-register the divergence criterion, e.g. grad-norm spike count). (ii) CL: per-head ω vs
  per-expert ω on E1/E2; **bar:** ACC parity + strictly better expert-count adaptation in E3
  (experts=5 on K=4 domains: one expert splits its heads across two domains — a qualitative,
  pre-registered check).
- **RISKS:** another normalization can destabilize training; per-head bookkeeping cost; risk of
  overclaiming "metaplasticity" for what is adaptive normalization — keep the ledger word
  "BCM-style".
- **Impact 7 · Feasibility 8.** Borrowed: BCM 1982; metaplasticity (Abraham & Bear 1996);
  Aitchison et al. Bayesian synapses (already in the CL ledger). New: head-granularity
  promotion; sliding threshold as the derived (not bolted) load-balancer.

### P6. Dopamine as Temporal-Difference of Free Energy (the global NM made an RPE) — RANK 6

- **WHAT:** Prizma-CL's master functional F gives, per batch, an energy `E_t`. Define the global
  neuromodulator scalar as the **TD of surprise**: `NM_t = E_{t−1} − E_t` (improvement, i.e.
  positive RPE over its own free energy), EMA-smoothed, gating the plasticity window
  `window(b_m)`. Phases: NM>0 sustained = "learning is working" (keep windows open);
  NM≈0 with high E = "stuck AND surprised" (novel domain — open recruitment);
  NM≈0 with low E = mastered (freeze). Three regimes from one scalar pair (E, dE) — the classic
  DA swing from RPE sign (Schultz, Dayan & Montague 1997).
- **WHY:** The doc's sign-tension fix reads E in two static signs; the TD version adds *temporal
  context*, which is what dopamine actually carries, and yields a principled, parameter-light
  phase detector that could replace/augment the (μ,σ) precision detector — the component the
  referee report identified as the real achievement.
- **HOW:** E is already computable; this is a readout change, ~zero new params. Combine with P1
  (θ modulation) — NM and ACh form the canonical DA/ACh dissociation.
- **FALSIFIABLE TEST (pre-register):** phase-detection accuracy on streams with known (synthetic)
  phase structure: **bar:** TD-based detector ≥ (μ,σ) detector's routing purity (100%) on E1 and
  degrades less under the E2 noise sweep, CI-separated from a random-phase-detector control;
  on interleaved streams, TD+dynamic-θ (P1) ≥ either alone (2×2 factorial ablation — cheap and
  decisive).
- **RISKS:** E is batch-dependent (batch-size sensitivity must be controlled); EMA horizon sets a
  confound with the contiguity assumption; risk of *no improvement* over (μ,σ) — which is a
  publishable negative only if powered.
- **Impact 7 · Feasibility 9.** Borrowed: Schultz RPE; Friston free energy; "learning progress"
  (Oudeyer & Kaplan intrinsic motivation). New: NM = dF/dt as the consolidation driver replacing
  hand thresholds.

### P7. Serotonin: Volatility-Adaptive State Timescale (α/τ as a 5-HT analog) — RANK 7

- **WHAT:** Estimate environmental volatility online (hazard rate of regime change — e.g.,
  an EWMA-CUSUM on the novelty rate from P1) and modulate the decay/forget gate: volatile →
  faster decay + wider plasticity windows (forget the stale world); stable → slower decay +
  consolidation (keep the world). One global scalar modulating `α_t` (Seq) and `ω` dynamics (CL).
- **WHY:** 5-HT's cleanest computational mapping is timescale/plasticity-rate control under
  change (e.g., Lottem et al. 2018; Doya's meta-learning of learning rates under a hazard-rate
  Bayesian model — Adams, MacKay 2007 style). The gated variant (`gated=True`) already exposes
  α_t as a lever; making it *environment-modulated* instead of purely input-learned is the
  neuromodulatory move, and gives Prizma a principled answer to "what happens at distribution
  shift mid-stream" — currently a failure mode the docs handle only implicitly.
- **HOW:** hazard estimator on per-head or global ε statistics; map hazard to a multiplier on
  the decay gate and on Prizma-CL's `α₀`. Defaults off = byte-identical.
- **FALSIFIABLE TEST (pre-register):** synthetic hazard streams: domains whose true switch rate
  changes between blocks (known ground-truth from construction). **Bar:** hazard-modulated arms
  achieve lower cumulative free energy than fixed-α arms on volatile blocks without losing on
  stable blocks beyond +0.02, ≥10 seeds, CI-separated; report the estimated-hazard vs
  true-hazard correlation. Failure mode (estimator lag) must be reported.
- **RISKS:** estimator lag/false alarms on short blocks; another moving part on a system whose
  fragility history (E2 collapse) argues for minimalism; interaction with P1's integrator must
  be ablated (they share the novelty-rate statistic).
- **Impact 6 · Feasibility 7.** Borrowed: hazard-rate optimal control (Adams & MacKay 2007;
  Doya 2002), 5-HT timescale accounts (Lottem et al. 2018). New: hazard-modulated state decay
  in a delta-state mixer + expert consolidation.

### P8. Three-Factor Rule with Free Energy Itself as the Third Factor (formalization + hardware) — RANK 8

- **WHAT:** Not a new mechanism but the *unifying formal statement* the repo is one paragraph
  away from: both threads already compute every factor of a local three-factor rule
  (`pre = φ(k)/f(z)`, `post = ε`, `modulator = Π/β/NM`), and the neuromodulator is *the model's
  own free-energy error* — no external reward. Write the paper-grade formulation:
  `ΔS = η·NM(F, dF)·[post ⊗ pre]` with NM a function of precision-weighted surprise and its
  delta; prove (or verify numerically, as the FD check did for the two-factor case, 5e-10) that
  this is ascent on F under stated conditions; map each factor to one analog-physics substrate
  element from §5 of docs/Prizma.md.
- **WHY:** This is the load-bearing *theory deliverable*: "the third factor is intrinsic"
  distinguishes Prizma from every RL-gated architecture (where the third factor must come from
  outside the network — Gerstner's fourth-factor debate) and from the Transformer (no third
  factor at all). It is also what a neuromorphic reviewer needs: each factor = one device physics.
- **HOW:** document + numerical verification harness, no architecture change. Cite Gerstner &
  Kistler three-factor reviews; Izhikevich; the repo's own PBWM borrowing.
- **FALSIFIABLE TEST:** numerical: FD verification of the three-factor ascent (<1e-8) across
  noise levels; empirical: show turning off any one factor collapses learning (the existing
  noRoute ablation is the modulator-off arm — complete the matrix: pre-off, post-off are
  degenerate by construction, state this honestly and test only non-degenerate ablations).
- **RISKS:** purely formal — risk of overclaiming novelty ("it's just the PC gradient
  re-branded"); the honest framing is *locality + intrinsic modulator*, not a new optimizer.
- **Impact 8 · Feasibility 9.** Borrowed: the rule family; New: the intrinsic-F NM synthesis and
  its hardware factor map.

---

## 3. Cross-cutting warnings from this lens

1. **The ablation seed problem recurs.** The R8/R9 surprise-gate ablation was n=2 and
   inconclusive; P4 is exactly that experiment powered. Do not add P5/P7 neuromodulatory levers
   to the S3 ablation matrix until each has its own powered gate — the S3 "exactly one novel-core
   lever" guard exists because of this class of bug.
2. **Modulator sharing = silent coupling.** P1 (θ), P6 (NM), and P7 (hazard) all want the
   novelty-rate statistic. Compute it once, centrally, with one EMA spec — otherwise three levers
   will each build a subtly different integrator and ablations become uninterpretable.
3. **Locality budget.** Every proposal above preserves no-weight-transport. The trace state (P2)
   and cascade (P3) are the only memory-growing additions — both must appear in
   `flop_ledger.py` before any headline cites them.
4. **Biology as falsifier, not decoration.** Each mapping table entry should carry a one-line
   "animal-level falsifier" (e.g., if vigilance were static, ART would not show list-strength
   recovery; if NE gating were input-only, uncertainty-injection experiments would show no
   difference). This keeps the brain-likeness claim empirical.

## 4. Borrowed-vs-new ledger (this report)

| Mechanism | Source | Status |
|---|---|---|
| Dynamic vigilance / arousal control | Carpenter & Grossberg 1987; Yu & Dayan 2005 | borrowed (P1) |
| Eligibility traces, three-factor rules | Gerstner & Kistler; Frémaux & Gerstner 2016; Izhikevich 2007 | borrowed (P2, P8) |
| Cascade consolidation, power-law memory | Fusi, Drew & Abbott 2005; Benna & Fusi 2016 | borrowed (P3) |
| NE uncertainty gating / network reset | Yu & Dayan 2005; Aston-Jones & Cohen 2005 | borrowed (P4) |
| BCM sliding threshold, metaplasticity | Bienenstock et al. 1982; Abraham & Bear 1996 | borrowed (P5) |
| Dopamine RPE, learning progress | Schultz et al. 1997; Oudeyer & Kaplan | borrowed (P6) |
| Hazard-rate timescale control | Adams & MacKay 2007; Doya 2002; Lottem et al. 2018 | borrowed (P7) |
| Trace-in-mixer delayed credit without backprop | — | **NEW** (P2) |
| State-intrinsic uncertainty readouts as gate sources | — | **NEW** (P4) |
| Graded (cascade) expert consolidation replacing hard freeze | — | **NEW synthesis** (P3-CL) |
| NM = TD of own free energy driving vigilance+consolidation | — | **NEW synthesis** (P6, with P1) |

---

## QUESTIONS FOR THE OWNER

1. **Memory budget doctrine:** P2 (trace) and P3 (cascade) both grow the O(1) state constant.
   Is a 2–4× larger *constant* state acceptable if it buys backprop-free credit assignment and
   power-law retention — or is "smallest constant" itself a pre-registered identity of the
   architecture?
2. **Continuity of the CL thread:** should Prizma-CL's binary freeze (ω → consol) be *replaced*
   by the graded cascade (P3-CL), or kept as the ablation baseline? Replacing touches the
   headline E1 numbers and would require re-running the referee gate.
3. **Interleaved-stream priority:** is breaking the contiguous-block assumption (via P1+P6) a
   first-class goal worth a new pre-registered benchmark leg, or out of scope for the next
   campaign?
4. **Negative-result protocol for P4:** if powered uncertainty-gate experiments again fail to
   beat controls, do we demote the neuromodulator *mapping* to "inspiration" in the README, or
   keep it as hypothesis with the negative result attached?
5. **Claim ceiling:** for P2's backprop-free sequence training arm, what gap vs BPTT would the
   owner consider *still worth publishing* (e.g., ≤0.2 BPC at small scale), so the bar can be
   pre-registered before we run it?
