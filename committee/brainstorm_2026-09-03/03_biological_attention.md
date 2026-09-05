# 03 — Biological Attention & Saliency Theorist

**Commission:** Prizma 14-scientist brainstorming commission, round 2026-09-03
**Author role:** Biological Attention & Saliency Theorist (thalamic gating, superior
colliculus priority maps, feature-similarity attention models, working-memory buffers,
feature-integration theory, top-down vs bottom-up attention)
**Classification (per brainstorming SKILL.md):** **Architectural-class
research-direction brainstorm.** Output is a report only — no code, no spec-to-plan
transition. All owner decisions are deferred to "QUESTIONS FOR THE OWNER" (no live
conversation available).
**Repo state read:** `README.md`, `docs/PRIZMA_SEQ_REPORT.md`, `docs/Prizma.md`,
`seq/prizma_seq.py` (config + mixer surface), `src/prizma.py` (vigilance routing).

---

## 0. Framing: does the brain do anything like softmax attention?

Almost certainly **no** — and the disanalogies are instructive, because each one points
at a mechanism Prizma already half-owns or could own cheaply.

| Transformer softmax attention | Brain |
|---|---|
| Global, dense, content-based lookup over a growing KV cache | Selective **routing/gating** through a hub (thalamus: <2% of cortical input passes relay; Sherman & Guillery 2002, "thalamic relay functions as a gate") |
| Every token can attend to every token | **Priority/salience maps** that are spatially/featurally structured and winner-take-more (superior colliculus: Fecteau & Munoz 2006; feature-similarity gain model: Treue & Martínez Trujillo 1999) |
| Softmax = a global normalizing competition over the whole cache | **Divisive normalization** over local pools (Heeger 1992; Carandini & Heeger 2012) — competition is a gain-control operating point, not a permutation over memory |
| Memory = KV cache (a verbatim trace of the past) | Working memory = **structured state buffers** with limited slots, scheduled writes, and decay (Baddeley; frontal-parietal WM buffers; Persistent activity + synaptic consolidation views) |
| Write = implicit; every token is stored | Write = **scheduled by salience/surprise** — novelty- and reward-gated encoding (hippocampal novelty gating: Ranganath & Rainer 2003; locus coeruleus phasic surprise → encoding boost: Aston-Jones & Cohen 2005) |
| Attention output = normalized dot product read | Read = recognition by **reconstruction** against a template with gain modulation (Prizma's `o = S q` read is *already* closer to this than softmax is) |

**The thesis of this report:** Prizma-Seq's decomposition — (i) a carried associative
working-memory state `S`, (ii) a precision/surprise-gated delta *write*, (iii) a small
local window (the "fovea"), and (iv) in Prizma-CL, a vigilance-threshold *router* — is
already the correct **skeleton of biological attention**. What it lacks to become a
*complete* brain-style attention substitute (and a Transformer-beater on recall) is:
a priority-map layer that decides **where in the state to write and read**, a
normalization-based readout instead of raw dot products, explicit decay/refresh of
state slots, and a gating hub that replaces the global-lookup role softmax plays.
Each proposal below is one such organ, transplanted with a falsifiable bar in the
repo's own culture.

**Central answer to the commission's question:** yes — Prizma's window head + delta
state + vigilance router can plausibly evolve into a complete attention substitute,
but *not by adding a softmax-like global lookup anywhere*. The biological route is
**structured state + scheduled writes + normalized competitive reads**, and the
recall-task wins should come from *write scheduling* (what gets into `S`) and
*read normalization* (what comes out), which is exactly where transformers are
weakest at O(1) memory.

---

## 1. Proposals (ranked by expected value)

Scoring: IMPACT (1–10) on the "inevitable rival + brain-like" twin goals;
FEASIBILITY (1–10) at the repo's current scale and honesty bar. Each proposal is
marked **Borrowed** (with citations) or **New/Adapted**.

---

### P1. Salience-scheduled state writes ("the brain writes what is surprising")
**Rank 1 — the single highest-leverage move.** IMPACT 9 / FEASIBILITY 8.

- **WHAT:** Replace the learned input-gated write `β_t = σ(W_β x_t)` with a write
  drive composed of a **bottom-up salience/surprise term** (normalized prediction
  error `‖ε_t‖ = ‖v_t − S_{t-1}k_t‖`, the quantity the delta rule already computes for
  free) plus a **top-down relevance term** (a slow signal from the router/task context,
  cf. Prizma-CL vigilance). Write strength `β_t ← f(ε̃_t, τ_t)` where `ε̃` is
  divisively normalized surprise and `τ_t` is a low-pass "relevance" gate.
- **WHY:** The current `surprise_gate` lever (Lever A in `prizma_seq.py`) is
  implemented but its evidence is INCONCLUSIVE (repo's own ablation note, n=2 smoke).
  The biological literature strongly predicts the mechanism only works when surprise
  is **normalized against the running error distribution** (LC phasic responses are
  invariant to raw magnitude; Aston-Jones & Cohen 2005, adaptive gain theory) and
  **modulated by top-down relevance** (Ranganath & Rainer 2003). Two knobs the current
  Lever A lacks. This is also the honest fix to the open question the report itself
  flags: "whether the surprise/precision signal used causally helps — NOT something
  B6 answered."
- **HOW:** (a) EMA of surprise per head → z-score/divisive-normalize; (b) multiplicative
  relevance gate trained from a cheap auxiliary "token will be queried" signal
  (trainable on MQAR where queries are known); (c) ablation ladder
  `{input-only (current), +surprise, +normalized-surprise, +relevance, full}` with
  the repo's `surprise_mode='random'/'constant'` controls extended; pre-register
  MQAR D=128 @130K solve-rate and selective-copy as the bar, n≥5 seeds. Prediction
  to falsify: normalized-surprise writes store fewer tokens at equal recall →
  measured as **write-sparsity at matched accuracy**.
- **FALSIFIABLE TEST:** At matched solve-rate ≥0.9 on MQAR D=128 and selective-copy,
  the full scheduled-write arm must show ≥30% reduction in total write mass
  `Σ‖β_t ε_t‖/‖ε_t‖` vs input-gated baseline; if it matches accuracy only by
  writing everything, the salience story is dead for this architecture.
- **RISKS:** Surprise-gating already trended *against* the mechanism in the existing
  inconclusive ablation — this could fail again; two-pass surprise needs the pre-write
  ε before β is applied (chunk-parallel training form may break — a real engineering
  cost the repo has disclosed before).
- **Borrowed/New:** Borrowed mechanisms (adaptive gain: Aston-Jones & Cohen 2005;
  novelty-gated encoding: Ranganath & Rainer 2003; divisive error normalization:
  Carandini & Heeger 2012); **New** as a *composition* with the delta write and as a
  causal write-mass metric.
- **Impact 9 / Feasibility 8.**

---

### P2. Divisive-normalization readout from S (competitive, non-softmax retrieval)
**Rank 2.** IMPACT 8 / FEASIBILITY 8.

- **WHAT:** Change the state read from `o_t = S_{t-1} q_t` to a two-stage read:
  (1) raw similarity scores `s = S q`, (2) **divisive normalization within
  learned pools** of state columns: `o = s / (c + Σ_{j∈pool(i)} s_j^γ)`, with per-head
  pools and an exponent γ (sublinear γ<1 = winner-take-more, the superior-colliculus
  regime, vs γ→∞ = winner-take-all).
- **WHY:** Biological attention is *not* a softmax — it is gain control (Carandini &
  Heeger 2012 normalization model; Reynolds & Heeger 2009 normalization model of
  attention). Winner-take-more dynamics suppress distractor traces in `S` without the
  global-renormalization pathology of softmax (which is what makes attention need its
  whole cache). This gives Prizma a *brain-grounded* answer to "what replaces
  softmax?": **nothing does — local divisive pools over a fixed-size state do.** It is
  also cheap: `S` is d_h×d_φ, pools cost O(d_φ).
- **HOW:** Add `readout_norm ∈ {none, rms (existing state_norm), divisive}` with
  pool-size and γ as config; pre-register: MQAR D=128 @130K (does normalization
  reduce crosstalk and hence raise solve-rate at the *lean* d_φ=137?), induction
  head retention at 8× length, and a **distractor-robustness bar** the repo doesn't
  yet have (MQAR with adversarial near-key distractors — where softmax's global
  renormalization is known to hurt and WTA should shine). Predicted falsifiable
  signature: divisive readout widens the capacity margin (Gershgorin `N < 1+1/cross(φ)`
  bound effectively loosened by suppressing sub-threshold traces).
- **FALSIFIABLE TEST:** On distractor-MQAR at D=128, divisive-WTM readout must beat
  `state_norm=RMS` readout by a pre-registered margin (e.g. ≥5 points solve-rate,
  n≥5) at matched params; else the normalization story is decoration.
- **RISKS:** Normalization can erase genuinely multi-trace answers (MQAR needs *two*
  associations retrieved and summed — too-sharp WTA could hurt exactly where quad2
  won); pool structure is a new hyperparameter with overfitting risk at small scale.
- **Borrowed/New:** Borrowed theory (Reynolds & Heeger 2009; Carandini & Heeger 2012;
  WTA/priority maps: Fecteau & Munoz 2006); **New** application to linear-associative
  state readouts — no efficient-attention work I know uses divisive-normalized state
  reads as a *capacity* lever.
- **Impact 8 / Feasibility 8.**

---

### P3. Thalamic gating hub: replace "global attention" with a two-layer relay
**Rank 3 — this is the architectural identity move.** IMPACT 9 / FEASIBILITY 5.

- **WHAT:** Insert a **thalamic-style gating layer** between pairs of Prizma blocks
  (or between state read and FFN): a bottleneck router that (a) computes a per-token
  **relay permission** `ρ_t ∈ [0,1]^H` (which heads' state-reads pass to the output),
  (b) optionally **re-addresses** the query (`q̃ = q + W r_t` where `r_t` is a slow
  top-down context vector accumulated across blocks — the cortical feedback signal),
  and (c) passes only a sparse winning subset of channels onward. The thalamus is
  literally a GATING hub, not a memory: relay cells are modulated (first-order) or
  drive-controlled (higher-order) by cortical feedback (Sherman & Guillery 2002;
  Saalmann & Kastner 2011 pulvinar attention routing).
- **WHY:** This answers the commission question directly: the brain never does
  content-based lookup over everything; it does **content-based permission-to-pass**
  over a small state. Prizma already has the state; what stands in for softmax's
  "select-what-matters" role should be a *gating* circuit over heads/channels, not a
  lookup. It also gives a natural home to top-down attention (the missing half of the
  biological story): the pulvinar's routing is **feedback-modulated**, and a slow
  context vector re-aiming queries across layers is the cheapest faithful analog.
- **HOW:** Add config `thalamic_gate: bool` with three pre-registered ablation arms:
  gate-only (ρ), gate+feedback-readdressing (ρ+r), and a shuffled-ρ control (the
  `precision_gate='random'` trick the repo already has). Bar: param-matched MQAR
  D=128 + char-LM text8 with the **full pre-registration discipline** (the repo's
  B4 lesson — both corpora, n≥3) to show the gate earns its params.
- **FALSIFIABLE TEST:** Gate-off must equal current model byte-identically (repo
  culture: `off == identical` guard, which prizma_seq.py already does for its levers
  — extend it). Gate-on must beat param-matched baseline by pre-registered margin on
  MQAR *and* not regress char-LM beyond the +0.05 margin. Shuffled-ρ control must
  fail (proves the gate is content-driven, the direct analog of the repo's
  `route_readout=False` noRouteReadout ablation).
- **RISKS:** More learned gating on top of W_β, out-gate, erase gate → gate zoo
  (DeltaNet already has 3 gates); risk of "gating soup" where no ablation is
  interpretable. Mitigate by one gate at a time with the repo's lever discipline.
  Also this is where the ~5× slow-training problem could worsen.
- **Borrowed/New:** Borrowed: thalamic gating (Sherman & Guillery 2002; Saalmann &
  Kastner 2011; HG BLT paper's LLM-thalamus analogy: Liu et al. 2024 "HLNet");
  **New:** the specific composition with pre-write delta state and the
  feedback-readdressed query.
- **Impact 9 / Feasibility 5.**

---

### P4. Priority-map state augmentation (a salience field co-evolving with S)
**Rank 4.** IMPACT 7 / FEASIBILITY 6.

- **WHAT:** Maintain, alongside each head's associative state `S`, a tiny
  **priority vector** `p ∈ R^{d_φ}` (a priority map over state columns, the analog
  of the superior colliculus map): `p_t = λ p_{t-1} + ‖u_t‖` (where `u_t = β_t ε_t`
  is the written trace), decaying at rate λ. Reads are modulated by `p`
  (features that were strongly written recently get read-gain — feature-similarity
  gain: attending *to a feature* boosts all channels carrying it, Treue & Martínez
  Trujillo 1999); writes to high-priority columns are decayed *slower* (protected
  traces) — i.e., **priority modulates the forget gate per state-dimension**, which
  the `gated`/`inctx_lr` per-channel machinery (Lever G) already has a hook for.
- **WHY:** Working memory in the brain is not a flat buffer — it has structured
  priority (Stokes 2015 "activity-silent prioritized WM"; Griffin & Nobre priority
  cues). Prizma's `S` is the WM buffer; it lacks the *prioritization field* that
  determines what decays and what is refreshed. The per-channel in-context learning
  rate η (Lever G, already implemented, mutually exclusive with surprise_gate) is
  exactly the right substrate: **set η per state-column from p instead of from x.**
- **HOW:** `p` costs d_φ extra floats/head — memory story intact. Reuse Lever G's
  per-channel η path with `η_t = β_cap · σ(W_p [x_t ; p_t])` vs current
  `η_t = σ(W_η x_t)`; ablation decides whether the priority field earns the bytes.
  Bar: selective-copy with **long interleaved filler** (where "protect the marked
  tokens" is precisely a priority problem) and length-extrapolation retention.
- **FALSIFIABLE TEST:** On long-filler selective-copy, priority-augmented η must
  beat input-only η at matched params/bytes by pre-registered margin; the
  `p`-shuffled control must collapse to input-only performance.
- **RISKS:** Lever G is scoped to n_delta=1 (code constraint); two novel-core levers
  colliding (the repo's own "silent footgun" note); λ adds a timescale hyperparameter.
- **Borrowed/New:** Borrowed: priority maps (Fecteau & Munoz 2006; Stokes 2015);
  feature-based gain (Treue & M-T 1999); **New:** priority as a *per-state-column
  write-strength field* co-evolving with the associative state — I know of no
  efficient-attention work doing this.
- **Impact 7 / Feasibility 6.**

---

### P5. Working-memory decay & refresh (consolidation dynamics on S)
**Rank 5.** IMPACT 6 / FEASIBILITY 7.

- **WHAT:** Give `S` an explicit **two-timescale WM lifetime**: a fast-decaying
  "active" component and a slow "consolidated" component (or, cheaper: a learned
  data-dependent decay gate — the `gated` flag already exposes GDN's α, currently
  off — driven by *priority/salience*, i.e., irrelevant context decays fast,
  task-relevant traces are refreshed). Biologically: synaptic consolidation +
  activity-silent WM (Stokes 2015; Mongillo et al. 2008 synaptic theory of WM).
- **WHY:** The brain's WM does not keep everything for free and does not softmax
  over a cache; it **actively forgets** and re-instantiates from silent traces.
  Prizma's constant-state claim is only fully honest if it can *also* claim
  controlled forgetting — right now unattended history lingers in `S` and causes
  exactly the distractor crosstalk that the FLOP ledger's capacity bounds quantify.
  Decay is also the causal fix for length-extrapolation (Prizma's ~0.40 absolute
  accuracy at 8× train length: a decay gate that adapts its timescale to content is
  the brain-flavored remedy).
- **HOW:** Enable + extend `gated` (α per-channel, driven by the P4 priority vector
  or by x). Pre-register length-extrapolation retention and long-filler selective-copy.
  Cheap, existing hook, honest memory story unchanged.
- **FALSIFIABLE TEST:** Adaptive-decay arm must improve 8×-length retention
  relative to fixed-decay by pre-registered margin AND not degrade short-context
  MQAR (a decay that helps long context by forgetting cannot pay for itself in
  recall regression).
- **RISKS:** The `gated` lever is known-good in GDN (borrowed) so novelty is thin —
  the novelty is *salience-driven* decay, which must be ablated against plain GDN
  decay or the borrowed-vs-new ledger is violated.
- **Borrowed/New:** Borrowed: GDN decay gate (Yang et al. 2024); Mongillo 2008,
  Stokes 2015 for the WM framing; **New:** salience-gated (not input-gated) decay.
- **Impact 6 / Feasibility 7.**

---

### P6. Foveal sharpening: make the window head a genuine saliency-selected fovea
**Rank 6.** IMPACT 5 / FEASIBILITY 7.

- **WHAT:** Reframe (and lightly upgrade) the w=16 exact local-window head as the
  **fovea**, and add a *saccade* mechanism: a per-token low-resolution "peripheral"
  signal (pooled from `S`'s read error) selects, for a small fraction of tokens, an
  **extra offset window** in the recent past to inspect exactly (inhibition-of-return
  style: don't re-inspect the same offset twice — superior colliculus saccade
  targeting with IOR, Fecteau & Munoz 2006; Findlay & Walker 1999).
- **WHY:** The brain's high-resolution access is narrow (fovea) but *actively
  repositioned* by a priority map; the transformer's local windows are fixed. This
  preserves exactness and O(1)-ish cost (a few extra bands) while making local
  access content-driven rather than position-driven — a genuinely brain-like middle
  path between "small window" and "global softmax."
- **HOW:** Extend `banded_window` to a *set* of (offset, width) bands chosen per
  token by a cheap argmax over a pooled salience signal; cap at k=2–3 bands. Bar:
  induction heads with distant noise; MQAR with keys near but not in w=16.
- **FALSIFIABLE TEST:** Saccadic-band arm must beat fixed w=64 (param/FLOP-matched)
  on offset-induction; if a single fixed wide window matches it, the saccade
  mechanism fails its reason to exist.
- **RISKS:** Band selection must be exact/deterministic to keep the repo's
  exactness culture; hardware-efficiency of irregular bands is a real kernel issue.
- **Borrowed/New:** Borrowed: saccade targeting/IOR (Fecteau & Munoz 2006); sparse
  banded attention kernels (Longformer-family); **New:** salience-selected
  per-token band composition over an exact banded kernel.
- **Impact 5 / Feasibility 7.**

---

### P7. Vigilance router → sequence-model novelty routing (unify the two Prizmas)
**Rank 7.** IMPACT 7 / FEASIBILITY 4.

- **WHAT:** Prizma-CL's ART-style vigilance routing (route on reconstruction
  surprise against a *committed expert*, recruit-on-novelty) is the most brain-like
  component in the repo and currently lives only in the continual-learning thread.
  Port the *principle* to Prizma-Seq: a small set of K delta-state "compartments"
  per layer with vigilance-based allocation of tokens (or segments) to compartments
  — a structured multi-slot WM instead of one flat buffer, with novelty-gated
  recruitment (hippocampal pattern separation; Carpenter & Grossberg 1987 ART;
  Kumaran & McClelland 2012 complementary learning systems).
- **WHY:** Biological WM has *multiple structured buffers* with competition between
  them, not one associative matrix. This also attacks the true ceiling of
  single-state models: capacity `N < 1+1/cross(φ)` — K compartments give K× the
  capacity at K× state (still O(1) in n) while softmax attention's cost stays O(n).
- **HOW:** Start with K=2–4 heads-of-states, route per *segment* (not per token) on
  mean reconstruction surprise; vigilance τ adaptive like Prizma-CL. Bar: MQAR at
  D=256/512 where single-state capacity provably binds — the D-frontier sweep the
  repo deferred (P3) becomes the natural arena.
- **FALSIFIABLE TEST:** At D=256, single-state arm must fail (capacity bound) while
  the K=4 compartment arm at ~4× state (still O(1), param-matched to a TF that
  passes) must pass; otherwise compartments are just params.
- **RISKS:** Hardest engineering lift here (segment routing, K-state kernels);
  novelty-recruitment dynamics are notoriously unstable to train with backprop —
  and the backprop-free variant is an open frontier, not a claim. Feasibility low
  this cycle.
- **Borrowed/New:** Borrowed: ART vigilance (Carpenter & Grossberg 1987), CLS
  (Kumaran & McClelland 2012), soft MoE routing (Shazeer 2017 — the non-biological
  cousin); **New:** vigilance-based *state-compartment* routing for sequence mixers.
- **Impact 7 / Feasibility 4.**

---

### P8. Feature-integration test bed: the binding problem as Prizma's benchmark moat
**Rank 8 (cheap, high signaling value).** IMPACT 5 / FEASIBILITY 9.

- **WHAT:** Add a **feature-binding benchmark suite** to `seq/`: tasks where the
  answer requires *conjunctive* recall (feature A × feature B bound at position t —
  the classic feature-integration-theory motivation, Treisman & Gelade 1980):
  MQAR variants where keys are conjunctions, "conjunctive MQAR," and a
  pop-out-vs-conjunction search task in token space. Run the existing quad2 arms vs
  the TF.
- **WHY:** FIT says binding is done by *spatially indexed attentional highlighting*
  — sequentially, via a WM buffer — not by parallel global lookup. The quad2
  monomials are literally a **conjunction detector** (products of features); if the
  biological story is right, Prizma should show a *disproportionate* advantage on
  conjunctive vs single-feature recall, and this is a cheap, novel, publication-grade
  diagnostic that no efficient-attention paper runs. It operationalizes "why a
  quadratic feature map is the brain's conjunction solution."
- **HOW:** Pure benchmark work, zero mixer changes; reuse `run_cell` scaffolding;
  pre-register the conjunctive:single-feature advantage ratio for quad2 vs none vs TF.
- **FALSIFIABLE TEST:** Quad2's advantage ratio on conjunctive tasks must exceed its
  advantage on standard MQAR by a pre-registered factor; if not, the "quadratic =
  binding" narrative is disconfirmed and should be retired from the story (honesty
  culture).
- **RISKS:** None mechanical; risk is a null result — which this repo should want.
- **Borrowed/New:** Borrowed: FIT (Treisman & Gelade 1980); **New:** conjunctive
  recall diagnostics for linear-attention feature maps.
- **Impact 5 / Feasibility 9.**

---

## 2. The integrated architecture sketch (what "Prizma as biological attention" looks like)

Layer structure at maturity (order of adoption per feasibility):

```
token x_t ──► short conv (fovea pre-processing)
   │
   ├─► P1 salience-normalized surprise ──┐
   ├─► P4 priority map p (col-level)  ───┤
   │                                     ▼
   │                    delta WRITE into S  (β from surprise×relevance;
   │                    per-column η from p; decay from P5 salience gate)
   │                                     │
   ├─► w=16 fovea (P6: + salience-selected saccade bands)
   │                                     │
   ▼                                     ▼
  read: S q ──► P2 divisive-WTM normalization ──► P3 thalamic gate ρ (+feedback q̃)
                                                          │
                                              (P7 vigilance compartments upstream)
```

Each organ has a **single analog in the brain** (salience gate = LC adaptive gain /
novelty gating; priority map = superior colliculus; WTM readout = cortical divisive
normalization; thalamic gate = pulvinar routing; compartments = hippocampal-CLS
buffers), and each is individually falsifiable under the repo's lever discipline.
That is exactly the property the Transformer lacks: no biological mapping of softmax
exists at any level of description that survives scrutiny — softmax is a global
renormalization with no gain-control, no priority structure, no write scheduling,
and an unbounded verbatim cache.

## 3. Borrowed-vs-new ledger (summary)

| Mechanism | Borrowed from | Status |
|---|---|---|
| Surprise/adaptive-gain writes | Aston-Jones & Cohen 2005; Ranganath & Rainer 2003 | borrowed principle, new composition |
| Divisive normalization / WTM readout | Carandini & Heeger 2012; Reynolds & Heeger 2009 | new application to state reads |
| Thalamic gating/routing hub | Sherman & Guillery 2002; Saalmann & Kastner 2011 | new as an LM gating layer |
| Priority maps | Fecteau & Munoz 2006; Stokes 2015; Treue & M-T 1999 | new as per-state-column field |
| Decay/refresh WM | Mongillo 2008; Stokes 2015; GDN gate (Yang 2024) | borrowed lever, new driver |
| Saccades/IOR fovea | Findlay & Walker 1999; Fecteau & Munoz 2006 | new over exact banded kernels |
| Vigilance compartments | Carpenter & Grossberg 1987; Kumaran & McClelland 2012 | new for sequence mixers |
| Conjunction/binding diagnostics | Treisman & Gelade 1980 | new benchmark family |

All citations are from memory and standard in the field; the owner should spot-check
DOIs before publication use.

## 4. QUESTIONS FOR THE OWNER

1. **Softmax abstinence:** Do you accept the thesis that Prizma should *never* grow a
   global softmax-like lookup, and instead treat the small w=16 window as the fovea
   and gating as the selectivity mechanism? Or is there a strategic reason to keep a
   global-lookup escape hatch (e.g., for long-range induction at scale)?
2. **Priority of P1 vs P2:** Both are cheap; P1 finishes the repo's own open question
   (surprise causality), P2 adds a new readout. One A100 campaign next — which?
3. **Write-sparsity as a pre-registered metric:** Is "total write mass at matched
   accuracy" an acceptable headline metric for the biological story, or do you want
   it confined to ablation tables (it invites "sparsity ≠ brain" referee objections)?
4. **P7 compartments:** This is the biggest capacity unlock and the biggest
   engineering risk. Fund it this cycle (with the deferred P3 D-frontier as its arena)
   or defer until P1/P2 land?
5. **Training-cost budget:** Thalamic gating (P3) and two-pass surprise (P1) may
   worsen the ~5× slow-training problem. What wall-clock budget is acceptable before
   a proposal is capped?
6. **Binding-benchmark moat (P8):** May I register the conjunctive-recall suite as a
   *pre-registered* diagnostic in the §4 bar family (it is cheap and could become the
   field's standard if quad2 wins it — or be honestly retired if not)?
7. **Cross-committee:** Proposals P1/P4/P5 share the priority/salience substrate and
   will collide with any memory-systems or predictive-coding member's proposals —
   should a joint substrate be co-owned rather than each member proposing variants?
