# Brainstorm — Neuromorphic & Edge-Hardware Systems lens
## (memristor crossbars, Loihi-class event-driven compute, in-memory computing, always-on edge AI, energy-per-inference economics)

Author: committee member 12 of 14 (Neuromorphic & Edge-Hardware Systems Engineer). 2026-09-03.
Scope interface: report `08_scaling_and_hardware_efficiency.md` owns GPU training throughput and
scaling-campaign economics. This report owns the *other* hardware frontier: deployment onto
analog/in-memory and event-driven substrates, and the always-on edge story. Where the two touch
(the 5x training gap), I defer to 08 and only note dependencies.

Classification: **Architectural-class research-direction brainstorm. HARD-GATE honored: no code
was written; everything below is design, analysis, and pre-registration proposals.** Numbers are
labeled: **[MEASURED]** = committed repo result (`results/`), **[ARITHMETIC]** = analytic
identity or arithmetic on disclosed parameters (not a measurement of any device). External
citations are from the author's knowledge — re-verify before README/paper use (marked
**[CITE-CHECK]**).

---

## 0. Executive frame

`docs/Prizma.md` §5 is the repo's first neuromorphic mapping. This brainstorm's job is to turn
that summary into a *deployment program* with the repo's culture: pre-registered falsifiable
bars, honest limits, no fabricated hardware numbers, an explicit borrowed-vs-new ledger.

The one-sentence thesis: **Prizma's two structural assets — a fixed-size state that is written and
read in place, and a learner whose plasticity is local and surprise-gated — are exactly the two
things analog/in-memory and event-driven hardware are good at, and exactly the two things the
Transformer's append-only KV-cache + global-backprop pipeline are structurally worst at.**
But the claim must be surgically stated (§2 steels the Transformer hard), and every hardware claim
must land as either **[ARITHMETIC]** with disclosed parameters or as the output of a
pre-registered emulation bar — never as a simulated-hardware "measurement" (the repo has burned
itself once on simulated rows; that must never recur in this lens, where the temptation is
strongest).

The single most valuable idea in this report (H1): **Prizma's consolidation mechanism is
simultaneously a learning rule and an endurance-management policy.** "Freeze an expert when its
domain passes" — read as an algorithm — is zero forgetting; read as hardware — it is "stop writing
conductances to that tile," which saves an endurance-limited memristor array. This is not an
analogy; it is one variable (write activity) with two consequences — structural and checkable in
the committed code (`src/prizma.py::train_batch` returns before any write once a committed expert
recognizes the batch).

---

## 1. Baseline — what the repo has actually established, and what it has not

**Established (build on these):**
- **[MEASURED]** Prizma-Seq inference is constant-memory and O(1)-per-token: `results/gpu_latency.json`
  shows `state_floats` constant across n=4k..64k (147,456 floats, small cell) and a measured
  per-step latency crossover (TF 16.95 ms vs Prizma 7.06 ms at n=65,536; Prizma 1.3–1.5x slower
  below ~16k). README quotes 17.9 MB state vs 1.25 GB KV-cache at 65k for the 50M pair
  (28–455x). The KV figure is an **[ARITHMETIC]** closed-form identity
  (`seq/scaling_analysis.py`), not an allocation measurement — the README says so.
- **[MEASURED]** The CL thread needs **no backprop at all**: decoder/head use the local PC/delta
  rule, encoder uses fixed-random feedback (DFA) — and the W^T-free variant *outperformed* the
  W^T one (0.834 vs 0.708, docs E4). No weight transport anywhere.
- **[MEASURED]** Routing is structurally sparse: one expert per domain; committed experts are never
  retrained; a recognizing committed expert short-circuits *before any write* (`train_batch`).
- **EXISTS, smoke-tested only:** the analog-degradation hooks — `weight_bits`, `act_bits`,
  `noise_in_std`, `noise_act_std`, `noise_weight_std` — are exercised by
  `tests/test_analog_neuromorphic.py` (7 tests: noise ordering, quantized-inference degradation
  ordering, QAT convergence, PC settling, DFA settling, Langevin sampling). These are per-expert
  smoke tests. **The full E1 campaign has never been run under any degradation** (H3 closes this).
- **EXISTS, contested:** the surprise-gated write lever (`surprise_gate`; Lever A) is OFF by
  default; report 11 (T4) argues the accuracy motivation is aimed at the wrong gate. This report
  *re-bases* that lever on hardware grounds, where the motivation does not depend on the
  contested accuracy claim (H4).
- **EXISTS:** a documented honesty precedent — simulated-CUDA rows were deleted with a public
  correction (README §3). This lens adopts that precedent as binding for all energy/endurance
  claims.

**Not established (must never be implied):**
- No emulation results beyond the smoke tests; no energy measurement or energy *arithmetic*
  anywhere; no spiking/event-driven code; no mapping beyond docs §5's qualitative table; no
  quantized state-S experiment (the hooks quantize weights/activations, not the recurrent
  state); no wear/endurance analysis.
- Nothing above ~1.4M params has been trained; no large-scale LM parity exists (README). The
  deployment story below describes the *mechanism class*, not "Prizma-70B on a memristor."

---

## 2. The honest wedge — steelman first

### 2.1 Steelman of the Transformer (what we must NOT claim)

1. **"Transformers can't run on edge" is FALSE.** Quantized 1–8B transformers run on phones and
   laptops today; short-context edge inference is a solved engineering problem. If our story
   requires the edge to be transformer-free below 16k context, it is dead on arrival.
2. **KV-cache compression narrows the constant factors.** 2–4-bit KV quantization (KIVI,
   Liu et al. 2024 **[CITE-CHECK]**), cache eviction (H2O, SnapKV **[CITE-CHECK]**), and
   sliding-window hybrids (Mistral/Gemma **[CITE-CHECK]**) cut KV memory and read traffic by
   ~4–16x. The *asymptotic* O(T)-bytes and O(T)-reads-per-token survive (some global attention
   must still read a cache that grows with context), but the honest crossover moves — Prizma's
   own measured latency crossover is n≥32k *today, at small scale* **[MEASURED]**. Below that,
   the transformer wins. Our wedge must live at long context and in always-on regimes, or
   nowhere.
3. **Spiking/Loihi attention exists.** Spikformer, SpikeGPT and successors
   (Zhou 2023; Zhu 2023 **[CITE-CHECK]**) run attention-family models on neuromorphic
   substrates. "Transformers cannot follow Prizma onto this hardware" is **false as stated**.
   What survives is narrower and still strong: (a) *training* still happens on GPUs with
   surrogate-gradient backprop — no deployed neuromorphic stack *continues learning on-chip* the
   way Prizma-CL's local rules would; (b) the KV-cache is *append-only* state — every token adds
   conductances and every token reads all of them — structurally hostile to endurance-limited
   analog cells and event-driven sparsity, whereas a *rewritable, bounded* state is friendly.
   The invariant that survives the steelman: **cache (append-only, global-read) vs state
   (in-place, local-read) — not transformer vs RNN.**
4. **On-device adaptation for transformers exists** (periodic, supervised on-device
   LoRA/fine-tuning pipelines). Prizma-CL's differentiation is not "can learn on device" but
   *continuous, unsupervised, boundary-free adaptation with zero backprop infrastructure* —
   adaptation as a running process, not a maintenance event. A real product asymmetry
   (always-on personalization with no training cluster), sold as a capability difference, not
   an impossibility theorem.
5. **Analog hardware punishes everyone.** Device variability, conductance drift, 1/f (RTN) noise
   (not white — `docs/Prizma.md` §5 already admits this), limited ADC/DAC precision and the fact
   that ADCs dominate crossbar area/power (ISAAC's own analysis; Chen et al. 2016
   **[CITE-CHECK]**). Prizma does not get a pass; the only honest question is whether its
   algorithmic structure *tolerates* these impairments better than alternatives — which is an
   empirical claim (H2, H3), not a birthright.

### 2.2 The wedge that survives the steelman — three claims we are allowed to want

- **W1 (data movement):** Per-token bytes moved grow O(context) for the transformer vs O(1) for
  Prizma-Seq. Since data movement, not arithmetic, dominates inference energy (Horowitz, ISSCC
  2014 **[CITE-CHECK]**), the long-context energy ratio inherits the README's 28–455x memory
  ratio *in a direction we can state*, with magnitude depending on per-byte energy — computed by
  the disclosed-parameter ledger (H5), never asserted.
- **W2 (always-on learning):** Prizma-CL trains with four local factors and no weight transport;
  there is no transformer-stack equivalent that adapts continuously without a backprop pipeline.
  The neuromorphic fit is qualitative today (docs §5); H3/H6 give it falsifiable content.
- **W3 (endurance structure):** Write traffic is the scarce resource on analog hardware. Prizma
  writes (i) a bounded state S every step (Seq) and (ii) only the active expert's weights (CL;
  zero writes to frozen experts). The KV-cache demands per-token writes that grow with context.
  **[ARITHMETIC]** consequences of this write asymmetry are the closest thing this lens has to
  an impossibility argument against "KV-cache on memristors" — stated with disclosed endurance
  parameters in H1, and explicitly NOT against "KV-cache on SRAM" (fine digitally; that is the
  steelman in §2.1-2).

Everything below turns W1–W3 into pre-registered, hardware-free-or-cheap programs.

---

## 3. Proposals (ranked)

Ranked by impact × feasibility × fit to repo culture (cheap, falsifiable, no hardware required).
**[B]** = borrowed (cited), **[N]** = new, **[B→N]** = borrowed component, new synthesis.

### H1. The two-state memory-technology assignment: S on volatile compute-in-memory, W on RRAM — the consolidation gate *is* the endurance gate  **[B→N]**  — IMPACT 9 / FEASIBILITY 8

**WHAT.** A mapping note (a `docs/` design document, not code) assigning Prizma's two memory
classes to two technologies: the recurrent delta state **S** (written every token, volatile,
rewritten forever) goes on SRAM-CIM / DRAM-CIM / digital SRAM (unlimited-endurance write class);
the learned weights **W** (written only during active learning, then frozen) go on RRAM
conductance cells (endurance-limited, non-volatile, stable at rest). Then formalize that
Prizma's consolidation (`frozen=True`) halts all writes to a tile — the learning mechanism
doubles as the wear-leveling policy.

**WHY.** **[ARITHMETIC]** (parameters disclosed, arithmetic not measurement): RRAM endurance is
commonly in the ~1e6–1e9 write-cycles band depending on technology (Ielmini & Wong 2018
**[CITE-CHECK]**). Prizma-Seq writes all of S once per token; at 100 tok/s a 1e6-endurance
technology exhausts in ~2.8 hours — S on RRAM is a non-starter, S on volatile CIM is fine
(unlimited writes). Conversely Prizma-CL writes only the active expert's W; a frozen expert's
tiles consume *zero* endurance forever (structural, checkable in `src/prizma.py`). If S shared
endurance-limited cells with W, the story collapses; the split is what makes the mapping
physically coherent. The brain parallel is real and borrowed: labile-to-stable synaptic
consolidation (Frey & Morris 1997; Fusi, Drew & Abbott 2005 **[CITE-CHECK]** — the repo already
cites Fusi/Benna-Fusi for metaplasticity).

**HOW.** (1) A `docs/neuromorphic/` mapping note: per-tensor technology table — Wenc/Wdec/Wcls →
RRAM tiles; the fixed FA matrices Bdec/Bcls → burned-in at fabrication, *never* written (a fit
no backprop model enjoys); S + ring buffers → SRAM-CIM or digital SRAM beside the crossbars.
(2) A write-traffic ledger: writes/token per tensor from the committed code paths (CL: 0 frozen /
>0 active; Seq: S every token, W never at inference) — **[ARITHMETIC]**, disclosed units.
(3) An endurance section under a "parameters, not measurements" banner. (4) Named risks: wear
concentrates on the active expert's tiles (logical→physical remapping = classical wear-leveling,
borrowed); RRAM read noise on the state read S·φ(q) feeds H2's emulation; ADC precision bounds.

**FALSIFIABLE TEST.** The mapping is falsified as *coherent* if any required operation violates
its technology class: e.g., if a committed expert must ever be rewritten (it must not — E1 code
shows it doesn't), or if S's update cannot be expressed as a bounded set of CIM ops (it can: a
rank-1 outer-product write + a global decay scale). Concretely: an independent auditor takes the
mapping note plus `src/prizma.py`/`seq/prizma_seq.py::step` and tries to break the assignment;
the note ships only if no break is found — the referee discipline the CL thread already used.

**RISKS.** Pure-design impact decays without H2/H5 making it testable; SRAM-CIM is
area-expensive per bit vs RRAM cells — disclosed as an open area trade; DRAM-CIM literature is
younger (mostly projected numbers **[CITE-CHECK]**).

### H2. "The delta rule is analog-robust": pre-registered state-quantization emulation bar for Prizma-Seq — delta vs additive write under 4–6-bit state  **[N]** (QA methods borrowed; the fitness *comparison* claim is new)  — IMPACT 8 / FEASIBILITY 9

**WHAT.** The first falsifiable experiment of this lens, zero hardware: extend the existing
emulation discipline (`weight_bits`/`act_bits` + the `step()==forward()` <1e-4 guard in
`seq/prizma_seq.py`) to the **recurrent state S**: quantize S to b ∈ {4,5,6,8} bits after every
write (`S = q(S)`), inject write noise of disclosed sigma, optionally clamp to conductance
range. Compare the two write modes the repo already implements — `write_mode="delta"` vs
`"additive"` (linear attention) — on MQAR / induction / selective-copy at the committed
parameter-matched configs.

**WHY.** The delta rule writes the prediction error itself (`u = β(v − S·k)`): each write
*corrects* the stored association against the state's current content — an error-correcting
memory write; additive linear attention can only accumulate. Hypothesis H2a: **under state
quantization and drift, delta degrades gracefully while additive collapses earlier** — the delta
update is not just constant-memory but *hardware-fitter* for analog state. Counter-hypothesis
H2b: the correction requires precisely reading S·k, so a quantized S corrupts the residual and
delta degrades as fast or faster. Either outcome is informative; H2a would be, to the author's
knowledge, unclaimed in the delta-rule/linear-attention literature (Gated DeltaNet, DeltaNet
**[CITE-CHECK]** — those papers do not test recurrent-state quantization) and a
*deployment-grade* differentiator that accuracy parity alone never provides.

**HOW.** (1) A state-quantization hook along the `step()` path (S quantized post-write; reads see
the quantized S — write-and-read both degraded, the honest emulation). (2) Pre-register the bar
BEFORE running at scale: "delta MQAR parity at b-bit state within x% of its own FP32 accuracy,
with additive at matched bits ≥y% worse, n≥3 seeds, non-overlapping CIs" — exact numbers frozen
by the owner before the run, as in every repo campaign. (3) Reuse the committed MQAR/induction
harness and param-match discipline; (4) raw JSON under `results/` with the machine-readable
config-fingerprint resume discipline (the contamination lesson, already fixed).

**FALSIFIABLE TEST.** The bar itself. If delta ≤ additive at matched state bits (H2b), the claim
dies and the repo records that — the honest outcome costs a CPU-week. Secondary: if delta only
survives at ≥8 bits, the "analog-fit" story retreats to digital-CIM precision (fine, but say so).

**RISKS.** Quantization-aware *training* would confound — run inference-time quantization first,
QAT optionally later; surprise/inctx levers stay OFF (byte-identity discipline); MPS/CPU float
nondeterminism needs the existing <1e-4 tolerance discipline.

### H3. The analog-degradation battery on E1: run the full continual-learning campaign under quantization + device noise + stuck-at faults, pre-registered  **[B]** (hooks are the repo's own; battery design borrowed from fault-injection practice; the pre-registration is new process)  — IMPACT 7 / FEASIBILITY 9

**WHAT.** The hooks exist (`weight_bits`, `act_bits`, three noise channels) and are smoke-tested
per-expert; the E1 campaign (structured-permuted, 10 seeds, FGT/ACC) has never been run under
them. Extend the fault model with what analog hardware does that the Gaussian hooks do not:
(i) **device variability** — a fixed per-synapse offset drawn once at init (persistent
miscalibration, not i.i.d. per-read noise); (ii) **stuck-at faults** — a yield parameter p% of
cells frozen at min/max conductance; (iii) **1/f-style noise** (Lorentzian-corner approximation
— docs §5 admits real RTN is not white). Then run E1 + E2 under a pre-registered grid.

**WHY.** docs/Prizma.md §5 claims analog fit; §8 honestly lists device variability and RTN as
idealizations; the only way the claim becomes evidence is this battery — and it is the cheapest
proposal here: pure CPU, reuses the committed harness, upgrades smoke tests into a *result*.
A second hypothesis worth pre-registering: routing reads *reconstruction surprise* (a difference
of two analog reads), so moderate noise may shift absolute surprise while preserving the
*ranking* across experts — a mechanistic reason why this learner suits analog: **comparison
survives where representation degrades**.

**HOW.** (1) Pre-register the grid: weight_bits ∈ {4,6,8,None} × act_bits ∈ {4,6,8,None} ×
variability σ ∈ {0, 2%, 5%} × stuck p ∈ {0, 0.5%, 2%} (factor-reduced; owner freezes the exact
lattice first); (2) success bar analogous to E1's — FGT stays 0 and ACC within a pre-registered
tolerance of FP32 across the mid grid, with routing accuracy (expert-assignment confusion)
reported SEPARATELY from ACC (the E2 lesson: ACC falls with noise for everyone; clean routing is
the real signal); (3) results in `results/analog_battery.json`, same raw-JSON discipline.

**FALSIFIABLE TEST.** The pre-registered grid bar. Routing-confusion-rate under 5% variability
is the most diagnostic number: if committed experts start mis-recognizing each other's domains,
FGT=0 collapses and the analog story is dead at that operating point — an honest, cheap
refutation.

**RISKS.** Negative results at aggressive degradation (4-bit) are likely and must be framed as
*frontier mapping*, not failure; any colored-noise generator is a disclosed modeling choice;
scope creep — freeze the lattice before running.

### H4. Surprise-gated writes, re-based: from a (contested) accuracy lever to a (necessary) endurance/energy lever — a Pareto claim, not a dominance claim  **[B→N]** (Lever A is the repo's own; the hardware re-basing and the activity-budget metric are new)  — IMPACT 7 / FEASIBILITY 8

**WHAT.** Report 11 (T4) argues the surprise gate's *optimization* motivation is mis-aimed.
Accept that, and re-base the same lever on hardware: on endurance-limited analog cells, **writes
are the scarce resource**, so "write ∝ surprise" converts into endurance and write-energy savings
*even if accuracy-neutral or slightly negative*. The claim becomes a **Pareto claim**: at
equal-or-acceptable accuracy, surprise gating reduces state/weight writes by a measured factor —
report the accuracy-vs-write-rate curve, not a point.

**WHY.** Two structural savings, one already committed: (i) **cross-domain** — frozen experts
consume zero writes forever; **[ARITHMETIC]** on committed code + E1's routing log: at M=5
domains, ≥(M−1)/M = 80% of weight-write hardware is idle at steady state. (ii) **within-domain**
— currently *unclaimed*: the active expert trains on every batch regardless of its surprise; the
docs' PGM window (`window(b_m)`, docs §3) and Seq's `surprise_gate` lever are the dormant
mechanisms for (ii). The activity rate — fraction of batches/tokens whose surprise exceeds
vigilance — is an *empirical property of the stream*, exactly the kind of honest quantity this
repo pre-registers. On Loihi-class hardware the same threshold is a spike threshold: only
surprising tokens wake the learning engine (H6). It is also the most brain-like property on
offer — predictive coding spends compute on prediction *errors*, not predictions (Rao & Ballard
1999 **[CITE-CHECK]**, already in the repo's ledger).

**HOW.** (1) Pre-register an **activity-budget metric**: for CL on E1's stream and Seq on text8,
measure the surprise-threshold sweep θ → {fraction of writes skipped, FGT/ACC/BPC} and commit
the full curve (no cherry-picked θ). (2) Disclosure rule: any "energy saved by gating" claim must
cite a point on this measured curve plus the H5 ledger — never a bare multiplier. (3) Optional:
a "write budget" mode living under w writes/token — endurance as a knob the algorithm optimizes
against (metaplasticity as wear management, closing the loop with H1).

**FALSIFIABLE TEST.** The curve: if surprise-gating is strictly dominated by ungated training at
every write-budget point, the Pareto claim is dead and the repo records it; the cross-domain
80%-idle claim survives regardless (structural, committed code).

**RISKS.** Threshold selection on the eval stream would be leakage — thresholds must come from
the expert's own precision EMAs (the existing `μ + z·σ` machinery), never tuned on test; gating
interacts with the phase detector (both read surprise) — orthogonal ablation arms required, same
discipline as the S3 novel-core ablation.

### H5. The energy-per-token ledger: analytical ops/bytes-per-token × cited energy/op bands, disclosed as arithmetic — the tool that makes W1 legal  **[B]** (Horowitz data-movement cost model borrowed; the disclosure apparatus is the repo's own honesty culture, applied to a new domain)  — IMPACT 8 / FEASIBILITY 8

**WHAT.** A small script (built only with owner approval — this report writes no code) computing,
per arm (Prizma-Seq, transformer at context T, Prizma-CL per active/frozen phase): ops/token and
bytes-moved/token from the *committed model definitions* (the closed-form discipline of
`seq/scaling_analysis.py`), multiplied by a **user-supplied, cited table** of energy-per-op
ranges — e.g., 32-bit DRAM read ~640 pJ, 32-bit SRAM read ~5 pJ, FP add ~0.9 pJ at 45 nm
(Horowitz 2014), analog-MVM claims fJ–pJ/MAC, Loihi synaptic-event energy (Davies et al. 2021)
— all **[CITE-CHECK]**. Output: a **band** per arm (min/max), never a point; every row carries
its citation; the script refuses to run on a missing citation.

**WHY.** W1 is the deployment story's spine, but energy claims are where this field lies to
itself, and the repo has already had to delete one set of fabricated hardware rows (README
correction note). A ledger tool that *structurally cannot emit an uncited number* turns "Prizma
is 100x more efficient" (fabricatable) into: "bytes/token ratio is 372x **[ARITHMETIC, repo]**;
energy ratio is 372x × energy-per-byte, which under cited ranges lands in band A–B, excluding
ADC/DAC overheads that worsen the analog arm and paging optimizations that help the transformer
arm — here is the sensitivity table." That last clause is the repo's culture, applied to watts.

**HOW.** (1) Ops/bytes inventory per arm from committed code (state read+write, window head,
projections, MLP); (2) the cited-range table with "source" and "claimed-vs-measured" columns;
(3) min/max bands + dominant-term sensitivity; (4) README integration ONLY as "analytical bands
under cited assumptions," never as results; (5) scope: single-stream batch-1 edge only — we do
not claim the batched datacenter.

**FALSIFIABLE TEST.** Self-falsifying by construction, a pre-registered *decision rule*: "we
claim a per-token energy advantage at 65k only if the LOWER edge of Prizma's band exceeds the
UPPER edge of the transformer band under the same cited table; otherwise we report overlap." If
the bands overlap, the energy claim is dropped (the memory claim, a closed-form identity,
survives).

**RISKS.** Cited energy/op numbers disagree by orders of magnitude across nodes and papers —
that is *why* bands; analog projections (vs measured silicon) must be flagged in-band, with the
ISAAC-style caveat that ADCs can dominate (Chen et al. 2016 **[CITE-CHECK]**); misuse is
mitigated by the refusal-to-run-on-missing-citation design and a
"projections-not-measurements" watermark on every emitted table.

### H6. Loihi 2 mapping for Prizma-CL: settle loop, WTA, local rules, and the surprise threshold as a spike threshold — plus the honest statement of where it does NOT fit  **[B]** (Loihi 2 architecture, Lazzaro WTA borrowed; the mapping table + activity-budget measurement are new)  — IMPACT 6 / FEASIBILITY 6

**WHAT.** A per-component mapping table from Prizma-CL's committed mechanics to Loihi 2
(Davies et al. 2021 **[CITE-CHECK]**): expert MVMs → synaptic cores; the PC settle loop →
iterative on-core dynamics (`n_settle_steps` → timesteps); fixed FA matrices Bdec/Bcls →
burned-in projections never written after fab (a fit no backprop model enjoys); recon surprise →
local readout against a programmable threshold (vigilance); routing over M experts → on-chip
winner-take-all (Lazzaro et al. 1989 **[CITE-CHECK]**; Loihi programmable-WTA idiom); the local
factors `(Πε) ⊗ z`-family → Loihi 2's microcode learning engine (2-factor rules with pre/post
traces; the neuromodulator scalar NM → the third factor); precision EMAs (μ,σ) → learning-rule
state variables. Then the honest half: the eligibility trace (flagged expensive in docs §5) maps
to Loihi's *trace registers* — free there, but only there; and **Prizma-Seq fits Loihi poorly**
— a dense rectangular state read every token is a dense-MVM workload, not an event-driven one;
Seq's substrate is the crossbar/CIM class (H1/H2), not the spike class. Say this explicitly; it
protects the story from the "everything runs on everything" overreach.

**WHY.** The event-driven claim needs a substrate story, and the CL thread's sparsity (one
active expert, threshold-gated plasticity) is the right *shape* for it. The quantified idle
claim available TODAY from committed code: at steady state with M=5 committed domains, ≥80% of
weight-update hardware receives no events (structural; H4 measures the within-domain fraction).
The always-on-learning asymmetry (§2.1-3a) is the capability no spiking-transformer result
demonstrates.

**HOW.** (1) The mapping note (H1's format discipline); (2) an **activity-budget measurement**
on the E1 stream: events/token under the vigilance sweep (H4's curve, Loihi units); (3) what a
real claim would require: Lava **[CITE-CHECK]** emulation of a *single expert* at smoke scale,
labeled capability-demonstration, never efficiency-evidence.

**FALSIFIABLE TEST.** Falsified if any committed CL operation has no Loihi counterpart. The
dangerous one is the exact local delta *write*: Loihi 2 supports synaptic-state writes via its
learning engine, but its weight state is digital — the analog-endurance story does not apply
there, and the table must disclose this. Pre-register: "each row carries a
supported/unsupported flag; any unsupported row kills its claim sentence."

**RISKS.** Wild Loihi efficiency numbers are workload-specific and partly vendor-adjacent — any
efficiency sentence must come from H5's cited bands; Lava emulation is real engineering for
marginal claim value now (hence feasibility 6); the note must never imply hardware access — the
repo owns no Loihi and says so.

### H7. Routing-is-free: M-expert parallel crossbar reads make sparse-MoE routing O(1) physical time — the analog answer to MoE's dispatch tax  **[B→N]** (crossbar parallelism borrowed; the routing-cost argument for expert architectures is new framing)  — IMPACT 6 / FEASIBILITY 7

**WHAT.** Digital sparse-MoE pays a dispatch tax: gating networks, all-to-all communication,
load-balancing losses. Prizma's routing reads all M experts' reconstruction surprises **in
parallel** — on a crossbar array with M adjacent tiles, M analog reads happen in O(1) physical
time with no inter-tile communication, followed by a local current-mode WTA (Lazzaro
**[CITE-CHECK]**). A short note quantifying **[ARITHMETIC]** digital routing overhead per token
(gate FLOPs + dispatch bytes) vs the analog equivalent (M parallel reads — spatial parallelism
provides them "for free" once the expert tiles exist).

**WHY.** It converts an incidental detail (Prizma evaluates all experts to take argmin) into a
substrate-native advantage — and a general statement about *expert* architectures on analog
substrates, positioning Prizma in a live hardware-design conversation (near-memory MoE at the
edge).

**HOW.** Arithmetic note beside H1's; one figure: per-token routing cost vs M, digital (gating +
dispatch bytes) vs crossbar (parallel reads + WTA), disclosed units, H5-cited constants.

**FALSIFIABLE TEST.** The arithmetic is auditable: an auditor recomputes the digital side from
committed code (the CL thread does exactly M recon reads + argmin) and the analog side from H1's
tile map. The batch-1 assumption is disclosed up front (edge single-stream is the target regime;
server batching amortizes the digital tax).

**RISKS.** At M=5–8 the digital overhead is small in absolute terms — frame as scaling behavior
and O(1) physical time, not a constant win today; analog read noise on recon surprises feeds
routing robustness — exactly what H3's routing-confusion metric measures; cite H3, don't
double-claim.

### H8. The brain-likeness ledger: name the measurable signatures so "brain-like" is a claim, not a vibe  **[N]** (process)  — IMPACT 5 / FEASIBILITY 9

**WHAT.** A one-page ledger binding each "brain-like" adjective to a pre-registered measurable:
(i) *local plasticity* → no weight transport (measured: DFA ≥ exact-W^T, E4); (ii) *event-driven
sparsity* → activity budget (H4's curve); (iii) *consolidation* → boundary-free freeze events
(E1's FGT=0) + labile/stable memory classes (H1's S/W split); (iv) *neuromodulation* → the
single scalar NM/third-factor (docs §2.4, dormant — honestly marked designed-not-tested); (v)
*predictive-coding economy* → compute ∝ prediction error (H4's curve, different caption). Each
row: mechanism → metric → committed result or "not yet tested."

**WHY.** The mission asks for "as brain-like as possible"; the failure mode of that ask is
decorative neuroscience. The ledger is the antidote — an auditable row-by-row claim that
surfaced (iv) as a designed-but-untested gap and doubles as the rebuttal kit against "just an
RNN with better marketing."

**HOW.** Owner-approved doc; every row cites its committed result file or is marked
NOT-YET-TESTED; no new experiments — H4 supplies the two missing rows.

**FALSIFIABLE TEST.** A row is falsified if its metric does not follow from the committed
mechanism (e.g., if gated writes prove accuracy-dominated, row (ii) must retreat to "the
architecture permits event-driven execution" — the ledger forces the retreat to be explicit).

**RISKS.** None technical; drift — rows must reference measured artifacts, and any row without
one stays marked untested (as (iv) does today).

---

## 4. What NOT to do (guardrails, inherited and extended)

1. **No simulated-hardware rows. Ever.** The repo deleted `CUDA (Simulated)` once and documented
   it (README). The equivalent sin here is "memristor-simulated accuracy" without a
   pre-registered degradation model — H2/H3 make emulation *legitimate*: model disclosed,
   parameters disclosed, labeled emulation, never measurement.
2. **No energy or endurance number without a band, a citation, and an ARITHMETIC label** —
   including README prose. Bands from H5 only; endurance arithmetic only with disclosed
   parameters.
3. **No "transformers cannot run on edge/neuromorphic hardware" claims.** The steelman (§2.1)
   kills both. Legal claims: O(1)-vs-O(T) data movement, append-only-vs-in-place write
   structure, and the absence of an always-on-learning equivalent in the transformer stack.
4. **No batched-server energy claims.** The wedge is single-stream, long-context, always-on
   edge; claiming the datacenter invites the batching rebuttal, which is correct.
5. **No hardware access assumptions** in any success criterion — every bar is satisfiable on
   CPU or by arithmetic (the repo owns no Loihi, no RRAM).
6. **Scope discipline vs report 08:** training throughput, Triton kernels, scaling campaigns
   belong to 08. The dependency is currently empty in both directions.

## 5. Borrowed-vs-new ledger (this report)

| Component | Source | Status |
|---|---|---|
| Data movement dominates energy | Horowitz, ISSCC 2014 [CITE-CHECK] | borrowed |
| RRAM crossbar accelerator architecture (ISAAC/PRIME), ADC dominance | Chen et al. 2016; Shafiee et al. 2016 [CITE-CHECK] | borrowed |
| RRAM endurance/variability facts | Ielmini & Wong 2018 [CITE-CHECK] | borrowed |
| Loihi / Loihi 2 architecture, learning engine, trace registers | Davies et al. 2018, 2021 [CITE-CHECK] | borrowed |
| Analog winner-take-all circuits | Lazzaro et al. 1989 [CITE-CHECK] | borrowed |
| Labile→stable consolidation, cascade metaplasticity | Frey & Morris 1997; Fusi et al. 2005 [CITE-CHECK] | borrowed (repo already cites Fusi for metaplasticity) |
| KV quantization / eviction / SWA (steelman only) | KIVI; H2O; SnapKV; Mistral [CITE-CHECK] | borrowed (used AGAINST us honestly) |
| Spiking transformers exist (steelman only) | Spikformer; SpikeGPT [CITE-CHECK] | borrowed |
| Quantization/fault-injection method | standard QA practice | borrowed |
| **Consolidation gate = endurance gate (one variable, two consequences)** | — | **NEW synthesis (H1)** |
| **Two-state technology assignment (S volatile CIM / W RRAM; burned-in FA matrices)** | — | **NEW mapping (H1)** |
| **Delta-vs-additive fitness under state quantization** | — | **NEW hypothesis + bar (H2)** |
| **Re-basing surprise gate from accuracy to Pareto endurance/energy claim** | — | **NEW reframing (H4)** |
| **Refusal-to-run-on-uncited-inputs energy ledger; band-only emission** | — | **NEW process (H5)** |
| **Activity-budget metric; brain-likeness signature ledger** | — | **NEW process (H4/H8)** |
| Prizma mechanisms themselves | this repo (docs §4) | built, cited in place |

---

## QUESTIONS FOR THE OWNER

1. **Pre-registration authority:** H2 (state-quantization bar) and H3 (degradation battery) are
   cheap and CPU-only. Do you authorize writing their pre-registrations (exact grids, bars,
   seed counts) BEFORE any run, as with §B4? Given the B4 history, I recommend the numbers be
   frozen by you, not by the implementer.
2. **The Pareto reframe (H4):** are you comfortable officially re-basing the surprise-gate lever
   from an accuracy claim (contested per report 11/T4) to an endurance/energy Pareto claim?
   This changes wording in the Seq spec's Lever A section.
3. **Energy ledger policy (H5):** shall the cited-range table be pinned to owner-approved
   sources (I propose Horowitz 2014 for digital, one measured-CIM paper, one Loihi paper), with
   citation additions requiring a README-visible change record?
4. **Naming and placement:** should the deployment notes (H1 mapping, H7 routing note, H6 Loihi
   table) live under `docs/neuromorphic/` as a clearly-labeled "design + arithmetic, no
   measurements" family, so no reader mistakes them for results?
5. **Brain-likeness ledger (H8):** ledger "brain-like" formally in the README (rows:
   measured / not-yet-tested), or confine to docs? I recommend: ledger in docs, one honest
   sentence in the README.
6. **Priority call:** with one CPU-week: H2 first (novel falsifiable hypothesis) or H3 first
   (finishes what the smoke tests started, protects the existing §5 claim)? I recommend H2 for
   novelty-per-hour; H3 is the safer institutional choice.
