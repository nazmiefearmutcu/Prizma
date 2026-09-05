# Brainstorm — Sparse Distributed Computing & Mixture-of-Experts lens: sparsify the delta state, grow experts by vigilance, budget compute by surprise

**Commission:** Prizma → inevitable Transformer rival + maximally brain-like
**Author:** Sparse Distributed Computing & MoE Architect (sparse coding, k-winner-take-all,
E/I balance, divisive normalization, lifelong MoE, expert capacity/aux losses, conditional compute)
**Date:** 2026-09-03
**Classification (per brainstorming skill):** **Architectural-class research-direction
brainstorm.** HARD-GATE honored: **no code, no implementation** — this report is a design/ideas
document only. The skill's clarifying-question step cannot reach the human owner from inside a
committee, so every direction-setting question is collected in **QUESTIONS FOR THE OWNER**.

Sources read before writing: `README.md`, `docs/Prizma.md`, `docs/PRIZMA_SEQ_REPORT.md`,
`flop_ledger.py` (line-level), `src/prizma.py` (line-level), skim of `seq/prizma_seq.py`,
and `committee/brainstorm_2026-09-03/01_computational_neuroscience.md` (format alignment only).

Repo-culture discipline used throughout: every proposal carries **WHAT / WHY (brain) / WHY
(beat Transformers) / HOW / FALSIFIABLE TEST (pre-registered bar) / RISKS / IMPACT 1–10 /
FEASIBILITY 1–10**, and every borrowed mechanism is cited. All FLOP arithmetic in §7 is
**analytical committee arithmetic derived from `flop_ledger.py`'s own counting conventions**
(MACs×2, causal-honest TF count, chunked-delta cost model). It is **not** a repo-produced
number; before any claim is booked, the sparse variants must be added to `flop_ledger.py` and
re-derived there. Nothing below is a result; everything is a proposal with a pre-registered bar.

---

## 0. Framing: sparsity is the brain's budget policy, and it is exactly where Prizma's two honest gaps live

Cortex runs at roughly **1–4% of neurons active at any instant** (Lennie 2003; sparse codes in
V1: Vinje & Gallant 2000; auditory cortex ~5%: Hromádka, DeWeese & Zador 2008). Sparse coding
is not a curiosity — it is the *cost model* (Olshausen & Field 1996; Kanerva 1988's Sparse
Distributed Memory makes the capacity/energy trade explicit), and it is enforced mechanically
by **k-winner-take-all inhibition** and **excitatory/inhibitory balance** (Douglas & Martin
2004; van Vreeswijk & Sompolinsky 1996), with **divisive normalization** as the gain-control
substrate (Carandini & Heeger 2012) and **homeostatic synaptic scaling** as the slow hygiene
(Turrigiano & Nelson 2004).

Read against the repo, this lens lands on the two numbers the repo itself flags as binding:

1. **"Prizma trains ~5× slower per step"** (B4: 1889 vs 361 s/arm; report's honest limit #2).
2. **"No per-FLOP claim"** (Prizma-quad2 = 2.14× the TF's forward FLOPs/token as-coded,
   1.78× with the ideal banded window, at d128L4H4 v1ref; the FLOP-matched TF arms were
   optimization-confounded and excluded).

My central quantitative claim, derived in §7: **the delta state is dense where the brain's
states are sparse, and that density is the single largest line in Prizma's own FLOP ledger**
(delta_state = ~43.6% of as-coded forward FLOPs at the headline scale, vs the MLP's 23.5%).
Top-k sparse writes/reads — the cortical move — shrink exactly that line, and at ~12.5% active
state columns land Prizma at **analytical FLOP parity** with the param-matched Transformer;
below ~10% it goes **sub-parity**. Combined with the already-built banded-window lever and a
surprise-budgeted layer skip, the analytical ceiling is ~0.73× the TF. That flips the burden:
Prizma becomes the *cheaper* arm, and "no per-FLOP claim" becomes a per-FLOP claim **in
Prizma's favor** — testable with the repo's existing harness discipline.

Second thesis: the repo already owns the thing modern MoE lacks. Dense-MoE Transformers need
**auxiliary load-balancing losses** (Shazeer et al. 2017; Switch capacity factors, Fedus et al.
2022; ST-MoE z-loss, Zoph et al. 2022) — extrinsic, non-local warts that fight the language
objective. Even the newest aux-loss-free method (Wang et al. 2024, DeepSeek-V3) still needs a
tuned per-expert bias update; expert-choice routing (Zhou et al. 2022) breaks autoregressive
causality. **Prizma-CL demonstrates balance-without-aux-loss in a small regime** — ART-style
vigilance routing + surprise-driven recruitment + consolidation freeze, 100% routing accuracy
in E1, all local, zero auxiliary loss. The missing step is **token-level** (not
domain-contiguous) routing in a *sequence* model — and Prizma-Seq's per-token write error
`ε_t = v_t − S_{t-1}k_t` is a free, already-computed per-token surprise signal that can drive
it. Fusing the two threads is, from this lens, the most strategically novel move available:
**lifelong experts recruited by vigilance, balanced by competition, consolidated by
surprise — the first MoE that needs no balancing loss because balance is architectural.**

Third thesis: the Transformer's two stabilization warts — LayerNorm (5 per block in the
baseline) and uniform per-token compute — both have cortical counterparts Prizma can claim:
divisive/E-I normalization instead of LayerNorm (§P4), and surprise-budgeted depth — predicted
stimuli evoke *less* cortical processing (Keller & Mrsic-Flogel 2018; Audette et al. 2022/23
voltage-imaging) — instead of compute-per-token constancy (§P3).

Honesty guard-rail from the repo that binds everything below: the **surprise-gating ablation
(`gpu_ablation.json`) is smoke, n=2, INCONCLUSIVE, and its point estimates run *against* the
mechanism** (constant gate 0.566 > random 0.527 > real surprise 0.517 > no gate 0.406). Any
proposal that *spends* the surprise signal (P2, P3) is therefore **premised on a signal whose
causal usefulness is currently undemonstrated**. The powered surprise ablation is thus
prerequisite #1 and is itself one of my ranked proposals (P3a).

---

## 1. Proposal inventory (each: WHAT/WHY/HOW/FALSIFIABLE/RISKS/scores)

### P1 — Top-k sparse delta state: k-winner writes, sparse reads (~2%-active carried memory)
**[N] new mechanism on [B] sparse-coding/k-WTA principles. RANK #1.**

- **WHAT.** Today every token writes a *dense* rank-1 outer product `ΔS = β_t·u_t k_tᵀ` into
  the per-head state `S ∈ R^{d_h×d_φ}` (d_φ=256 at v1ref) and reads with a dense query
  `o = S q_t`. Make the state sparse the way cortex is sparse: (a) **write-side top-k** —
  apply the delta update only to the k columns of S selected by a top-k on the *key feature
  vector* `φ(k_t)` (with straight-through gradients for the argmax); (b) **read-side top-k /
  block-sparse query** — read only the columns selected by top-k on `φ(q_t)`, or on a
  coarser per-token granularity to keep kernels regular; (c) optionally a **decay-and-threshold
  hygiene pass** per chunk: scale down state columns whose accumulated write mass falls below
  a running threshold ("synaptic scaling", Turrigiano & Nelson 2004) so activity stays ~k/d_φ
  by homeostasis rather than by a hand-set k. Config knob `state_topk: int = 0` (0 = dense,
  byte-identical default, per repo lever discipline).
- **WHY (brain).** Cortical memory traces are sparse and competitive: only a small winner
  population encodes an item, inhibition enforces the budget, and homeostatic scaling keeps
  total activity constant. Kanerva's SDM is the canonical statement that *address sparsity is
  what makes the memory cheap and interference-limited*.
- **WHY (beat Transformers).** It attacks **both** binding gaps at once. (i) FLOPs: the delta
  path is ~43.6% of as-coded forward FLOPs; top-k with active fraction f shrinks its dominant
  matmuls ~proportionally (§7 arithmetic: f=12.5% → ≈1.02× TF; f=6.25% → ≈0.96×, sub-parity).
  Attention cannot do this — attention's O(T²) score matrix is dense *by mechanism*; top-k
  attention variants (Child et al. 2019; Roy et al. 2021) pay router overhead and still carry
  an O(T) KV-cache. (ii) Capacity: sparsity *raises* effective key separation (less crosstalk
  between superposed traces — the SDM/quad2 crosstalk story the repo already measures), so
  MQAR-style recall should hold or improve at equal d_φ, not degrade.
- **HOW.** Three granularities, in ascending engineering cost: (1) **column-top-k** (gather/scatter
  on d_φ axis — exact delta-rule semantics on the selected columns); (2) **block-top-k**
  (8- or 16-column blocks so chunks stay tensor-core-friendly, cf. 2:4 structured sparsity
  practice, Mishra et al. 2021); (3) hard top-k in the *feature map* itself (top-k quadratic
  monomials — a "sparse quad2", zeroing d_φ structurally). Sparsified `chunked_delta` must keep
  the repo's guards: `step()==forward()` O(1) equivalence, off==dense byte-identical, and the
  chunk/recurrent dual-form agreement test at <1e-5.
- **FALSIFIABLE TEST (pre-registered bar).** At d64L2H2/130K and d128L4H4/860K, ≥5 seeds:
  (i) MQAR D=128 solve-rate within 0.05 of dense quad2 for f ∈ {25%, 12.5%} (solve ≥0.9);
  (ii) sparse-ledger (`flop_ledger.py` extension) confirms ≤1.1× TF forward FLOPs/token at
  f=12.5% with the banded window; (iii) **wall-clock bar** (binding, not FLOP): training
  tokens/s within 1.3× of the matched TF on A100 at the same scale — this is the honest test of
  the 5× claim; (iv) char-LM text8 BPC within +0.05 of dense Prizma. Pre-register the
  **f\* at which MQAR breaks** and report it as the sparsity-capacity frontier (never hide it).
- **RISKS.** (1) **Wall-clock ≠ FLOPs**: dynamic gathers are memory-bound; dense BF16 matmuls
  run at high tensor-core efficiency while top-k kernels may not win at small d_φ=256 — the
  5× slowdown is likely kernel-latency-dominated (Lever F's fused Triton kernel was deferred),
  so P1 must be paired with kernel work or its wall-clock bar will fail even if FLOPs drop.
  (2) Training-sparsity mismatch: top-k under straight-through estimators can be unstable at
  init; block granularity mitigates. (3) The chunked WY/UT parallel form assumes dense keys;
  sparsity must enter *before* the chunk products (sparse φ(k) blocks), not after, or the
  parallel form breaks — this is the main implementation subtlety. (4) Capacity loss at too-low
  f on long-context tasks (B5) — pre-register the frontier.
- **IMPACT 9 / FEASIBILITY 6.** Highest because it is the only proposal that plausibly closes
  *both* the 5× slowdown and the per-FLOP gap with one mechanism, while deepening (not diluting)
  the brain-likeness claim.

### P2 — Lifelong vigilance-MoE as the FFN: grow experts by surprise, balance without aux loss
**[B] ART + [B] MoE lineage, [N] fusion & token-level vigilance. RANK #2 (strategic).**

- **WHAT.** Replace each block's dense SwiGLU with a **growing mixture of small predictive-coding
  FFN experts** (e.g., E_max=8 slots of width d_ff/4, top-2 active). Routing signal = the
  layer's own **write-error energies** `E_t^{(h)} = ½‖ε_t‖²` — already computed by the mixer,
  zero extra router FLOPs — plus a tiny learned probe if the energies prove uninformative.
  Each expert keeps Prizma-CL's **precision EMA (μ, σ)** over its own output error; a token
  routes to the expert whose error is within `μ + zσ` (recognized); persistent novelty (EWMA of
  min-surprise above vigilance for W consecutive tokens) **recruits a fresh expert**; sustained
  low bid **freezes** it (consolidation). **No auxiliary load-balancing loss, ever** — balance
  is an architectural consequence of vigilance competition, *monitored* (max-expert-share) but
  never *optimized*. Knobs `moe_experts: int = 1` (1 = dense, byte-identical).
- **WHY (brain).** This is literally Prizma-CL's cortex: modules compete on precision-weighted
  error, novelty recruits (ART vigilance; Carpenter & Grossberg 1987), mastery consolidates.
  Extending it token-level turns "domain expert" into "regime expert" — cortical areas are
 Recruited continuously, not at task boundaries.
- **WHY (beat Transformers).** Modern MoE needs extrinsic balancing machinery (aux losses:
  Shazeer 2017, Fedus 2022, Zoph 2022; bias-balancing: Wang 2024) because a softmax router
  trained only on the task objective collapses to few experts. Expert-choice (Zhou 2022) avoids
  collapse but **violates autoregressive causality** (tokens chosen by experts, not vice versa).
  Prizma's vigilance is (i) **local and causal**, (ii) **aux-free by construction** (the E1
  ablation `PRIZMA_noRoute` shows routing is the causal mechanism in the CL regime), and
  (iii) **lifelong** — experts grow with the data distribution (contrast Lifelong-MoE, Shen et
  al. 2023, which needs task boundaries and a fixed expert budget; sparse upcycling,
  Komatsuzaki et al. 2023, grows once, offline). A token-level MoE that holds balance without
  any aux loss would be a **first**, and it is the clearest "inevitable-rival" differentiator
  in this report: it changes what an MoE *is*, not just its throughput.
- **HOW.** Keep the mixer untouched (only-the-token-mixer-differs discipline becomes
  only-the-FFN-differs for this lever). Expert = 2-layer low-rank FFN with the CL thread's
  local delta rule *optionally* available (backprop parity first — do not couple two novelties;
  per repo culture, one lever per bar). Growth implemented as static E_max slots with a
  recruited mask (honest: allocated-but-unrecruited slots cost parameter space, zero FLOPs).
  Routing is discrete top-2 with straight-through gradients; vigilance threshold z and window W
  are the only new hyperparameters.
- **FALSIFIABLE TEST.** char-LM text8, param-matched dense-vs-MoE and FLOP-matched at top-2,
  ≥5 seeds. Bars: (i) BPC within +0.05 of dense Prizma; (ii) **zero aux loss** and
  max-expert-load ≤ 4× uniform mean over the last 20% of training (if balance collapses
  without the loss — the Shazeer failure mode — the proposal is FALSIFIED, which is itself a
  publishable, honest result that validates the field's aux-loss practice from a surprising
  direction); (iii) expert recruitment count grows sub-linearly in corpus order (lifelong
  property) and tracks corpus regime changes (segment-level expert-usage purity > 0.6);
  (iv) ablation: routing signal = real write-error energies vs shuffled-energies control
  (mirroring the repo's `surprise_random`/`surprise_constant` discipline).
- **RISKS.** (1) **Known failure mode is the thesis**: balance without aux loss may simply
  fail at token level — vigilance was validated on *batch-level, input-distinguishable domains*
  with contiguous blocks; the repo itself documents interleaved-stream collapse (ACC ~0.58).
  Single-token reconstruction surprise is small/noisy — hence window-level (conv-smoothed)
  novelty. (2) Discrete routing breaks end-to-end differentiability — straight-through noise
  at char scale. (3) Static E_max + masks is not "true" growth; a TCO/honesty footnote is owed.
  (4) Two novelties (sparse state + MoE FFN) must be barred **separately**, never confounded.
- **IMPACT 9 / FEASIBILITY 5.** Strategically the highest; feasibility docked for the
  differentiability and balance risks and for char-scale noise in the vigilance signal.

### P3 — Surprise-budgeted conditional depth (+ the prerequisite powered surprise ablation)
**[B] ACT/CALM/MoD lineage + [B] predictive-suppression literature, [N] using the delta write error as the budget signal. RANK #3.**

- **P3a (prerequisite).** A **powered** ablation of the surprise signal itself: the repo's
  own honesty note says the only evidence (smoke n=2) points *against* surprise-gating being
  informative. Before any conditional-compute claim, run the existing `surprise_mode ∈
  {norm, random, constant}` ablation at n≥10 seeds on MQAR-D128 + text8 with CIs. Cost: small
  (existing knobs). If real surprise ≤ constant control at power, P2/P3's signal premise is
  falsified and must be re-scoped to *learned* routers. **This is the cheapest, most binding
  experiment in this report.**
- **P3b.** **Conditional compute via surprise**: predicted tokens skip deep processing. Per
  token, compute the mixer's write-energy `E_t`; if `E_t < θ_l` (well-predicted), skip layers
  ≥ l (residual pass-through; the carried state still decays but does not write — the state,
  not the token, carries continuity, which is exactly what makes skipping cheap in a
  recurrent-mixer architecture and expensive in attention, where every token must attend).
  Enforce a per-layer **token budget** (Mixture-of-Depths style, Raposo et al. 2024) to keep
  tensor shapes static: top-(1−r) tokens by E_t continue, the rest pass through.
- **WHY (brain).** Predictive processing's budget policy: expected stimuli evoke attenuated
  cortical responses (surprise suppression; Keller & Mrsic-Flogel 2018; Audette et al. 2022/23),
  i.e., the brain spends compute on *errors*, not on evidence it already predicts. Prizma's
  free-energy framing makes this natural: the delta write is already proportional to ε_t —
  skipping is `β_t → 0` made structural.
- **WHY (beat Transformers).** Transformers cannot cheaply skip depth (attention needs every
  token in every layer's K/V set or the cache breaks); MoD had to add routing networks and
  still needs careful causality handling. In Prizma, skipping layer ℓ for a token costs only
  that layer's MLP+out_proj+write (≈0.66 MFLOP/token/layer at the headline scale), and the
  budget makes wall-clock predictable — the MoD paper's own recipe shows ~1.5× end-to-end
  speedups at 12.5–25% skip rates in dense Transformers; the recurrent-mixer version should do
  at least as well per skipped FLOP (§7: 25% skip of layers 2–4 on the P1-sparse stack →
  ≈0.73× TF analytical).
- **HOW.** Budget router on the already-computed E_t (zero router params — contrast MoD's
  learned router). Keep shapes static via per-layer capacity r. Causality note: unlike MoD's
  attention interplay, skipped tokens write nothing, so downstream state reads are unaffected —
  the causal analysis is *simpler* here; still, pre-register a check that eval perplexity of
  low-E_t tokens does not regress (the tokens we starve must be the ones we can afford to).
- **FALSIFIABLE TEST.** text8 d128L4H4, r ∈ {0.75, 0.5}, ≥5 seeds: (i) BPC within +0.05 of
  full-depth Prizma; (ii) measured A100 tokens/s ≥ 1.3× full-depth (wall-clock, not FLOP);
  (iii) skip-rate vs token position entropy sanity (skips must concentrate on predictable
  text, measured, not asserted); (iv) P3a must have PASSED first.
- **RISKS.** (1) P3a could kill the premise (this is a feature of the process, not the plan).
  (2) Budget-forced skips on *surprising* tokens under distribution shift — guard with the
  (iii) measurement. (3) Static-capacity routing wastes slots at batch boundaries.
- **IMPACT 7 / FEASIBILITY 6** (feasibility includes P3a's near-certainty of clean execution).

### P4 — E/I balance & divisive normalization as the stabilization story (retire LayerNorm)
**[B] Carandini & Heeger, Douglas & Martin, van Vreeswijk & Sompolinsky; [N] as a replacement for pre-norm in a delta-state architecture. RANK #4 (cheapest brain-claim).**

- **WHAT.** Two moves. (a) **Divisive normalization on the state read**: `o = Sq ⊘ (γ +
  mean_topk(|Sq|))` — pool-normalized (an inhibitory pool over the head's read), replacing the
  pre-RMSNorm on the mixer branch; keep total read activity ~constant (k-WTA-style constant-
  sum normalization, Ahmad & Scheinkman 2019). (b) **Synaptic-scaling state hygiene**: per
  chunk, renormalize S's column norms toward a running target (bounds state growth that the
  delta rule's `‖k‖=1` assumption silently tolerates today). Then attempt **RMSNorm removal**
  from the mixer path entirely — the E/I circuit is doing the normalization.
- **WHY (brain).** Canonical microcircuit: roughly equal-strength inhibition divisively
  normalizes cortical responses (Douglas & Martin 2004; Carandini & Heeger 2012); balanced E/I
  keeps mean drive flat under strong input fluctuations (van Vreeswijk & Sompolinsky 1996);
  synaptic scaling is the slow homeostat (Turrigiano & Nelson 2004).
- **WHY (beat Transformers).** The baseline carries 2 RMSNorms/block as pure stabilization
  machinery with no functional story; LayerNorm-free training is a known fragility Transformers
  paper over with care. If Prizma's gate structure + DN trains stably *without any norm
  layer*, that is a mechanism-level claim attention cannot copy (its softmax scores are not
  pool-normalizable the same way without changing the function), and it saves parameters and
  bandwidth. It also directly serves the novelty premise: the repo's δ-state stability proof
  (‖J‖ ≤ α ≤ 1) suggests state dynamics *want* homeostatic control — P4 gives it a cortical
  name and a test.
- **FALSIFIABLE TEST.** text8 + MQAR at matched params, ≥5 seeds: (i) mixer-path-RMSNorm-free
  Prizma trains to within +0.05 BPC of normed Prizma with no more than baseline LR-ceiling
  reduction (measure max stable LR — the real stability metric); (ii) gradient-norm spike rate
  (count of >10× median steps) not above the normed baseline's; (iii) DN vs plain L2-read
  ablation isolates the pool term.
- **RISKS.** (1) Norms exist for optimization reasons; removing them may just need the LR sweep
  the repo has already learned to do honestly. (2) Modest novelty risk — normalization tricks
  are a crowded space; the *claim* must be scoped to "in a delta-state mixer, E/I-DN replaces
  pre-norm", not "DN beats LayerNorm" in general.
- **IMPACT 5.5 / FEASIBILITY 8.** Cheap, fast to falsify, upgrades the biological story of a
  component the architecture needs anyway (readout gain control).

### P5 — Amortized sparse state readout: prune S columns by write-mass, read the survivors
**[B] SDM/address-decoding idea + [N] amortized schedule. RANK #5.**

- **WHAT.** Complement to P1's per-token sparsity: maintain a per-head **column activity score**
  (EWMA of |φ(k)| contributions); every W tokens, keep the top-m columns and read only those
  (gather once per window, not per token). Reads cost ∝ m/d_φ; the selection amortizes over the
  window. Periodically (slow timescale) allow column *revival* if accumulated surprise
  concentrates on pruned columns — the reawakening term Prizma.md §3 already defines
  (`ω ← ω − κ·relu(conflict)`), applied to state columns instead of experts.
- **WHY (brain).** Memory traces compete for a fixed synaptic budget; unused traces decay
  below read threshold but can be re-potentiated (metaplasticity, Fusi/Benna–Fusi — already in
  the repo's ledger).
- **WHY (beat Transformers).** Zero analogue — the KV-cache cannot be structurally pruned
  online without recall loss; here the read is a matmul whose contraction dim we choose, and
  the pruning criterion is *usage*, not learned salience (cheap, stable).
- **FALSIFIABLE TEST.** With P1's bars as base: (i) MQAR D=128 solve at m = d_φ/2 within 0.05
  of dense; (ii) B5 long-context latency at n=65k improves by the predicted read-share factor
  (measured, single-A100 protocol as B5); (iii) revival ablation: pruning-without-revival vs
  with — pre-register whether revival matters at all (if not, simpler wins; report it).
- **RISKS.** Pruning mistakes are persistent (recall loss is not self-healing) — hence revival.
  Interaction with P1's per-token sparsity may double-count savings; bar them separately.
- **IMPACT 6 / FEASIBILITY 6.**

### P6 — k-WTA sparse FFN (the "2% cortex" MLP) — folded into P2 if P2 runs
**[B] k-WTA MLPs (Ahmad & Scheinkman 2019; Maass 2000), [N] integration. RANK #6.**

- **WHAT.** Replace SwiGLU's smooth nonlinearity with top-k hidden activation (k ≈ 2–5% of
  d_ff) with constant-total-activity rescaling. Only pays off wall-clock with sparse kernels
  (gathered rows); at small d_ff the dense matmul may win — be honest about this.
- **WHY.** The literal cortical activity budget (Lennie 2003); gives the FFN a sparsity story
  that pairs with P2's expert sparsity (within-expert sparsity vs between-expert sparsity).
- **FALSIFIABLE TEST.** text8 BPC within +0.05 at matched FLOPs (analytical), measured
  tokens/s reported regardless of outcome.
- **RISKS.** Known result: k-WTA MLPs train stably (Numenta's claims) but LLM-scale evidence
  is thin; at d_ff=344 the kernel overhead may dominate. **Do not run standalone if P2 runs**
  — same substrates, confounded levers.
- **IMPACT 5 / FEASIBILITY 7.**

---

## 2. The fused architecture this lens argues for ("Prizma-v3, sparse-workspace form")

No code — a target shape, every element separately barred by §1's proposals:

1. **Mixer**: Gated-DeltaNet + quad2 as today, but with **top-k sparse writes/reads** (P1)
   and **amortized column pruning** (P5); read gain set by **divisive normalization**
   (P4) instead of pre-RMSNorm; write gate β_t unchanged.
2. **FFN**: **vigilance-grown expert mixture** (P2) — experts recruited by surprise, frozen on
   mastery, balanced by competition without any aux loss; optionally k-WTA within experts (P6).
3. **Depth**: **surprise-budgeted token routing** (P3) — the per-token write energy decides
   which layers process the token; predicted tokens ride the residual + carried state.
4. **Stability**: E/I-DN + synaptic scaling (P4) as the only normalization; the existing
   ‖J‖≤α contractiveness proof as the analytical backbone.

Every Transformer wart replaced by a cortical mechanism with a pre-registered bar attached:
dense compute → sparse (P1), uniform depth → budgeted (P3), aux-loss-balanced MoE →
vigilance-grown (P2), LayerNorm → inhibition (P4). That is the "inevitable rival" pitch from
this lens: not "another DeltaNet variant", but the architecture whose *budget policies* are
cortical and whose efficiency claims follow from them.

---

## 3. Ranked summary

| Rank | Proposal | Borrowed core | Impact | Feasibility | Attacks |
|---|---|---|---|---|---|
| 1 | P1 top-k sparse delta state | k-WTA/SDM (Kanerva 1988; Olshausen & Field 1996) | 9 | 6 | 5× slowdown + FLOP gap + brain claim |
| 2 | P2 vigilance-grown MoE FFN (aux-loss-free, lifelong) | ART (Carpenter & Grossberg 1987); MoE lineage (Shazeer 2017; Wang 2024) | 9 | 5 | strategic differentiation; FFN cost |
| 3 | P3a powered surprise ablation (prereq) / P3b surprise-budgeted depth | MoD (Raposo 2024); CALM (Schuster 2022); ACT (Graves 2016); predictive suppression (Keller & Mrsic-Flogel 2018) | 7 | 6 | conditional compute; closes repo's honesty debt |
| 4 | P4 divisive norm + synaptic scaling, retire pre-norm | Carandini & Heeger 2012; Turrigiano & Nelson 2004 | 5.5 | 8 | stabilization story; brain claim |
| 5 | P5 amortized sparse state readout (+revival) | SDM; metaplasticity (Fusi; Benna–Fusi) | 6 | 6 | inference latency at long context |
| 6 | P6 k-WTA FFN (fold into P2) | Ahmad & Scheinkman 2019; Lennie 2003 | 5 | 7 | FFN cost; activity budget |

Recommended sequencing: **P3a first** (cheapest, binding, unblocks P2/P3's premise) → **P1**
(the quantitative lever; extend `flop_ledger.py` first, then block-top-k kernel) → **P4**
(quick win) → **P2** (strategic, on its own bar) → P5/P6 as follow-ons.

---

## 4. The ruthless FLOP arithmetic (analytical; committee arithmetic on `flop_ledger.py` conventions)

Per-layer, per-sequence FLOPs at the headline d128L4H4, v1ref d_φ=256, T=384, C=64, dh=32,
d_ff=344 (all MACs×2; my arithmetic — to be re-derived in the repo's ledger before any claim):

| component | MFLOP/layer/seq | share of as-coded total |
|---|---|---|
| **delta_state** (H·(T/C)·per_chunk = 4·6·7.865) | **188.8** | **43.6%** |
| window head (full-T² SDPA as-coded) | 75.5 | 17.5% |
| SwiGLU MLP | 101.5 | 23.5% |
| qkv + out_proj | 50.3 | 11.6% |
| conv + beta + phi_qk | 3.8 | 0.9% |
| (total/seq = 4·419.9 + head 50.3 = 1730 → **4504.6 kFLOP/tok**; TF = **2106.4**) | | |

`per_chunk` decomposition (MFLOP): KK 2.097 · KS0 1.049 · tri-solve 0.262 · O_inter 1.049 ·
QK 2.097 · O_intra 0.262 · S-update 1.049. Key-side sparsity (active fraction f of d_φ
columns) scales KK, KS0, O_inter, S-update by f; query-side sparsity (g) scales QK, O_inter;
tri-solve and O_intra are chunk-internal and unscaled.

**Scenario ladder (banded-window "ideal" accounting unless stated):**

| lever stack | kFLOP/tok | × TF (2106.4) |
|---|---|---|
| as-coded, dense (v1ref) | 4504.6 | 2.14× |
| + banded window only (repo's own ideal) | 3750.2 | 1.78× |
| + top-k writes+reads, f=g=25% | ≈2375 | ≈1.13× |
| + f=g=12.5% | ≈2144 | ≈1.02× (parity) |
| + f=g=6.25% | ≈2030 | ≈0.96× (sub-parity) |
| + 25% of tokens skip layers 2–4 (P3, on the f=6.25% stack) | ≈1530 | ≈0.73× |

Plus: P2 at top-2-of-8 quarter-width experts cuts the MLP's 23.5% by ~4× (≈ −178 kFLOP/tok);
the TF, by contrast, spends **50.2%** of its own FLOPs in the MLP and 9.3% in attention scores
at this T — so MoE/sparsity levers bite the TF baseline far harder than Prizma, another reason
the *comparison* (not just absolute Prizma cost) moves in Prizma's favor as sparsity deepens.

**Wall-clock honesty (binding):** FLOPs ≠ tokens/s. A100 dense BF16 peaks at ~312 TFLOPS;
2:4 structured sparsity delivers ~1.3–1.6× *real* speedup, not 2×; dynamic top-k gathers are
memory-bound; and the repo's own 5× slowdown is plausibly kernel-latency-dominated (sequential
WY, deferred fused kernel — Lever F). Therefore: (a) every FLOP claim above is analytical only;
(b) the binding bars in P1/P3 are **wall-clock tokens/s on A100**, measured with the repo's
existing separate-forward/backward, sync-barrier protocol; (c) the honest statement of P1 is
*"analytical sub-parity FLOPs, wall-clock bar of ≤1.3× TF, kernel work pre-declared as part of
the bar"* — never "sparse = faster" as an assertion. (d) The de-confounded per-FLOP TF arm
(fixing the repo's known confound) should use a per-width LR sweep or μP-style LR transfer
(Yang & Hu 2021) so the FLOP-match finally becomes claimable; with P1, Prizma is the *cheaper*
arm, which flips the direction of the confound risk and must be disclosed as such.

Backward pass note: training FLOPs ≈ forward + 2× forward for matmul-dominated graphs; all
ratios above carry over approximately. The MLP share of the TF is unchanged in backward; the
delta path's share grows slightly (the S-update has no full backward weight-grad term of the
same cost). Ledger must count backward explicitly before any "training-FLOP parity" phrasing.

---

## 5. Borrowed-vs-new ledger (this lens's additions, honest)

| Component | Source | Status |
|---|---|---|
| Sparse coding / activity budget | Olshausen & Field 1996; Lennie 2003; Vinje & Gallant 2000; Hromádka 2008 | borrowed |
| Sparse Distributed Memory (address sparsity ↔ capacity/interference) | Kanerva 1988 | borrowed |
| k-WTA with constant total activity (MLP form) | Ahmad & Scheinkman 2019; Maass 2000; McClelland & Rumelhart 1981 | borrowed |
| Divisive normalization / canonical E-I circuit / balanced network stability | Carandini & Heeger 2012; Douglas & Martin 2004; van Vreeswijk & Sompolinsky 1996 | borrowed |
| Homeostatic synaptic scaling | Turrigiano & Nelson 2004 | borrowed |
| ART vigilance, recruitment, consolidation | Carpenter & Grossberg 1987 (already in the repo's own ledger) | borrowed (repo-native) |
| MoE aux-loss practice and its costs | Shazeer 2017; Fedus 2022 (Switch); Zoph 2022 (ST-MoE); aux-loss-free bias (Wang 2024, DeepSeek-V3); expert choice (Zhou 2022); upcycling (Komatsuzaki 2023); Lifelong-MoE (Shen 2023) | borrowed (as the contrast class) |
| Conditional compute | ACT (Graves 2016); CALM (Schuster 2022); Mixture-of-Depths (Raposo 2024) | borrowed |
| Top-k/structured attention & sparsity practice | Child 2019; Roy 2021; Mishra 2021 (2:4) | borrowed |
| μP LR-transfer for de-confounded FLOP-matching | Yang & Hu 2021 | borrowed |
| **Top-k sparse delta-state writes/reads (sparse quad2 carried memory)** | — | **NEW** |
| **Amortized column pruning + reawakening revival of state columns** | — | **NEW** (extends the repo's own ω mechanism) |
| **Token-level vigilance-MoE FFN routed by the mixer's write-error energy, aux-loss-free, lifelong** | — | **NEW fusion** (parts borrowed; the fusion + causality-preserving budget is new) |
| **E/I-DN replacing pre-norm in a delta-state mixer + state synaptic scaling** | — | **NEW synthesis** |
| **Surprise-budgeted depth using the already-computed write energy (no router params)** | — | **NEW** (conditional on P3a passing) |

---

## 6. Pre-registration discipline notes (repo culture, applied to this report)

- One lever per bar; P1/P2/P3 barred on **separate** arms; never a combined "sparse stack"
  headline until each rung has cleared its own rung-bar.
- Every bar above is param-**and**-FLOP-matched with wall-clock reported alongside FLOPs;
  LR protocol disclosed per arm (the repo's known LR confound is the cautionary tale).
- The sparsity-capacity frontier (f\* where MQAR breaks; m\* where recall breaks) is
  pre-registered as a *reported* quantity, never omitted.
- The P2 falsification path (balance collapse without aux loss) is a **planned, publishable
  outcome** — the repo's tradition of honest refusals (B4 PARTIAL, quarantined recall gate)
  is the standard to keep.
- Nothing in §4 becomes citable until re-derived inside `flop_ledger.py` and committed as
  `flop_ledger_v3.json`; until then it is committee arithmetic only.

---

## QUESTIONS FOR THE OWNER

1. **Compute envelope.** The quarantine fix alone needs ~28 A100-hours; P1 (sparse-ledger +
   ≥5-seed MQAR/text8/wall-clock bars) and P3a (powered surprise ablation) add roughly 60–100
   more A100-hours by repo-protocol standards. What is the realistic budget (owned GPU? Colab
   cadence?) — and given a hard cap, the recommended order is P3a → P1-ledger+small-scale →
   P4 → P2: do you accept that ordering, or is the strategic MoE fusion (P2) higher priority
   than the efficiency lever (P1) despite its lower feasibility?
2. **Wall-clock bar appetite.** P1's honest wall-clock target (≤1.3× TF tokens/s) almost
   certainly requires the deferred fused Triton delta kernel (Lever F) as part of the bar, not
   as an afterthought. Is kernel engineering in scope for this project at all, or should P1 be
   scoped to analytical-FLOP + small-scale-quality bars with wall-clock explicitly deferred?
3. **Bar-protocol change.** If P1 lands, Prizma becomes the *cheaper* arm and "no per-FLOP
   claim" flips into a per-FLOP claim in Prizma's favor — this changes pre-registered bar
   language mid-project. Do you authorize drafting a v3 bar (param+wallclock+FLOP triple),
   knowing the repo's culture treats bar changes as events requiring disclosure?
4. **Thread topology.** Is the vigilance-MoE FFN (P2) a new thread (e.g., "Prizma-MoE") or a
   lever inside Prizma-Seq? The repo's discipline keeps threads separate with separate bars;
   P2 straddles them (it uses Prizma-CL's routing and Prizma-Seq's substrate).
5. **Failure publicization.** If P3a shows the surprise signal is uninformative at power
   (the current smoke evidence points that way), P2/P3's signal premise collapses. Do you want
   that outcome published as a refutation of the repo's "precision/surprise gating" novelty
   line (replacing it with learned routers), pre-committed now before the data exists?
