# Brainstorm — Mathematical Theory: Associative-Memory Capacity, Contraction & Expressivity of the rectangular delta state

**Commission:** Prizma → (1) inevitable Transformer rival + (2) maximally brain-like
**Author:** Mathematical Theorist (dynamical systems; linear/delta-rule state-space theory; associative
memory capacity bounds; fast weight programmers; expressivity separations; contraction analysis;
random-feature theory)
**Date:** 2026-09-03
**Classification (per brainstorming skill):** **Architectural-class research-direction brainstorm.**
HARD-GATE honored: **no code, no implementation** — proposals and derivations only. The skill's
interactive clarifying questions cannot reach the owner from inside a committee, so all
direction-setting questions are collected in **QUESTIONS FOR THE OWNER** at the end.

Sources read before writing: `README.md` (incl. §4 "Gradient Stability & Theoretical Convergence"
and the corrections/quarantine sections), `docs/quad2_theoretical_convergence.md` (Theorems 1–2,
Gershgorin capacity), `docs/PRIZMA_SEQ_REPORT.md` (§4 bar, P1–P5, deferred P3 D-frontier,
quarantined recall gate, borrowed-vs-new ledger), `seq/prizma_seq.py` (config, `_phi`, gates,
levers), `seq/delta.py` (reference + WY/UT chunked kernel, gate semantics), `feat_map_probe.py`
(crosstalk metric + random-vector floor `sqrt(2/(pi*d_phi))`), `cap_probe.py`, sibling report 07
(landscape). Claims from training knowledge rather than a live source are marked **(knowledge)**;
no numbers are invented.

Borrowed-vs-new discipline: every proposal marks **[B]** borrowed / **[N]** new /
**[B→N]** borrowed-component-with-new-framing. Everything derived in this document is explicitly
labeled *derivation sketch (new)* — none of it is a result until checked numerically and written
out in full proofs.

---

## 0. Executive frame: the repo has a stability theorem and a capacity *sketch*; it is missing the theory that would make the D-frontier predictable

The two existing theory assets are asymmetric in quality:

1. **Theorem 1 (Jacobian contractiveness) is exact and clean.** `‖J_t‖₂ = α_t·max(1,|1−β_e,t|) ≤ 1`
   for gates in [0,1]. Nothing to fix; only to *extend* (§3 below shows the bound secretly licenses
   a larger, strictly contractive gate family — reflections — that nobody in the delta-rule family
   is using, and that this is the sharpest expressivity lever available).
2. **Theorem 2 + the Gershgorin capacity bound `N < 1 + 1/cross(φ)` is a sketch with two holes** —
   and one of them is load-bearing for the repo's own headline:

   - **Hole 1 (honesty flag, see §1.3):** the bound *cannot* explain the repo's own D=128 result.
     With quad2 crosstalk ≈ 0.076 it gives **N < 14**, yet MQAR D=128 **PASSES**. The bound
     explains why *linear* fails at D=128 (8 << 128); it does **not** explain why quad2 succeeds.
     README §4 currently reads "…resolving the capacity block on MQAR D=128" — as written that is
     an overclaim by implication and deserves a corrections-section note (the repo punishes
     overclaiming, including its own).
   - **Hole 2 (the opportunity):** Gershgorin row sums bound `‖E‖₂ ≤ (N−1)·E|E_ij|` — this assumes
     **all crosstalk entries align in sign**. They do not; they are approximately zero-mean random
     variables whose *fluctuation* is what limits recall. Random-matrix reality gives
     `noise ∝ σ₂·√N` (σ₂ = per-entry standard deviation), not `N·cross`. The resulting capacity
     law has `1/σ₂²` scaling, not `1/cross` — a **factor-N difference in the exponent**, and it is
     exactly the gap between "explains one failure" and "**predicts the whole D-frontier before
     running it**". P3 (D-frontier) was deferred in the report as intractable; a predictive law
     turns the deferred run into the *decisive falsification test* of the theory. That inversion —
     theory predicts, one deferred experiment adjudicates — is the highest-value move available to
     this repo right now.

Secondary assets: Theorem 2 identifies the delta write with **one gradient step** on free energy
(already in the repo) — which means the *entire classical toolbox of optimization theory* (step-size
optimal schedules, conjugate gradients, condition numbers, Chebyshev acceleration) applies verbatim
to the write dynamics. Nobody in the GDN/Mamba/RWKV landscape owns that toolbox
(landscape report §0, corroborated here). And the delta read/write is a **Hopfield-family object**
(§4.6), which gives the brain-alignment axis its precise, citable skeleton.

One framing decision (stated up front, honest): per the brainstorm skill this is Architectural
class, but the deliverable here is a *theory program* — the unit of work is a theorem + a
pre-registered prediction + one cheap falsification run, not a subsystem. Every proposal below is
scoped so that its first falsification step is **CPU/numpy-cheap or reuses a harness that already
exists** (the probe, the deferred P3 grid, the owed surprise ablation).

---

## 1. What the theory currently says — precisely, and where it breaks

### 1.1 The recurrence and its two algebraic identities (established, repo)

State, per head, `S_t ∈ R^{d_v × d_φ}` (rectangular; `d_v = d_h`, `d_φ = d_h + feat_n2` for quad2):

```
S_t = α_t S_{t-1} + u_t k_tᵀ ,   u_t = β_w,t v_t − β_e,t α_t S_{t-1} k_t ,   ‖k_t‖ = ‖φ(k_t)‖ = 1
```

- Write-view: `S_t = S_{t-1} M_t + B_t` with `M_t = α_t(I − β_e,t k_t k_tᵀ)`.
- Read-view (pre-write): `o_t = S_{t-1} q_t = Σ_{s<t} (decay) u_s (k_sᵀ q_t)` — a kernel
  regression against a carried sufficient statistic.
- Theorem 1: `‖J_t‖₂ = α_t max(1, |1 − β_e,t|)`. (Note the *erase* gate is the one that enters the
  Jacobian; the *write* gate only moves the forcing term `B_t`. The doc states this for β_e = β_w;
  the decoupled-gate lever (`decoupled_gate`, already in code) makes the distinction operational.)
- Theorem 2: with β = 1, α = 1, one pass over N distinct keys, the recall residual is `R = −V E`
  where `E = G − I`, `G = ΦᵀΦ`. k passes: `R = −V(−E)^k`, rate `‖E‖₂`.

### 1.2 Where Theorem 2's use of Gershgorin breaks (derivation sketch, new)

`Gershgorin: ‖E‖₂ ≤ max_i Σ_{j≠i} |E_ij| ≈ (N−1)·cross(φ)` — a worst-case bound that is tight only
if all off-diagonal entries in a row share a sign. They are approximately i.i.d., zero-mean
(conditionally on key separations), with a small positive common-mode (below). For zero-mean
random ± entries the spectral norm of an N×N Wigner-type matrix is `≈ 2σ₂√N`, and the *per-row
recall noise* (which is the quantity MQAR actually scores) is even smaller:

```
recall of item i:   o = S φ(k_i) = v_i  +  Σ_{j≠i} (φ_jᵀφ_i) v_j  +  O(second order)
signal = 1;  noise² ≈ Σ_{j≠i} (φ_jᵀφ_i)² ‖v_j‖² ≈ (N−1)·E[E_ij²] = (N−1)·σ₂²
SNR(i) ≈ 1 / (σ₂ √(N−1))
```

**Capacity law (candidate, to be proven then pre-registered):**

```
N*(ε; φ)  ≈  1 + ε²/σ₂²        (SNR-limited regime)
N_max     =  min( N*(ε; φ), d_φ )   (hard rank cap: N > d_φ arbitrary pairs are not exactly
                                     storable by ANY rank-d_φ linear state)
```

with ε a single task-level tolerance constant (fit **once**, e.g. at D=64), and σ₂ the
*crosstalk fluctuation* `sqrt(E[E_ij²])` = `sqrt(cross² + Var)` measured by an extended
`feat_map_probe` (it currently reports only the mean `E|E_ij|`; the second moment is a two-line
addition — numpy, not GPU).

Sanity check against known numbers: σ₂ ≥ cross = 0.076 (quad2, d_φ=256) →
`N*(ε=1) ≤ 173`. D=128 PASS is consistent with ε ≈ 0.9–1.2; linear `none` (cross 0.142,
σ₂ ≈ 1/√32 = 0.177 by the random-vector floor) → `N*(1) ≈ 33` — fails D=128 ✓, and would predict
the linear baseline *passes* around D ≈ 32, which matches the pre-registered D*=32 expectation that
the experiment then *beat* with quad2. One law, three known facts, zero free parameters beyond ε.

**The mean/fluctuation split.** Conditionally on `ρ = x_iᵀx_j`, the doc's own approximation gives
`E[φ_iᵀφ_j | ρ] ≈ ((1−λ)ρ + λρ²)/(mixing norm)` with `λ ≈ 0.18` at `feat_n2=224, d_h=32`. Since
`E[ρ²] = 1/d_h > 0`, the off-diagonal Gram has a **nonzero mean component** `μ ≈ λ/d_h ≈ 5.6e−3`
plus the `I=J` self-monomials — a rank-one common-mode `μ·11ᵀ` on top of the fluctuation `W`:
`E ≈ μ11ᵀ + W`. The common-mode term contaminates every retrieval with the *same* vector
`(Σ_j v_j)` — key-independent, hence **absorbed by the learned output head** (and by `state_norm`);
only `W`'s fluctuations are irreversible capacity noise. The current doc never makes this split, and
its "convex combination" formula (`(1−λ)ρ + λρ²`) is an *expectation* statement — capacity is set by
the **conditional variance around that expectation**, which the doc never computes. Deriving
`Var[φ_iᵀφ_j | ρ]` in closed form for the sampled-monomial map (terms `x_{I_k}y_{I_k}x_{J_k}y_{J_k}`
are near-independent; dominant term `≈ 9·n2/d_h⁴`; the `I=J` monomials contribute a positive-mean
piece `3/(d_h(d_h+2))` per pair) is exactly the small, sharp computation that turns the probe into a
predictive instrument. *Caveat I flag against myself:* my own back-of-envelope from those moments
lands within a factor ~1.3–1.6 of the probe's measured 0.076 — closing that gap analytically
(self-monomial statistics, pair-collision effects at n2=224 draws from d_h²=1024 ordered pairs, and
the ρ-correlation of the monomial block) **is** the deliverable, and if the gap does not close, the
law is falsified at the theory level before any GPU is switched on. That is the correct order of
operations.

### 1.3 Honesty flags (for the corrections section, owner to authorize)

- **Flag A:** README §4 "Gershgorin … resolving the capacity block on MQAR D=128" — the bound
  `N < 14` cannot explain a D=128 PASS. Proposed wording: "Gershgorin explains the *linear*
  baseline's failure at high D; the quad2 success is *not yet* explained by a bound — see the
  capacity-law program (committee 11, proposal T1)."
- **Flag B:** `docs/quad2_theoretical_convergence.md` §5's use of `E[|E_ij|]` as if it were the
  convergence-limiting quantity conflates mean and fluctuation. Theorem 2 itself is correct (it is
  a norm bound); only the *capacity corollary* is loose by a factor that grows with N.
- **Flag C:** `feat_map_probe.py`'s "theoretical lower bound sqrt(2/(π·d_φ))" is the **mean
  absolute** cosine for purely random keys; the correct comparator for capacity is the second
  moment `σ₂`. Under the σ₂ metric the maps should be compared by a dimensionless
  **crosstalk efficiency** `η(φ) = 1/(σ₂²·d_φ)`: the random-vector floor has η = 1; measured quad2
  has `σ₂ ≈ cross·π/2 ≈ 0.119` → `σ₂²·d_φ ≈ 3.6` → **η ≈ 0.27** — quad2 pays ~3.6× more
  noise per dimension than the floor. `none` sits at η ≈ 0.63 (its E|·|≈0.8σ identity is why the
  uncorrected comparison flatters it). This η-gap is *headroom*, and proposal T3 monetizes it.

---

## 2. The four theory programs (derivations sketched; each becomes a proposal below)

### 2.1 From crosstalk mean to crosstalk spectrum → predictive D-frontier (→ T1, T2)

Combining §1.2: the law `N* = min(1 + ε²/σ₂², d_φ)` with gates (next paragraph) predicts, from
numpy-only probe measurements:
- the **deferred P3 grid** (D ∈ {16,32,64,128,256} × d_φ ∈ {32,128,137,256,512});
- the **quad2 vs quad2_lowrank equivalence** (both should land on the same N* when σ₂·d_φ match —
  the probe already has the keys to test this *before* any MQAR run);
- the **rand_linear control**: rank ≤ d_h → its Gram is the d_h-space Gram lifted; σ₂ should equal
  `none`'s to first order → predicts the B6 causal ordering *from theory*, not just empirics.

Gates enter through one factor. With decay `ᾱ` and one pass, a write of age `a` survives as
`ᾱ^a`; the SNR of the *oldest* stored item against the freshest noise is
`SNR = ᾱ^{A_max}/sqrt(Σ_a ᾱ^{2a}) ≈ ᾱ^{A_max}·sqrt(1−ᾱ²)`. So the law becomes
`N*_eff(ᾱ) = min(N*, 1/(1−ᾱ²))` — **memory window = SNR window**. The same product
`γ_t = Π α_τ` bounds the BPTT gradient (Theorem 1's §3 implication: gradient norm × γ_t). So:

> **Horizon identity (small theorem, new framing):** *the decay factor γ_t simultaneously (i)
> multiplies the stored-item magnitude, (ii) multiplies the BPTT gradient, and (iii) divides the
> effective capacity window. Memory horizon = gradient horizon. One number, three consequences —
> and the repo already measures (ii)'s stability and gates (i) for char-LM without connecting them.*

Prediction to pre-register: turning on `gated=True` with mean ᾱ = 0.999 collapses long-range MQAR
recall beyond ≈ 1/(1−ᾱ²) ≈ 500 effective pairs while leaving short-range intact. This connects the
two halves of the existing theory document with a single scalar.

### 2.2 The write gate as controlled contraction: admissible envelope, optimality, and why the surprise lever is aimed at the wrong gate (→ T4)

From `‖J‖₂ = α max(1, |1−β_e|)`:
- **Contractive envelope:** `0 ≤ β_e ≤ 2` keeps `max(1,|1−β_e|) = 1` → `‖J‖ = α ≤ 1`. The current
  code restricts `β ≤ 0.99` — the *safe half* of an interval that legally extends to 2.
- **β = 1 is exactly optimal for a single presentation** (any α): the post-write readback
  `S_t k_t = β v + (1 − β) α S_{t-1} k_t` equals `v` iff β = 1. β < 1 buys robustness to crosstalk
  (soft overwrites) at a quantifiable cost `1/(1−β)` in per-item residual — so the *optimal β under
  noise* is derivable as a bias–variance tradeoff: `β* ≈ 1/(1 + (N−1)σ₂²)`-type law (to derive
  properly). This gives the write gate a **principled target**, not just `σ(W_β x)`.
- **The surprise gate `g_t = 1 + tanh(‖ε_t‖)` multiplies the whole write** → effective
  `β_e ← g·β`. Contractiveness then requires `g·β ≤ 2`; with β_cap = 0.99 and g up to ≈ 2, the
  margin is **~2%**. Worse: since β = 1 is already the exact one-shot solver, *scaling the write
  away from 1 is provably counterproductive for storage* — the smoke ablation's ordering
  (constant ≥ random ≥ real signal, all above baseline) is exactly what "the lever is a no-op at
  best, margin-eating at worst" predicts. The honest status in the report ("points AGAINST, owed a
  powered ablation") can now be upgraded to a **pre-registered falsifiable prediction**: at
  β_cap ≈ 1, powered surprise-β ≈ constant-β ≥ surprise-random, all within noise of baseline;
  if the real signal beats the constant control by the pre-registered margin, my reading is
  falsified and the PC-surprise story survives with evidence — either way the repo wins.
- **Where surprise *should* go (theory-guided redirection):** the free-energy gradient step already
  writes ε itself; the *un-gated* quantity is **retention under interference** — i.e. α. A
  surprise-modulated *decay* (high ε on key k → keep α high for competing keys? or erase harder?)
  is the version with a clean Bellman interpretation (below), and it does not touch the
  contraction envelope. Brain tie-in: dopaminergic/ACP-style gating acts on **plasticity rate and
  retention windows**, not on doubling the Hebbian step amplitude.
- **Optimal schedule = scalar Bellman problem:** the myopic one-step-GD view (Theorem 2) ignores
  that today's write changes *future* free energy `Σ_τ γ^τ F_{t+τ}`. The optimal input-dependent
  `β_t*` is the solution of a scalar optimal-control problem on the state; its myopic special case
  is the classical Cauchy steepest-descent step `β = ‖r‖²/(rᵀGr)` (with `G` the chunk Gram —
  known in-chunk to the WY solver). Deriving `β_t*` for the two toy regimes (repeated keys;
  noise-limited fresh keys) and checking the learned `σ(W_β x)` correlates with it on a trained
  model is a mechanistic-interpretability test of the whole "write = gradient step" identity.

### 2.3 quadk and the cost–capacity frontier: quad2 is *not* on the optimal frontier (→ T3, T5)

Facts from the kernel: chunk cost is `O(T·(C·d_φ + d_v·C²))` — the C×C WY systems are d_φ-blind;
cost grows **linearly** in d_φ. State memory also grows linearly in d_φ. Capacity (law above)
grows like `min(d_φ, ε²/σ₂²)`. So the frontier metric is **capacity per unit state**, i.e. exactly
`η(φ) = 1/(σ₂² d_φ)` — crosstalk efficiency. Measured (§1.3): none 0.63, quad2 ≈ 0.27,
pure-random floor 1.0. Orderings and consequences:

1. **The optimal-mixing result (derivation sketch, new):** with the monomial block scaled by a
   fixed scalar γ before L2 normalization (currently γ = 1), the effective quadratic mixing
   `λ` is a tunable function of γ, and the σ₂ of the blended kernel
   `(1−λ)ρ + λρ² + fluctuation` is minimizable in closed form — the optimum balances the linear
   term's `Var(ρ) = 1/d_h` against the quadratic block's sampling variance `≈ 9n2/d_h⁴`.
   At d_h = 32 the linear term dominates σ₂ unless `λ → 1`, i.e. **the sampled quad2 lever
   primarily helps by dilution (bigger, more-random φ-space), not by ρ → ρ² compression** — the
   doc's own framing overweights the squaring. Testable instantly on the probe by sweeping γ.
2. **Full quadratic map** (`d_φ = d_h(d_h+1)/2 = 528` at d_h=32): kernel `≈ ρ²` + self-monomials,
   `σ₂ ≈ √2/d_h ≈ 0.044` → `N* ≈ 512` at η ≈ 0.97 — **near the floor, ~3.5× quad2's capacity at
   2.1× its d_φ**. Higher-order quadk: full degree-k has `d_φ ~ d_h^k/k!` and kernel `ρ^k` →
   σ₂ = O(d_h^{−k/2}) — capacity explodes BUT only if n2 covers enough monomials; *sampled* cubic
   monomials have per-term mass `E[x_i x_j x_l]² ~ d_h^{−3}` so after L2 the map collapses back to
   its linear part unless n2 ≳ d_h³ — **sampled quadk is provably dominated; full quad2 is on the
   frontier; lowrank-quad2 trades η for state size and lands where the probe says.** The
   report's three-way d_φ reconciliation (32/128/137/256) gets a one-number arbiter: η.
3. **Trained keys beat random keys (mechanistic validation):** the probe measures *random* keys.
   The law predicts trained-model σ₂ (measurable by dumping `phi(k)` from trained MQAR checkpoints)
   is smaller — the network whitens its keys — and that per-seed recall accuracy *correlates with
   measured trained σ₂* across seeds and d_φ. That would upgrade the capacity law from
   random-feature theory to a *measured mechanistic account of the trained model* — the kind of
   causal, honest evidence this repo is built for.

### 2.4 Expressivity: the honest map of who wins where, and the Householder lever nobody is pulling (→ T6, T7, T8)

Known-lands (all **knowledge**, cite-with-care): log-precision transformers sit inside
L-uniform TC⁰ (Merrill & Sabharwal 2023); linear/selective SSMs inherit similar
constant-depth limits for serial state tracking (Merrill, Petty & Sabharwal 2024, "Illusion of
State"); empirically transformers fail non-solvable-group word problems without chain-of-thought
while handling commutative aggregation, and fail length extrapolation on state tracking
(Delétang et al. 2023; Liu et al. chain-of-thought/state-tracking lines); the S₅ word problem is
NC¹-complete (Barrington 1989) so a *proven* fixed-depth separation likely requires unresolved
circuit complexity — I will NOT promise one. Conversely, attention provably beats bounded-state
recurrent models on sparse-index retrieval at polylog width (Sanford, Hsu & Telgarsky 2024
representational-capacity line) — and the rank argument here is elementary: **a rank-d_φ linear
state cannot represent N > d_φ arbitrary pairs exactly; the KV cache can.** So the separation is
bidirectional and *quantitative*, with crossover at N ≈ d_φ — which is precisely the capacity law's
rank cap. The two regimes:

- **N ≲ N\*(φ): compressed-regime** — Prizma O(1)-state wins on params/memory; TF needs no extra
  depth but pays O(T) cache.
- **N ≫ d_φ: cache-regime** — attention is *provably* the right architecture (rank); Prizma must
  lose. State it; own it; it is the honest boundary of the "inevitable rival" claim, and the phase
  diagram (T8) is a paper-grade figure.

**The new lever (derivation sketch, new; Householder reading borrowed):** `M = I − β_e k kᵀ` with
‖k‖=1 is a **Householder reflection** at `β_e = 2` (eigenvalues {−1, +1 …}), and products of
reflections generate the orthogonal group. Any permutation matrix (hence the regular representation
of ANY finite group generator set, orthogonalized by Cayley's theorem) factors into ≤ d_φ
reflections; a D₄ 90° rotation = 2 reflections; an S₅ 5-cycle = 4 transpositions = 4 reflections;
the standard generators ⟨(1 2), (1 2 3 4 5)⟩ need **n_delta ≤ 5**. The repo already has the
`n_delta` (DeltaProduct) lever, currently `β_cap = 0.99` — i.e. **the codebase is one constant away
from depth-1, O(1)-state, exact group-generator composition**, approximable from *inside* the
contractive set by `β_e = 2σ(ℓ) ∈ (0,2)` (Theorem 1 keeps `‖J‖ = α` throughout — reflections
approached from the stable side). Prediction: a matched TF needs chain-of-thought (O(T) cache) or
growing depth for D₄/S₅ state tracking over long horizons; Prizma-with-reflections solves D₄ at
d_h ≥ 2, n_delta = 2, β→2, and **length-extrapolates** (composition is exact, position-free).
The DeltaProduct paper's state-tracking claims (knowledge) make this *credible*; the β = 2
Householder reading + contractive-envelope proof + regular-representation argument is the new,
cheap, provable part.

### 2.5 Hopfield unification (→ T7)

**[B]** Ramsauer et al. 2020: transformer attention = one update of a continuous modern Hopfield
network (exponential capacity, O(N·d) memory). **[B]** Hopfield 1982 Hebbian storage; **[B]**
Kohonen/Anderson 1972 linear associative memory — whose incremental delta rule converges to the
pseudo-inverse solution `S* = VΦᵀ(ΦΦᵀ)⁺` (classical). The precise statement Prizma can own:

> Additive linear attention = *one-pass Hebbian* correlation memory (biased, SNR-limited,
> capacity ~ 1/σ₂² with σ₂ = 1/√d_φ floor). Delta rule = *online exact solver* of the same
> association problem (one pass with β=1 solves the orthonormal-key case exactly; iterated
> passes/replay converge to the pseudo-inverse — Theorem 2's `(−E)^k` is exactly the stationary
> iteration for it). Softmax attention = *batch, memory-unbounded, one-step* modern Hopfield with
> exponential capacity. The three are one family parameterized by (memory budget, passes,
> temperature); Prizma's contribution is the **online, memory-bounded corner with a proven SNR
> capacity law and a contraction proof** — the corner the brain is also in (a hippocampus does not
> carry a KV cache of everything).

Falsifiable pair: (i) N > d_φ arbitrary-pairs task where TF solves and Prizma provably cannot
(rank) — publish as the honest boundary; (ii) N ≤ d_φ at large T where cache-memory forces the TF
into the Prizma memory regime — the O(1) advantage quantified *as a function of the capacity law*,
not as a slogan.

---

## 3. Proposals (ranked by impact × feasibility)

---

### T1. The crosstalk-spectrum capacity law + the predicted D-frontier (headline) **[N]**

- **WHAT:** Replace the Gershgorin corollary with the two-term decomposition `E = μ11ᵀ + W`;
  derive `Var[φ_iᵀφ_j | ρ]` in closed form for quad2/lowrank/none; state and prove
  `N* = min(1 + ε²/σ₂², d_φ)`; extend `feat_map_probe` to report `σ₂`, `μ`, `η`; fit ε once at
  D=64; **pre-register** N* predictions for the deferred P3 grid (D ∈ {16,32,64,128,256} ×
  d_φ ∈ {32,128,137,256,512} incl. lowrank-137) with a pre-registered acceptance band (e.g.
  predicted-vs-observed solve-transition within ±½ octave on ≥ 4 of 5 arms); run P3 as the
  adjudicator.
- **WHY:** Converts the repo's only explanatory theory from post-hoc to *predictive*; resolves the
  N<14-vs-D=128 contradiction (Flags A/B); makes the already-built probe the theory's measuring
  instrument; the deferred-then-decisive structure is maximally credible.
- **HOW:** numpy derivation + probe extension (CPU) → derivation note in `docs/` → pre-registration
  committed before the A100 P3 run (the P1–P5 scaffolding already exists in `gpu_bench.py`).
- **FALSIFIABLE TEST:** the pre-registered band above; plus the theory-internal test that the
  closed-form moments reproduce the probe's measured 0.076 (currently my envelope is off by
  1.3–1.6× — if the analytic gap does not close, the law dies without a GPU).
- **RISKS:** trained-model effects (key whitening, readout margins) may make ε non-constant across
  configs — mitigation: also run the trained-σ₂ measurement (T3's mechanistic leg); MQAR's
  dense-query protocol may shift ε — fix the protocol across arms.
- **IMPACT 10 / FEASIBILITY 8.** Brain tie-in: capacity laws are the shared language of
  hippocampal memory (bounded synapses, measured item capacities); a *predicted-then-confirmed*
  capacity curve is the strongest brain-plausible claim available that is not vibes.

---

### T2. Gate-aware law: the horizon identity + the SNR–decay window curve **[N]**

- **WHAT:** Prove the three-way role of `γ_t = Π α_τ` (stored magnitude, BPTT gradient, capacity
  window); derive `N*_eff(ᾱ) = min(N*, 1/(1−ᾱ²))` and the age-dependent SNR
  `ᾱ^{A_max}·sqrt(1−ᾱ²)`; pre-register the gated-MQAR prediction (ᾱ ≈ 0.999 collapses recall
  beyond ≈ 500 effective pairs, short-range intact); derive the noise-optimal
  `β* ≈ 1/(1+(N−1)σ₂²)` bias–variance law and check the learned `σ(W_β x)` against it on trained
  checkpoints.
- **WHY:** Unifies the two halves of the existing theory doc with one scalar; gives the write gate
  a principled target (currently it is learned with no stated optimum); directly relevant to
  char-LM where `gated=True` is used and recall degrades for reasons nobody has quantified.
- **HOW:** paper math + one gated-MQAR run on the existing harness + a checkpoint probe for
  learned-β correlation.
- **FALSIFIABLE TEST:** the pre-registered decay-window transition; the learned-β vs β* correlation
  (prediction: positive rank correlation across positions with high crosstalk load).
- **RISKS:** data-dependent α (not constant ᾱ) complicates the clean law — handle via the
  `gamma`-ratio machinery the chunked kernel already computes exactly (log-space cumsum).
- **IMPACT 7 / FEASIBILITY 8.** Brain tie-in: synaptic decay/consolidation tradeoffs — the
  window identity is the Ebbinghaus-style forgetting curve of the architecture, stated exactly.

---

### T3. The η-frontier: crosstalk efficiency as the design arbiter; optimal monomial mixing; full-quad2 dominance check **[N, η-metric floor borrowed from random-feature theory]**

- **WHAT:** Adopt `η(φ) = 1/(σ₂² d_φ)` as the reported metric for every feature-map proposal; add
  the fixed monomial-scale γ as a parameter-free knob (γ is a constant, not learned) and derive
  the σ₂-optimal γ*; show sampled-quadk is dominated (moment collapse for n2 ≲ d_hᵏ); run
  full-quad2 (d_φ = 528 at d_h = 32) as the frontier arm; extend the probe to *trained-model* keys
  and test the σ₂↔accuracy correlation across seeds.
- **WHY:** The current selection logic (crosstalk mean at fixed d_φ) is dimensionally wrong per §1.3;
  η ordering says quad2 is 3.6× off the floor — either the η headroom is monetizable (full-quad2 or
  γ*) or the trained-model measurements will show the floor is irrelevant because learning whitens
  keys. Both outcomes are publishable; one is a capacity windfall.
- **HOW:** probe extension only (numpy) for γ*/η/trained-σ₂; one MQAR cell for full-quad2 and one
  for γ*-quad2 at *half* d_φ (prediction: matches quad2-256's accuracy).
- **FALSIFIABLE TEST:** γ*-quad2@d_φ≈128 ≈ quad2@256 on MQAR-D128, ≥3 seeds; trained σ₂ < random
  σ₂ on ≥ 8/10 checkpoints; η predicts the solve-transition ordering across all arms of T1's grid.
- **RISKS:** full-quad2 doubles state size (the O(1)-state selling point is d_φ-proportional —
  disclose, don't hide); trained-key whitening may erase the γ* effect at convergence (then γ*
  helps optimization speed, not asymptote — still measurable via steps-to-ignite, which the repo
  already tracks).
- **IMPACT 8 / FEASIBILITY 8.** Brain tie-in: random-feature/efficient-coding arguments —
  cortex maximizes information per synapse; η is exactly that, for the delta state.

---

### T4. Optimal write-schedule theory: envelope, redundancy of surprise-β, redirection to α, and the Bellman view **[N]**

- **WHAT:** Write the short theory note: (a) contractive envelope `β_e ∈ [0,2]` (erase) with
  `β_w ∈ [0,1]` free on stability grounds; (b) β = 1 is the exact one-shot write for any α →
  surprise-*scaling* of β is provably redundant for storage and eats the contraction margin
  (`g·β ≤ 2`, margin ~2% at current caps); (c) pre-register the powered surprise-ablation outcome
  the repo already owes (surprise-norm ≈ constant ≥ random ≈ baseline at β_cap ≈ 1); (d) derive
  the myopic-optimal Cauchy step and its Bellman extension for repeated keys; (e) propose the
  theory-consistent brain-like mechanism: surprise modulates **retention (α)** and/or **read
  routing**, not write amplitude.
- **WHY:** The surprise gate is the repo's one "new mechanism" and the only collected evidence
  points against it; theory should either rescue it (by relocating it) or bury it with a
  prediction. Both are wins; silence is the only loss. Also gives the PC/free-energy identity an
  optimality statement it currently lacks.
- **HOW:** math note + the already-built ablation apparatus (`gpu_ablation.py`) at power, plus a
  learned-β* correlation check (shared with T2).
- **FALSIFIABLE TEST:** the pre-registered powered-ablation ordering; the β_e ∈ (1,2] envelope
  arms are T6's business — here only β_e ≤ 1 claims are made.
- **RISKS:** the powered run may show a real surprise-β effect at intermediate β_cap (< 0.5) where
  soft writes matter — that is fine and pre-register it as a secondary arm.
- **IMPACT 7 / FEASIBILITY 9.** Brain tie-in: neuromodulatory gating of *plasticity windows and
  retention*, with the honest note that delta-writing-the-error already IS the Hebbian/
  predictive-coding term — the brain-like part is the rule, not the extra multiplier.

---

### T5. The two-regime phase diagram: compressed-state vs cache — the quantitative attention/Prizma boundary **[N, rank argument elementary; Sanford et al. line borrowed for the attention side]**

- **WHAT:** One figure + one theorem sketch: for arbitrary-pair MQAR-style storage, solve-probability
  as a function of N/d_φ for Prizma (all feat_maps) and TF (param-matched): theory says Prizma's
  transition sits at `min(1+ε²/σ₂², d_φ)` and the TF's at ∞ (cache rank = N); plot both; overlay
  the T1 law as a curve, not points.
- **WHY:** "Inevitable rival" requires knowing exactly where the rival *must* lose and saying it
  first. This is the figure that makes the honesty load-bearing and the law visual.
- **HOW:** reuses T1's runs + a TF arm at two depths; no new harness.
- **FALSIFIABLE TEST:** observed Prizma transitions land on the law's curve within the T1 band;
  the TF arm shows no transition in the tested range.
- **RISKS:** TF cache advantage needs the needle to be *arbitrary* pairs — use the same dense-MQAR
  protocol on both arms or the comparison is confounded.
- **IMPACT 7 / FEASIBILITY 9.** Brain tie-in: the compressed-state corner is the brain's corner;
  the figure quantifies what the hippocampal analogy costs.

---

### T6. Householder DeltaProduct: β_e → 2 reflections, depth-1 group state tracking, the provable expressivity win **[B→N: Householder/delta-product reading borrowed (DeltaProduct line, knowledge); envelope proof + regular-representation argument + contractive-from-inside parameterization new]**

- **WHAT:** Extend the erase-gate domain to `β_e = 2σ(ℓ) ∈ (0,2)` (one bound change; Theorem 1
  unchanged: `‖J‖ = α` for the whole interval); compose n_delta reflections per token; derive the
  generator factorizations (D₄: n_delta = 2, d_h ≥ 2; S₅ via ⟨(1 2), (1 2 3 4 5)⟩: n_delta = 5);
  run the D₄/S₅ word-problem suite (train short, test 8–16× extrapolation) vs param-matched TF.
- **WHY:** The only candidate for a *structural*, not just efficiency, separation available at
  fixed depth with O(1) state: exact non-commutative composition where attention needs CoT (O(T)
  cache) or depth. It also re-frames `n_delta` from "more gradient steps" to "wider gate family" —
  one lever, two theories.
- **HOW:** theory note (factorizations + envelope proof) first; then one lever change + one task
  suite (tasks.py pattern) on the existing harness.
- **FALSIFIABLE TEST:** Prizma-reflection solves D₄ word problems at L=1–2, d_h ≥ 2, n_delta = 2,
  with ≥ 8× length extrapolation; matched TF fails at 2× (prediction from the state-tracking
  literature) or needs CoT. If the TF solves it at matched params without CoT, the separation
  claim dies — publish that.
- **RISKS:** learnability of exact reflection normals from tokens (mitigation: d_h slack,
  curriculum, β annealed 1→2); β→2 near the envelope boundary may interact with fp noise in the
  WY solve (the `_solve_unit_lower` conditioning under near-parallel repeated keys should be
  analyzed — see T9); DeltaProduct literature may already contain the β=2 observation (I know the
  n_delta lever; I do NOT know a β=2-reflection statement — verify before claiming novelty; if
  found, the contribution collapses to the envelope proof + the S₅ factorization discipline).
- **IMPACT 9 / FEASIBILITY 6.** Brain tie-in: weakest of the set (cortical rotation/mental-pantomime
  analogies exist but are loose) — keep honest, market it as structure, not biology.

---

### T7. The Hopfield unification note: attention = batch one-step; Prizma = online bounded-memory; one family, three corners **[B: Hopfield '82; Kohonen/Anderson '72; Ramsauer et al. '20; Schlag/Schmidhuber FWP; new: the corner map + SNR law as the family's capacity theorem]**

- **WHAT:** A short paper-shaped note placing additive linear attention (Hebb corner), delta rule
  (online exact-solver corner; `(−E)^k` is the stationary iteration for the pseudo-inverse), and
  softmax attention (modern-Hopfield corner) in one (memory, passes, temperature) diagram, with
  T1's law as the capacity theorem of the online corner and T5's phase diagram as its boundary.
- **WHY:** Gives the free-energy identity a 70-year lineage (borrowed, citable) and gives the
  brain-alignment axis a precise skeleton (hippocampus = online bounded-memory associative memory
  with surprise-gated writes — the analogy becomes a theorem-shaped claim instead of vibes).
- **HOW:** writing + reusing T1/T5 artifacts; optionally a multi-pass replay experiment (in-context
  rehearsal) showing capacity extension toward the rank cap — the delta rule's version of
  hippocampal replay, and a genuinely brain-like capability the TF structurally lacks at O(1) state.
- **FALSIFIABLE TEST:** replay prediction: k passes reduce residual as `‖E‖₂^{k}` (Theorem 2) —
  measurable exactly on synthetic MQAR without any learning (pure kernel test, CPU-cheap).
- **RISKS:** reviewers may say "linear attention = kernel regression is known" — true; the new
  part is the *online-solver capacity law + contraction + replay* package, and the note must cite
  the kernel-regression line honestly.
- **IMPACT 8 / FEASIBILITY 7.** Brain tie-in: strongest framing asset of the set; replay-as-capacity
  is directly hippocampal (and testable without training).

---

### T8. Expressivity honesty note: what Prizma provably cannot do (rank cap), what attention provably cannot do at fixed depth without cache/CoT, and the open gap between them **[B: Merrill/Sabharwal, Delétang, Sanford et al. (knowledge); new: the rank-cap statement for rectangular delta states and the mapping to the tested suite]**

- **WHAT:** A one-page statement of limits, written *before* any separation claim: (i) rank cap —
  no rank-d_φ linear state stores N > d_φ arbitrary pairs (proof: dimension); (ii) fixed-depth
  delta-state models cannot benefit from cache-bought capacity; (iii) proven fixed-depth
  separations vs TC⁰-bounded transformers are NOT available without resolving TC⁰ vs NC¹-type
  questions — promise only the *structural* (rank) and *empirical* (T6) claims; (iv) map the
  existing §4 suite (induction, selective-copy) onto this map — and note honestly that B2/B3 were
  non-discriminating because both regimes solve them.
- **WHY:** The repo's credibility is its currency; a theory section that pre-declares its own
  inability to prove certain separations is worth more than one that implies them.
- **HOW:** writing only.
- **FALSIFIABLE TEST:** n/a (it is the falsifiability infrastructure itself).
- **RISKS:** none; cost is one page.
- **IMPACT 6 / FEASIBILITY 10.** Brain tie-in: states the true biological claim — brains are also
  rank- and SNR-bounded, which is *why* they replay, sleep, and forget; Prizma inherits the same
  three necessities, which is the deepest available brain-likeness statement.

---

### T9. Numerical contraction of the chunked solver: conditioning of `(I+A)` under repeated keys; predicted FP-instability thresholds **[N; the log-space gamma machinery already in code]**

- **WHAT:** Bound the condition number of the within-chunk WY system `(I + A)`, `A_ij ∝ β_e·(γ_i/γ_j)(k_j·k_i)`:
  worst case near-parallel repeated keys → `κ ≈ Π(1 + β_e·ratio)` growth; derive the
  chunk-size/adversarial-key condition `C·β_e·max-ratio ≲ 1` for float32; validate against
  deliberately adversarial chunks (the harness already has a repeated-key torture test pattern).
- **WHY:** The kernel's correctness guarantees are exact-equivalence tests at random keys; the
  *structured worst case* (what an adversary or a degenerate data stream supplies) is unquantified,
  and T6 pushes β_e toward the boundary where this bites first.
- **HOW:** math + a small numerical study (not a training run).
- **FALSIFIABLE TEST:** predicted FP error growth vs observed under adversarial chunks at
  C ∈ {16,32,64}, β_e ∈ {1, 1.5, 2−δ}.
- **RISKS:** may conclude "fine at float32, irrelevant until β→2" — then it is one paragraph in T6's
  appendix (still worth having before T6).
- **IMPACT 5 / FEASIBILITY 7.** Brain tie-in: none (pure systems hygiene); include because
  contraction claims should cover the solver, not just the recurrence.

---

## 4. Borrowed-vs-new ledger (theory program)

| Item | Source | Status |
|---|---|---|
| Jacobian contractiveness Theorem 1 | repo (`docs/quad2_theoretical_convergence.md`) | established |
| Residual recursion `R = −V(−E)^k` (Theorem 2) | repo; gradient-view of delta rule borrows DeltaNet/FWP lineage (Schmidhuber '92 delta-rule FWP; Schlag et al. '21; Yang et al. '24) | established (repo) / borrowed (lineage) |
| Gershgorin capacity corollary `N < 1+1/cross` | repo | established **but flagged** (Flags A–C) |
| Mean/fluctuation split `E = μ11ᵀ + W`; SNR law `N* = min(1+ε²/σ₂², d_φ)`; horizon identity | **new** (T1/T2) — Wigner/row-noise argument is standard random-matrix reasoning **[B-method]** | new derivation, unproven |
| Monomial moment computation, η metric, γ*-optimal mixing | **new** (T3); random-feature/kernel floor `σ₂ = 1/√d_φ` **[B-method]** | new derivation |
| Write-envelope `β_e ∈ [0,2]`, β=1 exactness, Cauchy/Bellman schedule | **new** (T4); Cauchy steepest-descent step **[B classical numerics]** | new derivation |
| Householder reflections, regular representation, NC¹/S₅ (Barrington), TF ⊆ TC⁰ (Merrill–Sabharwal), Illusion-of-State, Chomsky-hierarchy empirics, Sanford et al. sparse-retrieval separation | **[B]** all (knowledge) | applied to Prizma in T5/T6/T8 — the β_e→2 contractive-from-inside parameterization is the new bit |
| Hopfield corners map (Hebb / Kohonen pseudo-inverse / Ramsauer modern Hopfield) | **[B]** Hopfield '82, Kohonen/Anderson '72, Ramsauer '20, FWP line | new as an explicit *package* with the SNR law as its capacity theorem (T7) |
| WY solver conditioning bound | **new** (T9); WY representation **[B]** (DeltaNet) | new derivation |

## 5. What I deliberately do NOT claim

- No proven fixed-depth separation between Prizma and transformers at log precision — the
  circuit-complexity gap (TC⁰ vs NC¹-flavored questions) is open; T6's claim is *structural +
  empirical*, and T8 says so.
- No claim that the capacity law is proven — it is a derivation sketch with a numeric discrepancy
  (§1.2) that must close first; the proposal is structured so that closure is CPU-cheap and the
  falsification is pre-registered.
- No numbers for any experiment not yet run (P3 remains deferred; the surprise ablation remains
  owed and quarantined-adjacent in status).
- No claim that η or γ* helps *asymptotic* accuracy if trained keys whiten — only that the law
  predicts where transitions sit, whatever the trained σ₂ turns out to be.

---

## QUESTIONS FOR THE OWNER

1. **Corrections authorization (Flags A–C):** may I draft the one-paragraph corrections-section
   note for README §4 / the theory doc ("Gershgorin explains the linear failure, not the quad2
   success; capacity-law program pending")? The repo's own culture demands it, but it touches the
   headline README, so it is an owner call.
2. **Is "parameter-free feature map" a hard brand constraint?** T3's γ* and full-quad2 stay
   parameter-free (fixed buffers/constants); a *learned* φ would beat both on η but breaks the
   zero-parameter selling point. Hard constraint or disclosable trade?
3. **May β_cap leave [0,1]?** T6 requires `β_e ∈ (0,2)`. Theorem 1 survives unchanged, but
   "u_t is one gradient step on free energy" becomes "one (possibly over-relaxed) gradient step,
   β ∈ (0,2]" — a restatement of the PC identity, not just a constant. Owner-approved wording
   needed before any implementation exists to ablate.
4. **GPU budget:** T1's decisive run is the long-deferred P3 grid (~45 cells was called
   intractable). With the law pre-registered, a reduced decisive subset (~15 cells: 5 D × 3 d_φ)
   suffices. Is that budget acceptable, and should T1's pre-registration be committed before any
   GPU time is booked?
5. **Committee standard:** should `η(φ) = 1/(σ₂²·d_φ)` and the second-moment probe (σ₂, μ) become
   required reported metrics for every future feature-map proposal (analogous to the existing
   crosstalk pre-filter), so capacity claims are law-comparable across the commission's rounds?
6. **Surprise mechanism disposition:** if T4's pre-registered prediction confirms redundancy of
   surprise-β, is the owner's preference to (a) relocate the mechanism to α/retention as
   theory-consistent and re-run, or (b) retire the claim and keep the delta-write-as-free-energy
   identity as the sole PC novelty? (Both are honest outcomes; choosing in advance prevents
   motivated re-runs.)
