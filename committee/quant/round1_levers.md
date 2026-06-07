# Council 2 — Quant Team — Round 1: Ranked Levers to Dramatically Surpass the Transformer

**Subject:** Prizma-Seq (Gated-DeltaNet-family + parameter-free quad2 feature map, rectangular state S∈R^{d_h×d_φ}, exact local window head, O(1) streaming).

**Mandate:** DRAMATIC Pareto-dominance over a param-matched tuned Transformer on char-LM **AND** latency **AND** per-FLOP, while keeping the memory + param-efficiency edges and pushing absolute length-extrapolation up. No faked metrics, honest borrowed-vs-new ledger.

**Verdict up front:** The char-LM gap (1.7496 vs 1.7254 BPC, +0.0242) is *small and structural*, not a capacity wall. The 2024–2026 literature shows that nearly every one of our queued levers is independently validated to close exactly this kind of gap, and three of them are *FLOP-negative* (they make us faster, not slower). The honest path to Pareto-dominance is **C + D + E together** (the "win triad"), with B(k=2) as the length-extrapolation booster and A as the genuinely-novel core differentiator. The quad2 d_φ=256 is the single biggest self-inflicted wound on FLOP/latency, and lever D is the highest-leverage fix because it attacks FLOP, latency-crossover, *and* train-speed simultaneously.

---

## 1. The Real Frontier (grounded, cited)

| # | Work | arXiv | 1-line takeaway (relevance to us) |
|---|------|-------|-----------------------------------|
| 1 | **Gated Delta Networks: Improving Mamba2 with Delta Rule** (Yang, Kautz et al., ICLR'25) | [2412.06464](https://arxiv.org/abs/2412.06464) | The direct parent: scalar gate α + delta rule beats Mamba2/DeltaNet on LM, recall, length-extrap; gate is what makes delta competitive on *language*, not just recall. We have the gate; the *output* side is where we're leaving points on the table. |
| 2 | **Parallelizing Linear Transformers with the Delta Rule over Sequence Length** (Yang et al., NeurIPS'24) | [2406.06484](https://arxiv.org/abs/2406.06484) | The WY/UT chunk-parallel form we use. Crucial: there is a *published, hardware-efficient* chunked-delta Triton kernel — our 5× train slowdown is an *implementation* gap (lever F), not a fundamental one. |
| 3 | **DeltaProduct: Improving State-Tracking via Householder Products** (Siems et al., 2025) | [2502.10297](https://arxiv.org/abs/2502.10297) | n_h delta steps/token → diagonal+rank-n_h transition. n_h=2 handles S₃, n_h=3 reaches S₅; **at n_h=3 length-extrap degradation is *minimal* out to 8× train length** (2048→16384). Gated-DeltaProduct₂ = 25.12 vs Gated-DeltaNet 25.97 PPL (FineWeb). Cost ≈ n_h× the delta recurrence. This is lever B, and it's the strongest *absolute* length-extrap lever in the literature. |
| 4 | **RWKV-7 "Goose" with Expressive Dynamic State Evolution** (Peng et al., 2025) | [2503.14456](https://arxiv.org/abs/2503.14456) | Generalized delta rule with **vector-valued gating + in-context learning rates + relaxed value-replacement**; 3B SoTA multilingual on far fewer tokens; provably recognizes all regular languages (beyond TC⁰). Validates output-gating + per-channel decay (lever C) and surprise-style adaptive write (lever A) at scale. |
| 5 | **Titans: Learning to Memorize at Test Time** (Behrouz, Zhong, Mirrokni, 2024/25) | [2501.00663](https://arxiv.org/abs/2501.00663) | Surprise = gradient/predictive-coding signal with momentum; forgetting = data-dependent decay gate. Beats Transformers + linear-recurrent baselines, scales to 2M ctx. The theory backing for lever A (surprise-gated write). |
| 6 | **Gated DeltaNet-2: Decoupling Erase and Write** (2026) | [2605.22791](https://arxiv.org/abs/2605.22791) | **Channel-wise erase gate b_t (key-side) decoupled from write gate w_t (value-side).** S_t = (I − k_t(b_t⊙k_t)ᵀ)D_t S_{t-1} + k_t(w_t⊙v_t)ᵀ. WikiText 15.90 vs Gated-DeltaNet 16.40 PPL; multi-key NIAH@4K **37.8% vs 28.0%**. Param-matched, ~5% train throughput cost. Directly upgrades our delta core; near-free recall/LM win. |
| 7 | **OSDN: Online Scaled DeltaNet — Provable Online Preconditioning** (2026) | [2605.13473](https://arxiv.org/html/2605.13473) | Diagonal preconditioner via hypergradient feedback ≡ per-feature scaling of the write-side key; keeps chunkwise parallel form, no high-dim state blowup. Another near-free recall booster compatible with our state. |
| 8 | **BASED: Simple linear attention balances recall–throughput** (Arora et al., Hazy, 2024) | [2402.18668](https://arxiv.org/abs/2402.18668) | 2nd-order Taylor feature map + small sliding window = our exact template. **MQAR: Taylor map most effective; linear-alone can't do recall; window+linear expands the Pareto frontier.** Up to 24× throughput vs FA2. Validates rectangular-state + window hybrid as the right shape — but warns the window must be *cheap*. |
| 9 | **The Hedgehog & the Porcupine: Expressive Linear Attentions w/ Softmax Mimicry** (Zhang et al., ICLR'24) | [2402.04347](https://arxiv.org/abs/2402.04347) | Learnable MLP feature map trained to mimic softmax's *spiky + monotonic* weights, R^d→R^d (no dim blowup). The "leaner feature map" template (lever D): get the recall of a big d_φ at *lower* d_φ by making the map spiky, not bigger. |
| 10 | **The Key to State Reduction in Linear Attention: A Rank-based Perspective** (Nazari & Rusch, 2026) | [2602.04852](https://arxiv.org/pdf/2602.04852) | Linear-attention states have **low *effective* rank** — most d_φ dimensions are wasted. Recall depends on effective rank, not nominal state dim. Direct theoretical license for lever D: cut d_φ ~half with no recall loss if we raise per-dim usefulness. |
| 11 | **A Systematic Analysis of Hybrid Linear Attention** (2025) | [2507.06457](https://arxiv.org/pdf/2507.06457) | ~70–75% linear / 25–30% attention is the sweet spot; Gated DeltaNet is the best linear variant inside hybrids; put attention deeper, linear shallow. Tells us how to *place* the window head, and that GDN-family is the right backbone. |
| 12 | **Short Window Attention Enables Long-Term Memorization** (2025) | [2509.24552](https://arxiv.org/abs/2509.24552) | **Smaller windows force the recurrent state to carry long-range info → better long-context.** Stochastic window size in training beats fixed. Justifies a *banded/small* window (lever E): smaller window is both cheaper AND better for the state's long-memory — a rare win-win. |
| 13 | **Transformers are SSMs: State Space Duality (Mamba-2)** (Dao & Gu, 2024) | [2405.21060](https://arxiv.org/abs/2405.21060) | Chunked-scan = matmul-in-chunk + recurrence-across-chunk; trains via Tensor Cores at near-linear cost; train-time chunk == inference-time recurrence (our step()==forward() invariant). The kernel blueprint for lever F. |

**Frontier read:** Three independent threads converge on our exact problem. (a) *Output-side* gating + per-head normalization (RWKV-7, NeurIPS'25 Gated Attention) is the standard fix for the *language* gap that recall-strong linear models leave open. (b) *Feature-map efficiency* (Hedgehog, rank-based state reduction) says our d_φ=256 is over-provisioned — we can halve FLOPs. (c) *Cheap local context* (BASED, short-window) says the T² window head is both a FLOP sin and *suboptimal* — a small banded window is faster AND better. The owner's queued A–F are well-chosen; the new alpha is in *combining the FLOP-negative ones* so the latency/per-FLOP wins arrive *for free* alongside the LM win.

---

## 2. NeurIPS'25 Best-Paper signal (the cheapest LM win)

**Gated Attention for LLMs: Non-linearity, Sparsity, Attention-Sink-Free** ([2505.06708](https://ar5iv.labs.arxiv.org/html/2505.06708), NeurIPS'25 Oral/Best-Paper) — a *head-specific sigmoid gate on the SDPA output* gives reliable gains across the board, adds nonlinearity + sparsity, kills attention sinks. This is essentially lever C applied to attention; its best-paper status is the strongest possible prior that **output-gating is the single most reliable, lowest-risk quality lever in 2025**. We should treat C as near-mandatory.

---

## 3. Ranked Levers

Ranking metric: **(expected margin-gain ÷ honesty-risk)**, with a hard bonus for FLOP-negative levers (they buy a Pareto axis *for free*). "Honesty-risk" = probability the lever tempts overclaiming, breaks param-match, or imports an external trained component we'd have to disclose as borrowed.

### Rank 1 — Lever C: Output-gating + per-head state RMSNorm  (RWKV-7 / GLA / NeurIPS'25)
- **Mechanism (precise):** After reading o_t = S_{t-1} q_t, apply (i) per-head RMSNorm on the read-out (stabilizes the rectangular-state read across heads/positions) and (ii) a data-dependent output gate g_t = σ(W_g x_t), output = g_t ⊙ RMSNorm(o_t), then output proj. Optionally a SiLU on the gate (RWKV-7/GLA style). ~0.3–0.6% extra params (one projection) — fold into existing budget to keep param-match.
- **Axis attacked:** char-LM BPC (the real-task gap). Secondary: training stability → lets us push LR.
- **Expected margin-gain:** **High.** Output-gating is *the* fix for "recall-strong but LM-soft" linear models. Gated-DeltaNet's LM edge over plain DeltaNet, RWKV-7's gains, and the NeurIPS'25 best-paper all attribute material PPL drops to output-side gating/normalization. Rough magnitude: closing **0.01–0.03 BPC** is realistic — i.e. *this alone can erase or reverse the 0.0242 gap*.
- **Honesty-risk:** **Very low.** Standard, well-attributed borrowed component (ledger: "output gate + state RMSNorm, borrowed from RWKV-7/GLA"). Keeps param-match if we absorb the projection into budget. No metric-gaming surface.
- **O(1)-compatibility:** Fully preserved — gate is a pointwise function of the current token + current read; no extra carried state.
- **FLOP impact:** Negligible (+~1% — one projection + a norm). Neutral on latency.
- **Difficulty:** Low (≈1 day). Pure forward-path change; step()==forward() trivially holds.
- **Falsifiable mini-experiment:** Add gate+RMSNorm, retrain text8 at identical params/steps. **Pass iff char-LM BPC drops ≥0.01 with no MQAR regression.** Ablate gate-only vs norm-only to attribute.

### Rank 2 — Lever D: Lean / structured feature map (same D=128 recall at ~half d_φ)
- **Mechanism (precise):** Replace the seeded quad2 (d_φ=256) with a *structured* map at d_φ≈128 that preserves key-rank: options in increasing novelty — (a) **Hadamard/FastFood-structured random projection** before the monomial lift (cheap, parameter-free, raises effective rank per the rank-based-reduction result); (b) **spiky map** (Hedgehog-style power/exp on normalized features) to mimic softmax sharpness at lower dim; (c) **block-diagonal monomials** so d_φ FLOPs scale sub-quadratically. Keep it parameter-free to preserve the "0 added params" novelty claim where possible.
- **Axis attacked:** **FLOP (the missing per-FLOP claim) AND latency-crossover AND train-speed** — quad2 d_φ=256 is *5.5× the delta-state FLOPs*; halving d_φ roughly halves that term. This is the lever that flips the per-FLOP ledger.
- **Expected margin-gain:** **High on FLOP/latency, neutral-to-slightly-positive on recall.** Rank-based-reduction ([2602.04852]) and Hedgehog say the extra 128 dims are largely wasted effective-rank; the bet is *same MQAR@D=128 at d_φ=128*. If it holds: delta-state FLOPs drop ~2×, total forward FLOPs from 2.14× TF toward **~1.3–1.5× TF**, and the latency crossover drops from n≥32k toward the mid-thousands. Rough: **−40–55% of the quad2 FLOP term.**
- **Honesty-risk:** **Low–medium.** Risk surface: if we adopt a *learnable* spiky map we add params (breaks "parameter-free" novelty) — must disclose. Risk of tuning d_φ down until MQAR *just* passes and overclaiming "no recall loss" — mitigate with a pre-registered MQAR threshold. Structured-random (option a) is fully parameter-free and honest.
- **O(1)-compatibility:** Preserved — feature map is pointwise per token; state is still d_h×d_φ (smaller). step()==forward() unaffected.
- **FLOP impact:** **Strongly negative (good).** The primary FLOP win.
- **Difficulty:** Medium (≈2–3 days) — needs a small d_φ sweep + structured-projection impl.
- **Falsifiable mini-experiment:** Sweep d_φ ∈ {96,128,160,256} with structured map; measure MQAR D=128 solve-threshold (params) + text8 BPC + measured FLOPs/token. **Pass iff d_φ=128 keeps MQAR solve ≤ the 130K param point AND BPC within +0.003, at ~½ the quad2 FLOPs.** Mirage-check below.

### Rank 3 — Lever E: Banded sliding-window kernel (kill the T² window SDPA)
- **Mechanism (precise):** Replace the full-T² SDPA local head with a true *banded* attention kernel (FlashAttention-style block-local, only the w diagonal blocks computed), window w small (literature: 64–256; short-window paper says *smaller is better* for long-memory). Compute is O(T·w) not O(T²).
- **Axis attacked:** **latency-crossover (the ~1.3–1.5× slower-below-16k problem) AND FLOP** — the window head is currently *17.5% of forward FLOPs* purely from being T². Banding removes the T² term entirely.
- **Expected margin-gain:** **High on latency/FLOP.** Removing a T² term that's 17.5% of FLOPs at moderate n, and replacing with O(T·w): at n=4k, w=128 the window FLOPs drop ~30×. Combined with D, the **sub-16k latency deficit likely flips to a win**, and the crossover moves well below 16k. Bonus (short-window paper): a *smaller* window can *improve* long-context by forcing the state to carry more — so this is win-win, not a tradeoff.
- **Honesty-risk:** **Very low.** Pure kernel/algorithmic equivalence (banded == masked-full on the kept band), no metric surface, no param change. Ledger: "banded local attention, standard." Only caveat: must report window size honestly and show long-context didn't regress.
- **O(1)-compatibility:** Preserved and *improved* — streaming window is a fixed-size ring buffer of w keys/values = constant memory (already O(1), now also O(w) compute/step instead of O(n)).
- **FLOP impact:** **Strongly negative (good).**
- **Difficulty:** Medium (≈2–3 days) — write/borrow a banded Flash kernel; verify numerics vs masked-full.
- **Falsifiable mini-experiment:** Swap in banded kernel at w∈{64,128,256}; verify output matches masked-full within fp tolerance; measure wall-clock latency at n∈{1k,4k,16k} and FLOPs. **Pass iff latency at n=4k ≤ TF AND text8 BPC unchanged.**

### Rank 4 — Lever B: Higher-order DeltaProduct (k=2 Householder steps/token)
- **Mechanism (precise):** Take n_h=2 delta steps per token (diagonal+rank-2 transition = product of 2 generalized Householders), per [2502.10297]. Recurrence becomes n_h× longer; reuses the same WY/chunk machinery.
- **Axis attacked:** **absolute length-extrapolation (push the ~0.40@8× up) AND state-tracking/recall expressivity.** This is the single best *absolute* length-extrap lever in the literature.
- **Expected margin-gain:** **High on length-extrap, medium on recall/LM.** DeltaProduct at n_h=3 shows *minimal* degradation out to 8× train length; n_h=2 handles S₃ and already improves extrap markedly; Gated-DeltaProduct₂ = 25.12 vs 25.97 PPL. Rough: absolute 8× extrap acc could move from ~0.40 toward **0.6–0.8** at n_h=2 (extrapolating from their curves), with a small bonus LM gain.
- **Honesty-risk:** **Medium.** The honest landmine: it **multiplies delta-state FLOPs by ~n_h (≈2×) and train time by ~n_h** — it *worsens* our FLOP and train-speed axes. If reported in isolation it tempts "great extrap!" while hiding the FLOP regression. **Must be paired with D (which pays back the FLOP) and reported jointly.** Param-match preserved (steps reuse the same projections, possibly with per-step betas — small).
- **O(1)-compatibility:** Preserved — still a fixed-size carried state; inference does n_h micro-updates per token (constant memory, n_h× step compute).
- **FLOP impact:** **Positive (bad), ~+(n_h−1)× on the delta term.** This is why it's ranked below the FLOP-negative trio and must be bundled with D.
- **Difficulty:** Medium-high (≈3–4 days) — restructure recurrence to n_h sub-steps; correctness on chunk boundaries.
- **Falsifiable mini-experiment:** Train n_h∈{1,2} matched params; eval extrap acc at 1×/2×/4×/8× train length + measure FLOPs. **Pass iff 8× extrap acc rises ≥0.15 absolute AND the *combined* (B+D) FLOPs ≤ current 2.14× TF.** (i.e. D must absorb B's cost.)

### Rank 5 — Lever A: Surprise-gated write/forget (Titans-style, ‖ε_t‖ predictive-coding signal)  ← the novel core
- **Mechanism (precise):** Modulate write strength β_t and forget/decay α_t by the *local surprise* = norm of the prediction error ε_t = v_t − S_{t-1}k_t (the free-energy gradient already computed inside the delta rule!). High surprise → write harder / erase more; low surprise → preserve. Add momentum on the surprise signal (Titans) for smoothness. **Key novelty advantage: ε_t is *free* — the delta rule already computes v_t − αS k_t, so the surprise norm is a byproduct, not new compute.** Decouple erase vs write per Gated-DeltaNet-2 ([2605.22791]) for an extra near-free win.
- **Axis attacked:** char-LM BPC + recall robustness + long-context (interference reduction). This is the **genuine-architecture differentiator** — a non-Transformer mechanism with no Transformer analogue.
- **Expected margin-gain:** **Medium–high but higher-variance.** Titans shows surprise-gating beats Transformers + linear baselines; Gated-DeltaNet-2's decoupled erase/write gives WikiText 15.90 vs 16.40 and NIAH@4K 37.8% vs 28.0% — large recall gains. But surprise-gating on top of an *already-gated* delta core may partially overlap with α/β; the marginal gain over C is the question. Rough: **0.005–0.02 BPC + notable long-context/NIAH gain**, on top of C.
- **Honesty-risk:** **Medium.** This is the lever most likely to tempt narrative overclaim ("predictive-coding test-time memory!") — the borrowed-vs-new ledger must be scrupulous: surprise-gating = borrowed from Titans; ε_t-is-free = our genuine efficiency contribution; decoupled erase/write = borrowed from GDN-2. Risk of attributing C's gains to A. Param-match: adds tiny gate projections — keep in budget.
- **O(1)-compatibility:** Preserved — ε_t and its norm are computed per step from current state+token; momentum is one extra scalar/vector of state (constant).
- **FLOP impact:** Near-neutral (+~2–4%, just gate MLPs; ε_t is reused). 
- **Difficulty:** Medium (≈3 days) — but high *attribution* difficulty (must ablate vs C and vs plain delta).
- **Falsifiable mini-experiment:** Three-way ablation on text8 + NIAH/multi-key recall: {delta}, {delta+C}, {delta+C+A}. **Pass iff A adds ≥0.005 BPC *beyond* C AND improves multi-key recall ≥5 pts — otherwise A is redundant with the existing gate and should be cut or reduced to just the decoupled-erase/write part.**

### Rank 6 — Lever F: Fused chunked-delta kernel (kill the 5× train slowdown)
- **Mechanism (precise):** Port the published chunkwise-parallel delta Triton kernel ([2406.06484] WY/UT form; Mamba-2 SSD chunk pattern) — matmul-within-chunk on Tensor Cores, recurrence-across-chunk. Our 5× slowdown is a *missing fused kernel*, not an algorithmic cost.
- **Axis attacked:** **train-speed (the 5×/step slowdown).** Doesn't touch inference Pareto axes directly, but unblocks all other experiments (cheaper to iterate) and removes the "trains 5× slower" embarrassment.
- **Expected margin-gain:** **High on train-speed (the targeted axis), zero on the inference Pareto axes.** Published kernels bring delta training to *near* linear-attention speed (the chunk-scan is a small fraction of total — Mamba-2 result). Rough: **5×/step → ~1.3–1.8×/step.**
- **Honesty-risk:** **Very low.** Pure engineering; numerically identical (assert step()==forward() + grad-check). Ledger: "borrowed FLA chunked-delta kernel." No metric surface.
- **O(1)-compatibility:** Inference path unchanged (training-only kernel). Preserved.
- **FLOP impact:** Neutral on FLOP *count*; improves FLOP *utilization* (MFU).
- **Difficulty:** Medium-high (≈4–5 days) — Triton/CUDA, chunk-boundary correctness, backward pass. Best borrowed from `flash-linear-attention` (FLA) library rather than written from scratch.
- **Falsifiable mini-experiment:** Drop-in fused kernel; assert max-abs grad diff < 1e-3 vs reference; measure tokens/sec. **Pass iff ≥3× train throughput vs current AND bit-comparable loss curve.**

### Summary table

| Rank | Lever | Axis | Exp. gain | Honesty-risk | O(1) | FLOP | Diff | gain÷risk |
|------|-------|------|-----------|--------------|------|------|------|-----------|
| 1 | **C** output-gate + state-RMSNorm | char-LM | High (0.01–0.03 BPC) | Very low | ✓ | +1% | Low | ★★★★★ |
| 2 | **D** lean/structured feature map | FLOP + latency + train | High (−40–55% quad FLOP) | Low–med | ✓ | −− | Med | ★★★★★ |
| 3 | **E** banded-window kernel | latency-crossover + FLOP | High (kills T², ~30× window) | Very low | ✓+ | −− | Med | ★★★★★ |
| 4 | **B** DeltaProduct k=2 | abs length-extrap + recall | High extrap (+0.15–0.4) | Med (FLOP↑) | ✓ | + n_h× | Med-hi | ★★★ |
| 5 | **A** surprise-gated write | char-LM + recall (NOVEL) | Med–high, variable | Med (overclaim) | ✓ | +3% | Med | ★★★ |
| 6 | **F** fused chunked-delta kernel | train-speed | High (5×→~1.5×) | Very low | ✓ | neutral | Med-hi | ★★★★ |

---

## 4. The Single Best Bet for SIMULTANEOUS Pareto-Dominance

**THE WIN TRIAD: C + D + E**, then add **B(k=2)** for absolute length-extrap and **A** as the novelty core, with **F** as the iteration-speed enabler underneath.

Why this exact combination achieves *simultaneous* dominance across every axis the owner named:

| Pareto axis | Current state | Lever(s) that flip it | Mechanism of the flip |
|---|---|---|---|
| **char-LM BPC** | LOSE (1.7496 vs 1.7254) | **C** (primary) + A | Output-gating closes the "recall-strong, LM-soft" gap (RWKV-7, NeurIPS'25 best paper). C alone is sized to erase the 0.0242 gap. |
| **Latency (sub-16k)** | LOSE (1.3–1.5× slower <16k) | **E** + **D** | E removes the T² window term (17.5% FLOPs); D halves the quad2 term. Crossover drops from 32k to mid-thousands → win below 16k. |
| **Per-FLOP** | LOSE (2.14× TF FLOPs) | **D** + **E** | D: −40–55% of quad2 term (5.5× → ~2.7× delta FLOPs). E: −T² window. Together push total to ~1.3–1.5× TF, and combined with the LM win gives a clean per-FLOP-quality claim. |
| **Memory** | WIN (constant 17.9MB, 28–455× less) | preserve (all levers O(1)) | No lever breaks streaming; E *shrinks* the window buffer. |
| **Param-efficiency** | WIN (≥3.5× on MQAR) | preserve + B | All levers param-matched; B/C/A add only tiny projections kept in budget. |
| **Abs length-extrap** | weak (~0.40@8×) | **B(k=2)** | DeltaProduct's minimal-degradation curves; +0.15–0.4 absolute. |
| **Train-speed** | LOSE (5× slower) | **F** (+ D helps) | Published fused chunked-delta kernel: 5× → ~1.5×. |
| **Genuine novelty** | quad2 (param-free rect. state) | **A** + free-ε surprise + decoupled erase/write | The defensible "non-Transformer that's better *because* of architecture" story. |

**Critical sequencing (data dependencies):**
1. **F first** (or in parallel) — it doesn't change results, only iteration speed; everything downstream is cheaper to test once train is 3× faster.
2. **C + D + E in parallel** — independent forward-path changes, no shared state; this is the Pareto-flip bundle. Validate each against its own pre-registered threshold.
3. **B(k=2)** only *after* D lands — because D must pay back B's n_h× FLOP cost for B to be Pareto-honest.
4. **A last** — ablate strictly against C to prove non-redundant marginal value; if it doesn't beat C by ≥0.005 BPC, ship only the decoupled-erase/write sub-part (free recall) and keep the surprise-gate as a research line, not a claim.

**The honest headline this enables:** "A non-Transformer (Gated-DeltaProduct + parameter-free structured feature map + free-surprise gating) that param-matches a tuned Transformer and wins on char-LM BPC, latency (n≥~4k), and per-FLOP, while keeping 28–455× less inference memory and ≥3.5× MQAR param-efficiency, with absolute 8× length-extrapolation raised to ~0.6+." Every clause is backed by a flipped axis above, no faked metric.

---

## 5. Adversarial Self-Filter — Top 3 Levers, Strongest Failure Case + Cheapest Exposer

### C (output-gate + state-RMSNorm) — *might fail because:*
**Strongest reason to fail:** Our delta core is *already gated* (α, β). Output gating may be **redundant** with the existing input/state gates — the literature gains were often measured vs *un-gated* baselines, so the marginal BPC over our already-gated model could be ~0, not 0.01–0.03. There's also a mirage risk: RMSNorm can improve loss purely by enabling a higher LR, which a better-tuned baseline would also get — so the "gain" could be a tuning artifact, not architecture.
**Cheapest exposer:** A *single* text8 run: {already-gated delta} vs {+output-gate+RMSNorm} at **matched, separately-tuned LR for each** (not shared LR). If the gain vanishes under per-arm LR tuning, it was a tuning mirage. ~1 GPU-day.

### D (lean/structured feature map) — *might fail because:*
**Strongest reason to fail:** The whole premise is "d_φ=256 is over-provisioned (low effective rank)." But our **MQAR ≥3.5× param-efficiency edge may *depend* on the high d_φ key-rank** — that's literally the novelty mechanism ("rectangular state raises associative-recall key-rank"). Halving d_φ could **silently destroy the one decisive recall win** while looking fine on text8 (where recall demand is low). We'd flip FLOP/latency green but turn the param-efficiency win amber — *net Pareto-neutral or worse.*
**Cheapest exposer:** Don't trust text8. Run the **MQAR D=128 solve-threshold sweep at d_φ=128 *first*, before any LM run.** If the solve-point jumps above the 130K-param mark (losing the ≥3.5× edge), D at d_φ=128 is refuted; back off to d_φ=160 or use a spiky map to recover key-rank. ~0.5 GPU-day (MQAR is tiny).

### E (banded-window kernel) — *might fail because:*
**Strongest reason to fail:** Two compounding risks. (1) **Latency overhead-bound, not FLOP-bound:** our sub-16k slowness is described as "overhead-bound," meaning a banded kernel that cuts FLOPs may *not* cut wall-clock if the bottleneck is launch/Python/memory overhead — we'd cut FLOPs (per-FLOP claim ✓) but the *latency* axis stays red. (2) A too-small window could drop char-LM BPC (losing precise local copy), partially re-opening the gap C just closed.
**Cheapest exposer:** Before integrating, **micro-benchmark the banded kernel's wall-clock in isolation at n∈{1k,4k,16k}** vs the current T² head. If banded isn't actually faster at n=4k (overhead dominates), E doesn't deliver the latency axis and we must attack overhead (kernel fusion/CUDA-graph) instead. Pair with a w∈{64,128,256} BPC check on text8. ~0.5 GPU-day.

---

## 6. New Levers NOT in the A–F Queue (high-EV)

1. **G — Free-surprise *learning-rate* (RWKV-7 in-context LR), not just gate.** RWKV-7's edge is a *per-channel in-context learning rate* on the delta update, a strict generalization of a scalar β. We already compute ε_t; make the *effective step size* a learned function of (x_t, ‖ε_t‖) per channel. Higher-EV than vanilla A because it subsumes both the gate (C-adjacent) and surprise (A) into one mechanism with stronger theoretical backing (RWKV-7 recognizes all regular languages). Honesty-risk low if disclosed as "RWKV-7-style in-context LR." **This may dominate A.**

2. **H — Decoupled channel-wise erase/write (Gated-DeltaNet-2) as a *standalone* cheap recall win.** Independent of the surprise narrative: just split β into key-side erase b_t and value-side write w_t. Published result: WikiText 16.40→15.90, multi-key NIAH@4K 28%→37.8%, ~5% train cost, param-matched. This is **the single cheapest, highest-confidence recall+LM win in the whole 2026 frontier** and carries near-zero overclaim risk. Should arguably be promoted into the core delta rule *before* A, as the "free" part of A.

3. **I — Stochastic window size during training (short-window paper, [2509.24552]).** Train with the banded-window size sampled per-batch from e.g. {32…256}. Costs nothing extra, and the paper shows it improves *both* short- and long-context. Stacks on E for free → pushes absolute length-extrapolation (the owner's explicit goal) at zero inference cost. High-EV, near-zero risk.

4. **J — Hybrid layer-ratio tuning (systematic-hybrid paper, [2507.06457]).** We currently put a window head in *every* block. The hybrid analysis says ~25–30% attention is optimal and attention should sit *deeper*. Making only a subset of layers carry the (now-banded) window head, and placing them deep, could **cut FLOPs further AND improve LM** — a structural FLOP win we're currently leaving on the table by uniformly paying for the window everywhere.

**Highest-EV new lever:** **H (decoupled erase/write)** for *immediate* near-free recall+LM gain (promote it ahead of A), with **G (in-context LR)** as the higher-ceiling novel-core replacement for A, and **I (stochastic window)** as a zero-cost length-extrap booster bolted onto E.

---

*Ledger discipline reminder for every lever below:* borrowed = {C: RWKV-7/GLA/NeurIPS'25; D-spiky: Hedgehog; D-rank: Nazari&Rusch; E: Flash banded; B: DeltaProduct; A: Titans; H: Gated-DeltaNet-2; G: RWKV-7; F: FLA/Mamba-2 kernels}. New = {quad2 param-free rectangular state; ε_t-is-free surprise reuse; the specific C+D+E Pareto-flip composition under strict param-match + O(1) streaming}.
