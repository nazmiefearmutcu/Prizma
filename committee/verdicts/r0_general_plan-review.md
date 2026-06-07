# Council 1 (General Council) — Round 0 Gate Verdict

> **Decision under review:** Prizma-Seq v2 spec (`docs/superpowers/specs/2026-06-07-prizma-seq-v2-pareto-dominance-design.md`) + Phase 0/1 implementation plan (`docs/superpowers/plans/2026-06-07-prizma-seq-v2-pareto-dominance.md`), incl. all embedded code.
> **Date:** 2026-06-07 · **Council:** General (the gate) · **Members:** statistics/experimental-design · optimization · information-theory · reproducibility/integrity · ML-systems · adversarial-skeptic.

## VERDICT: **NEEDS-EXPERIMENTS / CONDITIONAL-REJECT**

The honest-science *framing* is strong and the borrowed-vs-new ledger discipline is genuinely better than most programs at this stage. But the plan ships **statistical machinery that is wrong in the anti-conservative direction** (it will manufacture false "we beat the TF" significance at n=10) and a **novel-core (lever A) chunked approximation that silently breaks on the exact task it is meant to win** (recall with repeated keys), with an O(1) guard that does not certify the training path. Phase 1 may NOT start until the binding remedies below are applied. This is a gate rejection of the *current artifacts*, not of the research direction.

The single most dangerous property of the plan as written: **every error we found points the same way — toward an easier, falsely-significant "win."** A gate exists to catch exactly that asymmetry.

---

## Lens 1 — Statistics / Experimental Design (REJECT)

### 1a. The normal-approx p-value is anti-conservative at n=10 (CRITICAL)
`seq/stats.py::_p_one_sided_from_t` uses `p = 1 - Φ(t)`, ignoring df, justified by the comment "df>=9 -> t ~ z within ~5%". That 5% is **on the statistic, not on the decision**, and at the decision boundary it is catastrophic:

- At df=9, the true one-sided 5% critical value is `t=1.833`. The normal approximation assigns `t=1.833 → p=0.0334`. So a result that a real t-test calls **non-significant (p=0.05)** is reported as **significant (p=0.033)**.
- The false-positive zone is the whole band `1.645 < t < 1.833` (df≈9): normal says "win", the t-test says "no win." With 10 seeds and Welch df typically 9–18, this is precisely the regime where a razor-thin char-LM result will land.
- Direction of error: **inflates Type-I error → manufactures "Prizma beats TF"**. This is the worst possible direction for a program whose explicit risk is overclaiming "dramatic" wins.

**Remedy (binding):** Replace the normal tail with an actual one-sided Student-t survival function. No SciPy is needed — use the regularized incomplete beta `I_x(a,b)` (Lentz continued fraction, ~30 lines) so `p = 1 - T_cdf(t, df)` for the upper tail; equivalently use `betainc(df/2, 1/2, df/(df+t^2))/2` for the tail. Validate against `scipy.stats.t.sf` in `tests/test_stats.py` at df ∈ {5,9,15,30} to <1e-4. Until then, no superiority p-value may be quoted.

### 1b. The TOST `crit*0.84` factor is a wrong-by-construction CI (CRITICAL)
`tost_equivalence` builds `crit = _t_crit(round(df)) * 0.84`, calling it "~90% CI half-width factor vs 95% table". Two errors:

1. **The constant is wrong.** `0.84 ≈ z_0.95/z_0.975 = 1.645/1.96` is the *normal* ratio; the t ratio `t_{0.95,df}/t_{0.975,df}` is df-dependent and *smaller*: 0.810 (df=9), 0.827 (df=20), 0.835 (df=60). The 0.84 constant overstates the half-width by +0.030 (df=9) to +0.013 (df=20) — i.e. it makes the 90% CI **too wide**, which (for equivalence, where the CI must fit inside ±margin) is conservative for declaring equivalence but is still **the wrong number** and will be wrong in the other direction the moment anyone reuses `_t_crit`'s 1.96 normal fallback at large df.
2. **TOST is not a "shrink the two-sided crit" operation.** Correct TOST = two one-sided tests at α each: equivalent iff `t_lower = (diff+margin)/se > +t_{1-α,df}` AND `t_upper = (diff-margin)/se < -t_{1-α,df}`. The CI-equivalent is the **(1-2α) CI built with `t_{1-α,df}` directly** — i.e. `crit = t_{0.95,df}` (the one-sided 95% / two-sided 90% point), NOT `t_{0.975,df}*0.84`.

**Remedy (binding):** Implement TOST as two explicit one-sided t-tests using the df-correct `t_{0.95,df}` from the t-survival function of 1a (add a `_T90`/one-sided table or compute it). Report both one-sided p-values and the (1-2α) CI. Add a unit test that the CI half-width equals `t_{0.95,df}*se` exactly.

### 1c. "Beats by ≥0.03 BPC" is under-specified as a superiority hypothesis
A one-sided superiority test (H1: mean(Prizma BPC) < mean(TF BPC)) tests *any* improvement, not ≥0.03. To claim a **≥0.03 margin**, you need a **non-inferiority/superiority test against a shifted null** (H0: μ_TF − μ_Prizma ≤ 0.03, reject to claim >0.03), i.e. a one-sided test on `(diff − 0.03)/se`. As written, the plan would declare "beats by ≥0.03" whenever the point estimate clears 0.03 and the p<0.05 superiority-vs-zero passes — that conflates a point estimate with a tested margin. Also note BPC is **lower-is-better**: confirm the sign convention in `superiority_test` callers (the plan's tests use accuracy, higher-is-better; the char-LM gate must flip it).

**Remedy:** Add a `margin_superiority(a, b, margin)` testing `mean(b) − mean(a) > margin` (df-correct t). Pre-register the sign per leg. The §3 "≥0.03 BPC" cell is only "won" when this test rejects, not when the point estimate clears.

### 1d. Multiple comparisons across legs are uncorrected
§3 has ≥7 axes, each with its own ≥10-seed test, plus per-lever ablations (A/B/C/D each with on/off + control). Running ~15+ significance tests at α=0.05 with no correction gives a family-wise false-positive probability ≈ 1−0.95^15 ≈ 54%. The "Pareto-dominance on every axis" claim is a **conjunction** (all must hold) so for the *positive* claims a conjunction is self-protecting — BUT the per-lever causal-attribution tests ("surprise beats random-scalar control", "k=2 makes selcopy discriminating") are independent discovery claims and DO need correction.

**Remedy:** Pre-register the leg→test map. Apply Holm-Bonferroni (or report adjusted p) across the *causal-attribution* family. State explicitly that Pareto-dominance positives are a conjunction (no inflation) but each discovery sub-claim is corrected.

### 1e. n≥10 + TOST-superiority is the right *story* but power is unverified
Ten seeds is a floor, not a guarantee. With per-seed BPC SD unknown (the report only has n=2: [1.751, 1.749] → SD≈0.0014, suspiciously tiny and likely under-dispersed because init was NOT seed-pinned), the detectable effect at n=10, 80% power, α=0.05 is roughly `2.5·SD/√10·... `. If true SD is ~0.01 BPC, n=10 detects ~0.013 BPC — fine for a 0.03 target. If SD is ~0.03 (plausible once init is genuinely seed-varied), n=10 is underpowered for 0.03.

**Remedy:** Run a 1-seed→3-seed pilot to estimate SD *after* the seed-pinning fix, then compute required n. Treat 10 as provisional; bump if the power calc demands it. Record the power calc in the verdict JSON.

---

## Lens 2 — Optimization (NEEDS-EXPERIMENTS)

### 2a. 5-point grid @ 1 seed is the right shape but the single seed is a confound
The per-config LR sweep (`sweep_lr`, grid `(5e-4,1e-3,1.5e-3,2e-3,3e-3)`, 1 seed → pick best, then ≥N seeds at the winner) is the correct two-stage design and directly addresses the report's documented optimization-confounding (the d208L4H4 0/3 that was excluded). **But:** MQAR/induction exhibit a *seed-dependent phase transition* (the report's own headline caveat — TF was 2/3 with a 0.785 at one config). Picking the best LR on **1 seed** can select an LR that happened to ignite on that seed and fails on others, or miss an LR that ignites on most seeds but not seed-0. Stage-1 LR selection on a bimodal task is itself noisy.

**Remedy:** For the bimodal diagnostic legs, Stage-1 must use ≥3 seeds per LR and select on **solve-rate then median** (not best_acc on 1 seed). For the smooth char-LM leg, 1 seed is acceptable (loss is smooth). Make this leg-conditional in `sweep_lr` (add a `stage1_seeds` arg). Record full grid incl. rejected LRs (already planned — keep).

### 2b. "Optimal LR ~ 1/width" is the right μP intuition but unverified here
The report attributes d208's failure to "optimal LR scales ~1/width per μP" — but explicitly says this is **inferred, not separately confirmed** (the per-width re-sweep was never run). The plan inherits this assumption as the *justification* for the sweep harness but never tests it. μP's LR-transfer also requires μP **initialization and per-layer LR scaling**, which the current `_init` (flat std=0.02 on all Linears) does NOT implement. So "1/width" may not even hold for this codebase.

**Remedy:** Do not assume 1/width. Sweep the full grid at every width (the harness already does). Either (a) implement actual μP (input/output/hidden-layer scaling) and *verify* coordinate-check / LR transfer empirically, or (b) drop the "μP-aware" label from the harness and call it what it is: a per-config grid sweep. Honesty: the plan/spec say "μP-aware" — that's an overclaim until coordinate-check passes.

### 2c. Banded-window + feature-map changes WILL shift the optimal recipe
Lever D (leaner feature map, d_φ 256→~128) changes the delta-state conditioning and effective key-rank; lever C (output gate + state RMSNorm) changes gradient scale at the read; lever A (surprise gate) changes the effective write LR per token. Each of these moves the loss landscape, so an LR swept on the *base* config is stale for the *combined* config.

**Remedy (binding):** The LR sweep must be re-run **for the exact lever combination being scored**, not once for the base. The Phase-1 per-lever A100 steps must each carry their own Stage-1 sweep. State this in the plan's per-task gates (currently only Task 0.4 mentions it generically).

---

## Lens 3 — Information Theory / Capacity (NEEDS-EXPERIMENTS)

### 3a. The rectangular-delta-state capacity argument is sound and quantified
Round-1's `S = Σ uᵢkᵢᵀ` online-ridgeless analysis (crosstalk var ≈ (D−1)/d_h, D* ≈ d_h for orthogonal keys, capacity linear in d_h) is correct textbook linear-associative-memory theory, and the quad2 monomials genuinely lift effective key-rank (crosstalk 0.141→0.076 at D=128, ≈ the true-d=128 oracle 0.071, with the rand_linear control showing no gain). This is the strongest part of the whole program. **No objection to the existing quad2 capacity claim.**

### 3b. Lever A (surprise gate) plausibly RE-WEIGHTS, does not add capacity (CRITICAL nuance)
The spec leads with A as "the novel core" adding "LM + long-context recall" capacity. Information-theoretically, `β_eff = β_t·(1+tanh‖ε_t‖)` is a **per-token scalar gain on the write** — it does not change the rank-d_h ceiling of `S`, the key-rank, or d_φ. It can only **re-allocate** the fixed capacity (write surprising tokens harder, decay stale ones faster). That is a real, useful inductive bias (Titans-style) but it is **not** a capacity increase, and the surprise-modulated forget (α release on large surprise) can *destroy* recall capacity if it evicts still-needed bindings. The spec's claim that A adds capacity is theoretically unsupported; A should be framed as a **capacity-allocation / salience** mechanism.

**Remedy:** Reframe A's claim in the spec from "add capacity" to "reallocate fixed capacity by salience." The §3 bar for A must be a *recall-under-distractors* or *LM* improvement at fixed d_h/d_φ, with the random-scalar control (already planned) AND a second control: a **constant β_eff = mean(1+tanh‖ε‖)** to separate "more average write" from "surprise-targeted write." Without the constant-mean control, a win could just be a higher effective write LR (which the LR sweep should have found anyway).

### 3c. Lever D (lean map, D=128 recall at half d_φ) — the capacity probe gate is correct but the threshold is asserted
Plan Task 1.D requires `quad2_lowrank` crosstalk ≤ 0.085 at D=128 ("within 0.01 of quad2's 0.076"). But crosstalk is *necessary, not sufficient* for end-to-end recall (round-1 guardrail explicitly says so). A low-rank-then-quadratic map at r-dim has `r(r+1)/2` monomials of rank ≤ r(r+1)/2 but spanning a structured (not generic) subspace; it may hit the crosstalk number while losing recall on adversarial key distributions.

**Remedy:** The D-gate must be **end-to-end MQAR-D128 solve-rate at ≥10 seeds**, not just the crosstalk probe. Keep the probe as a fast pre-filter, but the binding gate is recall parity with quad2-256/128 at matched seeds. Also resolve 3d first (you cannot claim "half d_φ" if the baseline d_φ is unknown).

### 3d. The d_φ value is INCONSISTENT across three documents (CRITICAL integrity)
- `committee/round1_synthesis.md` §C: "φ: d_h=32→**d_φ=128**", crosstalk 0.076.
- `docs/PRIZMA_SEQ_REPORT.md`: "**d_φ=256** throughout"; FLOP table "quad2-256"; "d_φ 32→256 multiplies delta-state FLOPs ~5.5×"; the entire **2.14× FLOP claim** is computed at d_φ=256.
- `seq/prizma_seq.py`: `feat_n2: int = 96` → `d_φ = d_h + 96 = 32+96 = **128**` (the code default).

So the headline 2.14× FLOP number (the thing levers D/E/F must beat down to ≤1.0×) is computed at a d_φ the code does not use by default, and "lean feature map to ~half d_φ (96–128 vs 256)" is incoherent if the deployed map is already 128. **The entire FLOP narrative rests on an unverified, inconsistent d_φ.**

**Remedy (binding):** Reconcile d_φ in code, report, and synthesis. Re-emit the FLOP ledger from the *actual* config used for the recall results. If the recall results were obtained at d_φ=256 (feat_n2=224), set that as the default and fix the code comment; if at 128, the report's 2.14× and "5.5×" must be recomputed and lever D's "half d_φ" target re-derived. No FLOP claim may be quoted until this is closed.

---

## Lens 4 — Reproducibility / Integrity (REJECT)

### 4a. Seed-before-build fix is correct but INCOMPLETE — three residual RNG-order leaks
`build_and_train` (seed → build → train) is the right primitive and the test (two runs, |Δloss|<1e-6) is a valid gate. But the fix is necessary, not sufficient:

1. **`train_model` re-seeds with the *training* seed (`set_seed(seed)`, common.py:152) AFTER the frozen eval set is drawn (`_frozen_eval_batches` calls `set_seed(cfg.eval_seed)`, common.py:135).** Good — eval is independent. But `build_and_train` calls `set_seed(seed)` then `model_fac()` then `train_model(...)` which calls `set_seed(cfg.eval_seed)` for eval batches and **then `set_seed(seed)` again** before the optimizer. So the *init* RNG (between the first `set_seed(seed)` and `model_fac`) is pinned (the fix works), but verify no `torch.randn`/dropout/`torch.rand` runs between them. The `precision_gate='random'` path (`prizma_seq.py:167`) draws `torch.rand` **inside forward every step**, with no generator → its stream depends on all prior CUDA RNG draws and is NOT reproducible across the eval/train re-seeds. The random-scalar *control* for lever A (the causal-attribution gate!) likely uses this path → the control itself is non-reproducible.
2. **The quad2 monomial buffers use a private `Generator().manual_seed(1234)` (prizma_seq.py:123)** — good, deterministic. But they are created inside `model_fac()` *after* the global `set_seed(seed)`; a private generator is fine, just confirm no global-RNG consumption order dependence.
3. **MPS vs CUDA non-determinism:** the guards run on MPS; headline numbers on A100/CUDA. `torch.use_deterministic_algorithms(True)` is not set anywhere; SDPA (window head) and several reductions are non-deterministic on CUDA by default. Bit-reproducibility across A100 runs is therefore NOT guaranteed even with seed-before-build.

**Remedy (binding):** (a) Route every stochastic op through an explicit `torch.Generator` (esp. `precision_gate='random'` and any lever-A random-scalar control). (b) Set `torch.use_deterministic_algorithms(True, warn_only=True)` + `CUBLAS_WORKSPACE_CONFIG` on the A100 runner and re-run the repro test on CUDA, not just MPS. (c) Extend `test_repro.py` to a Prizma model with `feat_map='quad2'` and the lever knobs on, on the headline device.

### 4b. The O(1) guard does NOT certify the surprise-gate's *training* path (CRITICAL)
This is the most serious integrity hole. The guards (`step()==forward()<1e-4`) only hold at **chunk=1** for lever A by the plan's own admission (Task 1.A Step 6: "guard at chunk=1 for the hard equality and document the chunked gap"). But **training runs at chunk=64** (`PrizmaSeqConfig.chunk=64`). So:

- The **inference** path (`step()`, chunk=1-equivalent) and the **training** path (chunked forward, chunk=64, frozen-S0 surprise) compute **different functions** whenever surprise is on. The model is trained as one function and deployed as another.
- The frozen-S0 within-chunk surprise is **not a small approximation on structured data.** I verified on the canonical recall pattern (a repeated key — the exact MQAR/induction signal): true sequential ‖ε‖ collapses 1.0 → 0.1 → 0.01 → 0.001 after each write, but the frozen-S0 surprise stays 1.0, 1.0, 1.0, 1.0. The gate `g=1+tanh‖ε‖` is then ~2.0 (frozen) vs ~1.0 (true) — a **100% error in the write gain** on precisely the tokens recall depends on. The plan's `test_surprise_chunk64_approx_within_tol < 5e-3` tests **random** q/k/v, which has no repeated structure, so it will pass while the training-relevant case fails silently.

**Remedy (binding):** Either (a) make the chunked surprise path **exact** (two-pass: first pass computes the true per-row S_{i-1} surprise, second pass applies it — the spec §9 already lists this fallback; promote it from fallback to default), OR (b) scope lever A to **inference-time only** with training on the un-gated path (also in §9), OR (c) reduce chunk to 1 for the surprise path (kills the speed lever F's premise for that arm). The chunk-64 frozen approximation as the *training* path is rejected. The equivalence test MUST include a **structured (repeated-key) case**, not just random tensors, with the same <1e-4 bar as every other guard — no special 5e-3 tolerance for the novel core.

### 4c. The §3 "DRAMATIC" bar invites overclaiming on three axes
- **Latency "faster at all n ≳ 2k":** the report's measured truth is Prizma is ~1.3–1.5× *slower* below n≈16k and crosses only at n≥32k, because it is overhead-bound (Python per-chunk loop + full-T² SDPA window). The 2k target depends ENTIRELY on lever E (banded) AND lever F (fused kernel) both landing. Pre-registering "all n ≳ 2k" before either kernel exists is an aspirational number masquerading as a bar.
- **Length-extrap "abs ≥0.70 @4×, ≥0.50 @8×":** current abs is 0.40 @8×. No lever in A–F targets absolute length-extrap. This number has no mechanism behind it.
- **Per-FLOP "≤1.0×":** rests on the unresolved d_φ (3d) and on D+E+F all succeeding.

**Remedy:** Mark each §3 target as **conditional on its enabling lever passing**, and require the bar to be **re-pre-registered after Phase 1** when the kernels exist (the spec already says Council 1/3 "refine and raise before each run" — make it explicit that targets without a landed mechanism are *hypotheses*, not bars). An axis with no mechanism (abs length-extrap) must be declared a Pareto knob now, not a win target.

---

## Lens 5 — ML-Systems (NEEDS-EXPERIMENTS)

### 5a. Banded window (lever E) — implementation is correct, but the latency premise is incomplete
`_window_banded` (chunked local SDPA, span ≤2w per query block) is numerically sound and the equivalence test (`<1e-4` vs `_window`) is the right gate. It cuts the window head's FLOPs (17.5%→~0.9%). **But** the report attributes Prizma's sub-16k slowness to *two* causes: the T² window AND the **sequential Python per-chunk delta loop**. Lever E fixes only the first. The delta loop is the dominant overhead at training and likely at the prefill that precedes decode. So "crossover 32k→~2k" from E alone is not credible; it needs F (fused kernel) too.

**Remedy:** Re-state the latency target as gated on E **and** F. Measure E's isolated effect (decode latency with banded window, fused off) and report it honestly — do not attribute the full 32k→2k move to E.

### 5b. Triton fused kernel (lever F) — feasibility is real but high-risk and under-specified
A fused chunked-delta WY/UT Triton kernel matching `chunked_delta` (rectangular state, gated path, surprise path, n_delta>1) to <1e-4 *including the backward pass* is a multi-week specialist task. The plan's Task 1.F is one paragraph ("implement a Triton fused kernel for the WY/UT chunk step") with no kernel design, no tiling/block-size plan, no backward derivation. The `_solve_unit_lower` triangular solve inside the chunk is the hard part to fuse (it's sequential within the chunk). FLA exists and is the realistic path (wrap `fla` ops) rather than hand-rolled Triton.

**Remedy:** Down-scope F to "integrate FLA's chunked-delta kernel where the math matches, else document the Python-loop tax honestly as a Pareto knob." The plan already says speed is "a Pareto knob, not a correctness gate" — good; make the default expectation "use FLA or disclose," not "write Triton." Equivalence-incl-grad on CUDA is the only hard gate.

### 5c. 4×A100-via-Colab + Chrome automation — operationally fragile, not a science risk
Driving 4 parallel Colab A100 sessions via logged-in Chrome is doable but flaky (session timeouts, preemption, the ~50 min/run the report already hit that made the D-frontier "intractable"). This is an execution risk, not a gate-blocker, IF checkpoints are truly crash-safe.

**Remedy:** Verify the atomic-JSON checkpoint actually resumes (a test that kills mid-run and resumes). Budget for preemption. Not a Phase-1 blocker.

### 5d. Latency target realism: "faster at all n ≳ 2k" — see 4c/5a
Below ~2k, Prizma carries a fixed per-step overhead (state read/write of d_h·d_φ per head per layer, the conv ring, the window softmax) that a single SDPA call (the TF at short n, fully fused) does not. At n=2k with small d, the TF's O(n²) has not yet dominated its constant. Beating the TF at 2k requires Prizma's per-step constant < the TF's per-step constant — plausible only with a fused kernel (F) and even then not guaranteed at tiny scale.

**Remedy:** Treat 2k as a hypothesis to be *measured* after E+F, not a pre-registered floor. Report the actual crossover, wherever it lands.

---

## Lens 6 — Adversarial Skeptic (the one most-likely misleading-win path)

**The single most likely way this program produces a false "we beat the Transformer" headline:**

> A combined model (C+A+D) clears char-LM "by ≥0.03 BPC" at 10 seeds with `superiority_test` p=0.041 — where (i) the p=0.041 is an artifact of the **normal-tail approximation** (true t-test p≈0.06, non-significant — Lens 1a), (ii) the seeds are **less dispersed than reality** because a residual RNG-order leak (the `precision_gate='random'` control, MPS→CUDA non-determinism) left init partially pinned (Lens 4a), shrinking the SE and inflating t, and (iii) the TF arm was **LR-swept on 1 bimodal seed** that happened to land on a weak LR (Lens 2a), quietly handicapping the baseline. Each effect is individually small and individually defensible; **stacked, they convert a true tie into a reported 0.03 win.** The "honest" ledger and O(1) guards all stay green because none of them test the statistics or the baseline's LR fairness.

**Secondary path:** The surprise-gate (the *novel* claim) "wins" because its chunk-64 training path silently runs a **2× effective write LR on recall tokens** (the frozen-S0 surprise bug, Lens 4b) — i.e. the "novel mechanism" is an accidental, non-reproducible LR hack that the random-scalar control does not catch (the control is a *random* scalar, not a *constant-mean* scalar), and that fails to reproduce at inference (chunk=1 step()).

**The gate that catches it (binding):**
1. **Re-run every headline significance test through `scipy.stats.t` (or a verified incomplete-beta t-CDF) and require the *t-test* p<0.05, not the normal-approx p.** Cross-check one result by hand.
2. **Adversarial baseline rule:** the TF arm gets the *same* multi-seed Stage-1 LR selection as Prizma (≥3 seeds for bimodal legs), and the chosen TF LR + full rejected grid is in the verdict JSON. A reviewer must be able to see the TF was not handicapped.
3. **Negative control that must FAIL:** run the *entire pipeline* with Prizma == a copy of the TF mixer (or two identical TFs with different seeds). If the pipeline ever reports a significant "win" of one identical model over the other, the stats are broken — this is the canary. Pre-register it.
4. **Lever A causal attribution needs the constant-mean control**, not only the random-scalar control (Lens 3b), AND must reproduce step()==forward() with the *exact* (two-pass) training path (Lens 4b).
5. **Seed dispersion sanity:** report per-seed raw values; if char-LM SD < 0.005 BPC across 10 genuinely-varied inits, treat as suspicious (under-dispersion → likely a repro leak).

---

## Required remedies before Phase 1 (binding, ranked)

| # | Remedy | Lens | Blocks |
|---|---|---|---|
| R1 | Replace normal-tail p-value with a verified one-sided **Student-t** survival fn (incomplete-beta); validate vs scipy at df∈{5,9,15,30}. No p-value quoted until done. | 1a | ALL accuracy claims |
| R2 | Reimplement TOST as two one-sided t-tests using df-correct `t_{0.95,df}`; delete the `0.84` constant; unit-test the (1-2α) CI. | 1b | char-LM equivalence/superiority |
| R3 | Make lever A's chunked surprise path **exact (two-pass)** OR scope A to inference-only OR chunk=1; add a **repeated-key structured** equivalence test at <1e-4 (not 5e-3 on random). | 4b, 3b | lever A (novel core) |
| R4 | Reconcile **d_φ** across code/report/synthesis; re-emit the FLOP ledger from the actual config; no FLOP claim until consistent. | 3d | all FLOP/per-FLOP claims, lever D target |
| R5 | Add `margin_superiority(a,b,margin)` (test diff>0.03, df-correct) and pre-register sign per leg (BPC lower-is-better). "Beats by ≥0.03" only when this rejects. | 1c | char-LM headline |
| R6 | Adversarial baseline fairness: ≥3-seed Stage-1 LR selection for bimodal legs (solve-rate→median, not best_acc@1seed); re-sweep LR per **lever combination**; "μP-aware" label dropped unless coordinate-check passes. | 2a, 2b, 2c | every head-to-head |
| R7 | Negative control (identical-model "win" must be non-significant) pre-registered + run as the stats canary; Holm correction on the causal-attribution family; multiple-comparison map pre-registered. | 6, 1d | causal-attribution claims |
| R8 | Route all stochastic ops through explicit `torch.Generator`; set deterministic algorithms + re-run the repro test on **CUDA**; extend it to a quad2+levers model. | 4a | reproducibility of every number |
| R9 | Lever A second control = **constant-mean β_eff**; lever D gate = end-to-end ≥10-seed MQAR-D128 recall, not just the crosstalk probe; reframe A in spec as capacity-reallocation, not capacity-add. | 3b, 3c | novelty claim, lever D |
| R10 | Re-mark §3 latency/abs-length-extrap/per-FLOP targets as **conditional on enabling levers landing**; declare abs-length-extrap a Pareto knob (no mechanism); re-pre-register the bar after Phase-1 kernels exist. | 4c, 5a, 5d | "DRAMATIC" bar integrity |

## What Council 1 WOULD approve (scope of conditional approval)
- The Phase-0 *structure* (seed-before-build wrapper, per-config LR sweep harness, param/FLOP ledger auto-emit, council charters, verdict log) — these are the right primitives and may be **built**, but R1/R2/R8 must be applied to `seq/stats.py` and `seq/common.py` **inside Phase 0** before any Phase-1 accuracy run consumes them.
- The lever **ordering** (C→E→A→B→D→F by value/risk) and the TDD/two-gate (G1 O(1), G2 chunked==reference) discipline.
- The existing **quad2 capacity result and its causal control** (Lens 3a) — untouched, credible.
- Levers **C and E** may proceed to implementation immediately after R1/R2/R8 land (they carry the lowest integrity risk; C is trainable-param-adding → confirm the ledger grows the TF in lockstep, which Task 1.C Step 7 already states).

**Lever A may NOT proceed to any accuracy run until R3 + R9 are applied. No FLOP/per-FLOP claim until R4. No significance claim until R1+R2+R5+R7.**

## Evidence references
- `seq/stats.py` (in plan §0.3) lines `_p_one_sided_from_t` (normal tail), `tost_equivalence` (`crit*0.84`).
- `seq/common.py:135,152` (set_seed ordering); `seq/prizma_seq.py:167` (`precision_gate='random'` no-generator), `:60` (`feat_n2=96`→d_φ=128), `:123` (seeded buffer).
- `docs/PRIZMA_SEQ_REPORT.md:46,143,161` (d_φ=256, 2.14× FLOP) vs `committee/round1_synthesis.md:53` (d_φ=128).
- plan Task 1.A Step 4/6 (frozen-S0 surprise, chunk=1 guard only, 5e-3 random-data test).
- `docs/PRIZMA_SEQ_REPORT.md:223,231` (latency: ~1.5× slower below 16k, crossover n≥32k) vs spec §3 ("faster at all n ≳ 2k").
- spec §2 lever A ("novel core", "add capacity") vs Lens 3b (re-weighting, not capacity).
