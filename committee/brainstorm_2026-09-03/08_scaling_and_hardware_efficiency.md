# Brainstorm — Scaling Laws & Training Dynamics / Hardware Efficiency lens

**Commission:** Prizma → (1) an *inevitable* Transformer rival + (2) as brain-like as possible
**Author:** Scaling Laws & Training Dynamics Scientist (recurrent/chunked-parallel optimization,
signal propagation, plateau transitions, hardware-aware kernels, throughput engineering)
**Date:** 2026-09-03
**Classification (per brainstorming skill):** **Architectural-class research-direction brainstorm.**
HARD-GATE honored: **no code, no implementation** — report only. The skill's interactive
clarifying questions cannot reach the owner from inside a committee; all direction-setting
questions are collected in **QUESTIONS FOR THE OWNER** at the end.

Sources read before writing: `README.md` (throughput table + fabricated-rows correction),
`docs/PRIZMA_SEQ_REPORT.md`, `docs/HANDOFF.md` (§ fair protocol lessons),
`docs/quad2_theoretical_convergence.md`, `seq/prizma_seq.py`, `seq/delta.py`,
`seq/delta_triton.py`, `seq/delta_fused.py`, `seq/throughput_benchmark.py`,
`seq/benchmark_results.md`, `seq/lrsweep.py`, `seq/common.py` (TrainConfig/plateau rule),
`flop_ledger.py`, `gpu_bench.py`, `results/gpu_latency.json` (raw A100 decode cells).

Borrowed-vs-new discipline: every proposal marks **[B]** borrowed / **[N]** new /
**[B→N]** borrowed-component-with-new-framing. Every number quoted below is from the repo's
committed artifacts; estimates are labeled *estimate*.

---

## 0. Executive frame: the 5× gap is the whole game — and today it silently cancels Prizma's one dynamics win

Two measured facts collide:

1. **Prizma ignites in ~⅓ the steps.** On matched MQAR D=128 (d128L4H4), Prizma-quad2 ignites at
   step 16–20k vs the Transformer's 44–80k (`gpu_bench.json`, P2 headline).
2. **Prizma trains ~5× slower per step** (char-LM arms: 1889 s vs 361 s on A100; `gpu_bench`
   cells ~57 min/run vs ~11 min equivalent).

Multiply them: **in wall-clock, Prizma's "⅓ the steps" ignition advantage is currently fictitious.**
16–20k Prizma steps at 5× step time = 80–100 TF-step-equivalents, i.e. roughly *parity to worse*
than the TF's 44–80k. The repo's honest-limits section discloses the 5× but never draws this
consequence: **the kernel gap does not merely slow Prizma down — it nullifies its one measured
training-dynamics superiority.** Conversely, closing the gap to ≤1.5× converts the same ignition
data into a genuine, quotable headline: *"reaches recall ~2× sooner in wall-clock than the tuned
Transformer."* This is why throughput engineering is not plumbing; it is the precondition for the
architecture's scientific claims to be visible.

Where does the 5× come from? Not from FLOPs — the causal FLOP ledger puts Prizma at 2.14× the
TF's forward FLOPs as-coded (1.78× ideal-banded) at d128L4H4, and the ratio *shrinks* with scale.
The remaining ~2.3× is **implementation, not mathematics**:

- The chunked WY/UT kernel (`seq/delta.py`) is a **Python for-loop over T/C chunks**, each chunk
  issuing ~10–15 small ops (two [C,d_φ,C] matmuls, log-space cumsum, exp of a [C,C] ratio matrix,
  tril masks, a batched `solve_triangular`, two [C,d_φ,d_v] matmuls, a state matmul). At T=1024,
  C=64 that is ~240 kernel launches per layer per forward, all serialized by the cross-chunk
  state carry — dispatch-bound on GPU, where the TF path is essentially one fused SDPA kernel.
- The elementwise chain (log/cumsum/exp/tril) is unfused in eager; the laptop already shows what
  fusion is worth: **1.55× forward / 2.43× backward** from `torch.compile` on *CPU*
  (`benchmark_results.md`). On CUDA the fusion+graph-capture upside is larger.
- The **as-coded window head is a full-T² SDPA** (17.5% of forward FLOPs) although an exact,
  already-tested banded kernel (`banded_window=True`, O(T·w)) sits in the codebase **default-off**.
- The TF comparison arm runs an optimized fused-attention stack; any honest "vs modern practice"
  claim will eventually also need bf16 on both sides (currently fp32 — symmetric, but it means the
  5× is measured against the *easiest* TF, and a bf16 TF would widen it).

The good news from an engineering standpoint: the sequential dependency between chunks carries
only a tiny state (d_v×d_φ per head), and *within* a chunk everything is large batched matmuls.
This is exactly the shape the flash-linear-attention (FLA) / DeltaNet kernel literature has
already industrialized (persistent kernel, chunk loop *inside* one program, WY/UT transform,
fused solve) *(knowledge)*. Prizma's one deviation — the rectangular quad2 state d_v×d_φ with
d_φ∈{128,137,256} — is a tiling problem, not a new algorithm.

An important and slightly uncomfortable status fact found while reading: **the repo's hand-written
Triton kernel cannot currently affect the 5×**, because `_should_use_triton` in
`seq/delta_triton.py` gates on `not grad_live` — every training call (grad enabled) routes to the
eager `chunked_delta`. A backward-capable `TritonChunkedDeltaFunction` *exists in the same file*
(contradicting the module's own "FORWARD-ONLY" header note), but it is unreachable during
training as gated. So the two artifacts that could close the gap — the draft Triton kernel and
`torch.compile` (`delta_fused.py`) — are both **parked behind an unexecuted A100 verification
step**. The single highest-leverage action available to this project is one GPU session that
resolves them, pass or fail.

---

## 1. Measured baseline (facts this brainstorm builds on)

| Fact | Value | Source |
|---|---|---|
| Training step-time gap | ~5× slower than matched TF (char-LM 1889 vs 361 s/arm) | README, report B4 |
| Ignition steps (MQAR D=128, d128L4H4) | Prizma 16–20k vs TF 44–80k | `gpu_bench.json` P2 |
| Decode latency crossover | n=32,768; 2.4× (small) / 2.8× (big) faster @65k; 1.3–1.5× slower below 16k | `gpu_latency.json`, B5 |
| Inference state @1.4M | 17.9 MB constant vs TF KV 562 MB @65k (31×, measured) | `gpu_latency.json` |
| Analytic state @50M/100M | 3.44 / 5.67 MB vs KV 1.25 / 2.06 GB @65k (**372×**, closed-form, never measured) | `scaling_analysis.json` |
| Forward-FLOP ratio | 2.14× as-coded / 1.78× banded-ideal @d128L4H4; window head = 17.5% as-coded | `flop_ledger.py` |
| Compile speedup (laptop CPU) | 1.55× fwd / 2.43× bwd; MPS ≈ eager | `benchmark_results.md` |
| Triton kernel status | DRAFT, pending A100 checklist; grad-gated out of training; backward kernel present but unreachable | `seq/delta_triton.py` |
| Gershgorin capacity bound | N < 1+1/cross(φ): none ≈ 8, lowrank ≈ 12.8, quad2 ≈ 14.2 | `quad2_theoretical_convergence.md` |
| **Bound-vs-measurement gap** | quad2 *measured* to solve D=128 at 130K params — 9× beyond the bound's N≈14 | report B1b |
| Gradient stability | ‖J‖ ≤ α ≤ 1 (contractive); no exploding gradients under BPTT | Theorem 1 |
| Plateau protocol facts | TF transition late+sharp; TF needs gen-warm (lr 1e-3, warmup 2000), Prizma prefers 2e-3; TF warmup-bimodal; engagement floor 0.5; FLOP-matched TF arms failed consistent with LR-transfer (inferred, unconfirmed) | HANDOFF §4, report P2b/P2c |
| Scaling campaign at 50M/100M | param counts + closed-form memory only; **zero training runs** | README §Apparatus |
| Reproducibility posture | resume-cache bug fixed (keyed on seed+config); fabricated CUDA rows deleted; landscape harness built, never run | README, report |

---

## 2. Proposals (ranked)

Each proposal: WHAT / WHY / HOW / FALSIFIABLE TEST / RISKS / IMPACT / FEASIBILITY.

---

### P1. The one-session CUDA kernel decision: verify the draft Triton path end-to-end (including its unreachable backward), else port the FLA-style persistent-kernel design — and then flip the training gate  **[B]** (FLA/DeltaNet kernel design borrowed; rectangular-state tiling is the new part)

**WHAT.** A single pre-registered A100 session that resolves, in order: (a) execute
`docs/superpowers/specs/triton_kernel_a100_checklist.md` against the existing draft kernels
(forward AND backward), (b) if the draft fails its gates, port the design rather than debug it
indefinitely — adopt the flash-linear-attention family's chunked-GDN persistent-kernel schedule
(chunk loop inside one program, fused log-space decay, fused UT solve, bf16 I/O with fp32
accumulators), re-targeted to Prizma's rectangular d_v×d_φ state, (c) either way, extend the
pass gate to *training* shapes and flip `_should_use_triton`'s `grad_live` clause once backward
is verified to <1e-4 fwd+grad, so the kernel actually participates in the 5× number.

**WHY.** Both candidate accelerators (draft Triton, `torch.compile`) are parked behind the same
unexecuted verification; the repo's own culture demands gating before trust, which is right — but
the gating step is currently the entire critical path of the project. The backward kernel already
exists in-file; verifying it is strictly cheaper than writing one. And the 5× gap is the blocker
for every other claim (see §0 — it even cancels the ignition advantage).

**HOW.** One A100 session, three phases, each with a committed pass/fail line:
1. Numerical: draft kernels vs `chunked_delta` on production shapes (C=64; d_φ∈{128,137,256};
   gated + ungated; rectangular d_v≠d_k; repeated keys; ragged tail; S0-carry), gates <1e-3 fwd /
   <1e-2 grad (the module's own stated tolerance) — record per-case, no cherry-picking.
2. Throughput: delta-op tokens/s and **end-to-end model step time** vs the matched TF at
   T∈{512,2048,8192}, bf16-TF and fp32-TF both reported. Pre-registered expectation from the
   compile evidence: ≥2× step-time improvement from kernel/fusion alone.
3. Decision rule (pre-registered): if draft passes gates → flip grad gate, adopt; if it fails on
   tiling (the known d_φ=137 non-power-of-2 concern; note `next_power_of_2`+masking may already
   suffice — the checklist resolves this) → FLA-port attempt in the same session with the same
   gates; if both fail → publish the failure honestly and keep `torch.compile` as the interim
   path with its measured number.

**FALSIFIABLE TEST.** Pre-register: "After P1, Prizma-Seq end-to-end training step time on A100
at d128L4H4, T=2048, bf16, is ≤1.5× the matched TF." If the verified kernel + compile cannot
reach ≤1.5×, the claim "the gap is implementation, not mathematics" is falsified and the honest
position reverts to "Prizma pays a real constant-factor training tax."

**RISKS.** Triangular-solve kernels are where blind implementations die (the repo knows this —
R3); repeated-key and ragged-tail cases are the landmines; register-pressure at BLOCK_DK=256 may
force d_φ=256 onto two passes. Mitigation: the fallback ladder (draft → FLA port → compile-only)
is pre-registered so a failure consumes one session, not a quarter.

**IMPACT: 9/10. FEASIBILITY: 7/10** (owner already has A100/L4 Colab credits; the artifacts and
checklist already exist).

---

### P2. Pre-register THE throughput bar and upgrade the throughput ledger to an end-to-end, same-harness comparator  **[N]** (process; tools borrowed from the repo's own honesty apparatus)

**WHAT.** A pre-registered, tiered throughput bar with a context-dependent structure, measured by
an extended `throughput_benchmark.py` that times **full models (Prizma vs TF) end-to-end in the
same harness**, on the device it names, with absent rows staying absent (the existing invariant):
- **Training:** ≤1.5× TF step time at T≥2k on A100 (bf16); ≤2× accepted as an explicitly-labeled
  intermediate milestone; parity-or-better claimed only where measured at T≥8k.
- **Decode:** retain the measured win at n≥32k and shrink the sub-16k deficit from 1.3–1.5× to
  ≤1.15× (kernel-launch elimination helps decode most, since per-step work is tiny).
- **Memory:** measured `max_memory_allocated` rows for training AND decode at 1.4M, 50M, 100M —
  converting the analytic 372× into a measured number (see P3, deliverable D3).
Protocol fixed in advance: median of N reps with warmup, sync barriers both sides (the fixed
backward-timing method), same shapes/tokens both arms, both fp32 and bf16 reported, TF-with-
compiled-SDPA as the reference arm, machine load disclosed, raw JSON committed.

**WHY.** "Inevitable rival" is a *throughput* claim as much as an accuracy claim — nobody adopts
an architecture that is 5× slower to train, whatever the recall numbers say. A pre-registered bar
converts the current soft narrative ("we're working on speed") into a falsifiable target, and it
is the natural place to enshrine the repo's deleted-fabricated-rows lesson as a standing protocol.

**HOW.** Extend the existing script (it already has the correct honesty skeleton); add the
model-level comparator arms and the memory probe; commit the bar wording in the report before the
A100 session of P1; report pass/fail per tier with no partial credit.

**FALSIFIABLE TEST.** The bar itself. Each tier is either met by a committed measurement row or
recorded as missed — with the current numbers (5×, 1.3–1.5×) as the named baseline to beat.

**RISKS.** Low scientific risk; the main risk is psychological — pre-registering a bar the first
session then misses. That is the point: a missed bar with a measured number is more credible
than an unmeasured hope, and the ladder (5×→2×→1.5×) gives intermediate honest waypoints.

**IMPACT: 9/10. FEASIBILITY: 9/10.**

---

### P3. The 50M scaling-law campaign: a pre-registered loss-vs-compute study with a claim menu, plus measured (not analytic) memory at scale  **[N]** process; **[B]** methods (Chinchilla-style L(C) fits, compute-matched arms)

**WHAT.** The repo's biggest honest hole for "inevitable rival" status: nothing above 1.4M params
has ever been trained. Run a compute-bounded, FLOP-matched, pre-registered campaign:
- **Sizes:** {≈2M, ≈8M, ≈25M, ≈50M} params × {Prizma-quad2(d_φ per Task-1.D lock), TF} × ≥3
  seeds. The 1.4M results anchor the low end.
- **Two matching modes per size:** (a) param-matched (the existing discipline) and (b)
  **compute-matched** (each arm trained to the *same measured FLOPs*, tokens chosen per arm) —
  this is the control whose absence made P2b/P2c "optimization-confounded, no per-FLOP claim."
  A compute-matched arm with per-width LR transfer neutralized (see P4) finally answers "is
  Prizma's win just spent FLOPs?" cleanly.
- **Corpus:** pre-registered *before* running — TinyStories or an OpenWebText subset at char/BPE
  scale, chosen once, with the B4 lesson (no corpus deviations after the fact) written into the
  pre-registration.
- **Fit:** L(C) = a·C^(−b) per architecture over the 4 sizes; report exponents with CIs.

**The claim menu (pre-registered, pick what the data supports, no upgrades after the fact):**
- **A (strong):** "Loss-vs-compute parity in regime: exponent b matches TF within 2 SE at ≤50M."
- **B (medium):** "Loss-vs-param parity at 50M + measured constant-memory advantage at 65k
  context" (the 31× at 1.4M generalizes or it doesn't — measured).
- **C (defensive):** "Divergence documented: b_Prizma/b_TF = X, loss gap grows/shrinks with
  scale" — a real scientific result either way, and vastly more informative than silence.

**Feasibility estimate (analytical, not measured):** 6·N·D-token FLOPs for the largest arm
(50M × 1B tokens ≈ 3×10^17 FLOPs) ≈ a few A100-hours per run at typical MFU; the full 4-size ×
2-arch × 3-seed grid ≈ **~25–80 A100-hours including Prizma's current 5× tax** — comfortably
inside the owner's demonstrated Colab budget, and P1 shrinks the Prizma share further.
*(estimate; verify with one probe run before committing.)*

**Deliverable D3 (cheap, independent):** measured memory rows at 50M/100M — instantiate, decode
65k tokens, record `torch.cuda.max_memory_allocated` for both architectures. Converts the
372× analytic ratio into a measured one without any training. One session, one hour.

**WHY.** Scaling exponents are the language in which architecture rivalry is settled; a
candidate that only ever speaks at 1.4M params is a toy no matter how honest. The existing
372× analytic claim is good *arithmetic* but the repo itself labels it as saying "nothing about
whether either model at this scale would learn anything" — only a measured campaign closes that.

**FALSIFIABLE TEST.** The claim menu is the test: pre-register the exponent-comparison rule and
the BPC/MQL tolerance per size before the first run; any post-hoc reinterpretation is a
deviation that must be labeled as B4 was.

**RISKS.** Char/BPE-corpus choice drives conclusions (mitigate: pre-register, keep text8 as a
secondary leg for continuity with B4); data-order seed sensitivity at small corpus; the 5× tax
makes Prizma arms slow until P1 lands (sequence P3 after P1); at 50M the TF may simply win on
loss — which is *claim C* and still worth publishing, because the memory+latency Pareto can
carry the rivalry argument even with a modest loss deficit (hybrid-slot positioning per the
landscape report 07).

**IMPACT: 10/10. FEASIBILITY: 7/10** (budget-feasible; gated on P1 for wall-clock sanity).

---

### P4. "One protocol, no per-model tuning": μP-style LR-transfer science + measured robustness curves as a headline claim  **[B→N]** (μP borrowed; applying transfer-exponent measurement to delta-state mixers and turning tuner-robustness into a first-class claim is new)

**WHAT.** Two sub-experiments that convert the fair-protocol folklore ("Prizma is robust to both;
TF is warmup-bimodal and LR-fragile") into measured, quotable curves:
- **D1 — Robustness curves:** solve-rate and steps-to-ignite as a function of lr ∈ the
  `lrsweep.py` grid × warmup ∈ {100, 500, 2000, 4000} for both architectures at d128L4H4.
  Pre-register: "Prizma solve-rate ≥0.9 across the full grid where the TF is bimodal below
  warmup=2000" (or whatever the curves show — currently this is HANDOFF folklore, unmeasured).
- **D2 — Transfer exponents:** measure the width-exponent of optimal LR per architecture
  (optimal lr ∝ width^(−a): fit a for TF and Prizma over d∈{64,128,208,256}). If both exhibit
  μP-like transfer (a≈1) with the *same* base protocol, a single (lr, warmup) chosen once at
  d64 transfers to all widths and both architectures — the basis for the headline
  **"one protocol, zero per-model tuning"** vs transformers, whose per-model LR/warmup needs are
  documented in this very repo (gen-warm; P2c's width-failure consistent with LR-transfer).
  If the exponents differ, that is itself a finding: report it and keep per-model sweeps.

**WHY.** The plateau-protocol lessons (per-model plateau early-stop, engagement floor, per-model
LR/warmup) are the repo's hardest-won knowledge, but today they read as *"comparisons need
hand-tuning"* — which reviewers hear as fragility. Inverting it — *Prizma is the architecture
you don't have to tune* — would be a genuinely distinctive, practical claim (tuning cost is real
money at scale) and is uniquely available to Prizma because its in-context plasticity (beta/eta
gates = per-token, per-channel learned write rates) plausibly *is* the robustness mechanism.
That link is also the brain-like story: metaplasticity — fast local plasticity stabilizing slow
system-level learning.

**HOW.** D1 is pure `lrsweep.py` usage (exists, audited); D2 adds a width dimension. Both are
MQAR-scale (hours), no new theory needed. Extend the plateau detector unchanged. Everything
runs before/alongside the P3 campaign and its results *set* P3's single shared protocol.

**FALSIFIABLE TEST.** D1: the pre-registered robustness dominance (or its failure). D2: "a
single (lr, warmup) selected at d64 on both architectures, transferred unchanged, yields final
metrics within +0.02 BPC / −0 solve-rate of each model's own tuned protocol at d208 and in the
P3 campaign." Failing that, the "no per-model tuning" headline is withdrawn — pre-committed.

**RISKS.** μP for delta-state mixers is unestablished in the literature — the transfer exponent
may simply differ from 1; TF bimodality may be intrinsic at small scale (which actually
*strengthens* the asymmetry in Prizma's favor). Watch the confound that killed P2c: here the
LR question *is* the experiment, so it is controlled by design.

**IMPACT: 8/10. FEASIBILITY: 6/10.**

---

### P5. Ignition science: predict the plateau transition from measured key-crosstalk, and reconcile the 9× bound-vs-measurement capacity gap  **[N]** (the theory doc's machinery turned into a dynamics instrument)

**WHAT.** Instrument training (eval-time hooks only, zero protocol change) to log, per eval
batch: (a) the *learned* key-crosstalk cross_t = E|φ(k_i)·φ(k_j)| over live keys (not the
random-key analytic value), (b) effective capacity N_eff(t) = 1 + 1/cross_t (Gershgorin),
(c) write-gate β and decay-α statistics, (d) read alignment (q-at-query vs k-at-storage cosine).
Then test whether the measured ignition step is *predicted* by the crosstalk trajectory — e.g.,
ignition occurs when (D_task − 1)·cross_t crosses below 1 (Theorem 2's residual-rate form).

**WHY.** Two open puzzles in one instrument. First, the repo's own theory says quad2 capacity is
N<14.2, yet quad2 *measured* solving D=128 at 130K params — a 9× gap between the Gershgorin
bound and measurement (the delta rule is an online optimizer that re-writes keys, which the
bound's one-shot assumption ignores; sharpening this is a real theory contribution). Second,
the MQAR transition is late+sharp for *both* architectures and dominated the fair-protocol work;
if Prizma can *predict* its transition from an internal measurable, the project gains: (i) an
a-priori D*(d_φ) recipe replacing post-hoc pre-registrations, (ii) a "why is Prizma 2–4× earlier
to ignite" mechanistic answer (quad2 starts with lower crosstalk → earlier threshold crossing —
a testable immediate hypothesis), and (iii) the first dynamics-level prediction in the
delta-rule family tied to its free-energy identity — squarely the theory slot the landscape
report (07) calls vacant.

**HOW.** Runs at D∈{32,64,128} × d_φ∈{32,137,256}, 3 seeds, MQAR recipe unchanged; correlate
predicted vs measured ignition steps; pre-register the correlation bar (e.g., r>0.8 across the
9 cells) and the *falsification* wording (if ignition tracks outer-loop circuit formation —
visible as β/alignment transitions preceding crosstalk crossings — the honest finding is
"capacity explains where ignition is *possible*, not when it happens," still publishable and
still novel for the family).

**FALSIFIABLE TEST.** As above: the pre-registered prediction rule either predicts ignition
steps out-of-sample or it doesn't; the D*(d_φ) recipe derived from it is then pre-registered for
the *next* task (e.g., D=256) before running it — a true out-of-sample theory test.

**RISKS.** The transition may be optimization-driven (read-projection circuit formation), not
capacity-driven — in which case the correlation fails and the project still gains an honest
negative plus the instrumentation. Cost is modest (MQAR-scale runs, reusable for P4/P3).

**IMPACT: 8/10. FEASIBILITY: 6/10.**

---

### P6. Chunk-size, banded-window, and compile-mode micro-science: the cheap, exact, immediate fraction of the 5×  **[B/N]** (all components exist in-repo; the systematic sweep and its pre-registration are new)

**WHAT.** Three zero-risk-or-exactness-guarded speedups, swept and locked before any new kernel
work:
1. **Turn `banded_window=True` on by default for long-T training.** The exact-equal banded
   kernel (already implemented and guarded, default off) cuts the window head from 17.5% to
   ~0.9% of forward FLOPs. At T=384 the saving is modest; at T≥2k training it is the difference
   between "window head" being a rounding error or a line item.
2. **Chunk-size sweep:** C ∈ {32, 64, 128, 256} × T ∈ {512, 2048, 8192} × d_φ ∈ {32, 128, 256},
   fwd+bwd tokens/s on A100 *and* the laptop MPS. chunk=64 was inherited from DeltaNet defaults
   and never swept in this repo. Exactness is C-independent (chunked==reference <1e-4 at any C),
   so this is pure measurement. Optimal C likely shifts upward with d_φ (bigger matmuls amortize
   the solve) — but measure, don't assume.
3. **Compile modes on CUDA:** `torch.compile(..., mode="reduce-overhead")` (CUDA-graph capture of
   the whole chunk loop) vs default compile vs eager, on top of whichever P1 outcome holds. The
   CPU compile evidence (1.55×/2.43×) says the eager dispatch overhead is large; CUDA graphs
   attack exactly that.

**WHY.** These are the fraction of the 5× that requires no new kernels, no numerics risk, and
roughly one benchmark session — and their results are *inputs* to P1's kernel design (the
persistent kernel should be tuned at the measured optimal C, not the inherited 64).

**FALSIFIABLE TEST.** Pre-register: "The three micro-optimizations jointly deliver ≥1.5×
end-to-end step-time improvement on A100 at d128L4H4/T=2048." (If they deliver less, that
quantifies how much of the gap genuinely requires the P1 kernel — also useful.)

**RISKS.** Near zero scientifically; the only watch-item is that compiled MPS fell back to eager
before — keep the "Compiled row ≈ Eager row means fallback" disclosure discipline.

**IMPACT: 6/10. FEASIBILITY: 9/10.**

---

### P7. Multi-timescale state decay (α_t): brain-like temporal hierarchy as an optimization stabilizer  **[B→N]** (multi-timescale SSM/decay initializations borrowed; the "decay spectrum as homeostat that reduces outer-loop tuning" framing and its measurement are new)

**WHAT.** Two variants, both behind off-path byte-identity levers per repo discipline:
1. **K-timescale heads (free):** partition or duplicate heads with log-spaced α floors
   {0.5, 0.9, 0.99, 0.999}, giving each head a designated temporal receptive field (scalar
   per-head decay — zero kernel cost, state grows ×K only if duplicating; disclose either way).
2. **Per-channel α_t ∈ R^{d_φ}** (Mamba-1-style): the chunked machinery for per-channel decay
   *already exists in-repo* — the eta lever's batched per-channel triangular solves
   ([B,H,d_v,C,C] systems) are the identical algebra with α in place of η. Higher kernel cost,
   pre-registered separately.

**WHY.** Brain grounding: cortical processing is organized as a temporal hierarchy with
systematically longer timescales in higher areas, and hippocampal traces are explicitly
multi-timescale *(knowledge)* — a delta-state model whose decay spectrum *is* a measured,
structured object is more brain-like in a falsifiable way than any vibe. Optimization grounding
(the lens of this report): long-α state channels act as internal momentum/integrators, plausibly
reducing outer-loop LR/warmup sensitivity — connecting directly to the P4 "one protocol"
headline. If Prizma's learned decay spectrum clusters at task-relevant timescales (n-gram vs
document), that is a measurable brain-alignment result with no behavioral disadvantage required.

**FALSIFIABLE TEST.** Pre-register all three: (a) char-LM BPC improves ≥0.01 at matched
params+FLOPs (K× state cost disclosed); (b) the warmup-robustness curve of P4-D1 shifts (less
warmup needed) with long-α channels; (c) the learned α-spectrum, measured as a histogram,
correlates with task temporal statistics (computed on text8: transition-length statistics).
Any two of three failing = the lever is honest dead weight, reported as such.

**RISKS.** K× or per-channel decay multiplies state size — directly diluting the 372× memory
headline (at K=4 it becomes ~93×; must be disclosed, never hidden); per-channel α is the
expensive kernel path; brain-likeness without measurement (c) is vibes. The repo already has
the discipline to scope this correctly.

**IMPACT: 7/10. FEASIBILITY: 5/10** (variant 1 is 8/10; variant 2 inherits eta-path machinery
but at real kernel cost).

---

### P8. Adjudicate the surprise lever *before* any kernel spend on it  **[N]** (process discipline; the ablation apparatus exists)

**WHAT.** The surprise-gated write currently forces the **exact sequential scan** (chunk-parallel
form provably unavailable; documented ~100% divergence on repeated keys for the frozen-chunk
approximation). A kernel effort to make surprise fast would therefore be a serious investment —
and the only evidence collected so far *points against the lever* (`gpu_ablation.json`, n=2
smoke: constant gate 0.566 and random gate 0.527 both above the real signal 0.517; every verdict
INCONCLUSIVE). Proposal: run the owed **powered** surprise ablation (≥10 seeds, the R9 controls
already coded) *first*; only if the real signal beats both controls with CI does any surprise-
fast-path work (two-pass fixed-point iteration with a disclosed approximation bar) enter the
roadmap — and even then as an *approximate* path gated against the exact scan, never silently.

**WHY.** This is a scaling-and-throughput decision, not just a modeling one: the surprise path
is the one lever whose semantics *defeat* chunk-parallelism, i.e., the one step backward from
the kernel strategy. Spending kernel effort on an unvalidated lever inverts the repo's own
pre-registration discipline. Cheap to resolve; expensive to skip.

**FALSIFIABLE TEST.** The powered ablation itself: real signal > random control and > constant
control, 95% CI, pre-registered before running.

**RISKS.** Minimal — the main risk is the answer being "no," which is exactly what the process
is for. (Lever G's per-channel eta, by contrast, is already chunk-parallel and exact — if a
novel-core lever earns kernel investment, it is that one.)

**IMPACT: 5/10** (7 if the lever survives and unlocks). **FEASIBILITY: 8/10.**

---

## 3. What NOT to do (guardrails inherited and extended)

1. **No throughput row without a device.** The deleted "CUDA (Simulated)" rows stay deleted; the
   P2 ledger keeps absent-rows-absent as an invariant, now covering model-level arms too.
2. **No per-FLOP claim until a clean compute-matched arm exists** (P3-b) — the P2b/P2c confound
   is documented; do not quietly promote param-matched wins into efficiency language.
3. **No kernel adoption without the exactness gate** (<1e-4 fwd+grad vs `chunked_delta`,
   including rectangular state, gated alpha, repeated keys, ragged tails) and no flipping of the
   `grad_live` gate until the *backward* kernel passes it — the current gate is correct behavior,
   not an oversight to remove early.
4. **No corpus or protocol deviations inside the scaling campaign** — B4 is the standing
   cautionary tale; pre-register corpus, seeds, claim menu, and the plateau/LR protocol before
   the first run, and label any deviation in the verdict itself.
5. **Do not report the ignition-step advantage without its wall-clock conversion** — by the same
   honesty that discloses the 5×, the "⅓ the steps" headline must always carry its step-time
   denominator until P1 changes the number.
6. **Keep `step()==forward() <1e-6` green** through every kernel and decay-variant change; the
   O(1) claim is load-bearing for B5 and for the memory scaling story.

---

## 4. QUESTIONS FOR THE OWNER

1. **Compute budget for the scaling campaign (P3):** the 4-size × 2-arch × 3-seed grid is
   estimated at ~25–80 A100-hours (plus reruns if a protocol bug appears). Is a one-week Colab
   budget acceptable, and may P3 pre-commit the corpus (TinyStories vs OpenWebText subset vs
   enwik8/text8 continuation) before any run?
2. **Kernel strategy preference (P1):** if the draft Triton kernel fails its A100 gates, is
   porting/adapting the existing flash-linear-attention open-source kernel family acceptable to
   the repo's borrowed-vs-new ledger (with attribution and the same exactness gates), or must
   the in-repo kernel be made to pass first?
3. **The throughput bar (P2):** is "≤1.5× TF training step time at T≥2k on A100 (bf16)" the
   bar the owner wants to pre-register, or should the milestone ladder (5×→2×→1.5×) be
   committed as the official sequence with dates/budgets attached?
4. **bf16 symmetry:** today both arms train fp32. For "vs modern practice" claims, do we adopt
   bf16-with-fp32-master for both arms (the standard) and restate the 5× in both precisions
   permanently, or keep fp32 as the canonical comparison regime?
5. **Headline risk appetite (P4):** "one protocol, zero per-model tuning" is a strong claim that
   requires μP-style transfer to *work* for delta mixers (unestablished). If D2 shows differing
   transfer exponents, is the owner comfortable publishing the negative and falling back to
   "measured robustness curves" as the claim?
6. **Multi-timescale memory dilution (P7):** variant 1 with duplicated heads multiplies state by
   K (372× → ~93× at K=4). Is trading measured memory advantage for BPC/robustness gains
   acceptable, or is the memory headline inviolate?
7. **Surprise lever (P8):** if the powered ablation confirms the smoke signal (real signal *worse*
   than random/constant controls), should the surprise lever be formally retired from the
   architecture (kept as an ablation footnote), freeing the roadmap from its sequential-scan tax?
