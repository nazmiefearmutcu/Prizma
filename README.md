# Prizma

[![CI](https://github.com/nazmiefearmutcu/Prizma/actions/workflows/ci.yml/badge.svg)](https://github.com/nazmiefearmutcu/Prizma/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Last commit](https://img.shields.io/github/last-commit/nazmiefearmutcu/Prizma)](https://github.com/nazmiefearmutcu/Prizma/commits)
[![Stars](https://img.shields.io/github/stars/nazmiefearmutcu/Prizma?style=social)](https://github.com/nazmiefearmutcu/Prizma/stargazers)

Two small-scale research threads built the same way: a **pre-registered falsifiable bar**, a
**parameter/FLOP-matched** baseline, an **adversarial referee audit**, and **honest, binding
limits**. No faked metrics — every number is produced by a reproducible script and the raw result
JSONs are committed under `results/`.

| Thread | Question | Verdict (in the tested regime) |
|---|---|---|
| **Prizma-Seq** | Can a parameter-free quadratic delta-state sequence mixer stand in for attention at small scale? | **Candidate** — clears the §4 diagnostic bar param-matched vs a tuned Transformer; constant-memory + long-context O(1)-latency edge; honest losses disclosed. |
| **Prizma** | Can a backprop-free, fully-local learner do task-boundary-free continual learning? | **Zero forgetting** in the input-distinguishable regime, beating backprop & EWC — no replay, no boundaries, no weight transport. |

---

## Prizma-Seq — efficient-attention-replacement candidate

Prizma-Seq is a Gated-DeltaNet-family sequence mixer whose novel lever is a **parameter-free
quadratic feature map (`quad2`)** that makes the per-head carried associative state *rectangular*
(`d_h × d_φ`, with the monomials as fixed seeded buffers → 0 added parameters). At small scale,
**parameter-matched** against a tuned decoder-only Transformer (RMSNorm + SwiGLU + RoPE), it clears
the project's pre-registered §4 bar:

| Leg | Verdict | Headline |
|---|---|---|
| MQAR (D=128) | **PASS** | parity @860K params; solves @130K where the matched TF needs ≥461K → ≥3.5× param-efficiency (coarse grid) |
| Induction | **PASS** | quad2 0.9995 (3/3) vs TF 0.996 |
| Selective-copy | **PASS** | selective 0.9991; a fixed-position control isolates content-selectivity |
| Char-LM (text8) | **PARTIAL — deviated from pre-registration** | Prizma 1.7496 vs TF 1.7254 BPC clears the +0.05 margin, but the pre-registered bar demanded **both** corpora with ≥3 seeds (≥5 for the closest leg). Delivered: **text8 only, n=2** — and the other corpus, tiny-shakespeare, **FAILED** (−0.09) with its raw data not retained. See [below](#b4-char-lm-a-partial-not-a-pass). |
| Inference | **PASS (memory)** | constant 17.9 MB state ∀n (28–455× less); measured **O(1)-latency crossover at n≥32k** (2.4–2.8× faster @65k) |
| Causal ablation | **PASS** | quad2 ≫ rand_linear ≈ none ≫ TF — the gain is the quadratic monomials, not "a bigger RNN" |
| Length-extrapolation | **WIN (relative)** | 10× better retention than a RoPE Transformer at 8× train length (absolute accuracy still only ~0.40) |

**Honest scope — a *candidate*, not a proven alternative.** Char-LM is a loss-within-margin; the
latency win is long-context-only (Prizma is ~1.3–1.5× *slower* below n≈16k) and Prizma trains ~5×
slower per step (sequential delta); the FLOP-matched TF arms were optimization-confounded so **no
per-FLOP claim** is made; n=2–3 seeds are descriptive (not powered equivalence); **large-scale LM
parity and backprop-free parity are NOT claimed** (open frontiers).

### Corrections and quarantined results (read this before citing anything)

<a name="b4-char-lm-a-partial-not-a-pass"></a>

**B4 char-LM is a PARTIAL, not a PASS.** Spec §B4 pre-registered: *test BPC ≤ T+0.05 on **BOTH**
corpora, ≥3 seeds (≥5 for the closest leg)*. What was delivered is **one** corpus (text8) at **n=2**,
after the other corpus (tiny-shakespeare) had **failed** by −0.09 under an overfitting recipe — and
that failing run's raw data was *not retained on disk*, so it cannot be re-examined. A later text8
run reached n=7 vs TF n=10 (Δ = +0.021, still within the margin, and the gap is statistically real:
Welch t≈6.7). Calling this leg PASS took credit for a bar that was not met. The result is real and
the margin is genuinely cleared **on the corpus that was run** — it is the *pre-registration* that
was not honoured, and the deviation now sits in the verdict rather than in the fine print.

**The powered n=10 recall gate is QUARANTINED — do not cite it.**
`results/campaign_2026-06-08/recall_gate.json` is not the n=10 run it describes. A resume-cache bug
in `seq/recall_gate.py` skipped seeds it had already seen **keyed on the seed number alone, with no
record of the configuration**, so an earlier tiny `--smoke` run supplied seeds 0–1 in **all ten
cells** — at a 4× smaller scale, and for the candidate arm with its key lever (`quad2`) switched
**off**. It also supplied the stage-1 LR sweep in all ten cells, meaning every arm's learning rate
was chosen on the wrong model over the wrong grid. Consequently the reported `params`, the standard
deviations, the CIs, and the published reading that *"the Transformer baseline is high-variance… on
induction TF is bimodal: ~half its seeds collapse to ~0.06"* are all untrustworthy — that last one is
a caching bug partly misdiagnosed as a property of the baseline (among the 8 uncontaminated seeds, 3
collapse, not 5 of 10). **The error direction was conservative**: it widened the CIs and handicapped
the candidate, so it produced an under-claim, not an over-claim. The artifact has **not** been re-run
and **no** replacement numbers have been invented — a clean campaign needs ~28 A100-hours. Full
disclosure: [`results/campaign_2026-06-08/CONTAMINATION.md`](results/campaign_2026-06-08/CONTAMINATION.md).
The bug is fixed (resume is now keyed on `(seed, config-fingerprint)`) with regression tests, and
the operational door is closed too: smoke and campaign runs now write **separate result files**
(`recall_gate_smoke.json` vs `recall_gate.json`; a smoke run aimed at the campaign ledger is
refused), every run archives its raw records before any verdict ([`docs/RETENTION.md`](docs/RETENTION.md)),
and claim-grade results are governed by the pre-registration registry ([`docs/preregistry/POLICY.md`](docs/preregistry/POLICY.md)).

**Baselines that were built and never run.** `seq/gla.py` (GLA), `seq/mamba2.py` (Mamba-2) and the
4-arm head-to-head harness `seq/landscape.py` are faithful, fully-tested implementations that have
produced **zero results**. No GLA or Mamba-2 number appears anywhere in this repo, and none is
claimed. The harness exists and has not been run at scale.

- Full writeup + adversarial referee trail → **[`docs/PRIZMA_SEQ_REPORT.md`](docs/PRIZMA_SEQ_REPORT.md)**
- Raw A100 results (auditable) → `results/gpu_{bench,diag,lengen,latency,charlm2}.json` + `results/v3_campaign_results.md`
- Code → `seq/` (mixer, tasks, transformer baseline), `gpu_*.py` (GPU runners), `PRIZMA_run_*.ipynb` (Colab bootstrap)

```bash
# local kernel self-tests / smoke (CPU/MPS), then the GPU runners on an A100:
PRIZMA_RESULTS=results python gpu_diag.py induction selcopy   # B2/B3
PRIZMA_RESULTS=results python gpu_charlm2.py --skip_none      # B4 (text8)
PRIZMA_RESULTS=results python gpu_latency.py                  # B5 latency/memory
PRIZMA_RESULTS=results python gpu_lengen.py                   # length-extrapolation
```

### Apparatus built beyond the tested regime — what has and has not been run

> **Read this heading literally.** The three subsections below are **apparatus and arithmetic, not
> results.** No model above ~1.4M parameters has been trained. No downstream benchmark has been run.
> The only measured numbers here are the throughput timings in §3, and they are laptop timings.

#### 1. Parameter-count scaling at 50M & 100M — **NO TRAINING RUN**
These models were **instantiated on CPU purely to count parameters** and to evaluate a closed-form
memory formula. They were never trained, never evaluated, and produced no accuracy or loss number of
any kind. What follows is a parameter count and an analytical memory identity:
* **50M Scale:** Prizma-Seq (**47,964,832** params; $d_{model}=512, L=10, H=8, d_{phi}=320, \text{window}=16$) vs. Transformer (**48,015,872** params; $d_{ff}=1376, \text{rope}=\text{True}$).
* **100M Scale:** Prizma-Seq (**102,602,760** params; $d_{model}=768, L=11, H=12, d_{phi}=320, \text{window}=16$) vs. Transformer (**102,653,184** params; $d_{ff}=2056, \text{rope}=\text{True}$).

**Analytical (closed-form, NOT measured) memory comparison ($B=1$, FP16):**
The memory ratio of Transformer KV-cache to Prizma-Seq state size ($\frac{2 \cdot T}{d_{phi} + 2 \cdot W}$) is independent of depth and width. This is an algebraic property of the two state definitions, evaluated on paper — it is not an allocation measurement, and it says nothing about whether either model at this scale would learn anything:
| Sequence Length ($T$) | Transformer KV-Cache (50M) | Transformer KV-Cache (100M) | Prizma State (50M / 100M) | Ratio |
|---|---|---|---|---|
| **1,024 Tokens** | 20.00 MB | 33.00 MB | **3.44 MB / 5.67 MB** | **5.82x** |
| **8,192 Tokens** | 160.00 MB | 264.00 MB | **3.44 MB / 5.67 MB** | **46.55x** |
| **65,536 Tokens** | 1.25 GB | 2.06 GB | **3.44 MB / 5.67 MB** | **372.36x** |

Details: [`seq/scaling_analysis.py`](seq/scaling_analysis.py) | JSON: [`results/scaling_analysis.json`](results/scaling_analysis.json) | Report: [`results/scaling_analysis.md`](results/scaling_analysis.md)

#### 2. Downstream evaluation (MMLU & GSM8k) — **PLANNED, NOT RUN**
[`seq/downstream.py`](seq/downstream.py) is a harness with **no committed results**. Nothing in this
repo has been pre-trained on OpenWebText/The Pile, and neither MMLU nor GSM8k has ever been scored.
The code exists; the evaluation does not:
* **Pre-Training:** A `StreamingTextDataset` for token packing standard corpus streams (OpenWebText/The Pile) and a PyTorch pre-training loop.
* **Downstream Tasks:** Few-shot/zero-shot evaluations for MMLU multiple-choice question-answering (via token log-probabilities) and GSM8k math word problems (via autoregressive causal/recurrent decoding).

#### 3. Training throughput (Eager vs. Compiled) — measured on a laptop
Wall-clock timings of the Prizma-Seq delta update, forward and backward. **Every row is measured on
the device it names.** There are no CUDA or Triton rows because this benchmark has not been run on a
CUDA box, and nothing is extrapolated to stand in for one.

| Execution Path | Pass | Device | Time (ms) | Throughput (tokens/s) | Speedup vs Eager CPU |
|:---|:---|:---|:---|:---|:---|
| **Eager** | Forward | CPU | 41.18 | 198,934.1 | 1.00x |
| **Eager** | Backward | CPU | 104.85 | 78,132.7 | 1.00x |
| **Compiled** | Forward | CPU | 26.57 | 308,341.2 | 1.55x |
| **Compiled** | Backward | CPU | 43.17 | 189,757.1 | 2.43x |
| **Eager** | Forward | MPS | 42.72 | 191,738.2 | 0.96x |
| **Eager** | Backward | MPS | 88.60 | 92,455.3 | 1.18x |
| **Compiled** | Forward | MPS | 42.00 | 195,061.5 | 0.98x |
| **Compiled** | Backward | MPS | 84.71 | 96,705.4 | 1.24x |

_B=8, H=4, T=1024, d=64, chunk=64 (8,192 tokens/pass); 5 warmup + 20 timed runs, mean; Apple M4,
torch 2.12. Forward and backward are timed in **separate** loops with synchronisation barriers, not
by subtracting one from a combined loop. Absolute times move by up to ~2x with machine load — read
the ordering and the ratios, not the milliseconds. On MPS the "Compiled" path falls back to eager
(see [`seq/benchmark_results.md`](seq/benchmark_results.md))._

> **Correction (superseded numbers).** An earlier version of this table carried six
> `CUDA (Simulated)` rows computed as `t_cpu / {120, 180, 210}` from hardcoded constants and printed
> to one decimal place next to the measured rows — they were arithmetic on a CPU timing, not
> measurements of any GPU, and they have been deleted. It also carried a row reading
> `Compiled | Backward | CPU | 0.6660 ms | 235.67x | Measured` — a backward pass ~70x *faster* than
> its own forward, which is not physically possible. That was a timing-method artifact
> (`t_bwd = t_combined - t_fwd` collapsing); the method has been fixed and the whole table re-measured.

Details: [`seq/throughput_benchmark.py`](seq/throughput_benchmark.py) | Report: [`seq/benchmark_results.md`](seq/benchmark_results.md)

#### 4. Gradient Stability & Theoretical Convergence
A mathematical analysis in [`docs/quad2_theoretical_convergence.md`](docs/quad2_theoretical_convergence.md) proves:
* **Jacobian Contractiveness:** The recurrent delta state transition Jacobian spectral norm is strictly bounded by the decay gate: $\| J_t \|_2 \le \alpha_t \le 1.0$, preventing exploding gradients during BPTT.
* **Gershgorin Capacity Bounds:** Using the Gershgorin Circle Theorem, we derive the capacity limit $N < 1 + 1/\text{cross}(\phi)$: with the repo's **measured** crosstalk for quadratic keys (`quad2` **0.117**, 1.54× the previously quoted `~0.076`), the bound caps $N^* \approx 9.5$ compared to $N < 8$ for linear keys (`none`). Per the addendum in [`docs/quad2_theoretical_convergence.md`](docs/quad2_theoretical_convergence.md), this bound does **not** by itself explain the repo's own MQAR $D=128$ PASS — explaining it is the open job of the capacity-law program (PR-02).

---


## Prizma — backprop-free, fully-local continual learning

A **backprop-free**, fully-local,
predictive-coding learning architecture targeting neuromorphic/analog hardware.

Prizma demonstrates **task-boundary-free, task-label-free continual learning**: in an
input-distinguishable (domain-incremental) stream it reaches **zero forgetting** while beating
naive backprop and (boundary-using) EWC — using only local learning rules (no backprop, no weight
transport; works with random-feedback DFA). Its limits are characterized honestly: it provides no
benefit in the fully-ambiguous regime (proven impossible for any single-head learner) and degrades
gracefully as domains overlap.

### Headline result (E1, structured-permuted, 10 seeds, ±95% CI)
| Learner | ACC | FGT (forgetting↓) | boundaries? | buffer? | W^T? |
|---|---|---|---|---|---|
| backprop MLP | 0.445 | 0.553 | — | — | — |
| EWC | 0.456 | 0.411 | **yes** | — | — |
| replay (buffer 1000) | 0.737 | 0.156 | **yes** | **yes** | — |
| oracle_multihead *(upper bound)* | 0.879 | 0.000 | **task-id given** | — | — |
| **Prizma (DFA, no W^T)** | **0.834** | **0.000** | **none** | **none** | **none** |
| Prizma (exact W^T) | 0.708 | 0.000 | none | none | yes |
| PRIZMA_noRoute *(ablation)* | 0.446 | 0.489 | — | — | — |

Prizma sits **between replay and the task-id-oracle**, matching the oracle's zero forgetting
*without being told the task id*, no replay, no boundaries, **no weight transport** (the W^T-free
DFA variant is the best). The ablation shows recognition-routing is the causal mechanism.
Adversarially audited by a 4-referee panel (no leakage, fair, reproduces, honest).

- **Full writeup — the most self-critical document in this repo — → [`docs/Prizma.md`](docs/Prizma.md)** (equations,
  borrowed-vs-new ledger, neuromorphic mapping, iteration log of what failed, limits). It is worth reading for
  §8 alone, where the author explains why the headline number above is nearly free: *"FGT=0 is an architectural
  quasi-tautology; the real achievement is the routing."* Written in Turkish; now translated in full, with the
  original preserved verbatim at [`docs/Prizma.tr.md`](docs/Prizma.tr.md).
- Code → `src/` (prizma + baselines + data + metrics), `experiments/` (E1–E5 suite + figure)

```bash
python3.13 -m venv .venv && ./.venv/bin/pip install numpy matplotlib
./.venv/bin/python experiments/run_continual.py   # ~2.5 min → results/results.json
./.venv/bin/python experiments/make_figure.py      # → results/figure.png
```

> Status: research prototypes. Neither thread claims large-scale parity; each is a falsifiability
> gate passed (or honestly refused) in a precisely-characterized small-scale regime.

---

## Registered pre-registration outcomes (2026-09 campaign)

All runs executed under the two-lane pre-registration registry
([docs/preregistry/INDEX.md](docs/preregistry/INDEX.md) — exact frozen protocols, fresh
seeds, pre-committed failure branches). Raw per-seed artifacts committed under `results/`.

| id | Question | Verdict |
|---|---|---|
| PR-03 | Is Prizma competitive with boundary-free regularizers in **single-pass** domain-incremental CL? | **CLAIMED (PASS)** — beats the best boundary-free arm by +0.070 ACC (Welch 95% CI [+0.024, +0.117]); FGT 0.192 vs 0.454. |
| PR-05 | Is the bounded-M_max=K expert economy (recruit-by-eviction) loss-free at home? | **CLAIMED (PASS)** — ΔACC = ΔFGT = +0.000000 on all 10 fresh seeds; 0 evictions; the cap is inert in the home regime. |
| PR-07′ | Does the O(1) state provide block-drift continual adaptation (char-LM, text8→shakespeare, no labels/boundaries/replay)? | **CLAIMED (PASS)** — FGT_A = −0.189 (B-training *improved* A); adaptation −3.18 BPC vs frozen control (CI [−3.420, −2.936]). Clears the flagship's re-scoped prerequisite at toy scale. |
| PR-04 | Is the delta write analog-robust (low-precision state + write noise)? | **NEGATIVE** — at the matched-clean operating point 4-bit delta retention is significantly *worse* (Δ = −0.0923, CI [−0.170, −0.014]); the correction needs a precise S·k read. Retired as a claim. |
| PR-06 (+ successors) | Can vigilance routing handle *interleaved* single-pass streams? | **NEGATIVE (terminal)** — interleaved-single-pass is information-bounded at the partition level: experts need block-coherent floor calibration (≳10² contiguous samples). In the block regime the same machinery becomes domain-coherent (expert corpus purity 0.997–1.0). |
| PR-08 | Does the fused column deliver lifelong behavior at many-block scale (3 corpora, returning domain): retention + adaptation + a clean routing ledger? | **NEGATIVE on the routing ledger** — B1 PASS (retention-A −0.273: negative forgetting at 3-block scale), B2 PASS (adaptation +0.298 vs frozen), and B4 measures the tissue bounding trunk drift (PRIM +1.20 vs shared-head +2.78 BPC degradation); but B3a FAIL: only 42.8% of early-C segments re-route to the A-expert (bar 0.5) → the 'lifelong routing' wording is retired per the pre-committed branch. First powered execution via the registered CPU fallback. |
| PR-09 | Is the PR-08 boundary-routing scatter caused by trunk-training re-calibrating the tissue's precision floors? | **NEGATIVE (mechanism narrowed)** — pinning floors to their block-A calibration left routing near-identical to the control (frac mean 0.405 vs 0.428, per-seed ~unchanged): the scatter is the trunk's shared-weight drift, not floor drift. Guards held (retention −0.270, adaptation +0.299); canary bit-identical to PR-08. Successor registered: trunk-lr scheduling (PR-10). |
| PR-10 | Is the boundary-routing scatter monotone in trunk plasticity — does scaling the backbone lr by 0.25 on the returning block repair the routing ledger without losing retention/adaptation? | **CLAIMED (PASS)** — frac mean 0.575 vs 0.428 control (4/5 seeds strongly improved), retention −0.282 and adaptation +0.308 intact (Holm p=0.021); bonus: trunk drift +1.20 → +0.72, the best B4 of any arm measured. The lifelong-routing repair is registered: C-block trunk-lr scaling enters PR-LM-1 designs. |
| PR-11 | Does the PR-10 repair transfer to the full 5-arm flagship and clear ALL gated bars? | **FAIL on B3 clause (b) — but clause (a) REPAIRED**: frac 0.575 ≥ 0.5 with B1 −0.282 and B2 +0.308 intact and B4 +0.72 (best arm); however killing pattern completion IMPROVES B-eval by 0.1995 (C segments leaking into B-experts via argmin contaminate them; the forced arm protects them). Both FROZEN canaries bit-identical to PR-08. Successor lead: domain-exclusive experts (own prereg). |
| PR-12 | Is the B-eval damage expert-head contamination (C leaking into B-experts), and does domain-exclusive C-routing close it? | **CLAIMED (PASS)** — with C segments barred from protected B-experts, PRIM's B-eval lands exactly at the forced-arm level (B4 +0.5213 vs +0.5214) while boundary routing RISES to 0.976 and retention/adaptation hold. The pattern-completion value gap is closed. Next: the full 5-arm flagship with exclusion (S2c). |
| PR-13 | Do BOTH registered repair levers compose? Does the repaired flagship clear ALL THREE gated bars at n=5? | **CLAIMED (PASS) — the flagship stands.** B1 −0.267, B2 +0.299 (Holm p=0.023), B3 frac 0.976 with forced_cost +0.0000 (pattern completion costs nothing); B4 +0.52 (best arm); both FROZEN canaries bit-identical to PR-08. **The 'lifelong routing' wording is reinstated with evidence** — the fused column delivers retention (negative forgetting at 3-block scale), adaptation, and a clean lifelong-routing ledger. |
| PR-14 | Does the repaired column ACCUMULATE across a 5-block stream with a literal returning-domain revisit? | **CLAIMED — ALL FIVE BARS** — revisit recovery +0.82 bpc (8x the bar), re-engagement frac 0.971, A-retention −0.32 through 5 blocks, exclusion costs nothing (harm −0.008), text8 accumulates. The lifelong claim now rests on 5 blocks with an honest re-encounter ledger. |
| PR-15 | Can a memory-matched sliding-window transformer (the standing control) hold domain A across the same 5-block curriculum? | **NEGATIVE for the claim as registered — and informative**: the control holds A too (both models show negative A-retention), because the curriculum re-exposes text8 at C and E — the bar was not discriminating. The real gap is at the boundaries: the control's B-drift damage is ~+1.15 bpc vs the repaired column's ~+0.60. Recorded honestly; the boundary-damage head-to-head becomes the next registration. |
| PR-16 | Does the repaired column's boundary-drift damage beat the memory-matched control's by a CI-established margin? | **CLAIMED** — delta +0.3685 bpc (control ~+1.01 vs column ~+0.64), Welch CI [0.27, 0.47]: the tissue HALVES B-boundary damage. The head-to-head comparison ledger's first established gap. |
| PR-17 | WHICH component carries the boundary-damage advantage — the lr schedule or the tissue? | **ATTRIBUTION: SCHEDULE-CARRIED** — a plain transformer that merely slows its post-A blocks matches the repaired column's damage bit-for-bit (C2 ~0) while the schedule alone establishes the C1 gap (+0.37, CI [0.27, 0.47]). Correction to PR-16's interpretation: the schedule protects, not the tissue; the tissue's established uniqueness remains the routing ledger. |
| PR-18 | Does the repaired flagship's all-bars verdict REPLICATE on fresh seeds? | **CLAIMED — REPLICATED at n=10 fresh (seeds 5-14)**: B1 −0.276, B2 +0.683 (CI [0.43, 0.93], Holm p=0.0003), B3 frac 0.965 with zero forced cost. The seeds-5-9 leg alone was B2-INCONCLUSIVE (floor variance); the pre-authorized seed extension resolved it. Cumulative evidence: 15 seeds, two independent samples, every bar passed in both. |
| PR-19 | Does the 5-block accumulation claim REPLICATE on fresh seeds? | **CLAIMED — REPLICATED at n=10 cumulative fresh** — every bar reproduces to 3-4 decimals on seeds 5-9 (recovery +0.81, frac 0.97, A-retention −0.32, harm −0.008). The lifelong-accumulation evidence now spans two independent seed samples. |
| PR-20 | Does the boundary-damage gap replicate on fresh seeds at the halving margin? | **NOT-ESTABLISHED (FINAL, pooled fresh n=10)** — the gap is directional in every fresh pair (+0.245 mean) but the 0.25 halving-margin is not CI-established: the fresh gap (~0.25) is ~half the original (~0.55) — the protection's size is seed-dependent. The honest ledger: accumulation parity, schedule-carried protection, tissue-only routing. |
| PR-21 | WHO owns the revisit recovery — the schedule, the tissue, or is it universal? | **CLAIMED — TISSUE-COSTS-RECOVERY (attribution matrix CLOSED)**: the plain control recovers MORE (+1.15 vs the column's +0.82, CI-established) while the scheduled control matches the column — recovery magnitude is a PLASTICITY trade (volatile learners swing hardest both ways: most damage AND most recovery). The column is the stable learner. |

Pending GPU-tier execution/confirmation: PR-01 (powered surprise-gating ablation, frozen
protocol), PR-02 (crosstalk capacity-law D-frontier), Tier-0 repairs (clean recall gate, B4
closure, first GLA/Mamba-2 landscape), kernel decision (≤1.5× TF step time), and the A100
confirmation of PR-LM-1 — the flagship continual-LM bar is REGISTERED (2026-09-08) and already
executed/claimed on CPU (PR-13, replicated at n=10 fresh by PR-18); only the GPU-tier
confirmation is pending.

---

## Reproducing the falsifiability harness

The invariant test suite (kernel guards, lever `off == identical` checks, O(1) `step == forward`
equivalence, and the anti-conservative statistics gate) runs on every push via
[CI](.github/workflows/ci.yml):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt pytest
pytest -q          # CPU-only (what CI runs): 563 collected -> 553 passed, 10 skipped
                   # (measured 2026-09-11; the 10 skips = 9 static CUDA skips + 1 scipy skip).
                   # Several kernel-equivalence tests are parametrised over devices, so a box
                   # with MPS available collects a different count — re-measure there.
```

_The "94 tests" figure this README used to quote was long out of date — CI was already collecting
over 200. The badge above was also green over a red run: CI had been failing since 2026-06-20 on a
missing `shakespeare.txt` fixture. That is fixed, and the counts above are from the passing run._

## License

Prizma is released under the [Apache License 2.0](LICENSE). © 2026 The Prizma Authors.
