# Prizma

[![CI](https://github.com/nazmiefearmutcu/Prizma/actions/workflows/ci.yml/badge.svg)](https://github.com/nazmiefearmutcu/Prizma/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

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
| Char-LM (text8) | **PASS** | Prizma 1.7496 vs TF 1.7254 BPC — within the pre-registered +0.05 bar (**does not beat** TF) |
| Inference | **PASS (memory)** | constant 17.9 MB state ∀n (28–455× less); measured **O(1)-latency crossover at n≥32k** (2.4–2.8× faster @65k) |
| Causal ablation | **PASS** | quad2 ≫ rand_linear ≈ none ≫ TF — the gain is the quadratic monomials, not "a bigger RNN" |
| Length-extrapolation | **WIN (relative)** | 10× better retention than a RoPE Transformer at 8× train length (absolute accuracy still only ~0.40) |

**Honest scope — a *candidate*, not a proven alternative.** Char-LM is a loss-within-margin; the
latency win is long-context-only (Prizma is ~1.3–1.5× *slower* below n≈16k) and Prizma trains ~5×
slower per step (sequential delta); the FLOP-matched TF arms were optimization-confounded so **no
per-FLOP claim** is made; n=2–3 seeds are descriptive (not powered equivalence); **large-scale LM
parity and backprop-free parity are NOT claimed** (open frontiers).

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

- Full writeup (equations, borrowed-vs-new ledger, neuromorphic mapping, limits) → **[`docs/Prizma.md`](docs/Prizma.md)**
- Code → `src/` (prizma + baselines + data + metrics), `experiments/` (E1–E5 suite + figure)

```bash
python3.13 -m venv .venv && ./.venv/bin/pip install numpy matplotlib
./.venv/bin/python experiments/run_continual.py   # ~2.5 min → results/results.json
./.venv/bin/python experiments/make_figure.py      # → results/figure.png
```

> Status: research prototypes. Neither thread claims large-scale parity; each is a falsifiability
> gate passed (or honestly refused) in a precisely-characterized small-scale regime.

---

## Reproducing the falsifiability harness

The invariant test suite (kernel guards, lever `off == identical` checks, O(1) `step == forward`
equivalence, and the anti-conservative statistics gate) runs on every push via
[CI](.github/workflows/ci.yml):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt pytest
pytest -q          # 94 tests, ~25 s on CPU
```

## License

Prizma is released under the [Apache License 2.0](LICENSE). © 2026 The Prizma Authors.
