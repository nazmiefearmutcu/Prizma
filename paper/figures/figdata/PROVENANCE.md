# Figure provenance — every plotted number traces to a committed `results/*.json`

All five paper figures derive from JSONs committed under `results/`. No figure uses
uncommitted ("Colab Drive") data; the honesty guardrail (footnote PAPER-ONLY points)
was **not** triggered — all values are reproducible from this repo.

| Fig | Content | Source(s) | Builder |
|---|---|---|---|
| **F1** | Rectangular delta-state schematic | — (conceptual diagram, no data) | `make_figures.py:fig1_architecture` |
| **F2** | Per-step decode latency vs `n` | `results/gpu_latency.json` → `cells.{TF,Prizma-quad2}.small.n{4096,8192,16384,32768,65536}.per_step_ms` | `make_figures.py:fig2_latency` |
| **F3** | Inference memory vs `n` (measured peak + analytic floats) | `results/gpu_latency.json` → `peak_bytes` (measured), `kv_floats`/`state_floats` (analytic) | `make_figures.py:fig3_memory` |
| **F4** | MQAR accuracy vs matched params (grid 130/461/857/3300 K) | `results/gpu_bench.json` (keys below) → `figdata/fig4_mqar.json` | `figdata/build_figdata.py` → `make_figures.py:fig4_mqar` |
| **F5** | Causal feature-map ablation at 130K | `results/gpu_bench.json` (keys below) → `figdata/fig5_ablation.json` | `figdata/build_figdata.py` → `make_figures.py:fig5_ablation` |

## F4 — MQAR-vs-params (all `results/gpu_bench.json`)

| Point | Key | median | solve |
|---|---|---|---|
| Prizma-quad2 131K | `p2eff.Prizma-quad2-d64L2H2.s{0,1,2}` | 0.9967 | 3/3 |
| Transformer 130K | `p1.TF.d64L2H2.s{0,1,2}` | 0.0163 | 0/3 |
| Transformer 461K (smallest TF solver) | `p1.TF.d128L2H4.s{0,1,2}` | 0.9999 | 3/3 |
| Prizma-quad2 862K | `p2.Prizma-quad2.s{0,1,2}` | 0.9983 | 3/3 |
| Transformer 857K | `p2.TF.s{0..4}` | 0.9995 | 3/3 |
| Transformer 3271K | `p1.TF.d256L4H8.s{0,1,2}` | 0.9999 | 3/3 |

Efficiency gap: Prizma solves at 131K; smallest TF solver 461K → **461/131 = 3.5×**.

## F5 — causal ablation at 130K (d64L2H2, MQAR D=128, 3 seeds; all `results/gpu_bench.json`)

| Condition | Key | median | seeds | solve |
|---|---|---|---|---|
| quad2 | `p2eff.Prizma-quad2-d64L2H2.s{0,1,2}` | 0.9967 | [0.999, 0.997, 0.935] | 3/3 |
| rand_linear | `p2eff.Prizma-randlin-d64L2H2.s{0,1,2}` | 0.5893 | [0.668, 0.408, 0.589] | 0/3 |
| none | `p2eff.Prizma-none-d64L2H2.s{0,1,2}` | 0.5183 | [0.518, 0.319, 0.718] | 0/3 |
| TF | `p1.TF.d64L2H2.s{0,1,2}` | 0.0163 | [0.016, 0.016, 0.016] | 0/3 |

These reproduce the paper's §7 ablation (`quad2 ≫ rand_linear ≈ none ≫ TF`) exactly.
Cross-checked against `results/v3_campaign_results.md` and `results/d128_sweep.json`.
