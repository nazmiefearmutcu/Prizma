# Prizma

[![CI](https://github.com/nazmiefearmutcu/Prizma/actions/workflows/ci.yml/badge.svg)](https://github.com/nazmiefearmutcu/Prizma/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Last commit](https://img.shields.io/github/last-commit/nazmiefearmutcu/Prizma)](https://github.com/nazmiefearmutcu/Prizma/commits)
[![Stars](https://img.shields.io/github/stars/nazmiefearmutcu/Prizma?style=social)](https://github.com/nazmiefearmutcu/Prizma/stargazers)

Research code for local continual learning and recurrent sequence modeling. Prizma includes model implementations, comparison baselines, experiment runners, pre-registered protocols, and raw result records. It is intended for reproducing and inspecting small-scale experiments.

## Research threads

| Thread | What it investigates | Implementation |
|---|---|---|
| **Prizma** | Learning successive, distinguishable domains without receiving task labels or task boundaries | NumPy predictive-coding experts, surprise-based routing, and local updates in [`src/`](src/) |
| **Prizma-Seq** | Associative recall and language modeling with a fixed-size recurrent state | PyTorch delta-state mixer, quadratic feature maps, and a local attention window in [`seq/`](seq/) |
| **Prizma-LM experiments** | Combining the sequence backbone with routed expert predictors during corpus changes and revisits | Fusion model, ablations, and registered campaigns in [`seq/fusion_probe.py`](seq/fusion_probe.py) and [`seq/prizma_lm_claim.py`](seq/prizma_lm_claim.py) |

The local Prizma learner uses backpropagation-free update rules. Prizma-Seq and the language-model experiments use gradient training; that distinction applies throughout the repository.

## Architecture

**Local learner.** Each expert contains an encoder, reconstruction decoder, and classifier. Reconstruction error determines which expert recognizes an input. Unrecognized inputs can recruit another expert; consolidation protects learned domains. The random-feedback variant updates the encoder without reading transposed forward weights. Routing depends on information in the input, so ambiguous inputs with conflicting target labels remain a limiting case.

**Sequence model.** Tokens pass through residual blocks containing an associative state updated by a gated delta rule, a short causal convolution, a local attention window, and a feed-forward layer. Optional fixed quadratic key/query features increase the state's feature dimension without adding trainable feature-map parameters. The recurrent `step()` path carries state whose size stays fixed as context length grows for a fixed model configuration.

**Continual language model.** A shared sequence backbone supplies features to a pool of expert residual predictors. Segments route by prediction surprise. Registered runs examine recruitment, protected experts, learning-rate schedules, retention, adaptation, and returning-domain routing against separate controls.

## Start locally

The CPU test workflow uses Python 3.11 and 3.12. Runtime dependencies are NumPy, PyTorch, and Matplotlib.

```bash
git clone https://github.com/nazmiefearmutcu/Prizma.git
cd Prizma
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on macOS/Linux or `.venv\Scripts\Activate.ps1` in PowerShell, then install the CPU environment:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt pytest
python -m pytest -q
```

Run the local continual-learning experiments and generate their figure:

```bash
python experiments/run_continual.py
python experiments/make_figure.py
```

These commands write `results/results.json` and `results/figure.png`. Use a separate checkout for new experiments when retaining the committed records.

For sequence experiments, see the [Prizma-Seq report](docs/PRIZMA_SEQ_REPORT.md), [protocol registry](docs/preregistry/INDEX.md), and `PRIZMA_run_*.ipynb` notebooks. GPU runners include `gpu_diag.py`, `gpu_charlm2.py`, `gpu_latency.py`, and `gpu_lengen.py`; hardware and dataset requirements depend on the experiment. Runners that support `PRIZMA_RESULTS` can write to a separate results directory.

## Evidence and limits

| Area | Current reading |
|---|---|
| Local continual learning | The repository reports retention and routing results on input-distinguishable synthetic domains. This does not establish general continual-learning performance. See the [local learner report](docs/Prizma.md). |
| Sequence diagnostics | Recall, copy, latency, memory, and language-model experiments cover small models. Long-context inference gains coexist with slower training and short-context losses. See the [sequence report](docs/PRIZMA_SEQ_REPORT.md). |
| Continual language modeling | Registered CPU campaigns report routing and revisit results at toy scale. Later controls attribute boundary-damage protection to the learning-rate schedule; the fresh-seed study did not establish its registered halving margin. See [PR-17](docs/preregistry/2026-09-10-scheduled-tf-attribution.md), [PR-20](docs/preregistry/2026-09-10-damage-gap-replication.md), and [PR-21](docs/preregistry/2026-09-11-recovery-attribution.md). |

Read these qualifications before citing results:

- The June 2026 powered recall-gate ledger is **quarantined** because cached smoke records contaminated the campaign. Its metrics and confidence intervals must not be used. See the [contamination notice](results/campaign_2026-06-08/CONTAMINATION.md).
- The original two-corpus language-model bar is **partial**: the retained text8 results do not complete the protocol, and the failed tiny-shakespeare run's raw records were lost. See the [retention policy](docs/RETENTION.md).
- Parameter counts and memory formulas for 50M/100M configurations are analytical comparisons. Those models were not trained; MMLU/GSM8k evaluations and GLA/Mamba-2 comparisons have no committed results.
- Large-scale language-model parity, a general replacement for attention, and deployed neuromorphic performance remain unestablished. GPU confirmation of the continual-LM campaign is pending.

## Repository guide

- [`src/`](src/): local learner, data generators, baselines, and metrics.
- [`seq/`](seq/): sequence models, kernels, comparison arms, and claim runners.
- [`experiments/`](experiments/): local continual-learning experiments and plotting.
- [`tests/`](tests/): kernel equivalence, causality, routing, statistics, and regression checks.
- [`docs/preregistry/`](docs/preregistry/): protocols, amendments, and outcome registry.
- [`results/`](results/): raw records, reports, and quarantine notices.

The [pre-registration policy](docs/preregistry/POLICY.md) and [artifact-retention policy](docs/RETENTION.md) describe how to separate exploratory work from registered claims and retain failed runs.

## License

[Apache License 2.0](LICENSE).

Created By Aylin (Nazmi Efe Armutcu)
