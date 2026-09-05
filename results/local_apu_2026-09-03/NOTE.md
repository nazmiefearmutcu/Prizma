# Local APU planning note (2026-09-03/05) — PLANNING NUMBERS, NOT CLAIMS

Machine: AMD Ryzen 7 PRO 5750G (8C/16T, Cezanne) + Radeon Graphics iGPU (~2 GB shared),
30 GB RAM, Windows, **no CUDA**. Purpose: how to run Prizma compute locally while the
zero-GPU owner decision is active.

## Measurements (scripts + raw JSON in this directory)

CPU (repo venv, torch 2.14.0+cpu, PrizmaSeqLM fwd+bwd, chunk=64, window=16, seq len 1024):

| threads | B×T | d_model×H | ms/step | tokens/s |
|---|---|---|---|---|
| 8 | 8×1024 | 64×2 | 262 | 31,244 |
| 8 | 8×1024 | 128×4 | 632 | 12,960 |
| 8 | 32×1024 | 128×4 | 2,425 | 13,515 |
| 16 | 8×1024 | 64×2 | 304 | 26,989 |
| 16 | 8×1024 | 128×4 | 752 | 10,895 |
| 16 | 32×1024 | 128×4 | 2,230 | 14,692 |

Finding 1 — **8 threads is the right default for small/medium shapes** (oversubscription
costs 8–16% at d=64/d=128 B=8; 16 threads wins ~8% only at B=32). Set `BENCH_THREADS=8` /
`torch.set_num_threads(8)` for local runs.

Finding 2 — **the iGPU cannot currently run the core kernel**: torch-directml (throwaway
venv `~/.prizma_dml_venv`, older pinned torch) raises `RuntimeError: The size of tensor a (0)
must match the size of tensor b (64) ... dimension 3` inside the chunked WY/UT delta path
(`torch.eye` composition in seq/delta.py) for every shape tried. Recorded verbatim in
`bench_dml.json`. An op-coverage gap of this kind is typical for DirectML on scan/WY-style
kernels. Consequences:

- Local compute = the CPU side of the APU. All wave-C experiments this month already ran
  there (citation battery ~4.5 min, expert economy ~8 s, analog probe ~93 min).
- The iGPU becomes usable only via (a) a DirectML-compatible fallback kernel path, or
  (b) WSL2 + ROCm on a supported APU (5750G is not officially ROCm-supported), or (c) a
  future torch-directml release covering the missing ops. None is worth engineering time
  while GPU budget exists for the decisive runs (Colab A100/L4 when the owner re-enables it).

## What the APU can and cannot carry (planning arithmetic, disclosed as such)

At ~13k tok/s (d=128): a 100-step toy sweep ≈ 20 min; the analog-probe grid was sized
exactly this way. NOT locally feasible: pre-registered claim-grade campaigns (n≥5 recall
gate ≈28 A100-h ≈ weeks on CPU), 50M scaling rungs, char-LM B4 closure — these stay
GPU-return work per the owner's locked order (synthesis §6).
