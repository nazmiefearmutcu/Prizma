# Lane 2 — Prizma-Seq CPU kernel (`fast_reads` + C1 solve simplification) — report

**Date:** 2026-09-11 · **Box:** Windows, AMD Ryzen 7 PRO 5750G (8C/16T), torch 2.14.0+cpu,
venv Python 3.12.10, `torch.set_num_threads(8)` · **HEAD at proof time:** `3a0e4ca`

**Status: COMPLETE.** All work items delivered; every default numerical path proven byte-identical
(== 0.0) against pre-edit HEAD; tests green; A/B measured and reported honestly below.

---

## 1. Delivered

### C2 — opt-in `fast_reads` (default OFF) in `chunked_delta`
- `seq/delta.py:375-376` — trailing kwarg `fast_reads: bool = False` (positional-compat safe; every
  call site uses keywords).
- `seq/delta.py:535-542` — gated WY/UT branch only: `read_ratio = ratio / Ac[..., :, None]`
  (= `(gamma_i/gamma_j)/alpha_i`, since `gamma_i = alpha_i * gamma_{i-1}`) instead of the second
  full `[B,H,C,C]` `sub+exp`. Removes 2 of ~30 kernels per chunk. `alpha in [0.5,1]` on the
  production config -> exact in real arithmetic; ~5e-6 float32 drift (NOT byte-identical -> opt-in).
- `seq/delta.py:406-414` — docstring: scope (gated WY/UT only; inert on pure/eta/surprise/
  surprise_norm/n_delta>=2) and the not-byte-identical disclosure.
- The `_chunked_delta_eta` twin (`seq/delta.py:337`) is deliberately NOT touched (survey-2 scope).

### C1 — bit-identical solve simplification (default-on)
- `seq/delta.py:270-276` — `_solve_unit_lower` now passes strictly-lower `Amat` directly with
  `unitriangular=True` (LAPACK ignores the diagonal); dropped `M = torch.eye(C) + Amat`.
  Measured `maxdiff == 0.0` vs the old expression (CPU), pinned by test.

### Tests (NEW)
- `tests/test_fast_reads.py` (4 tests): production-shape (B=2,H=3,T=256,d=16,C=64, alpha~U[0.5,1])
  fast-vs-default `max|dO| < 1e-5` (measured 4.8e-06), `max|dS| < 1e-6` (measured 0.0); gradient
  parity `< 1e-4` absolute **and** relative (mean-squared loss; measured worst abs 1.5e-08, rel
  9.2e-07); default-off byte identity `== 0.0` on 5 configs (pure/gated/additive/gated-additive/
  rectangular); `fast_reads` inert (`== 0.0`) where the gate is absent.
- `tests/test_delta.py` (6 tests): plain default `chunked_delta` vs `_delta_reference` parity
  (pure/gated/additive/rectangular) — the repo's missing pytest coverage; C1 direct-A solve
  equivalence `== 0.0`; Neumann fallback vs direct solve `< 1e-5` (forced via monkeypatched
  `solve_triangular`).

### Benchmark harness
- `seq/throughput_benchmark.py:172-179` — `--fast-reads` (opt-in) + `--ab-runs` (default 15).
- `seq/throughput_benchmark.py:237-263` — with the flag: extra `Eager+fast_reads` CPU rows via the
  same harness; **default rows unchanged**; without the flag the script behaves exactly as before.
- `seq/throughput_benchmark.py:122-161` — `benchmark_interleaved_forward`: order-balanced
  (A,B / B,A alternating) interleaved forward-only A/B, median of N pairs, same tensors/process.
- `seq/throughput_benchmark.py:364-378` — prints the A/B block; in `--fast-reads` mode
  `benchmark_results.md` is **never written** (publish-by-hand gate).

## 2. Default-path proof (no numerical path altered)

**Cross-version byte-identity** (`committee/campaign_2026-09-11/lane-2-default-path-proof.py`,
log `lane-2-default-path-proof.log`): loads `HEAD:seq/delta.py` (pre-edit) as a separate module and
compares outputs on identical inputs. All `max|dO| == max|dS| == 0.0`:

| branch | result |
|---|---|
| pure / gated / gated-floor(0.5) / additive / gated-additive | BIT-IDENTICAL |
| decoupled `beta_e` / eta gated / eta pure (batched solve path) | BIT-IDENTICAL |
| surprise norm / surprise_norm A4 / n_delta=2 | BIT-IDENTICAL |
| gated gradients (w.r.t. q,k,v,beta,alpha) | worst diff == 0.0 |

Plus: `tests/test_fast_reads.py::test_fast_reads_default_off_byte_identity` (== 0.0 on 5 branches),
`test_solve_direct_a_bit_identical_to_eye_plus_add` (== 0.0), and `python seq/delta.py` -> **ALL OK**.

## 3. Test gates

- Required batch (8 files: test_fast_reads, test_delta, test_inctx_lr, test_decoupled,
  test_surprise, test_deltaproduct, test_fused, test_delta_triton): **82 passed, 9 skipped**
  (CUDA-only skips), ~4s.
- New files: `test_fast_reads.py` 4 tests, `test_delta.py` 6 tests — all pass.
- `python seq/delta.py` self-test: **ALL OK** (all cases incl. repeated-key surprise and eta).
- Full-suite gate is coordinator-run per contract.

## 4. Benchmark — measured medians (raw, honest)

Shape B=8, H=4, T=1024, d=64, C=64 (8,192 tok/pass); interleaved order-balanced forward-only
median-of-15 per run; `torch.set_num_threads(8)`. `benchmark_results.md` sha256 unchanged
`7A8F3963...F06B0` before and after all runs (flag mode never writes it).

| run | default median | fast_reads median | median speedup | (default min → fast min) |
|---|---|---|---|---|
| 1 | 25.490 ms | 24.679 ms | **1.0329x** | 22.538 → 23.509 |
| 2 | 22.215 ms | 22.322 ms | **0.9952x** | 18.687 → 17.935 |
| 3 | 24.327 ms | 23.030 ms | **1.0563x** | 22.405 → 21.212 |

- Pooled ratio (sum of medians): 72.032 / 70.031 = **1.0286x**; median of run speedups = 1.0329x.
- Mean-timed table rows (20 runs, same harness as every other row): forward 35.36→32.53 (1.09x),
  31.01→29.08 (1.07x), 30.10→28.48 (1.06x); backward 1.05x / 1.01x / 0.96x (no consistent win —
  do not claim backward).
- **Verdict:** target ≥ 1.02x met in 2/3 interleaved runs and in the pooled ratio; run 2's 0.995x
  is within the box's documented noise (default medians themselves ranged 22.2–25.5 ms across runs;
  survey-2 measured 1.039x). The lever is real but small (~3% forward CPU wall on this box) and
  **must stay opt-in/default-OFF**. No training-step claim is made.

## 5. Artifacts

- `committee/campaign_2026-09-11/lane-2-fastreads.md` (this report)
- `committee/campaign_2026-09-11/lane-2-default-path-proof.py` + `.log`
- `committee/campaign_2026-09-11/lane-2-bench-run{1,2,3}.log` (raw benchmark output incl. table)

## 6. Notes / recommendations for the coordinator

- Keep `fast_reads` default-OFF (opt-in) and unadvertised in the README table; fold the median
  numbers above into docs only with the explicit opt-in caveat.
- C2 on the eta path (`_chunked_delta_eta`, line ~337) remains open as a future extension (survey
  scope decision).
- `torch.compile` on this box remains unavailable (InvalidCxxCompiler; no MSVC) — unchanged,
  unrelated to this lane.
- Files outside lane ownership were not touched (working tree also contains Lane 1/3 + coordinator
  edits).
