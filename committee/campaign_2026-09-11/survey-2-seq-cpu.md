# Survey 2 — Prizma-Seq CPU thread (read-only, 2026-09-11)

**Env**: Windows, venv Python 3.12.10, torch 2.14.0+cpu, 8 threads (Ryzen 8C/16T), no MSVC (`cl.exe` absent).
**Noise disclosure**: this box is shared; one and the same `chunked_delta` pass measured 34.9–198.6 ms across
runs. The repo itself documents ~2x run-to-run variance (`seq/benchmark_results.md:17-20`). All below-relative
numbers come from interleaved A/B runs or a single in-process profiler pass; absolute ms are indicative only.
No source file was modified (only this report written).

## 1. Baseline inventory

**Published numbers** (README.md:129-138 == seq/benchmark_results.md:27-36, byte-equal table):
Eager CPU fwd 41.18 ms / 198,934 tok/s; Eager CPU bwd 104.85 ms; Compiled CPU fwd 26.57 ms (1.55x),
bwd 43.17 ms (2.43x); MPS rows 42.72/88.60/42.00/84.71. Host "macOS-27.0-arm64 / Apple M4"
(benchmark_results.md:9); shape B=8 H=4 T=1024 d=64 chunk=64 = 8,192 tok/pass (benchmark_results.md:11-13);
5 warmup + 20 timed runs (benchmark_results.md:15).

**Harness**: `seq/throughput_benchmark.py` — timing loops :54-118 (separate fwd/bwd loops with sync barriers);
CPU "Compiled" row compiles `chunked_delta` directly (:171); MPS "Compiled" row goes through
`fused_chunked_delta(..., backend="compile")` (:209-210); report writer :254-325 (overwrites
`benchmark_results.md`).

**Hot loops (file:line anchors)**:
- `_delta_reference` per-token Python loop: `seq/delta.py:186-256` (ground truth; also the exact fallback for
  surprise / surprise_norm / eta+additive).
- `_solve_unit_lower` (unit-lower triangular solve + Neumann fallback): `seq/delta.py:260-284`
  (try `solve_triangular` :271-272; nilpotent fallback :273-284).
- `_chunked_delta_eta` batched per-channel solve: `seq/delta.py:287-367` (solve :351; 2nd exp at :337).
- `chunked_delta` WY/UT main loop: `seq/delta.py:492-550` (gated branch :501-532, pure branch :533-548;
  2nd full [B,H,C,C] exp for `read_ratio` at :523).
- dispatch: surprise_norm → reference `:409-417`; eta → eta-path `:433-439`; surprise → reference `:446-450`;
  n_delta>=2 token loop `:461-490`.
- fused/compiled: CUDA-only predicate `seq/delta_fused.py:169-176`; lazy `torch.compile` cache :122-130;
  eager mirror `_chunked_delta_fast_path` :50-114.
- model: `PrizmaSeqBlock.forward` `seq/prizma_seq.py:418-489` (delta call :469 for n_delta=1);
  streaming `step()` :513-643.

**Measured on this machine (2026-09-11)**:
- `chunked_delta` (B8 H4 T1024 d64 C64, gated): 35–72 ms (README's Mac: 41.2 ms).
- Solve share, instrumented inside the pass (`_solve_unit_lower` wall accumulator), 3 interleaved passes:
  **22–28% of wall** (67.7ms→19.0, 71.7→17.2, 57.4→12.8).
- `torch.profiler` CPU-time attribution (5 passes, thread-summed): `linalg_solve_triangular` **50.0%**,
  `mul` 15.4%, `bmm` 12.6%, `add` 5.5%, `sub` 5.0%, `tril` 3.9%, `exp` 3.0%, `copy_` 2.0%, `eye` 0.1%.
- Chunk-size sweep: C=16: 164 ms · C=32: 115 ms · **C=64: 66 ms** · C=128: 134 ms → C=64 already optimal.
- Threads: 4: 85 ms · 8: 70 ms · 16: 273 ms → default 8 already optimal.
- `torch.compile`: **FAILS here** — `InductorError: InvalidCxxCompiler: Compiler: cl is not found`.
  `backend="aot_eager"` is *slower* (chunked 0.69x, reference 0.81x).
- `_delta_reference` T=256: 101–118 ms → ≈400-470 ms extrapolated at T=1024 ≈ **6-8x** the chunked pass
  (README.md:39 claims ~5x from matched-TF GPU runs; direction consistent).

**Concentration**: the solve is the single largest wall component (~25%); the rest is ~30 eager kernels per
chunk × T/C chunks — i.e. elementwise `mul/add/sub/exp/tril` traffic + dispatch, not raw FLOPs. This is a
memory/dispatch-bound eager workload.

## 2. CPU-feasible candidates

**C1 — Bit-identical solve simplification (recommended rider, default-on).**
`seq/delta.py:270-272`: pass `Amat` (zero diagonal) directly with `unitriangular=True` and delete
`M = torch.eye(C)+Amat`; LAPACK ignores the diagonal under unitriangular (0 is read as 1).
Measured: **maxdiff exactly 0.0** vs current; isolated solve median 0.365 vs 0.377 ms (min 0.203 vs 0.222);
removes one [B,H,C,C] `add` + eye per chunk. Effect: ~1-2% of solve CPU, **<1% of pass wall (conservative)**.
Risk: minimal. Verify: existing self-test `python seq/delta.py` + `tests/test_inctx_lr.py`,
`tests/test_decoupled.py`, `tests/test_surprise.py`, `tests/test_deltaproduct.py` (no test currently pins
the solve; add a one-line assert).

**C2 — Opt-in fast reads (recommended headline; default-off).**
`seq/delta.py:523`: replace the second `[B,H,C,C]` `sub`+`exp` (`read_ratio = exp(clog_prev - clog)`) with
`read_ratio = ratio / Ac[..., :, None]` (γ_{i-1}/γ_j = (γ_i/γ_j)/α_i; α∈[0.5,1] → safe). Used only under
`tril(...,-1)`, exact in real arithmetic.
Measured (interleaved 15×A/B, same process): median **29.6 → 28.5 ms = 1.039x** forward (conservative claim
**2-4%**); removes 2 of ~30 per-chunk kernels. **Not byte-identical**: stated conservatively,
max|dO| = 7.6e-06, max|dS| = 0.0 (production shape) — well inside the repo's 1e-4 parity bar but must be
**default-off** so all existing artifacts stay valid. Same trick applies to `_chunked_delta_eta:337`
(extension, out of tonight's scope).
Risk: low (float division only). Verify: new test modeled on `tests/test_inctx_lr.py:52-140` pinning
fast-vs-default <1e-5 (fwd) and grad <1e-4; off-path `fast=False` byte-identity (== 0.0) asserted.

**C3 — Hoist `gated/erase/decoupled` branch dispatch out of the chunk loop** (`seq/delta.py:493-549`).
Bit-identical (same ops/order), but Python `if` cost is ns vs ~200 µs of ops per chunk → **<1%**. Do only if
already editing the function.

**C4 — Chunk size**: do not change; measured optimum C=64 (above). Could record the sweep in
benchmark_results.md as an evidence note.

**C5 — `torch.compile` on CPU**: **not feasible tonight** — needs MSVC Build Tools (admin install, ~GBs);
current error is InvalidCxxCompiler. The README's 1.55x/2.43x CPU rows are Mac-only and do not reproduce
here. `aot_eager` tested slower; `torch.jit.script` untested (not a proven win). If a ≥10% CPU number is
required, this is the only known route and it is an infrastructure task + full re-benchmark, not a code diff.

**C6 — Blocked forward substitution for the solve** (sb=16/32): measured **slower** (1.60/0.92 ms vs 0.377 ms)
— rejected. General `torch.linalg.solve` (LU) is slower than the triangular solve; flattened-batch layout is
neutral. No faster exact standard solve exists on this box.

**C7 — Sequential-scan speedups (`_delta_reference`)**: dispatch-bound (~0.4 ms/token); `aot_eager` 0.81x
(slower). Exact chunk-parallel form is impossible for state-dependent gates (documented `seq/delta.py:57-62`,
R3 repeated-key argument). No CPU win found tonight.

**C8 — Monomial-buffer "caching"**: `feat_P`/`feat_I`/`feat_J` are already registered buffers
(`seq/prizma_seq.py:300-308`); the monomial gather depends on the live input and cannot be cached. No.

## 3. Correctness/coverage gaps

Covered: chunked vs reference for surprise incl. repeated keys (`tests/test_surprise.py:79-131`), eta
(`tests/test_inctx_lr.py:52-140`), decoupled erase (`tests/test_decoupled.py:46-71`), n_delta
(`tests/test_deltaproduct.py:33-70`); off-path identities in the same files; fused fallback vs chunked
<1e-6 (`tests/test_fused.py:54-104`); CUDA fwd+bwd fused parity skips locally (`tests/test_fused.py:216-270`).
**Missing**:
1. **No pytest for the plain default `chunked_delta` vs `_delta_reference`** — the only check is the
   script-style `__main__` at `seq/delta.py:553-598` (not collected; `tests/` has no `test_delta.py`).
2. No test of the `_solve_unit_lower` Neumann fallback vs `solve_triangular` (nor of C1's direct-A
   equivalence).
3. No CPU compiled-vs-eager test (impossible here; was never asserted on the Mac either).
4. Benchmark table ↔ README sync is manual (no test).

## 4. Docs-truth spot check (Prizma-Seq thread)

- **STALE**: `README.md:159` quotes quad2 crosstalk `~0.076` → N<14. The repo's own addendum
  (`docs/quad2_theoretical_convergence.md:183-205`) records the measured value **0.11699** (ratio 1.54 vs the
  unreproducible 0.076) and `:261-272` repeats that the η/ε numbers inherit the stale figure. README should
  cite 0.117 (N*≈9.5) or point at the addendum.
- No numeric mismatch between README table and `seq/benchmark_results.md:27-36` (same numbers, Mac host).
- Caveat scoping: `benchmark_results.md:22-25` attributes the "Compiled fallback" only to
  `seq/delta_fused.py`; the CPU "Compiled" rows actually compile `chunked_delta` directly
  (`throughput_benchmark.py:171`) and **cannot run on Windows without MSVC** (today's InvalidCxxCompiler).
  A sentence stating that would close the reproducibility gap.
- `README.md:39` "~5x slower per step (sequential delta)": direction still honest — measured ~6-8x here at
  T=1024 — and it applies to the surprise/surprise_norm configs, not the default `chunked_delta` path
  (`PrizmaSeqConfig.precision_gate` default "input", `seq/prizma_seq.py:47`).

## 5. Recommended single improvement (tonight)

**Implement C2: an opt-in, default-off `fast_reads` fast path in `chunked_delta`** (derive
`read_ratio = ratio / Ac[..., :, None]` at `seq/delta.py:523`; thread a `fast_reads: bool = False` kwarg
from `seq/delta.py:370`). It is the largest measured win available without a compiler, has a clean
default-off story that preserves every existing artifact, and its numeric drift is ~13x below the repo's own
1e-4 parity bar. (Fold in C1's direct-A solve while in the function — bit-identical, free.)
If a larger single improvement is mandatory, C5 (MSVC + `torch.compile`) is the only known route and must be
a separate infrastructure task.

**Exact acceptance criteria**
1. Default path untouched: `fast_reads=False` (default) output is bit-identical to today (`maxdiff == 0.0`)
   on the production shape; all existing gates pass (`pytest tests/ -q`; `python seq/delta.py` → ALL OK).
2. New test `tests/test_fast_reads.py` (pattern: `tests/test_inctx_lr.py:52-140`): gated production shape
   (B=2, H=3, T=256, d=16, C=64, random alpha∈[0.5,1]) — `fast_reads=True` vs `False`:
   max|ΔO| < 1e-5 and max|ΔS| < 1e-6 (measured on the bigger 8/4/1024/64 shape: 7.6e-6 / 0.0),
   grad parity < 1e-4.
3. Benchmark proof (interleaved A/B, median-of-15, `torch.set_num_threads(8)`, same process; single-run
   comparisons are meaningless at ~2x machine noise):
   `python seq/throughput_benchmark.py --runs 20 --fast-reads` vs plain run → require **median forward
   speedup ≥ 1.02x** on the Eager CPU gated row (measured 1.039x); report min/median, no p-values.
   If the harness has no `--fast-reads` flag yet, add it (it must print both rows, default row unchanged).

**Do not claim**: any wall-time improvement on the model training step until a block-level A/B is measured
under quiet conditions; any byte-identity for `fast_reads=True`; any CPU compiled speedup (compiler absent).
