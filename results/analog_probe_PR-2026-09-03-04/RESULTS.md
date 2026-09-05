# RESULTS — PR-2026-09-03-04 (CLAIM): delta-vs-additive analog robustness of the carried state

**Registry id:** PR-2026-09-03-04 · **Lane:** CLAIM · **Executed:** 2026-09-05 22:28 → 2026-09-06 02:38 TSS
· **Runner:** `seq/analog_probe_claim.py` (new; reuses `seq/analog_probe.py` machinery verbatim
where the frozen skeleton is identical) · **Binding protocol:** executed VERBATIM from
[`../analog_probe_2026-09-03/RESULTS.md`](../analog_probe_2026-09-03/RESULTS.md) **§7** (the
registered draft; its INDEX row was IN-WRITE at run time — the maintainer freezes it from THIS
document). §7 named the artifact directory `results/analog_probe_claim/`; the run wrote to
`results/analog_probe_PR-2026-09-03-04/` (dispatch directive); the mapping is 1:1 — this IS the
§7 primary artifact.

---

## 0. VERDICT: **H2a NOT-SUPPORTED** at the registered matched-clean operating point

The frozen decision rule required Δret ≥ **+0.05** (delta − additive retention) with one-sided
Welch p < 0.05 (Holm over the 2 primaries) on **BOTH** primaries. Outcome at D=24 (the first
difficulty where the matched-clean gate passed for both arms):

| primary | Δret (delta − additive) | Welch 95% CI | 1-sided p | Holm p | bar (≥ +0.05) |
|---|---|---|---|---|---|
| **P1: 4-bit state, σ=0** | **−0.0923** | [−0.1704, −0.0142] | 0.9866 | 0.9866 | **FAIL** (sign flipped: delta significantly WORSE) |
| **P2: FP32 state, σ=0.05** | **+0.0010** | [−0.0111, +0.0130] | 0.4215 | 0.8431 | **FAIL** (exact parity) |

O(1)-sanity gate: **PASS** — |stream(bits=0,σ=0) − forward| = 0.0000 for **all 20 included
models** (exact to displayed precision on deterministic CPU). The failure is the bars, not the
plumbing. Per the frozen rule: **H2a is recorded NOT-SUPPORTED at this scale**, the report-12-H2
**H2b** reading is recorded ("the correction requires a precise read of `S·k`; a quantized S
corrupts the residual, so delta does NOT tolerate low-precision state better" — the P1 CI is
entirely negative, actively supporting H2b), and the deployment differentiator (analog/state-
precision robustness of the delta write) is not claimable anywhere.

## 1. What ran (registered protocol, executed verbatim)

- **Skeleton (§2-identical):** MixedMQAR(vocab=128, num_queries=64, gap=0, min_pairs=1);
  PrizmaSeqLM d64L2H2, feat_map='none'; AdamW 3000 steps, batch 64, warmup 300, cosine→0.1×;
  build_and_train seeding (seed BEFORE construction); train-FP32 / deploy-degraded through the
  post-fix `step()` path; frozen eval = 8 batches × 64 under eval_seed 12345; CPU, 8 threads.
- **Per-arm LR sweep** over `seq.lrsweep.DEFAULT_GRID` (5e-4, 1e-3, 1.5e-3, 2e-3, 3e-3) at
  sweep seed 0, best on the frozen eval, full grid recorded for audit (§7 LR-fairness). Chosen
  LR = **3e-3 for BOTH arms at BOTH difficulties** (delta D=32: 0.920 / D=24: 0.914; additive
  D=32: 0.931 / D=24: 0.975 — full grids in §4).
- **Matched-clean gate (binding):** clean streaming acc ≥ 0.80 on ≥ 4/5 seeds, per arm, at the
  shared difficulty; O(1)-sanity ≤ 0.005 per model.
- **n = 5 seeds per arm (0–4), no substitution.** Cells evaluated **primaries-first** per model
  (crash-safe per-run JSON, never overwritten; resume = re-run — used 3× after external kills,
  zero data loss, no seed reduced, no bar touched).

## 2. Stage 1 — D=32: matched-clean gate FAILED for delta → joint difficulty drop

| arm | chosen LR | clean streaming acc per seed (0–4) | ≥0.80 count | gate |
|---|---|---|---|---|
| delta | 3e-3 | 0.922, 0.780, 0.789, **0.419**, 0.852 | **2/5** | **FAIL** (incl. one outright non-solver) |
| additive | 3e-3 | 0.931, 0.821, 0.786, 0.959, 0.816 | 4/5 | PASS |

The sweep **fixed the additive solver rate** (exploratory: 1/3 solvers at frozen lr=2e-3; here
4/5 pass the gate at lr=3e-3 — exactly the confound §7's LR-fairness was designed to remove),
but **delta** became seed-fragile at D=32/lr=3e-3 (2/5, one seed at 0.419). Per §7 the
difficulty drops **identically for both arms** to D=24 (never per-arm); the D=32 runs are kept
on disk as the audit trail (`decision_d32.json`, `runs/run_d32_*`), and NO retention comparison
is claimed at D=32.

## 3. Stage 2 — D=24 (registered comparison): gate PASSED for both arms

| arm | chosen LR | clean streaming acc per seed (0–4) | ≥0.80 count | gate | O(1)-sanity |
|---|---|---|---|---|---|
| delta | 3e-3 | 0.916, 0.875, 0.822, 0.753, 0.924 | 4/5 | **PASS** | 0.0000 all 5 |
| additive | 3e-3 | 0.975, 0.872, 0.892, 0.991, 0.971 | 5/5 | **PASS** | 0.0000 all 5 |

Retention = acc(condition)/acc(own knobs-off streaming), per seed, n=5 per arm. Per-arm
retention mean ± sd [t-CI95], and the head-to-head (Welch, one-sided H1: delta > additive):

| condition | delta ret | additive ret | Δret | Welch 95% CI of Δ | 1-sided p | Holm p | bar |
|---|---|---|---|---|---|---|---|
| **4-bit, σ=0 (P1)** | 0.8407 ± 0.0408 [0.790, 0.891] | 0.9330 ± 0.0615 [0.857, 1.009] | **−0.0923** | [−0.170, −0.014] | 0.9866 | 0.9866 | **FAIL** |
| **FP32, σ=0.05 (P2)** | 0.9852 ± 0.0032 [0.981, 0.989] | 0.9842 ± 0.0099 [0.972, 0.996] | **+0.0010** | [−0.011, +0.013] | 0.4215 | 0.8431 | **FAIL** |

Per-seed retentions (audit; order = seeds 0–4):

- P1 delta: 0.8778, 0.8731, 0.8439, 0.7765, 0.8320 · P1 additive: 0.9520, 0.8252, 0.9454, 0.9768, 0.9654
- P2 delta: 0.9837, 0.9815, 0.9856, 0.9850, 0.9902 · P2 additive: 0.9865, 0.9797, 0.9695, 0.9938, 0.9917

Secondaries (§7-registered, descriptive only — no claim; Holm n/a) and extra cells
(exploratory-grid carryover, descriptive only):

| condition | delta ret | additive ret | Δret [CI95] |
|---|---|---|---|
| 6-bit, σ=0 (secondary) | 0.9873 ± 0.0027 | 0.9963 ± 0.0035 | −0.0090 [−0.014, −0.004] |
| 8-bit, σ=0 (secondary) | 0.9993 ± 0.0013 | 0.9994 ± 0.0010 | −0.0000 [−0.002, +0.002] |
| FP32, σ=0.01 (secondary) | 0.9997 ± 0.0006 | 0.9996 ± 0.0009 | +0.0001 [−0.001, +0.001] |
| 4-bit, σ=0.01 (secondary) | 0.8396 ± 0.0432 | 0.9310 ± 0.0634 | −0.0914 [−0.172, −0.010] |
| 6-bit, σ=0.01 (secondary) | 0.9863 ± 0.0026 | 0.9957 ± 0.0036 | −0.0094 [−0.014, −0.005] |
| 8-bit, σ=0.01 (secondary) | 0.9986 ± 0.0002 | 0.9988 ± 0.0014 | −0.0002 [−0.002, +0.002] |
| 4-bit, σ=0.05 (extra) | 0.8388 ± 0.0410 | 0.9128 ± 0.0650 | −0.0740 [−0.156, +0.008] |
| 6-bit, σ=0.05 (extra) | 0.9738 ± 0.0056 | 0.9811 ± 0.0135 | −0.0072 [−0.024, +0.009] |
| 8-bit, σ=0.05 (extra) | 0.9840 ± 0.0030 | 0.9833 ± 0.0106 | +0.0006 [−0.012, +0.014] |

## 4. LR sweep audit trail (full grids, best_acc on the train-time frozen eval, sweep seed 0)

| arm / D | 5e-4 | 1e-3 | 1.5e-3 | 2e-3 | 3e-3 | chosen |
|---|---|---|---|---|---|---|
| delta D=32 | 0.101 | 0.237 | 0.740 | 0.855 | **0.920** | 3e-3 |
| additive D=32 | 0.078 | 0.080 | 0.213 | 0.746 | **0.931** | 3e-3 |
| delta D=24 | 0.288 | 0.510 | 0.842 | 0.854 | **0.914** | 3e-3 |
| additive D=24 | 0.090 | 0.089 | 0.730 | 0.914 | **0.975** | 3e-3 |

## 5. Honest interpretation and limits

1. **The exploratory direction REVERSED at the registered operating point.** The exploratory
   probe's headline (+0.103 at 4-bit matched-clean) was built on an additive side of n=1 and a
   floor-confounded additive arm; with the LR sweep giving 5/5 (D=24) additive solvers, additive
   retention at 4-bit (0.933) is HIGHER than delta's (0.841). This is precisely the confound §7's
   matched-clean gate existed to remove — the gate worked, and the prior did not survive it.
2. **P1's CI is entirely negative** → at matched-clean, the delta write's self-correcting read
   (`S·k` residual) is genuinely MORE sensitive to a quantized carried state than additive
   accumulation — the H2b mechanism, now measured at n=5 with the O(1)-sanity gate green.
3. **Write noise σ=0.05 on an FP32 state is a parity cell** (Δ +0.001 [−0.011, +0.013]): both
   writes repair sub-2% write noise to ≤1.5% retention loss; there is no separation to claim.
4. **Operating-point shift (disclosed):** the registered comparison ran at D=24, one rung down
   §7's own ladder, because the gate (correctly) refused D=32 where delta went seed-fragile.
   The claim under test was always "at matched clean accuracy" — D is a protocol parameter, and
   the drop was applied identically to both arms per the frozen rule.
5. **step()-path-only limitation (unchanged, documented):** the analog levers are wired to the
   exact `step()` path only; the chunk-parallel training kernel never materializes per-token
   states. Train-FP32/deploy-degraded is the honest emulation of inference-time degradation, not
   QAT; conclusions are scoped to this deployment path.
6. **No retry at other bit widths / difficulties** without a NEW pre-registration (§7
   pre-committed negative handling). Per that clause: the report-12 H2b reading is recorded as
   the accepted interpretation, and the `docs/Prizma.md §5` analog wording is DUE to be scoped to
   exclude state-precision claims — NOT executed here (concurrent session owns `src/`-adjacent
   edits; left to the maintainer with this trigger on record). README/docs must not claim any
   analog/state-precision deployment differentiator for the delta write.
7. **Machine-sharing disclosure (wall-clock only):** the run shared the box with a concurrent
   RALPH iteration (PR-05) and one foreign pytest, was launched twice in duplicate by mistake
   (second copy killed 22:47 — deterministic fixed-seed training makes records identical, only
   wall times of the first contended units are inflated), and the harness killed the tracked
   task 3× (23:36, 00:43, 01:52) — each time resumed from crash-safe per-run JSON with ≤1 unit
   re-trained. Final leg ran detached (single process). Pure final-leg compute 2614 s; total
   training units across stages: 40 runs ≈ 3.9 h train-only wall (within §7's 2.5–4 h CPU
   estimate); no budget guard fired, n=5 never reduced, cells evaluated primaries-first.

## 6. Artifacts

- `meta.json` — frozen protocol record (grid, gate, bar, seeds, cell order). · `sweep_d{32,24}_{delta,additive}.json` — full LR grids.
- `runs/run_d{D}_{arm}_seed{s}.json` — 20 raw per-run records (train history, forward acc, 12
  cells each, sanity). Never overwritten; `_smoke/` = plumbing check only (30-step, never cite).
- `decision_d32.json` / `decision_d24.json` — gate outcomes per stage. · `summary.json` / `verdict.json` — machine-readable tables + frozen-rule verdict.
- `run.log`, `run_detached.log`, `run_detached.err.log` — execution logs (wall-clock evidence).
- Runner: `seq/analog_probe_claim.py`. Exploratory prior: `../analog_probe_2026-09-03/`.

## 7. Files changed by this execution (scope audit)

- NEW `seq/analog_probe_claim.py` (registered runner; imports skeleton constants + eval helpers
  from `seq/analog_probe.py` verbatim; `seq/prizma_seq.py`, `src/`, `tests/`, `docs/preregistry/*`
  untouched; no commits). NEW `results/analog_probe_PR-2026-09-03-04/` (this document + raw).
- Suite: **325 passed + 10 skipped, 0 failed** (2026-09-06 00:58 run, with this file present).
  Brief's baseline was 315P+10S; the +10 are the concurrent PR-05 iteration's
  `tests/test_mmax_lever.py` (not this wave's work). Green either way.

**For the maintainer:** INDEX row for PR-2026-09-03-04 can now be frozen to EXECUTED — NOT-SUPPORTED
(verdict §0), artifact path this directory. Pre-committed negative handling (§7): H2b recorded;
`docs/Prizma.md §5` scoping DUE (see §5.6).
