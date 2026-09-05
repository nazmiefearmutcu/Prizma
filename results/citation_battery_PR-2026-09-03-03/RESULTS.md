# PR-2026-09-03-03 — single-pass citation bar — RESULTS (registered run)

- Protocol: docs/CONTINUAL_CITATION_BAR.md §6 executed VERBATIM (single-pass battery,
  arms = Table-2 set, seeds 10-19, tuning on the run's first seed).
- Pre-run interpretation notes (fixed before verdict computation): the doc's 'seed 0'
  references denote the run's FIRST seed (10); the bar uses the literal
  two-sample Welch 95% CI on ACC_PRIZMA − ACC_B*; the paired-by-seed CI is reported
  as supplementary (pairing is by shared task sequence).

## Per-arm means (± 1.96 SEM)

| arm | ACC | FGT |
|---|---|---|
| backprop | 0.432 ± 0.028 | 0.451 ± 0.032 |
| EWC | 0.451 ± 0.024 | 0.365 ± 0.025 |
| MAS | 0.247 ± 0.021 | 0.466 ± 0.022 |
| SI | 0.453 ± 0.022 | 0.138 ± 0.011 |
| OnlineEWC(online) | 0.429 ± 0.028 | 0.454 ± 0.033 |
| OnlineEWC(boundary) | 0.450 ± 0.026 | 0.396 ± 0.028 |
| replay | 0.530 ± 0.023 | 0.215 ± 0.020 |
| PRIZMA(DFA) | 0.500 ± 0.033 | 0.192 ± 0.028 |

- **B*** = OnlineEWC(online) (higher first-seed ACC among the frozen boundary-free candidates).
- Primary: ACC_PRIZMA − ACC_B* = +0.070; Welch 95% CI [+0.024, +0.117] (df=17.6).
- Supplementary paired-by-seed 95% CI: [+0.037, +0.103].
- Secondary (not gated): FGT 0.192 vs FGT_B* 0.454 (≤ B*+0.05: yes).

## VERDICT: **PASS**

Prizma is competitive with boundary-free regularizers in single-pass domain-incremental CL (claim usable, with the doc's honest-limit notes).

Raw per-seed records: `raw_single_pass.json` (retention per docs/RETENTION.md).
