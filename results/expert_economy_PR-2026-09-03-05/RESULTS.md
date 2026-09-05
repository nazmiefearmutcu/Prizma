# PR-2026-09-03-05 — bounded-M economy lever (Policy A) — RESULTS (registered run)

- Protocol: docs/EXPERT_ECONOMY.md §3.3 Policy A + §5 operationalisation, executed
  VERBATIM on the shipped E1 stream (structured_permuted_tasks, K=5, d=24, 8 classes,
  n_samples=6000, h=48, M_unbounded=K+3=8, epochs=15, DFA, consolidate=True, z_novel=5.0).
- Arms: UNBOUNDED = shipped (m_max=0, pool M=8); BOUNDED = Policy A recruit-by-eviction
  at M_max=K=5 (pool allocation stays 8; slots >= 5 never used).
- Seeds 3-12 (n=10, fresh; the exploratory seeds 0-2 untouched).
- Wall-time estimate measured before the campaign: first E1-stream run = 2.71s; total campaign ≈ 52s.
- Eviction policy under test: at a capped recruit, evict the committed expert in
  [0, M_max) with the lowest lifetime routing share route_log[m]/max(sum,1),
  tie -> highest slot index (most recently recruited); reuse its slot.

## Arms table (ACC-vs-M_max reporting duty; mean ± 95% CI, run_continual.ci95)

| arm | M_max | ACC | FGT |
|---|---|---|---|
| UNBOUNDED (shipped) | — (pool 8, 5.0 experts used mean) | 0.8501 ± 0.0108 | 0.0000 ± 0.0000 |
| BOUNDED (Policy A) | 5 (5.0 experts used mean) | 0.8501 ± 0.0108 | 0.0000 ± 0.0000 |

## Verdict arithmetic

- ΔACC = +0.000000 (bar: |ΔACC| <= 0.02)
- ΔFGT = +0.000000 (bar: |ΔFGT| <= 0.02)
- Per-seed max |ΔACC| = 0.000000; per-seed max |ΔFGT| = 0.000000 (honest per-seed spread; the bar compares seed-summary means).
- Eviction fires (bounded arm, all seeds): **0** (the §2.1 measured expectation — the cap is inert at home).
- Per-expert train-fraction ledger (mean over seeds, slots 0-7):
    - UNBOUNDED: [0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0]
    - BOUNDED:   [0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0]

## Per-seed records

| seed | ACC_unb | ACC_bnd | ΔACC | FGT_unb | FGT_bnd | ΔFGT | evictions (victims) |
|---|---|---|---|---|---|---|---|
| 3 | 0.8453 | 0.8453 | +0.0000 | 0.0000 | 0.0000 | +0.0000 | 0 ([]) |
| 4 | 0.8673 | 0.8673 | +0.0000 | 0.0000 | 0.0000 | +0.0000 | 0 ([]) |
| 5 | 0.8448 | 0.8448 | +0.0000 | 0.0000 | 0.0000 | +0.0000 | 0 ([]) |
| 6 | 0.8155 | 0.8155 | +0.0000 | 0.0000 | 0.0000 | +0.0000 | 0 ([]) |
| 7 | 0.8590 | 0.8590 | +0.0000 | 0.0000 | 0.0000 | +0.0000 | 0 ([]) |
| 8 | 0.8543 | 0.8543 | +0.0000 | 0.0000 | 0.0000 | +0.0000 | 0 ([]) |
| 9 | 0.8257 | 0.8257 | +0.0000 | 0.0000 | 0.0000 | +0.0000 | 0 ([]) |
| 10 | 0.8655 | 0.8655 | +0.0000 | 0.0000 | 0.0000 | +0.0000 | 0 ([]) |
| 11 | 0.8607 | 0.8607 | +0.0000 | 0.0000 | 0.0000 | +0.0000 | 0 ([]) |
| 12 | 0.8630 | 0.8630 | +0.0000 | 0.0000 | 0.0000 | +0.0000 | 0 ([]) |

## VERDICT: **PASS**

Both deltas are within the frozen ±0.02 band: on the shipped E1 stream, bounded-M_max=K recruit-by-eviction is indistinguishable from unbounded M in ACC and FGT. Mechanistic account: 0 evictions fired, so the bounded arm never left the shipped code path on this stream (Policy A guards only fire at a capped recruit); per-seed ACC/FGT are exactly equal to the unbounded arm (bit-identical trajectories) — the measured realization of the §2.1 'cap is inert at home' expectation. The lever binds only in overlap/open-world regimes, where its forgetting cost must be measured, not assumed away.

Raw per-seed records (docs/RETENTION.md: raw first, verdict second): `raw_<arm>_seed_<sss>.json` in this directory (crash-safe per-seed writes, resumable), aggregated in `raw_all_seeds.json`.
