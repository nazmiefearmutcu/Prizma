# D — Disclosures report (review H-2 closure + M-9 disclosure), 2026-09-08

Executor: committee subagent (read-only outside its file zone; no git writes performed).
Repo: C:\Users\Kullanıcı\Prizma

## 1. Computed numbers (H-2)

Loaded from `results/blockdrift_PR-2026-09-03-07/raw.json` (READ-ONLY), computed with
`seq.stats` (t_sf upper-tail convention), full trace in
`committee/review_2026-09-08/h2_holm_recompute_OUTPUT.txt`:

- Bar 1 (retention, FGT_A, one-sample, n=5 seeds 0-4, df=4): mean = -0.189009748231,
  sd (ddof=1) = 0.056737841466, se = 0.025373934083, t = (0.05 - mean)/se = 9.419499059574,
  **p_raw1 = 3.540489e-04**, Holm-adjusted p = 3.540489e-04 (multiplier 1, step-down rank 2).
- Bar 2 (adaptation, BPC_B STREAM vs FROZEN, Welch two-sample): mean(STREAM) = 2.954381349637,
  mean(FROZEN) = 6.132161304947, delta = +3.177779955310 vs required +0.10,
  se_welch = 0.097402009141, df_welch = 5.660110836297, t = 31.598731714649,
  **p_raw2 = 6.944012e-08**, Holm-adjusted p = 1.388802e-07 (multiplier 2, step-down rank 1).
- Holm(2) at alpha=0.05: both hypotheses rejected; both RAW p are far below the binding
  first threshold alpha/2 = 0.025, so the registered-but-unimplemented Holm is a no-op and
  cannot flip the CLAIMED (PASS) outcome.
- Cross-checks: recomputed mean/CI reproduce the ledger's stored `fgt_mean`, `fgt_ci_hi`,
  `adapt_ci` exactly (difference 0.00e+00 on all three).

## 2. What was appended where

1. `docs/preregistry/2026-09-05-blockdrift-bar6.md` — appended
   `## Addendum 2026-09-08 (post-hoc, maintainer) — registered Holm implemented post-hoc; disclosure (review H-2)`
   (4 numbered points: disclosure, recomputed p's + Holm-adjusted p with real numbers,
   script/output pointer, no-side-effects statement). Placed AFTER the existing 2026-09-08
   post-run disclosure addendum.
2. `docs/preregistry/2026-09-03-surprise-gating-powered-ablation.md` — appended
   `## Addendum 2026-09-08 (pre-run, maintainer) — canary cost disclosure (review M-9)`
   (the "2 runs ~0.2 h" budget vs the named implementation's 2x(5 sweep + 2 seeds) = 14
   cells, ~+1 h on the 10-15 h estimate; no protocol change; PR-01 ledger will note the
   actual canary wall time at run time). Appended after the FROZEN footer.
3. NEW files (both inside committee/ only):
   - `committee/review_2026-09-08/h2_holm_recompute.py` (pure ASCII, read-only: the only
     `open()` is the ledger with mode "r"; no writes anywhere)
   - `committee/review_2026-09-08/h2_holm_recompute_OUTPUT.txt` (full console output, 54 lines)
   - `committee/campaign-2026-09-08/D-disclosures-report.md` (this report; dir created)

## 3. Commands + exit codes (run from repo root)

- `./.venv/Scripts/python.exe committee/review_2026-09-08/h2_holm_recompute.py` → EXIT=0
- `./.venv/Scripts/python.exe committee/review_2026-09-08/h2_holm_recompute.py > committee/review_2026-09-08/h2_holm_recompute_OUTPUT.txt 2>&1` → EXIT=0

## 4. Ledger structure notes (what the recompute had to work with)

- The ledger is ONE file (`raw.json`), not crash-safe per-seed files: `meta` (protocol
  string, pinned config repr, seeds, corpora sha256) + `arms.{STREAM, FROZEN, RESET_canary,
  WINDOW_TF}.seed_records` + a `verdict` block + `results_md` (embedded copy of RESULTS.md).
- Everything the registered bars need is present verbatim: per-seed `fgt_A`,
  `bpc_A_pre`/`bpc_A_post`, `bpc_B` for STREAM (n=5); per-seed `bpc_B` for FROZEN (n=5).
  No improvisation was needed; no alternative metric was substituted.
- Quirks (none blocking): (a) `RESET_canary.matches_STREAM` is stored as the STRING "True",
  not a JSON boolean; (b) FROZEN records duplicate STREAM's `bpc_A_pre` as `bpc_A`
  (identical floats, same seed/LR family — consistent with the arm design); (c) the runner
  computed per-seed adaptation diffs paired (STREAM_i - FROZEN_i) but built se/df from the
  two-sample Welch forms — my recompute used the registered two-sample Welch form and
  reproduced the ledger's stored CI exactly, so the two conventions agree numerically here;
  (d) WINDOW_TF ran 1 seed (descriptive only, gates nothing — per the registered partial
  order).

## 5. results/ integrity

Nothing under `results/` was modified or created; `results/blockdrift_PR-2026-09-03-07/`
still contains exactly RESULTS.md, corpora/, raw.json with their original timestamps
(Sep 7 01:23-01:30). No git commands were run (orchestrator integrates).
