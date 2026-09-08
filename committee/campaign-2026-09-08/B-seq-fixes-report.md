# B — seq fixes report (pre-GPU review M-5..M-12, M-16)

Agent B, 2026-09-08. Scope: review findings M-5, M-6, M-7, M-8, M-9, M-10, M-11, M-12, M-16.
M-9 is a documentation/budget-disclosure item (pre-run ledger note), not a code change — see
"deliberately NOT done" at the bottom. No git writes; no writes under `results/`.

## What changed per finding

### M-5 (root) — seq/stats.py `holm_correction`
`p_adj` is now the standard Holm running max of the step-down multipliers (capped at 1), so it is
monotone non-decreasing in ascending-p order and `p_adj < alpha` agrees with `reject` for every
hypothesis. `reject` semantics unchanged (step-down: reject the sorted prefix while the RAW
multiplier is < alpha, stop at the first failure). Verified the review's demonstrated case live:

```
holm_correction([0.0084, 0.0095, 0.02, 0.5, 0.7, 0.9], alpha=0.05)
-> [(0.0084, 0.0504, False), (0.0095, 0.0504, False), (0.02, 0.08, False),
    (0.5, 1.0, False), (0.7, 1.0, False), (0.9, 1.0, False)]
```

Old code gave p_adj=0.0475 for p=0.0095 (a `p_adj < alpha` consumer said "win" while step-down
said no rejections); now the running max carries 0.0504 and both decisions agree.

### M-5 (consumer) — seq/surprise_claim.py `surprise_verdict`
Win test switched from `h["p_adj"] < alpha` to `bool(h["reject"])` (+ comment). Equivalent after
the root fix, correct regardless. No existing test had to change: the §4 verdict tests assert
survives/retired outcomes, not p_adj numbers, and `test_holm_basic` pins only the `reject`
booleans (checked before editing). `tests/test_landscape.py` compares its rendering against
`holm_correction`'s own output, so it stays consistent by construction (ran it green anyway).

### M-6 — canary fingerprint (seq/surprise_claim.py + seq/gpu_harness.py)
- `seq/gpu_harness.py` `negative_control` gained an optional `cfgsig=None` parameter, THREADED
  into both arms' `sweep_then_seeds` calls (exactly how the arm cells do it), plus one docstring
  sentence. Default None keeps every other caller byte-identical (callers: seq/landscape.py
  x2, tests/test_gpu_harness.py — all pass).
- Call site in `run()` now builds `canary_cfgsig = _fingerprint({...})` with the canary's
  distinguishing constants: leg "integrity-canary", task, scale, cap/batch, recipe, sweep grid,
  seed pair [0,1], `NEGCTRL_SEED_OFFSET`, and `prizma_kw: {}` (the canary arm is the DEFAULT
  Prizma config — the empty dict is the honest payload), and passes it through.
- DISCLOSURE: `seq/gpu_harness.py` is outside my assigned file-zone list; the task item
  explicitly instructed threading the parameter through `negative_control`, which lives there.
  The edit is surgical (signature + two call forwards + docstring). If another agent owns that
  file, this is the one overlapping hunk to reconcile.
- NOTE: M-6 + M-7 both change fingerprints, so any pre-existing ledger's negctrl/gain cells will
  legitimately RECOMPUTE on the next resume instead of being reused — that is the fix working,
  not a regression.

### M-7 — gain-selection fingerprint payload (seq/surprise_claim.py)
`_train_gain_selection` gained a `recipe` kwarg (passed from `run()`) and its cfgsig payload now
includes `"recipe": recipe` — the training recipe `make_cfg` consumes. A warmup-recipe amendment
can no longer leave gain cells reusable while arm cells recompute.

### M-8 — archive before canary exit (seq/surprise_claim.py)
`archive_run` moved ABOVE the canary-exit branch: canary runs -> archive -> THEN the
`if nc is not None and not nc["pass"]` INCONCLUSIVE exit. Verdict logic unchanged. "Raw records
archived before any verdict" now holds on every path. Exercised end-to-end by the smoke run
(archive line present in output).

### M-10 — seq/blockdrift_claim.py
Deleted the dead wrong-condition `bar2 = (adapt_ci[0] <= -0.10)`. The real verdict at the bottom
(`bar1 and adapt_ci[1] <= -0.10`, INCONCLUSIVE via `adapt_ci[0]`) untouched.

### M-11 — seq/blockdrift_claim.py import + new regression test
DEVIATION from the literal instruction (disclosed): the instructed try/except dual import was
NOT added, because the file ALREADY has the working dual pattern at module level
(`sys.path.insert(0, _ROOT)` + `from seq.transformer import TFConfig, Transformer`, the same
effect as prizma_lm_claim.py's bootstrap — both invocation modes resolve it). The actual bug was
the redundant LOCAL bare import `from transformer import Transformer as _TF` inside `main()`.
Fix: deleted the local import; the WINDOW-TF leg now uses the module-level `Transformer`.
Added `tests/test_blockdrift_import.py`: `import seq.blockdrift_claim` as a package from repo
root must bind the identical `Transformer`/`TFConfig` objects from `seq.transformer`.

### M-12 — seq/analog_probe_claim.py `--force` guard
`__main__` now refuses `--seeds/--grid/--difficulties` overrides with SystemExit (exit 1, clear
message about the filename-keyed resume cache) UNLESS the new `--force` flag is passed; the
refusal fires before any training/paths are created. Verified live:
`python -m seq.analog_probe_claim --seeds 0 1` -> refusal message, exit 1.
- Default (no-override) invocations are byte-compatible: the guard only evaluates when an
  override flag is non-None.
- Checked tests/test_analog_lever.py and tests/test_analog_neuromorphic.py: NEITHER invokes the
  runner CLI (they test seq.prizma_seq levers / src.prizma directly), so no `--force` call-site
  updates were needed.

### M-16 — seq/dfrontier_claim.py `verify_frozen_against_ledger`
Deleted the dead unused `accs` collection (`accs = []` / `accs.append(rec["best"])`). The loop's
existence-check + SystemExit stays (it guards the eps recomputation below, which is unchanged).

## Tests added/updated
- tests/test_stats.py: +3 tests (no existing test changed — none pinned the old non-monotone
  p_adj numbers): `test_holm_p_adj_monotone_in_sorted_order`,
  `test_holm_p_adj_agrees_with_reject_knife_edge` (the review's p=[0.0084, 0.0095, ...] case,
  including the exact 0.0504 running-max pin), `test_holm_p_adj_agrees_with_reject_sweep`.
- tests/test_blockdrift_import.py: new file (M-11 regression, one test).
- No test anywhere else required changes.

## Verification (all from repo root, ./.venv/Scripts/python.exe)
| Command | Result | Exit |
|---|---|---|
| `-m pytest tests/test_stats.py tests/test_surprise_gate.py tests/test_dfrontier_smoke_refusal.py tests/test_analog_lever.py tests/test_analog_neuromorphic.py tests/test_blockdrift_import.py -q` | 92 passed, 1 skipped | 0 |
| `-c "import seq.stats, seq.surprise_claim, seq.blockdrift_claim, seq.dfrontier_claim, seq.analog_probe_claim; print('imports ok')"` | imports ok | 0 |
| `-m pytest tests/test_gpu_harness.py tests/test_recall_gate.py -q` (negative_control / archive neighbors) | 37 passed | 0 |
| `-m pytest tests/test_landscape.py -q` (other negative_control caller) | 15 passed | 0 |
| `PRIZMA_RESULTS=<temp> -m seq.surprise_claim --smoke` (ledger + archive redirected to temp; repo results/ untouched) | full gain-selection + 4-arm smoke green | 0 |
| `python -m seq.analog_probe_claim --seeds 0 1` (guard check) | refusal message, no training | 1 |

pytest printed summary lines normally this session.

## Deliberately NOT done
- M-9 (canary budget disclosure "2 runs ~0.2 h" vs the actual 14-cell sweep): documentation/
  pre-run-ledger note, not code; the reviewer's own recommendation is "note in the pre-run
  ledger at run time". Left to the orchestrator/integrator (I do not write results/).
- H-2 (PR-07' Holm disclosure addendum): assigned to the disclosures lane (agent D's report
  exists in this directory); I did not touch the prereg docs.
- M-12: `--steps` / `--eval-batches` remain unguarded — the review scoped the finding to
  --seeds/--grid/--difficulties and the task repeated exactly those three; extending the guard
  would be scope creep. Flagged here for the orchestrator if they want them covered.
- `python seq/blockdrift_claim.py` / `python -m seq.blockdrift_claim` end-to-end was NOT run
  (it downloads/trains the full PR-07' protocol); M-11 is covered by the package-import
  regression test plus the fact that the WINDOW-TF leg now uses the module-level import that
  the top-level bootstrap already exercises in both invocation modes.
