# Lane 1 — tissue multi-timescale (fast/slow) cascade — report

**Lane:** 1 (mechanism; campaign 2026-09-11 CONTRACT.md §"Lane 1")
**Status:** DONE. All deliverables implemented; all required test files green; smokes verified.
**Coordinator actions pending:** review + commit; run the default-vs-cascade `manyblock_probe.py`
exploratory comparison (contract §Coordinator); decide the bias question below before reading the
probe as a clean cascade signal.

## What changed (files, exact scope)

All changes are behind the guarded default-OFF lever; nothing was tuned (kappa/delta defaults
0.05/0.10 implemented verbatim).

### `seq/fusion_probe.py` — mechanism home
- New constants `CASCADE_TARGETS=("off","tissue")`, `CASCADE_KAPPA_DEFAULT=0.05`,
  `CASCADE_DELTA_DEFAULT=0.10`.
- `PCExpertHead.__init__(..., cascade=False, cascade_kappa, cascade_delta)`:
  - `cascade=True` adds zero-init `Wenc_fast`/`Wdec_fast` Parameters (`torch.zeros_like` — no RNG
    consumption, so the seed stream is preserved) and freezes the slow `Wenc.weight/Wdec.weight`
    (`requires_grad_(False)`: they can move ONLY via the fold).
  - `forward`: ON = `F.linear(tanh(F.linear(h, Wenc.w+Wenc_fast, Wenc.bias)), Wdec.w+Wdec_fast,
    Wdec.bias)`; OFF = the exact pre-lever `Wdec(tanh(Wenc(h)))`.
  - `fast_parameters()` (the optimizer set), `consolidate()` (the frozen-order fold
    `W += kappa*W_fast` using the PRE-decay fast value, then `W_fast *= (1-delta)`),
    `fast_norms()` (per-head L2 + max|.|).
- `FusionLM(...)`/`build_model(...)` accept and thread the cascade kwargs; `E==0` (`_PlainWrap`)
  untouched.
- `expert_cascade_kwargs(model)` — fresh heads inherit the lever (eviction / forced-recruit sites).
- `cascade_diagnostics(model)` — per-slot `fast_l2`, `fast_absmax`, `n_batches` + totals; returns
  `None` when off (OFF ledgers keep their exact field set).
- `_expert_train`: when the head has a fast component the optimizer receives ONLY it and
  `head.consolidate()` runs immediately after `opt.step()` (before the post-update floor
  recompute); OFF path keeps `head.parameters()` and operation order. Uses `getattr` so duck-typed
  test heads without the new API stay valid (this fixed a regression caught in
  `tests/test_floorfreeze_runner.py`, a shared-plumbing test).
- `ledger_snapshot` / `ledger_snapshot3`: append a `cascade` key ONLY when on.

### `seq/prizma_lm_claim.py` — flagship runner
- `CASCADE_TARGETS/KAPPA/DELTA` constants + pure `validate_cascade(target, kappa, delta)` guard
  (target domain + `[0,1]` range; SystemExit on violation).
- Parser: `--cascade-target {off,tissue}` (default off), `--cascade-kappa` (0.05),
  `--cascade-delta` (0.10); `main()` threads them into `run()`.
- `run()`: validates; records `meta["cascade"]` and `report["cascade"]` when ON (absent when off);
  adds the three keys to BOTH cell and lr-selection fingerprints (treated runs never resume from
  untreated cells); fusion-family LR selection uses the cascade model when ON; threads the flags
  to `run_routed`.
- `run_routed()`: builds the cascade model, records `cascade_after_A/B/C` per-block diagnostics
  (when on) and a `ledger.cascade` block via `ledger_snapshot_pr08`.
- Eviction / forced-recruit re-init sites now pass `**fp.expert_cascade_kwargs(model)` (fresh
  head inherits the lever; the pinned fresh-head seed formula is unchanged — zeros_like consumes
  no RNG).
- `ledger_snapshot_pr08` appends `cascade` only when on (shared with floorfreeze/domainexc/
  trunklr runners — inert for them, verified by their tests).

### `seq/manyblock_claim.py` — many-block runner
- Aliases the constants/guard from `plc`; `run()`/`run_cell()` accept and thread the flags;
  per-block `block_<TAG>.cascade` diagnostics; `meta`/`report`/fingerprint carry the lever when
  set; parser + `main()` wiring.

### `seq/manyblock_probe.py` — exploratory n=2 harness (now CLI)
- New argparse: `--cascade-target/-kappa/-delta`, `--smoke`; `--help` shows them.
- `run_seed(...)` builds the cascade model and records per-block
  `traj["blocks"]["post_<TAG>"]["cascade"]` + top-level probe meta when on.
- Output files never clobber: off → `probe.json` (the original behavior), tissue →
  `probe_cascade.json`, `--smoke` → `probe_smoke.json` (8 segments/block, seed 0).
- Default no-arg invocation is byte-identical to the 2026-09-09 probe (off path + same file).

### `tests/test_tissue_cascade.py` (NEW, 14 tests)
(a) OFF == no-flag exact equality on a tiny fusion training (identical construction state_dicts,
identical post-training params/`bpc`/ledger snapshot, no `cascade` key anywhere; two OFF runs also
compared); (b) fold equations + `forward == W+F` + optimizer-receives-only-W_fast + slow-W-moves-
only-by-fold + requires-grad structure + kappa=delta fold invariance; (c) parser defaults/
roundtrip/guard + fingerprint sensitivity + `inspect.signature` threading across all four files +
source pins; (d) diagnostics only on the ON path (`ledger_snapshot` and `ledger_snapshot_pr08`) +
Policy-A eviction re-init inherits the lever.

## Exact test results (venv `.venv\Scripts\python.exe`, 2026-09-10/11 local)

| command | result |
|---|---|
| `pytest tests/test_fused.py tests/test_prizma_lm_runner.py tests/test_manyblock_runner.py tests/test_pr11_repaired_flagship.py tests/test_pr13_exclusion_flagship.py tests/test_windowtf_manyblock.py tests/test_tissue_cascade.py -q` | **97 passed, 5 skipped** (4.1 s) |
| `pytest tests/test_tissue_cascade.py -q` (new file alone) | **14 passed** (3.1 s) |
| neighbor shared-plumbing files `tests/test_floorfreeze_runner.py tests/test_domainexc_runner.py tests/test_trunklr_runner.py` | **44 passed** (part of a 58-passed joint run) |
| artifact readers `tests/test_damage_gap_replication.py tests/test_boundary_damage.py tests/test_recovery_attribution.py` | **20 passed** |

Smoke evidence (registered ledgers untouched; claim/probe smokes wrote only to temp `--out` paths):
- `python seq/manyblock_probe.py --help` → shows `--cascade-target {off,tissue}`,
  `--cascade-kappa`, `--cascade-delta`, `--smoke`.
- `python seq/manyblock_probe.py --smoke` (off) → completes 1.7 s; the in-repo artifact
  `results/exploratory/manyblock_probe_2026-09-09/probe_smoke.json` is this OFF smoke (no cascade
  key anywhere; the on-disk file is new/untracked — coordinator may commit or discard).
  Copies: `%TEMP%\opencode\lane1\probe_smoke_off.json` and `probe_smoke_cascade.json`.
- `python seq/manyblock_probe.py --smoke --cascade-target tissue` → completes 1.7 s; JSON carries
  top-level `cascade{target,kappa,delta}` + per-block `post_A..post_E.cascade` with per-slot
  fast norms (post_E total_fast_l2 = 0.7409, max|F| = 0.0111; only the trained slot nonzero).
- `python seq/prizma_lm_claim.py --smoke --out <temp>` (off) and
  `--smoke --cascade-target tissue --out <temp>` → both complete (~25 s each); ON cell carries
  `cascade_after_A/B/C` + `ledger.cascade` (post-C total_fast_l2 = 0.6804) + `meta.cascade`;
  OFF cells carry none. Two independent OFF runs: **all scientific cell fields identical**
  (excluding `wall_s`), `cfgsig` equal; OFF vs ON: SHARED-HEAD science identical, only `cfgsig`
  differs (by design — the lever is in every fingerprint).
- `python seq/manyblock_claim.py --smoke --out <temp>` (off) and `--cascade-target tissue` →
  both complete; ON PLAIN and EX cells carry `block_A..block_E.cascade`; OFF none; cross-arm
  canary PASS; `report.cascade` + `meta.cascade` recorded when on.
- Side effect of the smoke runs (standard `archive_run` retention): 6 new untracked retention
  archives at `results/runs/prizma-lm-PR-2026-09-03-08-20260910T2239..T2242*.json` and
  `results/runs/manyblock-PR-2026-09-03-14-20260910T2242..T2243*.json` — coordinator's call
  whether to keep them.

## Expert optimizer weight_decay (asked explicitly)

`_expert_train` builds `torch.optim.AdamW(params, lr=lr)` with **no explicit `weight_decay`**;
torch 2.14.0's default is **`weight_decay=0.01`** (decoupled). Verified:
`inspect.signature(torch.optim.AdamW.__init__).parameters["weight_decay"].default == 0.01`.
Consequence for the probe: even at `kappa == delta` the cascade is **not** a pure
reparameterization — the decoupled decay shrinks the fast component
(`F <- (1 - lr*0.01) F` inside the optimizer) before the fold, and the decay is applied to `F`
only, not to `W+F`. This is a property of the existing code, not of the lever; the probe's
default pair (kappa=0.05 < delta=0.10) is genuinely transient regardless.

## Risks / uncertainty (for the coordinator)

1. **Biases are excluded under the frozen "ONLY the W_fast parameters" clause.** `PCExpertHead`
   also has `Wenc.bias`/`Wdec.bias` (zero-init, currently trainable in OFF). In cascade mode the
   optimizer receives exactly `[Wenc_fast, Wdec_fast]`, so the biases stay at zero. This is the
   literal frozen semantics ("ONLY the `W_fast` parameters"), but it is a real ON-vs-OFF
   difference beyond the cascade itself and a candidate confound for the probe (e.g.
   `Wdec.bias` is a vocab-shaped offset). If the coordinator intended fast copies/or continued
   bias training, it is a small extension — **flagged, not silently changed**. Pinned by
   `test_cascade_requires_grad_structure_and_optimizer_only_sees_fast`.
2. **Slow-W `requires_grad_(False)`** enforces "optimizer receives ONLY W_fast" structurally.
   The frozen text doesn't mention requires_grad; the numeric update path (`W += kappa*F` under
   `no_grad`) is identical either way.
3. **Fingerprint keys are unconditional** (following the PR-11/13/18 convention): the three
   cascade keys are in every cell + lr-selection payload, so pre-existing ledgers refuse resume
   at the same cell key (loud foreign-fingerprint refusal, never a silent resume). Point cascade
   runs at a fresh `--ledger-dir` (as the contract implies).
4. **`SharedHeadLM` (SHARED-HEAD arm) is not cascaded** — decided deliberately: it is the
   capacity control and has its own training path (`train_stream_shared`); cascading it would
   change the control and expand scope beyond the frozen "tissue expert heads" semantics.
   Verified: SHARED-HEAD science fields identical between off/on smoke runs.
5. **Per-block diagnostics** are recorded as `cascade_after_A/B/C` (plc), `block_<TAG>.cascade`
   (mbc), `post_<TAG>.cascade` (probe), plus `ledger.cascade` snapshots — all present only when
   on, so OFF record shapes are untouched.
6. `manyblock_probe.py` gained a CLI/`--smoke`; the default no-arg run keeps the old output path
   and behavior. Cascade and smoke outputs use distinct filenames by design.
7. Cosmetic: one em-dash in the probe `--help` string was replaced with ASCII (console codepage
   mojibake).

No change breaks the byte-identity requirement: the OFF path was verified at unit level (tiny
fusion training) and at smoke level (two independent OFF runs, all scientific fields identical).
