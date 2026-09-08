# E — PR-09 floor-freeze routing repair: implementation report (lane: CLAIM)

**Registry:** PR-2026-09-03-09 · **Date:** 2026-09-09 · **Implementer:** GLM 5.3 Flash subagent
(Zone E of the 4-parallel-agent wave). Implements the FROZEN protocol
`docs/preregistry/2026-09-09-floorfreeze-routing-repair.md` verbatim. **No powered run was
executed by this agent** — `--powered-cpu` is the orchestrator's official run. No git writes;
nothing written under repo `results/` (smoke went to a temp zone).

## 1. What shipped (file zone respected exactly)

| File | Change |
|---|---|
| `seq/floorfreeze_claim.py` | NEW — the PR-09 runner (562 lines) |
| `seq/fusion_probe.py` | ONE guarded insertion in `_expert_train` |
| `seq/prizma_lm_claim.py` | ONE guarded insertion in `route_pr08` |
| `tests/test_floorfreeze_runner.py` | NEW — 22 tests (pure layer + behavioral off-identity pins) |
| `tests/test_prizma_lm_runner.py` | UNTOUCHED (off-identity pin placed in the new file instead — no change was needed there) |

## 2. The two guarded lever insertions — exact points

**(a) `seq/fusion_probe.py` `_expert_train` — guard at lines 253–267.**
`freeze = getattr(model, "floor_freeze", None)` sits at **line 259**, and
`if not (freeze is not None and slot in freeze):` at **line 260** wraps BOTH floor-update
branches (first-calibration `if model.mu[slot] > 1e8:` at line 261 + the FLOOR_EMA `else`
at lines 264–267). Everything else in the function runs unchanged and OUTSIDE the guard, in
the original order: private-AdamW expert step (`opt.step()`, line 245), `n_batches`/`n_segments`
bookkeeping (lines 246–247), post-update surprise `r` and `ce_sum` (lines 249–251). The only
new unconditional operation is one `getattr` (returns `None` when the attribute is absent;
the condition then short-circuits without touching the set).

**(b) `seq/prizma_lm_claim.py` `route_pr08` — guarded discard at lines 507–509.**
Inside the Policy-A eviction re-init block, immediately AFTER
`model.mu[victim], model.var[victim] = 1e9, 1.0` (line 506) and BEFORE
`model.n_batches[victim] = 0` (line 510):
`fz = getattr(model, "floor_freeze", None)` / `if fz is not None: fz.discard(int(victim))`.

### DISCLOSED DEVIATION from the task snippet
The task snippet said `fz.pop(int(victim), None)`. The registered freeze is **a plain Python
set** of slot ints (task's own arm spec: `model.floor_freeze = {slots committed at A end}`),
and `set.pop(x, None)` raises `TypeError` (sets take no pop argument). The implemented
`fz.discard(int(victim))` is the exact remove-if-present semantics for a set, never raises,
and is pinned behaviorally (`test_route_pr08_eviction_pops_the_victim_from_the_freeze_set`,
`test_route_pr08_discard_of_an_absent_slot_is_a_noop`). The first behavioral test run caught
this before any run relied on it.

## 3. Runner architecture (`seq/floorfreeze_claim.py`)

- **Reuse BY IMPORT, no protocol constants copy-pasted:** `import seq.prizma_lm_claim as plc`;
  stream/slices = `ffc.pin_slices is plc.pin_slices` (true alias, pinned by test), tissue
  constants are the plc objects (`E_POOL is plc.E_POOL == 4`, `M_MAX==4`,
  `FREEZE_MIN_SEEN==300`, `Z_NOVEL==5.0`, `H_SMALL==64`, `FLOOR_EMA==0.05`,
  `FRESH_HEAD_SEED_BASE`, `B3_WINDOW_BATCHES==20`), cell plumbing = `plc.train_pr08`,
  `plc.route_pr08`, `plc._fresh_ledger`, `plc.ledger_snapshot_pr08`,
  `plc._onesample_margin`, `plc._welch_margin`, `plc._bar_status`,
  `plc.budget_projection`, `plc.needs_cuda` (TRUE alias, `ffc.needs_cuda is plc.needs_cuda`,
  pinned by test), `plc.CLAIM_SEEDS`, plus PR-08's `C_RANGE_REPAIR_NOTE`/`C_RET_PROSE_NOTE`
  carried into meta (PR-09 inherits those disclosures with the protocol). Bar math reuses
  plc's UPPER-TAIL t convention; Holm via `seq.stats.holm_correction`.
- **REGISTRY_ID** `PR-2026-09-03-09`; **LEDDIR** `floorfreeze_PR-2026-09-03-09`;
  smoke/powered ledgers separate (`smoke.json` / `powered.json`); smoke pointed at the
  powered ledger REFUSED unless `--force-smoke-path` (plc refusal pattern mirrored on the
  PR-09 led dir).
- **LR:** `LR_FROZEN = 3e-3` for BOTH arms, doc §2 inheritance. There is **NO lr-selection
  leg**; meta records `lr_rule` + `lr_inheritance` pointing at PR-08's
  `lr_selection.fusion` (verified: PR-08's powered cells carry `lr=0.003` for the fusion
  family — read from `results/prizma_lm_PR-2026-09-03-08/powered.json` before coding).
- **Arms:** `OFF` (freeze attribute never set) and `FROZEN-FLOORS`. `run_cell` mirrors
  `plc.run_routed`'s PRIM path operation-for-operation (same phase/eval order, same bpc
  keys, same boundary window, same B4 quantity); the treatment adds ONLY: after block-A
  training + the `bpc_A_preB` eval, `model.floor_freeze = {committed slots}` and an audit
  record `rec["floor_freeze"] = {slots, pinned_mu, pinned_var, note}`; at cell end,
  `slots_at_end` is appended. During B/C the freeze works exclusively through the guarded
  branches — zero per-batch runner code.
- **Fingerprints:** plc payload pattern (`registry`, `leg="claim"`, `arm`, `seed`,
  `lr=3e-3`, `smoke`, `vocab`, `slices`, `stream_lengths`, `bars`, `tissue` with a
  `floor_freeze` boolean inside). Foreign-fingerprint resume = hard `SystemExit` refuse
  (plc pattern). Crash-safe `_save` after every cell.
- **CANARY (doc §4, powered only):** after the OFF arm's 5 cells — i.e. between the arms —
  `run_canary` loads `results/prizma_lm_PR-2026-09-03-08/powered.json` (read-only) and
  requires every science field of `claim.OFF.s{seed}` == `claim.PRIM-LM.s{seed}` EXACTLY.
  Ignore set: `{"wall_s", "arm", "cellkey", "cfgsig", "complete", "forced_placement"}`;
  everything else (all 7 `bpc_*` floats, the nested `ledger` dict, `a_expert`, `fgt_A_full`,
  `b_degradation_B_postC`, plus config/E/seed/lr) compared with exact `==`. Missing PR-08
  ledger or any mismatch → loud `SystemExit` ABORT before the treatment arm.
- **Verdict (pure, unit-tested without torch):** P1 exact means (FROZEN-FLOORS frac mean
  ≥ 0.5; OFF frac mean recorded as control) — no CI, the PR-08 B3 form; G1 one-sample
  direction "below" (CI upper ≤ 0.05 AND Holm p < 0.05; CI lower > 0.05 → FAIL; straddle →
  INCONCLUSIVE); G2 Welch direction "above" (advantage = mean(PR-08 FROZEN-CHECKPOINT
  `bpc_Cret_postC`, REUSED from its powered ledger) − mean(FROZEN-FLOORS) ≥ 0.10, CI lower
  ≥ 0.10; CI upper < 0.10 → FAIL; straddle → INCONCLUSIVE). Holm family = [G1, G2].
  Overall precedence: any guard INCONCLUSIVE → INCONCLUSIVE (§5 n=10 pre-authorization
  echoed) > CLAIMED (P1+G1+G2 all PASS) > NEGATIVE-GUARD (P1 PASS + guard FAIL, names the
  broken guard + floor-scheduling lead) > NEGATIVE (P1 FAIL; trunk-lr scheduling named as
  the next prereg'd candidate). B4 per arm, churn (recruits/evictions/vetoed) per arm and
  per-seed fracs for both arms reported, never gated.
- **Retention:** `archive_run(res, label="floorfreeze-PR-2026-09-03-09")` BEFORE any verdict
  write (plc ordering). `--powered` refuses without CUDA (own `require_cuda`, plc byte-style)
  EXCEPT `--powered-cpu` (reuses `plc.needs_cuda`; meta gains `powered_cpu` +
  `compute_fallback_note` pointing at PR-09 doc §6 and the PR-08 addendum #2 precedent).
- **CLI:** `--smoke / --powered / --powered-cpu` mutually exclusive (exactly one required),
  `--out`, `--force-smoke-path`; unknown flags exit non-zero before anything runs.

## 4. Off-identity argument (default-off is provably inert)

1. **Static:** attribute absent → `getattr` returns `None` → (a) the guard condition
   `not (None is not None and ...)` is `False` without ever evaluating `slot in freeze`, so
   both floor-update branches execute with the identical operations in the identical order;
   (b) in `route_pr08` the `if fz is not None` is `False` → no-op. The only added
   unconditional work is one `getattr` call per `_expert_train` / per eviction — no state,
   no RNG, no tensor op.
2. **Behavioral pins (real torch, tiny tensors — tests run in the suite):**
   - `test_expert_train_off_identity_no_attribute_floors_still_update`: no attribute → floor
     first-calibration runs, `n_batches/n_segments/ce_sum` advance, expert weights move.
   - `test_expert_train_freeze_pins_floors_but_expert_still_trains`: `floor_freeze={0}` →
     `mu/var` EXACTLY untouched while the expert optimizer step still moves weights (only
     floors are frozen) and a non-frozen slot keeps updating.
   - `test_route_pr08_off_identity_no_attribute_no_crash_no_attribute_created`: full-pool
     eviction with no attribute → identical eviction ledger, and the guarded branch never
     conjures the attribute.
3. **End-to-end:** the smoke's two arms produced bit-equal `bpc_A_preB`
   (4.941736105231895 both — fail-loud assert in `_smoke_checks`), and the powered canary
   will prove OFF == PR-08 bit-identically against the stored ledger before the treatment
   arm runs (dry-run verified, §6).
4. **Smoke-proven mechanism (guard demonstrably FIRES when the attribute is present):**
   OFF's floor drifted during B/C (final mu 3.9472 / sigma 0.5348) while FROZEN-FLOORS'
   final floor equals the pinned A-end values (mu 4.1182 / sigma 0.4169) through the same
   6 post-A training batches — `_smoke_checks` verifies pinned-floor equality fail-loud.

## 5. Verification commands + exit codes (all run from repo root)

| Command | Result |
|---|---|
| `rm -rf "$LOCALAPPDATA/Temp/pr09_zone"` then `PRIZMA_RESULTS="$LOCALAPPDATA/Temp/pr09_zone" ./.venv/Scripts/python.exe -m seq.floorfreeze_claim --smoke` | **EXIT=0** — both arms complete, SMOKE banner, frozen-slots audit (`slots@A_end=[0] slots@end=[0] pinned_mu={'0': 4.118}`), A-phase bit-identity True, pinned floors verified through B/C `[0]`, raw archive written in the temp zone |
| `./.venv/Scripts/python.exe -m pytest tests/test_floorfreeze_runner.py tests/test_prizma_lm_runner.py -q` | **EXIT=0** — 46 passed (22 new + 24 existing) |
| `./.venv/Scripts/python.exe -m pytest tests/test_prizma_lm_runner.py -q` | **EXIT=0** — 24 passed (untouched suite) |
| `./.venv/Scripts/python.exe -m pytest tests/ -q -p no:cacheprovider` | **EXIT=0** — **435 passed, 10 skipped** in 7m30s (baseline 413P/10S + 22 new; zero regressions from the two shared-module insertions) |
| Canary dry-run (inline, read-only): PR-08 records fed verbatim as OFF cells → `run_canary` | CANARY PASS (5/5 cells); a 1-ULP `bpc_A_postC` drift → ABORT fires; PR-08 ledger missing (temp root) → ABORT fires |
| `--powered-cpu` | **NOT RUN** (the orchestrator's official run, per instructions) |

pytest on this box prints no reliable summary tail in some invocations; EXIT codes were read
via `echo "EXIT=${PIPESTATUS[0]}"` throughout (all shown above are real exit codes).

## 6. Honest disclosures / deliberately not done

1. **`discard` vs `pop(x, None)`** — see §2; behavioral deviation from the task snippet,
   required by the set type the task itself registered. Test-caught before any run.
2. **OFF cell's `config` field** is labeled `"PRIM-LM"` (treatment: `"PRIM-LM+floor-freeze"`)
   — mirroring plc's convention where `config` = model config and `arm` = experiment arm.
   This keeps the canary exact on the `config` field without widening the ignore set
   (`config` is NOT in the registered ignore list, so it is compared — and matches).
3. **Canary "science fields" = everything except the ignore set**, which is slightly
   stronger than the enumerated minimum (also compares `config`/`E`/`seed`/`lr`). These
   match naturally: PR-08's fusion-family lr is 0.003 (verified from its powered ledger),
   E=4, same seeds. The PR-08 `FROZEN-CHECKPOINT` cells have no `E` key — irrelevant: the
   canary only compares OFF vs PRIM-LM, and G2 reads only `bpc_Cret_postC` from them.
4. **`run_cell` drops plc's `frozen_after_A`/`forced_c` dead branches** (PR-09 arms are
   PRIM-family only; the frozen-B backbone snapshot and the forced-C placement machinery
   are unreachable in PR-09 and were not copied dead). The OFF path's executed operations
   are otherwise identical to `plc.run_routed`'s PRIM path in the same order — proven at
   run time by the powered canary.
5. **The off-identity behavioral tests use real torch** (tiny 4-dim tensors, no training
   loop, ~4s for all five): the task preferred a behavioral pin over source-grepping and
   asked to disclose the choice; the pure-layer tests remain torch-free and <1s.
6. **`needs_cuda` is a true alias** (`ffc.needs_cuda is plc.needs_cuda`), pinned by test —
   the task's "import and reuse" contract, not a fork.
7. **Not done on purpose:** any `--powered`/`--powered-cpu` execution; any write under repo
   `results/`; any change outside the declared file zone (notably `tests/test_prizma_lm_runner.py`
   stayed byte-identical — the off-identity pin lives in the new test file); any git command.
8. **Canary determinism caveat (registered risk, not new):** bit-identity of OFF vs PR-08
   assumes the same CPU/thread environment as the PR-08 run (BENCH_THREADS default 8, same
   box). If the official `--powered-cpu` run executes under a different thread count, torch
   CPU reduction order can differ and the canary will ABORT loudly — per doc §4 that is the
   registered fail-loud behavior (investigate, then re-run). Nothing in the runner relaxes
   this.
