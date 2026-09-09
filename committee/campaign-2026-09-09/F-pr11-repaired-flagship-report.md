# F — PR-2026-09-03-11 REPAIRED FLAGSHIP: implementation report (implementer lane F)

**Date:** 2026-09-09 · **Lane:** implementer (no git writes — orchestrator integrates)
**Frozen protocol:** `docs/preregistry/2026-09-09-repaired-flagship.md` (untouched, verbatim)
**Scope shipped:** the PR-11 repaired-flagship mode inside the PR-08 runner — PR-08 protocol
VERBATIM + ONE lever (block-C backbone lr 7.5e-4 = 3e-3 × 0.25 in every arm that trains a
backbone on C), guarded default-off, with fail-loud frozen-arm canaries.

## 1. Files touched (exact insertion points, file:line as of this report)

**`seq/fusion_probe.py`** (guarded addition ONLY):
- `train_stream_shared(...)` gains keyword-only `backbone_lr=None` (line 736); the BACKBONE
  AdamW alone uses `lr=lr if backbone_lr is None else backbone_lr` (lines 747-748) — the
  exact guarded pattern `train_pr08` already carries since PR-10. Shared-head optimizer keeps
  `lr`. With None the function is byte-identical (behaviorally pinned, §4).

**`seq/prizma_lm_claim.py`** (guarded additions ONLY; file now 1298 lines):
- Module docstring PR-11 section: lines 85-106 (mode, lever, canaries, honest ordering note).
- `_default_results_path(smoke, ledger_dir=None)`: line 198-201 (None ⇒ LEDDIR, byte-identical).
- `resolve_results_path(..., ledger_dir=None)`: line 204-214 (refusal reference threads ledger_dir).
- PURE helper `_backbone_lr_for_c(trunk_lr_c, frozen_after_A)`: lines 687-695 (truth table:
  None when lever off OR frozen_after_A; used by run_routed — load-bearing, source-pinned by a test).
- `run_routed(..., trunk_lr_c=None)`: line 697-699; C-phase conditional kwarg + audit key:
  lines 777-786. The `train_pr08` C call passes `backbone_lr` ONLY when
  `_backbone_lr_for_c(...) is not None`; otherwise the call is parameter-identical to PR-08
  (empty `**c_kwargs`) so FROZEN-TRUNK cells stay byte-comparable canaries. When applied:
  `rec["backbone_lr_c_applied"] = trunk_lr_c` (line 783).
- `run_shared(..., trunk_lr_c=None)`: line 805; C-phase call only: lines 827-831
  (`fp.train_stream_shared(..., backbone_lr=trunk_lr_c)` when ON; parameter-identical call
  when OFF); same audit key (line 829).
- `run_frozen_checkpoint`: **no change at all** (no C training).
- `_pr11_powered_path()`: lines 874-878 — always LEDDIR/`powered.json` (never ledger_dir):
  the canary reference is the untouched PR-08 powered ledger, read-only.
- `_pr11_canary(res, seeds, arm)`: lines 881-925 — fail-loud; comparator + ignore set REUSED
  from `seq/floorfreeze_claim.py` (`ffc.canary_mismatches`, `ffc.CANARY_IGNORE`; trunklr
  aliases the same objects). Missing PR-08 ledger, missing cell, or any science-field
  mismatch ⇒ SystemExit ABORT. Lazy `import seq.floorfreeze_claim` inside the function
  (ffc imports this module at its own module level — a top-level import would be circular).
- `run(..., trunk_lr_c=None, ledger_dir=None, provenance=None)`: line 928-929;
  path via ledger_dir: 939-940; `res["meta"]["pr11_repair"] = provenance` when provenance is
  not None: 1033-1034; lr-selection fingerprint gains `"trunk_lr_c": trunk_lr_c`: 1057;
  claim-cell fingerprint gains the key: 1127; arm dispatch threads trunk_lr_c
  (SHARED-HEAD 1142; routed arms 1143-1150, FROZEN-TRUNK nulled by the helper); canary gate
  `if not smoke and trunk_lr_c is not None and arm in ("FROZEN-TRUNK", "FROZEN-CHECKPOINT")`:
  1168-1175, stored as `res["canary.<ARM>"]`; report gains `"trunk_lr_c"` when set: 1190-1191.
- `_build_parser`: `--trunk-lr-c` (float, default None) line 1278-1282; `--ledger-dir`
  (default None) line 1283-1286. `main` threads both: line 1292-1294.

**`tests/test_pr11_repaired_flagship.py`** (NEW, 14 tests, 211 lines).
**`tests/test_prizma_lm_runner.py`** (ADD-ONLY): section 8 at line 335-344, one test
(`test_pr11_flags_default_to_none_so_pr08_defaults_are_unchanged`); no existing test modified.

Nothing else touched. The frozen PR-11 doc, `seq/trunklr_claim.py`, `seq/floorfreeze_claim.py`,
`results/` — all read-only or untouched. No git commands run.

## 2. Canary gating order — and its honest limitation

ARMS order is `(PRIM-LM, FROZEN-TRUNK, SHARED-HEAD, FROZEN-CHECKPOINT, FORCED-RECRUIT)`.
Gates (claim mode + lever ON only; a plain PR-08 rerun runs NO canary):

1. PRIM-LM runs FIRST — **before any gate** (see limitation below).
2. After FROZEN-TRUNK's 5 cells: canary #1 vs PR-08 `claim.FROZEN-TRUNK.s{0..4}` — gates
   SHARED-HEAD.
3. After FROZEN-CHECKPOINT's 5 cells: canary #2 vs PR-08 `claim.FROZEN-CHECKPOINT.s{0..4}` —
   gates FORCED-RECRUIT.

**Honest limitation (also stated in the module docstring, the abort message, and the code
comment):** PRIM-LM is a TREATED arm yet runs before the first canary gate. The FROZEN
canaries prove the lever's blast radius (a frozen arm has no C backbone step, so any
mismatch means the lever leaks outside the C backbone step), but they fire AFTER PRIM's 5
cells are already on disk. If either canary fails, the run aborts loudly and the already-run
PRIM cells are quarantine-suspect — they must be discarded, not cited. This ordering is
inherited from the PR-08 ARMS tuple, which the prereg inherits verbatim; changing arm order
was NOT allowed (protocol VERBATIM), so the limitation is documented instead of engineered
away.

Canary schema-cleanliness is enforced by DESIGN: `backbone_lr_c_applied` is deliberately NOT
in `ffc.CANARY_IGNORE` (pinned by test), so a FROZEN cell that ever carried the audit key
would fail the comparator loudly. FROZEN cells carry no audit key by construction (the pure
helper + run_shared's conditional never set it when the backbone step is absent).
`cfgsig` stays ignored: PR-11 recomputes it (new fingerprint key) but the frozen SCIENCE must
still compare clean.

## 3. Verification — commands + exit codes (all run from repo root)

| Command | Exit | Result |
|---|---|---|
| `./.venv/Scripts/python.exe -m pytest tests/test_prizma_lm_runner.py tests/test_pr11_repaired_flagship.py tests/test_trunklr_runner.py tests/test_floorfreeze_runner.py -q` | 0 | 72 passed (pytest DID print a summary here) |
| `./.venv/Scripts/python.exe -m pytest -q` | 0 | **461 passed, 10 skipped** (244.97s) — baseline 446 + 15 new (14 new-file + 1 ADD-only); 0 failures |
| `rm -rf "$LOCALAPPDATA/Temp/pr11_zone"` then `PRIZMA_RESULTS="$LOCALAPPDATA/Temp/pr11_zone" ./.venv/Scripts/python.exe -m seq.prizma_lm_claim --smoke --trunk-lr-c 0.0075 --ledger-dir prizma_lm_PR-2026-09-03-11` | 0 | all 5 arms completed; ledger at `<temp>/prizma_lm_PR-2026-09-03-11/smoke.json` |
| plain `--smoke` (no flags) into a second temp zone + read-only comparison vs `results/prizma_lm_PR-2026-09-03-08/smoke.json` | 0 | see below |

Smoke-with-lever evidence (verified programmatically against the temp ledger):
- `claim.PRIM-LM.s0`, `claim.FORCED-RECRUIT.s0`, `claim.SHARED-HEAD.s0` carry
  `backbone_lr_c_applied == 0.0075`; `claim.FROZEN-TRUNK.s0` and `claim.FROZEN-CHECKPOINT.s0`
  do NOT carry the key. ✓ (the required check)
- No canary keys in smoke (correct: canaries are claim-mode-only).
- `report.trunk_lr_c == 0.0075`; `meta.pr11_repair` absent (correct: provenance not passed).
- Lever FIRES: treated PRIM `bpc_B_postC` 5.2494 vs PR-08's 5.0411, while PRIM's `bpc_A_preB`
  and `bpc_B_postB` are bit-identical to PR-08's (divergence begins at C only — prereg §3
  canary (2) semantics, demonstrated at smoke scale).
- Treated FROZEN-TRUNK / FROZEN-CHECKPOINT cells compare with EMPTY mismatch lists against
  the PR-08 smoke records (`ffc.canary_mismatches`) — the canary premise holds under the lever.
- Plain (no-flag) smoke: no audit keys, no canaries, no `report.trunk_lr_c`, and its
  FROZEN-TRUNK/FROZEN-CHECKPOINT science fields are bit-identical to the repo's PR-08 smoke
  ledger (cross-run determinism re-proven; the no-flag path is unchanged end-to-end).
- Temp zones deleted after verification; repo `results/` never written by me.

`--powered-cpu` was NOT run (the orchestrator's official run).

## 4. Disclosures (honesty over polish)

1. **Fingerprint key side effect (deliberate, fail-loud):** the cell AND lr-selection
   fingerprint payloads now always include `"trunk_lr_c"` (None or the dose), per spec
   ("a treated rerun must never resume from untreated cells"). Because
   `config_fingerprint` hashes the whole payload, this changes EVERY cfgsig the modified code
   produces — including untreated ones. Consequence: re-running the modified code against a
   PRE-PR-11 ledger (e.g. the existing PR-08 powered.json) refuses resume with the standard
   loud foreign-fingerprint SystemExit instead of silently reusing those cells. The PR-11 run
   uses its own `--ledger-dir prizma_lm_PR-2026-09-03-11`, so this never bites the official
   run; the PR-08 ledger is only ever READ (canaries). `cfgsig` is in `CANARY_IGNORE`, so the
   canaries are unaffected.
2. **Fingerprint test approach (disclosed as the task requires):** `plc._fp` imports
   `seq.gpu_harness` → torch. Torch IS importable in this venv (CPU wheel; the behavioral
   test needs it anyway), so the test exercises the REAL fingerprint function
   (`test_fingerprint_is_sensitive_to_trunk_lr_c`: None vs 7.5e-4 give different sigs for
   both payload shapes, and both differ from the pre-PR-11 no-key payload), AND pins the
   `'"trunk_lr_c": trunk_lr_c'` key textually in `run()`'s source via `inspect.getsource`
   (count == 2: lr-selection + claim constructions) — the task's specified fallback, kept as
   a belt-and-suspenders regression pin.
3. **`provenance` param:** implemented (`meta["pr11_repair"]` when not None) but never exercised
   by me — the orchestrator passes it on the official run; smoke asserted it stays absent
   when not given.
4. **Canary runtime path is NOT yet exercised end-to-end with real ledgers** — by design it
   only runs in claim mode with the lever ON, which only the orchestrator's official
   `--powered-cpu` run does. What IS verified: the comparator objects (reused from PR-09/PR-10,
   proven in two prior powered runs), the missing-ledger ABORT (unit test), the schema
   cleanliness, and cross-run bit-identity of frozen cells at smoke scale (both plain and
   treated, vs the repo's PR-08 smoke ledger).
5. **Not done (deliberate):** no `--powered-cpu` run (orchestrator's); no changes to
   `trunklr_claim.py`/`floorfreeze_claim.py` (ffc canary reused via lazy import instead of
   re-exporting through tlc — tlc aliases the identical ffc objects anyway); no verdict/bar
   changes (PR-08 §4 bars inherited verbatim, untouched); FROZEN-TRUNK records no
   "lever-inapplicable" cell key (the prereg's "recorded as such" is satisfied by the schema-
   clean requirement + `meta["pr11_repair"]` provenance + the canary records
   `res["canary.<ARM>"]`, whose note states the lever is structurally inapplicable there).
6. One cosmetic edit mishap during development (a stray line briefly merged into
   `test_prizma_lm_runner.py`'s new section) was caught by the first test run and fixed
   before anything shipped; final tree is clean.
