# A — PR-08 CPU-fallback prep + pre-GPU-review fixes (M-1, M-2, M-3) — REPORT

Agent lane: A (Prizma file zone: `seq/prizma_lm_claim.py`, `tests/test_prizma_lm_runner.py`,
`docs/preregistry/2026-09-08-prizma-lm-flagship.md`). Date: 2026-09-08, evening session.
No git write commands were run (orchestrator integrates). Nothing was written under
`results/` (retention policy).

## 1. What changed, per finding

### DOC — addendum #2 appended VERBATIM
`docs/preregistry/2026-09-08-prizma-lm-flagship.md`: the exact text supplied by the
orchestrator (heading "Addendum 2026-09-08 (pre-run, maintainer #2) — §6 CPU-feasible
fallback authorized for tonight; operational disclosures" + items 1-4, with its leading
`---` rule, matching the doc's existing addendum convention) was appended after the
existing "(pre-run, maintainer)" addendum. Nothing added or removed; non-ASCII characters
(≈, §, en/em dashes) preserved exactly. Tail read back after the write — renders correctly.

### M-2 — bars constants into the claim-cell fingerprint
`seq/prizma_lm_claim.py` `run()`: the `_fp({...})` payload for `claim.<arm>.s<seed>` gains
`"bars": res["meta"]["bars"]`. Verified live: the old default smoke ledger's cell cfgsigs
(e.g. `40ec60d20456e558`) differ from the fresh redirected smoke's (e.g. `7449d8939d25c724`)
— the old-fingerprint ledger can never be silently reused; a stale cell now hard-refuses.

### M-3 — false docstring sentence corrected
`run_routed` docstring: "A and B phases are identical across the three" replaced with
"A and B phases are identical across PRIM-LM and FORCED-RECRUIT (FROZEN-TRUNK shares phase
A only; its B-phase backbone is frozen by design)". The rest of the docstring (seed-stream
determinism parenthetical, freeze and forced-recruit notes) is unchanged.

### M-1 — disclosure (no behavior change)
Module docstring gains a DISCLOSURE block after the two doc-vs-code notes: FORCED-RECRUIT
raises a fail-loud RuntimeError if the pool has no free slot at C start (review M-1,
pre-disclosed in the doc's 2026-09-08 maintainer addendum #2; deterministic, resume-safe).
The RuntimeError itself was NOT touched (post-freeze, any fallback = protocol change).

### NEW `--powered-cpu` mode (doc §6 fallback, addendum #2 point 1)
- `_build_parser()`: third member of the existing mutually-exclusive mode group;
  `--powered`'s help string is byte-identical (parser `description` gained one
  powered-cpu clause — that is the parser description, not --powered's help).
- `needs_cuda(mode: str) -> bool` pure helper added next to `require_cuda`
  ("powered" -> True; "powered-cpu"/"smoke" -> False); `run()` calls it so the CUDA-guard
  decision is unit-testable without torch.
- `run(*, smoke, results_path=None, force_smoke_path=False, powered_cpu=False)`;
  `main()` threads `args.powered_cpu`.
- In `run()`: powered-cpu behaves exactly like powered (smoke_segs=None, seeds=CLAIM_SEEDS,
  full LR selection, verdict computed) EXCEPT `require_cuda()` is skipped (guarded by
  `needs_cuda`), ledger meta gains `"powered_cpu": True` and `"compute_fallback_note"`
  (exact mandated string, pinned as module constant `POWERED_CPU_FALLBACK_NOTE`), and the
  report dict mirrors `powered_cpu: True`. The meta/report keys are added ONLY when
  powered_cpu=True so `--smoke` and `--powered` ledger content stays byte-identical to
  before. `--powered` refusal verified unchanged on this CPU box (same message, exit 1).
- Module docstring: "two modes" -> "three modes" + a `--powered-cpu` usage line.

### TESTS (added; no existing test modified — none needed fixing)
- `test_powered_cpu_alone_accepted_and_combinations_rejected`: `--powered-cpu` alone parses
  (and only it is set); rejected combined with `--smoke`, `--powered`, or both.
- `test_needs_cuda_only_for_the_powered_mode`: needs_cuda("powered") True;
  needs_cuda("powered-cpu") and needs_cuda("smoke") False.

## 2. Exact commands run (repo root `C:\Users\Kullanıcı\Prizma`)

1. `./.venv/Scripts/python.exe -m pytest tests/test_prizma_lm_runner.py -q`
   -> `20 passed in 0.94s`, **EXIT=0** (18 existing + 2 new; suite still pure-layer, <1s;
   pytest DID print a summary this run, and exit code was trusted per the standing rule).
2. `./.venv/Scripts/python.exe -m seq.prizma_lm_claim --powered` -> byte-identical CUDA
   refusal text, **EXIT=1** (refusal intact on a no-CUDA box).
3. `./.venv/Scripts/python.exe -m seq.prizma_lm_claim --powered-cpu --smoke` -> **EXIT=2**
   (argparse mutual-exclusion rejection; nothing launched).
4. End-to-end smoke (ledger redirected OUT of results/; target file deleted first):
   `rm -f "$LOCALAPPDATA/Temp/pr08_smoke_postfix.json"` then
   `PRIZMA_RESULTS="$LOCALAPPDATA/Temp/pr08_postfix_zone" ./.venv/Scripts/python.exe -m seq.prizma_lm_claim --smoke --out "$LOCALAPPDATA/Temp/pr08_smoke_postfix.json"`
   -> **EXIT=0**; all 5 arms completed; SMOKE banner printed; ledger at
   `C:\Users\Kullanıcı\AppData\Local/Temp/pr08_smoke_postfix.json`; A-phase bit-identity
   across fusion arms reproduced (A_preB=4.942 x3; FROZEN-CHECKPOINT 5.284 as before).

## 3. Deliberate deviations / NOT done (honesty over polish)

1. **`PRIZMA_RESULTS` was set to a temp dir for the verification smoke** (deviation from
   the literal VERIFY command): the runner's `archive_run` snapshot otherwise writes an
   append-only JSON under `<results-root>/runs/`, i.e. into the repo's `results/`, which my
   hard rules forbid. With the env override, BOTH the ledger (--out) and the archive
   snapshot landed in `%LOCALAPPDATA%\Temp`. Repo `results/` untouched (git status shows no
   results/ change; the pre-existing old-fingerprint `results/prizma_lm_PR-2026-09-03-08/smoke.json`
   was never used or modified). If the orchestrator prefers the literal command, the only
   difference is one archive snapshot under results/runs/ (the pre-GPU reviewer's smoke had
   the same side effect).
2. **Fingerprint-sensitivity test (task 2e-iii) SKIPPED, per the task's own provision:**
   `plc._fp` imports `seq.gpu_harness`, whose import chain pulls `seq.common` -> torch, so a
   real-`_fp` sensitivity test would require torch and break the test file's "no torch, <1s
   pure layer" discipline. The M-2 effect was instead verified live by comparing cfgsigs in
   the old vs the fresh smoke ledger (different, above).
3. **`--powered-cpu` was NOT executed end-to-end** — it IS the ~6-8 h claim campaign; only
   its parser/guard contracts were exercised. Execution tonight is the campaign session's job.
4. `git status --porcelain` shows OTHER modified files (surprise_claim, stats, ship script,
   other prereg docs, untracked review artifacts) — those belong to parallel lanes/review
   fixes, not to this agent; I touched only my three zone files + this report.
5. The report dict mirror is conditional (`if powered_cpu`), read as part of the "when
   powered_cpu is True" clause; flagging in case the orchestrator wanted it unconditional.

## 4. State for the campaign session

The registered fallback is executable tonight with:
`python -m seq.prizma_lm_claim --powered-cpu` (default ledger
`results/prizma_lm_PR-2026-09-03-08/powered.json`; any interrupted run resumes by
fingerprint; the old SMOKE ledger at the old fingerprint is superseded, never reused).
The 03:55 wind-down rule wins over everything, including the n=10 extension.
