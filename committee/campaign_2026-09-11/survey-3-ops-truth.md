# Survey 3 — Ops / truth drift audit (read-only)

Repo: `C:\Users\Kullanıcı\Prizma` · HEAD `3a0e4ca` (PR-21 EXECUTED) · tree clean · venv Python 3.12.10.
Method: file reads + grep only; `pytest --collect-only -q` (2.7 s); NO suite run, NO source modified.

## 1. CI truth

- `.github/workflows/ci.yml`: ubuntu-latest, matrix **Python 3.11 + 3.12** (line 25), CPU torch from
  `download.pytorch.org/whl/cpu` (lines 39-41), `pip install -r requirements.txt` + `pytest` (43-46),
  then `pytest -q` with `CUDA_VISIBLE_DEVICES=""` (48-51). CI asserts **no test count** — counts in docs
  cannot break CI, but they are the harness's public truth statement.
- Dependencies: `requirements.txt` = numpy>=1.24, torch>=2.2, matplotlib>=3.7. CI command matches the
  README recipe (README.md:251-254). Local venv is 3.12.10 (3.12 leg covered; 3.11 not runnable here).
- **Measured current collection: `532 tests collected in 2.67s`** (command below), vs README's 211.
  Skip split is consistent with the documented "10 skipped": 9 static CUDA skips
  (`tests/test_delta_triton.py:218` param ×3, `:245` ×1; `tests/test_fused.py:216` param ×5) + 1 scipy
  runtime skip (`tests/test_stats.py:68`) → expected pass today ≈ **522P / 10S**.

```
.venv\Scripts\python.exe -m pytest --collect-only -q   ->  532 tests collected in 2.67s
```

Stale numbers found (full list in §6): README.md:254 ("211 collected -> 201 passed, 10 skipped"),
README.md:255-256 (MPS "225 collected (216 passed, 9 skipped)"), notebook lines 48+87 ("478 passed,
10 skipped"), RALPH.md:395 ("expect 392 passed").

## 2. Registry consistency (INDEX.md vs RALPH.md vs README outcomes)

- INDEX.md has **18 data rows**: 01, 02, 03(IN-WRITE), 03(CLAIMED), 04, 05, 06, 07, 09, 21, 20, 19, 18,
  17, 16, 15, 14, 13. **Missing rows: PR-08, PR-10, PR-11, PR-12.**
- They existed until commit `05ad731` ("PR-13 EXECUTED"): that commit's INDEX diff **deleted the four
  rows plus the LANE-EXPLORATORY footnote** and inserted the PR-13 outcome row (1 insertion / 9
  deletions). Verified via `git show 05ad731^:docs/preregistry/INDEX.md` — all four rows present there.
  All four source docs still exist in `docs/preregistry/`, and RALPH.md + README outcomes document all
  four (PR-08 NEGATIVE, PR-10 CLAIMED, PR-11 NEGATIVE, PR-12 CLAIMED) → the deletion is an edit
  accident, not a policy retirement.
- Count contradiction: RALPH.md:254 states "Registry: 12 CLAIMED / 6 NEGATIVE". INDEX as displayed
  shows 10 CLAIMED / 4 NEGATIVE / 1 NOT-ESTABLISHED; including the four missing rows the true counts are
  12 CLAIMED / 6 NEGATIVE / 1 NOT-ESTABLISHED → RALPH right, INDEX wrong.
- Verdict cross-check of the rows that do exist: README.md:218-236 matches INDEX/RALPH on every outcome
  (PR-20 NOT-ESTABLISHED final, PR-13 CLAIMED, PR-17 SCHEDULE-CARRIED, PR-19/18 REPLICATED, PR-21
  TISSUE-COSTS-RECOVERY). **No contradicting verdicts found among existing rows.**
- Duplicate id: two rows for PR-2026-09-03-03 (INDEX.md:13 IN-WRITE + :14 CLAIMED) — lifecycle expects
  one row transitioned (POLICY.md:64).
- Status vocabulary: INDEX.md:21 uses `NOT-ESTABLISHED`, which is not in the status list of
  INDEX.md:5-7 nor POLICY.md (grep found it only in prerun docs as an outcome term).

## 3. Artifact existence spot-check (PR-18/19/20/21)

All referenced dirs and logs EXIST; per-seed records confirmed ("per_seed" keys present in each ledger):

- PR-18: `results/prizma_lm_PR-2026-09-03-18/` = powered.json (949,291 B) + pooled_fresh_verdict.json
  (outcome PASS, seeds 5-14, B1 -0.2759 CI [-0.2893,-0.2624]) + lock; 3 console logs at
  `results/pr18_replication_official_console{,2,3}.log` — matches INDEX ("+ 3 console logs"). OK.
- PR-19: `results/manyblock_PR-2026-09-03-19/` = powered.json (77,464 B) + lock; console logs at
  `results/manyblock_PR-2026-09-03-19_official_console{,2}.log`. OK.
- PR-20: `results/damage_gap_repl_PR-2026-09-03-20/` (powered.json 6,367 B) +
  `results/manyblock_PR-2026-09-03-20_ext/` (powered.json 77,966 B; **contains the pooled fresh n=10
  verdict** NOT-ESTABLISHED) + 2 console logs (damage_gap_repl…, manyblock_ext…). OK; only nit: the
  pooled verdict lives in the `_ext` ledger, not in `damage_gap_repl` (INDEX wording "pooled verdict
  in-ledger" is satisfied but slightly ambiguous about which sibling).
- PR-21: `results/recovery_attr_PR-2026-09-03-21/` = powered.json (5,807 B; outcome
  TISSUE-COSTS-RECOVERY) + lock; `results/recovery_attr_PR-2026-09-03-21_official_console.log`. OK.
- **No missing artifact found.** The old "owner-approved" strings inside PR-08 raw ledgers/archives are
  historical records — per retention rules they must NOT be edited (fixed in live code only).

## 4. Notebook truth (`PRIZMA_GPU_CAMPAIGN.ipynb`)

- Header stage table (line 38) row 6: `seq/prizma_lm_claim.py --powered --trunk-lr-c 7.5e-4
  --domain-exclusion --ledger-dir prizma_lm_PR-2026-09-03-13_gpu`, labeled CONFIRMATORY.
  **But the actual stage-6 code cell (line 237) runs `!python seq/prizma_lm_claim.py --powered`
  with no flags** → executes the ORIGINAL PR-08 protocol on the default `prizma_lm_PR-2026-09-03-08`
  ledger, contradicting the header and RALPH's S3-READINESS claim that stage 6 was updated.
- Stage-6 archive glob (line 247) still archives `prizma_lm_PR-2026-09-03-08` while the header names
  `prizma_lm_PR-2026-09-03-13_gpu` — archive would miss the actual (header-intended) ledger.
- Stage-6 markdown (lines 215-228) still titles itself "PR-2026-09-03-08 POWERED" and describes the
  original 5-arm/3-block run, without the header's "confirmatory, PR-13" framing.
- Line 225: **"owner-approved C-range repair"** — false provenance (Pre-GPU review H-1: no owner
  decision occurred; the seq/ strings were fixed to "maintainer addendum 2026-09-08"). Live notebook
  text still carries it.
- Expected sanity lines 48 and 87: "478 passed, 10 skipped" — stale (532 collected today).
- Header purpose (line 12) ends the locked order at "PR-08 powered"; the locked order per RALPH S3
  also names the kernel session — not represented as a stage (pre-existing, minor).

## 5. Dangling items (seq/ + tests/)

- No `TODO`/`FIXME`/`XXX`/`HACK`/known-broken markers in `seq/` or `tests/` (only an innocuous
  "protocol-neutral placeholder" comment, seq/prizma_seq.py:147).
- No test found pinning a superseded behavior. Checked the riskiest candidates:
  `tests/test_prizma_lm_runner.py:209-225` clause-(b) pins are consistent with the corrected sign
  semantics (`seq/prizma_lm_claim.py:373`: `b3b_ok = forced_b_mean >= prim_b_mean - B3_FORCED_COST`);
  `tests/test_recovery_attribution.py:83/92` match the PR-21 pre-committed outcomes;
  no test pins `0.1554`/`0.1995` or the old "clause (b) PASS" prose; no test pins stale counts.
- The only superseded string still live in a non-artifact file is notebook line 225 (see §4/D8).

## 6. Drift list (each: what it says → truth → fix)

1. **INDEX.md missing PR-08/PR-10/PR-11/PR-12 rows** (INDEX.md:19-20, deleted by `05ad731`; docs and
   README/RALPH carry all four) → restore the 4 rows from `git show 05ad731^:docs/preregistry/INDEX.md`
   and re-check 12C/6N counts.
2. README.md:254 "211 collected -> 201 passed, 10 skipped" → measured 532 collected (~522P+10S) →
   update after the coordinator's full run.
3. README.md:255-256 "225 collected where MPS is available (216 passed, 9 skipped)" → stale by the same
   delta → re-measure on an MPS box or delete the MPS parenthetical.
4. Notebook lines 48/87 "478 passed, 10 skipped" → 532 collected (~522P+10S) → update both.
5. RALPH.md:395 GPU-SESSION block "expect 392 passed" + stages 0-5 only → 532 collected and a stage 6
   exists → refresh the block (count + stage list).
6. Notebook line 38 vs line 237: header command has `--trunk-lr-c 7.5e-4 --domain-exclusion
   --ledger-dir prizma_lm_PR-2026-09-03-13_gpu`, the executed cell has none → align cell with header
   (or revert header, but RALPH:150-155 declares the header intent current).
7. Notebook line 247 archives `prizma_lm_PR-2026-09-03-08` while the header ledger is
   `prizma_lm_PR-2026-09-03-13_gpu` → change the archive glob to the actual ledger dir.
8. Notebook line 225 "owner-approved C-range repair" → maintainer addendum 2026-09-08 (Pre-GPU H-1) →
   replace the phrase; do not touch the historical raw ledgers.
9. INDEX.md:21 status `NOT-ESTABLISHED` not in the documented status vocabulary (INDEX.md:5-7,
   POLICY.md:64) → add it to the vocabulary or map PR-20 to an existing status with the outcome in the
   text.
10. INDEX.md:13-14 duplicate PR-2026-09-03-03 rows (IN-WRITE + CLAIMED) → collapse into one CLAIMED row.
11. README.md:238-241 lists PR-LM-1 as a "Pending GPU-tier registration" → it has been registered
   (2026-09-08) and executed/claimed on CPU (PR-13/PR-18); only the GPU confirmation is pending →
   reword to "pending GPU-tier execution/confirmation".

Not drift: ci.yml itself (commands/deps match the repo, no count assertions); artifact retention for
PR-18..21 (all present, per-seed records and console logs included); no dangling code markers.

— end of survey (read-only; no source file modified).
