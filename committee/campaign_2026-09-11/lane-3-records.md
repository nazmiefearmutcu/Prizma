# Lane 3 — records truth (README.md + PRIZMA_GPU_CAMPAIGN.ipynb)

Date: 2026-09-11 · Repo: `C:\Users\Kullanıcı\Prizma` · Base HEAD: `3a0e4ca`
Scope honored: edited **only** `README.md` and `PRIZMA_GPU_CAMPAIGN.ipynb`. No git writes.
No changes to `docs/preregistry/*`, `RALPH.md`, `tests/*`, `seq/*`, or any `results/` artifact —
historical raw records untouched.

## Verification summary

- `python -c "import json; d=json.load(open('PRIZMA_GPU_CAMPAIGN.ipynb',encoding='utf-8')); print(len(d['cells']))"` → `26`
- Content-level diff vs `HEAD:PRIZMA_GPU_CAMPAIGN.ipynb` (parsed JSON object equality): changed cells
  are exactly `[0, 4, 21, 22, 23]`; `metadata` identical; `nbformat`/`nbformat_minor` identical;
  file ends with `\n`, no CRLF.
- Notebook edited as JSON: `json.load` → mutate specific source strings → `json.dump(ensure_ascii=False, indent=1)`
  + trailing `\n` (matching existing conventions). Note: `ensure_ascii=False` rewrites `\uXXXX`
  escapes elsewhere in the file as literal UTF-8 — content-proven-neutral by the parsed-object
  comparison above (only the five intended cells differ).
- README: 3 targeted edits; `git diff --stat` = `README.md | 19 +++++-----` (nothing else).

---

## Fix 1 — README "Reproducing the falsifiability harness" (stale CPU + MPS counts)

**Before (README.md, code block):**

```
pytest -q          # CPU-only (what CI runs): 211 collected -> 201 passed, 10 skipped, ~3 min.
                   # 225 collected where MPS is available (216 passed, 9 skipped): several
                   # kernel-equivalence tests are parametrised over devices.
```

**After:**

```
pytest -q          # CPU-only (what CI runs): 532 collected -> 522 passed, 10 skipped
                   # (measured 2026-09-11; the 10 skips = 9 static CUDA skips + 1 scipy skip).
                   # Several kernel-equivalence tests are parametrised over devices, so a box
                   # with MPS available collects a different count — re-measure there.
```

**Choices made (no numbers invented):**
- MPS parenthetical: **reworded, not deleted** — all stale MPS numbers (`225 collected (216 passed,
  9 skipped)`) removed; the true statement that kernel-equivalence tests are device-parametrised is
  kept, with the instruction to re-measure instead of quoting an unverified count.
- Dropped `~3 min`: that wall-time was tied to the old 201-test run; no new wall-time was measured
  for the 532-test suite, so asserting one would be invented.

## Fix 2 — README line 159 quad2 crosstalk (`~0.076` → measured `0.117`)

**Before:**

```
* **Gershgorin Capacity Bounds:** Using the Gershgorin Circle Theorem, we derive the capacity limit $N < 1 + 1/\text{cross}(\phi)$, showing that quadratic keys (`quad2`, crosstalk ~0.076) push capacity bounds to $N < 14$ compared to $N < 8$ for linear keys (`none`), resolving the capacity block on MQAR $D=128$.
```

**After:**

```
* **Gershgorin Capacity Bounds:** Using the Gershgorin Circle Theorem, we derive the capacity limit $N < 1 + 1/\text{cross}(\phi)$: with the repo's **measured** crosstalk for quadratic keys (`quad2` **0.117**, 1.54× the previously quoted `~0.076`), the bound caps $N^* \approx 9.5$ compared to $N < 8$ for linear keys (`none`). Per the addendum in [`docs/quad2_theoretical_convergence.md`](docs/quad2_theoretical_convergence.md), this bound does **not** by itself explain the repo's own MQAR $D=128$ PASS — explaining it is the open job of the capacity-law program (PR-02).
```

**Disclosed deviation:** besides the number/citation fix, I replaced the tail
`resolving the capacity block on MQAR $D=128$` with the addendum-consistent statement, because the
addendum I was instructed to cite explicitly flags that reading as an overclaim by implication
(Flag A: a bound capping N<10 cannot explain a D=128 PASS). Keeping the old tail with the corrected
number would have contradicted the citation in the same sentence. Number provenance matches
`docs/quad2_theoretical_convergence.md` §Number provenance table (`quad2` artifact value `0.11699`,
ratio 1.54) and `N* = 1 + 1/0.11699 ≈ 9.55`.

## Fix 3 — README "Pending GPU-tier" paragraph (PR-LM-1 registered/executed on CPU)

**Before:**

```
Pending GPU-tier registrations: PR-01 (powered surprise-gating ablation, frozen protocol),
PR-02 (crosstalk capacity-law D-frontier), Tier-0 repairs (clean recall gate, B4 closure,
first GLA/Mamba-2 landscape), kernel decision (≤1.5× TF step time), PR-LM-1 (the flagship
continual-LM bar — unblocked on the block-drift regime by PR-07′).
```

**After:**

```
Pending GPU-tier execution/confirmation: PR-01 (powered surprise-gating ablation, frozen
protocol), PR-02 (crosstalk capacity-law D-frontier), Tier-0 repairs (clean recall gate, B4
closure, first GLA/Mamba-2 landscape), kernel decision (≤1.5× TF step time), and the A100
confirmation of PR-LM-1 — the flagship continual-LM bar is REGISTERED (2026-09-08) and already
executed/claimed on CPU (PR-13, replicated at n=10 fresh by PR-18); only the GPU-tier
confirmation is pending.
```

---

## Fix 4 — Notebook PRIZMA_GPU_CAMPAIGN.ipynb

Cells referenced by 0-based index in the notebook JSON: 0 = header markdown,
4 = stage-0d code, 21 = stage-6 markdown, 22 = stage-6 code, 23 = stage-6 archive code.

### (a) Stage-6 code cell (cell 22) — align with header command

**Before:** `!python seq/prizma_lm_claim.py --powered`

**After:** `!python seq/prizma_lm_claim.py --powered --trunk-lr-c 7.5e-4 --domain-exclusion --ledger-dir prizma_lm_PR-2026-09-03-13_gpu`

(Matches the header stage-table row 6 command, with the notebook's `!python` prefix. Flags verified to
exist in `seq/prizma_lm_claim.py`: `--trunk-lr-c` line 1397, `--ledger-dir` line 1406,
`--domain-exclusion` line 1410.)

### (b) Stage-6 archive glob (cell 23) — same ledger dir

**Before:**

```
# Stage 6 archive (copy-only)
archive_stage("pr08_prizma_lm", ["prizma_lm_PR-2026-09-03-08"])
```

**After:**

```
# Stage 6 archive (copy-only)
archive_stage("pr13_prizma_lm", ["prizma_lm_PR-2026-09-03-13_gpu"])
```

**Additional consistency edit (flagged):** the archive *label* was also renamed
`pr08_prizma_lm` → `pr13_prizma_lm`, so the archive folder name matches the PR-13 ledger it now
copies. (Task asked only for the glob; the label is the same stale PR-08 framing being corrected
in (c). Easy to revert if the coordinator prefers.)

### (c) Stage-6 markdown title/framing (cell 21) — confirmatory repaired flagship (PR-13)

**Before (title + first line):**

```
## Stage 6 — PR-2026-09-03-08 POWERED: the Prizma-LM flagship bar (many-block fused column)  (~45–90 A100-min)

`python seq/prizma_lm_claim.py --powered` executes `docs/preregistry/2026-09-08-prizma-lm-flagship.md`
```

**After (title + first lines):**

```
## Stage 6 — PR-13 CONFIRMATORY: the REPAIRED Prizma-LM flagship (many-block fused column)  (~45–90 A100-min)

`python seq/prizma_lm_claim.py --powered --trunk-lr-c 7.5e-4 --domain-exclusion --ledger-dir prizma_lm_PR-2026-09-03-13_gpu`
executes `docs/preregistry/2026-09-08-prizma-lm-flagship.md` with the two REGISTERED REPAIR LEVERS
composed — PR-10 C-block trunk-lr scaling (×0.25) + PR-12 domain-exclusive experts: 5 arms
(PRIM-LM / FROZEN-TRUNK / SHARED-HEAD / FROZEN-CHECKPOINT / FORCED-RECRUIT) × 5 seeds (0–4) over the
3-block stream text8[0:1M) → tiny-shakespeare → text8[1.1M:2.1M) with the PINNED eval slices
(A-eval [1.0M,1.1M), B-eval = shakespeare last 10%, C-retention [0.9M,1.0M)). Bars B1–B4 EXACT:
Holm over B1–B2 (upper-tail t_isf), B3 = routing ledger + forced-recruit cost, B4 = trunk-drift
accounting (reported, not gated); INCONCLUSIVE straddling-CI rule; §5 failure branches echoed in the
verdict. This A100 stage is CONFIRMATORY, not gating — PR-13 already CLAIMED all three bars on CPU
(2026-09-09) and PR-18 replicated the verdict at n=10 fresh seeds (5–14); the run uses its own ledger
(`prizma_lm_PR-2026-09-03-13_gpu`). Runs AFTER the PR-02 stage, BEFORE the final archive cell.
```

(Facts used: PR-13 CLAIMED 2026-09-09 on CPU, PR-18 replicated at n=10 fresh — both already recorded
in README's outcomes table and RALPH; no new numbers added.)

### (d) False provenance phrase (cell 21)

**Before:**

```
Disclosed (ledger meta carries both notes): (1) owner-approved C-range repair — the doc's literal
C = text8[1.0M,2.0M) would train on the A-eval slice and corrupt B1; the runner pins C = [1.1M,2.1M)
(probe-2 precedent); needs a dated maintainer addendum. (2) §2 prose bug — C-retention [0.9M,1.0M)
IS A-train's tail; slice implemented as pinned, B2 stays a fair comparison.
```

**After:**

```
Disclosed (ledger meta carries both notes): (1) maintainer addendum 2026-09-08 (Pre-GPU review H-1) —
the doc's literal C = text8[1.0M,2.0M) would train on the A-eval slice and corrupt B1; the runner
pins C = [1.1M,2.1M) (probe-2 precedent). (2) §2 prose bug — C-retention [0.9M,1.0M)
IS A-train's tail; slice implemented as pinned, B2 stays a fair comparison.
```

### (e) Sanity expectation counts — cells 0 and 4

**Cell 0 before:** `**Expected pytest sanity:** `478 passed, 10 skipped` (updated 2026-09-09 for the PR-09..PR-13 campaign` + `(+86 tests); the count only grows via tested additions — record the ACTUAL output).`

**Cell 0 after:** `**Expected pytest sanity:** `532 collected -> 522 passed, 10 skipped` on CPU (updated 2026-09-11;` + `the count only grows via tested additions — record the ACTUAL output).`

**Cell 4 before:** `# EXPECT the sanity line: "478 passed, 10 skipped" (updated 2026-09-09; the count only`

**Cell 4 after:** `# EXPECT the sanity line: "532 collected -> 522 passed, 10 skipped" (updated 2026-09-11; the count only`

---

## Not touched / reported instead

- Cell 0 purpose line 6 and stage-table row 6 still name the stage "PR-08 (Prizma-LM flagship)
  powered" (row 6 already carries the `--trunk-lr-c …` command + "REPAIRED flagship — PR-13 …
  CONFIRMATORY" note). Renaming the stage label in the purpose/table was not requested; left as-is.
- All raw ledgers/archives (including old "owner-approved" strings inside
  `results/prizma_lm_PR-2026-09-03-08/` and `results/runs/`) intentionally untouched per retention
  rules — only `seq/` live code was fixed earlier by the maintainer.
- README footnote "CI was already collecting over 200" left as-is (historical note about the old
  "94 tests" quote; contains no stale value in the fix list).
