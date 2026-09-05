# Artifact-retention policy

> **The rule in one line:** every run — pass or FAIL — writes its raw per-seed records under
> `results/` *before* any verdict is computed; every verdict references that artifact path; and
> deleting raw artifacts is a protocol violation.

This policy exists because we already lost data once by not having it. It is repo culture, not a
suggestion: it binds future campaigns the same way the pre-registration rules do
([`docs/preregistry/POLICY.md`](preregistry/POLICY.md)).

## Why (the incidents this insures against)

**B4 char-LM — the failed run that vanished.** Spec §B4 pre-registered test BPC ≤ T+0.05 on BOTH
corpora. The tiny-shakespeare leg failed by −0.09 under an overfitting recipe — and *that failing
run's raw data was not retained on disk*, so it cannot be re-examined today. Nobody can distinguish
"a real negative result" from "a broken recipe that would have passed when fixed", because the
records are gone. The surviving one-corpus partial was then presented as a PASS. A deleted failure
and an unexamined failure look identical, and both get laundered into claims.

**campaign 2026-06-08 — the quarantine that worked.** When the recall-gate contamination was found
(a `--smoke` run had supplied seeds 0–1 — and the LR sweeps — in all ten cells of the "powered"
campaign; see [`results/campaign_2026-06-08/CONTAMINATION.md`](../results/campaign_2026-06-08/CONTAMINATION.md)),
the artifact was **quarantined in place**: every file left exactly as produced, nothing re-run,
nothing deleted, and a notice written next to it. That is the correct behavior and the precedent.
The raw bytes of a failure are evidence; the correction lives in a notice, not in a deletion.

## The policy

1. **Raw first, verdict second.** Every run persists its raw per-seed records under `results/`
   *before* any verdict is computed. In `seq/recall_gate.py` this is structural: `_train_arm`
   streams each seed to the results ledger crash-safe (json → `.tmp` → `os.replace`) the moment it
   finishes, so a cell that crashes mid-campaign still leaves every earlier seed on disk.
2. **Verdicts reference the artifact.** The leg runner snapshots the raw records via
   `archive_run(records, out_dir, label=...)` before computing a verdict and stores the path in
   `res["meta"]["raw_archive"]`. A number that cannot point at its raw records is not citable.
3. **FAIL runs keep their artifacts.** A failed, crashed, or refused run archives its partial
   records like any other. Failures are results. Negative results are the only kind that cannot be
   re-produced on demand (nobody re-runs a failure), which makes their records the *most* valuable
   to keep, not the least.
4. **Deleting raw artifacts is a protocol violation.** `results/` is append-only. If an artifact is
   wrong, contaminated, or embarrassing, the remedy is a quarantine notice next to it (precedent:
   `CONTAMINATION.md`), never `rm`. Superseded results get superseded by *new* runs, not by
   editing or removing the old records.
5. **Smoke never writes the campaign ledger.** A `--smoke` run defaults to
   `results/recall_gate_smoke.json`; a full/campaign run defaults to `results/recall_gate.json`;
   an explicit `--out` overrides either; a smoke run pointed at the campaign ledger is refused
   unless `--force-smoke-path` is passed. Keeping the two streams in separate files is what makes
   the 2026-06-08 mixing impossible rather than merely detectable.

## Mechanics (where the enforcement lives)

| Mechanism | Where | What it guarantees |
|---|---|---|
| Crash-safe streaming saves per seed | `seq/recall_gate.py::_train_arm` (`_save`: json → `.tmp` → `os.replace`) | a mid-campaign crash leaves all completed seeds on disk |
| `archive_run(records, out_dir=None, *, label=...)` | `seq/recall_gate.py` | one atomic snapshot of the raw records; never overwrites an earlier archive; returns the artifact path |
| Verdict-referenced artifact path | `seq/recall_gate.py::_run_leg` → `res["meta"]["raw_archive"]` | every verdict points at the raw records it was computed from |
| Smoke/campaign file separation + refusal | `seq/recall_gate.py::_results_path` / `_resolve_results_path` | smoke plumbing evidence cannot land in the campaign ledger |

Regression tests: `tests/test_recall_gate.py` — the BAR-0 file-separation tests (smoke default ≠
campaign default; smoke-then-campaign leaves the campaign ledger untouched; the refusal fires) and
the retention tests (`archive_run` verbatim + never-overwrites; a simulated mid-campaign crash
still persists the earlier seeds and they archive).
