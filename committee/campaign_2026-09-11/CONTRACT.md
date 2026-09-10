# Campaign 2026-09-11 — CONTRACT (frozen before lanes start)

**Goal:** advance the transformer-rival ladder a measured step: (L1) a new biologically-grounded
mechanism in the fused tissue — multi-timescale (fast/slow) synaptic components, Benna-Fusi / CLS
lineage — as a default-OFF, tested lever, exploratory-probed tonight; (L2) an honest CPU efficiency
win for the Prizma-Seq kernel (opt-in `fast_reads` + a bit-identical solve simplification + the
missing parity tests); (L3) records/doc truth refresh. Owner directive: parallel agents, no errors,
stay inside the brain's machinery.

**Coordinator-owned files (agents must NOT touch):** `docs/preregistry/INDEX.md` (restored this
session), `tests/test_landscape_report.py` (locale fixes, done), `RALPH.md`, commits, probe/claim runs.

**Standing rules for every lane:**
- Agents NEVER run git write commands. Coordinator reviews + commits.
- Every new lever is **default-OFF** and must leave the shipped path bit-identical (guarded branch +
  a test that proves OFF == no-flag behavior).
- Litmus tests only (single test files, <3 min). Long jobs are coordinator-run.
- Write a lane report to `committee/campaign_2026-09-11/lane-<n>-<short>.md`; reply with <= 3 lines.
- Do not edit any file outside your lane's ownership; if a fix is needed elsewhere, report it.

## Lane 1 — tissue multi-timescale cascade (mechanism; owner: lane-1 report)

Files: `seq/fusion_probe.py`, `seq/prizma_lm_claim.py`, `seq/manyblock_claim.py`,
`seq/manyblock_probe.py`, `tests/test_tissue_cascade.py` (NEW).

Mechanism (frozen semantics):
- Applies to the tissue expert heads (`PCExpertHead`: Wenc/Wdec). Per head add a zero-init
  `W_fast` Parameter; the effective weight for forward/readout is `W + W_fast`.
- When enabled, each expert optimizer receives ONLY the `W_fast` parameters.
- After every optimizer step: `W <- W + kappa * W_fast` (consolidation) and
  `W_fast <- (1 - delta) * W_fast` (fast decay). `kappa`, `delta` in [0,1]; `delta > kappa`
  gives transient plasticity (not a reparameterization).
- Flags: `--cascade-target {off,tissue}` (default off), `--cascade-kappa` (default 0.05),
  `--cascade-delta` (default 0.10). Flags recorded in ledger meta + config fingerprint.
  `off` must be byte-identical to today.
- Diagnostics per block: L2 norm of W_fast (and its max) in the cell record, so the probe can
  show where the fast component holds information.

Acceptance (agent-run, litmus):
1. `tests/test_tissue_cascade.py`: (a) OFF == no-flag exact equality on a tiny fusion training
   (same object behavior, same bpc, same ledger fields); (b) mechanism math: after a manual
   step, W/F follow the cascade equations and forward uses W+F; (c) flags thread through all four
   files' entry points (signature-level + smoke-run smoke where cheap).
2. `python seq/manyblock_probe.py --help` shows the new flags; smoke invocation works (tiny).
3. Existing `tests/test_fused.py`, `tests/test_prizma_lm_runner.py`, `tests/test_syntax`? — run the
   fusion-adjacent test files and report green.

## Lane 2 — Prizma-Seq CPU kernel (owner: lane-2 report)

Files: `seq/delta.py`, `seq/throughput_benchmark.py`, `tests/test_fast_reads.py` (NEW),
`tests/test_delta.py` (NEW).

Work items (from survey-2, exact anchors):
1. C2 `fast_reads` (default OFF) in `chunked_delta` (`seq/delta.py:370,523`):
   `read_ratio = ratio / Ac[..., :, None]` replacing the second `[B,H,C,C]` sub+exp under
   `tril(...,-1)`. Thread a `fast_reads: bool = False` kwarg; default path byte-identical.
2. C1 bit-identical solve simplification (`seq/delta.py:270-272`): pass `Amat` directly with
   `unitriangular=True`, drop `torch.eye` + add. Verify maxdiff == 0.0 vs current.
3. NEW `tests/test_fast_reads.py`: fast vs default parity — production-shape gated
   (B=2,H=3,T=256,d=16,C=64, alpha in [0.5,1]): max|dO| < 1e-5, max|dS| < 1e-6, grad < 1e-4;
   off-path byte-identity == 0.0. NEW `tests/test_delta.py`: plain default `chunked_delta` vs
   `_delta_reference` parity (the missing coverage), plus the direct-A solve equivalence.
4. `seq/throughput_benchmark.py`: add `--fast-reads` flag (opt-in rows; default rows unchanged)
   and record the interleaved A/B medians in the report output. DO NOT overwrite
   `benchmark_results.md` with new numbers — the honest update is a snippet in your lane report:
   measured medians + date + box, for the coordinator to fold into docs.

Acceptance: new/related tests green; off-path parity == 0.0; benchmark flag prints both rows;
report the measured median speedup (target >= 1.02x on the gated eager CPU row; if not reproduced,
report the raw numbers honestly and recommend keeping the lever opt-in/annotated).

## Lane 3 — records truth (owner: lane-3 report)

Files: `README.md`, `PRIZMA_GPU_CAMPAIGN.ipynb`.

Fixes (from survey-3 drift list; exact anchors there):
1. README "Reproducing" block: replace `211 collected -> 201 passed, 10 skipped` with the measured
   `532 collected -> 522 passed, 10 skipped` (CPU-only, 2026-09-11; 9 static CUDA skips + 1 scipy
   skip); reword/delete the MPS parenthetical (do not invent MPS numbers).
2. README Prizma-Seq quad2 line (`README.md:159`): crosstalk `~0.076` -> the repo's own measured
   `0.117` (ratio 1.54; see docs/quad2_theoretical_convergence.md addendum; N*≈9.5).
3. README Pending-GPU paragraph (`README.md:238-241`): PR-LM-1 is REGISTERED and executed/claimed
   on CPU (PR-13/PR-18) — reword to "pending GPU-tier execution/confirmation".
4. Notebook: stage-6 code cell must match the header command
   (`--powered --trunk-lr-c 7.5e-4 --domain-exclusion --ledger-dir prizma_lm_PR-2026-09-03-13_gpu`);
   stage-6 archive glob must target the SAME ledger dir; stage-6 markdown title updated to the
   confirmatory framing; replace the false `"owner-approved C-range repair"` phrase with
   `"maintainer addendum 2026-09-08"` (Pre-GPU review H-1); sanity counts `478 passed, 10 skipped`
   -> `532 collected -> 522 passed, 10 skipped`. Do not touch raw result files.
5. Verify JSON validity after notebook edits (`python -c "import json; json.load(open(...))"`).

## Coordinator

- Registry repair (done), locale test fixes (done).
- Integration: full `pytest -q` gate; review each lane diff; fix cross-lane issues.
- Probe: run `manyblock_probe.py` default vs cascade (paired seeds 0-1) on CPU; compare
  recovery/damage/frac vs the registered PR-14 EX cells; record as LANE-EXPLORATORY.
- If (and only if) the probe shows a clear, direction-consistent mechanism effect: register PR-22
  with bars frozen from the probe, then run the claim. Otherwise: record the exploratory result
  honestly and leave the lever default-OFF for a future registration.
- Close-out: RALPH.md, commit plan, memory update. Wind-down 03:55 (03:58 shutdown is planned —
  never abort).
