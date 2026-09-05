# NOTICE — v1 additive-arm quarantine (step/forward divergence found and fixed)

**Date:** 2026-09-03. **Status:** the `additive` records of the first grid are quarantined in
place as `raw_analog_probe_v1_stepgap_additive.json` (verbatim copy of the first
`raw_analog_probe.json`). Per `docs/RETENTION.md` nothing was deleted; the main
`raw_analog_probe.json` was superseded-corrected (3 delta runs kept, additive re-run).

## What happened

The first grid's `additive` arm streamed its frozen-eval sequences through
`PrizmaSeqBlock.step()`, which — **pre-existing repo behavior, not introduced by the analog
levers** — ignored `write_mode='additive'` and applied the delta write
(`u = beta_w*v - beta_e*(alpha*S k)`) unconditionally. Training (`forward` → `chunked_delta`)
used the true additive write (`u = beta_w*v`, no erase read-back). Every `additive` cell of the
v1 grid therefore measured a model deployed under a write rule it was never trained with.

Evidence (v1 additive seed 0, knobs off): parallel `forward()` accuracy 0.742 vs streaming
`step()` accuracy 0.358 — a 38-point deployment gap. For comparison, every `delta` model
satisfied stream(bits=0) == forward() to ~1e-3 (the existing O(1) guard), so the gap is
specific to the additive path. No existing test covered `step()` with
`write_mode='additive'`; `tests/test_analog_lever.py::test_g1_step_equals_forward_additive`
now pins it (< 1e-4 after the fix).

## The fix

`seq/prizma_seq.py::PrizmaSeqBlock.step` now branches on `write_mode` in both the `n_delta>=2`
and `n_delta==1` paths, mirroring `_delta_reference` / `chunked_delta(write_mode='additive')`
exactly (`u = beta_w * v`; `u = eta ⊙ v` under Lever G). The delta path is byte-identical to
the pre-fix code (all 52 kernel/lever/analog tests green before and after).

## Consequences for the v1 numbers

- v1 **delta** records: VALID (path unchanged, bit-identical). They are kept in the main raw
  file and were not re-trained.
- v1 **additive** records: INVALID as an analog-robustness measurement (chimeric deployment
  rule); retained only as evidence of the step gap. Superseded by the post-fix re-run.
- Incidental by-product: the v1 gap is itself an honest data point — it measured "additive-
  trained weights driven by a delta write", and its 0.358-vs-0.742 divergence shows the
  deployment rule must match the training rule for the additive family. It is NOT cited as an
  analog-robustness number anywhere.
