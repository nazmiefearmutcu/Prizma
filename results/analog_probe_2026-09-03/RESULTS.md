# RESULTS — Analog-robustness probe (report 12-H2): delta vs additive write under state quantization + write noise

**Lane: LANE-EXPLORATORY (n=3 seeds) — hypothesis-flagging only, NOT a claim.**
Date: 2026-09-03 · Runner: `seq/analog_probe.py` · Raw records: `raw_analog_probe.json`
(crash-safe per-run saves; v1 additive arm quarantined — see
[`NOTICE_v1_additive_stepgap.md`](NOTICE_v1_additive_stepgap.md) and §3) · Summary machine table:
`summary.json` · Commission source: `committee/brainstorm_2026-09-03/12_neuromorphic_deployment.md`
H2 (pilot gate: one cell must train to >0.8 solve in <5 min CPU — PASSED, §2).

---

## 1. Hypothesis under test (pre-stated before the grid)

- **H2a:** the delta write stores the prediction error (`u_t = β_t·ε_t`, self-correcting), so a
  low-precision carried state S with noisy writes degrades **more gracefully** than an additive
  write (`u_t = β_t·v_t`, which can only accumulate errors).
- **H2b (counter):** the correction needs a precise read of `S·k`; a quantized S corrupts the
  residual, so delta degrades as fast or faster.
- Either outcome is informative per report 12 ("both outcomes publishable"); this run is the
  cheap exploratory probe that sizes the claim-grade pre-registration (§6).

**Protocol: train-FP32 / deploy-degraded.** Training uses the untouched FP32 chunk-parallel
kernel (`forward`); evaluation streams the frozen eval set through the exact O(1) `step()` path
with the analog levers set (`state_bits` b ∈ {0,4,6,8}, `write_noise_std` σ ∈ {0, 0.01, 0.05}).
Quantization is a per-head max-abs symmetric uniform quantizer applied to S **after every
write**, so subsequent pre-write reads see the degraded state (write-and-read both degraded,
the honest emulation per report 12-H2 HOW). Write noise is N(0, σ²) added to `u_t` before the
outer-product write. This is *inference-time* emulation — no QAT — exactly the report's
first-step protocol. Scope limitation: the levers are wired to `step()` ONLY (the WY/UT
chunk-parallel kernel never materializes per-token states; a shared quantization would require
re-deriving `seq/delta.py`). Documented in `PrizmaSeqConfig`; at defaults both knobs are
bit-identical (tests/test_analog_lever.py, 14 tests).

## 2. Frozen configuration (pilot-selected ONCE, then frozen — no per-cell tuning)

| item | value |
|---|---|
| Task | `MixedMQAR(vocab=128, max_pairs=32, num_queries=64, gap=0, min_pairs=1)` — **D=32** |
| Model | `PrizmaSeqLM` d64L2H2 (`d_model=64, n_layers=2, n_heads=2`), `feat_map='none'`, 106,824 params |
| Training | AdamW, lr **2e-3**, 3000 steps, batch 64, warmup 300, cosine to 0.1×, `build_and_train` seeding (seed before construction) |
| Seeds | (0, 1, 2) per write mode — **n=3** |
| Eval | frozen set: 8 batches × batch 64 under dedicated `eval_seed=12345`, `eval_sample` (fixed target difficulty); streamed token-by-token through `step()` |
| Grid | {delta, additive} × {0,4,6,8 bits} × {0, 0.01, 0.05} = 24 conditions per seed pair |

**D=64 was piloted and DROPPED for time — disclosed:** measured 254 ms/step (`none`) to
502 ms/step (`quad2_lowrank`) on this Ryzen 7 8C/16-thread box ⇒ the 6-run grid alone is
>1.7–3.4 h, over the ≤2 h hard budget. D=32 halves the sequence (T=128) and runs at 116 ms/step.
**Pilot (delta, seed 0, D=32):** lr ∈ {1e-3, 2e-3, 3e-3} → time-to->0.8: none / **281 s (0.867
at step 2000)** / none. lr=2e-3 frozen for all arms; 3000-step cap. Pilot gate "one cell >0.8 in
<5 min": **PASS**. Total probe compute: pilot 22 min + v1 grid 52 min + additive re-run 19 min
(§3) ≈ **93 min CPU**, inside budget; the 90-min seed-reduction guard never fired.

## 3. Incident discovered mid-probe: `step()` ignored `write_mode='additive'` (pre-existing, fixed)

The v1 grid's additive arm streamed at 0.358 while its parallel `forward()` scored 0.742
(seed 0, knobs off): `step()` applied the **delta** write unconditionally, so additive-trained
weights were deployed under a write rule they were never trained with. This is a **pre-existing
repo divergence** (no existing test covered `step()` × `write_mode='additive'`), not something
the analog levers introduced. Fix: `step()` now branches on `write_mode` mirroring
`_delta_reference` exactly; the delta path is byte-identical pre/post fix; post-fix
stream(bits=0) == forward() for additive too (0.742 == 0.742), pinned by
`tests/test_analog_lever.py::test_g1_step_equals_forward_additive`. Per `docs/RETENTION.md` the
v1 additive records are quarantined verbatim in `raw_analog_probe_v1_stepgap_additive.json`
(bug evidence; **never cite them as analog numbers**); the delta records were unaffected and
kept. Full account: [`NOTICE_v1_additive_stepgap.md`](NOTICE_v1_additive_stepgap.md).

## 4. Main table (post-fix; streaming masked accuracy on the frozen eval, mean±std over n=3)

`ret` = acc / own knobs-off streaming accuracy (matched-clean normalization, per seed then
averaged). Chance ≈ 0.016 (1/64 value tokens); the partial-recall floor sits well below solve.

| bits | noise | delta acc | additive acc | delta ret | additive ret |
|---|---|---|---|---|---|
| 0 | 0.00 | 0.813±0.033 | 0.485±0.211 | 1.000 | 1.000 |
| 0 | 0.05 | 0.791±0.032 | 0.437±0.183 | **0.974** | **0.909** |
| 4 | 0.00 | 0.588±0.012 | 0.326±0.102 | **0.724** | 0.730† |
| 4 | 0.05 | 0.589±0.016 | 0.310±0.091 | 0.725 | 0.700† |
| 6 | 0.00 | 0.795±0.031 | 0.467±0.203 | 0.978 | 0.967† |
| 6 | 0.05 | 0.776±0.030 | 0.423±0.174 | **0.955** | **0.887** |
| 8 | 0.00 | 0.810±0.034 | 0.483±0.211 | 0.997 | 0.995 |
| 8 | 0.05 | 0.790±0.032 | 0.436±0.182 | **0.973** | **0.908** |

σ=0.01 rows (in `summary.json`) track the σ=0 rows within ±0.003 — sub-1% write noise is
negligible at this scale.

**† The additive grid-level columns are floor-confounded:** additive solved MixedMQAR D=32 on
only **1 of 3 seeds** at this budget (train-best 0.746 / 0.488 / 0.226). A non-solver has no
recall circuit to degrade, so its retention ≈ 1 regardless of bits (v1 additive seed2 showed
ret 1.08 at 4-bit — quantization "helping" a floor model). The additive mean is therefore NOT a
matched-clean comparison. The matched-clean cells are §5.

## 5. Matched-clean analysis (the comparison H2 actually asks for)

Only additive **seed 0** reached the solved operating point (clean streaming 0.742, vs delta's
0.77–0.85 per seed). Matched-clean comparison, delta (n=3) vs additive (n=1):

| condition | delta ret (n=3) | additive ret (n=1, seed 0) | Δ (delta − additive) |
|---|---|---|---|
| 4-bit, no noise | 0.724 ± 0.005 | 0.621 | **+0.103** |
| 6-bit, no noise | 0.978 ± 0.004 | 0.974 | +0.004 (parity) |
| 8-bit, no noise | 0.997 ± 0.001 | 0.999 | −0.002 (parity) |
| FP32 state, σ=0.05 | 0.974 ± 0.003 | 0.888 | **+0.086** |
| 4-bit, σ=0.05 | 0.725 ± 0.003 | 0.578 | **+0.147** |

O(1)-sanity: every model's stream(bits=0, σ=0) equals its parallel `forward()` accuracy to
≤0.002 (delta 0.854/0.854, 0.812/0.812, 0.772/0.772; additive seed0 0.742/0.742).

## 6. Verdict (LANE-EXPLORATORY, honest)

1. **Directionally supports H2a; H2b rejected at these operating points.** The delta write
   degrades more gracefully than additive at matched clean accuracy under (a) 4-bit state
   quantization (+10.3 retention points, but additive side is n=1) and (b) write noise σ=0.05
   (+8.6 points on an FP32 state; +6.5–6.8 points at 6/8-bit, where the grid-level additive
   comparison is floor-conservative but still delta-favorable). The noise result is the
   cleaner of the two: write noise on a self-correcting write is partially repaired by
   subsequent writes; on an accumulating write it persists — exactly the H2a mechanism.
2. **Quantization is the binding impairment, and 4-bit is harsh for both.** Delta retains only
   0.724 at 4-bit (−22 points): the "analog-fit" story at 4-bit is *relative* robustness, not
   immunity. At 6–8 bits both modes are essentially undegraded without noise (ret ≥ 0.97) —
   the deployment-relevant separation under quantization lives at ≤4–5 bits.
3. **Sub-1% write noise is a non-event** at this scale (±0.003). A claim-grade design should
   sweep σ ≥ 0.02, or model ADC/DAC granularity instead of Gaussian noise alone.
4. **Power/effect honesty:** n=3 seeds overall; the matched-clean additive side is **n=1**
   (additive failed to solve 2/3 seeds at D=32/lr=2e-3/3000 steps — consistent with the known
   recall capacity/optimization fragility of additive linear attention, B6 `PRIZMA_noDelta`).
   No significance test is warranted at this n; nothing here is claim-grade. The exploratory
   effect sizes (+0.10 quantization, +0.09 noise at matched clean) are what §7's power
   analysis is built from.

## 7. PRE-REGISTRATION DRAFT — registry id **PR-2026-09-03-04**, status **IN-WRITE**

> To be promoted into `docs/preregistry/` (and the INDEX row added) by the registry maintainer;
> per the wave's scope split this document does NOT edit `docs/preregistry/INDEX.md`. Numbers
> below are frozen BEFORE any claim-grade run; the exploratory probe above is its only prior.

| field | value |
|---|---|
| Registry id | **PR-2026-09-03-04** |
| Date drafted | 2026-09-03 (status IN-WRITE — not yet registered/frozen) |
| Lane | **CLAIM** (two-lane policy, `docs/preregistry/POLICY.md`) |
| Title | Delta-vs-additive analog robustness of the carried state (state quantization + write noise), MixedMQAR |
| Primary artifact (to be written by the run) | `results/analog_probe_claim/` (raw + verdict, same crash-safe discipline) |
| Basis | Report 12-H2; exploratory probe = `results/analog_probe_2026-09-03/` (this directory) |

**H2a (claim under test):** at matched clean accuracy, the delta write degrades more gracefully
than the additive write under (i) 4-bit carried-state quantization and (ii) write noise
σ=0.05, each measured as retention = acc(condition)/acc(knobs-off) on the frozen eval.

**Arms & protocol (frozen in advance):**
- Identical model/task/training skeleton to §2, but: **per-arm LR sweep** over
  `seq.lrsweep.DEFAULT_GRID` (LR-fairness: no arm is denied an LR another gets; v1's additive
  non-solve is the disclosed motivation), then all claim seeds at the chosen LR.
- **Matched-clean gate (binding):** an arm enters the comparison only at a (difficulty, lr)
  where its clean streaming accuracy is ≥ 0.80 on ≥ 4/5 seeds. If additive cannot reach the
  gate at D=32, the difficulty drops (D=24, then D=16) **identically for both arms** until the
  gate passes for both — never per-arm.
- Conditions: {4-bit, σ=0} and {FP32, σ=0.05} primaries (the two exploratory signals); 6/8-bit
  and σ=0.01 secondaries. n = **5 seeds** per arm (seeds 0–4), no substitution.
- Deployment path: post-fix `step()` (additive-correct), knobs on the streaming path only;
  train-FP32/deploy-degraded; O(1)-sanity cell (stream-off == forward ≤ 0.005) required per
  model, else the run is invalid.

**Decision rule (frozen):** H2a SURVIVES iff, on BOTH primaries, delta retention exceeds
additive retention by ≥ **+0.05** (5 retention points) with one-sided Welch p < 0.05 across the
5v5 seed pairs (Holm-corrected over the 2 primaries), AND the O(1)-sanity gate passes for every
included model. Otherwise H2a is recorded NOT-SUPPORTED at this scale and the deployment
differentiator is not claimed anywhere (README/docs). MDE at n=5, σ(ret)≈0.01–0.03 (from the
exploratory seed spreads): detects Δ ≥ ~0.03–0.05 — the +0.05 bar is at the design's edge,
declared in advance.

**Budget:** ~6–10 min/run CPU × (2 arms × (5 seeds + ≤5 sweep runs) × ≤2 difficulties) ≈
2.5–4 h CPU; runs only after the two pilot gates (solve-time, matched-clean) pass.

**Pre-committed negative handling:** on NOT-SUPPORTED, report 12-H2's H2b reading is recorded
("the correction requires a precise S·k read; delta does not tolerate low-precision state
better"), the `docs/Prizma.md §5` analog wording is scoped to exclude state-precision claims,
and no retry at other bit widths without a NEW pre-registration.

## 8. Files changed by this wave (scope audit)

- `seq/prizma_seq.py` — `state_bits` / `write_noise_std` / `write_noise_seed` config knobs
  (defaults OFF = bit-identical), `_quantize_state_symmetric` helper, `step()` wiring (both
  n_delta branches), **plus the additive step/forward fix of §3** (pre-existing bug, found by
  this probe; delta path byte-identical).
- `seq/analog_probe.py` — NEW runner. `tests/test_analog_lever.py` — NEW (14 tests).
- `results/analog_probe_2026-09-03/` — this report + raw records + quarantine notice.
- NOT touched: everything on the concurrent-waves do-not-touch list (incl.
  `docs/preregistry/*`, `seq/recall_gate.py`, `seq/delta.py`, README).
