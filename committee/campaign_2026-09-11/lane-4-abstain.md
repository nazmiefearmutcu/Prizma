# Lane 4 — Open-world abstain / NOVEL gate (`abstain_z` + `route_or_novel`) — report

**Date:** 2026-09-11 · **Box:** Windows, venv Python, numpy-only · **HEAD at proof time:** `3a0e4ca`
(+ lane edits only: `src/prizma.py`, `tests/test_expert_abstain_open_world.py`)

**Status: COMPLETE.** The documented-but-unimplemented abstain rule (survey-1 §2b, EXPERT_ECONOMY
§3.4) is now implemented as an opt-in, default-safe mechanism; the §2.3 calibration was reproduced
at reduced shipped-scale; all gates green (new file + 2 adjacent suites + full `tests/`).

---

## 1. Delivered

### `src/prizma.py`
- `Prizma.__init__` (line 251): new trailing kwarg `abstain_z=None` (positional-compat safe —
  append-only). Validation at 496–500: `None` stays `None`; provided values are cast to float and
  rejected with `ValueError` if non-finite.
- New method `route_or_novel(X)` (562–597), frozen semantics from doc §3.4 / lane brief:
  1. raises `ValueError` when `abstain_z is None` (opt-in; nothing else changes);
  2. `idx, S = self.route_for_inference(X)` — literally the same code path (no fork/copy);
  3. `trained = [e.n_seen > 0]` (the existing trained mask); **no trained expert → `novel=True`
     with the same `idx/S`** (nothing claims the batch);
  4. per sample: `z[n, m] = (S[n, m] - mu_m) / max(var_m, 1e-12)**0.5` over trained experts;
     statistic = `mean_n( min_{m trained} z[n, m] )`; `novel = statistic > abstain_z`;
  5. empty batch → `novel=False` (no evidence; documented in the docstring — the one edge case the
     spec does not fix);
  6. numpy-only, side-effect free (no training, no bookkeeping mutation; `recon_error` may consume
     an expert's own rng under the noise/quantization knobs, exactly as `route_for_inference`
     already does).
- **Nothing else touched**: `route_for_inference`, `predict_logits`, `train_batch`, all levers,
  bit-identity paths unchanged. `git status`: `M src/prizma.py` (+50/−1), no other file.

### `tests/test_expert_abstain_open_world.py` (NEW, 7 tests)
1. `test_abstain_knob_leaves_route_for_inference_bit_identical` — two instances (with/without
   `abstain_z=4`), identical tiny training streams → `np.array_equal` on idx **and** S for every
   task, plus equal full parameter vector (weights + `mu`/`var`) and equal `state()`.
2. `test_calibrated_batches_separate_trained_from_never_trained[0/1/2]` — reduced-scale mirror of
   the §2.3 protocol (same generator, same shipped knobs: K=3, d=24, n_classes=8, h=48, epochs=15,
   n_samples=3000, z_novel=5.0, one never-trained 4th domain, full 128-sample batches): every
   trained-domain batch → `novel=False` and byte-equal idx/S vs `route_for_inference`; every
   never-trained batch → `novel=True` at z=4.
3. `test_route_or_novel_requires_opt_in_and_flags_untrained_pool_novel` — default `abstain_z=None`
   → `ValueError`; fresh pool (no trained expert) → `novel=True`, routing unchanged.
4. `test_route_or_novel_is_side_effect_free` — repeated calls identical; `state()` and parameter
   vector unchanged.
5. `test_non_finite_abstain_z_rejected` — `abstain_z=nan` → `ValueError` at construction.

## 2. Exact results

| command | result |
|---|---|
| `pytest tests/test_expert_abstain_open_world.py -q` | **7 passed in 1.24s** |
| `pytest tests/test_continual_baselines.py tests/test_expert_economy.py -q` | **21 passed in 1.34s** |
| all three together | **28 passed in 2.57s** |
| full `pytest tests -q` (safety, since `src/` changed) | **553 passed, 10 skipped in 221s** (was 546P/10S; +7 = this lane) |

**Calibration measured** (recomputed by the fixture's protocol; batch statistic as in the method):

| seed | trained / frozen | trained-domain batch stats (12 batches) | never-trained domain batch stats (4 batches) |
|---|---|---|---|
| 0 | 3 / 2 | min −0.51, **max 1.52** | 159.6, 156.1, 152.6, 166.0 |
| 1 | 3 / 2 | min −1.15, **max 1.63** | 170.6, 178.8, 162.2, 156.2 |
| 2 | 3 / 2 | min −0.66, **max 2.50** | 181.2, 176.7, 181.8, 174.7 |

At z=4: 0/36 trained batches flagged (0% false-NOVEL), 12/12 never-trained batches flagged
(100% detection). The margin is enormous (~60–120σ on the probe side), so the test is
deterministic and not threshold-overfit; z=4 is the doc's number, untuned.

## 3. Calibration caveat (honest, per the brief)

- The doc's §2.3 separation depends on **geometry**, not just the generator: at the naive tiny-test
  scale (K=3, d=8, h=24, epochs 3–8) a single expert absorbed the whole stream, and a
  never-trained permuted domain was **not** separable (probe batch stats ≈ 0.3–5.4, known stats up
  to ~4). Detection appeared reliably only once the expert pool actually recruited one expert per
  domain (domain-pure floors) — i.e. at reduced shipped geometry. The committed test therefore
  uses K=3/d=24/h=48/15 epochs (runtime ≈0.3 s per seed) instead of the usual d=8 unit-test scale;
  this is the robust construction, **not** a thresholds hack — the measured margins are orders of
  magnitude away from z=4 on both sides.
- What the test pins is the **mechanism + the §2.3 measurement quality in the same generator
  family at reduced scale**, not the shipped K=5/n=6000 numbers themselves (those live in the
  committed exploratory ledger `results/expert_economy_2026-09-03/growth.json`, whose probe used
  the same rule shape).
- Cross-generator generalization remains the doc's §6.5 honest limit (one generator family;
  z_abstain=4 is measured, not universal). Nothing here upgrades that to a claim.

## 4. Notes / no blocking items

- Empty-batch semantics (`novel=False`) and non-finite `abstain_z` rejection are the only two
  choices the brief left open; both are documented in code and pinned by tests.
- No other file needed changes; no deviation from lane scope.
