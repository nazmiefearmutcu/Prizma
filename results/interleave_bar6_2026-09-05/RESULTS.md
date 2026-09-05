# PR-2026-09-03-06 — BAR-6 claim run RESULTS (LANE-CLAIM)

| field | value |
|---|---|
| Protocol | `docs/preregistry/2026-09-03-interleaved-router-bar6.md` (REGISTERED, frozen; executed VERBATIM — no knob, seed, or bar changes) |
| Date | 2026-09-05 |
| Seeds | 10–14 (n=5, the doc's frozen contiguous set; disjoint from the exploratory 0–1) |
| Arms × streams | 4 arms (`shipped`, `+fixes`, `−route_stat`, `−hot_young`) × 3 streams (`mixed`, `roundrobin`, E1 block guard) = 60 runs, all at full n=5 (no budget reduction; wall 75 s CPU) |
| Code paths | `experiments/interleave_bar6_claim.py` reusing `experiments/interleave_fix_probe.py` (`run_mixed` / `run_roundrobin` / `run_e1_guard`) → `experiments/expert_economy.py` constructions; levers in `src/prizma.py` (default-off, bit-identical) |
| Raw artifacts (RETENTION.md: never delete) | per-seed crash-safe records: `results/interleave_bar6_2026-09-05/raw/<arm>__<stream>__seed<k>.json` (60 files); consolidated streamed ledger: **`results/interleave_fix_PR-2026-09-03-06/raw.json`** (the doc's pre-registered primary artifact path) + mirror `results/interleave_bar6_2026-09-05/raw.json`; computed verdict: `results/interleave_bar6_2026-09-05/verdict.json` |

## Verdict: **INSUFFICIENT** (the doc §7 pre-committed FAIL-of-bar-2 branch)

> "**FAIL of bar 2** … **the fix is declared INSUFFICIENT**. BAR-6 is recorded
> ANALYZED→NEGATIVE, **PR-LM-1 is BLOCKED** on the router, and the n=5 failure numbers
> (per-arm, per-stream, with CIs and route logs) become the **documented blocker**."

Bar 1 (gate) PASSED; bar 2 (primary) FAILED on BOTH interleaved streams; bar 3 computed but moot under the decision rule.

## Per arm × stream: mean ACC ± sd (n=5, seeds 10–14; Student-t/Welch 95% CI)

| arm | mixed | roundrobin | E1 block (ACC; FGT) |
|---|---|---|---|
| shipped | **0.587 ± 0.045** [0.531, 0.643] | **0.565 ± 0.042** [0.513, 0.617] | 0.841 ± 0.042 [0.789, 0.892]; FGT 0.000 |
| +fixes (hot 1.0 + sample_top) | **0.597 ± 0.040** [0.547, 0.646] | **0.592 ± 0.041** [0.541, 0.643] | 0.842 ± 0.042 [0.790, 0.894]; FGT 0.000 |
| −route_stat (hot only) | 0.587 ± 0.045 [0.532, 0.643] | 0.534 ± 0.079 [0.436, 0.632] | 0.841 ± 0.041 [0.790, 0.893]; FGT 0.000 |
| −hot_young (sample_top only) | 0.596 ± 0.040 [0.547, 0.646] | 0.592 ± 0.041 [0.541, 0.643] | 0.842 ± 0.042 [0.789, 0.894]; FGT 0.000 |

Sanity: shipped E1 0.841 vs the 10-seed headline 0.834 — within seed noise (doc §5 requirement met).

## Decision rule, clause by clause (doc §6–§7)

- **Bar 1 (gate) — PASS.** `+fixes` E1 mean ACC 0.8421 ∈ [0.814, 0.854] (±0.02 of 0.834) AND mean FGT 0.000 ≤ 0.02. The levers are shipped-safe (as the bit-identical tests predicted).
- **Bar 2 (primary) — FAIL on both streams.** `+fixes` mean ACC: mixed **0.5965** (Welch 95% CI [0.5471, 0.6459]), roundrobin **0.5916** ([0.5406, 0.6426]). Both point estimates are ~0.10 below the 0.70 bar, and even the upper CI bounds (0.646 / 0.643) fall short of it. → INSUFFICIENT branch fires.
- **Bar 3 (attribution) — computed, moot.** By the letter of the rule both minus-arms fail bar 2, so both levers flag "causally necessary"; the informative reading is the contrast: `route_stat` carries the whole effect (removing it costs +0.009 mixed / **+0.058 roundrobin**, the latter ≥ the 0.02 attribution cost; Welch Δ CIs [−0.053, 0.071] / [−0.040, 0.155]) while `hot_young` contributes **nothing** measurable (Δ = +0.0002 mixed / −0.0004 roundrobin; CIs straddle 0). No simplification to a passing config exists because no config passes bar 2.
- **Bar-1-fail consequences:** not triggered (gate passed); levers stay default-off as implemented.

## Blocker analysis (the doc's required ledger) — monolithic-ceiling hypothesis **CONFIRMED at n=5**

Mean per-expert training fraction (route_log share; per-seed raws in the artifacts):

| arm | stream | mean fraction per expert (e0…e7) | experts trained (mean) | recruited per seed |
|---|---|---|---|---|
| shipped | mixed | [0.9996, 0.0004, 0, 0, 0, 0, 0, 0] | 1.6 | 1,2,1,2,2 |
| shipped | roundrobin | [0.9987, 0.0013, 0, …] | 2.0 | 2,2,2,2,2 |
| **+fixes** | **mixed** | **[1.0000, 0, 0, 0, 0, 0, 0, 0]** | **1.0** | **1,1,1,1,1** |
| **+fixes** | **roundrobin** | **[1.0000, 0, 0, 0, 0, 0, 0, 0]** | **1.0** | **1,1,1,1,1** |
| −route_stat | mixed | [0.9996, 0.0004, 0, …] | 1.6 | 1,2,1,2,2 |
| −route_stat | roundrobin | [0.9354, 0.0404, 0.0233, 0.0009, 0, …] | 2.4 | 4,2,2,2,2 |
| −hot_young | mixed | [1.0000, 0, …] | 1.0 | 1,1,1,1,1 |
| −hot_young | roundrobin | [1.0000, 0, …] | 1.0 | 1,1,1,1,1 |

Reading (consistent across all 5 seeds; per-seed route logs in `raw.json`):

1. **The fix moves the system FROM "one expert catch-alls without training" TO "one expert trains on 100% of the stream"** — `sample_top` fixes the frozen-after-warmup failure (shipped: e0 trains 99.96% and extra recruits get only their 256-sample warmup) but lands exactly on the **monolithic-learner ceiling**: interleaved ACC 0.597/0.592 ≈ the exploratory ~0.60, far from block-stream 0.834.
2. **`hot_young` is inert at the claimed config** (Δ ≈ 0.000 both streams); where it does recruit (−route_stat, roundrobin seed 10: 4 recruits) it recruits *immature* experts that still route 93.5% of samples to e0 and **hurt** ACC (0.534 arm mean). Recruitment quantity is not the binding constraint — training granularity is.
3. Per-domain test purity is 1.0 trivially in these runs because only one expert is ever trained — inference-side per-sample routing (`predict_logits`) already exists; there is nothing worth routing to.

**What the fusion flagship actually needs next (mechanism class):** not threshold/statistic sensitivity (dynamic vigilance hurt; freeze veto ACC-neutral — exploratory §3, unchanged here) but **per-sample / expert-stream training granularity**: `train_batch` applies the whole batch's local delta to the single active expert (the batch-level training step in `src/prizma.py`), so a mixed batch can never be split across experts regardless of the recognition statistic. Candidate classes, each requiring its OWN pre-registration (doc §7): per-sample routed delta steps (route then train per sample/sub-batch), session/recruitment training windows, replay-before-freeze variants. Per doc §7, PR-LM-1 stays **BLOCKED on the router**; the n=5 numbers above are the documented blocker to be appended to the fusion design spec §3.3 discussion and the C4 follow-up list (maintainer action — this report does not edit the spec).

## Honest limits

Exactly the doc §8 list: toy-scale synthetic structured-permuted streams only; these two interleaved constructions are new exploratory streams; a negative here claims nothing about natural corpora, class-incremental settings, or pretrained backbones, and does not retract the shipped E1 result (0.841/FGT 0.000 reproduced at n=5).
