# CONTAMINATION NOTICE — `recall_gate.json` (campaign 2026-06-08)

**Status: this artifact is QUARANTINED. Do not cite the recall-gate numbers, the variance, the
confidence intervals, the per-cell `params`, or the "bimodal Transformer baseline" reading that was
drawn from them.**

`results/campaign_2026-06-08/recall_gate.json` is presented in the report as a powered n=10 A100 run
at scale `d128L2H4` with the candidate's feature map `quad2_lowrank`. It is not. It is a mixture of
two different experiments that were written to the same file.

The results JSON, the per-seed accuracies, and every other file in this directory are left exactly as
they were produced. Nothing has been re-run, deleted, or replaced. This notice is the correction.

---

## What happened

`seq/recall_gate.py` resumed a crashed run by skipping any seed already present in the results JSON.
The skip was keyed on **the seed number alone** — it carried no record of the configuration that had
produced the cached number:

```python
for s in seeds:
    sk = str(s)
    if sk in cell["seeds"] and "best" in cell["seeds"][sk]:
        continue          # <-- keyed on the seed, not on the config
```

An earlier `--smoke` run — a deliberately tiny CPU/MPS plumbing check — had written seeds 0 and 1 to
the same file. When the powered campaign ran, it found seeds 0 and 1 already present in every cell
and skipped them, adopting the smoke's numbers as its own.

The smoke configuration (`results/recall_gate.json`, `"smoke": true`) differs from the campaign's on
every axis that matters:

| | smoke run | campaign (as reported) |
|---|---|---|
| scale | `d64L2H2` | `d128L2H4` |
| candidate feature map | `feat_map: "none"` (the novel lever **OFF**) | `feat_map: "quad2_lowrank"` |
| LR grid | `[0.001, 0.002]` | `[0.0005, 0.001, 0.0015, 0.002, 0.003]` |
| solve threshold | 0.5 | 0.9 |
| seeds | 0–1 | 0–9 |

The seed-0 and seed-1 records in `campaign_2026-06-08/recall_gate.json` are **byte-identical** to the
smoke's (verified: 20/20 records across all 10 cells).

## What is affected

**All 10 cells. Every one.** In each, seeds 0–1 are the smoke's ~4×-smaller model and seeds 2–9 are
the campaign's model:

| Cell | reported `params` | seeds 0–1 (smoke) | seeds 2–9 (campaign) |
|---|---|---|---|
| `MQAR-HARD-FLIP.TF-big` | 404,096 | 404,096 | 1,701,120 |
| `MQAR-HARD.TF` | 101,696 | 101,696 | 461,440 |
| `MQAR-HARD.Prizma` | 102,728 | 102,728 | 464,016 |
| `MQAR-HARD.Hybrid` | 102,212 | 102,212 | 462,728 |
| `INDUCTION.TF` | 99,648 | 99,648 | 400,000 |
| `INDUCTION.Prizma` | 100,680 | 100,680 | 402,576 |
| `INDUCTION.Hybrid` | 100,164 | 100,164 | 401,288 |
| `SELECTIVE-COPY.TF` | 99,648 | 99,648 | 400,000 |
| `SELECTIVE-COPY.Prizma` | 100,680 | 100,680 | 402,576 |
| `SELECTIVE-COPY.Hybrid` | 100,164 | 100,164 | 401,288 |

Wall-clock confirms it independently: e.g. `MQAR-HARD.TF` seeds 0–1 took 2.4 s and 2.6 s; seeds 2–9
took 211–424 s.

**The `params` field reported for every cell is the contaminated (smoke) value** — it is read from
seed 0. The parameter-match claim computed from these numbers is a claim about the wrong models.

### The LR sweeps are contaminated too

This was not found in the original report and is the more damaging half of the bug. `_train_arm`
cached its **stage-1 learning-rate sweep** under the same key-only rule, so the campaign inherited
the smoke's sweep in every cell as well. Every arm's learning rate in this campaign was therefore
chosen:

* on the **4× smaller** smoke model, not the model that was actually trained, and
* over the smoke's **2-point** grid `[0.001, 0.002]`, not the 5-point grid `[0.0005, 0.001, 0.0015,
  0.002, 0.003]` that this file's own `meta.lr_grid` declares.

So the per-model LR-fairness protocol this campaign claims to have run was, in fact, not run at all.
This affects **all 10 seeds of every cell**, not only seeds 0–1.

## What this invalidates

1. **The reported per-cell `params`** — wrong model, in every cell.
2. **The variance, the standard deviations and the confidence intervals** — the spread is inflated by
   an artefact. Two of the ten "seeds" in each arm are a different, much smaller model; their low
   scores are a capacity result, not seed noise. The wide CIs that failed TOST on all three legs are
   therefore a pipeline artefact, not a measured property of the arms.
3. **The published interpretation of the induction leg.** The report said the Transformer baseline is
   *"high-variance … on induction TF is bimodal: ~half its seeds collapse to ~0.06."* That reading
   was drawn from a contaminated array. Precisely:
   * 5 of 10 reported TF seeds score below 0.5 — but **2 of those 5 (seeds 0, 1) are the smoke's
     99,648-param model**, which had roughly a quarter of the capacity. They are not evidence about
     the baseline that was supposedly under test.
   * Among the 8 uncontaminated seeds, 3 do collapse (seeds 2, 4, 6 at 0.066 / 0.065 / 0.064). So a
     bimodal failure mode is **visible in the valid seeds and is probably real** — but the "~half"
     rate is inflated by the bug, and the CI built on it is not trustworthy.

   Stated plainly: a caching bug was partly misdiagnosed as a property of the baseline.
4. **The LR-fairness claim for this campaign** (see above).

## What this does *not* invalidate — and the direction of the error

The error was **conservative**. It made the reported result *worse* for the candidate, not better:

* The contamination widened the CIs, which made TOST equivalence *harder* to certify. All three legs
  are recorded as FAILING the parity gate, and the honest claim was downgraded to "competitive".
* The candidate's own seeds 0–1 ran with its **novel lever switched off** (`feat_map: "none"`), which
  can only have hurt the candidate's numbers.

This is an under-claim caused by a bug, not an over-claim. That mitigates the harm; it does not make
the numbers usable.

## What was NOT done

**No re-run, and no replacement numbers.** A clean version of this campaign needs roughly 28
A100-hours, which is not available. Inventing, extrapolating or back-filling the missing cells would
be worse than having no result. The honest outcome is a disclosed, quarantined artifact, and that is
what this is.

## The fix in the code

`seq/gpu_harness.py` now provides `config_fingerprint()` and `cached_cell_is_reusable()`. The resume
skip in `seq/recall_gate.py::_train_arm` (and in `gpu_harness.sweep_then_seeds` / `run_cell` for
callers that opt in) is keyed on **(seed, config fingerprint)**. A cached cell — or a cached LR sweep
— is reused only when it records the fingerprint of the configuration being asked for. A cell with no
fingerprint is treated as unverifiable and recomputed, because silently trusting unlabelled cached
results is precisely the failure being fixed. `_train_arm` additionally asserts that all aggregated
seeds share one fingerprint and one parameter count, so this class of mixture now crashes the run
instead of being published.

Regression tests: `tests/test_recall_gate.py::test_smoke_run_cannot_poison_a_later_campaign_at_a_different_config`
reproduces exactly this scenario (a small smoke run followed by a bigger campaign against the same
file) and fails against the old key-only logic;
`test_resume_at_the_same_config_still_skips_cached_seeds` pins that legitimate crash-resume still
works.
