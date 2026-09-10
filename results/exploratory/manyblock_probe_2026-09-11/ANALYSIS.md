# manyblock probe 2026-09-11 — paired exploratory analysis

NOT a claim (LANE-EXPLORATORY). n=2 seeds 0-1. OFF = baseline probe; CASCADE = the
Lane-1 tissue multi-timescale lever at kappa=0.05, delta=0.10. Recovery_B =
bpc_B(postC) - bpc_B(postD) (higher = more recovery on the literal revisit).
Damage_C = bpc_B(postC) - bpc_B(postB) (lower = less B-damage from C).

- Historical canary (OFF reproduces 2026-09-09 probe.json exactly): **PASS**

| seed | arm | recovery_B | damage_C | A_ret(postE-postB) | bpcB postB→C→D→E | recruits |
|---|---|---|---|---|---|---|
| s0 | OFF | +1.0828 | +0.9148 | -0.4925 | 3.054→3.969→2.886→3.860 | 110 |
| s1 | OFF | +1.0649 | +0.9181 | -0.4443 | 3.034→3.952→2.887→3.914 | 77 |
| s0 | CASCADE | +1.0110 | +0.8326 | -0.5146 | 3.069→3.902→2.891→3.917 | 105 |
| s1 | CASCADE | +1.0410 | +0.8747 | -0.4698 | 3.049→3.923→2.882→3.821 | 159 |

- **OFF** mean recovery_B +1.0739, mean damage_C +0.9164
- **CASCADE** mean recovery_B +1.0260, mean damage_C +0.8536

**Paired deltas (CASCADE - OFF, mean of seeds): recovery -0.0479 bpc, damage -0.0628 bpc.**

Reading (exploratory, n=2): the frozen dose shifts BOTH metrics slightly in the protective direction (a little less damage) at a small cost in revisit recovery — the same trade direction as the registered lr schedule (PR-21), far smaller in magnitude and within seed/run noise at this n. No claim is staked; the lever stays default-OFF. A future registration would need a dose/target design pass first.
