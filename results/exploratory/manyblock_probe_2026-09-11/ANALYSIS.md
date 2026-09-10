# manyblock probe 2026-09-11 — paired exploratory analysis (n=5)

NOT a claim (LANE-EXPLORATORY). Paired seeds 0-4 (0-1 first run; 2-4 fresh).
CASCADE = Lane-1 tissue multi-timescale lever at kappa=0.05, delta=0.10.
Recovery_B = bpc_B(postC) - bpc_B(postD) (higher = more recovery on the revisit).
Damage_C = bpc_B(postC) - bpc_B(postB) (lower = less B-damage from C).

- Historical canary (seeds 0-1 OFF reproduce the committed 2026-09-09 probe.json exactly): **PASS**

| seed | OFF recovery | CAS recovery | Δrec | OFF damage | CAS damage | Δdmg | OFF bpcB postB→C→D | CAS bpcB postB→C→D |
|---|---|---|---|---|---|---|---|---|---|
| 0 | +1.0828 | +1.0110 | -0.0719 | +0.9148 | +0.8326 | -0.0822 | 3.054→3.969→2.886 | 3.069→3.902→2.891 |
| 1 | +1.0649 | +1.0410 | -0.0239 | +0.9181 | +0.8747 | -0.0434 | 3.034→3.952→2.887 | 3.049→3.923→2.882 |
| 2 | +1.1662 | +1.1443 | -0.0219 | +0.9584 | +0.9304 | -0.0280 | 3.080→4.038→2.872 | 3.096→4.027→2.883 |
| 3 | +1.0717 | +1.0496 | -0.0221 | +0.9078 | +0.8712 | -0.0366 | 3.071→3.978→2.907 | 3.085→3.956→2.907 |
| 4 | +1.0267 | +1.0719 | +0.0452 | +0.8740 | +0.9051 | +0.0311 | 3.029→3.903→2.876 | 3.045→3.950→2.878 |

- **paired Δrecovery_B (CASCADE − OFF, n=5)**: mean -0.0189, 95% CI [-0.0707, +0.0329]
- **paired Δdamage_C (CASCADE − OFF, n=5)**: mean -0.0318, 95% CI [-0.0825, +0.0189]

Direction reproduction: damage lower in 4/5 pairs; recovery lower in 4/5 pairs.

Reading (exploratory, n=5): the frozen dose keeps a small, direction-consistent protective shift that SHRINKS on fresh seeds (n=2 deltas −0.048/−0.063 → pooled −0.02/−0.03 bpc) and every pooled CI straddles zero. This is ~1/10 of the registered schedule's damage effect (PR-16: +0.37). No claim is staked; the lever stays default-OFF. A future registration needs a dose/target design pass AND a powered n; the honest priors from these five paired seeds are small-effect.
