# Pre-registration registry — INDEX

Registry of all pre-registrations under the two-lane policy in
[`POLICY.md`](POLICY.md). A LANE-CLAIM run may only start once its row is `REGISTERED` here and the
referenced document is committed. Statuses: `IN-WRITE` / `REGISTERED` / `ANALYZED` / `CLAIMED` /
`NEGATIVE` / `RETIRED` / `ABANDONED` / `AMENDED` (see POLICY.md for the lifecycle and the
immutability/addenda rule).

| id | date | lane | status | artifact | one-line claim/bar |
|----|------|------|--------|----------|--------------------|
| PR-2026-09-03-01 | 2026-09-03 | CLAIM | REGISTERED | [`2026-09-03-surprise-gating-powered-ablation.md`](2026-09-03-surprise-gating-powered-ablation.md) | Surprise-norm gating must beat BOTH constant and random controls at power (n≥5 seeds, Welch); otherwise the surprise mechanism is retired from README novelty claims (owner decision 4). Frozen 2026-09-03; runs when GPU budget returns. |
| PR-2026-09-03-02 | 2026-09-03 | CLAIM | IN-WRITE | [`../crosstalk_capacity_law.md`](../crosstalk_capacity_law.md) | Crosstalk-spectrum capacity law N*=min(1+ε²/σ₂², d_φ): fit ε at D∈{16,32,64}, freeze predictions, then the law must predict the MQAR solve/no-solve transition within ±1 grid step on ≥3 of 4 rungs D∈{96,128,192,256}; else the law is demoted to descriptive. |
| PR-2026-09-03-03 | 2026-09-03 | CLAIM | IN-WRITE | [`../CONTINUAL_CITATION_BAR.md`](../CONTINUAL_CITATION_BAR.md) | Single-pass domain-incremental: Prizma ACC within −0.05 of the best boundary-free regularizer (fresh seeds 10–19, Welch CI) for the CL headline to be citable in the single-pass regime; INCONCLUSIVE rule pre-committed. |
| PR-2026-09-03-04 | 2026-09-03 | CLAIM | IN-WRITE | [`../../results/analog_probe_2026-09-03/RESULTS.md`](../../results/analog_probe_2026-09-03/RESULTS.md) (§7 draft) | Analog robustness: at matched clean accuracy and n≥5, delta-mode retention under 4-bit state quantization must exceed additive by ≥ +0.05 (Welch, Holm) for the deployment-grade claim; exploratory direction was +0.10 (n=3, D=32). |
| PR-2026-09-03-05 | 2026-09-03 | CLAIM | IN-WRITE | [`../EXPERT_ECONOMY.md`](../EXPERT_ECONOMY.md) | Bounded-M_max=K expert economy with eviction achieves ACC/FGT within ±0.02 of unbounded growth on the shipped E1 stream (10 seeds) — the cap must be inert in the home regime; exploratory: 0 evictions at M_max=K. |
| PR-2026-09-03-06 | 2026-09-05 | CLAIM | REGISTERED | [`2026-09-03-interleaved-router-bar6.md`](2026-09-03-interleaved-router-bar6.md) | Interleaved-router BAR-6 toy check: frozen config hot_young=1.0 + route_stat=sample_top must reach ACC ≥ 0.70 on BOTH interleaved streams (mixed-batch, round-robin) at n=5 seeds 10–14, with no E1 regression (ACC 0.834±0.02, FGT ≤ 0.02); else the fix is declared insufficient and PR-LM-1 stays blocked. Exploratory (n=2, lane-exploratory): best 0.603/0.602 — the monolithic-learner ceiling; batch-level training granularity identified as the binding constraint. |

*Exploratory work (LANE-EXPLORATORY) does not get rows here by default; its results live under
`results/exploratory/` and never become claims. It gets a row only when it motivates a registered
pre-registration — and then the row points at the LANE-CLAIM document, not at the exploration.*
