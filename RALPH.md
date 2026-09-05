# RALPH.md — self-driving backlog (ralph-loop skill)

Owner directive 2026-09-05: run the remaining program in infinite turns without waiting for
"devam". This file is the loop's only memory. Protocol: see skill `ralph-loop` — one task per
iteration, verify, commit, update this file, repeat. Stop conditions: only-blocked backlog,
03:55 clock rule (nightly 03:58 shutdown is planned — never abort), user interrupt.

Standing rules: claims execute frozen protocols VERBATIM; registration precedes claim runs;
exploratory stays lane-exploratory; suite must end green every iteration; GPU-blocked tasks
wait with exact protocol pointers.

## BACKLOG

1. **[PR-03 claim] Single-pass citation bar** — status: `done (commit: see git log "PR-03 CLAIMED", 2026-09-05)`
   - Executed VERBATIM (seeds 10-19, 8 arms, single-pass). **VERDICT: PASS** — B*=OnlineEWC(online),
     diff +0.070, Welch 95% CI [+0.024, +0.117] (also excludes 0 → significant advantage);
     FGT 0.192 vs 0.454. INDEX → CLAIMED. Artifacts: results/citation_battery_PR-2026-09-03-03/.
   - Honesty note: initial verdict script had a t_isf tail-convention bug (zero-width CI);
     fixed before verdict accepted; raw records untouched by the bug (disclosed in INDEX).
   - Runner gained additive `seed_start`/`lane_label` params (default behavior unchanged).

2. **[PR-05 claim] Bounded-M economy** — status: `pending`
   - Protocol: docs/EXPERT_ECONOMY.md §5 (VERBATIM): shipped E1 settings; bounded arm =
     recruit-by-eviction M_max=5 (§3.3 policy) vs unbounded M=8; ≥10 seeds (fresh 3–12);
     PASS iff |ΔACC| ≤ 0.02 and |ΔFGT| ≤ 0.02; eviction count reported.
   - Runner: extend/reuse experiments/expert_economy.py (InstrumentedPrizma) into
     results/expert_economy_PR-2026-09-03-05/.
   - INDEX: PR-05 → ANALYZED with outcome.

3. **[PR-07' prereg+claim] Block-drift BAR-6′** — status: `pending`
   - Write + freeze docs/preregistry/2026-09-05-blockdrift-bar6.md (id PR-2026-09-03-07,
     lane CLAIM): sequential 3-corpus char-level continual stream (text8 → held-out slice →
     third corpus), no labels/boundaries/replay; arms {Prizma-LM toy (seq mixer + shipped
     router levers train_granularity+probation as needed), frozen-checkpoint control,
     memory-matched sliding-window control}; n=5 fresh seeds; bars: retention FGT ≤ 0.05 on
     corpus A after C, adaptation gain on C ≥ 0.10 BPC vs frozen control, E1-style home
     guard. Maintainer registers in INDEX, commits, THEN claim run executes.
   - NOTE (honest): streams at toy scale; corpus C needs a third committed text corpus
     (retain raw artifacts per docs/RETENTION.md).

4. **[PR-04 claim rerun] Analog robustness** — status: `pending` (~2.5–4 h CPU)
   - Protocol: results/analog_probe_2026-09-03/RESULTS.md §7 (VERBATIM): per-arm LR sweep,
     matched-clean gate, n=5, Δret ≥ +0.05 at 4-bit (Welch/Holm). Runner
     seq/analog_probe.py extended to the registered grid →
     results/analog_probe_PR-2026-09-03-04/.
   - Can run in BACKGROUND while other iterations proceed (accuracy-only, no timing claims).

5. **[exploratory] Toy Prizma-LM fusion proof-of-life** — status: `pending`
   - Prizma-Seq backbone (2-layer d=64) + vigilance-routed expert FFN (train_granularity +
     probation machinery) on a block-drift 2-corpus char stream; measure retention/adaptation;
     lane-exploratory only; motivates the real PR-LM-1 registration (GPU).

6. **[exploratory] PR-02 ε-fit at tiny D** — status: `pending`
   - CPU down-scale of the crosstalk capacity-law fit (D∈{16,32} MQAR grid, n=2) to sanity-
     check N* prediction direction BEFORE the GPU-tier registered grid. Lane-exploratory.

7. **[blocked: needs GPU] PR-01 surprise ablation** — frozen protocol
   (docs/preregistry/2026-09-03-surprise-gating-powered-ablation.md), ~10–15 A100-h.
8. **[blocked: needs GPU] Tier-0 repairs** — clean n=10 recall gate (~28h), init-fix reruns
   (~18h), B4 closure (~8h), GLA/Mamba-2 landscape (~60h). Order per synthesis §6.
9. **[blocked: needs GPU] Kernel decision session** — report 08-P1, bar ≤1.5× TF step time.
10. **[blocked: needs GPU] PR-LM-1** — gated by PR-07′ outcome.

## LOG

- (append per iteration: time, task, outcome, commit)
