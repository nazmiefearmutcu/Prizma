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

2. **[PR-05 claim] Bounded-M economy** — status: `done (commit ce7826b, 2026-09-06)`
   - Executed VERBATIM (seeds 3-12, E1 settings, unbounded M=8 vs bounded M_max=5 Policy A).
     **VERDICT: PASS** — ΔACC = ΔFGT = +0.000000 on all 10 seeds; 0 evictions (recruit #5
     lands in slot 4 < M_max); the cap is INERT at home — registered-grade. m_max lever
     shipped default-off + bit-identity-tested (10 tests). INDEX → CLAIMED.
   - Artifacts: results/expert_economy_PR-2026-09-03-05/. Suite 315→325P/10S.

3. **[PR-07′ prereg+claim] Block-drift BAR-6′** — status: `done — CLAIMED (PASS) (commit 4b9b737, 2026-09-07)`
   - Executed VERBATIM (+ pre-run Addendum 2026-09-07: RESET=canary, LR per arm-family,
     partial order). **VERDICT: PASS** — FGT_A mean −0.189 (NEGATIVE forgetting: B-training
     improved A-eval; CI upper −0.119 ≤ 0.05); adaptation 2.954 vs FROZEN 6.132 (diff −3.178,
     Welch CI [−3.420, −2.936]); RESET canary bit-matches STREAM; WINDOW-TF descriptive 3.112
     (competitive; STREAM leads). Flagship block-drift prerequisite CLEARS at toy scale.
   - Artifacts: results/blockdrift_PR-2026-09-03-07/ (corpora on disk, sha256 committed).
   - INDEX → CLAIMED. Runtime ~9 min (2-3h estimate was 15× conservative).

4. **[PR-04 claim rerun] Analog robustness** — status: `done — NEGATIVE (commit ce7826b)`
   - Executed VERBATIM (n=5, matched-clean gate forced D=24). **VERDICT: NOT-SUPPORTED** —
     4-bit Δret = −0.0923, Welch CI [−0.170, −0.014] (delta significantly WORSE); noise
     primary parity. H2b recorded; "delta rule is analog-robust" RETIRED (docs/Prizma.md §5
     addendum). INDEX → NEGATIVE. Artifacts: results/analog_probe_PR-2026-09-03-04/.

5. **[exploratory] Toy Prizma-LM fusion proof-of-life** — status: `pending`
   - Prizma-Seq backbone (2-layer d=64) + vigilance-routed expert FFN (train_granularity +
     probation machinery) on a block-drift 2-corpus char stream; measure retention/adaptation;
     lane-exploratory only; motivates the real PR-LM-1 registration (GPU).

6. **[exploratory] PR-02 ε-fit at tiny D** — status: `in-progress (launched 01:5x, 2026-09-07)`
   - experiments/e_fit_probe.py: D∈{16,32}, quad2 d_h=32/d_φ=256, n=2 seeds, 8000 steps,
     4 threads (claim run finished — no contention). σ₂(|cos|)=0.05212 measured →
     N*(ε=1)=256 (d_φ-capped). Crash-safe ledger: results/exploratory/e_fit_probe_2026-09-07/.
     If the 03:55 clock cuts it, partial per-run records are the exploratory deliverable.

7. **[blocked: needs GPU] PR-01 surprise ablation** — frozen protocol
   (docs/preregistry/2026-09-03-surprise-gating-powered-ablation.md), ~10–15 A100-h.
8. **[blocked: needs GPU] Tier-0 repairs** — clean n=10 recall gate (~28h), init-fix reruns
   (~18h), B4 closure (~8h), GLA/Mamba-2 landscape (~60h). Order per synthesis §6.
9. **[blocked: needs GPU] Kernel decision session** — report 08-P1, bar ≤1.5× TF step time.
10. **[blocked: needs GPU] PR-LM-1** — gated by PR-07′ outcome.

## LOG

- 2026-09-05 22:0x — loop armed: ralph-loop skill installed (~/.claude/skills/ralph-loop/),
  RALPH.md created, PR-03/PR-05 registered (INDEX), commit 09b5936 precursor.
- 2026-09-05 22:1x–22:3x — iter-1 PR-03: battery seeds 10-19 executed; verdict-script bug
  (t_isf tail) caught + fixed BEFORE verdict accepted; PASS (CI [+0.024,+0.117]). Commit 09b5936.
- 2026-09-05 22:3x–02:46 — iter-2 PR-05 (agent) + iter-4 PR-04 (agent, background): PASS /
  NEGATIVE. Suite 325P/10S. Commits ce7826b (+09b5936). m_max + replay + probation levers
  landed earlier (33f9d09, 7ce8867, 2e1b4ec).
- 2026-09-06 02:52 — PR-07′ REGISTERED (doc frozen, INDEX row); claim run deferred past the
  03:55 clock rule. HANDOFF: next session (or loop resume) = write seq/blockdrift_claim.py
  per doc §6, execute VERBATIM, INDEX PR-07′ → outcome, then backlog items 5-6 (exploratory).

## HANDOFF (2026-09-06 02:55 — clock-rule wind-down)

Suite last verified 325 passed + 10 skipped (agent run); re-verify with a full `pytest -q`
at session start. Working tree clean after the final commit. Resume point: RALPH.md task 3
(PR-07′ claim run — registration committed, nothing else needed first). Everything else on
the backlog is exploratory (5-6) or GPU-blocked (7-10).
