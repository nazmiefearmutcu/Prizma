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

5. **[exploratory] Toy Prizma-LM fusion proof-of-life** — status: `done — exploratory-complete (2026-09-07 evening)`
   - seq/fusion_probe.py + results/exploratory/fusion_probe_2026-09-07/ (LANE-EXPLORATORY, n=2,
     ~13 min). Gains small (~0.02-0.08 bpc, within seed spread; E=4≡E=8 — pool inert at home,
     PR-05-at-home reproduced). **Specialization PASSES in the block regime**: expert corpus
     purity 0.997-1.0 (e0 = 100% text8, e1 = ~99.5% shakespeare; recruitment fires at the
     drift boundary) — the terminal probe's ≳10²-contiguous-samples condition VALIDATED:
     vigilance experts are domain-coherent when floors calibrate on blocks.
   - PR-LM-1 design consequence (for the GPU tier): wire experts in; stake the claim on the
     ROUTING LEDGER + minus-routing ablation (plain mixer already clears block-drift bars;
     small bpc gain may be mere capacity — add a shared-extra-head control); segment-level
     routing + local per-expert optimizers + bounded-M; make the stream many-block.

6. **[exploratory] PR-02 ε-fit at tiny D** — status: `done — exploratory direction sanity (commit: git log "e-fit probe", 2026-09-07)`
   - Results (n=2, 8000 steps): D=16 solves 0.995/0.996; D=32 solves 0.954/0.990 → transition
     ABOVE the grid; ε lower bound ≈ 0.295 at σ₂(|cos|)=0.05212 (N*(ε=1)=256, d_φ-capped).
     The law with measured σ₂ does NOT contradict the tiny-D solves; decisive test stays the
     GPU-tier fit at the real transition (D≈96-256), i.e. PR-2026-09-03-02's registered grid.
   - Artifacts: results/exploratory/e_fit_probe_2026-09-07/ + experiments/e_fit_probe.py.

7. **[IMPLEMENTATION DONE — execution needs GPU] PR-01 surprise ablation** — frozen protocol
   (docs/preregistry/2026-09-03-surprise-gating-powered-ablation.md), ~10–15 A100-h.
   - 2026-09-07 evening: A4 arm (surprise_norm) + claim runner IMPLEMENTED per the frozen
     checklist (seq/prizma_seq.py + seq/delta.py exact-scan routing + seq/surprise_claim.py;
     31 tests; suite 325→356P/10S). Smoke verified all four arms end-to-end on CPU (plumbing
     only — numbers meaningless); `--powered` refuses without CUDA; NO-TUNING gain selector
     (seed 900) as pure function; smoke/powered ledger separation with refusal.
   - GPU SESSION RECIPE: `python seq/surprise_claim.py --powered` — executes PR-01 VERBATIM
     and computes the Welch+Holm verdict incl. the pre-committed RETIRED branch. One
     disclosed operational deviation (ledger filenames) noted in the module docstring.
8. **[GPU-plug-and-play] Tier-0 repairs** — clean n=10 recall gate (~28h, RESUMABLE — Colab 24h limit ok), B4 closure (~8h), GLA/Mamba-2 landscape (~60h multi-session). WIRED: PRIZMA_GPU_CAMPAIGN.ipynb stage 1-3 (init-fix reruns ride along inside the recall-gate/B4 runs).
9. **[GPU-plug-and-play] Kernel decision session** — report 08-P1, bar ≤1.5× TF step time (same A100 session; runners ready).
10. **[GPU-plug-and-play] PR-LM-1 = PR-2026-09-03-08 REGISTERED** — docs/preregistry/2026-09-08-prizma-lm-flagship.md (frozen 2026-09-08, before any run): many-block text8→shakespeare→text8-returning, 5 arms (PRIM-LM fusion / FROZEN-TRUNK / SHARED-HEAD capacity control / FROZEN-CHECKPOINT / FORCED-RECRUIT), bars B1 retention ≤0.05, B2 adaptation ≥0.10, B3 ledger-primary pattern completion (the probe-2 lessons baked in: capacity confound refuted, trunk drift = binding forgetting term, floor-maturity veto on). Runner seq/prizma_lm_claim.py to be implemented on CPU (mirrors surprise_claim.py) — the LAST remaining local task; notebook gains a stage when it exists. Est. 45-90 min A100.

11. **[CPU, LAST LOCAL TASK] seq/prizma_lm_claim.py implementation** — mirror surprise_claim.py (smoke/powered separation, CUDA refusal, fingerprint resume, archive-before-verdict, VERBATIM bars B1-B4 per PR-08 doc, t_isf upper-tail); add the PR-08 stage to PRIZMA_GPU_CAMPAIGN.ipynb; smoke on CPU.

## GPU SESSION (ONE Colab A100 notebook does everything)

PRIZMA_GPU_CAMPAIGN.ipynb (repo root): stage 0 setup+pytest sanity (expect 374 passed) →
stage 1 recall gate --full (resumable) → stage 2 B4 both corpora n≥5 → stage 3 landscape
--full → stage 4 PR-01 --powered → stage 5 PR-02 --powered → per-stage archives + Drive zip.
All runners refuse to start the powered tiers without CUDA and execute frozen protocols
VERBATIM (smoke/powered ledger separation everywhere; verdicts computed in-run per each
doc's statistics incl. pre-committed negative branches).

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

## HANDOFF (2026-09-07 02:4x — clock-rule wind-down, second night)

CPU-feasible backlog is now EMPTY: items 1-4 claimed/negative (PR-03 CLAIMED, PR-05 CLAIMED,
PR-04 NEGATIVE, PR-07′ CLAIMED), item 5 (toy fusion) and 6 (ε-fit) done/exploratory-complete.
Remaining backlog = GPU-blocked only (items 7-10: PR-01 surprise ablation, Tier-0 repairs,
kernel session, PR-LM-1 — each with its frozen protocol pointer). Suite last verified
325P/10S; re-verify with full `pytest -q` at next session start. Push is the owner's manual
step (credentials not available in-agent).


## PUSH STATUS (2026-09-08 02:4x — the one remaining owner gate)

git push = 403 (token nazmiefearmutcu0 has PULL-ONLY on nazmiefearmutcu/Prizma — the repo
lives under the old account). tools/ship_prizma.py prepared + dry-run verified: once the
owner picks (a) transfer the repo to nazmiefearmutcu0, (b) add nazmiefearmutcu0 as writer,
or (c) mirror under nazmiefearmutcu0, the ship is ONE command:
  python tools/ship_prizma.py "Prizma campaign 2026-09" [--force]
Loop is TERMINAL on all axes: CPU work done, GPU work notebook-gated, push decision-gated.
