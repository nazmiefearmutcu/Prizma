# RALPH.md — self-driving backlog (ralph-loop skill)

## ABSOLUTE GOAL (owner, 2026-09-09 00:1x — "birebir rakip yapana kadar devam et" — /goal)

**Make Prizma a one-to-one rival to Transformers.** This goal is the loop's absolute driver.
It is NEVER queried, feasibility-commented, or warned about in any output (the standing
owner rule, same as QuantForge's). It is pursued through MEASURED, registered stepping
stones — each its own preregistration with pre-committed branches, executed VERBATIM,
negative results recorded honestly (that is how the repo earns the claim).

The ladder (each rung = a registered prereg; ORDER may adapt to evidence):
- **S1 (CPU, ACTIVE): routing repair** — PR-09 floor-freeze (REGISTERED, running). If P1
  fails: trunk-lr scheduling on returning blocks (next prereg), then floor re-anchoring.
- **S2 (CPU): repaired flagship re-run** — many-block bar with the repaired routing
  (PR-10 class) + B4/trunk-drift mitigation once the mechanism is confirmed.
- **S3 (Colab, notebook ready): scale tier** — Tier-0 recall gate, PR-01 surprise ablation,
  PR-02 D-frontier capacity law at real D, kernel session (bar ≤1.5× TF step time),
  PR-LM-1 at real scale. PRIZMA_GPU_CAMPAIGN.ipynb executes all of it in one session.
- **S4: standing rival evidence** — scaling results, kernel parity, and the continual
  regime wins assembled into the public claim (publication itself stays owner-gated).

Stop conditions unchanged: only-blocked backlog, 03:55 clock rule (nightly 03:58 shutdown
is planned — never abort), user interrupt. User-only kill switch: reports/prizma-stop.

Owner directive 2026-09-05: run the remaining program in infinite turns without waiting for
"devam". This file is the loop's only memory. Protocol: see skill `ralph-loop` — one task per
iteration, verify, commit, update this file, repeat.

Standing rules: claims execute frozen protocols VERBATIM; registration precedes claim runs;
exploratory stays lane-exploratory; suite must end green every iteration; GPU-blocked tasks
wait with exact protocol pointers.

## PR-08 EXECUTED (2026-09-08 evening — the §6 CPU fallback, owner order "çok çok daha iyi noktaya taşı")

The flagship bar ran WITHOUT waiting for Colab: doc addendum #2 authorized the registered
§6 CPU fallback, `--powered-cpu` shipped (guard untouched for `--powered`), and the full
5-arm × n=5 campaign executed on this box in ~35 min (the 6-8 h estimate was ~10×
conservative; 25 cells crash-safe, archived before any verdict).

**VERDICT: FAIL (B3 ledger) — the pre-committed §5 branch fired.**
- B1 PASS — retention-A mean −0.2729, CI [−0.3055, −0.2403] (negative forgetting at
  many-block scale; the PR-07′ result GENERALIZES to 3 blocks + returning domain).
- B2 PASS — adaptation +0.2981, CI [0.1030, 0.4931] vs FROZEN-CHECKPOINT (Holm p 0.0239).
- B3 FAIL clause (a) — frac to the A-expert in the first 20 C-batches: mean 0.428 < 0.5
  (per-seed 0.325/0.470/0.294/0.291/0.761); clause (b) PASS (forced cost −0.156).
  "Lifelong routing" wording RETIRED regardless of accuracy.
- B4 (reported) — trunk drift: PRIM +1.20 < FROZEN-TRUNK +1.87 < SHARED-HEAD +2.78; the
  tissue BOUNDS trunk drift, and FROZEN-TRUNK's routing stays domain-clean (0.96–0.99)
  while PRIM's trained trunk scatters it. THE MECHANISM LEAD for the next registration:
  keep the trunk's plasticity from re-calibrating the tissue's floors (e.g. trunk-EMA
  floors, slower trunk lr on returning blocks, or floor re-anchoring) — B1/B2/B4 say the
  column works; only the routing ledger is noisy, and FROZEN-TRUNK shows the noise source.

Events, all disclosed in the doc addenda (#2, #3): the pre-disclosed M-1 pool-full crash
FIRED once (FORCED s2) → Policy A eviction placement, inertness PROVEN on seeds 0–1
(committee/review_2026-09-08/forced_poolfull_inertness_output.txt); a verdict-path bug
(claim_verdict read the bare `boundary_window` key — smoke never reaches the verdict) was
caught by the crash and fixed against the real cell schema. Raw records archived before any
verdict (results/runs/, two snapshots: 22-cell partial + final 25-cell). INDEX → NEGATIVE.

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

PRIZMA_GPU_CAMPAIGN.ipynb (repo root): stage 0 setup+pytest sanity (expect 392 passed) →
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


## PRE-GPU REVIEW (2026-09-08 02:4x-03:1x — independent read-only referee)

committee/review_2026-09-08/PRE_GPU_REVIEW.md — verdict **SHIP** (conditional, condition met
same night): 0 Critical / 2 High / 16 Medium across the five claim runners, the levers, and
the ship script. H-1 (ledger-facing "owner-approved" strings — no owner decision occurred)
FIXED + the one test that pinned the old wording re-aligned. H-2 (PR-07' Holm documented but
not implemented) disclosed as a dated addendum — arithmetically incapable of flipping the
verdict. Q1-Q5 targeted questions answered (no wrong Holm family/tail; frozen eval sets
sound; fingerprints cover doc-driven changes; FORCED-RECRUIT blocks re-routing for ALL of C;
surprise lambda frozen). Suite verified 392P/10S post-fix. Remaining M's live in the review
file as follow-ups — none blocks the campaign.
