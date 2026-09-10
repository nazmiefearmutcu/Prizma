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

**SINGLE-WRITER LOCK (collision safety between the hourly cron and live sessions):** before
ANY repo work, check logs/iteration.lock — if it exists and its mtime is younger than 45
minutes, EXIT this fire/session immediately (work already in flight). Otherwise create it
(content: session id + ISO timestamp) and DELETE it when the iteration/session ends. A claim
run left executing at wind-down is resume-safe; write the handoff, let 03:58 win.

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

## PR-09 EXECUTED (2026-09-09 01:0x — S1 rung, powered-cpu, ~28 min)

**VERDICT: NEGATIVE (P1 FAIL) — mechanistically decisive.** Floor freeze left boundary
routing near-identical to OFF (frac mean 0.405 vs 0.428; per-seed 0.325/0.470/0.167/0.291/
0.772 vs OFF 0.325/0.470/0.294/0.291/0.761) — the PR-08 scatter is NOT floor
re-calibration; it is the trunk's SHARED-WEIGHT drift (FROZEN-TRUNK's clean 0.96-0.99
routing = frozen weights). Guards held (G1 -0.2699 CI [-0.3030,-0.2368]; G2 +0.2994 CI
[0.1044,0.4943]); churn FELL (104 vs 179 recruits) without moving fracs. **CANARY PASS:
5 OFF cells bit-identical to PR-08 PRIM** (cross-process determinism re-proven). The
pre-committed branch fired: **S1b = PR-2026-09-03-10 trunk-lr scheduling REGISTERED**
(docs/preregistry/2026-09-09-trunklr-routing-repair.md — backbone lr x0.25 on C only;
OFF canary + TRUNK-LR-0.25C; same P1/G1/G2 bars). NEXT SESSION: implement
seq/trunklr_claim.py (mirror floorfreeze_claim.py; fingerprints include the C-lr) and
execute --powered-cpu, then INDEX + README + this file. Artifacts:
results/floorfreeze_PR-2026-09-03-09/ + official console log + archive under results/runs/.

## PR-10 EXECUTED (2026-09-09 18:5x — S1 rung COMPLETE, powered-cpu, ~13 min)

**VERDICT: CLAIMED (PASS) — the routing repair is real.** P1 PASS: frac mean 0.575 >= 0.5
(OFF 0.428; per-seed 0.259/0.689/0.631/0.608/0.686). G1 PASS (-0.2818, CI [-0.3078,
-0.2559]); G2 PASS (+0.3081, CI [0.1136, 0.5027], Holm p 0.0205). B4 (reported): trunk
drift +0.7209 vs OFF +1.2021 — best B4 of any arm measured. CANARY PASS (3rd independent
cross-process bit-identity proof). Mechanism ladder complete: interleaved=partition-bounded
(PR-06) -> floors exonerated (PR-09) -> trunk plasticity is the cause and the 0.25 C-dose
repairs it (PR-10). **S1 COMPLETE. NEXT = S2 (PR-2026-09-03-11, the REPAIRED FLAGSHIP):
PR-08 protocol VERBATIM with the registered lever wired into the C phases of PRIM-LM,
FORCED-RECRUIT and SHARED-HEAD (FROZEN-TRUNK/FROZEN-CHECKPOINT have no C backbone step);
bars B1-B4 as PR-08; the B3 ledger FAIL is expected to flip — a PASS would REINSTATE the
'lifelong routing' wording with evidence. Own ledger, own prereg; run --powered-cpu
(~35 min).**

## PR-11 EXECUTED (2026-09-09 19:3x — S2 rung, powered-cpu, ~31 min)

**VERDICT: FAIL (B3 clause (b)) — clause (a) REPAIRED, the value clause is the new binding
constraint.** B1 PASS (-0.2818), B2 PASS (+0.3081, Holm p 0.0205), B3 clause (a) PASS
(frac 0.575 — the PR-10 repair TRANSFERRED to the flagship), B3 clause (b) FAIL: forced
cost -0.1995 (killing pattern completion improves B-eval by 0.1995 > the 0.10 allowance).
Mechanism: PRIM's non-A C segments (42.5%) contaminate B-experts via argmin; FORCED
protects them. B4: PRIM +0.7209 best arm. CANARIES PASS x2 (FROZEN-TRUNK +
FROZEN-CHECKPOINT bit-identical to PR-08 — the lever's blast radius mechanically proven).
**CORRECTION (maintainer, 2026-09-09): my PR-08/PR-09 wrap-up prose said 'clause (b) PASS
(forced cost -0.156, costs nothing)' — the sign was misread; -0.1554 also VIOLATED clause
(b). OUTCOMES unaffected (clause (a) had failed independently).** **S2b LEAD (next prereg,
not yet registered): DOMAIN-EXCLUSIVE EXPERTS — C segments train only the A-expert or fresh
recruits, never B-committed experts; prediction: B3b flips (B-experts protected => PRIM_B
~ FORCED_B) while clause (a) holds.** Artifacts: results/prizma_lm_PR-2026-09-03-11/ +
results/pr11_repaired_flagship_official_console.log.

## PR-12 EXECUTED (2026-09-09 20:4x — S2b rung, powered-cpu, ~7 min)

**VERDICT: CLAIMED (PASS) — the contamination hypothesis is CONFIRMED and fixed.** P1 PASS:
PRIM-DE B mean 3.4124 vs FORCED 3.4124 (B4 +0.5213 == +0.5214 to 4 decimals — the B-expert
contamination is FULLY eliminated). P2 PASS: frac 0.976 (untreated 0.575). G1 -0.2638, G2
+0.2975 (Holm p 0.024). Per-cell A/B canaries PASS. **Disclosed side effect:** redirects=0;
in s2 the eviction-exemption collapsed the candidate pool to the a_expert slot (recycled
22x) — slot-index frac partially measures a C-warden slot; B-expert protection held in
EVERY seed (the P1 mechanism). **S2b COMPLETE. NEXT = S2c (PR-2026-09-03-13, the
EXCLUSION FLAGSHIP): the full 5-arm PR-08 protocol + exclusion lever + the eviction
refinement (when the only candidate is the a_expert, fall back to no-recruit/argmin —
preserve A-expert identity). Own prereg, 25 cells ~30 min. If B1-B3 all PASS there, the
'lifelong routing' wording is revisited WITH evidence.** Artifacts:
results/domainexc_PR-2026-09-03-12/ + official console logs.

## PR-13 EXECUTED (2026-09-09 night — S2c rung, powered-cpu, ~50 min; S2 COMPLETE)

**VERDICT: CLAIM PASS — ALL THREE BARS. The repaired flagship stands; 'lifelong routing'
REINSTATED WITH EVIDENCE.** B1 -0.2666 CI [-0.2956,-0.2376]; B2 +0.2994 CI [0.1046,0.4941]
(Holm p 0.0234); B3 frac 0.976 >= 0.5 AND forced_cost +0.0000 (PRIM B-eval == FORCED's);
B4 PRIM +0.5213 best arm. Both FROZEN canaries bit-identical to PR-08. **INCIDENT: the
first launch ran at a 10x CLI typo dose (0.0075 vs 7.5e-4) — INVALID, archived under
results/invalid_dose_archive/ (README explains; nothing deleted); the runner now REFUSES
unregistered doses (validate_trunk_lr_c, incident typo pinned as a test); the clean re-run
is the claim ledger.** Mechanism ladder, fully measured: interleaved=partition-bounded
(PR-06) -> floors exonerated (PR-09) -> trunk plasticity cause+repair (PR-10) ->
contamination cause+repair (PR-12) -> composition CLAIMED (PR-13). **S2 COMPLETE. NEXT =
S3 (GPU scale tier): PRIZMA_GPU_CAMPAIGN.ipynb on the owner's Colab — Tier-0 recall gate,
PR-01 surprise ablation, PR-02 D-frontier, kernel session, PR-LM-1 (designs inherit L1 +
L2 + the refinement). CPU-side, the next rung is optional polish; the program is
owner-GPU-gated from here.** Artifacts: results/prizma_lm_PR-2026-09-03-13/ +
results/pr13_exclusion_flagship_official_console.log + results/invalid_dose_archive/.

## S3-READINESS ITERATION (2026-09-09 22:3x — record consolidation, no new claims)

The notebook's sanity expectation was STALE (392 vs the actual 478) and its stage 6 still
pointed at the ORIGINAL PR-08 command. Updated: sanity -> 478 (2026-09-09), stage 6 ->
the REPAIRED flagship command (--trunk-lr-c 7.5e-4 --domain-exclusion, own
prizma_lm_PR-2026-09-03-13_gpu ledger, labeled CONFIRMATORY — the claim already stands on
PR-13's CPU ledger). PRIZMA_SEQ_REPORT.md gained the 09-09-night addendum (PR-09..PR-13 +
the sign-convention correction + the invalid-dose incident). Suite 478P/10S green.
**S3 remains owner-gated: PRIZMA_GPU_CAMPAIGN.ipynb on the owner's Colab (stages 1-5 are
the real payload; stage 6 confirmatory). CPU-side the registered program is complete
through S2; future CPU iterations = polish or NEW registrations only when a mechanism
lead justifies one.**

## MANYBLOCK PROBE (2026-09-09 23:0x — LANE-EXPLORATORY, n=2, ~8 min; PR-14 design lead)

seq/manyblock_probe.py, 5 blocks A=text8[0,1M) -> B=shakes -> C=text8[1.1M,2.1M) ->
D=shakes-REVISIT (the EXACT B slice) -> E=text8[2.1M,3.1M); L1 dose on every post-A
block; NO exclusion (measuring plain argmin). **Findings (both seeds agree):**
1. **The literal revisit WORKS under plain argmin**: B-eval ~3.96 post-C -> ~2.89 post-D
   (recovery -1.08 bpc, BELOW even the post-B level ~3.04) — the B-expert re-engages
   naturally on re-encounter. Seesaw pattern: leave B (+0.9), revisit (-1.08), leave again
   (+0.95) — fast, repeatable recovery.
2. **A-retention holds through 5 blocks**: A-eval 6.08 init -> 2.71 post-E (negative
   forgetting persists at many-block scale with L1 only).
3. **text8 accumulates**: Cret 2.85 -> 2.74 post-E.
4. High recruit/eviction churn continues (77-110 recruits) without breaking retention.
**PR-14 lead (register next): "many-block accumulation + true revisit" — 5-block stream,
bars frozen FROM THESE measurements: (i) revisit recovery >= 0.10 (measured ~1.08),
(ii) A-retention CI upper <= 0.05 through E (measured ~ -3.3), (iii) post-revisit B-eval
<= post-B level (second visit beats the first: 2.89 < 3.04), (iv) re-engagement ledger
(D-window frac to B-experts >= 0.5). Open design question: whether to add L2
domain-exclusion with per-block owner election, or run plain argmin (the probe suggests
argmin alone recovers on revisit; exclusion may still be needed to keep the recovery).
n=5, levers L1 (+L2 decision in the prereg), ~11 min CPU.** Artifacts:
results/exploratory/manyblock_probe_2026-09-09/ + console log.

## PR-14 EXECUTED (2026-09-09/10 night — S2.5 rung, powered-cpu, ~25 min)

**VERDICT: CLAIMED — ALL FIVE BARS. The lifelong claim now rests on 5 blocks with a true
revisit.** P1 revisit recovery +0.8195 (8x margin); P2 re-engagement frac 0.971; P3
A-retention -0.3226 (negative forgetting through 5 blocks); G1 harm -0.0083 (exclusion
costs nothing — marginally better than plain argmin); G2 text8 continuity -0.1173
(accumulating). PLAIN reproduced the exploratory probe BIT-FOR-BIT (3rd independent
cross-run determinism proof family). **INCIDENT (cosmetic only): two printer crashes AFTER
the verdict was ledgered (stale guard keys in _print_report) — fixed; claim cells never
recomputed.** **S2.5 COMPLETE — the CPU-side lifelong story is now: 5 blocks, 3 drift
boundaries, 1 true revisit, zero forgetting of A, full recovery of B on re-encounter, and
a re-engagement ledger that the routing machinery earns.** NEXT = S3 (owner Colab: Tier-0,
PR-01, PR-02, kernel, PR-LM-1 — designs inherit L1 generalized + L2 owner election +
refinement). Artifacts: results/manyblock_PR-2026-09-03-14/ + official console log.

## PR-15 EXECUTED (2026-09-10 00:5x — the first head-to-head control, powered-cpu, ~22 min)

**VERDICT: NEGATIVE (P1 FAIL) — the registered A-retention bar was NOT discriminating,
and the honest run found the real gap elsewhere.** The WINDOW-TF control holds A across
the 5-block curriculum (retention ~ -0.36, delta vs the column +0.0414 CI [+0.0087,
+0.0740] — marginally BETTER). Why: the curriculum re-exposes A's domain at C and E, so
both models are continuously rehearsed on text8 — A-retention cannot separate them here.
**The discriminating descriptives (measured, n=5 both arms): B-boundary drift damage
post-C: WINDOW-TF ~+1.15 bpc vs repaired column ~+0.60 bpc — the tissue HALVES boundary
damage.** Both recover on the literal revisit (TF postD 2.77). **PR-16 lead (register
next): the B-boundary drift-damage head-to-head — delta = TF damage - column damage, bar
CI lower >= 0.25 (half the measured ~0.5 gap), n=5, same reuse pattern.** LESSON for all
future controls: a retention bar only discriminates when the target domain does NOT recur
in the training stream. Artifacts:
results/windowtf_manyblock_PR-2026-09-03-15/ + official console log.

## PR-16 EXECUTED (2026-09-10 01:2x — the head-to-head ledger's first established gap)

**VERDICT: CLAIMED.** delta = +0.3685 (CONTROL damage [0.969, 0.994, 1.008, 0.940, 1.148]
vs COLUMN [0.601, 0.638, 0.671, 0.641, 0.666]); Welch CI [0.2704, 0.4666] — CI lower
0.2704 >= 0.25 (p_raw 0.0134). The tissue HALVES B-boundary drift damage vs the
memory-matched sliding-window control — established at claim grade over the registered
n=5 artifacts (PR-14 EX + PR-15 WINDOW-TF; post-hoc visibility disclosed, bar frozen at
half the point estimate). Runner seq/boundary_damage_claim.py (analysis-only, seconds).
**The comparison ledger vs the standing control: accumulation parity (PR-15: both hold
A), boundary damage 2x better (PR-16: established), revisit recovery both (descriptive).
Next head-to-head candidates (own preregs when a lead justifies): the B-drift gap at the
D/E boundaries, the window-matched TF WITH the tissue-style schedule, and the S3 GPU
tier.** Artifacts: results/boundary_damage_PR-2026-09-03-16/ + archive.

## PR-17 EXECUTED (2026-09-10 02:2x — the attribution control, powered-cpu, ~9 min)

**ATTRIBUTION: SCHEDULE-CARRIED (both CIs pre-committed, no cherry-picking).** C1
ESTABLISHED: the schedule alone reduces the plain TF's boundary damage by +0.3724 (CI
[0.2743, 0.4705]) — matching the column's PR-16 delta almost exactly. C2 ~0: SCHED-TF
damage == COLUMN damage (-0.0039, CI [-0.0444, +0.0365]). **CORRECTION to PR-16's
interpretation (dated): the boundary-damage protection is carried by the L1 LR SCHEDULE
(any model can adopt it), NOT the tissue. The tissue's established uniqueness = the
routing ledger (PR-13 B3: frac 0.976 + zero forced cost), specialization purity
(PR-12-era evidence), and the bounded-M economy (PR-05).** SCHED-TF also recovers on the
revisit (postC 3.74 -> postD 2.91). Runner: windowtf_manyblock gained the guarded
--post-a-lr/--ledger-dir flags (default None = byte-identical PR-15). **Next
head-to-head leads: the routing-ledger head-to-head (a plain TF has no ledger — the
tissue's home turf), the D/E-boundary audit, and S3 (owner Colab).** Artifacts:
results/windowtf_sched_PR-2026-09-03-17/ + official console log.

## NIGHT-CLOSE (2026-09-10 03:1x — clock-rule wind-down, records consolidated)

The 2026-09-09/10 campaign is COMPLETE through S2.5: PR-09 NEG (floors exonerated) ->
PR-10 CLAIMED (trunk-lr repair) -> PR-11 NEG (clause-b; flagging the leak) -> PR-12
CLAIMED (domain-exclusion) -> PR-13 CLAIM PASS ALL BARS (the flagship stands; 'lifelong
routing' reinstated) -> PR-14 CLAIMED ALL FIVE BARS (5-block accumulation + true revisit)
-> PR-15 NEG (the retention bar was non-discriminating — lesson recorded) -> PR-16
CLAIMED (boundary damage: the head-to-head gap) -> PR-17 SCHEDULE-CARRIED (the
attribution: the lr schedule protects, NOT the tissue; the tissue owns the routing
ledger). Registry: 12 CLAIMED / 6 NEGATIVE. Suite 504P/10S. Records consolidated:
PRIZMA_SEQ_REPORT.md carries all addenda; README's outcomes table is complete (PR-03..
PR-17); the notebook is current. **MORNING/NEXT-SESSION STATE: S3 = the owner's Colab
run of PRIZMA_GPU_CAMPAIGN.ipynb (stages 1-5 the payload; stage 6 confirmatory
repaired-flagship). CPU-side: no registered work pending; new registrations only when a
mechanism lead justifies one (leads: routing-ledger head-to-head design — structurally
tissue-only; D/E audit non-discriminating). Publication stays the owner's one decision
(tools/ship_prizma.py).** This file is current; good night.

## PR-18 EXECUTED (2026-09-10 night — the replication rung, powered-cpu, ~60 min over two legs)

**VERDICT: CLAIMED — REPLICATED at n=10 fresh (seeds 5-14); the flagship claim now stands
on 15 cumulative seeds.** The seeds-5-9 leg: B1 PASS (-0.268), B3 PASS (0.959/+0.0003),
B2 INCONCLUSIVE (mean +0.526 but a one-seed floor-variance CI [0.003, 1.049]) — the §5
pre-authorized extension (seeds 10-14) ran and PASSED alone (B2 +0.839, CI [0.627,
1.052]); the POOLED fresh adjudication (n=10): B1 -0.2759, B2 +0.6826 CI [0.4325, 0.9328]
(Holm p 0.0003), B3 0.965/+0.0001. LR-selection canary PASS (3e-3/3e-3/1e-2 exact);
structural FORCED==PRIM canary PASS; the FROZEN bit-identity canaries were RECORDED-SKIP
on fresh seeds (doc addendum #1). **INCIDENTS (cosmetic): two post-verdict printer
crashes (stale keys) + one planned-canary abort at FROZEN-TRUNK s5 (the fresh-seed scope
gap, fixed with the skip-record semantics) — claim cells never recomputed.** Runner:
prizma_lm_claim gained the guarded --seed-offset flag. **S2 is now REPLICATED-COMPLETE.
NEXT = S3 (owner Colab) — the CPU-side ladder has no pending registered work.**
Artifacts: results/prizma_lm_PR-2026-09-03-18/ (ledger + pooled_fresh_verdict.json) +
3 console logs.

## PR-19 EXECUTED (2026-09-10 night — the accumulation replication rung, ~25 min)

**VERDICT: CLAIMED — REPLICATED at n=10 cumulative fresh.** Every bar reproduces on
fresh seeds 5-9 to 3-4 decimals: P1 recovery +0.8067 (vs +0.82), P2 frac 0.971 (==),
P3 A-retention -0.3182 (vs -0.32), G1 harm -0.0083 (== PR-14's EXACTLY), G2 -0.1057.
The cross-arm A/B canary passed per cell; the fresh G1 compared fresh-PLAIN vs fresh-EX
(BOTH arms re-run per the prereg). **INCIDENT (cosmetic): one projection-pacing crash
(stale claim.PLAIN.s0 key at offset seeds) fixed BEFORE any cell ran; the PLAIN cells
from the crashed attempt were already ledgered and resumed.** Runner:
manyblock_claim gained the guarded --seed-offset/--ledger-dir flags. **The CPU-side
ladder now has BOTH crown jewels replicated: PR-18 (the 3-block flagship, n=10 fresh)
and PR-19 (the 5-block accumulation, n=10 cumulative fresh). NEXT = S3 (owner Colab).**
Artifacts: results/manyblock_PR-2026-09-03-19/ + official console logs.

## PR-20 EXECUTED + EXTENDED (2026-09-10 23:1x — the damage-gap replication, final)

**VERDICT: NOT-ESTABLISHED (FINAL at pooled fresh n=10).** The n=5 leg: delta +0.2746,
CI [0.21, 0.34] — point gap replicates but the 0.25 bar unestablished. The §4
pre-authorized seeds 10-14 extension ran (both arms, separate ledger per doc addendum
#1) and the POOLED fresh n=10 (seeds 5-14): delta +0.2453, CI [0.1969, 0.2936] —
directionally positive in EVERY fresh seed pair, but **the registered halving-margin
(0.25) is NOT CI-established: the fresh-sample gap (~0.25) is about half the
original-sample gap (~0.55) — the protection's SIZE is seed-dependent.** No further
extension pre-authorized. **THE COMPARISON LEDGER'S HONEST FINAL STATE: (1) accumulation
parity — both the column and the scheduled control hold A and recover B on revisit;
(2) boundary damage protection — directional, schedule-consistent, halving-withdrawn;
(3) routing ledger + specialization — tissue-only (structural). The CPU-side registered
program is COMPLETE. S3 (owner Colab) remains the ladder's next rung.** Artifacts:
results/damage_gap_repl_PR-2026-09-03-20/ + results/manyblock_PR-2026-09-03-20_ext/ +
pooled verdict in the PR-20 ledger.

## PR-21 EXECUTED (2026-09-11 00:2x — the attribution matrix CLOSES, ~seconds)

**VERDICT: CLAIMED — TISSUE-COSTS-RECOVERY. The attribution matrix is CLOSED.** C1
PARITY: SCHED-TF recovery == COLUMN (-0.04). C2 ESTABLISHED: PLAIN-TF recovery +1.147
vs COLUMN +0.820 (delta -0.328, CI [-0.422, -0.233]) — the PLAIN control recovers MORE.
**The honest mechanism story, fully attributed at claim grade: recovery magnitude is a
PLASTICITY property — the plain fast-learner swings hardest both ways (most boundary
damage +1.15, most revisit recovery +1.15); the schedule trades recovery for damage
protection; the tissue trades a little more recovery for the routing ledger. The column
is the STABLE learner; the plain TF is the VOLATILE learner.** The pre-committed branch
fired: the ledger-vs-recovery trade-off is a recorded design lead. **The CPU-side
registered program is now COMPLETE with the attribution matrix closed; S3 (owner Colab)
remains the ladder's only next rung.** Artifacts:
results/recovery_attr_PR-2026-09-03-21/ + official console log.

## CAMPAIGN 2026-09-11 (parallel-lane wave — owner: "çok çok çok daha iyi", stay inside the brain metaphors)

A three-lane wave (surveys → frozen contract → disjoint implementer lanes → coordinator review +
gates). No new claim was staked; one new mechanism was implemented and honestly probed.

1. **Lane 1 — tissue multi-timescale (fast/slow) cascade (Benna-Fusi lineage), DEFAULT-OFF.**
   `seq/fusion_probe.py` PCExpertHead gains zero-init `W_fast`; forward = W + W_fast; the expert
   optimizer receives the fast weights + the un-split biases (coordinator clarification: biases train
   in both modes, no confound); after each step `W += kappa*W_fast`, `W_fast *= (1-delta)`; flags
   `--cascade-target {off,tissue} --cascade-kappa --cascade-delta` threaded through both claim
   runners + the probe; fingerprints + per-block fast-norm diagnostics. `off` is byte-identical
   (pinned). 14 new tests. **EXPLORATORY PROBE (paired, seeds 0-1 first then FRESH 2-4, pooled n=5;
   `--out`/`--seeds` kept the historical artifact untouched): the frozen dose shifts BOTH metrics in
   the protective direction — pooled Δrecovery_B −0.019 [95% CI −0.071, +0.033], Δdamage_C −0.032
   [−0.083, +0.019]; damage lower in 4/5 pairs, recovery lower in 4/5 — a small, direction-consistent
   effect (~1/10 of the registered schedule's PR-16 damage gap) that SHRANK on the fresh seeds and
   whose CIs straddle zero. NOT claim-grade; the lever stays default-OFF and a dose/target design
   pass + a powered n is the prerequisite for any registration
   (`results/exploratory/manyblock_probe_2026-09-11/ANALYSIS.md`).** Historical canary PASS (the OFF
   runs reproduce the committed 2026-09-09 probe.json exactly).
2. **Lane 2 — Prizma-Seq CPU kernel (honest efficiency, default-off).** `chunked_delta` gains
   `fast_reads` (opt-in; read-ratio via `ratio / alpha_i`, removing a second [B,H,C,C] sub+exp;
   ~1e-5 drift, default path byte-identical); the unit-lower solve passes the strictly-lower matrix
   directly with `unitriangular=True` (maxdiff exactly 0.0). Closed the long-standing coverage gap:
   NEW `tests/test_delta.py` (chunked vs reference parity) + `tests/test_fast_reads.py`.
   Measured interleaved A/B medians: 1.033x / 0.995x / 1.056x (pooled 1.029x) — reported honestly,
   lever stays opt-in.
3. **Records truth + registry repair.** INDEX.md rows PR-08/PR-10/PR-11/PR-12 restored from
   `05ad731^` (they were accidentally deleted by that commit; registry now 12 CLAIMED / 6 NEGATIVE /
   1 NOT-ESTABLISHED + the exploratory footnote; duplicate PR-03 row collapsed; NOT-ESTABLISHED
   added to the vocabulary); README Prizma-Seq crosstalk corrected to the measured 0.117 + counts
   refreshed; notebook stage-6 command/archive/provenance aligned to the PR-13 confirmatory run;
   locale-robust test reads fixed (cp1254).
4. **Lane 4 — open-world abstain / NOVEL gate (opt-in, `src/prizma.py`).** The documented-but-
   unimplemented rule (docs/EXPERT_ECONOMY.md §3.4) now exists: `abstain_z=None` knob + new
   `route_or_novel(X)` (batch mean of per-sample min z over trained experts; novel above the
   calibrated z=4); default paths untouched (bit-identity test); calibration reproduced at reduced
   scale (0/36 false-NOVEL on trained domains, 12/12 detection on a never-trained domain, margins
   ~60–120σ). 7 new tests; calibration caveat recorded (the separation needs recruited,
   domain-pure floors — the committed test uses K=3/d=24/h=48 geometry, not the usual unit-test
   scale). Prune/merge stay specified-not-implemented (spec §3.1/§3.2).

**Suite at close: 563 collected -> 553 passed, 10 skipped (CPU, 3:45).** No prereg row changed; the
next registered rung remains S3 (owner Colab, PRIZMA_GPU_CAMPAIGN.ipynb). New leads: (i) the
cascade needs a dose/target design pass before any registration; (ii) L2 `fast_reads` could extend
to `_chunked_delta_eta` (survey-2 C2 extension, not done); (iii) prune/merge open-world lifecycle
(spec exists, not implemented).

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
