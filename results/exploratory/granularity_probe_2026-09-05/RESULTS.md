# Granularity probe (G1/G2) — LANE-EXPLORATORY — honest NEGATIVE: granularity alone does not clear BAR-6

| field | value |
|---|---|
| Date / lane | 2026-09-05, **LANE-EXPLORATORY** (n=2 seeds 0–1; never a claim — `docs/preregistry/POLICY.md`) |
| Script | `experiments/granularity_probe.py` (stream constructors reused verbatim from `experiments/interleave_fix_probe.py`; levers in `src/prizma.py`, default-off, bit-identity-tested) |
| Raw artifacts | `probe.json` (raw per-run records + frozen selection trace), `raw/<config>__<stream>__seed<k>.json` (16 crash-safe cells), `n_seen_ledger.json` (supplementary per-expert TRAINED-volume diagnostic) |
| Outcome | **NO PR DRAFT.** Best min interleaved ACC = 0.570 < 0.68 floor on both streams → the pre-committed no-draft branch of the frozen selection rule fired. PR-2026-09-03-07 is NOT drafted; PR-LM-1 stays blocked on the router. |

## Grid + frozen selection rule (written into the script header BEFORE running)

Configs (precedence: shipped < g2_w32 < g2_w8 < g1_sample) × streams {mixed, roundrobin}
× seeds {0, 1} = 16 runs (19.5 s CPU; budget 45 min, first-cell projection gate passed).
Rule: hard specialization filter (max per-expert train-fraction over mixed runs ≤ 0.85,
per the PR-06 lesson "check the TRAINING LEDGER, not just the accuracy"), then PASS-set
= filter AND ACC_mean ≥ 0.70 both streams; if empty → best-weakest min interleaved ACC;
draft PR-07 only if the reported best ≥ 0.68 on BOTH streams AND passes the ledger filter.

## Results (mean ACC over seeds 0–1; max per-expert train-fraction on mixed)

| config | mixed ACC | roundrobin ACC | max train-frac (mixed) | specialization | verdict |
|---|---|---|---|---|---|
| shipped | 0.570 [0.559, 0.580] | 0.586 [0.581, 0.592] | 1.000 | FAIL | monolithic (PR-06 blocker reproduced) |
| g2_w32 | 0.554 [0.549, 0.559] | 0.547 [0.535, 0.559] | 0.997 | FAIL | no recovery |
| g2_w8 | 0.474 [0.464, 0.484] | 0.586 [0.582, 0.589] | 0.989 | FAIL | **hurts** mixed |
| g1_sample | 0.560 [0.528, 0.593] | 0.566 [0.548, 0.583] | **0.706** | **PASS** | ledger fixed, ACC not |

E1 block-stream guard (no-regression direction, n=2): shipped 0.813 / FGT 0.000 (the
granularity levers are additive default-off knobs; G1/G2 E1 guards not triggered by the
rule because no candidate passed — available on request).

PASS-set was **empty**: no config reached 0.70 on both streams; the best-weakest config
by the frozen rule is `shipped` itself (min interleaved 0.570); the best GRANULARITY
candidate is `g1_sample` (min interleaved 0.560). Both are far below the 0.68 draft
floor → **no pre-registration is drafted**.

## The training ledger (the PR-06 lesson, applied): specialization DID happen under G1 — and did not pay

Supplementary per-expert TRAINED-volume diagnostic (`n_seen_ledger.json`, trained
fraction = expert's n_seen / total trained samples; `route_log` alone conflates trained
and protected-claim samples):

| run | trained fraction per expert (e0…e7) | committed / frozen |
|---|---|---|
| shipped, mixed, s0 | [0.998, 0.002, 0, …] | 1 / 1 (the monolith) |
| g1_sample, mixed, s0 | [0.442, 0.391, 0.166, 0, …] | 3 / 2 |
| g1_sample, mixed, s1 | [0.342, 0.202, 0.456, 0, …] | 3 / 2 |
| g1_sample, roundrobin, s0 | [0.338, 0.198, 0.464, 0, …] | 3 / 2 |
| g1_sample, roundrobin, s1 | [0.361, 0.639, 0, …] | 2 / 1 |

G1 converts the single 0.998-monolith into 2–3 experts sharing the stream (max fraction
0.64–0.71, every seed ≤ 0.81): the WHO-trains-on-WHAT fix works as designed.

## Mechanistic read: why granularity alone is insufficient

1. **Specialists are surprise-coherent, not domain-coherent.** G1's novel pool is defined
   as "samples far from every calibrated floor". The first floor is calibrated on a
   MIXTURE batch, so the partition it induces splits input space by distance-from-
   mixture; each recruit then trains on a fragment that still mixes domains (only 3 of
   the 5 domains ever get a dedicated expert — the ≥ 0.5 novel-fraction bar stops firing
   once 2–3 floors tile the space). Distributing the monolithic learner's mixture over
   3 fragments keeps ACC at the same ~0.56 ceiling.
2. **One unrepeated pass per fragment.** Recruitment commits the previous expert
   immediately (the shipped recruit machinery, reused as specified), so each specialist
   sees its fragment exactly once with no revisit; the block stream's 0.834 comes from
   ~15 epochs of sustained per-domain training. Committed experts never re-train on
   their recognized samples, so acquisition volume per domain stays ~1/3 of ONE epoch.
3. **G2's window unit is still mixture-blind at realistic W.** A 8- or 32-sample window
   of a 5-domain shuffled stream has near-stationary mean surprise → the ledger stays
   monolithic (0.99) and the extra commits only fragment training (W=8 mixed ACC 0.474,
   the worst cell in the grid). The batch-mean blindness shrinks with W arithmetic but
   not enough at this stream's per-domain batch share.

**Indication (owner decision, per the PR-06 successor list):** the binding constraint has
moved one level deeper — from training granularity (G1/G2, now implemented and honestly
negative at toy scale) to **per-domain training volume / revisit**: replay-before-freeze
(G3: recruits train to competence before committing; recognized samples of committed
experts periodically re-train instead of only routing) is the remaining mechanism class
of the spec Addendum's three. PR-LM-1 stays blocked on the router.

## Honest limits

Toy-scale synthetic structured-permuted streams only; n=2 exploratory seeds (0–1); these
two interleaved constructions are exploratory streams, not shipped benchmarks; no
natural-corpus, class-incremental, or LM claim of any kind; the shipped E1 result
(0.813/FGT 0.000 at this n) is untouched. Per POLICY.md no number in this file may be
cited as evidence — it motivates at most a future pre-registration (which the frozen
rule says should be G3, not a granularity retry).
