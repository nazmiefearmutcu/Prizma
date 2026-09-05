# Replay-before-freeze probe (G3a) — LANE-EXPLORATORY — honest NEGATIVE: generative replay at freeze does not clear BAR-6

| field | value |
|---|---|
| Date / lane | 2026-09-05, **LANE-EXPLORATORY** (n=2 seeds 0–1; never a claim — `docs/preregistry/POLICY.md`) |
| Script | `experiments/replay_probe.py` (stream constructors reused verbatim from `experiments/interleave_fix_probe.py`; G3a lever in `src/prizma.py`, default-off, bit-identity-tested by `tests/test_replay_levers.py`) |
| Raw artifacts | `probe.json` (raw per-run records + route_log/n_seen ledgers + G3 replay diagnostics + frozen selection trace), `raw/<config>__<stream>__seed<k>.json` (12 crash-safe cells) |
| Outcome | **NO PR DRAFT.** The pre-committed failure branch of the frozen selection rule fired: the reported best (`best_prior`) has min interleaved ACC 0.557 < 0.68 on both streams. PR-2026-09-03-07 is NOT drafted; the acquisition-volume hypothesis survives as a diagnosis, but replay-before-freeze in the tested form does not act on it. |

## Configs + frozen selection rule (written into the script header BEFORE running)

Configs (precedence: shipped < g3a < best_prior) × streams {mixed, roundrobin} × seeds
{0, 1} = 12 runs (17.3 s CPU; budget 45 min, first-cell projection gate passed).
G3a strength frozen a priori: `replay_passes=15` (the block stream's epoch count — the
acquisition-volume hypothesis gives the recruit the same NUMBER of passes),
`replay_items=256` (one batch-equivalent); hidden-statistic EMA rate 0.05, variance
floor 1e-4 frozen in `src/prizma.py`. Rule: hard specialization filter (max per-expert
train-fraction on mixed ≤ 0.85), PASS-set = filter AND ACC_mean ≥ 0.70 both streams AND
E1 guard (\|ACC−0.834\| ≤ 0.02, FGT ≤ 0.02); if empty → best-weakest among eligible;
draft PR-2026-09-03-07 only if the reported best ≥ 0.68 on BOTH streams AND passes the
ledger filter.

## Results (mean ACC over seeds 0–1; max per-expert train-fraction on mixed)

| config | mixed ACC | roundrobin ACC | max train-frac (mixed) | specialization | verdict |
|---|---|---|---|---|---|
| shipped | 0.570 | 0.586 | 1.000 | FAIL | monolithic baseline (reproduces granularity probe exactly) |
| g3a | 0.571 | 0.587 | 0.998 | FAIL | **+0.001 ACC** — replay bound to freeze events fires ~once per stream on the monolith; ledger unchanged |
| best_prior (G1+sample_top+G3a) | 0.557 | 0.564 | **0.727** | **PASS** | ledger fixed, ACC not — and slightly below G1 alone (0.560/0.566 in the granularity probe) |

E1 block-stream guard for the reported best (n=2): **ACC 0.666, FGT 0.003 — RISK FLAG**,
far below the ±0.02 band around 0.834 (shipped E1 at this n measured 0.813 in the
granularity probe). FGT ≈ 0 means nothing is forgotten; the damage is ACQUISITION on the
block stream itself. Attribution between G1 and G3a for this drop is unmeasured here
(the frozen rule guards only the chosen config) — recorded, not excused.

PASS-set was **empty**: no config reached 0.70 on either interleaved stream, let alone
both. Eligibility: only `best_prior` passed the ledger filter (0.727 ≤ 0.85); `g3a` at
batch granularity stays monolithic (0.998) because replay is bound to freeze events and
the monolith freezes ~once.

## Replay diagnostics (the G3-specific read-out; per run, means over freeze events)

| arm / stream | replay events | items generated | pseudo-set recon Δ (pre−post) | pseudo-set head-conf Δ |
|---|---|---|---|---|
| g3a, mixed | 1 per seed | 256 | +0.006 / +0.007 | −0.001 / −0.003 |
| g3a, roundrobin | 1 per seed | 256 | +0.005 / +0.004 | −0.002 / −0.003 |
| best_prior, mixed | 2–3 | 512–768 | +0.009 / +0.009 | −0.000 / −0.002 |
| best_prior, roundrobin | 1–2 | 256–512 | +0.010 / +0.009 | −0.001 / −0.000 |

The mechanism does exactly what it says and nothing more: the expert consolidates its
OWN manifold (pseudo-set reconstruction error drops consistently) while its self-labeled
head confidence does NOT rise (Δ ≤ 0). Total generated volume per stream (256–1280
items) is ~3 orders of magnitude below the stream's real sample-presentations (~450k),
because replay is bounded by the freeze-event count the router produces.

## Mechanistic read: why replay-before-freeze (this form) is insufficient

1. **Replay inherits the router's sparsity.** Bound to freeze events, G3a fires exactly
   as often as the (broken) router freezes: once on the monolithic ledger, 2–3 times per
   specialist ledger. A mechanism that grants "more passes" only at the router's
   commit points cannot out-volume the router's own failure to specialize. The 15×256
   generated updates per expert (~3.8k sample-steps) are a rounding error against the
   block stream's ~90k real sample-updates per domain.
2. **The information cap is measurable, not hypothetical.** On its own pseudo-set every
   replayed expert consolidates (recon ↓) — and its classifier head, self-distilled from
   the one-pass head, converges to what that one pass already knew (confidence flat to
   slightly down). ACC moves ≤ +0.001 (g3a) or −0.003 (best_prior vs G1 alone). This is
   the docstring's honest caveat borne out in data: consolidation without new information.
3. **The binding pair is (domain-coherent data per expert, real-data revisits).** G1
   fragments by surprise, not domain (granularity probe cause 1); G3a can only replay
   the fragment each expert already owns, so it sharpens the wrong partition. What the
   acquisition-volume hypothesis still leaves untried is REAL-data revisit — e.g. a
   probationary expert that keeps training on every batch it partially recognizes
   (G3b, owner decision pending) or routed re-training of committed experts — mechanisms
   that add real passes instead of generated ones. Each requires its own exploratory
   probe and pre-registration; PR-LM-1 stays blocked on the router.

## Honest limits

Toy-scale synthetic structured-permuted streams only; n=2 exploratory seeds (0–1); these
interleaved constructions are exploratory streams, not shipped benchmarks; the E1
risk-flag (0.666) is an n=2 number for the combined G1+G3a config and attributes nothing
to either lever alone; no natural-corpus, class-incremental, or LM claim of any kind;
the shipped E1 result is untouched (all levers default-off, bit-identity-tested). One
G3a strength (15×256) was frozen and run; the negative is a negative at the frozen
strength, softened only by diagnostic 1 (volume is router-bound by construction, so a
strength sweep inside the same mechanism could not change the fire count). Per POLICY.md
no number in this file may be cited as evidence — it motivates at most a future
pre-registration, which the frozen rule says should NOT be a replay-strength retry.
