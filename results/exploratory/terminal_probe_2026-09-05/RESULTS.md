# Terminal probe (G3b probation + settle depth) — LANE-EXPLORATORY — honest TERMINAL NEGATIVE: the interleaved regime is information-bounded for this architecture at toy scale

| field | value |
|---|---|
| Date / lane | 2026-09-05, **LANE-EXPLORATORY** (n=2 seeds 0–1; never a claim — `docs/preregistry/POLICY.md`) |
| Script | `experiments/terminal_probe.py` (stream constructors reused verbatim from `experiments/interleave_fix_probe.py`; G3b `probation` lever in `src/prizma.py`, default-off, bit-identity-tested by `tests/test_probation_lever.py`) |
| Raw artifacts | `probe.json` (raw per-run records + route_log/n_seen ledgers + end-state expert flags + frozen selection trace), `raw/<config>__<stream>__seed<k>.json` (20 crash-safe cells + 2 E1-guard cells) |
| Runtime | 129.2 s CPU total (budget 60 min; first-cell projection gate passed at 0.5 min projected) |
| Outcome | **NO PR DRAFT — TERMINAL.** The pre-committed failure branch of the frozen selection rule fired: the reported best (`probation`) has min interleaved ACC 0.582 < 0.68 and fails the E1 guard (0.668 vs the 0.814 floor). This is the last probe of the zero-GPU BAR-6 program: neither live re-processing (G3b) nor deeper per-exposure compute (settle 4/8) closes the acquisition-volume gap. The BAR-6 mechanism ladder terminates here. |

## Configs + frozen selection rule (written into the script header BEFORE running)

Configs (precedence: best_prior < probation < settle4 < settle8 < probation_settle4) ×
streams {mixed, roundrobin} × seeds {0, 1} = 20 runs. `best_prior` =
train_granularity="sample" + route_stat="sample_top" (route_stat INERT under G1 —
documented + tested — kept as the frozen prior stack). `probation=True` (G3b, NEW
default-off lever): a recruited expert stays PROBATIONARY and keeps training on every
subsequent LIVE sample it recognizes until the stream's surprise floor says its domain
passed (PROBATION_PATIENCE = 8 consecutive miss-batches, frozen a priori); only then
commit/freeze — and the freeze needs BOTH n_seen ≥ freeze_min_seen AND the exit test
(the LATER gate binds). Buffer-free by construction: it stores nothing; it trains only
on live samples as they arrive (single-pass compatible). `n_settle_steps ∈ {4, 8}`
(existing shipped knob, never before swept past 2): K settles = K latent gradient steps
on the same per-sample free energy — deeper iterative processing per exposure, K×
acquisition steps without any revisit.

Rule: hard specialization filter (max per-expert train-fraction on mixed ≤ 0.85),
PASS-set = filter AND ACC_mean ≥ 0.70 both streams AND E1 guard (\|ACC−0.834\| ≤ 0.02,
FGT ≤ 0.02); if empty → best-weakest among eligible; draft PR-2026-09-03-07 only if the
reported best ≥ 0.68 on BOTH streams AND the ledger filter AND the E1 guard hold.

## Results (mean ACC over seeds 0–1; max per-expert train-fraction on mixed)

| config | mixed ACC | roundrobin ACC | max train-frac (mixed) | specialization | verdict |
|---|---|---|---|---|---|
| best_prior | 0.560 | 0.566 | 0.706 | PASS | G1 baseline (reproduces the granularity probe) |
| **probation** | **0.590** | **0.582** | **0.284** | **PASS** | **largest mechanism gain of the ladder (+0.030/+0.016); ledger balanced across 5–7 experts; still far below the bar** |
| settle4 | 0.557 | 0.532 | 0.775 | PASS | deeper settling does NOT help (−0.003/−0.034 vs best_prior) |
| settle8 | 0.550 | 0.540 | 0.875 | FAIL | more depth, worse: 8× compute per exposure, ACC down, ledger re-concentrating |
| probation_settle4 | 0.592 | 0.551 | 0.197 | PASS | no synergy: mixed ≈ probation alone, roundrobin worse (0.551 vs 0.582) |

PASS-set was **empty**: no config reached 0.70 on either interleaved stream. The frozen
rule's best-weakest choice is `probation` (min interleaved 0.582). Its E1 block-stream
guard (n=2): **ACC 0.668, FGT 0.000 — RISK FLAG**, far outside ±0.02 of 0.834. Cross-probe
attribution: the replay probe measured `best_prior` (G1+sample_top+G3a) E1 at 0.666 —
statistically identical to probation's 0.668 — so the block-stream regression is caused
by the SHARED G1 per-sample granularity (which removes the young-expert whole-batch
training on block streams), NOT by probation; on the block stream probation behaves as
designed (domains end → exit test freezes → FGT 0.000). Both facts are recorded, neither
excused: any future claim vehicle built on G1 must first fix the G1 block-stream
acquisition regression.

## The mechanism worked exactly as designed — and the ceiling moved only +0.03

The probation ledgers prove the mechanism is real and volume is no longer scarce:

| run | n_seen per expert | end state | purity per domain |
|---|---|---|---|
| probation, mixed, s0 | [72.9k, 86.5k, 86.6k, 51.8k, 30.2k, 0…] | 5/5 probationary, 0 freezes | [0.51, 0.46, 0.63, 0.31, 0.42] |
| probation, roundrobin, s0 | [61.4k, 65.7k, 74.9k, 62.2k, 32.9k, 23.1k, 2.8k, 0] | 7/7 probationary, 0 freezes | [0.34, 0.46, 0.44, 0.33, 0.34] |
| probation_settle4, rr, s1 | [59.2k, 53.9k, 46.6k, 47.6k, 41.9k, 35.1k, 26.0k, 9.8k] | 8/8 probationary | [0.28, 0.39, 0.25, 0.23, 0.32] |
| best_prior, rr, s0 (contrast) | [33.3k, 19.6k, 45.7k, 0…] | 2 frozen, then catch-all duty | [0.58, 0.57, 0.52, 0.52, 0.69] |

Under shipped/G1 machinery a recruited specialist gets ONE unrepeated pass at recruit
and then freezes into claim-only duty (max train-fraction 0.71–1.00 across the ladder's
history). Under probation every specialist trains CONTINUOUSLY on live traffic — up to
15 epochs' worth of passes over its own fragment (n_seen 60–87k samples), the ledger
balanced to max-fraction 0.18–0.28, exit tests firing correctly when domains do end
(mixed s1: 5 mid-stream freezes, FGT-neutral). Every live sample is processed by the
expert that recognizes it, with nothing stored. The acquisition-volume hypothesis got
its strongest possible test on this substrate — and bought +0.030/+0.016 ACC.

## Terminal mechanistic verdict

**Live re-processing and deeper per-exposure compute do NOT close the gap; the
interleaved regime is information-bounded for this architecture at toy scale — bounded
at the PARTITION level, not the volume level.** The complete measured ladder on these
streams: thresholds +0.06 (PR-06) → granularity fixes the ledger, ACC flat (G1) →
generative replay ±0.001, consolidation without new information (G3a) → REAL-data
revisit at maximum volume +0.03 (G3b, this probe) → deeper settling per exposure
−0.00…−0.03 (settle 4/8). Every arm lands in 0.51–0.62 while the block stream, same
local rule, sits at 0.834. After probation, volume is exhausted as an explanation
(specialists train continuously; the ledger is balanced by construction), and settle
depth is exhausted as an explanation (8× compute per exposure moved ACC DOWN). What
remains, measurably, is the routing partition: fragment purity on the interleaved
streams is 0.23–0.63 — the floors still calibrate on mixture data and tile input space
by surprise, not by domain (the granularity probe's cause 1, now measured under
unlimited live volume). An expert that trains to competence on a 33%-pure fragment —
however many passes, however deep the settling — is a mixture learner, and the mixture
ceiling of this substrate is ~0.60. Probation's 0.590/0.582 is exactly that ceiling.

**Precisely what data-volume assumption would change the answer:** not total exposures
per domain — probation grants ~15 live passes per specialist fragment and moves +0.03,
so "more epochs / more presentations of the same mixed stream" is measured insufficient.
The assumption that would change the answer is DOMAIN-COHERENT EXPOSURE STRUCTURE: the
per-domain contiguous run-length at the moment the first precision floors calibrate.
Sample-level shuffling of 5 domains in 128-sample batches gives an expected contiguous
domain run of ~1.6 samples — no window can calibrate domain-pure floors, and fragment
purity is pinned near the mixing ratio regardless of stream length or epoch count. The
block stream is the existence proof: whole-domain contiguous blocks (run-length ≈ 960)
let the same rule reach 0.834. Concretely, the terminal verdict no longer applies when
the stream delivers per-domain contiguous blocks of ≳ 10² samples (domain-block-level
interleaving rather than sample-level) — under that presentation the existing
default-off ladder (G1 + probation + the exit test) is the natural, already-implemented
candidate, and it would need its own pre-registration on fresh seeds. Scaling unique
samples per domain at fixed sample-level mixing is NOT expected to help: purity stays
at the mixing ratio, so the ceiling stands.

## Honest limits

Toy-scale synthetic structured-permuted streams only (K=5, d=24, 4800 unique samples,
15 epochs); n=2 exploratory seeds (0–1); mixed/roundrobin are exploratory constructions,
not shipped benchmarks; one probation strength (PATIENCE=8) and two settle depths
{4, 8} frozen a priori and run — the negative is a negative at the frozen strengths,
with the structural argument (volume is no longer scarce under probation; purity is
pinned by the mixing ratio) softening strength-sweep concerns. The E1 risk flag (0.668)
is an n=2 number for G1+probation and attributes the block-stream drop to G1 only by
cross-probe comparison with the replay probe's 0.666 (both exploratory). The shipped E1
headline (0.834/FGT 0.000) is untouched: every lever here is default-off and
bit-identity-tested. Per POLICY.md no number in this file may be cited as evidence —
and per the pre-committed terminal branch, none will be: the BAR-6 mechanism ladder for
the interleaved-router problem is CLOSED at toy scale on this architecture. PR-LM-1
remains blocked on the router; the router's blocker is now precisely located in the
routing signal's domain information, not in training volume, granularity, replay, or
per-exposure compute.
