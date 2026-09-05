# PR-2026-09-03-06 — BAR-6: interleaved-router fix levers at toy scale

| field | value |
|---|---|
| Registry id | **PR-2026-09-03-06** |
| Date written | 2026-09-05 (motivation written after the exploratory runs it cites — that is what exploration is for; the run below uses seeds the exploration never touched) |
| Lane | **CLAIM** |
| Status | **IN-WRITE** (maintainer: freeze this text + add the INDEX row; only then may the run start. This file deliberately does NOT edit `INDEX.md`.) |
| Motivating explorations | `results/expert_economy_2026-09-03/growth.json` (collapse ledger), `results/exploratory/interleave_fix_probe_2026-09-05/probe.json` (lever selection study, n=2, seeds 0–1) |
| Claim vehicle | BAR-6 toy-scale interleaved-router check (`docs/superpowers/specs/2026-09-03-prizma-lm-fusion-design.md` §3.3 + GPU-return order item 4; commission synthesis §C4) |
| Primary artifact (to be written by the run) | `results/interleave_fix_PR-2026-09-03-06/raw.json` (per-seed records, crash-safe per `docs/RETENTION.md`; never deleted) |

## 1. Motivation (measured, not assumed)

Interleaved presentation collapses Prizma's vigilance routing by two documented mechanisms
(`docs/EXPERT_ECONOMY.md` §2.2, from live ledger records):

- **Mixed batches** (samples of all K domains shuffled into one stream): batch-mean
  reconstruction surprise is stationary, so batch-level vigilance is structurally blind.
  Measured `route_log = [359744, 256, 0, …]` — one expert trains exactly its warmup (256
  samples), freezes, then catch-alls the remaining ~360k samples **without ever training
  again**; recruited experts = 2/2/1 over seeds 0–2.
- **Round-robin batches** (domain-pure batches, shuffled order): the first expert freezes
  with an *immature* precision floor (n_seen = 256), and the loose floor then recognizes
  every later domain. Measured `route_log = [359744, 256, 0, …]`, recruited = 2/2/2.

Consequences, measured: routed ACC on these streams is **0.569 (mixed) / 0.586
(round-robin)** at n=2 (`probe.json`, shipped arm; consistent with the ~0.58 collapse
quoted in commission synthesis §C4) vs 0.834 on the shipped block stream. Related: the
single-pass citation battery (`docs/CONTINUAL_CITATION_BAR.md` §3.1) locates Prizma's
deficit in **acquisition** (LA 0.627 vs replay 0.708) — these levers attack acquisition on
the interleaved regime where it is worst. Per the fusion design spec §3.3 and the owner's
GPU-return order, BAR-6 (this check) is a prerequisite of PR-LM-1 (continual char-LM).

## 2. The levers (implemented, default-OFF, bit-identical)

Four additive `Prizma`-config knobs in `src/prizma.py` (plus the shipped
`n_settle_steps`, swept unmodified). At their default values every run is **bit-identical**
to the shipped model — verified by `tests/test_interleave_levers.py` (12 tests: identical
ACC/FGT AND identical parameter trajectories on block AND round-robin streams; unit tests
pin each lever's formula). Internal constants (τ=200, EMA rates 0.2/0.02, clamp [0.25, 4],
novel fraction 0.5) were frozen a priori, before any lever run.

| lever | off value | on semantics (exact) |
|---|---|---|
| `freeze_min_seen` | 0 | consolidation may not freeze an expert with `n_seen < floor` (commit/advance still happen) |
| `dynamic_vigilance` (dv) | 0.0 | causal novelty-rate EMAs (fast α=0.2, slow α=0.02, max = hysteresis); θ_eff = clamp(1 + dv·(rate − 0.5), 0.25, 4)·(μ + z·σ) |
| `hot_young` | 0.0 | every local delta step of an expert scales by 1 + hot·exp(−n_seen/200) (n_seen read pre-update) |
| `route_stat` | "batch_mean" | "sample_top": per-sample test against the same μ + z·σ threshold; batch recognized iff novel fraction < 0.5 (decision stays batch-level; no new calibration) |

## 3. Exploratory selection study (LANE-EXPLORATORY — motivation, never a claim)

`experiments/interleave_fix_probe.py`, full factorial {veto 0/300} × {dv 0/0.5} ×
{hot 0/1.0} × {route_stat batch_mean/sample_top} × {settle 0/2} = 32 configs × 2
interleaved streams × 2 seeds (0, 1) = 128 runs (+ E1 guards), 196 s CPU. Selection rule
was frozen in the script header **before running**; it fired its honest empty-PASS branch:

- **No config reached the 0.70 bar on both streams.** Best-weakest:
  **hot_young=1.0 + route_stat=sample_top → mixed 0.603, round-robin 0.602** (fewest
  levers among tied; E1 guard at that config: ACC 0.813, FGT 0.000, n=2 — below the bar-(i)
  window at n=2, recorded as a risk flag, not hidden).
- Mechanistic ledger reading (same JSON): under shipped routing the first expert stops
  training after its 256-sample warmup and the stream is never learned (0.569/0.586);
  under `sample_top` a single expert trains on ~100% of the stream (route
  [360000, 0, …]) and ACC rises to ~0.60 — i.e. **to the monolithic-learner ceiling**.
  Batch-level training granularity cannot give different experts different samples of a
  mixed batch, so per-domain specialization (block-stream 0.834-level) is not recovered
  by any batch-level decision statistic. `hot_young` alone ≈ shipped (0.570/0.586);
  `dv=0.5` alone **hurts** (0.515/0.540); the freeze veto is ACC-neutral in isolation and
  in combination (0.603/0.602 with and without) — as expected analytically, because a
  committed-but-unfrozen expert still protects via the committed scan.

## 4. The frozen claim configuration

```
Prizma(..., consolidate=True, feedback="random", z_novel=5.0,
       freeze_min_seen=0, dynamic_vigilance=0.0,
       hot_young=1.0, route_stat="sample_top", n_settle_steps=0)
```

All other settings identical to shipped E1 (`structured_permuted_tasks`, K=5, d=24,
8 classes, h=48, M=K+3=8 experts, epochs=15, batch=128, per-seed generator as in
`run_continual.run_prizma`).

## 5. Arms, streams, seeds

**Arms (attribution):**

| arm | config |
|---|---|
| `shipped` | all levers off (must reproduce the shipped E1 row within seed noise) |
| `+fixes` | the frozen claim configuration (§4) |
| `+fixes-minus-route_stat` | `+fixes` with `route_stat="batch_mean"` (isolates hot_young) |
| `+fixes-minus-hot_young` | `+fixes` with `hot_young=0.0` (isolates route_stat) |

*Drafting note (pre-registered here, not retro-fitted):* the originally sketched fourth arm
was `+fixes-minus-veto`. The exploratory study showed the veto ACC-neutral in isolation AND
in the chosen combination, so the frozen config **simplifies it away** (bar (iii) below),
and the freed attribution slot tests `hot_young` instead — every ACTIVE lever in the
frozen config gets a causal necessity test. `freeze_min_seen` and `dynamic_vigilance`
remain available default-off knobs in `src/` but are not part of the claimed fix.

**Streams:** `mixed` and `roundrobin` interleaved (constructed exactly as in
`experiments/expert_economy.py` — same generator, permutations, exposure; presentation
order only) + the shipped E1 block stream as the no-regression guard.

**Seeds:** **10–14 (n=5)**, contiguous, fixed now, no additions, no stopping rule other
than completion (the exploration used 0–1 precisely so these are uncontaminated).
Statistics: per-arm mean ACC with Welch 95% CI over seeds reported for every arm and
stream; the bars below are fixed absolute thresholds, not difference tests, so no
correction multiplicity applies to them; arm contrasts are attribution (reported, with CI).

## 6. The bars (falsifiable, exact)

1. **No-regression (gate):** `+fixes` on the E1 block stream at n=5: mean ACC within
   **±0.02 of 0.834** (the shipped 10-seed E1 headline) AND mean FGT **≤ 0.02**.
   If the gate fails, everything below is moot: the fix is rejected as shipped-regressing.
2. **Interleaved recovery (primary):** `+fixes` mean ACC **≥ 0.70 on BOTH interleaved
   streams** at n=5 (Welch CI reported; the bar applies to the point estimate, per BAR-6's
   toy-scale wording).
3. **Attribution:** each active lever (`route_stat`, `hot_young`) is **causally necessary**
   — removing it (its minus-arm) fails bar 2 on some stream, or costs ≥ 0.02 mean
   interleaved ACC on some stream — **or the config is simplified** to the levers that
   pass; a simplified config is the only one eligible for the claim wording, and if the
   simplification was not itself run at n=5 in this campaign, a follow-up n=5 run of the
   simplified config is required before any claim.

## 7. Decision rule (including the honest failure branch)

- **PASS** = bars 1 + 2 (+ 3 after simplification logic): the interleaved-router fix is
  declared sufficient **at toy scale**, BAR-6 is recorded ANALYZED→CLAIMED, and PR-LM-1's
  continual char-LM registration may proceed to its own pre-registration.
- **FAIL of bar 2** (the exploratory n=2 outcome predicts this): **the fix is declared
  INSUFFICIENT**. BAR-6 is recorded ANALYZED→NEGATIVE, **PR-LM-1 is BLOCKED** on the
  router, and the n=5 failure numbers (per-arm, per-stream, with CIs and route logs)
  become the **documented blocker** — appended to the fusion design spec §3.3 discussion
  and to the C4 follow-up list. The already-measured mechanistic diagnosis travels with
  it: batch-level routing cannot specialize experts on mixed batches (exploratory ceiling
  ~0.60 = monolithic learner); candidate next mechanisms (per-sample training
  granularity, session/recruitment windows, replay-before-freeze variants) each require
  their OWN pre-registration — never an amendment of this one.
- **FAIL of bar 1:** additionally revert the default recommendation: the levers stay
  default-off (they already are) and the README/design docs must not present them as
  fixes.

## 8. Honest limits (what will NOT be claimed even on a PASS)

- Toy-scale synthetic structured-permuted streams only; no natural-corpus, no
  class-incremental, no pretrained-backbone claim; no "task-free" wording (boundary-free,
  label-free only; K is known a priori via M=K+3).
- A PASS at 0.70 does NOT claim block-stream-level specialization (0.834) is recovered on
  interleaved streams, and does not claim any LM capability — only that the router
  recruits/trains enough under interleaving to clear the BAR-6 toy bar.
- Exploratory numbers in §3 (n=2, seeds 0–1) motivate; they are never evidence
  (POLICY.md). All claim evidence comes from the n=5 run above.
- The mixed-stream and round-robin constructions are new exploratory streams (not shipped
  benchmarks); the claim is scoped to exactly these constructions.

## 9. Provenance

- Levers + tests: `src/prizma.py` (4 knobs, default-off, bit-identical — suite 254P+10S
  before, 266P+10S after this task; `tests/test_interleave_levers.py` adds 12).
- Exploration: `experiments/interleave_fix_probe.py` →
  `results/exploratory/interleave_fix_probe_2026-09-05/probe.json` (raw, retained).
- Mechanism sources: `docs/EXPERT_ECONOMY.md` §2.2 (failure modes),
  `00_COMMISSION_SYNTHESIS.md` §C4, fusion design spec §3.3-A/B (settling, metaplasticity,
  dynamic vigilance 04-P1, replay-before-freeze 02-P4 lineage).
