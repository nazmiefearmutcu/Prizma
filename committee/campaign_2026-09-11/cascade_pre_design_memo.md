# Cascade pre-design memo (campaign 2026-09-11) — NOT a pre-registration

Status: internal design note. No run is authorized by this document; a registration (own doc +
INDEX row) is required before any claim. Written at 03:1x, 2026-09-11, after the paired
exploratory probe.

## 1. What was measured (paired, seeds 0-4, `kappa=0.05` `delta=0.10`)

| metric | OFF mean | CASCADE mean | paired Δ (n=5) | 95% CI | direction |
|---|---|---|---|---|---|
| recovery_B (postC−postD) | +1.0823 | +1.0634 | **−0.0189** | [−0.0707, +0.0329] | lower in 4/5 seeds |
| damage_C (postC−postB) | +0.9144 | +0.8829 | **−0.0318** | [−0.0825, +0.0189] | lower in 4/5 seeds |

Artifacts: `results/exploratory/manyblock_probe_2026-09-11/` (PROBE_off/cascade, fresh-seed
extension, ANALYSIS.md). Historical canary PASS (OFF reproduces the committed 2026-09-09 run).
Interpretation: a small, direction-consistent PROTECTIVE shift (~1/10 of the registered
schedule's PR-16 damage gap), CIs straddling zero. NOT claim-grade.

## 2. Why the effect is small (mechanism reasoning, to be tested — not asserted)

- The lever acts on the TISSUE expert heads (Wenc/Wdec) only.
- PR-17 established that the *damage* protection is carried by the L1 LR schedule (any model can
  adopt it), and PR-21 established that the *recovery* deficit vs the plain TF is schedule-carried
  too (the column ≈ the scheduled control; the tissue is neutral on recovery).
- Therefore a tissue-side mechanism has little leverage on the two registered metrics; the
  observable was expected to be small. The trunk side is where the schedule acts — and a trunk
  cascade would interact/confound with L1 (PR-17 turf), which is why it was deliberately out of
  scope for the first probe.
- Secondary: with `kappa <= delta` the fold mostly transfers a decaying fast component
  (transient filter). With kappa == delta the fold is a pure reparameterization except for
  AdamW's decoupled weight decay (0.01 default, verified) — a dose-design subtlety that must be
  documented in any future registration.

## 3. Power reality (for any future registration)

Paired sd of the observed deltas: recovery ~0.043, damage ~0.043 bpc. At the observed ~0.03 mean
effect, ~15 paired seeds are needed for 80% power (two-sided 0.05) — not practical. **A design
pass must first find a target/dose with a materially larger effect** (≥0.10 bpc) or the mechanism
stays descriptive-only.

## 4. Design axes (open questions for the maintainer/owner)

1. **Target**: tissue (done) | trunk (the schedule's turf — needs an explicit
   replace-vs-compose decision vs L1: e.g., cascade REPLACES the blanket 0.25 dose, or composes
   with it) | both.
2. **Dose semantics**: `kappa` (consolidation), `delta` (fast decay); the ratio regime
   (`delta > kappa` transient filter vs `kappa ≈ delta` reparameterization+decay). Note the
   existing dose was frozen a priori, never tuned; a design pass may NOT tune on the registered
   manyblock seeds without pre-registering the exploration.
3. **Coupling with the schedule**: does a trunk cascade recover the PR-21 deficit AND keep the
   PR-16 damage protection? If it only shifts both metrics like the schedule does, it adds no
   capability over L1 and should be retired.
4. **Metrics/arms**: manyblock P1–P3 + G1/G2 (registered bars), plus B4 trunk-drift accounting;
   OFF canary must bit-match the registered PR-14 cells; a fresh-seed sample for anything
   confirmatory.
5. **Cheapest informative next probe** (if pursued): trunk-target cascade at ONE pre-declared
   dose, paired n=2–3, same probe harness — measuring whether the trunk version has a larger
   lever arm than the tissue version. Only if that shows ≥0.10-ish directional movement does a
   registration design pass become worthwhile.

## 5. Recommendation

Park the tissue cascade as an available, tested, default-OFF mechanism (documented). Do not
register a claim at this effect size. If the mechanism is pursued, the trunk-target probe in §4.5
is the next concrete step; otherwise the CPU-side attention belongs on the remaining leads
(fast_reads extension, merge lifecycle) and the GPU ladder (S3, owner Colab).
