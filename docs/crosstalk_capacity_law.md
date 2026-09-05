# The Crosstalk Capacity Law: mean/fluctuation split, candidate `N* = min(1 + ε²/σ₂², d_φ)`, and the pre-registered D-frontier prediction

**Status:** CANDIDATE theory + PRE-REGISTERED protocol. Nothing here is a proven result; the
derivation is a sketch with measured-instrument support only. The one deferred GPU run
(the D-frontier grid) is the adjudicator, and it must not be run before the frozen-prediction
commit described in §5 exists.
**Registry lane:** `PR-2026-09-03-02`, **status = IN-WRITE** (this document is the pre-registration
draft; the maintainer merges the `docs/preregistry/INDEX.md` entry — per lane discipline this agent
does NOT edit `INDEX.md`; id `PR-2026-09-03-01` is already CLAIM/IN-WRITE for the surprise-gating
powered-ablation lane).
**Origin:** committee report 11 (`committee/brainstorm_2026-09-03/11_theory_capacity_expressivity.md`),
proposals T1/T3; honesty corrections in the addendum of `docs/quad2_theoretical_convergence.md`
(Flags A–C). Instrument: `feat_map_probe.py` (+ `results/feat_map_probe.json`,
`tests/test_crosstalk_metrics.py`).

---

## 1. Why: the Gershgorin corollary is scope-broken (Flags A/B, one paragraph)

The repo's capacity corollary `N < 1 + 1/cross(φ)` comes from a Gershgorin row-sum bound that is
tight only if all off-diagonal crosstalk entries in a row align in sign. They are approximately
zero-mean random variables; their *fluctuation*, not their mean, is what limits recall. With the
committed artifact's measured `cross(quad2) = 0.11699` the bound caps N < 9.55, while the repo's own
MQAR D=128 PASS (`results/gpu_bench.json`; N = D = 128 stored pairs, `seq/tasks.py`) requires
N* ≥ 128. The bound explains the **linear baseline's failure only**; it explains nothing about the
quad2 success. Full argument: addendum (Flags A–C) in `docs/quad2_theoretical_convergence.md`.

## 2. The mean/fluctuation split and the candidate law (derivation sketch — NEW; labeled)

One pass, β = 1, α = 1 (Theorem 2's setting), keys unit-normed in φ-space. For stored pairs
(k_i, v_i), i = 1..N, the readback of item i is:

```
o_i = S φ(k_i) = v_i + Σ_{j≠i} c_ij v_j ,      c_ij = φ(k_i)ᵀ φ(k_j)
```

Split the crosstalk matrix E = G − I into a common mode and a fluctuation (**[N]** as applied here):

```
E = μ·11ᵀ + W ,     μ = E[c_ij] (common mode) ,     W = zero-mean fluctuation
```

The common mode adds the *same* vector (Σ_j v_j) to every read — key-independent, hence absorbed by
the learned output head (and by `state_norm`); only W's fluctuation is irreversible capacity noise.

**SNR argument** (classical associative-memory crosstalk analysis — **[B-method]**: Hopfield '82/'84
line; the σ₂-measurement instrument and its application to the rectangular delta state are **[N]**):

```
signal = 1 ;   noise² ≈ Σ_{j≠i} c_ij² ‖v_j‖² ≈ (N−1)·E[c²] = (N−1)·σ₂_law²
SNR(i) ≈ 1 / (σ₂_law √(N−1))         with  σ₂_law = sqrt(E[c²]) = sqrt(σ₂_W² + μ²)
```

Requiring SNR ≥ ε (a single O(1) task-level tolerance constant absorbing ‖v‖², query-estimation
error, gate losses) gives the **candidate capacity law**:

```
N*(ε; φ) ≈ 1 + ε²/σ₂² ,      N_max(ε; φ) = min( N*(ε; φ), d_φ )
```

The `1 + ε²/σ²` form is the fluctuation analogue of worst-case row sums; the same
variance-threshold shape appears as May's stability criterion n·σ² < 1 in random-ecosystem theory
(**[B-method]** — analogy for the form only, not a derivation), and the √N-vs-N scaling gap against
Gershgorin is the standard Wigner row-noise fact ‖W‖₂ ≈ 2σ₂√N (**[B-method]** random-matrix theory).

**Not claimed (report 11 §5 discipline):** the law is not proven; ε is not known a priori; the probe
measures *random* keys and trained models may whiten them (then measured σ₂ overestimates the
trained model's — the fitted ε absorbs this at the pinned protocol, and per-seed trained-σ₂
measurement remains future work, report 11 T3); dense-query MQAR shifts ε, which is why the
protocol below pins one protocol across all arms.

## 3. The instrument (CPU, done): what the extended probe measures

`feat_map_probe.py` now reports, per feature map (exact definitions in its module docstring, unit
tests in `tests/test_crosstalk_metrics.py`, artifact `results/feat_map_probe.json`, regenerated
2026-09-04; legacy fields reproduce the 2026-06-08 artifact to 1 ULP [one last-digit float,
`quad2_lowrank.std`, see the addendum header note]): `mu` (= legacy cross, pooled), `mu_signed` (common mode),
`sigma2` (pre-registered probe definition: **sample std, ddof=1, of |cos| over the pooled i<j upper
triangle**), `sigma2_signed_fluct` (std of signed cosines = fluctuation of W), `eta` and
`eta_signed_fluct` (= 1/(σ₂²·d_φ)), `n_star` and `n_star_signed_fluct` (= min(1+1/σ₂², d_φ) at ε=1),
`offdiag_count`. Runtime ≈ 2.5 s CPU, seeded (42 / buffers 1234), numpy only.

Measured (D=128, d_h=32, 256 key draws of 128 keys each; floors = pure random-key values):

| map | d_φ | μ (=\|cos\| mean) | μ_signed | σ₂ (\|cos\| def) [floor] | σ₂_W (signed) [floor] | σ₂_law=√(σ₂W²+μ²) | η_W [floor 1] | N*_\|cos\| (ε=1) |
|---|---|---|---|---|---|---|---|---|
| `none` | 32 | 0.14228 [0.14105] | +0.00003 | 0.1052 [0.1066] | **0.1769** [0.1768] | 0.1769 | **0.998** | 32.0 (rank-capped) |
| `quad2` | 256 | 0.11699 [0.04987] | **+0.00639** | 0.0872 [0.0377] | 0.1458 [0.0625] | **0.1459** | 0.184 | 132.5 |
| `quad2_lowrank` | 137 | 0.12648 [0.06817] | +0.01601 | 0.0952 [0.0515] | 0.1575 [0.0854] | 0.1583 | 0.294 | 111.4 |

**Instrument validation against report 11's arithmetic — honest discrepancies included:**

1. **Validated:** `none` sits exactly on the random-key floors (σ₂_W = 0.1769 vs 1/√32 = 0.1768;
   η_W = 0.998 ≈ 1) — the signed-fluctuation metric is calibrated. The common-mode prediction
   μ ≈ λ/d_h ≈ 5.6e-3 for quad2 is **confirmed** at +6.39e-3 (ratio 1.14) — the E = μ11ᵀ + W split
   is real and measurable.
2. **Corrected (report 11 arithmetic slip):** the mean→fluctuation conversion is
   σ₂ ≈ cross·**√(π/2)** (≈1.2533), not cross·π/2 (≈1.5708). Verified on `none`:
   0.14228·√(π/2) = 0.1784 vs measured 0.1769 (−0.9%); the π/2 factor overestimates by +26%.
3. **Stale inputs (see addendum):** report 11's η ≈ 0.27 (quad2) and η ≈ 0.63 (`none`) and its
   "σ₂ ≥ cross = 0.076" all inherit cross values (0.076 / 0.085) that **no committed artifact
   reproduces**; the committed probe always measured 0.11699 / 0.12648. With measured values:
   η_W(quad2) = 0.184, η_W(none) = 0.998.
4. **Unsettled (the actual open question):** at ε = 1 with random-key σ₂, the law does **not**
   cleanly explain the D=128 PASS: the |cos|-definition gives N* = 132.5 (borderline), but the
   SNR-consistent σ₂_law = 0.1459 gives N* = **48.06 ≪ 128**. Report 11's "consistent with
   ε ≈ 0.9–1.2" relied on the unreproducible 0.076; the measured numbers imply ε ≈ 1.64 would be
   needed. This is exactly why §4 pre-registers the ε fit — and why ε̂ landing far from 1
   **demotes the law** rather than being argued away.

## 4. Definitions frozen BEFORE any MQAR run (pre-registration part 1)

- **σ₂ for the law** (per arm): `sigma2_law = sqrt(sigma2_signed_fluct² + mu_signed²)` from
  `results/feat_map_probe.json` (the SNR quantity E[c²]^½). Frozen values: `none` 0.17692,
  `quad2` 0.14591, `quad2_lowrank` 0.15829. (The |cos|-std `sigma2` stays a reported instrument
  metric; it is NOT the law's σ₂ — the SNR derivation uses the signed second moment.)
- **Rank cap:** d_φ = 32 / 137 / 256 for `none` / `quad2_lowrank` / `quad2` respectively.
- **Arms:** primary = `quad2` @ d_φ=256; secondary (non-binding) = `quad2_lowrank` @ d_φ=137,
  `none` @ d_φ=32. Buffers seeded exactly as the probe (seed 1234) so run keys and probe keys match
  the same map.
- **Solve criterion (per arm-rung):** rung SOLVED iff ≥ 2/3 seeds reach eval accuracy ≥ 0.90
  (repo precedent: B1 "3/3, 0.997" solves; 0.90 leaves margin above the chance/failed plateau
  ≈ 0.02–0.59 observed in gpu_bench failures). Seeds {0,1,2} minimum, identical across arms and
  rungs; harness = the existing MQAR A100 harness (B1/P3 machinery in `gpu_bench.py`), protocol
  pinned to the B1 configuration family (dense queries, matched-params config, shared-lr policy),
  identical budgets across rungs. **No protocol change after the first adjudication rung is run.**
- **Fit rungs:** D ∈ {16, 32, 64}. **Adjudication rungs:** D ∈ {96, 128, 192, 256}
  (union grid G = [16, 32, 64, 96, 128, 192, 256]; one grid step = adjacent entries).

## 5. Pre-registered prediction protocol (part 2: fit → freeze → adjudicate)

**Step 1 — fit ε once, at the D=64 grid (when GPU returns; ~hours: 3 arms × 3 fit rungs × 3 seeds
= 27 short runs).** Per arm: D*_fit = largest fit rung SOLVED. Then

```
ε̂(arm) = sqrt( (D*_fit − 1) · σ₂_law(arm)² )
```

(The law must hold *per arm with one shared constant* for the "single task-level tolerance"
reading; pre-registered reading is per-arm ε̂ with the consistency check |ε̂_quad2/ε̂_lowrank − 1| —
if per-arm ε̂ differ by > 2×, the "single constant" claim is demoted to per-map, reported honestly.)

**Step 2 — FREEZE predictions before any adjudication rung runs.** A follow-up commit to this file
(a) records ε̂ per arm, (b) fills the frozen table N*(arm) = min(1 + ε̂²/σ₂_law², d_φ) and the
per-rung predicted solve/not-solve for D ∈ {96, 128, 192, 256}, (c) records the commit hash. Any
adjudication rung started before that commit exists = protocol violation; results are quarantined.

**Step 3 — adjudicate** on the existing MQAR harness: 3 arms × 4 rungs × 3 seeds = 36 runs
(~12–18 A100-h with the B1 budget; the fit stage alone is the "~hours" tier).

**PASS bar (exact, primary arm `quad2` @ d_φ=256):** the law's per-rung solve prediction agrees
with observation on **≥ 3 of 4 adjudication rungs**. (Equivalent formulation: the observed
solve-transition among the adjudication rungs sits within ±1 grid step of the predicted
transition.) Secondary arms are reported under the same bar but do not gate the verdict;
`none` @ d_φ=32 doubles as a free sanity check — its prediction is "fails ALL four rungs" (rank
cap 32 < 96), and any `none` solve at ≥ 96 would falsify something deeper than this law.

**Kill conditions (pre-registered, any one triggers):**

- **K1 (fit-stage falsification):** ε̂ ∉ [0.25, 4.0] for the primary arm. ε was argued to be an
  O(1) constant; an extreme fitted value means the law has no predictive content beyond its fit
  rung. → law demoted, no adjudication run booked.
- **K2 (unidentifiable fit):** primary arm solves all of {16,32,64} or fails all → ε̂ not
  identified → outcome **INCOMPLETE** (reported as such; never counted as PASS).
- **K3 (non-monotone frontier):** an arm solves rung D_k but fails D_j < D_k under the frozen
  criterion → one re-run with fresh seeds; if it persists → protocol integrity failure, outcome
  INCOMPLETE + investigation (the law presumes a monotone solve frontier in D).
- **K4 (adjudication failure):** primary-arm agreement < 3 of 4 rungs → the law is **falsified as
  the predictive account** of the D-frontier. Flags A/B survive independently (they only require
  the Gershgorin corollary's scope to be corrected); the repo then states honestly that it has no
  predictive capacity account.

**Cost honesty:** ~63 runs total (27 fit + 36 adjudication) ≈ 12–18 A100-hours at the B1 budget;
the fit stage alone (~27 runs) is the cheap decisive filter — K1/K2 can kill the law for ~3–5 GPU-h
without touching the adjudication rungs.

## 6. Borrowed-vs-new ledger (this document)

| Item | Status |
|---|---|
| Gershgorin corollary `N < 1 + 1/cross(φ)` | repo, established **but scope-flagged** (Flags A/B) |
| Wigner row-noise spectral scaling ‖W‖₂ ≈ 2σ₂√N | **[B-method]** standard random-matrix reasoning |
| Crosstalk-noise-limited associative capacity | **[B-method]** classical Hopfield-line analysis; the σ₂-measuring probe + application to the rectangular delta state **[N]** |
| `1 + ε²/σ²`-form threshold | **[B-method]** analogy: May's n·σ² stability criterion; **[N]** as stated here |
| Mean/fluctuation split E = μ11ᵀ + W; common mode absorbed by readout | **[N]** derivation sketch (common-mode value validated numerically: 6.39e-3 vs 5.6e-3 predicted) |
| η(φ) = 1/(σ₂²·d_φ) efficiency metric; probe extension (μ, μ_signed, σ₂ both defs, η, N*) | **[N]** (random-floor calibration η_W(none) ≈ 1 measured) |
| ε as single fitted task tolerance + freeze-before-run protocol | **[N]** pre-registration discipline |

## 7. What this document does NOT claim

- No proof of the law; a derivation sketch with a calibrated instrument and one pre-registered
  falsification path.
- No claim that quad2's D=128 PASS is *explained* today — §3.4 shows the honest current status is
  "not yet explained by any bound in this repo"; the fit decides.
- No new GPU results, no re-analysis of existing artifacts, no claim about η-headroom
  monetization (full-quad2 / γ* arms are report 11 T3 and are NOT pre-registered here).
- No edits to `docs/preregistry/INDEX.md` (lane discipline): `PR-2026-09-03-02` merges by the
  maintainer from this document.
