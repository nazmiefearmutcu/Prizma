# 14 — Benchmark & Evaluation Methodology: the Decision-Grade Evaluation Program

**Commission:** Prizma → (1) inevitable Transformer rival + (2) as brain-like as possible
**Author role:** Benchmark & Evaluation Methodologist (pre-registration practice, statistical power,
seed variance, contamination control, fair-baseline design, evaluation ladders for sequence models)
**Date:** 2026-09-03
**Classification (per brainstorming SKILL.md):** **Architectural-class research-direction brainstorm.**
**HARD-GATE honored: report only — no code, no experiments launched, no files touched outside this report.**
**Repo state read in full:** `README.md` (incl. quarantine + B4 correction), `docs/PRIZMA_SEQ_REPORT.md`,
`docs/HANDOFF.md`, `results/campaign_2026-06-08/CONTAMINATION.md`, `seq/stats.py`, `seq/recall_gate.py`,
`seq/common.py`, `seq/tasks.py`, `seq/landscape.py` (docstring + arms), `seq/downstream.py` (head),
`calib_budget.py`, `seq/benchmark_results.md`, plus section-level skims of siblings 01–13.
**[B]** borrowed practice · **[N]** new to this repo · **[B→N]** borrowed component, new framing.

---

## 0. Executive frame: the evaluation debt IS the moat

Every sibling report proposes experiments. None of them can be cited until the *evaluation layer*
itself is decision-grade. The repo's two most damaging facts are both evaluation failures, not
architecture failures:

1. **B4 (char-LM): a pre-registered bar was not met, and the verdict briefly said PASS.** Both-corpus
   ≥3-seed bar delivered as one corpus at n=2, after the other corpus failed −0.09 and its raw data
   was not retained. Later text8 n=7 vs TF n=10 (Δ=+0.021, Welch t≈6.7) — still one corpus. This is
   *bar-shopping*: the only surviving sin in a repo whose entire brand is "pre-registered falsifiable bars."
2. **The n=10 recall gate is quarantined.** A resume cache keyed on seed alone contaminated 10/10
   cells with 4×-smaller smoke seeds (candidate arm with `quad2` OFF) *and* handed every arm its LR
   from a 2-point sweep on the wrong model. Conservative direction, but the repo's only powered,
   TOST-designed artifact is unusable. ~28 A100-hours fixes it.

The strategic conclusion: **Prizma cannot buy architecture credibility; it can only *earn* it with an
evaluation program nobody in the efficient-attention literature currently runs.** TOST equivalence
gates, flip-tests, identical-model negative controls, Holm-corrected verdicts, config-fingerprinted
resumes — this apparatus *already exists in the repo and is better than what most published
DeltaNet/SSM papers do*. The deficit is not apparatus; it is (a) unwrun experiments, (b) two process
holes (artifact retention, pre-registration discipline), (c) statistical power choices made per-crisis
rather than per-design. Fixing exactly those three things, at ~60 A100-hours of Colab credit, converts
"candidate" from self-description to claim.

Governing metric for every decision below: **information per GPU-hour.** The owner has limited Colab
A100/L4 credits; each campaign must be designed so that *any* outcome changes a documented belief.

---

## 1. The repair-first ledger (what MUST close before any new claim)

Ranked; each item states the hole, the repair, and the cost. **No new rung of the ladder (§2) opens
before its dependency here closes.**

| # | Hole | Why it blocks everything | Repair | Cost |
|---|---|---|---|---|
| H1 | Quarantined n=10 recall gate (`campaign_2026-06-08`) | The only powered artifact; "competitive" is a non-result forced by contaminated CIs | Clean re-run on the fixed `(seed, config-fingerprint)` resume path, **smoke and campaign writing to different result files** (the operational root cause; the fingerprint is necessary but file-separation is the cheap insurance) | ~28 A100-h (repo estimate) |
| H2 | `run_cell` init-before-`set_seed` pinning bug | B1–B3/B6 absolute solve-rates are not bit-reproducible; the report discloses this as open work | Seed before `model_fac`; re-run B1–B3/B6 once on the pinned path | ~15–20 A100-h |
| H3 | B4 PARTIAL: one corpus, under-seeded; tiny-shakespeare failure raw data **not retained** | Most quotable negative in the repo; fully self-inflicted | Diagnose the shakespeare overfitting recipe (dropout lever `16cdfdb` now exists, wd-only currently), then both corpora at n≥5; **guard band** on the contiguous text8 split (the <0.05% boundary leak is disclosed but free to eliminate) | ~8 A100-h (arm-times measured: 1889 s + 361 s per Prizma/TF arm-seed) |
| H4 | Surprise-gating ablation is n=2 smoke, verdict INCONCLUSIVE, ordering *constant > random > real* | The brand's one "new" mechanism is evidence-negative; it is the ceiling on siblings 01/03/04/05/06/09/12 and on every brain-likeness claim (§4) | Powered ablation: {surprise-norm, surprise-constant, surprise-random, none} × n≥5 seeds, LR-swept, on MQAR-D128 @130K + a char-LM leg | ~10 A100-h |
| H5 | GLA (`seq/gla.py`) and Mamba-2 (`seq/mamba2.py`): faithful, tested, **zero results ever produced** | Every comparison is vs the family Prizma claims to *leave* (attention), none vs the family it *belongs to* (linear/SSM). Reviewer question #1 has no answer | First-ever 4-arm landscape run via the never-run `seq/landscape.py` (recall legs + `--charlm` leg) | ~60 A100-h |
| H6 | No pre-registration registry; bars live in prose and can drift (the B4 mechanism) | Bar-shopping is *possible*; therefore every future bar is discountable | Registry (§5, proposal P2) — zero GPU | 0 |
| H7 | Artifact retention is aspirational; the B4 failure run's raw data vanished | Failures are the expensive half of information; losing them is paying full price for half | Binding retention policy (§5): every run (esp. failures) commits raw per-seed JSON + config + log + run manifest to `results/<campaign>/` | 0 |

**Repair-first total: ≈ 60–75 A100-hours.** This is squarely in the owner's demonstrated Colab-credit
regime, and it is the *only* spend that raises the value of everything the other 13 reports propose.

---

## 2. The evaluation ladder (rungs, arms, seeds, bars, budgets)

Design rules, all pre-registered per rung (see P2): **param-matched arms with disclosed FLOPs; per-arm
LR sweep with the full grid recorded; seed-pinned via `build_and_train`; frozen eval sets; plateau
early-stop with engagement floor 0.5; one PRIMARY endpoint per rung (locked before launch); all other
comparisons labeled exploratory; Holm family = the pairwise comparisons within one rung; TOST margins
anchored to measured seed-SD (§3).**

### L0 — Integrity rung (CPU/CI, ~0 GPU-h, permanent)
- Existing: kernel guards, `step==forward <1e-6`, lever-off==byte-identical, anti-conservative-stats
  gate, mixed-config aggregation assert, smoke-vs-campaign regression tests.
- **Add:** run-manifest in every results JSON (git SHA, config fingerprint, device, host, wall-clock,
  eval-set hash); identical-model negative control at *campaign* scale (not just landscape); a positive
  control (re-run one known cell each campaign; drift > tol ⇒ halt and audit).
- **Gate (canary bar):** any integrity test red ⇒ all citing claims freeze.

### L1 — Diagnostic recall gate (the repaired H1+H2; ~28 + ~18 A100-h)
- **Arms:** TF / Prizma-`quad2_lowrank` / Hybrid (3 arms × 3 legs) + TF-big flip-test, at d128L2H4
  param-matched (~461K/candidate ~464K), seeds 0–9, stage-1 LR sweep on the 5-point grid per arm per leg.
- **Legs & endpoints:** MQAR-HARD: TOST(±0.05) on accuracy + flip-test gate (clean attribution).
  INDUCTION: **primary = pre-registered extreme-margin solve-rate** (see §3.3 — bimodality makes mean-TOST
  unpowered here). SELECTIVE-COPY: TOST(±0.05).
- **Bar (pre-registered):** all three legs clean ⇒ the word "dominant" is earned; ≥2 of 3 ⇒ "Pareto-
  competitive (powered)"; <2 ⇒ "param-efficiency niche" and everything above rescope. **The downgrade
  branch is written down BEFORE the run** — it is what makes the pass credible.
- **Why n=10:** §3.1 shows n=10 resolves deltas ≈ 1 paired-SD; n=5 resolves ≈ 1.4 paired-SD — adequate
  for a ±0.05 margin only if paired SD ≤ 0.035, which is plausible at ceiling but unproven off-ceiling.

### L2 — Char-LM closure (H3; ~8 A100-h)
- **Arms:** TF / Prizma-v2 (gate-superset knobs) at d256L4H4 param-matched (−0.13–0.16% disclosed),
  dropout symmetric, n=5 per arm per corpus, val-selected test BPC, per-arm LR grid {2e-3, 3e-3}.
- **Corpora:** text8 (guard-banded split) + tiny-shakespeare (post-mortem the failed recipe first;
  report the diagnosis even if the rerun passes).
- **Endpoint (locked):** TOST on test BPC with margin ±0.05 — measured seed-SD ≈ 0.003–0.009 ⇒ n=5
  MDE ≈ 0.017 ≪ margin (§3.1): this rung is *cheap and decisive*, unlike the bimodal rungs.
- **Bar:** PASS = TOST-equivalent within ±0.05 on BOTH corpora (the original pre-registration, honoured
  at last). Anything else = the pre-registered honest verdict, whatever it is.

### L3 — SOTA landscape (H5; ~60 A100-h; apparatus complete, never run)
- **Arms:** TF / Prizma / GLA / Mamba-2 (optionally +2 nearly-free Prizma-variant arms per report 07
  R1: `inctx_lr`, `n_delta≥2`) on the recall legs + `--charlm` leg; Holm-corrected pairwise verdicts;
  identical-model negative control; param spreads disclosed per arm.
- **Endpoint (locked):** Pareto table + pairwise {BEATS/PARITY/WORSE/INCONCLUSIVE} per leg.
- **Bar:** no WORSE vs GLA and Mamba-2 on ≥2 of 3 recall legs, ≥1 BEATS somewhere ⇒ "belongs in the
  modern mixer family." WORSE on any ⇒ rescoped honest claim. **Either outcome is a first for the repo**
  and answers the reviewer question it currently cannot.

### L4 — Scale rung (50M; owner-budget decision; aligns with reports 07 R3 / 08 P3)
- **Arms:** TF / GDN / Mamba-2 / Prizma / Prizma-hybrid, param-matched, 5–10B tokens (FineWeb-Edu
  subset), μP-style LR transfer where feasible (report 08 P4), measured (not analytic) memory.
- **Endpoints:** loss-parity within pre-registered tolerance at matched params; throughput bar from
  report 08 P2 as a *separate co-registered* claim (never silently bundled).
- **Bar:** menu form — parity / within-margin / fail, each with its own wording. **This rung is the
  "inevitable" ceiling; it must not launch until L1–L3 close, or its results inherit the debt.**

### L5 — Downstream & long-context (only after L4; `seq/downstream.py` exists, never run)
- Few-shot (HellaSwag/ARC-easy/MMLU-mini/GSM8k-closed-form) + RULER/NIAH at 32k–128k + measured decode.
- **Bar:** report-only until L4 passes its menu; MMLU at 50M is near-chance for *all* arms — pre-register
  that reality (task floor) so the rung cannot produce a fake ranking.

### L-B — Brain-alignment add-on (§4; rides on L4; +3–5 GPU-h inference, zero training)

---

## 3. Statistical methodology (the part that makes the bars *bind*)

### 3.1 Power / MDE discipline — seed counts chosen by arithmetic, not tradition
For a two-sample Welch test at α=0.05, power 0.80: **MDE ≈ 2.96 × SD × √(1/n₁+1/n₂)**; with n=10+10 that
is ≈ 1.32·SD; **paired** (same seed both arms — legitimate here: identical frozen eval set, seed-pinned
data streams) with the usual positive cross-arm correlation ρ: SD_d = SD·√(2(1−ρ)) and n=10 pairs gives
**MDE ≈ 1.0 × SD_d** (ρ=0.6 ⇒ ≈0.9·SD). Consequences the program adopts:
- **Pair the seeds everywhere** (report paired-difference CIs alongside arm summaries) — free power.
- n=10 on near-ceiling legs (SD≈0.002–0.01) resolves 0.001–0.01: TOST ±0.05 is over-powered — good.
- n=10 on bimodal legs (SD≈0.3–0.4) resolves nothing below ~0.4: **mean-based TOST is the wrong
  endpoint there** — switch to solve-rate (below). This single observation explains the quarantined
  campaign's "TOST failed everywhere with absurd CIs" pattern and must be in the registry.
- Every campaign doc gets a **one-line MDE table**: assumed SD (from the campaign's own calibration
  runs, `calib_budget.py` precedent) → required n for the locked margin. If required n exceeds budget,
  the bar changes *before* launch or the rung does not run.

### 3.2 Endpoints must survive the distribution, not just the mean
- **Ceiling legs** (MQAR param-matched, selcopy): TOST on accuracy is correct; report seed ranges +
  paired-difference CI (the repo already learned the z-CI lesson — `seq/stats.py` R1).
- **Bimodal legs** (TF induction collapses ~3/8 uncontaminated seeds): primary endpoint =
  **solve-rate at 0.9**, compared with exact tests (Fisher / Newcombe-Wilson) — the repo's Wilson-CI
  culture extended to between-arm contrasts. At n=10 only *extreme* separations are detectable
  (e.g., ≥9/10 vs ≤3/10 ⇒ Fisher p≈0.011); pre-register exactly that extreme-margin form and accept
  that moderate differences are **INCONCLUSIVE by design, not by accident**.
- **Collapse must be reported as a phenomenon, not averaged away:** for bimodal legs also report
  ignition-step distributions (the `steps_to_plateau` audit field exists) — the honest signal that
  both architectures have phase transitions.

### 3.3 Equivalence margins get anchored, once
±0.05 accuracy and ±0.05 BPC are inherited, not derived. In the registry each margin carries its
anchor: BPC ±0.05 ≈ 5× the largest measured seed-SD (generous to the candidate ⇒ parity claims are
hard, not easy); accuracy ±0.05 ≈ the param-efficiency grid resolution at 130K–461K. Margins are
**never tightened or loosened after data**; a new margin = a new campaign.

### 3.4 Multiple comparisons without self-strangulation
- Families are declared per rung: L1 has 3 legs × ~2 arms = small; L3 pairwise family is already
  Holm-corrected in `landscape.py`. Primary endpoint = 1 per rung; everything else labeled
  exploratory and *reported but not claimed*. This is the cheap alternative to gaming p-values.
- **Negative controls** (identical-model, and `rand_linear`-style task controls) run inside every
  campaign — the repo already has the machinery; make them mandatory rows in the verdict JSON.

### 3.5 Sequential rules so pre-registration doesn't strangle iteration (the two-lane system)
- **Lane E (exploratory):** unlimited, cheap, n=2–3 triage runs, freely iterated, clearly labeled
  "E — not citable." This is where iteration lives.
- **Lane C (confirmatory):** registry entry, locked bar + endpoint + n + margins, **one-shot**.
- **Peeking without sinning:** group-sequential α-spending, O'Brien–Fleming style — with a planned
  interim at n=5: early-stop *for harm* allowed at interim p<0.005 (candidate worse beyond 2× margin ⇒
  halt, rescope, don't burn the second half); early-stop *for success* only at p<0.003; final decision
  at n=10 with α=0.047. Exact thresholds go in the registry entry before launch.
- **The B4 lesson as a rule:** Lane-E results may *motivate* a Lane-C entry; they may never *become*
  a claim. Converting lanes is the one act the registry is designed to make visible and shameful.

### 3.6 Contamination control is now three layers deep — keep all three
1. **Fingerprint resume** (done, tested) — necessary.
2. **File separation:** `--smoke` and full campaigns write to *different* result paths by default
   (the 2026-06-08 root cause was shared-path). One-line fix class, prevents the whole bug family.
3. **Audit hooks at aggregation time:** the param-count and fingerprint asserts in `recall_gate._train_arm`
   must be mirrored in every runner (`landscape`, `charlm`, any L4 harness) — a mixed cell should crash,
   never publish. Plus the L0 manifest + positive-control canary.

---

## 4. Brain-likeness evaluation: what is measurable, cheap, and honest

### 4.1 What the 2021–2026 literature actually does (grounding)
- **Brain scores / neural predictivity** (Schrimpf et al. 2021, PNAS "The neural architecture of
  language"; the **Brain-Score language** platform): fit a **ridge encoding model** from per-word LM
  representations to fMRI voxels / EEG sensors / reading times on public corpora (Pereira-2018 fMRI,
  Blank-2014 fMRI, Futrell-2018 reading times, Brennan-2019 EEG on the platform); score = held-out
  Pearson r (often ceiling-normalized), median across subjects. Inference-only: **no training, ~GPU-hours.**
- **Surprisal as the bridge variable:** word surprisal from an LM predicts N400-like EEG amplitudes and
  fMRI response in language regions (Frank-era reading-time work; Goldstein et al. 2022 word-level
  fMRI next-word prediction; Heilbron et al. 2022 Nat. Hum. Behav. — a *hierarchical*, multi-timescale
  predictive-coding account fits better than flat next-word objective; Caucheteux & King on depth and
  timescale structure).
- **2024–2026 refinements:** alignment is *not* reducible to next-word-prediction ability (EMNLP 2024,
  "…align due to more than next-word prediction"); larger LLMs align better on attention/saccade
  structure (Gao et al. 2025); and an interpretive backlash exists — prediction scores alone are being
  criticized as weak explanatory evidence (arXiv 2605.14025, "Prediction Scores Are Not [the story]").
  **Design implication:** a brain bar must be *mechanistic*, not just a leaderboard number.

### 4.2 The honest cheapness analysis
- Brain-score evaluation is **inference-only** (≈3–5 GPU-h at 50M scale: extract per-word layers, fit
  ridge, bootstrap CIs over subjects) — but word-level corpora require a **word-level model**, i.e. it
  rides on L4 (50M rung). At the current ≤1.4M char-level scale, a brain-parity claim would be
  **methodologically fake** (char-LMs cannot be scored on word-timed corpora without ad hoc surgery).
  **Pre-registered position: the brain bar activates at L4, never before.**
- What makes Prizma *uniquely* scoreable here — and cheap — is that its **novel signals are exposed
  scalars**: per-token write gate β_t, per-token delta error ε_t, per-head state decay. Three
  mechanistic hypotheses, each a regression, all inference-only:
  1. **β_t / ‖ε_t‖ ↔ N400 amplitude** (EEG, e.g. Brennan/Narratives-EEG): does the delta write's
     surprise track the human surprise response better than TF surprisal does? (Pre-register: partial
     correlation controlling for word length, position, frequency, lexical frequency.)
  2. **Layer-timescale profile ↔ Heilbron hierarchy:** per-layer effective memory (decay-spectrum
     statistics, report 09's observability audit) vs the brain's known timescale gradient — a
     correlation-without-training test of the "predictive-coding hierarchy" framing.
  3. **Brain-score parity:** Prizma's encoding r within the bootstrap CI of the param/FLOP-matched TF
     on ≥2 of {fMRI-language, EEG, reading-time} benchmarks. Parity bar, not superiority — superiority
     would be great and is *not* required for the claim to matter.

### 4.3 The pre-registered bar (realistic version)
> **B-BRAIN (activates at L4):** On Brain-Score-language's fMRI + EEG benchmarks, Prizma's neural
> predictivity is within the 95% bootstrap-over-subjects CI of the param/FLOP-matched TF, **and** on
> ≥1 mechanistic test (β_t↔N400 or timescale-hierarchy), Prizma's signal explains significant
> additional variance over TF surprisal (one-sided, Holm across the 2 mechanistic tests).
- **Cost:** ~3–5 GPU-h + CPU ridge fits. **Falsifiable test:** it either passes or it doesn't; the
  mechanistic half can fail *while* parity passes, and each outcome is separately reportable.
- **Risks:** encoding-score methodology is under active criticism (§4.1) — the bar therefore *pairs*
  the descriptive score with the mechanistic test; small models may simply have low absolute r
  (report the GPT-2 reference row so the number is interpretable); EEG corpora licensing/curation is
  an engineering cost outside my lane.
- **Impact on "inevitable":** moderate alone (6), but it is the **only existing quantitative framework**
  in which "brain-like" is a falsifiable claim rather than a metaphor — which is exactly what this
  commission's second goal needs (reports 01/02/03/05/09 all propose mechanisms this bar would test).

---

## 5. Red-teaming the evaluation itself

**Attacks on my own program, with defenses designed in:**

1. **Bar-shopping (the B4 failure mode).** Defense: the **pre-registration registry** (P2): dated,
   hash-committed bar text (`committee/prereg/2026-xx-xx-<rung>.md`, git commit as timestamp); any
   deviation stamped `DEVIATION:` in the verdict (the B4 verdict now does this — generalize it);
   loosening a bar post-hoc is prohibited; tightening is allowed but the original must still be reported.
2. **"Conservative error" complacency.** The contamination bug *happened* to hurt the candidate; that
   direction was luck, not design. Defense: audits are direction-blind (the CONTAMINATION.md analysis
   is the model); every campaign carries the negative control + positive-control canary so drift is
   caught regardless of whom it flatters.
3. **Baseline sabotage (subtle strawmanning).** Per-arm LR sweeps with the *full grid recorded and
   published for both arms* (lrsweep audit exists), plateau-stop with engagement floor identical across
   arms, and the flip-test prevent "the baseline was under-trained" attacks. Defense-in-depth: the
   referee panel reviews the sweep table, not just the headline.
4. **Evaluation-set overfitting.** Frozen eval sets exist for diagnostics; for LM rungs, add a
   **guard-banded split** (text8 boundary leak is disclosed but free to remove) and keep one
   **held-back eval variant** per rung (e.g., a second eval_seed never used during development;
   report both; if they disagree > tol, the rung's verdict auto-downgrades to INCONCLUSIVE).
5. **Endpoint drift.** Locked primary endpoints (§2) + MDE tables filed pre-launch. If an endpoint
   turns out unpowered (bimodal SDs), the verdict is INCONCLUSIVE-by-design — which is a *result*,
   and a respectable one.
6. **Selection on the triage lane.** Lane-E runs are n=2–3; selecting the best of them for Lane C is
   legitimate **only if the Lane-C bar is registered against the *selected* config, and the number of
   Lane-E candidates tried is disclosed** (the garden-of-forking-paths disclosure; one line in the
   registry: "selected from k exploratory configs").
7. **Compute-shopping.** Different arms on different hardware/seeds-by-availability. Defense: one
   campaign = one device class + manifest fields per cell; cross-device results never pooled (the
   recall-gate CPU/GPU mixture is the cautionary tale — the smoke seeds were also *different-device*
   seeds).

---

## 6. Proposals (ranked; each IS a falsifiable test)

### P1 — The Repair Campaign: clean recall gate + init-pinned B1–B3/B6 re-run **[B]** (repo's own apparatus)
- **WHAT:** H1+H2 in one Colab campaign: the n=10 TOST recall gate on the fingerprinted path (smoke
  separated into its own results file), then B1–B3/B6 re-run seed-pinned.
- **WHY:** everything else in this commission is priced in a currency ("candidate", "dominant") this
  repair mints. Until it runs, the repo's powered-claims balance is zero.
- **HOW:** order cells by information-per-hour: MQAR-HARD first (with flip-test; it's the decisive leg),
  then selcopy, then induction; B6 re-runs last (d64 cells are ~50 min each). Sequential rule from §3.5.
- **FALSIFIABLE TEST:** pre-registered bar — TOST(±0.05) clean on MQAR-HARD + SELECTIVE-COPY and
  extreme-margin solve-rate on INDUCTION ⇒ "dominant" is earned; ≥2/3 ⇒ "powered Pareto-competitive";
  <2/3 ⇒ "param-efficiency niche" (the downgrade wording ships with the launch).
- **RISKS:** outcome may be the downgrade branch (that is a feature); ~45 A100-h across two sessions;
  Colab disconnects — the crash-safe resume exists, use fresh campaign files.
- **IMPACT 10 · FEASIBILITY 8** (~45 A100-h).

### P2 — Pre-registration Registry + Two-Lane Policy **[B→N]** (clinical-trial/OSF practice; new to repo)
- **WHAT:** `committee/prereg/` with one immutable, git-hash-committed file per campaign: bar, primary
  endpoint, arms, seeds, margins with anchors, MDE table, sequential rules, downgrade wording. Lane-E
  results may motivate but never become claims (§3.5, §5.1).
- **WHY:** it is the structural fix for the only sin the repo ever committed (B4). Cheap insurance
  against the *next* B4, and it converts "trust us" into "check the hash."
- **FALSIFIABLE TEST:** procedural — every future campaign either has a registry entry or its claims
  are labeled exploratory; a claim without an entry is by definition over-claiming and gets retracted
  (one retraction rule, applied once to B4 historically, makes the policy credible).
- **RISKS:** bureaucracy cost (~1 h per campaign); temptation to write vague bars — mitigate with the
  template requiring exact numbers.
- **IMPACT 8 · FEASIBILITY 10 · COST 0 GPU-h.**

### P3 — Powered Surprise-Gating Ablation (the mechanism verdict) **[B]** (apparatus exists)
- **WHAT:** {surprise-norm, surprise-constant, surprise-random, none} × n≥5, LR-swept, on MQAR-D128
  @130K + char-LM leg; identical-model control. Constant and random gates are the load-bearing controls
  (their smoke-run point estimates currently beat the real signal).
- **WHY:** the single cheapest experiment in the entire commission and the **ceiling on every
  brain-likeness claim** (§4) and on siblings 01/03/04/05/06/09/12. Also the honest answer to red-team
  T7 (report 13).
- **FALSIFIABLE TEST (pre-registered):** mechanism SUPPORTED iff surprise-norm beats max(constant,
  random) with Holm-corrected p<0.05 AND beats none by ≥ the anchored margin; otherwise the ledger row
  flips to "mechanism refuted at tested scale" and the PC framing is retired to derivation-only.
- **RISKS:** likely outcome is refutation or null (the smoke points run against it); that is still the
  correct spend — an architecture whose brand mechanism is evidence-negative cannot be "inevitable."
- **IMPACT 9 · FEASIBILITY 9** (~10 A100-h).

### P4 — First-Ever 4-Arm SOTA Landscape Run **[B]** (harness complete, never executed)
- **WHAT:** `seq/landscape.py --full` + `--charlm`: TF/GLA/Mamba-2/Prizma, recall legs + char-LM,
  Holm-corrected pairwise verdicts + negative control (optionally the 6-arm extension of report 07 R1).
- **WHY:** answers the one question every reviewer asks ("how does it do against Mamba-2?") that the
  repo today cannot answer at all. Positions Prizma as a member of the winning family, not a
  Transformer-obsessed outsider.
- **FALSIFIABLE TEST (pre-registered):** no WORSE vs GLA and Mamba-2 on ≥2/3 recall legs with ≥1 BEATS
  somewhere ⇒ family-membership claim earned; any WORSE ⇒ the rescoped wording, published.
- **RISKS:** biggest budget item (~60 A100-h); Mamba-2/GLA tuning may need iteration (their sweeps are
  recorded, so iteration is honest); outcome may be "WORSE on induction" — still the first real number
  in the repo's landscape.
- **IMPACT 9 · FEASIBILITY 6** (~60 A100-h; run after P1 reuses its repaired protocol).

### P5 — Statistical Protocol Upgrade: paired endpoints, MDE tables, anchored margins, α-spending **[B→N]**
- **WHAT:** codify §3 into the verdict layer: paired-difference CIs; endpoint-per-leg locked in the
  registry; MDE table per campaign; margin anchors; O'Brien–Fleming interim rules; extreme-margin
  solve-rate endpoint for bimodal legs.
- **WHY:** the current stats are *correct* (`seq/stats.py` is genuinely better than most papers') but
  the *design* choices (n, endpoint, margin) were made ad hoc. This makes the numbers bind.
- **FALSIFIABLE TEST:** every campaign doc ships the MDE line "assumed SD → required n for margin m";
  a campaign without it fails the registry check (P2).
- **RISKS:** paired analyses need same-seed pairing discipline (build_and_train already pins seeds —
  the plumbing is there).
- **IMPACT 7 · FEASIBILITY 9 · COST 0 GPU-h.**

### P6 — B4 Closure + Artifact-Retention Policy **[B]** (bar exists; policy is new-to-repo)
- **WHAT:** shakespeare post-mortem (diagnose the −0.09 overfitting recipe; report it either way),
  then both corpora × n≥5 × symmetric dropout × guard-banded split (~8 A100-h). Bind the policy: every
  run — especially failures — commits raw per-seed JSON + config + log + manifest; deletion of a raw
  artifact is a repo-level sin.
- **WHY:** B4 is the most quotable negative and the cheapest positive to close. The retention policy
  is what makes every future failure recoverable.
- **FALSIFIABLE TEST:** the original bar, honoured: TOST(±0.05 BPC) on BOTH corpora. n=5 MDE≈0.017 ≪
  0.05 ⇒ decisively decidable for ~8 GPU-h.
- **RISKS:** shakespeare may fail again under the honest recipe — then B4 stays PARTIAL with two
  documented corpus results, which is still strictly stronger than today.
- **IMPACT 8 · FEASIBILITY 9.**

### P7 — Brain-Alignment Add-On Bar (activates at L4) **[B→N]** (Brain-Score platform borrowed;
mechanistic surprise/N400 and timescale-hierarchy bars are new to the repo)
- **WHAT:** §4.2–4.3: encoding-model brain-score parity vs matched TF on ≥2 benchmarks + ≥1 mechanistic
  test (β_t↔N400, decay-spectrum↔hierarchy), ~3–5 GPU-h inference at the 50M rung.
- **WHY:** the only quantitative, falsifiable form of "brain-like" available; inference-only cost; and
  it tests Prizma's *own* exposed signals, where TF has no analog — the one axis where a delta-state
  model can generate a headline a Transformer structurally cannot.
- **FALSIFIABLE TEST:** the B-BRAIN bar of §4.3, verbatim, registered at L4 launch.
- **RISKS:** needs a word-level model (dormant until L4); encoding-score methodology is under criticism
  (mitigated by the mechanistic pairing); subject-count limits power — bootstrap over subjects is the
  honest CI, and n<6 subjects ⇒ report as descriptive only.
- **IMPACT 6 (10 for the brain-like goal) · FEASIBILITY 6** (gated on L4).

### P8 — Information-per-GPU-Hour Scheduler + Evaluation Red-Team Checklist **[N]** (process)
- **WHAT:** (a) every campaign cell ordered by decision-value (decisive leg → controls → saturation
  legs); n=2 Lane-E triage before any Lane-C spend; (b) a one-page red-team checklist (§5) run by the
  referee panel *on the evaluation design before launch*, not only on results after.
- **WHY:** with limited Colab credits, the scarcest resource is owner hours + credits; a triage-first
  order routinely halves the cost of answering the same question (the flip-test-first ordering in
  `recall_gate.py` is exactly this pattern — generalize it).
- **FALSIFIABLE TEST:** procedural — no Lane-C launch without a triage result + checklist sign-off;
  measure: campaign aborts/mid-run rescopes drop to zero.
- **RISKS:** over-triage can bias toward cheap questions — mitigate by pre-committing the full ladder.
- **IMPACT 7 · FEASIBILITY 9 · COST 0 GPU-h.**

### Ranked summary
| Rank | Proposal | Impact | Feas | GPU-h | Type |
|---|---|---|---|---|---|
| 1 | P1 Repair campaign (clean gate + init-pin re-run) | 10 | 8 | ~45 | B |
| 2 | P2 Pre-registration registry + two-lane policy | 8 | 10 | 0 | B→N |
| 3 | P3 Powered surprise ablation (mechanism verdict) | 9 | 9 | ~10 | B |
| 4 | P4 4-arm SOTA landscape first run | 9 | 6 | ~60 | B |
| 5 | P6 B4 closure + retention policy | 8 | 9 | ~8 | B |
| 6 | P5 Statistical protocol upgrade | 7 | 9 | 0 | B→N |
| 7 | P8 Triage scheduler + eval red-team checklist | 7 | 9 | 0 | N |
| 8 | P7 Brain-alignment add-on bar (at L4) | 6 (10 for brain axis) | 6 | ~5 | B→N |

**Phase-1 spend (P1+P3+P6, all decisive, all cheap): ≈ 63 A100-hours.** P4 next (~60). P7 piggybacks L4.

---

## 7. Coverage map: how this program gates the siblings' experiments

| Sibling proposal | My gate |
|---|---|
| 03 P1 / 04 P1,P4 / 05 P1–P4 / 06 P2,P3 / 09 / 12 H4 build on the surprise signal | **P3 verdict sets their ceiling** — do not build on it before P3 lands |
| 07 R0–R5 ladder | Adopted as my L1–L5; my additions: two-lane policy, MDE tables, paired endpoints, retention policy |
| 08 P2 throughput bar | Co-registered but **separate** claim at L4 (never bundled into loss-parity — that bundling is how B4 happened) |
| 08 P3 / 07 R3 50M campaign | L4; carries the P7 brain add-on and the downstream guard rails |
| 11 T1/T2 D-frontier predictions | Diagnostic-leg extension at L1–L3; blind-predict D* from crosstalk in the registry, then run — the repo already did this correctly once (D*=32 pre-registration) |
| 02 P6 / 06 P3 surprise-modulated write | Same P3 gate |
| 10 P1 regime map, P4 citation bar | CL ladder is orthogonal; same registry + retention policy apply; E1 reruns must seed-pin (H2 fix) |
| 12 H2/H3 quantization & fault bars | Cheap CPU-side emulation; registry + canary discipline apply; nothing blocks them |
| 13 T4–T7 objections | P1 (evidence debt), P3 (T7), P2 (process); the "commodity" T1/T2/T3 objections are architecture questions outside my lane |

---

## QUESTIONS FOR THE OWNER

1. **Budget split:** do you authorize the ~63-A100-hour Phase-1 (P1 repair + P3 surprise + P6 B4) as
   the next three Colab campaigns, in that order? It is the minimum spend that makes any existing claim
   citable.
2. **Downgrade pre-commitment:** if the clean recall gate lands <2/3 legs, do you accept "param-efficiency
   niche" as the published wording *now* (pre-committed), so the result cannot be argued with later?
3. **B4 corpora:** is tiny-shakespeare worth the post-mortem, or would you substitute a second corpus
   (e.g., enwik8) for the both-corpora bar? (Substitution = a registry amendment, not a silent swap.)
4. **Landscape timing:** run P4 (~60 A100-h) immediately after Phase-1, or defer until a Triton kernel
   exists so throughput is not a confound in the landscape's char-LM leg?
5. **Registry policy:** are you willing to make the pre-registration registry binding on yourself
   (no claim without an entry, deviations stamped)? It only works if it binds the owner first.
6. **Brain bar appetite:** when the 50M rung (L4) becomes affordable, do you want the P7 brain-alignment
   add-on registered alongside it (+5 GPU-h), or should "brain-like" stay a design metaphor until then?
