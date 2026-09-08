# PR-2026-09-03-01 — Powered surprise-gating ablation (retire-if-negative)

| field | value |
|---|---|
| Registry id | **PR-2026-09-03-01** |
| Date written | 2026-09-03 |
| Lane | **CLAIM** (two-lane policy, [`POLICY.md`](POLICY.md)) |
| Status | **REGISTERED-FROZEN** (protocol frozen before any claim-grade run; runs only when GPU budget exists — owner decision: GPU = 0 this month) |
| Owner decision | Committee synthesis §6.4 / §C1 (`committee/brainstorm_2026-09-03/00_COMMISSION_SYNTHESIS.md`): the surprise-gating mechanism is **pre-registered retire-if-negative**. This document operationalizes that decision verbatim. |
| Primary artifact (to be written by the run) | `results/surprise_ablation_powered.json` |
| Claim ledger row | `docs/preregistry/INDEX.md` row `PR-2026-09-03-01` |

> **What this document is.** A LANE-CLAIM pre-registration under
> [`POLICY.md`](POLICY.md): the bars, arms, seeds, statistics, compute budget, and the
> pre-committed negative-result handling are written **before** the run and are **frozen**.
> Per the immutability rule, once results exist this text cannot be edited — only dated
> addenda may be appended. The protocol below executes verbatim or not at all.

---

## 1. Motivation & hypothesis

**The claim under test.** Prizma-Seq's delta write is
`u_t = β_t · ε_t` with prediction error `ε_t = v_t − α_t·S_{t−1}k_t` (with the diagnostic
`gated=False`, `ε_t = v_t − S_{t−1}k_t`). The repo's one claimed-novel mechanism is that the
write gate `β_t` should be **derived from the surprise signal** (`ε_t`): tokens with large
prediction error should be written more/less informatively than a fixed or random schedule
would write them — the predictive-coding/free-energy story ("precision/surprise signal used
causally for gating", `docs/PRIZMA_SEQ_REPORT.md` "Borrowed vs new ledger").

**Honest status of the existing evidence — it points AGAINST the mechanism.** The only
ablation run to date (`results/gpu_ablation.json`, `"smoke": true`) is **n=2, underpowered,
and every verdict is INCONCLUSIVE** — it establishes nothing. Its point estimates, however,
run *against* the mechanism: the **constant** gate scored highest (mean 0.566), then the
**random** gate (0.527), then the actual surprise signal (0.517), over the ungated baseline
(0.406). A constant gate beating the real signal is the opposite of what the claim needs.
The report already says this in the correction block of
`docs/PRIZMA_SEQ_REPORT.md`; this pre-registration does not reinterpret that smoke — it
supersedes it with a powered test whose outcome binds, whichever way it lands.

**What the shipped gate actually is (honest scoping of the mechanism).** The default write
gate in `seq/prizma_seq.py` is `precision_gate="input"`: `β_t = β_cap·σ(W_β x_t)` — a
**learned input-dependent gate**. It is *not* literally surprise-gated: `x_t` is the token
representation, not the state prediction error. The novelty claim is therefore specifically
that a *surprise-derived* gate would be better/more principled than such learned or
input-independent gates. That is the hypothesis tested here.

**H1 (the mechanism's hypothesis, falsifiable):** a write gate derived from the normalized
write-error energy carries information about *which* tokens to write, and therefore beats
both an input-independent **constant** gate and an input-independent **random** gate on
associative-recall tasks.
**H0:** it does not — any benefit of having *a* gate is fully captured by a constant or
random schedule (or by the learned input gate).

**Retire-if-negative (owner decision 4, binding):** the mechanism is claimed only if it
**survives** the pre-registered decision rule in §4. On any non-survival outcome the
surprise-gating mechanism is **retired from all README novelty claims**; relocation of the
surprise signal to the decay gate `α` may be proposed **only as a NEW pre-registration**,
never as a retroactive save of this one.

**Honest asymmetry (stated now, not after the run):** retirement does not prove the
mechanism cannot ever work — it removes the novelty *claim*. A claim needs positive
evidence; no evidence = no claim = retired. A null caused by low power is still a null for
claiming purposes, which is why the MDE table in §4 is part of the frozen protocol: the
community can see exactly what effect sizes this design could and could not have detected.

---

## 2. Exact arms

All four arms are Prizma-Seq (`PrizmaSeqLM`) at the **same scale** `d_model=64, n_layers=2,
n_heads=2` (`d_h=32`, ~130K params, the B6/P2eff causal-ablation scale), differing **only**
in the write-gate source. The gate is the per-head scalar `β_t ∈ (0, β_cap)` applied to the
delta write. `β_cap = 0.99` (existing `PrizmaSeqConfig.beta_cap`) in all arms.

**Shared config (identical across arms; every value pinned here):**
`feat_map="quad2_lowrank"` (`feat_rank=0` → effective `r=14`, `d_φ=137`, 0 trainable params),
`gated=False` (α = 1 — the diagnostic-task convention, so `ε_t = v_t − S_{t−1}k_t` exactly),
`decoupled_gate=False` (erase gate = write gate), `inctx_lr=False`, `n_delta=1`,
`surprise_gate=False` (the Lever-A multiplier machinery is NOT used; see A4),
`write_mode="delta"`, `use_workspace=True`, `use_window=True`, `route_readout=True`,
`short_conv=4`, `window=16`, `chunk=64`, `rope=False`, `learned_pos=False`, `out_gate=False`,
`state_norm=False`, `banded_window=False`, `dropout=0.0`.

Rationale for `feat_map="quad2_lowrank"`: the gate ablation must run where the carried state
actually solves recall (P2eff precedent: quad2-family maps solve MQAR D=128 at d64L2H2 3/3;
`feat_map="none"` fails there) — a floor-effect regime would make every leg uninformative.
The map is identical in all four arms, so it cannot explain an inter-arm difference. Results
are scoped to this base and this scale; nothing is claimed about the gate on other bases.

### A1 — `input` (the incumbent default)
- `PrizmaSeqConfig(precision_gate="input")`: `β_t = β_cap · σ(W_β x_t)`, `W_β ∈ R^{d×H}`
  learned. A learned, input-dependent gate. **Zero new code** (existing default knob).
- Role: the incumbent. It is a *context* arm: it shows where the shipped model sits. It does
  NOT enter the survival rule (§4) — the owner decision's rule consults only the constant and
  random controls. Its result is reported descriptively and constrains claim wording (§4,
  honest limits).

### A2 — `uniform` (the constant control)
- `PrizmaSeqConfig(precision_gate="uniform")`: `β_t = β_cap · σ(w_h)` per head `h`, where
  `w_h` is a single learned scalar (`beta_logit`). Input-independent, **learned constant**.
  **Zero new code** (existing B6 control knob).
- Honesty note: this is the *strongest* version of a constant control — the model may learn
  its own optimal constant. If the mechanism cannot beat this, it cannot beat anything a
  constant schedule offers. This makes the bar harder for the mechanism, i.e. conservative.

### A3 — `random` (the random control)
- `PrizmaSeqConfig(precision_gate="random")`: `β_t = β_cap · U[0,1)` i.i.d. per
  (batch, head, token) per forward, input-independent, no parameters. **Zero new code**
  (existing B6 control knob). Runs are reproducible under `set_seed` (global RNG stream).
- Role: separates "the signal helps" from "a high-variance write schedule helps".

### A4 — `surprise_norm` (the mechanism under test) — NEW code, spec frozen here
The gate is computed **inside the recurrence** from the running state, replacing the learned
gate entirely. One formula, frozen (report 04's EMA-normalization proposal, instantiated):

For each head (all quantities per batch-element `b`, head `h`; `t` = token index within the
sequence):

1. `s_t = ‖ε_t‖₂²` — the write-error energy, `ε_t = v_t − α_t·S_{t−1}φ(k_t)` computed at the
   current state **before** the write (with `gated=False`, `α_t = 1`).
2. EMA baseline, causal within the sequence: `m_1 = s_1`; for `t ≥ 2`:
   `m_t = (1−λ)·m_{t−1} + λ·s_t`, with **λ = 0.02 frozen a priori** (≈50-token effective
   horizon, context-scale for `T ∈ {256, 384}`; never tuned).
3. Normalized surprise: `s̃_t = s_t / max(m_t, 1e−6)`.
4. **The frozen gate: `β_t = β_cap · σ( a · (s̃_t − 1) )`**, with σ the logistic function and
   the gain `a` selected ONCE on the exploratory seed (below), then frozen.

Properties (why this form): scale-free (invariant to the overall error magnitude, which the
tanh-‖ε‖ variant of the smoke was not), rank-preserving (monotone in `s_t`), self-calibrating
(`s̃_t = 1` ⇒ `β_t = β_cap/2`; a token with twice its running-average error gets
`β_t = β_cap·σ(a)`), and differentiable (β_t is a function of the state, so gradients flow
through the gate — the mechanism must *learn to exploit* the signal, not receive it free).

**NO-TUNING rule (binding).** The gain `a` is set on **ONE exploratory seed** and then frozen:
- Exploratory procedure: seed **900** (not a claim seed), task **MQAR-D64 only**, recipe
  `lr = 1e-3` (the GENWARM default), candidate set **a ∈ {0.5, 1.0, 2.0, 4.0}** (four runs).
  Select the `a` with the highest `best_acc` on the frozen eval; **ties → smallest `a`**.
- The selected `a` is then frozen for **all** claim seeds (n=5), **all** arms' A4 runs, and
  **all** three tasks. No re-selection, no per-task gains, no post-hoc adjustment.
- The four exploratory runs' results go to `results/exploratory/surprise_gain_selection.json`
  and are **LANE-EXPLORATORY: never a claim**, never cited as evidence about the mechanism
  (POLICY.md). They exist only to pin `a` before claim data exists.

**Implementation spec (enough to build without further questions; an implementation agent
writes this code — this pre-registration itself writes none):**
1. `PrizmaSeqConfig`: allow `precision_gate="surprise_norm"` in the existing assertion; add
   fields `surprise_gain: float` and `surprise_ema_lambda: float = 0.02` (the gain supplied
   per-run from the frozen protocol value; defaults must leave every existing mode
   byte-identical).
2. `PrizmaSeqBlock._encode`: for `precision_gate="surprise_norm"` return `beta=None`, after
   asserting `surprise_gate=False`, `inctx_lr=False`, `decoupled_gate=False`, `n_delta==1`
   (the one-novel-lever rule; the existing assert pattern in `__post_init__` covers this).
3. `seq/delta.py`: when `beta is None` under this gate, route to the **exact sequential scan**
   (`_delta_reference`), exactly as `surprise=True` already does (the WY/UT chunk-parallel
   form is invalid: β_t depends on the running state via ε_t — the R3 repeated-key argument
   in the `delta.py` docstring applies verbatim). Inside the erase branch compute, per head:
   `Sk = S_{t−1}φ(k_t)`, `ε_t = v_t − α_t·Sk`, `s_t = ‖ε_t‖²`, update `m_t`, then
   `β_t = β_cap·σ(a·(s_t/max(m_t,1e−6) − 1))` and
   `u_t = β_t·v_t − β_t·(α_t·Sk)` (erase gate = write gate, `decoupled_gate=False`).
4. `PrizmaSeqBlock.step()`: mirror the computation exactly; extend the per-block streaming
   state tuple with the per-head EMA `m` (init zeros; first token sets `m = s`). The
   `step()==forward()` O(1) guard must hold to <1e-4 within a sequence.
5. Required tests (writer adds them; they do not exist today — `tests/test_surprise.py`
   covers the Lever-A multiplier machinery, **not** the `precision_gate` knobs):
   (a) off-identity — every existing `precision_gate`/`surprise_gate` path byte-identical
   after the change (existing suite must stay green); (b) `step()==forward()` <1e-4 for the
   new mode; (c) monotonicity — larger `s_t` at fixed `m_t` yields larger `β_t`;
   (d) repeated-key exactness vs `_delta_reference` mirrors the existing R3 test.
6. Runner: `gpu_surprise_ablation.py`, a thin variant of `gpu_ablation.py` reusing
   `seq.gpu_harness` (`make_arm`, `make_cfg`, `sweep_then_seeds`, `powered_summary`,
   `holm_family`, `negative_control`, `config_fingerprint`), with the **file separation and
   refusal** of `seq/recall_gate.py::_resolve_results_path`: smoke →
   `results/surprise_ablation_smoke.json`, campaign → `results/surprise_ablation_powered.json`,
   a smoke run pointed at the campaign ledger refused. Raw records archived before any
   verdict (`recall_gate.archive_run` mechanism; [`docs/RETENTION.md`](../RETENTION.md)).
   Resume keyed on `(seed, config-fingerprint)`; mixed-configuration aggregation raises.

**Could any arm be implemented two ways?** A4 could; it is pinned above: the gate *replaces*
the learned gate (not a multiplier on top of it); the signal is per-head (not shared);
ε_t uses the *current* pre-write state with `α_t = 1`; the EMA is causal, reset per sequence
in `forward()` and state-carried in `step()`; the σ argument is `a·(s̃_t − 1)` — not
`a·(log s̃_t)`, not `a·s̃_t − 1`; `β_cap` multiplies *after* σ; the normalization guard is
`max(m_t, 1e−6)`. A1–A3 are existing, already-tested knobs with a single implementation each.

---

## 3. Tasks & protocol

### Tasks (frozen)
| task | constructor | target difficulty (frozen eval) |
|---|---|---|
| MQAR-D64 | `MixedMQAR(vocab=256, max_pairs=64, num_queries=128, gap=0, min_pairs=1)` | eval fixed at `max_pairs=64` via `MixedMQAR.eval_sample` |
| MQAR-D128 | `MixedMQAR(vocab=512, max_pairs=128, num_queries=128, gap=0, min_pairs=1)` | eval fixed at `max_pairs=128` |
| SELECTIVE-COPY | `SelectiveCopy(vocab=32, mem_len=64, n_data=16, fixed=False)` | i.i.d. draw, held-out by construction |

`MixedMQAR` trains on the difficulty spectrum `d ~ U[min_pairs, max_pairs]` and evaluates
fixed at the target difficulty — the repo's fair protocol (dense supervision, identical for
every arm). `fixed=False` selective-copy requires content-selective gating (the one task
where an input-independent gate has a known structural reason to struggle).

### Training protocol (the repo's fair protocol, referenced — not re-invented)
- **Entry point:** `seq.common.build_and_train` — seeds **before** model construction, so
  per-seed init is pinned (the init-before-`set_seed` defect is fixed at this entry point).
- **Frozen eval set:** `seq.common.TrainConfig.eval_seed = 12345`, `eval_batches = 32`,
  `batch_size = 64` — built ONCE per cell under the dedicated eval RNG
  (`_frozen_eval_batches`), so the eval batches are **bit-identical across all four arms** and
  all seeds. No best-of-noisy-curve inflation is possible.
- **Per-model LR / warmup rules:** warmup = `max(warmup, warmup_frac·steps)` with the GENWARM
  recipe from `seq/gpu_harness.py` / `TrainConfig`: `warmup=2000`, `warmup_frac=0.0`,
  `min_lr_frac=0.1`, cosine decay, AdamW `betas=(0.9, 0.95)`, `weight_decay=0.01`,
  `grad_clip=1.0`. Per-model LR: stage-1 sweep per (arm × task) over
  `seq.lrsweep.DEFAULT_GRID = (5e-4, 1e-3, 1.5e-3, 2e-3, 3e-3)` at 1 seed, recording all
  rejected LRs (the LR-fairness audit; no architecture is denied an LR another gets);
  stage-2 runs all claim seeds at the chosen LR. The sweep's seed-0 run is discarded;
  stage-2 retrains every seed fresh at the chosen LR (the existing `sweep_then_seeds` /
  `_train_arm` behavior).
- **Plateau early-stop with engagement floor** (`TrainConfig`): `eval_every=2000`,
  `plateau_delta=0.003`, `plateau_floor=0.5`, `min_steps=4000`,
  `early_stop_patience=5` — a model stops only after best-acc stalls ≥5 evals **and** it is
  clearly learning (best ≥ 0.5); anything below the floor trains to the full cap, so a
  pre-phase-transition model is never cut off and misread as a failure. Identical rule for
  every arm.
- **Step budget (identical data budget):** `cap = 40000` steps for every arm × task × seed.
  Rationale, stated in advance: no Prizma-family arm at this scale has ever ignited after
  ~32k steps in any prior run (ignition 16–20k in B6/P2eff; solvers plateau by ~24–32k), so
  40k is ≥ 2× the latest observed ignition; the TF arms (which ignite later) are not part of
  this ablation. The cap deviation from the 80k precedent is a budget decision, frozen now,
  and applies identically to all arms.
- **Seeds (exact n, no stopping rule deviation):** claim seeds = **(0, 1, 2, 3, 4)** —
  **n = 5 per arm per task**. No seed is added, dropped, or substituted; a crashed seed is
  re-run as the same seed; a seed is never excluded post-hoc. LR-sweep seed = 0 (repo
  convention); the exploratory gain seed = 900 (never a claim seed).
- **Data note (honest):** the frozen eval is bit-identical across arms (dedicated eval RNG).
  The training stream is seeded identically per seed, but different arms consume different
  RNG amounts during model construction, so training batches are distribution-identical, not
  bit-identical, across arms. This is the existing repo behavior for all multi-arm campaigns;
  synthetic tasks draw i.i.d., so the comparison is fair.
- **Integrity canary:** the `negative_control` from `seq.gpu_harness` (two byte-identical
  `input` arms, MQAR-D64, claim seeds 0–1) must show **no** significant difference
  (one-sided Welch p ≥ 0.05). A canary FAIL invalidates the campaign's numbers: the verdict
  is INCONCLUSIVE and no claim is made until the harness bug is found and the campaign is
  re-run (recorded as a dated addendum; the original numbers are never cited).
- **Resume/integrity:** `(seed, config-fingerprint)` resume keys
  (`seq.gpu_harness.config_fingerprint`); mixed-configuration aggregation raises;
  smoke and campaign ledgers are separate files and a smoke run aimed at the campaign file is
  refused — the operational lessons of `results/campaign_2026-06-08/CONTAMINATION.md`,
  applied by construction.

### Coverage note and the first command a GPU session runs
`tests/test_surprise.py` covers the Lever-A surprise kernel machinery (off-identity, R3
repeated-key exactness, chunk independence, O(1) `step()` guard);
`tests/test_stats.py` covers the statistics layer. **There is no existing test coverage for
the `precision_gate` knob values themselves** — the new A4 mode must ship with the tests
spec'd in §2 A4 before any campaign run. GPU-session pre-flight (note only; nothing is
written here):

```
pytest -q tests/test_surprise.py tests/test_stats.py     # CPU, must be green first
python gpu_surprise_ablation.py --smoke                  # plumbing-only; writes the SMOKE ledger
```

Only after both pass does any A100 claim run start.

---

## 4. Statistics — frozen

### Endpoints
Per arm × task, over the 5 claim seeds:
- **Primary metric (feeds every test):** per-seed `best_acc` on the frozen eval set under the
  plateau early-stop protocol (the repo's standard recorded "best"; "plateau-final").
- Reported alongside: mean ± std of `best_acc`; **solve-rate** at the fixed step budget
  (`best_acc ≥ 0.9`, the repo's `SOLVE_THRESH`); and final-acc at the 40k cap + the
  `steps_to_plateau` audit field (both recorded, neither tested).
- A task on which every arm sits at exact ceiling cannot discriminate and will be reported as
  such (identical ceiling arrays yield Δ = 0 and cannot reject).

### Primary comparisons and correction (exact)
For each task `t ∈ {MQAR-D64, MQAR-D128, SELECTIVE-COPY}`:
- `P1_t = superiority_test(acc[surprise_norm, t], acc[uniform, t])` — one-sided Welch t,
  H1: mean(surprise_norm) > mean(uniform) (`seq.stats.superiority_test`).
- `P2_t = superiority_test(acc[surprise_norm, t], acc[random, t])` — same direction vs random.

**Family:** the 6 p-values {P1_t, P2_t : t} are Holm-Bonferroni-corrected at α = 0.05 with
`seq.stats.holm_correction(pvals, alpha=0.05)` (step-down; the reported decision uses each
comparison's Holm-adjusted p). There are **no other confirmatory tests**; the vs-`input`
comparison is descriptive only (§4 honest limits).

### Decision rule (frozen — this exact rule, no reinterpretation)
Let "win on task t vs control c" ≡ [Holm-adjusted p of the corresponding primary < 0.05]
**and** [point estimate mean(surprise_norm) > mean(control) on t].

**The mechanism SURVIVES iff:**
1. surprise_norm wins vs **uniform** AND vs **random** on **≥ 2 of the 3 tasks**, and
2. on the remaining third task (or, if it wins on all three, on every task), the reverse
   guard `superiority_test(acc[control, t], acc[surprise_norm, t])` is **not** significant at
   raw one-sided α = 0.05 for **both** controls (i.e. surprise_norm is never significantly
   *worse* than either control anywhere).

**Otherwise the mechanism is RETIRED** from README novelty claims (§6 patch applied; INDEX row
→ RETIRED; owner decision 4). This includes: winning vs only one control, winning on ≤1 task,
being significantly worse anywhere, an unrunnable or canary-failed campaign (no claim either
way), and a purely null result. **Failure to reject is not equivalence**: no TOST-equivalence
is claimed or computed for the decision; a tie earns the mechanism nothing.

### MDE table (n = 5 per arm, power 0.80, computed with `seq.stats.t_isf`)
Planning formula for two arms of equal n: `MDE = (t_{α*,ν} + t_{β,ν}) · σ · √(2/n)`, with
ν = 2(n−1) = 8 (equal-variance case; Welch df varies with the variance ratio — the
equal-variance df is the planning assumption). Computed values: `t_isf(0.05, 8) = 1.8595`,
`t_isf(0.20, 8) = 0.8889`, `t_isf(0.05/6, 8) = 3.0158` (Holm-worst-case: the smallest of the
6 p-values is tested at α/6 = 0.008333), `√(2/5) = 0.6325`.

| assumed per-seed σ (acc units) | MDE at raw α=0.05 | MDE at Holm-worst α/6 |
|---|---|---|
| 0.05 (within-regime) | 0.087 | 0.124 |
| 0.10 (mixed) | 0.174 | 0.247 |
| 0.20 (near bimodal) | 0.348 | 0.494 |

**σ assumptions, stated from repo evidence:** the only direct measurement of these arms' seed
variance is the n=2 smoke (sd ≈ 0.044–0.046, at a tiny 400-step scale — an underestimate);
converged solvers are tight (sd ≈ 0.003–0.02, e.g. the B4 n=7 seeds); cells straddling the
MQAR phase transition are bimodal (sd up to ~0.3+, e.g. the P1 TF 0.02-vs-0.96 cells).
The planning band {0.05, 0.10, 0.20} spans this honestly. **Honest reading of the table:** at
n=5 this design detects only large effects — roughly ≥ 9 accuracy points (raw α) to ≥ 12
points (Holm-worst) under optimistic σ; under bimodal σ it detects only near-total
separation. That is the power the owner's "n ≥ 5" mandate buys, and it is declared before the
run. The retire-if-negative rule remains valid at this power because retirement only removes
a claim.

### Honest limits — what will NOT be claimed even on a SURVIVE
- The rule consults only the constant and random controls (owner decision 4, verbatim).
  If surprise_norm survives but does **not** also beat the learned `input` incumbent
  (descriptive comparison), the claim is scoped to *"beats input-independent gates"* and any
  wording implying superiority over the shipped learned gate is prohibited; an upgrade to
  that stronger claim requires a new pre-registration.
- No claim beyond: this scale (d64L2H2), this base (`quad2_lowrank`), these three tasks,
  this gate formula (the frozen EMA-normalized form), n=5. No "brain-like" or
  neuromodulation claim (that is a mapping argument, not something this ablation tests).
  No claim that this formula is optimal among surprise-derived gates.
- Survival earns exactly the scoped sentence in §6 (SURVIVE branch), nothing more.

---

## 5. Compute budget (~10 A100-hours class; runs when GPU budget exists)

All runs at d64L2H2 (~130K params), cap 40k steps, batch 64, plateau early-stop. Per-run
wall-clock basis (estimate, to be re-measured at the exploratory session and recorded in the
ledger): extrapolated from the recorded d128L4H4 rate (~50 min / 80k steps), d64L2H2 is
roughly 2–3× cheaper per step ⇒ ~8–11 min at the full 40k cap, ~3–5 min typical with plateau
early-stop (most solvers stop ≤ 24k; non-solvers run to cap).

| block | runs | est. hours |
|---|---|---|
| A1 `input` — stage-1 sweep (3 tasks × 5 LRs) + stage-2 (3 × 5 seeds) | 30 | ~1.8 h |
| A2 `uniform` — same | 30 | ~1.8 h |
| A3 `random` — same | 30 | ~1.8 h |
| A4 `surprise_norm` — sweep + seeds (30) + exploratory gain selection (4) | 34 | ~4–10 h (exact sequential scan; disclosed 2–5× tax vs the WY/UT fast path — the dominant budget unknown, measured at the exploratory session) |
| Integrity canary (2 identical arms, 2 seeds, 1 task) | 2 | ~0.2 h |
| **Total** | **126** | **≈ 10–15 A100-h central (honest band 8–20 h)** |

Rules: the exploratory session (gain selection + one A4 timing run) happens **before** claim
seeds start, and its measured per-run cost is written into the ledger. If the measured total
implies > 20 A100-h, the owner is informed **before** further claim runs and the overage is
recorded — the protocol itself is **not edited** (no knob, seed, cap, or task changes under
budget pressure; the alternative to executing verbatim is not executing).

---

## 6. Pre-committed negative-result handling (drafted BEFORE the run)

### 6.1 README patch — ready to apply on RETIREMENT
Apply to `README.md`, inside section **"Corrections and quarantined results"**, appended
after the paragraph ending "…claim-grade results are governed by the pre-registration
registry":

```markdown
**The surprise-gating mechanism is RETIRED from novelty claims (PR-2026-09-03-01, RETIRED).**
The one mechanism Prizma-Seq held as novel-but-untested — a surprise-derived write gate
("Precision/surprise signal used causally for gating") — was tested at power under
pre-registration
[`docs/preregistry/2026-09-03-surprise-gating-powered-ablation.md`](docs/preregistry/2026-09-03-surprise-gating-powered-ablation.md)
(n=5 seeds/arm/task; EMA-normalized surprise gate vs learned-input, constant, and random gate
controls; one-sided Welch t, Holm-corrected; MQAR D=64, MQAR D=128, selective-copy). It
FAILED its pre-registered survival bar: the surprise-derived gate did not beat both the
constant and the random controls per the frozen decision rule. Verdict artifact:
`results/surprise_ablation_powered.json`. Per owner decision 4
(`committee/brainstorm_2026-09-03/00_COMMISSION_SYNTHESIS.md` §6.4), the mechanism is
retired from all novelty claims: no README, report, or abstract may describe Prizma-Seq's
write gate as surprise-gated, surprise-modulated, or precision-signal-driven. The shipped
write gate is a learned input gate `β_t = σ(W_β x_t)` — a Gated-DeltaNet-family standard
component. Relocation of the surprise signal to the decay gate α may be proposed only as a
NEW pre-registration, never as a retroactive save.
```

Companion hunk (same commit), `docs/PRIZMA_SEQ_REPORT.md`, "Borrowed vs new ledger" — the
row `| **Precision/surprise signal used causally for gating** | Prizma | **new — but UNTESTED
at power, and the one result there is points AGAINST it** … |` is replaced with:

```markdown
| **Precision/surprise signal used causally for gating** | Prizma | **RETIRED (PR-2026-09-03-01, RETIRED)** — failed its pre-registered powered bar (n=5 seeds/arm/task, Holm-corrected Welch): the surprise-derived gate did not beat both the constant and random controls; retired from novelty claims per owner decision 4. The shipped write gate is a learned input gate σ(W_β x_t). Verdict: `results/surprise_ablation_powered.json` |
```

If the outcome is SURVIVE instead, no README "corrections" hunk is applied; the claim text is
the §4-scoped sentence drafted into the INDEX CLAIMED row and nowhere stronger.

### 6.2 INDEX addendum text (maintainer merges; statuses per POLICY lifecycle)
On RETIREMENT, the `PR-2026-09-03-01` row becomes:
`| PR-2026-09-03-01 | 2026-09-03 | CLAIM | RETIRED | [2026-09-03-surprise-gating-powered-ablation.md](2026-09-03-surprise-gating-powered-ablation.md) | Surprise-norm gating FAILED to beat BOTH the constant and random controls at power (n=5, Holm-corrected Welch, 3 tasks); mechanism retired from README novelty claims per owner decision 4. Verdict: results/surprise_ablation_powered.json |`

On SURVIVAL, the row becomes:
`| PR-2026-09-03-01 | 2026-09-03 | CLAIM | CLAIMED | [2026-09-03-surprise-gating-powered-ablation.md](2026-09-03-surprise-gating-powered-ablation.md) | Surprise-norm gate beat BOTH constant and random controls (Holm-corrected Welch, n=5) on ≥2 of 3 tasks and never significantly worse on the third; claim scoped to §4 of the pre-registration. Verdict: results/surprise_ablation_powered.json |`

On a campaign that never executes, the row becomes ABANDONED with the reason.

### 6.3 Dependent proposals — re-anchoring rule
On RETIREMENT, every proposal that builds on surprise-derived **write** gating re-anchors on
this result: commission proposals 04-P4 (uncertainty-derived gating), 05-P2 (precision
audit), 06-P3a, 03-P2/P3 (thalamic gating of writes), and any thread-fusion component
assuming a surprise-driven `β_t` may **not** cite surprise-gating as a supported mechanism.
Two sanctioned paths forward: (a) re-derive the proposal without the retired mechanism; or
(b) file a NEW LANE-CLAIM pre-registration — including any proposal to relocate the surprise
signal to the decay gate `α` (owner decision 4) — which must cite PR-2026-09-03-01's negative
outcome as prior evidence and may not reuse this document's arms or bars.
On SURVIVAL, dependent proposals may cite the result only with the §4 scope wording.

---

## 7. Self-audit

- **Placeholder scan.** This document contains no TODO/TBD/XXX/FIXME and no fill-in slots:
  the §6 patch texts are fully literal (they are written for the retirement outcome and name
  the verdict artifact rather than quoting numbers that do not exist yet). All numeric
  constants (arms, seeds, cap, λ, α thresholds, MDE values) are pinned. The only quantity
  deliberately determined by data is the gain `a`, under the §2 NO-TUNING rule.
- **Ambiguity check (could any arm be implemented two ways?).** A1–A3 are single existing
  knobs (`precision_gate` ∈ {input, uniform, random}) with committed tests-backed behavior in
  `seq/prizma_seq.py::_encode`; no free parameters. A4 is pinned point-by-point in §2:
  gate replaces (not modulates); per-head signal; ε_t from the pre-write state with
  `α_t = 1`; causal per-sequence EMA (`m_1 = s_1`, λ = 0.02, carried in the streaming state);
  σ argument `a·(s̃_t − 1)` exactly; normalization guard `max(m_t, 1e−6)`; `β_cap` applied
  after σ; erase gate = write gate. Every shared config field is enumerated in §2. The
  statistics (endpoint, test direction, family, α, Holm utility, decision rule, guard level)
  are single-valued. The exploratory procedure (seed 900, task, LR, candidate set, tie-break)
  is single-valued.
- **"What would change our mind" — addendum policy.** Per the POLICY immutability rule, this
  document is never edited once results exist; changes are appended as
  `## ADDENDUM YYYY-MM-DD: <reason>` with the INDEX row marked AMENDED. Legitimate addendum
  triggers, declared now: a discovered infeasibility of a frozen task/config; a bug in the
  new A4 implementation found (and fixable) **before any claim seed runs**; an infra failure
  or canary FAIL. Illegitimate and therefore prohibited: changing `a` after claim seeds,
  adding/removing seeds or tasks, moving α, weakening the decision rule, or reinterpreting a
  RETIRE as a "partial save" — relocation to α requires a new pre-registration, period.

---

**FROZEN — this protocol is frozen as of 2026-09-03, owner decision recorded** (committee
synthesis §6.4; registry id PR-2026-09-03-01, lane CLAIM, status REGISTERED-FROZEN). It
executes verbatim when GPU budget exists, or not at all.

---

## Addendum 2026-09-08 (pre-run, maintainer) — canary cost disclosure (review M-9)

The §5 table budgets the integrity canary as "2 runs ~0.2 h", but the named implementation
(`seq/gpu_harness.negative_control`, which §5 itself names) calls `sweep_then_seeds` for BOTH
arms (negctrl.A and negctrl.B): 2 x (5-cell LR sweep + 2 seeds) = 14 cells, not 2. The
realistic canary cost is therefore ~+1 h on top of the 10-15 h §5 estimate. This is a cost
disclosure only — no protocol change (arms, seeds, statistics, and bars are untouched). The
PR-01 campaign ledger will note the actual canary wall time at run time. (Review:
committee/review_2026-09-08/PRE_GPU_REVIEW.md, M-9.)
