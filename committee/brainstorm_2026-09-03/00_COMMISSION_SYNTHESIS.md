# Prizma Commission — Brainstorm Synthesis (2026-09-03)

14 scientific-role agents, each loaded with the brainstorming skill, independently read the
repo and produced ranked proposals. This file synthesizes their findings, convergences, and
disagreements. Full reports: `01_*.md` … `14_*.md` in this directory. No code was written
(brainstorming HARD-GATE honored by all 14).

## 1. Commission

| # | Role | Report |
|---|------|--------|
| 01 | Computational neuroscience — canonical cortical microcircuits, hierarchical PC | 358 ln |
| 02 | Hippocampal memory & systems consolidation (CLS) | 428 ln |
| 03 | Biological attention & saliency (thalamic gating, WM buffers) | 408 ln |
| 04 | Neuromodulation & synaptic plasticity (ACh vigilance, DA surprise, eligibility) | 353 ln |
| 05 | Predictive processing / free-energy principle, active inference | 400 ln |
| 06 | Sparse coding, MoE, FLOP efficiency | 463 ln |
| 07 | Sequence-model competitive landscape (GDN/Mamba-2/RWKV-7/TTT/Titans) | 496 ln |
| 08 | Scaling laws & training/hardware efficiency | 466 ln |
| 09 | Oscillations, binding, multi-timescale phase dynamics | 559 ln |
| 10 | Continual & lifelong learning at LM scale | 555 ln |
| 11 | Mathematical theory — capacity, expressivity, dynamics | 577 ln |
| 12 | Neuromorphic & edge deployment | 489 ln |
| 13 | Red team / adversarial referee | 416 ln |
| 14 | Benchmark & evaluation methodology | 433 ln |

## 2. Where the commission CONVERGES (independent, repeated findings)

### C1. The surprise-gating ablation is the single most-repeated must-do (reports 01, 04, 05, 06, 08, 13, 14)
Prizma's one claimed-novel mechanism (surprise-gated writes) has only an n=2 ablation whose
point estimates run **against** the mechanism (constant gate beat the surprise signal). At
least 6 of 14 reports flag this as the ceiling on every brain-likeness claim. ~10 A100-h
powers it (surprise-norm vs constant vs random vs none, n≥5, pre-committed negative-result
protocol). Several proposals (05-P2 precision audit, 04-P4 NE gating, 06-P3a) are literally
this experiment with different framings.

### C2. "Replace attention" is over; the narrative must pivot (07, 13, 05, 10)
Gated DeltaNet is now production architecture (Qwen3-Next 80B ships a GDN-dominant hybrid;
Kimi Linear, RWKV-7 at scale). Diagnostics passes, O(1) memory, long-context decode are
**commodity**. The vacant thrones the commission identifies: (a) the free-energy/theory
unification of the whole delta-rule family, (b) lifecycle/always-on learning, (c) the phase
axis, (d) local learning, (e) the deployed-hybrid slot Prizma is already shaped to occupy.
Red team's survivable claim until bars pass: *"the free-energy-derived, locally-learnable,
lifecycle member of the family that won."*

### C3. The 5× training-step gap is THE quantitative blocker — and it is an engineering, not
scientific, problem (08, 06, 13)
It **cancels** Prizma's one dynamics win (ignites in ~⅓ the steps of the TF → wall-clock
parity or worse). 08 found the draft Triton kernel's backward is gated out of the training
path (`_should_use_triton` grad-gate) — so nothing currently attacks the gap; it needs one
decisive A100 verification session (bar: ≤1.5× TF step time). 06 independently derived that
top-k sparse delta writes (~6–12% active fraction) analytically reach **~1.0× TF FLOPs** —
cortically honest (~2% active in brain) and the only lever that attacks FLOPs, step-time,
and brain-likeness simultaneously.

### C4. Thread fusion — the "unified cortical column" — is the commission's biggest
architecture idea (01 P5 impact 10, 02, 06 P2, 10 P5, 13 convergence note)
Three reports independently proposed variants of the same flagship: Prizma-Seq backbone +
Prizma-CL vigilance routing/experts = a lifelong LM (hippocampus = carried state S; cortex =
experts/weights; consolidation = freeze/replay; vigilance = ACh) that keeps learning on-stream
with no replay, no boundaries — the one capability transformers structurally lack.
Smallest falsifiable version: continual char-LM over drifting corpora, pre-registered
retention bars. Caveats from 02/10: interleaved-stream collapse (ACC ~0.58) must be fixed
first (dynamic vigilance 04-P1, replay-before-freeze 02-P4); expert memory currently grows
unboundedly (10-P3 expert economy).

### C5. The theory has a hole the repo should patch honestly (11 — "most honest work of the
commission" per red team)
The Gershgorin bound (N<14 for quad2) **cannot explain the repo's own D=128 PASS** — it only
explains the linear baseline's failure. 11 derives a mean/fluctuation split of the crosstalk
matrix → sharp capacity law N* = min(1+ε²/σ₂², d_φ); fitted once at D=64 it would PREDICT the
entire deferred D-frontier before running it — predictive power as headline. Three honesty
flags (A–C) are drafted for the corrections section. Also: Householder DeltaProduct
(β→2 = reflections → depth-1 non-commutative state tracking where matched TFs need CoT).

### C6. Repair-first before any new claim (14, 07, 13)
≈63 A100-h of repairs close the evidence debt: clean n=10 recall-gate re-run (~28h),
init-before-set_seed fix re-runs (~18h), B4 closure both corpora n≥5 + artifact-retention
policy (~8h), first-ever GLA/Mamba-2 landscape run (~60h), powered surprise ablation (~10h),
plus zero-GPU items: two-lane pre-registration registry (exploratory may motivate, never
become claims — the B4 lesson), MDE/TOST tables, file-separation of smoke vs campaign
results (the operational root cause of the contamination).

### C7. Phase is a vacant axis (09)
No delta-rule family member carries phase; RoPE is a *frozen* phase code. A rotating-frame
decay (forgetting as precession, T1-feasible inside the chunk kernel) or chunk-local phase
precession (T0, zero params) targets the repo's weakest leg (length-extrapolation ~0.40
absolute at 8×). Bar: ≥0.55@8×, spectral-null test pre-registered.

### C8. Deployment wedge exists only at long-context + always-on edge (12, steelmanned)
"Transformers can't run on edge" is FALSE (spiking transformers exist). What survives: cache
(O(T), append-only) vs state (O(1), in-place), and always-on learning. Strongest mapping:
consolidation freeze doubles as RRAM wear-leveling (S volatile / W stable) — one mechanism,
two consequences; "delta rule is analog-robust" quantization bar (delta self-corrects,
additive accumulates) is cheap and could yield an unclaimed deployment-grade differentiator.
Energy claims only via a citation-gated ledger tool (no second simulated-CUDA incident).

### C9. Brain-likeness can be *measured*, cheaply, later (14, 01)
Brain-score-style encoding-model parity + mechanistic β_t↔N400 coupling tests exist at
~3–5 GPU-h (inference-only) but are only honest at the 50M word-level rung — registered as
dormant until that rung exists.

## 3. The commission's consolidated program (by tier)

**Tier 0 — REPAIR (≈63 A100-h + zero-GPU items; before ANY new claim):**
C1 surprise ablation (10h) · clean recall gate (28h) · init-fix re-runs (18h) · B4 closure
(8h) · GLA/Mamba-2 landscape first run (~60h) · pre-registration registry · smoke/campaign
file separation · artifact-retention policy.

**Tier 1 — THE DECISIVE BETS (the ones that could make Prizma "inevitable" at something):**
1. Kernel decision session (08-P1): ≤1.5× TF step-time bar; converts ⅓-steps ignition into a
   wall-clock win.
2. Top-k sparse delta state (06-P1): FLOP parity/sub-parity at f=6–12%; pre-register the
   write-sparsity-at-matched-accuracy headline.
3. Crosstalk capacity law + pre-registered D-frontier (11-T1): predict before you run.
4. Minimal Prizma-LM (C4): continual char-LM, no replay/boundaries/buffer, retention bars —
   after dynamic-vigilance fix.
5. Perpetual-inference drift bar (05-P4): streaming distribution-drift vs memory-matched
   sliding-window TF control — the structural moat claim.

**Tier 2 — STRUCTURE/IDENTITY (brain-shaped mechanisms):** deep FF/FB error hierarchy
(01-P1) · thalamic gating hub + divisive-normalized reads (03-P2/P3) · rotating-frame/phase
axis (09-P1/P6) · Benna–Fusi cascade for S (04-P3) · eligibility-trace three-factor sequence
learning (04-P2, the backprop-free sequence-trainer claim) · feature-map shootout + η-frontier
(07-P2, 11-T3) · vigilance-MoE FFN (06-P2).

**Tier 3 — DEPLOYMENT/ECONOMY:** consolidation=wear-leveling mapping · analog-robustness bar ·
energy ledger · expert economy (prune/merge/bounded) · regime map κ∈[0,1] with buffer-matched
DER++ (10-P1) · neuromorphic notes under docs/neuromorphic/ labeled "design + arithmetic".

## 4. Red team's bars that must hold (kill-criteria with pre-committed pivots)

BAR-0 seed-pin + result-file integrity · BAR-1 TOST parity vs GLA/Mamba-2 · BAR-2 B4 both
corpora ≥5 seeds · BAR-3 train step ≤1.5× TF · BAR-4 one real 50M run (+0.10 BPC) · BAR-5
surprise signal beats constant/random controls at power · BAR-6 interleaved CL ACC ≥0.70.
Until they run, the word "inevitable" is banned from the repo.

Red team's charges against the commission itself: 6 of 12 constructive reports build on the
untested surprise signal; 3 independently re-invented the same consolidation flagship (an
honest convergence signal, but the flagship is unproven); MQAR D=128 is home-turf — nobody
proposed the field's 1M-token recall arena; report 11 (fixing a theory that contradicts the
README) is the commission's most valuable single contribution.

## 5. Consolidated QUESTIONS FOR THE OWNER

Strategic:
1. Identity priority: (a) credible transformer-rival sequence model, (b) lifelong LM
   (thread fusion), (c) unified cortical column (both), (d) neuromorphic/edge specialist?
   (This decides Tier-1 ordering.)
2. Realistic monthly GPU budget (A100/L4 Colab hours)? The repair tier alone is ≈63h +
   60h landscape; the full program ≈150–250h.
3. The word "rival": banned until BAR-0..6 pass (red team's position), or kept as the
   internal north star while public wording stays scoped?
4. If the powered surprise ablation confirms the negative smoke signal — retire the
   mechanism (pre-commit now), or re-locate it (e.g. β→α)?
5. B4 corpus policy: re-run tiny-shakespeare post-mortem, or substitute enwik8?
6. Is Triton/CUDA kernel engineering in scope for you personally, or must the plan assume
   Colab-session-only engineering?
7. Backprop-free *sequence* training (eligibility traces, 04-P2): headline goal or
   optional module? It is the one claim transformers structurally cannot match — but it is
   feasibility 5/10.
8. Brain-likeness: framing-only for now, or fund the dormant brain-score/β↔N400 bars when
   the 50M rung exists?
9. May the repo publish capability *trades* (e.g. remapping losses in exchange for zero
   forgetting)?
10. Compute split between MPU (mechanism proofs) and the 150M-hybrid ladder (adoption)?

Report-specific questions are in each report's final section.

## 6. OWNER DECISIONS (2026-09-03, via interactive question round — BINDING)

1. **Identity = Lifelong LM (thread fusion).** Prizma-Seq backbone + Prizma-CL vigilance
   routing/experts. Tier-1 ordering re-centered on C4; "credible transformer-rival seq" and
   "unified column" become supporting claims, not the flag.
2. **GPU budget = ZERO this month.** Only zero-GPU work is authorized (see §7).
3. **Discourse = red team's rule.** The words "rival / inevitable" are banned in the repo
   until BAR-0..6 pass; internal north star only. Public wording: the free-energy-derived,
   locally-learnable, lifecycle member of the delta-rule family that won.
4. **Surprise-gating: pre-registered retirement.** The n≥5 powered ablation is pre-registered
   NOW with the outcome rule: if surprise-norm fails to beat constant AND random controls,
   the mechanism is retired from the README's novelty claims (relocation to α may be proposed
   as a NEW pre-registration, not a retroactive save).

## 7. This month's ZERO-GPU program (reordered per owner decisions)

1. **Pre-registration registry** (two-lane: exploratory vs claim-grade) + file-separation of
   smoke vs campaign results (the contamination root-cause insurance) + artifact-retention
   policy (failures are never deleted — the B4 lesson). Zero-GPU, repo culture infrastructure.
2. **Surprise ablation pre-registration** written and committed (protocol frozen; runs when
   GPU returns): arms {surprise-norm, constant, random, none}, n≥5, Welch + MDE table,
   pre-committed retirement rule (owner decision 4).
3. **Theory corrections** (report 11, Flags A–C): Gershgorin N<14 cannot explain the repo's
   own D=128 PASS; draft the mean/fluctuation crosstalk law (N* = min(1+ε²/σ₂², d_φ)) and
   extend `feat_map_probe.py` with σ₂/μ/η instrumentation — all CPU.
4. **Citation-bar CL battery** (report 10-P4): MAS/SI/online-EWC regularizers were never run;
   single-pass protocol variant; honest 2024–26 exemplar-free table (RanPAC/SHE/ACIL).
   Days of CPU work — the decidability work for the fusion flagship's citation claims.
5. **Expert economy analysis** (report 10-P3, CPU): quantify unbounded weight growth from
   existing `route_log` data; prune/merge/bounded-M_max design spec.
6. **Analog-robustness pre-registration** (report 12-H2): quantize S to 4–6 bits + write
   noise; delta vs additive write modes on the committed MQAR harness — CPU-runnable.
7. **Prizma-LM fusion design spec** (brainstorming architectural-path output, GPU-free):
   the minimal falsifiable lifelong-LM version — continual char-LM, no replay/boundaries/
   buffer, retention bars, dynamic-vigilance (04-P1) + replay-before-freeze (02-P4) as
   prerequisites — ready to execute the day GPU budget exists.
8. **BAR-0 infrastructure**: seed-pin regression tests, resume-cache fingerprint tests,
   paired-seed endpoints + MDE/TOST tables in `seq/stats.py` (CPU).

When GPU budget returns, the order is: Tier-0 repairs (§ C6) → surprise ablation (frozen
protocol) → kernel decision session (08-P1) → top-k sparse delta (06-P1) → minimal Prizma-LM.
