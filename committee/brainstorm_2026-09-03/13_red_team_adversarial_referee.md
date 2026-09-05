# 13 — Red Team / Adversarial Referee (hostile reviewer #2, falsification officer)

**Commission:** Prizma → "inevitable Transformer rival + maximally brain-like"
**Author role:** Adversarial Referee. My job is to find every reason the plan FAILS, then state
what would have to be true for it to survive.
**Classification (per brainstorming SKILL.md):** Architectural-class research-direction brainstorm.
**HARD-GATE honored: no code, no implementation — this report is the artifact.** Owner questions
are collected in **QUESTIONS FOR THE OWNER** at the end.
**Sources read in full:** `README.md` (incl. corrections/quarantine), `docs/PRIZMA_SEQ_REPORT.md`,
`docs/Prizma.md` (incl. §8), `docs/HANDOFF.md`, `committee/round1_synthesis.md`. Skimmed: all 12
sibling reports `01`–`12` in this folder. External 2025–26 facts are marked *(knowledge)*.

---

## 0. Verdict in one paragraph

The goal as stated — "make Prizma an **inevitable** rival to the Transformer" — **fails today, and
the repo's own documents prove it.** Every load-bearing claim is either small-scale-descriptive
(MQAR D=128, n=2–3, init not seed-pinned), a PARTIAL with a failed corpus (B4 char-LM), a
quarantined non-result (the n=10 recall gate), never-run apparatus (GLA/Mamba-2 landscape,
50M/100M models, MMLU/GSM8K harness), or — for the one mechanism the brand depends on (surprise
gating) — an ablation whose only point estimates run *against* the mechanism. Meanwhile the
external field shipped Gated DeltaNet inside Qwen3-Next-80B *(knowledge)*, making the entry-level
claim a commodity. "Inevitable" is not purchasable at 1.4M trained params by a single author.
**However:** the falsification attack does not kill the project — it kills the *word* "inevitable"
and replaces it with a set of pre-registered bars under which a narrower, still-valuable claim
("the locally-learnable, free-energy-derived member of the delta-rule family that won, with a
measured lifecycle axis") survives. Part B states those bars and what dies if each fails.

---

# PART A — FALSIFICATION ATTACK

## A.1 Objection ledger (all objections; severity 1–10; cheapest defuser)

| # | Objection (one line) | Sev | Cheapest defusing experiment / reframe |
|---|---|---|---|
| T1 | Linear attention is a commodity; GDN ships in Qwen3-Next-80B, Kimi Linear, Jamba, Samba, Falcon-H1 *(knowledge)* — Prizma's §4 PASSes are table stakes | **9** | Reposition: "hybrid-slot upgrade + unique axes" (07's wedge); run the built GLA/Mamba-2 arms to prove ≥ parity with the actual family |
| T2 | The ~5× training gap **nullifies** the "ignites in ⅓ the steps" dynamics win (80–100 TF-step-equivalents vs 44–80k) | **9** | One A100 session: unlock the training-path fused/Triton kernel (`delta_triton.py` gate excludes training) → bar ≤1.5×/step |
| T3 | O(1) state is an information bottleneck: recall-at-1M-token tasks (RULER-class) structurally favor the growing KV cache; capacity is `H·d_h·d_φ` floats, full stop | **8** | Pre-register the recall ceiling; publish the D-frontier; claim the serving/edge Pareto, not unbounded recall |
| T4 | B4 char-LM is a PARTIAL: the other corpus FAILED (−0.09) and its raw data was not retained; delivered n=2 (later n=7) vs pre-registered both-corpora ≥5 seeds | **8** | Re-run B4 properly (both corpora, ≥5 seeds, init pinned to seed) — the single most quotable negative, closeable for ~1 A100-day |
| T5 | The only powered multi-seed artifact (n=10 recall gate) is QUARANTINED; TOST parity was not met on any leg → the recall claim is a NON-result | **7** | Clean re-run (~28 A100-hours, bug already fixed, regression-tested) |
| T6 | GLA/Mamba-2 baselines: built, tested, **never run** — every existing win is vs the Transformer only (the family Prizma actually belongs to is unmeasured) | **8** | Run `seq/landscape.py`; it exists for exactly this |
| T7 | The novelty mechanism (surprise/precision gating) has one ablation: n=2 smoke, INCONCLUSIVE, point estimates run **against** it (constant 0.566 > random 0.527 > real 0.517 > none 0.406) | **8** | Powered ablation (≥5 seeds, LR-swept) — cheapest experiment in this entire report (CPU/small-GPU) |
| T8 | quad2 capacity theory does not explain its own headline: Gershgorin gives N<14, D=128 passes — README's "resolving the capacity block" is an overclaim by implication (report 11, Hole 1) | **7** | Random-matrix capacity law + pre-registered D-frontier prediction (report 11's P-series); CPU-cheap first |
| T9 | Latency loses below n≈16k; win only at ≥32k. Most real workloads live below that; and the "372× memory" rows are closed-form arithmetic, never measured at 50M/100M | **6** | Batch>1 serving benchmark (KV-cache dominates there); keep the 17.9MB measured curve as the honest headline |
| T10 | FLOP-matched TF arms are optimization-confounded (d128L9H4 1/3, d208L4H4 0/3) → **no per-FLOP claim**; Prizma is 2.14× TF forward-FLOPs/token as-coded | **7** | Per-width LR re-sweep at d208 (the decisive check the repo itself names); banded-window + fused kernel first |
| T11 | MQAR D=128 PASS is n=3 with **init not pinned by seed** (weights drawn before `set_seed`) → absolute solve-rates not reproducible; and TF is seed-fragile at this scale (P1 2/3 vs P2 3/3 same config) | **6** | Fix seed-order bug (one line, test-pinned), re-run the 3 headline cells |
| T12 | CL headline is quasi-tautological (author's own §8): FGT=0 is architecturally guaranteed once routing is perfect; the real claim is 100% task-identity inference in ONE narrow regime | **7** | Keep the honest reframe (it is already §8); extend regime: interleaved streams, natural data, K unknown (see T13) |
| T13 | CL regime is small: contiguous blocks only (interleaved ACC ~0.58), multi-epoch fits (epochs=10–15, block-online not single-pass), expert pool sized with K known a priori, freeze permanent (reawakening specified in §3 but never implemented), inference routing has no abstain, per-domain weights grow unboundedly | **7** | Interleaved-stream test with dynamic vigilance (report 04 P1); abstain output; a memory-growth table per domain |
| T14 | Vigilance routing lives or dies on input-distinguishability; the only degradation numbers are E2's Gaussian-noise sweep on one synthetic benchmark — no domain-overlap curve, no natural-data test | **6** | A single overlap-sweep experiment (interpolate covariances between domains; plot ACC/FGT vs overlap) |
| S1 | FEP branding risk: PC-as-LM-mechanism has prior art (PC transformers / Salvatori et al.; Millidge et al.; Ororbia's analog PC LMs) *(knowledge)*; the delta write IS one gradient step on free energy — but that is a derivation of a borrowed kernel, not an FEP architecture | **5** | Claim exactly the tested identity ("delta write = 1 GD step on F_t"), cite the PC-LM lineage, drop "FEP" as a banner word |
| S2 | Neuromorphic credibility: no hardware numbers at all; RTN is not white Gaussian; RRAM endurance ~1e6–1e9; the "noise=free sampler" line is an idealization the repo itself flags | **5** | Emulation campaign under labeled degradation (report 12 H3); never a simulated-hardware "measurement" again (the CUDA-rows precedent) |
| S3 | Reproduction-credibility pattern a hostile reviewer will cite: simulated-CUDA rows deleted; stale "94 tests" + green badge over red CI; failed corpus raw data not retained; cache bug misread as baseline bimodality; "PASS" booked on an unmet bar. Each was self-corrected — but the pattern is the attack surface | **6** | The corrections culture is the shield; institutionalize it: every headline verdict gets an independent re-run before it leaves the repo |
| G1 | Single-author compute vs frontier labs: nothing above ~1.4M params has ever been trained; 50M/100M pairs exist as parameter counts only; downstream harness has zero results. "Inevitable" is a claim about scale, and scale is exactly what cannot be bought here | **9** | Redefine inevitability as *mechanism-level adoption* (quad2 as an FLA-library variant; theory others cite) + ONE 50M training run as the credibility anchor |
| G2 | "Brain-like" attracts neuroscience pushback: oscillations/theta-gamma/FEP/vigilance rhetoric invites demands that the analogies *do work*; the new parts are untested (surprise) or tautological (FGT=0), the tested parts are borrowed (ART, DFA, delta rule) | **7** | Constrain brain claims to two tested sentences: (1) routing purity 100% label-free in-domain-incremental; (2) delta write = one GD step on per-token free energy. Everything else is labeled hypothesis |
| G3 | The honesty culture is a speed limit: pre-registration + refusal-to-extrapolate + ~28 A100-hour re-runs means slow cadence while the GDN commodity improves monthly | **6** | Split claim classes: engineering claims iterate fast without pre-registration; scientific claims keep the full ceremony |

## A.2 The five deadliest objections (detailed)

### T1 — The commodity objection: you are a late entrant claiming the winner's throne
Gated DeltaNet is not a competitor paper; it is **deployed** — the linear layer of Qwen3-Next-80B-A3B
at a 3:1 GDN:attention ratio, with Kimi Linear (KDA), Jamba/Samba/Falcon-H1/Granite-4 filling the
same slot *(knowledge, per report 07's sourcing)*. Every §4 PASS in this repo — MQAR param-efficiency,
induction, selective-copy, constant memory — is exactly what that family already demonstrates in
production. A hostile reviewer says: "Prizma-Seq is GDN + a random-projection feature map + a window
head. The repo's own ledger agrees: everything is borrowed except the framing." Severity 9 because
it caps the *entire* goal: no small-scale diagnostic suite can distinguish Prizma from a commodity
it is a member of.
**Would have to be true to survive:** Prizma must own at least one axis the commodity cannot copy
without abandoning its identity: (a) local/BPTT-free learning (nothing in the family has it),
(b) the free-energy/optimizer theory of *why* the family works, (c) the lifecycle axis (T12–T14),
(d) brain-grounded *falsifiable* predictions. And it must show ≥ TOST-parity with GLA/Mamba-2 —
currently unmeasured (T6).

### T2 — The wall-clock objection: the 5× gap cancels the one dynamics win
Report 08 found what the repo's honest-limits section discloses but never multiplies: Prizma
ignites in 16–20k steps vs the TF's 44–80k — but trains ~5× slower per step (1889 vs 361 s/arm at
char-LM scale). Product: **80–100 TF-step-equivalents vs 44–80k.** The training-dynamics headline
("converges in ⅓ the steps") is fictitious in wall-clock and arguably parity-to-worse. The cause is
implementation, not mathematics: ~240 serialized small-kernel launches per layer per forward in the
chunked WY/UT loop, unfused log/cumsum/exp chain, an as-coded full-T² window head. And the two
artifacts that could fix it are **parked**: the Triton kernel is gated out of every training call
(`_should_use_triton` requires `not grad_live`; a backward-capable function exists in the same file
but is unreachable), and `delta_fused` awaits an unexecuted A100 verification. Severity 9 because
every training-based claim (T4, T5, BAR-4 below) inherits the 5× tax.
**Defuser:** one GPU session, pass/fail: enable the compiled/backward kernel in training, verify
bit-equivalence, re-time. Bar: ≤1.5× TF step time, or the project's center of gravity moves to
inference/edge (BAR-3).

### T3 — The bottleneck objection: O(1) memory is a ceiling, not just a feature
Round-1 theory already established capacity ≈ linear in d_h (D* ≈ 0.5–1.0·d_h). quad2 raises the
*rank* of the fixed state, not its finiteness. The commission keeps winning on MQAR at D=128 — a
task chosen in 2024 because it broke RecNets. The field's current long-context arena is
recall-at-scale: multi-query/multi-hop retrieval at 10⁵–10⁶ tokens (RULER-class) *(knowledge)*,
where production hybrids *keep full-attention layers precisely because pure-linear recall degrades
there* (report 07 notes this from the hybrid papers' own ablations). A 1M-token needle task is a
structural kill: the KV cache holds the needle losslessly; S compresses it. Severity 8.
**Defuser:** do not fight on that field. Pre-register the state-recall ceiling as a *law*
(D* as a function of d_h, d_φ, H), publish the D-frontier, and claim the Pareto: for documents
where most context matters briefly, constant state + long-context latency/memory win + (future)
consolidation-into-weights. If a reviewer can defeat the ceiling with one task, the claim must be
scoped so that task is explicitly out of scope — otherwise the claim dies on contact.

### T4+T5+T6 — The evidence-debt cluster (these three together are why nothing is citable yet)
- **B4 char-LM:** pre-registered BOTH corpora ≥3 seeds (≥5 for this leg, the closest leg).
  Delivered: one corpus at n=2 after the other corpus failed −0.09 and its raw data was **not
  retained**. Later n=7 vs n=10 with a real gap (+0.021, Welch t≈6.7) — still one corpus, still a
  hair behind. This is the most quotable negative in the repo and it is fully self-inflicted.
- **Recall gate:** the only powered, Holm-corrected, TOST-designed artifact in the repo is
  quarantined; equivalence was not met on any leg; the honest recall claim is "competitive," a
  non-result. ~28 A100-hours to fix; not re-run.
- **GLA/Mamba-2:** faithful implementations, zero results. Every comparison in the repo is against
  the Transformer — i.e., against the architecture family Prizma claims to *leave*, never against
  the family it *belongs to*. A reviewer's first question is "how does it do against Mamba-2?" and
  the honest answer today is "nobody knows."
Severity 8 each; jointly they mean the repo currently cannot support even the modest word
"candidate" outside its own website.

### T7 — The mechanism objection: the brand's signal is evidence-negative
The two things docs/Prizma.md calls NEW are (a) the surprise-driven two-readout synthesis and
(b) the precision phase detector. The Seq-thread instantiation of (a) — the surprise gate — has
exactly one ablation: n=2, smoke, verdict INCONCLUSIVE, and its ordering is constant > random >
real > none. The report's own ledger now says: "asserted, not demonstrated; the only evidence
collected so far points the wrong way." Meanwhile half the commission (03 P1, 04, 05 P1–P4, 06 P2/P3,
09, 12 H4) **builds on this signal**. Severity 8 for the brain-like goal specifically: the one
mechanism that would make "predictive coding" a mechanism rather than a derivation is the one the
repo's own data currently votes against.
**Defuser:** the cheapest experiment in this entire report — a powered surprise ablation (≥5 seeds,
LR-swept, with constant/random controls, on MQAR + char-LM). Its outcome sets the ceiling on every
FEP-flavored claim in all 12 sibling reports.

## A.3 Scientific-priority and strategic objections (compressed)

**S1 (FEP branding, sev 5):** PC-for-LM has lineage *(knowledge: Salvatori et al.'s PC
transformers; Millidge/Tschantz/Buckley; Ororbia's analog PC LM work)*. The defensible asset is the
*exact identity* — the delta write is one gradient step on a per-token free energy — which nobody in
the GDN family states. Claim that, cite the lineage, and "FEP" as a banner word can be dropped
without losing anything real.

**S2 (Neuromorphic, sev 5):** zero hardware numbers, idealized noise story, endurance wall. Report
12's discipline ([MEASURED]/[ARITHMETIC]/[CITE-CHECK]) should be adopted repo-wide before any
neuromorphic sentence leaves the README.

**S3 (The honesty pattern, sev 6):** simulated-CUDA rows deleted; stale test count + green badge
over red CI; unretained failed corpus; cache artifact misread as baseline bimodality; a PASS booked
on an unmet bar. Every one was self-corrected — the corrections culture is genuinely rare and is
the project's best asset. But a hostile reviewer will use the *pattern*: "this lab's positive
results have a 100% history of needing post-hoc correction." The only durable answer is
independent re-runs of headline verdicts (see BAR-0).

**G1 (Compute asymmetry, sev 9):** the single deadliest strategic fact: the commission is writing
theory for a 1.4M-param empirical base while the commodity runs at 80B. No amount of committee
quality changes that the 50M/100M models have never been trained and the downstream harness has
zero results. "Inevitable rival" is a scale claim; scale is the one currency this project cannot
print. Reframe (BAR-4) or descope (BAR-3).

**G2 (Neuro-pushback, sev 7):** the more "brain-like" the rhetoric, the more the project borrows
neuroscience's burden of proof. The repo's honest parts (routing purity, tautology admission) will
be respected; the borrowed rhetoric (theta-gamma, grid cells as "nominally present — a frozen RBF
lift") will be attacked. Constrain brain claims to tested sentences (see ledger G2 defuser).

**G3 (Honesty as speed limit, sev 6):** pre-registration costs GPU-months and the GDN commodity
improves monthly. The asymmetry is real but the answer is not to loosen standards — it is to
**spend the pre-registration budget only where it pays** (Part B's bars), and let engineering
iterations run fast without booking verdicts.

## A.4 Red-teaming the commission itself (the 12 siblings)

### A.4.1 Convergence map (what the commission secretly agrees on)
Three convergences, in order of importance:
1. **The powered surprise ablation is the hidden prerequisite of at least 6 of 12 reports.**
   06 makes it "prerequisite #1" (its P3a); 03's rank-1 salience-write proposal, 04's NE mapping,
   05's entire precision-as-object program, 09's phase-gating and 12's H4 all *spend* the surprise
   signal. The commission has built a cathedral on the one beam whose only load test ran the wrong
   way. (This is also the commission's implicit answer to "which cheap must-do experiment?" —
   run it first.)
2. **"Run the apparatus you already built" appears in every strategic report.** 07 says it flatly
   (the 4-arm landscape is "the repo's own credibility debt"); 08's fused-kernel session; 06's
   banded-window + sparse-state ledger; 12's emulation campaign on existing hooks. Four reports,
   one message: **the repo does not need new mechanisms, it needs results from existing ones.**
3. **Three reports independently invent the same flagship** — state→weights consolidation /
   lifelong expert LM: 02's P1 (CLS loop), 10's P5/P6 (LM lifecycle), 06's vigilance-MoE fusion.
   Triple-counting one idea inflates the commission's apparent productivity; the idea also
   inherits CL's weakest flank (T13 interleaving) and carries the "replay in disguise" objection
   that 02 honestly flags itself.

### A.4.2 Hero-architecture bias
- **01 and 05 both rank "deep predictive hierarchy / cross-layer error propagation" #1** with
  impact 9 — each within its own lens, neither citing the other. Stacking a new vertical channel
  onto an architecture whose horizontal novelty (surprise gating) is undemonstrated is
  mechanism-stacking on an unverified foundation. Both proposals add params to the one asset whose
  selling point is byte-identical borrowed components, and both quietly break the "only the token
  mixer differs" experimental design that makes the repo's comparisons clean.
- **06's analytical FLOP ceiling (0.73× sub-parity)** is committee arithmetic on top of
  committee arithmetic (sparse top-k + banded + skip, all assumed to compose without loss).
  To its credit it flags this itself; but note this is the *same failure mode* as the deleted
  simulated-CUDA rows — extrapolated numbers that would flatter the project. The number must never
  be quoted outside a `flop_ledger.py` re-derivation.
- **11 is the exception that proves the rule:** its rank-1 proposal is to *fix the theory that
  disproves the README* (Gershgorin N<14 vs measured D=128). That is what intellectual honesty
  looks like; two other reports rank higher-impact mechanisms without confronting the failed load
  test underneath them.

### A.4.3 Brain-analogy overreach
- **09 (oscillations) is the most exposed.** It concedes the killer line itself: an external
  one-tick-per-token oscillator is in danger of being "position indexing wearing a lab coat" — and
  RoPE already is a phase code, while Mamba-2/S4D lineage already carries oscillatory decaying
  modes *(knowledge)*. Adding oscillation to a monotone-decay delta state is a real idea, but its
  brain-credit will be contested to zero if its *engineering* value does not show first; the
  report's own P9 null-harness is the right order of operations and should be a precondition for
  the rest.
- **02's grid-cell "HEAD"** exists in the code as a frozen RBF lift — the report says so honestly,
  then still proposes HEAD-adjacent mechanisms. The honest version of the hippocampal report is
  80% of its content and none of the cell-type vocabulary.
- **04's neuromodulator dictionary** is the best-disciplined of the brain reports (each mapping is
  marked present/static/absent) — but "Dopamine = the delta write" flatters a two-factor local rule
  that lacks every property dopamine actually has (political economy of circuits, timescale
  diversity, non-specificity). Usable as a search heuristic; not citable as biology.

### A.4.4 Benchmark cherry-picking
- The commission's collective experimental center of gravity is **MQAR D=128** — the exact task
  family where a rectangular-rank fixed state most outperforms its capacity class. No sibling
  report proposes the *field's* current long-context recall arena (RULER-class, 1M-token) except to
  avoid it, and only 07 admits that production hybrids keep attention layers because pure-linear
  recall degrades downstream. The commission should pre-register at least one hostile-field task
  per proposal, or the phrase "inevitable rival" is being tested on Prizma's home turf only.
- Several reports quote the **10× length-extrapolation retention** without the paired disclosure
  that Prizma's *absolute* accuracy at 8× is ~0.40 (the repo always pairs them; the commission must
  too).
- Nobody proposes re-running **tiny-shakespeare**. The failed corpus has become invisible in 12
  reports. That is exactly how B4's deviation happened the first time.

### A.4.5 The one-sentence commission verdict
The commission is excellent at generating mechanisms and weak at sequencing them: 12 reports
propose ~70 new ideas while the repo's binding constraints say **the next three moves are
re-runs, not inventions** (surprise ablation, clean recall gate, B4 redo, landscape run, kernel
session). My Part B is that sequence, made binding.

---

# PART B — WHAT MUST BE TRUE

## B.1 Pre-registered kill-criteria bars

The word "inevitable" is banned from all project documents until BAR-1 through BAR-4 all read PASS.
Each bar names what the project **pivots to** on failure — pivots are pre-committed, not punishments.

**BAR-0 (Integrity bar).** Before any headline verdict leaves the repo: the seed-order bug in
`run_cell` (init not pinned by seed) is fixed and the three headline MQAR cells re-run; every
"quarantined" artifact is either re-run clean or its subsection is deleted. *Fail ⇒* no
external communication of any Seq result until done. Cost: ~1 GPU-day. (This bar has no pivot;
it is a precondition.)

**BAR-1 (Family bar — the commodity test).** Clean 4-arm landscape run (TF / Prizma / GLA /
Mamba-2), powered seeds (≥5), Holm-corrected, LR-swept per arm, on MQAR-hard + induction +
selective-copy. **Pass:** Prizma-quad2 is TOST-parity (±0.05) or better vs **GLA and Mamba-2** on
≥2 of 3 legs. **Fail ⇒ pivot:** stop claiming architecture-level novelty; reposition Prizma-Seq as
(a) the theory paper (free-energy/optimizer view of the family) + (b) the local-learning axis, and
contribute quad2 upstream to the FLA ecosystem rather than as a rival architecture. Cost: ~30–40
A100-hours (harness exists).

**BAR-2 (LM bar — close B4 properly).** Both corpora (text8 + tiny-shakespeare, raw data
retained this time), ≥5 seeds, init pinned to seed, symmetric dropout lever on. **Pass:** test BPC
≤ TF + 0.05 on **both** corpora. **Fail ⇒ pivot:** the "credible attention-replacement candidate"
wording is deleted repo-wide; Prizma-Seq is rescored as a recall/synthetic-diagnostic specialist
with a long-context serving niche, and the LM axis is honestly labeled negative. Cost: ~2 GPU-days.

**BAR-3 (Speed bar — the wall-clock precondition).** One A100 session: enable the
compiled/backward-capable delta kernel in training, verify equivalence, re-time. **Pass:** train
step-time ≤1.5× matched TF at d128L4H4 and char-LM scale. **Fail ⇒ pivot:** the project's center of
gravity moves from "training rival" to **inference/edge deployment niche** (constant state + local
adaptation amortize a one-time training cost), and all training-dynamics headlines are withdrawn.
Cost: 1 GPU-session. (Report 08's analysis says PASS is plausible; the point is to make it binding.)

**BAR-4 (Scale bar — the smallest run that changes the story).** Train ONE 50M pair
(Prizma 47.96M vs TF 48.02M, configs already instantiated) on ~1B tokens of a standard stream;
pre-register: Prizma val BPC ≤ TF + 0.10 (relaxed margin, disclosed as such) AND the recall
diagnostic suite holds at 50M. **Pass:** the "small-scale-only" rider is deleted and the 372×
memory arithmetic gets a measured twin. **Fail ⇒ pivot:** permanent honesty rider "validated
≤1.4M params"; inevitability redefined as mechanism-level adoption only. Cost: the single largest
line item in this report (~2–4 GPU-weeks at owner's Colab cadence) — but it is the only experiment
that can move T1/G1 from severity 9.

**BAR-5 (Mechanism bar — the brand's load test).** Powered surprise-gate ablation (≥5 seeds,
LR-swept, constant + random controls, MQAR D=128 @130K + one char-LM). **Pass:** the real surprise
signal beats both controls with CI-separated margin on ≥1 task. **Fail ⇒ pivot:** every FEP /
"precision-gated" mechanism claim is deleted from branding (the PC identity survives only as the
exact derivation, which is true regardless), and the brain axis re-anchors solely to the CL
routing result. Cost: ~1 GPU-day.

**BAR-6 (CL bar — escape the narrow regime).** Interleaved-stream structured-permuted, K=5, ≥10
seeds, dynamic-vigilance arm (report 04 P1) vs static. **Pass:** ACC ≥ 0.70 with FGT ≤ 0.05,
CI-separated from static (which sits ~0.58), expert count ≤ 2K. **Fail ⇒ pivot:** the CL headline
is permanently scoped to "block-contiguous domain-incremental task-identity inference" — the §8
reframe becomes the whole claim, and the "continual learning" banner word is dropped. Cost: CPU-scale.

## B.2 Constructive counter-proposals that survive my own attack

### CP1 — "Debt-zero before mechanism-one" (the sequencing fix)
- **WHAT.** A single pre-registered campaign executing BAR-0/1/2/3/5 in that order; no new
  mechanism from reports 01–12 is implemented until the campaign closes.
- **WHY.** Every sibling proposal's expected value is currently uncomputable because the base
  rates are unknown (surprise signal evidence-negative; family parity unmeasured; kernel parked).
  Mechanism-stacking on unverified foundations is how the commission converts honest assets into
  future corrections. The debts are also *cheap* (~1–2 GPU-weeks total) relative to any new mechanism.
- **HOW.** Owner runs the campaign notebook(s) on Colab in 3–4 sessions; verdicts land in the
  report's §4 table with the same binding language as B4's correction.
- **FALSIFIABLE TEST.** The bars themselves; each has a pre-committed pivot.
- **RISKS.** Morale cost of a possible triple-negative month (recall non-result confirmed, B4
  marginal, kernel slow); mitigated by the pivot clauses — a failed bar still produces a
  publishable honest scope, which is this repo's brand.
- **IMPACT 10 / FEASIBILITY 9.** This is the highest expected-value action available to the project.

### CP2 — The one-scale-up run (BAR-4 as the project's credibility anchor)
- **WHAT.** The 50M training run described in BAR-4 — nothing above it, no multi-run sweep.
- **WHY.** It is the *only* experiment that attacks the two severity-9 objections (T1 commodity,
  G1 compute) that reframing alone cannot touch. A single 50M datapoint converts "never trained
  above 1.4M" into "has a 50M result" — one number, enormous narrative leverage, and it exercises
  the analytical 372× memory claim against reality.
- **HOW.** Owner Colab cadence; streaming dataset harness already exists (`seq/downstream.py`);
  pre-register the margin (+0.10) and the diagnostics before launch; retain raw artifacts (B4's lesson).
- **FALSIFIABLE TEST.** BAR-4 as written.
- **RISKS.** 2–4 GPU-weeks may fail mid-run (disconnect-truncation happened before — checkpoint
  crash-safely, per HANDOFF guardrail); a fail at 50M is a real negative result that must be
  reported (it would sit honestly in the README's corrections section).
- **IMPACT 9 / FEASIBILITY 5.**

### CP3 — Reframe inevitability: "the family's theory-and-lifecycle member" (drop the war Prizma can't win)
- **WHAT.** Officially adopt report 07's wedge: Prizma does not replace the delta-rule family —
  it is *the member of that family with* (a) an exact free-energy/optimizer theory (11's toolbox:
  the write is one GD step; step-size schedules, conjugate acceleration apply verbatim),
  (b) a local/BPTT-free training mode (unique in the landscape), (c) a lifecycle axis (02/10), and
  (d) falsifiable brain-grounding. Concretely: one theory paper + one FLA-library contribution of
  quad2 + the CL lifecycle demo; the phrase "Transformer rival" is retired.
- **WHY.** It is the only framing under which every existing measured asset is an asset rather
  than an embarrassment, and it converts the commodity threat (T1) into distribution (quad2-in-FLA
  = adoption = the only "inevitable" a solo project can actually achieve).
- **HOW.** Documentation reframe (a README surgery), theory paper from report 11's program,
  library contribution after BAR-1.
- **FALSIFIABLE TEST.** Adoption/external-citation of quad2 or the capacity law within 12 months;
  the theory paper's D-frontier predictions confirmed by the (currently deferred) P3 run.
- **RISKS.** Ego cost of retiring "rival"; library contribution may be ignored (that is a
  measurable fail of the test above, not a catastrophe).
- **IMPACT 8 / FEASIBILITY 7.**

### CP4 — The unified demonstration: continual char-LM with state→weights consolidation
- **WHAT.** Report 02's P1, executed *after* BAR-5 and on a scope that dodges its objections:
  corpus A → corpus B sequential training; S drains into slow weights via S-generated pseudo-items
  (closed buffer, never raw data); measure A-retention, B-learning, and state-freed recall.
- **WHY.** It is the one experiment where **no Transformer baseline can even enter without
  importing a replay buffer** — the only battlefield where Prizma's two threads compose into a
  structurally unmatched capability, and the strongest legitimate "brain-like" claim available
  (CLS, cited as prior art, novelty = state-generated replay inside a delta-state mixer).
- **HOW.** Off-by-default `consolidate()` hook; pre-registered bars from report 02 verbatim
  (A-retention within +0.10 of pre-B, no-loop control degrades ≥0.25, probe-recall drop <0.5).
- **FALSIFIABLE TEST.** Exactly those bars; failure = "consolidation moved nothing" reported.
- **RISKS.** "Replay in disguise" objection (mitigate: closed S-only buffer, disclosed); garbage
  reconstructions from low-rank S (confidence gate); multi-epoch CL confound from T13 — run
  single-pass or disclose epochs.
- **IMPACT 8 / FEASIBILITY 5.**

### CP5 — The hostile-field pre-registration (fix the commission's home-turf bias)
- **WHAT.** Before any new mechanism is booked, pre-register **one task chosen by an adversary** —
  drawn from the field's current arena (e.g., a RULER-style multi-hop retrieval at 65k–1M tokens,
  and one recall-intensive downstream proxy at 50M scale) — alongside the home tasks. Publish the
  failure curve with the wins.
- **WHY.** BAR-1/2 test Prizma against its own family on its own tasks; T3 says the *field* will
  test it at 1M tokens with a growing KV cache. A pre-registered hostile task converts the
  bottleneck objection from a referee's ambush into a disclosed Pareto boundary — which is the
  difference between "they hid the ceiling" and "they mapped the ceiling."
- **HOW.** Add two cells to the existing landscape harness; pre-register the expected failure
  (recall ≥ some D degrades; retention vs the TF's O(t) cost at matched accuracy is the honest
  axis).
- **FALSIFIABLE TEST.** The ceiling law itself: measured D* at 50M must track the capacity
  prediction (report 11's random-matrix law) — this doubles as 11's decisive falsification run.
- **RISKS.** Publishing a bad number on the field's favorite task invites mockery; answered by the
  Pareto framing and the fact that the commodity hybrids *also* fail pure-linear there (07's
  observation — cite their own ablations).
- **IMPACT 7 / FEASIBILITY 6.**

## A.5/B summary — what would have to be true, in one paragraph

For "Prizma is an inevitable Transformer rival" to survive, all of the following must hold
simultaneously: parity with the *family* (not just the Transformer) at powered seeds; LM
competitiveness on **both** corpora with retained raw data; a training step-time within 1.5× of
attention; at least one real-scale (50M) datapoint; and a mechanism ablation where the brand's
signal beats its own controls. That conjunction is currently 0/5 measured. The honest, survivable
version — "the free-energy-derived, locally-learnable, lifecycle-capable member of the delta-rule
family that won, with every ceiling mapped and every negative result retained" — is achievable at
this project's actual compute budget, and this commission's 12 reports are worth precisely nothing
until BAR-0 through BAR-6 have run.

---

## QUESTIONS FOR THE OWNER

1. **Compute reality:** what is the true monthly A100/Colab budget in hours? (BAR-1/2/4 are sized
   against it; if it is <20 h/month, BAR-4 must move to a grant/collaboration track and the pivot
   in BAR-3 becomes the default strategy.)
2. **Goal hierarchy:** if forced to choose, is the priority (a) architecture adoption
   (quad2-in-FLA, theory citations), (b) the neuromorphic hardware story, or (c) the continual-LM
   lifecycle demo? Each implies a different bar ordering; CP3 assumes (a).
3. **The "inevitable" word:** do you accept retiring "rival to the Transformer" in favor of
   "the family's theory/lifecycle member" *before* the bars run — or must the bars fail first?
   (This changes BAR wording, not the experiments.)
4. **B4's failed corpus:** tiny-shakespeare's raw data was not retained. Do you accept "rerun and
   retain" as BAR-2, or is there a reason (e.g., known recipe pathology) to replace the corpus —
   and if so, which one, pre-registered now?
5. **Surprise-gate stakes:** if BAR-5 fails (the signal loses to constant/random controls at
   power), do you authorize deleting the precision/surprise mechanism claims from all branding in
   the same commit that records the failure? (12 sibling reports depend on your answer.)
6. **Timeline tolerance:** a clean Debt-zero campaign (CP1) is 3–4 GPU sessions and possibly three
   negative verdicts. Is a negative-heavy month acceptable if every verdict lands with a
   pre-committed pivot — or does the project need a positive headline sooner (in which case BAR-3,
   the kernel session, is the most likely PASS to schedule first)?
