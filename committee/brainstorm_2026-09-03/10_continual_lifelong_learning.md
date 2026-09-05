# Brainstorm — Continual & Lifelong Learning lens: from a toy-regime routing machine to a lifelong LM

**Commission:** Prizma → (1) inevitable Transformer rival + (2) maximally brain-like
**Author:** Continual & Lifelong Learning Specialist (catastrophic forgetting literature; task-free CL;
DER++/ER; EWC/SI/MAS; progressive nets; gating networks; continual pretraining of LLMs;
domain-/class-/task-incremental taxonomy)
**Date:** 2026-09-03
**Classification (per brainstorming skill):** **Architectural-class research-direction brainstorm.**
HARD-GATE honored: **no code, no implementation** — strategy/ideas document only. The skill's
interactive clarifying questions cannot reach the owner from inside a committee, so all
direction-setting questions are collected in **QUESTIONS FOR THE OWNER** at the end.

Sources read before writing: `README.md` (both threads, incl. quarantine + B4 partial-verdict
sections), `docs/Prizma.md` in full (esp. §7 iteration log, §8 honest assessment, §4 borrowed-vs-new
ledger), `src/prizma.py`, `src/baselines.py`, `experiments/run_continual.py`, `docs/HANDOFF.md`
(Seq-thread state), and the sibling committee reports 01–09 (format + to avoid duplication with
06/07). Web search was rate-limited during this session; **every claim about 2024–2026 work is
marked *(knowledge)* and is qualitative only — no numbers are invented.** Classic CL citations are
from established literature (pre-2024) and are given author-year.

Borrowed-vs-new discipline: every proposal marks **[B]** borrowed / **[N]** new / **[B→N]**
borrowed-component-with-new-framing, per repo culture.

---

## 0. Executive frame: what Prizma actually owns, and where the money is

The one capability where the Transformer **structurally** fails and brains succeed is not recall
(commodity, see report 07) — it is **lifecycle**. A trained checkpoint is a file. Everything a
deployed LLM "knows" after training is frozen; in-context learning and RAG are *ephemeral state*,
not plasticity, and nothing a Transformer picks up in context ever *moves into weights*. Move a
checkpoint into a new domain by naive gradient descent and you get catastrophic forgetting
(McCloskey & Cohen 1989; French 1999) — which is why every industrial lab does continual
pretraining with a replay percentage and still measures degradation *(knowledge; e.g. IBM's
continual-pretraining study — Ibrahim et al. 2024 — reported that even ~1% replay of original-corpus
data is needed to keep perplexity from exploding; exact figures *(knowledge)*). Brains, by contrast,
run **complementary learning systems** (CLS; McClelland, McNaughton & O'Reilly 1995): fast
hippocampal acquisition, slow neocortical consolidation, no catastrophic overwrite. Prizma is, by
construction, the only member of this commission's portfolio whose *learning rules* are built for a
lifecycle rather than a run.

Now the honest center of gravity, which the author himself stated in `docs/Prizma.md` §8: **FGT=0
is an architectural quasi-tautology; the real achievement is the routing.** In my taxonomy terms,
Prizma is not primarily a *continual-learning method*; it is a **task-identity inference machine**
— it recovers the oracle multi-head's task id (recon-surprise argmin) with 100% purity, label-free
and boundary-free, in the input-distinguishable regime — and FGT=0 is the corollary of perfect
task inference plus frozen experts. This is a genuine and rare mechanism. But its current territory
is small and, candidly, a regime where the community's hardest open problems do not live:

- The benchmark is domain-incremental with **fully distinguishable** domains (permutation-structured
  covariance), **contiguous** blocks, **batch-level** novelty decisions, and **multi-epoch** fits
  (`fit_task`, epochs=10 default, 15 in E1) — i.e. the stream is block-online, not single-pass.
- The expert pool is sized with the number of domains **known a priori** (`n_experts = K + 3`).
- E5 honestly proves no claim in the ambiguous (class-incremental-like) regime; §8 honestly reports
  collapse (~ACC 0.58) under interleaving.

Every proposal below therefore does one of three things: **(A) map the territory honestly** (P1,
P2, P4), **(B) make the routing achievement predictive and durable** (P2, P3, P7, P8, P9), or
**(C) carry the territory-holder into the LM lifecycle** where forgetting is unsolved and the
economic prize is enormous (P5, P6). Ranking is by expected value (IMPACT × FEASIBILITY), with the
flagship fusion deliberately ranked mid-list because its feasibility is real but GPU-bound.

A terminology warning used throughout: the CL literature's "task-free" (Aljundi et al. 2019) means
no boundaries **and** no task-count knowledge **and** single-pass online exposure. Prizma is
task-free in boundaries and labels only. This gap, not FGT, is the citability gap.

---

## 1. Code-level audit from the CL lens (facts the proposals build on)

From `src/prizma.py` / `experiments/run_continual.py` / `docs/Prizma.md`:

1. **Freeze is permanent.** `_train_expert` returns immediately if `e.frozen`; reawakening never
   happens in code. The design doc §3 *specifies* a reawakening rule (`ω_m ← ω_m − κ·relu(conflict)`)
   — it is **specified but never implemented**. This is a design-to-code gap and the natural entry
   point for every overlapping-domain proposal (P7).
2. **Inference routing has no abstain.** `route_for_inference` takes argmin over trained experts;
   a genuinely novel input at test time is force-routed to the nearest expert and answered
   confidently. Open-world deployment needs a vigilance rejection output ("NOVEL") — ART has it;
   the prototype dropped it at inference.
3. **Memory grows linearly and without bound in the number of domains.** Per expert: encoder +
   decoder + head = `h·d + h + d·h + d + K·h + K` (= 2,768 floats at d=24, h=48, K=8) **plus** two
   fixed FA matrices (`Bdec` h×d, `Bcls` h×K; +1,536). Total ≈ 4,304 floats/domain, forever.
   "No buffer" is true; "no memory growth" is not. Replay's buffer is budgeted; Prizma's weights
   are not. Quantification and a bound are overdue (P3).
4. **Usage is already tracked.** `route_log` accumulates per-expert sample counts — the instrument
   needed for pruning/merging statistics already exists (P3).
5. **Batch-level decisions, not sample-level.** Novelty is decided on batch-mean recon surprise
   with a per-expert precision test (`μ + z·σ`, z=5, warmup 256). This was *learned the hard way*
   (§7: per-sample vigilance thrashing in v2). Any proposal that reintroduces sample-level routing
   must cite that failure.
6. **Multi-epoch exposure.** Every arm gets `epochs=15` per task. Standard online-CL protocols
   (er-ACE, DER++, OCM) are single-pass. The "no buffer" claim is stronger than the protocol
   demonstrates (P4).
7. **The EWC comparator uses task boundaries** (offline Fisher, `baselines.py`), while
   boundary-free regularizers (MAS, online EWC, SI-online) were never run (P4).

---

## 2. Proposals

Format: WHAT / WHY / HOW / FALSIFIABLE TEST / RISKS / IMPACT / FEASIBILITY / provenance.
Ranked order appears in §3; the proposals are written in logical order.

---

### P1. The Regime Map — a controlled distinguishability→overlap degradation study with buffer-matched, task-free SOTA baselines

**WHAT.** Generalize the benchmark from a binary (distinguishable / impossible) to a **continuum**:
parameterize domain overlap κ ∈ [0, 1] where task t's inputs are drawn from
`κ·N(μ_t, Σ_t) + (1−κ)·N(μ_t', Σ_t')` (structured-permuted base retained). κ=0 reproduces E1's
regime; κ→0.5 approaches E5's ambiguity. For each κ, measure a pre-registered dashboard:
routing purity (expert↔domain confusion), ACC, FGT, and *where each new expert got recruited*
(recruitment-point drift). Run the full battery of 2020s baselines **buffer-matched to Prizma's
actual weight growth** (see §1.3): naive, EWC (boundary), online EWC / SI / MAS (boundary-free),
ER / DER++ / MIR / GDumb at buffer sizes whose sample-storage bytes equal Prizma's
`(K+3) × 4,304`-float expert memory, SHE *(knowledge: task-free exemplar-free class-incremental,
prototype heads)*, and the task-id oracle.

**WHY.** The repo *gestures* at this (E2 noise sweep: FGT stays ≤0.077 up to noise 1.2 with ~5
experts committed, ACC decays gracefully) but noise injection is a weak blurring axis — it shrinks
all margins uniformly rather than creating *partial* overlap where two domains share most of their
mass. The κ-axis asks the question that matters scientifically: **does vigilance routing degrade
smoothly (a phase-transition curve), or does it catastrophically merge experts at a critical κ\*?**
ART theory predicts a critical similarity (vigilance) threshold; nobody has measured Prizma's.

**HOW.** All numpy, CPU, same scale. New `src/overlap.py`-style task generator (κ-mixtures of the
existing structured-permuted base); extend `metrics.py` with routing purity; a `E6_regime_map.py`
in the exact falsifiability style of `run_continual.py`. Buffer-matching arithmetic is closed-form
from §1.3. ~1–2 days of compute on a laptop.

**FALSIFIABLE TEST.** Pre-register: (i) there exists κ\* (95% CI over seeds) where routing purity
drops below 90%; (ii) Prizma's FGT stays ≤ 0.6·naive-FGT for all κ < κ\*; (iii) Prizma beats
DER++-buffer-matched at equal memory for κ < κ\* — **if DER++ at matched memory beats Prizma at
κ=0, the entire headline is in trouble and must be re-framed** (this is the control the literature
will demand first, because GDumb showed big-buffer replay is embarrassingly strong *(knowledge)*).
(iv) The routing-purity curve against κ matches the P2 prediction (see below).

**RISKS.** Results may show the honest territory is narrower than E1 suggests (that is the point —
the repo culture prefers a precise small claim to a vague big one). Mixture tasks with shared
labels may be label-ambiguous like E5 — pilot first to keep the *classification* target
well-defined per mixture component.

**IMPACT 9 / FEASIBILITY 9.** [B] κ-mixtures (standard distribution-shift construction), ER/DER++/
MIR/GDumb/SHE protocols (Buzzega 2020; Prabhu 2020; Aljundi 2019; De Lange & Tuytelaar 2021);
[N] the κ-continuum applied to vigilance-recruited experts and the buffer-equals-weights budget rule.

---

### P2. The Routing Law — a Chernoff/error-exponent theory that *predicts* the map before running it

**WHAT.** Formalize the §8 insight ("the real achievement is unsupervised task-id inference") as a
hypothesis-testing statement, and derive, per domain pair (D_i, D_j), the probability that a batch
from D_j passes expert i's vigilance test (recon error ≤ μ_i + z·σ_i). For the Gaussian benchmark
family these mis-recognition probabilities are closed-form overlap integrals; the asymptotic decay
rate is the Chernoff information between the two domains' recon-surprise distributions
(Chernoff 1952; Wald 1945 for the sequential-test framing). Deliverable: a plotted **predicted**
routing-failure curve, overlaid on P1's measured curve.

**WHY.** Two payoffs. (a) It converts the "quasi-tautology" into a *quantitative separability
criterion*: Prizma works iff batch-scale Chernoff information between consecutive domains exceeds
≈ log(K+3) + log(1/δ) — a citable, falsifiable law, and the first predictive account of *when*
surprise-routing CL works. (b) It is the theory spine the Seq-fusion needs: at LM scale, language
domains share ~99% of their statistics, so everyone will ask "how would vigilance ever fire?" —
P2 is the instrument that answers with a number instead of a hope.

**HOW.** Pure math + a validation notebook. Recon surprise under a linear autoencoder of a Gaussian
input is a quadratic form → its distribution is a generalized chi-square; pairwise overlaps are
numerically computable even where closed forms are messy. No GPU.

**FALSIFIABLE TEST.** (i) Measured routing purity per κ ≥ predicted purity − 2σ across ≥10 seeds;
(ii) a *designed* domain pair built exactly at the predicted critical overlap flips Prizma between
clean-routing and merged-expert behavior with high probability; (iii) the z=5 default is shown to be
near-optimal under the derived loss curve (or is shown suboptimal — either result is publishable).

**RISKS.** The prototype's learned recon distributions may drift from Gaussianity during training
(theory on the *fixed* substrate vs *learned* substrate gap) — mitigate by stating the theory for
the pre-training substrate and validating drift empirically. Risk of the theory being "correct but
uninformative" if the Gaussian family is too easy — mitigation: instantiate it also on the κ-mixed
family of P1.

**IMPACT 9 / FEASIBILITY 6.** [B] Chernoff information, SPRT, Gaussian discrimination theory;
[N] vigilance-routing regime prediction in CL ("Chernoff-vigilance law") — genuinely new positioning
of the author's §8 reading; would be the *theory section* of the CL paper this repo deserves.

---

### P3. Expert Economy — pruning, merging, recycling: a memory bound for an open world

**WHAT.** Close the unbounded-growth hole (§1.3) with the three standard organs the design lacks:
(a) **prune**: an expert whose `route_log` mass has been ~0 for W samples and whose domain has
re-decayed is recycled (weights re-initialized, `committed=False`); (b) **merge**: when two frozen
experts' routing decisions co-fire (confusion-graph edge weight > θ) or their recon-surprise
distributions are within z-merge, merge by weight-averaging + union of routing mass
(DEN's network-splitting in reverse — Yoon et al. 2017; Compressed/Distributed ARTMAP category
merging — Carpenter; SHY-style downscaling — Tononi & Cirelli 2014); (c) **hard budget**: M_max
with queue discipline, reporting ACC-vs-M_max the way E3 reports ACC-vs-M.

**WHY.** Brains prune and re-allocate; the current design only accretes. Also PR: "no buffer" is
currently purchased with unbounded weights — a reviewer will find §1.3 arithmetic in an afternoon.
A bounded-memory Prizma with quantified ACC cost converts that attack surface into a result.

**HOW.** `route_log` already carries usage (§1.4); merging needs only the confusion statistics of
`_recognizes` across expert pairs — all present machinery, numpy-only. Pre-register the merge
criterion before looking at ACC effects (repo culture).

**FALSIFIABLE TEST.** On P1's κ < κ\* regime plus a long multi-domain stream (K=20 ≫ M_max=8):
bounded Prizma retains ≥ 95% of unbounded ACC; merged experts' joint ACC ≥ 0.95× pre-merge ACC on
their two domains; pruning *never* deletes an expert whose domain reappears within the stream
(measured reappearance-hits ≥ 0 false prunes with 95% CI) — plus the open-world extension of
routing abstention from §1.2 (NOVEL output) scored with open-set metrics.

**RISKS.** Merging two PCs autoencoders by averaging is only principled if their latent bases align
(FA matrices are random and expert-specific — merging encoders is *not* obviously sound). Fallback:
merge only heads + route mass, keep decoders separate (cheaper memory win, still bounded). Pruning
wrongly = catastrophic forgetting of a still-live domain — this is why the falsifiable test
demands zero false prunes.

**IMPACT 7 / FEASIBILITY 8.** [B] DEN splitting/pruning (Yoon 2017), PackNet-style budgeting
(Mallya & Lazebnik 2018), ART category compression, SHY; [N] usage/confusion-driven expert economy
for vigilance-recruited predictive experts.

---

### P4. The Citation Bar — boundary-free regularizers, single-pass protocol, and the 2024–2026 SOTA checklist

**WHAT.** Three protocol upgrades that cost days and decide citability: (a) add the
boundary-free regularizer family — MAS (Aljundi et al. 2018), Synaptic Intelligence (Zenke et al.
2017, online form), online EWC (Chaudhry et al. 2018) — since the current "beats EWC" compares
only against a *boundary-using* privileged baseline; (b) add a **single-pass streaming variant**
(each sample seen once; online accuracy + forgetting per er-ACE/OCM protocol *(knowledge)*), keeping
the multi-epoch result as a separate, clearly-labeled protocol; (c) publish the 2024–2026
exemplar-free comparison table honestly: RanPAC-style frozen-features + closed-form/prototype heads
*(knowledge: McDonnell et al. NeurIPS 2023, near-oracle exemplar-free)*, SHE, analytic-CL family
(ACIL — Zhuang et al. 2022, recursive least-squares heads, philosophically adjacent to Prizma's
local delta heads *(knowledge)*) — with a written statement of which of those results Prizma does
and does not touch.

**WHY.** This is the difference between "impressive internal table" and "citable result". Reviewers
from the CL community will check exactly these three boxes before reading §6.5. Also self-defense:
RanPAC's frozen-random-features + local-head substrate is *philosophically the same family* as
Prizma's fixed-FA + local heads — better that Prizma says this first and states its delta
(vigilance recruitment + consolidation machinery), or someone else says it later.

**HOW.** All numpy; the hardest piece is a faithful MAS/SI implementation on the existing hand-rolled
MLP (the codebase already hand-rolls backprop, so the footing stays identical). Single-pass variant
is a flag in `fit_task` semantics (batch stream without epoch loop). No GPU.

**FALSIFIABLE TEST.** Pre-registered S1–S6-style criteria against each new baseline: e.g.
`ACC_Prizma ≥ ACC_MAS − 0.02` **and** `FGT_Prizma < FGT_MAS` in the κ=0 regime (else the
"locality advantage" story weakens); single-pass Prizma keeps routing purity ≥ 0.95 at κ=0 (the
warmup=256 machinery was built for block streams; single-pass is where it may crack — honest
prediction: it will need batch-window redesign, and finding that is a result, not a failure).

**RISKS.** Results may show task-free regularizers at boundary-free parity with Prizma in some
band — again, better from inside. MAS/SI are designed for supervised networks; on a
predictive-coding substrate they need careful adaptation (that adaptation itself is a small
contribution).

**IMPACT 7 / FEASIBILITY 9.** [B] MAS/SI/online-EWC/er-ACE/OCM protocols, RanPAC/SHE/ACIL
comparisons; [N] first single-pass + boundary-free-regularizer audit of a vigilance-routed
predictive-coding CL system.

---

### P5. Prizma-LM — the thread fusion: a lifelong char-LM that keeps learning, no replay, no boundaries (flagship)

**WHAT.** The smallest falsifiable fusion of the two threads. **Backbone:** Prizma-Seq char-LM
(GDN-family mixer, quad2 rectangular state; ~5–15M params — the scale the Seq thread has actually
trained). **Continual machinery:** a small pool of *adapter experts* at the top blocks, each a
delta-rule FFN side-module trained by the Seq thread's own local kernels; **recruitment** when the
sequence-level aggregate per-token loss (precision-normalized by the running (μ, σ) machinery —
token-level vigilance is explicitly forbidden, §1.5/iteration-log v2) stays above the active
regime's floor for a window; **consolidation freeze** on the same precision phase detector.
**Data:** one concatenated char stream, four phases of *unequal* length (e.g. text8 slice →
tiny-shakespeare → simple-wiki slice → Python source), fixed order, no boundary tokens, no task
labels, no buffer. Metrics: per-corpus held-out BPC at stream end vs best-during-tenure;
forgetting_j = BPC_j(final) − min_t BPC_j(t).

**WHY.** This is the mission's biggest prize: the delta-state backbone has *constant memory*
(measured 28–455× vs KV-cache) and the CL thread owns the only tested no-replay no-boundary
acquisition mechanism; a checkpoint that keeps learning on-stream at O(1) memory is a capability
**no member of the 2024–2026 hybrid-linear family has** (they all freeze after pretraining —
report 07's "local/online learning slot is completely vacant"). It is also the honest answer to
"in what sense is Prizma a rival and not a diagnostic model?" — rivals need a lifecycle.

**HOW.** Pre-registered arms (repo style, ≥3 seeds, 95% CI): naive continued pretraining (lower
bound), Prizma-LM, replay-1% char buffer (the industrial default *(knowledge)*), O-LoRA-style
orthogonal-subspace adapters *(knowledge; Wang et al. 2023)*, LLaMA-Pro-style block expansion with
oracle boundaries *(knowledge; 2024)* as the privileged upper, and the **task-id oracle**
(adapter-per-corpus, id given — mirroring E1's oracle structure so results are comparable across
threads). Grep gate: no boundary marker in the Prizma-LM code path. Budget estimate: comparable to
the Seq thread's ~28 A100-hour clean campaign, ×3–5 arms.

**FALSIFIABLE TEST.** Pre-register: mean forgetting (BPC) of Prizma-LM ≤ 0.05 while naive ≥ 0.20
(numbers fixed only after a pilot run sets the plausible band — B4's lesson: pre-register *both*
corpora and seeds before seeing results); Prizma-LM ≥ replay-1% on mean forgetting; adapter
overhead ≤ 1.3× backbone params; and the **null control nobody runs**: measure how fast the delta
state itself forgets a prior corpus (BPC on corpus-1 text immediately after phase shift vs after
all phases) — if the state retained anything, the routing story is unnecessary and must be
re-framed; expected result: decay makes state retention negligible over 10⁷-char gaps, which is
precisely why experts are needed.

**RISKS.** (i) Char-level 10M-param models may show *weak* forgetting under naive continued
training at these corpus sizes (the task might be too easy — pilot first; if naive forgetting < 0.1
BPC, scale corpora or shrink replay arms). (ii) Vigilance on language surprise: cross-corpus
surprise differences may be smaller than the (μ, σ) floors — the P2 machinery is what makes this
calculable in advance. (iii) B4 history: pre-registration deviations are already in this repo's
public record — this proposal must ship with a quarantine policy agreed *in advance* (question for
the owner). (iv) Compute: GPU-bound; without A100 access this slips behind every CPU proposal.

**IMPACT 10 / FEASIBILITY 4.** [B] continual-pretraining protocol + replay-1% baseline
(*(knowledge)* standard practice; Ibrahim et al. 2024), O-LoRA / LLaMA-Pro / adapter-expert
lineage (BTM/Branch-Train-Merge *(knowledge)*, PHATGOOSE routing *(knowledge)*); [N] task-free,
boundary-free, replay-free expert *recruitment* for a delta-state LM — the vacant slot in report
07's landscape, and the only proposal here that addresses "inevitable rival" head-on.

---

### P6. Adapter-scale LLM continual pretraining — the bridge version of P5 (GPT-2/125M, task-free)

**WHAT.** The same recruitment/freeze machinery on a *commodity* backbone (GPT-2-scale, 124M,
public checkpoint) with LoRA-like side experts: stream three domain corpora (e.g. generic web →
biomedical abstracts → code) with no boundaries; experts recruited on corpus-level surprise
statistics; frozen on mastery; routing at inference by precision-normalized aggregate loss.
Compare against: naive continued training, replay-1%, O-LoRA, LLaMA-Pro (oracle boundaries), and
Branch-Train-Merge as the industrial forget-free-but-forked alternative *(knowledge)*.

**WHY.** If P5 is too ambitious to fund first, this is its cheaper sibling with a *much* more
credible backbone: "125M params, three domains, zero boundaries, zero replay, matches
replay-1% forgetting" is a citable LLM-continual-learning result even without touching the Seq
thread. It also derisks every design decision (windowing, floors, freeze timing) before P5 spends
A100-hours.

**HOW.** Public GPT-2 + PEFT-style adapters; vigilance uses *aggregate* next-token loss statistics
over windows (never token-level); all training is standard backprop (honesty note: this arm
**abandons** the backprop-free property to buy scale — it must be labeled as the
"Prizma-routing, standard-gradient" hybrid, not sold as local learning).

**FALSIFIABLE TEST.** Pre-register: mean normalized forgetting across domains ≤ 0.5× naive and ≤
1.2× replay-1% (replay may win — if it does, report it; the claim is then "replay-free parity"),
adapter overhead ≤ 15% of backbone, routing purity ≥ 0.9 on held-out domain probes, and a
boundary-leak grep gate as in P5.

**RISKS.** Domain shifts in real text are smooth (register/topic), so recruitment timing may smear
across corpora — the precision floor machinery plus P2's overlap analysis is the mitigation.
Reviewer objection: "this is PHATGOOSE/LoRA-expert routing with a new trigger" *(knowledge)* — the
response is the trigger: task-free, boundary-free, precision-tested recruitment is the new part,
and E1 is its existence proof.

**IMPACT 8 / FEASIBILITY 6.** [B] LoRA experts, O-LoRA, LLaMA-Pro, BTM, PHATGOOSE (all *(knowledge)*);
[N] vigilance-recruited adapter experts with precision-phase consolidation for task-free LLM CL.

---

### P7. Mixture-of-Responsibilities — partial plasticity for overlapping domains (and implementing the design's unwritten reawakening)

**WHAT.** Between "recognized → frozen/exploit" and "novel → recruit", add the middle branch the
iteration log abandoned in v1: when a batch is recognized by a committed expert but *also* sits
outside the active expert's floor (the P1 overlap band), assign soft responsibility — active expert
trains at β·window(novelty) while the recognized expert's **head only** (decoder frozen) takes
local delta updates; and implement the design doc's specified-but-unwritten reawakening rule
`ω_m ← ω_m − κ·relu(conflict)` so a frozen expert can re-enter plasticity under sustained conflict.

**WHY.** This is the mechanism that turns P1's κ\* cliff (if it exists) into a slope: today, two
overlapping domains either merge into one expert (ACC collapse) or thrash recruitment (expert
burn). The design doc already claims the mechanism; the code never got it (§1.1) — closing that
gap is both honest and cheap.

**HOW.** numpy; the v1 failure mode (§7: soft MoE collapsed to uniform, "low FGT for the wrong
reason") is the named hazard — the anti-collapse guard is that responsibilities are now
precision-tested per expert (v4 machinery) rather than globally annealed, plus the head-only
constraint (which preserves the FGT guarantee for the frozen decoder).

**FALSIFIABLE TEST.** On P1's κ ∈ (0, κ\*) band: Prizma-MoR's ACC at κ = 0.8·κ\* ≥ unmodified
Prizma + 0.05 with FGT ≤ 0.1; no expert burns (recruitments per domain ≤ 2); and the FGT=0
guarantee is *stated honestly as relaxed* in the overlap band (pre-register FGT ≤ 0.1 there, not 0).

**RISKS.** High — the iteration log records a real collapse in the naive version of this idea.
That history is an asset: the proposal is a *targeted retry* with the v4 precision machinery the
v1 attempt lacked, and its failure would itself bound the approach (report as such).

**IMPACT 6 / FEASIBILITY 6.** [B] soft responsibility / MoE gates, metaplastic reawakening (design
§3, Fusi/Benna metaplasticity); [N] precision-tested partial plasticity with head-only writes.

---

### P8. Wake-Sleep Consolidation — replay-free offline reorganization via state summaries and generative self-replay

**WHAT.** A two-phase daily cycle. **Wake:** stream learning exactly as now. **Sleep:** no raw data;
(1) each frozen expert regenerates *approximate* domain inputs from its own decoder (the expert IS
a local generative model — generative replay, Shin et al. 2017 / van de Ven et al. 2020, but
*self*-replay from consolidated modules rather than a co-trained generator); (2) inter-expert
interference is measured on synthetic batches; (3) consolidation actions run: gate re-balancing,
merge/prune statistics (P3), ω-decay downscaling (SHY-flavored synaptic homeostasis), and — in the
Seq fusion — **state re-consolidation**: brief re-encoding passes that re-write the delta state
summaries so the fast state tracks the *consolidated* expert set, not the wake-phase stream.

**WHY.** Brains consolidate offline (hippocampal replay, Wilson & McNaughton 1994; wake-sleep
algorithm, Hinton 1995; SHY, Tononi & Cirelli 2014) — this is the single largest missing organ in
the brain-like ledger, and it is the honest answer to the interleaved-stream collapse (§8): the
wake phase sees contiguous blocks; the sleep phase can reorganize *between* blocks so that the
next wake period starts re-organized. It also gives P7's soft responsibility a safe venue (risky
plasticity happens offline on synthetic data, never on the live stream).

**HOW.** Phase 1 (numpy, CPU): sleep-time merge/prune/downscale from already-computed surprise
statistics — cheap, pre-registerable. Phase 2 (the bold part): decoder-sampled pseudo-inputs for
interference measurement; honesty label: this *is* replay in the generative-replay family (borrowed
family), the new parts are (a) replay sources are the consolidated experts themselves, (b) the
replayed signal drives *gates/merges*, not weight gradient descent, so FGT=0 is structurally
preserved.

**FALSIFIABLE TEST.** Sleep phase on P1's overlap band improves ACC ≥ +0.04 without any raw-data
access (a logger proves no raw batch is read during sleep — the repo's grep/audit culture
extended to a data-access gate); SHY downscaling after a long stream recovers ≥ 5% of the accuracy
lost to expert-saturation at M_max (with P3); in the Seq fusion, sleep re-consolidation keeps
corpus-1 BPC within +0.05 of its tenure minimum after corpus-4.

**RISKS.** Decoder samples from under-trained experts are poor pseudo-data (quality drift);
generative-replay lineage makes "replay-free" marketing impossible — the claim must be narrowed to
"raw-data-free" (accurate and still valuable). The bold phase-2 may simply not help at toy scale —
pre-register it as an exploratory leg, not a gate.

**IMPACT 7 / FEASIBILITY 5.** [B] generative replay (Shin 2017; van de Ven 2020), wake-sleep
(Hinton 1995), SHY, hippocampal replay; [N] expert-as-replay-source driving *routing/consolidation*
updates (not gradients) with a data-access audit gate.

---

### P9. The Class-Incremental Honesty Probe — does vigilance add anything on frozen features + prototype heads?

**WHAT.** A deliberately adversarial internal control. Build the strongest trivial competitor in
the modern exemplar-free style: frozen random features (the FA matrices / RBF lift Prizma already
contains) + a *growing prototype/ridge head* that simply accumulates per-class local models with no
routing and no vigilance (RanPAC/SHE/ACIL family *(knowledge)*). Run it on the κ-continuum
alongside Prizma. If the trivial grower matches Prizma at κ=0, then vigilance's value is entirely
in the overlap band and open world (P1's κ > 0, P3's budget) — and the paper must say so.

**WHY.** RanPAC showed frozen random projections + closed-form heads reach near-oracle exemplar-free
class-incremental results *(knowledge)* — uncomfortably close to "what Prizma would be without its
routing". Establishing the delta *internally*, with pre-registered bars, is the difference between
Prizma's routing being a mechanism or an ornament. This is repo-culture adversarial auditing turned
on the CL thread itself.

**HOW.** numpy at toy scale (prototype heads are closed-form); optional torch variant on Split-CIFAR
if the owner wants a standard benchmark citation. Cheap.

**FALSIFIABLE TEST.** Pre-register: at κ=0, prototype-grower ACC ≥ Prizma − 0.02 (expectation: it
may match — the task is separable); for κ ≥ 0.8·κ\*, Prizma ≥ grower + 0.05 (the routing earns its
keep exactly where domains overlap and a shared head must arbitrate); in open-world streams with
M_max budget (P3), Prizma ≥ grower + 0.05 (allocation beats accumulation under scarcity). Any leg
that fails is reported as a boundary of the mechanism, per repo style.

**RISKS.** The honest risk is the expected one: the grower wins at κ=0. The value of the proposal is
precisely that it *demands* the overlap and open-world legs pass for the routing claim to survive —
turning a possible embarrassment into the experiment that localizes Prizma's true contribution.

**IMPACT 6 / FEASIBILITY 6.** [B] RanPAC/SHE/ACIL-style prototype heads *(knowledge)*; [N] the
κ-localization audit of vigilance value against its own trivial ablation.

---

## 3. Ranked summary (by IMPACT × FEASIBILITY)

| Rank | Proposal | Impact | Feasibility | EV | Compute | Thread |
|---|---|---|---|---|---|---|
| 1 | P1 Regime Map + buffer-matched SOTA | 9 | 9 | 81 | CPU, days | CL |
| 2 | P4 Citation bar (MAS/SI/online-EWC, single-pass, 2024–26 table) | 7 | 9 | 63 | CPU, days | CL |
| 3 | P3 Expert economy (prune/merge/budget + open-world abstain) | 7 | 8 | 56 | CPU, days | CL |
| 4 | P2 Chernoff routing law (theory spine) | 9 | 6 | 54 | CPU, weeks | CL |
| 5 | P6 Adapter-scale LLM CL (125M, task-free) | 8 | 6 | 48 | 1× GPU | bridge |
| 6 | P5 Prizma-LM flagship fusion (char-LM lifelong) | 10 | 4 | 40 | multi-A100 | Seq+CL |
| 7 | P7 Mixture-of-Responsibilities + reawakening | 6 | 6 | 36 | CPU | CL |
| 8 | P9 Class-incremental honesty probe (prototype-grower control) | 6 | 6 | 36 | CPU, days | CL |
| 9 | P8 Wake-sleep consolidation (self-replay, SHY) | 7 | 5 | 35 | CPU→GPU | CL+brain |

Sequencing logic: P1 → P2 → P9 form one paper (map + law + control: "when does surprise-routing
continual learning work?"); P4 lands before any of it is submitted; P3 makes every other claim
bounded; P6 de-risks P5; P5 is the flagship the "inevitable rival" story needs; P8 is the
brain-likeness capstone once P1–P3 give it real phenomena to act on.

---

## 4. The named citation bar (what must be on the comparison table, per regime)

- **Domain-incremental / distinguishable (Prizma's home):** naive SGD, EWC (boundary, as now),
  online EWC (Chaudhry 2018), SI (Zenke 2017), MAS (Aljundi 2018), ER / DER++ / MIR / GDumb at
  buffer budgets *matched to Prizma's weight growth*, SHE *(knowledge)*, expert-expansion
  lineage (PNN — Rusu 2016; DEN — Yoon 2017; Expert Gate — Aljundi 2017, the closest historical
  ancestor of recon-based expert routing — must be cited in the related-work ledger).
- **Overlap band (new, P1):** same battery vs κ; plus P9's prototype-grower control.
- **Single-pass online (new, P4):** er-ACE / OCM-style protocol *(knowledge)*.
- **LLM continual pretraining (P5/P6):** replay-1%, O-LoRA, LLaMA-Pro, Branch-Train-Merge,
  PHATGOOSE *(knowledge for all 2024 items)*; metric = normalized forgetting + adaptation parity,
  never BPC alone.
- **Standing bar for any "beats X" sentence:** ≥10 seeds, non-overlapping 95% CI, budget-matched,
  boundary-leak grep gate, pre-registered criteria — the repo's existing S1–S6 discipline,
  extended, never relaxed.

---

## 5. Brain-likeness ledger from the CL lens (claimed vs implemented)

| Brain organ/property | Design slot (docs/Prizma.md) | Code status | Proposal |
|---|---|---|---|
| Fast acquisition / slow consolidation (CLS) | fast experts ↔ ω-consolidation | partial (freeze, no reawakening) | P7 |
| Novelty/vigilance (hippocampal mismatch) | ART vigilance + precision test | implemented (train only; no test-time abstain) | P3 (abstain) |
| Systems consolidation during sleep | broadcast/efference copy only | absent | P8 |
| Hippocampal replay | explicitly rejected as requirement | absent (by design) | P8 (raw-data-free variant) |
| Synaptic homeostasis (SHY downscaling) | not in doc | absent | P3 + P8 |
| Metaplastic reawakening (cascade synapses; Fusi 2005; Benna & Fusi 2016) | §3 rule specified | **not implemented** | P7 |
| Neuromodulation (NM gating plasticity) | §2.4 NM scalar | absent (constant lr) | (lens 04's territory; P7 touches) |
| Multiple timescales in one substrate | settling τ_z / gates / ω / eligibility τ_e | design only (prototype has lr + ω) | P5 (state-as-eligibility-trace) |
| No weight transport (local learning) | DFA | **implemented and measured** (rare!) | keep as headline property |

The rarest brain-like property Prizma already owns is the last row: a CL system whose learning is
fully local and transport-free, with evidence that removing weight transport *helps* (E4/E1:
DFA 0.834 > exact 0.708). No EWC/SI/MAS/DER++/RanPAC variant has any analog of this. It deserves
top billing in every future writeup, second only to the routing mechanism itself.

---

## 6. Cross-committee notes

- **To report 06 (sparse MoE):** P3's merge/prune is your load-balancing story at the *stream*
  timescale; shared machinery, please co-design the usage statistics.
- **To report 07 (Seq landscape):** P5/P6 are your "vacant local/online learning slot" made
  concrete; the state-as-eligibility-trace claim inside P5 needs your BPTT-vs-local benchmark.
- **To report 04 (neuromodulation):** P7's reawakening rule and the missing NM scalar are one
  mechanism; the v1-collapse history suggests the conflict signal, not the gate, is the hard part.
- **To reports 05 (free energy):** P2's Chernoff framing is the same surprise signal read as a
  sequential test — please check the consistency of the two derivations.

---

## QUESTIONS FOR THE OWNER

1. **Claim repositioning:** §8 already says FGT=0 is a quasi-tautology. Do you authorize demoting
   the README headline to "task-identity inference without labels or boundaries (FGT=0 as
   corollary)" — the framing P1–P4 and the literature require? (If not, the citation bar in §4
   cannot be met honestly.)
2. **Compute sequencing:** P1–P4 are CPU-only and weeks of laptop time; P5–P6 need A100 budget
   (P6 ~tens of GPU-hours; P5 ~100+). Do we sequence CPU-first this quarter, or is GPU budget
   available now for the P6 bridge?
3. **Benchmark diet:** stay synthetic/numpy until the Regime Map is done (control, cheap, but not
   citable to CL-community benchmarks), or adopt one standard benchmark (Split-CIFAR/miniImageNet)
   early even though it dilutes the neuromorphic narrative?
4. **Pre-registration failure policy:** given the B4 history, what is the standing rule when a
   pre-registered leg fails — quarantine + disclose (as done), or re-register with justification?
   P5's bars should be set under whichever rule you pick, before the pilot runs.
5. **Memory bound:** is unbounded expert growth acceptable while claims stay "bounded-domain
   streams", or must P3's M_max budget ship before any open-world sentence appears in docs?
6. **Priority under scarcity:** if only one flagship is funded this year — P5 (Prizma-LM, the
   "inevitable rival" capability) or P8 (wake-sleep, the largest missing brain organ) — which
   mission objective wins?
