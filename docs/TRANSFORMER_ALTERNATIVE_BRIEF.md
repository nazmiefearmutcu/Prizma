# PRISM → Transformer-Alternative — Committee Brief (self-contained)

**Read this fully before doing your task. You share NO conversation context; everything you need is here.**

## 0. The mission (from the project owner)
Take the existing PRISM project (at `/Users/nazmi/Desktop/Projeler/proje/PRISM`) and push it to a
**realistic point where it is a genuine alternative to the Transformer architecture**. The owner's
explicit success floor: *"PRISM can stand in for the Transformer."* A rigorous committee loops,
improving the method and the report, and **stops only when the committee agrees PRISM is a credible
competitor to the Transformer architecture.** Hard rule from the owner: **no faked metrics, no
hand-waving** — but also **do not treat "can't be done" as the endpoint; engineer a path.**

## 1. What PRISM currently IS (accurate, no spin)
PRISM today is a **backprop-free, fully-local, predictive-coding continual-learning method** on a
**shallow MLP substrate**, validated on a **synthetic static-classification** benchmark.

- **Core mechanism:** a mixture of predictive-coding auto-encoder *experts*. Routing = ART-style
  vigilance on each expert's reconstruction *surprise* (label-free). Consolidation = "Precision-
  Gated Metaplasticity" (PGM): a mastered expert produces low surprise → low bid → loses the
  competition → freezes. No task labels, no task boundaries, no replay.
- **Learning is local:** decoder/head use exact PC/delta rules; encoder uses **fixed-random
  feedback (Feedback Alignment / DFA)** so **no weight transport (W^T)** anywhere.
- **Headline result (E1, structured-permuted, 10 seeds, 95% CI):** PRISM(DFA) ACC **0.834**,
  **FGT (forgetting) = 0.000**; backprop MLP 0.445 / FGT 0.553; EWC 0.456 / FGT 0.411; replay 0.737
  / FGT 0.156; oracle-multihead (task-id GIVEN) 0.879 / FGT 0. PRISM matches the oracle's
  zero-forgetting *without being told the task id*, and DFA (no W^T) **beats** exact-W^T.
- **Honestly characterized limits (already documented):** NOT a scaling claim, NO backprop parity
  shown; works only in *input-distinguishable domain-incremental* streams with *temporally
  contiguous* domains; fully-ambiguous regime proven impossible; P1 (scaling) & P2 (weight
  transport) explicitly unsolved.

**Files:** `src/prism.py` (the method), `src/baselines.py` (MLP+EWC, hand-coded backprop),
`src/data.py` (synthetic benchmarks), `src/metrics.py`, `experiments/run_continual.py`,
`docs/PRISM.md` (full writeup), `committee/reports.json` (the 6 design reports it was built from).
Read any of these as needed.

## 2. The brutal gap (why current PRISM is NOT a Transformer alternative)
A Transformer's defining capability is **sequence modeling via attention**: content-based,
data-dependent mixing across a sequence → in-context learning, associative recall, language
modeling. Current PRISM does **static classification with continual-learning routing**. It has:
no notion of a sequence, no position, no causal mixing, no in-context recall, no autoregression.
"Beating EWC at zero-forgetting on synthetic Gaussians" is **orthogonal** to "replacing attention."

## 3. The strategic reframing the committee should adopt (and pressure-test)
"Transformer alternative" in the literature means an **architecture that replaces the attention
*mechanism*** (how tokens mix), trainable by the standard recipe — **not** a new learning rule.
Mamba, S4/S5, RWKV, RetNet, linear-attention, Hyena, Based are all **backprop-trained** sequence
architectures that replace O(n²) attention with cheaper input-dependent mixing.

**PRISM's genuinely novel core IS a candidate attention-replacement:** the **cortical-workspace
broadcast** (a small shared latent `a ∈ R^k`, `k≪n`, every module reads/writes it → **O(n·k)
linear** token mixing) + **precision-weighted recognition-routing** (input-dependent, content-based
gating of who writes to the workspace). This is structurally a *linear-cost, content-based,
input-dependent mixing operator* — exactly the class attention belongs to.

**Therefore the realistic, honest, ambitious target is `PRISM-Seq`:**
> A predictive-coding cortical-workspace **sequence architecture** that replaces self-attention
> with **precision-routed, input-dependent workspace mixing at O(n·k) / O(1)-per-step linear cost**.
> Primary axis = **architecture, backprop-trainable**, parameter-matched against a Transformer.
> Differentiators Transformers lack = (a) linear time & constant-memory autoregressive inference,
> (b) an optional **local / backprop-free training mode** with a *quantified* accuracy tax, (c)
> **task-free continual learning** (inherited from current PRISM). Large-scale LM parity = stated
> open frontier.

This is the only path that is simultaneously honest (it's what "alternative" actually means and is
empirically testable here) and achievable. **Backprop-free + Transformer-parity at scale is an
unsolved frontier problem; do NOT make it the gating claim** — make the architecture the gating
claim, and carry local-learning as a characterized bonus.

## 4. The falsifiable bar (the committee must agree on the exact pass/fail BEFORE building)
A research prototype counts as a *credible Transformer alternative in the tested regime* if, at
**small scale and parameter-matched against a standard Transformer**, it clears the field's
**attention-diagnostic suite** + a small LM, AND offers a structural advantage. Proposed concrete
gate (refine it, keep it field-standard and non-strawman):

1. **Associative recall / MQAR** (multi-query associative recall) — *the* test that separates real
   attention-alternatives from fakes. PRISM-Seq ≥ Transformer (within noise) at matched params.
2. **Induction** (in-context copy of `… [A][B] … [A] → [B]`) — the ICL primitive. Must solve it.
3. **Selective copying** — input-dependent gating / content-selective memory.
4. **Char/byte-level LM** (e.g. Shakespeare / enwik8-subset / text8-subset): bits-per-char or
   perplexity **competitive** with a parameter-matched Transformer (define "competitive": within a
   small explicit margin, or better).
5. **Structural advantage demonstrated & measured:** O(n) (or O(1)/step) inference cost vs the
   Transformer's O(n²)/O(n)-cache — show the actual latency/memory curve.
6. **Honesty controls:** parameter & FLOP matching audited; no test leakage; ≥3 seeds with CIs;
   ablations showing the PRISM mechanism (not just "a bigger RNN") is causal; explicit statement of
   where it still loses and what scaling is unproven.

The committee may **add** the local-learning bonus axis (PRISM-Seq trained with local/PC/DFA rules
vs backprop, quantifying the tax) and the **continual-learning** axis (the unique selling point).

## 5. Compute reality (what experiments are feasible — design within this)
- Apple Silicon Mac, **torch 2.12.0 with MPS available**, Python 3.13, numpy 2.2.6, **10 CPU, 16 GB
  RAM**, single GPU (MPS). No CUDA, no cluster.
- Feasible: small models (d_model ~64–256, 2–6 layers, seq len ~64–1024, vocab small/byte),
  synthetic tasks generated on the fly, char-LM on a few-MB corpus, a few thousand steps each →
  **minutes per run**, not hours. Keep every experiment in the minutes regime; serialize GPU jobs.
- Use PyTorch for PRISM-Seq + the Transformer baseline so the comparison is on identical footing and
  fast on MPS. (The current numpy PC code is the *conceptual* ancestor, not the perf substrate.)

## 6. Non-negotiables (owner's integrity rules)
- **Real, reproducible numbers only.** Every claim backed by a runnable script + seeds + CIs.
- Parameter/FLOP-matched, non-strawman Transformer baseline (proper implementation, tuned LR).
- Keep PRISM's honest-limits culture: a "borrowed vs new" ledger; explicit failure modes; scaling
  stated as open.
- The differentiator must be *real* (linear cost / continual learning / local learning), measured,
  not asserted.

## 7. Your job depends on which agent you are — see your specific prompt.
Anchor everything to: **does this move PRISM-Seq toward clearing the Section-4 bar, honestly?**
