# Council 3 — The Standards & Landscape Council (high bar)

**Role:** judge whether a claimed advance is real against the actual state of the field, and set the
standard for what "dramatically surpassed the Transformer" must mean. Prevents strawman victories.
High standards; evidence-based with citations. Judges at every phase boundary and may **raise** the bar.

## Members (lenses)
1. **Landscape analyst** — where the field is (pure SSM/linear-attn vs hybrids vs test-time-memory).
2. **Benchmark authority** — which axes a win is meaningful on; real numbers for the deltas.
3. **Senior skeptic jury** — the scope rider; the honest word ("dominant" vs "competitive" vs "efficient").

## The three meta-questions (answered each round)
1. Has the Transformer been surpassed — where, by what, on which axes; where does attention still win?
2. By how much (numerically)?
3. Industry direction — what does a credible 2026 "Transformer alternative" look like?

## Round 1 result (2026-06-07) — raises the bar
- **Verdict on the field:** the TF is *matched* (not surpassed) at small/medium LM scale; alternatives
  win decisively on **decode throughput (~4–5×), KV/memory (~10×), long-context cost (~4× context/VRAM)**;
  attention **provably dominates recall/copying** ("Repeat After Me", arXiv:2402.01032). At frontier
  scale every competitive "alternative" is a **hybrid that keeps ~7–8% attention** (Jamba, Nemotron-H).
  Note: the TF is ~1.8× *faster at short seq* — the latency win is length-dependent.
- **Bar additions (binding on the §3 target):**
  1. **Recall is a hard pass/fail GATE** — MQAR(hard rung) + induction + selective-copy must reach
     **≥ tuned-TF parity via TOST**, hard rung protected by the optimization-vs-capacity flip-test.
     If recall is only "not much worse", the honest word is **Pareto-competitive**, not dominant.
  2. **per-FLOP ≤1.0× is currently UNMET (2.1–2.7×)** → "dramatic" is conditional until D/F deliver it;
     all axes must hold *simultaneously and powered* or downgrade to "Pareto-efficient in the tested regime".
  3. **Scope rider mandatory, in the same breath:** "≤2M params (+1 confirmation 10–50M), char-LM +
     diagnostics — NOT a frontier / MMLU / NL-long-context claim."
  4. **Add a matched tiny-HYBRID baseline arm** (Samba / GatedDeltaNet-H style). Since hybrids dominate
     both pure-attention and pure-linear, Prizma must be at least Pareto-competitive with a matched tiny
     hybrid — else the honest framing is "best *pure*-O(1) point", not "beats the Transformer".
- **Net:** §3 is honest and well-constructed; if fully met as written **plus** the recall-gate,
  iso-FLOP, scope-rider, and tiny-hybrid additions, it supports a credible *scoped* "dramatically
  Pareto-dominant in the tested regime" claim.
