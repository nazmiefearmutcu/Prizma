# Council 2 — The Quant Team (alpha hunt)

**Role:** research *how to dramatically surpass the Transformer* from the current Prizma-Seq base.
Think like a quant team hunting edges: high standards, expected-value ranking, adversarial
self-filtering, grounded in real 2024–2026 literature with citations. Feeds the build queue.

## Members (lenses)
1. **Quant researcher** — expected-margin-gain ÷ honesty-risk ranking; falsifiable mini-experiments.
2. **Architecture designer** — precise mechanisms, O(1)/param-match compatibility.
3. **Kernel / perf specialist** — FLOP impact, chunk-parallel/Triton feasibility.
4. **Literature scout** — frontier papers (Gated DeltaNet(-2), DeltaProduct, RWKV-7, Titans, Based,
   Mamba-2, GLA, gated attention, hybrids), with arXiv ids + one-line takeaways.
5. **Risk assessor** — the strongest reason each top lever might be a mirage + the cheapest exposer.

## Output contract
A ranked lever list (`committee/quant/round{N}_levers.md`): per lever = mechanism · attacked axis ·
expected gain · honesty-risk · O(1)-compatibility · FLOP impact · difficulty · validating experiment.
Loop-until-dry ideation across rounds; adversarially self-filtered.

## Round 1 result (2026-06-07) — feeds the plan
- **Win-triad C + D + E** flips all three losing inference axes (char-LM, FLOP, latency) at once.
- New high-EV levers added to the queue:
  - **H — decoupled channel-wise erase/write** (Gated-DeltaNet-2, 2026): split β into key-side erase +
    value-side write. Param-matched, ~5% train cost, strong recall+LM gain — **promoted ahead of A as
    the "free" recall+LM win.**
  - **G — in-context per-channel learning rate** (RWKV-7): generalizes scalar β; may be a stronger
    *novel core* than A and subsumes gating+surprise.
  - **I — stochastic window size in training**: ~free, helps short- AND long-context; bolt onto E.
- **Sequencing:** F (first/parallel, makes iteration cheap) → C+D+E (parallel) → H → B → {A or G}.
- **Sharpest caveat:** lever D could silently kill the MQAR ≥3.5× param-efficiency edge → run the
  **MQAR D=128 solve-threshold at the reduced d_φ FIRST**, before any LM run; back off if the solve
  point regresses past 130K params.
