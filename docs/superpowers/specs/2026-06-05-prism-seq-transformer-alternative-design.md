# PRISM-Seq → Transformer Alternative — Design Spec (2026-06-05)

> Goal (owner, approved scope = **"bar + aggressive frontier"**): develop PRISM-Seq at full speed
> until it is a **credible Transformer alternative in the tested regime** per the project's own
> pre-registered falsifiable bar (`docs/TRANSFORMER_ALTERNATIVE_BRIEF.md` §4), then push the
> aggressive frontier. Data-driven: run at matched params first; grow the architecture only if a
> leg fails. No faked metrics; param/FLOP-matched non-strawman TF; preserve the measured O(1) edge.

## 0. Where we already are (done, referee-gated — `results/gpu_bench.json`)
- **MQAR D=128 ✓** (1 of the 4 diagnostic legs): at matched params PRISM-quad2 ≈ tuned TF
  (descriptive parity, 3 seeds), ~3.5× param-efficiency (solves @130K where matched TF fails),
  constant-MEMORY inference edge (28× @n=4096, analytic), causal (quad2 ≫ rand_linear ≈ none ≫ TF).
- Honesty fixes applied (memory=analytic not measured; n=3 descriptive-parity; coarse-grid efficiency;
  candidate-not-alternative; model-init caveat). Adversarial referee panel passed as-scoped.

## 1. Success criteria — the §4 bar (pre-registered pass/fail; do NOT inflate)
A credible alternative *in the tested regime* must, at **small scale, param-matched vs a tuned TF**:
1. **MQAR** — ✓ DONE (parity @ D=128).
2. **Induction** (in-context `…[A][B]…[A]→[B]`): PRISM-quad2 **≥0.98**; holds across prefix length
   64→256 (≥0.95 at the long gap). ≥3 seeds, solve-rate + median.
3. **Selective copying** (Mamba's content-selective copy through filler): selective variant **≥0.97**
   and ≥ TF−0.02; the `fixed`-position control must be ~equal for both (isolates content-selectivity).
4. **Char-LM** (tiny-shakespeare; + text8-subset for the frontier): test **BPC ≤ TF + 0.05** (explicit
   margin), param-matched, ≥2 seeds.
5. **Structural advantage** — ✓ memory; **strengthen** (see Phase C): show the O(1) *latency* crossover,
   not just memory; length-extrapolation.
6. **Honesty controls** — param/FLOP-matched audited; ≥3 seeds + CIs (or seed ranges at n=3); causal
   ablation that the PRISM mechanism (not "a bigger RNN") is responsible; explicit failure/limits.

**Verdict rule:** PRISM-Seq earns "credible alternative in tested regime" iff legs 1–4 PASS at matched
params with the structural advantage (5) and honesty controls (6). Any leg that fails at matched params
triggers Phase D (grow architecture) before any claim; if a leg cannot be cleared even with levers
#2/#3, the claim is rescoped to what passed (never hidden).

## 2. Phased plan
- **Phase A — finish the diagnostic suite (the gap).** GPU runner `gpu_diag.py` (analogous to the
  verified `gpu_bench.py`): Induction + SelectiveCopy, PRISM-quad2 vs PRISM-none(ablation) vs tuned TF,
  param-matched, gen-warm + per-model plateau, ≥3 seeds, streamed crash-safe JSON. Tasks already exist
  in `seq/tasks.py` (`Induction`, `SelectiveCopy`).
- **Phase B — char-LM leg.** GPU runner `gpu_charlm.py` (wraps `seq/charlm.py`): tiny-shakespeare
  char-LM, param-matched PRISM-quad2 vs TF, report test BPC + the margin, ≥2 seeds. Frontier: text8
  subset, deeper/wider, longer training.
- **Phase C — strengthen structural advantage.** (1) `flop_match_width.py` d208 control (already in
  gist, unrun); (2) **latency-crossover demo** — extend P5 to n∈{4k,8k,16k,32k} and/or a bigger model
  so the TF's O(t) per-step term dominates and PRISM's O(1)-per-step yields a real wall-clock win
  (currently only memory wins at n≤4096); (3) length-extrapolation (train len L, eval 2–4×L) on MQAR/
  induction — a known SSM/attention differentiator.
- **Phase D — architecture iteration (ONLY if a leg fails).** Build committee lever **#2 decoupled
  `d_state`** (per-head state dim decoupled from d_model//H; `delta.py` already value-dim-aware) and/or
  **#3 GlobalDeltaMemory** (one asymmetric high-key-rank delta head). Grow the TF in lockstep to stay
  param-matched. Re-run the failed leg → repeat until cleared (the `/goal` loop).
- **Phase E — final adversarial referee gate + report.** Update `docs/PRISM_SEQ_REPORT.md` (B1–B6
  table → real numbers), run the parallel adversarial referee panel, fix findings, re-approve.
- **Frontier (after the bar passes):** bigger multi-layer char-LM (text8, longer); throughput/wall-clock
  head-to-head ledger; optional backprop-free (DFA) mode with a *quantified* accuracy tax (bonus axis,
  never a gate — backprop-free parity is an explicit OPEN FRONTIER, not claimed).

## 3. Execution model
- **Code-building** (GPU runners, levers #2/#3, charlm wrapper): `/subagent-driven-development` —
  fresh implementer subagent per task + 2-stage review (spec compliance, then code quality); TDD where
  it fits; keep `step()==forward()<1e-6` O(1) guard green before any accuracy number.
- **GPU runs:** Colab A100 via Claude-for-Chrome (secret-gist bootstrap, the verified pattern). Stream
  crash-safe JSON to Drive; resumable. PRISM ~30–57 min/run on A100 (sequential chunked-delta) → use 3
  seeds for diagnostics, fewer for the frontier; disclose seed counts.
- **Local (Mac MPS):** kernel self-tests, FLOP ledger, smoke tests, parity checks before each GPU run.

## 4. Honesty guardrails (binding, from committee + the referee panel)
Param/FLOP-match every head-to-head (grow TF for any paid PRISM arm); measured FLOP+memory ledger
(memory = analytic float counts, label it so; latency = measured wall-clock); ≥3 seeds + solve-rate +
median + ranges (n=3 underpowered → descriptive, not "indistinguishable"); causal ablation
(quad2 vs none vs rand_linear) per new leg; never "attention fails" (say "did not learn under recipe R
at this scale"); pre-register thresholds before multi-seed; verify BOTH sides of a contrast; commit the
raw results JSON to `results/` (auditable); `quad2` novelty = rectangular-delta-state framing, not the
kernel; large-scale LM + backprop-free parity NOT claimed (open frontiers).

## 5. Non-goals (this campaign)
- Large-scale LM parity (>~few-M params); production training; new datasets beyond shakespeare/text8.
- A new attention PRIMITIVE (PRISM-Seq is a synthesis + the quad2 lever + framing).
- Backprop-free parity as a gate (only an optional, quantified bonus axis if time permits).
