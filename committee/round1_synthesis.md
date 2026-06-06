# PRISM-Seq Committee — Round 1 Synthesis (durable record)

> Produced by the round-1 committee (fairness referee · recall-capacity theorist · 3 architecture
> proposers · chief architect). This is the authoritative summary that drives the aggressive
> D=128-parity loop. Owner goal (locked): **full parity with a tuned Transformer even on the
> hardest MQAR rung (D=128)**, big architectural changes allowed, multi-session compute; **no faked
> metrics**, param/FLOP-matched non-strawman baseline, **preserve the measured O(1) inference
> differentiator**.

## Verdict
The harness is *structurally* honest (byte-identical FFN/norm/embed/head, ~1.01% param match,
correct disjoint-range dense-query MQAR, symmetric best-LR selection, a genuinely measured O(1)
path) but its CURRENT settings were **unfair to the Transformer**, which fully explains the
provisional "PRISM wins rung2/3" headline. And the pre-registered recall ceiling is real:
`H·d_h = 64` cannot hold 128 near-orthogonal bindings, so **D=128 parity requires architectural
change**.

## A. Fairness fixes (implemented in `seq/common.py`; verified)
Three compounding unfairnesses, all now fixed:
1. **Undertraining.** Fixed 6–8k step budgets sat near PRISM's fast convergence; the Transformer was
   still climbing (TF=0.095 @7k on rung2) — yet `results/validate_disjoint.txt` shows a matched TF
   hits **1.000 by step 3000** given a good LR. → **Relative per-model plateau early-stop** (large
   per-rung cap + stop when best-acc gain < `plateau_delta=0.003` over `patience=5` evals, `min_steps=4000`),
   applied identically to both. `eval_every=500` for resolution.
2. **LR grid asymmetry.** `{1e-3,2e-3}`: 2e-3 collapses the TF to 0.149 on rung1 while PRISM is robust
   at both → "best-of-2" gave PRISM a 2-shot draw, the TF a 1-shot. → **Wider shared grid
   `{5e-4,1e-3,1.5e-3,2e-3,3e-3}`, two-stage** (full grid @1 seed → best LR → ≥5 seeds), reject
   collapsing LRs.
3. **Absolute early-stop (0.995).** Let the fast model stop, truncated the slow one. → replaced by the
   relative plateau rule above.

Also: **generous absolute warmup (600 / 0.15)**; **frozen reproducible eval set** (`eval_seed`,
≥32 batches, identical across models/LRs/seeds) selecting best on the same held-out batches;
**≥5 seeds** on decisive MQAR rungs with **solve-rate + median + CI**, reporting the TF's bimodality
as a documented failure mode; a **first-class D=128 rung4** + a **gapped (recall-over-distance) rung**;
a **measured param+FLOP+throughput ledger**; and **per-experiment JSON checkpointing** (crash-safe).
Both models are bimodal/LR-fragile on MQAR at this scale (`stability_test.txt`: TF 2/5, PRISM 5/5 on
one config; the reverse on another) — the protocol must be symmetric and seed-robust.

## B. Recall-capacity theory (numerically verified probe)
Delta-state `S = Σ uᵢ kᵢᵀ` (‖k‖=1) is online ridgeless least-squares; read crosstalk variance
≈ `(D−1)/d_h`. Hard-recall (the MQAR decision) ≥0.97 at **D\* ≈ 0.5·d_h** (random keys) → **~1.0·d_h**
(orthogonal/learned keys). Capacity is **linear in d_h**, rank-ceiling `d_h`.
- (d_h=32,H=2): **D\* ≈ 24–32** (matches pre-registration).
- (d_h=64,H=2): **D\* ≈ 48–64** (still short of 128).
- **D=128 from fixed state needs effective rank ≥128** → ~d_state 128–256 OR a key-rank trick.
- Transformer = n-growing lossless KV cache → unbounded recall; D=128 trivial. The structural trade:
  PRISM compresses D bindings into a fixed `d_h²` lossy codebook in exchange for O(1)/step inference.

## C. Ranked improvements (toward D=128, honesty-weighted)
| # | Lever | Param cost | O(1)? | Status |
|---|---|---|---|---|
| **1** | **Quadratic feature map** φ: d_h=32→d_φ=128, rectangular state S∈R^{d_h×d_φ} | **0** (buffers) | **yes** | **IMPLEMENTED + verified** |
| 2 | Decoupled `d_state` (widen per-head state, e.g. 64→cap 128) | +32% (grow TF too) | yes | designed |
| 3 | GlobalDeltaMemory: 1 asymmetric high-key-rank (d_k=128) delta head | +18% (grow TF too) | yes | designed |
| 4 | Bounded global-attention / global-token hybrid | match | **partially** | reserved (explicit O(1)-trade) |

**first_build = #1.** Best capacity-gain-per-honesty-risk: zero trainable params (byte-identical
match preserved, no TF re-grow), O(1) intact (state fixed-size in n; constant grows d_h²=1024 →
d_h·d_φ=4096 floats/head — disclosed in B5), default-off (no regression), lowest correctness risk
(kernel verified dimension-agnostic; rectangular self-test added to `delta.py`). Crosstalk evidence
is decisive: D=128 off-diagonal 0.141 → **0.076** ≈ true-d=128 oracle (0.071); a random *linear*
32→128 projection gives no gain (0.153) — proving the gain is the *quadratic monomials*.
**Capacity is necessary, not sufficient — the end-to-end recall lift MUST be measured** (`cap_probe.py`,
then `run_bar` B1b). Next levers staged: #2 as the param-honest "grow-both" Pareto arm, #3 as the
cheaper-than-rank-doubling fallback, #4 only if pure-O(1) plateaus. **Honest end-state may be a
Pareto pair** — a pure-O(1) variant AND, if needed, a bounded-hybrid that matches D=128 at a stated,
measured inference cost — not a single model.

## D. Honesty guardrails (binding on every reported number)
1. **Param match** every head-to-head; any paid arm (#2/#3/#4) grows the Transformer in lockstep — never 130K-PRISM vs 102K-TF.
2. **Measured FLOP + throughput ledger** (PRISM carries an extra window head + conv at equal params; disclose direction+magnitude of any >10% gap; PRISM is ~3–5× slower wall-clock — disclose).
3. **O(1) disclosure**: report the per-model decode-state size in B5; state it is still O(1) in n and ≪ the TF's O(t) cache. A shipped hybrid's O(n) layer is a SEPARATE labeled B5 curve.
4. **O(1) equality guard**: streaming `step()` == parallel `forward()` to <1e-4 for every mechanism BEFORE any accuracy number (hard blocker; the `prism_seq.py` __main__ assert).
5. **Causal ablation** under the same two-stage LR sweep: `quad2` vs `none` + a `random-linear-32→128` control (must show no gain) to prove the quadratic monomials are the cause.
6. **No best-of-noisy-curve**: select on the frozen eval set; report frozen best AND end-of-plateau acc.
7. **Report failure modes**: TF bimodality explicit; solve-rate + median + CI over ≥5 seeds; **pre-register D\*** for every new arm before its multi-seed run; if a lever lands below 0.97 at D=128, rescope to the achieved D\* — never claim unmeasured parity.
8. **Both sides of the contrast**: verify the matched TF actually reaches near-ceiling at D=128 under the fair protocol before claiming "PRISM degrades, TF doesn't".
9. **Seed-frozen architecture choices**: the φ monomial indices are seeded (1234) and disclosed — never tuned per task.
10. **Crash-safe audit trail**: per-experiment atomic JSON checkpoints incl. steps-to-plateau, full stage-1 LR sweep + rejected LRs, warmup/plateau/eval_seed, and the param+FLOP ledger.

## Implementation status (this session)
- `seq/common.py`: plateau early-stop + generous warmup + frozen eval set + `steps_to_plateau` — **done, verified** (smoke OK).
- `seq/delta.py`: value-dim-aware init (rectangular state) + rectangular self-test — **done, verified** (ALL OK, CPU+MPS).
- `seq/prism_seq.py`: `feat_map='quad2'` (config, fixed monomial buffers, `_phi`, rectangular state in forward/step/init_state) — **done, verified**: feat_map=quad2 → d_φ=128, params 102,728 (unchanged), step==forward 3.6e-7.
- `cap_probe.py`: decisive D∈{64,128} capacity validation under the fair protocol — **running**.
- TODO: `run_bar.py` full protocol patch (two-stage LR, rung4 D=128, gapped rung, checkpointing, FLOP ledger, B5/B6 fixes) → full matrix; then adversarial referee panel.
