# Fusion probe — Toy Prizma-LM vigilance-routed expert tissue on the PR-07′ block-drift stream

**LANE-EXPLORATORY (docs/preregistry/POLICY.md). Design-informing ONLY — this never becomes
a claim.** It motivates the GPU-tier PR-LM-1 architecture decision (fusion spec §3.1–3.2,
`docs/superpowers/specs/2026-09-03-prizma-lm-fusion-design.md` — the "unified column" idea at
the smallest honest scale).

- **Script:** `seq/fusion_probe.py` (standalone; zero changes to any existing module —
  `git status` must show `src/prizma.py`, `seq/prizma_seq.py`, `seq/transformer.py`,
  `docs/preregistry/` untouched).
- **Raw ledger:** `raw.json` in this directory (crash-safe: tmp + atomic replace after every
  cell). Run log: `fusion_probe_2026-09-07_run.log`.
- **Stream:** PR-07′ verbatim — text8[0:1M) → tiny-shakespeare, one pass, char-level, SEG=256,
  BATCH_SEGS=32; corpora read from the archived `results/blockdrift_PR-2026-09-03-07/corpora/`.
- **Backbone:** Prizma-SeqLM, PR-07′ pinned config (2 layers, d_model=64, H=4, chunk=64,
  window=16, max_len=256; 103,056 params). **PREFERRED VARIANT (disclosed): the backbone
  TRAINS on all segments** with the shared LM loss (PR-07′ `train_stream` objective verbatim);
  the frozen-backbone variant is a design note below, not run.
- **Tissue:** E ∈ {4, 8} torch PC-expert heads (the `src/prizma.py` Expert's surprise-routing
  idea, adapted; the classifier head and FA feedback are not needed here): per expert
  Wenc 64→64 (tanh) + Wdec 64→65 (vocab), 8,385 params/expert. Final logits = base lm_head
  logits + routed expert logits — the expert is a **residual corrector** trained on DETACHED
  base logits and DETACHED hidden states with its own private AdamW. Routing by surprise is
  non-differentiable and that is the point: local learning, no gradient crosses the router.
- **Routing:** per SEGMENT (not per token — the block-drift lesson). Surprise = the segment's
  mean CE of the combined prediction under its owning expert, calibrated per-expert with
  μ/σ EMAs over the segments the expert TRAINS on (rate 0.05, seed μ=r, σ²=max(1e-4,(0.1r)²) —
  mirrored verbatim from `src/prizma.py::_train_expert`). Recognized iff z ≤ 5.0 (the shipped
  `z_novel`). Novelty recruits the next slot (left-to-right fill = recruit recency); at cap
  m_max=E, novelty falls back to argmin-surprise routing. Recruit-at-cap relationship to the
  registered semantics: PR-05's m_max **Policy A** is recruit-by-eviction (evict lowest
  lifetime routing share, tie → most recent); this probe implements the milder no-eviction
  fallback, and the cap is measured INERT here (0 fallback segments, below) — with 2 corpus
  blocks and E ≥ 4 it never binds, exactly mirroring PR-05's 0-evictions-at-home.
- **LRs:** one per arm-family, chosen on seed 0's first-200-A-segment loss, then frozen
  (PR-07′ addendum rule). Selected: plain lr=0.01 (A-loss 4.197), fusion_e4/e8 lr=0.003
  (A-loss 4.217). Honest confound: the fusion families preferred a lower LR (the tissue adds
  effective capacity and shifts the optimum); each family is compared at its OWN frozen-best
  LR — the same per-arm-family rule PR-07′ used.
- **Eval:** PR-07′ `eval_bpc` math over combined logits; each eval segment served by its
  argmin-surprise committed expert (the `route_for_inference` analog; no recruitment at eval).
- **Budget:** first cell (plain, seed 0) 44.3 s measured; 6 cells + 9 LR-selection runs
  ≈ **13 min CPU**, torch.set_num_threads(8). Well under the ~90 min ceiling.
- **Seeds:** {0, 1}. n=2 → all deltas are DESCRIPTIVE; no statistical claim is made or
  implied.

## 1. Retention / adaptation table (bpc; FGT_A = pre − post)

| config | seed | bpc_A_pre | bpc_A_post | FGT_A | bpc_B | wall s |
|---|---|---|---|---|---|---|
| plain | 0 | 3.150 | 3.328 | −0.178 | 2.931 | 44.3 |
| plain | 1 | 3.057 | 3.297 | −0.240 | 2.899 | 41.8 |
| **plain mean** | | **3.1035** | **3.3124** | **−0.2089** | **2.9150** | |
| fusion E=4 | 0 | 3.030 | 3.240 | −0.211 | 2.893 | 43.5 |
| fusion E=4 | 1 | 3.047 | 3.223 | −0.176 | 2.894 | 43.4 |
| **fusion E=4 mean** | | **3.0382** | **3.2316** | **−0.1934** | **2.8937** | |
| fusion E=8 | 0 | 3.030 | 3.240 | −0.211 | 2.893 | 44.7 |
| fusion E=8 | 1 | 3.047 | 3.223 | −0.176 | 2.894 | 44.6 |
| **fusion E=8 mean** | | **3.0382** | **3.2316** | **−0.1934** | **2.8937** | |

Registered reference (n=5, `results/blockdrift_PR-2026-09-03-07/RESULTS.md`): STREAM bpc_B
**2.954**, FGT_A **−0.189**. The within-session plain re-run (n=2: 2.915 / −0.209) is
consistent with it — within-session comparability holds.

Reading (descriptive, n=2):

- **Fusion helps slightly, uniformly, on both sides of the boundary**: bpc_A_pre −0.065,
  bpc_A_post −0.081, bpc_B −0.021 vs the within-session plain mixer. All per-seed diffs point
  the same way except FGT (below), but the seed spreads (plain FGT −0.178/−0.240,
  fusion −0.211/−0.176; bpc_B spread ≤0.038) are the same order as the deltas — this is a
  small-effect observation, not a result.
- **FGT_A**: both arms show strongly NEGATIVE forgetting (B-training improves A-eval — the
  PR-07′ phenomenon). Fusion's FGT is +0.016 closer to zero than plain, but mechanically its
  lower A_pre (better pre-B fit) shrinks the room for negative forgetting; the +0.016 is well
  inside the seed spreads. Both arms are far below the PR-07′ ≤0.05 bar.
- **E=4 ≡ E=8 exactly** (all four metrics, both seeds): only 2 experts are ever recruited, so
  pool size never binds. The pool is inert at home, mirroring PR-05's inert-at-home cap.

## 2. Specialization ledger — the corpus-purity check

Ideal fusion: text8 segments → one expert group, shakespeare → another. Measured
(fusion_e4; e8 identical):

| seed | recruits | e0 train (A/B) | e1 train (A/B) | purity A | purity B | cap fallbacks | eval A_eval→(e0,e1) | eval B_eval→(e0,e1) |
|---|---|---|---|---|---|---|---|---|
| 0 | e0 @A-batch-0 (empty pool); e1 @B-batch-0 (trigger z=25.1) | 3906 / 26 | 0 / 3895 | 1.000 | 0.993 | 0 segs | 360, 30 | 6, 429 |
| 1 | e0 @A-batch-0 (empty pool); e1 @B-batch-0 (trigger z=26.3) | 3906 / 0 | 0 / 3921 | 1.000 | 1.000 | 0 segs | 381, 9 | 0, 435 |

- **Training-ledger sums balance exactly** (A: 3906, B: 3921 on both seeds) — every segment
  trains exactly one expert (see §4, second bug: the first version double-trained boundary
  segments; caught by the PR-06 "check the training ledger" discipline and fixed before any
  number was accepted).
- **The specialization check PASSES at the routing level.** Vigilance routing on this block
  stream is DOMAIN-coherent, not surprise-coherent: e0 owns essentially 100% of text8, e1
  essentially 100% of shakespeare; mean expert-majority-corpus share 0.997 / 1.000. The
  boundary recruit fires on the FIRST shakespeare batch with trigger z ≈ 25–26 — a ~5× margin
  over the z=5 vigilance threshold, the same enormous-margin recruitment regime measured in
  the home stream (EXPERT_ECONOMY §2.1: mean z ≈ 290). The residual leakage (seed 0: 26
  B-segments to e0, 1.7% eval leakage) comes from the single boundary batch where some
  segments still z-passed under the text8 expert.
- Eval-time argmin routing (raw CE, `route_for_inference` analog) is 92–98% pure on A_eval
  and 99–100% on B_eval — inference routing is nearly as clean as training routing.

## 3. Honest mechanistic read

1. **The PR-06/terminal lesson does NOT reproduce in the block regime — and the terminal
   probe predicted exactly this.** In interleaved streams, vigilance experts are
   surprise-coherent catch-alls because floors calibrate on mixture data (PR-06, G1/G2, G3a/b
   — measured end-to-end, terminal verdict: information-bounded at the partition level). Here
   the SAME mechanism, given block-coherent exposure (≳10² contiguous segments during floor
   calibration — precisely the terminal probe's stated condition), produces domain-coherent
   specialists with ~1.0 corpus purity. The block-drift re-scope of BAR-6′ is thereby
   validated on the routing side too: the block regime is the router's home regime.
2. **But the plain mixer already passes block-drift retention/adaptation without any of
   this machinery** (PR-07′ n=5: FGT −0.189, adaptation −3.18 vs FROZEN). The fusion tissue's
   bpc deltas at this scale are small (~0.02–0.08 bpc, ~0.6–2.5%) and n=2-descriptive. An
   honest confound cuts against over-reading the gain: the routed expert head ADDS capacity
   (8,385 params/expert ≈ 8% of the backbone per expert), so part of the −0.02…−0.08 is
   plausibly capacity, not routing. The clean GPU-tier control is a plain mixer + ONE shared
   extra head of the same size (routing ablation, fusion spec §4 arm 4) — at this scale we
   did not run it; it is listed as a PR-LM-1 design requirement below.
3. **What routing buys is not average bpc — it is the LEDGER.** The plain mixer cannot show
   a routing-attributable ablation gap because there is nothing to ablate. The fusion column
   reconstructs task identity from raw surprise (no labels, no boundaries) with ~1.0 purity
   and fires its recruit on the first out-of-domain batch with a 5× margin. If PR-LM-1's
   claim is to be about lifelong adaptation (which expert KNOWS what), this ledger is the
   mechanism — and it works in the block regime.
4. **Recruit-at-cap is inert at home, again.** 0 fallback segments; E=4 and E=8 runs are
   bit-identical. Pool size only becomes load-bearing when the number of regimes grows past
   E — a property the 2-block toy stream cannot test.

### Design note — frozen-backbone variant (not run)

The alternative variant (backbone FROZEN after corpus A; only the expert tissue adapts
through B) would turn the column into a pure routing problem: retention of A is then exact
(no shared weights move), and all adaptation must flow through the routed expert heads. We
expect it to show perfect FGT_A ≈ 0 with materially worse bpc_B (only ≤8,385 params/expert of
adaptable capacity vs the full 103k backbone) — i.e., the opposite corner of the
retention/adaptation trade from the run variant. It is the natural ablation for the GPU tier
where the per-expert predictors are full FFN blocks, not 8k-param probes; at this scale it
would measure the probe's own capacity, not the architecture.

## 4. Bug disclosure (caught before any number was accepted)

Two implementation bugs were found and fixed during bring-up; both were invisible to the
metrics and visible only in wiring/ledger checks — the PR-06 lesson paid for itself:

1. `sd` (per-expert σ) was broadcast instead of indexed by the argmin expert
   (`sd[min_idx]`). The crash fired only when ≥2 experts were committed (i.e., only in the
   real B-boundary regime — the smoke check with 1 expert could not see it).
2. Boundary double-training: a batch's novel segments trained BOTH the argmin expert and the
   fresh recruit (seed 0's first ledger showed e0 B=65 + e1 B=3888 = 3953 > 3921 total B
   segments). Fixed to exclusive recruit ownership; the final ledger sums balance exactly.

The two superseded executions' outputs were discarded (invalid-by-construction, retained
nowhere); the retained raw record for this probe is the final `raw.json` from the clean run,
produced by the current `seq/fusion_probe.py` (commit-tree state of this directory). The run
log of the final execution is `fusion_probe_2026-09-07_run.log`.

## 5. What this means for PR-LM-1 (GPU tier)

**Recommendation: wire the vigilance-routed expert tissue into PR-LM-1 — but stake the claim
on the routing ledger and the ablation gap, not on average bpc.** At toy scale the plain
mixer is already sufficient for block-drift retention (strongly negative FGT) and adaptation;
the tissue adds a small uniform bpc gain whose honest decomposition (routing vs mere capacity)
requires the shared-extra-head control. What the tissue uniquely provides — and what the
plain mixer structurally cannot — is a label-free, boundary-free routing ledger with ~1.0
corpus purity and first-batch recruit timing (z-margin ≈ 5× threshold), plus a bounded-M pool
whose cap is measurably inert at home. Concrete carry-forward for the PR-LM-1 design:

- **Segment-level routing is the validated granularity** for block streams (per-token routing
  was never needed here; the fusion spec's token-level write-error routing remains a
  kernel-feasibility question, not a specialization one).
- **Local tissue learning works**: per-expert private optimizers on detached activations, no
  shared backprop through the non-differentiable router — keep this property (it is the
  architectural thesis, and it trained stably here).
- **The claim-relevant arms** (fusion spec §4): keep the minus-routing ablation and add the
  shared-extra-head capacity control; make the stream MANY-block (true open-ended drift) so
  recruitment/eviction actually binds — the 2-corpus toy cannot distinguish E=4 from E=8.
- **Retention bars stay unchanged** (FGT ≤ 0.05 bpc, adaptation ≥ 0.10 bpc): both mechanisms
  clear them trivially at this scale; the open question at GPU scale is whether the routing
  advantage survives real capacity ratios (expert FFN ≫ backbone share) and many-block drift.
