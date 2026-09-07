# Fusion probe 2 — three completion probes sharpening the PR-LM-1 design (frozen-backbone, shared-extra-head, returning-domain)

**LANE-EXPLORATORY (docs/preregistry/POLICY.md). Design-informing ONLY — every number below
is EXPLORATORY, n=2 seeds {0,1}, descriptive; no statistical claim is made or implied.**
Extends the 2026-09-07 fusion probe (`results/exploratory/fusion_probe_2026-09-07/RESULTS.md`)
ahead of the PR-LM-1 GPU-tier registration.

- **Script:** `seq/fusion_probe.py` (ADDITIVE extension; the original probe's code path,
  defaults and no-arg CLI behavior are untouched — `python seq/fusion_probe.py` reproduces the
  2026-09-07 probe). New modes: `frozen` | `shared` | `return` | `all2`.
- **Raw ledgers (crash-safe, tmp + os.replace per cell):** `frozen.json`, `shared.json`,
  `return.json` in this directory. Run logs: `frozen_run.log`, `shared_run.log`, `return_run.log`.
- **Backbone / tissue / routing:** identical to the 2026-09-07 probe (Prizma-SeqLM 2L d=64
  H=4 chunk=64 window=16, 103,056 params; PCExpertHead Wenc 64→64 tanh + Wdec 64→65, 8,385
  params/expert; per-segment vigilance z≤5.0, μ/σ EMAs rate 0.05, recruit next slot, private
  per-batch AdamW on detached base+hidden, no router gradient).
- **Stream (P1/P2):** PR-07′ verbatim 2-block (text8[0:1M) → tiny-shakespeare).
  **Stream (P3):** three blocks text8[0:1M) → shakespeare → text8[1.1M:2.0M), one pass;
  C_eval = text8[2.0M:2.1M). **C disclosure:** C starts at 1.1M, NOT 1.0M — text8[1.0M:1.1M)
  is A_eval in the PR-07′/fusion protocol and training on it would corrupt the A-retention
  probe; C is therefore fully text-disjoint from A_train AND A_eval, same domain (text8).
  Vocab 65 in all modes (C adds no new chars).
- **LRs:** the frozen PR-07′ rule (per arm-family, seed-0 first-200-A-segment loss, grid
  {1e-3, 3e-3, 1e-2}), re-run in-session. Selected: plain **0.01** (A-loss 4.197);
  fusion_e4 / frozen_e4 / shared_h64 **0.003** (4.217); shared_h256 **0.003** (4.258).
  P1 disclosure: the frozen arm's B-phase expert LR = the fusion-family value (an A-phase
  selection cannot distinguish a frozen-B arm; disclosed). P3 uses the same fusion/plain
  family LRs on the identical A block.
- **Determinism check:** the P2 plain and fusion_e4 re-runs reproduce the 2026-09-07 raw.json
  to all four decimals on every metric (plain bpc_B 2.9312/2.8988, fusion_e4 2.8934/2.8940) —
  within- and cross-session comparability holds.
- **Budget:** first cells measured (frozen 35.1 s, shared 42.3 s, return 71.5 s); total cell
  wall 834 s + LR selections ≈ **~17 min CPU**, torch.set_num_threads(8) — well under the
  ~100 min ceiling; no blocks were shrunk.

## 1. P1 — Frozen-backbone variant (backbone frozen after corpus A; only experts learn through B)

| config | seed | bpc_A_pre | bpc_A_post | FGT_A | bpc_B (combined) | bpc_B (base-only) | backbone frozen? | wall s |
|---|---|---|---|---|---|---|---|---|
| frozen_e4 | 0 | 3.030 | 3.028 | **+0.001** | 3.544 | 5.293 | YES (param-identity) | 35.1 |
| frozen_e4 | 1 | 3.047 | 3.043 | **+0.003** | 3.495 | 5.256 | YES (param-identity) | 32.9 |
| **mean** | | 3.0382 | 3.0358 | **+0.0024** | **3.5193** | **5.2747** | | |

References: backbone-TRAIN fusion E=4 bpc_B **2.8937** (same session, §2); PR-07′ FROZEN arm
base-only bpc_B 6.132 (n=5, different LR family).

Read (descriptive, n=2):

- **Retention is exact**: FGT_A +0.002/+0.003 (vs −0.19..−0.21 when the trunk trains). The
  residual is eval-routing, not weight drift — e0 trained on 0 B segments (ledger), the
  backbone is parameter-identity verified; the ±0.002 comes from a few A_eval segments
  argmin-routing to the newer slots (ev_A 377/11/2 and 365/24/1/0).
- **The tissue alone carries substantial adaptation**: with the trunk frozen, the two
  8,385-param expert heads take bpc_B 5.275 → 3.519 (−1.755), closing **~74%** of the gap to
  the full backbone-train fusion (2.894). The fusion design's division of labor (trunk =
  general, experts = domain tissue) is a REAL operating point, not just a story: perfect
  retention, most of the adaptation, from ~16k moving params.
- Honest scale caveat: at this probe scale the tissue is only ~8% of the backbone per expert;
  the GPU tier's full-FFN experts will shift this fraction — measure it there, don't carry
  the 74% as a prediction.

## 2. P2 — Shared-extra-head capacity control (the confound the 2026-09-07 probe flagged)

Same extra-parameter budget as fusion E=4, but ONE shared head (no routing, no recruitment)
trained on ALL segments with the exact `_expert_train` rule. Two widths: h=64 (one expert's
size, 8,385 params) and h=256 (~the E=4 pool budget: 33,345 vs 4×8,385 = 33,540).

| config | seed | bpc_A_pre | bpc_A_post | FGT_A | bpc_B | wall s |
|---|---|---|---|---|---|---|
| plain | 0 | 3.150 | 3.328 | −0.178 | 2.931 | 42.7 |
| plain | 1 | 3.057 | 3.297 | −0.240 | 2.899 | 42.9 |
| fusion E=4 | 0 | 3.030 | 3.241 | −0.211 | **2.893** | 45.5 |
| fusion E=4 | 1 | 3.047 | 3.223 | −0.176 | **2.894** | 48.4 |
| shared h=64 | 0 | 3.030 | 3.284 | −0.254 | 2.904 | 42.3 |
| shared h=64 | 1 | 3.047 | 3.342 | −0.295 | 2.903 | 43.0 |
| shared h=256 | 0 | 3.057 | 3.348 | −0.291 | 2.929 | 43.9 |
| shared h=256 | 1 | 3.076 | 3.372 | −0.296 | 2.920 | 44.4 |

Means: plain 2.9150 / fusion_e4 **2.8937** / shared_h64 2.9032 / shared_h256 2.9245.

**Capacity-confound verdict: the fusion gain is NOT capacity — routing carries it.**

- At the SAME total extra-parameter budget (shared_h256 ≈ E=4 pool), the shared head is
  **WORSE than fusion on bpc_B by +0.031** (2.9245 vs 2.8937), same direction on both seeds.
- More shared capacity actively hurts: shared_h256 > shared_h64 by +0.021 on bpc_B — a single
  head serving both domains degrades as it grows (one correction field must fit two domains).
- Honest caveat: fusion vs the SMALL shared_h64 is only −0.0095 (2.8937 vs 2.9032), inside the
  seed spread — the load-bearing comparison is the same-budget h256, which both seeds put
  firmly on fusion's side. FGT note: shared arms show more NEGATIVE forgetting (−0.27/−0.29
  vs −0.19) — their head keeps improving A through B; descriptive only.

## 3. P3 — Returning-domain probe (text8 → shakespeare → fresh text8; the many-block question)

Arms: `return_vigilant` (natural vigilance), `return_forced` (C force-recruited to a fresh
slot — the treat-as-novel counterfactual; routing-only override, identical backbone
trajectory — A_pre/A_postB/B_postB are bit-identical across fusion arms), `plain3`.

| arm | seed | A_pre | A_postB | **A_final** | B_postB | **B_final** | C_preC | **C_final** |
|---|---|---|---|---|---|---|---|---|
| return_vigilant | 0 | 3.030 | 3.240 | 2.797 | 2.893 | 4.214 | 3.348 | **2.963** |
| return_vigilant | 1 | 3.047 | 3.223 | 2.771 | 2.894 | 4.144 | 3.339 | **2.945** |
| return_forced | 0 | 3.030 | 3.240 | 2.806 | 2.893 | 3.997 | 3.348 | **2.977** |
| return_forced | 1 | 3.047 | 3.223 | 2.778 | 2.894 | 3.939 | 3.339 | **2.954** |
| plain3 | 0 | 3.150 | 3.328 | 2.834 | 2.931 | 4.578 | 3.413 | **2.992** |
| plain3 | 1 | 3.057 | 3.297 | 2.800 | 2.899 | 4.150 | 3.380 | **2.977** |

Means: A_final 2.784 / 2.792 / 2.817; B_final 4.179 / **3.968** / 4.364; C_final **2.954** /
2.965 / 2.985 (vigilant / forced / plain).

**Pattern-completion verdict: the return is RECOGNIZED at the boundary, but the C ledger is
NOT domain-clean — mixed, with fractions.**

- (a) **Boundary recognition (the pattern-completion event): YES — 2/2 seeds.** The FIRST C
  batch fires NO recruit: under the existing text8 expert e0, batch-0 C surprise is
  z_mean 1.14 / 0.76, z_max 3.52 / 3.33 — every batch-0 segment z-passes (< 5,
  frac_novel 0.0). The router treats the returning domain as KNOWN at the boundary.
- (a′) **But the full-C training ledger splits**: pooled over seeds, 6,562/7,030 C segments
  (**93.4%**) train PRE-C experts and only 468 (**6.7%**) land on fresh recruits — yet of the
  pre-C share, the SHAKESPEARE expert e1 takes 3,996 (**56.8%**) vs the text8 expert e0's
  2,566 (**36.5%**) (per-seed: e1 takes 48.4% / 65.2% of C). Two mid-C recruits fire
  (slots 2 at C-batch 0, z≈5.5; slot 3 at C-batch 32, z≈6.2–10.2) from SINGLE-segment novelty
  under e1's floor — a recruit cascade, not domain novelty (e2 trains 2/1 segments; e3 takes
  428/37). Mechanism read: experts couple to the trunk state at calibration time — after B
  moved the trunk, e1 (the freshest correction) wins argmin on new text8, and immature-expert
  floors trigger spurious recruits. The 2-block regime's ~1.0 corpus purity does NOT survive
  the third block.
- (b) **C bpc under re-routed vs fresh-recruited handling: negligible cost either way.**
  vigilant 2.9539 vs forced 2.9651 (Δ −0.011, inside the seed spread); both beat plain by
  ~0.02–0.03. Re-mastery curves are nearly identical (vigilant 3.045→2.989→2.986→2.963;
  forced 3.059→3.006→2.995→2.977 at 25/50/75/100% of C; plain reaches 2.9846): a returning
  domain is cheap to re-master through ANY of these mechanisms — the lifelong win is in the
  LEDGER (no new expert needed for a seen domain), not in bpc.
- (c) **A/C retention after the full stream**: A_final shows strongly negative forgetting in
  all arms (2.78–2.82 vs A_postB 3.23–3.31 — C-training on same-domain text improves A-eval);
  the tissue arms lead (2.784 vigilant / 2.792 forced vs 2.817 plain). **The real retention
  event is B**: the TRUNK catastrophically forgets shakespeare during C in every arm
  (bpc_B 2.894 → 4.18 vigilant / 3.97 forced / 4.36 plain) — residual-corrector tissue cannot
  stop trunk drift. And expert REUSE matters: the UNTOUCHED shakespeare expert (forced arm,
  e1 trains 0 C segments) serves B_eval best — training an expert on other domains overwrites
  its original correction; an idle one retains it (forced beats vigilant by −0.21, plain by
  −0.40; direction consistent on both seeds).
- (d) **Ledger-per-block**: recorded in full in `return.json` (per-block train counts with
  exact balance checks — every segment trains exactly one expert; recruits with corpus tags
  and trigger z; per-block majority shares A 1.000 / B 0.993–1.000 / C 0.485–0.652; eval
  routes per block). Note: INFERENCE routing also loses domain purity in the returning regime
  (A_eval segments split across e0/e1/e3 by argmin CE) yet combined bpc stays good — the
  residual design makes mis-routing survivable at this scale, but argmin-CE routing is
  demonstrably NOT domain-identity routing once the trunk has moved.

## 4. What this means for PR-LM-1 (GPU tier) — registration recommendation

**Register the BACKBONE-TRAIN fusion variant as the primary PR-LM-1 arm.** The three probes
together: (1) capacity is REFUTED as the fusion gain's explanation — a same-budget single
shared head is worse (P2), so the routing/specialization mechanism is what earns the tissue
its bpc and its ledger; (2) backbone-train dominates the corners — frozen-backbone gives
exact retention (FGT +0.002) but leaves ~26% of the adaptation gap unclosed at probe scale,
so keep frozen-backbone as the RETENTION-CORNER ablation (and as a concrete many-block lever:
alternate trunk-frozen phases between blocks where exact retention is required), and drop
shared-head as a registered arm (keep it as the capacity control, where it did its job);
(3) the returning-domain probe validates the many-block design's central bet — a seen domain
RETURNING is recognized at the boundary (no spurious recruit, 2/2 seeds) and re-masters
cheaply — but it also adds two registration-grade requirements: **(i) stake the lifelong
claim on boundary-recruit timing + per-block train ledgers + the forced-recruit ablation,
NOT on per-expert purity** (purity collapses past two blocks via trunk-coupled argmin
leakage and immature-floor recruit cascades — add a floor-maturity veto: an expert below a
minimum n_segments must not trigger recruits); and **(ii) treat trunk drift as the dominant
forgetting term in many-block streams** (bpc_B +1.3 in every arm while the tissue stood
still or even idled) — the many-block protocol should include an expert-preservation or
trunk-frozen phase arm, because expert reuse (idle experts keep their correction) was worth
−0.21..−0.40 bpc here.

## 5. Disclosure

- Smoke wiring checks ran first for all three modes (separate `*_smoke.json`, discarded);
  they caught one real bug class before any number was accepted: `build_model`/`select_lr`
  take the vocab SIZE, not the dict (crash on first call — fixed; no invalid execution
  produced output). At smoke scale fusion_e4 ≡ shared_h64 bit-identically (no recruit fires
  on 64-segment blocks, so fusion degenerates to one shared expert) — an init-consistency
  check, not a result.
- P1's frozen arm shows a stray mid-B recruit (1 segment, z=5.387, seed 0) and seed 1 a
  4th slot — with the trunk frozen, e0's floor stops improving and single-segment vigilance
  noise recruits; harmless here, symptomatic of the same floor-maturity issue as P3.
- All numbers EXPLORATORY, n=2, descriptive. No existing module was modified; suite verified
  after this probe (see final pytest run record in the session log).
