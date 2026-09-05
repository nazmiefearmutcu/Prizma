# Expert Economy — growth quantification, merge/prune/bounded-M design, open-world abstain

**Status: EXPLORATORY (LANE-EXPLORATORY).** This document contains (i) measured results from
real runs of the repo's own stream, (ii) design specifications that are **not implemented in
`src/`**, and (iii) a pre-registration **draft**. Per `docs/preregistry/POLICY.md`, exploratory
results may motivate but never become claims; the only future-claim vehicle here is the draft
in §5 (registry id **PR-2026-09-03-05, status IN-WRITE** — `INDEX.md` intentionally untouched).

**Wave-C C3**, commissioned from `committee/brainstorm_2026-09-03/10_continual_lifelong_learning.md`
(report 10, §1.3 + P3) and `00_COMMISSION_SYNTHESIS.md` §C4 caveat ("expert memory currently
grows unboundedly").

- **Script:** `experiments/expert_economy.py` (all instrumentation lives there, in an
  `InstrumentedPrizma` subclass; `src/prizma.py` and `experiments/run_continual.py` end this
  task byte-identical — verified by `git status`).
- **Raw ledger:** `results/expert_economy_2026-09-03/growth.json` (3 seeds × 3 stream kinds,
  whole analysis ≈ 8 s CPU).
- **Streams:** the shipped E-suite constructor `src.data.structured_permuted_tasks` at the
  shipped E1 settings (K=5 domains, d=24, 8 classes, h=48, M=K+3=8 experts, epochs=15, DFA
  feedback, consolidate=True, z_novel=5.0 — identical to `run_continual.E1_main`). The two
  interleaved variants are **new exploratory streams** (same generator and exposure, different
  presentation order) and are labelled as such everywhere; they are not shipped benchmarks.
- **`route_log`:** report 10's audit ("usage is already tracked") confirmed — the attribute
  exists in `src/prizma.py` and is consumed here unchanged; inference-time routing/surprise
  statistics (which `route_log` does not carry) were collected by the script's subclass
  wrapper around `train_batch` and by reusing `route_for_inference`'s existing return value.

---

## 1. Verified arithmetic — the commission's 4,304 floats/domain is correct

Counted from live `Expert` array shapes (not from the doc), at the shipped E-suite shapes
d=24, h=48, K=8:

| tensor | shape | floats |
|---|---|---|
| Wenc | h×d | 1,152 |
| benc | h | 48 |
| Wdec | d×h | 1,152 |
| bdec | d | 24 |
| Wcls | K×h | 384 |
| bcls | K | 8 |
| **trainable subtotal** | | **2,768** |
| Bdec (fixed FA feedback) | h×d | 1,152 |
| Bcls (fixed FA feedback) | h×K | 384 |
| **fixed-FA subtotal** | | **1,536** |
| **total per expert** | | **4,304** |

Report 10 §1.3 quoted "≈4,304 floats/domain (2,768 trainable + 1,536 fixed FA)": **verified,
exact match** (`growth.json → summary.floats_per_expert.verified_match = true`).

**Measured growth curve** (block E1-style stream, mean over seeds 0/1/2 — identical on every
seed): one expert recruited per domain, so parameters grow linearly at exactly 4,304
floats/domain:

| after domain | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| experts recruited | 1 | 2 | 3 | 4 | 5 |
| total floats (trainable+FA) | 4,304 | 8,608 | 12,912 | 17,216 | **21,520** |

**Against the backprop MLP baseline** (`[24,128,128,8]`, fixed at **20,744** trainable params,
same closed form as `run_continual.param_count_mlp`):

- recruited trainable-only / MLP = 13,840 / 20,744 = **0.667**
- recruited incl. fixed-FA / MLP = 21,520 / 20,744 = **1.037**
- pre-allocated pool incl. FA (M=8) / MLP = 34,432 / 20,744 = **1.660**

Honest reading: at K=5 the whole expert economy is still *smaller* than the fixed baseline
(trainable-only). The unbounded-growth attack surface is about the slope, not the level:
linear extrapolation of the measured 4,304/domain crosses the baseline's footprint at K≈8
domains and reaches ~4.15× it at K=20 (arithmetic extrapolation of a measured slope, not a
measurement). Also honest: today's code pre-allocates M=K+3 with K known a priori (report 10
audit point 3); the "open-world" accounting above counts only *recruited* experts — the
allocation policy itself is part of the gap P3 closes.

---

## 2. RESULTS (exploratory; 3 seeds, shipped E1 settings; `growth.json`)

### 2.1 The home regime is recruitment-clean — and that is why every economy organ is inert

Block E1-style stream, all seeds identical on the economy ledger:

| quantity | value (seeds 0/1/2) |
|---|---|
| experts recruited at stream end | 5 / 5 / 5 (exactly K) |
| frozen / committed | 4 / 4 |
| routing purity (each domain's test set → majority expert) | 1.0 on all 5 domains, all seeds |
| silently dropped batches (pool exhaustion) | 0 |
| eviction fires under M_max = K | **0** |
| eviction fires under M_max = K−1 | 1 (recruit #5; victim = most recently recruited among 4 experts tied at identical lowest share) |
| mergeable frozen pairs (Jaccard > 0.5 gate) | **0** — all 6 pairs have Jaccard exactly 0.0 |
| prune candidates (zero routing over whole stream) | exactly the 3 never-recruited pool slots |

**Surprise distribution at recruit events** (the trigger batch, evaluated on every expert with
a precision floor): the frozen experts' z-scores on the batch that triggers a recruit are
enormous — **mean 290.4, min 214.2, max 398.7** over 30 recruit events (3 seeds × 10
non-recruiting expert measurements). Recruitment in this regime is decided by margins ~40× the
z=5 vigilance threshold; there is no ambiguity to arbitrate, which is precisely why merge and
eviction never fire. **The cap and the merge criterion are inert in the home regime — that is
the quantified finding, and it is the point:** any future bounded-M claim (§5) must show the
machinery stays inert where Prizma already works, and only binds in overlap/open-world regimes
(report 10 P1's κ-band).

### 2.2 Interleaved presentation defeats recruitment by two different mechanisms (EXPLORATORY streams)

Synthesis §C4 flags interleaved collapse as the open caveat; these two new exploratory streams
probe the *economy* side of that caveat (accuracy was not the target here and was not scored):

- **Mixed-batch interleave** (samples of all K domains shuffled into one stream; identical
  exposure): recruited 2 / 2 / 1 across seeds. `route_log = [359744, 256, 0, …]` — one
  catch-all expert absorbs the stream; a second expert is recruited, trains exactly its
  warmup (256 samples) and then never again. Mechanism: every batch is a *mixture*, so the
  batch-mean reconstruction surprise is stationary — batch-level vigilance (report 10 audit
  point 5) is structurally blind to mixing.
- **Round-robin interleave** (domain-pure batches, batch order shuffled; identical exposure):
  recruited 2 / 2 / 2, same `route_log` shape. Mechanism: the first expert freezes early with
  an *immature* precision floor (high μ, high σ); the loose floor then recognizes every later
  domain, so it becomes a catch-all and no further recruitment fires.

In both modes memory stays small (~1–2 experts) **but the routing achievement — the thing that
makes FGT=0 meaningful — silently vanishes**: purity is trivially 1.0 only because a single
expert serves everything. This is a mechanism-level account (from the recorded ledger, not an
ACC claim) consistent with the C4 caveat, and it sharpens P3's scope: an expert economy can
only be as good as the recruitment signal that feeds it. Any merge/prune/budget machinery
must therefore log *recognizer immaturity* (freeze-time `n_seen` vs warmup) as a veto.

### 2.3 Open-world gap quantified — argmin never abstains today (probe: never-trained 6th domain)

A 6th structured-permuted domain (same generator, never trained on) is routed at test time.
Today's `route_for_inference` force-routes it: the probe's "purity" is a meaningless 0.28
spread across experts, each answering with confident logits. The proposed NOVEL rule
(§3.4) separates cleanly — min-surprise z-score over trained experts:

| rule | known domains: false-NOVEL | probe: detected NOVEL |
|---|---|---|
| per-sample, z=2 | 14.9% | 100.0% |
| per-sample, z=3 | 12.7% | 100.0% |
| per-sample, z=4 | 11.3% | 100.0% |
| **batch (128), z=2** | 3.0% | 100.0% |
| **batch (128), z=3** | 0.74% | 100.0% |
| **batch (128), z=4** | **0.0%** | **100.0%** |

Per-sample thresholds have a fat false-flag tail (~11–15%) on *known* domains — the empirical
reason the repo's v2 lesson (batch-level vigilance decisions only; report 10 audit point 5)
must extend to abstention. At batch level the rule is perfect in this regime: z=4 flags 100%
of the novel domain's batches and 0% of known batches.

---

## 3. DESIGN SPECS (not implemented; all numpy, all from existing machinery)

### 3.1 Vigilance-compatible merge

Run **offline from the ledger** (verified feasible: the script's `simulate_merges` already does
exactly this from recorded snapshots, no `src/` changes):

```
at freeze events (or every W_freeze samples), for each pair (m1, m2) of FROZEN experts:
  J = Jaccard(routed_set(m1), routed_set(m2))            # stage 1: gate
  if J <= tau (default 0.5): skip
  U = union(routed_set(m1), routed_set(m2))
  if Welch(S_m1(U), S_m2(U)) p > 0.05 and |cohen_d| < 0.5:   # stage 2: indistinguishable
    merged = mean(Wenc, benc, Wdec, bdec, Wcls, bcls)        # stage 3: feasibility
    if recon_mean(merged, set_1) <= mu_1 + z*sqrt(var_1)
       and recon_mean(merged, set_2) <= mu_2 + z*sqrt(var_2):
        replace {m1, m2} <- merged (route mass unioned)
```

- **Measured verdict on the E-suite:** never fires (Jaccard ≡ 0.0). Inert at home, by
  construction — experts are domain-pure. The intended habitat is report 10 P1's overlap band
  (κ > 0), where co-fired experts actually exist.
- **Guard (from report 10's own risk note):** merging two PC autoencoders by averaging is only
  principled if their latent bases align; the FA feedback matrices are expert-specific random
  bases and are **never averaged** (they are also unused by the settle-0 substrate the merge
  check runs on — documented assumption). Fallback if stage 3 fails on aligned bases: merge
  heads + route mass only, keep decoders separate (cheaper memory win, bounded either way).
- Statistical test: Welch t with Welch–Satterthwaite df; p-values computed with the repo's own
  `seq/stats.py` (`t_sf`), imported analysis-side only (`src/` does not import `seq/`;
  direction verified). Effect sizes reported regardless.

### 3.2 Prune (use-it-or-lose-it)

```
expert m is prunable iff frozen AND route_log[m] == 0 over a window of W samples
prune = release the slot (re-initialise, committed=False); a later recruit re-uses it
```

Borrowed framing: selective stabilisation of synapses (Changeux & Danchin 1976); computational
pruning accounts (Chechik, Meilijson & Ruppin 1998). Measured on the E-suite: only the three
never-recruited slots qualify — i.e. pruning as specified is a **no-op on content experts** in
the home regime; its risk (deleting a domain that reappears — catastrophic, replay-free) is
bounded by the pre-registered zero-false-prune bar if P1's reappearance streams are run.

### 3.3 Bounded M_max: recruit-by-eviction vs hard-cap refusal

- **Policy A — recruit-by-eviction:** on a recruit when the pool is at M_max, evict the expert
  with the lowest lifetime routing share (tie → most recently recruited: least consolidated,
  least time to have served other domains), re-use its slot. **Honest risk: eviction
  reintroduces forgetting by construction** — the evicted domain's weights are the only record
  of it and there is no replay to recover it. Quantified on the E-suite: with M_max = K the
  policy fires **0 times in 3/3 seeds** (§2.1) — the cap is *inert at home*; it only binds in
  open-world/overlap regimes where its cost must be measured, not assumed away.
- **Policy B — hard-cap refusal:** note that the current code already has a de facto cap
  behaviour and it is the worst of the options: after pool exhaustion
  (`if self.active >= self.M: return`) incoming batches are **silently dropped** (the
  InstrumentedPrizma drop-counter exists precisely to expose this; 0 drops on the block
  stream, by construction of M=K+3). If refusal is ever needed, it should be: stay in argmin
  mode for answering, log the batch as NOVEL-UNHANDLED, never discard it silently.
- Reporting duty (borrowed from E3): any bounded-M run reports ACC-vs-M_max the way E3
  reports ACC-vs-M, plus eviction-fire counts and victims.

### 3.4 Test-time abstain / NOVEL (open-world option; spec, not implemented)

```
route_for_inference(X):                     # X = a BATCH; batch-level by design
    S = recon_matrix(X); m = trained experts
    z[n, m] = (S[n, m] - mu_m) / sigma_m    # per-expert precision floors, already in code
    if mean_n(min_m z[n, m]) > z_abstain:   # z_abstain = 4, calibrated in §2.3
        return "NOVEL"                      # abstain: no expert claims the batch
    return argmin_m S[n, m], S              # exactly today's behaviour otherwise
```

This closes report 10's audit gap §1.2 ("inference routing has no abstain … answered
confidently") with machinery that already exists (`mu`, `var`, `z_novel` — the same precision
test used at train time). Calibration measured in §2.3: batch-level z=4 → 0% false-NOVEL on
known domains, 100% detection on a never-trained domain (3 seeds). Open metric for a future
claim: open-set recognition score (AUROC of the min-z statistic), not accuracy alone.

---

## 4. Borrowed-vs-new ledger

| organ / policy | borrowed from | status here |
|---|---|---|
| Category merge under a vigilance test | [B] ART family — fuzzy ART category dynamics (Carpenter, Grossberg & Rosen 1991); Distributed ARTMAP (Carpenter, Milenova & Noeske 1998) | spec §3.1; inert at home (measured) |
| Expand / prune / select lifecycle | [B] DEN, dynamically expandable networks (Yoon et al. 2017); Expert Gate (Aljundi, Chakravarty & Tuytelaars 2017 — the closest historical ancestor of recon-based expert routing, per report 10 §4); PNN (Rusu et al. 2016); PackNet budgeting (Mallya & Lazebnik 2018) | spec §3.2–3.3 |
| Pruning, use-it-or-lose-it | [B] selective stabilisation (Changeux & Danchin 1976); computational pruning (Chechik, Meilijson & Ruppin 1998) | spec §3.2 |
| Sleep-time downscaling / reorganisation | [B] SHY (Tononi & Cirelli 2014); the merge/prune window is a natural wake-sleep venue (report 10 P8) | noted, not designed here |
| Expert turnover | [B] adult hippocampal neurogenesis turnover, activity-dependent (qualitative, *(knowledge)*) | framing only |
| Usage/confusion-driven economy for vigilance-recruited PC experts; eviction-inertness quantification; batch-level NOVEL calibration on a real run; catch-all collapse ledger for interleaved presentation | [N] new | §2 results, §3 specs |

## 5. Pre-registration DRAFT — PR-2026-09-03-05 (status IN-WRITE; NOT registered in INDEX.md)

> **Draft bar (verbatim):** *on the shipped E1 stream, bounded-M_max=K (the stream's true
> domain count) with eviction achieves ACC/FGT within ±0.02 of unbounded at 10 seeds.*
>
> **Operationalisation (draft):** shipped E1 settings (`structured_permuted_tasks`, K=5, d=24,
> h=48, M_unbounded=K+3, epochs=15, DFA); bounded arm = recruit-by-eviction at M_max=5 with the
> §3.3 policy; ≥10 seeds; report ACC and FGT per arm with 95% CI; **pass** iff
> |ACC_bounded − ACC_unbounded| ≤ 0.02 and |FGT_bounded − FGT_unbounded| ≤ 0.02 at every seed-
> summary level; **falsification** = either delta exceeds 0.02, or any eviction fires and
> degrades a re-tested domain (eviction-fire count is a reported covariate: measured
> expectation on this stream is 0, §2.1).
>
> **Status:** IN-WRITE. This document lives in the exploratory lane; `docs/preregistry/INDEX.md`
> is deliberately untouched. Registration (if the owner promotes it) means moving this draft
> into the registry and committing it **before** the bounded-M implementation exists.

## 6. Honest limits

1. **Exploratory lane.** All §2 numbers are n=3 seeds, single protocol (block, multi-epoch,
   batch-level decisions). They quantify mechanisms; they support no headline. The
   multi-epoch protocol caveat (report 10 audit point 6) applies to every run here.
2. **Interleaved streams are new and exploratory**, constructed from the shipped generator by
   changing presentation order only. They are not the E5 checkerboard construction, and no ACC
   was scored here — the catch-all finding is a ledger observation (route_log shape), not an
   accuracy claim, and its scope is one K/d/h configuration.
3. **Merge feasibility assumes the settle-0 substrate** (`n_settle_steps=0`, no noise or
   quantization), which is exact for the E-suite but would need re-derivation if settling is
   switched on.
4. **Routed sets are test-set routing**, used as a proxy for the routed-input distribution;
   a production merge criterion would maintain running routed-set sketches online.
5. **The abstain calibration is one generator family.** z_abstain=4 is measured, not
   universal; report 10 P2's Chernoff machinery is the right tool for predicting it per
   domain pair before deployment.
6. **No `src/` changes.** `git status` must show `src/prizma.py` and
   `experiments/run_continual.py` unmodified; all instrumentation is the script's
   `InstrumentedPrizma` subclass. The full test suite (219 passed + 10 skipped before this
   task) must still pass with `tests/test_expert_economy.py` added (7 tests, <1 s).
