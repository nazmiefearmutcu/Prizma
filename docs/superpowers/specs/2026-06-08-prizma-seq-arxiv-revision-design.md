# Prizma-Seq arXiv Revision — Design / Spec

- **Date:** 2026-06-08
- **Status:** Approved (brainstorming) → ready for writing-plans
- **Scope decision:** Existing data only (no new compute). Target standard: **solid arXiv preprint** (clean, figure-rich, internally consistent, error-free). NOT a workshop/conference submission.
- **Hard constraint:** This design phase changes **no line of `paper/main.tex`**. All edits happen in the implementation phase.
- **Target artifact:** `paper/main.tex` (+ new `paper/figures/`), rebuilt `paper/main.pdf`, refreshed `paper/prizma_seq_arxiv_submission.tar.gz`.

---

## 1. Context / current state

Prizma-Seq is a Gated-DeltaNet-family sequence mixer whose only lever is a **parameter-free
quadratic feature map (`quad2`)** that makes the carried delta-state *rectangular* (`d_h × d_φ`,
monomials as fixed seeded buffers → 0 added params), preserving `O(1)` inference.

Current paper: `paper/main.tex`, 422 lines, 9-page PDF, single author "Aylin / Independent
Researcher", 23-entry `refs.bib` (complete). Public repo: `github.com/nazmiefearmutcu/Prizma`
with raw result JSONs committed under `results/`.

**The science is sound and the paper is unusually honest** (pre-registered falsifiable bar, binding
limitations, no inflated claims). The work needed is *polish + one real math fix + figures*, not
re-research.

---

## 2. Diagnosis (verified against raw `results/*.json`)

### 2.1 Verified CORRECT (every headline number traced to data)

| Claim | Source | Verdict |
|---|---|---|
| Free-energy gradient `∂F/∂S = −ε·kᵀ` | analytic | ✓ |
| Dim count: 32 linear + 224 cross = 256 (`d_φ`) | analytic | ✓ |
| Param-match +0.6% (857,216 vs 862,368) | `flop_ledger_v2.txt` | ✓ |
| Latency table; crossover @32k; 2.4× @65k; 1.3–1.6× slower <16k | `gpu_latency.json` | ✓ exact |
| Memory float-ratio 28.4× (4k) → 455× (65k) | `kv_floats/state_floats` (4.19M/147,456; 67.1M/147,456) | ✓ |
| FLOP 2.14× at headline d128L4H4 | `flop_ledger_v2.txt` (`quad2_d256_v1ref`) | ✓ **as-coded** (ideal 1.78×) |
| text8 BPC 1.7496 vs 1.7254; random 4.7549 = log₂27 | `v3_campaign_results.md`, `gpu_charlm2.json` | ✓ |
| Causal ablation quad2≫rand_linear≈none≫TF | (locate exact JSON in W2) | ✓ values consistent |
| Length-extrap 10× (0.398 vs 0.041) | text/JSON | ✓ |

### 2.2 Issues to fix

| # | Type | Issue | Severity |
|---|---|---|---|
| **I1** | **Math error** | Eq (3) defines the rectangular state but does **not** redefine `ε_t`. Reusing Eq (1)'s `ε_t = v_t − S_{t-1}k_t` is **dimensionally invalid** (`S_{t-1}∈ℝ^{d_h×d_φ}` · `k_t∈ℝ^{d_h}`). Must be `ε_t = v_t − S_{t-1}φ(k_t)`. | High |
| **I2** | Correctness/clarity | "peak state is a constant **17.9 MB** … **147,456 state floats**" conflates **measured peak process memory** (17.9 MB) with the **analytic state** (147,456 fp32 = **0.56 MB**). Also "**28–455×** less than KV" is the analytic float ratio, while **measured peak ratio is 2.8–31×**. Both appear without separation. | High |
| **I3** | Consistency | Param-match sign: MQAR "+0.6%" vs char-LM "−0.13%" — both have Prizma larger; convention contradicts itself. | Low |
| **I4** | Precision | "2.14×" should be labeled **as-coded**; cite **1.78× kernel-ideal** for optimization headroom. | Low |
| **I5** | **Presentation (biggest gap)** | **Zero figures.** No architecture schematic, no latency/memory plots, no MQAR efficiency curve, no ablation chart. `results/figure.png` belongs to a *different* project (GRAIL continual-learning) — unusable. | High |
| **I6** | Reproducibility | No code/data availability statement; no arXiv category metadata; tables show medians only (seed ranges live in prose). | Medium |

None of these falsify the core result. I1 is a genuine notation error; the rest are polish + the figures gap.

---

## 3. Goals / Non-goals

**Goals**
- Fix all of I1–I6 using existing data only.
- Add 5 figures (1 schematic + 4 data plots) from committed JSONs.
- Keep the disciplined, honest tone and the ~9–11 page length.
- Produce a clean-compiling submission tarball with figures + `.bbl`.

**Non-goals (explicitly out of scope)**
- No new experiments / GPU runs (no denser TF grid, no ≥5-seed runs, no powered TOST).
- No restructuring of the argument; no numeric related-work table (that was the workshop option, declined).
- No new claims. The verdict set is frozen.

---

## 4. Workstreams (ordered)

Execution order: **W1 → W2 → W3 → W4 → W5**. All edits land on a single `paper/main.tex` plus a new `paper/figures/`.

### W1 — Math & correctness fixes (text only; no data changes)

**I1 — Eq (3) dimensional fix.** Current (`main.tex` ~144–149):
```latex
S_t = \alpha_t S_{t-1} + \beta_t \varepsilon_t \phi(k_t)^{\top},
\quad S_t \in \R^{\dhd\times\dphi}, \quad o_t = S_{t-1}\phi(q_t).
```
Add the lifted error term explicitly, e.g.:
```latex
S_t = \alpha_t S_{t-1} + \beta_t \varepsilon_t^{\phi} \phi(k_t)^{\top},
\quad \varepsilon_t^{\phi} = v_t - S_{t-1}\phi(k_t),
\quad S_t \in \R^{\dhd\times\dphi}, \quad o_t = S_{t-1}\phi(q_t).
```
Also note (near the free-energy paragraph ~114–116) that the lifted write descends
`F_t(S)=½‖v_t − Sφ(k_t)‖²`, whose gradient is `−ε_t^φ φ(k_t)^⊤` — keeping the predictive-coding
reading consistent in the rectangular case.

**I2 — Memory disambiguation.** Three touch points:
- Abstract (~39–41): replace "its state is a constant 17.9 MB … 28–455× less than a growing
  KV-cache" with wording that separates **analytic state-float ratio (28–455×)** from
  **measured peak-memory ratio (2.8–31×)**, and states the pure state is **0.56 MB** (147,456 fp32).
- §6 Inference (~289–295): label 17.9 MB as **measured peak process memory** (weights+activations+
  state); give analytic state bytes separately; present both ratios side by side.
- Table 1 "Inference" row (~236–238) and Table 2 caption (~300–304): clarify "peak mem" = measured
  process peak, and add the analytic float-ratio as the headline capacity number.

**I3 — Sign convention.** Make the char-LM param-match delta sign consistent with MQAR (~181–182):
if MQAR is "+0.6%" (Prizma larger), char-LM (Prizma 3,224,608 > TF 3,220,480) is **+0.13%**.

**I4 — FLOP precision.** At the two FLOP mentions (~162, ~357): write "**2.14× (as-coded; 1.78×
kernel-ideal)**". Source: `flop_ledger_v2.txt` config `quad2_d256_v1ref`.

### W2 — Figures (from existing data)

New dir `paper/figures/`. A small reproducible script (e.g. `paper/make_figures.py`) reads
`results/*.json` and emits PDFs. **Data-archaeology guardrail:** before plotting F4/F5, map each
plotted point to its exact JSON; if a clean grid is not reconstructable from committed JSONs, plot
the coarse grid the paper already cites and footnote single-seed points. **Never synthesize data.**

| Fig | Content | Data source |
|---|---|---|
| **F1** | Architecture schematic: square `d_h×d_h` state → `φ` lift → rectangular `d_h×d_φ`; delta write `+β ε φ(k)ᵀ` / read `S φ(q)`. Conceptual (TikZ or matplotlib). | none (schematic) |
| **F2** | Per-step decode latency vs `n` (flat Prizma ~7 ms × rising TF; crossover marked @32k). Small + big config. | `gpu_latency.json` |
| **F3** | Inference memory vs `n`, log-y, two panels: (a) **measured peak** (17.9 MB vs 49.5→561.7 MB), (b) **analytic** state-floats vs KV-floats (28–455×). Visually resolves I2. | `gpu_latency.json` |
| **F4** | MQAR accuracy vs params: Prizma solves D=128 @130K; TF needs ≥461K; coarse grid {130,461,857,3300}K → 3.5× gap. | consolidate `scale_frontier.json` + `tf_frontier.json` + `d128_sweep.json` + `cap_probe.json` |
| **F5** | Causal ablation bar: quad2 (0.997) ≫ rand_linear (0.589) ≈ none (0.518) ≫ TF (0.016), with seed error bars. | locate exact JSON (likely `feat_map_probe.json` / `recall_gate.json`; NOT `gpu_ablation.json`, which is a different surprise-norm ablation) |

Insert figures at natural points: F1 in §2 (Method), F2+F3 in §6 (Inference), F4 in §5 (Results/MQAR),
F5 in §7 (Ablation).

### W3 — Tables & statistics (existing dispersion only)

- Table 1: add a **seed-range / min–max** column beside medians (values already in prose/JSON; no new runs).
- Table 2: tighten config labeling (small=d128L4H4; big config named).
- No new seeds, no fabricated CIs. Only surface the dispersion already in `results/`.

### W4 — New content

- **Reproducibility & Availability** paragraph (end of §3 or before Conclusion): link
  `github.com/nazmiefearmutcu/Prizma`, point to `results/` JSONs and the exact configs/seeds.
- **Funding / Acknowledgements**: one line, "no funding to disclose."
- Optional one-sentence note that all numbers are script-produced and JSON-committed.

### W5 — arXiv mechanics

- Primary category **cs.LG**, secondary **cs.CL**.
- Choose license (e.g. arXiv default / CC BY 4.0).
- Verify abstract length within arXiv limit (~1920 chars).
- `\author` + affiliation final check.
- Rebuild: `pdflatex → bibtex → pdflatex ×2`; scan for overfull hboxes; confirm figures embed.
- Refresh `prizma_seq_arxiv_submission.tar.gz` to include `figures/` + `main.bbl` + `refs.bib` + `main.tex`.

---

## 5. Acceptance criteria

1. I1 fixed: every equation is dimensionally consistent; `ε_t^φ` defined for the rectangular state.
2. I2 fixed: measured-peak vs analytic-state numbers are never conflated; both ratios stated explicitly; pure state = 0.56 MB called out.
3. I3/I4 fixed: consistent param-match sign; FLOP labeled as-coded with kernel-ideal cited.
4. ≥5 figures present, each traceable to a committed JSON (or schematic), captioned, referenced in text.
5. Reproducibility/availability paragraph + arXiv metadata present.
6. `main.pdf` compiles clean; submission tarball contains all sources + figures.
7. **No claim changed.** The verdict set (MQAR parity/efficiency, induction, selective-copy, char-LM within-margin, inference, ablation, length-extrap) is identical to the current paper.

---

## 6. Risks / open items

- **F4/F5 data dispersion.** Mitigation: data-archaeology step first; honesty guardrail (coarse grid + footnote) if a clean grid is not reconstructable. This is the main execution risk.
- **F5 source ambiguity.** `gpu_ablation.json` is a *different* ablation; the quad2/rand_linear/none/TF data must be located in another file before plotting.
- **Schematic effort (F1).** TikZ vs matplotlib — pick whichever compiles cleanly in the arXiv toolchain; F1 is the only non-data figure.
- **arXiv endorsement.** First cs.LG submission may need endorsement; flagged for the author, not a blocker for paper readiness.
