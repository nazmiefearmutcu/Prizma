# Prizma-Seq v2 — Handoff Report (next task: the 4×A100 validation campaign)

> Written 2026-06-07 for the agent that picks this up. Fully self-contained — assume you share NO
> conversation context. Read this top-to-bottom before acting.

---

## 0. OPERATIONAL — READ FIRST (your green light)
- **Your next task is now DOABLE: run the 4×A100 validation campaign on Google Colab.**
- **You MAY use the Claude-in-Chrome extension** to drive the browser (native `claude-in-chrome`;
  fall back to the `chrome-devtools` / `playwright` MCP servers if needed).
- **Colab is already LOGGED IN** in the normal Chrome browser — the owner's account is authenticated,
  so notebooks open and A100 runtimes attach without a login step. Use the normal Chrome profile (do
  NOT spawn a fresh/headless browser — the account would not be signed in there).
- The owner authorized **up to 4 parallel A100 sessions**. Use them.
- **Never idle:** while an A100 run is in flight, do local work (Lever G, Lever F Triton kernel,
  Phase-2 prep, report drafting, council rounds). 2+ independent tasks ⇒ one message, multiple
  parallel `Agent` calls.
- **Do not ask "should I continue?"** — execute. Stop only on a genuine blocker or owner instruction.

---

## 1. Mission (owner, locked)
Take **Prizma-Seq** — a non-Transformer sequence architecture (a Gated-DeltaNet-family token mixer
with a parameter-free quadratic feature map giving a rectangular associative "delta" state, plus a
verified O(1)/constant-memory streaming inference path) — from a "razor-thin / within-margin
candidate" to **dramatically Pareto-dominant** vs a parameter/FLOP-matched **tuned** Transformer at
small scale, then ONE scale-up confirmation. "Better than the Transformer but NOT a Transformer."
Owner choices: **Pareto-dominance** + **genuine novelty** + **stabilize-then-scale**.

Non-negotiable integrity rules: no faked metrics; every number from a reproducible script with
seeds + CIs; param/FLOP-matched non-strawman baseline; powered statistics; `step()==forward()<1e-4`
O(1) guard before any accuracy number; honest borrowed-vs-new ledger; explicit scope rider.

## 2. Where the project lives
- **Repo (note the SPACE in the path — always quote it in shell):**
  `/Volumes/disk 2/Desktop_Migrate_2026-05-28/Projeler/proje/PRISM`
  (The old `/Users/nazmi/Desktop/Projeler/proje/PRISM` path is GONE — the project was migrated.)
- **Branch:** `v2-pareto-dominance` (do NOT work on `main`; the v2 code lives here).
- **Env:** system `python3.13`, torch 2.12 + MPS locally (CUDA/A100 only on Colab), numpy, scipy present.
  Run tests `python -m pytest tests/ -q` and module guards `python -m seq.delta` / `python -m seq.prizma_seq`
  from the repo root.

## 3. What is already DONE (committed on `v2-pareto-dominance`, 60 tests green)
**Phase 0 — honest-science harness + councils:**
- `seq/common.py::build_and_train(model_fac, task, cfg, device, seed=0, **fac_kw)` — seeds BEFORE
  building the model (fixes the v1 init-not-seed-pinned defect). **Use this for every run.**
- `seq/stats.py` — **powered statistics (validated vs scipy to ~1e-16):** `t_sf`, `t_isf` (real
  Student-t via incomplete beta), `summarize` (t-CI, median, solve-rate), `superiority_test`
  (one-sided Welch t), `margin_superiority(a,b,margin)` (BPC lower-is-better: tests mean(b)-mean(a)>margin),
  `tost_equivalence(a,b,margin)` (two one-sided t at the df-correct critical), `holm_correction`,
  `solve_rate`. The v1 stats were ANTI-CONSERVATIVE (normal-tail p-value reported p=0.033 for a true
  p=0.05 at n=10) — that is fixed; **quote no significance unless you use these functions.**
- `seq/lrsweep.py::sweep_lr(...)` — per-config LR sweep recording rejected LRs (de-confounds FLOP arms).
- `seq/ledger.py::param_match_report(tf_model, pz_model, tol=0.02)` — param-match auditor.
- **3 councils chartered** (`committee/charters/council{1,2,3}_*.md`) with a verdict log
  (`committee/verdicts/`, template `_TEMPLATE.json`). Round-0 outputs:
  `committee/verdicts/r0_general_plan-review.{md,json}`, `committee/quant/round1_levers.md`,
  `committee/standards/round1_landscape.md`, synthesis `committee/round0_v2_synthesis.md`.

**Phase 1 — six architecture levers, each LOCAL + O(1)/equivalence-guarded + committed** (all are
**default-OFF config knobs** on `PrizmaSeqConfig`; OFF = byte-identical to the v1 model):
| Lever | Config knob(s) | Guard result | Commit |
|---|---|---|---|
| C output-gate + per-head state RMSNorm | `out_gate: bool`, `state_norm: bool` | step==forward 6e-7; off=identical | `4e513d7` |
| E banded sliding-window | `banded_window: bool` | banded==full 3e-7 | `2f87b43` |
| H decoupled erase/write (GDN-2) | `decoupled_gate: bool` (+ `W_e`) | off=1e-6, chunked==ref 1e-4 | `62c717c` |
| B higher-order DeltaProduct | `n_delta: int` | k=1 identical, k=2 chunked==ref 1e-4 | `36f2be5` |
| D lean feature map | `feat_map='quad2_lowrank'`, `feat_rank: int` | d_φ=137 (vs 256), 0 added params, crosstalk pre-filter PASS | `55e03ac` |
| A surprise-gated write (NOVEL CORE) | `surprise_gate: bool`, `surprise_mode: 'norm'|'random'|'constant'` | **repeated-key exact = machine-zero**; controls reproducible | `56ebe85` |

Notes: trainable gates (C's `W_g`, H's `W_e`, G's `W_η` later) add params → **grow the TF in
lockstep** to keep param-match. Lever A's surprise-on forward is **sequential** (exact; a disclosed
speed cost — a Pareto knob a fused kernel can fix). Lever B's k≥2 uses a documented sequential-within-
chunk fallback (correctness met; speed later).

## 4. Binding council guidance you MUST honor (the bar)
From `committee/round0_v2_synthesis.md` (Council-1 gate remedies R1–R10, Council-3 bar):
1. **Statistics (DONE):** use `seq/stats.py`; ≥10 seeds for any decisive claim; report solve-rate +
   median + CI; superiority/TOST/margin tests; Holm-correct the causal-attribution family; run the
   **identical-model negative control** (two identical models must NOT show a "significant win" — if
   they do, the pipeline is broken → reject the result).
2. **Recall is a HARD TOST-parity GATE (Council-3):** MQAR (hard rung) + induction + selective-copy
   must reach **≥ tuned-TF parity via `tost_equivalence`**, with the optimization-vs-capacity
   **flip-test** on the hard rung (show a bigger TF DOES solve it). If recall is only "not much
   worse," the honest word is **Pareto-competitive**, not "dominant."
3. **per-FLOP ≤1.0× is currently UNMET (2.1–2.7×):** "dramatic" stays **conditional** until D+E(+F)
   deliver it AND all axes hold **simultaneously and powered**; otherwise downgrade to
   "Pareto-efficient/-competitive in the tested regime."
4. **R4 — reconcile d_φ BEFORE any FLOP claim:** code default `feat_n2=96`→d_φ=128, but the v1 report's
   headline FLOP/recall numbers used d_φ=256. Decide the canonical config, fix code/report/synthesis,
   and **re-emit the FLOP ledger from the actual config** (`flop_ledger.py`).
5. **Add a matched tiny-HYBRID baseline arm (Council-3):** a Samba/GatedDeltaNet-H-style block (mostly
   Prizma layers + ~1 attention layer). Prizma must be at least Pareto-competitive with it, else the
   honest framing is "best pure-O(1) point," not "beats the Transformer."
6. **Mandatory scope rider on every claim:** "≤2M params (+1 confirmation 10–50M), char-LM +
   diagnostics — NOT a frontier / MMLU / NL-long-context claim."
7. **Lever D real gate:** the crosstalk pre-filter PASSED locally, but the binding gate is the
   **end-to-end ≥10-seed MQAR-D128 solve-rate at d_φ=137**, run BEFORE any LM run. Fallback d_φ=168
   (r=16) if the solve point regresses past 130K params.
8. **Lever A (DONE on exactness):** in the A100 ablation prove surprise-TARGETING is causal by beating
   BOTH controls (`surprise_mode='random'` and `'constant'`) — not just "more average write."

## 5. YOUR NEXT TASK — the 4×A100 Colab validation campaign (detailed)
Goal: turn the six implemented levers from "guarded" into **powered, decisive, honestly-scoped wins**.
Drive Colab via Claude-in-Chrome (logged in). Calibrate-then-matrix; crash-safe atomic JSON
checkpoints; stream results back into `results/`. Suggested 4-session split:

- **S1 — char-LM (text8):** TF (grown for the gate params) vs Prizma-v2 (`out_gate=True, state_norm=True,
  decoupled_gate=True, gated=True`). Per-model LR sweep (`sweep_lr`), **≥10 seeds**, then
  `margin_superiority(prizma_bpc, tf_bpc, 0.03)` + `tost_equivalence`. Target: Prizma beats TF by
  **≥0.03 BPC**, p<0.05. (Flips the v1 −0.024 loss.)
- **S2 — recall GATE (run FIRST / highest priority):** MQAR D=128 solve-rate for `none` vs `quad2`
  (d_φ=256) vs `quad2_lowrank` (d_φ=137) vs tuned TF; the flip-test (bigger TF solves); the
  param-efficiency crossover on a DENSE grid. This decides (a) does the lean map D keep the ≥3.5×
  edge, (b) recall TOST-parity vs TF. Also induction + selective-copy with `n_delta=2` (does
  DeltaProduct make them discriminating vs Prizma-none?).
- **S3 — novel-core ablation:** `surprise_gate` `norm` vs `random` vs `constant` controls (Lever A
  causal attribution) AND Lever G (RWKV-7 in-context per-channel LR — implement locally first, see §6)
  vs baseline. Keep whichever wins the "novel core" slot; Council-1 must sign off the causal claim.
- **S4 — efficiency + structure:** banded-window all-n latency (target faster than TF at **all n ≳ 2k**,
  vs the v1 crossover at 32k); FLOP ledger (AFTER R4); constant-memory curve; length-extrapolation
  ABSOLUTE (target ≥0.70 @4×, ≥0.50 @8×); the tiny-hybrid baseline arm.

After the campaign: Council-1 reviews each causal claim; Council-3 judges the combined model vs the
SOTA landscape and sets the Phase-2 bar; then re-run `writing-plans` for Phases 2 (consolidate),
3 (scale-up 10–50M), 4 (report + adversarial referee). Update `docs/PRIZMA_SEQ_REPORT.md`.

Existing GPU runners to adapt (they already do A100 work): `gpu_bench.py` (MQAR), `gpu_diag.py`
(induction/selcopy), `gpu_charlm2.py` (char-LM), `gpu_latency.py`, `gpu_lengen.py`; Colab bootstraps
`PRIZMA_run_*.ipynb` + `PRIZMA_D128_GPU.ipynb`. They must be extended to expose the new v2 config
knobs + use `seq/stats.py` (powered) + `sweep_lr` (per-model/per-combination LR) + crash-safe JSON.
(This extension is Task #12 "Colab A100 runner harness" in the task list.)

## 6. Local work to do WHILE A100 runs (never idle)
- **Lever G** (Task #14 family): RWKV-7-style in-context per-channel learning rate `η_t=σ(W_η x)∈R^{d_h}`
  modulating the delta update per channel; `inctx_lr: bool` config; off=identical (<1e-6); chunked==ref
  (<1e-4); O(1) guard. It competes with Lever A for the novel-core slot.
- **Lever F** (Task #11): Triton fused chunked-delta kernel (CUDA) in `seq/delta_fused.py`,
  equivalence-gated `== chunked_delta < 1e-4`, MPS/CPU fallback = `chunked_delta`. Target train
  ≤1.5× TF. Speed is a Pareto knob, not a correctness gate.
- **R4 d_φ reconciliation** + re-emit `flop_ledger.py`. **Recall-gate runner** + **tiny-hybrid arm**
  (`seq/hybrid.py`). Phase-2 consolidation prep.

## 7. Honest status (do not overstate)
The six levers are **implemented and correctness-guarded**, NOT yet validated as accuracy wins. No
"dramatically beats the Transformer" claim is licensed until the A100 campaign delivers powered
(≥10-seed, TOST/margin) results that clear §4's bar, with the scope rider attached. The constant-
memory and (coarse) parameter-efficiency edges are the only currently-solid wins; char-LM, all-n
latency, and per-FLOP are the targets the campaign must actually win.

## 8. Task list & file map
- Live task list: see TaskList (IDs 1–15). Done: harness, councils R0, gate remedies, levers C/E/H/B/D/A.
  Pending: G, F, Colab runner (#12), the A100 campaign, recall-gate, tiny-hybrid, Phase-1 exit (#13).
- Spec: `docs/superpowers/specs/2026-06-07-prizma-seq-v2-pareto-dominance-design.md` (incl. §11 amendments).
- Plan: `docs/superpowers/plans/2026-06-07-prizma-seq-v2-pareto-dominance.md` (incl. Council-0 amendments).
- Councils: `committee/charters/`, `committee/verdicts/`, `committee/quant/`, `committee/standards/`,
  `committee/round0_v2_synthesis.md`.
- Model: `seq/prizma_seq.py` (mixer + config + O(1) step), `seq/delta.py` (delta kernels),
  `seq/transformer.py` (baseline), `seq/common.py` (harness), `seq/{stats,lrsweep,ledger}.py`.
- Probes/results: `feat_map_probe.py`, `cap_probe.py`, `flop_ledger.py`, `results/`.
