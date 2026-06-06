# PRISM-Seq → Transformer-Alternative — SESSION HANDOFF

> **Read this first, then `committee/round1_synthesis.md` and `docs/TRANSFORMER_ALTERNATIVE_BRIEF.md`.**
> A fresh session can resume from this file alone. Date of handoff: 2026-06-04.

---

## 0. TL;DR — where we are
- **Goal (locked by owner):** push PRISM-Seq to be a *credible Transformer alternative*, proven on the
  field-standard attention-diagnostic bar, with the **most aggressive** target: genuinely **solve MQAR
  D=128** at a scale where the Transformer is a strong baseline. Big architectural changes allowed,
  multi-session compute OK. **Non-negotiables:** no faked metrics; param/FLOP-matched, non-strawman
  Transformer; preserve a **measured O(1)/constant-memory inference** advantage.
- **Scientific core WORKS:** a **parameter-free quadratic feature map** (`feat_map="quad2"`) makes the
  Gated-DeltaNet carried state *rectangular* (`S ∈ R^{d_h×d_φ}`), raising associative-recall key-rank
  at **zero added params** with **O(1) inference intact** (verified step==forward <1e-6).
- **Seed-0 directional results (matched params, mixed-D training, fair protocol):**
  - **D=64:** Transformer **1.000**, PRISM-quad2 **1.000**, PRISM-none 0.931 → *parity*.
  - **D=128:** tiny matched TF **0.016** (can't learn at this scale), PRISM-quad2-256 **0.999**,
    PRISM-none 0.369. + flip-test confirms tiny TF stays flat even at constant high LR / 120k steps.
- **Committee R2 (adversarial) verdict = `NEEDS-EXPERIMENTS-FIRST`.** The D=128 claim is NOT yet
  publishable. Two serious gaps: (1) *optimization-vs-capacity* — must show a **bigger** TF DOES solve
  D=128 (so the matched-tiny failure is "under-capacity at this size", not "attention can't"); + multi-seed.
  (2) **FLOP gap undisclosed** — PRISM-quad2-256 = **2.1–2.7× TF FLOPs/token** (rectangular state =
  buffers, 0 params but ~5.5× delta-FLOP). Need FLOP ledger + a FLOP-matched TF arm.
- **NEXT: run `gpu_bench.py` on a CUDA GPU (Colab)** — it does all the rigor (scale search, ≥5-seed
  matched + FLOP-matched head-to-head, D-frontier, ablations incl. `rand_linear` control, O(1)
  latency/memory). The notebook `PRISM_D128_GPU.ipynb` is built & ready. Then update the report and run
  the final adversarial referee panel.

## 1. Honest claim wording (committee-vetted; do NOT exceed this until experiments land)
> "At matched ~130K params (2-layer, d=64), under a single fair protocol and seed 0, a parameter-free
> quadratic feature map making a Gated-DeltaNet's carried state rectangular (d_φ=256) **solved MQAR
> D=128 (0.999) where the param-matched, under-capacity tiny Transformer did not (0.016 in this run)**,
> while preserving an O(1)-in-sequence-length constant-memory inference advantage. Small-scale,
> single-seed, directional; borrows the Based/Hedgehog feature-map family (novelty = the
> **rectangular-delta-state framing**, not the kernel); it does **NOT** show attention cannot do recall
> — confirming that needs the bigger-TF D=128 control + ≥3–5 seeds with CIs + a FLOP-matched comparison."

## 2. Environment & how to run
- **Local (Mac):** use system Python `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13`
  (torch 2.12 + MPS). The project `.venv` LOST torch — do NOT use it. Always run with
  `PYTHONPATH=/Users/nazmi/Desktop/Projeler/proje/PRISM`. PRISM is ~4–5× slower than the TF per step on MPS.
- **GPU (Colab):** owner has credits for hours of A100/L4. The benchmark targets CUDA via `gpu_bench.py`.
- **Browser/Colab automation:** `chrome-devtools` and `playwright` MCPs both launch an *isolated*
  automation browser → **Google blocks login** ("browser may not be secure"). To drive Colab you need
  the **"Claude for Chrome" extension** connected to the session (drives the user's real, logged-in
  Chrome) — NOT available in the session that wrote this handoff. Otherwise the **owner runs the
  notebook** (one upload + Run all) and we read results back from Drive.
- **Google Drive MCP works** (`mcp__claude_ai_Google_Drive__*`): can list/search/create files and
  read back results. The owner's "Colab Notebooks" Drive folder id = `1EuDxroP_fyv8wzsiXb3FRCYjahq5Oe6j`.

## 3. Architecture (`seq/`)
- **`prism_seq.py`** — PRISM-Seq = Gated-DeltaNet-family mixer: per-head carried state `S` (precision-gated
  delta rule, L2 keys), read `o = S·q` (pre-write), + short causal depthwise conv (k=4) + small exact
  local-window attention head (w=16). RoPE OFF on delta keys. **Lever:** `feat_map ∈ {none, quad2,
  rand_linear}`, `feat_n2` (d_φ = d_h + feat_n2). `quad2` = fixed random quadratic monomials (seeded
  BUFFERS → 0 params) → rectangular state. `rand_linear` = fixed random linear map d_h→d_φ (CONTROL:
  rank≤d_h → expected NO gain; proves the monomials are causal). Exact O(1) `step()` path
  (verified ==forward <1e-6 for none/quad2/rand_linear, all params identical).
- **`delta.py`** — chunked WY/UT delta kernel; made **value-dim-aware** so the state can be rectangular
  `S∈R^{d_v×d_k}` (byte-identical when d_v==d_k). Self-test (`python -m seq.delta`) covers square +
  rectangular cases, CPU+MPS, `ALL OK`.
- **`transformer.py`** — clean Llama-style decoder baseline (causal SDPA, RoPE, RMSNorm, SwiGLU, tied
  head); **verified non-strawman**. KV-cached O(t) decode.
- **`tasks.py`** — `MQAR` (Zoology-standard, disjoint key/value ranges, dense queries) and **`MixedMQAR`**
  (samples #pairs ~U[1,max] per batch → high-D becomes learnable; eval fixed at target D). This mixed-D
  training is REQUIRED: fixed-high-D training stalls at chance for both architectures.
- **`common.py`** — the **fair protocol**: `MixedMQAR` + `train_model` with (a) **frozen reproducible eval
  set** (eval_seed, same batches for all models/LRs/seeds), (b) **relative per-model PLATEAU early-stop**
  with an **engagement floor=0.5** (don't cut a model off during MQAR's flat pre-transition), (c)
  generous warmup. `TrainConfig` fields: lr, warmup, warmup_frac, cosine, min_lr_frac, plateau_delta,
  plateau_floor, min_steps, early_stop_patience, eval_seed, eval_batches.

## 4. The fair protocol — hard-won lessons (do NOT regress)
1. **Train each model to its own PLATEAU** (large cap + plateau-stop). Under-budgeting was the #1
   false-fail (the TF's MQAR transition is late+sharp; D=64 ignites ~step 8k).
2. **Per-model optimizer tuning** (LR *and* warmup): the TF needs **gen-warm** (lr=1e-3, warmup=2000);
   short "punchy" warmup destabilizes it (loss oscillates 0.8↔4.5). PRISM is robust to both. "Identical
   protocol" = same task/budget/eval/plateau-rule; NOT identical optimizer constants.
3. **Mixed-D training** (MixedMQAR) is mandatory for high-D learnability.
4. **Engagement floor on early-stop** (don't stop a model that's still at chance pre-transition).
5. **≥5 seeds + solve-rate + CI** for decisive rungs (both models are seed-bimodal on MQAR).
6. **Measured FLOP ledger** in every result (param-match ≠ FLOP-match).

## 5. Results so far (all SEED 0, directional; files in `results/`)
- `parity_d64.json`: D=64 gen-warm — Transformer 1.000, **PRISM-quad2 0.99951**, PRISM-none 0.931.
- `d128_sweep.json`: D=128 gen-warm — Transformer **0.016**, **PRISM-quad2-256 0.99947**, PRISM-none 0.369.
- `tf_verify_d128.json` / `.log`: flip-test — matched tiny TF stays flat (~0.016) at D=128 under
  constant lr=1e-3 (120k) and lr=2e-3; `big_d128L2H4` arm may still be running.
- `flop_ledger.py` output: PRISM-quad2-256 / TF forward-FLOP ratio = **2.14× (ideal) – 2.67× (as-coded)**;
  the as-coded window head is an unoptimized full-T² SDPA (20% — can be banded to ~1%).
- Earlier scans (`tf_frontier`, `scale_frontier`, `mixed_test`, `tf_stabilize`, `tf_long`, `calib_budget`,
  `cap_probe`) document the diagnosis path (tiny TF solves D=16/D=64 but not D=128; fixed-D unlearnable).

## 6. Committee
- **R1** (`committee/round1_synthesis.md`): fairness audit + recall-capacity theory (matched D*≈24–32;
  D=128 needs architectural change) + ranked levers: **#1 quad2 feature map (built)**, #2 decoupled
  `d_state` (NOT built), #3 GlobalDeltaMemory (NOT built), #4 bounded global-attention hybrid (last
  resort, spends O(1)). + 10 honesty guardrails.
- **R2** (adversarial verify, in the transcript): verdict `needs-experiments-first`. **CLEARED:** no
  leakage; eval fair (TF scores 0.016 on the same frozen set quad2 scores 0.999 on); window-head NOT
  causal (none has it, plateaus 0.369); quad2 truly param-free; O(1) verified. **GAPS:** optim-vs-capacity
  + FLOP disclosure (see §0).

## 7. GPU plan — `gpu_bench.py` (CUDA) + `PRISM_D128_GPU.ipynb`
`gpu_bench.py` phases (resumable; writes `$PRISM_RESULTS/gpu_bench.json` incrementally):
1. **TF D=128 solving-scale search** {d64L2H2, d128L2H4, d128L4H4, d256L4H8} × lr × seeds → find the
   smallest TF that SOLVES D=128 (the fair arena + the flip-test).
2. **Matched + FLOP-comparable head-to-head @ D=128** at that scale, ≥5 seeds → solve-rate + median + CI.
3. **D-frontier {16,32,64,128,256}** capacity curve.
4. **Ablations @ D=128:** quad2 vs none vs **rand_linear** control; window on/off.
5. **Measured O(1) decode latency + memory vs sequence length.**

`PRISM_D128_GPU.ipynb` (at repo root, 91KB, 19 cells, validated) self-contains everything: writes the
verified `seq/*` + `gpu_bench.py` + `flop_ledger.py` via `%%writefile`, mounts Drive, runs kernel
self-tests, the FLOP ledger, then all phases; saves to `Drive/MyDrive/prism_results/gpu_bench.json` and
prints a `===RESULTS_JSON===` block. Regenerate with `python3.13 build_notebook.py`.

**To run on Colab:** owner opens it in their logged-in Chrome → File→Upload notebook →
`/Users/nazmi/Desktop/Projeler/proje/PRISM/PRISM_D128_GPU.ipynb` → Runtime=A100/L4 → Run all → approve
Drive mount. (If the session has the "Claude for Chrome" extension, the assistant can drive this directly.)

## 8. Read results back (assistant, autonomously)
- Poll Drive: `Google_Drive search_files` query `title contains 'gpu_bench' ` (or list the prism_results
  folder), then read it. NOTE `read_file_content` mime support is limited (docs/sheets/pdf/images, not raw
  json) — use `download_file_content` if available, else have the owner paste the printed `===RESULTS_JSON===`.

## 9. Running jobs at handoff
- **Mac `bm7tyb93q`** (`tf_verify_d128.py`): flip-test still running (constLR2e-3 ~step 42k = 0.016, then
  big_d128L2H4 arm). Complementary small-scale data. Safe to let finish or `pkill -f tf_verify_d128.py`.
- An isolated chrome-devtools browser is open at a Google login page (useless; ignore/close).

## 10. EXACT NEXT STEPS (prioritized)
1. **Get the GPU run done** (Colab): drive the browser if "Claude for Chrome" is connected; else confirm
   the owner ran the notebook. Read `gpu_bench.json` back from Drive.
2. **Adjudicate the flip-test from Phase 1:** does a bigger TF SOLVE D=128? If yes → honest framing =
   "matched-tiny TF under-capacity; PRISM-quad2 solves at matched params + O(1) + far cheaper than scaling
   attention". If even big TFs fail → investigate (task realism) before any claim.
3. **Phase 2 (≥5-seed matched + FLOP-matched) is the headline.** Report solve-rate + median + 95% CI.
4. **Update `docs/PRISM_SEQ_REPORT.md`** with the real numbers (it still has `_pending_` rows), the FLOP
   ledger, the O(1) curve, and the honest claim wording (§1). Add quad2 to the borrowed-vs-new ledger
   (Based/Hedgehog kernel family; novelty = rectangular-delta-state framing).
5. **Final adversarial referee panel** (a committee Workflow) against the §4 bar of the brief.
6. If quad2 plateaus below target at some D: implement **lever #2 `d_state`** (decouple per-head state
   dim from d_model; delta.py already handles it) and/or **#3 GlobalDeltaMemory**; keep #4 hybrid as the
   explicit O(1)-trading Pareto variant only if needed.

## 11. Honesty guardrails (binding)
Param-match every head-to-head (grow the TF for any paid PRISM arm); ship a measured FLOP+throughput
ledger; disclose the O(1) state-constant growth (d_h·d_φ floats) but note it's constant in n and ≪ the
TF's O(t) KV-cache; keep the step==forward<1e-6 O(1) guard green before reporting any accuracy; multi-seed
+ solve-rate; pre-register D* per new arm; verify BOTH sides of any contrast; never say "attention fails"
(say "did not learn under recipe R at this scale"); quad2 novelty = framing not kernel; persist crash-safe
checkpoints. Memory file: `~/.claude/projects/-Users-nazmi/memory/prism-seq-transformer-alternative.md`.
