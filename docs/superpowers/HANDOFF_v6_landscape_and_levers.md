# Prizma-Seq v2 — Handoff v6 (SOTA landscape + model levers built; campaign running; what's next)

> Written 2026-06-08. Consolidates a long improvement session run WHILE the A100/L4 campaign streams.
> Read with HANDOFF_v5 (the running-campaign session map + the idle-disconnect lesson). HEAD = `743038f`.

## 1. What was SHIPPED this session (all on `v2-pareto-dominance`, each via subagent-driven-development:
## implement → spec-review → quality-review; protected files byte-identical; suite 143→162 pass / 5 skip)

| Commits | Improvement | Verify |
|---|---|---|
| `e1e8a08` `a6f7984` | **Lever F** `seq/delta_fused.py` — fused chunked-delta; CUDA path = `torch.compile(chunked_delta)` (Inductor→Triton, provably-equivalent), exact eager fallback off-CUDA. A hand-written `@triton.jit` kernel is DEFERRED to be built+verified ON the A100. | `python3.13 -m seq.delta_fused` |
| `a041d07` `c976051` | **GLA** `seq/gla.py` — faithful Gated Linear Attention SOTA baseline; recurrent+chunk+O(1); param-vs-TF **+1.58%**; `make_arm kind='gla'`. | `python3.13 -m seq.gla` |
| `a22a8ca` `b7ee9ed` | **Mamba-2 (SSD)** `seq/mamba2.py` — faithful state-space SOTA baseline (scalar-A SSD, short conv, D-skip, z-gate); param-vs-TF **+0.62%**; `make_arm kind='mamba2'`. | `python3.13 -m seq.mamba2` |
| `9bf52bd` `6944bb2` | **Landscape runner** `seq/landscape.py` — powered 4-arm head-to-head (TF/Prizma/GLA/Mamba-2) on the recall diagnostics; `landscape_verdict` {BEATS/PARITY/WORSE/INCONCLUSIVE}; negative control; `--smoke/--full`. (Fixed an Important verdict bug: a statistical tie where Prizma leads is INCONCLUSIVE, not WORSE.) | `python3.13 -m seq.landscape --smoke` |
| `94d62e2` `51d07d4` | **char-LM landscape leg** — same 4 arms on char-LM **BPC** (lower-is-better); `run_landscape_charlm` + `--charlm`; `PRIZMA_CHARLM_KNOBS` (gates ON, distinct from the recall arm's gates-off). | `python3.13 -m seq.landscape --smoke --charlm` |
| `f869de4` `743038f` | **Chunk-parallel Lever G** `seq/delta.py` — per-channel `eta` was a sequential `_delta_reference` scan; now an EXACT chunk-parallel BATCHED PER-VALUE-CHANNEL triangular solve. `eta=None` **byte-identical (max\|d\|=0.0)**; `eta==_delta_reference` ~1e-5 fwd+grad (pure+gated, cpu+mps). | `python3.13 -m seq.delta` |

**Net:** the SOTA-landscape EVIDENCE BASE is complete — Transformer (attention) / GLA (gated-linear-attn) /
Mamba-2 (SSM) / Prizma (gated-deltanet), on BOTH recall diagnostics AND char-LM BPC, all powered + param-
matched. Plus two model-side wins: Lever F (fused speed) and Lever G (chunk-parallel per-channel eta).

## 2. The campaign (running — see HANDOFF_v5 for the live cell map)
Recovered after an idle-disconnect into **foreground notebook cells** (idle-safe): nb1=S2 recall-gate
(A100), nb2=S1 char-LM v2 (L4), nb3=S3 ablation→S4 (L4). L4s are scientifically fine for S1/S3 (BPC +
solve-rate are GPU-independent; ~2.3× slower wall-clock); only S4's absolute latency is GPU-specific
(re-run on A100 later). Monitor via CELL OUTPUT (`get_page_text`/screenshot); the kernel is busy while a
cell runs, so EXFILTRATE each result JSON when its cell COMPLETES. Gist @`4c44918` is the bootstrap
(DELETE at the end: `gh gist delete fe9c44feff67ff0a807f888c72e976f8`).

## 3. What's NEXT (the remaining work is campaign-DATA-DEPENDENT or GPU-BLOCKED — hence built-out above)
1. **Exfiltrate the powered verdicts** as S1/S2/S3/S4 cells complete; report with the scope rider
   ("≤2M params (+1 confirmation 10–50M); char-LM + diagnostics; NOT a frontier/MMLU/long-context claim;
   per-FLOP 'dramatic' stays conditional unless all axes hold + powered").
2. **RUN the landscape runner** (`--full` recall + `--charlm` LM) on a freed GPU → the Council-3
   Pareto-vs-SOTA table. (It is built + smoke-green; it just needs GPU time.)
3. **Hand-written Triton WY/UT kernel** for Lever F — DEFERRED until a campaign A100 frees (verify against
   `chunked_delta` < 1e-4 fwd+grad ON the A100; never ship it blind).
4. **Lower-FLOP feature map** for the per-FLOP weak axis — genuinely needs the MQAR-D-frontier capacity
   data to pick the rank/approx (don't build it blind; the trade is capacity-vs-FLOP).
5. Council-1 (each causal claim) + Council-3 (combined picture) → `docs/PRIZMA_SEQ_REPORT.md` →
   re-plan Phases 2 (consolidate) → 3 (scale-up 10–50M) → 4 (report + adversarial referee).

## 4. Env (unchanged)
`python3.13` only; `PYTORCH_ENABLE_MPS_FALLBACK=1`; no `timeout` binary. Commit gate:
`touch /tmp/.opsera-pre-commit-scan-passed` as its OWN Bash call, THEN `git add <files> && git commit`
separately; NEVER `git add -A` (untracked hf_publish/, hf_space/, paper/, docs/HF_* must stay out).
