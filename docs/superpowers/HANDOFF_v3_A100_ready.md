# Prizma-Seq v2 — Handoff v3 (local prerequisites DONE → RUN the A100 campaign)

> Written 2026-06-07 for the agent that picks this up. Self-contained — assume NO shared conversation
> context. Supersedes `HANDOFF_v2_A100_campaign.md` for the "what's ready" picture; that file's mission,
> council bar (§4), and integrity rules still bind. Read this top-to-bottom before acting.

---

## 0. OPERATIONAL — READ FIRST
- **The local-prerequisite sprint is COMPLETE.** Every runner the A100 campaign needs is implemented,
  seed-pinned, powered-stats-wired, crash-safe, and reviewed. **Your task is to RUN the 4×A100 Colab
  campaign** and feed results back, then convene the councils for the Phase-1 exit gate.
- **Repo path (IMPORTANT):** the real directory is
  `/Volumes/disk 2/Desktop_Migrate_2026-05-28/Projeler/proje/Prizma` (capital **P**, lowercase rest).
  A `PRISM` symlink may exist but it **dropped once during an external-volume remount** — always target
  the real `Prizma` path in long-running automation, never the symlink. Always quote the space.
- **Branch:** `v2-pareto-dominance` (NOT `main`). **Env:** `python3.13` (bare `python` is absent);
  prefix torch commands with `PYTORCH_ENABLE_MPS_FALLBACK=1` locally; A100 = CUDA on Colab. `timeout` is
  NOT installed on the local mac — don't wrap commands in it.
- **Drive permissions:** when a Google-Drive auth/permission prompt appears (Colab→Drive mount,
  `PRIZMA_RESULTS` persistence, Drive MCP OAuth) the owner has standing instruction to **grant it
  yourself** — do not hand it back to the owner. (Owner's own account; not a 3rd-party data upload.)
- **Never idle / don't ask "should I continue?"** — execute; while a GPU run is in flight do local work
  (Lever F Triton kernel, report drafting, council prep).

---

## 1. What is DONE this sprint (all on `v2-pareto-dominance`, reviewed, 106 tests green)

Subagent-driven-development: each task = implementer → spec-review → quality-review (fix loops), and
**every commit preserved off-path byte-identity** (T8 verified this with SHA256 fingerprints of all 7
config forwards), so the published v1 *Prizma* numbers and all prior results remain valid.

| Commit | What |
|---|---|
| `c9fae02` | **R4 FLOP truth**: `flop_ledger.py` re-emits a per-config ledger; pins every number to its exact `(feat_map, feat_n2/feat_rank, d_φ)`; writes `results/flop_ledger_v2.{json,txt}`; report+synthesis annotated. Canonical d_φ left to the A100 D-gate. |
| `9bde66c` | **Lever G** `inctx_lr`: RWKV-7 per-channel in-context LR through `_delta_reference`+`chunked_delta`+`step()`; off=identical <1e-6, chunked==ref <1e-4 (sequential fallback), step==forward <1e-4. |
| `7a71e65` | **Tiny-hybrid arm** `seq/hybrid.py`: Samba-style mostly-Prizma + 1 attention layer; `hybrid_factory`; param-matched (+0.48% vs TF); Council-3 3rd baseline. |
| `2522fab` | **Recall-gate runner** `seq/recall_gate.py`: MQAR-hard + induction + selcopy; **TOST-parity** verdict + optimization-vs-capacity **flip-test**; `build_and_train` (seed-pinned) + `sweep_lr` + powered `seq.stats`; crash-safe JSON; pure verdict layer unit-tested. |
| `a193bc4`,`239493f` | **v2 campaign harness** `seq/gpu_harness.py` (the keystone): `run_cell` (seed-pinned), `sweep_then_seeds`, `powered_summary`, `h2h`, declarative `make_arm` (all v2 knobs), **identical-model `negative_control`** canary (PASSES p≈0.5), atomic JSON. + `gpu_ablation.py` (S3 novel-core). |
| `f569a20` | **surprise_gen fix**: `surprise_mode='random'` now runs through the model (reproducible generator, `surprise_seed`), so Lever A's random control is usable. Runs on MPS+CUDA. |
| `caaf998` | **S1 char-LM v2** `gpu_charlm2.py --v2`: Prizma-v2 arm (out_gate+state_norm+decoupled_gate+gated) + hybrid arm + param-matched TF; powered BPC verdict (`margin_superiority(0.03)` + TOST, correct lower-is-better sign). Fixed a real cache-key collision (v2 TF was reusing the legacy TF cell). |
| `d76f5e8` | **T8 hardening**: `inctx_lr`+`surprise_gate` mutual-exclusion assert; recall-gate CLI arg-guard (unknown arg → exit 2, never a multi-hour run); refreshed ablation artifact (random runnable); reviewer nits. |

(Levers C/E/H/B/D/A from the prior sprint — `4e513d7`..`56ebe85` — remain in place, default-OFF + guarded.)

---

## 2. YOUR TASK — the 4×A100 Colab campaign (each runner is READY)

Drive Colab via Claude-in-Chrome (owner logged in). Set `PRIZMA_RESULTS` to a Drive-mounted dir for
crash-safe persistence. Every runner is resumable (skips completed cells) so a disconnect never loses
progress. Use **≥10 seeds** for any decisive claim. Suggested 4-session split:

### S1 — char-LM (the LM headline)
```
PRIZMA_RESULTS=<drive> python3 gpu_charlm2.py --v2 --seeds 0 1 2 3 4 5 6 7 8 9 --corpus text8
```
Arms: `Prizma-v2` (gates) vs param-matched `TF-v2` vs `Hybrid-v2`. Gate: `charlm_v2_verdict` →
**BEATS** iff `margin_superiority(prizma_bpc, tf_bpc, 0.03)` p<0.05; else PARITY via TOST. Target: flip
the v1 −0.024 BPC into ≥0.03 BPC win. Output `results/gpu_charlm2.json::v2_verdict`.

### S2 — recall gate (run FIRST; the hard pass/fail for "dominant")
```
PRIZMA_RESULTS=<drive> python3 seq/recall_gate.py --full   # (no-arg also runs full; --smoke = tiny CPU plumbing)
```
Legs: MQAR-hard (MixedMQAR D=128), induction (mixed-len), selective-copy. Arms: TF / Prizma
(`quad2_lowrank` d137 by default) / Hybrid. Each leg must reach **TOST-parity to the tuned TF**, with the
**flip-test** (a bigger TF DOES solve MQAR-hard → tiny-TF failure is under-capacity, not "attention
can't"). `combine_gate` → `dominant` iff all legs pass, else `competitive`. Output
`results/recall_gate.json`. This also settles the **canonical v2 d_φ** (R4): confirm `quad2_lowrank`
d137 keeps the ≥3.5× param-efficiency + recall at ≤1.0× FLOP; back off to d168 if the solve point
regresses past ~130K params.

### S3 — novel-core ablation (which novel core ships)
```
PRIZMA_RESULTS=<drive> python3 gpu_ablation.py --seeds 0 1 2 3 4
```
Arms: `baseline` vs `surprise_norm` / `surprise_random` / `surprise_constant` (Lever A + its TWO
mandatory R9 controls — all now runnable) vs `inctx_lr` (Lever G). Holm-corrected `h2h` vs baseline +
the `negative_control`. **Lever A's causal claim requires surprise_norm to beat BOTH controls**; keep
whichever of {surprise, inctx_lr} wins the slot (remember: they're mutually exclusive now). Output
`results/gpu_ablation.json::s3_report`.

### S4 — efficiency / structure (measurements; inherently on-GPU)
```
PRIZMA_RESULTS=<drive> python3 gpu_latency.py    # banded_window=True; target faster than TF at all n >~ 2k
PRIZMA_RESULTS=<drive> python3 gpu_lengen.py     # length-extrapolation (target >=0.70 @4x, >=0.50 @8x)
python3 flop_ledger.py                            # already analytical + per-config (results/flop_ledger_v2.json)
```
Constant-memory curve + the FLOP ledger are already in hand; latency/length-extrap need the A100.

---

## 3. The ONE remaining build (do on the A100, where it's testable)
- **Lever F — fused chunked-delta Triton kernel** `seq/delta_fused.py` (Task 1.F). Deliberately NOT
  built locally: a Triton kernel can only be equivalence-tested against real CUDA
  (`fused_delta == chunked_delta < 1e-4`, fwd+grad). Build it on Colab, gate on equivalence, target
  train ≤1.5× TF. Speed is a Pareto knob, not a correctness gate; document the gap honestly if Triton
  parity is hard. MPS/CPU fallback = `return chunked_delta(...)`.

---

## 4. The bar every claim must clear (unchanged; from HANDOFF_v2 §4 + council synthesis)
- Powered stats ONLY (`seq/stats.py`): ≥10 seeds for decisive claims; solve-rate + median + CI;
  superiority/TOST/margin; **Holm-correct** the causal-attribution family; the **identical-model negative
  control** must come back NOT-significant (it does — built into `gpu_harness`).
- Recall is a **hard TOST-parity gate** (S2). per-FLOP ≤1.0× stays **conditional** until D+E(+F) deliver
  it AND all axes hold simultaneously+powered; else downgrade "dramatic" → "Pareto-competitive in the
  tested regime."
- **Mandatory scope rider on every claim:** "≤2M params (+1 confirmation 10–50M), char-LM + diagnostics
  — NOT a frontier / MMLU / NL-long-context claim."
- No faked metrics; `step()==forward()<1e-4` O(1) guard before any accuracy number; honest
  borrowed-vs-new ledger.

## 5. After the campaign
Council-1 reviews each causal claim; Council-3 judges the combined model vs the SOTA landscape and sets
the Phase-2 bar; then re-run `writing-plans` for Phases 2 (consolidate), 3 (scale-up 10–50M), 4 (report +
adversarial referee). Update `docs/PRIZMA_SEQ_REPORT.md`.
