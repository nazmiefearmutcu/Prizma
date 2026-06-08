# Prizma-Seq v2 — Handoff v7 (RESUME the running campaign + finish it)

> Written 2026-06-08 ~mid-campaign. SELF-CONTAINED — assume NO shared conversation context. This is the
> authoritative "what is running, how to monitor/exfiltrate, how to migrate, what's left" doc. Read with
> HANDOFF_v5 (idle-disconnect lesson) and HANDOFF_v6 (the 7 shipped improvements). HEAD = `6267392`,
> branch `v2-pareto-dominance` (NOT main).

---

## 0. THE ONE-PARAGRAPH PICTURE
A 4-session honest-science campaign (recall-gate / char-LM / ablation / latency) is RUNNING on Colab as
**foreground notebook cells** (idle-safe) on account `nazmiefearmutcu@gmail.com` (Colab Pro+). S2 (recall)
is on an **A100** and ~3–5 h from completing; S1 (char-LM) and S3 (ablation) are on **L4** GPUs (A100s were
scarce at launch) and are **TOO SLOW to finish within Colab's ~24 h max** at L4 speed. The plan: when S2
completes and frees its A100, **migrate S1 onto it** (lossless, crash-safe JSON resume) so the char-LM
verdict can finish ~5× faster; S3 may end up partial unless a 2nd A100 is obtained. Results so far: Prizma
does NOT reach powered TOST-parity with the tuned Transformer on MQAR-hard or induction (honest, not a
win). All monitoring is via reading the **cell output** (the kernel is busy while a cell runs, so
exfiltrate each result JSON when its cell COMPLETES).

---

## 1. REGAIN CONTROL OF THE BROWSER (do this first in a new session)
The campaign runs in Chrome via the **claude-in-chrome** extension. Tab IDs are session-specific and will
NOT carry over. To reconnect:
1. `list_connected_browsers` → if the owner's Chrome ("here!" = the `nazmiefearmutcu@gmail.com` account) is
   listed, `select_browser` its deviceId; else `switch_browser` (broadcasts a Connect prompt; owner clicks
   Connect in the correct-account Chrome). MANDATORY: present the browser list via AskUserQuestion first.
2. `tabs_context_mcp` to get the current tab IDs, then open/identify the three campaign notebooks by their
   **stable Drive URLs**:
   - **S2 recall-gate (A100):** https://colab.research.google.com/drive/1M1rllmgqND6jtD4fab--gxJSOFepBR5G
   - **S1 char-LM (L4):**       https://colab.research.google.com/drive/17af-btTgFFgOHQMZDicIOGjhBxkJ2QA0
   - **S3 ablation→S4 (L4):**   https://colab.research.google.com/drive/1VLhgp9vFiqhOcOdUXuFJWm5U5Bfg-K7V
3. Each notebook has ONE foreground code cell running its workload (status bar bottom-right shows
   "Executing (Nh Nm Ns)"). Monitor by `computer screenshot` or `get_page_text` on the tab. DO NOT re-run
   or interrupt a healthy cell.

---

## 2. WHAT EACH SESSION IS RUNNING (the cell command + the result JSON)
Each cell was launched as: `!mkdir -p /content/prizma && curl -sL <GIST> | base64 -d | tar xz -C
/content/prizma && cd /content/prizma && python3 -u <script> 2>&1 | tee <log>`. GIST raw URL:
`https://gist.githubusercontent.com/nazmiefearmutcu/fe9c44feff67ff0a807f888c72e976f8/raw/264408a24c469d06557991a7e47c2beb49ef3784/prizma_v2_b64.txt`
(code @ commit `4c44918`; DELETE at the very end: `gh gist delete fe9c44feff67ff0a807f888c72e976f8`).

| Session | Script | Result JSON | Notes |
|---|---|---|---|
| **S2** | `python3 -u seq/recall_gate.py --full` | `results/recall_gate.json` | A100; ~3–5h left (on SELECTIVE-COPY leg) |
| **S1** | `python3 -u gpu_charlm2.py --v2 --seeds 0 1 2 3 4 5 6 7 8 9` | `results/gpu_charlm2.json` | L4; TF-v2 arm done (10 seeds cached), Prizma-v2 ~5× slower on L4 (~2.2h/run) → unfinishable on L4 |
| **S3** | `python3 -u gpu_ablation.py` (then `gpu_latency.py`, `gpu_lengen.py` chained) | `results/gpu_ablation.json` (+ `gpu_latency.json`, `gpu_lengen.json`) | L4; baseline arm done, surprise_norm arm sweeping (~6h/arm × 6 arms) → unfinishable on L4 |

---

## 3. RESULTS SO FAR — report HONESTLY, no spin (scope rider mandatory)
Scope rider on EVERY claim: "≤2M params (+1 confirmation 10–50M); char-LM + diagnostics; NOT a
frontier/MMLU/long-context claim; per-FLOP 'dramatic' stays conditional unless all axes hold + powered."

**S2 recall-gate (d128L2H4, 10 seeds, A100) — 2 of 3 leg verdicts in:**
- **MQAR-HARD:** `leg_pass=False parity=False flip_solved=True` (delta=+0.0398, ci90=(-0.1954, 0.275),
  not_worse=False). Best-acc: TF≈1.000 vs Prizma≈0.997 — Prizma a hair behind; NO powered parity.
- **INDUCTION:** `leg_pass=False parity=False flip_solved=True` (delta=+0.2845, ci90=(-0.0563, 0.6252),
  not_worse=False). Best-acc: TF **bimodal** (~0.06 or ~0.998 by seed, mean≈0.65) vs Prizma consistent
  ≈1.000. Prizma leads on the mean BUT the ci90 crosses 0 (TF's seed-variance is huge), so NO powered
  win and NO parity. (The delta sign maps to the gate's internal acc metric, not raw best-acc — report
  the flags + raw accuracies, do not over-read the sign.)
- **SELECTIVE-COPY:** in progress (TF arm solving it trivially ≈1.0).
- HONEST headline: **Prizma does NOT reach powered TOST-parity with the tuned Transformer on the recall
  diagnostics so far.** Not a win. Wait for the SELCOPY leg verdict + the full JSON before any summary.

S1/S3 have produced only partial sub-results (S1: TF-v2 BPC seeds ~1.71–1.77; S3: baseline arm
mean=0.749, solve=0.60) — no finalized verdicts yet.

---

## 4. THE MONITOR/EXFILTRATE LOOP (every ~28–30 min)
Screenshot/get_page_text each of the 3 cells. For each:
- **Still "Executing" + streaming output** → healthy, do nothing.
- **Cell STOPPED (kernel free, `[1]` shows a finished run-time + the final powered verdict in its
  output)** → it COMPLETED. EXFILTRATE: in a NEW cell on that notebook run
  `!base64 -w0 /content/prizma/results/<name>.json`, `get_page_text` the output, base64-decode it, and
  save to `/Volumes/disk 2/Desktop_Migrate_2026-05-28/Projeler/proje/Prizma/results/campaign_2026-06-08/<name>.json`.
  Then report the FULL powered verdict with the scope rider, HONESTLY.
- **Runtime DISCONNECTED** ("Reconnect" / "Runtime disconnected") → reconnect, then RELAUNCH IN A
  FOREGROUND CELL (NOT terminal+nohup — see HANDOFF_v5 idle lesson), FIRST restoring that session's last
  exfiltrated JSON into `/content/prizma/results/` (echo its base64 through `base64 -d`) so the crash-safe
  runner RESUMES (skips cached cells) instead of restarting.
- ⚠️ NEVER stop an L4 cell before exfiltrating its JSON — a stopped kernel goes idle → the VM is torn down
  → the ephemeral JSON (no Drive) is LOST.

---

## 5. THE S1 MIGRATION (do this when S2 completes and frees its A100)
1. Exfiltrate + report S2's `recall_gate.json` (§4).
2. The A100 (S2's notebook) is now free. MIGRATE S1 there (lossless — gpu_charlm2.json has the 10 TF-v2
   seeds cached):
   a. In S1's L4 notebook: **stop** the running cell (Runtime → Interrupt). Then in a NEW cell:
      `!base64 -w0 /content/prizma/results/gpu_charlm2.json` → get_page_text → save the base64 string.
   b. In S2's now-free A100 notebook, a FRESH FOREGROUND cell (one line, no quotes/parens so Monaco
      won't auto-close): re-bootstrap from the gist, recreate `/content/prizma/results/gpu_charlm2.json`
      by piping the saved base64 through `base64 -d`, THEN
      `cd /content/prizma && python3 -u gpu_charlm2.py --v2 --seeds 0 1 2 3 4 5 6 7 8 9 2>&1 | tee logs_s1.txt`.
      It RESUMES (skips the cached TF-v2 seeds) and runs Prizma-v2 + Hybrid-v2 at ~5× the L4 speed (~5–6 h).
3. S3 (ablation) stays on L4 (likely partial) unless a 2nd A100 is obtained — then migrate it the same way
   (its `gpu_ablation.json` has the baseline arm cached).

**Speed-vs-safety decision the owner is weighing** (timeline they asked about): (A) try to grab fresh
A100s NOW for S1/S3 (faster ~6–8h total, small risk: A100 may be unavailable → revert to L4, lose an
in-progress LR-sweep run); (B) safe plan above (~15–20h, S3-bottlenecked); (C) drop S1/S3, run the full
4-arm `seq.landscape --full`+`--charlm` on S2's freed A100 instead (fastest clean Council-3 Pareto table).

---

## 6. AFTER THE CAMPAIGN (the remaining honest-science work)
1. Report all powered verdicts with the scope rider: S2 recall (per-leg TOST-parity + flip-test),
   S1 char-LM BPC (margin_superiority+TOST, param-matched), S3 ablation (surprise_norm must beat BOTH
   controls; inctx_lr vs baseline; identical-model negative-control PASS), S4 latency crossover +
   length-extrap + O(1) (NOTE: S4 latency is GPU-specific — re-run on A100 for the canonical number).
2. Run the SOTA-landscape on a freed A100: `python3 -m seq.landscape --full` and `--charlm`, then render
   with `python3 -m seq.landscape_report` → the Council-3 Pareto table (TF/Prizma/GLA/Mamba-2 × recall +
   char-LM). (This is BUILT + smoke-validated; it just needs GPU time.)
3. DEFERRED GPU build: a hand-written `@triton.jit` WY/UT kernel for Lever F — develop+verify ON the A100
   against `chunked_delta` (==<1e-4 fwd+grad) before trusting it. Never ship blind.
4. DATA-DEPENDENT: a lower-FLOP feature map for the per-FLOP weak axis — needs the MQAR-D capacity data to
   pick the rank/approx (don't build blind).
5. Council-1 (each causal claim) + Council-3 (combined picture) → update `docs/PRIZMA_SEQ_REPORT.md`
   "Results vs the bar" → re-run writing-plans for Phases 2–4 → DELETE the gist.

---

## 7. SHIPPED THIS SESSION (HEAD `6267392`; all subagent-reviewed; suite 162 pass / 5 skip)
Lever F `e1e8a08`+`a6f7984` (`seq/delta_fused.py`, torch.compile→Triton fused, eager fallback) · GLA
`a041d07`+`c976051` (`seq/gla.py`, +1.58%) · Mamba-2 `a22a8ca`+`b7ee9ed` (`seq/mamba2.py`, +0.62%) ·
landscape runner `9bf52bd`+`6944bb2` + char-LM leg `94d62e2`+`51d07d4` (`seq/landscape.py`, recall+BPC,
INCONCLUSIVE verdict bucket) · chunk-parallel Lever G `f869de4`+`743038f` (`seq/delta.py` eta = exact
batched per-channel triangular solve, eta=None byte-identical 0.0) · landscape report `3a0ff14`+`37add47`
(`seq/landscape_report.py`). Docs: v5 `6f207f6`, v6 `dffaefb`, report methods `6267392`.

## 8. ENV + GATES (unchanged)
`python3.13` only (bare `python` absent); prefix torch cmds `PYTORCH_ENABLE_MPS_FALLBACK=1`; no `timeout`
binary on the mac. Commit gate: `touch /tmp/.opsera-pre-commit-scan-passed` as its OWN Bash call, THEN
`git add <files> && git commit` as a SEPARATE call; NEVER `git add -A` (untracked hf_publish/, hf_space/,
paper/, docs/HF_*, results artifacts must stay out — stage only files you changed). Repo:
`/Volumes/disk 2/Desktop_Migrate_2026-05-28/Projeler/proje/Prizma` (capital P; quote the space; the
`PRISM` symlink can drop on remount — target the real dir).
