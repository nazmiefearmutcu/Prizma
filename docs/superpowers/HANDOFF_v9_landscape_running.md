# HANDOFF v9 — Prizma-Seq v2 campaign: core verdicts LANDED, 4-arm SOTA landscape RUNNING

**Date:** 2026-06-09 · **Branch:** `v2-pareto-dominance` · **HEAD:** `df395f0`
**Repo (load-bearing, quote the space):** `/Volumes/disk 2/Desktop_Migrate_2026-05-28/Projeler/proje/Prizma`
(`PRISM` is a drop-prone symlink — always target the real `Prizma` dir.)

This is a self-contained resume doc. Read it fully before touching anything.

---

## 1. TL;DR — where things stand

The Prizma-Seq v2 Colab campaign's **two core scientific verdicts are DONE, committed, and folded into the report — both honestly "competitive" (NOT a win, NOT a tested parity).** Two new model levers shipped. The only remaining piece is the **4-arm SOTA landscape** (adds GLA + Mamba-2 to the TF-vs-Prizma comparison), which was **running on Colab A100(s)** but is a fragile multi-session marathon blocked on **Drive-backing** (a manual user step). Core results are safe regardless of what happens to the landscape.

| Leg | Verdict | Commit | Artifact |
|---|---|---|---|
| **Recall (S2)** | GATE NOT MET on all 3 legs → **"competitive"** (no TOST-parity vs tuned TF) | `04c39f9` | `results/campaign_2026-06-08/recall_gate.json` (30948B) |
| **char-LM (S1)** | n=7 Prizma vs n=10 TF, Δ=**+0.0205 BPC** (Welch t≈6.7 — real but within the ±0.05 B4 bar) → **"competitive"** | `df395f0` | `results/campaign_2026-06-08/charlm_partial_n7.json` |
| **Triton kernel** | DRAFT, hardened by 28-agent review (4 bugs fixed); **PENDING A100 verify** | `f2fa0a0` | `seq/delta_triton.py` + `docs/superpowers/specs/triton_kernel_a100_checklist.md` |
| **opt-in dropout** | core lever, default 0.0 **byte-identical**; 203 tests | `16cdfdb` | `seq/prizma_seq.py` |
| **4-arm landscape** | **PENDING** — running on A100(s), Drive-blocked | — | `results/gpu_landscape.json` (recall), `results/gpu_landscape_charlm.json` (char-LM) |

Honest one-liners (use verbatim, no spin):
- **Recall:** Prizma is *competitive* with the tuned Transformer on MQAR-hard / induction / selective-copy — its mean+solve-rate are ≥ TF and it is the more *reliable* arm (TF induction is bimodal), but NO leg clears powered TOST-equivalence at ±0.05. Not a win.
- **char-LM:** Prizma is *competitive within the ±0.05 B4 bar* on text8 BPC — **reliably ~0.02 BPC behind** the tuned TF (a real gap, t≈6.7), not as good, but within tolerance. n=7 (disconnect-truncated at Prizma seed 7); the 7 seeds are extremely tight (sd 0.0028) so the conclusion is robust.

**Scope rider (binding — attach to every claim):** ≤2M params (+1 confirmation 10–50M); char-LM + diagnostics; NOT a frontier/MMLU/long-context claim; per-FLOP "dramatic" stays conditional unless all axes hold + powered.

---

## 2. The only remaining task: the 4-arm SOTA landscape

**Goal:** run `seq/landscape.py` to get the powered 4-arm head-to-head (TF / Prizma-v2 / GLA / Mamba-2), which ADDS GLA+Mamba-2 to what S2/S1 already did for TF-vs-Prizma. Two legs, two output files (resumable, cache-by-key, respect `$PRIZMA_RESULTS`):
- `python3 -u -m seq.landscape --full`   → RECALL leg  → `results/gpu_landscape.json`   (~16h on A100)
- `python3 -u -m seq.landscape --charlm` → CHAR-LM leg → `results/gpu_landscape_charlm.json` (~24h on A100; Prizma ~58min/seed is the bottleneck)
- CLI also has `--smoke` (CPU/MPS plumbing-only, do NOT cite).

**Per-cell printed results are the rescue net** (each `[...|arm|role|seed] best_bpc=...` / `best=... solve=...` line). Capture them every tick.

### THE BLOCKER — Drive-backing (needs the USER, one-time)
Each leg is **longer than Colab's ~10h session limit**, so it MUST resume across disconnects. Resume needs the crash-safe JSON to survive a VM teardown → it must be written to **Google Drive**, not local `/content`. The runner honors `PRIZMA_RESULTS`, so the fix is:
```
PRIZMA_RESULTS=/content/drive/MyDrive/PrizmaCampaign/landscape_results python3 -u -m seq.landscape --charlm ...
```
BUT `drive.mount()` opens an OAuth **account-chooser in a separate browser window that is OUTSIDE Chrome-MCP's tab group** — automation physically cannot click it (verified: `ValueError: mount failed`). **The user must mount Drive manually** (📁 Files pane → "Mount Drive" → pick the Colab account `nazmiefearmutcu@gmail.com` → Allow). THEN relaunch Drive-backed:
```
!mkdir -p /content/drive/MyDrive/PrizmaCampaign/landscape_results && cd /content/prizma \
 && cp -n results/gpu_landscape*.json /content/drive/MyDrive/PrizmaCampaign/landscape_results/ 2>/dev/null \
 && PRIZMA_RESULTS=/content/drive/MyDrive/PrizmaCampaign/landscape_results \
    python3 -u -m seq.landscape --charlm 2>&1 | tee /content/drive/MyDrive/PrizmaCampaign/logs_landscape.txt
```
The `cp -n` seeds the Drive dir with any local progress so the resume keeps it. **If the user won't/can't mount Drive: the landscape cannot complete the full 4-arm run** (each session restarts from scratch and never reaches GLA/Mamba-2) — fall back to rescuing the printed BPCs per tick (partial), and say so honestly.

### When a leg COMPLETES (Pareto / verdict printed, or its JSON done)
1. Exfiltrate the JSON: in a cell `!echo ===B64=== ; base64 -w0 results/gpu_landscape_charlm.json ; echo ; echo ===B64END===`, `get_page_text`, decode, save to `results/campaign_2026-06-08/`. (If the kernel is busy, the Colab **terminal** (right pane) is independent of the kernel and can `base64` the file.)
2. Render: `python3.13 -m seq.landscape_report` (torch-free; renders the persisted verdict JSONs into the Council-3 Pareto markdown).
3. Fold the 4-arm Pareto into `docs/PRIZMA_SEQ_REPORT.md` (v2 §"Results vs the bar") — honest BEATS/PARITY/WORSE/INCONCLUSIVE per Holm-corrected pairwise, GLA/Mamba-2 vs Prizma vs TF, with the scope rider. NO per-FLOP spin.
4. Referee-verify every number vs the JSON (a quick subagent or the Bash recompute), then **Opsera-gated commit**.
5. When BOTH legs land → campaign COMPLETE → delete the gists (below).

---

## 3. Colab operational lessons (learned the hard way — DON'T repeat)

1. **Idle-disconnect:** a connected-but-IDLE runtime is torn down in minutes ("inactivity" toast). NEVER leave a fresh runtime idle. **Run/queue the launch cell immediately** — running a cell while the runtime is "Allocating/Connecting" QUEUES it and it auto-runs on connect (no idle gap). A *running foreground cell* = activity = idle-safe (Pro+ background exec survives tab-close).
2. **~10h teardown:** long runs die around 10h. A **queued completion-checkpoint cell does NOT fire on a teardown** (only on clean completion) — so it does not protect a mid-run disconnect.
3. **Busy kernel can't run another cell:** an exfil cell QUEUES behind the running cell → you CANNOT exfil the JSON mid-run via a cell. The **Colab terminal (right pane) is a separate shell, independent of the kernel** — use it to `base64`/`cp` the live JSON while the kernel runs. (Reading large xterm output back is finicky; small is fine.)
4. **Drive popup is unreachable** (see §2) — user-only.
5. **GPU availability:** "A100 not available → connected to L4" happens; BPC/accuracy are GPU-independent (identical numbers) so an L4 is scientifically fine, just ~2.3× slower (char-LM ~46h on L4 vs ~24h on A100). S4 latency is the only GPU-specific leg.
6. **Chrome-MCP tab group + tab IDs reset frequently** (extension reconnects). Re-establish via `tabs_context_mcp` (createIfEmpty:true). If the user's Colab tabs aren't in the group, `navigate` a group tab to the notebook URL — BUT a fresh tab loads the SAVED notebook DISCONNECTED (it does not show the live runtime's cells). **Respect the user's browser** — they recently denied a tab action; ASK before re-taking browser control if there's any doubt.
7. **JS `get_page_text` on a landscape cell trips the cookie/query-string filter** (the HF/gist URLs). Extract ONLY result tokens via JS regex (`arm/role/sSEED:val` + boolean markers), never raw lines/URLs.
8. **tar warnings** `Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.provenance'` are HARMLESS (macOS-built tarball; GNU tar still exits 0 and extracts fine).

### Notebook URLs & bootstrap
- nb1 = `https://colab.research.google.com/drive/1M1rllmgqND6jtD4fab--gxJSOFepBR5G` (Untitled2)
- nb2 = `https://colab.research.google.com/drive/17af-btTgFFgOHQMZDicIOGjhBxkJ2QA0` (Untitled3)
- nb3 = `https://colab.research.google.com/drive/1VLhgp9vFiqhOcOdUXuFJWm5U5Bfg-K7V` (Untitled4) — last known to hold the A100 running `--charlm`.
- Bootstrap a fresh VM: `!mkdir -p /content/prizma && curl -sL <RAW_GIST> | base64 -d | tar xz -C /content/prizma` then `cd /content/prizma && python3 -u -m seq.landscape <flag>`.
- **Gists (transient — DELETE at campaign end):**
  - `4490202ec32af7db33d3f626c69295bc` — landscape bootstrap, code @`16cdfdb` (file `prizma_landscape_b64.txt`). RAW: `https://gist.githubusercontent.com/nazmiefearmutcu/4490202ec32af7db33d3f626c69295bc/raw/prizma_landscape_b64.txt`
  - `fe9c44feff67ff0a807f888c72e976f8` — original campaign bootstrap @`4c44918`.
  - Delete: `gh gist delete 4490202ec32af7db33d3f626c69295bc` ; `gh gist delete fe9c44feff67ff0a807f888c72e976f8`.

---

## 4. Rescue data already captured (so it is NOT lost)

char-LM (text8, d256L4H4, val-selected best_bpc), captured from the cell output before the A100 teardown — already saved in `charlm_partial_n7.json` and committed:
- **TF-v2 (n=10):** `[1.7061, 1.7224, 1.7214, 1.7132, 1.7287, 1.7283, 1.7355, 1.7352, 1.7327, 1.7198]` → mean **1.7243**, sd 0.0091
- **Prizma-v2 (n=7, s0–s6):** `[1.7436, 1.7464, 1.7417, 1.7426, 1.7507, 1.7444, 1.7447]` → mean **1.7449**, sd 0.0028
- Δ = +0.0205 BPC, Welch t ≈ 6.7, within ±0.05 → "competitive".

Recall (`recall_gate.json`, n=10, A100) per-leg Δ(Prizma−TF): MQAR-hard +0.040 ci90(−0.195,+0.275); induction +0.285 ci90(−0.056,+0.625) (TF bimodal — half its seeds collapse to ~0.06); selcopy +0.022 ci90(−0.009,+0.052) not_worse=True. All NOT MET → "competitive".

---

## 5. Environment & guardrails

- `python3.13` (bare `python` absent); `PYTORCH_ENABLE_MPS_FALLBACK=1` locally; **no `timeout` binary** on the mac.
- **Opsera pre-commit gate:** `touch /tmp/.opsera-pre-commit-scan-passed` as its OWN Bash call, THEN `git add <specific files> && git commit` as a separate call. The flag clears between commits. **NEVER `git add -A`** (the tree has unrelated dirty/untracked files: `results/figure.png`, the stale local `results/gpu_charlm2.json`, `hf_publish/`, `hf_space/`, `paper/` — leave them).
- Branch `v2-pareto-dominance`: **other sessions also commit arXiv `paper/` work here** — pull/rebase-aware, **never force-push**, never `-A`.
- **Off-path byte-identity is sacred:** any new lever must keep `eta=None`/default paths in `seq/delta.py` + the default `PrizmaSeqConfig` forward byte-identical (verified by the value-pinned tests). Full suite currently **203 passed, 9 skipped** (`PYTORCH_ENABLE_MPS_FALLBACK=1 python3.13 -m pytest -q`).
- Drive permission / computer-use is pre-authorized by the user ([[feedback_computer_use_allowed]]) EXCEPT the Drive-mount popup which is technically unreachable (user-only).

## 6. Deferred / future (not blocking)
- **Triton A100 verification** — run the 13-item checklist in `docs/superpowers/specs/triton_kernel_a100_checklist.md` on an A100 (first fix the non-pow2 `d_phi` tiling so it compiles; then fwd/grad parity gates incl. gated/repeated-key/ragged-tail/S0-carry/bf16). Only then drop the DRAFT status + trust the kernel.
- **Fair symmetric-dropout char-LM rerun** — the new `dropout` lever (`16cdfdb`) closes Prizma's regularizer gap vs TFConfig; a both-arms-dropout rerun could test whether it narrows the ~0.02 BPC gap (GPU question, not claimed).
- Council-1/Council-3 referee panels; arXiv paper integration of the landed verdicts.

---

**Status of THIS session at handoff:** monitoring loop stopped at the user's request; two core verdicts + two levers committed; landscape was running on A100(s) but Drive-blocked; the user was actively reconnecting/managing runtimes and chose to pause + hand off. Pick up by re-establishing browser access (gently — ask first), checking which notebook holds the live A100 + the `seq.landscape` cell, capturing printed results, and — once the user has mounted Drive — relaunching Drive-backed so the marathon can finish. The science is already safe; the landscape is the bonus 4-arm extension.
