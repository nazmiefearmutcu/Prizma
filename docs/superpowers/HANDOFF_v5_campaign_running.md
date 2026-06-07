# Prizma-Seq v2 — Handoff v5 (the campaign is RUNNING — monitor → exfiltrate → conclude)

> Written 2026-06-08. Self-contained. Supersedes HANDOFF_v4 for "what to do next": the 4 scientific
> sessions are LAUNCHED and HEALTHY on 3 parallel A100s on the CORRECT Colab account. Your job now is to
> MONITOR them, EXFILTRATE each powered verdict JSON as it completes (results are ephemeral — no Drive),
> report honestly with the scope rider, then run the councils + update the report + delete the gist.

---

## 0. EXACTLY WHERE I STOPPED (your green light)
All 4 campaign sessions are running across **3 A100 notebooks** on Colab account **`nazmiefearmutcu@gmail.com`**
(Pro+, 596 compute units at launch, burn ≈ 6.8 units/hr each ⇒ ~20/hr total ⇒ >24h headroom). The
**claude-in-chrome** browser is connected (the owner named it **"here!"**). Last health poll (VM clock
~22:36, 2026-06-07 — the VM clock runs a few hours behind real time) showed all three GPU-busy and alive,
NO crashes. Nothing has produced a FINAL verdict yet — the runs are genuinely multi-hour (each seed ≈ 10
min; 10 seeds × multiple arms × an LR sweep).

**Your very next action:** poll each of the 3 terminals (see §3) to check for completion/crash; when a
session finishes, `cat` its result JSON, save it locally, and report the powered verdict (§4).

---

## 1. The session map (3 notebooks, 4 sessions)
Browser tabs are in the claude-in-chrome MCP group. If tab IDs changed, re-`tabs_context_mcp`; the Drive
URLs below are stable.

| NB | Tab id | Drive URL | Session(s) | Launch PID | VM host | GPU | Log(s) | Result JSON |
|----|--------|-----------|-----------|-----------|---------|-----|--------|-------------|
| nb1 | 823974880 | drive/1M1rllmgqND6jtD4fab--gxJSOFepBR5G | **S2** recall-gate `--full` (10 seeds) | py 1345 | e05369e335b1 | A100-80GB High-RAM | `logs_s2.txt` | `results/recall_gate.json` (incremental) |
| nb2 | 823974881 | drive/17af-btTgFFgOHQMZDicIOGjhBxkJ2QA0 | **S1** `gpu_charlm2.py --v2 --seeds 0..9` (text8) | py 3092 | 8a5133bebca3 | A100-40GB | `logs_s1.txt` | `results/gpu_charlm2.json` (incremental) |
| nb3 | 823974882 | drive/1VLhgp9vFiqhOcOdUXuFJWm5U5Bfg-K7V | **S3** ablation → **S4** latency → **S4** lengen (chained) | bash 974 / py 975 | 6ac2796ce3d2 | A100-40GB | `logs_s3.txt`, `logs_s4lat.txt`, `logs_s4len.txt` | `results/gpu_ablation.json`, `gpu_latency.json`, `gpu_lengen.json` |

Each notebook bootstrapped the code into `/content/prizma` from the gist (see §5) and verified
`DEV cuda`. Each session was launched via `nohup ... &` in the **Pro+ Terminal** (survives cell
completion; independent of the kernel). Background-tab `computer type` into a terminal WORKS — you do not
need to foreground a tab to poll it.

### ⚠️ STALE-JSON GOTCHA (integrity-critical)
The git archive bundles PRE-EXISTING committed artifacts `results/gpu_ablation.json` (~13 KB),
`gpu_latency.json` (~10 KB), `gpu_lengen.json` (~4.8 KB), all dated **Jun 7 19:44** — i.e. BEFORE the
22:20 launch. **File presence is NOT a completion signal.** Use these signals instead:
- **S3/S4 (nb3):** the chain writes `s34_done.txt` (`ALLDONE`) only when ALL of gpu_ablation→latency→
  lengen finish. Also confirm each JSON's **mtime is AFTER launch** and the python procs are gone.
- **S2 (nb1) / S1 (nb2):** the runs WRITE their JSON incrementally from the start, so check (a) the
  python proc is GONE (`pgrep -af recall_gate` / `charlm2` empty) AND (b) the log tail shows a FINAL
  powered verdict block (not a mid-sweep line).

---

## 2. What each session must PRODUCE (the powered verdicts — report HONESTLY + scope rider)
Scope rider on EVERY claim: "≤2M params (+1 confirmation 10–50M); char-LM + diagnostics; NOT a
frontier/MMLU/long-context claim; per-FLOP 'dramatic' stays conditional unless all axes hold + powered."
- **S2 recall-gate** → per-leg (MQAR-hard / induction / selcopy) **TOST-parity** verdict + the
  **optimization-vs-capacity flip-test** → "dominant" vs "competitive". (Seen live: TF-big flip seeds
  hitting plateau=1.000 @26–32k steps.)
- **S1 char-LM v2** → BPC verdict BEATS/PARITY/WORSE via `margin_superiority(0.03)` + TOST, param-matched
  (live: TF-v2 3,490,816 vs Prizma-v2 3,496,256 = **−0.16%, OK ≤1%**; TF-v2 s0 best_bpc 1.7362).
- **S3 ablation** → which novel core wins: `surprise_norm` must beat BOTH controls (random + constant);
  `inctx_lr` vs baseline; **identical-model negative-control must PASS**.
- **S4** → all-n latency crossover + length-extrapolation (10×) + constant-memory (O(1)) confirmation.

---

## 3. Monitor protocol (poll ~every 20–30 min; the runs are multi-hour)
For each tab, click the terminal (~`1200,450`) then `type` ONE line + Return + screenshot:
- **S2 (nb1 823974880):** `echo ===S2 $(date +%H:%M)===; tail -6 logs_s2.txt; ls -la results/recall_gate.json; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; pgrep -af recall_gate|head -1||echo PROC_GONE`
- **S1 (nb2 823974881):** same with `logs_s1.txt`, `results/gpu_charlm2.json`, `pgrep -af charlm2`.
- **S34 (nb3 823974882):** `echo ===S34 $(date +%H:%M)===; tail -4 logs_s3.txt; tail -3 logs_s4lat.txt 2>/dev/null; tail -3 logs_s4len.txt 2>/dev/null; cat s34_done.txt 2>/dev/null||echo NOT_DONE; nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader; pgrep -af 'gpu_ablation|gpu_latency|gpu_lengen'|head -2||echo PROC_GONE`

If a VM resets (files gone / "PROC_GONE" with no JSON + empty logs): re-bootstrap from the gist (§5) and
relaunch that session (commands in §6).

---

## 4. EXFILTRATE on completion (results are ephemeral — no Drive)
When a session is DONE, capture its JSON to disk (don't trust the canvas to read a big JSON):
`cat results/<name>.json` in the terminal then read the screenshot, OR better, base64 it for a clean
copy: `base64 -w0 results/<name>.json` and reconstruct locally. Save each verdict JSON under
`Prizma/results/campaign_2026-06-08/<name>.json` on disk 2. Then report the powered verdict + scope rider.

---

## 5. The bootstrap gist (delete at the very end)
- id `fe9c44feff67ff0a807f888c72e976f8` @ commit `4c44918`; raw:
  `https://gist.githubusercontent.com/nazmiefearmutcu/fe9c44feff67ff0a807f888c72e976f8/raw/264408a24c469d06557991a7e47c2beb49ef3784/prizma_v2_b64.txt`
- Re-bootstrap one-liner: `mkdir -p /content/prizma && curl -sL <raw> | base64 -d | tar xz -C /content/prizma && cd /content/prizma && echo BOOTSTRAPPED && python3 -c "from seq.common import get_device; print('DEV', get_device())"`
- **Delete when the campaign ends:** `gh gist delete fe9c44feff67ff0a807f888c72e976f8` (gh authed as `nazmiefearmutcu`).
- NOTE: the gist is @`4c44918` and does NOT include Lever F (`seq/delta_fused.py`) — that's fine, the
  campaign sessions don't use it. To verify Lever F's CUDA path on a freed A100 later, re-`git archive`
  the current HEAD and make a fresh gist.

## 6. Relaunch commands (if a VM resets)
- S2: `cd /content/prizma && nohup python3 -u seq/recall_gate.py --full > logs_s2.txt 2>&1 & echo S2_PID $!`
- S1: `cd /content/prizma && nohup python3 -u gpu_charlm2.py --v2 --seeds 0 1 2 3 4 5 6 7 8 9 > logs_s1.txt 2>&1 & echo S1_PID $!`
- S34: `cd /content/prizma && nohup bash -c 'python3 -u gpu_ablation.py > logs_s3.txt 2>&1; python3 -u gpu_latency.py > logs_s4lat.txt 2>&1; python3 -u gpu_lengen.py > logs_s4len.txt 2>&1; echo ALLDONE > s34_done.txt' >/dev/null 2>&1 & echo S34_PID $!`

---

## 7. LOCAL state (repo) — what changed this session
- Branch `v2-pareto-dominance`. **HEAD = `a6f7984`** (was `ae1f63c` at handoff-v4).
- `e1e8a08` **Lever F** `seq/delta_fused.py` (Task 1.F): fused chunked-delta — CUDA path via
  `torch.compile(chunked_delta)` (TorchInductor→Triton; provably equivalent by construction), EXACT
  eager fallback on CPU/MPS + surprise/eta/n_delta≥2/backend="eager". `seq/delta.py` UNTOUCHED
  (byte-identity preserved). Built via subagent-driven-development (implement→spec-review→quality-review,
  both passed). A hand-written `@triton.jit` kernel is DEFERRED to be developed+verified ON the A100.
- `a6f7984` **Lever F polish** (code-review nits): CUDA fwd+grad parity now also grad-gates the
  `additive` and decoupled `beta_e` branches; `ValueError` on unknown `backend`; pytest `monkeypatch`
  fixture. Full suite **125 passed, 5 skipped** (5 = CUDA-gated parity, run on the A100). `tests/test_fused.py`.
- Env reminders: `python3.13` only; prefix torch cmds `PYTORCH_ENABLE_MPS_FALLBACK=1`; no `timeout`
  binary; commit gate = `touch /tmp/.opsera-pre-commit-scan-passed` as its OWN Bash call THEN
  `git add <files> && git commit` separately; NEVER `git add -A` (untracked hf_publish/, hf_space/,
  paper/, docs/HF_*, modified results/figure.png must stay out).

---

## 8. After the campaign
Report S1–S4 powered verdicts with the scope rider; keep per-FLOP conditional. Then: Council-1 reviews
each causal claim; Council-3 judges the combined picture; re-run writing-plans for Phase 2 (consolidate)
→ 3 (scale-up 10–50M) → 4 (report + adversarial referee); update `docs/PRIZMA_SEQ_REPORT.md` + the
`prizma_v2_pareto_sprint` memory; **delete the transient gist** (§5).
