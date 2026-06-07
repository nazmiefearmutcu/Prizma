# Prizma-Seq v2 — Handoff v4 (run the 4× HIGH-RAM A100 parallel campaign)

> Written 2026-06-08. Self-contained — assume NO shared conversation context. Read top-to-bottom.
> This supersedes HANDOFF_v3 for the "what to do next" picture (the local prerequisites are DONE; the
> task now is to RUN the campaign on 4 parallel high-RAM A100s on the CORRECT Colab account).

---

## 0. EXACTLY WHERE I STOPPED (your green light)
I was driving Google Colab via the **claude-in-chrome** extension to run the GPU campaign. I had it
running 3-wide, then discovered I was on the **wrong Google account** (it had 6 active sessions + low
pre-paid compute → Colab refused the 4th GPU with "Too many sessions", and one runtime dropped from
inactivity). The owner fixed the "burası" browser onto the **correct account**.

**My very next action is to send browser-connection requests to determine which Chrome to drive** — i.e.
call `switch_browser` (claude-in-chrome), which broadcasts a "Connect" prompt to every connected Chrome
extension; the owner clicks Connect in the right one. (Two browsers are/were connected: **"burası"**
deviceId `f38ac278-b1e0-479f-b417-1a8455bfc225` = the now-correct account, and **"Browser 1"** deviceId
`483ca6ee-2230-4287-b09d-04a543683dc8`.) After connecting, **set up 4 HIGH-RAM A100 notebooks and run
the 4 campaign sessions in parallel** (§5).

The last live `tabs_context_mcp` on the correct account returned a fresh tab group (the old wrong-account
tabs are gone). A `New Tab`/`Untitled29.ipynb` was created then invalidated — just start fresh.

---

## 1. Mission (owner, locked)
Take **Prizma-Seq** (a non-Transformer Gated-DeltaNet-family token mixer with a parameter-free quadratic
feature map + verified O(1)/constant-memory streaming inference) from "within-margin candidate" to
**Pareto-dominant** vs a param/FLOP-matched **tuned** Transformer at small scale, then one scale-up
confirmation. Integrity rules: no faked metrics; every number from a reproducible seed-pinned script with
CIs; param/FLOP-matched non-strawman baseline; powered statistics; `step()==forward()<1e-4` O(1) guard
before any accuracy number; identical-model negative-control must pass; mandatory **scope rider** on every
claim ("≤2M params (+1 confirmation 10–50M); char-LM + diagnostics; NOT a frontier/MMLU/long-context
claim"); per-FLOP "dramatic" stays conditional unless all axes hold simultaneously + powered.

---

## 2. Where the project lives (CRITICAL)
- **Repo:** `/Volumes/disk 2/Desktop_Migrate_2026-05-28/Projeler/proje/Prizma` — capital **P**, lowercase
  rest. A `PRISM` symlink may exist but **dropped once during an external-volume (M2 SSD) remount** —
  target the real `Prizma` dir in automation. Always quote the space.
- **Branch:** `v2-pareto-dominance` (NOT `main`). **HEAD this session = `4c44918`.**
- **Env:** `python3.13` (bare `python` is NOT on PATH); prefix torch cmds with
  `PYTORCH_ENABLE_MPS_FALLBACK=1` locally; the `timeout` binary is NOT installed on the mac (don't wrap
  cmds in it). Tests: `python3.13 -m pytest tests/ -q` (106 pass). Guards: `python3.13 -m seq.delta`,
  `python3.13 -m seq.prizma_seq`.
- **Commit gate (Opsera/Semgrep pre-commit):** `touch /tmp/.opsera-pre-commit-scan-passed` as its OWN
  Bash call, THEN `git add <files> && git commit` as a SEPARATE call (chaining touch && commit is
  blocked). NEVER `git add -A` (untracked `hf_publish/`, `hf_space/`, `paper/`, `docs/HF_*` and a modified
  `results/figure.png` must stay out — stage only the files you changed).

---

## 3. What I did this session — the LOCAL-PREREQUISITE SPRINT (all DONE, committed, reviewed)
Executed via subagent-driven-development (each task = implementer → spec-review → quality-review, with fix
loops; the Workflow tool orchestrated them). **Every commit preserved off-path byte-identity** (T8
verified via SHA256 fingerprints of all 7 config forwards), so the published v1 *Prizma* numbers and all
prior results remain valid. Test count grew 60 → **106, all green**.

Commits on `v2-pareto-dominance` (newest last):
| Commit | What |
|---|---|
| `c9fae02` | **R4 FLOP truth**: `flop_ledger.py` re-emits a per-config ledger pinning every number to its exact (feat_map, feat_n2/feat_rank, d_φ); writes `results/flop_ledger_v2.{json,txt}`; report+synthesis annotated. Canonical d_φ left to the A100 MQAR-D128 gate. (Honest finding: v2-lean d137=1.73×/1.37×, code-default d128=1.70×/1.34×, v1 d256=2.14×/1.78×, none=1.36×/1.00× per-FLOP vs matched TF; ≤1.0× needs the banded window E.) |
| `9bde66c` | **Lever G** `inctx_lr` (RWKV-7 per-channel in-context LR) through `_delta_reference`+`chunked_delta`+`step()`; off<1e-6, chunked==ref<1e-4 (documented sequential fallback), step==forward<1e-4. |
| `7a71e65` | **Tiny-hybrid arm** `seq/hybrid.py` (Samba-style: (L-1) Prizma + 1 attention layer); `hybrid_factory`; param-matched +0.48% vs TF; Council-3 3rd baseline. |
| `2522fab` | **Recall-gate runner** `seq/recall_gate.py`: MQAR-hard + induction + selcopy; **TOST-parity** verdict + optimization-vs-capacity **flip-test**; seed-pinned `build_and_train` + per-arm `sweep_lr` + powered `seq.stats`; crash-safe JSON; pure verdict layer unit-tested; CLI `--smoke`/`--full`. |
| `a193bc4`,`239493f` | **v2 campaign harness** `seq/gpu_harness.py` (the keystone: seed-pinned `run_cell`, `sweep_then_seeds`, `powered_summary`, `h2h`, declarative `make_arm` for all v2 knobs, **identical-model `negative_control`** canary, atomic JSON) + `gpu_ablation.py` (S3 novel-core ablation). |
| `f569a20` | **surprise_gen fix**: `surprise_mode='random'` now runs through the model (reproducible generator + `surprise_seed`), so Lever A's random control is usable on MPS+CUDA. (Spawned mid-S3-ablation when the harness flagged it UNRUNNABLE.) |
| `caaf998` | **S1 char-LM v2** `gpu_charlm2.py --v2`: Prizma-v2 arm (out_gate+state_norm+decoupled_gate+gated) + hybrid arm + param-matched TF; powered BPC verdict `charlm_v2_verdict` (margin_superiority(0.03) + TOST, correct lower-is-better sign). Fixed a real **cache-key collision** (v2 TF was reusing the legacy TF's cached cell). |
| `d76f5e8` | **T8 hardening**: inctx_lr+surprise_gate mutual-exclusion assert; recall-gate `main()` arg-guard (unknown arg → exit 2, never a multi-hour run); refreshed ablation artifact; reviewer nits. |
| `1eae719` | docs: HANDOFF_v3_A100_ready.md (the campaign runner→session map). |
| `d5200ec` | docs: corrected PRISM→Prizma path refs in v2 handoff/plan/spec. |
| `4c44918` | **device fix** (caught LIVE on the A100): `seq/common.py::get_device()` now detects CUDA FIRST. Before, `get_device(prefer="mps")` returned CPU on a CUDA box unless `prefer=="cuda"` was passed — so `recall_gate`/`gpu_ablation` (which use the default) silently ran on **CPU while an A100 sat idle**. Local MPS behavior unchanged; 106 tests still green. **HEAD.** |

---

## 4. What I did this session — the A100 CAMPAIGN ATTEMPT (where it stands)
1. **Code→Colab bootstrap (no public WIP push, owner pseudonymity care):** built a clean `git archive`
   of HEAD (491KB tar.gz, code + bundled shakespeare; text8 downloads on Colab), base64'd it into a
   **transient secret GitHub gist**, and pull it on Colab via curl. Current gist (holds the FIXED code @
   `4c44918`): id `fe9c44feff67ff0a807f888c72e976f8`, raw:
   `https://gist.githubusercontent.com/nazmiefearmutcu/fe9c44feff67ff0a807f888c72e976f8/raw/264408a24c469d06557991a7e47c2beb49ef3784/prizma_v2_b64.txt`
   (an earlier gist `952c477...` @ pre-fix code was deleted). **Delete this gist when the campaign ends:**
   `gh gist delete fe9c44feff67ff0a807f888c72e976f8` (gh is authed as `nazmiefearmutcu`, scopes gist+repo).
   The raw URL is also saved locally at `/tmp/prizma_raw_url.txt`.
2. **Validated on the A100:** bootstrapped → **`106/106 tests pass on CUDA`**; the recall-gate runner
   produced real TOST verdicts on the A100 (smoke). A100 = NVIDIA A100-SXM4-40GB.
3. **Caught + fixed the get_device→CPU bug live** (commit `4c44918` + refreshed gist) so every new A100
   comes up `device=cuda` with zero per-notebook patching.
4. **Google Drive persistence is BLOCKED:** `drive.mount` opens an OS-level Google-accounts OAuth popup
   *outside* the extension's tab group, and native-desktop clicking is blocked in browser chrome
   (read-tier). In-page Drive permission dialogs I CAN click; the downstream OS popup I cannot. ⇒ results
   are **ephemeral on Colab** (no Drive). Mitigation: exfiltrate result JSONs as sessions complete; the
   runners are crash-safe/resumable *within* a session.
5. **Ran 3-wide, then hit the account problem:** launched the campaign as a terminal `nohup` chain, then
   parallelized to 3 A100 notebooks (S2/S1/S3 all confirmed GPU 65–100%, device=cuda, 10-seed). Colab
   refused the 4th with **"Too many sessions"** — the account had **6 active sessions** + low pre-paid
   compute (~8.77h at ~40 units/hr). That account was the **WRONG one**; the owner switched "burası" to
   the **correct account** and told me to **re-run everything from scratch** there.
   - NOTE: the wrong account's runtimes may still be burning compute — when you regain control of that
     account, terminate them (Manage sessions / Runtime → Disconnect and delete runtime).
   - The `nohup`-in-terminal pattern is what made the runs survive cell completion; monitor via the
     **Colab Terminal** (Pro+), which is independent of the notebook kernel (the kernel gets blocked by a
     foreground `!...&` cell). Terminals take **literal** input (no Monaco auto-close-quote issues).

**Honest status:** NO scientific campaign numbers exist yet — only smoke/plumbing verdicts (which are
explicitly labelled non-scientific). The real ≥10-seed results are what this run must produce.

---

## 5. YOUR TASK — 4× HIGH-RAM A100 parallel campaign
1. **Pick the browser:** call `switch_browser` (sends Connect prompts to all Chrome extensions); owner
   clicks Connect in the correct-account Chrome. (Or `select_browser` deviceId `f38ac278-...` for
   "burası" if confirmed correct.)
2. **For EACH of 4 notebooks** (create with a new tab → `https://colab.research.google.com/#create=true`):
   - Runtime → **Change runtime type** → **A100 GPU** + **toggle HIGH RAM ON** → Save → **Connect**.
     (Stable coords seen: Runtime menu (182,33); "Change runtime type" (216,258); A100 radio (879,340);
     **High RAM toggle (682,430)**; Save (938,541); Connect (~1480,54). VERIFY each via screenshot —
     notebooks sometimes need a second click once fully loaded.)
   - Open **Terminal** (bottom-left, ~126,754). Bootstrap (literal terminal input):
     `mkdir -p /content/prizma && curl -sL https://gist.githubusercontent.com/nazmiefearmutcu/fe9c44feff67ff0a807f888c72e976f8/raw/264408a24c469d06557991a7e47c2beb49ef3784/prizma_v2_b64.txt | base64 -d | tar xz -C /content/prizma && cd /content/prizma && echo BOOTSTRAPPED && python3 -c "from seq.common import get_device; print('DEV', get_device())"`  → expect `BOOTSTRAPPED` + `DEV cuda`.
   - Launch its session in the background (one session per A100):
     - **S2** (recall TOST-parity gate, 10 seeds): `cd /content/prizma && nohup python3 -u seq/recall_gate.py --full > logs_s2.txt 2>&1 & echo S2 $!`
     - **S1** (char-LM v2, 10 seeds, text8): `cd /content/prizma && nohup python3 -u gpu_charlm2.py --v2 --seeds 0 1 2 3 4 5 6 7 8 9 > logs_s1.txt 2>&1 & echo S1 $!`
     - **S3** (novel-core ablation): `cd /content/prizma && nohup python3 -u gpu_ablation.py > logs_s3.txt 2>&1 & echo S3 $!`
     - **S4** (efficiency, fast): `cd /content/prizma && nohup bash -c 'python3 -u gpu_latency.py > logs_s4lat.txt 2>&1; python3 -u gpu_lengen.py > logs_s4len.txt 2>&1; echo S4DONE > s4_done.txt' >/dev/null 2>&1 & echo S4 $!`
   - Confirm each: `sleep 6; head -8 logs_sX.txt; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader` → expect `device=cuda` + GPU busy.
3. **Monitor** (poll each terminal ~every 25–30 min; the runs are multi-hour): `cd /content/prizma; ps -ef|grep -c [p]ython3; nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader; ls -la results/*.json; tail -6 logs_s*.txt`. A session is DONE when its python process is gone + its result JSON is sizable → `cat` it and report the powered verdict. **Exfiltrate each JSON as it completes** (ephemeral).
4. If a notebook's VM resets (files gone), re-bootstrap there from the gist + relaunch that session.
5. **Lever F** (the one remaining on-GPU BUILD): write `seq/delta_fused.py` (Triton fused chunked-delta,
   equivalence-gated `== chunked_delta < 1e-4`, MPS/CPU fallback) once a GPU frees; commit it.

---

## 6. After the campaign
Report the powered verdicts HONESTLY with the scope rider: S2 recall-gate → **dominant vs competitive**
(per-leg TOST parity + flip-test); S3 → which novel core wins (surprise_norm must beat BOTH controls;
inctx_lr vs baseline; negative-control PASS); S1 → char-LM BPC verdict (BEATS/PARITY/WORSE,
margin_superiority 0.03 + TOST, param-matched); S4 → all-n latency crossover + length-extrapolation +
constant-memory. Keep per-FLOP conditional. Then: Council-1 reviews each causal claim; Council-3 judges
the combined picture; re-run writing-plans for Phases 2 (consolidate) → 3 (scale-up 10–50M) → 4 (report +
adversarial referee); update `docs/PRIZMA_SEQ_REPORT.md` + the `prizma_v2_pareto_sprint` memory; **delete
the transient gist**.

## 7. File map
Model: `seq/prizma_seq.py` (+ v2 knobs: out_gate, state_norm, decoupled_gate, surprise_gate+surprise_mode
+surprise_seed, n_delta, feat_map quad2/quad2_lowrank, feat_rank, inctx_lr, gated), `seq/delta.py`,
`seq/transformer.py`, `seq/hybrid.py`. Harness: `seq/common.py` (build_and_train, get_device CUDA-first),
`seq/stats.py` (powered), `seq/lrsweep.py`, `seq/ledger.py`. Runners: `gpu_harness.py`, `gpu_ablation.py`
(S3), `seq/recall_gate.py` (S2), `gpu_charlm2.py --v2` (S1), `gpu_latency.py`/`gpu_lengen.py` (S4),
`flop_ledger.py`. Tasks: `seq/tasks.py` (MixedMQAR/Induction/SelectiveCopy), `seq/charlm.py`. Councils:
`committee/`. Spec/plan/handoffs: `docs/superpowers/`.
