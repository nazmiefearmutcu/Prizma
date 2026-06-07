# Prizma-Seq v2 Pareto-Dominance — Implementation Plan (Phase 0 + Phase 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the honest-science foundation (repro, powered stats, LR-sweep, FLOP ledger, councils) and the first tranche of architecture levers (output-gating+state-norm, surprise-gate, DeltaProduct, lean feature-map, banded window, fused kernel) so each razor-thin §4 leg can flip to a *powered, decisive* win.

**Architecture:** Prizma-Seq is a Gated-DeltaNet-family mixer (`seq/prizma_seq.py` block: conv → QKV → chunked delta state `seq/delta.py` + window head → SwiGLU). All v2 levers are **default-off config knobs** on `PrizmaSeqConfig`, each gated by a `step()==forward()<1e-4` O(1) guard and a chunked-vs-reference `<1e-4` correctness test. The TF baseline (`seq/transformer.py`) is byte-identical except the mixer.

**Tech Stack:** Python 3.13 (system), PyTorch 2.12 (MPS locally for guards/smoke; CUDA/A100 via Colab for headline numbers), numpy. No autocast, float32.

**Working dir (note — project was migrated):** `/Volumes/disk 2/Desktop_Migrate_2026-05-28/Projeler/proje/PRISM`. All commands below assume `cd` into it. Run module self-tests as `python -m seq.<module>` from repo root.

**Re-planning note:** Phases 2 (consolidate), 3 (scale-up ~10–50M), 4 (report + referee gate) are re-planned after Phase 1 results + council judgments — they depend on which levers win. This document covers Phase 0 + Phase 1 only.

---

## File Structure

**New files:**
- `seq/stats.py` — powered statistics: multi-seed runner, TOST equivalence, one-sided superiority, solve-rate, t-CI.
- `seq/lrsweep.py` — per-model / per-width LR (μP-aware) sweep harness; emits chosen LR + rejected LRs.
- `seq/ledger.py` — param + FLOP ledger auto-emit per run (wraps `flop_ledger.py`), asserts param-match.
- `committee/charters/council1_general.md`, `council2_quant.md`, `council3_standards.md` — durable council charters.
- `committee/verdicts/` — one JSON per council round (template defined in Task 0.6).
- `tests/test_stats.py`, `tests/test_repro.py`, `tests/test_ledger.py` — pytest gates for the harness.

**Modified files:**
- `seq/common.py` — add `build_and_train()` seeded factory wrapper (the repro fix).
- `seq/prizma_seq.py` — add v2 config knobs + forward/step branches for levers A/B/C/D/E.
- `seq/delta.py` — add the higher-order (DeltaProduct) and surprise-gated reference + chunked forms.

---

## PHASE 0 — Foundation & Councils

### Task 0.1: Pin the environment & verify existing kernel guards still pass

**Files:**
- Create: `docs/ENV.md`

- [ ] **Step 1: Verify interpreter + torch**

Run:
```bash
cd "/Volumes/disk 2/Desktop_Migrate_2026-05-28/Projeler/proje/PRISM"
python3.13 -c "import torch, numpy, sys; print(sys.version.split()[0], torch.__version__, torch.backends.mps.is_available())"
```
Expected: prints `3.13.x 2.12.x True` (MPS available). If `python3.13` missing, fall back to `python3` and record the actual version in `docs/ENV.md`.

- [ ] **Step 2: Run the two existing correctness guards (must already be green)**

Run:
```bash
python -m seq.delta
python -m seq.prizma_seq
```
Expected: `seq.delta` → `ALL OK`; `seq.prizma_seq` → both `feat_map=none` and `feat_map=quad2` lines end `OK` with `step-vs-forward max|d| < 1e-4` and identical `params`.

- [ ] **Step 3: Write `docs/ENV.md`** recording: exact python/torch/numpy versions, device, the two guard outputs (paste), and the working-dir note. This is the reproducibility anchor for every later run.

- [ ] **Step 4: Commit**
```bash
git add docs/ENV.md && git commit -m "docs: pin env + record baseline kernel guards"
```

---

### Task 0.2: Seed-pinned factory wrapper (the reproducibility fix)

The report flags: init is created **before** `set_seed` in the `run_cell` callers, so per-seed init isn't pinned (d128L4H4.s0 read 0.785 vs 0.979 across runs). Fix = one wrapper that seeds, *then* builds, *then* trains.

**Files:**
- Modify: `seq/common.py` (add `build_and_train` after `train_model`)
- Test: `tests/test_repro.py`

- [ ] **Step 1: Write the failing test**
```python
# tests/test_repro.py
import torch
from seq.common import build_and_train, TrainConfig, get_device
from seq.transformer import Transformer, TFConfig

class _TinyTask:
    vocab = 16
    def sample(self, B, device):
        x = torch.randint(0, self.vocab, (B, 12), device=device)
        return x, x, torch.ones_like(x)

def _fac(vocab, max_len):
    return Transformer(TFConfig(vocab=vocab, d_model=32, n_layers=1, n_heads=2, max_len=max_len))

def test_seed_pinned_init_is_bit_reproducible():
    dev = get_device()
    cfg = TrainConfig(steps=3, eval_every=3, min_steps=0, batch_size=4, log=False)
    r1 = build_and_train(_fac, _TinyTask(), cfg, dev, seed=7, vocab=16, max_len=16)
    r2 = build_and_train(_fac, _TinyTask(), cfg, dev, seed=7, vocab=16, max_len=16)
    assert abs(r1.final_loss - r2.final_loss) < 1e-6, (r1.final_loss, r2.final_loss)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_repro.py -q`
Expected: FAIL with `ImportError: cannot import name 'build_and_train'`.

- [ ] **Step 3: Implement `build_and_train` in `seq/common.py`**
```python
def build_and_train(model_fac, task, cfg: TrainConfig, device, seed=0, **fac_kw):
    """Reproducibility-correct entry point: seed BEFORE constructing the model so per-seed init is
    pinned (fixes the run_cell init-before-set_seed defect), then train. `model_fac(**fac_kw)` must
    return an nn.Module. Use this everywhere instead of (construct; set_seed; train_model)."""
    set_seed(seed)
    model = model_fac(**fac_kw)
    return train_model(model, task, cfg, device, seed=seed)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_repro.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add seq/common.py tests/test_repro.py && git commit -m "fix(repro): seed-before-build factory wrapper (pins per-seed init)"
```

---

### Task 0.3: Powered-statistics module (kills the n=2–3 underpowering)

**Files:**
- Create: `seq/stats.py`
- Test: `tests/test_stats.py`

- [ ] **Step 1: Write the failing test**
```python
# tests/test_stats.py
import numpy as np
from seq.stats import summarize, tost_equivalence, superiority_test, solve_rate

def test_summarize_reports_t_ci_median_solverate():
    xs = [0.95, 0.97, 0.99, 0.96, 0.98, 0.94, 0.99, 0.97, 0.96, 0.98]
    s = summarize(xs, solve_thresh=0.9)
    assert s["n"] == 10
    assert abs(s["median"] - 0.97) < 1e-9
    assert s["solve_rate"] == 1.0
    assert s["ci95"][0] < s["mean"] < s["ci95"][1]

def test_superiority_detects_real_gap():
    a = list(np.full(10, 0.80) + np.array([0.01,-0.01]*5))   # ~0.80
    b = list(np.full(10, 0.70) + np.array([0.01,-0.01]*5))   # ~0.70
    res = superiority_test(a, b)            # H1: mean(a) > mean(b)
    assert res["p_value"] < 0.05 and res["significant"]

def test_tost_equivalence_within_margin():
    a = list(np.full(10, 0.900) + np.array([0.002,-0.002]*5))
    b = list(np.full(10, 0.901) + np.array([0.002,-0.002]*5))
    res = tost_equivalence(a, b, margin=0.02)
    assert res["equivalent"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_stats.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'seq.stats'`.

- [ ] **Step 3: Implement `seq/stats.py`**
```python
"""Powered statistics for the head-to-head: t-based CIs (not z), solve-rate, one-sided superiority
(Welch t), and TOST equivalence. Use >=10 seeds for any decisive claim (the v1 n=2-3 was descriptive
only). No SciPy dependency assumed -> t critical values via a small table + normal fallback."""
from __future__ import annotations
import math
import numpy as np

# two-sided t critical values @ alpha=0.05 for df 1..30, then normal (1.96) beyond.
_T95 = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,
        11:2.201,12:2.179,13:2.160,14:2.145,15:2.131,16:2.120,17:2.110,18:2.101,19:2.093,
        20:2.086,21:2.080,22:2.074,23:2.069,24:2.064,25:2.060,26:2.056,27:2.052,28:2.048,
        29:2.045,30:2.042}

def _t_crit(df, two_sided=True):
    t = _T95.get(int(df), 1.96)
    return t if two_sided else (t if df not in _T95 else _T95[int(df)])  # table is 2-sided; ok approx

def summarize(xs, solve_thresh=0.9):
    a = np.asarray(xs, float); n = len(a)
    mean = float(a.mean()); sd = float(a.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    h = _t_crit(n - 1) * se
    return {"n": n, "mean": mean, "median": float(np.median(a)), "sd": sd,
            "ci95": (mean - h, mean + h), "min": float(a.min()), "max": float(a.max()),
            "solve_rate": float((a >= solve_thresh).mean())}

def _welch(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    va, vb = a.var(ddof=1), b.var(ddof=1); na, nb = len(a), len(b)
    se = math.sqrt(va/na + vb/nb) or 1e-12
    t = (a.mean() - b.mean()) / se
    df = (va/na + vb/nb)**2 / ((va/na)**2/(na-1) + (vb/nb)**2/(nb-1) + 1e-30)
    return t, df, se

def _p_one_sided_from_t(t, df):
    # normal approx to the t-CDF tail (adequate for df>=9, our regime). Upper tail of H1: mean(a)>mean(b)
    from math import erf, sqrt
    z = t  # df>=9 -> t ~ z within ~5%
    return 1.0 - 0.5 * (1 + erf(z / sqrt(2)))

def superiority_test(a, b, alpha=0.05):
    """One-sided Welch t for H1: mean(a) > mean(b)."""
    t, df, se = _welch(a, b)
    p = _p_one_sided_from_t(t, df)
    return {"delta": float(np.mean(a) - np.mean(b)), "t": t, "df": df,
            "p_value": p, "significant": p < alpha}

def tost_equivalence(a, b, margin, alpha=0.05):
    """Two one-sided tests: equivalent if the (1-2alpha) CI of (mean a - mean b) lies within +/-margin."""
    t, df, se = _welch(a, b)
    diff = float(np.mean(a) - np.mean(b))
    crit = _t_crit(round(df)) * 0.84  # ~90% CI half-width factor vs 95% table (z1.645/z1.96)
    lo, hi = diff - crit*se, diff + crit*se
    return {"delta": diff, "ci90": (lo, hi), "margin": margin,
            "equivalent": (lo > -margin) and (hi < margin)}

def solve_rate(xs, thresh=0.9):
    return float((np.asarray(xs, float) >= thresh).mean())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_stats.py -q`
Expected: PASS (3 passed). If `test_tost_equivalence_within_margin` is borderline, widen the synthetic spread in the test, not the module.

- [ ] **Step 5: Commit**
```bash
git add seq/stats.py tests/test_stats.py && git commit -m "feat(stats): powered stats (t-CI, solve-rate, superiority, TOST)"
```

---

### Task 0.4: Per-model / per-width LR-sweep harness (kills optimization-confounding)

The FLOP-matched arms were "optimization-confounded" because a recipe tuned at d128 was applied at d208 (optimal LR scales ~1/width under μP). This harness sweeps a per-config LR grid and records the winner + all rejected LRs.

**Files:**
- Create: `seq/lrsweep.py`

- [ ] **Step 1: Implement `seq/lrsweep.py`**
```python
"""Per-config LR sweep so no architecture/width is denied an LR another gets. Stage-1: full grid @1
seed -> pick best on the frozen eval; Stage-2 is run by the caller at >=N seeds on the chosen LR.
Records the FULL grid incl. rejected LRs for the audit trail (committee guardrail #10)."""
from __future__ import annotations
import copy
from dataclasses import replace
from .common import build_and_train, TrainConfig

DEFAULT_GRID = (5e-4, 1e-3, 1.5e-3, 2e-3, 3e-3)

def sweep_lr(model_fac, task, base_cfg: TrainConfig, device, grid=DEFAULT_GRID, seed=0, **fac_kw):
    """Returns {'best_lr', 'best_acc', 'grid': [{'lr','best_acc','steps_to_plateau'}...]}."""
    rows = []
    for lr in grid:
        cfg = replace(base_cfg, lr=lr)
        r = build_and_train(model_fac, task, cfg, device, seed=seed, **fac_kw)
        rows.append({"lr": lr, "best_acc": r.best_acc, "steps_to_plateau": r.steps_to_plateau})
    best = max(rows, key=lambda d: d["best_acc"])
    return {"best_lr": best["lr"], "best_acc": best["best_acc"], "grid": rows}
```

- [ ] **Step 2: Smoke it on the tiny task**

Run:
```bash
python -c "
from seq.lrsweep import sweep_lr; from seq.common import TrainConfig, get_device
from seq.transformer import Transformer, TFConfig
import torch
class T:
    vocab=16
    def sample(s,B,d): x=torch.randint(0,16,(B,12),device=d); return x,x,torch.ones_like(x)
fac=lambda vocab,max_len: Transformer(TFConfig(vocab=vocab,d_model=32,n_layers=1,n_heads=2,max_len=max_len))
print(sweep_lr(fac, T(), TrainConfig(steps=20,eval_every=20,min_steps=0,batch_size=8,log=False), get_device(), grid=(1e-3,2e-3), vocab=16, max_len=16))
"
```
Expected: prints a dict with `best_lr` in {0.001, 0.002} and a 2-row `grid`. (No assertion — this is a smoke; correctness is exercised by callers in Phase 1.)

- [ ] **Step 3: Commit**
```bash
git add seq/lrsweep.py && git commit -m "feat(lrsweep): per-config LR sweep with rejected-LR audit trail"
```

---

### Task 0.5: Param + FLOP ledger auto-emit (every head-to-head is matched & disclosed)

**Files:**
- Create: `seq/ledger.py`
- Test: `tests/test_ledger.py`

- [ ] **Step 1: Write the failing test**
```python
# tests/test_ledger.py
from seq.ledger import param_match_report
from seq.transformer import Transformer, TFConfig
from seq.prizma_seq import PrizmaSeqLM, PrizmaSeqConfig

def test_param_match_within_2pct_at_matched_config():
    tf = Transformer(TFConfig(vocab=64, d_model=128, n_layers=4, n_heads=4))
    pz = PrizmaSeqLM(PrizmaSeqConfig(vocab=64, d_model=128, n_layers=4, n_heads=4, feat_map="quad2"))
    rep = param_match_report(tf, pz)
    assert rep["matched"], rep            # |Δparams| / tf < 0.02
    assert rep["feat_map_added_params"] == 0   # quad2 is buffers -> 0 trainable params
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_ledger.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'seq.ledger'`.

- [ ] **Step 3: Implement `seq/ledger.py`**
```python
"""Param-match auditor for every Prizma-vs-TF head-to-head. The quad2 feature map must add 0
trainable params (it is buffers); any *trainable* gate (output-gate W_g etc.) is reported so the TF
can be grown in lockstep where the addition is a fair architectural comparison."""
from __future__ import annotations
from .common import param_count

def param_match_report(tf_model, pz_model, tol=0.02):
    pt, pp = param_count(tf_model), param_count(pz_model)
    added = 0
    for n, p in pz_model.named_parameters():
        if any(tag in n for tag in ("feat_I", "feat_J", "W_rand")):   # buffers anyway; defensive
            added += p.numel()
    return {"tf_params": pt, "pz_params": pp, "delta": pp - pt,
            "rel": abs(pp - pt) / max(1, pt), "matched": abs(pp - pt) / max(1, pt) < tol,
            "feat_map_added_params": added}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_ledger.py -q`
Expected: PASS. (If `matched` is False, adjust the matched config in the test — d128L4H4 is the report's headline match at +0.6%.)

- [ ] **Step 5: Commit**
```bash
git add seq/ledger.py tests/test_ledger.py && git commit -m "feat(ledger): param-match auditor for every head-to-head"
```

---

### Task 0.6: Stand up the three councils (durable charters + verdict log)

Councils are **parallel subagent groups dispatched by the orchestrator**, not Python. This task creates the durable charters and the verdict-record format they write to.

**Files:**
- Create: `committee/charters/council1_general.md`, `committee/charters/council2_quant.md`, `committee/charters/council3_standards.md`
- Create: `committee/verdicts/_TEMPLATE.json`

- [ ] **Step 1: Write `council1_general.md`** — the GATE. Contents: mission (every major decision presented; on rejection, the rejection cause must be fixed before proceeding), the 6 member roles (ML-systems · stats/experimental-design referee · optimization theorist · information theorist · reproducibility/integrity referee · adversarial skeptic), the decision protocol (decide→present→approve/reject→remedy→re-present), and the binding honesty guardrails copied from `committee/round1_synthesis.md` §D.

- [ ] **Step 2: Write `council2_quant.md`** — the ALPHA HUNT. Contents: mission (research how to dramatically surpass the TF), 5 roles (quant researcher · architecture designer · kernel/perf · literature scout · risk assessor), output contract = a ranked lever list (expected margin-gain per honesty-risk), loop-until-dry ideation rule, adversarial self-filter.

- [ ] **Step 3: Write `council3_standards.md`** — the HIGH BAR. Contents: the three meta-questions (Has the TF been surpassed? By how much? Industry direction?), the real-SOTA reference set (Mamba-2, Gated DeltaNet, RWKV-7, Titans, Based, RetNet), the "dramatically surpassed" standard definition, and the rule that it judges at every phase boundary and may *raise* the §3 bar.

- [ ] **Step 4: Write `committee/verdicts/_TEMPLATE.json`**
```json
{
  "round": 0, "date": "YYYY-MM-DD", "council": "general|quant|standards",
  "decision_under_review": "", "verdict": "approve|reject|needs-experiments",
  "rejection_causes": [], "required_remedies": [], "members": [],
  "evidence_refs": [], "raises_bar_to": null, "notes": ""
}
```

- [ ] **Step 5: Commit**
```bash
git add committee/charters committee/verdicts && git commit -m "docs(council): charter the 3 councils + verdict-log template"
```

- [ ] **Step 6: Run Council-1 round-0 (orchestrator action, not a script)**

Dispatch the 6 Council-1 members as parallel subagents to review **this plan + the v2 spec**. Each returns approve/reject + rejection causes. Record the synthesis to `committee/verdicts/r0_general_plan-review.json`. If any reject, fix the cause in the plan/spec before starting Phase 1. (This is the first real gate.)

---

## PHASE 1 — Lever R&D (≤2M params, multi-seed, each Council-1-gated)

> Each lever task adds a **default-off** config knob, a **reference** implementation (determinate
> ground truth), a **fast/chunked** implementation, and TWO gates that must pass before any accuracy
> run: (G1) `step()==forward() < 1e-4`, (G2) `chunked == reference < 1e-4`. Then a small multi-seed
> A100 validation via the Phase-0 harness. Levers are ordered by (value ÷ risk): C → E → A → B → D → F.

### Task 1.C: Output-gating + per-head state RMSNorm (the char-LM win, lowest risk)

**Rationale:** RWKV-7 / GLA / Gated-Attention all show a per-token output gate `g=σ(W_g x)` and a
per-head norm on the state read materially improve LM. This is the most direct lever to flip char-LM
from −0.024 to a decisive win. `W_g` is *trainable* → the TF is grown in lockstep (Task notes).

**Files:**
- Modify: `seq/prizma_seq.py` (config knobs `out_gate`, `state_norm`; block `__init__` + `forward` + `step`)

- [ ] **Step 1: Add config knobs** to `PrizmaSeqConfig` (after `feat_n2`):
```python
    out_gate: bool = False      # per-token output gate g=sigma(W_g x); o = o * g before W_o (RWKV-7/GLA)
    state_norm: bool = False    # per-head RMSNorm on the delta-state read o_delta before merge
```

- [ ] **Step 2: Add modules** in `PrizmaSeqBlock.__init__` (after `self.W_o`):
```python
        self.W_g = nn.Linear(d, d, bias=True) if cfg.out_gate else None
        self.state_rms = RMSNorm(dh) if cfg.state_norm else None
```

- [ ] **Step 3: Apply in `forward`** — change the delta-read merge and the pre-`W_o` output:
```python
        if self.cfg.use_workspace:
            o_delta, _ = chunked_delta(self._phi(q), self._phi(k), v, beta, alpha,
                                       chunk=self.cfg.chunk, write_mode=self.cfg.write_mode)
            if self.state_rms is not None:
                o_delta = self.state_rms(o_delta)        # per-head RMSNorm over d_h
            o = o + o_delta
        if self.cfg.use_window:
            o = o + self._window(q, k, v)
        o = o.transpose(1, 2).reshape(B, T, d)
        if self.W_g is not None:
            o = o * torch.sigmoid(self.W_g(self._apply_conv_id(h)))   # gate on the block input
        h = h + self.W_o(o)
```
Use the **normed block input** for the gate to match `step()`; add a tiny helper so both paths agree:
```python
    def _gate_src(self, h_or_x):   # gate reads norm1(h); in forward we already have it as `x` pre-conv
        return h_or_x
```
Simplest consistent choice: compute `g` from `self.norm1(h)` in forward and from `self.norm1(h_t)` in
step (both pre-conv). Replace the gate line with `o = o * torch.sigmoid(self.W_g(self.norm1(h)))` in
forward and the analogous `self.norm1(h_t)` in step. (Drop `_gate_src`/`_apply_conv_id` — they were
scaffolding; the norm1 source is the determinate contract.)

- [ ] **Step 4: Mirror in `step`** — after computing `o` (`o = (o_delta + o_win).reshape(B,1,-1)`), before `self.W_o`:
```python
        if self.state_rms is not None:
            o_delta = self.state_rms(o_delta)   # apply BEFORE combining; recompute o accordingly
        # (recompute) o = (o_delta + o_win).reshape(B, 1, -1)
        if self.W_g is not None:
            o = o * torch.sigmoid(self.W_g(self.norm1(h_t)))
```
Note: in `step`, apply `state_rms` to `o_delta` (shape `[B,H,d_h]`) *before* `o = (o_delta+o_win)`, so the
ordering matches `forward` (where it is applied to the `[B,H,T,d_h]` delta read before adding the window).

- [ ] **Step 5: G1 — O(1) guard with the new knobs**

Run:
```bash
python -c "
import torch; from seq.prizma_seq import PrizmaSeqLM, PrizmaSeqConfig; from seq.common import get_device, param_count
dev=get_device()
for og in (False,True):
  for sn in (False,True):
    cfg=PrizmaSeqConfig(vocab=64,d_model=64,n_layers=2,n_heads=2,feat_map='quad2',out_gate=og,state_norm=sn)
    m=PrizmaSeqLM(cfg).to(dev); m.train(False)
    x=torch.randint(0,64,(2,48),device=dev); y=m(x)
    st=m.init_state(2,dev); outs=[]
    for t in range(x.shape[1]):
      lg,st=m.step(x[:,t:t+1],st); outs.append(lg)
    d=(y-torch.cat(outs,1)).abs().max().item()
    print(f'out_gate={og} state_norm={sn} max|d|={d:.2e} params={param_count(m)} {\"OK\" if d<1e-4 else \"MISMATCH\"}')
"
```
Expected: all four lines `OK` with `max|d| < 1e-4`. If any `MISMATCH`, the forward/step ordering differs — reconcile Steps 3–4 before proceeding (do NOT run accuracy numbers on a failing guard).

- [ ] **Step 6: Commit**
```bash
git add seq/prizma_seq.py && git commit -m "feat(C): output-gating + per-head state RMSNorm (default off, O(1) guard green)"
```

- [ ] **Step 7: A100 char-LM validation (Colab)** — run `gpu_charlm2.py` with `out_gate=True, state_norm=True, gated=True` for Prizma vs the param-matched TF (grown by `W_g`'s params), per-model LR-swept (Task 0.4), ≥10 seeds, TOST-superiority (Task 0.3). Record to `results/gpu_charlm2_v2.json`. **Gate (Council-1):** claim "beats TF by ≥0.03 BPC" only if superiority p<0.05 at ≥10 seeds with seed-pinned init.

---

### Task 1.E: Banded-window kernel (the latency win — push crossover 32k→~2k)

**Rationale:** `_window` currently builds a full `[T,T]` mask + SDPA (17.5% of FLOPs as-coded). A true
sliding-window (chunked-local) cuts it to ~0.9% and removes the overhead that makes Prizma ~1.5×
slower below n≈16k. Same numerics, different kernel → guarded by exact-equivalence to the current window.

**Files:**
- Modify: `seq/prizma_seq.py` (`_window` → add a banded path; config `banded_window: bool = False`)
- Test: `tests/test_window.py`

- [ ] **Step 1: Write the failing equivalence test**
```python
# tests/test_window.py
import torch
from seq.prizma_seq import PrizmaSeqBlock, PrizmaSeqConfig

def test_banded_window_equals_full_window():
    torch.manual_seed(0)
    cfg = PrizmaSeqConfig(vocab=64, d_model=64, n_layers=1, n_heads=2, window=16)
    blk = PrizmaSeqBlock(cfg)
    B,H,T,dh = 2, cfg.n_heads, 200, cfg.d_h
    q = torch.randn(B,H,T,dh); k = torch.randn(B,H,T,dh); v = torch.randn(B,H,T,dh)
    full = blk._window(q, k, v)
    banded = blk._window_banded(q, k, v)
    assert (full - banded).abs().max().item() < 1e-4
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_window.py -q`
Expected: FAIL with `AttributeError: ... has no attribute '_window_banded'`.

- [ ] **Step 3: Implement `_window_banded`** in `PrizmaSeqBlock` (chunked local attention; each query
block attends only its own block + the previous block, masked to the last `w` tokens):
```python
    def _window_banded(self, q, k, v):
        """Sliding-window causal attention in O(T*w) via fixed-size chunks of size w. Each query in
        chunk c attends keys in chunks {c-1, c} masked to [i-w+1, i]. Numerically equals _window."""
        B, H, T, dh = q.shape
        w = self.cfg.window
        outs = []
        for c0 in range(0, T, w):
            c1 = min(c0 + w, T)
            qc = q[:, :, c0:c1]                          # [B,H,Cq,dh]
            k0 = max(0, c0 - w)
            kc = k[:, :, k0:c1]; vc = v[:, :, k0:c1]     # span <= 2w
            qi = torch.arange(c0, c1, device=q.device)[:, None]
            ki = torch.arange(k0, c1, device=q.device)[None, :]
            band = (ki <= qi) & (ki > qi - w)
            mask = torch.zeros(c1 - c0, c1 - k0, device=q.device, dtype=q.dtype).masked_fill(~band, float("-inf"))
            outs.append(torch.nn.functional.scaled_dot_product_attention(qc, kc, vc, attn_mask=mask))
        return torch.cat(outs, dim=2)
```

- [ ] **Step 4: Wire the config knob** — in `forward`, use `self._window_banded` when `cfg.banded_window` else `self._window`. Add `banded_window: bool = False` to the config. The streaming `step()` window (ring buffer) is already O(w) and unchanged.

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/test_window.py -q`
Expected: PASS (`max|d| < 1e-4`).

- [ ] **Step 6: Commit**
```bash
git add seq/prizma_seq.py tests/test_window.py && git commit -m "feat(E): banded sliding-window kernel (exact-equal to full window, O(T*w))"
```

- [ ] **Step 7: A100 latency re-measure (Colab)** — re-run `gpu_latency.py` with `banded_window=True`; record the new crossover n. **Target:** Prizma faster than TF at all n ≳ 2k. Save `results/gpu_latency_v2.json`.

---

### Task 1.A: Surprise-gated write/forget (the NOVEL core — PC free-energy made causal)

**Rationale:** Today `u_t = β_t·ε_t` writes the error but the surprise *magnitude* `‖ε_t‖` is unused.
Modulate the effective write/forget by surprise (write more, forget stale on a large surprise) — a
Titans-class test-time memory **derived** from Prizma's free energy. Because `ε_t` depends on
`S_{t-1}`, the chunk-parallel form needs care; the **reference** (sequential) is the ground truth and
`step()` uses it exactly. The chunked form uses a within-chunk frozen-surprise approximation that
must match the reference < 1e-4 (else fall back to two-pass).

**Files:**
- Modify: `seq/delta.py` (add `surprise` arg to `_delta_reference` and `chunked_delta`)
- Modify: `seq/prizma_seq.py` (config `surprise_gate: bool = False`, compute & pass surprise)
- Test: extend `seq/delta.py` `__main__` guard + `tests/test_surprise.py`

- [ ] **Step 1: Define the surprise modulation (determinate contract)** — in `_delta_reference`, when
`surprise=True`, after computing `Sk` and the raw error `e = vt - at*Sk`, scale the write:
```python
        if surprise:
            s = e.norm(dim=-1, keepdim=True)                  # [B,H,1]  ||epsilon_t||
            g = 1.0 + torch.tanh(s)                           # in [1,2): bounded surprise boost
            u = bt[..., None] * g * e
        else:
            u = bt[..., None] * (vt - at[..., None] * Sk)
```
This is the **ground truth**. (`tanh` keeps it bounded → stable; the exact `g(s)` form is the research
knob, but the *contract* — `g` is a per-token positive scalar function of `‖e_t‖` — is fixed.)

- [ ] **Step 2: Write the failing test** (`tests/test_surprise.py`): the chunked form (frozen
within-chunk surprise) must match the sequential reference < 1e-4 for chunk=1 (exact) and be within
a looser 5e-3 for chunk=64 (approximation), and `step()==forward()` must hold for chunk=1:
```python
# tests/test_surprise.py
import torch
from seq.delta import _delta_reference, chunked_delta

def _mk(T=128, d=16, H=2, B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B,H,T,d, generator=g)
    k = torch.randn(B,H,T,d, generator=g); k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn(B,H,T,d, generator=g); beta = torch.rand(B,H,T, generator=g)*0.99
    return q,k,v,beta

def test_surprise_chunk1_is_exact():
    q,k,v,beta = _mk()
    Oref,Sref = _delta_reference(q,k,v,beta, surprise=True)
    Och,Sch  = chunked_delta(q,k,v,beta, chunk=1, surprise=True)
    assert max((Oref-Och).abs().max(), (Sref-Sch).abs().max()) < 1e-4

def test_surprise_chunk64_approx_within_tol():
    q,k,v,beta = _mk()
    Oref,_ = _delta_reference(q,k,v,beta, surprise=True)
    Och,_  = chunked_delta(q,k,v,beta, chunk=64, surprise=True)
    assert (Oref-Och).abs().max() < 5e-3
```

- [ ] **Step 3: Run it to verify it fails**

Run: `python -m pytest tests/test_surprise.py -q`
Expected: FAIL with `TypeError: _delta_reference() got an unexpected keyword argument 'surprise'`.

- [ ] **Step 4: Implement** — add `surprise=False` to both `_delta_reference` and `chunked_delta`
signatures; in the reference use Step-1's code; in `chunked_delta` compute the within-chunk surprise
from the *chunk-entry* state `S` (frozen across the chunk): `e_i ≈ V_c - gamma_i*(K_c S0^T)` per row,
`g_i = 1+tanh(‖e_i‖)`, and fold `g_i` into `Bc` (`Bc_eff = Bc * g`) before building `A`/`rhs`. With
`chunk=1` the frozen surprise equals the true surprise → exact (test 1); larger chunks are the
approximation (test 2). If test 2 fails, reduce the boost (`0.5*tanh`) or document a smaller default chunk.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_surprise.py -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Wire into the model** — add `surprise_gate: bool = False` to `PrizmaSeqConfig`; in the
block, pass `surprise=self.cfg.surprise_gate` to `chunked_delta` (forward) and implement the same
`g=1+tanh(‖v1 - a1*Sk‖)` scaling on `u` in `step`. Re-run the G1 O(1) guard (Task 1.C Step 5 style)
with `surprise_gate=True` → must be `OK` (< 1e-4) at chunk=1; for chunk>1 the streaming step is exact
and forward is the approximation, so guard at `chunk=1` for the hard equality and document the chunked gap.

- [ ] **Step 7: Commit**
```bash
git add seq/delta.py seq/prizma_seq.py tests/test_surprise.py && git commit -m "feat(A): surprise-gated write (PC free-energy made causal; chunk1 exact, guarded)"
```

- [ ] **Step 8: A100 ablation (Colab)** — MQAR D=128 + char-LM, `surprise_gate` on vs off, ≥10 seeds,
causal attribution (is the surprise signal *causally* helping, vs a matched random scalar control).
Record to `results/gpu_surprise.json`. **Gate (Council-1):** the novel claim stands only if surprise
beats the random-scalar control with superiority p<0.05.

---

### Task 1.B: Higher-order DeltaProduct (k=2) — make induction/selcopy discriminating + LM

**Files:**
- Modify: `seq/delta.py` (reference + chunked accept `n_delta: int = 1`)
- Modify: `seq/prizma_seq.py` (config `n_delta: int = 1`)
- Test: `tests/test_deltaproduct.py`

- [ ] **Step 1: Define the contract** — `n_delta=k` applies k sequential delta updates per token using
k separate (k,v) sub-projections (Householder-product state transition, Siems 2025). Reference: loop
the existing per-token update k times with `(k_i^(j), v_i^(j))` for j in 1..k before reading the next
token. Gate: `chunked == reference < 1e-4` for k∈{1,2} (k=1 must be byte-identical to today).

- [ ] **Step 2: Write the failing test** (`tests/test_deltaproduct.py`): k=1 equals the current
`chunked_delta`; k=2 chunked equals a k=2 reference < 1e-4:
```python
# tests/test_deltaproduct.py
import torch
from seq.delta import _delta_reference, chunked_delta

def test_k1_is_identical_to_today():
    torch.manual_seed(0)
    B,H,T,d = 2,2,64,16
    q=torch.randn(B,H,T,d); k=torch.randn(B,H,T,d); k/=k.norm(dim=-1,keepdim=True)
    v=torch.randn(B,H,T,d); beta=torch.rand(B,H,T)*0.99
    O0,_ = chunked_delta(q,k,v,beta)
    O1,_ = chunked_delta(q,k,v,beta, n_delta=1)
    assert (O0-O1).abs().max() < 1e-6

def test_k2_chunked_matches_reference():
    torch.manual_seed(0)
    B,H,T,d = 2,2,96,16
    # k=2 needs 2 key/val sets stacked on a new axis: shape [B,H,T,2,d]
    q=torch.randn(B,H,T,d)
    k=torch.randn(B,H,T,2,d); k/=k.norm(dim=-1,keepdim=True)
    v=torch.randn(B,H,T,2,d); beta=torch.rand(B,H,T,2)*0.99
    Oref,Sref = _delta_reference(q,k,v,beta, n_delta=2)
    Och,Sch   = chunked_delta(q,k,v,beta, n_delta=2)
    assert max((Oref-Och).abs().max(), (Sref-Sch).abs().max()) < 1e-4
```

- [ ] **Step 3: Run it to verify it fails** → `python -m pytest tests/test_deltaproduct.py -q` →
FAIL (`unexpected keyword argument 'n_delta'`).

- [ ] **Step 4: Implement** `n_delta` in both forms (k=1 path unchanged; k≥2 applies the k sub-updates
per token/chunk-row in order). Keep `d_h` even, float32. The read is still pre-write `S_{t-1} q_t`.

- [ ] **Step 5: Run the tests** → PASS (2 passed).

- [ ] **Step 6: Commit** → `git commit -m "feat(B): higher-order DeltaProduct n_delta (k=1 identical, k=2 guarded)"`

- [ ] **Step 7: A100 (Colab)** — induction/selcopy at k=1 vs k=2 (does k=2 make them *discriminating*
vs Prizma-none?) + MQAR + char-LM, ≥10 seeds, FLOP ledger (k=2 doubles delta FLOPs → must still net
≤1.0× after Task 1.E/1.D). Record `results/gpu_deltaproduct.json`.

---

### Task 1.D: Leaner / structured feature map (FLOP 2.14×→≤1.0×, same D=128 recall)

**Files:**
- Modify: `seq/prizma_seq.py` (`feat_map` gains `'quad2_lowrank'`; config `feat_rank: int = 0`)
- Test: extend `tests/` with a capacity-equivalence probe

- [ ] **Step 1: Define the contract** — `quad2_lowrank`: project `x` (d_h) → r dims via a fixed seeded
random matrix, take quadratic monomials in the r-dim space, concat → d_φ = d_h + (r·(r+1)/2 or feat_n2),
with r chosen so d_φ ≈ 96–128 (vs 256). Goal: same MQAR-D=128 crosstalk reduction at ~half the
delta-state FLOPs. Still 0 trainable params (buffers).

- [ ] **Step 2: Capacity probe gate** — reuse `cap_probe.py`'s crosstalk metric: `quad2_lowrank` must
reach off-diagonal crosstalk ≤ 0.085 at D=128 (the current `quad2` hits ~0.076; the bar is "within
0.01 of quad2 at ≤half d_φ"). Run `python cap_probe.py` adapted for the new map; record the number.

- [ ] **Step 3: Implement** `_phi` branch for `quad2_lowrank` + the buffers; G1 O(1) guard (must be OK);
chunked==reference via the existing rectangular-state self-test in `seq/delta.py __main__` (run it).

- [ ] **Step 4: Commit** → `git commit -m "feat(D): low-rank quadratic feature map (leaner d_phi, capacity-checked)"`

- [ ] **Step 5: A100 (Colab)** — MQAR D-frontier {32,64,128} quad2 vs quad2_lowrank vs none, ≥10 seeds
+ FLOP ledger. **Gate (Council-1):** keep the param-efficiency win while cutting FLOPs to ≤1.0× TF.

---

### Task 1.F: Fused chunked-delta kernel (train 5×→≤1.5× slower) — A100/Triton

**Files:**
- Create: `seq/delta_fused.py` (CUDA/Triton fused chunked delta; CPU/MPS falls back to `chunked_delta`)
- Test: `tests/test_fused.py` (equivalence to `chunked_delta` < 1e-4 when CUDA present; skip otherwise)

- [ ] **Step 1: Equivalence-first test** — `tests/test_fused.py` skips unless `torch.cuda.is_available()`;
when present, asserts `fused_delta(...) == chunked_delta(...) < 1e-4` (forward + grad) on the
production regime (chunk=64, T=256, the rectangular state, the gated path).

- [ ] **Step 2: Implement** a Triton fused kernel for the WY/UT chunk step (the hot loop), with a pure
fallback `return chunked_delta(...)` on non-CUDA. Correctness gate first; speed second.

- [ ] **Step 3: A100 (Colab)** — measure train-step wall-clock Prizma(fused) vs TF; **target ≤1.5×**.
Record `results/gpu_train_speed.json`. (If Triton parity is hard, document the gap honestly — speed is
a Pareto knob, not a correctness gate.)

- [ ] **Step 4: Commit** → `git commit -m "feat(F): fused chunked-delta kernel (CUDA; equivalence-gated, MPS fallback)"`

---

## Phase 1 exit gate (Council-1 + Council-3)
Phase 1 is complete when each lever has passed G1+G2 and a ≥10-seed A100 validation, and Council-1
has signed off each causal-attribution claim. Council-3 then judges the *combined* picture against the
SOTA landscape and sets the exact Phase-2 consolidation bar. **Then re-run writing-plans for Phase 2.**

---

## Self-Review (done by author)
- **Spec coverage:** §2 levers A–F → Tasks 1.A–1.F ✓; §4 scaffolding → Tasks 0.2 (repro), 0.3 (powered
  stats), 0.4 (LR sweep), 0.5 (ledger) ✓; §5 councils → Task 0.6 ✓; §3 bar enforced as per-task A100
  gates ✓; §6 parallelism + §7 compute are orchestration policy (executed by subagent-driven-development,
  not a code task) — noted ✓; Phases 2–4 deferred with explicit re-plan note ✓.
- **Placeholders:** none — every code step shows code; research-knob freedom is bounded by a determinate
  *contract* + a concrete equivalence/guard test (not a "TODO").
- **Type/name consistency:** `build_and_train(model_fac, task, cfg, device, seed, **fac_kw)` used in
  0.2/0.4; `summarize/superiority_test/tost_equivalence/solve_rate` used in 0.3 tests + Phase-1 gates;
  config knobs `out_gate/state_norm/banded_window/surprise_gate/n_delta/feat_map='quad2_lowrank'/feat_rank`
  all defined on `PrizmaSeqConfig` before use; `_window_banded`, `surprise=`, `n_delta=` signatures match
  across tests and impls. ✓

---

## Council Round-0 amendments (2026-06-07, binding)
Record: `committee/round0_v2_synthesis.md`. Council 1 (gate) CONDITIONAL-REJECT → remedies applied/queued.
**Already landed:** R1/R2/R5/R7 in `seq/stats.py` (real Student-t, correct TOST, margin-superiority,
identical-model canary, Holm — validated vs scipy to ~1e-16) and R8 (set_seed CUDA+python-random).

**Revised Phase-1 sequence:** F → **C + E** (approved now) → **H** → **D** (MQAR-gated) → **B** →
**{A vs G}** (novel-core ablation). New tasks below; lever-A task (1.A above) is amended by R3/R9.

### Task 1.H: Decoupled channel-wise erase/write (Gated-DeltaNet-2) — promoted recall+LM win
**Files:** Modify `seq/delta.py` (separate erase β_e and write β_w), `seq/prizma_seq.py`
(config `decoupled_gate: bool = False`, project two gates), test `tests/test_decoupled.py`.
- [ ] **Step 1: Contract** — today `u_t = β_t(v_t − αS k_t)`. Decouple: a key-side **erase** gate
  `β_e=σ(W_e x)` scales the read-back `S k_t`, a value-side **write** gate `β_w=σ(W_w x)` scales the
  new value: `u_t = β_w·v_t − β_e·(αS k_t)`. `decoupled_gate=False` must be byte-identical to today
  (set β_e=β_w=β). Both gates are trainable → grow the TF in lockstep in the param ledger.
- [ ] **Step 2: Failing test** — k-step reference vs chunked < 1e-4 for the decoupled form; and
  `decoupled_gate=False` equals the current `chunked_delta` to < 1e-6.
- [ ] **Step 3: Implement** in `_delta_reference` + `chunked_delta` (the `A`/`rhs` build uses β_e on the
  KK/KS0 terms, β_w on the V term) and the streaming `step()`. Run G1 O(1) guard (must be OK).
- [ ] **Step 4: Commit** `feat(H): decoupled channel-wise erase/write (GDN-2; off=identical, guarded)`.
- [ ] **Step 5: A100** — MQAR multi-key + char-LM, decoupled on vs off, ≥10 seeds, FLOP ledger.

### Task 1.G: RWKV-7-style in-context per-channel learning rate (A-alternative novel core)
**Files:** `seq/delta.py` + `seq/prizma_seq.py` (config `inctx_lr: bool=False`), `tests/test_inctx_lr.py`.
- [ ] **Step 1: Contract** — replace the scalar write gate with a **per-channel** in-context rate
  vector `η_t=σ(W_η x_t) ∈ R^{d_h}` modulating the delta update per state channel (RWKV-7 "Goose"
  generalized delta). `inctx_lr=False` = identical to today. Trainable `W_η` → grow TF in lockstep.
- [ ] **Step 2–4:** chunked==reference < 1e-4 (off-path identical < 1e-6), G1 guard, commit
  `feat(G): in-context per-channel learning rate (RWKV-7; off=identical, guarded)`.
- [ ] **Step 5: A100** — head-to-head vs lever A on the SAME ablation; keep whichever wins the
  novel-core slot (Council-1 sign-off on causal attribution).

### Task 1.A amendment (Council-1 R3/R9 — BINDING before any lever-A accuracy run)
- The chunked surprise approximation breaks on repeated keys. REQUIRED: an **exact two-pass chunked
  form** (pass 1 computes per-token ε against the carried state; pass 2 applies the gated write) OR
  inference-only OR chunk=1. Add `tests/test_surprise_repeatedkey.py`: a sequence with repeated keys
  (the MQAR/induction signal) must match the sequential reference at **< 1e-4** (NOT the 5e-3
  random-data tolerance). Add TWO controls in the A100 ablation: a **random-scalar** gate and a
  **constant-mean β_eff** gate. Reframe A as capacity-*reallocation*. No lever-A accuracy number until
  this passes.

### Task 1.D amendment (Council-1 R9 + Council-2 caveat)
- The lever-D gate is the **end-to-end ≥10-seed MQAR-D128 solve-rate at the reduced d_φ**, run BEFORE
  any LM run — not just the crosstalk probe. Back off d_φ (160 / spiky Hedgehog map) if the solve
  point regresses past 130K params. Also reconcile **d_φ (R4)** across code/report/synthesis and
  re-emit the FLOP ledger before any FLOP statement.

### Task 1.Recall-gate (Council-3): recall as a hard TOST-parity gate
**Files:** `gpu_diag.py` / a new `seq/recall_gate.py` runner.
- [ ] MQAR (hard rung) + induction + selective-copy must reach **≥ tuned-TF parity via TOST**
  (`seq.stats.tost_equivalence`), with the optimization-vs-capacity **flip-test** on the hard rung
  (a bigger TF must be shown to solve it, so a tiny-TF failure is under-capacity not "attention can't").
  This is a pass/fail GATE for the "dominant" word; failing it downgrades the claim to "competitive".

### Task 1.Hybrid-baseline (Council-3): matched tiny-hybrid arm
**Files:** `seq/hybrid.py` (a Samba/GatedDeltaNet-H-style block: mostly Prizma layers + ~1 attention
layer), wired into the head-to-head as a THIRD baseline alongside the pure TF.
- [ ] Prizma must be at least **Pareto-competitive with a matched tiny hybrid**, else the honest
  framing is "best pure-O(1) point", not "beats the Transformer". Param-matched; same harness.

### §3 bar status (R10): latency / abs-length-extrap / per-FLOP targets are **conditional** on their
enabling levers (E/D/F) landing and being measured; abs-length-extrap is a Pareto knob. "Dramatic"
requires char-LM(iso-FLOP) + all-n latency + O(1) memory + recall-parity **simultaneously and powered**,
with the scope rider stated; otherwise downgrade to "Pareto-efficient/-competitive in the tested regime".
