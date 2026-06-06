"""PRISM-Seq vs Transformer — the remaining diagnostic legs (INDUCTION + SELECTIVE-COPY), for a
CUDA GPU (Colab). Closes the §1/§2 "diagnostic suite" gap that `gpu_bench.py` left open (it covered
MQAR D=128). Mirrors `gpu_bench.py` EXACTLY in structure/protocol and REUSES its helpers
(`run_cell`, `solve_stats`, `_load`, `_save`, `tf_factory`, `ps_factory`, `GENWARM`, `DEV`) so the
two runners compose into the same `$PRISM_RESULTS` directory.

Pre-registered protocol (`docs/superpowers/specs/2026-06-05-...-design.md` §1), matched + fair:

  PARAM-MATCHED ARENA   primary scale d128L2H4 (~461K params) — the SAME scale at which the tuned TF
                        cleanly SOLVES MQAR D=128 in gpu_bench.py (best~1.0), so it is a genuine
                        non-strawman baseline for these (easier) tasks. Arms:
                          TF           = tf_factory(d,L,H)                       (tuned reference)
                          PRISM-quad2  = ps_factory(..., feat_map='quad2', feat_n2=224)
                          PRISM-none   = ps_factory(...)                         (ablation control)
                        All three are param-matched to within ~1% (printed at startup).

  LEG: INDUCTION        in-context induction-head probe. Trained on a MIXED-length distribution over
                        prefix lengths {64,128,256} (a thin in-file wrapper that COMPOSES the existing
                        seq.tasks.Induction — the model/task code is untouched), so a single model is
                        fairly graded at every length. EVALUATED at fixed prefix lengths 64 / 128 / 256
                        on a FROZEN reproducible eval set (same `cfg.eval_seed` batches as training's
                        frozen eval -> reproducible, no best-of-noisy-curve inflation). Bar (§1.2):
                        PRISM-quad2 >= 0.98, and >= 0.95 at the 256 gap. 3 seeds. solve threshold 0.98.

  LEG: SELECTIVE-COPY   Mamba's content-selective copy. Trained + scored on SelectiveCopy(fixed=False)
                        (selective) AND a fixed=True CONTROL (must be ~equal across arms -> isolates
                        content-selectivity, not raw copy). Bar (§1.3): selective >= 0.97 and >= TF-0.02.
                        3 seeds.

Identical for all arms: same task / budget (cap + plateau early-stop) / frozen eval / plateau rule.
Per-model optimizer constants follow gpu_bench's pattern (GENWARM). max_len is set to cover each
task's seq_len (+8) by the imported factories. O(1) discipline preserved (models untouched).

Results stream incrementally + crash-safe to $PRISM_RESULTS/gpu_diag.json via the imported `_save`
(completed cells skipped by cellkey -> a Colab disconnect never loses progress; resumable). A
`diag_summary` records per-leg solve_rate + median + bests for each arm.

Env: set PRISM_RESULTS to a Drive-mounted dir for persistence (default ./results).
Run: python3 gpu_diag.py                 # both legs
     python3 gpu_diag.py induction        # only listed legs (also accepts: selcopy)
     python3 gpu_diag.py induction selcopy
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

# Reuse the VERIFIED gpu_bench machinery verbatim (do NOT duplicate run_cell / solve_stats / IO /
# factories): this guarantees identical protocol + that both runners write to the same results dir.
import gpu_bench
from gpu_bench import (  # noqa: F401  (re-exported for symmetry / external import)
    run_cell, solve_stats, tf_factory, ps_factory, _median, ci95, GENWARM, DEV,
)
from seq.common import TrainConfig, train_model, param_count, set_seed, masked_acc
from seq.tasks import Induction, SelectiveCopy
from seq.transformer import Transformer, TFConfig
from seq.prism_seq import PRISMSeqLM, PRISMSeqConfig

# Dedicated results file (sibling of gpu_bench.json, same dir). We rebind gpu_bench.OUT to this so
# the imported `run_cell`/`_save` (which close over gpu_bench.OUT) stream here, crash-safe.
RES = os.environ.get("PRISM_RESULTS", os.path.join(os.path.dirname(__file__), "results"))
os.makedirs(RES, exist_ok=True)
OUT = os.path.join(RES, "gpu_diag.json")
gpu_bench.OUT = OUT          # <- imported _load/_save/run_cell now target gpu_diag.json
_load = gpu_bench._load
_save = gpu_bench._save

# Pre-registered primary scale: the param-matched arena where the tuned TF cleanly solves MQAR D=128.
SCALE = (128, 2, 4)          # d_model, n_layers, n_heads  (~461K params, per gpu_bench p1.TF.d128L2H4)
FEAT_N2 = 224                # quad2 monomials -> rectangular delta state (matches gpu_bench p2 default)
SEEDS = (0, 1, 2)
IND_LENS = (64, 128, 256)    # induction prefix lengths to train-mixed over + evaluate at
IND_VOCAB = 32               # seq.tasks.Induction default vocab
SC_VOCAB, SC_MEM, SC_K = 32, 64, 16   # SelectiveCopy: filler=0, marker=V-1; seq_len = mem+1+k = 81


# --------------------------------------------------------------------------------------------- #
# Induction trained over a SPECTRUM of prefix lengths. This COMPOSES seq.tasks.Induction (one
# instance per length, built once); it does NOT modify the task. Mirrors the MixedMQAR pattern in
# seq/tasks.py: `sample()` draws a random difficulty (here: prefix length) per batch so one model
# learns the induction circuit at every tested length; `eval_sample()` is FIXED at the longest gap
# (the hardest), so the training-frozen-eval number is the worst-case length. The per-length report
# (eval at 64 / 128 / 256) is computed separately in `induction()` on its own frozen eval set.
class _MixedInduction:
    def __init__(self, vocab=IND_VOCAB, lens=IND_LENS):
        self.vocab = vocab
        self.lens = tuple(sorted(lens))
        self._tasks = {L: Induction(vocab=vocab, seq_len=L) for L in self.lens}
        # seq_len drives max_len sizing in the factories -> size for the LONGEST prefix (+ query).
        self.seq_len = max(t.seq_len for t in self._tasks.values())
        self.name = f"MixedInduction(V={vocab},lens={self.lens})"

    def sample(self, B, device):                       # TRAINING: a random prefix length per batch
        L = int(self.lens[int(torch.randint(0, len(self.lens), (1,)).item())])
        return self._tasks[L].sample(B, device)

    def eval_sample(self, B, device):                  # FROZEN EVAL: fixed at the longest (hardest)
        return self._tasks[self.lens[-1]].sample(B, device)


def _ind_factory(d, L, H, prism=None):
    """Model factory sized for the MIXED-induction longest prefix. `prism=None` -> TF; else kwargs."""
    if prism is None:
        return tf_factory(d, L, H)
    return ps_factory(d, L, H, **prism)


def _run_ind_cell(res, cellkey, model_fac, task_fac, cap, seed, recipe, eval_every,
                  vocab, lens, bs, eb, eseed):
    """Induction cell runner. SAME caching contract as gpu_bench.run_cell (cache by cellkey, crash-
    safe `_save`, resumable), but trains the model ONCE and computes BOTH the worst-case-length
    'best' (= train_model's frozen eval, fixed at the longest prefix) AND the per-length eval from
    that SAME trained model -> no wasteful retrain. Mirrors run_cell's record shape + adds per_len."""
    if cellkey in res and "best" in res[cellkey] and "per_len" in res[cellkey]:
        return res[cellkey]
    task = task_fac()
    model = model_fac(vocab, task.seq_len)
    p = param_count(model)
    cfg = TrainConfig(steps=cap, batch_size=bs, log=False, eval_every=eval_every, **recipe)
    t0 = time.time()
    r = train_model(model, task, cfg, DEV, seed=seed)            # frozen eval == longest-prefix acc
    per_len = {str(k): round(v, 4)
               for k, v in _eval_at_lengths(model, vocab, lens, bs, eb, eseed, DEV).items()}
    rec = {"best": r.best_acc, "plateau": r.steps_to_plateau, "params": p,
           "sec": round(time.time() - t0, 1), "seed": seed, "cap": cap, "per_len": per_len}
    res[cellkey] = rec
    _save(res)
    print(f"   [{cellkey}] best={rec['best']:.3f} plateau@{rec['plateau']} per-len={per_len} "
          f"({rec['sec']}s, {p}p)", flush=True)
    return rec


# --------------------------------------------------------------------------------------------- #
@torch.no_grad()
def _eval_at_lengths(model, vocab, lens, batch_size, eval_batches, eval_seed, device):
    """Per-length frozen masked-accuracy for a TRAINED induction model. Each length gets its own
    FROZEN eval set built under `eval_seed` (the SAME seed train_model uses for its frozen eval) ->
    reproducible across arms / seeds. Returns {length: acc}. Uses the imported `masked_acc`."""
    model.train(False)
    out = {}
    for L in lens:
        task = Induction(vocab=vocab, seq_len=L)
        set_seed(eval_seed)                            # frozen, reproducible held-out batches
        batches = [tuple(task.sample(batch_size, device)) for _ in range(eval_batches)]
        out[L] = float(np.mean([masked_acc(model(x), y, m) for (x, y, m) in batches]))
    return out


def _print_param_match(res):
    """Print param counts for all three arms at the primary scale; confirm match (<~1%)."""
    d, L, H = SCALE
    V = IND_VOCAB
    probes = {
        "TF":          Transformer(TFConfig(vocab=V, d_model=d, n_layers=L, n_heads=H, max_len=300, rope=True)),
        "PRISM-quad2": PRISMSeqLM(PRISMSeqConfig(vocab=V, d_model=d, n_layers=L, n_heads=H, max_len=300,
                                                 feat_map="quad2", feat_n2=FEAT_N2)),
        "PRISM-none":  PRISMSeqLM(PRISMSeqConfig(vocab=V, d_model=d, n_layers=L, n_heads=H, max_len=300)),
    }
    counts = {k: param_count(m) for k, m in probes.items()}
    base = counts["TF"]
    print(f"  param-match @ d{d}L{L}H{H} (V={V}):", flush=True)
    for k, p in counts.items():
        print(f"    {k:<12} {p:>8,}p  ({100.0 * (p - base) / base:+.2f}% vs TF)", flush=True)
    spread = (max(counts.values()) - min(counts.values())) / base
    print(f"    -> max spread {100.0 * spread:.2f}% of TF "
          f"({'MATCHED <=1%' if spread <= 0.01 else 'NOTE: >1% (see report)'})", flush=True)
    res["param_match"] = {"scale": f"d{d}L{L}H{H}", "vocab": V, "counts": counts,
                          "max_spread_frac": round(spread, 4)}
    _save(res)
    return counts


# --------------------------------------------------------------------------------------------- #
def induction(res, scale=SCALE, feat_n2=FEAT_N2, seeds=SEEDS, cap=60000, eval_every=2000):
    """INDUCTION leg: train mixed over prefix {64,128,256}; eval per-length. Bar: quad2 >=0.98, and
    >=0.95 @256. 3 seeds. solve threshold 0.98 for this leg's solve-rate."""
    d, L, H = scale
    print(f"\n==== LEG: INDUCTION @ d{d}L{L}H{H} (mixed-len {IND_LENS}, {len(seeds)} seeds) ====", flush=True)
    task_fac = lambda: _MixedInduction(vocab=IND_VOCAB, lens=IND_LENS)
    arms = {
        "TF":          _ind_factory(d, L, H, None),
        "PRISM-quad2": _ind_factory(d, L, H, dict(feat_map="quad2", feat_n2=feat_n2)),
        "PRISM-none":  _ind_factory(d, L, H, dict()),
    }
    # eval-set knobs MUST match TrainConfig defaults so the per-length frozen eval mirrors train's.
    bs, eb, eseed = TrainConfig.batch_size, TrainConfig.eval_batches, TrainConfig.eval_seed
    summary = {}
    for aname, fac in arms.items():
        recs = []
        per_len_all = {L_: [] for L_ in IND_LENS}
        for s in seeds:
            ck = f"diag.ind.{aname}.s{s}"
            # Trains ONCE; caches 'best' (worst-case-length frozen eval) + per_len. Resumable.
            rec = _run_ind_cell(res, ck, fac, task_fac, cap, s, GENWARM, eval_every,
                                IND_VOCAB, IND_LENS, bs, eb, eseed)
            recs.append(rec)
            for L_ in IND_LENS:
                per_len_all[L_].append(rec["per_len"][str(L_)])
        st = solve_stats(recs)
        # leg-specific solve-rate at the 0.98 bar (solve_stats uses >0.9; recompute at the pre-reg bar)
        bests = [r["best"] for r in recs]
        st["solve_rate_098"] = f"{sum(b >= 0.98 for b in bests)}/{len(bests)}"
        st["per_len_median"] = {str(L_): round(_median(per_len_all[L_]), 4) for L_ in IND_LENS}
        st["per_len_min"] = {str(L_): round(min(per_len_all[L_]), 4) for L_ in IND_LENS}
        summary[aname] = st
        print(f"  -> {aname}: solve@0.98={st['solve_rate_098']} median={st['median']} "
              f"per-len-median={st['per_len_median']}", flush=True)
    # PASS check (descriptive; pre-registered bar §1.2)
    q = summary.get("PRISM-quad2", {})
    pl_med = q.get("per_len_median", {})
    passes = (q.get("median", 0) >= 0.98) and (pl_med.get("256", 0) >= 0.95)
    summary["_bar"] = {"rule": "PRISM-quad2 median>=0.98 AND per-len-median@256>=0.95",
                       "pass": bool(passes)}
    print(f"  ==> INDUCTION bar (quad2 >=0.98, >=0.95@256): {'PASS' if passes else 'NOT MET'}", flush=True)
    res["diag_induction_summary"] = summary
    _save(res)
    return summary


def selcopy(res, scale=SCALE, feat_n2=FEAT_N2, seeds=SEEDS, cap=60000, eval_every=2000):
    """SELECTIVE-COPY leg: selective (fixed=False) + a fixed=True control, per arm. Bar: selective
    >=0.97 and >= TF-0.02; fixed control ~equal across arms (isolates content-selectivity)."""
    d, L, H = scale
    print(f"\n==== LEG: SELECTIVE-COPY @ d{d}L{L}H{H} (selective + fixed control, {len(seeds)} seeds) ====", flush=True)
    variants = {  # variant_name -> task factory
        "selective": lambda: SelectiveCopy(vocab=SC_VOCAB, mem_len=SC_MEM, n_data=SC_K, fixed=False),
        "fixed":     lambda: SelectiveCopy(vocab=SC_VOCAB, mem_len=SC_MEM, n_data=SC_K, fixed=True),
    }
    arms = {
        "TF":          _ind_factory(d, L, H, None),
        "PRISM-quad2": _ind_factory(d, L, H, dict(feat_map="quad2", feat_n2=feat_n2)),
        "PRISM-none":  _ind_factory(d, L, H, dict()),
    }
    summary = {}
    for vname, task_fac in variants.items():
        summary[vname] = {}
        for aname, fac in arms.items():
            recs = [run_cell(res, f"diag.sc.{vname}.{aname}.s{s}", fac, task_fac, cap, s,
                             recipe=GENWARM, eval_every=eval_every) for s in seeds]
            st = solve_stats(recs)
            bests = [r["best"] for r in recs]
            st["solve_rate_097"] = f"{sum(b >= 0.97 for b in bests)}/{len(bests)}"
            summary[vname][aname] = st
            print(f"  -> {vname}/{aname}: solve@0.97={st['solve_rate_097']} "
                  f"median={st['median']} bests={st['bests']}", flush=True)
    # PASS check (descriptive; pre-registered bar §1.3)
    sel = summary.get("selective", {})
    q_med = sel.get("PRISM-quad2", {}).get("median", 0.0)
    tf_med = sel.get("TF", {}).get("median", 0.0)
    fixed = summary.get("fixed", {})
    fx = {a: fixed.get(a, {}).get("median", 0.0) for a in arms}
    fixed_spread = (max(fx.values()) - min(fx.values())) if fx else 1.0
    passes = (q_med >= 0.97) and (q_med >= tf_med - 0.02)
    summary["_bar"] = {
        "rule": "selective PRISM-quad2 median>=0.97 AND >= TF_median-0.02; fixed control ~equal",
        "selective_quad2_median": round(q_med, 4), "selective_tf_median": round(tf_med, 4),
        "fixed_medians": {a: round(v, 4) for a, v in fx.items()},
        "fixed_spread": round(fixed_spread, 4), "pass": bool(passes),
    }
    print(f"  ==> SELECTIVE-COPY bar (sel quad2>=0.97 & >=TF-0.02): {'PASS' if passes else 'NOT MET'}"
          f" | fixed-control spread={fixed_spread:.3f}", flush=True)
    res["diag_selcopy_summary"] = summary
    _save(res)
    return summary


LEGS = {"induction": induction, "selcopy": selcopy}


def main():
    args = [a.lower() for a in sys.argv[1:]] or ["induction", "selcopy"]
    wanted = [a for a in args if a in LEGS]
    if not wanted:
        print(f"no valid legs in {args}; choose from {list(LEGS)}", flush=True)
        return
    print(f"device={DEV} torch={torch.__version__} results={OUT} legs={wanted}", flush=True)
    if DEV.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    res = _load()
    _print_param_match(res)
    for leg in wanted:
        LEGS[leg](res)
    # Compose a top-level diag_summary (per-leg solve_rate + median + bests for each arm).
    diag = {}
    if "diag_induction_summary" in res:
        s = res["diag_induction_summary"]
        diag["induction"] = {a: {"solve_rate_098": s[a].get("solve_rate_098"),
                                 "median": s[a].get("median"), "bests": s[a].get("bests"),
                                 "per_len_median": s[a].get("per_len_median")}
                             for a in ("TF", "PRISM-quad2", "PRISM-none") if a in s}
        diag["induction"]["_bar"] = s.get("_bar")
    if "diag_selcopy_summary" in res:
        s = res["diag_selcopy_summary"]
        diag["selcopy"] = {v: {a: {"solve_rate_097": s[v][a].get("solve_rate_097"),
                                   "median": s[v][a].get("median"), "bests": s[v][a].get("bests")}
                               for a in ("TF", "PRISM-quad2", "PRISM-none") if a in s.get(v, {})}
                           for v in ("selective", "fixed") if v in s}
        diag["selcopy"]["_bar"] = s.get("_bar")
    res["diag_summary"] = diag
    _save(res)
    print("\n==== DONE. Summary keys:", [k for k in res if k.endswith("_summary") or k == "param_match"], flush=True)
    print(f"saved -> {OUT}", flush=True)
    print("\n===DIAG_RESULTS===", flush=True)
    print(json.dumps({"param_match": res.get("param_match"), "diag_summary": diag}, indent=2), flush=True)


if __name__ == "__main__":
    main()
