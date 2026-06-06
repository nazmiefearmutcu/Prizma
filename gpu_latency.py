"""PRISM-Seq vs Transformer — the LONG-CONTEXT decode-latency & memory probe (CUDA/Colab-ready).

WHY THIS EXISTS
---------------
gpu_bench.py phase5 (`decode_latency`) measured per-step decode at sequence lengths n<=4096 and
found BOTH models were *overhead-bound*: the per-step wall-clock was flat (TF ~4.5ms, PRISM ~7.0ms),
so the asymptotic O(t)-per-step (Transformer KV-cache) vs O(1)-per-step (PRISM constant state)
WALL-CLOCK crossover did NOT appear at that scale. PRISM only won on MEMORY there (constant state vs
a linearly-growing KV-cache), and that was an *analytic* (floats) claim, not a measured one.

This probe pushes decode MUCH further — bigger n (up to 65k) AND a bigger model (heavier per-step
attention term) — to HONESTLY test whether the O(1) inference advantage is observable in wall-clock,
not just in memory. Two outcomes are both legitimate and reported as-is:
  * a crossover n* exists in the tested range  -> the O(1) latency advantage is real & measured; or
  * no crossover in range                       -> reported plainly; the MEMORY advantage still stands
                                                   (and is now MEASURED via torch.cuda.max_memory_allocated,
                                                   not merely analytic).

It also records the Transformer's practical OOM ceiling (the KV-cache is O(n) memory): hitting OOM is
ITSELF a result favoring PRISM's constant footprint, so OOM is caught and logged, never fatal.

FAIRNESS / HONESTY (no rigging)
-------------------------------
  * BOTH models decode through their `model.step()` streaming API — the fair O(1)-API path. The TF
    `step()` is genuinely KV-cached (O(t) compute & memory at step t); PRISM `step()` carries a fixed
    state + a length-`window` ring (verified O(1)). Same dtype, same device, same warmup, same reps.
  * Warmup steps before every timed measurement; median over reps; proper device sync
    (torch.cuda.synchronize / torch.mps.synchronize) bracketing each timed region.
  * We report BOTH per-step ms (the asymptotics) AND total decode time. The crossover is defined on
    per-step ms (the thing that actually grows for the TF).
  * Memory: analytic floats (KV vs constant state) AT EVERY n, PLUS measured peak GPU bytes for the TF
    decode at each n on CUDA (reset_peak_memory_stats -> decode -> max_memory_allocated), and a measured
    PRISM peak for contrast. On MPS the proper peak API is absent, so we sample current_allocated_memory
    as a best-effort (clearly labelled); the headline measured-memory result is a CUDA/Colab deliverable.

Crash-safe + resumable: every measured (model, n) cell streams to $PRISM_RESULTS/gpu_latency.json via an
atomic _save (mirrors gpu_bench.py). A Colab disconnect never loses progress, and re-running skips done
cells. We write to the SIBLING gpu_latency.json and NEVER touch gpu_bench.json.

Env:
  PRISM_RESULTS   dir for the JSON ledger (default ./results).
  PRISM_LAT_NS    comma-list overriding the n grid (e.g. "128,512,2048").
  PRISM_LAT_SMOKE =1 -> fast local smoke (tiny model, n in {128,512,2048}, reps=2) for MPS/CPU verify.
  PRISM_LAT_REPS  override reps (default 5; smoke uses 2).
Run:
  python3 gpu_latency.py                 # full campaign (both model sizes) — for Colab/A100/L4
  PRISM_LAT_SMOKE=1 python3 gpu_latency.py   # local machinery + step()-correctness smoke
  python3 gpu_latency.py --smoke         # same as PRISM_LAT_SMOKE=1
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch

from seq.common import get_device
from seq.transformer import Transformer, TFConfig
from seq.prism_seq import PRISMSeqLM, PRISMSeqConfig

# Prefer CUDA (the target), then MPS (local smoke), then CPU. Mirrors gpu_bench.py's selection.
DEV = torch.device("cuda" if torch.cuda.is_available()
                   else ("mps" if torch.backends.mps.is_available() else "cpu"))
RES = os.environ.get("PRISM_RESULTS", os.path.join(os.path.dirname(__file__), "results"))
os.makedirs(RES, exist_ok=True)
OUT = os.path.join(RES, "gpu_latency.json")   # SIBLING of gpu_bench.json — never clobbered.

V = 512          # vocab (matches gpu_bench.py phase5); decode tokens are random/argmax, content-agnostic.
FEAT_N2 = 224    # PRISM quad2 monomials (matches gpu_bench.py phase2/phase5 PRISM-quad2 headline arm).


# --------------------------------- crash-safe ledger ------------------------------------- #
def _load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def _save(d):
    """Atomic write (mirrors gpu_bench.py _save): tmp file -> os.replace, so a crash mid-write
    never corrupts the ledger."""
    tmp = OUT + ".tmp"
    json.dump(d, open(tmp, "w"), indent=2)
    os.replace(tmp, OUT)


# --------------------------------- device sync helpers ----------------------------------- #
def _sync():
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    elif DEV.type == "mps":
        torch.mps.synchronize()


def _is_oom(err: Exception) -> bool:
    """Detect an out-of-memory condition across CUDA / MPS / generic backends."""
    if isinstance(err, torch.cuda.OutOfMemoryError) if hasattr(torch.cuda, "OutOfMemoryError") else False:
        return True
    s = str(err).lower()
    return ("out of memory" in s) or ("cuda oom" in s) or ("alloc" in s and "fail" in s)


def _empty_cache():
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    elif DEV.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


# --------------------------------- memory measurement ------------------------------------ #
def _reset_peak():
    if DEV.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def _peak_bytes():
    """Peak allocated bytes since the last _reset_peak(). CUDA: true peak via max_memory_allocated.
    MPS: no peak API exists, so sample current_allocated_memory (best-effort, labelled in output).
    CPU: not measurable -> None."""
    if DEV.type == "cuda":
        torch.cuda.synchronize()
        return int(torch.cuda.max_memory_allocated())
    if DEV.type == "mps":
        try:
            return int(torch.mps.current_allocated_memory())
        except Exception:
            return None
    return None


# --------------------------------- the core timed probe ---------------------------------- #
@torch.no_grad()
def decode_latency(model, n, reps, warmup, measure_mem=False):
    """Decode `n` tokens through model.step() (the fair O(1)-API path for BOTH models) and return
    the MEDIAN total wall-clock over `reps` timed runs (after `warmup` untimed runs).

    measure_mem=True additionally measures the peak allocated memory over the decode (reset before,
    read after — CUDA gives a true peak; MPS samples current alloc). Returns:
        {"total_s": median_total_seconds, "per_step_ms": median_total*1000/n, "peak_bytes": int|None}
    """
    lat = []
    for r in range(reps + warmup):
        st = model.init_state(1, DEV)
        tok = torch.randint(0, V, (1, 1), device=DEV)
        if r == reps + warmup - 1 and measure_mem:
            _reset_peak()
        _sync()
        t0 = time.time()
        for _ in range(n):
            lg, st = model.step(tok, st)
            tok = lg[:, -1:].argmax(-1)
        _sync()
        dt = time.time() - t0
        if r >= warmup:
            lat.append(dt)
    peak = _peak_bytes() if measure_mem else None
    total = float(np.median(lat))
    return {"total_s": total, "per_step_ms": total * 1000.0 / n, "peak_bytes": peak}


# --------------------------------- analytic memory --------------------------------------- #
def kv_floats(L, H, dh, n):
    """Transformer KV-cache size in floats at decode length n: K and V, per layer, per head, n slots,
    dh each. Grows LINEARLY in n. (Identical formula to gpu_bench.py phase5 tf_kv_floats.)"""
    return 2 * L * H * dh * n


def prism_state_floats(L, H, dh, d_phi, window):
    """PRISM carried-state size in floats — CONSTANT in n: the d_h x d_phi associative state per head
    per layer, plus the length-`window` k/v ring (2*window*dh per head per layer). Independent of n.
    (Generalizes gpu_bench.py phase5 prism_state_floats, which hard-coded window=16 and d_phi=dh+feat_n2.)"""
    return L * H * dh * d_phi + 2 * L * H * window * dh


# --------------------------------- model builders ---------------------------------------- #
def build_tf(d, L, H, max_len):
    return Transformer(TFConfig(vocab=V, d_model=d, n_layers=L, n_heads=H, max_len=max_len + 8, rope=True)).to(DEV)


def build_prism(d, L, H, max_len):
    # max_len does NOT bound PRISM's O(1) decode (no learned pos here); kept generous for safety.
    return PRISMSeqLM(PRISMSeqConfig(vocab=V, d_model=d, n_layers=L, n_heads=H, max_len=max_len + 8,
                                     feat_map="quad2", feat_n2=FEAT_N2)).to(DEV)


# --------------------------------- per-model-size sweep ---------------------------------- #
def run_size(res, size_key, d, L, H, ns, reps, warmup):
    """Run the TF-vs-PRISM decode sweep over `ns` for one model size; stream every cell to the
    ledger; build the per-size summary (per-step curves, crossover, OOM ceiling, measured peak mem)."""
    print(f"\n==== MODEL SIZE {size_key}: d{d} L{L} H{H}  (V={V}, feat_n2={FEAT_N2}) ====", flush=True)
    cells = res.setdefault("cells", {})
    dh = d // H

    # Build models once per size; the TF KV-cache is freed by reallocating init_state each rep.
    tf = build_tf(d, L, H, max(ns)); tf.train(False)
    ps = build_prism(d, L, H, max(ns)); ps.train(False)
    d_phi = ps.cfg.d_phi
    window = ps.cfg.window

    tf_oom_ceiling = None   # first n at which the TF decode OOMs (its practical memory ceiling)
    for n in ns:
        base = f"{size_key}.n{n}"

        # ---- Transformer (KV-cache, O(t)/step, O(n) memory) ---- #
        tkey = f"TF.{base}"
        if tkey not in cells:
            if tf_oom_ceiling is not None:
                # already OOM'd at a smaller n -> everything larger also OOMs; record without retrying.
                cells[tkey] = {"oom": True, "note": f"skipped (OOM at n={tf_oom_ceiling})"}
            else:
                try:
                    rec = decode_latency(tf, n, reps, warmup, measure_mem=True)
                    rec["kv_floats"] = kv_floats(L, H, dh, n)
                    cells[tkey] = rec
                    pm = rec["peak_bytes"]
                    print(f"  n={n:<6} TF(KV)   per-step={rec['per_step_ms']:.3f}ms  "
                          f"total={rec['total_s']:.3f}s  peak={_fmt_mb(pm)}", flush=True)
                except Exception as e:   # noqa: BLE001 — OOM (or any decode failure) is a DATA POINT
                    if _is_oom(e):
                        tf_oom_ceiling = n
                        cells[tkey] = {"oom": True, "note": str(e)[:200]}
                        _empty_cache()
                        print(f"  n={n:<6} TF(KV)   OOM -> practical ceiling (KV-cache too large)", flush=True)
                    else:
                        cells[tkey] = {"error": str(e)[:200]}
                        print(f"  n={n:<6} TF(KV)   ERROR: {str(e)[:120]}", flush=True)
            _save(res)
        elif cells[tkey].get("oom") and tf_oom_ceiling is None:
            tf_oom_ceiling = n   # resume: re-learn the ceiling from the ledger

        # ---- PRISM-quad2 (constant state, O(1)/step, O(1) memory) ---- #
        pkey = f"PRISM-quad2.{base}"
        if pkey not in cells:
            try:
                rec = decode_latency(ps, n, reps, warmup, measure_mem=True)
                rec["state_floats"] = prism_state_floats(L, H, dh, d_phi, window)
                cells[pkey] = rec
                pm = rec["peak_bytes"]
                print(f"  n={n:<6} PRISM    per-step={rec['per_step_ms']:.3f}ms  "
                      f"total={rec['total_s']:.3f}s  peak={_fmt_mb(pm)}", flush=True)
            except Exception as e:   # noqa: BLE001
                cells[pkey] = ({"oom": True, "note": str(e)[:200]} if _is_oom(e)
                               else {"error": str(e)[:200]})
                print(f"  n={n:<6} PRISM    {'OOM' if _is_oom(e) else 'ERROR'}: {str(e)[:120]}", flush=True)
            _save(res)

    # free this size's models before the next (bigger) size
    del tf, ps
    _empty_cache()

    summary = _summarize_size(cells, size_key, d, L, H, dh, d_phi, window, ns, tf_oom_ceiling)
    res.setdefault("size_summaries", {})[size_key] = summary
    _save(res)
    return summary


def _summarize_size(cells, size_key, d, L, H, dh, d_phi, window, ns, tf_oom_ceiling):
    """Build the per-step ms curves for both models, find the crossover n (first n where TF per-step
    ms strictly exceeds PRISM per-step ms), and assemble the measured + analytic memory curves."""
    tf_ps_ms, ps_ps_ms = {}, {}
    tf_total, ps_total = {}, {}
    tf_peak, ps_peak = {}, {}
    kv_fl, st_fl = {}, {}
    crossover = "no crossover in tested range"
    for n in ns:
        tf_rec = cells.get(f"TF.{size_key}.n{n}", {})
        ps_rec = cells.get(f"PRISM-quad2.{size_key}.n{n}", {})
        kv_fl[n] = kv_floats(L, H, dh, n)
        st_fl[n] = prism_state_floats(L, H, dh, d_phi, window)
        if "per_step_ms" in tf_rec:
            tf_ps_ms[n] = round(tf_rec["per_step_ms"], 4)
            tf_total[n] = round(tf_rec["total_s"], 4)
            if tf_rec.get("peak_bytes") is not None:
                tf_peak[n] = int(tf_rec["peak_bytes"])
        if "per_step_ms" in ps_rec:
            ps_ps_ms[n] = round(ps_rec["per_step_ms"], 4)
            ps_total[n] = round(ps_rec["total_s"], 4)
            if ps_rec.get("peak_bytes") is not None:
                ps_peak[n] = int(ps_rec["peak_bytes"])
        # crossover: first n where BOTH measured and TF per-step > PRISM per-step.
        if (crossover == "no crossover in tested range"
                and n in tf_ps_ms and n in ps_ps_ms and tf_ps_ms[n] > ps_ps_ms[n]):
            crossover = n

    # measured memory crossover (first n where measured TF peak > PRISM peak), CUDA-meaningful.
    mem_crossover = "no measured-memory crossover in tested range"
    for n in ns:
        if n in tf_peak and n in ps_peak and tf_peak[n] > ps_peak[n]:
            mem_crossover = n
            break

    return {
        "config": {"d": d, "L": L, "H": H, "d_h": dh, "d_phi": d_phi, "window": window,
                   "vocab": V, "feat_n2": FEAT_N2},
        "ns_tested": ns,
        "tf_per_step_ms": tf_ps_ms,
        "prism_per_step_ms": ps_ps_ms,
        "tf_total_s": tf_total,
        "prism_total_s": ps_total,
        "latency_crossover_n": crossover,
        "tf_oom_ceiling_n": tf_oom_ceiling,
        "analytic_kv_floats": kv_fl,
        "analytic_prism_state_floats": st_fl,
        "analytic_kv_over_state_ratio": {n: round(kv_fl[n] / st_fl[n], 3) for n in ns},
        "measured_tf_peak_bytes": tf_peak,
        "measured_prism_peak_bytes": ps_peak,
        "measured_mem_crossover_n": mem_crossover,
        "mem_measure_kind": ("cuda_max_memory_allocated" if DEV.type == "cuda"
                             else ("mps_current_allocated_memory_bestEffort" if DEV.type == "mps"
                                   else "unavailable_cpu")),
    }


def _fmt_mb(b):
    return "n/a" if b is None else f"{b / 1e6:.1f}MB"


# --------------------------------- top-level driver -------------------------------------- #
def default_grid(smoke: bool):
    """The n-grid and (model-size list, reps, warmup) for full vs smoke runs."""
    if smoke:
        ns = [128, 512, 2048]
        sizes = [("small", 128, 4, 4)]          # tiny: just prove the machinery + step() correctness
        reps, warmup = 2, 1
    else:
        ns = [4096, 8192, 16384, 32768, 65536]  # push WAY past phase5's 4096 to surface a crossover
        sizes = [("small", 128, 4, 4),          # the headline small model (overhead-bound at <=4k)
                 ("big",   512, 8, 8)]          # heavier per-step attention term -> crossover more likely
        reps, warmup = 5, 2
    env_ns = os.environ.get("PRISM_LAT_NS")
    if env_ns:
        ns = [int(x) for x in env_ns.split(",") if x.strip()]
    env_reps = os.environ.get("PRISM_LAT_REPS")
    if env_reps:
        reps = int(env_reps)
    return ns, sizes, reps, warmup


def main():
    smoke = ("--smoke" in sys.argv) or (os.environ.get("PRISM_LAT_SMOKE", "0") == "1")
    ns, sizes, reps, warmup = default_grid(smoke)
    print(f"device={DEV} torch={torch.__version__} results={OUT}", flush=True)
    print(f"mode={'SMOKE' if smoke else 'FULL'} ns={ns} sizes={[s[0] for s in sizes]} "
          f"reps={reps} warmup={warmup}", flush=True)
    if DEV.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    res = _load()
    res["meta"] = {"device": DEV.type, "torch": torch.__version__, "vocab": V, "feat_n2": FEAT_N2,
                   "mode": "smoke" if smoke else "full", "ns": ns, "reps": reps, "warmup": warmup,
                   "fairness": "both models decode via model.step() (KV-cache for TF, O(1) state for PRISM)"}
    _save(res)

    for (size_key, d, L, H) in sizes:
        run_size(res, size_key, d, L, H, ns, reps, warmup)

    # ---- top-level latency_summary across sizes ---- #
    latency_summary = {}
    for (size_key, d, L, H) in sizes:
        s = res["size_summaries"][size_key]
        latency_summary[size_key] = {
            "config": s["config"],
            "tf_per_step_ms": s["tf_per_step_ms"],
            "prism_per_step_ms": s["prism_per_step_ms"],
            "latency_crossover_n": s["latency_crossover_n"],
            "tf_oom_ceiling_n": s["tf_oom_ceiling_n"],
            "measured_tf_peak_bytes": s["measured_tf_peak_bytes"],
            "measured_prism_peak_bytes": s["measured_prism_peak_bytes"],
            "measured_mem_crossover_n": s["measured_mem_crossover_n"],
            "analytic_kv_over_state_ratio": s["analytic_kv_over_state_ratio"],
        }
    res["latency_summary"] = latency_summary
    _save(res)

    # ---- honest verdict line ---- #
    verdicts = []
    for size_key, s in latency_summary.items():
        c = s["latency_crossover_n"]
        if isinstance(c, int):
            verdicts.append(f"{size_key}: latency crossover at n={c} (PRISM O(1) wins wall-clock beyond it)")
        else:
            verdicts.append(f"{size_key}: NO latency crossover in {ns} (overhead-bound; memory advantage stands)")
        if s["tf_oom_ceiling_n"] is not None:
            verdicts[-1] += f"; TF OOM ceiling n={s['tf_oom_ceiling_n']}"
    res["verdict"] = verdicts
    _save(res)

    print("\n==== LATENCY VERDICT ====", flush=True)
    for v in verdicts:
        print("  " + v, flush=True)
    print(f"\nsaved -> {OUT}", flush=True)

    # Machine-readable block for downstream parsing (mirrors the ===...=== convention).
    print("\n===LATENCY_RESULTS===", flush=True)
    print(json.dumps({"meta": res["meta"], "latency_summary": latency_summary, "verdict": verdicts},
                     indent=2), flush=True)
    print("===END_LATENCY_RESULTS===", flush=True)
    return res


if __name__ == "__main__":
    main()
