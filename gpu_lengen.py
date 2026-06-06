"""Prizma-Seq vs Transformer — LENGTH EXTRAPOLATION benchmark, CUDA/Colab-ready (also runs on MPS/CPU).

The classic attention-vs-SSM differentiator: TRAIN at one sequence length L, then EVALUATE the SAME
frozen weights at LONGER lengths {2x, 4x, 8x}L WITHOUT retraining, and read off the extrapolation
curve (accuracy vs eval_len/train_len). The question this answers:

    Does Prizma-Seq (position-free delta state + a small LOCAL window head) generalize to
    longer-than-trained sequences better or worse than a tuned RoPE Transformer at matched params?

Why this is a clean probe for these two architectures (verified against the code):
  * Transformer (seq/transformer.py): RoPE is recomputed per-forward from the actual T
    (`_tf_rope_cache(T,...)`), NOT a fixed table indexed by max_len -> the forward accepts ANY length
    with no crash; `max_len` only sizes the (unused, rope=True) learned pos-embedding. So the TF
    *can* run at longer T; whether ACCURACY holds at unseen positions is the open question (RoPE is
    known to degrade past its trained range without interpolation tricks).
  * Prizma-Seq (seq/prizma_seq.py): the delta state S in R^{d_h x d_phi} is POSITION-FREE and constant
    in T (RoPE is OFF on delta keys by design); position only enters via the short causal conv +
    causal write-order. The local window head builds a [T,T] band mask from the actual T -> also
    accepts any length. So Prizma is structurally length-agnostic; the empirical question is whether a
    state TRAINED on length-L statistics still recalls correctly when the recall distance grows.

PRIMARY TASK = Induction (the cleanest length-gen probe): a unique bigram [A,B] sits somewhere in a
length-L prefix, the final token is the query A, predict B (mask only the final position). Train at
prefix L=64; evaluate at prefix {64,128,256,512} = up to 8x the train length. As L grows the bigram
can sit further from the query, so the recall distance grows with eval length -> a true length test.

SECOND TASK (optional, cheap) = MQAR length-gen: train D key->value pairs at one filler gap, then
evaluate the SAME D pairs at LONGER filler gaps {0, gap, 2gap, ...} so the query-to-binding distance
grows while the task (recall D values) is unchanged. Same frozen-weights, longer-sequence protocol.

PROTOCOL (apples-to-apples, no leakage):
  * Models are built with max_len = (longest eval length + 8) so nothing can crash on the long evals.
  * A SINGLE model is trained to plateau at the TRAIN length only (via seq.common.train_model).
  * That ONE frozen model is then evaluated at each eval length on a FROZEN, reproducible eval set
    built with a fixed eval_seed PER LENGTH that is SHARED across arms -> TF and Prizma see the
    byte-identical eval batches at every length. (set_seed(eval_seed+len) before drawing each set.)
  * Arms: TF (RoPE) and Prizma-quad2, parameter-matched (counts printed). 3 seeds by default.
  * Reported per arm: train-length acc + acc at each longer eval length = the extrapolation curve,
    plus `lengen_summary` with the per-arm per-length MEDIAN over seeds. Honest read: who degrades
    less as eval_len/train_len grows.

Crash-safe + resumable: every (arm x seed) cell streams to $PRIZMA_RESULTS/gpu_lengen.json and is
skipped if already present, so a Colab disconnect never loses progress. Does NOT touch gpu_bench.json.

Env: PRIZMA_RESULTS -> a Drive-mounted dir for persistence (default ./results).
Run:
    python gpu_lengen.py                # induction (primary)
    python gpu_lengen.py induction      # same, explicit
    python gpu_lengen.py mqar           # the MQAR length-gen task
    python gpu_lengen.py both           # induction then mqar
    python gpu_lengen.py induction --smoke   # tiny d64L2H2, 1 seed, fast local sanity
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch

# REUSE the gpu_bench template wholesale: device, factories, param_count, the cell cache pattern.
from gpu_bench import DEV, tf_factory, ps_factory  # noqa: F401  (DEV/factories reused verbatim)
from seq.common import TrainConfig, train_model, param_count, set_seed, masked_acc
from seq.tasks import Induction, MQAR

# Distinct results file — NEVER clobber gpu_bench.json.
RES = os.environ.get("PRIZMA_RESULTS", os.path.join(os.path.dirname(__file__), "results"))
os.makedirs(RES, exist_ok=True)
OUT = os.path.join(RES, "gpu_lengen.json")

# Same gen-warm recipe family the rest of the repo uses for diagnostics (long absolute warmup so the
# LR-fragile Transformer is treated symmetrically; cosine floor so neither model is cut mid-climb).
GENWARM = dict(lr=1e-3, warmup=2000, warmup_frac=0.0, min_lr_frac=0.1)


def _load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def _save(d):
    tmp = OUT + ".tmp"
    json.dump(d, open(tmp, "w"), indent=2)
    os.replace(tmp, OUT)


def _median(xs):
    s = sorted(xs); n = len(s)
    if n == 0:
        return float("nan")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def ci95(xs):
    xs = np.asarray(xs, float)
    if len(xs) < 2:
        return float(xs.mean()), 0.0
    return float(xs.mean()), float(1.96 * xs.std(ddof=1) / math.sqrt(len(xs)))


# --------------------------------------------------------------------------------------------- #
# Task builders. `train_fac` builds the TRAIN-length task (used by train_model, which also drives
# its own internal frozen eval at the TRAIN length). `eval_fac(L)` builds the EVAL task at an
# arbitrary length L; the SAME trained weights are scored on it. Both arms get identical eval tasks.
# --------------------------------------------------------------------------------------------- #
def induction_spec(vocab=32, train_len=64, eval_lens=(64, 128, 256, 512)):
    """Induction length-gen: eval length == the PREFIX length (the +1 query is added by the task).
    `xkey` is the x-axis (eval_len / train_len) for the extrapolation curve."""
    longest = max(eval_lens)
    return dict(
        name="induction",
        train_len=train_len,
        eval_lens=list(eval_lens),
        # longest TOTAL sequence the model must accept = longest prefix + 1 (query). max_len adds +8.
        longest_T=longest + 1,
        train_fac=lambda: Induction(vocab=vocab, seq_len=train_len),
        eval_fac=lambda L: Induction(vocab=vocab, seq_len=L),
        xkey=lambda L: round(L / train_len, 3),
        meta=dict(vocab=vocab),
    )


def mqar_spec(vocab=256, num_pairs=16, num_queries=32, train_gap=0,
              eval_gaps=(0, 64, 192, 448)):
    """MQAR length-gen: train D pairs at `train_gap`, evaluate the SAME D pairs at LONGER filler gaps
    so the binding->query distance (= sequence length) grows while the task is unchanged. The
    "length" we report is the TOTAL sequence length 2D + gap + M; the x-axis is total_len/train_len.
    train_len here is the total sequence length at train_gap."""
    train_len = 2 * num_pairs + train_gap + num_queries
    eval_lens = [2 * num_pairs + g + num_queries for g in eval_gaps]
    longest = max(eval_lens)
    return dict(
        name="mqar",
        train_len=train_len,
        eval_lens=eval_lens,
        eval_gaps=list(eval_gaps),
        longest_T=longest,
        train_fac=lambda: MQAR(vocab=vocab, num_pairs=num_pairs, num_queries=num_queries, gap=train_gap),
        # eval_fac takes the TOTAL length but we map it back to the gap that produced it.
        eval_fac=lambda L: MQAR(vocab=vocab, num_pairs=num_pairs, num_queries=num_queries,
                                gap=L - 2 * num_pairs - num_queries),
        xkey=lambda L: round(L / train_len, 3),
        meta=dict(vocab=vocab, num_pairs=num_pairs, num_queries=num_queries),
    )


# --------------------------------------------------------------------------------------------- #
@torch.no_grad()
def _eval_at_lengths(model, spec, eval_seed, eval_batches, batch_size, device):
    """Score a FROZEN model at every eval length on a FROZEN, reproducible eval set.

    KEY CORRECTNESS: for each length L we reseed with (eval_seed + L) BEFORE drawing that length's
    eval batches. The reseed depends only on (eval_seed, L) — NOT on the arm or the model — so the
    Transformer and Prizma are scored on byte-identical batches at every length (apples-to-apples).
    The trained weights are never updated here; this is pure frozen evaluation at unseen lengths.
    """
    model.train(False)
    accs = {}
    for L in spec["eval_lens"]:
        task = spec["eval_fac"](L)
        # eval_sample (fixed difficulty) if the task exposes it; else sample. Induction/MQAR -> sample.
        sample_fn = getattr(task, "eval_sample", task.sample)
        set_seed(eval_seed + L)                       # length-keyed, arm-INDEPENDENT frozen RNG
        batch = [tuple(sample_fn(batch_size, device)) for _ in range(eval_batches)]
        a = float(np.mean([masked_acc(model(x), y, m) for (x, y, m) in batch]))
        # sanity: the model forward genuinely accepted this length (would have raised otherwise).
        accs[str(L)] = round(a, 4)
    return accs


def run_arm(res, spec, arm_name, model_fac, cap, seed, eval_batches, batch_size,
            recipe=GENWARM, eval_every=2000):
    """Train ONE (arm x seed) at the train length to plateau, then evaluate the SAME frozen weights
    at every eval length. Cache the whole cell by cellkey (resumable)."""
    cellkey = f"{spec['name']}.{arm_name}.s{seed}"
    if cellkey in res and "by_len" in res[cellkey]:
        return res[cellkey]
    train_task = spec["train_fac"]()
    # Build the model with max_len = longest eval length (+8) so the LONG evals can never crash on a
    # position table / cache that is too small. For rope=True TF and default Prizma this only sizes
    # an unused learned-pos embedding, but it is the correct, defensive contract the task demands.
    model = model_fac(train_task.vocab, spec["longest_T"])
    p = param_count(model)
    cfg = TrainConfig(steps=cap, batch_size=batch_size, log=False, eval_every=eval_every,
                      eval_batches=eval_batches, **recipe)
    t0 = time.time()
    # train_model drives training + an internal FROZEN eval at the TRAIN length (its reported best_acc
    # is the train-length number). We then do our own multi-length frozen eval on the returned model.
    r = train_model(model, train_task, cfg, DEV, seed=seed)
    by_len = _eval_at_lengths(model, spec, cfg.eval_seed, eval_batches, batch_size, DEV)
    train_len = spec["train_len"]
    rec = {
        "by_len": by_len,                                  # {eval_len(str): acc}
        "xs": {str(L): spec["xkey"](L) for L in spec["eval_lens"]},   # {eval_len: eval/train ratio}
        "train_len": train_len,
        "train_acc": by_len.get(str(train_len), r.best_acc),  # acc AT the train length (curve anchor)
        "train_best": round(r.best_acc, 4),                # train-length best from train_model's eval
        "plateau": r.steps_to_plateau,
        "params": p,
        "sec": round(time.time() - t0, 1),
        "seed": seed,
        "cap": cap,
    }
    res[cellkey] = rec
    _save(res)
    curve = "  ".join(f"L{L}={by_len[str(L)]:.3f}" for L in spec["eval_lens"])
    print(f"   [{cellkey}] params={p}  plateau@{rec['plateau']}  {curve}  ({rec['sec']}s)", flush=True)
    return rec


def _arms(d, L, H, feat_n2):
    """Param-matched arms: RoPE Transformer vs Prizma-quad2 (quadratic feature map = 0 extra params)."""
    return {
        "TF": tf_factory(d, L, H),
        "Prizma-quad2": ps_factory(d, L, H, feat_map="quad2", feat_n2=feat_n2),
    }


def summarize(res, spec, arms, seeds):
    """Per-arm, per-length MEDIAN (and mean+CI) over seeds -> the extrapolation curve summary."""
    out = {"task": spec["name"], "train_len": spec["train_len"],
           "eval_lens": spec["eval_lens"],
           "xs": {str(L): spec["xkey"](L) for L in spec["eval_lens"]},
           "arms": {}}
    for arm in arms:
        recs = [res[f"{spec['name']}.{arm}.s{s}"] for s in seeds
                if f"{spec['name']}.{arm}.s{s}" in res]
        per_len = {}
        for L in spec["eval_lens"]:
            vals = [r["by_len"][str(L)] for r in recs if str(L) in r["by_len"]]
            mean, ci = ci95(vals) if vals else (float("nan"), 0.0)
            per_len[str(L)] = {"median": round(_median(vals), 4) if vals else None,
                               "mean_ci95": [round(mean, 4), round(ci, 4)] if vals else None,
                               "vals": [round(v, 3) for v in vals]}
        # retention = median acc at the LONGEST eval length / median acc at the train length.
        tl = str(spec["train_len"])
        ll = str(max(spec["eval_lens"]))
        mt = per_len.get(tl, {}).get("median")
        ml = per_len.get(ll, {}).get("median")
        retention = round(ml / mt, 3) if (mt and ml is not None and mt > 1e-9) else None
        out["arms"][arm] = {"per_len": per_len,
                            "params": recs[0]["params"] if recs else None,
                            "retention_longest_over_train": retention}
    return out


def run_task(res, spec, scale, feat_n2, seeds, cap, eval_batches, batch_size):
    d, L, H = scale
    print(f"\n==== LENGTH-EXTRAPOLATION: task={spec['name']}  scale d{d}L{L}H{H}  "
          f"train_len={spec['train_len']}  eval_lens={spec['eval_lens']}  seeds={list(seeds)} ====",
          flush=True)
    print(f"     x-axis (eval_len/train_len): "
          f"{ {L_: spec['xkey'](L_) for L_ in spec['eval_lens']} }", flush=True)
    arms = _arms(d, L, H, feat_n2)
    # Param-match report up front (built at the longest_T with the train vocab, exactly as trained).
    vocab = _probe_vocab(spec)
    pc = {arm: param_count(fac(vocab, spec["longest_T"])) for arm, fac in arms.items()}
    print(f"     PARAM-MATCH: " + "  ".join(f"{a}={n}" for a, n in pc.items()), flush=True)
    if len(set(pc.values())) > 1:
        lo, hi = min(pc.values()), max(pc.values())
        print(f"     (param spread {hi - lo} = {100*(hi-lo)/lo:.2f}% — feat_n2 tunes Prizma width to "
              f"match TF; quad2 monomials add 0 params)", flush=True)
    for arm, fac in arms.items():
        for s in seeds:
            run_arm(res, spec, arm, fac, cap, s, eval_batches, batch_size)
    summary = summarize(res, spec, arms, seeds)
    res[f"{spec['name']}_lengen_summary"] = summary
    _save(res)
    # human-readable curve table
    print(f"\n  ---- {spec['name']} extrapolation curve (median acc over seeds) ----", flush=True)
    header = "  arm           " + "".join(f"  L={L_:<5}(x{spec['xkey'](L_):g})" for L_ in spec["eval_lens"])
    print(header, flush=True)
    for arm in arms:
        row = f"  {arm:<13}"
        for L_ in spec["eval_lens"]:
            md = summary["arms"][arm]["per_len"][str(L_)]["median"]
            row += f"  {('%.3f' % md) if md is not None else ' n/a ':<12}"
        ret = summary["arms"][arm]["retention_longest_over_train"]
        row += f"   retention(longest/train)={ret}"
        print(row, flush=True)
    return summary


def _probe_vocab(spec):
    """Vocab the train task will report (factories need it to size the embedding for the param probe)."""
    return spec["train_fac"]().vocab


# --------------------------------------------------------------------------------------------- #
def main():
    args = [a for a in sys.argv[1:]]
    smoke = "--smoke" in args
    args = [a for a in args if not a.startswith("--")]
    which = args[0].lower() if args else "induction"
    if which not in ("induction", "mqar", "both"):
        print(f"unknown task '{which}' — use: induction | mqar | both", flush=True)
        sys.exit(2)

    print(f"device={DEV} torch={torch.__version__} results={OUT} task={which} smoke={smoke}", flush=True)
    if DEV.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    if smoke:
        # TINY + FAST local sanity (MPS): does it train, eval at 64/128/256 without crash, and
        # produce an extrapolation curve? Numbers need not converge.
        scale, feat_n2, seeds = (64, 2, 2), 24, (0,)
        cap, eval_batches, batch_size = 1500, 8, 64
        ind = induction_spec(vocab=16, train_len=64, eval_lens=(64, 128, 256))
        mq = mqar_spec(vocab=64, num_pairs=8, num_queries=16, train_gap=0, eval_gaps=(0, 64, 192))
    else:
        # Full campaign defaults (do NOT auto-run on a laptop). 3 seeds, real caps.
        scale, feat_n2, seeds = (128, 4, 4), 224, (0, 1, 2)
        cap, eval_batches, batch_size = 40000, 32, 64
        ind = induction_spec(vocab=32, train_len=64, eval_lens=(64, 128, 256, 512))
        mq = mqar_spec(vocab=256, num_pairs=16, num_queries=32, train_gap=0, eval_gaps=(0, 64, 192, 448))

    res = _load()
    tasks = {"induction": ind, "mqar": mq}
    order = ["induction", "mqar"] if which == "both" else [which]
    summaries = {}
    for t in order:
        summaries[t] = run_task(res, tasks[t], scale, feat_n2, seeds, cap, eval_batches, batch_size)

    print("\n===LENGEN_RESULTS===", flush=True)
    print(json.dumps({"device": str(DEV), "smoke": smoke, "scale": scale, "feat_n2": feat_n2,
                      "seeds": list(seeds), "summaries": summaries}, indent=2), flush=True)
    print("===END_LENGEN_RESULTS===", flush=True)
    print(f"\nsaved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
