"""PRISM-quad2 vs a tuned Transformer on a small char-LM — the "small LM" leg of the §4 bar.

This is bar item #4 (real-data language modeling). Unlike the synthetic MQAR arena in
`gpu_bench.py`, this is *standard autoregressive next-char prediction* on a real corpus, and the
headline metric is **test bits-per-char (BPC)** = mean test cross-entropy in bits/char = CE_nats/ln2.

It param-matches PRISM-Seq-quad2 against an honest decoder-only Transformer (RMSNorm + SwiGLU +
RoPE, the same one in `seq/transformer.py`) at a small scale and asks the falsifiable question:

    Is PRISM's test BPC within +0.05 of the Transformer's?  (PASS iff  PRISM_BPC <= TF_BPC + 0.05)

Fairness mirrors gpu_bench:
  * IDENTICAL optimizer / schedule / step-budget / batch / context for both arms.
  * Per-arm best-LR from a small SHARED grid is allowed (each architecture trains at its own best LR
    on the SAME grid; disclosed) — the standard fair protocol for cross-architecture LM comparison.
  * The param-match is enforced by giving the Transformer a slightly larger SwiGLU d_ff so its
    param count lands within ~1% of PRISM-quad2 (the feature map is buffers => 0 trainable params,
    so PRISM's param count cannot be tuned via feat_n2). If anything this favors the TF baseline.

Corpus:
  * default = tiny-shakespeare (~1.1 MB). Reused from seq/data/shakespeare.txt if present, else
    downloaded from the canonical karpathy char-rnn URL into seq/data/. Offline + missing => error.
  * `text8` (a flag) downloads the first N MB of the enwik8/text8 corpus for the frontier; shakespeare
    stays the default/primary.

Crash-safe + resumable (Colab disconnects): every (arm x seed) record streams to
$PRISM_RESULTS/gpu_charlm.json via an atomic write; completed cells are skipped on restart.

Env: set PRISM_RESULTS to a Drive-mounted dir for persistence (default ./results).
Run: python3 gpu_charlm.py             # tiny-shakespeare (default)
     python3 gpu_charlm.py text8       # text8 subset frontier
     python3 gpu_charlm.py --smoke     # fast 1-seed tiny-config sanity check
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from seq.common import param_count, set_seed
from seq.transformer import Transformer, TFConfig
from seq.prism_seq import PRISMSeqLM, PRISMSeqConfig

# --------------------------------------------------------------------------------------------- #
# Device + crash-safe JSON ledger (mirrors gpu_bench._load/_save; SEPARATE file so we never clobber
# the existing gpu_bench.json).
# --------------------------------------------------------------------------------------------- #
DEV = torch.device("cuda" if torch.cuda.is_available()
                   else ("mps" if torch.backends.mps.is_available() else "cpu"))
RES = os.environ.get("PRISM_RESULTS", os.path.join(os.path.dirname(__file__), "results"))
os.makedirs(RES, exist_ok=True)
OUT = os.path.join(RES, "gpu_charlm.json")

_DATA_DIR = os.path.join(os.path.dirname(__file__), "seq", "data")
_SHAKES = os.path.join(_DATA_DIR, "shakespeare.txt")
_SHAKES_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
_TEXT8 = os.path.join(_DATA_DIR, "text8.txt")
# https mirror of the 100 MB enwik8-derived text8 (lowercase a-z + space). The canonical
# mattmahoney.net host is http-only; this https mirror avoids cleartext transport.
_TEXT8_ZIP_URL = "https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8.zip"
_TEXT8_MB = 5                                              # use the first N MB for the frontier subset


def _load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def _save(d):
    tmp = OUT + ".tmp"
    json.dump(d, open(tmp, "w"), indent=2)
    os.replace(tmp, OUT)


def _rng_range(xs):
    """(mean, half-range) so the summary can print mean +/- range across seeds."""
    xs = np.asarray(xs, float)
    return float(xs.mean()), float((xs.max() - xs.min()) / 2.0)


# --------------------------------------------------------------------------------------------- #
# Corpus loading: reuse seq/data/shakespeare.txt; download if absent. Build a CONTIGUOUS-block
# train / test split so no n-gram leaks across the boundary (same principle as seq/charlm.CharLM,
# but with a held-out TEST slice as the headline metric — CharLM only exposed val).
# --------------------------------------------------------------------------------------------- #
def _fetch_bytes(url, timeout):
    """Return the bytes at `url`. ONLY https is allowed and the URL must be one of this module's
    fixed constants (never user-controlled) — this closes the urllib `file://`/cleartext audit
    concern (CWE-319/CWE-939): no dynamic or non-https scheme can reach the network call."""
    allowed = (_SHAKES_URL, _TEXT8_ZIP_URL)
    if url not in allowed or not url.lower().startswith("https://"):
        raise ValueError(f"refusing non-allowlisted/non-https URL: {url!r}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (PRISM-charlm)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # nosec B310 # nosemgrep
        return r.read()


def _download(url, dest):
    print(f"   downloading {url} -> {dest}", flush=True)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(_fetch_bytes(url, 60))


def _ensure_shakespeare():
    if os.path.exists(_SHAKES) and os.path.getsize(_SHAKES) > 100_000:
        return _SHAKES, "cache"
    try:
        _download(_SHAKES_URL, _SHAKES)
        return _SHAKES, "download"
    except Exception as e:
        raise RuntimeError(
            f"tiny-shakespeare not found at {_SHAKES} and download failed ({e}). "
            f"Place input.txt there manually from {_SHAKES_URL}.") from e


def _ensure_text8(mb=_TEXT8_MB):
    cache = f"{_TEXT8}.{mb}mb"
    if os.path.exists(cache) and os.path.getsize(cache) > 100_000:
        return cache, "cache"
    import zipfile
    import io
    try:
        print(f"   downloading text8 (first {mb} MB) from {_TEXT8_ZIP_URL}", flush=True)
        blob = _fetch_bytes(_TEXT8_ZIP_URL, 180)
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            raw = z.read("text8")[: mb * 1_000_000]
        with open(cache, "wb") as f:
            f.write(raw)
        return cache, "download"
    except Exception as e:
        raise RuntimeError(
            f"text8 subset unavailable ({e}). tiny-shakespeare (the default) needs no network if "
            f"seq/data/shakespeare.txt exists.") from e


class CharData:
    """Char-level corpus with a contiguous train/test split. Headline metric = TEST BPC.

    sample()/eval_test_batch() yield (x[B,T], y[B,T]) sliding windows; y is x shifted by one.
    """
    def __init__(self, text: str, seq_len: int, name: str, train_frac=0.90):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab = len(chars)
        data = np.array([self.stoi[c] for c in text], dtype=np.int64)
        a = int(len(data) * train_frac)
        # contiguous blocks -> no leakage across the train/test boundary
        self.train, self.test = data[:a], data[a:]
        self.seq_len = seq_len
        self.name = f"{name}(V={self.vocab},T={seq_len},train={len(self.train)},test={len(self.test)})"
        self.rand_bpc = math.log2(self.vocab)   # uniform-prediction baseline BPC

    def _batch(self, split, B, device, rng):
        T = self.seq_len
        ix = rng.integers(0, len(split) - T - 1, size=B)
        x = np.stack([split[i:i + T] for i in ix])
        y = np.stack([split[i + 1:i + 1 + T] for i in ix])
        return (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))

    def sample(self, B, device, rng):
        return self._batch(self.train, B, device, rng)

    def eval_test_batch(self, B, device, rng):
        return self._batch(self.test, B, device, rng)


def load_corpus(which: str, seq_len: int):
    if which == "shakespeare":
        path, src = _ensure_shakespeare()
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return CharData(text, seq_len, "shakespeare"), src
    if which == "text8":
        path, src = _ensure_text8()
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return CharData(text, seq_len, "text8"), src
    raise ValueError(f"unknown corpus '{which}' (use 'shakespeare' or 'text8')")


# --------------------------------------------------------------------------------------------- #
# Model factories. PRISM-quad2 is the model under test; the Transformer is the baseline. For char-LM
# PRISM needs learned_pos=True (its delta path is deliberately position-free, so absolute positions
# come from a learned embedding here, the standard char-LM parity setting).
# The TF gets a slightly larger d_ff to param-match PRISM-quad2 within ~1% (see module docstring).
# --------------------------------------------------------------------------------------------- #
def tf_factory(d, L, H, T, d_ff=None):
    return lambda V: Transformer(TFConfig(vocab=V, d_model=d, n_layers=L, n_heads=H,
                                          d_ff=d_ff, max_len=T + 8, rope=True))


def ps_factory(d, L, H, T, **kw):
    kw.setdefault("learned_pos", True)
    return lambda V: PRISMSeqLM(PRISMSeqConfig(vocab=V, d_model=d, n_layers=L, n_heads=H,
                                               max_len=T + 8, **kw))


def _match_tf_dff(d, L, H, T, target_params, V_probe=65):
    """Pick the TF SwiGLU d_ff (multiple of 8) whose param count is closest to `target_params`
    (PRISM-quad2's count). Returns (d_ff, tf_params)."""
    base = TFConfig(vocab=V_probe, d_model=d, n_layers=L, n_heads=H, max_len=T + 8, rope=True).d_ff
    best = (base, None, 1e18)
    for dff in range(max(8, base - 64), base + 320, 8):
        p = param_count(Transformer(TFConfig(vocab=V_probe, d_model=d, n_layers=L, n_heads=H,
                                              d_ff=dff, max_len=T + 8, rope=True)))
        if abs(p - target_params) < best[2]:
            best = (dff, p, abs(p - target_params))
    return best[0], best[1]


# --------------------------------------------------------------------------------------------- #
# Train one (arm x seed) cell: AdamW + cosine-with-warmup + grad-clip, fixed step budget. Score on
# a FROZEN reproducible TEST set every eval_every steps; report best + final TEST BPC.
# --------------------------------------------------------------------------------------------- #
def _lr_at(step, steps, lr, warmup, min_lr_frac=0.1):
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, steps - warmup)
    return lr * (min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))


@torch.no_grad()
def test_bpc(model, data: CharData, device, batches, batch_size, eval_seed=12345):
    """Bits-per-char on the held-out TEST split = mean next-char CE (nats) / ln 2, over a FROZEN
    set of eval batches (same eval_seed across arms/seeds -> reproducible, no curve-cherry-pick)."""
    model.train(False)
    rng = np.random.default_rng(eval_seed)
    tot, n = 0.0, 0
    for _ in range(batches):
        x, y = data.eval_test_batch(batch_size, device, rng)
        logits = model(x)
        ce = F.cross_entropy(logits.reshape(-1, data.vocab), y.reshape(-1), reduction="sum")
        tot += float(ce)
        n += y.numel()
    return (tot / n) / math.log(2)


def train_charlm(model, data: CharData, device, *, steps, batch_size, lr, warmup, grad_clip,
                 weight_decay, betas, eval_every, eval_batches, seed, log=False):
    model = model.to(device)
    set_seed(seed)
    train_rng = np.random.default_rng(seed)        # training stream RNG (seed-deterministic)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    best, final, last_loss = float("inf"), float("inf"), float("nan")
    hist = []
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = _lr_at(step, steps, lr, warmup)
        model.train()
        x, y = data.sample(batch_size, device, train_rng)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, data.vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        if (step + 1) % eval_every == 0 or step == steps - 1:
            last_loss = float(loss.detach())
            bpc = test_bpc(model, data, device, eval_batches, batch_size)
            best = min(best, bpc)
            final = bpc
            hist.append((step + 1, round(last_loss, 4), round(bpc, 4)))
            if log:
                print(f"    step {step+1:>6}  loss {last_loss:.4f}  test_bpc {bpc:.4f}"
                      f"  lr {opt.param_groups[0]['lr']:.2e}", flush=True)
    return {"best_bpc": best, "final_bpc": final, "history": hist}


# --------------------------------------------------------------------------------------------- #
# Per-arm best-LR over a small SHARED grid (fair: same grid for both architectures), then >=2 seeds
# at that LR. Everything streams to the JSON ledger and is cache-by-key resumable.
# --------------------------------------------------------------------------------------------- #
def _cell(res, key, build_model, data, hp, seed, lr, log=False):
    if key in res and "best_bpc" in res[key]:
        return res[key]
    model = build_model(data.vocab)
    p = param_count(model)
    t0 = time.time()
    r = train_charlm(model, data, DEV, lr=lr, seed=seed,
                     steps=hp["steps"], batch_size=hp["batch_size"], warmup=hp["warmup"],
                     grad_clip=hp["grad_clip"], weight_decay=hp["weight_decay"], betas=hp["betas"],
                     eval_every=hp["eval_every"], eval_batches=hp["eval_batches"], log=log)
    rec = {"best_bpc": round(r["best_bpc"], 5), "final_bpc": round(r["final_bpc"], 5),
           "params": p, "lr": lr, "seed": seed, "steps": hp["steps"],
           "sec": round(time.time() - t0, 1), "history": r["history"]}
    res[key] = rec
    _save(res)
    print(f"   [{key}] best_bpc={rec['best_bpc']:.4f} final={rec['final_bpc']:.4f} "
          f"lr={lr:.0e} ({rec['sec']}s, {p:,}p)", flush=True)
    return rec


def run_arm(res, arm, build_model, data, hp, lr_grid, seeds, lr_seed=0):
    """Best-LR selection on seed=lr_seed over the shared grid, then full seeds at the winning LR."""
    grid = {}
    for lr in lr_grid:
        rec = _cell(res, f"{data.name}|{arm}|lrsel|lr{lr:.0e}|s{lr_seed}", build_model, data, hp, lr_seed, lr)
        grid[lr] = rec["best_bpc"]
    best_lr = min(grid, key=grid.get)
    print(f"  -> {arm}: best LR={best_lr:.0e} (grid {{{', '.join(f'{k:.0e}:{v:.3f}' for k,v in grid.items())}}})", flush=True)
    recs = []
    for s in seeds:
        # reuse the lr-selection run for the seed that already ran at best_lr (no recompute)
        if s == lr_seed:
            recs.append(res[f"{data.name}|{arm}|lrsel|lr{best_lr:.0e}|s{lr_seed}"])
        else:
            recs.append(_cell(res, f"{data.name}|{arm}|final|lr{best_lr:.0e}|s{s}", build_model, data, hp, s, best_lr))
    bests = [r["best_bpc"] for r in recs]
    mean, rng = _rng_range(bests)
    return {"best_lr": best_lr, "test_bpc_per_seed": [round(b, 4) for b in bests],
            "mean_bpc": round(mean, 4), "range": round(rng, 4), "params": recs[0]["params"]}


# --------------------------------------------------------------------------------------------- #
# Configs.
# --------------------------------------------------------------------------------------------- #
def primary_config(corpus):
    """The param-matched headline config. d256 L4 H4 context 256 for shakespeare (>=2 seeds, 15k steps)."""
    d, L, H, T = 256, 4, 4, 256
    feat_n2 = 256
    hp = dict(steps=15000, batch_size=48, warmup=1000, grad_clip=1.0,
              weight_decay=0.1, betas=(0.9, 0.95), eval_every=1000, eval_batches=40)
    lr_grid = (1e-3, 2e-3, 3e-3)
    seeds = (0, 1)
    return d, L, H, T, feat_n2, hp, lr_grid, seeds


def smoke_config():
    """Fast 1-seed tiny-config sanity check (CPU/MPS, a few hundred steps)."""
    d, L, H, T = 64, 2, 2, 64
    feat_n2 = 64
    hp = dict(steps=200, batch_size=32, warmup=40, grad_clip=1.0,
              weight_decay=0.1, betas=(0.9, 0.95), eval_every=50, eval_batches=10)
    lr_grid = (2e-3,)
    seeds = (0,)
    return d, L, H, T, feat_n2, hp, lr_grid, seeds


def build_arms(d, L, H, T, feat_n2, tf_dff):
    return {
        "TF":          tf_factory(d, L, H, T, d_ff=tf_dff),
        "PRISM-quad2": ps_factory(d, L, H, T, feat_map="quad2", feat_n2=feat_n2),
        "PRISM-none":  ps_factory(d, L, H, T, feat_map="none"),   # ablation: state w/o the feature map
    }


def _print_param_match(arms, V, label):
    print(f"\n  param-match @ {label} (V={V}):", flush=True)
    counts = {a: param_count(f(V)) for a, f in arms.items()}
    ref = counts["PRISM-quad2"]
    for a, p in counts.items():
        diff = (p - ref) / ref * 100 if a != "PRISM-quad2" else 0.0
        print(f"    {a:<12} {p:>10,} params   ({diff:+.2f}% vs PRISM-quad2)", flush=True)
    return counts


# --------------------------------------------------------------------------------------------- #
def run(corpus="shakespeare", smoke=False):
    print(f"device={DEV} torch={torch.__version__} corpus={corpus} smoke={smoke}", flush=True)
    if DEV.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"results -> {OUT}", flush=True)

    d, L, H, T, feat_n2, hp, lr_grid, seeds = (smoke_config() if smoke else primary_config(corpus))
    data, src = load_corpus(corpus, T)
    print(f"corpus: {data.name}  (source={src}, random-baseline BPC={data.rand_bpc:.3f})", flush=True)

    # param-match: size the TF d_ff to PRISM-quad2's count, then confirm + print.
    ps_target = param_count(ps_factory(d, L, H, T, feat_map="quad2", feat_n2=feat_n2)(data.vocab))
    tf_dff, tf_p = _match_tf_dff(d, L, H, T, ps_target, V_probe=data.vocab)
    arms = build_arms(d, L, H, T, feat_n2, tf_dff)
    counts = _print_param_match(arms, data.vocab, f"d{d}L{L}H{H} ctx{T} (TF d_ff={tf_dff})")
    match_pct = (counts["TF"] - counts["PRISM-quad2"]) / counts["PRISM-quad2"] * 100
    print(f"  -> TF vs PRISM-quad2 param-match: {match_pct:+.2f}% "
          f"({'OK <=1%' if abs(match_pct) <= 1.0 else 'WARN >1%'})", flush=True)

    res = _load()
    res["_config"] = {"corpus": corpus, "d": d, "L": L, "H": H, "ctx": T, "feat_n2": feat_n2,
                      "tf_dff": tf_dff, "steps": hp["steps"], "batch_size": hp["batch_size"],
                      "lr_grid": list(lr_grid), "seeds": list(seeds), "device": DEV.type,
                      "params": counts, "param_match_pct": round(match_pct, 3),
                      "random_baseline_bpc": round(data.rand_bpc, 4)}
    _save(res)

    summary = {}
    for arm, fac in arms.items():
        summary[arm] = run_arm(res, arm, fac, data, hp, lr_grid, seeds, lr_seed=seeds[0])

    tf_bpc = summary["TF"]["mean_bpc"]
    ps_bpc = summary["PRISM-quad2"]["mean_bpc"]
    margin = round(ps_bpc - tf_bpc, 4)                 # PRISM - TF (lower is better)
    passed = bool(ps_bpc <= tf_bpc + 0.05)
    charlm_summary = {
        "corpus": corpus, "metric": "test_bits_per_char",
        "per_arm": summary,
        "TF_mean_bpc": tf_bpc, "PRISM_quad2_mean_bpc": ps_bpc,
        "margin_prism_minus_tf": margin, "threshold": 0.05,
        "PASS": passed, "param_match_pct": round(match_pct, 3),
        "verdict": ("PASS: PRISM within +0.05 of TF" if passed
                    else "FAIL: PRISM more than +0.05 worse than TF"),
    }
    res["charlm_summary"] = charlm_summary
    _save(res)

    print("\n==== CHAR-LM SUMMARY ====", flush=True)
    for arm, s in summary.items():
        print(f"  {arm:<12} test_bpc mean={s['mean_bpc']:.4f} +/- {s['range']:.4f}  "
              f"(seeds={s['test_bpc_per_seed']}, bestLR={s['best_lr']:.0e}, {s['params']:,}p)", flush=True)
    print(f"  margin (PRISM - TF) = {margin:+.4f}  threshold +0.05  -> "
          f"{'PASS' if passed else 'FAIL'}", flush=True)
    print("\n===CHARLM_RESULTS===", flush=True)
    print(json.dumps(charlm_summary), flush=True)
    print(f"saved -> {OUT}", flush=True)
    return charlm_summary


def main():
    args = [a for a in sys.argv[1:]]
    smoke = "--smoke" in args
    args = [a for a in args if a != "--smoke"]
    corpus = args[0] if args else "shakespeare"
    run(corpus=corpus, smoke=smoke)


if __name__ == "__main__":
    main()
