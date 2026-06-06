"""PRISM-quad2 vs a tuned Transformer char-LM — a CREDIBLE, REGULARIZED, configurable rebuild.

This is a hardened successor to `gpu_charlm.py` (the §4 "small LM" bar leg). It param-matches
PRISM-Seq-quad2 against an honest decoder-only Transformer (RMSNorm + SwiGLU + RoPE, the one in
`seq/transformer.py`) at a small scale and asks the falsifiable, pre-registered question:

    PASS iff  PRISM-quad2 best_bpc <= TF best_bpc + 0.05   (param-matched, >= 2 seeds)

where the headline metric is **test bits-per-char (BPC)** = mean held-out next-char CE (nats)/ln2.

────────────────────────────────────────────────────────────────────────────────────────────────
WHY THIS FILE EXISTS (the overfitting fix)
────────────────────────────────────────────────────────────────────────────────────────────────
The original `gpu_charlm.py` recipe OVERFITS catastrophically on tiny-shakespeare (~1M train chars):
a ~3.2M-param model over 15000 steps with eval only every 1000 steps drives train-loss to ~0.085
while TEST BPC rises monotonically from ~2.2 (step 1000) to ~7.1 (worse than the random baseline).
The recorded best_bpc lands at the very FIRST checkpoint, so the model is essentially untrained at
its own optimum — non-credible. This rebuild fixes that WITHOUT favoring either architecture:

  1. ANTI-OVERFITTING, applied FAIRLY to BOTH models:
       * AdamW weight_decay (CLI --weight_decay, default 0.1). Optimizer-level => IDENTICAL pressure
         on PRISM and the TF, no architecture edits, perfectly symmetric.
       * DROPOUT IS DELIBERATELY NOT USED.  Audit of the constructors:
            - seq/transformer.py  : TFConfig HAS `dropout` (attention dropout via SDPA dropout_p).
            - seq/prism_seq.py    : PRISMSeqConfig has NO dropout anywhere (neither the delta path,
                                    the window SDPA head, nor an embedding/residual dropout).
         Dropout therefore exists on ONLY ONE side. Enabling it would apply regularization pressure
         to the Transformer that PRISM cannot receive — an UNFAIR contrast that would flatter PRISM.
         Per the fairness contract we rely on weight_decay ONLY (architecture-agnostic) and record
         this choice in the JSON `_config` (`dropout_used=false`, `dropout_reason=...`). A bigger,
         more diverse corpus (text8, default) is the primary, fair overfitting control.
  2. FINER eval cadence: --eval_every default 250 (was 1000) so the true early-stop optimum is
     captured, and we record both best_bpc (min over all eval points) AND best_step.
  3. CONFIGURABLE corpus: --corpus {shakespeare,text8}, DEFAULT text8 (100M chars, V=27). text8 is
     large enough that, with weight_decay, test BPC stays bounded (verified in the smoke test).
  4. Reduced but overridable steps for text8 (--steps default 20000).

Everything else MIRRORS `gpu_charlm.py` and is held IDENTICAL across the two arms: optimizer (AdamW),
cosine-with-warmup schedule, grad-clip, step budget, batch size, context, the SHARED per-arm best-LR
grid (each architecture trains at its own best LR on the SAME grid — the standard fair cross-arch LM
protocol; disclosed), the FROZEN reproducible eval set, deterministic seeding before BOTH data
sampling and model construction, and the crash-safe streaming-JSON cache-by-key resumable ledger.

Param-match: the feature map is buffers (0 trainable params), so PRISM-quad2's count is fixed; the
TF's SwiGLU d_ff is sized so its param count lands within ~1% of PRISM-quad2 (recorded in _config as
param_match_pct). If anything that favors the TF baseline.

Crash-safe + resumable: every (arm x seed) record streams to $PRISM_RESULTS/gpu_charlm2.json via an
atomic write; completed cells are skipped on restart. SEPARATE file from gpu_charlm.json so a live
run writing that file is never clobbered.

Env:  set PRISM_RESULTS to a Drive-mounted dir for persistence (default ./results).
Run:  python3 gpu_charlm2.py                          # text8 subset (default), 2 seeds
      python3 gpu_charlm2.py --corpus shakespeare     # tiny-shakespeare
      python3 gpu_charlm2.py --smoke                   # fast 1-seed tiny-config sanity check
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import time
import urllib.request
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from seq.common import param_count, set_seed
from seq.transformer import Transformer, TFConfig
from seq.prism_seq import PRISMSeqLM, PRISMSeqConfig

# --------------------------------------------------------------------------------------------- #
# Device + crash-safe JSON ledger (mirrors gpu_charlm._load/_save; SEPARATE file gpu_charlm2.json
# so we NEVER clobber the existing gpu_charlm.json that may be written live by another run).
# --------------------------------------------------------------------------------------------- #
DEV = torch.device("cuda" if torch.cuda.is_available()
                   else ("mps" if torch.backends.mps.is_available() else "cpu"))
RES = os.environ.get("PRISM_RESULTS", os.path.join(os.path.dirname(__file__), "results"))
os.makedirs(RES, exist_ok=True)
OUT = os.path.join(RES, "gpu_charlm2.json")

_DATA_DIR = os.path.join(os.path.dirname(__file__), "seq", "data")
_SHAKES = os.path.join(_DATA_DIR, "shakespeare.txt")
_SHAKES_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# text8 raw (100M lowercase a-z + space). Cache the RAW file under $PRISM_RESULTS so a resumed run
# never re-downloads. Try the HF raw mirror first (plain text, no unzip), then the canonical zip.
# NOTE: the verified-live HF raw file is the `ardMLX/text8` repo's plain `text8` object (302->CDN,
# Content-Length=100,000,000). The `.zip` variant and the `afmck` raw path do NOT exist (404), so
# they are not used.
_TEXT8_CACHE = os.path.join(RES, "text8")                       # raw 100M-char text8, cached here
_TEXT8_RAW_URL = "https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8"  # plain text, 100MB
_TEXT8_ZIP_URLS = (
    "http://mattmahoney.net/dc/text8.zip",                     # canonical mattmahoney, http-only fallback
)


def _load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def _save(d):
    tmp = OUT + ".tmp"
    json.dump(d, open(tmp, "w"), indent=2)
    os.replace(tmp, OUT)


def _median(xs):
    return float(np.median(np.asarray(xs, float)))


def _rng_range(xs):
    """(mean, half-range) so the summary can print mean +/- range across seeds."""
    xs = np.asarray(xs, float)
    return float(xs.mean()), float((xs.max() - xs.min()) / 2.0)


# --------------------------------------------------------------------------------------------- #
# Corpus loading.
#   shakespeare: reuse seq/data/shakespeare.txt; download from karpathy URL if absent (same path as
#                gpu_charlm.py). CONTIGUOUS 90/10 train/test split (no n-gram leak across boundary).
#   text8:       download ONCE (HF raw -> zip fallback) and cache RAW to $PRISM_RESULTS/text8.
#                Use a SUBSET: first `train_chars` train / next 500k val / next 500k test. Vocab is
#                built deterministically from the data (sorted unique chars => V=27 for text8).
# --------------------------------------------------------------------------------------------- #
def _fetch_bytes(url, timeout):
    """Return the bytes at `url`. The URL must be one of this module's FIXED constants (never
    user-controlled). https is preferred; the canonical mattmahoney text8 mirror is http-only and is
    explicitly allow-listed as a last-resort fallback (the payload is a public, hash-stable corpus)."""
    allowed = (_SHAKES_URL, _TEXT8_RAW_URL) + _TEXT8_ZIP_URLS
    if url not in allowed:
        raise ValueError(f"refusing non-allowlisted URL: {url!r}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (PRISM-charlm2)"})
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


def _ensure_text8_raw():
    """Return (path_to_raw_text8, source). Caches the full 100M-char text8 to $PRISM_RESULTS/text8."""
    if os.path.exists(_TEXT8_CACHE) and os.path.getsize(_TEXT8_CACHE) > 50_000_000:
        return _TEXT8_CACHE, "cache"
    os.makedirs(os.path.dirname(_TEXT8_CACHE), exist_ok=True)
    # 1) plain-text HF mirror (no unzip needed)
    try:
        print(f"   downloading text8 (raw) from {_TEXT8_RAW_URL}", flush=True)
        raw = _fetch_bytes(_TEXT8_RAW_URL, 600)
        if len(raw) < 50_000_000:
            raise RuntimeError(f"raw text8 too small ({len(raw)} bytes) — wrong/HTML payload")
        with open(_TEXT8_CACHE, "wb") as f:
            f.write(raw)
        return _TEXT8_CACHE, "download(raw)"
    except Exception as e_raw:
        print(f"   raw text8 fetch failed ({e_raw}); trying zip mirrors...", flush=True)
    # 2) zip mirrors (unzip in Python)
    last = None
    for url in _TEXT8_ZIP_URLS:
        try:
            print(f"   downloading text8.zip from {url}", flush=True)
            blob = _fetch_bytes(url, 600)
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                raw = z.read("text8")
            if len(raw) < 50_000_000:
                raise RuntimeError(f"unzipped text8 too small ({len(raw)} bytes)")
            with open(_TEXT8_CACHE, "wb") as f:
                f.write(raw)
            return _TEXT8_CACHE, "download(zip)"
        except Exception as e_zip:
            last = e_zip
            print(f"     -> {url} failed: {e_zip}", flush=True)
    raise RuntimeError(
        f"text8 unavailable from all mirrors (last error: {last}). On Colab the HF raw URL works; "
        f"locally you can pre-place the raw 100M-char text8 file at {_TEXT8_CACHE}.")


class CharData:
    """Char-level corpus with CONTIGUOUS train/(val)/test splits. Headline metric = TEST BPC.

    Two construction modes:
      * fractional (shakespeare): 90/10 contiguous train/test from a single text.
      * explicit  (text8):        first `train_chars` train / next `val_chars` val / next `test_chars`
                                  test, taken as contiguous blocks of the (deterministically encoded)
                                  corpus. Vocab is the sorted unique chars of the FULL encoded text.

    sample()/eval_test_batch() yield (x[B,T], y[B,T]) sliding windows; y is x shifted by one. The
    eval set is frozen via a fixed eval_seed (same across arms/seeds) so best_bpc is not cherry-picked.
    """
    def __init__(self, text: str, seq_len: int, name: str, *, train_frac=0.90,
                 train_chars=None, val_chars=0, test_chars=None):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab = len(chars)
        data = np.array([self.stoi[c] for c in text], dtype=np.int64)
        if train_chars is None:
            # fractional contiguous split (shakespeare path, identical to gpu_charlm.py)
            a = int(len(data) * train_frac)
            self.train, self.val, self.test = data[:a], data[a:a], data[a:]
            split_desc = f"train={len(self.train)},test={len(self.test)}"
        else:
            # explicit contiguous subset (text8 path)
            a = min(train_chars, len(data))
            b = min(a + val_chars, len(data))
            c = min(b + (test_chars if test_chars is not None else 0), len(data))
            self.train, self.val, self.test = data[:a], data[a:b], data[b:c]
            split_desc = f"train={len(self.train)},val={len(self.val)},test={len(self.test)}"
        if len(self.test) <= seq_len + 1:
            raise ValueError(f"test split too small ({len(self.test)}) for seq_len {seq_len}")
        self.seq_len = seq_len
        self.has_val = len(self.val) > seq_len + 1   # val present => unbiased early-stop is possible
        self.name = f"{name}(V={self.vocab},T={seq_len},{split_desc})"
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

    def eval_val_batch(self, B, device, rng):
        return self._batch(self.val, B, device, rng)


def load_corpus(which: str, seq_len: int, *, text8_train_chars=10_000_000,
                text8_val_chars=500_000, text8_test_chars=500_000):
    if which == "shakespeare":
        path, src = _ensure_shakespeare()
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return CharData(text, seq_len, "shakespeare"), src
    if which == "text8":
        path, src = _ensure_text8_raw()
        need = text8_train_chars + text8_val_chars + text8_test_chars
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read(need)                # read ONLY the subset we need (text8 is one long line)
        return CharData(text, seq_len, "text8", train_chars=text8_train_chars,
                        val_chars=text8_val_chars, test_chars=text8_test_chars), src
    raise ValueError(f"unknown corpus '{which}' (use 'shakespeare' or 'text8')")


# --------------------------------------------------------------------------------------------- #
# Model factories. PRISM-quad2 is the model under test; the Transformer is the baseline. For char-LM
# PRISM needs learned_pos=True (its delta path is position-free, so absolute positions come from a
# learned embedding here, the standard char-LM parity setting). The TF gets a slightly larger d_ff to
# param-match PRISM-quad2 within ~1% (the feature map is buffers => 0 params, so PRISM's count is
# fixed and the match must be made on the TF side).
#
# DROPOUT: not wired. TFConfig supports `dropout` but PRISMSeqConfig does NOT, so passing it would
# regularize only the TF (unfair). We pass NOTHING dropout-related to either side. See module docstring.
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
# Train one (arm x seed) cell: AdamW + cosine-with-warmup + grad-clip, fixed step budget. Score on a
# FROZEN reproducible TEST set every eval_every steps; report best (min) + final TEST BPC + best_step.
# --------------------------------------------------------------------------------------------- #
def _lr_at(step, steps, lr, warmup, min_lr_frac=0.1):
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, steps - warmup)
    return lr * (min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))


@torch.no_grad()
def split_bpc(model, data: CharData, device, batches, batch_size, split="test", eval_seed=12345):
    """Bits-per-char on a held-out split = mean next-char CE (nats)/ln2, over a FROZEN set of eval
    batches (same eval_seed across arms/seeds -> reproducible, no curve-cherry-pick). `split` in
    {'test','val'}: VAL is used ONLY for unbiased early-stopping (model selection); TEST is the
    headline score, reported at the val-selected checkpoint so we never early-stop on the test set."""
    model.train(False)
    rng = np.random.default_rng(eval_seed)
    batch_fn = data.eval_val_batch if split == "val" else data.eval_test_batch
    tot, n = 0.0, 0
    for _ in range(batches):
        x, y = batch_fn(batch_size, device, rng)
        logits = model(x)
        ce = F.cross_entropy(logits.reshape(-1, data.vocab), y.reshape(-1), reduction="sum")
        tot += float(ce)
        n += y.numel()
    return (tot / n) / math.log(2)


def train_charlm(model, data: CharData, device, *, steps, batch_size, lr, warmup, grad_clip,
                 weight_decay, betas, eval_every, eval_batches, seed, log=False):
    """Train one arm; SELECT the checkpoint by VAL BPC (unbiased) and report TEST BPC at that step.
    Corpora without a val split (e.g. shakespeare) fall back to min-over-test (documented). The
    oracle min-over-test is always recorded as `min_test_bpc` for transparency."""
    model = model.to(device)
    set_seed(seed)                                 # deterministic BEFORE model-init-dependent RNG ...
    train_rng = np.random.default_rng(seed)        # ... and the training data stream (seed-det.)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    use_val = data.has_val                         # val-based selection avoids early-stop-on-test bias
    best_val, sel_step, sel_test = float("inf"), -1, float("inf")   # the val-selected checkpoint
    min_test, min_test_step = float("inf"), -1                       # diagnostic: oracle min-over-test
    final, last_loss = float("inf"), float("nan")
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
            test_b = split_bpc(model, data, device, eval_batches, batch_size, split="test")
            val_b = (split_bpc(model, data, device, eval_batches, batch_size, split="val")
                     if use_val else None)
            if use_val and val_b < best_val:       # select on VAL, remember the matching TEST
                best_val, sel_step, sel_test = val_b, step + 1, test_b
            if test_b < min_test:
                min_test, min_test_step = test_b, step + 1
            final = test_b
            hist.append((step + 1, round(last_loss, 4), round(test_b, 4),
                         (round(val_b, 4) if use_val else None)))
            if log:
                vstr = f" val {val_b:.4f}" if use_val else ""
                print(f"    step {step+1:>6}  loss {last_loss:.4f}  test_bpc {test_b:.4f}{vstr}"
                      f"  (sel test {sel_test:.4f}@{sel_step})  lr {opt.param_groups[0]['lr']:.2e}",
                      flush=True)
    if use_val:
        best_bpc, best_step, policy = sel_test, sel_step, "val"        # TEST @ best-VAL checkpoint
    else:
        best_bpc, best_step, policy = min_test, min_test_step, "test"  # no val -> min-over-test
    return {"best_bpc": best_bpc, "best_step": best_step, "final_bpc": final,
            "min_test_bpc": round(min_test, 5), "early_stop": policy, "history": hist}


# --------------------------------------------------------------------------------------------- #
# Per-arm best-LR over a small SHARED grid (fair: same grid for both architectures), then >=2 seeds
# at that LR. Everything streams to the JSON ledger and is cache-by-key resumable.
# --------------------------------------------------------------------------------------------- #
def _cell(res, key, build_model, data, hp, seed, lr, log=False):
    if key in res and "best_bpc" in res[key]:
        return res[key]
    set_seed(seed)                                 # deterministic model construction (init RNG)
    model = build_model(data.vocab)
    p = param_count(model)
    t0 = time.time()
    r = train_charlm(model, data, DEV, lr=lr, seed=seed,
                     steps=hp["steps"], batch_size=hp["batch_size"], warmup=hp["warmup"],
                     grad_clip=hp["grad_clip"], weight_decay=hp["weight_decay"], betas=hp["betas"],
                     eval_every=hp["eval_every"], eval_batches=hp["eval_batches"], log=log)
    rec = {"best_bpc": round(r["best_bpc"], 5), "best_step": r["best_step"],
           "final_bpc": round(r["final_bpc"], 5), "min_test_bpc": r["min_test_bpc"],
           "early_stop": r["early_stop"], "params": p, "lr": lr, "seed": seed,
           "steps": hp["steps"], "sec": round(time.time() - t0, 1), "history": r["history"]}
    res[key] = rec
    _save(res)
    print(f"   [{key}] best_bpc={rec['best_bpc']:.4f}@{rec['best_step']} final={rec['final_bpc']:.4f} "
          f"lr={lr:.0e} ({rec['sec']}s, {p:,}p)", flush=True)
    return rec


def run_arm(res, arm, build_model, data, hp, lr_grid, seeds, lr_seed=0, log=False):
    """Best-LR selection on seed=lr_seed over the shared grid, then full seeds at the winning LR."""
    grid = {}
    for lr in lr_grid:
        rec = _cell(res, f"{data.name}|{arm}|lrsel|lr{lr:.0e}|s{lr_seed}",
                    build_model, data, hp, lr_seed, lr, log=log)
        grid[lr] = rec["best_bpc"]
    best_lr = min(grid, key=grid.get)
    print(f"  -> {arm}: best LR={best_lr:.0e} "
          f"(grid {{{', '.join(f'{k:.0e}:{v:.3f}' for k,v in grid.items())}}})", flush=True)
    recs = []
    for s in seeds:
        # reuse the lr-selection run for the seed that already ran at best_lr (no recompute)
        if s == lr_seed:
            recs.append(res[f"{data.name}|{arm}|lrsel|lr{best_lr:.0e}|s{lr_seed}"])
        else:
            recs.append(_cell(res, f"{data.name}|{arm}|final|lr{best_lr:.0e}|s{s}",
                              build_model, data, hp, s, best_lr, log=log))
    bests = [r["best_bpc"] for r in recs]
    mean, rng = _rng_range(bests)
    return {"best_lr": best_lr,
            "test_bpc_per_seed": [round(b, 4) for b in bests],
            "best_step_per_seed": [r["best_step"] for r in recs],
            "final_bpc_per_seed": [round(r["final_bpc"], 4) for r in recs],
            "min_test_bpc_per_seed": [r.get("min_test_bpc") for r in recs],
            "early_stop": recs[0].get("early_stop"),
            "mean_bpc": round(mean, 4), "median_bpc": round(_median(bests), 4),
            "range": round(rng, 4), "params": recs[0]["params"]}


# --------------------------------------------------------------------------------------------- #
# Configs. CLI flags override these; --smoke swaps in a tiny CPU/MPS-friendly config.
# --------------------------------------------------------------------------------------------- #
def primary_hp(args):
    """The param-matched headline config built from CLI args (d256 L4 H4 ctx256 feat_n2256 default)."""
    hp = dict(steps=args.steps, batch_size=args.batch_size, warmup=args.warmup, grad_clip=args.grad_clip,
              weight_decay=args.weight_decay, betas=(0.9, 0.95),
              eval_every=args.eval_every, eval_batches=args.eval_batches)
    return args.d, args.L, args.H, args.ctx, args.feat_n2, hp, tuple(args.lr_grid), tuple(args.seeds)


def smoke_hp(args):
    """Fast tiny-config sanity check (CPU/MPS, a few hundred steps). Honors any explicit CLI overrides
    for d/L/H/feat_n2/steps/eval_every/seeds so the documented smoke command lands a fast end-to-end."""
    d = args.d if args.d != 256 else 64
    L = args.L if args.L != 4 else 2
    H = args.H if args.H != 4 else 2
    feat_n2 = args.feat_n2 if args.feat_n2 != 256 else 16
    T = args.ctx if args.ctx != 256 else 64
    steps = args.steps if args.steps != 20000 else 200
    eval_every = args.eval_every if args.eval_every != 250 else 50
    hp = dict(steps=steps, batch_size=args.batch_size if args.batch_size != 48 else 32,
              warmup=min(args.warmup, max(1, steps // 5)), grad_clip=args.grad_clip,
              weight_decay=args.weight_decay, betas=(0.9, 0.95),
              eval_every=eval_every, eval_batches=args.eval_batches if args.eval_batches != 40 else 10)
    lr_grid = tuple(args.lr_grid) if len(args.lr_grid) == 1 else (2e-3,)
    seeds = tuple(args.seeds)
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
def run(args):
    smoke = args.smoke
    corpus = args.corpus
    print(f"device={DEV} torch={torch.__version__} corpus={corpus} smoke={smoke}", flush=True)
    if DEV.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"results -> {OUT}", flush=True)

    d, L, H, T, feat_n2, hp, lr_grid, seeds = (smoke_hp(args) if smoke else primary_hp(args))
    data, src = load_corpus(corpus, T, text8_train_chars=args.text8_train_chars,
                            text8_val_chars=args.text8_val_chars, text8_test_chars=args.text8_test_chars)
    print(f"corpus: {data.name}  (source={src}, random-baseline BPC={data.rand_bpc:.3f})", flush=True)

    # param-match: size the TF d_ff to PRISM-quad2's count, then confirm + print.
    ps_target = param_count(ps_factory(d, L, H, T, feat_map="quad2", feat_n2=feat_n2)(data.vocab))
    tf_dff, tf_p = _match_tf_dff(d, L, H, T, ps_target, V_probe=data.vocab)
    arms = build_arms(d, L, H, T, feat_n2, tf_dff)
    if args.skip_none:
        arms.pop("PRISM-none", None)   # leg-4 needs only TF vs PRISM-quad2; none is the (optional) ablation
    counts = _print_param_match(arms, data.vocab, f"d{d}L{L}H{H} ctx{T} (TF d_ff={tf_dff})")
    match_pct = (counts["TF"] - counts["PRISM-quad2"]) / counts["PRISM-quad2"] * 100
    print(f"  -> TF vs PRISM-quad2 param-match: {match_pct:+.2f}% "
          f"({'OK <=1%' if abs(match_pct) <= 1.0 else 'WARN >1%'})", flush=True)

    res = _load()
    res["_config"] = {
        "runner": "gpu_charlm2.py", "corpus": corpus, "d": d, "L": L, "H": H, "ctx": T,
        "feat_n2": feat_n2, "tf_dff": tf_dff, "steps": hp["steps"], "batch_size": hp["batch_size"],
        "warmup": hp["warmup"], "grad_clip": hp["grad_clip"], "weight_decay": hp["weight_decay"],
        "betas": list(hp["betas"]), "eval_every": hp["eval_every"], "eval_batches": hp["eval_batches"],
        "lr_grid": list(lr_grid), "seeds": list(seeds), "device": DEV.type,
        "params": counts, "param_match_pct": round(match_pct, 3),
        "random_baseline_bpc": round(data.rand_bpc, 4),
        # --- model-selection honesty disclosure (no early-stopping on the test set) ---
        "early_stop_policy": ("headline best_bpc = TEST BPC at the min-VAL checkpoint (val-based model "
                              "selection; unbiased). Corpora without a val split fall back to "
                              "min-over-test; the oracle min_test_bpc is recorded either way."),
        # --- regularization honesty disclosure ---
        "dropout_used": False,
        "dropout_reason": ("TFConfig supports dropout but PRISMSeqConfig does not; enabling it would "
                           "regularize only the Transformer (unfair). Anti-overfitting is weight_decay "
                           "(optimizer-level, identical for both) + a larger corpus (text8) ONLY."),
        "weight_decay_applied_to": ["TF", "PRISM-quad2", "PRISM-none"],
        "text8_subset": ({"train_chars": args.text8_train_chars, "val_chars": args.text8_val_chars,
                          "test_chars": args.text8_test_chars} if corpus == "text8" else None),
    }
    _save(res)

    summary = {}
    for arm, fac in arms.items():
        summary[arm] = run_arm(res, arm, fac, data, hp, lr_grid, seeds, lr_seed=seeds[0], log=args.log)

    # --- honesty guard: warn if any model failed to beat the uniform baseline ---
    rand_bpc = data.rand_bpc
    for arm, s in summary.items():
        if s["median_bpc"] > rand_bpc:
            print(f"  *** WARNING: {arm} median best_bpc {s['median_bpc']:.4f} EXCEEDS random "
                  f"baseline {rand_bpc:.4f} — this arm did not learn (check recipe). ***", flush=True)

    tf_med = summary["TF"]["median_bpc"]
    ps_med = summary["PRISM-quad2"]["median_bpc"]
    margin = round(tf_med - ps_med, 4)             # TF_best - PRISM_best (positive => PRISM ahead)
    passed = bool(ps_med <= tf_med + 0.05)
    bar = {
        "rule": "PRISM-quad2 best_bpc <= TF best_bpc + 0.05 (param-matched, >=2 seeds)",
        "metric": "test_bits_per_char",
        "pass": passed,
        "TF_median_best_bpc": tf_med,
        "PRISM_quad2_median_best_bpc": ps_med,
        "margin_tf_minus_prism": margin,           # TF - PRISM (lower BPC is better => + favors PRISM)
        "threshold": 0.05,
        "n_seeds": len(seeds),
        "param_match_pct": round(match_pct, 3),
        "random_baseline_bpc": round(rand_bpc, 4),
        "dropout_used": False,
        "model_selection": "val (TEST BPC @ min-VAL checkpoint); min-over-test kept as diagnostic",
    }

    charlm_summary = {
        "corpus": corpus, "metric": "test_bits_per_char", "per_arm": summary,
        "TF_median_bpc": tf_med, "PRISM_quad2_median_bpc": ps_med,
        "TF_mean_bpc": summary["TF"]["mean_bpc"], "PRISM_quad2_mean_bpc": summary["PRISM-quad2"]["mean_bpc"],
        "margin_tf_minus_prism": margin, "threshold": 0.05, "PASS": passed,
        "param_match_pct": round(match_pct, 3), "random_baseline_bpc": round(rand_bpc, 4),
        "verdict": ("PASS: PRISM-quad2 within +0.05 of TF" if passed
                    else "FAIL: PRISM-quad2 more than +0.05 worse than TF"),
    }
    res["_bar"] = bar
    res["charlm_summary"] = charlm_summary
    _save(res)

    print("\n=== RESULTS ===", flush=True)
    print(f"corpus={corpus}  device={DEV.type}  steps={hp['steps']}  eval_every={hp['eval_every']}  "
          f"weight_decay={hp['weight_decay']}  dropout=OFF(weight_decay-only)", flush=True)
    print(f"random-baseline BPC = {rand_bpc:.4f}   param-match (TF vs PRISM-quad2) = {match_pct:+.2f}%",
          flush=True)
    for arm, s in summary.items():
        flag = "  <-- > random!" if s["median_bpc"] > rand_bpc else ""
        print(f"  {arm:<12} best_bpc/seed={s['test_bpc_per_seed']}  "
              f"median={s['median_bpc']:.4f}  mean={s['mean_bpc']:.4f} +/-{s['range']:.4f}  "
              f"(bestLR={s['best_lr']:.0e}, best_step={s['best_step_per_seed']}, "
              f"final/seed={s['final_bpc_per_seed']}, {s['params']:,}p){flag}", flush=True)
    print(f"\n  _bar rule : {bar['rule']}", flush=True)
    print(f"  TF median best_bpc    = {tf_med:.4f}", flush=True)
    print(f"  PRISM median best_bpc = {ps_med:.4f}", flush=True)
    print(f"  margin (TF - PRISM)   = {margin:+.4f}   threshold +0.05   -> "
          f"{'PASS' if passed else 'FAIL'}", flush=True)
    print("\n===CHARLM2_BAR===", flush=True)
    print(json.dumps(bar), flush=True)
    print("===CHARLM2_RESULTS===", flush=True)
    print(json.dumps(charlm_summary), flush=True)
    print(f"saved -> {OUT}", flush=True)
    return charlm_summary


def build_parser():
    p = argparse.ArgumentParser(
        description="Credible regularized PRISM-quad2 vs Transformer char-LM (test BPC, param-matched).")
    p.add_argument("--corpus", choices=["shakespeare", "text8"], default="text8",
                   help="corpus (default: text8 — large enough to control overfitting)")
    # anti-overfitting (fair, optimizer-level)
    p.add_argument("--weight_decay", type=float, default=0.1,
                   help="AdamW weight decay, applied IDENTICALLY to both models (default 0.1)")
    p.add_argument("--dropout", type=float, default=0.1,
                   help="(ACCEPTED BUT NOT USED) PRISMSeqConfig has no dropout, so wiring it would be "
                        "unfair to the TF-only side; we rely on weight_decay. Recorded in _config.")
    # eval cadence + budget
    p.add_argument("--eval_every", type=int, default=250,
                   help="evaluate test BPC every N steps (default 250; finer than the old 1000)")
    p.add_argument("--eval_batches", type=int, default=40, help="frozen eval batches (default 40)")
    p.add_argument("--steps", type=int, default=20000, help="training steps (default 20000)")
    # model / scale (mirror gpu_charlm.py defaults)
    p.add_argument("--d", type=int, default=256, help="d_model (default 256)")
    p.add_argument("--L", type=int, default=4, help="n_layers (default 4)")
    p.add_argument("--H", type=int, default=4, help="n_heads (default 4)")
    p.add_argument("--ctx", type=int, default=256, help="context length T (default 256)")
    p.add_argument("--feat_n2", type=int, default=256, help="PRISM-quad2 monomials (default 256)")
    # optimization knobs
    p.add_argument("--batch_size", type=int, default=48, help="batch size (default 48)")
    p.add_argument("--warmup", type=int, default=1000, help="LR warmup steps (default 1000)")
    p.add_argument("--grad_clip", type=float, default=1.0, help="grad-norm clip (default 1.0)")
    p.add_argument("--lr_grid", type=float, nargs="+", default=[1e-3, 2e-3, 3e-3],
                   help="shared per-arm LR grid (default 1e-3 2e-3 3e-3)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1],
                   help=">=2 seeds; LR is selected on seeds[0], then all seeds run at the winner")
    # text8 subset sizing
    p.add_argument("--text8_train_chars", type=int, default=10_000_000,
                   help="text8 train chars (default 10,000,000)")
    p.add_argument("--text8_val_chars", type=int, default=500_000, help="text8 val chars (default 500k)")
    p.add_argument("--text8_test_chars", type=int, default=500_000,
                   help="text8 test chars (default 500k)")
    # misc
    p.add_argument("--skip_none", action="store_true",
                   help="skip the PRISM-none ablation arm (leg-4 needs only TF vs PRISM-quad2; "
                        "the feat_map causal-ablation is already covered by the MQAR B6 control)")
    p.add_argument("--smoke", action="store_true", help="fast tiny-config 1-seed sanity check")
    p.add_argument("--log", action="store_true", help="verbose per-eval logging")
    return p


def main():
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
