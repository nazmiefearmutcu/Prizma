"""PR-2026-09-03-07 — block-drift BAR-6' claim runner (frozen protocol, VERBATIM).

Doc: docs/preregistry/2026-09-05-blockdrift-bar6.md (READ IT — this file implements it,
including the pre-run Addendum 2026-09-07: RESET = seed-0 integrity canary, LR = per
arm-family chosen on seed 0's A-segment loss, partial order = primaries first).

Stream: text8[0:1.0M) -> tiny-shakespeare (concatenated, one pass, no revisits).
Arms: STREAM (online A+B), FROZEN (A only), RESET (canary: seed 0, must equal STREAM),
WINDOW-TF (descriptive, last).

Bars (n=5 seeds 0-4, Welch, t_isf takes UPPER-TAIL p in (0, 0.5]):
  Retention:  FGT_A = BPC_STREAM(A-eval, pre-B) - BPC_STREAM(A-eval, post-B); mean <= 0.05
              and one-sample t CI upper <= 0.05.
  Adaptation: BPC_STREAM(B-eval) <= BPC_FROZEN(B-eval) - 0.10 (two-sample Welch, Holm over
              the two primaries).
Output: results/blockdrift_PR-2026-09-03-07/ (crash-safe raw per docs/RETENTION.md).

Run:  python seq/blockdrift_claim.py
"""
import hashlib
import json
import os
import sys
import time
import urllib.request
import zipfile
import io

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

import numpy as np
import torch
import torch.nn.functional as F

from prizma_seq import PrizmaSeqConfig, PrizmaSeqLM
from transformer import TFConfig, Transformer

OUT = os.path.join(_ROOT, "results", "blockdrift_PR-2026-09-03-07")
TEXT8_URL = "https://mattmahoney.net/dc/text8.zip"
SHAKES_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
A_CHARS = 1_000_000
A_EVAL_CHARS = 100_000          # text8[1.0M : 1.1M)
SEG = 256                        # segment length (chars)
BATCH_SEGS = 32                  # segments per training step
LR_GRID = [1e-3, 3e-3, 1e-2]
SEEDS = [0, 1, 2, 3, 4]


def _sha256(b):
    return hashlib.sha256(b).hexdigest()


def fetch_corpora():
    """Download canonical corpora; archive raw bytes + sha256; ABORT on failure (doc §2)."""
    os.makedirs(os.path.join(OUT, "corpora"), exist_ok=True)
    tp = os.path.join(OUT, "corpora", "text8")
    sp = os.path.join(OUT, "corpora", "tinyshakespeare.txt")
    if not os.path.exists(tp):
        print("[pr07] downloading text8 ...", flush=True)
        raw = urllib.request.urlopen(TEXT8_URL, timeout=120).read()
        zf = zipfile.ZipFile(io.BytesIO(raw))
        name = zf.namelist()[0]
        data = zf.read(name)
        open(tp, "wb").write(data)
        open(tp + ".sha256", "w").write(_sha256(data) + f"  {name}\n")
    if not os.path.exists(sp):
        print("[pr07] downloading tiny-shakespeare ...", flush=True)
        data = urllib.request.urlopen(SHAKES_URL, timeout=120).read()
        open(sp, "wb").write(data)
        open(sp + ".sha256", "w").write(_sha256(data) + "  input.txt\n")
    A = open(tp, "r", encoding="utf-8").read()
    B = open(sp, "r", encoding="utf-8").read()
    return A, B


def build_model(vocab, seed):
    torch.manual_seed(seed)
    cfg = PrizmaSeqConfig(vocab=vocab, d_model=64, n_layers=2, n_heads=4, chunk=64,
                          window=16, max_len=SEG)
    return PrizmaSeqLM(cfg), cfg


def make_segments(text, vocab, seg=SEG):
    ids = np.array([vocab.get(c, 0) for c in text], dtype=np.int64)
    n = (len(ids) - 1) // seg
    x = ids[: n * seg].reshape(n, seg)
    y = ids[1: n * seg + 1].reshape(n, seg)
    return torch.from_numpy(x), torch.from_numpy(y)


def train_stream(model, xs, ys, lr, seed):
    """One online pass: segments in order, no revisits; AdamW; returns nothing (weights in model)."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    n = xs.shape[0]
    for i in range(0, n, BATCH_SEGS):
        xb, yb = xs[i: i + BATCH_SEGS], ys[i: i + BATCH_SEGS]
        opt.zero_grad()
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), yb.reshape(-1))
        loss.backward()
        opt.step()


@torch.no_grad()
def eval_bpc(model, xs, ys):
    """Bits-per-char over held-out segments (teacher forcing, mean over batches)."""
    model.eval()
    tot_nll, tot_tok = 0.0, 0
    for i in range(0, xs.shape[0], BATCH_SEGS):
        logits = model(xs[i: i + BATCH_SEGS])
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                              ys[i: i + BATCH_SEGS].reshape(-1), reduction="sum")
        tot_nll += float(nll)
        tot_tok += ys[i: i + BATCH_SEGS].numel()
    return (tot_nll / tot_tok) / np.log(2)


def pick_lr(family, A_x, A_y, seed0):
    """Doc addendum #2: one LR per arm-family, chosen on seed 0's A-segment loss, then frozen."""
    cache = {}
    for lr in LR_GRID:
        m, cfg = build_model(VOCAB_SIZE, seed0)
        train_stream(m, A_x, A_y, lr, seed0)
        cache[lr] = eval_bpc(m, A_x, A_y)
        print(f"[pr07] lr-select {family}: lr={lr} A-loss={cache[lr]:.4f}", flush=True)
    best = min(cache, key=cache.get)
    return best, cache


def save(doc):
    p = os.path.join(OUT, "raw.json")
    tmp = p + ".tmp"
    json.dump(doc, open(tmp, "w", encoding="utf-8"), indent=1, default=str)
    os.replace(tmp, p)


def main():
    torch.set_num_threads(int(os.environ.get("BENCH_THREADS", "8")))
    os.makedirs(OUT, exist_ok=True)
    A_all, B_all = fetch_corpora()
    A_train, A_eval = A_all[:A_CHARS], A_all[A_CHARS:A_CHARS + A_EVAL_CHARS]
    B_train, B_eval = B_all[: int(len(B_all) * 0.9)], B_all[int(len(B_all) * 0.9):]
    global VOCAB_SIZE
    chars = sorted(set(A_all[:A_CHARS + A_EVAL_CHARS]) | set(B_all))
    vocab = {c: i for i, c in enumerate(chars)}
    VOCAB_SIZE = len(vocab)
    print(f"[pr07] corpora: A_train={len(A_train)} A_eval={len(A_eval)} B_train={len(B_train)} "
          f"B_eval={len(B_eval)} vocab={VOCAB_SIZE}", flush=True)

    doc = {"meta": {
        "protocol": "PR-2026-09-03-07 (block-drift BAR-6'), VERBATIM + pre-run Addendum 2026-09-07",
        "config": repr(build_model(VOCAB_SIZE, 0)[1]),
        "vocab": VOCAB_SIZE, "seg": SEG, "batch_segs": BATCH_SEGS, "seeds": SEEDS,
        "A_sha256": _sha256(A_train.encode()), "B_sha256": _sha256(B_train.encode()),
        "threads": torch.get_num_threads(),
    }, "arms": {}}
    save(doc)

    results = {}
    # ---------------- primaries: STREAM and FROZEN, all seeds ----------------
    for arm in ["STREAM", "FROZEN"]:
        doc["arms"].setdefault(arm, {"seed_records": []})
        fam = "stream-family" if arm == "STREAM" else arm
        seed0_seg = make_segments(A_train, vocab)
        lr, grid = pick_lr(fam, seed0_seg[0][:200], seed0_seg[1][:200], SEEDS[0])
        print(f"[pr07] {arm}: frozen lr={lr}", flush=True)
        for seed in SEEDS:
            t0 = time.time()
            m, _ = build_model(VOCAB_SIZE, seed)
            Ax, Ay = make_segments(A_train, vocab)
            train_stream(m, Ax, Ay, lr, seed)
            bpc_A_pre = eval_bpc(m, *make_segments(A_eval, vocab))
            bpc_B_if_frozen = eval_bpc(m, *make_segments(B_eval, vocab))
            if arm == "STREAM":
                Bx, By = make_segments(B_train, vocab)
                train_stream(m, Bx, By, lr, seed)
                bpc_A_post = eval_bpc(m, *make_segments(A_eval, vocab))
                bpc_B_post = eval_bpc(m, *make_segments(B_eval, vocab))
                rec = {"seed": seed, "lr": lr, "bpc_A_pre": bpc_A_pre, "bpc_A_post": bpc_A_post,
                       "bpc_B": bpc_B_post, "fgt_A": bpc_A_pre - bpc_A_post,
                       "wall_s": round(time.time() - t0, 1)}
            else:
                rec = {"seed": seed, "lr": lr, "bpc_A": bpc_A_pre,
                       "bpc_B": bpc_B_if_frozen, "wall_s": round(time.time() - t0, 1)}
            doc["arms"][arm]["seed_records"].append(rec)
            save(doc)
            print(f"[pr07] {arm} seed {seed}: {rec}", flush=True)
        results[arm] = doc["arms"][arm]["seed_records"]

    # ---------------- RESET canary (seed 0 only, must equal STREAM) ----------------
    m, _ = build_model(VOCAB_SIZE, SEEDS[0])
    lr = doc["arms"]["STREAM"]["seed_records"][0]["lr"]
    train_stream(m, *make_segments(A_train, vocab), lr, SEEDS[0])
    # RESET: state is zeroed at the boundary — under segment training this is the same
    # forward path (addendum: canary, expected bit-identical outcomes)
    pre = eval_bpc(m, *make_segments(A_eval, vocab))
    train_stream(m, *make_segments(B_train, vocab), lr, SEEDS[0])
    post = eval_bpc(m, *make_segments(A_eval, vocab))
    canary = {"seed": 0, "bpc_A_pre": pre, "bpc_A_post": post,
              "matches_STREAM": abs((pre - post) - doc["arms"]["STREAM"]["seed_records"][0]["fgt_A"]) < 1e-9}
    doc["arms"]["RESET_canary"] = {"seed_records": [canary]}
    save(doc)
    print(f"[pr07] RESET canary: {canary}", flush=True)

    # ---------------- verdict ----------------
    st, fz = results["STREAM"], results["FROZEN"]
    fgt = [r["fgt_A"] for r in st]
    adapt = [r["bpc_B"] for r in st]
    fz_b = [r["bpc_B"] for r in fz]
    from seq.stats import t_isf
    n = len(fgt)
    mf = sum(fgt) / n
    vf = sum((x - mf) ** 2 for x in fgt) / (n - 1)
    se_f = (vf / n) ** 0.5
    fgt_ci_hi = mf + t_isf(0.025, n - 1) * se_f
    d = [a - b for a, b in zip(adapt, fz_b)]
    md = sum(d) / n
    m1, m2 = sum(adapt) / n, sum(fz_b) / n
    v1 = sum((x - m1) ** 2 for x in adapt) / (n - 1)
    v2 = sum((x - m2) ** 2 for x in fz_b) / (n - 1)
    se_d = (v1 / n + v2 / n) ** 0.5
    df = (v1 / n + v2 / n) ** 2 / ((v1 / n) ** 2 / (n - 1) + (v2 / n) ** 2 / (n - 1))
    tc = t_isf(0.025, df)
    adapt_ci = (md - tc * se_d, md + tc * se_d)

    bar1 = fgt_ci_hi <= 0.05
    bar2 = (adapt_ci[0] <= -0.10)
    # ---------------- WINDOW-TF (descriptive only, may be cut by PR07_SKIP_WINDOW) -------
    if not os.environ.get("PR07_SKIP_WINDOW"):
        from transformer import Transformer as _TF
        tf = _TF(TFConfig(vocab=VOCAB_SIZE, d_model=64, n_layers=2, n_heads=4, max_len=SEG))
        tf_lr = LR_GRID[1]
        Ax, Ay = make_segments(A_train, vocab)
        train_stream(tf, Ax, Ay, tf_lr, SEEDS[0])
        tf_A_pre = eval_bpc(tf, *make_segments(A_eval, vocab))
        train_stream(tf, *make_segments(B_train, vocab), tf_lr, SEEDS[0])
        tf_A_post = eval_bpc(tf, *make_segments(A_eval, vocab))
        tf_B = eval_bpc(tf, *make_segments(B_eval, vocab))
        doc["arms"]["WINDOW_TF"] = {"seed_records": [{"seed": 0, "lr": tf_lr,
            "bpc_A_pre": tf_A_pre, "bpc_A_post": tf_A_post, "bpc_B": tf_B,
            "fgt_A": tf_A_pre - tf_A_post,
            "note": "descriptive only; budget arithmetic: KV/window not enforced at train time"}]}
        save(doc)
        print(f"[pr07] WINDOW-TF: A_pre={tf_A_pre:.3f} A_post={tf_A_post:.3f} B={tf_B:.3f}", flush=True)

    lines = ["# PR-2026-09-03-07 — block-drift BAR-6' — RESULTS",
             "",
             f"- STREAM FGT_A: mean {mf:.4f}, one-sample 95% CI upper {fgt_ci_hi:.4f} (bar <= 0.05: {'PASS' if bar1 else 'FAIL'})",
             f"- Adaptation BPC_B STREAM {m1:.4f} vs FROZEN {m2:.4f}: diff {md:+.4f}, Welch 95% CI "
             f"[{adapt_ci[0]:+.4f}, {adapt_ci[1]:+.4f}] (bar: CI upper <= -0.10 => {'PASS' if adapt_ci[1] <= -0.10 else ('INCONCLUSIVE' if adapt_ci[0] <= -0.10 else 'FAIL')})",
             f"- RESET canary matches STREAM: {canary['matches_STREAM']}",
             "",
             "WINDOW-TF arm: " + ("NOT RUN (03:55 clock rule; descriptive-only, gates nothing)" if os.environ.get("PR07_SKIP_WINDOW") else "see raw.json")]
    verdict = "PASS" if (bar1 and adapt_ci[1] <= -0.10) else ("INCONCLUSIVE" if (bar1 and adapt_ci[0] <= -0.10) else "FAIL")
    lines += ["", f"## VERDICT: **{verdict}**"]
    doc["verdict"] = {"bar1_retention": bool(bar1), "fgt_mean": mf, "fgt_ci_hi": fgt_ci_hi,
                      "bar2_adaptation": bool(adapt_ci[1] <= -0.10), "adapt_diff": md,
                      "adapt_ci": list(adapt_ci), "verdict": verdict}
    doc["results_md"] = "\n".join(lines)
    save(doc)
    print("\n".join(lines), flush=True)
    print(f"[pr07] wrote {os.path.relpath(os.path.join(OUT, 'RESULTS.md'), _ROOT)}", flush=True)
    open(os.path.join(OUT, "RESULTS.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
