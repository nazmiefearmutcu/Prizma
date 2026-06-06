"""
Char-level language modeling on tiny-shakespeare (~1.1 MB) — bar item #4 (real-data LM).
Same (inputs, targets, mask) interface, but the headline metric is bits-per-char (BPC) on a
held-out split, not token accuracy. A model trained here exercises genuine next-char prediction.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch

_DATA = os.path.join(os.path.dirname(__file__), "data", "shakespeare.txt")


class CharLM:
    def __init__(self, seq_len=256, path=_DATA, splits=(0.90, 0.05, 0.05)):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab = len(chars)
        data = np.array([self.stoi[c] for c in text], dtype=np.int64)
        n = len(data)
        a = int(n * splits[0]); b = a + int(n * splits[1])
        # CONTIGUOUS blocks -> no n-gram leakage across the train/val/test boundaries
        self.train, self.val, self.test = data[:a], data[a:b], data[b:]
        self.seq_len = seq_len
        self.name = f"CharLM({os.path.basename(path)},V={self.vocab},T={seq_len})"

    def _batch(self, split, B, device):
        T = self.seq_len
        ix = np.random.randint(0, len(split) - T - 1, size=B)
        x = np.stack([split[i:i + T] for i in ix])
        y = np.stack([split[i + 1:i + 1 + T] for i in ix])
        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)
        m = torch.ones(B, T, device=device)
        return x, y, m

    def sample(self, B, device):
        return self._batch(self.train, B, device)

    def eval_sample(self, B, device):
        return self._batch(self.val, B, device)


@torch.no_grad()
def val_bpc(model, charlm: CharLM, device, batches=40, batch_size=64):
    """Bits-per-char on the validation split = mean next-char CE (nats) / ln(2)."""
    model.train(False)
    model = model.to(device)
    import torch.nn.functional as F
    tot, n = 0.0, 0
    for _ in range(batches):
        x, y, _ = charlm.eval_sample(batch_size, device)
        logits = model(x)
        ce = F.cross_entropy(logits.reshape(-1, charlm.vocab), y.reshape(-1), reduction="sum")
        tot += float(ce)
        n += y.numel()
    return (tot / n) / math.log(2)


if __name__ == "__main__":
    c = CharLM(seq_len=128)
    print(c.name, "train", len(c.train), "val", len(c.val))
    x, y, m = c.sample(4, torch.device("cpu"))
    print("x", tuple(x.shape), "sample text:", repr("".join(c.itos[int(t)] for t in x[0][:60])))
