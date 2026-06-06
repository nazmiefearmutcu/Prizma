"""
Family-control sequence baselines for B6 (is PRISM-Seq more than 'a bigger RNN'?).
  * GRULM        — a plain gated RNN (no associative state, no attention).
  * LinAttnLM    — a minimal (non-delta) linear-attention block: additive S = sum phi(k) v^T,
                   read phi(q) S. This is the 'no targeted erase' memory family.
Both share the embedding/FFN/head scaffold so the comparison stays param-fair-ish.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import RMSNorm, SwiGLU, TFConfig


class GRULM(nn.Module):
    def __init__(self, vocab, d_model=64, n_layers=2, **_):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.rnn = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.nf = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, idx):
        h, _ = self.rnn(self.tok(idx))
        return self.head(self.nf(h))


class _LinAttnBlock(nn.Module):
    def __init__(self, cfg: TFConfig):
        super().__init__()
        d, H = cfg.d_model, cfg.n_heads
        self.H, self.dh = H, d // H
        self.norm1 = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.norm2 = RMSNorm(d)
        self.mlp = SwiGLU(cfg)

    def forward(self, h):
        B, T, d = h.shape
        x = self.norm1(h)
        q, k, v = self.qkv(x).view(B, T, self.H, 3, self.dh).unbind(3)
        q = F.elu(q.transpose(1, 2)) + 1                       # [B,H,T,dh] positive feature map
        k = F.elu(k.transpose(1, 2)) + 1
        v = v.transpose(1, 2)
        kv = torch.einsum("bhtd,bhte->bhtde", k, v)            # outer products
        kv = kv.cumsum(dim=2)                                  # additive causal state (no erase)
        num = torch.einsum("bhtd,bhtde->bhte", q, kv)
        ksum = k.cumsum(dim=2)
        den = torch.einsum("bhtd,bhtd->bht", q, ksum).clamp_min(1e-6)[..., None]
        o = (num / den).transpose(1, 2).reshape(B, T, d)
        h = h + self.o(o)
        return h + self.mlp(self.norm2(h))


class LinAttnLM(nn.Module):
    def __init__(self, vocab, d_model=64, n_layers=2, n_heads=2, **_):
        super().__init__()
        cfg = TFConfig(vocab=vocab, d_model=d_model, n_layers=n_layers, n_heads=n_heads)
        self.tok = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList([_LinAttnBlock(cfg) for _ in range(n_layers)])
        self.nf = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, idx):
        h = self.tok(idx)
        for b in self.blocks:
            h = b(h)
        return self.head(self.nf(h))
