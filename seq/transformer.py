"""
A clean, properly-implemented decoder-only Transformer baseline (NOT a strawman).
Causal multi-head self-attention via scaled_dot_product_attention, RMSNorm pre-norm,
SwiGLU MLP, learned absolute positions. This is the honest reference PRISM-Seq must match
at equal parameter count.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TFConfig:
    vocab: int = 64
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = None          # default 8/3 * d_model rounded (SwiGLU keeps ~4x FLOPs)
    max_len: int = 1024
    dropout: float = 0.0
    tie_embeddings: bool = True
    rope: bool = True         # use RoPE (parameter-free) instead of a learned pos-embedding

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = int(round(8 / 3 * self.d_model / 8) * 8)


def _tf_rope_cache(T, hd, device, dtype):
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=device, dtype=torch.float32) / hd))
    ang = torch.outer(torch.arange(T, device=device, dtype=torch.float32), inv)
    return torch.cos(ang).to(dtype), torch.sin(ang).to(dtype)


def _tf_apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return n * self.w


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TFConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.nh = cfg.n_heads
        self.hd = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = cfg.dropout
        self.rope = cfg.rope

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.nh, self.hd).transpose(1, 2)
        k = k.view(B, T, self.nh, self.hd).transpose(1, 2)
        v = v.view(B, T, self.nh, self.hd).transpose(1, 2)
        if self.rope:
            cos, sin = _tf_rope_cache(T, self.hd, x.device, x.dtype)
            q = _tf_apply_rope(q, cos, sin)
            k = _tf_apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.drop if self.training else 0.0)
        o = o.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(o)

    @torch.no_grad()
    def step(self, x_t, cache):
        """x_t:[B,1,C]; cache=(K,V) each [B,nh,t,hd] or None. KV-cached O(t) decode."""
        B, _, C = x_t.shape
        q, k, v = self.qkv(x_t).split(C, dim=2)
        q = q.view(B, 1, self.nh, self.hd).transpose(1, 2)
        k = k.view(B, 1, self.nh, self.hd).transpose(1, 2)
        v = v.view(B, 1, self.nh, self.hd).transpose(1, 2)
        t = 0 if cache is None else cache[0].shape[2]
        if self.rope:
            cos, sin = _tf_rope_cache(t + 1, self.hd, x_t.device, x_t.dtype)
            q = _tf_apply_rope(q, cos[t:t + 1], sin[t:t + 1])
            k = _tf_apply_rope(k, cos[t:t + 1], sin[t:t + 1])
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=False)   # q attends all cached (<=t)
        o = o.transpose(1, 2).reshape(B, 1, C)
        return self.proj(o), (k, v)


class SwiGLU(nn.Module):
    def __init__(self, cfg: TFConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.wo = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.wo(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self, cfg: TFConfig):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x

    @torch.no_grad()
    def step(self, x_t, cache):
        o, cache = self.attn.step(self.n1(x_t), cache)
        x_t = x_t + o
        return x_t + self.mlp(self.n2(x_t)), cache


class Transformer(nn.Module):
    def __init__(self, cfg: TFConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos = None if cfg.rope else nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.nf = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.tok.weight
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx)
        if self.pos is not None:
            x = x + self.pos(torch.arange(T, device=idx.device))[None]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.nf(x))

    @torch.no_grad()
    def init_state(self, batch, device):
        return [None for _ in self.blocks]              # per-layer KV cache (grows with t)

    @torch.no_grad()
    def step(self, tok, state):
        """tok:[B,1] -> (logits[B,1,V], new_state). KV-cached: O(t) compute, O(t) memory at step t."""
        x = self.tok(tok)
        if self.pos is not None:
            p = 0 if state[0] is None else state[0][0].shape[2]
            x = x + self.pos(torch.tensor([p], device=tok.device))[None]
        new = []
        for blk, c in zip(self.blocks, state):
            x, c2 = blk.step(x, c)
            new.append(c2)
        return self.head(self.nf(x)), new


if __name__ == "__main__":
    from common import param_count
    cfg = TFConfig(vocab=64, d_model=128, n_layers=2, n_heads=4)
    m = Transformer(cfg)
    x = torch.randint(0, 64, (2, 32))
    print("logits", tuple(m(x).shape), "params", param_count(m))
