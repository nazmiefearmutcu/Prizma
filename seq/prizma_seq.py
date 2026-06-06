"""
Prizma-Seq — a Predictive-Coding Gated-DeltaNet sequence model (committee spec PRIZMA_SEQ_SPEC.md).

Mixer = a carried associative workspace state S_t in R^{d_h x d_h} per head, updated by a
precision-gated targeted erase-and-write (the delta rule), which is exactly ONE gradient step on
Prizma's per-token free energy F_t(S)=1/2||v_t - S k_t||^2.  Read = S_{t-1} q_t (recognition-by-
reconstruction, strictly pre-write/causal) + a small exact local window head. FFN is byte-identical
to the Transformer baseline so ONLY the mixer differs.

Honest design note: the write gate beta is INPUT-dependent (sigma(W_beta x_t)), which keeps the
chunk-parallel training form valid. Surprise-proportionality is intrinsic: the delta write is
u_t = beta_t * (v_t - S_{t-1} k_t) = beta_t * epsilon_t — it writes the prediction error itself
(Prizma's dW ~ (Pi*eps) (x) r). An optional surprise-gated variant (two-pass) is exposed for B6.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import RMSNorm, SwiGLU, TFConfig
from .delta import chunked_delta


@dataclass
class PrizmaSeqConfig:
    vocab: int = 64
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 2
    chunk: int = 64
    window: int = 16
    d_ff: int = None
    max_len: int = 1024
    rope: bool = False           # delta keys/queries are POSITION-FREE (DeltaNet/Mamba standard):
                                 #   RoPE makes the recall dot-product distance-dependent and
                                 #   scrambles content-based recall. Position comes from the conv +
                                 #   causal write-order. (RoPE on state keys empirically blocks MQAR.)
    beta_cap: float = 0.99
    gated: bool = False          # data-dependent forget gate alpha (Gated-DeltaNet); off for diagnostics
    learned_pos: bool = False    # add a learned absolute pos-embedding (used for char-LM parity)
    short_conv: int = 4          # short causal depthwise conv before qkv (Mamba/Based/DeltaNet std);
                                 #   lets a token's k/v encode its predecessor -> enables recall. 0=off.
    # --- ablation knobs (B6) ---
    precision_gate: str = "input"   # 'input' (sigma(W_beta x)) | 'uniform' | 'random' (B6 controls)
    write_mode: str = "delta"       # 'delta' (targeted erase-and-write) | 'additive' (linear-attn)
    use_workspace: bool = True      # False -> no carried state (window head only)
    use_window: bool = True         # False -> no local window head (state only)
    route_readout: bool = True      # False -> read state with a FIXED (input-independent) query
                                    #          (B6 noRouteReadout: kills content-based recall)
    # --- capacity lever: parameter-free quadratic key/query feature map (committee R1, rank #1) - #
    feat_map: str = "none"          # 'none' | 'quad2'. 'quad2' expands the DELTA keys/queries from
                                    #   d_h to d_phi = d_h + feat_n2 via FIXED random quadratic
                                    #   monomials -> rectangular carried state S in R^{d_h x d_phi},
                                    #   raising associative-recall key-rank toward D=128 at ZERO
                                    #   trainable params (the indices are buffers, not Parameters)
                                    #   with O(1)/constant-in-n inference intact (state stays fixed).
    feat_n2: int = 96               # number of quadratic monomials (d_phi = d_h + feat_n2)

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = int(round(8 / 3 * self.d_model / 8) * 8)
        assert self.d_model % self.n_heads == 0
        self.d_h = self.d_model // self.n_heads
        assert self.d_h % 2 == 0, "d_h must be even for RoPE"
        assert self.feat_map in ("none", "quad2", "rand_linear")
        # d_phi = delta key/query dim after the optional feature map (= d_h when 'none').
        # 'rand_linear' = a FIXED random linear map d_h->d_phi (a CONTROL: it stays in a d_h-rank
        # subspace so it must give NO capacity gain, proving the quad2 MONOMIALS are what help).
        self.d_phi = self.d_h + (self.feat_n2 if self.feat_map in ("quad2", "rand_linear") else 0)


# ----------------------------------- RoPE ------------------------------------------------- #
def _rope_cache(T, d_h, device, dtype, offset=0):
    inv = 1.0 / (10000 ** (torch.arange(0, d_h, 2, device=device, dtype=torch.float32) / d_h))
    pos = torch.arange(offset, offset + T, device=device, dtype=torch.float32)
    ang = torch.outer(pos, inv)                       # [T, d_h/2]
    return torch.cos(ang).to(dtype), torch.sin(ang).to(dtype)


def _apply_rope(x, cos, sin):
    # x: [B,H,T,d_h]; cos/sin: [T,d_h/2]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = rx1
    out[..., 1::2] = rx2
    return out


def _l2(x, eps=1e-6):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


# --------------------------------- the block ---------------------------------------------- #
class PrizmaSeqBlock(nn.Module):
    def __init__(self, cfg: PrizmaSeqConfig):
        super().__init__()
        self.cfg = cfg
        d, H, dh = cfg.d_model, cfg.n_heads, cfg.d_h
        self.H, self.dh = H, dh
        self.norm1 = RMSNorm(d)
        self.kc = cfg.short_conv
        if self.kc > 0:
            self.conv = nn.Conv1d(d, d, self.kc, groups=d, bias=True)   # depthwise causal short conv
        self.W_qkv = nn.Linear(d, 3 * d, bias=False)
        self.W_beta = nn.Linear(d, H, bias=True)
        self.beta_logit = nn.Parameter(torch.zeros(H))      # uniform (input-independent) gate, B6
        self.q_fixed = nn.Parameter(torch.randn(H, dh) * 0.02)   # noRouteReadout fixed query, B6
        self.W_alpha = nn.Linear(d, H, bias=True) if cfg.gated else None
        self.W_o = nn.Linear(d, d, bias=False)
        self.norm2 = RMSNorm(d)
        self.mlp = SwiGLU(TFConfig(d_model=d, d_ff=cfg.d_ff))
        self.win_scale = dh ** -0.5
        self.d_phi = cfg.d_phi
        if cfg.feat_map == "quad2":
            # FIXED random quadratic monomials phi(x) = [x ; x[I]*x[J]] -> d_phi. Seeded => a fixed
            # architectural choice (NOT tuned per task; disclosed). Registered as BUFFERS, so
            # param_count is unchanged (byte-identical param-match to the Transformer preserved).
            g = torch.Generator().manual_seed(1234)
            self.register_buffer("feat_I", torch.randint(0, dh, (cfg.feat_n2,), generator=g))
            self.register_buffer("feat_J", torch.randint(0, dh, (cfg.feat_n2,), generator=g))
        elif cfg.feat_map == "rand_linear":
            # CONTROL (committee): fixed random linear map d_h -> d_phi. rank <= d_h, so it CANNOT
            # raise recall key-rank -> expected NO gain over 'none'. Buffer => param_count unchanged.
            g = torch.Generator().manual_seed(1234)
            self.register_buffer("W_rand", torch.randn(dh, self.d_phi, generator=g) * (dh ** -0.5))

    def _apply_conv(self, x):
        """Causal depthwise short conv + SiLU on the normed input (Mamba/Based-style). x:[B,T,d]."""
        if self.kc == 0:
            return x
        xc = F.pad(x.transpose(1, 2), (self.kc - 1, 0))          # left-pad -> causal
        return F.silu(self.conv(xc).transpose(1, 2))             # [B,T,d]

    def _phi(self, x):
        """Quadratic key/query feature map for the DELTA path only. x:[...,d_h] (already L2-normed)
        -> [...,d_phi]. Identity when feat_map='none'; else _l2([x ; x[I]*x[J]]) over FIXED random
        monomial indices, which escapes the d_h subspace and cuts associative-recall crosstalk
        (D=128: ~0.141 -> ~0.076, ~matching a true d=128 key set). The final _l2 preserves ||k||=1,
        the invariant the delta kernel relies on. The local-window head keeps the linear L2 keys."""
        if self.cfg.feat_map == "none":
            return x
        if self.cfg.feat_map == "rand_linear":
            return _l2(x @ self.W_rand)               # control: rank <= d_h -> no capacity gain
        two = x[..., self.feat_I] * x[..., self.feat_J]
        return _l2(torch.cat([x, two], dim=-1))

    def _encode(self, x, cos, sin):
        B, T, _ = x.shape
        qkv = self.W_qkv(x).view(B, T, self.H, 3, self.dh)
        q, k, v = qkv.unbind(3)                                   # each [B,T,H,dh]
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)   # [B,H,T,dh]
        if self.cfg.rope:
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)
        q, k = _l2(q), _l2(k)
        B, T = x.shape[0], x.shape[1]
        if not self.cfg.route_readout:    # B6: read with a FIXED query -> no content-based recall
            q = _l2(self.q_fixed[None, :, None, :].expand(B, self.H, T, self.dh).contiguous())
        if self.cfg.precision_gate == "uniform":
            beta = torch.sigmoid(self.beta_logit)[None, :, None].expand(B, self.H, T) * self.cfg.beta_cap
        elif self.cfg.precision_gate == "random":   # input-independent random write gate (B6 control)
            beta = torch.rand(B, self.H, T, device=x.device, dtype=x.dtype) * self.cfg.beta_cap
        else:
            beta = torch.sigmoid(self.W_beta(x)).transpose(1, 2) * self.cfg.beta_cap   # [B,H,T]
        if self.W_alpha is not None:
            alpha = torch.sigmoid(self.W_alpha(x)).transpose(1, 2)                 # [B,H,T]
            alpha = 0.5 + 0.5 * alpha           # keep decay in [0.5,1] (stability)
        else:
            alpha = None
        return q, k, v, beta, alpha

    def _window(self, q, k, v):
        """Exact causal attention restricted to the last `w` tokens (incl. self). Fused SDPA + band
        mask (fast on MPS); scaling 1/sqrt(d_h) matches the streaming step() path."""
        T = q.shape[2]
        w = self.cfg.window
        idx = torch.arange(T, device=q.device)
        band = (idx[None, :] <= idx[:, None]) & (idx[None, :] > idx[:, None] - w)  # [T,T] True=allow
        mask = torch.zeros(T, T, device=q.device, dtype=q.dtype).masked_fill(~band, float("-inf"))
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)            # [B,H,T,dh]

    def forward(self, h):
        B, T, d = h.shape
        x = self._apply_conv(self.norm1(h))
        cos, sin = _rope_cache(T, self.dh, h.device, h.dtype) if self.cfg.rope else (None, None)
        q, k, v, beta, alpha = self._encode(x, cos, sin)
        o = torch.zeros(B, self.H, T, self.dh, device=h.device, dtype=h.dtype)
        if self.cfg.use_workspace:
            # delta state keyed by phi(q),phi(k) (dim d_phi); values stay d_h -> state [B,H,d_h,d_phi]
            o_delta, _ = chunked_delta(self._phi(q), self._phi(k), v, beta, alpha,
                                       chunk=self.cfg.chunk, write_mode=self.cfg.write_mode)  # [B,H,T,d_h]
            o = o + o_delta
        if self.cfg.use_window:
            o = o + self._window(q, k, v)            # window head keeps the LINEAR L2 keys (dim d_h)
        o = o.transpose(1, 2).reshape(B, T, d)                                     # merge heads
        h = h + self.W_o(o)
        h = h + self.mlp(self.norm2(h))
        return h

    # ---- O(1)-per-step inference path (for B5 latency / true streaming) ---- #
    @torch.no_grad()
    def step(self, h_t, state):
        """h_t:[B,1,d]; state=(S, ring_k, ring_v, conv_ring, pos). Returns o_t, new_state. O(1)."""
        B = h_t.shape[0]
        S, rk, rv, cring, pos = state
        xin = self.norm1(h_t)                                    # [B,1,d]
        if self.kc > 0:
            buf = torch.cat([cring, xin], dim=1)                 # [B,kc,d]
            w = self.conv.weight.squeeze(1)                      # [d,kc]
            xc = (buf.transpose(1, 2) * w).sum(-1) + self.conv.bias   # [B,d]
            x = F.silu(xc)[:, None, :]                           # [B,1,d]
            cring = buf[:, 1:, :]                                # keep last kc-1
        else:
            x = xin
        cos, sin = _rope_cache(1, self.dh, h_t.device, h_t.dtype, offset=pos) if self.cfg.rope else (None, None)
        q, k, v, beta, alpha = self._encode(x, cos, sin)         # [B,H,1,dh], beta [B,H,1]
        q1, k1, v1 = q[:, :, 0], k[:, :, 0], v[:, :, 0]          # [B,H,dh] (linear L2; window keys)
        q1p, k1p = self._phi(q)[:, :, 0], self._phi(k)[:, :, 0]  # [B,H,d_phi] (delta state keys)
        b1 = beta[:, :, 0]                                       # [B,H]
        a1 = alpha[:, :, 0] if alpha is not None else torch.ones_like(b1)
        o_delta = torch.einsum("bhij,bhj->bhi", S, q1p)          # pre-write read S_{t-1} phi(q)
        Sk = torch.einsum("bhij,bhj->bhi", S, k1p)               # [B,H,d_h]
        u = b1[..., None] * (v1 - a1[..., None] * Sk)            # [B,H,d_h]
        S = a1[..., None, None] * S + torch.einsum("bhi,bhj->bhij", u, k1p)   # [B,H,d_h,d_phi]
        # window ring
        rk = torch.cat([rk, k1[:, :, None]], dim=2)[:, :, -self.cfg.window:]
        rv = torch.cat([rv, v1[:, :, None]], dim=2)[:, :, -self.cfg.window:]
        sc = torch.einsum("bhd,bhwd->bhw", q1, rk) * self.win_scale
        aw = torch.softmax(sc, dim=-1)
        o_win = torch.einsum("bhw,bhwd->bhd", aw, rv)
        o = (o_delta + o_win).reshape(B, 1, -1)
        h = h_t + self.W_o(o)
        h = h + self.mlp(self.norm2(h))
        return h, (S, rk, rv, cring, pos + 1)


class PrizmaSeqLM(nn.Module):
    def __init__(self, cfg: PrizmaSeqConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model) if cfg.learned_pos else None
        self.blocks = nn.ModuleList([PrizmaSeqBlock(cfg) for _ in range(cfg.n_layers)])
        self.nf = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
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
        h = self.tok(idx)
        if self.pos is not None:
            h = h + self.pos(torch.arange(T, device=idx.device))[None]
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.nf(h))

    @torch.no_grad()
    def init_state(self, batch, device):
        st = []
        kc1 = max(self.cfg.short_conv - 1, 0)
        for _ in self.blocks:
            S = torch.zeros(batch, self.cfg.n_heads, self.cfg.d_h, self.cfg.d_phi, device=device)
            rk = torch.zeros(batch, self.cfg.n_heads, 0, self.cfg.d_h, device=device)
            rv = torch.zeros(batch, self.cfg.n_heads, 0, self.cfg.d_h, device=device)
            cring = torch.zeros(batch, kc1, self.cfg.d_model, device=device)
            st.append((S, rk, rv, cring, 0))
        return st

    @torch.no_grad()
    def step(self, tok, state):
        """tok:[B,1] -> (logits[B,1,V], new_state). O(1) in sequence length."""
        h = self.tok(tok)
        if self.pos is not None:
            p = state[0][4] if state else 0
            h = h + self.pos(torch.tensor([p], device=tok.device))[None]
        new = []
        for blk, st in zip(self.blocks, state):
            h, st2 = blk.step(h, st)
            new.append(st2)
        return self.head(self.nf(h)), new


def prizma_seq_factory(d_model=64, n_layers=2, n_heads=2, **kw):
    def f(vocab, max_len):
        return PrizmaSeqLM(PrizmaSeqConfig(vocab=vocab, d_model=d_model, n_layers=n_layers,
                                         n_heads=n_heads, max_len=max_len + 8, **kw))
    return f


if __name__ == "__main__":
    from .common import param_count, get_device
    dev = get_device()
    # O(1) GUARD (committee guardrail): for BOTH feat_map settings the streaming step() must equal
    # the parallel forward() to <1e-4, AND param_count must be identical (feature map = 0 params).
    for feat in ("none", "quad2"):
        cfg = PrizmaSeqConfig(vocab=64, d_model=64, n_layers=2, n_heads=2, feat_map=feat)
        m = PrizmaSeqLM(cfg).to(dev)
        x = torch.randint(0, 64, (2, 48), device=dev)
        y = m(x)
        m.train(False)
        st = m.init_state(2, dev)
        outs = []
        for t in range(x.shape[1]):
            lg, st = m.step(x[:, t:t + 1], st)
            outs.append(lg)
        yo = torch.cat(outs, dim=1)
        d = (y - yo).abs().max().item()
        print(f"[feat_map={feat:<5} d_phi={cfg.d_phi}] forward {tuple(y.shape)} params {param_count(m)} "
              f"step-vs-forward max|d|={d:.2e} {'OK' if d < 1e-4 else 'MISMATCH'}")
