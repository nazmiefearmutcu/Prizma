"""
Tiny-HYBRID baseline arm (Council-3; plan "Task 1.Hybrid-baseline"; council record
committee/round0_v2_synthesis.md item 9c).

WHY THIS EXISTS — an adversarial honesty check, NOT an O(1) candidate.
A Samba / GatedDeltaNet-H-style block — MOSTLY Prizma layers + ~1 attention layer — is the
strongest *cheap* baseline between the pure Transformer and pure-O(1) Prizma. It is added as a
THIRD baseline arm, param-matched, on the SAME harness. If Prizma is NOT at least
Pareto-competitive with this tiny hybrid, the honest framing of the result is "best pure-O(1)
point", not "beats the Transformer". So the hybrid keeps us honest.

ARCHITECTURE.
  embed (reuses the PrizmaSeqLM embedding/tie/head/init conventions)
    -> n_layers blocks, where a SPECIFIED subset of layer indices are Transformer attention
       Blocks (seq.transformer.Block, built from a TFConfig matching d_model/n_heads/d_ff/
       max_len/rope) and the REST are PrizmaSeqBlocks (built from the given PrizmaSeqConfig)
    -> RMSNorm final + tied head (identical to PrizmaSeqLM)
  Default: exactly 1 attention layer at the MIDDLE index (n_layers // 2). Configurable via
  attn_layers (int or tuple) or n_attn (auto-placed, centered).

  Both block types are residual (h + sublayer) at the SAME d_model and the attention Block's
  FFN (SwiGLU) is sized from the SAME d_ff as the Prizma config, so they compose in one
  nn.ModuleList and the whole model lands within ~2-3% of the matched TF baseline.

STEP / O(1) NOTE.
  This hybrid is a QUALITY / FLOP baseline, NOT an O(1) candidate: the one attention layer makes
  autoregressive decode O(n) (the KV cache grows). step()/init_state are therefore implemented by
  composing PrizmaSeqBlock.step (O(1) per layer) + Block.step (O(t) KV-cached) per layer, and the
  WHOLE model is O(n) overall by construction — there is deliberately NO O(1) guard for it (that
  would be a lie). The streaming path is provided only for latency probing / completeness.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import Block, RMSNorm, TFConfig, SwiGLU
from .prizma_seq import PrizmaSeqBlock, PrizmaSeqConfig
from .delta import chunked_delta


def _resolve_attn_layers(n_layers: int, n_attn: int, attn_layers):
    """Resolve which layer indices are attention layers.

    - attn_layers given (int or iterable) -> use exactly those indices (sorted, unique).
    - else n_attn auto-placed: centered, roughly even. n_attn=1 -> [n_layers//2].
    Returns a sorted tuple of valid indices in [0, n_layers).
    """
    if attn_layers is not None:
        if isinstance(attn_layers, int):
            idxs = (attn_layers,)
        else:
            idxs = tuple(attn_layers)
        idxs = tuple(sorted(set(int(i) for i in idxs)))
        for i in idxs:
            assert 0 <= i < n_layers, f"attn_layers index {i} out of range [0,{n_layers})"
        return idxs
    assert 0 <= n_attn <= n_layers, f"n_attn={n_attn} out of range [0,{n_layers}]"
    if n_attn == 0:
        return ()
    if n_attn == 1:
        return (n_layers // 2,)
    # n_attn >= 2: evenly spaced, centered (Samba/GDN-H style interleave)
    step = n_layers / n_attn
    idxs = sorted({min(n_layers - 1, int(round((j + 0.5) * step))) for j in range(n_attn)})
    # de-collision: if rounding collided, fill from the middle outward
    if len(idxs) < n_attn:
        pool = sorted(range(n_layers), key=lambda i: abs(i - (n_layers - 1) / 2))
        for i in pool:
            if i not in idxs:
                idxs.append(i)
            if len(idxs) == n_attn:
                break
        idxs = sorted(idxs)
    return tuple(idxs)


def resolve_hybrid_config(n_layers: int, n_heads: int, n_attn: int, attn_layers, attn_heads):
    """Resolves attention head indices for each layer.

    Returns a dict mapping layer index to a sorted tuple of attention head indices.
    """
    if attn_heads is not None:
        if isinstance(attn_heads, dict):
            layer_attn_heads = {i: tuple(sorted(set(attn_heads.get(i, ())))) for i in range(n_layers)}
        elif isinstance(attn_heads, (list, tuple)) and len(attn_heads) == n_layers and all(isinstance(x, (list, tuple, set)) for x in attn_heads):
            layer_attn_heads = {i: tuple(sorted(set(attn_heads[i]))) for i in range(n_layers)}
        else:
            # Single sequence of head indices applied to all layers
            single_heads = tuple(sorted(set(int(h) for h in attn_heads)))
            layer_attn_heads = {i: single_heads for i in range(n_layers)}

        for i, heads in layer_attn_heads.items():
            for h in heads:
                assert 0 <= h < n_heads, f"head index {h} in layer {i} out of range [0, {n_heads})"
        return layer_attn_heads

    # Fallback to standard layer-wise behavior
    resolved_layers = _resolve_attn_layers(n_layers, n_attn, attn_layers)
    layer_attn_heads = {}
    for i in range(n_layers):
        if i in resolved_layers:
            layer_attn_heads[i] = tuple(range(n_heads))
        else:
            layer_attn_heads[i] = ()
    return layer_attn_heads


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


class HybridBlock(nn.Module):
    """A hybrid layer containing a flexible mixture of attention heads and delta mixer heads."""

    def __init__(self, cfg: PrizmaSeqConfig, attn_heads: tuple[int, ...], tf_rope: bool = True):
        super().__init__()
        self.cfg = cfg
        self.tf_rope = tf_rope
        self.H = cfg.n_heads
        self.dh = cfg.d_h
        
        self.attn_heads = tuple(sorted(set(attn_heads)))
        self.delta_heads = tuple(sorted(set(h for h in range(self.H) if h not in self.attn_heads)))
        
        self.H_attn = len(self.attn_heads)
        self.H_delta = len(self.delta_heads)
        
        d = cfg.d_model
        dh = cfg.d_h
        
        self.norm1 = RMSNorm(d)
        
        # 1. Attention-specific parameters
        if self.H_attn > 0:
            self.W_qkv_attn = nn.Linear(d, 3 * self.H_attn * dh, bias=False)
            
        # 2. Delta-specific parameters
        if self.H_delta > 0:
            self.kc = cfg.short_conv
            if self.kc > 0:
                self.conv = nn.Conv1d(d, d, self.kc, groups=d, bias=True)
            self.W_qkv_delta = nn.Linear(d, 3 * self.H_delta * dh, bias=False)
            self.W_beta = nn.Linear(d, self.H_delta, bias=True)
            self.beta_logit = nn.Parameter(torch.zeros(self.H_delta))
            self.q_fixed = nn.Parameter(torch.randn(self.H_delta, dh) * 0.02)
            self.W_alpha = nn.Linear(d, self.H_delta, bias=True) if cfg.gated else None
            self.state_rms = RMSNorm(dh) if cfg.state_norm else None
            self.W_e = nn.Linear(d, self.H_delta, bias=True) if cfg.decoupled_gate else None
            self.W_eta = nn.Linear(d, self.H_delta * dh, bias=True) if cfg.inctx_lr else None
            
            if cfg.n_delta >= 2:
                self.W_kv_extra = nn.ModuleList([
                    nn.Linear(d, 2 * self.H_delta * dh, bias=False) for _ in range(cfg.n_delta - 1)
                ])
                self.W_beta_extra = nn.ModuleList([
                    nn.Linear(d, self.H_delta, bias=True) for _ in range(cfg.n_delta - 1)
                ])
            else:
                self.W_kv_extra = None
                self.W_beta_extra = None
                
            self.d_phi = cfg.d_phi
            if cfg.feat_map == "quad2":
                g = torch.Generator().manual_seed(1234)
                self.register_buffer("feat_I", torch.randint(0, dh, (cfg.feat_n2,), generator=g))
                self.register_buffer("feat_J", torch.randint(0, dh, (cfg.feat_n2,), generator=g))
            elif cfg.feat_map == "quad2_lowrank":
                r = cfg._feat_rank_eff
                g = torch.Generator().manual_seed(1234)
                self.register_buffer("feat_P", torch.randn(dh, r, generator=g) * (dh ** -0.5))
                n_pairs = r * (r + 1) // 2
                I_lr = torch.tensor([i for i in range(r) for j in range(i, r)], dtype=torch.long)
                J_lr = torch.tensor([j for i in range(r) for j in range(i, r)], dtype=torch.long)
                self.register_buffer("feat_I_lr", I_lr)
                self.register_buffer("feat_J_lr", J_lr)
            elif cfg.feat_map == "rand_linear":
                g = torch.Generator().manual_seed(1234)
                self.register_buffer("W_rand", torch.randn(dh, self.d_phi, generator=g) * (dh ** -0.5))
                
        # 3. Shared output parameters
        self.W_o = nn.Linear(d, d, bias=False)
        self.W_g = nn.Linear(d, d, bias=True) if cfg.out_gate else None
        self.norm2 = RMSNorm(d)
        self.mlp = SwiGLU(TFConfig(d_model=d, d_ff=cfg.d_ff))
        self.drop = nn.Dropout(cfg.dropout)
        self.win_scale = dh ** -0.5

    def _phi(self, x):
        if self.cfg.feat_map == "none":
            return x
        if self.cfg.feat_map == "rand_linear":
            return _l2(x @ self.W_rand)
        if self.cfg.feat_map == "quad2_lowrank":
            z = x @ self.feat_P
            two = z[..., self.feat_I_lr] * z[..., self.feat_J_lr]
            return _l2(torch.cat([x, two], dim=-1))
        two = x[..., self.feat_I] * x[..., self.feat_J]
        return _l2(torch.cat([x, two], dim=-1))

    def _window(self, q, k, v):
        T = q.shape[2]
        w = self.cfg.window
        idx = torch.arange(T, device=q.device)
        band = (idx[None, :] <= idx[:, None]) & (idx[None, :] > idx[:, None] - w)
        mask = torch.zeros(T, T, device=q.device, dtype=q.dtype).masked_fill(~band, float("-inf"))
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    def _window_banded(self, q, k, v):
        B, H, T, dh = q.shape
        w = self.cfg.window
        outs = []
        for c0 in range(0, T, w):
            c1 = min(c0 + w, T)
            qc = q[:, :, c0:c1]
            k0 = max(0, c0 - w)
            kc = k[:, :, k0:c1]; vc = v[:, :, k0:c1]
            qi = torch.arange(c0, c1, device=q.device)[:, None]
            ki = torch.arange(k0, c1, device=q.device)[None, :]
            band = (ki <= qi) & (ki > qi - w)
            mask = torch.zeros(c1 - c0, c1 - k0, device=q.device, dtype=q.dtype).masked_fill(~band, float("-inf"))
            outs.append(F.scaled_dot_product_attention(qc, kc, vc, attn_mask=mask))
        return torch.cat(outs, dim=2)

    def forward(self, h):
        B, T, d = h.shape
        h_norm = self.norm1(h)
        
        o_attn = None
        if self.H_attn > 0:
            qkv_attn = self.W_qkv_attn(h_norm).view(B, T, self.H_attn, 3, self.dh)
            q_attn, k_attn, v_attn = qkv_attn.unbind(3)
            q_attn = q_attn.transpose(1, 2)
            k_attn = k_attn.transpose(1, 2)
            v_attn = v_attn.transpose(1, 2)
            
            if self.tf_rope:
                cos, sin = _rope_cache(T, self.dh, h.device, h.dtype)
                q_attn = _apply_rope(q_attn, cos, sin)
                k_attn = _apply_rope(k_attn, cos, sin)
                
            o_attn = F.scaled_dot_product_attention(
                q_attn, k_attn, v_attn, is_causal=True,
                dropout_p=self.cfg.dropout if self.training else 0.0
            )
            
        o_delta = None
        if self.H_delta > 0:
            x_delta = h_norm
            if self.kc > 0:
                xc = F.pad(h_norm.transpose(1, 2), (self.kc - 1, 0))
                x_delta = F.silu(self.conv(xc).transpose(1, 2))
                
            qkv_delta = self.W_qkv_delta(x_delta).view(B, T, self.H_delta, 3, self.dh)
            q, k, v = qkv_delta.unbind(3)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            if self.cfg.rope:
                cos, sin = _rope_cache(T, self.dh, h.device, h.dtype)
                q = _apply_rope(q, cos, sin)
                k = _apply_rope(k, cos, sin)
                
            q, k = _l2(q), _l2(k)
            
            if not self.cfg.route_readout:
                q = _l2(self.q_fixed[None, :, None, :].expand(B, self.H_delta, T, self.dh).contiguous())
                
            if self.cfg.precision_gate == "uniform":
                beta = torch.sigmoid(self.beta_logit)[None, :, None].expand(B, self.H_delta, T) * self.cfg.beta_cap
            elif self.cfg.precision_gate == "random":
                beta = torch.rand(B, self.H_delta, T, device=h.device, dtype=h.dtype) * self.cfg.beta_cap
            else:
                beta = torch.sigmoid(self.W_beta(x_delta)).transpose(1, 2) * self.cfg.beta_cap
                
            if self.W_alpha is not None:
                alpha = torch.sigmoid(self.W_alpha(x_delta)).transpose(1, 2)
                alpha = 0.5 + 0.5 * alpha
            else:
                alpha = None
                
            if self.W_e is not None:
                beta_e = torch.sigmoid(self.W_e(x_delta)).transpose(1, 2) * self.cfg.beta_cap
            else:
                beta_e = None
                
            if self.W_eta is not None:
                eta = torch.sigmoid(self.W_eta(x_delta)).view(B, T, self.H_delta, self.dh).transpose(1, 2)
                eta = eta * self.cfg.beta_cap
            else:
                eta = None
                
            o_workspace = torch.zeros(B, self.H_delta, T, self.dh, device=h.device, dtype=h.dtype)
            if self.cfg.use_workspace:
                if self.cfg.n_delta >= 2:
                    ks = [k]
                    vs = [v]
                    bs = [beta]
                    for i, (wkv, wbeta) in enumerate(zip(self.W_kv_extra, self.W_beta_extra)):
                        kv_i = wkv(x_delta).view(B, T, self.H_delta, 2, self.dh)
                        k_i, v_i = kv_i.unbind(3)
                        k_i = k_i.transpose(1, 2)
                        v_i = v_i.transpose(1, 2)
                        k_i = _l2(k_i)
                        b_i = torch.sigmoid(wbeta(x_delta)).transpose(1, 2) * self.cfg.beta_cap
                        ks.append(k_i)
                        vs.append(v_i)
                        bs.append(b_i)
                    k_nd = torch.stack(ks, dim=3)
                    v_nd = torch.stack(vs, dim=3)
                    b_nd = torch.stack(bs, dim=3)
                    o_workspace, _ = chunked_delta(q, k_nd, v_nd, b_nd, alpha,
                                                   chunk=self.cfg.chunk, write_mode=self.cfg.write_mode,
                                                   beta_e=None, n_delta=self.cfg.n_delta)
                else:
                    surprise_gen = None
                    if self.cfg.surprise_gate and self.cfg.surprise_mode == 'random':
                        surprise_gen = torch.Generator(device=q.device).manual_seed(self.cfg.surprise_seed)
                    o_workspace, _ = chunked_delta(self._phi(q), self._phi(k), v, beta, alpha,
                                                   chunk=self.cfg.chunk, write_mode=self.cfg.write_mode,
                                                   beta_e=beta_e,
                                                   surprise=self.cfg.surprise_gate,
                                                   surprise_mode=self.cfg.surprise_mode,
                                                   surprise_gen=surprise_gen,
                                                   eta=eta)
                if self.state_rms is not None:
                    o_workspace = self.state_rms(o_workspace)
                    
            o_win = torch.zeros(B, self.H_delta, T, self.dh, device=h.device, dtype=h.dtype)
            if self.cfg.use_window:
                win_fn = self._window_banded if self.cfg.banded_window else self._window
                o_win = win_fn(q, k, v)
                
            o_delta = o_workspace + o_win
            
        o = torch.empty(B, self.H, T, self.dh, device=h.device, dtype=h.dtype)
        if self.H_attn > 0:
            o[:, list(self.attn_heads)] = o_attn
        if self.H_delta > 0:
            o[:, list(self.delta_heads)] = o_delta
            
        o = o.transpose(1, 2).reshape(B, T, d)
        if self.W_g is not None:
            o = o * torch.sigmoid(self.W_g(self.norm1(h)))
        h = h + self.drop(self.W_o(o))
        h = h + self.drop(self.mlp(self.norm2(h)))
        return h

    @torch.no_grad()
    def step(self, h_t, state):
        B = h_t.shape[0]
        attn_cache, delta_state = state
        h_norm = self.norm1(h_t)
        
        o_attn = None
        new_attn_cache = None
        if self.H_attn > 0:
            qkv_attn = self.W_qkv_attn(h_norm).view(B, 1, self.H_attn, 3, self.dh)
            q_a, k_a, v_a = qkv_attn.unbind(3)
            q_a = q_a.transpose(1, 2)
            k_a = k_a.transpose(1, 2)
            v_a = v_a.transpose(1, 2)
            
            t = 0 if attn_cache is None else attn_cache[0].shape[2]
            if self.tf_rope:
                cos, sin = _rope_cache(t + 1, self.dh, h_t.device, h_t.dtype)
                q_a = _apply_rope(q_a, cos[t:t + 1], sin[t:t + 1])
                k_a = _apply_rope(k_a, cos[t:t + 1], sin[t:t + 1])
                
            if attn_cache is not None:
                k_a = torch.cat([attn_cache[0], k_a], dim=2)
                v_a = torch.cat([attn_cache[1], v_a], dim=2)
            new_attn_cache = (k_a, v_a)
            
            o_attn = F.scaled_dot_product_attention(q_a, k_a, v_a, is_causal=False)
            
        o_delta = None
        new_delta_state = None
        if self.H_delta > 0:
            S, rk, rv, cring, pos = delta_state
            
            x_delta = h_norm
            if self.kc > 0:
                buf = torch.cat([cring, h_norm], dim=1)
                w = self.conv.weight.squeeze(1)
                xc = (buf.transpose(1, 2) * w).sum(-1) + self.conv.bias
                x_delta = F.silu(xc)[:, None, :]
                cring = buf[:, 1:, :]
                
            qkv_delta = self.W_qkv_delta(x_delta).view(B, 1, self.H_delta, 3, self.dh)
            q, k, v = qkv_delta.unbind(3)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            if self.cfg.rope:
                cos, sin = _rope_cache(1, self.dh, h_t.device, h_t.dtype, offset=pos)
                q = _apply_rope(q, cos, sin)
                k = _apply_rope(k, cos, sin)
                
            q, k = _l2(q), _l2(k)
            q1, k1, v1 = q[:, :, 0], k[:, :, 0], v[:, :, 0]
            
            if self.cfg.precision_gate == "uniform":
                b1 = torch.sigmoid(self.beta_logit)[None, :].expand(B, self.H_delta) * self.cfg.beta_cap
            elif self.cfg.precision_gate == "random":
                b1 = torch.rand(B, self.H_delta, device=h_t.device, dtype=h_t.dtype) * self.cfg.beta_cap
            else:
                b1 = torch.sigmoid(self.W_beta(x_delta))[:, 0, :] * self.cfg.beta_cap
                
            if self.W_alpha is not None:
                a1 = torch.sigmoid(self.W_alpha(x_delta))[:, 0, :]
                a1 = 0.5 + 0.5 * a1
            else:
                a1 = torch.ones_like(b1)
                
            if self.W_e is not None:
                be1 = torch.sigmoid(self.W_e(x_delta))[:, 0, :] * self.cfg.beta_cap
            else:
                be1 = b1
                
            if self.W_eta is not None:
                eta1 = torch.sigmoid(self.W_eta(x_delta)).view(B, self.H_delta, self.dh) * self.cfg.beta_cap
            else:
                eta1 = None
                
            if self.cfg.use_workspace:
                if self.cfg.n_delta >= 2:
                    o_workspace = torch.einsum("bhij,bhj->bhi", S, q1)
                    
                    k1_step = k1
                    v1_step = v1
                    Sk = torch.einsum("bhij,bhj->bhi", S, k1_step)
                    u = b1[..., None] * v1_step - b1[..., None] * (a1[..., None] * Sk)
                    S = a1[..., None, None] * S + torch.einsum("bhi,bhj->bhij", u, k1_step)
                    
                    x1 = x_delta[:, 0, :]
                    for wkv, wbeta in zip(self.W_kv_extra, self.W_beta_extra):
                        kv_j = wkv(x1).view(B, self.H_delta, 2, self.dh)
                        k_j, v_j = kv_j[:, :, 0], kv_j[:, :, 1]
                        k_j = _l2(k_j)
                        b_j = torch.sigmoid(wbeta(x1)).view(B, self.H_delta) * self.cfg.beta_cap
                        Sk_j = torch.einsum("bhij,bhj->bhi", S, k_j)
                        u_j = b_j[..., None] * (v_j - Sk_j)
                        S = S + torch.einsum("bhi,bhj->bhij", u_j, k_j)
                else:
                    q1p = self._phi(q)[:, :, 0]
                    o_workspace = torch.einsum("bhij,bhj->bhi", S, q1p)
                    
                    k1p = self._phi(k)[:, :, 0]
                    Sk = torch.einsum("bhij,bhj->bhi", S, k1p)
                    eps1 = v1 - a1[..., None] * Sk
                    if eta1 is not None:
                        u = eta1 * eps1
                    elif self.cfg.surprise_gate:
                        g1 = (1.0 + torch.tanh(eps1.norm(dim=-1)))[..., None]
                        u = g1 * (b1[..., None] * v1 - be1[..., None] * (a1[..., None] * Sk))
                    else:
                        u = b1[..., None] * v1 - be1[..., None] * (a1[..., None] * Sk)
                    S = a1[..., None, None] * S + torch.einsum("bhi,bhj->bhij", u, k1p)
                if self.state_rms is not None:
                    o_workspace = self.state_rms(o_workspace)
            else:
                o_workspace = torch.zeros(B, self.H_delta, self.dh, device=h_t.device, dtype=h_t.dtype)
                
            if self.cfg.use_window:
                rk = torch.cat([rk, k1[:, :, None]], dim=2)[:, :, -self.cfg.window:]
                rv = torch.cat([rv, v1[:, :, None]], dim=2)[:, :, -self.cfg.window:]
                sc = torch.einsum("bhd,bhwd->bhw", q1, rk) * self.win_scale
                aw = torch.softmax(sc, dim=-1)
                o_win = torch.einsum("bhw,bhwd->bhd", aw, rv)
            else:
                o_win = torch.zeros(B, self.H_delta, self.dh, device=h_t.device, dtype=h_t.dtype)
                
            o_delta = (o_workspace + o_win)[:, :, None, :]
            new_delta_state = (S, rk, rv, cring, pos + 1)
            
        o = torch.empty(B, self.H, 1, self.dh, device=h_t.device, dtype=h_t.dtype)
        if self.H_attn > 0:
            o[:, list(self.attn_heads)] = o_attn
        if self.H_delta > 0:
            o[:, list(self.delta_heads)] = o_delta
            
        o = o.transpose(1, 2).reshape(B, 1, -1)
        if self.W_g is not None:
            o = o * torch.sigmoid(self.W_g(self.norm1(h_t)))
            
        h = h_t + self.drop(self.W_o(o))
        h = h + self.drop(self.mlp(self.norm2(h)))
        return h, (new_attn_cache, new_delta_state)


class HybridSeqLM(nn.Module):
    """Mostly-Prizma + ~1 attention layer baseline (param-matched to the Transformer).

    Args:
      prizma_cfg: the PrizmaSeqConfig used for every NON-attention layer (carries d_model,
                  n_layers, n_heads, max_len, vocab + all Prizma lever knobs).
      n_attn:     number of attention layers (auto-placed, centered) when attn_layers is None.
      attn_layers: explicit attention layer index/indices (int or iterable); overrides n_attn.
      tf_rope:    RoPE on the attention layers (matches the TF baseline default True).
      attn_heads: mapping/sequence specifying custom attention head indices per layer.
                  e.g., dict mapping {layer_idx: [head_idx, ...]} or tuple of head indices.
    """

    def __init__(self, prizma_cfg: PrizmaSeqConfig, n_attn: int = 1, attn_layers=None,
                 tf_rope: bool = True, attn_heads=None):
        super().__init__()
        self.cfg = prizma_cfg
        n_layers = prizma_cfg.n_layers
        self.layer_attn_heads = resolve_hybrid_config(n_layers, prizma_cfg.n_heads, n_attn, attn_layers, attn_heads)
        self.attn_layers = tuple(sorted(i for i in range(n_layers) if len(self.layer_attn_heads[i]) > 0))

        # Attention Block config: match d_model / n_heads / max_len / rope to the Transformer
        # baseline, and size its SwiGLU FFN from the SAME d_ff the Prizma layers use so the FFN
        # is byte-identical across all layers (keeps the param-match to TF tight).
        tf_cfg = TFConfig(
            vocab=prizma_cfg.vocab,
            d_model=prizma_cfg.d_model,
            n_layers=n_layers,          # informational only; we build Blocks individually
            n_heads=prizma_cfg.n_heads,
            d_ff=prizma_cfg.d_ff,
            max_len=prizma_cfg.max_len,
            rope=tf_rope,
        )
        self.tf_cfg = tf_cfg

        # Embedding / positions: reuse the PrizmaSeqLM conventions exactly.
        self.tok = nn.Embedding(prizma_cfg.vocab, prizma_cfg.d_model)
        self.pos = nn.Embedding(prizma_cfg.max_len, prizma_cfg.d_model) if prizma_cfg.learned_pos else None

        blocks = []
        for i in range(n_layers):
            heads = self.layer_attn_heads[i]
            if len(heads) == prizma_cfg.n_heads:
                blocks.append(Block(tf_cfg))
            elif len(heads) == 0:
                blocks.append(PrizmaSeqBlock(prizma_cfg))
            else:
                blocks.append(HybridBlock(prizma_cfg, attn_heads=heads, tf_rope=tf_rope))
        self.blocks = nn.ModuleList(blocks)

        self.nf = RMSNorm(prizma_cfg.d_model)
        self.head = nn.Linear(prizma_cfg.d_model, prizma_cfg.vocab, bias=False)
        self.head.weight = self.tok.weight              # tied head (matches PrizmaSeqLM)
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
            h = blk(h)                                  # all block types: forward(h) -> [B,T,d]
        return self.head(self.nf(h))

    # ---- streaming decode (O(n) OVERALL: the attention layer's KV cache grows). --------------- #
    # Provided for latency-probe completeness only. NOT an O(1) path; no O(1) guard exists for it.
    @torch.no_grad()
    def init_state(self, batch, device):
        """Per-layer state: PrizmaSeqBlock layers get the O(1) Prizma tuple; attention layers get a
        growing KV cache (None at t=0). pos tracks the learned-pos offset (if any)."""
        cfg = self.cfg
        kc1 = max(cfg.short_conv - 1, 0)
        state_k_dim = cfg.d_h if cfg.n_delta >= 2 else cfg.d_phi
        st = []
        for blk in self.blocks:
            if isinstance(blk, Block):
                st.append(None)                          # attention KV cache (grows with t) -> O(t)
            elif isinstance(blk, PrizmaSeqBlock):
                S = torch.zeros(batch, cfg.n_heads, cfg.d_h, state_k_dim, device=device)
                rk = torch.zeros(batch, cfg.n_heads, 0, cfg.d_h, device=device)
                rv = torch.zeros(batch, cfg.n_heads, 0, cfg.d_h, device=device)
                cring = torch.zeros(batch, kc1, cfg.d_model, device=device)
                st.append((S, rk, rv, cring, 0))
            elif isinstance(blk, HybridBlock):
                if blk.H_delta > 0:
                    S = torch.zeros(batch, blk.H_delta, cfg.d_h, state_k_dim, device=device)
                    rk = torch.zeros(batch, blk.H_delta, 0, cfg.d_h, device=device)
                    rv = torch.zeros(batch, blk.H_delta, 0, cfg.d_h, device=device)
                    cring = torch.zeros(batch, kc1, cfg.d_model, device=device)
                    st.append((None, (S, rk, rv, cring, 0)))
                else:
                    st.append((None, None))
        return st

    @torch.no_grad()
    def step(self, tok, state):
        """tok:[B,1] -> (logits[B,1,V], new_state). O(n) OVERALL (one attention layer is KV-cached).

        Composes PrizmaSeqBlock.step (O(1) per layer) with Block.step (O(t) KV-cached per layer).
        """
        h = self.tok(tok)
        if self.pos is not None:
            # learned-pos offset: read from the first Prizma layer's state tuple (pos slot) if present,
            # else fall back to the attention KV length, else 0.
            p = 0
            for blk, st in zip(self.blocks, state):
                if isinstance(blk, PrizmaSeqBlock) and st is not None:
                    p = st[4]
                    break
                elif isinstance(blk, HybridBlock) and st is not None and st[1] is not None:
                    p = st[1][4]
                    break
            else:
                for blk, st in zip(self.blocks, state):
                    if isinstance(blk, Block) and st is not None:
                        p = st[0].shape[2]
                        break
                    elif isinstance(blk, HybridBlock) and st is not None and st[0] is not None:
                        p = st[0][0].shape[2]
                        break
            h = h + self.pos(torch.tensor([p], device=tok.device))[None]
        new = []
        for blk, st in zip(self.blocks, state):
            h, st2 = blk.step(h, st)                     # Block.step + PrizmaSeqBlock.step share (h_t, st)->(h, st)
            new.append(st2)
        return self.head(self.nf(h)), new


def hybrid_factory(d, L, H, n_attn=1, attn_layers=None, tf_rope=True, attn_heads=None, **prizma_kw):
    """Factory mirroring ps_factory: hybrid_factory(d, L, H, ...) -> (lambda V, T: HybridSeqLM(...)).

    Drops into run_cell / recall-gate / gpu_bench as a THIRD arm exactly like ps_factory.
    `prizma_kw` are forwarded to PrizmaSeqConfig (feat_map, gated, window, etc.).
    """
    def f(vocab, max_len):
        cfg = PrizmaSeqConfig(vocab=vocab, d_model=d, n_layers=L, n_heads=H,
                              max_len=max_len + 8, **prizma_kw)
        return HybridSeqLM(cfg, n_attn=n_attn, attn_layers=attn_layers, tf_rope=tf_rope, attn_heads=attn_heads)
    return f


def _print_param_match(d=128, L=4, H=4, vocab=64, max_len=300, **prizma_kw):
    """Print Hybrid vs TF param counts at a scale and confirm the spread (mirrors gpu_diag.py)."""
    from .transformer import Transformer
    from .common import param_count
    hyb = hybrid_factory(d, L, H, **prizma_kw)(vocab, max_len)
    tf = Transformer(TFConfig(vocab=vocab, d_model=d, n_layers=L, n_heads=H, max_len=max_len + 8, rope=True))
    p_hyb, p_tf = param_count(hyb), param_count(tf)
    rel = (p_hyb - p_tf) / p_tf
    print(f"  param-match @ d{d}L{L}H{H} (V={vocab}):", flush=True)
    print(f"    {'TF':<14} {p_tf:>9,}p  (  +0.00% vs TF)", flush=True)
    print(f"    {'Hybrid':<14} {p_hyb:>9,}p  ({100.0 * rel:+.2f}% vs TF)  "
          f"[attn_layers={hyb.attn_layers}]", flush=True)
    tag = "MATCHED <=3%" if abs(rel) <= 0.03 else "NOTE: >3% (baseline; disclosed)"
    print(f"    -> hybrid-vs-TF spread {100.0 * rel:+.2f}% of TF ({tag})", flush=True)
    return p_hyb, p_tf, rel


if __name__ == "__main__":
    import torch as _t
    # 1. Test standard factory
    m = hybrid_factory(128, 4, 4)(64, 128)
    x = _t.randint(0, 64, (2, 48))
    types = [type(b).__name__ for b in m.blocks]
    print("blocks:", types, "attn_layers:", m.attn_layers)
    print("logits", tuple(m(x).shape))
    _print_param_match()

    # 2. Test flexible head-wise mixing
    print("\n--- Testing head-wise mixing step vs forward parity ---")
    # Say 4 heads total, layer 0 has heads (0, 1) as attention, layer 2 has heads (2, 3) as attention.
    # Other layers are fully delta or fully attention.
    attn_heads_cfg = {0: (0, 1), 1: (), 2: (2, 3), 3: (0, 1, 2, 3)}
    m_mixed = hybrid_factory(128, 4, 4, attn_heads=attn_heads_cfg)(64, 128)
    m_mixed.train(False)
    
    types_mixed = [type(b).__name__ for b in m_mixed.blocks]
    print("Mixed blocks:", types_mixed)
    # Check that mixed layers are of type HybridBlock
    assert types_mixed[0] == "HybridBlock", f"Expected HybridBlock, got {types_mixed[0]}"
    assert types_mixed[1] == "PrizmaSeqBlock", f"Expected PrizmaSeqBlock, got {types_mixed[1]}"
    assert types_mixed[2] == "HybridBlock", f"Expected HybridBlock, got {types_mixed[2]}"
    assert types_mixed[3] == "Block", f"Expected Block, got {types_mixed[3]}"
    
    x_mixed = _t.randint(0, 64, (2, 40))
    y_mixed = m_mixed(x_mixed)
    st_mixed = m_mixed.init_state(2, _t.device("cpu"))
    
    outs_mixed = []
    for t in range(x_mixed.shape[1]):
        lg_mixed, st_mixed = m_mixed.step(x_mixed[:, t:t + 1], st_mixed)
        outs_mixed.append(lg_mixed)
    d_mixed = (y_mixed - _t.cat(outs_mixed, dim=1)).abs().max().item()
    print(f"Step vs forward max difference: {d_mixed:.2e}")
    assert d_mixed < 1e-4, "Mixed hybrid step vs forward mismatch!"
    print("Mixed hybrid step-vs-forward OK!")
