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
    feat_map: str = "none"          # 'none' | 'quad2' | 'quad2_lowrank' | 'rand_linear'.
                                    #   'quad2': expands delta keys/queries from d_h to
                                    #     d_phi = d_h + feat_n2 via FIXED random quadratic monomials
                                    #     -> rectangular state S in R^{d_h x d_phi}, raising
                                    #     associative-recall key-rank toward D=128 at ZERO trainable
                                    #     params (buffers, not Parameters). O(1) inference intact.
                                    #   'quad2_lowrank': leaner variant — projects x (d_h) -> r dims
                                    #     via a FIXED seeded matrix P in R^{d_h x r}, then takes ALL
                                    #     r*(r+1)/2 quadratic monomials in the r-dim projected space,
                                    #     giving d_phi = d_h + r*(r+1)/2. Default r=feat_rank (or 14
                                    #     when feat_rank=0), which yields d_phi=137 (~54% of quad2's
                                    #     256 at d_h=32) while preserving recall capacity (crosstalk
                                    #     gap from quad2 < 0.01). Still ZERO trainable params —
                                    #     P and monomial indices are buffers seeded at 1234.
    feat_n2: int = 96               # number of quadratic monomials (d_phi = d_h + feat_n2); used by
                                    #   'quad2' and 'rand_linear' only (ignored by 'quad2_lowrank').
    feat_rank: int = 0              # low-rank projection dim r for 'quad2_lowrank'. When 0, defaults
                                    #   to 14 -> d_phi = d_h + 14*15//2 = d_h + 105 (≈137 for d_h=32).
                                    #   Effective d_phi = d_h + r*(r+1)//2 (documented in __post_init__).
    out_gate: bool = False      # per-token output gate g=sigma(W_g x); o = o * g before W_o (RWKV-7/GLA)
    state_norm: bool = False    # per-head RMSNorm on the delta-state read o_delta before merge
    banded_window: bool = False # O(T*w) banded sliding-window kernel (exact-equal to _window, default off)
    decoupled_gate: bool = False  # GDN-2: decouple erase gate beta_e (key-side) from write gate beta_w
    n_delta: int = 1              # DeltaProduct: number of sequential delta sub-steps per token (k=1 = today)
    # --- surprise-gated write (Lever A) ---
    surprise_gate: bool = False   # if True, scale write by g_t = 1+tanh(||eps_t||) (default OFF = identical)
    surprise_mode: str = 'norm'   # 'norm' | 'random' | 'constant' — controls for R9 ablation

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = int(round(8 / 3 * self.d_model / 8) * 8)
        assert self.d_model % self.n_heads == 0
        self.d_h = self.d_model // self.n_heads
        assert self.d_h % 2 == 0, "d_h must be even for RoPE"
        assert self.feat_map in ("none", "quad2", "quad2_lowrank", "rand_linear")
        assert self.surprise_mode in ('norm', 'random', 'constant'), \
            f"surprise_mode must be 'norm', 'random', or 'constant', got {self.surprise_mode!r}"
        # d_phi = delta key/query dim after the optional feature map (= d_h when 'none').
        # 'rand_linear' = a FIXED random linear map d_h->d_phi (a CONTROL: it stays in a d_h-rank
        # subspace so it must give NO capacity gain, proving the quad2 MONOMIALS are what help).
        # 'quad2_lowrank': effective r = feat_rank if feat_rank > 0 else 14 (default);
        #   d_phi = d_h + r*(r+1)//2  (all upper-triangular monomial pairs in the projected space).
        if self.feat_map == "quad2_lowrank":
            _r = self.feat_rank if self.feat_rank > 0 else 14
            self._feat_rank_eff = _r
            self.d_phi = self.d_h + _r * (_r + 1) // 2
        elif self.feat_map in ("quad2", "rand_linear"):
            self._feat_rank_eff = 0
            self.d_phi = self.d_h + self.feat_n2
        else:
            self._feat_rank_eff = 0
            self.d_phi = self.d_h


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
        self.W_g = nn.Linear(d, d, bias=True) if cfg.out_gate else None
        self.state_rms = RMSNorm(dh) if cfg.state_norm else None
        self.W_e = nn.Linear(d, H, bias=True) if cfg.decoupled_gate else None   # erase gate beta_e
        # DeltaProduct n_delta>=2: (n_delta-1) additional kv projections + beta heads.
        # Each extra sub-step j>=1 gets its own W_kv_j and W_beta_j (independent params).
        # Total param cost: (n_delta-1) * (2*d*d + H) trainable weights per block.
        # NOTE: the plan explicitly allows repeated projections for simplicity (documented).
        self.W_kv_extra = nn.ModuleList([
            nn.Linear(d, 2 * d, bias=False) for _ in range(cfg.n_delta - 1)
        ]) if cfg.n_delta >= 2 else None
        self.W_beta_extra = nn.ModuleList([
            nn.Linear(d, H, bias=True) for _ in range(cfg.n_delta - 1)
        ]) if cfg.n_delta >= 2 else None
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
        elif cfg.feat_map == "quad2_lowrank":
            # Leaner variant (Task 1.D): project x (d_h) -> r dims via FIXED seeded P in R^{d_h x r},
            # then take ALL r*(r+1)/2 upper-triangular monomial pairs in the r-dim space.
            # d_phi = d_h + r*(r+1)//2. At r=14 (default): d_phi = 32+105=137 for d_h=32.
            # P is a buffer (ZERO trainable params), seeded same discipline as quad2 (seed 1234).
            r = cfg._feat_rank_eff
            g = torch.Generator().manual_seed(1234)
            self.register_buffer("feat_P", torch.randn(dh, r, generator=g) * (dh ** -0.5))
            n_pairs = r * (r + 1) // 2
            I_lr = torch.tensor([i for i in range(r) for j in range(i, r)], dtype=torch.long)
            J_lr = torch.tensor([j for i in range(r) for j in range(i, r)], dtype=torch.long)
            self.register_buffer("feat_I_lr", I_lr)
            self.register_buffer("feat_J_lr", J_lr)
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
        (D=128: ~0.142 -> ~0.117 for quad2/d_phi=256, key_crosstalk metric). The final _l2 preserves
        ||k||=1, the invariant the delta kernel relies on. The local-window head keeps linear L2 keys.
        'quad2_lowrank': project x -> z=x@P (r-dim), take all r*(r+1)/2 monomials z[I]*z[J],
        concat [x; monomials] -> d_phi = d_h + r*(r+1)/2 (leaner than quad2 at ~half d_phi)."""
        if self.cfg.feat_map == "none":
            return x
        if self.cfg.feat_map == "rand_linear":
            return _l2(x @ self.W_rand)               # control: rank <= d_h -> no capacity gain
        if self.cfg.feat_map == "quad2_lowrank":
            z = x @ self.feat_P                        # [..., r]  (fixed seeded projection)
            two = z[..., self.feat_I_lr] * z[..., self.feat_J_lr]   # [..., n_pairs]
            return _l2(torch.cat([x, two], dim=-1))   # [..., d_phi = d_h + n_pairs]
        # quad2: fixed random monomials from the d_h-dim input
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
        if self.W_e is not None:
            beta_e = torch.sigmoid(self.W_e(x)).transpose(1, 2) * self.cfg.beta_cap   # [B,H,T]
        else:
            beta_e = None          # => chunked_delta will use beta_e = beta (byte-identical)
        return q, k, v, beta, alpha, beta_e

    def _window(self, q, k, v):
        """Exact causal attention restricted to the last `w` tokens (incl. self). Fused SDPA + band
        mask (fast on MPS); scaling 1/sqrt(d_h) matches the streaming step() path."""
        T = q.shape[2]
        w = self.cfg.window
        idx = torch.arange(T, device=q.device)
        band = (idx[None, :] <= idx[:, None]) & (idx[None, :] > idx[:, None] - w)  # [T,T] True=allow
        mask = torch.zeros(T, T, device=q.device, dtype=q.dtype).masked_fill(~band, float("-inf"))
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)            # [B,H,T,dh]

    def _window_banded(self, q, k, v):
        """Sliding-window causal attention in O(T*w) via fixed-size chunks of size w. Each query in
        chunk c attends keys in chunks {c-1, c} masked to [i-w+1, i]. Numerically equals _window."""
        B, H, T, dh = q.shape
        w = self.cfg.window
        outs = []
        for c0 in range(0, T, w):
            c1 = min(c0 + w, T)
            qc = q[:, :, c0:c1]                           # [B,H,Cq,dh]
            k0 = max(0, c0 - w)
            kc = k[:, :, k0:c1]; vc = v[:, :, k0:c1]     # span <= 2w
            qi = torch.arange(c0, c1, device=q.device)[:, None]
            ki = torch.arange(k0, c1, device=q.device)[None, :]
            band = (ki <= qi) & (ki > qi - w)
            mask = torch.zeros(c1 - c0, c1 - k0, device=q.device, dtype=q.dtype).masked_fill(~band, float("-inf"))
            outs.append(F.scaled_dot_product_attention(qc, kc, vc, attn_mask=mask))
        return torch.cat(outs, dim=2)

    def forward(self, h):
        B, T, d = h.shape
        x = self._apply_conv(self.norm1(h))
        cos, sin = _rope_cache(T, self.dh, h.device, h.dtype) if self.cfg.rope else (None, None)
        q, k, v, beta, alpha, beta_e = self._encode(x, cos, sin)
        o = torch.zeros(B, self.H, T, self.dh, device=h.device, dtype=h.dtype)
        if self.cfg.use_workspace:
            # DeltaProduct: for n_delta>=2, build multi-sub-step k,v,beta tensors
            if self.cfg.n_delta >= 2:
                # sub-step 0 uses the main projection; sub-steps 1..n_delta-1 use W_kv_extra
                ks = [k]    # each [B,H,T,d_h]
                vs = [v]
                bs = [beta]
                for i, (wkv, wbeta) in enumerate(zip(self.W_kv_extra, self.W_beta_extra)):
                    kv_i = wkv(x).view(B, T, self.H, 2, self.dh)   # [B,T,H,2,dh]
                    k_i, v_i = kv_i.unbind(3)                       # each [B,T,H,dh]
                    k_i = k_i.transpose(1, 2); v_i = v_i.transpose(1, 2)   # [B,H,T,dh]
                    k_i = _l2(k_i)                                   # unit-norm
                    b_i = torch.sigmoid(wbeta(x)).transpose(1, 2) * self.cfg.beta_cap  # [B,H,T]
                    ks.append(k_i); vs.append(v_i); bs.append(b_i)
                # Stack on a new sub-step axis: [B,H,T,n_delta,d_h]
                k_nd = torch.stack(ks, dim=3)
                v_nd = torch.stack(vs, dim=3)
                b_nd = torch.stack(bs, dim=3)
                # phi applied per sub-step; for simplicity phi(k_sub_0) uses main k (already phi'd below)
                # For k>=2 we apply phi to sub-step 0 key; extra sub-steps use linear keys (d_h, not d_phi)
                # NOTE: n_delta>=2 uses d_h-dimensional state (no feat_map for sub-steps 1+, linear keys)
                # Sub-step 0 key goes through phi; but the state dim must be consistent: we use d_h for all
                # sub-steps when n_delta>=2 (phi expansion is not combined with n_delta in this impl).
                o_delta, _ = chunked_delta(q, k_nd, v_nd, b_nd, alpha,
                                           chunk=self.cfg.chunk, write_mode=self.cfg.write_mode,
                                           beta_e=None, n_delta=self.cfg.n_delta)  # [B,H,T,d_h]
            else:
                # n_delta=1: standard path (byte-identical when surprise_gate=False, with phi and beta_e)
                o_delta, _ = chunked_delta(self._phi(q), self._phi(k), v, beta, alpha,
                                           chunk=self.cfg.chunk, write_mode=self.cfg.write_mode,
                                           beta_e=beta_e,
                                           surprise=self.cfg.surprise_gate,
                                           surprise_mode=self.cfg.surprise_mode)   # [B,H,T,d_h]
            # delta state keyed by phi(q),phi(k) (dim d_phi); values stay d_h -> state [B,H,d_h,d_phi]
            if self.state_rms is not None:
                o_delta = self.state_rms(o_delta)    # per-head RMSNorm over d_h
            o = o + o_delta
        if self.cfg.use_window:
            win_fn = self._window_banded if self.cfg.banded_window else self._window
            o = o + win_fn(q, k, v)                  # window head keeps the LINEAR L2 keys (dim d_h)
        o = o.transpose(1, 2).reshape(B, T, d)                                     # merge heads
        if self.W_g is not None:
            o = o * torch.sigmoid(self.W_g(self.norm1(h)))   # gate on block input (pre-conv normed)
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
        q, k, v, beta, alpha, beta_e = self._encode(x, cos, sin)  # [B,H,1,dh], beta [B,H,1]
        q1, k1, v1 = q[:, :, 0], k[:, :, 0], v[:, :, 0]          # [B,H,dh] (linear L2; window keys)
        b1 = beta[:, :, 0]                                       # [B,H]  write gate beta_w
        a1 = alpha[:, :, 0] if alpha is not None else torch.ones_like(b1)
        # pre-write read (always uses state from end of previous token)
        if self.cfg.n_delta >= 2:
            o_delta = torch.einsum("bhij,bhj->bhi", S, q1)        # [B,H,d_h] (d_h state for n_delta>=2)
        else:
            q1p = self._phi(q)[:, :, 0]                           # [B,H,d_phi] (delta state keys)
            o_delta = torch.einsum("bhij,bhj->bhi", S, q1p)       # pre-write read S_{t-1} phi(q)
        if self.cfg.n_delta >= 2:
            # DeltaProduct: apply n_delta sub-steps sequentially (mirrors the chunked/reference form)
            # Sub-step 0: main kv + alpha decay
            k1_step = k1; v1_step = v1
            Sk = torch.einsum("bhij,bhj->bhi", S, k1_step)
            u = b1[..., None] * v1_step - b1[..., None] * (a1[..., None] * Sk)
            S = a1[..., None, None] * S + torch.einsum("bhi,bhj->bhij", u, k1_step)
            # Sub-steps 1..(n_delta-1): extra projections, no additional alpha decay
            x1 = x[:, 0, :]                                       # [B,d] (squeeze T=1 dim)
            for wkv, wbeta in zip(self.W_kv_extra, self.W_beta_extra):
                kv_j = wkv(x1).view(B, self.H, 2, self.dh)        # [B,H,2,dh]
                k_j, v_j = kv_j[:, :, 0], kv_j[:, :, 1]           # [B,H,dh]
                k_j = _l2(k_j)
                b_j = torch.sigmoid(wbeta(x1)).view(B, self.H) * self.cfg.beta_cap  # [B,H]
                Sk_j = torch.einsum("bhij,bhj->bhi", S, k_j)
                u_j = b_j[..., None] * (v_j - Sk_j)
                S = S + torch.einsum("bhi,bhj->bhij", u_j, k_j)
        else:
            k1p = self._phi(k)[:, :, 0]                           # [B,H,d_phi] (delta state keys)
            be1 = beta_e[:, :, 0] if beta_e is not None else b1   # [B,H]  erase gate beta_e
            Sk = torch.einsum("bhij,bhj->bhi", S, k1p)            # [B,H,d_h]
            # Prediction error (free-energy gradient at S_{t-1}): eps = v - alpha*S*k
            eps1 = v1 - a1[..., None] * Sk                        # [B,H,d_h]
            if self.cfg.surprise_gate:
                # g_t = 1 + tanh(||eps_t||); same formula as _delta_reference / _surprise_gate 'norm'
                # (only 'norm' mode is used in step() — the 'random'/'constant' modes are ablation-only
                # and are not wired through step() since they are not meaningful at inference time).
                # For step==forward parity with surprise_mode='norm', this is exact.
                g1 = (1.0 + torch.tanh(eps1.norm(dim=-1)))[..., None]   # [B,H,1]
                # Apply g to full write vector: u = g * (beta_w*v - beta_e*alpha*Sk)
                u = g1 * (b1[..., None] * v1 - be1[..., None] * (a1[..., None] * Sk))
            else:
                # decoupled: u = beta_w * v  -  beta_e * (alpha * S k)
                u = b1[..., None] * v1 - be1[..., None] * (a1[..., None] * Sk)   # [B,H,d_h]
            S = a1[..., None, None] * S + torch.einsum("bhi,bhj->bhij", u, k1p)   # [B,H,d_h,d_phi]
        if self.state_rms is not None:
            o_delta = self.state_rms(o_delta)    # per-head RMSNorm [B,H,d_h], mirrors forward
        # window ring
        rk = torch.cat([rk, k1[:, :, None]], dim=2)[:, :, -self.cfg.window:]
        rv = torch.cat([rv, v1[:, :, None]], dim=2)[:, :, -self.cfg.window:]
        sc = torch.einsum("bhd,bhwd->bhw", q1, rk) * self.win_scale
        aw = torch.softmax(sc, dim=-1)
        o_win = torch.einsum("bhw,bhwd->bhd", aw, rv)
        o = (o_delta + o_win).reshape(B, 1, -1)
        if self.W_g is not None:
            o = o * torch.sigmoid(self.W_g(self.norm1(h_t)))   # gate on block input, mirrors forward
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
        # For n_delta>=2, state uses d_h-dim keys (no phi expansion); for n_delta=1 uses d_phi
        state_k_dim = self.cfg.d_h if self.cfg.n_delta >= 2 else self.cfg.d_phi
        for _ in self.blocks:
            S = torch.zeros(batch, self.cfg.n_heads, self.cfg.d_h, state_k_dim, device=device)
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
    # O(1) GUARD (committee guardrail): for ALL feat_map settings the streaming step() must equal
    # the parallel forward() to <1e-4, AND param_count must be identical (feature map = 0 params).
    ref_params = None
    for feat in ("none", "quad2", "quad2_lowrank"):
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
        p = param_count(m)
        if ref_params is None:
            ref_params = p
        param_ok = (p == ref_params)
        print(f"[feat_map={feat:<12} d_phi={cfg.d_phi:<4}] forward {tuple(y.shape)} "
              f"params {p} {'(MATCH)' if param_ok else '(MISMATCH!)'} "
              f"step-vs-forward max|d|={d:.2e} {'OK' if d < 1e-4 else 'MISMATCH'}")
    # SURPRISE GATE O(1) guard: surprise_gate=True step()==forward() < 1e-4
    cfg_s = PrizmaSeqConfig(vocab=64, d_model=64, n_layers=2, n_heads=2,
                            feat_map='none', surprise_gate=True, surprise_mode='norm')
    m_s = PrizmaSeqLM(cfg_s).to(dev)
    m_s.train(False)
    torch.manual_seed(7)
    x_s = torch.randint(0, 64, (2, 48), device=dev)
    y_s = m_s(x_s)
    st_s = m_s.init_state(2, dev)
    outs_s = []
    for t in range(x_s.shape[1]):
        lg_s, st_s = m_s.step(x_s[:, t:t + 1], st_s)
        outs_s.append(lg_s)
    yo_s = torch.cat(outs_s, dim=1)
    d_s = (y_s - yo_s).abs().max().item()
    print(f"[surprise_gate=True norm] step-vs-forward max|d|={d_s:.2e} {'OK' if d_s < 1e-4 else 'MISMATCH'}")
