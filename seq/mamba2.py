"""
A FAITHFUL Mamba-2 (SSD / State-Space Duality) baseline — Dao & Gu, "Transformers are SSMs:
Generalized Models and Efficient Algorithms Through Structured State Space Duality" (ICML 2024).

Mamba-2 is THE dominant non-attention SOTA architecture (a STRUCTURED STATE-SPACE MODEL), the
counterpart to the GLA gated-linear-attention baseline in this study. It is built as a NON-strawman:
the SHORT DEPTHWISE CAUSAL CONV (Mamba's local mixing), the GATED OUTPUT (z gate) AND the D SKIP are
ALL included — omitting any of Mamba's components silently weakens the baseline and would corrupt the
Pareto comparison.

SCALAR-A SSD DESIGN (the minimal-faithful published SSD form; documented choices).
  Per layer, input x in R^{B x T x d} (d=d_model), H heads, head channel dim P = d/H, SSM state dim
  N = d_state (default 16). Per head h (all in-projs are bias-free Linear unless noted):
    xin = W_x x   reshaped [B,H,T,P]   (SSM input / "value" path)
    z   = W_z x   reshaped [B,H,T,P]   (output gate)
    B_t = W_B x   reshaped [B,H,T,N]   (data-dependent, per head)
    C_t = W_C x   reshaped [B,H,T,N]   (data-dependent, per head)
    dt  = softplus(W_dt x + dt_bias)   reshaped [B,H,T]   (per-head scalar step, > 0; W_dt has bias)
  A = -exp(A_log) is a learned per-head NEGATIVE scalar (A_log a Parameter[H]); so A < 0.
  SHORT DEPTHWISE CAUSAL CONV on xin (Mamba's local mixing — NOT omitted): a depthwise conv1d
  (kernel k=4, groups=channels, causal left-pad k-1) over T, then silu. The streaming cache of the
  last k-1 timesteps is part of the O(1) state.
  DISCRETIZED SSD RECURRENCE, per-head state Hst in R^{P x N}, Hst_0 = 0:
    a_t   = exp(dt_t * A)                         # scalar in (0,1) per (head,t)  [ZOH on A]
    Hst_t = a_t * Hst_{t-1} + dt_t * outer(xin_t, B_t)   # [P,N]; input term dt_t*(xin ⊗ B) [ZOH on B]
    y_t   = Hst_t @ C_t                           # contract over N -> [P]
    y_t   = y_t + D * xin_t                       # D skip, D a Parameter[H,P] per channel
  OUTPUT: y = y * silu(z) (gated), concat heads -> out_proj W_o (d -> d).

  DOCUMENTED CHOICES (where the published SSD form leaves a detail to the implementer).
   * SCALAR (per-head) A and dt — the SSD scalar-identity form (Dao & Gu Sec. 3-4): the recurrence
     decay a_t is a single scalar per (head,t), which is exactly what makes the chunk form a clean
     decayed-attention (no per-channel gate bookkeeping, unlike GLA). The "selective" data-dependence
     enters through dt_t (softplus of a data-dependent projection) and through B_t, C_t.
   * The short conv is applied to xin ONLY (the value path). Mamba-2's reference also convs B and C;
     conv'ing xin is the MINIMAL faithful choice that keeps the local-mixing inductive bias and the
     O(1) conv cache while keeping the param-match to the Transformer tight — documented here, and the
     recurrent/chunk/step forms are all consistent with this choice.
   * ZOH discretization: a_t = exp(dt_t*A) on the state decay and the input term scaled by dt_t (the
     B-side ZOH simplification dt*B used by the Mamba-2 reference rather than the (a-1)/A*B form).
   * D skip is per-(head,channel) (Parameter[H,P]); the output gate z is the full per-channel SiLU
     gate (the Mamba "gated MLP-like" output), NOT a scalar.
   * BLOCK: a Mamba-2 mixer in a pre-norm Transformer block for a fair param-matched comparison —
     x = x + Mamba2mixer(RMSNorm1(x)); x = x + SwiGLU(RMSNorm2(x)). NO positional embedding (the SSM
     is implicitly positional via the causal scan + per-step decay + the causal conv).

THREE FORMS + the O(1) rigor bar (the project's non-negotiable gate).
  1. RECURRENT reference (_recurrent): a sequential scan over T implementing the recurrence above
     exactly. Vectorized over [B,H,P,N]; python loop over t. Ground truth; used by step().
  2. CHUNK-PARALLEL forward (_chunk): scalar-A makes this a clean decayed causal attention. Per-head
     cumulative log-decay L_i = sum_{s<=i} dt_s * A (<= 0). Intra-chunk:
        y_i = sum_{s<=i} (C_i . B_s) * exp(L_i - L_s) * dt_s * xin_s
     (the recombined decay exp(L_i - L_s) <= 1 only for s <= i -> lower-tri mask). Inter-chunk: carry
     the real state Hst across chunks. A self-check in __main__/tests asserts chunk == recurrent
     < 1e-4 (T>=256, chunk=64, float32). The chunk form is the shipped forward().
  3. step()/init_state(): TRUE O(1) streaming — state is the constant-size per-layer (Hst [B,H,P,N] +
     conv cache [B,H,k-1,P]). step() does ONE token of the recurrence incl. the conv. step()==forward()
     < 1e-4.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import RMSNorm, SwiGLU, TFConfig


@dataclass
class Mamba2Config:
    vocab: int = 64
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_state: int = 16            # SSM state dim N (small default, the Mamba-2 reference uses 16/64/128)
    d_conv: int = 4              # short depthwise causal conv kernel k (Mamba default 4)
    chunk: int = 64
    d_ff: int = None             # default 8/3 * d_model rounded (SwiGLU keeps ~4x FLOPs) — TF-matched
    max_len: int = 1024
    tie_embeddings: bool = True

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = int(round(8 / 3 * self.d_model / 8) * 8)
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = self.d_model // self.n_heads   # P = d/H (per-head SSM input channels)


# --------------------------------- the mixer ---------------------------------------------- #
class Mamba2Block(nn.Module):
    """One Mamba-2 block: x = x + Mamba2mixer(RMSNorm1(x)); x = x + SwiGLU(RMSNorm2(x))."""

    def __init__(self, cfg: Mamba2Config):
        super().__init__()
        self.cfg = cfg
        d, H = cfg.d_model, cfg.n_heads
        self.H, self.P, self.N = H, cfg.head_dim, cfg.d_state
        self.k = cfg.d_conv
        self.norm1 = RMSNorm(d)
        # in-proj (bias-free): value path xin (d->d), gate z (d->d), per-head B and C (d->H*N each)
        self.W_x = nn.Linear(d, d, bias=False)
        self.W_z = nn.Linear(d, d, bias=False)
        self.W_B = nn.Linear(d, H * cfg.d_state, bias=False)
        self.W_C = nn.Linear(d, H * cfg.d_state, bias=False)
        # per-head scalar dt: data-dependent projection (with bias) + a learned dt_bias, then softplus
        self.W_dt = nn.Linear(d, H, bias=True)
        self.dt_bias = nn.Parameter(torch.zeros(H))
        # learned per-head NEGATIVE scalar A = -exp(A_log)  (A_log a Parameter[H])
        self.A_log = nn.Parameter(torch.zeros(H))
        # per-(head,channel) D skip
        self.D = nn.Parameter(torch.ones(H, cfg.head_dim))
        # SHORT DEPTHWISE CAUSAL CONV on xin: depthwise conv1d over the d=H*P channels, kernel k.
        # groups = channels -> per-channel (depthwise). No bias (Mamba's conv is biased; we keep it
        # bias-free here so the conv adds exactly k params/channel — documented; immaterial to faithfulness).
        self.conv = nn.Conv1d(d, d, kernel_size=self.k, groups=d, bias=True, padding=0)
        # output projection d -> d
        self.W_o = nn.Linear(d, d, bias=False)
        self.norm2 = RMSNorm(d)
        self.mlp = SwiGLU(TFConfig(d_model=d, d_ff=cfg.d_ff))
        self._reset_ssm_params()

    def _reset_ssm_params(self):
        # A_log init so A = -exp(A_log) is a modest negative (Mamba init: A in 1..H roughly). Keep it
        # simple + deterministic: A_log ~ log(uniform in [1, H]) gives a spread of decay rates per head.
        with torch.no_grad():
            a_init = torch.arange(1, self.H + 1, dtype=torch.float32)
            self.A_log.copy_(torch.log(a_init))
            # dt_bias init so softplus(dt_bias) starts at a small positive step (~0.01..0.1 range).
            self.dt_bias.uniform_(math.log(math.expm1(0.01)), math.log(math.expm1(0.1)))

    # --- projections ------------------------------------------------------------------------ #
    def _project(self, x):
        """x:[B,T,d] -> xin,z:[B,H,T,P]; B_t,C_t:[B,H,T,N]; dt:[B,H,T] (>0)."""
        B, T, _ = x.shape
        H, P, N = self.H, self.P, self.N
        xin = self.W_x(x).view(B, T, H, P).transpose(1, 2)        # [B,H,T,P]
        z = self.W_z(x).view(B, T, H, P).transpose(1, 2)          # [B,H,T,P]
        Bt = self.W_B(x).view(B, T, H, N).transpose(1, 2)         # [B,H,T,N]
        Ct = self.W_C(x).view(B, T, H, N).transpose(1, 2)         # [B,H,T,N]
        dt = F.softplus(self.W_dt(x) + self.dt_bias)              # [B,T,H] > 0
        dt = dt.transpose(1, 2)                                   # [B,H,T]
        return xin, z, Bt, Ct, dt

    def _conv_silu(self, xin):
        """Short depthwise CAUSAL conv on xin:[B,H,T,P] (over T), then silu. Causal left-pad k-1.
        Reshape heads*channels -> conv channels (depthwise) -> back. -> [B,H,T,P]."""
        B, H, T, P = xin.shape
        u = xin.transpose(1, 2).reshape(B, T, H * P).transpose(1, 2)   # [B, d=H*P, T]
        u = F.pad(u, (self.k - 1, 0))                                  # causal left-pad k-1 over T
        u = self.conv(u)                                              # [B, d, T] (depthwise)
        u = u.transpose(1, 2).view(B, T, H, P).transpose(1, 2)        # [B,H,T,P]
        return F.silu(u)

    # --- recurrent reference (ground truth) ------------------------------------------------- #
    def _recurrent(self, xc, Bt, Ct, dt):
        """Ground-truth sequential scan. xc:[B,H,T,P] (post-conv-silu SSM input); Bt,Ct:[B,H,T,N];
        dt:[B,H,T]. Per-head state Hst[P,N], Hst_0=0:
          a_t = exp(dt_t*A); Hst_t = a_t*Hst_{t-1} + dt_t*outer(xin_t,B_t); y_t = Hst_t@C_t + D*xin_t.
        -> Y:[B,H,T,P]."""
        Bsz, H, T, P = xc.shape
        N = self.N
        A = -torch.exp(self.A_log)                                   # [H] negative
        Hst = torch.zeros(Bsz, H, P, N, dtype=xc.dtype, device=xc.device)
        outs = []
        for t in range(T):
            a = torch.exp(dt[:, :, t] * A[None, :])                  # [B,H] scalar decay in (0,1)
            xin_t = xc[:, :, t]                                      # [B,H,P]
            B_t = Bt[:, :, t]                                        # [B,H,N]
            C_t = Ct[:, :, t]                                        # [B,H,N]
            dt_t = dt[:, :, t]                                       # [B,H]
            # Hst = a*Hst + dt * outer(xin, B)
            Hst = a[..., None, None] * Hst \
                + dt_t[..., None, None] * (xin_t[..., :, None] * B_t[..., None, :])   # [B,H,P,N]
            y = torch.einsum("bhpn,bhn->bhp", Hst, C_t)             # Hst @ C  -> [B,H,P]
            y = y + self.D[None] * xin_t                           # D skip (per head,channel)
            outs.append(y)
        return torch.stack(outs, dim=2)                             # [B,H,T,P]

    # --- chunk-parallel SSD (shipped forward) ----------------------------------------------- #
    def _chunk(self, xc, Bt, Ct, dt):
        """Chunk-parallel SSD. Scalar-A => a clean decayed causal attention. Per-head cumulative
        log-decay L_i = sum_{s<=i} dt_s*A (<= 0). Intra-chunk:
          y_i = sum_{s<=i} (C_i . B_s) * exp(L_i - L_s) * dt_s * xin_s
        (exp(L_i - L_s) <= 1 only for s<=i -> lower-tri mask). Inter-chunk: carry the real state Hst.
        -> Y:[B,H,T,P] (== _recurrent). float32-stable: the per-token decay is bounded (dt*A<=0) and
        the recombined ratio exp(L_i-L_s) on the causal region is <=1; the carry uses exp(L_C - L_s)
        (also <=1 for s<=C), mirroring the gated chunk-carry pattern in seq/gla.py / seq/delta.py."""
        Bsz, H, T, P = xc.shape
        N = self.N
        C = self.cfg.chunk
        A = -torch.exp(self.A_log)                                   # [H] negative
        dtA = dt * A[None, :, None]                                  # [B,H,T]  per-token log-decay <=0
        Hst = torch.zeros(Bsz, H, P, N, dtype=xc.dtype, device=xc.device)  # carried inter-chunk state
        outs = []
        for c0 in range(0, T, C):
            c1 = min(c0 + C, T)
            Cc = c1 - c0
            xq = xc[:, :, c0:c1]                                     # [B,H,Cc,P]
            Bq = Bt[:, :, c0:c1]                                     # [B,H,Cc,N]
            Cq = Ct[:, :, c0:c1]                                     # [B,H,Cc,N]
            dtc = dt[:, :, c0:c1]                                    # [B,H,Cc]
            la = dtA[:, :, c0:c1]                                    # [B,H,Cc]  per-token log-decay
            # within-chunk INCLUSIVE cumulative log-decay L_i = sum_{s<=i} dt_s*A
            L = torch.cumsum(la, dim=2)                              # [B,H,Cc]
            # --- intra-chunk causal decayed attention ---
            # scores[i,s] = (C_i . B_s) ; decay[i,s] = exp(L_i - L_s) for s<=i ; weight by dt_s.
            scores = torch.einsum("bhin,bhsn->bhis", Cq, Bq)        # [B,H,Cc,Cc]  C_i . B_s
            decay = torch.exp(L[:, :, :, None] - L[:, :, None, :])  # [B,H,Cc,Cc] exp(L_i - L_s)
            tri = torch.tril(torch.ones(Cc, Cc, dtype=torch.bool, device=xc.device))  # s<=i incl diag
            w = scores * decay * dtc[:, :, None, :]                 # weight source s by dt_s
            w = w.masked_fill(~tri, 0.0)
            y_intra = torch.einsum("bhis,bhsp->bhip", w, xq)        # sum_s w[i,s] * xin_s -> [B,H,Cc,P]
            # --- inter-chunk: read carried state Hst (from end of previous chunks) ---
            # y_inter_i = exp(L_i) * (Hst @ C_i)   (L_i = within-chunk cumulative decay applied to carry)
            y_carry = torch.einsum("bhpn,bhin->bhip", Hst, Cq)      # (Hst @ C_i) -> [B,H,Cc,P]
            y_inter = torch.exp(L)[..., None] * y_carry             # scale by exp(L_i)
            y = y_intra + y_inter + self.D[None, :, None, :] * xq   # D skip per (head,channel)
            outs.append(y)
            # --- carry state to next chunk ---
            # Hst_new = exp(L_C) * Hst + sum_s exp(L_C - L_s) * dt_s * outer(xin_s, B_s)
            LC = L[:, :, -1:]                                        # [B,H,1] chunk-final cumdecay
            gC = torch.exp(LC)[..., None]                           # [B,H,1,1]
            ratio = torch.exp(LC - L) * dtc                         # [B,H,Cc]  exp(L_C-L_s)*dt_s  (<=*dt)
            Hst = gC * Hst + torch.einsum("bhs,bhsp,bhsn->bhpn", ratio, xq, Bq)
        return torch.cat(outs, dim=2)                               # [B,H,T,P]

    # --- mixer ------------------------------------------------------------------------------ #
    def _mix(self, x, use_chunk=True):
        """The Mamba-2 mixer on a normed input x:[B,T,d] -> [B,T,d]. use_chunk picks chunk vs recurrent
        (the two MUST agree < 1e-4 — gated by tests). Applies conv+silu, the SSD core, then the z gate."""
        B, T, d = x.shape
        xin, z, Bt, Ct, dt = self._project(x)
        xc = self._conv_silu(xin)                                   # short causal conv + silu
        core = self._chunk if use_chunk else self._recurrent
        y = core(xc, Bt, Ct, dt)                                    # [B,H,T,P]
        y = y * F.silu(z)                                           # gated output (z gate)
        y = y.transpose(1, 2).reshape(B, T, d)                     # concat heads -> [B,T,d]
        return self.W_o(y)                                          # d -> d

    def forward(self, h):
        h = h + self._mix(self.norm1(h), use_chunk=True)
        h = h + self.mlp(self.norm2(h))
        return h

    # ---- TRUE O(1)-per-step streaming inference ---- #
    @torch.no_grad()
    def step(self, h_t, state):
        """h_t:[B,1,d]; state = (Hst:[B,H,P,N], conv_cache:[B,H,k-1,P]) — constant size. Returns
        h_out, new_state. O(1) in sequence length (constant memory)."""
        B = h_t.shape[0]
        Hst, conv_cache = state
        x = self.norm1(h_t)                                        # [B,1,d]
        xin, z, Bt, Ct, dt = self._project(x)                     # xin,z:[B,H,1,P]; Bt,Ct:[B,H,1,N]; dt:[B,H,1]
        H, P, N = self.H, self.P, self.N
        # --- one step of the depthwise causal conv: window = [conv_cache (k-1) | xin_now] ---
        win = torch.cat([conv_cache, xin], dim=2)                 # [B,H,k,P]
        wd = win.transpose(1, 2).reshape(B, self.k, H * P).permute(0, 2, 1)  # [B, d=H*P, k]
        cw = self.conv.weight.view(H * P, self.k)                 # depthwise weight [d,k]
        conv_out = (wd * cw[None]).sum(-1) + self.conv.bias       # [B, d]
        xc = F.silu(conv_out).view(B, H, P).unsqueeze(2)         # [B,H,1,P]
        new_conv_cache = win[:, :, 1:]                            # drop oldest -> keep last k-1 [B,H,k-1,P]
        # --- one token of the recurrence ---
        A = -torch.exp(self.A_log)                                # [H]
        a = torch.exp(dt[:, :, 0] * A[None, :])                  # [B,H]
        xin_t = xc[:, :, 0]                                       # [B,H,P]
        B_t = Bt[:, :, 0]                                         # [B,H,N]
        C_t = Ct[:, :, 0]                                         # [B,H,N]
        dt_t = dt[:, :, 0]                                        # [B,H]
        Hst = a[..., None, None] * Hst \
            + dt_t[..., None, None] * (xin_t[..., :, None] * B_t[..., None, :])   # [B,H,P,N]
        y = torch.einsum("bhpn,bhn->bhp", Hst, C_t) + self.D[None] * xin_t       # [B,H,P]
        y = y * F.silu(z[:, :, 0])                               # z gate
        y = y.reshape(B, 1, self.cfg.d_model)                    # concat heads -> [B,1,d]
        h = h_t + self.W_o(y)
        h = h + self.mlp(self.norm2(h))
        return h, (Hst, new_conv_cache)


class Mamba2LM(nn.Module):
    """Mamba-2 language model. Mirrors transformer.Transformer: token embedding, n_layers Mamba-2
    blocks, final RMSNorm, tied Linear head. NO positional embedding (the SSM is implicitly positional
    via the causal scan + per-step decay + the causal conv)."""

    def __init__(self, cfg: Mamba2Config):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d_model)
        self.blocks = nn.ModuleList([Mamba2Block(cfg) for _ in range(cfg.n_layers)])
        self.nf = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.tok.weight
        # init weights normal_(0.02), THEN restore the SSM params (A_log/dt_bias/D) the blocks set in
        # their constructor — apply(_init) would otherwise clobber them via the generic param paths.
        self.apply(self._init)
        for blk in self.blocks:
            blk._reset_ssm_params()

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        h = self.tok(idx)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.nf(h))

    @torch.no_grad()
    def init_state(self, batch, device):
        """Per-layer constant-size state: (SSM state Hst[B,H,P,N], conv cache[B,H,k-1,P]).
        CONSTANT size in T (NOT a growing KV cache)."""
        H, P, N, k = self.cfg.n_heads, self.cfg.head_dim, self.cfg.d_state, self.cfg.d_conv
        return [(torch.zeros(batch, H, P, N, device=device),
                 torch.zeros(batch, H, k - 1, P, device=device))
                for _ in self.blocks]

    @torch.no_grad()
    def step(self, tok, state):
        """tok:[B,1] -> (logits[B,1,V], new_state). O(1) in sequence length (constant memory)."""
        h = self.tok(tok)
        new = []
        for blk, st in zip(self.blocks, state):
            h, st2 = blk.step(h, st)
            new.append(st2)
        return self.head(self.nf(h)), new


def mamba2_factory(d_model=128, n_layers=2, n_heads=4, **kw):
    """(lambda V, T: Mamba2LM) factory with max_len = T + 8 (mirrors gla_factory / prizma_seq_factory)."""
    def f(vocab, max_len):
        return Mamba2LM(Mamba2Config(vocab=vocab, d_model=d_model, n_layers=n_layers,
                                     n_heads=n_heads, max_len=max_len + 8, **kw))
    return f


if __name__ == "__main__":
    from .common import param_count, get_device
    from .transformer import Transformer, TFConfig
    dev = get_device()

    # CHUNK == RECURRENT (production regime: T=256, chunk=64, float32)
    torch.manual_seed(0)
    cfg = Mamba2Config(vocab=64, d_model=64, n_layers=1, n_heads=4, chunk=64)
    blk = Mamba2Block(cfg).to(dev)
    blk.train(False)
    h = torch.randn(2, 256, 64, device=dev)
    with torch.no_grad():
        oc = blk._mix(blk.norm1(h), use_chunk=True)
        orf = blk._mix(blk.norm1(h), use_chunk=False)
    dcr = (oc - orf).abs().max().item()
    print(f"[chunk==recurrent T=256 C=64] max|d|={dcr:.2e} {'OK' if dcr < 1e-4 else 'MISMATCH'}")

    # O(1) GUARD: streaming step() == forward() < 1e-4
    cfg2 = Mamba2Config(vocab=64, d_model=64, n_layers=2, n_heads=4)
    m = Mamba2LM(cfg2).to(dev)
    m.train(False)
    x = torch.randint(0, 64, (2, 48), device=dev)
    with torch.no_grad():
        y = m(x)
        st = m.init_state(2, dev)
        outs = []
        for t in range(x.shape[1]):
            lg, st = m.step(x[:, t:t + 1], st)
            outs.append(lg)
        yo = torch.cat(outs, dim=1)
    d1 = (y - yo).abs().max().item()
    print(f"[O(1) step==forward] forward {tuple(y.shape)} max|d|={d1:.2e} "
          f"{'OK' if d1 < 1e-4 else 'MISMATCH'}")

    # PARAM-vs-TF print (same d/L/H/vocab)
    V, d, L, H = 256, 128, 3, 4
    tf = Transformer(TFConfig(vocab=V, d_model=d, n_layers=L, n_heads=H, rope=True))
    mamba = Mamba2LM(Mamba2Config(vocab=V, d_model=d, n_layers=L, n_heads=H))
    p_tf, p_m = param_count(tf), param_count(mamba)
    print(f"[param-vs-TF d{d}L{L}H{H} V{V}] TF={p_tf} Mamba2={p_m} "
          f"spread={(p_m - p_tf) / p_tf:+.2%}")
