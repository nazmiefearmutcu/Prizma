"""
A FAITHFUL Gated Linear Attention (GLA) baseline — Yang, Wang, Shen, Panda, Kim,
"Gated Linear Attention Transformers with Hardware-Efficient Training" (ICML 2024).

This is THE canonical modern gated-linear-attention / SSM SOTA baseline. It is built as a
NON-strawman: the LOW-RANK data-dependent per-key-channel forget gate AND the full GLA output
head (per-head RMSNorm + swish output gate + output projection) are both included — omitting
either silently weakens the baseline and would corrupt the Pareto comparison.

CANONICAL GLA DESIGN (matches fla-org/flash-linear-attention defaults; documented choices).
  Per layer, input x in R^{B x T x d} (d=d_model), H heads.
    expand_k = 0.5, expand_v = 1.0  =>  key_dim = d/2, value_dim = d  (the published defaults).
    head_k = (d/2)/H,  head_v = d/H.  The carried state S is RECTANGULAR: [head_k x head_v].
  Projections (bias-free Linear, reshaped to [B,H,T,head_*]):
      q = W_q x  (d -> d/2),   k = W_k x  (d -> d/2),   v = W_v x  (d -> d).
  Data-dependent per-KEY-channel forget gate, LOW-RANK (gate_low_rank_dim=16, the GLA default):
      gk = logsigmoid( W_g2( W_g1 x ) ) / gate_logit_normalizer            # W_g1: d->16, W_g2: 16->d/2
      alpha_t = exp(gk_t)   in (0,1)^{head_k}     (gate_logit_normalizer=16, the published default)
  State recurrence, S_t in R^{head_k x head_v}, S_0 = 0:
      S_t[i,:] = alpha_t[i] * S_{t-1}[i,:] + k_t[i] * v_t     (row i = key channel i, decays by alpha_t[i])
      o_t[j]   = sum_i q_t[i] * S_t[i,j]                       # o_t = q_t^T S_t  (POST-write read; GLA std)
  Output head (faithful GLA — NOT omitted):
      o = rmsnorm_perhead(o_t) * silu(W_r x_t)                 # per-head RMSNorm + swish output gate
      out = W_o( concat_heads(o) )                             # W_r: d->d (output gate), W_o: d->d

  DOCUMENTED CHOICES.
   * Forget gate is the canonical LOW-RANK form W_g = W_g2 @ W_g1 (gate_low_rank_dim=16) with a
     gate_logit_normalizer of 16 and a logsigmoid activation — the published default. This keeps
     the param-match to the Transformer tight (the full-rank sigmoid(W_g x) alternative inflates
     params ~16% and is NOT the canonical default).
   * expand_k=0.5 / expand_v=1.0 are the GLA defaults; key/query head-dim is half the value
     head-dim and the state is rectangular [head_k x head_v]. head_v MUST be even (production
     regime); head_k = head_v/2 follows.
   * The output read uses the POST-write state S_t (q_t^T S_t), the published GLA convention —
     this differs from Prizma-Seq's PRE-write delta read on purpose (GLA is an ADDITIVE, gated
     linear-attention update, not a delta/erase-and-write rule).

THREE FORMS + the O(1) rigor bar (the project's non-negotiable gate).
  1. RECURRENT reference (_recurrent): a sequential scan over T implementing the recurrence
     above exactly. Vectorized over [B,H,head_*]; python loop over t. Ground truth; used by step().
  2. CHUNK-PARALLEL forward (_chunk): per-key-channel cumulative gate products in LOG space
     (mirrors seq/delta.py's gated chunked path: all gate RATIOS formed as exp(log-diff) so they
     stay <= 1 on the causal region -> float32-stable). Used for parallel training.
     A self-check in __main__/tests asserts chunk == recurrent < 1e-4 (T>=256, chunk=64).
  3. step()/init_state(): TRUE O(1) streaming — state is the constant-size per-layer
     S [B,H,head_k,head_v]. step() does ONE token of the recurrence. step()==forward() < 1e-4.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import RMSNorm, SwiGLU, TFConfig


@dataclass
class GLAConfig:
    vocab: int = 64
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    chunk: int = 64
    expand_k: float = 0.5            # GLA default: key/query dim = expand_k * d_model
    expand_v: float = 1.0            # GLA default: value dim = expand_v * d_model
    gate_low_rank_dim: int = 16      # GLA default: low-rank forget gate W_g = W_g2 @ W_g1
    gate_logit_normalizer: float = 16.0   # GLA default: gk = logsigmoid(.) / normalizer
    d_ff: int = None                 # default 8/3 * d_model rounded (SwiGLU keeps ~4x FLOPs) — TF-matched
    max_len: int = 1024
    tie_embeddings: bool = True

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = int(round(8 / 3 * self.d_model / 8) * 8)
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.key_dim = int(self.d_model * self.expand_k)
        self.value_dim = int(self.d_model * self.expand_v)
        assert self.key_dim % self.n_heads == 0, "key_dim must be divisible by n_heads"
        assert self.value_dim % self.n_heads == 0, "value_dim must be divisible by n_heads"
        self.head_k = self.key_dim // self.n_heads
        self.head_v = self.value_dim // self.n_heads
        assert self.head_v % 2 == 0, "head_v (value head-dim) must be even (production-regime invariant)"


# --------------------------------- the mixer ---------------------------------------------- #
class GLABlock(nn.Module):
    """One GLA block: x = x + GLAmixer(RMSNorm1(x)); x = x + SwiGLU(RMSNorm2(x))."""

    def __init__(self, cfg: GLAConfig):
        super().__init__()
        self.cfg = cfg
        d, H = cfg.d_model, cfg.n_heads
        self.H, self.hk, self.hv = H, cfg.head_k, cfg.head_v
        self.norm1 = RMSNorm(d)
        self.W_q = nn.Linear(d, cfg.key_dim, bias=False)
        self.W_k = nn.Linear(d, cfg.key_dim, bias=False)
        self.W_v = nn.Linear(d, cfg.value_dim, bias=False)
        # LOW-RANK per-key-channel forget gate W_g = W_g2 @ W_g1 (the canonical GLA gate).
        self.W_g1 = nn.Linear(d, cfg.gate_low_rank_dim, bias=False)
        self.W_g2 = nn.Linear(cfg.gate_low_rank_dim, cfg.key_dim, bias=True)
        self.W_r = nn.Linear(d, cfg.value_dim, bias=True)     # swish output gate (GLA output head)
        self.W_o = nn.Linear(cfg.value_dim, d, bias=False)    # output projection value_dim -> d
        self.gnorm = RMSNorm(cfg.head_v)                      # per-head RMSNorm on the read o_t
        self.norm2 = RMSNorm(d)
        self.mlp = SwiGLU(TFConfig(d_model=d, d_ff=cfg.d_ff))

    def _project(self, x):
        """x:[B,T,d] -> q,k:[B,H,T,head_k]; v:[B,H,T,head_v]; log_alpha:[B,H,T,head_k]; r:[B,T,value_dim]."""
        B, T, _ = x.shape
        q = self.W_q(x).view(B, T, self.H, self.hk).transpose(1, 2)   # [B,H,T,head_k]
        k = self.W_k(x).view(B, T, self.H, self.hk).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.H, self.hv).transpose(1, 2)   # [B,H,T,head_v]
        # Canonical GLA forget gate: gk = logsigmoid(W_g2(W_g1 x)) / gate_logit_normalizer.
        # log_alpha = gk (<=0); alpha = exp(gk) in (0,1). logsigmoid is the stable log of sigmoid.
        gk = F.logsigmoid(self.W_g2(self.W_g1(x))) / self.cfg.gate_logit_normalizer
        log_alpha = gk.view(B, T, self.H, self.hk).transpose(1, 2)    # [B,H,T,head_k]
        return q, k, v, log_alpha

    def _recurrent(self, q, k, v, log_alpha):
        """Ground-truth sequential scan. q,k,log_alpha:[B,H,T,head_k]; v:[B,H,T,head_v].
        S_t[i,:] = alpha_t[i]*S_{t-1}[i,:] + k_t[i]*v_t ; o_t = q_t^T S_t (post-write). -> O:[B,H,T,head_v]."""
        B, H, T, hk = q.shape
        hv = v.shape[-1]
        S = torch.zeros(B, H, hk, hv, dtype=q.dtype, device=q.device)   # [B,H,head_k,head_v]
        alpha = torch.exp(log_alpha)
        outs = []
        for t in range(T):
            a = alpha[:, :, t]                                   # [B,H,head_k] per-key-channel decay
            kt, vt = k[:, :, t], v[:, :, t]
            # S[i,:] = a[i]*S[i,:] + k[i]*v   (outer(k,v) added per key-row)
            S = a[..., None] * S + kt[..., None] * vt[..., None, :]
            o = torch.einsum("bhi,bhij->bhj", q[:, :, t], S)    # POST-write read q_t^T S_t -> [B,H,head_v]
            outs.append(o)
        return torch.stack(outs, dim=2)                          # [B,H,T,head_v]

    def _chunk(self, q, k, v, log_alpha):
        """Chunk-parallel GLA. Per-key-channel cumulative gates in LOG space. The RECOMBINED causal
        ratio exp(B_i - B_s) is <=1 only for s<=i (hence the lower-triangular mask). The DECOMPOSED
        factors q_i*exp(B_i) and k_s*exp(-B_s) are NOT individually bounded by 1 — exp(-B_s) grows as
        the cumulative log-gate B_s goes negative. Numerical stability therefore rests on float32's
        dynamic range PLUS `gate_logit_normalizer` bounding how negative each per-token log-gate can
        get (not on the factors being <=1). Carries a real state S across chunks (true O(T) training).
        -> O:[B,H,T,head_v] (== _recurrent)."""
        B, H, T, hk = q.shape
        C = self.cfg.chunk
        hv = v.shape[-1]
        S = torch.zeros(B, H, hk, hv, dtype=q.dtype, device=q.device)   # carried inter-chunk state
        outs = []
        for c0 in range(0, T, C):
            c1 = min(c0 + C, T)
            qc = q[:, :, c0:c1]                                  # [B,H,Cc,head_k]
            kc = k[:, :, c0:c1]
            vc = v[:, :, c0:c1]                                  # [B,H,Cc,head_v]
            la = log_alpha[:, :, c0:c1]                          # [B,H,Cc,head_k]
            # within-chunk INCLUSIVE cumulative log-gate b_i[ch] = sum_{s<=i} log alpha_s[ch]
            cb = torch.cumsum(la, dim=2)                         # [B,H,Cc,head_k]  (B_i)
            # --- intra-chunk causal attention ---
            # A[i,s] = sum_ch q_i[ch] * exp(B_i[ch] - B_s[ch]) * k_s[ch]   for s <= i.
            # Decompose the recombined ratio: q~_i = q_i*exp(B_i), k~_s = k_s*exp(-B_s) so
            # q~_i . k~_s = sum_ch q_i k_s exp(B_i - B_s). The PRODUCT exp(B_i - B_s) <= 1 only for
            # s <= i, so we MASK to the lower triangle (incl diag). The individual k~_s = k_s*exp(-B_s)
            # is NOT <=1 (it grows as B_s goes negative) — stability rests on float32 range +
            # gate_logit_normalizer (see the method docstring), not on the factors being bounded.
            q_dec = qc * torch.exp(cb)                           # [B,H,Cc,head_k]   q_i * b_i
            k_dec = kc * torch.exp(-cb)                          # [B,H,Cc,head_k]   k_s / b_s
            scores = torch.matmul(q_dec, k_dec.transpose(-1, -2))   # [B,H,Cc,Cc]  A[i,s]
            Cc = c1 - c0
            causal = torch.tril(torch.ones(Cc, Cc, dtype=torch.bool, device=q.device))  # s<=i incl diag
            scores = scores.masked_fill(~causal, 0.0)
            o_intra = torch.matmul(scores, vc)                  # [B,H,Cc,head_v]
            # --- inter-chunk: read carried state S (from end of previous chunks) ---
            # o_inter_i = sum_ch q_i[ch] * b_i[ch] * S[ch,:]   (b_i = within-chunk cumulative gate)
            o_inter = torch.einsum("bhic,bhcv->bhiv", q_dec, S)  # [B,H,Cc,head_v]
            outs.append(o_intra + o_inter)
            # --- carry state to next chunk ---
            # S_new[ch,:] = exp(B_C[ch]) * S[ch,:] + sum_s exp(B_C[ch]-B_s[ch]) k_s[ch] v_s[:]
            #   (exp(B_C - B_s) <= 1 for s <= C -> stable, mirrors delta.py's gamma_C/gamma_i carry).
            cbC = cb[:, :, -1:]                                  # [B,H,1,head_k]  B_C (chunk-final cumgate)
            S = torch.exp(cbC).transpose(2, 3) * S \
                + torch.einsum("bhsc,bhsv->bhcv", kc * torch.exp(cbC - cb), vc)
        return torch.cat(outs, dim=2)                            # [B,H,T,head_v]

    def _mix(self, x, use_chunk=True):
        """The GLA mixer on a normed input x:[B,T,d] -> [B,T,d]. use_chunk picks chunk vs recurrent
        (the two MUST agree < 1e-4 — gated by tests). Applies the full GLA output head."""
        B, T, d = x.shape
        q, k, v, log_alpha = self._project(x)
        core = self._chunk if use_chunk else self._recurrent
        o = core(q, k, v, log_alpha)                            # [B,H,T,head_v]  post-write read
        o = self.gnorm(o)                                       # per-head RMSNorm (GLA output head)
        o = o.transpose(1, 2).reshape(B, T, self.cfg.value_dim)   # concat heads -> [B,T,value_dim]
        o = o * F.silu(self.W_r(x))                             # swish output gate
        return self.W_o(o)                                      # value_dim -> d

    def forward(self, h):
        h = h + self._mix(self.norm1(h), use_chunk=True)
        h = h + self.mlp(self.norm2(h))
        return h

    # ---- TRUE O(1)-per-step streaming inference ---- #
    @torch.no_grad()
    def step(self, h_t, state):
        """h_t:[B,1,d]; state = S:[B,H,head_k,head_v] (constant size). Returns o_t, new_state. O(1)."""
        B = h_t.shape[0]
        S = state
        x = self.norm1(h_t)                                     # [B,1,d]
        q, k, v, log_alpha = self._project(x)                  # q,k:[B,H,1,head_k] v:[B,H,1,head_v]
        q1 = q[:, :, 0]; k1 = k[:, :, 0]; v1 = v[:, :, 0]
        a1 = torch.exp(log_alpha[:, :, 0])                     # [B,H,head_k] per-key-channel decay
        S = a1[..., None] * S + k1[..., None] * v1[..., None, :]   # one token of the recurrence
        o = torch.einsum("bhi,bhij->bhj", q1, S)              # POST-write read [B,H,head_v]
        o = self.gnorm(o)                                      # per-head RMSNorm, mirrors forward
        o = o.reshape(B, 1, self.cfg.value_dim)              # [B,1,value_dim]
        o = o * F.silu(self.W_r(x))                           # swish output gate
        h = h_t + self.W_o(o)
        h = h + self.mlp(self.norm2(h))
        return h, S


class GLALM(nn.Module):
    """GLA language model. Mirrors transformer.Transformer: token embedding, n_layers GLA blocks,
    final RMSNorm, tied Linear head. NO positional embedding (linear attention is implicitly
    positional via the per-channel decay + causal write-order)."""

    def __init__(self, cfg: GLAConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d_model)
        self.blocks = nn.ModuleList([GLABlock(cfg) for _ in range(cfg.n_layers)])
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
        h = self.tok(idx)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.nf(h))

    @torch.no_grad()
    def init_state(self, batch, device):
        """Per-layer constant-size GLA state S:[B,H,head_k,head_v] (NOT a growing KV cache)."""
        return [torch.zeros(batch, self.cfg.n_heads, self.cfg.head_k, self.cfg.head_v, device=device)
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


def gla_factory(d_model=128, n_layers=2, n_heads=4, **kw):
    """(lambda V, T: GLALM) factory with max_len = T + 8 (mirrors prizma_seq_factory)."""
    def f(vocab, max_len):
        return GLALM(GLAConfig(vocab=vocab, d_model=d_model, n_layers=n_layers,
                               n_heads=n_heads, max_len=max_len + 8, **kw))
    return f


if __name__ == "__main__":
    from .common import param_count, get_device
    from .transformer import Transformer, TFConfig
    dev = get_device()

    # CHUNK == RECURRENT (production regime: T=256, chunk=64, head_v even, float32)
    torch.manual_seed(0)
    cfg = GLAConfig(vocab=64, d_model=64, n_layers=1, n_heads=4, chunk=64)
    blk = GLABlock(cfg).to(dev)
    blk.train(False)
    h = torch.randn(2, 256, 64, device=dev)
    with torch.no_grad():
        oc = blk._mix(blk.norm1(h), use_chunk=True)
        orf = blk._mix(blk.norm1(h), use_chunk=False)
    dcr = (oc - orf).abs().max().item()
    print(f"[chunk==recurrent T=256 C=64] max|d|={dcr:.2e} {'OK' if dcr < 1e-4 else 'MISMATCH'}")

    # O(1) GUARD: streaming step() == forward() < 1e-4
    cfg2 = GLAConfig(vocab=64, d_model=64, n_layers=2, n_heads=4)
    m = GLALM(cfg2).to(dev)
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
    gla = GLALM(GLAConfig(vocab=V, d_model=d, n_layers=L, n_heads=H))
    p_tf, p_gla = param_count(tf), param_count(gla)
    print(f"[param-vs-TF d{d}L{L}H{H} V{V}] TF={p_tf} GLA={p_gla} "
          f"spread={(p_gla - p_tf) / p_tf:+.2%}")
