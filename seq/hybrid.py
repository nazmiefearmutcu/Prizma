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

from .transformer import Block, RMSNorm, TFConfig
from .prizma_seq import PrizmaSeqBlock, PrizmaSeqConfig


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


class HybridSeqLM(nn.Module):
    """Mostly-Prizma + ~1 attention layer baseline (param-matched to the Transformer).

    Args:
      prizma_cfg: the PrizmaSeqConfig used for every NON-attention layer (carries d_model,
                  n_layers, n_heads, max_len, vocab + all Prizma lever knobs).
      n_attn:     number of attention layers (auto-placed, centered) when attn_layers is None.
      attn_layers: explicit attention layer index/indices (int or iterable); overrides n_attn.
      tf_rope:    RoPE on the attention layers (matches the TF baseline default True).
    """

    def __init__(self, prizma_cfg: PrizmaSeqConfig, n_attn: int = 1, attn_layers=None,
                 tf_rope: bool = True):
        super().__init__()
        self.cfg = prizma_cfg
        n_layers = prizma_cfg.n_layers
        self.attn_layers = _resolve_attn_layers(n_layers, n_attn, attn_layers)

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
            if i in self.attn_layers:
                blocks.append(Block(tf_cfg))
            else:
                blocks.append(PrizmaSeqBlock(prizma_cfg))
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
            h = blk(h)                                  # both block types: forward(h) -> [B,T,d]
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
            else:
                S = torch.zeros(batch, cfg.n_heads, cfg.d_h, state_k_dim, device=device)
                rk = torch.zeros(batch, cfg.n_heads, 0, cfg.d_h, device=device)
                rv = torch.zeros(batch, cfg.n_heads, 0, cfg.d_h, device=device)
                cring = torch.zeros(batch, kc1, cfg.d_model, device=device)
                st.append((S, rk, rv, cring, 0))
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
                if not isinstance(blk, Block) and st is not None:
                    p = st[4]
                    break
            else:
                for blk, st in zip(self.blocks, state):
                    if isinstance(blk, Block) and st is not None:
                        p = st[0].shape[2]
                        break
            h = h + self.pos(torch.tensor([p], device=tok.device))[None]
        new = []
        for blk, st in zip(self.blocks, state):
            h, st2 = blk.step(h, st)                     # Block.step + PrizmaSeqBlock.step share (h_t, st)->(h, st)
            new.append(st2)
        return self.head(self.nf(h)), new


def hybrid_factory(d, L, H, n_attn=1, attn_layers=None, tf_rope=True, **prizma_kw):
    """Factory mirroring ps_factory: hybrid_factory(d, L, H, ...) -> (lambda V, T: HybridSeqLM(...)).

    Drops into run_cell / recall-gate / gpu_bench as a THIRD arm exactly like ps_factory.
    `prizma_kw` are forwarded to PrizmaSeqConfig (feat_map, gated, window, etc.).
    """
    def f(vocab, max_len):
        cfg = PrizmaSeqConfig(vocab=vocab, d_model=d, n_layers=L, n_heads=H,
                              max_len=max_len + 8, **prizma_kw)
        return HybridSeqLM(cfg, n_attn=n_attn, attn_layers=attn_layers, tf_rope=tf_rope)
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
    m = hybrid_factory(128, 4, 4)(64, 128)
    x = _t.randint(0, 64, (2, 48))
    types = [type(b).__name__ for b in m.blocks]
    print("blocks:", types, "attn_layers:", m.attn_layers)
    print("logits", tuple(m(x).shape))
    _print_param_match()
