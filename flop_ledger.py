"""SKEPTIC-C FLOP / throughput ledger — PARAMETERIZED over scale.

Analytical per-token forward FLOPs for the matched-param TF vs Prizma-quad2 at seq T,
per scale (d_model, H, L), plus the lever's marginal cost and the window-head share.

Counts MACs*2 = FLOPs. Matmul [m,k]x[k,n] = 2*m*k*n FLOPs. Counted per SEQUENCE (all T
tokens) for the training forward, then /T for per-token. Attention is counted at its
TRUE causal O(T^2/2) training cost (honest, no over-count). Prizma's chunked_delta is
counted at its true chunk-matmul cost; the window head both as-coded (full SDPA) and at
its ideal banded cost.

Run: python3 flop_ledger.py            # prints both the legacy d64L2H2 and headline d128L4H4
"""
import torch
from seq.transformer import Transformer, TFConfig
from seq.prizma_seq import PrizmaSeqLM, PrizmaSeqConfig
from seq.common import param_count

V = 512
T = 384            # MixedMQAR(max_pairs=128): seq_len = 2*128 + 128 = 384


def dff_of(d):
    return int(round(8 / 3 * d / 8) * 8)


def mm(m, k, n):   # FLOPs of [m,k]@[k,n]
    return 2 * m * k * n


def tf_layer_flops_causal(T, d, H, dff):
    """Honest causal count: only lower-triangular pairs ~ T*(T+1)/2."""
    dh = d // H
    pairs = T * (T + 1) / 2
    return {
        "qkv_proj": mm(T, d, 3 * d),
        "attn_scores": H * 2 * pairs * dh,   # sum over causal (i,j) of 2*dh
        "attn_AV": H * 2 * pairs * dh,
        "attn_out": mm(T, d, d),
        "mlp": mm(T, d, dff) + mm(T, d, dff) + mm(T, dff, d),  # SwiGLU w1,w2,wo
    }


def prizma_layer_flops(T, d, H, dff, d_phi, C=64, w=16, kc=4):
    dh = d // H
    f = {}
    f["conv"] = 2 * T * d * kc                         # depthwise causal conv k=4
    f["qkv_proj"] = mm(T, d, 3 * d)
    f["beta"] = mm(T, d, H)
    n2 = d_phi - dh
    f["phi_qk"] = H * T * (n2 * 1 + d_phi * 3) * 2     # quadratic monomials + l2 norm (generous)
    nC = T // C
    per_chunk = (mm(C, d_phi, C)          # KK
                 + mm(C, d_phi, dh)        # KS0
                 + 2 * C * C * dh          # triangular solve (approx)
                 + mm(C, d_phi, dh)        # O_inter
                 + mm(C, d_phi, C)         # QK
                 + 2 * C * C * dh          # O_intra
                 + mm(dh, C, d_phi))       # S update
    f["delta_state"] = H * nC * per_chunk
    f["window_full_TT"] = H * (mm(T, dh, T) + mm(T, T, dh))   # as-coded full SDPA + mask
    band_pairs = T * w - w * (w - 1) / 2
    f["window_band_ideal"] = H * 2 * band_pairs * dh * 2      # band-only scores+AV
    f["out_proj"] = mm(T, d, d)
    f["mlp"] = mm(T, d, dff) + mm(T, d, dff) + mm(T, dff, d)
    return f


def total(fd, exclude=()):
    return sum(v for k, v in fd.items() if k not in exclude)


def ledger(d, H, L, d_phi, T=T, label="", show_components=True):
    dff = dff_of(d)
    dh = d // H
    head_flops = mm(T, d, V)
    print(f"\n############### SCALE {label}: d={d}, H={H}, L={L}, d_h={dh}, d_ff={dff}, T={T} ###############")

    tfl = tf_layer_flops_causal(T, d, H, dff)
    tf_total = L * total(tfl) + head_flops
    if show_components:
        print("TRANSFORMER per-layer (causal-honest):")
        for k, v in tfl.items():
            print(f"   {k:<14} {v/1e6:8.2f} MFLOP")
    print(f"   TF: layer {total(tfl)/1e6:8.2f} MFLOP  x{L}+head = {tf_total/1e6:8.2f} MFLOP/seq  "
          f"= {tf_total/T/1e3:7.1f} kFLOP/token")

    pf = prizma_layer_flops(T, d, H, dff, d_phi=d_phi)
    ps_ascoded = total(pf, exclude=("window_band_ideal",))
    ps_ideal = total(pf, exclude=("window_full_TT",))
    ps_total_ascoded = L * ps_ascoded + head_flops
    ps_total_ideal = L * ps_ideal + head_flops
    if show_components:
        print(f"Prizma-quad2 (d_phi={d_phi}) per-layer:")
        for k, v in pf.items():
            print(f"   {k:<18} {v/1e6:8.2f} MFLOP")
    print(f"   Prizma as-coded: x{L}+head = {ps_total_ascoded/1e6:8.2f} MFLOP/seq = "
          f"{ps_total_ascoded/T/1e3:7.1f} kFLOP/token")
    print(f"   Prizma ideal   : x{L}+head = {ps_total_ideal/1e6:8.2f} MFLOP/seq = "
          f"{ps_total_ideal/T/1e3:7.1f} kFLOP/token")
    print(f"   RATIO Prizma/TF forward:  as-coded {ps_total_ascoded/tf_total:4.2f}x   "
          f"ideal {ps_total_ideal/tf_total:4.2f}x")

    # lever marginal cost
    pf_none = prizma_layer_flops(T, d, H, dff, d_phi=dh)
    print(f"   lever (d_phi {d_phi} vs {dh}): delta-state x{pf['delta_state']/pf_none['delta_state']:.1f}  "
          f"| window_full share = {100*L*pf['window_full_TT']/ps_total_ascoded:4.1f}% as-coded, "
          f"band = {100*L*pf['window_band_ideal']/ps_total_ideal:4.1f}% ideal")
    return {"tf_per_tok": tf_total / T, "ps_ascoded_per_tok": ps_total_ascoded / T,
            "ps_ideal_per_tok": ps_total_ideal / T,
            "ratio_ascoded": ps_total_ascoded / tf_total, "ratio_ideal": ps_total_ideal / tf_total}


def flop_matched_tf_search(d, H, L, d_phi, target_ratio, T=T):
    """Find a bigger TF (grow d_ff, then layers) whose forward FLOPs >= target_ratio x base TF.
    Returns candidates so the headline FLOP-matched arm can be sized from MEASURED FLOPs."""
    dff = dff_of(d)
    head = mm(T, d, V)
    base_tf = L * total(tf_layer_flops_causal(T, d, H, dff)) + head
    pf = prizma_layer_flops(T, d, H, dff, d_phi=d_phi)
    prizma_ascoded = L * total(pf, exclude=("window_band_ideal",)) + head
    print(f"\n=== FLOP-matched TF search @ d={d} H={H} L={L} (match Prizma-quad2 as-coded "
          f"{prizma_ascoded/T/1e3:.1f} kFLOP/tok) ===")
    # option A: deeper TF (more layers) at same width
    for L2 in range(L, 4 * L + 1):
        tot = L2 * total(tf_layer_flops_causal(T, d, H, dff)) + head
        if tot >= prizma_ascoded:
            print(f"   deeper: TF d{d}L{L2}H{H}  -> {tot/T/1e3:6.1f} kFLOP/tok  "
                  f"({tot/base_tf:.2f}x base, {tot/prizma_ascoded:.2f}x Prizma)")
            break
    # option B: wider d_model (keep L,H ratio) — FLOP ~ d^2
    for dm in range(d, 3 * d + 1, 16):
        if dm % H:
            continue
        tot = L * total(tf_layer_flops_causal(T, dm, H, dff_of(dm))) + mm(T, dm, V)
        if tot >= prizma_ascoded:
            print(f"   wider : TF d{dm}L{L}H{H}  -> {tot/T/1e3:6.1f} kFLOP/tok  "
                  f"({tot/base_tf:.2f}x base, {tot/prizma_ascoded:.2f}x Prizma)")
            break


if __name__ == "__main__":
    print("Param counts (matched):")
    for (d, L, H) in [(64, 2, 2), (128, 4, 4)]:
        tf = Transformer(TFConfig(vocab=V, d_model=d, n_layers=L, n_heads=H, max_len=T + 8, rope=True))
        ps = PrizmaSeqLM(PrizmaSeqConfig(vocab=V, d_model=d, n_layers=L, n_heads=H, max_len=T + 8,
                                       feat_map="quad2", feat_n2=224))
        print(f"   d{d}L{L}H{H}: TF {param_count(tf):,}  Prizma-quad2 {param_count(ps):,}")

    ledger(64, 2, 2, d_phi=256, label="LEGACY d64L2H2")
    h = ledger(128, 4, 4, d_phi=256, label="HEADLINE d128L4H4")
    flop_matched_tf_search(128, 4, 4, d_phi=256, target_ratio=h["ratio_ascoded"])
