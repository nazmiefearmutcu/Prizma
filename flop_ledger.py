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
import json
import os

import torch
from seq.transformer import Transformer, TFConfig
from seq.prizma_seq import PrizmaSeqLM, PrizmaSeqConfig
from seq.common import param_count
from seq.ledger import param_match_report

V = 512
T = 384            # MixedMQAR(max_pairs=128): seq_len = 2*128 + 128 = 384

# ---- R4 d_phi reconciliation: the FOUR canonical configs, each pinned to its exact -------- #
# (feat_map, feat_n2/feat_rank) -> d_phi. d_phi is NOT a free knob: it is DERIVED in
# PrizmaSeqConfig.__post_init__ (quad2: d_h+feat_n2 ; quad2_lowrank: d_h + r*(r+1)//2 ; none: d_h).
# We list the d_phi that each config yields at d_h=32 so the analytical ledger() (which takes d_phi
# directly) and the REAL constructed module agree by construction.
#   d_phi=256 (feat_n2=224) = v1 PUBLISHED reference ("full quad2") — kept, RELABELED, not deleted.
#   d_phi=128 (feat_n2=96)  = current CODE DEFAULT ("quad2 code-default").
#   d_phi=137 (quad2_lowrank, r=14) = v2 LEAN target.
#   d_phi=32  (feat_map='none') = no-feature-map baseline.
# The FINAL canonical v2 d_phi is LOCKED by a pending A100 >=10-seed MQAR-D128 solve-rate gate
# (plan Task 1.D); this ledger does NOT pick a winner — it makes every FLOP number unambiguous.
PER_CONFIG = [
    # (label,                        feat_map,         feat_n2, feat_rank, d_phi_at_dh32)
    ("none_d32",                     "none",           0,       0,         32),
    ("quad2_d128_codedefault",       "quad2",          96,      0,         128),
    ("quad2_d256_v1ref",             "quad2",          224,     0,         256),
    ("quad2_lowrank_d137_v2lean",    "quad2_lowrank",  0,       14,        137),
]


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
    return {"d_phi": d_phi, "tf_per_tok": tf_total / T, "ps_ascoded_per_tok": ps_total_ascoded / T,
            "ps_ideal_per_tok": ps_total_ideal / T,
            "ratio_ascoded": ps_total_ascoded / tf_total, "ratio_ideal": ps_total_ideal / tf_total}


def flop_matched_tf_search(d, H, L, d_phi, target_ratio=None, T=T, verbose=True):
    """Find a bigger TF (grow d_ff, then layers) whose forward FLOPs >= the Prizma as-coded cost.
    Returns the deeper + wider candidates (dict) so the headline FLOP-matched arm can be sized from
    the same analytical FLOPs and so the per-config ledger can record them machine-readably."""
    dff = dff_of(d)
    head = mm(T, d, V)
    base_tf = L * total(tf_layer_flops_causal(T, d, H, dff)) + head
    pf = prizma_layer_flops(T, d, H, dff, d_phi=d_phi)
    prizma_ascoded = L * total(pf, exclude=("window_band_ideal",)) + head
    if verbose:
        print(f"\n=== FLOP-matched TF search @ d={d} H={H} L={L} d_phi={d_phi} (match Prizma "
              f"as-coded {prizma_ascoded/T/1e3:.1f} kFLOP/tok) ===")
    out = {"deeper": None, "wider": None}
    # option A: deeper TF (more layers) at same width
    for L2 in range(L, 4 * L + 1):
        tot = L2 * total(tf_layer_flops_causal(T, d, H, dff)) + head
        if tot >= prizma_ascoded:
            out["deeper"] = {"label": f"d{d}L{L2}H{H}", "d": d, "L": L2, "H": H,
                             "kflop_per_tok": tot / T / 1e3, "x_base": tot / base_tf,
                             "x_prizma": tot / prizma_ascoded}
            if verbose:
                print(f"   deeper: TF {out['deeper']['label']}  -> {tot/T/1e3:6.1f} kFLOP/tok  "
                      f"({tot/base_tf:.2f}x base, {tot/prizma_ascoded:.2f}x Prizma)")
            break
    # option B: wider d_model (keep L,H ratio) — FLOP ~ d^2
    for dm in range(d, 3 * d + 1, 16):
        if dm % H:
            continue
        tot = L * total(tf_layer_flops_causal(T, dm, H, dff_of(dm))) + mm(T, dm, V)
        if tot >= prizma_ascoded:
            out["wider"] = {"label": f"d{dm}L{L}H{H}", "d": dm, "L": L, "H": H,
                            "kflop_per_tok": tot / T / 1e3, "x_base": tot / base_tf,
                            "x_prizma": tot / prizma_ascoded}
            if verbose:
                print(f"   wider : TF {out['wider']['label']}  -> {tot/T/1e3:6.1f} kFLOP/tok  "
                      f"({tot/base_tf:.2f}x base, {tot/prizma_ascoded:.2f}x Prizma)")
            break
    return out


def build_matched_pair(d, H, L, feat_map, feat_n2, feat_rank):
    """Construct the REAL param-matched (Transformer, PrizmaSeqLM) pair for one config and return
    the constructed config's effective d_phi + a param-match report. This ties every FLOP number in
    the ledger to actual nn.Modules, not just an analytical d_phi."""
    tf = Transformer(TFConfig(vocab=V, d_model=d, n_layers=L, n_heads=H, max_len=T + 8, rope=True))
    kw = {}
    if feat_map in ("quad2", "rand_linear"):
        kw["feat_n2"] = feat_n2
    if feat_map == "quad2_lowrank":
        kw["feat_rank"] = feat_rank
    cfg = PrizmaSeqConfig(vocab=V, d_model=d, n_layers=L, n_heads=H, max_len=T + 8,
                          feat_map=feat_map, **kw)
    ps = PrizmaSeqLM(cfg)
    rep = param_match_report(tf, ps)
    return cfg, tf, ps, rep


def emit_per_config_ledger(d=128, H=4, L=4, verbose=True):
    """R4: emit the FLOP ledger for the FOUR canonical d_phi configs at scale d,H,L.

    For EACH config: print/return per-token kFLOP (as-coded & ideal), ratio vs the param-matched TF
    (as-coded & ideal), run flop_matched_tf_search, AND construct the REAL param-matched modules and
    assert |TF-Prizma|/TF < 0.02 so the ledger is tied to actual modules. These FLOP numbers are
    ANALYTICAL (allowed/honest); they are NOT measured-accuracy metrics. Returns a dict keyed by
    config label."""
    out = {}
    for (label, feat_map, feat_n2, feat_rank, d_phi) in PER_CONFIG:
        if verbose:
            print(f"\n=================== CONFIG {label}: feat_map={feat_map} "
                  f"feat_n2={feat_n2} feat_rank={feat_rank} -> d_phi={d_phi} ===================")
        cfg, tf, ps, rep = build_matched_pair(d, H, L, feat_map, feat_n2, feat_rank)
        # the constructed module's effective d_phi MUST equal the analytical d_phi we ledger.
        eff_dphi = cfg.d_phi
        assert eff_dphi == d_phi, (
            f"{label}: constructed d_phi={eff_dphi} != ledgered d_phi={d_phi}")
        assert rep["matched"], f"{label}: param-match FAILED {rep}"
        led = ledger(d, H, L, d_phi=d_phi, label=f"{label} (d{d}L{L}H{H})",
                     show_components=verbose)
        matched_tf = flop_matched_tf_search(d, H, L, d_phi=d_phi, verbose=verbose)
        rec = dict(led)
        rec.update({
            "feat_map": feat_map, "feat_n2": feat_n2, "feat_rank": feat_rank,
            "scale": f"d{d}L{L}H{H}",
            "param_match": {"tf_params": rep["tf_params"], "pz_params": rep["pz_params"],
                            "delta": rep["delta"], "rel": rep["rel"], "matched": rep["matched"],
                            "feat_map_added_params": rep["feat_map_added_params"]},
            "matched_tf": matched_tf,
        })
        out[label] = rec
    return out


def _write_outputs(per_cfg, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "flop_ledger_v2.json")
    payload = {
        "_about": ("R4 d_phi reconciliation. ANALYTICAL per-token forward FLOPs (causal-honest), "
                   "one entry per canonical (feat_map, feat_n2/feat_rank, d_phi) config at the "
                   "headline scale d128L4H4. These are analytical FLOP counts, NOT measured-accuracy "
                   "metrics. The canonical v2 d_phi is LOCKED by a pending A100 >=10-seed MQAR-D128 "
                   "solve-rate gate (plan Task 1.D); this ledger does not pick a winner."),
        "scale": "d128L4H4", "vocab": V, "T": T,
        "labels": {
            "none_d32": "no-feature-map baseline (feat_map='none')",
            "quad2_d128_codedefault": "current CODE DEFAULT (feat_n2=96 -> d_phi=128)",
            "quad2_d256_v1ref": "v1 PUBLISHED reference / full quad2 (feat_n2=224 -> d_phi=256)",
            "quad2_lowrank_d137_v2lean": "v2 LEAN target (quad2_lowrank, r=14 -> d_phi=137)",
        },
        "configs": per_cfg,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    return json_path


if __name__ == "__main__":
    import io
    from contextlib import redirect_stdout

    here = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(here, "results")

    buf = io.StringIO()

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, s):
            for st in self.streams:
                st.write(s)

        def flush(self):
            for st in self.streams:
                st.flush()

    import sys
    tee = _Tee(sys.stdout, buf)
    with redirect_stdout(tee):
        print("Param counts (matched, all four configs share params — feat map is buffers):")
        for (d, L, H) in [(64, 2, 2), (128, 4, 4)]:
            tf = Transformer(TFConfig(vocab=V, d_model=d, n_layers=L, n_heads=H, max_len=T + 8,
                                      rope=True))
            ps = PrizmaSeqLM(PrizmaSeqConfig(vocab=V, d_model=d, n_layers=L, n_heads=H,
                                             max_len=T + 8, feat_map="quad2", feat_n2=224))
            print(f"   d{d}L{L}H{H}: TF {param_count(tf):,}  Prizma {param_count(ps):,}")

        # legacy scale (single d_phi=256 reference, kept verbatim for back-compat)
        ledger(64, 2, 2, d_phi=256, label="LEGACY d64L2H2 (d_phi=256, v1 ref)")

        # R4: the FOUR canonical configs at the headline scale, each FLOP number pinned to its exact
        # (feat_map, feat_n2/feat_rank, d_phi) config and tied to a REAL param-matched module pair.
        print("\n" + "#" * 30 + " R4 PER-CONFIG LEDGER (headline d128L4H4) " + "#" * 30)
        per_cfg = emit_per_config_ledger(d=128, H=4, L=4, verbose=True)

        print("\n" + "=" * 78)
        print("R4 SUMMARY — per-config FLOP ratios (Prizma / param-matched TF) @ d128L4H4:")
        print(f"   {'config':<30} {'d_phi':>5} {'as-coded':>9} {'ideal':>7} "
              f"{'ascoded x':>10} {'ideal x':>8}")
        for label, rec in per_cfg.items():
            print(f"   {label:<30} {rec['d_phi']:>5} "
                  f"{rec['ps_ascoded_per_tok']/1e3:>8.1f} {rec['ps_ideal_per_tok']/1e3:>6.1f}k "
                  f"{rec['ratio_ascoded']:>9.2f}x {rec['ratio_ideal']:>7.2f}x")
        print("   (canonical v2 d_phi pending A100 >=10-seed MQAR-D128 gate — plan Task 1.D)")

    json_path = _write_outputs(per_cfg, results_dir)
    txt_path = os.path.join(results_dir, "flop_ledger_v2.txt")
    os.makedirs(results_dir, exist_ok=True)
    with open(txt_path, "w") as f:
        f.write(buf.getvalue())
    print(f"\n[wrote] {json_path}")
    print(f"[wrote] {txt_path}")
