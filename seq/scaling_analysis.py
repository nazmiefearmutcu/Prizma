"""
Prizma-Seq vs Transformer Scaling Analysis.
Calculates parameter counts and analytic memory (KV-cache vs carried recurrent state)
at 50M and 100M parameter scales across sequence lengths from 1024 to 65536.
"""

import os
import sys
import json

# Ensure parent directory is in path for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from seq.transformer import Transformer, TFConfig
from seq.prizma_seq import PrizmaSeqLM, PrizmaSeqConfig
from seq.common import param_count

def match_tf_dff(vocab, d_model, n_layers, n_heads, target_params):
    """Finds the Transformer SwiGLU d_ff that matches the target parameter count."""
    base_d_ff = int(round(8 / 3 * d_model / 8) * 8)
    best_d_ff = base_d_ff
    best_params = 0
    min_diff = float("inf")
    
    # Search range around the base d_ff
    for d_ff in range(max(8, base_d_ff - 256), base_d_ff + 512, 8):
        cfg = TFConfig(vocab=vocab, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, rope=True)
        m = Transformer(cfg)
        p = param_count(m)
        diff = abs(p - target_params)
        if diff < min_diff:
            min_diff = diff
            best_d_ff = d_ff
            best_params = p
            
    return best_d_ff, best_params

def compute_kv_cache_floats(n_layers, d_model, seq_len):
    """Transformer KV cache floats = 2 * L * d_model * seq_len"""
    return 2 * n_layers * d_model * seq_len

def compute_prizma_state_floats(n_layers, d_model, d_phi, window):
    """Prizma state floats = L * d_model * d_phi + 2 * L * d_model * window"""
    return n_layers * d_model * d_phi + 2 * n_layers * d_model * window

def format_memory(floats, precision="fp16"):
    """Formats floats to MB or GB representation."""
    bytes_per_float = 2 if precision == "fp16" else 4
    bytes_total = floats * bytes_per_float
    if bytes_total >= 1024**3:
        return f"{bytes_total / 1024**3:.2f} GB"
    else:
        return f"{bytes_total / 1024**2:.2f} MB"

def main():
    vocab = 32000
    window = 16
    feat_n2 = 256
    
    # Define targets
    scales = {
        "50M": {
            "d_model": 512,
            "n_layers": 10,
            "n_heads": 8,
        },
        "100M": {
            "d_model": 768,
            "n_layers": 11,
            "n_heads": 12,
        }
    }
    
    seq_lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    
    results = {}
    
    print("=" * 80)
    print("Prizma-Seq vs Transformer Scaling Analysis (50M & 100M)")
    print("=" * 80)
    
    for scale_name, conf in scales.items():
        d_model = conf["d_model"]
        n_layers = conf["n_layers"]
        n_heads = conf["n_heads"]
        dh = d_model // n_heads
        d_phi = dh + feat_n2
        
        # Instantiate Prizma-Seq
        ps_cfg = PrizmaSeqConfig(
            vocab=vocab,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            feat_map="quad2",
            feat_n2=feat_n2,
            window=window
        )
        ps_model = PrizmaSeqLM(ps_cfg)
        ps_params = param_count(ps_model)
        
        # Parameter-match Transformer
        tf_d_ff, tf_params = match_tf_dff(vocab, d_model, n_layers, n_heads, ps_params)
        tf_cfg = TFConfig(
            vocab=vocab,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=tf_d_ff,
            rope=True
        )
        tf_model = Transformer(tf_cfg)
        
        param_diff_pct = (tf_params - ps_params) / ps_params * 100
        
        print(f"\nScale: {scale_name}")
        print(f"  Prizma-Seq Config:  d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}, d_h={dh}, d_phi={d_phi}")
        print(f"  Prizma-Seq Params:  {ps_params:,}")
        print(f"  Transformer Config: d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}, d_ff={tf_d_ff}")
        print(f"  Transformer Params: {tf_params:,} ({param_diff_pct:+.3f}% match)")
        
        # Memory Analysis
        ps_state_fl = compute_prizma_state_floats(n_layers, d_model, d_phi, window)
        
        scale_results = {
            "config": {
                "vocab": vocab,
                "d_model": d_model,
                "n_layers": n_layers,
                "n_heads": n_heads,
                "dh": dh,
                "d_phi": d_phi,
                "window": window,
                "prizma_params": ps_params,
                "transformer_params": tf_params,
                "transformer_d_ff": tf_d_ff,
                "param_diff_pct": param_diff_pct
            },
            "prizma_state_floats": ps_state_fl,
            "comparisons": []
        }
        
        print("\n  Memory Comparison:")
        print(f"    {'Seq Len':<10} | {'TF KV-Cache':<22} | {'Prizma State':<22} | {'Ratio':<10}")
        print(f"    {'-'*10} | {'-'*22} | {'-'*22} | {'-'*10}")
        
        for seq_len in seq_lengths:
            tf_kv_fl = compute_kv_cache_floats(n_layers, d_model, seq_len)
            ratio = tf_kv_fl / ps_state_fl
            
            # Format outputs
            tf_mem_str = f"{tf_kv_fl:,} fl ({format_memory(tf_kv_fl, 'fp16')})"
            ps_mem_str = f"{ps_state_fl:,} fl ({format_memory(ps_state_fl, 'fp16')})"
            
            print(f"    {seq_len:<10} | {tf_mem_str:<22} | {ps_mem_str:<22} | {ratio:.2f}x")
            
            scale_results["comparisons"].append({
                "seq_len": seq_len,
                "tf_kv_cache_floats": tf_kv_fl,
                "ratio": ratio
            })
            
        results[scale_name] = scale_results
        
    # Save to JSON
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "scaling_analysis.json")
    
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    
    # Generate Markdown Table
    md_lines = []
    md_lines.append("# Prizma-Seq vs Transformer Scaling Analysis")
    md_lines.append("")
    md_lines.append("## Model Configurations & Parameter Counts")
    md_lines.append("")
    md_lines.append("| Scale | Model | Layers | Width ($d_{model}$) | Heads | FFN Dim ($d_{ff}$) | Extra Params / Features | Parameter Count | Match Delta |")
    md_lines.append("|---|---|---|---|---|---|---|---|---|")
    
    for scale_name, data in results.items():
        cfg = data["config"]
        md_lines.append(f"| {scale_name} | Prizma-Seq | {cfg['n_layers']} | {cfg['d_model']} | {cfg['n_heads']} | {cfg['d_model']*8//3} (SwiGLU) | $d_{{phi}}$={cfg['d_phi']}, window={cfg['window']} | {cfg['prizma_params']:,} | Reference |")
        md_lines.append(f"| {scale_name} | Transformer | {cfg['n_layers']} | {cfg['d_model']} | {cfg['n_heads']} | {cfg['transformer_d_ff']} | RoPE | {cfg['transformer_params']:,} | {cfg['param_diff_pct']:+.3f}% |")
        
    md_lines.append("")
    md_lines.append("## Analytical State/KV-Cache Memory Size Comparison")
    md_lines.append("")
    md_lines.append("> [!NOTE]")
    md_lines.append("> Memory sizes are calculated per sequence batch element ($B=1$) assuming FP16 precision (2 bytes per float).")
    md_lines.append("> Prizma-Seq state size includes the recurrent delta-state and the window local ring buffer, which remains constant across sequence lengths.")
    md_lines.append("")
    
    for scale_name, data in results.items():
        md_lines.append(f"### {scale_name} Scale Memory Size Comparison")
        md_lines.append("")
        md_lines.append("| Sequence Length ($T$) | Transformer KV-Cache (floats) | Transformer KV-Cache (FP16) | Prizma-Seq State (floats) | Prizma-Seq State (FP16) | Memory Ratio (TF / Prizma) |")
        md_lines.append("|---|---|---|---|---|---|")
        
        ps_fl = data["prizma_state_floats"]
        ps_mem = format_memory(ps_fl, "fp16")
        
        for comp in data["comparisons"]:
            seq_len = comp["seq_len"]
            tf_fl = comp["tf_kv_cache_floats"]
            tf_mem = format_memory(tf_fl, "fp16")
            ratio = comp["ratio"]
            md_lines.append(f"| {seq_len:,} | {tf_fl:,} | {tf_mem} | {ps_fl:,} | {ps_mem} | {ratio:.2f}x |")
            
        md_lines.append("")
        
    # Write to a summary file or stdout
    md_content = "\n".join(md_lines)
    print("\n" + "=" * 80)
    print("Generated Markdown Table:")
    print("=" * 80)
    print(md_content)
    
    # Save markdown table to results/scaling_analysis.md for easy viewing
    md_path = os.path.join(out_dir, "scaling_analysis.md")
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"\nMarkdown table saved to {md_path}")

if __name__ == '__main__':
    main()
