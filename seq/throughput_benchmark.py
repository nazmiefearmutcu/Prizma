"""
Throughput benchmarking script for Prizma-Seq delta updates.
Compares eager, compiled, and Triton execution paths (both forward and backward passes).
Computes latency, tokens/second, and relative speedup.
MEASURED-ONLY. Every row this script emits is wall-clock timed on the device it names. If CUDA or
Triton is unavailable, those rows are simply ABSENT -- nothing is extrapolated, simulated, or scaled
from CPU timings. (An earlier version synthesised "CUDA (Simulated)" rows by dividing the CPU time by
hardcoded constants and printed them, to one decimal place, in the same table as the measured rows.
Those rows were not measurements of anything and have been removed.)

BACKWARD TIMING. Forward and backward are timed in SEPARATE loops, each with its own synchronisation
barrier, and the backward loop times only the `.backward()` call (its graph is rebuilt outside the
timer). The previous method timed a combined forward+backward loop and reported
`t_bwd = t_combined - t_fwd`; that subtraction is unreliable and produced, among other things, a
"backward 70x faster than the corresponding forward" row, which is not physically possible.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import torch

# Ensure the parent directory of 'seq' is in the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seq.delta import chunked_delta
from seq.delta_fused import fused_chunked_delta
from seq.delta_triton import triton_chunked_delta, _HAS_TRITON

try:
    from seq.delta_triton import _triton_chunked_delta
except ImportError:
    _triton_chunked_delta = None


def get_current_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def benchmark_path(name, fn, device, B, H, T, d, chunk, warmup=5, runs=20):
    """Benchmarks forward and backward pass for a given implementation function."""
    torch.manual_seed(42)
    
    # Initialize tensors on the target device
    try:
        q = torch.randn(B, H, T, d, device=device, requires_grad=True)
        k = torch.randn(B, H, T, d, device=device)
        k = (k / k.norm(dim=-1, keepdim=True)).requires_grad_(True)
        v = torch.randn(B, H, T, d, device=device, requires_grad=True)
        beta = (torch.rand(B, H, T, device=device) * 0.99).requires_grad_(True)
        alpha = (0.5 + 0.5 * torch.rand(B, H, T, device=device)).requires_grad_(True)
    except Exception as e:
        print(f"[{name}] Tensor initialization failed on {device}: {e}")
        return None, None, False

    try:
        # 1. Warmup Forward
        for _ in range(warmup):
            o, s = fn(q, k, v, beta, alpha, chunk=chunk)
        
        sync_device(device)
        
        # 2. Time Forward
        t0 = time.perf_counter()
        for _ in range(runs):
            o, s = fn(q, k, v, beta, alpha, chunk=chunk)
        sync_device(device)
        t_fwd = (time.perf_counter() - t0) / runs

        # 3. Warmup Backward
        grad_o = torch.randn_like(o)
        grad_s = torch.randn_like(s)
        for _ in range(warmup):
            o, s = fn(q, k, v, beta, alpha, chunk=chunk)
            torch.autograd.backward([o, s], [grad_o, grad_s])

        sync_device(device)

        # 4. Time the BACKWARD pass IN ISOLATION.
        #    The graph is rebuilt outside the timer and the device is synchronised on both sides of
        #    the .backward() call, so what is measured is the backward pass and nothing else. Do NOT
        #    go back to timing a combined loop and subtracting t_fwd: that subtraction is dominated
        #    by run-to-run variance and by lazy evaluation, and it can (and did) yield a "backward
        #    faster than forward" result that is not physically possible.
        t_bwd_total = 0.0
        for _ in range(runs):
            o, s = fn(q, k, v, beta, alpha, chunk=chunk)
            # Clear grads to avoid memory inflation (though negligible for small tests)
            q.grad = None
            k.grad = None
            v.grad = None
            beta.grad = None
            alpha.grad = None
            sync_device(device)                       # barrier: forward fully materialised
            t0 = time.perf_counter()
            torch.autograd.backward([o, s], [grad_o, grad_s])
            sync_device(device)                       # barrier: backward fully materialised
            t_bwd_total += time.perf_counter() - t0
        t_bwd = t_bwd_total / runs

        return t_fwd, t_bwd, True
    except Exception as e:
        print(f"[{name}] Run failed on {device}: {e}")
        return None, None, False


def main():
    parser = argparse.ArgumentParser(description="Prizma-Seq Throughput Benchmark")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (B)")
    parser.add_argument("--heads", type=int, default=4, help="Number of heads (H)")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length (T)")
    parser.add_argument("--dim", type=int, default=64, help="Head dimension (d)")
    parser.add_argument("--chunk", type=int, default=64, help="Chunk size (C)")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--runs", type=int, default=20, help="Number of timed benchmark runs")
    args = parser.parse_args()

    B, H, T, d, chunk = args.batch, args.heads, args.seq_len, args.dim, args.chunk
    total_tokens = B * T

    print("=" * 60)
    print(" PRIZMA-SEQ DELTA UPDATES THROUGHPUT BENCHMARK")
    print("=" * 60)
    print(f"Shape: B={B}, H={H}, T={T}, d={d}, chunk={chunk}")
    print(f"Tokens per Forward Pass: {total_tokens:,}")
    print(f"Warmup runs: {args.warmup}, Timed runs: {args.runs}")
    
    device = get_current_device()
    print(f"Current detected hardware: {device.type.upper()}")
    print("-" * 60)

    results = []

    # --- 1. Eager CPU (Baseline) ---
    print("Benchmarking Eager on CPU...")
    cpu_device = torch.device("cpu")
    t_fwd_cpu, t_bwd_cpu, ok = benchmark_path("Eager CPU", chunked_delta, cpu_device, B, H, T, d, chunk, args.warmup, args.runs)
    if ok:
        results.append({
            "path": "Eager", "pass": "Forward", "device": "CPU", 
            "time_ms": t_fwd_cpu * 1000.0, "tokens_sec": total_tokens / t_fwd_cpu,
            "type": "Measured", "raw_time": t_fwd_cpu
        })
        results.append({
            "path": "Eager", "pass": "Backward", "device": "CPU", 
            "time_ms": t_bwd_cpu * 1000.0, "tokens_sec": total_tokens / t_bwd_cpu,
            "type": "Measured", "raw_time": t_bwd_cpu
        })
    else:
        print("CRITICAL ERROR: CPU benchmark failed.")
        sys.exit(1)

    # --- 2. Compiled on CPU (if supported/works) ---
    print("Benchmarking Compiled on CPU (if possible)...")
    try:
        # Wrap compiled helper
        compiled_helper = torch.compile(chunked_delta, fullgraph=False)
        t_fwd_comp_cpu, t_bwd_comp_cpu, ok = benchmark_path("Compiled CPU", compiled_helper, cpu_device, B, H, T, d, chunk, args.warmup, args.runs)
        if ok:
            results.append({
                "path": "Compiled", "pass": "Forward", "device": "CPU", 
                "time_ms": t_fwd_comp_cpu * 1000.0, "tokens_sec": total_tokens / t_fwd_comp_cpu,
                "type": "Measured", "raw_time": t_fwd_comp_cpu
            })
            results.append({
                "path": "Compiled", "pass": "Backward", "device": "CPU", 
                "time_ms": t_bwd_comp_cpu * 1000.0, "tokens_sec": total_tokens / t_bwd_comp_cpu,
                "type": "Measured", "raw_time": t_bwd_comp_cpu
            })
    except Exception as e:
        print(f"Skipping CPU Compilation benchmark: {e}")

    # --- 3. Run GPU measurements (CUDA / MPS) if available ---
    has_gpu = device.type in ("cuda", "mps")
    
    if has_gpu:
        print(f"Benchmarking on GPU ({device.type.upper()})...")
        
        # GPU Eager
        t_fwd_gpu, t_bwd_gpu, ok = benchmark_path(f"Eager {device.type.upper()}", chunked_delta, device, B, H, T, d, chunk, args.warmup, args.runs)
        if ok:
            results.append({
                "path": "Eager", "pass": "Forward", "device": device.type.upper(), 
                "time_ms": t_fwd_gpu * 1000.0, "tokens_sec": total_tokens / t_fwd_gpu,
                "type": "Measured", "raw_time": t_fwd_gpu
            })
            results.append({
                "path": "Eager", "pass": "Backward", "device": device.type.upper(), 
                "time_ms": t_bwd_gpu * 1000.0, "tokens_sec": total_tokens / t_bwd_gpu,
                "type": "Measured", "raw_time": t_bwd_gpu
            })
            
            # GPU Compiled
            # We can use fused_chunked_delta (with compile backend) or compile chunked_delta
            def compile_wrapper(*args, **kwargs):
                return fused_chunked_delta(*args, **kwargs, backend="compile")
                
            t_fwd_comp_gpu, t_bwd_comp_gpu, ok = benchmark_path(f"Compiled {device.type.upper()}", compile_wrapper, device, B, H, T, d, chunk, args.warmup, args.runs)
            if ok:
                results.append({
                    "path": "Compiled", "pass": "Forward", "device": device.type.upper(), 
                    "time_ms": t_fwd_comp_gpu * 1000.0, "tokens_sec": total_tokens / t_fwd_comp_gpu,
                    "type": "Measured", "raw_time": t_fwd_comp_gpu
                })
                results.append({
                    "path": "Compiled", "pass": "Backward", "device": device.type.upper(), 
                    "time_ms": t_bwd_comp_gpu * 1000.0, "tokens_sec": total_tokens / t_bwd_comp_gpu,
                    "type": "Measured", "raw_time": t_bwd_comp_gpu
                })

            # GPU Triton (Only if CUDA + Triton available)
            if device.type == "cuda" and _HAS_TRITON and _triton_chunked_delta is not None:
                t_fwd_triton, t_bwd_triton, ok = benchmark_path("Triton CUDA", _triton_chunked_delta, device, B, H, T, d, chunk, args.warmup, args.runs)
                if ok:
                    results.append({
                        "path": "Triton", "pass": "Forward", "device": "CUDA", 
                        "time_ms": t_fwd_triton * 1000.0, "tokens_sec": total_tokens / t_fwd_triton,
                        "type": "Measured", "raw_time": t_fwd_triton
                    })
                    results.append({
                        "path": "Triton", "pass": "Backward", "device": "CUDA", 
                        "time_ms": t_bwd_triton * 1000.0, "tokens_sec": total_tokens / t_bwd_triton,
                        "type": "Measured", "raw_time": t_bwd_triton
                    })
            else:
                print("Triton CUDA path not available on this GPU configuration.")

    # --- 4. Absent execution paths stay ABSENT ---
    # There is deliberately no extrapolation step here. A previous version of this script, when CUDA
    # or Triton was unavailable, synthesised "CUDA (Simulated)" rows as t_cpu / {120, 180, 210} using
    # hardcoded constants and emitted them into the same table as the measured rows, formatted to the
    # same precision. Those numbers were arithmetic on a CPU timing, not a measurement of any GPU, and
    # a reader could not tell the difference at a glance. If a CUDA/Triton number is wanted, run this
    # script on a CUDA box; until then the row does not exist.
    if device.type != "cuda":
        print("\nNOTE: no CUDA device -> no CUDA rows. Nothing is extrapolated or simulated.")
    elif not _HAS_TRITON:
        print("\nNOTE: CUDA present but Triton missing -> no Triton rows. Nothing is extrapolated.")

    # --- 5. Generate and Output the Table ---
    # Calculate speedup relative to CPU Eager of the same pass type (Forward or Backward)
    fwd_baseline = t_fwd_cpu
    bwd_baseline = t_bwd_cpu

    print("\n" + "=" * 80)
    print(" BENCHMARK RESULTS")
    print("=" * 80 + "\n")
    
    # Sort results for readability: Device, Path, Pass
    def sort_key(r):
        dev_order = {"CPU": 0, "MPS": 1, "CUDA": 2}
        path_order = {"Eager": 0, "Compiled": 1, "Triton": 2}
        pass_order = {"Forward": 0, "Backward": 1}
        return (dev_order.get(r["device"], 9), path_order.get(r["path"], 9), pass_order.get(r["pass"], 9))
    
    sorted_results = sorted(results, key=sort_key)

    # Print Table
    markdown_lines = []
    markdown_lines.append("| Execution Path | Pass | Device | Time (ms) | Throughput (tokens/s) | Speedup vs Eager CPU | Type |")
    markdown_lines.append("|:---|:---|:---|:---|:---|:---|:---|")
    
    for r in sorted_results:
        baseline = fwd_baseline if r["pass"] == "Forward" else bwd_baseline
        speedup = baseline / r["raw_time"]
        speedup_str = f"{speedup:.2f}x"
        
        line = f"| {r['path']} | {r['pass']} | {r['device']} | {r['time_ms']:.4f} | {r['tokens_sec']:,.1f} | {speedup_str} | {r['type']} |"
        markdown_lines.append(line)
        
    table_str = "\n".join(markdown_lines)
    print(table_str)
    print("\n" + "=" * 80)
    
    # Write report file to same directory
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.md")
    try:
        import platform
        import subprocess
        cpu = platform.processor() or platform.machine()
        if sys.platform == "darwin":                      # platform.processor() is just "arm" on macOS
            try:
                cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                     capture_output=True, text=True, timeout=5).stdout.strip() or cpu
            except Exception:
                pass
        host = f"{platform.platform()} / {cpu} ({os.cpu_count()} cores)"
        with open(report_path, "w") as f:
            f.write("# Prizma-Seq Delta Updates Benchmark Results\n\n")
            f.write("> **Every row below is measured wall-clock time on the device it names.** Rows for\n")
            f.write("> hardware this machine does not have are absent, not extrapolated. Forward and\n")
            f.write("> backward are timed in separate loops with synchronisation barriers on both sides\n")
            f.write("> of the timed region; the backward time is NOT obtained by subtracting a forward\n")
            f.write("> time from a combined forward+backward time.\n\n")
            f.write(f"**Host**: {host}, torch {torch.__version__}\n\n")
            f.write(f"**Shape Config**: Batch size = {B}, Heads = {H}, Seq len = {T}, Dim = {d}, Chunk = {chunk}\n\n")
            f.write(f"**Tokens/pass**: {total_tokens:,}\n\n")
            f.write(f"**Protocol**: {args.warmup} warmup + {args.runs} timed runs per cell, mean.\n\n")
            f.write("**Caveat on absolute times**: this is a wall-clock benchmark on a shared, unpinned\n")
            f.write("machine with no CI gate on it. Run-to-run variation of ~2x has been observed on the\n")
            f.write("CPU rows depending on what else was running. Treat the ORDERING and the rough ratios\n")
            f.write("as the signal; do not read the absolute milliseconds as a hardware characterisation.\n\n")
            f.write("**Caveat on the `Compiled` rows**: off CUDA, `seq/delta_fused.py` may fail to\n")
            f.write("compile and fall back to the eager kernel (it emits a RuntimeWarning when it does).\n")
            f.write("A `Compiled` row whose time matches its `Eager` row is that fallback, not a\n")
            f.write("compiled speedup of ~1.0x. Check the run log.\n\n")
            f.write(table_str)
            f.write("\n")
        print(f"Results written to {report_path}")
    except Exception as e:
        print(f"Could not write benchmark_results.md: {e}")


if __name__ == "__main__":
    main()
