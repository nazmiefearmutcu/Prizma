# Prizma-Seq Delta Updates Benchmark Results

> **Every row below is measured wall-clock time on the device it names.** Rows for
> hardware this machine does not have are absent, not extrapolated. Forward and
> backward are timed in separate loops with synchronisation barriers on both sides
> of the timed region; the backward time is NOT obtained by subtracting a forward
> time from a combined forward+backward time.

**Host**: macOS-27.0-arm64-arm-64bit-Mach-O / Apple M4 (10 cores), torch 2.12.0

**Shape Config**: Batch size = 8, Heads = 4, Seq len = 1024, Dim = 64, Chunk = 64

**Tokens/pass**: 8,192

**Protocol**: 5 warmup + 20 timed runs per cell, mean.

**Caveat on absolute times**: this is a wall-clock benchmark on a shared, unpinned
machine with no CI gate on it. Run-to-run variation of ~2x has been observed on the
CPU rows depending on what else was running. Treat the ORDERING and the rough ratios
as the signal; do not read the absolute milliseconds as a hardware characterisation.

**Caveat on the `Compiled` rows**: off CUDA, `seq/delta_fused.py` may fail to
compile and fall back to the eager kernel (it emits a RuntimeWarning when it does).
A `Compiled` row whose time matches its `Eager` row is that fallback, not a
compiled speedup of ~1.0x. Check the run log.

| Execution Path | Pass | Device | Time (ms) | Throughput (tokens/s) | Speedup vs Eager CPU | Type |
|:---|:---|:---|:---|:---|:---|:---|
| Eager | Forward | CPU | 41.1795 | 198,934.1 | 1.00x | Measured |
| Eager | Backward | CPU | 104.8472 | 78,132.7 | 1.00x | Measured |
| Compiled | Forward | CPU | 26.5680 | 308,341.2 | 1.55x | Measured |
| Compiled | Backward | CPU | 43.1710 | 189,757.1 | 2.43x | Measured |
| Eager | Forward | MPS | 42.7249 | 191,738.2 | 0.96x | Measured |
| Eager | Backward | MPS | 88.6049 | 92,455.3 | 1.18x | Measured |
| Compiled | Forward | MPS | 41.9970 | 195,061.5 | 0.98x | Measured |
| Compiled | Backward | MPS | 84.7109 | 96,705.4 | 1.24x | Measured |
