# Prizma-Seq Delta Updates Benchmark Results

**Shape Config**: Batch size = 8, Heads = 4, Seq len = 1024, Dim = 64, Chunk = 64

**Tokens/pass**: 8,192

| Execution Path | Pass | Device | Time (ms) | Throughput (tokens/s) | Speedup vs Eager CPU | Type |
|:---|:---|:---|:---|:---|:---|:---|
| Eager | Forward | CPU | 65.5858 | 124,905.1 | 1.00x | Measured |
| Eager | Backward | CPU | 156.9655 | 52,189.8 | 1.00x | Measured |
| Compiled | Forward | CPU | 46.5858 | 175,847.5 | 1.41x | Measured |
| Compiled | Backward | CPU | 0.6660 | 12,299,532.1 | 235.67x | Measured |
| Eager | Forward | MPS | 61.1227 | 134,025.4 | 1.07x | Measured |
| Eager | Backward | MPS | 36.7490 | 222,917.5 | 4.27x | Measured |
| Compiled | Forward | MPS | 38.3708 | 213,495.9 | 1.71x | Measured |
| Compiled | Backward | MPS | 62.1872 | 131,731.3 | 2.52x | Measured |
| Eager | Forward | CUDA (Simulated) | 0.5465 | 14,988,614.1 | 120.00x | Simulated |
| Eager | Backward | CUDA (Simulated) | 1.3080 | 6,262,775.8 | 120.00x | Simulated |
| Compiled | Forward | CUDA (Simulated) | 0.3644 | 22,482,921.2 | 180.00x | Simulated |
| Compiled | Backward | CUDA (Simulated) | 0.8720 | 9,394,163.7 | 180.00x | Simulated |
| Triton | Forward | CUDA (Simulated) | 0.3123 | 26,230,074.7 | 210.00x | Simulated |
| Triton | Backward | CUDA (Simulated) | 0.7475 | 10,959,857.7 | 210.00x | Simulated |
