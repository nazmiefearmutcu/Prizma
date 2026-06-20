# Prizma-Seq vs Transformer Scaling Analysis

## Model Configurations & Parameter Counts

| Scale | Model | Layers | Width ($d_{model}$) | Heads | FFN Dim ($d_{ff}$) | Extra Params / Features | Parameter Count | Match Delta |
|---|---|---|---|---|---|---|---|---|
| 50M | Prizma-Seq | 10 | 512 | 8 | 1365 (SwiGLU) | $d_{phi}$=320, window=16 | 47,964,832 | Reference |
| 50M | Transformer | 10 | 512 | 8 | 1376 | RoPE | 48,015,872 | +0.106% |
| 100M | Prizma-Seq | 11 | 768 | 12 | 2048 (SwiGLU) | $d_{phi}$=320, window=16 | 102,602,760 | Reference |
| 100M | Transformer | 11 | 768 | 12 | 2056 | RoPE | 102,653,184 | +0.049% |

## Analytical State/KV-Cache Memory Size Comparison

> [!NOTE]
> Memory sizes are calculated per sequence batch element ($B=1$) assuming FP16 precision (2 bytes per float).
> Prizma-Seq state size includes the recurrent delta-state and the window local ring buffer, which remains constant across sequence lengths.

### 50M Scale Memory Size Comparison

| Sequence Length ($T$) | Transformer KV-Cache (floats) | Transformer KV-Cache (FP16) | Prizma-Seq State (floats) | Prizma-Seq State (FP16) | Memory Ratio (TF / Prizma) |
|---|---|---|---|---|---|
| 1,024 | 10,485,760 | 20.00 MB | 1,802,240 | 3.44 MB | 5.82x |
| 2,048 | 20,971,520 | 40.00 MB | 1,802,240 | 3.44 MB | 11.64x |
| 4,096 | 41,943,040 | 80.00 MB | 1,802,240 | 3.44 MB | 23.27x |
| 8,192 | 83,886,080 | 160.00 MB | 1,802,240 | 3.44 MB | 46.55x |
| 16,384 | 167,772,160 | 320.00 MB | 1,802,240 | 3.44 MB | 93.09x |
| 32,768 | 335,544,320 | 640.00 MB | 1,802,240 | 3.44 MB | 186.18x |
| 65,536 | 671,088,640 | 1.25 GB | 1,802,240 | 3.44 MB | 372.36x |

### 100M Scale Memory Size Comparison

| Sequence Length ($T$) | Transformer KV-Cache (floats) | Transformer KV-Cache (FP16) | Prizma-Seq State (floats) | Prizma-Seq State (FP16) | Memory Ratio (TF / Prizma) |
|---|---|---|---|---|---|
| 1,024 | 17,301,504 | 33.00 MB | 2,973,696 | 5.67 MB | 5.82x |
| 2,048 | 34,603,008 | 66.00 MB | 2,973,696 | 5.67 MB | 11.64x |
| 4,096 | 69,206,016 | 132.00 MB | 2,973,696 | 5.67 MB | 23.27x |
| 8,192 | 138,412,032 | 264.00 MB | 2,973,696 | 5.67 MB | 46.55x |
| 16,384 | 276,824,064 | 528.00 MB | 2,973,696 | 5.67 MB | 93.09x |
| 32,768 | 553,648,128 | 1.03 GB | 2,973,696 | 5.67 MB | 186.18x |
| 65,536 | 1,107,296,256 | 2.06 GB | 2,973,696 | 5.67 MB | 372.36x |
