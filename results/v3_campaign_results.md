# Prizma-Seq v3 campaign — raw results digest (auditable)

Source: 5 concurrent Colab A100/L4 runs, 2026-06-05/06. Raw JSONs on Drive folder
`prizma_results` (the project Colab Drive account). gpu_charlm2.json was written LOCAL on the Colab
runtime (read from the cell printout). gpu_bench.json is committed at `results/gpu_bench.json`.
These are the numbers transcribed into `docs/PRIZMA_SEQ_REPORT.md`; this file lets a referee
cross-check report ⇄ raw without the live notebook.

## B2 Induction — gpu_diag.json (param-matched d128L2H4, vocab32, 0.64% match, 3 seeds, cap 60k)
Best accuracy per seed (per_len 64/128/256):
- TF:          s0 0.998 (.991/.997/.998), s1 0.996 (.991/.996/.995), s2 0.903 (.908/.902/.903) → median 0.9956, 2/3 ≥0.98
- Prizma-quad2: s0 0.9995 (1.0/.9995/.9995), s1 1.0 (1.0/.9995/1.0), s2 0.9995 (1.0/.9985/.9995) → median 0.9995, 3/3 ≥0.98
- Prizma-none:  s0 1.0, s1 1.0, s2 0.999 → median 1.0, 3/3   (note: none ALSO solves induction)
- _bar: quad2 median≥0.98 AND per-len-median@256≥0.95 → pass=true (quad2 @256 median 0.9995)

## B3 Selective-copy — gpu_diag.json (same config)
selective variant best per seed:
- TF:          0.9994, 1.0, 0.9989 → median 0.9994
- Prizma-quad2: 0.9991, 0.9991, 0.9992 → median 0.9991
- Prizma-none:  0.9994, 0.9993, 0.9995 → median 0.9994   (none ALSO solves)
fixed-position control (isolates content-selectivity): TF 1.0 / quad2 0.9999 / none 1.0 → spread 1e-4
- _bar: selective quad2 median 0.9991 ≥0.97 AND ≥ TF_median−0.02 (0.9794); fixed ~equal → pass=true

## B4 Char-LM (CREDIBLE) — gpu_charlm2.json (text8 10M/0.5M/0.5M, V=27, d256L4H4, wd=0.1,
##   eval@250, val-based early-stop = TEST BPC @ min-VAL ckpt, param-match −0.13%, 2 seeds, 12k steps)
random-baseline BPC = 4.7549
- TF best_bpc:          s0 1.7253 (@9500), s1 1.7256 (@10500) → median 1.7254; final 1.7354/1.7323 (NO overfit)
- Prizma-quad2 best_bpc: s0 1.7505 (@9500), s1 1.7488 (@10000) → median 1.7496; final 1.7649/1.7619 (NO overfit)
- margin TF−Prizma = −0.0242  (threshold +0.05) → PASS
- timing: TF arm ~361s, Prizma arm ~1889s (~5.2× slower to train)
- SUPERSEDED shakespeare run (gpu_charlm.json, overfit recipe eval@1000): TF best ~2.21 vs Prizma best ~2.276
  (Prizma −0.07, FAIL) + final blew up to 6.5–7.3 ≫ random 6.02 (memorization). Not used for the verdict.

## B5 Inference latency+memory — gpu_latency.json (decode via model.step(); KV-cache for TF, O(1) for Prizma)
small (d128L4H4, state 147,456 floats): per-step ms TF {4k:4.47, 8k:4.46, 16k:5.39, 32k:8.95, 64k:16.95}
  vs Prizma {4k:6.96, 8k:6.95, 16k:6.95, 32k:6.95, 64k:7.06} → latency_crossover_n = 32768; @64k 2.40×.
  peak MB TF {4k:49.5 … 64k:561.7} vs Prizma 17.9 ∀n → mem crossover @4096; 31.3× @64k. analytic KV/state 28.4→455×.
big (state 1,310,720 floats): per-step ms TF {4k:8.89, 8k:9.12, 16k:12.30, 32k:20.79, 64k:38.90}
  vs Prizma {4k:14.04, 8k:14.02, 16k:14.09, 32k:14.00, 64k:13.87} → crossover ~32k; @64k 2.80× (measured).
  peak MB TF {4k:491 … 64k:4532} vs Prizma 232 ∀n → 19.5× @64k.

## Length-extrapolation (frontier) — gpu_lengen.json (induction, train L=64, eval 64/128/256/512, 3 seeds)
median acc by eval length (1×/2×/4×/8× train length):
- TF(RoPE):    64:1.0, 128:0.4849, 256:0.0874, 512:0.0405   → retention(512/64)=0.041
- Prizma-quad2: 64:0.998, 128:0.9663, 256:0.7197, 512:0.397  → retention(512/64)=0.398  (≈10× TF)

## d208 width-FLOP-match control — gpu_bench.json p2c (2.18M params, ~2.5× the matched count, 3 seeds)
- TF-flopmatch-width-d208L4H4: 0.0224, 0.0221, 0.0220 → 0/3, median 0.0221 (CI ±0.0002, deterministic fail)
- companion depth match d128L9H4 (p2b, 1.85M): 0.021, 0.964, 0.020 → 1/3
- Interpretation: optimization-confounded (LR-transfer; same-depth-narrower d128L4H4/857K SOLVES). EXCLUDED from verdict.

## Param-matched MQAR headline (B1) — gpu_bench.json p2 (d128L4H4 ~860K, 3 seeds)
- TF: 0.9787, 0.9995, 0.9998 (+s3 0.986, s4 0.9995) → median 0.9995
- Prizma-quad2: 0.9991, 0.9983, 0.9955 → median 0.9983 ; Prizma-none: 0.959, 0.958, 0.948 → median 0.9583
- param-eff (B1b, d64L2H2 130K): quad2 0.999/0.997/0.935 (3/3) vs TF 0.016 (0/3); rand_linear 0/3, none 0/3
