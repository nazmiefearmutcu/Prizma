# Survey 1 — Mechanism rung: lever inventory, integration points, harnesses, frozen bar inputs

Read-only survey of `C:\Users\Kullanıcı\Prizma` @ HEAD `3a0e4ca` (tree clean), 2026-09-11.
All anchors are `path:line` at this tree. No source file was modified; the only write is this report.
Legend: OFF-ID = default leaves the shipped path bit-identical (guarded branch, tests claim it).

## 1. Lever inventory (implemented knobs only)

### 1a. `src/prizma.py` (numpy PC-experts; E1 substrate)
| lever | anchor | default | semantics (one line) | pinned by |
|---|---|---|---|---|
| `route` | 260 | True | False = no routing, one monolithic local learner (ablation) | — (implicit in E1 runs) |
| `z_novel` | 255 | 5.0 | per-expert novelty z over recon precision (mu+z·sigma) | — (shipped E1) |
| `consolidate` / `omega_consol` | 259 / 243,261 | True / 3.0 | freeze-on-commit pairing; sets `omega` at freeze | test_continual_baselines.py |
| `freeze_min_seen` | 290 (use 879,943,1044) | 0 | freeze-immaturity veto: no freeze below N seen | test_interleave_levers.py |
| `dynamic_vigilance` | 291 (512–535, 765–772) | 0.0 | novelty-EMA-scaled vigilance threshold, clamped [0.25,4] | test_interleave_levers.py |
| `hot_young` | 292 (501–510, 578–582) | 0.0 | metaplasticity: lr ×(1+hot·e^-n/200) for young experts | test_interleave_levers.py |
| `route_stat` | 293 (774–797) | "batch_mean" | "sample_top" = novel-fraction statistic per sample | test_interleave_levers.py |
| `train_granularity` | 326 (807–812) | "batch" | "sample" = per-sample routed training (G1) | test_granularity_levers.py |
| `session_window` | 327 (813–817) | 0 | W>0 = window-level decisions (G2); ⊕ G1 exclusive | test_granularity_levers.py |
| `replay_passes`/`replay_items` | 371–372 (657–706) | 0 / 256 | G3a generative replay-before-freeze (buffer-free) | test_replay_levers.py |
| `probation` | 438 (957–1050) | False | G3b probationary commit = live-revisit until floor-exit | test_probation_lever.py |
| `m_max` | 478 (709–763) | 0 | bounded pool; Policy A recruit-by-eviction (PR-05) | test_mmax_lever.py, test_expert_economy.py |
| Expert analog knobs `weight_bits/act_bits/noise_in/act/weight_std` | 66–69 (117,134,142,150) | None/0.0 | weight/act quantization + noise (neuromorphic emulation) | — (no dedicated pin) |
| `n_settle_steps/eta_settle/langevin_temp` | 244–246 (158–214) | 0/0.1/0.0 | PC settling + Langevin sampling (0 = one-pass) | test_analog_neuromorphic.py (langevin) |

State fields (not knobs): `Expert.omega` 82, `mu/var` 87–88, `committed/frozen` 80–81, `n_seen` 83,
G3a `h_mean/h_var` 96–97, G3b `prob_since_hit` 103, `route_log` 263, `_nov_fast/_nov_slow` 296–297, `eviction_log` 479.
Note: `eta_c` (243,261) and `omega` are written but NEVER read by `_train_expert` (560) — the
docs/Prizma.md §3 `α=α0/(1+ω)` rule is not active in the prototype; reawakening (docs:139) exists only in
legacy `src/learners.py:462`.

### 1b. `seq/prizma_seq.py` (`PrizmaSeqConfig` / block)
| lever | anchor | default | semantics | pinned by |
|---|---|---|---|---|
| `precision_gate` | 47 (354–371) | "input" | write gate source; `'surprise_norm'` = PR-01 A4 frozen formula | test_surprise_gate.py, test_surprise.py |
| `write_mode` | 60 | "delta" | "delta" erase-write vs "additive" linear-attn write | test_fused.py |
| `use_workspace` / `use_window` | 61 / 62 | True | delta-state head / local window head toggles | — (implicit; step-guards) |
| `route_readout` | 63 (352–353) | True | False = fixed query (kills content recall, B6) | — |
| `feat_map`/`feat_n2`/`feat_rank` | 66,80,82 (322–340) | "none" | 0-param quadratic key map (capacity) | test_featmap.py |
| `out_gate` / `state_norm` | 85 / 86 | False | RWKV-7-style output gate / state RMSNorm | test_landscape.py |
| `banded_window` | 87 (400–416) | False | O(T·w) window kernel, exact-equal off | test_window.py |
| `decoupled_gate` | 88 (377–380) | False | decouple erase βe from write βw | test_decoupled.py |
| `n_delta` | 89 (426–449) | 1 | DeltaProduct sub-steps per token | test_deltaproduct.py |
| `surprise_gate/mode/seed` | 91–93 (452–475, 599–606) | False/'norm'/1234 | g=1+tanh(||eps||) write scaling (Lever A) | test_surprise_gate.py |
| `inctx_lr` | 98 (381–387, 593–598) | False | per-value-channel in-context LR (Lever G) | test_inctx_lr.py |
| `dropout` | 102 | 0.0 | residual+embedding dropout (0.0 = identity) | test_dropout.py |
| `state_bits` / `write_noise_std` | 122 / 127 (492–509, 626–628) | 0 / 0.0 | deployment emulation on step() ONLY (chunk kernel untouched) | test_analog_lever.py |
| `surprise_gain` / `surprise_ema_lambda` | 142 / 148 (607–619) | 1.0 / 0.02 | A4 gate gain (frozen on seed 900) / causal EMA rate | test_surprise_gate.py |

Block anchors: `__init__` 252; `_encode` 342; `forward` 418 (chunked_delta call 469–475);
`step` 513 (state tuple 523; write 559–627; return 643); `PrizmaSeqLM.init_state` 679 (6-slot state 685–692);
`step` 696. Kernel: `seq/delta.py::_delta_reference` 122, `chunked_delta` 370.

### 1c. Fusion runner `seq/prizma_lm_claim.py` (+ tissue `seq/fusion_probe.py`)
| lever | anchor | default | semantics | pinned by |
|---|---|---|---|---|
| frozen tissue constants `E_POOL/M_MAX/FREEZE_MIN_SEEN/Z_NOVEL/H_SMALL/H_SHARED/FLOOR_EMA` | 147–154 | 4/4/300/5.0/64/256/0.05 | pool, cap, veto, z, widths, floor EMA | test_prizma_lm_runner.py |
| `--trunk-lr-c` | 1397, 235, 238, 845–853 | None | PR-11 block-C backbone lr (registered dose 7.5e-4, guarded) | test_pr11_repaired_flagship.py |
| `--domain-exclusion` | 1410, 834–842, 607–622 | False | PR-13 L2: protected C slots, recruit suppressed | test_pr13_exclusion_flagship.py |
| `--seed-offset` | 1402, 1043 (canary 1197–1207) | 0 | PR-18 fresh-seed replication; LR canary | test_manyblock_runner.py, test_prizma_lm_runner.py |
| `--ledger-dir` | 1406, 198–201 | None | PR-11 own-ledger redirect | test_prizma_lm_runner.py |
| `--force-smoke-path` (refusal) | 204–221 | off | smoke/powered ledger separation | test_prizma_lm_runner.py |
| `model.floor_freeze` consumer | fusion_probe.py:259–267 | absent | PR-09 floor pinning (set by seq/floorfreeze_claim.py) | test_floorfreeze_runner.py |
| `model.domain_protect` consumer | 607–622 | absent | PR-13 protected-slot redirects | test_pr13_exclusion_flagship.py |
| bars B1/B2/B3 margins | 166–171 | 0.05/0.10/0.5/20/0.10 | frozen verdict constants | test_prizma_lm_runner.py |

Tissue anchors: `PCExpertHead` fusion_probe.py:99–114 (Wenc/Wdec only); `_expert_train` 230–268 (one AdamW
step; post-update floor EMA 261–267 is the ONLY slow state); `route_batch` 173–227; `FusionLM` 117–150.

## 2. NEW-mechanism integration points

### (a) Benna-Fusi fast+slow cascade in the fused LM protocol
Motive (measured): PR-21 tissue recovery 0.819 vs plain 1.147 bpc (`results/recovery_attr_PR-2026-09-03-21/powered.json`);
PR-16 tissue damage 0.643 vs control 1.012 (delta +0.368, CI [0.270,0.467]).
- Tissue plug (lowest risk): `seq/fusion_probe.py:99–114` add per-param slow copy (`W_slow`); `_expert_train`
  230–268 fold the cascade after the AdamW step (there is a natural precedent for a slow variable: floor
  EMA 261–267). Existing state: only `Wenc/Wdec` params + `mu/var` floors (117–138). Nothing to remove.
- Delta-state plug (higher risk): `seq/prizma_seq.py` state tuple gains a slow S at 523/643/685–692;
  writes at 627; the chunk kernel `seq/delta.py::chunked_delta` 370 (WY/UT) would need re-derivation — the
  repo-documented precedent is `state_bits` wired to `step()` only (`prizma_seq.py:110–121`).
- Trunk plug: `seq/prizma_lm_claim.py::train_pr08` AdamW 649–650; the only existing multi-timescale control is
  the trunk_lr_c schedule 845–853 — a trunk cascade duplicates the PR-17 attribution ("SCHEDULE-CARRIED").
- Existing src consolidation machinery to port (no seq analog, disclosed at `prizma_lm_claim.py:70–71`):
  omega 82, replay-before-freeze 657–706, probation 957–1050, novelty EMAs 512–524.
- Must add: guarded config field + fingerprint entry (cfgsig 1216–1230), default-off bit-identity test, and
  step()==forward() guard if the delta state changes (pattern: tests/test_analog_lever.py).

### (b) Expert economy (docs/EXPERT_ECONOMY.md §3) in `src/prizma.py`
- merge: absent. Would be a new `Prizma` method over `Expert` weights (72–79), `mu/var` 87–88,
  `route_log` 263, frozen flags 81; offline sim already exists `experiments/expert_economy.py:313`
  (`simulate_merges`); spec §3.1 (docs/EXPERT_ECONOMY.md:153–180); guard: FA matrices never averaged.
- prune: absent. Slot-release machinery to mirror: `_evict_slot` 745–763 (but it keeps committed=True wiring
  and is eviction-driven); spec §3.2 (182–193); zero-routing source = `route_log` 263.
- abstain: absent. `route_for_inference` 540–548 always argmins; spec §3.4 (213–228) + measured calibration
  §2.3 (133–147: batch z=4 → 0% false-NOVEL, 100% detection). Pieces already present: `mu/var` 87–88,
  `z_novel` 255, `_threshold` 765, `_recon_matrix` 537 (batch-level `min_m z` mean).
- bounded-M Policy A already implemented + claimed: 709–763, `m_max` 478 (PR-05).

### (c) src machinery with NO seq-fusion analog (anchors)
| mechanism | src anchor | seq side status |
|---|---|---|
| omega slow consolidation | Expert 82; writes 882,946,1049; ctor 243 | absent; and never read even in src (see §1a note) |
| reawakening (`ω -= κ·relu(conflict)`) | not in src/prizma.py; legacy learners.py:462 | absent everywhere current |
| dynamic_vigilance | 291, 512–535, 765–772 | fixed `Z_NOVEL=5.0` (`prizma_lm_claim.py:150`; fusion_probe route 513) |
| hot_young metaplasticity | 501–510, 578–582 | absent (tissue uses flat lr) |
| freeze_min_seen veto | 879, 943, 1044 | PRESENT as floor-maturity veto (149, 515–528) |
| G3a replay / G3b probation / novelty EMA | 657–706 / 957–1050 / 512–524 | absent in fusion tissue |
| (reverse) floor_freeze / trunk_lr_c | — | seq-only (`fusion_probe.py:259`; `prizma_lm_claim.py:845`) |

## 3. Measurement harnesses (existing runners, measured CPU costs on this box)

Cost basis — per-cell `wall_s` read from the ledgers: flagship PR-18 (`results/prizma_lm_PR-2026-09-03-18/powered.json`):
PRIM-LM 72 s, FROZEN-TRUNK 59 s, SHARED-HEAD 72 s, FROZEN-CHECKPOINT 22 s, FORCED-RECRUIT 71 s (mean over 10 cells);
PR-08 smoke (`results/prizma_lm_PR-2026-09-03-08/smoke.json`): 2.5–7 s/cell. manyblock PR-14: 111–135 s/cell, smoke
13–15 s. WINDOW-TF PR-15: 139–145 s; PR-17: 98–101 s. All at `BENCH_THREADS=8`.

| candidate | command | smoke | n=2 probe | n=5/n=10 claim | risks |
|---|---|---|---|---|---|
| (a) cascade | `python seq/prizma_lm_claim.py --powered-cpu --ledger-dir prizma_lm_PR-<new>` | `--smoke` (~30 s, 5 cells) | seeds via a new `--seed`/probe flag or `manyblock_probe.py` | 25 cells ≈ 25 min + LR-sel (9 cells) | default-off bit-identity + fingerprint (1216–1230); chunk-kernel re-derivation if delta state changes (110–121); trunk cascade confounds with PR-17 schedule; FROZEN-arm canaries 948–1003 abort on mismatch |
| (a) recovery read-out | `python seq/manyblock_claim.py --powered-cpu --ledger-dir manyblock_PR-<new>` then `python seq/recovery_attribution.py --powered-cpu` | 13–15 s/cell | `seq/manyblock_probe.py` | 2 arms × 5 ≈ 21 min; attribution = seconds | PR-21 sources are FROZEN registered artifacts (recovery_attribution.py:50–54) — a new bar must cite a fresh ledger, not mutate sources |
| (b) economy | `python experiments/run_continual.py` (E1_main 122; 10 seeds) / instrumented probe `python experiments/expert_economy.py` (main 621; whole analysis ≈ 8 s, doc line 17) | same scripts are n=3 exploratory (doc §6.1) | — | seconds–minutes (d=24) | settle-0-only merge feasibility (EX doc 269–271); FA mats never averaged (171–180); prune = replay-free deletion; abstain calibration one generator family (274) |
| (c) dynamic_vigilance in seq | port into `route_pr08` novel test (`prizma_lm_claim.py:513`) then flagship command above | ~30 s | — | ~25 min | PR-09 measured floor pinning does not move fracs (`results/floorfreeze_PR-2026-09-03-09`) — likely inert; low expected value |

Claim-integrity rules that apply to every new lever: ledger separation (204–221), config fingerprint in every
cell (1216–1230), raw archive before verdict (1292–1294), INCONCLUSIVE never PASS (321–337), and the registered
dose guard (238–249).

## 4. Frozen bar inputs (exact, for a new pre-registration)

### PR-21 recovery per seed — `results/recovery_attr_PR-2026-09-03-21/powered.json` (verdict)
| arm | s0 | s1 | s2 | s3 | s4 | mean |
|---|---|---|---|---|---|---|
| COLUMN (PR-14 EX) | 0.7804 | 0.7925 | 0.8810 | 0.8142 | 0.8294 | **0.81949** |
| SCHED-TF (PR-17) | 0.8314 | 0.8398 | 0.9214 | 0.8517 | 0.8532 | **0.85951** |
| PLAIN-TF (PR-15) | 1.1014 | 1.1303 | 1.1497 | 1.0786 | 1.2754 | **1.14709** |

C1 (SCHED−COLUMN) delta −0.04002 CI [−0.09485, +0.01481] → PARITY. C2 (PLAIN−COLUMN) delta **−0.32761**
CI [−0.42206, −0.23315] → ESTABLISHED (TISSUE-COSTS-RECOVERY). Frozen attribution rule: gap needs CI∌0 AND
|point| ≥ 0.10 (`recovery_attribution.py:90–96`).

### Damage at the C boundary (B_postC − B_postB), same three ledgers
| arm | s0 | s1 | s2 | s3 | s4 | mean |
|---|---|---|---|---|---|---|
| COLUMN | 0.6006 | 0.6383 | 0.6711 | 0.6411 | 0.6656 | 0.6433 |
| SCHED-TF | 0.6026 | 0.6323 | 0.6782 | 0.6349 | 0.6490 | 0.6394 |
| PLAIN-TF | 0.9687 | 0.9944 | 1.0082 | 0.9398 | 1.1478 | 1.0118 |

PR-16 head-to-head: delta +0.36849 CI [0.27037, 0.46660] PASS vs 0.25 bar
(`results/boundary_damage_PR-2026-09-03-16/powered.json`).
Fractions: only COLUMN carries a routing ledger — PR-14 P2 frac_per_seed [0.975, 0.98125, 0.95625, 0.975, 0.9671875],
mean 0.97094; WINDOW-TF arms have NO fraction (no committed pool) — a new bar must not demand one of them.
PR-14 frozen bars: P1 margin 0.10, P2 frac ≥ 0.5, P3 margin 0.05, G1 allowance −0.10, G2 margin 0.05
(`results/manyblock_PR-2026-09-03-14/powered.json` meta.bars; P1 mean −0.81949, P3 mean −0.32264).

### PR-18 pooled fresh verdict (seeds 5–14, n=10) — `results/prizma_lm_PR-2026-09-03-18/pooled_fresh_verdict.json`
| bar | value | CI | status |
|---|---|---|---|
| B1 retention-A | mean −0.27586 | [−0.28930, −0.26243] | PASS (p_holm 1.1e−12) |
| B2 adaptation-C | Welch delta +0.68264 | [+0.43249, +0.93279] | PASS (p_holm 2.6e−4) |
| B3 routing ledger | frac_mean 0.96548 ≥ 0.5; forced_cost +0.00013 ≤ 0.10 | per-seed fracs 0.8625…0.9906 | PASS |
| B4 (reported) | PRIM +0.5389 / FORCED +0.5390 / FROZEN-TRUNK +1.9906 / SHARED-HEAD +1.9258 | — | — |

## 5. Ranking (CPU-feasible tonight)
1. (a) tissue fast+slow cascade — directly attacks the two CI-established deficits (PR-21 recovery gap,
   PR-16 damage) on the strongest existing harness (~25 min flagship, 21 min manyblock, seconds attribution).
2. (b) abstain (+prune) economy in `src/prizma.py` — cheapest to implement and already calibrated
   (EXPERT_ECONOMY §2.3), closes a documented audit gap; value is open-world robustness, not the LM ladder.
3. (c) omega/hot_young/dynamic_vigilance port into seq — lowest expected value: PR-09 showed the floor/
   threshold family inert for routing fractions; omega in src is currently a dead write.
