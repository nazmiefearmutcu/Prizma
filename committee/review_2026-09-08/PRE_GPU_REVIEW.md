# PRE-GPU REVIEW — PR-2026-09-03-01 / -02 / -07 / -04 / -08 + levers + ship script

**Reviewer:** independent pre-GPU review agent (read-only; CPU tests/smokes only)
**Date:** 2026-09-08 · **Tree reviewed:** `C:\Users\Kullanıcı\Prizma` @ HEAD `2520eb5` ("fix(seq): correct provenance wording…"), tree clean, all local.
**Scope:** the 5 GPU-bound claim runners + 9 default-off levers + `tools/ship_prizma.py`, each against its frozen protocol doc, before the A100 campaign executes them.

---

## VERDICT SUMMARY

| Severity | Count | One-line |
|---|---|---|
| **C** (corrupts a verdict / wastes GPU) | **0** | — |
| **H** (misleads a reader/maintainer) | **2** | false "owner-approved" provenance written into the PR-08 powered ledger; PR-07′ registered Holm never implemented (verdict stands on decisive margins) |
| **M** (style/robustness) | **16** | Holm `p_adj`-vs-`reject` edge, unfingerprinted canary cells, ship-script cwd/dirty-tree footguns, dead code, import fragility, budget under-disclosure |

### Recommendation: **SHIP** — conditional on one 2-line commit (H-1) landing before the campaign runs.

No finding corrupts any registered verdict math or wastes GPU hours on the expected path. The three powered runners' decision logic, fingerprints, refusals, freeze semantics, and lever off-identity all check out against their frozen docs (details below). H-1 must be fixed first because the PR-08 powered ledger would otherwise permanently record a provenance claim the project itself has ruled false. H-2 is post-hoc (PR-07′ already claimed; addendum, not a blocker).

---

## FINDINGS TABLE

| ID | Sev | File:Line | Why | Minimal fix |
|---|---|---|---|---|
| **H-1** | H | `seq/prizma_lm_claim.py:41` and `:142` | The C-range repair is still labeled **"owner-approved, 2026-09-07 session"** in the module docstring and in `C_RANGE_REPAIR_NOTE` — the string written **verbatim into the powered ledger** (`meta["c_range_repair"]`) and the report. HEAD `2520eb5` (whose own message says "maintainer addendum, not an owner decision") fixed only the `C_OFFSET` comment (line 103); the two ledger-facing strings still assert an owner decision that never happened (no user access that session; the doc's Addendum records a *maintainer correction*). A registered artifact carrying a false approval claim is exactly the honesty failure this repo's POLICY exists to prevent. | Replace both occurrences of "(owner-approved, 2026-09-07 session…)" with "(maintainer addendum 2026-09-08, docs/preregistry/2026-09-08-prizma-lm-flagship.md)". Commit before `--powered`. |
| **H-2** | H | `seq/blockdrift_claim.py:210-231, 258` vs `docs/preregistry/2026-09-05-blockdrift-bar6.md` §4 | The registered adaptation bar says "two-sample Welch, **Holm over the two primaries**"; the runner computes raw Welch CIs and **never calls Holm**. Mitigations: the claimed PASS margins are enormous (FGT CI upper −0.119 vs ≤0.05; adaptation diff CI [−3.420, −2.936] vs ≤−0.10), so Holm(2) cannot flip the outcome; the CI bars are themselves conservative. Still: the registered statistic was not the computed statistic. | Dated addendum: disclose the CI-based rule, and show both raw one-sided p-values are ≪ 0.025 so Holm(2) is a no-op. No re-run. |
| M-1 | M | `seq/prizma_lm_claim.py:645-647` | FORCED-RECRUIT raises `RuntimeError` if the pool has no free slot at C start. Deterministic (PRIM/FORCED share the A/B trajectory), fail-loud, ledger stays valid — but if B ever recruits 3 experts the campaign crashes at that cell on every resume and the GPU session dies mid-campaign. | None post-freeze (any fallback = protocol change). Disclose in the pre-run notes; if it fires, resume-safe + dated addendum. |
| M-2 | M | `seq/prizma_lm_claim.py:900-905` | Claim-cell fingerprint covers slices/stream/tissue/lr but **not the bars constants** — while `B3_WINDOW_BATCHES` is applied at *train* time (`_count_boundary`, line 391). A doc-driven window change between resume attempts would mix cells counted at different windows. | Add `res["meta"]["bars"]` to the claim-cell cfgsig payload. |
| M-3 | M | `seq/prizma_lm_claim.py:599-603` | `run_routed` docstring: "A and B phases are identical across the three" — false for FROZEN-TRUNK (its B is frozen by design; the LR-rule disclosure even says so). Smoke confirms PRIM/FORCED share A bit-identically (A_preB=4.942 on all three). | Reword: "identical across PRIM-LM and FORCED-RECRUIT; FROZEN-TRUNK shares phase A only." |
| M-4 | M | flagship doc §4 vs runner | §4 mentions "WINDOW-TF / FROZEN-CHECKPOINT descriptive"; the runner implements no WINDOW-TF arm (§3's five arms are what shipped). No bar gates on it. | Optional one-line doc note; otherwise ignore. |
| M-5 | M | `seq/surprise_claim.py:217` + `seq/stats.py:215-244` | Win test uses `h["p_adj"] < alpha`; `holm_correction`'s `p_adj` is not made monotone (no running max). Demonstrated: p=[0.0084, 0.0095,…] → p_adj=0.0475 < 0.05 while step-down `reject=False` for all (the first p already failed at 0.0504). Divergence only in a knife-edge band and only ever *toward SURVIVES*; the doc's literal wording ("Holm-adjusted p < 0.05") arguably licenses it, and the analog runner uses `h["reject"]` correctly. | Use `h["reject"]` in `surprise_verdict`, or make `holm_correction`'s `p_adj` a running max (fixes every consumer). |
| M-6 | M | `seq/surprise_claim.py:445-446` | The integrity canary is called **without a fingerprint**, so its internal `sweep_then_seeds` runs at `cfgsig=None` → legacy key-only reuse of `negctrl.A/B` cells. Inside this registry-scoped ledger with frozen constants the practical risk is low, but it is the one cell family that can silently adopt stale cells. | Thread a fingerprint through `negative_control`. |
| M-7 | M | `seq/surprise_claim.py:319-323` | Gain-selection cells' fingerprint omits the training recipe (`warmup/warmup_frac/min_lr_frac`) that `make_cfg` consumes — a protocol amendment to warmup would leave gain cells reusable while arm cells recompute, so `frozen_a` could derive from stale-recipe accs. | Add `"recipe": recipe` to the payload. |
| M-8 | M | `seq/surprise_claim.py:449-459` | On canary FAIL the runner writes verdict INCONCLUSIVE and exits **before** `archive_run` — the doc's "raw records archived before any verdict" is violated on that one path (retention only; INCONCLUSIVE numbers are never citable). | Move `archive_run` above the canary-exit branch. |
| M-9 | M | prereg §5 table vs `seq/gpu_harness.negative_control` | Doc budgets the canary as "2 runs ~0.2 h"; the implementation (which the doc itself names) sweeps both arms → 2×(5 sweep + 2 seeds) = 14 cells ≈ +1 h on the 10–15 h estimate. Not disclosed. | Note in the pre-run ledger; no protocol change. |
| M-10 | M | `seq/blockdrift_claim.py:231` | Dead, wrong-condition `bar2 = (adapt_ci[0] <= -0.10)` (the real verdict at :258 correctly uses ci[1] for PASS, ci[0] only for INCONCLUSIVE). Misleads a reader of the code. | Delete the dead variable. |
| M-11 | M | `seq/blockdrift_claim.py:234` | `from transformer import Transformer` resolves only when run as a bare script (`sys.path[0]`=seq/). Verified: root import fails, seq-dir-on-path import works. `python -m seq.blockdrift_claim` would train all primaries, then crash at the WINDOW-TF leg **before writing any verdict**. | `from seq.transformer import Transformer`. |
| M-12 | M | `seq/analog_probe_claim.py:114-130, 149-158, 364-389` | Resume caches by file existence with no config fingerprint, and `--seeds/--grid/--difficulties` overrides are accepted unguarded — a re-run with changed knobs silently adopts stale cells (meta.json records what ran). Finished CPU claim (PR-04, NEGATIVE): low risk. | Fingerprint in run filenames, or refuse overrides without `--force`. |
| M-13 | M | `tools/ship_prizma.py:88` | Ships `git ls-tree -r HEAD` with **no dirty-tree check**: uncommitted working-tree edits (e.g., a last-minute addendum) silently do not ship. | Refuse (or warn+confirm) when `git status --porcelain` is non-empty. |
| M-14 | M | `tools/ship_prizma.py:88` | No cwd guard: run from a subdirectory, `git ls-tree -r HEAD` lists only that subtree and `--force` would **replace the entire remote tree with it**. Documented usage is repo-root only. | Assert `os.getcwd() == git rev-parse --show-toplevel`. |
| M-15 | M | `tools/ship_prizma.py:104-115` | No post-ship tree-equality verification (the flowmap-solana recipe this is modeled on had one). Failures are loud, but a final `GET /trees/<sha>?recursive=1` compare is cheap insurance for a one-way force-ref operation. | Add the compare; print TREE EQUAL / mismatch. |
| M-16 | M | `seq/dfrontier_claim.py:573-581` | `verify_frozen_against_ledger` collects `accs` that is never used (cosmetic; the eps recomputation below is the real check and is correct). | Delete the dead loop. |

---

## WHAT WAS VERIFIED CORRECT (the load-bearing positives)

**PR-01 `surprise_claim.py` vs prereg.** Verdict family is exactly the 6 primaries {P1_t, P2_t}, one-sided Welch in the registered direction (`superiority_test(surprise, control)`, H1 mean>), Holm at α=0.05, win = Holm+point-estimate (M-5 edge aside). The SURVIVE rule (wins vs BOTH controls on ≥2 of 3 tasks + reverse guard) is implemented equivalently to §4 — the "guard implied on won tasks" argument is mathematically sound (forward significance at 0.05 forces the reverse p>0.5, same samples/df). NO-TUNING gain selection: seed 900, MQAR-D64 only, lr 1e-3, ascending candidates with strict `>` ⇒ ties→smallest; exploratory results quarantined to the LANE-EXPLORATORY file, never touched by smoke; frozen `a` enters the arm fingerprint. Smoke/powered separation + refusal verified; archive-before-verdict holds on the normal path (M-8 is the one gap); canary first-class with the INCONCLUSIVE branch; MDE checksum pins the `t_isf` upper-tail convention (recomputed live: all four constants match).

**PR-02 `dfrontier_claim.py` vs crosstalk doc.** ε̂ = √((D*−1)·σ₂_law²) exact (`fit_epsilon`). σ₂_law read from the committed instrument artifact and **refused on drift** vs the doc's 5-dp constants (tol 2e-4). Freeze-before-adjudicate: `frozen_predictions` written before any adj cell; resume reuses and **re-derives ε̂ from ledger fit cells, refusing any mismatch** — the doc's "adjudication before freeze" violation is mechanically impossible. Bar = primary agreement ≥3/4 (the doc's exact bar; ±1-grid-step reported, not substituted). K2 checked before K1; K1/K2 hard-stop powered before adjudication; K3 = one pre-pinned fresh-seed re-run {10,11,12} then INCOMPLETE; K4 wording verbatim. Per-cell fingerprints are the tightest of the three runners (leg/arm/D/seed/scale/d_φ/feat_kw/cap/batch/recipe/queries/solve/grid_tag). Smoke exercised freeze→adjudicate→verdict end-to-end (redirected ledger; K2-INCOMPLETE plumbing path correct).

**PR-08 `prizma_lm_claim.py` vs flagship doc.** Slices pinned exactly per the maintainer addendum: A-eval [1.0M,1.1M), C = [1.1M,2.1M), C-ret [0.9M,1.0M), with `pin_slices()` asserting the disjointness invariants (A-eval∩C-train=∅ etc.). The C-ret/A-train overlap is disclosed in-ledger. Arms: PRIM-LM's tissue reuses fusion_probe verbatim — per-expert **local AdamW on detached activations** (`_expert_train`: `h.detach()`, detached base logits, no grad to backbone; backbone step first, whole-batch shared loss); FROZEN-TRUNK freezes with a parameter-identity check; SHARED-HEAD matches the dominant param term (4×64V vs 256V) with both counts recorded; FROZEN-CHECKPOINT aliases post-A evals honestly; **FORCED-RECRUIT forces ALL C batches through the forced slot (early return before any vigilance logic — see Q4)**. Bars: B1 one-sample CI-upper ≤0.05 with correct upper-tail p via flipped hypothesis; B2 Welch advantage ≥0.10 in the correct direction (frozen−prim); **Holm over exactly B1–B2**; B3 = mean frac ≥0.5 AND forced cost ≤0.10 (means, correctly CI-free); B4 reported-not-gated; straddling CI ⇒ INCONCLUSIVE, never PASS. floor-maturity veto (freeze_min_seen=300) and Policy-A eviction (lowest share, tie→highest slot, pinned re-seed formula) implemented in `route_pr08` with full event ledgers. 5-arm smoke passed end-to-end (redirected), ~25 s.

**PR-07′ / PR-04 (lighter pass).** PR-07′: verdict math otherwise correct (H-2 Holm gap and M-10/11 aside); RESET canary per the pre-run addendum; corpora by canonical URL + sha256, ABORT on failure. PR-04: §7 gates implemented faithfully — matched-clean ladder drops difficulty identically for both arms, per-model O(1)-sanity required, both primaries delta ≥+0.05 AND one-sided Welch Holm via `h["reject"]`, retention normalized by each arm's OWN knobs-off accuracy. (Already executed; NEGATIVE recorded.)

**Levers off-identity (task 5).** `seq/prizma_seq.py`: `state_bits`/`write_noise_std` are read **only** in `step()` via `_quantized`/`_noisy`, both early-returning at defaults with no rng and no ops; `forward()`/`chunked_delta` never read them — no off-lever path can alter results. `surprise_norm`: `_encode` returns `beta=None` only under the gate; `chunked_delta` routes to `_delta_reference` only when `surprise_norm` is not None (byte-identical fast path otherwise); `delta.py` implements the frozen formula exactly (`clamp_min(1e-6)`, β_cap after σ, `a·(s̃−1)`, m₁=s₁, erase=write); step() mirrors via shared helpers with the EMA in the state's 6th slot; the one-novel-lever rule is enforced at config, `_encode`, and `_delta_reference` with loud errors, never silent degradation. `src/prizma.py`: all levers default-off behind guarded branches; **all three recruit sites** (`_train_unit` else-branch, `_train_batch_by_sample`, probation dispatch) run the same freeze-veto→replay→advance-or-evict sequence; the probation demote path intentionally skips commit/freeze (documented) and its exit test applies BOTH gates plus replay. No recruit path missing a lever branch.

**`tools/ship_prizma.py` (task 6).** Ships the **committed** object store (`git ls-tree -r HEAD` + `cat-file blob`), so no CRLF drift; gitignored corpora correctly excluded (sha256 committed instead); modes preserved; pull-only access → exit 3 with options (verified logic); dry-run default, `--force` required for the ref PATCH; history rebuild documented. Gaps: M-13/M-14/M-15 — fix before the owner's one-shot publication.

---

## TARGETED QUESTIONS

**Q1 — Holm over the wrong family or wrong tail?** No runner computes Holm over a wrong family or tail. PR-01: exactly the 6 primaries, correct one-sided direction. PR-08: exactly B1–B2 (B3 is means-based and correctly untested). PR-04: the 2 primaries with `h["reject"]` (the standard decision). PR-02: no Welch family by design (registered agreement count). The only Holm issues are H-2 (PR-07′: registered Holm never implemented — cannot flip the claimed result) and M-5 (PR-01's `p_adj<alpha` vs step-down `reject` edge). Tails: every consumer uses `t_isf/t_sf` upper-tail correctly, pinned by the MDE checksum.

**Q2 — frozen eval set shifting between arms?** No. Synthetic runners: `TrainConfig.eval_seed=12345`, dedicated eval RNG, same task+protocol across arms ⇒ bit-identical eval batches. PR-08/PR-07′: fixed char slices pinned in code (PR-08 with disjointness asserts); the C-range repair guarantees B1's premise (A-eval never trained). PR-08's LR selection scores the first 200 **A-train** segments (not held-out) — identical rule per arm-family, per the PR-07′ addendum.

**Q3 — resume replaying stale cells after a doc-driven knob change?** Largely no: all three GPU runners fingerprint claim cells and **hard-refuse** foreign fingerprints (no silent replay); `run_cell` recomputes on mismatch. Four narrow gaps, none silently corrupting: M-6 (surprise canary cells unfingerprinted), M-7 (surprise gain-selection fingerprint omits the recipe), M-2 (PR-08 fingerprint omits bar constants incl. the train-time B3 window), M-12 (analog probe, filename-only, finished CPU claim). PR-02 additionally re-derives the frozen ε̂ from ledger fit cells on every resume and refuses mismatches — the strongest resume integrity of the five.

**Q4 — does FORCED-RECRUIT actually prevent re-routing beyond C-batch-0?** **Yes.** `route_pr08` checks `force_slot is not None and corpus == "C"` **first** and returns `[(force_slot, all B segments)]` for *every* C batch (prizma_lm_claim.py:420-427) — the vigilance/z-test/argmin machinery is never reached for C while forced. A/B calls pass `force_slot=None`, so B routing is untouched. The only failure mode is the missing free slot at C start (M-1, fail-loud).

**Q5 — is the surprise EMA λ=0.02 actually frozen in code?** **Yes.** `PrizmaSeqConfig.surprise_ema_lambda: float = 0.02` (seq/prizma_seq.py:148-150, "FROZEN A PRIORI … never tuned"); the runner pins it explicitly per-arm (`surprise_claim.py:129`); the NO-TUNING selector touches only `surprise_gain` (candidates {0.5,1,2,4}, seed 900, ties→smallest — verified in `select_gain`); `delta.py` receives λ only through the frozen tuple. No runner path can tune it.

---

## RUN EVIDENCE (all CPU, ≤5 min total; ledgers redirected to %TEMP%)

- `pytest -q tests/test_prizma_lm_runner.py tests/test_dfrontier_smoke_refusal.py tests/test_surprise_gate.py tests/test_stats.py` → **85 passed, 1 skipped**.
- `prizma_lm_claim --smoke` (redirected) → all 5 arms end-to-end; A-phase bit-identity across fusion arms (A_preB=4.942 ×3); FORCED frac=0.0 vs PRIM frac=1.0 as expected.
- `dfrontier_claim --smoke` (redirected) → fit→freeze→K2-INCOMPLETE→verdict plumbing correct.
- `surprise_claim --smoke` (redirected) → gain selection + 4-arm sweep completed green.
- `mde_checksum()` recomputed live → all four frozen constants match (upper-tail convention intact).
- Holm edge demonstrated numerically (see M-5).

**Reviewer side effects (disclosed):** the smokes' `archive_run` wrote three new append-only snapshots under `results/runs/` (retention policy forbids deletion, so they were left in place; no existing file was modified; ledgers themselves went to %TEMP%).

---

## PRE-FLIGHT CHECKLIST BEFORE THE A100 SESSION

1. **Required:** commit H-1 (2-line provenance rewording in `seq/prizma_lm_claim.py`). *Blocker for PR-08's ledger honesty only — not for the math.*
2. Recommended (same commit, minutes): M-5 (`h["reject"]`), M-8 (archive before canary exit), M-10/M-11 (blockdrift dead var + import), M-13/M-14 (ship-script guards).
3. Post-hoc (no re-run): H-2 disclosure addendum for PR-07′; note M-9's canary cost in the PR-01 ledger at run time.
4. Accept and pre-disclose M-1 (FORCED-RECRUIT pool-full crash = fail-loud, resume-safe).

**FINAL CALL: SHIP (conditional on item 1). Zero C findings; the registered verdict math, freeze/fingerprint/refusal machinery, and lever off-identity are sound across all five runners.**
