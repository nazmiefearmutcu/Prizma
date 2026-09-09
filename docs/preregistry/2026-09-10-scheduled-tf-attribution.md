# PR-2026-09-03-17 — scheduled-TF attribution control (does the lr schedule or the tissue carry the boundary-damage advantage?)

**Lane:** CLAIM (attribution) · **Status:** REGISTERED 2026-09-10 02:1x (frozen before any run) · **Written:** 2026-09-10
**Provenance:** PR-16 CLAIMED (the repaired column's B-boundary damage beats the plain
WINDOW-TF control's by +0.3685, CI [0.27, 0.47]). PR-10 established that the backbone-lr
schedule alone repairs ROUTING; the open attribution question: **is the boundary-damage
advantage carried by the lr schedule alone (which any transformer could also use), or does
the tissue contribute beyond it?** This is the control the attribution claim needs before
the mechanism is named.

## 1. Claim under test (attribution)

A WINDOW-TF that adopts the SAME schedule (backbone lr ×0.25 on every post-A block) —
SCHED-TF — is compared against (a) the plain WINDOW-TF (PR-15 cells) and (b) the repaired
column (PR-14 EX cells). Three mutually exclusive attributions, decided by the two
pre-fixed CIs below.

## 2. Protocol

Stream, slices, corpora, seeds 0–4: EXACTLY PR-14/PR-15. SCHED-TF: the PR-07′ WINDOW-TF
config VERBATIM + the L1 schedule (backbone lr 7.5e-4 on blocks B–E, 3e-3 on A). No
tissue (the control stays routing-free — that is the point).

## 3. Comparisons (exact; n=5; Welch; t_isf UPPER-TAIL)

- **C1 (schedule effect on the control):** delta = mean(PLAIN-TF damage) − mean(SCHED-TF
  damage), damage = bpc_B(post_C) − bpc_B(post_B). CI reported; CI lower ≥ 0.25 ⇒ the
  schedule alone reduces the control's boundary damage.
- **C2 (the tissue's residual):** delta = mean(SCHED-TF damage) − mean(COLUMN damage)
  (COLUMN = PR-14 EX cells reused, disclosed). CI reported; CI lower ≥ 0.25 ⇒ the tissue
  carries damage protection BEYOND the schedule.
- Both CIs are ATTRIBUTIONS, not pass/fail gates: the outcome is the pair
  (C1-attribution, C2-attribution), reported verbatim. Descriptive: SCHED-TF's full
  trajectories + its revisit behavior (bpc_B postD vs postC).

## 4. Pre-committed attribution table (both CIs two-sided 95%; "gap" = the named delta)

- C1 gap established AND C2 gap ~0 (CI straddles 0 or upper < 0.25) ⇒ **SCHEDULE-CARRIED**:
  the boundary-damage advantage is the lr schedule; the tissue's role is the routing
  ledger/recovery — the comparison record names the schedule as the active ingredient.
- C1 gap ~0 AND C2 gap established ⇒ **TISSUE-CARRIED**: the tissue protects beyond any
  schedule a plain LM could adopt.
- BOTH established ⇒ **COMPOUND** — schedule + tissue each contribute; the ledger records
  the split.
- NEITHER established ⇒ **UNRESOLVED at n=5** — the n=10 seed extension (seeds 5–9, new
  SCHED-TF training) is PRE-AUTHORIZED as the next iteration's first action.

## 5. Budget & runner

5 cells ≈ 12–15 min CPU. Runner: seq/windowtf_manyblock.py gains a guarded
`--post-a-lr` flag (default None = byte-identical PR-15 behavior; fingerprint includes
it) writing to its OWN ledger dir `windowtf_sched_PR-2026-09-03-17`. Smoke on CPU first.

## 6. Self-audit

One knob (the schedule, PR-10's frozen dose), one new arm, two CI comparisons, and a
pre-committed attribution table that cannot be cherry-picked post hoc. The baselines are
the registered artifacts themselves. What would change our mind: only dated addenda.
