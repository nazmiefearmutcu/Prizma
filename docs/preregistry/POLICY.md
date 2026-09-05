# Pre-registration policy — the two-lane registry

> **The rule in one line:** exploratory work is free, but it is never a claim; a claim-grade result
> must trace to a pre-registration written *before* the run, with exact bars, arms, seeds and
> statistics, registered in [`INDEX.md`](INDEX.md), and frozen once results exist.

This repo's culture is pre-registered falsifiable bars and honest limits. This document makes that
culture mechanical instead of aspirational. Companion policy: [`docs/RETENTION.md`](../RETENTION.md)
(raw artifacts are never deleted — a pre-registration you cannot audit is worth nothing).

## Why

Two already-paid tuitions motivate every rule below:

* **B4 char-LM**: the spec pre-registered "BOTH corpora, ≥3 seeds (≥5 for the closest leg)". What
  was delivered was one corpus at n=2, and the deviation sat in the verdict instead of the fine
  print. A pre-registration that can be reinterpreted after seeing the results is decoration.
* **campaign 2026-06-08**: a `--smoke` run silently supplied seeds and LR sweeps to a "powered"
  campaign. The fix is structural (seed+fingerprint resume keys, separate smoke/campaign files),
  but the *protocol* fix is the registry: what a run is allowed to claim must be written down
  before the run, not reconstructed after it.

## The two lanes

### LANE-EXPLORATORY

* Purpose: generate and motivate new pre-registrations. Probes, smoke runs, plot-driven digging,
  "what happens if" analysis.
* **NEVER becomes a claim.** No exploratory number may be cited in the README, a report abstract, a
  comparison table, or any external statement as evidence about a mechanism. The strongest allowed
  wording is "in an exploratory run (unpowered, post-hoc), we observed X — see PR-…".
* Results live under `results/exploratory/` (separate from campaign ledgers, same spirit as the
  smoke/campaign file separation).
* When an exploratory finding looks real, the correct output is a *new* LANE-CLAIM pre-registration
  with fresh falsifiable bars — on data/configs the exploration did not select on, whenever
  feasible. Exploration is the motivation section of the next pre-registration, not the result
  section of a claim.

### LANE-CLAIM

A claim-grade pre-registration is written **BEFORE the run** and contains, exactly:

1. **The bar(s)**: the falsifiable pass/fail criterion(ia), numeric, with the decision rule
   (including what happens on a FAIL — a pre-committed pivot or retirement, e.g. owner decision 4
   on surprise-gating).
2. **The arms**: every comparator, incl. controls (constant/random/none where a mechanism is
   claimed), and the param/FLOP-matching scheme.
3. **The seeds**: how many, which, and the stopping rule (no "add seeds until significant").
4. **The statistics**: the exact test, margin, alpha/power, and the multiple-comparison stance.
5. **The artifact plan**: which ledger(s) the run writes and how raw records are retained
   (see [`docs/RETENTION.md`](../RETENTION.md)).
6. **Honest-limit notes**: what will NOT be claimed even on a PASS.

It gets an entry in [`INDEX.md`](INDEX.md) (status `REGISTERED`) **before** the run starts.

## Immutability rule

A LANE-CLAIM document **cannot be edited after results exist**. If the protocol genuinely needs
changing — a discovered confound, an infeasible bar, a power correction — the change is appended as
a **dated addendum** (a new section `## ADDENDUM YYYY-MM-DD: <reason>`, and the INDEX row notes
`AMENDED`), never a silent rewrite of the registered text. The pre-results text stays byte-identical
and auditable in git history. Breaking a bar is an honest result; quietly moving a bar is not.

Status lifecycle in INDEX.md: `IN-WRITE` → `REGISTERED` (committed, run may start) → `ANALYZED`
(outcome recorded per the bars, whatever they are) → `CLAIMED` / `RETIRED` / `NEGATIVE` (decision
applied). A row may also go `ABANDONED` (run never executed; say why).

## File conventions

* One document per pre-registration: `docs/preregistry/YYYY-MM-DD-<slug>.md`.
* Every document names its lane explicitly at the top (LANE-CLAIM docs must, LANE-EXPLORATORY
  motivation notes may also live here if they motivate a registration).
* Every LANE-CLAIM run's verdict text cites its INDEX id and artifact path.
