# Lane 5 — Use-it-or-lose-it prune (`prune_slots`) — report

**Date:** 2026-09-11 · **Box:** Windows, venv Python, numpy-only · **HEAD at proof time:** `3f28f8e`
(+ lane edits only: `src/prizma.py`, `tests/test_expert_prune_lifecycle.py`)

**Status: COMPLETE.** The §3.2 slot-release lifecycle (survey-1 §2b, `docs/EXPERT_ECONOMY.md` §3.2)
is implemented as a METHOD-ONLY, opt-in operation; no constructor knob, no automatic cadence, no
default-path behavior change. All targeted suites + the full `tests/` suite are green.

---

## 1. Delivered

### `src/prizma.py` (pure insertion after `_evict_slot`, +113/−0; nothing else touched)
- `_release_slot(m, window, kept_by_guard=None)` (line 815): resets slot `m` to a FRESH `Expert`
  built exactly as `__init__` / `_evict_slot` build it — deterministic seed formula
  `self._base_seed + 100*(m+1)` + `**self._expert_kwargs` — which clears committed / frozen /
  omega / n_seen / init_recon / the precision floors (mu=1e9, var=1.0), the G3a h-statistics and
  the G3b probation counter. `route_log[m]` is zeroed (the only per-slot bookkeeping the recruit
  path clears), and one audit record is appended to `prune_log`. `prune_log` is created lazily on
  first release (no `__init__` change → the operation stays method-only).
- `prune_slots(window=None)` (line 840): public lifecycle operation. Releases every expert that is
  (a) frozen and (b) zero-routed in the window; returns the ascending list of released slots.

### `route_log` / `window` semantics (the documented conservative reading)
The shipped `route_log` is a **lifetime cumulative per-slot ledger** (`np.int64`, sample counts;
every `route_log[m] += n` site adds samples, no timestamps, no per-event history). It therefore
**cannot decide** "zero routing in the last `window` routing events" exactly. The conservative
SOUND mapping was used, documented verbatim in the method docstring:
- a slot is releasable **only when its recorded count is ZERO** — zero over the entire recorded
  history is a superset of any finite window, so every release is valid for every `window` and a
  false prune is impossible by construction;
- **any nonzero count protects** the slot (it may include routing inside the caller's window; it
  cannot be proven stale);
- the predicate is consequently **window-independent BY DESIGN**; `window` is still **required**
  (`None → ValueError`), validated (positive `int`; bool/float/str/≤0 rejected) and recorded in
  `prune_log` so the operation is always explicit and auditable.
- A future per-event routing ledger would refine the predicate without an API change.

### Safety guards (enforced + tested)
1. **Frozen only**: uncommitted / probationary experts and fresh pool slots are never candidates.
2. **Never prune below one trained expert** (`n_seen > 0`): if every trained expert is a candidate,
   the most-recently-used candidate is kept — highest recorded route count, tie → highest slot
   index (the same recruitment-recency proxy `_evict_victim` uses; under the conservative mapping
   candidate counts are zero, so the tie-break decides).
3. **`window` required** (see above).

### Re-use (mirrors the recruit/eviction path)
After release, the pool cursor is rewound to the **lowest free slot** iff the cursor is not
mid-training: cursor exhausted (`active >= M`), pointing at a frozen expert, or sitting on a fresh
never-trained slot — exactly the `self.active = victim` move the eviction path makes. Otherwise the
cursor is left untouched (an in-progress recruit is never abandoned) and released slots are blank
free capacity that the **next safe `prune_slots` call** rewinds to (the shipped cursor only
advances, so this rewind is what makes released capacity reachable at all). This also fixes the
otherwise-orphaned-slot hole when the release call happens with an unsafe cursor.

### Tests — `tests/test_expert_prune_lifecycle.py` (NEW, 6 tests, ~0.1 s)
1. frozen + zero-routed slot released; cursor rewired (exhausted) → the next recruit trains in the
   released slot; audit record carries pre-release state; survivor expert untouched;
2. protection: frozen + used (nonzero ledger) and non-frozen / mid-training experts survive, no
   cursor movement, bit-identical state across two no-op calls;
3. all-prunable guard: 3 trained zero-routed frozen candidates → exactly one (highest index) kept,
   ≥1 trained expert always survives; no keep when a trained survivor exists outside candidates;
4. freshness: released slot byte-identical to `Expert(d, h, K, seed+100*(slot+1), act_bits=8)`
   (n_seen=0, mu=1e9, var=1.0, not committed/frozen); mid-training cursor not abandoned; later safe
   call rewinds; training continues on the released slot (n_seen/route/floor re-calibrate);
5. window semantics pinned: `None`/0/−1/0.5/`"8"`/`True` raise; a single recorded routing protects
   at `window=1` and `window=1e9`; zero recorded routing releases at both (window-independent by
   design);
6. default path: two identically-seeded tiny streams bit-identical (weights/floors/flags/route_log/
   cursor/`state()`); `prune_log` never appears without an explicit call; routing unchanged.

## 2. Exact results

| command | result |
|---|---|
| `pytest tests/test_expert_prune_lifecycle.py tests/test_expert_abstain_open_world.py tests/test_expert_economy.py tests/test_continual_baselines.py tests/test_mmax_lever.py -q` | **44 passed in 2.71s** |
| `pytest tests -q` (full suite, since `src/` changed) | **559 passed, 10 skipped in 220.85s** (was 553P/10S; +6 = this lane) |

Lane-4's `abstain_z` + `route_or_novel` are untouched (verified by the pure-insertion diff and its
suite green).

## 3. Caveats / notes (honest)

- **White-box lifecycle states in tests.** The shipped machinery essentially never produces a
  frozen + zero-lifetime-routed expert organically (`route_log` only grows once an expert is
  active, and freezing implies prior training; the only organic path is the degenerate
  `commit_after=0` config). Like `tests/test_mmax_lever.py` does for eviction, the tests construct
  the state explicitly and pin the release/re-use contract; this is a property of the cumulative
  ledger, not of the implementation. If a future lane records per-event routing, the candidate
  predicate can be strengthened and the test construction becomes organic.
- **`window` is currently an explicit audit parameter, not a semantic cutoff** — deliberate under
  the conservative mapping (see §1). Reported here so no downstream doc claims exact-window
  pruning.
- No other file needed changes; no deviation from lane scope.
