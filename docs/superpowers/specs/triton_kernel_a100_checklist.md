# `seq/delta_triton.py` — Triton chunked-delta DRAFT: hardening report + A100 verification checklist

**Status:** DRAFT, **NOT TRUSTED / NOT for production** until the A100 checklist below passes.
**Provenance:** 28-agent adversarial review (6 lenses → per-finding adversarial verify → synthesis), 2026-06-08. 21 findings raised, **15 confirmed**.
**Reachability (governs severity):** `delta_triton` has **zero non-test importers** — the production model (`seq/prizma_seq.py`) calls `chunked_delta` directly and exposes no `backend=` switch, and the Triton path additionally requires `q.is_cuda AND _HAS_TRITON`, so it is **never executed locally** and cannot affect any committed/trusted result today. Every item below is a *latent A100-gate defect*, not a live production bug — but several are hard blockers that would make the A100 verification fail at launch or at the parity assert.

## Locally-safe fixes ALREADY APPLIED (this commit; none need CUDA; cannot regress the CPU byte-identical fallback)
1. **grad-gate** (`_should_use_triton`): the forward-only raw `@triton.jit` launch carries no `grad_fn`; the predicate now also requires *no grad live* (grad disabled or no input requires grad), so any training step routes to the differentiable eager `chunked_delta`. Closes **grad-1/grad-2**.
2. **`clogC = tl.min(...)`** (was `tl.max(...)`): `clog = cumsum(log α)` is non-increasing (α≤1), so γ_C is the **last/min** row, matching eager `clog[..., -1:]`. The `tl.max` form took the *first* row → corrupted `S_end` on every gated chunk. Closes **math-1/mask-1/triton-2** (HIGH; numerically verified gC 0.86 vs eager 0.039).
3. **`input_precision='ieee'`** on all six `tl.dot`: forces true-fp32 to match eager `torch.matmul` instead of A100 default TF32, so the <1e-3 gate measures algebra not TF32 drift. Closes **prec-1/triton-4**.
4. **alpha-stride splat** (`*((...) if gated else (0,0,0))`): the launcher passed 3 strides as one tuple into a kernel expecting 3 scalars (arity shortfall / arg misbinding). Closes **triton-1** (launch failure).
5. **CPU regression tests** added (`tests/test_delta_triton.py` §5): grad-gate predicate, explicit-`S0` carry byte-identity, non-pow2 `chunk ∈ {32,48,64,96}` (ragged), eager-fallback differentiability. Closes **grad-2/test-1/test-3** at the CPU level.

## STILL DEFERRED to the A100 (need CUDA to validate — do these on the GPU, in order)
- **triton-3 (COMPILE BLOCKER):** `tl.arange(0, DK/DV)` uses the **raw** channel dim; Triton requires power-of-2 arange and ≥16 / 16-aligned `tl.dot` dims. The real production key dim is `d_phi ∈ {128, 137, 160…}` (137 is neither pow-2 nor mult-of-16) → **cannot compile as-is**. Must introduce real `BLOCK_DK=next_power_of_2(d_k)` / `BLOCK_DV=next_power_of_2(d_v)` tile constants, build `arange(0, BLOCK_*)`, and rely on the (currently dead — **triton-6**) `dk<DK`/`dv<DV` padding masks.
- **triton-5 (robustness):** the forward-substitution `for i in range(0, BLOCK_C)` fully unrolls at BLOCK_C=64 with a per-row `[C,C]` gather → register pressure / long compile. Restructure the gather if it spills.
- **prec-4 (low-precision scope):** output buffers are `dtype=q.dtype`; bf16/fp16 parity is untested/unbounded while A100 training is typically bf16. Decide before adoption.
- **test-4:** no test exercises `d ≥ 64` (let alone production `d_phi`).

---

## A100 verification checklist (ordered; do NOT commit/trust the kernel until ALL pass)
0. **PRECONDITION:** re-confirm the locally-safe fixes are in and the CPU suite is green (fallback byte-identity `== 0.0`, delegation, dispatch, grad-gate/S0/chunk tests) before touching the GPU.
1. **LAUNCH:** kernel compiles + launches with no arity/stride error for the pure square case (gated=False, dk=dv=16, T=256, chunk=64) — verifies the alpha-stride splat (triton-1) + signature binding.
2. **DIMENSION TILING:** implement `BLOCK_DK/BLOCK_DV` (triton-3); confirm compile + correctness for non-pow-2 dims — at minimum dk=48,dv=16 **and the real production dims `d_phi ∈ {128,137,160}` with dv=d_h (32/64)**. Confirm `dk<DK`/`dv<DV` masks are now load-bearing (triton-6). MUST NOT crash at d_phi=137.
3. **FORWARD PARITY (pure):** `max|O_t−O_e| < 1e-3` **AND** `max|S_t−S_e| < 1e-3` for α=None, square + rectangular state, T=256/chunk=64. (Assert on **both** dO and dS.)
4. **FORWARD PARITY (gated):** same <1e-3 on dO **AND** dS for α∈[0.5,1] random **and** the α=0.5 constant worst-case floor — this is the gate that catches the clogC bug. **The dS assertion is mandatory** (the carry corruption shows in S_end first).
5. **REPEATED-KEY PARITY:** rank-deficient KK/QK case <1e-3 on dO **AND** dS — gates the γ_{i-1} pre-write read + inter-chunk carry.
6. **RAGGED-TAIL PARITY:** add chunk=48 and/or T not divisible by chunk (e.g. T=200) so C<BLOCK_C is exercised; <1e-3 on dO **AND** dS (closes test-3).
7. **EXPLICIT S0 CARRY:** pass non-None S0 (randn*0.1) into the CUDA parity test (pure + gated); <1e-3 on dO **AND** dS (closes test-1 — with S0=None all three S0 terms vanish).
8. **TF32 vs IEEE:** with `input_precision='ieee'` on every `tl.dot`, re-confirm all asserts pass at <1e-3; record whether removing the flag breaks the gate. **Do NOT loosen the tolerance to mask TF32 drift.**
9. **LOW-PRECISION SCOPE:** either add a bf16/fp16 parity test (document achievable tolerance) or assert `q.dtype==float32` on the Triton path (closes prec-4). A100 training is typically bf16 → decide before adoption.
10. **GRADIENT STRATEGY:** confirm the grad-gate routes grad-enabled CUDA calls to eager (q/k/v/beta grads populated, matching chunked_delta ~1e-5). If a native backward (autograd.Function) is added later, gate grad parity ~1e-2. Until then, **training MUST use the eager fallback** — verify no silent-zero-grad path exists.
11. **PERFORMANCE/ROBUSTNESS:** profile register usage + compile time of the unrolled forward-substitution at BLOCK_C=64 (triton-5); restructure if spilling. Only after correctness, benchmark vs `torch.compile(chunked_delta)` to justify adoption (no perf claim until then).
12. **SIGN-OFF:** only after 1–11 pass on the A100 may the kernel be marked trusted + committed; update the module docstring to drop DRAFT/PENDING and record the measured tolerances.
