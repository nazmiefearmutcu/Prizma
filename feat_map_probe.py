"""Capacity pre-filter for feat_map variants (Task 1.D, Council-1 R9).

Measures the off-diagonal key crosstalk of each feature map at D=128, d_h=32 (= d_model=64, n_heads=2).

Metric: mean absolute off-diagonal cosine similarity of the EXPANDED KEYS phi(k_i) (unit-normed),
  cross(phi) = E[|phi(k_i) . phi(k_j)|]  for i != j, averaged over random key draws.
This equals sqrt(2/(pi * d_phi)) for purely random keys in R^{d_phi}, giving a theoretical lower
bound. Structured feature maps (monomials) deviate upward due to correlations; the improvement over
'none' (d_phi=d_h=32 -> ~0.142) shows whether the map decorrelates keys in the extended space.

PASS CRITERION (local pre-filter; the end-to-end MQAR gate runs on A100):
  quad2_lowrank crosstalk <= 0.085 (absolute bar from plan) AND within 0.010 of quad2's value.
  NOTE: the absolute bar 0.085 was calibrated by the plan author against a prior code path; if
  our metric gives different absolute values, the RELATIVE criterion (within 0.010) is binding.

Mean/fluctuation law metrics (Wave-B B2, committee report 11, 2026-09-03; additive — every legacy
field above keeps its exact semantics):
  Split the crosstalk matrix E = G - I into a common mode and a fluctuation: E = mu*11^T + W.
  From the POOLED i<j upper-triangle entries of |phi_i . phi_j| (and the signed cosines) over all
  trials we report, per feature map:
    mu                  mean |off-diag cosine| over the pooled i<j upper triangle
                        (== legacy cross(phi) in exact arithmetic; legacy "mean" averages the
                        per-trial means over trials -- same value, same number)
    mu_signed           mean SIGNED cosine over the same pooled triangle: the common-mode mu of
                        the split E = mu*11^T + W (report 11 predicts mu ~ lambda/d_h ~ 5.6e-3
                        for quad2; ~0 by symmetry for 'none')
    sigma2              SAMPLE std (ddof=1) of the |cosines| over the same pooled i<j triangle.
                        EXACT DEFINITION (pre-registered): sigma2 = std_{t,i<j}(|phi_i.phi_j|),
                        ddof=1, pooled over all trials. This is the std of ABSOLUTE cosines.
                        NOTE: the random-key floor for THIS definition is sqrt((1-2/pi)/d_phi)
                        (std of a half-normal), and its eta floor is 1/(1-2/pi) ~= 2.752 -- NOT
                        the eta=1 floor of report 11 (that floor belongs to the signed def).
    sigma2_signed_fluct SAMPLE std (ddof=1) of the SIGNED cosines over the same pooled triangle:
                        the fluctuation of W under E = mu*11^T + W. Random-key floor = 1/sqrt(d_phi)
                        (verified to 4 decimals by 'none' below); the report-11 SNR quantity
                        sqrt(E[c^2]) is exactly sqrt(sigma2_signed_fluct^2 + mu_signed^2).
                        For a zero-mean, |cos|-half-normal map, sigma2_signed_fluct ~= mu*sqrt(pi/2)
                        -- the CORRECT conversion factor is sqrt(pi/2) ~= 1.2533, NOT the pi/2
                        used in committee report 11 (verified numerically on 'none').
    eta                 1/(sigma2^2 * d_phi)           (crosstalk efficiency, |cos| def)
    eta_signed_fluct    1/(sigma2_signed_fluct^2 * d_phi) (report-11-comparable; pure random-key
                        floor has eta_signed_fluct = 1)
    n_star              min(1 + 1/sigma2^2, d_phi)     (candidate capacity law N* = min(1+eps^2/
                        sigma2^2, d_phi) at eps=1 for unit-norm keys; CANDIDATE -- not yet fitted
                        or validated, see docs/crosstalk_capacity_law.md)
    n_star_signed_fluct min(1 + 1/sigma2_signed_fluct^2, d_phi)
    offdiag_count       number of pooled upper-triangle samples behind the estimates

Writes results to results/feat_map_probe.json.

Run: python feat_map_probe.py
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")

D = 128       # number of MQAR key-value pairs (the hard rung)
D_H = 32      # d_h = d_model // n_heads = 64 // 2  (config used in all capacity probes)
N_TRIALS = 256  # random key draws for stable mean (each trial is independent)
SEED = 42

RES = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RES, exist_ok=True)
OUT = os.path.join(RES, "feat_map_probe.json")


# ── helper ────────────────────────────────────────────────────────────────────

def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(n > 0, n, 1.0)


def _key_crosstalk(K_phi: np.ndarray) -> float:
    """Mean absolute off-diagonal cosine similarity of the unit-normed keys K_phi (D x d_phi).
    Lower is better: fewer false associations when reading a stored key back."""
    Kn = _l2(K_phi)           # re-normalise (should already be unit, but guard fp drift)
    sim = Kn @ Kn.T           # D x D; diagonal = 1 (self-similarity)
    off_mask = 1.0 - np.eye(K_phi.shape[0])
    return float(np.abs(sim * off_mask).sum() / (K_phi.shape[0] * (K_phi.shape[0] - 1)))


# ── mean/fluctuation law metrics (pure, importable; tested in tests/test_crosstalk_metrics.py) ──

def offdiag_upper(sim: np.ndarray) -> np.ndarray:
    """Signed off-diagonal upper-triangle entries sim[i, j] for i < j, flattened (deterministic
    np.triu_indices order)."""
    sim = np.asarray(sim, dtype=float)
    iu = np.triu_indices(sim.shape[0], k=1)
    return sim[iu]


def n_star_law(sigma2: float, d_phi: int, eps: float = 1.0) -> float:
    """Candidate mean/fluctuation capacity law (committee report 11, CANDIDATE -- unvalidated):

        N*(eps; phi) = min(1 + eps^2 / sigma2^2, d_phi)

    eps = task-level signal tolerance (eps=1 for unit-norm keys); sigma2 = crosstalk fluctuation
    (per-entry std, NOT the mean); d_phi = hard rank cap (a rank-d_phi linear state cannot store
    more than d_phi arbitrary pairs exactly). sigma2 == 0 (orthonormal or identical keys) degenerates
    to the rank cap d_phi."""
    if sigma2 <= 0.0:
        return float(d_phi)
    return float(min(1.0 + eps * eps / (sigma2 * sigma2), float(d_phi)))


def crosstalk_metrics_from_values(abs_vals: np.ndarray, signed_vals: np.ndarray,
                                  d_phi: int) -> dict:
    """Mean/fluctuation metrics from pooled off-diagonal upper-triangle cosine values.

    abs_vals    pooled |phi_i . phi_j| (i<j) samples
    signed_vals pooled  phi_i . phi_j  (i<j) samples
    d_phi       expanded key dimension (for eta and the rank cap)

    Exact definitions (also in the module docstring):
      mu                  = mean(|c|)                                 [== cross(phi)]
      sigma2              = std(|c|, ddof=1)                          [std of ABSOLUTE cosines]
      sigma2_signed_fluct = std(c, ddof=1)                            [fluctuation of W in
                                                                       E = mu*11^T + W]
      eta                 = 1/(sigma2^2 * d_phi)                      [inf when sigma2 == 0]
      eta_signed_fluct    = 1/(sigma2_signed_fluct^2 * d_phi)         [random-key floor == 1]
      n_star              = min(1 + 1/sigma2^2, d_phi)                [eps=1; rank cap when sigma2==0]
      n_star_signed_fluct = min(1 + 1/sigma2_signed_fluct^2, d_phi)
    """
    abs_vals = np.asarray(abs_vals, dtype=float).ravel()
    signed_vals = np.asarray(signed_vals, dtype=float).ravel()
    n = abs_vals.size
    mu = float(abs_vals.mean()) if n > 0 else 0.0
    mu_signed = float(signed_vals.mean()) if n > 0 else 0.0
    sigma2 = float(abs_vals.std(ddof=1)) if n > 1 else 0.0
    sigma2_fluct = float(signed_vals.std(ddof=1)) if n > 1 else 0.0
    eta = float(1.0 / (sigma2 * sigma2 * d_phi)) if sigma2 > 0.0 else float("inf")
    eta_fluct = (float(1.0 / (sigma2_fluct * sigma2_fluct * d_phi))
                 if sigma2_fluct > 0.0 else float("inf"))
    return {
        "mu": mu,
        "mu_signed": mu_signed,
        "sigma2": sigma2,
        "sigma2_signed_fluct": sigma2_fluct,
        "eta": eta,
        "eta_signed_fluct": eta_fluct,
        "n_star": n_star_law(sigma2, d_phi),
        "n_star_signed_fluct": n_star_law(sigma2_fluct, d_phi),
        "offdiag_count": int(n),
    }


def crosstalk_metrics(K_phi: np.ndarray) -> dict:
    """All mean/fluctuation crosstalk metrics for one key matrix K_phi (D x d_phi).
    Keys need not be unit-normed (re-normalised here, same guard as _key_crosstalk)."""
    K_phi = np.asarray(K_phi, dtype=float)
    Kn = _l2(K_phi)
    sim = Kn @ Kn.T
    vals = offdiag_upper(sim)
    return crosstalk_metrics_from_values(np.abs(vals), vals, K_phi.shape[1])


# ── fixed buffers (seeded exactly as in prizma_seq.py: seed 1234, same generator order) ──

def _make_quad2_buffers(d_h: int, feat_n2: int) -> tuple[np.ndarray, np.ndarray]:
    """Replicate the torch.Generator(seed=1234) randint sequence for quad2."""
    rng = np.random.default_rng(1234)
    feat_I = rng.integers(0, d_h, feat_n2)
    feat_J = rng.integers(0, d_h, feat_n2)
    return feat_I, feat_J


def _make_lowrank_buffers(d_h: int, r: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replicate the torch.Generator(seed=1234) randn + index sequence for quad2_lowrank."""
    rng = np.random.default_rng(1234)
    P = rng.standard_normal((d_h, r)) * (d_h ** -0.5)
    I_lr = np.array([i for i in range(r) for j in range(i, r)], dtype=np.intp)
    J_lr = np.array([j for i in range(r) for j in range(i, r)], dtype=np.intp)
    return P, I_lr, J_lr


# ── per-map phi functions ──────────────────────────────────────────────────────

def _phi_none(K: np.ndarray, _buffers) -> np.ndarray:
    return K   # identity; keys stay in R^{d_h}


def _phi_quad2(K: np.ndarray, buffers) -> np.ndarray:
    feat_I, feat_J = buffers
    two = K[:, feat_I] * K[:, feat_J]
    return _l2(np.concatenate([K, two], axis=1))


def _phi_quad2_lowrank(K: np.ndarray, buffers) -> np.ndarray:
    P, I_lr, J_lr = buffers
    z = K @ P                          # (D, r)  —  fixed projection
    two = z[:, I_lr] * z[:, J_lr]     # (D, n_pairs)
    return _l2(np.concatenate([K, two], axis=1))


# ── main probe ────────────────────────────────────────────────────────────────

def probe_crosstalk(phi_fn, buffers, d_h: int = D_H, n_trials: int = N_TRIALS,
                    seed: int = SEED, n_keys: int = D) -> dict:
    """Crosstalk of `phi_fn` over n_trials independent random key sets (n_keys unit-norm keys each).

    Legacy fields (unchanged semantics): mean/std/min/max over the PER-TRIAL _key_crosstalk values,
    n_trials, d_phi. Additive fields: mu, sigma2, sigma2_signed_fluct, eta, eta_signed_fluct,
    n_star, n_star_signed_fluct, offdiag_count (pooled i<j upper-triangle metrics; exact
    definitions in the module docstring and crosstalk_metrics_from_values)."""
    rng = np.random.default_rng(seed)
    xts = []
    abs_pool = []
    signed_pool = []
    for _ in range(n_trials):
        K = rng.standard_normal((n_keys, d_h))
        K = _l2(K)                         # unit-norm raw keys (pre-map, as in prizma_seq._encode)
        K_phi = phi_fn(K, buffers)
        xts.append(_key_crosstalk(K_phi))
        vals = offdiag_upper(_l2(K_phi) @ _l2(K_phi).T)
        abs_pool.append(np.abs(vals))
        signed_pool.append(vals)
    arr = np.array(xts)
    out = {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_trials": n_trials,
        "d_phi": K_phi.shape[1],   # actual expanded dimension
    }
    out.update(crosstalk_metrics_from_values(np.concatenate(abs_pool),
                                             np.concatenate(signed_pool), out["d_phi"]))
    return out


def main():
    # ── quad2 reference: feat_n2=224, d_phi=32+224=256  (canonical "full" quad2 from plan) ──
    feat_n2_quad2 = 224       # d_h=32 + 224 = 256
    r_lowrank = 14            # default feat_rank; d_phi = 32 + 14*15//2 = 32+105 = 137

    bufs_none = None
    bufs_quad2 = _make_quad2_buffers(D_H, feat_n2_quad2)
    bufs_lr = _make_lowrank_buffers(D_H, r_lowrank)

    print(f"Capacity pre-filter: feat_map crosstalk at D={D}, d_h={D_H} "
          f"({N_TRIALS} random key draws each)", flush=True)
    print(f"  quad2 config: feat_n2={feat_n2_quad2}, d_phi={D_H + feat_n2_quad2}")
    print(f"  quad2_lowrank config: feat_rank={r_lowrank}, d_phi={D_H + r_lowrank*(r_lowrank+1)//2}")
    print()

    results = {}

    for name, fn, bufs in [
        ("none",          _phi_none,          bufs_none),
        ("quad2",         _phi_quad2,         bufs_quad2),
        ("quad2_lowrank", _phi_quad2_lowrank, bufs_lr),
    ]:
        r = probe_crosstalk(fn, bufs)
        results[name] = r
        print(f"  {name:<16}: d_phi={r['d_phi']:<4}  crosstalk={r['mean']:.4f} ± {r['std']:.4f}")

    # ── pass / fail judgement ──────────────────────────────────────────────────
    quad2_xt  = results["quad2"]["mean"]
    lr_xt     = results["quad2_lowrank"]["mean"]
    none_xt   = results["none"]["mean"]
    lr_d_phi  = results["quad2_lowrank"]["d_phi"]
    q2_d_phi  = results["quad2"]["d_phi"]

    gap_from_quad2 = lr_xt - quad2_xt
    abs_pass  = lr_xt <= 0.085
    rel_pass  = abs(gap_from_quad2) <= 0.010
    half_pass = lr_d_phi <= q2_d_phi * 0.6   # ≤60% of quad2 d_phi counts as "~half"
    overall   = rel_pass and half_pass        # relative criterion is binding (see docstring)

    theory_none = float(np.sqrt(2 / (np.pi * D_H)))
    theory_q2   = float(np.sqrt(2 / (np.pi * q2_d_phi)))
    theory_lr   = float(np.sqrt(2 / (np.pi * lr_d_phi)))

    print()
    print("──── PRE-FILTER VERDICT ────────────────────────────────────────────")
    print(f"  none  d_phi={D_H:<4}: crosstalk={none_xt:.4f}  (theory for pure R^{D_H}: {theory_none:.4f})")
    print(f"  quad2 d_phi={q2_d_phi:<4}: crosstalk={quad2_xt:.4f}  (theory for pure R^{q2_d_phi}: {theory_q2:.4f})")
    print(f"  lr    d_phi={lr_d_phi:<4}: crosstalk={lr_xt:.4f}  (theory for pure R^{lr_d_phi}: {theory_lr:.4f})")
    print(f"  gap(lr - quad2) = {gap_from_quad2:+.4f}  (criterion: |gap| <= 0.010)")
    print(f"  d_phi ratio lr/quad2 = {lr_d_phi}/{q2_d_phi} = {lr_d_phi/q2_d_phi:.2f}  (criterion: <= 0.60)")
    print(f"  absolute bar (<=0.085, metric-relative): {'PASS' if abs_pass else 'FAIL (metric differs from plan baseline)'}")
    print(f"  relative criterion (|gap|<=0.010):        {'PASS' if rel_pass else 'FAIL'}")
    print(f"  d_phi ≤ 60% of quad2:                    {'PASS' if half_pass else 'FAIL'}")
    print(f"  OVERALL (relative + half-d_phi):          {'PASS' if overall else 'FAIL'}")

    print()
    print("──── MEAN/FLUCTUATION LAW (candidate, committee report 11; eps=1) ─────────")
    for name in ("none", "quad2", "quad2_lowrank"):
        r = results[name]
        dp = r["d_phi"]
        t_abs = float(np.sqrt((1 - 2 / np.pi) / dp))   # std of |cos| for pure random keys
        t_sgn = float(np.sqrt(1.0 / dp))               # signed fluctuation floor 1/sqrt(d_phi)
        print(f"  {name:<16}: sigma2={r['sigma2']:.4f} (|cos| floor {t_abs:.4f})  "
              f"sig_fluct={r['sigma2_signed_fluct']:.4f} (floor {t_sgn:.4f})  "
              f"mu_signed={r['mu_signed']:+.5f}  eta_fl={r['eta_signed_fluct']:.3f}  "
              f"N*={r['n_star']:.1f} (d_phi={dp})")
    print("  NOTE: N* at eps=1 with RANDOM-key sigma2 is a candidate diagnostic only; the law's")
    print("  eps must be fitted at D=64 before any D-frontier prediction (docs/crosstalk_capacity_law.md).")

    results["_meta"] = {
        "D": D, "d_h": D_H, "n_trials": N_TRIALS, "seed": SEED,
        "feat_n2_quad2": feat_n2_quad2, "feat_rank_lowrank": r_lowrank,
        "gap_lr_vs_quad2": gap_from_quad2,
        "abs_pass": abs_pass, "rel_pass": rel_pass, "half_pass": half_pass, "overall": overall,
        "theory_none": theory_none, "theory_quad2": theory_q2, "theory_lowrank": theory_lr,
        "n_star_none": results["none"]["n_star"],
        "n_star_quad2": results["quad2"]["n_star"],
        "n_star_lowrank": results["quad2_lowrank"]["n_star"],
        "eta_none": results["none"]["eta"],
        "eta_quad2": results["quad2"]["eta"],
        "eta_lowrank": results["quad2_lowrank"]["eta"],
        "theory_sigma2_abs_floor_none": float(np.sqrt((1 - 2 / np.pi) / D_H)),
        "theory_sigma2_abs_floor_quad2": float(np.sqrt((1 - 2 / np.pi) / q2_d_phi)),
        "theory_sigma2_abs_floor_lowrank": float(np.sqrt((1 - 2 / np.pi) / lr_d_phi)),
        "theory_sigma2_signed_floor_none": float(np.sqrt(1.0 / D_H)),
        "theory_sigma2_signed_floor_quad2": float(np.sqrt(1.0 / q2_d_phi)),
        "theory_sigma2_signed_floor_lowrank": float(np.sqrt(1.0 / lr_d_phi)),
        "law_definition_sigma2": (
            "sample std (ddof=1) of |phi_i.phi_j| over the i<j upper triangle, pooled over all "
            "trials; sigma2_signed_fluct is the std of the SIGNED cosines (fluctuation of W in "
            "E = mu*11^T + W). N* = min(1 + eps^2/sigma2^2, d_phi) at eps=1 (unit-norm keys); "
            "CANDIDATE law, eps not yet fitted -- see docs/crosstalk_capacity_law.md."
        ),
        "note": (
            "Absolute bar 0.085 was calibrated by plan author against a different probe baseline. "
            "The BINDING criterion here is rel_pass (|gap|<=0.010) + half_pass (d_phi<=60% of quad2). "
            "For the end-to-end gate, see Task 1.D amendment: >=10-seed MQAR-D128 on A100. "
            "HISTORY NOTE (2026-09-03): README/docs quote cross(quad2)~0.076 and cross(lowrank)~0.085, "
            "but this committed artifact has always measured 0.1170/0.1265 (only cross(none)~0.142 "
            "matches). The 0.076/0.085 figures trace to an unreproducible prior code path -- see the "
            "addendum in docs/quad2_theoretical_convergence.md."
        ),
    }

    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nsaved -> {OUT}", flush=True)
    return overall


if __name__ == "__main__":
    main()
