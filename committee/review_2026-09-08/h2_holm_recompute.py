"""H-2 closure: recompute the registered Holm(2) for PR-2026-09-03-07, post-hoc.

READ-ONLY script. It loads the raw claim ledger
results/blockdrift_PR-2026-09-03-07/raw.json and recomputes the TWO registered
primary p-values exactly per docs/preregistry/2026-09-05-blockdrift-bar6.md
section 4, then runs the Holm step-down at alpha = 0.05 that the protocol
registered but the runner (seq/blockdrift_claim.py) never implemented
(review finding H-2, committee/review_2026-09-08/PRE_GPU_REVIEW.md).

Registered bars, mirrored exactly (doc section 4; n = 5 seeds 0-4):

  Bar 1 (retention, FGT_A): FGT_A = BPC_STREAM(A-eval, pre-B) - BPC_STREAM(A-eval,
      post-B) per seed; one-sample t. H0: mean >= 0.05 vs H1: mean < 0.05.
      p_raw1 = t_sf((0.05 - mean) / se, n - 1).
      seq.stats convention: t_sf(t, df) is the UPPER-TAIL survival P(T > t), so for
      t >= 0 it returns p in (0, 0.5]; with t = (margin - mean)/se this is exactly
      the lower-tail p of the flipped hypothesis (small = significantly below the
      margin).

  Bar 2 (adaptation, BPC_B): H0: mean(BPC_FROZEN) - mean(BPC_STREAM) <= 0.10 vs
      H1: > 0.10 (two-sample Welch; required advantage 0.10 in the improving
      direction for a lower-is-better metric). delta = mean(FROZEN) - mean(STREAM);
      se_welch and df_welch as in seq.stats._welch / the runner;
      p_raw2 = t_sf((delta - 0.10) / se_welch, df_welch).

  Holm(2) at alpha = 0.05 over [p_raw1, p_raw2]: sorted ascending, multipliers
  2 then 1, step-down (seq.stats.holm_correction).

This script WRITES NOTHING. It only reads the ledger and prints every number it
used (per-seed values, means, sd, se, df, t, raw p, Holm-adjusted p) so the
computation is auditable. Run from the repo root:

  ./.venv/Scripts/python.exe committee/review_2026-09-08/h2_holm_recompute.py
"""
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)  # same bootstrap as seq/blockdrift_claim.py (repo root)

from seq.stats import holm_correction, t_isf, t_sf  # noqa: E402

LEDGER_PATH = os.path.join(_ROOT, "results", "blockdrift_PR-2026-09-03-07", "raw.json")

ALPHA = 0.05        # registered family-wise alpha
MARGIN_FGT = 0.05   # bar 1: FGT_A mean <= 0.05
MARGIN_ADAPT = 0.10 # bar 2: mean(BPC_FROZEN) - mean(BPC_STREAM) >= 0.10


def one_sample(xs):
    """Return (n, mean, var_ddof1, se) exactly as seq/blockdrift_claim.py computes them."""
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    se = math.sqrt(var / n)
    return n, mean, var, se


def main():
    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        ledger = json.load(f)

    arms = ledger["arms"]
    for name in ("STREAM", "FROZEN"):
        if name not in arms:
            raise SystemExit("FATAL: ledger arm %r missing - not the expected ledger" % name)
    stream_recs = arms["STREAM"]["seed_records"]
    frozen_recs = arms["FROZEN"]["seed_records"]
    if len(stream_recs) != 5 or len(frozen_recs) != 5:
        raise SystemExit(
            "FATAL: expected n=5 per primary arm, got STREAM=%d FROZEN=%d"
            % (len(stream_recs), len(frozen_recs)))

    print("=== H-2 closure: registered Holm(2) recomputed from the raw ledger ===")
    print("ledger    : %s" % os.path.relpath(LEDGER_PATH, _ROOT))
    print("protocol  : %s" % ledger["meta"]["protocol"])
    print("seeds     : %s" % ledger["meta"]["seeds"])
    print("alpha     : %s ; Holm family: [bar1 FGT_A, bar2 adaptation], m=2" % ALPHA)
    print()

    # ---------------- Bar 1: FGT_A, one-sample (STREAM arm) ----------------
    fgt = [r["fgt_A"] for r in stream_recs]
    n1, mean1, var1, se1 = one_sample(fgt)
    df1 = n1 - 1
    t1 = (MARGIN_FGT - mean1) / se1
    p1 = t_sf(t1, df1)
    ci_hi1 = mean1 + t_isf(0.025, df1) * se1

    print("--- Bar 1 (retention, FGT_A): one-sample t, H0: mean >= 0.05 vs H1: mean < 0.05 ---")
    print("per-seed FGT_A (STREAM arm):")
    for r in stream_recs:
        print("  seed %d: fgt_A = %+.12f   (bpc_A_pre %.12f - bpc_A_post %.12f)"
              % (r["seed"], r["fgt_A"], r["bpc_A_pre"], r["bpc_A_post"]))
    print("n            = %d" % n1)
    print("mean         = %.12f" % mean1)
    print("sd (ddof=1)  = %.12f" % math.sqrt(var1))
    print("se           = %.12f" % se1)
    print("df           = %d" % df1)
    print("t            = (0.05 - mean) / se = %.12f" % t1)
    print("p_raw1       = t_sf(t, df) = %.6e   (upper-tail convention; small = below margin)" % p1)
    print("cross-check  : recomputed 95%% CI upper %.12f vs ledger fgt_ci_hi %.12f (diff %.2e)"
          % (ci_hi1, ledger["verdict"]["fgt_ci_hi"], abs(ci_hi1 - ledger["verdict"]["fgt_ci_hi"])))
    print("cross-check  : recomputed mean %.12f vs ledger fgt_mean %.12f"
          % (mean1, ledger["verdict"]["fgt_mean"]))
    print()

    # -------- Bar 2: adaptation, Welch two-sample (STREAM vs FROZEN, BPC_B) --------
    stream_b = [r["bpc_B"] for r in stream_recs]
    frozen_b = [r["bpc_B"] for r in frozen_recs]
    n_s, mean_s, var_s, _ = one_sample(stream_b)
    n_f, mean_f, var_f, _ = one_sample(frozen_b)
    delta = mean_f - mean_s                       # required advantage direction (FROZEN worse)
    se_w = math.sqrt(var_s / n_s + var_f / n_f)
    df_w = (var_s / n_s + var_f / n_f) ** 2 / (
        (var_s / n_s) ** 2 / (n_s - 1) + (var_f / n_f) ** 2 / (n_f - 1))
    t2 = (delta - MARGIN_ADAPT) / se_w
    p2 = t_sf(t2, df_w)
    tc = t_isf(0.025, df_w)
    adapt_ci = (mean_s - mean_f - tc * se_w, mean_s - mean_f + tc * se_w)

    print("--- Bar 2 (adaptation, BPC_B): Welch two-sample, H0: mean(FROZEN)-mean(STREAM) <= 0.10 ---")
    print("per-seed BPC_B (STREAM arm):")
    for r in stream_recs:
        print("  seed %d: bpc_B = %.12f" % (r["seed"], r["bpc_B"]))
    print("per-seed BPC_B (FROZEN arm):")
    for r in frozen_recs:
        print("  seed %d: bpc_B = %.12f" % (r["seed"], r["bpc_B"]))
    print("mean(STREAM) = %.12f  (n=%d, var=%.12f)" % (mean_s, n_s, var_s))
    print("mean(FROZEN) = %.12f  (n=%d, var=%.12f)" % (mean_f, n_f, var_f))
    print("delta        = mean(FROZEN) - mean(STREAM) = %.12f" % delta)
    print("se_welch     = %.12f" % se_w)
    print("df_welch     = %.12f" % df_w)
    print("t            = (delta - 0.10) / se_welch = %.12f" % t2)
    print("p_raw2       = t_sf(t, df) = %.6e" % p2)
    print("cross-check  : recomputed Welch 95%% CI of (STREAM-FROZEN) = [%.12f, %.12f]"
          % adapt_ci)
    print("               vs ledger adapt_ci = [%.12f, %.12f]  (adapt_diff %.12f)"
          % (ledger["verdict"]["adapt_ci"][0], ledger["verdict"]["adapt_ci"][1],
             ledger["verdict"]["adapt_diff"]))
    print()

    # ---------------- Holm(2) step-down at alpha = 0.05 ----------------
    holm = holm_correction([p1, p2], alpha=ALPHA)  # returned in input order [bar1, bar2]
    print("--- Holm(2) step-down at alpha = 0.05 over [p_raw1, p_raw2] ---")
    order = sorted(range(2), key=lambda i: (p1, p2)[i])
    for rank, idx in enumerate(order):
        mult = 2 - rank
        pv = (p1, p2)[idx]
        thresh = ALPHA / mult
        print("  rank %d: bar%d raw p = %.6e  x %d = %.6e  (threshold alpha/%d = %.4f) -> %s"
              % (rank + 1, idx + 1, pv, mult, min(pv * mult, 1.0), mult, thresh,
                 "REJECT H0" if pv * mult < ALPHA else "FAIL TO REJECT"))
    print()
    print("bar 1 (FGT_A):     raw p = %.6e   Holm-adjusted p = %.6e   reject = %s"
          % (p1, holm[0]["p_adj"], holm[0]["reject"]))
    print("bar 2 (adaptation): raw p = %.6e   Holm-adjusted p = %.6e   reject = %s"
          % (p2, holm[1]["p_adj"], holm[1]["reject"]))
    print()

    both_below = max(p1, p2) < ALPHA / 2.0
    print("VERDICT: %s" % (
        "BOTH raw p-values (%.3e and %.3e) are far below the binding first Holm threshold"
        " alpha/2 = 0.025, so the registered-but-unimplemented Holm(2) is a NO-OP here and"
        " CANNOT flip the CLAIMED (PASS) outcome. The claim verdict, which used the"
        " conservative CI rules, stands." % (p1, p2)
        if both_below else
        "at least one raw p is NOT below alpha/2 = 0.025 - the Holm no-op argument does NOT"
        " hold from this ledger; do not claim closure."))

    return 0


if __name__ == "__main__":
    sys.exit(main())
