"""PR-2026-09-03-03 claim run — single-pass citation bar (docs/CONTINUAL_CITATION_BAR.md §6, VERBATIM).

Registered protocol: single-pass battery, arms = the full Table-2 set, seeds 10-19 (fresh,
never touched by the seeds 0-9 exploration), runner conventions of
experiments/run_citation_battery.py (identical substrate/stream/tuning rule; the only
addition is seed_start, which shifts the seed block without changing anything else).

Bar (frozen): B* = boundary-free regularizer arm in {MAS, OnlineEWC(online)} with the higher
ACC on the run's FIRST seed (the doc's "seed 0" of the tuning rule — first seed of the run);
PASS iff ACC_PRIZMA - ACC_B* mean >= -0.05 AND the two-sided Welch 95% CI on the seed-level
difference excludes values below -0.05. CIs straddling -0.05 => INCONCLUSIVE (never PASS).
Secondary (reported, not gated): FGT_PRIZMA <= FGT_B* + 0.05.

Run:  python experiments/run_pr03_claim.py
"""
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from run_citation_battery import CONFIGS, run_protocol  # noqa: E402

OUT_DIR = os.path.join(_ROOT, "results", "citation_battery_PR-2026-09-03-03")
SEED_START, N_SEEDS = 10, 10
LANE = "CLAIM PR-2026-09-03-03 (registered 2026-09-05; protocol frozen in docs/CONTINUAL_CITATION_BAR.md §6)"
BOUNDARY_FREE_CANDIDATES = ["MAS", "OnlineEWC(online)"]
DELTA = 0.05


def _accs_by_seed(raw, arm):
    recs = sorted(raw["arms"][arm]["seed_records"], key=lambda r: r["seed"])
    return {r["seed"]: r["ACC"] for r in recs}, [r["ACC"] for r in recs], [r["FGT"] for r in recs]


def _welch_ci(a, b):
    """Two-sample Welch t 95% CI for mean(a) - mean(b), Welch-Satterthwaite df."""
    from seq.stats import t_isf  # repo's own Student-t machinery (no scipy); t_isf takes the UPPER-TAIL p in (0, 0.5]
    n1, n2 = len(a), len(b)
    m1, m2 = sum(a) / n1, sum(b) / n2
    v1 = sum((x - m1) ** 2 for x in a) / (n1 - 1)
    v2 = sum((x - m2) ** 2 for x in b) / (n2 - 1)
    se = math.sqrt(v1 / n1 + v2 / n2)
    df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    tc = t_isf(0.025, df)  # two-sided 95% -> 0.025 upper tail
    d = m1 - m2
    return d, d - tc * se, d + tc * se, df


def _paired_ci(a, b):
    """One-sample t 95% CI on paired differences a_i - b_i (supplementary — pairing is by seed)."""
    from seq.stats import t_isf
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    m = sum(diffs) / n
    v = sum((x - m) ** 2 for x in diffs) / (n - 1)
    se = math.sqrt(v / n)
    tc = t_isf(0.025, n - 1)  # two-sided 95% -> 0.025 upper tail
    return m, m - tc * se, m + tc * se


def main():
    raw_path = os.path.join(OUT_DIR, "raw_single_pass.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(raw_path):
        print(f"[pr03] running registered battery: single-pass, seeds {SEED_START}-"
              f"{SEED_START + N_SEEDS - 1}, 8 arms -> {os.path.relpath(OUT_DIR, _ROOT)}")
        run_protocol("single_pass", CONFIGS["e1"], OUT_DIR, n_seeds=N_SEEDS,
                     seed_start=SEED_START, lane_label=LANE)
    raw = json.load(open(raw_path, encoding="utf-8"))
    meta, arms = raw["meta"], raw["arms"]

    needed = {"backprop", "replay", "PRIZMA(DFA)"} | set(BOUNDARY_FREE_CANDIDATES)
    missing = needed - set(arms)
    assert not missing, f"raw ledger missing arms: {missing}"

    accs, fgt = {}, {}
    for arm in arms:
        by_seed, acc_list, fgt_list = _accs_by_seed(raw, arm)
        assert len(acc_list) == N_SEEDS, f"{arm}: expected {N_SEEDS} seeds, got {len(acc_list)}"
        assert min(by_seed) == SEED_START and max(by_seed) == SEED_START + N_SEEDS - 1
        accs[arm], fgt[arm] = acc_list, fgt_list

    first_seed = SEED_START
    b_star = max(BOUNDARY_FREE_CANDIDATES, key=lambda a: accs[a][0])
    print(f"[pr03] B* (boundary-free candidate with higher FIRST-seed ACC, seed {first_seed}): "
          f"{b_star} ({accs[b_star][0]:.3f} vs {accs['MAS'][0]:.3f} / {accs['OnlineEWC(online)'][0]:.3f})")

    p = accs["PRIZMA(DFA)"]
    d, lo, hi, df = _welch_ci(p, accs[b_star])
    pm, plo, phi = _paired_ci(p, accs[b_star])
    fgt_p = sum(fgt["PRIZMA(DFA)"]) / N_SEEDS
    fgt_b = sum(fgt[b_star]) / N_SEEDS

    if lo >= -DELTA:
        verdict = "PASS"
    elif hi >= -DELTA:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "FAIL"
    secondary_ok = fgt_p <= fgt_b + 0.05

    lines = [
        "# PR-2026-09-03-03 — single-pass citation bar — RESULTS (registered run)",
        "",
        f"- Protocol: docs/CONTINUAL_CITATION_BAR.md §6 executed VERBATIM (single-pass battery,",
        f"  arms = Table-2 set, seeds {SEED_START}-{SEED_START + N_SEEDS - 1}, tuning on the run's first seed).",
        f"- Pre-run interpretation notes (fixed before verdict computation): the doc's 'seed 0'",
        f"  references denote the run's FIRST seed ({first_seed}); the bar uses the literal",
        f"  two-sample Welch 95% CI on ACC_PRIZMA − ACC_B*; the paired-by-seed CI is reported",
        f"  as supplementary (pairing is by shared task sequence).",
        "",
        "## Per-arm means (± 1.96 SEM)",
        "",
        "| arm | ACC | FGT |",
        "|---|---|---|",
    ]
    import run_citation_battery as rc
    for arm in rc.SINGLE_ARMS:
        a, ac = rc.ci95(accs[arm])
        f, fc = rc.ci95(fgt[arm])
        lines.append(f"| {arm} | {a:.3f} ± {ac:.3f} | {f:.3f} ± {fc:.3f} |")
    lines += [
        "",
        f"- **B*** = {b_star} (higher first-seed ACC among the frozen boundary-free candidates).",
        f"- Primary: ACC_PRIZMA − ACC_B* = {d:+.3f}; Welch 95% CI [{lo:+.3f}, {hi:+.3f}] (df={df:.1f}).",
        f"- Supplementary paired-by-seed 95% CI: [{plo:+.3f}, {phi:+.3f}].",
        f"- Secondary (not gated): FGT {fgt_p:.3f} vs FGT_B* {fgt_b:.3f} "
        f"(≤ B*+0.05: {'yes' if secondary_ok else 'no'}).",
        "",
        f"## VERDICT: **{verdict}**",
        "",
        {"PASS": "Prizma is competitive with boundary-free regularizers in single-pass "
                 "domain-incremental CL (claim usable, with the doc's honest-limit notes).",
         "INCONCLUSIVE": "CIs straddle −0.05 — the claim is NOT established and NOT retired; "
                         "a follow-up needs ~n=15-20 seeds (maintainer addendum territory).",
         "FAIL": "The single-pass competitiveness sentence is RETIRED from README/abstract use; "
                 "the multi-epoch E1 claim stands untouched (different protocol, different claim)."}[verdict],
        "",
        f"Raw per-seed records: `raw_single_pass.json` (retention per docs/RETENTION.md).",
    ]
    out_md = os.path.join(OUT_DIR, "RESULTS.md")
    open(out_md, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines[lines.index("## Per-arm means (± 1.96 SEM)"):]))
    print(f"\n[pr03] wrote {os.path.relpath(out_md, _ROOT)} — VERDICT: {verdict}")


if __name__ == "__main__":
    main()
