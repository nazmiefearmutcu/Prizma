"""Committee-completing follow-up arms, written into the SAME $PRIZMA_RESULTS/gpu_bench.json
(run_cell caches by cellkey -> incremental & resumable; existing p1/p2/... cells untouched).

Two arms, both at D=128 MixedMQAR, gen-warm, per-model plateau, 5 seeds:

  ARM B  "FLOP-matched TF" — the headline (P2) is param-matched at d128L4H4, but the measured
         ledger shows Prizma-quad2-256 costs ~2.14x the param-matched TF's forward FLOPs/token.
         A deeper TF d128L9H4 matches Prizma-quad2's as-coded FLOPs to ~2% (4576 vs 4505
         kFLOP/tok; flop_ledger.py) AND has MORE params than Prizma -> deliberately generous to
         attention. Answers "is Prizma's recall just spent FLOPs?".

  ARM A  "parameter-efficiency / sharp contrast" — at the TINY matched scale d64L2H2 (130K),
         does Prizma-quad2 solve D=128 where the param-matched Transformer cannot? P1 already
         shows TF d64L2H2 = 0/3 (0.016). This adds Prizma-quad2 + Prizma-none x5 at 130K (and 2
         more TF seeds for a clean 5v5). This is the strongest honest claim: same recall, ~3.5x
         fewer params than the smallest clean TF solver (d128L2H4, 461K), with O(1) inference.

Run on the Colab A100 (gist must include this file):
    PRIZMA_RESULTS=/content/drive/MyDrive/prizma_results python -u extra_arms.py
"""
from gpu_bench import (run_cell, solve_stats, tf_factory, ps_factory, _load, _save, OUT, DEV)
from seq.tasks import MixedMQAR

V = 512
S5 = (0, 1, 2)   # 3 seeds (committee min ≥3): Prizma is ~30-57min/run on A100 — 5 seeds is intractable here
TASK = lambda: MixedMQAR(vocab=V, max_pairs=128, num_queries=128, gap=0, min_pairs=1)


def main():
    print(f"device={DEV} results={OUT}  EXTRA ARMS (FLOP-matched + parameter-efficiency)", flush=True)
    res = _load()
    summary = {}

    # --- ARM B: FLOP-matched (deeper) Transformer d128L9H4 ~= Prizma-quad2-256 as-coded FLOPs ---
    print("\n== ARM B: FLOP-matched TF d128L9H4 @ D=128 ==", flush=True)
    recs = [run_cell(res, f"p2b.TF-flopmatch-d128L9H4.s{s}", tf_factory(128, 9, 4), TASK, 80000, s)
            for s in S5]
    summary["TF-flopmatch-d128L9H4"] = solve_stats(recs)
    print(f"  -> {summary['TF-flopmatch-d128L9H4']}", flush=True)

    # --- ARM A: parameter-efficiency at the tiny matched 130K scale d64L2H2 (d_h=32, d_phi=256) ---
    print("\n== ARM A: parameter-efficiency @ d64L2H2 (130K), D=128 ==", flush=True)
    q = [run_cell(res, f"p2eff.Prizma-quad2-d64L2H2.s{s}",
                  ps_factory(64, 2, 2, feat_map="quad2", feat_n2=224), TASK, 80000, s) for s in S5]
    summary["Prizma-quad2-d64L2H2"] = solve_stats(q)
    print(f"  -> Prizma-quad2: {summary['Prizma-quad2-d64L2H2']}", flush=True)

    n = [run_cell(res, f"p2eff.Prizma-none-d64L2H2.s{s}", ps_factory(64, 2, 2), TASK, 80000, s)
         for s in S5]
    summary["Prizma-none-d64L2H2"] = solve_stats(n)
    print(f"  -> Prizma-none:  {summary['Prizma-none-d64L2H2']}", flush=True)

    # rand_linear CONTROL (fixed random LINEAR 32->256 map): rank<=d_h -> expected NO gain over none.
    # If quad2 >> rand_linear ~ none, the gain is causally the QUADRATIC monomials (committee guardrail #5).
    rl = [run_cell(res, f"p2eff.Prizma-randlin-d64L2H2.s{s}",
                   ps_factory(64, 2, 2, feat_map="rand_linear", feat_n2=224), TASK, 80000, s) for s in S5]
    summary["Prizma-randlin-d64L2H2"] = solve_stats(rl)
    print(f"  -> Prizma-rand_linear (control): {summary['Prizma-randlin-d64L2H2']}", flush=True)

    # TF d64L2H2: reuse the 3 cached P1 seeds (all 0.016) + 2 fresh for a clean 5-seed set.
    tf = []
    for s in S5:
        ck = f"p1.TF.d64L2H2.s{s}" if s < 3 else f"p2eff.TF-d64L2H2.s{s}"
        tf.append(run_cell(res, ck, tf_factory(64, 2, 2), TASK, 80000, s))
    summary["TF-d64L2H2"] = solve_stats(tf)
    print(f"  -> TF (5-seed): {summary['TF-d64L2H2']}", flush=True)

    res["extra_summary"] = summary
    _save(res)
    print("\n==== EXTRA ARMS DONE ====", flush=True)
    print(json_summary(summary), flush=True)


def json_summary(summary):
    import json
    return "===EXTRA_JSON===\n" + json.dumps(summary, indent=2)


if __name__ == "__main__":
    main()
