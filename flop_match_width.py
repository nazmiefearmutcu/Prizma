"""Phase 2c — the FAIR (width-scaled) FLOP-matched Transformer arm.

The depth-scaled FLOP-match (p2b, TF d128L9H4) confounds "more FLOPs" with "harder to optimize"
(deep TFs fail on MQAR at this scale — see P1 depth-bimodality: L2 clean, L4 bimodal, L9→0.02).
The FAIR FLOP-match keeps depth fixed (L=4) and scales WIDTH: TF d208L4H4 ≈ 4994 kFLOP/tok
(1.11× Prizma-quad2's as-coded forward FLOPs; flop_ledger.py). This avoids the depth-trainability
confound, so it is the honest test of "is Prizma's D=128 recall just spent compute?":
  - if d208L4H4 ALSO fails  -> Prizma-quad2's matched-param recall is not reachable by a same-budget
                               attention model at this scale (strong for Prizma);
  - if d208L4H4 SOLVES      -> equal FLOPs via width lets attention solve too; Prizma's edge is then
                               parameter-efficiency + O(1) memory, NOT raw capability (honest, still real).

Writes p2c.* into the SAME $PRIZMA_RESULTS/gpu_bench.json (resumable). Run on Colab after extra_arms:
    PRIZMA_RESULTS=/content/drive/MyDrive/prizma_results python -u flop_match_width.py
"""
from gpu_bench import run_cell, solve_stats, tf_factory, _load, _save, OUT, DEV
from seq.tasks import MixedMQAR

V = 512
S3 = (0, 1, 2)
TASK = lambda: MixedMQAR(vocab=V, max_pairs=128, num_queries=128, gap=0, min_pairs=1)


def main():
    print(f"device={DEV} results={OUT}  Phase 2c: FAIR width-scaled FLOP-matched TF d208L4H4 @ D=128", flush=True)
    res = _load()
    recs = [run_cell(res, f"p2c.TF-flopmatch-width-d208L4H4.s{s}", tf_factory(208, 4, 4), TASK, 80000, s)
            for s in S3]
    summary = {"TF-flopmatch-width-d208L4H4": solve_stats(recs)}
    res["p2c_summary"] = summary
    _save(res)
    print(f"  -> {summary['TF-flopmatch-width-d208L4H4']}", flush=True)
    print("\n==== Phase 2c DONE ====", flush=True)


if __name__ == "__main__":
    main()
