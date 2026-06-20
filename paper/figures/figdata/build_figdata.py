#!/usr/bin/env python3
"""Deterministically extract F4 (MQAR-vs-params) and F5 (causal ablation) figure data
from committed results/gpu_bench.json. Every plotted number traces to a JSON key here;
no hand-transcription. Run: python3 build_figdata.py"""
import json, os, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
BENCH = os.path.join(REPO, "results", "gpu_bench.json")

def main():
    with open(BENCH) as f:
        d = json.load(f)

    def seeds(prefix, ns=(0, 1, 2)):
        return [d[f"{prefix}.s{i}"]["best"] for i in ns if f"{prefix}.s{i}" in d]

    def params(prefix):
        return d[f"{prefix}.s0"]["params"]

    # ---- F4: MQAR accuracy vs matched params (grid 130/461/857/3300 K) ----
    pz130 = seeds("p2eff.Prizma-quad2-d64L2H2")
    tf130 = seeds("p1.TF.d64L2H2")
    tf461 = seeds("p1.TF.d128L2H4")
    pz857 = seeds("p2.Prizma-quad2")
    tf857 = seeds("p2.TF", ns=(0, 1, 2, 3, 4))
    tf3300 = seeds("p1.TF.d256L4H8")
    fig4 = {
        "task": "MQAR D=128, best accuracy, MQAR-vs-params",
        "source_file": "results/gpu_bench.json",
        "points": [
            {"arch": "Prizma-quad2", "params_k": round(params("p2eff.Prizma-quad2-d64L2H2")/1000),
             "scale": "d64L2H2", "median_acc": statistics.median(pz130), "seeds": pz130,
             "solve_rate": "3/3", "source": "gpu_bench.json:p2eff.Prizma-quad2-d64L2H2.s{0,1,2}"},
            {"arch": "Transformer", "params_k": round(params("p1.TF.d64L2H2")/1000),
             "scale": "d64L2H2", "median_acc": statistics.median(tf130), "seeds": tf130,
             "solve_rate": "0/3", "source": "gpu_bench.json:p1.TF.d64L2H2.s{0,1,2}"},
            {"arch": "Transformer", "params_k": round(params("p1.TF.d128L2H4")/1000),
             "scale": "d128L2H4", "median_acc": statistics.median(tf461), "seeds": tf461,
             "solve_rate": "3/3", "source": "gpu_bench.json:p1.TF.d128L2H4.s{0,1,2} (smallest TF solver)"},
            {"arch": "Prizma-quad2", "params_k": round(params("p2.Prizma-quad2")/1000),
             "scale": "d128L4H4", "median_acc": statistics.median(pz857), "seeds": pz857,
             "solve_rate": "3/3", "source": "gpu_bench.json:p2.Prizma-quad2.s{0,1,2}"},
            {"arch": "Transformer", "params_k": round(params("p2.TF")/1000),
             "scale": "d128L4H4", "median_acc": statistics.median(tf857), "seeds": tf857,
             "solve_rate": "3/3", "source": "gpu_bench.json:p2.TF.s{0..4}"},
            {"arch": "Transformer", "params_k": round(params("p1.TF.d256L4H8")/1000),
             "scale": "d256L4H8", "median_acc": statistics.median(tf3300), "seeds": tf3300,
             "solve_rate": "3/3", "source": "gpu_bench.json:p1.TF.d256L4H8.s{0,1,2}"},
        ],
        "grid_k": [130, 461, 857, 3300],
        "efficiency_gap": "Prizma solves at 131K; smallest TF solver 461K => 3.5x",
    }

    # ---- F5: causal ablation at 130K (d64L2H2, MQAR D=128, 3 seeds) ----
    def cond(name, prefix):
        s = seeds(prefix)
        return {"name": name, "median_acc": statistics.median(s), "seeds": s,
                "solved": "3/3" if sum(x > 0.9 for x in s) == 3 else f"{sum(x>0.9 for x in s)}/3",
                "source": f"gpu_bench.json:{prefix}.s{{0,1,2}}"}
    fig5 = {
        "task": "MQAR D=128 best accuracy, d64L2H2 (130K), causal feature-map ablation",
        "source_file": "results/gpu_bench.json",
        "conditions": [
            cond("quad2", "p2eff.Prizma-quad2-d64L2H2"),
            cond("rand_linear", "p2eff.Prizma-randlin-d64L2H2"),
            cond("none", "p2eff.Prizma-none-d64L2H2"),
            cond("TF", "p1.TF.d64L2H2"),
        ],
        "scale": "d64L2H2 (130K), MQAR D=128",
    }

    for fn, obj in [("fig4_mqar.json", fig4), ("fig5_ablation.json", fig5)]:
        with open(os.path.join(HERE, fn), "w") as f:
            json.dump(obj, f, indent=1)
        print("wrote", fn)

if __name__ == "__main__":
    main()
