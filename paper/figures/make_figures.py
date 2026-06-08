#!/usr/bin/env python3
"""Generate all Prizma-Seq paper figures from committed results/ JSONs. No new compute.

  python3 figdata/build_figdata.py   # refresh fig4/fig5 data from gpu_bench.json
  python3 make_figures.py            # emit fig1..fig5 PDFs

Every data figure traces to results/*.json (see figdata/PROVENANCE.md)."""
import json, os, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(REPO, "results")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "figure.dpi": 150})
C_TF, C_PZ = "#c0392b", "#1f6f3f"   # transformer red, prizma green


def _load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def fig1_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 2.7)); ax.axis("off")
    ax.set_xlim(0, 12); ax.set_ylim(0, 4)

    def box(x, y, w, h, t, fc):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                     fc=fc, ec="black", lw=1.1))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8.5)

    box(0.2, 1.4, 1.8, 1.1, "$k_t,q_t$\n$\\in\\mathbb{R}^{d_h=32}$", "#eaf2ff")
    ax.annotate("", (3.2, 1.95), (2.0, 1.95), arrowprops=dict(arrowstyle="-|>", lw=1.2))
    box(3.2, 1.4, 2.5, 1.1, "$\\phi$: fixed seeded\nmonomials\n(0 params)", "#fff2cc")
    ax.annotate("", (6.4, 1.95), (5.7, 1.95), arrowprops=dict(arrowstyle="-|>", lw=1.2))
    box(6.4, 1.4, 2.0, 1.1, "$\\phi(k_t)$\n$\\in\\mathbb{R}^{d_\\phi=256}$", "#eaf2ff")
    ax.annotate("", (9.3, 1.95), (8.4, 1.95), arrowprops=dict(arrowstyle="-|>", lw=1.2))
    box(9.3, 0.5, 2.5, 3.0,
        "rectangular state\n$S_t\\in\\mathbb{R}^{32\\times256}$\n\n"
        "write $+\\,\\beta_t\\varepsilon^{\\phi}_t\\phi(k_t)^{\\!\\top}$\n"
        "read $S_{t-1}\\phi(q_t)$", "#e6ffe6")
    ax.text(6.0, 0.15,
            "square $d_h\\times d_h$  $\\rightarrow$  rectangular $d_h\\times d_\\phi$"
            "   ($O(1)$ inference preserved, $+0$ params)",
            ha="center", fontsize=8, style="italic")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig1_architecture.pdf")); plt.close(fig)


def fig2_latency():
    d = _load("gpu_latency.json"); ns = d["meta"]["ns"]; c = d["cells"]
    tf = [c[f"TF.small.n{n}"]["per_step_ms"] for n in ns]
    pz = [c[f"Prizma-quad2.small.n{n}"]["per_step_ms"] for n in ns]
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    ax.plot(ns, tf, "o-", color=C_TF, label="Transformer (KV-cache)")
    ax.plot(ns, pz, "s-", color=C_PZ, label="Prizma-Seq ($O(1)$ state)")
    ax.axvline(32768, ls="--", color="gray", lw=1)
    ax.annotate("crossover\n$n=32$k", (32768, 12.5), fontsize=8, color="gray", ha="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns); ax.set_xticklabels([f"{n//1024}k" for n in ns])
    ax.set_xlabel("sequence length $n$"); ax.set_ylabel("per-step decode latency (ms)")
    ax.set_title("Decode latency vs. context (small config, A100)")
    ax.legend(fontsize=8, loc="upper left"); fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig2_latency.pdf")); plt.close(fig)


def fig3_memory():
    d = _load("gpu_latency.json"); ns = d["meta"]["ns"]; c = d["cells"]
    tf_mb = [c[f"TF.small.n{n}"]["peak_bytes"] / 1e6 for n in ns]
    pz_mb = [c[f"Prizma-quad2.small.n{n}"]["peak_bytes"] / 1e6 for n in ns]
    tf_fl = [c[f"TF.small.n{n}"]["kv_floats"] for n in ns]
    pz_fl = [c[f"Prizma-quad2.small.n{n}"]["state_floats"] for n in ns]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.3))
    for ax in (a1, a2):
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(ns); ax.set_xticklabels([f"{n//1024}k" for n in ns]); ax.set_xlabel("$n$")
    a1.plot(ns, tf_mb, "o-", color=C_TF, label="Transformer")
    a1.plot(ns, pz_mb, "s-", color=C_PZ, label="Prizma-Seq")
    a1.set_ylabel("peak process memory (MB)")
    a1.set_title("Measured peak (2.8$\\times$ at 4k $\\to$ 31$\\times$ at 64k)"); a1.legend(fontsize=8)
    a2.plot(ns, tf_fl, "o-", color=C_TF, label="KV-cache floats")
    a2.plot(ns, pz_fl, "s-", color=C_PZ, label="Prizma state floats")
    a2.set_ylabel("carried floats")
    a2.set_title("Analytic state vs. KV (28$\\times$ $\\to$ 455$\\times$)"); a2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig3_memory.pdf")); plt.close(fig)


def fig4_mqar():
    with open(os.path.join(HERE, "figdata", "fig4_mqar.json")) as f:
        d = json.load(f)
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    for arch, mk, col in [("Prizma-quad2", "s", C_PZ), ("Transformer", "o", C_TF)]:
        pts = sorted([p for p in d["points"] if p["arch"] == arch], key=lambda p: p["params_k"])
        xs = [p["params_k"] for p in pts]; ys = [p["median_acc"] for p in pts]
        ax.plot(xs, ys, mk + "-", color=col, label=arch)
    ax.axhline(0.9, ls="--", color="gray", lw=1)
    ax.text(150, 0.84, "solve bar (0.9)", fontsize=8, color="gray")
    ax.annotate("Prizma solves\nat 131K", (131, 0.997), (170, 0.62),
                fontsize=8, color=C_PZ, arrowprops=dict(arrowstyle="->", color=C_PZ))
    ax.annotate("smallest TF\nsolver 461K", (461, 0.9999), (520, 0.45),
                fontsize=8, color=C_TF, arrowprops=dict(arrowstyle="->", color=C_TF))
    ax.set_xscale("log"); ax.set_xlabel("matched parameters (K)")
    ax.set_ylabel("MQAR $D{=}128$ accuracy (median)"); ax.set_ylim(-0.05, 1.08)
    ax.set_title("Parameter-efficiency: $\\geq$3.5$\\times$ (131K vs. 461K)")
    ax.legend(fontsize=8, loc="center right"); fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig4_mqar.pdf")); plt.close(fig)


def fig5_ablation():
    with open(os.path.join(HERE, "figdata", "fig5_ablation.json")) as f:
        d = json.load(f)
    conds = d["conditions"]
    labels = {"quad2": "quad2", "rand_linear": "rand_linear", "none": "none", "TF": "TF"}
    names = [labels[c["name"]] for c in conds]
    vals = [c["median_acc"] for c in conds]
    errs = [[c["median_acc"] - min(c["seeds"]) for c in conds],
            [max(c["seeds"]) - c["median_acc"] for c in conds]]
    cols = [C_PZ, "#7f8c8d", "#b0b0b0", C_TF]
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    ax.bar(names, vals, yerr=errs, capsize=5, color=cols, edgecolor="black", lw=0.6)
    ax.axhline(0.9, ls="--", color="gray", lw=1); ax.text(2.6, 0.92, "solve bar", fontsize=8, color="gray")
    for i, c in enumerate(conds):
        ax.text(i, max(c["seeds"]) + 0.04, c["solved"], ha="center", fontsize=8)
    ax.set_ylabel("MQAR $D{=}128$ accuracy (median, 130K)"); ax.set_ylim(0, 1.12)
    ax.set_title("Causal ablation: the gain is the quadratic monomials")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig5_ablation.pdf")); plt.close(fig)


if __name__ == "__main__":
    fig1_architecture(); fig2_latency(); fig3_memory(); fig4_mqar(); fig5_ablation()
    print("wrote fig1_architecture, fig2_latency, fig3_memory, fig4_mqar, fig5_ablation (.pdf)")
