"""Produce the headline figure for PRISM: retention curves + forgetting + separability."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import structured_permuted_tasks
from prism import PRISM
from baselines import MLP, EWC
from metrics import AccuracyMatrix, accuracy

ROOT = os.path.join(os.path.dirname(__file__), "..")
K = 5

def retention(kind):
    tasks = structured_permuted_tasks(n_tasks=K, d=24, n_classes=8, seed=0)
    R = AccuracyMatrix(K)
    rng = np.random.default_rng(0)
    if kind == "PRISM":
        p = PRISM(d=24, h=48, K=8, n_experts=K + 3, seed=0)
        for i, t in enumerate(tasks):
            p.fit_task(t.Xtr, t.ytr, epochs=15, rng=rng)
            for j, tt in enumerate(tasks):
                R.record(i, j, accuracy(p.predict_logits(tt.Xte), tt.yte))
    else:
        m = MLP([24, 128, 128, 8], seed=0)
        for i, t in enumerate(tasks):
            m.fit_task(t.Xtr, t.ytr, epochs=15, lr=0.1, rng=rng)
            for j, tt in enumerate(tasks):
                R.record(i, j, accuracy(m.predict_logits(tt.Xte), tt.yte))
    return R.R

Rp, Rb = retention("PRISM"), retention("backprop")
res = json.load(open(os.path.join(ROOT, "results", "results.json")))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))

# Panel A: retention of task 0,1,2 as training proceeds (accuracy measured after each stage)
stages = np.arange(K)
for j, c in zip(range(3), ["#e63946", "#457b9d", "#2a9d8f"]):
    ax[0].plot(stages[j:], Rb[j:, j], "o--", color=c, alpha=0.55, lw=1.6,
               label=f"backprop T{j}")
    ax[0].plot(stages[j:], Rp[j:, j], "o-", color=c, lw=2.4, label=f"PRISM T{j}")
ax[0].set_title("A. Per-task retention through training\n(solid=PRISM, dashed=backprop)")
ax[0].set_xlabel("after training task #"); ax[0].set_ylabel("test accuracy on task j")
ax[0].set_ylim(0, 1); ax[0].legend(fontsize=7, ncol=2); ax[0].grid(alpha=0.25)

# Panel B: forgetting bars (E1)
rows = res["E1_main"]["rows"]
names = ["backprop", "EWC", "replay(boundaries)", "PRISM(DFA,no W^T)", "PRISM_noRoute(ablation)"]
fgt = [rows[n]["FGT"][0] for n in names]
err = [rows[n]["FGT"][1] for n in names]
cols = ["#9aa0a6", "#f4a261", "#e9c46a", "#2a9d8f", "#b0b0b0"]
ax[1].bar(range(len(names)), fgt, yerr=err, color=cols, capsize=4)
ax[1].set_xticks(range(len(names)))
ax[1].set_xticklabels(["backprop", "EWC\n(uses\nbound.)", "replay\n(uses\nbuffer)", "PRISM\n(no labels)", "PRISM\nnoRoute"], fontsize=7.5)
ax[1].set_title("B. Forgetting (lower better), 10 seeds ±95% CI")
ax[1].set_ylabel("FGT"); ax[1].grid(alpha=0.25, axis="y")

# Panel C: separability sweep (E2)
sw = res["E2_separability"]
noise = [r["noise"] for r in sw]
ax[2].plot(noise, [r["PRISM_ACC"] for r in sw], "o-", color="#2a9d8f", lw=2.2, label="PRISM ACC")
ax[2].plot(noise, [r["backprop_ACC"] for r in sw], "o--", color="#9aa0a6", lw=1.8, label="backprop ACC")
ax[2].plot(noise, [r["PRISM_FGT"] for r in sw], "s-", color="#e63946", lw=2.2, label="PRISM FGT")
ax[2].plot(noise, [r["backprop_FGT"] for r in sw], "s--", color="#e07a7a", lw=1.8, label="backprop FGT")
ax[2].set_title("C. Separability sweep\n(noise blurs domains)")
ax[2].set_xlabel("input noise_std"); ax[2].set_ylabel("ACC / FGT")
ax[2].set_ylim(0, 1); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.25)

plt.tight_layout()
out = os.path.join(ROOT, "results", "figure.png")
plt.savefig(out, dpi=130)
print("saved", out)
