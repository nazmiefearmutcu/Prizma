"""Generate a self-contained Colab notebook (PRIZMA_D128_GPU.ipynb) that recreates the EXACT verified
seq/* package + gpu_bench.py + flop_ledger.py via %%writefile cells, mounts Drive, and runs the full
rigorous D=128 benchmark on a CUDA GPU, streaming results to Drive. No repo/clone needed.
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
MODULES = [
    ("seq/__init__.py", "seq/__init__.py"),
    ("seq/common.py", "seq/common.py"),
    ("seq/delta.py", "seq/delta.py"),
    ("seq/transformer.py", "seq/transformer.py"),
    ("seq/prizma_seq.py", "seq/prizma_seq.py"),
    ("seq/tasks.py", "seq/tasks.py"),
    ("gpu_bench.py", "gpu_bench.py"),
    ("flop_ledger.py", "flop_ledger.py"),
]


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src if isinstance(src, list) else [src]}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src if isinstance(src, list) else [src]}


def writefile_cell(target, path):
    with open(os.path.join(ROOT, path)) as f:
        content = f.read()
    if content and not content.endswith("\n"):
        content += "\n"
    return code([f"%%writefile {target}\n"] + content.splitlines(keepends=True))


cells = [
    md(["# Prizma-Seq vs Transformer — D=128 GPU benchmark\n",
        "**Run:** Runtime → Change runtime type → **GPU (A100/L4)** → then Runtime → **Run all**.\n",
        "Results stream to `Drive/MyDrive/prizma_results/gpu_bench.json` (resumable; safe to re-run after a disconnect).\n"]),
    code(["import torch, subprocess\n",
          "print('torch', torch.__version__, 'cuda', torch.cuda.is_available())\n",
          "print(subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],\n",
          "                     capture_output=True, text=True).stdout)\n",
          "assert torch.cuda.is_available(), 'Enable GPU: Runtime > Change runtime type > GPU'\n"]),
    code(["import os\n",
          "from google.colab import drive\n",
          "drive.mount('/content/drive')\n",
          "os.environ['PRIZMA_RESULTS'] = '/content/drive/MyDrive/prizma_results'\n",
          "os.makedirs(os.environ['PRIZMA_RESULTS'], exist_ok=True)\n",
          "os.makedirs('seq', exist_ok=True)\n",
          "print('results ->', os.environ['PRIZMA_RESULTS'])\n"]),
]
cells += [writefile_cell(t, p) for (t, p) in MODULES]
cells += [
    md(["## Self-tests (kernels) — should print ALL OK + step==forward <1e-6\n"]),
    code(["!python -m seq.delta | tail -4\n",
          "!python -m seq.prizma_seq\n"]),
    md(["## FLOP ledger (analytical disclosure: Prizma/TF forward-FLOP ratio)\n"]),
    code(["!python flop_ledger.py | grep -E 'per-token|RATIOS|as-coded|ideal'\n"]),
    md(["## Run the full benchmark (phases 1-5). Streams to Drive; hours on A100/L4.\n"]),
    code(["import sys\n",
          "sys.argv = ['gpu_bench']      # no args -> all phases (or e.g. ['gpu_bench','1','2'])\n",
          "import gpu_bench\n",
          "gpu_bench.main()\n"]),
    md(["## Final summary (also copied here for easy read-back)\n"]),
    code(["import json, os\n",
          "d = json.load(open(os.path.join(os.environ['PRIZMA_RESULTS'], 'gpu_bench.json')))\n",
          "print('===RESULTS_JSON_BEGIN===')\n",
          "print(json.dumps({k: v for k, v in d.items() if k.endswith('_summary') or k == 'p5_latency'}, indent=2))\n",
          "print('===RESULTS_JSON_END===')\n"]),
]

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "toc_visible": True},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

out = os.path.join(ROOT, "PRIZMA_D128_GPU.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print(f"wrote {out}  ({os.path.getsize(out)} bytes, {len(cells)} cells)")
