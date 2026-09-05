"""Local APU planning benchmark: Prizma-Seq LM fwd+bwd throughput on CPU vs the Radeon iGPU.

Machine: AMD Ryzen 7 PRO 5750G (8C/16T) with Radeon Graphics iGPU; Windows; no CUDA.
Repo venv: torch 2.14.0+cpu (CPU rows). Throwaway venv ~/.prizma_dml_venv: torch-directml
(iGPU rows). torch-directml pins an older torch — the exact versions each row ran with are
recorded in the JSON. These are PLANNING numbers for local zero-GPU work, not claims; run
from the repo root:  python results/local_apu_2026-09-03/bench_local.py cpu|dml
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from seq.prizma_seq import PrizmaSeqConfig, PrizmaSeqLM

MODE = sys.argv[1] if len(sys.argv) > 1 else "cpu"


def build_device():
    if MODE == "dml":
        import torch_directml

        return torch_directml.device(), torch.__version__
    torch.set_num_threads(int(os.environ.get("BENCH_THREADS", "8")))
    return "cpu", torch.__version__


def bench(B, T, d, H, device, n_warm=3, n_timed=10):
    cfg = PrizmaSeqConfig(vocab=64, d_model=d, n_layers=2, n_heads=H, chunk=64, window=16, max_len=T)
    m = PrizmaSeqLM(cfg).to(device)
    x = torch.randint(0, 64, (B, T), device=device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

    def step():
        opt.zero_grad()
        loss = m(x).float().pow(2).mean()
        loss.backward()
        opt.step()

    for _ in range(n_warm):
        step()
    if str(device) != "cpu":
        torch_directml.synchronize()  # noqa: F821 (dml mode only)
    t0 = time.perf_counter()
    for _ in range(n_timed):
        step()
    if str(device) != "cpu":
        torch_directml.synchronize()  # noqa: F821
    dt = (time.perf_counter() - t0) / n_timed
    return round(dt * 1000, 1), round(B * T / dt)


def main():
    device, torch_ver = build_device()
    rows = []
    for B, T, d, H in [(8, 1024, 64, 2), (8, 1024, 128, 4), (32, 1024, 128, 4)]:
        try:
            ms, tps = bench(B, T, d, H, device)
            rows.append(dict(mode=MODE, threads=os.environ.get("BENCH_THREADS", "-"), B=B, T=T,
                             d_model=d, H=H, ms_per_step=ms, tokens_per_s=tps, torch=torch_ver))
            print(rows[-1])
        except Exception as e:  # record, don't hide — DirectML op gaps are a real finding
            rows.append(dict(mode=MODE, B=B, T=T, d_model=d, H=H, error=f"{type(e).__name__}: {e}"))
            print(rows[-1])
    out = os.path.join(os.path.dirname(__file__), f"bench_{MODE}{'_' + os.environ.get('BENCH_THREADS', '') if MODE == 'cpu' else ''}.json")
    json.dump(rows, open(out, "w"), indent=2)
    print("WROTE", out)


if __name__ == "__main__":
    main()
