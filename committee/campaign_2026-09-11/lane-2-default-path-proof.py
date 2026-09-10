"""Default-path byte-identity proof for Lane 2 (campaign 2026-09-11).

Loads the PRE-EDIT seq/delta.py from git HEAD (before Lane 2's edits) as a separate module and
runs every default branch of `chunked_delta` against the CURRENT working-tree implementation on
identical inputs, asserting max|dO| == 0.0 and max|dS| == 0.0 (bit-identical), plus one gradient
check. `fast_reads=True` is excluded by design (opt-in, documented NOT byte-identical).

Run BEFORE the lane is committed (HEAD must still be the pre-edit delta.py):
    .venv\\Scripts\\python.exe committee\\campaign_2026-09-11\\lane-2-default-path-proof.py
"""
import importlib.util
import os
import subprocess
import sys
import tempfile

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from seq.delta import chunked_delta as new_chunked, _delta_reference  # noqa: E402

head_src = subprocess.run(["git", "show", "HEAD:seq/delta.py"], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout
tmp = os.path.join(tempfile.gettempdir(), "delta_preedit_head.py")
with open(tmp, "w", encoding="utf-8") as f:
    f.write(head_src)
spec = importlib.util.spec_from_file_location("delta_preedit", tmp)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
old_chunked = old.chunked_delta

torch.set_num_threads(8)
g = torch.Generator().manual_seed(20260911)
B, H, T, d = 2, 3, 300, 16
q = torch.randn(B, H, T, d, generator=g)
k = torch.randn(B, H, T, d, generator=g)
k = k / k.norm(dim=-1, keepdim=True)
v = torch.randn(B, H, T, d, generator=g)
beta = torch.rand(B, H, T, generator=g) * 0.99
alpha = 0.5 + 0.5 * torch.rand(B, H, T, generator=g)
eta = torch.rand(B, H, T, d, generator=g) * 0.99
beta_e = torch.rand(B, H, T, generator=g) * 0.99
g2 = torch.Generator().manual_seed(7)
k2 = torch.randn(B, H, T, 2, d, generator=g2)
k2 = k2 / k2.norm(dim=-1, keepdim=True)   # contraction needs ||k||=1 (as in tests/test_deltaproduct.py)
v2 = torch.randn(B, H, T, 2, d, generator=g2)
b2 = torch.rand(B, H, T, 2, generator=g2) * 0.99

cases = [
    ("pure", dict(alpha=None, chunk=64)),
    ("gated", dict(alpha=alpha, chunk=64)),
    ("gated floor", dict(alpha=torch.full((B, H, T), 0.5), chunk=64)),
    ("additive", dict(alpha=None, chunk=64, write_mode="additive")),
    ("gated additive", dict(alpha=alpha, chunk=64, write_mode="additive")),
    ("decoupled beta_e", dict(alpha=alpha, chunk=64, beta_e=beta_e)),
    ("eta gated", dict(alpha=alpha, chunk=64, eta=eta)),
    ("eta pure", dict(alpha=None, chunk=64, eta=eta)),
    ("surprise norm", dict(alpha=alpha, chunk=16, surprise=True, surprise_mode="norm")),
    ("surprise_norm A4", dict(alpha=None, chunk=64, surprise_norm=(8.0, 0.02, 0.9))),
    ("n_delta=2", dict(alpha=alpha, chunk=32, n_delta=2)),
]

allok = True
for name, kw in cases:
    if name == "n_delta=2":
        args = (q, k2, v2, b2)
    elif name == "surprise_norm A4":
        args = (q, k, v, None)      # the A4 gate REPLACES the learned beta (asserted in the kernel)
    else:
        args = (q, k, v, beta)
    O_new, S_new = new_chunked(*args, **kw)
    O_old, S_old = old_chunked(*args, **kw)
    dO = (O_new - O_old).abs().max().item()
    dS = (S_new - S_old).abs().max().item()
    ok = (dO == 0.0) and (dS == 0.0)
    allok &= ok
    print(f"[{name:<18}] max|dO|={dO:.3e} max|dS|={dS:.3e}  {'BIT-IDENTICAL' if ok else 'DIFFERS'}")


def grads(fn, fast):
    qq, kk, vv = q.clone().requires_grad_(True), k.clone().requires_grad_(True), v.clone().requires_grad_(True)
    bb = beta.clone().requires_grad_(True)
    aa = alpha.clone().requires_grad_(True)
    O, S = fn(qq, kk, vv, bb, alpha=aa, chunk=64, **({"fast_reads": fast} if fast is not None else {}))
    (O.square().mean() + S.square().mean()).backward()
    return qq.grad, kk.grad, vv.grad, bb.grad, aa.grad


old_g = grads(old_chunked, None)
new_g = grads(new_chunked, None)
worst = max((a - b).abs().max().item() for a, b in zip(old_g, new_g))
ok = worst == 0.0
allok &= ok
print(f"[grads gated       ] worst grad maxdiff={worst:.3e}  {'BIT-IDENTICAL' if ok else 'DIFFERS'}")
print("ALL OK" if allok else "FAILURES PRESENT")
sys.exit(0 if allok else 1)
