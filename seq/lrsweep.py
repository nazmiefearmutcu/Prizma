"""Per-config LR sweep so no architecture/width is denied an LR another gets. Stage-1: full grid @1
seed -> pick best on the frozen eval; Stage-2 is run by the caller at >=N seeds on the chosen LR.
Records the FULL grid incl. rejected LRs for the audit trail (committee guardrail #10)."""
from __future__ import annotations
import copy
from dataclasses import replace
from .common import build_and_train, TrainConfig

DEFAULT_GRID = (5e-4, 1e-3, 1.5e-3, 2e-3, 3e-3)

def sweep_lr(model_fac, task, base_cfg: TrainConfig, device, grid=DEFAULT_GRID, seed=0, **fac_kw):
    """Returns {'best_lr', 'best_acc', 'grid': [{'lr','best_acc','steps_to_plateau'}...]}."""
    rows = []
    for lr in grid:
        cfg = replace(base_cfg, lr=lr)
        r = build_and_train(model_fac, task, cfg, device, seed=seed, **fac_kw)
        rows.append({"lr": lr, "best_acc": r.best_acc, "steps_to_plateau": r.steps_to_plateau})
    best = max(rows, key=lambda d: d["best_acc"])
    return {"best_lr": best["lr"], "best_acc": best["best_acc"], "grid": rows}
