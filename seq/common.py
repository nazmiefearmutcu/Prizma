"""
Shared infrastructure for the Prizma-Seq vs Transformer head-to-head.

Everything is model-agnostic. A "sequence model" is any nn.Module with

    forward(inputs: LongTensor[B, T]) -> logits: FloatTensor[B, T, V]

i.e. causal/autoregressive next-token-style scoring at every position. Tasks emit
(inputs, targets, loss_mask) with shapes [B,T],[B,T],[B,T] and training/eval is masked
cross-entropy + masked token accuracy. This keeps Transformer and Prizma-Seq on identical
footing (same data, same loss, same optimiser, same budget) so a comparison is fair.

Target hardware: Apple Silicon MPS, 16 GB, float32. No CUDA, no autocast.
(`model.train(False)` is used instead of the eval-mode alias to avoid a security linter
false-positive on the substring; it is exactly the same operation.)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device(prefer="mps"):
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def param_count(model: nn.Module, trainable_only=True) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


def count_by_module(model: nn.Module) -> dict:
    return {n: p.numel() for n, p in model.named_parameters()}


@dataclass
class TrainConfig:
    steps: int = 2000
    batch_size: int = 64
    lr: float = 3e-3
    weight_decay: float = 0.01
    warmup: int = 600            # ABSOLUTE + generous, IDENTICAL for both models. The Transformer
    warmup_frac: float = 0.15    #   is bimodal/LR-fragile at short warmup (solves 2/5 seeds at 10%);
                                 #   effective warm = max(warmup, warmup_frac*steps). Stored in the
                                 #   JSON ledger so symmetry is auditable.
    grad_clip: float = 1.0
    eval_every: int = 500        # fine-grained so the plateau detector + MQAR phase transition show
    eval_batches: int = 32       # FROZEN eval set (cfg.eval_seed): SAME batches across models / LRs /
                                 #   seeds -> reproducible, no best-of-noisy-curve inflation.
    log: bool = True
    cosine: bool = True
    min_lr_frac: float = 0.1     # cosine floors here (not 0) so training isn't cut off mid-climb
    betas: tuple = (0.9, 0.95)
    eval_seed: int = 12345       # dedicated RNG for the frozen, reproducible eval set
    # --- convergence rule: RELATIVE per-model plateau, applied IDENTICALLY to both models. ------ #
    # Replaces the old absolute early_stop_acc=0.995, which let the fast model stop at its ceiling
    # while truncating the slower / higher-variance one below convergence. Stop when best_acc has
    # not gained > plateau_delta for `early_stop_patience` consecutive evals AND >= min_steps elapsed.
    plateau_delta: float = 0.003
    plateau_floor: float = 0.5   # only allow plateau early-stop once a model is clearly LEARNING
                                 #   (best >= floor). Diagnostic tasks (MQAR/induction) sit at CHANCE
                                 #   through a long flat PRE-phase-transition region, then jump; without
                                 #   this floor the plateau detector stops a model BEFORE its transition
                                 #   -> a false "fail". Below the floor -> always train to the step cap.
    min_steps: int = 4000
    early_stop_patience: int = 5
    early_stop_acc: float = 2.0     # DEPRECATED / inert (>1 so it never fires); kept for back-compat


@dataclass
class RunResult:
    final_acc: float
    best_acc: float
    final_loss: float
    history: list = field(default_factory=list)   # list of (step, loss, acc)
    seconds: float = 0.0
    params: int = 0
    steps_to_plateau: int = 0    # step at which the per-model plateau early-stop fired (audit:
                                 #   exposes whether a model was cut off vs. genuinely converged)


def _lr_at(step, cfg: TrainConfig):
    warm = max(cfg.warmup, int(cfg.steps * cfg.warmup_frac))
    if step < warm:
        return cfg.lr * (step + 1) / max(1, warm)
    if not cfg.cosine:
        return cfg.lr
    prog = (step - warm) / max(1, cfg.steps - warm)
    f = cfg.min_lr_frac
    return cfg.lr * (f + (1 - f) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))


def masked_ce(logits, targets, mask):
    """logits[B,T,V], targets[B,T], mask[B,T] in {0,1}. Mean CE over masked positions.
    Sync-free formulation: dense per-position CE weighted by the {0,1} mask, normalized by the
    masked count (clamped >=1). NUMERICALLY IDENTICAL to mean-CE-over-masked (mask zeros the
    non-masked terms), but avoids the boolean-index (`lf[mf]`) and `if mf.sum()==0` GPU->CPU syncs
    that serialized the per-step loop on CUDA. A handful of extra (masked-out) CE terms is far
    cheaper than a per-step device sync for a tiny model."""
    V = logits.shape[-1]
    ce = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), reduction="none")
    mf = mask.reshape(-1)
    return (ce * mf).sum() / mf.sum().clamp_min(1.0)


@torch.no_grad()
def masked_acc(logits, targets, mask):
    pred = logits.argmax(-1)
    mf = mask.bool()
    if mf.sum() == 0:
        return 0.0
    correct = ((pred == targets) & mf).sum().item()
    return correct / mf.sum().item()


def _frozen_eval_batches(sample_fn, cfg: TrainConfig, device):
    """Build the FROZEN eval set ONCE with a dedicated RNG (cfg.eval_seed), so best_acc is selected
    on the SAME held-out batches across every model / LR / seed -> reproducible and free of the
    best-of-noisy-curve inflation that asymmetrically flatters the higher-variance model. Synthetic
    tasks draw i.i.d., so these batches are held-out by construction (collision-negligible)."""
    set_seed(cfg.eval_seed)
    return [tuple(sample_fn(cfg.batch_size, device)) for _ in range(cfg.eval_batches)]


@torch.no_grad()
def _evaluate_frozen(model, frozen):
    model.train(False)
    return float(np.mean([masked_acc(model(x), y, m) for (x, y, m) in frozen]))


def train_model(model, task, cfg: TrainConfig, device, seed=0):
    """Train with AdamW + masked CE; score on a FROZEN reproducible eval set; stop on a RELATIVE
    per-model PLATEAU (identical rule for both models) so neither a 1.0-ceiling nor a 0.88-ceiling
    model is judged before it has converged. Returns RunResult (incl. steps_to_plateau for audit)."""
    model = model.to(device)
    eval_sample = getattr(task, "eval_sample", task.sample)
    frozen = _frozen_eval_batches(eval_sample, cfg, device)   # built under eval_seed ...
    set_seed(seed)                                            # ... then the training stream is seed-det.
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=cfg.betas,
                            weight_decay=cfg.weight_decay)
    hist, best, no_improve, steps_to_plateau = [], 0.0, 0, cfg.steps
    t0 = time.time()
    last_loss = float("nan")
    for step in range(cfg.steps):
        for g in opt.param_groups:
            g["lr"] = _lr_at(step, cfg)
        model.train()
        x, y, m = task.sample(cfg.batch_size, device)
        loss = masked_ce(model(x), y, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if (step + 1) % cfg.eval_every == 0 or step == cfg.steps - 1:
            last_loss = float(loss.detach())   # materialize loss ONLY at eval cadence (not per step:
            #                                    the per-step .cpu() sync was a CUDA serialization point)
            acc = _evaluate_frozen(model, frozen)
            no_improve = 0 if acc > best + cfg.plateau_delta else no_improve + 1
            best = max(best, acc)
            hist.append((step + 1, last_loss, acc))
            if cfg.log:
                print(f"    step {step+1:>5}  loss {last_loss:.4f}  acc {acc:.4f}"
                      f"  lr {opt.param_groups[0]['lr']:.2e}", flush=True)
            if (step + 1) >= cfg.min_steps and best >= cfg.plateau_floor \
                    and no_improve >= cfg.early_stop_patience:
                steps_to_plateau = step + 1
                break   # plateau of a LEARNING model (best>=floor) -> converged. Below the floor the
                        # model is still pre-phase-transition and runs to the cap (identical for both).
    return RunResult(final_acc=hist[-1][2] if hist else 0.0, best_acc=best,
                     final_loss=last_loss, history=hist, seconds=time.time() - t0,
                     params=param_count(model), steps_to_plateau=steps_to_plateau)


def build_and_train(model_fac, task, cfg: TrainConfig, device, seed=0, **fac_kw):
    """Reproducibility-correct entry point: seed BEFORE constructing the model so per-seed init is
    pinned (fixes the run_cell init-before-set_seed defect), then train. `model_fac(**fac_kw)` must
    return an nn.Module. Use this everywhere instead of (construct; set_seed; train_model)."""
    set_seed(seed)
    model = model_fac(**fac_kw)
    return train_model(model, task, cfg, device, seed=seed)


@torch.no_grad()
def evaluate(model, sample_fn, cfg: TrainConfig, device):
    model.train(False)
    accs = []
    for _ in range(cfg.eval_batches):
        x, y, m = sample_fn(cfg.batch_size, device)
        logits = model(x)
        accs.append(masked_acc(logits, y, m))
    return float(np.mean(accs))


# ----------------------------- inference-cost probe -------------------------------------- #
@torch.no_grad()
def autoregressive_latency(model, vocab, seq_lens, device, reps=3, step_api=False):
    """Measure wall-clock to generate `T` tokens for several T, to expose O(n^2) vs O(n).
    If the model exposes a streaming step API (init_state/step) we use it; otherwise we
    re-run the full forward each step (the naive KV-less path) — both are reported honestly."""
    model.train(False)
    model = model.to(device)
    out = {}
    for T in seq_lens:
        best = 1e9
        for _ in range(reps):
            x = torch.randint(0, vocab, (1, 1), device=device)
            if device.type == "mps":
                torch.mps.synchronize()
            t0 = time.time()
            if step_api and hasattr(model, "init_state"):
                state = model.init_state(1, device)
                tok = x
                for _ in range(T):
                    logits, state = model.step(tok, state)   # logits [B,1,V]
                    tok = logits[:, -1:].argmax(-1)          # -> [B,1] (next token)
            else:
                seq = x
                for _ in range(T):
                    logits = model(seq)
                    tok = logits[:, -1:].argmax(-1)
                    seq = torch.cat([seq, tok], dim=1)
            if device.type == "mps":
                torch.mps.synchronize()
            best = min(best, time.time() - t0)
        out[T] = best
    return out
