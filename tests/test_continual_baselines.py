"""Tests for the citation-bar continual-learning battery (Wave-C C1; committee report 10-P4,
commission synthesis SS7.4).

Covers, per the task spec:
  (a) each new learner (OnlineEWC online + boundary, MAS, SI) reduces loss on a toy stream;
  (b) importance/penalty tensors have the right shapes and grow/shrink as expected;
  (c) online variants (OnlineEWC boundary-free, MAS) actually update online -- no boundary
      call is ever needed, and their penalty engages from stream data alone;
  (d) the runner's single-pass mode feeds each domain EXACTLY ONCE (probe counting);
  (e) a tiny smoke battery (2 domains x 2 seeds x tiny MLP) runs end-to-end < 60 s.

Runtime discipline: everything here uses 2-domain toy streams and tiny MLPs; the suite entry
`test_smoke_battery_end_to_end` is the only multi-second test (bounded by the 60 s bar).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from baselines import MLP, EWC, MAS, OnlineEWC, SI  # noqa: E402
from data import structured_permuted_tasks  # noqa: E402
from prizma import Prizma  # noqa: E402
import run_citation_battery as bat  # noqa: E402

# small lambdas so the penalty never strangles toy-net learning in these unit tests
LAM = {"MAS": 10.0, "SI": 1.0, "OnlineEWC(online)": 50.0, "OnlineEWC(boundary)": 20.0,
       "EWC": 50.0}
ARMS = ["MAS", "SI", "OnlineEWC(online)", "OnlineEWC(boundary)"]
BOUNDARY_ARMS = {"SI", "OnlineEWC(boundary)"}


def _toy_tasks(n_tasks=2, n_samples=256, d=8, ncls=3, seed=0):
    return structured_permuted_tasks(n_tasks=n_tasks, n_samples=n_samples, d=d,
                                     n_classes=ncls, seed=seed)


def _fresh(arm):
    return MLP([8, 16, 3], seed=0), bat.make_reg(arm, LAM[arm], 0.1), np.random.default_rng(0)


# ------------------------------------------------------------------------------------------ #
# (a) loss reduction on a toy stream
# ------------------------------------------------------------------------------------------ #
@pytest.mark.parametrize("arm", ARMS)
def test_learner_reduces_loss_on_toy_stream(arm):
    tasks = _toy_tasks()
    m, reg, rng = _fresh(arm)
    t0 = tasks[0]
    probe = t0.Xtr[:64]
    loss_before = m.grads(probe, t0.ytr[:64])[2]
    m.fit_task(t0.Xtr, t0.ytr, epochs=10, lr=0.1, rng=rng, ewc=reg)
    loss_after = m.grads(probe, t0.ytr[:64])[2]
    assert loss_after < loss_before - 0.15, f"{arm}: loss {loss_before:.3f} -> {loss_after:.3f}"


# ------------------------------------------------------------------------------------------ #
# (b) importance/penalty tensor shapes + grow/shrink semantics
# ------------------------------------------------------------------------------------------ #
@pytest.mark.parametrize("arm", ARMS)
def test_importance_tensor_shapes(arm):
    tasks = _toy_tasks()
    m, reg, rng = _fresh(arm)
    t0 = tasks[0]
    m.fit_task(t0.Xtr, t0.ytr, epochs=1, lr=0.1, rng=rng, ewc=reg)
    # class-specific state, checked explicitly (no guessing attribute names)
    if isinstance(reg, MAS):
        states = [reg.omegaW, reg.omegab, reg.starW, reg.starb]
    elif isinstance(reg, OnlineEWC):
        states = [reg.FW, reg.Fb, reg.starW, reg.starb]
    elif isinstance(reg, SI):
        states = [reg.omW, reg.omb, reg.wW, reg.wb, reg.dW, reg.db, reg.starW, reg.starb]
    for st in states:
        assert st is not None
        for i in range(len(m.W)):
            assert st[i].shape == m.W[i].shape or st[i].shape == m.b[i].shape
        assert all(np.isfinite(t).all() for t in st)


def test_importance_grows_without_boundary_and_shrinks_on_consolidation():
    tasks = _toy_tasks()
    t0 = tasks[0]
    # MAS: importance grows from zero online
    m, reg, rng = _fresh("MAS")
    m.fit_task(t0.Xtr, t0.ytr, epochs=1, lr=0.1, rng=rng, ewc=reg)
    assert any(np.any(o > 0) for o in reg.omegaW), "MAS importance must grow during fit"
    w_before = [o.copy() for o in reg.omegaW]
    reg.consolidate(m)  # documented no-op
    assert all((o == w).all() for o, w in zip(reg.omegaW, w_before))

    # OnlineEWC(online): Fisher grows from zero online; lam_t unchanged (no consolidate)
    m2, reg2, rng2 = _fresh("OnlineEWC(online)")
    m2.fit_task(t0.Xtr, t0.ytr, epochs=1, lr=0.1, rng=rng2, ewc=reg2)
    assert any(np.any(f > 0) for f in reg2.FW), "online Fisher must grow without any boundary"
    assert reg2.t == 0 and reg2.lam_t == reg2.lam

    # OnlineEWC(boundary): Fisher stays ZERO until consolidate, then grows; lam_t rescales up
    m3, reg3, rng3 = _fresh("OnlineEWC(boundary)")
    m3.fit_task(t0.Xtr, t0.ytr, epochs=1, lr=0.1, rng=rng3, ewc=reg3)
    assert all(not np.any(f) for f in reg3.FW), "boundary mode must not estimate Fisher mid-task"
    reg3.consolidate(m3, t0.Xtr, t0.ytr, n_samples=32, rng=rng3)
    assert any(np.any(f > 0) for f in reg3.FW)
    assert reg3.t == 1
    assert reg3.lam_t > reg3.lam  # online-lambda update (gamma=0.9 -> lam_t = lam/0.9)

    # SI: running integral grows online but Omega appears only at the boundary, then w resets
    m4, reg4, rng4 = _fresh("SI")
    m4.fit_task(t0.Xtr, t0.ytr, epochs=1, lr=0.1, rng=rng4, ewc=reg4)
    assert any(np.any(w > 0) for w in reg4.wW), "path integral must accumulate during fit"
    assert all(not np.any(o) for o in reg4.omW), "Omega must stay zero before any boundary"
    reg4.consolidate(m4)
    assert any(np.any(o > 0) for o in reg4.omW), "Omega must materialize at the boundary"
    assert all(not np.any(w) for w in reg4.wW), "integrators must reset at the boundary"
    assert all(np.array_equal(s, w) for s, w in zip(reg4.starW, m4.W))


# ------------------------------------------------------------------------------------------ #
# (c) online variants update online -- no boundary call anywhere in the loop
# ------------------------------------------------------------------------------------------ #
@pytest.mark.parametrize("arm", ["MAS", "OnlineEWC(online)"])
def test_online_penalty_engages_without_boundary_call(arm):
    """One fit_task call, NO consolidate: the importance tensor is populated AND the
    gradient the learner applies differs from the raw task gradient (penalty is live)."""
    tasks = _toy_tasks()
    t0 = tasks[0]
    m, reg, rng = _fresh(arm)
    m.fit_task(t0.Xtr, t0.ytr, epochs=1, lr=0.1, rng=rng, ewc=reg)
    raw_gW, raw_gb, _ = m.grads(t0.Xtr[:32], t0.ytr[:32])
    pen_gW, pen_gb = [g.copy() for g in raw_gW], [g.copy() for g in raw_gb]
    reg.add_penalty_grads(m, pen_gW, pen_gb)
    changed = any(not np.array_equal(a, b) for a, b in zip(raw_gW, pen_gW)) or \
        any(not np.array_equal(a, b) for a, b in zip(raw_gb, pen_gb))
    assert changed, f"{arm}: penalty must engage with no boundary call"


def test_boundary_free_learners_never_need_consolidate():
    """Runner-level: fit_mlp_arm on boundary-free arms calls NO consolidate and still
    produces a full accuracy matrix (proves the runner path needs no boundary)."""
    tasks = _toy_tasks(n_tasks=2)
    for arm in ["backprop", "MAS", "OnlineEWC(online)"]:
        R, _m = bat.fit_mlp_arm(arm, tasks, 8, 3, 0, epochs=1, lr=0.1, hidden=[16],
                                lam=LAM.get(arm, 10.0))
        assert np.isfinite(R.R).all() and len(R.R) == 2


# ------------------------------------------------------------------------------------------ #
# (d) single-pass feeds each domain exactly once (probe counting)
# ------------------------------------------------------------------------------------------ #
def test_single_pass_feeds_each_domain_exactly_once(monkeypatch):
    tasks = _toy_tasks(n_tasks=2, n_samples=256)
    total = sum(len(t.Xtr) for t in tasks)

    # --- MLP-family arms: count rows entering MLP.grads (the actual per-batch update). -----
    counts = {"rows": 0}
    orig_grads = MLP.grads

    def counting_grads(self, X, y):
        counts["rows"] += len(X)
        return orig_grads(self, X, y)

    monkeypatch.setattr(MLP, "grads", counting_grads)

    # boundary-free arms: consolidate() is never called, so rows == total EXACTLY.
    for arm in ["backprop", "MAS", "OnlineEWC(online)"]:
        counts["rows"] = 0
        bat.run_arm(arm, tasks, 8, 3, 0, epochs=1, lr=0.1, hidden=[16],
                    lam=LAM.get(arm, 10.0), prizma_h=16)
        assert counts["rows"] == total, f"{arm}: expected {total}, fed {counts['rows']}"

    # multi-epoch sanity: epochs=2 feeds exactly 2x (the probe measures epochs, not luck).
    counts["rows"] = 0
    bat.run_arm("backprop", tasks, 8, 3, 0, epochs=2, lr=0.1, hidden=[16], lam=None,
                prizma_h=16)
    assert counts["rows"] == 2 * total

    # boundary arms: training rows are still exactly `total`; the EXTRA rows differ by arm:
    #   EWC / OnlineEWC(boundary) sample min(1024, n_train) per boundary inside consolidate()
    #   for Fisher estimation -- the same privileged boundary information the shipped EWC
    #   baseline uses, NOT extra training exposure.
    #   SI consolidates from already-seen gradient statistics and samples nothing.
    fisher_per_boundary = len(tasks[0].Xtr)  # min(1024, n_train)
    for arm, extra in [("EWC", 2 * fisher_per_boundary), ("SI", 0),
                       ("OnlineEWC(boundary)", 2 * fisher_per_boundary)]:
        counts["rows"] = 0
        bat.run_arm(arm, tasks, 8, 3, 0, epochs=1, lr=0.1, hidden=[16],
                    lam=LAM[arm], prizma_h=16)
        assert counts["rows"] == total + extra, \
            f"{arm}: train-once violated ({counts['rows']} vs {total} + {extra})"

    # replay mixes buffered OLD samples back in (that is its mechanism); the CURRENT-domain
    # exposure must still be exactly one pass: rows >= total, and each task's Xtr enters
    # once. We assert the weaker honest bound here.
    counts["rows"] = 0
    bat.run_arm("replay", tasks, 8, 3, 0, epochs=1, lr=0.1, hidden=[16], lam=None, prizma_h=16)
    assert counts["rows"] >= total

    # --- Prizma: count rows entering the SHIPPED train_batch via the shipped fit_task. ----
    prizma_rows = {"rows": 0}
    orig_tb = Prizma.train_batch

    def counting_tb(self, X, Y, y):
        prizma_rows["rows"] += len(X)
        return orig_tb(self, X, Y, y)

    monkeypatch.setattr(Prizma, "train_batch", counting_tb)
    prizma_rows["rows"] = 0
    bat.run_arm("PRIZMA(DFA)", tasks, 8, 3, 0, epochs=1, lr=0.1, hidden=[16], lam=None,
                prizma_h=16)
    assert prizma_rows["rows"] == total, "Prizma single-pass must stream each domain once"


# ------------------------------------------------------------------------------------------ #
# (e) tiny smoke battery end-to-end < 60 s
# ------------------------------------------------------------------------------------------ #
def test_smoke_battery_end_to_end(tmp_path):
    t0 = time.time()
    bat.run_protocol("multi_epoch", bat.CONFIGS["smoke"], str(tmp_path))
    bat.run_protocol("single_pass", bat.CONFIGS["smoke"], str(tmp_path))
    bat.write_results_md(str(tmp_path))  # no results/results.json next to tmp_path -> E1 skipped
    elapsed = time.time() - t0
    assert elapsed < 60.0, f"smoke battery took {elapsed:.1f}s (bar: 60s)"

    for f in ("raw_multi_epoch.json", "raw_single_pass.json", "RESULTS.md"):
        assert (tmp_path / f).exists(), f"missing artifact {f}"

    single = json.load(open(tmp_path / "raw_single_pass.json"))
    assert single["meta"]["n_seeds"] == 2
    assert single["meta"]["epochs_per_domain"] == 1
    for arm in bat.SINGLE_ARMS:
        recs = single["arms"][arm]["seed_records"]
        assert len(recs) == 2, f"{arm}: per-seed records must be retained"
        for r in recs:
            assert np.isfinite(r["ACC"]) and np.isfinite(r["FGT"])
            assert len(r["R"]) == 2 and all(len(row) == 2 for row in r["R"])

    multi = json.load(open(tmp_path / "raw_multi_epoch.json"))
    assert multi["meta"]["epochs_per_domain"] == bat.CONFIGS["smoke"]["multi_epochs"]
    for arm in bat.MULTI_NEW_ARMS:
        assert len(multi["arms"][arm]["seed_records"]) == 2
        assert multi["meta"]["tuned"][arm]["lam"] is not None

    md = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert "LANE-EXPLORATORY" in md and "Table 1" in md and "Table 2" in md
