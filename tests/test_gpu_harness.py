"""Fast tests for the v2 campaign harness (seq/gpu_harness.py).

Mostly synthetic / minimal-training so the whole file runs in seconds. The harness composes the
already-tested primitives (build_and_train, seq.stats, seq.lrsweep); these tests pin the GLUE:

  * powered_summary merges seq.stats.summarize with solve_rate correctly.
  * h2h: a clear winner is superiority-significant; a tie is TOST-equivalent; verdict string sane.
  * negative-control LOGIC: superiority_test on two near-identical arrays is NOT significant
    (the integrity canary direction — identical models must not show a 'win').
  * make_arm builds valid 'tf' / 'prizma'(+ a v2 knob) / 'hybrid' models whose forward -> [B,T,V].
  * run_cell is SEED-PINNED: same seed + tiny config -> bit-identical best (mirrors test_repro.py).
  * run_cell caches by cellkey and writes crash-safe JSON.
"""
import json
import os
import tempfile

import torch

from seq import gpu_harness as h
from seq.common import TrainConfig, get_device


# --------------------------------------------------------------- powered_summary --
def test_powered_summary_on_known_array():
    accs = [1.0, 0.95, 0.92, 0.4, 0.8]            # 3 of 5 >= 0.9
    s = h.powered_summary(accs, solve_thresh=0.9)
    assert s["n"] == 5
    assert abs(s["mean"] - (sum(accs) / 5)) < 1e-9
    assert abs(s["median"] - 0.92) < 1e-9
    assert abs(s["solve_rate"] - 0.6) < 1e-9       # 3/5
    assert "ci95" in s and len(s["ci95"]) == 2


# --------------------------------------------------------------------------- h2h --
def test_h2h_clear_winner_is_superiority_significant():
    cand = [0.99, 0.98, 0.99, 1.0, 0.97]
    base = [0.50, 0.52, 0.48, 0.51, 0.49]
    r = h.h2h(cand, base, margin=0.05)
    assert bool(r["superiority"]["significant"]) is True
    assert r["superiority"]["p_value"] < 0.05
    assert "WIN" in r["verdict"].upper()


def test_h2h_tie_is_tost_equivalent():
    cand = [0.80, 0.81, 0.79, 0.80, 0.805]
    base = [0.80, 0.795, 0.81, 0.802, 0.798]
    r = h.h2h(cand, base, margin=0.05)
    assert bool(r["superiority"]["significant"]) is False
    assert bool(r["tost"]["equivalent"]) is True
    assert "EQUIVALENT" in r["verdict"].upper()


def test_h2h_lower_is_better_uses_margin_superiority():
    # BPC-style: lower is better. candidate clearly below baseline by > margin.
    cand = [1.00, 1.01, 0.99, 1.00, 1.02]          # candidate BPC (lower=better)
    base = [1.30, 1.31, 1.29, 1.30, 1.32]          # baseline BPC
    r = h.h2h(cand, base, margin=0.05, lower_is_better=True)
    assert bool(r["margin_superiority"]["significant"]) is True
    assert "WIN" in r["verdict"].upper()


def test_h2h_holm_reject_overrides_raw_significance_for_verdict():
    # A RAW-significant superiority that the family-wise Holm correction REJECTS must NOT read 'WIN'.
    # This is the verdict/holm reconciliation: verdict is computed from the corrected decision.
    cand = [0.99, 0.98, 0.99, 1.0, 0.97]
    base = [0.50, 0.52, 0.48, 0.51, 0.49]
    raw = h.h2h(cand, base, margin=0.05)
    assert bool(raw["superiority"]["significant"]) is True      # raw test IS significant
    corrected = h.h2h(cand, base, margin=0.05, holm_reject=False)
    # raw test object is preserved verbatim ...
    assert bool(corrected["superiority"]["significant"]) is True
    # ... but the verdict reflects the (failed) family-wise decision, never 'WIN'.
    assert "WIN" not in corrected["verdict"].upper()
    assert corrected["win_basis"] == "holm"


def test_h2h_holm_reject_true_yields_corrected_win():
    cand = [0.99, 0.98, 0.99, 1.0, 0.97]
    base = [0.50, 0.52, 0.48, 0.51, 0.49]
    r = h.h2h(cand, base, margin=0.05, holm_reject=True)
    assert "WIN" in r["verdict"].upper()
    assert "HOLM" in r["verdict"].upper()
    assert r["win_basis"] == "holm"


def test_h2h_default_verdict_is_labelled_uncorrected():
    cand = [0.99, 0.98, 0.99, 1.0, 0.97]
    base = [0.50, 0.52, 0.48, 0.51, 0.49]
    r = h.h2h(cand, base, margin=0.05)            # no holm_reject -> standalone single-test basis
    assert r["win_basis"] == "uncorrected"
    assert "UNCORRECTED" in r["verdict"].upper()


# --------------------------------------------------- negative-control canary LOGIC --
def test_negative_control_logic_two_near_identical_arrays_not_significant():
    a = [0.90, 0.91, 0.89, 0.905, 0.895]
    b = [0.901, 0.908, 0.892, 0.903, 0.897]
    from seq.stats import superiority_test
    st = superiority_test(a, b)
    assert bool(st["significant"]) is False
    assert st["p_value"] > 0.05         # canary direction: identical-ish -> NOT a win


def test_negative_control_draws_different_seeds_for_arm_b():
    # The canary must keep the architecture identical but VARY the seeds for arm B, otherwise the
    # two arms are bit-identical (delta=0, p=0.5) and the test is a tautology. Use a real (tiny)
    # MixedMQAR task so per-seed init genuinely diverges -> non-degenerate within-arm variance.
    from seq.tasks import MixedMQAR
    dev = get_device()
    task_fac = lambda: MixedMQAR(vocab=32, max_pairs=8, num_queries=16, gap=0, min_pairs=1)
    seeds = (0, 1, 2)
    cfg = TrainConfig(steps=60, eval_every=60, min_steps=0, batch_size=16, log=False)
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "nc.json")
        res = {}
        nc = h.negative_control(res, (64, 2, 2), task_fac, cfg, dev, seeds, out, seed_offset=100)
    # arm B drew DIFFERENT seeds than arm A — this is the whole fix (no longer a tautology).
    assert nc["seeds_a"] == [0, 1, 2]
    assert nc["seeds_b"] == [100, 101, 102]
    assert nc["seeds_a"] != nc["seeds_b"]
    # the ledger recorded arm B's cells under the OFFSET seeds (not arm A's seeds).
    assert res["negctrl.B.s100"]["seed"] == 100
    assert res["negctrl.A.s0"]["seed"] == 0
    # because the seeds differ, the two arms are NOT bit-identical: the per-seed accs are not the
    # frozen, exactly-equal pair the old degenerate control produced (accs_a == accs_b, delta 0.0).
    assert nc["accs_a"] != nc["accs_b"]
    assert nc["delta"] != 0.0
    # the canary still must NOT manufacture a 'win' from identical-architecture seed noise.
    assert nc["pass"] is True
    assert nc["p_value"] > 0.05


# --------------------------------------------------------------------- make_arm --
def _forward_shape_ok(model, V, T=24, B=2):
    x = torch.randint(0, V, (B, T))
    out = model(x)
    return tuple(out.shape) == (B, T, V)


def test_make_arm_tf_builds_valid_model():
    name, fac = h.make_arm("tf", d=32, L=2, H=2)
    model = fac(V := 24, T := 24)
    assert _forward_shape_ok(model, V, T)
    assert isinstance(name, str) and name


def test_make_arm_prizma_with_v2_knob_builds_valid_model():
    name, fac = h.make_arm("prizma", d=32, L=2, H=2, inctx_lr=True)
    model = fac(V := 24, T := 24)
    assert _forward_shape_ok(model, V, T)
    # the v2 knob must actually have propagated into the config
    assert model.cfg.inctx_lr is True


def test_make_arm_hybrid_builds_valid_model():
    name, fac = h.make_arm("hybrid", d=32, L=2, H=2)
    model = fac(V := 24, T := 24)
    assert _forward_shape_ok(model, V, T)


# ---------------------------------------------------------- run_cell seed-pinning --
def test_save_serializes_numpy_scalars_from_powered_stats():
    # h2h's output embeds numpy scalars (np.bool_ / np.float64) from seq.stats. The crash-safe
    # ledger MUST persist them; _save's numpy-aware default makes that round-trip.
    cand = [0.99, 0.98, 0.99, 1.0, 0.97]
    base = [0.50, 0.52, 0.48, 0.51, 0.49]
    payload = {"h2h": h.h2h(cand, base, margin=0.05),
               "summary": h.powered_summary(cand, solve_thresh=0.9)}
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "p.json")
        h._save(payload, out)
        back = json.load(open(out))
    assert back["h2h"]["superiority"]["significant"] is True   # numpy bool -> python bool
    assert isinstance(back["summary"]["solve_rate"], float)


class _TinyTask:
    vocab = 16
    seq_len = 12

    def sample(self, B, device):
        x = torch.randint(0, self.vocab, (B, self.seq_len), device=device)
        return x, x, torch.ones_like(x).float()


def _tiny_cfg():
    return TrainConfig(steps=3, eval_every=3, min_steps=0, batch_size=4, log=False)


def test_run_cell_is_seed_pinned_bit_reproducible():
    dev = get_device()
    _, fac = h.make_arm("tf", d=32, L=1, H=2)
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "r.json")
        res1, res2 = {}, {}
        r1 = h.run_cell(res1, "c", fac, _TinyTask, _tiny_cfg(), dev, seed=7, out_path=out)
        r2 = h.run_cell(res2, "c", fac, _TinyTask, _tiny_cfg(), dev, seed=7, out_path=out)
    assert abs(r1["best"] - r2["best"]) < 1e-6, (r1["best"], r2["best"])


def test_run_cell_caches_and_writes_crash_safe_json():
    dev = get_device()
    _, fac = h.make_arm("tf", d=32, L=1, H=2)
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "r.json")
        res = {}
        r1 = h.run_cell(res, "key1", fac, _TinyTask, _tiny_cfg(), dev, seed=0, out_path=out)
        # file exists and round-trips
        assert os.path.exists(out)
        on_disk = json.load(open(out))
        assert "key1" in on_disk and "best" in on_disk["key1"]
        # second call with same key returns the cached record (does NOT retrain)
        r2 = h.run_cell(res, "key1", fac, _TinyTask, _tiny_cfg(), dev, seed=0, out_path=out)
        assert r2 is res["key1"]
        assert r1["best"] == r2["best"]
