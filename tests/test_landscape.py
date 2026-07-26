"""Tests for the SOTA-LANDSCAPE powered head-to-head runner (seq/landscape.py).

seq/landscape.py is seq/recall_gate.py GENERALIZED from TF-vs-Prizma to the 4-arm SOTA landscape
(TF / Prizma-v2 / GLA / Mamba-2). It REUSES the campaign primitives in seq/gpu_harness.py
(make_arm, sweep_then_seeds, powered_summary, h2h, holm_family, negative_control, _save,
load_results) — these tests pin the GLUE that is new here:

  (A) PURE VERDICT LAYER  landscape_verdict(arm_accs, ...)  — training-free, deterministic:
        * ranks the arms into a Pareto table (sorted by mean accuracy, higher-is-acc default),
        * computes Prizma-vs-EACH-baseline pairwise verdicts (BEATS / PARITY / WORSE) using
          h2h (margin_superiority + TOST) with Holm family correction over the pair p-values,
        * handles the lower-is-LOSS sign (lower_is_better=True flips BEATS/WORSE correctly).
      Tested on SYNTHETIC per-arm accuracy arrays — no torch, no training.

  (B) SMOKE END-TO-END  run_landscape(smoke=True)  — CPU-fast (<~60s): tiny d/L/H, 2 seeds, 1
      task, small cap. Asserts it completes, writes the JSON ledger with all 4 arms + per-task
      pairwise verdicts + the negative-control leg, and is RESUMABLE (a second run SKIPs cached
      cells).

  (C) CLI ARG-GUARD  main()  — an unknown flag exits NON-ZERO (never silently launches the
      multi-hour full landscape), mirroring recall_gate T8.

  (D) CHAR-LM LEG  run_landscape_charlm(...)  — the LANGUAGE-MODELING axis of the same 4 arms, on
      char-LM BPC (lower-is-better). A CPU-fast smoke trains all 4 arms on a tiny shakespeare slice,
      writes results/gpu_landscape_charlm.json with the per-pair BPC verdicts + the negative control,
      and is resumable. The PRIZMA_CHARLM_KNOBS gate-superset (gates ON for char-LM) is pinned DISTINCT
      from the recall PRIZMA_V2_KNOBS (gates OFF for clean overwrite), and a synthetic lower-is-better
      verdict sanity reuses the shared landscape_verdict.
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest


# =============================================================== module surface ==
def test_module_imports_and_public_surface():
    import seq.landscape as ls
    assert callable(ls.landscape_verdict), "pure verdict layer must exist"
    assert callable(ls.run_landscape), "training runner must exist"
    assert callable(ls.main), "CLI entry must exist"
    assert callable(ls._build_parser), "arg-guard parser must be unit-testable"
    # The Prizma arm config must mirror seq/recall_gate.py's REAL-run Prizma (this runner uses recall
    # tasks): feat_map='quad2_lowrank' with the forget/output GATES OFF — recall diagnostics need a clean
    # overwrite, not the char-LM gate superset. (See PRIZMA_V2_KNOBS docstring + seq/delta.py.)
    assert isinstance(ls.PRIZMA_V2_KNOBS, dict)
    assert ls.PRIZMA_V2_KNOBS.get("feat_map") == "quad2_lowrank"
    for k in ("out_gate", "state_norm", "decoupled_gate", "gated"):
        assert k not in ls.PRIZMA_V2_KNOBS, f"recall Prizma must NOT set the char-LM gate knob {k}"


# ===================================================== PURE VERDICT: ranking =====
def test_verdict_ranks_arms_by_mean_accuracy_higher_is_acc():
    """Higher-is-accuracy: the Pareto table is sorted best-first by mean accuracy."""
    from seq.landscape import landscape_verdict
    rng = np.random.default_rng(0)
    arm_accs = {
        "Prizma": list(0.97 + 0.004 * rng.standard_normal(8)),
        "TF":     list(0.95 + 0.004 * rng.standard_normal(8)),
        "GLA":    list(0.80 + 0.01 * rng.standard_normal(8)),
        "Mamba2": list(0.90 + 0.01 * rng.standard_normal(8)),
    }
    params = {"Prizma": 30000, "TF": 26000, "GLA": 28000, "Mamba2": 28800}
    v = landscape_verdict(arm_accs, cand_key="Prizma", params=params, margin=0.05)
    table = v["pareto_table"]
    means = [row["mean"] for row in table]
    assert means == sorted(means, reverse=True), f"table must be best-first, got {means}"
    # best arm here is Prizma; worst is GLA.
    assert table[0]["name"] == "Prizma"
    assert table[-1]["name"] == "GLA"
    # every row carries the powered summary fields + params + per-seed accs.
    for row in table:
        assert "ci95" in row and "solve_rate" in row and "median" in row
        assert "params" in row and row["params"] == params[row["name"]]
        assert "accs" in row and len(row["accs"]) == 8


def test_verdict_ranks_lower_is_loss_correctly():
    """Lower-is-LOSS (e.g. BPC): the table must rank the LOWEST value best-first."""
    from seq.landscape import landscape_verdict
    rng = np.random.default_rng(1)
    arm_accs = {
        "Prizma": list(1.10 + 0.01 * rng.standard_normal(6)),   # best (lowest loss)
        "TF":     list(1.20 + 0.01 * rng.standard_normal(6)),
        "GLA":    list(1.50 + 0.02 * rng.standard_normal(6)),   # worst (highest loss)
    }
    v = landscape_verdict(arm_accs, cand_key="Prizma", margin=0.05, lower_is_better=True)
    table = v["pareto_table"]
    means = [row["mean"] for row in table]
    assert means == sorted(means), f"lower-is-loss table must be ascending (best=lowest), got {means}"
    assert table[0]["name"] == "Prizma"
    assert table[-1]["name"] == "GLA"


# ============================================ PURE VERDICT: pairwise BEATS/PARITY/WORSE ==
def test_verdict_labels_beats_parity_worse_higher_is_acc():
    """Prizma clearly > GLA (BEATS), Prizma ~ Mamba2 (PARITY), Prizma < TF (WORSE)."""
    from seq.landscape import landscape_verdict
    rng = np.random.default_rng(2)
    arm_accs = {
        "Prizma": list(0.90 + 0.004 * rng.standard_normal(10)),
        "TF":     list(0.99 + 0.004 * rng.standard_normal(10)),   # TF clearly above Prizma -> WORSE
        "GLA":    list(0.50 + 0.01 * rng.standard_normal(10)),    # far below Prizma -> BEATS
        "Mamba2": list(0.902 + 0.004 * rng.standard_normal(10)),  # ~ Prizma -> PARITY
    }
    v = landscape_verdict(arm_accs, cand_key="Prizma", margin=0.05)
    pairs = v["pairwise"]   # {baseline -> {verdict, ...}}
    assert pairs["GLA"]["verdict"] == "BEATS", pairs["GLA"]
    assert pairs["Mamba2"]["verdict"] == "PARITY", pairs["Mamba2"]
    assert pairs["TF"]["verdict"] == "WORSE", pairs["TF"]
    # the candidate is never compared against itself.
    assert "Prizma" not in pairs


def test_verdict_lower_is_loss_flips_beats_and_worse():
    """With lower_is_better=True the SIGN flips: a LOWER Prizma loss vs a baseline is BEATS."""
    from seq.landscape import landscape_verdict
    rng = np.random.default_rng(3)
    arm_accs = {
        "Prizma": list(1.00 + 0.01 * rng.standard_normal(8)),   # low loss = good
        "GLA":    list(1.40 + 0.01 * rng.standard_normal(8)),   # high loss -> Prizma BEATS it
        "TF":     list(0.60 + 0.01 * rng.standard_normal(8)),   # lower loss than Prizma -> Prizma WORSE
        "Mamba2": list(1.005 + 0.01 * rng.standard_normal(8)),  # ~ Prizma -> PARITY
    }
    v = landscape_verdict(arm_accs, cand_key="Prizma", margin=0.05, lower_is_better=True)
    pairs = v["pairwise"]
    assert pairs["GLA"]["verdict"] == "BEATS", pairs["GLA"]
    assert pairs["TF"]["verdict"] == "WORSE", pairs["TF"]
    assert pairs["Mamba2"]["verdict"] == "PARITY", pairs["Mamba2"]


def test_verdict_applies_holm_family_correction():
    """The pairwise p-values must be Holm-corrected as a FAMILY, and the verdict must read from the
    family-wise decision (a comparison Holm rejects can never read BEATS for the wrong reason)."""
    from seq.landscape import landscape_verdict
    from seq.stats import superiority_test, holm_correction
    rng = np.random.default_rng(4)
    arm_accs = {
        "Prizma": list(0.95 + 0.004 * rng.standard_normal(10)),
        "TF":     list(0.80 + 0.01 * rng.standard_normal(10)),
        "GLA":    list(0.60 + 0.01 * rng.standard_normal(10)),
        "Mamba2": list(0.40 + 0.01 * rng.standard_normal(10)),
    }
    v = landscape_verdict(arm_accs, cand_key="Prizma", margin=0.05)
    pairs = v["pairwise"]
    # every pair carries the Holm-adjusted p-value and rejection decision.
    for b in ("TF", "GLA", "Mamba2"):
        assert "holm_p_adj" in pairs[b] and "holm_reject" in pairs[b]
    # cross-check: the Holm-adjusted p-values match seq.stats.holm_correction over the raw pair p's
    # (in the module's own baseline ordering, surfaced as v["pair_order"]).
    order = v["pair_order"]
    raw = [superiority_test(arm_accs["Prizma"], arm_accs[b])["p_value"] for b in order]
    holm = holm_correction(raw)
    for b, hr in zip(order, holm):
        assert pairs[b]["holm_p_adj"] == pytest.approx(hr["p_adj"])
        assert pairs[b]["holm_reject"] == bool(hr["reject"])


def test_verdict_uses_powered_stats_not_normal_approx():
    """Per-arm ci95 must match seq.stats.summarize exactly (real Student-t, not a 1.96 z-CI)."""
    from seq.landscape import landscape_verdict
    from seq.stats import summarize
    rng = np.random.default_rng(5)
    arm_accs = {
        "Prizma": list(0.95 + 0.003 * rng.standard_normal(7)),
        "TF":     list(0.93 + 0.003 * rng.standard_normal(7)),
        "GLA":    list(0.88 + 0.003 * rng.standard_normal(7)),
        "Mamba2": list(0.90 + 0.003 * rng.standard_normal(7)),
    }
    v = landscape_verdict(arm_accs, cand_key="Prizma", margin=0.05)
    rows = {row["name"]: row for row in v["pareto_table"]}
    exp = summarize(arm_accs["TF"], 0.9)
    assert rows["TF"]["ci95"] == pytest.approx(exp["ci95"])
    assert rows["TF"]["mean"] == pytest.approx(exp["mean"])


def test_verdict_requires_candidate_present():
    from seq.landscape import landscape_verdict
    with pytest.raises(KeyError):
        landscape_verdict({"TF": [0.9, 0.9], "GLA": [0.8, 0.8]}, cand_key="Prizma", margin=0.05)


# ===================================================== SMOKE END-TO-END (CPU) =====
def test_smoke_end_to_end_writes_json_with_all_arms_and_resumes():
    """A CPU-fast --smoke run must complete, write a JSON ledger containing all 4 arms + per-task
    pairwise verdicts + the negative-control leg, and be RESUMABLE (a second run skips cached cells).
    Kept tiny (d/L/H small, 2 seeds, 1 task, short cap) so it runs in well under a minute."""
    from seq import landscape as ls
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "gpu_landscape.json")
        report = ls.run_landscape(smoke=True, results_path=out)

        assert os.path.exists(out), "smoke run must write the results ledger"
        on_disk = json.load(open(out))

        # report + ledger agree and carry the landscape report block.
        assert report["smoke"] is True
        rep = on_disk["landscape_report"]
        assert rep["smoke"] is True

        # exactly one task in the smoke config, and it has all 4 arms ranked + pairwise verdicts.
        tasks = rep["tasks"]
        assert len(tasks) == 1, f"smoke runs exactly one task, got {list(tasks)}"
        (task_name, task_block), = tasks.items()
        arm_names = {row["name"].split(".")[0] for row in task_block["pareto_table"]}
        # canonical kinds present (TF / Prizma / GLA / Mamba2), tolerating skipped-broken arms.
        present = task_block["arms_present"]
        for kind in ("tf", "prizma", "gla", "mamba2"):
            assert kind in present, f"{kind} must appear (present or skipped-with-reason): {present}"
        # 4 runnable arms expected on CPU at this tiny scale.
        assert len(task_block["pareto_table"]) == 4, task_block["pareto_table"]
        # Prizma is compared against each of the 3 baselines (BEATS/PARITY/WORSE/INCONCLUSIVE labels).
        # the pairwise dict is keyed by the FULL arm name (e.g. 'TF.d48L1H2'); match by kind prefix.
        pair_kinds = {k.split(".")[0]: pv for k, pv in task_block["pairwise"].items()}
        assert set(pair_kinds) == {"TF", "GLA", "Mamba2"}, pair_kinds
        for b in ("TF", "GLA", "Mamba2"):
            assert pair_kinds[b]["verdict"] in ("BEATS", "PARITY", "WORSE", "INCONCLUSIVE")

        # NEGATIVE-CONTROL integrity leg present (two byte-identical arms must not differ).
        nc = rep["negative_control"]
        assert "p_value" in nc and "pass" in nc

        # ---- RESUMABILITY: a second smoke run must SKIP cached cells (no recompute). ----
        # snapshot the cached arm-cell keys, then re-run and assert no new training cells appeared
        # and the per-seed accuracies are bit-identical (cache hit, not a fresh train).
        before_keys = set(on_disk.keys())
        report2 = ls.run_landscape(smoke=True, results_path=out)
        rep2 = json.load(open(out))["landscape_report"]
        after_keys = set(json.load(open(out)).keys())
        assert after_keys == before_keys, "resume must not introduce new top-level cells"
        accs1 = rep["tasks"][task_name]["pairwise"]
        accs2 = rep2["tasks"][task_name]["pairwise"]
        assert accs1.keys() == accs2.keys()
        # the per-arm accuracies are unchanged across the resumed run (cache hit -> identical).
        t1 = {r["name"]: r["accs"] for r in rep["tasks"][task_name]["pareto_table"]}
        t2 = {r["name"]: r["accs"] for r in rep2["tasks"][task_name]["pareto_table"]}
        assert t1 == t2, "resumed run must reuse cached accuracies (no retrain)"


# ============================================================ CLI ARG-GUARD =======
def test_cli_unknown_arg_does_not_trigger_full_run():
    from seq.landscape import _build_parser
    parser = _build_parser()
    with pytest.raises(SystemExit) as ei:
        parser.parse_args(["--full-landscape"])   # typo of --full: rejected, never launched
    assert ei.value.code != 0
    with pytest.raises(SystemExit) as ei2:
        parser.parse_args(["bogus"])               # stray positional: rejected
    assert ei2.value.code != 0


def test_cli_known_flags_parse_and_select_mode():
    from seq.landscape import _build_parser
    parser = _build_parser()
    assert parser.parse_args(["--smoke"]).smoke is True
    full = parser.parse_args(["--full"])
    assert full.smoke is False and full.full is True
    none = parser.parse_args([])
    assert none.smoke is False and none.full is False
    with pytest.raises(SystemExit):
        parser.parse_args(["--smoke", "--full"])


def test_cli_charlm_flag_parses_and_is_smoke_composable():
    """--charlm selects the char-LM leg; --smoke --charlm composes (tiny char-LM smoke). Unknown
    flags still exit non-zero (arg-guard), and the existing recall flags are unchanged."""
    from seq.landscape import _build_parser
    parser = _build_parser()
    assert parser.parse_args(["--charlm"]).charlm is True
    both = parser.parse_args(["--smoke", "--charlm"])
    assert both.smoke is True and both.charlm is True
    # the default recall path is unchanged: no --charlm => charlm False.
    assert parser.parse_args([]).charlm is False
    assert parser.parse_args(["--smoke"]).charlm is False
    # arg-guard still rejects an unknown flag even alongside --charlm.
    with pytest.raises(SystemExit) as ei:
        parser.parse_args(["--charlm", "--bogus"])
    assert ei.value.code != 0


# ===================================================== CHAR-LM: gate-superset knob distinction =====
def test_prizma_charlm_knobs_gate_superset_and_differs_from_recall():
    """PRIZMA_CHARLM_KNOBS is the char-LM gate SUPERSET (4 gates ON + quad2_lowrank feature map), and it
    DIFFERS from the recall PRIZMA_V2_KNOBS (which has the gates OFF for clean overwrite). This pins the
    distinction the contract requires: char-LM benefits from forget/output gates; recall does not."""
    import seq.landscape as ls
    ck = ls.PRIZMA_CHARLM_KNOBS
    assert isinstance(ck, dict)
    # the 4 gate knobs are ON for char-LM ...
    for k in ("out_gate", "state_norm", "decoupled_gate", "gated"):
        assert ck.get(k) is True, f"char-LM Prizma must set gate knob {k}=True, got {ck.get(k)!r}"
    # ... and the feature map is the v2 lean recall lever, shared with the campaign char-LM arm.
    assert ck.get("feat_map") == "quad2_lowrank"
    # the recall arm (PRIZMA_V2_KNOBS) must NOT carry any of those gate knobs (gates OFF for recall).
    for k in ("out_gate", "state_norm", "decoupled_gate", "gated"):
        assert k not in ls.PRIZMA_V2_KNOBS, f"recall Prizma must NOT set the char-LM gate knob {k}"
    # the two configs are genuinely DIFFERENT (the char-LM arm is not the recall arm).
    assert ck != ls.PRIZMA_V2_KNOBS, "char-LM knobs must differ from recall knobs"
    # cross-check against gpu_charlm2's V2 gate set (single source of truth for the char-LM gates).
    import gpu_charlm2
    for k, v in gpu_charlm2.V2_GATE_KW.items():
        assert ck.get(k) == v, f"char-LM gate {k} must match gpu_charlm2.V2_GATE_KW"


# ===================================================== CHAR-LM: lower-is-better verdict sanity =====
def test_charlm_verdict_lower_bpc_prizma_beats_baseline():
    """A clearly-lower-BPC Prizma vs a baseline must read BEATS under lower_is_better=True, reusing the
    SAME landscape_verdict the recall path uses (the char-LM leg adds NO new verdict layer)."""
    from seq.landscape import landscape_verdict
    rng = np.random.default_rng(7)
    arm_accs = {
        "Prizma": list(1.10 + 0.01 * rng.standard_normal(8)),   # clearly lower BPC = better
        "TF":     list(1.40 + 0.01 * rng.standard_normal(8)),   # higher BPC -> Prizma BEATS it
    }
    v = landscape_verdict(arm_accs, cand_key="Prizma", margin=0.05, lower_is_better=True)
    assert v["lower_is_better"] is True
    assert v["pareto_table"][0]["name"] == "Prizma"             # best-first = lowest BPC
    assert v["pairwise"]["TF"]["verdict"] == "BEATS", v["pairwise"]["TF"]


# ===================================================== CHAR-LM: SMOKE END-TO-END (CPU) =====
def test_charlm_smoke_end_to_end_writes_bpc_verdicts_and_resumes():
    """A CPU-fast --smoke --charlm run must complete, write results/gpu_landscape_charlm.json containing
    all 4 arms + per-pair BPC verdicts (lower_is_better) + the negative control, and be RESUMABLE.

    Kept tiny (small char corpus slice, tiny d/T/cap, 2 seeds, 1-LR grid) so it runs CPU-fast.
    tiny-shakespeare is NOT committed to this repo (.gitignore excludes seq/data/shakespeare.txt; it is
    fetched on demand by gpu_charlm*.py), so on a fresh clone / in CI the smoke takes the deterministic
    synthetic fallback corpus instead. Either way the smoke is network-free, and this test asserts the
    PLUMBING only -- which corpus was actually used is recorded in the results JSON."""
    from seq import landscape as ls
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "gpu_landscape_charlm.json")
        report = ls.run_landscape_charlm(smoke=True, results_path=out)

        assert os.path.exists(out), "char-LM smoke must write the results ledger"
        on_disk = json.load(open(out))

        # report + ledger agree and carry the char-LM landscape block, flagged smoke + lower-is-better.
        assert report["smoke"] is True
        rep = on_disk["landscape_charlm_report"]
        assert rep["smoke"] is True
        assert rep["lower_is_better"] is True
        assert rep["metric"] == "bits_per_char"

        # exactly one task (char-LM) with all 4 arms ranked + pairwise BPC verdicts.
        tasks = rep["tasks"]
        assert len(tasks) == 1, f"char-LM smoke runs exactly one LM task, got {list(tasks)}"
        (task_name, task_block), = tasks.items()
        present = task_block["arms_present"]
        for kind in ("tf", "prizma", "gla", "mamba2"):
            assert kind in present, f"{kind} must appear (present or skipped-with-reason): {present}"
        assert len(task_block["pareto_table"]) == 4, task_block["pareto_table"]
        # the verdict is lower-is-better (BPC) end-to-end.
        assert task_block["lower_is_better"] is True
        # Prizma vs each of the 3 baselines, keyed by full arm name -> match by kind prefix.
        pair_kinds = {k.split(".")[0]: pv for k, pv in task_block["pairwise"].items()}
        assert set(pair_kinds) == {"TF", "GLA", "Mamba2"}, pair_kinds
        for b in ("TF", "GLA", "Mamba2"):
            assert pair_kinds[b]["verdict"] in ("BEATS", "PARITY", "WORSE", "INCONCLUSIVE")
            assert pair_kinds[b]["lower_is_better"] is True

        # the Prizma char-LM arm carries the gate superset (gates ON) in its disclosed knobs.
        assert rep["prizma_knobs"]["out_gate"] is True
        assert rep["prizma_knobs"]["feat_map"] == "quad2_lowrank"

        # NEGATIVE-CONTROL integrity leg present (two byte-identical arms must not differ).
        nc = rep["negative_control"]
        assert "p_value" in nc and "pass" in nc

        # a loud non-scientific DISCLAIMER is recorded for the smoke result.
        assert rep.get("smoke_disclaimer"), "smoke must record a non-scientific disclaimer"

        # ---- RESUMABILITY: a second smoke run must SKIP cached cells (bit-identical BPCs). ----
        before_keys = set(on_disk.keys())
        report2 = ls.run_landscape_charlm(smoke=True, results_path=out)
        after_keys = set(json.load(open(out)).keys())
        assert after_keys == before_keys, "resume must not introduce new top-level cells"
        t1 = {r["name"]: r["accs"] for r in report["tasks"][task_name]["pareto_table"]}
        t2 = {r["name"]: r["accs"] for r in report2["tasks"][task_name]["pareto_table"]}
        assert t1 == t2, "resumed char-LM run must reuse cached BPCs (no retrain)"
