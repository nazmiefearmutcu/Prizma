"""FAST unit tests for the RECALL TOST-PARITY GATE pure verdict logic (Council-3; plan
"Task 1.Recall-gate"; council record committee/round0_v2_synthesis.md item 9a).

These tests exercise ONLY the deterministic, training-free verdict logic in seq/recall_gate.py:
  - recall_gate_verdict(arm_accs, ...) : per-leg verdict from synthetic per-seed accuracy arrays
  - combine_gate(legs)                 : top-level gate that ANDs all legs and emits the downgrade word

NO training happens here (it must run in milliseconds). The real parity verdict requires the A100
>=10-seed run; this file pins the LOGIC that turns those numbers into a pass/fail "dominant" claim.

The gate semantics under test:
  * parity      : Prizma is TOST-equivalent to the tuned TF within tost_margin (the council bar).
  * flip-test   : a leg only counts as a CLEAN gate when a bigger TF solved the hard rung
                  (flip_solved=True) -> a tiny-TF failure is attributable to capacity, not "attention
                  can't". If flip_solved is False/None on the hard rung, the leg is 'inconclusive'.
  * combine     : gate_pass requires ALL legs pass; downgrade_word is 'dominant' if pass else
                  'competitive'.
"""
from __future__ import annotations

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# Module imports + public surface
# --------------------------------------------------------------------------- #
def test_module_imports_and_public_surface():
    import seq.recall_gate as rg
    assert callable(rg.recall_gate_verdict)
    assert callable(rg.combine_gate)
    assert callable(rg.run_recall_gate), "run_recall_gate must exist (training runner; not run here)"


# --------------------------------------------------------------------------- #
# (1) PARITY case: cand ~ tf (both ~0.99), flip_solved=True -> leg_pass True, equivalent True
# --------------------------------------------------------------------------- #
def test_parity_case_passes():
    from seq.recall_gate import recall_gate_verdict
    rng = np.random.default_rng(0)
    tf = list(0.99 + 0.002 * rng.standard_normal(10))
    cand = list(0.99 + 0.002 * rng.standard_normal(10))
    v = recall_gate_verdict(
        {"TF": tf, "Prizma": cand},
        tf_key="TF", cand_key="Prizma",
        tost_margin=0.05, solve_thresh=0.9, flip_solved=True,
    )
    assert v["equivalent"] is True, f"expected TOST-equivalent, got {v}"
    assert v["parity"] is True
    assert v["leg_pass"] is True
    assert v["flip_solved"] is True
    # per-arm summaries present for both arms with the powered summarize() keys
    assert set(["TF", "Prizma"]).issubset(v["per_arm"].keys())
    assert "ci95" in v["per_arm"]["TF"] and "solve_rate" in v["per_arm"]["TF"]
    # the TOST delta + ci90 are surfaced for the audit trail
    assert "delta" in v and "ci90" in v


# --------------------------------------------------------------------------- #
# (2) FAIL case: cand much lower than tf (0.5 vs 0.99), flip_solved=True ->
#     leg_pass False, equivalent False; combine_gate => word 'competitive'
# --------------------------------------------------------------------------- #
def test_fail_case_not_equivalent():
    from seq.recall_gate import recall_gate_verdict, combine_gate
    rng = np.random.default_rng(1)
    tf = list(0.99 + 0.002 * rng.standard_normal(10))
    cand = list(0.50 + 0.01 * rng.standard_normal(10))
    v = recall_gate_verdict(
        {"TF": tf, "Prizma": cand},
        tf_key="TF", cand_key="Prizma",
        tost_margin=0.05, solve_thresh=0.9, flip_solved=True,
    )
    assert v["equivalent"] is False, f"0.5 vs 0.99 must NOT be equivalent, got {v}"
    assert v["parity"] is False
    assert v["leg_pass"] is False

    g = combine_gate({"MQAR-HARD": v})
    assert g["gate_pass"] is False
    assert g["downgrade_word"] == "competitive", f"a failed leg must downgrade, got {g}"


# --------------------------------------------------------------------------- #
# (3) INCONCLUSIVE case: cand ~ tf BUT flip_solved=False on the hard rung ->
#     leg verdict 'inconclusive' (NOT a clean pass)
# --------------------------------------------------------------------------- #
def test_inconclusive_when_flip_not_solved():
    from seq.recall_gate import recall_gate_verdict
    rng = np.random.default_rng(2)
    tf = list(0.99 + 0.002 * rng.standard_normal(10))
    cand = list(0.99 + 0.002 * rng.standard_normal(10))
    v = recall_gate_verdict(
        {"TF": tf, "Prizma": cand},
        tf_key="TF", cand_key="Prizma",
        tost_margin=0.05, solve_thresh=0.9, flip_solved=False,
    )
    # even though the arms are equivalent, a flip-test failure makes the leg inconclusive
    assert v["flip_solved"] is False
    assert v["leg_pass"] is False, "a non-flip-solved hard rung must NOT be a clean pass"
    assert "inconclusive" in v["reason"].lower(), f"reason must say inconclusive, got {v['reason']!r}"

    # flip_solved=None is treated the same as False (no bigger-TF evidence)
    v_none = recall_gate_verdict(
        {"TF": tf, "Prizma": cand},
        tf_key="TF", cand_key="Prizma",
        tost_margin=0.05, solve_thresh=0.9, flip_solved=None,
    )
    assert v_none["leg_pass"] is False
    assert "inconclusive" in v_none["reason"].lower()


# --------------------------------------------------------------------------- #
# (4) combine_gate: all legs pass -> gate_pass True, word 'dominant';
#     one leg fails -> False, word 'competitive'
# --------------------------------------------------------------------------- #
def test_combine_gate_all_pass_is_dominant():
    from seq.recall_gate import recall_gate_verdict, combine_gate
    rng = np.random.default_rng(3)

    def _pass_leg(seed):
        r = np.random.default_rng(seed)
        tf = list(0.99 + 0.002 * r.standard_normal(10))
        cand = list(0.99 + 0.002 * r.standard_normal(10))
        return recall_gate_verdict(
            {"TF": tf, "Prizma": cand},
            tf_key="TF", cand_key="Prizma",
            tost_margin=0.05, solve_thresh=0.9, flip_solved=True,
        )

    legs = {"MQAR-HARD": _pass_leg(10), "INDUCTION": _pass_leg(11), "SELECTIVE-COPY": _pass_leg(12)}
    for v in legs.values():
        assert v["leg_pass"] is True

    g = combine_gate(legs)
    assert g["gate_pass"] is True
    assert g["downgrade_word"] == "dominant", f"all-pass must say dominant, got {g}"
    assert set(legs.keys()).issubset(g["per_leg"].keys())


def test_combine_gate_one_fail_is_competitive():
    from seq.recall_gate import recall_gate_verdict, combine_gate

    def _leg(tf_mean, cand_mean, seed, flip=True):
        r = np.random.default_rng(seed)
        tf = list(tf_mean + 0.002 * r.standard_normal(10))
        cand = list(cand_mean + 0.01 * r.standard_normal(10))
        return recall_gate_verdict(
            {"TF": tf, "Prizma": cand},
            tf_key="TF", cand_key="Prizma",
            tost_margin=0.05, solve_thresh=0.9, flip_solved=flip,
        )

    legs = {
        "MQAR-HARD": _leg(0.99, 0.99, 20),       # pass
        "INDUCTION": _leg(0.99, 0.55, 21),       # FAIL (cand far below TF)
        "SELECTIVE-COPY": _leg(0.99, 0.99, 22),  # pass
    }
    g = combine_gate(legs)
    assert g["gate_pass"] is False
    assert g["downgrade_word"] == "competitive"


# --------------------------------------------------------------------------- #
# (5) An inconclusive leg also blocks the gate (not just an explicit fail)
# --------------------------------------------------------------------------- #
def test_combine_gate_inconclusive_blocks_gate():
    from seq.recall_gate import recall_gate_verdict, combine_gate
    r = np.random.default_rng(30)
    tf = list(0.99 + 0.002 * r.standard_normal(10))
    cand = list(0.99 + 0.002 * r.standard_normal(10))
    incon = recall_gate_verdict(
        {"TF": tf, "Prizma": cand},
        tf_key="TF", cand_key="Prizma",
        tost_margin=0.05, solve_thresh=0.9, flip_solved=None,
    )
    clean = recall_gate_verdict(
        {"TF": tf, "Prizma": cand},
        tf_key="TF", cand_key="Prizma",
        tost_margin=0.05, solve_thresh=0.9, flip_solved=True,
    )
    g = combine_gate({"MQAR-HARD": incon, "INDUCTION": clean})
    assert g["gate_pass"] is False
    assert g["downgrade_word"] == "competitive"


# --------------------------------------------------------------------------- #
# (6) The verdict uses the POWERED stats functions, not a normal-approx CI.
#     Sanity: per-arm ci95 must match seq.stats.summarize exactly.
# --------------------------------------------------------------------------- #
def test_verdict_uses_powered_stats():
    from seq.recall_gate import recall_gate_verdict
    from seq.stats import summarize, tost_equivalence
    r = np.random.default_rng(40)
    tf = list(0.99 + 0.003 * r.standard_normal(8))
    cand = list(0.97 + 0.003 * r.standard_normal(8))
    v = recall_gate_verdict(
        {"TF": tf, "Prizma": cand},
        tf_key="TF", cand_key="Prizma",
        tost_margin=0.05, solve_thresh=0.9, flip_solved=True,
    )
    exp_tf = summarize(tf, 0.9)
    assert v["per_arm"]["TF"]["ci95"] == pytest.approx(exp_tf["ci95"])
    assert v["per_arm"]["TF"]["mean"] == pytest.approx(exp_tf["mean"])
    exp_tost = tost_equivalence(cand, tf, 0.05)
    assert v["delta"] == pytest.approx(exp_tost["delta"])
    assert v["equivalent"] == exp_tost["equivalent"]


# --------------------------------------------------------------------------- #
# (7) CLI arg-guard: an UNKNOWN arg must NOT trigger the multi-hour full gate.
#     argparse rejects it (SystemExit, non-zero) before any run starts. --smoke
#     and --full parse cleanly; no-arg defaults to the full gate (smoke=False).
#     Training-free: we only exercise the parser, never run_recall_gate.
# --------------------------------------------------------------------------- #
def test_cli_unknown_arg_does_not_trigger_full_run():
    from seq.recall_gate import _build_parser
    parser = _build_parser()
    # an unknown/typo'd flag -> SystemExit with a NON-ZERO code (never silently runs the full gate)
    with pytest.raises(SystemExit) as ei:
        parser.parse_args(["--full-gate"])     # typo of --full: must be rejected, not launched
    assert ei.value.code != 0
    with pytest.raises(SystemExit) as ei2:
        parser.parse_args(["bogus"])           # stray positional: also rejected
    assert ei2.value.code != 0


def test_cli_known_flags_parse_and_select_mode():
    from seq.recall_gate import _build_parser
    parser = _build_parser()
    assert parser.parse_args(["--smoke"]).smoke is True
    full = parser.parse_args(["--full"])
    assert full.smoke is False and full.full is True
    # no-arg -> full gate (smoke False); main() runs the full gate in this case.
    none = parser.parse_args([])
    assert none.smoke is False and none.full is False
    # --smoke and --full are mutually exclusive -> rejected together.
    with pytest.raises(SystemExit):
        parser.parse_args(["--smoke", "--full"])


# --------------------------------------------------------------------------- #
# (8) Process-safe locking: verify atomic JSON write behavior under concurrent
#     access. Writes must never result in half-written/corrupted files.
# --------------------------------------------------------------------------- #
def test_process_safe_locking_via_atomic_replace(tmp_path):
    import json
    import random
    import concurrent.futures
    from seq.recall_gate import _save, _load

    file_path = str(tmp_path / "recall_gate_test.json")

    def writer(thread_id):
        for i in range(50):
            # Create a dictionary of random size to simulate realistic writes
            data = {
                "thread_id": thread_id,
                "iteration": i,
                "payload": "a" * random.randint(100, 2000),
                "nested": {"list": [random.random() for _ in range(10)]}
            }
            _save(file_path, data)

    def reader():
        for _ in range(100):
            try:
                data = _load(file_path)
                if data:  # If file has been written, it must be valid and contain our keys
                    assert "thread_id" in data
                    assert "payload" in data
            except json.JSONDecodeError as e:
                pytest.fail(f"File corruption detected: {e}")
            except Exception as e:
                if not isinstance(e, FileNotFoundError):
                    pytest.fail(f"Unexpected read error: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = []
        for i in range(4):
            futures.append(executor.submit(writer, i))
        futures.append(executor.submit(reader))

        # Wait and raise exceptions if any occurred
        concurrent.futures.wait(futures)
        for f in futures:
            f.result()


# --------------------------------------------------------------------------- #
# (9) LR sweep seed pinning: verify that the LR sweep is pinned to seeds[0],
#     and Stage-2 runs are correctly pinned to their respective seeds.
# --------------------------------------------------------------------------- #
def test_lr_sweep_seed_pinning(tmp_path):
    from unittest.mock import patch, MagicMock
    from seq.recall_gate import _train_arm

    file_path = str(tmp_path / "recall_gate_test.json")
    res = {}

    # Dummy factories and inputs
    dummy_model_fac = MagicMock()
    dummy_task = MagicMock()
    dummy_task.vocab = 10
    dummy_task.seq_len = 5
    dummy_task_fac = MagicMock(return_value=dummy_task)

    seeds = (42, 100, 200)
    lr_grid = (1e-3, 2e-3)

    # Mock return values for build_and_train
    mock_run_result = MagicMock()
    mock_run_result.best_acc = 0.95
    mock_run_result.steps_to_plateau = 120
    mock_run_result.params = 500

    # Mock return for sweep_lr
    mock_sweep_result = {
        "best_lr": 1e-3,
        "best_acc": 0.95,
        "grid": [
            {"lr": 1e-3, "best_acc": 0.95, "steps_to_plateau": 120},
            {"lr": 2e-3, "best_acc": 0.90, "steps_to_plateau": 150}
        ]
    }

    with patch("seq.lrsweep.sweep_lr", return_value=mock_sweep_result) as mock_sweep, \
         patch("seq.common.build_and_train", return_value=mock_run_result) as mock_build:

        _train_arm(
            res=res,
            results_path=file_path,
            leg="MQAR-HARD",
            arm="Prizma",
            model_fac=dummy_model_fac,
            task_fac=dummy_task_fac,
            device="cpu",
            cap=10,
            seeds=seeds,
            lr_grid=lr_grid,
            recipe={},
            eval_every=5,
            batch_size=4
        )

        # 1. Assert sweep_lr was called once, and seed is pinned to 0
        mock_sweep.assert_called_once()
        sweep_kwargs = mock_sweep.call_args[1]
        assert sweep_kwargs["seed"] == 0, f"Expected sweep seed to be pinned to 0, got {sweep_kwargs['seed']}"
        assert sweep_kwargs["grid"] == lr_grid

        # 2. Assert build_and_train was called for each seed in Stage-2 (42, 100, 200)
        assert mock_build.call_count == len(seeds)
        called_seeds = [call[1]["seed"] for call in mock_build.call_args_list]
        assert called_seeds == list(seeds), f"Expected Stage-2 seeds {list(seeds)}, got {called_seeds}"

        # 3. Assert cell in res is populated correctly
        cell = res["cells"]["MQAR-HARD.Prizma"]
        assert cell["best_lr"] == 1e-3
        assert cell["best_accs"] == [0.95, 0.95, 0.95]
        assert cell["params"] == 500

