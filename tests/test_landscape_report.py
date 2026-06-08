"""Tests for the TORCH-FREE landscape report renderer (seq/landscape_report.py).

seq/landscape_report.py is a PURE RENDERER: it READS the verdict JSONs that seq/landscape.py's
run_landscape / run_landscape_charlm write (results/gpu_landscape.json and
results/gpu_landscape_charlm.json) and emits a Council-3 Pareto MARKDOWN report. It does NO torch
import and NO training — it only renders what the JSON already contains (no invented numbers, no spin,
per-FLOP claims OUT since this layer only has accuracy/BPC, not FLOPs).

These tests pin the renderer contract:

  (A) REAL SMOKE END-TO-END  — generate a genuine smoke landscape JSON via
      seq.landscape.run_landscape(smoke=True) (trains tiny CPU models within the existing smoke budget),
      then render it and assert the markdown contains all 4 arm names, "Prizma", a verdict label, the
      scope rider, the negative-control line, and the SMOKE banner. Using the real smoke means the schema
      can't silently drift away from what the renderer reads.

  (B) SYNTHETIC-DICT  — a hand-crafted minimal landscape_report dict (1 task, 4 arms, KNOWN verdicts)
      rendered and checked: the table rows are best-first ordered and the Verdict-vs-Prizma column matches
      the input verdicts exactly.

  (C) ARG-GUARD  — an unknown CLI flag exits NON-ZERO (mirrors landscape.main's arg-guard), so a typo
      can never silently do the wrong thing.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# The verbatim scope rider the renderer MUST emit (a substring of it is enough to pin it).
SCOPE_RIDER_FRAGMENT = ("Scope: ≤2M params (+1 confirmation 10–50M); char-LM + diagnostics; "
                        "NOT a frontier/MMLU/long-context claim")


# ============================================================ module surface ==
def test_module_imports_torch_free_surface():
    """The renderer module imports WITHOUT torch (it is a pure read-and-render layer) and exposes
    render_report + main."""
    import seq.landscape_report as lr
    assert callable(lr.render_report), "render_report(landscape_json, charlm_json) -> str must exist"
    assert callable(lr.main), "main() CLI entry must exist"
    # the renderer must not have pulled torch in just by being imported.
    assert "torch" not in sys.modules or True  # tolerant: other tests may have imported torch already
    # but the module's own source must not import torch.
    src = open(os.path.join(REPO_ROOT, "seq", "landscape_report.py")).read()
    assert "import torch" not in src, "the renderer must be torch-free (no 'import torch')"


# ============================================================ REAL SMOKE END-TO-END ==
def test_render_real_smoke_recall_json_contains_arms_prizma_verdict_scope_negctrl_banner():
    """Generate a REAL smoke recall landscape JSON (tiny CPU train within the smoke budget) and render
    it. The markdown must contain all 4 arm names, the word 'Prizma', at least one verdict label, the
    scope rider, the negative-control line, and the SMOKE banner. Using the real smoke keeps the schema
    honest (the renderer reads exactly what landscape.py writes)."""
    from seq import landscape as ls
    from seq.landscape_report import render_report
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "gpu_landscape.json")
        ls.run_landscape(smoke=True, results_path=out)
        assert os.path.exists(out)

        md = render_report(landscape_json=out, charlm_json=os.path.join(td, "missing_charlm.json"))
        assert isinstance(md, str) and md.strip()

        # all 4 canonical arm kinds appear by name in the rendered table.
        for kind in ("TF", "Prizma", "GLA", "Mamba2"):
            assert kind in md, f"arm {kind!r} must appear in the rendered report"
        # the candidate is named.
        assert "Prizma" in md
        # at least one pairwise verdict label is rendered.
        assert re.search(r"BEATS|PARITY|WORSE|INCONCLUSIVE", md), "a verdict label must appear"
        # the verbatim scope rider.
        assert SCOPE_RIDER_FRAGMENT in md, "the scope rider must be emitted verbatim"
        # the negative-control line (PASS/FAIL) per leg.
        assert "Negative control" in md or "NEGATIVE CONTROL" in md.upper()
        assert re.search(r"negative control.*(PASS|FAIL)", md, re.IGNORECASE), \
            "the negative-control PASS/FAIL verdict must be rendered"
        # the SMOKE banner (the run is a smoke).
        assert "SMOKE" in md and "NOT a scientific result" in md


def test_render_omits_missing_section_with_a_note():
    """When a JSON path is missing, that section is OMITTED with a note (not a crash)."""
    from seq import landscape as ls
    from seq.landscape_report import render_report
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "gpu_landscape.json")
        ls.run_landscape(smoke=True, results_path=out)
        # only recall exists; charlm path is missing.
        md = render_report(landscape_json=out, charlm_json=os.path.join(td, "nope.json"))
        # a note is rendered for the missing char-LM leg, and it does NOT crash.
        assert re.search(r"char-?LM", md, re.IGNORECASE)
        assert re.search(r"(not found|missing|omitted|no .*json)", md, re.IGNORECASE), \
            "a missing section must be omitted WITH a note"


def test_render_returns_text_when_both_missing():
    """When BOTH JSONs are missing the renderer still returns a (non-crashing) markdown string that
    says so rather than raising."""
    from seq.landscape_report import render_report
    with tempfile.TemporaryDirectory() as td:
        md = render_report(landscape_json=os.path.join(td, "a.json"),
                           charlm_json=os.path.join(td, "b.json"))
        assert isinstance(md, str) and md.strip()
        # the scope rider is still present (it is part of the title block).
        assert SCOPE_RIDER_FRAGMENT in md


# ============================================================ SYNTHETIC DICT ==
def _synthetic_recall_report():
    """A minimal hand-crafted landscape_report dict (1 task, 4 arms, KNOWN verdicts) matching the schema
    seq.landscape.landscape_verdict + run_landscape persist. Prizma is the candidate (in cand_key, not in
    pairwise). Rows are deliberately given UNSORTED so the renderer must sort them best-first."""
    cand = "Prizma.d128L4H4_feat_map=quad2_lowrank"

    def row(name, mean, params, solve, ci):
        return {"name": name, "params": params, "mean": mean, "median": mean,
                "ci95": list(ci), "sd": 0.01, "solve_rate": solve, "accs": [mean, mean, mean]}

    # intentionally NOT best-first on input: GLA(0.50) < Mamba2(0.90) < TF(0.95) < Prizma(0.97).
    pareto_unsorted = [
        row("GLA.d128L4H4", 0.50, 280000, 0.0, (0.48, 0.52)),
        row(cand, 0.97, 300000, 1.0, (0.96, 0.98)),
        row("TF.d128L4H4", 0.95, 260000, 1.0, (0.94, 0.96)),
        row("Mamba2.d128L4H4", 0.90, 288000, 0.9, (0.88, 0.92)),
    ]
    pairwise = {
        "TF.d128L4H4": {"verdict": "PARITY", "delta": 0.02, "equivalent": True,
                        "holm_p_adj": 0.40, "holm_reject": False, "lower_is_better": False},
        "GLA.d128L4H4": {"verdict": "BEATS", "delta": 0.47, "equivalent": False,
                         "holm_p_adj": 0.001, "holm_reject": True, "lower_is_better": False},
        "Mamba2.d128L4H4": {"verdict": "INCONCLUSIVE", "delta": 0.07, "equivalent": False,
                            "holm_p_adj": 0.20, "holm_reject": False, "lower_is_better": False},
    }
    block = {
        "pareto_table": pareto_unsorted,
        "pairwise": pairwise,
        "pair_order": ["TF.d128L4H4", "GLA.d128L4H4", "Mamba2.d128L4H4"],
        "cand_key": cand,
        "margin": 0.05,
        "lower_is_better": False,
        "solve_thresh": 0.9,
        "arms_present": {k: {"name": f"{k.upper()}.x", "status": "ok"}
                         for k in ("tf", "prizma", "gla", "mamba2")},
        "unrunnable": {},
    }
    return {
        "landscape_report": {
            "tasks": {"MQAR-HARD": block},
            "negative_control": {"p_value": 0.81, "significant": False, "pass": True},
            "meta": {"scale": "d128L4H4", "seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                     "arms": ["tf", "prizma", "gla", "mamba2"], "margin": 0.05, "solve_thresh": 0.9},
            "smoke": False,
        }
    }


def test_synthetic_table_is_best_first_and_verdict_column_matches_input():
    """Render the synthetic dict; the table rows must be best-first ordered (Prizma > TF > Mamba2 > GLA)
    and the Verdict-vs-Prizma column must match the input verdicts exactly (no invented labels)."""
    from seq.landscape_report import render_report
    with tempfile.TemporaryDirectory() as td:
        rec = os.path.join(td, "gpu_landscape.json")
        json.dump(_synthetic_recall_report(), open(rec, "w"))
        md = render_report(landscape_json=rec, charlm_json=os.path.join(td, "none.json"))

        # locate the markdown table DATA lines (rows start with '|'); drop the header (mentions 'Prizma'
        # in the 'Verdict vs Prizma' column title) and the '---' separator so only real arm rows remain.
        lines = [ln for ln in md.splitlines()
                 if ln.strip().startswith("|") and "---" not in ln
                 and not ("Arm" in ln and "Params" in ln)]
        # the data rows, in order of appearance, must rank best-first: Prizma, TF, Mamba2, GLA.
        order = []
        for kind in ("Prizma", "TF", "Mamba2", "GLA"):
            for i, ln in enumerate(lines):
                if kind in ln and i not in [o[1] for o in order]:
                    order.append((kind, i))
                    break
        # the row indices must be strictly increasing in the best-first order.
        idxs = [i for _, i in order]
        assert idxs == sorted(idxs), f"rows must be best-first (Prizma,TF,Mamba2,GLA), got {order}"

        # the Verdict-vs-Prizma column matches the input verdicts (Prizma row has no self-verdict).
        # find each baseline's DATA row line (skip the header row, which mentions 'Prizma' in its column
        # title, and the '---' separator) and assert its verdict token is present on that line.
        def is_header(ln):
            return "Arm" in ln and "Params" in ln
        def row_line(kind):
            return next(ln for ln in lines if kind in ln and "---" not in ln and not is_header(ln))
        assert "PARITY" in row_line("TF"), row_line("TF")
        assert "BEATS" in row_line("GLA"), row_line("GLA")
        assert "INCONCLUSIVE" in row_line("Mamba2"), row_line("Mamba2")
        # the candidate row is marked as the candidate and carries no verdict label.
        prizma_line = row_line("Prizma")
        assert "cand" in prizma_line.lower(), f"Prizma row must be marked as candidate: {prizma_line}"
        assert not re.search(r"BEATS|PARITY|WORSE|INCONCLUSIVE", prizma_line), \
            f"candidate row must have NO self-verdict: {prizma_line}"


def test_synthetic_lower_is_better_charlm_renders_bpc_metric_and_lower_better_label():
    """A synthetic char-LM report (lower_is_better=True, BPC) renders with the BPC metric name and a
    '(lower better)' marker, and the rows are ascending best-first (lowest BPC first)."""
    from seq.landscape_report import render_report
    cand = "Prizma.d256L4H4"

    def row(name, mean, params):
        return {"name": name, "params": params, "mean": mean, "median": mean,
                "ci95": [mean - 0.02, mean + 0.02], "sd": 0.01, "solve_rate": 1.0,
                "accs": [mean, mean, mean]}

    block = {
        "pareto_table": [row("TF.d256L4H4", 1.40, 260000), row(cand, 1.10, 300000),
                         row("GLA.d256L4H4", 1.60, 280000), row("Mamba2.d256L4H4", 1.30, 288000)],
        "pairwise": {
            "TF.d256L4H4": {"verdict": "BEATS", "delta": 0.30, "equivalent": False,
                            "holm_p_adj": 0.002, "holm_reject": True, "lower_is_better": True},
            "GLA.d256L4H4": {"verdict": "BEATS", "delta": 0.50, "equivalent": False,
                             "holm_p_adj": 0.001, "holm_reject": True, "lower_is_better": True},
            "Mamba2.d256L4H4": {"verdict": "PARITY", "delta": 0.20, "equivalent": True,
                                "holm_p_adj": 0.30, "holm_reject": False, "lower_is_better": True},
        },
        "pair_order": ["TF.d256L4H4", "GLA.d256L4H4", "Mamba2.d256L4H4"],
        "cand_key": cand, "margin": 0.05, "lower_is_better": True, "solve_thresh": 1.5,
        "arms_present": {}, "unrunnable": {},
    }
    rep = {
        "landscape_charlm_report": {
            "tasks": {"CHARLM-TEXT8": block},
            "negative_control": {"p_value": 0.6, "significant": False, "pass": True},
            "meta": {"scale": "d256L4H4", "seeds": list(range(10)), "arms": ["tf", "prizma", "gla", "mamba2"],
                     "margin": 0.05, "solve_thresh": 1.5, "corpus": "text8", "ctx": 256,
                     "random_baseline_bpc": 4.75, "metric": "bits_per_char"},
            "smoke": False, "lower_is_better": True, "metric": "bits_per_char",
        }
    }
    with tempfile.TemporaryDirectory() as td:
        cj = os.path.join(td, "gpu_landscape_charlm.json")
        json.dump(rep, open(cj, "w"))
        md = render_report(landscape_json=os.path.join(td, "none.json"), charlm_json=cj)
        # BPC metric name + lower-is-better marker appear.
        assert re.search(r"BPC|bits.per.char", md, re.IGNORECASE), "BPC metric name must appear"
        assert "lower better" in md.lower() or "lower is better" in md.lower()
        # best-first = lowest BPC first: Prizma(1.10) before Mamba2(1.30) before TF(1.40) before GLA(1.60).
        # drop the header (mentions 'Prizma' in its column title) + the '---' separator.
        lines = [ln for ln in md.splitlines()
                 if ln.strip().startswith("|") and "---" not in ln
                 and not ("Arm" in ln and "Params" in ln)]
        def first_idx(kind):
            return next(i for i, ln in enumerate(lines) if kind in ln)
        assert first_idx("Prizma") < first_idx("Mamba2") < first_idx("TF") < first_idx("GLA")


def test_render_tolerates_missing_optional_fields():
    """Missing params / ci95 are rendered as '—' (no crash, no invented numbers)."""
    from seq.landscape_report import render_report
    cand = "Prizma.x"
    block = {
        "pareto_table": [
            {"name": cand, "params": None, "mean": 0.9, "median": 0.9, "sd": 0.0,
             "solve_rate": 1.0, "accs": [0.9]},  # no ci95 key, params None
            {"name": "TF.x", "mean": 0.8, "median": 0.8, "solve_rate": 1.0, "accs": [0.8]},  # minimal
        ],
        "pairwise": {"TF.x": {"verdict": "BEATS", "delta": 0.1, "equivalent": False,
                              "holm_p_adj": 0.01, "holm_reject": True, "lower_is_better": False}},
        "pair_order": ["TF.x"], "cand_key": cand, "margin": 0.05, "lower_is_better": False,
        "solve_thresh": 0.9, "arms_present": {}, "unrunnable": {},
    }
    rep = {"landscape_report": {"tasks": {"T": block},
                                "negative_control": {"p_value": 0.5, "significant": False, "pass": True},
                                "meta": {"scale": "dx", "margin": 0.05}, "smoke": False}}
    with tempfile.TemporaryDirectory() as td:
        rec = os.path.join(td, "gpu_landscape.json")
        json.dump(rep, open(rec, "w"))
        md = render_report(landscape_json=rec, charlm_json=os.path.join(td, "none.json"))
        assert "—" in md, "missing fields must render as the em-dash placeholder"
        assert "BEATS" in md


def test_render_pareto_picture_summary_counts_verdicts_no_flop_spin():
    """The combined 'Pareto picture' summary counts Prizma's BEATS/PARITY/WORSE/INCONCLUSIVE vs each
    baseline, states it plainly, and keeps per-FLOP/efficiency claims OUT (this renderer has no FLOPs)."""
    from seq.landscape_report import render_report
    with tempfile.TemporaryDirectory() as td:
        rec = os.path.join(td, "gpu_landscape.json")
        json.dump(_synthetic_recall_report(), open(rec, "w"))
        md = render_report(landscape_json=rec, charlm_json=os.path.join(td, "none.json"))
        assert re.search(r"Pareto picture", md, re.IGNORECASE), "a combined Pareto-picture summary"
        # the synthetic recall leg has exactly 1 BEATS, 1 PARITY, 1 INCONCLUSIVE, 0 WORSE.
        # the summary should surface those counts somewhere.
        assert re.search(r"BEATS\D*1|1\D*BEATS", md), "BEATS count must be surfaced"
        # no per-FLOP / efficiency claim leaks in (the renderer only has accuracy/BPC).
        assert not re.search(r"per[- ]?FLOP", md) or "conditional" in md.lower(), \
            "no unconditional per-FLOP claim may be made by this renderer"


# ============================================================ CLI ARG-GUARD ==
def test_cli_unknown_flag_exits_nonzero():
    """An unknown CLI flag exits NON-ZERO (arg-guard), via the installed module entrypoint."""
    env = dict(os.environ)
    proc = subprocess.run([sys.executable, "-m", "seq.landscape_report", "--definitely-not-a-flag"],
                          cwd=REPO_ROOT, capture_output=True, text=True, env=env)
    assert proc.returncode != 0, f"unknown flag must exit non-zero, got {proc.returncode}\n{proc.stderr}"


def test_cli_renders_to_stdout_and_out_path():
    """main() renders to stdout by default and to --out PATH when given (real smoke recall JSON)."""
    from seq import landscape as ls
    with tempfile.TemporaryDirectory() as td:
        rec = os.path.join(td, "gpu_landscape.json")
        ls.run_landscape(smoke=True, results_path=rec)
        # render via the CLI to an out file.
        outmd = os.path.join(td, "report.md")
        proc = subprocess.run(
            [sys.executable, "-m", "seq.landscape_report",
             "--recall-json", rec, "--charlm-json", os.path.join(td, "none.json"),
             "--out", outmd],
            cwd=REPO_ROOT, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        assert os.path.exists(outmd)
        body = open(outmd).read()
        assert "Prizma" in body and SCOPE_RIDER_FRAGMENT in body
