"""Unit tests for tools/ship_prizma.py guards (pre-GPU review M-13 / M-14 / M-15).

Pure helper tests only: subprocess is monkeypatched and the remote-tree verifier
takes an injectable fetch callable — NO network calls, NO token file access.
"""
import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parent.parent / "tools" / "ship_prizma.py"
_spec = importlib.util.spec_from_file_location("ship_prizma_under_test", _TOOLS)
ship = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ship)

A, B, C = "a" * 40, "b" * 40, "c" * 40  # fake blob shas


def _fake_git(stdout: bytes):
    def fake_check_output(cmd, **kwargs):
        return stdout
    return fake_check_output


# ---------------------------------------------------------------- M-13: dirty tree

def test_dirty_tree_refused(monkeypatch):
    monkeypatch.setattr(ship.subprocess, "check_output",
                        _fake_git(b" M docs/addendum.md\n?? new-file.py\n"))
    with pytest.raises(SystemExit) as exc:
        ship._assert_clean_tree(allow_dirty=False)
    assert exc.value.code not in (0, None)


def test_dirty_tree_allowed_with_allow_dirty(monkeypatch):
    monkeypatch.setattr(ship.subprocess, "check_output",
                        _fake_git(b" M docs/addendum.md\n"))
    ship._assert_clean_tree(allow_dirty=True)  # must not raise


def test_clean_tree_passes(monkeypatch):
    monkeypatch.setattr(ship.subprocess, "check_output", _fake_git(b""))
    ship._assert_clean_tree(allow_dirty=False)  # must not raise


# ---------------------------------------------------------------- M-14: cwd guard

def test_cwd_not_toplevel_refused(monkeypatch):
    monkeypatch.setattr(ship.subprocess, "check_output",
                        _fake_git(b"C:/some/other/repo\n"))
    with pytest.raises(SystemExit) as exc:
        ship._assert_repo_root()
    assert exc.value.code not in (0, None)


def test_cwd_toplevel_passes_with_forward_slashes(monkeypatch):
    # git reports the toplevel with forward slashes on Windows; normalization must cope.
    cwd = os.getcwd()
    monkeypatch.setattr(ship.subprocess, "check_output",
                        _fake_git(cwd.replace("\\", "/").encode() + b"\n"))
    ship._assert_repo_root()  # must not raise


def test_not_a_git_repo_refused(monkeypatch):
    def boom(cmd, **kwargs):
        raise subprocess.CalledProcessError(128, cmd)
    monkeypatch.setattr(ship.subprocess, "check_output", boom)
    with pytest.raises(SystemExit) as exc:
        ship._assert_repo_root()
    assert exc.value.code not in (0, None)


# ---------------------------------------------------------------- M-15: tree verify

def test_compare_tree_sets_equal():
    assert ship._compare_tree_sets({A, B}, {B, A}) == []
    assert ship._compare_tree_sets([], []) == []


def test_compare_tree_sets_difference():
    assert ship._compare_tree_sets({A, B}, {B, C}) == sorted([A, C])
    assert ship._compare_tree_sets([A], []) == [A]
    assert ship._compare_tree_sets([], [C]) == [C]


def test_verify_remote_tree_equal(capsys):
    seen = {}

    def fetch(method, url):
        seen["call"] = (method, url)
        return {"truncated": False,
                "tree": [{"type": "tree", "sha": "t1"},
                         {"type": "blob", "sha": B},
                         {"type": "blob", "sha": A}]}

    assert ship._verify_remote_tree("tok", "https://x/git", "tre_sha", {A, B},
                                    fetch=fetch) == []
    assert seen["call"] == ("GET", "https://x/git/trees/tre_sha?recursive=1")
    assert "TREE EQUAL" in capsys.readouterr().out


def test_verify_remote_tree_mismatch_exits_nonzero(capsys):
    def fetch(method, url):
        return {"truncated": False, "tree": [{"type": "blob", "sha": C}]}

    with pytest.raises(SystemExit) as exc:
        ship._verify_remote_tree("tok", "https://x/git", "tre_sha", {A}, fetch=fetch)
    assert exc.value.code not in (0, None)
    assert "TREE MISMATCH" in capsys.readouterr().out


# ---------------------------------------------------------------- CLI parsing

def test_parser_accepts_allow_dirty():
    ns = ship._build_parser().parse_args(["msg", "--allow-dirty"])
    assert ns.allow_dirty is True

    ns2 = ship._build_parser().parse_args(["msg"])
    assert ns2.allow_dirty is False
    assert ns2.force is False

    ns3 = ship._build_parser().parse_args(
        ["msg", "--force", "--allow-dirty", "--repo", "owner/name"])
    assert ns3.force is True and ns3.repo == "owner/name"
