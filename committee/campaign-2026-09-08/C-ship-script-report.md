# C-ship-script report — M-13 / M-14 / M-15 guards landed in tools/ship_prizma.py

**Date:** 2026-09-08 · **Zone touched:** `tools/ship_prizma.py` (edited), `tests/test_ship_prizma.py` (new), this report. Nothing else.
**Source findings:** `committee/review_2026-09-08/PRE_GPU_REVIEW.md` M-13 (dirty tree), M-14 (cwd), M-15 (post-ship verify).

## What changed, per finding

**M-13 — dirty-tree guard.** New `_assert_clean_tree(allow_dirty)`: runs `git status --porcelain`; non-empty output → `SystemExit` (exit 1) listing the dirty count + first 5 paths, telling the operator to commit/stash or pass the new `--allow-dirty` flag. Wired in `main()` only on the `--force` path, before `_read_token()` (so a refused run touches no token, no network). **Design decision, disclosed:** the guard does NOT gate the dry run — a dry run ships nothing, and the campaign's own acceptance check ("dry-run from repo root must still exit 3 with the pull-only diagnosis") is only satisfiable this way while the tree is dirty (it was: 14 dirty paths from parallel sessions at verify time, plus this task's own uncommitted files). A dry-run→`--force` progression still cannot skip the guard.

**M-14 — cwd guard.** New `_assert_repo_root()`: `git rev-parse --show-toplevel` compared against `os.getcwd()` via `_norm_path()` (`normcase(realpath(...))`, with abspath fallback) — needed because git reports the toplevel with forward slashes (`C:/Users/Kullanıcı/Prizma`) while `os.getcwd()` uses backslashes; realpath also folds 8.3 short paths (`KULLAN~1`). First statement of `main()`, before argparse. Not a git repo at all (rev-parse fails) → clear error, exit 1.

**M-15 — post-ship verification.** After the `--force` ref PATCH: `_verify_remote_tree(token, api_url, tree_sha, local_blob_shas)` does `GET /trees/<new_tree_sha>?recursive=1` through the script's existing `api()` (retries on 409/502/503 inherited), extracts blob shas, and compares via the pure `_compare_tree_sets(local, remote) -> sorted symmetric difference` ([] == equal). Prints `TREE EQUAL (N blobs)` or `TREE MISMATCH` + short diff summary (counts + up to 5 shas per side), and raises `SystemExit` (exit 1) on mismatch with an honest note that the ref WAS already patched. A `truncated: true` recursive listing is treated as a verification failure (exit 1, distinct message) — fail-closed; practically impossible at this repo size but cheap insurance. Local blob shas are collected during the existing `ls-tree` upload loop (set, so duplicated-blob-at-multiple-paths is fine; submodules/trees excluded, matching what gets uploaded). Verify runs only on the real ship path, per the review's "post-ship" wording — the dry run's behavior is byte-identical to before.

**Testability:** all new logic lives in small helpers; the only network-touching function (`_verify_remote_tree`) takes an injectable `fetch(method, url)` callable and is thin over the pure compare. Tests load the script by path with importlib (tools/ has no `__init__.py`).

## Tests (tests/test_ship_prizma.py — 11 tests)

Covers the four required areas plus extras; all subprocess calls monkeypatched, verify tested with an injected fetch. NO network, NO token file access.
(a) dirty tree refused / allowed with allow_dirty / clean tree passes; (b) cwd != toplevel refused, forward-slash toplevel passes (normalization), not-a-git-repo refused; (c) `_compare_tree_sets` [] on equal, differing shas otherwise; (d) parser accepts `--allow-dirty` (and default False, plus combos with --force/--repo). Extras: `_verify_remote_tree` TREE EQUAL path (pins the exact `?recursive=1` URL) and mismatch path (SystemExit non-zero + "TREE MISMATCH" printed).

## Commands + exit codes (all run from repo root unless noted)

| Command | Result |
|---|---|
| `./.venv/Scripts/python.exe -m pytest tests/test_ship_prizma.py -q` | `11 passed in 0.35s`, **EXIT=0** |
| `./.venv/Scripts/python.exe tools/ship_prizma.py --help` | usage shows `[--allow-dirty]` + help text, **EXIT=0** |
| `./.venv/Scripts/python.exe tools/ship_prizma.py "guard verification dry run"` | pull-only diagnosis (3 options), **EXIT=3** — unchanged, as required |
| `./.venv/Scripts/python.exe tools/ship_prizma.py "would-be ship" --force` | refused by M-13 guard (14 dirty paths listed), **EXIT=1**, before any token read/network |
| (from `tools/` subdir) `../.venv/Scripts/python.exe ship_prizma.py "subdir run"` | refused by M-14 guard with toplevel named, **EXIT=1** |

The `--force` and subdir runs are safe demonstrations: both exit inside the new guards, before `_read_token()` and before any HTTP call. The dry run performs only the two pre-existing read-only GETs (`/user`, `/repos/...`) exactly as before. The real ship was NOT run; the token file was never opened by me or printed.

## Deliberately not done

- **No git writes** — orchestrator integrates (per hard rule). So at report time `tools/ship_prizma.py`, the test file, and this report are themselves uncommitted; the M-13 guard is exactly what will make a premature `--force` refuse until the orchestrator commits. Working as intended.
- Dry-run does not run the M-15 verification (review scopes it post-PATCH; keeps dry-run output identical to before).
- No interactive warn+confirm variant of M-13 (review offered "refuse (or warn+confirm)"; refusal-only is the simpler, safer branch and matches the task text).
- Mismatch exit code is 1 (via `SystemExit(str)`); only the pull-only diagnosis keeps its existing code 3.
- Pre-existing dirty files (14 paths: docs/preregistry, seq/…) belong to other sessions/lanes — not mine, untouched.
- The other review items (M-1..M-12, M-16, H-1/H-2) are outside this task's file zone; not touched.
