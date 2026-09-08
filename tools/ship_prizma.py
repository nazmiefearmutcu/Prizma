#!/usr/bin/env python3
"""Ship the current Prizma HEAD tree to GitHub via the REST API (git push is broken on this
machine — see the flowmap-solana tools/ship-api.py header for the full story; same recipe).

Prizma-specific pre-check: the local token (nazmiefearmutcu0) currently has PULL-ONLY
access to nazmiefearmutcu/Prizma (the repo lives under the older account). Without write
access the tool prints the three remediation options and exits 3:
  (a) transfer nazmiefearmutcu/Prizma to nazmiefearmutcu0 (repo Settings -> Transfer), or
  (b) add nazmiefearmutcu0 as a collaborator with write on nazmiefearmutcu/Prizma, or
  (c) create nazmiefearmutcu0/Prizma and ship there (--repo nazmiefearmutcu0/Prizma).

Usage:
    python tools/ship_prizma.py "commit message" [--repo OWNER/NAME] [--force] [--allow-dirty]

Guards (fail closed, exit non-zero):
  - must run from the repo toplevel: from a subdir `git ls-tree -r HEAD` lists only that
    subtree and --force would replace the whole remote tree with it (review M-14);
  - with --force, refuses a dirty working tree unless --allow-dirty is passed —
    uncommitted edits are NOT shipped, the script ships `git ls-tree -r HEAD` (M-13).
After a --force PATCH, the remote tree is re-fetched (GET /trees/<sha>?recursive=1) and
its blob-sha set compared against local HEAD: prints TREE EQUAL or TREE MISMATCH plus a
short diff summary, and exits non-zero on mismatch (M-15).

Without --force: dry run (builds everything, touches no ref). With --force: PATCHes the
branch. Token: GH_TOKEN env or C:/Users/Kullanıcı/gh.token (never printed).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

DEFAULT_REPO = "nazmiefearmutcu/Prizma"
TOKEN_FALLBACK_PATH = r"C:/Users/Kullanıcı/gh.token"


def _read_token() -> str:
    token = os.environ.get("GH_TOKEN", "").strip()
    if token:
        return token
    with open(TOKEN_FALLBACK_PATH, encoding="utf-8") as fh:
        return fh.read().strip()


def api(token: str, method: str, url: str, payload=None, tries: int = 5):
    for i in range(tries):
        req = urllib.request.Request(
            url, method=method,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Authorization": f"token {token}",
                     "Accept": "application/vnd.github+json",
                     "User-Agent": "prizma-ship"},
        )
        try:
            with urllib.request.urlopen(req) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (409, 502, 503) and i < tries - 1:
                time.sleep(3 * (i + 1))
                continue
            raise


def _norm_path(path: str) -> str:
    """Normalize for comparison (Windows: case, separators, 8.3 short names)."""
    try:
        return os.path.normcase(os.path.realpath(path))
    except OSError:
        return os.path.normcase(os.path.abspath(path))


def _assert_repo_root() -> None:
    """M-14: refuse to run from a subdirectory.

    From a subdir `git ls-tree -r HEAD` lists only that subtree and --force would
    replace the whole remote tree with it.
    """
    try:
        toplevel = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"]).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        raise SystemExit("error: not inside a git repository "
                         "(git rev-parse --show-toplevel failed) — run from the repo root")
    if _norm_path(os.getcwd()) != _norm_path(toplevel):
        raise SystemExit(f"error: cwd {os.getcwd()} is not the repo toplevel ({toplevel}) — "
                         "from a subdir the tree listing would cover only that subtree and "
                         "--force would replace the whole remote tree with it. "
                         "Run from the repo root.")


def _assert_clean_tree(allow_dirty: bool) -> None:
    """M-13: refuse to ship with uncommitted working-tree edits.

    The script ships `git ls-tree -r HEAD`, so dirty edits silently do not ship.
    Enforced on the --force path only (a dry run ships nothing).
    """
    status = subprocess.check_output(["git", "status", "--porcelain"]).decode()
    dirty = [ln for ln in status.splitlines() if ln.strip()]
    if dirty and not allow_dirty:
        preview = "\n".join("  " + ln for ln in dirty[:5])
        more = f"\n  ... and {len(dirty) - 5} more" if len(dirty) > 5 else ""
        raise SystemExit(f"error: working tree is dirty ({len(dirty)} path(s)); uncommitted "
                         "edits do NOT ship (the script ships `git ls-tree -r HEAD`). "
                         "Commit or stash first, or pass --allow-dirty to ship anyway.\n"
                         + preview + more)


def _compare_tree_sets(local_blobs, remote_blobs) -> list:
    """Pure compare: sorted symmetric difference of the two blob-sha sets ([] == equal)."""
    return sorted(set(local_blobs) ^ set(remote_blobs))


def _verify_remote_tree(token: str, api_url: str, tree_sha: str, local_blobs,
                        fetch=None) -> list:
    """M-15: after the ref PATCH, re-fetch the remote tree and compare blob shas.

    Prints TREE EQUAL, or TREE MISMATCH plus a short diff summary, and raises
    SystemExit (non-zero) on mismatch. `fetch` is injectable for tests; defaults
    to the module's api().
    """
    if fetch is None:
        def fetch(method: str, url: str):
            return api(token, method, url)
    remote = fetch("GET", f"{api_url}/trees/{tree_sha}?recursive=1")
    if remote.get("truncated"):
        raise SystemExit("error: remote tree listing was truncated — tree-equality "
                         "verification cannot complete; investigate before trusting the ship")
    remote_blobs = [e["sha"] for e in remote.get("tree", []) if e.get("type") == "blob"]
    diff = _compare_tree_sets(local_blobs, remote_blobs)
    if diff:
        only_local = sorted(set(local_blobs) - set(remote_blobs))
        only_remote = sorted(set(remote_blobs) - set(local_blobs))
        print("TREE MISMATCH")
        print(f"  blob sha(s) in local HEAD but not in remote tree: {len(only_local)}")
        for sha in only_local[:5]:
            print(f"    - {sha}")
        print(f"  blob sha(s) in remote tree but not in local HEAD: {len(only_remote)}")
        for sha in only_remote[:5]:
            print(f"    - {sha}")
        raise SystemExit("error: remote tree does NOT match the local HEAD blob set "
                         "(the ref WAS patched — fix and re-ship; do not assume success)")
    print(f"TREE EQUAL ({len(set(local_blobs))} blobs)")
    return diff


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ship Prizma HEAD to GitHub via REST API.")
    parser.add_argument("message")
    parser.add_argument("--repo", default=os.environ.get("GH_REPO", DEFAULT_REPO))
    parser.add_argument("--force", action="store_true",
                        help="REQUIRED to actually PATCH the branch ref; without it: dry run")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="with --force: permit shipping despite a dirty working tree "
                             "(uncommitted edits are NOT shipped)")
    return parser


def main() -> int:
    _assert_repo_root()  # M-14: before anything else
    args = _build_parser().parse_args()
    if args.force:
        _assert_clean_tree(args.allow_dirty)  # M-13: only the real ship needs a clean tree

    token = _read_token()
    who = api(token, "GET", "https://api.github.com/user")["login"]
    repo = api(token, "GET", f"https://api.github.com/repos/{args.repo}")
    perms = repo.get("permissions", {})
    print(f"token account: {who} | target: {args.repo} | push: {perms.get('push')}")
    if not perms.get("push"):
        print("\nNO WRITE ACCESS — nothing was shipped. Pick one:")
        print(f"  (a) transfer {args.repo} to {who} (Settings -> Transfer), or")
        print(f"  (b) add {who} as a collaborator with write on {args.repo}, or")
        print(f"  (c) create {who}/Prizma and re-run with --repo {who}/Prizma")
        return 3
    if repo.get("default_branch") != "main" or repo.get("size", 0) == 0:
        raise SystemExit("error: target repo has no main branch — seed it first "
                         "(single Contents PUT of README.md), then re-run")

    api_url = f"https://api.github.com/repos/{args.repo}/git"
    parent = api(token, "GET", f"{api_url}/ref/heads/{repo['default_branch']}")["object"]["sha"]
    print("remote base:", parent)

    files = subprocess.check_output(["git", "ls-tree", "-r", "HEAD"]).decode().strip().splitlines()
    tree_entries, local_blob_shas, n = [], set(), 0
    for line in files:
        meta, path = line.split("\t", 1)
        mode, typ, sha = meta.split(" ")
        if typ != "blob":
            continue
        local_blob_shas.add(sha)
        raw = subprocess.check_output(["git", "cat-file", "blob", sha])  # committed bytes (no CRLF drift)
        r = api(token, "POST", f"{api_url}/blobs",
                {"content": base64.b64encode(raw).decode(), "encoding": "base64"})
        tree_entries.append({"path": path, "mode": mode, "type": "blob", "sha": r["sha"]})
        n += 1
        if n % 20 == 0:
            print(f"blobs {n}")
    print("total blobs:", n)

    tree = api(token, "POST", f"{api_url}/trees", {"tree": tree_entries})
    commit = api(token, "POST", f"{api_url}/commits",
                 {"message": args.message, "tree": tree["sha"], "parents": [parent]})
    if args.force:
        api(token, "PATCH", f"{api_url}/refs/heads/{repo['default_branch']}",
            {"sha": commit["sha"], "force": True})
        print("SHIPPED commit:", commit["sha"])
        print("NOTE: this rebuilds remote history (tree contents match local HEAD; commit "
              "SHAs differ from local by design).")
        _verify_remote_tree(token, api_url, tree["sha"], local_blob_shas)  # M-15
    else:
        print("DRY RUN — ref NOT updated. Built commit:", commit["sha"])
        print(f"re-run with --force to PATCH {args.repo} -> {commit['sha']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
