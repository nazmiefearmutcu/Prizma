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
    python tools/ship_prizma.py "commit message" [--repo OWNER/NAME] [--force]

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Ship Prizma HEAD to GitHub via REST API.")
    parser.add_argument("message")
    parser.add_argument("--repo", default=os.environ.get("GH_REPO", DEFAULT_REPO))
    parser.add_argument("--force", action="store_true",
                        help="REQUIRED to actually PATCH the branch ref; without it: dry run")
    args = parser.parse_args()

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
    tree_entries, n = [], 0
    for line in files:
        meta, path = line.split("\t", 1)
        mode, typ, sha = meta.split(" ")
        if typ != "blob":
            continue
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
    else:
        print("DRY RUN — ref NOT updated. Built commit:", commit["sha"])
        print(f"re-run with --force to PATCH {args.repo} -> {commit['sha']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
