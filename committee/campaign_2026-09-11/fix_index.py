"""One-off registry repair (campaign 2026-09-11).

Restores the PR-08/PR-10/PR-11/PR-12 rows that commit 05ad731 accidentally deleted from
docs/preregistry/INDEX.md (1 insertion / 9 deletions in that commit), removes the duplicate
PR-2026-09-03-03 IN-WRITE row (POLICY: one row transitions), adds NOT-ESTABLISHED to the
documented status vocabulary, and re-appends the LANE-EXPLORATORY footnote.

Idempotent: refuses to double-insert (checks for existing rows first). Read-only re: history;
writes only docs/preregistry/INDEX.md.
"""
from __future__ import annotations

import os
import subprocess

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INDEX = os.path.join(REPO, "docs", "preregistry", "INDEX.md")
OLD_REV = "05ad731^"


def main() -> int:
    old = subprocess.run(
        ["git", "show", f"{OLD_REV}:docs/preregistry/INDEX.md"],
        cwd=REPO, capture_output=True, check=True,
    ).stdout.decode("utf-8")
    cur = open(INDEX, encoding="utf-8").read()

    def find_row(text: str, pid: str) -> str:
        for line in text.splitlines():
            if line.startswith(f"| PR-2026-09-03-{pid} |"):
                return line
        raise SystemExit(f"row PR-2026-09-03-{pid} not found in {OLD_REV}")

    missing = ["12", "11", "10", "08"]
    to_add = [find_row(old, p) for p in missing]

    lines = cur.splitlines(keepends=True)
    out: list[str] = []
    inserted = False
    removed_dup = False
    vocab_done = False
    for line in lines:
        # drop the duplicate PR-03 IN-WRITE row (a CLAIMED -03 row also exists).
        if (not removed_dup and line.startswith("| PR-2026-09-03-03 |")
                and "| IN-WRITE |" in line):
            removed_dup = True
            continue
        # status vocabulary: add NOT-ESTABLISHED (PR-20's terminal status).
        if (not vocab_done and line.startswith("Statuses:")):
            line = line.replace(
                "`IN-WRITE` / `REGISTERED` / `ANALYZED` / `CLAIMED` / `NEGATIVE` /",
                "`IN-WRITE` / `REGISTERED` / `ANALYZED` / `CLAIMED` / `NEGATIVE` / `NOT-ESTABLISHED` /",
            )
            vocab_done = True
        out.append(line)
        if not inserted and line.startswith("| PR-2026-09-03-13 |"):
            out.extend(r + "\n" for r in to_add)
            inserted = True

    if not inserted:
        raise SystemExit("PR-13 row anchor not found; refusing to guess")
    new = "".join(out)

    for line in to_add:
        if line in new.split("\n")[:0]:
            pass
    # idempotence guard: if the added rows are already present twice, bail.
    for r in to_add:
        if new.count(r) != 1:
            raise SystemExit(f"row multiplicity != 1 after merge: {r[:60]}")

    footnote = ("\n*Exploratory work (LANE-EXPLORATORY) does not get rows here by default; its results live under\n"
                "`results/exploratory/` and never become claims. It gets a row only when it motivates a registered\n"
                "pre-registration — and then the row points at the LANE-CLAIM document, not at the exploration.*\n")
    if "LANE-EXPLORATORY" not in new:
        if not new.endswith("\n"):
            new += "\n"
        new += footnote

    open(INDEX, "w", encoding="utf-8", newline="\n").write(new)
    n_rows = sum(1 for ln in new.splitlines() if ln.startswith("| PR-2026-09-03-"))
    print(f"OK: restored {missing}, dup-removed={removed_dup}, vocab={vocab_done}, "
          f"footnote={'LANE-EXPLORATORY' in new}, registry rows={n_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
