INVALID-DOSE ARCHIVE (2026-09-09 ~21:3x, maintainer)

The first PR-2026-09-03-13 launch passed --trunk-lr-c 0.0075 to the CLI — 10x the
registered PR-10/PR-11 dose (7.5e-4 = 3e-3 x 0.25). The runner accepted the raw float
without validating the registered dose (fixed same night: it now refuses any value other
than 7.5e-4). The ledger + console here are kept per docs/RETENTION.md (nothing deleted):
they are a valid EXECUTION of an UNREGISTERED dose and must never be cited as PR-13.
Side observation only (unregistered, n=1 run): at 0.0075 every arm's B-eval degraded
~+2.0 BPC post-C — a 10x C-dose destroys B retention in ALL arms, FORCED included.

Valid PR-2026-09-03-13 artifacts live in results/prizma_lm_PR-2026-09-03-13/ (re-run at
the registered dose after the guard landed).
