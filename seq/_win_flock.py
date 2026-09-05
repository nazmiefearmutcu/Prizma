"""Windows fallback for fcntl.flock (used by seq/gpu_harness.py and seq/recall_gate.py).

Both call sites follow the same pattern: open a `<path>.lock` file, flock it around a short
read-modify-write or read-only critical section, release implicitly by closing the handle.

Semantics provided on Windows via msvcrt.locking:
- LOCK_EX: LK_LOCK — blocking exclusive lock (retries every 1s up to 10 attempts, then
  raises OSError); released when the locking handle is closed, which matches the call
  sites' `with open(...)` release-on-close discipline.
- LOCK_SH: degrades to the same exclusive lock — Windows has no shared byte-range lock.
  This is a safe over-lock (serializes readers that flock would have let run concurrently),
  never an under-lock. Contention in this repo's usage is a single training process, so the
  throughput difference is nil.
Unix path: the real fcntl is imported and this module is never loaded.
"""
import msvcrt as _msvcrt

LOCK_EX = "EX"
LOCK_SH = "SH"


def flock(fileno, mode):
    _msvcrt.locking(fileno, _msvcrt.LK_LOCK, 1)
