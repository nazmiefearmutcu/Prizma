"""PR-2026-09-03-01 runner — the pre-registration's NAMED entry point (§2 item 6 / §3 pre-flight).

Thin alias: the implementation lives in seq/surprise_claim.py (one script, --smoke / --powered;
separate ledgers under results/surprise_ablation_PR-2026-09-03-01/ with the smoke->powered refusal;
the directory is the owner's commission for this implementation — disclosed in that module's
docstring). Run either of:

  python gpu_surprise_ablation.py --smoke      # pre-reg §3 pre-flight: CPU plumbing, minutes
  python seq/surprise_claim.py    --powered    # the A100 claim campaign (refuses without CUDA)
"""
from seq.surprise_claim import main

if __name__ == "__main__":
    main()
