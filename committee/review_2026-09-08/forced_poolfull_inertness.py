"""Inertness proof for the FORCED-RECRUIT pool-full contingency (doc addendum #3).

Re-runs FORCED-RECRUIT seeds 0 and 1 under the AMENDED run_routed (Policy A eviction fallback)
in a FRESH process and compares every science-carrying field against the original powered.json
cells. Only wall_s (timing) and the new forced_placement provenance key may differ. If anything
else differs, the amendment is NOT inert and the contingency must be re-examined before any
verdict is recorded. Read-only over results/.
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from seq import prizma_lm_claim as plc          # noqa: E402
from seq import blockdrift_claim as claim       # noqa: E402

led_path = os.path.join(ROOT, "results", "prizma_lm_PR-2026-09-03-08", "powered.json")
with open(led_path, encoding="utf-8") as f:
    res = json.load(f)
lr = res["lr_selection.fusion"]["lr"]

slices = plc.pin_slices()
A_all, B_all = claim.fetch_corpora()
A_train = A_all[slices["A_train"][1]:slices["A_train"][2]]
A_eval = A_all[slices["A_eval"][1]:slices["A_eval"][2]]
C_ret = A_all[slices["C_retention"][1]:slices["C_retention"][2]]
C_train = A_all[slices["C_train"][1]:slices["C_train"][2]]
B_train = B_all[: int(len(B_all) * slices["B_train"][2])]
B_eval = B_all[int(len(B_all) * slices["B_eval"][1]):]
chars = sorted(set(A_train) | set(A_eval) | set(C_ret) | set(C_train) | set(B_all))
vocab = {c: i for i, c in enumerate(chars)}
V = len(vocab)
Ax, Ay = claim.make_segments(A_train, vocab)
Bx, By = claim.make_segments(B_train, vocab)
Cx, Cy = claim.make_segments(C_train, vocab)
Aex, Aey = claim.make_segments(A_eval, vocab)
Bex, Bey = claim.make_segments(B_eval, vocab)
Crx, Cry = claim.make_segments(C_ret, vocab)
data = (Ax, Ay, Bx, By, Cx, Cy, Aex, Aey, Bex, Bey, Crx, Cry)

import torch            # noqa: E402
torch.set_num_threads(8)

ok = True
for seed in (0, 1):
    new = plc.run_routed(V, seed, data, lr, forced_c=True)
    old = res[f"claim.FORCED-RECRUIT.s{seed}"]
    # run_routed's OWN output keys are the science fields; arm/cellkey/cfgsig/complete are
    # cell-loop wrapper keys absent from the raw return (asserted separately below).
    mismatched = [k for k in sorted(set(new))
                  if k not in ("wall_s", "forced_placement") and old.get(k) != new.get(k)]
    # the wrapper fields the loop would re-attach must also match (cfgsig recomputed from the
    # exact same payload run() builds — this proves the amendment did not shift the fingerprint)
    sig = plc._fp({"leg": "claim", "arm": "FORCED-RECRUIT",
                   "family": plc.ARM_FAMILY["FORCED-RECRUIT"], "seed": seed, "lr": lr,
                   "smoke": False, "vocab": V, "smoke_segs": None, "seg": plc.SEG,
                   "batch_segs": plc.BATCH_SEGS, "slices": res["meta"]["pinned_slices"],
                   "stream_lengths": res["meta"]["stream_lengths"],
                   "tissue": res["meta"]["tissue"], "bars": res["meta"]["bars"]})
    wrapper_ok = (old.get("arm") == "FORCED-RECRUIT"
                  and old.get("cellkey") == f"claim.FORCED-RECRUIT.s{seed}"
                  and old.get("complete") is True and old.get("cfgsig") == sig)
    ok = ok and not mismatched and wrapper_ok
    print(f"seed {seed}: forced_slot old={old.get('forced_slot')} new={new.get('forced_slot')}"
          f" placement_mode={new['forced_placement']['mode']}"
          f" wall old={old.get('wall_s')}s new={new['wall_s']}s"
          f" mismatched_keys={mismatched if mismatched else 'NONE'}"
          f" wrapper(cfgsig==recomputed)={wrapper_ok}", flush=True)
print("INERTNESS:", "PROVEN — amendment byte-identical on the free-slot path" if ok
      else "FAILED — amendment changed a science-carrying field; DO NOT adjudicate")
sys.exit(0 if ok else 1)
