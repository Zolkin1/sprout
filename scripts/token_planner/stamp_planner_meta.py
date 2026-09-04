# Copyright 2026 Zachary Olkin. All rights reserved.

"""Write a recording's contract into a planner checkpoint that predates it being saved.

``load_planner_context`` needs the recording contract -- joint order, the predicted frame
names, the camera contract, the robot fingerprint, the re-plan cadence -- and takes it from
the checkpoint's ``meta``. Checkpoints written before the trainer saved that key have none,
and the Isaac command then dies on an empty joint list.

Everything needed lives in the recording's HDF5 ATTRS, so this reads no array data and takes
about a second. Prefer re-training over stamping where that is cheap; this exists because a
run in flight should not have to be restarted to be watched.

    python scripts/token_planner/stamp_planner_meta.py \\
        --checkpoint logs/token_planner/<run>/planner.pt \\
        --recording  logs/rsl_rl/g1/<teacher>/recorded_data/<recorded>.h5
"""

from __future__ import annotations

import argparse
import shutil

import torch

from robot_rl.token_planner.online import _meta_from_recording


def main() -> None:
    """Stamp the contract and rewrite the checkpoint in place."""
    p = argparse.ArgumentParser(description="Stamp a recording contract into a planner .pt")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--recording", required=True,
                   help="The recording the planner trained on -- its FIRST --data entry, "
                        "which is where the flow trainer takes its meta from too.")
    p.add_argument("--out", default=None, help="Write here instead of in place.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite a contract the checkpoint already carries.")
    a = p.parse_args()

    meta = _meta_from_recording(a.recording)
    if not meta:
        raise SystemExit(f"[stamp] {a.recording} is not readable as a recording.")
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    if ckpt.get("meta") and not a.force:
        raise SystemExit(
            "[stamp] this checkpoint already carries a contract; pass --force to replace it. "
            "Replacing one silently is how a checkpoint ends up describing a recording it "
            "was not trained on.")
    ckpt["meta"] = meta
    ckpt.setdefault("target_frame_names", list(meta.get("target_frame_names") or []))
    out = a.out or a.checkpoint
    if out == a.checkpoint:
        shutil.copyfile(a.checkpoint, a.checkpoint + ".bak")
    torch.save(ckpt, out)
    print(f"[stamp] {out}: {len(meta.get('state_joint_names') or [])} joints, "
          f"{len(meta.get('target_frame_names') or [])} frames, "
          f"fm_snap_every={meta.get('fm_snap_every')}, "
          f"robot_asset={'yes' if meta.get('robot_asset') else 'no'}")


if __name__ == "__main__":
    main()
