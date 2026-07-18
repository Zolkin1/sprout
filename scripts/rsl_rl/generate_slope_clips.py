# Copyright 2026 Zachary Olkin. All rights reserved.

"""Turn a FLAT clip library into a SLOPE clip library by rigidly rotating each clip.

A slope clip is a flat clip whose whole body is pitched onto a uniform tilted plane
(``motion_matching.slope_rotate.rotate_clip_to_slope``): the feet that walked on ``z=0``
now climb the plane, the tilt lives in the clip's roll/pitch, joints (the gait) are
unchanged, and the contact schedule carries over (rotation-invariant).  Each requested
slope angle becomes one terrain tile (``terrain_id``); every flat clip is rotated onto
every angle, tagged ``terrain_type="slope"`` with the slope anchor mask
(FREE x,y; z/roll/pitch/yaw clip/terrain-fixed).  A ``terrain_manifest`` with one
``SlopeBlock`` entry per angle is embedded for :class:`ClipTerrainImporter`.

Example::

    PYTHONPATH=source/robot_rl:source/motion_gen python scripts/rsl_rl/generate_slope_clips.py \\
        --in-lib experiments/flat_v1.h5 --out experiments/slope_v1.h5 \\
        --slope-degs 5,10,15,20
"""

from __future__ import annotations

import argparse
import math

import torch

from motion_gen.clips.clip_io import ClipLibraryWriter, load_clip_library
from motion_gen.motion_matching import features as F
from motion_gen.motion_matching.clip_output import ClipOutputBuilder
from motion_gen.motion_matching.slope_rotate import (
    rotate_clip_to_slope,
    slope_pitch_for_climb,
    upright_slope_clip,
)
from motion_gen.world_lift import _ANCHOR_MASK_SLOPE

# Uphill slopes only make sense up to a modest grade; the reference is a rotated flat
# walk (an approximation), so we cap the generated grade.
_MAX_SLOPE_DEG = 10.0


def _pelvis_xy_idx(pos_names: list[str]) -> tuple[int, int]:
    if "pelvis_link:pos_x" not in pos_names:
        raise ValueError("library has no 'pelvis_link:pos_x' channel; cannot pick a rotation pivot.")
    return pos_names.index("pelvis_link:pos_x"), pos_names.index("pelvis_link:pos_y")


def _pelvis_span(y_des: torch.Tensor, px: int, py: int) -> tuple[float, float]:
    """Max (x-span, y-span) the pelvis covers across all clips (for tile sizing)."""
    x_span = float((y_des[:, :, px].amax(1) - y_des[:, :, px].amin(1)).amax())
    y_span = float((y_des[:, :, py].amax(1) - y_des[:, :, py].amin(1)).amax())
    return x_span, y_span


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-lib", required=True, help="Input FLAT clip library .h5.")
    p.add_argument("--out", required=True, help="Output SLOPE clip library .h5.")
    p.add_argument("--slope-degs", default="5,10",
                   help=f"Comma-separated uphill slope angles in degrees (one terrain tile each); "
                        f"capped at {_MAX_SLOPE_DEG:g} deg (rotated-flat reference is only valid on modest grades).")
    p.add_argument("--width-margin", type=float, default=1.0,
                   help="Extra walkable width (m) beyond the clips' lateral span for the slope tile.")
    p.add_argument("--span-margin", type=float, default=1.0,
                   help="Extra along-slope length (m) beyond the clips' forward span for the tile.")
    p.add_argument("--rigid", action="store_true",
                   help="Rigidly tilt the whole body onto the ramp (torso leans back). Default is "
                        "UPRIGHT: keep the pelvis world-vertical + flex the hips so only the legs "
                        "follow the incline (a feasible-looking climb).")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    print("[slope] WARNING: slope generation is DEFERRED / INCOMPLETE (see slope_rotate.py TODO).\n"
          "        The upright-pelvis hip compensation is only correct for clips walking STRAIGHT\n"
          "        up/down the slope; turning/strafing source clips are left leaning sideways\n"
          "        (cross-slope tilt is ROLL, not pitch). Do not train on these clips yet.")

    degs = [float(d) for d in args.slope_degs.split(",") if d.strip()]
    if not degs:
        raise ValueError("no slope angles given (--slope-degs).")
    over = [d for d in degs if d > _MAX_SLOPE_DEG + 1e-6]
    if over:
        raise ValueError(f"slope angles {over} exceed the {_MAX_SLOPE_DEG:g} deg cap "
                         f"(rotated-flat reference is only valid on modest grades).")

    lib = load_clip_library(args.in_lib, args.device)
    if lib.terrain_id.numel() and not bool((lib.terrain_id.to(torch.long) == -1).all()):
        raise ValueError("--in-lib must be a FLAT library (all terrain_id == -1).")
    pos_names, vel_names = lib.meta["pos_output_names"], lib.meta["vel_output_names"]
    px, py = _pelvis_xy_idx(pos_names)
    n_flat = lib.num_clips
    x_span, y_span = _pelvis_span(lib.y_des, px, py)
    size_x = x_span + args.span_margin
    size_y = y_span + args.width_margin

    meta = {
        "dt": lib.meta["dt"], "clip_length_s": lib.meta.get("clip_length_s", lib.num_steps * lib.meta["dt"]),
        "num_steps": lib.num_steps,
        "pos_output_names": list(pos_names), "vel_output_names": list(vel_names),
        "contact_frames": list(lib.meta["contact_frames"]), "ref_frame": lib.meta["ref_frame"],
        "skill_list": list(lib.meta.get("skill_list", [])),
    }
    total = n_flat * len(degs)
    upright = not args.rigid
    mode = "upright pelvis + hip flex" if upright else "rigid body tilt"
    print(f"[slope] {n_flat} flat clips x {len(degs)} angles -> {total} slope clips "
          f"(tile {size_x:.2f} x {size_y:.2f} m; {mode})")

    # Upright mode rebuilds via FK (feet from the compensated legs); set up the builder
    # + the pelvis/joint channel indices it needs.
    if upright:
        builder = ClipOutputBuilder(pos_names, vel_names,
                                    contact_frames=list(lib.meta["contact_frames"]), device=args.device)
        pelvis_pos_i = [pos_names.index(f"pelvis_link:pos_{a}") for a in "xyz"]
        pelvis_ori_i = [pos_names.index(f"pelvis_link:ori_{a}") for a in ("w", "x", "y", "z")]
        joint_i = [pos_names.index(f"joint:{n}") for n in F.G1_JOINT_NAMES]
        hip_idx = (F.G1_JOINT_NAMES.index("left_hip_pitch_joint"),
                   F.G1_JOINT_NAMES.index("right_hip_pitch_joint"))

    manifest = []
    with ClipLibraryWriter(args.out, total, meta, compression="lzf") as w:
        written = 0
        for tid, deg in enumerate(degs):
            rad = math.radians(deg)
            pitch = slope_pitch_for_climb(rad)
            rise = size_x * math.tan(rad)          # total z climb across the tile span
            manifest.append({
                "terrain_id": tid, "type": "slope",
                "size_x": float(size_x), "size_y": float(size_y),
                "dims": {"angle_deg": float(deg), "rise": float(rise), "w": float(size_y)},
                "bands": [{"block_type": "SlopeBlock", "aabb": [0.0, float(size_x)],
                           "rise": float(rise), "w": float(size_y)}],
            })
            # Put every flat clip onto this slope; keep the flat contact (invariant).
            ys, dys = [], []
            for c in range(n_flat):
                pivot = (float(lib.y_des[c, 0, px]), float(lib.y_des[c, 0, py]))
                if upright:
                    yr, dyr = upright_slope_clip(
                        builder, lib.y_des[c][:, pelvis_pos_i], lib.y_des[c][:, pelvis_ori_i],
                        lib.y_des[c][:, joint_i], hip_idx, pitch, pivot, float(lib.meta["dt"]))
                else:
                    yr, dyr = rotate_clip_to_slope(lib.y_des[c], lib.dy_des[c], pos_names, vel_names,
                                                   pitch, pivot)
                ys.append(yr); dys.append(dyr)
            batch = {
                "y_des": torch.stack(ys), "dy_des": torch.stack(dys),
                "contact": lib.contact.clone(), "skill_id": lib.skill_id.clone(),
                "stride_idx": lib.stride_idx.clone(),
                "target_vel": lib.target_vel.clone() if lib.target_vel.numel() else torch.zeros(n_flat, lib.num_steps, 3),
                "terrain_id": torch.full((n_flat,), tid, dtype=torch.int32),
                "terrain_type": "slope",
                "anchor_mask": torch.full((n_flat,), _ANCHOR_MASK_SLOPE, dtype=torch.uint8),
            }
            w.write_batch(written, batch)
            written += n_flat
            print(f"[slope]  tile {tid}: {deg:.1f} deg  rise={rise:.2f} m  ({n_flat} clips)")
        w.set_attr("terrain_manifest", manifest)
        w.set_attr("generator", "slope_rotate")

    out = load_clip_library(args.out)
    print(f"[slope] wrote {args.out}: {out.num_clips} clips, "
          f"terrain_type_code unique={sorted(set(out.terrain_type_code.tolist()))}, "
          f"anchor_mask unique={sorted(set(out.anchor_mask.tolist()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
