# Copyright 2026 Zachary Olkin. All rights reserved.

"""Shared plumbing for the terrain-consistency reward: reading rollouts, placing tiles.

Kept out of ``inspect_terrain_reward.py`` so the viser viewer can reuse it without importing
a script full of argparse.  Imports no IsaacLab beyond the window LAYOUT
(``robot_rl.flow_matching.window``, which needs only ``isaaclab.utils.math``) -- so this runs
on a training box without launching a simulator, which is the whole point of grading a
recorded plan offline.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import h5py
import numpy as np
import torch

from motion_gen.terrain.box_sdf import BoxUnionTerrain, pad_box_sets
from motion_gen.terrain.overlay_boxes import (
    collider_boxes_from_manifest_tile,
    flat_tile_collider_boxes,
)

#: The block selection every shipped generator predicts.  Read from the recording where
#: possible; this is the fallback for a file that predates the attribute.
DEFAULT_BLOCKS: tuple[str, ...] = ("base_pose", "base_vel", "joint_pos")

#: ``terrain_id`` of a generated flat tile, which has no manifest entry.
FLAT_TERRAIN_ID: int = -1

#: Footprint (m) assumed for a flat tile when the library does not state one.
#:
#: Only the GROUND matters for a flat tile and it is level everywhere, so the extent decides
#: nothing except where the SDF stops being ground -- which no plan on a flat tile reaches.
FLAT_TILE_FALLBACK_FOOTPRINT: tuple[float, float] = (60.0, 60.0)


@dataclass(frozen=True)
class TileTable:
    """Per-tile collider boxes in TILE-LOCAL coordinates, plus the route each tile declares.

    ``terrain_id -> row`` so an episode's ``terrain_id`` indexes it directly, and the routes
    are what recover a tile's world corner from a recorded goal (see
    :func:`corner_from_goal`).
    """

    terrain_ids: tuple[int, ...]
    """Tile ids, in row order."""

    boxes: torch.Tensor
    """``(T, B, 6)`` padded tile-local collider boxes."""

    mask: torch.Tensor
    """``(T, B)`` bool validity."""

    routes: dict[int, dict]
    """``terrain_id -> route`` block, for the tiles that declare one."""

    def row_of(self, terrain_id: int) -> int:
        """Row index of a tile id.

        Args:
            terrain_id: The tile id.

        Returns:
            Its row in :attr:`boxes`.

        Raises:
            KeyError: If the library has no such tile -- a silent fallback here would grade
                every plan on that tile against the wrong geometry.
        """
        try:
            return self.terrain_ids.index(int(terrain_id))
        except ValueError as exc:
            raise KeyError(
                f"terrain_id {terrain_id} is not in this library (has "
                f"{list(self.terrain_ids)}). The rollout and the clip library disagree about "
                "the terrain; check the recording's clip_library_path attribute."
            ) from exc


def load_tile_table(clip_path: str, dtype: torch.dtype = torch.float64) -> TileTable:
    """Build the per-tile collider table from a clip library's ``terrain_manifest``.

    Args:
        clip_path: The ``.h5`` library.
        dtype: Dtype for the box tensor.

    Returns:
        The :class:`TileTable`, including a flat-tile row when the library has flat clips.

    Raises:
        ValueError: If the library carries no ``terrain_manifest``.
    """
    with h5py.File(clip_path, "r") as f:
        raw = f.attrs.get("terrain_manifest")
        if raw is None:
            raise ValueError(f"{clip_path} has no terrain_manifest attribute.")
        manifest = json.loads(raw if isinstance(raw, str) else raw.decode())
        has_flat = ("terrain_id" in f
                    and bool(np.any(f["terrain_id"][:] == FLAT_TERRAIN_ID)))

    ids, sets, routes = [], [], {}
    for entry in manifest:
        tid = int(entry["terrain_id"])
        ids.append(tid)
        sets.append(collider_boxes_from_manifest_tile(entry))
        if "route" in entry:
            routes[tid] = entry["route"]
    if has_flat:
        ids.append(FLAT_TERRAIN_ID)
        sets.append(flat_tile_collider_boxes(*FLAT_TILE_FALLBACK_FOOTPRINT))

    boxes, mask = pad_box_sets(sets, dtype=dtype)
    return TileTable(tuple(ids), boxes, mask, routes)


def corner_from_goal(
    table: TileTable,
    terrain_id: np.ndarray,
    goal_pos_w: np.ndarray,
) -> np.ndarray:
    """Each episode's tile corner in the world, from its recorded goal position.

    A rollout stores every episode's world goal and the manifest states that goal in
    tile-local coordinates, so the corner is their difference.  That is exact and needs
    neither the terrain assembler nor IsaacLab -- which matters because Stage 1 has to grade
    rollouts that were recorded before a corner was ever written down.

    **z is taken as 0, not differenced.**  ``tile_corner`` is defined with ``z = 0`` by the
    importer, while a recorded ``goal_pos_w_z`` is a PELVIS height (a clip-free route
    declares ``goal_z`` as the walkable surface and the goal anchor adds the standing
    height), so differencing z would bake a ~0.79 m offset into every tile.
    :func:`placement_residual` is the check that this is right rather than assumed.

    Args:
        table: The tile table, for the routes.
        terrain_id: ``(E,)`` per-episode tile id.
        goal_pos_w: ``(E, 3)`` per-episode world goal position.

    Returns:
        ``(E, 3)`` world corner per episode, with ``z = 0``.

    Raises:
        KeyError: If a tile in ``terrain_id`` declares no route, so its corner cannot be
            recovered this way.
    """
    corner = np.zeros((len(terrain_id), 3), dtype=np.float64)
    for tid in np.unique(terrain_id):
        tid = int(tid)
        sel = terrain_id == tid
        if tid == FLAT_TERRAIN_ID:
            continue                       # a flat tile is level everywhere; corner is moot
        route = table.routes.get(tid)
        if route is None:
            raise KeyError(
                f"terrain_id {tid} declares no route, so its world corner cannot be "
                "recovered from a recorded goal. Re-record with cell_corner stored, or "
                "reconstruct the layout with assemble_clip_terrain."
            )
        corner[sel, 0] = goal_pos_w[sel, 0] - float(route["goal_x"])
        corner[sel, 1] = goal_pos_w[sel, 1] - float(route["goal_y"])
    return corner


def read_window_block(
    path: str,
    names: list[str],
    lo: int,
    hi: int,
) -> dict[str, np.ndarray]:
    """Read a contiguous slice of several rollout-window datasets.

    Args:
        path: The rollout ``.h5``.
        names: Dataset names to read.
        lo / hi: Row range.

    Returns:
        Mapping name -> array for that slice.
    """
    with h5py.File(path, "r") as f:
        return {n: f[n][lo:hi] for n in names}


def rollout_meta(path: str) -> dict:
    """The rollout's own attributes, JSON-decoded where they are JSON.

    Args:
        path: The rollout ``.h5``.

    Returns:
        The attribute mapping, with ``columns`` / ``*_layout`` parsed.
    """
    out: dict = {}
    with h5py.File(path, "r") as f:
        for k, v in f.attrs.items():
            if isinstance(v, (str, bytes)):
                s = v if isinstance(v, str) else v.decode()
                try:
                    out[k] = json.loads(s)
                except (json.JSONDecodeError, ValueError):
                    out[k] = s
            else:
                out[k] = v.tolist() if isinstance(v, np.ndarray) else v
        out["_num_written"] = int(f.attrs.get("num_windows_written",
                                              f["episode_uid"].shape[0]))
        out["_plan_nodes"] = int(f["fm_plan"].shape[1])
        out["_plan_width"] = int(f["fm_plan"].shape[2])
    return out


def plan_to_world(
    plan: torch.Tensor,
    anchor_pos: torch.Tensor,
    anchor_quat: torch.Tensor,
    num_joints: int,
    num_ee_frames: int,
    blocks: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lift a recorded plan into the world through the window's own inverse map.

    Uses ``robot_rl.flow_matching.window.window_to_world`` rather than re-deriving the lift:
    the anchor frame is YAW-ONLY (the anchor's roll and pitch are deliberately dropped, the
    reference's are not), and restating that here is exactly the kind of frame detail that
    goes wrong silently.

    Args:
        plan: ``(N, F, D)`` recorded ``fm_plan``.
        anchor_pos: ``(N, 3)`` world anchor position.
        anchor_quat: ``(N, 4)`` world anchor orientation, xyzw.
        num_joints: ``nj``.
        num_ee_frames: EE frames the layout sizes for.
        blocks: The block selection the plan was packed with.

    Returns:
        ``(base_pos_w (N, F, 3), base_rot_w (N, F, 3, 3), joint_pos (N, F, nj),
        base_quat_w (N, F, 4) xyzw)`` -- the rotation twice, because the collider FK wants a
        matrix and posing a URDF wants a quaternion, and converting between them at the call
        site is where a wxyz/xyzw mix-up would silently twist the robot.
    """
    from isaaclab.utils.math import matrix_from_quat            # noqa: PLC0415
    from robot_rl.flow_matching.window import window_to_world   # noqa: PLC0415

    # The anchor needs its own node axis: ``window_to_world`` broadcasts it over nodes, so a
    # bare (N, 3) is read as (N,) anchors of width F and refused.
    world = window_to_world(plan.float(), anchor_pos.float()[:, None, :],
                            anchor_quat.float()[:, None, :],
                            num_joints, num_ee_frames, list(blocks))
    n, f = world["base_pos_w"].shape[:2]
    rot = matrix_from_quat(world["base_quat_w"].reshape(n * f, 4)).reshape(n, f, 3, 3)
    return (world["base_pos_w"].to(plan.dtype), rot.to(plan.dtype),
            world["joint_pos"].to(plan.dtype), world["base_quat_w"].to(plan.dtype))


def placement_residual(
    sole_z: np.ndarray,
    terrain_top: np.ndarray,
) -> dict[str, float]:
    """How far the plan's first-node soles sit off the terrain they were placed on.

    The frame check.  Every element of the placement -- the tile corner, the anchor, the
    yaw-frame lift, the FK -- lands in this one number, so a systematic offset here means
    the reward is grading a displacement rather than a plan.  ``augment_registration.py``
    makes the same argument for its own filter: a placement check has to be a measurement.

    Args:
        sole_z: ``(M,)`` world z of the lowest sole corner at node 0.
        terrain_top: ``(M,)`` terrain surface height under it.

    Returns:
        Median / quantile summary of ``sole_z - terrain_top`` (m).
    """
    d = np.asarray(sole_z, dtype=np.float64) - np.asarray(terrain_top, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {"n": 0}
    return {
        "n": int(d.size),
        "median_m": float(np.median(d)),
        "mean_m": float(d.mean()),
        "p05_m": float(np.quantile(d, 0.05)),
        "p95_m": float(np.quantile(d, 0.95)),
        "frac_within_5cm": float(np.mean(np.abs(d) < 0.05)),
    }


def terrain_for_rows(table: TileTable, terrain_id: np.ndarray, corner: np.ndarray,
                     device: torch.device | str = "cpu") -> BoxUnionTerrain:
    """One world-frame :class:`BoxUnionTerrain` row per input row.

    Args:
        table: The tile table.
        terrain_id: ``(N,)`` tile id per row.
        corner: ``(N, 3)`` world tile corner per row.
        device: Device for the result.

    Returns:
        A batched terrain of length ``N``.
    """
    rows = torch.as_tensor([table.row_of(int(t)) for t in terrain_id], dtype=torch.long)
    tiles = BoxUnionTerrain(table.boxes, table.mask)
    return tiles.index(rows).translated(
        torch.as_tensor(corner, dtype=table.boxes.dtype)).to(device)

#: Where a rollout's own recorded joint order is stamped, once the recorder stamps it.
ROLLOUT_JOINT_ORDER_ATTR: str = "joint_names_isaac"


def isaac_joint_names(rollout_path: str, override: str = "") -> list[str]:
    """The joint order a rollout's ``fm_plan`` / ``proprio`` blocks are actually in.

    **This is not the clip order, and confusing the two is the single most dangerous mistake
    available here.**  The policy's order is interleaved
    (``left_hip_pitch, right_hip_pitch, waist_yaw, ...``) while a clip library's ``y_des``
    joints are per limb (``left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, ...``);
    27 of the G1's 29 positions differ.  FK'ing a plan under the clip order yields a
    PLAUSIBLE WRONG SKELETON -- a robot standing roughly in the right place with the wrong
    posture -- so every geometric number comes back finite, differently wrong, and
    believable.  It got past a placement check that requires 90% of soles to be over their
    own tile.

    Resolution order, and there is deliberately **no fallback**:

    1. the rollout's own ``joint_names_isaac`` attribute, once the recorder stamps it;
    2. ``override``, a ``policy_parameters.yaml`` or a run directory;
    3. the ``exported/policy_parameters.yaml`` of the run that owns the rollout's
       ``checkpoint_path``.

    Args:
        rollout_path: The rollout ``.h5``.
        override: An explicit ``policy_parameters.yaml``, ``exported/`` dir, or run dir.

    Returns:
        The joint names, in the order the recorded blocks use.

    Raises:
        FileNotFoundError: If no artifact states the order.  Guessing it is exactly the
            failure this function exists to prevent.
    """
    with h5py.File(rollout_path, "r") as f:
        stamped = f.attrs.get(ROLLOUT_JOINT_ORDER_ATTR)
    if stamped is not None:
        raw = stamped if isinstance(stamped, str) else stamped.decode()
        return list(json.loads(raw)) if raw.lstrip().startswith("[") else list(stamped)

    candidates: list[pathlib.Path] = []
    if override:
        p = pathlib.Path(override)
        candidates += [p] if p.suffix in (".yaml", ".yml") else [
            p / "policy_parameters.yaml", p / "exported" / "policy_parameters.yaml"]
    ckpt = trio_meta_checkpoint(rollout_path)
    if ckpt:
        run = pathlib.Path(ckpt).parent
        candidates.append(run / "exported" / "policy_parameters.yaml")

    for cand in candidates:
        if cand.exists():
            import yaml                                              # noqa: PLC0415
            names = yaml.safe_load(cand.read_text()).get("joint_names_isaac")
            if names:
                return list(names)
    raise FileNotFoundError(
        f"cannot determine the joint order for {rollout_path}. Tried its own "
        f"{ROLLOUT_JOINT_ORDER_ATTR!r} attribute and {[str(c) for c in candidates]}. "
        "Pass --isaac_joint_names <policy_parameters.yaml>. There is no fallback ON PURPOSE: "
        "the clip order differs from the policy order on 27 of 29 joints, and FK'ing a plan "
        "under the wrong one returns a plausible, wrong skeleton with no error."
    )


def trio_meta_checkpoint(rollout_path: str) -> str:
    """The ``checkpoint_path`` attribute of a rollout, or ``""``.

    Args:
        rollout_path: The rollout ``.h5``.

    Returns:
        The recorded checkpoint path.
    """
    with h5py.File(rollout_path, "r") as f:
        v = f.attrs.get("checkpoint_path", "")
    if isinstance(v, bytes):
        v = v.decode()
    return str(v).strip('"')
