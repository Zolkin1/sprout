# Copyright 2026 Zachary Olkin. All rights reserved.

"""Turn recorded teacher rollouts into autoregressive reference-window FM training items.

The dataset operates on already-loaded numpy arrays plus a ``meta`` object (duck
-typed; only its layout attributes are read), so it has no dependency on the
recording scripts and is unit-testable with synthetic data. The training script
loads the ``.npz`` via ``recorded_data_io.load_recorded_data`` and passes
``(data, meta)`` here.

Paradigm (``target_source="ego_anchored_clip_future"``): each training item is
sampled at a **snap (re-anchor) event** of a recorded teacher rollout. It pairs

* ``history``: the ``N`` reference nodes APPLIED at the ``N`` steps before the snap --
  what the tracker had just been tracking -- each expressed relative to the robot as it
  was at *that* step;
* ``height``: the exteroception tokens at the snap -- the pelvis height scan (one
  token) or the two depth cameras (one token each), per ``cfg.extero_source``;
* ``cmd_tokens``: the strided FUTURE velocity command over the F-step window;
* ``state`` (the target): the ``F``-node clip future **ego-anchored onto the robot** at
  the snap (recorded ``ref_future``), all in the snap robot's frame -- node 0's base
  offset is the live tracking error, ≈ 0 by construction.

Both blocks are the same thing (:mod:`~robot_rl.flow_matching.window` nodes -- the exact
vector ``mdp.reference_window`` feeds the policy), so history and target share one
normalizer, and at deploy the model conditions on nodes it emitted itself. The measured
robot enters only as the frame each block is expressed relative to, and it cancels.

Nothing here needs a re-record beyond the window geometry: ``record_data.py`` already
writes the applied-reference ring (``history_ref`` / per-step ``ref_state``) alongside the
measured one, column-for-column in the same layout.

The exposure-bias caveat is worth stating plainly: this history is the CLIP TEACHER's
applied reference, never the model's own output, so training and deploy see different
history distributions. ``cfg.null_cond_prob`` and the per-channel history noise
(:meth:`FutureStateDataset._state_noise_std`) are what stand between them.
"""

from __future__ import annotations

import json
import math
from typing import Any, Sequence


import numpy as np
import torch
from torch.utils.data import Dataset

from .config import FlowMatchingConfig
from .normalizer import Normalizer
from .transforms import sanitize_extero
from .window import (
    REFERENCE_WINDOW_NOISE,
    expand_window_blocks,
    resolve_reference_blocks,
    window_block_slices,
    window_dim,
    world_to_window,
)


#: Loss weight -> the window blocks the term relates. A term whose blocks are not all
#: selected has nothing to tie together, so :func:`resolve_block_selection` zeroes its
#: weight rather than letting it index columns that are not there.
_LOSS_TERM_BLOCKS: dict[str, tuple[str, ...]] = {
    "vel_consistency_weight": ("base_pose", "base_vel"),
    "joint_vel_consistency_weight": ("joint_pos", "joint_vel"),
    "fk_pose_weight": ("joint_pos", "ee_pos"),
    "body_pose_root_weight": ("joint_pos",),
    "body_pose_world_weight": ("joint_pos", "base_pose"),
}


#: Which ``RecordingMeta`` field carries each source's sensor contract. Both state
#: ``height``/``width``, so one image-shape resolver serves both.
_EXTERO_SPEC_FIELD: dict[str, str] = {
    "depth": "depth_camera_spec",
    "height_scan": "height_scan_spec",
}


def _check_scan_grid_closes(entry: dict, sensor_dim: int) -> None:
    """Require a scan spec's PHYSICAL fields to reproduce its stated fold.

    This is the check ``extero_sensor_dim`` cannot make. A scan spec states its grid three
    independent ways -- ``resolution`` + ``size``, ``(height, width)``, and ``num_rays`` -- so
    recomputing the fold from the physical pair and comparing catches a wrong extent or
    resolution, which would otherwise sail through every width-based assertion and hand the
    convolution a correctly-shaped image of the wrong geometry.

    A no-op for a depth spec, which has no ``resolution``/``size`` to recompute from (its
    equivalent guarantee is the pinhole intrinsics agreement on the deploy path).

    Args:
        entry: One sensor's spec dict.
        sensor_dim: Recorded row width for that sensor.

    Raises:
        ValueError: If the physical fields do not reproduce the stated fold or ray count.
    """
    if "resolution" not in entry or "size" not in entry:
        return
    from .transforms import scan_grid_shape

    stated = (int(entry["height"]), int(entry["width"]))
    recomputed = scan_grid_shape(
        float(entry["resolution"]), entry["size"], str(entry.get("ordering", "xy")))
    if recomputed != stated:
        raise ValueError(
            f"scan '{entry.get('sensor')}' states a {stated[0]}x{stated[1]} fold, but "
            f"{entry['size']} m at {entry['resolution']} m with ordering "
            f"{entry.get('ordering', 'xy')!r} is {recomputed[0]}x{recomputed[1]}. The "
            "recording's geometry and its fold disagree, so one of them is wrong and the "
            "image would be built from the wrong one."
        )
    if "num_rays" in entry and int(entry["num_rays"]) != sensor_dim:
        raise ValueError(
            f"scan '{entry.get('sensor')}' states num_rays={entry['num_rays']} but the "
            f"recorded row is {sensor_dim} wide."
        )


def declared_state_scales(
    cfg: FlowMatchingConfig, num_joints: int, num_extra_frames: int,
    blocks: Sequence[str] | None,
) -> torch.Tensor | None:
    """Per-channel declared scale over one window node, ``0`` where the block stays FITTED.

    Expanded through :func:`expand_window_blocks`, the same helper the noise and observation
    scale tables use, so a layout or block-selection change cannot land here and not there.

    Args:
        cfg: Training config (the ``state_scale_*`` fields are read).
        num_joints: ``nj``.
        num_extra_frames: ``K``.
        blocks: The window's block selection.

    Returns:
        ``(D,)`` scales, or ``None`` when no block declares one.

    Raises:
        ValueError: On a non-positive declared scale, which would divide by zero or flip sign.
    """
    table = {
        "base_pos": cfg.state_scale_base_pos,
        # Never declared: the rot6d block is already exempt from standardization.
        "base_ori": None,
        "base_lin_vel": cfg.state_scale_base_lin_vel,
        "base_ang_vel": cfg.state_scale_base_ang_vel,
        "joint_pos": cfg.state_scale_joint_pos,
        "joint_vel": cfg.state_scale_joint_vel,
        "ee_pos": cfg.state_scale_ee_pos,
    }
    for key, value in table.items():
        if value is not None and float(value) < 0.0:
            raise ValueError(
                f"state_scale_{key}={value} must be positive, or 0/None to fit that block.")
    # 0 and None both mean "fit this block per channel" -- 0 so the CLI has an off switch.
    zeroed = {k: (0.0 if v is None else float(v)) for k, v in table.items()}
    if not any(v > 0.0 for v in zeroed.values()):
        return None
    return torch.tensor(
        expand_window_blocks(num_joints, num_extra_frames, zeroed, blocks), dtype=torch.float32)


def apply_declared_scales(normalizer: Normalizer, declared: torch.Tensor | None) -> Normalizer:
    """Override the fitted statistics wherever a block declares its own scale.

    A declared channel becomes ``mean = 0, std = scale``; the rest keep what was fitted. The
    normalizer's SHAPE is unchanged, so the checkpoint, the folded ONNX buffers and the whole
    deploy path are identical either way -- only the numbers move.

    Args:
        normalizer: The freshly fitted state normalizer.
        declared: ``(D,)`` from :func:`declared_state_scales`, or ``None``.

    Returns:
        The same normalizer when nothing is declared, else a new one.
    """
    if declared is None:
        return normalizer
    sel = declared > 0
    return Normalizer(
        torch.where(sel, torch.zeros_like(normalizer.mean), normalizer.mean),
        torch.where(sel, declared.to(normalizer.std.dtype), normalizer.std),
        normalizer.standardize_mask.clone(),
    )


def declared_scale_summary(
    block_slices: dict[str, slice], declared: torch.Tensor, fitted_std: torch.Tensor,
) -> str:
    """``name[k/width] scale(s) vs fitted [min,max]`` per block, over the DECLARED channels.

    Restricting the fitted range to the channels a declaration actually REPLACES is the
    whole point. Taken over the full block slice it reported numbers no declaration touches:
    ``base_pose`` spans position + rot6d, and the rot6d columns are exempt from
    standardization, so their std is the mask-forced 1.0 -- which is what made the shipped
    line read ``base_pose 0.4 vs [0.090,1.000]`` and the declared 0.4 look like it was
    overriding a fitted 1.0 when the position channels it replaces fit to 0.442.

    Two smaller things it also gets right: a block can hold MORE than one declared scale
    (``base_vel`` is lin 0.5 + ang 0.4, and printing ``declared[s.start]`` showed only the
    lin one), and a block whose first channel is undeclared but whose later ones are (only
    ``state_scale_base_ang_vel`` set) was dropped from the line entirely.

    Args:
        block_slices: Block name -> its slice within one window node.
        declared: ``(D,)`` declared scales, ``0`` where the channel stays fitted.
        fitted_std: ``(D,)`` std as FITTED, before the declarations were applied.

    Returns:
        A ``"; "``-joined summary, empty when nothing is declared.
    """
    parts = []
    for name, s in block_slices.items():
        # A block can be ZERO WIDTH (ee_pos with no extra frames), and its slice then starts
        # at state_dim -- indexing it would overrun.
        if s.stop <= s.start:
            continue
        pick = declared[s] > 0
        if not bool(pick.any()):
            continue
        scales = ",".join(dict.fromkeys(f"{float(v):g}" for v in declared[s][pick]))
        kept = fitted_std[s][pick]
        parts.append(f"{name}[{int(pick.sum())}/{s.stop - s.start}] {scales} vs fitted "
                     f"[{float(kept.min()):.3f},{float(kept.max()):.3f}]")
    return "; ".join(parts)


def _yaw_from_quat_xyzw(quat: torch.Tensor) -> torch.Tensor:
    """Heading angle (radians) of an ``(..., 4)`` xyzw quaternion.

    The same extraction ``yaw_quat`` performs, as a scalar rather than a rotation, so the
    angle between the terrain's mask frame and the window's own frame can be taken directly.
    """
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _resolve_extero_coord_extent(spec: list[dict] | None) -> list[float] | None:
    """The metric ``[size_x, size_y]`` of a robot-centred sensor grid, or ``None``.

    Present only for a ray-cast grid whose pixels are ground POINTS at known offsets from the
    robot; a depth pixel is a ray and has no such planar coordinate, so depth returns ``None``
    and its encoder is unchanged.

    Args:
        spec: The resolved sensor contract, or ``None``.

    Returns:
        ``[size_x, size_y]`` in metres, or ``None``.

    Raises:
        ValueError: If several sensors disagree on the extent -- they share one coordinate
            grid, so there is no single answer to give the encoder.
    """
    if not spec or "size" not in spec[0]:
        return None
    extents = {(float(s["size"][0]), float(s["size"][1])) for s in spec}
    if len(extents) != 1:
        raise ValueError(
            f"sensors state different metric extents {sorted(extents)}; they share one "
            "coordinate grid in the encoder, so give them one geometry."
        )
    sx, sy = extents.pop()
    return [sx, sy]


#: Ray count above which a height scan is "dense" and ``extero_encoder='auto'`` is refused.
#: The validated coarse grid is 176 rays; a metric scan is 121x121 = 14641. Anything in
#: between is closer to the metric case than to the coarse one, and the cost of guessing wrong
#: is a hundred epochs.
_DENSE_SCAN_RAYS: int = 1000


#: How far a DECLARED ``extero_xyz_offset`` may sit from the contract-derived one before the
#: dataset build says so. Generous on purpose: this is a stale-geometry alarm, not a
#: tolerance. The failure it exists for -- the far camera going 35 -> 15 deg -- moves the
#: offset by ~2.4 m, so anything at this scale is a different camera, not a rounding.
_XYZ_OFFSET_DRIFT_WARN_M = 0.25


def _select_depth_lag(cfg, data, rows, hs, n_sensors, sensor_dim, snap_meas, ds):
    """Draw a camera-latency lag per window and swap in that frame and its capture state.

    Writes ``ds._lagged_hs`` / ``ds._lagged_pose`` and returns the resolved, normalised
    probability tuple for the checkpoint to record.

    Args:
        cfg: The run config; ``depth_latency_probs`` indexes lag in control steps.
        data: The opened recording.
        rows: Window row indices into the recording.
        hs: The extero rows already selected at the snap, ``[Nw, n_sensors * sensor_dim]``.
        n_sensors: Number of sensors.
        sensor_dim: Width of one sensor reading.
        snap_meas: State rows at the snap, the fallback pose source.
        ds: The dataset, written through.

    Raises:
        ValueError: If the recording predates the lagged arrays, or holds fewer lags than the
            distribution asks for. Never silently truncates: a latency model that quietly
            became a no-op would be indistinguishable from one that worked.
    """
    probs = [float(p) for p in cfg.depth_latency_probs]
    if min(probs) < 0.0 or sum(probs) <= 0.0:
        raise ValueError(f"depth_latency_probs must be non-negative and sum > 0, got {probs}")
    probs = [p / sum(probs) for p in probs]
    if data.get("depth_lag") is None:
        raise ValueError(
            "depth_latency_probs is set but this recording has no 'depth_lag' array, so the "
            "older frames it would sample were never stored. Re-record with the current "
            "record_data.py (it keeps DEPTH_LAG_STEPS frames), or clear depth_latency_probs."
        )
    n_lags = int(data["depth_lag"].shape[1])
    if len(probs) > n_lags:
        raise ValueError(
            f"depth_latency_probs asks for {len(probs)} lags but the recording holds "
            f"{n_lags}. Re-record with a larger DEPTH_LAG_STEPS or shorten the distribution."
        )
    idx = rows.numpy() if torch.is_tensor(rows) else rows
    gen = np.random.default_rng(int(cfg.seed))
    lag = gen.choice(len(probs), size=len(idx), p=probs)
    lagged = torch.from_numpy(
        np.ascontiguousarray(data["depth_lag"][idx][np.arange(len(idx)), lag])).float()
    ds._lagged_hs = lagged.reshape(lagged.shape[0], n_sensors * sensor_dim)
    state = data.get("depth_lag_state")
    if state is None:
        raise ValueError(
            "'depth_lag' is present but 'depth_lag_state' is not, so a lagged frame has no "
            "capture pose to unproject with. Re-record."
        )
    ds._lagged_pose = torch.from_numpy(
        np.ascontiguousarray(state[idx][np.arange(len(idx)), lag])).float()
    hist = [int((lag == k).sum()) for k in range(len(probs))]
    mean_ms = 20.0 * float((lag * 1.0).mean())
    print(f"[fm-data] camera latency: lags {hist} of {len(idx)} windows "
          f"(target {[round(p, 3) for p in probs]}), mean {mean_ms:.1f} ms at 20 ms/step. "
          "Each lagged frame is unprojected with its own capture pose and left in that frame.")
    return tuple(probs)


def _resolve_xyz_contract(
    cfg: FlowMatchingConfig, spec: list[dict], joint_names: list[str]
) -> tuple[list[list[float]], float]:
    """Derive the declared xyz offsets and clamp from the camera contract. No data involved.

    The offset for a camera is where its BORESIGHT meets the nominal ground plane, in the
    root's yaw-aligned frame, with the robot at zero joints and upright. Every term of that is
    a constant of the robot and the mount: FK from the pelvis to the mount body, the contract's
    nominal offset, and a ground plane taken from the ankle at zero joints. Nothing consults
    the recording, which is the point -- the same robot yields the same constants in every
    dataset, so a checkpoint's notion of "one metre from the robot" does not depend on which
    terrain happened to be under it the day the normalizer was fitted.

    The boresight rather than the cloud's centroid, because the centroid is a fact about
    terrain: a wide downward FOV maps the upper image rows much farther out than the lower
    ones, so on flat ground the centroid already sits well past the boresight, and over a
    drop-off it runs away entirely. The boresight intersection is the one point in the view
    that is defined by the camera alone.

    The clamp is sized to each camera's full max-range CONE post-offset, so it cannot bind on
    a physically reachable point -- ``extero_clip`` already bounds range long before this does.

    Args:
        cfg: Training config; ``extero_xyz_scale``, ``extero_nominal_ground_z`` and
            ``fk_model_path`` are read.
        spec: The per-camera contract, in row order.
        joint_names: The recording's joint order, for the FK chain.

    Returns:
        ``(offsets, clip)`` -- one ``[x, y, z]`` per camera in metres, and the symmetric bound
        on the normalised xyz channels.

    Raises:
        ValueError: If a camera's boresight does not point downward, so no ground intersection
            exists and the nominal offset is undefined.
    """
    from motion_gen.fk import FKChain  # noqa: PLC0415

    from robot_rl.tasks.manager_based.robot_rl.mdp.observations.depth import (  # noqa: PLC0415
        pinhole_ray_directions,
    )

    nj = len(joint_names)
    ground = cfg.extero_nominal_ground_z
    if ground is None:
        # The sole at zero joints: the ankle roll link, dropped by the collider half-height.
        chain = FKChain(cfg.fk_model_path, list(joint_names), ["left_ankle_roll_link"],
                        base_link="pelvis_link", dtype=torch.float32)
        pos, _, _, _ = chain.pose_and_jacobian(torch.zeros(1, nj))
        ground = float(pos[0, 0, 2]) - 0.03
        cfg.extero_nominal_ground_z = round(ground, 4)

    poses = FutureStateDataset._camera_pose_yaw_frame(
        spec, torch.tensor([[0.0, 0.0, 0.0, 1.0]]), torch.zeros(1, nj), joint_names,
        cfg.fk_model_path,
    )
    scale = float(cfg.extero_xyz_scale)
    offsets, envelope = [], 0.0
    for cam, (p, rot) in zip(spec, poses):
        p, rot = p[0], rot[0]
        boresight = rot[:, 0]                                   # camera +x is the optical axis
        if float(boresight[2]) >= -1e-6:
            raise ValueError(
                f"camera {cam.get('sensor')!r} has a boresight of {boresight.tolist()}, which "
                "does not point downward, so it never meets the nominal ground plane and has "
                "no geometric offset. Set extero_xyz_offset explicitly."
            )
        hit = p + boresight * ((ground - float(p[2])) / float(boresight[2]))
        offsets.append([round(float(v), 2) for v in hit])       # rounded: it is a declared datum
        dirs = pinhole_ray_directions(
            int(cam["width"]), int(cam["height"]), float(cam["fx"]), float(cam["fy"]),
            float(cam["cx"]), float(cam["cy"]), p.device, p.dtype,
        )                                                        # (K, 3) in the camera frame
        # The furthest a pixel can REACH, which is not ``max_dist``. Two ways a ray gets long:
        # an invalid pixel is placed at ``max_dist`` of ray length exactly, and a valid one
        # carries a Z-DEPTH bounded by the clip band, which becomes a ray length once divided
        # by the cosine to the optical axis -- so a frame CORNER reaches far past the band's
        # nominal range. Sizing this from ``max_dist`` alone under-bounds the cone by that
        # cosine and the clamp then bites real geometry. It also means the depth band and this
        # envelope are coupled: widening ``extero_clip`` widens the reach.
        band_m = float(cfg.resolve_extero_clip()) / float(cam["scale"]) + float(cam["offset"])
        reach = torch.clamp(band_m / dirs[:, 0], min=float(cam["max_dist"]))
        cone = p + (dirs @ rot.T) * reach[:, None]
        envelope = max(envelope, float((cone - torch.tensor(offsets[-1])).abs().max()))
    return offsets, math.ceil(envelope / scale * 2.0) / 2.0


def _extero_contract_normalizer(
    normalizer: Normalizer, cfg: FlowMatchingConfig, n_sensors: int, sensor_dim: int
) -> Normalizer:
    """Build the exteroception normalizer from DECLARED constants, discarding the fit entirely.

    ``d`` and ``v`` become identities and ``xyz`` gets the contract's per-camera offset over one
    shared scale. See :attr:`FlowMatchingConfig.extero_norm_mode` for why each is what it is.

    Riding the ``Normalizer`` rather than transforming the rows is what keeps this free
    downstream: it is still one ``(x - mean) / std`` of shape ``(T*K,)``, so the checkpoint, the
    affine folded into ``fm.onnx`` and the whole deploy path are untouched -- the numbers are
    simply declared instead of measured. Nothing new has to be plumbed to a robot.

    Args:
        normalizer: The fitted normalizer, read only for dtype/device/shape.
        cfg: Training config; the resolved xyz constants and channel names are read.
        n_sensors: Cameras packed into the flat width.
        sensor_dim: Elements per camera, channels included.

    Returns:
        A normalizer of identical shape carrying only declared constants.

    Raises:
        ValueError: If the declared offsets do not cover every sensor.
    """
    names = [str(n) for n in cfg.extero_channel_names]
    channels = len(names)
    n_px = sensor_dim // channels
    offsets = cfg.extero_xyz_offset or []
    if names != ["d"] and len(offsets) != n_sensors:
        raise ValueError(
            f"extero_xyz_offset has {len(offsets)} entry/entries for {n_sensors} sensor(s); "
            "the contract normalization needs one per camera."
        )
    dtype, device = normalizer.mean.dtype, normalizer.mean.device
    mean = torch.zeros(n_sensors, channels, n_px, dtype=dtype, device=device)
    std = torch.ones(n_sensors, channels, n_px, dtype=dtype, device=device)
    mask = torch.zeros(n_sensors, channels, n_px, dtype=torch.bool, device=device)
    scale = float(cfg.extero_xyz_scale)
    for axis, name in enumerate(names):
        if name not in ("x", "y", "z"):
            continue                                     # d and v are identities, by contract
        for s in range(n_sensors):
            mean[s, axis] = float(offsets[s]["xyz".index(name)])
            std[s, axis] = scale
            mask[s, axis] = True
    print(f"[fm-data] extero_norm_mode=contract: d and v IDENTITY; xyz over one declared scale "
          f"{scale:g} m with per-camera offsets {offsets} m (boresight on the nominal ground "
          f"at z={cfg.extero_nominal_ground_z}). No statistic is fitted.")
    return Normalizer(mean.reshape(-1).contiguous(), std.reshape(-1).contiguous(),
                      mask.reshape(-1).contiguous())


def _pool_extero_per_channel(
    normalizer: Normalizer, cfg: FlowMatchingConfig, n_sensors: int, sensor_dim: int
) -> Normalizer:
    """One statistic per sensor per CHANNEL GROUP, uniform over pixels.

    The mode that makes a multi-channel depth row coherent. A 5-channel pixel is three
    different quantities -- normalised depth, metres of position, a 0/1 indicator -- so one
    statistic per sensor (``per_sensor``) produces a std dominated by xyz that compresses
    depth and squashes validity toward invisibility.

    The grouping comes from :meth:`FlowMatchingConfig.extero_channel_groups` and is NOT simply
    per channel: ``x, y, z`` share one SCALE so a metre stays a metre in every direction.
    Fitting the scale separately would rescale the axes against each other -- lateral spread is
    the largest and vertical the smallest -- and destroy exactly the metric comparability the
    channels were added to provide. The OFFSET is still per axis, because subtracting one is a
    rigid translation that preserves all of that while actually centring the channel.

    ``v`` is left as an IDENTITY: it is already 0/1, and fitting it would map "valid" and
    "invalid" onto two arbitrary values that move with the invalid RATE of whichever recording
    was used, so a cleaner recording would hand the network a different number for the same
    fact. This is the one place the ``standardize_mask`` is narrowed rather than inherited.

    Args:
        normalizer: The freshly fitted per-element normalizer.
        cfg: Training config; the channel names and ``extero_norm_eps`` are read.
        n_sensors: Number of sensor rows packed into the flat width.
        sensor_dim: Elements per sensor, channels included.

    Returns:
        A normalizer whose statistics are constant within each (sensor, group), same
        ``(T*K,)`` shape as the input.

    Raises:
        ValueError: If the channel count does not divide ``sensor_dim``.
    """
    names = [str(n) for n in cfg.extero_channel_names]
    channels = len(names)
    if channels < 1 or sensor_dim % channels:
        raise ValueError(
            f"extero_channel_names {names} implies {channels} channel(s), which does not "
            f"divide a sensor row of {sensor_dim}."
        )
    n_px = sensor_dim // channels
    # (sensors, channels, pixels) -- CHANNEL-MAJOR, matching the row the encoder folds. Reading
    # this as (pixels, channels) would pool across channels while looking correct.
    mean = normalizer.mean.reshape(n_sensors, channels, n_px).clone()
    std = normalizer.std.reshape(n_sensors, channels, n_px).clone()
    mask = normalizer.standardize_mask.reshape(n_sensors, channels, n_px).clone()
    eps = float(cfg.extero_norm_eps)

    report: list[str] = []
    for group in cfg.extero_channel_groups():
        idx = [names.index(n) for n in group]
        if group == ("v",):
            mean[:, idx] = 0.0
            std[:, idx] = 1.0
            mask[:, idx] = False
            report.append("v: identity (0/1 indicator, not standardized)")
            continue
        # The STD is pooled over the group; the MEAN is not, and the asymmetry is the point.
        #
        # Sharing the scale is what keeps the axes metrically comparable -- one divisor, so a
        # metre reads the same in x, y and z. Sharing the OFFSET buys nothing and costs
        # conditioning: subtracting a per-axis mean is a rigid TRANSLATION of the cloud, which
        # preserves every relative distance exactly, while one shared offset leaves the axes
        # sitting wherever their own means happened to fall. Measured on the far camera, x has
        # mean +6.66 m and z -3.88 m against a pooled +0.79: a shared offset parks those two
        # channels at +1.94 and -1.55 sigma instead of at zero, before the first convolution.
        g_mean = mean[:, idx].mean(dim=2, keepdim=True)                       # per channel
        g_std = std[:, idx].square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(eps)
        mean[:, idx] = g_mean.expand_as(mean[:, idx])
        std[:, idx] = g_std.expand_as(std[:, idx])
        report.append(
            f"{''.join(group)}: mean {[round(float(v), 4) for v in g_mean.reshape(-1)]}, "
            f"std(shared) {[round(float(v), 4) for v in g_std.reshape(-1)]}"
        )
    print(f"[fm-data] extero_norm_mode=per_channel: {n_sensors} sensor(s) x {channels} "
          f"channel(s) in {len(cfg.extero_channel_groups())} group(s) -- " + "; ".join(report))
    return Normalizer(
        mean.reshape(-1).contiguous(), std.reshape(-1).contiguous(), mask.reshape(-1).contiguous()
    )


def _pool_extero_statistics(
    normalizer: Normalizer, cfg: FlowMatchingConfig, n_sensors: int, sensor_dim: int
) -> Normalizer:
    """Collapse a per-element exteroception normalizer to one statistic per sensor or group.

    Motivation, in full, is on :attr:`FlowMatchingConfig.extero_norm_mode`; the short version
    is that a per-pixel affine is a spatially varying transform and a convolution's weight
    sharing assumes the transform is uniform, so EVERY mode here is uniform over pixels.

    The pooled value is broadcast back over the sensor's elements, so the returned normalizer
    has the SAME ``(T*K,)`` shape as the fitted one -- the checkpoint, the folded ONNX
    normalization and the deploy path are all shape-identical in every mode.

    The std is pooled as the RMS of the per-element stds rather than their mean, because what
    should be preserved is the total variance the sensor carries, not the average scale; the
    eps floor is re-applied afterwards so a near-constant sensor still cannot blow the
    normalization up.

    Args:
        normalizer: The freshly fitted per-element normalizer.
        cfg: Training config (``extero_norm_mode``, ``extero_norm_eps``, and for
            ``per_channel`` the channel names) are read.
        n_sensors: Number of sensor rows packed into the flat width.
        sensor_dim: Elements per sensor, CHANNELS INCLUDED (it is the row width).

    Returns:
        A normalizer with statistics constant over each sensor or channel group.

    Raises:
        ValueError: On an unrecognised ``extero_norm_mode``, or a channel layout that does not
            divide the row width.
    """
    cfg.check_extero_norm_mode()
    mode = str(cfg.extero_norm_mode)
    if mode == "metric":
        # A FIXED declared affine, discarding the fit entirely. The stored statistics keep
        # their (T*K,) shape so the checkpoint, the folded ONNX normalization and the deploy
        # path are untouched -- only the numbers change, and now they are the same numbers for
        # every recording of the same world rather than a property of this training set.
        offset = float(cfg.extero_norm_offset)
        scale = float(cfg.extero_norm_scale)
        width = n_sensors * sensor_dim
        print(f"[fm-data] extero_norm_mode=metric: DECLARED affine (x - {offset:g}) / "
              f"{scale:g}, no fitting. The network input is metres/{scale:g}, identical "
              "across recordings.")
        # Device from the normalizer being replaced, for the same reason: a fresh tensor
        # otherwise lands on the default device rather than where the data is.
        return Normalizer(
            torch.full((width,), offset, dtype=normalizer.mean.dtype,
                       device=normalizer.mean.device),
            torch.full((width,), scale, dtype=normalizer.std.dtype,
                       device=normalizer.std.device),
            normalizer.standardize_mask.clone(),
        )
    if mode == "contract":
        return _extero_contract_normalizer(normalizer, cfg, n_sensors, sensor_dim)
    if mode == "per_channel":
        return _pool_extero_per_channel(normalizer, cfg, n_sensors, sensor_dim)

    mean = normalizer.mean.reshape(n_sensors, sensor_dim)
    std = normalizer.std.reshape(n_sensors, sensor_dim)
    pooled_mean = mean.mean(dim=1, keepdim=True).expand_as(mean)
    pooled_std = std.square().mean(dim=1, keepdim=True).sqrt().clamp_min(
        float(cfg.extero_norm_eps)
    ).expand_as(std)
    print(f"[fm-data] extero_norm_mode=per_sensor: {n_sensors} sensor(s) standardized by one "
          f"(mean, std) each -- mean {[round(float(v), 4) for v in pooled_mean[:, 0]]}, "
          f"std {[round(float(v), 4) for v in pooled_std[:, 0]]}.")
    return Normalizer(
        pooled_mean.reshape(-1).contiguous(),
        pooled_std.reshape(-1).contiguous(),
        normalizer.standardize_mask,
    )


def resolve_block_selection(cfg: FlowMatchingConfig, meta: Any) -> tuple[str, ...]:
    """Settle which window blocks this model predicts, and disable what they cannot support.

    Order of authority: an explicit ``cfg.reference_blocks``, else the teacher that
    produced the recording (``meta.reference_blocks``), else all five -- which is what a
    recording predating the field necessarily was.

    The selection is written back onto ``cfg`` so the checkpoint records what was actually
    trained, and any consistency weight whose blocks are absent is zeroed here (with a
    printed line). Leaving such a weight live is the failure this whole plumbing exists to
    prevent: the term would index columns belonging to a different block and quietly
    regularize toward nonsense.

    Args:
        cfg: The training config; ``reference_blocks`` and the loss weights are updated.
        meta: Recording layout metadata (only ``reference_blocks`` is read).

    Returns:
        The selection in canonical packing order.
    """
    recorded = list(getattr(meta, "reference_blocks", None) or []) or None
    requested = list(cfg.reference_blocks or []) or None
    selected = resolve_reference_blocks(requested if requested is not None else recorded)
    if requested is not None and recorded is not None and tuple(recorded) != selected:
        # Legal -- the FM can be trained for a teacher other than the recorded one -- but
        # it is the setup where a width mismatch bites at play time, so say it out loud.
        print(f"[fm-data] reference_blocks={list(selected)} OVERRIDES the recording's "
              f"teacher, which observed {recorded}. The generator will only drive a "
              "policy whose reference_window packs the requested blocks.")
    elif recorded is None and requested is None:
        # Only on the FIRST resolution -- the val/eval datasets re-enter this with the
        # selection already written back, and repeating the note would read as a fresh
        # finding about a different recording.
        print(f"[fm-data] the recording states no reference_blocks (it predates the "
              f"field, so its teacher observed all five); using {list(selected)}.")
    cfg.reference_blocks = list(selected)

    for weight_name, needed in _LOSS_TERM_BLOCKS.items():
        missing = [b for b in needed if b not in selected]
        if missing and float(getattr(cfg, weight_name)) > 0.0:
            print(f"[fm-data] {weight_name}={getattr(cfg, weight_name)} -> 0.0: it ties "
                  f"{list(needed)} and this window has no {missing}.")
            setattr(cfg, weight_name, 0.0)
    return selected


def resolve_term_slice(term_names: list[str], term_dims: list[int], target: str) -> tuple[int, int]:
    """Return ``(start, end)`` of a named term within a concatenated vector.

    Args:
        term_names: Ordered term names.
        term_dims: Per-term widths (same order).
        target: The term to locate.

    Raises:
        KeyError: If ``target`` is not present.
    """
    offset = 0
    for name, dim in zip(term_names, term_dims):
        if name == target:
            return offset, offset + dim
        offset += dim
    raise KeyError(f"term {target!r} not found in {term_names}")


def _stack_frame_positions(split_fn, frame_names: list[str]) -> torch.Tensor:
    """Gather extra-frame world POSITIONS as ``(..., K, 3)`` for :func:`world_to_window`.

    The recording carries each frame's full ``pos/quat/lin_vel/ang_vel``, but the window's
    ee block is position-only (see :mod:`~robot_rl.flow_matching.window`), so the other
    three are simply not read -- which is why widening the ee block back out would need no
    re-record.

    Args:
        split_fn: ``name -> (..., dim)`` slicer over the windowed rows.
        frame_names: Ordered extra body-frame names; ``[]`` gives a ``(..., 0, 3)`` array.

    Returns:
        ``frame_pos_w`` ``(..., K, 3)``.
    """
    if not frame_names:
        probe = split_fn("root_pos_w")
        return probe.new_zeros(probe.shape[:-1] + (0, 3))
    return torch.stack([split_fn(f"{f}:pos_w") for f in frame_names], dim=-2)


#: Raw ``state_term_names`` a MEASURED history row takes from the APPLIED REFERENCE instead
#: of from the robot, because no robot can measure them.
#:
#: The window-level names are ``window.MEASURED_HISTORY_FALLBACK`` -- ``base_pos`` and
#: ``base_lin_vel``. These are the raw terms ``world_to_window`` reads each of them out of,
#: so splicing here is equivalent to splicing the built window's spans and costs one pass
#: instead of two. Kept as a separate tuple rather than derived, because the mapping from a
#: window half to a state term is not something either module can compute: it is the
#: definition of the block.
MEASURED_HISTORY_FALLBACK_TERMS: tuple[str, ...] = ("root_pos_w", "root_link_lin_vel_w")


def _check_measured_lag_alignment(dls: torch.Tensor, hs_sel: torch.Tensor, H: int,
                                  s_slc: dict[str, tuple[int, int]]) -> None:
    """Assert ``depth_lag_state`` lag ``k`` really is ``history_state`` row ``H-1-k``.

    The measured history build rests entirely on that identity, and it is an identity
    between two arrays the recorder fills in different places -- so it is checked on the
    data rather than trusted from the code. Two halves, and BOTH are needed:

    * lag 0 must equal row ``H-1`` in FULL. That row is the snap, which the recorder leaves
      measured, so this pins the lag indexing with no substitution in the way.
    * for lag ``k >= 1``, the PRESERVED columns (everything past ``root_quat_w``) must match
      row ``H-1-k`` exactly, while the pose columns must DIFFER -- the latter being the plan
      anchor the recorder substituted. Without the "must differ" half the check would pass
      on a recording that never substituted anything, where these rows would not be plan-
      anchored and the whole history frame would be wrong.

    Args:
        dls: ``(Nw, L, node)`` ``depth_lag_state`` for the selected windows.
        hs_sel: ``(Nw, H, node)`` ``history_state`` for the same windows.
        H: History-ring length.
        s_slc: Term-name -> ``(start, stop)`` for one state row.

    Raises:
        ValueError: If either half fails.
    """
    pose_end = s_slc["root_quat_w"][1]
    if not torch.equal(dls[:, 0, :], hs_sel[:, H - 1, :]):
        d = float((dls[:, 0, :] - hs_sel[:, H - 1, :]).abs().max())
        raise ValueError(
            f"depth_lag_state lag 0 != history_state row {H - 1} (max |diff| {d:.3e}). Those "
            "are both the measured state at the snap, so the lag index is not a history-row "
            "index in this recording and the measured history would be built from the wrong "
            "steps.")
    for k in range(1, int(dls.shape[1])):
        row = H - 1 - k
        if row < 0:
            break
        keep = (dls[:, k, pose_end:] - hs_sel[:, row, pose_end:]).abs().max()
        if float(keep) != 0.0:
            raise ValueError(
                f"depth_lag_state lag {k} and history_state row {row} disagree on the "
                f"PRESERVED columns (max |diff| {float(keep):.3e}); they should be the same "
                "measured row. The lag/row alignment does not hold in this recording.")
        pose = (dls[:, k, :pose_end] - hs_sel[:, row, :pose_end]).abs().max()
        if float(pose) == 0.0:
            raise ValueError(
                f"depth_lag_state lag {k} and history_state row {row} AGREE on the root "
                "pose, but the recorder is supposed to have substituted the plan anchor "
                "there. This recording's history rows are therefore not plan-anchored, so "
                "the frame the measured rows would be observed in is not the trained one.")


def _rows_to_window(
    split_fn,
    anchor_pos: torch.Tensor,
    anchor_quat: torch.Tensor,
    frame_names: list[str],
    blocks: Sequence[str],
) -> torch.Tensor:
    """Turn recorded world-frame reference rows into window nodes.

    The one place the recorded column layout is mapped onto the window representation, used
    for BOTH blocks the model sees: the ``F``-node future (one anchor for all nodes) and the
    ``N``-node autoregressive history (a per-row anchor). The only difference between the
    two calls is the shape of the anchor, which :func:`world_to_window` broadcasts.

    Args:
        split_fn: ``name -> (..., dim)`` slicer over the recorded rows, in the
            ``meta.state_term_names`` layout.
        anchor_pos: ``(..., 3)`` measured robot base position each node is expressed
            relative to.
        anchor_quat: ``(..., 4)`` measured robot base orientation (xyzw).
        frame_names: Ordered extra body-frame names.
        blocks: The block selection to pack.

    Returns:
        ``(..., D)`` window nodes.
    """
    # The recorded columns for a dropped block are read but never used, which costs one
    # slice of an already-materialised array -- ``world_to_window`` does not compute the
    # block itself. Reading them unconditionally keeps this mapping in one shape.
    return world_to_window(
        anchor_pos, anchor_quat,
        split_fn("root_pos_w"), split_fn("root_quat_w"),
        split_fn("root_link_lin_vel_w"), split_fn("root_link_ang_vel_w"),
        split_fn("joint_pos"), split_fn("joint_vel"),
        _stack_frame_positions(split_fn, frame_names),
        blocks=blocks,
    )


def snap_windows(
    episode_id: np.ndarray,
    step_in_episode: np.ndarray,
    done: np.ndarray,
    reanchor_active: np.ndarray,
    num_history: int,
    stride: int = 1,
) -> list[np.ndarray]:
    """History-row windows ending at each snap (re-anchor) event.

    Returns one int array of length ``num_history`` per usable snap: the ``N``
    consecutive rows of one episode ending at a row where ``reanchor_active`` is set
    (the FM's deploy re-inference points). A snap is usable iff it has ``N-1`` prior
    consecutive steps in the same episode with no ``done`` before the snap.

    Args:
        episode_id / step_in_episode / done / reanchor_active: Per-pair flat arrays.
        num_history: ``N`` history steps (the snap is the last of the ``N``).
        stride: Keep every ``stride``-th usable snap (per episode).

    Returns:
        List of int arrays, each of length ``num_history`` (last entry = the snap row).
    """
    order = np.lexsort((step_in_episode, episode_id))
    ep_sorted = episode_id[order]
    step_sorted = step_in_episode[order]
    done_sorted = done[order]
    snap_sorted = reanchor_active[order].astype(bool)

    out: list[np.ndarray] = []
    n = len(order)
    start = 0
    while start < n:
        end = start
        while end + 1 < n and ep_sorted[end + 1] == ep_sorted[start]:
            end += 1
        ep_rows = order[start : end + 1]
        ep_steps = step_sorted[start : end + 1]
        ep_done = done_sorted[start : end + 1]
        ep_snap = snap_sorted[start : end + 1]
        length = len(ep_rows)
        kept = 0
        for j in range(num_history - 1, length):
            if not ep_snap[j]:
                continue
            sl = slice(j - (num_history - 1), j + 1)
            if not np.all(np.diff(ep_steps[sl]) == 1):
                continue
            if np.any(ep_done[sl][:-1] != 0):  # no reset within the history window
                continue
            if kept % stride == 0:
                out.append(ep_rows[sl].copy())
            kept += 1
        start = end + 1
    return out


class FutureStateDataset(Dataset):
    """Robot-state-conditioned FM items: (history, height, cmd) -> ego-anchored clip future."""

    def __init__(
        self,
        data: dict[str, np.ndarray],
        meta: Any,
        cfg: FlowMatchingConfig,
        windows: list[np.ndarray],
        state_normalizer: Normalizer | None = None,
        cmd_normalizer: Normalizer | None = None,
        extero_normalizer: Normalizer | None = None,
        add_noise: bool = False,
    ):
        """Build and (optionally) fit normalizers over the provided snap windows.

        Args:
            data: Recorded arrays (``state``, ``ref_future``, ``cmd_future``).
            meta: Layout metadata (``state_term_names/dims``, ``state_frame_names``,
                ``ref_future_term_names/dims``, ``ref_future_len``, ``reference_blocks``).
            cfg: Config; ``reference_blocks``, ``state_dim``, ``cmd_dim``, ``history_dim``
                and any unsupportable loss weight are written back here.
            windows: History-row windows (length ``N``) from :func:`snap_windows`.
            state_normalizer / cmd_normalizer / extero_normalizer: reused on the val
                split; fit on the train split when ``None``.
            add_noise: Add resampled observation noise to the history tokens in
                ``__getitem__`` (train split only); gated by ``cfg.obs_noise``.
        """
        self.cfg = cfg
        f = cfg.future_len
        n = cfg.num_history_states
        self.N = n
        if n < 1:
            raise ValueError(f"num_history_states={n} must be >= 1.")
        if cfg.target_source != "ego_anchored_clip_future":
            raise NotImplementedError(
                f"FutureStateDataset on this branch supports only "
                f"target_source='ego_anchored_clip_future' (got {cfg.target_source!r}); "
                "the robot-state-conditioned FM trains on recorded ego-anchored clip futures."
            )
        if data.get("ref_future") is None:
            raise ValueError(
                "target_source='ego_anchored_clip_future' requires a 'ref_future' array; "
                "re-record with scripts/rsl_rl/record_data.py (ego_anchor_fraction=1.0)."
            )
        # The history rows are only ever read as ANCHOR poses, so a recording whose rows are
        # anchored per step has the same shape as one anchored on the plan and loads clean --
        # while meaning something the generator is never served. At deploy the history IS a
        # slice of the previous plan, in that plan's frame, with no transformation applied.
        anchor = str(getattr(meta, "history_anchor", "") or "")
        if anchor != "plan":
            raise ValueError(
                f"history_anchor={anchor or '<absent>'!r}, but the generator conditions on a "
                "PLAN-frame history (its own previous prediction, untransformed). A per-step "
                "anchored recording differences each reference against its own contemporaneous "
                "robot pose, which cancels the reference's motion: measured, its per-lag means "
                "are FLAT (0.0196/0.0197/0.0199/0.0199 m) and the rows sit 6.9 mm apart, "
                "against 0.1231->0.1657 m and 45 mm plan-anchored. Re-record with "
                "scripts/rsl_rl/record_data.py --num_windows (an absent field means the "
                "recording predates the change, so it is per-step)."
            )
        # NOTE: there is deliberately no train-time exteroception noise, and therefore no gate
        # here reconciling it with the recording. The sensor model lives in the RL env (see
        # FlowMatchingConfig, where noise_depth / noise_height_scan used to be): the cameras'
        # DR is already baked into what was recorded, and the privileged scan is exact on
        # purpose. ``meta.extero_corrupted`` / ``meta.extero_noise_free`` survive as statements
        # about the recording; nothing on this side reacts to them any more.
        cfg.check_extero_source()
        cfg.check_extero_norm_mode()
        if data.get(cfg.extero_source) is None:
            hint = (
                "a height_scanner sensor active (e.g. --task G1-teacher-clf-record-play)"
                if cfg.extero_source == "height_scan"
                else "the depth cameras active (--task G1-teacher-clf-terrain-depth-play); "
                     "note record_data.py stores 'depth' only when the task has them"
            )
            raise ValueError(
                f"extero_source={cfg.extero_source!r} requires a '{cfg.extero_source}' array "
                f"in the recording; re-record with {hint}. "
                f"Present keys: {sorted(k for k in data if data.get(k) is not None)}"
            )
        if len(windows) == 0:
            raise ValueError("No valid snap windows; check reanchor_active vs num_history_states.")

        # -- resolve state layout (the node layout of ref_future == state layout) --
        s_names, s_dims = list(meta.state_term_names), list(meta.state_term_dims)
        s_slc = {name: resolve_term_slice(s_names, s_dims, name) for name in s_names}
        nj = s_dims[s_names.index("joint_pos")]
        self.num_joints = nj
        recorded_frames = list(getattr(meta, "state_frame_names", []) or [])
        self.target_frame_names = list(recorded_frames)
        self.num_extra_frames = len(self.target_frame_names)
        # Which blocks a node packs -- the teacher's selection, resolved from the recording
        # unless the config names one. Everything downstream (D, the slices, the noise
        # table, the losses, the export) derives from this one call.
        self.blocks = resolve_block_selection(cfg, meta)
        self.state_dim = window_dim(nj, self.num_extra_frames, self.blocks)
        self.block_slices = window_block_slices(nj, self.num_extra_frames, self.blocks)
        cfg.state_dim = self.state_dim
        cfg.history_dim = self.state_dim

        rf = data["ref_future"]
        if rf.shape[1] < f:
            raise ValueError(
                f"ref_future has F={rf.shape[1]} nodes but cfg.future_len={f}; the recording "
                "is SHORTER than the horizon being trained. Re-record with "
                f"--ref_future_len >= {f}."
            )
        if rf.shape[1] > f:
            # Truncating is safe and deliberate: node k is the reference k steps after the
            # snap, so the first f of a longer recording ARE the f-node horizon. Recording
            # with headroom therefore lets one file train every F <= it -- which matters
            # because the horizon is set by the tracker's re-plan period
            # (F >= max_window_offset + fm_update_every) and that is a thing you sweep.
            print(f"[fm-data] ref_future has {rf.shape[1]} nodes; truncating to "
                  f"future_len={f}.")
            rf = rf[:, :f]

        # -- command (strided future velocity profile from cmd_future) --
        self.cmd_array = cfg.cmd_array
        cmd_future = data.get("cmd_future")
        if self.cmd_array is not None and cmd_future is None:
            raise ValueError("cmd_array set but the recording has no 'cmd_future' array.")
        cmd_dim = int(cmd_future.shape[-1]) if (self.cmd_array is not None and cmd_future is not None) else 0
        self.cmd_dim = cmd_dim
        cfg.cmd_dim = cmd_dim
        self.cmd_steps = list(range(0, f, cfg.cmd_stride))
        self.num_cmd_tokens = len(self.cmd_steps) if cmd_dim > 0 else 0

        # Two input formats produce the SAME raw tensors; everything downstream
        # (world_to_window, normalizers, striding, noise) is shared.
        #
        # BOTH blocks the model sees are the APPLIED REFERENCE -- the history is the
        # reference this generator emitted at the previous steps, the target is the
        # reference over the future window -- and the measured robot appears only as the
        # ANCHOR each block is expressed relative to. That is what makes the model
        # autoregressive over its own output at deploy: it pushes back the node it just
        # emitted and needs no state estimate.
        #
        # The two anchors differ, and the difference is the whole design:
        #   * TARGET: one anchor, the robot at the snap, shared by all F nodes -- so the
        #     future is one rigid plan in the robot's current frame, exactly what the
        #     policy's ``reference_window`` observation is.
        #   * HISTORY: a PER-ROW anchor, the robot as it was at that row's own step -- so
        #     each history token is literally the node this model emitted then. Re-anchoring
        #     them onto the robot NOW would instead need the egomotion since, which is the
        #     odometry the deploy path no longer carries.
        #
        # Row indexing: the recorded rings are pushed at one instant per control step
        # (record_data.collect_windows), and ``command_manager.compute()`` runs before
        # ``observation_manager.compute()`` inside ``env.step``, so ``history_ref[H-1]``,
        # ``history_state[H-1]`` and ``ref_future[0]`` are all the SAME step. Node 0 of the
        # target is therefore the CURRENT reference, and the N history rows are the N
        # BEFORE it: indices ``H-1-n .. H-2``. (Guard 3 below pins this.)
        #
        #   per-step (.npz): ``windows`` = length-(N+1) row-index arrays; ``[:-1]`` are the
        #     history rows, ``[-1]`` the snap.
        #   windowed (.h5): ``windows`` = a 1-D array of window indices into the pre-formed
        #     arrays; ``history_state``/``history_ref``/``ref_future`` are already snap-formed.
        windowed = "history_state" in data
        self.window_sel: np.ndarray | None = None
        root_pos_slc, root_quat_slc = s_slc["root_pos_w"], s_slc["root_quat_w"]
        _p = lambda t: t[..., root_pos_slc[0]:root_pos_slc[1]]
        _q = lambda t: t[..., root_quat_slc[0]:root_quat_slc[1]]
        rf_t = torch.from_numpy(np.ascontiguousarray(rf)).float()  # (P|W, F, node)
        if windowed:
            sel = np.asarray(windows, dtype=np.int64)  # (Nw,) window selector
            # Kept so a caller can map dataset item i back to recorded window sel[i] and join
            # per-window labels (e.g. the transition mask) onto val results. Only the windowed
            # path has a 1-D selector; the per-step path leaves this None.
            self.window_sel = sel
            n_win = int(sel.shape[0])
            hs_state = torch.from_numpy(np.ascontiguousarray(data["history_state"])).float()
            H = int(hs_state.shape[1])
            if n + 1 > H:
                raise ValueError(
                    f"num_history_states={n} needs history_len >= {n + 1} (N context rows "
                    f"plus the snap row they precede), but the recording has {H}; "
                    f"re-record with --history_len >= {n + 1}."
                )
            if data.get("history_ref") is None:
                raise ValueError(
                    "The autoregressive history is the APPLIED REFERENCE, which lives in the "
                    "'history_ref' array; this recording has none. Re-record with the current "
                    "record_data.py (it writes history_ref unconditionally)."
                )
            hr = torch.from_numpy(np.ascontiguousarray(data["history_ref"])).float()
            # The recorder writes the PLAN ANCHOR into rows 0..H-2, so these n rows all carry
            # the same pose -- the robot at the last re-plan. Row H-1 stays measured.
            hw_meas = hs_state[sel][:, H - 1 - n:H - 1, :]   # (Nw, n, node) anchors, lags n..1
            hw_ref = hr[sel][:, H - 1 - n:H - 1, :]          # (Nw, n, node) applied reference
            snap_meas = hs_state[sel][:, H - 1, :]           # (Nw, node) robot at the snap
            snap_ref = hr[sel][:, H - 1, :]                  # (Nw, node) reference at the snap
            tw = rf_t[sel]                                   # (Nw, F, node)
            # -- history_source="measured": the ROBOT'S OWN rows ------------------------
            # ``history_state`` cannot supply them: the recorder substitutes the plan anchor
            # into rows 0..H-2, cols 0:7, so the measured root POSE of every non-snap row is
            # gone from there. ``depth_lag_state`` is the same ``_assemble_state`` cloned
            # BEFORE that substitution, at lags 0..depth_lag_steps, and lag k is row H-1-k --
            # verified bit-exact on the shipped recordings (lag 0 vs row H-1 is 0.000e+00,
            # and lags 1..3 match rows 6/5/4 on every preserved column while differing on
            # 0:7, which is the substitution).
            hw_src = hw_ref
            if cfg.history_source == "measured":
                if data.get("depth_lag_state") is None:
                    raise ValueError(
                        "history_source='measured' needs the un-overwritten measured state, "
                        "which lives in 'depth_lag_state'; this recording has none (it was "
                        "written without depth). Re-record, or train with "
                        "history_source='reference'.")
                dls = torch.from_numpy(
                    np.ascontiguousarray(data["depth_lag_state"])).float()[sel]
                cfg.check_history_source(available_lags=int(dls.shape[1]) - 1)
                # Lags n..1, oldest first, matching the history block's row order.
                hw_src = dls[:, torch.arange(n, 0, -1), :].clone()
                _check_measured_lag_alignment(dls, hs_state[sel], H, s_slc)
                # THE FALLBACK, spliced in the RAW row rather than in the built window.
                # ``world_to_window`` reads the base position out of ``root_pos_w`` and the
                # linear velocity out of ``root_link_lin_vel_w``, so substituting those two
                # terms from the reference row yields exactly the node the deploy runtime
                # serves: there the same two blocks come from the SERVED node, whose
                # base_pose[0:3] is R_planyawT (p_ref - p_plan) -- the identical quantity,
                # since a served node IS the applied reference. Splicing here also keeps one
                # ``_rows_to_window`` call instead of building the window twice and
                # overwriting half of it.
                for term in MEASURED_HISTORY_FALLBACK_TERMS:
                    a, b = s_slc[term]
                    hw_src[..., a:b] = hw_ref[..., a:b]
        else:
            idx_all = np.stack(windows).astype(np.int64)     # (Nw, N+1); last = snap
            idx_hist, snap_rows = idx_all[:, :-1], idx_all[:, -1]
            n_win = idx_all.shape[0]
            state_t = torch.from_numpy(np.ascontiguousarray(data["state"])).float()
            if data.get("ref_state") is None:
                raise ValueError(
                    "The autoregressive history is the APPLIED REFERENCE, which lives in the "
                    "per-step 'ref_state' array; this recording has none. Re-record with the "
                    "current record_data.py."
                )
            ref_t = torch.from_numpy(np.ascontiguousarray(data["ref_state"])).float()
            if cfg.history_source == "measured":
                raise ValueError(
                    "history_source='measured' is not implemented for the legacy per-step "
                    "(.npz) format. It would be easy and pointless: that path anchors every "
                    "row on its OWN step rather than on the plan, so a measured row's base "
                    "block would be ~zero by construction and carry nothing. Use a windowed "
                    "recording.")
            hw_meas = state_t[idx_hist]                      # (Nw, N, node)
            hw_ref = ref_t[idx_hist]                         # (Nw, N, node)
            hw_src = hw_ref
            snap_meas = state_t[snap_rows]                   # (Nw, node)
            snap_ref = ref_t[snap_rows]
            tw = rf_t[snap_rows]                             # (Nw, F, node)

        p_anchor, q_anchor = _p(snap_meas), _q(snap_meas)    # (Nw, 3) / (Nw, 4)

        # -- history: lags n..1, all in the PLAN's frame (windowed; the legacy per-step path
        # still anchors each row on its own step). ``hw_src`` is the applied reference under
        # ``history_source="reference"`` and the robot's own state (bar the two unmeasurable
        # blocks) under ``"measured"``; the ANCHOR is the plan pose either way, which is what
        # every history row is observed in.
        history = _rows_to_window(
            lambda name: hw_src[..., s_slc[name][0]:s_slc[name][1]],
            _p(hw_meas), _q(hw_meas), self.target_frame_names, self.blocks,
        ).contiguous()                                        # (Nw, n, D)

        # -- target: the F-node ego-anchored clip future, all in the snap robot's frame --
        target = _rows_to_window(
            lambda name: tw[..., s_slc[name][0]:s_slc[name][1]],
            p_anchor[:, None, :], q_anchor[:, None, :], self.target_frame_names, self.blocks,
        ).contiguous()                                        # (Nw, F, D)

        # Under ``measured`` the newest history row is adjacent to the MEASURED snap, not to
        # the reference target -- see guard 4. Frame-invariant block, so the raw joints are
        # the window's joints with no transform.
        _adj = (snap_meas[:, s_slc["joint_pos"][0]:s_slc["joint_pos"][1]]
                if cfg.history_source == "measured" else None)
        self._check_window_invariants(target, history, snap_ref, tw[:, 0], s_slc, f, _adj)

        # Exteroception readings at the snap row. The recording carries either
        # ``height_scan`` (P|W, K) -- one sensor -- or ``depth`` (P|W, T, K) with one row
        # per camera, so the sensor count and per-sensor width both come from the array's
        # shape rather than being configured.
        raw_extero = data[cfg.extero_source]
        n_sensors = int(raw_extero.shape[1]) if raw_extero.ndim == 3 else 1
        sensor_dim = int(raw_extero.shape[-1])
        # SELECT before sanitizing, and sanitize IN PLACE. The naive order (read whole ->
        # nan_to_num -> clamp -> fancy-index) holds up to four simultaneous full-size copies of
        # the array, which at a dense scan is tens of GB before training starts. Here the fancy
        # index is the only copy and the clamp is applied to it.
        rows = sel if windowed else snap_rows
        hs = torch.from_numpy(
            np.ascontiguousarray(raw_extero[rows.numpy() if torch.is_tensor(rows) else rows])
        ).float()
        hs = hs.reshape(hs.shape[0], n_sensors * sensor_dim)
        # CAMERA LATENCY. Swap in a frame captured a few control steps earlier, per window,
        # drawn from cfg.depth_latency_probs (index = lag in control steps). ``pose_meas``
        # follows it: the xyz channels below are unprojected from the camera pose, so a lagged
        # frame has to be unprojected with the pose it was CAPTURED with or every pixel is
        # displaced by the egomotion -- the registration error the channels exist to remove.
        # It is then left in that capture-time frame: re-expressing it at the re-plan would
        # need the root displacement over the lag, and this robot has no base-velocity
        # estimate to integrate.
        #
        # Drawn ONCE per window at build, not per epoch -- the extero array is materialised
        # here. With a recording of this size every lag is thoroughly represented across
        # windows; what it does not do is show the same window at two different lags.
        pose_meas = snap_meas
        self.depth_latency_probs = None
        if cfg.depth_latency_probs is not None:
            self.depth_latency_probs = _select_depth_lag(
                cfg, data, rows, hs, n_sensors, sensor_dim, snap_meas, self)
            hs, pose_meas = self._lagged_hs, self._lagged_pose
        # Sanitize before the normalizer fit so a non-finite entry cannot poison the
        # per-dimension mean and yield a NaN loss (matches the deploy token builder). The band
        # is the config's, so a run can widen it past the per-source default -- a 1 m band
        # destroys ~11% of a 3 m scan on this terrain, concentrated downrange.
        band = cfg.resolve_extero_clip()
        # STAMPED back, so the artifact states the band it trained under instead of leaving a
        # None for two different fallback tables to resolve. They no longer agree:
        # ``config.EXTERO_CLIP`` says 5.575 for depth while ``deploy.depth_obs.EXTERO_CLIP_NP``
        # says 2.0, and each is right for the artifacts of its own era -- so a None reaching
        # deploy trained at 5.575 and ran at 2.0, saturating range at 4.75 m and putting a
        # false wall in the xyz cloud. Same contract ``extero_encoder`` and ``extero_channels``
        # already follow: resolve once, write it down, and let every consumer read a number.
        cfg.extero_clip = float(band)
        # MEASURE before clamping, and report. The band is the single most consequential
        # unmeasured constant on this path: at the scan's historical 1.0 m default a 3 m window
        # over stairs loses ~11% of its pixels, concentrated downrange where the planning
        # happens, and nothing warned. A run's log should state what its band cost it.
        finite = torch.isfinite(hs)
        n_total = hs.numel()
        n_miss = int((~finite).sum())
        vals = hs[finite]
        n_sat = int((vals.abs() > band).sum())
        if n_total:
            q = torch.tensor([0.001, 0.5, 0.999])
            # SUBSAMPLE for the quantiles: torch.quantile refuses an input past ~16M elements
            # ("input tensor is too large"), and a real extero array is far past that -- 282M
            # for the depth pair at 180k windows, 2.6B for the dense scan. A uniform sample of
            # a few million puts every percentile here well inside its own rounding.
            _cap = 4_000_000
            _s = (vals if vals.numel() <= _cap
                  else vals[torch.randint(vals.numel(), (_cap,), device=vals.device)])
            lo, mid, hi = (torch.quantile(_s.float(), q).tolist()
                           if vals.numel() else (float("nan"),) * 3)
            # DEPTH's invalid pixels are FINITE, so ``n_miss`` counts ~0 of them and they are
            # otherwise tallied as ordinary in-band data -- in the very number
            # ``--extero_clip``'s help says to pick the band from. Measured on the shipped
            # pair: 15% of the near camera's pixels and 40% of the far camera's carry no
            # reading, against the 2% a sentinel-equality test finds (that 2% is the noiser's
            # per-frame dropout, which lands exactly on the sentinel; a geometric miss is
            # blurred off it). Report them as their own class.
            invalid = ""
            if cfg.extero_source == "depth" and meta.depth_camera_spec:
                from robot_rl.tasks.manager_based.robot_rl.mdp.observations.depth import (
                    invalid_depth_mask,
                )
                per_cam = hs.reshape(hs.shape[0], n_sensors, sensor_dim)
                frac = [float(invalid_depth_mask(cam, per_cam[:, i]).float().mean())
                        for i, cam in enumerate(meta.depth_camera_spec)]
                invalid = (" | no reading (by range): "
                           + ", ".join(f"{c.get('obs_term', i)} {100.0 * v:.1f}%"
                                       for i, (c, v) in enumerate(
                                           zip(meta.depth_camera_spec, frac))))
            print(f"[fm-data] extero clip band +-{band:g}: "
                  f"{100.0 * n_sat / n_total:.2f}% of entries saturated, "
                  f"{100.0 * n_miss / n_total:.2f}% missed rays (-> +{band:g}), "
                  f"finite range [{float(vals.min()) if vals.numel() else float('nan'):.3f}, "
                  f"{float(vals.max()) if vals.numel() else float('nan'):.3f}] "
                  f"(p0.1 {lo:.3f} / p50 {mid:.3f} / p99.9 {hi:.3f})." + invalid)
        torch.nan_to_num_(hs, nan=0.0, posinf=band, neginf=band)
        hs.clamp_(-band, band)
        # Contract of the sensor behind those readings, carried from the recording so the
        # trained checkpoint (and the ONNX exported from it) states what it conditions on --
        # ``depth_camera_spec`` for depth, ``height_scan_spec`` for the scan. Resolved BEFORE
        # the channel widening below, which needs the intrinsics and the mount out of it.
        self.extero_spec = self._resolve_extero_spec(cfg, meta, n_sensors, sensor_dim)
        # Widen one pixel from a scalar to a vector, when the run asks for it. The extra
        # planes are DERIVED here rather than recorded: the recording already carries the root
        # orientation and the joint angles, and the mount is nominal, so an existing .h5 needs
        # no re-recording to train a 4-channel model. Done AFTER the clamp on purpose -- see
        # _depth_xyz_rows.
        # Resolved from the source when the run did not say (5 for depth, 1 for a scan), and
        # written BACK so the checkpoint records a number rather than "whatever the default
        # meant on the day" -- the same contract ``extero_encoder="auto"`` follows.
        row_channels = cfg.resolve_extero_channels()
        cfg.extero_channels = row_channels
        cfg.extero_channel_names = ["d", "x", "y", "z", "v"][:row_channels]
        if row_channels > 1:
            # ``pose_meas`` is ``snap_meas`` unless camera latency swapped in an older
            # frame, in which case it is the state that frame was captured with.
            poses = self._camera_pose_yaw_frame(
                self.extero_spec, _q(pose_meas), pose_meas[:, s_slc["joint_pos"][0]:
                                                           s_slc["joint_pos"][1]],
                [str(j) for j in meta.state_joint_names], cfg.fk_model_path,
            )
            hs = self._depth_xyz_rows(hs, self.extero_spec, poses, n_sensors, sensor_dim,
                                      channels=row_channels)
            if cfg.extero_norm_mode == "contract":
                # Derived from the contract, BEFORE the normalizer is built and independently
                # of anything in `hs`. Reported against the data's own centroid because the
                # declared offset centres the nominal FLAT-GROUND view: on stairs and drop-offs
                # the real centroid sits some way off, and that residual is the honest price of
                # a constant that does not move between recordings. Visible, not assumed.
                # Derive ALWAYS, even when a value is already set, so the contract's own
                # answer is on screen next to whatever is being used. A declared offset is
                # legitimate (--extero_xyz_offset exists), but one carried over from a run
                # whose CAMERA has since moved is not, and the two are indistinguishable in
                # the config: moving the far camera 35 -> 15 deg walks its boresight ground
                # intersection from 1.8 m to about 4.3 m while the stored number sits still.
                derived, envelope = _resolve_xyz_contract(
                    cfg, self.extero_spec, [str(j) for j in meta.state_joint_names]
                )
                if cfg.extero_xyz_offset is None:
                    cfg.extero_xyz_offset, cfg.extero_xyz_envelope = derived, envelope
                else:
                    # The envelope is derived from the camera CONE, not from the offset, so a
                    # declared offset does not imply a declared envelope -- and it used to be
                    # assigned only on the branch above. Declaring an offset therefore left it
                    # None and the ``float(...)`` below raised a bare TypeError, immediately
                    # after a warning saying a declared offset is fine. There is no
                    # ``--extero_xyz_envelope`` flag to set it by hand either, so the
                    # documented escape hatch from a stale offset could not be used at all.
                    if cfg.extero_xyz_envelope is None:
                        cfg.extero_xyz_envelope = envelope
                    drift = max(
                        (sum((a - b) ** 2 for a, b in zip(d, c)) ** 0.5
                         for d, c in zip(derived, cfg.extero_xyz_offset)), default=0.0)
                    if drift > _XYZ_OFFSET_DRIFT_WARN_M:
                        print(f"[fm-data] WARNING: the declared extero_xyz_offset is "
                              f"{drift:.2f} m from what this recording's camera contract "
                              f"derives ({derived}). A declared offset is fine; one left over "
                              "from a different camera geometry is not, and nothing else "
                              "distinguishes them. Clear it to re-derive.")
                view = hs.reshape(hs.shape[0], n_sensors, row_channels, -1)
                for s, cam in enumerate(self.extero_spec):
                    got = view[:, s, 1:4].mean(dim=(0, 2))
                    want = torch.tensor(cfg.extero_xyz_offset[s], dtype=got.dtype)
                    print(f"[fm-data]   {cam.get('sensor')}: declared offset "
                          f"{cfg.extero_xyz_offset[s]} m vs measured centroid "
                          f"{[round(float(v), 2) for v in got]} m -- residual "
                          f"{float((got - want).norm()):.2f} m "
                          f"({float((got - want).norm()) / float(cfg.extero_xyz_scale):.2f} of "
                          "the declared scale)")
                reach = float((view[:, :, 1:4].abs().max()))
                bound = float(cfg.extero_xyz_envelope) * float(cfg.extero_xyz_scale)
                if reach > bound:
                    raise ValueError(
                        f"an xyz channel reaches {reach:.2f} m but the declared clamp allows "
                        f"{bound:.2f} m. The clamp is sized to the max-range cone and must not "
                        "bind on real geometry -- the camera contract and the data disagree."
                    )
            sensor_dim = row_channels * sensor_dim      # the ROW width the model receives
            print(f"[fm-data] extero_channels={row_channels}: each pixel carries "
                  f"{cfg.extero_channel_names}, so a sensor row is {sensor_dim} wide "
                  f"({row_channels} x {sensor_dim // row_channels} px), channel-major.")
        self.extero = hs                                            # (Nw, T*C*K)
        self.extero_clip = band
        self.extero_sensor_dim = sensor_dim
        self.extero_sensor_count = n_sensors
        cfg.extero_sensor_dim = sensor_dim
        cfg.extero_sensor_count = n_sensors
        # The per-sensor IMAGE SHAPE, derived from that contract rather than configured --
        # the flat array cannot state it, and a convolutional encoder cannot fold without
        # it. Written onto cfg so the checkpoint records the architecture it trained with,
        # and re-verified wherever it is read again (``online.load_fm_context``).
        # Against the PIXEL count, not the row width: with channels the row is a stack of
        # images, and comparing the whole width against one image's shape would reject the
        # very geometry it exists to describe.
        self.extero_image_hw = self._resolve_extero_image_hw(
            self.extero_spec, sensor_dim // row_channels
        )
        cfg.extero_image_hw = ([list(hw) for hw in self.extero_image_hw]
                               if self.extero_image_hw else None)
        # And the METRIC extents, for a grid whose pixels are ground points rather than rays.
        # This is what lets the encoder give the convolution each pixel's (x, y) offset from
        # the robot in metres -- the positional information a translation-equivariant kernel
        # structurally cannot have, and the whole quantity the generator fails to register.
        # Derived from the same spec; ``None`` for depth, whose pixels have no planar coordinate.
        cfg.extero_coord_extent_m = _resolve_extero_coord_extent(self.extero_spec)
        if cfg.extero_coord_extent_m is not None:
            cfg.extero_grid_ordering = str(self.extero_spec[0].get("ordering", "xy"))
        # A CAMERA's pixels are rays, so their position is an ANGLE rather than a ground
        # coordinate. Per camera, because the two are different lenses -- one shared set would
        # label the far camera's tokens 1.43x too wide, and nothing would notice.
        cfg.extero_camera_intrinsics = (
            [[float(c["fx"]), float(c["fy"]), float(c["cx"]), float(c["cy"])]
             for c in self.extero_spec]
            if (cfg.extero_coord_extent_m is None and self.extero_spec
                and all("fx" in c for c in self.extero_spec))
            else None
        )
        # Resolve "auto" HERE, where the source is finally known, and write it back so the
        # checkpoint says which encoder it trained -- ``from_dict`` must never re-resolve it.
        #
        # But refuse to resolve it for a DENSE scan. "auto" gives a height scan the flat MLP,
        # which is right for the validated 176-ray grid and a trap for a metric one: a
        # 121x121 scan through one Linear is ~7.5M parameters that are permutation-invariant
        # over the grid, i.e. structurally unable to say WHERE a step edge is -- the exact
        # failure the convolutional encoder exists to fix. Flipping the default instead would
        # silently re-architect every legacy scan run, whose recordings carry no grid at all.
        # So the ambiguous case is the one that is refused, and stating either encoder works.
        if (cfg.extero_encoder == "auto" and cfg.extero_source == "height_scan"
                and sensor_dim >= _DENSE_SCAN_RAYS):
            raise ValueError(
                f"extero_encoder='auto' on a {sensor_dim}-ray height scan is ambiguous. "
                f"'auto' means the flat MLP, which is correct for the coarse grid but on this "
                f"one is a single Linear of ~{sensor_dim * cfg.d_model / 1e6:.1f}M parameters "
                "that cannot represent WHERE anything is (it is permutation-invariant over "
                "the rays), which is what the convolutional encoder was added to fix.\n"
                "  State it: --extero_encoder cnn (with --extero_cnn_strides sized to the "
                "grid, e.g. '1,2,2,2' for 121x121 -> 16x16 = 256 tokens), or --extero_encoder "
                "mlp to take the flat path deliberately."
            )
        cfg.extero_encoder = cfg.resolve_extero_encoder()
        cfg.check_extero_encoder()
        # Where a MISSED ray lands after sanitize + normalization, so the encoder can mark it
        # invalid instead of letting a hole share a number with real geometry. Only computable
        # under the declared affine, where the sentinel is a known constant rather than a
        # function of the fitted statistics.
        cfg.extero_miss_sentinel = (
            (band - float(cfg.extero_norm_offset)) / float(cfg.extero_norm_scale)
            if (cfg.extero_norm_mode == "metric" and cfg.extero_source == "height_scan")
            else None
        )

        # Reference pose per window (the snap-step robot base), raw for eval/denorm.
        self.p_ref = p_anchor.contiguous()
        self.q_ref = q_anchor.contiguous()

        # -- pinned-axis loss weighting: the per-node anchor mask, and the terrain frame it is
        # named in, expressed RELATIVE to this window's own frame --
        #
        # Stored compactly (a uint8 per node plus one angle per window) and expanded into
        # channel weights inside the loss. A dense (Nw, F, D) weight would be 2 GB at 200k
        # windows for information that is one byte per node.
        self.anchor_mask = None
        self.mask_frame_yaw_rel = None
        weighted = float(cfg.pinned_axis_loss_weight) != 1.0
        if weighted or cfg.pinned_axis_metrics:
            if weighted and not windowed:
                raise ValueError(
                    "pinned_axis_loss_weight needs the per-window clip index, which only the "
                    "windowed .h5 recordings carry. Record with --num_windows."
                )
            try:
                if not windowed:
                    raise ValueError("the per-window clip index needs a windowed recording")
                n_all = int(np.asarray(data["clip_id"]).shape[0])
                masks = window_anchor_masks(data, meta, cfg, n_all)[sel]
                frame_yaw = window_mask_frame_yaw(data, meta, cfg, n_all)[sel]
            except Exception as exc:
                # The LOSS cannot silently lose its weighting; a metric can.
                if weighted:
                    raise
                print(f"[fm-data] pinned/free val split OFF: {type(exc).__name__}: {exc}")
                masks = frame_yaw = None
        else:
            masks = frame_yaw = None
        if masks is not None:
            # The mask's downrange/lateral axes live in the TERRAIN's frame; the window's x/y
            # are the snap robot's yaw frame. What the loss needs is the angle between them,
            # so it is resolved here where both are known rather than assumed to be zero (the
            # terrain frame is world-aligned in today's libraries, but the ROBOT's heading is
            # not, so this angle is never trivially zero).
            snap_yaw = _yaw_from_quat_xyzw(q_anchor)
            self.anchor_mask = torch.from_numpy(masks.astype(np.int64))       # (Nw, F)
            self.mask_frame_yaw_rel = (
                torch.from_numpy(frame_yaw).to(snap_yaw.dtype) - snap_yaw)    # (Nw,)
            from motion_gen.world_lift import anchor_mask_is_pinned
            frac = float(anchor_mask_is_pinned(self.anchor_mask).float().mean())
            how = (f"their pinned root DOFs carry {cfg.pinned_axis_loss_weight:g}x the flow "
                   "loss (weights renormalized to mean 1, so the loss scale is unchanged)"
                   if float(cfg.pinned_axis_loss_weight) != 1.0
                   else "the loss is UNWEIGHTED; the mask is loaded only to report "
                        "val/flow_mse_masked vs flow_mse_unmasked")
            print(f"[fm-data] anchor mask: {100.0 * frac:.1f}% of predicted nodes are "
                  f"terrain-registered; {how}.")

        # Strided future command tokens.
        if cmd_dim > 0:
            cf = torch.from_numpy(np.ascontiguousarray(cmd_future)).float()  # (P|W, F, cmd)
            cf = cf[sel] if windowed else cf[snap_rows]                       # (Nw, F, cmd)
            cmd_strided = cf[:, self.cmd_steps, :].contiguous()
        else:
            cmd_strided = None

        # -- normalizers (fit on train) --
        # base_pose's orientation half is EXEMPT from standardization -- the reason the
        # normalizer carries a mask at all (see ``normalizer`` module docstring). Its six
        # channels are two orthonormal columns: already O(1) and unit-scaled, and a
        # per-channel affine would shear them off the manifold, so the model would be
        # regressing towards targets that are no longer rotations and the Gram-Schmidt on
        # the way out would silently absorb the difference. Everything else is standardized.
        #
        # Note the fit is over all (Nw * F) nodes, so base_pose's position std is set by the
        # FAR nodes -- node 0's near-zero residual ends up squashed, and node 0 is the one
        # the tracker's CLF reads. Unchanged by the representation change, but worth
        # revisiting with per-node loss weighting.
        state_mask = torch.ones(self.state_dim, dtype=torch.bool)
        if "base_pose" in self.block_slices:
            bp = self.block_slices["base_pose"].start
            state_mask[bp + 3:bp + 9] = False
        if state_normalizer is None:
            state_normalizer = Normalizer.fit(target.reshape(-1, self.state_dim), state_mask)
            declared = declared_state_scales(
                cfg, self.num_joints, self.num_extra_frames, self.blocks)
            if declared is not None:
                fitted_std = state_normalizer.std.clone()
                state_normalizer = apply_declared_scales(state_normalizer, declared)
                sel = declared > 0
                print(f"[fm-data] declared state scales on {int(sel.sum())}/{self.state_dim} "
                      "channels (mean 0) -- the rest stay fitted. Per block, declared "
                      "scale(s) vs the fitted std they replace: "
                      + declared_scale_summary(self.block_slices, declared, fitted_std))
        self.state_normalizer = state_normalizer

        if extero_normalizer is None:
            width = self.extero.shape[1]
            mask = torch.ones(width, dtype=torch.bool)
            if cfg.extero_norm_mode == "metric":
                # Skip the fit entirely rather than fitting and discarding: on a dense scan
                # the fit is a full pass over a ~15k-column array whose result is unused.
                #
                # ON THE DATA'S DEVICE, explicitly. The fitted branch below inherits it from
                # the array it reads; this one builds fresh tensors, which land on the
                # DEFAULT device -- and train.py sets that to cuda while the dataset stays in
                # host memory, so ``__getitem__`` died on "at least two devices". A
                # normalizer belongs where the data it normalizes lives.
                extero_normalizer = Normalizer(
                    torch.zeros(width, device=self.extero.device, dtype=self.extero.dtype),
                    torch.ones(width, device=self.extero.device, dtype=self.extero.dtype),
                    mask,
                )
            else:
                extero_normalizer = Normalizer.fit(
                    self.extero, mask, eps=float(cfg.extero_norm_eps))
            extero_normalizer = _pool_extero_statistics(
                extero_normalizer, cfg, n_sensors, sensor_dim
            )
        self.extero_normalizer = extero_normalizer

        if cmd_dim > 0:
            if cmd_normalizer is None:
                if cfg.cmd_scale is not None:
                    scale = [float(v) for v in cfg.cmd_scale]
                    if len(scale) != cmd_dim:
                        raise ValueError(
                            f"cmd_scale has {len(scale)} entries but the command is {cmd_dim} "
                            "wide; give one scale per channel (vx, vy, wz).")
                    if any(v <= 0.0 for v in scale):
                        raise ValueError(f"cmd_scale entries must be positive, got {scale}.")
                    # Divide by the DESIGNED maximum, mean 0: the command range is a generator
                    # setting rather than an observed distribution, so this puts it in [-1, 1]
                    # by construction AND keeps a zero command at zero. Standardizing would
                    # not: the fitted vx mean is ~1.25 m/s, so "stand still" would arrive as
                    # a large negative number rather than as nothing.
                    cmd_normalizer = Normalizer(
                        torch.zeros(cmd_dim), torch.tensor(scale, dtype=torch.float32),
                        torch.ones(cmd_dim, dtype=torch.bool))
                    obs = cmd_strided.reshape(-1, cmd_dim)
                    frac = (obs.abs() > torch.tensor(scale)).float().mean(0)
                    print(f"[fm-data] cmd_scale={scale} (declared max, mean 0): "
                          f"observed max |cmd| "
                          f"{[round(float(v), 4) for v in obs.abs().max(0).values]}, "
                          f"fraction outside the scale "
                          f"{[f'{100 * float(v):.3f}%' for v in frac]}.")
                else:
                    cmd_normalizer = Normalizer.fit(
                        cmd_strided.reshape(-1, cmd_dim), torch.ones(cmd_dim, dtype=torch.bool)
                    )
            self.cmd_normalizer = cmd_normalizer
            self.cmd_tokens = cmd_normalizer.normalize(cmd_strided).contiguous()
        else:
            self.cmd_normalizer = None
            self.cmd_tokens = None

        # Pre-normalize once.
        self.state = state_normalizer.normalize(target)           # (Nw, F, D) target
        self.history = state_normalizer.normalize(history)        # (Nw, N, D) conditioning
        # The exteroception is NOT pre-normalized: it is by far the widest array here (a dense
        # scan is ~15k columns against the state's ~120), so a normalized copy alongside the raw
        # one doubles the dataset's footprint for an elementwise affine that costs nothing per
        # item. Applied in __getitem__ instead -- same numbers, same order, one array.

        # -- observation noise (train-time augmentation; conditioning tokens only) --
        self.add_noise = bool(add_noise and cfg.obs_noise)
        self.state_noise = (
            self._state_noise_std(state_normalizer) if self.add_noise else None
        )
        # No exteroception noise: the sensor model is the RL env's DR, already in the recording.

    def _check_window_invariants(
        self,
        target: torch.Tensor,
        history: torch.Tensor,
        snap_ref: torch.Tensor,
        future_node0: torch.Tensor,
        s_slc: dict[str, tuple[int, int]],
        f: int,
        adjacent_joints: torch.Tensor | None = None,
    ) -> None:
        """Fail on the recording defects that would otherwise train a plausible wrong model.

        Every one of these is silent: the arrays are the right shape and the loss converges
        either way, and the damage only shows up as a generator that steers oddly.

        Guards 1, 2 and 4 read a block out of the packed window, so they are skipped when
        the selection has dropped it. Guard 3 works on the RAW recorded rows and always
        runs -- it is the one that pins the recorder's off-by-one, and it is worth having
        whatever the window is packing.

        Args:
            target: ``(Nw, F, D)`` future window nodes.
            history: ``(Nw, N, D)`` autoregressive history nodes.
            snap_ref: ``(Nw, node)`` recorded APPLIED reference at the snap step, raw world
                layout -- the cross-check for the row alignment.
            future_node0: ``(Nw, node)`` recorded ``ref_future`` node 0, same raw layout.
            s_slc: Recorded term name -> ``(start, end)`` column slice.
            f: ``cfg.future_len``.

        Raises:
            ValueError: On a violated invariant.
        """
        if "base_pose" in self.block_slices:
            bp = self.block_slices["base_pose"]
            pos0 = target[:, 0, bp.start:bp.start + 3]
            ori6d = target[..., bp.start + 3:bp.start + 9]

            # 1. Node-0 ego anchor. The clip anchored at the snap sits on the robot, so
            #    node 0's base_pose position is the (small) live tracking error. The
            #    paradigm assumes it: the policy reads node 0 as "where the reference is
            #    right now".
            node0_off = pos0.norm(dim=-1).mean().item()
            if node0_off > 0.15:
                raise ValueError(
                    f"target node-0 base position offset mean={node0_off:.3f} m (> 0.15): the "
                    "recorded ref_future is not ego-anchored onto the robot at the snap. "
                    "Re-record with ego_anchor_fraction=1.0 on the teacher command."
                )

            # 2. The orientation half really is a rotation. The TARGET is built from
            #    recorded unit quaternions, so its two 6D columns must be orthonormal to
            #    float precision -- the model's output need not be, but the thing it
            #    regresses towards does. This replaces the old pi-branch-cut guard, which
            #    the 6D representation makes unfireable: what it protected against was a
            #    log map wrapping, and there is no longer a log map. A packing bug (wrong
            #    slice, transposed columns, a stale 6-wide recording read as 9) lands here
            #    instead of surviving as a plausible-looking target.
            c0, c1 = ori6d[..., 0:3], ori6d[..., 3:6]
            worst = max(
                (c0.norm(dim=-1) - 1.0).abs().max().item(),
                (c1.norm(dim=-1) - 1.0).abs().max().item(),
                (c0 * c1).sum(dim=-1).abs().max().item(),
            )
            if worst > 1e-3:
                raise ValueError(
                    f"target base_pose orientation is not a rotation: worst deviation from "
                    f"two orthonormal columns is {worst:.2e} (> 1e-3). The 6D block is "
                    "built from recorded unit quaternions, so this is a packing bug, not "
                    "data -- check the base_pose slice and the recording's width."
                )

        # 3. History/target row alignment -- the highest-value guard here. ``history_ref``
        #    and ``ref_future`` come from different accessors (the per-step world cache vs
        #    the look-ahead gather), and the recorder pushes its rings one step per control
        #    step, so an off-by-one is entirely possible and entirely invisible. They must
        #    describe the SAME step: history row H-1 (which the history block excludes) is
        #    the reference at the snap, and so is ref_future node 0. Comparing them also
        #    catches history_ref carrying the GLOBALLY-placed clip base rather than the
        #    anchored one, and a joint-order scramble between the two accessors.
        a, b = s_slc["root_pos_w"]
        jp_a, jp_b = s_slc["joint_pos"]
        d_pos = (snap_ref[:, a:b] - future_node0[:, a:b]).norm(dim=-1).max().item()
        d_joint = (snap_ref[:, jp_a:jp_b] - future_node0[:, jp_a:jp_b]).abs().max().item()
        if d_pos > 1e-3 or d_joint > 1e-3:
            raise ValueError(
                "history_ref[H-1] != ref_future[0]: the recorded applied-reference ring and "
                f"the look-ahead future disagree at the snap step (base {d_pos:.4f} m, joints "
                f"{d_joint:.4f} rad). They must be the same step -- the history block is the N "
                "rows BEFORE node 0. Either the recorder's ring is off by one against the "
                "look-ahead, or history_ref is carrying the globally-placed clip base instead "
                "of the anchored one."
            )

        # 4. The history's last row is ONE control step before node 0, so the gap between
        #    them must look like one step. Calibrated against the gap between consecutive
        #    TARGET nodes -- which is one step by construction -- rather than an absolute
        #    threshold: the raw number is dominated by how fast the clip's joints move (a
        #    18 rad/s joint covers 0.36 rad in a step all by itself), so any fixed tolerance
        #    is either loose on slow clips or wrong on fast ones. An off-by-k history shows
        #    up as a gap ~k times the one-step gap, which this catches at any joint speed.
        #
        #    WHAT IT IS COMPARED AGAINST depends on the history source, and the choice is
        #    the difference between a real check and a guaranteed failure. The gap only
        #    means "one step" when both sides come from the SAME trajectory:
        #
        #    * ``reference`` -- both are the applied reference, so ``target[0]`` is the
        #      adjacent step and this is a straight comparison.
        #    * ``measured`` -- the history row is the ROBOT'S joints and ``target[0]`` is
        #      the REFERENCE's, so their gap is the TRACKING ERROR, not a step of motion.
        #      Measured on the shipped recording that is 0.070 rad against 0.016 rad per
        #      step, i.e. it reads as 4.4 steps of misalignment and fails every window.
        #      The adjacent quantity is instead the MEASURED state at the snap, which the
        #      caller passes in -- same trajectory, one step apart, so the off-by-k
        #      sensitivity is exactly as strong as for the reference arm.
        if "joint_pos" not in self.block_slices:
            return
        jp = self.block_slices["joint_pos"]
        one_step = (target[:, 1, jp] - target[:, 0, jp]).abs().mean().item()
        against = target[:, 0, jp] if adjacent_joints is None else adjacent_joints
        hist_gap = (history[:, -1, jp] - against).abs().mean().item()
        if hist_gap > 3.0 * one_step + 1e-4:
            what = "target[0]" if adjacent_joints is None else "the measured snap row"
            raise ValueError(
                f"history[-1] is {hist_gap / max(one_step, 1e-9):.1f} control steps away from "
                f"{what} in joint_pos (gap {hist_gap:.4f} rad vs {one_step:.4f} rad per "
                "step): the autoregressive history rows are misaligned against the future "
                "window. The history block is the N rows BEFORE node 0."
            )

    @staticmethod
    def _resolve_extero_spec(
        cfg: FlowMatchingConfig, meta: Any, n_sensors: int, sensor_dim: int
    ) -> list[dict] | None:
        """The sensor contract behind the exteroception readings, validated against them.

        Each source has its own contract field -- ``depth_camera_spec`` for depth,
        ``height_scan_spec`` for the scan -- and both state ``height``/``width``, which is what
        lets one image-shape resolver serve both.

        Absent contract: ``None`` when the resolved encoder is the flat MLP, which keeps every
        legacy height-scan recording (they carry only a ray count) training exactly as before.
        A HARD ERROR when the encoder is convolutional, because a bare width cannot say what
        shape the grid was and a wrong fold is scrambled input at entirely correct sizes.

        Args:
            cfg: The training config (its ``extero_source`` selects the sensor).
            meta: Recording layout metadata.
            n_sensors: Sensor count found in the recorded array.
            sensor_dim: Per-sensor width found in the recorded array.

        Returns:
            The per-sensor contract, or ``None``.

        Raises:
            ValueError: If a convolutional run's recording carries no contract, or if the
                contract disagrees with the shape of the recorded array, or if a scan's three
                statements of its grid do not close.
        """
        field = _EXTERO_SPEC_FIELD[cfg.extero_source]
        spec = getattr(meta, field, None)
        if not spec:
            if cfg.resolve_extero_encoder() != "cnn":
                return None
            raise ValueError(
                f"extero_encoder='cnn' needs the recording's '{field}' to state the "
                f"(height, width) fold of its {sensor_dim}-wide rows, and this recording has "
                "none. A ray/pixel count alone cannot distinguish a grid from its transpose, "
                "and under a convolution a wrong fold is scrambled input with every tensor the "
                "right size -- so this is refused rather than guessed. Re-record, or train this "
                "file with --extero_encoder mlp."
            )
        if len(spec) != n_sensors:
            raise ValueError(
                f"{field} lists {len(spec)} sensor(s) but the recorded "
                f"{cfg.extero_source} has {n_sensors} sensor row(s)."
            )
        for entry in spec:
            if int(entry["width"]) * int(entry["height"]) != sensor_dim:
                raise ValueError(
                    f"sensor '{entry.get('sensor')}' is {entry['width']}x{entry['height']} "
                    f"= {int(entry['width']) * int(entry['height'])} samples but the recorded "
                    f"{cfg.extero_source} row is {sensor_dim} wide."
                )
            _check_scan_grid_closes(entry, sensor_dim)
        return [dict(entry) for entry in spec]

    @staticmethod
    def _resolve_extero_image_hw(
        spec: list[dict] | None, sensor_dim: int
    ) -> list[tuple[int, int]] | None:
        """Per-sensor ``(height, width)`` from the camera contract.

        The recorded depth array is flat per camera, so the 2-D shape exists only here. That
        was immaterial while the embedding was a flat MLP; a convolution has to fold the row
        back into an image, and folding it with the axes swapped scrambles every kernel's
        neighbourhood while leaving all the shapes -- and the loss curve -- looking right.
        Hence: derived from the one producer that states it, never configured.

        Args:
            spec: The per-camera contract, or ``None`` (height scan).
            sensor_dim: Per-sensor flat width, for the cross-check.

        Returns:
            One ``(height, width)`` per sensor in token order, or ``None`` when there is no
            camera contract to derive from.

        Raises:
            ValueError: If a contract entry does not unfold to ``sensor_dim``.  (The
                per-camera check in :meth:`_resolve_extero_spec` already covers this for
                the pixel COUNT; this one is here so the tuple built below cannot be built
                from a mismatched entry regardless of call order.)
        """
        if not spec:
            return None
        out: list[tuple[int, int]] = []
        for cam in spec:
            h, w = int(cam["height"]), int(cam["width"])
            if h * w != sensor_dim:
                raise ValueError(
                    f"camera '{cam.get('sensor')}' contract {h}x{w} does not unfold the "
                    f"{sensor_dim}-wide recorded row."
                )
            out.append((h, w))
        return out

    @staticmethod
    def _camera_pose_yaw_frame(
        spec: list[dict], root_quat_w: torch.Tensor, joint_pos: torch.Tensor,
        joint_names: list[str], fk_model_path: str,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Each camera's pose in the root's yaw-aligned frame, from RECORDED state alone.

        The offline twin of ``mdp.observations.depth.depth_camera_pose_yaw_frame``, which can
        read the live articulation; here there is only the recording. It needs no more than a
        robot has: the root's orientation (gravity, i.e. the IMU), the joint angles (encoders)
        and the NOMINAL mount from the contract. That is why the existing recordings suffice
        and nothing has to be re-recorded -- and why the same three inputs reproduce this on
        hardware.

        The heading cancels out. Writing ``R_world_root = Rz(psi) @ R_rp``, the yaw-frame
        coordinates of a point are ``R_rp @ p_root``, in which ``psi`` never appears -- so a
        drifting yaw estimate cannot corrupt the cloud, and no odometry is involved.

        **Nominal, never the domain-randomised mount.** The DR perturbs the IMAGE relative to
        the stated geometry, which is exactly the calibration error hardware has; using the
        true randomised pose would teach the generator a registration it will never be given.

        Args:
            spec: The per-camera contract (``mount_body``, ``offset_pos``,
                ``offset_quat_xyzw``).
            root_quat_w: World root orientation ``(Nw, 4)``, scalar-LAST.
            joint_pos: Joint angles ``(Nw, nj)`` in the recording's joint order.
            joint_names: That joint order, from ``meta.state_joint_names``.
            fk_model_path: MJCF the root->mount chain is read from.

        Returns:
            One ``(pos (Nw, 3), rot (Nw, 3, 3))`` per camera, in contract order.

        Raises:
            ValueError: If the cameras do not share one mount body -- the FK below resolves a
                single chain, and silently using it for a second mount would misplace that
                camera by the offset between them.
        """
        from isaaclab.utils.math import matrix_from_quat, quat_inv, quat_mul, yaw_quat

        from motion_gen.fk import FKChain

        mounts = {str(cam["mount_body"]) for cam in spec}
        if len(mounts) != 1:
            raise ValueError(
                f"cameras report different mount bodies {sorted(mounts)}; the offline "
                "unprojection resolves one FK chain, so they must share a mount."
            )
        mount = mounts.pop()
        # Roll/pitch only: the frame is level (its z IS gravity), so what survives here is the
        # robot's tilt WITHIN that frame -- which is the whole point. It moves the CAMERA, and
        # moving the camera by exactly the tilt is what leaves static terrain sitting at the
        # coordinates it already had.
        q_rp = quat_mul(quat_inv(yaw_quat(root_quat_w)), root_quat_w)
        r_rp = matrix_from_quat(q_rp)                                        # (Nw, 3, 3)

        chain = FKChain(fk_model_path, list(joint_names), [mount],
                        base_link="pelvis_link", dtype=joint_pos.dtype)
        mount_pos, mount_rot, _jv, _jw = chain.pose_and_jacobian(joint_pos)
        mount_pos, mount_rot = mount_pos[:, 0], mount_rot[:, 0]              # (Nw, 3/3x3)

        out = []
        for cam in spec:
            off_p = torch.tensor([float(v) for v in cam["offset_pos"]],
                                 dtype=joint_pos.dtype, device=joint_pos.device)
            off_q = torch.tensor([float(v) for v in cam["offset_quat_xyzw"]],
                                 dtype=joint_pos.dtype, device=joint_pos.device)
            off_r = matrix_from_quat(off_q.expand(mount_rot.shape[0], 4))
            # root frame -> yaw frame, mount -> camera. Composed in that order.
            cam_pos_root = mount_pos + torch.einsum("bij,j->bi", mount_rot, off_p)
            out.append((torch.einsum("bij,bj->bi", r_rp, cam_pos_root),
                        r_rp @ mount_rot @ off_r))
        return out

    @staticmethod
    def _depth_xyz_rows(
        hs: torch.Tensor, spec: list[dict], poses: list[tuple[torch.Tensor, torch.Tensor]],
        n_sensors: int, sensor_dim: int, channels: int = 4,
    ) -> torch.Tensor:
        """Widen depth rows to ``[d, x, y, z]`` (optionally ``+ v``) per pixel, CHANNEL-MAJOR.

        Derived from the SAME post-clip array the depth channel carries, so the cloud has its
        wall at exactly the distance that channel claims; unprojecting an unclipped image
        instead would describe a different surface than the number beside it.

        Args:
            hs: Sanitized rows ``(Nw, n_sensors * sensor_dim)``.
            spec: The per-camera contract, in row order.
            poses: Per-camera ``(pos, rot)`` in the yaw frame, from
                :meth:`_camera_pose_yaw_frame`.
            n_sensors: Cameras per row.
            sensor_dim: Pixels per camera.
            channels: 4 for ``[d, x, y, z]``, 5 to append the validity plane. Must match what
                the live term (``mdp.depth_xyz_stack``) emits for the same checkpoint.

        Returns:
            ``(Nw, n_sensors * channels * sensor_dim)``, each camera's block laid out as all
            of ``d``, then all of ``x``, ``y``, ``z`` -- the layout ``images_from_flat`` folds
            and every other producer emits.
        """
        from robot_rl.tasks.manager_based.robot_rl.mdp.observations.depth import (
            invalid_depth_mask,
            unproject_to_yaw_frame,
        )

        img = hs.reshape(hs.shape[0], n_sensors, sensor_dim)
        blocks = []
        for i, cam in enumerate(spec):
            pos, rot = poses[i]
            pts = unproject_to_yaw_frame(cam, img[:, i], pos, rot)          # (Nw, 3, K)
            planes = [img[:, i : i + 1], pts]
            if channels == 5:
                # By RANGE, not by sentinel equality: the recording is noised, and the
                # noiser's blur leaves a missed pixel near the sentinel rather than on it.
                valid = (~invalid_depth_mask(cam, img[:, i])).to(img.dtype)
                planes.append(valid.reshape(hs.shape[0], 1, sensor_dim))
            blocks.append(torch.cat(planes, dim=1))                         # (Nw, C, K)
        return torch.stack(blocks, dim=1).reshape(hs.shape[0], -1)

    def _state_noise_std(self, state_normalizer: Normalizer) -> torch.Tensor:
        """Per-channel uniform noise half-width (normalized space) for a history token.

        Expands :data:`~robot_rl.flow_matching.window.REFERENCE_WINDOW_NOISE` over the
        window layout, scales by ``cfg.noise_scale``, then divides by the normalizer std.

        It is the SAME table the policy's ``reference_window`` observation is corrupted
        with, deliberately: both model the generator's prediction error, and at deploy this
        model's history is its own past output, so that error is exactly what corrupts it.
        This noise is also the main thing standing between the training history (the clip
        teacher's applied reference) and the deploy history (the model's own) -- the
        exposure-bias gap -- so it matters more here than it did when the history was
        measured state.
        """
        amp = torch.tensor(
            expand_window_blocks(self.num_joints, self.num_extra_frames,
                                 REFERENCE_WINDOW_NOISE, self.blocks),
            dtype=self.state.dtype,
        )
        return (amp * self.cfg.noise_scale) / state_normalizer.std

    def __len__(self) -> int:
        return self.state.shape[0]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        history = self.history[i]
        # Normalized per item rather than held as a second full-width copy (see __init__).
        extero = self.extero_normalizer.normalize(self.extero[i])
        if self.add_noise:
            # HISTORY only. This noise models the generator's own prediction error feeding back
            # into its history at deploy (exposure bias); the exteroception is a real sensor
            # whose noise the RL env already applied, so nothing is added to it here.
            history = history + (torch.rand(self.N, self.state_dim) * 2.0 - 1.0) * self.state_noise
        item = {
            "state": self.state[i],          # (F, D) ego-anchored clip future target
            "history": history,              # (N, D) real-robot phase conditioning
            "extero": extero,                # (T*K,) flat exteroception, normalized
            "p_ref": self.p_ref[i],
            "q_ref": self.q_ref[i],
        }
        if self.anchor_mask is not None:
            # Compact: one bitmask per predicted node plus one angle. The loss expands these
            # into channel weights for the batch it is given.
            item["anchor_mask"] = self.anchor_mask[i]              # (F,) int64
            item["mask_frame_yaw_rel"] = self.mask_frame_yaw_rel[i]
        if self.cmd_tokens is not None:
            item["cmd_tokens"] = self.cmd_tokens[i]
        return item


def build_train_val(
    data: dict[str, np.ndarray], meta: Any, cfg: FlowMatchingConfig
) -> tuple[FutureStateDataset, FutureStateDataset | None]:
    """Build train/val datasets, splitting by ``episode_id`` to avoid leakage.

    Uses snap-event windows (the robot-state-conditioned FM paradigm). Normalizers are
    fit on the train split and shared with val. ``train_ds.val_episode_ids`` records the
    held-out episode ids so the test bench can label train vs eval. If the val split has
    no windows, ``None`` is returned for it.

    Args:
        data: Recorded arrays (must include ``reanchor_active`` + ``ref_future``).
        meta: Layout metadata.
        cfg: Config.

    Returns:
        ``(train_ds, val_ds_or_None)``.
    """
    # Pre-formed windowed (.h5) recordings skip the snap_windows pass entirely.
    if "history_state" in data:
        return _build_train_val_windowed(data, meta, cfg)
    # No window-filter guard is needed on this legacy per-step path, though it looks like it
    # wants one: both filters live in the WINDOWED branch, so a per-step recording would
    # silently ignore skip_distractors / skip_families. It cannot get that far --
    # _build_train_val_windowed refuses any recording whose history_anchor is not "plan", and
    # a per-step recording has no history_anchor at all, so the format is already rejected
    # outright and for a stronger reason. A second gate here would only shadow that message.
    reanchor = data.get("reanchor_active")
    if reanchor is None:
        raise ValueError(
            "Recording has no 'reanchor_active' flag; the robot-state-conditioned FM trains "
            "on snap (re-anchor) events. Re-record with ego_anchor_fraction=1.0."
        )
    # N+1 rows: the N autoregressive context rows plus the snap row they precede (the
    # snap is node 0 of the future, so it is NOT part of the history block).
    windows = snap_windows(
        data["episode_id"], data["step_in_episode"], data["done"],
        reanchor, cfg.num_history_states + 1, cfg.stride,
    )
    if len(windows) == 0:
        raise ValueError(
            "No valid snap windows; every snap lacks N+1 prior consecutive steps. "
            "Check ego_anchor_every > num_history_states and the recording length."
        )

    if str(cfg.val_split_by) != "episode":
        raise ValueError(
            f"val_split_by={cfg.val_split_by!r} is only implemented for the windowed .h5 "
            "path; this is the legacy per-step format. Drawing an episode split here instead "
            "would report a clip-disjoint split that was never drawn."
        )
    ep_of_window = np.array([data["episode_id"][w[-1]] for w in windows])  # episode of the snap
    unique_eps = np.unique(ep_of_window)
    rng = np.random.default_rng(cfg.seed)
    n_val = int(round(cfg.val_fraction * len(unique_eps)))
    val_eps = set(rng.choice(unique_eps, size=n_val, replace=False).tolist()) if n_val > 0 else set()

    train_windows = [w for w, e in zip(windows, ep_of_window) if e not in val_eps]
    val_windows = [w for w, e in zip(windows, ep_of_window) if e in val_eps]
    print(f"[fm-data] {len(windows)} snap windows ({len(train_windows)} train / "
          f"{len(val_windows)} val) over {len(unique_eps)} episodes.")

    train_ds = FutureStateDataset(data, meta, cfg, train_windows, add_noise=True)
    train_ds.val_episode_ids = sorted(int(e) for e in val_eps)  # for the bench's eval split
    val_ds = None
    if len(val_windows) > 0:
        val_ds = FutureStateDataset(
            data, meta, cfg, val_windows,
            state_normalizer=train_ds.state_normalizer,
            cmd_normalizer=train_ds.cmd_normalizer,
            extero_normalizer=train_ds.extero_normalizer,
        )
    return train_ds, val_ds


#: How the held-out split is drawn. See :func:`draw_val_split`.
VAL_SPLIT_KEYS: tuple[str, ...] = ("episode", "clip")


def draw_val_split(
    data: dict[str, np.ndarray], cfg: FlowMatchingConfig, num_windows: int
) -> tuple[np.ndarray, str, np.ndarray]:
    """Draw the held-out window mask, keyed on episodes or on CLIPS.

    ``"episode"`` is the historical behaviour and it has a measured hole. ``episode_id`` is
    numbered PER RECORDING and ``train.py`` offsets each subsequent file past the running
    maximum, so when a run trains on a recording plus an augmented re-render of it, an
    episode and its own twin carry different ids and land on opposite sides of the draw. Worse,
    a clip played by more than one episode leaks the same way inside a single file.

    Measured on the shipped ``measured_n3`` pair: **740 of its 855 held-out clips (86.5%) also
    appear in a training window**, and the leak is not cosmetic -- the generator's residual on
    those rows is 0.41-0.78x its residual on genuinely unseen clips, group by group, and 18x
    smaller at the median on ``joint_pos``. Its reported ``valema_rmse_base_pos`` of 0.0348
    sits between the leaked rows' 0.0264 and the clean rows' 0.0441, exactly where a mixture
    should: the published held-out error is optimistic.

    ``"clip"`` closes it. ``clip_id`` indexes the clip LIBRARY, so it is the same integer for
    an episode, for every other episode that played the same motion, and for every re-render
    of either -- which makes it the only key under which "the generator has not seen this
    motion" is answerable at all.

    Args:
        data: The recording arrays, already filtered.
        cfg: The run config; ``val_split_by``, ``val_fraction`` and ``seed`` are read.
        num_windows: ``W``, the filtered window count.

    Returns:
        ``(in_val, key_name, held_out_ids)`` -- a ``(W,)`` bool mask, the key that was used,
        and the sorted ids held out under it.

    Raises:
        ValueError: If ``val_split_by`` is not one of :data:`VAL_SPLIT_KEYS`, or if it asks
            for ``"clip"`` on a recording carrying no ``clip_id``. Falling back to episodes
            there would silently restore the leak this option exists to remove.
    """
    key = str(cfg.val_split_by)
    if key not in VAL_SPLIT_KEYS:
        raise ValueError(f"val_split_by={key!r} is not one of {list(VAL_SPLIT_KEYS)}.")
    column = "episode_id" if key == "episode" else "clip_id"
    if column not in data:
        raise ValueError(
            f"val_split_by={key!r} needs a per-window '{column}' array and this recording has "
            "none. Splitting on the other key instead would silently reinstate the leak this "
            "option exists to remove."
        )
    ids = np.asarray(data[column]).reshape(-1)[:num_windows]
    unique = np.unique(ids)
    rng = np.random.default_rng(cfg.seed)
    n_val = int(round(cfg.val_fraction * len(unique)))
    held = rng.choice(unique, size=n_val, replace=False) if n_val > 0 else unique[:0]
    in_val = np.isin(ids, held) if n_val > 0 else np.zeros(num_windows, dtype=bool)
    return in_val, key, np.sort(held)


def _distractor_keep_mask(
    data: dict[str, np.ndarray], cfg: FlowMatchingConfig, num_windows: int
) -> np.ndarray | None:
    """Boolean keep-mask dropping DISTRACTOR-family windows, or ``None`` to keep everything.

    Distractor clips are flat walking routed around obstacle boxes. They are ego-anchored like
    flat clips (so they DO produce honest snap windows), but whether they help the generator is
    an open question -- hence the opt-out. ``record_data.py`` tags every window with
    ``is_distractor``, so this is a train-time switch needing no re-record.

    Args:
        data: The loaded recording arrays.
        cfg: Config; ``skip_distractors`` enables the filter.
        num_windows: ``W``, for validating the tag's length.

    Returns:
        ``(W,)`` bool array (True = keep), or ``None`` when the filter is off / a no-op.

    Raises:
        ValueError: If the filter is requested but the recording carries no tag, or the tag
            would drop every window.
    """
    if not getattr(cfg, "skip_distractors", False):
        return None
    tag = data.get("is_distractor")
    if tag is None:
        raise ValueError(
            "--skip_distractors requested but this recording has no 'is_distractor' tag (it "
            "predates the field). Either re-record with the current record_data.py, or drop the "
            "flag -- silently training on the distractors would misreport what was excluded."
        )
    tag = np.asarray(tag).reshape(-1).astype(bool)
    if tag.shape[0] != num_windows:
        raise ValueError(
            f"'is_distractor' has {tag.shape[0]} entries but there are {num_windows} windows."
        )
    n_drop = int(tag.sum())
    if n_drop == 0:
        print("[fm-data] --skip_distractors: no distractor windows in this recording (no-op).")
        return None
    if n_drop == num_windows:
        raise ValueError(
            f"--skip_distractors would drop ALL {num_windows} windows (the whole recording is "
            f"distractors); nothing left to train on."
        )
    print(f"[fm-data] --skip_distractors: dropping {n_drop}/{num_windows} distractor windows "
          f"({100.0 * n_drop / num_windows:.1f}%).")
    return ~tag


def resolve_skip_families(wanted: list[str], have: list[str]) -> tuple[list[str], list[str]]:
    """Split a requested family-exclusion list into (present, absent), raising on a typo.

    Strictness is asymmetric BY NAME, which is what lets the rule live here rather than in
    whichever caller happens to know how the list was spelled. A name in
    :data:`~robot_rl.flow_matching.config.DEFAULT_SKIP_FAMILIES` is a standing REQUEST --
    "drop the backwards jump-off windows wherever they exist" -- and a recording that has
    none of them is not an error. Any OTHER name was typed by a caller who believes it is
    there, so its absence is a typo, and silently dropping nothing would read as success.

    The absent case is not just stale recordings. A flat-only library has no terrain
    manifest, so ``record_data._clip_family_codes`` writes no family table at all -- a
    perfectly current flat recording legitimately carries no families to name.

    Args:
        wanted: Requested family names.
        have: The recording's own ``family_names`` (empty when it carries no table).

    Returns:
        ``(present, absent)``, both in the requested order.

    Raises:
        ValueError: If a name outside the default set is absent from the recording.
    """
    from .config import DEFAULT_SKIP_FAMILIES

    wanted = [f for f in (wanted or []) if f]
    have = list(have or [])
    absent = [f for f in wanted if f not in have]
    typo = [f for f in absent if f not in DEFAULT_SKIP_FAMILIES]
    if typo:
        listing = have or "(none -- no family table, which is what a flat-only library gives)"
        raise ValueError(
            f"--skip_families named {typo}, which this recording does not contain. Its "
            f"families are {listing}. A typo here would silently drop nothing, which reads "
            "as success."
        )
    return [f for f in wanted if f in have], absent


def _family_keep_mask(
    data: dict[str, np.ndarray], meta: Any, cfg: FlowMatchingConfig, num_windows: int
) -> np.ndarray | None:
    """Boolean keep-mask dropping the ``cfg.skip_families`` clip families, or ``None``.

    The general form of :func:`_distractor_keep_mask`. Reads the per-window ``clip_family``
    code and ``meta.family_names``, both written at record time, so any family is a
    train-time choice.

    Args:
        data: The loaded recording arrays.
        meta: Layout metadata (``family_names`` is read).
        cfg: Config; ``skip_families`` names what to drop.
        num_windows: ``W``, for validating the tag's length.

    Returns:
        ``(W,)`` bool array (True = keep), or ``None`` when nothing is requested and when
        nothing requested is present (a flat-only recording under the default).

    Raises:
        ValueError: If a family the caller NAMED is absent (see
            :func:`resolve_skip_families` for why a defaulted name is not), if the tag's
            length disagrees with ``num_windows``, or if the filter would drop every window.
            Each of those would otherwise look like a successful no-op.
    """
    requested = [f for f in (cfg.skip_families or []) if f]
    if not requested:
        return None
    codes = data.get("clip_family")
    names = list(getattr(meta, "family_names", []) or [])
    # A name only the DEFAULT asks for is dropped where it exists and reported where it does
    # not; a name a caller typed must be there. Applied here rather than only in train.py
    # because build_train_val is a library entry point -- the probes, the composer and the
    # tests all reach it directly, and a default that hard-failed every table-less recording
    # would break all of them.
    wanted, absent = resolve_skip_families(requested, names if codes is not None else [])
    if absent:
        print(f"[fm-data] --skip_families {absent}: not in this recording "
              f"({names or 'no family table'}); nothing to drop for them.")
    if not wanted:
        return None
    codes = np.asarray(codes).reshape(-1).astype(np.int64)
    if codes.shape[0] != num_windows:
        raise ValueError(
            f"'clip_family' has {codes.shape[0]} entries but there are {num_windows} windows."
        )
    drop = np.isin(codes, [names.index(f) for f in wanted])
    n_drop = int(drop.sum())
    if n_drop == num_windows:
        raise ValueError(
            f"--skip_families {wanted} would drop ALL {num_windows} windows; nothing left."
        )
    if n_drop == 0:
        print(f"[fm-data] --skip_families {wanted}: no such windows in this recording (no-op).")
        return None
    print(f"[fm-data] --skip_families {wanted}: dropping {n_drop}/{num_windows} windows "
          f"({100.0 * n_drop / num_windows:.1f}%).")
    return ~drop


#: ``(abspath, size, mtime_ns) -> sha256`` for clip libraries verified this process.
_LIBRARY_SHA_CACHE: dict[tuple, str] = {}


def resolve_clip_library(meta: Any, cfg: FlowMatchingConfig, why: str):
    """Open the clip library the recorded ``clip_id`` indexes into, verifying its identity.

    Shared by every offline join back to the library (transition flags, the per-node anchor
    mask). Each failure is raised rather than defaulted, because a silent fallback reports a
    run that used a feature it never actually used.

    Args:
        meta: Recording metadata (``clip_library_path`` / ``clip_library_sha256``).
        cfg: Config; ``clip_library_path`` overrides the recorded path.
        why: Name of the feature needing the join, for the message.

    Returns:
        The loaded library, with ``anchor_mask`` / ``anchor_mask_t`` / ``clip_len`` present.

    Raises:
        ValueError: If no path is known, or the library on disk is not the recorded one.
        FileNotFoundError: If the path does not exist.
    """
    import hashlib
    import os

    path = cfg.clip_library_path or getattr(meta, "clip_library_path", "") or ""
    if not path:
        raise ValueError(
            f"{why} needs the clip library that clip_id indexes into, but the recording "
            "carries no clip_library_path. Pass --clip_library_path explicitly."
        )
    if not os.path.exists(path):
        raise FileNotFoundError(f"clip library '{path}' (from the recording) does not exist.")
    want = getattr(meta, "clip_library_sha256", "") or ""
    if want:
        # Cached per (path, size, mtime) for the process: the library is multi-GB and both the
        # train and the val dataset resolve it, so an uncached check hashes ~14 GB per run to
        # answer the same question twice. Keyed on the stat so a file replaced mid-run is still
        # re-hashed rather than trusted.
        st = os.stat(path)
        key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
        got = _LIBRARY_SHA_CACHE.get(key)
        if got is None:
            got = hashlib.sha256(open(path, "rb").read()).hexdigest()
            _LIBRARY_SHA_CACHE[key] = got
        if got != want:
            raise ValueError(
                f"clip library '{path}' has sha256 {got[:12]} but the recording was made "
                f"against {want[:12]}. Clip ids are positions in ONE library and a "
                "regenerated library reuses them for different motions, so this join would "
                "silently mislabel every window."
            )
    from motion_gen.clips.clip_io import load_clip_library

    return load_clip_library(path, device="cpu",
                             keys=["anchor_mask", "anchor_mask_t", "clip_len"])


def window_anchor_masks(
    data: dict[str, np.ndarray], meta: Any, cfg: FlowMatchingConfig, num_windows: int
) -> np.ndarray:
    """Per-window, per-node anchor bitmask ``[W, F]`` (uint8; bit SET = that DOF is FREE).

    The ground truth for which root DOFs a node's reference is REGISTERED TO THE TERRAIN
    rather than re-anchored onto the robot -- flat nodes free downrange/lateral/yaw and pin
    only height, terrain nodes free lateral alone. Joined through the recorded
    ``clip_id``/``clip_step``, exactly as the transition flags are, so it needs no new
    recorded array and works on any recording carrying the clip index.

    The clip tail is repeat-last padded, so a node past the clip's real length takes the last
    real step's mask -- the same convention the reference itself is padded under.

    Args:
        data: Recorded window arrays; ``clip_id`` and ``clip_step`` are read.
        meta: Recording metadata; supplies the library path and hash.
        cfg: Config; ``future_len`` sets the horizon.
        num_windows: ``W``, for shape validation.

    Returns:
        ``[W, F]`` uint8 anchor masks.

    Raises:
        ValueError: If the recording predates ``clip_id``/``clip_step``.
    """
    if "clip_id" not in data or "clip_step" not in data:
        raise ValueError(
            "the pinned-axis loss weight needs the per-window 'clip_id'/'clip_step' index, "
            "which this recording predates. Re-record with the current record_data.py."
        )
    lib = resolve_clip_library(meta, cfg, "the pinned-axis loss weight")
    masks = lib.anchor_mask_per_step().numpy()              # (num_clips, T) uint8
    clip_len = lib.clip_len_or_full().numpy()

    cid = np.asarray(data["clip_id"]).reshape(-1).astype(np.int64)
    cstep = np.asarray(data["clip_step"]).reshape(-1).astype(np.int64)
    if cid.shape[0] != num_windows:
        raise ValueError(f"'clip_id' has {cid.shape[0]} entries but there are {num_windows} "
                         "windows.")
    steps = np.minimum(cstep[:, None] + np.arange(int(cfg.future_len))[None, :],
                       (clip_len[cid] - 1)[:, None])
    return masks[cid[:, None], steps].astype(np.uint8)      # (W, F)


def window_mask_frame_yaw(
    data: dict[str, np.ndarray], meta: Any, cfg: FlowMatchingConfig, num_windows: int
) -> np.ndarray:
    """Per-window terrain-frame heading ``[W]`` (radians, world), for the anchor mask's axes.

    The mask's ``downrange`` / ``lateral`` are named in the TERRAIN's frame, not the world's
    and not the robot's -- which world axes they are is authored per terrain as ``frame_yaw``
    in the library's manifest. Getting this from the tile rather than assuming ``+x`` is the
    same correction the mask itself documents: a 90-degree-rotated terrain family once had its
    traversal axis read as its lateral offset.

    Args:
        data: Recorded window arrays; ``clip_id`` is read.
        meta: Recording metadata.
        cfg: Config.
        num_windows: ``W``, for shape validation.

    Returns:
        ``[W]`` float32 world-frame yaw of each window's terrain frame. Zero for clips whose
        terrain declares none (flat ground has no traversal axis).
    """
    import h5py

    path = cfg.clip_library_path or getattr(meta, "clip_library_path", "") or ""
    cid = np.asarray(data["clip_id"]).reshape(-1).astype(np.int64)
    with h5py.File(path, "r") as f:
        terrain_id = f["terrain_id"][:] if "terrain_id" in f else None
        raw = f.attrs.get("terrain_manifest")
    yaw_by_terrain: dict[int, float] = {}
    if raw is not None:
        manifest = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
        yaw_by_terrain = {int(m["terrain_id"]): float(m.get("frame_yaw", 0.0))
                          for m in manifest}
    if terrain_id is None or not yaw_by_terrain:
        # Every manifest predating the field was generated under world-axis masking, which is
        # frame_yaw = 0 -- so this is the correct value, not a fallback.
        return np.zeros(num_windows, dtype=np.float32)
    # terrain_id is -1 for a clip on no terrain (flat), which has no traversal axis and no
    # manifest entry. Indexing a LUT with -1 would silently wrap to the LAST terrain's yaw, so
    # the lookup is masked rather than relying on the index.
    tid = terrain_id[cid].astype(np.int64)
    lut = np.zeros(max(int(terrain_id.max()), 0) + 1, dtype=np.float32)
    for terrain, yaw in yaw_by_terrain.items():
        if 0 <= terrain < lut.shape[0]:
            lut[terrain] = yaw
    out = np.zeros(num_windows, dtype=np.float32)
    valid = tid >= 0
    out[valid] = lut[tid[valid]]
    return out


def transition_window_mask(
    data: dict[str, np.ndarray], meta: Any, cfg: FlowMatchingConfig, num_windows: int
) -> np.ndarray:
    """Boolean ``[W]``: does each window cross a terrain transition?

    Ground truth, not a height heuristic. A window crosses a transition when the clip's
    ``anchor_mask_t`` changes pinned-ness strictly inside ``[clip_step, clip_step + F)`` --
    i.e. the reference goes from ego-anchored flat walking to terrain-registered (or back)
    somewhere in the horizon the FM predicts. Joined through the recorded ``clip_id`` /
    ``clip_step``, so it needs the library those ids index into.

    Args:
        data: Recorded window arrays; ``clip_id`` and ``clip_step`` are read.
        meta: Recording metadata; supplies the library path and its hash.
        cfg: Config; ``clip_library_path`` overrides the recorded path, ``future_len`` sets
            the horizon.
        num_windows: ``W``, for shape validation.

    Returns:
        ``[W]`` bool array, True where the window crosses a transition.

    Raises:
        ValueError: If the recording predates ``clip_id``/``clip_step``, if no library path
            is known, or if the library on disk does not match the recorded hash. Each is
            raised rather than defaulting to uniform sampling, because a silent fallback
            would report a transition-weighted run that never weighted anything.
    """
    import hashlib
    import os

    if "clip_id" not in data or "clip_step" not in data:
        raise ValueError(
            "transition_fraction needs the per-window 'clip_id'/'clip_step' index, which "
            "this recording predates. Re-record with the current record_data.py (it sets "
            "tag_clip_index=True), or leave transition_fraction=None."
        )
    path = cfg.clip_library_path or getattr(meta, "clip_library_path", "") or ""
    if not path:
        raise ValueError(
            "transition_fraction needs the clip library that clip_id indexes into, but the "
            "recording carries no clip_library_path. Pass --clip_library_path explicitly."
        )
    if not os.path.exists(path):
        raise FileNotFoundError(f"clip library '{path}' (from the recording) does not exist.")
    want = getattr(meta, "clip_library_sha256", "") or ""
    if want:
        got = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if got != want:
            raise ValueError(
                f"clip library '{path}' has sha256 {got[:12]} but the recording was made "
                f"against {want[:12]}. Clip ids are positions in ONE library and a "
                "regenerated library reuses them for different motions, so this join would "
                "silently mislabel every window."
            )
    from motion_gen.clips.clip_io import load_clip_library
    from motion_gen.world_lift import anchor_mask_is_pinned

    lib = load_clip_library(path, device="cpu",
                            keys=["anchor_mask", "anchor_mask_t", "clip_len"])
    pinned = anchor_mask_is_pinned(lib.anchor_mask_per_step()).numpy()
    clip_len = lib.clip_len_or_full().numpy()

    cid = np.asarray(data["clip_id"]).reshape(-1).astype(np.int64)
    cstep = np.asarray(data["clip_step"]).reshape(-1).astype(np.int64)
    if cid.shape[0] != num_windows:
        raise ValueError(f"'clip_id' has {cid.shape[0]} entries but there are {num_windows} "
                         "windows.")
    # Boundary count on [start+1, end) via a cumulative sum of per-step changes -- the clip
    # tail is repeat-last padded, so the span is clamped to the clip's real length.
    change = np.zeros_like(pinned, dtype=np.int64)
    change[:, 1:] = (pinned[:, 1:] != pinned[:, :-1]).astype(np.int64)
    cum = np.concatenate([np.zeros((change.shape[0], 1), np.int64),
                          np.cumsum(change, axis=1)], axis=1)
    end = np.minimum(cstep + int(cfg.future_len), clip_len[cid])
    mask = (cum[cid, end] - cum[cid, np.minimum(cstep + 1, end)]) > 0
    return mask & ((end - cstep) >= 2)


def transition_sampler_weights(
    data: dict[str, np.ndarray], meta: Any, cfg: FlowMatchingConfig,
    num_windows: int, train_idx: np.ndarray,
) -> np.ndarray | None:
    """Per-train-sample weights making transitions ``cfg.transition_fraction`` of an epoch.

    Args:
        data: Recorded window arrays.
        meta: Recording metadata.
        cfg: Config; ``transition_fraction`` is the target fraction (``None`` -> no weighting).
        num_windows: ``W``, for shape validation.
        train_idx: Indices of the TRAIN split, so val stays at the natural distribution.

    Returns:
        ``[len(train_idx)]`` float64 weights, or ``None`` when no weighting is requested.

    Raises:
        ValueError: If the target fraction is not in ``(0, 1)``, or the train split contains
            no transition windows (or nothing but), which no reweighting can fix.
    """
    p = cfg.transition_fraction
    if p is None:
        return None
    if not 0.0 < float(p) < 1.0:
        raise ValueError(f"transition_fraction must be in (0, 1), got {p}.")
    mask = transition_window_mask(data, meta, cfg, num_windows)[train_idx]
    n_t = int(mask.sum())
    n_f = int(mask.size - n_t)
    if n_t == 0 or n_f == 0:
        raise ValueError(
            f"transition_fraction={p} needs both kinds in the train split, but it has "
            f"{n_t} transition and {n_f} non-transition windows."
        )
    w_t = float(p) * n_f / ((1.0 - float(p)) * n_t)
    weights = np.ones(mask.size, dtype=np.float64)
    weights[mask] = w_t
    print(f"[fm-data] transition_fraction={p}: {n_t}/{mask.size} "
          f"({100.0 * n_t / mask.size:.1f}%) of train windows cross a terrain transition; "
          f"weighting them {w_t:.2f}x (each seen ~{w_t:.1f}x per epoch).")
    return weights


def _build_train_val_windowed(
    data: dict[str, np.ndarray], meta: Any, cfg: FlowMatchingConfig
) -> tuple[FutureStateDataset, FutureStateDataset | None]:
    """Train/val split for the pre-formed windowed (``.h5``) format.

    The snap windows are already formed at record time (:class:`RecordedWindowWriter`),
    so there is no :func:`snap_windows` pass: split the ``W`` windows by their per-window
    snap ``episode_id`` (leakage-free, same RNG/``val_fraction`` as the per-step path) and
    pass a 1-D window-index selector to :class:`FutureStateDataset`. ``cfg.stride`` applies
    as a global train-time subsample of each split (a no-op at the default ``stride=1``).
    """
    W = int(data["history_state"].shape[0])
    ep = np.asarray(data["episode_id"]).reshape(-1)  # (W,) snap-row episode id per window
    keep = _distractor_keep_mask(data, cfg, W)
    fam_keep = _family_keep_mask(data, meta, cfg, W)
    keep = fam_keep if keep is None else (keep if fam_keep is None else (keep & fam_keep))
    if keep is not None:
        ep = ep[keep]
        data = {**data, **{k: v[keep] for k, v in data.items()
                           if isinstance(v, np.ndarray) and v.shape[:1] == (W,)}}
        W = int(ep.shape[0])
    in_val, split_key, held_ids = draw_val_split(data, cfg, W)
    all_idx = np.arange(W, dtype=np.int64)
    stride = max(int(cfg.stride), 1)
    train_idx = all_idx[~in_val][::stride]
    val_idx = all_idx[in_val][::stride]
    # The EPISODES that ended up held out are recorded whichever key drew the split, because
    # every downstream consumer reads `val_episode_ids` -- but under a clip split they are a
    # CONSEQUENCE of the draw, not the draw itself, which is what `val_split_key` says.
    val_eps = np.unique(ep[in_val]) if in_val.any() else ep[:0]
    print(f"[fm-data] {W} snap windows ({train_idx.size} train / {val_idx.size} val) "
          f"over {np.unique(ep).size} episodes, split by {split_key.upper()} "
          f"({held_ids.size} held out) [windowed .h5].")
    if split_key == "clip":
        print("[fm-data] the held-out set is clip-disjoint from training: no episode, repeat "
              "play or augmented re-render of a val clip reaches the train split.")

    train_ds = FutureStateDataset(data, meta, cfg, train_idx, add_noise=True)
    train_ds.val_episode_ids = sorted(int(e) for e in val_eps)  # for the bench's eval split
    train_ds.val_split_key = split_key
    train_ds.val_split_ids = [int(i) for i in held_ids]
    # Sampler weights for cfg.transition_fraction, on the TRAIN split only -- val must keep
    # the natural distribution or its metrics stop being comparable across runs.
    train_ds.sampler_weights = transition_sampler_weights(data, meta, cfg, W, train_idx)
    val_ds = None
    if val_idx.size > 0:
        val_ds = FutureStateDataset(
            data, meta, cfg, val_idx,
            state_normalizer=train_ds.state_normalizer,
            cmd_normalizer=train_ds.cmd_normalizer,
            extero_normalizer=train_ds.extero_normalizer,
        )
    return train_ds, val_ds
