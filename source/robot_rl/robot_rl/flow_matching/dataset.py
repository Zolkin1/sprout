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


def _pool_extero_statistics(
    normalizer: Normalizer, cfg: FlowMatchingConfig, n_sensors: int, sensor_dim: int
) -> Normalizer:
    """Collapse a per-element exteroception normalizer to one statistic per SENSOR.

    A no-op unless ``cfg.extero_norm_mode == "per_sensor"``. Motivation, in full, is on
    :attr:`FlowMatchingConfig.extero_norm_mode`; the short version is that a per-pixel
    affine is a spatially varying transform, and a convolution's weight sharing assumes the
    transform is uniform.

    The pooled value is broadcast back over the sensor's elements, so the returned
    normalizer has the SAME ``(T*K,)`` shape as the fitted one -- the checkpoint, the folded
    ONNX normalization and the deploy path are all shape-identical either way.

    The std is pooled as the RMS of the per-element stds rather than their mean, because
    what should be preserved is the total variance the sensor carries, not the average
    scale; the eps floor is re-applied afterwards so a near-constant sensor still cannot
    blow the normalization up.

    Args:
        normalizer: The freshly fitted per-element normalizer.
        cfg: Training config (``extero_norm_mode`` and ``extero_norm_eps`` are read).
        n_sensors: Number of sensor rows packed into the flat width.
        sensor_dim: Elements per sensor.

    Returns:
        Either ``normalizer`` unchanged, or one with per-sensor-constant statistics.

    Raises:
        ValueError: On an unrecognised ``extero_norm_mode``.
    """
    cfg.check_extero_norm_mode()
    mode = str(cfg.extero_norm_mode)
    if mode == "per_element":
        return normalizer
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
        # Do NOT stack the sim's sensor DR with the train-time additive noise. When the
        # recording already carries the noiser's corruption (structured: per-episode plane +
        # blur + camera pose, per-step additive + dropout), adding a scalar on top over-noises
        # the one signal the generator uses to place the reference on terrain. Same gating
        # style as the block-tied loss weights below, and only depth is affected: the recorder
        # captures height_scan through the raw obs FUNCTION, which bypasses the term's noise.
        if getattr(meta, "extero_corrupted", False) and cfg.extero_source == "depth":
            if cfg.noise_depth > 0.0:
                print(f"[fm-data] noise_depth={cfg.noise_depth} -> 0.0: this recording already "
                      "carries the sim's depth DR (extero_corrupted=True), so the train-time "
                      "noise would double-count it.")
                cfg.noise_depth = 0.0
        # The complementary gate, and the one the depth-shaped flag above cannot cover: a
        # recording whose exteroception is noise-free BY CONSTRUCTION must not have train-time
        # noise added to it. Without this a privileged scan run silently keeps
        # noise_height_scan=0.1 m -- four grid cells of vertical error on a 0.025 m grid, i.e.
        # the vertical channel quantised coarser than the step edge being localised, which is
        # incoherent next to a dense grid and quietly defeats the whole experiment.
        #
        # Written on the FIELD rather than through the `noise_extero` property, which is
        # read-only, and printed either way so a run's log states the amplitude it used.
        if getattr(meta, "extero_noise_free", False):
            field = ("noise_height_scan" if cfg.extero_source == "height_scan"
                     else "noise_depth")
            if float(getattr(cfg, field)) > 0.0:
                print(f"[fm-data] {field}={getattr(cfg, field)} -> 0.0: this recording's "
                      "exteroception is noise-free BY CONSTRUCTION (extero_noise_free=True: no "
                      "sensor drift, no DR event, read through the raw obs function), so adding "
                      "noise would corrupt a signal recorded precisely to be exact.")
                setattr(cfg, field, 0.0)
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
            hw_meas = state_t[idx_hist]                      # (Nw, N, node)
            hw_ref = ref_t[idx_hist]                         # (Nw, N, node)
            snap_meas = state_t[snap_rows]                   # (Nw, node)
            snap_ref = ref_t[snap_rows]
            tw = rf_t[snap_rows]                             # (Nw, F, node)

        p_anchor, q_anchor = _p(snap_meas), _q(snap_meas)    # (Nw, 3) / (Nw, 4)

        # -- history: the emitted reference at lags n..1, all in the PLAN's frame (windowed;
        # the legacy per-step path still anchors each row on its own step) --
        history = _rows_to_window(
            lambda name: hw_ref[..., s_slc[name][0]:s_slc[name][1]],
            _p(hw_meas), _q(hw_meas), self.target_frame_names, self.blocks,
        ).contiguous()                                        # (Nw, n, D)

        # -- target: the F-node ego-anchored clip future, all in the snap robot's frame --
        target = _rows_to_window(
            lambda name: tw[..., s_slc[name][0]:s_slc[name][1]],
            p_anchor[:, None, :], q_anchor[:, None, :], self.target_frame_names, self.blocks,
        ).contiguous()                                        # (Nw, F, D)

        self._check_window_invariants(target, history, snap_ref, tw[:, 0], s_slc, f)

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
        # Sanitize before the normalizer fit so a non-finite entry cannot poison the
        # per-dimension mean and yield a NaN loss (matches the deploy token builder). The band
        # is the config's, so a run can widen it past the per-source default -- a 1 m band
        # destroys ~11% of a 3 m scan on this terrain, concentrated downrange.
        band = cfg.resolve_extero_clip()
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
        row_channels = int(cfg.extero_channels)
        if row_channels > 1:
            if cfg.extero_source != "depth":
                raise ValueError(
                    f"extero_channels={row_channels} but extero_source="
                    f"'{cfg.extero_source}'. The extra planes are a pixel's xyz from the "
                    "pinhole model; a height scan's rays already ARE ground points."
                )
            if row_channels not in (4, 5):
                raise ValueError(
                    f"extero_channels={row_channels} is not supported; the depth path emits "
                    "1 (depth alone), 4 (depth + xyz) or 5 (plus validity)."
                )
            poses = self._camera_pose_yaw_frame(
                self.extero_spec, _q(snap_meas), snap_meas[:, s_slc["joint_pos"][0]:
                                                           s_slc["joint_pos"][1]],
                [str(j) for j in meta.state_joint_names], cfg.fk_model_path,
            )
            hs = self._depth_xyz_rows(hs, self.extero_spec, poses, n_sensors, sensor_dim,
                                      channels=row_channels)
            sensor_dim = row_channels * sensor_dim      # the ROW width the model receives
            cfg.extero_channel_names = ["d", "x", "y", "z", "v"][:row_channels]
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
        # Height-scan noise (normalized space = raw m / std), per ray. The height token is a
        # real (noisy) sensor at deploy, so it is noised in training like the history tokens.
        self.extero_noise = (
            (float(cfg.noise_extero) * float(cfg.noise_scale)) / extero_normalizer.std
            if self.add_noise else None
        )

    def _check_window_invariants(
        self,
        target: torch.Tensor,
        history: torch.Tensor,
        snap_ref: torch.Tensor,
        future_node0: torch.Tensor,
        s_slc: dict[str, tuple[int, int]],
        f: int,
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
        if "joint_pos" not in self.block_slices:
            return
        jp = self.block_slices["joint_pos"]
        one_step = (target[:, 1, jp] - target[:, 0, jp]).abs().mean().item()
        hist_gap = (history[:, -1, jp] - target[:, 0, jp]).abs().mean().item()
        if hist_gap > 3.0 * one_step + 1e-4:
            raise ValueError(
                f"history[-1] is {hist_gap / max(one_step, 1e-9):.1f} control steps away from "
                f"target[0] in joint_pos (gap {hist_gap:.4f} rad vs {one_step:.4f} rad per "
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
        # robot's tilt WITHIN that frame -- which is the whole point, since a pelvis pitch
        # moves the points while the terrain did not move.
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
            history = history + (torch.rand(self.N, self.state_dim) * 2.0 - 1.0) * self.state_noise
            extero = extero + (torch.rand(self.extero.shape[1]) * 2.0 - 1.0) * self.extero_noise
        item = {
            "state": self.state[i],          # (F, D) ego-anchored clip future target
            "history": history,              # (N, D) real-robot phase conditioning
            "extero": extero,                # (T*K,) flat exteroception (noised in train)
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
        ``(W,)`` bool array (True = keep), or ``None`` when nothing is requested.

    Raises:
        ValueError: If the recording predates the tag, a named family is absent from it, or
            the filter would drop every window. Each of those would otherwise look like a
            successful no-op.
    """
    wanted = [f for f in (cfg.skip_families or []) if f]
    if not wanted:
        return None
    codes = data.get("clip_family")
    names = list(getattr(meta, "family_names", []) or [])
    if codes is None or not names:
        raise ValueError(
            f"--skip_families {wanted} needs the per-window 'clip_family' tag and the "
            "recording's family_names; this recording predates them. Re-record with the "
            "current record_data.py (it writes both unconditionally), or drop the flag -- "
            "silently training on those families would misreport what was excluded."
        )
    unknown = [f for f in wanted if f not in names]
    if unknown:
        raise ValueError(
            f"--skip_families named {unknown}, which this recording does not contain. Its "
            f"families are {names}. A typo here would silently drop nothing."
        )
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
    unique_eps = np.unique(ep)
    rng = np.random.default_rng(cfg.seed)
    n_val = int(round(cfg.val_fraction * len(unique_eps)))
    val_eps = set(rng.choice(unique_eps, size=n_val, replace=False).tolist()) if n_val > 0 else set()

    all_idx = np.arange(W, dtype=np.int64)
    in_val = np.isin(ep, list(val_eps)) if val_eps else np.zeros(W, dtype=bool)
    stride = max(int(cfg.stride), 1)
    train_idx = all_idx[~in_val][::stride]
    val_idx = all_idx[in_val][::stride]
    print(f"[fm-data] {W} snap windows ({train_idx.size} train / {val_idx.size} val) "
          f"over {len(unique_eps)} episodes [windowed .h5].")

    train_ds = FutureStateDataset(data, meta, cfg, train_idx, add_noise=True)
    train_ds.val_episode_ids = sorted(int(e) for e in val_eps)  # for the bench's eval split
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
