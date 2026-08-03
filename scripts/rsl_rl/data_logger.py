# Copyright 2026 Zachary Olkin. All rights reserved.

"""Data logger for play-time trajectory recording.

Collects per-timestep data from the Isaac Lab environment during policy playback.
Metadata (axis names, joint names, dt) is captured once at initialization.
Per-step tensors are accumulated and finalized into numpy arrays.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from robot_rl.tasks.manager_based.robot_rl.mdp.observations.observations import (
    base_pose_error,
    ee_ref_relative_to_base,
    joint_pos_ref,
    joint_vel_ref,
    ref_base_velocity_local,
    reference_window,
    reference_window_layout,
    reference_window_to_world,
)

# Trajectory command term names to search for (in priority order)
_TRAJ_TERM_NAMES = ("traj_ref", "hlip_ref")


def _obs_term_slice(unwrapped, term: str) -> tuple[str, slice] | None:
    """Locate ``term`` inside whichever concatenated observation group carries it.

    Args:
        unwrapped: The unwrapped Isaac Lab env.
        term: Observation term name.

    Returns:
        ``(group_name, slice)`` into that group's flat vector, or ``None`` if no active
        group has the term.
    """
    om = unwrapped.observation_manager
    for group, names in om.active_terms.items():
        if term not in names:
            continue
        dims = [int(np.prod(s)) for s in om.group_obs_term_dim[group]]
        i = list(names).index(term)
        start = int(sum(dims[:i]))
        return group, slice(start, start + dims[i])
    return None


def _group_tensor(obs, group: str) -> torch.Tensor | None:
    """The concatenated tensor for one observation group, from whatever ``obs`` is.

    rsl-rl hands back a plain tensor (single group), a dict of groups, or a TensorDict
    (NOT a ``dict`` subclass), so probe by subscripting rather than by type.
    """
    if obs is None:
        return None
    if hasattr(obs, "ndim") and not hasattr(obs, "keys"):
        return obs
    try:
        return obs[group]
    except Exception:  # noqa: BLE001 - group absent, or an obs type we do not know
        return None


def _split_reference_window(
    window: torch.Tensor, layout: list[tuple[str, int]], num_steps: int
) -> dict[str, torch.Tensor]:
    """Split a flat ``reference_window`` into one ``future_<block>`` tensor per block.

    The observation packs the window as ``[step0(D), step1(D), ...]``; the plots want
    one signal per physical quantity with the look-ahead index as the inner axis. Each
    returned tensor is ``[N, num_steps * width]``, laid out ``[step0, step1, ...]``.

    Args:
        window: ``[N, num_steps * D]`` reference window observation.
        layout: ``[(block_name, width), ...]`` from ``reference_window_layout``.
        num_steps: Number of samples in the window.

    Returns:
        Mapping of ``"future_<block_name>"`` to its ``[N, num_steps * width]`` slice.
    """
    n = window.shape[0]
    per_step = sum(width for _, width in layout)
    stacked = window.view(n, num_steps, per_step)
    out: dict[str, torch.Tensor] = {}
    offset = 0
    for block_name, width in layout:
        out[f"future_{block_name}"] = stacked[:, :, offset:offset + width].reshape(n, -1).clone()
        offset += width
    return out

# Attributes to pull from the trajectory command term each step.
# ``current_domain`` / ``cur_ref_frame_idx`` used to be logged here; the foot-stance path that
# updated them is gone, so they were constants and are no longer allocated at all.
_TRAJ_ATTRS = (
    "y_des", "y_act", "dy_des", "dy_act",
    "v", "vdot", "phasing_var",
    "ref_poses",
)


class DataLogger:
    """Collects per-timestep data from the environment during policy playback.

    Usage::

        logger = DataLogger(env)
        for step in range(N):
            ...
            logger.collect_step(env)
        data, metadata = logger.finalize()
    """

    def __init__(self, env, log_fm_window: bool = False, log_world: bool = False) -> None:
        """Initialize the logger and capture one-time metadata from the env.

        Args:
            env: The RslRlVecEnvWrapper-wrapped environment.
            log_fm_window: If True and the trajectory term is a flow-matching
                generator command, additionally log the full F-step predicted
                window (``fm_pred_y`` / ``fm_pred_dy`` / ``fm_pred_idx``) each
                step so ``plot_fm_predictions`` can fan the predictions off the
                tracked reference.  Heavier (``[T, N, F, P]``), so it's opt-in;
                ``fm_active`` (the cheap source mask used for the transition
                line) is always logged when available.
            log_world: If True, additionally log everything a 3-D replay needs -- the
                measured root pose, tracked body positions, the exteroception readings,
                the per-env clip/tile assignment, the generator's predicted horizon in
                world coordinates, and the reference window lifted back to world for BOTH
                sources.  See :meth:`_collect_world`.  Used by
                ``scripts/rsl_rl/fm_debug.py``; off for ``play.py``, whose plots need none
                of it and whose rollouts are long.
        """
        self._data: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._step_count: int = 0
        self._warned_sources: set[str] = set()
        self._log_fm_window = log_fm_window
        self._log_world = log_world
        # Reference-window shape; overridden from the obs cfg in _collect_metadata so the
        # logged obs match exactly what the policy is fed.
        self._ref_num_steps: int = 11
        self._ref_stride: int = 5
        # Which reference blocks the term actually packs; None = the full default set.
        # Read from the obs cfg below, so a task that drops blocks is sliced correctly.
        self._ref_blocks: tuple[str, ...] | None = None
        self.metadata: dict[str, Any] = {}
        # Per-step termination signals, stacked in finalize() into a
        # "successful tracking" mask (steps in episodes that ended via
        # time_out vs early termination).
        self._dones: list[torch.Tensor] = []
        self._time_outs: list[torch.Tensor] = []

        self._collect_metadata(env)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def num_steps(self) -> int:
        """Number of timesteps collected so far."""
        return self._step_count

    def collect_distillation_actions(
        self,
        student_actions: torch.Tensor,
        teacher_actions: torch.Tensor,
    ) -> None:
        """Log raw student and teacher policy outputs for one step.

        Args:
            student_actions: Student network output for this step, shape ``[N_envs, n_actions]``.
            teacher_actions: Teacher network output for this step, shape ``[N_envs, n_actions]``.
        """
        self._data["student_actions"].append(student_actions.detach().clone())
        self._data["teacher_actions"].append(teacher_actions.detach().clone())

    def collect_step(self, env, policy_obs=None) -> None:
        """Capture all loggable data for the current timestep.

        Args:
            env: The RslRlVecEnvWrapper-wrapped environment.
            policy_obs: The observation dict computed from the SAME simulator state the
                rest of this row describes -- i.e. what ``env.step`` just returned, not the
                pre-step observation the action was drawn from.  Only used with
                ``log_world``; it is the one signal that carries the observation
                CORRUPTION (noise + clipping), which nothing recomputed here can recover.
        """
        unwrapped = env.unwrapped
        step: dict[str, torch.Tensor] = {}

        # 0. Termination signals for this step. Read post-step (before the next
        # env.step overwrites them) and stored aligned with the per-step data so
        # finalize() can mask stats down to episodes that ended via time_out
        # (successful tracking) vs early termination (e.g. a fall).
        try:
            tm = unwrapped.termination_manager
            self._dones.append(tm.dones.clone())
            self._time_outs.append(tm.time_outs.clone())
        except Exception as exc:
            self._warn("term_signals", f"Could not read termination signals: {exc}")

        # 1. Trajectory command term attributes
        if self._traj_term_name is not None:
            try:
                ref = unwrapped.command_manager.get_term(self._traj_term_name)
                for attr in _TRAJ_ATTRS:
                    val = getattr(ref, attr, None)
                    if val is not None and isinstance(val, torch.Tensor):
                        step[attr] = val.clone()
                    elif val is None:
                        self._warn(f"traj_attr:{attr}",
                                   f"Trajectory term '{self._traj_term_name}' has no attribute '{attr}'")

                # Per-step trajectory indices (multiskill only — others have
                # a single global trajectory and don't need this).
                manager = getattr(ref, "manager", None)
                if manager is not None and hasattr(manager, "get_current_trajectory_indices"):
                    try:
                        step["traj_idx"] = manager.get_current_trajectory_indices().clone()
                    except Exception as exc:
                        self._warn("traj_idx", f"Could not read traj_idx: {exc}")

                # Contact-gate state (multiskill phase-based command only).
                # ``gate_rel_phi_pre_snap`` is the post-update, pre-snap
                # value the gate logic actually decided on — what we want
                # to display so dots land at the true fire-decision
                # offset (otherwise we'd show the post-snap value, which
                # is one step off and below the early-fire window).  Fall
                # back to ``gate_rel_phi`` if the pre-snap snapshot isn't
                # exposed (older managers).
                if manager is not None:
                    grp_pre = getattr(manager, "gate_rel_phi_pre_snap", None)
                    grp = getattr(manager, "gate_rel_phi", None)
                    if isinstance(grp_pre, torch.Tensor):
                        step["gate_rel_phi"] = grp_pre.clone()
                    elif isinstance(grp, torch.Tensor):
                        step["gate_rel_phi"] = grp.clone()
                    ngi = getattr(manager, "next_gate_idx", None)
                    if isinstance(ngi, torch.Tensor):
                        step["next_gate_idx"] = ngi.clone()

                    # Early-fire window threshold per env: ``-W * pre_size``
                    # of the currently armed gate.  Lets the plot draw the
                    # window edge alongside ``gate_rel_phi`` so you can see
                    # exactly when the window opens and closes per cycle.
                    cfg = getattr(ref, "cfg", None)
                    W = getattr(cfg, "contact_gate_window_frac", None) if cfg else None
                    pre_table = getattr(manager, "_gate_pre_size_table", None)
                    if (W is not None and pre_table is not None
                            and isinstance(ngi, torch.Tensor)):
                        try:
                            traj_idx = manager.get_current_trajectory_indices()
                            active = ngi >= 0
                            safe = torch.clamp(ngi, min=0)
                            pre_size = pre_table[traj_idx, safe]
                            thresh = -float(W) * pre_size
                            thresh = torch.where(active, thresh, torch.zeros_like(thresh))
                            step["gate_early_window"] = thresh.clone()
                        except Exception as exc:
                            self._warn("gate_early_window",
                                       f"Could not compute gate_early_window: {exc}")

                # Fallback: if the command doesn't expose phasing_var (e.g.
                # multiskill, which keeps phase state on the manager), read
                # ``manager.phase`` directly so downstream plots still work.
                if (
                    "phasing_var" not in step
                    and hasattr(ref, "manager")
                    and getattr(ref.manager, "phase", None) is not None
                ):
                    try:
                        step["phasing_var"] = ref.manager.phase.clone()
                    except Exception as exc:
                        self._warn("phasing_var_fallback",
                                   f"Could not read manager.phase: {exc}")
            except Exception as exc:
                self._warn("traj_term", f"Failed to read trajectory term: {exc}")

        # 1b. Commanded (joystick-level) base velocity for motion-sequence
        # clips: the upstream (vx, vy, wz) command baked into the clip library
        # (``target_vel``).  This is the velocity we are *trying* to track -- the
        # closest analog to the old ``base_velocity`` command term -- and is
        # paired with the robot's measured body-frame velocity (section 4) for
        # the velocity-tracking plot.
        if self._traj_term_name is not None:
            try:
                ref = unwrapped.command_manager.get_term(self._traj_term_name)
                if getattr(ref, "command_velocity_available", False):
                    step["target_vel"] = ref.get_command_velocity().clone()
            except Exception as exc:
                self._warn("target_vel", f"Could not read commanded target velocity: {exc}")

        # 1b'. Trajectory-reference POLICY OBSERVATIONS (computed clean, no obs
        # corruption) -- the exact base-relative reference signals the policy
        # consumes, for obs-level diagnostics (e.g. spotting lookahead truncation
        # / re-plan jumps in the FM-in-loop). Keyed ``obs_*``; plotted by the
        # "trajectory_obs" play plot. Skipped silently for commands that don't
        # expose the reference (the helpers raise).
        if self._traj_term_name is not None:
            name = self._traj_term_name
            ns, st, blocks = self._ref_num_steps, self._ref_stride, self._ref_blocks
            def _traj_obs() -> dict[str, torch.Tensor]:
                out = {
                    "base_pose_error": base_pose_error(unwrapped, name).clone(),
                    "base_vel_ref": ref_base_velocity_local(unwrapped, name).clone(),
                    "joint_pos_ref": joint_pos_ref(unwrapped, name).clone(),
                    "joint_vel_ref": joint_vel_ref(unwrapped, name).clone(),
                    "ee_ref": ee_ref_relative_to_base(unwrapped, name).clone(),
                }
                # The policy's merged look-ahead, split back into named per-block
                # signals so each keeps its own plot. Index 0 of every block is the
                # CURRENT step (the window starts at offset 0), so block 0 should
                # track the matching current-step term above.
                cmd = unwrapped.command_manager.get_term(name)
                win = reference_window(unwrapped, name, num_steps=ns, stride=st, blocks=blocks)
                out.update(_split_reference_window(
                    win, reference_window_layout(cmd, blocks), ns))
                return out
            try:
                for k, v in _traj_obs().items():
                    step[f"obs_{k}"] = v
                # If the command can also serve the CLIP reference (the FM-in-loop
                # FmTrackingCommand), log the clip's version of every obs under
                # the same robot state -> the FM (obs_*) and clip (obs_clip_*) signals
                # overlay in the trajectory_obs plot, the obs-level FM-vs-clip view.
                ref = unwrapped.command_manager.get_term(name)
                if hasattr(ref, "as_clip_reference"):
                    with ref.as_clip_reference():
                        for k, v in _traj_obs().items():
                            step[f"obs_clip_{k}"] = v
                        # Also capture the clip's full WORLD reference output (y_des/
                        # dy_des) so the positions/velocities plots can overlay the clip
                        # reference alongside the FM one (world-frame FM-vs-clip view).
                        # as_clip_reference() swaps ref.y_des/dy_des to the clip.
                        if isinstance(getattr(ref, "y_des", None), torch.Tensor):
                            step["y_des_clip"] = ref.y_des.clone()
                        if isinstance(getattr(ref, "dy_des", None), torch.Tensor):
                            step["dy_des_clip"] = ref.dy_des.clone()
                # Symmetrically capture the FM reference: the per-term reference OBS
                # (obs_fm_*, for the clip-vs-FM obs-diff plot) and the full world
                # reference output (y_des_fm/dy_des_fm, for the positions overlay). The
                # passive FM in drive_with_clip mode; == applied otherwise. Logging both
                # obs_clip_* and obs_fm_* makes the comparison work in either direction.
                if hasattr(ref, "as_fm_reference"):
                    with ref.as_fm_reference():
                        for k, v in _traj_obs().items():
                            step[f"obs_fm_{k}"] = v
                        if isinstance(getattr(ref, "y_des", None), torch.Tensor):
                            step["y_des_fm"] = ref.y_des.clone()
                        if isinstance(getattr(ref, "dy_des", None), torch.Tensor):
                            step["dy_des_fm"] = ref.dy_des.clone()
            except Exception as exc:
                self._warn("traj_obs", f"Could not compute trajectory reference obs: {exc}")

        # 1c. Flow-matching generator source mask + (optionally) the predicted
        # window.  ``fm_active`` ([N] bool) marks the envs whose reference is
        # being driven by the FM (vs the clip) this step -- the play plots use it
        # to draw the clip->FM transition line and to know that ``y_des`` is the
        # FM prediction, not the clip, while it's True.  When ``log_fm_window`` is
        # set we also capture the full F-step prediction in the y_des/dy_des
        # layout plus the currently-applied window index, so ``plot_fm_predictions``
        # can fan each fresh prediction forward off the tracked line.  Both are
        # no-ops for clip-only commands (no ``fm_active`` attribute).
        if self._traj_term_name is not None:
            try:
                ref = unwrapped.command_manager.get_term(self._traj_term_name)
                fm_active = getattr(ref, "fm_active", None)
                if isinstance(fm_active, torch.Tensor):
                    step["fm_active"] = fm_active.clone()
                    # Applied window index, logged every step (cheap, [N] long): a
                    # re-plan is where it resets to fm_novel_start (non-increase), used
                    # to mark the FM/clip re-anchor instants on the trajectory_obs plots.
                    pred_idx = getattr(ref, "fm_pred_idx", None)
                    if isinstance(pred_idx, torch.Tensor):
                        step["fm_pred_idx"] = pred_idx.clone()
                    if self._log_fm_window and hasattr(ref, "predicted_outputs_window"):
                        y_win, dy_win = ref.predicted_outputs_window()
                        step["fm_pred_y"] = y_win.clone()
                        step["fm_pred_dy"] = dy_win.clone()
            except Exception as exc:
                self._warn("fm_source", f"Could not read FM source / predicted window: {exc}")

        # 2. Base velocity command
        try:
            step["base_velocity"] = unwrapped.command_manager.get_command("base_velocity").clone()
        except Exception as exc:
            self._warn("base_velocity", f"Could not read base_velocity command: {exc}")

        # 3. Action targets
        try:
            action_term = unwrapped.action_manager.get_term("joint_pos")
            step["action_targets"] = action_term._processed_actions.clone()
        except Exception as exc:
            self._warn("action_targets", f"Could not read action targets: {exc}")

        # 4. Robot articulation data
        try:
            robot = unwrapped.scene.articulations["robot"]
            step["joint_pos"] = _to_torch(robot.data.joint_pos).clone()
            step["applied_torque"] = _to_torch(robot.data.applied_torque).clone()
            # Measured body-frame base velocity as (vx forward, vy lateral, wz
            # yaw rate) to pair 1:1 with the clip's (vx, vy, wz) ``target_vel``
            # in the velocity-tracking plot.
            lin_b = _to_torch(robot.data.root_lin_vel_b)
            ang_b = _to_torch(robot.data.root_ang_vel_b)
            step["robot_base_vel"] = torch.stack([lin_b[:, 0], lin_b[:, 1], ang_b[:, 2]], dim=-1).clone()
        except Exception as exc:
            self._warn("robot_data", f"Could not read robot articulation data: {exc}")

        # 5. Skill + terrain row/col (only for envs that use the
        # multiskill stack: ``MultiskillVelocityTrackingCommand`` for the
        # informational sampled bucket, ``BatchedMultiSkillCommand`` for
        # the active skill + pending state, ``MetaTerrainImporter`` for
        # cell coordinates).
        try:
            vel_cmd = unwrapped.command_manager.get_term("base_velocity")
            if hasattr(vel_cmd, "sampled_skill_id"):
                step["sampled_skill_id"] = vel_cmd.sampled_skill_id.clone()
        except Exception as exc:
            self._warn(
                "sampled_skill_id",
                f"Could not read vel_cmd.sampled_skill_id: {exc}",
            )

        try:
            traj_cmd = unwrapped.command_manager.get_term(self._traj_term_name)
            if hasattr(traj_cmd, "skill_id"):
                step["active_skill_id"] = traj_cmd.skill_id.clone()
            pending = getattr(traj_cmd, "pending", None)
            if pending is not None:
                # -1 when no pending entry is queued, else the queued skill idx.
                pid = torch.where(
                    pending.active,
                    pending.skill_id,
                    torch.full_like(pending.skill_id, -1),
                )
                step["pending_skill_id"] = pid.clone()

            # Cross-fade diagnostics: log alpha, the active mask, and the
            # pure-new / pure-old evaluations of y_des at the current
            # phase.  These let the plotter overlay (old, new, blended)
            # so the operator can visually verify the cross-fade math.
            transition = getattr(traj_cmd, "transition", None)
            if transition is not None:
                step["transition_active"] = transition.active.clone()
                step["transition_old_phase"] = transition.old_phase.clone()
                step["transition_old_traj_idx"] = transition.old_traj_idx.clone()
                blend_end = traj_cmd.cfg.transition_blend_end_phi
                if blend_end > 0.0:
                    all_envs = torch.arange(
                        unwrapped.num_envs, device=transition.active.device
                    )
                    full_alpha = transition.alpha(all_envs, blend_end)
                    # Mask out non-active envs with -1 so the plotter
                    # can hide them.
                    full_alpha = torch.where(
                        transition.active, full_alpha,
                        torch.full_like(full_alpha, -1.0),
                    )
                    step["transition_alpha"] = full_alpha.clone()

                    # Pure-new y_des = the cross-fade-aware helper called
                    # with alpha forced to 1 (the new traj only).  Same
                    # as ``manager.get_output`` for the current traj, but
                    # we route through the helper for a single shared
                    # eval pathway.
                    phase = traj_cmd.manager.phase
                    ones = torch.ones(
                        unwrapped.num_envs, device=phase.device
                    )
                    zeros = torch.zeros(
                        unwrapped.num_envs, device=phase.device
                    )
                    y_pos_new_only, _ = traj_cmd.compute_blended_outputs_at(
                        phase, all_envs, alpha_override=ones,
                    )
                    step["y_des_new_only"] = y_pos_new_only.clone()

                    # Pure-old y_des = blend with alpha=0.  For non-
                    # transitioning envs the helper returns the new
                    # eval (no transition active → no old to evaluate),
                    # so for those envs y_des_old_only == y_des_new_only
                    # by construction.  Operator should filter on
                    # ``transition_active`` when interpreting.
                    y_pos_old_only, _ = traj_cmd.compute_blended_outputs_at(
                        phase, all_envs, alpha_override=zeros,
                    )
                    step["y_des_old_only"] = y_pos_old_only.clone()
        except Exception as exc:
            self._warn(
                "active_pending_skill",
                f"Could not read traj_cmd skill_id / pending / transition: {exc}",
            )

        try:
            terrain = unwrapped.scene["terrain"]
            robot = unwrapped.scene.articulations["robot"]
            if hasattr(terrain, "world_xy_to_cell"):
                xy_w = _to_torch(robot.data.root_pos_w)[:, :2]
                rows, cols = terrain.world_xy_to_cell(xy_w)
                step["terrain_row"] = rows.clone()
                step["terrain_col"] = cols.clone()
        except Exception as exc:
            self._warn("terrain_rc", f"Could not compute terrain (row, col): {exc}")

        if self._log_world:
            self._collect_world(unwrapped, step, policy_obs)

        for key, val in step.items():
            self._data[key].append(val)

        self._step_count += 1

    # ------------------------------------------------------------------
    # World / replay block (opt-in; see ``log_world``)
    # ------------------------------------------------------------------
    def _collect_world(self, unwrapped, step: dict[str, torch.Tensor], policy_obs) -> None:
        """Add everything a 3-D replay needs, in WORLD coordinates.

        Kept in one method (and one flag) rather than sprinkled through
        :meth:`collect_step` so there is a single capture path feeding both the plots and
        the viewer -- a second recorder would drift from this one the first time the
        reference layout changed.

        Each sub-block is guarded independently: a scene without cameras, without a height
        scanner or without a generator still produces a usable bundle.

        Args:
            unwrapped: The unwrapped Isaac Lab env.
            step: This step's output dict, extended in place.
            policy_obs: The observation the env just returned, or ``None``.
        """
        robot = unwrapped.scene.articulations["robot"]
        try:
            step["root_pos_w"] = _to_torch(robot.data.root_pos_w).clone()
            step["root_quat_w"] = _to_torch(robot.data.root_quat_w).clone()   # xyzw
            step["root_lin_vel_w"] = _to_torch(robot.data.root_link_lin_vel_w).clone()
            step["root_ang_vel_w"] = _to_torch(robot.data.root_link_ang_vel_w).clone()
            step["joint_vel"] = _to_torch(robot.data.joint_vel).clone()
            step["projected_gravity"] = _to_torch(robot.data.projected_gravity_b).clone()
            if self._world_body_idx is not None:
                step["body_pos_w"] = _to_torch(robot.data.body_pos_w)[:, self._world_body_idx].clone()
        except Exception as exc:
            self._warn("world_robot", f"Could not read world robot state: {exc}")

        cmd = (unwrapped.command_manager.get_term(self._traj_term_name)
               if self._traj_term_name is not None else None)
        if cmd is not None:
            try:
                for name in ("clip_id", "start_step", "clip_world_pos", "clip_world_quat"):
                    value = getattr(cmd, name, None)
                    if isinstance(value, torch.Tensor):
                        step[name] = value.clone()
                if isinstance(getattr(cmd, "_step", None), torch.Tensor):
                    step["clip_step"] = cmd._step.clone()
            except Exception as exc:
                self._warn("world_clip", f"Could not read clip assignment: {exc}")

        # Exteroception, raw from the sensor (the observation-side copy travels inside
        # ``policy_obs`` below, carrying its noise and clipping).
        try:
            from isaaclab.envs.mdp import height_scan as _height_scan
            from isaaclab.managers import SceneEntityCfg

            if "height_scanner" in unwrapped.scene.sensors:
                step["height_scan_raw"] = _height_scan(
                    unwrapped, SceneEntityCfg("height_scanner")).clone()
                hits = _to_torch(unwrapped.scene.sensors["height_scanner"].data.ray_hits_w)
                step["scan_hits_w"] = hits.clone()
        except Exception as exc:
            self._warn("world_scan", f"Could not read the height scan: {exc}")

        if self._has_depth:
            try:
                from robot_rl.tasks.manager_based.robot_rl.mdp.observations.depth import (
                    depth_obs_stack,
                )
                step["depth_raw"] = depth_obs_stack(unwrapped).clone()
            except Exception as exc:
                self._warn("world_depth", f"Could not read the depth cameras: {exc}")

        # The generator's predicted horizon, in world coordinates (not the y_des layout --
        # the viewer poses a robot from it, so it wants root pose + joints directly).
        if cmd is not None and getattr(cmd, "fm_enabled", False):
            try:
                step["fm_pred_root_pos"] = cmd._pred_root_pos.clone()
                step["fm_pred_root_quat"] = cmd._pred_root_quat.clone()
                step["fm_pred_joint_pos"] = cmd._pred_jp[..., cmd._fm_to_robot_jperm].clone()
                step["fm_pred_frame_pos"] = cmd._pred_frame_pos.clone()
            except Exception as exc:
                self._warn("world_fm_pred", f"Could not read the FM prediction window: {exc}")

        self._collect_reference_windows(unwrapped, cmd, step, policy_obs)

    def _collect_reference_windows(self, unwrapped, cmd, step, policy_obs) -> None:
        """Lift the reference window back to world -- per source, and as OBSERVED.

        Three variants, and the differences between them are the point:

        * ``refwin_<src>_*`` -- each source's own look-ahead, straight from the command;
        * ``refwin_obs_*`` -- the APPLIED source's window round-tripped through
          :func:`~robot_rl...observations.reference_window` and back, which must reproduce
          the applied one exactly (it is the same geometry, so a mismatch is a frame bug);
        * ``refwin_noisy_*`` -- the same, but reconstructed from the observation the POLICY
          actually received, so it carries the noise and the +-20 clip. This is the only one
          that shows what the encoder was really fed.

        The scale is divided back out before reconstructing; noise and clipping are not
        invertible and are not meant to be.
        """
        if cmd is None or not hasattr(cmd, "get_future_world_base"):
            return
        nj = len(cmd.robot.joint_names)
        frames = cmd.resolve_ee_frames(None)
        ns, st = self._ref_num_steps, self._ref_stride

        def _capture(prefix: str, available: tuple[str, ...] | None = None) -> None:
            """Store the served source's look-ahead window in world coordinates.

            Sample 0 is the ANCHORED current-step reference -- what the policy observes --
            which is NOT the same thing as ``y_des``'s base channels on the clip path:
            those are re-pointed at the GLOBALLY-placed clip by
            ``MotionSequenceCommand._write_global_base_des`` so the reward and the drift
            termination grade the same datum. Both are worth having; conflating them makes
            accumulated global drift look like a reference error.

            Args:
                prefix: Key prefix naming the source.
                available: Window blocks this source can actually serve, or ``None`` for
                    "all of them". The GENERATOR is limited to the blocks it was trained
                    to predict, and its accessors raise rather than invent the rest -- so
                    asking for an ee frame from a generator that predicts none would
                    abort the whole capture, including the clip source's.
            """
            def has(block: str) -> bool:
                """Whether this source carries ``block``."""
                return available is None or block in available

            offs = torch.arange(ns, device=cmd.device, dtype=torch.long) * st
            pos_w, quat_w, _, _ = cmd.get_future_world_base(offs)
            step[f"refwin_{prefix}_base_pos"] = pos_w.clone()
            step[f"refwin_{prefix}_base_quat"] = quat_w.clone()
            # [N, n_offsets, n_frames, 13] -- offset first, matching the layout
            # ``reference_window_to_world`` returns, so the two are directly comparable.
            if has("ee_pos"):
                ee = cmd.get_future_world_frames(frames, offs)
                step[f"refwin_{prefix}_ee_pos"] = ee[..., 0:3].clone()
            if has("joint_pos"):
                # Positions only -- the velocity half comes back NaN on a window without
                # joint_vel, and this is what the viewer poses its ghost from.
                joint_pos, _ = cmd.get_future_joint_state(offs)      # robot joint order
                step[f"refwin_{prefix}_joint_pos"] = joint_pos.clone()

        try:
            if hasattr(cmd, "as_clip_reference"):
                managers = {"clip": cmd.as_clip_reference, "fm": cmd.as_fm_reference}
                for src in (["clip", "fm"] if getattr(cmd, "fm_enabled", False) else ["clip"]):
                    with managers[src]():
                        # The clip carries the whole reference; the generator carries only
                        # the blocks it predicts (== the observation term's, checked at
                        # load), so that is what it is asked for.
                        _capture(src, self._ref_blocks if src == "fm" else None)
            else:
                _capture("applied")        # a plain clip command: one source, no swapping
        except Exception as exc:
            self._warn("refwin_sources", f"Could not capture the per-source windows: {exc}")

        # The clean window round-tripped: identical to the applied source by construction,
        # so any disagreement localises a frame bug to reference_window itself.
        try:
            root_pos = _to_torch(cmd.robot.data.root_pos_w)
            root_quat = _to_torch(cmd.robot.data.root_quat_w)
            # The term's OWN block selection on both halves: build the clean window the
            # way the policy's term does, and take it apart the same way. Defaulting to
            # all five here would both mis-size the window and invent channels the policy
            # never saw.
            blocks = self._ref_blocks
            clean = reference_window(unwrapped, self._traj_term_name, num_steps=ns, stride=st,
                                     blocks=blocks)
            for tag, win in (("obs", clean), ("noisy", self._observed_window(unwrapped, policy_obs))):
                if win is None:
                    continue
                world = reference_window_to_world(win, root_pos, root_quat, ns, nj, len(frames),
                                                  blocks)
                # Only what the selection carries -- a dropped block is absent from
                # ``world`` rather than zero, so this logs nothing instead of a zero the
                # plots would draw as a reference sitting at the origin.
                for key, name in (("base_pos_w", "base_pos"), ("base_quat_w", "base_quat"),
                                  ("ee_pos_w", "ee_pos"), ("joint_pos", "joint_pos")):
                    if key in world:
                        step[f"refwin_{tag}_{name}"] = world[key].clone()
        except Exception as exc:
            self._warn("refwin_obs", f"Could not lift the reference window to world: {exc}")

    def _observed_window(self, unwrapped, policy_obs) -> torch.Tensor | None:
        """The ``reference_window`` slice of ``policy_obs``, with the term scale undone.

        Returns the raw-unit window the policy was fed, still carrying its noise and
        clipping, or ``None`` when no observation was supplied.
        """
        if policy_obs is None or self._refwin_obs is None:
            return None
        group, sl = self._refwin_obs
        tensor = _group_tensor(policy_obs, group)
        if tensor is None:
            return None
        win = tensor[:, sl].float()
        return win if self._refwin_scale is None else win / self._refwin_scale

    def finalize(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Stack all timestep data into numpy arrays.

        Returns:
            Tuple of ``(data, metadata)`` where data values have shape
            ``[T, ...]`` (e.g. ``[T, N_envs, n_dims]``).
        """
        data: dict[str, np.ndarray] = {}
        for key, tensors in self._data.items():
            stacked = torch.stack(tensors, dim=0)
            data[key] = stacked.cpu().numpy()

        # Build the "successful tracking" mask: True for logged steps that
        # belong to an episode that ended via time_out (truncation), False for
        # steps in early-terminated (fallen) episodes and the trailing
        # still-running episode. Lets stats.txt report tracking quality
        # conditioned on not having failed. Only built when per-step
        # termination signals were captured for every step.
        if self._dones and len(self._dones) == self._step_count:
            dones = torch.stack(self._dones, dim=0).cpu().numpy().astype(bool)
            time_outs = torch.stack(self._time_outs, dim=0).cpu().numpy().astype(bool)
            data["_timeout_mask"] = _build_timeout_mask(dones, time_outs)
            # The raw signals too: a diagnostic wants to mark WHERE an episode ended, not
            # only whether the steps before it belonged to a successful one.
            data["dones"] = dones
            data["time_outs"] = time_outs

        # Note: per-trajectory CLF/tracking + termination summaries now live in
        # a dedicated headless eval, which would have far better clip coverage than an
        # interactive play rollout. This logger only collects the per-step
        # ``traj_idx``/``v`` series used by the play plots.

        return data, self.metadata

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collect_metadata(self, env) -> None:
        """One-time capture of axis names, joint names, dt, etc."""
        unwrapped = env.unwrapped
        self.metadata["dt"] = unwrapped.step_dt

        # Discover trajectory command term
        self._traj_term_name: str | None = None
        active = unwrapped.command_manager.active_terms
        for name in _TRAJ_TERM_NAMES:
            if name in active:
                self._traj_term_name = name
                break

        # Multiskill trajectory metadata for per-trajectory summary.
        self._num_trajectories: int | None = None
        self._traj_names: list[str] | None = None

        if self._traj_term_name is not None:
            try:
                ref = unwrapped.command_manager.get_term(self._traj_term_name)
                if hasattr(ref, "ordered_pos_output_names"):
                    self.metadata["pos_names"] = list(ref.ordered_pos_output_names)
                if hasattr(ref, "ordered_vel_output_names"):
                    self.metadata["vel_names"] = list(ref.ordered_vel_output_names)
                if hasattr(ref, "ref_frames"):
                    self.metadata["ref_frames"] = list(ref.ref_frames)
                if hasattr(ref, "resolve_ee_frames"):  # ee_ref obs column labels
                    self.metadata["ee_frames"] = list(ref.resolve_ee_frames())
                # Flow-matching prediction-window geometry (generator command only):
                # ``fm_window_len`` = F steps stored per inference, ``fm_novel_start``
                # = window index N where the novel future begins.  Lets the
                # peel-off plot align each prediction's window steps to sim time.
                if hasattr(ref, "fm_window_len") and hasattr(ref, "fm_novel_start"):
                    self.metadata["fm_window_len"] = int(ref.fm_window_len)
                    self.metadata["fm_novel_start"] = int(ref.fm_novel_start)
                # Which source drives the robot -- lets the positions/velocities plots
                # label the applied line, and tells the viewer which overlay is the one
                # the policy actually tracked.
                cmd_cfg = getattr(ref, "cfg", None)
                if cmd_cfg is not None and hasattr(cmd_cfg, "reference_source"):
                    self.metadata["reference_source"] = str(cmd_cfg.reference_source)
                    self.metadata["drive_with_clip"] = cmd_cfg.reference_source == "clip"
                    self.metadata["fm_update_every"] = int(getattr(cmd_cfg, "fm_update_every", 1))
                if getattr(ref, "fm_enabled", False):
                    ctx = ref.ctx
                    self.metadata["fm_horizon"] = int(ctx.F)
                    self.metadata["fm_frame_names"] = list(ctx.frame_names)
                    self.metadata["fm_num_history"] = int(ctx.N)
                    self.metadata["fm_extero_source"] = str(ctx.extero_source)
                manager = getattr(ref, "manager", None)
                if manager is not None and hasattr(manager, "num_trajectories"):
                    self._num_trajectories = int(manager.num_trajectories)
                    self._traj_names = self._collect_traj_names(manager)
                    self.metadata["num_trajectories"] = self._num_trajectories
                    if self._traj_names is not None:
                        self.metadata["per_traj_names"] = list(self._traj_names)
            except Exception as exc:
                print(f"[WARN DataLogger] Could not read trajectory metadata: {exc}")
        else:
            print(f"[WARN DataLogger] No trajectory command term found among {active}. "
                  f"Trajectory-related data will not be logged.")

        # Reference-window shape: read from whichever obs group carries the term (the
        # actor's reference group, or the critic if the reference group was dropped) so
        # the logged obs match exactly what the policy is fed.
        try:
            obs_mgr = unwrapped.observation_manager
            for group_name, group_terms in obs_mgr.active_terms.items():
                if "reference_window" in group_terms:
                    cfg = obs_mgr._group_obs_term_cfgs[group_name][
                        list(group_terms).index("reference_window")
                    ]
                    self._ref_num_steps = int(cfg.params.get("num_steps", self._ref_num_steps))
                    self._ref_stride = int(cfg.params.get("stride", self._ref_stride))
                    blocks = cfg.params.get("blocks")
                    self._ref_blocks = tuple(blocks) if blocks else None
                    break
            self.metadata["reference_num_steps"] = self._ref_num_steps
            self.metadata["future_stride"] = self._ref_stride
            self.metadata["reference_blocks"] = list(
                self._ref_blocks) if self._ref_blocks else "all"
        except Exception as exc:
            print(f"[WARN DataLogger] Could not read reference-window params: {exc}")

        # Joint names
        try:
            robot = unwrapped.scene.articulations["robot"]
            self.metadata["joint_names"] = list(robot.data.joint_names)
        except Exception as exc:
            print(f"[WARN DataLogger] Could not read joint names: {exc}")

        # Skill + terrain metadata (only populated for envs that use
        # ``MultiskillVelocityTrackingCommand`` + ``MetaTerrainImporter``).
        # ``_skill_list`` is private but it's the canonical ordered list the
        # command pre-builds; no public alias exists today.
        try:
            vel_cmd = unwrapped.command_manager.get_term("base_velocity")
            if hasattr(vel_cmd, "_skill_list"):
                self.metadata["skill_list"] = list(vel_cmd._skill_list)
        except Exception as exc:
            print(f"[WARN DataLogger] Could not read vel command skill_list: {exc}")

        try:
            terrain = unwrapped.scene["terrain"]
            if hasattr(terrain, "terrain_origins") and terrain.terrain_origins is not None:
                num_rows, num_cols = terrain.terrain_origins.shape[:2]
                self.metadata["terrain_num_rows"] = int(num_rows)
                self.metadata["terrain_num_cols"] = int(num_cols)
            if hasattr(terrain, "_spawnable_columns") and terrain._spawnable_columns is not None:
                spawn_cols = set(terrain._spawnable_columns.cpu().tolist())
                border_cols = sorted(
                    c for c in range(self.metadata.get("terrain_num_cols", 0))
                    if c not in spawn_cols
                )
                self.metadata["border_cols"] = border_cols
        except Exception as exc:
            print(f"[WARN DataLogger] Could not read terrain grid metadata: {exc}")

        self._collect_world_metadata(unwrapped)

    def _collect_world_metadata(self, unwrapped) -> None:
        """Resolve everything :meth:`_collect_world` needs once, up front.

        Always runs (the attributes must exist even when ``log_world`` is off), but only
        the world block reads them.
        """
        self._world_body_idx: torch.Tensor | None = None
        self._world_body_names: list[str] = []
        self._has_depth = False
        self._refwin_obs: tuple[str, slice] | None = None
        self._refwin_scale: torch.Tensor | None = None
        if not self._log_world:
            return
        try:
            robot = unwrapped.scene.articulations["robot"]
            cmd = unwrapped.command_manager.get_term(self._traj_term_name)
            # The base plus the ee reference frames -- the same set the reference window
            # carries, so measured and reference are directly comparable per frame.
            names = [cmd.cfg.base_frame_name] + list(cmd.resolve_ee_frames(None))
            self._world_body_names = [n for n in names if n in robot.body_names]
            self._world_body_idx = torch.tensor(
                [robot.body_names.index(n) for n in self._world_body_names],
                dtype=torch.long, device=unwrapped.device,
            )
            self.metadata["world_body_names"] = list(self._world_body_names)
            self.metadata["base_frame_name"] = str(cmd.cfg.base_frame_name)
        except Exception as exc:
            print(f"[WARN DataLogger] Could not resolve world body frames: {exc}")

        obs_names = set()
        for names in unwrapped.observation_manager.active_terms.values():
            obs_names.update(names)
        self._has_depth = bool({"depth_near", "depth_far"} & obs_names)
        if self._has_depth:
            try:
                from robot_rl.tasks.manager_based.robot_rl.mdp.observations.depth import (
                    depth_camera_specs,
                )
                # The whole camera contract, so the viewer can reshape the flat token axis
                # back into images (and denormalise them) without restating any geometry.
                self.metadata["depth_camera_spec"] = depth_camera_specs(unwrapped)
            except Exception as exc:
                print(f"[WARN DataLogger] Could not read the depth camera spec: {exc}")

        # Where the reference window sits in the observation the policy receives, and the
        # per-channel scale to divide back out.
        try:
            self._refwin_obs = _obs_term_slice(unwrapped, "reference_window")
            if self._refwin_obs is not None:
                group, _ = self._refwin_obs
                om = unwrapped.observation_manager
                names = list(om.active_terms[group])
                cfg = om._group_obs_term_cfgs[group][names.index("reference_window")]
                if cfg.scale is not None:
                    self._refwin_scale = torch.as_tensor(
                        cfg.scale, dtype=torch.float32, device=unwrapped.device)
                self.metadata["reference_window_group"] = group
        except Exception as exc:
            print(f"[WARN DataLogger] Could not locate the reference_window observation: {exc}")

        # Static scene geometry the viewer rebuilds the terrain from.
        try:
            importer = getattr(unwrapped.scene, "terrain", None)
            cmd = unwrapped.command_manager.get_term(self._traj_term_name)
            if importer is not None and hasattr(importer, "env_terrain_id"):
                self.metadata["env_terrain_id"] = importer.env_terrain_id.cpu().tolist()
                self.metadata["env_cell_corner"] = importer.env_cell_corner.cpu().tolist()
            lib = getattr(cmd, "_clip_lib", None)
            if lib is not None:
                self.metadata["terrain_manifest"] = lib.attrs.get("terrain_manifest", [])
                self.metadata["clip_len"] = cmd.manager.clip_len.cpu().tolist()
        except Exception as exc:
            print(f"[WARN DataLogger] Could not read terrain / clip geometry: {exc}")

    def _collect_traj_names(self, manager) -> list[str] | None:
        """Build a per-trajectory display name list from the multiskill manager.

        Format: ``"<skill>/vx=<vel_x>"`` so trajectories that share a YAML
        name (very common) remain distinguishable. Falls back to
        ``"<skill>/<idx>"`` or ``"traj_<idx>"`` when conditioning data is
        missing.
        """
        T = int(getattr(manager, "num_trajectories", 0))
        if T <= 0:
            return None

        skills = list(getattr(manager, "skill_labels", []) or [])
        cond = getattr(manager, "_global_conditioning", None)
        vel_x: list[float] | None = None
        if cond is not None:
            try:
                vel_x = cond[:, 0].detach().cpu().tolist()
            except Exception:
                vel_x = None

        names: list[str] = []
        for i in range(T):
            skill = skills[i] if i < len(skills) else "?"
            if vel_x is not None:
                names.append(f"{skill}/vx={vel_x[i]:.2f}")
            else:
                names.append(f"{skill}/{i}")
        return names

    def _warn(self, source: str, msg: str) -> None:
        """Print a warning once per source key."""
        if source not in self._warned_sources:
            print(f"[WARN DataLogger] {msg}")
            self._warned_sources.add(source)


def _to_torch(data: Any) -> torch.Tensor:
    """Convert data to a torch tensor, handling warp arrays."""
    if isinstance(data, torch.Tensor):
        return data
    try:
        import warp as wp
        return wp.to_torch(data)
    except (ImportError, TypeError):
        return torch.as_tensor(data)


def _build_timeout_mask(dones: np.ndarray, time_outs: np.ndarray) -> np.ndarray:
    """Mark each logged step as belonging to a time_out (successful) episode.

    A ``done`` at step ``t`` means the env reset *during* step ``t``; Isaac Lab
    recomputes the command (and hence ``v``, ``y_act``, ...) after the in-step
    reset, so the data logged at step ``t`` for that env already belongs to the
    *next* episode. A logged step ``s`` therefore belongs to the episode that
    ends at the first ``done`` strictly after ``s``, and is "successful" iff
    that terminating ``done`` was a ``time_out``. Steps with no later ``done``
    (the trailing in-progress episode) are marked False. This boundary
    convention never leaks a fallen episode's divergence into the success set.

    Args:
        dones: ``[T, N]`` bool array, True where env ``n`` reset at step ``t``.
        time_outs: ``[T, N]`` bool array, True where that reset was a time_out.

    Returns:
        ``[T, N]`` bool mask, True for steps in successful (time_out) episodes.
    """
    num_steps, num_envs = dones.shape
    mask = np.zeros((num_steps, num_envs), dtype=bool)
    pending = np.zeros(num_envs, dtype=bool)      # time_out-ness of nearest done at idx > s
    has_pending = np.zeros(num_envs, dtype=bool)  # whether any done exists at idx > s
    for s in range(num_steps - 1, -1, -1):
        mask[s] = has_pending & pending
        d = dones[s]
        pending = np.where(d, time_outs[s], pending)
        has_pending = has_pending | d
    return mask
