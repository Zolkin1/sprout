# Copyright 2026 Zachary Olkin. All rights reserved.

from __future__ import annotations

import torch
import warp as wp
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    quat_box_minus,
    quat_conjugate,
    quat_mul,
    wrap_to_pi,
)

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


def base_pose_error_termination(
    env,
    command_name: str = "traj_ref",
    max_pos_dist: float = 0.25,
    max_ori_rad: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the base pose drifts too far from the clip reference.

    The single failure termination for open-loop motion-sequence tracking
    (replaces the pelvis-upright / base-height / frame-drift terminations).
    Compares the robot's world base pose against the clip's reference base
    pose (from :meth:`MotionSequenceCommand.reference_world_state`):

      * position: terminate if ``||ref_pos - root_pos|| > max_pos_dist`` (m).
      * orientation: terminate if the geodesic angle between the reference
        and actual base quaternions exceeds ``max_ori_rad`` (rad).

    Args:
        env: IsaacLab environment.
        command_name: The :class:`MotionSequenceCommand` term name.
        max_pos_dist: Max base position error before terminating (m).
        max_ori_rad: Max base orientation error before terminating (rad).
        asset_cfg: Robot scene entity.

    Returns:
        Boolean ``[num_envs]`` -- True for envs that should terminate.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    state = cmd.reference_world_state()
    ref_pos_w = state[:, 0:3]
    ref_quat_w = state[:, 3:7]                      # wxyz
    root_pos_w = wp.to_torch(asset.data.root_pos_w)
    root_quat_w = wp.to_torch(asset.data.root_quat_w)  # wxyz

    pos_err = torch.norm(ref_pos_w - root_pos_w, dim=-1)
    # ||quat_box_minus|| is the geodesic rotation angle (rad).
    ori_err = torch.norm(quat_box_minus(root_quat_w, ref_quat_w), dim=-1)
    return (pos_err > max_pos_dist) | (ori_err > max_ori_rad)


def clip_end_reached(env, command_name: str = "traj_ref") -> torch.Tensor:
    """Truncate (good termination) when the clip's last step is reached.

    Wire this as ``DoneTerm(..., time_out=True)`` so RSL-RL bootstraps the
    value from the final state rather than treating clip-end as a failure.
    Keyed on the clip step counter so it is correct even when episodes start
    mid-clip (``random_start_phase``).

    Args:
        env: IsaacLab environment.
        command_name: The :class:`MotionSequenceCommand` term name.

    Returns:
        Boolean ``[num_envs]`` -- True for envs at/after the clip's last step.
    """
    cmd = env.command_manager.get_term(command_name)
    last_step = cmd.manager.num_steps - 1
    return (cmd.start_step + env.episode_length_buf) >= last_step

def no_progress(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Terminates the episode early if the robot is not making enough progress
    compared to expected distance at current time step.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")

    # Distance traveled from starting point
    root_pos = wp.to_torch(asset.data.root_pos_w)[:, :2]
    origin = env.scene.env_origins[:, :2]
    distance = torch.norm(root_pos - origin, dim=1)

    # Expected distance so far = commanded_speed * time_elapsed
    commanded_speed = torch.norm(command[:, :2], dim=1)
    elapsed_time = env.episode_length_buf * env.step_dt  # [num_envs]
    expected_distance = commanded_speed * elapsed_time

    # Flag for insufficient progress
    behind_schedule = distance < (0.5 * expected_distance)

    # Optional: only trigger after a minimum time has passed (e.g., 30% of episode)
    enough_time_passed = env.episode_length_buf > (0.5 * env.max_episode_length)
    no_progress_flag = behind_schedule & enough_time_passed

    return no_progress_flag

def base_orientation(env, cmd_name: str, roll_limit_deg: float = 30.0, pitch_limit_deg: float = 30.0,
                     base_link: str = "pelvis_link",
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the base roll/pitch deviates from the reference by more than the limits.

    The reference orientation is stored as a quaternion ``(x, y, z, w)`` inside ``cmd.y_des``.
    Error is taken as the world-frame relative quaternion ``q_act * q_ref^-1`` and decomposed
    into XYZ Euler angles to recover proper roll/pitch.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(cmd_name)
    ref_traj = cmd.y_des
    output_names = cmd.ordered_pos_output_names

    ori_idx = [output_names.index(f"{base_link}:ori_{a}") for a in ("x", "y", "z", "w")]
    ref_quat = ref_traj[:, ori_idx]
    act_quat = wp.to_torch(asset.data.root_quat_w)  # already (x, y, z, w)

    q_err = quat_mul(act_quat, quat_conjugate(ref_quat))
    roll_err, pitch_err, _ = euler_xyz_from_quat(q_err, wrap_to_2pi=False)
    roll_err = wrap_to_pi(roll_err)
    pitch_err = wrap_to_pi(pitch_err)

    roll_limit = torch.deg2rad(torch.tensor(roll_limit_deg, device=q_err.device))
    pitch_limit = torch.deg2rad(torch.tensor(pitch_limit_deg, device=q_err.device))

    return (roll_err.abs() > roll_limit) | (pitch_err.abs() > pitch_limit)


def base_orientation_from_upright(
    env,
    roll_limit_deg: float = 70.0,
    pitch_limit_deg: float = 70.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the base roll or pitch deviates from upright (world z) by more than the limits.

    Unlike :func:`base_orientation`, the reference here is the world-frame
    upright (identity rotation) — yaw is ignored.  Useful as a face-plant /
    fallen-robot detector that is independent of the trajectory reference,
    and therefore not subject to whatever grace gating the deviation-from-
    reference termination uses.

    Args:
        env: IsaacLab environment.
        roll_limit_deg: Maximum allowed |roll| from upright in degrees.
        pitch_limit_deg: Maximum allowed |pitch| from upright in degrees.
        asset_cfg: Robot scene entity.

    Returns:
        Boolean tensor of shape ``[num_envs]`` — ``True`` for envs that
        should terminate this step.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    act_quat = wp.to_torch(asset.data.root_quat_w)  # (x, y, z, w)
    roll, pitch, _ = euler_xyz_from_quat(act_quat, wrap_to_2pi=False)
    roll = wrap_to_pi(roll)
    pitch = wrap_to_pi(pitch)

    roll_limit = torch.deg2rad(torch.tensor(roll_limit_deg, device=act_quat.device))
    pitch_limit = torch.deg2rad(torch.tensor(pitch_limit_deg, device=act_quat.device))
    return (roll.abs() > roll_limit) | (pitch.abs() > pitch_limit)


def base_fallen(
    env,
    min_height: float = 0.5,
    roll_limit_deg: float = 70.0,
    pitch_limit_deg: float = 70.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the base has fallen: too low, or tilted too far from upright.

    A trajectory-independent fall detector: fires when the base world height
    drops below ``min_height`` (m) **or** the base roll/pitch deviates from
    world upright (world z) by more than ``roll_limit_deg`` / ``pitch_limit_deg``
    (yaw ignored).  The orientation half reuses
    :func:`base_orientation_from_upright`, so the two share one definition.

    Independent of the tracking reference (unlike
    :func:`base_pose_error_termination`), so it catches a collapsed / face-
    planted robot regardless of where the reference is.  On flat ground the
    base world ``z`` is the height above the ground plane.

    Args:
        env: IsaacLab environment.
        min_height: Minimum base world height before terminating (m).
        roll_limit_deg: Maximum allowed |roll| from upright (deg).
        pitch_limit_deg: Maximum allowed |pitch| from upright (deg).
        asset_cfg: Robot scene entity.

    Returns:
        Boolean ``[num_envs]`` -- True for envs that should terminate.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    too_low = wp.to_torch(asset.data.root_pos_w)[:, 2] < min_height
    tilted = base_orientation_from_upright(env, roll_limit_deg, pitch_limit_deg, asset_cfg)
    return too_low | tilted


def frame_deviation_from_reference(
    env,
    cmd_name: str,
    frame_names: list[str],
    max_frac: float,
    min_dist: float = 0.0,
    grace_period_s: float = 0.0,
    debug: bool = False,
) -> torch.Tensor:
    """Terminate when any listed frame deviates from its reference by more
    than ``max(max_frac * chord, min_dist)``.

    For each frame the deviation is the (ref-frame-local) position-error
    norm ``||y_act_xyz - y_des_xyz||``.  The reference scale is the
    **current domain's** Bezier chord length per frame
    (``||last_cp - first_cp||``) — the same scalar the debug-viz sphere
    uses.  Because stance frames have chord ≈ 0 in their stance domain,
    ``min_dist`` (metres) sets an absolute floor on the threshold so
    those frames don't terminate trivially.  The effective threshold per
    env per frame is ``max(max_frac * chord, min_dist)``.

    Args:
        env: The IsaacLab environment.
        cmd_name: Trajectory command term exposing ``y_act``, ``y_des``,
            ``ordered_pos_output_names``, and a multiskill ``manager``.
        frame_names: Frames to monitor.  Each must have
            ``{frame_name}:pos_{x,y,z}`` in the command's
            ``ordered_pos_output_names``.
        max_frac: Allowed fractional deviation (e.g. ``0.5`` for 50%).
        min_dist: Absolute minimum threshold in metres.  Default ``0.0``
            disables the floor — set this above the noise floor / typical
            stance-foot drift to avoid spurious termination.
        grace_period_s: Seconds after an in-episode trajectory swap
            (within-skill bucket swap or full skill change) during which this
            termination is suppressed.  Fresh episode resets are NOT granted
            grace — the reset spawns the robot on the reference, so there is no
            transition transient and the termination arms immediately.  Reads
            ``cmd.time_since_traj_change_s`` (populated by
            ``BatchedMultiSkillCommand._pre_update_phase``).  Default 0
            disables the grace.  Silently no-ops if the command term does
            not expose ``time_since_traj_change_s`` (e.g. single-skill
            commands).
        debug: If True, print per-frame ``(err_x, err_y, err_z, ||err||,
            threshold)`` for every env that terminates this step.  Off by
            default — enable from the cfg via ``params={"debug": True}``.

    Returns:
        Boolean tensor of shape ``[num_envs]`` — ``True`` for envs that
        should terminate this step.

    Raises:
        ValueError: If any name in ``frame_names`` lacks a complete
            ``pos_{x,y,z}`` triplet in the command's outputs.
    """
    cmd = env.command_manager.get_term(cmd_name)
    output_names = cmd.ordered_pos_output_names

    pos_indices: list[list[int]] = []
    missing: list[str] = []
    for name in frame_names:
        try:
            xyz = [output_names.index(f"{name}:pos_{a}") for a in ("x", "y", "z")]
        except ValueError:
            missing.append(name)
            continue
        pos_indices.append(xyz)
    if missing:
        available = sorted({
            n.split(":", 1)[0] for n in output_names
            if ":pos_" in n and not n.startswith("joint:")
        })
        raise ValueError(
            f"Frames {missing} do not have a full ``pos_{{x,y,z}}`` triplet "
            f"in '{cmd_name}'.ordered_pos_output_names. "
            f"Frames with a pos triplet available: {available}."
        )

    device = cmd.y_des.device
    pos_idx_t = torch.tensor(pos_indices, dtype=torch.long, device=device)    # (F, 3)

    # Position error per frame (ref-frame-local coords).
    err = (cmd.y_act - cmd.y_des)[:, pos_idx_t]                                # (N, F, 3)
    err_norm = torch.linalg.norm(err, dim=-1)                                  # (N, F)

    # Current-target chord per frame.  Prefer the cmd's accessor when it
    # exists — for ``BatchedMultiSkillCommand`` that returns an
    # alpha-blended chord during a cross-fade transition, keeping the
    # scale consistent with the (likewise blended) ``y_des`` the numerator
    # is measured against.  Fallback (single-skill commands without the
    # accessor) computes the raw current-domain chord on the new traj.
    if hasattr(cmd, "current_chord_per_frame"):
        chord_len = cmd.current_chord_per_frame(pos_idx_t)                     # (N, F)
    else:
        manager = cmd.manager
        traj_idx = manager.get_current_trajectory_indices()
        domain_idx = manager._get_domain_indices(manager.phase, traj_idx)
        coeffs_pos = manager.data["coeffs_pos"][traj_idx, domain_idx]
        frame_coeffs = coeffs_pos[:, pos_idx_t, :]
        chord_len = torch.linalg.norm(
            frame_coeffs[..., -1] - frame_coeffs[..., 0], dim=-1
        )                                                                       # (N, F)

    threshold = torch.clamp(max_frac * chord_len, min=min_dist)                # (N, F)
    terminate = (err_norm > threshold).any(dim=-1)                             # (N,)

    if grace_period_s > 0.0:
        time_since = getattr(cmd, "time_since_traj_change_s", None)
        if time_since is not None:
            terminate = terminate & (time_since >= grace_period_s)

    if debug and terminate.any():
        bad_envs = terminate.nonzero(as_tuple=False).flatten().tolist()
        for env_id in bad_envs:
            parts = []
            for f, name in enumerate(frame_names):
                ex, ey, ez = err[env_id, f].tolist()
                n = err_norm[env_id, f].item()
                t = threshold[env_id, f].item()
                marker = "!!" if n > t else "  "
                parts.append(
                    f"{marker} {name}: err=({ex:+.3f}, {ey:+.3f}, {ez:+.3f}) "
                    f"||err||={n:.3f} thresh={t:.3f}"
                )
            print(f"[frame_dev term] env {env_id}: " + " | ".join(parts))

    return terminate


def frame_drift_in_skill(
    env,
    skill_name: str,
    cmd_name: str = "traj_ref",
    base_term: str = "frame_drift",
) -> torch.Tensor:
    """Subset of ``base_term``'s frame-drift done, masked to envs whose active
    skill is ``skill_name`` — purely for a per-skill breakdown in the logs.

    IsaacLab's ``TerminationManager`` logs every term as
    ``Episode_Termination/<name>`` (the fraction of envs whose most recent
    episode ended via that term), so registering one of these per skill yields
    ``Episode_Termination/frame_drift_<skill>`` series that partition the parent
    ``frame_drift`` fraction. Because each is an exact subset of ``base_term``
    (already an active termination term), the union of all done signals — and
    therefore the training dynamics — is unchanged.

    The already-computed ``base_term`` done is read via the manager's public
    ``get_term`` accessor, so the underlying frame-drift logic (including its
    grace period) is not recomputed. This requires ``base_term`` to be computed
    earlier in the step than these per-skill terms, which holds as long as it is
    registered before them in the terminations cfg.

    Args:
        env: The IsaacLab environment.
        skill_name: Skill whose frame-drift terminations to isolate. Must be
            present in the command's ``_skill_list``.
        cmd_name: Trajectory command term exposing ``skill_id`` and
            ``_skill_list`` (the multiskill command).
        base_term: Name of the parent frame-drift termination term.

    Returns:
        Boolean tensor of shape ``[num_envs]`` — ``True`` for envs that
        terminated via ``base_term`` this step while ``skill_name`` was active.
    """
    # Safe no-op if the parent term is not active (e.g. disabled in a subclass).
    if base_term not in env.termination_manager.active_terms:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    drift = env.termination_manager.get_term(base_term)                       # (N,) bool
    cmd = env.command_manager.get_term(cmd_name)
    skill_idx = cmd._skill_list.index(skill_name)
    return drift & (cmd.skill_id == skill_idx)


def frame_drift_post_grace(
    env,
    cmd_name: str = "traj_ref",
    window_s: float = 0.25,
    base_term: str = "frame_drift",
    grace_period_s: float | None = None,
) -> torch.Tensor:
    """Subset of ``base_term``'s frame-drift done, masked to envs that
    terminated within ``window_s`` seconds *after* the grace period ended.

    A frame-drift termination this soon after a trajectory swap's grace window
    expires is strong evidence the transition itself failed (it just took
    slightly longer than the grace period to blow up), rather than steady-state
    tracking drift. Logged by the ``TerminationManager`` as
    ``Episode_Termination/frame_drift_post_grace``; compare against
    ``Episode_Termination/frame_drift`` for the suspected-transition-failure
    fraction.

    Like :func:`frame_drift_in_skill` this is an exact subset of ``base_term``
    (an already-active term) read via ``get_term``, so the union of all done
    signals — and therefore training dynamics — is unchanged, and the grace
    logic is not recomputed. This requires ``base_term`` to be computed earlier
    in the step than this term, which holds as long as it is registered before
    it in the terminations cfg.

    The window is ``[grace_period_s, grace_period_s + window_s)`` in seconds,
    measured by ``cmd.time_since_traj_change_s`` (reset to 0 on an in-episode
    trajectory swap, 1e6 on a fresh reset — so envs that never transitioned are
    naturally excluded). The lower bound is redundant while ``frame_drift`` can
    only fire after grace, but is kept explicit so the metric stays correct if
    the parent gate ever changes.

    Args:
        env: The IsaacLab environment.
        cmd_name: Trajectory command term exposing ``time_since_traj_change_s``
            (the multiskill command).
        window_s: Width (seconds) of the post-grace window to flag.
        base_term: Name of the parent frame-drift termination term.
        grace_period_s: Grace window (seconds). Defaults to ``None``, in which
            case it is read from ``base_term``'s own ``grace_period_s`` param via
            ``get_term_cfg`` so it tracks the parent automatically.

    Returns:
        Boolean tensor of shape ``[num_envs]`` — ``True`` for envs that
        terminated via ``base_term`` this step within the post-grace window.
    """
    # Safe no-op if the parent term is not active (e.g. disabled in a subclass).
    if base_term not in env.termination_manager.active_terms:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    drift = env.termination_manager.get_term(base_term)                       # (N,) bool
    cmd = env.command_manager.get_term(cmd_name)
    # Silent no-op for command terms without a transition clock (single-skill).
    time_since = getattr(cmd, "time_since_traj_change_s", None)
    if time_since is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if grace_period_s is None:
        grace_period_s = env.termination_manager.get_term_cfg(base_term).params.get(
            "grace_period_s", 0.0
        )
    in_window = (time_since >= grace_period_s) & (time_since < grace_period_s + window_s)
    return drift & in_window


def illegal_terrain_contact(
    env, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Terminate when the peak contact force on any of the sensor's bodies
    exceeds ``threshold``.

    Mirrors :func:`isaaclab.envs.mdp.terminations.illegal_contact` and reads
    unfiltered :attr:`ContactSensor.data.net_forces_w_history` — the total
    force from all contact partners, self-collision included.

    Note: filtering this to terrain-only contact is not viable on the PhysX
    GPU pipeline — it cannot filter against the static terrain mesh (logs
    ``GPU contact filter for collider '/World/ground' is not supported`` and
    zeroes the force matrix), and filtering against the robot's own bodies to
    subtract self-contact does not capture intra-articulation contacts
    reliably.  Use an orientation-based fall detector
    (:func:`base_orientation_from_upright`) instead of trying to isolate
    terrain contact by force.  This function remains for non-terrain hard
    contacts where ``threshold`` can simply be set above self-collision noise.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = wp.to_torch(contact_sensor.data.net_forces_w_history)
    body_slice = sensor_cfg.body_ids if sensor_cfg.body_ids is not None else slice(None)
    return torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, body_slice], dim=-1), dim=1)[0] > threshold,
        dim=1,
    )