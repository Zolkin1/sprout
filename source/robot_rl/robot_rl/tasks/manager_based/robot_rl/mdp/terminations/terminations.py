# Copyright 2026 Zachary Olkin. All rights reserved.

from __future__ import annotations

import torch
import warp as wp
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    quat_apply,
    quat_box_minus,
    wrap_to_pi,
)

from motion_gen.world_lift import anchor_mask_from_free

from ..events.resets import freeform_route, tile_local_xy

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


#: Anchor-mask bits, computed rather than hand-typed so they stay in lock-step with the bit
#: order in ``motion_gen.world_lift``.  Bit SET = that DOF is FREE.
_MASK_DOWNRANGE = anchor_mask_from_free({"downrange"})
_MASK_LATERAL = anchor_mask_from_free({"lateral"})
_MASK_HEIGHT = anchor_mask_from_free({"height"})
_MASK_ROLL_PITCH = anchor_mask_from_free({"roll", "pitch"})
_MASK_YAW = anchor_mask_from_free({"yaw"})


def base_pose_error_termination(
    env,
    command_name: str = "traj_ref",
    max_pos_dist: float = 0.25,
    max_ori_rad: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the base drifts too far from the reference, on the PINNED axes only.

    Compares the robot's world base pose against
    :meth:`MotionSequenceCommand.termination_reference_world_state` -- the **globally-placed**
    clip, deliberately NOT the per-step re-anchored reference the reward grades against.

    Only the anchor mask's PINNED DOFs are measured.  A FREE DOF is one the clip was slid
    along to sit on the robot, so error there is zero by construction and invisible to the
    policy: the reference window shows it a base error of ~0 in exactly those directions.
    Bounding something unobservable only truncates episodes.  What remains is the part of the
    pose the clip genuinely pins down, which is both graded by the CLF and visible:

    ==========================  ====================================================
    mask                        measured
    ==========================  ====================================================
    flat (downrange/lat/yaw     height, roll, pitch  -- i.e. "don't sink, stay upright"
    free)
    stairs / boxes (lateral     downrange, height, roll, pitch, yaw
    free)
    ==========================  ====================================================

    On flat that is exactly a height + orientation check, which is all SONIC terminates on.
    On terrain it additionally holds position along the treads and heading.  Because the free
    axes are no longer measured, ONE tight ``max_pos_dist`` works everywhere -- there is no
    longer a reason to inflate the bound on flat to stop it firing on drift nobody grades.

    Frames: the translational bits are named in the TERRAIN's frame, so the horizontal error
    is rotated by ``-mask_frame_yaw`` before being split into downrange / lateral.  Height is
    gravity-aligned and needs no rotation.  Orientation splits into TILT (roll+pitch together,
    as the angle between the two up-axes -- frame-free) and YAW (the heading difference).
    Roll and pitch share one test because every mask in use pins both.

    Box clips carry a PER-FRAME mask (ego approach -> pinned mount), so which axes are
    measured follows the phase within a single episode.

    Args:
        env: IsaacLab environment.
        command_name: The :class:`MotionSequenceCommand` term name.
        max_pos_dist: Max position error over the PINNED translational axes (m).
        max_ori_rad: Max tilt, and max yaw error where yaw is pinned (rad).
        asset_cfg: Robot scene entity.

    Returns:
        Boolean ``[num_envs]`` -- True for envs that should terminate.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    state = cmd.termination_reference_world_state()
    ref_pos_w, ref_quat_w = state[:, 0:3], state[:, 3:7]   # xyzw (Isaac Lab 3.0)
    root_pos_w = wp.to_torch(asset.data.root_pos_w)
    root_quat_w = wp.to_torch(asset.data.root_quat_w)      # xyzw

    mask = cmd.manager.gather_anchor_mask_at_step(cmd.clip_id, cmd._step).to(root_pos_w.device)
    pinned = lambda bit: (mask & bit) == 0                                       # noqa: E731

    # --- position, in the TERRAIN frame -------------------------------------------------
    d = ref_pos_w - root_pos_w
    yaw_f = cmd._mask_frame_yaw.to(d.device)
    cos_y, sin_y = torch.cos(yaw_f), torch.sin(yaw_f)
    downrange = cos_y * d[:, 0] + sin_y * d[:, 1]
    lateral = -sin_y * d[:, 0] + cos_y * d[:, 1]
    comps = torch.stack([downrange, lateral, d[:, 2]], dim=-1)
    keep = torch.stack([pinned(_MASK_DOWNRANGE), pinned(_MASK_LATERAL),
                        pinned(_MASK_HEIGHT)], dim=-1)
    pos_err = torch.norm(comps * keep.to(comps.dtype), dim=-1)

    # --- orientation: tilt (roll+pitch) and yaw, tested separately ----------------------
    _, _, ref_yaw = euler_xyz_from_quat(ref_quat_w, wrap_to_2pi=False)
    _, _, act_yaw = euler_xyz_from_quat(root_quat_w, wrap_to_2pi=False)
    yaw_err = wrap_to_pi(ref_yaw - act_yaw).abs()
    # Tilt = angle between the two body-z axes. Independent of yaw, so the two tests do not
    # double-count, and gravity-aligned so no terrain-frame rotation is needed.
    up = torch.tensor([0.0, 0.0, 1.0], device=d.device).expand(d.shape[0], 3)
    cos_tilt = (quat_apply(ref_quat_w, up) * quat_apply(root_quat_w, up)).sum(dim=-1)
    tilt_err = torch.acos(cos_tilt.clamp(-1.0, 1.0))

    over_tilt = (tilt_err > max_ori_rad) & pinned(_MASK_ROLL_PITCH)
    over_yaw = (yaw_err > max_ori_rad) & pinned(_MASK_YAW)
    return (pos_err > max_pos_dist) | over_tilt | over_yaw


def tracked_frame_error_termination(
    env,
    command_name: str = "traj_ref",
    frame_names: tuple[str, ...] = (
        "left_ankle_roll_link", "right_ankle_roll_link",
        "left_wrist_yaw_link", "right_wrist_yaw_link",
    ),
    max_pos_dist: float = 0.25,
    max_ori_rad: float | Sequence[float | None] = 0.8,
) -> torch.Tensor:
    """Terminate when a tracked end-effector drifts too far from ITS reference.

    Complements :func:`base_pose_error_termination`, which watches the base against the
    globally-placed clip. This one is LOCAL: it reads ``y_act`` / ``y_des``, so the clip
    root has already been re-datumed onto the robot and what remains is limb error. The two
    catch different failures -- a robot can hold its base pose perfectly while dragging a
    foot through the wrong place, and it can track every limb while walking off the clip.

    Terminates if ANY named frame exceeds either bound.

    Args:
        env: IsaacLab environment.
        command_name: The :class:`MotionSequenceCommand` term name.
        frame_names: Frames to watch. Defaults to the four end effectors.
        max_pos_dist: Max per-frame position error before terminating (m).
        max_ori_rad: Max per-frame geodesic orientation error before terminating (rad).
            Either one value for every frame, or one PER frame in ``frame_names`` order,
            where ``None`` disables the orientation bound for that frame (its POSITION bound
            still applies).  Per-frame exists because the wrists and the ankles are not
            comparable: profiled at 2.6k iterations, orientation drove 492 of 629 ee trips
            and ``left_wrist_yaw`` alone was 305 of them, of which 286 were on TERRAIN and
            only 4 on flat.  A wrist rides a longer kinematic chain than an ankle and its yaw
            is not pinned by ground contact, so one shared bound made this mostly a
            left-wrist-on-terrain detector.  Raising the bound does not fix that -- 0.8 ->
            1.0 rad moved it 305 -> 290 trips, so the error is far past the bound rather than
            marginally over it, and the honest options are to disable it or to fix whatever
            the terrain clips are asking of that wrist.

    Returns:
        Boolean ``[num_envs]`` -- True for envs that should terminate.

    Raises:
        ValueError: If a per-frame ``max_ori_rad`` does not match ``frame_names``.
    """
    cmd = env.command_manager.get_term(command_name)
    act_pos, act_quat, des_pos, des_quat = cmd.tracked_frame_pose(list(frame_names))
    n, f = act_pos.shape[0], act_pos.shape[1]
    pos_err = torch.norm(act_pos - des_pos, dim=-1)                                  # [N, F]
    ori_err = torch.norm(
        quat_box_minus(act_quat.reshape(n * f, 4), des_quat.reshape(n * f, 4)), dim=-1
    ).view(n, f)

    if isinstance(max_ori_rad, (int, float)):
        ori_lim = torch.full((f,), float(max_ori_rad), device=ori_err.device,
                             dtype=ori_err.dtype)
    else:
        vals = [float("inf") if v is None else float(v) for v in max_ori_rad]
        ori_lim = torch.as_tensor(vals, device=ori_err.device, dtype=ori_err.dtype)
        if ori_lim.numel() != f:
            raise ValueError(
                f"max_ori_rad has {ori_lim.numel()} entries but frame_names has {f}: "
                f"{list(frame_names)}"
            )
    return ((pos_err > max_pos_dist) | (ori_err > ori_lim)).any(dim=-1)


def clip_end_reached(env, command_name: str = "traj_ref", reserve_lookahead: bool = True) -> torch.Tensor:
    """Truncate (good termination) when the clip's last USABLE step is reached.

    Wire this as ``DoneTerm(..., time_out=True)`` so RSL-RL bootstraps the
    value from the final state rather than treating clip-end as a failure.
    Keyed on the clip step counter so it is correct even when episodes start
    mid-clip (``random_start_phase``).

    With ``reserve_lookahead`` the episode ends one full ``reference_window`` **before**
    the clip's last frame, rather than at it. The look-ahead gather clamps to the library
    length (``_future_step_indices``), so without the margin the final
    ``(num_steps - 1) * stride`` = 50 steps of every clip feed the policy a window that is
    progressively frozen on the last frame -- and then teach the generator to predict that
    "the reference stops moving". Reserving the margin costs the tail 1 s of every clip and
    means no observed sample is ever a clamp artefact. The margin is read off the live
    ``reference_window`` term, so it cannot drift from the observation.

    Args:
        env: IsaacLab environment.
        command_name: The :class:`MotionSequenceCommand` term name.
        reserve_lookahead: End the episode a full observation window early (default). Set
            False to run to the clip's true last frame and accept the saturated tail.

    Returns:
        Boolean ``[num_envs]`` -- True for envs at/after the clip's last usable step.
    """
    cmd = env.command_manager.get_term(command_name)
    # Each clip's OWN last real frame (clip_len - 1), so a short clip padded into a longer library
    # truncates at its real end instead of playing the held padding tail to num_steps - 1.  clip_len
    # lives on the clip-library device (may be CPU via --clip_device), so index it with clip_id on
    # THAT device before moving to the env device (device-safe gather).
    clip_len = cmd.manager.clip_len
    last_step = clip_len[cmd.clip_id.to(clip_len.device)].to(cmd.start_step.device) - 1
    if reserve_lookahead:
        offsets = cmd.obs_window_offsets()
        if offsets is not None and offsets.numel() > 0:
            # clamp_min(0): a clip shorter than the window has no unsaturated step at all,
            # so it ends immediately rather than silently serving a frozen reference.
            last_step = (last_step - int(offsets.max().item())).clamp_min(0)
    return (cmd.start_step + env.episode_length_buf) >= last_step

def base_height_below_termination(
    env, min_height: float = 0.3, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the base world height drops below ``min_height``: the fall detector.

    The height half of :func:`base_fallen` with the attitude half left out, because a
    generator-driven run grades attitude against the REFERENCE instead
    (:func:`fm_base_tilt_error_termination`) and the two would otherwise double-count.

    **Absolute, deliberately, and it is the reference-relative version that fails here.**
    Height error against the served reference looks like the consistent choice, but under
    ``ego_anchor_fraction=1.0`` the reference is re-datumed onto the robot at every re-plan,
    so that error collapses toward zero every ``fm_update_every`` steps -- a slow sink walks
    straight through it. World z has no such hole.

    The cost is that this assumes the terrain never goes below world zero: if it did, a
    healthy robot descending far enough would trip it. That holds for the shipped libraries
    (every tile builds upward from the ground plane) and is why the bound can sit as low as
    0.3 m -- deep under any standing pose, so it fires on a collapse rather than on a stoop.

    Args:
        env: IsaacLab environment.
        min_height: Base world height below which the episode ends (m).
        asset_cfg: Robot scene entity.

    Returns:
        Boolean ``[num_envs]`` -- True for envs that should terminate.
    """
    robot = env.scene[asset_cfg.name]
    return wp.to_torch(robot.data.root_pos_w)[:, 2] < min_height


def fm_base_tilt_error_termination(
    env, command_name: str = "traj_ref", max_tilt_rad: float = 1.047,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the base tilts too far from the REFERENCE's attitude (yaw ignored).

    :func:`base_orientation_from_upright` measures against world upright, which is the wrong
    datum once the reference itself leans -- climbing and jumping both pitch the base on
    purpose, so a fixed cone around vertical either fires on healthy motion or has to be
    slackened until it catches nothing.  This measures the angle between the robot's own up
    axis and the REFERENCE's, so the allowance travels with the motion.

    Yaw drops out by construction: comparing up axes says nothing about heading, which is the
    same reason :func:`base_orientation_from_upright` ignores it.

    Args:
        env: IsaacLab environment.
        command_name: The tracking command term.
        max_tilt_rad: Maximum angle between the two up axes (rad); 1.047 = 60 deg.
        asset_cfg: Robot scene entity.

    Returns:
        Boolean ``[num_envs]`` -- True for envs that should terminate.
    """
    cmd = env.command_manager.get_term(command_name)
    robot = env.scene[asset_cfg.name]
    up = torch.zeros(env.num_envs, 3, device=env.device)
    up[:, 2] = 1.0
    ref_up = quat_apply(cmd.reference_world_state()[:, 3:7], up)
    act_up = quat_apply(wp.to_torch(robot.data.root_quat_w), up)
    cos = (ref_up * act_up).sum(-1).clamp(-1.0, 1.0)
    return torch.arccos(cos) > max_tilt_rad


def fm_tracking_violation_termination(
    env,
    command_name: str = "traj_ref",
    spec: "object | None" = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate where the robot leaves the STRICT tracking bounds around the served node.

    The union of the three checks in :func:`mdp.tracking_error.violation_checks` -- lateral,
    height, tilt -- evaluated against the reference the generator actually served this step.
    It goes through those functions rather than restating the tests, so this termination and
    every offline consumer of ``first_violation_step`` cannot disagree about what a violation
    is; the bounds likewise come from ``BOOTSTRAP_SPEC`` rather than being written here.

    **Why this is a TERMINATION and not just the offline latch it used to be.**  The recording
    used to run on past a violation and every consumer then cut the episode there offline,
    which left the two halves incoherent: the run reached the goal and banked the progress
    reward's success bonus, and the offline cut then deleted the very row that bonus was paid
    on -- measured at 95.3% of arriving episodes.  Terminating makes the episode END where
    every consumer already thought it ended, so the bonus is simply never earned by a run that
    left the bounds, no truncation is needed anywhere, and ``reached_goal`` and
    ``rollout_outcomes.terminal_reward`` become the same question.

    **It cannot fire before the generator has planned.**  ``_pred_valid`` is False until the
    first re-plan, and before that there is no served reference to be far from -- a real state
    rather than an error.  Without this gate every episode would terminate on step 0, which
    is indistinguishable from a policy that cannot stand up.

    **The height bound is doing the work here, and it is chunk-local.**  Under
    ``window_anchor="plan"`` the served node is expressed against the robot's own root at the
    last re-plan, so a height excursion has to accumulate inside one re-plan (0.24 s at
    K=12).  That is why this does not contradict
    :func:`base_height_below_termination`'s docstring, which rejects a reference-relative
    height bound as a FALL detector: a slow sink re-datums away and is invisible here, which
    is exactly why the absolute-height term stays.

    Args:
        env: IsaacLab environment.
        command_name: The tracking command term.
        spec: A ``tracking_error.TrackingErrorSpec``, or ``None`` for ``BOOTSTRAP_SPEC``.
        asset_cfg: Robot scene entity.

    Returns:
        Boolean ``[num_envs]`` -- True for envs that should terminate.
    """
    from ..tracking_error import BOOTSTRAP_SPEC, tracking_error, violates  # noqa: PLC0415

    cmd = env.command_manager.get_term(command_name)
    robot = env.scene[asset_cfg.name]
    sp = BOOTSTRAP_SPEC if spec is None else spec
    # Offset 0 -- the node being tracked THIS step, which is what the accumulator grades and
    # what `first_violation_step` is recorded against.
    zero = torch.zeros(1, dtype=torch.long, device=env.device)
    pos_w, quat_w, lin_w, ang_w = cmd.get_future_world_base(zero)
    jp, _ = cmd.get_future_joint_state(zero)
    # **`robot.data`'s buffers are WARP arrays here and must be converted.** `eval.accumulate`
    # reads them through a `_Data` shim that calls `wp.to_torch` on every attribute access, so
    # copying its accessor NAMES without its shim gets a `vec3f` where a tensor is wanted and
    # raises inside `tracking_error`'s first subtraction -- at the first step of collection,
    # after the simulator has launched. Every other termination in this file converts
    # explicitly (see `base_height_below_termination`), which is the idiom.
    #
    # `root_link_*`, NOT `root_lin_vel_w`: the latter is the CoM twist. The same choice as
    # `eval.accumulate`, and mixing the two would put a systematic offset on the velocity
    # term in one place only.
    err = tracking_error(
        pos_w[:, 0], quat_w[:, 0], lin_w[:, 0], ang_w[:, 0], jp[:, 0],
        wp.to_torch(robot.data.root_pos_w), wp.to_torch(robot.data.root_quat_w),
        wp.to_torch(robot.data.root_link_lin_vel_w),
        wp.to_torch(robot.data.root_link_ang_vel_w),
        wp.to_torch(robot.data.joint_pos))
    hit = violates(err, sp)
    # THE GATE IS REQUIRED, not optional. `_pred_valid` is allocated only when the command has
    # a generator (`fm_enabled`); on the clip arm `_post_init` returns before that, so a
    # `getattr(..., None)` fallback would SKIP the gate exactly where there is no served
    # prediction to be far from -- terminating from step 0 on a task nothing here measured,
    # which reads as a policy that cannot stand up. Refuse instead: this term belongs to the
    # generator arm and says so.
    valid = getattr(cmd, "_pred_valid", None)
    if valid is None:
        raise ValueError(
            f"command {command_name!r} has no `_pred_valid`, so it is serving no generator "
            "prediction (reference_source is not 'fm', or no fm_checkpoint was given). The "
            "strict tracking bounds are defined against the SERVED node, so there is nothing "
            "for this termination to measure -- remove `tracking_violation` from the "
            "terminations for this arm rather than letting it fire from step 0.")
    return hit & valid.reshape(-1).to(hit.device)


def base_orientation_from_upright(
    env,
    roll_limit_deg: float = 70.0,
    pitch_limit_deg: float = 70.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """True where the base roll or pitch deviates from world upright by more than the limits.

    The reference is the world-frame upright (identity rotation), so YAW IS IGNORED: this is a
    face-plant / fallen detector that does not depend on the tracking reference, and so is not
    subject to whatever grace gating :func:`base_pose_error_termination` uses.

    Restored 2026-07-29: this was deleted in ``a754728`` (the online reference-engine removal)
    while :func:`base_fallen` kept calling it, leaving that termination -- and therefore the
    whole ``G1-tracker-clf-fm-*`` task -- raising ``NameError`` on its first step. Implementation
    is the pre-deletion one verbatim, so termination behaviour is unchanged from when it worked.

    Args:
        env: IsaacLab environment.
        roll_limit_deg: Maximum allowed |roll| from upright in degrees.
        pitch_limit_deg: Maximum allowed |pitch| from upright in degrees.
        asset_cfg: Robot scene entity.

    Returns:
        Boolean tensor of shape ``[num_envs]`` -- ``True`` for envs that should terminate.
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
    # ``TODO(align-tilt-convention)``: the orientation half is PER-AXIS, while the eval
    # harness's derived ``fell`` column uses the TOTAL angle from upright -- so this
    # termination and the column meant to mirror it disagree, measured at 0.045 against 0.054
    # on one rollout. Over 2e6 uniform orientations the per-axis test is a strict SUBSET of
    # the total-angle one (zero counter-examples), they differ on 1.54% of the sphere, and the
    # worst pose this admits is 75.6 deg total. Aligning is monotone -- it can only fire
    # earlier -- but it changes WHEN episodes end and therefore every recording and every
    # published rate, so it is a deliberate open item rather than a silent edit. The strict
    # reference-relative checks (``mdp.tracking_error``) are the primary criterion now; this
    # is the backstop for a reference that is broken or absent.
    asset: Articulation = env.scene[asset_cfg.name]
    too_low = wp.to_torch(asset.data.root_pos_w)[:, 2] < min_height
    tilted = base_orientation_from_upright(env, roll_limit_deg, pitch_limit_deg, asset_cfg)
    return too_low | tilted



def freeform_goal_reached(
    env,
    min_half_width: float = 2.5,
    height_tol_m: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Truncate (good termination) when the robot has walked its free-form section.

    The clip-free replacement for :func:`clip_end_reached`, which bounds an episode by the
    length of a motion that no longer exists. Wire it as ``DoneTerm(..., time_out=True)``:
    walking the section IS the task, so its end is a truncation and RSL-RL bootstraps the
    value from the final state rather than scoring it as a failure. Keep the task's plain
    ``time_out`` alongside -- this fires only on success.

    **A wide BAND across the section, not a disc**, and that is the lesson mjeval paid for:
    a 0.5 m circle on the centreline scored most of its timeouts as "did not get through"
    when the robot had cleared the terrain cleanly and passed BESIDE the disc, because a few
    degrees of accumulated heading error over six metres is more than half a metre of lateral
    miss (``mjeval.terrains.GOAL_HALF_WIDTH_FLOOR``). Measuring the absence of a heading
    controller is not what this is for. So: any lateral position within the band, and any
    downrange position at or past the goal line.

    **Downrange it is a HALF-PLANE, where mjeval uses a 2 m-deep band.** mjeval polls
    ``in_goal`` as a verdict on a trial that keeps running, so it needs a far edge; here the
    first step past the line ENDS the episode, so a far edge could only add a way to step over
    the goal and miss it.

    **The HEIGHT test is OPTIONAL and off by default** (``height_tol_m = 0``), and the reason
    to turn it on depends entirely on the terrain.

    On an AUTHORED section the hole it closes is small: the band overhangs the tile floor by at
    most 0.1395 m per side on 5.9% of tiles, stairs only
    (``motion_gen.terrain.freeform.GOAL_HALF_WIDTH_FLOOR_M`` carries the sweep), and inside
    that overhang a robot on the floor scores the same truncation as one on the runout.

    On a section whose goal is UP -- a raised landing, a box top -- the hole is not small at
    all, and it is the whole measurement. A clip-library ``stairs_up`` tile has 0.90 m of open
    floor either side of a 2.25 m flight, and ``min_half_width`` clamps the band to at least
    that, so a robot that never left the floor can pass the goal line and score. The two
    DESCENDING families are immune for a different reason -- their goal is at floor level and
    their spawn is on the platform, so the height change is unavoidable -- which is why this is
    a parameter rather than a constant: it is needed exactly where the goal is above the spawn.

    Off by DEFAULT because turning it on silently would change the termination of every
    free-form task, including arms already measured, and a success rate that moved for that
    reason is indistinguishable from one that moved because the policy did.

    The datum is the PELVIS, matching :func:`~.goal_region.freeform_goal_anchor_world`: the
    route's ``goal_z`` is the walkable SURFACE, and a standing robot's root sits a nominal
    height above it, so comparing raw ``root_z`` against ``goal_z`` would put every genuine
    arrival ~0.79 m outside a 0.5 m tolerance. That asymmetry already cost this stack a run
    where every arrival recorded as a failure.

    Args:
        env: IsaacLab environment (``env.scene.terrain`` must carry ``terrain_route``).
        min_half_width: Floor on the band's half-extent (m), applied over each section's own
            declared ``goal_half_width``. mjeval's floor, for its reason: the widest feature in
            that catalogue is 3.96 m across, and 2.5 m clears everything with margin.
        height_tol_m: Also require the PELVIS within this many metres of the goal's own
            standing height.  ``0`` (default) disables the test entirely and reproduces the
            downrange-and-lateral behaviour every existing task was measured under.
        asset_cfg: Robot scene entity.

    Returns:
        Boolean ``[num_envs]`` -- True for envs at or past their section's goal line.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos_w = wp.to_torch(asset.data.root_pos_w)
    route = freeform_route(env.scene.terrain,
                           torch.arange(env.num_envs, device=root_pos_w.device))
    local = tile_local_xy(route["corner"], route["frame_yaw"], root_pos_w)
    half_w = torch.clamp(route["goal_half_width"], min=float(min_half_width))
    hit = ((local[:, 0] >= route["goal_x"])
           & ((local[:, 1] - route["goal_y"]).abs() <= half_w))
    if float(height_tol_m) > 0.0:
        # The goal's own STANDING height: the tile corner, plus the route's walkable `goal_z`,
        # plus the asset's nominal root height -- the same three terms
        # `freeform_goal_anchor_world` sums, so the line the episode ends on and the anchor the
        # outcome is folded against cannot disagree about where "there" is.
        # Indexed rather than `.get(..., 0)`: the defensive read meant this datum was the tile
        # FLOOR on every library, since `goal_z` was not a route field at all.
        gz = route["goal_z"].to(root_pos_w.device)
        nominal_z = wp.to_torch(asset.data.default_root_pose)[:, 2].to(root_pos_w.device)
        goal_h = route["corner"][:, 2].to(root_pos_w.device) + nominal_z + gz
        hit = hit & ((root_pos_w[:, 2] - goal_h).abs() <= float(height_tol_m))
    return hit
