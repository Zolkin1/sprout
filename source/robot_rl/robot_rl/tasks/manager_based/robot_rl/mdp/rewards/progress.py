# Copyright 2026 Zachary Olkin. All rights reserved.

"""The PROGRESS reward, live, once per re-plan interval.

The torch twin of ``scripts/rsl_rl/inspect_terrain_reward.py``'s ``_progress_terms``, which
grades a RECORDING offline and is what the off-policy fine-tune (AWR and its ladder) fits a
critic to.  Both read their five constants from
:data:`~motion_gen.progress_reward.PROGRESS_REWARD_SPEC` and both gate through
:func:`~motion_gen.progress_reward.terrain_gate`, so the only way they can disagree is in the
plumbing this module supplies -- which is what
``source/robot_rl/tests/test_progress_reward_parity.py`` pins.

Per chunk::

    penalty    = contact_penalty * 68.0 + penetration_sum_m * 0.107   # TerrainConsistencySpec
    R          = exp(-penalty / tau)
    gate       = sin(pi/2 * R**2)
    v_fwd      = (root_now - anchor) / (K * dt), in the yaw frame of the anchor
    vel        = exp(-|v_fwd - cmd_vx| / 0.5)
    gate_worst = sin(pi/2 * min(R over the episode so far)**3)
    reward     = R + gate * vel + gate_worst * C * arrived * is_last

**Why this can be read at the END of the interval and still describe its START.**  ``_pred_window``
has exactly three writers -- the allocation, ``_run_fm``'s sample, and ``apply_plan_residual``'s
copy -- and the anchor is latched only in ``_run_fm``.  Inside :meth:`AdjusterEnv.step` the
re-plan happens in the POST-loop ``command_manager.compute``, after rewards, so at reward time
the plan and its anchor are still the ones this interval was served from.  The grade is
therefore of the post-residual plan, which is what the action is responsible for.

**The last chunk has no velocity term**, matching the offline grader, which marks the final
window ``valid=False`` because it has no successor anchor to difference against.  Online the
displacement is computable, but the episode ended somewhere inside the interval and the robot
kept moving under a dead episode, so the number would not mean what the offline one means.

**``gate_worst`` is causal at the only point it is used.**  Offline it is the minimum over the
whole episode; here it is the minimum SO FAR.  On the terminal chunk -- the only chunk that ever
pays the bonus -- those are the same number.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from motion_gen.progress_reward import PROGRESS_REWARD_SPEC, ProgressRewardSpec, terrain_gate
from motion_gen.terrain_consistency import TerrainConsistencySpec

from .rewards import plan_terrain_terms

# TYPE-ONLY, as everywhere else in this package. `ManagerBasedRLEnv` is the CONCRETE env class
# and reaches `pxr`, which hydra would bind before AppLauncher runs -- pip's usd-core instead of
# Kit's, and Kit segfaults a second into startup. Every cfg module imports `mdp`, so an eager
# import here reaches all of them. `test_no_pre_app_pxr_import` is what says so.
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _progress_terms(env: ManagerBasedRLEnv, command_name: str,
                    spec: ProgressRewardSpec) -> dict[str, torch.Tensor]:
    """Every piece of this interval's progress reward, computed once and cached.

    Three reward terms read this and the manager calls each separately, so without the cache
    the SDF pass behind ``plan_terrain_terms`` would run three times for one answer.  Keyed on
    the env's step counter, exactly as :func:`~.rewards._terrain_consistency` is, so a stale
    result cannot outlive the plan it describes.

    Args:
        env: The :class:`~robot_rl...envs.AdjusterEnv`.  A plain
            :class:`~isaaclab.envs.ManagerBasedRLEnv` is refused: the velocity term needs the
            interval's opening anchor, which only an interval-stepping env can latch.
        command_name: The :class:`FmTrackingCommand` term name.
        spec: The reward's constants.

    Returns:
        ``terrain``, ``gate``, ``vel_track``, ``vel_err_mps``, ``gate_vel``, ``success``,
        ``worst`` and ``is_last`` -- all ``[N]``.

    Raises:
        RuntimeError: If the env does not latch the interval's opening anchor.  Returning a
            zero velocity term instead would read as a robot that stood still.
    """
    cached = getattr(env, "_progress_reward_cache", None)
    step = int(env.common_step_counter)
    if cached is not None and cached[0] == step:
        return cached[1]

    anchor_pos = getattr(env, "progress_anchor_pos", None)
    if anchor_pos is None:
        raise RuntimeError(
            "the progress reward needs the interval's OPENING anchor, and this env does not "
            "latch one. It is written for an env whose step spans a whole re-plan interval "
            "(AdjusterEnv); on a per-control-step env the velocity term has no interval to "
            "average over. A zero here would read as a robot that stood still."
        )

    tc = TerrainConsistencySpec()
    terms = plan_terrain_terms(env, command_name)
    penalty = (terms["contact_penalty"] * tc.contact_weight
               + terms["penetration_sum_m"] * tc.penetration_weight)
    reward_terrain = torch.exp(-penalty / float(spec.tau))

    # The episode's worst chunk SO FAR, including this one -- the offline minimum is over every
    # graded chunk, so the update happens before the read.
    worst = env.worst_terrain_R
    torch.minimum(worst, reward_terrain, out=worst)

    # A true interval average: the anchor is `root_pos_w` at the re-plan that opened this
    # interval, and `root_pos_w` now is what the re-plan about to happen will latch. The
    # recorded `base_twist_yaw` the offline grader declined to use would be an instantaneous
    # sample, and a CoM twist where this is the frame-origin velocity the command means.
    root_pos_w = env.scene["robot"].data.root_pos_w
    if not isinstance(root_pos_w, torch.Tensor):
        import warp as wp                                                   # noqa: PLC0415
        root_pos_w = wp.to_torch(root_pos_w)
    dt = float(env.replan_period) * float(env.step_dt)
    v_w = (root_pos_w - anchor_pos) / dt

    # Into the yaw frame of the anchor the interval STARTED at, because the command is a
    # heading-frame joystick quantity. Only the yaw is removed: a full-quaternion rotation
    # would swing the forward component on every pitch.
    #
    # XYZW, w LAST -- `FmTrackingCommand` initialises `_anchor_quat[:, 3] = 1.0`, and the
    # offline grader verified the same layout against a real recording. Written out rather
    # than routed through `isaaclab.utils.math.yaw_quat`, whose docstring states (w, x, y, z):
    # the two conventions disagree, and the failure mode is a velocity rotated by the robot's
    # own tilt with entirely plausible magnitudes.
    q = env.progress_anchor_quat
    qw, qx, qy, qz = q[:, 3], q[:, 0], q[:, 1], q[:, 2]
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    cos_y, sin_y = torch.cos(-yaw), torch.sin(-yaw)
    v_fwd = cos_y * v_w[:, 0] - sin_y * v_w[:, 1]

    # The NOMINAL command, latched at the re-plan: `_override_cmd_speed` corrupts only the copy
    # handed to the generator as conditioning, and the speed-mismatch arm is an input
    # corruption to REJECT rather than a command to follow.
    err = (v_fwd - env.progress_cmd_vx).abs()
    vel = torch.exp(-err / float(spec.vel_sigma_mps))

    gate = terrain_gate(reward_terrain, spec.gate_sharpness)
    is_last = env.reset_terminated | env.reset_time_outs
    gate_vel = gate * vel * (~is_last).to(vel.dtype)

    arrived = env.ended_by["goal_reached"].to(reward_terrain.dtype)
    gate_worst = terrain_gate(worst, spec.success_gate_sharpness)
    success = gate_worst * float(spec.success_const) * arrived * is_last.to(arrived.dtype)

    out = {"terrain": reward_terrain, "gate": gate, "vel_track": vel, "vel_err_mps": err,
           "gate_vel": gate_vel, "success": success, "worst": worst.clone(),
           "is_last": is_last}
    env._progress_reward_cache = (step, out)
    return out


def progress_terrain(env: ManagerBasedRLEnv, command_name: str = "traj_ref",
                     spec: ProgressRewardSpec = PROGRESS_REWARD_SPEC) -> torch.Tensor:
    """``R = exp(-penalty / tau)``: does this chunk's plan agree with the ground it is on?

    Args:
        env: The interval-stepping env.
        command_name: The :class:`FmTrackingCommand` term name.
        spec: The reward's constants.

    Returns:
        ``[N]`` in ``(0, 1]``.  1 is a geometrically perfect plan.
    """
    return _progress_terms(env, command_name, spec)["terrain"]


def progress_velocity(env: ManagerBasedRLEnv, command_name: str = "traj_ref",
                      spec: ProgressRewardSpec = PROGRESS_REWARD_SPEC) -> torch.Tensor:
    """The commanded-speed term, GATED by the same chunk's terrain grade.

    Gated rather than paid outright because the bare terrain grade is very nearly maximised by
    standing still, and an ungated speed term is maximised by charging through the boxes.  The
    product asks for both.

    Zero on the interval that ends the episode; see the module docstring.

    Args:
        env: The interval-stepping env.
        command_name: The :class:`FmTrackingCommand` term name.
        spec: The reward's constants.

    Returns:
        ``[N]`` in ``[0, 1]``.
    """
    return _progress_terms(env, command_name, spec)["gate_vel"]


def progress_success(env: ManagerBasedRLEnv, command_name: str = "traj_ref",
                     spec: ProgressRewardSpec = PROGRESS_REWARD_SPEC) -> torch.Tensor:
    """The arrival bonus, paid ONCE, gated by the episode's WORST chunk.

    **Scaled by ``(1 - gamma)``, and that is not a tuning choice.**  The offline reducer builds
    the ``(1-gamma)``-weighted average ``G = (1-gamma) * sum gamma^i r``, so a one-off bonus
    enters AWR's target with weight ``(1-gamma)``.  Here ``goal_reached`` is a TRUNCATION, and
    rsl-rl adds ``gamma * V(s_t)`` at a truncation -- at a goal-adjacent state ``V`` is defined
    through itself, with fixed point ``(r + b) / (1 - gamma)``, so a bonus paid at face value
    would recur with weight ``1 / (1 - gamma)`` = 20x what the offline arm sees.  Paying
    ``(1 - gamma) * gate_worst * C`` removes that 20x.

    **It does NOT make the two returns identical, and an earlier version of this docstring
    claimed it did.**  The identity

        (1-gamma) * U_t  ==  G_t   for all t

    holds only if the offline tail continues at the TERMINAL chunk's ongoing reward, i.e. at
    ``tail_chunks == 1``.  On the progress branch it does not: ``fit_value.py`` calls
    ``bootstrapped_window_return`` without passing ``tail_chunks``, so it takes that function's
    default of 3 and averages the last three ongoing rewards -- while ``--tail_chunks`` exists,
    defaults to 1, and is threaded into the OTHER reducer only.  The residual is a systematic
    bias on arrivals and budget time-outs (it is exactly zero on failures, where both sides take
    an absorbing 0 tail), it grows toward the terminal chunk, and it is of the same order as the
    bonus itself.

    Left alone on purpose: the four off-policy arms were all fitted under ``tail_chunks = 3``,
    so changing it now would align this arm with a reducer that produced none of the numbers it
    is compared against.  Closing it properly means passing ``tail_chunks=1`` on the progress
    branch and re-running every arm together.

    ``gamma`` is stated ONCE, on ``AdjusterEnvCfg.progress_gamma``, and read from there via
    ``env.progress_gamma_scale``; ``scripts/rsl_rl/train.py`` refuses a run whose agent cfg
    discounts by a different number, because that is the only place both objects are in hand.

    Args:
        env: The interval-stepping env.
        command_name: The :class:`FmTrackingCommand` term name.
        spec: The reward's constants.

    Returns:
        ``[N]``, zero except on the chunk that ends an ARRIVING episode.
    """
    return _progress_terms(env, command_name, spec)["success"] * float(env.progress_gamma_scale)
