# Copyright 2026 Zachary Olkin. All rights reserved.

"""Configuration for the robot-state-conditioned flow-matching tracking command."""

from isaaclab.utils import configclass

from .motion_sequence_cmd_cfg import MotionSequenceCommandCfg


@configclass
class FmTrackingCommandCfg(MotionSequenceCommandCfg):
    """Config for :class:`FmTrackingCommand` (closed-loop, robot-state-conditioned FM).

    The FM conditions on the **live robot state** history (N most-recent measured steps,
    in the FM's xyz+yaw frame), an exteroception token, and the future velocity command;
    it predicts the F-step ego-anchored clip future, re-anchored on the robot each
    re-plan. The CLF tracking error is computed in the world frame (inherited).

    The clip library supplies **only** the reset spawn pose (and the terrain meshed from
    it) and the ``target_vel`` command -- there is no clip fallback for the reference
    itself, not even during warmup. See :attr:`strict_fm_only`.
    """

    class_type: type | str = (
        "robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking"
        ".fm_tracking_cmd:FmTrackingCommand"
    )

    fm_checkpoint: str = ""
    """Path to the trained robot-state-conditioned flow-matching ``.pt`` checkpoint
    (``target_source=ego_anchored_clip_future``). Required."""

    fm_update_every: int = 1
    """Re-run flow-matching inference every ``K`` control steps (the receding-horizon
    re-plan cadence); the applied index advances through the F-step window in between.
    ``1`` = re-plan every step, so the cursor never advances and the reference is always
    node 0 of a fresh prediction. Must be ``>= 1``.

    This need NOT match the recording's ``fm_snap_every``; the agreement check that used to
    enforce it was removed. Note what that means in each direction:

    * FASTER than the recording (``K < fm_snap_every``) is horizon-SAFE -- the required
      horizon is ``max_offset + K``, so a smaller K needs less of it -- and the query
      states are a superset of the snapped ones rather than different in kind, since the
      teacher re-anchors every step regardless.
    * SLOWER (``K > fm_snap_every``) is the direction that bites: the cursor runs further
      into the window between re-plans, and ``_check_horizon`` is now the only thing
      standing between that and a silently frozen reference tail.

    Cost scales directly with this: the generator is ~99% of per-step inference, so K=1
    makes every control tick pay the full FM cost instead of one tick in K."""

    fm_num_sampling_steps: int | None = None
    """Override the checkpoint's Euler sampling-step count at inference (``None`` = trained)."""

    fm_device: str | None = None
    """Device to load the flow-matching model on (``None`` uses the env device)."""

    strict_fm_only: bool = True
    """Poison every ``y_des``/``dy_des`` channel the FM did not write with NaN, each step.

    The barrier between the generator path and the offline clips. The FM writes the base
    frame, its end-effector frames and the joints; everything else in the output vector
    (the COM block, the ``@root`` body-link blocks, any frame the checkpoint does not
    predict) is clip-derived and must not reach a reward, an observation or a
    termination here. Filling those with NaN turns an accidental read into an immediate,
    obvious failure instead of a plausible-looking number from the wrong source.

    Turn it off only to debug what a clip-sourced channel *would* have contained."""

    fm_zero_noise: bool = False
    """Integrate the flow from ``x0 = 0`` instead of Gaussian noise, so the prediction is a
    deterministic function of the conditioning.

    Default ``False`` = training-matched stochastic sampling; leave it there for training and
    for any run meant to represent real behaviour. Set it only for A-B diagnostics where the
    per-call noise draw would mask the effect being measured -- notably sim2sim, whose MuJoCo
    counterpart (``GenTrackerPolicy(zero_noise=...)``, exposed as ``--zero_noise`` /
    ``--gaussian_noise``) defaults it ON, so a default Isaac run and a default MuJoCo run are
    NOT comparing like with like. ``x0 = 0`` is off the training distribution."""

    # --- Debug visualization ---
    debug_vis_show_fm_prediction: bool = True
    """Draw the FM's predicted world-frame trajectory as a ribbon of spheres (one color per
    frame). Overlay it on the yellow observation-window ribbon (``debug_vis_obs_window``) to
    check that what the policy observes is what the generator predicted."""

    debug_vis_fm_marker_radius: float = 0.02
    """Sphere radius (m) for the FM-prediction ribbon points."""
