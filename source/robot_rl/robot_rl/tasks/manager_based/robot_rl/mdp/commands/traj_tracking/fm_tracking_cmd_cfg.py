# Copyright 2026 Zachary Olkin. All rights reserved.

"""Configuration for the robot-state-conditioned flow-matching tracking command."""

from isaaclab.utils import configclass

from .motion_sequence_cmd_cfg import MotionSequenceCommandCfg


@configclass
class FmTrackingCommandCfg(MotionSequenceCommandCfg):
    """Config for :class:`FmTrackingCommand` (closed-loop, robot-state-conditioned FM).

    The FM conditions on the **live robot state** history (N most-recent measured steps,
    in the FM's xyz+yaw frame), a root-height token, and the future velocity command; it
    predicts the F-step ego-anchored clip future, re-anchored on the robot each re-plan.
    The clip library supplies only the reset spawn pose + the ``target_vel`` command; the
    clip reference is the warmup fallback until the N-step history fills. The CLF tracking
    error is computed in the world frame (inherited).
    """

    class_type: type | str = (
        "robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking"
        ".fm_tracking_cmd:FmTrackingCommand"
    )

    fm_checkpoint: str = ""
    """Path to the trained robot-state-conditioned flow-matching ``.pt`` checkpoint
    (``target_source=ego_anchored_clip_future``). Required."""

    fm_update_every: int = 12
    """Re-run flow-matching inference every ``K`` control steps (the receding-horizon
    re-plan cadence = a snap); the applied index advances through the F-step window in
    between. Should match the recording's ``ego_anchor_every``. Must be ``>= 1``."""

    fm_num_sampling_steps: int | None = None
    """Override the checkpoint's Euler sampling-step count at inference (``None`` = trained)."""

    fm_device: str | None = None
    """Device to load the flow-matching model on (``None`` uses the env device)."""

    fm_null_history_warmup: bool = True
    """Sample the FM from control step 0 using the model's learned ``null_history`` embedding
    while the N-step history window is still filling, instead of falling back to the clip.

    The generator was trained with ``null_cond_prob`` of its samples history-nulled precisely so
    it can be conditioned without a history, so this is the in-distribution way to cover the
    post-reset steps. With it off, the reference falls back to the clip for the first ``N-1``
    steps (the previous behaviour), which makes the RL and deploy paths disagree: the deployed
    generator has no clip to fall back on and instead fabricates a repeated static standing
    history. Keep it on so IsaacLab, MuJoCo and hardware share one flow."""

    fm_zero_noise: bool = False
    """Integrate the flow from ``x0 = 0`` instead of Gaussian noise, so the prediction is a
    deterministic function of the conditioning.

    Default ``False`` = training-matched stochastic sampling; leave it there for training and
    for any run meant to represent real behaviour. Set it only for A-B diagnostics where the
    per-call noise draw would mask the effect being measured -- notably sim2sim, whose MuJoCo
    counterpart (``GenTrackerPolicy(zero_noise=...)``, exposed as ``--zero_noise`` /
    ``--gaussian_noise``) defaults it ON, so a default Isaac run and a default MuJoCo run are
    NOT comparing like with like. ``x0 = 0`` is off the training distribution."""

    # --- Debug visualization (augments the inherited clip-ribbon viz) ---
    debug_vis_show_fm_prediction: bool = True
    """Draw the FM's predicted world-frame trajectory as a ribbon of spheres (one color per
    frame, matching the clip ribbon), for envs whose FM prediction is live (past warmup) --
    so you can see the FM driving the robot rather than the clip warmup fallback."""

    debug_vis_show_source: bool = True
    """Float a marker above each robot: green = FM (prediction live), yellow = warming up
    (history not yet full, still on the clip)."""

    debug_vis_fm_marker_radius: float = 0.02
    """Sphere radius (m) for the FM-prediction ribbon points."""

    debug_vis_source_marker_radius: float = 0.05
    """Sphere radius (m) for the per-env tracking-source indicator."""

    debug_vis_source_marker_height: float = 0.4
    """Height (m) above the robot root to float the tracking-source indicator."""
