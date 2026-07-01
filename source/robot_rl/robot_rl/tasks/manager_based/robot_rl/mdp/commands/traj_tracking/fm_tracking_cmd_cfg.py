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

    fm_update_every: int = 10
    """Re-run flow-matching inference every ``K`` control steps (the receding-horizon
    re-plan cadence = a snap); the applied index advances through the F-step window in
    between. Should match the recording's ``ego_anchor_every``. Must be ``>= 1``."""

    fm_num_sampling_steps: int | None = None
    """Override the checkpoint's Euler sampling-step count at inference (``None`` = trained)."""

    fm_device: str | None = None
    """Device to load the flow-matching model on (``None`` uses the env device)."""
