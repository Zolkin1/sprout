# Copyright 2026 Zachary Olkin. All rights reserved.

"""Configuration for the flow-matching generator-tracking command."""

from isaaclab.utils import configclass

from .motion_sequence_cmd_cfg import MotionSequenceCommandCfg


@configclass
class GeneratorTrackingCommandCfg(MotionSequenceCommandCfg):
    """Config for :class:`GeneratorTrackingCommand`.

    Extends the open-loop motion-sequence command: every env is still reset onto
    a clip (which supplies the FM velocity command ``target_vel`` and the reset
    pose), but a configurable fraction of envs track the **flow-matching
    generator's** predicted future state instead of the clip reference.  The CLF
    tracking error is computed in the **world frame** for both sources.
    """

    class_type: type | str = (
        "robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking"
        ".generator_tracking_cmd:GeneratorTrackingCommand"
    )

    fm_checkpoint: str = ""
    """Path to the trained (extended, future-command) flow-matching ``.pt`` checkpoint
    whose predictions the FM-tracking envs follow.  Required."""

    clip_fraction: float = 0.35
    """Fraction of envs that track the pre-made motion clip (the stable teacher
    signal); the remaining ``1 - clip_fraction`` track the FM generator.  The
    per-env assignment is re-rolled on every reset."""

    fm_update_every: int = 1
    """Re-run flow-matching inference every ``K`` control steps for FM-tracking
    envs; the predicted next reference advances through the F-step prediction
    between re-infers (``K = 1`` re-infers every step).  Amortizes FM compute."""

    fm_num_sampling_steps: int | None = None
    """Override the checkpoint's Euler sampling-step count at inference (or ``None``
    to use the trained value).  Fewer steps = cheaper, coarser samples."""

    fm_device: str | None = None
    """Device to load the flow-matching model on (``None`` uses the env device)."""
