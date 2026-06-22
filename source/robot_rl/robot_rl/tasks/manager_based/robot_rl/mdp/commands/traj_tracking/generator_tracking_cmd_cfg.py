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

    # --- Debug visualization (generator-tracking specific) ---
    # These augment the inherited clip-ribbon viz (cylinder segments, drawn for
    # all envs as the clip baseline). Set ``debug_vis = True`` to enable.
    debug_vis_show_fm_prediction: bool = True
    """Draw the flow-matching generator's predicted world-frame trajectory as a
    ribbon of spheres (one color per frame, matching the clip ribbon's colors),
    for the envs currently tracking the FM (``_fm_mask & _pred_valid``). Lets you
    compare the FM prediction (spheres) against the clip baseline (cylinders)."""

    debug_vis_show_source: bool = True
    """Draw a floating marker above each robot indicating which reference it is
    tracking: green = FM, white = clip, yellow = FM-but-warming-up (history not
    yet full, so still falling back to the clip)."""

    debug_vis_fm_marker_radius: float = 0.02
    """Sphere radius (m) for the FM-prediction ribbon points."""

    debug_vis_source_marker_radius: float = 0.05
    """Sphere radius (m) for the per-env tracking-source indicator."""

    debug_vis_source_marker_height: float = 0.4
    """Height (m) above the robot root to float the tracking-source indicator."""

    debug_vis_show_base_velocity: bool = True
    """Draw two world-frame base-velocity arrows per env: green = reference base
    velocity (the source-selected clip / FM target), blue = the robot's actual
    base velocity. The gap between them visualizes the base-velocity tracking
    error. Drawn for every env regardless of source."""

    debug_vis_velocity_height: float = 0.6
    """Height (m) above the robot root to float the base-velocity arrows."""

    debug_vis_velocity_scale: float = 3.0
    """Arrow-length gain: the x-scale is ``speed (m/s) * this`` (matches the
    stock velocity-command arrow scaling)."""
