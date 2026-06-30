# Copyright 2026 Zachary Olkin. All rights reserved.

"""Configuration for the flow-matching generator-tracking command."""

from isaaclab.utils import configclass

from .motion_sequence_cmd_cfg import MotionSequenceCommandCfg


@configclass
class GeneratorTrackingCommandCfg(MotionSequenceCommandCfg):
    """Config for :class:`GeneratorTrackingCommand`.

    Extends the open-loop motion-sequence command into a single, robot-decoupled
    mode: every env runs the **flow-matching generator** autoregressively (the FM
    conditions on its OWN past reference output, seeded from the clip at reset and
    during the warmup -- it never sees the robot), and each sampled trajectory is
    re-anchored onto the robot root (xy + z + heading, keeping only the FM's tilt)
    -- a receding horizon. The clip only supplies the reset spawn
    pose and the ``target_vel`` velocity command. The CLF tracking error is
    computed in the world frame.
    """

    class_type: type | str = (
        "robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking"
        ".generator_tracking_cmd:GeneratorTrackingCommand"
    )

    fm_checkpoint: str = ""
    """Path to the trained (clip-conditioned, command-conditioned) flow-matching
    ``.pt`` checkpoint whose predictions the reference follows.  Required."""

    fm_update_every: int = 1
    """Re-run flow-matching inference every ``K`` control steps; the predicted
    reference advances through the F-step prediction window between re-infers
    (``K = 1`` re-infers every step).  Amortizes FM compute.  Must be ``>= 1``."""

    fm_num_sampling_steps: int | None = None
    """Override the checkpoint's Euler sampling-step count at inference (or ``None``
    to use the trained value).  Fewer steps = cheaper, coarser samples."""

    fm_device: str | None = None
    """Device to load the flow-matching model on (``None`` uses the env device)."""

    fm_fraction: float = 1.0
    """Fraction of envs whose reference is driven by the flow-matching generator
    (ego-anchored onto the robot root); re-rolled per reset (see
    :func:`reset_on_motion_sequence`).  The remaining envs track the clip, split into
    world- vs ego-anchored by ``ego_anchor_fraction``, giving three buckets:
    world-anchored clip, ego-anchored clip, and ego-anchored FM.  Only the FM-bucket
    envs pay the FM inference cost.  ``1.0`` (default) puts every env on the FM (the
    deploy / play behaviour)."""

    drive_with_clip: bool = False
    """Diagnostic mode: drive the robot with the **clip** while running the FM
    **passively**.  When True the applied reference (what the policy tracks, what the
    CLF/rewards grade, the ``y_des`` obs) is the ego-anchored clip -- exactly the
    clip-tracking deploy -- but the flow-matching generator still runs in its full
    deploy configuration (autoregressive on its own output, re-anchored on the robot)
    and its prediction is cached so it can be recorded/plotted alongside the clip via
    :meth:`GeneratorTrackingCommand.as_fm_reference`.  This is the inverse of the
    default (FM drives, clip is the comparison overlay): here the clip drives and the
    FM is the comparison overlay, so you can compare the two with the robot safely on
    the clip.  Default False (FM drives)."""

    # --- Debug visualization (generator-tracking specific) ---
    # These augment the inherited clip-ribbon viz (cylinder segments). Set
    # ``debug_vis = True`` to enable.
    debug_vis_show_fm_prediction: bool = True
    """Draw the flow-matching generator's predicted world-frame trajectory as a
    ribbon of spheres (one color per frame, matching the clip ribbon's colors), for
    the envs whose FM prediction is live (past warmup). Lets you compare the FM
    prediction (spheres) against the clip baseline (cylinders)."""

    debug_vis_show_source: bool = True
    """Draw a floating marker above each robot indicating its reference source:
    green = FM (prediction live), yellow = warming up (history not yet full, still
    falling back to the clip)."""

    debug_vis_fm_marker_radius: float = 0.02
    """Sphere radius (m) for the FM-prediction ribbon points."""

    debug_vis_source_marker_radius: float = 0.05
    """Sphere radius (m) for the per-env tracking-source indicator."""

    debug_vis_source_marker_height: float = 0.4
    """Height (m) above the robot root to float the tracking-source indicator."""

    debug_vis_show_base_velocity: bool = True
    """Draw two world-frame base-velocity arrows per env: green = the joystick-level
    velocity command (``target_vel``, rotated from the base-yaw heading frame into
    the world frame -- the signal the velocity reward grades and the FM is
    conditioned on), blue = the robot's actual base velocity. The gap between them
    visualizes the high-level command tracking error."""

    debug_vis_velocity_height: float = 0.6
    """Height (m) above the robot root to float the base-velocity arrows."""

    debug_vis_velocity_scale: float = 3.0
    """Arrow-length gain: the x-scale is ``speed (m/s) * this`` (matches the
    stock velocity-command arrow scaling)."""
