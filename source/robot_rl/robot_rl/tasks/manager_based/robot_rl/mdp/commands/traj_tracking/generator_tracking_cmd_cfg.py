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

    closed_loop_fraction: float = 0.5
    """Fraction of the **FM-tracking** envs that are *closed-loop* (the rest are
    *open-loop*).  Both buckets still track the FM generator's prediction; they
    differ only in the velocity command fed to the FM and in the pose-drift
    termination reference:

      * **open-loop FM** (``1 - closed_loop_fraction`` of the FM envs): the FM is
        fed the clip's open-loop ``target_vel`` and the pose-drift termination is
        graded against the FM reference (the original behavior);
      * **closed-loop FM** (``closed_loop_fraction`` of the FM envs): the FM is
        fed a feedforward + proportional position controller that tracks the
        clip's *moving* position reference, and the pose-drift termination is
        graded against the **clip** reference -- so a robot that stalls while the
        clip marches on accrues a growing corrective command and terminates.

    The actor/critic see no flag identifying the bucket.  ``0`` ⇒ all FM envs are
    open-loop (the original behavior).  Re-rolled (within the FM envs) on every
    reset."""

    pd_pos_kp: float = 1.5
    """Proportional gain (1/s) on the world-frame base position error for the
    closed-loop velocity command."""

    pd_pos_kd: float = 0.0
    """Derivative gain on the robot's base-frame planar velocity for the
    closed-loop velocity command (damping; ``0`` disables)."""

    pd_yaw_kp: float = 1.0
    """Proportional gain (1/s) on the wrapped base yaw error for the closed-loop
    yaw-rate command."""

    pd_yaw_kd: float = 0.0
    """Derivative gain on the robot's base yaw rate for the closed-loop yaw-rate
    command (damping; ``0`` disables)."""

    pd_max_lin_vel: float = 2.0
    """Symmetric clamp (m/s) on each planar component of the closed-loop velocity
    command.  Should bracket the clip library's ``target_vel`` range so the FM's
    command conditioning stays in-distribution."""

    pd_max_ang_vel: float = 2.0
    """Symmetric clamp (rad/s) on the closed-loop yaw-rate command."""

    fm_handoff_time_s: float = -1.0
    """If ``>= 0``, every env tracks the clip for the first ``fm_handoff_time_s`` seconds
    of the episode, then hands off to the flow-matching generator -- a single time-based
    clip->FM switch within one episode (overrides the per-env ``clip_fraction`` roll).
    Mirrors the MuJoCo runner's ``--clip_handoff_time`` so the clip->FM handoff can be
    compared apples-to-apples across IsaacLab and sim2sim.  ``< 0`` (default) uses the
    normal random per-env clip/FM split (training).  (Float, not Optional, so a hydra
    ``env.commands.traj_ref.fm_handoff_time_s=...`` override type-checks.)"""

    fm_update_every: int = 1
    """Re-run flow-matching inference every ``K`` control steps for FM-tracking
    envs; the predicted next reference advances through the F-step prediction
    between re-infers (``K = 1`` re-infers every step).  Amortizes FM compute."""

    fm_history_source: str = "robot"
    """What the FM conditions on as history: ``"robot"`` (the robot's measured
    state each step -- the original generator-tracking behaviour, for an
    FM trained on recorded rollouts) or ``"reference"`` (the FM's OWN past
    reference output, seeded from the clip at reset and during the warmup).

    ``"reference"`` makes the generator a true autoregressive, **robot-decoupled**
    receding-horizon reference source -- it never sees the robot. Use it with a
    clip-conditioned FM (history = past clip; see ``clip_to_fm_data.py``) and
    ``clip_fraction = 0`` / ``closed_loop_fraction = 0`` so every env runs the FM
    open-loop. The tracker policy still tracks the emitted reference."""

    fm_anchor_on_prev_traj: bool = False
    """If True, FM-tracking envs plant each newly sampled trajectory on the *previous*
    generated trajectory (at the step it has advanced to, ``dt = fm_update_every``
    steps in) instead of on the robot's current pose. Only the ground plane (xy +
    yaw) is chained from the prior reference; height (z) and tilt (roll/pitch) stay
    grounded to the robot each inference. This lets the FM reference drift globally
    away from a poorly-tracking robot, like the pre-made motion clips. The first
    inference after a reset (no valid prior prediction) still anchors on the robot.
    ``False`` (default) keeps the original robot-anchored behavior."""

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
    """Draw two world-frame base-velocity arrows per env: green = the joystick-level
    velocity command (``target_vel``, rotated from the base-yaw heading frame into
    the world frame -- the signal the velocity reward grades and the FM is
    conditioned on), blue = the robot's actual base velocity. The gap between them
    visualizes the high-level command tracking error. Drawn for every env regardless
    of source."""

    debug_vis_velocity_height: float = 0.6
    """Height (m) above the robot root to float the base-velocity arrows."""

    debug_vis_velocity_scale: float = 3.0
    """Arrow-length gain: the x-scale is ``speed (m/s) * this`` (matches the
    stock velocity-command arrow scaling)."""
