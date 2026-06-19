# Copyright 2026 Zachary Olkin. All rights reserved.

"""Configuration for the open-loop motion-sequence command."""

from isaaclab.utils import configclass

from .base_trajectory_cmd_cfg import BaseTrajectoryCommandCfg


@configclass
class MotionSequenceCommandCfg(BaseTrajectoryCommandCfg):
    """Config for :class:`MotionSequenceCommand`.

    ``path`` points at a dense clip-library ``.h5`` (produced by
    ``scripts/rsl_rl/generate_clips.py``), not a stride-library folder.
    The conditioner / gate / skill-bucket fields of the multi-skill command
    do not apply -- a clip is tracked open-loop by a step index.
    """

    class_type: type | str = (
        "robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking"
        ".motion_sequence_cmd:MotionSequenceCommand"
    )

    base_frame_name: str = "pelvis_link"
    """Trajectory output frame used as the robot root (must expose pos/ori)."""

    random_start_phase: bool = False
    """If True, episodes start at a random step within the clip (else step 0)."""

    random_start_min_remaining_s: float = 2.0
    """When ``random_start_phase`` is True, the random start is drawn uniformly
    from ``[0, clip_length - this]`` seconds, so at least this many seconds of
    clip remain after the start (avoids near-zero-length episodes that would
    immediately hit ``clip_end_reached``). Ignored when ``random_start_phase``
    is False."""

    randomize_heading: bool = False
    """If True, each clip is placed at a random world yaw at reset."""

    clip_device: str | None = None
    """Device to hold the clip library on. ``None`` uses the env/sim device.
    Keep on CPU for very large libraries (the per-step gather still works,
    paying a host->device copy each step)."""

    clip_fingerprint: dict | None = None
    """Recorded clip-library fingerprint (set at train, verified at play).
    Populated by the command at construction; serialised into env.yaml."""

    # --- Debug visualization (set ``debug_vis = True`` to enable) ---
    debug_vis_horizon_s: float = 3.0
    """Forward window (seconds) of the reference trajectory drawn as markers."""

    debug_vis_stride: int = 5
    """Draw every Nth step within the debug window (sparser = fewer markers)."""

    debug_vis_marker_radius: float = 0.03
    """Sphere radius (m) for the per-frame reference-trajectory markers."""
