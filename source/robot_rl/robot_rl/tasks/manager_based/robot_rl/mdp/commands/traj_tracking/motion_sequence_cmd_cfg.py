# Copyright 2026 Zachary Olkin. All rights reserved.

"""Configuration for the open-loop motion-sequence command."""

from isaaclab.utils import configclass

from robot_rl.flow_matching.window import DEFAULT_WINDOW_ANCHOR

from .base_trajectory_cmd_cfg import BaseTrajectoryCommandCfg


@configclass
class MotionSequenceCommandCfg(BaseTrajectoryCommandCfg):
    """Config for :class:`MotionSequenceCommand`.

    ``path`` points at a dense clip-library ``.h5`` (produced by
    ``scripts/rsl_rl/generate_mb_library.py``), not a stride-library folder.
    The conditioner / gate / skill-bucket fields of the multi-skill command
    do not apply -- a clip is tracked open-loop by a step index.
    """

    class_type: type | str = (
        "robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking"
        ".motion_sequence_cmd:MotionSequenceCommand"
    )

    base_frame_name: str = "pelvis_link"
    """Trajectory output frame used as the robot root (must expose pos/ori)."""

    tracking_mode: str = "joint"
    """Which reference parameterization the CLF grades -- the joint-vs-body A/B switch.

    ``y_des`` / ``y_act`` always carry every channel (the reset, the world-state
    reconstruction and the FM path all read ``joint:*`` regardless); this only selects
    what enters eta:

    * ``"joint"`` -- ``pelvis_link:*`` + the world end-effector frames + all ``joint:*``.
      The original parameterization.
    * ``"body"`` -- ``pelvis_link:*`` + the world end-effector frames + the root-frame body
      links (``<body>@root:*``), with ``joint:*`` dropped.  The DeepMimic / Sonic style
      Cartesian error metric.  Requires a clip library carrying root-frame channels (see
      ``scripts/rsl_rl/backfill_root_frames.py``); the world-frame skeleton is kept so
      global foot placement is still penalised against the terrain.
    """

    random_start_phase: bool = False
    """If True, episodes start at a random step within the clip (else step 0)."""

    random_start_min_remaining_s: float = 2.0
    """When ``random_start_phase`` is True, the random start is drawn uniformly
    from ``[0, clip_length - this]`` seconds, so at least this many seconds of
    clip remain after the start (avoids near-zero-length episodes that would
    immediately hit ``clip_end_reached``). Ignored when ``random_start_phase``
    is False."""

    random_start_before_terrain_start: bool = False
    """Draw the random start BEFORE the terrain begins, so every episode opens on flat.

    For a generator-driven run this is about the learned ``null_history`` embedding, which
    covers the first N steps of every episode and is fitted overwhelmingly on flat ground --
    which is also how the robot starts on hardware. Opening already on terrain asks it for the
    one case it never saw.

    A clip with no flat lead-in (``terrain_start_step`` at or near 0, i.e. the ``*_pure``
    stair families and a few ``jump_off*`` clips) cannot satisfy this; those start as early as
    the history allows instead, and are the ones
    :attr:`~.fm_tracking_cmd_cfg.FmTrackingCommandCfg.seed_history_from_clip` covers.

    Independent of :attr:`random_start_before_terrain_end`, which bounds the other end."""

    random_start_before_terrain_end: bool = True
    """Cap the random start at the last frame of the clip's terrain-PINNED run.

    A transition or box clip is ego approach, then a pinned run over the terrain (the stair
    climb, the box mount or dismount), then ego walking on the raised top or away from it.
    Without this cap a start drawn past the pinned run spawns the robot AFTER its one chance
    to interact with the terrain, and the episode is just flat walking that the flat clips
    already cover -- on ``g1_mb_v5_body.h5`` that is 8.5% of all resets, and up to 70% on
    individual jump-off clips.  With it, a start may land anywhere from frame 0 through the
    last pinned frame, and never after.

    No-op for clips with no pinned run (flat, distractors) and for clips pinned to their end
    (pure stairs, one-sided box), where ``random_start_min_remaining_s`` already binds first.
    Only the terrain reset event honours this; the flat teacher has no terrain to miss.
    Post-terrain frames are still TRAINED -- episodes run forward through them; they just
    stop being entry points.  Ignored when ``random_start_phase`` is False."""

    randomize_heading: bool = False
    """If True, each clip is placed at a random world yaw at reset."""

    clip_device: str | None = None
    """Device to hold the clip library on. ``None`` uses the env/sim device.
    Keep on CPU for very large libraries (the per-step gather still works,
    paying a host->device copy each step)."""

    clip_fingerprint: dict | None = None
    """Recorded clip-library fingerprint (set at train, verified at play).
    Populated by the command at construction; serialised into env.yaml."""

    # --- Ego-centric (receding-horizon) clip re-anchoring ---
    ego_anchor_fraction: float = 0.0
    """Fraction of envs whose clip placement is periodically re-anchored onto the
    robot's current root link (planar position + heading), turning the reference
    into an ego-centric, receding-horizon trajectory.  The assignment is re-rolled
    per reset (like the tracker's FM bucket).  ``0.0`` (default) keeps every env on
    the fixed world placement chosen at reset."""

    ego_anchor_every: int = 12
    """Re-anchor cadence in control steps for ego-anchored envs: the clip is
    re-placed on the robot's root every this many steps (a receding-horizon
    re-plan).  Smaller = more frequent, smaller per-step reference jumps.  Only
    used when ``ego_anchor_fraction > 0``.

    ``1`` makes the reference fully LOCAL: the free root
    DOFs coincide with the robot every step, so tracking error is measured
    locally and planar drift is not double-counted.  Drift is still graded --
    by the termination, against the globally-fixed clip.  See
    :meth:`MotionSequenceCommand.termination_reference_world_state`."""

    fm_snap_every: int = 12
    """Cadence in control steps for the FM SNAP flag (``_reanchor_active``), which the
    recorder logs and the flow-matching dataset turns into one training window each.

    Deliberately SEPARATE from :attr:`ego_anchor_every`.  They used to share one timer,
    which was fine when re-placement was occasional but breaks at every-step anchoring:
    the recorder would emit a window per env per step, and the FM target would degenerate
    to nothing (node 0 always coincides with the robot, so the window would encode no
    placement decision -- the very thing the FM predicts).  Keep this at the value the FM
    checkpoint was recorded with; it must match ``fm_update_every`` on the tracker."""

    window_anchor: str = DEFAULT_WINDOW_ANCHOR
    """Which pose the reference window's ``base_pose`` POSITION is measured from.

    * ``"robot_now"`` (default) -- ``p_ref - p_actual(t)``: the tracking error, measured
      against the robot as it is this step. What every run before 2026-08-20 trained on.
    * ``"plan"`` -- ``p_ref - p_plan``: the reference's position in the frame of the last
      re-plan. A pure function of the reference, so nothing has to know where the robot is.

    **Why "plan" exists: it removes odometry from deployment.** The generator already
    predicts in the plan frame, so under ``"plan"`` its output IS the observation and the
    deploy runtime serves it unchanged. Under ``"robot_now"`` the runtime must convert, which
    means reading the robot's measured world position every control step -- real odometry on
    hardware. See :data:`robot_rl.flow_matching.window.BASE_POSE_CONVENTIONS`.

    The two are the same width and differ only in meaning, so the choice is stamped into the
    run's ``params/`` and into ``tracker.onnx``, and refused on mismatch at load.

    **The default must stay ``"robot_now"``**: runs trained before this field existed have no
    ``window_anchor`` in their saved config, so they resolve to the default, and their
    recorded convention string has to keep matching."""

    stand_start: bool = False
    """Spawn STANDING -- default joint pose, zero velocity, yaw-only orientation.

    Off by default so every existing task is bit-identical. The clip's first frame is a
    snapshot of a motion already in progress: measured on the 4-family library, about half of
    every family's clips open with |v| > 0.3 m/s and some at 2.27 m/s, so the default reset
    drops the robot mid-stride at speed. That is a state no deployment ever starts from -- a
    real robot stands up and then walks -- and the generator's first inference has to be made
    from it, under the null-history embedding, which is its weakest configuration.

    The clip's PLACEMENT (xy and heading) is kept, because it is what puts the robot on the
    right tile facing the traversal; only the pose and motion are replaced.

    Pairs with :attr:`cmd_hold_s`: standing still while being commanded 1.5 m/s is its own
    off-distribution state, so the settle window is what makes this coherent.
    """

    cmd_hold_s: float = 0.0
    """Hold the commanded velocity at ZERO for this long at the start of an episode.

    Applies to the horizon as well as the current step, so the generator conditions on a
    command that is zero now and resumes later -- which is what will actually happen -- rather
    than on a constant command it is not being asked to follow yet.

    0.0 (the default) reproduces every existing task exactly. ~0.3 s covers the first re-plan
    interval at ``fm_update_every=12``.
    """

    # --- Debug visualization (set ``debug_vis = True`` to enable) ---
    debug_vis_horizon_s: float = 3.0
    """Forward window (seconds) of the reference trajectory drawn as markers."""

    debug_vis_stride: int = 5
    """Draw every Nth step within the debug window (sparser = fewer markers)."""

    debug_vis_marker_radius: float = 0.01
    """Cylinder radius (m) for the per-frame reference-trajectory ribbon
    segments (thin = line-like, matching the online command's marker style)."""

    debug_vis_show_hands: bool = False

    debug_vis_body_spheres: bool = True
    """Draw a sphere pair per TRACKED body: measured (green) vs the reference the reward
    grades against (magenta), i.e. the clip after the per-step re-datum onto the robot.

    The gap between each pair IS the reward's per-body error, so this is the direct read on
    whether body-link tracking is working.  Follows ``tracking_mode`` automatically -- it
    reads the tracked channels out of ``y_act``/``y_des`` rather than restating a body list."""

    debug_vis_body_sphere_radius: float = 0.03
    """Radius (m) of the tracked-body spheres."""

    debug_vis_obs_window: bool = True

    #: Path to a ``goal_regions.json`` (see ``scripts/rsl_rl/build_goal_regions.py``). When
    #: set, the debug visualiser draws each env's GOAL REGION as a translucent gate: a slab
    #: at the region's near face, spanning its lateral band and its height band, square to
    #: the terrain frame.
    #:
    #: This exists to be LOOKED AT before anything is measured. The region is the only
    #: spatial success test in the tracker task -- every other outcome signal compares step
    #: indices -- and the ways it can be subtly wrong all produce a perfectly plausible
    #: number: the wrong yaw frame shows only on tiles whose heading is not world-x, an
    #: anchor taken at the clip's padded tail rather than at ``last_usable_step`` sits a
    #: whole observation window too far downrange, and a band whose height is off puts the
    #: gate under the terrain. None of those is visible in a table.
    #:
    #: **The DEFAULT tolerances are drawn, not any per-family override.** The thing worth
    #: eyeballing is the anchor and the frame, which are family-independent; read the JSON
    #: for a family's own numbers.
    #:
    #: **``play.py`` needs ``--viz kit``.** This IsaacLab defaults to headless ("No
    #: visualizer was selected, so running in headless mode"), and headless never fires
    #: ``_debug_vis_callback`` at all -- so the gate silently does not draw, along with every
    #: other debug marker. That reads exactly like a region placed somewhere off screen,
    #: which is the one conclusion this visualiser must not allow. ``--video`` does not
    #: substitute: it renders offscreen without firing the callback either.
    #:
    #: Empty = off, which is every existing task.
    debug_vis_goal_regions: str = ""
    """Path to a ``goal_regions.json`` whose regions to draw; empty = off.

    The two docstrings that used to sit here belonged to OTHER fields -- one to
    :attr:`debug_vis_obs_window` above, and one to a ``debug_vis_ee_frames`` that no longer
    exists -- so this flag was documented as "draw a ribbon along the base of the reference
    window", which is not what it does. Nothing read the missing field, so the only cost was
    the wrong description on the flag a reader reaches for. See the ``#:`` comments above for
    what this actually draws and why it needs ``--viz kit``."""
