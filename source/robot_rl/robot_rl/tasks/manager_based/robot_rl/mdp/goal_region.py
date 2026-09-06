# Copyright 2026 Zachary Olkin. All rights reserved.

"""The spatial goal region: where a clip-driven episode has to physically GET TO.

Everything else the tracker task can terminate or score on is a **clip-cursor** condition.
``mdp.clip_end_reached`` fires on ``start_step + episode_length_buf >= clip_len - 1 -
max_offset``; ``eval.schema.terrain_columns`` compares ``steps_survived`` against step
indices.  The clip cursor advances once per control step no matter where the robot is, so
"survived to clip end" means *stayed upright for N steps*, not *got anywhere*.  A robot that
stalls in place, lags three metres behind, or drifts laterally off the flight scores a
success under every one of them.

That loophole is fatal to any method that reshapes the generator's output distribution
toward "outcomes that score well", because standing still is a trivially reachable optimum
with none of the risk of walking up a staircase.  So success needs a test the robot cannot
pass by waiting, and that is what this module is.

**The region, in the terrain's own frame:**

* a **downrange half-plane** -- the robot got at least ``approach_margin_m`` short of where
  the clip itself finishes.  A half-plane rather than a box because overshoot is not a
  failure: walking off the far end of a finite tile already trips ``base_fallen``, so
  bounding it above would only add a second way to be wrong.
* a **lateral band** of half-width ``half_width_m``, because nothing in this stack steers
  and a robot that clears the terrain routinely ends up beside the clip rather than on it.
* a **height band** of ``height_tol_m``.  This one is not optional and is the failure the
  otherwise-identical ``freeform_goal_reached`` leaves open: a robot that falls off the side
  of an elevated tile lands on the ground plane roughly 0.79 m up, which does NOT trip a
  0.40 m absolute-height fall test, and could then satisfy downrange and lateral while
  standing on the floor below the staircase it was supposed to climb.

**The frame resolution is lifted from** :func:`~...terminations.base_pose_error_termination`,
which is the only place in the repo that gets it right: the anchor is the **globally placed**
clip (``_reset_world_pos`` / ``_reset_world_yaw``, written once per episode by
``_commit_clip_reset`` and never touched by re-anchoring), and the horizontal error is rotated
by ``-_mask_frame_yaw`` into downrange / lateral before being tested.  Using the per-step
re-anchored reference instead would collapse the whole test: the free root DOFs are slid onto
the robot every step, so the error there is zero by construction and a stalled robot would
read as perfectly on-goal.

What is deliberately NOT taken from that function is its threshold.  ``max_pos_dist=0.25`` is
a *drift* bound, and it stays disabled as a termination on the FM tracker for the reason it
was disabled there: it fires exactly when the generator emits an unusual chunk, which converts
"unusual" into "failed".  The tolerances here are wide, and they are MEASURED -- see
``scripts/rsl_rl/build_goal_regions.py``, which sizes them from where surviving *clip-arm*
episodes actually end, so the region can never be tighter than the tracker's own demonstrated
ability and a miss is attributable to the generator.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

import torch

#: Families whose tolerances may be overridden individually, spelled the way the recording's
#: ``RecordingMeta.family_names`` spells them minus the ``family_`` prefix -- i.e. what
#: ``eval.sampler.normalize_family`` returns.  Not an allow-list (an unknown family falls back
#: to the default); just the set the calibration script reports on.
KNOWN_FAMILIES: tuple[str, ...] = (
    "flat", "stairs_up", "stairs_down", "jump_on", "jump_off", "jump_off_front",
    "distractor", "stairs_dn_pure", "stairs_up_pure",
)


@dataclass(frozen=True)
class GoalTolerances:
    """How far off the clip's own endpoint still counts as arriving.

    Attributes:
        approach_margin_m: How far SHORT of the clip's terminal downrange coordinate the
            half-plane boundary sits.  The robot must reach ``goal_downrange -
            approach_margin_m`` or beyond.
        half_width_m: Lateral half-width of the band, about the clip's terminal lateral
            coordinate.
        height_tol_m: Absolute band on ``|robot_z - goal_z|``.  Catches the robot that
            reached the right (x, y) while standing on the floor beside or below the terrain.
    """

    approach_margin_m: float = 1.0
    half_width_m: float = 2.5
    height_tol_m: float = 0.5


@dataclass
class GoalRegionSpec:
    """A calibrated set of tolerances, plus whatever produced it.

    Serialised to ``goal_regions.json`` and hashed into the run manifest, so every stage of a
    collection/eval campaign resolves the identical definition rather than each re-deriving
    one.  The ANCHOR is not stored: it is the placed clip's own terminal base pose, recovered
    exactly at any step from ``_reset_world_pos`` / ``_reset_world_yaw``, so storing it would
    duplicate the clip library.

    Attributes:
        default: Tolerances used for any family without an override.
        by_family: Per-family overrides, keyed as :data:`KNOWN_FAMILIES` spells them.
        provenance: Free-form record of the calibration run -- checkpoint, clip library sha,
            episode counts, and the measured inside-fraction per family.
    """

    default: GoalTolerances = field(default_factory=GoalTolerances)
    by_family: dict[str, GoalTolerances] = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def for_family(self, family: str) -> GoalTolerances:
        """Tolerances for one family, falling back to :attr:`default`.

        Args:
            family: Normalised family name (no ``family_`` prefix).

        Returns:
            The override if one exists, else :attr:`default`.
        """
        return self.by_family.get(str(family), self.default)

    @classmethod
    def bootstrap(cls) -> "GoalRegionSpec":
        """A deliberately UNCALIBRATED spec, for the run that produces the real one.

        The calibration is circular on its face: the tolerances are fitted from where
        surviving clip-arm episodes end, but nothing records where an episode ended unless a
        goal region is attached.  This breaks it.  The three raw terminal offsets and
        ``max_goal_downrange_m`` do not depend on the tolerances at all, so ANY spec records
        what ``build_goal_regions.py`` needs to fit; this one just says out loud that its own
        numbers are placeholders.

        The tolerances are FINITE rather than infinite so the region is drawable -- the whole
        point of the first pass is to look at the gate in Isaac and check that the anchor,
        the terrain frame and the height all land where they should, and an infinite region
        renders as nothing.  The values are :class:`GoalTolerances`' own defaults.

        ``provenance["bootstrap"]`` is True, ``runner.py`` copies it into the table's meta,
        and ``report.py`` refuses to derive spatial success from a table carrying it.  Without
        that, a run under placeholder tolerances would report a success rate that looks
        exactly like a result.

        Returns:
            The spec.
        """
        return cls(
            default=GoalTolerances(),
            provenance={
                "bootstrap": True,
                "note": ("UNCALIBRATED placeholder tolerances. Finite so the gate can be "
                         "SEEN in Isaac, and sized only to be plausible. Records the goal "
                         "geometry so build_goal_regions.py has something to fit; its own "
                         "reached_goal means nothing and report.py refuses it."),
            },
        )

    def to_dict(self) -> dict:
        """JSON-ready dict."""
        return {
            "default": asdict(self.default),
            "by_family": {k: asdict(v) for k, v in sorted(self.by_family.items())},
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "GoalRegionSpec":
        """Rebuild from :meth:`to_dict` output.

        Args:
            raw: The parsed JSON.

        Returns:
            The spec.

        Raises:
            ValueError: If ``default`` is missing.  There is no implicit fallback: a spec
                without one would silently hand every family the dataclass defaults, which
                are placeholders rather than measurements.
        """
        if "default" not in raw:
            raise ValueError(
                "goal region spec has no 'default' block; it must state measured tolerances "
                "rather than inherit the placeholder dataclass defaults. Re-run "
                "scripts/rsl_rl/build_goal_regions.py."
            )
        return cls(
            default=GoalTolerances(**raw["default"]),
            by_family={k: GoalTolerances(**v) for k, v in (raw.get("by_family") or {}).items()},
            provenance=dict(raw.get("provenance") or {}),
        )

    def save(self, path: str) -> None:
        """Write to ``path``, creating parent directories."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "GoalRegionSpec":
        """Read a spec written by :meth:`save`."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


def last_usable_step(cmd) -> torch.Tensor:
    """The step a SURVIVING episode ends on, per env, lookahead reserve included.

    The exact quantity ``mdp.clip_end_reached`` truncates at, re-derived here from the same
    two sources (the clip's own length and the live ``reference_window`` term's furthest
    offset) rather than restated, so the goal can never be anchored on a step the episode does
    not reach.  Anchoring on ``clip_len - 1`` instead would place the goal up to one full
    observation window past where any episode can get, and every arm would score zero.

    Args:
        cmd: A :class:`MotionSequenceCommand` (or subclass).

    Returns:
        ``[N]`` long, on ``cmd.clip_id``'s device, never below 0.
    """
    if not cmd.has_clips:
        # No clip, so no clip length. The quantity this stands for is "the last step of this
        # episode that could still serve a full look-ahead window", and on a clip-free task the
        # episode is bounded by the time-out horizon rather than by a motion -- so that is what
        # it resolves to. Constant across envs, which is correct: every section shares it.
        horizon = int(getattr(cmd._env, "max_episode_length", 0))
        step = torch.full_like(cmd.clip_id, max(horizon - 1, 0))
        offsets = cmd.obs_window_offsets()
        if offsets is not None and offsets.numel() > 0:
            step = step - int(offsets.max().item())
        return step.clamp_min(0).long()
    clip_len = cmd.manager.clip_len
    step = clip_len[cmd.clip_id.to(clip_len.device)].to(cmd.clip_id.device) - 1
    offsets = cmd.obs_window_offsets()
    if offsets is not None and offsets.numel() > 0:
        step = step - int(offsets.max().item())
    return step.clamp_min(0).long()


def freeform_goal_anchor_world(cmd) -> tuple[torch.Tensor, torch.Tensor]:
    """The goal anchor for a CLIP-FREE section, read off the manifest's route block.

    :func:`goal_anchor_world` recovers the anchor from the placed clip's own terminal pose, and
    on a manifest-only library there is no clip to ask -- every gather refuses by name
    (``MotionClipManager._require_clips``).  A free-form section declares its goal instead, in
    tile-local coordinates, and this is the inverse of the lift
    ``mdp.terminations.freeform_goal_reached`` applies to the robot: local ``(goal_x, goal_y,
    goal_z)`` rotated by the tile's ``frame_yaw`` and translated to its corner.

    **The two must agree, and they do by construction**: both read the SAME route row through
    :func:`~.events.resets.freeform_route`, so the anchor the outcome is folded against and
    the line the episode terminates on cannot drift apart.  The termination tests a half-plane
    plus a lateral band while the fold measures a signed displacement from this point; that is
    the same geometry read two ways, not two goals.

    Args:
        cmd: A :class:`MotionSequenceCommand` (or subclass) whose env carries a route-bearing
            terrain.

    Returns:
        ``(goal_pos_w [N, 3], frame_yaw [N])`` on the env device.

    Raises:
        TypeError: If the terrain carries no ``terrain_route`` -- a clip library, or a
            free-form library built before the route block existed.
    """
    from .events.resets import freeform_route  # noqa: PLC0415

    dev = cmd._mask_frame_yaw.device
    n = cmd._mask_frame_yaw.shape[0]
    route = freeform_route(cmd._env.scene.terrain, torch.arange(n, device=dev))
    yaw = route["frame_yaw"]
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    gx, gy = route["goal_x"], route["goal_y"]
    # `goal_z` rides beyond FREEFORM_ROUTE_FIELDS, so read it defensively: a library written
    # before it existed still has a usable x/y goal, and the height term is the one the
    # termination does not test either.
    gz = route.get("goal_z")
    corner = route["corner"]
    # **The anchor is a PELVIS height, not a ground height**, because that is what the clip
    # anchor is -- `goal_anchor_world` returns the clip's terminal BASE pose -- and
    # `height_tol_m` is applied to `|root_z - goal_z|`. A route's `goal_z` is the walkable
    # SURFACE there, so using it raw would put a standing robot ~0.79 m outside a 0.5 m
    # tolerance at every goal: `in_goal` could never fire while `freeform_goal_reached` still
    # terminated, and the recorder would label every arrival a failure.
    # The standing height is read off the asset's own `default_root_pose`, which is the same
    # quantity `reset_on_freeform_section` adds to `start_z` when it places the spawn -- so
    # the two ends of the episode are measured against one number rather than two.
    import warp as wp  # noqa: PLC0415

    asset = cmd._env.scene["robot"]
    nominal_z = wp.to_torch(asset.data.default_root_pose)[:, 2].to(dev)
    pos = torch.stack((corner[:, 0] + cos_y * gx - sin_y * gy,
                       corner[:, 1] + sin_y * gx + cos_y * gy,
                       corner[:, 2] + (gz if gz is not None else torch.zeros_like(gx))
                       + nominal_z), dim=-1)
    return pos, yaw.to(pos.device)


def goal_anchor_world(cmd) -> tuple[torch.Tensor, torch.Tensor]:
    """Where the clip itself finishes, in the world, and the terrain heading to measure in.

    **Dispatches on whether there is a clip at all.**  A manifest-only library refuses every
    gather, so a clip-free run reads its declared route goal instead
    (:func:`freeform_goal_anchor_world`).  Branching here rather than at each of the four call
    sites keeps "where is the goal" one question with one answer.

    Uses the GLOBAL placement -- ``_reset_world_pos`` / ``_reset_world_yaw``, the reset-time
    terrain-cell anchor -- exactly as :func:`base_pose_error_termination` does, and for the
    same reason: the per-step re-anchored placement slides the clip's free root DOFs onto the
    robot every step, so a goal measured against it would follow a stalled robot around and
    always be reachable.

    Args:
        cmd: A :class:`MotionSequenceCommand` (or subclass).

    Returns:
        ``(goal_pos_w [N, 3], frame_yaw [N])`` on the env device.
    """
    from isaaclab.utils.math import quat_from_angle_axis

    if not cmd.has_clips:
        return freeform_goal_anchor_world(cmd)
    yaw = cmd._reset_world_yaw
    world_quat = quat_from_angle_axis(yaw, cmd._world_z.expand(yaw.shape[0], 3))
    state = cmd.reference_world_state_at(
        cmd.clip_id, last_usable_step(cmd), cmd._reset_world_pos, world_quat
    )
    return state[:, 0:3], cmd._mask_frame_yaw.to(state.device)


def goal_travel_m(cmd) -> torch.Tensor:
    """How far the clip ITSELF still travels downrange, from where the episode opens.

    The goal is a forward half-plane at the clip's terminal position, which quietly assumes
    the clip makes forward progress along the terrain frame.  When it does not, the region
    sits BEHIND the spawn: the robot is inside it on its first step, stays there, and scores
    a success having done nothing.  Nothing about the resulting table looks wrong.

    Measured on the shipped library, median travel per episode:

    ==================  ==========  ==================
    family              travel      spawned inside
    ==================  ==========  ==================
    jump_on             +2.68 m     0%
    stairs_down         +1.90 m     0%
    ``jump_off``        **-0.41 m**  **80%**
    ``distractor``      +0.40 m     **80%**
    ==================  ==========  ==================

    ``jump_off`` is the jump-off-a-box-BACKWARDS family, so its downrange coordinate
    decreases by construction; ``distractor`` travels less than a typical approach margin, so
    the margin swallows the whole task.  Both are already excluded everywhere else in this
    pipeline (``DEFAULT_SKIP_FAMILIES`` and ``skip_distractors``), which is the tell.

    This function is what makes that visible for a library nobody has measured yet, rather
    than only for the two cases already known.

    Args:
        cmd: A :class:`MotionSequenceCommand` (or subclass).

    Returns:
        ``[N]`` metres, positive when the goal is ahead of where the episode opened.
    """
    from isaaclab.utils.math import quat_from_angle_axis

    if not cmd.has_clips:
        # The clip-free answer is the SECTION's own downrange run: spawn to goal line, both
        # declared in the route block. There is no motion to ask how far it travels, and the
        # quantity this feeds -- the degenerate-goal / `spawned_in_goal` check -- is asking
        # whether the goal sits ahead of the spawn at all, which the route answers directly.
        from .events.resets import freeform_route  # noqa: PLC0415

        dev = cmd._mask_frame_yaw.device
        route = freeform_route(cmd._env.scene.terrain,
                               torch.arange(cmd._mask_frame_yaw.shape[0], device=dev))
        return route["goal_x"] - route["start_x"]
    goal, yaw = goal_anchor_world(cmd)
    world_quat = quat_from_angle_axis(cmd._reset_world_yaw,
                                      cmd._world_z.expand(cmd._reset_world_yaw.shape[0], 3))
    opened = cmd.reference_world_state_at(
        cmd.clip_id, cmd.start_step.to(cmd.clip_id.device), cmd._reset_world_pos, world_quat)
    down, _, _ = goal_frame_offsets(opened[:, 0:3], goal, yaw)
    return -down


def goal_frame_offsets(
    root_pos_w: torch.Tensor, goal_pos_w: torch.Tensor, frame_yaw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Robot-relative-to-goal displacement, resolved in the TERRAIN's frame.

    Pure geometry, so the debug visualiser, the accumulator and the tests all read one
    definition.  Sign convention: ``downrange`` is positive when the robot is PAST the goal,
    matching how a half-plane test reads.

    Args:
        root_pos_w: ``[N, 3]`` robot base position, world.
        goal_pos_w: ``[N, 3]`` goal anchor, world.
        frame_yaw: ``[N]`` terrain heading (rad), the ``_mask_frame_yaw`` the anchor mask's
            downrange / lateral bits are already resolved in.

    Returns:
        ``(downrange [N], lateral [N], height_err [N])``.  Height is gravity-aligned and
        needs no rotation; it is signed, positive when the robot is above the goal.
    """
    d = root_pos_w - goal_pos_w
    cos_y, sin_y = torch.cos(frame_yaw), torch.sin(frame_yaw)
    downrange = cos_y * d[:, 0] + sin_y * d[:, 1]
    lateral = -sin_y * d[:, 0] + cos_y * d[:, 1]
    return downrange, lateral, d[:, 2]


def _tol_columns(
    tol, like: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Broadcast a tolerance spec to three ``[N]`` tensors.

    Accepts a scalar :class:`GoalTolerances` (one region for every env, which is what a test
    or a single-family probe wants) or an ``[N, 3]`` tensor of
    ``(approach_margin_m, half_width_m, height_tol_m)`` (which is what the eval accumulator
    holds, because the tolerances are per FAMILY and a batch spans several).  One code path
    either way, so the batched call site cannot drift from the scalar one the tests pin.

    Args:
        tol: A :class:`GoalTolerances`, or an ``[N, 3]`` tensor in that column order.
        like: Any ``[N]`` tensor, for device and dtype.

    Returns:
        ``(margin, half_width, height_tol)``, each ``[N]``.

    Raises:
        ValueError: On a tensor of the wrong width -- silently reading three columns out of
            a four-column tensor would apply the wrong bound to two of the three axes.
    """
    if isinstance(tol, GoalTolerances):
        f = lambda v: torch.full_like(like, float(v))  # noqa: E731
        return f(tol.approach_margin_m), f(tol.half_width_m), f(tol.height_tol_m)
    t = torch.as_tensor(tol, device=like.device, dtype=like.dtype)
    if t.ndim != 2 or t.shape[-1] != 3:
        raise ValueError(
            f"per-env tolerances must be [N, 3] (margin, half_width, height_tol); got "
            f"{tuple(t.shape)}."
        )
    return t[:, 0], t[:, 1], t[:, 2]


def in_goal(
    downrange: torch.Tensor, lateral: torch.Tensor, height_err: torch.Tensor, tol,
) -> torch.Tensor:
    """Whether the base is inside the region.

    Args:
        downrange: ``[N]`` from :func:`goal_frame_offsets`.
        lateral: ``[N]``.
        height_err: ``[N]``.
        tol: A :class:`GoalTolerances` or an ``[N, 3]`` per-env tensor -- see
            :func:`_tol_columns`.

    Returns:
        ``[N]`` bool.
    """
    margin, width, height = _tol_columns(tol, downrange)
    return (downrange >= -margin) & (lateral.abs() <= width) & (height_err.abs() <= height)


#: Default clip on the steering controller's yaw rate (rad/s).  0.68 is the largest ``|wz|``
#: the training library ever commands -- on the box family; the two stair families are
#: commanded exactly 0.0 over 3500 clips, so any non-zero rate there is extrapolation whatever
#: the clip.  Matches ``transfer/sim/mjeval/runner.STEER_WZ_MAX`` so the two simulators steer
#: identically.
STEER_WZ_MAX: float = 0.68


def steer_yaw_rate(root_pos_w, root_quat_w, goal_pos_w, in_goal_mask, kp: float,
                   wz_max: float = STEER_WZ_MAX):
    """Proportional yaw rate that points each robot at its goal.

    Nothing in this stack steers: the command is a body-frame velocity and the generator plans
    from depth, so a few degrees of accumulated heading error over several metres becomes a
    lateral miss -- and on a thin flight of stairs a lateral miss is a fall.  This closes that
    one loop so a course is scored on whether the robot could traverse it rather than on
    whether it happened to hold the centreline.

    **It is OFF-DISTRIBUTION on stairs, and that is not a detail.**  Measured over the training
    library the two stair families carry ``|wz| max = 0.0000``; ``flat`` and ``box`` do turn.
    A stair number measured with this on is a different measurement, not a cleaner one, which
    is why every caller must opt in and say so.

    The heading error is taken as the goal's BEARING IN THE BODY FRAME rather than by
    extracting a world yaw and subtracting: ``atan2`` of a body-frame offset is already wrapped
    to ``(-pi, pi]``, and it cannot be got wrong by a quaternion convention.

    Args:
        root_pos_w: ``[N, 3]`` base position, world.
        root_quat_w: ``[N, 4]`` base orientation, world, as ``robot.data.root_quat_w``.
        goal_pos_w: ``[N, 3]`` goal position, world.
        in_goal_mask: ``[N]`` bool; the rate is 0 wherever it is set, so the controller stops
            fighting for the centreline once the course is already cleared.
        kp: Proportional gain (1/s).  ``0`` disables and returns zeros.
        wz_max: Symmetric clip on the output (rad/s).

    Returns:
        ``[N]`` commanded yaw rate (rad/s).
    """
    import torch
    from isaaclab.utils.math import subtract_frame_transforms

    if kp == 0.0:
        return torch.zeros(root_pos_w.shape[0], device=root_pos_w.device)
    goal_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, goal_pos_w)
    err = torch.atan2(goal_b[:, 1], goal_b[:, 0])
    wz = torch.clamp(kp * err, -float(wz_max), float(wz_max))
    return torch.where(in_goal_mask.reshape(-1), torch.zeros_like(wz), wz)


def goal_distance_m(
    downrange: torch.Tensor, lateral: torch.Tensor, height_err: torch.Tensor, tol,
) -> torch.Tensor:
    """Distance to the region, 0 inside it.

    Reported so a near-miss timeout is distinguishable from a robot that never left the spawn
    -- the same reason ``mjeval`` carries ``min_dist_to_target_m``.  Each axis contributes its
    own excess beyond the tolerance, so the metric is the Euclidean distance to the region's
    boundary rather than to its anchor.

    Args:
        downrange: ``[N]`` from :func:`goal_frame_offsets`.
        lateral: ``[N]``.
        height_err: ``[N]``.
        tol: A :class:`GoalTolerances` or an ``[N, 3]`` per-env tensor.

    Returns:
        ``[N]`` non-negative metres.
    """
    margin, width, height = _tol_columns(tol, downrange)
    short = (-margin - downrange).clamp_min(0.0)
    wide = (lateral.abs() - width).clamp_min(0.0)
    high = (height_err.abs() - height).clamp_min(0.0)
    return torch.sqrt(short * short + wide * wide + high * high)
