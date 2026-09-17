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

#: Seconds of the clip's TAIL the downrange anchor is maximised over, ending at its LAST
#: frame -- see :func:`goal_anchor_step`.
GOAL_ANCHOR_WINDOW_S: float = 1.5

#: Stride, in control steps, at which that window is sampled.  3 steps is 60 ms of a 1.5 s
#: window (26 samples); the anchor moves by at most one sample of clip travel, ~5 cm at
#: 2.6 m/s, which is far inside the fitted downrange margin.  Sampling exists because the
#: search runs per reset and each sample is a library gather.
GOAL_ANCHOR_STRIDE: int = 3

#: Seconds the episode is allowed to run PAST the clip's last usable step, so the anchor
#: :func:`goal_anchor_step` returns is actually REACHABLE.
#:
#: ``last_usable_step`` is ``clip_len - 1 - max_offset``, i.e. 1.02 s before the clip's real
#: end, and the anchor now sits in that final window -- so with no slack the goal would be
#: past the deadline and no arm could ever score (measured: the clip arm itself scored
#: **0.001**).  Both the adjuster task and the eval harness's ``clip_end`` term read THIS
#: constant, because a goal definition and the deadline that makes it reachable are one
#: decision and were previously two.
#:
#: **1.5 s, not 3.0, and the difference is measured.**  Past the clip's end the cursor clamps
#: and the command HOLDS the last commanded velocity, so slack is time the robot spends
#: walking with no reference left -- and on a finite tile it walks off.  At 3.0 s the clip
#: arm's fall rate went 0.121 -> 0.352 and the refitted margins blew back out to metre scale
#: (``jump_off_front`` 2.04 m), undoing the very thing the new anchor buys.  1.5 s covers the
#: 1.02 s reserve plus the tracker's own lag with margin to spare: even at the library's top
#: commanded speed of 2.6 m/s it is 3.9 m of travel against a 2.65 m reserve.
GOAL_DEADLINE_SLACK_S: float = 1.5


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

    approach_margin_m: float = 0.25
    half_width_m: float = 2.5
    height_tol_m: float = 0.15


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


def clip_family_codes(cmd) -> tuple[torch.Tensor | None, list[str]]:
    """Per-CLIP family index plus the family names, from the library's ``terrain_manifest``.

    One producer, because two consumers now need it for different reasons and a family read
    two ways is a family that can be read two ways WRONG: ``record_data`` stamps the code into
    every recorded window so ``train.py --skip_families`` can drop a family later, and the
    goal termination looks the tolerances up per family. Both want the same mapping and the
    same ``.h5``-stripped spelling.

    Clips on a tile with no manifest entry -- and every clip of a flat-only library, which
    carries no manifest at all -- get code 0 = ``"unknown"``, which no filter matches.

    Args:
        cmd: The resolved trajectory command term, or anything exposing ``manager.lib``.
            ``None`` (no clips) returns the empty answer rather than raising.

    Returns:
        ``([num_clips] long codes, [family names])``, with ``names[0] == "unknown"``.
        ``(None, [])`` when the library carries no manifest or no terrain ids.
    """
    lib = getattr(getattr(cmd, "manager", None), "lib", None)
    if lib is None or getattr(lib, "terrain_id", None) is None or lib.terrain_id.numel() == 0:
        return None, []
    manifest = lib.attrs.get("terrain_manifest") if hasattr(lib, "attrs") else None
    if not manifest:
        return None, []
    if isinstance(manifest, str):
        manifest = json.loads(manifest)

    # Family name -> code, with 0 reserved for "no manifest entry".
    names = ["unknown"]
    tid_to_code: dict[int, int] = {}
    for entry in manifest:
        if "terrain_id" not in entry:
            continue
        fam = str(entry.get("family") or "unknown")
        # Strip the .h5 the composer stamps, so a filter reads family_jump_off not
        # family_jump_off.h5 -- the extension is an artefact of how the family was stored.
        fam = fam[:-3] if fam.endswith(".h5") else fam
        if fam not in names:
            names.append(fam)
        tid_to_code[int(entry["terrain_id"])] = names.index(fam)

    tid = lib.terrain_id.to(torch.long)
    codes = torch.zeros(tid.shape[0], dtype=torch.long, device=tid.device)
    for t, code in tid_to_code.items():
        codes[tid == t] = code
    return codes, names


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
    clip_len = cmd.manager.clip_len
    step = clip_len[cmd.clip_id.to(clip_len.device)].to(cmd.clip_id.device) - 1
    offsets = cmd.obs_window_offsets()
    if offsets is not None and offsets.numel() > 0:
        step = step - int(offsets.max().item())
    return step.clamp_min(0).long()


def _anchor_step_for(cmd, ids: torch.Tensor) -> torch.Tensor:
    """The max-downrange step in the tail window, for the envs in ``ids``.

    Args:
        cmd: A :class:`MotionSequenceCommand` (or subclass).
        ids: ``[K]`` env indices to solve for.

    Returns:
        ``[K]`` long clip-step indices, on ``cmd.clip_id``'s device.
    """
    from isaaclab.utils.math import quat_from_angle_axis

    dev = cmd.clip_id.device
    clip_len = cmd.manager.clip_len
    clip_id = cmd.clip_id[ids]
    end = (clip_len[clip_id.to(clip_len.device)].to(dev) - 1).clamp_min(0).long()

    dt = float(getattr(cmd.manager, "dt", 0.02))
    n_back = max(int(round(float(GOAL_ANCHOR_WINDOW_S) / dt)), 0)
    back = torch.arange(0, n_back + 1, int(GOAL_ANCHOR_STRIDE), device=dev, dtype=torch.long)
    steps = (end.unsqueeze(-1) - back.unsqueeze(0)).clamp_min(0)          # [K, S]

    yaw = cmd._reset_world_yaw[ids]
    world_quat = quat_from_angle_axis(yaw, cmd._world_z.expand(yaw.shape[0], 3))
    world_pos = cmd._reset_world_pos[ids]
    fy = cmd._mask_frame_yaw[ids].to(dev)
    cos_y, sin_y = torch.cos(fy), torch.sin(fy)

    best = torch.full((ids.shape[0],), -float("inf"), device=dev, dtype=torch.float32)
    best_step = steps[:, 0].clone()
    for j in range(steps.shape[1]):
        pos = cmd.reference_world_state_at(
            clip_id, steps[:, j], world_pos, world_quat)[:, 0:3].to(dev)
        # Only the PROJECTION onto the downrange axis is compared, so the goal position it
        # would be measured against cancels and no anchor is needed to find the anchor.
        down = cos_y * pos[:, 0] + sin_y * pos[:, 1]
        take = down > best
        best = torch.where(take, down, best)
        best_step = torch.where(take, steps[:, j], best_step)
    return best_step


def goal_anchor_step(cmd) -> torch.Tensor:
    """The clip step the goal is anchored on: FURTHEST DOWNRANGE in the clip's tail window.

    Not ``last_usable_step``, and the difference is a measured defect rather than a
    refinement.  ``last_usable_step`` reserves one observation window (1.02 s at
    ``max_offset = 50``), and on the ``jump_on`` family the box mount happens inside that
    reserve: over 120 clips the anchor there sat a median 0.276 m above the spawn against
    0.496 m at the clip's real end, and on **45%** of clips it was below half the clip's own
    maximum rise -- i.e. the mount had not happened yet, so the goal was a point in mid-air
    before the box.  Combined with the old 1.0 m approach margin, **62.5%** of ``jump_on``
    clips let a robot standing on the ground a metre short of the box sit fully inside the
    region and score ``goal_reached`` without ever jumping.

    Anchoring on the tail window instead puts the goal where the clip ENDS UP -- on the box
    it climbed onto -- which is what makes both the downrange half-plane and the height band
    discriminate.  The MAXIMUM over the window rather than its last frame is what keeps that
    true for a clip whose tail drifts backwards; on ``jump_on`` the tail is monotonically
    forward (measured max-minus-final gap 0.000 m), so it costs nothing there and is a
    guard for the families nobody has looked at.

    **This step is past ``last_usable_step``, so the deadline must be extended to reach it**
    -- see :data:`GOAL_DEADLINE_SLACK_S`, which both consumers read.

    Cached per env and recomputed only when ``clip_id`` changes, because the answer is a pure
    function of the clip: the window search costs one library gather per sample and this is
    called every control step by the termination, the critic observation and the accumulator.

    Args:
        cmd: A :class:`MotionSequenceCommand` (or subclass).

    Returns:
        ``[N]`` long, on ``cmd.clip_id``'s device.
    """
    clip_id = cmd.clip_id
    cached = getattr(cmd, "_goal_anchor_step", None)
    cached_id = getattr(cmd, "_goal_anchor_clip_id", None)
    if cached is None or cached_id is None or cached.shape != clip_id.shape:
        stale = torch.ones_like(clip_id, dtype=torch.bool)
        cached = torch.zeros_like(clip_id)
    else:
        stale = cached_id != clip_id
    if bool(stale.any()):
        ids = torch.nonzero(stale).flatten()
        cached = cached.clone()
        cached[ids] = _anchor_step_for(cmd, ids)
        cmd._goal_anchor_step = cached
        cmd._goal_anchor_clip_id = clip_id.clone()
    return cached


def goal_anchor_world(cmd) -> tuple[torch.Tensor, torch.Tensor]:
    """Where the clip ENDS UP, in the world, and the terrain heading to measure in.

    "Ends up" rather than "finishes": the step is :func:`goal_anchor_step`, the furthest
    downrange point of the clip's tail window, NOT ``last_usable_step`` -- which on the box
    families falls before the mount and put the goal in mid-air short of the box.

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

    yaw = cmd._reset_world_yaw
    world_quat = quat_from_angle_axis(yaw, cmd._world_z.expand(yaw.shape[0], 3))
    state = cmd.reference_world_state_at(
        cmd.clip_id, goal_anchor_step(cmd), cmd._reset_world_pos, world_quat
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
