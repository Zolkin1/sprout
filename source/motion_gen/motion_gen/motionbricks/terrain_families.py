# Copyright 2026 Zachary Olkin. All rights reserved.

"""What differs between terrain skills, in one place: the family table + the boundary-frame gates.

Every Motion-Bricks terrain clip has the SAME shape -- ``MB approach -> replayed terrain -> MB exit`` --
and the families differ only in a handful of declarative facts (does the traversal loop, which walkable
level does each end run on, which way does each end walk) plus HOW the two boundary frames of the replay
are chosen:

* **Episodic** interactions (a box mount / dismount) are played whole, so the boundaries are simply the
  first and last frame (:func:`episodic_boundaries`).
* **Periodic** interactions (a stair stride looped up/down a staircase) have no natural first/last frame:
  cutting anywhere would start a foot inside a tread or end it on a phantom stair past the tile.  So the
  boundary frames are FILTERED by contact + geometry (:func:`boundary_candidates`): a foot planted on a
  walkable level with the other foot a small height above it -- an ordinary single-support mid-stride pose
  the flat MB segment can hand off to / pick up from.

TWO VELOCITIES, ONE AUTHORED.  A terrain clip has a velocity the GENERATOR uses and a velocity the RL
SEES, and they are deliberately different:

* The generator's command is whatever the terrain needs -- notably it ramps DOWN into a box takeoff,
  because Motion Bricks pins the root to the exact-velocity rollout of the command, so a steady command
  walks the robot into the box at full speed (measured: arrives at 1.14 m/s where the replay's entry
  frame is at 0.11).  Releasing that root pin near the boundary does not fix it; the deceleration has to
  be prescribed.  It is derived per clip from the chosen reference, and never leaves the generator.
* The RL / flow-matching command is what a USER or PLANNER would ask for: one steady velocity, ignorant
  of the terrain.  Adapting it to the terrain is the flow model's job, and the FM can only learn that if
  the clips pair a terrain-agnostic command with a terrain-aware motion.

So a family carries no "what does the RL see over the replay" knob -- that is one constant per clip,
written by :mod:`scripts.rsl_rl.mb_terrain_clips`.

This module is numpy-only (no torch, no Motion Bricks, no IsaacLab imports) so it loads in the Motion
Bricks conda env as well as in ``env_isaaclab`` -- the terrain plan (:mod:`.terrain_plan`) is built on one
side and consumed on the other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Height of the ankle_roll_link frame origin above the sole with the foot flat on a surface (m).  Sole
# height = ankle-frame z - this, which is what the contact gates compare against a tread / box top.
SOLE_Z = 0.034


@dataclass(frozen=True)
class TerrainFamily:
    """Declarative description of one terrain skill for the Motion-Bricks terrain pipeline.

    ``lower`` / ``upper`` name the two walkable LEVELS a tile has: for stairs the bottom platform (flush
    with the lowest tread) and the top platform (flush with the highest tread); for a box the ground and
    the box top (which likewise carries a platform flush with it, so the elevated segment can walk).  A
    family says which level its approach runs on and which its exit runs on -- everything directional
    (which tread the traversal peels off, which surface each MB segment is grounded to, which side of the
    box its platform extends to) follows from that, so no generator carries per-skill branches.
    """

    skill: str                  # terrain library skill name (manifest "skill" column)
    geom: str                   # "stairs" | "box" -- which tile geometry to author
    periodic: bool              # True => loop the stride onto the tile; False => play once (episodic)
    going_up: bool              # stairs: +x runs uphill.  Unused (False) for the box family
    approach_level: str         # "lower" | "upper" -- level the MB approach walks/stands on
    exit_level: str             # "lower" | "upper" -- level the MB exit walks/stands on
    approach_mode: str          # "walk" (locomote into the terrain) | "stand" (settle in place on it)
    exit_mode: str              # "walk" (locomote away) | "stand" (arrived; hold position)
    approach_sign: float        # approach travel direction along tile +x: +1 forward, -1 backward
    exit_sign: float            # exit travel direction along tile +x: +1 forward, -1 backward
    pin_future: bool            # approach pin tracks the replay's FUTURE frames (see below) vs holds the entry
    blurb: str                  # one-line human description (printed by the drivers)


# ``pin_future``: the approach's target keyframe sits ~1 s ahead of the robot, so near the hand-off it can
# either keep walking forward into the replay's own future frames or hold the entry pose.  Tracking the
# future keeps the arrival MOVING at the entry velocity, which is what an ASCENT wants (its future rises, and
# Motion Bricks partly follows the rise into the climb).  A DESCENT's future drops BELOW the platform the
# approach is still walking on, so tracking it drags the robot off its surface before the seam; and an
# EPISODIC boundary's future is the takeoff itself, so tracking it generates a half-jump the splice then has
# to undo.  Measured boundary-pose error (joint L2 at the seam, lower is better):
#   stairs_up   future 0.66 rad | hold 1.05 | unpinned 1.56
#   stairs_down future 1.70 rad | hold 0.79 | unpinned 1.68


# ``jump_on`` is the only MOUNT in the library (there is no backward mount), so the box family covers
# mount + both dismount directions: ``jump_off`` steps off backwards, ``jump_off_front`` forwards.
#
# Every box family WALKS on both sides, exactly like the stair families: the tile authors a short walkable
# platform flush with the box top (:func:`~motion_gen.terrain.box_clip.box_tile_boxes`), so the
# elevated segment has somewhere to go.  It used to be a standing hold -- a mount ended by standing on the
# box, a dismount began by standing on it -- and because both MB segments carry the ego anchor mask, a
# standing hold RE-ANCHORS onto whatever pose the robot actually reached: missing the box cost nothing and
# the RL command fell to zero.  A walking end demands the traversal be completed, as a staircase does.
FAMILIES: dict[str, TerrainFamily] = {
    "stairs_up": TerrainFamily(
        skill="stairs_up", geom="stairs", periodic=True, going_up=True,
        approach_level="lower", exit_level="upper", approach_mode="walk", exit_mode="walk",
        approach_sign=1.0, exit_sign=1.0, pin_future=True,
        blurb="walk in on the bottom platform, climb the treads, walk off the top platform"),
    "stairs_down": TerrainFamily(
        skill="stairs_down", geom="stairs", periodic=True, going_up=False,
        approach_level="upper", exit_level="lower", approach_mode="walk", exit_mode="walk",
        approach_sign=1.0, exit_sign=1.0, pin_future=False,
        blurb="walk in on the top platform, descend the treads, walk off the bottom platform"),
    "jump_on": TerrainFamily(
        skill="jump_on", geom="box", periodic=False, going_up=False,
        approach_level="lower", exit_level="upper", approach_mode="walk", exit_mode="walk",
        approach_sign=1.0, exit_sign=1.0, pin_future=False,
        blurb="walk in on the ground, jump onto the box, walk on along the box top"),
    "jump_off": TerrainFamily(
        skill="jump_off", geom="box", periodic=False, going_up=False,
        approach_level="upper", exit_level="lower", approach_mode="walk", exit_mode="walk",
        approach_sign=-1.0, exit_sign=-1.0, pin_future=False,
        blurb="walk BACKWARDS along the box top, drop off backwards, walk away backwards"),
    "jump_off_front": TerrainFamily(
        skill="jump_off_front", geom="box", periodic=False, going_up=False,
        approach_level="upper", exit_level="lower", approach_mode="walk", exit_mode="walk",
        approach_sign=1.0, exit_sign=1.0, pin_future=False,
        blurb="walk FORWARDS along the box top, drop off forwards, walk away forwards"),
}


def family_for(skill: str) -> TerrainFamily:
    """The :class:`TerrainFamily` for a terrain library ``skill``, or a clear error listing what exists."""
    try:
        return FAMILIES[skill]
    except KeyError:
        raise KeyError(f"no Motion-Bricks terrain family for skill {skill!r}; "
                       f"have {sorted(FAMILIES)}") from None


def elevated_side(family: TerrainFamily) -> str:
    """Which MB segment runs on the UPPER level -- ``"approach"`` or ``"exit"``.

    The one fact a ONE-SIDED clip needs.  "Terrain plus the ground side only" means omitting the segment
    on the upper level, and which one that is differs per family: a MOUNT lands up there so it is the
    exit, a DISMOUNT starts up there so it is the approach.  Reading it off ``*_level`` keeps the
    one-sided path family-agnostic instead of branching on the skill name.

    Raises:
        ValueError: if neither end is elevated (both levels ``"lower"``), which no family does today.
    """
    if family.exit_level == "upper":
        return "exit"
    if family.approach_level == "upper":
        return "approach"
    raise ValueError(f"{family.skill}: neither end runs on the upper level, so no side is elevated")


def ground_side(family: TerrainFamily) -> str:
    """Which MB segment runs on the LOWER level -- the complement of :func:`elevated_side`."""
    return "approach" if elevated_side(family) == "exit" else "exit"


def box_platform_sign(family: TerrainFamily) -> float:
    """Which side of the jump box (``+1`` = ``+x``, ``-1`` = ``-x``) the walkable TOP PLATFORM extends to.

    Derived rather than declared, because it is fully determined by which end of the interaction is
    elevated and which way that end travels:

    * a MOUNT (``exit_level == "upper"``) lands near the face it cleared and walks AWAY along
      ``exit_sign``, so the platform must lie ahead of the landing, at ``+exit_sign``;
    * a DISMOUNT (``approach_level == "upper"``) arrives at the edge it drops off travelling along
      ``approach_sign``, so the platform it walked in on lies BEHIND it, at ``-approach_sign``.

    Either way the platform is on the opposite side of the box from the face the jump uses, which is what
    leaves the takeoff / landing footprint exactly as the reference solved it.

    Raises:
        ValueError: if ``family`` is not a box family (a staircase's platforms come from its tread layout).
    """
    if family.geom != "box":
        raise ValueError(f"{family.skill}: box top platforms are a box-family notion, not {family.geom!r} "
                         f"(stair platforms come from stair_tile_boxes)")
    return float(family.exit_sign) if family.exit_level == "upper" else -float(family.approach_sign)


@dataclass(frozen=True)
class BoundaryGate:
    """Result of a boundary-frame filter: which frames the replay may start / stop at, and how it got them."""

    frames: np.ndarray          # [K] int candidate frame indices (ascending, possibly empty)
    strict: bool                # True => single-support candidates; False => the contact-only fallback
    n_planted: int              # frames with SOME foot planted on a level (before the swing-foot test)
    reason: str                 # one-line diagnostic for the generator log


def sole_heights(y: np.ndarray, pos_names: list[str],
                 frames: tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link"),
                 sole_offset: float = SOLE_Z) -> np.ndarray:
    """Per-frame SOLE heights ``[T, 2]`` (left, right) from a built clip-output block ``y[T, P]``.

    ``y`` is what :class:`~motion_gen.clip_output.ClipOutputBuilder` produces (the
    ``pos_output_names`` layout); the ankle-frame z is lowered by ``sole_offset`` so the value can be
    compared directly against a tread / box top height.
    """
    cols = [pos_names.index(f"{f}:pos_z") for f in frames]
    return np.asarray(y, dtype=np.float64)[:, cols] - float(sole_offset)


def boundary_candidates(soles: np.ndarray, contact: np.ndarray, levels, *,
                        tol: float = 0.02, swing_clear: float = 0.05, swing_band: float = 0.15,
                        window: tuple[int, int] | None = None) -> BoundaryGate:
    """Frames where the traversal is in a clean SINGLE-SUPPORT pose on a walkable level.

    The hand-off test, in both directions: one foot is IN CONTACT and its sole sits (within ``tol``) on one
    of the walkable ``levels``, while the other (swing) foot is clear ABOVE that same level -- lifted by
    more than ``swing_clear`` (so an eased seam cannot drag it through the step) but by no more than
    ``swing_band`` (so it is not near its apex a full riser up, which a flat segment would have to yank
    back down).  That window reads like an ordinary flat mid-stride, which is exactly what makes it
    spliceable onto a flat Motion-Bricks segment.

    ``levels`` may be a single height (the one surface the boundary must sit on -- e.g. the top tread an
    ascent peels off) or a list of heights (any tread, e.g. for an ENTRY whose stance tread is whichever
    the placement will anchor onto the tile's first tread).

    If no frame passes, falls back to the contact-only test (some foot planted on a level) and says so via
    :attr:`BoundaryGate.strict`, so the caller can warn instead of failing outright.

    Args:
        soles: ``[T, 2]`` sole heights (see :func:`sole_heights`).
        contact: ``[T, 2]`` per-foot in-contact flags (nonzero == in contact).
        levels: walkable surface height(s) the stance foot may rest on (m).
        tol: how close the stance sole must be to a level to count as planted (m).
        swing_clear / swing_band: allowed swing-foot height above the stance level (m).
        window: optional ``(lo, hi)`` half-open frame range to restrict candidates to.

    Returns:
        :class:`BoundaryGate`.
    """
    soles = np.asarray(soles, dtype=np.float64)
    contact = np.asarray(contact)
    if soles.ndim != 2 or soles.shape[1] != 2:
        raise ValueError(f"soles must be [T, 2], got {soles.shape}")
    if contact.shape[:1] != soles.shape[:1]:
        raise ValueError(f"contact {contact.shape} does not match soles {soles.shape}")
    lv = np.atleast_1d(np.asarray(levels, dtype=np.float64)).reshape(-1)
    T = soles.shape[0]

    nearest = lv[np.abs(soles[:, :, None] - lv[None, None, :]).argmin(axis=-1)]   # [T,2] level under each foot
    planted = (contact[:, :2] > 0) & (np.abs(soles - nearest) < float(tol))       # [T,2]
    swing = soles[:, ::-1] - nearest                     # the OTHER foot's height above this foot's level
    ok_swing = (swing > float(swing_clear)) & (swing <= float(swing_band))
    strict_mask = (planted & ok_swing).any(axis=1)
    planted_mask = planted.any(axis=1)

    keep = np.ones(T, dtype=bool)
    if window is not None:
        lo, hi = int(window[0]), int(window[1])
        keep[:] = False
        keep[max(0, lo):max(0, hi)] = True

    frames = np.nonzero(strict_mask & keep)[0]
    if frames.size:
        return BoundaryGate(frames=frames.astype(np.int64), strict=True, n_planted=int(planted_mask.sum()),
                            reason=f"{frames.size} single-support candidates on levels "
                                   f"{np.round(lv, 3).tolist()}")
    frames = np.nonzero(planted_mask & keep)[0]
    return BoundaryGate(
        frames=frames.astype(np.int64), strict=False, n_planted=int(planted_mask.sum()),
        reason=(f"no single-support frame (swing band ({swing_clear:.3f}, {swing_band:.3f}] m); "
                f"fell back to {frames.size} contact-only candidates"))


def episodic_boundaries(n_frames: int) -> tuple[int, int]:
    """Boundary frames of an EPISODIC interaction: the whole thing plays, so ``(0, n_frames - 1)``.

    A box mount starts standing on the ground and ends standing on the box (and a dismount the reverse):
    both ends are already settled double-support poses solved for exactly that surface, so there is
    nothing to filter -- the approach hands off at frame 0 and the exit picks up at the last frame.
    """
    if n_frames < 2:
        raise ValueError(f"episodic interaction needs >= 2 frames, got {n_frames}")
    return 0, n_frames - 1


def pick_boundary(gate: BoundaryGate, *, prefer: str = "first", rng=None) -> int:
    """Choose ONE frame from a candidate gate -- the hand-picked hard-pin frame.

    ``prefer="first"`` takes the earliest candidate: for an ENTRY that is the earliest clean hand-off in
    the first stride (least flat walking replaced by replayed terrain), and for an EXIT the first clean
    arrival on the far surface (the traversal stops as soon as it has landed, never stepping onto a
    phantom tread past the tile).  ``prefer="last"`` takes the latest, ``prefer="middle"`` the median
    candidate (a phase in the interior of the window, away from both edges).

    ``prefer="sample"`` draws uniformly from the candidates using ``rng``, and is what an ENTRY should
    use.  The entry frame IS the gait phase at which the approach hands over to the terrain, and it is
    an input we choose, not something recovered afterwards -- with a fixed preference every clip sharing
    a reference enters at the identical phase, so the teacher and the flow-matching model only ever see
    one hand-off pose out of the whole gate.  The gate has already filtered the candidates down to
    frames that stand on a tread with the swing foot clear, so any of them is a legal entry.
    """
    if gate.frames.size == 0:
        raise ValueError(f"no boundary candidates to pick from ({gate.reason})")
    if prefer == "first":
        return int(gate.frames[0])
    if prefer == "last":
        return int(gate.frames[-1])
    if prefer == "middle":
        return int(gate.frames[gate.frames.size // 2])
    if prefer == "sample":
        if rng is None:
            raise ValueError("prefer='sample' needs an rng")
        return int(gate.frames[int(rng.integers(0, gate.frames.size))])
    raise ValueError(f"unknown boundary preference {prefer!r} (first | last | middle | sample)")
