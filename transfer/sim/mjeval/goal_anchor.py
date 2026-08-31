# Copyright 2026 Zachary Olkin. All rights reserved.

"""Where a clip-driven episode has to physically GET TO, made available to MuJoCo.

The failure critic (``robot_rl.flow_matching.value``) reads three scalar goal features --
downrange / lateral / height error against a per-tile anchor -- and they are Isaac quantities:
``mdp.goal_region.goal_anchor_world`` derives them from a live ``MotionSequenceCommand``.  The
mjeval courses have no clip and no command, which is why the critic could not be scored in
MuJoCo and why ``fm_sample_explorer.py`` records outcomes rather than critic readings.

**The anchor is recoverable exactly, and it is not a fit.**  ``goal_anchor_world`` places the
clip's own terminal pelvis pose at the reset-time terrain-cell anchor, so in the TILE's frame
it is simply that terminal pose.  Measured over the v2 rollout -- 14703 episodes, 2289 clips,
75 real tiles::

    goal_pos_w - y_des[clip, last_usable_step, 0:3]   is CONSTANT within a tile
        max within-tile spread   x 7.6e-06   y 1.5e-05   z 0.0   m

and that constant is the tile corner.  So the anchor in tile-local coordinates IS the clip's
terminal pelvis position, reachable from the clip library alone -- no layout rebuild, no torch,
no IsaacLab, nothing that needs a simulator.

**Do NOT use ``Course.target_xy`` for this.**  It is the far walkable platform's CENTRE
(``terrains.isaac_tile_course``), and those platforms are 7.5-10.1 m deep while the clip steps
only 0.8-2.1 m onto them.  Measured bias against the real anchor, per family::

    stairs_up      -3.056 m   (-4.187 .. -2.785)
    stairs_down    -3.166 m   (-5.070 .. -2.605)
    jump_on        -0.442 m   (-1.202 .. -0.359)
    jump_off_front +4.917 m   -- sign flips; excluded, see below

Against ``GOAL_SCALE_M`` that is 0.61-0.63 of a normalized unit of pure offset on a feature the
critic reads directly.  ``target_xy`` keeps owning ``in_goal`` / success and is untouched, so
every mjeval number measured before this module stays comparable; the two goals answer
different questions and are deliberately not unified.

**What a per-tile MEDIAN costs, measured against the per-clip Isaac anchor** (``|x - median|``
over each tile's clips -- the error the critic feature actually carries)::

    family        median   p90     max   |  target_xy bias   ratio at p90
    jump_on        0.000  0.000   0.000  |          0.442    exact
    stairs_down    0.083  0.487   1.520  |          3.166    6.5x better
    stairs_up      0.176  0.718   1.652  |          3.056    4.3x better

so 0.017-0.035 context units typical against ``target_xy``'s flat 0.61-0.63.  The residual is
real clip-to-clip variation in where the motion ends, and it is **inside the distribution the
critic was fitted on** -- Isaac's anchor is per CLIP, so the critic already saw that spread
across the clips of a tile.  ``target_xy``'s error is a systematic offset, which it did not.

**Tile -1 is refused, not approximated.**  It is the FLAT sentinel: those clips carry
``terrain_id == -1`` and are placed with no tile frame at all, so the identity above is
undefined there rather than merely noisy (measured spread 129 m, against 1.5e-05 on every real
tile).  162 clips of the v2 rollout are in that bucket.

**``jump_off_front`` is excluded** on three independent grounds: the residual sign flips, 7 of
its 15 tiles cannot be built as a course at all (``spawn_point`` puts the spawn outside the
tile footprint), and the generator the critic was fitted against is ``..._nojumpoff``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

#: Divisor turning metres into the critic's context units.  Mirrors ``fit_value.GOAL_SCALE_M``;
#: a DECLARED scale rather than a fitted statistic, so "far" is a property of the robot rather
#: than of whichever rollout was on disk -- the same idiom as the generator's `contract`
#: exteroception normalization.
GOAL_SCALE_M = 5.0

#: Families whose anchor does not transfer.  See the module docstring.
EXCLUDED_FAMILIES = ("jump_off_front", "jump_off")

#: Largest within-tile spread of the recovered anchor that still counts as "constant".  The
#: measured worst case is 1.5e-05 m, so this is four orders of slack -- it is a tripwire for
#: the identity having broken (a different placement rule, a re-generated library), NOT a
#: tolerance anything is expected to sit near.
_CONSTANT_ATOL_M = 1e-3


@dataclass(frozen=True)
class TileAnchor:
    """One tile's goal anchor, in the tile's own local-corner frame.

    Attributes:
        terrain_id: The tile this describes.
        anchor: ``(3,)`` median terminal pelvis position over the tile's clips, metres.
        spread: ``(3,)`` peak-to-peak of that position across those clips, metres.  This is
            real clip-to-clip variation in where the motion ends, NOT placement error -- the
            placement contributes 1.5e-05 m.  Isaac's anchor is per CLIP, so the critic was
            trained across exactly this spread and a per-tile median sits inside its
            distribution rather than beside it.
        n_clips: How many clips the median is over.
    """

    terrain_id: int
    anchor: tuple[float, float, float]
    spread: tuple[float, float, float]
    n_clips: int


def last_usable_step(clip_len: np.ndarray, max_offset: int) -> np.ndarray:
    """The step a SURVIVING episode ends on, lookahead reserve included.

    The numpy twin of ``mdp.goal_region.last_usable_step``, re-derived from the same two
    sources rather than restated: the clip's own length and the reference window's furthest
    offset.  Anchoring on ``clip_len - 1`` instead would place the goal up to one full
    observation window past where any episode can reach.

    Args:
        clip_len: ``(C,)`` per-clip length in steps.
        max_offset: The reference window's furthest look-ahead offset, in steps.

    Returns:
        ``(C,)`` int64, never below 0.
    """
    return np.maximum(clip_len.astype(np.int64) - 1 - int(max_offset), 0)


def tile_anchors(clip_h5: str, *, max_offset: int,
                 terrain_ids: np.ndarray | None = None) -> dict[int, TileAnchor]:
    """Recover the per-tile goal anchor from a clip library.

    Args:
        clip_h5: A clip library carrying ``y_des``, ``clip_len`` and ``terrain_id``.
        max_offset: The reference window's furthest offset (50 for every current tracker).
            Stated rather than guessed: it changes ``last_usable_step`` and therefore the
            anchor, and a wrong value produces a plausible anchor a metre or so off.
        terrain_ids: Restrict to these tiles, or ``None`` for every real tile in the library.

    Returns:
        ``{terrain_id: TileAnchor}``, tile-local frame.  Tile ``-1`` is never included.

    Raises:
        ValueError: If a tile's recovered anchor is not constant to
            :data:`_CONSTANT_ATOL_M` -- which would mean the clip-terminal identity this
            module rests on no longer holds for that library.
    """
    import h5py  # noqa: PLC0415  -- keeps the module importable without h5py for the offsets half

    with h5py.File(clip_h5, "r") as f:
        clip_len = f["clip_len"][:]
        tid = f["terrain_id"][:]
        step = last_usable_step(clip_len, max_offset)
        want = (np.unique(tid[tid >= 0]) if terrain_ids is None
                else np.asarray(sorted({int(t) for t in terrain_ids})))
        out: dict[int, TileAnchor] = {}
        for t in want:
            t = int(t)
            if t < 0:
                # The flat sentinel. Refused rather than approximated -- see the docstring.
                raise ValueError(
                    "terrain_id -1 is the FLAT sentinel: its clips are placed with no tile "
                    "frame, so a goal anchor is undefined there rather than merely noisy "
                    "(measured within-'tile' spread 129 m against 1.5e-05 m on real tiles).")
            rows = np.flatnonzero(tid == t)
            if rows.size == 0:
                raise ValueError(f"clip library {clip_h5} has no clips on terrain_id {t}")
            # Read one clip at a time: y_des is (C, T, 155) and slicing the whole tile would
            # pull hundreds of MB for three columns.
            p = np.stack([f["y_des"][int(c), int(step[int(c)]), 0:3] for c in rows]).astype(np.float64)
            spread = p.max(axis=0) - p.min(axis=0)
            out[t] = TileAnchor(terrain_id=t, anchor=tuple(np.median(p, axis=0)),
                                spread=tuple(spread), n_clips=int(rows.size))
    return out


def frame_offsets(root_pos: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """``(downrange, lateral, height_err)`` -- the numpy twin of ``goal_frame_offsets``.

    The Isaac form rotates the displacement into the terrain's heading.  **Every mjeval course
    runs +x and every Isaac episode in the v2 rollout recorded ``goal_frame_yaw == 0.0``
    exactly** (checked over all 15129 of them), so on both sides the rotation is the identity
    and this is plain subtraction.  That is a measured coincidence of the current terrain
    generator rather than a law, which is why :func:`assert_yaw_free` exists and why this
    function takes no yaw: silently applying a zero rotation would hide a library that
    started authoring turned tiles.

    Sign convention matches Isaac's: ``downrange`` is positive when the robot is PAST the
    goal, and ``height_err`` is positive when it is above.

    Args:
        root_pos: ``(..., 3)`` robot base position, in the same frame as ``anchor``.
        anchor: ``(3,)`` or ``(..., 3)`` goal anchor.

    Returns:
        ``(..., 3)`` stacked ``(downrange, lateral, height_err)``, metres.
    """
    return np.asarray(root_pos, dtype=np.float64) - np.asarray(anchor, dtype=np.float64)


def context_goal(root_pos: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """The critic's first three context channels: :func:`frame_offsets` on the declared scale.

    Mirrors ``fit_value._batch``, including its ``nan_to_num`` -- an episode with no goal
    records NaN there, and one non-finite value takes the whole batch's logit with it.

    Args:
        root_pos: ``(..., 3)`` robot base position.
        anchor: ``(3,)`` or ``(..., 3)`` goal anchor.

    Returns:
        ``(..., 3)`` float32, divided by :data:`GOAL_SCALE_M`.
    """
    g = frame_offsets(root_pos, anchor) / GOAL_SCALE_M
    return np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def assert_yaw_free(goal_frame_yaw: np.ndarray, *, atol: float = 1e-9) -> None:
    """Refuse a rollout whose terrain frames are turned.

    :func:`frame_offsets` drops the rotation because it is measurably the identity today.  If
    a future library authors turned tiles, that assumption becomes a silent lateral/downrange
    swap of up to the full displacement -- shape-correct, finite, and wrong.  Call this
    wherever a recorded ``goal_frame_yaw`` is in hand.

    Args:
        goal_frame_yaw: Recorded terrain headings, radians.
        atol: Largest heading still treated as zero.

    Raises:
        ValueError: If any heading is non-zero.
    """
    y = np.abs(np.asarray(goal_frame_yaw, dtype=np.float64))
    if y.size and y.max() > atol:
        raise ValueError(
            f"goal_frame_yaw is not identically zero (max |yaw| = {y.max():.3e} rad over "
            f"{y.size} entries). `frame_offsets` drops the terrain rotation because it has "
            "measured as exactly zero on every tile; a turned tile makes that a silent "
            "downrange/lateral swap. Port the rotation before using this module on it.")


def save(anchors: dict[int, TileAnchor], path: str, *, provenance: dict) -> None:
    """Write an anchor table as JSON, with the provenance that makes it checkable.

    The table is an ARTIFACT rather than a recomputation so a course and a critic reading can
    be traced to the library they came from -- the same reason the clip pack and the camera
    contract are artifacts.

    Args:
        anchors: What :func:`tile_anchors` returned.
        path: Destination ``.json``.
        provenance: At minimum the source library path, its sha256, and ``max_offset``.
    """
    for k in ("clip_h5", "clip_sha256", "max_offset"):
        if k not in provenance:
            raise ValueError(f"provenance must carry {k!r}: an anchor table that cannot name "
                             "the library and lookahead it came from is not checkable, and a "
                             "wrong max_offset yields a plausible anchor about a metre off")
    doc = {"schema": 1, "provenance": provenance, "goal_scale_m": GOAL_SCALE_M,
           "excluded_families": list(EXCLUDED_FAMILIES),
           "tiles": {str(t): {"anchor": list(a.anchor), "spread": list(a.spread),
                              "n_clips": a.n_clips} for t, a in sorted(anchors.items())}}
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)


def load(path: str) -> tuple[dict[int, TileAnchor], dict]:
    """Read back what :func:`save` wrote.

    Args:
        path: The ``.json``.

    Returns:
        ``({terrain_id: TileAnchor}, provenance)``.
    """
    with open(path) as f:
        doc = json.load(f)
    tiles = {int(k): TileAnchor(terrain_id=int(k), anchor=tuple(v["anchor"]),
                                spread=tuple(v["spread"]), n_clips=int(v["n_clips"]))
             for k, v in doc["tiles"].items()}
    return tiles, doc["provenance"]
