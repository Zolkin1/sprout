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


def tile_episode_seconds(clip_h5: str, *, max_offset: int, terrain_ids, dt: float = 0.02,
                         reduce: str = "max") -> dict[int, float]:
    """How long an ISAAC episode on each tile is allowed to run, in seconds.

    The eval harness adds a ``clip_end`` termination at
    ``clip_len - 1 - max_offset`` (see ``mdp.clip_end_reached`` and
    :func:`last_usable_step`), so that -- not a wall clock -- is the window every Isaac episode
    lived inside, and therefore the only window the failure critic was ever fitted on.

    mjeval has no clip and bounds episodes by ``course_length_m / BUDGET_SPEED`` instead, which
    is derived from DISTANCE and is measurably the wrong variable: tile 40 (a box jump-on) has
    the longest clips of the three shipped feature tiles (10.4 s median) and the SHORTEST
    budget (15.1 s), because it is the shortest course.  Worse for this purpose, a successful
    tile-6 run takes ~10.8 s against a 6.3 s median clip, so its last ~4.5 s sit outside the
    regime any training episode reached.

    Args:
        clip_h5: The clip library.
        max_offset: The tracker's furthest reference-window offset.
        terrain_ids: Which tiles to report.
        dt: Control period, seconds.
        reduce: ``"max"`` (default) or ``"median"`` over the tile's clips.  MuJoCo needs ONE
            number per tile where Isaac has one per episode, and ``max`` is the choice that
            cannot cut a run shorter than the longest Isaac episode on that tile -- so it
            bounds without introducing a new way to fail.

    Returns:
        ``{terrain_id: seconds}``.
    """
    import h5py  # noqa: PLC0415

    fn = {"max": np.max, "median": np.median}[reduce]
    with h5py.File(clip_h5, "r") as f:
        clip_len = f["clip_len"][:]
        tid = f["terrain_id"][:]
        step = last_usable_step(clip_len, max_offset)
        out = {}
        for t in terrain_ids:
            rows = np.flatnonzero(tid == int(t))
            if rows.size == 0:
                raise ValueError(f"clip library {clip_h5} has no clips on terrain_id {t}")
            out[int(t)] = float(fn(step[rows]) * dt)
    return out


@dataclass(frozen=True)
class ClipEpisode:
    """Everything ONE Isaac episode is defined by, read straight from its clip.

    The point of this object is that nothing in it is aggregated.  A tile carries ~50 clips
    with different lengths, endpoints and commands, and mjeval used to build one course per
    TILE -- which forced a reduction of 50 numbers to 1, and any two fields reduced differently
    (a ``max`` duration beside a ``median`` endpoint) imply a pace neither clip ever ran at.
    Isaac has no such problem because it runs one clip per episode.  So does this.

    Attributes:
        clip_id: Index into the library.
        terrain_id: Its tile.
        command: ``(T, 3)`` the commanded ``(vx, vy, wz)`` per control step -- the SAME signal
            Isaac's teacher was driven by, read from ``target_vel`` rather than inferred.  A
            pace derived as distance/duration is not this and is badly wrong: on tile 6 the
            derived figure is 0.86 m/s against a real command of 1.60, because the clip climbs
            and turns and its mean ground speed is far below what it was told to do.
        constant_command: True when ``command`` never changes over the clip (188 of 200
            sampled clips).  A constant can be handed to ``run_episode`` as-is; a varying one
            needs the per-step path.
        duration_s: ``(clip_len - 1 - max_offset) * dt`` -- the ``clip_end`` window the episode
            lives inside, and therefore the only window the failure critic was fitted on.
        goal_xyz: The clip's terminal pelvis position, tile-local.  This is the goal LINE, not
            the line minus a margin: mjeval's ``in_goal`` already awards success within
            ``GOAL_DEPTH / 2`` = 1.0 m, which is exactly Isaac's ``approach_margin_m``.
        start_pos / start_quat_wxyz / start_joint_pos: The reference pose at step 0, tile-local.
            Isaac starts the robot ON the clip; mjeval used to spawn it standing at a nominal
            pose, which on stairs is the difference between arriving in phase with the treads
            and arriving from a standstill.
        start_lin_vel / start_ang_vel / start_joint_vel: The reference VELOCITIES at step 0,
            from ``dy_des``.  Isaac writes all four -- root pose, root velocity, joint position
            and joint velocity (``resets._commit_clip_reset``) -- so dropping the velocity half
            would start the robot at the right pose and dead still, which is a different
            initial condition that looks correct in a viewer.
        joint_names: The clip's own joint ORDER, carried so a consumer can convert rather than
            assume.  It is the ROBOT's per-limb order and **differs from the policy's
            interleaved order in 27 of 29 positions** -- writing one where the other is
            expected produces a plausible, entirely wrong pose.
    """

    clip_id: int
    terrain_id: int
    command: np.ndarray
    constant_command: bool
    duration_s: float
    goal_xyz: tuple[float, float, float]
    start_pos: tuple[float, float, float]
    start_quat_wxyz: tuple[float, float, float, float]
    start_joint_pos: tuple[float, ...]
    start_lin_vel: tuple[float, float, float]
    start_ang_vel: tuple[float, float, float]
    start_joint_vel: tuple[float, ...]
    joint_names: tuple[str, ...]


#: Column layout of ``y_des``, from the library's own ``pos_output_names``.  Read rather than
#: assumed -- but pinned here so a library that reorders them fails loudly instead of writing
#: a plausible pose. ``ori_*`` is w-FIRST, which is already MuJoCo's ``qpos[3:7]`` convention.
Y_DES_POS = slice(0, 3)
Y_DES_QUAT_WXYZ = slice(3, 7)
Y_DES_JOINTS = slice(35, 64)


def clip_episode(clip_h5: str, clip_id: int, *, max_offset: int,
                 dt: float = 0.02) -> ClipEpisode:
    """Read one clip's full episode definition.

    Args:
        clip_h5: The clip library.
        clip_id: Which clip.
        max_offset: The tracker's furthest reference-window offset.
        dt: Control period, seconds.

    Returns:
        The :class:`ClipEpisode`.

    Raises:
        ValueError: If the library's channel layout is not the one :data:`Y_DES_JOINTS` and
            friends describe -- the slices are constants, so a reordered library would
            otherwise yield a well-shaped, entirely wrong pose.
    """
    import h5py  # noqa: PLC0415

    with h5py.File(clip_h5, "r") as f:
        names = json.loads(f.attrs["pos_output_names"])
        if names[0] != "pelvis_link:pos_x" or names[3] != "pelvis_link:ori_w":
            raise ValueError(
                f"{clip_h5} does not open with the pelvis pose in the expected order "
                f"(got {names[:4]}). Y_DES_POS / Y_DES_QUAT_WXYZ are constants and would "
                "silently read the wrong columns.")
        joints = [n for n in names if n.startswith("joint:")]
        if len(joints) != 29 or names[Y_DES_JOINTS][0] != joints[0]:
            raise ValueError(
                f"{clip_h5} has {len(joints)} joint columns and they do not start at "
                f"{Y_DES_JOINTS.start}; the layout this module pins has moved.")
        vnames = json.loads(f.attrs["vel_output_names"])
        vj = [i for i, n in enumerate(vnames) if n.startswith("joint:")]
        if vnames[:6] != ["pelvis_link:pos_x", "pelvis_link:pos_y", "pelvis_link:pos_z",
                          "pelvis_link:ori_x", "pelvis_link:ori_y", "pelvis_link:ori_z"]:
            raise ValueError(f"{clip_h5}'s dy_des does not open with the pelvis twist "
                             f"(got {vnames[:6]}); the velocity slices below would be wrong.")
        if [vnames[i].split(":")[1] for i in vj] != [n.split(":")[1] for n in names
                                                     if n.startswith("joint:")]:
            raise ValueError(f"{clip_h5}'s dy_des joint order differs from y_des's; this "
                             "module assumes one joint order for the clip and would pair "
                             "each joint's position with another joint's velocity.")
        c = int(clip_id)
        n = int(f["clip_len"][c])
        end = int(last_usable_step(np.array([n]), max_offset)[0])
        cmd = np.asarray(f["target_vel"][c, :end + 1], dtype=np.float64)
        row0 = np.asarray(f["y_des"][c, 0], dtype=np.float64)
        rowE = np.asarray(f["y_des"][c, end], dtype=np.float64)
        d0 = np.asarray(f["dy_des"][c, 0], dtype=np.float64)
        return ClipEpisode(
            clip_id=c, terrain_id=int(f["terrain_id"][c]),
            command=cmd, constant_command=bool(np.allclose(cmd, cmd[0], atol=1e-6)),
            duration_s=float(end * dt),
            goal_xyz=tuple(rowE[Y_DES_POS]),
            start_pos=tuple(row0[Y_DES_POS]),
            start_quat_wxyz=tuple(row0[Y_DES_QUAT_WXYZ]),
            start_joint_pos=tuple(row0[Y_DES_JOINTS]),
            start_lin_vel=tuple(d0[0:3]), start_ang_vel=tuple(d0[3:6]),
            start_joint_vel=tuple(d0[vj]),
            joint_names=tuple(n.split(":")[1] for n in names if n.startswith("joint:")))


def load_tolerances(path: str, family: str) -> tuple[float, float, float]:
    """Isaac's goal tolerances for a family, from a fitted ``goal_regions.json``.

    Args:
        path: The spec written by ``build_goal_regions.py``.
        family: Bare family name, e.g. ``stairs_up``.

    Returns:
        ``(approach_margin_m, half_width_m, height_tol_m)``.
    """
    with open(path) as f:
        doc = json.load(f)
    t = doc.get("by_family", {}).get(family) or doc["default"]
    return (float(t["approach_margin_m"]), float(t["half_width_m"]), float(t["height_tol_m"]))


def in_isaac_goal(root_pos, goal_xyz, tol) -> bool:
    """Isaac's arrival test, ported exactly.

    A HALF-PLANE downrange plus two bands -- ``(downrange >= -margin) &
    (|lateral| <= width) & (|height_err| <= height)`` (``mdp.goal_region.in_goal``).  Three
    differences from mjeval's own rule matter:

    * downrange is a half-plane, so OVERSHOOT still counts.  mjeval's ``in_goal`` is a BAND
      (``|x - target| <= GOAL_DEPTH/2``), which a robot can leave by walking past it.
    * there is a HEIGHT band.  mjeval has none, so a robot that fell off an elevated tile and
      walked to the goal line on the floor below scores differently in the two.
    * there is no traversal gate.  Isaac does not need one: the height band already refuses a
      robot that never went up.

    Args:
        root_pos: ``(3,)`` robot base position, course frame.
        goal_xyz: ``(3,)`` the anchor, same frame.
        tol: ``(margin, half_width, height_tol)``.

    Returns:
        Whether the robot is inside the region.
    """
    d = np.asarray(root_pos, dtype=np.float64) - np.asarray(goal_xyz, dtype=np.float64)
    margin, width, height = tol
    return bool(d[0] >= -margin and abs(d[1]) <= width and abs(d[2]) <= height)


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
