# Copyright 2026 Zachary Olkin. All rights reserved.

"""Place a PLAYED terrain interaction onto a FIXED tile -- the shared source of truth.

Every terrain clip generator (the DP matcher in ``scripts/rsl_rl/generate_box_clips.py``, the Motion
Bricks drivers in ``scripts/rsl_rl/mb_terrain_*.py``) plays the same traj-opt terrain references onto
the same authored tile geometry (:mod:`.box_clip`), so the *placement* -- looping a periodic stride onto
exact treads, anchoring an episodic jump onto the tile box -- must be ONE implementation:

* :func:`place_periodic_on_tile` -- a periodic stair stride looped down/up the tile's treads with an
  integer-tread per-period offset (drift-free re-pin), anchored so the stance foot starts on the tile's
  first tread exactly as the source solved it.
* :func:`place_episodic_on_tile` -- an episodic box interaction (``jump_on`` / ``jump_off*``) played once
  and anchored by its solved box onto the tile's fixed box.
* :func:`episodic_box_extents` -- how big that solved box is in the frame the placement will put it in,
  so the tile can be authored before the interaction is placed on it.
* :func:`select_terrain_reference` -- the two-stage reference pick (nearest geometry, then nearest speed)
* :func:`terrain_speed_cell` -- the speeds that pick would return the SAME reference for
* :func:`speed_cell_from` -- the midpoint rule underneath it, shared with the pure-stair path
  both generators use.

All functions are pure torch + math: they take an ingested :class:`~.motion_db.MotionSource` plus the
authored tile geometry and return the placed ``(root_pos, root_quat, joints, contact)`` in the TILE
frame.  Emitting the geometry / anchor mask / manifest is the caller's job.
"""

from __future__ import annotations

import math

import torch

from .box_clip import align_and_concat, loop_periodic, periodic_tread_offset, yaw_normalize


def _tread_boxes(boxes: list[dict]) -> list[int]:
    """Indices of the NARROW tread boxes in ``boxes`` (i.e. not the wide approach / recovery platforms).

    A transition tile's platforms are authored FLUSH with the extreme treads (same top z) but WIDE, so an
    unrestricted ``argmax/argmin(top z)`` ties and can pick a platform -- whose center x is far from the
    tread column -- anchoring the whole traversal onto the platform instead of the stairs.  Every stage of
    the pipeline tells treads from platforms by width, so that test lives here once.
    """
    min_sx = min(float(b["size"][0]) for b in boxes)
    return [i for i in range(len(boxes)) if float(boxes[i]["size"][0]) <= 1.5 * min_sx]


def place_periodic_on_tile(ts, tile, boxes, dev, k: int = 0, truncate: bool = True,
                           going_up: bool = False, end_buffer_treads: int = 0):
    """Loop the source stair stride from frame ``k`` and place it on the FIXED tile treads.

    ``truncate`` (pure-tile path): cut the traversal once it reaches the far tread, so the clip ends
    there.  The transition path passes ``truncate=False`` and instead peels off via a contact gate (which
    needs the settled frames ON the far tread, so it keeps the full looped traversal).

    Loops enough source strides to cover the tile's N treads (``n_loops`` = tile #steps / the source
    stride's treads-per-period; NOT ``box_clip.n_periodic_loops``, whose box-CENTER z-span under-reads our
    solid floor-to-top tread boxes ~2x).  The per-loop offset is the SOURCE stride's own-frame integer-tread
    step (:func:`.box_clip.periodic_tread_offset`); :func:`.box_clip.yaw_normalize` then rotates the whole
    looped traversal to +x.  Finally ANCHORS the source's frame-``k`` stance tread (the tread it stands on
    in the source's OWN geometry) onto the tile's first tread by tread-TOP, preserving the source's
    foot->tread stance -- so the robot starts ON that tread exactly as the trajectory source has it,
    not floating mid-staircase.  Any ``k`` in the period is a valid start: the anchor moves with it.

    Args:
        ts: ingested terrain :class:`~.motion_db.MotionSource` (periodic stair stride, with contacts).
        tile: the sampled :class:`~.terrain_sampler.TerrainSpec` (dims + #steps).
        boxes: the authored tile geometry (:func:`.box_clip.stair_tile_boxes`).
        dev: torch device for the placed tensors.
        k: source frame to start the traversal at (the ENTRY frame).
        truncate: stop at the far tread (pure path) instead of running onto phantom treads.
        going_up: ascent (+x uphill); descent otherwise.
        end_buffer_treads: with ``truncate``, leave this many treads ahead of the robot at the clip end.

    Returns:
        ``(root_pos[T,3], root_quat[T,4], joints[T,29], contact[T,2] | None)`` in the tile frame.
    """
    src_boxes = ts.terrain_meta.get("terrain", {}).get("boxes", [])
    # Treads the source covers per PERIOD -- measured over the full period (frame 0 -> last), NOT from
    # ``k``: a mid-period start covers less z, which would inflate the loop count for no reason.
    stride_dz = abs(float(ts.root_pos[0, 2] - ts.root_pos[-1, 2]))
    ptreads = max(1, round(stride_dz / float(tile.step_height)))
    # Loops must cover the tile's treads with margin, so ROUND UP and add one period.  Rounding to nearest
    # leaves no margin whenever N/ptreads lands on .5 (N=5 with the usual 2-treads-per-stride: 2 loops = 4
    # treads, which only just reaches the 5th from the first) -- and the traversal starts at frame ``k``,
    # PARTWAY through the first period, so that period contributes less than its full ptreads and the
    # traversal ends a tread short.  The far-tread contact gate then never fires and the clip is lost.
    # Over-covering is free: the pure path truncates at the far tread, and the transition path is cut at its
    # peel-off / arrival frame, so the extra phantom treads are discarded either way.
    n = int(tile.stair_steps[0])
    n_loops = max(1, -(-n // ptreads) + 1)                  # ceil(n / ptreads) + 1 period of margin
    period_offset = periodic_tread_offset(ts.root_pos, "periodic", src_boxes)
    rp, rq, jt, terr_contact = loop_periodic(
        ts.root_pos.to(dev), ts.root_quat.to(dev), ts.joint_pos.to(dev),
        None if ts.contact is None else ts.contact.to(dev), n_loops, offset=period_offset, start=k)
    Wp, Wq, Wj, _, _ = align_and_concat([(rp, rq, jt)])
    Wp, Wq, _ = yaw_normalize(Wp, Wq, [], Wq[0])            # traversal heading -> +x (R = -facing)
    src_steps = [src_boxes[i] for i in _tread_boxes(src_boxes)]
    rk = ts.root_pos[k]
    src_start = min(src_steps, key=lambda b: (float(b["pos"][0]) - float(rk[0])) ** 2
                                             + (float(b["pos"][1]) - float(rk[1])) ** 2)
    sst = [float(src_start["pos"][0]), float(src_start["pos"][1]),
           float(src_start["pos"][2]) + float(src_start["size"][2]) / 2.0]      # source start-tread TOP
    qw, qx, qy, qz = (float(v) for v in ts.root_quat[k])
    fy = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))  # source facing yaw
    c, s = math.cos(-fy), math.sin(-fy)                                          # R = rot(-facing)
    rst = [c * sst[0] - s * sst[1], s * sst[0] + c * sst[1], sst[2]]             # R(src_start_top)
    # The tile's START tread: a descent starts on the HIGHEST tread (top of the stairs), an ascent on the
    # LOWEST (bottom).  Restricted to the narrow treads so a flush-but-wide platform can never win the tie.
    tread_ids = _tread_boxes(boxes)
    _top_z = lambda i: float(boxes[i]["pos"][2]) + float(boxes[i]["size"][2]) / 2.0
    first = (min if going_up else max)(tread_ids, key=_top_z)
    t0 = [float(boxes[first]["pos"][0]), float(boxes[first]["pos"][1]), _top_z(first)]
    sh = torch.tensor([t0[j] - rst[j] for j in range(3)], device=Wp.device, dtype=Wp.dtype)
    Wp = Wp + sh

    if truncate:
        # Stop ``end_buffer_treads`` treads BEFORE the far end so the clip always leaves that many stairs
        # ahead of the robot (a landing buffer, never ending on the last tread at a cliff edge).
        N = int(tile.stair_steps[0])
        risers = max(1, (N - 1) - int(end_buffer_treads))   # treads traversed before stopping (>=1)
        tol = 0.02 * float(tile.step_height)
        if going_up:
            z_target = float(Wp[0, 2]) + risers * float(tile.step_height)
            reached = (Wp[:, 2] >= z_target - tol).nonzero(as_tuple=False)
        else:
            z_target = float(Wp[0, 2]) - risers * float(tile.step_height)
            reached = (Wp[:, 2] <= z_target + tol).nonzero(as_tuple=False)
        if reached.numel() > 0:
            cut = int(reached[0].item()) + 1
            Wp, Wq, Wj = Wp[:cut], Wq[:cut], Wj[:cut]
            if terr_contact is not None:
                terr_contact = terr_contact[:cut]
    return Wp, Wq, Wj, terr_contact


def episodic_box_extents(ts, align: str = "facing") -> list[float]:
    """Full extents ``[sx, sy, sz]`` of an episodic source's solved box, in the frame it will be PLACED in.

    The tile box has to be authored before the interaction can be placed on it, so this answers "how big
    is the box along the robot's travel direction?" without running the placement.  With
    ``align="facing"`` (what :func:`place_episodic_on_tile` uses for the Motion-Bricks path) the box is
    rotated with the motion onto ``+x``, so ``sx`` is its depth ALONG TRAVEL and ``sy`` its lateral width
    -- which for every jump reference in the library is the SWAP of the raw solved extents, because the
    robot walks along the source's ``y``.  The rotated footprint is returned as its axis-aligned bounding
    box, since tile boxes are axis-aligned (``yaw = 0``); for the near-90-degree rotations in the library
    that inflates the extents by well under a centimetre.

    ``align="box"`` reproduces the raw solved extents (the frame the DP path places in).

    Args:
        ts: ingested terrain :class:`~.motion_db.MotionSource` (episodic ``jump_on`` / ``jump_off*``).
        align: ``"facing"`` (travel frame) or ``"box"`` (the source's own frame).

    Returns:
        ``[sx, sy, sz]`` full extents (m).
    """
    src = dict(ts.terrain_meta.get("terrain", {}).get("boxes", [{}])[0])
    sx, sy, sz = (float(v) for v in src["size"])
    if align == "box":
        return [sx, sy, sz]
    qw, qx, qy, qz = (float(v) for v in ts.root_quat[0])
    fy = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))   # source facing yaw
    th = float(src.get("yaw", 0.0)) - fy                    # the box's yaw after yaw_normalize("facing")
    c, s = abs(math.cos(th)), abs(math.sin(th))
    return [c * sx + s * sy, s * sx + c * sy, sz]           # AABB of the rotated footprint


def place_episodic_on_tile(ts, tile_box, dev, k: int = 0, align: str = "box"):
    """Play an episodic box interaction from frame ``k`` and anchor its solved box onto the tile box.

    The box-family mirror of :func:`place_periodic_on_tile`: play once (no looping),
    :func:`.box_clip.yaw_normalize` to put the interaction in the tile frame, then translate so the source
    box CENTER coincides with the tile's fixed box center -- so every clip that shares the tile interacts
    with the SAME box, and a mount lands on / a dismount drops off it exactly as it was solved.

    ``align`` picks WHICH rotation normalizes the clip:

    * ``"facing"`` -- cancel the robot's heading, so the interaction TRAVELS along ``+x`` exactly as a
      stair traversal does, and the solved box rotates with it.  The tile must then be authored from
      :func:`episodic_box_extents` (the travel-frame extents), not from the raw solved ``size``.
    * ``"box"`` -- cancel the solved box's own yaw.  For a box that yaw is always 0, so this is a no-op
      that leaves the clip on its source heading (~-91 deg for every jump reference in the library):
      the interaction then runs along world ``y`` while the tile is laid out along world ``x``.  Kept as
      the default for the DP generator, which authors its tile boxes in that same unrotated frame.

    NOTE: the tile box must keep the source box's DEPTH (travel direction) and HEIGHT; only the lateral
    width may be widened (see ``generate_box_clips._box_tile_worker``).  A center-align is only correct
    while depth/height match the source -- a deeper box would leave its near footprint under the run-up,
    which the approach then walks through.

    Args:
        ts: ingested terrain :class:`~.motion_db.MotionSource` (episodic ``jump_on`` / ``jump_off*``).
        tile_box: the authored tile box (:func:`.box_clip.box_tile_box` / ``box_tile_boxes``' index 0).
        dev: torch device for the placed tensors.
        k: source frame to start the interaction at (the ENTRY frame).
        align: normalizing rotation, ``"box"`` (default, the DP path) or ``"facing"`` (see above).

    Returns:
        ``(root_pos[T,3], root_quat[T,4], joints[T,29], contact[T,2] | None)`` in the tile frame (tile
        ground at z=0).
    """
    rp, rq, jt, terr_contact = loop_periodic(
        ts.root_pos.to(dev), ts.root_quat.to(dev), ts.joint_pos.to(dev),
        None if ts.contact is None else ts.contact.to(dev), 1, offset=None, start=k)   # episodic: play once
    Wp, Wq, Wj, _, _ = align_and_concat([(rp, rq, jt)])
    src = dict(ts.terrain_meta.get("terrain", {}).get("boxes", [{}])[0])
    Wp, Wq, aligned = yaw_normalize(Wp, Wq, [src], Wq[0], align=align)
    sc = aligned[0]["pos"]                                  # source box CENTER in the aligned frame
    sh = torch.tensor([float(tile_box["pos"][j]) - float(sc[j]) for j in range(3)],
                      device=Wp.device, dtype=Wp.dtype)
    Wp = Wp + sh                                            # land the interaction's box on the fixed tile box
    return Wp, Wq, Wj, terr_contact


def speed_cell_from(candidate_speeds, chosen: float) -> tuple:
    """The midpoint interval around ``chosen`` among ``candidate_speeds`` -- the Voronoi rule itself.

    Split out so the two callers cannot drift: :func:`terrain_speed_cell` applies it after the
    selector's geometry filter, and the PURE stair generator applies it to the candidate list its own
    inline nearest-geometry pick produces. Both are "which speeds would have chosen this same
    reference", and there is one answer.

    Speeds are compared unsigned; the outermost cells are open, which is the point.
    """
    spd = sorted({round(abs(float(v)), 6) for v in candidate_speeds})
    mine = round(abs(float(chosen)), 6)
    if mine not in spd:
        raise ValueError(f"chosen speed {chosen} is not among the candidates {spd}")
    j = spd.index(mine)
    lo = 0.0 if j == 0 else 0.5 * (spd[j - 1] + spd[j])
    hi = float("inf") if j == len(spd) - 1 else 0.5 * (spd[j] + spd[j + 1])
    return (float(lo), float(hi))


def terrain_speed_cell(sources: list, chosen: int, *, riser: float | None = None,
                       tread: float | None = None, box_height: float | None = None) -> tuple:
    """Speeds for which :func:`select_terrain_reference` returns ``chosen`` -- its Voronoi cell.

    A terrain clip's COMMAND is allowed to differ from the motion it plays, and on the traversal it
    must: the replay is one fixed solved motion, so a clip that only ever commands that motion's own
    speed teaches a generator nothing about SELECTING. What it may not do is command a speed that
    points at a DIFFERENT reference -- that would teach the opposite of the intended lesson ("you
    asked for something nearer the other motion, and we gave you this one").

    So the rule is CONSISTENCY, not agreement, and this is the set it is consistent with. Derived from
    the selector's own two stages rather than restated: filter to the same nearest-geometry candidates
    (their speeds are the only ones reachable on this tile's geometry), then take the midpoints to the
    neighbouring candidate speeds. The outermost cells are open, which is the point -- on a stair tile
    whose solved speeds are {0.31, 0.69} a clip playing 0.69 may legitimately be commanded 2.6, because
    2.6 is still nearer 0.69 than 0.31.

    The cell SHRINKS as the solved library gains speeds, which is the right direction: more motions to
    select among means finer selection, and the freedom to disagree narrows exactly as the reason for
    it does.

    Args:
        sources: the same ingested list passed to :func:`select_terrain_reference`.
        chosen: the index it returned.
        riser / tread / box_height: the same geometry it was called with.

    Returns:
        ``(lo, hi)`` in m/s, unsigned. ``lo`` is 0.0 and/or ``hi`` is ``inf`` at the extremes.
    """
    if not sources:
        raise ValueError("no terrain references to select from")

    def _geom_dist(s) -> float:
        tm = s.terrain_meta.get("terrain", {})
        d = 0.0
        if riser is not None:
            d += abs(float(tm.get("riser", 0.0)) - riser)
        if tread is not None:
            d += abs(float(tm.get("tread", 0.0)) - tread)
        if box_height is not None:
            d += abs(float(tm.get("box_height", 0.0)) - box_height)
        return d

    dgeom = [_geom_dist(s) for s in sources]
    dmin = min(dgeom)
    cand = [i for i, d in enumerate(dgeom) if abs(d - dmin) < 1e-3]
    if chosen not in cand:
        raise ValueError(f"reference {chosen} is not among this geometry's candidates {cand}")
    spd = [abs(float(sources[i].steady_vel[0])) if sources[i].steady_vel is not None else 0.0
           for i in cand]
    mine = abs(float(sources[chosen].steady_vel[0])) if sources[chosen].steady_vel is not None else 0.0
    return speed_cell_from(spd, mine)


def select_terrain_reference(sources: list, *, riser: float | None = None, tread: float | None = None,
                             box_height: float | None = None, speed: float | None = None) -> int:
    """Two-stage terrain-reference pick: nearest GEOMETRY, then nearest steady SPEED within it.

    Feet only sit on the treads (or land on the box) when the played pose matches the geometry it was
    solved for, so geometry dominates: filter to the sources whose ``(riser, tread)`` (stairs) or
    ``box_height`` (box) is nearest the target, then take the one whose steady forward speed is nearest
    ``speed``.  Sources with no recorded command are treated as 0 m/s.

    Args:
        sources: ingested terrain :class:`~.motion_db.MotionSource` list (one skill).
        riser / tread: target stair geometry (m); omit for the box family.
        box_height: target box height (m); omit for the stair family.
        speed: target forward speed (m/s); omitted -> the first nearest-geometry source.

    Returns:
        Index into ``sources``.
    """
    if not sources:
        raise ValueError("no terrain references to select from")

    def _geom_dist(s) -> float:
        tm = s.terrain_meta.get("terrain", {})
        d = 0.0
        if riser is not None:
            d += abs(float(tm.get("riser", 0.0)) - riser)
        if tread is not None:
            d += abs(float(tm.get("tread", 0.0)) - tread)
        if box_height is not None:
            d += abs(float(tm.get("box_height", 0.0)) - box_height)
        return d

    dgeom = [_geom_dist(s) for s in sources]
    dmin = min(dgeom)
    cand = [i for i, d in enumerate(dgeom) if abs(d - dmin) < 1e-3]
    if speed is None:
        return cand[0]
    spd = [abs(float(sources[i].steady_vel[0])) if sources[i].steady_vel is not None else 0.0 for i in cand]
    return cand[min(range(len(cand)), key=lambda j: abs(spd[j] - abs(speed)))]
