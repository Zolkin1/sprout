# Copyright 2026 Zachary Olkin. All rights reserved.

"""Join a Motion-Bricks approach + a replayed terrain traversal + a Motion-Bricks exit into one clip.

The replayed terrain segment is the ANCHOR: it sits on the tile's treads / box exactly as it was solved
and must not move (the RL anchor mask world-pins it).  The two flat Motion-Bricks segments are placed
around it, and there are two ways to do that -- the comparison this module exists to make:

* ``"weld"`` (the hard pin) -- rigid SE(2)+z weld of the boundary frame, ZERO ease.  Exact continuity at the
  seam by construction; only as good as the MB segment's own boundary pose, which is why Stage 1 pins the
  terrain boundary as a target keyframe (approach) / seeds the MB context from it (exit).  A pose mismatch
  shows up as the segment being lifted off (or pushed into) its own walking surface -- reported as
  ``*_surface_err``.
* ``"ease"`` (the soft pin) -- **ground** the MB segment to its walkable surface (feet keep sitting on the
  platform / ground, exactly as z=0 does for a flat clip) and let
  :func:`~motion_gen.playback.inertialize_seam` decay the seam discontinuity over a blend
  window, the same inertialization the motion matcher applies at a cut.

Both keep the terrain segment pristine and yaw+xy align the MB segment onto its boundary frame, so the only
difference is how the boundary residual is absorbed: welded into the placement, or eased out over
``blend_n`` frames.  The seam treatment is INDEPENDENT of how Stage 1 generated the segment, so a pinned
segment can also be eased -- in practice the best of both (an in-betweened / seeded pose leaves little for
the ease to absorb, and the ease keeps the feet on the surface).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..math_utils import quat_apply, quat_mul
from ..terrain.box_clip import align_and_concat, se2z_align  # noqa: F401  (se2z_align re-exported)
from ..playback import apply_seam_offset, inertialize_seam, seam_offset

Segment = tuple[torch.Tensor, torch.Tensor, torch.Tensor]   # (root_pos[T,3], root_quat[T,4], joints[T,29])


def _transform(seg: Segment, gq: torch.Tensor, t_xyz: torch.Tensor) -> Segment:
    """Apply a rigid ``(gq pure-yaw, t_xyz)`` transform to every frame of a segment."""
    P, Q, J = seg
    n = P.shape[0]
    gqe = gq.unsqueeze(0).expand(n, 4)
    return quat_apply(gqe, P) + t_xyz, quat_mul(gqe, Q), J


def align_boundary(seg: Segment, boundary: int, tgt_p: torch.Tensor, tgt_q: torch.Tensor,
                   *, ground_z: float | None) -> tuple[Segment, float]:
    """Yaw+xy align ``seg``'s frame ``boundary`` onto ``(tgt_p, tgt_q)``; set z by weld or by grounding.

    Args:
        seg: the segment to move (a flat MB approach / exit; its own ground is at z ~ 0).
        boundary: index of the frame that must coincide with the target (``-1`` for an approach's last
            frame, ``0`` or the overlap-matched frame for an exit).
        tgt_p / tgt_q: the terrain boundary frame's root position / orientation (tile frame).
        ground_z: ``None`` => WELD (take the target's z exactly, the "hard" pin).  A height => GROUND (add
            that surface to the segment's own feet-at-zero height, the "soft" pin -- the same role z=0
            plays for a flat clip).

    Returns:
        ``(placed_segment, weld_z, gq, t_xyz)`` where ``weld_z`` is the z shift a full weld applies.  Since
        a flat MB segment carries its feet at z ~ 0, ``weld_z - surface_height`` is exactly the
        boundary-pose height residual: how far a hard pin lifts the segment off (or sinks it into) its own
        walking surface, which is what the hard pin trades the seam pop for.  ``(gq, t_xyz)`` is the rigid
        placement actually applied, so a caller can move things that ride ALONG with the segment but are
        not part of it -- the pinned keyframes the viser overlay draws, above all.
    """
    P, Q, _J = seg
    gq, t = se2z_align(P[boundary], Q[boundary], tgt_p, tgt_q)
    weld_z = float(t[2])
    t_xyz = torch.tensor([float(t[0]), float(t[1]), weld_z if ground_z is None else float(ground_z)],
                         device=P.device, dtype=P.dtype)
    return _transform(seg, gq, t_xyz), weld_z, gq, t_xyz


def match_overlap(seg: Segment, tgt_p: torch.Tensor, tgt_q: torch.Tensor, tgt_j: torch.Tensor,
                  search: int = 12) -> int:
    """Index of the frame in ``seg[:search]`` that best reproduces the terrain boundary pose.

    A context-seeded Motion Bricks generation RE-PREDICTS its seed frames first (the pre-filter blends the
    output's opening frames back toward the supplied history) before the genuinely new motion, so the
    output's leading frames replay the terrain's last frames.  Concatenating from frame 0 would replay the
    end of the traversal; this finds where the overlap ends -- the frame that matches the boundary pose in
    root position AND joint configuration (root alone is ambiguous for a near-stationary seed, e.g.
    standing on a box).
    """
    P, _Q, J = seg
    k = max(1, min(int(search), P.shape[0]))
    d = torch.linalg.norm(P[:k] - tgt_p.unsqueeze(0), dim=1) \
        + 0.5 * torch.linalg.norm(J[:k] - tgt_j.unsqueeze(0), dim=1)
    return int(torch.argmin(d))


def match_arrival(seg: Segment, tgt_j: torch.Tensor, z_target: float, search: int = 30,
                  w_z: float = 3.0) -> int:
    """Index of the frame in ``seg``'s LAST ``search`` frames that best reproduces the boundary POSE.

    The approach-side mirror of :func:`match_overlap`.  A Motion Bricks in-betweening does not land on its
    pinned keyframe at an exact frame -- it passes near it -- so the arrival frame is *found* rather than
    assumed to be the last one.  Only the joint configuration and the surface-relative root HEIGHT are
    scored: the segment's xy is free (the splice aligns it), so including it would just penalise how far the
    walk happened to travel.

    Args:
        seg: the approach segment, on its own ground (feet at z ~ 0).
        tgt_j: the terrain entry frame's joint vector.
        z_target: the entry root height RELATIVE to the surface the approach walks on.
        search: how many trailing frames to consider (1 = take the last frame).
        w_z: weight of the height error (m) against the joint error (rad).
    """
    P, _Q, J = seg
    k = max(1, min(int(search), P.shape[0]))
    d = torch.linalg.norm(J[-k:] - tgt_j.unsqueeze(0), dim=1) \
        + w_z * (P[-k:, 2] - float(z_target)).abs()
    return int(P.shape[0] - k + int(torch.argmin(d)))


@dataclass
class Piece:
    """One segment of a spliced clip, plus how it joins its neighbours.

    A clip is an ordered list of these.  Terrain is ``[approach, replay, exit]``; a flat clip is
    ``[stand, transient, loop, transient, loop, ...]``.  Two independent flags, because they answer
    different questions:

    * ``anchored`` -- may this piece be MOVED?  The terrain replay sits on its treads exactly as solved,
      so it defines the output frame and everything else is placed onto it.  A flat clip has no anchored
      piece at all: nothing is world-pinned, so the chain simply starts wherever the first piece is.
    * ``rigid`` -- may this piece be MODIFIED by seam easing?  Loops and replays are exact motion whose
      whole point is that nobody touches them; transients are the give.  A seam eases its non-rigid side.
    """

    seg: Segment
    kind: str = "transient"            # "loop" | "transient" | "replay" | "stand"
    rigid: bool = False                # never modified by easing
    anchored: bool = False             # never moved; defines the output frame
    ground_z: float | None = None      # None welds the boundary z; a value places this piece's own
    #                                    ground (feet at z ~ 0) at that height instead
    trim: tuple[int, int] = (0, 0)     # (lead_skip, trail_cut), applied BEFORE placement
    mask: int = 0                      # per-frame anchor-mask value for build_anchor_mask_t
    stride_idx: int = -1               # loops carry their bank index; everything else -1


def _trimmed(p: Piece) -> Segment:
    lead, trail = int(p.trim[0]), int(p.trim[1])
    end = p.seg[0].shape[0] - trail
    if end - lead < 1:
        raise ValueError(f"piece {p.kind!r}: trim {p.trim} leaves nothing of {p.seg[0].shape[0]} frames")
    return tuple(x[lead:end] for x in p.seg)


def splice_pieces(pieces: list[Piece], *, blend_n: int = 0, dt: float = 0.02):
    """Join an arbitrary list of :class:`Piece` into one world-space clip.

    Generalizes :func:`splice_terrain`'s fixed ``approach + terrain + exit`` to the
    ``[loop, transient, loop, ...]`` shape flat generation needs.  Three stages:

    1. **Trim**, so a piece's boundary frame is the one that will actually meet its neighbour.
    2. **Place.**  Free runs are chained with
       :func:`~motion_gen.terrain.box_clip.align_and_concat` (each piece's first frame rigidly welded onto
       the running last frame, duplicate dropped).  A run adjacent to the anchored piece is then moved as
       a whole onto it -- backwards for the run before, forwards for the run after -- so the anchor never
       moves.  With no anchored piece the whole list is one chain and the first piece sets the frame.
    3. **Ease.**  Every seam's offset is captured from the UNMODIFIED concatenation first, then applied
       over windows clamped so two seams sharing a free piece cannot overlap.  Order-independent by
       construction; easing lazily would let the second seam reopen the first.

    Returns ``(root_pos[T,3], root_quat[T,4], joints[T,29], bounds, seams, diag)`` -- ``bounds[i]`` the
    ``(start, end)`` slice of piece ``i``, ``seams[k]`` a per-seam dict, ``diag`` clip-level totals.
    """
    if not pieces:
        raise ValueError("splice_pieces needs at least one piece")
    anchors = [i for i, p in enumerate(pieces) if p.anchored]
    if len(anchors) > 1:
        raise NotImplementedError(
            f"{len(anchors)} anchored pieces: a run between two fixed pieces is over-constrained (it "
            f"would have to arrive at exactly the right place on its own). Split into one clip per "
            f"anchored interaction, or leave all but one un-anchored.")
    segs = [_trimmed(p) for p in pieces]
    a = anchors[0] if anchors else None

    def _chain(run_idx: list[int]):
        """align_and_concat a run of pieces -> (segment, per-piece bounds within the run)."""
        rp, rq, rj, sb, _ = align_and_concat([segs[i] for i in run_idx])
        return (rp, rq, rj), sb

    placed: list[tuple[Segment, list]] = []          # (segment, per-piece bounds) in output order
    place: dict = {}                                 # placement info for the anchor-adjacent runs
    if a is None:
        placed.append(_chain(list(range(len(pieces)))))
        order = [list(range(len(pieces)))]
        welded = set(range(1, len(pieces)))          # every seam is an align_and_concat weld
    else:
        order = []
        # Seams inside a chained run are position-welded; the two that MEET the anchor are grounded onto
        # it, so those carry a real residual and keep the positional ease.
        welded = {i for i in range(1, len(pieces)) if i != a and i != a + 1}
        if a > 0:                                    # everything before the anchor, moved back onto it
            run = list(range(a))
            seg, sb = _chain(run)
            seg, weld_z, gq, t_xyz = align_boundary(seg, -1, segs[a][0][0], segs[a][1][0],
                                                    ground_z=pieces[a - 1].ground_z)
            # Residual AT the intended boundary, i.e. the frame that was aligned -- measured before it is
            # dropped, so it reports how well the segment reached the pose rather than how far the frame
            # before it sits from the anchor.
            place["before"] = {"weld_z": weld_z, "xform": (gq.tolist(), [float(v) for v in t_xyz]),
                               "joint_res": float(torch.linalg.norm(seg[2][-1] - segs[a][2][0]))}
            seg = tuple(x[:-1] for x in seg)         # drop the frame duplicating the anchor's first
            sb = [(s, min(e, seg[0].shape[0])) for s, e in sb]
            placed.append((seg, sb)); order.append(run)
        placed.append((segs[a], [(0, segs[a][0].shape[0])])); order.append([a])
        if a < len(pieces) - 1:                      # everything after, moved forward onto it
            run = list(range(a + 1, len(pieces)))
            seg, sb = _chain(run)
            seg, weld_z, gq, t_xyz = align_boundary(seg, 0, segs[a][0][-1], segs[a][1][-1],
                                                    ground_z=pieces[a + 1].ground_z)
            place["after"] = {"weld_z": weld_z, "xform": (gq.tolist(), [float(v) for v in t_xyz]),
                              "joint_res": float(torch.linalg.norm(seg[2][0] - segs[a][2][-1]))}
            seg = tuple(x[1:] for x in seg)          # drop the frame duplicating the anchor's last
            sb = [(max(0, s - 1), e - 1) for s, e in sb]
            placed.append((seg, sb)); order.append(run)

    wp = torch.cat([s[0] for s, _ in placed])
    wq = torch.cat([s[1] for s, _ in placed])
    wj = torch.cat([s[2] for s, _ in placed])

    bounds: list[tuple[int, int]] = [(0, 0)] * len(pieces)
    cursor = 0
    for (seg, sb), run in zip(placed, order):
        for i, (s, e) in zip(run, sb):
            bounds[i] = (cursor + s, cursor + e)
        cursor += seg[0].shape[0]

    # --- ease: capture every offset BEFORE touching anything ------------------------------------
    seams: list[dict] = []
    plans = []
    for k in range(len(pieces) - 1):
        before, after = pieces[k], pieces[k + 1]
        idx = bounds[k + 1][0]
        rec = {"at": idx, "between": (before.kind, after.kind)}
        if bounds[k][1] == bounds[k][0] or bounds[k + 1][1] == bounds[k + 1][0]:
            seams.append({**rec, "eased": False, "why": "empty neighbour"}); continue
        rec["joint_res"] = float(torch.linalg.norm(wj[idx] - wj[idx - 1]))
        if before.rigid == after.rigid:
            # Both exact (nothing to give) or both free (no side to preserve): report, do not ease.
            seams.append({**rec, "eased": False,
                          "why": "both rigid" if before.rigid else "neither rigid"})
            continue
        anchored_after = after.rigid
        free = k if anchored_after else k + 1
        plans.append((len(seams), idx, anchored_after, free, (k + 1) in welded))
        seams.append({**rec, "eased": True, "anchored_after": anchored_after,
                      "pos_welded": (k + 1) in welded})

    shared: dict[int, int] = {}
    for _, _, _, free, _ in plans:
        shared[free] = shared.get(free, 0) + 1
    offs = [(sl, idx, aa, free, seam_offset(wp, wq, wj, idx, anchored_after=aa, welded=w))
            for sl, idx, aa, free, w in plans]
    for sl, idx, aa, free, off in offs:
        avail = (bounds[free][1] - bounds[free][0]) // shared[free]
        b = max(0, min(int(blend_n), avail))
        seams[sl]["blend_n_used"] = b
        seams[sl]["clamped"] = b < int(blend_n)
        if b >= 1:
            wp, wq, wj = apply_seam_offset(wp, wq, wj, idx, off, b, anchored_after=aa)
    for s in seams:
        i = s["at"]
        if 0 < i < wp.shape[0]:
            s["seam_speed"] = float(torch.linalg.norm(wp[i] - wp[i - 1]) / dt)

    diag = {"n_pieces": len(pieces), "n_frames": int(wp.shape[0]),
            "anchored": None if a is None else a, "placements": place,
            "seam_speed_max": max([s.get("seam_speed", 0.0) for s in seams], default=0.0),
            "joint_res_max": max([s.get("joint_res", 0.0) for s in seams], default=0.0)}
    return wp, wq, wj, bounds, seams, diag


def splice_terrain(terrain: Segment, approach: Segment | None, exit_seg: Segment | None, *,
                   approach_surf_z: float, exit_surf_z: float, blend_n: int,
                   approach_seam: str = "weld", exit_seam: str = "weld",
                   approach_frame: str = "flat", exit_frame: str = "flat",
                   exit_seeded: bool = False, weld_search: int = 12, arrival_search: int = 1,
                   dt: float = 0.02):
    """Splice ``approach + terrain + exit`` into one world-space clip, per-side seam treatment.

    The terrain segment is never moved.  Each MB segment is aligned onto its boundary frame with
    :func:`align_boundary` (welded for ``"hard"``, grounded for ``"soft"``), the duplicated boundary frame
    is dropped, and a soft seam is eased with the matcher's inertialization.

    Args:
        terrain: the placed replay ``(P, Q, J)``, already cut to its boundary frames.
        approach / exit_seg: the MB segments, or ``None`` to omit that side.
        approach_surf_z / exit_surf_z: walkable surface heights the two MB segments belong on (tile frame).
        blend_n: seam inertialization length (frames); ignored on a welded seam.
        approach_seam / exit_seam: ``"weld"`` (rigid, no ease) or ``"ease"`` (inertialize the residual).
        approach_frame / exit_frame: the segment's own coordinate convention -- ``"flat"`` (generated with
            its walking surface at z ~ 0, so an eased placement LIFTS it onto that surface) or ``"tile"``
            (already in tile coordinates, so it must not be lifted again).  A property of how Stage 1
            generated the segment, independent of the seam treatment -- which is what makes "pinned
            generation + eased seam" expressible.
        exit_seeded: the exit was CONTEXT-SEEDED from the terrain, so its opening frames re-predict the seed
            and must be skipped (see :func:`match_overlap`); a free flat exit starts new motion at frame 0.
        weld_search: how many leading exit frames to search for the context-seed overlap.
        arrival_search: how many trailing approach frames to search for the in-betweened arrival
            (1 = take the last frame as generated).
        dt: control period, only used to report seam speeds (m/s).

    Returns:
        ``(root_pos[T,3], root_quat[T,4], joints[T,29], bounds, diag)`` where ``bounds`` is the three
        ``(start, end)`` slices ``[approach, terrain, exit]`` (empty slices for omitted sides) and ``diag``
        holds the per-seam placement diagnostics.
    """
    for name, val, allowed in (("approach_seam", approach_seam, ("weld", "ease")),
                               ("exit_seam", exit_seam, ("weld", "ease")),
                               ("approach_frame", approach_frame, ("flat", "tile")),
                               ("exit_frame", exit_frame, ("flat", "tile"))):
        if val not in allowed:
            raise ValueError(f"{name} must be one of {allowed}, got {val!r}")
    tP, tQ, tJ = terrain
    dev, dtype = tP.device, tP.dtype
    diag: dict = {"approach_seam": approach_seam, "exit_seam": exit_seam,
                  "approach_frame": approach_frame, "exit_frame": exit_frame}

    def _to_dev(seg):
        return tuple(x.to(device=dev, dtype=dtype) if x.dtype.is_floating_point else x.to(dev)
                     for x in seg)

    def _ground(seam: str, frame: str, surf: float) -> float | None:
        """z handling: ``None`` welds the boundary height; a value is the additive lift for that frame."""
        if seam == "weld":
            return None
        return 0.0 if frame == "tile" else float(surf)

    def _surface_err(seam: str, frame: str, weld_z: float, surf: float) -> float:
        """Where a WELD puts the segment's own ground relative to the surface it belongs on (0 if eased)."""
        if seam != "weld":
            return 0.0
        return weld_z - float(surf) if frame == "flat" else weld_z

    pieces: list[Piece] = []
    # --- approach: its LAST frame meets the terrain's FIRST -------------------------------------
    if approach is not None:
        aseg = _to_dev(approach)
        trail = 0
        if arrival_search > 1:
            z_ref = float(tP[0, 2]) - (0.0 if approach_frame == "tile" else float(approach_surf_z))
            b = match_arrival(aseg, tJ[0], z_ref, search=arrival_search)
            trail = aseg[0].shape[0] - 1 - b          # frames dropped after the in-betweened arrival
            diag["approach_arrival_cut"] = trail
        pieces.append(Piece(seg=aseg, kind="transient", trim=(0, trail),
                            ground_z=_ground(approach_seam, approach_frame, approach_surf_z)))
    pieces.append(Piece(seg=terrain, kind="replay", rigid=True, anchored=True))
    # --- exit: its boundary frame meets the terrain's LAST ---------------------------------------
    if exit_seg is not None:
        eseg = _to_dev(exit_seg)
        j = 0
        if exit_seeded:
            # A context-seeded exit opens by RE-PREDICTING its seed frames, so skip that overlap; a free
            # flat exit starts new motion at frame 0.  The seed is surface-relative, so score the overlap
            # against the boundary in the SAME frame (drop the surface from the terrain's z).
            tgt_p = tP[-1].clone()
            if exit_frame == "flat":
                tgt_p[2] = tgt_p[2] - float(exit_surf_z)
            j = match_overlap(eseg, tgt_p, tQ[-1], tJ[-1], search=weld_search)
        diag["exit_overlap_skip"] = j
        pieces.append(Piece(seg=eseg, kind="transient", trim=(j, 0),
                            ground_z=_ground(exit_seam, exit_frame, exit_surf_z)))

    eased = int(blend_n) if (approach_seam == "ease" or exit_seam == "ease") else 0
    Wp, Wq, Wj, pb, _seams, pdiag = splice_pieces(pieces, blend_n=eased, dt=dt)

    # Re-express the generic result in this function's fixed [approach, terrain, exit] shape.
    n_a = 0 if approach is None else pb[0][1] - pb[0][0]
    n_t = tP.shape[0]
    n_e = 0 if exit_seg is None else pb[-1][1] - pb[-1][0]
    bounds = [(0, n_a), (n_a, n_a + n_t), (n_a + n_t, n_a + n_t + n_e)]
    place = pdiag["placements"]
    if approach is not None:
        b4 = place["before"]
        diag["approach_xform"] = b4["xform"]
        diag["approach_surface_err"] = _surface_err(approach_seam, approach_frame, b4["weld_z"],
                                                    approach_surf_z)
        diag["approach_joint_res"] = b4["joint_res"]
        diag["approach_seam_speed"] = float(torch.linalg.norm(Wp[n_a] - Wp[n_a - 1]) / dt)
    if exit_seg is not None:
        af = place["after"]
        diag["exit_xform"] = af["xform"]
        diag["exit_surface_err"] = _surface_err(exit_seam, exit_frame, af["weld_z"], exit_surf_z)
        diag["exit_joint_res"] = af["joint_res"]
        diag["exit_seam_speed"] = float(torch.linalg.norm(Wp[n_a + n_t] - Wp[n_a + n_t - 1]) / dt)
    return Wp, Wq, Wj, bounds, diag


def footprint_recenter(Wp: torch.Tensor, boxes: list[dict], margin: float):
    """Shift a clip so the tile's FIXED geometry sits at a clip-independent tile-local origin.

    The datum is the boxes' xy bounding box minus ``margin`` (never the motion), so every clip authored on
    the same tile emits the same geometry.  Returns ``(Wp_shifted, corner, size_x, size_y)`` where the tile
    footprint spans the union of the geometry and the motion, plus ``margin`` on all four sides.
    """
    bxy = torch.tensor([[float(b["pos"][0]), float(b["pos"][1])] for b in boxes])
    bhf = torch.tensor([[float(b["size"][0]) / 2.0, float(b["size"][1]) / 2.0] for b in boxes])
    box_lo, box_hi = (bxy - bhf).amin(0), (bxy + bhf).amax(0)
    lo = torch.minimum(Wp[:, :2].amin(0).cpu(), box_lo)
    hi = torch.maximum(Wp[:, :2].amax(0).cpu(), box_hi)
    corner = (lo - margin)
    out = Wp.clone()
    out[:, 0] -= float(corner[0]); out[:, 1] -= float(corner[1])
    return out, (float(corner[0]), float(corner[1])), \
        float(hi[0] - lo[0]) + 2.0 * margin, float(hi[1] - lo[1]) + 2.0 * margin


#: Optional support-structure keys an authored box may carry (``stair_tile_boxes(float_thickness=...)``,
#: ``resolve_box_variant``).  Copied VERBATIM into the manifest overlay so the RL importer and the viewers
#: rebuild the same sub-boxes; absent keys leave the loader on its "solid" defaults, which keeps an
#: undecorated tile byte-identical to prior libraries.
_VARIANT_KEYS = ("variant", "slab_thickness", "leg_size", "leg_inset")


def overlays_for(boxes: list[dict], corner: tuple[float, float]) -> list[dict]:
    """Terrain-manifest overlay dicts for ``boxes``, shifted into the tile-local frame of ``corner``."""
    return [{"kind": "box",
             "pos_xy": [float(b["pos"][0]) - corner[0], float(b["pos"][1]) - corner[1]],
             "size_xyz": [float(v) for v in b["size"]],
             "base_z": float(b["pos"][2]) - float(b["size"][2]) / 2.0,
             "walkable_top": True, "yaw": float(b.get("yaw", 0.0)),
             **{k: b[k] for k in _VARIANT_KEYS if k in b}} for b in boxes]
