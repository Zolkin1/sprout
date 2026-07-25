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
  :func:`~motion_gen.motion_matching.playback.inertialize_seam` decay the seam discontinuity over a blend
  window, the same inertialization the motion matcher applies at a cut.

Both keep the terrain segment pristine and yaw+xy align the MB segment onto its boundary frame, so the only
difference is how the boundary residual is absorbed: welded into the placement, or eased out over
``blend_n`` frames.  The seam treatment is INDEPENDENT of how Stage 1 generated the segment, so a pinned
segment can also be eased -- in practice the best of both (an in-betweened / seeded pose leaves little for
the ease to absorb, and the ease keeps the feet on the surface).
"""

from __future__ import annotations

import torch

from ..math_utils import quat_apply, quat_mul
from ..motion_matching.box_clip import se2z_align
from ..motion_matching.playback import inertialize_seam

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
        ``(placed_segment, weld_z)`` where ``weld_z`` is the z shift a full weld applies.  Since a flat MB
        segment carries its feet at z ~ 0, ``weld_z - surface_height`` is exactly the boundary-pose height
        residual: how far a hard pin lifts the segment off (or sinks it into) its own walking surface, which
        is what the hard pin trades the seam pop for.
    """
    P, Q, _J = seg
    gq, t = se2z_align(P[boundary], Q[boundary], tgt_p, tgt_q)
    weld_z = float(t[2])
    t_xyz = torch.tensor([float(t[0]), float(t[1]), weld_z if ground_z is None else float(ground_z)],
                         device=P.device, dtype=P.dtype)
    return _transform(seg, gq, t_xyz), weld_z


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

    # --- approach: its LAST frame meets the terrain's FIRST -------------------------------------
    if approach is not None:
        aseg = tuple(x.to(device=dev, dtype=dtype) if x.dtype.is_floating_point else x.to(dev)
                     for x in approach)
        if arrival_search > 1:
            z_ref = float(tP[0, 2]) - (0.0 if approach_frame == "tile" else float(approach_surf_z))
            b = match_arrival(aseg, tJ[0], z_ref, search=arrival_search)
            diag["approach_arrival_cut"] = aseg[0].shape[0] - 1 - b   # frames dropped after the arrival
            aseg = (aseg[0][:b + 1], aseg[1][:b + 1], aseg[2][:b + 1])
        (aP, aQ, aJ), weld_z = align_boundary(
            aseg, -1, tP[0], tQ[0], ground_z=_ground(approach_seam, approach_frame, approach_surf_z))
        diag["approach_surface_err"] = _surface_err(approach_seam, approach_frame, weld_z, approach_surf_z)
        diag["approach_joint_res"] = float(torch.linalg.norm(aJ[-1] - tJ[0]))
        aP, aQ, aJ = aP[:-1], aQ[:-1], aJ[:-1]                 # drop the duplicated boundary frame
    else:
        aP, aQ, aJ = tP[:0], tQ[:0], tJ[:0]

    # --- exit: its boundary frame meets the terrain's LAST ---------------------------------------
    if exit_seg is not None:
        eseg = tuple(x.to(device=dev, dtype=dtype) if x.dtype.is_floating_point else x.to(dev)
                     for x in exit_seg)
        # A context-seeded exit opens by RE-PREDICTING its seed frames, so skip that overlap; a free flat
        # exit starts new motion at frame 0 and needs no skip.  The seed is surface-relative, so score the
        # overlap against the boundary in the SAME frame (drop the surface from the terrain's z).
        j = 0
        if exit_seeded:
            tgt_p = tP[-1].clone()
            if exit_frame == "flat":
                tgt_p[2] = tgt_p[2] - float(exit_surf_z)
            j = match_overlap(eseg, tgt_p, tQ[-1], tJ[-1], search=weld_search)
        diag["exit_overlap_skip"] = j
        eseg = (eseg[0][j:], eseg[1][j:], eseg[2][j:])
        (eP, eQ, eJ), weld_z = align_boundary(
            eseg, 0, tP[-1], tQ[-1], ground_z=_ground(exit_seam, exit_frame, exit_surf_z))
        diag["exit_surface_err"] = _surface_err(exit_seam, exit_frame, weld_z, exit_surf_z)
        diag["exit_joint_res"] = float(torch.linalg.norm(eJ[0] - tJ[-1]))
        eP, eQ, eJ = eP[1:], eQ[1:], eJ[1:]                    # drop the duplicated boundary frame
    else:
        eP, eQ, eJ = tP[:0], tQ[:0], tJ[:0]

    Wp = torch.cat([aP, tP, eP]); Wq = torch.cat([aQ, tQ, eQ]); Wj = torch.cat([aJ, tJ, eJ])
    na, nt, ne = aP.shape[0], tP.shape[0], eP.shape[0]
    bounds = [(0, na), (na, na + nt), (na + nt, na + nt + ne)]

    # Ease only the EASED seams; the terrain is the anchored side on both.
    if blend_n >= 2:
        if na >= 1 and approach_seam == "ease":
            Wp, Wq, Wj = inertialize_seam(Wp, Wq, Wj, na, blend_n, anchored_after=True)
        if ne >= 1 and exit_seam == "ease":
            Wp, Wq, Wj = inertialize_seam(Wp, Wq, Wj, na + nt, blend_n, anchored_after=False)
    if na >= 1:
        diag["approach_seam_speed"] = float(torch.linalg.norm(Wp[na] - Wp[na - 1]) / dt)
    if ne >= 1:
        diag["exit_seam_speed"] = float(torch.linalg.norm(Wp[na + nt] - Wp[na + nt - 1]) / dt)
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


def overlays_for(boxes: list[dict], corner: tuple[float, float]) -> list[dict]:
    """Terrain-manifest overlay dicts for ``boxes``, shifted into the tile-local frame of ``corner``."""
    return [{"kind": "box",
             "pos_xy": [float(b["pos"][0]) - corner[0], float(b["pos"][1]) - corner[1]],
             "size_xyz": [float(v) for v in b["size"]],
             "base_z": float(b["pos"][2]) - float(b["size"][2]) / 2.0,
             "walkable_top": True, "yaw": float(b.get("yaw", 0.0))} for b in boxes]
