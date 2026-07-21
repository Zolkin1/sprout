# Copyright 2026 Zachary Olkin. All rights reserved.

"""Splice a box / terrain traversal clip: ``DP(approach) + PLAY(terrain) + DP(exit)``.

The terrain interaction (a box jump, a stair stride) is PLAYED, not matched -- the DP only
fills the free-locomotion approach and exit around it (see :mod:`.dp`).  This module joins
the three world-space segments into one continuous ``(root_pos, root_quat, joints)`` clip and
carries the terrain geometry + per-frame anchor mask along:

* :func:`align_and_concat` -- rigid SE(2)+z align each segment's first frame onto the previous
  segment's last frame (same transform :meth:`MotionMatchClipGenerator._align` uses) and drop
  the duplicated seam frame, so the join is C0-continuous and the played terrain segment (and
  its geometry) is yaw-normalized to the approach heading.
* :func:`transform_boxes` -- apply that same yaw+translation to the exported terrain geometry so
  the box / treads stay glued to the feet (geometry rides with the motion).
* :func:`build_anchor_mask_t` -- stamp the per-frame ``[T]`` anchor mask by segment (jump-on
  world-pinned, approach / on-top-walk / recovery ego).

All functions are pure (torch + math); the CLI that produces the DP segments and writes the
library composes them (see ``scripts/rsl_rl/generate_box_clips.py``).
"""

from __future__ import annotations

import math

import torch

from ..math_utils import quat_apply, quat_inv, quat_mul, yaw_quat
from .stair_geom import stair_tread_layout


def _yaw_of(q: torch.Tensor) -> float:
    """Yaw angle (rad) of a wxyz quaternion (pure-yaw here)."""
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def se2z_align(src_root: torch.Tensor, src_quat: torch.Tensor,
               tgt_pos: torch.Tensor, tgt_quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rigid SE(2)+z ``(gq pure-yaw, t)`` mapping ``(src_root, src_quat) -> (tgt_pos, tgt_quat)``.

    A source frame ``P`` maps to ``quat_apply(gq, P) + t`` and its quat to ``quat_mul(gq, Q)``.
    Mirrors :meth:`MotionMatchClipGenerator._align`.
    """
    gq = quat_mul(yaw_quat(tgt_quat.unsqueeze(0)), quat_inv(yaw_quat(src_quat.unsqueeze(0))))[0]
    t = tgt_pos - quat_apply(gq.unsqueeze(0), src_root.unsqueeze(0))[0]
    return gq, t


def align_and_concat(
    segments: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, int]], list[tuple[torch.Tensor, torch.Tensor]]]:
    """Align + concatenate world-space segments into one continuous clip.

    Args:
        segments: list of ``(root_pos[Ti,3], root_quat[Ti,4] wxyz, joints[Ti,21])``.

    Each segment after the first is rigid-SE(2)+z aligned so its first frame coincides with the
    running last frame, then its duplicate first frame is dropped (clean C0 join).

    Returns ``(root_pos[T,3], root_quat[T,4], joints[T,21], seg_bounds, transforms)`` where
    ``seg_bounds[i] = (start, end)`` is segment ``i``'s slice in the concatenated timeline and
    ``transforms[i] = (gq, t)`` is the applied SE(2)+z (identity for segment 0).
    """
    if not segments:
        raise ValueError("align_and_concat needs at least one segment.")
    out_p: list[torch.Tensor] = []
    out_q: list[torch.Tensor] = []
    out_j: list[torch.Tensor] = []
    bounds: list[tuple[int, int]] = []
    transforms: list[tuple[torch.Tensor, torch.Tensor]] = []
    cursor = 0
    ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device=segments[0][0].device)
    for i, (rp, rq, jt) in enumerate(segments):
        if i == 0:
            gq, t = ident, torch.zeros(3, device=rp.device)
            arp, arq, ajt = rp, rq, jt
        else:
            gq, t = se2z_align(rp[0], rq[0], out_p[-1][-1], out_q[-1][-1])
            n = rp.shape[0]
            arp = quat_apply(gq.unsqueeze(0).expand(n, 4), rp) + t
            arq = quat_mul(gq.unsqueeze(0).expand(n, 4), rq)
            ajt = jt
            arp, arq, ajt = arp[1:], arq[1:], ajt[1:]        # drop the duplicated seam frame
        out_p.append(arp); out_q.append(arq); out_j.append(ajt)
        bounds.append((cursor, cursor + arp.shape[0]))
        cursor += arp.shape[0]
        transforms.append((gq, t))
    return torch.cat(out_p), torch.cat(out_q), torch.cat(out_j), bounds, transforms


def n_periodic_loops(root_pos: torch.Tensor, wrap: str, boxes: list[dict]) -> int:
    """How many times to loop a PERIODIC terrain stride to traverse its geometry (episodic -> 1).

    A periodic stride (e.g. one stair descent cycle) covers only part of a multi-step staircase; loop
    it until the geometry's vertical span is covered.  ``N = round(total tread z-span / stride net
    z-travel)`` (>=1).  Geometry-driven, so a taller staircase gets more loops with no per-terrain code.
    """
    if wrap != "periodic" or not boxes:
        return 1
    stride_dz = abs(float(root_pos[0, 2] - root_pos[-1, 2]))
    if stride_dz < 1e-3:
        return 1
    zs = [float(b["pos"][2]) for b in boxes]
    return max(1, round((max(zs) - min(zs)) / stride_dz))


def periodic_tread_offset(root_pos: torch.Tensor, wrap: str, boxes: list[dict]):
    """Per-period translation that lands each loop on an EXACT tread (``None`` if not applicable).

    From the uniform stair geometry: the mean tread->tread step vector (in descent order) times the
    number of treads one period covers (``round(period z-travel / riser)``).  Feeding this to
    :func:`loop_periodic` places every period exactly on a tread instead of drifting by the stride's
    own slightly-off net travel.  Wide top/bottom PLATFORMS (size_x >> the step depth) are excluded so
    the step vector is a true tread-to-tread rise.
    """
    if wrap != "periodic" or not boxes:
        return None
    min_sx = min(float(b["size"][0]) for b in boxes)
    steps = sorted((b for b in boxes if float(b["size"][0]) <= 1.5 * min_sx),
                   key=lambda b: -float(b["pos"][2]))               # high -> low (descent order)
    if len(steps) < 2:
        return None
    pos = torch.tensor([[float(v) for v in b["pos"]] for b in steps])
    step_vec = (pos[1:] - pos[:-1]).mean(0)                         # per-tread step, down the descent
    riser = abs(float(step_vec[2]))
    if riser < 1e-6:
        return None
    ptreads = max(1, round(abs(float(root_pos[0, 2] - root_pos[-1, 2])) / riser))
    return (ptreads * step_vec).tolist()


def stair_tile_boxes(
    *,
    step_height: float,
    step_depth: float,
    num_steps: int,
    tread_width: float,
    ttype: str = "terrain",
    top_platform_len: float = 0.0,
    bottom_platform_len: float = 0.0,
    nav_width: float | None = None,
) -> list[dict]:
    """FIXED tile geometry for a uniform stair DESCENT (tile-local, axis-aligned, +x = travel).

    Authored ONCE per tile and shared by all of its clips (no per-clip transform) -- this is what
    makes many clips reuse one terrain.  ``+x`` runs downhill: ``[top platform (high)] -> [treads
    descending] -> [bottom platform (floor, z=0)]``.  The highest tread top sits at ``N*h`` (flush
    with the top platform); the lowest at ``h`` (one riser above the z=0 floor, where the bottom
    platform / recovery sits) -- mirroring the v8 leveling conventions so no post-hoc leveling is
    needed.  Tread x-spacing comes from :func:`stair_geom.stair_tread_layout` (the shared layout SoT).

    Boxes are ``{pos:[x,y,z], size:[sx,sy,sz], yaw:0}``.  Steps are NARROW (``size_x == d``); the
    platforms are WIDE, so the downstream pipeline tells them apart by width.  ``ttype == "terrain"``
    (pure stairs) emits the treads only -- no platforms.

    Args:
        step_height / step_depth: Uniform riser / tread depth (m).
        num_steps: Number of treads ``N``.
        tread_width: Tread y-width (m).
        ttype: ``"terrain"`` (treads only) or ``"transition"`` (add top + bottom platforms).
        top_platform_len / bottom_platform_len: Platform x-lengths (m); typically sized to the
            longest approach / recovery that will run on the tile (transition only).
        nav_width: Platform y-width (m); defaults to ``tread_width``.
    """
    h, d, N = float(step_height), float(step_depth), int(num_steps)
    w = float(tread_width)
    nav = float(nav_width) if nav_width else w
    top_len = float(top_platform_len) if ttype != "terrain" else 0.0
    # start_z_zero => lowest tread top at 0, highest at (N-1)h; shift +h so the floor (bottom
    # platform / recovery) is at z=0 one riser below the lowest tread.
    lay = stair_tread_layout(size_x=N * d, size_y=w, step_height=h, step_depth_nominal=d,
                             going_up=False, x_offset=top_len, y_offset=-w / 2.0, start_z_zero=True)
    boxes: list[dict] = []
    for c in lay.tread_top_centers:
        cx, top = float(c[0]), float(c[2]) + h                # solid tread box from z=0 up to its top
        boxes.append({"pos": [cx, 0.0, top / 2.0], "size": [d, w, top], "yaw": 0.0})
    if ttype != "terrain":
        top_top = N * h                                       # top platform flush with the highest tread
        boxes.insert(0, {"pos": [top_len / 2.0, 0.0, top_top / 2.0],
                         "size": [max(top_len, 1e-3), nav, top_top], "yaw": 0.0})
        bx0 = top_len + N * d
        bot_len = max(float(bottom_platform_len), 1e-3)
        # Bottom platform flush with the LOWEST tread (top z=h), symmetric with the top platform: the
        # recovery walks off the last step onto it continuously (no float).  (A floor-level landing one
        # riser lower would need the descent to include a final step-down -- a later refinement.)
        boxes.append({"pos": [bx0 + bot_len / 2.0, 0.0, h / 2.0],
                      "size": [bot_len, nav, h], "yaw": 0.0})
    return boxes


def loop_periodic(rp: torch.Tensor, rq: torch.Tensor, jt: torch.Tensor,
                  contact: torch.Tensor | None, n_loops: int, offset=None, start: int = 0):
    """Play the FULL periodic stride from frame ``start``, then wrap it forward ``n_loops-1`` more times.

    ``rp/rq/jt`` are the whole stride (``n`` frames).  The first (partial) period runs frames
    ``[start, n)`` -- so the approach can hand off at ANY ``start`` in the period, not just frame 0 --
    and each following FULL period ``[0, n)`` wraps on (its duplicate seam frame dropped, C0).

    Two placements for the wrapped periods:

    * ``offset=None`` -- rigid-aligned onto the previous period's last frame (``align_and_concat``),
      continuing by the stride's OWN net travel.  Simple, but tiny per-stride error accumulates.
    * ``offset`` (a 3-vector) -- each period translated by an EXACT ``i * offset`` (orientation +
      joints repeat unchanged): an integer-tread offset from the uniform stair geometry, so every
      period lands exactly on a tread -- the per-period re-projection that stops drift accumulating.

    ``n_loops <= 1`` is a single play from ``start`` (an episodic box jump from its matched entry).
    Returns ``(rp, rq, jt, contact)``.
    """
    if n_loops <= 1:
        s = slice(start, None)
        return rp[s], rq[s], jt[s], (None if contact is None else contact[s])
    if offset is None:
        segs = [(rp[start:], rq[start:], jt[start:])] + [(rp, rq, jt)] * (n_loops - 1)
        lrp, lrq, ljt, _, _ = align_and_concat(segs)
    else:
        off = torch.as_tensor(offset, device=rp.device, dtype=rp.dtype)
        lrp = torch.cat([rp[start:]] + [rp[1:] + i * off for i in range(1, n_loops)])
        lrq = torch.cat([rq[start:]] + [rq[1:]] * (n_loops - 1))    # pure translation between periods
        ljt = torch.cat([jt[start:]] + [jt[1:]] * (n_loops - 1))
    lc = None if contact is None else torch.cat([contact[start:]] + [contact[1:]] * (n_loops - 1))
    return lrp, lrq, ljt, lc


def _quat_yaw_wxyz(theta: float) -> torch.Tensor:
    """Pure-yaw wxyz quaternion for angle ``theta`` about +z."""
    return torch.tensor([math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0)])


def yaw_normalize(root_pos: torch.Tensor, root_quat: torch.Tensor, boxes: list[dict],
                  ref_gq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Rotate the whole clip + its boxes about the world origin so the TERRAIN is axis-aligned.

    Since the DP builds backward from the (globally placed) terrain interaction, the terrain
    GEOMETRY sets the frame: rotate so the box / tread yaw -> 0, and the rigidly-attached played
    motion (and the DP approach spliced onto it) rotate with it.  This lands the treads axis-aligned
    (width across world-y, depth/descent along world-x) so the descent falls on the anchor mask's
    PINNED x-axis and the axis-aligned ``OverlayVolumeCfg`` / viewer render them correctly.

    Aligning by the box yaw (not ``ref_gq``, the arbitrary approach/splice heading) is what fixes a
    staircase: its treads carry a native world yaw (~-85 deg) that ``ref_gq`` does NOT cancel, which
    left the tread slab rotated ~90 deg (long axis along the descent) and put the descent on the
    y-free anchor axis.  ``ref_gq`` is the fallback heading when the clip carries no boxes.
    """
    ref_yaw = float(boxes[0]["yaw"]) if boxes else _yaw_of(ref_gq)
    gq = _quat_yaw_wxyz(-ref_yaw).to(root_pos.device)
    n = root_pos.shape[0]
    rp = quat_apply(gq.unsqueeze(0).expand(n, 4), root_pos)
    rq = quat_mul(gq.unsqueeze(0).expand(n, 4), root_quat)
    return rp, rq, transform_boxes(boxes, gq, torch.zeros(3, device=root_pos.device))


def transform_boxes(boxes: list[dict], gq: torch.Tensor, t: torch.Tensor) -> list[dict]:
    """Apply an SE(2)+z transform ``(gq pure-yaw, t)`` to a list of ``{pos, size, yaw}`` boxes.

    Rotates each box center about z by the transform's yaw and translates it, and adds the yaw
    to the box orientation -- so the exported terrain geometry stays glued to the (aligned)
    played motion.  Full extents (``size``) are unchanged.
    """
    yaw = _yaw_of(gq)
    c, s = math.cos(yaw), math.sin(yaw)
    tx, ty, tz = float(t[0]), float(t[1]), float(t[2])
    out = []
    for b in boxes:
        px, py, pz = (float(v) for v in b["pos"])
        out.append({
            "pos": [c * px - s * py + tx, s * px + c * py + ty, pz + tz],
            "size": [float(v) for v in b["size"]],
            "yaw": float(b.get("yaw", 0.0)) + yaw,
        })
    return out


def build_anchor_mask_t(T: int, seg_bounds: list[tuple[int, int]], seg_masks: list[int],
                        device: str | torch.device = "cpu") -> torch.Tensor:
    """Per-frame anchor mask ``[T]`` uint8: fill each segment's slice with its mask.

    ``seg_masks[i]`` applies to ``seg_bounds[i]`` (e.g. FLAT for approach / on-top-walk /
    recovery, BOX for the jump-on interior).
    """
    if len(seg_bounds) != len(seg_masks):
        raise ValueError("seg_bounds and seg_masks must have the same length.")
    m = torch.zeros(T, dtype=torch.uint8, device=device)
    for (a, b), mask in zip(seg_bounds, seg_masks):
        m[a:b] = int(mask)
    return m
