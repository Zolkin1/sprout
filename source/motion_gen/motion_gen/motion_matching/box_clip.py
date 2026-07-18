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


def _quat_yaw_wxyz(theta: float) -> torch.Tensor:
    """Pure-yaw wxyz quaternion for angle ``theta`` about +z."""
    return torch.tensor([math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0)])


def yaw_normalize(root_pos: torch.Tensor, root_quat: torch.Tensor, boxes: list[dict],
                  ref_gq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Rotate the whole clip + its boxes about the world origin by ``-yaw(ref_gq)``.

    ``ref_gq`` is the played terrain segment's alignment quaternion, whose yaw is the arbitrary
    approach heading the traversal ended up at.  Removing it makes the terrain **axis-aligned**
    (box yaw -> ~0), matching the axis-aligned terrain-tile convention (and the axis-aligned
    ``OverlayVolumeCfg`` / viewer) so the box isn't left rotated under the motion.
    """
    gq = _quat_yaw_wxyz(-_yaw_of(ref_gq)).to(root_pos.device)
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
