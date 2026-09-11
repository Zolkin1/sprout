# Copyright 2026 Zachary Olkin. All rights reserved.

"""Exact signed distance to a union of axis-aligned boxes, in batched torch.

The terrain this repo trains on IS a union of axis-aligned boxes and nothing else: every
manifest tile is a ground slab plus a list of overlay volumes, and a non-zero overlay ``yaw``
is hard-refused by the Isaac loader, by the MuJoCo emitter and by
:func:`~motion_gen.terrain.overlay_boxes.collider_boxes_from_manifest_tile`.  So the signed
distance is a closed form rather than an approximation, and it needs neither a raycast nor a
simulator -- which is what lets one definition serve the RL reward, an offline inspector and a
unit test.

This fills a seam that was deliberately left open.  ``flow_matching/terrain.py`` was deleted
in ``27723572`` because its floor was a single scalar ``ground_z``, "which is wrong on the
stairs/boxes the clip library actually trains on"; its own docstring named the replacement --
*"a future signed-distance-field / heightfield terrain drops in behind the same interface
(return ``(-sdf(world_xyz)).clamp_min(0)``)"*.  :meth:`BoxUnionTerrain.penetration` is that
interface, and :meth:`BoxUnionTerrain.sdf` is that SDF.

**What ``min`` over boxes is exact about.**  Outside the union it is the true distance to the
surface.  Inside, it is the distance to the nearest face of the single most-penetrated box,
which UNDER-reads the true penetration depth near an internal seam where two boxes abut (a
point just inside one tread near its shared face with the tread below reads a small depth,
where the union's own boundary is further away).  That regime starts at half a tread and this
reward never reaches it; the bound is stated rather than hidden.
"""

from __future__ import annotations

import numpy as np
import torch


def box_union_sdf(
    points: torch.Tensor,
    boxes: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Signed distance from each point to the union of axis-aligned boxes.

    For one box with center ``c`` and half-extents ``h``, with ``q = |p - c| - h``::

        sd = ||max(q, 0)|| + min(max(q.x, q.y, q.z), 0)

    which is negative inside, zero on the surface and positive outside.  The union takes the
    ``min`` over boxes (see the module docstring for what that is exact about).

    Args:
        points: ``(N, P, 3)`` query points, in the SAME frame as ``boxes``.
        boxes: ``(N, B, 6)`` as ``(cx, cy, cz, sx, sy, sz)`` -- centers and **full** extents,
            matching :func:`~motion_gen.terrain.overlay_boxes.overlay_sub_boxes`.
        mask: ``(N, B)`` bool, ``True`` for a real box.  ``None`` means every box is real.
            Padding is what lets per-env tiles with different box counts share one tensor;
            a masked-out box is given ``+inf`` rather than being dropped, so it can never
            win the ``min``.

    Returns:
        ``(N, P)`` signed distances, in metres.

    Raises:
        ValueError: On a shape disagreement, which is the one failure mode that would
            otherwise broadcast into a plausible wrong answer.
    """
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"points must be (N, P, 3); got {tuple(points.shape)}")
    if boxes.ndim != 3 or boxes.shape[-1] != 6:
        raise ValueError(f"boxes must be (N, B, 6); got {tuple(boxes.shape)}")
    if points.shape[0] != boxes.shape[0]:
        raise ValueError(
            f"points and boxes disagree on the batch: {points.shape[0]} vs {boxes.shape[0]}"
        )
    if mask is not None and tuple(mask.shape) != tuple(boxes.shape[:2]):
        raise ValueError(
            f"mask must be (N, B) = {tuple(boxes.shape[:2])}; got {tuple(mask.shape)}"
        )

    center = boxes[:, None, :, :3]                      # (N, 1, B, 3)
    half = boxes[:, None, :, 3:] * 0.5                  # (N, 1, B, 3)

    # Accumulate over the three axes instead of materialising (N, P, B, 3). At 512 envs x
    # 1860 points x 16 boxes that is 61 MB per axis rather than 183 MB in one tensor, for
    # three cheap iterations.
    sq_outside = None
    max_q = None
    for a in range(3):
        q = (points[:, :, None, a] - center[..., a]).abs() - half[..., a]   # (N, P, B)
        pos = q.clamp_min(0.0)
        sq_outside = pos * pos if sq_outside is None else sq_outside + pos * pos
        max_q = q if max_q is None else torch.maximum(max_q, q)

    sd = torch.sqrt(sq_outside) + max_q.clamp_max(0.0)                      # (N, P, B)
    if mask is not None:
        sd = sd.masked_fill(~mask[:, None, :], float("inf"))
    return sd.amin(dim=-1)


def pad_box_sets(
    box_sets: list[np.ndarray] | list[torch.Tensor],
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack per-element box lists of differing length into one padded batch.

    Args:
        box_sets: One ``(B_i, 6)`` array per batch element.
        device: Device for the result.
        dtype: Dtype for the boxes.

    Returns:
        ``(boxes (N, B_max, 6), mask (N, B_max) bool)``.

    Raises:
        ValueError: If ``box_sets`` is empty or any entry is not ``(B, 6)``.
    """
    if not box_sets:
        raise ValueError("box_sets is empty; there is no terrain to build an SDF over.")
    arrays = [torch.as_tensor(np.asarray(b), dtype=dtype) for b in box_sets]
    for i, a in enumerate(arrays):
        if a.ndim != 2 or a.shape[-1] != 6:
            raise ValueError(f"box_sets[{i}] must be (B, 6); got {tuple(a.shape)}")
    n, b_max = len(arrays), max(int(a.shape[0]) for a in arrays)
    boxes = torch.zeros(n, b_max, 6, dtype=dtype)
    mask = torch.zeros(n, b_max, dtype=torch.bool)
    for i, a in enumerate(arrays):
        boxes[i, : a.shape[0]] = a
        mask[i, : a.shape[0]] = True
    return boxes.to(device), mask.to(device)


class BoxUnionTerrain:
    """A batch of box-union terrains, one per batch element (an env, or a rollout window).

    Holds ``(N, B, 6)`` boxes plus a validity mask in a fixed frame -- WORLD frame for the RL
    reward, since a plan is placed in the world by its anchor.  Translating a tile's local
    boxes onto its corner is :meth:`translated`.
    """

    def __init__(self, boxes: torch.Tensor, mask: torch.Tensor | None = None):
        """Store the boxes and mask.

        Args:
            boxes: ``(N, B, 6)`` centers + full extents.
            mask: ``(N, B)`` bool, or ``None`` for all-valid.

        Raises:
            ValueError: On a malformed shape.
        """
        if boxes.ndim != 3 or boxes.shape[-1] != 6:
            raise ValueError(f"boxes must be (N, B, 6); got {tuple(boxes.shape)}")
        if mask is not None and tuple(mask.shape) != tuple(boxes.shape[:2]):
            raise ValueError(f"mask must be {tuple(boxes.shape[:2])}; got {tuple(mask.shape)}")
        self.boxes = boxes
        self.mask = mask

    @property
    def device(self) -> torch.device:
        """The device the boxes live on."""
        return self.boxes.device

    def __len__(self) -> int:
        """The batch size N."""
        return int(self.boxes.shape[0])

    def to(self, device: torch.device | str) -> "BoxUnionTerrain":
        """Move to ``device``, returning a new instance (the boxes are read-only)."""
        return BoxUnionTerrain(self.boxes.to(device),
                               None if self.mask is None else self.mask.to(device))

    def index(self, idx: torch.Tensor) -> "BoxUnionTerrain":
        """Gather a sub-batch (or re-order / repeat one) along N.

        This is how a per-TILE box table becomes a per-ENV one: index the tile table by each
        env's ``env_terrain_id`` slot.

        Args:
            idx: ``(M,)`` long indices into the batch axis.

        Returns:
            A new :class:`BoxUnionTerrain` of batch size ``M``.
        """
        return BoxUnionTerrain(self.boxes[idx],
                               None if self.mask is None else self.mask[idx])

    def translated(self, offset: torch.Tensor) -> "BoxUnionTerrain":
        """Translate every box by a per-element offset, leaving extents alone.

        Args:
            offset: ``(N, 3)`` translation (m) -- for a tile, its world corner.

        Returns:
            A new :class:`BoxUnionTerrain` in the translated frame.

        Raises:
            ValueError: If ``offset`` is not ``(N, 3)``.
        """
        if tuple(offset.shape) != (self.boxes.shape[0], 3):
            raise ValueError(
                f"offset must be ({self.boxes.shape[0]}, 3); got {tuple(offset.shape)}")
        moved = self.boxes.clone()
        moved[..., :3] = moved[..., :3] + offset[:, None, :].to(moved.dtype)
        return BoxUnionTerrain(moved, self.mask)

    def sdf(self, points: torch.Tensor) -> torch.Tensor:
        """Signed distance from ``(N, P, 3)`` points to this terrain; ``(N, P)``."""
        return box_union_sdf(points, self.boxes, self.mask)

    def penetration(self, points: torch.Tensor) -> torch.Tensor:
        """Depth below the surface for ``(N, P, 3)`` points, ``>= 0``; ``(N, P)``.

        The interface the deleted ``flow_matching.terrain.FlatTerrain`` defined and named an
        SDF as the way to fill.
        """
        return (-self.sdf(points)).clamp_min(0.0)

    def top_z_at(self, xy: torch.Tensor) -> torch.Tensor:
        """Highest box top face covering each ``(N, P, 2)`` xy, or ``-inf`` where none does.

        The 2.5-D reading of the same boxes, kept beside the SDF because it is what
        ``generate_clip_tile_scene.terrain_top_at`` answers and is therefore the cross-check
        that the two agree.  It is NOT what the reward uses: a box's vertical face has no
        "top", and a plan can drive a shin into one.

        Args:
            xy: ``(N, P, 2)`` query points.

        Returns:
            ``(N, P)`` surface height (m).
        """
        if xy.ndim != 3 or xy.shape[-1] != 2:
            raise ValueError(f"xy must be (N, P, 2); got {tuple(xy.shape)}")
        c = self.boxes[:, None, :, :3]
        h = self.boxes[:, None, :, 3:] * 0.5
        inside = ((xy[:, :, None, 0] - c[..., 0]).abs() <= h[..., 0]) & \
                 ((xy[:, :, None, 1] - c[..., 1]).abs() <= h[..., 1])
        if self.mask is not None:
            inside = inside & self.mask[:, None, :]
        top = c[..., 2] + h[..., 2]
        return top.masked_fill(~inside, float("-inf")).amax(dim=-1)
