# Copyright 2026 Zachary Olkin. All rights reserved.

"""Dependency-light stair projector for offline clip generation.

The online terrain-aware reference snapping projects a world foot pose onto
the stair tread it rests on via
:meth:`MetaCompositeTerrainImporter._project_world`.  That importer is
sim-bound (it owns the physical terrain and a grid of sub-terrains).  Offline
clip generation builds one *uniform* staircase at a time in its own local
frame, so it needs the same projection math without the grid/tiling and
without importing IsaacLab.

:func:`make_offline_stair_projector` returns a ``project(points) -> points``
callable that is a faithful single-span port of ``_project_world``: the query
z picks the step (tread / entry / exit level) whose top-z is nearest, so a
foot whose ankle xy still sits over the flat below a riser snaps to the step
it is physically on rather than a whole step too low.  Coordinates are
sub-terrain-**local corner** coordinates (the clip-world frame used by the
offline generator), so there is no cell offset.

Kept torch-only (no IsaacLab import) so it is unit-testable in a bare
interpreter and cheap to call inside the batched rollout.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def make_offline_stair_projector(
    stair_top_centers: torch.Tensor,
    step_depth: float,
    entry_z: float,
    span_xmin: float,
    span_xmax: float,
    z_bias_frac: float = 0.5,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a single-span stair projector matching ``_project_world``.

    The tread is chosen by nearest top-z to the query z (as in the importer).
    Offline this is fed the trajectory's stance ankle, which -- because the
    ankle frame sits ~half a step-height ABOVE the tread the foot rests on --
    lands almost exactly on the midpoint between two tread tops and rounds UP to
    the step *ahead*, so the re-anchor teleports the clip forward a tread (the
    online path avoids this by snapping a genuinely in-contact / planted foot).
    To compensate, the query z is biased DOWN by ``z_bias_frac`` of a step
    height before the nearest-tread pick, so it rounds to the tread the foot is
    actually on.  Keep ``z_bias_frac`` in ``(0, 1)`` -- < 1 step so it never
    rounds to a tread *below* the foot (which would sink/stall the climb);
    ~0.5 (≈ the ankle offset) is the robust middle.  The bias affects only the
    tread/entry SELECTION; the returned coordinates are the tread's own values.

    Args:
        stair_top_centers: ``[num_steps, 3]`` tread-top centers (cx, cy, top_z)
            in local-corner coordinates, in ascending-x order — exactly the
            ``extras["stair_top_centers"]`` a :class:`StairBlock` emits.
        step_depth: Tread depth (m); the fictitious entry/exit levels are
            placed half a tread beyond the span edges, as in the importer.
        entry_z: Connecting-ground z (``BlockOutput.entry_z``) — one full step
            below the first tread top for a composite stair block.
        span_xmin: Span (block) lower x edge in local-corner coordinates.
        span_xmax: Span (block) upper x edge in local-corner coordinates.
        z_bias_frac: Fraction of a step height subtracted from the query z
            before the nearest-tread pick (see above).  Default ``0.5``.

    Returns:
        ``project(points[N, 3]) -> [N, 3]`` mapping local-corner foot poses to
        their snapped stair (x, y, z).  Only x and z are meaningful to
        :func:`stair_stance_ref` (it keeps the foot's own y); the y column
        mirrors ``_project_world`` (tread center y on the tread branch, the
        foot's y on the entry/exit branches).
    """
    if stair_top_centers.dim() != 2 or stair_top_centers.shape[-1] != 3:
        raise ValueError(
            "stair_top_centers must be [num_steps, 3]; got "
            f"{tuple(stair_top_centers.shape)}"
        )
    centers = stair_top_centers.detach().clone()
    step_depth = float(step_depth)
    entry_z = float(entry_z)
    span_xmin = float(span_xmin)
    span_xmax = float(span_xmax)
    exit_z = float(centers[-1, 2])
    # Step height = tread-top z spacing (fall back to the first tread's rise
    # above the entry level when there is only one tread).
    step_height = (
        float(centers[1, 2] - centers[0, 2]) if centers.shape[0] >= 2
        else float(centers[0, 2] - entry_z)
    )
    z_bias = float(z_bias_frac) * abs(step_height)

    def project(points: torch.Tensor) -> torch.Tensor:
        """Snap local-corner foot poses ``[N, 3]`` to their stair step."""
        if points.dim() != 2 or points.shape[-1] != 3:
            raise ValueError(
                f"project expects points of shape (N, 3); got {tuple(points.shape)}"
            )
        c = centers.to(device=points.device, dtype=points.dtype)  # [S, 3]
        local_x = points[:, 0]
        local_y = points[:, 1]
        # Bias the query z DOWN so the ankle-frame offset doesn't round the tread
        # pick up to the step ahead (see the function docstring).
        query_z = points[:, 2] - z_bias

        # Among treads, pick the one whose top-z is nearest the query z.
        d_tread = (c[:, 2].unsqueeze(0) - query_z.unsqueeze(1)).abs()  # [N, S]
        best_tread_d, best_tread_k = d_tread.min(dim=1)                # [N]
        tread_sel = c[best_tread_k]                                    # [N, 3]

        # Fictitious entry level (at the first riser).  Keep the foot's own x
        # when it lies inside the span (on a start platform); else snap just
        # before the riser.
        in_span = (local_x >= span_xmin) & (local_x < span_xmax)
        entry_x = torch.where(
            in_span, local_x, torch.full_like(local_x, span_xmin - 0.5 * step_depth)
        )
        entry_sel = torch.stack(
            [entry_x, local_y, torch.full_like(local_x, entry_z)], dim=-1
        )
        use_entry = ((entry_z - query_z).abs() < best_tread_d).unsqueeze(-1)
        z_sel = torch.where(use_entry, entry_sel, tread_sel)

        # Exit-side fictitious level shares the top tread's z, so height can't
        # distinguish it — gate on xy being past the span.
        exit_sel = torch.stack(
            [
                torch.full_like(local_x, span_xmax + 0.5 * step_depth),
                local_y,
                torch.full_like(local_x, exit_z),
            ],
            dim=-1,
        )
        past_exit = (local_x >= span_xmax).unsqueeze(-1)
        return torch.where(past_exit, exit_sel, z_sel)

    return project


def make_offline_multi_stair_projector(
    spans: list[dict],
    z_bias_frac: float = 0.5,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Compose per-band single-span projectors for a multi-staircase corridor.

    Each entry in ``spans`` describes one staircase band (ordered along +x) with
    keys ``stair_top_centers`` ``[ns, 3]``, ``step_depth``, ``entry_z``,
    ``span_xmin``, ``span_xmax``.  A query point is routed to the band it is on
    or *approaching*: the first band whose ``span_xmax`` is greater than the
    point's x (via ``searchsorted``), clamped to the last band.  So a point
    inside band k goes to band k (its treads), a point on the flat *after* band
    k (and before band k+1) goes to band k+1 (its entry — the staircase being
    approached), a point before all bands goes to band 0, and a point past the
    last band goes to that band's exit level.  This "next stair ahead"
    ownership is what the approach slow-down needs (the pre-riser flat foot must
    project to the upcoming staircase's entry, not the previous exit).

    Args:
        spans: Ordered list of per-band span dicts (ascending ``span_xmax``).

    Returns:
        ``project(points[N, 3]) -> [N, 3]`` dispatching each point to its band.
    """
    if not spans:
        raise ValueError("make_offline_multi_stair_projector needs >= 1 span")
    xmaxs = [float(s["span_xmax"]) for s in spans]
    if any(b <= a for a, b in zip(xmaxs, xmaxs[1:])):
        raise ValueError(f"stair spans must be ordered by ascending span_xmax; got {xmaxs}")
    projectors = [
        make_offline_stair_projector(
            s["stair_top_centers"], s["step_depth"], s["entry_z"], s["span_xmin"], s["span_xmax"],
            z_bias_frac=z_bias_frac,
        )
        for s in spans
    ]
    n = len(spans)
    xmax_t = torch.tensor(xmaxs, dtype=torch.float32)

    def project(points: torch.Tensor) -> torch.Tensor:
        if points.dim() != 2 or points.shape[-1] != 3:
            raise ValueError(
                f"project expects points of shape (N, 3); got {tuple(points.shape)}"
            )
        if n == 1:
            return projectors[0](points)
        band = torch.searchsorted(
            xmax_t.to(points.device), points[:, 0].contiguous(), right=True,
        ).clamp_(max=n - 1)
        out = torch.empty_like(points)
        for k in range(n):
            mask = band == k
            if bool(mask.any()):
                out[mask] = projectors[k](points[mask])
        return out

    return project
