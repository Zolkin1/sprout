# Copyright 2026 Zachary Olkin. All rights reserved.

"""A forward-biased ray FAN for terrain sensing, as a ``RayCaster`` pattern.

The stock ``GridPatternCfg`` is a uniform lattice, which spends its samples the same way
everywhere: at 0.1 m resolution over 1.5 x 1.0 m it is 176 rays that never see past about a
metre. For a policy correcting a 1.24 s plan that is the wrong budget -- the far half of the
plan lands on terrain the scan cannot reach, and the near half is sampled far finer than the
0.05 m the feet actually care about.

This pattern instead sweeps a few ANGLES and spends its samples along each one, densely near
the robot and logarithmically out to 3.5 m::

    yaw offsets   -30, -15, 0, +15, +30 degrees        (5 rays, ascending)
    distances     0.05 .. 1.00 m at 0.05 m             (20, the foot-placement band)
                  1.07 .. 3.50 m log-spaced            (20, the approach band)

200 rays covering ~3.5 m forward, against 176 covering ~1 m. The layout is ``(ray, distance)``
and is a CONTRACT: an observation built from it is a flat vector, so a reordering here would
silently permute the critic's input while every width still matched.

Two details that are easy to get wrong:

* **The log leg starts one step PAST 1.0 m.** A naive ``geomspace(1.0, 3.5, 20)`` begins at
  exactly 1.00 m, which the dense leg already sampled -- a duplicated ray dressed as extra
  coverage.
* **Rays start ABOVE the base**, not at it, so a ray whose start would otherwise be inside a
  riser still casts downward onto the tread rather than missing the mesh entirely.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.sensors.ray_caster.patterns import PatternBaseCfg
from isaaclab.utils import configclass


def ray_fan_pattern(cfg: RayFanPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the fan's ray starts and directions.

    Args:
        cfg: The pattern configuration.
        device: Device to build the tensors on.

    Returns:
        ``(ray_starts, ray_directions)``, both ``[num_rays, 3]``, ordered ``(ray, distance)``.

    Raises:
        ValueError: If the distance schedule is empty or non-increasing, which would make the
            layout ambiguous rather than merely odd.
    """
    near = torch.arange(cfg.near_start, cfg.near_stop + 1e-9, cfg.near_step, device=device)
    # One step PAST the dense leg's end: `geomspace(a, b, n)` includes `a`, and `a` is the
    # dense leg's last sample, so taking it again would duplicate a ray.
    far_all = torch.logspace(
        math.log10(cfg.near_stop), math.log10(cfg.far_stop), cfg.far_num + 1, device=device
    )
    distances = torch.cat([near, far_all[1:]])
    if distances.numel() == 0 or bool((distances[1:] <= distances[:-1]).any()):
        raise ValueError(
            f"ray fan distances must be strictly increasing and non-empty, got "
            f"{distances.tolist()}. The observation's layout is (ray, distance), so a "
            "repeated or unordered distance is a silently duplicated channel."
        )

    yaws = torch.tensor([math.radians(d) for d in cfg.yaw_offsets_deg], device=device)
    # (ray, distance) order, flattened. `outer` keeps ray-major.
    x = torch.outer(torch.cos(yaws), distances).reshape(-1)
    y = torch.outer(torch.sin(yaws), distances).reshape(-1)
    z = torch.full_like(x, float(cfg.start_height))

    starts = torch.stack([x, y, z], dim=-1)
    directions = torch.zeros_like(starts)
    directions[:, 2] = -1.0
    return starts, directions


@configclass
class RayFanPatternCfg(PatternBaseCfg):
    """Configuration for :func:`ray_fan_pattern`.

    Defaults give 5 rays x 40 distances = 200 rays reaching 3.5 m. The sensor this feeds must
    use ``ray_alignment="yaw"`` so the fan is expressed in the base's YAW frame: with the full
    orientation the whole fan would pitch with the torso and a lean would read as terrain.
    """

    func = ray_fan_pattern

    yaw_offsets_deg: Sequence[float] = (-30.0, -15.0, 0.0, 15.0, 30.0)
    """Ray azimuths relative to base yaw, ascending. Order is part of the layout contract."""

    near_start: float = 0.05
    """First dense sample, in metres."""

    near_stop: float = 1.0
    """Last dense sample, and where the log leg picks up (exclusive there)."""

    near_step: float = 0.05
    """Dense spacing. 0.05 m is roughly a foot length, the scale of a placement decision."""

    far_num: int = 20
    """Log-spaced samples strictly beyond :attr:`near_stop`."""

    far_stop: float = 3.5
    """Furthest sample. ~3.5 m is about the far camera's usable ground footprint."""

    start_height: float = 1.0
    """Height above the sensor frame to cast from, in metres.

    Not zero: a ray originating at base height inside a stair riser would miss the mesh and
    return an infinity, which is exactly the sample a stair-climbing policy needs most.
    """
