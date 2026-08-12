# Copyright 2026 Zachary Olkin. All rights reserved.

"""The height-scan geometry contract, and the single reader for it.

The counterpart of :mod:`.depth` for downward ray-cast grids, with one structural
difference that shapes the whole module: **a height scan is read from the SENSOR, not from
an observation term.**  ``record_data.py`` and ``FmTrackingCommand._read_extero`` both call
``mdp.height_scan`` directly, deliberately, so the reading bypasses the observation
manager's noise and clip pipeline and is the clean raw value on both sides.  Depth is the
opposite -- its noise lives *inside* ``mdp.depth_image`` -- which is why that module
discovers cameras by walking obs terms and this one resolves scanners by name.

Consequences worth stating, because they are what make this module small:

* the generator's dense scanner needs **no observation term and no observation group**, so
  nothing here touches obs-group plumbing, the sagittal-reflection augmentation, or
  ``export_parameters``' policy-group lookup;
* a group's ``enable_corruption=False`` would be moot anyway -- reading the sensor skips
  the manager entirely;
* scoping is therefore by sensor name (:func:`fm_scan_sensor` / :func:`rl_scan_sensor`)
  rather than by obs group.

What this module does NOT carry is the *affine* that maps a reading into the FM's input
units.  That is a property of the model's input representation, not of the sensor, so it
lives in :class:`~robot_rl.flow_matching.config.FlowMatchingConfig` (``extero_clip`` and
the metric scale) and travels in the checkpoint.  The spec carries geometry plus
``sensor_offset`` -- the constant ``mdp.height_scan`` subtracts -- which is everything
needed to say *where each ray sampled the ground*.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import GridPatternCfg

from robot_rl.flow_matching.transforms import scan_grid_shape

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


RL_SCAN_SENSOR: str = "height_scanner"
"""The policy's height scanner, the only one that existed before the generator got its own.

Named rather than discovered because four call sites already hardcode it and the sensor is
a fixture of :class:`~robot_rl.tasks.manager_based.robot_rl.g1.g1_teacher_env_cfg.G1TeacherSceneCfg`.
"""

FM_SCAN_PREFIX: str = "fm_"
"""Prefix marking a sensor as the GENERATOR's own sampling rather than the policy's.

The same convention ``g1_depth_env_cfg`` uses for the FM camera pair, and the thing
:func:`height_scan_specs` derives ``privileged`` from -- a dense noise-free scan is a
diagnostic instrument that no deployable artifact can reproduce.
"""

FM_SCAN_SENSOR: str = f"{FM_SCAN_PREFIX}height_scanner"
"""The generator's dense scanner, added by ``add_fm_height_scan``."""


def _sensor_offset_default() -> float:
    """The ``offset`` ``mdp.height_scan`` subtracts when the caller passes none.

    Read off the function's own signature rather than restated, so an exported contract
    cannot claim a value the reader does not actually use.  Same trick as
    :func:`.depth._normalization_params`.
    """
    from isaaclab.envs.mdp.observations import height_scan as _height_scan

    default = inspect.signature(_height_scan).parameters["offset"].default
    return float(default)


def fm_scan_sensor(env: "ManagerBasedRLEnv") -> str | None:
    """The scanner the GENERATOR should be conditioned on, or ``None`` if there is none.

    The dense FM scanner when the task has one, else the policy's.  That fallback is what
    keeps every call site free of a flag: with no FM scanner present this returns exactly
    what the four hardcoded reads used to name, so a task that never opts in is unchanged.

    Args:
        env: The environment instance.

    Returns:
        A scene sensor name, or ``None`` when the task has no height scanner at all.
    """
    sensors = env.scene.sensors
    if FM_SCAN_SENSOR in sensors:
        return FM_SCAN_SENSOR
    return RL_SCAN_SENSOR if RL_SCAN_SENSOR in sensors else None


def rl_scan_sensor(env: "ManagerBasedRLEnv") -> str | None:
    """The POLICY's scanner, never the generator's.

    For consumers that describe what a *deployed* policy reads -- ``export_parameters``,
    the hardware sensor configs.  Deliberately not :func:`fm_scan_sensor`: emitting the
    generator's privileged grid as a deploy contract would tell the robot to reproduce a
    signal it cannot.
    """
    return RL_SCAN_SENSOR if RL_SCAN_SENSOR in env.scene.sensors else None


def height_scan_specs(
    env: "ManagerBasedRLEnv", sensor_names: list[str] | str | None = None
) -> list[dict]:
    """The geometry contract for one or more height scanners.

    Everything needed to say *where on the ground each ray landed*, derived from the
    ``RayCasterCfg`` and never hand-copied: the grid, the fold, the mount, and the constant
    the reader subtracts.  Recorded as ``RecordingMeta.height_scan_spec`` so a training run
    can check that the array it loaded has the shape its geometry implies -- ``num_rays``
    alone cannot distinguish a grid from its transpose.

    Args:
        env: The environment instance.
        sensor_names: Scanner name(s) to describe.  ``None`` resolves to
            :func:`fm_scan_sensor`, i.e. the one the generator would be conditioned on.

    Returns:
        One dict per scanner, in the order given.  Empty when there is no scanner.

    Raises:
        ValueError: If a scanner is configured in a way the fold or the reading does not
            survive -- a non-grid pattern, a non-downward cast, a non-yaw alignment, a ray
            count that disagrees with the derived grid, or ray drift on a privileged sensor.
    """
    if sensor_names is None:
        resolved = fm_scan_sensor(env)
        names = [resolved] if resolved is not None else []
    elif isinstance(sensor_names, str):
        names = [sensor_names]
    else:
        names = list(sensor_names)

    sensor_offset = _sensor_offset_default()
    specs: list[dict] = []
    for name in names:
        sensor = env.scene.sensors[name]
        cfg = sensor.cfg
        privileged = name.startswith(FM_SCAN_PREFIX)
        pattern = cfg.pattern_cfg
        if not isinstance(pattern, GridPatternCfg):
            raise ValueError(
                f"height scanner '{name}' has pattern_cfg "
                f"{type(pattern).__name__}, not GridPatternCfg; there is no (H, W) fold for "
                "it, so it cannot feed a convolutional encoder."
            )
        direction = tuple(float(v) for v in pattern.direction)
        if direction != (0.0, 0.0, -1.0):
            raise ValueError(
                f"height scanner '{name}' casts along {direction}; the reading "
                "'pos_z - hit_z - offset' only means ground clearance for a straight-down "
                "cast (0, 0, -1)."
            )
        alignment = str(getattr(cfg, "ray_alignment", "yaw"))
        if alignment != "yaw":
            raise ValueError(
                f"height scanner '{name}' has ray_alignment='{alignment}'; the FM's window "
                "frame is xyz + yaw, so only a yaw-aligned grid puts each ray at a FIXED "
                "point in that frame. A 'base' grid is roll/pitch-coupled and its pixels "
                "would move under the robot."
            )
        height, width = scan_grid_shape(pattern.resolution, pattern.size, pattern.ordering)
        num_rays = height * width
        # Cross-check against the LIVE cast when there is one. The cfg is authoritative (it is
        # what an off-sim consumer can reproduce), but a live sensor is the one thing that can
        # catch the fold arithmetic drifting from the library's, so check it whenever a real
        # sensor is present -- absent for the sim-free `*_from_cfg` twin, which has only cfgs.
        live = getattr(getattr(sensor, "data", None), "ray_hits_w", None)
        if live is not None and int(live.shape[1]) != num_rays:
            raise ValueError(
                f"height scanner '{name}': the cfg implies a {height}x{width} grid "
                f"({num_rays} rays) but the sensor cast {int(live.shape[1])}. The fold and the "
                "reading disagree, so any image built from this row would be scrambled."
            )
        drift = tuple(float(v) for v in getattr(cfg, "drift_range", (0.0, 0.0)))
        ray_drift = {
            axis: [float(v) for v in bounds]
            for axis, bounds in (getattr(cfg, "ray_cast_drift_range", None) or {}).items()
        }
        if privileged and (any(drift) or any(any(b) for b in ray_drift.values())):
            raise ValueError(
                f"height scanner '{name}' is the generator's privileged sampling but carries "
                f"drift_range={drift} / ray_cast_drift_range={ray_drift}. The point of it is "
                "to be exact -- randomising it silently reintroduces the perception error it "
                "exists to remove."
            )
        specs.append({
            "sensor": name,
            # The mount prim's leaf name, which is also the rigid-body name off-sim.
            "mount_body": cfg.prim_path.rstrip("/").split("/")[-1],
            "offset_pos": [float(v) for v in cfg.offset.pos],
            "ray_alignment": alignment,
            "direction": list(direction),
            "max_distance": float(cfg.max_distance),
            "resolution": float(pattern.resolution),
            "size": [float(pattern.size[0]), float(pattern.size[1])],
            "ordering": str(pattern.ordering),
            "num_x": width if pattern.ordering == "xy" else height,
            "num_y": height if pattern.ordering == "xy" else width,
            "num_rays": num_rays,
            # Named to match the camera spec so one image-shape resolver serves both.
            "height": height,
            "width": width,
            "sensor_offset": sensor_offset,
            "drift_range": list(drift),
            "ray_cast_drift_range": ray_drift,
            "update_period": float(cfg.update_period),
            "privileged": privileged,
        })
    return specs


def scan_obs_stack(
    env: "ManagerBasedRLEnv", sensor_names: list[str] | str | None = None
) -> torch.Tensor:
    """Stacked height-scan readings, shape ``(num_envs, num_sensors, num_rays)``.

    The single reader.  Goes through ``mdp.height_scan``, so the value is
    ``pos_z - hit_z - offset`` with **no** observation-term noise and **no** clip on either
    side -- which is exactly what ``record_data.py`` has always captured, and therefore what
    makes a recording and a live rollout comparable.

    The sensor axis is kept even at one sensor so the shape matches the depth path's
    ``(B, T, K)``; ``FmTrackingCommand._check_extero_shape`` reads it the same way for both.

    Args:
        env: The environment instance.
        sensor_names: Scanner name(s) to read.  ``None`` resolves to :func:`fm_scan_sensor`.

    Returns:
        ``(num_envs, num_sensors, num_rays)``.

    Raises:
        RuntimeError: When there is no scanner to read.
        ValueError: When the named scanners cast different ray counts, which cannot be
            stacked into one array.
    """
    if sensor_names is None:
        resolved = fm_scan_sensor(env)
        if resolved is None:
            raise RuntimeError(
                "no height scanner in this task's scene, so there is no scan to read. Either "
                "the task is depth-only exteroception or the scanner was trimmed."
            )
        names = [resolved]
    elif isinstance(sensor_names, str):
        names = [sensor_names]
    else:
        names = list(sensor_names)
        if not names:
            raise RuntimeError("scan_obs_stack was given an empty sensor list.")

    rows = [_read_one(env, n) for n in names]
    widths = {int(r.shape[-1]) for r in rows}
    if len(widths) != 1:
        raise ValueError(
            f"scanners {names} cast {sorted(widths)} rays respectively and cannot be stacked "
            "into one array. Read them separately, or give them one grid."
        )
    return torch.stack(rows, dim=1)


def _read_one(env: "ManagerBasedRLEnv", name: str) -> torch.Tensor:
    """One scanner's raw ground-clearance reading, shape ``(num_envs, num_rays)``."""
    from isaaclab.envs.mdp.observations import height_scan as _height_scan

    return _height_scan(env, SceneEntityCfg(name))
