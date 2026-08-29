# Copyright 2026 Zachary Olkin. All rights reserved.

"""The generator's own dense height scanner -- a PRIVILEGED diagnostic, not a deployable input.

Why this exists: the flow-matching generator does not register terrain.  Displace the robot
downrange and its predicted stair segment slides with it, where the anchor mask requires it
to stay put.  The open question is whether the model *cannot* localise terrain or merely
cannot see it, and today's exteroception -- two 26x30 depth images, domain-randomized,
noisy, self-occluding -- cannot separate those.  This module builds the upper bound: a
robot-centred metric height map, cast exactly, with no noise and no randomisation.  If the
generator still fails to register terrain given this, the fault is the model or the
objective, not the input.

**Nothing off-sim can produce this signal.**  A 3 m x 3 m grid at 0.025 m is 14,641 rays of
exact ground truth, which is why the spec it emits carries ``privileged: True`` and why the
ONNX / MuJoCo / hardware paths refuse it rather than substituting a coarser grid.

The design mirrors ``install_fm_depth_group`` in :mod:`.g1_depth_env_cfg`, with one structural
simplification: a height scan is read from the SENSOR, not from an observation term (see
:mod:`..mdp.observations.height_scan`), so this adds **no observation term and no
observation group**.  That keeps the sagittal-reflection augmentation, ``export_parameters``
and all obs-group plumbing entirely out of it.

The one hard constraint: **the policy's own ``height_scanner`` must not be touched.**  The
depth teacher's critic observes it, so widening it changes the critic's input width and
``runner.load()`` refuses the checkpoint at record time (``PPORunnerCfg`` has no
``load_critic = False``).  A second sensor is the only safe route, and
:func:`add_fm_height_scan` asserts the policy's is unchanged on the way out.
"""

from __future__ import annotations

from types import SimpleNamespace

from isaaclab.sensors import RayCasterCfg

from robot_rl.tasks.manager_based.robot_rl import mdp
from robot_rl.tasks.manager_based.robot_rl.mdp.observations.height_scan import (
    FM_SCAN_SENSOR,
    RL_SCAN_SENSOR,
)

# The shipped grid. 3.0 m x 3.0 m at 0.025 m is 121 x 121 = 14,641 rays, which is exactly
# what the encoder's stride-(1,2,2,2) trunk folds to a 16 x 16 = 256-token grid -- the token
# count and the ray count are coupled through the conv stack, so changing one without the
# other silently changes the sequence length. 0.025 m over 3.0 m is the combination that
# lands on 256 tokens at a sane extent.
_FM_SCAN_SIZE: tuple[float, float] = (3.0, 3.0)
_FM_SCAN_RESOLUTION: float = 0.025

# Robot-centred. The policy's scanner sits 0.75 m FORWARD of the pelvis, which for a 1.5 m
# grid keeps it over the ground the robot is about to step on; inheriting that offset for a
# 3 m grid would put the window at x in [-0.75, +2.25] instead of symmetric, and nothing
# downstream could detect it. Symmetric also keeps the positional encoding's metre grid
# centred on the frame the FM's window lives in.
_FM_SCAN_OFFSET_XY: tuple[float, float] = (0.0, 0.0)

# `ray_caster.py` hands every finite hit to one point instancer. At 24 play envs a dense
# grid is ~350k sphere instances and the viewer stops responding, so debug_vis is refused
# above this rather than merely defaulted off.
_DEBUG_VIS_RAY_BUDGET: int = 2000


def _make_fm_height_scanner(
    policy_scanner: RayCasterCfg,
    size: tuple[float, float],
    resolution: float,
    offset_xy: tuple[float, float],
) -> RayCasterCfg:
    """The generator's scanner, derived from the POLICY's by replacing only the grid.

    Everything physical is shared *by construction* rather than restated: the mount prim, the
    yaw alignment, the target mesh, ``max_distance``, both drift ranges and the 20 m ray-start
    lift all come from ``policy_scanner``.  Only the pattern and the in-plane offset differ,
    and both are arguments.  This is the same trick ``_make_depth_camera`` uses with
    ``_CameraSpec`` -- two samplings of one physical thing cannot drift apart if only one of
    them states the physics.

    ``RayCaster.meshes`` is a class-level cache keyed by prim path and both scanners target
    ``/World/ground``, so the second sensor reuses the already-converted warp mesh.

    Args:
        policy_scanner: The scene's existing ``height_scanner`` cfg.
        size: ``(size_x, size_y)`` full extents in metres; ``size[0]`` is forward.
        resolution: Grid spacing in metres.
        offset_xy: In-plane grid centre relative to the mount prim.

    Returns:
        A new ``RayCasterCfg``; the input is not mutated.
    """
    from isaaclab.sensors.ray_caster import patterns

    return policy_scanner.replace(
        pattern_cfg=patterns.GridPatternCfg(resolution=float(resolution), size=list(size)),
        offset=RayCasterCfg.OffsetCfg(
            # Keep the policy scanner's ray-start height; only the in-plane centre moves.
            pos=(float(offset_xy[0]), float(offset_xy[1]), float(policy_scanner.offset.pos[2])),
            rot=policy_scanner.offset.rot,
        ),
        debug_vis=False,
        update_period=0.0,
    )


def add_fm_height_scan(
    cfg,
    size: tuple[float, float] = _FM_SCAN_SIZE,
    resolution: float = _FM_SCAN_RESOLUTION,
    offset_xy: tuple[float, float] = _FM_SCAN_OFFSET_XY,
    debug_vis: bool = False,
) -> str | None:
    """Give the generator its own dense height scanner, leaving the policy's untouched.

    A no-op when the requested grid already matches the policy's: with no ``fm_height_scanner``
    in the scene, :func:`mdp.fm_scan_sensor` resolves to the policy's scanner, which is
    exactly what every reader named before this existed.  So a task that does not opt in is
    bit-identical to today and no consumer needs a flag.

    Args:
        cfg: The env cfg to mutate in place.
        size: ``(size_x, size_y)`` full extents in metres; ``size[0]`` is forward/downrange.
        resolution: Grid spacing in metres.
        offset_xy: Grid centre relative to the mount prim, in the sensor's yaw-aligned plane.
        debug_vis: Draw the hit points.  Refused above ``_DEBUG_VIS_RAY_BUDGET`` rays.

    Returns:
        The sensor name that was added, or ``None`` when this was a no-op.

    Raises:
        ValueError: When the scene has no ``height_scanner`` to derive from; when the FM
            scanner is already present; when ``debug_vis`` is asked for on too many rays; or
            when an event term randomises either scanner (which would defeat the point).
    """
    from robot_rl.flow_matching.transforms import scan_grid_shape

    scene = getattr(cfg, "scene", None)
    policy_scanner = getattr(scene, RL_SCAN_SENSOR, None) if scene is not None else None
    if policy_scanner is None:
        raise ValueError(
            f"this task's scene has no '{RL_SCAN_SENSOR}', so there is nothing to derive the "
            "generator's scanner from. The dense scan shares its mount, alignment and target "
            "mesh with the policy's by construction rather than restating them."
        )
    if getattr(scene, FM_SCAN_SENSOR, None) is not None:
        raise ValueError(
            f"'{FM_SCAN_SENSOR}' is already on this scene. add_fm_height_scan is not "
            "idempotent -- it is reachable from record_data.py, play.py and a task "
            "__post_init__, and adding it twice would leave two sensors the readers cannot "
            "choose between."
        )

    pat = policy_scanner.pattern_cfg
    same_grid = (
        (float(resolution), tuple(float(v) for v in size)) ==
        (float(pat.resolution), tuple(float(v) for v in pat.size))
        and tuple(float(v) for v in offset_xy) == tuple(float(v) for v in policy_scanner.offset.pos[:2])
    )
    if same_grid:
        print(f"[fm-scan] requested grid {size} @ {resolution} m matches "
              f"'{RL_SCAN_SENSOR}'; no separate scanner added (readers fall back to it).")
        return None

    height, width = scan_grid_shape(resolution, size, pat.ordering)
    num_rays = height * width
    if debug_vis and num_rays > _DEBUG_VIS_RAY_BUDGET:
        n_envs = int(getattr(scene, "num_envs", 1) or 1)
        raise ValueError(
            f"debug_vis on a {num_rays}-ray scanner would draw {num_rays * n_envs} marker "
            f"instances ({n_envs} envs). Raise _DEBUG_VIS_RAY_BUDGET only if you have also "
            "made the visualiser stride its points."
        )

    # No DR may touch either scanner. There is none today, so this is a tripwire for the day
    # someone copies a camera-randomisation term: the inverse of install_fm_depth_group, which
    # has to EXTEND its DR events to cover both samplings, because here the whole value is
    # that the reading is exact.
    events = getattr(cfg, "events", None)
    if events is not None:
        for term_name, term in vars(events).items():
            params = getattr(term, "params", None) or {}
            named = []
            for key in ("sensor_cfg", "sensor_cfgs"):
                entry = params.get(key)
                if entry is None:
                    continue
                named += [getattr(e, "name", None) for e in
                          (entry if isinstance(entry, (list, tuple)) else [entry])]
            if RL_SCAN_SENSOR in named or FM_SCAN_SENSOR in named:
                raise ValueError(
                    f"event '{term_name}' randomises a height scanner ({named}). The dense "
                    "scan exists to be exact -- randomising it silently reintroduces the "
                    "perception error the experiment is trying to remove. Disable the term "
                    "for this run, or scope it to the policy's scanner only and re-check "
                    "height_scan_specs' privileged-drift guard."
                )

    setattr(scene, FM_SCAN_SENSOR, _make_fm_height_scanner(
        policy_scanner, size, resolution, offset_xy))

    # The policy's scanner is the one thing that must not have moved: the depth teacher's
    # critic observes it, so a changed width makes runner.load() reject the checkpoint.
    assert policy_scanner.pattern_cfg is pat, "the policy scanner's pattern was replaced"
    assert (float(pat.resolution), tuple(pat.size)) == (
        float(policy_scanner.pattern_cfg.resolution),
        tuple(policy_scanner.pattern_cfg.size)), "the policy scanner's grid changed"

    pol_h, pol_w = scan_grid_shape(pat.resolution, pat.size, pat.ordering)
    pol_rays = pol_h * pol_w
    print(f"[fm-scan] added '{FM_SCAN_SENSOR}': {size[0]}x{size[1]} m @ {resolution} m, "
          f"fold (H, W) = ({height}, {width}) = {num_rays} rays, centre {offset_xy}, "
          f"{num_rays * 4 / 1024:.1f} KiB/window f32 ({num_rays * 2 / 1024:.1f} KiB f16). "
          f"{num_rays / pol_rays:.0f}x the policy's {pol_rays} rays. "
          "PRIVILEGED: exact and noise-free, so not deployable.")
    return FM_SCAN_SENSOR


def height_scan_specs_from_cfg(
    scene=None, sensor_names: list[str] | None = None
) -> list[dict]:
    """The scan contract for real cfgs, with no simulator running.

    Runs the production producer (:func:`mdp.height_scan_specs`) over instantiated cfgs, so an
    off-sim tool and the contract tests see the same numbers the training task declares rather
    than a restatement that can go stale.  The sim-free path skips only the live ray-count
    cross-check, since there is no cast to compare against.

    Args:
        scene: A scene cfg holding the scanner(s).  Defaults to the depth teacher's scene with
            the FM scanner added, i.e. the configuration a recording would use.
        sensor_names: Scanner names to describe.  Defaults to every ``RayCasterCfg`` on the
            scene whose name mentions a height scanner.

    Returns:
        One dict per scanner, in the order given.
    """
    if scene is None:
        from robot_rl.tasks.manager_based.robot_rl.g1.g1_depth_env_cfg import (
            G1TeacherTerrainDepthEnvCfg,
        )
        env_cfg = G1TeacherTerrainDepthEnvCfg()
        add_fm_height_scan(env_cfg)
        scene = env_cfg.scene
    if sensor_names is None:
        sensor_names = [n for n in (RL_SCAN_SENSOR, FM_SCAN_SENSOR)
                        if getattr(scene, n, None) is not None]
    sensors = {n: SimpleNamespace(cfg=getattr(scene, n)) for n in sensor_names}
    env = SimpleNamespace(scene=SimpleNamespace(sensors=sensors))
    return mdp.height_scan_specs(env, sensor_names)
