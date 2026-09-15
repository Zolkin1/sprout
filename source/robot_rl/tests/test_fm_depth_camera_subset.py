# Copyright 2026 Zachary Olkin. All rights reserved.

"""A near-only GENERATOR can be served, and only because its own checkpoint says so.

The camera ablation drops the far camera at DATASET build (``apply_camera_subset``), so the
resulting generator conditions on one sensor row while every task installs the ``fm_depth``
group over both cameras. ``FmTrackingCommand._check_extero_shape`` refuses that pair -- rightly,
and three stages downstream of anything that explains it, which is how a near-only fine-tune
died at GRADE on a missing recording.

So ``install_fm_depth_group`` grew a ``cameras`` argument. The danger in that is obvious and is
what these tests are actually about: ``assert_fm_depth_group_covers_every_camera`` exists because
a half-blind generator is SILENT -- the recording, the checkpoint and ``fm.onnx`` all end up
agreeing with each other about one camera, and the result is indistinguishable from a deliberate
one-camera design. An argument that can switch that guard off is exactly the shape of the bug it
was written to stop.

What keeps it honest is PROVENANCE: the subset comes from the checkpoint's own ``extero_spec``
and from nowhere else. There is no CLI flag for it, no task default, and a checkpoint that names
both cameras still gets both. These tests pin the mechanism (the group really does drop the term,
and the guard really does still fire on an undeclared omission) and the provenance (the only
callers derive it from an artifact).

Simulator-free: the cfgs are plain dataclasses and the contract producer has an off-sim entry
point, which is also what the existing depth-contract tests use.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_NEAR = "depth_camera_near"
_FAR = "depth_camera_far"


def _cfg():
    """A minimal env cfg carrying the real cameras, DR events and an observations holder.

    Returns:
        An object with ``scene`` / ``observations`` / ``events``, enough for
        :func:`install_fm_depth_group`.
    """
    from types import SimpleNamespace

    from robot_rl.tasks.manager_based.robot_rl.g1.g1_depth_env_cfg import (
        G1TeacherTerrainDepthEventsCfg,
        G1TeacherTerrainDepthObservationsCfg,
        G1TeacherTerrainDepthSceneCfg,
    )

    return SimpleNamespace(
        scene=G1TeacherTerrainDepthSceneCfg(num_envs=1, env_spacing=2.5),
        observations=G1TeacherTerrainDepthObservationsCfg(),
        events=G1TeacherTerrainDepthEventsCfg(),
    )


def _observed(cfg) -> list[str]:
    """The scene sensors the generator's group actually names, in observation order.

    Args:
        cfg: An env cfg the installer has run on.

    Returns:
        Sensor names, ``fm_`` prefix stripped.
    """
    from isaaclab.managers import ObservationTermCfg as ObsTerm

    from robot_rl.tasks.manager_based.robot_rl.mdp import FM_DEPTH_OBS_GROUP

    group = getattr(cfg.observations, FM_DEPTH_OBS_GROUP)
    return [
        term.params["sensor_cfg"].name.removeprefix("fm_")
        for term in vars(group).values()
        if isinstance(term, ObsTerm) and "sensor_cfg" in (term.params or {})
    ]


def test_the_default_is_unchanged_and_still_covers_both() -> None:
    """No ``cameras`` argument installs exactly what it always did."""
    from robot_rl.tasks.manager_based.robot_rl.g1.g1_depth_env_cfg import (
        install_fm_depth_group,
    )

    cfg = _cfg()
    install_fm_depth_group(cfg)
    assert _observed(cfg) == [_NEAR, _FAR]


def test_a_declared_subset_drops_the_term_and_keeps_the_order() -> None:
    """``cameras=[near]`` leaves one term, and it is the near one.

    ORDER is the second half of this and is not decoration: the FM binds one CNN trunk per
    camera BY INDEX, each with its own angular position embedding (the cones are 100x70 and
    70x42 deg), so a set that was right but ordered wrong would mislabel a camera's tokens by
    1.43x with every shape check satisfied.
    """
    from robot_rl.tasks.manager_based.robot_rl.g1.g1_depth_env_cfg import (
        depth_camera_specs_from_cfg,
        install_fm_depth_group,
    )
    from robot_rl.tasks.manager_based.robot_rl.mdp import FM_DEPTH_OBS_GROUP

    cfg = _cfg()
    install_fm_depth_group(cfg, cameras=[_NEAR])
    assert _observed(cfg) == [_NEAR]
    specs = depth_camera_specs_from_cfg(observations=cfg.observations, scene=cfg.scene,
                                        group=FM_DEPTH_OBS_GROUP)
    assert [s["sensor"] for s in specs] == [_NEAR], (
        "the CONTRACT must follow the group: it is what the recording stamps and what the "
        "exported fm.onnx carries, so a spec still naming two cameras would describe a rig "
        "the generator never saw.")


def test_the_installer_is_reentrant_across_the_subset() -> None:
    """A subset install followed by a full one restores both terms.

    The tasks install the group in ``__post_init__`` and the scripts re-install it from the
    checkpoint, so both directions happen in one process.
    """
    from robot_rl.tasks.manager_based.robot_rl.g1.g1_depth_env_cfg import (
        install_fm_depth_group,
    )

    cfg = _cfg()
    install_fm_depth_group(cfg, cameras=[_NEAR])
    install_fm_depth_group(cfg)
    assert _observed(cfg) == [_NEAR, _FAR]


def test_an_empty_or_unknown_camera_set_is_refused() -> None:
    """The two ways a subset could mean nothing, both of which would otherwise be silent.

    An empty list would leave the generator with no exteroception and every shape check still
    passing at zero sensors; a misspelled name would match nothing, and "matches nothing"
    read as "keep everything" is precisely the two-camera arm this exists to be distinct from.
    """
    from robot_rl.tasks.manager_based.robot_rl.g1.g1_depth_env_cfg import (
        install_fm_depth_group,
    )

    with pytest.raises(ValueError, match="no exteroception"):
        install_fm_depth_group(_cfg(), cameras=[])
    with pytest.raises(ValueError, match="does not carry"):
        install_fm_depth_group(_cfg(), cameras=["depth_camera_middle"])


def test_an_undeclared_omission_is_still_a_fault() -> None:
    """The guard has not lost its teeth: a group short a camera, with nothing declaring it.

    This is the original failure -- the recorder captures one camera, stamps a one-camera
    contract, trains a generator, exports a matching ``fm.onnx``, and every artifact agrees
    with every other about a half-blind generator.
    """
    from robot_rl.tasks.manager_based.robot_rl.g1.g1_depth_env_cfg import (
        assert_fm_depth_group_covers_every_camera,
        install_fm_depth_group,
    )

    cfg = _cfg()
    install_fm_depth_group(cfg, cameras=[_NEAR])
    with pytest.raises(ValueError, match="must see every camera"):
        assert_fm_depth_group_covers_every_camera(cfg)          # no `expect`: full coverage


def test_a_subset_naming_a_camera_the_scene_lacks_is_refused() -> None:
    """A checkpoint and a task describing different robots is not something to reconcile."""
    from robot_rl.tasks.manager_based.robot_rl.g1.g1_depth_env_cfg import (
        assert_fm_depth_group_covers_every_camera,
        install_fm_depth_group,
    )

    cfg = _cfg()
    install_fm_depth_group(cfg)
    with pytest.raises(ValueError, match="different robots"):
        assert_fm_depth_group_covers_every_camera(cfg, expect=[_NEAR, "depth_camera_rear"])


def test_the_subset_is_only_ever_read_off_an_artifact() -> None:
    """Nothing passes ``cameras=`` from a flag, a literal or a task default.

    THE PROVENANCE IS THE WHOLE SAFETY ARGUMENT. A subset that a person can type is a subset
    that can be wrong, and the resulting half-blind generator is self-consistent everywhere
    downstream. The only legitimate source is the checkpoint's own ``extero_spec``, which is
    what ``read_fm_extero_meta`` returns as ``cameras``.
    """
    callers = [
        _ROOT / "scripts/rsl_rl/play.py",
        _ROOT / "scripts/rsl_rl/record_rollouts.py",
        _ROOT / "scripts/rsl_rl/record_data.py",
        _ROOT / "scripts/rsl_rl/augment_registration.py",
        _ROOT / "scripts/rsl_rl/check_depth_xyz_agreement.py",
        (_ROOT / "source/robot_rl/robot_rl/tasks/manager_based/robot_rl/g1"
         / "g1_finetune_env_cfg.py"),
        (_ROOT / "source/robot_rl/robot_rl/tasks/manager_based/robot_rl/g1"
         / "g1_tracker_env_cfg.py"),
    ]
    offenders: list[str] = []
    for path in callers:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name != "install_fm_depth_group":
                continue
            for kw in node.keywords:
                if kw.arg != "cameras":
                    continue
                # The only accepted spelling: a subscript of a meta dict, e.g.
                # ``fm_extero["cameras"]`` / ``meta["cameras"]``.
                ok = (isinstance(kw.value, ast.Subscript)
                      and isinstance(kw.value.slice, ast.Constant)
                      and kw.value.slice.value == "cameras")
                if not ok:
                    offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        "install_fm_depth_group(cameras=...) is passed something other than a checkpoint's "
        "own declared camera list at: " + ", ".join(offenders))


def test_read_fm_extero_meta_reports_the_cameras_and_abstains_when_it_cannot() -> None:
    """``cameras`` is the spec's ``sensor`` field, or ``None`` when any entry lacks it.

    Abstention is the SAFE direction: ``None`` means "no subset declared", the env installs
    every camera, and the command term's own row-count check fires with a message naming the
    mismatch. Guessing a name from ``obs_term`` would hand ``install_fm_depth_group`` a
    sensor to build that may not exist.
    """
    import dataclasses

    import torch

    from robot_rl.flow_matching.config import FlowMatchingConfig
    from robot_rl.flow_matching.online import read_fm_extero_meta

    tmp = pathlib.Path(__file__).resolve().parent / "_tmp_fm_meta.pt"
    base = dataclasses.asdict(FlowMatchingConfig(
        extero_source="depth", extero_sensor_count=1, extero_sensor_dim=3900,
        extero_channels=5))
    cam = {"sensor": _NEAR, "obs_term": "fm_depth_near", "width": 30, "height": 26}
    try:
        torch.save({"config": base, "meta": {"extero_spec": [cam]}}, tmp)
        assert read_fm_extero_meta(str(tmp))["cameras"] == [_NEAR]
        torch.save({"config": base,
                    "meta": {"extero_spec": [{k: v for k, v in cam.items()
                                              if k != "sensor"}]}}, tmp)
        assert read_fm_extero_meta(str(tmp))["cameras"] is None
    finally:
        tmp.unlink(missing_ok=True)
