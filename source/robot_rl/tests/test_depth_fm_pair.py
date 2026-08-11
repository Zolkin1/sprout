# Copyright 2026 Zachary Olkin. All rights reserved.

"""The generator's own depth camera pair: a second SAMPLING of the policy's lenses.

The FM's depth resolution is decoupled from the policy's so the generator's vision input
can change without retraining the teacher. What must NOT decouple is the physics: same
mount, same cone, same range, same normalization. These tests pin that -- and pin the
no-op case, since "nothing is added when the resolutions match" is what keeps every
consumer's group scoping free.

Runs without a simulator: the cfg classes are instantiated and the production contract
producer is run over them, exactly as ``depth_camera_specs_from_cfg`` does.
"""

from __future__ import annotations

import math

import pytest

from isaaclab.sensors.ray_caster import MultiMeshRayCasterCameraCfg

import robot_rl.tasks.manager_based.robot_rl.g1.g1_depth_env_cfg as dcfg
from robot_rl.tasks.manager_based.robot_rl.mdp.observations.depth import FM_DEPTH_OBS_GROUP

_FM_RES = (60, 52)


def _env_cfg():
    """A minimal stand-in carrying the three cfg objects ``add_fm_depth_pair`` touches."""

    class _Cfg:
        def __init__(self):
            self.scene = dcfg.G1TeacherTerrainDepthSceneCfg(num_envs=4, env_spacing=2.5)
            self.observations = dcfg.G1TeacherTerrainDepthObservationsCfg()
            self.events = dcfg.G1TeacherTerrainDepthEventsCfg()

    return _Cfg()


def _cameras(cfg) -> dict[str, MultiMeshRayCasterCameraCfg]:
    """Every depth camera on a scene cfg, by field name."""
    return {n: s for n, s in vars(cfg.scene).items()
            if isinstance(s, MultiMeshRayCasterCameraCfg)}


def _half_angles(cam: dict) -> tuple[float, float, float, float]:
    """The four pinhole half-angles (left, right, up, down) in radians.

    Resolution-INVARIANT by construction: with ``fx = W*f/aperture`` and ``cx = W/2``,
    ``atan(cx/fx) = atan(aperture/2f)``, in which the width cancels. This is the quantity
    that says two cameras look through the same lens whatever grid they sample it on.
    """
    return (math.atan(cam["cx"] / cam["fx"]), math.atan((cam["width"] - cam["cx"]) / cam["fx"]),
            math.atan(cam["cy"] / cam["fy"]), math.atan((cam["height"] - cam["cy"]) / cam["fy"]))


##
# The no-op case
##

def test_matching_resolution_adds_nothing():
    """The gating story: no second pair, no group, so consumers fall through to today's path.

    ``mdp.fm_depth_groups`` returns ``None`` when the group is absent, and ``None`` means
    "all groups" -- so every call site that scopes itself is a no-op here without any flag
    being plumbed to it.
    """
    cfg = _env_cfg()
    before = set(_cameras(cfg))
    assert dcfg.add_fm_depth_pair(cfg, dcfg._DEPTH_W, dcfg._DEPTH_H) is None
    assert set(_cameras(cfg)) == before
    assert getattr(cfg.observations, FM_DEPTH_OBS_GROUP, None) is None


def test_a_task_without_cameras_is_refused():
    """Mirroring cameras that are not there is a task mix-up, not something to skip."""
    cfg = _env_cfg()
    cfg.scene.depth_camera_near = None
    with pytest.raises(ValueError, match="depth cameras"):
        dcfg.add_fm_depth_pair(cfg, *_FM_RES)


##
# What the pair is
##

def test_the_pair_is_added_with_its_own_resolution():
    cfg = _env_cfg()
    assert dcfg.add_fm_depth_pair(cfg, *_FM_RES) == FM_DEPTH_OBS_GROUP
    cams = _cameras(cfg)
    assert set(cams) == {"depth_camera_near", "depth_camera_far",
                         "fm_depth_camera_near", "fm_depth_camera_far"}
    for name, cam in cams.items():
        want = _FM_RES if name.startswith("fm_") else (dcfg._DEPTH_W, dcfg._DEPTH_H)
        assert (cam.pattern_cfg.width, cam.pattern_cfg.height) == want, name


def test_the_policy_group_is_untouched():
    """The whole point: the policy's observation vector must not change."""
    cfg, ref = _env_cfg(), _env_cfg()
    dcfg.add_fm_depth_pair(cfg, *_FM_RES)
    assert vars(cfg.observations.policy).keys() == vars(ref.observations.policy).keys()
    for term in ("depth_near", "depth_far"):
        got, want = getattr(cfg.observations.policy, term), getattr(ref.observations.policy, term)
        assert got.params == want.params and got.clip == want.clip


@pytest.mark.parametrize("suffix", ["near", "far"])
def test_both_samplings_describe_one_physical_camera(suffix):
    """Identical cone and identical physical fields -- only the SAMPLING may differ.

    Asserted through the real contract producer rather than on the cfg objects, because the
    contract is what travels into the ONNX and what the deploy compares.
    """
    cfg = _env_cfg()
    dcfg.add_fm_depth_pair(cfg, *_FM_RES)
    rl = {c["sensor"]: c for c in
          dcfg.depth_camera_specs_from_cfg(cfg.observations, cfg.scene, group="policy")}
    fm = {c["sensor"]: c for c in
          dcfg.depth_camera_specs_from_cfg(cfg.observations, cfg.scene, group=FM_DEPTH_OBS_GROUP)}
    a, b = rl[f"depth_camera_{suffix}"], fm[f"fm_depth_camera_{suffix}"]

    assert (a["width"], a["height"]) == (dcfg._DEPTH_W, dcfg._DEPTH_H)
    assert (b["width"], b["height"]) == _FM_RES
    for x, y in zip(_half_angles(a), _half_angles(b)):
        assert x == pytest.approx(y, abs=1e-12), "the two samplings are not the same lens"
    # Everything that is not naming or sampling must be equal.
    sampling = {"obs_term", "sensor", "width", "height", "fx", "fy", "cx", "cy", "update_period"}
    for field in set(a) - sampling:
        assert a[field] == b[field], field


def test_the_two_groups_do_not_mix():
    """Each group resolves exactly its own two cameras, in order."""
    cfg = _env_cfg()
    dcfg.add_fm_depth_pair(cfg, *_FM_RES)
    rl = dcfg.depth_camera_specs_from_cfg(cfg.observations, cfg.scene, group="policy")
    fm = dcfg.depth_camera_specs_from_cfg(cfg.observations, cfg.scene, group=FM_DEPTH_OBS_GROUP)
    assert [c["sensor"] for c in rl] == ["depth_camera_near", "depth_camera_far"]
    assert [c["sensor"] for c in fm] == ["fm_depth_camera_near", "fm_depth_camera_far"]


##
# The shared DR draw
##

@pytest.mark.parametrize("term_name", ["randomize_camera_extrinsics_near",
                                       "randomize_camera_intrinsics_near",
                                       "resample_depth_noise_near"])
def test_dr_terms_are_extended_not_duplicated(term_name):
    """One draw covers both samplings, so there is no second term to order against the first.

    Drawing independently would make the two views ±1 cm of mount and ±2 % of focal length
    apart -- two different physical cameras, which is precisely the silent inconsistency the
    camera contract exists to prevent.
    """
    cfg = _env_cfg()
    before = set(vars(cfg.events))
    dcfg.add_fm_depth_pair(cfg, *_FM_RES)
    assert set(vars(cfg.events)) == before, "a new event term appeared; extend, do not add"

    params = getattr(cfg.events, term_name).params
    assert "sensor_cfg" not in params
    assert [s.name for s in params["sensor_cfgs"]] == ["depth_camera_near",
                                                       "fm_depth_camera_near"]


def test_intrinsics_dr_stays_relative():
    """The shared intrinsics draw is only correct because the operation is a RATIO.

    ``fx/fy/cx/cy`` all scale with resolution, so an absolute draw could not transfer
    between the two samplings. The event refuses a non-scale operation for a list, and this
    pins the cfg side of that precondition.
    """
    cfg = _env_cfg()
    for suffix in ("near", "far"):
        assert getattr(cfg.events, f"randomize_camera_intrinsics_{suffix}").params[
            "operation"] == "scale"


def test_noise_cfg_leaves_the_resolution_to_the_camera():
    """One noise cfg serves both samplings, so it must not restate a resolution."""
    noise = dcfg._make_depth_noise()
    assert noise.height is None and noise.width is None


##
# Play trims
##

def test_play_trims_null_every_depth_event_including_shared_ones():
    """Derived from the event functions, not a list of names that goes stale."""
    cfg = _env_cfg()
    dcfg.add_fm_depth_pair(cfg, *_FM_RES)
    dcfg.apply_depth_play_trims(cfg)
    live = [n for n, t in vars(cfg.events).items()
            if t is not None
            and getattr(getattr(t, "func", None), "__module__", "").endswith("mdp.events.depth")]
    assert not live, f"depth DR still live in a play cfg: {live}"


def test_play_trims_show_only_the_policy_cameras():
    """Both pairs sit on the same mounts, so drawing both just doubles every frustum."""
    cfg = _env_cfg()
    dcfg.add_fm_depth_pair(cfg, *_FM_RES)
    dcfg.apply_depth_play_trims(cfg)
    vis = {n: c.debug_vis for n, c in _cameras(cfg).items()}
    assert vis == {"depth_camera_near": True, "depth_camera_far": True,
                   "fm_depth_camera_near": False, "fm_depth_camera_far": False}


def test_adding_the_pair_after_the_trims_finds_no_events_to_extend():
    """A trimmed play cfg has no DR terms left; extending must skip them, not crash."""
    cfg = _env_cfg()
    dcfg.apply_depth_play_trims(cfg)
    assert dcfg.add_fm_depth_pair(cfg, *_FM_RES) == FM_DEPTH_OBS_GROUP


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
