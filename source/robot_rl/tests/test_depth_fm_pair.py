# Copyright 2026 Zachary Olkin. All rights reserved.

"""The generator's depth group: BOTH cameras, at the policy's grid or its own.

The policy reads the NEAR camera alone -- the far one reads terrain ahead, which is what
places a reference and is therefore the generator's job. So the tracker's camera set is a
strict, name-matched SUBSET of the generator's, and ``install_fm_depth_group`` is what puts
the far camera back in front of an observation term. These tests pin that subset relation,
pin that the generator's group covers every camera the scene carries (the failure it does
not is silent: a one-camera recording, a one-camera checkpoint and a one-camera ``fm.onnx``
all agree with each other), and pin the older invariant that a SECOND sampling of one lens
may differ in the ray grid and in nothing else.

Also pinned here: one physical camera yields ONE frame per control step. At the shared grid
the near camera is read by the policy's term AND the generator's in the same step, where the
hardware hands them one image.

Runs without a simulator: the cfg classes are instantiated and the production contract
producer is run over them, exactly as ``depth_camera_specs_from_cfg`` does.
"""

from __future__ import annotations

import math

import pytest
import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCameraCfg

import robot_rl.tasks.manager_based.robot_rl.g1.g1_depth_env_cfg as dcfg
from robot_rl.tasks.manager_based.robot_rl.mdp.observations import depth as depth_obs
from robot_rl.tasks.manager_based.robot_rl.mdp.observations.depth import FM_DEPTH_OBS_GROUP
from robot_rl.tasks.manager_based.robot_rl.mdp.sensors.noise import ensure_depth_noiser

_FM_RES = (60, 52)
_POLICY_RES = (dcfg._DEPTH_W, dcfg._DEPTH_H)


def _env_cfg():
    """A minimal stand-in carrying the three cfg objects the installer touches."""

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


def _sensors(cfg, group: str) -> list[str]:
    """The sensors one observation group's depth terms resolve to, in observation order."""
    return [c["sensor"]
            for c in dcfg.depth_camera_specs_from_cfg(cfg.observations, cfg.scene, group=group)]


def _half_angles(cam: dict) -> tuple[float, float, float, float]:
    """The four pinhole half-angles (left, right, up, down) in radians.

    Resolution-INVARIANT by construction: with ``fx = W*f/aperture`` and ``cx = W/2``,
    ``atan(cx/fx) = atan(aperture/2f)``, in which the width cancels. This is the quantity
    that says two cameras look through the same lens whatever grid they sample it on.
    """
    return (math.atan(cam["cx"] / cam["fx"]), math.atan((cam["width"] - cam["cx"]) / cam["fx"]),
            math.atan(cam["cy"] / cam["fy"]), math.atan((cam["height"] - cam["cy"]) / cam["fy"]))


##
# The policy's cameras are a strict subset of the generator's
##

def test_the_policy_reads_the_near_camera_alone():
    """The teacher's own contract, straight out of the shipped cfgs."""
    cfg = _env_cfg()
    assert _sensors(cfg, "policy") == ["depth_camera_near"]
    assert [n for n, _ in dcfg._POLICY_CAMERAS] == ["depth_camera_near"]


def test_the_image_variant_reads_the_same_cameras_as_the_flat_one():
    """The two depth teachers differ in observation SHAPE and in nothing else.

    Letting them disagree about how many cameras the policy reads would turn the CNN-vs-MLP
    comparison into a two-variable one, and would export two different tracker contracts for
    one robot.
    """
    flat = dcfg.G1TeacherTerrainDepthObservationsCfg()
    image = dcfg.G1TeacherTerrainDepthImageObservationsCfg()
    scene = dcfg.G1TeacherTerrainDepthSceneCfg(num_envs=1, env_spacing=2.5)

    flat_sensors = [c["sensor"]
                    for c in dcfg.depth_camera_specs_from_cfg(flat, scene, group="policy")]
    image_groups = [g for g in vars(image) if g.startswith("depth_") and g.endswith("_image")]
    image_sensors = [c["sensor"] for g in image_groups
                     for c in dcfg.depth_camera_specs_from_cfg(image, scene, group=g)]
    assert flat_sensors == image_sensors == ["depth_camera_near"]


def test_the_policys_cameras_are_a_strict_subset_of_the_generators():
    """One physical pair; the policy reads part of it and the generator reads all of it.

    Checked through the real contract producer on both sides, because the contract is what
    travels into ``tracker.onnx`` / ``fm.onnx`` and what the deploy runtime compares.
    """
    cfg = _env_cfg()
    dcfg.install_fm_depth_group(cfg)
    policy, generator = _sensors(cfg, "policy"), _sensors(cfg, FM_DEPTH_OBS_GROUP)
    assert set(policy) < set(generator), "the policy must read FEWER cameras, not other ones"
    assert generator == [n for n, _ in dcfg._DEPTH_CAMERAS]


def test_the_generator_group_must_cover_every_camera_on_the_scene():
    """The silent failure, named: a half-blind generator whose artifacts all agree.

    Without the group, ``mdp.fm_depth_groups`` returns ``None`` = "every group", which on
    this task is the policy's near camera alone -- so the recorder would stamp a one-camera
    ``depth_camera_spec`` and everything downstream would be consistent with it.
    """
    cfg = _env_cfg()
    with pytest.raises(ValueError, match="fm_depth"):
        dcfg.assert_fm_depth_group_covers_every_camera(cfg)

    dcfg.install_fm_depth_group(cfg)
    dcfg.assert_fm_depth_group_covers_every_camera(cfg)

    # A group that has lost a camera is refused just as loudly as a missing one.
    getattr(cfg.observations, FM_DEPTH_OBS_GROUP).fm_depth_far = None
    with pytest.raises(ValueError, match="every camera"):
        dcfg.assert_fm_depth_group_covers_every_camera(cfg)


##
# The shared arm: the generator reads the POLICY's cameras
##

def test_the_matching_resolution_reads_the_scenes_own_cameras():
    """No second sampling, so no second ray cast and no second DR draw to order."""
    cfg = _env_cfg()
    before_cams, before_events = set(_cameras(cfg)), {
        n: dict(t.params) for n, t in vars(cfg.events).items() if hasattr(t, "params")
    }

    assert dcfg.install_fm_depth_group(cfg, *_POLICY_RES) == FM_DEPTH_OBS_GROUP
    assert set(_cameras(cfg)) == before_cams, "a second camera pair appeared at one grid"
    assert _sensors(cfg, FM_DEPTH_OBS_GROUP) == ["depth_camera_near", "depth_camera_far"]
    assert {n: dict(t.params) for n, t in vars(cfg.events).items()
            if hasattr(t, "params")} == before_events, "DR moved with nothing to share"


def test_the_resolution_defaults_to_the_policys():
    """Leaving it unset is the normal case and must not mean 'no group'."""
    cfg, explicit = _env_cfg(), _env_cfg()
    dcfg.install_fm_depth_group(cfg)
    dcfg.install_fm_depth_group(explicit, *_POLICY_RES)
    assert (getattr(cfg.observations, FM_DEPTH_OBS_GROUP).to_dict()
            == getattr(explicit.observations, FM_DEPTH_OBS_GROUP).to_dict())


def test_a_task_without_cameras_is_refused():
    """Mirroring cameras that are not there is a task mix-up, not something to skip."""
    cfg = _env_cfg()
    cfg.scene.depth_camera_near = None
    with pytest.raises(ValueError, match="depth cameras"):
        dcfg.install_fm_depth_group(cfg, *_FM_RES)


##
# The split arm: the generator samples its own grid
##

def test_the_pair_is_added_with_its_own_resolution():
    """At a DIFFERENT grid the group gets its own ``fm_``-prefixed cameras, both of them.

    The policy's two sensors keep the policy's resolution and the duplicates carry the
    generator's, so one lens is sampled twice rather than resampled -- which is what lets the
    generator's vision input change without retraining the teacher.
    """
    cfg = _env_cfg()
    assert dcfg.install_fm_depth_group(cfg, *_FM_RES) == FM_DEPTH_OBS_GROUP
    cams = _cameras(cfg)
    assert set(cams) == {"depth_camera_near", "depth_camera_far",
                         "fm_depth_camera_near", "fm_depth_camera_far"}
    for name, cam in cams.items():
        want = _FM_RES if name.startswith("fm_") else _POLICY_RES
        assert (cam.pattern_cfg.width, cam.pattern_cfg.height) == want, name


def test_the_policy_group_is_untouched():
    """The whole point: the policy's observation vector must not change."""
    cfg, ref = _env_cfg(), _env_cfg()
    dcfg.install_fm_depth_group(cfg, *_FM_RES)
    assert vars(cfg.observations.policy).keys() == vars(ref.observations.policy).keys()
    got, want = cfg.observations.policy.depth_near, ref.observations.policy.depth_near
    assert got.params == want.params and got.clip == want.clip


@pytest.mark.parametrize("suffix", ["near", "far"])
def test_both_samplings_describe_one_physical_camera(suffix):
    """Identical cone and identical physical fields -- only the SAMPLING may differ.

    Asserted through the real contract producer rather than on the cfg objects, because the
    contract is what travels into the ONNX and what the deploy compares. The reference side
    is the SHARED arm's contract, which is the scene's own cameras.
    """
    shared, split = _env_cfg(), _env_cfg()
    dcfg.install_fm_depth_group(shared)
    dcfg.install_fm_depth_group(split, *_FM_RES)
    a = {c["sensor"]: c for c in dcfg.depth_camera_specs_from_cfg(
        shared.observations, shared.scene, group=FM_DEPTH_OBS_GROUP)}[f"depth_camera_{suffix}"]
    b = {c["sensor"]: c for c in dcfg.depth_camera_specs_from_cfg(
        split.observations, split.scene,
        group=FM_DEPTH_OBS_GROUP)}[f"fm_depth_camera_{suffix}"]

    assert (a["width"], a["height"]) == _POLICY_RES
    assert (b["width"], b["height"]) == _FM_RES
    for x, y in zip(_half_angles(a), _half_angles(b)):
        assert x == pytest.approx(y, abs=1e-12), "the two samplings are not the same lens"
    # Everything that is not naming or sampling must be equal.
    sampling = {"obs_term", "sensor", "width", "height", "fx", "fy", "cx", "cy", "update_period"}
    for field in set(a) - sampling:
        assert a[field] == b[field], field


def test_the_two_groups_do_not_mix():
    """Each group resolves exactly its own cameras, in order."""
    cfg = _env_cfg()
    dcfg.install_fm_depth_group(cfg, *_FM_RES)
    assert _sensors(cfg, "policy") == ["depth_camera_near"]
    assert _sensors(cfg, FM_DEPTH_OBS_GROUP) == ["fm_depth_camera_near", "fm_depth_camera_far"]


def test_installing_again_tears_down_the_previous_sampling():
    """Re-entrant, because it really is called twice.

    A task installs the group at the policy's resolution in ``__post_init__`` and ``play.py``
    re-installs it at whatever the checkpoint states. Leaving the first call's ``fm_``
    cameras behind would cast a grid nothing reads, every step.
    """
    cfg, fresh = _env_cfg(), _env_cfg()
    dcfg.install_fm_depth_group(cfg, *_FM_RES)
    dcfg.install_fm_depth_group(cfg)
    assert set(_cameras(cfg)) == set(_cameras(fresh))
    assert _sensors(cfg, FM_DEPTH_OBS_GROUP) == ["depth_camera_near", "depth_camera_far"]
    for suffix in ("near", "far"):
        params = getattr(cfg.events, f"randomize_camera_extrinsics_{suffix}").params
        assert "sensor_cfgs" not in params
        assert params["sensor_cfg"].name == f"depth_camera_{suffix}"


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
    dcfg.install_fm_depth_group(cfg, *_FM_RES)
    assert set(vars(cfg.events)) == before, "a new event term appeared; extend, do not add"

    params = getattr(cfg.events, term_name).params
    assert "sensor_cfg" not in params
    assert [s.name for s in params["sensor_cfgs"]] == ["depth_camera_near",
                                                       "fm_depth_camera_near"]


def test_the_far_camera_keeps_its_dr_though_the_policy_never_reads_it():
    """It is unobserved on the teacher, not absent -- and the generator will observe it.

    Dropping its DR terms would also drop its noiser (built by ``resample_depth_noise``) and
    quietly hand the generator a noise-free far camera.
    """
    cfg = _env_cfg()
    for term_name in ("randomize_camera_extrinsics_far", "randomize_camera_intrinsics_far",
                      "resample_depth_noise_far"):
        assert getattr(cfg.events, term_name) is not None
    assert cfg.scene.depth_camera_far is not None


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
# One physical camera, ONE frame per control step
##

class _Scene:
    """Just enough scene for ``depth_image`` and ``ensure_depth_noiser``."""

    def __init__(self, sensors: dict):
        """Args:
            sensors: Sensor stand-ins by scene name.
        """
        self.sensors = sensors

    def __getitem__(self, name: str):
        """Args:
            name: Scene name of the sensor.

        Returns:
            The sensor stand-in.
        """
        return self.sensors[name]


def _noise_env(num_envs: int = 3):
    """An env stand-in whose one camera returns a fixed image, with the real noiser attached.

    Returns:
        ``(env, params)`` -- the stand-in and the observation-term keyword arguments.
    """
    cam_cfg = dcfg._make_depth_camera(dcfg._NEAR)
    width, height = cam_cfg.pattern_cfg.width, cam_cfg.pattern_cfg.height
    depth = torch.rand(num_envs, height, width, 1) * 4.0 + 0.5

    class _Sensor:
        cfg = cam_cfg
        data = type("_D", (), {"output": {"distance_to_image_plane": depth}})()

    class _Env:
        pass

    env = _Env()
    env.num_envs = num_envs
    env.device = "cpu"
    env.common_step_counter = 7
    env.scene = _Scene({"depth_camera_near": _Sensor()})
    ensure_depth_noiser(env, "depth_camera_near", dcfg._make_depth_noise())
    return env, dcfg._depth_obs_params(dcfg._NEAR)


def test_one_camera_yields_one_frame_per_control_step():
    """The policy's term and the generator's read one lens; hardware hands them one image.

    Two independent draws would differ by a 2.5% dropout pattern and a per-pixel additive
    term, and would make ``--dump_replay`` report a divergence that is the harness's fault:
    the C++ runtime rebuilds the generator's extero row from the tracker's recorded
    ``depth_raw``.
    """
    env, params = _noise_env()
    sensor_cfg = SceneEntityCfg("depth_camera_near")
    first = depth_obs.depth_image(env, sensor_cfg, **params)
    second = depth_obs.depth_image(env, sensor_cfg, **params)
    assert torch.equal(first, second), "two reads of one camera in one step disagree"

    env.common_step_counter += 1
    assert not torch.equal(first, depth_obs.depth_image(env, sensor_cfg, **params)), (
        "the next control step must redraw -- per-pixel noise and dropout are per FRAME"
    )


def test_the_shared_frame_survives_the_image_shaped_term():
    """``depth_image_4d`` is the same frame in another shape, not another frame."""
    env, params = _noise_env()
    sensor_cfg = SceneEntityCfg("depth_camera_near")
    flat = depth_obs.depth_image(env, sensor_cfg, **params)
    image = depth_obs.depth_image_4d(env, sensor_cfg, height=dcfg._DEPTH_H,
                                     width=dcfg._DEPTH_W, **params)
    assert torch.equal(flat, image.reshape(env.num_envs, -1))


def test_a_reset_redraws_the_frame():
    """The per-episode camera state moves, so the frame drawn against it must not persist."""
    env, params = _noise_env()
    sensor_cfg = SceneEntityCfg("depth_camera_near")
    first = depth_obs.depth_image(env, sensor_cfg, **params)
    getattr(env, "_robot_rl_depth_noisers")["depth_camera_near"].reset()
    assert not torch.equal(first, depth_obs.depth_image(env, sensor_cfg, **params))


##
# Play trims
##

def test_play_trims_null_every_depth_event_including_shared_ones():
    """Derived from the event functions, not a list of names that goes stale."""
    cfg = _env_cfg()
    dcfg.install_fm_depth_group(cfg, *_FM_RES)
    dcfg.apply_depth_play_trims(cfg)
    live = [n for n, t in vars(cfg.events).items()
            if t is not None
            and getattr(getattr(t, "func", None), "__module__", "").endswith("mdp.events.depth")]
    assert not live, f"depth DR still live in a play cfg: {live}"


def test_play_trims_show_only_the_policy_cameras():
    """Both pairs sit on the same mounts, so drawing both just doubles every frustum."""
    cfg = _env_cfg()
    dcfg.install_fm_depth_group(cfg, *_FM_RES)
    dcfg.apply_depth_play_trims(cfg)
    vis = {n: c.debug_vis for n, c in _cameras(cfg).items()}
    assert vis == {"depth_camera_near": True, "depth_camera_far": True,
                   "fm_depth_camera_near": False, "fm_depth_camera_far": False}


def test_adding_the_pair_after_the_trims_finds_no_events_to_extend():
    """A trimmed play cfg has no DR terms left; extending must skip them, not crash."""
    cfg = _env_cfg()
    dcfg.apply_depth_play_trims(cfg)
    assert dcfg.install_fm_depth_group(cfg, *_FM_RES) == FM_DEPTH_OBS_GROUP


##
# The tasks that run a generator install the group themselves
##

@pytest.mark.parametrize("factory", ["tracker", "finetune"])
def test_a_task_with_the_generator_in_the_loop_installs_the_group(factory):
    """Nothing on those paths may depend on a CLI flag to give the generator both cameras.

    And decoration goes back ON there: the teacher's training scene renders none of it, while
    the generator reads terrain from the cameras and was recorded with it visible.
    """
    if factory == "tracker":
        from robot_rl.tasks.manager_based.robot_rl.g1.g1_tracker_env_cfg import (
            G1FmTrackerTerrainDepthEnvCfg as Cfg,
        )
    else:
        from robot_rl.tasks.manager_based.robot_rl.g1.g1_finetune_env_cfg import (
            G1FinetuneTerrainDepthEnvCfg as Cfg,
        )
    cfg = Cfg()
    dcfg.assert_fm_depth_group_covers_every_camera(cfg)
    assert _sensors(cfg, FM_DEPTH_OBS_GROUP) == ["depth_camera_near", "depth_camera_far"]
    assert _sensors(cfg, "policy") == ["depth_camera_near"]
    assert cfg.scene.terrain.render_decor is True


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
