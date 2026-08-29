# Copyright 2026 Zachary Olkin. All rights reserved.

"""The generator's own dense height scanner: a PRIVILEGED second sampling of the ground.

The dense scan exists to remove perception as a confound, so what these tests pin is mostly
about it staying exact and staying separate:

* the policy's scanner is untouched -- the depth teacher's critic observes it, so a changed
  width makes ``runner.load()`` reject the checkpoint at record time;
* the fold ``(H, W)`` agrees with **IsaacLab's own** ``grid_pattern``, not with our reading of
  it, so a library change to the ``ordering`` default fails here rather than silently
  transposing every image;
* nothing randomises or noises it;
* the no-op case, since "nothing is added when the grid already matches" is what keeps every
  reader free of a flag.

Runs without a simulator: the cfg classes are instantiated and the production contract
producer is run over them, exactly as ``height_scan_specs_from_cfg`` does.
"""

from __future__ import annotations

import copy

import pytest
import torch

from isaaclab.sensors import RayCasterCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import GridPatternCfg

import robot_rl.tasks.manager_based.robot_rl.g1.g1_depth_env_cfg as dcfg
import robot_rl.tasks.manager_based.robot_rl.g1.g1_fm_scan_env_cfg as scfg
from robot_rl.flow_matching.transforms import (
    scan_grid_counts,
    scan_grid_shape,
    scan_lateral_dim,
)
from robot_rl.tasks.manager_based.robot_rl.mdp.observations.height_scan import (
    FM_SCAN_SENSOR,
    RL_SCAN_SENSOR,
)


def _env_cfg():
    """A minimal stand-in carrying the two cfg objects ``add_fm_height_scan`` touches."""

    class _Cfg:
        def __init__(self):
            self.scene = dcfg.G1TeacherTerrainDepthSceneCfg(num_envs=4, env_spacing=2.5)
            self.events = dcfg.G1TeacherTerrainDepthEventsCfg()

    return _Cfg()


def _grid_of(scanner: RayCasterCfg) -> tuple:
    """A scanner's grid identity, for equality assertions."""
    pat = scanner.pattern_cfg
    return (float(pat.resolution), tuple(float(v) for v in pat.size), str(pat.ordering),
            tuple(float(v) for v in scanner.offset.pos))


##
# The fold, pinned against the library
##

@pytest.mark.parametrize("resolution, size, ordering", [
    (0.1, [1.5, 1.0], "xy"),          # today's 176-ray policy scan
    (0.025, [3.0, 3.0], "xy"),        # the shipped dense grid
    (0.05, [2.5, 1.6], "xy"),         # non-square, both orderings
    (0.05, [2.5, 1.6], "yx"),
])
def test_fold_matches_isaaclab_grid_pattern(resolution, size, ordering):
    """``scan_grid_shape`` must reproduce ``grid_pattern``'s actual flat ray order.

    The load-bearing test of the whole feature. ``num_rays`` cannot distinguish a grid from
    its transpose, and under a convolution a transposed grid is scrambled input with correct
    shapes and a falling loss. So assert against the LIBRARY: fold row-major and require each
    ray to land where the fold says it does. If IsaacLab ever changes the ``ordering`` default
    or the meshgrid indexing, this fails instead of the model quietly getting worse.
    """
    cfg = GridPatternCfg(resolution=resolution, size=tuple(size), ordering=ordering)
    starts, dirs = patterns.grid_pattern(cfg, "cpu")

    height, width = scan_grid_shape(resolution, size, ordering)
    n_x, n_y = scan_grid_counts(resolution, size)
    assert height * width == starts.shape[0]
    assert {height, width} == {n_x, n_y}

    # Fold the library's flat rays ROW-MAJOR using the returned (H, W) -- the same
    # `images_from_flat` reshape the encoder does -- and require the physical coordinates to
    # line up with the axes the fold claims. Using `height`/`width` here is the point: an
    # arithmetic-only check on the ray COUNT is transpose-blind, so the fold has to be
    # exercised, not merely multiplied out.
    x = torch.arange(-size[0] / 2, size[0] / 2 + 1e-9, resolution)
    y = torch.arange(-size[1] / 2, size[1] / 2 + 1e-9, resolution)
    folded = starts.reshape(height, width, 3)
    if ordering == "xy":
        # rows = lateral (y), columns = forward (x): x varies along a row, y down a column.
        want_x = x.view(1, -1).expand(height, width)
        want_y = y.view(-1, 1).expand(height, width)
    else:
        # rows = forward (x), columns = lateral (y).
        want_x = x.view(-1, 1).expand(height, width)
        want_y = y.view(1, -1).expand(height, width)
    assert torch.allclose(folded[..., 0], want_x, atol=1e-7), "forward axis is not where the fold says"
    assert torch.allclose(folded[..., 1], want_y, atol=1e-7), "lateral axis is not where the fold says"

    # And the axis a sagittal reflection flips must be the one that actually carries y:
    # y varies ALONG it and is constant across the other. `scan_lateral_dim` indexes a
    # (batch, H, W) tensor, so drop one for this batch-less fold.
    lat_axis = scan_lateral_dim(ordering) - 1
    other_axis = 1 - lat_axis
    assert folded[..., 1].std(dim=other_axis).max() < 1e-7, "y is not constant across the fold"
    assert folded[..., 1].diff(dim=lat_axis).abs().min() > 0, "y does not vary along lateral_dim"

    # The reading 'pos_z - hit_z - offset' only means ground clearance straight down.
    assert torch.allclose(dirs, torch.tensor([0.0, 0.0, -1.0]).expand_as(dirs))


def test_lateral_dim_follows_the_ordering():
    """The axis a sagittal reflection flips, derived from the same ``ordering`` as the fold."""
    assert scan_lateral_dim("xy") == 1      # (batch, n_y, n_x): rows are lateral
    assert scan_lateral_dim("yx") == 2      # (batch, n_x, n_y): columns are lateral


@pytest.mark.parametrize("ordering", ["", "XY", "xz", None])
def test_unknown_ordering_is_refused(ordering):
    """A guessed fold transposes the image, so an unrecognised ordering must raise."""
    with pytest.raises(ValueError, match="ordering"):
        scan_grid_shape(0.05, [1.0, 1.0], ordering)


def test_the_shipped_grid_lands_on_the_intended_token_count():
    """3.0 m @ 0.025 m must fold to 121x121, which the conv trunk takes to 16x16 = 256.

    The ray count and the token count are coupled through the stride-(1,2,2,2) trunk: 121 ->
    121 -> 61 -> 31 -> 16. Changing the grid without the strides silently changes the
    transformer's sequence length, so pin the arithmetic that ties them.
    """
    height, width = scan_grid_shape(scfg._FM_SCAN_RESOLUTION, scfg._FM_SCAN_SIZE, "xy")
    assert (height, width) == (121, 121)
    assert height * width == 14641

    grid_h, grid_w = height, width
    for stride in (1, 2, 2, 2):
        grid_h = -(-grid_h // stride)      # ceil-halving, as conv_grid_shape folds it
        grid_w = -(-grid_w // stride)
    assert (grid_h, grid_w) == (16, 16), "the trunk no longer lands on 256 tokens"


##
# The no-op case
##

def test_matching_grid_adds_nothing():
    """Asking for the policy's own grid must add no sensor and return ``None``.

    That is what lets every reader stay unscoped: with no ``fm_height_scanner`` present,
    ``fm_scan_sensor`` resolves to the policy's scanner, which is what the four historical
    hardcoded reads named.
    """
    cfg = _env_cfg()
    pat = getattr(cfg.scene, RL_SCAN_SENSOR).pattern_cfg
    added = scfg.add_fm_height_scan(
        cfg, size=tuple(pat.size), resolution=pat.resolution,
        offset_xy=tuple(getattr(cfg.scene, RL_SCAN_SENSOR).offset.pos[:2]))
    assert added is None
    assert getattr(cfg.scene, FM_SCAN_SENSOR, None) is None


def test_refuses_a_task_with_no_policy_scanner():
    """Without a scanner to derive from there is no shared mount/alignment/mesh to inherit."""
    cfg = _env_cfg()
    setattr(cfg.scene, RL_SCAN_SENSOR, None)
    with pytest.raises(ValueError, match=RL_SCAN_SENSOR):
        scfg.add_fm_height_scan(cfg)


def test_refuses_being_added_twice():
    """Reachable from record_data, play and a task __post_init__; two would be unresolvable."""
    cfg = _env_cfg()
    assert scfg.add_fm_height_scan(cfg) == FM_SCAN_SENSOR
    with pytest.raises(ValueError, match="already"):
        scfg.add_fm_height_scan(cfg)


##
# The runner.load guard
##

def test_the_policy_scanner_is_untouched():
    """The depth teacher's CRITIC observes the policy scanner.

    Widen it and the critic's observation width changes, which makes ``runner.load()`` reject
    the teacher checkpoint at record time -- ``PPORunnerCfg`` does not set
    ``load_critic = False``. This is the single constraint that forces a separate sensor, so
    assert the grid identity is byte-equal across the call.
    """
    cfg = _env_cfg()
    before = _grid_of(getattr(cfg.scene, RL_SCAN_SENSOR))
    scfg.add_fm_height_scan(cfg)
    assert _grid_of(getattr(cfg.scene, RL_SCAN_SENSOR)) == before


def test_the_two_scanners_share_their_physics():
    """Only the grid and the in-plane centre may differ; everything physical is inherited.

    Built with ``.replace()`` off the policy's cfg, so this is a property of construction
    rather than of two lists being kept in sync -- and that is what the test is really pinning.
    """
    cfg = _env_cfg()
    scfg.add_fm_height_scan(cfg)
    pol, fm = getattr(cfg.scene, RL_SCAN_SENSOR), getattr(cfg.scene, FM_SCAN_SENSOR)
    assert fm.prim_path == pol.prim_path
    assert fm.ray_alignment == pol.ray_alignment
    assert list(fm.mesh_prim_paths) == list(pol.mesh_prim_paths)
    assert fm.max_distance == pol.max_distance
    assert fm.drift_range == pol.drift_range
    assert fm.ray_cast_drift_range == pol.ray_cast_drift_range
    # The ray-start lift is inherited; only x/y move (the dense grid is robot-centred).
    assert fm.offset.pos[2] == pol.offset.pos[2]
    assert tuple(fm.offset.pos[:2]) == (0.0, 0.0)
    assert fm.pattern_cfg.resolution != pol.pattern_cfg.resolution


##
# Noise-free, and provably so
##

def test_the_dense_scanner_carries_no_drift():
    """Zero by inheritance, and the producer refuses a privileged sensor that has any."""
    cfg = _env_cfg()
    scfg.add_fm_height_scan(cfg)
    fm = getattr(cfg.scene, FM_SCAN_SENSOR)
    assert tuple(fm.drift_range) == (0.0, 0.0)
    assert all(tuple(b) == (0.0, 0.0) for b in (fm.ray_cast_drift_range or {}).values())


def test_privileged_sensor_with_drift_is_refused():
    """The tripwire: randomising the dense scan defeats the whole experiment."""
    cfg = _env_cfg()
    scfg.add_fm_height_scan(cfg)
    getattr(cfg.scene, FM_SCAN_SENSOR).drift_range = (-0.05, 0.05)
    with pytest.raises(ValueError, match="privileged"):
        scfg.height_scan_specs_from_cfg(scene=cfg.scene)


def test_a_scanner_dr_event_is_refused():
    """The inverse of ``install_fm_depth_group``: nothing may randomise either scanner.

    There is no such event today. This exists so that the day someone copies a
    camera-randomisation term onto a scanner, the dense-scan path refuses to be built rather
    than quietly reintroducing the perception error it removes.
    """
    from isaaclab.managers import EventTermCfg, SceneEntityCfg

    cfg = _env_cfg()
    cfg.events.jitter_the_scan = EventTermCfg(
        func=lambda *a, **k: None, mode="reset",
        params={"sensor_cfg": SceneEntityCfg(RL_SCAN_SENSOR)},
    )
    with pytest.raises(ValueError, match="randomises a height scanner"):
        scfg.add_fm_height_scan(cfg)


def test_debug_vis_is_off_and_refused_at_scale():
    """~350k marker instances at play env counts; the visualiser has no stride."""
    cfg = _env_cfg()
    scfg.add_fm_height_scan(cfg)
    assert getattr(cfg.scene, FM_SCAN_SENSOR).debug_vis is False
    with pytest.raises(ValueError, match="marker"):
        scfg.add_fm_height_scan(_env_cfg(), debug_vis=True)


def test_play_trims_do_not_enable_the_dense_scanner():
    """``-play`` cfgs set ``height_scanner.debug_vis = True`` unconditionally.

    They name the POLICY's scanner, so the dense one must stay dark. Order matters too: the
    trims run on an already-``__post_init__``'d cfg, before this sensor exists.
    """
    cfg = _env_cfg()
    scfg.add_fm_height_scan(cfg)
    dcfg.apply_depth_play_trims(cfg)
    assert getattr(cfg.scene, FM_SCAN_SENSOR).debug_vis is False


##
# The contract
##

def test_the_contract_states_the_derived_geometry():
    """Geometry, fold and ray count, all derived -- and the fold closes against the count."""
    specs = {s["sensor"]: s for s in scfg.height_scan_specs_from_cfg()}
    fm = specs[FM_SCAN_SENSOR]
    assert (fm["height"], fm["width"]) == (121, 121)
    assert fm["num_rays"] == fm["height"] * fm["width"] == 14641
    assert fm["resolution"] == 0.025 and fm["size"] == [3.0, 3.0]
    assert fm["ray_alignment"] == "yaw" and fm["direction"] == [0.0, 0.0, -1.0]
    assert fm["offset_pos"][:2] == [0.0, 0.0]
    assert fm["sensor_offset"] == 0.5      # read off mdp.height_scan's own signature
    assert fm["privileged"] is True
    # The policy's is described too, and is NOT privileged -- a deployable artifact may
    # reproduce that grid, and export_parameters relies on the distinction.
    assert specs[RL_SCAN_SENSOR]["privileged"] is False
    assert specs[RL_SCAN_SENSOR]["num_rays"] == 176


def test_the_grid_closes_for_a_non_square_size():
    """A wrong ``size`` or ``resolution`` must not survive: recompute and compare.

    This is the check ``extero_sensor_dim`` cannot make. A non-square grid also makes the
    transpose visible in the numbers, which a square one never does.
    """
    cfg = _env_cfg()
    scfg.add_fm_height_scan(cfg, size=(2.5, 1.6), resolution=0.05)
    spec = {s["sensor"]: s for s in
            scfg.height_scan_specs_from_cfg(scene=cfg.scene)}[FM_SCAN_SENSOR]
    assert (spec["height"], spec["width"]) == (33, 51)
    assert spec["num_rays"] == 1683
    assert (spec["num_y"], spec["num_x"]) == (33, 51)
    # Recomputing from the spec's own physical fields must reproduce its stated fold.
    assert scan_grid_shape(spec["resolution"], spec["size"], spec["ordering"]) == (
        spec["height"], spec["width"])


def test_a_non_yaw_alignment_is_refused():
    """The FM's window frame is xyz + yaw; only a yaw grid pins each ray to a fixed point."""
    cfg = _env_cfg()
    scfg.add_fm_height_scan(cfg)
    getattr(cfg.scene, FM_SCAN_SENSOR).ray_alignment = "base"
    with pytest.raises(ValueError, match="ray_alignment"):
        scfg.height_scan_specs_from_cfg(scene=cfg.scene)


def test_a_non_grid_pattern_is_refused():
    """There is no (H, W) fold for a non-grid pattern, so it cannot feed a convolution."""
    cfg = _env_cfg()
    scfg.add_fm_height_scan(cfg)
    fm = getattr(cfg.scene, FM_SCAN_SENSOR)
    fm.pattern_cfg = patterns.PinholeCameraPatternCfg()
    with pytest.raises(ValueError, match="GridPatternCfg"):
        scfg.height_scan_specs_from_cfg(scene=cfg.scene)


def test_sensor_resolution_prefers_the_dense_scanner():
    """``fm_scan_sensor`` picks the FM's when present and falls back to the policy's."""
    from types import SimpleNamespace

    from robot_rl.tasks.manager_based.robot_rl import mdp

    def _env(names):
        return SimpleNamespace(scene=SimpleNamespace(sensors={n: object() for n in names}))

    assert mdp.fm_scan_sensor(_env([RL_SCAN_SENSOR])) == RL_SCAN_SENSOR
    assert mdp.fm_scan_sensor(_env([RL_SCAN_SENSOR, FM_SCAN_SENSOR])) == FM_SCAN_SENSOR
    assert mdp.fm_scan_sensor(_env([])) is None
    # The deploy-facing resolver never returns the privileged one.
    assert mdp.rl_scan_sensor(_env([RL_SCAN_SENSOR, FM_SCAN_SENSOR])) == RL_SCAN_SENSOR
    assert mdp.rl_scan_sensor(_env([])) is None


def test_the_dense_scanner_does_not_touch_observations():
    """No obs term and no obs group: the readers read the SENSOR.

    This is what keeps the sagittal reflector, ``export_parameters``' policy-group lookup and
    the whole obs-group pipeline out of the dense-scan path -- and it means the policy's
    observation vector provably cannot change.
    """
    cfg = _env_cfg()
    cfg.observations = dcfg.G1TeacherTerrainDepthObservationsCfg()
    before = copy.deepcopy(vars(cfg.observations.policy))
    scfg.add_fm_height_scan(cfg)
    assert set(vars(cfg.observations.policy)) == set(before)
    assert not any("scan" in g for g in vars(cfg.observations) if g.startswith("fm_"))
