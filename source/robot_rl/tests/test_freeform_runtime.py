# Copyright 2026 Zachary Olkin. All rights reserved.

"""The CLIP-FREE runtime: spawn, goal-band truncation, and the critic's substitute datum.

The free-form terrain set ships as a manifest-only library, so the three clip-keyed pieces of
the fine-tune's runtime need replacements that read the manifest instead of a clip. This file
pins the parts that are pure tensor logic -- no simulator, no clip library -- with the same
stubbing idiom the rest of this directory uses (``monkeypatch`` makes ``wp.to_torch`` the
identity, so a stub articulation can hand back plain tensors).

The last test is the one that matters most: every flag added here is OFF by default, so the
clip tasks must reach exactly the code they reached before.
"""

from __future__ import annotations

import pytest
import torch
import warp as wp

from types import SimpleNamespace

from robot_rl.tasks.manager_based.robot_rl.mdp.events import resets
from robot_rl.tasks.manager_based.robot_rl.mdp.observations import observations as obs
from robot_rl.tasks.manager_based.robot_rl.mdp.terminations import terminations as term

#: One section, its start platform at the tile's own datum and its goal 6 m downrange.
_ROUTE = {
    "start_x": 1.0, "start_y": 2.0, "start_z": 0.0, "start_yaw": 0.0,
    "goal_x": 7.0, "goal_y": 2.0, "goal_half_width": 1.5,
}
#: A second section whose lowest platform is the tile datum but which STARTS a flight up, so
#: its start surface sits above it -- the descending-first case the z rule exists for.
_ROUTE_RAISED = dict(_ROUTE, start_z=1.2, goal_x=5.0)

_NOMINAL_BASE_Z = 0.785      # the G1's ``init_state`` base height
_N_JOINTS = 4


def _plain_warp(monkeypatch) -> None:
    """Make ``wp.to_torch`` the identity so stub buffers can be plain tensors."""
    monkeypatch.setattr(wp, "to_torch", lambda a: a)


class _StubImporter:
    """Minimal ``ClipTerrainImporter`` stand-in: the env->tile map plus the route dicts."""

    def __init__(self, terrain_id: torch.Tensor, corner: torch.Tensor,
                 frame_yaw: torch.Tensor, routes: dict[int, dict]) -> None:
        """Store the per-env tile arrays and the per-terrain-id routes."""
        self.env_terrain_id = terrain_id
        self.env_cell_corner = corner
        self.env_terrain_yaw = frame_yaw
        self.terrain_route = routes


class _StubAsset:
    """Articulation stand-in that records what was written to the sim."""

    def __init__(self, n: int) -> None:
        """Default root pose at the nominal standing height, defaults for the joints."""
        pose = torch.zeros(n, 7)
        pose[:, 2] = _NOMINAL_BASE_Z
        pose[:, 3] = 1.0                                   # xyzw identity
        self.data = SimpleNamespace(
            default_root_pose=pose,
            default_joint_pos=torch.arange(n * _N_JOINTS, dtype=torch.float32).view(n, _N_JOINTS),
            default_joint_vel=torch.zeros(n, _N_JOINTS),
            root_pos_w=torch.zeros(n, 3),
            root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(n, 1),
        )
        self.written: dict[str, torch.Tensor] = {}

    def write_root_pose_to_sim_index(self, root_pose, env_ids):
        """Record the spawn pose."""
        self.written["root_pose"] = root_pose.clone()

    def write_root_link_velocity_to_sim_index(self, root_velocity, env_ids):
        """Record the spawn root velocity."""
        self.written["root_vel"] = root_velocity.clone()

    def write_joint_position_to_sim_index(self, position, env_ids):
        """Record the spawn joint positions."""
        self.written["joint_pos"] = position.clone()

    def write_joint_velocity_to_sim_index(self, velocity, env_ids):
        """Record the spawn joint velocities."""
        self.written["joint_vel"] = velocity.clone()


class _StubEnv:
    """``ManagerBasedEnv`` stand-in holding one robot and one terrain."""

    def __init__(self, importer: _StubImporter, asset: _StubAsset) -> None:
        """Wire the scene lookup the reset and the termination both use."""
        self.device = "cpu"
        self.num_envs = importer.env_terrain_id.shape[0]
        self.scene = _SceneShim(importer, asset)


class _SceneShim:
    """``env.scene`` with both attribute access (``terrain``) and ``scene["robot"]``."""

    def __init__(self, terrain: _StubImporter, asset: _StubAsset) -> None:
        """Hold the terrain and the one asset."""
        self.terrain = terrain
        self._asset = asset

    def __getitem__(self, name: str) -> _StubAsset:
        """Every asset name resolves to the one stub robot."""
        return self._asset


def _env(n: int = 4, routes: dict[int, dict] | None = None,
         frame_yaw: float = 0.0, corner=(10.0, -5.0, 0.0)) -> tuple[_StubEnv, _StubAsset]:
    """An ``n``-env scene, every env on terrain 0 unless ``routes`` says otherwise."""
    routes = routes if routes is not None else {0: dict(_ROUTE)}
    tid = torch.zeros(n, dtype=torch.long)
    if len(routes) > 1:
        tid = torch.arange(n, dtype=torch.long) % len(routes)
    importer = _StubImporter(
        tid,
        torch.tensor([list(corner)], dtype=torch.float32).repeat(n, 1),
        torch.full((n,), float(frame_yaw)),
        routes,
    )
    asset = _StubAsset(n)
    return _StubEnv(importer, asset), asset


# ----------------------------------------------------------------------
# The goal band
# ----------------------------------------------------------------------

def _place(env: _StubEnv, asset: _StubAsset, local_xy: list[tuple[float, float]]) -> None:
    """Put each robot at a tile-LOCAL (x, y), so a test states section coordinates."""
    corner = env.scene.terrain.env_cell_corner
    yaw = env.scene.terrain.env_terrain_yaw
    local = torch.tensor(local_xy, dtype=torch.float32)
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    pos = corner.clone()
    pos[:, 0] += cos_y * local[:, 0] - sin_y * local[:, 1]
    pos[:, 1] += sin_y * local[:, 0] + cos_y * local[:, 1]
    asset.data.root_pos_w = pos


def test_the_goal_band_spans_the_section_width(monkeypatch):
    """Anywhere across the band counts: nothing here holds a robot on the centreline."""
    _plain_warp(monkeypatch)
    env, asset = _env(n=4)
    # Goal line at x=7, centre y=2, declared half-width 1.5 -> floored to 2.5 by min_half_width.
    _place(env, asset, [(7.5, 2.0), (7.5, 4.4), (7.5, -0.4), (7.5, 5.0)])
    done = term.freeform_goal_reached(env, min_half_width=2.5)
    assert done.tolist() == [True, True, True, False]


def test_the_goal_band_does_not_fire_before_the_line(monkeypatch):
    """Downrange is a half-plane at the goal line, so nothing short of it counts."""
    _plain_warp(monkeypatch)
    env, asset = _env(n=3)
    _place(env, asset, [(0.0, 2.0), (6.99, 2.0), (7.0, 2.0)])
    assert term.freeform_goal_reached(env).tolist() == [False, False, True]


def test_the_goal_band_has_no_far_edge(monkeypatch):
    """Unlike mjeval's 2 m-deep rectangle: the first step past the line ends the episode."""
    _plain_warp(monkeypatch)
    env, asset = _env(n=2)
    _place(env, asset, [(7.0, 2.0), (50.0, 2.0)])
    assert term.freeform_goal_reached(env).tolist() == [True, True]


def test_the_goal_band_is_in_the_tile_frame(monkeypatch):
    """A rotated tile frame rotates the band with it, rather than measuring world x."""
    _plain_warp(monkeypatch)
    env, asset = _env(n=2, frame_yaw=torch.pi / 2)
    # Local (7.5, 2) and local (0, 2): the first is past the goal, the second is at the start.
    _place(env, asset, [(7.5, 2.0), (0.0, 2.0)])
    assert term.freeform_goal_reached(env).tolist() == [True, False]
    # ...and the placement really did move in world y, not world x (yaw = 90 deg).
    assert asset.data.root_pos_w[0, 1].item() == pytest.approx(-5.0 + 7.5, abs=1e-5)


def test_the_declared_half_width_wins_when_it_is_wider(monkeypatch):
    """``min_half_width`` is a FLOOR, not the band: a wide section keeps its own width."""
    _plain_warp(monkeypatch)
    env, asset = _env(n=2, routes={0: dict(_ROUTE, goal_half_width=4.0)})
    _place(env, asset, [(7.5, 5.9), (7.5, 6.1)])
    assert term.freeform_goal_reached(env, min_half_width=2.5).tolist() == [True, False]


# ----------------------------------------------------------------------
# The spawn
# ----------------------------------------------------------------------

def test_the_spawn_sits_on_the_declared_surface_plus_the_z_offset(monkeypatch):
    """Base z = tile ground + the section's start surface + standing height + the offset.

    The 0.005 m is not cosmetic: with the soles exactly on the surface PhysX ejects the robot
    and every episode opens with a kick.
    """
    _plain_warp(monkeypatch)
    env, asset = _env(n=2, routes={0: dict(_ROUTE), 1: dict(_ROUTE_RAISED)},
                      corner=(10.0, -5.0, 0.25))
    resets.reset_on_freeform_section(
        env, torch.arange(2), spawn_z_offset=0.005, pos_jitter_m=(0.0, 0.0), yaw_jitter_rad=0.0,
    )
    z = asset.written["root_pose"][:, 2]
    assert z[0].item() == pytest.approx(0.25 + 0.0 + _NOMINAL_BASE_Z + 0.005, abs=1e-6)
    assert z[1].item() == pytest.approx(0.25 + 1.2 + _NOMINAL_BASE_Z + 0.005, abs=1e-6)


def test_the_spawn_lands_on_the_declared_start_pose(monkeypatch):
    """With the jitter off, the robot is exactly at ``(start_x, start_y)`` on its tile."""
    _plain_warp(monkeypatch)
    env, asset = _env(n=2)
    resets.reset_on_freeform_section(
        env, torch.arange(2), pos_jitter_m=(0.0, 0.0), yaw_jitter_rad=0.0,
    )
    pose = asset.written["root_pose"]
    assert pose[:, 0].tolist() == pytest.approx([11.0, 11.0])     # corner.x + start_x
    assert pose[:, 1].tolist() == pytest.approx([-3.0, -3.0])     # corner.y + start_y
    # Identity heading (frame_yaw 0, start_yaw 0, no jitter): xyzw.
    assert pose[0, 3:7].tolist() == pytest.approx([0.0, 0.0, 0.0, 1.0], abs=1e-6)


def test_the_spawn_jitter_stays_inside_its_bounds(monkeypatch):
    """Uniform in the TILE frame, and never wider than asked -- the box must fit the platform."""
    _plain_warp(monkeypatch)
    torch.manual_seed(0)
    n = 512
    env, asset = _env(n=n)
    resets.reset_on_freeform_section(
        env, torch.arange(n), pos_jitter_m=(0.25, 0.4), yaw_jitter_rad=0.2,
    )
    pose = asset.written["root_pose"]
    dx = pose[:, 0] - 11.0
    dy = pose[:, 1] - (-3.0)
    assert dx.abs().max().item() <= 0.25 + 1e-6
    assert dy.abs().max().item() <= 0.4 + 1e-6
    # Yaw about z only: the xyzw quaternion's x and y components stay zero.
    assert pose[:, 3:5].abs().max().item() == pytest.approx(0.0, abs=1e-6)
    yaw = 2.0 * torch.atan2(pose[:, 5], pose[:, 6])
    assert yaw.abs().max().item() <= 0.2 + 1e-6
    # ...and the set is not degenerate: a spawn distribution that collapsed would give the
    # generator one image per tile, which is the thing the jitter exists to prevent.
    assert dx.std().item() > 0.05 and yaw.std().item() > 0.02


def test_the_spawn_writes_the_default_joint_state_and_zero_velocity(monkeypatch):
    """No clip means no reference pose to borrow: the robot spawns standing, at rest."""
    _plain_warp(monkeypatch)
    env, asset = _env(n=3)
    resets.reset_on_freeform_section(env, torch.arange(3))
    assert torch.equal(asset.written["joint_pos"], asset.data.default_joint_pos)
    assert torch.equal(asset.written["joint_vel"], asset.data.default_joint_vel)
    assert asset.written["root_vel"].abs().max().item() == 0.0


def test_the_spawn_honours_the_tile_frame_yaw(monkeypatch):
    """``start_yaw`` is relative to the tile frame, so the two compose."""
    _plain_warp(monkeypatch)
    env, asset = _env(n=1, routes={0: dict(_ROUTE, start_yaw=0.5)}, frame_yaw=1.0)
    resets.reset_on_freeform_section(
        env, torch.arange(1), pos_jitter_m=(0.0, 0.0), yaw_jitter_rad=0.0,
    )
    q = asset.written["root_pose"][0, 3:7]
    assert (2.0 * torch.atan2(q[2], q[3])).item() == pytest.approx(1.5, abs=1e-5)


def test_an_empty_reset_is_a_no_op(monkeypatch):
    """A step where nothing terminated must not touch the sim."""
    _plain_warp(monkeypatch)
    env, asset = _env(n=2)
    resets.reset_on_freeform_section(env, torch.zeros(0, dtype=torch.long))
    assert asset.written == {}


# ----------------------------------------------------------------------
# The manifest contract
# ----------------------------------------------------------------------

def test_a_terrain_without_routes_is_refused():
    """A clip library reaching the free-form runtime fails by name, not by NaN spawn."""
    env, _ = _env(n=1)
    env.scene.terrain.terrain_route = {}
    with pytest.raises(TypeError, match="terrain_route"):
        resets.freeform_route(env.scene.terrain, torch.arange(1))


def test_a_route_missing_a_field_is_refused():
    """Every field is required: a partial route would spawn or truncate on a NaN."""
    env, _ = _env(n=1, routes={0: {"start_x": 0.0}})
    with pytest.raises(ValueError, match="missing"):
        resets.freeform_route(env.scene.terrain, torch.arange(1))


def test_a_tile_with_no_route_is_refused():
    """A tile the mesh placed but the route table does not cover cannot be spawned on."""
    env, _ = _env(n=2, routes={0: dict(_ROUTE)})
    env.scene.terrain.env_terrain_id = torch.tensor([0, 3])
    with pytest.raises(ValueError, match="NO route"):
        resets.freeform_route(env.scene.terrain, torch.arange(2))


def test_a_goal_behind_the_start_is_refused():
    """Otherwise every episode truncates on step 0 and reads as a policy that never walks."""
    env, _ = _env(n=1, routes={0: dict(_ROUTE, goal_x=0.5)})
    with pytest.raises(ValueError, match="BEHIND"):
        resets.freeform_route(env.scene.terrain, torch.arange(1))


def test_the_route_table_is_built_once():
    """The goal test reads it every step, so the validation cannot be per-step work."""
    env, _ = _env(n=2)
    resets.freeform_route(env.scene.terrain, torch.arange(2))
    # Corrupting the source dict after the first build must not change the answer.
    env.scene.terrain.terrain_route[0]["goal_x"] = -99.0
    route = resets.freeform_route(env.scene.terrain, torch.arange(2))
    assert route["goal_x"].tolist() == pytest.approx([7.0, 7.0])


# ----------------------------------------------------------------------
# The critic's substitute datum
# ----------------------------------------------------------------------

class _StubCmd:
    """``FmTrackingCommand`` stand-in: a plan anchor, a robot, and a clip datum."""

    def __init__(self, n: int, window_anchor: str = "plan", fm_enabled: bool = True) -> None:
        """Anchor the plan 1 m ahead of a robot sitting at the origin."""
        self.cfg = SimpleNamespace(window_anchor=window_anchor)
        self.fm_enabled = fm_enabled
        quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(n, 1)
        self.robot = SimpleNamespace(data=SimpleNamespace(
            root_pos_w=torch.zeros(n, 3), root_quat_w=quat.clone(),
        ))
        self._anchor_pos = torch.tensor([[1.0, 0.0, 0.0]]).repeat(n, 1)
        self._anchor_quat = quat.clone()
        self._clip_state = torch.cat([torch.full((n, 3), 5.0), quat.clone()], dim=-1)

    def window_anchor(self):
        """The plan anchor, as the real command serves it under ``window_anchor='plan'``."""
        return self._anchor_pos, self._anchor_quat

    def termination_reference_world_state(self):
        """The globally-placed clip pose, as the clip datum reads it."""
        return self._clip_state


def _obs_env(cmd: _StubCmd) -> SimpleNamespace:
    """An env whose command manager serves ``cmd``."""
    return SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda name: cmd),
    )


def test_the_anchor_drift_has_the_width_of_the_term_it_replaces(monkeypatch):
    """6 wide, both arms -- what keeps a resumed teacher loading with no checkpoint surgery."""
    _plain_warp(monkeypatch)
    cmd = _StubCmd(n=5)
    env = _obs_env(cmd)
    clip = obs.tracking_error_global(env, datum="clip")
    plan = obs.tracking_error_global(env, datum="plan_anchor")
    assert clip.shape == (5, 6)
    assert plan.shape == clip.shape


def test_the_anchor_drift_is_reference_minus_actual(monkeypatch):
    """Same convention as the clip arm, so the critic reads one sign everywhere."""
    _plain_warp(monkeypatch)
    cmd = _StubCmd(n=2)
    cmd.robot.data.root_pos_w = torch.tensor([[0.25, 0.0, 0.0], [1.5, 0.0, 0.0]])
    out = obs.tracking_error_global(_obs_env(cmd), datum="plan_anchor")
    assert out[:, 0].tolist() == pytest.approx([0.75, -0.5])
    assert out[:, 3:].abs().max().item() == pytest.approx(0.0, abs=1e-6)


def test_the_anchor_drift_is_not_the_clip_datum(monkeypatch):
    """The two answer different questions; a task must not get one thinking it asked for the
    other."""
    _plain_warp(monkeypatch)
    cmd = _StubCmd(n=2)
    env = _obs_env(cmd)
    assert not torch.allclose(obs.tracking_error_global(env, datum="clip"),
                              obs.tracking_error_global(env, datum="plan_anchor"))


@pytest.mark.parametrize("anchor,fm", [("robot_now", True), ("plan", False)])
def test_the_anchor_drift_refuses_a_command_whose_anchor_is_the_robot(monkeypatch, anchor, fm):
    """Under ``robot_now`` (or with no generator) the drift is identically zero.

    Silently feeding the critic a zero column is exactly the zero-fill this term exists to
    avoid, so it is an error rather than a degenerate answer.
    """
    _plain_warp(monkeypatch)
    cmd = _StubCmd(n=2, window_anchor=anchor, fm_enabled=fm)
    with pytest.raises(ValueError, match="plan_anchor"):
        obs.tracking_error_global(_obs_env(cmd), datum="plan_anchor")


def test_an_unknown_datum_is_refused(monkeypatch):
    """A typo must not silently fall through to the clip branch."""
    _plain_warp(monkeypatch)
    with pytest.raises(ValueError, match="datum"):
        obs.tracking_error_global(_obs_env(_StubCmd(n=1)), datum="anchor")


# ----------------------------------------------------------------------
# The clip tasks are untouched
# ----------------------------------------------------------------------

def test_every_freeform_switch_is_off_by_default(monkeypatch):
    """The clip runtime must reach exactly the code it reached before.

    Not a style check: all three additions are gated, and the gate is the whole reason a clip
    teacher and the free-form fine-tune can share these modules.
    """
    import inspect

    # The critic term defaults to the clip datum, and takes that path with no terrain at all.
    assert inspect.signature(obs.tracking_error_global).parameters["datum"].default == "clip"
    _plain_warp(monkeypatch)
    cmd = _StubCmd(n=2)
    assert torch.equal(obs.tracking_error_global(_obs_env(cmd)),
                       obs.tracking_error_global(_obs_env(cmd), datum="clip"))

    # The spawn and the goal band are NEW functions, so nothing a clip task wires can reach
    # them: the clip reset and clip_end are untouched, and neither gained a parameter.
    clip_reset = inspect.signature(resets.reset_on_terrain_motion_sequence).parameters
    assert set(clip_reset) == {
        "env", "env_ids", "command_name", "base_frame_name", "asset_cfg", "pin_clip_ids",
        "pin_start_step", "spawn_z_offset", "clip_draw",
    }
    assert set(inspect.signature(term.clip_end_reached).parameters) == {
        "env", "command_name", "reserve_lookahead",
    }
