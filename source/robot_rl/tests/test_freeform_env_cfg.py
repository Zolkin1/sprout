# Copyright 2026 Zachary Olkin. All rights reserved.

"""The free-form fine-tune task, and the seam where three independent halves have to agree.

Two different things are pinned here.

**The TASK.** ``G1-teacher-clf-terrain-depth-finetune-freeform`` removes the clip from all
four of the roles it still played in the clip fine-tune -- terrain aside -- and each removal
is a separate switch on a separate manager. Any one of them left at the clip's value is
silent: the env still builds, still trains, and either reads a clip that is not there or
grades against a datum that means nothing. So each is asserted, and the CLIP fine-tune is
asserted unchanged in the same four respects beside it, because the whole design rests on
these being two configurations of one task rather than a fork.

**The SEAM.** The section route is authored by an IsaacLab-free generator
(``motion_gen.terrain.freeform``), carried by a manifest, copied through
``ClipTerrainLayout`` / ``ClipTerrainImporter``, and read by two consumers that were written
against it independently -- the spawn/goal runtime and the steering command. Three of those
four hops are in different packages. A field renamed at one end is an ``AttributeError`` at
best and a route silently absent at worst (``terrain_route`` empty reads as "this is a clip
library"), so the round trip is exercised on a REAL generated manifest rather than a stub.
"""

from __future__ import annotations

import importlib.util
import math
from types import SimpleNamespace

import pytest

pytest.importorskip("isaaclab", reason="the env configs import IsaacLab")

import gymnasium as gym                                                      # noqa: E402
import torch                                                                 # noqa: E402

from motion_gen.features import _REPO_ROOT                                   # noqa: E402
from motion_gen.terrain import freeform as ff                                # noqa: E402

import robot_rl.tasks  # noqa: F401,E402  - registers every G1 task
from robot_rl.tasks.manager_based.robot_rl.g1.g1_finetune_env_cfg import (   # noqa: E402
    FREEFORM_GOAL_MIN_HALF_WIDTH_M,
    G1FinetuneFreeformEnvCfg,
    G1FinetuneTerrainDepthEnvCfg,
)
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.fm_tracking_cmd import (  # noqa: E402,E501
    FmTrackingCommand,
)
from robot_rl.tasks.manager_based.robot_rl.mdp.events import resets          # noqa: E402
from robot_rl.tasks.manager_based.robot_rl.terrains.clip_terrain_loader import (  # noqa: E402
    ClipTerrainLayout,
    assemble_clip_terrain,
    default_num_flat_tiles,
)

TASK_ID = "G1-teacher-clf-terrain-depth-finetune-freeform"
CLIP_TASK_ID = "G1-teacher-clf-terrain-depth-finetune"


@pytest.fixture(scope="module")
def cfgs():
    """The free-form task's config and the clip fine-tune it derives from."""
    return G1FinetuneFreeformEnvCfg(), G1FinetuneTerrainDepthEnvCfg()


def _active(group) -> list[str]:
    """Names of a manager config's live terms, in declaration order.

    Args:
        group: A configclass instance whose attributes are term configs.

    Returns:
        The names whose term is not ``None``.
    """
    return [k for k, v in vars(group).items()
            if not k.startswith("_") and v is not None and hasattr(v, "func")]


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------

def test_the_task_is_registered_against_the_freeform_cfg_and_the_finetune_runner():
    """Both entry points, because a wrong runner is as silent as a wrong env.

    ``FinetunePPORunnerCfg`` is what makes a RESUME a fine-tune rather than a fresh run; with
    the plain ``PPORunnerCfg`` the task would train from scratch on a terrain set that exists
    only to fine-tune on.
    """
    kwargs = gym.registry[TASK_ID].kwargs
    assert kwargs["env_cfg_entry_point"].endswith(":G1FinetuneFreeformEnvCfg")
    assert kwargs["rsl_rl_cfg_entry_point"].endswith(":FinetunePPORunnerCfg")


#: Every task allowed to manufacture its own velocity command, and each is a FREE-FORM task
#: whose library carries no clip to take one from. Named rather than counted: the guard is
#: "no CLIP task turned this on", and a bare count would pass the day someone adds a
#: free-form task and silently flips a clip task at the same time.
_FREEFORM_COMMAND_TASKS: tuple[str, ...] = (
    "G1-teacher-clf-terrain-depth-finetune-freeform",
    # The same fine-tune pinned to the fixed two-box course, with the policy's far camera
    # restored so an existing two-camera tracker can run it.
    "G1-box2-finetune",
)


def test_only_the_freeform_tasks_manufacture_their_own_command():
    """Enumerated over the REGISTRY, so a task added later cannot slip past this.

    The free-form command replaces every command-velocity accessor. On a clip task that would
    silently discard the library's ``target_vel`` -- the joystick track the FM is conditioned
    on -- and nothing downstream would report it, because the widths are identical.
    """
    turned_on = []
    for task_id in [t for t in gym.registry if t.startswith("G1-")]:
        entry = gym.registry[task_id].kwargs.get("env_cfg_entry_point")
        if not isinstance(entry, str):
            continue
        module_name, _, cls_name = entry.partition(":")
        cls = getattr(importlib.import_module(module_name), cls_name)
        traj_ref = getattr(cls().commands, "traj_ref", None)
        if getattr(traj_ref, "freeform_command", False):
            turned_on.append(task_id)
    assert sorted(turned_on) == sorted(_FREEFORM_COMMAND_TASKS)


# ----------------------------------------------------------------------
# The task: the clip is gone from all four of its non-terrain roles
# ----------------------------------------------------------------------

def test_the_generator_is_still_the_graded_reference(cfgs):
    """The one field the whole fine-tune rests on survives the free-form variant.

    Free-form terrain only makes sense BECAUSE the reference is predicted from the cameras.
    Back at ``reference_source="clip"`` this task would be tracking nothing at all.
    """
    freeform, _ = cfgs
    assert freeform.commands.traj_ref.reference_source == "fm"
    assert freeform.commands.traj_ref.history_from == "generator"


def test_the_velocity_command_is_manufactured_and_the_clip_task_still_reads_the_library(cfgs):
    """Role 1 of 4. A manifest-only library carries no ``target_vel`` to read."""
    freeform, clip = cfgs
    assert freeform.commands.traj_ref.freeform_command is True
    assert clip.commands.traj_ref.freeform_command is False


def test_the_spawn_comes_off_the_manifest_and_the_clip_reset_is_gone(cfgs):
    """Role 2 of 4.

    Disabled rather than re-pointed: both clip resets SAMPLE a clip, which raises the moment a
    tile's terrain id has no clips -- correct for a clip library, fatal on this one. A reset
    term left in place beside the new one would fail at the first episode.
    """
    freeform, clip = cfgs
    assert freeform.events.reset_on_clip is None
    assert freeform.events.reset_on_section.func is resets.reset_on_freeform_section
    assert freeform.events.reset_on_section.mode == "reset"
    assert clip.events.reset_on_clip.func is resets.reset_on_terrain_motion_sequence
    assert not hasattr(clip.events, "reset_on_section")


def test_the_episode_is_bounded_by_the_goal_band_not_by_a_clip_running_out(cfgs):
    """Role 3 of 4, and the goal is a TRUNCATION.

    ``time_out=True`` bootstraps the value from the final state, so walking the section is
    neither punished nor paid and the return does not become a function of section LENGTH
    (10-38 m across this terrain set). The THREE failure terms are inherited untouched -- none
    of them ever referred to a clip.

    `base_upright` is the third: a slack 70 deg world-upright backstop covering the blind spot
    in the absolute height term, which cannot see a collapse on RAISED ground (the
    up-into-down platform is 0.90 m up, a box top 0.40). It is pinned here because the
    termination SET is what decides how an episode can end, and a term silently added or
    dropped changes every outcome table without changing any number's shape.

    `tracking_violation` is the FOURTH failure term, added 2026-09-09: the strict tracking
    bounds around the served node (lateral 0.20 / height 0.12 / tilt 30 deg). It is the one
    term here that is not a fall detector -- the height bound separates a correct box dismount
    from a stairs-down-off-a-box one -- and it exists as a TERMINATION rather than the offline
    latch it used to be because the latch made the pipeline incoherent: the run reached the
    goal and banked the progress reward's success bonus, then every consumer cut the episode at
    the latch and deleted the row that bonus was paid on, on 95.3% of arriving episodes.

    It is deliberately NOT `time_out`: a violation is a failure, and marking it a truncation
    would make ``rollout_outcomes.terminal_reward`` score it 0 rather than -1.
    """
    freeform, clip = cfgs
    assert freeform.terminations.clip_end is None
    assert freeform.terminations.goal_reached.time_out is True
    assert _active(freeform.terminations) == [
        "time_out", "base_height", "base_tilt_error", "base_upright", "tracking_violation",
        "goal_reached",
    ]
    assert freeform.terminations.tracking_violation.time_out is not True
    # The CLIP fine-tune is untouched: its reference is a placed clip, so a bound built for the
    # generator's served node has no meaning there and adding it would change a task this
    # decision was never about.
    assert _active(clip.terminations) == [
        "time_out", "clip_end", "base_height", "base_tilt_error", "base_upright",
    ]


def test_the_critic_measures_plan_anchor_drift_and_keeps_its_term_list(cfgs):
    """Role 4 of 4, and the reason it is a PARAMETER rather than a second term.

    The free-form cfg overrides one ``params`` entry, so the critic group's term list --
    names and order -- is identical to the clip fine-tune's and a resumed teacher checkpoint
    loads with no surgery. The clip branch is not merely unavailable here but wrong:
    ``termination_reference_world_state`` keeps indexing a clip by step under
    ``reference_source="fm"``, so it marches away from the robot on its own.
    """
    freeform, clip = cfgs
    assert freeform.observations.critic.tracking_error_global.params["datum"] == "plan_anchor"
    assert clip.observations.critic.tracking_error_global.params.get("datum", "clip") == "clip"
    assert _active(freeform.observations.critic) == _active(clip.observations.critic)


def test_the_goal_band_floor_is_the_generators_own_and_not_the_functions_default(cfgs):
    """The one number two lanes disagreed about, resolved toward the manifest.

    ``mdp.freeform_goal_reached`` defaults its floor to mjeval's 2.5 m, for mjeval's reason:
    nothing in that stack steers, so a band narrower than the accumulated heading error
    measures the absence of a controller. Here the steering controller runs and the generator
    already floors every declared band at 0.75 m, so passing that same number makes the clamp
    a NO-OP. At 2.5 m a 1.219 m staircase would be scored through a 5 m-wide gate, which a
    robot that fell off the side and drifted past the goal line can pass.
    """
    freeform, _ = cfgs
    assert FREEFORM_GOAL_MIN_HALF_WIDTH_M == ff.GOAL_HALF_WIDTH_FLOOR_M
    assert freeform.terminations.goal_reached.params["min_half_width"] == ff.GOAL_HALF_WIDTH_FLOOR_M
    assert freeform.terminations.goal_reached.params["min_half_width"] < 2.5


def test_the_wall_collision_penalty_is_quoted_at_its_EFFECTIVE_weight(cfgs):
    """The free-form stairwell walls COLLIDE, and two docstrings quote what that costs.

    Both ``motion_gen.terrain.freeform.make_stairs_tile`` (where the walls are authored) and
    ``G1FinetuneFreeformEnvCfg`` (where they are rendered) argue "nothing in the MDP teaches
    obstacle avoidance" from the size of ``undesired_contacts`` against ``clf_reward``. The
    weight they must quote is the one this cfg CARRIES, and it is not the one declared beside
    the term: ``g1_clf_tracking_base`` declares -0.1 and ``G1TeacherEnvCfg.__post_init__``
    overwrites it with -0.5, which every terrain cfg down to this one inherits.

    Pinned because the number was already corrected the WRONG way once -- from the effective
    -0.5 to the declared -0.1, on the strength of reading the declaration -- which understates
    the penalty 5x and makes that argument look stronger than it is. A ratio quoted in prose
    has no other guard.
    """
    freeform, clip = cfgs
    assert freeform.rewards.undesired_contacts.weight == -0.5
    assert freeform.rewards.clf_reward.weight == 10.0
    # ...and the declaration it is NOT, so a future reader who finds -0.1 knows where it lives.
    from robot_rl.tasks.manager_based.robot_rl.g1.g1_clf_tracking_base import (
        G1ClfTrackingEnvCfg,
    )
    assert G1ClfTrackingEnvCfg().rewards.undesired_contacts.weight == -0.1
    # The clip fine-tune is the same robot in the same world; the walls are the only difference.
    assert clip.rewards.undesired_contacts.weight == freeform.rewards.undesired_contacts.weight


def test_every_clip_phase_knob_is_off(cfgs):
    """None of these can mean anything without clips, and each would fail differently.

    ``seed_history_from_clip`` is the one that matters: its mechanism is reading the clip's
    steps BEFORE the episode start, so on this library it has nothing to read. Every free-form
    episode therefore opens on the learned null-history embedding, which is the deploy
    condition -- and is why the sections are authored with a flat approach apron rather than
    starting on the terrain.
    """
    freeform, clip = cfgs
    cmd = freeform.commands.traj_ref
    assert cmd.seed_history_from_clip == "never"
    assert cmd.random_start_phase is False
    assert cmd.random_start_before_terrain_start is False
    assert cmd.random_start_min_remaining_s == 0.0
    # The clip fine-tune keeps all four, which is what makes this a variant and not a fork.
    assert clip.commands.traj_ref.seed_history_from_clip == "on_terrain_start"
    assert clip.commands.traj_ref.random_start_phase is True


def test_the_generator_still_sees_both_cameras_in_a_decorated_world(cfgs):
    """Neither of the two things running a GENERATOR requires may be lost in the subclass.

    The policy reads the near camera alone, so without the installed group the generator would
    condition on half the robot's cameras -- silently, with every artifact self-consistent
    about it. And the teacher's own training scene renders no decoration while the recordings
    the generator was fitted on did, so a vertical surface it has never seen is what it
    misreads as a box.
    """
    freeform, _ = cfgs
    assert hasattr(freeform.observations, "fm_depth")
    assert freeform.scene.terrain.render_decor is True
    # Both of the shipped cameras, not just the policy's.
    assert len(_active(freeform.observations.fm_depth)) == 2


def test_no_default_library_path_is_baked_in(cfgs):
    """``--clip`` states the terrain, here as everywhere: a stale default is undetectable."""
    freeform, _ = cfgs
    assert freeform.commands.traj_ref.path == ""
    assert freeform.commands.traj_ref.fm_checkpoint == ""


# ----------------------------------------------------------------------
# The seam: generator -> manifest -> layout -> importer -> runtime
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def freeform_library(tmp_path_factory):
    """A small real free-form library on disk, plus the manifest it was written from.

    Generated rather than fixtured: the point of these tests is that the generator's output
    and the RL side's reader agree TODAY, and a checked-in fixture would freeze one half.
    """
    maker_path = _REPO_ROOT / "scripts" / "rsl_rl" / "make_freeform_terrain.py"
    spec = importlib.util.spec_from_file_location("make_freeform_terrain", maker_path)
    maker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(maker)

    cfg = ff.FreeformConfig(seed=3, num_flat=2, num_boxes=2, num_stairs=2)
    manifest = ff.generate_freeform_manifest(cfg)
    ff.validate_manifest(manifest)
    path = str(tmp_path_factory.mktemp("freeform") / "lib.h5")
    # The bare SCHEMA -- see test_clip_free_library. Nothing here resolves a joint
    # output, so the layout `library_meta` refuses is exactly the right one to use.
    maker.write_terrain_library(path, manifest, maker.schema_only_layout(), cfg)
    return path, manifest


def test_the_route_field_names_are_one_set_shared_by_all_three_lanes():
    """The generator writes what the RL side reads, by identity rather than by convention.

    ``motion_gen`` cannot import ``robot_rl`` (it must stay IsaacLab-free), so the two tuples
    are declared separately and this is what holds them equal. The steering command consumes
    the SAME dicts through ``resets.freeform_route``, so all three lanes move together or not
    at all.
    """
    assert tuple(ff.ROUTE_FIELDS) == tuple(resets.FREEFORM_ROUTE_FIELDS)


def test_the_manifests_route_survives_the_trip_into_the_layout(freeform_library):
    """The connector that had been missing: ``ClipTerrainLayout.terrain_route``.

    Both RL lanes were written against ``ClipTerrainImporter.terrain_route`` while nothing
    populated it, so the free-form runtime raised ``TypeError`` on its first reset. This is
    the round trip end to end -- generator -> ``.h5`` -> ``assemble_clip_terrain`` -- checked
    field by field, because a route that arrives with a field dropped fails much later, at a
    per-step gather.
    """
    path, manifest = freeform_library
    from motion_gen.clips.clip_io import load_clip_library

    lib = load_clip_library(path, "cpu")
    _mesh, _origins, layout = assemble_clip_terrain(lib, num_rows=1, num_flat_tiles=0)

    assert isinstance(layout, ClipTerrainLayout)
    assert len(layout.terrain_route) == len(manifest)
    for tile in manifest:
        got = layout.terrain_route[int(tile["terrain_id"])]
        for name in resets.FREEFORM_ROUTE_FIELDS:
            assert got[name] == pytest.approx(tile["route"][name]), name


def test_a_clip_library_carries_no_route_at_all(freeform_library):
    """The ABSENCE is load-bearing: it is how the runtime tells the two library kinds apart.

    ``resets._freeform_route_table`` raises a named ``TypeError`` on a falsy ``terrain_route``
    rather than spawning at NaN, so a clip library must produce an empty dict here -- not a
    dict of empty routes, and not a missing attribute.
    """
    path, manifest = freeform_library
    from motion_gen.clips.clip_io import load_clip_library

    stripped = [{k: v for k, v in tile.items() if k != "route"} for tile in manifest]
    lib = load_clip_library(path, "cpu")
    lib.attrs["terrain_manifest"] = stripped
    _mesh, _origins, layout = assemble_clip_terrain(lib, num_rows=1, num_flat_tiles=0)
    assert layout.terrain_route == {}


def test_every_tile_the_scene_places_has_a_route(freeform_library):
    """No env may land on a tile with no spawn and no goal, and the filler tile is the hazard.

    ``assemble_clip_terrain`` synthesises flat FILLER tiles, which are not authored and so
    carry no route -- an env on one would hit the named "no route" ``ValueError`` at reset. A
    free-form library asks for zero of them (``generator_config.counts.flat`` is 0, since its
    flat sections are real authored tiles), and that is what keeps the hazard theoretical.
    """
    path, manifest = freeform_library
    from motion_gen.clips.clip_io import load_clip_library

    lib = load_clip_library(path, "cpu")
    assert default_num_flat_tiles(lib) == 0
    _mesh, _origins, layout = assemble_clip_terrain(lib, num_rows=1)
    placed = {int(t) for t in layout.tile_terrain_id.flatten().tolist()}
    assert placed == {int(tile["terrain_id"]) for tile in manifest}
    assert placed <= set(layout.terrain_route)


# ----------------------------------------------------------------------
# The steering centreline is the ray the GENERATOR authored
# ----------------------------------------------------------------------

class _StubImporter:
    """The four attributes ``resets.freeform_route`` reads off a ``ClipTerrainImporter``.

    A real importer needs a simulator; the route lookup needs only these, and feeding it the
    LAYOUT's own route dict keeps the manifest half of the trip real.
    """

    def __init__(self, routes, corner, frame_yaw) -> None:
        """Hold one env per terrain id, in sorted-id order.

        Args:
            routes: The layout's ``terrain_route``.
            corner: ``[k, 3]`` world position of each env's tile corner.
            frame_yaw: ``[k]`` tile frame heading (rad).
        """
        self.terrain_route = routes
        self.env_terrain_id = torch.tensor(sorted(routes), dtype=torch.long)
        self.env_cell_corner = corner
        self.env_terrain_yaw = frame_yaw


def _bare_freeform_command(importer):
    """An ``FmTrackingCommand`` carrying only what ``_default_freeform_section`` reads.

    Args:
        importer: The stub terrain importer.

    Returns:
        The bare instance, with ``freeform_command`` on.
    """
    n = int(importer.env_terrain_id.numel())

    class _Bare(FmTrackingCommand):
        """Shadows the env-backed ``device`` / ``num_envs`` properties so a bare instance works."""

        device = torch.device("cpu")
        num_envs = n
        #: ``CommandTerm.__del__`` touches this; a bare instance never ran the ctor that sets it.
        _debug_vis_handle = None

    cmd = object.__new__(_Bare)
    cmd.cfg = SimpleNamespace(freeform_command=True)
    cmd._ff_origin_w = torch.zeros(n, 2)
    cmd._ff_heading_w = torch.zeros(n)
    cmd.env = SimpleNamespace(scene=SimpleNamespace(terrain=importer, env_origins=None))
    return cmd


def _layout_of(path: str):
    """Assemble a written free-form library and return its layout.

    Args:
        path: The ``.h5`` written by :func:`freeform_library`.

    Returns:
        The :class:`ClipTerrainLayout`.
    """
    from motion_gen.clips.clip_io import load_clip_library

    _mesh, _origins, layout = assemble_clip_terrain(
        load_clip_library(path, "cpu"), num_rows=1, num_flat_tiles=0)
    return layout


def test_the_steering_centreline_is_the_ray_the_generator_authored(freeform_library):
    """One line, stated twice in two packages, pinned equal on every family.

    ``motion_gen.terrain.freeform``'s ``route_project`` / ``route_lookahead`` define a section's
    centreline as the ray from ``(start_x, start_y)`` along ``start_yaw``, and name the command
    as their consumer. ``start_yaw`` is the entire mechanism by which the boxes family meets an
    AXIS-ALIGNED box off-square (``OverlayVolumeCfg`` has no yaw and the loader refuses a
    non-zero one, so the APPROACH is the only place an angle can live) and by which the flat
    family walks a 3-18 deg line rather than a straight one.

    Measured on the default config before this was checked, with the command reading
    ``(start_x, goal_y)`` at ``frame_yaw`` instead: the 5 m aim point sat up to 3.94 m (flat)
    and 2.57 m (boxes) from the authored one, and EXACTLY on it for stairs, where
    ``start_yaw == 0``. The one family that looked fine is the one this terrain set is least
    about, which is why the pin is per tile rather than aggregate.
    """
    layout = _layout_of(freeform_library[0])
    ids = sorted(layout.terrain_route)
    # A non-trivial tile frame in BOTH axes, so a dropped rotation and a dropped translation
    # are separately visible. The shipped manifests are axis-aligned, which is exactly why the
    # lift must be tested against a frame that is not.
    corner = torch.tensor([[3.0 + i, -7.0 - i, 0.0] for i in range(len(ids))])
    frame_yaw = torch.full((len(ids),), 0.4)
    cmd = _bare_freeform_command(_StubImporter(layout.terrain_route, corner, frame_yaw))

    origin_w, heading_w = cmd._default_freeform_section(torch.arange(len(ids)))

    cos_f, sin_f = torch.cos(frame_yaw), torch.sin(frame_yaw)
    for i, tid in enumerate(ids):
        route = layout.terrain_route[tid]
        probe = (route["start_x"] + 0.3, route["start_y"] - 0.2)   # off the line on purpose
        aim_x, aim_y = ff.route_lookahead(route, probe, 5.0)       # the generator's answer
        # The command's line, carried back into the tile frame and stepped the same way.
        d = origin_w[i] - corner[i, :2]
        ox = float(cos_f[i] * d[0] + sin_f[i] * d[1])
        oy = float(-sin_f[i] * d[0] + cos_f[i] * d[1])
        yaw_local = float(heading_w[i] - frame_yaw[i])
        ux, uy = math.cos(yaw_local), math.sin(yaw_local)
        along = (probe[0] - ox) * ux + (probe[1] - oy) * uy
        assert ox + (along + 5.0) * ux == pytest.approx(aim_x, abs=1e-4), f"tile {tid} x"
        assert oy + (along + 5.0) * uy == pytest.approx(aim_y, abs=1e-4), f"tile {tid} y"


def test_the_approach_angle_actually_reaches_the_controller(freeform_library):
    """A guard on the pin above, which stairs alone would satisfy vacuously.

    If ``start_yaw`` were dropped again every heading would come back equal to ``frame_yaw``,
    and on a stairs-only manifest that is indistinguishable from correct. So assert the
    manifest really contains angled sections and that their angle survives the trip.
    """
    layout = _layout_of(freeform_library[0])
    ids = sorted(layout.terrain_route)
    cmd = _bare_freeform_command(_StubImporter(
        layout.terrain_route, torch.zeros(len(ids), 3), torch.zeros(len(ids))))
    _origin_w, heading_w = cmd._default_freeform_section(torch.arange(len(ids)))

    angled = [i for i, tid in enumerate(ids)
              if abs(layout.terrain_route[tid]["start_yaw"]) > 1e-3]
    assert angled, "this manifest has no angled section, so the guard would be vacuous"
    for i in angled:
        assert float(heading_w[i]) == pytest.approx(
            layout.terrain_route[ids[i]]["start_yaw"], abs=1e-6)


def test_the_goal_height_test_is_off_by_default_and_closes_the_ascent_hole():
    """`freeform_goal_reached`'s height term must be opt-in, and must use the PELVIS datum.

    Off by default because turning it on silently would change the termination of every
    free-form task including arms already measured, and a success rate that moved for that
    reason is indistinguishable from one that moved because the policy did.

    Needed at all because the downrange-and-lateral test has no height: on a clip-library
    `stairs_up` tile there is 0.90 m of open floor either side of a 2.25 m flight and
    `min_half_width` clamps the band at least that wide, so a robot that never left the floor
    passes the goal line and scores. The DESCENDING families are immune for a different
    reason -- goal at floor level, spawn on the platform -- which is why this is a parameter.
    """
    import inspect

    from robot_rl.tasks.manager_based.robot_rl.mdp.terminations import terminations as T

    sig = inspect.signature(T.freeform_goal_reached)
    assert "height_tol_m" in sig.parameters, "the height term is gone"
    assert sig.parameters["height_tol_m"].default == 0.0, (
        "the height term must default OFF: enabling it changes every free-form task's "
        "termination, and every number measured before it")

    src = inspect.getsource(T.freeform_goal_reached)
    # The datum is the pelvis, not the walkable surface. `goal_z` is a SURFACE height and a
    # standing root sits ~0.79 m above it, so a raw `root_z` vs `goal_z` comparison puts
    # every genuine arrival outside a 0.5 m tolerance -- the asymmetry that once made every
    # box2 arrival record as a failure.
    assert "default_root_pose" in src, (
        "the height test must add the asset's nominal root height, or it compares a pelvis "
        "against a floor")
    assert 'route["corner"]' in src and "goal_z" in src, (
        "the goal height must sum the tile corner, the route's goal_z and the standing "
        "height -- the same three terms freeform_goal_anchor_world uses")
    # And it must still be an AND with the lateral/downrange test, not a replacement.
    assert "hit = hit &" in src or "hit &" in src


def test_the_finetune_env_turns_the_height_test_ON_by_default() -> None:
    """The FUNCTION defaults the height test off; the fine-tune ENV must not.

    Two different decisions, and conflating them is what left the check off everywhere.  The
    library default is off so that enabling a height term cannot silently move every free-form
    task's termination -- see
    :func:`test_the_goal_height_term_is_opt_in_and_uses_the_pelvis`.  The fine-tune env is a
    concrete task whose four courses (box2, up-into-down, varied, in-distribution) all declare
    0.5 in their own goal specs, so a DIRECT ``train.py`` / ``play.py`` launch should get the
    check those courses were designed with rather than the library's safe-for-migration default.

    0.5 specifically, because it must not bind on the DESCENDING courses, whose goal sits at
    floor level and whose height change is unavoidable.  Measured at arrival: box2 tops out at
    0.203 m of ``|height_err|`` and up-into-down at 0.399, while in-distribution binds at
    exactly 0.500 -- so one value serves ascending and descending alike.

    What it buys: the downrange-and-lateral test has no height, so on a ``stairs_up`` tile with
    0.90 m of open floor either side of the flight, a robot that never left the floor passes the
    goal line and scores a traverse.
    """
    import inspect

    from robot_rl.tasks.manager_based.robot_rl.g1 import g1_finetune_env_cfg as C

    src = inspect.getsource(C)
    assert '"height_tol_m": 0.5' in src, (
        "the fine-tune env's goal_reached must declare the height band ON at 0.5; 0.0 turns "
        "the test off and makes the floor-walker a scored traverse")
    # DECLARED rather than omitted: hydra's struct mode refuses a key that is absent, so
    # dropping it would make the band unsettable from the CLI and keep it out of the run's own
    # params/env.yaml -- the only record of what a run was configured with.
    assert "height_tol_m" in src
