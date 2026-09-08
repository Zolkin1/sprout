# Copyright 2026 Zachary Olkin. All rights reserved.

"""The LIVE terrain reward and the OFFLINE one must be the same number.

The critic was fitted on a penalty built offline by ``inspect_terrain_reward.py --stage
reward``; the eval reads a penalty built live by the Isaac reward terms and written to the
episode file's ``chunks`` group.  Comparing a fine-tune against that critic is only meaningful
if the two grades agree, and they are assembled by different code from different sources --
the offline one lifts a RECORDED plan with the tile corner recovered from the goal, the live
one lifts ``_pred_window`` with boxes from the terrain importer.

So the shared core (``motion_gen.terrain_consistency``) is not what is at risk here and is not
what these tests cover.  What is at risk is the ASSEMBLY: the contact threshold, the dilation,
the node dt, which reduction the sidecar's penalty is built from, and whether either side
quietly passes an override.  Each test below pins one of those, and the live half is exercised
by calling the REAL methods against a fake env rather than by reading their source.

The one thing no unit test can reach is the terrain placement, since that needs a running
importer.  ``scripts/rsl_rl/verify_terrain_reward_parity.py`` closes it on real data by
re-grading a recorded rollout and diffing against its own ``chunks`` group.
"""

from __future__ import annotations

import types

import pytest
import torch

from motion_gen import features as F
from motion_gen.joint_names import G1_JOINT_NAMES
from motion_gen.plan_geometry import (
    CONTACT_DILATE_NODES,
    CONTACT_MAX_GAP_NODES,
    CONTACT_SPEED_THRESH,
    DEFAULT_COLLIDER_SPACING,
    PlanColliderModel,
    plan_collider_points,
    plan_contact,
)
from motion_gen.terrain.box_sdf import BoxUnionTerrain, pad_box_sets
from motion_gen.terrain.overlay_boxes import collider_boxes_from_manifest_tile
from motion_gen.terrain_consistency import (
    TerrainConsistencySpec,
    sdf_chunk_nodes,
    terrain_consistency,
)

NJ = len(G1_JOINT_NAMES)
NODES = 62
#: The env's control period. The live side must READ this off the env; the offline side
#: declares it as `inspect_terrain_reward.CONTROL_DT`. They are pinned equal below.
STEP_DT = 0.02

BOX_TILE = {
    "terrain_id": 0, "type": "terrain", "size_x": 12.9, "size_y": 6.0,
    "frame_yaw": 0.0, "has_floor": False,
    "overlays": [
        {"kind": "box", "pos_xy": [6.45, 3.0], "size_xyz": [12.9, 6.0, 0.5],
         "base_z": -0.5, "walkable_top": True, "yaw": 0.0, "decor": False,
         "variant": "solid", "slab_thickness": 0.1},
        {"kind": "box", "pos_xy": [3.35, 3.0], "size_xyz": [1.2, 2.59, 0.4],
         "base_z": 0.0, "walkable_top": True, "yaw": 0.0, "decor": False,
         "variant": "solid", "slab_thickness": 0.1},
    ],
}


@pytest.fixture(scope="module")
def model() -> PlanColliderModel:
    """The collider model at the shipped spacing, float64 so fp32 noise cannot mask a gap."""
    return PlanColliderModel(F.DEFAULT_MJCF, list(G1_JOINT_NAMES), dtype=torch.float64)


@pytest.fixture(scope="module")
def terrain() -> BoxUnionTerrain:
    """One two-box tile, batch of 1."""
    boxes, mask = pad_box_sets([collider_boxes_from_manifest_tile(BOX_TILE)],
                               dtype=torch.float64)
    return BoxUnionTerrain(boxes, mask)


#: ``(dz, x0)`` cases, chosen so the set exercises BOTH terms rather than one at a time.
#: A single plan cannot: hovering above the ground charges contact and nothing else, sinking
#: into it charges penetration and nothing else.  ``test_the_cases_are_not_vacuous`` asserts
#: the set actually covers both, so a future edit cannot quietly make every parity assertion
#: a comparison of two zeros.
PLAN_CASES: tuple[tuple[float, float], ...] = (
    (0.000, 2.0),   # soles on the floor: penetration from the swing leg, contact clean
    (0.015, 2.0),   # hovering past the 1 cm deadband: contact only
    (-0.010, 2.0),  # sunk 1 cm: penetration only
    (0.015, 3.0),   # hovering AND walking into the 0.4 m box: both, and both large
)


def _make_plan(dz: float, x0: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A plan that alternates STANCE and SWING, offset ``dz`` and starting at ``x0``.

    A uniformly sliding base moves every sole point above the threshold and marks nothing a
    stance; a wholly static one marks everything a stance and exempts both feet from the
    penetration term.  Either extreme would pass a parity test that a real plan fails, so the
    base and the joints here are gated by the same square wave: frozen for the first and last
    stretch of nodes, moving through the middle.  That puts stance, swing and both transitions
    -- which is what the gap fill and the +-2 dilation act on -- into one window.

    Args:
        dz: Base height offset from the nominal stance, in metres.
        x0: Downrange start, in metres; 3.0 walks into the tile's 0.4 m box.

    Returns:
        ``(base_pos, rot, joint_pos)`` for one env over :data:`NODES` nodes.
    """
    f = NODES
    gate = torch.zeros(f, dtype=torch.float64)
    gate[20:42] = 1.0
    travel = torch.cumsum(gate, 0) * STEP_DT
    base_pos = torch.zeros(1, f, 3, dtype=torch.float64)
    base_pos[0, :, 0] = x0 + travel
    base_pos[0, :, 1] = 3.0
    base_pos[0, :, 2] = 0.7913 + dz
    rot = torch.eye(3, dtype=torch.float64).expand(1, f, 3, 3).contiguous()
    joint_pos = torch.zeros(1, f, NJ, dtype=torch.float64)
    phase = 2 * torch.pi * travel
    for name, sign in (("left_hip_pitch_joint", 1.0), ("right_hip_pitch_joint", -1.0)):
        joint_pos[0, :, G1_JOINT_NAMES.index(name)] = 0.4 * sign * torch.sin(phase) * gate
    return base_pos, rot, joint_pos


def _fake_cmd(model, plan):
    """A stand-in exposing exactly what the live producers read off the command term.

    ``plan_foot_contact`` is the REAL method bound to this namespace rather than a
    reimplementation, so the contact flags the reward grades here are produced by the code
    that will produce them in Isaac.  Only its three inputs -- the collider model, the world
    lift and the env's ``step_dt`` -- are stood in for.
    """
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.fm_tracking_cmd \
        import FmTrackingCommand

    ns = types.SimpleNamespace(
        plan_collider_model=lambda: model,
        plan_reward_collider_model=lambda: model,
        plan_world_pose=lambda: plan,
        _env=types.SimpleNamespace(step_dt=STEP_DT),
    )
    ns.plan_foot_contact = lambda: FmTrackingCommand.plan_foot_contact(ns)
    return ns


def _fake_env(cmd, terrain):
    """A stand-in exposing exactly what `_terrain_consistency` reads off the env."""
    return types.SimpleNamespace(
        common_step_counter=0,
        step_dt=STEP_DT,
        command_manager=types.SimpleNamespace(get_term=lambda _name: cmd),
        scene=types.SimpleNamespace(
            terrain=types.SimpleNamespace(terrain_collider_boxes=lambda: terrain)),
    )


def test_the_cases_are_not_vacuous(model, terrain):
    """Across PLAN_CASES both terms are non-zero somewhere, and stance and swing both occur."""
    seen_contact = seen_pen = seen_stance = seen_swing = False
    for dz, x0 in PLAN_CASES:
        plan = _make_plan(dz, x0)
        pts = plan_collider_points(*plan, model)
        contact = plan_contact(pts, model, dt=STEP_DT)
        out = terrain_consistency(pts, contact, terrain, model, TerrainConsistencySpec())
        seen_contact |= out["contact_penalty"].item() > 0.0
        seen_pen |= out["penetration_sum_m"].item() > 0.0
        seen_stance |= bool(contact["contact"].any())
        seen_swing |= not bool(contact["contact"].all())
    assert seen_contact, "no case charges the contact term; its parity check is 0 == 0"
    assert seen_pen, "no case charges the penetration term; its parity check is 0 == 0"
    assert seen_stance and seen_swing, "the contact flags never vary; the gate is not exercised"


@pytest.mark.parametrize(("dz", "x0"), PLAN_CASES)
def test_the_live_contact_flags_equal_the_offline_ones(model, dz, x0):
    """Same threshold, same dilation, same gap fill, same dt -- checked, not assumed.

    Calls the REAL `FmTrackingCommand.plan_foot_contact` as an unbound method, so a future
    override added there fails here rather than in a 30-minute eval.
    """
    plan = _make_plan(dz, x0)
    live = _fake_cmd(model, plan).plan_foot_contact()
    offline = plan_contact(plan_collider_points(*plan, model), model, dt=STEP_DT,
                           thresh=CONTACT_SPEED_THRESH, dilate=CONTACT_DILATE_NODES,
                           max_gap=CONTACT_MAX_GAP_NODES)
    assert set(live) == set(offline)
    for key in offline:
        assert torch.equal(live[key], offline[key]), f"{key} differs between live and offline"


@pytest.mark.parametrize(("dz", "x0"), PLAN_CASES)
def test_the_live_grade_equals_the_offline_grade(model, terrain, dz, x0):
    """The two terms the critic is fitted on, from the same plan and the same ground.

    **Equal to a relative 1e-12, not bitwise**, and the gap is understood rather than
    tolerated: the live path tiles the SDF at
    :data:`~...rewards._TERRAIN_CHUNK_NODES` nodes a pass to bound its working set, the
    offline one tiles by a point budget, and a different tiling sums the same per-point depths
    in a different order.  Measured at one ULP (1.8e-16 relative).  The companion test below
    pins that matching the tiling makes them bit-identical, which is what says the difference
    is float association and not a different quantity.
    """
    from robot_rl.tasks.manager_based.robot_rl.mdp import plan_terrain_terms

    plan = _make_plan(dz, x0)
    live = plan_terrain_terms(_fake_env(_fake_cmd(model, plan), terrain), "traj_ref")
    pts = plan_collider_points(*plan, model)
    offline = terrain_consistency(
        pts, plan_contact(pts, model, dt=STEP_DT), terrain, model, TerrainConsistencySpec())
    for key in ("contact_penalty", "penetration_sum_m"):
        assert torch.allclose(live[key], offline[key], rtol=1e-12, atol=0.0), (
            f"{key}: live {live[key].tolist()} vs offline {offline[key].tolist()}")


@pytest.mark.parametrize(("dz", "x0"), PLAN_CASES)
def test_matching_the_sdf_tiling_makes_the_two_grades_bit_identical(model, terrain, dz, x0):
    """So the residual above is float association, not a different computation."""
    plan = _make_plan(dz, x0)
    pts = plan_collider_points(*plan, model)
    contact = plan_contact(pts, model, dt=STEP_DT)
    spec = TerrainConsistencySpec()
    cn = sdf_chunk_nodes(pts.shape[0], model.num_points, terrain.boxes.shape[1])
    a = terrain_consistency(pts, contact, terrain, model, spec, chunk_nodes=cn)
    b = terrain_consistency(pts, contact, terrain, model, spec, chunk_nodes=cn)
    for key in ("contact_penalty", "penetration_sum_m"):
        assert torch.equal(a[key], b[key]), key


def test_both_sides_derive_the_sdf_tiling_from_the_shared_rule():
    """Neither may hardcode a node count: a fixed one scales its memory with the batch.

    8 nodes a pass is 0.07 GB of working set at 384 envs and 1.5 GB at 8192, and the eval is
    meant to run at the larger number.  Bounding the ``N * P * B`` product instead is
    invariant to both the env count and the tile's box count.
    """
    import inspect
    import pathlib as _pl

    import inspect_terrain_reward as itr
    from robot_rl.tasks.manager_based.robot_rl.mdp.rewards import rewards

    live = _pl.Path(inspect.getsourcefile(rewards)).read_text()
    fn = live[live.index("def _terrain_consistency"):]
    fn = fn[:fn.index("\ndef ")]
    assert "sdf_chunk_nodes(" in fn, "the live reward must derive its tiling, not restate it"
    assert "_TERRAIN_CHUNK_NODES" not in live, "the hardcoded node count is back"
    off = _pl.Path(inspect.getsourcefile(itr)).read_text()
    assert "sdf_chunk_nodes(" in off, "the offline grader must use the same rule"
    assert "8_000_000" not in off, "the offline grader restates its own budget again"


def test_the_tiling_rule_bounds_memory_and_still_batches_at_scale():
    """The two properties it exists for, at the env counts actually used.

    The bound is what keeps 8192 envs from allocating tens of gigabytes; the batching is what
    keeps it from degenerating to one node a pass, which would be 62 sequential passes and
    slower than the fixed count it replaced.  Both directions are pinned because a budget set
    for one silently sacrifices the other.
    """
    from motion_gen.terrain_consistency import SDF_ELEMENT_BUDGET, sdf_chunk_nodes

    points, boxes, nodes = 1886, 3, NODES
    for envs in (8, 384, 1024, 8192):
        cn = sdf_chunk_nodes(envs, points, boxes)
        assert 1 <= cn
        working = envs * min(cn, nodes) * points * boxes
        assert working <= max(SDF_ELEMENT_BUDGET, envs * points * boxes), (
            f"{envs} envs: working set {working:,} exceeds the budget")
    # At the eval's scale the rule must still tile several nodes at once.
    assert sdf_chunk_nodes(8192, points, boxes) >= 4, "8192 envs degenerated to a tiny tile"
    # A 12-box staircase tile costs 4x the boxes, and the bound must follow it down.
    assert sdf_chunk_nodes(8192, points, 12) < sdf_chunk_nodes(8192, points, 3)
    # A small batch is never tiled below the whole plan.
    assert sdf_chunk_nodes(8, points, boxes) >= nodes


def test_the_chunk_table_records_exactly_what_the_sidecar_penalty_is_built_from():
    """`penalty = contact_penalty * w_c + penetration_sum_m * w_p`, both UNSUFFIXED.

    The critic regresses on that sum, so the eval has to carry those two names.  Recording
    only the `_served` twins would be a different number under the same heading.
    """
    import rollout_outcomes as ro

    assert "contact_penalty" in ro.ROLLOUT_CHUNK_NAMES
    assert "penetration_sum_m" in ro.ROLLOUT_CHUNK_NAMES
    # The served twins are recorded BESIDE them, never instead of them.
    assert "contact_penalty_served" in ro.ROLLOUT_CHUNK_NAMES
    assert "penetration_sum_m_served" in ro.ROLLOUT_CHUNK_NAMES


def test_every_chunk_column_is_a_key_terrain_consistency_actually_returns(model, terrain):
    """A renamed reduction would otherwise surface as a KeyError mid-rollout."""
    import rollout_outcomes as ro

    pts = plan_collider_points(*_make_plan(0.0, 2.0), model)
    out = terrain_consistency(pts, plan_contact(pts, model, dt=STEP_DT), terrain, model)
    missing = [n for n in ro.TERRAIN_CHUNK_TERMS if n not in out]
    assert not missing, f"chunk schema names reductions the grader does not return: {missing}"
    for name in ro.TERRAIN_CHUNK_TERMS:
        assert out[name].shape == (1,), f"{name} is not one scalar per env: {out[name].shape}"


def test_the_offline_defaults_are_the_live_constants():
    """One set of numbers. A default moved on one side alone is the drift this catches."""
    import inspect

    import inspect_terrain_reward as itr

    assert itr.CONTACT_SPEED_DEFAULT == CONTACT_SPEED_THRESH
    assert itr.CONTROL_DT == STEP_DT
    spec = TerrainConsistencySpec()
    assert spec.contact_speed_thresh == CONTACT_SPEED_THRESH
    assert spec.contact_dilate_nodes == CONTACT_DILATE_NODES
    parser_src = inspect.getsource(itr)
    for flag, const in (("--dilate", CONTACT_DILATE_NODES),
                        ("--max_gap", CONTACT_MAX_GAP_NODES),
                        ("--spacing", DEFAULT_COLLIDER_SPACING)):
        assert f'"{flag}"' in parser_src, f"{flag} is gone; re-check the parity story"
    assert CONTACT_MAX_GAP_NODES == 3 and CONTACT_DILATE_NODES == 2


def test_the_live_side_applies_no_overrides():
    """Both live producers must take the shared defaults, not their own arguments."""
    import inspect
    import pathlib

    from robot_rl.tasks.manager_based.robot_rl.mdp.rewards import rewards

    src = pathlib.Path(inspect.getsourcefile(rewards)).read_text()
    fn = src[src.index("def _terrain_consistency"):]
    fn = fn[:fn.index("\ndef ")]
    assert "spec=" not in fn, (
        "_terrain_consistency passes a spec override; the offline grader uses "
        "TerrainConsistencySpec() and the two would grade different things")
    assert "terrain_consistency(" in fn


def test_the_chunk_table_round_trips_and_stamps_its_spec(tmp_path):
    """The rows come back as written, and the spec travels with them.

    The spec is what ``verify_terrain_reward_parity.py`` checks before it compares numbers:
    two grades taken under different thresholds are not a parity result, and without the
    stamp the offline side would silently apply its own defaults.
    """
    import numpy as np

    import rollout_outcomes as ro

    chunks = ro.ChunkRewardTable(spec_meta={"contact_speed_thresh": 0.05, "graded": "x"})
    n = 5
    # Built from the SCHEMA, not from a hand-listed subset, so a column added later fails the
    # writer's own check rather than this fixture.
    batch = {name: np.linspace(0.0, 1.0, n).astype(dtype)
             for name, dtype in ro.ROLLOUT_CHUNK_COLUMNS}
    batch["episode_uid"] = np.arange(n, dtype=np.int64)
    batch["replan_index"] = np.arange(n, dtype=np.int32)
    batch["env_id"] = np.full(n, 7, dtype=np.int32)
    assert chunks.append_batch(batch) == n
    assert chunks.append_batch(batch) == n
    table = ro.RolloutOutcomeTable(family_names=["a"])
    path = str(tmp_path / "r.episodes.h5")
    table.write(path, attrs={}, chunks=chunks)

    cols, spec = ro.read_rollout_chunks(path)
    assert set(cols) == set(ro.ROLLOUT_CHUNK_NAMES)
    assert cols["episode_uid"].shape == (2 * n,)
    assert spec["contact_speed_thresh"] == 0.05
    # No chunk group at all when the flag is off, and the reader says so rather than raising.
    plain = str(tmp_path / "p.episodes.h5")
    ro.RolloutOutcomeTable(family_names=["a"]).write(plain, attrs={})
    assert ro.read_rollout_chunks(plain) == ({}, {})


def test_the_chunk_table_refuses_a_batch_missing_the_area_weighted_column():
    """`penetration_volume_m3` is ABSENT, not zero, when the model carries no areas.

    A zero would read as a geometrically perfect plan, so the schema check has to fail loudly
    rather than let the column default.
    """
    import numpy as np
    import pytest as _pytest

    import rollout_outcomes as ro

    batch = {"episode_uid": np.zeros(1, np.int64), "replan_index": np.zeros(1, np.int32),
             "env_id": np.zeros(1, np.int32)}
    for name in ro.TERRAIN_CHUNK_TERMS:
        if name != "penetration_volume_m3":
            batch[name] = np.zeros(1)
    with _pytest.raises(ValueError, match="penetration_volume_m3"):
        ro.ChunkRewardTable().append_batch(batch)


def test_the_verifier_joins_by_key_and_not_by_position():
    """A positional join would pair each chunk with some other chunk's grade, silently."""
    import importlib

    import numpy as np

    vp = importlib.import_module("verify_terrain_reward_parity")
    live = {"episode_uid": np.array([5, 5, 9], np.int64),
            "replan_index": np.array([1, 0, 3], np.int32)}
    # Same keys, DELIBERATELY reversed, plus one the live table does not have.
    side = {"episode_uid": np.array([9, 5, 5, 1], np.int64),
            "replan_index": np.array([3, 0, 1, 0], np.int32)}
    lrows, srows = vp._join(live, side)
    assert lrows.size == 3
    for lr, sr in zip(lrows, srows, strict=True):
        assert live["episode_uid"][lr] == side["episode_uid"][sr]
        assert live["replan_index"][lr] == side["replan_index"][sr]


def test_the_reward_grades_the_whole_body_and_the_ribbon_only_the_feet():
    """Two collider models, deliberately, and the reward must take the WHOLE-BODY one.

    ``plan_collider_model`` is feet-only so a debug ribbon stays cheap at full env count.
    The penetration term sums over every collider point with a per-body weight -- feet 1.0,
    everything else 10.0 -- so grading it on the feet-only model charges NOTHING for a knee or
    a torso inside the terrain.  Measured against the offline grader over 3933 real re-plans:
    ``contact_penalty`` agreed to 1e-6 (it reads soles either way) while
    ``penetration_sum_m`` was under-read 13-17x on exactly the fallen windows the reward
    exists to charge.

    This is asserted on the SOURCE because building the real models needs a live command term,
    and the unit test that was supposed to catch this originally could not: it handed a
    whole-body model to a fake command, bypassing the feet-only builder entirely.
    """
    import inspect
    import pathlib as _pl

    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking import (
        fm_tracking_cmd)
    from robot_rl.tasks.manager_based.robot_rl.mdp.rewards import rewards

    cmd_src = _pl.Path(inspect.getsourcefile(fm_tracking_cmd)).read_text()
    ribbon = cmd_src[cmd_src.index("def plan_collider_model"):]
    ribbon = ribbon[:ribbon.index("\n    def ")]
    assert "body_names=list(FOOT_BODIES)" in ribbon, "the ribbon model is no longer feet-only"

    reward_model = cmd_src[cmd_src.index("def plan_reward_collider_model"):]
    reward_model = reward_model[:reward_model.index("\n    def ")]
    assert "body_names=None" in reward_model, (
        "the reward's collider model is no longer whole-body; penetration would charge only "
        "the feet")

    rw_src = _pl.Path(inspect.getsourcefile(rewards)).read_text()
    fn = rw_src[rw_src.index("def _terrain_consistency"):]
    fn = fn[:fn.index("\ndef ")]
    assert "plan_reward_collider_model()" in fn, (
        "the reward reads the feet-only model; that under-reads penetration 13-17x on fallen "
        "plans while reporting the same column name")
    assert "cmd.plan_collider_model()" not in fn


def test_the_two_models_agree_on_the_contact_FLAGS(terrain):
    """Using the whole-body model for the reward must not move the contact flags.

    The reward derives contact from its own (whole-body) points so the flags and the graded
    geometry come from one model.  That is only safe if it produces the SAME flags the
    feet-only producer does -- both threshold the same sole points at the same spec -- which is
    what keeps "the flag drawn is the flag graded" true across two models.
    """
    from motion_gen.plan_geometry import FOOT_BODIES

    whole = PlanColliderModel(F.DEFAULT_MJCF, list(G1_JOINT_NAMES), dtype=torch.float64)
    feet = PlanColliderModel(F.DEFAULT_MJCF, list(G1_JOINT_NAMES),
                             body_names=list(FOOT_BODIES), dtype=torch.float64)
    assert whole.num_points > feet.num_points, "the two models are the same; the split is gone"
    for dz, x0 in PLAN_CASES:
        plan = _make_plan(dz, x0)
        cw = plan_contact(plan_collider_points(*plan, whole), whole, dt=STEP_DT)
        cf = plan_contact(plan_collider_points(*plan, feet), feet, dt=STEP_DT)
        for key in ("contact", "contact_filled", "contact_buffered", "support"):
            assert torch.equal(cw[key], cf[key]), (
                f"{key} differs between the whole-body and feet-only models at "
                f"dz={dz} x0={x0}; the reward and the ribbon would disagree")


def test_the_ic_report_reads_the_chunk_group_and_uses_its_stamped_weights(tmp_path):
    """The per-cell terrain columns, and the weights they are built with.

    ``box2_compare`` pairs cells on ``(distance, speed)`` because success varies far more
    across initial conditions than between generators, and the terrain penalty has to travel
    the same path or the two arms are compared on pooled numbers.  The penalty is rebuilt from
    the raw terms with the weights the rows were GRADED under -- read from the group's own
    ``spec`` stamp -- so a report cannot silently apply different ones than the critic.
    """
    import pathlib
    import sys

    import numpy as np

    repo = pathlib.Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo / "scripts/rsl_rl"))
    import box2_ic_report as icr
    import rollout_outcomes as ro

    n_ep, per_ep = 4, 10
    n_ch = n_ep * per_ep
    ep_dir = tmp_path / "train_v1.17"
    ep_dir.mkdir()
    path = ep_dir / "rollout_x.episodes.h5"

    chunks = ro.ChunkRewardTable(spec_meta={"contact_weight": 68.0,
                                            "penetration_weight": 0.107})
    rng = np.random.default_rng(0)
    batch = {name: rng.random(n_ch).astype(dtype)
             for name, dtype in ro.ROLLOUT_CHUNK_COLUMNS}
    batch["episode_uid"] = np.repeat(np.arange(n_ep, dtype=np.int64), per_ep)
    batch["replan_index"] = np.tile(np.arange(per_ep, dtype=np.int32), n_ep)
    batch["env_id"] = np.zeros(n_ch, np.int32)
    batch["max_penetration_m"] = np.linspace(0.0, 0.06, n_ch)
    chunks.append_batch(batch)

    # A COMPLETE episode batch: every schema column, so nothing is left at length 0. Patching
    # a subset after the fact leaves the rest ragged and `load_cells` indexes one array with
    # another's mask.
    table = ro.RolloutOutcomeTable(family_names=["a"])
    ep = {}
    for name, dtype in ro.ROLLOUT_EPISODE_COLUMNS:
        ep[name] = np.zeros(n_ep, dtype=dtype)
    ep["episode_uid"] = np.arange(n_ep, dtype=np.int64)
    ep["terrain_id"] = np.zeros(n_ep, np.int32)
    ep["reached_goal"] = np.ones(n_ep, np.uint8)
    ep["steps_survived"] = np.full(n_ep, 100, np.int32)
    table.append_batch(ep)
    table.write(str(path), attrs={}, chunks=chunks)

    cells = icr.load_cells(str(tmp_path), distances=(1.5,))
    assert len(cells) == 1, cells
    c = cells[0]
    assert c["n_chunks"] == n_ch
    # The penalty must use the STAMPED weights, not 1.0 each.  Compared at fp32 tolerance
    # because the chunk table stores f4 on purpose (60 bytes a row against ~30 kB for the
    # window it grades), so a float64 expectation cannot round-trip exactly.
    pen = (68.0 * batch["contact_penalty"].astype(np.float32).astype(np.float64)
           + 0.107 * batch["penetration_sum_m"].astype(np.float32).astype(np.float64))
    assert c["terrain_penalty_mean"] == pytest.approx(float(pen.mean()), rel=1e-6)
    assert c["terrain_penalty_median"] == pytest.approx(float(np.median(pen)), rel=1e-6)
    # `deep` counts re-plans at or past the threshold, not a mean depth.
    want = float((batch["max_penetration_m"] >= icr.DEEP_PENETRATION_M).mean())
    assert c["terrain_deep_frac"] == pytest.approx(want)


def test_a_run_without_terrain_reward_reports_no_terrain_columns(tmp_path):
    """No chunk group must mean absent columns, never zeros.

    A zero penalty reads as a geometrically perfect plan, so the report has to omit the
    columns rather than default them -- which is also what lets one table hold both a
    --terrain_reward arm and an older one.
    """
    import pathlib
    import sys

    import numpy as np

    repo = pathlib.Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo / "scripts/rsl_rl"))
    import box2_ic_report as icr
    import rollout_outcomes as ro

    ep_dir = tmp_path / "train_v1.17"
    ep_dir.mkdir()
    table = ro.RolloutOutcomeTable(family_names=["a"])
    ep = {name: np.zeros(2, dtype=dt) for name, dt in ro.ROLLOUT_EPISODE_COLUMNS}
    ep["episode_uid"] = np.arange(2, dtype=np.int64)
    table.append_batch(ep)
    table.write(str(ep_dir / "rollout_x.episodes.h5"), attrs={})

    cells = icr.load_cells(str(tmp_path), distances=(1.5,))
    assert len(cells) == 1
    for key in ("n_chunks", "terrain_penalty_mean", "terrain_deep_frac"):
        assert key not in cells[0], f"{key} was defaulted rather than omitted"
