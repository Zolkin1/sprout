# Copyright 2026 Zachary Olkin. All rights reserved.

"""The MuJoCo domain randomization, against Isaac's own ranges and MuJoCo's own traps.

Three of the four things pinned here were wrong or nearly wrong when first written, and each
would have produced a plausible robot rather than an error:

* **compounding.** ``mj_resetData`` does not touch ``mjModel``, so a per-episode ``scale``
  draw applied without re-basing is a log-random walk: 100 episodes of ``U(0.9, 1.1)`` leaves
  a robot nothing drew.
* **inertia.** MuJoCo does not scale ``body_inertia`` when ``body_mass`` changes. Scaling mass
  alone gives a body 1.15x heavier with its original inertia tensor -- a different body, not a
  heavier one -- and ``body_subtreemass`` stays stale until ``mj_setConst``.
* **joint order.** Gains are written by NAME. Reading them back POSITIONALLY reports a
  77.7 error on a correct application, because 27 of 29 actuator indices do not match the
  policy's joint order. That is the same trap as the clip spawn, in a different file.

The fourth, armature being DOF-indexed (``nv`` 35 against ``nu`` 29, the free joint's six at
the front), is pinned by construction: :class:`domain_rand.MujocoDR` maps through
``jnt_dofadr`` and this checks the mapped entries move while the free-joint DOFs do not.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(os.path.dirname(_SIM))
for _p in (_SIM, os.path.dirname(_SIM), os.path.join(_SIM, "mjeval"),
           os.path.join(_ROOT, "source", "robot_rl"),
           os.path.join(_ROOT, "source", "motion_gen"),
           os.path.join(_ROOT, "scripts", "rsl_rl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

LIB = os.path.join(_ROOT, "clips", "g1_mb_v5_body_mode11_decor.h5")
pytestmark = pytest.mark.skipif(not os.path.exists(LIB), reason="needs the clip library")


def test_draws_land_inside_isaacs_ranges():
    """Every term must stay inside the range Isaac declared -- these are the target."""
    import domain_rand as DR

    rng = np.random.default_rng(0)
    ds = [DR.draw(rng, 29, s) for s in range(300)]
    for name, get in (("joint_friction", lambda d: d.joint_friction),
                      ("joint_armature", lambda d: d.joint_armature),
                      ("stiffness", lambda d: d.stiffness),
                      ("damping", lambda d: d.damping),
                      ("joint_default_pos", lambda d: d.joint_default_offset)):
        lo, hi = DR.ISAAC_DR[name if name != "joint_default_pos" else "joint_default_pos"]["range"]
        v = np.concatenate([get(d) for d in ds])
        assert v.min() >= lo and v.max() <= hi, f"{name} left [{lo}, {hi}]"
    ms = np.array([d.mass_scale for d in ds])
    lo, hi = DR.ISAAC_DR["torso_mass"]["range"]
    assert ms.min() >= lo and ms.max() <= hi


def test_one_mass_scale_is_shared_by_the_three_bodies():
    """Isaac's term lists them together, so a per-body draw would randomize their RATIO."""
    import domain_rand as DR

    d = DR.draw(np.random.default_rng(3), 29, 3)
    assert isinstance(d.mass_scale, float), (
        "mass_scale must be ONE number: torso and both backpack bodies scale together in "
        "Isaac's add_torso_mass, so drawing per body would randomize a mass ratio its own "
        "draw holds fixed.")


def test_draws_are_reproducible_from_their_seed():
    """A recorded draw must be re-derivable, or a stored dr_draw is not checkable."""
    import domain_rand as DR

    a = DR.draw(np.random.default_rng(11), 29, 11)
    b = DR.draw(np.random.default_rng(11), 29, 11)
    assert np.array_equal(a.joint_armature, b.joint_armature)
    assert a.mass_scale == b.mass_scale
    assert "mass_scale" in a.to_json()


@pytest.fixture(scope="module")
def rigged():
    """A loaded arm plus its DR handle.

    Returns:
        ``(MujocoDR, robot, policy)``.
    """
    import dataclasses
    import types

    import domain_rand as DR
    import terrains
    from fm_noise_attribution import build_arm

    exp = os.path.join(_ROOT, "logs/rsl_rl/g1/2026-08-20_16-36-10_mode11_decor_plan_v1/exported")
    tracker = os.path.join(exp, "tracker.onnx")
    fm = os.path.join(exp, "2026-08-25_09-03-39_plan_v1_h64_27M_nojumpoff_fm.onnx")
    if not (os.path.exists(tracker) and os.path.exists(fm)):
        pytest.skip("needs the exported pair")
    course = terrains.isaac_tile_course(LIB, 6, speeds=(0.8,), approach=2.5)
    course = dataclasses.replace(course, name="tile6_drpytest")
    _cfg, policy, _lag, _spawn, rig = build_arm(
        types.SimpleNamespace(tracker_onnx=tracker, fm_onnx=fm, out=".", providers="auto",
                              fall_height=0.2, fall_tilt_deg=70.0, cmd_hold_s=0.0), course)
    return DR.MujocoDR(rig.robot, policy), rig.robot, policy


def test_applying_twice_does_not_compound(rigged):
    """mjModel survives mj_resetData, so every apply must re-base from nominal."""
    import domain_rand as DR

    dr, robot, _ = rigged
    m = robot.mj_model
    d = DR.draw(np.random.default_rng(7), 29, 7)
    dr.apply(d)
    first = (float(np.sum(m.body_mass)), np.array(m.dof_armature, copy=True))
    dr.apply(d)
    assert float(np.sum(m.body_mass)) == first[0]
    assert np.array_equal(np.array(m.dof_armature), first[1]), (
        "armature changed on a repeat application, so the draw is COMPOUNDING. Over an "
        "episode sweep that is a log-random walk away from the range Isaac drew from.")
    dr.restore_nominal()


def test_mass_inertia_and_subtreemass_move_together(rigged):
    """MuJoCo scales none of these for you; a mass-only change is an unphysical body."""
    import domain_rand as DR

    dr, robot, _ = rigged
    m = robot.mj_model
    d = DR.draw(np.random.default_rng(7), 29, 7)
    dr.apply(d)
    b = dr.torso_id
    assert m.body_mass[b] / dr.nominal["body_mass"][b] == pytest.approx(d.mass_scale)
    assert (m.body_inertia[b] / dr.nominal["body_inertia"][b]) == pytest.approx(
        d.mass_scale), "body_inertia did not scale with body_mass"
    assert float(m.body_subtreemass[0]) == pytest.approx(float(np.sum(m.body_mass))), (
        "body_subtreemass is stale -- mj_setConst was not called, and that value is the "
        "cost-of-transport denominator")
    dr.restore_nominal()


def test_gains_are_written_by_name_not_position(rigged):
    """The 27-of-29 ordering trap, asserted in both directions."""
    import mujoco

    import domain_rand as DR

    dr, robot, _ = rigged
    m = robot.mj_model
    d = DR.draw(np.random.default_rng(7), 29, 7)
    dr.apply(d)
    act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{n}_pos")
           for n in dr.joint_names]
    by_name = np.array([m.actuator_gainprm[a, 0] for a in act])
    want = dr.nominal_kp * d.stiffness
    assert np.abs(by_name - want).max() == pytest.approx(0.0, abs=1e-9)
    # And the wrong read must be VISIBLY wrong, or this test cannot see the error it guards.
    positional = np.array([m.actuator_gainprm[i, 0] for i in range(29)])
    assert np.abs(positional - want).max() > 1.0, (
        "a positional read now agrees with the by-name one, so actuator order has come to "
        "match policy joint order and this test no longer guards anything")
    dr.restore_nominal()


def test_armature_is_dof_indexed_not_joint_indexed(rigged):
    """nv (35) exceeds nu (29) by the free joint's six, which must not be touched."""
    import domain_rand as DR

    dr, robot, _ = rigged
    m = robot.mj_model
    before = np.array(m.dof_armature, copy=True)
    dr.apply(DR.draw(np.random.default_rng(5), 29, 5))
    changed = np.flatnonzero(np.array(m.dof_armature) != before)
    assert set(changed.tolist()) <= set(dr.dof.tolist()), (
        "a DOF outside the mapped joint set changed; dof_armature is indexed by DOF and a "
        "joint id would silently perturb another joint")
    assert not (set(changed.tolist()) & set(range(6))), "the free joint's DOFs were touched"
    dr.restore_nominal()
