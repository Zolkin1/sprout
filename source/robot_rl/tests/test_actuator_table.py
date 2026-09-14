# Copyright 2026 Zachary Olkin. All rights reserved.

"""Pin every joint's motor constants, because nothing else in the repo does.

``assets/robots/g1_21j.py`` derives stiffness and damping from a per-motor-class armature
(``k = J w^2``, ``d = 2 z J w``), and derives the ACTION SCALE from
``0.125 * effort_limit / stiffness``. So a single edit to one armature or one effort limit
silently rescales an action channel -- and before this file, the entire test suite passed
either way. Changing the hip-pitch gear ratio (14.3 -> 22.5) is exactly that edit.

The table below is the resolved, per-joint truth. It is deliberately explicit rather than
recomputed from the cfg: a test that re-derives the same numbers the same way cannot catch
a wrong number, only a crash.

**When a motor spec legitimately changes, this test SHOULD fail.** Update the table in the
same commit, and read the printed action-scale delta before you do -- that number is what
every existing checkpoint was trained against.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import pytest

from motion_gen.features import _REPO_ROOT
from motion_gen.joint_names import G1_JOINT_NAMES
from robot_rl.assets.robots.g1_21j import (
    ARMATURE_5020,
    ARMATURE_7520_14,
    ARMATURE_7520_22,
    G1_ACTION_SCALE,
    G1_CFG,
)

#: joint -> (effort_limit_sim, velocity_limit_sim, stiffness, damping, armature).
#: Stiffness/damping are quoted to 6 dp; they are exactly ``J w^2`` and ``2 z J w`` with
#: ``w = 2 pi 10 Hz`` and ``z = 2``.
ACTUATORS: dict[str, tuple[float, float, float, float, float]] = {
    # legs: hip yaw alone on the 7520_14 class; hip pitch, hip roll and knee on 7520_22.
    "left_hip_pitch_joint": (139.0, 20.0, 99.098428, 6.308802, ARMATURE_7520_22),
    "left_hip_roll_joint": (139.0, 20.0, 99.098428, 6.308802, ARMATURE_7520_22),
    "left_hip_yaw_joint": (88.0, 32.0, 40.179238, 2.557890, ARMATURE_7520_14),
    "left_knee_joint": (139.0, 20.0, 99.098428, 6.308802, ARMATURE_7520_22),
    "right_hip_pitch_joint": (139.0, 20.0, 99.098428, 6.308802, ARMATURE_7520_22),
    "right_hip_roll_joint": (139.0, 20.0, 99.098428, 6.308802, ARMATURE_7520_22),
    "right_hip_yaw_joint": (88.0, 32.0, 40.179238, 2.557890, ARMATURE_7520_14),
    "right_knee_joint": (139.0, 20.0, 99.098428, 6.308802, ARMATURE_7520_22),
    # feet: twice the 5020 class.
    "left_ankle_pitch_joint": (35.0, 30.0, 28.501246, 1.814446, 2.0 * ARMATURE_5020),
    "left_ankle_roll_joint": (35.0, 30.0, 28.501246, 1.814446, 2.0 * ARMATURE_5020),
    "right_ankle_pitch_joint": (35.0, 30.0, 28.501246, 1.814446, 2.0 * ARMATURE_5020),
    "right_ankle_roll_joint": (35.0, 30.0, 28.501246, 1.814446, 2.0 * ARMATURE_5020),
    # waist.
    "waist_yaw_joint": (88.0, 32.0, 40.179238, 2.557890, ARMATURE_7520_14),
    # kd AUTHORED at 5.0 (2026-09-13), the one joint off the zeta=2.0 law -- see
    # g1_21j.WAIST_ROLL_DAMPING. The zeta=2.0 value was 2.557890.
    "waist_roll_joint": (35.0, 30.0, 40.179238, 5.0, ARMATURE_7520_14),
    "waist_pitch_joint": (35.0, 30.0, 40.179238, 2.557890, ARMATURE_7520_14),
    # arms + wrists: the 5020 class throughout.  Wrist pitch/yaw are the light 5 N*m
    # motors and also turn slower (22 rad/s) than everything else on that class.
    "left_shoulder_pitch_joint": (25.0, 37.0, 14.250623, 0.907223, ARMATURE_5020),
    "left_shoulder_roll_joint": (25.0, 37.0, 14.250623, 0.907223, ARMATURE_5020),
    "left_shoulder_yaw_joint": (25.0, 37.0, 14.250623, 0.907223, ARMATURE_5020),
    "left_elbow_joint": (25.0, 37.0, 14.250623, 0.907223, ARMATURE_5020),
    "left_wrist_roll_joint": (25.0, 37.0, 14.250623, 0.907223, ARMATURE_5020),
    "left_wrist_pitch_joint": (5.0, 22.0, 14.250623, 0.907223, ARMATURE_5020),
    "left_wrist_yaw_joint": (5.0, 22.0, 14.250623, 0.907223, ARMATURE_5020),
    "right_shoulder_pitch_joint": (25.0, 37.0, 14.250623, 0.907223, ARMATURE_5020),
    "right_shoulder_roll_joint": (25.0, 37.0, 14.250623, 0.907223, ARMATURE_5020),
    "right_shoulder_yaw_joint": (25.0, 37.0, 14.250623, 0.907223, ARMATURE_5020),
    "right_elbow_joint": (25.0, 37.0, 14.250623, 0.907223, ARMATURE_5020),
    "right_wrist_roll_joint": (25.0, 37.0, 14.250623, 0.907223, ARMATURE_5020),
    "right_wrist_pitch_joint": (5.0, 22.0, 14.250623, 0.907223, ARMATURE_5020),
    "right_wrist_yaw_joint": (5.0, 22.0, 14.250623, 0.907223, ARMATURE_5020),
}

URDF_PATH = (_REPO_ROOT / "robot_assets" / "g1" / "g1_21j"
             / "g1_21j_box_merged_29j_mode11.urdf")

#: The MuJoCo model the OBELISK sim drives the robot with. Its actuator gains are checked
#: below because they are the one copy of the PD gains nothing else overrides -- see
#: test_mjcf_actuator_gains_match_the_cfg.
MJCF_PATH = _REPO_ROOT / "transfer" / "sim" / "robots" / "g1" / "g1_perceptive.xml"

#: Joints where the cfg deliberately disagrees with the URDF, and why.  The cfg is the one
#: PhysX uses; the URDF value only reaches the sim through the USD's joint limits, which the
#: actuator cfg overrides.
#:
#: EMPTY, and that is the point.  It used to carry hip roll, where our URDF said 88 N*m and
#: the cfg 139 -- the URDF was the stale side.  ``apply_mode11_deltas.py`` reconciled it, so
#: the cfg and the asset now agree on every joint with nothing recorded as an exception.
EFFORT_OVERRIDES_URDF: dict[str, tuple[float, float]] = {}


def _resolve(pattern_map, joint: str):
    """The value a per-joint-regex cfg field resolves to for ``joint`` (or the scalar)."""
    if not isinstance(pattern_map, dict):
        return pattern_map
    for pat, val in pattern_map.items():
        if re.fullmatch(pat, joint):
            return val
    return None


@pytest.fixture(scope="module")
def resolved() -> dict[str, tuple]:
    """``{joint: (effort, velocity, stiffness, damping, armature)}`` resolved from ``G1_CFG``."""
    out: dict[str, tuple] = {}
    for act in G1_CFG.actuators.values():
        for joint in G1_JOINT_NAMES:
            if not any(re.fullmatch(p, joint) for p in act.joint_names_expr):
                continue
            assert joint not in out, f"{joint} is claimed by two actuator groups"
            out[joint] = (
                _resolve(act.effort_limit_sim, joint), _resolve(act.velocity_limit_sim, joint),
                _resolve(act.stiffness, joint), _resolve(act.damping, joint),
                _resolve(act.armature, joint),
            )
    return out


def test_every_joint_is_claimed_by_exactly_one_actuator_group(resolved):
    """An unclaimed joint gets no PD gains at all, and nothing else would say so."""
    assert set(resolved) == set(G1_JOINT_NAMES)
    assert set(ACTUATORS) == set(G1_JOINT_NAMES)


@pytest.mark.parametrize("joint", sorted(ACTUATORS))
def test_motor_constants_match_the_table(joint, resolved):
    fields = ("effort_limit_sim", "velocity_limit_sim", "stiffness", "damping", "armature")
    for name, want, got in zip(fields, ACTUATORS[joint], resolved[joint]):
        assert got == pytest.approx(want, rel=1e-6), f"{joint}.{name}: {got} != {want}"


#: Joints whose damping is AUTHORED, so the ``d = 2 z J w`` half of the law does not apply.
#: Listed here rather than skipped inside the test so the exception is a value someone has to
#: edit, not a condition that quietly covers a future joint too. Mirrors
#: ``g1_21j.WAIST_ROLL_DAMPING`` and ``make_g1_mjcf.JOINT_DAMPING_OVERRIDE``; all three must
#: agree, which is what `test_the_mjcf_gains_match_the_rl_table` below actually enforces.
DAMPING_EXCEPTIONS: dict[str, float] = {"waist_roll_joint": 5.0}


def test_the_derived_gain_formula_still_holds(resolved):
    """``k = J w^2`` and ``d = 2 z J w`` with w = 2 pi 10 Hz, z = 2 -- the whole gain table
    is one number per motor class, so an armature edit moves stiffness AND damping.

    STIFFNESS is still universal. DAMPING has exactly the exceptions in
    :data:`DAMPING_EXCEPTIONS`, each pinned to its authored value here, so raising one joint's
    kd stays a deliberate edit in three files rather than a drift that no test notices.
    """
    w = 10 * 2.0 * 3.1415926535
    for joint, (_e, _v, k, d, j) in resolved.items():
        assert k == pytest.approx(j * w ** 2, rel=1e-9), joint
        want = DAMPING_EXCEPTIONS.get(joint, 2.0 * 2.0 * j * w)
        assert d == pytest.approx(want, rel=1e-9), joint


def test_the_damping_exceptions_are_all_real_joints():
    """A typo in :data:`DAMPING_EXCEPTIONS` would silently exempt nothing and pin nothing."""
    assert set(DAMPING_EXCEPTIONS) <= set(G1_JOINT_NAMES), (
        f"unknown joint(s) {sorted(set(DAMPING_EXCEPTIONS) - set(G1_JOINT_NAMES))}")


@pytest.mark.parametrize("joint", sorted(ACTUATORS))
def test_action_scale_matches_effort_over_stiffness(joint, resolved):
    """``G1_ACTION_SCALE`` is ``0.125 * effort / stiffness``, keyed by regex.

    This is the number that makes a gear-ratio change invalidate a checkpoint rather than
    merely mistune it: the same network output means a different joint target afterwards.
    """
    effort, _v, stiffness, _d, _a = resolved[joint]
    matches = [v for pat, v in G1_ACTION_SCALE.items() if re.fullmatch(pat, joint)]
    assert matches, f"{joint} matches no G1_ACTION_SCALE pattern -- the action would be unscaled"
    assert len(set(matches)) == 1, f"{joint} matches conflicting scales {set(matches)}"
    assert matches[0] == pytest.approx(0.125 * effort / stiffness, rel=1e-9)


def test_effort_limits_track_the_urdf_except_where_recorded(resolved):
    """The cfg overrides the USD's joint limits, so a silent disagreement is invisible in sim."""
    root = ET.parse(URDF_PATH).getroot()
    urdf_effort = {
        j.get("name"): float(j.find("limit").get("effort"))
        for j in root.findall("joint") if j.get("type") == "revolute"
    }
    for joint, (effort, *_rest) in resolved.items():
        want_urdf, want_cfg = EFFORT_OVERRIDES_URDF.get(joint, (None, None))
        if want_urdf is not None:
            assert urdf_effort[joint] == pytest.approx(want_urdf), (
                f"{joint}: the URDF changed to {urdf_effort[joint]} -- if it now agrees with the "
                "cfg, drop it from EFFORT_OVERRIDES_URDF"
            )
            assert effort == pytest.approx(want_cfg)
            continue
        assert effort == pytest.approx(urdf_effort[joint]), (
            f"{joint}: cfg {effort} N*m vs URDF {urdf_effort[joint]} N*m. Either fix one, or "
            "record it in EFFORT_OVERRIDES_URDF with the reason."
        )


def test_velocity_limits_track_the_urdf(resolved):
    """The cfg's velocity limits are the URDF's, joint for joint.

    This check used to be a PIN rather than a comparison: our pre-mode_11 URDF carried a
    blanket 12 rad/s (22 on the wrists) that no motor has, so the two could not be compared
    and the test recorded the gap instead, saying to convert it if the URDF's values ever
    became real. mode_11's are real -- 20 on the 22.5-gear legs, 32 on hip yaw and waist
    yaw, 30 on the ankles and waist roll/pitch, 37 on the arms, 22 on the light wrists --
    so it is now a comparison, and the third copy of these numbers has a guard.
    """
    root = ET.parse(URDF_PATH).getroot()
    urdf_vel = {
        j.get("name"): float(j.find("limit").get("velocity"))
        for j in root.findall("joint") if j.get("type") == "revolute"
    }
    assert len(set(urdf_vel.values())) > 2, (
        f"the URDF's velocity limits collapsed to {sorted(set(urdf_vel.values()))} -- that is "
        "the pre-mode_11 blanket, so this file is the wrong asset"
    )
    for joint, (_e, vel, *_rest) in sorted(resolved.items()):
        assert vel == pytest.approx(urdf_vel[joint]), (
            f"{joint}: the cfg says {vel} rad/s and {URDF_PATH.name} says {urdf_vel[joint]}. "
            "PhysX uses the cfg, so this divergence would be silent."
        )


def _mjcf_actuator_gains() -> dict[str, tuple[float, float]]:
    """``{joint: (kp, kv)}`` as ``g1_perceptive.xml`` will actually apply them.

    Resolved the way MuJoCo resolves them: an attribute on the ``<position>`` element wins,
    otherwise the ``<default class=...>`` it names supplies it.
    """
    root = ET.parse(MJCF_PATH).getroot()
    classes = {
        d.get("class"): (p.get("kp"), p.get("kv"))
        for d in root.iter("default") if d.get("class")
        for p in d.findall("position")
    }
    out: dict[str, tuple[float, float]] = {}
    for actuator in root.iter("actuator"):
        for position in actuator.findall("position"):
            kp, kv = classes.get(position.get("class"), (None, None))
            kp = position.get("kp", kp)
            kv = position.get("kv", kv)
            assert kp is not None and kv is not None, (
                f"{position.get('joint')} has no resolvable kp/kv in {MJCF_PATH.name}")
            out[position.get("joint")] = (float(kp), float(kv))
    return out


def test_mjcf_actuator_gains_match_the_cfg(resolved):
    """The obelisk sim's PD gains must be the ones the policy was trained with.

    This is the ONE copy of the gains nothing else overrides. ``transfer/sim`` replaces them
    from the policy (``Robot.set_pd_gains_from_policy``) and the hardware driver takes
    ``kp``/``kd`` off the wire, but obelisk's MuJoCo sim writes ``u_mujoco`` straight into
    ``data.ctrl`` and never reads the published gains -- so the MJCF alone decides how stiff
    the robot is there.

    It went stale exactly the way this test now prevents. ``make_g1_mjcf.py`` copies
    ``<default>`` verbatim from its template (``g1_29j.xml``, the PREVIOUS robot), so when
    mode_11 moved hip pitch from the 14.3 gear to the 22.5 the armature was updated and the
    actuator gains were not: kp 40.18 against the 99.10 the policy exports, a hip pitch 2.46x
    too soft on the joint that drives fore/aft leg swing. Every shape check passed, sim2sim
    was unaffected, and it presented only as obelisk tracking worse than sim2sim.

    Tolerance is 0.5%: the MJCF stores the gains to 6 significant figures.
    """
    mjcf = _mjcf_actuator_gains()
    assert set(mjcf) == set(resolved), (
        "the MJCF actuates a different joint set than the cfg: "
        f"only in MJCF {sorted(set(mjcf) - set(resolved))}, "
        f"only in cfg {sorted(set(resolved) - set(mjcf))}")
    for joint, (_e, _v, stiffness, damping, _a) in sorted(resolved.items()):
        kp, kv = mjcf[joint]
        assert kp == pytest.approx(stiffness, rel=5e-3), (
            f"{joint}: {MJCF_PATH.name} has kp={kp} but the cfg trained at {stiffness}. "
            "Regenerate with transfer/sim/robots/g1/make_g1_mjcf.py, which derives the gains "
            "from JOINT_ARMATURE rather than copying them from the template.")
        assert kv == pytest.approx(damping, rel=5e-3), (
            f"{joint}: {MJCF_PATH.name} has kv={kv} but the cfg trained at {damping}.")
