# Copyright 2026 Zachary Olkin. All rights reserved.

"""The Isaac URDF and the MuJoCo MJCF must describe the same robot.

There are two independent descriptions of the G1 in this repo and nothing kept them in
step:

* ``robot_assets/g1/g1_21j/g1_21j_box_merged_29j_hands.urdf`` -> the USD Isaac spawns;
* ``transfer/sim/robots/g1/g1_29j.xml`` -> MuJoCo sim2sim, and ALSO
  :data:`motion_gen.features.DEFAULT_MJCF`, i.e. the model every clip library's
  ``<body>@root`` and world end-effector channel is forward-kinematicked through.

The second point is what makes this more than tidiness: a disagreement between the two is
train/serve skew that shows up as a tracking error nobody can attribute. These tests pin
the agreement, and pin the disagreements that are real and known so that closing one is a
deliberate act rather than a surprise.

FK is compared by compiling the URDF with MuJoCo's own URDF front-end and running BOTH
models through :class:`motion_gen.fk.FKChain` -- the same parser the clip generator uses.
The compile is hermetic: ``discardvisual="true"`` means no STL is read, so this needs no
mesh directory and takes milliseconds.
"""

from __future__ import annotations

import os
import re
import tempfile
import xml.etree.ElementTree as ET

import pytest
import torch

from motion_gen import features as F
from motion_gen.fk import FKChain
from motion_gen.joint_names import G1_BODY_TRACK_FRAMES, G1_JOINT_NAMES

mujoco = pytest.importorskip("mujoco")

URDF_PATH = str(
    F._REPO_ROOT / "robot_assets" / "g1" / "g1_21j" / "g1_21j_box_merged_29j_hands.urdf"
)

#: Compiler flags that make MuJoCo's URDF front-end preserve the link tree.
#: ``fusestatic="false"`` is load-bearing -- without it MuJoCo merges every fixed joint and
#: the 46-link model collapses to 30, silently taking the hands with it.
_URDF_MUJOCO_BLOCK = (
    '  <mujoco><compiler fusestatic="false" discardvisual="true" strippath="true"/></mujoco>\n'
)

#: MuJoCo names the palm ``*_palm_link``; the URDF/USD name it ``*_hand_palm_link``. Documented
#: at ``motion_gen/joint_names.py``, and the reason the palms are excluded from
#: ``G1_BODY_TRACK_FRAMES`` -- the trackable set is the INTERSECTION of the two models.
PALM_RENAMES = {"left_hand_palm_link": "left_palm_link",
                "right_hand_palm_link": "right_palm_link"}

#: Links whose mass the two models deliberately distribute differently: the URDF lumps the
#: whole hand into the wrist and gives the 14 finger links a token 1e-6 kg (see
#: ``make_hand_links_urdf.py`` -- they exist only so the ray caster can see them), while the
#: MJCF gives each finger a real mass. The TOTAL is what must agree, and
#: :func:`test_the_hand_lumping_conserves_mass` checks exactly that.
_HAND_LUMPED = {f"{side}_{part}" for side in ("left", "right") for part in (
    "wrist_yaw_link", "hand_palm_link",
    "hand_thumb_0_link", "hand_thumb_1_link", "hand_thumb_2_link",
    "hand_middle_0_link", "hand_middle_1_link",
    "hand_index_0_link", "hand_index_1_link",
)}

#: MEASURED mass the MJCF has and the URDF does not, in kg, per MuJoCo-resolved body.
#:
#: All three of these bodies carry NO ``<inertial>`` in the MJCF, so MuJoCo derives their
#: mass from mesh volume times the default density of 1000 kg/m^3 -- which is a modelling
#: accident for anything that is not solid water.
#:
#: * the two palms are known and deliberate (``g1_29j.xml`` documents that setting
#:   ``density="0"`` there "would silently make the palms massless");
#: * ``head_link`` is NOT. The URDF merged every fixed joint, folding the head into
#:   ``torso_link``, so Isaac carries the head's mass in the torso while MuJoCo adds a
#:   separate 1.80 kg head on top -- it is counted twice across the two sims.
#:
#: Together they make MuJoCo's G1 **2.62 kg (7.2%) heavier than Isaac's**. Recorded here
#: rather than silently corrected: changing either model's mass changes every sim2sim
#: number already measured, so it is a deliberate decision, not a test fix.
KNOWN_MJCF_ONLY_MASS_KG = {
    "head_link": 1.8025,
    "left_palm_link": 0.41119,
    "right_palm_link": 0.41119,
}


def _urdf_as_mjcf_path(tmpdir: str) -> str:
    """Compile the live URDF through MuJoCo and save the resulting MJCF.

    Args:
        tmpdir: Directory to write the intermediate files into.

    Returns:
        Path to the saved MJCF.
    """
    text = open(URDF_PATH).read()
    text, n = re.subn(r"(<robot[^>]*>\s*\n)", r"\1" + _URDF_MUJOCO_BLOCK, text, count=1)
    assert n == 1, "could not find the <robot> tag to inject the MuJoCo compiler block into"
    src = os.path.join(tmpdir, "urdf_with_mujoco_block.urdf")
    with open(src, "w") as f:
        f.write(text)
    model = mujoco.MjModel.from_xml_path(src)
    out = os.path.join(tmpdir, "from_urdf.xml")
    mujoco.mj_saveLastXML(out, model)
    return out


def _revolute_joints(urdf_path: str) -> dict[str, dict]:
    """``{joint_name: {lower, upper, effort, velocity, axis}}`` from a URDF."""
    root = ET.parse(urdf_path).getroot()
    out = {}
    for j in root.findall("joint"):
        if j.get("type") != "revolute":
            continue
        lim, ax = j.find("limit"), j.find("axis")
        out[j.get("name")] = {
            "lower": float(lim.get("lower")), "upper": float(lim.get("upper")),
            "effort": float(lim.get("effort")), "velocity": float(lim.get("velocity")),
            "axis": tuple(float(v) for v in (ax.get("xyz") if ax is not None else "0 0 1").split()),
        }
    return out


def _mjcf_bodies_and_joints(mjcf_path: str) -> tuple[dict[str, dict], dict[str, dict]]:
    """``({body: {mass, com}}, {joint: {lower, upper, axis}})`` from an MJCF."""
    root = ET.parse(mjcf_path).getroot()
    bodies, joints = {}, {}

    def walk(el):
        for b in el.findall("body"):
            inert = b.find("inertial")
            if inert is not None:
                bodies[b.get("name")] = {
                    "mass": float(inert.get("mass")),
                    "com": tuple(float(v) for v in inert.get("pos").split()),
                }
            for j in b.findall("joint"):
                if j.get("type") not in (None, "hinge"):
                    continue
                rng = j.get("range")
                joints[j.get("name")] = {
                    "lower": float(rng.split()[0]) if rng else None,
                    "upper": float(rng.split()[1]) if rng else None,
                    "axis": tuple(float(v) for v in (j.get("axis") or "0 0 1").split()),
                }
            walk(b)

    walk(root.find("worldbody"))
    return bodies, joints


def _compiled_masses(model) -> dict[str, float]:
    """``{body_name: mass}`` from a COMPILED model.

    Read off the compiled model rather than the XML because MuJoCo derives a body's mass
    from its geoms' density when it has no ``<inertial>``, and three bodies in the live MJCF
    do exactly that -- a text-level parse reports them as massless and misses 2.62 kg.
    """
    out = {}
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and name != "world":
            out[name] = float(model.body_mass[i])
    return out


@pytest.fixture(scope="module")
def facts():
    """Parsed facts from both live models, with masses from the compiled models."""
    with tempfile.TemporaryDirectory() as tmp:
        urdf_model = mujoco.MjModel.from_xml_path(_urdf_as_mjcf_path(tmp))
    return {
        "urdf_joints": _revolute_joints(URDF_PATH),
        "urdf_mass": _compiled_masses(urdf_model),
        "mjcf_mass": _compiled_masses(mujoco.MjModel.from_xml_path(F.DEFAULT_MJCF)),
        "mjcf": _mjcf_bodies_and_joints(F.DEFAULT_MJCF),
    }


##
# Kinematics -- the half that feeds every clip channel.
##


def test_forward_kinematics_agree_exactly():
    """The strong one: same joint angles, same body poses, through the same parser.

    Every ``<body>@root`` and world end-effector channel in a clip library is FK through
    ``DEFAULT_MJCF``, while the policy is trained on the USD built from the URDF. If these
    two disagree the reference the CLF grades is not the pose the robot can reach, and
    nothing else in the pipeline would say so.
    """
    frames = list(G1_BODY_TRACK_FRAMES) + list(F.FOOT_FRAMES)
    with tempfile.TemporaryDirectory() as tmp:
        from_urdf = _urdf_as_mjcf_path(tmp)
        a = FKChain(F.DEFAULT_MJCF, G1_JOINT_NAMES, frames, base_link="pelvis_link")
        b = FKChain(from_urdf, G1_JOINT_NAMES, frames, base_link="pelvis_link")
    gen = torch.Generator().manual_seed(0)
    q = (torch.rand(32, len(G1_JOINT_NAMES), generator=gen) - 0.5) * 1.4
    pa, ra, _, _ = a.pose_and_jacobian(q)
    pb, rb, _, _ = b.pose_and_jacobian(q)
    worst = float((pa - pb).abs().max())
    assert worst < 1e-6, f"pelvis-frame FK differs by up to {worst * 1000:.4f} mm"
    assert float((ra - rb).abs().max()) < 1e-6


def test_the_urdf_compiles_with_its_link_tree_intact():
    """``fusestatic="false"`` is what keeps the hands; without it 46 links become 30."""
    with tempfile.TemporaryDirectory() as tmp:
        model = mujoco.MjModel.from_xml_path(_urdf_as_mjcf_path(tmp))
    assert model.njnt == len(G1_JOINT_NAMES)
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)}
    assert len([n for n in names if "hand" in n]) == 16, "the hand links were fused away"


##
# The joint set.
##


def test_both_models_declare_the_same_joints(facts):
    assert set(facts["urdf_joints"]) == set(facts["mjcf"][1]) == set(G1_JOINT_NAMES)


def test_joint_axes_agree(facts):
    for name, u in facts["urdf_joints"].items():
        m = facts["mjcf"][1][name]
        assert u["axis"] == pytest.approx(m["axis"], abs=1e-9), name


def test_joint_limits_agree(facts):
    """To 1e-5 rad: the MJCF rounds to six significant figures (wrist roll 1.972222 -> 1.97222)."""
    for name, u in facts["urdf_joints"].items():
        m = facts["mjcf"][1][name]
        if m["lower"] is None:
            continue
        assert u["lower"] == pytest.approx(m["lower"], abs=1e-5), name
        assert u["upper"] == pytest.approx(m["upper"], abs=1e-5), name


##
# Mass. The distribution differs by design in one place; the total may not.
##


def test_total_mass_agrees_once_the_known_divergences_are_removed(facts):
    """Everything except the three density-derived bodies must add up."""
    urdf_total = sum(facts["urdf_mass"].values())
    mjcf_total = sum(facts["mjcf_mass"].values()) - sum(KNOWN_MJCF_ONLY_MASS_KG.values())
    assert urdf_total == pytest.approx(mjcf_total, abs=5e-3), (
        f"URDF {urdf_total:.4f} kg vs MJCF-minus-known {mjcf_total:.4f} kg"
    )


def test_the_mujoco_robot_is_still_exactly_2_62_kg_heavier(facts):
    """KNOWN GAP, pinned with numbers so that closing it is deliberate.

    MuJoCo's G1 weighs 38.79 kg; Isaac's weighs 36.17. The whole 2.62 kg is three bodies
    with no ``<inertial>``, whose mass MuJoCo derives from mesh volume x 1000 kg/m^3:
    both palms (documented in ``g1_29j.xml``) and ``head_link`` (not documented anywhere --
    the URDF folded the head into ``torso_link`` when it merged fixed joints, so across the
    two sims the head is counted twice).

    7.2% of body mass is not a rounding error, and it sits on the torso and the hands --
    exactly where an arm-swing or a payload change is felt. Fixing it changes every sim2sim
    number measured so far, so it belongs in the robot-model pass, not in a test edit.
    """
    for name, expected in KNOWN_MJCF_ONLY_MASS_KG.items():
        assert facts["mjcf_mass"][name] == pytest.approx(expected, abs=1e-4), name
        assert facts["mjcf"][0].get(name) is None, (
            f"{name} now has an explicit <inertial> -- good; drop it from "
            "KNOWN_MJCF_ONLY_MASS_KG and re-check the totals"
        )
    delta = sum(facts["mjcf_mass"].values()) - sum(facts["urdf_mass"].values())
    assert delta == pytest.approx(2.6219, abs=5e-3), (
        f"the URDF/MJCF mass gap moved to {delta:+.4f} kg; if that was intended, update this "
        "test and KNOWN_MJCF_ONLY_MASS_KG"
    )


def test_per_body_mass_agrees_outside_the_hand(facts):
    """Everything the two models agree to model the same way, to 5 g."""
    for name, m in facts["urdf_mass"].items():
        if name in _HAND_LUMPED:
            continue
        other = facts["mjcf_mass"].get(PALM_RENAMES.get(name, name))
        if other is None:
            continue
        assert m == pytest.approx(other, abs=5e-3), f"{name}: URDF {m} vs MJCF {other}"


def test_the_hand_lumping_conserves_mass_apart_from_the_palm(facts):
    """The URDF lumps the hand into the wrist; the MJCF distributes it along the fingers.

    Same total per side once the density-derived palm is set aside -- so Isaac carries the
    hand's mass at the wrist and MuJoCo carries it out at the fingers. The mass agrees; the
    arm's INERTIA does not. Small, real, and worth knowing before blaming a tracker.
    """
    for side in ("left", "right"):
        urdf_side = sum(m for n, m in facts["urdf_mass"].items()
                        if n in _HAND_LUMPED and n.startswith(side))
        mjcf_side = sum(m for n, m in facts["mjcf_mass"].items()
                        if n.startswith(side) and ("hand" in n or n.endswith("wrist_yaw_link")))
        assert urdf_side == pytest.approx(mjcf_side, abs=1e-4), (
            f"{side} wrist+fingers: URDF {urdf_side:.6f} kg vs MJCF {mjcf_side:.6f} kg"
        )


##
# Known, deliberate divergences. Each is a tripwire: closing one must fail this file.
##


def test_the_body_name_divergence_is_exactly_the_palms_and_the_head(facts):
    """The trackable-frame ceiling is the INTERSECTION of the two models' body names."""
    urdf_names, mjcf_names = set(facts["urdf_mass"]), set(facts["mjcf_mass"])
    only_urdf = urdf_names - mjcf_names
    only_mjcf = mjcf_names - urdf_names
    assert only_urdf == set(PALM_RENAMES), f"new URDF-only body: {only_urdf}"
    assert only_mjcf == set(PALM_RENAMES.values()) | {"head_link"}, (
        f"new MJCF-only body: {only_mjcf - set(PALM_RENAMES.values()) - {'head_link'}}"
    )
    assert not ((set(PALM_RENAMES) | only_mjcf) & set(G1_BODY_TRACK_FRAMES)), (
        "a body that exists in only one model must not be in the tracked set -- the clip "
        "generator FKs it from the MJCF while the RL side resolves it through robot.body_names"
    )


def test_the_mjcf_still_uses_one_blanket_armature():
    """KNOWN GAP, pinned so that fixing it is deliberate.

    Isaac gives each motor class its own reflected inertia (0.0036 - 0.0251 kg m^2, see
    ``assets/robots/g1_21j.py``); the MJCF sets a single ``armature="0.01"`` on the ``g1``
    default class and no per-joint override. That is up to a 2.5x mismatch on the joints
    that matter most, and it is a sim2sim divergence nothing else reports.

    When the MJCF learns per-joint armature, this test SHOULD fail -- update it then.
    """
    root = ET.parse(F.DEFAULT_MJCF).getroot()
    per_joint = [j.get("armature") for b in root.iter("body") for j in b.findall("joint")]
    assert set(per_joint) == {None}, (
        "the MJCF now sets armature per joint -- good; update this test and check the values "
        "against assets/robots/g1_21j.py"
    )
    defaults = {d.get("class"): d.find("joint") for d in root.iter("default")}
    g1 = defaults.get("g1")
    assert g1 is not None and g1.get("armature") == "0.01"
