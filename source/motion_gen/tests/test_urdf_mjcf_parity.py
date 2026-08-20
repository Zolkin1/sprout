# Copyright 2026 Zachary Olkin. All rights reserved.

"""The Isaac URDF and the MuJoCo MJCF must describe the same robot.

There are two descriptions of the G1 in this repo:

* ``robot_assets/g1_perceptive/g1_perceptive.urdf`` -> the USD Isaac spawns;
* ``transfer/sim/robots/g1/g1_perceptive.xml`` -> MuJoCo sim2sim, and ALSO
  :data:`motion_gen.features.DEFAULT_MJCF`, i.e. the model every clip library's
  ``<body>@root`` and world end-effector channel is forward-kinematicked through.

The second point is what makes this more than tidiness: a disagreement between the two is
train/serve skew that shows up as a tracking error nobody can attribute.

**The MJCF is now GENERATED from the URDF** (``make_g1_mjcf.py``), so these are no longer
two hand-maintained files that happen to agree -- and this file's job changed with them.
It used to pin a table of known divergences: a 2.62 kg mass gap, a head counted twice
across the two sims, the palms' density-derived mass, the URDF lumping the hand into the
wrist while the MJCF spread it down 16 finger links, and one blanket ``armature="0.01"``
against Isaac's four per-motor values. Every one of those is gone. What is left is the
strict form: the same 39 bodies, the same masses to a milligram, and identical FK.

Regenerate with::

    python transfer/sim/robots/g1/make_g1_mjcf.py \\
        --urdf robot_assets/g1_perceptive/g1_perceptive.urdf \\
        --template transfer/sim/robots/g1/g1_29j.xml \\
        --out transfer/sim/robots/g1/g1_perceptive.xml

FK is compared by compiling the URDF with MuJoCo's own URDF front-end and running BOTH
models through :class:`motion_gen.fk.FKChain` -- the same parser the clip generator uses.
The compile is hermetic: ``discardvisual="true"`` means no STL is read, so this needs no
mesh directory and takes milliseconds.
"""

from __future__ import annotations

import importlib.util
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

URDF_PATH = str(F._REPO_ROOT / "robot_assets" / "g1_perceptive" / "g1_perceptive.urdf")

#: Compiler flags that make MuJoCo's URDF front-end preserve the link tree.
#: ``fusestatic="false"`` is load-bearing -- without it MuJoCo merges every fixed joint and
#: the 46-link model collapses to 30, silently taking the hands with it.
_URDF_MUJOCO_BLOCK = (
    '  <mujoco><compiler fusestatic="false" discardvisual="true" strippath="true"/></mujoco>\n'
)

#: Per-joint reflected inertia the generator writes, keyed by joint-name substring.
#:
#: Imported from the generator rather than restated: the MJCF used to carry ONE blanket
#: ``armature="0.01"`` against Isaac's four per-motor-class values -- up to a 2.5x sim2sim
#: mismatch that nothing reported. ``assets/robots/g1_21j.py`` holds the Isaac side and
#: ``source/robot_rl/tests/test_actuator_table.py`` pins it there.
_MJCF_GEN = F._REPO_ROOT / "transfer" / "sim" / "robots" / "g1" / "make_g1_mjcf.py"
_spec = importlib.util.spec_from_file_location("_make_g1_mjcf", _MJCF_GEN)
_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen)
JOINT_ARMATURE = _gen.JOINT_ARMATURE


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
    """``fusestatic="false"`` is what keeps the fixed-joint links from being welded away.

    Everything bolted to this robot is a fixed joint -- both rubber hands, the two camera
    mounts, the compute backpack and its two payload masses. Fuse them and the depth cameras
    stop seeing the hands (and the ray caster has no prim to target), while the backpack's
    2.64 kg silently moves into the torso's inertial. The hands are the canary because they
    are leaves that the self-scan explicitly targets.
    """
    with tempfile.TemporaryDirectory() as tmp:
        model = mujoco.MjModel.from_xml_path(_urdf_as_mjcf_path(tmp))
    assert model.njnt == len(G1_JOINT_NAMES)
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)}
    assert {"left_rubber_hand", "right_rubber_hand"} <= names, "the hands were fused away"
    for bolted in ("camera_lower_mount_link", "camera_upper_mount_link",
                   "backpack_shell_link", "backpack_thor_link", "backpack_zed_link"):
        assert bolted in names, f"{bolted} was fused into its parent"


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
# Mass. The two models are now the same file compiled twice; they must agree outright.
##


def test_total_mass_agrees(facts):
    """No exemption list. There used to be one worth 2.62 kg (7.2%)."""
    urdf_total = sum(facts["urdf_mass"].values())
    mjcf_total = sum(facts["mjcf_mass"].values())
    assert urdf_total == pytest.approx(mjcf_total, abs=1e-4), (
        f"URDF {urdf_total:.6f} kg vs MJCF {mjcf_total:.6f} kg"
    )


def test_per_body_mass_agrees(facts):
    """Every body, to a milligram -- the generator copies the inertial across verbatim."""
    for name, m in facts["urdf_mass"].items():
        other = facts["mjcf_mass"][name]
        assert m == pytest.approx(other, abs=1e-6), f"{name}: URDF {m} vs MJCF {other}"


def test_the_body_name_sets_are_IDENTICAL(facts):
    """The trackable-frame ceiling used to be the INTERSECTION of two models. Now it is both.

    The old divergence was three bodies: the two palms, which MuJoCo named ``*_palm_link``
    while the URDF said ``*_hand_palm_link``, and ``head_link``, which existed only in the
    MJCF because the URDF had folded the head into ``torso_link`` when it merged fixed
    joints -- so the head's 1.80 kg was counted in the torso on one side and separately on
    the other. Generating the MJCF from the URDF removes both by construction.
    """
    urdf_names, mjcf_names = set(facts["urdf_mass"]), set(facts["mjcf_mass"])
    assert urdf_names == mjcf_names, (
        f"only in the URDF: {sorted(urdf_names - mjcf_names)}; "
        f"only in the MJCF: {sorted(mjcf_names - urdf_names)}"
    )
    assert set(G1_BODY_TRACK_FRAMES) <= urdf_names, (
        "a tracked frame is missing from the models: the clip generator FKs it from the "
        "MJCF while the RL side resolves it through robot.body_names"
    )


def test_no_body_derives_its_mass_from_mesh_volume(facts):
    """A body with no ``<inertial>`` gets volume x 1000 kg/m^3 -- water, not aluminium.

    Three did, worth 2.62 kg between them, and the number reached sim2sim unremarked. The
    generator now carries an explicit inertial for every body, so an omission is a real
    modelling gap rather than a silent default.
    """
    missing = [n for n in facts["mjcf_mass"] if facts["mjcf"][0].get(n) is None]
    assert missing == [], f"no <inertial> on {missing}; MuJoCo will invent one from density"


##
# Actuator dynamics -- the half MuJoCo needs and the URDF cannot express.
##


def test_the_mjcf_carries_per_joint_armature(facts):
    """Isaac gives each motor class its own reflected inertia; the MJCF must match it.

    It used to set a single ``armature="0.01"`` on the ``g1`` default class, against Isaac's
    0.0036-0.0251 kg m^2 -- up to 2.5x on the joints that matter most, and a sim2sim
    divergence nothing else reported.
    """
    root = ET.parse(F.DEFAULT_MJCF).getroot()
    seen = {j.get("name"): j.get("armature")
            for b in root.iter("body") for j in b.findall("joint")}
    for name in G1_JOINT_NAMES:
        assert name in seen, f"{name} is not a hinge in the MJCF"
        expected = next((v for k, v in JOINT_ARMATURE.items() if k in name), None)
        assert expected is not None, f"the generator has no armature rule for {name}"
        assert seen[name] is not None, f"{name} fell back to the blanket class armature"
        assert float(seen[name]) == pytest.approx(expected, rel=1e-9), name
