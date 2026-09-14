# Copyright 2026 Zachary Olkin. All rights reserved.

"""Generate the MuJoCo body tree from the URDF, instead of maintaining it by hand.

Run from the repo root::

    python transfer/sim/robots/g1/make_g1_mjcf.py \\
        --urdf robot_assets/g1_perceptive/g1_perceptive.urdf \\
        --template transfer/sim/robots/g1/g1_29j.xml \\
        --out transfer/sim/robots/g1/g1_perceptive.xml

There are two descriptions of this robot's kinematics -- a URDF that IsaacLab converts to
USD, and an MJCF that MuJoCo sim2sim and ``motion_gen``'s FK read -- and until now both were
edited by hand. Every clip in the library carries ``<body>@root`` channels computed by FK
through the MJCF, so a divergence between the two does not announce itself: it shows up as a
tracker that follows a reference nobody can reproduce.

**MuJoCo compiles the URDF itself.** The body tree, the mesh assets and the inertials all
come out of ``mujoco.MjModel.from_xml_path`` plus ``mj_saveLastXML``, so there is no
translation layer here to be wrong. Three compiler flags make that work, and each one is
load-bearing:

* ``discardvisual="false"`` -- otherwise every visual mesh is thrown away, and with it the
  self-occlusion the depth cameras depend on (``transfer/sim/depth_camera.py`` casts geom
  group 1, which IS the visual meshes);
* ``fusestatic="false"`` -- otherwise the fixed-joint links are welded into their parents,
  and the hands stop being bodies at all;
* ``balanceinertia="true"`` -- tolerates the token inertias on the frame-only links.

**The output is ONE self-contained file, not a template plus an include.** MuJoCo supports
``<mujocoinclude>`` and it would be the tidier split, but ``motion_gen.fk._parse_mjcf``
reads the MJCF with ``xml.etree`` and does not follow includes -- it would find no
``<worldbody>`` and every FK consumer would break. Verified, not assumed.

**What carries over from the template, and what does not.** Everything outside
``<worldbody>`` -- the servo gain classes, actuators, sensors, keyframe, options -- is
copied verbatim, so none of it is transcribed here and none of it can drift by being
retyped. From inside the worldbody only the SITES come across, re-attached to the same
bodies by name. Geoms, bodies, joints and inertials are the URDF's.

That drops the ``nocontact`` class as a concept: it existed because the hand-maintained
MJCF gave the palm and finger visual meshes real colliders, which the training USD does not
have. A URDF-derived model cannot make that mistake, because a ``<visual>`` compiles to
``contype="0" conaffinity="0" group="1" density="0"`` and only a ``<collision>`` collides.

**Joint dynamics are per-joint, not a blanket.** The template's ``class="g1"`` sets
``armature="0.01"`` on all 29 joints while IsaacLab uses 0.0036 to 0.0251 depending on the
motor -- a standing sim2sim mismatch of up to 2.5x on the term that most affects how a joint
responds. :data:`JOINT_ARMATURE` states the real values and
``source/robot_rl/tests/test_urdf_mjcf_parity.py`` checks them against the Isaac actuator
config, which is the only copy that can be called authoritative.
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import tempfile
import xml.etree.ElementTree as ET

import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))

#: Compiler settings injected into the URDF before MuJoCo reads it. See the module
#: docstring: dropping any of the three silently removes something the sim depends on.
URDF_COMPILER_ATTRS: dict[str, str] = {
    "meshdir": ".",
    "discardvisual": "false",
    "fusestatic": "false",
    "strippath": "false",
    "balanceinertia": "true",
}

#: The floating base. MuJoCo's URDF path gives the root body no joint, so the robot would be
#: welded to the world; the template's hand-written tree carried this explicitly.
ROOT_BODY = "pelvis_link"
FREE_JOINT_NAME = "floating_base_joint"

#: Rotor inertia per joint, in kg m^2, reflected through the gearbox -- the same constants
#: IsaacLab's actuator config uses, where they are named ARMATURE_*. Stated here because
#: ``transfer/`` cannot import ``robot_rl`` (it must run without IsaacLab), and CHECKED
#: against it by test_urdf_mjcf_parity so the two cannot drift.
#:
#: The template applied a blanket 0.01 to everything. For a hip pitch on the 22.5 gear that
#: is 2.5x too small.
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925

JOINT_ARMATURE: dict[str, float] = {
    **{f"{s}_hip_yaw_joint": ARMATURE_7520_14 for s in ("left", "right")},
    **{f"{s}_hip_roll_joint": ARMATURE_7520_22 for s in ("left", "right")},
    **{f"{s}_hip_pitch_joint": ARMATURE_7520_22 for s in ("left", "right")},
    **{f"{s}_knee_joint": ARMATURE_7520_22 for s in ("left", "right")},
    **{f"{s}_ankle_pitch_joint": 2.0 * ARMATURE_5020 for s in ("left", "right")},
    **{f"{s}_ankle_roll_joint": 2.0 * ARMATURE_5020 for s in ("left", "right")},
    "waist_yaw_joint": ARMATURE_7520_14,
    "waist_roll_joint": ARMATURE_7520_14,
    "waist_pitch_joint": ARMATURE_7520_14,
    **{f"{s}_{j}_joint": ARMATURE_5020 for s in ("left", "right")
       for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                 "wrist_roll", "wrist_pitch", "wrist_yaw")},
}

#: The PD gain law, matching ``robot_rl.assets.robots.g1_21j``: ``kp = J w^2`` and
#: ``kd = 2 z J w`` with ``w = 2 pi 10 Hz`` and ``z = 2``. One number per motor class --
#: :data:`JOINT_ARMATURE` -- decides both, which is why the gains are DERIVED here rather
#: than carried as their own table. ``test_actuator_table.py`` pins the result against the
#: RL side, which is the definition.
NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0

#: Joints whose kd is AUTHORED rather than derived from the law above, mirroring
#: ``robot_rl.assets.robots.g1_21j.WAIST_ROLL_DAMPING``.  Read that constant for why.
#:
#: This table exists because the generator DERIVES gains instead of importing them -- it must
#: stay `motion_gen`-free and cannot reach the RL package -- so a per-joint override on the RL
#: side would otherwise never reach MuJoCo and sim2sim would quietly run a different plant.
#: That is not hypothetical: `3ddab280` fixed exactly this, two hip pitches left 2.46x too soft
#: because the MJCF was maintained separately.  `test_actuator_table.py` is what keeps the two
#: honest, and it compares against the RL side, which is the definition.
JOINT_DAMPING_OVERRIDE: dict[str, float] = {
    "waist_roll_joint": 5.0,
}

#: Bodies whose visual mesh must NOT be ray-cast, and the group to park it in.
#:
#: ``transfer/sim/depth_camera.py`` casts geom groups {1 = per-link visual meshes,
#: 2 = terrain}, so anything left in group 1 appears in the depth image. On the Isaac side
#: the equivalent decision is ``g1_self_scan.UNSCANNED_LINKS``, and the two MUST agree:
#: a body cast by one sim and not the other is a pixel-level divergence in the exact signal
#: the generator uses to place a reference, which is the class of bug the hands cost a
#: debugging cycle for.
#:
#: The camera mounts are excluded because a camera has no use for a picture of its own
#: housing, and they are the heaviest meshes on the robot. Group 3 rather than 0: group 0
#: means "collision geometry" here, and these do not collide.
NO_RAYCAST_BODIES: tuple[str, ...] = ("camera_lower_mount_link", "camera_upper_mount_link")
NO_RAYCAST_GROUP = "3"

#: Sections copied verbatim from the template, in the order MuJoCo expects them.
TEMPLATE_SECTIONS: tuple[str, ...] = (
    "compiler", "option", "size", "visual", "statistic", "default",
    "contact", "equality", "actuator", "sensor", "keyframe",
)


def compile_urdf(urdf_path: str) -> mujoco.MjModel:
    """Compile ``urdf_path`` with the flags this robot needs.

    The ``<mujoco>`` block is added to a COPY, written beside the original so relative mesh
    paths still resolve, and deleted afterwards -- the checked-in URDF stays free of
    simulator-specific settings.

    Args:
        urdf_path: The URDF to compile.

    Returns:
        The compiled model.
    """
    tree = ET.parse(urdf_path)
    block = ET.SubElement(tree.getroot(), "mujoco")
    ET.SubElement(block, "compiler", dict(URDF_COMPILER_ATTRS))
    directory = os.path.dirname(os.path.abspath(urdf_path))
    handle = tempfile.NamedTemporaryFile("w", suffix=".urdf", dir=directory, delete=False)
    try:
        tree.write(handle.name, encoding="unicode")
        handle.close()
        return mujoco.MjModel.from_xml_path(handle.name)
    finally:
        os.unlink(handle.name)


def saved_xml(model: mujoco.MjModel) -> ET.Element:
    """``mj_saveLastXML`` output, parsed.

    Args:
        model: A compiled model.

    Returns:
        Its ``<mujoco>`` root element.
    """
    with tempfile.NamedTemporaryFile("r", suffix=".xml", delete=False) as handle:
        path = handle.name
    try:
        mujoco.mj_saveLastXML(path, model)
        return ET.parse(path).getroot()
    finally:
        os.unlink(path)


def add_free_joint(worldbody: ET.Element, body: str = ROOT_BODY,
                   name: str = FREE_JOINT_NAME) -> None:
    """Give the root body a free joint, so the robot is not welded to the world.

    Args:
        worldbody: The generated ``<worldbody>``.
        body: Root body name.
        name: Joint name to emit.

    Raises:
        ValueError: If the root body is absent or already has a free joint.
    """
    root = next((b for b in worldbody.iter("body") if b.get("name") == body), None)
    if root is None:
        raise ValueError(f"no <body name={body!r}> in the generated tree")
    if any(j.get("type") == "free" for j in root.findall("joint")):
        raise ValueError(f"{body} already has a free joint")
    joint = ET.Element("joint", {"name": name, "type": "free",
                                 "limited": "false", "actuatorfrclimited": "false"})
    # After <inertial>, before the geoms, matching how the template read.
    insert_at = 1 if root.find("inertial") is not None else 0
    root.insert(insert_at, joint)


def carry_root_pose(gen_world: ET.Element, tpl_world: ET.Element,
                    body: str = ROOT_BODY) -> None:
    """Copy the template's root-body ``pos``/``quat`` onto the generated root.

    The MJCF states where the robot STARTS: the template has
    ``<body name="pelvis_link" pos="0 0 0.793">``, and a URDF has nowhere to say that, so
    the generated root comes out at the origin. Without this every site and every body sits
    0.793 m low at qpos0 -- the robot spawns inside the floor rather than standing on it.

    Args:
        gen_world: The generated ``<worldbody>``, edited in place.
        tpl_world: The template's ``<worldbody>``.
        body: Root body name.

    Raises:
        ValueError: If either tree lacks the root body.
    """
    src = next((b for b in tpl_world.iter("body") if b.get("name") == body), None)
    dst = next((b for b in gen_world.iter("body") if b.get("name") == body), None)
    if src is None or dst is None:
        raise ValueError(f"{body} is missing from the template or the generated tree")
    for attr in ("pos", "quat"):
        if src.get(attr) is not None:
            dst.set(attr, src.get(attr))
    print(f"[mjcf] root     {body} pos={dst.get('pos')} (from the template)")


def collect_sites(worldbody: ET.Element) -> dict[str, list[ET.Element]]:
    """Map body name -> its ``<site>`` children, from the template's tree.

    Sites are the one thing inside ``<worldbody>`` that is not derived from the URDF: they
    are mount points for sensors, the mocap frame and the height probe, and they exist only
    in the MJCF. Carried across by body NAME rather than by position, so a body moving in
    the tree does not strand its sites.

    Args:
        worldbody: The template's ``<worldbody>``.

    Returns:
        Body name -> list of site elements.
    """
    out: dict[str, list[ET.Element]] = {}
    for body in worldbody.iter("body"):
        sites = body.findall("site")
        if sites:
            out[body.get("name")] = sites
    return out


def attach_sites(worldbody: ET.Element, sites: dict[str, list[ET.Element]]) -> list[str]:
    """Re-attach carried-over sites to the generated bodies. Returns any that had no home.

    Args:
        worldbody: The generated ``<worldbody>``, edited in place.
        sites: From :func:`collect_sites`.

    Returns:
        Names of sites whose body no longer exists -- reported, not dropped silently,
        because a site is something else's mount point.
    """
    by_name = {b.get("name"): b for b in worldbody.iter("body")}
    orphaned: list[str] = []
    for body_name, elements in sites.items():
        target = by_name.get(body_name)
        for site in elements:
            if target is None:
                orphaned.append(f"{site.get('name')} (was on {body_name})")
            else:
                target.append(site)
    return orphaned


def exclude_from_raycast(worldbody: ET.Element, bodies=NO_RAYCAST_BODIES,
                         group: str = NO_RAYCAST_GROUP) -> int:
    """Move the named bodies' visual geoms out of the depth cameras' cast group.

    Args:
        worldbody: The generated ``<worldbody>``, edited in place.
        bodies: Body names to exclude.
        group: Group to move their group-1 geoms into.

    Returns:
        How many geoms were moved.

    Raises:
        ValueError: If a named body is absent -- silently skipping it would put the mount
            back in the depth image with nothing to say so.
    """
    by_name = {b.get("name"): b for b in worldbody.iter("body")}
    moved = 0
    for name in bodies:
        body = by_name.get(name)
        if body is None:
            raise ValueError(
                f"{name} is not in the generated tree, so its visual cannot be excluded "
                "from the ray cast. If the body was renamed, update NO_RAYCAST_BODIES to "
                "match -- and g1_self_scan.UNSCANNED_LINKS with it."
            )
        for geom in body.findall("geom"):
            if geom.get("group") == "1":
                geom.set("group", group)
                moved += 1
    return moved


def apply_actuator_gains(root: ET.Element, armature=JOINT_ARMATURE) -> dict[str, tuple]:
    """Rewrite the position actuators' PD gains from the armature table.

    THE GAINS MUST NOT BE COPIED FROM THE TEMPLATE. The template is the PREVIOUS robot
    (``g1_29j.xml``), and ``<default>`` is in :data:`TEMPLATE_SECTIONS` -- so when mode_11
    moved hip pitch from the 14.3 gear to the 22.5, its armature was updated here (it feeds
    ``<joint armature=...>``) while its ACTUATOR gains kept the old class's numbers. The
    result was a hip pitch 2.46x too soft: kp 40.18 where the policy exports 99.10.

    That was invisible almost everywhere. ``transfer/sim`` overwrites these gains from the
    policy (``Robot.set_pd_gains_from_policy``) and the hardware driver takes ``kp``/``kd``
    off the wire, so the ONE consumer of this block is the obelisk MuJoCo sim -- whose
    ``ApplyControl`` writes ``u_mujoco`` straight into ``data.ctrl`` and never reads the
    published gains. A stale number here therefore shows up only as a robot that tracks
    slightly worse in obelisk than in sim2sim, which is exactly how it presented.

    Args:
        root: The assembled MJCF root, edited in place.
        armature: Joint name -> rotor inertia; the same table :func:`apply_joint_dynamics`
            stamps onto the hinges, so the two cannot describe different motors.

    Returns:
        ``{default class: (kp, kv)}`` as written.

    Raises:
        ValueError: If an actuated joint has no armature, if a position actuator declares no
            default class, or if one class is shared by joints whose gains differ -- that
            last one means the class needs splitting, and silently writing one joint's gains
            over another's is the failure this function exists to prevent.
    """
    per_class: dict[str, dict[str, tuple]] = {}
    missing: list[str] = []
    for actuator in root.iter("actuator"):
        for position in actuator.findall("position"):
            joint = position.get("joint")
            cls = position.get("class")
            if joint not in armature:
                missing.append(str(joint))
                continue
            if cls is None:
                raise ValueError(
                    f"position actuator for {joint} declares no default class, so there is "
                    "nowhere to write its gains.")
            kp = armature[joint] * NATURAL_FREQ ** 2
            kv = JOINT_DAMPING_OVERRIDE.get(
                joint, 2.0 * DAMPING_RATIO * armature[joint] * NATURAL_FREQ)
            per_class.setdefault(cls, {})[joint] = (kp, kv)
    if missing:
        raise ValueError(
            f"no armature declared for actuated joint(s) {sorted(set(missing))}, so their PD "
            "gains cannot be derived. Add them to JOINT_ARMATURE.")

    written: dict[str, tuple] = {}
    for cls, joints in per_class.items():
        gains = set(joints.values())
        if len(gains) > 1:
            raise ValueError(
                f"default class {cls!r} is shared by joints with different gains: "
                f"{joints}. Split the class -- writing one joint's gains over another's is "
                "silent and is exactly what this check is for.")
        kp, kv = next(iter(gains))
        found = False
        for default in root.iter("default"):
            if default.get("class") != cls:
                continue
            for position in default.findall("position"):
                position.set("kp", f"{kp:.6g}")
                position.set("kv", f"{kv:.6g}")
                found = True
        if not found:
            raise ValueError(
                f"no <default class={cls!r}> with a <position> in the template, so the gains "
                f"for {sorted(joints)} would fall back to MuJoCo's defaults.")
        written[cls] = (kp, kv)
    return written


def apply_joint_dynamics(worldbody: ET.Element, armature=JOINT_ARMATURE) -> list[str]:
    """Stamp each hinge with its own armature. Returns the joints that had none declared.

    Args:
        worldbody: The generated ``<worldbody>``, edited in place.
        armature: Joint name -> rotor inertia.

    Returns:
        Names of hinges missing from ``armature``.
    """
    missing: list[str] = []
    for joint in worldbody.iter("joint"):
        name = joint.get("name")
        if joint.get("type") == "free":
            continue
        if name in armature:
            joint.set("armature", f"{armature[name]:.9g}")
        else:
            missing.append(name)
    return missing


def build(urdf: str, template: str, out: str) -> None:
    """Write ``out``: the template's non-body sections around a URDF-derived body tree.

    Args:
        urdf: Source URDF.
        template: Existing MJCF to take everything outside ``<worldbody>`` from.
        out: Output path.

    Raises:
        ValueError: If the template has no ``<worldbody>``, or a site is orphaned, or a
            hinge has no declared armature.
    """
    model = compile_urdf(urdf)
    generated = saved_xml(model)
    template_root = ET.parse(template).getroot()

    tpl_world = template_root.find("worldbody")
    gen_world = generated.find("worldbody")
    if tpl_world is None or gen_world is None:
        raise ValueError("template or generated model has no <worldbody>")

    print(f"[mjcf] urdf     {urdf}")
    print(f"[mjcf] template {template}")
    print(f"[mjcf] compiled {model.nbody - 1} bodies, {model.njnt} joints, "
          f"{model.ngeom} geoms, {model.nmesh} meshes")

    carry_root_pose(gen_world, tpl_world)
    add_free_joint(gen_world)
    orphaned = attach_sites(gen_world, collect_sites(tpl_world))
    if orphaned:
        raise ValueError(
            f"{len(orphaned)} site(s) have no body in the new tree: {orphaned}. A site is a "
            "mount point for a sensor or a frame -- dropping one silently would take that "
            "consumer with it."
        )
    moved = exclude_from_raycast(gen_world)
    print(f"[mjcf] moved {moved} visual geom(s) to group {NO_RAYCAST_GROUP} "
          f"(not ray-cast): {list(NO_RAYCAST_BODIES)}")
    missing = apply_joint_dynamics(gen_world)
    if missing:
        raise ValueError(
            f"no armature declared for {missing}. The template's blanket class would have "
            "given them 0.01 regardless of the motor, which is what this replaces."
        )

    root = ET.Element("mujoco", {"model": os.path.splitext(os.path.basename(out))[0]})
    for tag in TEMPLATE_SECTIONS:
        for element in template_root.findall(tag):
            root.append(copy.deepcopy(element))
    gains = apply_actuator_gains(root)
    print(f"[mjcf] PD gains derived from JOINT_ARMATURE for {len(gains)} actuator class(es): "
          + ", ".join(f"{c} kp={kp:.4g}" for c, (kp, _kv) in sorted(gains.items())))
    _point_meshdir_at_the_urdf(root, urdf, out)
    for element in generated.findall("asset"):
        root.append(element)
    root.append(gen_world)

    n_sites = sum(1 for _ in gen_world.iter("site"))
    groups: dict[int, int] = {}
    for geom in gen_world.iter("geom"):
        key = int(geom.get("group", 0))
        groups[key] = groups.get(key, 0) + 1
    print(f"[mjcf] wrote {sum(1 for _ in gen_world.iter('body'))} bodies, "
          f"{n_sites} sites, geoms by group {dict(sorted(groups.items()))}")

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(out, encoding="utf-8", xml_declaration=True)
    _assert_no_double_hyphen_comment(out)
    print(f"[mjcf] wrote {out}")


def _point_meshdir_at_the_urdf(root: ET.Element, urdf: str, out: str) -> str:
    """Resolve mesh assets against the URDF's directory, not the template's.

    The generated ``<asset>`` entries keep the paths the URDF used (``strippath="false"``),
    and those are relative to the URDF. The template's ``meshdir="meshes"`` would prepend
    its own, asking for ``meshes/meshes/left_hip_roll_link.STL``.

    A single prefix is enough even though the perceptive URDF draws meshes from TWO places
    (its own, for the mounts and the backpack; the base asset's, for the other 50): both are
    written relative to the URDF, so pointing ``meshdir`` there covers both.

    DERIVED with ``relpath`` rather than written as a count of ``..`` hops, per the repo's
    path rule, so moving either tree keeps this correct.

    Args:
        root: The output ``<mujoco>`` element, edited in place.
        urdf: Source URDF path.
        out: Where the MJCF is being written.

    Returns:
        The meshdir that was set.
    """
    meshdir = os.path.relpath(os.path.dirname(os.path.abspath(urdf)),
                              os.path.dirname(os.path.abspath(out)))
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("meshdir", meshdir)
    print(f"[mjcf] meshdir  {meshdir}")
    return meshdir


def _assert_no_double_hyphen_comment(path: str) -> None:
    """Refuse an output whose comments contain ``--``.

    MuJoCo tolerates it; ``xml.etree`` does not, and ``motion_gen.fk`` parses this same file
    with ``xml.etree`` for the FK chain. The failure is a hard parse error a long way from
    the comment that caused it.

    Args:
        path: The written file.

    Raises:
        ValueError: If any comment body contains a double hyphen.
    """
    text = open(path, encoding="utf-8").read()
    for comment in re.findall(r"<!--(.*?)-->", text, flags=re.S):
        if "--" in comment:
            raise ValueError(
                f"{path}: a comment contains '--', which xml.etree rejects: "
                f"{comment.strip()[:80]!r}"
            )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urdf", required=True, help="Source URDF.")
    parser.add_argument("--template", default=os.path.join(HERE, "g1_29j.xml"),
                        help="MJCF to copy everything outside <worldbody> from.")
    parser.add_argument("--out", required=True, help="Output MJCF.")
    args = parser.parse_args()
    build(os.path.abspath(args.urdf), os.path.abspath(args.template),
          os.path.abspath(args.out))


if __name__ == "__main__":
    main()
