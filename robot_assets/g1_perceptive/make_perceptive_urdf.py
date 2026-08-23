# Copyright 2026 Zachary Olkin. All rights reserved.

"""Build the perceptive-G1 URDF: our G1, as distinct from the stock one.

Three differences from the merged mode_11 base asset, all of them fixed-joint additions
that leave the 29 actuated joints untouched in name and order:

* two depth-camera mounts, with an optical frame each (:data:`MOUNTS`);
* a compute backpack: two masses and one shell (:data:`PAYLOADS`, :data:`SHELLS`);
* the base asset's meshes are referenced where they live, not copied
  (:func:`rebase_inherited_meshes`).

The motors, the waist geometry and the passive RUBBER hands are NOT here: they define which
G1 this is rather than what we bolted to it, so they live in
``robot_assets/g1/g1_21j/apply_mode11_deltas.py``, which patches the unmerged source against
a vendored upstream mode_11 and hands its output to ``merge_urdf_joints.py``. Doing the
hands there also removes an un-merge step this file used to need: the merge folded the Dex3
into ``wrist_yaw``, so swapping afterwards meant subtracting it back out.

Run from anywhere::

    python robot_assets/g1_perceptive/make_perceptive_urdf.py

Re-runnable. The mount placement is PINNED to the numbers the mount was designed to, not
derived from anything inside the mesh: each STL is exported with its origin ON the left
imager -- the point rays are cast from -- so the mesh goes straight to that declared
position. Deriving the placement by finding the imager solids inside the assembly was the
first design and was dropped: it makes the camera position a function of the camera
module's tessellation, so a re-export that changes the module, merges it into the bracket
or re-meshes it would silently relocate the camera. That search survives only as a printed
check (:func:`_report_imager_check`), which reports a drifted CAD datum without moving
anything.

So the whole transform is the given ``(x, y, z)`` plus the given pitch. The one thing that
is NOT free is the axis convention: the CAD is modelled in an OPTICAL frame with the
boresight on its own +z, while the sim's ray caster is +x boresight / +y left / +z up
(``g1_depth_env_cfg``'s ``convention="world"``). :func:`mount_mesh_rpy` is that fixed
remap and nothing else.

Two wrong readings of it cost a cycle each, and both rendered plausibly:

1. placing the mesh with identity rotation, on the theory that the bracket already carried
   the tilt -- it does not, and a level mount next to a 65 deg camera looks unremarkable;
2. placing it without the 180 deg roll, which put the right lens where the left belongs --
   nearly invisible on a near-symmetric bracket.

Hence the boresight post-condition in :func:`build`: a position check cannot catch either.

Measured off the shipped exports (see :func:`find_camera_pair`):

* they are in MILLIMETRES, so a ``scale`` is emitted; :func:`detect_units` handles a metric
  re-export, which is preferred (all 50 G1 meshes are metric and the base URDF carries no
  ``scale`` at all);
* each assembly contains exactly two identical 19.50 x 17.15 x 12.00 mm bodies -- the
  stereo pair, 50 mm baseline on the lower and 120 mm on the upper.

Two frames are emitted per camera, at the SAME position and differing only in orientation:

* ``<name>_mount_link`` -- the mesh, in the CAD's own z-forward frame;
* ``<name>_optical_link`` -- an empty link in the sim's x-forward frame. This is the ray
  origin. Empty links are how upstream Unitree carries ``d435_link`` and ``mid360_link``,
  so the idiom is not new here.

The optical links are frames, NOT sensor mounts: the depth sensors stay parented to
``torso_link`` with an offset, because ``FutureStateDataset._camera_pose_yaw_frame``
resolves ONE FK chain and refuses two different mount bodies.
"""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.join(
    HERE, os.pardir, "g1", "g1_21j", "g1_21j_box_merged_29j_mode11.urdf")
DEFAULT_DST = os.path.join(HERE, "g1_perceptive.urdf")

#: The body the mounts bolt to.
PARENT_LINK = "torso_link"


def mount_mesh_rpy(pitch_rad: float) -> tuple[float, float, float]:
    """URDF ``rpy`` placing a mount mesh so its camera looks ``pitch_rad`` below horizontal.

    The CAD is modelled as an OPTICAL frame -- +z along the boresight -- while the sim's ray
    caster uses +x boresight, +y left, +z up (``g1_depth_env_cfg``'s ``convention="world"``).
    Mapping +z onto +x is a +90 deg rotation about y; the CAD is additionally rolled 180 deg
    about the boresight relative to the robot, so the composition is::

        Ry(pitch) @ Rx(180) @ Ry(90)  ==  Rz(0) @ Ry(pitch - 90) @ Rx(180)

    which is the fixed-axis rpy ``(pi, pitch - 90 deg, 0)``. The roll is what puts the
    camera's LEFT imager on the robot's left; without it the pair is upside down and the
    right lens sits where the left should be -- a mistake that looks almost right, because
    the mount is nearly symmetric and the pair centre lands correctly either way.

    Args:
        pitch_rad: Downward pitch of the optical axis.

    Returns:
        ``(roll, pitch, yaw)`` for the URDF joint origin.
    """
    return (np.pi, pitch_rad - np.pi / 2.0, 0.0)


#: Extents (mm) of one imager module, used to find the stereo pair inside an assembly.
_IMAGER_EXTENTS_MM = (19.50, 17.15, 12.00)
_IMAGER_TOL_MM = 0.05

#: How far the STL origin may sit from the left imager before the datum report complains.
#: ``_DATUM_OK_MM`` is "indistinguishable from on it"; up to ``_DATUM_CLOSE_MM`` is a
#: modelling choice about where inside a sensor the origin belongs (die, front face, cover
#: glass) and is reported without alarm. Beyond that it is a datum error: the ones this
#: check was written for were 50 mm and 308 mm.
_DATUM_OK_MM = 0.5
_DATUM_CLOSE_MM = 3.0


@dataclass(frozen=True)
class MountSpec:
    """One camera mount: its mesh, where it casts rays from, and how far it looks down.

    Attributes:
        name: Link-name prefix.
        stl: Mesh filename under ``meshes/``.
        raycast_xyz: The RAYCAST POINT in the ``torso_link`` frame (m) -- the left imager of
            the stereo pair, which is the frame a stereo depth camera reports depth in, and
            which the CAD places its own origin on. The mesh is pinned here directly.
        pitch_deg: Downward pitch of the optical axis (deg).
    """

    name: str
    stl: str
    raycast_xyz: tuple[float, float, float]
    pitch_deg: float


#: The shipped pair, as measured in CAD. ``raycast_xyz`` is the mount centre the design is
#: built around plus the left imager's half-baseline offset in y (+25 mm / +60 mm), i.e. the
#: numbers exactly as specified: lower (0.107, 0, 0.045) and upper (0.085, 0, 0.285).
MOUNTS = (
    MountSpec("camera_lower", "PART_lower_camera_assembly.STL", (0.107, 0.025, 0.045), 65.0),
    MountSpec("camera_upper", "PART_upper_camera_assembly.STL", (0.085, 0.060, 0.285), 15.0),
)


@dataclass(frozen=True)
class PayloadSpec:
    """One rigid mass bolted to the torso -- a compute box, not a moving part.

    The backpack is ONE mesh but NOT one rigid body's worth of uniform stuff: the Thor and
    the ZED sit in different places and weigh very different amounts, so they are modelled
    as two links. A URDF link carries exactly one ``<inertial>``, and lumping them would
    put ~2.6 kg at a fictitious midpoint. Both are welded to ``torso_link``, so the split
    costs nothing dynamically and keeps each number traceable to what was weighed.

    Attributes:
        name: Link-name prefix.
        com_xyz: Centre of mass in the ``torso_link`` frame (m). The link origin IS the
            centre of mass, so the ``<inertial>`` sits at the link origin and the joint
            carries the whole offset -- the number in the URDF is the number that was
            measured, with no arithmetic in between.
        mass_kg: Mass of this body.
        size_xyz: Extents (m) of the solid box the inertia tensor is computed from, about
            ``com_xyz``, axis-aligned with the torso. Taken from the matching block in the
            shell mesh, which :func:`_report_shell_datum` re-checks: both are solid
            rectangular blocks (bounding-box fill 0.9999 and 0.9956), so the box is the
            SHAPE exactly and the only modelling assumption left is uniform density.
        size_is_estimate: ``True`` while ``size_xyz`` is a stand-in rather than a CAD
            figure. Printed at build, because an inertia nobody has checked should not look
            like one that has been.
    Note:
        These bodies carry MASS ONLY. The shell they live in is one mesh spanning both, and
        it is a separate link (:data:`SHELLS`) because its CAD datum is its own fact -- tying
        the mesh to a mass would make a re-export able to move a centre of mass.
    """

    name: str
    com_xyz: tuple[float, float, float]
    mass_kg: float
    size_xyz: tuple[float, float, float]
    size_is_estimate: bool


#: The compute backpack, as weighed, relative to the ``torso_link`` origin.
#:
#: x AND y are negated from the measured pair (102.284, 0.3727, 127.1) and
#: (177.594, 1.4421, 63.861). Those were taken with the pack on the robot's FRONT; it has
#: since been swung to the back, and the transform that did it is Rz(180) about the robot's
#: centreline -- a yaw, so x and y flip and z does not. That is not inferred from the word
#: "backpack": the shell mesh puts the ZED block 75.310 mm behind, 1.0694 mm RIGHT of and
#: 63.239 mm below the Thor block, and the given numbers put it the same distance behind,
#: 1.0694 mm LEFT and 63.239 below. Both signs flip, z agrees, and negating x alone leaves
#: a 2.139 mm residual in y that vanishes to 0.0001 mm under the full yaw.
#:
#: +x is forward -- checked against the foot collision box, whose centre sits 35.9 mm ahead
#: of the ankle because the toe is longer than the heel, so the sign does not rest on the
#: cameras. "Everything else" (mounting plate, cabling, fasteners) is lumped into the Thor
#: mass rather than modelled, which is why 1.940 kg is more than a bare Thor -- note the
#: inertia box below is the Thor BLOCK, so that remainder's spread is not in the tensor.
PAYLOADS = (
    PayloadSpec("backpack_thor", (-0.102284, -0.0003727, 0.1271), 1.940,
                (0.05638, 0.11240, 0.24086), False),
    PayloadSpec("backpack_zed", (-0.177594, -0.0014421, 0.063861), 0.700,
                (0.07100, 0.09200, 0.11750), False),
)


@dataclass(frozen=True)
class ShellSpec:
    """A visual-only mesh welded to the torso: geometry with no mass of its own.

    Attributes:
        name: Link-name prefix.
        stl: Mesh filename under ``meshes/``.
        xyz: Where the STL's OWN origin goes, in the ``PARENT_LINK`` frame (m).
        rpy: Orientation of the mesh in that frame (rad).
        collision_size: Extents (m) of a box collider, in the LINK frame, or ``None`` for a
            visual-only shell.
        collision_origin: Centre of that box in the link frame (m).
    """

    name: str
    stl: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    collision_size: tuple[float, float, float] | None = None
    collision_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)


#: The backpack shell, placed at the Thor centre because that is where its datum now is:
#: the re-export puts the STL origin exactly on the centre of the Thor block (0.000 mm), and
#: the ZED block's centre sits at the declared ZED offset to 0.001 mm. The pack is modelled
#: on the BACK in this export, so the yaw that turned the old front-mounted one around is
#: gone. :func:`_report_shell_datum` re-checks both of those every build.
#: The yaw is the same front-to-back correction the masses get, done as a RIGID rotation
#: rather than a mirror: Rz(180) maps the pod's x span [+1.2, +167.3] mm to [-167.3, -1.2],
#: which keeps the face that was nearest the torso nearest the torso -- a pod turned around,
#: not a reflected one. A mirror would need a negative mesh scale, which inverts every face
#: normal. The pod is symmetric in y to 0.07 mm, so the yaw's y flip costs nothing. This is
#: a placement correction, NOT a substitute for moving the pack in CAD: re-export with it on
#: the back and the yaw goes to zero.
#: The collider exists because ``backpack_contact`` needs something to fire on: a contact
#: sensor watching a body with no collider is silently dead, which is the state
#: ``torso_contact`` has been in (its box sits on ``waist_yaw_link``). It is also physically
#: right -- 2.6 kg standing 0.2 m behind the torso is the first thing to touch a wall the
#: robot backs into, and the first thing to hit the floor in a backwards fall.
#:
#: A box rather than the mesh: the shell is a plate plus a frame plus two blocks, the convex
#: hull of that is nearly the box anyway, and a box is what every other collider here is.
#:
#: It is the mesh's bounding box with the FRONT FACE PULLED BACK 12.1 mm, which is not
#: cosmetic. At the full bbox the pack overlapped the torso box by 2.09 mm across a
#: 180 x 190 mm face, and since the two are rigidly welded PhysX cannot resolve it by moving
#: them apart -- it just pushes. Measured: the robot drifted +0.38 m FORWARD in one second
#: of settling, when the backpack's own mass should carry it backwards, and it did drift
#: -0.19 m backwards before the collider existed. Welded bodies must not touch, and nothing
#: filters this pair: PhysX only auto-filters ADJACENT links, and these are two joints
#: apart. The pull-back leaves 10 mm of clearance; ``test_colliders_do_not_interpenetrate``
#: fails if any pair closes up again.
SHELLS = (
    ShellSpec("backpack_shell", "ASSY_backpack.STL", PAYLOADS[0].com_xyz,
              collision_size=(0.15401, 0.18748, 0.25307),
              collision_origin=(-0.040805, -0.00007, -0.006105)),
)


def box_inertia(mass_kg: float, size_xyz) -> tuple[float, float, float]:
    """Diagonal inertia of a uniform solid box about its own centre.

    Args:
        mass_kg: Body mass.
        size_xyz: Full extents along x, y, z (m).

    Returns:
        ``(ixx, iyy, izz)`` in kg m^2. Off-diagonals are zero by construction, which is the
        approximation: the box is taken axis-aligned with the torso.

    Note:
        A box is a coarse model of a compute brick, and deliberately so -- the term that
        matters here is the PARALLEL-AXIS one, ``m * d^2`` about the torso, which is exact
        because ``com_xyz`` is measured. For the Thor that is ~0.052 kg m^2 against a
        self-inertia near 0.003, so a factor-of-two error in the box changes the robot's
        rotational dynamics by well under a percent. Replace with the CAD tensor when it is
        available; do not agonise over it before then.
    """
    x, y, z = (float(v) for v in size_xyz)
    k = float(mass_kg) / 12.0
    return (k * (y * y + z * z), k * (x * x + z * z), k * (x * x + y * y))


def detect_units(mesh: trimesh.Trimesh) -> float:
    """Metres-per-unit for a mount mesh: 1.0 if already metric, 0.001 if millimetres.

    Detected rather than assumed, and PRINTED by the caller, because the two conventions
    differ by 1000x -- an error too large to be subtle in a viewer but perfectly capable of
    reaching the USD conversion and the ray-cast target list.

    Metres is the convention here: all 50 G1 meshes are metric and the base URDF carries no
    ``scale`` attribute at all, so a scaled mesh is a special case that a re-export can drop.

    Args:
        mesh: The loaded mount mesh.

    Returns:
        ``1.0`` or ``0.001``.

    Raises:
        ValueError: If the extents match neither convention for a part of this size.
    """
    longest = float(np.max(mesh.bounds[1] - mesh.bounds[0]))
    if 0.01 <= longest <= 1.0:
        return 1.0
    if 10.0 <= longest <= 1000.0:
        return 0.001
    raise ValueError(
        f"mount mesh has a longest extent of {longest:g}, which is neither metres nor "
        "millimetres for a camera bracket; check the export units."
    )


def find_camera_pair(mesh_path: str) -> tuple[np.ndarray, float, float]:
    """Locate the two imager modules inside a mount assembly.

    Args:
        mesh_path: Path to the mount STL, in metres or millimetres.

    Returns:
        ``(centres, baseline, metres_per_unit)`` -- ``centres`` is a ``(2, 3)`` array of the
        two imager centres in METRES, in the STL's own frame, in no particular order.

        Deliberately NOT labelled left/right here: "left" is a fact about the robot, and the
        CAD's roll about the boresight decides which STL imager ends up there. Labelling by
        STL ``+y`` assumed a roll that turned out to be inverted, which put the right lens
        where the left belongs. :func:`_report_imager_check` resolves it in the torso frame.

        Used only by that check -- placement does not depend on it; see the module docstring.

    Raises:
        ValueError: If the assembly does not contain exactly two imager-sized bodies. That
            is the check worth having: silently taking the wrong body would place the
            camera somewhere plausible and wrong.
    """
    mesh = trimesh.load(mesh_path, force="mesh")
    mpu = detect_units(mesh)
    want = sorted(np.array(_IMAGER_EXTENTS_MM) * 0.001 / mpu)
    cams = [p for p in mesh.split(only_watertight=False)
            if np.allclose(sorted(p.bounds[1] - p.bounds[0]), want, atol=_IMAGER_TOL_MM * 0.001 / mpu)]
    if len(cams) != 2:
        raise ValueError(
            f"{os.path.basename(mesh_path)}: expected exactly 2 bodies of "
            f"{_IMAGER_EXTENTS_MM} mm (the stereo pair), found {len(cams)}. If the camera "
            "module changed, update _IMAGER_EXTENTS_MM; if the export changed, check that "
            "the imagers are still separate solids."
        )
    centres = np.array([(p.bounds[0] + p.bounds[1]) / 2.0 for p in cams]) * mpu
    return centres, float(np.linalg.norm(centres[0] - centres[1])), mpu


#: Token inertial for a welded, effectively massless link, matching
#: ``make_hand_links_urdf.py``'s ``_MASS`` / ``_INERTIA``.
#:
#: NOT decoration. With ``merge_fixed_joints=False`` (what
#: ``urdf_to_usd_conversion.py`` uses) every fixed-joint link becomes its own PhysX rigid
#: body, and ``UrdfConverterCfg.link_density`` defaults to 0.0 -- so a link with no
#: ``<inertial>`` becomes a ZERO-MASS rigid body, which is a solver hazard rather than a
#: free frame. ``g1_29j.xml`` carries a comment about the same trap for ``left_palm_link``.
#:
#: These are placeholders: the mounts and their cameras have real mass, and it belongs here
#: once CAD provides it. A token value is the safe default meanwhile -- it keeps the bodies
#: legal without inventing an inertia tensor.
_TOKEN_MASS_KG = 1e-6
_TOKEN_INERTIA = 1e-9


def _token_inertial() -> ET.Element:
    """A near-massless ``<inertial>``; see :data:`_TOKEN_MASS_KG` for why it is not omitted."""
    inertial = ET.Element("inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": f"{_TOKEN_MASS_KG:g}"})
    ET.SubElement(inertial, "inertia", {
        "ixx": f"{_TOKEN_INERTIA:g}", "ixy": "0", "ixz": "0",
        "iyy": f"{_TOKEN_INERTIA:g}", "iyz": "0", "izz": f"{_TOKEN_INERTIA:g}"})
    return inertial


def _inertial(mass_kg: float, size_xyz) -> ET.Element:
    """An ``<inertial>`` at the link origin for a uniform box of ``mass_kg``."""
    ixx, iyy, izz = box_inertia(mass_kg, size_xyz)
    inertial = ET.Element("inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": f"{mass_kg:.6g}"})
    ET.SubElement(inertial, "inertia", {
        "ixx": f"{ixx:.9g}", "ixy": "0", "ixz": "0",
        "iyy": f"{iyy:.9g}", "iyz": "0", "izz": f"{izz:.9g}"})
    return inertial


def _roty(rad: float) -> np.ndarray:
    """Rotation matrix about +y by ``rad``. Positive pitches +x downward."""
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rpy_to_mat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis ``rpy`` -> rotation matrix, i.e. ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""
    cr, sr, cp, sp, cy, sy = (np.cos(roll), np.sin(roll), np.cos(pitch),
                              np.sin(pitch), np.cos(yaw), np.sin(yaw))
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def _xyz(v) -> str:
    """URDF-formatted ``x y z``."""
    return " ".join(f"{float(c):.9g}" for c in v)


def _link_with_mesh(name: str, stl_rel: str, metres_per_unit: float,
                    inertial: ET.Element | None = None) -> ET.Element:
    """A link carrying one mesh at its own origin.

    A ``scale`` attribute is emitted ONLY for a millimetre mesh, so a metric export produces
    a URDF as scale-free as the rest of the robot's 50 meshes.

    Args:
        name: Link name.
        stl_rel: Mesh path relative to the URDF.
        metres_per_unit: Mesh scale, from :func:`detect_units`.
        inertial: The link's ``<inertial>``; a token near-massless one when omitted.
    """
    link = ET.Element("link", {"name": name})
    link.append(_token_inertial() if inertial is None else inertial)
    # NAME the visual after its link. The converter names the inner geometry Xform after the
    # visual, falling back to the MESH FILENAME when unnamed -- and every stock G1 mesh
    # happens to be called after its link, so that fallback looks like a rule until a mesh
    # called ASSY_backpack.STL lands under backpack_shell_link and the prim comes out
    # "ASSY_backpack". g1_self_scan builds ray-cast targets from LINK names, so the target
    # would name a prim that does not exist -- and MultiMeshRayCaster answers that with a
    # log line, not an error, leaving the body silently missing from every depth image.
    visual = ET.SubElement(link, "visual", {"name": name})
    ET.SubElement(visual, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    geom = ET.SubElement(visual, "geometry")
    attrs = {"filename": stl_rel}
    if metres_per_unit != 1.0:
        attrs["scale"] = " ".join([f"{metres_per_unit:g}"] * 3)
    ET.SubElement(geom, "mesh", attrs)
    mat = ET.SubElement(visual, "material", {"name": "camera_mount_grey"})
    ET.SubElement(mat, "color", {"rgba": "0.25 0.25 0.28 1"})
    return link


def _fixed_joint(name: str, parent: str, child: str, xyz, rpy=(0.0, 0.0, 0.0)) -> ET.Element:
    joint = ET.Element("joint", {"name": name, "type": "fixed"})
    ET.SubElement(joint, "origin", {"xyz": _xyz(xyz), "rpy": _xyz(rpy)})
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    return joint


def _report_imager_check(stl_abs: str, m: MountSpec, mpu: float) -> None:
    """Report how far the STL origin is from the imager it is supposed to sit on.

    A CHECK, never a dependency. Placement is pinned to the declared raycast point, so this
    cannot move the robot -- it only tells you whether the CAD datum landed where intended.
    Deriving placement from these blocks was the previous design and was rejected for good
    reason: a re-export that changes the camera model, merges the imagers into the bracket,
    or re-tessellates them would silently relocate the camera.

    Non-fatal by design: if the blocks cannot be found, that is a fact about the export, not
    an error in the URDF being written.

    Args:
        stl_abs: Path to the mount mesh.
        m: The mount being placed.
        mpu: Metres per STL unit, for the report.
    """
    try:
        centres, baseline, _ = find_camera_pair(stl_abs)
    except ValueError as exc:                          # noqa: BLE001 - reported, not raised
        print(f"    imager check             : SKIPPED ({exc})\n")
        return
    # Which imager is the robot's left depends on the CAD roll, so resolve it the same way
    # the placement does: rotate into the torso frame and take the greater +y.
    R = _rpy_to_mat(*mount_mesh_rpy(np.radians(m.pitch_deg)))
    in_torso = np.array([R @ c for c in centres])
    left_stl = centres[int(np.argmax(in_torso[:, 1]))]
    d = np.round(left_stl * 1000.0, 3)
    err = float(np.linalg.norm(d))
    if err <= _DATUM_OK_MM:
        verdict = "OK -- STL origin is ON the left imager"
    elif err <= _DATUM_CLOSE_MM:
        # A millimetre or two is a modelling choice (sensor die vs front face), not a fault;
        # calling it one trains you to ignore the line that matters.
        verdict = f"OK -- STL origin is {err:.2f} mm from the left imager, within tolerance"
    else:
        verdict = (f"OFF by {err:.2f} mm -- the mesh renders displaced by this until the CAD "
                   "datum is moved")
    print(f"    imager check             : {verdict}")
    print(f"                               offset {d} mm, baseline "
          f"{baseline * 1000:.1f} mm\n")


def rebase_inherited_meshes(root: ET.Element, src: str, dst: str) -> int:
    """Repoint the base robot's mesh paths at where they actually live.

    ``src``'s mesh filenames are relative to ``src``'s own directory, so writing the tree
    out at ``dst`` breaks every one of them. The alternative -- copying all 50 G1 meshes in
    beside the new URDF -- was tried and dropped: it is 30 MB of byte-identical duplicates,
    and two copies of one mesh that nothing compares is precisely the drift this repo keeps
    getting bitten by. Only the genuinely new meshes (mounts, backpack) live here.

    The prefix is DERIVED from the two paths rather than written as a count of ``..`` hops,
    per the repo's path rule, so moving either directory keeps this correct.

    Args:
        root: The URDF ``<robot>`` element, edited in place.
        src: The base URDF the tree was parsed from.
        dst: Where it is about to be written.

    Returns:
        ``(count, prefix)`` -- how many references were repointed, and the prefix, which
        anything added LATER from the base asset's mesh folder needs too.
    """
    prefix = os.path.relpath(os.path.dirname(src), os.path.dirname(dst))
    n = 0
    for mesh in root.iterfind(".//geometry/mesh"):
        fn = mesh.get("filename")
        if fn and not os.path.isabs(fn) and "://" not in fn:
            mesh.set("filename", os.path.normpath(os.path.join(prefix, fn)))
            n += 1
    return n, prefix


def _add_payloads(root: ET.Element, payloads) -> float:
    """Weld each payload to :data:`PARENT_LINK` and report the total mass added.

    Args:
        root: The URDF ``<robot>`` element, appended to in place.
        payloads: The :class:`PayloadSpec` list.

    Returns:
        Total mass added, kg.
    """
    total = 0.0
    for pl in payloads:
        link = ET.Element("link", {"name": f"{pl.name}_link"})
        link.append(_inertial(pl.mass_kg, pl.size_xyz))
        root.append(link)
        root.append(_fixed_joint(f"{pl.name}_joint", PARENT_LINK, f"{pl.name}_link",
                                 pl.com_xyz))
        total += pl.mass_kg

        ixx, iyy, izz = box_inertia(pl.mass_kg, pl.size_xyz)
        note = "  <- ESTIMATED, replace with the CAD figure" if pl.size_is_estimate else ""
        print(f"  {pl.name}  {pl.mass_kg * 1000:.0f} g")
        print(f"    centre of mass  : {_xyz(pl.com_xyz)}  ({PARENT_LINK} frame)")
        print(f"    inertia box     : {' x '.join(f'{v * 1000:.0f}' for v in pl.size_xyz)} mm"
              f"{note}")
        print(f"    -> ixx iyy izz  : {ixx:.6g} {iyy:.6g} {izz:.6g} kg m^2\n")
    return total


def _add_shells(root: ET.Element, shells, payloads) -> None:
    """Weld each visual-only shell to :data:`PARENT_LINK`, and check its datum.

    Args:
        root: The URDF ``<robot>`` element, appended to in place.
        shells: The :class:`ShellSpec` list.
        payloads: The payloads whose centres should fall inside a shell, for the check.
    """
    for sh in shells:
        stl_abs = os.path.join(HERE, "meshes", sh.stl)
        if not os.path.isfile(stl_abs):
            # Not fatal: the masses are already in, and they are what training needs. A
            # missing visual is worth saying out loud, not worth failing the build over.
            print(f"  {sh.name}: mesh '{sh.stl}' NOT FOUND under meshes/ -- shell skipped; "
                  "re-run once the STL is in place.\n")
            continue
        mesh = trimesh.load(stl_abs, force="mesh")
        mpu = detect_units(mesh)
        link = _link_with_mesh(f"{sh.name}_link", f"meshes/{sh.stl}", mpu)
        if sh.collision_size is not None:
            collision = ET.SubElement(link, "collision")
            ET.SubElement(collision, "origin",
                          {"xyz": _xyz(sh.collision_origin), "rpy": "0 0 0"})
            ET.SubElement(ET.SubElement(collision, "geometry"), "box",
                          {"size": _xyz(sh.collision_size)})
        root.append(link)
        root.append(_fixed_joint(f"{sh.name}_joint", PARENT_LINK, f"{sh.name}_link",
                                 sh.xyz, sh.rpy))
        units = "metres" if mpu == 1.0 else f"MILLIMETRES (scale {mpu:g})"
        kind = "visual only" if sh.collision_size is None else "visual + box collider"
        print(f"  {sh.name}  ({sh.stl}, {units}, {kind})")
        print(f"    STL origin goes at: {_xyz(sh.xyz)}  ({PARENT_LINK} frame), "
              f"rpy {' '.join(f'{np.degrees(v):.0f}' for v in sh.rpy)} deg")
        if sh.collision_size is not None:
            # The collider may be SMALLER than the mesh (clearance against a welded
            # neighbour), never larger -- a box sticking out past the geometry it stands for
            # would collide with things the robot visibly misses.
            lo, hi = np.asarray(mesh.bounds) * mpu
            c, h = np.asarray(sh.collision_origin), np.asarray(sh.collision_size) / 2.0
            out = np.maximum(np.maximum(lo - (c - h), (c + h) - hi), 0.0) * 1000.0
            shrink = ((hi - lo) - np.asarray(sh.collision_size)) * 1000.0
            # 50 um: the declared extents are rounded to 5 dp, so an exact-zero test just
            # reports the rounding as an overhang.
            verdict = ("inside the mesh" if out.max() < 0.05 else
                       f"STICKS OUT past the mesh by {np.round(out, 3)} mm")
            print(f"    collider box      : "
                  f"{' x '.join(f'{v * 1000:.0f}' for v in sh.collision_size)} mm at "
                  f"{_xyz(sh.collision_origin)}")
            print(f"                        {verdict}; "
                  f"{np.round(shrink, 2)} mm smaller than the mesh bbox")
        _report_shell_datum(mesh, mpu, sh, payloads)


def _report_shell_datum(mesh: trimesh.Trimesh, mpu: float, shell: ShellSpec,
                        payloads) -> None:
    """Check the placed shell against the masses it is supposed to contain.

    Two things, both of which have been wrong at some point in this asset's life:

    1. the shell's overall footprint on the robot, so a datum error of the size seen in the
       earlier exports (hundreds of mm) is impossible to miss;
    2. per payload, the nearest BLOCK inside the shell -- its centre against the declared
       centre of mass, and its extents against the declared inertia box. That is what turns
       :attr:`PayloadSpec.size_xyz` from a guess into a measurement with a standing check.

    Derived, and deliberately only ever printed. Placement and mass both come from declared
    numbers; a re-export that re-tessellates or re-splits the mesh must not be able to move
    a centre of mass, so this reports disagreement rather than resolving it.

    Args:
        mesh: The loaded shell mesh, in its own units.
        mpu: Metres per mesh unit.
        shell: The shell being placed.
        payloads: Payloads expected to sit inside it.
    """
    # Rotate the eight corners, not the bounds: an AABB of a rotated box is not the rotated
    # AABB, and this check exists precisely to be believed.
    rot = _rpy_to_mat(*shell.rpy)
    b = np.asarray(mesh.bounds) * mpu
    corners = np.array([[b[i, 0], b[j, 1], b[k, 2]]
                        for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    placed = corners @ rot.T + np.asarray(shell.xyz)
    lo, hi = placed.min(axis=0), placed.max(axis=0)
    print(f"    occupies (torso)  : x[{lo[0]:+.4f},{hi[0]:+.4f}] y[{lo[1]:+.4f},{hi[1]:+.4f}] "
          f"z[{lo[2]:+.4f},{hi[2]:+.4f}] m")

    parts = mesh.split(only_watertight=False)
    blocks = [((p.bounds[0] + p.bounds[1]) / 2.0 * mpu, (p.bounds[1] - p.bounds[0]) * mpu,
               abs(p.volume) / max(float(np.prod(p.bounds[1] - p.bounds[0])), 1e-30))
              for p in parts]
    for pl in payloads:
        # Into the shell's own frame, where the blocks live.
        c = rot.T @ (np.asarray(pl.com_xyz) - np.asarray(shell.xyz))
        i = int(np.argmin([np.linalg.norm(bc - c) for bc, _, _ in blocks]))
        bc, bext, fill = blocks[i]
        d_c = np.linalg.norm(bc - c) * 1000.0
        d_e = np.abs(np.asarray(pl.size_xyz) - bext) * 1000.0
        print(f"      {pl.name}")
        print(f"        nearest block centre off by {d_c:6.3f} mm "
              f"({np.round((bc - c) * 1000, 3)})")
        print(f"        its extents {np.round(bext * 1000, 2)} mm vs declared "
              f"{np.round(np.asarray(pl.size_xyz) * 1000, 2)} -> off by "
              f"{np.round(d_e, 3)} mm")
        print(f"        bounding-box fill {fill:.4f}"
              f"{' (a solid block, so the box inertia is the real shape)' if fill > 0.99 else ''}")
    print()


def urdf_total_mass(root: ET.Element) -> float:
    """Sum every link's ``<inertial><mass>``, in kg."""
    return sum(float(m.get("value", 0.0))
               for m in root.iterfind("link/inertial/mass"))


def build(src: str, dst: str, mounts=MOUNTS, payloads=PAYLOADS,
          shells=SHELLS) -> None:
    """Write ``dst``: ``src`` plus one mesh link and one optical frame per mount."""
    tree = ET.parse(src)
    root = tree.getroot()
    existing = {l.get("name") for l in root.findall("link")}
    if PARENT_LINK not in existing:
        raise ValueError(f"{src} has no '{PARENT_LINK}' to mount the cameras on.")

    base_mass = urdf_total_mass(root)
    print(f"[perceptive] base {os.path.relpath(src, HERE)}  ({base_mass:.3f} kg)")
    # Before appending anything: this must see only the INHERITED meshes. The mount and
    # shell meshes are ours and are already relative to this file.
    n, base_prefix = rebase_inherited_meshes(root, src, dst)
    print(f"[perceptive] repointed {n} inherited mesh paths at the base asset's own folder")


    print(f"[perceptive] mounting on {PARENT_LINK}\n")
    for m in mounts:
        stl_abs = os.path.join(HERE, "meshes", m.stl)
        mpu = detect_units(trimesh.load(stl_abs, force="mesh"))
        pitch = np.radians(m.pitch_deg)
        xyz = np.asarray(m.raycast_xyz, dtype=float)

        # PINNED, not derived. The STL origin IS the raycast point by design, so the mesh
        # goes exactly at the declared position -- no search, nothing that a re-export can
        # move. Both links share that origin; they differ only in ORIENTATION, and only
        # because the CAD frame is z-forward while the sim's ray caster is x-forward
        # (see mount_mesh_rpy). That remap is a fixed axis convention, not a measurement.
        mesh_rpy = mount_mesh_rpy(pitch)
        optical_rpy = (0.0, pitch, 0.0)

        root.append(_link_with_mesh(f"{m.name}_mount_link", f"meshes/{m.stl}", mpu))
        root.append(_fixed_joint(f"{m.name}_mount_joint", PARENT_LINK,
                                 f"{m.name}_mount_link", xyz, mesh_rpy))
        optical = ET.Element("link", {"name": f"{m.name}_optical_link"})
        optical.append(_token_inertial())
        root.append(optical)
        root.append(_fixed_joint(f"{m.name}_optical_joint", PARENT_LINK,
                                 f"{m.name}_optical_link", xyz, optical_rpy))

        # POST-CONDITION on the axis remap: the mesh's own boresight (+z, rotated) must
        # equal the optical frame's (+x, rotated). Cheap, and it is what caught the mount
        # being placed level, and later upside down.
        R_mesh = _rpy_to_mat(*mesh_rpy)
        mesh_bore = R_mesh @ np.array([0.0, 0.0, 1.0])
        optical_bore = _roty(pitch) @ np.array([1.0, 0.0, 0.0])
        if not np.allclose(mesh_bore, optical_bore, atol=1e-9):
            raise ValueError(
                f"{m.name}: the mesh's boresight {np.round(mesh_bore, 6)} disagrees with the "
                f"optical frame's {np.round(optical_bore, 6)}. mount_mesh_rpy is wrong for "
                "this export -- the mount would render aimed somewhere the camera is not."
            )

        units = "metres" if mpu == 1.0 else f"MILLIMETRES (scale {mpu:g})"
        print(f"  {m.name}  ({m.stl}, {units})")
        print(f"    raycast point (pinned)   : {_xyz(xyz)}  (torso frame)")
        print(f"    optical frame  rpy       : 0 {m.pitch_deg} 0 deg   (+x boresight, sim)")
        print(f"    mesh           rpy       : "
              f"{' '.join(f'{np.degrees(v):.1f}' for v in mesh_rpy)} deg   (+z boresight, CAD)")
        print(f"    boresight (mesh +z == optical +x): {np.round(mesh_bore, 4)}")
        _report_imager_check(stl_abs, m, mpu)

    print(f"[perceptive] payloads on {PARENT_LINK}\n")
    added = _add_payloads(root, payloads)
    _add_shells(root, shells, payloads)
    print(f"[perceptive] mass {base_mass:.3f} -> {urdf_total_mass(root):.3f} kg "
          f"(+{added:.3f} kg payload)")

    ET.indent(tree, space="  ")
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    print(f"[perceptive] wrote {dst}")


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=DEFAULT_SRC, help="Base 29-DOF URDF to extend.")
    p.add_argument("--dst", default=DEFAULT_DST, help="Output URDF.")
    args = p.parse_args()
    build(os.path.abspath(args.src), os.path.abspath(args.dst))


if __name__ == "__main__":
    main()
