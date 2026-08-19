# Copyright 2026 Zachary Olkin. All rights reserved.

"""Bring ``g1_21j_29dof.urdf`` up to Unitree's ``g1_29dof_mode_11``.

Run from anywhere::

    python robot_assets/g1/g1_21j/apply_mode11_deltas.py

mode_11 is the variant this robot IS: high-torque hip (the 22.5 gear ratio), revolute-35
waist, light wrists. Our source predates it, so a set of motor limits, three inertials, a
handful of frame offsets and the hands all need to move together. They land in ONE patch
because they force one retrain between them and there is no value in doing that twice.

Every number is DERIVED from ``reference/g1_29dof_mode_11.urdf``, vendored beside this
script, rather than transcribed. What is transcribed is only the SCOPE -- which fields we
take -- because our source carries local customisations upstream knows nothing about
(``pelvis_link``, the box foot collider, the heel/toe frames) and a blind sync would
silently revert them. See that folder's README for the full keep-list.

The four delta classes, all measured against the reference:

* **joint limits** -- effort and velocity, 26 joints. Notably hip pitch 88 -> 139 N.m with
  velocity 32 -> 20, which is the gear change (139/88 = 1.579, 22.5/14.3 = 1.573);
* **joint frames** -- the waist chain moves the torso 10 mm DOWN
  (``waist_roll`` z +9 mm, ``waist_pitch`` z -19 mm) and the shoulder, logo and head
  mounts each move +10 mm back UP inside it. At zero waist angles those cancel exactly, so
  the arms do not move; under waist roll/pitch they do not, because the two offsets are
  applied in differently-rotated frames;
* **link inertials** -- ``torso_link`` (8.562 -> 6.780 kg), ``waist_yaw_link``,
  ``waist_roll_link``;
* **hands** -- the Dex3-1 subtree is replaced by the passive rubber hand, which is what the
  hardware has and therefore what the depth cameras must see.

The mass reconciliation is a POST-CONDITION, not a comment. :func:`expected_delta_kg` adds
up what the four classes SHOULD move -- from the same two files, so nothing is rounded --
and the patch refuses if the actual total does not match to a nanogram. A delta silently
missed or applied twice fails the build rather than shipping. It is deliberately not the
difference of the two files' totals: upstream also dropped two 1 g visual tokens we keep
and added sensor frames we do not take.

Downstream, in order::

    apply_mode11_deltas.py           -> g1_21j_29dof_mode11.urdf
    merge_urdf_joints.py --keep ...  -> g1_21j_box_merged_29j_mode11.urdf
    g1_perceptive/make_perceptive_urdf.py  -> adds cameras + backpack

The ``--keep`` matters: the merge folds every unlisted fixed joint into its parent, and
folding the rubber hand into ``wrist_yaw`` would erase the separate link the ray caster
needs to see a hand at all.
"""

from __future__ import annotations

import argparse
import copy
import os
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.join(HERE, "g1_21j_29dof.urdf")
DEFAULT_DST = os.path.join(HERE, "g1_21j_29dof_mode11.urdf")
DEFAULT_REF = os.path.join(HERE, "reference", "g1_29dof_mode_11.urdf")

#: Joint origins to take from the reference. Explicit, because our source moves other
#: frames on purpose: taking every differing origin would drag in the hip and waist parent
#: renames and undo the ``pelvis_link`` convention the whole stack keys off.
FRAME_JOINTS: tuple[str, ...] = (
    "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "logo_joint", "head_joint",
)

#: Link inertials to take from the reference.
INERTIAL_LINKS: tuple[str, ...] = ("torso_link", "waist_yaw_link", "waist_roll_link")

#: The Dex3-1 subtree our source carries, removed wholesale.
DEX3_LINKS: tuple[str, ...] = tuple(
    f"{side}_hand_{part}_link"
    for side in ("left", "right")
    for part in ("palm", "thumb_0", "thumb_1", "thumb_2",
                 "middle_0", "middle_1", "index_0", "index_1")
)

#: Its replacement, taken from the reference by name.
RUBBER_HANDS: tuple[str, ...] = ("left_rubber_hand", "right_rubber_hand")

#: Fixed joints the merge must NOT fold away, for
#: :mod:`merge_urdf_joints` -- the hands have to stay their own links so the depth cameras
#: can see them (``MultiMeshRayCaster`` only targets real link prims).
KEEP_JOINTS: tuple[str, ...] = ("left_hand_palm_joint", "right_hand_palm_joint")

#: How exactly the mass post-condition must close. Tight, because both sides of it are read
#: out of the same two files -- there is no rounding to absorb. Transcribing the expected
#: figure instead cost exactly this: a 1 ug rounding of the Dex3 mass tripped the guard.
MASS_TOL_KG = 1e-9


def expected_delta_kg(ours: ET.Element, ref: ET.Element, links=INERTIAL_LINKS,
                      dex3=DEX3_LINKS, hands=RUBBER_HANDS) -> float:
    """What the four delta classes should come to, derived from the two files.

    NOT the difference between the two totals: upstream also dropped
    ``pelvis_contour_link`` and ``waist_support_link`` (1 g visual tokens) and added sensor
    frames, none of which we take. This sums only what the patch actually does, so it is an
    independent statement of intent to check the edits against.

    Args:
        ours: Our ``<robot>``, BEFORE patching.
        ref: The reference ``<robot>``.
        links: Links whose inertial is replaced.
        dex3: Links removed.
        hands: Links added.

    Returns:
        Expected change in total mass, kg.
    """
    def mass(root: ET.Element, name: str) -> float:
        e = _index(root, "link")[name].find("inertial/mass")
        return 0.0 if e is None else float(e.get("value"))

    return (sum(mass(ref, n) - mass(ours, n) for n in links)
            - sum(mass(ours, n) for n in dex3)
            + sum(mass(ref, n) for n in hands))


def _index(root: ET.Element, tag: str) -> dict[str, ET.Element]:
    """Map name -> element for every ``tag`` child of ``root``."""
    return {e.get("name"): e for e in root.findall(tag)}


def total_mass(root: ET.Element) -> float:
    """Sum every link's ``<inertial><mass>``, in kg."""
    return sum(float(m.get("value", 0.0)) for m in root.iterfind("link/inertial/mass"))


def _fmt(elem: ET.Element | None, *keys: str) -> str:
    """Compact ``k=v`` rendering of an element's attributes, for the change log."""
    if elem is None:
        return "(absent)"
    return " ".join(f"{k}={elem.get(k)}" for k in keys if elem.get(k) is not None)


def patch_limits(ours: ET.Element, ref: ET.Element) -> int:
    """Take every differing ``effort``/``velocity`` from the reference. Returns the count.

    Only those two attributes: ``lower``/``upper`` agree everywhere already, and taking
    them wholesale would be a silent channel for a range change nobody reviewed.

    Args:
        ours: Our ``<robot>`` element, edited in place.
        ref: The reference ``<robot>``.

    Returns:
        How many joints changed.
    """
    ref_j = _index(ref, "joint")
    n = 0
    for name, joint in _index(ours, "joint").items():
        lim, r = joint.find("limit"), ref_j.get(name)
        if lim is None or r is None or r.find("limit") is None:
            continue
        rlim = r.find("limit")
        changed = {k: rlim.get(k) for k in ("effort", "velocity")
                   if rlim.get(k) is not None and rlim.get(k) != lim.get(k)}
        if not changed:
            continue
        print(f"    {name:32s} {_fmt(lim, 'effort', 'velocity')}"
              f"  ->  {_fmt(rlim, 'effort', 'velocity')}")
        for k, v in changed.items():
            lim.set(k, v)
        n += 1
    return n


def patch_frames(ours: ET.Element, ref: ET.Element, joints=FRAME_JOINTS) -> int:
    """Take the listed joint origins from the reference. Returns the count."""
    ref_j = _index(ref, "joint")
    n = 0
    for name in joints:
        joint, r = _index(ours, "joint").get(name), ref_j.get(name)
        if joint is None or r is None:
            raise ValueError(f"{name} is missing from our source or the reference.")
        o, ro = joint.find("origin"), r.find("origin")
        if o.get("xyz") == ro.get("xyz") and o.get("rpy", "0 0 0") == ro.get("rpy", "0 0 0"):
            continue
        print(f"    {name:32s} xyz {o.get('xyz')}  ->  {ro.get('xyz')}")
        o.set("xyz", ro.get("xyz"))
        if ro.get("rpy") is not None:
            o.set("rpy", ro.get("rpy"))
        n += 1
    return n


def patch_inertials(ours: ET.Element, ref: ET.Element, links=INERTIAL_LINKS) -> int:
    """Take the listed link inertials from the reference. Returns the count."""
    ref_l = _index(ref, "link")
    our_l = _index(ours, "link")
    n = 0
    for name in links:
        link, r = our_l.get(name), ref_l.get(name)
        if link is None or r is None:
            raise ValueError(f"{name} is missing from our source or the reference.")
        old, new = link.find("inertial"), r.find("inertial")
        print(f"    {name:20s} mass {old.find('mass').get('value'):>8s} -> "
              f"{new.find('mass').get('value'):>8s}   com "
              f"{old.find('origin').get('xyz')}  ->  {new.find('origin').get('xyz')}")
        link.remove(old)
        link.insert(0, copy.deepcopy(new))
        n += 1
    return n


def patch_hands(ours: ET.Element, ref: ET.Element, dex3=DEX3_LINKS,
                hands=RUBBER_HANDS) -> tuple[int, int]:
    """Swap the Dex3-1 subtree for the reference's rubber hands.

    The rubber hand takes the Dex3's own ``*_hand_palm_joint``, at the same origin, because
    it bolts to the same face -- so the joint is RETARGETED rather than replaced, and its
    origin is then taken from the reference like any other frame.

    Args:
        ours: Our ``<robot>`` element, edited in place.
        ref: The reference ``<robot>``.
        dex3: Link names to remove.
        hands: Reference link names to bring across.

    Returns:
        ``(links removed, links added)``.

    Raises:
        ValueError: If our source does not carry the expected Dex3 subtree, or the
            reference does not carry the hands.
    """
    our_l, ref_l = _index(ours, "link"), _index(ref, "link")
    missing = [n for n in dex3 if n not in our_l]
    if missing:
        raise ValueError(f"our source is missing {len(missing)} Dex3 links ({missing[:3]}); "
                         "it is not the asset this patch assumes.")
    absent = [n for n in hands if n not in ref_l]
    if absent:
        raise ValueError(f"the reference has no {absent}; wrong upstream variant?")

    doomed = set(dex3)
    for link in [l for l in ours.findall("link") if l.get("name") in doomed]:
        ours.remove(link)
    removed = len(doomed)
    # Drop the Dex3's INTERNAL joints but keep the palm joint, which the rubber hand reuses.
    ref_j = _index(ref, "joint")
    for joint in list(ours.findall("joint")):
        child = joint.find("child").get("link")
        if child in doomed and joint.get("name") not in ref_j:
            ours.remove(joint)

    for name in hands:
        ours.append(copy.deepcopy(ref_l[name]))
        palm = _index(ours, "joint")[f"{name.split('_')[0]}_hand_palm_joint"]
        old_child = palm.find("child").get("link")
        palm.find("child").set("link", name)
        r_origin = ref_j[palm.get("name")].find("origin")
        palm.find("origin").set("xyz", r_origin.get("xyz"))
        i = ref_l[name].find("inertial")
        print(f"    {palm.get('name'):26s} child {old_child} -> {name}"
              f"  ({float(i.find('mass').get('value')) * 1000:.0f} g at "
              f"{r_origin.get('xyz')})")
    return removed, len(hands)


def apply(src: str, dst: str, ref_path: str) -> None:
    """Write ``dst``: ``src`` with every mode_11 delta applied.

    Args:
        src: Our 29-DOF source URDF.
        dst: Output path.
        ref_path: The vendored upstream mode_11 URDF.

    Raises:
        ValueError: If the resulting mass change does not match
            :data:`EXPECTED_DELTA_KG`, which means a delta was missed or applied twice.
    """
    tree = ET.parse(src)
    ours = tree.getroot()
    ref = ET.parse(ref_path).getroot()
    before = total_mass(ours)
    expected = expected_delta_kg(ours, ref)
    print(f"[mode11] source    {os.path.relpath(src, HERE)}  ({before:.4f} kg, "
          f"{len(ours.findall('link'))} links)")
    print(f"[mode11] reference {os.path.relpath(ref_path, HERE)}  "
          f"({total_mass(ref):.4f} kg, {len(ref.findall('link'))} links)\n")

    print("  -- joint limits (effort / velocity) --")
    n_lim = patch_limits(ours, ref)
    print(f"\n  -- joint frames --")
    n_frm = patch_frames(ours, ref)
    print(f"\n  -- link inertials --")
    n_in = patch_inertials(ours, ref)
    print(f"\n  -- hands: Dex3-1 -> passive rubber --")
    n_del, n_add = patch_hands(ours, ref)

    after = total_mass(ours)
    delta = after - before
    print(f"\n[mode11] {n_lim} limits, {n_frm} frames, {n_in} inertials, "
          f"{n_del} links removed, {n_add} added")
    print(f"[mode11] mass {before:.4f} -> {after:.4f} kg  ({delta:+.9f}, "
          f"expected {expected:+.9f})")
    if abs(delta - expected) > MASS_TOL_KG:
        raise ValueError(
            f"mass moved {delta:+.9f} kg but the four delta classes account for "
            f"{expected:+.9f}. A delta was missed, applied twice, or the source changed. "
            "Re-derive against the reference before trusting this file."
        )
    ET.indent(tree, space="  ")
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    print(f"[mode11] wrote {dst}")
    print(f"[mode11] next: merge_urdf_joints.py {os.path.basename(dst)} <merged> "
          f"--keep {' '.join(KEEP_JOINTS)}")


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=DEFAULT_SRC, help="Our 29-DOF source URDF.")
    p.add_argument("--dst", default=DEFAULT_DST, help="Output URDF.")
    p.add_argument("--ref", default=DEFAULT_REF, help="Vendored upstream mode_11 URDF.")
    args = p.parse_args()
    apply(os.path.abspath(args.src), os.path.abspath(args.dst), os.path.abspath(args.ref))


if __name__ == "__main__":
    main()
