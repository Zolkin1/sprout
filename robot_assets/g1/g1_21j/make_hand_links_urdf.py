# Copyright 2026 Zachary Olkin. All rights reserved.

"""Promote the G1's hand VISUALS into real fixed-joint links.

The stock URDF hangs all eight hand meshes per side off ``*_wrist_yaw_link`` as extra
``<visual>`` elements.  The URDF importer turns each one into a bare Xform parented to the
wrist's OUTER Xform -- one level shallower than a real link, and carrying a non-identity
mesh->body offset.  Those are the only meshes in the robot with a non-identity offset, and
they are the only ones ``MultiMeshRayCaster`` fails to intersect: the depth cameras see
every limb but not the hands (``compare_depth_sim2sim.py`` shows MuJoCo hitting the palm on
50 far-camera pixels where IsaacLab hits the floor 1 m beyond).

Promoting each visual to its own link joined by a FIXED joint gives the hands exactly the
structure every working link already has -- own link Xform, own inner visual Xform, identity
mesh offset -- and matches the MuJoCo model, which has them as real bodies
(``<body name="left_palm_link" pos="0.0415 0.003 0">``).

The joint carries the visual's original origin, and the visual is re-zeroed inside the new
link, so the composed world pose is unchanged.  Geometry is untouched: the meshes are the
same files at the same places.

Mass: each new link gets a token inertial.  The wrist's existing inertial already accounts
for the hand, so giving these real mass would double-count it and change the dynamics the
policies were trained under.  A near-zero mass on a fixed-joint child keeps the articulation
inertially identical.

Usage (from the repo root)::

    python robot_assets/g1/g1_21j/make_hand_links_urdf.py
    python robot_assets/g1/g1_21j/urdf_to_usd_conversion.py --urdf <the new file>
"""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET

_HERE = os.path.dirname(os.path.abspath(__file__))

#: Token inertial for a welded, massless visual shell (see the module docstring).
_MASS = 1e-6
_INERTIA = 1e-9


def _indent(elem: ET.Element, level: int = 0) -> None:
    """Pretty-print an ElementTree in place, so the diff stays readable."""
    pad = "\n" + "  " * level
    if len(elem):
        if not (elem.text or "").strip():
            elem.text = pad + "  "
        for child in elem:
            _indent(child, level + 1)
        if not (elem[-1].tail or "").strip():
            elem[-1].tail = pad
    if level and not (elem.tail or "").strip():
        elem.tail = pad


def promote(src: str, dst: str) -> int:
    """Rewrite ``src`` into ``dst`` with every hand visual promoted to its own link.

    Args:
        src: Input URDF path.
        dst: Output URDF path.

    Returns:
        Number of links created.
    """
    tree = ET.parse(src)
    root = tree.getroot()
    created = 0

    for link in list(root.findall("link")):
        parent = link.get("name")
        for vis in list(link.findall("visual")):
            mesh = vis.find("geometry/mesh")
            if mesh is None:
                continue
            name = mesh.get("filename").rsplit("/", 1)[-1]
            if "hand" not in name:
                continue
            child = name[: -len(".STL")]
            origin = vis.find("origin")
            xyz = origin.get("xyz", "0 0 0")
            rpy = origin.get("rpy", "0 0 0")

            # Move the visual out of the wrist and zero its origin: the joint now
            # carries the placement.
            link.remove(vis)
            origin.set("xyz", "0 0 0")
            origin.set("rpy", "0 0 0")

            new_link = ET.SubElement(root, "link", {"name": child})
            inertial = ET.SubElement(new_link, "inertial")
            ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
            ET.SubElement(inertial, "mass", {"value": repr(_MASS)})
            ET.SubElement(inertial, "inertia", {
                "ixx": repr(_INERTIA), "ixy": "0", "ixz": "0",
                "iyy": repr(_INERTIA), "iyz": "0", "izz": repr(_INERTIA),
            })
            new_link.append(vis)

            joint = ET.SubElement(root, "joint",
                                  {"name": f"{child}_fixed_joint", "type": "fixed"})
            ET.SubElement(joint, "origin", {"xyz": xyz, "rpy": rpy})
            ET.SubElement(joint, "parent", {"link": parent})
            ET.SubElement(joint, "child", {"link": child})
            created += 1

    _indent(root)
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    return created


def main() -> None:
    """Generate the hand-link URDF next to the source."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=os.path.join(_HERE, "g1_21j_box_merged_29j.urdf"))
    ap.add_argument("--dst", default=os.path.join(_HERE, "g1_21j_box_merged_29j_hands.urdf"))
    args = ap.parse_args()

    n = promote(args.src, args.dst)
    tree = ET.parse(args.dst)
    links = tree.getroot().findall("link")
    joints = tree.getroot().findall("joint")
    movable = [j for j in joints if j.get("type") != "fixed"]
    print(f"wrote {args.dst}")
    print(f"  promoted {n} hand visuals to links")
    print(f"  links {len(links)}   joints {len(joints)}   movable joints {len(movable)}")


if __name__ == "__main__":
    main()
