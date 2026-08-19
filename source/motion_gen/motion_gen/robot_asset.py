# Copyright 2026 Zachary Olkin. All rights reserved.

"""Portable content fingerprint for the ROBOT the artifacts were made with.

Every load-time refusal in this repo -- the clip-library sha256, the clip fingerprint,
``fm.onnx``'s rejected pre-rename metadata, the tracker/FM depth-spec agreement -- guards
the *data*. None of them guards the *robot*. So after the model changes, an old tracker
checkpoint, an old recording, an old clip library and an old ``fm.onnx`` all load cleanly
against a different robot, and the one near-guard that might have noticed
(``joint_names_isaac``) matches, because adding a backpack, swapping the hands and changing
a gear ratio touch link geometry, mass and limits -- not joint names or their order.

What this hashes is deliberately BOTH:

* **content** -- sha256 of each model description on disk. Strict, so a regenerated USD is a
  different robot even if the change was meant to be cosmetic. A whitespace edit trips it;
  that is the correct trade for a tripwire whose failure mode is otherwise silent.
* **semantics** -- joint names in order, body names, and total mass. These make the
  *message* actionable ("the backpack is missing", "29 joints became 30") instead of "two
  hex strings differ", and they are what a consumer can report even when the model files
  are not on the machine at all, as on a deploy box.

Stdlib-only apart from an optional ``mujoco`` import used solely to resolve the MJCF's total
mass, so the standalone checkers and ``source/motion_gen`` stay IsaacLab-free.
"""

from __future__ import annotations

import hashlib
import json
import os

# Bump when the digest inputs or the canonicalisation below change, so fingerprints from
# two algorithm versions can never collide.
ROBOT_ASSET_FINGERPRINT_VERSION = 1

#: Read in this many bytes at a time when hashing; the USD geometry payload is ~10 MB.
_CHUNK = 1 << 20


def _canonical(obj) -> str:
    """Deterministic JSON for hashing (sorted keys, compact, stable floats)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_file(path: str) -> str:
    """Streamed sha256 of one file.

    Args:
        path: File to hash.

    Returns:
        The hex digest, or ``""`` when the file does not exist -- a missing model is a
        legitimate state on a deploy box that carries only the ONNX pair, and the caller
        decides whether that is acceptable.
    """
    if not path or not os.path.isfile(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def sha256_tree(root: str) -> str:
    """Streamed sha256 over every file under ``root``, path-ordered.

    A USD asset is a directory of payload layers, not one file, so hashing the root layer
    alone would miss a regenerated ``geometries.usd``.

    Args:
        root: Directory to hash.

    Returns:
        The hex digest, or ``""`` when the directory does not exist.
    """
    if not root or not os.path.isdir(root):
        return ""
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            h.update(os.path.relpath(full, root).replace(os.sep, "/").encode())
            with open(full, "rb") as f:
                for block in iter(lambda: f.read(_CHUNK), b""):
                    h.update(block)
    return h.hexdigest()


def fingerprint_robot_asset(
    *,
    urdf_path: str | None = None,
    mjcf_path: str | None = None,
    usd_path: str | None = None,
    joint_names: list[str] | None = None,
    body_names: list[str] | None = None,
    total_mass_kg: float | None = None,
) -> dict:
    """Compute a portable fingerprint for the robot model an artifact was made with.

    Every argument is optional: a producer records what it can see. ``export_parameters``
    has the live articulation (so joint and body names, and the USD it spawned); the clip
    generator has only the FK MJCF. A fingerprint built from a subset is still comparable --
    :func:`verify_robot_asset` compares the fields both sides carry -- but see the
    ``covers`` field, which says what was actually looked at so a sparse fingerprint cannot
    agree by silence.

    Args:
        urdf_path: The URDF the USD was converted from.
        mjcf_path: The MuJoCo model (also ``motion_gen``'s FK model).
        usd_path: The ``.usda`` root layer; its containing directory is hashed, so the
            payload layers are covered too.
        joint_names: Actuated joint names, IN ORDER. Order is part of the identity: it is
            the PhysX BFS order every exported artifact indexes by.
        body_names: Rigid-body names; order is not significant, so they are sorted.
        total_mass_kg: Total body mass. The cheapest semantic check that notices a payload.

    Returns:
        Dict with ``hash``, ``short``, ``version``, ``covers`` (the fields folded in), and
        the individual facts, so a mismatch message can say WHICH one moved.
    """
    facts: dict[str, object] = {}
    if urdf_path:
        facts["urdf_sha256"] = sha256_file(urdf_path)
        facts["urdf_name"] = os.path.basename(urdf_path)
    if mjcf_path:
        facts["mjcf_sha256"] = sha256_file(mjcf_path)
        facts["mjcf_name"] = os.path.basename(mjcf_path)
    if usd_path:
        facts["usd_sha256"] = sha256_tree(os.path.dirname(usd_path))
        facts["usd_name"] = os.path.basename(usd_path)
    if joint_names is not None:
        facts["joint_names"] = list(joint_names)
    if body_names is not None:
        facts["body_names"] = sorted(body_names)
    if total_mass_kg is not None:
        facts["total_mass_kg"] = round(float(total_mass_kg), 6)

    covers = sorted(facts)
    h = hashlib.sha256()
    h.update(f"robot-asset-fingerprint;version={ROBOT_ASSET_FINGERPRINT_VERSION};".encode())
    h.update(_canonical(facts).encode())
    digest = h.hexdigest()
    return {"hash": digest, "short": digest[:12],
            "version": ROBOT_ASSET_FINGERPRINT_VERSION, "covers": covers, **facts}


#: Fields worth naming individually in a mismatch message, most legible first.
_REPORTABLE = ("joint_names", "total_mass_kg", "body_names",
               "urdf_sha256", "mjcf_sha256", "usd_sha256")


def verify_robot_asset(recorded: dict | None, current: dict | None) -> tuple[bool, str]:
    """Compare a recorded robot fingerprint against the current robot.

    Compares only the fields BOTH sides carry, so a producer that could see just the MJCF
    still checks against one that saw everything. A pair that overlaps in nothing is a
    failure, not a pass: agreeing by silence is the failure mode this whole module exists
    to remove.

    Args:
        recorded: Fingerprint stored in the artifact (``None`` / ``{}`` when the artifact
            predates this check).
        current: Fingerprint of the robot about to be used.

    Returns:
        ``(ok, message)``. ``ok`` is False whenever a shared field differs, whenever there
        is no shared field to compare, or whenever either side is absent -- the caller
        decides whether an absent one is fatal, since every artifact made before this
        existed has none.
    """
    if not recorded:
        return False, ("no robot_asset fingerprint recorded in this artifact -- it predates "
                       "the check, so the robot it was made with cannot be verified")
    if not current:
        return False, "no robot_asset fingerprint for the current robot to compare against"

    if recorded.get("version") != current.get("version"):
        return False, (
            f"robot_asset fingerprint version {recorded.get('version')} vs "
            f"{current.get('version')}; recompute the recorded one or accept the mismatch"
        )

    shared = [k for k in _REPORTABLE if k in recorded and k in current]
    if not shared:
        return False, (
            f"robot_asset fingerprints share no comparable field (recorded covers "
            f"{recorded.get('covers')}, current covers {current.get('covers')}), so they "
            "cannot agree -- treating that as a mismatch rather than a pass"
        )

    diffs = [k for k in shared if recorded[k] != current[k]]
    if not diffs:
        return True, (f"robot asset match on {', '.join(shared)} "
                      f"({current.get('short', '')})")

    lines = [f"ROBOT ASSET MISMATCH: this artifact was made with a different robot "
             f"({recorded.get('short', '?')} vs {current.get('short', '?')})."]
    for k in diffs:
        rec, cur = recorded[k], current[k]
        if k == "joint_names":
            lines.append(f"  joint_names: {len(rec)} -> {len(cur)}"
                         + (f", first change at index {_first_diff(rec, cur)}"
                            if len(rec) == len(cur) else ""))
        elif k == "body_names":
            gone, added = sorted(set(rec) - set(cur)), sorted(set(cur) - set(rec))
            lines.append(f"  body_names: -{gone or '[]'} +{added or '[]'}")
        elif k == "total_mass_kg":
            lines.append(f"  total_mass_kg: {rec} -> {cur} ({cur - rec:+.4f} kg)")
        else:
            lines.append(f"  {k}: {str(rec)[:12]} -> {str(cur)[:12]}")
    return False, "\n".join(lines)


def _first_diff(a: list, b: list) -> int | str:
    """Index of the first differing element in two equal-length lists."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return "none"
