# Copyright 2026 Zachary Olkin. All rights reserved.

"""Read a trajectory library's *layout* metadata without parsing the whole library.

The offline clip generators need exactly three things from the reference trajectory library:
the ordered position output names, the ordered velocity output names, and the skill list.
They then write those into the clip ``.h5`` so the RL side agrees on the layout.  That is a
few hundred bytes of schema; it does not require building the library's merged Bezier
tensors, which is what instantiating the full trajectory manager used to do.

The output-name derivation is the SAME function the trajectory manager used
(:func:`output_names_from_bezier`, moved here verbatim), so the ordering contract is preserved
by construction rather than by reimplementation.  Ordering matters: the names are positional
indices into every clip's ``y_des`` / ``dy_des``, so a reordering silently corrupts every clip
generated afterwards.  Note the asymmetry -- ``pos`` includes each frame's ``ori_w`` while
``vel`` omits it, because an angular velocity has only three components.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .library_fingerprint import fingerprint_skill_folders

# libyaml-backed loader where available (same choice the trajectory manager made).
try:
    _YAML_LOADER = yaml.CSafeLoader
except AttributeError:  # PyYAML built without libyaml
    _YAML_LOADER = yaml.SafeLoader


@dataclass(frozen=True)
class LibraryMetadata:
    """The layout contract a clip library inherits from its source trajectory library."""

    root: Path
    pos_output_names: list[str]
    vel_output_names: list[str]
    skill_list: list[str]
    fingerprint: dict   # see library_fingerprint.fingerprint_skill_folders

    @property
    def num_pos_outputs(self) -> int:
        return len(self.pos_output_names)

    @property
    def num_vel_outputs(self) -> int:
        return len(self.vel_output_names)


def output_names_from_bezier(bezier: dict) -> tuple[list[str], list[str]]:
    """Ordered ``(pos_output_names, vel_output_names)`` for one domain's bezier coefficients.

    Preserves YAML key order for both frames and joints.  ``pos`` includes every axis;
    ``vel`` drops each frame's ``ori_w`` (angular velocity has 3 components, not 4).

    Args:
        bezier: A domain's ``bezier_coeffs`` mapping, with ``frames`` and ``joints`` keys.

    Returns:
        The two ordered name lists, formatted ``"<frame>:<axis>"`` and ``"joint:<name>"``.
    """
    pos_output_names: list[str] = []
    vel_output_names: list[str] = []

    for frame in bezier["frames"].keys():
        for axis in bezier["frames"][frame].keys():
            pos_output_names.append(f"{frame}:{axis}")
            if "ori_w" not in axis:
                vel_output_names.append(f"{frame}:{axis}")

    for joint in bezier["joints"].keys():
        pos_output_names.append(f"joint:{joint}")
        vel_output_names.append(f"joint:{joint}")

    return pos_output_names, vel_output_names


def discover_skill_folders(root: Path) -> dict[str, Path]:
    """Map skill name -> folder for every immediate subdirectory holding YAMLs.

    Sorted, so the skill list (and therefore every clip's ``skill_id``) is deterministic.
    Falls back to a single ``"default"`` skill when the root itself holds the YAMLs.
    """
    if not root.exists():
        raise FileNotFoundError(f"Skill root folder not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root}")

    skill_folders: dict[str, Path] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if list(child.glob("*.yaml")) + list(child.glob("*.yml")):
            skill_folders[child.name] = child

    if not skill_folders:
        if list(root.glob("*.yaml")) + list(root.glob("*.yml")):
            skill_folders["default"] = root
        else:
            raise ValueError(f"No skill subfolders with YAML files found under: {root}")
    return skill_folders


def _first_bezier(skill_folders: dict[str, Path]) -> dict:
    """The first domain's ``bezier_coeffs`` from the first trajectory that has them.

    Not every trajectory carries them -- a ``standing`` perpetual solution, for instance, has
    no Bezier outputs at all -- so this scans in deterministic order until it finds one rather
    than assuming the first file is representative.  The trajectory manager verified that all
    domains of all trajectories agree on frames / axes / joints, so any one of them defines the
    layout; that cross-library check is not repeated here.
    """
    for skill in sorted(skill_folders):
        for path in sorted(skill_folders[skill].glob("*.yaml")) + sorted(skill_folders[skill].glob("*.yml")):
            with open(path, "r") as f:
                data = yaml.load(f, Loader=_YAML_LOADER)
            for domain in data.get("domain_sequence", []):
                bezier = data.get(domain, {}).get("bezier_coeffs")
                if bezier and "frames" in bezier and "joints" in bezier:
                    return bezier
    raise ValueError(
        "No trajectory in the library carries 'bezier_coeffs' with 'frames' + 'joints'; "
        "cannot determine the output-name layout."
    )


def read_library_metadata(source: str | Path, hf_repo: str | None = None,
                          verify_fingerprint: str | None = None) -> LibraryMetadata:
    """Read the output-name layout + skill list for a trajectory library.

    Args:
        source: Library root, or the in-repo path to fetch when ``hf_repo`` is given.
        hf_repo: Optional HuggingFace repo to fetch from (cached under ``./hf``).
        verify_fingerprint: When given, a full sha256 to check the library against; raises on mismatch.
            This is the guard against the local ``hf/`` cache having silently diverged from
            another machine's copy -- the loader reuses the cache and never re-checks the
            remote, so two machines can hold different libraries under the same path.

    Returns:
        The :class:`LibraryMetadata` for the resolved library.
    """
    from .hf_cache import resolve_library_path

    root = resolve_library_path(source, hf_repo)
    skill_folders = discover_skill_folders(root)
    pos_names, vel_names = output_names_from_bezier(_first_bezier(skill_folders))
    fingerprint = fingerprint_skill_folders(skill_folders)

    if verify_fingerprint is not None and fingerprint["hash"] != verify_fingerprint:
        raise ValueError(
            f"Trajectory library fingerprint mismatch for {root}.\n"
            f"  expected: {verify_fingerprint}\n"
            f"  actual:   {fingerprint['hash']}\n"
            "The local hf/ cache has diverged from the library this was recorded against."
        )

    return LibraryMetadata(
        root=root,
        pos_output_names=pos_names,
        vel_output_names=vel_names,
        skill_list=sorted(skill_folders),
        fingerprint=fingerprint,
    )
