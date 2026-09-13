# Copyright 2026 Zachary Olkin. All rights reserved.
"""Shared plumbing for writing a MANIFEST-ONLY clip library, and checking it loads.

Every fixed-course generator in `scripts/rsl_rl/` writes the same shape of artifact -- terrain
and zero clips -- and needs the same two things around it: the channel-layout metadata a
clip-free library must still carry, and a load-back check that the thing it just wrote can be
opened, managed and turned into a mesh.

**This lives here because it was living in `make_freeform_terrain.py`.** Four current builders
imported `library_meta` and `verify` from that script, so the free-form section generator --
which nothing fine-tunes on any more -- was a load-bearing dependency of every course that
replaced it. Moving it makes the deprecation possible and the dependency honest.

Nothing here knows what is IN the manifest, which is what lets one copy serve the two-box
course, the up-into-down course, the varied set and the in-distribution routes.
"""

from __future__ import annotations

import json
import pathlib
import sys

import h5py

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _pkg in ("motion_gen", "robot_rl"):
    _p = str(_REPO_ROOT / "source" / _pkg)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from motion_gen.clips.clip_io import load_clip_library  # noqa: E402
from motion_gen.clips.clip_manager import MotionClipManager  # noqa: E402

#: The metadata KEY SET a clip-free library must declare, with a coherent minimal value for
#: each.  It is the schema, and it is NOT a usable layout: see :func:`library_meta` for why
#: ``--like`` is mandatory.
_MINIMAL_META: dict = {
    "dt": 0.02,
    "num_steps": 1,
    "clip_length_s": 0.02,
    "pos_output_names": ["pelvis_link:pos_x", "pelvis_link:pos_y", "pelvis_link:pos_z",
                         "pelvis_link:ori_w", "pelvis_link:ori_x", "pelvis_link:ori_y",
                         "pelvis_link:ori_z"],
    "vel_output_names": ["pelvis_link:pos_x", "pelvis_link:pos_y", "pelvis_link:pos_z",
                         "pelvis_link:ori_x", "pelvis_link:ori_y", "pelvis_link:ori_z"],
    "contact_frames": ["left_ankle_roll_link", "right_ankle_roll_link"],
    "ref_frame": "pelvis_link",
    "skill_list": ["standing"],
}

_META_KEYS: tuple[str, ...] = tuple(_MINIMAL_META)


def library_meta(like: str) -> dict:
    """The channel-layout metadata to stamp on the terrain-only library.

    **``like`` is MANDATORY, and the reason is a three-minute silent failure.** The layout was
    documented here as "pure bookkeeping -- nothing reads it without clips to index", and that
    is false: ``BaseTrajectoryCommand._reference_world_indices`` resolves one output index per
    ROBOT JOINT at command construction, whether or not a clip is ever gathered.  A library
    carrying only the seven base-frame channels therefore writes, loads, passes every check in
    :func:`verify`, and then dies inside Isaac with ``Joint output
    'joint:left_hip_pitch_joint' missing from trajectory outputs`` -- after the simulator has
    booted.  Every clip-free course on disk that has ever run carries the full 155/137 layout.

    Args:
        like: A source library whose channel layout to copy.  Any real 29-joint library will
            do -- the layout is identical across them -- but prefer the one the run's tracker
            was trained on.

    Returns:
        The metadata dict.

    Raises:
        ValueError: If ``like`` is ``None``, or names a library with no joint channels.
        KeyError: If ``like`` is missing one of the layout attributes.
    """
    if like is None:
        raise ValueError(
            "library_meta needs a --like library to copy the channel layout from. The minimal "
            "layout is a schema, not a usable one: a clip-free library still has to name one "
            "output per robot joint, because the trajectory command resolves them at "
            "construction. Pass e.g. --like clips/g1_mb_v6g_decor.h5.")
    with h5py.File(like, "r") as f:
        meta = {k: json.loads(f.attrs[k]) for k in _META_KEYS}
    assert_layout_is_usable(meta, source=like)
    return meta


def assert_layout_is_usable(meta: dict, source: str) -> None:
    """Refuse a channel layout the trajectory command cannot be constructed on.

    Args:
        meta: A layout dict as returned by :func:`library_meta`.
        source: Where it came from, for the message.

    Raises:
        ValueError: If the position outputs carry no ``joint:`` channels.
    """
    n_joint = sum(1 for n in meta["pos_output_names"] if str(n).startswith("joint:"))
    if n_joint == 0:
        raise ValueError(
            f"{source} declares {len(meta['pos_output_names'])} position outputs and NONE of "
            "them is a 'joint:' channel. The trajectory command resolves one index per robot "
            "joint at construction, so this layout cannot build an env -- it fails after the "
            "simulator has booted, not here.")



def verify(path: str, manifest: list[dict]) -> None:
    """Load the file back, build a clip manager on it, and build its mesh.

    Three checks, each catching a different way this file can be wrong, and the middle one is
    the reason the whole free-form task could not be constructed before: a clip-free library
    must still answer every LAYOUT question :class:`MotionClipManager` asks, because the command
    term builds one before anything knows whether a clip will ever be read.  Failing here costs
    a second; failing in Isaac costs a simulator launch.

    ``assemble_clip_terrain`` needs ``isaaclab.utils`` but no simulator.  Where it is missing the
    first two halves still run and the mesh half is reported as skipped rather than passed.

    Args:
        path: The written library.
        manifest: The manifest that was written, to check the mesh against.
    """
    lib = load_clip_library(path)
    assert lib.num_clips == 0, f"expected a clip-free library, got {lib.num_clips} clips"
    assert not lib.has_clips, "has_clips must be False on a clip-free library"
    assert lib.attrs["terrain_manifest"], "terrain_manifest did not survive the round trip"
    # The RANK is the point: ``num_steps`` is read off this axis, and an omitted dataset loads
    # rank-1 and takes it away.
    assert lib.y_des.ndim == 3, f"y_des must keep its (N, T, P) rank, got {tuple(lib.y_des.shape)}"
    print(f"[terrain] load-back OK: {lib.num_clips} clips, "
          f"{len(lib.attrs['terrain_manifest'])} tiles, "
          f"y_des {tuple(lib.y_des.shape)}, num_steps {lib.num_steps}")

    manager = MotionClipManager(lib)
    assert not manager.has_clips
    assert manager.num_steps == lib.num_steps
    # Everything the command term reads at construction, driven here so a layout that cannot
    # answer them is caught by the generator rather than by the env.
    manager.order_outputs(manager.pos_output_names, manager.vel_output_names)
    # ``order_outputs`` is happy with a base-frame-only layout; the COMMAND is not.  Checking
    # it here is what turns "dies three minutes into an Isaac launch" into "fails at build".
    assert_layout_is_usable({"pos_output_names": list(manager.pos_output_names)}, source=path)
    print(f"[terrain] clip manager OK: dt {manager.dt}, T {manager.num_steps}, "
          f"{manager.num_pos_outputs} pos / {manager.num_vel_outputs} vel outputs, "
          f"ref_frame {manager.ref_frame}, total_time {float(manager.get_total_time()):.3f} s "
          f"(every gather refuses; --like copies a real library's channel layout)")
    try:
        from robot_rl.tasks.manager_based.robot_rl.terrains.clip_terrain_loader import (
            assemble_clip_terrain, default_num_flat_tiles,
        )
    except ImportError as e:
        print(f"[terrain] mesh check SKIPPED (no isaaclab: {e})")
        return
    n_flat = default_num_flat_tiles(lib)
    assert n_flat == 0, f"expected 0 synthesised flat tiles, got {n_flat}"
    mesh, origins, layout = assemble_clip_terrain(lib)

    # The highest surface the manifest claims, against the mesh's own z extent: the two must
    # agree, or a slab was authored at a height the loader did not build.
    #
    # **Split by ``decor``, because the two land in DIFFERENT meshes.** ``assemble_clip_terrain``
    # returns the ground mesh and puts every ``decor=True`` overlay in ``layout.decor_mesh``, so
    # pooling them compares a walkable claim against a mesh the walls are not in. The free-form
    # families never emit decoration -- their stairwell walls are collidable -- so this only
    # bites a FIXED course that mirrors mjeval's visual-only walls
    # (``motion_gen.terrain.step_courses``), where it reported the 2.0 m wall top against a
    # 0.9 m platform. Both halves are checked, so a wall silently landing in the ground mesh
    # (or vice versa) fails here rather than at the first depth frame.
    def _top(want_decor: bool) -> float | None:
        """Highest surface among the manifest's decor / non-decor overlays, or None."""
        tops = [float(o["base_z"]) + float(o["size_xyz"][2])
                for m in manifest for o in m["overlays"]
                if bool(o.get("decor", False)) is want_decor]
        return max(tops) if tops else None

    claim = _top(False)
    got = float(mesh.bounds[1][2])
    assert claim is not None, "every tile is decoration; there is no ground to stand on"
    assert abs(claim - got) < 1e-6, f"ground mesh top {got} != manifest's highest walkable {claim}"
    decor_claim = _top(True)
    decor_mesh = getattr(layout, "decor_mesh", None)
    assert (decor_claim is None) == (decor_mesh is None), (
        f"the manifest has {'no ' if decor_claim is None else ''}decoration but the loader "
        f"built {'none' if decor_mesh is None else 'some'}")
    if decor_claim is not None:
        decor_got = float(decor_mesh.bounds[1][2])
        assert abs(decor_claim - decor_got) < 1e-6, (
            f"decor mesh top {decor_got} != manifest's highest decoration {decor_claim}")
    print(f"[terrain] mesh OK: {len(mesh.faces)} faces, {layout.num_tiles} tiles, "
          f"origins {tuple(origins.shape)}, z in [{mesh.bounds[0][2]:.3f}, {got:.3f}]"
          + ("" if decor_claim is None
             else f"; decor {len(decor_mesh.faces)} faces to z {decor_claim:.3f}"))
