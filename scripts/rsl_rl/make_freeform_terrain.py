# Copyright 2026 Zachary Olkin. All rights reserved.

"""Write a MANIFEST-ONLY clip library: free-form terrain with no clips in it.

The clip library is the spine -- it carries both the reference motions and the terrain they were
generated on -- and everything downstream reaches terrain through it: the RL importer, both
MuJoCo emitters, both viser viewers, ``terrain_fingerprint`` and the eval's tile sampler.  The
FM-driven fine-tune needs the second half and none of the first: its reference is generated
perceptively (``reference_source="fm"``), so no clip is ever read.

So this writes a library with a ``terrain_manifest`` and ZERO CLIPS -- the clip datasets are
present with their full channel layout and zero extent -- and it is passed as ``--clip`` exactly
like any other.  Most of the terrain side needed no special case: ``assert_manifest_covers_clips``
no-ops on an empty ``terrain_id`` and ``default_num_flat_tiles`` returns 0 with no flat clips and
``counts.flat == 0``.  What DID need stating is what "zero clips" means to the RL runtime, and it
is stated in exactly two places: ``ClipLibraryData.has_clips`` (the predicate) and
``MotionClipManager._require_clips`` (the refusal every gather goes through).  A clip-consuming
term left wired on this task therefore fails by name instead of quietly indexing nothing.

Usage::

    python scripts/rsl_rl/make_freeform_terrain.py --out clips/freeform_v1.h5 \\
        --seed 0 --flat 8 --boxes 16 --stairs 24

    # copy an existing library's channel layout, for a consumer that inspects it
    python scripts/rsl_rl/make_freeform_terrain.py --out clips/freeform_v1.h5 \\
        --like clips/g1_mb_v5_body_mode11_decor.h5

Then::

    python scripts/rsl_rl/view_terrain_viser.py --clip clips/freeform_v1.h5

.. note::
    The file goes through :class:`~motion_gen.clips.clip_io.ClipLibraryWriter` like every other
    library, at ``num_clips=0``.  It used to be written with raw ``h5py`` because that writer
    chunks each clip dataset ``(1, ...)``, which h5py refuses against a zero-extent clip axis
    ("Chunk shape must not be greater than data shape") -- so the datasets were simply OMITTED.
    That was the wrong half to give up: the loader's placeholder for an absent dataset is rank-1,
    and ``num_steps`` is read off ``y_des.shape[1]``, so ``MotionClipManager`` raised
    ``IndexError`` on the first line that touched it and the task could not be constructed at
    all.  The writer now drops the CHUNKING at zero clips (there is nothing to chunk) and keeps
    the SHAPES, which is what makes the library well-defined rather than merely loadable.

.. note::
    The terrain FINGERPRINT is printed and deliberately NOT stored.  ``ClipTerrainImporter``
    recomputes it from the manifest at load, so a stored copy would be a second statement of the
    same thing that any later manifest edit (``decorate_terrain.py``, ``author_clutter.py``)
    would leave stale.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import h5py

# <repo>/scripts/rsl_rl/make_freeform_terrain.py -> parents[2] == <repo>.  Derived rather than
# counting ``..`` hops, so moving this script cannot silently break the path.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _pkg in ("motion_gen", "robot_rl"):
    _p = str(_REPO_ROOT / "source" / _pkg)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from motion_gen.clips.clip_fingerprint import fingerprint_terrain_manifest  # noqa: E402
from motion_gen.clips.clip_io import ClipLibraryWriter, load_clip_library  # noqa: E402
from motion_gen.clips.clip_manager import MotionClipManager  # noqa: E402
from motion_gen.terrain.freeform import (  # noqa: E402
    FAMILIES,
    GENERATOR_TAG,
    FreeformConfig,
    generate_freeform_manifest,
    validate_manifest,
)

#: Channel layout for a library that carries no motion.  It is pure bookkeeping here -- nothing
#: reads it without clips to index -- but it has to be a coherent, non-empty layout, because the
#: loader's metadata keys are not optional and a zero-width name list would make the shapes
#: degenerate.  ``--like`` copies a real library's layout for a consumer that inspects it.
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


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="Output .h5 (a manifest-only clip library).")
    p.add_argument("--seed", type=int, default=0,
                   help="Base seed. A tile's geometry depends only on (seed, terrain_id).")
    p.add_argument("--flat", type=int, default=8, help="Flat sections.")
    p.add_argument("--boxes", type=int, default=16, help="Box-corridor sections.")
    p.add_argument("--stairs", type=int, default=24, help="Stair sections.")
    p.add_argument("--like", default=None,
                   help="Copy the channel layout (pos/vel names, dt, contact frames) from this "
                        "library instead of the minimal one. Motion is never copied.")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip the load-back + assemble_clip_terrain round trip.")
    return p.parse_args()


def library_meta(like: str | None) -> dict:
    """The channel-layout metadata to stamp on the terrain-only library.

    Args:
        like: A source library whose layout to copy, or ``None`` for :data:`_MINIMAL_META`.

    Returns:
        The metadata dict.

    Raises:
        KeyError: If ``like`` is missing one of the layout attributes.
    """
    if like is None:
        return dict(_MINIMAL_META)
    with h5py.File(like, "r") as f:
        return {k: json.loads(f.attrs[k]) for k in _META_KEYS}


def write_terrain_library(path: str, manifest: list[dict], meta: dict, cfg: FreeformConfig) -> None:
    """Write the manifest-only ``.h5``.

    Through the shared :class:`ClipLibraryWriter` at ``num_clips=0``, so the channel layout,
    the ``terrain_type_names`` list and the clip datasets' RANK are all stated by the one
    writer every other library uses -- and there is no second place stating what a library
    looks like.

    Args:
        path: Output file.
        manifest: The terrain manifest.
        meta: Channel-layout metadata (see :func:`library_meta`).
        cfg: The generation config, recorded so the file states how it was made.
    """
    counts = {"flat": int(cfg.num_flat), "boxes": int(cfg.num_boxes),
              "stairs": int(cfg.num_stairs)}
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with ClipLibraryWriter(path, num_clips=0, meta=meta) as writer:
        writer.set_attr("terrain_manifest", manifest)
        writer.set_attr("generator", GENERATOR_TAG)
        writer.set_attr("generator_config", {
            "seed": int(cfg.seed),
            # ``counts.flat`` is what ``default_num_flat_tiles`` reads, and it means flat CLIP
            # tiles -- the synthesised patches a flat clip is dropped onto.  There are none: the
            # flat SECTIONS here are manifest tiles like any other, so this is 0 and the
            # free-form counts live under their own key.
            "counts": {"flat": 0},
            "freeform": {"section_counts": counts},
        })


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
    print(f"[freeform] load-back OK: {lib.num_clips} clips, "
          f"{len(lib.attrs['terrain_manifest'])} tiles, "
          f"y_des {tuple(lib.y_des.shape)}, num_steps {lib.num_steps}")

    manager = MotionClipManager(lib)
    assert not manager.has_clips
    assert manager.num_steps == lib.num_steps
    # Everything the command term reads at construction, driven here so a layout that cannot
    # answer them is caught by the generator rather than by the env.
    manager.order_outputs(manager.pos_output_names, manager.vel_output_names)
    print(f"[freeform] clip manager OK: dt {manager.dt}, T {manager.num_steps}, "
          f"{manager.num_pos_outputs} pos / {manager.num_vel_outputs} vel outputs, "
          f"ref_frame {manager.ref_frame}, total_time {float(manager.get_total_time()):.3f} s "
          f"(every gather refuses; --like copies a real library's channel layout)")
    try:
        from robot_rl.tasks.manager_based.robot_rl.terrains.clip_terrain_loader import (
            assemble_clip_terrain, default_num_flat_tiles,
        )
    except ImportError as e:
        print(f"[freeform] mesh check SKIPPED (no isaaclab: {e})")
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
    print(f"[freeform] mesh OK: {len(mesh.faces)} faces, {layout.num_tiles} tiles, "
          f"origins {tuple(origins.shape)}, z in [{mesh.bounds[0][2]:.3f}, {got:.3f}]"
          + ("" if decor_claim is None
             else f"; decor {len(decor_mesh.faces)} faces to z {decor_claim:.3f}"))


def main() -> int:
    """Generate, validate, write and verify."""
    args = parse_args()
    cfg = FreeformConfig(seed=args.seed, num_flat=args.flat, num_boxes=args.boxes,
                         num_stairs=args.stairs)
    manifest = generate_freeform_manifest(cfg)
    validate_manifest(manifest)
    write_terrain_library(args.out, manifest, library_meta(args.like), cfg)

    fp = fingerprint_terrain_manifest(manifest)
    per_family = {fam: sum(1 for m in manifest if m["family"] == fam) for fam in FAMILIES}
    arms = {a: sum(1 for m in manifest if m.get("stair_sampling_arm") == a)
            for a in ("regulation", "trained")}
    print(f"[freeform] wrote {args.out}: {len(manifest)} tiles {per_family}, "
          f"stair arms {arms}, {fp['num_overlays']} overlays, "
          f"terrain fingerprint {fp['short']} (recomputed at load; not stored)")
    if not args.no_verify:
        verify(args.out, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
