# Copyright 2026 Zachary Olkin. All rights reserved.

"""Terrain importer that assembles a clip-library terrain directly (no grid).

Instead of IsaacLab's uniform :class:`TerrainGeneratorCfg` grid (which forces one
footprint on every cell -> heavy flat padding), this importer builds the terrain
mesh from a clip library's manifest with per-class tile footprints
(:func:`...terrains.clip_terrain_loader.assemble_clip_terrain`) and hands it to the
base :class:`TerrainImporter` via the same two public calls the grid path uses:
:meth:`import_mesh` + :meth:`configure_env_origins`.

It exposes, per env (following the ``terrain_levels`` / ``terrain_types`` mapping
the base curriculum sets up):

* :attr:`env_terrain_id` -- the terrain id of the env's tile (``-1`` = flat),
  matching the clips' per-clip ``terrain_id`` for assignment;
* :attr:`env_cell_corner` / :attr:`env_cell_center` -- the tile's local-corner /
  center in world, for placing a clip on the tile
  (``clip_terrain_loader.clip_frame_origin``).
"""

from __future__ import annotations

import torch
from typing_extensions import override

import isaaclab.sim as sim_utils
from isaaclab.terrains.terrain_importer import TerrainImporter

from motion_gen.clips.clip_fingerprint import fingerprint_terrain_manifest
from motion_gen.clips.clip_io import load_clip_library
from .clip_terrain_loader import assemble_clip_terrain


#: One tiny triangle, 1 km below the world, standing in for "no decoration".
#:
#: The decoration prim cannot simply be omitted, and cannot be an empty Xform either.  Every
#: depth camera lists it as a ray-cast target, and ``MultiMeshRayCaster`` fails BOTH ways: a
#: target matching no prim raises ``"Failed to find a prim at path expression"``, and a target
#: matching a prim that holds no meshes is warned about, skipped -- and then reaches the
#: second pass with ``n_meshes = 0``, where ``pos_w.view(-1, 0, 3)`` raises
#: ``"shape '[-1, 0, 3]' is invalid for input of size 3"``.  So the target must resolve to at
#: least one real mesh.
#:
#: This one can never be hit: it sits 1 km down, so any ray reaching it has already passed
#: the ground, and even through a ``has_floor=False`` void it is two orders of magnitude
#: beyond ``max_dist`` and reads as the range sentinel exactly like a miss.  It is
#: non-degenerate on purpose -- a zero-area triangle is a coin flip in a BVH build.
_PLACEHOLDER_POINTS = ((0.0, 0.0, -1000.0), (0.001, 0.0, -1000.0), (0.0, 0.001, -1000.0))


def create_decor_prim(decor_mesh, prim_path: str) -> int:
    """Create the visual-only decoration prim, with a placeholder when there is nothing to show.

    ``create_prim_from_mesh`` hardcodes ``collision_enabled=True``, so it cannot be used:
    decoration that collides is a wall the robot walks into having never been trained to avoid
    one.  This builds the same prim structure with the collider left off.

    Shared with the flat-mesh probe scenes (``verify_depth_self_scan.py``,
    ``compare_depth_sim2sim.py``) via ``g1_depth_env_cfg.ensure_decor_prim``, which build the
    real cameras over a plain ``TerrainImporter`` and so have no decoration of their own.

    Args:
        decor_mesh: The assembled decoration, or ``None`` for none at all.
        prim_path: Where to build it, normally :attr:`ClipTerrainImporter.DECOR_PRIM_PATH`.

    Returns:
        The face count actually imported; ``0`` when the placeholder was used.
    """
    import numpy as np
    from pxr import UsdGeom

    empty = decor_mesh is None or not len(decor_mesh.faces)
    points = _PLACEHOLDER_POINTS if empty else decor_mesh.vertices
    faces = np.asarray([[0, 1, 2]]) if empty else decor_mesh.faces

    sim_utils.create_prim(prim_path, "Xform")
    prim = sim_utils.create_prim(
        f"{prim_path}/mesh", "Mesh",
        attributes={
            "points": points,
            "faceVertexIndices": np.asarray(faces).flatten(),
            "faceVertexCounts": np.asarray([3] * len(faces)),
            "subdivisionScheme": "bilinear",
        },
    )
    # Deliberately NO define_collision_properties: this geometry must not collide.
    colour = UsdGeom.Primvar(prim.GetAttribute("primvars:displayColor"))
    colour.Set([(0.45, 0.42, 0.50)])
    return 0 if empty else len(faces)


class ClipTerrainImporter(TerrainImporter):
    """Custom-mesh importer for clip-library terrains with per-env terrain id."""

    cfg: "object"

    env_terrain_id: torch.Tensor
    """[num_envs] int64 terrain id of each env's tile (``-1`` = flat)."""

    env_cell_corner: torch.Tensor
    """[num_envs, 3] world position of each env's tile local (0,0) corner (z=0)."""

    env_cell_center: torch.Tensor
    """[num_envs, 3] world position of each env's tile center (z=0)."""

    env_clip_x_offset: torch.Tensor
    """[num_envs] x-shift of the clip within its tile (transition back-buffer)."""

    env_terrain_yaw: torch.Tensor
    """[num_envs] heading (rad) of each env's terrain frame -- the axis the traversal crosses the
    terrain along ("downrange").  The clip-tracking command resolves its anchor mask's downrange /
    lateral bits in this frame, so a terrain is never assumed to be laid out along world x.  0 for
    flat tiles and for manifests predating the field."""

    def __init__(self, cfg) -> None:
        # Mirror TerrainImporter.__init__'s buffer setup, but import a custom
        # assembled mesh instead of dispatching on cfg.terrain_type.
        self.cfg = cfg
        self.device = sim_utils.SimulationContext.instance().device  # type: ignore
        self.terrain_prim_paths = []
        self.terrain_origins = None
        self.env_origins = None
        self._terrain_flat_patches = {}

        if not isinstance(cfg.clip_library_path, str) or not cfg.clip_library_path:
            raise ValueError(
                "ClipTerrainImporter has no clip_library_path: the terrain mesh is BUILT from "
                "the clip library, and there is deliberately no default (a stale one silently "
                "builds the wrong terrain). Pass the library on the command line with "
                "--clip <lib.h5> -- it sets scene.terrain.clip_library_path and the tracking "
                "command's commands.traj_ref.path to the same file."
            )
        lib = load_clip_library(cfg.clip_library_path, device="cpu")
        mesh, origins, layout = assemble_clip_terrain(
            lib, num_rows=cfg.num_rows, num_flat_tiles=cfg.num_flat_tiles,
            col_gap=cfg.col_gap, row_gap=cfg.row_gap, size_margin=cfg.size_margin,
            flat_buffer=cfg.flat_buffer, transition_back_buffer=cfg.transition_back_buffer,
        )
        self.import_mesh("terrain", mesh)
        self._import_decor(layout.decor_mesh)
        # Base curriculum origin distribution: terrain_levels (rows) + terrain_types
        # (columns == tiles); env_origins = origins[level, type] = tile center.
        self.configure_env_origins(origins)

        # WHICH TERRAIN this env is standing on, as a content hash of the manifest the mesh
        # was just built from. The clip library's own `clip_fingerprint` does not cover the
        # terrain (and in practice is never written at all), so an offline stage that
        # rewrites tile geometry while leaving the motions alone -- terrain decoration --
        # is otherwise invisible to every artifact downstream. Recorded into tracker.onnx
        # by export_parameters; not enforced here, since the importer has nothing to
        # compare against at env build.
        self.terrain_fingerprint = fingerprint_terrain_manifest(lib.attrs.get("terrain_manifest"))

        self._layout = layout
        self._tile_terrain_id = layout.tile_terrain_id.to(self.device)
        self._tile_corner = layout.tile_corner.to(self.device)
        self._tile_footprint = layout.tile_footprint.to(self.device)
        self._tile_clip_x_offset = layout.tile_clip_x_offset.to(self.device)
        self.clip_size_y = layout.clip_size_y
        self.terrain_type_code = layout.terrain_type_code
        self.terrain_frame_yaw = layout.terrain_frame_yaw
        # Per-tile terrain-frame heading, tabled for the per-env gather below. The tile's mesh is
        # emitted axis-aligned and translated onto its corner, so a terrain's world frame yaw IS its
        # authored tile-local frame yaw.
        self._tile_frame_yaw = torch.tensor(
            [[float(layout.terrain_frame_yaw.get(int(t), 0.0)) for t in row]
             for row in layout.tile_terrain_id.tolist()],
            dtype=torch.float32, device=self.device,
        )
        self._refresh_env_arrays()

        self.set_debug_vis(cfg.debug_vis)

    #: Where the visual-only decoration goes. A SIBLING of the ground, not a child.
    #:
    #: ``import_mesh`` builds under ``cfg.prim_path`` (``/World/ground``), and the height
    #: scanner is a ``RayCaster`` pointed at exactly that path -- so a decor mesh imported the
    #: normal way would be scanned along with the terrain, which is the one thing the channel
    #: exists to prevent. The depth cameras take a LIST of prims and are given this one
    #: explicitly (``g1_depth_env_cfg``).
    DECOR_PRIM_PATH = "/World/decor"

    def _import_decor(self, decor_mesh) -> None:
        """Import the decoration mesh as a visual-only prim.

        Args:
            decor_mesh: The assembled decoration, or ``None`` for an undecorated library.
        """
        n = create_decor_prim(decor_mesh, self.DECOR_PRIM_PATH)
        if n:
            print(f"[ClipTerrainImporter] decoration: {n} faces at {self.DECOR_PRIM_PATH} "
                  "(visual only -- no collider, not in the height scan)")
        else:
            print(f"[ClipTerrainImporter] no decoration in this library; "
                  f"{self.DECOR_PRIM_PATH} carries the unhittable placeholder")

    @override
    def update_env_origins(self, env_ids: torch.Tensor, move_up: torch.Tensor, move_down: torch.Tensor):
        super().update_env_origins(env_ids, move_up, move_down)
        if self.terrain_origins is not None:
            self._refresh_env_arrays()

    def _refresh_env_arrays(self) -> None:
        """Per-env terrain id + tile corner/center from the env->tile mapping."""
        lv, ty = self.terrain_levels, self.terrain_types
        self.env_terrain_id = self._tile_terrain_id[lv, ty]
        self.env_cell_corner = self._tile_corner[lv, ty]
        self.env_clip_x_offset = self._tile_clip_x_offset[lv, ty]
        self.env_terrain_yaw = self._tile_frame_yaw[lv, ty]
        fp = self._tile_footprint[lv, ty]
        center = self.env_cell_corner.clone()
        center[:, 0] += fp[:, 0] / 2.0
        center[:, 1] += fp[:, 1] / 2.0
        self.env_cell_center = center
