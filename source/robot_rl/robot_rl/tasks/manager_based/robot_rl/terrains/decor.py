# Copyright 2026 Zachary Olkin. All rights reserved.

"""The decoration prim's path, in a module that imports nothing.

:class:`~...terrains.clip_terrain_importer.ClipTerrainImporter` owns the decoration prim,
but the depth camera cfgs need its PATH at module scope to list it as a ray-cast target --
and importing the importer to read that one string drags in
``isaaclab.terrains.terrain_importer`` -> ``isaaclab.markers.visualization_markers`` ->
``pxr``.  Hydra composes the env cfg BEFORE ``AppLauncher`` runs, so a pre-app ``pxr`` bind
resolves to whatever USD is on ``sys.path`` (pip's ``usd-core``) rather than Kit's, and Kit
then boots a mismatched C++ core and segfaults.  See the note in ``terrains/__init__.py``.

Both sides read the constant from here so there is still exactly one source of truth.
"""

from __future__ import annotations

#: The visual-only decoration prim the depth cameras cast against.
DECOR_PRIM_PATH: str = "/World/decor"
