# Copyright 2026 Zachary Olkin. All rights reserved.

"""Listing ``/World/decor`` as a ray-cast target makes the prim MANDATORY.

``MultiMeshRayCaster`` fails BOTH ways. A target matching no prim raises "Failed to find a
prim at path expression". A target matching a prim that holds no MESHES is only warned about
-- and then reaches the second pass with ``n_meshes = 0``, where ``pos_w.view(-1, 0, 3)``
raises "shape '[-1, 0, 3]' is invalid for input of size 3". So an empty Xform is not a way
out: the target has to resolve to a real mesh. Three scenes therefore had to start building
one the moment the depth cameras gained the decoration target:

* ``ClipTerrainImporter`` on an UNDECORATED library -- it used to return before creating the
  prim when there was no decoration, which would have made ``g1_mb_v5_body_mode11.h5`` (or any
  pre-decoration library) unusable with every depth task;
* the flat-mesh probe scenes in ``verify_depth_self_scan.py`` and ``compare_depth_sim2sim.py``,
  which build the real cameras over a plain ``TerrainImporter`` and have no decoration path at
  all. Both died on this, which is how it was found -- and both are Part 2.7 acceptance gates,
  so they are exactly the scripts that must not be broken by a terrain change.

The failure is loud, but it is loud at ``sim.reset()`` inside Kit, minutes into a run.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

pytest.importorskip("isaaclab", reason="needs the full Isaac install")

from robot_rl.tasks.manager_based.robot_rl.g1 import g1_depth_env_cfg as depth_cfg  # noqa: E402
from robot_rl.tasks.manager_based.robot_rl.terrains import (  # noqa: E402
    clip_terrain_importer as importer_mod)
from robot_rl.tasks.manager_based.robot_rl.terrains.clip_terrain_importer import (  # noqa: E402
    ClipTerrainImporter)


class _Sink:
    """Stands in for a USD Primvar: swallows the display-colour write."""

    def Set(self, _value):
        return None

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "rsl_rl"


def test_the_two_names_are_one_name():
    """The cfg re-exports the importer's constant rather than restating the string."""
    assert depth_cfg.DECOR_PRIM_PATH == ClipTerrainImporter.DECOR_PRIM_PATH


def test_every_depth_camera_targets_it():
    """If this ever stops being true the rest of the file is moot -- and so is the decoration."""
    for spec in (depth_cfg._NEAR, depth_cfg._FAR):
        paths = depth_cfg._make_depth_camera(spec).mesh_prim_paths
        assert depth_cfg.DECOR_PRIM_PATH in paths, spec


def test_an_undecorated_library_still_gets_the_prim(monkeypatch):
    """The early return must come AFTER the Xform, not before it.

    Exercised directly rather than through a sim: ``_import_decor`` needs nothing off the
    instance but the class attribute, so a stand-in plus a stubbed ``create_prim`` is a real
    behavioural check with no Kit involved.
    """
    created: list[tuple[str, str, dict]] = []

    class _FakePrim:
        def GetAttribute(self, _name):
            return None

    def _fake_create(path, kind, **kw):
        created.append((path, kind, kw.get("attributes", {})))
        return _FakePrim()

    monkeypatch.setattr(
        "robot_rl.tasks.manager_based.robot_rl.terrains.clip_terrain_importer.sim_utils"
        ".create_prim", _fake_create)
    monkeypatch.setattr("pxr.UsdGeom.Primvar", lambda _attr: _Sink())

    n = importer_mod.create_decor_prim(None, ClipTerrainImporter.DECOR_PRIM_PATH)
    assert n == 0, "no decoration means no real faces"
    kinds = [(p, k) for p, k, _ in created]
    assert kinds == [(ClipTerrainImporter.DECOR_PRIM_PATH, "Xform"),
                     (f"{ClipTerrainImporter.DECOR_PRIM_PATH}/mesh", "Mesh")], (
        "an undecorated library must still leave a MESH behind -- an empty Xform reaches the "
        "caster's second pass with n_meshes = 0 and dies on the reshape"
    )
    attrs = created[-1][2]
    assert len(attrs["faceVertexCounts"]) == 1 and len(attrs["points"]) == 3
    zs = [p[2] for p in attrs["points"]]
    assert max(zs) < -100.0, (
        f"the placeholder must sit far below the world so no ray can reach it, got z={zs}")


@pytest.mark.parametrize("script", ["verify_depth_self_scan.py", "compare_depth_sim2sim.py"])
def test_the_flat_mesh_probe_scenes_create_it(script):
    """These build the real cameras with no terrain importer to make the prim for them."""
    src = (_SCRIPTS / script).read_text(encoding="utf-8")
    assert "_make_depth_camera" in src, f"{script} no longer builds the real cameras"
    assert "ensure_decor_prim()" in src, (
        f"{script} builds the depth cameras over a plain TerrainImporter, so it must call "
        "ensure_decor_prim() before sim.reset() or the caster raises on /World/decor"
    )
    # Before, not after: the sensors initialise inside sim.reset().
    assert src.index("ensure_decor_prim()") < src.index("sim.reset()"), (
        f"{script} calls ensure_decor_prim() after sim.reset(), which is too late -- the "
        "ray-caster resolves its targets in the reset callback"
    )


def test_ensure_decor_prim_is_idempotent():
    """It runs in scripts that may already have a decorated terrain; twice must be safe."""
    src = inspect.getsource(depth_cfg.ensure_decor_prim)
    assert "is_prim_path_valid" in src, (
        "ensure_decor_prim must check before creating: called on a scene that already has "
        "decoration it would otherwise stomp the real mesh with the placeholder"
    )
