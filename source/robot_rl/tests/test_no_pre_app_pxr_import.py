# Copyright 2026 Zachary Olkin. All rights reserved.

"""No env cfg module may import ``pxr`` before ``AppLauncher`` runs.

Hydra composes the env cfg BEFORE ``launch_simulation``, so every module named by a
registered ``env_cfg_entry_point`` is imported while Kit is still down.  At that point the
only ``pxr`` on ``sys.path`` is pip's ``usd-core`` (25.8, pulled in by the asset pipeline --
``mujoco-usd-converter``, ``urdf-usd-converter``, ``usd-exchange``), not Kit's (25.11).  The
import binds ``sys.modules['pxr.*']`` to 25.8, Kit then boots a 25.11 C++ core against those
25.8 Python wrappers, every schema extension fails to register its wrappers, and the process
dies with SIGSEGV about a second into startup.

``terrains/__init__.py`` states the rule in a docstring.  Commit b21972e7 walked past it --
``g1_depth_env_cfg`` grew a module-scope
``from ...terrains.clip_terrain_importer import ClipTerrainImporter, create_decor_prim``
to read one string, ``ClipTerrainImporter.DECOR_PRIM_PATH``, at module scope -- and every
depth task became unlaunchable.  A docstring cannot fail a build; this can.

The check MUST run in a subprocess.  Importing anything that touches
``isaaclab.markers.visualization_markers`` pollutes ``sys.modules`` for the rest of the
session, and sibling tests in this directory do exactly that at their own module scope, so
an in-process assertion would pass or fail on test ordering rather than on the code.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

pytest.importorskip("isaaclab", reason="needs the full Isaac install")

import gymnasium as gym  # noqa: E402

import robot_rl  # noqa: F401, E402 -- registers the gym environments


#: Imports each module in turn, reporting the first one to pull in ``pxr``.
_PROBE = """
import importlib, json, sys

first_offender = None
for name in sys.argv[1:]:
    importlib.import_module(name)
    if first_offender is None and any(
        m == "pxr" or m.startswith("pxr.") for m in sys.modules
    ):
        first_offender = name

print(json.dumps({"first_offender": first_offender}))
"""


def _registered_env_cfg_modules() -> list[str]:
    """Every module hydra imports to compose a registered env cfg.

    Returns:
        Sorted, de-duplicated module names taken from the ``env_cfg_entry_point`` of every
        registered task, which are ``"<module>:<class>"`` strings.
    """
    modules: set[str] = set()
    for spec in gym.registry.values():
        entry = (spec.kwargs or {}).get("env_cfg_entry_point")
        if isinstance(entry, str) and ":" in entry and entry.startswith("robot_rl"):
            modules.add(entry.split(":", 1)[0])
    return sorted(modules)


def _first_module_to_import_pxr(modules: list[str]) -> str | None:
    """Import *modules* in a clean interpreter and name the first that binds ``pxr``.

    Args:
        modules: Module names to import, in order.

    Returns:
        The offending module name, or ``None`` if none of them imported ``pxr``.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, *modules],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])["first_offender"]


def test_there_are_registered_env_cfg_modules_to_check() -> None:
    """A silent empty list would make the real test vacuously true."""
    modules = _registered_env_cfg_modules()
    assert modules, "no registered env_cfg_entry_point found -- the probe below proves nothing"


def test_no_registered_env_cfg_module_imports_pxr() -> None:
    """Composing any registered env cfg must leave ``pxr`` unbound."""
    modules = _registered_env_cfg_modules()
    offender = _first_module_to_import_pxr(modules)
    assert offender is None, (
        f"{offender} imports pxr at module scope. Hydra imports it before AppLauncher runs, "
        f"so this binds pip's usd-core instead of Kit's USD and Kit segfaults at startup. "
        f"Move the offending import inside the function that needs it, or read the value from "
        f"a dependency-free module (see terrains/decor.py)."
    )


def test_importing_robot_rl_does_not_import_pxr() -> None:
    """``train.py`` imports ``robot_rl`` at module scope to register the environments."""
    assert _first_module_to_import_pxr(["robot_rl"]) is None, (
        "importing robot_rl binds pxr; train.py does this at module scope, before AppLauncher"
    )
