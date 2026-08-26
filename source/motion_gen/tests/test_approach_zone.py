# Copyright 2026 Zachary Olkin. All rights reserved.

"""The approach zone is ONE constant with two consumers, and it must stay that way.

The clip generator decides where an approach stops tracking the command and starts transitioning
into the terrain; the visual-clutter pass vetoes an interactable-looking decoy whose approach zone
the robot's path would enter. Those are the same distance, and the whole reason the constant was
declared in ``clutter.py`` before anything honoured it was so that they could not diverge:

    "One declared number... the same distance at which a clip should stop tracking the
    steady-state command and start following the terrain motion. Today the clips have no such
    contract... so this is the definition arriving before the thing it will govern."

Two copies agreeing today is not the same property as one copy, which is what this file checks --
the same discipline the camera contract follows.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

from motion_gen.motionbricks import approach_zone
from motion_gen.terrain import clutter


def _repo_root() -> pathlib.Path:
    for parent in pathlib.Path(__file__).resolve().parents:
        if (parent / "scripts" / "rsl_rl").is_dir() and (parent / "source" / "motion_gen").is_dir():
            return parent
    raise RuntimeError(f"no checkout above {__file__}")


_REPO = _repo_root()


def test_the_clutter_pass_reads_the_generator_s_constant():
    assert clutter.APPROACH_ZONE_M is approach_zone.APPROACH_ZONE_M
    assert approach_zone.APPROACH_ZONE_M == 2.0


def test_nothing_declares_a_SECOND_copy():
    """Grep, because that is the failure: a module quietly assigning its own value.

    ``clutter.py`` used to hold 2.5 while nothing else held anything; the moment the generator
    started honouring a zone, that became two numbers that had to be kept equal by hand.
    """
    offenders = []
    for path in list((_REPO / "source").rglob("*.py")) + list((_REPO / "scripts").rglob("*.py")):
        if path.name in ("approach_zone.py", "test_approach_zone.py"):
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("APPROACH_ZONE_M") and "=" in stripped and "import" not in stripped:
                offenders.append(f"{path.relative_to(_REPO)}:{i}: {stripped}")
    assert not offenders, (
        "APPROACH_ZONE_M is assigned outside motionbricks/approach_zone.py:\n  "
        + "\n  ".join(offenders))


def test_it_loads_BY_PATH_for_the_motionbricks_conda_env():
    """Pass B runs in a separate conda env that cannot import the package, so it reads the
    generation contracts by path via ``mb_env.load_schema_module``. A module that only works as a
    package import would fail there and nowhere else -- and pass B is the GPU stage, so the failure
    would land after the expensive part had already started.
    """
    code = ("import sys; sys.path.insert(0, 'scripts/rsl_rl'); import mb_env; "
            "print(mb_env.load_schema_module('approach_zone').APPROACH_ZONE_M)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(_REPO),
                         env={"PYTHONPATH": str(_REPO / "source" / "motion_gen"),
                             "PATH": "/usr/bin:/bin", "HOME": str(pathlib.Path.home())})
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "2.0", out.stdout


def test_the_module_has_no_intra_package_imports():
    """Its siblings in ``motionbricks/`` are loaded by path, which gives them no package context, so
    a relative import would raise there while working fine under a normal import."""
    src = (_REPO / "source" / "motion_gen" / "motion_gen" / "motionbricks"
           / "approach_zone.py").read_text()
    for line in src.splitlines():
        s = line.strip()
        assert not s.startswith("from ."), f"relative import breaks the by-path load: {s}"
        assert not s.startswith("from motion_gen"), f"package import breaks the by-path load: {s}"
