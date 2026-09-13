# Copyright 2026 Zachary Olkin. All rights reserved.

"""``ballast`` must be able to read a pre-training recording under its REAL caller's sys.path.

``build_ballast_dataset`` defers ``from recorded_data_io import load_recorded_windows`` to the
bottom of the function. That module lives in ``scripts/rsl_rl``; ``ballast`` lives in
``scripts/flow_matching``; and its only importer, ``finetune_generator.py``, puts
``source/robot_rl`` and ``source/motion_gen`` on the path and nothing else. So the import
resolved only when something else had already added ``scripts/rsl_rl`` -- which is not the
case in the fine-tune loop.

**Why it survived a smoke test.** The ballast path has exactly one caller: of the four
fine-tune configs only ``multi`` sets ``ballast_chunks > 0``. The other three never reach the
import at all, so three clean runs say nothing about it. And because it is deferred to the END
of stage 8 of 8, ``multi`` collected 4000 episodes, graded them, built a 113k-window buffer,
fitted both critic rounds and computed the AWR weights -- 67 minutes of an H100 -- before
dying on ``ModuleNotFoundError: No module named 'recorded_data_io'``.

The test runs in a SUBPROCESS with a reconstructed path, because once any test in this session
has imported a module from ``scripts/rsl_rl`` the failure cannot be reproduced in-process.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_FM = _ROOT / "scripts" / "flow_matching"

#: Exactly what ``finetune_generator.py`` puts on ``sys.path`` before importing ``ballast``:
#: its own directory (implicit, since it is the script being run) plus the two source trees.
#: ``scripts/rsl_rl`` is deliberately ABSENT -- reproducing it is the point.
_CALLER_PATH: tuple[str, ...] = (
    str(_FM), str(_ROOT / "source" / "robot_rl"), str(_ROOT / "source" / "motion_gen"),
)

_PROBE = """
import sys
sys.path = {path!r} + [p for p in sys.path if p.endswith(('site-packages', 'lib-dynload'))
                       or 'python3' in p]
import ballast                      # must not need anything else on the path
import recorded_data_io             # the deferred import build_ballast_dataset performs
assert hasattr(recorded_data_io, 'load_recorded_windows')
print('OK')
"""


def test_importing_ballast_makes_its_deferred_dependency_resolvable() -> None:
    """The regression: ``ballast`` owns the dependency, so it must own the path entry."""
    code = _PROBE.format(path=list(_CALLER_PATH))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=str(_ROOT), timeout=120)
    assert r.returncode == 0 and "OK" in r.stdout, (
        "ballast cannot reach recorded_data_io under its real caller's sys.path. That is a "
        "ModuleNotFoundError at the END of stage 8 of 8, after the whole iteration has been "
        f"computed.\n--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")


def test_ballast_declares_the_path_itself_rather_than_relying_on_a_caller() -> None:
    """Pinned as a property of the module, so a tidy-up cannot move it back to the caller.

    Putting it in ``finetune_generator`` would work today and break for the next importer --
    and the next importer would find out the same expensive way.
    """
    sys.path.insert(0, str(_FM))
    ballast = pytest.importorskip("ballast")
    assert hasattr(ballast, "_RSL_RL"), (
        "ballast no longer declares the scripts/rsl_rl path entry it needs")
    assert pathlib.Path(ballast._RSL_RL).name == "rsl_rl"
    assert (pathlib.Path(ballast._RSL_RL) / "recorded_data_io.py").exists(), (
        f"ballast._RSL_RL points at {ballast._RSL_RL}, which holds no recorded_data_io.py")
