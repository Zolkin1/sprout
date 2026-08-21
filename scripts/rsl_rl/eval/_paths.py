# Copyright 2026 Zachary Olkin. All rights reserved.

"""``sys.path`` bootstrap for the eval package.

Scripts under ``scripts/rsl_rl/`` import their siblings by bare module name
(``from provenance import git_snapshot``), which works because the running script's own
directory is ``sys.path[0]``.  This package sits one level deeper, so that directory is
``scripts/rsl_rl/eval`` and every one of those imports would fail.  Importing this module
first puts both directories on the path, which is what keeps ``provenance``, ``cli_args``,
``play_plots``, ``recorded_data_io``, ``profiling`` and ``plot_style`` reachable without
moving or editing any of them.

Precedents for the same manoeuvre: ``scripts/flow_matching/fm_augment_probe.py`` inserts
``scripts/rsl_rl``, and ``scripts/rsl_rl/command_distribution_stats.py`` inserts its own
directory before importing a sibling.

Import for effect, before any sibling import::

    import _paths  # noqa: F401
"""

from __future__ import annotations

import pathlib
import sys

# <repo>/scripts/rsl_rl/eval/_paths.py -> parents[0] == eval, [1] == rsl_rl, [3] == <repo>.
# Counted from THIS file rather than assembled from ``os.pardir`` hops, and then verified
# against markers below, so moving this module cannot silently yield a plausible-but-wrong
# root (which has happened in this repo before -- see motion_gen.features._REPO_ROOT).
EVAL_DIR: pathlib.Path = pathlib.Path(__file__).resolve().parent
SCRIPTS_RSL_RL_DIR: pathlib.Path = EVAL_DIR.parent
REPO_ROOT: pathlib.Path = EVAL_DIR.parents[2]

_MARKERS = ("source/robot_rl", "source/motion_gen", "scripts/rsl_rl")


def _verify_root() -> None:
    """Fail loudly if the derived repo root is not this repo.

    A wrong root is far worse than a missing one: every path built from it would point
    somewhere that plausibly exists, so the failure would surface much later as a stale
    library or a missing module rather than here.

    Raises:
        RuntimeError: If any expected marker directory is absent.
    """
    missing = [m for m in _MARKERS if not (REPO_ROOT / m).is_dir()]
    if missing:
        raise RuntimeError(
            f"{__file__} derived REPO_ROOT={REPO_ROOT}, which is missing {missing}. "
            "This module must live at <repo>/scripts/rsl_rl/eval/_paths.py; if it moved, "
            "fix the parents[] index above rather than adding another '..' hop."
        )


def bootstrap() -> None:
    """Prepend the eval dir and ``scripts/rsl_rl`` to ``sys.path`` (idempotent)."""
    _verify_root()
    for d in (SCRIPTS_RSL_RL_DIR, EVAL_DIR):  # eval ends up first
        s = str(d)
        if s in sys.path:
            sys.path.remove(s)
        sys.path.insert(0, s)


bootstrap()
