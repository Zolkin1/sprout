# Copyright 2026 Zachary Olkin. All rights reserved.

"""Pin the distractor family's anchor mask to FLAT (ego), in BOTH generators.

Distractors must be ego-anchored like flat walking. A world-pinned distractor is not merely
useless to the flow-matching generator, it is actively harmful: FM training windows are cut at
re-anchor (snap) events, and ``ego_anchor_fraction`` is 1.0 when recording, so the snap DOES
fire for a pinned clip -- but its placement cannot move, so the recorded target is accumulated
robot-vs-clip tracking drift instead of a placement decision. Nothing downstream can tell those
windows apart from good ones.

These tests exist so that reverting the mask (or adding a third distractor generator that
forgets) fails here rather than silently poisoning a recording several hours later.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

from motion_gen.world_lift import _ANCHOR_MASK_FLAT, anchor_mask_free_names

_SCRIPTS = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "rsl_rl"

# Every module that writes clips for the distractor family, and the constant it must use.
_GENERATORS = ("generate_distractor_clips", "mb_distractor_clips")


def _load(name: str):
    """Import a ``scripts/rsl_rl`` module by path (they are scripts, not an installed package)."""
    path = _SCRIPTS / f"{name}.py"
    if not path.exists():                                    # pragma: no cover - layout guard
        pytest.skip(f"{path} not present")
    # These scripts import each other as siblings, so the directory must be importable. Add it
    # only if absent and NEVER remove it: other test modules in this suite put it there too, and
    # popping a shared entry breaks whichever module imports after us.
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name", _GENERATORS)
def test_distractor_mask_is_flat(name: str) -> None:
    """Both distractor generators must emit ``_ANCHOR_MASK_FLAT``."""
    mod = _load(name)
    assert hasattr(mod, "_DISTRACTOR_MASK"), (
        f"{name} has no _DISTRACTOR_MASK; if the constant was renamed, update this test so the "
        f"mask stays pinned rather than dropping the check"
    )
    assert int(mod._DISTRACTOR_MASK) == int(_ANCHOR_MASK_FLAT), (
        f"{name}._DISTRACTOR_MASK is {int(mod._DISTRACTOR_MASK)}, expected "
        f"{int(_ANCHOR_MASK_FLAT)} (FLAT). Distractors must be ego-anchored -- see this module's "
        f"docstring for why a pinned distractor poisons FM recording."
    )


@pytest.mark.parametrize("name", _GENERATORS)
def test_distractor_mask_frees_some_root_dof(name: str) -> None:
    """The mask must free at least one root DOF -- that is what makes a clip re-anchorable."""
    mod = _load(name)
    free = anchor_mask_free_names(int(mod._DISTRACTOR_MASK))
    assert free, f"{name}._DISTRACTOR_MASK frees NO root DOF: it can never be a re-anchor event"
    assert {"downrange", "lateral", "yaw"} <= set(free), f"{name} freed only {sorted(free)}"


def test_both_generators_agree() -> None:
    """The two sources must agree, so a mixed library has one distractor anchoring rule."""
    masks = {n: int(_load(n)._DISTRACTOR_MASK) for n in _GENERATORS}
    assert len(set(masks.values())) == 1, f"distractor generators disagree: {masks}"
