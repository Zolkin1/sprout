# Copyright 2026 Zachary Olkin. All rights reserved.

"""The shipped clip-gen recipes must load, and the loader must refuse to guess.

The point of the config is that a library's distribution is stated, not defaulted -- so the
STRICTNESS is the feature under test here, not just the happy path. A silently-defaulted parameter
produces a library that looks correct and trains on the wrong thing.
"""
from __future__ import annotations

import os
import pathlib
import sys

import pytest
import yaml

def _repo_root() -> pathlib.Path:
    """Walk up from this file to the checkout holding both assets this test needs.

    Not a fixed number of ``..`` hops (that breaks silently when a file moves -- see CLAUDE.md) and
    NOT derived from ``motion_gen.__file__`` either: the editable install points at the MAIN
    checkout, so in a git worktree that would resolve to the wrong tree and quietly test the other
    copy's configs. Searching for the assets is self-validating -- it either finds a tree with both
    or fails.
    """
    for parent in pathlib.Path(__file__).resolve().parents:
        if (parent / "configs" / "clip_gen").is_dir() and (parent / "scripts" / "rsl_rl").is_dir():
            return parent
    raise RuntimeError(f"no checkout with configs/clip_gen + scripts/rsl_rl above {__file__}")


_REPO = str(_repo_root())
sys.path.insert(0, os.path.join(_REPO, "scripts", "rsl_rl"))

from clip_gen_config import config_keys, load_clip_gen_config  # noqa: E402

_CONFIG_DIR = os.path.join(_REPO, "configs", "clip_gen")
_SHIPPED = ["g1_mb_v6.yaml", "quick.yaml"]


@pytest.mark.parametrize("name", _SHIPPED)
def test_shipped_config_loads(name):
    """Every recipe in the repo validates against the schema."""
    cfg = load_clip_gen_config(os.path.join(_CONFIG_DIR, name))
    for key in config_keys():
        assert hasattr(cfg, key), f"{name} produced no {key}"


def test_quick_differs_from_full_only_in_scale():
    """quick.yaml is a SCALE variant: every distribution parameter must match the full recipe.

    It is a standalone file precisely so it cannot silently miss a knob the way the old --quick flag
    did, but that only helps if the two stay equal everywhere scale is not involved -- otherwise a
    smoke run measures a distribution the real run will not produce.
    """
    full = load_clip_gen_config(os.path.join(_CONFIG_DIR, "g1_mb_v6.yaml"))
    quick = load_clip_gen_config(os.path.join(_CONFIG_DIR, "quick.yaml"))
    scale = {"clips_per_tile", "stair_tiles", "pure_stair_tiles", "box_tiles", "flat_tiles",
             "distractor_tiles"}
    for key in config_keys():
        if key in scale:
            continue
        assert getattr(full, key) == getattr(quick, key), (
            f"{key} differs between the full and quick recipes ({getattr(full, key)!r} vs "
            f"{getattr(quick, key)!r}); only scale keys {sorted(scale)} may differ")


def test_unknown_key_is_rejected(tmp_path):
    """A typo must fail loudly rather than fall back to a default."""
    good = yaml.safe_load(open(os.path.join(_CONFIG_DIR, "quick.yaml")))
    good["stair_vel_rnge"] = [0.2, 2.6]                       # typo
    path = tmp_path / "typo.yaml"
    path.write_text(yaml.safe_dump(good))
    with pytest.raises(SystemExit, match="stair_vel_rnge"):
        load_clip_gen_config(str(path))


def test_missing_key_is_rejected(tmp_path):
    """An omitted parameter must fail loudly -- the whole recipe has to be explicit."""
    good = yaml.safe_load(open(os.path.join(_CONFIG_DIR, "quick.yaml")))
    del good["stair_vel_range"]
    path = tmp_path / "missing.yaml"
    path.write_text(yaml.safe_dump(good))
    with pytest.raises(SystemExit, match="stair_vel_range"):
        load_clip_gen_config(str(path))


def test_transposed_range_is_rejected(tmp_path):
    """``[max, min]`` is the easy mistake in a hand-edited band, and it silently empties the range."""
    good = yaml.safe_load(open(os.path.join(_CONFIG_DIR, "quick.yaml")))
    good["stair_vel_range"] = [2.6, 0.2]
    path = tmp_path / "flipped.yaml"
    path.write_text(yaml.safe_dump(good))
    with pytest.raises(SystemExit, match="transposed"):
        load_clip_gen_config(str(path))


@pytest.mark.parametrize("key,bad", [
    ("clips_per_tile", 4.5),                 # float where an int is required
    ("rl_cmd_vary", "yes"),                  # string where a bool is required
    ("box_tiles", [20, 15]),                 # wrong arity
    ("rl_vels", "interpolated"),             # not a valid choice
    ("stair_vel_range", [0.2, 2.6, 3.0]),    # not a pair
])
def test_bad_value_is_rejected(tmp_path, key, bad):
    """Shallow type / arity checks catch the mistakes a hand-edited recipe actually makes."""
    good = yaml.safe_load(open(os.path.join(_CONFIG_DIR, "quick.yaml")))
    good[key] = bad
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(good))
    with pytest.raises(SystemExit, match=key):
        load_clip_gen_config(str(path))
