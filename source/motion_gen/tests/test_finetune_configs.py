# Copyright 2026 Zachary Olkin. All rights reserved.
"""The four shipped fine-tune configs, as a set.

``finetune_config.py`` validates ONE config against the schema -- types, arity, ordering. This
checks the properties that only exist ACROSS the four, which is where they are load-bearing:
``make_finetune_terrain.py`` builds each library from its config, so a band edited in one file
and not the others silently makes the multi-terrain arm a different experiment from the
single-terrain arms it is supposed to be compared against.

The claim being pinned is the one ``multi.yaml`` makes in its own header: **config 4 is configs
1-3 plus the entry stairs.** That is only true if the geometry the three drawn types are drawn
from is IDENTICAL across the files, and nothing but a test can hold four YAML files to that.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "scripts" / "flow_matching"))

from finetune_config import config_keys, load_finetune_config  # noqa: E402

CONFIG_DIR = _ROOT / "configs" / "finetune"
SINGLES = ("boxes", "updown", "indist")
ALL = SINGLES + ("multi",)

#: The geometry every drawn tile is drawn from. Must be identical in all four files, or the
#: multi arm samples a different distribution from the arms it is compared against.
_SHARED_GEOMETRY = (
    "updown_steps_range", "updown_width_range_m", "updown_platform_range_m",
    "updown_wall_height_range_m", "boxes_per_tile_range", "box_height_range_m",
    "box_depth_range_m", "box_width_range_m", "box_gap_range_m",
    "commanded_speed_range_mps",
)


def _cfg(name: str):
    """One shipped config."""
    return load_finetune_config(str(CONFIG_DIR / f"{name}.yaml"))


@pytest.mark.parametrize("name", ALL)
def test_every_shipped_config_loads(name: str) -> None:
    """The loader is strict in both directions, so this also pins that none has drifted from
    the schema -- an unknown key is a typo that would do nothing, a missing one takes a
    default the file exists to state."""
    assert _cfg(name).name == name


@pytest.mark.parametrize("name", ALL)
def test_every_config_states_every_key(name: str) -> None:
    """No config may omit a decision, even an inert one."""
    import yaml

    with open(CONFIG_DIR / f"{name}.yaml") as fh:
        raw = yaml.safe_load(fh)
    assert sorted(raw) == config_keys()


@pytest.mark.parametrize("key", _SHARED_GEOMETRY)
def test_the_drawn_geometry_is_identical_across_all_four_configs(key: str) -> None:
    """Otherwise 'config 4 is configs 1-3 plus the entry stairs' is false in the files.

    An inert band still has to match: boxes.yaml declares stair ranges it never uses, and the
    moment someone tunes updown.yaml's and not boxes.yaml's, a later reader cannot tell which
    was the experiment.
    """
    values = {n: getattr(_cfg(n), key) for n in ALL}
    first = values[ALL[0]]
    assert all(v == first for v in values.values()), (
        f"{key} differs across the configs: {values}")


def test_the_stair_cells_match_between_updown_and_multi() -> None:
    """The two configs that actually BUILD up-into-down tiles must draw from one grid."""
    for key in ("updown_cells", "updown_measured_cells", "updown_measured_frac"):
        assert getattr(_cfg("updown"), key) == getattr(_cfg("multi"), key), key


@pytest.mark.parametrize("name,expect", [("boxes", "tiles_boxes"), ("updown", "tiles_updown"),
                                         ("indist", "tiles_indist")])
def test_each_single_terrain_config_builds_exactly_one_type(name: str, expect: str) -> None:
    """A single-terrain arm with a second type in it is not a single-terrain arm."""
    c = _cfg(name)
    counts = {k: getattr(c, k) for k in
              ("tiles_boxes", "tiles_updown", "tiles_indist", "tiles_entry_stairs")}
    assert counts[expect] > 0
    assert sum(v for k, v in counts.items() if k != expect) == 0, counts


def test_the_multi_config_carries_every_type() -> None:
    """Including both entry-stairs directions, which needs an even count of at least two."""
    c = _cfg("multi")
    for k in ("tiles_boxes", "tiles_updown", "tiles_indist", "tiles_entry_stairs"):
        assert getattr(c, k) > 0, f"multi is missing {k}"
    assert c.tiles_entry_stairs % 2 == 0, (
        "an odd entry-stairs count gives the two directions different tile counts, so the "
        "per-type breakdown compares them at different sample sizes")


def test_the_entry_stairs_get_enough_tiles_to_read_a_rate_off() -> None:
    """They are the type the config exists to test, and the tiles are a SAMPLING WEIGHT: all
    tiles of a direction are the same building, so the count buys env share and therefore
    episodes, not geometric variety."""
    c = _cfg("multi")
    per_direction = c.tiles_entry_stairs // 2
    assert per_direction >= 5, (
        f"{per_direction} tiles per direction is a thin slice to read a per-type success rate "
        "off; envs are round-robin over tiles, so this is the family's share of training time")


def test_only_the_multi_config_carries_ballast() -> None:
    """Ballast exists so AWR pulling toward one course cannot cost the pre-training
    distribution. A single-terrain arm that carried it would not be the same experiment."""
    assert _cfg("multi").ballast_chunks > 0
    assert 0.0 < _cfg("multi").ballast_flat_frac <= 1.0
    for n in SINGLES:
        assert _cfg(n).ballast_chunks == 0, f"{n} carries ballast"


def test_the_multi_buffer_scales_with_its_type_count() -> None:
    """Four types share one buffer, so the same churn would refresh a quarter as much of each
    type per iteration and the critic would be fitted to a policy several iterations stale."""
    multi, single = _cfg("multi"), _cfg(SINGLES[0])
    assert multi.buffer > single.buffer and multi.refresh > single.refresh
    assert multi.buffer / multi.refresh == pytest.approx(single.buffer / single.refresh), (
        "the buffer and the churn must scale together, or the multi arm holds its episodes "
        "for a different number of iterations than the single arms do")


@pytest.mark.parametrize("name", ALL)
def test_each_config_names_its_own_library_and_spec(name: str) -> None:
    """The pairing nothing else enforces: CLIP and GOALS were independent shell variables, so
    a half-updated command ran one course against another's regions and reported a rate."""
    c = _cfg(name)
    assert c.clip.endswith("_train.h5") and name in c.clip
    assert c.goals.endswith(".json") and name in c.goals


def test_no_two_configs_write_the_same_library() -> None:
    """A shared output path means the last build silently wins."""
    paths = [(_cfg(n).clip, _cfg(n).goals) for n in ALL]
    assert len({p[0] for p in paths}) == len(ALL)
    assert len({p[1] for p in paths}) == len(ALL)
