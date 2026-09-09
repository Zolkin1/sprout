# Copyright 2026 Zachary Olkin. All rights reserved.

"""The tile budget IS the training distribution, so it gets pinned.

``compose_terrain_library`` states the reason: "The RL importer distributes envs ROUND-ROBIN across
all tiles, so a family's share of training = (its tile count) / (total tiles)". A tile is also how
the RL and the FM see GEOMETRY -- each carries its own corridor width, platform widths, step count,
floor-or-void and box height -- which is independent of how many solved MOTION cells exist: 26 stair
tiles replaying 8 ``(riser, tread)`` cells still present 26 different staircases to the camera.

So the counts are a deliberate allocation and not an incidental total, and a change to one should
be a change someone made on purpose.
"""

from __future__ import annotations

import pathlib
import sys

import pytest


def _repo_root() -> pathlib.Path:
    for parent in pathlib.Path(__file__).resolve().parents:
        if (parent / "configs" / "clip_gen").is_dir() and (parent / "scripts" / "rsl_rl").is_dir():
            return parent
    raise RuntimeError(f"no checkout above {__file__}")


_REPO = _repo_root()
sys.path.insert(0, str(_REPO / "scripts" / "rsl_rl"))

from clip_gen_config import config_keys, load_clip_gen_config  # noqa: E402


def _cfg(name):
    return load_clip_gen_config(str(_REPO / "configs" / "clip_gen" / name))


def _tiles(c):
    return 2 * c.stair_tiles + 2 * c.pure_stair_tiles + sum(c.box_tiles) + c.flat_tiles


def test_the_full_recipe_is_200_tiles_and_10000_clips():
    """The scale the whole pipeline is costed against -- 21 min to build, and the number every
    per-family share is read against."""
    c = _cfg("g1_mb_v6.yaml")
    assert _tiles(c) == 200, (
        f"tile budget is {_tiles(c)}, not 200: transition {2 * c.stair_tiles}, "
        f"pure {2 * c.pure_stair_tiles}, box {c.box_tiles}, flat {c.flat_tiles}")
    assert _tiles(c) * c.clips_per_tile == 10_000


def test_the_terrain_families_got_the_distractor_tiles():
    """The reallocation, stated as the shares it produces rather than as the deltas.

    Two rounds of it. First the distractor family's 20 tiles went to terrain (before: 20+20
    transition, 15+15 pure, 20/15/15 box, 60 flat, 20 distractor). Then jump_off was cut to ZERO and
    pure stairs halved, and both refunds went to the four families that matter on hardware.
    """
    c = _cfg("g1_mb_v6.yaml")
    assert c.stair_tiles == 33, "transition stairs, per direction"
    assert list(c.box_tiles) == [32, 0, 26], "jump_on / jump_off / jump_off_front"
    assert c.pure_stair_tiles == 8, "halved -- a pure tile is one long staircase and carries less "\
                                    "geometric variety per tile than a transition tile"
    assert c.flat_tiles == 60, "deliberately unchanged"


def test_jump_off_is_cut_to_zero_on_purpose():
    """It jumps BACKWARDS off a box and the robot has no rear-facing camera.

    So the generator can never see what it is being asked to do there -- which is why it was already
    in ``DEFAULT_SKIP_FAMILIES`` for FM training. Cutting it to zero stops the TEACHER spending
    capacity on it too. Pinned as an explicit 0 rather than a missing entry, so it reads as a
    decision and not as a reallocation accident.
    """
    c = _cfg("g1_mb_v6.yaml")
    assert c.box_tiles[1] == 0


def test_the_distractor_family_is_unreachable():
    """No recipe key, and no driver stage.

    The visual-clutter pass now delivers "a box you see but do not touch" on EVERY tile instead of
    on 10% of them, at higher diversity and with no motion generation; and the family's clips were
    ``anchor_mask = 0`` (fully world-pinned), which was its own defect. The scripts stay on disk --
    ``mb_distractor_clips.py`` is the second consumer of ``flat_plan.plan_from_holds``, so deleting
    it would take that function's only non-flat test coverage.
    """
    assert not {"distractor_tiles", "distractor_s"} & set(config_keys())
    driver = (_REPO / "scripts" / "rsl_rl" / "generate_mb_library.py").read_text()
    for token in ("cfg.distractor_tiles", "cfg.distractor_s"):
        assert token not in driver, f"the driver still reads {token!r}"
    # Checked as an INVOCATION rather than a mention: the docstring names the script to explain why
    # it survives, and a bare substring test would flag its own explanation.
    assert '_run("distract' not in driver, "the driver still runs a distractor stage"
    assert '"mb_distractor_clips.py",' not in driver, "the driver still invokes mb_distractor_clips"
    # ...but the scripts are still there, on purpose.
    assert (_REPO / "scripts" / "rsl_rl" / "mb_distractor_clips.py").is_file()


@pytest.mark.parametrize("recipe", ["g1_mb_v6.yaml", "quick.yaml"])
def test_every_family_gets_at_least_one_tile(recipe):
    """A family at zero tiles is a family absent from training, which should be an explicit 0 in
    the recipe rather than something a reallocation did by accident."""
    c = _cfg(recipe)
    assert c.stair_tiles > 0 and c.pure_stair_tiles > 0 and c.flat_tiles > 0
    # jump_off (box_tiles[1]) is a DELIBERATE zero -- see test_jump_off_is_cut_to_zero_on_purpose.
    # Every other box family must still be present.
    assert c.box_tiles[0] > 0 and c.box_tiles[2] > 0, c.box_tiles
