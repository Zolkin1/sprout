# Copyright 2026 Zachary Olkin. All rights reserved.
"""The four-type multi-terrain library: drawn boxes, drawn up-into-down, the measured entrance
in both directions, and copied in-distribution tiles.

What this file is aimed at is the JOIN. The four types are each tested where they are built;
what is new here is putting them in one manifest, and every failure mode of that is a
renumbering or a bookkeeping one -- a duplicate id orphans a tile, a non-contiguous family
block makes the per-type report pool two families, an unbalanced in-distribution draw moves the
denominators between libraries, and a shallow copy lets the multi library mutate the source
manifest it was built from.

The per-type breakdown is the whole point of the config: a pooled success rate over a mix moves
when the MIX moves, with no policy change at all, so the id blocks have to be right or the
multi-terrain arm cannot be compared against the single-terrain ones.
"""

from __future__ import annotations

import copy

import pytest

from motion_gen.terrain.entry_stairs import (
    ENTRY_STAIRS_DOWN_FAMILY,
    ENTRY_STAIRS_UP_FAMILY,
)
from motion_gen.terrain.multi_courses import (
    FAMILY_ORDER,
    MultiCounts,
    family_blocks,
    generate_multi_manifest,
    multi_generator_config,
    multi_goal_region_spec,
    select_indist_tiles,
    tile_key,
    validate_multi_manifest,
)
from motion_gen.terrain.varied_courses import (
    BOXES_FAMILY,
    GOAL_HALF_WIDTH_M,
    UPDOWN_FAMILY,
    VariedCoursesCfg,
)

#: The shipped in-distribution library's own shape: four families, unevenly filled.
_INDIST_SHAPE = {"family_jump_off_front.h5": 15, "family_jump_on.h5": 20,
                 "family_stairs_down.h5": 20, "family_stairs_up.h5": 20}


def _indist_source() -> list[dict]:
    """A stand-in in-distribution manifest with the shipped one's family shape."""
    tiles, tid = [], 0
    for fam, n in _INDIST_SHAPE.items():
        for _ in range(n):
            tiles.append({
                "terrain_id": tid, "type": "box", "family": fam, "has_floor": True,
                "size_x": 12.0, "size_y": 6.0, "frame_yaw": 0.0,
                "source_terrain_id": tid,
                "overlays": [{"kind": "box", "pos_xy": [6.0, 3.0],
                              "size_xyz": [12.0, 6.0, 0.1], "base_z": 0.0,
                              "walkable_top": True, "yaw": 0.0}],
                "route": {"start_x": 1.0, "start_y": 3.0, "start_z": 0.0, "start_yaw": 0.0,
                          "goal_x": 9.0, "goal_y": 3.0, "goal_half_width": 0.75,
                          "goal_z": 0.0},
            })
            tid += 1
    return tiles


def _manifest(counts: MultiCounts | None = None, seed: int = 0, source=None) -> list[dict]:
    """A built manifest at one seed."""
    counts = counts or MultiCounts(boxes=4, updown=4, entry_stairs=6, indist=8)
    return generate_multi_manifest(counts, VariedCoursesCfg(seed=seed),
                                   _indist_source() if source is None else source)


# ---------------------------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------------------------
def test_a_negative_count_is_refused() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        MultiCounts(boxes=-1)


def test_an_empty_library_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one tile"):
        MultiCounts()


@pytest.mark.parametrize("n,up,down", [(0, 0, 0), (1, 1, 0), (2, 1, 1), (6, 3, 3), (7, 4, 3)])
def test_the_entry_stairs_split_between_the_two_directions(n, up, down) -> None:
    """An odd count gives the extra tile to the ascending direction, and both halves sum."""
    c = MultiCounts(boxes=1, entry_stairs=n)
    assert (c.entry_up, c.entry_down) == (up, down)
    assert c.entry_up + c.entry_down == n


def test_both_directions_actually_appear_in_the_manifest() -> None:
    """The thing the split exists for -- a library that silently built only one direction
    would still have the right tile COUNT."""
    man = _manifest(MultiCounts(boxes=1, entry_stairs=4, indist=0), source=[])
    fams = [t["family"] for t in man]
    assert fams.count(ENTRY_STAIRS_UP_FAMILY) == 2
    assert fams.count(ENTRY_STAIRS_DOWN_FAMILY) == 2


# ---------------------------------------------------------------------------------------------
# The join: ids and family blocks
# ---------------------------------------------------------------------------------------------
def test_the_ids_are_contiguous_from_zero() -> None:
    man = _manifest()
    assert [t["terrain_id"] for t in man] == list(range(len(man)))


def test_every_family_occupies_one_contiguous_block() -> None:
    """The reduction key for the per-type report. A gap pools two families silently."""
    man = _manifest()
    blocks = family_blocks(man)
    for fam, (lo, hi) in blocks.items():
        ids = sorted(t["terrain_id"] for t in man if t["family"] == fam)
        assert ids == list(range(lo, hi + 1))
    spans = sorted(blocks.values())
    for (_, hi), (lo, _) in zip(spans, spans[1:]):
        assert lo == hi + 1, "the family blocks do not tile the id range"


def test_the_drawn_families_come_first_in_the_declared_order() -> None:
    """So a manifest and FAMILY_ORDER cannot disagree about the layout."""
    man = _manifest()
    blocks = family_blocks(man)
    present = [f for f in FAMILY_ORDER if f in blocks]
    assert [blocks[f][0] for f in present] == sorted(blocks[f][0] for f in present)
    assert min(blocks[f][0] for f in present) == 0


def test_a_non_contiguous_family_is_refused() -> None:
    """Because an id-RANGE reduction would pool it with whatever sits in the gap."""
    man = _manifest()
    a = next(i for i, t in enumerate(man) if t["family"] == BOXES_FAMILY)
    b = next(i for i, t in enumerate(man) if t["family"] == UPDOWN_FAMILY)
    man[a]["family"], man[b]["family"] = man[b]["family"], man[a]["family"]
    with pytest.raises(ValueError, match="non-contiguous"):
        family_blocks(man)


def test_a_duplicate_id_is_refused() -> None:
    """What renumbering the copied tiles gets wrong; one tile is built and the rest orphaned."""
    man = _manifest()
    man[-1]["terrain_id"] = man[0]["terrain_id"]
    with pytest.raises(ValueError, match="duplicate terrain_id"):
        validate_multi_manifest(man)


def test_the_manifest_validates() -> None:
    validate_multi_manifest(_manifest())


def test_the_freeform_validator_still_runs_on_the_authored_tiles() -> None:
    """The copied tiles are exempt from it by SCHEMA, not the authored ones -- so a real fault
    in a drawn tile must still be caught rather than waved through with them."""
    man = _manifest()
    box = next(t for t in man if t["family"] == BOXES_FAMILY)
    box["overlays"][0]["yaw"] = 0.5          # built unrotated in Isaac, rotated in MuJoCo
    with pytest.raises(ValueError):
        validate_multi_manifest(man)


# ---------------------------------------------------------------------------------------------
# The copied in-distribution tiles
# ---------------------------------------------------------------------------------------------
def test_the_indist_sample_is_balanced_across_its_families() -> None:
    """The source is 15/20/20/20, so a uniform draw would return a different mix every seed
    and make the per-type denominators wobble between libraries."""
    picked = select_indist_tiles(_indist_source(), 8, seed=0)
    counts = {f: sum(1 for t in picked if t["family"] == f) for f in _INDIST_SHAPE}
    assert set(counts.values()) == {2}, counts


def test_an_uneven_count_spreads_the_remainder_rather_than_starving_a_family() -> None:
    picked = select_indist_tiles(_indist_source(), 6, seed=0)
    counts = sorted(sum(1 for t in picked if t["family"] == f) for f in _INDIST_SHAPE)
    assert counts == [1, 1, 2, 2], counts


def test_the_sample_is_reproducible_and_the_seed_moves_it() -> None:
    src = _indist_source()
    a = [t["terrain_id"] for t in select_indist_tiles(src, 12, seed=0)]
    assert a == [t["terrain_id"] for t in select_indist_tiles(src, 12, seed=0)]
    assert a != [t["terrain_id"] for t in select_indist_tiles(src, 12, seed=7)]


def test_the_copied_tiles_are_deep_copies() -> None:
    """A shallow copy lets the multi library mutate the manifest it was built from -- and the
    renumbering does exactly that kind of write."""
    src = _indist_source()
    before = copy.deepcopy(src)
    man = generate_multi_manifest(MultiCounts(boxes=2, indist=8), VariedCoursesCfg(seed=0), src)
    assert src == before, "building the multi manifest mutated the source tiles"
    assert max(t["terrain_id"] for t in man) == len(man) - 1


def test_the_copied_tiles_keep_their_route_and_provenance() -> None:
    """They are the control arm: the same question on the same ground, so nothing about them
    may be re-derived."""
    man = _manifest()
    copied = [t for t in man if t["family"] not in FAMILY_ORDER]
    assert copied
    for t in copied:
        assert t["route"]["goal_half_width"] == pytest.approx(0.75)
        assert "source_terrain_id" in t, "the tile's original id is its provenance"


def test_asking_for_indist_tiles_without_supplying_any_is_refused() -> None:
    """Rather than quietly building a library that is missing its control arm."""
    with pytest.raises(ValueError, match="no in-distribution tiles"):
        generate_multi_manifest(MultiCounts(boxes=2, indist=4), VariedCoursesCfg(), [])


def test_asking_for_more_indist_tiles_than_exist_is_refused() -> None:
    with pytest.raises(ValueError, match="only .* are available"):
        select_indist_tiles(_indist_source(), 1000)


def test_a_holdout_excluding_the_training_picks_is_DISJOINT() -> None:
    """Without this the two draws are independent samples of one pool, and overlap badly.

    Measured on the shipped library at 60 of 75 tiles: the "holdout" shared 80% of its tiles
    with training, and jump_off_front -- 15 tiles, all of them taken by both -- shared 100%.
    A file named holdout that reports mostly training tiles is worse than no holdout.
    """
    src = _indist_source()
    train = select_indist_tiles(src, 20, seed=0)
    spent = {tile_key(t) for t in train}
    held = select_indist_tiles(src, 20, seed=0, exclude=spent)
    assert not (spent & {tile_key(t) for t in held}), "the holdout reuses training tiles"
    assert len(held) == 20


def test_a_holdout_that_cannot_be_disjoint_is_REFUSED_not_silently_overlapped() -> None:
    """The in-distribution config asks for 60 of 75 real tiles, which leaves 15 -- so it has
    to be told, rather than handed a file that measures what it was fitted on."""
    src = _indist_source()                       # 75 tiles
    spent = {tile_key(t) for t in select_indist_tiles(src, 60, seed=0)}
    with pytest.raises(ValueError, match="already spent on the training library"):
        select_indist_tiles(src, 60, seed=0, exclude=spent)


def test_the_exclusion_keys_off_the_SOURCE_id_not_the_renumbered_one() -> None:
    """terrain_id is renumbered on the way into a multi manifest, so it cannot identify a tile
    across libraries -- keying on it would make every exclusion a no-op or a wrong match."""
    src = _indist_source()
    man = generate_multi_manifest(MultiCounts(boxes=2, indist=8), VariedCoursesCfg(seed=0), src)
    copied = [t for t in man if t["family"] not in FAMILY_ORDER]
    assert any(t["terrain_id"] != t["source_terrain_id"] for t in copied), (
        "nothing was renumbered, so this test proves nothing")
    held = select_indist_tiles(src, 8, seed=0, exclude={tile_key(t) for t in copied})
    assert not ({tile_key(t) for t in copied} & {tile_key(t) for t in held})


def test_an_indist_tile_without_a_route_is_refused() -> None:
    """A route is what make_indist_routes.py adds; without it the tile has no goal at all."""
    src = _indist_source()
    del src[3]["route"]
    with pytest.raises(ValueError, match="carry no route"):
        select_indist_tiles(src, 8)


# ---------------------------------------------------------------------------------------------
# The declared spec and config
# ---------------------------------------------------------------------------------------------
def test_the_goal_spec_is_the_librarys_one_band() -> None:
    """Every tile here is BUILT to GOAL_HALF_WIDTH_M, which is what lets one number serve."""
    counts = MultiCounts(boxes=4, updown=4, entry_stairs=6, indist=8)
    spec = multi_goal_region_spec(counts)
    assert spec["default"]["half_width_m"] == pytest.approx(GOAL_HALF_WIDTH_M)
    man = _manifest(counts)
    for t in man:
        assert t["route"]["goal_half_width"] == pytest.approx(GOAL_HALF_WIDTH_M)


def test_the_goal_spec_never_declares_a_zero_height_tolerance() -> None:
    """0 is an exact-equality test in `in_goal`, not a disabled one, and it voided a whole arm.

    It matters more here than on the drawn courses: this library has tiles whose goal is a
    storey above the spawn, so the band is what separates climbing the flight from walking
    past it on the floor beside it.
    """
    assert multi_goal_region_spec(MultiCounts(boxes=1))["default"]["height_tol_m"] > 0.0


def test_the_spec_and_config_record_what_was_built() -> None:
    counts = MultiCounts(boxes=4, updown=4, entry_stairs=6, indist=8)
    spec = multi_goal_region_spec(counts)
    assert spec["provenance"]["counts"]["entry_stairs_up"] == 3
    assert spec["provenance"]["counts"]["entry_stairs_down"] == 3
    cfg = multi_generator_config(counts, VariedCoursesCfg(seed=3), indist_source="clips/x.h5")
    assert cfg["seed"] == 3
    assert cfg["multi_courses"]["indist_source"] == "clips/x.h5"
    assert cfg["freeform"]["commanded_speed_range_mps"][0] > 0.0


def test_the_command_band_is_where_the_reader_looks() -> None:
    """`clip_io.read_commanded_speed_range` reads `freeform.commanded_speed_range_mps`; box2
    and the curb shipped a ladder under a different key for weeks and nothing read it."""
    cfg = multi_generator_config(MultiCounts(boxes=1))
    assert "commanded_speed_range_mps" in cfg["freeform"]


def test_a_tile_without_a_source_id_is_refused_rather_than_keyed_on_its_renumbered_one() -> None:
    """The fallback that would have made the holdout silently overlap again.

    `terrain_id` is renumbered on the way into a multi manifest, so keying on it compares a
    renumbered id against a source id: the exclusion set matches nothing, the "already spent"
    refusal never fires, and the holdout quietly reuses the training tiles -- the exact
    failure the exclusion exists to prevent. Same lesson as `goal_z`'s defensive `.get`.
    """
    src = _indist_source()
    del src[0]["source_terrain_id"]
    with pytest.raises(ValueError, match="no source_terrain_id"):
        tile_key(src[0])


def test_the_multi_spec_does_not_inherit_the_varied_librarys_family_block() -> None:
    """It described a library that does not exist here, which is worse than no record."""
    spec = multi_goal_region_spec(MultiCounts(boxes=2, entry_stairs=2))
    assert "families" not in spec["provenance"]
    assert spec["provenance"]["source"].endswith("multi_courses")
