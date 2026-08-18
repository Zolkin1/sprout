# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the dataset composer's selection maths.

The composer's whole claim is that a quota comes out spread evenly over clips and evenly
along each clip -- so these check the two spreads directly, plus the degenerate cases that
would silently skew a dataset instead of failing: a bucket with less supply than its quota,
clips of wildly different lengths, and a quota below the clip count.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_rl.flow_matching.compose import (
    allocate_even,
    bucket_supply,
    pick_evenly_spaced,
    join_buckets,
    select_windows,
    window_buckets,
    window_buckets_by_terrain_span,
    window_buckets_by_type,
)


# --- allocate_even ---------------------------------------------------------------------


def test_allocate_even_splits_equally_when_supply_allows():
    take = allocate_even(np.array([100, 100, 100, 100]), 40)
    assert take.tolist() == [10, 10, 10, 10]


def test_allocate_even_redistributes_a_short_item():
    # Item 0 can only give 2; the other three absorb the remaining 28 evenly.
    take = allocate_even(np.array([2, 100, 100, 100]), 30)
    assert take[0] == 2
    assert take[1:].tolist() == [10, 10, 10] or sorted(take[1:].tolist()) == [9, 9, 10]
    assert int(take.sum()) == 30


def test_allocate_even_clamps_to_total_supply():
    take = allocate_even(np.array([3, 4, 5]), 1000)
    assert take.tolist() == [3, 4, 5]


def test_allocate_even_remainder_goes_to_the_roomiest_items():
    # 5 picks over 4 items: one each, then the extra to the item with the most room.
    take = allocate_even(np.array([10, 1, 1, 1]), 5)
    assert int(take.sum()) == 5
    assert take[0] == 2  # the only item with room left after one apiece


def test_allocate_even_below_item_count_spreads_one_each():
    take = allocate_even(np.array([50, 50, 50, 50, 50]), 3)
    assert int(take.sum()) == 3
    assert set(take.tolist()) == {0, 1}


def test_allocate_even_rejects_negatives():
    with pytest.raises(ValueError):
        allocate_even(np.array([1, 2]), -1)
    with pytest.raises(ValueError):
        allocate_even(np.array([-1, 2]), 1)


# --- pick_evenly_spaced ----------------------------------------------------------------


def test_pick_evenly_spaced_is_strictly_increasing_and_in_range():
    idx = pick_evenly_spaced(100, 7)
    assert idx.size == 7
    assert np.all(np.diff(idx) > 0)
    assert idx.min() >= 0 and idx.max() < 100


def test_pick_evenly_spaced_takes_everything_when_asked_for_more():
    assert pick_evenly_spaced(5, 9).tolist() == [0, 1, 2, 3, 4]


def test_pick_evenly_spaced_avoids_the_endpoints_for_a_single_pick():
    # One pick lands mid-clip, not on frame 0 -- the whole point of bin sampling.
    assert pick_evenly_spaced(100, 1).tolist() == [50]


def test_pick_evenly_spaced_offset_shifts_within_the_bin():
    assert pick_evenly_spaced(100, 1, offset=0.0).tolist() == [0]
    assert pick_evenly_spaced(100, 1, offset=0.9).tolist() == [90]
    assert np.all(np.diff(pick_evenly_spaced(100, 7, offset=0.1)) > 0)


def test_pick_evenly_spaced_rejects_an_out_of_range_offset():
    with pytest.raises(ValueError, match="offset"):
        pick_evenly_spaced(10, 2, offset=1.0)


def test_pick_evenly_spaced_is_uniform():
    idx = pick_evenly_spaced(1000, 10)
    gaps = np.diff(idx)
    assert gaps.min() >= 99 and gaps.max() <= 101


# --- select_windows --------------------------------------------------------------------


def _synthetic(n_clips_per_bucket=(("flat", 4, 200), ("stairs", 6, 50))):
    """Build (clip_id, clip_step, bucket) for clips of a stated per-bucket length."""
    clip_id, clip_step, bucket = [], [], []
    next_clip = 0
    for name, n_clips, n_steps in n_clips_per_bucket:
        for _ in range(n_clips):
            clip_id.append(np.full(n_steps, next_clip))
            clip_step.append(np.arange(n_steps) * 12)
            bucket.append(np.full(n_steps, name, dtype=object))
            next_clip += 1
    return (np.concatenate(clip_id), np.concatenate(clip_step),
            np.concatenate(bucket).astype(object))


def test_select_windows_meets_each_quota():
    cid, cstep, buck = _synthetic()
    rows, report = select_windows(cid, cstep, buck, {"flat": 100, "stairs": 120})
    assert report["flat"]["taken"] == 100
    assert report["stairs"]["taken"] == 120
    assert rows.size == 220
    assert np.all(np.diff(rows) > 0)  # strictly increasing, as HDF5 requires


def test_select_windows_spreads_over_every_clip_in_the_bucket():
    cid, cstep, buck = _synthetic()
    rows, report = select_windows(cid, cstep, buck, {"stairs": 120})
    _, counts = np.unique(cid[rows], return_counts=True)
    assert report["stairs"]["clips_used"] == 6
    assert counts.tolist() == [20] * 6  # 120 over 6 clips, exactly even


def test_select_windows_is_uniform_along_each_clip():
    cid, cstep, buck = _synthetic((("flat", 1, 600),))
    rows, _ = select_windows(cid, cstep, buck, {"flat": 60})
    steps = np.sort(cstep[rows])
    gaps = np.diff(steps)
    # 60 picks over 600 windows spaced 12 apart -> one every ~120 steps, no clustering.
    assert gaps.min() >= 108 and gaps.max() <= 132


def test_select_windows_one_per_clip_still_spans_the_phase_range():
    # The failure this guards: a quota below the bucket's clip count takes exactly one
    # window per clip, and with a fixed bin offset every one of them is that clip's
    # MIDPOINT -- a dataset even across clips sitting entirely at phase 0.5.
    cid, cstep, buck = _synthetic((("stairs", 200, 300),))
    rows, report = select_windows(cid, cstep, buck, {"stairs": 200})
    assert report["stairs"]["clips_used"] == 200
    phase = cstep[rows] / cstep.max()
    counts, _ = np.histogram(phase, bins=10, range=(0.0, 1.0))
    assert counts.min() >= 10, f"phase deciles not covered: {counts.tolist()}"


def test_select_windows_ignores_buckets_with_no_quota():
    cid, cstep, buck = _synthetic()
    rows, report = select_windows(cid, cstep, buck, {"stairs": 60})
    assert set(buck[rows]) == {"stairs"}
    assert "flat" not in report


def test_select_windows_reports_a_short_bucket_instead_of_overdrawing():
    cid, cstep, buck = _synthetic((("stairs", 2, 10),))
    rows, report = select_windows(cid, cstep, buck, {"stairs": 500})
    assert report["stairs"]["taken"] == 20  # everything there is
    assert report["stairs"]["taken"] < report["stairs"]["requested"]
    assert rows.size == 20


def test_select_windows_rejects_an_unknown_bucket():
    cid, cstep, buck = _synthetic()
    with pytest.raises(ValueError, match="no windows for bucket"):
        select_windows(cid, cstep, buck, {"lava": 10})


def test_select_windows_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        select_windows(np.zeros(5), np.zeros(4), np.array(["a"] * 5, dtype=object), {"a": 1})


def test_select_windows_never_repeats_a_row():
    cid, cstep, buck = _synthetic()
    rows, _ = select_windows(cid, cstep, buck, {"flat": 800, "stairs": 300})
    assert np.unique(rows).size == rows.size


# --- bucketing -------------------------------------------------------------------------


def test_window_buckets_maps_codes_through_family_names():
    out = window_buckets(np.array([0, 2, 1, 2]), ["unknown", "family_stairs_up", "family_jump_on"])
    assert out.tolist() == ["unknown", "family_jump_on", "family_stairs_up", "family_jump_on"]


def test_window_buckets_rejects_a_code_with_no_name():
    with pytest.raises(ValueError, match="disagree"):
        window_buckets(np.array([0, 5]), ["unknown", "family_stairs_up"])


def test_window_buckets_by_type_joins_through_clip_id():
    codes = np.array([0, 4, 2])                      # clip 0 flat, clip 1 box, clip 2 transition
    names = ["flat", "terrain", "transition", "slope", "box"]
    out = window_buckets_by_type(np.array([2, 0, 1, 1]), codes, names)
    assert out.tolist() == ["transition", "flat", "box", "box"]


def test_window_buckets_by_type_rejects_a_foreign_library():
    with pytest.raises(ValueError, match="different library"):
        window_buckets_by_type(np.array([0, 7]), np.array([0, 1]), ["flat", "terrain"])


def test_bucket_supply_counts_windows_and_clips():
    cid, _, buck = _synthetic()
    supply = bucket_supply(buck, cid)
    assert supply["flat"] == {"windows": 800, "clips": 4}
    assert supply["stairs"] == {"windows": 300, "clips": 6}
    assert list(supply) == ["flat", "stairs"]  # ordered by descending window count


# --- window-level terrain span ---------------------------------------------------------


def _pinned_clip(n_steps: int, pinned_from: int, pinned_to: int) -> np.ndarray:
    """One clip's per-step pinned flag: flat, then a pinned run, then flat again."""
    p = np.zeros((1, n_steps), dtype=bool)
    p[0, pinned_from:pinned_to] = True
    return p


def test_terrain_span_classifies_the_three_cases():
    # Clip 0: flat 0-99, pinned 100-199, flat 200-299. Span of 20 nodes.
    pinned = _pinned_clip(300, 100, 200)
    cid = np.zeros(4, dtype=np.int64)
    step = np.array([0, 90, 120, 250])          # flat / crossing in / on terrain / flat
    out = window_buckets_by_terrain_span(cid, step, pinned, span_len=20)
    assert out.tolist() == ["flat", "crossing", "on_terrain", "flat"]


def test_terrain_span_catches_the_exit_boundary_too():
    pinned = _pinned_clip(300, 100, 200)
    out = window_buckets_by_terrain_span(
        np.zeros(1, dtype=np.int64), np.array([190]), pinned, span_len=20)
    assert out.tolist() == ["crossing"]


def test_terrain_span_clamps_into_the_clips_real_frames():
    # The library pads by repeating the last frame; without the clamp a span running past
    # clip_len would read padding and extend the final segment for free.
    pinned = np.zeros((1, 300), dtype=bool)
    pinned[0, 90:100] = True                     # pinned run ends AT the clip's real end
    out = window_buckets_by_terrain_span(
        np.zeros(1, dtype=np.int64), np.array([95]), pinned, span_len=50,
        clip_len=np.array([100]),
    )
    assert out.tolist() == ["on_terrain"]        # padding is the held last (pinned) frame


def test_terrain_span_is_blocked_consistently():
    rng = np.random.default_rng(0)
    pinned = rng.random((5, 400)) > 0.5
    cid = rng.integers(0, 5, 500)
    step = rng.integers(0, 300, 500)
    a = window_buckets_by_terrain_span(cid, step, pinned, 30, block=7)
    b = window_buckets_by_terrain_span(cid, step, pinned, 30, block=10_000)
    assert a.tolist() == b.tolist()


def test_terrain_span_rejects_a_foreign_library():
    with pytest.raises(ValueError, match="different library"):
        window_buckets_by_terrain_span(
            np.array([0, 9]), np.array([0, 0]), np.zeros((2, 50), dtype=bool), 10)


def test_terrain_span_rejects_a_nonpositive_span():
    with pytest.raises(ValueError, match="span_len"):
        window_buckets_by_terrain_span(
            np.array([0]), np.array([0]), np.zeros((1, 50), dtype=bool), 0)


# --- axis joining ----------------------------------------------------------------------


def test_join_buckets_builds_a_compound_key():
    a = np.array(["transition", "box"], dtype=object)
    b = np.array(["crossing", "flat"], dtype=object)
    assert join_buckets(a, b).tolist() == ["transition/crossing", "box/flat"]


def test_join_buckets_passes_a_single_axis_through():
    a = np.array(["flat", "box"], dtype=object)
    assert join_buckets(a).tolist() == ["flat", "box"]


def test_join_buckets_rejects_mismatched_axes():
    with pytest.raises(ValueError, match="disagree in length"):
        join_buckets(np.array(["a"], dtype=object), np.array(["a", "b"], dtype=object))


def test_join_buckets_needs_an_axis():
    with pytest.raises(ValueError, match="at least one axis"):
        join_buckets()


def test_compound_buckets_select_independently():
    """A quota on 'transition/crossing' must not pull in 'transition/flat' windows."""
    cid = np.repeat(np.arange(4), 50)
    cstep = np.tile(np.arange(50) * 12, 4)
    clip_axis = np.where(cid < 2, "transition", "box").astype(object)
    span_axis = np.where(cstep < 300, "flat", "crossing").astype(object)
    bucket = join_buckets(clip_axis, span_axis)
    rows, report = select_windows(cid, cstep, bucket, {"transition/crossing": 20})
    assert report["transition/crossing"]["taken"] == 20
    assert set(bucket[rows]) == {"transition/crossing"}
    assert set(cid[rows].tolist()) == {0, 1}
