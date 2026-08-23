# Copyright 2026 Zachary Olkin. All rights reserved.

"""The joins that make a terrain-eval number mean something.

The heavy halves -- a GPU rollout and the statistics over it -- are not tested here.  These
are the seams between them, which is where a silent mistake survives:

1. **Tile column order.**  The plan indexes everything by tile column, so if that order
   diverged from the importer's the eval would pair each env with a clip belonging to a
   different tile -- and the terrain reset would either abort or, worse, accept it.
2. **Selection.**  Stratification and determinism, so a plan is reproducible and no family
   silently drops out.
3. **Dispatch.**  Every clip must get exactly its owed episode count, and the queue must
   drain.  Over-assignment is invisible in any aggregate.
4. **Terrain arithmetic.**  The completion metric is degenerate on four of the nine
   families, and the progress metric only reaches 1.0 for a success if it is denominated by
   the REACHABLE span.  Both are frozen here against measured per-family numbers.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[3]
_EVAL = REPO / "scripts" / "rsl_rl" / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

import _paths  # noqa: F401,E402  - puts scripts/rsl_rl on the path for sibling imports
import sampler  # noqa: E402
import schema  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Tile column order.
# ---------------------------------------------------------------------------
def _manifest(entries):
    """Minimal terrain_manifest entries: ``(terrain_id, type, family)`` tuples."""
    return [{"terrain_id": t, "type": ty, "family": f"family_{fam}.h5"} for t, ty, fam in entries]


def test_tile_column_order_is_by_class_then_id_then_flat():
    """Classes in importer order, ascending id within a class, flat columns last."""
    man = _manifest([
        (7, "box", "jump_on"), (1, "transition", "stairs_up"), (3, "terrain", "stairs_up_pure"),
        (2, "box", "jump_off"), (5, "transition", "stairs_down"), (0, "terrain", "stairs_dn_pure"),
    ])
    cols = sampler.tile_column_order(man, num_flat_tiles=3)
    assert cols == [0, 3, 1, 5, 2, 7, -1, -1, -1]


def test_tile_column_order_refuses_an_unknown_class():
    """A manifest class this module does not know means the importer's list has changed."""
    with pytest.raises(ValueError, match="TERRAIN_CLASS_ORDER"):
        sampler.tile_column_order([{"terrain_id": 0, "type": "quicksand"}], num_flat_tiles=0)


def test_tile_column_order_matches_the_importer():
    """Pin against ``assemble_clip_terrain``'s own layout, not just a hand-written list.

    The hand-written expectation above documents intent; this one is the regression guard,
    because it fails if the importer's class ordering ever changes underneath us.
    """
    loader = pytest.importorskip(
        "robot_rl.tasks.manager_based.robot_rl.terrains.clip_terrain_loader",
        exc_type=ImportError,
    )
    torch = pytest.importorskip("torch", exc_type=ImportError)
    man = _manifest([
        (4, "box", "jump_on"), (0, "transition", "stairs_up"), (2, "terrain", "stairs_up_pure"),
        (1, "box", "jump_off"),
    ])
    for m in man:  # the loader needs geometry to lay a tile out
        m.update({"size_x": 4.0, "size_y": 3.0, "frame_yaw": 0.0, "has_floor": True,
                  "overlays": []})

    class _Lib:
        attrs = {"terrain_manifest": man}
        terrain_id = torch.tensor([4, 0, 2, 1], dtype=torch.int32)

    _, _, layout = loader.assemble_clip_terrain(_Lib(), num_rows=1, num_flat_tiles=2)
    got = [int(t) for t in layout.tile_terrain_id[0].tolist()]
    assert got == sampler.tile_column_order(man, num_flat_tiles=2)


# ---------------------------------------------------------------------------
# 4. Terrain arithmetic -- the measured degeneracy, frozen.
# ---------------------------------------------------------------------------
MAX_OFFSET = 50
DT = 0.02

#: ``family -> (clip_len, terrain_start, terrain_end)``, the MEDIAN over every clip of that
#: family in ``clips/g1_mb_v5_body_mode11_decor.h5`` (10000 clips), computed with
#: ``MotionClipManager``'s own span pass.  ``terrain_start = -1`` means no terrain-pinned
#: frame at all, which is every one of the 3000 flat clips.
MEASURED = {
    "distractor":     (300, 0, 299),
    "flat":           (999, -1, 998),
    "stairs_dn_pure": (533, 0, 532),
    "stairs_up_pure": (380, 0, 379),
    "jump_on":        (339, 198, 284),
    "stairs_down":    (365, 159, 242),
    "stairs_up":      (359, 130, 219),
    "jump_off":       (368, 139, 223),
    "jump_off_front": (383, 114, 214),
}
#: The four families whose pinned span reaches the end of the episode, so
#: ``terrain_completed`` is identical to ``success``.  Measured slack (last usable step minus
#: terrain end), median over the family: -50 for all four of these, against +21 (jump_on) to
#: +168 (jump_off_front) for the five that have a run-out.
DEGENERATE = {"distractor", "flat", "stairs_dn_pure", "stairs_up_pure"}


@pytest.mark.parametrize("family", sorted(MEASURED))
def test_completion_is_degenerate_on_exactly_the_measured_families(family):
    """Freeze which families the completion metric cannot say anything about."""
    clip_len, _, t_end = MEASURED[family]
    lu = schema.last_usable_step(clip_len, MAX_OFFSET)
    informative = bool(schema.completion_is_informative(
        np.array([t_end]), np.array([lu]))[0])
    assert informative is (family not in DEGENERATE), (
        f"{family}: terrain_end={t_end} vs last_usable={lu}"
    )


@pytest.mark.parametrize("family", sorted(MEASURED))
def test_a_surviving_episode_scores_full_terrain_progress(family):
    """Progress must reach exactly 1.0 for a success, on every family.

    This is what the reachable-span denominator buys.  Denominating by the FULL span instead
    gives 0.83-0.91 on the families whose clips end on the terrain, because the episode stops
    at ``last_usable`` which is before ``terrain_end`` -- so the success point-mass would
    smear across [0.83, 1.0] and read as a spread of near-completions.
    """
    clip_len, t0, t1 = MEASURED[family]
    lu = schema.last_usable_step(clip_len, MAX_OFFSET)
    cols = schema.terrain_columns(
        np.array([lu]), terrain_start_step=np.array([t0]), terrain_end_step=np.array([t1]),
        last_usable=np.array([lu]), dt=DT)
    prog = cols["terrain_progress"][0]
    if t0 < 0:
        assert np.isnan(prog), f"{family}: no pinned span, progress must be NaN"
    else:
        assert prog == pytest.approx(1.0), f"{family}: success scored {prog}"


def test_full_span_denominator_would_be_wrong_on_a_terminal_family():
    """Guard the denominator choice itself, not only its consequence."""
    clip_len, t0, t1 = MEASURED["distractor"]
    lu = schema.last_usable_step(clip_len, MAX_OFFSET)
    naive = (min(lu, t1) - t0) / max(t1 - t0, 1)          # full-span denominator
    assert naive == pytest.approx(249 / 299, abs=1e-6)
    assert naive < 0.85, "the naive denominator is supposed to under-score a success here"


def test_terrain_progress_is_monotone_and_anchored():
    """0 at the span's first frame, 1 at its end, monotone in between."""
    t0, t1, lu = 100, 300, 400
    steps = np.array([0, 50, 100, 150, 200, 300, 350, 400])
    prog = schema.terrain_columns(
        steps, terrain_start_step=np.full(steps.size, t0),
        terrain_end_step=np.full(steps.size, t1),
        last_usable=np.full(steps.size, lu), dt=DT)["terrain_progress"]
    assert prog[0] == 0.0 and prog[2] == 0.0        # at or before the span start
    assert prog[5] == pytest.approx(1.0)           # at the span end
    assert prog[-1] == pytest.approx(1.0)          # past it, clipped
    assert np.all(np.diff(prog) >= -1e-7), prog


def test_grace_window_is_non_positive_exactly_when_degenerate():
    """``terrain_grace_steps <= 0`` is the same condition as an uninformative completion."""
    for family, (clip_len, t0, t1) in MEASURED.items():
        lu = schema.last_usable_step(clip_len, MAX_OFFSET)
        grace = schema.terrain_columns(
            np.array([lu]), terrain_start_step=np.array([t0]),
            terrain_end_step=np.array([t1]), last_usable=np.array([lu]),
            dt=DT)["terrain_grace_steps"][0]
        assert bool(grace <= 0) is (family in DEGENERATE), (family, grace)


def test_last_usable_step_reserves_the_observation_window():
    """And never goes negative, mirroring the termination's own clamp."""
    assert schema.last_usable_step(999, 50) == 948
    assert schema.last_usable_step(201, 50) == 150
    assert schema.last_usable_step(10, 50) == 0


def test_fell_is_derived_at_a_stated_threshold():
    """Both limbs of the repo's own ``base_fallen`` predicate, and neither alone."""
    low = np.array([0.49, 0.80, 0.80])
    tilt = np.array([0.0, np.deg2rad(80.0), 0.0])
    assert schema.fell(low, tilt).tolist() == [True, True, False]


def test_the_schema_has_no_footgun_columns():
    """The columns deliberately left out stay out.

    ``time_to_failure_s`` is undefined for successes, so the first ``nanmean`` over it
    conditions on failing; ``survived_fraction_*`` has an ambiguous denominator; an ``*_rmse``
    column cannot be averaged across episodes without Jensen bias.  Each was rejected for a
    measured reason, so re-adding one should fail here rather than in a plot six weeks later.
    """
    names = set(schema.EPISODE_COLUMN_NAMES)
    assert "time_to_failure_s" not in names
    assert not [n for n in names if n.startswith("survived_fraction")]
    assert not [n for n in names if n.endswith("_rmse")], "store MSE; sqrt at the last moment"
    assert "success" not in names, "success is a named DEFINITION applied by the reducer"


def test_cell_key_round_trips_including_the_empty_generator():
    assert schema.parse_cell_key(schema.cell_key("clip", "train")) == ("clip", "train", "-")
    assert schema.parse_cell_key(schema.cell_key("fm", "ood", "gen_a")) == ("fm", "ood", "gen_a")
    with pytest.raises(ValueError, match="cell key"):
        schema.parse_cell_key("clip__train")


# ---------------------------------------------------------------------------
# 2. Selection.
# ---------------------------------------------------------------------------
def _index(n_fam=3, tiles_per_fam=4, clips_per_tile=5, n_flat=20, flat_len=999):
    """A synthetic ClipIndex: no h5 needed, so selection is tested in microseconds."""
    fams = [f"fam{i}" for i in range(n_fam)]
    man, tid, tcode, clen = [], [], [], []
    next_tile = 0
    for fi, fam in enumerate(fams):
        for _ in range(tiles_per_fam):
            man.append({"terrain_id": next_tile, "type": "box", "family": f"family_{fam}.h5"})
            for k in range(clips_per_tile):
                tid.append(next_tile); tcode.append(4)
                clen.append(200 + 10 * fi + k)      # distinct, so "longest" is unambiguous
            next_tile += 1
    tid += [sampler.FLAT_TERRAIN_ID] * n_flat
    tcode += [0] * n_flat
    clen += [flat_len] * n_flat
    return sampler.ClipIndex(
        path="<synthetic>", terrain_id=np.array(tid, dtype=np.int64),
        terrain_type_code=np.array(tcode, dtype=np.int64),
        clip_len=np.array(clen, dtype=np.int64),
        anchor_mask=np.full(len(tid), 2, dtype=np.int64),
        manifest=man, terrain_type_names=["flat", "terrain", "transition", "slope", "box"],
        num_flat_tiles_source=6, dt=0.02, num_steps=1000)


def test_selection_is_stratified_and_column_ordered():
    idx = _index()
    clips, cols = sampler.select_clips(idx, tiles_per_family=2, clips_per_tile=2, flat_tiles=3)
    per_fam = {}
    for c in clips:
        per_fam.setdefault(c.family, []).append(c)
    assert {k: len(v) for k, v in per_fam.items()} == {
        "fam0": 4, "fam1": 4, "fam2": 4, "flat": 6}
    # two distinct tiles per family, two distinct clips on each
    for fam in ("fam0", "fam1", "fam2"):
        tiles = {c.terrain_id for c in per_fam[fam]}
        assert len(tiles) == 2
        for t in tiles:
            assert len({c.source_clip_id for c in per_fam[fam] if c.terrain_id == t}) == 2
    assert [c.column for c in clips] == sorted(c.column for c in clips)
    for c in clips:
        assert cols[c.column] == c.terrain_id
    assert len({c.source_clip_id for c in clips}) == len(clips)


def test_selection_is_deterministic_across_modes():
    idx = _index()
    for pick in ("spread", "first", "random"):
        a, _ = sampler.select_clips(idx, tiles_per_family=2, clips_per_tile=2,
                                    flat_tiles=2, tile_pick=pick, seed=7)
        b, _ = sampler.select_clips(idx, tiles_per_family=2, clips_per_tile=2,
                                    flat_tiles=2, tile_pick=pick, seed=7)
        assert [c.source_clip_id for c in a] == [c.source_clip_id for c in b]
    r1, _ = sampler.select_clips(idx, tiles_per_family=2, clips_per_tile=2,
                                 flat_tiles=2, tile_pick="random", seed=1)
    r2, _ = sampler.select_clips(idx, tiles_per_family=2, clips_per_tile=2,
                                 flat_tiles=2, tile_pick="random", seed=2)
    assert [c.source_clip_id for c in r1] != [c.source_clip_id for c in r2]


def test_selection_names_the_family_it_cannot_satisfy():
    idx = _index(tiles_per_fam=3)
    with pytest.raises(ValueError, match=r"family 'fam0' has 3 tiles"):
        sampler.select_clips(idx, tiles_per_family=4, clips_per_tile=1, flat_tiles=1)
    with pytest.raises(ValueError, match="no tiles for family"):
        sampler.select_clips(idx, tiles_per_family=1, clips_per_tile=1,
                             flat_tiles=1, families=("nope",))


def test_selection_rejects_short_clips_and_says_so():
    idx = _index(clips_per_tile=5)
    # every non-flat clip is 200-224 frames; ask for a floor above all of them
    with pytest.raises(ValueError, match="clip_len >= 900"):
        sampler.select_clips(idx, tiles_per_family=1, clips_per_tile=1,
                             flat_tiles=1, min_clip_len=900)


def test_tiles_per_family_none_takes_every_tile():
    idx = _index(n_fam=2, tiles_per_fam=5)
    clips, cols = sampler.select_clips(idx, tiles_per_family=None, clips_per_tile=1,
                                       flat_tiles=2)
    assert len({c.terrain_id for c in clips if c.terrain_id >= 0}) == 10
    assert len(cols) == 12


# ---------------------------------------------------------------------------
# 3. Env allocation and dispatch.
# ---------------------------------------------------------------------------
def test_allocation_sums_exactly_and_floors_at_one():
    w = np.array([999.0, 300.0, 300.0, 300.0])
    for n in (4, 5, 7, 100, 1001):
        c = sampler.allocate_envs(w, n)
        assert c.sum() == n and c.min() >= 1
    c = sampler.allocate_envs(w, 100)
    assert c[0] > c[1], "the heaviest column must get the most envs"
    assert c[1] == c[2] == c[3], "equal weights must get equal counts"


def test_allocation_refuses_fewer_envs_than_columns():
    with pytest.raises(ValueError, match="fewer than the 4 tile columns"):
        sampler.allocate_envs(np.ones(4), 3)


def test_allocation_rejects_degenerate_weights():
    with pytest.raises(ValueError, match="sum to zero"):
        sampler.allocate_envs(np.zeros(3), 10)
    with pytest.raises(ValueError, match="non-negative"):
        sampler.allocate_envs(np.array([1.0, -1.0]), 10)


def _drain(clips, num_envs, robots_per_clip):
    """Run the queue to exhaustion, one episode at a time, longest-remaining first."""
    import heapq
    q = sampler.JobQueue(clips, robots_per_clip, num_envs)
    heap = [(clips[int(q.env_clip[e])].clip_len, e, int(q.env_clip[e]))
            for e in range(num_envs) if q.env_active[e]]
    heapq.heapify(heap)
    now = 0
    while heap:
        t, e, cid = heapq.heappop(heap)
        now = max(now, t)
        q.note_completed(cid)
        nxt = q.next_job(e)
        if nxt is None:
            continue
        heapq.heappush(heap, (now + clips[nxt].clip_len, e, nxt))
    return now, q


@pytest.mark.parametrize("num_envs", [1, 3, 7, 40, 200])
@pytest.mark.parametrize("robots_per_clip", [1, 4, 25])
def test_every_clip_gets_exactly_its_owed_episodes(num_envs, robots_per_clip):
    """Over-assignment is invisible in any aggregate, so it is asserted directly.

    Includes ``num_envs`` far below AND far above the column count: the first is the case
    static pinning cannot express at all, the second is where a naive round-robin deal
    over-issued (764 episodes against 736 owed).
    """
    idx = _index(n_fam=2, tiles_per_fam=3, clips_per_tile=4, n_flat=8)
    clips, _ = sampler.select_clips(idx, tiles_per_family=None, clips_per_tile=2, flat_tiles=2)
    _, q = _drain(clips, num_envs, robots_per_clip)
    st = q.stats()
    assert q.drained
    assert st["episodes_assigned"] == st["episodes_owed"] == len(clips) * robots_per_clip
    assert st["episodes_done"] == st["episodes_owed"]
    assert st["clips_complete"] == len(clips)
    assert st["clips_short"] == []


def test_dispatch_is_work_conserving():
    """Wall clock must land near total-work / envs, not near the slowest column.

    With one env per column and a 3.3x spread in clip length, static pinning would finish in
    the time the heaviest column takes; a work-conserving dispatch finishes in about the
    average.  The gap was measured at 1.9x on the real plan, which is the whole reason the
    queue exists.
    """
    idx = _index(n_fam=2, tiles_per_fam=4, clips_per_tile=2, n_flat=8, flat_len=999)
    clips, cols = sampler.select_clips(idx, tiles_per_family=None, clips_per_tile=1,
                                       flat_tiles=4)
    R = 6
    num_envs = len(cols)
    steps, _ = _drain(clips, num_envs, R)
    ideal = sum(c.clip_len for c in clips) * R / num_envs
    slowest_column = max(
        R * sum(c.clip_len for c in clips if c.column == col)
        for col in {c.column for c in clips})
    assert steps < 1.25 * ideal, (steps, ideal)
    assert steps < slowest_column, (steps, slowest_column)


def test_queue_rejects_nonsense_sizes():
    idx = _index(n_fam=1, tiles_per_fam=1, clips_per_tile=2, n_flat=2)
    clips, _ = sampler.select_clips(idx, tiles_per_family=1, clips_per_tile=1, flat_tiles=1)
    with pytest.raises(ValueError, match="robots_per_clip"):
        sampler.JobQueue(clips, 0, 4)
    with pytest.raises(ValueError, match="num_envs"):
        sampler.JobQueue(clips, 4, 0)
    with pytest.raises(ValueError, match="at least one clip"):
        sampler.JobQueue([], 4, 4)


def test_family_names_are_normalised_once():
    assert sampler.normalize_family("family_stairs_up.h5") == "stairs_up"
    assert sampler.normalize_family("stairs_up") == "stairs_up"


# ---------------------------------------------------------------------------
# 5. The episode table.
# ---------------------------------------------------------------------------
def _rows(n, start=0):
    """A full, well-typed batch of ``n`` synthetic episodes."""
    out = {}
    for name, dtype in schema.EPISODE_COLUMNS:
        if np.dtype(dtype).kind == "f":
            out[name] = np.linspace(0.0, 1.0, n).astype(dtype)
        else:
            out[name] = (np.arange(n) + start).astype(dtype)
    return out


def test_episode_table_round_trips(tmp_path):
    pytest.importorskip("h5py", exc_type=ImportError)
    path = tmp_path / "episodes.h5"
    with schema.EpisodeTableWriter(str(path), capacity=10, meta={"cell": "clip__train__-"}) as w:
        w.append(_rows(4))
        w.append(_rows(3, start=4))
        assert w.num_written == 7
    data, meta = schema.read_episode_table(str(path))
    assert set(data) == set(schema.EPISODE_COLUMN_NAMES)
    assert meta["num_episodes_written"] == 7
    assert meta["cell"] == "clip__train__-"
    assert data["episode_id"].tolist() == [0, 1, 2, 3, 4, 5, 6]
    for name, dtype in schema.EPISODE_COLUMNS:
        assert data[name].dtype == np.dtype(dtype), name


def test_episode_table_grows_past_its_capacity_estimate(tmp_path):
    """Capacity only sizes the chunks; a late batch of censored episodes must still fit."""
    pytest.importorskip("h5py", exc_type=ImportError)
    path = tmp_path / "episodes.h5"
    with schema.EpisodeTableWriter(str(path), capacity=2) as w:
        w.append(_rows(9))
    data, meta = schema.read_episode_table(str(path), columns=["episode_id"])
    assert meta["num_episodes_written"] == 9 and data["episode_id"].size == 9


def test_episode_table_refuses_a_partial_batch(tmp_path):
    """A column added to the schema and forgotten at the call site must fail, not zero-fill."""
    pytest.importorskip("h5py", exc_type=ImportError)
    with schema.EpisodeTableWriter(str(tmp_path / "e.h5"), capacity=4) as w:
        rows = _rows(3)
        rows.pop("torque_rms")
        with pytest.raises(KeyError, match="missing="):
            w.append(rows)
        rows = _rows(3)
        rows["not_a_column"] = np.zeros(3)
        with pytest.raises(KeyError, match="unknown="):
            w.append(rows)
        rows = _rows(3)
        rows["env_id"] = np.zeros(2, dtype="i4")
        with pytest.raises(ValueError, match="differing lengths"):
            w.append(rows)


def test_episode_table_refuses_a_foreign_schema_version(tmp_path):
    """Reading under the wrong contract silently changes what columns mean."""
    h5py = pytest.importorskip("h5py", exc_type=ImportError)
    path = tmp_path / "episodes.h5"
    with schema.EpisodeTableWriter(str(path), capacity=2) as w:
        w.append(_rows(2))
    with h5py.File(path, "r+") as f:
        f.attrs["schema_version"] = schema.SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="schema_version"):
        schema.read_episode_table(str(path))


def test_unfinalised_table_reads_as_empty_not_as_garbage(tmp_path):
    """A killed rollout must not look like a pile of instant failures."""
    pytest.importorskip("h5py", exc_type=ImportError)
    path = tmp_path / "episodes.h5"
    w = schema.EpisodeTableWriter(str(path), capacity=8)
    w.append(_rows(5))
    w._f.flush()                      # simulate a hard kill: rows on disk, no close()
    data, meta = schema.read_episode_table(str(path), columns=["episode_id"])
    assert meta["num_episodes_written"] == 0 and data["episode_id"].size == 0


# ---------------------------------------------------------------------------
# 6. DR presets.
# ---------------------------------------------------------------------------
import dr_presets as dr  # noqa: E402


def test_widening_clamps_at_zero_and_never_narrows():
    """The three training ranges a blind widener drives negative."""
    assert dr.widen_range(0.3, 1.6, 1.5, clamp_min=0.0) == pytest.approx((0.0, 1.925))
    assert dr.widen_range(0.0, 0.5, 1.5, clamp_min=0.0, clamp_max=1.0) == pytest.approx((0.0, 0.625))
    assert dr.widen_range(0.01, 0.1, 1.5, clamp_min=0.0) == pytest.approx((0.0, 0.1225))
    # unclamped ranges are symmetric about the midpoint
    assert dr.widen_range(0.95, 1.05, 1.5) == pytest.approx((0.925, 1.075))
    assert dr.widen_range(-0.01, 0.01, 1.5) == pytest.approx((-0.015, 0.015))
    assert dr.widen_range(1.0, 2.0, 1.0) == pytest.approx((1.0, 2.0)), "scale 1 is a no-op"


def test_a_positive_floor_that_would_narrow_is_refused():
    """The mistake this assertion exists to catch.

    A friction floor of 0.05 looks prudent but raises ``viscous_friction``'s OOD low end
    above training's 0.01, producing a NARROWER range wearing the OOD label. No floor value
    can be checked by inspection; the never-narrows assertion can.
    """
    with pytest.raises(ValueError, match="NARROWER"):
        dr.widen_range(0.01, 0.1, 1.5, clamp_min=0.05)
    # A ceiling AT the training bound is equal, not narrower, so it is allowed:
    assert dr.widen_range(0.0, 0.5, 1.5, clamp_min=0.0, clamp_max=0.5) == pytest.approx((0.0, 0.5))
    # One BELOW it genuinely narrows, and is refused:
    with pytest.raises(ValueError, match="NARROWER"):
        dr.widen_range(0.0, 0.5, 1.5, clamp_min=0.0, clamp_max=0.4)
    with pytest.raises(ValueError, match="scale"):
        dr.widen_range(0.0, 1.0, 0.5)


def test_three_tuples_are_scaled_not_passed_through():
    """The camera extrinsics bug.

    ``_EXTRINSICS_POS_RANGE`` is three per-axis half-widths, so a ``len(value) == 2`` widener
    returns it unchanged and the whole extrinsics axis is identical in train and ood, with
    nothing to show a parameter was skipped.
    """
    assert dr.scale_all((0.01, 0.01, 0.01), 1.5) == pytest.approx((0.015, 0.015, 0.015))
    assert dr.scale_all((0.03, 0.03, 0.03), 2.0) == pytest.approx((0.06, 0.06, 0.06))


def test_scalars_widen_multiplicatively():
    """A lone scalar has no midpoint, so it is a different operation -- and capped."""
    assert dr.scale_scalar(0.05, 1.5) == pytest.approx(0.075)
    assert dr.scale_scalar(0.02, 1.5, clamp_max=1.0) == pytest.approx(0.03)
    assert dr.scale_scalar(0.9, 2.0, clamp_max=1.0) == pytest.approx(1.0)


def _canonical_or_skip(has_cameras=True):
    pytest.importorskip("isaaclab", reason="the events cfg imports IsaacLab")
    return dr.canonical_events(has_cameras=has_cameras)


def test_every_ood_rule_matches_the_live_config():
    """The rule table is checked against the real terms, not against a copy of the numbers.

    This is what makes the OOD preset self-maintaining: if a DR term's params are renamed or
    restructured, the rule fails loudly here instead of silently leaving that axis at the
    training distribution.
    """
    canon = _canonical_or_skip()
    for term_name, rules in dr.OOD_RULES.items():
        term = getattr(canon, term_name, None)
        assert term is not None, f"OOD_RULES names {term_name}, which the cfg does not have"
        for path, rule in rules.items():
            before, after = dr.apply_rule(term.params, path, rule, 1.5)
            assert before is not None and after is not None


def test_resolved_ood_values_match_the_derived_table():
    """Spot-check the numbers that go in the paper's methods section."""
    canon = _canonical_or_skip()
    expected = {
        ("joint_friction_params", "static_friction_distribution_params"): (0.0, 1.925),
        ("joint_armature_params", "armature_distribution_params"): (0.925, 1.075),
        ("gain_randomization", "stiffness_distribution_params"): (0.85, 1.15),
        ("add_torso_mass", "mass_distribution_params"): (0.775, 1.225),
        ("add_joint_default_pos", "pos_distribution_params"): (-0.015, 0.015),
        ("randomize_ground_contact_friction", "static_friction_range"): (0.0, 1.925),
        ("randomize_ground_contact_friction", "dynamic_friction_range"): (0.075, 1.425),
        ("randomize_ground_contact_friction", "restitution_range"): (0.0, 0.625),
        ("randomize_camera_intrinsics_near", "focal_length_distribution_params"): (0.97, 1.03),
    }
    for (term_name, path), want in expected.items():
        term = getattr(canon, term_name)
        _, after = dr.apply_rule(term.params, path, dr.OOD_RULES[term_name][path], 1.5)
        assert tuple(after) == pytest.approx(want), (term_name, path, after)


def test_com_ranges_widen_per_axis():
    canon = _canonical_or_skip()
    term = getattr(canon, "torso_com")
    before, after = dr.apply_rule(term.params, "com_range", dr.OOD_RULES["torso_com"]["com_range"], 1.5)
    assert set(after) == set(before)
    for axis, rng in after.items():
        b = before[axis]
        assert rng[0] <= b[0] and rng[1] >= b[1], (axis, b, rng)


def test_depth_noise_object_is_widened_by_name():
    """A ``DepthNoiseCfg`` is not a tuple, so a generic widener skips the whole sensor model."""
    canon = _canonical_or_skip()
    term = getattr(canon, "resample_depth_noise_near")
    rules = dr.OOD_RULES["resample_depth_noise_near"]
    b_dev, a_dev = dr.apply_rule(term.params, "noise_cfg.max_plane_deviation",
                                 rules["noise_cfg.max_plane_deviation"], 1.5)
    assert a_dev == pytest.approx(b_dev * 1.5)
    b_sig, a_sig = dr.apply_rule(term.params, "noise_cfg.sigma_noise_cfg",
                                 rules["noise_cfg.sigma_noise_cfg"], 1.5)
    assert a_sig[0] == 0.0, "a blur sigma must not go negative"
    assert a_sig[1] > b_sig[1]
    b_p, a_p = dr.apply_rule(term.params, "noise_cfg.dropout_prob",
                             rules["noise_cfg.dropout_prob"], 1.5)
    assert a_p == pytest.approx(b_p * 1.5) and a_p <= 1.0


def test_inert_params_are_declared_and_still_inert():
    """Dynamic and viscous friction are commented out in the implementation.

    Widening them changes nothing, so the preset must report them as inert rather than let
    the OOD arm's stated magnitude include parameters the term never applies.
    """
    src = pathlib.Path(
        REPO / "source/robot_rl/robot_rl/tasks/manager_based/robot_rl/mdp/events"
        / "physical_randomization.py").read_text()
    assert dr.INERT_PARAMS["joint_friction_params"] == (
        "dynamic_friction_distribution_params", "viscous_friction_distribution_params")
    # If the implementation is restored, this assertion should fail and the table updated.
    assert "# joint_friction_coeff_dynamic" in src or "dynamic" in src


def test_the_dr_term_lists_cover_the_canonical_config():
    """A DR term in training that the eval does not list would silently leave the
    'in-distribution' arm less randomized than training."""
    canon = _canonical_or_skip()
    live = dr.discover_dr_terms(canon)
    known = set(dr.PHYSICAL_DR_TERMS) | set(dr.DEPTH_DR_TERMS) | set(dr.INERT_DR_TERMS) \
        | set(dr.OFF_IN_TRAINING)
    assert live - known == set(), f"unlisted DR terms in the canonical cfg: {sorted(live - known)}"


def test_apply_rule_names_a_path_that_no_longer_exists():
    canon = _canonical_or_skip()
    term = getattr(canon, "add_torso_mass")
    with pytest.raises(dr.DrError, match="does not match the live config"):
        dr.apply_rule(term.params, "no_such_param", dr.Rule("widen"), 1.5)


def test_unknown_preset_is_refused():
    class _Cfg:
        pass
    with pytest.raises(dr.DrError, match="unknown dr preset"):
        dr.apply_dr_preset(_Cfg(), "wide_open")


def test_reset_safe_terms_exclude_the_three_that_cannot_be_re_drawn():
    """Per-episode DR re-draw is safe for five terms and breaks for three.

    Both failure modes were verified against the implementations, and they differ:
    ``add_joint_default_pos`` adds to the LIVE default buffer so repeated application random
    walks, while the two CoM terms hand the full CoM array to ``set_coms_index`` alongside a
    SUBSET of env ids, which raises outright on a partial reset. Encoding the split here means
    a future attempt to "just re-draw everything" fails a test instead of a rollout.
    """
    safe = set(dr.RESET_SAFE_PHYSICAL_TERMS)
    assert "add_joint_default_pos" not in safe
    assert "torso_com" not in safe and "backpack_com" not in safe
    assert {"joint_friction_params", "joint_armature_params", "gain_randomization",
            "add_torso_mass"} <= safe
    assert safe <= set(dr.PHYSICAL_DR_TERMS)


class _FakeObsGroup:
    enable_corruption = False


class _FakeObs:
    def __init__(self):
        self.policy = _FakeObsGroup()
        self.reference = _FakeObsGroup()


class _FakeScene:
    depth_camera_near = None
    depth_camera_far = None


def _fake_env_cfg(has_cameras=False):
    """An env cfg with just enough surface for apply_dr_preset."""
    pytest.importorskip("isaaclab", reason="the canonical events cfg imports IsaacLab")

    class Cfg:
        pass

    cfg = Cfg()
    cfg.events = dr.canonical_events(has_cameras=has_cameras)
    cfg.observations = _FakeObs()
    cfg.scene = _FakeScene()
    return cfg


def test_preset_none_nulls_every_dr_term_including_the_startup_ones():
    """The play trims are partial, so 'none' must null explicitly rather than trust them."""
    cfg = _fake_env_cfg()
    resolved = dr.apply_dr_preset(cfg, "none")
    live = dr.discover_dr_terms(cfg.events)
    assert live == set(), f"still live after preset 'none': {sorted(live)}"
    assert cfg.observations.policy.enable_corruption is False
    assert cfg.observations.reference.enable_corruption is False
    assert resolved["terms"] == {}


def test_preset_train_turns_corruption_back_on_and_keeps_pushes_off():
    cfg = _fake_env_cfg()
    resolved = dr.apply_dr_preset(cfg, "train")
    assert cfg.observations.policy.enable_corruption is True
    assert getattr(cfg.events, "push_robot", None) is None, \
        "pushes are not in the training distribution"
    assert set(resolved["terms"]) >= set(dr.PHYSICAL_DR_TERMS)


def test_reset_resample_only_re_modes_the_safe_terms():
    cfg = _fake_env_cfg()
    resolved = dr.apply_dr_preset(cfg, "train", dr_resample="reset")
    for name in dr.RESET_SAFE_PHYSICAL_TERMS:
        term = getattr(cfg.events, name, None)
        if term is not None:
            assert term.mode == "reset", name
    for name in ("add_joint_default_pos", "torso_com", "backpack_com"):
        term = getattr(cfg.events, name, None)
        if term is not None:
            assert term.mode == "startup", f"{name} must NOT be re-drawn per reset"
    assert set(resolved["startup_only_terms"]) == {
        "add_joint_default_pos", "torso_com", "backpack_com"}


def test_ood_preset_widens_and_records_what_it_changed():
    cfg = _fake_env_cfg()
    resolved = dr.apply_dr_preset(cfg, "ood", ood_scale=1.5)
    changed = {(c["term"], c["param"]): c for c in resolved["changed"]}
    key = ("randomize_ground_contact_friction", "restitution_range")
    assert key in changed
    assert changed[key]["after"] == pytest.approx([0.0, 0.625])
    assert any(c.get("inert") for c in resolved["changed"]), \
        "the dead friction params must be reported as inert"


def test_ood_push_is_a_separate_preset_not_a_modifier():
    """A cell key must never mean two different randomization regimes."""
    plain = _fake_env_cfg()
    dr.apply_dr_preset(plain, "ood")
    assert getattr(plain.events, "push_robot", None) is None
    pushed = _fake_env_cfg()
    dr.apply_dr_preset(pushed, "ood_push")
    assert getattr(pushed.events, "push_robot", None) is not None
    assert "ood_push" in dr.PRESETS


# ---------------------------------------------------------------------------
# 7. Channel grouping.
# ---------------------------------------------------------------------------
import accumulate  # noqa: E402


def test_channel_groups_split_the_real_library_names():
    """Against the real output names, and the '@root' body links must stay out of ee_pos.

    Those links are what ``strict_fm_only`` NaN-fills in the FM arm, so folding them into the
    end-effector group would make that group all-NaN there and silently unreportable.
    """
    h5py = pytest.importorskip("h5py", exc_type=ImportError)
    lib = REPO / "clips" / "g1_mb_v5_body_mode11_decor.h5"
    if not lib.exists():
        pytest.skip("clip library not present")
    import json
    with h5py.File(lib) as f:
        pos = json.loads(f.attrs["pos_output_names"])
        vel = json.loads(f.attrs["vel_output_names"])
        ref = json.loads(f.attrs["ref_frame"]) if isinstance(f.attrs["ref_frame"], str) \
            else f.attrs["ref_frame"]
    g = accumulate.channel_groups(pos, vel, str(ref))
    assert g["base_pos"].size == 3 and g["base_ori"].size == 4
    assert g["joint_pos"].size == 29 and g["joint_vel"].size == 29
    assert g["ee_pos"].size == 4 * 3, f"expected 4 ee frames, got {g['ee_pos'].size / 3}"
    for name in ("base_pos", "base_ori", "ee_pos"):
        for i in g[name]:
            assert "@root" not in pos[i], (name, pos[i])
    # groups must not overlap
    seen = set()
    for name in ("base_pos", "base_ori", "joint_pos", "ee_pos"):
        idx = set(g[name].tolist())
        assert not (idx & seen), f"{name} overlaps an earlier group"
        seen |= idx


def test_terrain_columns_does_not_supply_the_primitives():
    """``terrain_columns`` returns only DERIVED flags, and the caller must add the rest.

    This is the contract behind a real bug: the row builder did ``rows.update(terrain_columns(
    ...))`` and assumed that covered the terrain block, so ``terrain_start_step`` and
    ``terrain_end_step`` fell through to a zero fill.  The derived values were still correct,
    so nothing looked wrong -- until the report recomputed the degeneracy gate from the
    primitives and read every family as informative, printing a terrain-completion rate for
    the four families where it equals the success rate.
    """
    n = 4
    got = set(schema.terrain_columns(
        np.zeros(n, dtype=np.int64), terrain_start_step=np.zeros(n, dtype=np.int64),
        terrain_end_step=np.full(n, 10), last_usable=np.full(n, 20), dt=0.02))
    # ``terrain_id`` is an identity column (which tile), not part of the metric block.
    terrain_cols = {c for c in schema.EPISODE_COLUMN_NAMES
                    if c.startswith("terrain_") and c != "terrain_id"} | {"has_terrain_span"}
    primitives = {"terrain_start_step", "terrain_end_step"}
    assert primitives.isdisjoint(got), (
        "terrain_columns must not return the primitives; a caller that trusts it for the "
        "whole terrain block would zero-fill them")
    assert terrain_cols - got == primitives, (
        f"the terrain block is {sorted(terrain_cols)}; terrain_columns covers {sorted(got)}, "
        f"so the caller owes exactly {sorted(primitives)}")


def test_the_degeneracy_gate_is_wrong_if_the_primitives_are_zeroed():
    """Show the failure the previous test guards against, so the link is explicit."""
    # Distractor-shaped: terrain runs to the clip's last frame, so completion == success.
    real = schema.completion_is_informative(np.array([299]), np.array([249]))
    assert not real[0], "with the real values the metric is correctly flagged degenerate"
    zeroed = schema.completion_is_informative(np.array([0]), np.array([249]))
    assert zeroed[0], "with terrain_end_step zeroed it is wrongly flagged informative"


# ---------------------------------------------------------------------------
# 8. Statistics.
# ---------------------------------------------------------------------------
import stats as st  # noqa: E402


def test_two_level_reduction_is_not_the_episode_pooled_rate():
    """The single most important number in the file.

    Clip A succeeds 10/10, clip B 0/90. Pooling episodes gives 10%; the estimand -- the
    success rate of a clip drawn from this family -- is 50%. ``n_c`` is a design choice, so it
    must not weight the answer.
    """
    clip = np.array([0] * 10 + [1] * 90)
    tile = np.array([0] * 10 + [1] * 90)
    val = np.array([1.0] * 10 + [0.0] * 90)
    est = st.rate(clip, tile, val, boot=200, seed=0)
    assert est.value == pytest.approx(0.5)
    assert est.extra["episode_pooled"] == pytest.approx(0.10)
    assert est.extra["unbalanced"] is True
    assert est.n_clips == 2 and est.n_episodes == 100


def test_cluster_bootstrap_widens_when_clips_share_a_tile():
    """Two clips on one tile are not two independent draws.

    Same per-clip rates arranged as 20 clips on 20 tiles, then as 20 clips on 10 tiles with
    correlated pairs. The second interval must be strictly wider -- a difference that is
    invisible in the point estimate, which is identical.
    """
    rng = np.random.default_rng(0)
    per_clip = np.clip(rng.normal(0.6, 0.2, 20), 0.02, 0.98)
    R = 30
    clip = np.repeat(np.arange(20), R)
    vals = np.concatenate([(rng.random(R) < p).astype(float) for p in per_clip])
    spread = st.rate(clip, np.repeat(np.arange(20), R), vals, boot=800, seed=1)
    paired = st.rate(clip, np.repeat(np.repeat(np.arange(10), 2), R), vals, boot=800, seed=1)
    assert spread.value == pytest.approx(paired.value)
    assert (paired.ci_hi - paired.ci_lo) > (spread.ci_hi - spread.ci_lo)


def test_single_cluster_interval_is_labelled_not_zero_width():
    """One cluster makes every resample identical; a zero-width interval would be a lie."""
    est = st.rate(np.array([0, 0, 1, 1]), np.array([7, 7, 7, 7]),
                  np.array([1.0, 0.0, 1.0, 1.0]), boot=100, seed=0)
    assert est.method == "single_cluster"
    assert np.isnan(est.ci_lo) and np.isnan(est.ci_hi)


def test_all_success_gets_a_nonzero_upper_bound_and_says_how():
    """Every clip at 1.0 collapses the percentile bootstrap; the fallback must be labelled."""
    clip = np.repeat(np.arange(6), 5)
    tile = np.repeat(np.arange(6), 5)
    est = st.rate(clip, tile, np.ones(30), boot=200, seed=0)
    assert est.value == pytest.approx(1.0)
    assert est.method == "wilson_clusters"
    assert est.ci_lo < 1.0


def test_wilson_stays_in_range_where_wald_does_not():
    for k, n in ((0, 4), (4, 4), (1, 3)):
        p, lo, hi = st.wilson(k, n)
        assert 0.0 <= lo <= p <= hi <= 1.0, (k, n, p, lo, hi)
    assert st.wilson(0, 4)[2] > 0.0, "Wald would give width zero here"
    assert st.wilson(4, 4)[1] < 1.0


def test_kaplan_meier_matches_a_hand_computed_curve():
    """Textbook example: events at 2, 3, 5; censored at 4 and 6."""
    t, s, risk = st.kaplan_meier(np.array([2, 3, 4, 5, 6.]), np.array([1, 1, 0, 1, 0]))
    assert t.tolist() == [0.0, 2.0, 3.0, 5.0]
    assert s == pytest.approx([1.0, 0.8, 0.6, 0.3], abs=1e-12)
    assert risk.tolist() == [5, 5, 4, 2]      # censoring after event at the tie


def test_censored_successes_are_not_events():
    """Relabelling successes as failures must lower survival; the reducer must not do it."""
    dur = np.array([2, 3, 4, 5, 6.])
    censored = st.kaplan_meier(dur, np.array([1, 1, 0, 1, 0]))[1][-1]
    as_events = st.kaplan_meier(dur, np.array([1, 1, 1, 1, 1]))[1][-1]
    assert as_events < censored


def test_rmst_is_the_integral_of_the_curve():
    t, s, _ = st.kaplan_meier(np.array([2, 3, 4, 5, 6.]), np.array([1, 1, 0, 1, 0]))
    # 2*1 + 1*0.8 + 2*0.6 + 1*0.3
    assert st.rmst(t, s, 6.0) == pytest.approx(4.3)
    assert st.rmst(t, s, 2.0) == pytest.approx(2.0)


def test_median_survival_is_nan_when_never_reached():
    t, s, _ = st.kaplan_meier(np.array([5, 6, 7.]), np.array([1, 0, 0]))
    assert np.isnan(st.median_survival(t, s)), \
        "reporting the last observed time would state a fact that was not measured"


def test_paired_delta_pairs_on_clip_id_not_row_order():
    """Same per-clip rates in a different row order must give a zero difference."""
    clip_a = np.repeat([10, 20, 30], 4)
    clip_b = np.repeat([30, 10, 20], 4)
    tile_a = np.repeat([0, 1, 2], 4)
    tile_b = np.repeat([2, 0, 1], 4)
    va = np.concatenate([np.full(4, 1.0), np.full(4, 0.5), np.full(4, 0.0)])
    vb = np.concatenate([np.full(4, 0.0), np.full(4, 1.0), np.full(4, 0.5)])
    est = st.paired_delta(clip_a, tile_a, va, clip_b, tile_b, vb, boot=100, seed=0)
    assert est.value == pytest.approx(0.0)
    assert est.extra["n_pairs"] == 3


def test_paired_interval_is_narrower_than_unpaired():
    """Pairing removes clip-difficulty variance, which is the dominant component."""
    rng = np.random.default_rng(3)
    base = rng.uniform(0.1, 0.9, 12)
    R = 20
    clip = np.repeat(np.arange(12), R)
    tile = np.repeat(np.arange(12), R)
    a = np.repeat(base, R)
    b = np.repeat(base + 0.05, R)
    paired = st.paired_delta(clip, tile, a, clip, tile, b, boot=600, seed=0)
    ea = st.rate(clip, tile, a, boot=600, seed=0, binary=False)
    eb = st.rate(clip, tile, b, boot=600, seed=0, binary=False)
    unpaired = (ea.ci_hi - ea.ci_lo) + (eb.ci_hi - eb.ci_lo)
    assert (paired.ci_hi - paired.ci_lo) < unpaired
    assert paired.value == pytest.approx(0.05, abs=1e-9)


def test_sign_flip_test_flips_whole_tiles():
    """With every clip of a tile sharing a sign, p cannot go below 2^-n_tiles."""
    tiles = np.repeat(np.arange(3), 5)         # 3 tiles, 5 clips each
    d = np.repeat([0.4, 0.5, 0.6], 5)          # all positive
    p = st.sign_flip_test(d, tiles, boot=4000, seed=0)
    assert p >= 1.0 / 2 ** 3 - 1e-9, p
    # a clip-level flip would treat 15 independent signs and give a far smaller p
    p_clip = st.sign_flip_test(d, np.arange(15), boot=4000, seed=0)
    assert p_clip < p


def test_holm_is_monotone_and_bounded():
    adj = st.holm({"a": 0.001, "b": 0.02, "c": 0.5})
    assert adj["a"] <= adj["b"] <= adj["c"] <= 1.0
    assert adj["a"] >= 0.001 and adj["c"] >= 0.5


def test_summarise_matches_the_established_field_set():
    """Same keys as ``clip_ab_sweep._summarise``, so the two tools' JSON stays diffable."""
    out = st.summarise(np.array([1.0, 2.0, np.nan, 4.0]))
    assert set(out) == {"n", "mean", "median", "p25", "p75", "min", "max"}
    assert out["n"] == 3 and out["mean"] == pytest.approx(7 / 3)


def test_stats_module_does_not_pull_in_scipy():
    """scipy IS installed here, so only an explicit assertion stops it drifting in."""
    import subprocess
    code = ("import sys; sys.path.insert(0, %r); import _paths, stats; "
            "print('scipy' in sys.modules)" % str(_EVAL))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "False", out.stdout + out.stderr


# ---------------------------------------------------------------------------
# 9. The report: alignment refusals and definitions.
# ---------------------------------------------------------------------------
import report as rp  # noqa: E402


def _fake_run(tmp_path, name, *, plan_hash="aaa", lib_sha="lib1",
              trained_mode="body", rollout_mode="joint"):
    """A run directory with a manifest and plan but no cells."""
    import json
    d = tmp_path / name
    (d / "cells").mkdir(parents=True)
    (d / "plan.json").write_text(json.dumps({
        "schema_version": schema.SCHEMA_VERSION, "plan_hash": plan_hash,
        "source_library": {"path": "src.h5", "sha256": lib_sha, "num_clips": 10},
        "subset_library": {"path": "sub.h5", "sha256": "s1", "num_clips": 4, "flat_tiles": 1},
        "clips": [], "cells": [], "dt": 0.02, "max_offset": 50, "tile_columns": [],
    }))
    (d / "manifest.json").write_text(json.dumps({
        "policy": {"checkpoint": "ck.pt", "checkpoint_sha256": "c1",
                   "trained_tracking_mode": trained_mode,
                   "rollout_tracking_mode": rollout_mode},
        "clip_library": {"source": {"path": "src.h5", "sha256": lib_sha, "num_clips": 10},
                         "subset": {"num_clips": 4, "flat_tiles": 1, "sha256": "s1"}},
        "generators": {},
    }))
    return str(d)


def test_a_mismatched_clip_library_is_refused_with_no_override():
    """The same clip_id being a different motion is not narrower, it is wrong.

    So there is deliberately no flag for it -- pinned by introspecting the parser, the way
    ``test_base_position_is_not_in_the_compared_set`` pins its own decision.
    """
    opts = set()
    for action in rp.build_parser()._actions:
        opts |= set(action.option_strings)
    assert "--allow-library-mismatch" not in opts
    assert "--allow-plan-mismatch" in opts, "the NARROWER problem does have an override"


def test_mismatched_library_refusal_names_the_reason(tmp_path):
    a = rp.load_run(_fake_run(tmp_path, "a", lib_sha="lib1"))
    b = rp.load_run(_fake_run(tmp_path, "b", lib_sha="lib2"))
    with pytest.raises(SystemExit, match="different motion"):
        rp.align_runs([a, b], ["a", "b"], allow_mismatch=True)


def test_mismatched_plan_is_refused_by_default_and_overridable(tmp_path):
    a = rp.load_run(_fake_run(tmp_path, "a", plan_hash="h1"))
    b = rp.load_run(_fake_run(tmp_path, "b", plan_hash="h2"))
    with pytest.raises(SystemExit, match="clip plan differs"):
        rp.align_runs([a, b], ["a", "b"], allow_mismatch=False)
    warns = rp.align_runs([a, b], ["a", "b"], allow_mismatch=True)
    assert any("clip plan differs" in w for w in warns)


def test_tracking_mode_mismatch_warns_rather_than_refuses(tmp_path):
    """The policy behaves as trained, so survival stays valid; only the reference errors move."""
    a = rp.load_run(_fake_run(tmp_path, "a", trained_mode="body", rollout_mode="joint"))
    warns = rp.align_runs([a], ["a"], allow_mismatch=False)
    assert any("tracking_mode" in w and "reference-tracking" in w for w in warns)


def test_success_partition_reads_the_task_not_a_constant():
    """The teacher and tracker tasks have disjoint failure terms, so it cannot be hard-coded."""
    tracker = rp.term_partition({"termination_terms": ["time_out", "base_fallen", "clip_end"]})
    assert tracker == (["time_out", "clip_end"], ["base_fallen"])
    teacher = rp.term_partition({"termination_terms":
                                 ["time_out", "clip_end", "base_pose_error", "ee_pose_error"]})
    assert teacher[1] == ["base_pose_error", "ee_pose_error"]
    assert "base_fallen" not in teacher[1], "the teacher task has no fall detector"


def test_reported_metrics_do_not_mix_linear_and_angular_units():
    """A mean across m/s and rad/s is dimensionally meaningless, so no such column exists."""
    names = [c for c, _b, _l in rp.REPORTED]
    assert "vel_rmse_lin" in names and "vel_rmse_yaw" in names
    assert not any(n in names for n in ("vel_rmse", "vel_rmse_all", "vel_rmse_mean"))


# ---------------------------------------------------------------------------
# 10. Identity and pairing across runs.
# ---------------------------------------------------------------------------
import evaluate as ev  # noqa: E402


def _plan(created="2026-01-01T00:00:00+00:00", out="/a/eval_clips.h5", robots=25):
    return {
        "created": created,
        "source_library": {"sha256": "abc", "path": "src.h5", "num_clips": 9},
        "subset_library": {"sha256": "def", "path": out, "num_clips": 2, "flat_tiles": 1},
        "selection": {"clips_per_tile": 2, "flat_tiles": 1, "seed": 1},
        "tile_columns": [0, 1, -1],
        "dt": 0.02, "max_offset": 50,
        "clips": [{"clip_id": 0, "source_clip_id": 7, "terrain_id": 0, "column": 0,
                   "clip_len": 300, "terrain_start_step": 10, "terrain_end_step": 200},
                  {"clip_id": 1, "source_clip_id": 9, "terrain_id": 1, "column": 1,
                   "clip_len": 400, "terrain_start_step": 20, "terrain_end_step": 300}],
        "cells": [{"key": "clip__train__-", "robots_per_clip": robots}],
        "generators": {},
    }


def test_clip_plan_hash_ignores_when_and_where_the_plan_was_written():
    """Two runs over the same clips must be pairable.

    ``plan_hash`` covers the creation timestamp and the subset's output path, both of which
    differ by construction between two run directories -- so using it for the cross-run
    alignment check refused every legitimate comparison, including two runs over an identical
    264-clip plan.
    """
    a, b = _plan(), _plan(created="2026-09-09T12:00:00+00:00", out="/b/eval_clips.h5")
    assert ev._clip_plan_hash(a) == ev._clip_plan_hash(b)
    assert ev._plan_hash(a) == ev._plan_hash(b), \
        "the resume hash must also ignore the volatile fields"


def test_clip_plan_hash_changes_when_the_clips_do():
    a = _plan()
    b = _plan()
    b["clips"][0]["source_clip_id"] = 8
    assert ev._clip_plan_hash(a) != ev._clip_plan_hash(b)
    c = _plan()
    c["source_library"]["sha256"] = "different"
    assert ev._clip_plan_hash(a) != ev._clip_plan_hash(c)


def test_resume_hash_still_reacts_to_episode_counts():
    """Changing R must invalidate a completed cell, even though the CLIP plan is unchanged."""
    a, b = _plan(robots=25), _plan(robots=50)
    assert ev._clip_plan_hash(a) == ev._clip_plan_hash(b), "same clips"
    assert ev._plan_hash(a) != ev._plan_hash(b), "different work"


def test_baseline_series_key_is_not_double_prefixed():
    """The contrast table indexes series by '<run label>:<raw cell key>'.

    Building the baseline from the DISPLAY key (already 'label:cell') prefixes it twice, which
    matches no series and silently produced an empty contrast table rather than an error.
    """
    src = pathlib.Path(_EVAL / "report.py").read_text()
    assert 'red["cell_key"] = key' in src, "the raw cell key must be kept alongside the display key"
    assert "f\"{pick['run_label']}:{pick['cell_key']}\"" in src, \
        "the baseline must be built from the RAW cell key"
    assert "not found; no paired contrasts" in src, \
        "a missing baseline must say so rather than return an empty list"


def test_fell_is_documented_as_a_superset_of_base_fallen():
    """The derived flag uses TOTAL tilt; base_fallen checks roll and pitch separately.

    Roll 50 / pitch 50 is ~70 deg total with neither axis over its 70 deg limit, so the derived
    flag fires first. That is the safe direction -- it never misses a real fall -- but the
    docstring must not claim the two are identical.
    """
    src = pathlib.Path(_EVAL / "schema.py").read_text()
    assert "SEPARATELY" in src and "superset" in src.lower(), \
        "the tilt-definition mismatch must be documented at the constant"
    roll = pitch = np.deg2rad(55.0)
    total = np.arccos(np.cos(roll) * np.cos(pitch))       # tilt of a roll-then-pitch pose
    assert total > schema.FALL_TILT_RAD, "70.8 deg total: the derived flag fires"
    assert roll < schema.FALL_TILT_RAD and pitch < schema.FALL_TILT_RAD, \
        "...while neither axis exceeds the per-axis limit"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
