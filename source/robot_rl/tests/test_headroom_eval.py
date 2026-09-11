# Copyright 2026 Zachary Olkin. All rights reserved.

"""The seams of the best-of-K headroom harness that a wrong answer would survive.

The rollout itself needs a simulator and is not tested here.  What is tested is every place
where a mistake produces a plausible number rather than an error:

1. **The state walk.**  It missed the generator's history ring on the first run -- the ring
   is a ``deque`` of ``dict`` of tensors hanging off a plain object, and the walk followed
   only tensors and containers, so the attribute was a dead end.  Every branch would have
   been conditioned on its own stale past while every shape agreed.  Pinned by shape here,
   and by ``REQUIRED_STATE`` at runtime.
2. **Common random numbers.**  The whole measurement rests on the K branches differing in
   exactly one draw.  Three properties are asserted: a continuation draw is shared by a
   group, candidates differ from each other, and the parent shares candidate 0's seed.
3. **Best-of-K arithmetic.**  Computed in closed form; checked against brute-force
   enumeration of every k-subset.
4. **Row order.**  The reduce reshapes the flat table into ``[n_states, K+1]`` and relies on
   the candidate-then-parent order; a silent transposition there would mix candidates with
   parents in every table.
5. **The commit rule.**  A candidate that survived must never lose to one that fell.
"""

from __future__ import annotations

import collections
import itertools
import pathlib
import sys

import numpy as np
import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[3]
_EVAL = REPO / "scripts" / "rsl_rl" / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

import _paths  # noqa: F401,E402
import headroom_noise  # noqa: E402
import headroom_report  # noqa: E402
import headroom_schema as hs  # noqa: E402
import headroom_state  # noqa: E402


# ---------------------------------------------------------------------------
# 1. The state walk.
# ---------------------------------------------------------------------------
class _Ring:
    """Stand-in for ``BatchedHistory``: a deque of dicts of per-env tensors."""

    def __init__(self, n: int) -> None:
        self._steps = collections.deque(
            [{"node": torch.zeros(n, 3), "extra": torch.zeros(n, 2)} for _ in range(4)])
        self._steps_since_reset = torch.zeros(n, dtype=torch.long)


def _walk(obj, path, n):
    """Run the module's walk and return the inventory paths it found."""
    out: list = []
    headroom_state._walk(obj, path, n, headroom_state._MAX_DEPTH, out, set())
    return sorted(s.path for s in out)


def test_the_walk_reaches_a_deque_of_dicts_of_tensors():
    """The generator's history ring shape. This is the one the first run silently missed."""
    found = _walk(_Ring(8), "cmd._history", 8)
    assert "cmd._history._steps_since_reset" in found
    assert "cmd._history._steps[0]['node']" in found
    assert "cmd._history._steps[3]['extra']" in found


def test_the_walk_does_not_recurse_into_plain_objects():
    """Why the history ring has to be a NAMED owner rather than found through ``cmd``."""

    class Outer:
        def __init__(self):
            self.ring = _Ring(8)
            self.direct = torch.zeros(8, 3)

    found = _walk(Outer(), "cmd", 8)
    assert found == ["cmd.direct"], (
        "the walk followed a plain object; if that changes, the explicit ('cmd._history') "
        "owner is redundant but harmless -- update this test rather than deleting it."
    )


def test_the_walk_skips_tensors_whose_leading_dim_is_not_num_envs():
    """Shape is the only test, so a shared constant must not be the wrong length."""

    class Owner:
        def __init__(self):
            self.per_env = torch.zeros(8, 3)
            self.shared = torch.zeros(1, 3)
            self.scalarish = torch.zeros(3)

    assert _walk(Owner(), "o", 8) == ["o.per_env"]


def test_the_walk_lists_a_shared_buffer_once():
    """Two attributes pointing at one tensor must not be copied twice under two names."""

    class Owner:
        def __init__(self):
            self.a = torch.zeros(8, 3)
            self.b = self.a

    assert len(_walk(Owner(), "o", 8)) == 1


def test_skip_attrs_are_honoured():
    """The clip library is enormous and shared; copying it would be a slow no-op at best."""

    class Owner:
        def __init__(self):
            self._clip_lib = torch.zeros(8, 3)
            self.keep = torch.zeros(8, 3)

    assert _walk(Owner(), "cmd", 8) == ["cmd.keep"]


def test_required_state_names_only_owners_the_mirror_actually_walks():
    """A required path under an owner that is never visited could never be satisfied."""
    import headroom_runner  # noqa: PLC0415

    prefixes = ("cmd.", "cmd._history.", "actions.", "env.episode_length_buf")
    for path in headroom_runner.REQUIRED_STATE:
        assert path.startswith(prefixes), f"{path} is under no walked owner"


# ---------------------------------------------------------------------------
# 2. Common random numbers.
# ---------------------------------------------------------------------------
def _crn(k=4, groups=2, seed=11):
    """A small CRN instance: ``groups`` groups of ``1 parent + k`` envs."""
    return headroom_noise.CrnNoise(num_envs=groups * (k + 1), group_size=k + 1,
                                   future_len=3, state_dim=2, device="cpu", run_seed=seed)


def test_a_continuation_draw_is_shared_by_a_group_and_differs_across_groups():
    """The property the whole ranking rests on: only the branched chunk may differ."""
    c = _crn()
    c.arm_continuation(2)
    out = c(torch.arange(10))
    for env in range(1, 5):
        assert torch.equal(out[0], out[env]), "group 0 members got different continuations"
    assert not torch.equal(out[0], out[5]), "two groups share a continuation draw"


def test_the_parent_shares_candidate_zero_s_branch_draw():
    """Candidate 0 must be the parent's own chunk, or the noise floor measures nothing."""
    c = _crn()
    c.arm_branch(0)
    out = c(torch.arange(10))
    assert torch.equal(out[0], out[1]), "parent and candidate 0 drew different chunks"
    assert not torch.equal(out[1], out[2]), "two candidates drew the same chunk"


def test_seeds_are_derived_not_counted():
    """A rerun, or a resumed run, must draw the same numbers regardless of call order."""
    a, b = _crn(), _crn()
    a.arm_branch(0)
    a(torch.arange(10))          # burn a call on one instance only
    a.arm_continuation(1)
    b.arm_continuation(1)
    assert torch.equal(a(torch.arange(10)), b(torch.arange(10)))


def test_a_partial_group_is_refused():
    """A partial group would leave some candidate index with no env, silently."""
    with pytest.raises(ValueError, match="not a multiple"):
        headroom_noise.CrnNoise(num_envs=10, group_size=4, future_len=2, state_dim=2,
                                device="cpu", run_seed=0)


def test_the_draw_is_an_ordinary_gaussian():
    """Not ``fm_zero_noise``: only WHICH sample is controlled, not its distribution."""
    c = headroom_noise.CrnNoise(num_envs=2, group_size=2, future_len=64, state_dim=64,
                                device="cpu", run_seed=3)
    c.arm_branch(0)
    x = c(torch.arange(2))[0]
    assert abs(float(x.mean())) < 0.05
    assert 0.9 < float(x.std()) < 1.1


# ---------------------------------------------------------------------------
# 3. Best-of-K arithmetic.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m", list(range(6)))
def test_best_of_k_matches_brute_force_enumeration(m):
    """Closed form against every k-subset, for each possible survivor count."""
    big_k = 5
    surv = np.zeros((1, big_k), dtype=np.int64)
    surv[0, :m] = 1
    got = headroom_report.best_of_k(surv, k_grid=(1, 2, 4))
    for k, val in got.items():
        subsets = list(itertools.combinations(range(big_k), k))
        want = np.mean([any(surv[0, i] for i in s) for s in subsets])
        assert val[0] == pytest.approx(want), f"k={k}, m={m}"


def test_best_of_k_drops_k_above_the_candidate_count():
    """Asking for K=16 from an 8-candidate run must omit it, not extrapolate."""
    got = headroom_report.best_of_k(np.ones((3, 8), dtype=np.int64))
    assert max(got) == 8


# ---------------------------------------------------------------------------
# 4. Row order and the record schema.
# ---------------------------------------------------------------------------
def test_interleave_puts_the_parent_last_in_every_state():
    """The order the whole reduce reshapes by."""
    import headroom_runner  # noqa: PLC0415

    cand = np.array([10, 11, 20, 21], dtype=np.float64)     # 2 states x k=2
    parent = np.array([19, 29], dtype=np.float64)
    out = headroom_runner._interleave(cand, parent, n_states=2, k=2)
    assert out.tolist() == [10, 11, 19, 20, 21, 29]


def test_candidate_grid_refuses_a_row_order_it_does_not_recognise():
    """A ragged or reordered file must fail loudly, not reshape into a plausible grid."""
    k = 2
    d = {"candidate_idx": np.array([0, 1, hs.PARENT_ROW, 1, 0, hs.PARENT_ROW],
                                   dtype=np.uint8),
         "survived": np.arange(6)}
    with pytest.raises(SystemExit, match="candidate_idx"):
        headroom_report.candidate_grid(d, "survived", k)


def test_record_writer_round_trip(tmp_path):
    """Every column written is read back, and the metadata survives."""
    path = tmp_path / "states.h5"
    w = hs.RecordWriter(str(path), meta={"k": 3, "note": "unit"})
    rows = {name: np.arange(4, dtype=np.dtype(dt)) for name, dt in hs.COLUMNS}
    w.append(rows)
    w.append(rows)
    w.close()
    back = hs.read_records(str(path))
    assert back["meta"]["k"] == 3
    assert back["state_id"].shape == (8,)
    assert back["candidate_idx"].tolist() == [0, 1, 2, 3] * 2


def test_record_writer_names_a_missing_column(tmp_path):
    """A zero-fill here would read downstream as a measurement."""
    w = hs.RecordWriter(str(tmp_path / "s.h5"), meta={})
    rows = {name: np.zeros(2, dtype=np.dtype(dt)) for name, dt in hs.COLUMNS}
    rows.pop("err_base_pos")
    with pytest.raises(KeyError, match="err_base_pos"):
        w.append(rows)
    w.close()


def test_reading_a_foreign_schema_version_is_refused(tmp_path):
    """Columns are read by NAME, but a version bump means one of them changed meaning."""
    import h5py  # noqa: PLC0415

    path = tmp_path / "s.h5"
    w = hs.RecordWriter(str(path), meta={})
    w.close()
    with h5py.File(path, "a") as f:
        f.attrs["schema_version"] = hs.SCHEMA_VERSION + 1
    with pytest.raises(SystemExit, match="schema_version"):
        hs.read_records(str(path))


# ---------------------------------------------------------------------------
# 5. Phase labelling and the commit rule.
# ---------------------------------------------------------------------------
def test_phase_code_labels_each_region_and_the_flat_clip():
    """A flat clip has no span at all, and must not fall through to 'exit'."""
    code = hs.phase_code(
        clip_step=[10, 100, 150, 250, 400, 42],
        terrain_start_step=[100, 100, 100, 100, 100, -1],
        terrain_end_step=[300, 300, 300, 300, 300, -1])
    names = [hs.PHASES[c] for c in code]
    assert names == ["approach", "early_feature", "early_feature", "mid_feature", "exit",
                     "no_terrain"]


def test_commit_score_never_ranks_a_faller_above_a_survivor():
    """Falling ends the episode; no tracking error can be worth that."""
    import headroom_score  # noqa: PLC0415

    cols = {
        "survived": np.array([1, 0]),
        "window_steps": np.array([60, 60]),
        "steps_before_term": np.array([60, 59]),
        # The survivor tracks appallingly, the faller perfectly.
        **{c: np.array([10.0, 0.0]) for c in headroom_score.COMMIT_SCALES},
    }
    score = headroom_score.commit_score(cols)
    assert score[0] > score[1]


def test_tracking_cost_ignores_a_channel_the_generator_never_predicts():
    """``ee_pos`` is NaN for every row on this checkpoint; it must not poison the sum."""
    import headroom_score  # noqa: PLC0415

    cols = {"err_base_pos": np.array([0.1, 0.2]), "err_ee_pos": np.array([np.nan, np.nan])}
    cost = headroom_score.tracking_cost(cols)
    assert np.isfinite(cost).all()
    assert cost[1] > cost[0]


# ---------------------------------------------------------------------------
# 6. Job allocation.
# ---------------------------------------------------------------------------
def test_every_clip_gets_its_quota_and_the_column_follows_the_clip():
    """Over-assignment is invisible in any aggregate; a stale column is refused by the reset."""
    import headroom_runner  # noqa: PLC0415

    clips = [{"column": i % 3} for i in range(6)]
    alloc = headroom_runner.GroupAllocator(clips, episodes_per_clip=2, num_groups=2, seed=1)
    seen: dict[int, int] = {}
    for g in range(2):
        seen[int(alloc.group_clip[g])] = seen.get(int(alloc.group_clip[g]), 0) + 1
    while not alloc.drained:
        for g in range(2):
            c = alloc.advance(g)
            seen[c] = seen.get(c, 0) + 1
    assert all(v <= 3 for v in seen.values()), "a clip was over-assigned"
    assert set(seen) == set(range(6)), "a clip was never assigned"
    for g in range(2):
        clip = int(alloc.group_clip[g])
        assert alloc.column_of(clip) == clips[clip]["column"]


def test_override_args_supplies_every_field_the_eval_override_reads():
    """``override_args``' docstring promises completeness; this is what enforces it.

    ``_apply_env_overrides`` belongs to the EPISODE harness and grows fields on its own
    schedule.  When it gained ``fall_min_height`` this namespace did not, and the probe died
    with an ``AttributeError`` raised deep inside the override -- a failure that names the
    attribute but not the reason, and that no unit test could see because the function is
    only reachable with IsaacLab imported.

    A source scan rather than a call, for that same reason.  It is a heuristic: a field read
    through ``getattr(args, ...)`` is invisible to it, exactly as with
    ``test_stage_grid_forwards_every_flag_the_cell_stage_reads``.
    """
    import inspect
    import re

    import headroom_runner
    import runner as eval_runner

    src = inspect.getsource(eval_runner._apply_env_overrides)
    needed = set(re.findall(r"\bargs\.([a-z_][a-z0-9_]*)", src))
    supplied = set(re.findall(r"\b([a-z_][a-z0-9_]*)=",
                              inspect.getsource(headroom_runner.override_args)))
    missing = sorted(needed - supplied)
    assert not missing, (
        f"override_args does not supply {missing}, which _apply_env_overrides reads. Add "
        "each to the namespace AND to headroom.py's parser with the same default evaluate.py "
        "uses, or the two harnesses stop being comparable."
    )
