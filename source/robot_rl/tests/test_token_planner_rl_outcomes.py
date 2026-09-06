# Copyright 2026 Zachary Olkin. All rights reserved.

"""How an ending is bucketed -- the convention, and the two ways it was got wrong.

The bug this pins: `--validate_only` derived success as a RESIDUAL
(``done - fell - timeout``) while subtracting only ``tracking_violation``. When the task grew
a second fall term (``base_fallen``), every global-fall ending fell through the subtraction
and was reported as a completed traverse. The training loop had the mirror defect -- an
arrival that also fell was counted into BOTH the goal and the fall bucket, so the two rates
summed past 1.0.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from robot_rl.token_planner.rl_outcomes import EndingCounts  # noqa: E402


def _flags(n: int, **keys: list[int]) -> dict[str, torch.Tensor]:
    """Build an `ended_by`-shaped dict from per-key index lists.

    Args:
        n: Number of envs.
        **keys: Termination key -> env indices that tripped it.

    Returns:
        Key -> ``[n]`` bool tensor.
    """
    out = {}
    for k, idx in keys.items():
        t = torch.zeros(n, dtype=torch.bool)
        t[idx] = True
        out[k] = t
    return out


def test_a_global_fall_is_not_a_success() -> None:
    """The regression: an ending on `base_fallen` ALONE must be a fall, not a residual success."""
    n = 4
    done = torch.tensor([True, True, True, True])
    trunc = torch.tensor([False, False, False, False])
    by = _flags(n, base_fallen=[0, 1], tracking_violation=[2], time_out=[3])
    c = EndingCounts()
    c.update(done, trunc, by)
    assert c.episodes == 4
    assert c.fell == 3, "both fall terms count as falls"
    assert c.success == 0, "no episode reached the goal, so none can be a success"
    assert c.timeout == 1
    assert c.unattributed == 0


def test_arrival_wins_over_a_simultaneous_fall() -> None:
    """Reaching the goal and falling on the same step is a SUCCESS, counted once."""
    n = 3
    done = torch.tensor([True, True, True])
    trunc = torch.tensor([True, True, False])
    by = _flags(n, base_fallen=[1, 2])
    c = EndingCounts()
    c.update(done, trunc, by)
    assert c.success == 2
    assert c.goal_and_fall == 1
    assert c.fell == 1, "only the non-arriving fall is a fall"
    # The buckets partition the endings -- they do not overlap, unlike `by_key`.
    assert c.success + c.fell + c.timeout + c.unattributed == c.episodes


def test_the_buckets_always_partition_the_endings() -> None:
    """Whatever the flags say, the four buckets sum to the episode count."""
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        n = 16
        done = torch.rand(n, generator=g) < 0.5
        trunc = (torch.rand(n, generator=g) < 0.3) & done
        by = {k: torch.rand(n, generator=g) < 0.3
              for k in ("tracking_violation", "base_fallen", "time_out", "clip_end")}
        c = EndingCounts()
        c.update(done, trunc, by)
        assert c.success + c.fell + c.timeout + c.unattributed == c.episodes


def test_an_unclassified_termination_key_is_reported_not_absorbed() -> None:
    """A key nobody buckets lands in `unattributed`, never silently in fell/timeout."""
    n = 2
    done = torch.tensor([True, True])
    trunc = torch.tensor([False, False])
    by = _flags(n, some_new_term=[0, 1])
    c = EndingCounts()
    c.update(done, trunc, by)
    assert c.unattributed == 2
    assert c.fell == 0 and c.timeout == 0 and c.success == 0
    assert c.by_key["some_new_term"] == 2


def test_only_ended_envs_are_counted() -> None:
    """Flags on envs that did not end this step contribute nothing."""
    n = 4
    done = torch.tensor([True, False, False, False])
    trunc = torch.tensor([False, True, False, False])
    by = _flags(n, base_fallen=[0, 1, 2, 3])
    c = EndingCounts()
    c.update(done, trunc, by)
    assert c.episodes == 1
    assert c.fell == 1
    assert c.by_key["base_fallen"] == 1, "the other three envs are still running"


def test_no_endings_is_a_no_op() -> None:
    """A step where nothing ended must not move any counter."""
    c = EndingCounts()
    c.update(torch.zeros(8, dtype=torch.bool), torch.zeros(8, dtype=torch.bool),
             _flags(8, base_fallen=[0, 1]))
    assert c.episodes == 0 and c.by_key == {}
    assert c.rates()["success"] == 0.0, "rates must not divide by zero"


def test_a_fall_key_beats_a_timeout_key_on_the_same_step() -> None:
    """`time_out` fires on the step budget; a fall on that step is still a fall."""
    n = 1
    done = torch.tensor([True])
    trunc = torch.tensor([False])
    by = _flags(n, base_fallen=[0], time_out=[0])
    c = EndingCounts()
    c.update(done, trunc, by)
    assert c.fell == 1 and c.timeout == 0


def test_an_env_with_no_ended_by_map_is_unattributed_not_a_success() -> None:
    """With no termination flags at all, an ending cannot be scored as a traverse."""
    done = torch.tensor([True, True])
    trunc = torch.tensor([False, False])
    for by in (None, {}):
        c = EndingCounts()
        c.update(done, trunc, by)
        assert c.success == 0 and c.unattributed == 2
