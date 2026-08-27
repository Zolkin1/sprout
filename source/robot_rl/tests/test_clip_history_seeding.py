# Copyright 2026 Zachary Olkin. All rights reserved.

"""Teacher forcing seeds the generator's history from the clip; the closed loop does not.

A null-history inference is phase-agnostic by construction. Under ``history_from="clip"`` the
robot spawns at a definite gait phase and the reward follows it, so an unseeded opening plan
disagrees with both for a whole re-plan period -- every episode. Seeding removes that.

The closed loop must KEEP the null embedding: on a robot there is no clip to seed from, so the
two arms are deliberately allowed to differ here.
"""

from __future__ import annotations

import pytest
import torch

from robot_rl.flow_matching.online import BatchedHistory


def test_seed_installs_rows_and_clears_the_mask():
    """Seeded envs read back their rows and are valid immediately; others stay masked."""
    n_env, maxlen, dim = 6, 4, 3
    hist = BatchedHistory(n_env, maxlen, torch.device("cpu"))
    ids = torch.tensor([1, 4])
    rows = torch.arange(2 * maxlen * dim, dtype=torch.float32).reshape(2, maxlen, dim)

    hist.seed(ids, {"node": rows})

    assert not bool(hist.null_mask()[ids].any())          # seeded -> usable now
    assert bool(hist.null_mask()[[0, 2, 3, 5]].all())     # everyone else untouched
    torch.testing.assert_close(hist.stack("node")[ids], rows)


def test_seed_pads_an_underfilled_deque_to_num_envs_not_maxlen():
    """The padding slot is ``(num_envs, ...)``.

    Sizing it from the row block instead gives ``(maxlen, ...)``, which indexes out of bounds
    the moment an env id exceeds ``maxlen`` -- as a CUDA device-side assert far from the
    cause. Regression: this was the first version of the code.
    """
    n_env, maxlen, dim = 16, 4, 3
    hist = BatchedHistory(n_env, maxlen, torch.device("cpu"))
    ids = torch.tensor([15])                              # > maxlen on purpose
    hist.seed(ids, {"node": torch.ones(1, maxlen, dim)})

    assert hist.stack("node").shape == (n_env, maxlen, dim)
    torch.testing.assert_close(hist.stack("node")[15], torch.ones(maxlen, dim))


def test_seed_rejects_a_wrong_length_block():
    """A short block would leave stale rows in the ring while the mask says they are real."""
    hist = BatchedHistory(4, 4, torch.device("cpu"))
    with pytest.raises(ValueError, match="expected maxlen"):
        hist.seed(torch.tensor([0]), {"node": torch.zeros(1, 2, 3)})


def test_seed_of_nothing_is_a_no_op():
    """No env reset this step is the common case and must not disturb the ring."""
    hist = BatchedHistory(4, 4, torch.device("cpu"))
    hist.seed(torch.zeros(0, dtype=torch.long), {"node": torch.zeros(0, 4, 3)})
    assert bool(hist.null_mask().all())


class _FakeCmd:
    """The seeding-mode properties over stand-ins for ctx / cfg / clip state."""

    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.fm_tracking_cmd import (  # noqa: E501
        FmTrackingCommand as _C,
    )
    min_start_step = _C.min_start_step
    _seeding_enabled = _C._seeding_enabled
    _envs_needing_seed = _C._envs_needing_seed


def _cmd(mode: str, history_from: str = "generator"):
    """A fake command with the given seeding mode."""
    pytest.importorskip("isaaclab")
    c = _FakeCmd()
    c.ctx = type("Ctx", (), {"N": 4})()
    c.cfg = type("Cfg", (), {"seed_history_from_clip": mode, "history_from": history_from})()
    return c


@pytest.mark.parametrize(
    "mode,history_from,enabled",
    [("never", "clip", False), ("never", "generator", False),
     ("always", "generator", True), ("on_terrain_start", "generator", True),
     ("auto", "clip", True), ("auto", "generator", False)],
)
def test_seeding_mode_resolution(mode, history_from, enabled):
    """``auto`` tracks teacher forcing; everything else is explicit.

    ``auto`` exists so the teacher-forced arm keeps its seed without every task having to
    restate it -- that arm cannot hold phase on its first inference without one.
    """
    assert _cmd(mode, history_from)._seeding_enabled is enabled


def test_an_unknown_seeding_mode_raises_rather_than_disabling_the_seed():
    """A typo must not read as "off": that failure is invisible in every shape."""
    with pytest.raises(ValueError, match="seed_history_from_clip"):
        _cmd("on_terain_start")._seeding_enabled


def test_min_start_step_reserves_exactly_the_history_length():
    """The reserve is what makes the seed safe, so it tracks N and nothing else.

    Without it a start inside the first N steps indexes backwards -- the clip-step gather
    clamps only at the TOP -- and reads the clip's END as its history, silently.
    """
    assert _cmd("always")._seeding_enabled and _cmd("always").min_start_step == 4
    assert _cmd("never").min_start_step == 0
    assert _cmd("auto", "clip").min_start_step == 4
    assert _cmd("auto", "generator").min_start_step == 0


def test_on_terrain_start_seeds_only_the_envs_with_no_flat_lead_in():
    """Flat clips (``terrain_start_step = -1``) keep the null embedding, as deploy does.

    Seeding those too would train a policy that never sees the opening it actually gets on a
    robot; seeding none would mean dropping every clip whose terrain begins at step 0.
    """
    torch = pytest.importorskip("torch")
    c = _cmd("on_terrain_start")
    c.device = "cpu"
    c.clip_id = torch.arange(4)
    c._step = torch.tensor([10, 10, 4, 10])       # start step per env
    #                        flat  pre  on   past
    t_start = torch.tensor([-1, 50, 0, 5])
    c.manager = type("M", (), {"gather_terrain_start_step": staticmethod(lambda cid: t_start[cid])})()

    got = c._envs_needing_seed(torch.arange(4)).tolist()
    assert got == [2, 3]        # env0 flat, env1 starts before its terrain -> both left null


def test_other_modes_do_not_narrow_the_env_set():
    """``always`` means always; the terrain test belongs to one mode only."""
    torch = pytest.importorskip("torch")
    c = _cmd("always")
    ids = torch.arange(3)
    assert c._envs_needing_seed(ids).tolist() == [0, 1, 2]
