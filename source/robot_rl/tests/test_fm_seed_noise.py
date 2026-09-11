# Copyright 2026 Zachary Olkin. All rights reserved.

"""Pre-flight 1: the noise source is addressable, shared where it must be, and distinct
where it must be. Isaac-free."""

from __future__ import annotations

import os
import sys

import pytest

torch = pytest.importorskip("torch")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for _p in (os.path.join(_ROOT, "scripts", "rsl_rl", "eval"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fm_seed_noise as fsn  # noqa: E402


def _src(seed, buf, replan_every=12, F=4, D=3):
    return fsn.FmSeedNoise(fm_seed=seed, future_len=F, state_dim=D, device="cpu",
                           replan_every=replan_every, episode_len_buf=lambda: buf)


def test_envs_at_the_same_point_of_their_episode_share_one_draw():
    """THE floor property. Several robots on one clip under one seed must be handed the
    identical noise, or their spread contains the effect it is meant to bound."""
    buf = torch.tensor([24, 24, 24, 24])
    out = _src(7, buf)(torch.arange(4))
    for i in range(1, 4):
        assert torch.equal(out[0], out[i])


def test_envs_at_different_points_get_different_draws():
    buf = torch.tensor([0, 12, 24])
    out = _src(7, buf)(torch.arange(3))
    assert not torch.equal(out[0], out[1])
    assert not torch.equal(out[1], out[2])


def test_the_index_is_the_replan_count_not_the_step():
    """Steps 24..35 are all the SAME re-plan, so they must draw the same noise."""
    a = _src(7, torch.tensor([24]))(torch.tensor([0]))
    b = _src(7, torch.tensor([35]))(torch.tensor([0]))
    c = _src(7, torch.tensor([36]))(torch.tensor([0]))
    assert torch.equal(a[0], b[0])
    assert not torch.equal(a[0], c[0])


def test_different_seeds_give_different_noise():
    """If this failed the whole experiment would be measuring nothing, quietly."""
    buf = torch.tensor([24])
    assert not torch.equal(_src(1, buf)(torch.tensor([0]))[0],
                           _src(2, buf)(torch.tensor([0]))[0])


def test_one_seed_reproduces_across_instances():
    """Re-run determinism at the source level: a fresh process must draw the same stream."""
    buf = torch.tensor([0, 12, 24])
    assert torch.equal(_src(9, buf)(torch.arange(3)), _src(9, buf)(torch.arange(3)))


def test_an_env_that_resets_starts_its_stream_again():
    """Derived from episode_length_buf, so a reset returns to index 0 with no bookkeeping."""
    s = _src(3, torch.tensor([48]))
    late = s(torch.tensor([0]))[0]
    s._buf = lambda: torch.tensor([0])
    assert torch.equal(s(torch.tensor([0]))[0], s.draw(0))
    assert not torch.equal(late, s.draw(0))


def test_the_draw_is_an_ordinary_gaussian():
    """It must be a normal sample, not a fixed or zeroed vector -- fm_zero_noise is a
    different, off-distribution thing."""
    s = fsn.FmSeedNoise(fm_seed=0, future_len=64, state_dim=44, device="cpu",
                        replan_every=12, episode_len_buf=lambda: torch.tensor([0]))
    x = torch.stack([s.draw(i) for i in range(40)])
    assert abs(float(x.mean())) < 0.02 and abs(float(x.std()) - 1.0) < 0.02


def test_ids_are_honoured_in_order():
    """The command uses the returned rows positionally against the ids it passed."""
    buf = torch.tensor([0, 12, 24, 36])
    s = _src(5, buf)
    out = s(torch.tensor([3, 1]))
    assert torch.equal(out[0], s.draw(3)) and torch.equal(out[1], s.draw(1))
