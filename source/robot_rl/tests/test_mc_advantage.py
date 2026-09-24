# Copyright 2026 Zachary Olkin. All rights reserved.

"""The Monte-Carlo advantage, ``A = G - V``.

It used to be a blend: ``A = w_track * (G - G_hat) + w_fail * (P(fail) - 1{failed})``, with a
survival half read off the critic's second head. That head is gone -- the fine-tune dropped its
loss, so no reported result used it -- and with it the blend, the ``w_fail`` knob and the
n-step alternatives that bootstrapped on the critic's own prediction at ``t + n``.

What remains is the estimator the loop always ran: difference the REALIZED return against the
critic's out-of-fold baseline, no bootstrap, no ``n`` to tune.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_rl.flow_matching import advantage as adv


def test_it_is_exactly_G_minus_V():
    """The whole estimator, so there is nothing to get subtly wrong."""
    g = np.array([0.9, 0.4, 0.55])
    v = np.array([0.5, 0.5, 0.55])
    a, ok = adv.mc_advantage(g, v)
    assert np.allclose(a, [0.4, -0.1, 0.0])
    assert ok.all()


def test_a_perfectly_calibrated_critic_scores_zero():
    """The property that makes the advantage a SELECTION signal rather than a level.

    If V predicts G exactly, every chunk is average and nothing is preferred -- which is what
    makes a non-zero advantage mean "this chunk beat the baseline" rather than "this state was
    easy".
    """
    g = np.linspace(0.0, 1.0, 16)
    a, ok = adv.mc_advantage(g, g.copy())
    assert np.allclose(a, 0.0)
    assert ok.all()


def test_a_row_with_no_measured_return_is_UNLABELLED_not_zero():
    """``valid=False`` must not read as "average".

    A zero advantage is a real, meaningful value -- it says the chunk matched the baseline --
    so a row without a return has to be excluded by the MASK rather than given a zero. Under
    the old blend such a row survived on its survival half; there is no half left to survive
    on, so it drops out entirely and the caller's filter is what respects that.
    """
    g = np.array([0.8, 0.0, 0.3])
    v = np.array([0.5, 0.5, 0.5])
    valid = np.array([True, False, True])
    a, ok = adv.mc_advantage(g, v, valid=valid)
    assert ok.tolist() == [True, False, True]
    assert a[1] == 0.0, "the value is zeroed, but ok[1] is what says it means nothing"
    assert np.allclose(a[[0, 2]], [0.3, -0.2])


def test_it_ignores_the_episode_layout_entirely():
    """No bootstrap means no dependence on order, neighbours or episode boundaries.

    This is the estimator's one structural advantage over the n-step form that was deleted:
    a row's label depends on that row alone, so a shuffled table gives shuffled labels rather
    than different ones.
    """
    rng = np.random.default_rng(0)
    g, v = rng.uniform(size=32), rng.uniform(size=32)
    a, _ = adv.mc_advantage(g, v)
    perm = rng.permutation(32)
    a_perm, _ = adv.mc_advantage(g[perm], v[perm])
    assert np.allclose(a_perm, a[perm])


def test_ragged_inputs_are_refused():
    """Lengths that disagree are a join error upstream; guessing which is right is worse."""
    with pytest.raises(ValueError, match="agree in length"):
        adv.mc_advantage(np.zeros(4), np.zeros(5))


def test_the_labeller_offers_only_the_mc_estimator():
    """The n-step choice is gone from the CLI, not merely unused.

    Leaving ``--estimator n_step`` selectable would let a run ask for an estimator whose
    implementation was deleted -- an argparse-accepted flag that fails somewhere later.
    """
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[3] / "scripts" / "flow_matching"
           / "label_advantage.py").read_text()
    assert 'choices=("mc",)' in src
    assert "adv_mod.mc_advantage(" in src
    # USES, not mentions: the file explains in prose why these are gone, and that prose
    # must not fail the test that protects it.
    assert "args.w_fail" not in src, "the survival weight is back"
    assert "adv_mod.n_step_advantage" not in src, "a deleted estimator is called"
    assert "adv_mod.combined_n_step_advantage" not in src, "a deleted estimator is called"
