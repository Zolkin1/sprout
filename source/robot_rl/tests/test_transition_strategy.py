# Copyright 2026 Zachary Olkin. All rights reserved.

"""Pure (no-library) tests for the pluggable skill-transition strategies.

Exercise the ``raw`` (hard cut) and ``linear`` (online-parity cross-fade)
strategies directly with a tiny stub manager, so the alpha ramp / blend math /
window-end deactivation are verified without a stride library.
"""

from __future__ import annotations

import pytest
import torch

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.clips.transition_strategy import (
    LinearInterpolationTransition,
    QuadraticInterpolationTransition,
    RawStitchTransition,
    TransitionStrategy,
    make_transition_strategy,
)

_P, _V = 4, 3
_OLD, _NEW = 1.0, 0.0  # constant old / new output values (linear blend -> 1 - alpha)


class _StubManager:
    """Returns constant ``_OLD`` outputs for the "old" stride regardless of phase."""

    def set_trajectory_indices(self, idx: torch.Tensor) -> None:
        self._m = idx.shape[0]

    def get_output(self, phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        m = phase.shape[0]
        return torch.full((m, _P), _OLD), torch.full((m, _V), _OLD)


def _linear(dt: float = 0.1, blend_end_phi: float = 0.3) -> LinearInterpolationTransition:
    """A linear strategy with unit stride time (so phi advances by ``dt`` / step)."""
    return LinearInterpolationTransition(
        manager=_StubManager(),
        total_time=torch.ones(8),          # stride duration 1 s -> dphi = dt
        ori_groups=torch.zeros((0, 4), dtype=torch.long),  # no quats -> pure linear
        dt=dt, blend_end_phi=blend_end_phi, device="cpu",
    )


def _new_outputs(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.full((n, _P), _NEW), torch.full((n, _V), _NEW)


def test_raw_is_identity_and_does_not_arm():
    raw = make_transition_strategy("raw", manager=None, total_time=None, ori_groups=None,
                                   dt=0.1, blend_end_phi=0.3, device="cpu")
    assert isinstance(raw, RawStitchTransition) and raw.arms is False
    raw.reset(2)
    raw.arm(torch.tensor([0]), torch.tensor([0]), torch.tensor([0.0]))  # no-op
    yp, yv = _new_outputs(2)
    out_p, out_v = raw.apply(yp, yv)
    assert torch.equal(out_p, yp) and torch.equal(out_v, yv)


def test_linear_alpha_ramp_and_deactivation():
    lin = _linear(dt=0.1, blend_end_phi=0.3)
    assert lin.arms is True
    lin.reset(2)
    # Only clip 0 transitions; clip 1 stays pure-new the whole time.
    lin.arm(torch.tensor([0]), torch.tensor([0]), torch.tensor([0.0]))

    # alpha = phi / 0.3, phi advances by dt=0.1 each apply -> 0, 1/3, 2/3, then off.
    expected_c0 = [1.0, 1.0 - 1.0 / 3.0, 1.0 - 2.0 / 3.0, _NEW]  # 1 - alpha (old=1,new=0)
    for step, exp in enumerate(expected_c0):
        yp, yv = _new_outputs(2)
        out_p, _ = lin.apply(yp, yv)
        assert out_p[0, 0].item() == pytest.approx(exp, abs=1e-5), f"step {step}"
        assert out_p[1, 0].item() == pytest.approx(_NEW, abs=1e-5), "non-transitioning clip changed"
    # After the window the transition is inactive.
    assert not bool(lin._active.any())


def test_quadratic_eases_in_holding_old_longer():
    """Quadratic transition blends with ``alpha = frac**2`` -> holds old longer.

    Same machinery as linear; only the easing differs.  With window 0.3 and
    dt 0.1 the fractions are 0, 1/3, 2/3, so alpha = 0, 1/9, 4/9 and the blended
    output (old=1, new=0) is ``1 - alpha`` -- and stays nearer the old stride
    than linear (whose alpha would be 0, 1/3, 2/3) at the same steps.
    """
    q = QuadraticInterpolationTransition(
        manager=_StubManager(), total_time=torch.ones(8),
        ori_groups=torch.zeros((0, 4), dtype=torch.long),
        dt=0.1, blend_end_phi=0.3, device="cpu",
    )
    assert q.name == "quadratic" and q.arms is True
    q.reset(2)
    q.arm(torch.tensor([0]), torch.tensor([0]), torch.tensor([0.0]))
    expected = [1.0, 1.0 - 1.0 / 9.0, 1.0 - 4.0 / 9.0, _NEW]  # 1 - frac**2
    for step, exp in enumerate(expected):
        yp, yv = _new_outputs(2)
        out_p, _ = q.apply(yp, yv)
        assert out_p[0, 0].item() == pytest.approx(exp, abs=1e-5), f"step {step}"
        # Holds the old (value 1) closer than linear (1 - frac) would at each step.
        assert out_p[0, 0].item() >= (1.0 - step / 3.0) - 1e-6
    assert not bool(q._active.any())  # deactivates at the window end


def test_make_transition_strategy_resolves_quadratic():
    q = make_transition_strategy(
        "quadratic", manager=_StubManager(), total_time=torch.ones(4),
        ori_groups=torch.zeros((0, 4), dtype=torch.long), dt=0.1, blend_end_phi=0.3, device="cpu",
    )
    assert isinstance(q, QuadraticInterpolationTransition)


def test_linear_pos_shift_translates_old_continuation():
    """A mid-cycle seam's ``pos_shift`` translates the old stride's position
    channels (so its stance foot re-aligns to the new landing) before blending.
    """
    pos_groups = torch.tensor([[0, 1, 2]], dtype=torch.long)  # one frame, pos_x/y/z
    lin = LinearInterpolationTransition(
        manager=_StubManager(), total_time=torch.ones(4),
        ori_groups=torch.zeros((0, 4), dtype=torch.long),
        dt=0.1, blend_end_phi=0.3, device="cpu", pos_groups=pos_groups,
    )
    lin.reset(2)
    shift = torch.tensor([[0.5, -0.2, 0.1]])
    lin.arm(torch.tensor([0]), torch.tensor([0]), torch.tensor([0.0]), pos_shift=shift)
    yp, yv = _new_outputs(2)  # _NEW = 0
    out_p, _ = lin.apply(yp, yv)
    # First apply -> phi=0 -> alpha=0 -> pure (shifted) old (old=_OLD=1.0).
    assert out_p[0, 0].item() == pytest.approx(_OLD + 0.5)
    assert out_p[0, 1].item() == pytest.approx(_OLD - 0.2)
    assert out_p[0, 2].item() == pytest.approx(_OLD + 0.1)
    assert out_p[0, 3].item() == pytest.approx(_OLD)   # col 3 is not a pos channel
    assert out_p[1, 0].item() == pytest.approx(_NEW)   # non-armed clip unchanged


def test_linear_rejects_out_of_range_window():
    # blend_end_phi is a fraction of a STEP -> must be in (0, 1].
    with pytest.raises(ValueError):
        _linear(blend_end_phi=0.0)
    with pytest.raises(ValueError):
        _linear(blend_end_phi=1.5)


def test_seg_len_makes_blend_step_relative():
    """``arm(seg_len=s)`` scales the window: ``blend_end_phi`` is a fraction of the STEP.

    With ``blend_end_phi=0.6`` and ``seg_len=0.5`` the window is 0.6*0.5 = 0.3 of
    full-period phase, so the ramp matches a full-period window of 0.3 -- i.e. the
    knob is a fraction of the (half-cycle) step, not of the whole two-stride period.
    """
    lin = _linear(dt=0.1, blend_end_phi=0.6)
    lin.reset(1)
    lin.arm(torch.tensor([0]), torch.tensor([0]), torch.tensor([0.0]),
            seg_len=torch.tensor([0.5]))
    expected = [1.0, 1.0 - 1.0 / 3.0, 1.0 - 2.0 / 3.0, _NEW]  # window 0.3 -> alpha 0,1/3,2/3,1
    for step, exp in enumerate(expected):
        yp, yv = _new_outputs(1)
        out_p, _ = lin.apply(yp, yv)
        assert out_p[0, 0].item() == pytest.approx(exp, abs=1e-5), f"step {step}"
    assert not bool(lin._active.any())


def test_factory_passthrough_and_unknown():
    custom = RawStitchTransition()
    assert make_transition_strategy(custom, manager=None, total_time=None, ori_groups=None,
                                    dt=0.1, blend_end_phi=0.3, device="cpu") is custom
    assert isinstance(custom, TransitionStrategy)
    with pytest.raises(ValueError):
        make_transition_strategy("motion_matching", manager=None, total_time=None, ori_groups=None,
                                 dt=0.1, blend_end_phi=0.3, device="cpu")
