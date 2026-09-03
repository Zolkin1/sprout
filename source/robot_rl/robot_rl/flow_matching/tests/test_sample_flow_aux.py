# Copyright 2026 Zachary Olkin. All rights reserved.

"""``return_aux`` -- handing out the final Euler step's velocity field.

The RL adjuster's query tokens carry two things per node: the window the generator predicted,
and the flow velocity still acting there at the last integration step. The second is the ``v``
the sampler already computed, so exposing it costs nothing -- and the property that has to
hold is exactly that: **turning it on must not change the sample.** An aux output that
perturbed the plan would make every adjuster measurement incomparable with the bare
generator's, quietly.

The other thing pinned here is the SPACE. ``sample_window`` denormalizes the window and
Gram-Schmidts its rotation block; the field gets neither, and both omissions are deliberate --
it is a rate, so the normalizer's offset does not apply to it, and it is a velocity in the
ambient 9-vector space rather than a rotation, so projecting it onto SO(3) would be a category
error rather than a refinement.
"""

from __future__ import annotations

import pytest
import torch

from robot_rl.flow_matching.flow import sample_trajectory
from robot_rl.flow_matching.model import FlowMatchingTransformer

F, D, N, NUM_STEPS = 6, 12, 2, 4
SENSOR_DIM, SENSORS = 8, 1


def _model(attention_mode: str = "prefix") -> FlowMatchingTransformer:
    """A tiny transformer of the shipped shape, small enough to run on a CPU."""
    torch.manual_seed(0)
    return FlowMatchingTransformer(
        state_dim=D, future_len=F, num_history_states=N,
        extero_sensor_dim=SENSOR_DIM, extero_sensor_count=SENSORS,
        cmd_dim=3, cmd_stride=3, d_model=32, nhead=4, num_layers=2, dim_feedforward=64,
        extero_encoder="mlp", attention_mode=attention_mode,
    ).eval()


def _conditioning(batch: int = 3) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(1)
    return {
        "history": torch.randn(batch, N, D, generator=gen),
        "extero": torch.randn(batch, SENSORS * SENSOR_DIM, generator=gen),
        "cmd_tokens": torch.randn(batch, len(range(0, F, 3)), 3, generator=gen),
    }


def _sample(model, *, return_aux: bool, guidance_scale: float = 1.0):
    """One sample from a fixed noise draw, so two calls are comparable."""
    cond = _conditioning()
    noise = torch.randn(3, F, D, generator=torch.Generator().manual_seed(2))
    return sample_trajectory(
        model, cond["history"], cond["extero"], cond["cmd_tokens"], F, D, NUM_STEPS,
        noise=noise, guidance_scale=guidance_scale, return_aux=return_aux,
    )


@pytest.mark.parametrize("attention_mode", ["prefix", "full"])
def test_asking_for_the_field_does_not_change_the_sample(attention_mode: str) -> None:
    """Bit-identical, on both attention paths."""
    model = _model(attention_mode)
    plain = _sample(model, return_aux=False)
    window, field = _sample(model, return_aux=True)

    assert torch.equal(window, plain), "the aux output perturbed the sample"
    assert field.shape == window.shape
    assert torch.isfinite(field).all()


def test_the_field_carried_the_sample_over_its_final_step() -> None:
    """Exactly, using a one-step run where the state before the step is the noise itself.

    At ``num_steps = 1`` the integration is the single update ``x = x0 + 1.0 * v``, and ``x0``
    is the noise the caller passed -- so ``v`` is pinned to ``x_final - noise`` with nothing
    reconstructed and nothing assumed. Reconstructing ``x_before`` from the returned field
    instead would be a tautology: it would hold for any field whatsoever.
    """
    model = _model()
    cond = _conditioning()
    noise = torch.randn(3, F, D, generator=torch.Generator().manual_seed(2))

    final, field = sample_trajectory(
        model, cond["history"], cond["extero"], cond["cmd_tokens"], F, D, 1,
        noise=noise, return_aux=True,
    )
    torch.testing.assert_close(field, final - noise)
    assert not torch.allclose(field, torch.zeros_like(field))


def test_the_field_is_the_LAST_step_not_the_first() -> None:
    """A multi-step run's field is evaluated late in the flow, not at ``tau = 0``.

    The two are the same shape and both finite, so nothing structural separates them. What
    does is that the late field is evaluated at a different flow time AND from a partly
    integrated state, so it cannot coincide with the first step's.
    """
    model = _model()
    cond = _conditioning()
    noise = torch.randn(3, F, D, generator=torch.Generator().manual_seed(2))

    args = (model, cond["history"], cond["extero"], cond["cmd_tokens"], F, D)
    _, first_step_field = sample_trajectory(*args, 1, noise=noise, return_aux=True)
    _, last_step_field = sample_trajectory(*args, NUM_STEPS, noise=noise, return_aux=True)

    assert not torch.allclose(last_step_field, first_step_field), (
        "the multi-step run returned the tau=0 field, so the aux output is the FIRST step's"
    )


def test_the_field_is_coherent_under_guidance() -> None:
    """Under CFG the field is the COMBINED one that was integrated, so no arm is refused.

    Guidance changes both the sample and the field; what matters is that the pair still
    describes one integration, which the reconstruction above checks. This asserts the arm
    runs at all and moves the answer -- an early ``raise`` here would have been a defensible
    design and is deliberately not the one chosen.
    """
    model = _model()
    plain_w, plain_v = _sample(model, return_aux=True, guidance_scale=1.0)
    guided_w, guided_v = _sample(model, return_aux=True, guidance_scale=2.0)

    assert torch.isfinite(guided_v).all()
    assert not torch.allclose(guided_w, plain_w), "guidance did not reach the sample"
    assert not torch.allclose(guided_v, plain_v), "guidance did not reach the field"
