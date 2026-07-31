# Copyright 2026 Zachary Olkin. All rights reserved.

"""The reference-encoder actor, exercised on plain tensors (no simulator).

:class:`~robot_rl.rl.ref_encoder_model.RefEncoderMLPModel` only needs a live observation
dict to size itself, so the whole model -- group split, encoder, mirrored decoder and
both export wrappers -- can be checked without IsaacSim.  The one thing worth guarding
hardest is the export path: it takes a SINGLE flat observation vector and splits it
internally, so a mistake there produces a policy that runs but reads garbage.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from robot_rl.rl.ref_encoder_model import RefEncoderMLPModel

POLICY_DIM, REFERENCE_DIM, NUM_ACTIONS = 37, 23, 5
ENCODER_DIMS = [16, 12, 8]
GROUPS = {"actor": ["policy", "reference"]}
DIST_CFG = {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "log"}


def _obs(num_envs: int = 6) -> TensorDict:
    """A two-group observation batch shaped like the teacher's actor input."""
    torch.manual_seed(0)
    return TensorDict(
        {"policy": torch.randn(num_envs, POLICY_DIM), "reference": torch.randn(num_envs, REFERENCE_DIM)},
        batch_size=[num_envs],
    )


def _model(**overrides) -> RefEncoderMLPModel:
    """Build a small reference-encoder actor over :func:`_obs`."""
    kwargs = dict(
        hidden_dims=[24, 16],
        activation="elu",
        distribution_cfg=dict(DIST_CFG),
        encoder_obs_groups=["reference"],
        encoder_dims=list(ENCODER_DIMS),
    )
    kwargs.update(overrides)
    return RefEncoderMLPModel(_obs(), GROUPS, "actor", NUM_ACTIONS, **kwargs)


def _linear_shapes(mlp) -> list[tuple[int, int]]:
    """``(in, out)`` of every Linear layer in an MLP, in order."""
    return [(m.in_features, m.out_features) for m in mlp if isinstance(m, torch.nn.Linear)]


def test_group_split_and_trunk_width():
    """The reference group is encoded; the trunk sees the rest plus the latent."""
    model = _model()
    assert model.obs_groups == ["policy"]
    assert model.obs_dim == POLICY_DIM
    assert model.encoder_obs_groups == ["reference"]
    assert model.encoder_obs_dim == REFERENCE_DIM
    # Last encoder_dims entry is the latent width, not a hidden layer.
    assert model.encoder_latent_dim == ENCODER_DIMS[-1]
    assert model._get_latent_dim() == POLICY_DIM + ENCODER_DIMS[-1]
    assert _linear_shapes(model.mlp)[0][0] == POLICY_DIM + ENCODER_DIMS[-1]
    # The flat export vector is every actor group concatenated in obs_groups order.
    assert model.flat_obs_dim == POLICY_DIM + REFERENCE_DIM


def test_encoder_shape():
    """``encoder_dims`` maps to hidden layers plus a latent output."""
    model = _model()
    assert _linear_shapes(model.encoder) == [(REFERENCE_DIM, 16), (16, 12), (12, 8)]


def test_forward_shape():
    """The model still produces one action vector per env."""
    obs = _obs()
    assert _model()(obs).shape == (obs.batch_size[0], NUM_ACTIONS)


def test_no_decoder_by_default():
    """Reconstruction is off unless asked for -- it is the ablation, not the default."""
    model = _model()
    assert model.decoder is None
    with pytest.raises(RuntimeError, match="reconstruction_loss requires the decoder"):
        model.reconstruction_loss(_obs())


def test_decoder_mirrors_the_encoder():
    """The default decoder is the encoder's transpose, derived rather than restated."""
    model = _model(reconstruction=True)
    assert _linear_shapes(model.decoder) == [(8, 12), (12, 16), (16, REFERENCE_DIM)]


def test_decoder_dims_override():
    """An explicit decoder shape wins over the mirror."""
    model = _model(reconstruction=True, reconstruction_dims=[7])
    assert _linear_shapes(model.decoder) == [(8, 7), (7, REFERENCE_DIM)]


def test_reconstruction_loss_reaches_the_encoder():
    """The auxiliary loss is finite and its gradient flows back into the encoder."""
    model = _model(reconstruction=True)
    loss = model.reconstruction_loss(_obs())
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    first = next(m for m in model.encoder if isinstance(m, torch.nn.Linear))
    assert first.weight.grad is not None
    assert torch.linalg.vector_norm(first.weight.grad) > 0.0


def test_reconstruction_loss_ignores_the_trunk():
    """The auxiliary term supervises the encoder only, not the action trunk."""
    model = _model(reconstruction=True)
    model.reconstruction_loss(_obs()).backward()
    assert all(
        m.weight.grad is None or torch.linalg.vector_norm(m.weight.grad) == 0.0
        for m in model.mlp
        if isinstance(m, torch.nn.Linear)
    )


@pytest.mark.parametrize("export", ["as_jit", "as_onnx"])
def test_export_matches_forward_on_the_flat_vector(export):
    """Both export wrappers reproduce the TensorDict forward from one flat input.

    This is the deploy contract: the runtime concatenates the exported term table into a
    single vector, so the exported graph has to split it back exactly as the training
    model did.
    """
    model = _model(reconstruction=True).eval()
    obs = _obs()
    flat = torch.cat([obs["policy"], obs["reference"]], dim=-1)
    exported = getattr(model, export)().eval()
    with torch.no_grad():
        assert torch.equal(exported(flat), model(obs))


def test_jit_scripts():
    """The JIT wrapper survives ``torch.jit.script`` (how play.py writes policy.pt)."""
    model = _model().eval()
    obs = _obs()
    scripted = torch.jit.script(model.as_jit())
    with torch.no_grad():
        assert torch.equal(scripted(torch.cat([obs["policy"], obs["reference"]], dim=-1)), model(obs))


def test_onnx_wrapper_advertises_one_flat_input():
    """The ONNX wrapper keeps the single-``obs``-input contract, just wider."""
    onnx_model = _model().as_onnx()
    dummy = onnx_model.get_dummy_inputs()
    assert onnx_model.input_names == ["obs"] and onnx_model.output_names == ["actions"]
    assert len(dummy) == 1 and dummy[0].shape == (1, POLICY_DIM + REFERENCE_DIM)


def test_normalization_covers_the_reference():
    """Turning normalization on must not leave the encoder branch unnormalized."""
    model = _model(obs_normalization=True)
    obs = _obs()
    model.update_normalization(obs)
    assert int(model.encoder_obs_normalizer.count.item()) == obs.batch_size[0]


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"encoder_dims": [8]}, "encoder_dims"),
        ({"encoder_obs_groups": ["absent"]}, "not in the"),
        ({"encoder_obs_groups": ["policy", "reference"]}, "trunk"),
    ],
)
def test_misconfiguration_raises(overrides, match):
    """Bad group / width configs fail at construction rather than at the first step."""
    with pytest.raises(ValueError, match=match):
        _model(**overrides)
