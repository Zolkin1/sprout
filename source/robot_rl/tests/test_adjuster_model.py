# Copyright 2026 Zachary Olkin. All rights reserved.

"""The adjuster actor: what it borrows, what it trains, and what it must not own.

Built against a small synthetic generator of the shipped SHAPE, so these run in a second and
still exercise the real ``encode_prefix`` / ``_run_layer`` path.

Five properties, each of which fails silently if it breaks:

* **The mean residual is exactly zero at initialisation**, so the served plan starts
  bit-identical to the bare generator's. That is the baseline every measurement of this policy
  is against, and over thousands of action dimensions a careless head swamps the plan on
  iteration one.
* **The head is SHARED across nodes.** Permuting the input tokens must permute the output
  blocks identically. A flat trunk would have to learn that correspondence instead, from an
  ``F * D``-dimensional policy gradient.
* **The generator is borrowed, not owned**: absent from ``state_dict()`` and from
  ``parameters()``, which PPO's single optimizer and its gradient clipping both walk. A 27M
  copy in the checkpoint would be demanded back by ``load(strict=True)`` and would silently
  restore the wrong generator on a resume against a different one.
* **Gradients reach the input embedding.** The borrowed layers are frozen by
  ``requires_grad_(False)`` and NOT by ``no_grad``, because ``token_in`` sits upstream of
  them -- wrapping them would sever the graph and train nothing but the head, while still
  producing a model that runs and a loss that falls.
* **The trainable copies are actually trainable.** ``deepcopy`` inherits ``requires_grad``
  from the source, which is frozen by then, so without an explicit re-enable the "two tuned
  layers" would be two frozen ones.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")
pytest.importorskip("rsl_rl.models", exc_type=ImportError)
pytest.importorskip("isaaclab.utils.math", exc_type=ImportError)

from tensordict import TensorDict  # noqa: E402

from robot_rl.flow_matching import online  # noqa: E402
from robot_rl.flow_matching.model import FlowMatchingTransformer  # noqa: E402
from robot_rl.flow_matching.normalizer import Normalizer  # noqa: E402
from robot_rl.rl.adjuster_model import WindowAdjusterModel  # noqa: E402

# The shipped SHAPE at a runnable size: 3 history nodes, an 8-node plan, 2 layers.
N, F, D, NJ = 3, 8, 12, 4
SENSORS, SENSOR_DIM = 2, 10
CMD_DIM, CMD_STRIDE = 3, 4
D_MODEL, LAYERS = 32, 2
N_ENVS = 5

COND_DIM = N * D + SENSORS * SENSOR_DIM + len(range(0, F, CMD_STRIDE)) * CMD_DIM
PLAN_DIM = 2 * F * D
ACTION_DIM = F * D


def _context(attention_mode: str = "prefix") -> online.FmOnlineContext:
    """A minimal published context around a small generator of the shipped shape."""
    torch.manual_seed(0)
    model = FlowMatchingTransformer(
        state_dim=D, future_len=F, num_history_states=N,
        extero_sensor_dim=SENSOR_DIM, extero_sensor_count=SENSORS,
        cmd_dim=CMD_DIM, cmd_stride=CMD_STRIDE,
        d_model=D_MODEL, nhead=4, num_layers=LAYERS, dim_feedforward=64,
        extero_encoder="mlp", attention_mode=attention_mode,
    ).eval()
    # Mirror `load_fm_context`, which freezes what it loads. Not cosmetic: freezing flips
    # `nn.MultiheadAttention` onto its fused fast path, whose float32 rounding differs from
    # the unfused one by ~5e-07 -- so a fixture left unfrozen would make every before/after
    # comparison here measure that instead of what the test is about.
    for p in model.parameters():
        p.requires_grad_(False)
    ident = lambda n: Normalizer(torch.zeros(n), torch.ones(n), torch.ones(n, dtype=torch.bool))  # noqa: E731
    cmd_steps = list(range(0, F, CMD_STRIDE))
    return online.FmOnlineContext(
        model=model, state_norm=ident(D), cmd_norm=ident(CMD_DIM), extero_norm=ident(SENSOR_DIM),
        N=N, F=F, D=D, nj=NJ, target_num_extra_frames=0, num_steps=4,
        cmd_dim=CMD_DIM, cmd_steps=cmd_steps, num_cmd_tokens=len(cmd_steps),
        joint_names=[f"j{i}" for i in range(NJ)], target_frame_names=[],
        device=torch.device("cpu"),
        extero_sensor_count=SENSORS, extero_sensor_dim=SENSOR_DIM,
    )


@pytest.fixture()
def published():
    """Publish a context for the duration of one test, then clear it."""
    ctx = _context()
    online.set_active_fm_context(ctx)
    try:
        yield ctx
    finally:
        online.set_active_fm_context(None)


def _obs(batch: int = N_ENVS) -> TensorDict:
    gen = torch.Generator().manual_seed(1)
    return TensorDict(
        {
            "fm_cond": torch.randn(batch, COND_DIM, generator=gen),
            "fm_plan": torch.randn(batch, PLAN_DIM, generator=gen) * 0.1,
        },
        batch_size=[batch],
    )


def _model(obs: TensorDict, **kwargs) -> WindowAdjusterModel:
    return WindowAdjusterModel(
        obs, {"actor": ["fm_cond", "fm_plan"]}, "actor", ACTION_DIM,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5,
                          "std_type": "log"},
        num_frozen_layers=LAYERS - 1,
        **kwargs,
    )


def test_the_initial_residual_is_exactly_zero(published) -> None:
    """So the served plan starts bit-identical to the bare generator's."""
    obs = _obs()
    model = _model(obs)
    out = model(obs)                       # deterministic by default
    assert out.shape == (N_ENVS, ACTION_DIM)
    assert float(out.detach().abs().max()) == 0.0, "the head must be zero-initialised"


def test_the_head_is_shared_across_nodes(published) -> None:
    """Permuting the input nodes permutes the output blocks identically.

    Checked after perturbing the head off zero, since at exactly zero every permutation is
    trivially equal and the test would pass on a flat trunk too.
    """
    obs = _obs()
    model = _model(obs)
    with torch.no_grad():
        model.mlp.proj.weight.normal_(std=0.1)
        model.mlp.proj.bias.normal_(std=0.1)
        # Kill the node-position embedding as well: it legitimately breaks permutation
        # equivariance (node 0 is one control step out, node F-1 over a second), so leaving it
        # in would test the embedding rather than the head.
        model.node_pos.zero_()

    latent = model.get_latent(obs).view(N_ENVS, F, -1)
    perm = torch.randperm(F, generator=torch.Generator().manual_seed(3))
    direct = model.mlp(latent.reshape(N_ENVS, -1)).view(N_ENVS, F, D)
    permuted = model.mlp(latent[:, perm].reshape(N_ENVS, -1)).view(N_ENVS, F, D)
    torch.testing.assert_close(permuted, direct[:, perm])


def test_the_generator_is_borrowed_not_owned(published) -> None:
    """Absent from state_dict and parameters, so it stays out of every PPO checkpoint."""
    obs = _obs()
    model = _model(obs)

    fm_params = {id(p) for p in published.model.parameters()}
    assert fm_params, "the fixture generator must have parameters to look for"
    assert not (fm_params & {id(p) for p in model.parameters()}), (
        "a generator parameter is reachable from model.parameters(); PPO's optimizer and its "
        "gradient clipping both walk that, and alg.save() would embed all 27M weights"
    )
    keys = list(model.state_dict())
    assert not any(k.startswith("_fm") or k.startswith("fm.") for k in keys)
    # And its parameter count is the adjuster's own, not the generator's.
    own = sum(p.numel() for p in model.parameters())
    assert own < sum(p.numel() for p in published.model.parameters())


def test_gradients_reach_the_input_embedding_and_the_tuned_layers(published) -> None:
    """The frozen layers are frozen by requires_grad, NOT by no_grad.

    If they were wrapped in ``no_grad`` the model would still run and a loss would still fall
    -- on the head alone. This is the test that separates those two.
    """
    obs = _obs()
    model = _model(obs)
    with torch.no_grad():
        model.mlp.proj.weight.normal_(std=0.1)      # off zero, or every grad is zero

    model(obs).square().sum().backward()

    assert model.token_in.weight.grad is not None
    assert float(model.token_in.weight.grad.abs().sum()) > 0.0, (
        "no gradient reached token_in -- the borrowed layers severed the graph"
    )
    assert model.node_pos.grad is not None and float(model.node_pos.grad.abs().sum()) > 0.0
    tuned = next(model.tuned_layers.parameters())
    assert tuned.grad is not None and float(tuned.grad.abs().sum()) > 0.0

    # And nothing accumulated on the generator itself.
    assert all(p.grad is None for p in published.model.parameters())


def test_the_tuned_copies_are_trainable_and_the_borrowed_layers_are_not(published) -> None:
    """deepcopy inherits requires_grad from a source that is frozen by then."""
    obs = _obs()
    model = _model(obs)

    assert len(model.tuned_layers) == 1
    assert all(p.requires_grad for p in model.tuned_layers.parameters())
    assert not any(p.requires_grad for p in published.model.parameters())

    # The copy starts EQUAL to the layer it warm-started from, and is a different tensor.
    src = published.model.encoder.layers[model.num_frozen_layers]
    for (name, a), (_, b) in zip(model.tuned_layers[0].named_parameters(),
                                 src.named_parameters()):
        torch.testing.assert_close(a, b, msg=f"{name} did not warm-start from the generator")
        assert a.data_ptr() != b.data_ptr(), f"{name} shares storage with the generator"


def test_training_the_adjuster_does_not_change_the_generator(published) -> None:
    """The property the whole design turns on, asserted end to end.

    The last layers are the adjuster's own COPIES, warm-started from the generator. Had they
    been shared, an optimizer step would move the generator too -- which is the generator
    fine-tune that was already tried and hit a measured ceiling, not a residual adjuster. And
    it would be invisible: the plan would still be a plan, and every number would describe a
    generator that no longer matched its checkpoint.
    """
    from robot_rl.flow_matching.flow import sample_trajectory

    fm = published.model
    cond = _obs()
    n_hist = N * D
    n_ext = SENSORS * SENSOR_DIM
    history = cond["fm_cond"][:, :n_hist].reshape(N_ENVS, N, D)
    extero = cond["fm_cond"][:, n_hist:n_hist + n_ext]
    cmd = cond["fm_cond"][:, n_hist + n_ext:].reshape(N_ENVS, published.num_cmd_tokens, CMD_DIM)
    noise = torch.randn(N_ENVS, F, D, generator=torch.Generator().manual_seed(7))

    def plan() -> torch.Tensor:
        return sample_trajectory(fm, history, extero, cmd, F, D, 4, noise=noise)

    before = plan()

    model = _model(cond)
    with torch.no_grad():
        model.mlp.proj.weight.normal_(std=0.1)
    model(cond).square().sum().backward()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    opt.step()

    after = plan()
    assert torch.equal(before, after), (
        "an optimizer step on the adjuster moved the generator's own output -- the tuned "
        "layers are shared with it rather than copied"
    )


def test_a_full_attention_generator_is_refused() -> None:
    """There is no fixed encoding to borrow, and the two modes share one parameter set."""
    ctx = _context(attention_mode="full")
    online.set_active_fm_context(ctx)
    try:
        with pytest.raises(ValueError, match=r"prefix-attention"):
            _model(_obs())
    finally:
        online.set_active_fm_context(None)


def test_the_action_width_must_match_the_generators_layout(published) -> None:
    """The action IS the per-node residual, so a mismatch is a wrong-width residual."""
    obs = _obs()
    with pytest.raises(ValueError, match=r"output_dim .* != F \* D"):
        WindowAdjusterModel(obs, {"actor": ["fm_cond", "fm_plan"]}, "actor", ACTION_DIM + 1)


def test_a_group_of_the_wrong_width_names_the_generators_own_layout(published) -> None:
    """Every group here is a slice of the generator's state; a fitting width is not enough."""
    obs = _obs()
    bad_plan = obs.clone()
    bad_plan["fm_plan"] = torch.zeros(N_ENVS, PLAN_DIM - D)
    with pytest.raises(ValueError, match=r"'fm_plan' is .* wide but the generator's plan"):
        _model(bad_plan)

    bad_cond = obs.clone()
    bad_cond["fm_cond"] = torch.zeros(N_ENVS, COND_DIM - 1)
    with pytest.raises(ValueError, match=r"'fm_cond' is .* wide but this generator's"):
        _model(bad_cond)


def test_observation_normalization_is_refused(published) -> None:
    """Both groups arrive in the generator's DECLARED normalized space."""
    with pytest.raises(ValueError, match=r"obs_normalization=False"):
        WindowAdjusterModel(_obs(), {"actor": ["fm_cond", "fm_plan"]}, "actor", ACTION_DIM,
                            obs_normalization=True)


def test_a_missing_context_says_what_to_pass() -> None:
    """The registry raises rather than handing back None."""
    online.set_active_fm_context(None)
    with pytest.raises(RuntimeError, match=r"fm_checkpoint"):
        _model(_obs())
