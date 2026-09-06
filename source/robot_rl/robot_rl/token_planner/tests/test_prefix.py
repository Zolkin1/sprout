# Copyright 2026 Zachary Olkin. All rights reserved.

"""The conditioning prefix must be the flow-matching generator's, exactly.

The AR planner exists to be compared with the FM, and that comparison only means anything if
the two see the same conditioning. The prefix assembly is restated in ``token_planner.prefix``
rather than imported (the generator's copy is a method on a class that also builds a flow
trunk), so nothing but a test can keep the two from drifting -- and every way they could drift
leaves the prefix the right SHAPE: a missing type embedding, a positional offset off by
``num_history_states``, a sensor id broadcast down the wrong axis.
"""

from __future__ import annotations

import pytest
import torch

from robot_rl.flow_matching.config import FlowMatchingConfig
from robot_rl.flow_matching.model import FlowMatchingTransformer
from robot_rl.token_planner.prefix import ConditioningPrefix


def depth_config(**kw) -> FlowMatchingConfig:
    """A small depth-camera config in the shipped generator's shape.

    Args:
        **kw: Field overrides.

    Returns:
        The config. Small enough to build in milliseconds, but with T=2 sensors and a conv
        trunk, so ``extero_id`` and the spatial slots are both exercised.
    """
    base = dict(
        state_dim=44, history_dim=44, future_len=16, num_history_states=4,
        extero_source="depth", extero_encoder="cnn", extero_sensor_count=2,
        extero_channels=5, extero_sensor_dim=5 * 6 * 8, extero_image_hw=[[6, 8], [6, 8]],
        extero_cnn_channels=[8, 16], extero_cnn_strides=[1, 2],
        extero_camera_intrinsics=[[4.0, 5.0, 3.5, 2.5], [7.0, 9.0, 3.5, 2.5]],
        cmd_dim=3, cmd_stride=5, d_model=32, nhead=4, num_layers=2, dim_feedforward=64,
        attention_mode="prefix",
    )
    base.update(kw)
    return FlowMatchingConfig(**base)


def prefix_from(cfg: FlowMatchingConfig) -> ConditioningPrefix:
    """Build the planner's prefix from a flow-matching config.

    Args:
        cfg: The generator's config.

    Returns:
        A prefix module with the same architecture the generator would build.
    """
    return ConditioningPrefix(
        history_dim=cfg.history_dim, num_history_states=cfg.num_history_states,
        future_len=cfg.future_len, extero_sensor_dim=cfg.extero_sensor_dim,
        extero_sensor_count=cfg.extero_sensor_count, d_model=cfg.d_model,
        extero_encoder=cfg.extero_encoder, extero_image_hw=cfg.extero_image_hw,
        extero_cnn_channels=cfg.extero_cnn_channels, extero_cnn_grid=cfg.extero_cnn_grid,
        extero_cnn_strides=cfg.extero_cnn_strides,
        extero_coord_extent_m=cfg.extero_coord_extent_m,
        extero_grid_ordering=cfg.extero_grid_ordering,
        extero_fourier_bands=cfg.extero_fourier_bands,
        extero_miss_sentinel=cfg.extero_miss_sentinel,
        extero_channels=cfg.resolve_extero_channels(),
        extero_camera_intrinsics=cfg.extero_camera_intrinsics,
        cmd_dim=cfg.cmd_dim, cmd_stride=cfg.cmd_stride,
    )


@pytest.mark.parametrize("encoder,sensors", [("cnn", 2), ("cnn", 1), ("mlp", 1), ("mlp", 2)])
def test_the_prefix_matches_the_flow_matching_models_embedding(encoder, sensors):
    """Bit-identical output from identical weights, on every encoder/sensor combination.

    The ``mlp`` and ``T=1`` arms are here because they take structurally different paths --
    ``extero_id`` does not exist at ``T=1``, and the MLP path fabricates a length-1 slot axis
    -- so a bug could hide in any one of them while the shipped ``cnn``/``T=2`` arm passed.
    """
    hw = [[6, 8]] * sensors
    cfg = depth_config(
        extero_encoder=encoder, extero_sensor_count=sensors,
        extero_sensor_dim=5 * 6 * 8, extero_image_hw=(hw if encoder == "cnn" else None),
        extero_camera_intrinsics=[[4.0, 5.0, 3.5, 2.5]] * sensors,
    )
    fm = FlowMatchingTransformer.from_config(cfg).eval()
    prefix = prefix_from(cfg).eval()

    # RANDOMISE THE ZERO-INITIALISED TAGS FIRST. `hist_type`, `extero_type`, `cmd_type`,
    # `extero_id` and `null_history` are all zeros in a fresh model, so copying them across
    # and comparing outputs compares zero against zero -- and the test passes just as
    # happily with those terms deleted outright or with `extero_id` broadcast down the wrong
    # axis, which is measured: all four arms below passed with two of them removed. Since
    # this test is the ENTIRE safety net for restating the assembly, it has to be able to
    # see the drifts its own docstring names.
    torch.manual_seed(11)
    with torch.no_grad():
        for name in ("hist_type", "extero_type", "cmd_type", "null_history", "extero_id"):
            tag = getattr(fm, name, None)
            if tag is not None:
                tag.normal_(0.0, 0.5)

    shared = {k: v for k, v in fm.state_dict().items() if k in dict(prefix.state_dict())}
    missing, unexpected = prefix.load_state_dict(shared, strict=False)
    # Nothing may be left at its own initialisation: an un-copied parameter would make the
    # comparison below a comparison of two random modules that happen to agree on shape.
    #
    # The three exemptions are INDEX/POSITION buffers, not learned content: `pos` is a fixed
    # sinusoidal table, and `cmd_steps` / `plan_steps` are the timeline indices their streams
    # sit on. `plan_steps` is additionally EMPTY here -- the plan stream is an adjuster-only
    # option and the generator has no counterpart to copy from, which is exactly why it
    # defaults off. Anything with a weight in it must still transfer.
    assert not [m for m in missing if m not in ("pos", "cmd_steps", "plan_steps")], missing
    assert prefix.plan_in is None, (
        "the parity check must run with the plan stream OFF -- the generator has no plan "
        "stream, so a prefix carrying one cannot be bit-identical to its embedding.")
    # `unexpected` cannot fire (``shared`` is filtered to the prefix's own keys), so the
    # real check is that every tag the FM has was actually transferred.
    have = dict(prefix.state_dict())
    for name in ("hist_type", "extero_type", "cmd_type", "null_history", "extero_id"):
        if getattr(fm, name, None) is not None:
            assert name in have, f"the prefix has no {name}; the streams are untagged"
            assert torch.equal(have[name], getattr(fm, name)), f"{name} did not transfer"

    torch.manual_seed(0)
    b = 3
    history = torch.randn(b, cfg.num_history_states, cfg.history_dim)
    extero = torch.randn(b, cfg.extero_sensor_count, cfg.extero_sensor_dim)
    cmd = torch.randn(b, len(range(0, cfg.future_len, cfg.cmd_stride)), cfg.cmd_dim)
    null = torch.tensor([False, True, False])

    with torch.no_grad():
        want = fm._embed_prefix(history, extero, cmd, null)
        have = prefix(history, extero, cmd, null)
    assert have.shape == want.shape == (b, fm.prefix_len, cfg.d_model)
    assert torch.equal(have, want)
    assert prefix.prefix_len == fm.prefix_len
    assert prefix.extero_seq_tokens == fm.extero_seq_tokens


def test_the_null_history_mask_actually_replaces_the_history():
    """The null embedding covers the first N steps after a reset, when no history exists.

    A mask that silently did nothing would be invisible in training (the loss would simply be
    computed on real history) and would then deny the model the learned null row at exactly
    the steps it was meant for.
    """
    cfg = depth_config()
    prefix = prefix_from(cfg).eval()
    with torch.no_grad():
        prefix.null_history.copy_(torch.full_like(prefix.null_history, 3.0))
        history = torch.randn(2, cfg.num_history_states, cfg.history_dim)
        extero = torch.randn(2, cfg.extero_sensor_count, cfg.extero_sensor_dim)
        cmd = torch.randn(2, len(range(0, cfg.future_len, cfg.cmd_stride)), cfg.cmd_dim)
        out = prefix(history, extero, cmd, torch.tensor([True, False]))
    n = cfg.num_history_states
    # Row 0 is the null embedding plus the timeline position and the type tag; row 1 is not.
    base = out[0, :n] - prefix.pos[0, :n] - prefix.hist_type[0, 0]
    assert torch.allclose(base, torch.full_like(base, 3.0), atol=1e-5)
    assert not torch.allclose(out[1, :n], out[0, :n])


def test_a_command_conditioned_prefix_refuses_to_run_without_command_tokens():
    cfg = depth_config()
    prefix = prefix_from(cfg)
    with pytest.raises(ValueError, match="requires cmd_tokens"):
        prefix(torch.randn(1, cfg.num_history_states, cfg.history_dim),
               torch.randn(1, cfg.extero_sensor_count, cfg.extero_sensor_dim), None)
