# Copyright 2026 Zachary Olkin. All rights reserved.

"""Serving the planner online: the duck-type against the flow-matching generator.

``FmTrackingCommand`` is written against a CONTEXT and one sampler, not against a flow model.
These tests hold the two halves of that claim: the planner's context carries every field the
flow context does (so the command and the three shared token builders work unchanged), and
its ``sample_window`` returns the same thing in the same layout.

The builders are the interesting half. They are imported from ``flow_matching.online`` and
run here on a PLANNER context -- if a field name drifted, they would fail on the real serving
path and nowhere else.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from robot_rl.flow_matching.normalizer import Normalizer
from robot_rl.flow_matching.online import (
    FmOnlineContext,
    build_cmd_tokens,
    build_extero_tokens,
    build_history_tokens,
)
from robot_rl.token_planner.config import PlannerConfig, TokenizerConfig
from robot_rl.token_planner.online import (
    TokenPlannerContext,
    load_planner_context,
    sample_window,
)
from robot_rl.token_planner.planner import TokenPlanner
from robot_rl.token_planner.tokenizer import TrajectoryTokenizer

NJ, K = 4, 0
NODE_DIM = 9 + 6 + NJ            # base_pose + base_vel + joint_pos


def make_checkpoint(tmp_path, **cfg_kw) -> str:
    """Write a small but structurally real planner checkpoint.

    Args:
        tmp_path: pytest tmp dir.
        **cfg_kw: :class:`PlannerConfig` overrides.

    Returns:
        Path to the written ``planner.pt``.
    """
    tok = TokenizerConfig(
        node_dim=NODE_DIM, chunk_len=16, downsample=4, channels=16,
        num_blocks_high=1, num_blocks_low=1, fsq_levels=[4, 4, 3],
        num_joints=NJ, num_ee_frames=K,
        reference_blocks=["base_pose", "base_vel", "joint_pos"],
    )
    cfg = PlannerConfig(
        tokenizer=tok.to_dict(), history_dim=NODE_DIM, num_history_states=3, future_len=16,
        extero_encoder="cnn", extero_sensor_count=2, extero_channels=5,
        extero_sensor_dim=5 * 6 * 8, extero_image_hw=[[6, 8], [6, 8]],
        extero_cnn_channels=[8, 16], extero_cnn_strides=[1, 2],
        extero_camera_intrinsics=[[4.0, 5.0, 3.5, 2.5], [7.0, 9.0, 3.5, 2.5]],
        cmd_dim=3, cmd_stride=5, d_model=32, nhead=4, num_layers=2, dim_feedforward=64,
        dropout=0.0,
    )
    for k, v in cfg_kw.items():
        setattr(cfg, k, v)
    model = TokenPlanner(cfg, TrajectoryTokenizer(tok))

    d, e, c = NODE_DIM, cfg.extero_sensor_count * cfg.extero_sensor_dim, cfg.cmd_dim
    payload = {
        "model": model.state_dict(),
        "config": cfg.to_dict(),
        "epoch": 7,
        "state_normalizer": Normalizer(
            torch.zeros(d), torch.full((d,), 0.5), torch.ones(d, dtype=torch.bool)
        ).state_dict(),
        "extero_normalizer": Normalizer(
            torch.zeros(e), torch.ones(e), torch.ones(e, dtype=torch.bool)
        ).state_dict(),
        "cmd_normalizer": Normalizer(
            torch.zeros(c), torch.tensor([3.5, 1.0, 0.8]), torch.ones(c, dtype=torch.bool)
        ).state_dict(),
        "num_joints": NJ,
        "num_ee_frames": K,
        "reference_blocks": ["base_pose", "base_vel", "joint_pos"],
        "target_frame_names": [],
        "fm_config": {"history_source": "reference", "extero_clip": 5.575,
                      "depth_latency_probs": [0.0, 0.1, 0.4, 0.5]},
        "meta": {"fm_snap_every": 12, "state_joint_names": [f"j{i}" for i in range(NJ)]},
    }
    path = str(tmp_path / "planner.pt")
    torch.save(payload, path)
    return path


def test_the_context_carries_every_field_the_flow_context_does():
    """The command and the token builders read the flow context's field names.

    A field dropped here fails on the real serving path and nowhere else -- there is no
    shape to catch it, because the failure is an ``AttributeError`` inside a builder that
    only ever runs against a live simulator.
    """
    fm = {f.name for f in dataclasses.fields(FmOnlineContext)}
    tp = {f.name for f in dataclasses.fields(TokenPlannerContext)}
    assert not (fm - tp), f"missing from the planner context: {sorted(fm - tp)}"


def test_loading_gives_a_servable_context(tmp_path):
    ctx = load_planner_context(make_checkpoint(tmp_path), device="cpu")
    assert ctx.N == 3 and ctx.F == 16 and ctx.D == NODE_DIM and ctx.nj == NJ
    assert ctx.num_steps == 4                      # N tokens = chunk_len / downsample
    assert ctx.cmd_dim == 3 and ctx.cmd_steps == [0, 5, 10, 15]
    assert ctx.num_cmd_tokens == 4
    assert ctx.reference_blocks == ("base_pose", "base_vel", "joint_pos")
    assert ctx.history_source == "reference"
    assert ctx.fm_snap_every == 12
    assert ctx.extero_clip == pytest.approx(5.575)
    assert ctx.depth_latency_probs == (0.0, 0.1, 0.4, 0.5)
    assert not ctx.model.training, "a served model must be in eval mode"
    assert not any(p.requires_grad for p in ctx.model.parameters())


def test_the_flow_matchings_own_builders_work_on_a_planner_context(tmp_path):
    """The three conditioning builders are IMPORTED from ``flow_matching.online``.

    Running them here is the test that the duck-type is real rather than a naming
    coincidence: they touch ``ctx.{D, N, state_norm, extero_norm, extero_source,
    extero_clip, cmd_dim, cmd_norm, cmd_steps}``.
    """
    ctx = load_planner_context(make_checkpoint(tmp_path), device="cpu")
    v = 3
    nodes = torch.randn(v, ctx.N, ctx.D)
    hist = build_history_tokens(ctx, nodes)
    assert hist.shape == (v, ctx.N, ctx.D)

    raw = torch.randn(v, ctx.extero_sensor_count, ctx.extero_sensor_dim)
    extero = build_extero_tokens(ctx, raw)
    assert extero.shape == (v, ctx.extero_sensor_count * ctx.extero_sensor_dim)
    assert torch.isfinite(extero).all()

    cmd = build_cmd_tokens(ctx, torch.randn(v, ctx.F, ctx.cmd_dim))
    assert cmd.shape == (v, ctx.num_cmd_tokens, ctx.cmd_dim)


def test_sample_window_returns_a_reference_window_with_real_rotations(tmp_path):
    """The shape and layout the tracker is handed, and a pose it can actually use."""
    ctx = load_planner_context(make_checkpoint(tmp_path), device="cpu", greedy=True)
    v = 3
    hist = build_history_tokens(ctx, torch.randn(v, ctx.N, ctx.D))
    extero = build_extero_tokens(
        ctx, torch.randn(v, ctx.extero_sensor_count, ctx.extero_sensor_dim))
    cmd = build_cmd_tokens(ctx, torch.randn(v, ctx.F, ctx.cmd_dim))

    w = sample_window(ctx, hist, extero, cmd)
    assert w.shape == (v, ctx.F, ctx.D)
    assert torch.isfinite(w).all()
    # base_pose is first, so its rot6d half is columns 3:9 -- and must be orthonormal, or
    # the tracker is being served something that is not a pose.
    a, b = w[..., 3:6], w[..., 6:9]
    assert float((a.norm(dim=-1) - 1).abs().max()) < 1e-5
    assert float((b.norm(dim=-1) - 1).abs().max()) < 1e-5
    assert float((a * b).sum(-1).abs().max()) < 1e-5


def test_greedy_serving_is_deterministic_and_the_null_history_is_live(tmp_path):
    """Determinism, and that the null-history path is actually wired through the seam.

    The null flag is checked on the PREFIX EMBEDDING rather than on the served plan. A
    discrete bottleneck makes the plan piecewise-constant in its conditioning: a change that
    flips no token gives a bit-identical trajectory, so asserting the plan must move is
    asserting that a token happened to flip. Measured on this fixture, the flag moves the
    prefix by ~1.0 and leaves the four argmax decisions untouched.
    """
    ctx = load_planner_context(make_checkpoint(tmp_path), device="cpu", greedy=True)
    v = 2
    hist = build_history_tokens(ctx, torch.randn(v, ctx.N, ctx.D))
    extero = build_extero_tokens(
        ctx, torch.randn(v, ctx.extero_sensor_count, ctx.extero_sensor_dim))
    cmd = build_cmd_tokens(ctx, torch.randn(v, ctx.F, ctx.cmd_dim))
    assert torch.equal(sample_window(ctx, hist, extero, cmd),
                       sample_window(ctx, hist, extero, cmd))

    with torch.no_grad():
        plain = ctx.model.prefix(hist, extero, cmd, torch.zeros(v, dtype=torch.bool))
        nulled = ctx.model.prefix(hist, extero, cmd, torch.ones(v, dtype=torch.bool))
    assert not torch.allclose(plain, nulled), "the null-history embedding is not reachable"
    # And it is servable end to end without raising.
    w = sample_window(ctx, hist, extero, cmd,
                      history_null_mask=torch.ones(v, dtype=torch.bool))
    assert w.shape == (v, ctx.F, ctx.D) and torch.isfinite(w).all()


def test_what_has_no_analogue_is_refused_rather_than_ignored(tmp_path):
    """A probe that injects a fixed ``noise`` to hold the draw constant across arms would
    otherwise vary nothing and report the difference as zero."""
    ctx = load_planner_context(make_checkpoint(tmp_path), device="cpu")
    v = 2
    hist = build_history_tokens(ctx, torch.randn(v, ctx.N, ctx.D))
    extero = build_extero_tokens(
        ctx, torch.randn(v, ctx.extero_sensor_count, ctx.extero_sensor_dim))
    cmd = build_cmd_tokens(ctx, torch.randn(v, ctx.F, ctx.cmd_dim))
    with pytest.raises(ValueError, match="takes no `noise`"):
        sample_window(ctx, hist, extero, cmd, noise=torch.randn(v, ctx.F, ctx.D))
    with pytest.raises(ValueError, match="no flow field"):
        sample_window(ctx, hist, extero, cmd, return_flow=True)
    with pytest.raises(ValueError, match="cmd_tokens is None"):
        sample_window(ctx, hist, extero, None)
