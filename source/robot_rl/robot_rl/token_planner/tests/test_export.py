# Copyright 2026 Zachary Olkin. All rights reserved.

"""The exported graph must be the torch path, and must carry its own contract.

Two claims, and the first is the one everything downstream rests on: a plan produced by
``fm_session``'s single ``session.run`` has to be the plan ``online.sample_window`` would have
produced from the same conditioning. If it is not, every closed-loop number measured in
MuJoCo or on hardware is about a different model than the one that was trained and evaluated.

The second is the Gumbel-max identity. ``argmax(logits/T + g)`` with ``g ~ Gumbel(0,1)`` is an
EXACT sample from ``softmax(logits/T)`` -- that is what lets a data-dependent sampling loop
become a static graph with the randomness supplied as an input, and it is what keeps the
``noise`` slot the flow model's runtimes already bind. Zeros must give exactly greedy.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

onnxruntime = pytest.importorskip("onnxruntime")

from robot_rl.flow_matching.online import (  # noqa: E402
    build_extero_tokens,
    build_history_tokens,
)
from robot_rl.token_planner.online import load_planner_context, sample_window  # noqa: E402
from robot_rl.token_planner.tests.test_online import NODE_DIM, make_checkpoint  # noqa: E402


def export_pair(tmp_path, temperature: float = 1.0):
    """Build a checkpoint, export it, and return ``(ctx, session, raw_inputs)``.

    Args:
        tmp_path: pytest tmp dir.
        temperature: Baked-in graph temperature.

    Returns:
        ``(ctx, session, (history, extero, cmd))`` with the three RAW conditioning tensors.
    """
    from scripts_token_planner_export import export  # noqa: PLC0415

    ckpt = make_checkpoint(tmp_path)
    out = str(tmp_path / "planner_fm.onnx")
    export(ckpt, out, temperature=temperature)
    ctx = load_planner_context(ckpt, device="cpu", greedy=True)
    sess = onnxruntime.InferenceSession(out, providers=["CPUExecutionProvider"])
    torch.manual_seed(5)
    v = 1
    raw = (
        torch.randn(v, ctx.N, ctx.D),
        torch.randn(v, ctx.extero_sensor_count * ctx.extero_sensor_dim),
        torch.randn(v, ctx.num_cmd_tokens, ctx.cmd_dim),
    )
    return ctx, sess, raw


def run_graph(sess, ctx, raw, noise: np.ndarray | None = None, null: float = 0.0):
    """Run the exported graph once.

    Args:
        sess: An ``onnxruntime.InferenceSession``.
        ctx: The loaded context, for shapes.
        raw: ``(history, extero, cmd)`` raw tensors.
        noise: Gumbel noise, or ``None`` for zeros (greedy).
        null: The history-null flag.

    Returns:
        ``(1, F, D)`` numpy trajectory.
    """
    h, e, c = (t.numpy().astype(np.float32) for t in raw)
    if noise is None:
        noise = np.zeros((1, ctx.num_steps, ctx.model.vocab_size), dtype=np.float32)
    feed = {"history": h, "extero": e, "cmd_tokens": c, "noise": noise,
            "history_null": np.array([[null]], dtype=np.float32)}
    return sess.run(["trajectory"], feed)[0]


def test_the_graph_matches_the_torch_path(tmp_path):
    """The whole point of the artifact. A mismatch here means every closed-loop number in
    MuJoCo or on hardware is about a different model than the one that was evaluated."""
    ctx, sess, raw = export_pair(tmp_path)
    got = run_graph(sess, ctx, raw)

    hist = build_history_tokens(ctx, raw[0])
    extero = build_extero_tokens(
        ctx, raw[1].reshape(1, ctx.extero_sensor_count, ctx.extero_sensor_dim))
    # The graph is fed ALREADY-STRIDED command tokens, where `build_cmd_tokens` strides a
    # full-horizon command -- so normalize them the way that builder's tail would.
    cmd = ctx.cmd_norm.normalize(raw[2])
    want = sample_window(ctx, hist, extero, cmd).numpy()

    assert got.shape == want.shape == (1, ctx.F, ctx.D)
    assert np.abs(got - want).max() < 2e-4, f"max |graph - torch| = {np.abs(got-want).max():.3e}"


def test_zero_noise_is_exactly_greedy_and_is_deterministic(tmp_path):
    ctx, sess, raw = export_pair(tmp_path)
    a = run_graph(sess, ctx, raw)
    b = run_graph(sess, ctx, raw)
    assert np.array_equal(a, b), "a zero-noise graph run must be deterministic"


def test_gumbel_noise_actually_changes_the_plan(tmp_path):
    """The randomness has to REACH the output -- a graph that ignored `noise` would look
    perfectly healthy and would silently make every sampling experiment a no-op."""
    ctx, sess, raw = export_pair(tmp_path)
    greedy = run_graph(sess, ctx, raw)
    rng = np.random.default_rng(0)
    u = rng.uniform(1e-9, 1.0, size=(1, ctx.num_steps, ctx.model.vocab_size))
    gumbel = (-np.log(-np.log(u))).astype(np.float32)
    sampled = run_graph(sess, ctx, raw, gumbel)
    assert not np.allclose(greedy, sampled), "gumbel noise did not reach the plan"
    # And a different draw gives a different plan again.
    u2 = rng.uniform(1e-9, 1.0, size=u.shape)
    other = run_graph(sess, ctx, raw, (-np.log(-np.log(u2))).astype(np.float32))
    assert not np.allclose(sampled, other)


def test_the_history_null_flag_reaches_the_graph(tmp_path):
    """It covers the first ``N`` control steps after every reset, so it has to be live.

    Checked by AGREEMENT WITH TORCH under both flag values, not by "the plan changed".
    A discrete bottleneck makes the plan piecewise-constant in its conditioning: measured
    here, flipping the flag moves the prefix embedding by 1.004 and yet leaves the four
    argmax decisions -- and therefore the whole trajectory -- bit-identical. Asserting that
    the output must move would be testing whether a token happened to flip.
    """
    ctx, sess, raw = export_pair(tmp_path)
    from scripts_token_planner_export import _PlannerSamplerOnnx  # noqa: PLC0415

    wrapper = _PlannerSamplerOnnx(ctx, 1.0).eval()
    zeros = torch.zeros(1, ctx.num_steps, ctx.model.vocab_size)
    for flag in (0.0, 1.0):
        with torch.no_grad():
            want = wrapper(raw[0], raw[1], raw[2], zeros,
                           torch.full((1, 1), flag)).numpy()
        got = run_graph(sess, ctx, raw, null=flag)
        assert np.abs(got - want).max() < 2e-4, f"null={flag}: graph and torch disagree"


def test_the_plan_is_piecewise_constant_in_its_conditioning(tmp_path):
    """A property of the discrete bottleneck, pinned because it is easy to misread as a bug.

    A conditioning perturbation that flips no token produces a BIT-IDENTICAL plan. That is
    why the null-flag test above compares against torch rather than looking for a change,
    and it is worth knowing before anyone builds a sensitivity probe here and reads a step
    function as a broken gradient.
    """
    ctx, sess, raw = export_pair(tmp_path)
    base = run_graph(sess, ctx, raw)
    tiny = (raw[0] + 1e-6, raw[1], raw[2])
    assert np.array_equal(run_graph(sess, ctx, tiny), base), (
        "a perturbation far too small to flip a token must give the identical plan")

    # That the conditioning matters AT ALL is checked one level down, on the LOGITS, and
    # deliberately not on the plan: on an untrained model the argmax over a near-uniform
    # head is dominated by the head's own initialisation, so even a large conditioning
    # change flips no token here. The logits are where the dependence is visible before
    # training concentrates the distribution enough for it to reach the argmax.
    with torch.no_grad():
        tokens = torch.zeros(1, ctx.num_steps, dtype=torch.long)
        hist = build_history_tokens(ctx, raw[0])
        extero = build_extero_tokens(
            ctx, raw[1].reshape(1, ctx.extero_sensor_count, ctx.extero_sensor_dim))
        cmd = ctx.cmd_norm.normalize(raw[2])
        a = ctx.model(tokens, hist, extero, cmd)
        b = ctx.model(tokens, hist * 0.0 + 5.0, extero, cmd)
    assert not torch.allclose(a, b), "the conditioning must reach the logits"


def test_the_graph_serves_a_real_rotation(tmp_path):
    """The runtime does no rotation maths, so the projection has to be inside the graph."""
    ctx, sess, raw = export_pair(tmp_path)
    w = run_graph(sess, ctx, raw)
    a, b = w[..., 3:6], w[..., 6:9]
    assert np.abs(np.linalg.norm(a, axis=-1) - 1).max() < 1e-4
    assert np.abs(np.linalg.norm(b, axis=-1) - 1).max() < 1e-4
    assert np.abs((a * b).sum(-1)).max() < 1e-4


def test_the_metadata_states_what_a_runtime_cannot_assume(tmp_path):
    """`noise_shape` above all: the flow model's is (F, D) and this one's is (N_tok, vocab),
    so a runtime that computes it instead of reading it feeds the wrong tensor."""
    import json

    import onnx

    from scripts_token_planner_export import export  # noqa: PLC0415

    ckpt = make_checkpoint(tmp_path)
    out = str(tmp_path / "meta_fm.onnx")
    export(ckpt, out)
    m = onnx.load(out)
    meta = json.loads(
        {e.key: e.value for e in m.metadata_props}["fm_meta"])
    assert meta["generator_kind"] == "token_planner"
    assert meta["noise_distribution"] == "gumbel"
    assert meta["noise_shape"] == [meta["num_plan_tokens"], meta["vocab_size"]]
    # The contract keys `fm_session` refuses to load without.
    for key in ("window_blocks", "extero_sensor_dim", "extero_sensor_count", "N", "F", "D",
                "nj", "cmd_dim", "cmd_steps", "num_cmd_tokens", "joint_names",
                "target_frame_names", "has_history_null", "history_source"):
        assert key in meta, f"fm_meta is missing the load-bearing key {key!r}"
    assert meta["window_blocks"] == ["base_pose", "base_vel", "joint_pos"]
    assert meta["D"] == NODE_DIM
