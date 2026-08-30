# Copyright 2026 Zachary Olkin. All rights reserved.

"""Which ARM a conditioned generator serves must travel in the artifact, not the call site.

One parameter set answers to all three labels, so the arm is not recoverable from the
weights -- exactly the trap ``attention_mode`` and ``extero_clip`` are already documented
for.  The failure it produces is the dangerous kind: serving ``null`` where ``positive`` was
meant gives a perfectly plausible generator that is simply not the thing under test, with
nothing in the run to show it.

So ``load_fm_context`` REFUSES a conditioned checkpoint that names no arm.  Not a default:
``null`` and ``positive`` are both real labels, so any default silently converts the
deliverable into a control or the reverse.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from robot_rl.flow_matching.config import FlowMatchingConfig  # noqa: E402
from robot_rl.flow_matching.model import FlowMatchingTransformer  # noqa: E402
from robot_rl.flow_matching.normalizer import Normalizer  # noqa: E402
from robot_rl.flow_matching.online import load_fm_context, sample_window  # noqa: E402

NJ, F, N, CMD = 2, 6, 2, 3
K_EXT = 12


def _norm(d: int) -> dict:
    """A pass-through normalizer of width ``d``, as a state dict."""
    return Normalizer(torch.zeros(d), torch.ones(d), torch.ones(d, dtype=torch.bool)).state_dict()


def _ckpt(advantage_cond: bool, serve_label: str | None = None) -> dict:
    """A minimal in-memory checkpoint ``load_fm_context`` can consume."""
    cfg = FlowMatchingConfig(
        num_history_states=N, future_len=F, cmd_array="target_vel", cmd_stride=2,
        extero_source="height_scan", extero_sensor_dim=K_EXT, extero_sensor_count=1,
        d_model=16, nhead=2, num_layers=1, dim_feedforward=32,
        reference_blocks=("base_pose", "base_vel"), num_sampling_steps=2,
        advantage_cond=advantage_cond, advantage_serve_label=serve_label,
    )
    cfg.state_dim = 15                       # base_pose(9) + base_vel(6), no joints
    # `from_config` expects the encoder already RESOLVED -- the dataset writes "auto" out at
    # build time, and `load_fm_context` does the same on the way in.
    cfg.extero_encoder = cfg.resolve_extero_encoder()
    cfg.extero_channels = cfg.resolve_extero_channels()
    model = FlowMatchingTransformer.from_config(cfg).eval()
    if advantage_cond:                       # a zero table would make the arms indistinguishable
        with torch.no_grad():
            model.advantage_emb.weight.normal_(0.0, 0.5)
    return {
        "config": cfg.to_dict(),
        "meta": {"state_joint_names": [f"j{i}" for i in range(NJ)], "target_frame_names": []},
        "num_joints": NJ,
        "state_normalizer": _norm(cfg.state_dim),
        "cmd_normalizer": _norm(CMD),
        "extero_normalizer": _norm(K_EXT),
        "model": model.state_dict(),
    }


def test_a_conditioned_checkpoint_that_names_no_arm_is_REFUSED():
    """There is no safe default: `null` and `positive` are both real labels."""
    with pytest.raises(ValueError, match="does not say WHICH arm"):
        load_fm_context(_ckpt(True), device="cpu")


@pytest.mark.parametrize("name,index", [("negative", 0), ("positive", 1), ("null", 2)])
def test_the_arm_travels_in_the_ARTIFACT(name, index):
    """The checkpoint states its own arm, so no call site can state it differently."""
    ctx = load_fm_context(_ckpt(True, name), device="cpu")
    assert ctx.advantage_label == index


def test_an_explicit_override_beats_the_artifact():
    """The exporter and arm-sweeping diagnostics legitimately state the label themselves."""
    ctx = load_fm_context(_ckpt(True, "null"), device="cpu", advantage_label="positive")
    assert ctx.advantage_label == 1


def test_an_unconditioned_checkpoint_is_UNAFFECTED_and_refuses_a_label():
    """The pretrained generator must keep loading exactly as before, and a label on it is a
    caller error rather than a no-op -- it would otherwise be dropped in silence."""
    ctx = load_fm_context(_ckpt(False), device="cpu")
    assert ctx.advantage_label is None
    with pytest.raises(ValueError, match="advantage_cond=False"):
        load_fm_context(_ckpt(False), device="cpu", advantage_label="positive")


def test_the_arm_REACHES_the_sampler():
    """The point of all of the above: two arms must produce different served windows.

    Same conditioning, same noise draw, so the label is the only difference. Without this
    the plumbing could be complete and correct up to `sample_window` and still serve one arm.
    """
    served = {}
    for name in ("positive", "negative"):
        ctx = load_fm_context(_ckpt(True, name), device="cpu")
        g = torch.Generator().manual_seed(0)
        served[name] = sample_window(
            ctx,
            torch.randn(3, ctx.N, ctx.D, generator=g),
            torch.randn(3, K_EXT, generator=g),
            torch.randn(3, ctx.num_cmd_tokens, ctx.cmd_dim, generator=g),
            noise=torch.randn(3, ctx.F, ctx.D, generator=torch.Generator().manual_seed(1)),
        )
    assert not torch.allclose(served["positive"], served["negative"])
