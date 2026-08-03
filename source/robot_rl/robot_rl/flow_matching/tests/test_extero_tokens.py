# Copyright 2026 Zachary Olkin. All rights reserved.

"""The FM's exteroception slot: one token for the height scan, one per depth camera.

The height scan must stay behaviourally identical (it is the validated path and existing
checkpoints have to keep loading), while depth becomes selectable via
``cfg.extero_source``.  These tests pin both halves.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from robot_rl.flow_matching.config import EXTERO_SOURCES, FlowMatchingConfig
from robot_rl.flow_matching.dataset import build_train_val
from robot_rl.flow_matching.tests.recording_fixtures import per_step_recording
from robot_rl.flow_matching.model import FlowMatchingTransformer
from robot_rl.flow_matching.transforms import (
    DEPTH_CLIP,
    HEIGHT_SCAN_CLIP,
    sanitize_extero,
)


def _camera_spec(n_cameras: int, token_dim: int) -> list[dict]:
    """A minimal depth-camera contract matching a ``(n_cameras, token_dim)`` depth array.

    Field set and values mirror :func:`mdp.depth_camera_specs` (the real producer) closely
    enough to satisfy both the dataset's validation and
    ``deploy.depth_obs.parse_depth_spec``, so a spec built here would survive the real
    export -> deploy path.

    The GEOMETRY is a placeholder: ``token_dim x 1`` pixels with unit intrinsics, chosen
    only so ``width * height == token_dim`` for any ``token_dim`` these tests pick.  They
    exercise token plumbing (counts, widths, source selection, sanitising), not camera
    reproduction -- nothing here asserts the numbers describe a real camera.

    Args:
        n_cameras: Number of cameras, one per depth token.
        token_dim: Per-camera token width (pixels).

    Returns:
        One contract dict per camera, in token order.
    """
    return [
        {
            "obs_term": f"depth_cam{i}",
            "sensor": f"cam{i}",
            "mount_body": "torso_link",
            "offset_pos": [0.0, 0.0, 0.0],
            "offset_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            "width": token_dim,
            "height": 1,
            "fx": 1.0,
            "fy": 1.0,
            "cx": token_dim / 2.0,
            "cy": 0.5,
            "max_distance": 9.0,
            "update_period": 0.0,
            "data_type": "distance_to_image_plane",
            "depth_clipping_behavior": "max",
            "clip": None,
            # normalize_depth's defaults, as _normalization_params would resolve them.
            "offset": 0.75,
            "scale": 0.5,
            "min_dist": 0.2,
            "max_dist": 9.0,
            "min_clip_val": 0.0,
            "max_clip_val": 0.0,
        }
        for i in range(n_cameras)
    ]


def _synthetic_recording(n_ep=3, L=24, nj=4, K=2, F=6, Ksc=7, Kstep=4, depth=None, seed=0):
    """Synthetic recording with a height scan and optionally a depth array.

    Delegates the trajectory itself to the shared fixture builder -- the dataset validates
    the recording now, so it has to be a plausible rollout (see
    :mod:`~robot_rl.flow_matching.tests.recording_fixtures`).

    Args:
        depth: ``(num_cameras, per_camera_dim)`` to add a ``depth`` array, or ``None``.

    Returns:
        ``(data, meta, nj, K, F, node_dim)``.
    """
    data, stn, std, frames = per_step_recording(
        n_ep=n_ep, length=L, nj=nj, k=K, future_len=F, snap_every=Kstep,
        num_scan_rays=Ksc, depth=depth, seed=seed,
    )
    meta = SimpleNamespace(state_term_names=stn, state_term_dims=std, state_frame_names=frames,
                           state_joint_names=[f"j{i}" for i in range(nj)], ref_future_len=F,
                           # Present only alongside a depth array, exactly as the recorder
                           # behaves -- so the depth=None case still fails on the MISSING
                           # ARRAY (what test_missing_array_names_the_task_to_record_with
                           # pins) rather than on the missing contract.
                           depth_camera_spec=_camera_spec(depth[0], depth[1]) if depth else None)
    return data, meta, nj, K, F, sum(std)


def _cfg(**kw) -> FlowMatchingConfig:
    base = dict(num_history_states=2, future_len=6, cmd_array="target_vel", cmd_stride=4,
                target_source="ego_anchored_clip_future", val_fraction=0.34, seed=0)
    base.update(kw)
    return FlowMatchingConfig(**base)


##
# Config
##

def test_default_source_is_depth():
    """The DEPLOYABLE signal is the default: the robot has cameras, not a heightmap.

    Defaulting to the height scan made it too easy to train an FM that could never leave
    the simulator, so the sim-only signal is now the one you opt into.
    """
    assert FlowMatchingConfig().extero_source == "depth"


def test_noise_knob_follows_the_source():
    """Each source has its own amplitude -- metres vs normalised depth units."""
    assert _cfg(extero_source="height_scan").noise_extero == pytest.approx(
        FlowMatchingConfig().noise_height_scan)
    assert _cfg(extero_source="depth").noise_extero == pytest.approx(
        FlowMatchingConfig().noise_depth)


def test_unknown_source_is_rejected():
    """A typo must fail loudly rather than silently conditioning on nothing."""
    with pytest.raises(ValueError, match="extero_source"):
        _cfg(extero_source="lidar").check_extero_source()


def test_old_checkpoint_config_still_loads():
    """A config dict predating the selector resolves to the height-scan path."""
    d = FlowMatchingConfig().to_dict()
    for k in ("extero_source", "extero_tokens", "noise_depth"):
        d.pop(k, None)
    cfg = FlowMatchingConfig.from_dict(d)
    assert cfg.extero_source == "height_scan" and cfg.extero_tokens == 1


def test_terrain_penetration_loss_is_gone():
    """The flat-floor terrain losses were deleted (wrong on the stairs/boxes we train on).

    Re-adding a penetration penalty is fine, but it must use a LOCAL floor (SDF or
    height-scan-derived), so a resurrected flat-``ground_z`` knob should fail here.
    """
    names = {f.name for f in dataclasses.fields(FlowMatchingConfig)}
    assert not [n for n in names if n.startswith("terrain_")]


def test_stale_checkpoint_config_with_terrain_keys_loads():
    """Checkpoints trained before the removal carry ``terrain_*`` keys; ignore them."""
    d = FlowMatchingConfig(extero_source="height_scan").to_dict()
    d["terrain_penetration_weight"] = 200.0
    d["terrain_penetration_mode"] = "ankle"
    cfg = FlowMatchingConfig.from_dict(d)
    # The unknown keys are dropped, and every known one survives round-tripping -- the
    # recorded source in particular, which must NOT be re-defaulted on load.
    assert cfg.extero_source == "height_scan"
    assert not hasattr(cfg, "terrain_penetration_weight")


##
# Sanitizer clip bands
##

def test_depth_band_is_wider_than_the_scan_band():
    """Reusing the scan's 1.0 m band would silently clip real depth readings."""
    assert DEPTH_CLIP > HEIGHT_SCAN_CLIP
    v = torch.tensor([[1.5]])
    assert sanitize_extero(v, "depth").item() == pytest.approx(1.5)
    assert sanitize_extero(v, "height_scan").item() == pytest.approx(HEIGHT_SCAN_CLIP)


@pytest.mark.parametrize("source", EXTERO_SOURCES)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_is_sanitized_for_both_sources(source, bad):
    """One non-finite entry would make the normalizer mean non-finite -> NaN loss."""
    out = sanitize_extero(torch.tensor([[bad, 0.25]]), source)
    assert torch.isfinite(out).all()


def test_unknown_source_has_no_clip_band():
    with pytest.raises(KeyError):
        sanitize_extero(torch.zeros(1, 2), "lidar")


##
# Model
##

def test_single_token_model_has_no_extra_parameters():
    """T=1 must keep the exact parameter set so height-scan checkpoints load."""
    common = dict(state_dim=12, future_len=6, num_history_states=2, cmd_dim=3)
    m1 = FlowMatchingTransformer(height_dim=176, extero_tokens=1, **common)
    m2 = FlowMatchingTransformer(height_dim=780, extero_tokens=2, **common)
    k1 = set(dict(m1.named_parameters()))
    k2 = set(dict(m2.named_parameters()))
    assert not any("extero_id" in k for k in k1)
    assert k2 - k1 == {"extero_id"}


@pytest.mark.parametrize("tokens,dim", [(1, 176), (2, 780), (3, 64)])
def test_forward_accepts_flat_and_token_shaped_input(tokens, dim):
    """(B, T*dim) and (B, T, dim) must both work -- the recorder writes the latter."""
    m = FlowMatchingTransformer(state_dim=12, future_len=6, num_history_states=2, cmd_dim=3,
                                height_dim=dim, extero_tokens=tokens)
    b = 4
    x, tau = torch.zeros(b, 6, 12), torch.rand(b)
    hist, cmd = torch.zeros(b, 2, 12), torch.zeros(b, len(m.cmd_steps), 3)
    flat = m(x, tau, hist, torch.zeros(b, tokens * dim), cmd)
    shaped = m(x, tau, hist, torch.zeros(b, tokens, dim), cmd)
    assert flat.shape == (b, 6, 12) == shaped.shape
    torch.testing.assert_close(flat, shaped)


def test_token_count_changes_the_sequence_length():
    """Extra extero tokens must actually reach the encoder, not be folded together."""
    common = dict(state_dim=12, future_len=6, num_history_states=2, cmd_dim=0, height_dim=8)
    seen = {}
    for tokens in (1, 3):
        m = FlowMatchingTransformer(extero_tokens=tokens, **common)
        captured = []
        m.encoder.register_forward_pre_hook(lambda _m, inp: captured.append(inp[0].shape[1]))
        m(torch.zeros(2, 6, 12), torch.rand(2), torch.zeros(2, 2, 12), torch.zeros(2, tokens, 8))
        seen[tokens] = captured[0]
    assert seen[3] - seen[1] == 2  # N + T + F, so +2 tokens


def test_per_token_id_distinguishes_cameras():
    """Without a per-token id the two cameras would be interchangeable to the model."""
    m = FlowMatchingTransformer(state_dim=12, future_len=6, num_history_states=2, cmd_dim=0,
                                height_dim=8, extero_tokens=2)
    args = (torch.zeros(2, 6, 12), torch.rand(2), torch.zeros(2, 2, 12), torch.zeros(2, 2, 8))
    before = m(*args)
    with torch.no_grad():
        m.extero_id[0, 1] += 1.0
    assert not torch.allclose(before, m(*args))


def test_swapping_two_cameras_changes_the_prediction():
    """Order matters: near and far must not be a symmetric pair."""
    torch.manual_seed(0)
    m = FlowMatchingTransformer(state_dim=12, future_len=6, num_history_states=2, cmd_dim=0,
                                height_dim=8, extero_tokens=2)
    with torch.no_grad():  # break the zero-init symmetry of extero_id
        m.extero_id.normal_()
    near, far = torch.randn(2, 1, 8), torch.randn(2, 1, 8)
    common = (torch.zeros(2, 6, 12), torch.rand(2), torch.zeros(2, 2, 12))
    a = m(*common, torch.cat([near, far], dim=1))
    b = m(*common, torch.cat([far, near], dim=1))
    assert not torch.allclose(a, b)


##
# Dataset
##

def test_height_scan_dataset_resolves_one_token():
    """The scan path resolves T=1 and the full ray count as the token width."""
    data, meta, *_ = _synthetic_recording()
    cfg = _cfg(extero_source="height_scan")      # explicit: the default is depth
    train, _ = build_train_val(data, meta, cfg)
    ksc = data["height_scan"].shape[1]
    assert cfg.extero_tokens == 1 and cfg.height_dim == ksc
    assert train[0]["height"].shape == (ksc,)


def test_depth_dataset_resolves_one_token_per_camera():
    """A (rows, T, dim) depth array yields T tokens of width dim, flattened per item."""
    data, meta, *_ = _synthetic_recording(depth=(2, 15))
    cfg = _cfg(extero_source="depth")
    train, _ = build_train_val(data, meta, cfg)
    assert cfg.extero_tokens == 2 and cfg.height_dim == 15
    assert train[0]["height"].shape == (30,)


def test_source_selects_which_array_is_used():
    """Both arrays present -> the config decides, and the widths differ observably."""
    data, meta, *_ = _synthetic_recording(depth=(2, 15))
    scan_cfg = _cfg(extero_source="height_scan")
    depth_cfg = _cfg(extero_source="depth")
    scan_ds, _ = build_train_val(data, meta, scan_cfg)
    depth_ds, _ = build_train_val(data, meta, depth_cfg)
    assert scan_ds[0]["height"].shape != depth_ds[0]["height"].shape


def test_missing_array_names_the_task_to_record_with():
    """Selecting depth on a scan-only recording must say how to fix it."""
    data, meta, *_ = _synthetic_recording()          # no depth array
    with pytest.raises(ValueError, match="depth"):
        build_train_val(data, meta, _cfg(extero_source="depth"))


def test_dataset_sanitizes_the_selected_array():
    """A non-finite recording must not poison the normalizer."""
    data, meta, *_ = _synthetic_recording(depth=(2, 15))
    data["depth"][0, 0, 0] = float("inf")
    train, _ = build_train_val(data, meta, _cfg(extero_source="depth"))
    assert torch.isfinite(train.height).all()
    assert torch.isfinite(train.height_normalizer.mean).all()
