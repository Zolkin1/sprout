# Copyright 2026 Zachary Olkin. All rights reserved.

"""The residual arm: the composition rule, the normalizer, the cache, and the split.

Everything here pins a property whose violation is SHAPE-CORRECT and therefore silent:

* a composition that perturbs the plan at zero residual -- a 1e-7 seed on a robot;
* a residual normalizer that moves the origin, so "leave the generator alone" stops being
  representable;
* a residual cache loaded as an absolute one, whose tensors have identical shape and a
  completely different meaning;
* a held-out split that leaks a clip into training, which was MEASURED to deflate the
  generator's residual to 0.41-0.78x its honest value and is invisible in every count.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from robot_rl.flow_matching.config import FlowMatchingConfig
from robot_rl.flow_matching.dataset import VAL_SPLIT_KEYS, draw_val_split
from robot_rl.flow_matching.transforms import matrix_to_rot6d, so3_exp
from robot_rl.flow_matching.window import (
    RESIDUAL_MAX_TILT_RAD,
    compose_window_residual,
    decompose_window_residual,
    window_block_slices,
    window_rotation_span,
)
from robot_rl.token_planner.residual import (
    ABSOLUTE_ARM,
    RESIDUAL_ARM,
    ResidualChunkData,
    declared_residual_normalizer,
    fit_residual_normalizer,
    load_residual_cache,
    residual_cache_key,
    save_residual_cache,
)

BLOCKS = ["base_pose", "base_vel", "joint_pos"]
NJ, NEE, H = 29, 4, 16
D = 9 + 6 + NJ


def _plan(n: int = 3, seed: int = 0) -> torch.Tensor:
    """A ``(n, H, D)`` window whose rot6d block is a real rotation, as a served plan is."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n, H, D, generator=g) * 0.1
    span = window_rotation_span(NJ, NEE, BLOCKS)
    rot = so3_exp(torch.randn(n, H, 3, generator=g) * 0.2)
    w[..., span] = matrix_to_rot6d(rot)
    return w


def test_a_zero_residual_returns_the_plan_bit_identically():
    """Not "close": Gram-Schmidt moves an orthonormal frame ~2.4e-7 in float32, and a served
    plan that differs from the bare generator's by 1e-7 is a chaotic divergence on a robot,
    not a rounding difference. This is the property the whole zero-action control rests on."""
    plan = _plan()
    out = compose_window_residual(plan, torch.zeros_like(plan), NJ, NEE, BLOCKS)
    assert torch.equal(out, plan)


def test_alpha_zero_is_the_identity_however_large_the_residual():
    """``alpha = 0`` is how the adjuster initialises as a no-op, so it must be exact even when
    the residual it is scaling is enormous."""
    plan = _plan()
    out = compose_window_residual(plan, torch.randn_like(plan) * 10.0, NJ, NEE, BLOCKS,
                                  alpha=0.0)
    assert torch.equal(out, plan)


def test_decompose_inverts_compose_on_every_linear_channel():
    """The linear half must round-trip exactly -- that is what makes the residual
    reconstruction error equal the end-to-end plan error there."""
    plan, target = _plan(seed=1), _plan(seed=2)
    delta = decompose_window_residual(plan, target)
    out = compose_window_residual(plan, delta, NJ, NEE, BLOCKS)
    span = window_rotation_span(NJ, NEE, BLOCKS)
    lin = torch.ones(D, dtype=torch.bool)
    lin[span] = False
    torch.testing.assert_close(out[..., lin], target[..., lin], rtol=0, atol=1e-6)


def test_the_rotation_half_round_trips_inside_the_tilt_bound():
    """Inside the bound the composition is the identity on the rotation too, so a residual
    fitted offline reproduces its target rather than being clamped toward the plan."""
    plan = _plan(seed=3)
    span = window_rotation_span(NJ, NEE, BLOCKS)
    small = so3_exp(torch.full((plan.shape[0], H, 3), RESIDUAL_MAX_TILT_RAD / 8))
    target = plan.clone()
    target[..., span] = matrix_to_rot6d(small @ _rot(plan, span))
    delta = decompose_window_residual(plan, target)
    out = compose_window_residual(plan, delta, NJ, NEE, BLOCKS)
    torch.testing.assert_close(out, target, rtol=0, atol=1e-5)


def _rot(w: torch.Tensor, span: slice) -> torch.Tensor:
    """The window's rotation block as matrices."""
    from robot_rl.flow_matching.transforms import rot6d_to_matrix

    return rot6d_to_matrix(w[..., span])


def test_the_tilt_clamp_only_ever_shrinks():
    """A residual asking for more than the bound is limited to it, and the limited rotation is
    never FURTHER from the plan than the request was."""
    plan = _plan(seed=4)
    span = window_rotation_span(NJ, NEE, BLOCKS)
    big = so3_exp(torch.full((plan.shape[0], H, 3), 1.0))          # ~1.73 rad, way past 15 deg
    target = plan.clone()
    target[..., span] = matrix_to_rot6d(big @ _rot(plan, span))
    out = compose_window_residual(plan, decompose_window_residual(plan, target),
                                  NJ, NEE, BLOCKS)
    from robot_rl.flow_matching.transforms import so3_log

    moved = so3_log(_rot(out, span) @ _rot(plan, span).transpose(-1, -2))
    assert float(moved.norm(dim=-1).max()) <= RESIDUAL_MAX_TILT_RAD * (1 + 1e-5)


def test_compose_refuses_a_broadcastable_residual():
    """Broadcasting would apply one node's correction to every node, silently."""
    plan = _plan()
    with pytest.raises(ValueError, match="must match exactly"):
        compose_window_residual(plan, plan[:, :1], NJ, NEE, BLOCKS)


def test_the_residual_normalizer_pins_the_origin_and_standardizes_every_channel():
    """A fitted mean would move "the generator was right" off zero, and an unstandardized
    rot6d block -- correct for the STATE normalizer -- would leave the rotation residual at a
    fifth of base_pos's magnitude and let the loss ignore it."""
    delta = torch.randn(500, H, D) * torch.linspace(0.01, 2.0, D) + 3.0
    norm = fit_residual_normalizer(delta)
    assert torch.equal(norm.mean, torch.zeros(D))
    assert bool(norm.standardize_mask.all())
    # de-normalizing recovers physical units exactly, which is what makes the reported RMSE
    # comparable to the generator's own held-out error.
    torch.testing.assert_close(norm.denormalize(norm.normalize(delta)), delta,
                               rtol=1e-5, atol=1e-5)


def test_a_zero_residual_normalizes_to_zero():
    """The property ZERO_CODE, the injected all-zero chunks and the alpha=0 init all rest on."""
    norm = fit_residual_normalizer(torch.randn(200, H, D) * 5.0 + 7.0)
    assert torch.equal(norm.normalize(torch.zeros(1, H, D)), torch.zeros(1, H, D))


def test_the_declared_band_refuses_a_selection_it_has_no_scale_for():
    """``joint_vel`` and ``ee_pos`` are inert in the adjuster's authority table, so a declared
    normalizer over them would divide by zero rather than say so."""
    declared_residual_normalizer(NJ, NEE, BLOCKS)                      # the shipped selection
    with pytest.raises(ValueError, match="declared residual band is zero"):
        declared_residual_normalizer(NJ, NEE, [*BLOCKS, "joint_vel"])


def _residual_chunks(**over) -> ResidualChunkData:
    """A minimal residual build, for the cache round trip."""
    n = 4
    norm = fit_residual_normalizer(torch.randn(50, H, D))
    kw = dict(
        train_nodes=torch.randn(n, H, D), val_nodes=torch.randn(2, H, D),
        train_family=torch.zeros(n, dtype=torch.long),
        val_family=torch.zeros(2, dtype=torch.long),
        state_normalizer_state=norm.state_dict(), family_names=["unknown"],
        reference_blocks=list(BLOCKS), num_joints=NJ, num_ee_frames=NEE,
        chunk_len=H, node_dim=D, dt=0.02, val_episode_ids=[1, 2],
        provenance={"data_sha256": "abc", "seed": 0, "val_fraction": 0.15},
        train_plan=torch.randn(n, H, D), val_plan=torch.randn(2, H, D),
        absolute_normalizer_state=norm.state_dict(),
        fm_provenance={"fm_run": "/run", "fm_sha256": "dead", "draw": "zero_noise",
                       "history_source": "measured", "honesty": "clean", "fm_epoch": 94},
    )
    kw.update(over)
    return ResidualChunkData(**kw)


def test_the_residual_cache_round_trips(tmp_path):
    """The plans and the generator provenance have to survive, or the plan-space gate has
    nothing to score against and the cache cannot say which generator it describes."""
    chunks = _residual_chunks()
    path = str(tmp_path / "res.pt")
    save_residual_cache(path, chunks)
    key = residual_cache_key(chunks.provenance, chunks.chunk_len, chunks.reference_blocks,
                             chunks.fm_provenance)
    back = load_residual_cache(path, key)
    assert torch.equal(back.train_plan, chunks.train_plan)
    assert back.fm_provenance["history_source"] == "measured"


def test_a_residual_cache_is_refused_by_the_absolute_loader(tmp_path):
    """Both hold ``(Nw, H, D)`` float tensors; only their MEANING differs, so nothing about
    the shapes would stop a run training on the wrong quantity to completion."""
    from robot_rl.token_planner.data import cache_key, load_chunk_cache

    path = str(tmp_path / "res.pt")
    chunks = _residual_chunks()
    save_residual_cache(path, chunks)
    want = cache_key(chunks.provenance, chunks.chunk_len, chunks.reference_blocks)
    with pytest.raises(ValueError, match="ABSOLUTE build was asked for"):
        load_chunk_cache(path, want)


def test_an_absolute_cache_is_refused_by_the_residual_loader(tmp_path):
    """The mirror direction, so neither arm can be reached by accident."""
    from robot_rl.token_planner.data import save_chunk_cache

    chunks = _residual_chunks()
    plain = {k: v for k, v in vars(chunks).items()
             if k not in ("train_plan", "val_plan", "absolute_normalizer_state",
                          "fm_provenance")}
    from robot_rl.token_planner.data import ChunkData

    path = str(tmp_path / "abs.pt")
    save_chunk_cache(path, ChunkData(**plain))
    key = residual_cache_key(chunks.provenance, chunks.chunk_len, chunks.reference_blocks,
                             chunks.fm_provenance)
    with pytest.raises(ValueError, match="RESIDUAL build was asked for"):
        load_residual_cache(path, key)


def test_the_cache_key_names_the_generator_and_its_arm():
    """A residual is defined by the generator that produced it, how it was sampled and which
    rows were judged honest -- so a cache built under a different history arm, a different
    draw or a different honesty set must not satisfy a request for this one."""
    chunks = _residual_chunks()
    base = residual_cache_key(chunks.provenance, chunks.chunk_len, chunks.reference_blocks,
                              chunks.fm_provenance)
    for field, other in (("history_source", "reference"), ("draw", "gaussian"),
                         ("honesty", "clean+twin"), ("fm_sha256", "beef")):
        moved = dict(chunks.fm_provenance, **{field: other})
        assert residual_cache_key(chunks.provenance, chunks.chunk_len,
                                  chunks.reference_blocks, moved) != base, field


def test_the_clip_split_is_disjoint_where_the_episode_split_leaks():
    """The measured failure, reproduced: a clip played by several episodes -- or re-rendered
    into a second file whose episode ids train.py offsets -- puts the SAME motion on both
    sides of an episode-keyed draw. On the shipped measured_n3 pair that was 740 of 855
    held-out clips."""
    rng = np.random.default_rng(0)
    w = 20000
    clip = rng.integers(0, 500, size=w)
    ep = clip * 4 + rng.integers(0, 4, size=w)
    ep[w // 2:] += 100_000                       # the per-file offset train.py applies
    data = {"episode_id": ep, "clip_id": clip}

    by_ep, _, _ = draw_val_split(data, FlowMatchingConfig(val_split_by="episode", seed=0), w)
    by_clip, key, held = draw_val_split(
        data, FlowMatchingConfig(val_split_by="clip", seed=0), w)

    leaked_ep = np.intersect1d(np.unique(clip[by_ep]), np.unique(clip[~by_ep]))
    leaked_clip = np.intersect1d(np.unique(clip[by_clip]), np.unique(clip[~by_clip]))
    assert leaked_ep.size > 0, "the episode split should leak on this construction"
    assert leaked_clip.size == 0
    assert key == "clip" and held.size == np.unique(clip[by_clip]).size
    # and it still holds out roughly what was asked for
    assert 0.05 * w < by_clip.sum() < 0.2 * w


def test_the_split_is_deterministic_and_refuses_what_it_cannot_key_on():
    """A silent fallback to episodes would reinstate the leak while reporting a clip split."""
    rng = np.random.default_rng(1)
    w = 2000
    data = {"episode_id": rng.integers(0, 300, w), "clip_id": rng.integers(0, 80, w)}
    cfg = FlowMatchingConfig(val_split_by="clip", seed=0)
    assert torch.equal(torch.from_numpy(draw_val_split(data, cfg, w)[0]),
                       torch.from_numpy(draw_val_split(data, cfg, w)[0]))
    with pytest.raises(ValueError, match="not one of"):
        draw_val_split(data, FlowMatchingConfig(val_split_by="nonsense"), w)
    with pytest.raises(ValueError, match="needs a per-window 'clip_id'"):
        draw_val_split({"episode_id": data["episode_id"]}, cfg, w)
    assert set(VAL_SPLIT_KEYS) == {"episode", "clip"}


def test_a_suppressed_geodesic_prints_as_absent_not_as_zero():
    """The residual arm drops ``ori_geodesic_rad`` because the angle between two DIFFERENCES
    of rotation frames is not a rotation angle. A defaulted 0.0 would print as a perfect
    score for a quantity nobody measured -- which is the exact shape of every silent-default
    trap in this repo."""
    from robot_rl.token_planner.metrics import format_report

    rep = {
        "chunks": 1, "group_units": {"base_pos": "m"}, "rmse": {"base_pos": 0.1},
        "rmse_by_family": {}, "rmse_by_position": {"base_pos": [0.1]},
        "boundary_penalty": {}, "node_rmse_p99": {"base_pos": 0.2},
        "node_rmse_max": {"base_pos": 0.3},
        "codebook": {"distinct": 1.0, "vocab_size": 2.0, "utilisation": 0.5,
                     "entropy_bits": 1.0, "entropy_frac": 1.0, "perplexity": 2.0},
    }
    out = format_report(rep)
    assert "n/a (residual arm" in out
    assert "0.00000 rad" not in out
    assert "0.05000 rad" in format_report({**rep, "ori_geodesic_rad": 0.05})
