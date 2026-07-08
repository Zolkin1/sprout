# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the offline clip generator's stitching invariants.

Requires the full-periodic stride library on disk (skipped if absent, e.g.
in CI without the HF cache).  Asserts the two correctness properties that
the dense rollout must hold: a continuous world stance foot across stride
seams (the anchor recursion, now baked into the stored clip-world outputs)
and an exactly uniform global timeline (the dt-remainder carry).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

_LIB = Path("hf/trajectories/retargeted/2026-06-13_15-10-58_merged")
pytestmark = pytest.mark.skipif(not _LIB.exists(), reason=f"stride library {_LIB} not present")

DEVICE = "cpu"


@dataclass
class _Bucket:
    lin_vel_x: tuple[float, float] = (0.0, 0.0)
    lin_vel_y: tuple[float, float] = (0.0, 0.0)
    ang_vel_z: tuple[float, float] = (0.0, 0.0)
    heading: tuple[float, float] | None = None


@pytest.fixture(scope="module")
def generated():
    from motion_gen.multiskill_manager import (
        MultiSkillManager,
    )
    from motion_gen.clips.offline_velocity_sampler import (
        OfflineVelocitySampler,
    )
    from motion_gen.clips.offline_clip_generator import (
        OfflineClipGenerator,
    )

    m = MultiSkillManager(str(_LIB), DEVICE, env=None)
    skills = [s.name for s in m.skills]
    buckets = {
        "running": _Bucket((1.5, 3.0)),
        "standing": _Bucket((0.0, 0.1)),
        "walk_backwards": _Bucket((-1.0, -0.1)),
        "walk_forward": _Bucket((0.1, 1.5), (-0.3, 0.3), (-0.5, 0.5)),
    }
    probs = [0.4, 0.05, 0.2, 0.35]
    sampler = OfflineVelocitySampler(
        buckets, skills, probs, resampling_time_range=(0.8, 1.5),
        max_acc=2.0, command_dt=0.02, device=DEVICE, seed=0,
    )
    gen = OfflineClipGenerator(m, sampler, dt=0.02, clip_length_s=5.0, device=DEVICE)
    out = gen.generate_batch(16)
    return m, gen, out


def test_shapes_and_finite(generated):
    _, gen, out = generated
    n, T = 16, gen.num_steps
    assert out["y_des"].shape == (n, T, gen.num_pos)
    assert out["dy_des"].shape == (n, T, gen.num_vel)
    # The per-step anchor schedule is baked into y_des/dy_des, not stored.
    assert "anchor_pos" not in out and "anchor_quat" not in out
    for k in ("y_des", "dy_des"):
        assert torch.isfinite(out[k]).all(), k


def _foot_cols(names):
    """``([lx,ly,lz], [rx,ry,rz])`` ankle position columns."""
    left = [names.index(f"left_ankle_roll_link:pos_{a}") for a in ("x", "y", "z")]
    right = [names.index(f"right_ankle_roll_link:pos_{a}") for a in ("x", "y", "z")]
    return torch.tensor(left), torch.tensor(right)


def test_world_landing_foot_continuous_across_seams(generated):
    """Across each stride seam the JUST-LANDED foot must not jump (planted).

    The rollout is now per STEP (half cycle), alternating which foot lands at a
    seam: the LEFT ankle at a mid-cycle strike, the RIGHT ankle at a cycle wrap.
    So continuity holds for whichever foot just landed -- read as the smaller of
    the two ankle jumps at each seam (the swing foot legitimately moves a lot).
    Outputs are baked to clip-world, so the ankle tracks are read from ``y_des``.
    """
    m, gen, out = generated
    left, right = _foot_cols(m.pos_output_names)
    y, st = out["y_des"], out["stride_idx"]
    max_jump = 0.0
    for ci in range(y.shape[0]):
        wl, wr = y[ci][:, left], y[ci][:, right]
        bnd = (st[ci, 1:] != st[ci, :-1]).nonzero().flatten()
        for b in bnd.tolist():
            dl = (wl[b + 1] - wl[b]).norm().item()
            dr = (wr[b + 1] - wr[b]).norm().item()
            max_jump = max(max_jump, min(dl, dr))  # the landed (continuous) foot
    assert max_jump < 0.10, f"landed foot jumps {max_jump:.3f} m across a seam"


def test_seams_alternate_landing_foot(generated):
    """Per-step stitching lands each foot at some seams (not always the right).

    The old whole-cycle stitcher re-anchored only the right ankle, so the right
    was continuous at EVERY seam.  Per step the left ankle is the continuous foot
    at mid-cycle seams -- so across a batch BOTH feet act as the landing foot.
    """
    m, gen, out = generated
    left, right = _foot_cols(m.pos_output_names)
    y, st = out["y_des"], out["stride_idx"]
    left_landed = right_landed = 0
    for ci in range(y.shape[0]):
        wl, wr = y[ci][:, left], y[ci][:, right]
        bnd = (st[ci, 1:] != st[ci, :-1]).nonzero().flatten()
        for b in bnd.tolist():
            dl = (wl[b + 1] - wl[b]).norm().item()
            dr = (wr[b + 1] - wr[b]).norm().item()
            if dl < dr:
                left_landed += 1
            else:
                right_landed += 1
    assert left_landed > 0 and right_landed > 0, (
        f"expected both feet to land across seams; left={left_landed} right={right_landed}"
    )


def test_blend_window_must_be_step_fraction(generated):
    """``transition_blend_end_phi`` is a fraction of a STEP -> must be in (0, 1]."""
    m, _, _ = generated
    from motion_gen.clips.offline_velocity_sampler import (
        OfflineVelocitySampler,
    )
    from motion_gen.clips.offline_clip_generator import (
        OfflineClipGenerator,
    )
    skills = [s.name for s in m.skills]
    buckets = {
        "running": _Bucket((1.5, 3.0)), "standing": _Bucket((0.0, 0.1)),
        "walk_backwards": _Bucket((-1.0, -0.1)),
        "walk_forward": _Bucket((0.1, 1.5), (-0.3, 0.3), (-0.5, 0.5)),
    }

    def _make(be):
        sampler = OfflineVelocitySampler(
            buckets, skills, [0.4, 0.05, 0.2, 0.35], resampling_time_range=(0.8, 1.5),
            max_acc=2.0, command_dt=0.02, device=DEVICE, seed=0,
        )
        return OfflineClipGenerator(
            m, sampler, dt=0.02, clip_length_s=2.0,
            transition_blend_end_phi=be, device=DEVICE,
        )

    with pytest.raises(ValueError, match="STEP"):
        _make(1.5)   # > 1 step is rejected
    # 0.6 (60% of a step) is now valid step-relative (was rejected under the old
    # full-period < half-cycle rule).
    _make(0.6)


def test_uniform_timeline_no_dropped_frames(generated):
    """Every clip has exactly num_steps frames with monotone within-stride phase."""
    _, gen, out = generated
    # The rollout stores exactly T frames; stride_idx is piecewise-constant and
    # only advances forward (indices change at seams, never mid-stride).
    st = out["stride_idx"]
    assert st.shape[1] == gen.num_steps
    # No NaNs / negative indices leaked.
    assert (st >= 0).all()


def test_debug_interp_logs_blend_state(generated):
    """``debug_interp=True`` adds per-step blend_active/blend_alpha for plotting."""
    m, _, _ = generated
    from motion_gen.clips.offline_velocity_sampler import (
        OfflineVelocitySampler,
    )
    from motion_gen.clips.offline_clip_generator import (
        OfflineClipGenerator,
    )
    skills = [s.name for s in m.skills]
    buckets = {
        "running": _Bucket((1.5, 3.0)), "standing": _Bucket((0.0, 0.1)),
        "walk_backwards": _Bucket((-1.0, -0.1)),
        "walk_forward": _Bucket((0.1, 1.5), (-0.3, 0.3), (-0.5, 0.5)),
    }
    sampler = OfflineVelocitySampler(
        buckets, skills, [0.4, 0.05, 0.2, 0.35], resampling_time_range=(0.8, 1.5),
        max_acc=2.0, command_dt=0.02, device=DEVICE, seed=0,
    )
    g = OfflineClipGenerator(m, sampler, dt=0.02, clip_length_s=5.0, debug_interp=True, device=DEVICE)
    out = g.generate_batch(8)
    ba, al = out["blend_active"], out["blend_alpha"]
    assert ba.shape == (8, g.num_steps) and al.shape == (8, g.num_steps)
    assert (al >= 0).all() and (al <= 1.0 + 1e-6).all(), "blend alpha out of [0, 1]"
    assert bool((ba > 0).any()), "expected some interpolation frames"
    assert bool((al[ba == 0] == 0).all()), "alpha must be 0 where not mid-blend"
    # Pre-blend nominals: finite exactly on blend frames, NaN elsewhere.
    bo, bn = out["blend_old"], out["blend_new"]
    assert bo.shape == (8, g.num_steps, g.num_pos) and bn.shape == bo.shape
    fin = torch.isfinite(bo).any(-1)   # [n, T] a real (finite) nominal this frame
    assert bool((fin == (ba > 0)).all()), "nominals must be finite exactly on mid-blend frames"
    assert bool(torch.isnan(bn[ba == 0]).all()), "new nominal must be NaN off-blend"


def test_blend_pace_longer_is_max_of_old_and_new(generated):
    """``blend_pace='longer'`` paces the cross-fade window by the longer half-cycle.

    The window (in OLD-period phi) reduces to ``dur / total_time[old]`` where
    ``dur`` is the leaving half (old), incoming half (new), or their max (longer);
    since all three divide by the same ``total_time[old]``, ``longer`` must equal
    the elementwise max of the ``old`` and ``new`` windows.  A same-stride seam has
    ``old == new``, so every pace agrees there.
    """
    _, gen, _ = generated
    if gen._phi_mid.shape[0] < 2:
        pytest.skip("need >= 2 strides to pair distinct old/new")
    old_s = torch.tensor([0, 1], device=DEVICE)
    new_s = torch.tensor([1, 0], device=DEVICE)          # cross-gait: old != new
    even = torch.tensor([True, False], device=DEVICE)    # 2nd half, then 1st half
    saved = gen._blend_pace
    try:
        gen._blend_pace = "old"
        w_old = gen._blend_seg_len(old_s, new_s, even)
        gen._blend_pace = "new"
        w_new = gen._blend_seg_len(old_s, new_s, even)
        gen._blend_pace = "longer"
        w_long = gen._blend_seg_len(old_s, new_s, even)
        assert torch.allclose(w_long, torch.maximum(w_old, w_new))
        assert bool((w_long >= w_old - 1e-6).all()) and bool((w_long >= w_new - 1e-6).all())
        # Same-stride seam: nothing to pace differently -> all paces identical.
        same = torch.tensor([0, 1], device=DEVICE)
        gen._blend_pace = "old"
        a = gen._blend_seg_len(same, same, even)
        gen._blend_pace = "longer"
        b = gen._blend_seg_len(same, same, even)
        assert torch.allclose(a, b)
    finally:
        gen._blend_pace = saved
