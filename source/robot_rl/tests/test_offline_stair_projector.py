# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the offline single-span stair projector.

Pure tensor-level tests on :func:`make_offline_stair_projector` in
``robot_rl...traj_tracking.clips.offline_stair_projector`` — no IsaacLab, no
sim.  A hand-built 4-tread staircase (in local-corner coordinates) exercises
the tread / entry / exit routing, including the defining property that the
query **z** picks the step even when the ankle xy sits over a lower tread.
"""

from __future__ import annotations

import torch

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.clips.offline_stair_projector import (
    make_offline_multi_stair_projector,
    make_offline_stair_projector,
)

DEVICE = "cpu"

# A staircase: 0.5 m start platform, then 4 treads of depth 0.30 / rise 0.16.
# Tread i center x = 0.5 + (i + 0.5) * 0.30;  tread top z = (i + 1) * 0.16.
STEP_DEPTH = 0.30
STEP_HEIGHT = 0.16
ENTRY_Z = 0.0
SPAN_XMIN = 0.0
SPAN_XMAX = 1.7  # 0.5 platform + 4 * 0.30 treads
STAIR_TOP_CENTERS = torch.tensor(
    [
        [0.65, 0.0, 0.16],
        [0.95, 0.0, 0.32],
        [1.25, 0.0, 0.48],
        [1.55, 0.0, 0.64],
    ],
    device=DEVICE,
)


def _projector():
    # z_bias_frac=0.0 -> raw z-nearest routing (these tests feed z AT the tread
    # tops and assert that tread).  The default projector biases the query z
    # down by half a step; the bias is covered by the dedicated tests below.
    return make_offline_stair_projector(
        STAIR_TOP_CENTERS, STEP_DEPTH, ENTRY_Z, SPAN_XMIN, SPAN_XMAX, z_bias_frac=0.0
    )


def test_foot_on_a_tread_snaps_to_that_tread_center():
    """A foot at a tread's height snaps to that tread's center x and top z."""
    project = _projector()
    # Slightly noisy z near tread 2 (top 0.48), xy near tread 2.
    pts = torch.tensor([[1.24, 0.05, 0.47]], device=DEVICE)
    out = project(pts)
    assert torch.allclose(out[0, 0], torch.tensor(1.25), atol=1e-5)  # tread center x
    assert torch.allclose(out[0, 2], torch.tensor(0.48), atol=1e-5)  # tread top z


def test_z_selects_step_even_when_xy_sits_over_a_lower_tread():
    """The query z, not xy, picks the step (the whole point of z-routing)."""
    project = _projector()
    # xy over tread 1 (x=0.95) but height of tread 2 (z=0.48) -> tread 2.
    pts = torch.tensor([[0.95, 0.0, 0.48]], device=DEVICE)
    out = project(pts)
    assert torch.allclose(out[0, 0], torch.tensor(1.25), atol=1e-5)
    assert torch.allclose(out[0, 2], torch.tensor(0.48), atol=1e-5)


def test_foot_on_start_platform_keeps_its_x_at_entry_z():
    """In-span foot at entry height stays put (start-platform behavior)."""
    project = _projector()
    pts = torch.tensor([[0.30, 0.12, 0.02]], device=DEVICE)  # on the platform
    out = project(pts)
    assert torch.allclose(out[0, 0], torch.tensor(0.30), atol=1e-5)  # x kept
    assert torch.allclose(out[0, 1], torch.tensor(0.12), atol=1e-5)  # y kept
    assert torch.allclose(out[0, 2], torch.tensor(0.0), atol=1e-5)   # entry z


def test_foot_approaching_from_outside_snaps_before_the_first_riser():
    """Out-of-span foot at entry height snaps to span_xmin - depth/2."""
    project = _projector()
    pts = torch.tensor([[-0.20, 0.0, 0.0]], device=DEVICE)
    out = project(pts)
    assert torch.allclose(out[0, 0], torch.tensor(SPAN_XMIN - 0.5 * STEP_DEPTH), atol=1e-5)
    assert torch.allclose(out[0, 2], torch.tensor(0.0), atol=1e-5)


def test_foot_past_the_span_snaps_to_exit_level():
    """A foot past span_xmax snaps to the fictitious exit tread at exit z."""
    project = _projector()
    pts = torch.tensor([[2.0, 0.3, 0.64]], device=DEVICE)
    out = project(pts)
    assert torch.allclose(out[0, 0], torch.tensor(SPAN_XMAX + 0.5 * STEP_DEPTH), atol=1e-5)
    assert torch.allclose(out[0, 2], torch.tensor(0.64), atol=1e-5)  # last tread top


def test_batched_mixed_cases():
    """A batch of all four routings resolves independently per row."""
    project = _projector()
    pts = torch.tensor(
        [
            [1.24, 0.0, 0.47],  # tread 2
            [0.30, 0.0, 0.02],  # start platform (entry, in-span)
            [-0.20, 0.0, 0.0],  # approaching (entry, out-of-span)
            [2.00, 0.0, 0.64],  # past exit
        ],
        device=DEVICE,
    )
    out = project(pts)
    assert out.shape == (4, 3)
    assert torch.allclose(out[:, 2], torch.tensor([0.48, 0.0, 0.0, 0.64]), atol=1e-5)


def test_rejects_wrong_shape():
    project = _projector()
    try:
        project(torch.zeros(3, device=DEVICE))
    except ValueError:
        return
    raise AssertionError("expected ValueError on non-(N,3) input")


# --- Multi-span projector: two staircases with an inter-band flat -----------
# Band 0: 1 m lead-in, 3 treads d=0.3 h=0.15 -> tops 0.15/0.30/0.45, span [1.0,1.9].
# Inter-flat [1.9, 3.0] at z=0.45.  Band 1: 3 treads -> tops 0.60/0.75/0.90,
# span [3.0, 3.9], entry_z 0.45.
_SPANS = [
    {"stair_top_centers": torch.tensor([[1.15, 0.0, 0.15], [1.45, 0.0, 0.30], [1.75, 0.0, 0.45]]),
     "step_depth": 0.3, "entry_z": 0.0, "span_xmin": 1.0, "span_xmax": 1.9},
    {"stair_top_centers": torch.tensor([[3.15, 0.0, 0.60], [3.45, 0.0, 0.75], [3.75, 0.0, 0.90]]),
     "step_depth": 0.3, "entry_z": 0.45, "span_xmin": 3.0, "span_xmax": 3.9},
]


def test_multi_span_routes_each_point_to_its_band():
    project = make_offline_multi_stair_projector(_SPANS, z_bias_frac=0.0)  # raw routing
    pts = torch.tensor([
        [0.50, 0.0, 0.00],   # before band 0 -> band 0 entry (span_xmin0 - d/2 = 0.85, z 0)
        [1.44, 0.0, 0.29],   # on band 0 tread 1 -> (1.45, 0.30)
        [2.50, 0.0, 0.45],   # inter-flat -> APPROACHING band 1 -> band 1 entry (2.85, 0.45)
        [3.44, 0.0, 0.74],   # on band 1 tread 1 -> (3.45, 0.75)
        [4.50, 0.0, 0.90],   # past band 1 -> band 1 exit (3.9 + d/2 = 4.05, 0.90)
    ])
    out = project(pts)
    exp_x = torch.tensor([0.85, 1.45, 2.85, 3.45, 4.05])
    exp_z = torch.tensor([0.00, 0.30, 0.45, 0.75, 0.90])
    assert torch.allclose(out[:, 0], exp_x, atol=1e-4), out[:, 0]
    assert torch.allclose(out[:, 2], exp_z, atol=1e-4), out[:, 2]


def test_multi_span_single_band_matches_single_projector():
    multi = make_offline_multi_stair_projector([_SPANS[0]], z_bias_frac=0.0)
    single = make_offline_stair_projector(
        _SPANS[0]["stair_top_centers"], 0.3, 0.0, 1.0, 1.9, z_bias_frac=0.0,
    )
    pts = torch.tensor([[1.44, 0.0, 0.29], [0.5, 0.0, 0.0], [2.5, 0.0, 0.45]])
    assert torch.allclose(multi(pts), single(pts), atol=1e-6)


def test_multi_span_rejects_unordered_spans():
    try:
        make_offline_multi_stair_projector([_SPANS[1], _SPANS[0]])
    except ValueError:
        return
    raise AssertionError("expected ValueError on unordered spans")


# --- z-bias: don't skip the tread the foot actually rests on ----------------
# The offline re-anchor feeds the stance ANKLE, which sits ~half a step ABOVE
# its tread, so the raw z lands near the midpoint between two tread tops and
# rounds UP to the step ahead -> a per-stride forward teleport.  The default
# downward z-bias makes the pick round to the resting tread instead.


def test_z_bias_snaps_to_resting_tread_not_the_step_ahead():
    ankle = torch.tensor([[0.95, 0.0, 0.41]], device=DEVICE)  # ~ above tread 1 (top 0.32)
    raw = make_offline_stair_projector(
        STAIR_TOP_CENTERS, STEP_DEPTH, ENTRY_Z, SPAN_XMIN, SPAN_XMAX, z_bias_frac=0.0
    )
    biased = make_offline_stair_projector(  # default z_bias_frac = 0.5
        STAIR_TOP_CENTERS, STEP_DEPTH, ENTRY_Z, SPAN_XMIN, SPAN_XMAX
    )
    # Raw z-nearest rounds UP to tread 2 (the teleport).
    assert torch.allclose(raw(ankle)[0], torch.tensor([1.25, 0.0, 0.48]), atol=1e-5)
    # Biased snaps to the resting tread 1 (center x 0.95, top z 0.32).
    out = biased(ankle)
    assert torch.allclose(out[0, 0], torch.tensor(0.95), atol=1e-5)
    assert torch.allclose(out[0, 2], torch.tensor(0.32), atol=1e-5)


def test_z_bias_does_not_sink_a_planted_foot():
    """An ankle a half-step above its tread still snaps to THAT tread, not below."""
    biased = make_offline_stair_projector(
        STAIR_TOP_CENTERS, STEP_DEPTH, ENTRY_Z, SPAN_XMIN, SPAN_XMAX
    )
    ankle = torch.tensor([[1.25, 0.0, 0.48 + 0.5 * STEP_HEIGHT]], device=DEVICE)  # above tread 2
    out = biased(ankle)
    assert torch.allclose(out[0, 0], torch.tensor(1.25), atol=1e-5)
    assert torch.allclose(out[0, 2], torch.tensor(0.48), atol=1e-5)


def test_z_bias_preserves_flat_approach_to_entry():
    """The bias must not knock a flat, out-of-span approach off the entry level."""
    biased = make_offline_stair_projector(
        STAIR_TOP_CENTERS, STEP_DEPTH, ENTRY_Z, SPAN_XMIN, SPAN_XMAX
    )
    out = biased(torch.tensor([[-0.20, 0.0, 0.0]], device=DEVICE))
    assert torch.allclose(out[0, 0], torch.tensor(SPAN_XMIN - 0.5 * STEP_DEPTH), atol=1e-5)
    assert torch.allclose(out[0, 2], torch.tensor(0.0), atol=1e-5)
