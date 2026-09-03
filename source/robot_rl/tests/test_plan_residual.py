# Copyright 2026 Zachary Olkin. All rights reserved.

"""``FmTrackingCommand.apply_plan_residual`` -- the adjuster's write onto the live plan.

Pure buffer arithmetic, so these drive the real methods on a bare instance carrying only the
buffers each one reads. What they pin:

* **The write lands in ``_pred_window``, in place.** That buffer is the only one every
  consumer reads -- the tracker's ``reference`` observation, ``y_des`` and hence the CLF,
  ``served_tracking_error`` and hence the dense reward and the failure termination, the
  generator's own history row. Writing anywhere else (inside ``reference_window_obs``, say)
  would reach the observation alone and leave every graded quantity describing a plan nobody
  followed -- with no shape or width change to reveal it.
* **The rotation half comes back a rotation, and a BOUNDED one.** Adding six free numbers to
  an orthonormal pair leaves SO(3); Gram-Schmidt puts it back but says nothing about how far
  the frame turned, and the failure termination's tilt check is measured against this very
  reference.
* **``plan_residual`` reports what was SERVED, not what was asked for.** The request goes
  through a squash, a per-channel scale, a projection and an angle clamp; only the far end of
  that chain is the physical quantity the L2 reward is meant to price. Reading the action
  buffer instead would under-price a request the clamp cut down.
* **``_prev_served_node`` is re-taken.** ``_snapshot_and_infer`` latches it at the tail of the
  previous step from the RAW plan, so without the re-take the generator's history row for node
  0 is unadjusted while rows 1..K-1 are adjusted.
* **``plan_window`` indexes ABSOLUTELY**, so the adjuster can see the whole plan including the
  tail that ``reference_window_obs`` refuses as past the look-ahead horizon.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab.utils.math", exc_type=ImportError)

from robot_rl.flow_matching.transforms import (  # noqa: E402
    quat_to_rot6d,
    rot6d_to_matrix,
    so3_log,
)
from robot_rl.flow_matching.window import (  # noqa: E402
    RESIDUAL_MAX_TILT_RAD,
    residual_scale_vector,
    window_block_slices,
    window_dim,
)
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.fm_tracking_cmd import (  # noqa: E402
    FmTrackingCommand,
)

N_ENVS, NJ, NEE, F = 4, 29, 4, 62
BLOCKS = ("base_pose", "base_vel", "joint_pos")
D = window_dim(NJ, NEE, BLOCKS)
_ORI = window_block_slices(NJ, NEE, BLOCKS)["base_pose"].start + 3


class _BareFmCommand(FmTrackingCommand):
    """The real class with ``device`` / ``num_envs`` shadowed so a bare instance can exist."""

    device = torch.device("cpu")
    num_envs = N_ENVS
    _debug_vis_handle = None


def _stub(pred_idx: int = 0, valid: bool = True) -> FmTrackingCommand:
    """A bare command carrying only what the residual path reads."""
    cmd = object.__new__(_BareFmCommand)
    cmd.cfg = SimpleNamespace(fm_update_every=12, reference_source="fm")
    cmd.ctx = SimpleNamespace(
        F=F, D=D, nj=NJ, num_extra_frames=NEE, target_num_extra_frames=NEE,
        reference_blocks=BLOCKS,
    )
    cmd._fm_blocks = BLOCKS
    cmd._arange_n = torch.arange(N_ENVS)
    cmd._pred_idx = torch.full((N_ENVS,), pred_idx, dtype=torch.long)
    cmd._pred_valid = torch.full((N_ENVS,), valid, dtype=torch.bool)
    cmd._plan_residual = None
    cmd._prev_served_node = torch.zeros(N_ENVS, D)

    # A plausible plan: small values everywhere, with real rotations in the orientation half.
    gen = torch.Generator().manual_seed(11)
    window = torch.randn(N_ENVS, F, D, generator=gen) * 0.05
    quat = torch.randn(N_ENVS, F, 4, generator=gen)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    window[..., _ORI:_ORI + 6] = quat_to_rot6d(quat)
    cmd._pred_window = window.contiguous()
    return cmd


def _residual(amplitude: float, seed: int = 12) -> torch.Tensor:
    """A saturated-action residual in RAW units, the way the action term will build one."""
    gen = torch.Generator().manual_seed(seed)
    action = torch.tanh(torch.randn(N_ENVS, F, D, generator=gen) * amplitude)
    return action * torch.tensor(residual_scale_vector(NJ, NEE, BLOCKS))


def _tilt(adjusted: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    r_new = rot6d_to_matrix(adjusted[..., _ORI:_ORI + 6])
    r_old = rot6d_to_matrix(original[..., _ORI:_ORI + 6])
    return torch.linalg.norm(so3_log(r_new @ r_old.transpose(-1, -2)), dim=-1)


def test_the_residual_lands_in_the_plan_buffer_in_place() -> None:
    """The plan changes, and it is the SAME storage -- diagnostics hold views of it."""
    cmd = _stub()
    before = cmd._pred_window.clone()
    storage = cmd._pred_window.data_ptr()

    FmTrackingCommand.apply_plan_residual(cmd, _residual(0.5))

    assert cmd._pred_window.data_ptr() == storage, "rebound the buffer instead of writing it"
    assert not torch.allclose(cmd._pred_window, before), "the residual did not reach the plan"
    # Non-rotation channels move by exactly the requested amount; only the rotation half is
    # re-projected and bounded.
    delta = cmd._pred_window - before
    want = _residual(0.5)
    torch.testing.assert_close(delta[..., :_ORI], want[..., :_ORI])
    torch.testing.assert_close(delta[..., _ORI + 6:], want[..., _ORI + 6:])


def test_the_rotation_half_stays_a_bounded_rotation() -> None:
    """Orthonormal after the write, and never turned further than the angle cap."""
    cmd = _stub()
    before = cmd._pred_window.clone()
    FmTrackingCommand.apply_plan_residual(cmd, _residual(3.0))

    matrix = rot6d_to_matrix(cmd._pred_window[..., _ORI:_ORI + 6])
    eye = torch.eye(3).expand_as(matrix)
    torch.testing.assert_close(matrix @ matrix.transpose(-1, -2), eye, atol=1e-5, rtol=1e-5)
    assert float(_tilt(cmd._pred_window, before).max()) <= RESIDUAL_MAX_TILT_RAD + 1e-5


def test_plan_residual_reports_what_was_served_not_what_was_asked() -> None:
    """The reported residual equals the realised change, so the L2 term prices the serve.

    Asserted where the clamp BINDS, since that is the only regime where the two differ -- a
    request that survives untouched would make this test vacuous.
    """
    cmd = _stub()
    before = cmd._pred_window.clone()
    request = _residual(3.0)
    FmTrackingCommand.apply_plan_residual(cmd, request)

    reported = FmTrackingCommand.plan_residual(cmd)
    torch.testing.assert_close(reported, cmd._pred_window - before)

    ori = slice(_ORI, _ORI + 6)
    bound = _tilt(cmd._pred_window, before) >= RESIDUAL_MAX_TILT_RAD - 1e-4
    assert bool(bound.any()), "fixture must over-drive the rotation for this to mean anything"
    assert not torch.allclose(reported[..., ori][bound], request[..., ori][bound]), (
        "the reported residual equals the request even where the clamp cut it down"
    )


def test_an_env_with_no_live_plan_is_skipped() -> None:
    """No prediction means no correction -- and a zero reported residual, not a stale one."""
    cmd = _stub(valid=False)
    before = cmd._pred_window.clone()
    FmTrackingCommand.apply_plan_residual(cmd, _residual(1.0))

    torch.testing.assert_close(cmd._pred_window, before)
    assert float(FmTrackingCommand.plan_residual(cmd).abs().max()) == 0.0


def test_plan_residual_is_zero_before_any_is_applied() -> None:
    """The reward term reads this every step, including before the first residual."""
    cmd = _stub()
    residual = FmTrackingCommand.plan_residual(cmd)
    assert residual.shape == (N_ENVS, F, D)
    assert float(residual.abs().max()) == 0.0


@pytest.mark.parametrize("pred_idx", [0, 7])
def test_the_served_node_stash_is_retaken_from_the_adjusted_plan(pred_idx: int) -> None:
    """Otherwise the generator's history row for the served node is the discarded plan."""
    cmd = _stub(pred_idx=pred_idx)
    FmTrackingCommand.apply_plan_residual(cmd, _residual(1.0))

    expected = cmd._pred_window[torch.arange(N_ENVS), cmd._pred_idx]
    torch.testing.assert_close(cmd._prev_served_node, expected)
    assert cmd._prev_served_node.data_ptr() != cmd._pred_window.data_ptr(), (
        "the stash must be a clone; a view would follow the next residual"
    )


def test_a_mis_shaped_residual_is_refused() -> None:
    """Loudly, because a broadcastable-but-wrong shape would silently correct the wrong nodes."""
    cmd = _stub()
    with pytest.raises(ValueError, match=r"plan residual has shape"):
        FmTrackingCommand.apply_plan_residual(cmd, torch.zeros(N_ENVS, F - 1, D))
    with pytest.raises(ValueError, match=r"plan residual has shape"):
        FmTrackingCommand.apply_plan_residual(cmd, torch.zeros(N_ENVS, F, D - 1))


def test_plan_window_indexes_absolutely_and_covers_the_whole_horizon() -> None:
    """The adjuster sees the plan as a plan, including the tail the policy's window refuses.

    ``reference_window_obs`` resolves offsets against the served cursor and is bounded by
    ``F - 1 - (K - 1)`` so the look-ahead stays valid for a whole interval. The adjuster is
    correcting the plan itself, so it has no such horizon -- node ``F - 1`` is fair game.
    """
    cmd = _stub(pred_idx=5)
    assert FmTrackingCommand.plan_window(cmd).shape == (N_ENVS, F, D)

    idx = torch.tensor([0, 30, F - 1])
    picked = FmTrackingCommand.plan_window(cmd, idx)
    assert picked.shape == (N_ENVS, 3, D)
    for k, node in enumerate(idx.tolist()):
        # ABSOLUTE: node 0 is the head of the plan, not the served node at cursor 5.
        torch.testing.assert_close(picked[:, k], cmd._pred_window[:, node])


def test_the_flow_stash_refuses_to_read_before_it_is_enabled() -> None:
    """Zeros would read as a converged flow, which is plausible and wrong.

    The stash is opt-in because it is a whole window per env, so the failure mode is a caller
    that forgot to enable it -- and a silent zero field is exactly the shape of answer the
    adjuster cannot distinguish from a real one.
    """
    cmd = _stub()
    cmd._pred_flow = None
    with pytest.raises(RuntimeError, match=r"enable_flow_stash"):
        FmTrackingCommand.plan_flow(cmd)


def test_enabling_the_flow_stash_is_idempotent() -> None:
    """Allocated once, at the plan's shape; a second call keeps the buffer it has."""
    cmd = _stub()
    cmd._pred_flow = None
    FmTrackingCommand.enable_flow_stash(cmd)
    first = cmd._pred_flow
    assert first.shape == cmd._pred_window.shape
    assert float(first.abs().max()) == 0.0

    FmTrackingCommand.enable_flow_stash(cmd)
    assert cmd._pred_flow is first


def test_applying_a_zero_residual_changes_nothing() -> None:
    """The identity the zero-residual rollout test depends on, at the buffer level."""
    cmd = _stub()
    before = cmd._pred_window.clone()
    FmTrackingCommand.apply_plan_residual(cmd, torch.zeros(N_ENVS, F, D))
    torch.testing.assert_close(cmd._pred_window, before, atol=0.0, rtol=0.0)
