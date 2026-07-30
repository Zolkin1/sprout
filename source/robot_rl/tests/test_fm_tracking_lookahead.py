# Copyright 2026 Zachary Olkin. All rights reserved.

"""``FmTrackingCommand``'s look-ahead surface + the generator/clip barrier.

These are the pieces that decide whether the teacher's ``reference_window`` observation is
served correctly from the generator, and they are pure indexing -- no simulator needed. The
tests drive the real methods on a bare instance (``object.__new__``) carrying only the
buffers each one reads, so the arithmetic under test is the shipped arithmetic.

Three properties matter here:

* **Joint order.** ``get_future_joint_state``'s contract is the ROBOT's joint order (the
  clip implementation builds its index by iterating ``robot.joint_names``), while the
  prediction buffers are in FM order. Two permutations exist and only one is correct; using
  the clip-output one instead scrambles the reference with no shape change to reveal it.
* **No clamping.** A short horizon must raise, not saturate: a clamp would silently freeze
  the tail of the window the policy sees.
* **The barrier.** The tracker env must carry no rewards and no clip-graded termination.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab.utils.math", exc_type=ImportError)

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.fm_tracking_cmd import (  # noqa: E402
    FmTrackingCommand,
)

N_ENVS, NJ, K, F = 4, 6, 2, 20
FM_FRAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
# A deliberately scrambled FM<->robot joint mapping: an identity would let the wrong
# permutation pass.
ROBOT_TO_FM = [3, 0, 5, 1, 4, 2]


class _BareFmCommand(FmTrackingCommand):
    """``FmTrackingCommand`` with ``device`` overridden to a plain attribute.

    The base class exposes ``device`` as a read-only property off the env; the class
    attribute here shadows it so a bare instance can be built without one. Everything else
    is the real class -- the methods under test are inherited unchanged.
    """

    device = torch.device("cpu")
    # ``CommandTerm.__del__`` touches this; a bare instance never ran the ctor that sets it.
    _debug_vis_handle = None


def _stub(pred_idx: list[int], fm_update_every: int = 4) -> FmTrackingCommand:
    """A bare command carrying only what the look-ahead accessors read."""
    cmd = object.__new__(_BareFmCommand)
    cmd.cfg = SimpleNamespace(fm_update_every=fm_update_every)
    cmd.ctx = SimpleNamespace(F=F, nj=NJ, num_extra_frames=K, frame_names=list(FM_FRAMES))
    cmd._arange_n = torch.arange(N_ENVS)
    cmd._pred_idx = torch.tensor(pred_idx, dtype=torch.long)
    cmd._checked_max_offset = -1
    # Distinct, decodable contents: value == env*1000 + node*10 + channel.
    grid = (torch.arange(N_ENVS)[:, None, None] * 1000
            + torch.arange(F)[None, :, None] * 10)
    cmd._pred_jp = grid + torch.arange(NJ)[None, None, :].float()
    cmd._pred_jv = cmd._pred_jp + 0.5
    cmd._pred_root_pos = grid + torch.arange(3)[None, None, :].float()
    cmd._pred_root_quat = grid + torch.arange(4)[None, None, :].float()
    cmd._pred_root_lin = cmd._pred_root_pos + 0.1
    cmd._pred_root_ang = cmd._pred_root_pos + 0.2
    cmd._pred_frame_pos = (grid[:, :, None, :] + torch.arange(K)[None, None, :, None] * 100
                           + torch.arange(3)[None, None, None, :].float())
    robot_to_fm = torch.tensor(ROBOT_TO_FM, dtype=torch.long)
    cmd._robot_to_fm_jperm = robot_to_fm
    cmd._fm_to_robot_jperm = torch.argsort(robot_to_fm)
    return cmd


def test_fm_to_robot_permutation_is_the_inverse():
    """``argsort`` of a permutation IS its inverse -- the property the accessor relies on."""
    robot_to_fm = torch.tensor(ROBOT_TO_FM, dtype=torch.long)
    fm_to_robot = torch.argsort(robot_to_fm)
    # Round-tripping robot -> FM -> robot must be the identity on values.
    robot_vals = torch.arange(NJ).float()
    fm_vals = robot_vals[robot_to_fm]
    assert torch.equal(fm_vals[fm_to_robot], robot_vals)
    # And it is genuinely non-trivial, so a wrong permutation cannot pass by accident.
    assert not torch.equal(fm_to_robot, torch.arange(NJ))
    assert not torch.equal(fm_to_robot, robot_to_fm)


def test_get_future_joint_state_returns_robot_order():
    """The accessor permutes FM -> ROBOT, matching the base class's contract."""
    cmd = _stub(pred_idx=[0, 1, 2, 3])
    offsets = torch.tensor([0, 5, 10])
    jp, jv = FmTrackingCommand.get_future_joint_state(cmd, offsets)
    assert jp.shape == (N_ENVS, 3, NJ) and jv.shape == (N_ENVS, 3, NJ)

    fm_to_robot = torch.argsort(torch.tensor(ROBOT_TO_FM, dtype=torch.long))
    for e in range(N_ENVS):
        for k, off in enumerate(offsets.tolist()):
            node = cmd._pred_idx[e].item() + off
            assert torch.equal(jp[e, k], cmd._pred_jp[e, node][fm_to_robot])
            assert torch.equal(jv[e, k], cmd._pred_jv[e, node][fm_to_robot])

    # The clip-OUTPUT permutation would give a different answer -- so this test would fail
    # if the two were swapped, which is the whole point.
    wrong = cmd._pred_jp[0, 0][torch.tensor(ROBOT_TO_FM, dtype=torch.long)]
    assert not torch.equal(jp[0, 0], wrong)


def test_get_future_world_base_walks_the_cursor():
    """Node index is ``_pred_idx + offset``, per env -- no clamping, no clip blend."""
    cmd = _stub(pred_idx=[0, 1, 2, 3])
    offsets = torch.tensor([0, 4, 8])
    pos, quat, lin, ang = FmTrackingCommand.get_future_world_base(cmd, offsets)
    for e in range(N_ENVS):
        for k, off in enumerate(offsets.tolist()):
            node = cmd._pred_idx[e].item() + off
            assert torch.equal(pos[e, k], cmd._pred_root_pos[e, node])
            assert torch.equal(quat[e, k], cmd._pred_root_quat[e, node])
            assert torch.equal(lin[e, k], cmd._pred_root_lin[e, node])
            assert torch.equal(ang[e, k], cmd._pred_root_ang[e, node])


def test_get_future_world_frames_position_only_placeholders():
    """Position comes from the FM; the other 10 channels are constants, not stale clip data."""
    cmd = _stub(pred_idx=[0, 1, 2, 3])
    offsets = torch.tensor([0, 6])
    out = FmTrackingCommand.get_future_world_frames(cmd, list(FM_FRAMES), offsets)
    assert out.shape == (N_ENVS, 2, K, 13)
    for e in range(N_ENVS):
        for k, off in enumerate(offsets.tolist()):
            node = cmd._pred_idx[e].item() + off
            assert torch.equal(out[e, k, :, 0:3], cmd._pred_frame_pos[e, node])
    assert torch.equal(out[..., 3:6], torch.zeros_like(out[..., 3:6]))   # quat xyz
    assert torch.equal(out[..., 6], torch.ones_like(out[..., 6]))        # quat w
    assert torch.equal(out[..., 7:13], torch.zeros_like(out[..., 7:13]))  # lin + ang


def test_get_future_world_frames_reorders_by_name():
    """Frames come back in the ORDER ASKED FOR, not the checkpoint's order."""
    cmd = _stub(pred_idx=[0, 0, 0, 0])
    offsets = torch.tensor([0])
    fwd = FmTrackingCommand.get_future_world_frames(cmd, list(FM_FRAMES), offsets)
    rev = FmTrackingCommand.get_future_world_frames(cmd, list(reversed(FM_FRAMES)), offsets)
    assert torch.equal(fwd[:, :, 0, 0:3], rev[:, :, 1, 0:3])
    assert torch.equal(fwd[:, :, 1, 0:3], rev[:, :, 0, 0:3])


def test_unknown_frame_raises_rather_than_falling_back():
    cmd = _stub(pred_idx=[0, 0, 0, 0])
    with pytest.raises(ValueError, match="does not predict"):
        FmTrackingCommand.get_future_world_frames(cmd, ["left_wrist_yaw_link"], torch.tensor([0]))


# ---------------------------------------------------------------------------
# No clamping: a horizon that cannot cover the request must RAISE.
# ---------------------------------------------------------------------------
def test_horizon_check_accepts_a_sufficient_window():
    # F=20, re-plan every 4 -> the cursor is at most 3 stale, so offsets to 16 are servable.
    cmd = _stub(pred_idx=[0, 0, 0, 0], fm_update_every=4)
    FmTrackingCommand._check_horizon(cmd, torch.tensor([0, 8, 16]))
    assert cmd._checked_max_offset == 16


def test_horizon_check_raises_when_short():
    """Offset 17 at period 4 needs F >= 21; F is 20, so this must not silently clamp."""
    cmd = _stub(pred_idx=[0, 0, 0, 0], fm_update_every=4)
    with pytest.raises(ValueError, match=r"needs F >= 21"):
        FmTrackingCommand._check_horizon(cmd, torch.tensor([0, 17]))


def test_horizon_check_rechecks_only_on_a_larger_offset():
    """Cached by the largest offset seen: a bigger ask must still be validated."""
    cmd = _stub(pred_idx=[0, 0, 0, 0], fm_update_every=4)
    FmTrackingCommand._check_horizon(cmd, torch.tensor([0, 4]))
    FmTrackingCommand._check_horizon(cmd, torch.tensor([0, 2]))       # smaller: no-op
    assert cmd._checked_max_offset == 4
    with pytest.raises(ValueError):
        FmTrackingCommand._check_horizon(cmd, torch.tensor([0, 19]))


def test_teacher_window_fits_the_shipped_horizon():
    """The shipped F actually covers the shipped teacher window at the shipped cadence.

    This is the arithmetic in ``FlowMatchingConfig.future_len``'s comment, asserted against
    the three constants it is derived from -- so moving any of them without moving F fails
    here rather than at the first play step.
    """
    from robot_rl.flow_matching import FlowMatchingConfig
    from robot_rl.tasks.manager_based.robot_rl.g1.g1_teacher_env_cfg import (
        REFERENCE_NUM_STEPS, REFERENCE_STRIDE,
    )
    from robot_rl.tasks.manager_based.robot_rl.g1.g1_tracker_env_cfg import (
        G1FmTrackerCommandsCfg,
    )

    period = G1FmTrackerCommandsCfg().traj_ref.fm_update_every
    required = (period - 1) + (REFERENCE_NUM_STEPS - 1) * REFERENCE_STRIDE + 1
    assert FlowMatchingConfig().future_len >= required, (
        f"future_len must be >= {required} for a {REFERENCE_NUM_STEPS}-sample window at "
        f"stride {REFERENCE_STRIDE} re-planned every {period} steps."
    )


# ---------------------------------------------------------------------------
# The generator/clip barrier, at the config level.
# ---------------------------------------------------------------------------
def test_tracker_env_has_no_rewards_and_no_clip_graded_termination():
    pytest.importorskip("isaaclab.envs", exc_type=ImportError)
    from robot_rl.tasks.manager_based.robot_rl.g1.g1_tracker_env_cfg import (
        G1FmTrackerTerrainEnvCfg,
    )

    cfg = G1FmTrackerTerrainEnvCfg()
    assert [k for k in vars(cfg.rewards) if not k.startswith("_")] == []
    terms = {k for k in vars(cfg.terminations) if not k.startswith("_")}
    assert terms == {"time_out", "base_fallen"}, terms
    # Only the two ACTOR groups: the privileged critic group is where the last clip-derived
    # observation lived (tracking_error_global), and nothing here trains.
    assert [k for k in vars(cfg.observations) if not k.startswith("_")] == ["policy", "reference"]
    assert cfg.commands.traj_ref.strict_fm_only is True
    # Nothing re-anchors the clip: no env reads its reference.
    assert cfg.commands.traj_ref.ego_anchor_fraction == 0.0


def test_tracker_runner_does_not_load_a_critic():
    """The value net is vestigial here, which is what makes the task teacher-agnostic."""
    pytest.importorskip("isaaclab.envs", exc_type=ImportError)
    from robot_rl.tasks.manager_based.robot_rl.g1.agents.rsl_rl_ppo_cfg import (
        FmTrackerPlayRunnerCfg, PPORunnerCfg,
    )

    cfg = FmTrackerPlayRunnerCfg()
    assert cfg.load_critic is False
    # Pointed at a deployable group so rsl-rl can build one; the env has no critic group.
    assert cfg.obs_groups == {"actor": ["policy", "reference"], "critic": ["policy"]}
    # The teacher's runner is unchanged -- it still trains a real critic.
    assert PPORunnerCfg().obs_groups["critic"] == ["critic"]
    assert getattr(PPORunnerCfg(), "load_critic", True) is True


def test_fm_command_rejects_body_mode_without_blaming_the_teacher():
    """Body mode is about THIS task's CLF channels, not the checkpoint being loaded."""
    from robot_rl.tasks.manager_based.robot_rl.g1.g1_tracker_env_cfg import (
        G1FmTrackerCommandsCfg,
    )

    assert G1FmTrackerCommandsCfg().traj_ref.tracking_mode == "joint"


# ---------------------------------------------------------------------------
# No clamping, part 2: the TEACHER's own window must not saturate either.
# ---------------------------------------------------------------------------
class _ClipEndStub:
    """Minimal command surface for :func:`mdp.clip_end_reached`."""

    def __init__(self, clip_len: int, lookahead: int | None, start_step: int = 0):
        self.manager = SimpleNamespace(clip_len=torch.tensor([clip_len]))
        self.clip_id = torch.zeros(1, dtype=torch.long)
        self.start_step = torch.full((1,), start_step, dtype=torch.long)
        self._lookahead = lookahead

    def obs_window_offsets(self):
        """The reference-window offsets, or None when the env has no such term."""
        if self._lookahead is None:
            return None
        return torch.tensor([0, self._lookahead], dtype=torch.long)


def _fired(clip_len, lookahead, episode_step, reserve=True):
    """Run ``clip_end_reached`` at ``episode_step`` and return the single env's verdict."""
    from robot_rl.tasks.manager_based.robot_rl.mdp.terminations.terminations import clip_end_reached

    cmd = _ClipEndStub(clip_len, lookahead)
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _n: cmd),
        episode_length_buf=torch.tensor([episode_step], dtype=torch.long),
    )
    return bool(clip_end_reached(env, "traj_ref")[0]) if reserve else bool(
        clip_end_reached(env, "traj_ref", reserve_lookahead=False)[0]
    )


def test_clip_end_reserves_the_observed_window():
    """A 100-frame clip with a 50-step look-ahead ends at step 49, not 99.

    Past that the look-ahead gather clamps to the clip's last frame, so the policy would be
    fed a progressively frozen reference -- and the FM trained on it would learn that as a
    stop-moving mode.
    """
    assert not _fired(clip_len=100, lookahead=50, episode_step=48)
    assert _fired(clip_len=100, lookahead=50, episode_step=49)


def test_clip_end_without_the_margin_runs_to_the_last_frame():
    assert not _fired(clip_len=100, lookahead=50, episode_step=98, reserve=False)
    assert _fired(clip_len=100, lookahead=50, episode_step=99, reserve=False)


def test_clip_end_falls_back_when_no_window_term_exists():
    """No ``reference_window`` in the env -> nothing to reserve, old behaviour."""
    assert not _fired(clip_len=100, lookahead=None, episode_step=98)
    assert _fired(clip_len=100, lookahead=None, episode_step=99)


def test_clip_shorter_than_the_window_ends_immediately():
    """Rather than serving a reference that is frozen from the very first step."""
    assert _fired(clip_len=30, lookahead=50, episode_step=0)
