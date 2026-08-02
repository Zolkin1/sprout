# Copyright 2026 Zachary Olkin. All rights reserved.

"""``FmTrackingCommand``'s look-ahead surface + the generator/clip barrier.

These are the pieces that decide whether the teacher's ``reference_window`` observation is
served correctly from the generator, and they are pure indexing -- no simulator needed. The
tests drive the real methods on a bare instance (``object.__new__``) carrying only the
buffers each one reads, so the arithmetic under test is the shipped arithmetic.

Four properties matter here:

* **Joint order.** ``get_future_joint_state``'s contract is the ROBOT's joint order (the
  clip implementation builds its index by iterating ``robot.joint_names``), while the
  prediction buffers are in FM order. Two permutations exist and only one is correct; using
  the clip-output one instead scrambles the reference with no shape change to reveal it.
* **No clamping.** A short horizon must raise, not saturate: a clamp would silently freeze
  the tail of the window the policy sees.
* **The measured side runs.** ``_update_command`` must produce ``y_act`` and the CLF on
  EVERY step, whatever the checkpoint's ``history_source``. It once did not: the
  ``compute_measured_output`` + ``compute_vdot`` block was left below the early ``return``
  in ``_stash_applied_reference``, so on the default ``history_source="measured"`` it never
  ran -- ``y_act`` stayed at its initialised zeros and ``V`` at 0.0. Nothing crashed and the
  policy still acted (the actor reads the reference window, not ``y_act``), so every
  tracking number the task reported was silently wrong.
* **The barrier.** Both tracker envs must carry no rewards and no clip-graded termination.
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
    """``FmTrackingCommand`` with ``device`` / ``num_envs`` overridden to plain attributes.

    The base class exposes both as read-only properties off the env; the class attributes
    here shadow them so a bare instance can be built without one. Everything else is the
    real class -- the methods under test are inherited unchanged.
    """

    device = torch.device("cpu")
    num_envs = N_ENVS
    # ``CommandTerm.__del__`` touches this; a bare instance never ran the ctor that sets it.
    _debug_vis_handle = None


def _stub(pred_idx: list[int], fm_update_every: int = 4) -> FmTrackingCommand:
    """A bare command carrying only what the look-ahead accessors read."""
    cmd = object.__new__(_BareFmCommand)
    cmd.cfg = SimpleNamespace(fm_update_every=fm_update_every, reference_source="fm")
    cmd.ctx = SimpleNamespace(F=F, nj=NJ, num_extra_frames=K, frame_names=list(FM_FRAMES))
    # No diagnostic override in force: the accessors answer for cfg.reference_source.
    cmd._serve_override = None
    cmd.fm_enabled = True
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
# The measured side + the CLF must run on every step, for every history_source.
# ---------------------------------------------------------------------------
# Wide enough for the real channel layout the window test scatters into: base + K ee
# frames (pos 3 + ori 4 each), the NJ joints, and one channel the generator never writes.
POS_W, VEL_W = 30, 26


#: What the stubbed CLIP half writes into ``y_des`` -- a value the generator half can never
#: produce, so "which source is applied" is decidable by inspection.
CLIP_FILL = 9.0


def _update_stub(history_source: str, reference_source: str = "fm") -> FmTrackingCommand:
    """A bare command with everything ``_update_command`` touches, stubbed and recorded.

    The CLIP half is stubbed to a constant fill (the real one is the inherited clip path,
    covered by ``test_segment_placement.py``); the GENERATOR half and the source selection
    run for real.

    Args:
        history_source: The checkpoint's history source to simulate.
        reference_source: Which reference the command should apply.

    Returns:
        The instance; ``cmd.calls`` records which stubbed collaborators fired.
    """
    cmd = object.__new__(_BareFmCommand)
    cmd.calls: list[str] = []
    cmd.cfg = SimpleNamespace(fm_update_every=4, strict_fm_only=True,
                              reference_source=reference_source)
    cmd.ctx = SimpleNamespace(F=F, nj=NJ, num_extra_frames=K, frame_names=list(FM_FRAMES),
                              history_source=history_source)
    cmd._resampling = False
    cmd._building = None
    cmd._serve_override = None
    cmd.fm_enabled = True
    cmd.y_des = torch.zeros(N_ENVS, POS_W)
    cmd.dy_des = torch.zeros(N_ENVS, VEL_W)
    cmd.y_act = torch.zeros(N_ENVS, POS_W)
    cmd.dy_act = torch.zeros(N_ENVS, VEL_W)
    cmd._wf_world_frames = torch.zeros(N_ENVS, 1 + K, 13)
    cmd._joint_slots_pos = torch.arange(NJ)
    cmd._joint_slots_vel = torch.arange(NJ)
    cmd._fm_jperm = torch.arange(NJ)
    # One channel the FM does not write, so the strict_fm_only fill is exercised too.
    cmd._unwritten_pos = torch.tensor([POS_W - 1])
    cmd._unwritten_vel = torch.tensor([VEL_W - 1])
    cmd._zeros3 = torch.zeros(N_ENVS, 3)
    cmd._ident_quat = torch.zeros(N_ENVS, 4)
    # As _post_init leaves them: no reference applied yet.
    cmd._applied_jp = cmd._applied_jv = cmd._applied_frames = None

    def _clip_half():
        cmd.calls.append("clip")
        cmd.y_des.fill_(CLIP_FILL)
        cmd.dy_des.fill_(CLIP_FILL)
        cmd._y_des_clip = cmd.y_des.clone()
        cmd._dy_des_clip = cmd.dy_des.clone()
        cmd._clip_wf_frames = cmd._wf_world_frames

    cmd._build_clip_reference = _clip_half
    cmd._advance_step = lambda: cmd.calls.append("advance")
    cmd._snapshot_and_infer = lambda: cmd.calls.append("infer")
    cmd._write_world_des = lambda frames: cmd.calls.append("write_des")
    cmd._fm_current_outputs = lambda: (
        torch.zeros(N_ENVS, 13), torch.zeros(N_ENVS, K, 13),
        torch.zeros(N_ENVS, NJ), torch.zeros(N_ENVS, NJ),
    )

    def _measured(pos, quat):
        cmd.calls.append("measured")
        cmd.y_act = torch.full((N_ENVS, POS_W), 2.0)      # a stand-in for the real read
        cmd.dy_act = torch.full((N_ENVS, VEL_W), 3.0)

    cmd.compute_measured_output = _measured
    cmd.clf = SimpleNamespace(compute_vdot=lambda ya, yd, dya, dyd: (
        cmd.calls.append("clf") or (torch.full((N_ENVS,), -1.0), torch.full((N_ENVS,), 7.0))
    ))
    return cmd


@pytest.mark.parametrize("history_source", ["measured", "reference_partial"])
def test_update_command_always_computes_measured_output_and_clf(history_source):
    """The regression: this block used to be stranded inside ``_stash_applied_reference``.

    Below that method's ``history_source != "reference_partial"`` early return, so on the
    DEFAULT source it never ran and every tracking metric read zero.
    """
    cmd = _update_stub(history_source)
    FmTrackingCommand._update_command(cmd)

    assert "measured" in cmd.calls, f"compute_measured_output never ran ({history_source})"
    assert "clf" in cmd.calls, f"the CLF never ran ({history_source})"
    # Order matters: the CLF grades the measured output produced this step, not last step's.
    assert cmd.calls.index("measured") < cmd.calls.index("clf")
    assert torch.equal(cmd.v, torch.full((N_ENVS,), 7.0))
    assert torch.equal(cmd.vdot, torch.full((N_ENVS,), -1.0))


def test_applied_reference_is_stashed_only_for_a_partially_autoregressive_checkpoint():
    """The stash itself stays conditional -- that part of the early return was correct."""
    measured = _update_stub("measured")
    FmTrackingCommand._update_command(measured)
    assert measured._applied_jp is None

    partial = _update_stub("reference_partial")
    FmTrackingCommand._update_command(partial)
    assert partial._applied_jp is not None
    assert partial._applied_frames is not None


def test_strict_fm_only_nans_run_before_the_clf():
    """So a graded channel the FM does not write shows up as a NaN ``V``, not a plausible one."""
    cmd = _update_stub("measured")
    FmTrackingCommand._update_command(cmd)
    assert torch.isnan(cmd.y_des[:, POS_W - 1]).all()
    assert torch.isnan(cmd.dy_des[:, VEL_W - 1]).all()


# ---------------------------------------------------------------------------
# Dual reference source: both are computed, exactly one is applied.
# ---------------------------------------------------------------------------
def test_both_references_are_computed_every_step():
    """Whichever source is applied, both halves run and both caches are filled.

    That is what makes the FM-vs-clip diagnostic possible without a second rollout: the two
    references are produced from the SAME robot state on the same tick.
    """
    for source in ("fm", "clip"):
        cmd = _update_stub("measured", reference_source=source)
        FmTrackingCommand._update_command(cmd)
        assert "clip" in cmd.calls and "infer" in cmd.calls, (source, cmd.calls)
        # The clip half runs first: the generator must be able to overwrite it.
        assert cmd.calls.index("clip") < cmd.calls.index("infer")
        assert torch.equal(cmd._y_des_clip[:, :NJ], torch.full((N_ENVS, NJ), CLIP_FILL))
        assert torch.equal(cmd._y_des_fm[:, :NJ], torch.zeros(N_ENVS, NJ))


def test_reference_source_selects_what_is_applied():
    """``y_des`` is the named source's cache -- the joint slots tell the two apart."""
    fm = _update_stub("measured", reference_source="fm")
    FmTrackingCommand._update_command(fm)
    assert torch.equal(fm.y_des[:, :NJ], torch.zeros(N_ENVS, NJ))

    clip = _update_stub("measured", reference_source="clip")
    FmTrackingCommand._update_command(clip)
    assert torch.equal(clip.y_des[:, :NJ], torch.full((N_ENVS, NJ), CLIP_FILL))
    # The clip reference is not NaN-poisoned: strict_fm_only guards the GENERATOR cache,
    # and NaNing the applied clip would take the CLF down with it.
    assert torch.isfinite(clip.y_des).all()
    assert torch.isnan(clip._y_des_fm[:, POS_W - 1]).all()


@pytest.mark.parametrize("source", ["fm", "clip"])
def test_as_other_reference_swaps_and_restores(source):
    """The context managers serve the other source, then put the configured one back."""
    other = "clip" if source == "fm" else "fm"
    cmd = _update_stub("measured", reference_source=source)
    FmTrackingCommand._update_command(cmd)
    applied = cmd.y_des.clone()

    manager = cmd.as_clip_reference() if other == "clip" else cmd.as_fm_reference()
    with manager:
        assert cmd._serving() == other
        expected = cmd._y_des_clip if other == "clip" else cmd._y_des_fm
        torch.testing.assert_close(cmd.y_des, expected, equal_nan=True)
    assert cmd._serving() == source
    torch.testing.assert_close(cmd.y_des, applied, equal_nan=True)


def test_apply_source_keeps_one_stable_y_des_tensor():
    """``get_desired_outputs`` assigns into ``y_des`` by index, so it cannot be rebound.

    A rebinding ``_apply_source`` would alias the live vector onto a cache and the next
    step's clip gather would write straight into last step's snapshot.
    """
    cmd = _update_stub("measured")
    before = cmd.y_des
    FmTrackingCommand._update_command(cmd)
    assert cmd.y_des is before
    assert cmd.y_des is not cmd._y_des_fm and cmd.y_des is not cmd._y_des_clip


def test_serve_rejects_the_generator_when_none_was_loaded():
    """The pure-clip control has no generator to report on, and says so."""
    cmd = _update_stub("measured", reference_source="clip")
    cmd.fm_enabled = False
    with pytest.raises(RuntimeError, match="no fm_checkpoint"):
        with cmd.as_fm_reference():
            pass


@pytest.mark.parametrize(
    "accessor,args",
    [("get_future_world_base", ()), ("get_future_joint_state", ()),
     ("get_future_world_frames", (list(FM_FRAMES),))],
)
def test_serving_dispatches_the_lookahead_to_the_clip(monkeypatch, accessor, args):
    """Under a ``clip`` override every window accessor defers to the inherited clip path.

    ``mdp.reference_window`` is built entirely out of these three, so if one of them kept
    reading the prediction buffers the "clip" observation would be a silent hybrid: the
    clip's base with the generator's joints, or similar.
    """
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.motion_sequence_cmd import (
        MotionSequenceCommand,
    )

    cmd = _stub(pred_idx=[0, 0, 0, 0])
    seen: list[str] = []
    monkeypatch.setattr(MotionSequenceCommand, accessor,
                        lambda self, *a: seen.append("clip") or torch.zeros(1))

    offsets = torch.tensor([0, 4])
    getattr(FmTrackingCommand, accessor)(cmd, *args, offsets)
    assert seen == [], "the FM source must NOT go through the clip implementation"

    cmd._serve_override = "clip"
    getattr(FmTrackingCommand, accessor)(cmd, *args, offsets)
    assert seen == ["clip"]


def test_predicted_outputs_window_lands_in_the_y_des_layout():
    """The whole horizon, scattered into the same channels the applied reference uses.

    ``plot_fm_predictions`` fans these off the tracked line and can only do that if node
    ``_pred_idx`` of the window equals what ``_fm_current_outputs`` applies -- same channel
    slots, same joint permutation, same NaN mask.
    """
    cmd = _stub(pred_idx=[0, 1, 2, 3])
    cmd.cfg.strict_fm_only = True
    cmd.y_des = torch.zeros(N_ENVS, POS_W)
    cmd.dy_des = torch.zeros(N_ENVS, VEL_W)
    # base + K ee frames, laid out [base_xyz, ee0_xyz, ...] / [base_quat, ee0_quat, ...].
    cmd._wf_pos_flat = torch.arange(3 * (1 + K))
    cmd._wf_ori_flat = torch.arange(3 * (1 + K), 3 * (1 + K) + 4 * (1 + K))
    cmd._wf_lin_flat = torch.arange(3 * (1 + K))
    cmd._wf_ang_flat = torch.arange(3 * (1 + K), 6 * (1 + K))
    cmd._joint_slots_pos = torch.arange(POS_W - NJ - 1, POS_W - 1)
    cmd._joint_slots_vel = torch.arange(VEL_W - NJ - 1, VEL_W - 1)
    cmd._fm_jperm = torch.tensor(ROBOT_TO_FM, dtype=torch.long)
    cmd._unwritten_pos = torch.tensor([POS_W - 1])
    cmd._unwritten_vel = torch.tensor([VEL_W - 1])
    cmd._ee_zero_quat = torch.zeros(N_ENVS, K, 4)
    cmd._ee_zero_quat[..., 3] = 1.0
    cmd._ee_zero_vec = torch.zeros(N_ENVS, K, 3)

    y, dy = FmTrackingCommand.predicted_outputs_window(cmd)
    assert y.shape == (N_ENVS, F, POS_W) and dy.shape == (N_ENVS, F, VEL_W)
    # The base position of every node is the prediction buffer, unpermuted.
    torch.testing.assert_close(y[:, :, 0:3], cmd._pred_root_pos)
    torch.testing.assert_close(y[:, :, 3:6], cmd._pred_frame_pos[:, :, 0])
    # Joints go through the clip-OUTPUT permutation, as the applied write does.
    torch.testing.assert_close(y[:, :, cmd._joint_slots_pos], cmd._pred_jp[..., cmd._fm_jperm])
    torch.testing.assert_close(dy[:, :, cmd._joint_slots_vel], cmd._pred_jv[..., cmd._fm_jperm])
    assert torch.isnan(y[:, :, POS_W - 1]).all() and torch.isnan(dy[:, :, VEL_W - 1]).all()

    # Node _pred_idx of the window == what _fm_current_outputs applies this step.
    base, ee, jp, jv = FmTrackingCommand._fm_current_outputs(cmd)
    for e in range(N_ENVS):
        node = int(cmd._pred_idx[e])
        torch.testing.assert_close(y[e, node, 0:3], base[e, 0:3])
        torch.testing.assert_close(y[e, node, 3:6], ee[e, 0, 0:3])
        torch.testing.assert_close(y[e, node, cmd._joint_slots_pos], jp[e][cmd._fm_jperm])
        torch.testing.assert_close(dy[e, node, cmd._joint_slots_vel], jv[e][cmd._fm_jperm])


# ---------------------------------------------------------------------------
# The generator/clip barrier, at the config level.
# ---------------------------------------------------------------------------
def _tracker_cfgs():
    """Both tracker env cfgs, so the barrier is asserted on each exteroception variant."""
    from robot_rl.tasks.manager_based.robot_rl.g1.g1_tracker_env_cfg import (
        G1FmTrackerTerrainDepthEnvCfg, G1FmTrackerTerrainEnvCfg,
    )

    return [G1FmTrackerTerrainEnvCfg, G1FmTrackerTerrainDepthEnvCfg]


@pytest.mark.parametrize("index", [0, 1], ids=["height_scan", "depth"])
def test_tracker_env_has_no_rewards_and_no_clip_graded_termination(index):
    pytest.importorskip("isaaclab.envs", exc_type=ImportError)

    cfg = _tracker_cfgs()[index]()
    assert [k for k in vars(cfg.rewards) if not k.startswith("_")] == []
    terms = {k for k in vars(cfg.terminations) if not k.startswith("_")}
    assert terms == {"time_out", "base_fallen"}, terms
    # Only the two ACTOR groups: the privileged critic group is where the last clip-derived
    # observation lived (tracking_error_global), and nothing here trains.
    assert [k for k in vars(cfg.observations) if not k.startswith("_")] == ["policy", "reference"]
    assert cfg.commands.traj_ref.strict_fm_only is True
    # Nothing re-anchors the clip: no env reads its reference.
    assert cfg.commands.traj_ref.ego_anchor_fraction == 0.0
    # No default generator: a stale FM loads and predicts and is silently wrong.
    assert cfg.commands.traj_ref.fm_checkpoint == ""
    # FmTrackingCommand refuses body mode (it predicts no '@root' links), so both variants
    # must land on joint -- the depth env derives from the BODY teacher, which sets body.
    assert cfg.commands.traj_ref.tracking_mode == "joint"


def test_depth_tracker_carries_the_cameras_its_generator_needs():
    """The reason this variant exists: a depth FM reads cameras off the SCENE.

    ``FmTrackingCommand._read_extero`` picks the sensor from the CHECKPOINT, so a
    depth-trained generator in the height-scan task raises looking for terms that are not
    there. Depth is ``FlowMatchingConfig``'s default, so without this env a default-trained
    FM had no task to run in at all.
    """
    pytest.importorskip("isaaclab.envs", exc_type=ImportError)
    from robot_rl.tasks.manager_based.robot_rl.g1.g1_tracker_env_cfg import (
        G1FmTrackerTerrainDepthEnvCfg, G1FmTrackerTerrainEnvCfg,
    )

    depth = G1FmTrackerTerrainDepthEnvCfg()
    assert depth.scene.depth_camera_near is not None
    assert depth.scene.depth_camera_far is not None
    assert depth.observations.policy.depth_near is not None
    assert depth.observations.policy.depth_far is not None
    # Depth REPLACES the scan in the policy branch, as on the depth teacher.
    assert depth.observations.policy.height_scan is None

    scan = G1FmTrackerTerrainEnvCfg()
    assert not hasattr(scan.scene, "depth_camera_near")
    assert scan.observations.policy.height_scan is not None


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
