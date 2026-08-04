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
wp = pytest.importorskip("warp", exc_type=ImportError)
pytest.importorskip("isaaclab.utils.math", exc_type=ImportError)
wp.init()

from robot_rl.flow_matching.window import (
    REFERENCE_WINDOW_BLOCK_NAMES as ALL_BLOCKS,
    window_block_slices,
    window_dim,
    window_to_world,
)
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.fm_tracking_cmd import (  # noqa: E402
    FmTrackingCommand,
)

N_ENVS, NJ, K, F = 4, 6, 2, 20
D = window_dim(NJ, K)     # one reference-window node, all five blocks
FM_FRAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]


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


def _stub(fm_update_every: int = 1, pred_idx: int = 0) -> FmTrackingCommand:
    """A bare command carrying only what the look-ahead accessors read.

    The prediction is ONE ``[N, F, D]`` buffer of reference-window nodes -- the same layout
    the policy observes -- rather than the seven world-frame buffers the old lift needed.
    ``robot.data`` is stubbed because the diagnostic accessors lift through the measured
    pose; ``reference_window_obs`` does not touch it at all, which is the property that
    matters at deploy.
    """
    cmd = object.__new__(_BareFmCommand)
    cmd.cfg = SimpleNamespace(fm_update_every=fm_update_every, reference_source="fm")
    cmd.ctx = SimpleNamespace(F=F, D=D, nj=NJ, num_extra_frames=K, frame_names=list(FM_FRAMES),
                              reference_blocks=ALL_BLOCKS)
    cmd._fm_blocks = ALL_BLOCKS
    # No diagnostic override in force: the accessors answer for cfg.reference_source.
    cmd._serve_override = None
    cmd.fm_enabled = True
    cmd._arange_n = torch.arange(N_ENVS)
    cmd._checked_max_offset = -1
    cmd._zero_offset = torch.zeros(1, dtype=torch.long)
    # Cursor + the pose the live prediction is anchored on. At K=1 the anchor IS the robot
    # now and re-anchoring is a no-op; at K>1 they diverge and the served window has to be
    # corrected back onto the current pose before the policy sees it.
    cmd._pred_idx = torch.full((N_ENVS,), pred_idx, dtype=torch.long)
    cmd._anchor_pos = torch.zeros(N_ENVS, 3)
    cmd._anchor_quat = torch.zeros(N_ENVS, 4)
    cmd._anchor_quat[:, 3] = 1.0
    # Distinct, decodable contents: value == env*1000 + node*10 + channel.
    cmd._pred_window = (torch.arange(N_ENVS)[:, None, None] * 1000.0
                        + torch.arange(F)[None, :, None] * 10.0
                        + torch.arange(D)[None, None, :].float())
    # Small, plausible values in the blocks the world lift actually reads, so the lift is
    # well-posed (the decodable ramp above would be a metres-scale base offset).
    cmd._pred_window[..., 0:12] *= 1e-4
    root_quat = torch.zeros(N_ENVS, 4)
    root_quat[:, 3] = 1.0
    cmd.robot = SimpleNamespace(data=SimpleNamespace(
        root_pos_w=wp.from_torch(torch.zeros(N_ENVS, 3).contiguous()),
        root_quat_w=wp.from_torch(root_quat.contiguous()),
    ))
    return cmd


def test_reference_window_obs_is_the_prediction_when_the_anchor_is_current():
    """At ``fm_update_every=1`` the re-anchor is a no-op, so the served window IS the
    prediction.

    The anchor is captured at the re-plan, and at K=1 that is this very step -- so the
    round trip out to world and back lands where it started. Held to float tolerance rather
    than equality because it really is a round trip through ``quat_box_minus``/``plus``.
    """
    cmd = _stub(fm_update_every=1)          # anchor == robot now (both identity/zero)
    offsets = torch.tensor([0, 5, 10])
    win = FmTrackingCommand.reference_window_obs(cmd, offsets)
    assert win.shape == (N_ENVS, 3, D)
    for k, off in enumerate(offsets.tolist()):
        torch.testing.assert_close(win[:, k], cmd._pred_window[:, off], atol=1e-5, rtol=1e-5)


def test_reference_window_obs_reanchors_onto_the_current_robot():
    """At K>1 the prediction is stale, and the served window must be corrected onto NOW.

    This is the property that replaced "the window ignores the robot pose". It had to: the
    window means "reference relative to the robot's CURRENT base", so once the robot has
    walked away from the re-plan anchor, serving the raw prediction would hand the policy a
    window wrong by exactly that displacement. Correcting it is what costs odometry at
    deploy -- see ``_reanchor_to_now``.
    """
    cmd = _stub(fm_update_every=6, pred_idx=7)
    offsets = torch.tensor([0, 5])
    before = FmTrackingCommand.reference_window_obs(cmd, offsets).clone()

    # The robot walks 0.5 m since the re-plan; the anchor stays where it was.
    moved = torch.zeros(N_ENVS, 3); moved[:, 0] = 0.5
    cmd.robot.data.root_pos_w = wp.from_torch(moved.contiguous())
    after = FmTrackingCommand.reference_window_obs(cmd, offsets)

    # It MUST move, and by exactly the displacement, in the base position channels.
    assert not torch.allclose(before, after, atol=1e-4)
    torch.testing.assert_close(after[..., 0:3], before[..., 0:3] - moved[:, None, :],
                               atol=1e-5, rtol=1e-5)
    # Frame-invariant blocks are untouched by where the robot is.
    jp = window_block_slices(NJ, K)["joint_pos"]
    torch.testing.assert_close(after[..., jp], before[..., jp], atol=1e-6, rtol=1e-6)


def test_reference_window_obs_refuses_a_selection_the_checkpoint_did_not_train_on():
    """The observation term and the generator must name the SAME blocks.

    The prediction is fed to the term verbatim, so a disagreement is not something to drop
    or reorder around -- the policy would read one block's numbers under another's name.
    Two same-width blocks (``joint_pos`` / ``joint_vel``) make it a silent swap rather than
    a shape error, which is why this is checked by name.
    """
    cmd = _stub()                                  # trained on all five
    with pytest.raises(ValueError, match="asks for blocks"):
        FmTrackingCommand.reference_window_obs(
            cmd, torch.tensor([0]), blocks=("base_pose", "base_vel", "joint_pos"))
    # Naming the same five in a different order is the same window, and is accepted.
    win = FmTrackingCommand.reference_window_obs(
        cmd, torch.tensor([0]), blocks=tuple(reversed(ALL_BLOCKS)))
    assert win.shape == (N_ENVS, 1, D)


def test_reference_window_obs_serves_a_slim_window_at_its_own_width():
    """A generator trained on a subset serves that subset, re-anchored the same way.

    The blocks that survive are exactly the ones the teacher packs; the re-anchor has to
    keep working without the dropped ones, which is why it is per-block rather than a
    world round trip (that leg needs every block to reconstruct the node).
    """
    slim = ("base_pose", "base_vel", "joint_pos")
    d_slim = window_dim(NJ, K, slim)
    cmd = _stub(fm_update_every=6, pred_idx=7)
    cmd._fm_blocks = slim
    cmd.ctx.reference_blocks = slim
    cmd.ctx.D = d_slim
    cmd._pred_window = cmd._pred_window[..., :d_slim].contiguous()

    offsets = torch.tensor([0, 5])
    before = FmTrackingCommand.reference_window_obs(cmd, offsets, blocks=slim).clone()
    assert before.shape == (N_ENVS, 2, d_slim) and d_slim < D

    moved = torch.zeros(N_ENVS, 3); moved[:, 0] = 0.5
    cmd.robot.data.root_pos_w = wp.from_torch(moved.contiguous())
    after = FmTrackingCommand.reference_window_obs(cmd, offsets, blocks=slim)
    torch.testing.assert_close(after[..., 0:3], before[..., 0:3] - moved[:, None, :],
                               atol=1e-5, rtol=1e-5)
    jp = window_block_slices(NJ, K, slim)["joint_pos"]
    torch.testing.assert_close(after[..., jp], before[..., jp], atol=1e-6, rtol=1e-6)


def test_clip_history_node_is_built_in_the_checkpoints_layout():
    """``history_from="clip"`` must push a node the FM's own history ring can hold.

    The teacher-forcing row comes from the CLIP branch, which defaults to the full
    five-block layout -- so on a slim checkpoint it produced an 82-wide node to stack
    against 41-wide predictions, and the run died inside ``BatchedHistory.stack_padded``
    on the first inference. Only this path builds a clip node, so nothing else catches it.
    """
    slim = ("base_pose", "base_vel", "joint_pos")
    d_slim = window_dim(NJ, K, slim)
    cmd = _stub()
    cmd._fm_blocks = slim
    cmd.ctx.reference_blocks = slim

    seen = {}

    def _fake_super(offsets, blocks=None):
        """Stand in for the clip branch, recording the selection it was asked for."""
        seen["blocks"] = blocks
        return torch.zeros(N_ENVS, len(offsets), window_dim(NJ, K, blocks))

    # _clip_history_node reaches the clip branch through super(); patch the bound name it
    # resolves to, then check both the request and the width that comes back.
    import robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.motion_sequence_cmd as ms
    original = ms.MotionSequenceCommand.reference_window_obs
    ms.MotionSequenceCommand.reference_window_obs = staticmethod(_fake_super)
    try:
        node = FmTrackingCommand._clip_history_node(cmd)
    finally:
        ms.MotionSequenceCommand.reference_window_obs = original

    assert tuple(seen["blocks"]) == slim, "the clip node was built with the wrong selection"
    assert node.shape == (N_ENVS, d_slim), (
        f"clip history node is {node.shape[-1]} wide but the ring holds {d_slim}")


def test_cursor_offsets_index_from_the_served_node():
    """Offsets are relative to the CURSOR, not the start of the window.

    With K>1 the served node walks forward between re-plans, so "offset 0" has to keep
    meaning "now". Reading from node 0 instead would replay the re-plan instant over and
    over and freeze the reference.
    """
    cmd = _stub(fm_update_every=6, pred_idx=7)   # F=20 here, so 9 + 6 fits
    ar, idx = FmTrackingCommand._future_indices(cmd, torch.tensor([0, 4, 9]))
    assert idx[0].tolist() == [7, 11, 16]
    assert torch.equal(ar[:, 0], torch.arange(N_ENVS))


def test_get_future_joint_state_returns_robot_order_unpermuted():
    """The joint block is served straight through: robot order by construction.

    The window's ``joint_pos`` block IS in the robot's order (that is the
    ``reference_window`` contract), so the FM predicts it that way and there is no
    permutation left to get wrong -- ``_post_init`` refuses a checkpoint that disagrees.
    """
    cmd = _stub()
    offsets = torch.tensor([0, 5, 10])
    jp, jv = FmTrackingCommand.get_future_joint_state(cmd, offsets)
    assert jp.shape == (N_ENVS, 3, NJ) and jv.shape == (N_ENVS, 3, NJ)
    blocks = window_block_slices(NJ, K)
    for k, off in enumerate(offsets.tolist()):
        assert torch.equal(jp[:, k], cmd._pred_window[:, off, blocks["joint_pos"]])
        assert torch.equal(jv[:, k], cmd._pred_window[:, off, blocks["joint_vel"]])


def test_world_accessors_are_the_inverse_of_the_window():
    """The diagnostic lift must reproduce exactly what the window encodes.

    These accessors exist only for the CLF, the debug ribbon and the recorder -- simulation
    has the measured pose the window was expressed against, and nothing off-sim calls them.
    """
    cmd = _stub()
    offsets = torch.tensor([0, 4, 8])
    pos, quat, lin, ang = FmTrackingCommand.get_future_world_base(cmd, offsets)
    rp = wp.to_torch(cmd.robot.data.root_pos_w)[:, None, :]
    rq = wp.to_torch(cmd.robot.data.root_quat_w)[:, None, :]
    want = window_to_world(cmd._pred_window[:, offsets], rp, rq, NJ, K)
    torch.testing.assert_close(pos, want["base_pos_w"])
    torch.testing.assert_close(lin, want["base_lin_vel_w"])
    torch.testing.assert_close(ang, want["base_ang_vel_w"])
    torch.testing.assert_close(quat, want["base_quat_w"])


def test_get_future_world_frames_position_only_placeholders():
    """Position comes from the FM; the other 10 channels are constants, not stale clip data."""
    cmd = _stub()
    offsets = torch.tensor([0, 6])
    out = FmTrackingCommand.get_future_world_frames(cmd, list(FM_FRAMES), offsets)
    assert out.shape == (N_ENVS, 2, K, 13)
    rp = wp.to_torch(cmd.robot.data.root_pos_w)[:, None, :]
    rq = wp.to_torch(cmd.robot.data.root_quat_w)[:, None, :]
    want = window_to_world(cmd._pred_window[:, offsets], rp, rq, NJ, K)
    torch.testing.assert_close(out[..., 0:3], want["ee_pos_w"])
    assert torch.equal(out[..., 3:6], torch.zeros_like(out[..., 3:6]))   # quat xyz
    assert torch.equal(out[..., 6], torch.ones_like(out[..., 6]))        # quat w
    assert torch.equal(out[..., 7:13], torch.zeros_like(out[..., 7:13]))  # lin + ang


def test_get_future_world_frames_reorders_by_name():
    """Frames come back in the ORDER ASKED FOR, not the checkpoint's order."""
    cmd = _stub()
    offsets = torch.tensor([0])
    fwd = FmTrackingCommand.get_future_world_frames(cmd, list(FM_FRAMES), offsets)
    rev = FmTrackingCommand.get_future_world_frames(cmd, list(reversed(FM_FRAMES)), offsets)
    assert torch.equal(fwd[:, :, 0, 0:3], rev[:, :, 1, 0:3])
    assert torch.equal(fwd[:, :, 1, 0:3], rev[:, :, 0, 0:3])


def test_unknown_frame_raises_rather_than_falling_back():
    cmd = _stub()
    with pytest.raises(ValueError, match="does not predict"):
        FmTrackingCommand.get_future_world_frames(cmd, ["left_wrist_yaw_link"], torch.tensor([0]))


# ---------------------------------------------------------------------------
# No clamping: a horizon that cannot cover the request must RAISE.
# ---------------------------------------------------------------------------
def test_horizon_check_accepts_a_sufficient_window():
    # F=20 and the re-plan is every step, so offsets up to 19 are servable.
    cmd = _stub()
    FmTrackingCommand._check_horizon(cmd, torch.tensor([0, 8, 19]))
    assert cmd._checked_max_offset == 19


def test_horizon_check_raises_when_short():
    """Offset 20 needs F >= 21; F is 20, so this must not silently clamp."""
    cmd = _stub()
    with pytest.raises(ValueError, match=r"needs F >= 21|F >= 21"):
        FmTrackingCommand._check_horizon(cmd, torch.tensor([0, 20]))


def test_horizon_check_rechecks_only_on_a_larger_offset():
    """Cached by the largest offset seen: a bigger ask must still be validated."""
    cmd = _stub()
    FmTrackingCommand._check_horizon(cmd, torch.tensor([0, 4]))
    FmTrackingCommand._check_horizon(cmd, torch.tensor([0, 2]))       # smaller: no-op
    assert cmd._checked_max_offset == 4
    with pytest.raises(ValueError):
        FmTrackingCommand._check_horizon(cmd, torch.tensor([0, 25]))


def test_teacher_window_fits_the_shipped_horizon():
    """The shipped F actually covers the shipped teacher window at the shipped cadence.

    This is the arithmetic in ``FlowMatchingConfig.future_len``'s comment, asserted against
    the constants it is derived from -- so widening the teacher's window without moving F
    fails here rather than at the first play step.

    The requirement is ``max(offsets) + 1``, NOT ``len(offsets)``: the window is
    non-uniform (offsets 0, 1, 2 then every 5th step), so what the generator has to cover
    is the furthest offset, and adding near offsets costs no horizon at all.
    """
    from robot_rl.flow_matching import FlowMatchingConfig
    from robot_rl.flow_matching.config import MAX_SERVABLE_REPLAN_PERIOD
    from robot_rl.tasks.manager_based.robot_rl.g1.g1_teacher_env_cfg import (
        REFERENCE_OFFSETS,
        REPLAN_PERIOD_K,
    )
    from robot_rl.tasks.manager_based.robot_rl.g1.g1_teacher_env_cfg import (
        G1TeacherCommandsCfg,
    )
    from robot_rl.tasks.manager_based.robot_rl.g1.g1_tracker_env_cfg import (
        G1FmTrackerCommandsCfg,
    )

    # ONE constant drives all three cadences. They were allowed to disagree and did:
    # ego_anchor_every=1 against fm_snap_every=12, so the teacher trained on a reference
    # refreshed every step while the generator serves a plan up to K-1 steps stale.
    period = G1FmTrackerCommandsCfg().traj_ref.fm_update_every
    teacher = G1TeacherCommandsCfg().traj_ref
    assert period == REPLAN_PERIOD_K, "the tracker's re-plan period must be REPLAN_PERIOD_K"
    assert teacher.fm_snap_every == REPLAN_PERIOD_K, "the recorder cuts windows at K"
    assert teacher.ego_anchor_every == REPLAN_PERIOD_K, (
        "the teacher must re-place the clip on the SAME cadence the generator re-plans on, or "
        "its observation never carries the drift a stale plan has at deploy."
    )
    # The history is one plan's tail, so K must be 1 or >= N; FmTrackingCommand enforces it.
    assert period == 1 or period >= FlowMatchingConfig().num_history_states, (
        f"K={period} straddles two plans' frames for N="
        f"{FlowMatchingConfig().num_history_states} history tokens."
    )
    assert period <= MAX_SERVABLE_REPLAN_PERIOD, (
        f"K={period} exceeds the horizon the generator is sized for."
    )
    # The deployed period is 1, but the horizon must cover the largest period we ever want
    # to be able to SELECT, because a checkpoint cannot gain horizon after the fact: at
    # period K the cursor sits up to K-1 nodes past the re-plan, so the furthest sample
    # reads node max_offset + K. Sizing F to max_offset + 1 silently makes a checkpoint
    # K=1-only, which happened once and cost a retrain to undo.
    required = max(REFERENCE_OFFSETS) + MAX_SERVABLE_REPLAN_PERIOD
    assert FlowMatchingConfig().future_len >= required, (
        f"future_len={FlowMatchingConfig().future_len} must be >= {required} = "
        f"max(REFERENCE_OFFSETS) {max(REFERENCE_OFFSETS)} + MAX_SERVABLE_REPLAN_PERIOD "
        f"{MAX_SERVABLE_REPLAN_PERIOD}, or generators trained with it cannot be served at "
        f"K={MAX_SERVABLE_REPLAN_PERIOD} at all."
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


def _update_stub(reference_source: str = "fm") -> FmTrackingCommand:
    """A bare command with everything ``_update_command`` touches, stubbed and recorded.

    The CLIP half is stubbed to a constant fill (the real one is the inherited clip path,
    covered by ``test_segment_placement.py``); the GENERATOR half and the source selection
    run for real.

    Args:
        reference_source: Which reference the command should apply.

    Returns:
        The instance; ``cmd.calls`` records which stubbed collaborators fired.
    """
    cmd = object.__new__(_BareFmCommand)
    cmd.calls: list[str] = []
    cmd.cfg = SimpleNamespace(fm_update_every=4, strict_fm_only=True,
                              reference_source=reference_source)
    cmd.ctx = SimpleNamespace(F=F, D=D, nj=NJ, num_extra_frames=K, frame_names=list(FM_FRAMES),
                              reference_blocks=ALL_BLOCKS)
    cmd._fm_blocks = ALL_BLOCKS
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
        {"joint_pos": torch.zeros(N_ENVS, NJ), "joint_vel": torch.zeros(N_ENVS, NJ)},
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


def test_update_command_always_computes_measured_output_and_clf():
    """The regression: this block was once stranded behind a conditional early return, so
    on the default configuration it never ran and every tracking metric read zero."""
    cmd = _update_stub()
    FmTrackingCommand._update_command(cmd)

    assert "measured" in cmd.calls, "compute_measured_output never ran"
    assert "clf" in cmd.calls, "the CLF never ran"
    # Order matters: the CLF grades the measured output produced this step, not last step's.
    assert cmd.calls.index("measured") < cmd.calls.index("clf")
    assert torch.equal(cmd.v, torch.full((N_ENVS,), 7.0))
    assert torch.equal(cmd.vdot, torch.full((N_ENVS,), -1.0))


def test_strict_fm_only_nans_run_before_the_clf():
    """So a graded channel the FM does not write shows up as a NaN ``V``, not a plausible one."""
    cmd = _update_stub()
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
        cmd = _update_stub(reference_source=source)
        FmTrackingCommand._update_command(cmd)
        assert "clip" in cmd.calls and "infer" in cmd.calls, (source, cmd.calls)
        # The clip half runs first: the generator must be able to overwrite it.
        assert cmd.calls.index("clip") < cmd.calls.index("infer")
        assert torch.equal(cmd._y_des_clip[:, :NJ], torch.full((N_ENVS, NJ), CLIP_FILL))
        assert torch.equal(cmd._y_des_fm[:, :NJ], torch.zeros(N_ENVS, NJ))


def test_reference_source_selects_what_is_applied():
    """``y_des`` is the named source's cache -- the joint slots tell the two apart."""
    fm = _update_stub(reference_source="fm")
    FmTrackingCommand._update_command(fm)
    assert torch.equal(fm.y_des[:, :NJ], torch.zeros(N_ENVS, NJ))

    clip = _update_stub(reference_source="clip")
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
    cmd = _update_stub(reference_source=source)
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
    cmd = _update_stub()
    before = cmd.y_des
    FmTrackingCommand._update_command(cmd)
    assert cmd.y_des is before
    assert cmd.y_des is not cmd._y_des_fm and cmd.y_des is not cmd._y_des_clip


def test_serve_rejects_the_generator_when_none_was_loaded():
    """The pure-clip control has no generator to report on, and says so."""
    cmd = _update_stub(reference_source="clip")
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

    cmd = _stub()
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

    ``plot_fm_predictions`` fans these off the tracked line and can only do that if node 0
    of the window equals what ``_fm_current_outputs`` applies -- same channel slots, same
    joint permutation, same NaN mask. Diagnostic only: it lifts through the measured pose.
    """
    cmd = _stub()
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
    cmd._fm_jperm = torch.arange(NJ)
    cmd._unwritten_pos = torch.tensor([POS_W - 1])
    cmd._unwritten_vel = torch.tensor([VEL_W - 1])
    cmd._ee_zero_quat = torch.zeros(N_ENVS, K, 4)
    cmd._ee_zero_quat[..., 3] = 1.0
    cmd._ee_zero_vec = torch.zeros(N_ENVS, K, 3)

    y, dy = FmTrackingCommand.predicted_outputs_window(cmd)
    assert y.shape == (N_ENVS, F, POS_W) and dy.shape == (N_ENVS, F, VEL_W)
    # Every node's base + ee position is the window lifted through the measured pose.
    rp = wp.to_torch(cmd.robot.data.root_pos_w)[:, None, :]
    rq = wp.to_torch(cmd.robot.data.root_quat_w)[:, None, :]
    want = window_to_world(cmd._pred_window, rp, rq, NJ, K)
    torch.testing.assert_close(y[:, :, 0:3], want["base_pos_w"])
    torch.testing.assert_close(y[:, :, 3:6], want["ee_pos_w"][:, :, 0])
    # Joints go through the clip-OUTPUT permutation, as the applied write does.
    torch.testing.assert_close(y[:, :, cmd._joint_slots_pos],
                               want["joint_pos"][..., cmd._fm_jperm])
    torch.testing.assert_close(dy[:, :, cmd._joint_slots_vel], want["joint_vel"][..., cmd._fm_jperm])
    assert torch.isnan(y[:, :, POS_W - 1]).all() and torch.isnan(dy[:, :, VEL_W - 1]).all()

    # Node 0 of the window == what _fm_current_outputs applies this step. (Always node 0:
    # fm_update_every is pinned to 1, so there is no cursor to chase.)
    base, ee, joints = FmTrackingCommand._fm_current_outputs(cmd)
    torch.testing.assert_close(y[:, 0, 0:3], base[:, 0:3])
    torch.testing.assert_close(y[:, 0, 3:6], ee[:, 0, 0:3])
    torch.testing.assert_close(y[:, 0, cmd._joint_slots_pos],
                               joints["joint_pos"][..., cmd._fm_jperm])
    torch.testing.assert_close(dy[:, 0, cmd._joint_slots_vel],
                               joints["joint_vel"][..., cmd._fm_jperm])


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


def test_full_window_lift_is_independent_of_the_serving_horizon():
    """The debug ribbon draws ALL F nodes, and must not be gated by the serving check.

    Regression. The ribbon used to ask for offsets ``0..F-1`` through ``_future_indices``,
    which applies the SERVING requirement ``F >= offset + fm_update_every`` -- so at K>1 the
    last offset needed ``F + K - 1`` nodes and the callback raised on every draw. It only
    escaped notice because the ribbon returns early when there is no viewer, so every
    headless run skipped it.

    The second half of the bug was quieter: ``_check_horizon`` caches the largest offset it
    has validated, so a ribbon-sized offset would mark the horizon "checked" far beyond
    anything the observation asks for and the real check would stop running.
    """
    cmd = _stub(fm_update_every=6)
    nj, k = NJ, K
    rp = cmd._anchor_pos[:, None, :]
    rq = cmd._anchor_quat[:, None, :]

    # The whole window lifts, whatever K is -- no cursor, no horizon check.
    world = window_to_world(cmd._pred_window, rp, rq, nj, k)
    assert world["base_pos_w"].shape == (N_ENVS, F, 3)
    assert world["ee_pos_w"].shape == (N_ENVS, F, k, 3)
    assert cmd._checked_max_offset == -1, "drawing must not touch the serving horizon cache"

    # ...while SERVING still refuses an offset the horizon cannot cover at this K.
    with pytest.raises(ValueError, match="needs F >="):
        FmTrackingCommand._check_horizon(cmd, torch.tensor([F - 1]))
