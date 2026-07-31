# Copyright 2026 Zachary Olkin. All rights reserved.

"""RL-side root-frame body tracking, without IsaacLab's simulator.

Three things are pinned here:

* **The CLF channel subset.** ``y``/``dy`` always carry every channel; ``tracking_mode``
  selects what enters eta.  ``"joint"`` must reproduce the pre-existing behaviour EXACTLY
  (same P, same eta, same subgroups) or the A/B's control arm is not a control.
* **The comparison frame.** Both sides of the CLF are WORLD quantities: the measured block
  is a plain world read, and the desired side is lifted out of its pelvis-relative storage
  by ``_lift_root_frames_to_world``.  ``@root`` names how the CLIP STORES a channel (which
  is what makes it immune to re-anchoring and slope rotation), not the frame it is graded
  in.  The lift's ``omega x r`` cross term is easy to omit and hard to notice -- drop it and
  the positions still look perfect while the velocities are quietly wrong -- so it is
  pinned by a round-trip test.
* **Frame-list stability.** ``resolve_ee_frames`` must keep returning only the WORLD
  end-effectors: it sizes ``ee_ref``, drives the per-step world lift, and gates FM checkpoint
  compatibility.

The real classes are driven on a ``SimpleNamespace`` robot carrying only the attributes the
methods under test touch, following ``test_segment_placement.py``.
"""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul

from motion_gen.joint_names import (
    G1_BODY_TRACK_FRAMES,
    G1_JOINT_NAMES,
    is_root_frame,
    root_frame_name,
)
from robot_rl.tasks.manager_based.robot_rl.g1.clf_weights import (
    WALKING_Q_weights,
    WALKING_R_weights,
)
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking import (
    base_trajectory_cmd as btc,
)
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.base_trajectory_cmd import (
    BaseTrajectoryCommand,
)
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.clf import CLF
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.motion_sequence_cmd import (
    MotionSequenceCommand,
)

DEVICE = "cpu"
WORLD_FRAMES = ["pelvis_link", "left_ankle_roll_link", "right_ankle_roll_link",
                "right_wrist_yaw_link", "left_wrist_yaw_link"]
# Real joint names: the CLF looks every tracked channel up in the Q/R weight dicts.
JOINTS = list(G1_JOINT_NAMES)
# Distinct body names so a body-index mix-up cannot pass by coincidence.
BODY_NAMES = ["pelvis_link"] + [f for f in WORLD_FRAMES if f != "pelvis_link"] + [
    b for b in G1_BODY_TRACK_FRAMES if b not in WORLD_FRAMES
]


def _clip_pos_names(with_root: bool) -> list[str]:
    """The clip's ``pos_output_names``, optionally widened with root-frame channels."""
    names = []
    for f in WORLD_FRAMES:
        names += [f"{f}:pos_{a}" for a in "xyz"] + [f"{f}:ori_{a}" for a in ("w", "x", "y", "z")]
    names += [f"joint:{j}" for j in JOINTS]
    if with_root:
        for b in G1_BODY_TRACK_FRAMES:
            rf = root_frame_name(b)
            names += [f"{rf}:pos_{a}" for a in "xyz"]
            names += [f"{rf}:ori_{a}" for a in ("w", "x", "y", "z")]
    return names


def _fake_cmd(with_root: bool, tracking_mode: str = "joint") -> SimpleNamespace:
    """A command stub with the real ``_parse_outputs`` / ``_resolve_tracked_vel_names`` bound."""
    cmd = SimpleNamespace(
        robot=SimpleNamespace(body_names=BODY_NAMES, joint_names=JOINTS),
        cfg=SimpleNamespace(base_frame_name="pelvis_link", tracking_mode=tracking_mode),
        device=DEVICE,
        num_envs=4,
    )
    result = BaseTrajectoryCommand._parse_outputs(cmd, _clip_pos_names(with_root))
    (cmd.joint_idx, cmd.body_idx, cmd.use_com, cmd.ordered_pos_output_names,
     cmd.ordered_vel_output_names, cmd.body_type, cmd.root_body_idx,
     cmd.root_body_frames) = result
    cmd._root_base_body_idx = BODY_NAMES.index("pelvis_link")
    cmd.tracked_vel_names = BaseTrajectoryCommand._resolve_tracked_vel_names(cmd)
    return cmd


# ----------------------------------------------------------------------------------
# Output parsing
# ----------------------------------------------------------------------------------

def test_parse_splits_world_and_root_groups():
    """Root-frame frames land in their own group, resolved to the underlying body index."""
    cmd = _fake_cmd(with_root=True)
    assert cmd.root_body_frames == [root_frame_name(b) for b in G1_BODY_TRACK_FRAMES]
    assert cmd.root_body_idx == [BODY_NAMES.index(b) for b in G1_BODY_TRACK_FRAMES]
    # The world group is untouched by the widening.
    assert cmd.body_idx == [BODY_NAMES.index(f) for f in WORLD_FRAMES]


def test_parse_without_root_channels_is_unchanged():
    """A pre-widening library parses exactly as before, with no root group."""
    cmd = _fake_cmd(with_root=False)
    assert cmd.root_body_idx is None
    assert cmd.root_body_frames == []
    assert not any(is_root_frame(n) for n in cmd.ordered_pos_output_names)


def test_parse_rejects_a_partial_root_frame():
    """A root frame missing channels is an error, not a silently misaligned block write."""
    names = _clip_pos_names(with_root=False)
    names += [f"{root_frame_name('left_knee_link')}:pos_{a}" for a in "xyz"]   # no ori
    cmd = SimpleNamespace(
        robot=SimpleNamespace(body_names=BODY_NAMES, joint_names=JOINTS),
        cfg=SimpleNamespace(base_frame_name="pelvis_link"), device=DEVICE, num_envs=1,
    )
    with pytest.raises(ValueError, match="must expose exactly"):
        BaseTrajectoryCommand._parse_outputs(cmd, names)


# ----------------------------------------------------------------------------------
# tracking_mode -> tracked channel selection
# ----------------------------------------------------------------------------------

def test_joint_mode_selects_world_and_joints():
    """``joint`` grades the world skeleton + every joint, and no root-frame channel."""
    cmd = _fake_cmd(with_root=True, tracking_mode="joint")
    tracked = cmd.tracked_vel_names
    assert not any(is_root_frame(n) for n in tracked)
    assert sum(n.startswith("joint:") for n in tracked) == len(JOINTS)
    assert len(tracked) == 6 * len(WORLD_FRAMES) + len(JOINTS)


def test_body_mode_swaps_joints_for_root_frames():
    """``body`` drops the joints, adds the body links, and DEDUPES the shared frames.

    Ankles and wrists exist in both the world set and the ``@root`` set. Grading both would
    double their weight and grade them against two different reference velocities (world =
    finite differences, ``@root`` = FK Jacobian). Only the ``@root`` copy is kept.
    """
    cmd = _fake_cmd(with_root=True, tracking_mode="body")
    tracked = cmd.tracked_vel_names
    assert not any(n.startswith("joint:") for n in tracked)
    assert sum(is_root_frame(n) for n in tracked) == 6 * len(G1_BODY_TRACK_FRAMES)

    shared = [f for f in WORLD_FRAMES if f in G1_BODY_TRACK_FRAMES]
    assert shared, "fixture should exercise the dedupe"
    for f in shared:
        assert f"{f}:pos_x" not in tracked, f"{f} world copy must be deduped"
        assert f"{root_frame_name(f)}:pos_x" in tracked
    # Frames with no @root twin (the base) stay, so global placement is still penalised.
    for f in WORLD_FRAMES:
        if f not in G1_BODY_TRACK_FRAMES:
            assert f"{f}:pos_x" in tracked

    kept_world = [f for f in WORLD_FRAMES if f not in G1_BODY_TRACK_FRAMES]
    assert len(tracked) == 6 * (len(kept_world) + len(G1_BODY_TRACK_FRAMES))


def test_body_mode_needs_a_widened_library():
    """Asking for body mode against a narrow library fails loudly with the fix in the message."""
    with pytest.raises(ValueError, match="backfill_root_frames"):
        _fake_cmd(with_root=False, tracking_mode="body")


def test_unknown_mode_rejected():
    """A typo in tracking_mode is an error, not a silent fallback."""
    with pytest.raises(ValueError, match="must be 'joint' or 'body'"):
        _fake_cmd(with_root=True, tracking_mode="bodies")


# ----------------------------------------------------------------------------------
# CLF channel subsetting
# ----------------------------------------------------------------------------------

def _clf(cmd, eta_scales: dict[str, float] | None = None) -> CLF:
    """The real CLF over a stub command's tracked subset."""
    return CLF(sim_dt=0.02, batch_size=cmd.num_envs,
               ordered_vel_output_names=cmd.ordered_vel_output_names,
               ordered_pos_output_names=cmd.ordered_pos_output_names,
               Q_weights=WALKING_Q_weights, R_weights=WALKING_R_weights,
               tracked_vel_names=cmd.tracked_vel_names, eta_scales=eta_scales,
               device=DEVICE)


def test_joint_mode_reproduces_the_unwidened_clf_exactly():
    """The control arm is a true control: same P and same V on a widened library."""
    narrow = _fake_cmd(with_root=False, tracking_mode="joint")
    wide = _fake_cmd(with_root=True, tracking_mode="joint")
    clf_n, clf_w = _clf(narrow), _clf(wide)

    assert clf_n.n_outputs == clf_w.n_outputs
    torch.testing.assert_close(clf_n.P, clf_w.P)
    assert clf_w.ordered_vel_output_names == clf_n.ordered_vel_output_names

    torch.manual_seed(0)
    n = narrow.num_envs
    y_n = torch.randn(n, len(narrow.ordered_pos_output_names))
    dy_n = torch.randn(n, len(narrow.ordered_vel_output_names))
    # Embed the narrow vectors into the wide layout; the extra columns are pure noise the
    # CLF must ignore.
    idx_p = [wide.ordered_pos_output_names.index(nm) for nm in narrow.ordered_pos_output_names]
    idx_v = [wide.ordered_vel_output_names.index(nm) for nm in narrow.ordered_vel_output_names]
    y_w = torch.randn(n, len(wide.ordered_pos_output_names))
    dy_w = torch.randn(n, len(wide.ordered_vel_output_names))
    y_w[:, idx_p] = y_n
    dy_w[:, idx_v] = dy_n

    zeros_p_n = torch.zeros_like(y_n)
    zeros_v_n = torch.zeros_like(dy_n)
    zeros_p_w = torch.zeros_like(y_w)
    zeros_v_w = torch.zeros_like(dy_w)
    zeros_p_w[:, idx_p] = zeros_p_n
    zeros_v_w[:, idx_v] = zeros_v_n

    v_n = clf_n.compute_v(y_n, zeros_p_n, dy_n, zeros_v_n)
    v_w = clf_w.compute_v(y_w, zeros_p_w, dy_w, zeros_v_w)
    torch.testing.assert_close(v_n, v_w)
    assert set(clf_n.v_subgroups) == set(clf_w.v_subgroups)
    for k in clf_n.v_subgroups:
        torch.testing.assert_close(clf_n.v_subgroups[k], clf_w.v_subgroups[k], msg=k)


def test_body_mode_clf_has_no_joint_energy_and_a_root_group():
    """Body-mode eta contains the root-frame links and no joint channel."""
    cmd = _fake_cmd(with_root=True, tracking_mode="body")
    clf = _clf(cmd)
    # Deduped: the shared ankle/wrist frames are graded once, via their @root copy.
    kept_world = [f for f in WORLD_FRAMES if f not in G1_BODY_TRACK_FRAMES]
    assert clf.n_outputs == 2 * 6 * (len(kept_world) + len(G1_BODY_TRACK_FRAMES))
    assert clf.subgroup_indices["joint_pos"].numel() == 0
    assert clf.subgroup_indices["root_body_pos"].numel() == 3 * len(G1_BODY_TRACK_FRAMES)
    assert clf.subgroup_indices["root_body_ori"].numel() == 3 * len(G1_BODY_TRACK_FRAMES)

    torch.manual_seed(1)
    y = torch.randn(cmd.num_envs, len(cmd.ordered_pos_output_names))
    dy = torch.randn(cmd.num_envs, len(cmd.ordered_vel_output_names))
    v = clf.compute_v(y, torch.zeros_like(y), dy, torch.zeros_like(dy))
    assert torch.isfinite(v).all() and (v > 0).all()
    assert float(clf.v_subgroups["root_body_pos"].sum()) > 0.0
    # Perturbing ONLY joint channels must not move V in body mode.
    jcols = [i for i, n in enumerate(cmd.ordered_pos_output_names) if n.startswith("joint:")]
    y2 = y.clone()
    y2[:, jcols] += 5.0
    torch.testing.assert_close(clf.compute_v(y2, torch.zeros_like(y2), dy, torch.zeros_like(dy)), v)


def test_clf_weights_cover_every_root_channel():
    """Every tracked channel needs a Q/R entry or CLF construction KeyErrors."""
    cmd = _fake_cmd(with_root=True, tracking_mode="body")
    for name in cmd.tracked_vel_names:
        assert name in WALKING_Q_weights, name
        assert name in WALKING_R_weights, name


def test_batched_box_minus_matches_per_body():
    """The batched quaternion error equals the per-body loop it replaced.

    ``compute_y_err`` operates on the TRACKED channel width (it runs after the gather in
    ``compute_v``), so the inputs here are built at that width, not the command's full one.
    """
    from isaaclab.utils.math import quat_box_minus
    cmd = _fake_cmd(with_root=True, tracking_mode="body")
    clf = _clf(cmd)
    torch.manual_seed(2)
    n, p = cmd.num_envs, len(clf.ordered_pos_output_names)
    y_a, y_d = torch.randn(n, p), torch.randn(n, p)
    # Normalise the quaternion blocks so box-minus is well posed.
    for cols in clf.ori_body_indices_manifold.values():
        for tensor in (y_a, y_d):
            tensor[:, cols] = torch.nn.functional.normalize(tensor[:, cols], dim=-1)
    got = clf.compute_y_err(y_a, y_d)
    kept_world = [f for f in WORLD_FRAMES if f not in G1_BODY_TRACK_FRAMES]
    assert len(clf.ori_body_indices_manifold) == len(kept_world) + len(G1_BODY_TRACK_FRAMES)
    for body, cols in clf.ori_body_indices_manifold.items():
        tan = clf.ori_body_indices_tangent[body]
        torch.testing.assert_close(got[:, tan], quat_box_minus(y_a[:, cols], y_d[:, cols]),
                                   msg=body)


# ----------------------------------------------------------------------------------
# Measured root-frame block
# ----------------------------------------------------------------------------------

@pytest.fixture
def plain_tensors(monkeypatch):
    """Let ``compute_measured_output`` read plain torch tensors off the stub robot.

    The real method pulls articulation buffers through ``wp.to_torch``; there is no Warp
    array here, so the conversion is the identity for this test.
    """
    monkeypatch.setattr(btc.wp, "to_torch", lambda a: a)


def _random_body_state(n: int, nb: int, seed: int = 0):
    """Random world poses/twists for ``nb`` bodies over ``n`` envs."""
    torch.manual_seed(seed)
    pos = torch.randn(n, nb, 3)
    quat = torch.nn.functional.normalize(torch.randn(n, nb, 4), dim=-1)   # xyzw
    vel = torch.randn(n, nb, 6)
    return pos, quat, vel


def test_measured_body_block_is_the_world_pose_and_twist(plain_tensors):
    """The measured body block is a plain WORLD read, in the caller's reference frame.

    The comparison frame moved: both sides of the CLF are now world quantities (the
    desired side is lifted out of its pelvis-relative storage by
    ``MotionSequenceCommand._lift_root_frames_to_world``).  ``@root`` names how the CLIP
    STORES these channels, not the frame they are graded in.
    """
    cmd = _fake_cmd(with_root=True, tracking_mode="body")
    n, nb_all = cmd.num_envs, len(BODY_NAMES)
    pos_w, quat_w, vel_w = _random_body_state(n, nb_all, seed=3)

    cmd.robot.data = SimpleNamespace(
        body_pos_w=pos_w, body_quat_w=quat_w, body_link_vel_w=vel_w,
        joint_pos=torch.randn(n, len(JOINTS)), joint_vel=torch.randn(n, len(JOINTS)),
    )
    cmd.y_act = torch.zeros(n, len(cmd.ordered_pos_output_names))
    cmd.dy_act = torch.zeros(n, len(cmd.ordered_vel_output_names))
    cmd.body_type = torch.full((len(WORLD_FRAMES),), 2, dtype=torch.int)

    BaseTrajectoryCommand.compute_measured_output(
        cmd, torch.zeros(n, 3), torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(n, 4)
    )

    names, vnames = cmd.ordered_pos_output_names, cmd.ordered_vel_output_names
    for body in G1_BODY_TRACK_FRAMES:
        i = BODY_NAMES.index(body)
        rf = root_frame_name(body)
        got_p = cmd.y_act[:, [names.index(f"{rf}:pos_{a}") for a in "xyz"]]
        got_q = cmd.y_act[:, [names.index(f"{rf}:ori_{a}") for a in ("x", "y", "z", "w")]]
        got_v = cmd.dy_act[:, [vnames.index(f"{rf}:pos_{a}") for a in "xyz"]]
        got_w = cmd.dy_act[:, [vnames.index(f"{rf}:ori_{a}") for a in "xyz"]]
        torch.testing.assert_close(got_p, pos_w[:, i], msg=f"{body} pos")
        torch.testing.assert_close(got_q, quat_w[:, i], msg=f"{body} ori")
        torch.testing.assert_close(got_v, vel_w[:, i, :3], msg=f"{body} lin vel")
        torch.testing.assert_close(got_w, vel_w[:, i, 3:], msg=f"{body} ang vel")


def test_measured_body_block_moves_with_the_robot(plain_tensors):
    """World quantities are pose-EQUIvariant, not pose-invariant.

    The old pelvis-frame block was invariant to rigidly moving the robot.  The world block
    must not be: displacing the robot displaces the tracked links by the same amount, which
    is exactly what lets the anchored reference cancel the free root DOFs and leave the
    z/roll/pitch discrepancy standing.
    """
    cmd = _fake_cmd(with_root=True, tracking_mode="body")
    n, nb_all = cmd.num_envs, len(BODY_NAMES)
    pos_w, quat_w, vel_w = _random_body_state(n, nb_all, seed=4)
    shift = torch.tensor([2.0, -1.0, 0.5])

    def measured(p):
        cmd.robot.data = SimpleNamespace(
            body_pos_w=p, body_quat_w=quat_w, body_link_vel_w=vel_w,
            joint_pos=torch.zeros(n, len(JOINTS)), joint_vel=torch.zeros(n, len(JOINTS)),
        )
        cmd.y_act = torch.zeros(n, len(cmd.ordered_pos_output_names))
        cmd.dy_act = torch.zeros(n, len(cmd.ordered_vel_output_names))
        cmd.body_type = torch.full((len(WORLD_FRAMES),), 2, dtype=torch.int)
        BaseTrajectoryCommand.compute_measured_output(
            cmd, torch.zeros(n, 3), torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(n, 4)
        )
        rp = [i for i, nm in enumerate(cmd.ordered_pos_output_names)
              if is_root_frame(nm) and ":pos_" in nm]
        return cmd.y_act[:, rp].clone()

    base = measured(pos_w)
    shifted = measured(pos_w + shift)
    nb = len(G1_BODY_TRACK_FRAMES)
    torch.testing.assert_close(shifted, base + shift.repeat(nb)[None, :].expand(n, 3 * nb))


def test_lift_inverts_the_pelvis_relative_storage(plain_tensors):
    """Lifting ``@root`` into world and undoing it recovers the stored pose AND twist.

    Guards the cross terms in ``_lift_root_frames_to_world``: drop the ``w0 x (R p_r)``
    and the positions still look perfect while the velocities are quietly wrong, which is
    exactly the bug that would show up only as a mysteriously unlearnable reward.
    """
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.motion_sequence_cmd import (
        MotionSequenceCommand,
    )

    cmd = _fake_cmd(with_root=True, tracking_mode="body")
    n = cmd.num_envs
    b = len(cmd.root_body_frames)
    torch.manual_seed(11)

    # Stored (pelvis-relative) values, and an arbitrary non-trivial base pose+twist.
    p_r = torch.randn(n, b, 3)
    q_r = torch.randn(n, b, 4)
    q_r = q_r / torch.linalg.vector_norm(q_r, dim=-1, keepdim=True)
    v_r, w_r = torch.randn(n, b, 3), torch.randn(n, b, 3)
    base = torch.randn(n, 13)
    base[:, 3:7] /= torch.linalg.vector_norm(base[:, 3:7], dim=-1, keepdim=True)

    cmd.y_des = torch.zeros(n, len(cmd.ordered_pos_output_names))
    cmd.dy_des = torch.zeros(n, len(cmd.ordered_vel_output_names))
    cmd._rf_maps = "unset"
    # The lift resolves its index maps through self._root_frame_maps().
    cmd._root_frame_maps = MethodType(MotionSequenceCommand._root_frame_maps, cmd)
    pos_i, ori_i, lin_i, ang_i = cmd._root_frame_maps()
    cmd.y_des[:, pos_i] = p_r.reshape(n, -1)
    cmd.y_des[:, ori_i] = q_r.reshape(n, -1)
    cmd.dy_des[:, lin_i] = v_r.reshape(n, -1)
    cmd.dy_des[:, ang_i] = w_r.reshape(n, -1)

    MotionSequenceCommand._lift_root_frames_to_world(cmd, base)

    p0, q0, v0, w0 = base[:, 0:3], base[:, 3:7], base[:, 7:10], base[:, 10:13]
    inv = quat_inv(q0)[:, None, :].expand(n, b, 4).reshape(n * b, 4)
    p_w = cmd.y_des[:, pos_i].view(n, b, 3)
    off = p_w - p0[:, None, :]

    # Undo the lift and compare with what went in.
    torch.testing.assert_close(quat_apply(inv, off.reshape(n * b, 3)).view(n, b, 3), p_r)
    torch.testing.assert_close(
        quat_mul(inv, cmd.y_des[:, ori_i].view(n * b, 4)).view(n, b, 4), q_r)
    v_w = cmd.dy_des[:, lin_i].view(n, b, 3)
    lin_back = v_w - v0[:, None, :] - torch.linalg.cross(w0[:, None, :].expand_as(off), off)
    torch.testing.assert_close(quat_apply(inv, lin_back.reshape(n * b, 3)).view(n, b, 3), v_r)
    w_back = cmd.dy_des[:, ang_i].view(n, b, 3) - w0[:, None, :]
    torch.testing.assert_close(quat_apply(inv, w_back.reshape(n * b, 3)).view(n, b, 3), w_r)


# ----------------------------------------------------------------------------------
# Frame-list stability
# ----------------------------------------------------------------------------------

def test_resolve_ee_frames_excludes_root_frames():
    """``ee_ref`` / the world lift / FM checkpoint compat must not see the new frames."""
    cmd = _fake_cmd(with_root=True)
    frames = MotionSequenceCommand.resolve_ee_frames(cmd)
    assert frames == sorted(f for f in WORLD_FRAMES if f != "pelvis_link")
    assert len(frames) == 4
    # Same answer with and without the widening -- that is the invariant.
    assert frames == MotionSequenceCommand.resolve_ee_frames(_fake_cmd(with_root=False))


# ----------------------------------------------------------------------------------
# Per-class channel scaling (``eta_scales``)
# ----------------------------------------------------------------------------------
#
# V sums raw squares over metres, radians, m/s and rad/s as though they were one quantity,
# so the numerically biggest class wins by default regardless of how well it is tracked.
# ``eta_scales`` is the missing unit conversion, folded into P as ``S^T P S``.  What has to
# hold: each class moves only its own channels, V's share moves as the SQUARE of the factor,
# and the result is still a genuine CARE solution (so V stays a valid CLF and
# ``clf_decreasing_condition`` keeps its guarantee) rather than an ad-hoc reweighting.

_SCALES = {"pos": 1.0, "ori": 0.45, "lin_vel": 0.7, "ang_vel": 0.2}


def _class_eta_indices(clf: CLF, cls: str) -> list[int]:
    """Eta indices belonging to one scale class, derived independently of the fold."""
    out = []
    for i, name in enumerate(clf.ordered_vel_output_names):
        pose_key, twist_key = CLF._channel_classes(name)
        if pose_key == cls:
            out.append(2 * i)
        if twist_key == cls:
            out.append(2 * i + 1)
    return out


def test_eta_scales_default_is_a_no_op():
    """The default must not perturb P, or every non-teacher caller silently changes."""
    cmd = _fake_cmd(with_root=True, tracking_mode="body")
    torch.testing.assert_close(_clf(cmd).P, _clf(cmd, eta_scales={}).P, rtol=0, atol=0)
    torch.testing.assert_close(
        _clf(cmd).P, _clf(cmd, eta_scales={"ori": 1.0}).P, rtol=0, atol=0)


def test_eta_scales_hit_their_own_class_and_nothing_else():
    """``P_scaled == outer(s, s) * P``, with s built independently of the implementation."""
    cmd = _fake_cmd(with_root=True, tracking_mode="body")
    base, scaled = _clf(cmd), _clf(cmd, eta_scales=_SCALES)

    s_vec = torch.ones(base.n_outputs, device=base.P.device)
    for cls, factor in _SCALES.items():
        idx = _class_eta_indices(base, cls)
        assert idx, f"body mode must track {cls} channels for this test to mean anything"
        s_vec[idx] = factor
    torch.testing.assert_close(scaled.P, base.P * torch.outer(s_vec, s_vec))

    # Spelled out on the diagonal, which is what the V split is actually built from.
    for cls, factor in _SCALES.items():
        idx = _class_eta_indices(base, cls)
        torch.testing.assert_close(scaled.P[idx, idx], base.P[idx, idx] * factor * factor)


def test_eta_scales_leave_joint_channels_alone_by_default():
    """The joint-vs-body A/B's control arm must not move unless scaled deliberately."""
    cmd = _fake_cmd(with_root=True, tracking_mode="joint")
    base, scaled = _clf(cmd), _clf(cmd, eta_scales={"ori": 0.45, "ang_vel": 0.2})
    joints = _class_eta_indices(base, "joint_pos") + _class_eta_indices(base, "joint_vel")
    assert joints, "joint mode must track joint channels"
    torch.testing.assert_close(
        scaled.P[joints, joints], base.P[joints, joints], rtol=0, atol=0)


def test_eta_scales_move_a_class_share_of_V_as_the_square():
    """The tuning law the config comments quote, on a real V rather than on P alone."""
    torch.manual_seed(0)
    cmd = _fake_cmd(with_root=True, tracking_mode="body")
    base, scaled = _clf(cmd), _clf(cmd, eta_scales={"ori": 0.45})
    eta = torch.randn(4, base.n_outputs, device=base.P.device)

    idx = _class_eta_indices(base, "ori")
    share = lambda clf: torch.einsum(                                        # noqa: E731
        "bi,ij,bj->b", eta[:, idx], clf.P[idx][:, idx], eta[:, idx])
    torch.testing.assert_close(share(scaled), share(base) * 0.45**2)


def test_eta_scales_stay_a_care_solution():
    """``S^T P S`` solves the CARE for ``(a^3/b*q_p, a*b*q_v, b^3/a*r)`` -- V remains a CLF.

    This is the whole justification for the knob: it is a reparameterization of the optimal
    cost-to-go, not an arbitrary reweighting.  Checked on a channel whose two halves are
    scaled by DIFFERENT factors, which is the general case.
    """
    import numpy as np
    from scipy.linalg import solve_continuous_are

    cmd = _fake_cmd(with_root=True, tracking_mode="body")
    scaled = _clf(cmd, eta_scales=_SCALES)
    i = _class_eta_indices(scaled, "ang_vel")[0]        # velocity half of an ori channel
    a, b = _SCALES["ori"], _SCALES["ang_vel"]

    name = scaled.ordered_vel_output_names[i // 2]
    q_p, q_v = WALKING_Q_weights[name]
    (r,) = WALKING_R_weights[name]

    mat_a = np.array([[0.0, 1.0], [0.0, 0.0]])
    mat_b = np.array([[0.0], [1.0]])
    expect = solve_continuous_are(
        mat_a, mat_b, np.diag([a**3 / b * q_p, a * b * q_v]), np.diag([b**3 / a * r]))
    np.testing.assert_allclose(
        scaled.P[i - 1:i + 1, i - 1:i + 1].cpu().numpy(), expect, rtol=1e-5, atol=1e-6)


def test_eta_scales_keep_P_positive_definite():
    """V must stay a valid CLF, i.e. P symmetric positive definite.

    ``S^T P S`` is a congruence transform, so by Sylvester's law of inertia a NONSINGULAR S
    preserves inertia and P stays SPD however lopsided the factors get.  Cholesky is the
    check that actually fails when it does not.
    """
    cmd = _fake_cmd(with_root=True, tracking_mode="body")
    for scales in ({}, _SCALES, {"ori": 0.05, "ang_vel": 0.02}, {"pos": 4.0}):
        P = _clf(cmd, eta_scales=scales).P.double()
        torch.testing.assert_close(P, P.T, rtol=0, atol=0)
        assert torch.linalg.eigvalsh(P)[0] > 0.0, f"P not positive definite at {scales}"
        torch.linalg.cholesky(P)   # raises if it is not


@pytest.mark.parametrize("bad", [0.0, -0.3])
def test_eta_scales_reject_singular_and_negative(bad):
    """The two values that quietly break the CLF rather than loudly failing.

    ``0`` makes S singular, so P goes positive SEMI-definite and a nonzero eta can score
    V = 0.  A negative factor keeps P definite but flips the P12 cross term, rewarding pose
    error and velocity being aligned instead of opposed.
    """
    cmd = _fake_cmd(with_root=True, tracking_mode="body")
    with pytest.raises(ValueError, match="must be > 0"):
        _clf(cmd, eta_scales={"ang_vel": bad})


def test_eta_scales_reject_unknown_class():
    """A typo'd class name must fail loudly, not silently grade nothing."""
    cmd = _fake_cmd(with_root=True, tracking_mode="body")
    with pytest.raises(ValueError, match="unknown eta_scales class"):
        _clf(cmd, eta_scales={"ang_velocity": 0.2})
