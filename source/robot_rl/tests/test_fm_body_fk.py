# Copyright 2026 Zachary Olkin. All rights reserved.

"""The body CLF's reference links, derived by FK from the GENERATOR's predicted joints.

Under ``reference_source="fm"`` the generator supplies ``base_pose`` and ``joint_pos`` and
nothing else, but the body CLF grades 13 ``<body>@root`` links.  Those are pure forward
kinematics of the base and the joints, so ``FmTrackingCommand._write_fm_root_body_frames``
derives them in the loop.  Nothing tested that until this file; its correctness rested on an
offline measurement recorded in a docstring and kept nowhere.

**Every way of getting it wrong is silent**, which is why these are pinned:

* **The quaternion convention crosses a boundary here.**  The OFFLINE producer
  (``backfill_root_frames.root_frame_block``) writes ``motion_gen``'s ``matrix_to_quat``,
  which is scalar-FIRST, into columns named ``ori_w, ori_x, ori_y, ori_z`` -- in that file
  order.  The LIVE producer writes ``isaaclab``'s ``quat_from_matrix``, which is
  scalar-LAST, through ``_root_frame_maps``' ``("x", "y", "z", "w")`` gather.  The two
  disagree about component ORDER at both ends and agree about the VALUE, which is the only
  thing that makes them interchangeable -- and getting it wrong is not a crash, it is a
  plausible wrong attitude.  So the agreement is asserted, and so is the failure of the
  misread.
* **The write must precede the world lift.**  ``_write_world_des`` is what lifts ``@root``
  through the anchored base; called after it, the CLF reads pelvis-relative poses as world
  ones -- finite, enormous, and visible only as ``clf_reward`` pinned at 0.  Pinned
  positionally, because the two orderings produce identical shapes.
* **The slots must be registered as WRITTEN**, or ``strict_fm_only`` NaNs precisely the
  channels the body CLF then grades and ``v`` goes NaN with them.

The fixture is synthetic and built by the REAL offline producer, which is what makes the
convention check meaningful without depending on whichever library is on disk: the property
under test is that the live consumer reproduces what the offline producer wrote.
"""

from __future__ import annotations

import importlib.util
import inspect
import pathlib
import sys
from types import MethodType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from isaaclab.utils.math import quat_from_matrix  # noqa: E402

from motion_gen import features as F  # noqa: E402
from motion_gen.clip_output import root_frame_channels  # noqa: E402
from motion_gen.fk import FKChain  # noqa: E402
from motion_gen.joint_names import (  # noqa: E402
    G1_BODY_TRACK_FRAMES,
    root_frame_name,
    strip_root_marker,
)
from motion_gen.math_utils import matrix_to_quat  # noqa: E402
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.fm_tracking_cmd import (  # noqa: E402
    FmTrackingCommand,
)
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.motion_sequence_cmd import (  # noqa: E402
    MotionSequenceCommand,
)

DEVICE = "cpu"
DT = 0.02
#: float32 FK through a ~8-link chain and back out of a quaternion.  The measured worst case
#: against a real library's own stored channels is 1.2e-07 on pose and 3.6e-07 on twist.
TOL = 5e-6
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _load_backfill():
    """Import ``scripts/rsl_rl/backfill_root_frames.py`` by path -- ``scripts/`` is no package.

    Returns:
        The loaded module, whose ``root_frame_block`` is the offline producer of exactly the
        channels this file's subject reproduces in the loop.
    """
    path = _REPO_ROOT / "scripts" / "rsl_rl" / "backfill_root_frames.py"
    if not path.is_file():
        pytest.skip(f"{path} is missing; it is the offline producer under comparison")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("backfill_root_frames", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _smooth_joints(n_steps: int, joint_names: list[str], seed: int = 0) -> torch.Tensor:
    """A smooth, non-degenerate joint trajectory ``[T, nj]``.

    Smooth because the offline block differentiates it, and non-degenerate because a
    near-identity posture would let a wrong quaternion order pass: at ``q ~= [1,0,0,0]``
    every ordering agrees to within the tolerance.

    Args:
        n_steps: Trajectory length.
        joint_names: Joints to generate, giving the column count.
        seed: RNG seed.

    Returns:
        ``[n_steps, len(joint_names)]`` angles in radians.
    """
    g = torch.Generator().manual_seed(seed)
    nj = len(joint_names)
    t = torch.arange(n_steps, dtype=torch.float32)[:, None] * DT
    amp = 0.35 + 0.45 * torch.rand(1, nj, generator=g)
    freq = 0.8 + 2.2 * torch.rand(1, nj, generator=g)
    phase = 6.283 * torch.rand(1, nj, generator=g)
    return amp * torch.sin(freq * t + phase)


def _stub_command(joint_names: list[str], n_envs: int, chain: FKChain) -> SimpleNamespace:
    """A command stub carrying the real ``_root_frame_maps`` and the 13-link channel layout.

    Only the attributes ``_write_fm_root_body_frames`` touches are provided, following
    ``test_root_frame_tracking.py``.

    Args:
        joint_names: The generator's joint order (``ctx.joint_names``).
        n_envs: Row count for ``y_des`` / ``dy_des``.
        chain: The FK chain to serve as ``self._body_fk``.

    Returns:
        The stub, with ``_write_fm_root_body_frames`` bound and ``_pred_world`` left unset.
    """
    root_pos, root_vel = root_frame_channels(list(G1_BODY_TRACK_FRAMES))
    # A base frame and the joints ahead of the '@root' block, exactly as widen_output_names
    # appends it -- so an index map built for a bare '@root' layout cannot pass by accident.
    pos_names = (
        [f"pelvis_link:{c}" for c in ("pos_x", "pos_y", "pos_z",
                                      "ori_w", "ori_x", "ori_y", "ori_z")]
        + [f"joint:{j}" for j in F.G1_JOINT_NAMES] + root_pos
    )
    vel_names = (
        [f"pelvis_link:{c}" for c in ("pos_x", "pos_y", "pos_z", "ori_x", "ori_y", "ori_z")]
        + [f"joint:{j}" for j in F.G1_JOINT_NAMES] + root_vel
    )
    cmd = SimpleNamespace(
        device=DEVICE,
        num_envs=n_envs,
        cfg=SimpleNamespace(base_frame_name="pelvis_link", tracking_mode="body"),
        ctx=SimpleNamespace(joint_names=list(joint_names)),
        manager=SimpleNamespace(dt=DT),
        root_body_frames=[root_frame_name(b) for b in G1_BODY_TRACK_FRAMES],
        ordered_pos_output_names=pos_names,
        ordered_vel_output_names=vel_names,
        y_des=torch.zeros(n_envs, len(pos_names)),
        dy_des=torch.zeros(n_envs, len(vel_names)),
        _body_fk=chain,
        _first_two_offsets=torch.arange(2),
    )
    cmd._root_frame_maps = MethodType(MotionSequenceCommand._root_frame_maps, cmd)
    cmd._write_fm_root_body_frames = MethodType(
        FmTrackingCommand._write_fm_root_body_frames, cmd
    )
    return cmd


def _write_one(chain: FKChain, window: torch.Tensor,
               joint_names: list[str] | None = None) -> SimpleNamespace:
    """Run the production write for a single env over one ``[1, 2, nj]`` node window.

    Args:
        chain: The FK chain to serve.
        window: ``[1, 2, nj]`` predicted joint angles -- node 0 is the served node.
        joint_names: The generator's joint order; defaults to the clip library's.

    Returns:
        The stub, after ``_write_fm_root_body_frames``.
    """
    cmd = _stub_command(joint_names or list(F.G1_JOINT_NAMES), 1, chain)
    cmd._pred_world = lambda offsets: {"joint_pos": window}
    cmd._write_fm_root_body_frames()
    return cmd


@pytest.fixture(scope="module")
def chain() -> FKChain:
    """The production chain: ``DEFAULT_MJCF``, the 13 tracked links, pelvis base."""
    if not pathlib.Path(F.DEFAULT_MJCF).is_file():
        pytest.skip(f"{F.DEFAULT_MJCF} is missing; it is the FK model under test")
    return FKChain(str(F.DEFAULT_MJCF), list(F.G1_JOINT_NAMES),
                   list(G1_BODY_TRACK_FRAMES), base_link="pelvis_link").to(DEVICE)


@pytest.fixture(scope="module")
def offline(chain):
    """The offline ``@root`` block for one smooth trajectory, plus the joints that made it.

    Returns:
        ``(joints [T, nj], qdot [T, nj], pos_block [T, 7B], vel_block [T, 6B])`` from
        ``backfill_root_frames.root_frame_block`` -- the producer of the very columns a real
        library stores.
    """
    bf = _load_backfill()
    joints = _smooth_joints(48, list(F.G1_JOINT_NAMES))
    pos_block, vel_block = bf.root_frame_block(joints, DT, chain,
                                               list(G1_BODY_TRACK_FRAMES))
    return joints, bf._central_diff(joints, DT), pos_block, vel_block


def _blocks_by_body(pos_block, vel_block, step):
    """One step of the offline blocks, split per body into ``(pos, quat_wxyz, lin, ang)``."""
    nb = len(G1_BODY_TRACK_FRAMES)
    p = pos_block[step].view(nb, 7)
    v = vel_block[step].view(nb, 6)
    return p[:, 0:3], p[:, 3:7], v[:, 0:3], v[:, 3:6]


# ----------------------------------------------------------------------------------
# The convention, at both ends of the boundary
# ----------------------------------------------------------------------------------

def test_the_two_quaternion_helpers_disagree_about_component_order():
    """``quat_from_matrix`` is scalar-LAST and ``matrix_to_quat`` scalar-FIRST.

    The premise of every other test here, and the reason the ``@root`` ori channels need a
    permuting gather at all.  Asserted rather than assumed because a convention flip in
    either library would silently invert the block that gets written.
    """
    rz90 = torch.tensor([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    c = 0.5 ** 0.5
    assert torch.allclose(quat_from_matrix(rz90)[0],
                          torch.tensor([0.0, 0.0, c, c]), atol=1e-6), "expected xyzw"
    assert torch.allclose(matrix_to_quat(rz90)[0],
                          torch.tensor([c, 0.0, 0.0, c]), atol=1e-6), "expected wxyz"


def test_the_ori_gather_turns_the_files_w_first_columns_scalar_last(chain):
    """``_root_frame_maps`` gathers ``ori`` as ``(x, y, z, w)``, so a w-first file reads xyzw.

    That permutation is the whole reason the live path can write ``quat_from_matrix``'s
    scalar-last output straight into those slots, and it is invisible in any shape check.
    """
    cmd = _stub_command(list(F.G1_JOINT_NAMES), 1, chain)
    _pos_i, ori_i, _lin_i, _ang_i = cmd._root_frame_maps()
    names = cmd.ordered_pos_output_names
    rf = root_frame_name(G1_BODY_TRACK_FRAMES[0])
    assert [names[int(k)] for k in ori_i[:4]] == [f"{rf}:ori_{a}" for a in ("x", "y", "z", "w")]
    # ...while the FILE stores that body's four columns w-first.
    assert [n for n in names if n.startswith(f"{rf}:ori_")] == [
        f"{rf}:ori_{a}" for a in ("w", "x", "y", "z")]


# ----------------------------------------------------------------------------------
# The live path against the offline producer
# ----------------------------------------------------------------------------------

def test_the_live_fk_reproduces_the_offline_root_pose_block(chain, offline):
    """``_write_fm_root_body_frames`` writes what the offline producer stored.

    Position and orientation, through the production code path, for the 13 links the body
    CLF grades.  The orientation half is the cross-convention check: the offline value is
    scalar-first in the file's own column order, the live one scalar-last through the gather,
    and the two must name the same rotation.
    """
    joints, _qdot, pos_block, vel_block = offline
    step = 17
    q = joints[step][None, None, :].expand(1, 2, joints.shape[1]).clone()
    cmd = _write_one(chain, q)

    pos_i, ori_i, _lin_i, _ang_i = cmd._root_frame_maps()
    nb = len(G1_BODY_TRACK_FRAMES)
    got_pos = cmd.y_des[0, pos_i].view(nb, 3)
    got_quat = cmd.y_des[0, ori_i].view(nb, 4)                      # xyzw, by the gather
    exp_pos, exp_wxyz, _lin, _ang = _blocks_by_body(pos_block, vel_block, step)
    exp_xyzw = exp_wxyz[:, [1, 2, 3, 0]]

    assert torch.allclose(got_pos, exp_pos, atol=TOL), \
        f"position: worst {(got_pos - exp_pos).abs().max():.3e}"
    sgn = torch.sign((got_quat * exp_xyzw).sum(-1, keepdim=True))   # q and -q are one rotation
    assert torch.allclose(got_quat * sgn, exp_xyzw, atol=TOL), \
        f"orientation: worst {(got_quat * sgn - exp_xyzw).abs().max():.3e}"


def test_reading_the_written_orientation_scalar_first_is_badly_wrong(chain, offline):
    """The negative control: the misread is a plausible attitude, not a crash.

    Interpreting the gathered (xyzw) block as the file's w-first order is off by order 1 --
    measured at 1.18 against a real library -- while the POSITIONS stay exact.  That
    asymmetry is how the bug presented, so a test checking only positions would have passed.
    """
    joints, _qdot, pos_block, vel_block = offline
    step = 17
    q = joints[step][None, None, :].expand(1, 2, joints.shape[1]).clone()
    cmd = _write_one(chain, q)

    _pos_i, ori_i, _lin_i, _ang_i = cmd._root_frame_maps()
    nb = len(G1_BODY_TRACK_FRAMES)
    got_quat = cmd.y_des[0, ori_i].view(nb, 4)
    _p, exp_wxyz, _l, _a = _blocks_by_body(pos_block, vel_block, step)
    sgn = torch.sign((got_quat * exp_wxyz).sum(-1, keepdim=True))
    worst = (got_quat * sgn - exp_wxyz).abs().max()
    assert worst > 0.1, (
        f"the scalar-first misread is only {worst:.3e} away, so this fixture cannot tell the "
        "two orders apart -- the posture is too close to identity to be a control")


def test_the_twist_is_the_jacobian_times_a_forward_difference(chain, offline):
    """The ``@root`` twists are ``J . qdot`` with ``qdot`` differenced across the window.

    The generator predicts no ``joint_vel``, so the live path differences its own two leading
    nodes.  Feeding a window whose forward difference equals the offline central difference
    makes the two exactly comparable, which isolates the JACOBIAN convention -- the half that
    must be exact -- from the differencing, which is a documented approximation.
    """
    joints, qdot, pos_block, vel_block = offline
    step = 17
    q = torch.stack([joints[step], joints[step] + DT * qdot[step]])[None]   # [1, 2, nj]
    cmd = _write_one(chain, q)

    _pos_i, _ori_i, lin_i, ang_i = cmd._root_frame_maps()
    nb = len(G1_BODY_TRACK_FRAMES)
    got_lin = cmd.dy_des[0, lin_i].view(nb, 3)
    got_ang = cmd.dy_des[0, ang_i].view(nb, 3)
    _p, _q, exp_lin, exp_ang = _blocks_by_body(pos_block, vel_block, step)

    assert torch.allclose(got_lin, exp_lin, atol=TOL), \
        f"linear: worst {(got_lin - exp_lin).abs().max():.3e}"
    assert torch.allclose(got_ang, exp_ang, atol=TOL), \
        f"angular: worst {(got_ang - exp_ang).abs().max():.3e}"


def test_a_static_window_gives_exactly_zero_twist(chain, offline):
    """Two identical nodes must difference to zero, not to noise.

    Cheap, and the one case where the forward difference has a known answer -- so it
    separates "the difference is taken across the window" from "the difference is taken
    against something else".
    """
    joints, _qdot, _pb, _vb = offline
    q = joints[9][None, None, :].expand(1, 2, joints.shape[1]).clone()
    cmd = _write_one(chain, q)
    _p, _o, lin_i, ang_i = cmd._root_frame_maps()
    assert cmd.dy_des[0, lin_i].abs().max() == 0.0
    assert cmd.dy_des[0, ang_i].abs().max() == 0.0


def test_the_pose_comes_from_node_zero_not_node_one(chain, offline):
    """The served pose is the window's FIRST node; node 1 exists only for the difference.

    Both nodes are the same width and both are plausible postures, so reading the wrong one
    serves a reference one control step ahead of the robot -- with no symptom beyond a
    slightly worse reward.
    """
    joints, _qdot, pos_block, vel_block = offline
    a, b = 17, 31
    q = torch.stack([joints[a], joints[b]])[None]
    cmd = _write_one(chain, q)
    pos_i, _ori_i, _l, _an = cmd._root_frame_maps()
    nb = len(G1_BODY_TRACK_FRAMES)
    got = cmd.y_des[0, pos_i].view(nb, 3)
    exp_a, _qa, _la, _aa = _blocks_by_body(pos_block, vel_block, a)
    exp_b, _qb, _lb, _ab = _blocks_by_body(pos_block, vel_block, b)
    assert torch.allclose(got, exp_a, atol=TOL)
    assert not torch.allclose(got, exp_b, atol=1e-3), \
        "the two nodes are too similar for this to be a control"


# ----------------------------------------------------------------------------------
# The chain is built from the GENERATOR's joint order
# ----------------------------------------------------------------------------------

def test_the_chain_covers_the_thirteen_links_with_the_marker_stripped(chain):
    """``_body_fk`` resolves ``root_body_frames`` to bare body names for the MJCF lookup.

    ``FKChain`` raises on a frame that is not a body, so ``left_knee_link@root`` cannot be
    passed through -- but that failure would be at construction, far from here, and only in
    body mode.
    """
    cmd = _stub_command(list(F.G1_JOINT_NAMES), 1, chain)
    del cmd._body_fk                                   # force the real property to build one
    cmd._body_fk_chain = None
    built = FmTrackingCommand._body_fk.fget(cmd)
    assert built.frame_names == [strip_root_marker(f) for f in cmd.root_body_frames]
    assert built.frame_names == list(G1_BODY_TRACK_FRAMES)


def test_the_chain_follows_the_generators_joint_order_not_the_librarys(chain):
    """The FK joint order comes from ``ctx.joint_names``, never from ``G1_JOINT_NAMES``.

    The generator's ``joint_pos`` is in the POLICY's interleaved order and the clip library's
    is per-limb; 27 of 29 differ.  FK'ing under the wrong one gives a plausible wrong
    skeleton -- it has already produced a false finding elsewhere in this repo -- so the
    binding is pinned by permuting the order and requiring the answer to follow.
    """
    perm = list(F.G1_JOINT_NAMES)
    perm = perm[7:] + perm[:7]                         # a real reordering, same set
    cmd = _stub_command(perm, 1, chain)
    del cmd._body_fk
    cmd._body_fk_chain = None
    built = FmTrackingCommand._body_fk.fget(cmd)

    joints = _smooth_joints(4, list(F.G1_JOINT_NAMES), seed=3)[2]
    idx = [F.G1_JOINT_NAMES.index(j) for j in perm]
    ref, _r, _jv, _jw = chain.pose_and_jacobian(joints[None])
    got, _r2, _jv2, _jw2 = built.pose_and_jacobian(joints[idx][None])
    assert torch.allclose(got, ref, atol=TOL), (
        "FK under a permuted joint order did not reproduce the reference; the chain is not "
        "reading ctx.joint_names")


# ----------------------------------------------------------------------------------
# Ordering and registration -- both silent, both pinned positionally
# ----------------------------------------------------------------------------------

def test_the_fk_write_precedes_the_world_lift():
    """``_write_fm_root_body_frames`` must be called BEFORE ``_write_world_des``.

    ``_write_world_des`` lifts the ``@root`` channels through the anchored base
    (``_lift_root_frames_to_world``).  Written after it, the CLF reads pelvis-relative poses
    as world ones: finite, enormous, and visible only as ``clf_reward`` pinned at 0 -- no
    exception, no shape change, no NaN.

    Positional, because the failure IS the ordering and both orderings type-check.
    """
    src = inspect.getsource(FmTrackingCommand._build_fm_reference)
    fk = src.index("self._write_fm_root_body_frames()")
    lift = src.index("self._write_world_des(")
    assert fk < lift, (
        "_write_fm_root_body_frames must precede _write_world_des; after it the '@root' "
        "channels stay pelvis-relative while the CLF grades them as world")


def test_the_root_body_slots_are_registered_as_written_in_body_mode():
    """Body mode must add the ``@root`` maps to BOTH written lists in ``_build_strict_masks``.

    The generator does not predict these channels, so without the registration
    ``strict_fm_only`` NaNs them -- and they are exactly what the body CLF then reads, so
    ``v`` goes NaN and the reward with it.  Both halves are needed: pose alone leaves every
    twist NaN.
    """
    src = inspect.getsource(FmTrackingCommand._build_strict_masks)
    assert "rf_maps = self._root_frame_maps()" in src
    pos_reg = src.index("pos_written += [rf_maps[0], rf_maps[1]]")
    vel_reg = src.index("vel_written += [rf_maps[2], rf_maps[3]]")
    # Each registration is gated on body mode, and the vel one lands after the vel list exists.
    assert src.index("vel_written: list[torch.Tensor] = []") < vel_reg
    assert pos_reg < vel_reg


def test_body_mode_refuses_a_generator_that_predicts_no_joint_pos():
    """There is no posture to run FK on, so the pair is refused at construction.

    Guarded in ``_post_init`` rather than at the first write: a checkpoint without
    ``joint_pos`` would otherwise reach ``_write_fm_root_body_frames`` and fail deep in the
    step, thousands of iterations from the config that caused it.
    """
    src = inspect.getsource(FmTrackingCommand._post_init)
    guard = src.index('"joint_pos" not in self._fm_blocks')
    assert 'tracking_mode", "joint") == "body"' in src[:guard]
    assert 'reference_source == "fm"' in src[:guard]
    assert "raise ValueError" in src[guard:guard + 400]


def test_the_thirteen_links_account_for_the_documented_channel_counts():
    """13 links are 91 position and 78 velocity channels -- the widths the counts rest on.

    ``_build_strict_masks`` prints "writes N/155 position", and the body-mode figure of
    127/155 is ``base 7 + joints 29 + root 91``; 84/137 is ``base_vel 6 + root_vel 78``.
    Pinning the block width here is what makes those printed numbers checkable at a glance.
    """
    pos, vel = root_frame_channels(list(G1_BODY_TRACK_FRAMES))
    assert len(G1_BODY_TRACK_FRAMES) == 13
    assert len(pos) == 13 * 7 == 91
    assert len(vel) == 13 * 6 == 78
    assert 7 + len(F.G1_JOINT_NAMES) + len(pos) == 127
    assert 6 + len(vel) == 84
