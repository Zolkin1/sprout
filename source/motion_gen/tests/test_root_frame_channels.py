# Copyright 2026 Zachary Olkin. All rights reserved.

"""Root-frame body-link channels: generation, invariance, and the offline backfill.

Covers the generator half of the joint-vs-body tracking A/B:

* the ``<body>@root:*`` channels a clip carries really are the FK pose in the pelvis frame,
  and their velocities really are the FK Jacobian times the joint velocities;
* they are INVARIANT under any rigid world transform of the clip -- the property every
  re-datum / splice / slope-rotate site depends on, and the one that silently corrupts data
  if a translation forgets to exclude them;
* ``scripts/rsl_rl/backfill_root_frames.py`` reproduces exactly what the generator would have
  written, so a library widened offline is interchangeable with a regenerated one.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import torch

from motion_gen import features as F
from motion_gen.clip_output import ClipOutputBuilder, widen_output_names
from motion_gen.joint_names import (
    G1_BODY_TRACK_FRAMES,
    is_root_frame,
    world_frame_names,
    world_position_channels,
)
from motion_gen.math_utils import quat_apply, quat_mul

_WORLD_FRAMES = ["pelvis_link", "left_ankle_roll_link", "right_ankle_roll_link",
                 "right_wrist_yaw_link", "left_wrist_yaw_link"]
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _world_names() -> tuple[list[str], list[str]]:
    """A source library's world channel layout (what ``read_library_metadata`` supplies)."""
    pos, vel = [], []
    for f in _WORLD_FRAMES:
        pos += [f"{f}:pos_{a}" for a in "xyz"] + [f"{f}:ori_{a}" for a in ("w", "x", "y", "z")]
        vel += [f"{f}:pos_{a}" for a in "xyz"] + [f"{f}:ori_{a}" for a in "xyz"]
    return pos, vel


def _motion(t_steps: int = 48, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A smooth synthetic world trajectory ``(root_pos, root_quat wxyz, joints)``.

    Joint frequencies are capped at ~2 rad/s so a central difference at dt=0.02 is an
    accurate derivative -- the Jacobian-vs-finite-difference check needs the FD side to be
    trustworthy, and per-joint frequencies scaled by joint index would run past Nyquist.
    """
    torch.manual_seed(seed)
    nj = len(F.G1_JOINT_NAMES)
    t = torch.linspace(0.0, 2.0, t_steps)
    freq = 0.5 + 1.5 * torch.arange(nj, dtype=torch.float32) / nj
    joints = 0.3 * torch.sin(t[:, None] * freq)
    root_pos = torch.stack([t, 0.1 * torch.sin(t), 0.75 + 0.02 * torch.cos(t)], dim=-1)
    ang = 0.4 * t
    root_quat = torch.stack([torch.cos(ang / 2), torch.zeros_like(ang),
                             torch.zeros_like(ang), torch.sin(ang / 2)], dim=-1)
    return root_pos, root_quat, joints


@pytest.fixture(scope="module")
def builder() -> ClipOutputBuilder:
    """A builder with the default root-frame body set."""
    pos, vel = _world_names()
    return ClipOutputBuilder(pos, vel, device="cpu")


def test_layout_widens_by_the_body_set(builder):
    """P/V grow by 7/6 per root-frame body, and world channels keep their indices."""
    pos, vel = _world_names()
    nb = len(G1_BODY_TRACK_FRAMES)
    assert builder.P == len(pos) + len(F.G1_JOINT_NAMES) + 7 * nb
    assert builder.V == len(vel) + len(F.G1_JOINT_NAMES) + 6 * nb
    assert builder.pos_names[:len(pos)] == pos
    assert builder.vel_names[:len(vel)] == vel


def test_fk_chains_are_deduplicated(builder):
    """The 4 end-effectors are also root-frame bodies; each body gets ONE FK chain."""
    expected = list(dict.fromkeys(builder.ee_frames + list(G1_BODY_TRACK_FRAMES)))
    assert builder._fk_bodies == expected
    assert len(builder._fk_bodies) == len(G1_BODY_TRACK_FRAMES)


def test_root_pose_equals_raw_fk(builder):
    """``<body>@root`` position/orientation is the unlifted FK pose in the pelvis frame."""
    root_pos, root_quat, joints = _motion()
    y, _, _ = builder.build(root_pos, root_quat, joints, dt=0.02)
    pos_p, rot_p, _, _ = builder.fk.pose_and_jacobian(joints)
    from motion_gen.math_utils import matrix_to_quat
    for body in G1_BODY_TRACK_FRAMES:
        i = builder._fk_idx[body]
        rf = f"{body}@root"
        torch.testing.assert_close(y[:, builder._p_pos[rf]], pos_p[:, i])
        torch.testing.assert_close(y[:, builder._p_ori[rf]], matrix_to_quat(rot_p[:, i]))


def test_root_velocity_is_the_jacobian_product(builder):
    """``dy`` for a root frame is ``J(q) qdot``, i.e. exactly ``d/dt`` of the root-frame pose."""
    from motion_gen.ingest import ingest_common as IC
    root_pos, root_quat, joints = _motion()
    y, dy, _ = builder.build(root_pos, root_quat, joints, dt=0.02)
    _, _, jv, jw = builder.fk.pose_and_jacobian(joints)
    qd = IC.central_diff(joints, 0.02)
    for body in G1_BODY_TRACK_FRAMES:
        i, rf = builder._fk_idx[body], f"{body}@root"
        torch.testing.assert_close(dy[:, builder._v_lin[rf]],
                                   torch.einsum("taj,tj->ta", jv[:, i], qd))
        torch.testing.assert_close(dy[:, builder._v_ang[rf]],
                                   torch.einsum("taj,tj->ta", jw[:, i], qd))
        # ...and it agrees with a finite difference of the stored root-frame position, which
        # is the property the CLF's double-integrator error model needs.
        fd = IC.central_diff(y[:, builder._p_pos[rf]], 0.02)
        torch.testing.assert_close(fd[2:-2], dy[:, builder._v_lin[rf]][2:-2], atol=2e-3, rtol=2e-3)


def test_world_ee_equals_root_pose_composed_with_root_frame(builder):
    """A body tracked BOTH ways is consistent: world = root_pose x root-frame pose."""
    root_pos, root_quat, joints = _motion()
    y, _, _ = builder.build(root_pos, root_quat, joints, dt=0.02)
    rp = y[:, builder._p_pos["pelvis_link"]]
    rq = y[:, builder._p_ori["pelvis_link"]]
    for body in builder.ee_frames:
        wp = y[:, builder._p_pos[body]]
        wq = y[:, builder._p_ori[body]]
        lp = y[:, builder._p_pos[f"{body}@root"]]
        lq = y[:, builder._p_ori[f"{body}@root"]]
        torch.testing.assert_close(rp + quat_apply(rq, lp), wp, atol=1e-5, rtol=1e-5)
        pred = quat_mul(rq, lq)
        # Quaternion double cover: compare up to sign.
        assert torch.minimum((pred - wq).abs().amax(-1), (pred + wq).abs().amax(-1)).max() < 1e-5


@pytest.mark.parametrize("shift", [(5.0, -3.0, 0.0), (0.0, 0.0, 1.25)])
def test_root_block_invariant_to_world_translation(builder, shift):
    """Translating the clip in the world leaves every root-frame channel bit-identical."""
    root_pos, root_quat, joints = _motion()
    y0, dy0, _ = builder.build(root_pos, root_quat, joints, dt=0.02)
    y1, dy1, _ = builder.build(root_pos + torch.tensor(shift), root_quat, joints, dt=0.02)
    rp = [i for i, n in enumerate(builder.pos_names) if is_root_frame(n)]
    rv = [i for i, n in enumerate(builder.vel_names) if is_root_frame(n)]
    assert torch.equal(y0[:, rp], y1[:, rp])
    assert torch.equal(dy0[:, rv], dy1[:, rv])


def test_root_block_invariant_to_world_yaw(builder):
    """Rotating the clip about world z leaves every root-frame channel unchanged."""
    root_pos, root_quat, joints = _motion()
    y0, dy0, _ = builder.build(root_pos, root_quat, joints, dt=0.02)
    h = 0.5 * 1.1
    yaw = torch.tensor([torch.cos(torch.tensor(h)), 0.0, 0.0, torch.sin(torch.tensor(h))])
    yaw_b = yaw[None].expand(root_pos.shape[0], 4)
    y1, dy1, _ = builder.build(quat_apply(yaw_b, root_pos), quat_mul(yaw_b, root_quat),
                               joints, dt=0.02)
    rp = [i for i, n in enumerate(builder.pos_names) if is_root_frame(n)]
    rv = [i for i, n in enumerate(builder.vel_names) if is_root_frame(n)]
    torch.testing.assert_close(y0[:, rp], y1[:, rp], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(dy0[:, rv], dy1[:, rv], atol=1e-6, rtol=1e-6)


def test_translation_channel_helper_excludes_root(builder):
    """``world_position_channels`` is what a clip translation must move -- world only."""
    for axis in "xyz":
        cols = world_position_channels(builder.pos_names, axis)
        assert cols, axis
        assert all(not is_root_frame(builder.pos_names[c]) for c in cols)
        assert len(cols) == len(_WORLD_FRAMES)
    assert world_frame_names(builder.pos_names) == _WORLD_FRAMES


def test_slope_rotate_leaves_the_root_block_alone(builder):
    """The rigid slope rotation must not touch pelvis-relative channels."""
    from motion_gen.motion_matching.slope_rotate import rotate_clip_to_slope
    root_pos, root_quat, joints = _motion()
    y, dy, _ = builder.build(root_pos, root_quat, joints, dt=0.02)
    yr, dyr = rotate_clip_to_slope(y, dy, builder.pos_names, builder.vel_names,
                                   pitch_rad=0.15, pivot_xy=(0.0, 0.0))
    rp = [i for i, n in enumerate(builder.pos_names) if is_root_frame(n)]
    rv = [i for i, n in enumerate(builder.vel_names) if is_root_frame(n)]
    assert torch.equal(y[:, rp], yr[:, rp])
    assert torch.equal(dy[:, rv], dyr[:, rv])
    # ...but it DID rotate the world block, so this is not a vacuous pass.
    px = builder._p_pos["pelvis_link"]
    assert not torch.allclose(y[:, px], yr[:, px])


def test_widen_output_names_refuses_to_double_add():
    """Widening an already-widened layout is a hard error, not a silent duplicate."""
    pos, vel = _world_names()
    wp, wv = widen_output_names(pos, vel, G1_BODY_TRACK_FRAMES)
    with pytest.raises(ValueError, match="already declares root-frame channels"):
        widen_output_names(wp, wv, G1_BODY_TRACK_FRAMES)


def test_opt_out_reproduces_the_original_layout():
    """``root_frames=()`` gives byte-for-byte the pre-root-frame channel layout."""
    pos, vel = _world_names()
    b = ClipOutputBuilder(pos, vel, root_frames=(), device="cpu")
    assert b.pos_names == pos + [f"joint:{j}" for j in F.G1_JOINT_NAMES]
    assert b.P == len(pos) + len(F.G1_JOINT_NAMES)
    assert not any(is_root_frame(n) for n in b.pos_names)


def _load_backfill_module():
    """Import ``scripts/rsl_rl/backfill_root_frames.py`` by path (scripts/ is not a package)."""
    path = os.path.join(_REPO, "scripts", "rsl_rl", "backfill_root_frames.py")
    spec = importlib.util.spec_from_file_location("_backfill_root_frames", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_backfill_matches_the_generator(builder, tmp_path):
    """The offline widener reproduces exactly what the generator writes.

    This is what lets an existing library be widened instead of regenerated -- which matters
    because the raw Motion Bricks inputs are no longer kept in the tree, and regenerating
    would resample the motions and break the A/B's controlled comparison.
    """
    from motion_gen.clips.clip_io import ClipLibraryWriter, load_clip_library

    bf = _load_backfill_module()
    root_pos, root_quat, joints = _motion()
    t = root_pos.shape[0]

    # Write a NARROW library (no root-frame channels), as an older generator would have.
    pos, vel = _world_names()
    narrow = ClipOutputBuilder(pos, vel, root_frames=(), device="cpu")
    y, dy, contact = narrow.build(root_pos, root_quat, joints, dt=0.02)
    meta = {"dt": 0.02, "clip_length_s": t * 0.02, "num_steps": t,
            "pos_output_names": list(narrow.pos_names), "vel_output_names": list(narrow.vel_names),
            "contact_frames": ["left_ankle_roll_link", "right_ankle_roll_link"],
            "ref_frame": "pelvis_link", "skill_list": ["walk"]}
    src = str(tmp_path / "narrow.h5")
    with ClipLibraryWriter(src, 1, meta, compression=None) as w:
        w.write_batch(0, {
            "y_des": y[None], "dy_des": dy[None], "contact": contact[None],
            "skill_id": torch.zeros(1, t, dtype=torch.uint8),
            "stride_idx": torch.full((1, t), -1, dtype=torch.int32),
            "target_vel": torch.zeros(1, t, 3),
        })

    dst = str(tmp_path / "wide.h5")
    bf.backfill(src, dst, list(G1_BODY_TRACK_FRAMES), batch=4, mjcf_path=None, device="cpu")
    wide = load_clip_library(dst)

    # Same channel layout as a natively-generated wide library...
    assert wide.meta["pos_output_names"] == builder.pos_names
    assert wide.meta["vel_output_names"] == builder.vel_names
    # ...same values in the root block...
    y_ref, dy_ref, _ = builder.build(root_pos, root_quat, joints, dt=0.02)
    torch.testing.assert_close(wide.y_des[0], y_ref, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(wide.dy_des[0], dy_ref, atol=1e-6, rtol=1e-6)
    # ...and the pre-existing columns are untouched.
    assert torch.equal(wide.y_des[0][:, :narrow.P], y)
    assert torch.equal(wide.dy_des[0][:, :narrow.V], dy)


def test_backfill_refuses_to_overwrite(tmp_path):
    """An existing output is an error -- widening a library must not clobber one."""
    bf = _load_backfill_module()
    dst = tmp_path / "exists.h5"
    dst.write_bytes(b"")
    with pytest.raises(FileExistsError):
        bf.backfill("unused.h5", str(dst), list(G1_BODY_TRACK_FRAMES), 4, None, "cpu")
