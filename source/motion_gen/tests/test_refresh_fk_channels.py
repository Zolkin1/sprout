# Copyright 2026 Zachary Olkin. All rights reserved.

"""``refresh_fk_channels`` must rewrite the FK channels and nothing else.

The script edits a multi-gigabyte library in one pass, and the channels it touches are
indistinguishable by eye from the ones it must not: they are all smooth float32 columns of
the same array. Getting a slice index wrong would silently overwrite the joint trajectory
with a body pose, and the result would still train.

So the properties worth pinning are the boundaries, not the arithmetic:

* refreshing against the SAME model the library was built with is a no-op on ``@root``;
* ``joint:*`` and ``pelvis_link:*`` come out BIT-identical, always;
* the world blocks are recomposed from the pelvis and the freshly written ``@root``, so the
  two can never disagree;
* the world VELOCITY block is left alone -- deliberately, because it does not compose (see
  the script's module docstring), and a future refactor "fixing" that would silently
  redefine 24 channels.

Built on a synthetic 3-clip library rather than the real one: the real library is 6.8 GB and
lives outside the repo, and these are statements about the code, not about that data.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
torch = pytest.importorskip("torch")

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_MJCF = _REPO_ROOT / "transfer" / "sim" / "robots" / "g1" / "g1_29j.xml"
_MJCF_NEW = _REPO_ROOT / "transfer" / "sim" / "robots" / "g1" / "g1_perceptive.xml"


def _load_script():
    """Import the script by path -- ``scripts/`` is not a package."""
    path = _REPO_ROOT / "scripts" / "rsl_rl" / "refresh_fk_channels.py"
    sys.path.insert(0, str(_REPO_ROOT / "scripts" / "rsl_rl"))
    sys.path.insert(0, str(_REPO_ROOT / "source" / "motion_gen"))
    spec = importlib.util.spec_from_file_location("refresh_fk_channels", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    """The loaded refresh script."""
    if not _MJCF.is_file():
        pytest.fail(f"{_MJCF} is missing; the FK model is the point of this script")
    return _load_script()


@pytest.fixture
def library(tmp_path, script):
    """A tiny library with the real channel layout and random-but-smooth joint motion."""
    from motion_gen import features as F
    from motion_gen.fk import FKChain
    from motion_gen.math_utils import quat_apply, quat_mul

    root_frames = ["left_knee_link", "torso_link", "left_wrist_yaw_link"]
    world_bodies = ["left_wrist_yaw_link"]
    pos_names = (
        [f"pelvis_link:{c}" for c in ("pos_x", "pos_y", "pos_z",
                                      "ori_w", "ori_x", "ori_y", "ori_z")]
        + [f"{b}:{c}" for b in world_bodies
           for c in ("pos_x", "pos_y", "pos_z", "ori_w", "ori_x", "ori_y", "ori_z")]
        + [f"joint:{j}" for j in F.G1_JOINT_NAMES]
        + [f"{b}@root:{c}" for b in root_frames
           for c in ("pos_x", "pos_y", "pos_z", "ori_w", "ori_x", "ori_y", "ori_z")]
    )
    vel_names = (
        [f"pelvis_link:{c}" for c in ("pos_x", "pos_y", "pos_z", "ori_x", "ori_y", "ori_z")]
        + [f"{b}:{c}" for b in world_bodies
           for c in ("pos_x", "pos_y", "pos_z", "ori_x", "ori_y", "ori_z")]
        + [f"joint:{j}" for j in F.G1_JOINT_NAMES]
        + [f"{b}@root:{c}" for b in root_frames
           for c in ("pos_x", "pos_y", "pos_z", "ori_x", "ori_y", "ori_z")]
    )
    n_clips, n_steps, dt = 3, 24, 0.02
    rng = np.random.default_rng(0)
    y = np.zeros((n_clips, n_steps, len(pos_names)), np.float32)
    dy = np.zeros((n_clips, n_steps, len(vel_names)), np.float32)

    t = np.linspace(0.0, 1.0, n_steps)
    jstart = len(pos_names) - 7 * len(root_frames) - len(F.G1_JOINT_NAMES)
    for c in range(n_clips):
        y[c, :, 0:3] = rng.normal(size=(1, 3))                      # pelvis position
        y[c, :, 3] = 1.0                                            # pelvis identity quat
        for j in range(len(F.G1_JOINT_NAMES)):
            y[c, :, jstart + j] = 0.4 * np.sin(2 * np.pi * t + j)
    dy[...] = rng.normal(size=dy.shape).astype(np.float32)          # untouched blocks

    # Fill the FK blocks so the fixture is self-consistent under the OLD model.
    fk = FKChain(str(_MJCF), F.G1_JOINT_NAMES, root_frames, base_link="pelvis_link")
    jcols = list(range(jstart, jstart + len(F.G1_JOINT_NAMES)))
    rp0 = len(pos_names) - 7 * len(root_frames)
    rv0 = len(vel_names) - 6 * len(root_frames)
    for c in range(n_clips):
        joints = torch.as_tensor(y[c][:, jcols], dtype=torch.float32)
        pos_blk, vel_blk = script.root_frame_block(joints, dt, fk, root_frames)
        y[c, :, rp0:] = pos_blk.numpy()
        dy[c, :, rv0:] = vel_blk.numpy()
        b = root_frames.index("left_wrist_yaw_link")
        q_pel = torch.as_tensor(y[c][:, 3:7], dtype=torch.float64)
        p_pel = torch.as_tensor(y[c][:, 0:3], dtype=torch.float64)
        p_r = torch.as_tensor(pos_blk.numpy()[:, b * 7:b * 7 + 3], dtype=torch.float64)
        q_r = torch.as_tensor(pos_blk.numpy()[:, b * 7 + 3:b * 7 + 7], dtype=torch.float64)
        y[c, :, 7:10] = (p_pel + quat_apply(q_pel, p_r)).numpy()
        y[c, :, 10:14] = quat_mul(q_pel, q_r).numpy()

    path = tmp_path / "lib.h5"
    with h5py.File(path, "w") as f:
        f.attrs["pos_output_names"] = json.dumps(pos_names)
        f.attrs["vel_output_names"] = json.dumps(vel_names)
        f.attrs["dt"] = json.dumps(dt)
        f.create_dataset("y_des", data=y)
        f.create_dataset("dy_des", data=dy)
        f.create_dataset("clip_len", data=np.full(n_clips, n_steps, np.int32))
    return path, pos_names, vel_names


def _read(path):
    with h5py.File(path, "r") as f:
        return np.array(f["y_des"]), np.array(f["dy_des"])


def test_refreshing_against_the_same_model_is_a_noop(library, script, tmp_path):
    """The whole approach rests on this: same model in, same numbers out."""
    src, pos_names, _ = library
    dst = tmp_path / "same.h5"
    script.refresh(str(src), str(dst), str(_MJCF), batch=2, device="cpu")
    y0, d0 = _read(src)
    y1, d1 = _read(dst)
    for i, name in enumerate(pos_names):
        if "@root" in name:
            assert np.array_equal(y0[..., i], y1[..., i]), f"{name} moved under the same model"
    assert np.array_equal(d0, d1), "velocities moved under the same model"


def test_motion_source_channels_are_bit_identical(library, script, tmp_path):
    """joint:* and pelvis_link:* are the motion, not the model. They must never change."""
    src, pos_names, vel_names = library
    dst = tmp_path / "new.h5"
    script.refresh(str(src), str(dst), str(_MJCF_NEW), batch=2, device="cpu")
    y0, d0 = _read(src)
    y1, d1 = _read(dst)
    for i, name in enumerate(pos_names):
        if name.startswith("joint:") or name.startswith("pelvis_link:"):
            assert np.array_equal(y0[..., i], y1[..., i]), f"y_des {name} changed"
    for i, name in enumerate(vel_names):
        if name.startswith("joint:") or name.startswith("pelvis_link:"):
            assert np.array_equal(d0[..., i], d1[..., i]), f"dy_des {name} changed"


def test_world_velocity_is_left_alone(library, script, tmp_path):
    """Not recoverable from pelvis + @root -- see the script docstring. Pinned so a later
    'tidy-up' cannot silently redefine 24 channels."""
    src, _, vel_names = library
    dst = tmp_path / "new.h5"
    script.refresh(str(src), str(dst), str(_MJCF_NEW), batch=2, device="cpu")
    d0, d1 = _read(src)[1], _read(dst)[1]
    for i, name in enumerate(vel_names):
        if "@root" not in name and not name.startswith(("joint:", "pelvis_link:")):
            assert np.array_equal(d0[..., i], d1[..., i]), f"world velocity {name} changed"


def test_world_pose_is_recomposed_from_the_refreshed_root(library, script, tmp_path):
    """The world block must agree with the @root block it was written beside."""
    from motion_gen.math_utils import quat_apply, quat_mul

    src, pos_names, _ = library
    dst = tmp_path / "new.h5"
    script.refresh(str(src), str(dst), str(_MJCF_NEW), batch=2, device="cpu")
    y = _read(dst)[0]
    idx = {n: i for i, n in enumerate(pos_names)}
    body = "left_wrist_yaw_link"
    p_pel = torch.as_tensor(y[..., [idx[f"pelvis_link:pos_{a}"] for a in "xyz"]],
                            dtype=torch.float64)
    q_pel = torch.as_tensor(y[..., [idx[f"pelvis_link:ori_{a}"] for a in "wxyz"]],
                            dtype=torch.float64)
    p_r = torch.as_tensor(y[..., [idx[f"{body}@root:pos_{a}"] for a in "xyz"]],
                          dtype=torch.float64)
    q_r = torch.as_tensor(y[..., [idx[f"{body}@root:ori_{a}"] for a in "wxyz"]],
                          dtype=torch.float64)
    want_p = (p_pel + quat_apply(q_pel, p_r)).numpy()
    want_q = quat_mul(q_pel, q_r).numpy()
    got_p = y[..., [idx[f"{body}:pos_{a}"] for a in "xyz"]]
    got_q = y[..., [idx[f"{body}:ori_{a}"] for a in "wxyz"]]
    # float32 storage of a float64 composition
    assert np.abs(got_p - want_p).max() < 1e-5
    assert np.abs(got_q - want_q).max() < 1e-5


def test_it_refuses_to_overwrite(library, script, tmp_path):
    """The input is irreplaceable; a clobbered library is a re-generation, not a re-run."""
    src, _, _ = library
    dst = tmp_path / "out.h5"
    dst.write_bytes(b"")
    with pytest.raises(FileExistsError):
        script.refresh(str(src), str(dst), str(_MJCF), batch=2, device="cpu")


def test_it_refuses_a_library_with_no_root_block(library, script, tmp_path):
    """Nothing to refresh is a mistake worth naming, not a silent copy."""
    src, pos_names, vel_names = library
    stripped = tmp_path / "stripped.h5"
    with h5py.File(src, "r") as a, h5py.File(stripped, "w") as b:
        keep_p = [i for i, n in enumerate(pos_names) if "@root" not in n]
        keep_v = [i for i, n in enumerate(vel_names) if "@root" not in n]
        b.attrs["pos_output_names"] = json.dumps([pos_names[i] for i in keep_p])
        b.attrs["vel_output_names"] = json.dumps([vel_names[i] for i in keep_v])
        b.attrs["dt"] = a.attrs["dt"]
        b.create_dataset("y_des", data=np.array(a["y_des"])[..., keep_p])
        b.create_dataset("dy_des", data=np.array(a["dy_des"])[..., keep_v])
    with pytest.raises(ValueError, match="no <body>@root channels"):
        script.refresh(str(stripped), str(tmp_path / "o.h5"), str(_MJCF),
                       batch=2, device="cpu")
