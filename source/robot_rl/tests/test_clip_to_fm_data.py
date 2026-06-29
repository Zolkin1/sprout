# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the clip-library -> flow-matching data converter.

Numpy-only (no IsaacLab / torch / simulator): synthesize a tiny clip ``.h5`` with
known, per-channel-identifiable values, run ``clip_to_fm_data.convert`` (+ the
``recorded_data_io`` round-trip), and verify the clip's ``pos_output_names`` /
``vel_output_names`` outputs are re-mapped straight into the ``record_data.py``
``state`` layout (base pos/ori from ``y_des``, lin/ang vel from ``dy_des``, joints,
and the ee frames), that episodes are one-per-clip with truncation at the clip end,
that ``target_vel`` is carried, that no ``ref_state`` is written (so train.py uses
``target_source='actual'``), and that step subsampling (``--stride``) works.
"""

from __future__ import annotations

import json
import os
import sys

import h5py
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts", "flow_matching"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts", "rsl_rl"))
import clip_to_fm_data as c2f  # noqa: E402
from recorded_data_io import load_recorded_data  # noqa: E402

BASE = "pelvis_link"
JOINTS = ["j0", "j1"]
EE = ["left_ankle_roll_link", "right_ankle_roll_link"]  # sorted; both have pos+ori


def _names() -> tuple[list[str], list[str]]:
    """Clip pos/vel output names: base (pos+ori), joints, two ee frames (pos+ori)."""
    pos, vel = [], []
    for fr in [BASE] + EE:
        pos += [f"{fr}:pos_{a}" for a in ("x", "y", "z")]
        pos += [f"{fr}:ori_{a}" for a in ("x", "y", "z", "w")]
        vel += [f"{fr}:pos_{a}" for a in ("x", "y", "z")]   # linear vel
        vel += [f"{fr}:ori_{a}" for a in ("x", "y", "z")]   # angular vel
    pos += [f"joint:{j}" for j in JOINTS]
    vel += [f"joint:{j}" for j in JOINTS]
    return pos, vel


def _write_clip(path: str, n_clips: int = 3, T: int = 5) -> tuple[list[str], list[str]]:
    """Write a clip h5 where every (clip, step, col) cell is uniquely identifiable."""
    pos_names, vel_names = _names()
    P, V = len(pos_names), len(vel_names)
    # y_des cell = 1000*clip + 10*step + col ; dy_des = negated, +0.5 offset (distinct from y).
    cc, tt = np.arange(n_clips)[:, None, None], np.arange(T)[None, :, None]
    y = (1000.0 * cc + 10.0 * tt + np.arange(P)[None, None, :]).astype(np.float64)
    dy = -(1000.0 * cc + 10.0 * tt + np.arange(V)[None, None, :]).astype(np.float64) - 0.5
    c2, t2 = np.arange(n_clips)[:, None], np.arange(T)[None, :]  # (n_clips,1), (1,T)
    z = np.zeros((n_clips, T))
    vx, vy, wz = c2 + 0.1 * t2 + z, 0.2 * t2 + z, -0.3 * c2 + z  # each (n_clips, T)
    tv = np.stack([vx, vy, wz], axis=-1).astype(np.float64)       # (n_clips, T, 3)
    with h5py.File(path, "w") as f:
        f.create_dataset("y_des", data=y)
        f.create_dataset("dy_des", data=dy)
        f.create_dataset("target_vel", data=tv)
        f.attrs["pos_output_names"] = json.dumps(pos_names)
        f.attrs["vel_output_names"] = json.dumps(vel_names)
        f.attrs["dt"] = json.dumps(0.02)
    return pos_names, vel_names


def _col(names, frame, comps, kind):
    return [names.index(f"{frame}:{kind}_{c}") for c in comps]


def _term_slice(term_names, term_dims, target):
    off = 0
    for n, d in zip(term_names, term_dims):
        if n == target:
            return off, off + d
        off += d
    raise KeyError(target)


def test_ee_frames_sorted_minus_base():
    pos_names, _ = _names()
    assert c2f.resolve_ee_frames(pos_names, BASE) == EE  # sorted, base excluded


def test_channel_mapping_and_layout(tmp_path):
    clip = str(tmp_path / "tiny.h5")
    pos_names, vel_names = _write_clip(clip, n_clips=3, T=5)
    data, joints, ee, dt = c2f.convert(clip, BASE, max_clips=None, stride=1)

    assert joints == JOINTS and ee == EE and dt == 0.02
    meta = c2f.build_meta(clip, joints, ee, dt, data["state"].shape[0], "")
    assert sum(meta.state_term_dims) == data["state"].shape[1]   # layout sums to state_dim
    assert data.get("ref_state") is None                          # -> target_source 'actual'

    # Re-read the source cells and check each state term picks the right clip column.
    with h5py.File(clip, "r") as f:
        y, dy = f["y_des"][...], f["dy_des"][...]
    sd = data["state"].shape[1]
    cell = lambda c, t: data["state"][c * 5 + t]  # T == 5
    checks = {
        "root_pos_w": (y, _col(pos_names, BASE, ("x", "y", "z"), "pos")),
        "root_quat_w": (y, _col(pos_names, BASE, ("x", "y", "z", "w"), "ori")),
        "root_link_lin_vel_w": (dy, _col(vel_names, BASE, ("x", "y", "z"), "pos")),
        "root_link_ang_vel_w": (dy, _col(vel_names, BASE, ("x", "y", "z"), "ori")),
        "joint_pos": (y, [pos_names.index(f"joint:{j}") for j in JOINTS]),
        "joint_vel": (dy, [vel_names.index(f"joint:{j}") for j in JOINTS]),
        f"{EE[0]}:pos_w": (y, _col(pos_names, EE[0], ("x", "y", "z"), "pos")),
        f"{EE[1]}:ang_vel_w": (dy, _col(vel_names, EE[1], ("x", "y", "z"), "ori")),
    }
    for c in range(3):
        for t in range(5):
            row = cell(c, t)
            for term, (src, cols) in checks.items():
                a, b = _term_slice(meta.state_term_names, meta.state_term_dims, term)
                assert np.allclose(row[a:b], src[c, t, cols]), f"{term} @clip{c} step{t}"


def test_episode_metadata_and_target_vel(tmp_path):
    clip = str(tmp_path / "tiny.h5")
    _write_clip(clip, n_clips=3, T=5)
    data, _, _, _ = c2f.convert(clip, BASE, max_clips=None, stride=1)

    assert data["episode_id"].tolist() == [c for c in range(3) for _ in range(5)]
    assert data["step_in_episode"].tolist() == [t for _ in range(3) for t in range(5)]
    # done/time_out only at each clip's last step; nothing terminated.
    assert data["done"].reshape(3, 5)[:, :-1].sum() == 0
    assert data["done"].reshape(3, 5)[:, -1].tolist() == [1, 1, 1]
    assert data["terminated"].sum() == 0
    assert data["target_vel"].shape == (15, 3)
    # target_vel carried in step order (clip 2, step 3 -> [2+0.3, 0.6, -0.6]).
    assert np.allclose(data["target_vel"][2 * 5 + 3], [2.3, 0.6, -0.6])


def test_stride_subsamples_steps(tmp_path):
    clip = str(tmp_path / "tiny.h5")
    _write_clip(clip, n_clips=2, T=6)
    data, _, _, _ = c2f.convert(clip, BASE, max_clips=None, stride=2)
    # 6 steps, stride 2 -> 3 kept per clip; 2 clips -> 6 rows.
    assert data["state"].shape[0] == 6
    assert data["step_in_episode"].tolist() == [0, 1, 2, 0, 1, 2]


def test_max_clips(tmp_path):
    clip = str(tmp_path / "tiny.h5")
    _write_clip(clip, n_clips=5, T=4)
    data, _, _, _ = c2f.convert(clip, BASE, max_clips=2, stride=1)
    assert int(data["episode_id"].max()) + 1 == 2 and data["state"].shape[0] == 8


def test_roundtrip_load(tmp_path):
    clip = str(tmp_path / "tiny.h5")
    _write_clip(clip, n_clips=2, T=5)
    data, joints, ee, dt = c2f.convert(clip, BASE, max_clips=None, stride=1)
    meta = c2f.build_meta(clip, joints, ee, dt, data["state"].shape[0], c2f.fingerprint_arrays(data))
    out = str(tmp_path / "fm.npz")
    c2f.save_recorded_data(out, data, meta)

    loaded, lmeta = load_recorded_data(out)
    assert lmeta.state_quat_format == "xyzw" and lmeta.state_joint_names == JOINTS
    assert lmeta.state_frame_names == EE and lmeta.dt == 0.02
    assert loaded["ref_state"] is None and loaded["target_vel"] is not None
    assert np.allclose(loaded["state"], data["state"])
