# Copyright 2026 Zachary Olkin. All rights reserved.

"""Test the FM-rollout -> clip channel scatter (inverse of clip_to_fm_data's map).

Numpy-only: synthesize a world-frame rollout dict with per-channel-identifiable
values, scatter it into the clip ``y_des`` / ``dy_des`` layout, and verify every
base / joint / ee channel lands in the right ``pos_output_names`` /
``vel_output_names`` column -- the inverse of the forward extraction in
``clip_to_fm_data`` (joints/ee in FM order map back to the clip's output names).
"""

from __future__ import annotations

import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts", "flow_matching"))
import fm_rollout_to_clip as r2c  # noqa: E402

BASE = "pelvis_link"
JOINTS = ["j0", "j1", "j2"]
EE = ["left_ankle_roll_link", "right_ankle_roll_link"]


def _names():
    pos, vel = [], []
    for fr in [BASE] + EE:
        pos += [f"{fr}:pos_{a}" for a in ("x", "y", "z")] + [f"{fr}:ori_{a}" for a in ("x", "y", "z", "w")]
        vel += [f"{fr}:pos_{a}" for a in ("x", "y", "z")] + [f"{fr}:ori_{a}" for a in ("x", "y", "z")]
    pos += [f"joint:{j}" for j in JOINTS]
    vel += [f"joint:{j}" for j in JOINTS]
    return pos, vel


def test_scatter_to_clip_inverse_mapping():
    pos_names, vel_names = _names()
    T, nj, K = 4, len(JOINTS), len(EE)
    rng = np.arange  # deterministic, distinct per channel
    roll = {
        "root_pos": rng(T * 3).reshape(T, 3) + 0.1,
        "root_quat": rng(T * 4).reshape(T, 4) + 0.2,
        "root_lin": rng(T * 3).reshape(T, 3) + 0.3,
        "root_ang": rng(T * 3).reshape(T, 3) + 0.4,
        "jp": rng(T * nj).reshape(T, nj) + 0.5,
        "jv": rng(T * nj).reshape(T, nj) + 0.6,
        "frame_pos": rng(T * K * 3).reshape(T, K, 3) + 0.7,
        "frame_quat": rng(T * K * 4).reshape(T, K, 4) + 0.8,
        "frame_lin": rng(T * K * 3).reshape(T, K, 3) + 0.9,
        "frame_ang": rng(T * K * 3).reshape(T, K, 3) + 1.0,
    }
    y, dy = r2c.scatter_to_clip(roll, pos_names, vel_names, BASE, EE, JOINTS)
    assert y.shape == (T, len(pos_names)) and dy.shape == (T, len(vel_names))

    c = lambda names, fr, comps, kind: [names.index(f"{fr}:{kind}_{a}") for a in comps]
    assert np.allclose(y[:, c(pos_names, BASE, ("x", "y", "z"), "pos")], roll["root_pos"])
    assert np.allclose(y[:, c(pos_names, BASE, ("x", "y", "z", "w"), "ori")], roll["root_quat"])
    assert np.allclose(dy[:, c(vel_names, BASE, ("x", "y", "z"), "pos")], roll["root_lin"])
    assert np.allclose(dy[:, c(vel_names, BASE, ("x", "y", "z"), "ori")], roll["root_ang"])
    for j, jn in enumerate(JOINTS):
        assert np.allclose(y[:, pos_names.index(f"joint:{jn}")], roll["jp"][:, j])
        assert np.allclose(dy[:, vel_names.index(f"joint:{jn}")], roll["jv"][:, j])
    for k, fr in enumerate(EE):
        assert np.allclose(y[:, c(pos_names, fr, ("x", "y", "z"), "pos")], roll["frame_pos"][:, k])
        assert np.allclose(y[:, c(pos_names, fr, ("x", "y", "z", "w"), "ori")], roll["frame_quat"][:, k])
        assert np.allclose(dy[:, c(vel_names, fr, ("x", "y", "z"), "pos")], roll["frame_lin"][:, k])
        assert np.allclose(dy[:, c(vel_names, fr, ("x", "y", "z"), "ori")], roll["frame_ang"][:, k])
