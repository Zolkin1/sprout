# Copyright 2026 Zachary Olkin. All rights reserved.

"""Parity test: GeneratorTrackerPolicy obs terms vs the real observations.py.

Dev-only (imports isaaclab + robot_rl + the clip library for the CLF output
ordering). Feeds *identical* synthetic world references into the real Isaac
observation functions and into ``GeneratorTrackerPolicy._build_obs``, then
compares the command/reference-derived obs terms term-by-term. This pins down the
quaternion conventions, joint/frame orderings, and per-term formulas of the port
without launching the simulator.

It also exercises the ``joint_out_names`` dependency: the CLF output joint order
(from the clip) is injected into the policy; with the wrong order
``joint_pos_ref``/``joint_vel_ref`` would mismatch.

Run: ``python transfer/sim/tests/test_gen_tracker_obs_parity.py``
"""

from __future__ import annotations

import json
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # transfer/sim
from gen_tracker_policy import GeneratorTrackerPolicy  # noqa: E402

import warp as wp  # noqa: E402
import robot_rl.tasks.manager_based.robot_rl.mdp.observations.observations as O  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CLIP = os.path.join(ROOT, "clips", "g1_flat_v1.h5")
RUN = os.path.join(ROOT, "logs/rsl_rl/g1/2026-06-22_09-47-52_TM2_generator_tracking_test/exported")
FM = os.path.join(ROOT, "logs/flow_matching/2026-06-21_17-07-59/ckpt_89.pt")

BASE = "pelvis_link"
HORIZON_S, STRIDE, DT = 0.5, 5, 0.02
N = 4  # synthetic envs


def _load_output_names() -> tuple[list[str], list[str]]:
    """Read the clip's CLF output name ordering (pos + vel)."""
    with h5py.File(CLIP, "r") as h:
        pos = json.loads(str(np.asarray(h.attrs["pos_output_names"]).item()))
        vel = json.loads(str(np.asarray(h.attrs["vel_output_names"]).item()))
    return pos, vel


def _rand_quat(n: int, g: torch.Generator) -> torch.Tensor:
    q = torch.randn(n, 4, generator=g, dtype=torch.float64)
    return q / q.norm(dim=-1, keepdim=True)


class _MockCmd:
    """Minimal stand-in exposing what the observation functions read."""

    def __init__(self, pos_names, vel_names, ee_frames, joint_out, truth):
        self.ordered_pos_output_names = pos_names
        self.ordered_vel_output_names = vel_names
        self._ee_frames = ee_frames
        self._joint_out = joint_out
        self.cfg = type("cfg", (), {"base_frame_name": BASE})
        self.manager = type("mgr", (), {"dt": DT})
        self.device = truth["root_quat"].device
        self.robot = type("robot", (), {})()
        self.robot.data = type("d", (), {})()
        self.robot.data.root_pos_w = truth["root_pos"]
        self.robot.data.root_quat_w = truth["root_quat"]
        self._t = truth
        self.y_des = self._build_y_des()
        self.dy_des = self._build_dy_des()

    def resolve_ee_frames(self, frame_names=None):
        return list(frame_names) if frame_names is not None else list(self._ee_frames)

    def _pos_idx(self, name):
        return self.ordered_pos_output_names.index(name)

    def _vel_idx(self, name):
        return self.ordered_vel_output_names.index(name)

    def _build_y_des(self) -> torch.Tensor:
        t, P = self._t, len(self.ordered_pos_output_names)
        y = torch.zeros(N, P, dtype=torch.float64, device=self.device)
        for j, a in enumerate("xyz"):
            y[:, self._pos_idx(f"{BASE}:pos_{a}")] = t["ref_pos"][:, j]
        for j, a in enumerate("xyzw"):
            y[:, self._pos_idx(f"{BASE}:ori_{a}")] = t["ref_quat"][:, j]
        for kf, f in enumerate(self._ee_frames):
            for j, a in enumerate("xyz"):
                y[:, self._pos_idx(f"{f}:pos_{a}")] = t["ref_ee_pos"][:, kf, j]
        for jj, jn in enumerate(self._joint_out):  # ref_jp is isaac order -> place by joint name
            y[:, self._pos_idx(f"joint:{jn}")] = t["ref_jp_isaac"][:, t["isaac"].index(jn)]
        return y

    def _build_dy_des(self) -> torch.Tensor:
        t, V = self._t, len(self.ordered_vel_output_names)
        dy = torch.zeros(N, V, dtype=torch.float64, device=self.device)
        for j, a in enumerate("xyz"):
            dy[:, self._vel_idx(f"{BASE}:pos_{a}")] = t["ref_lin"][:, j]
            dy[:, self._vel_idx(f"{BASE}:ori_{a}")] = t["ref_ang"][:, j]
        for jn in self._joint_out:
            dy[:, self._vel_idx(f"joint:{jn}")] = t["ref_jv_isaac"][:, t["isaac"].index(jn)]
        return dy

    def reference_world_state(self):
        t = self._t
        return torch.cat([t["ref_pos"], t["ref_quat"], t["ref_lin"], t["ref_ang"]], dim=-1)

    def get_command_velocity(self):
        return self._t["command"]

    def get_future_command_velocity(self, offs):
        return self._t["fut_cmd"]  # [N, K, 3]

    def get_future_world_base(self, offs):
        t = self._t
        return t["fut_pos"], t["fut_quat"], t["fut_lin"], t["fut_ang"]


def test_obs_parity() -> None:
    """Compare each command/reference obs term: Isaac functions vs the port."""
    pos_names, vel_names = _load_output_names()
    isaac = [n.split(":", 1)[1] for n in pos_names if n.startswith("joint:")]  # CLF order == used for joint refs
    ee_frames = sorted({n.split(":", 1)[0] for n in pos_names if ":pos_" in n} - {BASE})
    nj = len(isaac)

    pol = GeneratorTrackerPolicy(f"{RUN}/policy_parameters.yaml", f"{RUN}/policy.pt", FM, device="cpu")
    # Inject the real CLF output joint order (simulates the export_parameters.py change).
    pol._rl.policy_params["joint_out_names"] = isaac
    pol.load()
    isaac_jn = pol.get_joint_names()  # robot/isaac joint order (== FM order)
    K = len(pol.future_offsets)

    g = torch.Generator().manual_seed(3)
    truth = {
        "isaac": isaac_jn,
        "root_pos": torch.randn(N, 3, generator=g, dtype=torch.float64),
        "root_quat": _rand_quat(N, g),
        "ref_pos": torch.randn(N, 3, generator=g, dtype=torch.float64),
        "ref_quat": _rand_quat(N, g),
        "ref_lin": torch.randn(N, 3, generator=g, dtype=torch.float64),
        "ref_ang": torch.randn(N, 3, generator=g, dtype=torch.float64),
        "ref_jp_isaac": torch.randn(N, nj, generator=g, dtype=torch.float64),
        "ref_jv_isaac": torch.randn(N, nj, generator=g, dtype=torch.float64),
        "ref_ee_pos": torch.randn(N, len(ee_frames), 3, generator=g, dtype=torch.float64),
        "command": torch.randn(N, 3, generator=g, dtype=torch.float64),
        "fut_pos": torch.randn(N, K, 3, generator=g, dtype=torch.float64),
        "fut_quat": _rand_quat(N * K, g).reshape(N, K, 4),
        "fut_lin": torch.randn(N, K, 3, generator=g, dtype=torch.float64),
        "fut_ang": torch.randn(N, K, 3, generator=g, dtype=torch.float64),
    }
    # Held command over the horizon so future_command_vel matches the deployment path.
    truth["fut_cmd"] = truth["command"][:, None, :].expand(N, K, 3).contiguous()

    cmd = _MockCmd(pos_names, vel_names, ee_frames, isaac, truth)
    env = type("env", (), {"num_envs": N, "device": "cpu",
                           "command_manager": type("cm", (), {"get_term": lambda self, n: cmd})()})()

    # --- Isaac reference terms (the real observation functions) ---
    orig = wp.to_torch
    wp.to_torch = lambda a: a if isinstance(a, torch.Tensor) else orig(a)
    try:
        isaac_terms = {
            "base_pose_error": O.base_pose_error(env, "x"),
            "joint_pos_ref": O.joint_pos_ref(env, "x"),
            "joint_vel_ref": O.joint_vel_ref(env, "x").clamp(-20.0, 20.0),
            "ee_ref": O.ee_ref_relative_to_base(env, "x"),
            "command_vel": O.command_velocity(env, "x"),
            "future_command_vel": O.future_command_velocity(env, "x", HORIZON_S, STRIDE),
            "future_base_pose": O.future_base_pose(env, "x", HORIZON_S, STRIDE),
            "future_base_vel": O.future_base_velocity(env, "x", HORIZON_S, STRIDE),
        }
    finally:
        wp.to_torch = orig

    # --- My port: build the same reference per env and slice the assembled obs ---
    offsets = list(range(STRIDE, int(round(HORIZON_S / DT)) + 1, STRIDE))
    term_dims = {t["name"]: t["shape"] for t in pol._rl.get_obs_terms()}
    term_off, acc = {}, 0
    for t in pol._rl.get_obs_terms():
        term_off[t["name"]] = (acc, acc + t["shape"]); acc += t["shape"]

    mine = {k: [] for k in isaac_terms}
    for e in range(N):
        reference = {
            "current": {
                "root_pos": truth["ref_pos"][e], "root_quat": truth["ref_quat"][e],
                "jp": truth["ref_jp_isaac"][e], "jv": truth["ref_jv_isaac"][e],
                "frame_pos": truth["ref_ee_pos"][e],
            },
            "future": [{
                "root_pos": truth["fut_pos"][e, o], "root_quat": truth["fut_quat"][e, o],
                "root_lin": truth["fut_lin"][e, o], "root_ang": truth["fut_ang"][e, o],
            } for o in range(K)],
        }
        state = {
            "root_pos": truth["root_pos"][e].numpy(), "root_quat": truth["root_quat"][e].numpy(),
            "base_ang_vel": np.zeros(3), "joint_pos": np.zeros(nj), "joint_vel": np.zeros(nj),
            "command": truth["command"][e].numpy(),
        }
        pol._rl.action_isaac = np.zeros(nj)
        obs = pol._build_obs(state, reference)[0].numpy()
        for name in isaac_terms:
            lo, hi = term_off[name]
            mine[name].append(obs[lo:hi])

    print(f"K (future offsets) = {K}, offsets = {offsets}, ee_frames = {ee_frames}")
    ok = True
    for name in isaac_terms:
        a = torch.tensor(np.stack(mine[name]), dtype=torch.float64)
        b = isaac_terms[name].to(torch.float64)
        err = (a - b).abs().max().item()
        status = "ok " if err < 1e-5 else "FAIL"
        if err >= 1e-5:
            ok = False
        print(f"  {status} {name:22s} dim={term_dims[name]:<3} max_err={err:.2e}")
    assert ok, "obs parity mismatch (see FAIL rows above)"
    print("ALL OBS PARITY CHECKS PASSED")


if __name__ == "__main__":
    test_obs_parity()
