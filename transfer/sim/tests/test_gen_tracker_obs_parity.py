# Copyright 2026 Zachary Olkin. All rights reserved.

"""Parity test: the deployed obs assembly vs the real observations.py.

Dev-only (imports isaaclab + robot_rl for the real observation functions). Feeds
*identical* synthetic world references into those functions and into
``deploy.gen_tracker.GenTrackerPolicy._build_obs``, then compares the
command/reference-derived obs terms term-by-term. This pins down the quaternion
conventions, joint/frame orderings, and per-term formulas of the deployment port without
launching the simulator.

It also exercises the ``joint_out_names`` dependency (with the wrong CLF output order
``joint_pos_ref``/``joint_vel_ref`` would mismatch) and, when the run has depth cameras,
that each camera's image lands in its own slice of the observation vector.

Everything comes from the exported ``tracker.onnx`` metadata, so the test needs only that
artifact and skips cleanly without one. Run from ``transfer/sim``:
``python tests/test_gen_tracker_obs_parity.py [--run <run dir>]``
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

_SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # transfer/sim
sys.path.insert(0, _SIM)
sys.path.insert(0, os.path.dirname(_SIM))                            # transfer/
from deploy.gen_tracker import GenTrackerPolicy  # noqa: E402

import warp as wp  # noqa: E402
import robot_rl.tasks.manager_based.robot_rl.mdp.observations.observations as O  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

BASE = "pelvis_link"
HORIZON_S, STRIDE, DT = 0.5, 5, 0.02
N = 4  # synthetic envs


def _find_run(explicit: str | None) -> str | None:
    """Resolve an exported run directory holding a ``tracker.onnx``.

    Args:
        explicit: A run directory (or its ``exported`` subdirectory) to use, or ``None``
            to take the most recent run that has been exported.

    Returns:
        The ``exported`` directory path, or ``None`` if nothing is exported.
    """
    if explicit:
        cand = explicit if os.path.basename(explicit) == "exported" else os.path.join(explicit, "exported")
        return cand if os.path.exists(os.path.join(cand, "tracker.onnx")) else None
    hits = glob.glob(os.path.join(ROOT, "logs", "rsl_rl", "*", "*", "exported", "tracker.onnx"))
    return os.path.dirname(max(hits, key=os.path.getmtime)) if hits else None


def _output_names(joint_out: list[str], ee_frames: list[str]) -> tuple[list[str], list[str]]:
    """Synthesise the CLF output-name layout from the tracker's own metadata.

    The observation functions index ``y_des`` / ``dy_des`` by NAME, so any consistent
    ordering works as long as every channel is present. Building the layout from the
    tracker's ``joint_out_names`` + ``ee_frame_names`` (rather than from a clip library)
    keeps the test self-sufficient AND pins the ordering the deployment actually uses --
    a clip whose joint set differs from the policy's would otherwise silently mis-scope
    this comparison.

    Args:
        joint_out: CLF output joint order from the tracker metadata.
        ee_frames: End-effector frame names from the tracker metadata.

    Returns:
        ``(ordered_pos_output_names, ordered_vel_output_names)``.
    """
    pos = [f"{BASE}:pos_{a}" for a in "xyz"] + [f"{BASE}:ori_{a}" for a in "xyzw"]
    for frame in ee_frames:
        pos += [f"{frame}:pos_{a}" for a in "xyz"]
    pos += [f"joint:{n}" for n in joint_out]
    vel = [f"{BASE}:pos_{a}" for a in "xyz"] + [f"{BASE}:ori_{a}" for a in "xyz"]
    vel += [f"joint:{n}" for n in joint_out]
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


def test_obs_parity(run: str | None = None) -> None:
    """Compare each command/reference obs term: Isaac functions vs the deployment port."""
    exported = _find_run(run)
    if exported is None:
        print("[SKIP] No exported tracker.onnx found under logs/rsl_rl/*/*/exported/; "
              "export one with scripts/rsl_rl/export_tracker_onnx.py.")
        return
    print(f"[INFO] tracker: {exported}")

    # Tracker-only: no fm.onnx, so the reference is injected directly (as clip playback does).
    pol = GenTrackerPolicy(os.path.join(exported, "tracker.onnx"), None)
    pol.load()
    isaac_jn = pol.get_joint_names()  # robot/isaac joint order
    K = len(pol.future_offsets)
    depth_spec = pol.get_depth_camera_spec()

    isaac = pol._trk.get_joint_out_names() or isaac_jn   # CLF output order
    ee_frames = pol._trk.get_ee_frame_names() or []
    nj = len(isaac_jn)
    pos_names, vel_names = _output_names(isaac, ee_frames)
    if sorted(isaac) != sorted(isaac_jn):
        print(f"[SKIP] joint_out_names ({len(isaac)}) is not a permutation of the "
              f"tracker's joint order ({nj}).")
        return

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

    # --- The deployment port: same reference per env, then slice the assembled obs ---
    offsets = list(range(STRIDE, int(round(HORIZON_S / DT)) + 1, STRIDE))
    term_dims = {t["name"]: t["shape"] for t in pol._trk.get_obs_terms()}
    term_off, acc = {}, 0
    for t in pol._trk.get_obs_terms():
        term_off[t["name"]] = (acc, acc + t["shape"]); acc += t["shape"]

    # Synthetic depth, one row per camera in contract order (what the caster returns), with
    # distinct content per camera so a swapped or misplaced image shows up.
    rng = np.random.default_rng(0)
    depth_images = np.stack([
        rng.uniform(-1.0, 1.0, int(cam["width"]) * int(cam["height"])) for cam in depth_spec
    ]) if depth_spec else None

    mine = {k: [] for k in isaac_terms}
    obs_last = None
    for e in range(N):
        reference = {
            "current": {
                "root_pos": truth["ref_pos"][e], "root_quat": truth["ref_quat"][e],
                "root_lin": truth["ref_lin"][e], "root_ang": truth["ref_ang"][e],
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
        if depth_images is not None:
            state["depth"] = depth_images
        if "height_scan" in term_dims:
            state["height_scan"] = np.zeros(term_dims["height_scan"])
        pol._trk.action_isaac = np.zeros(nj)
        obs_last = pol._build_obs(state, reference)[0]
        for name in isaac_terms:
            lo, hi = term_off[name]
            mine[name].append(obs_last[lo:hi])

    print(f"K (future offsets) = {K}, offsets = {offsets}, ee_frames = {ee_frames}")
    ok = True
    for name in isaac_terms:
        a = torch.tensor(np.stack(mine[name]), dtype=torch.float64)
        b = isaac_terms[name].to(torch.float64)
        err = (a - b).abs().max().item()
        status = "ok " if err < 1e-5 else "FAIL"
        if err >= 1e-5:
            ok = False
        print(f"  {status} {name:22s} dim={term_dims[name]:<5} max_err={err:.2e}")

    # Depth placement: each camera's image must occupy its own slice, clipped as trained
    # (the clip comes from the term's own exported metadata, like every other term's).
    term_clip = {t["name"]: t["clip"] for t in pol._trk.get_obs_terms()}
    for i, cam in enumerate(depth_spec):
        term = cam["obs_term"]
        lo, hi = term_off[term]
        want = np.asarray(depth_images[i], dtype=np.float32)
        clip = term_clip[term]
        if clip is not None:
            want = np.clip(want, -clip, clip)
        err = float(np.abs(obs_last[lo:hi] - want).max())
        status = "ok " if err < 1e-6 else "FAIL"
        if err >= 1e-6:
            ok = False
        print(f"  {status} {term:22s} dim={term_dims[term]:<5} max_err={err:.2e} "
              f"(slice {lo}:{hi}, {cam['width']}x{cam['height']} on {cam['mount_body']}, "
              f"clip {clip})")
    if not depth_spec:
        print("  --  no depth cameras in this run's contract (height-scan or scan-free policy)")

    assert ok, "obs parity mismatch (see FAIL rows above)"
    print("ALL OBS PARITY CHECKS PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None,
                        help="Run dir (or its exported/) to test; default: latest exported run.")
    test_obs_parity(parser.parse_args().run)
