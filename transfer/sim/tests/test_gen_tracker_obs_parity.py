# Copyright 2026 Zachary Olkin. All rights reserved.

"""Parity test: the deployed obs assembly vs the real observations.py.

Dev-only (imports isaaclab + robot_rl for the real observation functions). Feeds
*identical* synthetic world references into those functions and into
``deploy.gen_tracker.GenTrackerPolicy._build_obs``, then compares the
reference-derived obs terms elementwise. This pins down the quaternion conventions,
joint/frame orderings, and per-step block layout of the deployment port without
launching the simulator.

Three things it exists to catch, all of which are silent rather than loud:

* the ``reference_window`` joint blocks are in the ROBOT's joint order, while the
  retired ``joint_pos_ref`` was in the CLF-output order -- swapping them still
  produces a correctly-sized observation;
* the window's ``ee_pos`` block is relative to *that step's* reference base, not the
  robot's;
* the proprioception terms are history-stacked oldest-first, and the per-channel
  ``scale`` has to be tiled across that stack.

Everything comes from the exported ``tracker.onnx`` metadata, so the test needs only
that artifact and skips cleanly without one. Run from ``transfer/sim``:
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
DT = 0.02
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


def _rand_quat(n: int, g: torch.Generator) -> torch.Tensor:
    """``n`` uniformly-random unit quaternions ``[n, 4]`` (xyzw)."""
    q = torch.randn(n, 4, generator=g, dtype=torch.float64)
    return q / q.norm(dim=-1, keepdim=True)


class _MockCmd:
    """Minimal command-term stand-in exposing what the window observations read.

    ``reference_window`` / ``command_window`` reach the reference only through the
    ``get_future_*`` accessors and ``robot.data``, so serving those from a fixed truth
    dict is enough to drive the real observation functions with no simulator.
    """

    def __init__(self, ee_frames, joint_names, truth):
        """Wire the accessors to ``truth``.

        Args:
            ee_frames: End-effector frame names, in reference order.
            joint_names: The robot's joint names (the window's joint-block order).
            truth: The synthetic reference tensors shared with the deployment side.
        """
        self._ee_frames = list(ee_frames)
        self.cfg = type("cfg", (), {"base_frame_name": BASE})
        self.manager = type("mgr", (), {"dt": DT})
        self.device = truth["root_quat"].device
        self.robot = type("robot", (), {})()
        self.robot.joint_names = list(joint_names)
        self.robot.data = type("d", (), {})()
        self.robot.data.root_pos_w = truth["root_pos"]
        self.robot.data.root_quat_w = truth["root_quat"]
        self._t = truth

    def resolve_ee_frames(self, frame_names=None):
        """The reference's ee frame names (all of them when ``frame_names`` is None)."""
        return list(frame_names) if frame_names is not None else list(self._ee_frames)

    def get_future_world_base(self, offs):
        """World-frame reference base over the window: ``(pos, quat, lin, ang)``."""
        t = self._t
        return t["w_pos"], t["w_quat"], t["w_lin"], t["w_ang"]

    def get_future_joint_state(self, offs):
        """Reference joint angles + velocities over the window, ``[N, K, nj]`` each."""
        return self._t["w_jp"], self._t["w_jv"]

    def get_future_world_frames(self, frames, offs):
        """World-frame ee states over the window, ``[N, K, len(frames), 13]``."""
        return self._t["w_ee"]

    def get_future_command_velocity(self, offs):
        """Commanded base velocity over the window, ``[N, K, 3]``."""
        return self._t["w_cmd"]


def _term_offsets(pol) -> dict[str, tuple[int, int]]:
    """Map each obs term name to its ``[lo, hi)`` slice of the assembled vector."""
    out, acc = {}, 0
    for t in pol._trk.get_obs_terms():
        out[t["name"]] = (acc, acc + t["shape"])
        acc += t["shape"]
    return out


def _apply_export_postprocess(pol, name: str, raw: torch.Tensor) -> torch.Tensor:
    """Clip and scale ``raw`` the way the exported term table says training did.

    The Isaac observation functions return the raw quantity; ``_assemble`` applies the
    term's ``clip`` then its ``scale``. Comparing against the raw value would let a
    dropped clip or a mis-broadcast per-channel scale pass.

    Args:
        pol: The loaded :class:`GenTrackerPolicy`.
        name: Observation term name.
        raw: The Isaac function's output, ``[N, dim]``.

    Returns:
        The post-processed tensor, ``[N, dim]``.
    """
    term = next(t for t in pol._trk.get_obs_terms() if t["name"] == name)
    out = raw
    if term["clip"] is not None:
        out = out.clamp(-term["clip"], term["clip"])
    scale = term["scale"]
    if isinstance(scale, (list, tuple, np.ndarray)):
        scale = torch.as_tensor(np.asarray(scale, dtype=np.float64))
    return out * scale


def test_obs_parity(run: str | None = None) -> None:
    """Compare the reference obs terms: Isaac functions vs the deployment port."""
    exported = _find_run(run)
    if exported is None:
        print("[SKIP] No exported tracker.onnx found under logs/rsl_rl/*/*/exported/; "
              "export one by running scripts/rsl_rl/play.py.")
        return
    print(f"[INFO] tracker: {exported}")

    # Tracker-only: no fm.onnx, so the reference is injected directly (as clip playback
    # does), and the FM->isaac joint permutation is the identity.
    pol = GenTrackerPolicy(os.path.join(exported, "tracker.onnx"), None)
    pol.load()
    isaac_jn = pol.get_joint_names()
    nj = len(isaac_jn)
    ee_frames = pol._trk.get_ee_frame_names() or []
    spec = pol.reference_window_spec
    num_steps, stride = int(spec["num_steps"]), int(spec["stride"])
    offs = pol.window_offsets
    K = len(offs)
    depth_spec = pol.get_depth_camera_spec()
    term_dims = {t["name"]: t["shape"] for t in pol._trk.get_obs_terms()}
    term_off = _term_offsets(pol)

    print(f"window: num_steps={num_steps} stride={stride} offsets={offs}")
    print(f"blocks: {spec['blocks']}  ee_frames={ee_frames}  joint_order={spec['joint_order']}")

    g = torch.Generator().manual_seed(3)
    n_ee = len(ee_frames)
    truth = {
        "root_pos": torch.randn(N, 3, generator=g, dtype=torch.float64),
        "root_quat": _rand_quat(N, g),
        "w_pos": torch.randn(N, K, 3, generator=g, dtype=torch.float64),
        "w_quat": _rand_quat(N * K, g).reshape(N, K, 4),
        "w_lin": torch.randn(N, K, 3, generator=g, dtype=torch.float64),
        "w_ang": torch.randn(N, K, 3, generator=g, dtype=torch.float64),
        "w_jp": torch.randn(N, K, nj, generator=g, dtype=torch.float64),
        "w_jv": torch.randn(N, K, nj, generator=g, dtype=torch.float64),
        "w_cmd": torch.randn(N, K, 3, generator=g, dtype=torch.float64),
    }
    # World ee states [N, K, n_ee, 13]; only the leading pos(3) is read, but fill it all
    # so a port that grabbed the wrong 3 columns would be caught.
    truth["w_ee"] = torch.randn(N, K, max(n_ee, 1), 13, generator=g, dtype=torch.float64)[:, :, :n_ee]

    cmd = _MockCmd(ee_frames, isaac_jn, truth)
    env = type("env", (), {"num_envs": N, "device": "cpu",
                           "command_manager": type("cm", (), {"get_term": lambda self, n: cmd})()})()

    # --- Isaac reference terms (the real observation functions) ---
    orig = wp.to_torch
    wp.to_torch = lambda a: a if isinstance(a, torch.Tensor) else orig(a)
    try:
        isaac_terms = {
            "reference_window": O.reference_window(env, "x", num_steps, stride),
            "command_window": O.command_window(env, "x", num_steps, stride),
        }
    finally:
        wp.to_torch = orig
    isaac_terms = {k: _apply_export_postprocess(pol, k, v) for k, v in isaac_terms.items()}

    # --- The deployment port: same reference per env, then slice the assembled obs ---
    # Synthetic depth, one row per camera in contract order (what the caster returns), with
    # distinct content per camera so a swapped or misplaced image shows up.
    rng = np.random.default_rng(0)
    depth_images = np.stack([
        rng.uniform(-1.0, 1.0, int(cam["width"]) * int(cam["height"])) for cam in depth_spec
    ]) if depth_spec else None

    mine = {k: [] for k in isaac_terms}
    obs_last = None
    for e in range(N):
        reference = {"window": [{
            "root_pos": truth["w_pos"][e, k], "root_quat": truth["w_quat"][e, k],
            "root_lin": truth["w_lin"][e, k], "root_ang": truth["w_ang"][e, k],
            "jp": truth["w_jp"][e, k], "jv": truth["w_jv"][e, k],
            "frame_pos": truth["w_ee"][e, k, :, 0:3],
            "command": truth["w_cmd"][e, k],
        } for k in range(K)]}
        state = {
            "root_pos": truth["root_pos"][e].numpy(), "root_quat": truth["root_quat"][e].numpy(),
            "base_ang_vel": np.zeros(3), "joint_pos": np.zeros(nj), "joint_vel": np.zeros(nj),
            "command": truth["w_cmd"][e, 0].numpy(),
        }
        if depth_images is not None:
            state["depth"] = depth_images
        if "height_scan" in term_dims:
            state["height_scan"] = np.zeros(term_dims["height_scan"])
        pol._trk.action_isaac = np.zeros(nj)
        # Each env is an independent comparison, so the history must not carry across them.
        pol.reset()
        obs_last = pol._build_obs(state, reference)[0]
        for name in isaac_terms:
            lo, hi = term_off[name]
            mine[name].append(obs_last[lo:hi])

    ok = True
    for name in isaac_terms:
        a = torch.tensor(np.stack(mine[name]), dtype=torch.float64)
        b = isaac_terms[name].to(torch.float64)
        if a.shape != b.shape:
            print(f"  FAIL {name:22s} shape {tuple(a.shape)} != isaac {tuple(b.shape)}")
            ok = False
            continue
        err = (a - b).abs().max().item()
        status = "ok " if err < 1e-5 else "FAIL"
        ok = ok and err < 1e-5
        print(f"  {status} {name:22s} dim={term_dims[name]:<5} max_err={err:.2e}")

    ok = _check_history(pol, term_off, term_dims, nj) and ok

    # Depth placement: each camera's image must occupy its own slice, clipped as trained
    # (the clip comes from the term's own exported metadata, like every other term's).
    term_clip = {t["name"]: t["clip"] for t in pol._trk.get_obs_terms()}
    for i, cam in enumerate(depth_spec or []):
        term = cam["obs_term"]
        lo, hi = term_off[term]
        want = np.asarray(depth_images[i], dtype=np.float32)
        clip = term_clip[term]
        if clip is not None:
            want = np.clip(want, -clip, clip)
        err = float(np.abs(obs_last[lo:hi] - want).max())
        status = "ok " if err < 1e-6 else "FAIL"
        ok = ok and err < 1e-6
        print(f"  {status} {term:22s} dim={term_dims[term]:<5} max_err={err:.2e} "
              f"(slice {lo}:{hi}, {cam['width']}x{cam['height']} on {cam['mount_body']}, "
              f"clip {clip})")
    if not depth_spec:
        print("  --  no depth cameras in this run's contract (height-scan or scan-free policy)")

    assert ok, "obs parity mismatch (see FAIL rows above)"
    print("ALL OBS PARITY CHECKS PASSED")


def _check_history(pol, term_off, term_dims, nj) -> bool:
    """Verify the observation-history stacking against Isaac Lab's CircularBuffer rules.

    Two properties, both silent when wrong: the FIRST sample must fill the whole window
    (so step 0 is a static window, not zero-padded), and the flattened window must read
    OLDEST-first. A reversed window still has the right width and the right values.

    Uses ``base_ang_vel`` (an unclipped, scale-1.0, 3-wide term) so the check is about
    ordering rather than post-processing.

    Args:
        pol: The loaded :class:`GenTrackerPolicy`.
        term_off: Term name -> ``[lo, hi)`` slice map.
        term_dims: Term name -> flat width map.
        nj: Joint count, for the synthetic state.

    Returns:
        True if the history behaves as Isaac Lab does.
    """
    term = next((t for t in pol._trk.get_obs_terms()
                 if t["name"] == "base_ang_vel" and t["history_length"] > 0), None)
    if term is None:
        print("  --  no history_length on base_ang_vel; history stacking not exercised")
        return True
    hist = term["history_length"]
    lo, hi = term_off["base_ang_vel"]
    pol.reset()

    def _obs_at(step: int) -> np.ndarray:
        """One control step whose base_ang_vel is the constant ``step``."""
        state = {
            "root_pos": np.zeros(3), "root_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "base_ang_vel": np.full(3, float(step)),
            "joint_pos": np.zeros(nj), "joint_vel": np.zeros(nj), "command": np.zeros(3),
        }
        if "height_scan" in term_dims:
            state["height_scan"] = np.zeros(term_dims["height_scan"])
        if pol.get_depth_camera_spec():
            state["depth"] = np.stack([np.zeros(int(c["width"]) * int(c["height"]))
                                       for c in pol.get_depth_camera_spec()])
        ref = {"window": [{
            "root_pos": np.zeros(3), "root_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "root_lin": np.zeros(3), "root_ang": np.zeros(3),
            "jp": np.zeros(pol.ctx.nj), "jv": np.zeros(pol.ctx.nj),
            "frame_pos": np.zeros((pol.ctx.target_num_extra_frames, 3)),
        } for _ in pol.window_offsets]}
        pol._trk.action_isaac = np.zeros(nj)
        return pol._build_obs(state, ref)[0][lo:hi]

    ok = True
    first = _obs_at(1)
    if not np.allclose(first, 1.0):
        print(f"  FAIL history warm-start: first push must fill all {hist} slots, got {first}")
        ok = False
    else:
        print(f"  ok  history warm-start     first push fills all {hist} slots")

    for step in range(2, hist + 2):
        got = _obs_at(step)
    # After pushing 1..hist+1, the window holds the last `hist` values, oldest first.
    want = np.repeat(np.arange(step - hist + 1, step + 1, dtype=np.float32), 3)
    if not np.allclose(got, want):
        print(f"  FAIL history order        got {got[:6]}... want {want[:6]}... "
              "(must be OLDEST-first, matching CircularBuffer.buffer)")
        ok = False
    else:
        print(f"  ok  history order         oldest-first over {hist} steps")
    pol.reset()
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None,
                        help="Run dir (or its exported/) to test; default: latest exported run.")
    test_obs_parity(parser.parse_args().run)
