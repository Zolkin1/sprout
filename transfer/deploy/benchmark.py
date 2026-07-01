# Copyright 2026 Zachary Olkin. All rights reserved.

"""Benchmark the torch-free ONNX deploy path vs the torch path (CPU latency).

Times three things and reports mean per-call latency + speedup:
  1. Tracker net      : onnxruntime tracker.onnx  vs  torch.jit policy.pt
  2. FM sampler        : onnxruntime fm.onnx        vs  torch model.encode + Euler loop
  3. End-to-end step   : numpy GenTrackerPolicy     vs  torch GeneratorTrackerPolicy
                         (create_obs + get_action; FM fires every fm_update_every)

Run from ``transfer/sim``:
    python ../deploy/benchmark.py --run_dir <abs run dir> --fm_pt <abs fm .pt>
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "transfer"))
sys.path.insert(0, os.path.join(ROOT, "transfer", "sim"))


def _time(fn, iters: int, warmup: int = 20) -> float:
    """Return mean wall-clock ms per call of ``fn`` over ``iters`` (after warmup)."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e3


def _synthetic_state(nj: int, K: int, rng) -> dict:
    rq = lambda: (lambda v: v / np.linalg.norm(v))(rng.standard_normal(4))
    s = {
        "root_pos": np.array([0.0, 0.0, 0.79]), "root_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "root_lin_vel_w": rng.standard_normal(3) * 0.1, "root_ang_vel_w": rng.standard_normal(3) * 0.1,
        "base_ang_vel": rng.standard_normal(3) * 0.1,
        "joint_pos": rng.standard_normal(nj) * 0.1, "joint_vel": rng.standard_normal(nj) * 0.1,
        "command": np.array([0.5, 0.0, 0.0]),
    }
    if K > 0:
        s.update(frame_pos=rng.standard_normal((K, 3)) * 0.1, frame_quat=np.stack([rq() for _ in range(K)]),
                 frame_lin=np.zeros((K, 3)), frame_ang=np.zeros((K, 3)))
    return s


def main() -> None:
    """Run the three benchmarks and print a comparison table."""
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="Absolute tracker run dir.")
    p.add_argument("--fm_pt", required=True, help="Absolute FM .pt checkpoint (for the torch path).")
    p.add_argument("--iters", type=int, default=300)
    args = p.parse_args()

    import torch
    from deploy.gen_tracker import GenTrackerPolicy
    from gen_tracker_policy import GeneratorTrackerPolicy

    exp = os.path.join(args.run_dir, "exported")
    rng = np.random.default_rng(0)

    # ---- 1. tracker net ----
    import onnxruntime as ort
    from deploy.tracker_session import make_session
    sess = make_session(os.path.join(exp, "tracker.onnx"))
    jit = torch.jit.load(os.path.join(exp, "policy.pt"), map_location="cpu").eval()
    obs_np = rng.standard_normal((1, 207)).astype(np.float32)
    obs_t = torch.from_numpy(obs_np)
    in_name, out_name = sess.get_inputs()[0].name, sess.get_outputs()[0].name
    ort_trk = _time(lambda: sess.run([out_name], {in_name: obs_np}), args.iters * 3)
    with torch.no_grad():
        torch_trk = _time(lambda: jit(obs_t), args.iters * 3)

    # ---- 2. full end-to-end step (the real deploy cost) ----
    npd = GenTrackerPolicy(os.path.join(exp, "tracker.onnx"), os.path.join(exp, "fm.onnx"))
    npd.load()
    nj, K = npd.ctx.nj, npd.ctx.num_extra_frames
    state = _synthetic_state(nj, K, rng)
    npd.warm_start(state)

    pt = GeneratorTrackerPolicy(os.path.join(exp, "policy_parameters.yaml"),
                                os.path.join(exp, "policy.pt"), args.fm_pt, device="cpu")
    pt.load()
    pt.warm_start(state)

    def np_step():
        npd.get_action(npd.create_obs(state), npd.get_joint_names())

    def torch_step():
        pt.get_action(pt.create_obs(state), pt.get_joint_names())

    ort_e2e = _time(np_step, args.iters)
    torch_e2e = _time(torch_step, args.iters)

    def row(name, ort_ms, torch_ms):
        sp = torch_ms / ort_ms if ort_ms > 0 else float("nan")
        print(f"  {name:22s} ORT {ort_ms:7.3f} ms   torch {torch_ms:7.3f} ms   speedup x{sp:.2f}")

    print(f"\n=== CPU latency (mean over {args.iters}+ calls) ===")
    row("tracker net", ort_trk, torch_trk)
    row("end-to-end step", ort_e2e, torch_e2e)
    print("  (end-to-end fires the FM every fm_update_every control steps)")


if __name__ == "__main__":
    main()
