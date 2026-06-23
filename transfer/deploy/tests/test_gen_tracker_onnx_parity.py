# Copyright 2026 Zachary Olkin. All rights reserved.

"""Parity: numpy+ORT GenTrackerPolicy obs vs the torch GeneratorTrackerPolicy obs.

Dev-only. Feeds an identical synthetic (state, FM-reference) into both the
torch obs builder (``transfer/sim/gen_tracker_policy``, validated against Isaac
to ~1e-7) and the torch-free deploy builder (``transfer/deploy/gen_tracker``),
and asserts the 207-dim observation matches. The CLF output joint order is
injected into both (the TM2 export predates ``joint_out_names``).

Requires (build first):
  - tracker.onnx : python scripts/rsl_rl/export_tracker_onnx.py --run_dir <run>
  - fm.onnx      : python scripts/flow_matching/export_fm_onnx.py --checkpoint <ckpt> --out <run>/exported/fm.onnx

Run: ``python transfer/deploy/tests/test_gen_tracker_onnx_parity.py``
"""

from __future__ import annotations

import json
import os
import sys

import h5py
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "transfer"))         # deploy, fm
sys.path.insert(0, os.path.join(ROOT, "transfer", "sim"))  # gen_tracker_policy (torch)

from deploy.gen_tracker import GenTrackerPolicy  # noqa: E402
from gen_tracker_policy import GeneratorTrackerPolicy  # noqa: E402

CLIP = os.path.join(ROOT, "clips", "g1_flat_v1.h5")
RUN = os.path.join(ROOT, "logs/rsl_rl/g1/2026-06-22_09-47-52_TM2_generator_tracking_test/exported")
FM_PT = os.path.join(ROOT, "logs/flow_matching/2026-06-21_17-07-59/ckpt_89.pt")
N_NJ = 21


def _clip_joint_out() -> list[str]:
    with h5py.File(CLIP, "r") as h:
        pos = json.loads(str(np.asarray(h.attrs["pos_output_names"]).item()))
    return [n.split(":", 1)[1] for n in pos if n.startswith("joint:")]


def test_gen_tracker_onnx_parity() -> None:
    """numpy+ORT obs == torch obs on the same synthetic state + FM reference."""
    joint_out = _clip_joint_out()

    # torch oracle (policy.pt + fm .pt); inject the CLF joint order.
    pt = GeneratorTrackerPolicy(f"{RUN}/policy_parameters.yaml", f"{RUN}/policy.pt", FM_PT, device="cpu")
    pt._rl.policy_params["joint_out_names"] = joint_out
    pt.load()

    # torch-free deploy (tracker.onnx + fm.onnx); inject the same order.
    npd = GenTrackerPolicy(f"{RUN}/tracker.onnx", f"{RUN}/fm.onnx")
    npd.load()
    isaac = npd.get_joint_names()
    npd.joint_out_names = joint_out
    npd._fm_to_out = np.array([isaac.index(n) for n in joint_out], dtype=np.int64)

    nj, K, off = N_NJ, pt.ctx.target_num_extra_frames, pt.future_offsets
    g = np.random.default_rng(4)
    rq = lambda: (lambda v: v / np.linalg.norm(v))(g.standard_normal(4))

    state = {
        "root_pos": g.standard_normal(3), "root_quat": rq(),
        "base_ang_vel": g.standard_normal(3),
        "joint_pos": g.standard_normal(nj), "joint_vel": g.standard_normal(nj),
        "command": g.standard_normal(3),
    }
    cur = {"root_pos": g.standard_normal(3), "root_quat": rq(),
           "jp": g.standard_normal(nj), "jv": g.standard_normal(nj),
           "frame_pos": g.standard_normal((K, 3))}
    future = [{"root_pos": g.standard_normal(3), "root_quat": rq(),
               "root_lin": g.standard_normal(3), "root_ang": g.standard_normal(3)} for _ in off]

    # numpy reference (arrays) and torch reference (tensors) from the SAME values.
    ref_np = {"current": cur, "future": future}
    ref_t = {
        "current": {k: torch.tensor(v) for k, v in cur.items()},
        "future": [{k: torch.tensor(v) for k, v in s.items()} for s in future],
    }
    pt._rl.action_isaac = np.zeros(nj)
    npd._trk.action_isaac = np.zeros(nj)

    obs_t = pt._build_obs(state, ref_t)[0].detach().numpy()
    obs_n = npd._build_obs(state, ref_np)[0]

    # term-by-term report
    off_i, ok = 0, True
    for term in npd._trk.get_obs_terms():
        name, shape = term["name"], term["shape"]
        err = float(np.abs(obs_t[off_i:off_i + shape] - obs_n[off_i:off_i + shape]).max())
        status = "ok " if err < 1e-5 else "FAIL"
        if err >= 1e-5:
            ok = False
        print(f"  {status} {name:22s} dim={shape:<3} max_err={err:.2e}")
        off_i += shape
    print(f"TOTAL obs max_err={np.abs(obs_t - obs_n).max():.2e}")
    assert ok, "numpy+ORT obs != torch obs (see FAIL rows)"
    print("ALL GEN-TRACKER ONNX PARITY CHECKS PASSED")


if __name__ == "__main__":
    test_gen_tracker_onnx_parity()
