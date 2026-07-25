#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Gate-1 probe (Motion Bricks env): validate the encode/decode round-trip on OUR stride qpos.

Phase-1 already proved the normal generate->predict->decode path. The one genuinely new piece for
Phase 2 is using ``convert_mujoco_qpos_to_motion_transforms`` as an *encoder* on our own optimized
strides (the stride bank from ``export_stride_bank.py``) and driving ``motion_inference.predict``
directly with hand-built constraints. This script isolates exactly that:

    stride qpos window [8,36]  -> converter -> (global_joint_positions/rotations)
      -> build the 8-frame predict inputs EXACTLY as full_agent._generate_inbetween_frames (:400-457)
      -> predict(..., all has_*=True)  -> convert_motion_features_to_mujoco_qpos -> reorder quat
      -> compare the (pinned) context frames of the output to the input window.

A clean round-trip (finite output; low context-frame qpos error) confirms the converter, the channel
layout, the coordinate/quaternion conventions, and the decoder all work on our data -- the green light
to build the full ``mb_encode.py`` driver. Run in the ``motionbricks`` conda env::

    conda run -n motionbricks python scripts/rsl_rl/mb_encode_probe.py \
        --mb-root ../GR00T-WholeBodyControl/motionbricks \
        --stride-bank mb_raw/stride_bank_v2.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch as t

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mb_env  # noqa: E402  (reuse the exact navigation_demo arg layout)

from motionbricks.motionlib.core.utils.rotations import matrix_to_cont6d  # noqa: E402


def _load_bank(path: str):
    """Load a stride bank .npz -> (list of qpos[Fi,36], steady_vel[N,3], periodic[N], meta)."""
    d = np.load(path, allow_pickle=False)
    qpos_all, offsets = d["qpos_all"], d["offsets"]
    strides = [qpos_all[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]
    return strides, d["steady_vel"], d["periodic"], json.loads(str(d["meta"]))


def _build_predict_inputs(agent, ctx_qpos: t.Tensor, tgt_qpos: t.Tensor):
    """Build the 8-frame (4 context + 4 target) predict inputs from two qpos windows.

    Mirrors ``full_navigation_agent._generate_inbetween_frames`` (full_agent.py:400-457), with the
    target treated symmetrically to the context (root taken straight from the converted transforms;
    no spring model, no realignment) so the round-trip is a faithful identity test.
    """
    nfpt, fps, dev = agent.NUM_FRAMES_PER_TOKEN, agent._fps, agent._device
    root = 0
    cgp, cgr = agent._converter.convert_mujoco_qpos_to_motion_transforms(ctx_qpos)  # [1,4,34,3], [1,4,34,3,3]
    tgp, tgr = agent._converter.convert_mujoco_qpos_to_motion_transforms(tgt_qpos)

    def root_values(gp, gr):
        rp = gp[:, :, root, :]                                               # [1,4,3] = [x, y(height), z]
        ang = t.atan2(gr[:, :, root, 0, 2], gr[:, :, root, 2, 2])            # heading about Y
        return t.cat([rp, t.cos(ang)[..., None], t.sin(ang)[..., None]], dim=-1), ang

    def local_root(rv, ang):
        lr = t.zeros([1, nfpt, 4], device=dev)
        lr[:, :nfpt - 1, 0] = (((ang[:, 1:] - ang[:, :-1] + t.pi) % (2 * t.pi)) - t.pi) * fps
        lr[:, :nfpt - 1, 1:3] = (rv[:, 1:, [0, 2]] - rv[:, :-1, [0, 2]]) * fps
        lr[:, :nfpt - 1, 3] = rv[:, :nfpt - 1, 1]
        return lr

    def local_poses(gp, gr, recenter):
        jp = gp[:, :, 1:, :].clone()                                        # drop root -> [1,4,33,3]
        if recenter:
            jp[..., 0] = gp[:, :, 1:, 0] - gp[:, :, :1, 0]
            jp[..., 2] = gp[:, :, 1:, 2] - gp[:, :, :1, 2]
        ortho6d = matrix_to_cont6d(gr)                                      # [1,4,34,6]
        return t.cat([jp.reshape(1, nfpt, -1), ortho6d.reshape(1, nfpt, -1)], dim=-1)  # [1,4,303]

    cgrv, cang = root_values(cgp, cgr)
    tgrv, tang = root_values(tgp, tgr)
    global_root_values = t.cat([cgrv, tgrv], dim=1)                          # [1,8,5]
    local_root_values = t.cat([local_root(cgrv, cang), local_root(tgrv, tang)], dim=1)  # [1,8,4]
    lp = t.cat([local_poses(cgp, cgr, recenter=True), local_poses(tgp, tgr, recenter=False)], dim=1)  # [1,8,303]

    has_g = t.ones_like(global_root_values[:, :, 0], dtype=t.bool)
    has_lr = t.ones_like(local_root_values[:, :, 0], dtype=t.bool)
    has_lp = t.ones_like(lp[:, :, 0], dtype=t.bool)
    has_lr[:, nfpt - 1] = False                                             # context's last velocity is invalid
    return global_root_values, has_g, local_root_values, has_lr, lp, has_lp


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mb-root", required=True, help="GR00T motionbricks package root.")
    p.add_argument("--stride-bank", required=True, help="Stride-bank .npz from export_stride_bank.py.")
    p.add_argument("--controller", default="random")   # unused, keeps _build_demo_args happy
    p.add_argument("--allowed-mode", default=None)
    p.add_argument("--stride-idx", type=int, default=None, help="Which stride to probe (default: pick a fwd-walk one).")
    args = p.parse_args()

    from motionbricks.motion_backbone.demo.utils import navigation_demo

    strides, steady_vel, periodic, meta = _load_bank(args.stride_bank)
    print(f"[probe] bank: {len(strides)} strides, dt={meta.get('dt')}, joint_order={meta.get('joint_order')}")

    # default: a clean forward-walk stride (vx ~ 1.0, small |vy|,|wz|) that is long enough
    if args.stride_idx is None:
        score = np.abs(steady_vel[:, 0] - 1.0) + np.abs(steady_vel[:, 1]) + np.abs(steady_vel[:, 2])
        score[[i for i, s in enumerate(strides) if s.shape[0] < 8]] = 1e9
        idx = int(np.argmin(score))
    else:
        idx = args.stride_idx
    stride = strides[idx]
    print(f"[probe] stride {idx}: '{meta['names'][idx]}' Fi={stride.shape[0]} "
          f"steady_vel={steady_vel[idx].round(3).tolist()} periodic={int(periodic[idx])}")

    demo_agent = navigation_demo(mb_env.build_demo_args(args))
    agent = demo_agent.full_agent
    agent.reset()
    dev = agent._device

    win = t.as_tensor(stride[:8], dtype=t.float32, device=dev)               # [8,36]
    ctx_qpos, tgt_qpos = win[None, :4], win[None, 4:8]                        # [1,4,36] each

    with t.no_grad():
        g, hg, lr, hlr, lp, hlp = _build_predict_inputs(agent, ctx_qpos, tgt_qpos)
        masked = agent._inferencer._root_model.backbone_net.MASKED_NUM_TOKENS
        num_tokens = t.full([1], masked, dtype=t.int32, device=dev)
        config = {"num_inference_step": 1, "smooth_root_traj": False, "allow_pred_out_of_reach_num_tokens": False,
                  "pose_token_sampling_use_argmax": True, "skip_ending_target_cond": agent.SKIP_ENDING_TARGET_COND}
        pred, num_pred_tokens = agent._inferencer.predict(g, hg, lr, hlr, lp, hlp, num_tokens, config=config, info={})
        n_valid = int(agent.NUM_FRAMES_PER_TOKEN * num_pred_tokens.item())
        qpos_out = agent._converter.convert_motion_features_to_mujoco_qpos(pred, agent._motion_rep, False)
        rr = qpos_out[:, :, 3:7].clone()
        qpos_out[:, :, 3:7] = rr[:, :, [3, 0, 1, 2]]                          # [x,y,z,w] -> [w,x,y,z]
    qpos_out = qpos_out[0].cpu().numpy()
    pred = pred.detach().cpu().numpy()

    finite = bool(np.isfinite(pred).all()) and bool(np.isfinite(qpos_out).all())
    print(f"[probe] predict -> pred{tuple(pred.shape)} tokens={int(num_pred_tokens.item())} "
          f"valid_frames={n_valid}  finite={finite}")

    # the 4 context frames are fully pinned (has_*=True) -> the decoded first 4 frames should match input.
    nfpt = agent.NUM_FRAMES_PER_TOKEN
    ctx_in = win[:nfpt].cpu().numpy()
    ctx_out = qpos_out[:nfpt]
    err_root = float(np.abs(ctx_out[:, :3] - ctx_in[:, :3]).mean())
    err_joints = float(np.abs(ctx_out[:, 7:] - ctx_in[:, 7:]).mean())
    quat_dot = float(np.abs((ctx_out[:, 3:7] * ctx_in[:, 3:7]).sum(-1)).mean())  # ~1.0 if aligned
    print(f"[probe] context round-trip (pinned frames):")
    print(f"[probe]   root-pos MAE = {err_root:.4f} m   joint-angle MAE = {err_joints:.4f} rad   "
          f"quat|dot| = {quat_dot:.4f}")
    print(f"[probe]   in  pelvis_z={ctx_in[:, 2].round(3).tolist()}")
    print(f"[probe]   out pelvis_z={ctx_out[:, 2].round(3).tolist()}")
    ok = finite and err_root < 0.10 and err_joints < 0.20 and quat_dot > 0.95
    print(f"[probe] RESULT: {'PASS' if ok else 'CHECK'}  (finite + context reconstructed)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
