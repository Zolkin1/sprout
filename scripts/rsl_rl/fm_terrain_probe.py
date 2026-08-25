#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Does the generator read a WALL as terrain? One forward pass per condition.

The field observation this exists for: on the first armed hardware run the robot "looked like
it wanted to run at a wall", which reads as a wall being mistaken for a box or a staircase.
``eval/terrain_class.py`` turns "what kind of motion did it plan" into a number; this supplies
the stimulus, under conditions the training set does not contain.

It is worth being precise about why a wall is out of distribution, because it is not an
oversight so much as a measured hole. On the flat tiles -- 30% of training -- ``decorate.py``
records that at the shipped 20 m buffer **0.0% of flat clips have a wall within camera
reach**. On the terrain tiles, 71 of 133 carry a wall at the far end and the motion stops a
median 9.6 m short of it, against a near camera that reaches 8 m. The only close-range
obstacle anywhere is the distractor box, at 0.36 m, and the generator is trained with
``skip_distractors``. So the generator's entire experience of tall geometry inside camera
range is *a box it is about to jump onto* or *a staircase it is about to climb*, and a wall at
2 m resembles both.

**Every env differs only in what its cameras see.** The scenery is DECOR -- no collider, not
in the height scan -- and all N envs replay the same clip at the same step on the same ground,
so the physics, the reference, the command and the privileged scan are identical across them.
``make_terrain_probe_clips.py`` builds that library.

**Probing at ``--settle_steps 0`` is deliberate.** With no history the prediction is a
function of exteroception and command alone, which is the cleanest form of the question. The
harness has already established that the confusion is closed-loop
(0.174 against a 0.104 open-loop and a 0.103 ground-truth control), so a null result here does
NOT exonerate perception -- it says the geometry alone is not sufficient, and the drifted
history is where to look next. A positive result is the stronger read.

**Read every rate against ``empty``, never absolutely.** The classifier has its own confusion
(measured 0.050 stairs->jump, 0.083 flat->jump on held-out clip windows), so only the
difference between a condition and the control means anything.

    python scripts/rsl_rl/make_terrain_probe_clips.py --clip <lib>.h5 --out clips/_terrain_probe.h5
    python scripts/rsl_rl/fm_terrain_probe.py --headless \\
        --fm_checkpoint <fm>/best.pt --clip clips/_terrain_probe.h5 \\
        --terrain_ref terrain_ref.npz --out probe.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="G1-tracker-clf-fm-terrain-depth-play")
parser.add_argument("--fm_checkpoint", required=True, help="Flow-matching .pt.")
parser.add_argument("--clip", required=True, help="Probe library from make_terrain_probe_clips.py.")
parser.add_argument("--terrain_ref", default=None,
                    help="Optional reference .npz from terrain_class.py. Adds the motion-KIND "
                         "readout on top of the plan-difference one. Off by default: the "
                         "paired difference answers the question without needing a label, "
                         "and a classifier can only add its own error to the measurement.")
parser.add_argument("--out", default=None, help="Result .json (default: beside the checkpoint).")
parser.add_argument("--repeats", type=int, default=8,
                    help="Envs per condition. The plan is deterministic given the observation, "
                         "but the depth DR is not, so repeats sample the sensor draw rather "
                         "than the model. Total envs = repeats x conditions.")
parser.add_argument("--settle_steps", type=int, default=0,
                    help="Steps before the probe. 0 = null history, so the prediction is a "
                         "function of exteroception and command alone.")
parser.add_argument("--no_depth_dr", action="store_true",
                    help="Disable the depth sensor DR. REQUIRED to read the stimulus check: "
                         "the mount/intrinsics draw and the per-step noise are per ENV, so "
                         "with DR live two envs differ by their sensor draw as well as by "
                         "their scenery and the depth delta is mostly noise.")
parser.add_argument("--zero_noise", action="store_true",
                    help="Integrate the flow from ZERO instead of a fresh Gaussian draw. "
                         "sample_trajectory redraws its noise every call, so two envs with "
                         "byte-identical conditioning return different plans -- with a "
                         "handful of repeats that sampling noise swamps the effect of the "
                         "scenery entirely. This makes the plan a deterministic function of "
                         "the conditioning, which is the causal read the probe wants; it is "
                         "NOT what deploy does, so a rate over real draws needs many more "
                         "repeats (or paired noise, which sample_window cannot yet accept).")
parser.add_argument("--dump_viz", default=None,
                    help="Write an .npz for view_terrain_probe_viser.py: per condition, the "
                         "depth image each camera returned and the plan the generator drew, "
                         "both alongside the control's. Everything is in the ROBOT's frame "
                         "at the probe instant, which every condition shares.")
parser.add_argument("--clip_device", default="cpu")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import robot_rl.sensors._recursive_contact_sensor_impl  # noqa: F401,E402
import robot_rl.tasks  # noqa: F401,E402
from robot_rl.tasks.manager_based.robot_rl import mdp  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval"))
from terrain_class import COARSE_CLASSES, TerrainClassifier  # noqa: E402
from robot_rl.flow_matching.window import window_block_slices  # noqa: E402


def main() -> int:
    """Build the probe scene, take one plan per env, and classify it."""
    with h5py.File(args_cli.clip, "r") as f:
        spec = json.loads(f.attrs["terrain_probe"])
    conds: list[str] = list(spec["conditions"])
    n_cond, rep = len(conds), max(1, args_cli.repeats)
    n_env = n_cond * rep
    print(f"[terrain-probe] {n_cond} conditions x {rep} envs = {n_env}; "
          f"clip {spec['clip_id']} step {spec['start_step']}")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=n_env)
    env_cfg.commands.traj_ref.path = args_cli.clip
    env_cfg.commands.traj_ref.clip_device = args_cli.clip_device
    env_cfg.scene.terrain.clip_library_path = args_cli.clip
    env_cfg.commands.traj_ref.fm_checkpoint = args_cli.fm_checkpoint
    # K=1: this measures the PREDICTION, not the serving cadence.
    env_cfg.commands.traj_ref.fm_update_every = 1
    # Every env on the same clip at the same step -- the scenery is the only variable.
    env_cfg.events.reset_on_clip.params["pin_clip_ids"] = " ".join(
        str(i % n_cond) for i in range(n_env))
    env_cfg.events.reset_on_clip.params["pin_start_step"] = int(spec["start_step"])
    if args_cli.no_depth_dr:
        # DERIVED from the event functions, not a name list -- a hardcoded guess once matched
        # nothing and left the camera pose draw live through every "DR off" run.
        killed = [n for n, t in list(vars(env_cfg.events).items())
                  if getattr(getattr(t, "func", None), "__module__", "").endswith(
                      "mdp.events.depth")]
        for n in killed:
            setattr(env_cfg.events, n, None)
        if not killed:
            raise SystemExit("--no_depth_dr: this task declares no depth DR events at all.")
        print(f"[terrain-probe] depth DR OFF: {killed}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    terrain = u.scene.terrain

    # env i -> flat tile (i % n_cond) -> decor variant (i % n_cond). Without this the importer
    # assigns tiles at random and every env's condition would be unknown. The probe library
    # holds ONLY flat tiles, so the type index IS the flat-tile ordinal the round-robin uses.
    dev = terrain.terrain_levels.device
    terrain.terrain_levels = torch.zeros(n_env, dtype=torch.long, device=dev)
    terrain.terrain_types = torch.arange(n_env, dtype=torch.long, device=dev) % n_cond
    terrain.env_origins = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    terrain._refresh_env_arrays()
    # No need to push this into the scene: InteractiveScene.env_origins is a read-only
    # property that reads through to the terrain importer.

    cmd = u.command_manager.get_term("traj_ref")
    if args_cli.zero_noise:
        cmd.ctx.zero_noise = True
    # env.reset(), not u.reset(): the settle loop steps through the gymnasium wrapper, whose
    # OrderEnforcing does not see a reset on the unwrapped env and refuses the first step.
    # Latent until settle_steps went above its default of 0.
    env.reset()
    for _ in range(max(0, args_cli.settle_steps)):
        with torch.inference_mode():
            env.step(torch.zeros(n_env, u.action_manager.total_action_dim, device=u.device))

    # FORCE the ray-casters, and read TWICE keeping the second. They carry update_period=0.02
    # and believe they are current, so without this the read returns the PREVIOUS batch's
    # image -- every condition would then be classified from the same stale depth and the
    # probe would report a flat null no matter what the scenery was. This is the same lag
    # augment_registration documents at length; measured there as the whole of a bimodal 50%
    # failure. It is not optional and it is not a slow path.
    groups = mdp.fm_depth_groups(u)
    sensors = [name for name, _ in mdp.find_depth_obs_terms(u, groups)]
    if not sensors:
        raise SystemExit("no depth observation terms on this task; --task must be a depth task.")
    with torch.inference_mode():
        for _read in range(2):
            u.scene.update(u.physics_dt)
            for sname in sensors:
                u.scene.sensors[sname].update(u.physics_dt, force_recompute=True)
            depth = mdp.depth_obs_stack(u, groups)
    depth = (depth if isinstance(depth, torch.Tensor) else torch.as_tensor(depth))
    depth = depth.reshape(n_env, -1).detach().cpu().numpy()

    # STIMULUS CHECK, before any response is read. A condition whose depth is identical to the
    # control's did not reach the cameras, and classifying it would report a null that is a
    # property of this script rather than of the generator. Compared within a repeat index so
    # the per-env depth DR draw is held fixed: env i and env (i % n_cond == 0) of the same
    # repeat differ only in scenery.
    d_ctrl = depth.reshape(rep, n_cond, -1)[:, 0, :]
    seen = {}
    for j, c in enumerate(conds):
        dd = np.abs(depth.reshape(rep, n_cond, -1)[:, j, :] - d_ctrl)
        seen[c] = {"mean_abs_delta": float(dd.mean()),
                   "frac_pixels_changed": float((dd > 1e-6).mean())}
    # Split PER CAMERA and per channel. A whole-row mean hides the thing that matters: the
    # near lens looks 65 deg down and reaches ~2.4 m of ground, so it can only ever see very
    # close geometry, while the far lens at 15 deg is the one that should register a wall at
    # range. Reporting them together made a wall at 1.5 m look "stronger" than an enclosure at
    # 2.5 m, which is an artefact of averaging over a camera that cannot see either.
    cams = mdp.depth_camera_specs(u, groups)
    ch = int(depth.shape[1] // sum(int(c["width"]) * int(c["height"]) for c in cams))
    per_cam, at = [], 0
    for cspec in cams:
        n_px = int(cspec["width"]) * int(cspec["height"])
        # Name it from the SENSOR list rather than the contract dict, which carries mount and
        # intrinsics but no name -- "cam0" in a table is exactly the ambiguity this is about.
        nm = sensors[len(per_cam)] if len(per_cam) < len(sensors) else f"cam{len(per_cam)}"
        per_cam.append((nm.replace("depth_camera_", ""), slice(at, at + ch * n_px), n_px))
        at += ch * n_px
    d3 = depth.reshape(rep, n_cond, -1)
    print("\n=== stimulus check: does each condition actually reach the cameras? ===")
    print("  (delta is normalised depth; sentinel->surface is ~4.7, ground shortening is ~0.1-1)")
    head = "".join(f"{n[:11]:>13s}" for n, _, _ in per_cam)
    print(f"  {'condition':<22s}{head}{'all mean':>11s}{'px changed':>12s}")
    for j, c in enumerate(conds):
        dd = np.abs(d3[:, j, :] - d3[:, 0, :])
        cells = ""
        seen[c]["per_camera"] = {}
        for name, sl, n_px in per_cam:
            # Channel 0 of the channel-major row is the DEPTH plane; the rest are xyz+validity.
            dch = dd[:, sl].reshape(rep, ch, n_px)[:, 0, :]
            seen[c]["per_camera"][name] = {"mean_abs_delta": float(dch.mean()),
                                           "frac_pixels_changed": float((dch > 1e-6).mean())}
            cells += f"{dch.mean():13.4f}"
        print(f"  {c:<22s}{cells}{seen[c]['mean_abs_delta']:11.4f}"
              f"{seen[c]['frac_pixels_changed']:12.3f}")
    blind = [c for c in conds[1:] if seen[c]["frac_pixels_changed"] < 1e-4]
    if blind:
        print(f"  !! {blind} changed no pixels -- their rows below mean nothing.")

    if args_cli.zero_noise:
        # Deterministic, but NOT the mean of the distribution -- integrating the flow from
        # zero is an arbitrary point in it, and measured here it made the empty control
        # classify as "stairs" on bare flat ground. Kept as an option, not the default.
        cmd.ctx.zero_noise = True
        print("[terrain-probe] zero-noise sampling: deterministic, but off-distribution.")
    else:
        # PAIRED noise. sample_trajectory redraws per call, so without this the scenery's
        # effect is measured on top of the sampler's own variance rather than against it.
        # Every condition in a repeat integrates from the SAME rows, which turns a
        # between-subjects comparison into a within-subjects one; repeats then sample the
        # draw. Patched here rather than hooked into the command term, because a probe
        # should not put a test seam in the runtime it is measuring.
        import robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.fm_tracking_cmd as _fmc
        base_noise = torch.randn(rep, u.command_manager.get_term("traj_ref").ctx.F,
                                 cmd.ctx.D, device=u.device)
        paired = base_noise.repeat_interleave(n_cond, dim=0)          # env i -> repeat i//n_cond
        _orig_sample_window = _fmc.sample_window

        def _paired_sample_window(ctx, history, extero, cmd_tokens, history_null_mask=None,
                                  noise=None):
            """Serve the paired rows, and refuse if the batch is not the whole env set."""
            if history.shape[0] != n_env:
                raise RuntimeError(
                    f"paired noise needs every env to re-plan on the same step (got "
                    f"{history.shape[0]} of {n_env}); the pairing maps ROW i to env i and "
                    "a partial batch breaks that. Run with fm_update_every=1 from a reset.")
            return _orig_sample_window(ctx, history, extero, cmd_tokens,
                                       history_null_mask=history_null_mask, noise=paired)

        _fmc.sample_window = _paired_sample_window
        print(f"[terrain-probe] paired noise: {rep} draw(s), each shared by all {n_cond} "
              "conditions.")
    with torch.inference_mode():
        u.command_manager.compute(dt=u.step_dt)
        win = cmd._pred_window.detach().cpu().numpy()          # (n_env, F, D) raw window nodes

    # ---- the measurement: how far did the SCENERY move the plan? ---------------------------
    # No classifier. Every env in a repeat shares the clip, the step, the ground, the command
    # and (paired or zero) the sampling noise, so the only thing separating condition j from
    # the control is its decor -- which makes ``plan_j - plan_0`` attributable by construction
    # and needs no reference set, no k-NN and no confusion matrix to read it against.
    #
    # Reported PER BLOCK in physical units rather than as one norm over the node: a window
    # node concatenates metres, a dimensionless rot6d, radians and rad/s, so a flat norm over
    # it is a units salad whose value depends on the joint count.
    #
    # What the numbers mean, given nothing in this stack avoids obstacles: a WALL should
    # ideally move the plan very little (the robot has no reason to change gait for it),
    # while STAIRS and a BOX should move it a lot. A wall that shifts the plan as much as a
    # staircase does is the misidentification, measured without ever naming a motion.
    slices = window_block_slices(cmd.ctx.nj, cmd.ctx.target_num_extra_frames,
                                 cmd.ctx.reference_blocks)
    w3 = win.reshape(rep, n_cond, win.shape[-2], win.shape[-1])
    base_pos = slices["base_pose"]
    pos_sl = slice(base_pos.start, base_pos.start + 3)
    delta = {}
    for j, c in enumerate(conds):
        d = w3[:, j] - w3[:, 0]                                   # (rep, F, D)
        row = {}
        dp = np.linalg.norm(d[..., pos_sl], axis=-1)              # (rep, F) metres per node
        row["base_pos_mean_m"] = float(dp.mean())
        row["base_pos_max_m"] = float(dp.max(axis=-1).mean())
        row["base_pos_final_m"] = float(dp[:, -1].mean())
        for nm, sl in slices.items():
            row[f"{nm}_rms"] = float(np.sqrt((d[..., sl] ** 2).mean()))
        delta[c] = row

    print("\n=== how far the SCENERY moved the plan (paired against the control) ===")
    print(f"  {'condition':<22s}{'base dxy mean':>15s}{'max':>9s}{'at node F':>11s}"
          f"{'joint rms':>11s}")
    print(f"  {'':<22s}{'(m)':>15s}{'(m)':>9s}{'(m)':>11s}{'(rad)':>11s}")
    for c in conds:
        r = delta[c]
        print(f"  {c:<22s}{r['base_pos_mean_m']:15.4f}{r['base_pos_max_m']:9.4f}"
              f"{r['base_pos_final_m']:11.4f}{r.get('joint_pos_rms', float('nan')):11.4f}")

    if args_cli.dump_viz:
        # Repeat 0 only: with paired (or zero) noise the conditions within a repeat are the
        # comparable set, and one of them is all the eye can take in at a time.
        d3 = depth.reshape(rep, n_cond, -1)[0]
        imgs, hw = [], []
        for _name, sl, n_px in per_cam:
            block = d3[:, sl].reshape(n_cond, ch, n_px)[:, 0, :]        # depth plane only
            cw = next(int(c["width"]) for c in cams
                      if int(c["width"]) * int(c["height"]) == n_px)
            chh = n_px // cw
            imgs.append(block.reshape(n_cond, chh, cw))
            hw.append((chh, cw))
        # base_pose's position block is ALREADY the path relative to the robot at the
        # re-plan, in its yaw-aligned frame -- which is the same frame the conditions were
        # authored in, so plans and boxes overlay with no transform at either end.
        plans = w3[0][:, :, pos_sl]                                     # (n_cond, F, 3)
        # The generator predicts BASE POSE AND JOINTS ONLY -- the served window is
        # 9 + 6 + 29 = 44 wide with no ``ee_pos`` block, even though ``target_frame_names``
        # lists four extra frames (they are tracked, not predicted). So a foot path has to
        # come from forward kinematics of the predicted joints, carried into the probe frame
        # by the predicted base pose; there is no shortcut through the model's own output.
        if "ee_pos" in slices:
            raise SystemExit(
                "this window predicts an ee_pos block, so the viewer's forward kinematics "
                "would be a second opinion on something the model already states -- dump "
                "the block instead of FK'ing the joints.")
        ori6 = w3[0][:, :, slice(pos_sl.stop, base_pos.stop)].astype(np.float32)
        np.savez_compressed(
            args_cli.dump_viz, conditions=np.asarray(conds),
            cameras=np.asarray([n for n, _, _ in per_cam]),
            depth=np.stack(imgs, axis=1), cam_hw=np.asarray(hw),
            plans=plans.astype(np.float32),
            plan_ori6=ori6,
            joint_names=np.asarray([str(j) for j in cmd.ctx.joint_names]),
            plan_joints=w3[0][:, :, slices["joint_pos"]].astype(np.float32),
            clip_band=float(cams[0].get("clip", [0.0, 5.575])[1]),
            conditions_local=json.dumps(spec.get("conditions_local", [])),
            probe=json.dumps(spec))
        print(f"[terrain-probe] wrote {args_cli.dump_viz} for view_terrain_probe_viser.py")

    codes = None
    if args_cli.terrain_ref:
        clf = TerrainClassifier.load(args_cli.terrain_ref)
        codes = np.asarray(clf.classify(win)).reshape(-1)
    # IsaacLab's data views are warp-backed on the GPU pipeline, so this cannot be sliced
    # directly (same reason fm_registration_probe carries a _torch helper).
    root_pos = u.scene["robot"].data.root_pos_w
    if not isinstance(root_pos, torch.Tensor):
        import warp as wp
        root_pos = wp.to_torch(root_pos)
    robot_xy = root_pos[:, :2].detach().cpu().numpy()
    # PLACEMENT CHECK. The conditions are authored relative to the tile CENTRE, using an
    # offset derived offline from the flat re-centring rule. If the runtime offset disagrees,
    # every box is somewhere other than where its name says and the enclosure sanity row is
    # what shows it first.
    cc = terrain.env_cell_center[:, :2].detach().cpu().numpy()
    got = (robot_xy - cc)[0]
    want = np.asarray(spec["offset_xy"], dtype=float)
    print(f"\n[placement] robot - tile_centre: runtime [{got[0]:+.3f}, {got[1]:+.3f}] m vs "
          f"authored [{want[0]:+.3f}, {want[1]:+.3f}] m  (error {np.linalg.norm(got - want):.3f} m)")

    rows = {}
    if codes is None:
        out = args_cli.out or os.path.join(
            os.path.dirname(os.path.abspath(args_cli.fm_checkpoint)), "fm_terrain_probe.json")
        with open(out, "w") as fh:
            json.dump({"conditions": conds, "repeats": rep, "probe": spec,
                       "fm_checkpoint": os.path.abspath(args_cli.fm_checkpoint),
                       "settle_steps": int(args_cli.settle_steps),
                       "zero_noise": bool(args_cli.zero_noise),
                       "stimulus": seen, "delta": delta, "rows": rows,
                       "robot_xy": robot_xy.tolist()}, fh, indent=2)
        print(f"\n[terrain-probe] wrote {out}")
        env.close()
        return 0

    # Per condition: the distribution over planned motion kinds, and the excess over `empty`.
    per = {c: np.zeros(len(COARSE_CLASSES)) for c in conds}
    for i, k in enumerate(codes):
        per[conds[i % n_cond]][int(k)] += 1
    frac = {c: (v / v.sum()).tolist() for c, v in per.items()}
    base = np.asarray(frac[conds[0]])
    terrainish = [COARSE_CLASSES.index("jump"), COARSE_CLASSES.index("stairs")]

    print(f"\n=== planned motion kind, {rep} envs per condition "
          f"(read every row against '{conds[0]}') ===")
    hdr = "".join(f"{c:>9s}" for c in COARSE_CLASSES)
    print(f"  {'condition':<22s}{hdr}{'jump+stairs':>13s}{'vs control':>12s}")
    for c in conds:
        f_ = np.asarray(frac[c])
        t = float(f_[terrainish].sum())
        d = t - float(base[terrainish].sum())
        rows[c] = {"fractions": f_.tolist(), "terrainish": t, "excess_vs_control": d}
        print(f"  {c:<22s}" + "".join(f"{v:9.3f}" for v in f_) + f"{t:13.3f}{d:+12.3f}")

    out = args_cli.out or os.path.join(
        os.path.dirname(os.path.abspath(args_cli.fm_checkpoint)), "fm_terrain_probe.json")
    with open(out, "w") as fh:
        json.dump({"conditions": conds, "repeats": rep, "classes": list(COARSE_CLASSES),
                   "stimulus": seen, "delta": delta, "probe": spec, "fm_checkpoint": os.path.abspath(args_cli.fm_checkpoint),
                   "terrain_ref": os.path.abspath(args_cli.terrain_ref),
                   "settle_steps": int(args_cli.settle_steps),
                   "zero_noise": bool(args_cli.zero_noise),
                   "robot_xy": robot_xy.tolist(), "rows": rows}, fh, indent=2)
    print(f"\n[terrain-probe] wrote {out}")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    sys.exit(code)
