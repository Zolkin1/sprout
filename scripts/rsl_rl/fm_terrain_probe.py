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
parser.add_argument("--terrain_ref", required=True, help="Reference .npz from terrain_class.py.")
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

    u.reset()
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
    print("\n=== stimulus check: does each condition actually reach the cameras? ===")
    print(f"  {'condition':<22s}{'mean |d-delta|':>16s}{'pixels changed':>16s}")
    for c in conds:
        print(f"  {c:<22s}{seen[c]['mean_abs_delta']:16.5f}{seen[c]['frac_pixels_changed']:16.3f}")
    blind = [c for c in conds[1:] if seen[c]["frac_pixels_changed"] < 1e-4]
    if blind:
        print(f"  !! {blind} changed no pixels -- their rows below mean nothing.")

    cmd = u.command_manager.get_term("traj_ref")
    with torch.inference_mode():
        u.command_manager.compute(dt=u.step_dt)
        win = cmd._pred_window.detach().cpu().numpy()          # (n_env, F, D) raw window nodes

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
    rows = {}
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
                   "stimulus": seen, "probe": spec, "fm_checkpoint": os.path.abspath(args_cli.fm_checkpoint),
                   "terrain_ref": os.path.abspath(args_cli.terrain_ref),
                   "settle_steps": int(args_cli.settle_steps),
                   "robot_xy": robot_xy.tolist(), "rows": rows}, fh, indent=2)
    print(f"\n[terrain-probe] wrote {out}")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    sys.exit(code)
