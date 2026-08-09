#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Augment a recording by DISPLACING THE ROBOT, so the generator has to localise terrain.

The generator was measured not to register terrain at all: displace the robot downrange and
its predicted stair segment slides along with it (0.098 / 0.168 m for a 0.10 / 0.20 m shift,
where the anchor mask requires 0.000 -- ``fm_registration_probe.py``). The cause is in the
data, not the model. Every recorded window has the robot sitting exactly on its clip, so
depth and the correct answer are perfectly correlated and depth can be ignored outright.

This breaks that correlation offline. For each source window it re-places the robot at its
recorded pose PLUS a sampled ``(downrange, lateral, yaw)`` offset, re-casts the depth cameras
from there, and writes a new window. No physics, no controller, no rollout -- Isaac is used
only as a renderer.

What changes, and what deliberately does not:

* ``depth`` -- re-rendered from the displaced pose. The ONLY channel carrying the offset.
* ``history_state[H-1]`` root position + yaw -- the SNAP ANCHOR, the frame the dataset
  anchors ``ref_future`` against (``world_to_window``'s anchor is YAW-ONLY, so roll/pitch
  there are never read). Displacing it is what turns the unchanged world-frame clip into a
  CORRECTION back onto the terrain. No target arithmetic, so no sign to get backwards.
* ``ref_future`` / ``cmd_future`` -- untouched. They are stored in WORLD; the clip does not
  move just because the robot did.
* the HISTORY (rows ``H-1-n .. H-2`` and all of ``history_ref``) -- untouched, and this is
  load-bearing rather than incidental. The history is the generator's own previous output
  expressed relative to its own frame, so it is invariant to a rigid displacement anyway --
  and if the offset DID leak into it, the model could recover the offset without ever
  consulting depth, which is precisely the shortcut this augmentation exists to remove.

Terrain must be IDENTICAL to the recording: the recorded poses are absolute world, so the
tiles have to be where they were. The library is taken from the recording's own
``clip_library_path`` and its sha256 is verified, and ``--num_envs`` defaults to the
recording's, because env origins enter the tile layout.

    # 1. prove the pipeline: re-render at ZERO offset and diff against the recorded depth
    python scripts/rsl_rl/augment_registration.py --headless --data <rec>.h5 --check_only

    # 2. then augment
    python scripts/rsl_rl/augment_registration.py --headless --data <rec>.h5 \\
        --out <aug>.h5 --num_augments 2 --downrange 0.15 --lateral 0.20 --yaw 0.15
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="G1-teacher-clf-terrain-depth")
parser.add_argument("--data", required=True, help="Recording to augment (needs clip_id).")
parser.add_argument("--out", default=None, help="Output .h5 (omit with --check_only).")
parser.add_argument("--num_augments", type=int, default=1,
                    help="New points generated per SOURCE point.")
parser.add_argument("--downrange", type=float, default=0.15,
                    help="Symmetric +- bound (m) along the terrain's traversal axis. Keep "
                         "SMALL: past roughly a stride the correct behaviour stops being "
                         "'the same trajectory, corrected' and becomes 'take another step "
                         "first', so a large bound teaches the terrain motion to start early.")
parser.add_argument("--lateral", type=float, default=0.20,
                    help="Symmetric +- bound (m) across it. Tolerates more than downrange -- "
                         "the anchor mask leaves lateral FREE on terrain segments.")
parser.add_argument("--yaw", type=float, default=0.15,
                    help="Symmetric +- bound (rad) on heading. The mask PINS yaw on terrain, "
                         "so a yaw offset is a correction the generator must undo.")
parser.add_argument("--max_source", type=int, default=0,
                    help="Cap source windows processed (0 = all). Augmented output is this "
                         "times --num_augments.")
parser.add_argument("--batch", type=int, default=0,
                    help="Windows per render pass (0 = use num_envs).")
parser.add_argument("--num_envs", type=int, default=0,
                    help="0 = the recording's own num_envs, which is what reproduces the "
                         "tile layout the recorded world poses live in.")
parser.add_argument("--check_only", action="store_true",
                    help="Re-render at ZERO offset and report the diff against the recorded "
                         "depth. Proves placement, tile alignment and sensor refresh before "
                         "any augmented number is trusted. Writes nothing.")
parser.add_argument("--no_depth_dr", action="store_true",
                    help="Disable depth sensor DR (mount randomisation + noise) for this run. "
                         "REQUIRED to interpret --check_only: the recording carries "
                         "extero_corrupted=True, so a fresh render draws different noise and "
                         "the delta-0 diff can never be zero with DR live.")
parser.add_argument("--quat_wxyz", action="store_true",
                    help="Write the root quaternion permuted to wxyz. A/B for the convention: "
                         "the recording states xyzw, the sim API's expectation is what this "
                         "settles empirically against a DR-free delta-0 check.")
parser.add_argument("--env_seed", type=int, default=42,
                    help="Env seed; MUST equal the recording's --seed or the terrain tiles "
                         "are laid out differently and absolute world poses are meaningless.")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# After the SimulationApp, or the per-link contact sensors find no bodies (same as play.py).
import robot_rl.sensors._recursive_contact_sensor_impl  # noqa: F401,E402
import robot_rl.tasks  # noqa: F401,E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from recorded_data_io import (  # noqa: E402
    RecordedWindowWriter,
    load_recorded_windows,
)


def term_slice(names, dims, target: str) -> slice:
    """Column slice of a named term inside a concatenated node row."""
    off = 0
    for n, d in zip(names, dims):
        if n == target:
            return slice(off, off + d)
        off += d
    raise KeyError(f"{target!r} not in {names}")


def yaw_of(quat_xyzw: np.ndarray) -> np.ndarray:
    """Heading (rad) from an xyzw quaternion array ``[..., 4]``."""
    x, y, z, w = (quat_xyzw[..., i] for i in range(4))
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_to_quat(yaw: np.ndarray) -> np.ndarray:
    """xyzw quaternion for a pure-yaw rotation."""
    out = np.zeros(yaw.shape + (4,), dtype=np.float32)
    out[..., 2] = np.sin(yaw / 2.0)
    out[..., 3] = np.cos(yaw / 2.0)
    return out


def quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of xyzw quaternions, ``a * b``."""
    ax, ay, az, aw = (a[..., i] for i in range(4))
    bx, by, bz, bw = (b[..., i] for i in range(4))
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def main() -> int:
    """Re-render depth at displaced robot poses and write the augmented recording."""
    import h5py

    rng = np.random.default_rng(args_cli.seed)
    data, meta = load_recorded_windows(args_cli.data)
    W = int(data["history_state"].shape[0])
    H = int(data["history_state"].shape[1])

    lib = meta.clip_library_path
    if not lib:
        raise SystemExit("recording carries no clip_library_path; re-record with the current "
                         "record_data.py.")
    if meta.clip_library_sha256:
        import hashlib
        got = hashlib.sha256(open(lib, "rb").read()).hexdigest()
        if got != meta.clip_library_sha256:
            raise SystemExit(
                f"clip library '{lib}' is sha256 {got[:12]} but the recording was made "
                f"against {meta.clip_library_sha256[:12]}. The recorded poses are ABSOLUTE "
                "world, so a different library puts the tiles somewhere else and every "
                "re-rendered depth image would be of the wrong geometry.")

    n_env = args_cli.num_envs or int(meta.num_envs)
    batch = args_cli.batch or n_env
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=n_env)
    # The env SEED must match the recording's. ClipTerrainImporter's tile->world layout is
    # built at env creation, so an unseeded run lays the tiles out differently and the
    # recording's ABSOLUTE world poses then land on different geometry entirely.
    env_cfg.seed = int(args_cli.env_seed)
    env_cfg.commands.traj_ref.path = lib
    env_cfg.commands.traj_ref.clip_device = "cpu"
    terrain = getattr(getattr(env_cfg, "scene", None), "terrain", None)
    if terrain is not None and hasattr(terrain, "clip_library_path"):
        terrain.clip_library_path = lib
    if args_cli.no_depth_dr:
        # The REAL event names. An earlier guess ("randomize_depth_mount_*") matched nothing,
        # so camera EXTRINSICS/INTRINSICS randomization stayed live through every "DR off" run
        # -- a per-episode camera pose draw, which is why the re-render came out vertically
        # shifted while the scene structure matched.
        for ev in ("resample_depth_noise_near", "resample_depth_noise_far",
                   "randomize_camera_extrinsics_near", "randomize_camera_extrinsics_far",
                   "randomize_camera_intrinsics_near", "randomize_camera_intrinsics_far"):
            if not hasattr(env_cfg.events, ev):
                raise SystemExit(f"--no_depth_dr: event '{ev}' not on this task's cfg; the "
                                 "name list is stale and DR would silently stay ON.")
            setattr(env_cfg.events, ev, None)
    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    robot = u.scene["robot"]
    def to_torch(x):
        """Sim data view (torch / warp / numpy) as a torch tensor."""
        if isinstance(x, torch.Tensor):
            return x
        try:
            import warp as wp
            return wp.to_torch(x)
        except Exception:
            return torch.as_tensor(np.asarray(x), device=u.device)

    u.reset()

    # Joint ORDER: the recording's joint_pos columns are meaningless unless they line up with
    # this articulation's. A silent mismatch would scramble the posture, which moves the
    # torso-mounted cameras as well as the self-scan -- i.e. it would corrupt exactly what we
    # are trying to render.
    rec_joints = list(meta.state_joint_names)
    if rec_joints != list(robot.joint_names):
        raise SystemExit(
            "joint order mismatch between the recording and the robot.\n"
            f"  recording: {rec_joints[:4]} ... ({len(rec_joints)})\n"
            f"  robot    : {list(robot.joint_names)[:4]} ... ({len(robot.joint_names)})")

    # clip_id -> terrain_id -> the envs sitting on that tile. This is the mapping the RL's own
    # reset uses, so it is exact; picking a tile by PROXIMITY to the recorded pose would be a
    # guess that fails whenever the robot is far along its tile.
    terr = u.scene.terrain
    env_tid = to_torch(terr.env_terrain_id).detach().cpu().numpy().reshape(-1)
    env_org = to_torch(terr.env_origins).detach().cpu().numpy().reshape(-1, 3)
    with h5py.File(lib, "r") as _lf:
        clip_tid = _lf["terrain_id"][:]
    envs_for_tid: dict[int, np.ndarray] = {
        int(t): np.flatnonzero(env_tid == int(t)) for t in np.unique(env_tid)
    }
    print(f"[augment] {len(envs_for_tid)} distinct terrain ids across {len(env_tid)} envs")

    names, dims = list(meta.state_term_names), list(meta.state_term_dims)
    s_pos, s_quat = term_slice(names, dims, "root_pos_w"), term_slice(names, dims, "root_quat_w")
    s_jp = term_slice(names, dims, "joint_pos")

    src = np.arange(W if not args_cli.max_source else min(W, args_cli.max_source))
    reps = 1 if args_cli.check_only else int(args_cli.num_augments)
    hs_all = data["history_state"]

    writer = None
    if not args_cli.check_only:
        if not args_cli.out:
            raise SystemExit("--out is required unless --check_only.")
        depth_shape = (int(data["depth"].shape[1]), int(data["depth"].shape[2]))
        writer = RecordedWindowWriter(
            args_cli.out, len(src) * reps, H, int(hs_all.shape[2]),
            int(data["ref_future"].shape[1]),
            int(data["height_scan"].shape[1]) if "height_scan" in data else 0,
            int(data["cmd_future"].shape[2]) if "cmd_future" in data else 0,
            meta, depth_shape=depth_shape, tag_distractors=True, tag_families=True,
            tag_clip_index=True, record_history_ref=True,
        )

    diffs = []
    samples = []
    perwin = []
    allren = []
    allsrc = []
    n_done = 0
    for rep in range(reps):
        for lo in range(0, len(src), batch):
            idx = src[lo:lo + batch]
            k = len(idx)
            snap = hs_all[idx, H - 1, :].astype(np.float64)
            pos = snap[:, s_pos].copy()
            quat = snap[:, s_quat].copy()
            jp = snap[:, s_jp].copy()

            if args_cli.check_only:
                d_dr = d_lat = d_yaw = np.zeros(k)
            else:
                d_dr = rng.uniform(-args_cli.downrange, args_cli.downrange, k)
                d_lat = rng.uniform(-args_cli.lateral, args_cli.lateral, k)
                d_yaw = rng.uniform(-args_cli.yaw, args_cli.yaw, k)
            # Offsets are in the ROBOT's own heading frame -- on the axis-aligned tiles built
            # today this equals the terrain frame the anchor mask names its DOFs in, and it
            # stays meaningful if a tile is ever rotated.
            hd = yaw_of(quat)
            new_pos = pos.copy()
            new_pos[:, 0] += d_dr * np.cos(hd) - d_lat * np.sin(hd)
            new_pos[:, 1] += d_dr * np.sin(hd) + d_lat * np.cos(hd)
            new_quat = quat_mul_xyzw(yaw_to_quat(d_yaw).astype(np.float64), quat)

            # Place: full pose (the cameras are torso-mounted, so roll/pitch matter to the
            # render) and the recorded joint pose. Envs beyond k keep whatever they had; only
            # the first k rows are read back.
            # Route each window to an env whose tile has the SAME terrain_id, keeping the
            # pose TILE-LOCAL. Tiles of one terrain_id are geometrically identical, so which
            # instance we land on cannot change the depth image -- only the terrain_id can.
            # The source env is RECORDED, so there is no instance to infer. clip_id ->
            # terrain_id is still checked below, as a guard that the layout really matches.
            src_envs = np.asarray(data["env_id"])[idx].astype(int)
            tids = clip_tid[np.asarray(data["clip_id"])[idx]]
            slots, keep = [], []
            used_envs: set[int] = set()
            for j, t in enumerate(tids):
                e = int(src_envs[j])
                if e < 0 or e >= len(env_tid):
                    continue
                if int(env_tid[e]) != int(t):
                    raise SystemExit(
                        f"env {e} sits on terrain {int(env_tid[e])} but window {j}'s clip "
                        f"wants terrain {int(t)}. The tile layout does not match the "
                        "recording -- check that --env_seed and --num_envs equal the "
                        "recording's, or the recorded world poses are meaningless here.")
                if e in used_envs:
                    continue                      # one window per env per render pass
                used_envs.add(e)
                slots.append((e, e))              # place back into the very env it came from
                keep.append(j)
            if not keep:
                continue
            keep = np.asarray(keep)
            tgt_env = np.array([a for a, _ in slots])
            src_env = np.array([b for _, b in slots])
            local = new_pos[keep] - env_org[src_env]
            place_pos = local + env_org[tgt_env]

            root = to_torch(robot.data.root_state_w).clone()
            ti = torch.as_tensor(tgt_env, device=root.device, dtype=torch.long)
            root[ti, 0:3] = torch.as_tensor(place_pos, device=root.device, dtype=root.dtype)
            # CONVENTION: the recording states quats as xyzw (meta.state_quat_format), while
            # `root_state_w` carries them wxyz. Writing one into the other silently yields a
            # ~180 deg rotation for a near-upright robot -- the camera points elsewhere and the
            # body appears mirrored front-to-back, which is exactly what the first visual diff
            # showed. Converted here, and ROUND-TRIPPED below so the convention can never be
            # guessed wrong again.
            q_write = (new_quat[keep][:, [3, 0, 1, 2]] if args_cli.quat_wxyz
                       else new_quat[keep])
            root[ti, 3:7] = torch.as_tensor(q_write, device=root.device, dtype=root.dtype)
            root[ti, 7:] = 0.0
            robot.write_root_state_to_sim(root)
            if not hasattr(main, "_quat_checked"):
                # SEMANTIC check, not a round trip: does the robot now HOLD the recorded
                # orientation? Comparing what we wrote against what we read back proves only
                # that the API stores what it is given -- it round-trips under ANY convention
                # and so cannot catch a convention error at all.
                back = to_torch(robot.data.root_quat_w)[ti].detach().cpu().numpy()
                rec_q = new_quat[keep]
                d_same = np.abs(np.abs((back * rec_q).sum(-1)) - 1.0).max()
                d_conv = np.abs(np.abs((back * rec_q[:, [3, 0, 1, 2]]).sum(-1)) - 1.0).max()
                print(f"[augment] quat check: |read . recorded_xyzw|-1 = {d_same:.5f}, "
                      f"vs wxyz-permuted = {d_conv:.5f}  (the SMALLER names the convention "
                      f"root_quat_w reports in)")
                main._quat_checked = True
            jpos = to_torch(robot.data.joint_pos).clone()
            jpos[ti] = torch.as_tensor(jp[keep], device=jpos.device, dtype=jpos.dtype)
            robot.write_joint_position_to_sim(jpos)
            robot.write_joint_velocity_to_sim(torch.zeros_like(jpos))
            # Target the written posture so the PD does not pull away during the step.
            # MEASURED: this changes nothing -- the depth output is bit-identical with and
            # without it and the drift stays at 3.6 mrad median / 47 mrad max -- so the
            # residual joint motion is the solver settling contacts, not the actuators.
            # Kept because holding the pose we asked for is the correct intent regardless.
            robot.set_joint_position_target(jpos)
            # Step ENOUGH times for the ray-casters' update_period to elapse, re-asserting the
            # pose before each step so physics cannot drift it.
            #
            # One step per batch was the bug behind the bimodal delta-0 result: the casters run
            # at update_period 0.02 against a smaller physics_dt, so a single step ticks them
            # only every other batch. Since windows are written grouped by snap event and the
            # batch size equals num_envs, each batch IS one snap index -- which made the
            # good/bad split alternate exactly with window-position-within-episode (pos 0: 1%
            # good, pos 1: 95%, pos 2: 1%, pos 3: 96%). Half the frames were the PREVIOUS
            # batch's image, which no amount of placement accuracy could fix.
            _period = float(getattr(u.scene.sensors["depth_camera_near"].cfg, "update_period",
                                    0.0) or 0.0)
            # x2: the casters lag one PASS behind this loop regardless of how much sim time
            # elapses inside it, so a single pass reads the previous batch's image for about
            # half the windows. Doubling the cycle costs one extra step and removes the lag.
            _n_steps = 2 * max(2, int(round(_period / float(u.physics_dt))) + 1)
            for _ in range(_n_steps):
                robot.write_root_state_to_sim(root)
                robot.write_joint_position_to_sim(jpos)
                robot.write_joint_velocity_to_sim(torch.zeros_like(jpos))
                u.sim.step(render=True)
            u.scene.update(u.physics_dt)

            if not hasattr(main, "_pose_checked"):
                # Did the robot actually END UP where we put it? Compares the ACHIEVED root
                # pose against the recorded one, so a depth mismatch can be attributed to the
                # placement or exonerated from it. Saved, not printed: IsaacSim eats stdout.
                aq = to_torch(robot.data.root_quat_w)[ti].detach().cpu().numpy()
                ap = to_torch(robot.data.root_pos_w)[ti].detach().cpu().numpy()
                aj = to_torch(robot.data.joint_pos)[ti].detach().cpu().numpy()
                lk = to_torch(robot.data.root_link_pos_w)[ti].detach().cpu().numpy()
                cm = to_torch(robot.data.root_com_pos_w)[ti].detach().cpu().numpy()
                np.savez((args_cli.out or args_cli.data) + ".pose_check.npz",
                         achieved_quat=aq, wrote_quat=q_write, achieved_pos=ap,
                         wrote_pos=place_pos, achieved_jp=aj, wrote_jp=jp[keep],
                         link_pos=lk, com_pos=cm)
                main._pose_checked = True
            from robot_rl.tasks.manager_based.robot_rl import mdp
            # FORCE the ray-casters to re-cast. They carry update_period=0.02 and are normally
            # driven by env.step()'s sensor pass; a bare sim.step() leaves them believing they
            # are already current, so the read returns STALE data -- which looked exactly like
            # a placement bug (near-uniform error, max == the far camera's max range).
            # Read the sensors TWICE, keep the SECOND. The casters lag one READ CYCLE:
            # measured, 89% of mismatched frames were bit-equal to the SAME ENV's PREVIOUS
            # placement -- the first read returns the prior batch's image. Extra sim steps do
            # not help (the lag is in the read, not in elapsed time); a second update+read
            # does. This alone was the whole bimodal ~50% delta-0 failure.
            for _read in range(2):
                u.scene.update(u.physics_dt)
                for _sname in ("depth_camera_near", "depth_camera_far"):
                    if _sname not in u.scene.sensors:
                        raise SystemExit(f"sensor '{_sname}' absent; this list is stale and "
                                         "the depth read would be silently wrong.")
                    u.scene.sensors[_sname].update(u.physics_dt, force_recompute=True)
                # Same producer the recorder used, so this is the OBSERVATION value,
                # normalized identically -- anything else would not be like for like.
                dep = to_torch(mdp.depth_obs_stack(u))[ti].detach().cpu().numpy()

            if args_cli.check_only:
                rec = data["depth"][idx[keep]]
                diffs.append(np.abs(dep - rec))
                perwin.append(np.stack([idx[keep],
                                        np.abs(dep - rec).mean(axis=(1, 2))], axis=1))
                allren.append(dep.astype(np.float32))
                allsrc.append(idx[keep])
                if len(samples) < 6:      # keep a few raw pairs for visual inspection
                    for j in range(min(6 - len(samples), dep.shape[0])):
                        samples.append((rec[j].copy(), dep[j].copy()))
            else:
                sel_idx = idx[keep]
                out = {kk: np.asarray(v[sel_idx]).copy() for kk, v in data.items()
                       if isinstance(v, np.ndarray) and v.shape[:1] == (W,)}
                out["depth"] = dep.astype(np.float32)
                # ONLY the snap anchor moves: row H-1's position + yaw. History rows and
                # history_ref stay put so the offset cannot leak in through them.
                out["history_state"][:, H - 1, s_pos] = place_pos.astype(np.float32)
                out["history_state"][:, H - 1, s_quat] = new_quat[keep].astype(np.float32)
                writer.write_batch(out)
            n_done += len(keep)
            if n_done % (batch * 8) < batch:
                print(f"[augment] {n_done}/{len(src) * reps}")

    if args_cli.check_only:
        d = np.concatenate([x.reshape(-1) for x in diffs])
        # To a FILE, not just stdout: IsaacSim's exit path discards buffered Python stdout,
        # so a printed result routinely vanishes from a redirected log even on a clean run.
        stats = dict(mean=float(d.mean()), p50=float(np.median(d)),
                     p99=float(np.percentile(d, 99)), max=float(d.max()),
                     frac_gt_1mm=float((d > 1e-3).mean()), n_windows=int(n_done))
        cpath = (args_cli.out or os.path.splitext(args_cli.data)[0]) + ".delta0_check.npz"
        if allren:
            np.savez_compressed((args_cli.out or args_cli.data) + ".renders.npz",
                                ren=np.concatenate(allren), src=np.concatenate(allsrc))
        if perwin:
            stats["perwin"] = np.concatenate(perwin)      # (n, 2): source index, mean |err|
        if samples:
            stats["rec"] = np.stack([a for a, _ in samples])
            stats["ren"] = np.stack([b for _, b in samples])
        np.savez(cpath, **stats)
        print(f"[augment] wrote {cpath}")
        print(f"\n=== delta-0 ROUND TRIP over {n_done} windows ===")
        print(f"  |re-rendered - recorded| depth:  mean {d.mean():.5f}  p50 "
              f"{np.median(d):.5f}  p99 {np.percentile(d, 99):.5f}  max {d.max():.5f}")
        print(f"  fraction of pixels > 1e-3: {100.0 * (d > 1e-3).mean():.2f}%")
        print("  (near-zero => placement, tile alignment and sensor refresh are all correct)")
    else:
        writer.close()
        print(f"[augment] wrote {args_cli.out}: {n_done} windows "
              f"({len(src)} source x {reps})")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    sys.exit(code)
