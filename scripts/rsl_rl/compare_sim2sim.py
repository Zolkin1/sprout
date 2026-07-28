#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Run the FM generator + teacher tracker in BOTH Isaac Lab and MuJoCo and compare them.

The point is a like-for-like comparison, so the two sims must agree on three things:

1. **The same start state.** Both spawn on a clip's start frame (Isaac via the clip reset,
   MuJoCo via ``ClipReference.spawn_state``).
2. **The same velocity command.** Both read it from that clip's ``target_vel`` track, so the
   command sequence is identical by construction rather than by two matching profiles.
3. **The same terrain.** ``--stage prep`` extracts ONE clip into a single-clip library and
   emits a MuJoCo scene built from that clip's own tile overlays, so the geometry is the same
   boxes in the same places -- not a parametric staircase that merely resembles it.

The single-clip library is what makes (1) and (2) airtight: with one clip in the library
neither sim has anything else to pick, so no clip-index plumbing has to agree.

Stages (run in order; each writes files the next reads)::

    # 1. build a flat and a stair-transition case from a full library
    python scripts/rsl_rl/compare_sim2sim.py --stage prep \\
        --lib clips/g1_mb_v1_nodistract.h5 --out-dir logs/sim2sim

    # 2. Isaac Lab rollout (delegates to debug_episodes.py)
    python scripts/rsl_rl/compare_sim2sim.py --stage isaac --out-dir logs/sim2sim \\
        --case flat --checkpoint <policy.pt> --fm_checkpoint <fm.pt>

    # 3. MuJoCo rollout (delegates to transfer/sim/gen_tracker_runner.py)
    python scripts/rsl_rl/compare_sim2sim.py --stage mujoco --out-dir logs/sim2sim \\
        --case flat --run-name <rl_run_dir_name>

    # 4. overlay plots + PNGs (no sim)
    python scripts/rsl_rl/compare_sim2sim.py --stage plot --out-dir logs/sim2sim --case flat

Plots (per case): commanded vs achieved velocity in both sims, reference vs actual root
height and pelvis/feet tracking, joint angles, root position, and the height scan (mean/min/max
envelope plus the per-step difference between the two sims' scans).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
_CASES = ("flat", "stairs")


# --------------------------------------------------------------------------- prep

def _pick_clip(lib_path: str, case: str) -> tuple[int, int]:
    """Choose a representative clip for ``case``; return ``(clip_index, terrain_id)``.

    flat   -> a clip whose ``terrain_id`` is -1 (no tile geometry).
    stairs -> a clip on a ``transition`` tile (flat approach, stairs, exit) -- the case where
              the two sims' terrain must line up or nothing else is comparable.
    """
    import h5py

    with h5py.File(lib_path, "r") as f:
        tid = np.asarray(f["terrain_id"][:])
        tcode = np.asarray(f["terrain_type_code"][:])
        names = json.loads(f.attrs["terrain_type_names"])
        clip_len = np.asarray(f["clip_len"][:]) if "clip_len" in f else None
    if case == "flat":
        cand = np.nonzero(tid == -1)[0]
    else:
        want = names.index("transition") if "transition" in names else None
        cand = np.nonzero(tcode == want)[0] if want is not None else np.array([], dtype=int)
    if cand.size == 0:
        raise SystemExit(f"[prep] no clip found for case {case!r} in {lib_path}")
    if clip_len is not None:                       # prefer a long clip: more to compare
        cand = cand[np.argsort(-clip_len[cand])]
    return int(cand[0]), int(tid[cand[0]])


def _write_single_clip_library(src: str, dst: str, clip_i: int) -> dict:
    """Copy ``src`` keeping only clip ``clip_i`` (and only its tile in the manifest)."""
    import h5py

    with h5py.File(src, "r") as fi, h5py.File(dst, "w") as fo:
        tid = int(np.asarray(fi["terrain_id"][:])[clip_i])
        for k in fi.keys():
            d = fi[k]
            fo.create_dataset(k, data=np.asarray(d[clip_i:clip_i + 1]), dtype=d.dtype)
        for k, v in fi.attrs.items():
            fo.attrs[k] = v
        man = [m for m in json.loads(fi.attrs["terrain_manifest"])
               if int(m.get("terrain_id", -2)) == tid]
        fo.attrs["terrain_manifest"] = json.dumps(man)
        fo.attrs["num_clips"] = 1
        fo.attrs["num_clips_written"] = 1
        # The importer sizes its FLAT tile count from generator_config.counts.flat in
        # preference to the actual clip count (clip_terrain_loader.default_num_flat_tiles), so
        # a copied "flat: 60" would build 60 flat tiles for a library holding one stair clip --
        # and the terrain reset then aborts ("terrain id(s) [-1] have an env tile but NO
        # clips"). Rewrite it to what this library really contains.
        gc = fi.attrs.get("generator_config")
        gcfg = json.loads(gc) if isinstance(gc, str) else (gc or {})
        if not isinstance(gcfg, dict):
            gcfg = {}
        gcfg.setdefault("counts", {})
        gcfg["counts"]["flat"] = 1 if tid == -1 else 0
        fo.attrs["generator_config"] = json.dumps(gcfg)
    return {"clip_index": clip_i, "terrain_id": tid, "tiles": len(man), "path": dst}


def _mujoco_scene_from_tile(man: list[dict], tid: int, robot_xml: str, out_path: str) -> tuple[int, bool]:
    """Emit a MuJoCo scene whose geometry is EXACTLY one clip tile's overlays.

    The tile is authored in tile-local coordinates with the clip placed on it, and the Isaac
    importer meshes precisely these boxes -- so re-emitting them (rather than generating a
    parametric staircase) is what makes the two sims' terrain identical.

    The tile's ``has_floor`` flag is honoured: many stair/transition tiles are generated
    ISOLATED (no ground plane), so the robot is supported only by these boxes and stepping off
    means falling into the void. Adding MuJoCo's infinite floor to such a tile would give the
    robot support Isaac never had -- a silent sim2sim divergence.

    Returns ``(n_box_geoms, has_floor)``.
    """
    geoms = []
    has_floor = True
    for e in man:
        if int(e.get("terrain_id", -2)) != tid:
            continue
        has_floor = bool(e.get("has_floor", True))
        for i, ov in enumerate(e.get("overlays", [])):
            (px, py), (sx, sy, sz) = ov["pos_xy"], ov["size_xyz"]
            bz = float(ov.get("base_z", 0.0))
            geoms.append(
                f'        <geom name="tile{tid}_{i}" type="box" material="box_material" '
                f'group="2" size="{sx / 2:.5f} {sy / 2:.5f} {sz / 2:.5f}" '
                f'pos="{px:.5f} {py:.5f} {bz + sz / 2:.5f}"/>'
            )
    body = "\n".join(geoms)
    floor = ('        <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" '
             'group="2"/>\n') if has_floor else (
             "        <!-- tile has_floor=False: NO ground plane, matching the isolated Isaac "
             "tile (stepping off the boxes falls into the void). -->\n")
    with open(out_path, "w") as f:
        f.write(f"""<mujoco model="sim2sim_tile{tid}">
    <!-- Generated by compare_sim2sim.py: the EXACT overlay boxes of clip tile {tid}, so the
         MuJoCo geometry matches what the Isaac ClipTerrainImporter meshes for the same clip. -->
    <include file="{robot_xml}"/>
    <statistic center="1.0 0.7 1.0" extent="0.8"/>
    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0.9 0.9 0.9"/>
        <rgba haze="0.15 0.25 0.35 1"/>
        <global azimuth="-140" elevation="-20"/>
    </visual>
    <asset>
        <texture type="skybox" builtin="flat" rgb1="0 0 0" rgb2="0 0 0" width="512" height="3072"/>
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
        <material name="box_material" shininess="0.1" specular="0.5" roughness="0.3" rgba="0.75 0.6 0.42 1"/>
    </asset>
    <sensor>
        <framepos objtype="site" objname="left_ankle_site" name="left_ankle_pos"/>
        <framepos objtype="site" objname="right_ankle_site" name="right_ankle_pos"/>
    </sensor>
    <worldbody>
        <light pos="1 0 3.5" dir="0 0 -1" directional="true"/>
{floor}{body}
    </worldbody>
</mujoco>
""")
    return len(geoms), has_floor


def stage_prep(args) -> int:
    """Build a single-clip library + a matching MuJoCo scene for each case."""
    import h5py

    os.makedirs(args.out_dir, exist_ok=True)
    scene_dir = os.path.join(_REPO, "transfer", "sim", "robots", "g1")
    summary = {}
    for case in _CASES:
        clip_i, tid = _pick_clip(args.lib, case)
        lib_out = os.path.join(args.out_dir, f"s2s_{case}.h5")
        info = _write_single_clip_library(args.lib, lib_out, clip_i)
        with h5py.File(lib_out, "r") as f:
            man = json.loads(f.attrs["terrain_manifest"])
            steps = int(np.asarray(f["clip_len"][:])[0]) if "clip_len" in f else int(f.attrs["num_steps"])
            tv = np.asarray(f["target_vel"][0])
        scene_name = f"s2s_{case}_scene"
        n_geom, has_floor = _mujoco_scene_from_tile(
            man, tid, f"{args.robot}.xml",
            os.path.join(scene_dir, f"{args.robot}_{scene_name}.xml"))
        summary[case] = {**info, "scene": scene_name, "tile_geoms": n_geom,
                         "has_floor": has_floor, "clip_len": steps,
                         "cmd_vx_range": [float(tv[:steps, 0].min()), float(tv[:steps, 0].max())]}
        print(f"[prep] {case}: clip {clip_i} (tile {tid}, {steps} steps, {n_geom} tile geoms, "
              f"has_floor={has_floor}) "
              f"-> {lib_out} + robots/g1/{args.robot}_{scene_name}.xml")
        print(f"        commanded vx {summary[case]['cmd_vx_range'][0]:.2f}..."
              f"{summary[case]['cmd_vx_range'][1]:.2f} m/s")
    with open(os.path.join(args.out_dir, "cases.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return 0


# --------------------------------------------------------------------------- rollouts

def _cases(out_dir: str) -> dict:
    with open(os.path.join(out_dir, "cases.json")) as f:
        return json.load(f)


def stage_isaac(args) -> int:
    """Record the Isaac Lab rollout by delegating to debug_episodes.py."""
    case = _cases(args.out_dir)[args.case]
    # Both cases run the SAME task: the ClipTerrainImporter builds its mesh from whatever
    # single-clip library ``prep`` wrote, so the flat case gets a flat mesh and the stair
    # case a staircase. Using one task also means the Isaac and MuJoCo sides differ only in
    # geometry, never in env wiring.
    task = "G1-tracker-clf-fm-terrain-play"
    cmd = [sys.executable, os.path.join(_REPO, "scripts", "rsl_rl", "debug_episodes.py"),
           "--task", task, "--checkpoint", args.checkpoint,
           "--clip", os.path.abspath(case["path"]), "--clip_device", "cpu",
           "--num_envs", "4", "--episodes", "1", "--steps", str(case["clip_len"]),
           "--out", os.path.join(args.out_dir, f"isaac_{args.case}.npz")]
    if args.fm_checkpoint:
        cmd += ["--fm_checkpoint", args.fm_checkpoint]
    if not args.gaussian_noise:
        cmd += ["--fm_zero_noise"]
    print("[isaac] " + " ".join(cmd))
    return subprocess.call(cmd, cwd=_REPO, env={**os.environ, "PYTHONPATH": os.pathsep.join(
        [os.path.join(_REPO, "source", "robot_rl"), os.path.join(_REPO, "source", "motion_gen"),
         os.path.join(_REPO, "scripts", "rsl_rl"), os.environ.get("PYTHONPATH", "")])})


def stage_mujoco(args) -> int:
    """Record the MuJoCo rollout by delegating to gen_tracker_runner.py (clip mode)."""
    case = _cases(args.out_dir)[args.case]
    # ``--log`` is a BOOLEAN in the runner: it writes to
    # <run_dir>/mujoco_gen_tracker_logs/sim_log.csv (+ a .scan.npy sibling). Run it, then copy
    # both next to the Isaac trace so --stage plot finds a matched pair per case.
    cmd = [sys.executable, "gen_tracker_runner.py", "--mode", "eval",
           "--scene", case["scene"], "--robot_name", args.robot,
           "--load_run", args.run_name, "--clip", os.path.abspath(case["path"]),
           "--clip_id", "0", "--clip_handoff_time", str(args.clip_handoff_time),
           "--fm_update_every", str(args.fm_update_every),
           ("--gaussian_noise" if args.gaussian_noise else "--zero_noise"),
           "--headless", "--total_time", str(case["clip_len"] * 0.02), "--log"]
    print("[mujoco] " + " ".join(cmd))
    rc = subprocess.call(cmd, cwd=os.path.join(_REPO, "transfer", "sim"))
    if rc != 0:
        return rc
    src = os.path.join(_REPO, "logs", "rsl_rl", "g1", args.run_name,
                       "mujoco_gen_tracker_logs", "sim_log.csv")
    if not os.path.exists(src):
        print(f"[mujoco] FAIL: expected the runner's log at {src}")
        return 1
    dst = os.path.join(args.out_dir, f"mujoco_{args.case}.csv")
    shutil.copyfile(src, dst)
    print(f"[mujoco] copied {src} -> {dst}")
    if os.path.exists(f"{src}.scan.npy"):
        shutil.copyfile(f"{src}.scan.npy", f"{dst}.scan.npy")
        print(f"[mujoco] copied the height-scan rows -> {dst}.scan.npy")
    else:
        print("[mujoco] WARN: no .scan.npy beside the log; the scan plot will be skipped.")
    # Per-step FM predicted windows (root + extra frames) for the viser per-step ribbons.
    for ext in (".pred.npy", ".predf.npy"):
        if os.path.exists(f"{src}{ext}"):
            shutil.copyfile(f"{src}{ext}", f"{dst}{ext}")
            print(f"[mujoco] copied the FM predicted windows -> {dst}{ext}")
        else:
            print(f"[mujoco] WARN: no {ext} beside the log; viser will show no MuJoCo FM window.")
    return 0


# --------------------------------------------------------------------------- compare

def _load_isaac(path: str) -> dict:
    """Isaac trace (episode 0) from a debug_episodes bundle -> the common schema.

    Positions are converted to CLIP coordinates by subtracting the clip's world placement.
    Isaac reports WORLD coordinates and places the clip somewhere in a large multi-tile grid,
    while the generated MuJoCo scene is authored in clip coordinates -- so comparing raw
    positions puts the two robots tens of metres apart even when they are on the identical spot.
    The datum depends on where the clip lives, because a clip's ``y_des`` is authored against
    whichever origin its anchor mask pins it to:

    * **overlay (terrain) tile** -> the tile's ``cell_corner``. Not ``clip_world_pos``: that also
      carries the reset's sampled in-tile placement (stair masks leave y free), which the
      authored ``y_des`` already accounts for -- using it shifts the trace by that offset,
      ~0.39 m in y on a stair tile.
    * **flat tile** -> ``clip_world_pos``, the env origin. The importer fills ``env_cell_corner``
      only for tiles built from a manifest overlay, so a flat env's corner stays exactly zero
      while its placement sits tens of metres out in the tile grid. (The env's ``terrain_id`` does
      NOT distinguish the two -- flat tiles are numbered from 0 as well.)

    Both branches are verified to put the Isaac and MuJoCo spawns within ~3 mm.
    """
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta_json"]))
    ep = 0
    corner = np.asarray(d["cell_corner"][ep], dtype=np.float64)
    has_overlay_tile = bool(np.any(corner != 0.0))
    key = "cell_corner" if has_overlay_tile or "clip_world_pos" not in d else "clip_world_pos"
    off = np.asarray(d[key][ep], dtype=np.float64)                 # [3]
    return {
        "t": np.arange(d["cmd_vel"].shape[0]) * float(meta["dt"]),
        "cmd": d["cmd_vel"][:, ep],                       # (vx, vy, wz)
        "ach": d["ach_vel"][:, ep][:, [0, 1, 3]],         # (vx, vy, wz) from (vx, vy, vz, wz)
        "root_pos": d["root_pos"][:, ep] - off,
        "root_quat": d["root_quat"][:, ep],               # xyzw (frame-invariant: corner is a translation)
        "joint_pos": d["joint_pos"][:, ep],
        "ref_pos": d["ref_pos"][:, ep] - off,             # [T, F, 3] pelvis + feet
        "body_pos": d["body_pos"][:, ep] - off,
        "scan": d["scan_raw"][:, ep] if "scan_raw" in d else None,
        "done": (np.asarray(d["done"])[:, ep].astype(bool) if "done" in d
                 else np.zeros(d["cmd_vel"].shape[0], dtype=bool)),
        "joint_names": list(meta["joint_names"]), "frames": list(meta["frames"]),
        "cell_corner": off, "datum": key,
        # Per-step FM predicted window, shifted into the same clip frame as everything else.
        "fm_pred": (d["fm_pred"][:, ep] - off) if "fm_pred" in d else None,
        "fm_pred_frames": ((d["fm_pred_frames"][:, ep] - off) if "fm_pred_frames" in d
                           and d["fm_pred_frames"].ndim == 5 else None),
        "fm_pred_idx": (np.asarray(d["fm_pred_idx"])[:, ep] if "fm_pred_idx" in d else None),
        "fm_frame_names": list(meta.get("fm_frame_names", []) or []),
    }


def _align(iso: dict, mjc: dict) -> tuple[int, str]:
    """Number of steps the two traces may be compared over, plus a human-readable reason.

    Comparison stops at the FIRST Isaac ``done``. A terminated episode resets to the clip start
    mid-trace, so the frames after it belong to a fresh rollout -- overlaying them reports a
    metres-large "divergence" that is really a teleport back to the spawn.

    Note the trailing frame is dropped even without a ``done``: the recorder's last captured step
    is post-reset when the clip runs to its end.
    """
    n = min(len(iso["t"]), len(mjc["t"])) - 1
    why = "full overlap"
    hit = np.flatnonzero(iso["done"][:n])
    if hit.size:
        n = int(hit[0])
        why = (f"Isaac TERMINATED at step {int(hit[0])} (t={iso['t'][int(hit[0])]:.2f}s) and reset; "
               "comparing only up to there")
    return max(n, 1), why


def _load_mujoco(csv_path: str) -> dict:
    """MuJoCo trace from the runner's CSV (+ .scan.npy) -> the common schema."""
    import csv as _csv

    with open(csv_path) as f:
        rows = list(_csv.reader(f))
    head, data = rows[0], np.array([[float(x) for x in r] for r in rows[1:]])
    col = {n: i for i, n in enumerate(head)}
    # The runner emits per-joint ``jpact_<joint>`` (measured) and ``jpref_<joint>`` (the
    # reference it tracked) -- not a bare ``jp_`` prefix.
    jp_names = [n[len("jpact_"):] for n in head if n.startswith("jpact_")]
    jp_cols = [col[f"jpact_{n}"] for n in jp_names]
    jpref_cols = [col[f"jpref_{n}"] for n in jp_names]
    scan_path = f"{csv_path}.scan.npy"
    return {
        "t": data[:, col["time"]],
        "cmd": data[:, [col["cmd_vx"], col["cmd_vy"], col["cmd_wz"]]],
        "ach": data[:, [col["vx_b"], col["vy_b"], col["wz_b"]]],
        "root_pos": data[:, [col["act_pos_x"], col["act_pos_y"], col["act_pos_z"]]],
        "root_rpy": data[:, [col["act_roll"], col["act_pitch"], col["act_yaw"]]],
        "ref_root": data[:, [col["ref_pos_x"], col["ref_pos_y"], col["ref_pos_z"]]],
        "joint_pos": data[:, jp_cols] if jp_cols else None,
        "joint_ref": data[:, jpref_cols] if jpref_cols else None,
        "scan": np.load(scan_path) if os.path.exists(scan_path) else None,
        "jp_names": jp_names,
        # Per-step FM predicted window sidecars (NaN-padded before the generator is valid).
        "fm_pred": (np.load(f"{csv_path}.pred.npy")
                    if os.path.exists(f"{csv_path}.pred.npy") else None),
        "fm_pred_frames": (np.load(f"{csv_path}.predf.npy")
                           if os.path.exists(f"{csv_path}.predf.npy") else None),
    }


def stage_plot(args) -> int:
    """Overlay the two traces and write PNGs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iso = _load_isaac(os.path.join(args.out_dir, f"isaac_{args.case}.npz"))
    mjc = _load_mujoco(os.path.join(args.out_dir, f"mujoco_{args.case}.csv"))
    png_dir = os.path.join(args.out_dir, f"compare_{args.case}")
    os.makedirs(png_dir, exist_ok=True)
    n, why = _align(iso, mjc)
    print(f"[plot] {args.case}: isaac {len(iso['t'])} steps, mujoco {len(mjc['t'])} steps "
          f"-> comparing the first {n} ({why})")
    mj_fell = np.flatnonzero(np.abs(mjc["root_rpy"][:n, 0]) > 1.0)
    if mj_fell.size:
        print(f"  [WARN] MuJoCo |roll| exceeds 1.0 rad from t={mjc['t'][int(mj_fell[0])]:.2f}s "
              "-- it is on its side. The runner's survival check is height-based and does not "
              "catch this on raised terrain.")

    def save(fig, name):
        p = os.path.join(png_dir, f"{name}.png")
        fig.savefig(p, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {p}")

    # 1. velocity tracking, both sims vs the shared command
    fig, ax = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    for k, lbl in enumerate(("vx [m/s]", "vy [m/s]", "wz [rad/s]")):
        ax[k].plot(iso["t"][:n], iso["cmd"][:n, k], "k", lw=2.0, label="command (shared)")
        ax[k].plot(iso["t"][:n], iso["ach"][:n, k], lw=1.3, label="Isaac Lab")
        ax[k].plot(mjc["t"][:n], mjc["ach"][:n, k], lw=1.3, ls="--", label="MuJoCo")
        ax[k].set_ylabel(lbl)
        ax[k].grid(alpha=0.3)
    ax[0].set_title(f"{args.case}: velocity tracking, Isaac vs MuJoCo (same command, same start)")
    ax[0].legend(fontsize=8, loc="upper right")
    ax[-1].set_xlabel("time [s]")
    save(fig, "velocity")

    # 2. root position + the reference each sim tracked
    fig, ax = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    for k, lbl in enumerate(("x [m]", "y [m]", "z [m]")):
        ax[k].plot(iso["t"][:n], iso["root_pos"][:n, k], lw=1.4, label="Isaac actual")
        ax[k].plot(mjc["t"][:n], mjc["root_pos"][:n, k], lw=1.4, ls="--", label="MuJoCo actual")
        ax[k].plot(iso["t"][:n], iso["ref_pos"][:n, 0, k], lw=1.0, alpha=0.7, label="Isaac reference")
        if "ref_root" in mjc:
            ax[k].plot(mjc["t"][:n], mjc["ref_root"][:n, k], lw=1.0, alpha=0.7, ls="--",
                       label="MuJoCo reference")
        ax[k].set_ylabel(lbl)
        ax[k].grid(alpha=0.3)
    ax[0].set_title(f"{args.case}: root position + tracked reference")
    ax[0].legend(fontsize=7, loc="upper left", ncol=2)
    ax[-1].set_xlabel("time [s]")
    save(fig, "root_and_reference")

    # 3. joint angles (a representative subset, matched by name)
    if mjc["joint_pos"] is not None:
        want = [j for j in ("left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint",
                            "waist_yaw_joint", "left_shoulder_pitch_joint", "left_elbow_joint")
                if j in iso["joint_names"] and j in mjc["jp_names"]]
        fig, ax = plt.subplots(len(want), 1, figsize=(11, 2.0 * len(want)), sharex=True)
        ax = np.atleast_1d(ax)
        for a, j in zip(ax, want):
            a.plot(iso["t"][:n], iso["joint_pos"][:n, iso["joint_names"].index(j)], lw=1.3,
                   label="Isaac")
            a.plot(mjc["t"][:n], mjc["joint_pos"][:n, mjc["jp_names"].index(j)], lw=1.3, ls="--",
                   label="MuJoCo")
            if mjc.get("joint_ref") is not None:
                a.plot(mjc["t"][:n], mjc["joint_ref"][:n, mjc["jp_names"].index(j)], lw=0.9,
                       alpha=0.6, color="k", label="MuJoCo ref")
            a.set_ylabel(j.replace("_joint", "").replace("left_", "L "), fontsize=8)
            a.grid(alpha=0.3)
        ax[0].set_title(f"{args.case}: joint angles [rad]")
        ax[0].legend(fontsize=8, loc="upper right")
        ax[-1].set_xlabel("time [s]")
        save(fig, "joint_angles")

    # 4. height scan: envelope in both sims + the per-step disagreement
    if iso["scan"] is not None and mjc["scan"] is not None:
        m = min(n, mjc["scan"].shape[0])
        a, b = iso["scan"][:m], mjc["scan"][:m]
        fa, fb = np.isfinite(a), np.isfinite(b)
        fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
        for arr, fin, lbl, c in ((a, fa, "Isaac", "tab:blue"), (b, fb, "MuJoCo", "tab:red")):
            lo = np.where(fin.any(1), np.nanmin(np.where(fin, arr, np.nan), 1), np.nan)
            hi = np.where(fin.any(1), np.nanmax(np.where(fin, arr, np.nan), 1), np.nan)
            mid = np.where(fin.any(1), np.nanmean(np.where(fin, arr, np.nan), 1), np.nan)
            ax[0].fill_between(iso["t"][:m], lo, hi, alpha=0.22, color=c, label=f"{lbl} min-max")
            ax[0].plot(iso["t"][:m], mid, lw=1.3, color=c, label=f"{lbl} mean")
        ax[0].set_ylabel("ground clearance [m]")
        ax[0].set_title(f"{args.case}: height scan, Isaac vs MuJoCo "
                        f"({a.shape[1]} vs {b.shape[1]} rays)")
        ax[0].legend(fontsize=8, loc="upper right")
        ax[0].grid(alpha=0.3)
        if a.shape[1] == b.shape[1]:
            both = fa & fb
            diff = np.where(both, np.abs(a - b), np.nan)
            ax[1].plot(iso["t"][:m], np.nanmean(diff, 1), lw=1.3, label="mean |Isaac - MuJoCo|")
            ax[1].plot(iso["t"][:m], np.nanmax(diff, 1), lw=1.0, alpha=0.7, label="max")
            ax[1].set_ylabel("scan diff [m]")
            ax[1].legend(fontsize=8, loc="upper right")
        else:
            ax[1].text(0.5, 0.5, f"ray counts differ ({a.shape[1]} vs {b.shape[1]}); "
                                 "per-ray diff not comparable", ha="center", va="center",
                       transform=ax[1].transAxes)
        ax[1].set_xlabel("time [s]")
        ax[1].grid(alpha=0.3)
        save(fig, "height_scan")
    else:
        print("  [WARN] height scan missing on one side; skipping the scan plot.")
    return 0


def _rpy_to_wxyz(rpy: np.ndarray) -> np.ndarray:
    """Roll-pitch-yaw (rad) -> ``(w, x, y, z)`` quaternion, row-wise."""
    r, p, y = rpy[:, 0] / 2, rpy[:, 1] / 2, rpy[:, 2] / 2
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.stack([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy], axis=-1)


def stage_viser(args) -> int:
    """Replay BOTH rollouts in viser, side by side on the shared terrain.

    Two robots posed from their own traces on the same clock, the tile they both ran on, root
    paths and tracked references as ribbons, plus the comparison PNGs in the GUI. The MuJoCo
    ghost can be offset laterally so the two are separable, or overlaid at 0 to read the
    divergence directly.
    """
    import time

    import viser
    from viser.extras import ViserUrdf

    sys.path.insert(0, os.path.join(_REPO, "scripts", "rsl_rl"))
    from robot_rl.flow_matching.viz.robot import load_urdf, recording_to_urdf_perm, xyzw_to_wxyz
    from view_clip_viser import _tube_mesh
    from view_debug_episodes import _terrain_mesh

    case = _cases(args.out_dir)[args.case]
    iso = _load_isaac(os.path.join(args.out_dir, f"isaac_{args.case}.npz"))
    mjc = _load_mujoco(os.path.join(args.out_dir, f"mujoco_{args.case}.csv"))
    n, why = _align(iso, mjc)
    import h5py
    with h5py.File(case["path"], "r") as f:
        man = json.loads(f.attrs["terrain_manifest"])
    print(f"[viser] {args.case}: {n} shared steps ({why}); terrain tile {case['terrain_id']}")

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    urdf = load_urdf(args.urdf)
    iso_perm = recording_to_urdf_perm(iso["joint_names"], urdf)
    mjc_perm = recording_to_urdf_perm(mjc["jp_names"], urdf) if mjc["joint_pos"] is not None else None
    r_iso = ViserUrdf(server, urdf, root_node_name="/isaac")
    r_mjc = ViserUrdf(server, urdf, root_node_name="/mujoco")

    tm = _terrain_mesh(man, int(case["terrain_id"]))
    if tm is not None:
        v, f = tm
        server.scene.add_mesh_simple("/terrain", vertices=v, faces=f, color=(0.55, 0.55, 0.58))
    # Whole-episode ACTUAL paths (context; toggleable). The reference ribbons are NOT drawn
    # whole-episode -- they are per-step FM windows, rebuilt in ``pose`` below.
    path_handles = []
    for name, pts, col in (("/path/isaac", iso["root_pos"][:n], (0.20, 0.45, 1.00)),
                           ("/path/mujoco", mjc["root_pos"][:n], (0.95, 0.55, 0.10))):
        vv, ff = _tube_mesh(np.asarray(pts, dtype=np.float64), 0.012)
        path_handles.append(server.scene.add_mesh_simple(name, vertices=vv, faces=ff, color=col))

    # Which predicted extra frames are the feet, by name (the FM also predicts the wrists).
    fm_names = list(iso["fm_frame_names"])
    foot_idx = [i for i, nm in enumerate(fm_names) if "ankle" in nm or "foot" in nm]
    wrist_idx = [i for i, nm in enumerate(fm_names) if i not in foot_idx]
    have_win = iso["fm_pred"] is not None and mjc["fm_pred"] is not None
    print(f"[viser] FM predicted frames: {fm_names or '(none recorded)'}; feet -> {foot_idx}")
    if not have_win:
        print("[viser] WARN: no per-step FM window recorded on one/both sides -- re-run "
              "--stage isaac and --stage mujoco to capture fm_pred/.pred.npy, else only the "
              "actual paths are shown.")

    gui = server.gui
    step = gui.add_slider("step", min=0, max=n - 1, step=1, initial_value=0)
    playing = gui.add_checkbox("play", initial_value=False)
    speed = gui.add_slider("speed x", min=0.1, max=4.0, step=0.1, initial_value=1.0)
    y_off = gui.add_slider("MuJoCo y-offset [m]", min=0.0, max=3.0, step=0.1, initial_value=1.5)
    show_paths = gui.add_checkbox("whole-episode actual paths", initial_value=True)
    show_wrists = gui.add_checkbox("FM window: also wrist frames", initial_value=False)
    win_r = gui.add_slider("FM ribbon radius [m]", min=0.004, max=0.03, step=0.002,
                           initial_value=0.010)
    readout = gui.add_markdown("")
    png_dir = os.path.join(args.out_dir, f"compare_{args.case}")
    with gui.add_folder("comparison plots"):
        tabs = gui.add_tab_group()
        try:
            import imageio.v3 as iio
            for nm in ("velocity", "root_and_reference", "joint_angles", "height_scan"):
                fp = os.path.join(png_dir, f"{nm}.png")
                if os.path.exists(fp):
                    with tabs.add_tab(nm):
                        gui.add_image(np.asarray(iio.imread(fp))[..., :3])
        except Exception as exc:                                          # noqa: BLE001
            gui.add_markdown(f"plots not embedded ({exc}); see `{png_dir}`")

    mjc_q = _rpy_to_wxyz(mjc["root_rpy"][:n])
    win_handles: dict[str, object] = {}

    def _draw_window(tag: str, pts: np.ndarray, col: tuple[float, float, float],
                     dy: float) -> None:
        """Draw one FM-window polyline as a tube, or hide it if the window is not valid.

        ``pts`` is a ``[F, 3]`` slice of a single step's prediction. Rows are NaN before the
        generator has produced anything (MuJoCo pads that way), and a degenerate window would
        make the tube builder fail, so both cases just hide the handle instead.
        """
        ok = pts is not None and np.isfinite(pts).all() and len(pts) >= 2
        if ok:
            q = np.asarray(pts, dtype=np.float64).copy()
            q[:, 1] += dy
            ok = float(np.linalg.norm(q[-1] - q[0])) > 1e-6
        if not ok:
            h = win_handles.get(tag)
            if h is not None:
                h.visible = False
            return
        vv, ff = _tube_mesh(q, float(win_r.value))
        win_handles[tag] = server.scene.add_mesh_simple(f"/fmwin/{tag}", vertices=vv, faces=ff,
                                                       color=col)
        win_handles[tag].visible = True

    def pose(k: int) -> None:
        """Pose both robots at shared step ``k`` and redraw that step's FM windows."""
        r_iso.update_cfg(iso["joint_pos"][k][iso_perm])
        server.scene.add_frame("/isaac", show_axes=False,
                               position=tuple(iso["root_pos"][k]),
                               wxyz=tuple(xyzw_to_wxyz(iso["root_quat"][k][None, :])[0]))
        if mjc_perm is not None:
            r_mjc.update_cfg(mjc["joint_pos"][k][mjc_perm])
        p = np.array(mjc["root_pos"][k], dtype=float)
        p[1] += y_off.value
        server.scene.add_frame("/mujoco", show_axes=False, position=tuple(p),
                               wxyz=tuple(mjc_q[k]))
        for h in path_handles:
            h.visible = bool(show_paths.value)

        # The FM's OWN OUTPUT at this step: the F-frame predicted window for the pelvis and
        # each foot, per sim. Refreshed only every fm_update_every steps, so scrubbing shows a
        # window hold for ~12 steps then jump as the generator re-plans off the robot's pose.
        idx = list(foot_idx) + (list(wrist_idx) if show_wrists.value else [])
        for sim, src, dy, c_root, c_fr in (
                ("isaac", iso, 0.0, (0.20, 0.45, 1.00), (0.30, 0.85, 1.00)),
                ("mujoco", mjc, y_off.value, (0.95, 0.55, 0.10), (1.00, 0.85, 0.20))):
            root_w = src["fm_pred"][k] if src["fm_pred"] is not None else None
            _draw_window(f"{sim}/pelvis", root_w, c_root, dy)
            frames_w = src["fm_pred_frames"]
            for j in idx:
                seg = (frames_w[k, :, j] if frames_w is not None and j < frames_w.shape[2]
                       else None)
                _draw_window(f"{sim}/{fm_names[j] if j < len(fm_names) else j}", seg, c_fr, dy)
            for j in (wrist_idx if not show_wrists.value else []):
                h = win_handles.get(f"{sim}/{fm_names[j] if j < len(fm_names) else j}")
                if h is not None:
                    h.visible = False

        dv = np.abs(iso["ach"][k] - mjc["ach"][k])
        widx = (int(iso["fm_pred_idx"][k]) if iso["fm_pred_idx"] is not None else -1)
        readout.content = (
            f"**step {k}/{n - 1}**  t={iso['t'][k]:.2f}s  \n"
            f"cmd (vx,vy,wz) = {np.round(iso['cmd'][k][:3], 2).tolist()}  \n"
            f"Isaac  achieved = {np.round(iso['ach'][k], 2).tolist()}  \n"
            f"MuJoCo achieved = {np.round(mjc['ach'][k], 2).tolist()}  \n"
            f"|diff| = {np.round(dv, 2).tolist()}  \n"
            f"root z: Isaac {iso['root_pos'][k, 2]:.3f} / MuJoCo {mjc['root_pos'][k, 2]:.3f} m  \n"
            f"Isaac FM window: applying frame {widx}/{(iso['fm_pred'].shape[1] - 1) if iso['fm_pred'] is not None else '?'}"
            f" (re-plans every {args.fm_update_every})")

    for _ctl in (step, y_off, show_paths, show_wrists, win_r):
        _ctl.on_update(lambda _evt: pose(int(step.value)))

    pose(0)
    print(f"[viser] serving on http://localhost:{args.port}  (blue = Isaac, orange = MuJoCo)")
    dt = float(iso["t"][1] - iso["t"][0]) if n > 1 else 0.02
    while True:
        if playing.value:
            nxt = int(step.value) + 1
            step.value = 0 if nxt > n - 1 else nxt
        time.sleep(dt / max(speed.value, 1e-3))
    return 0


def main() -> int:
    """Dispatch a stage."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True,
                   choices=("prep", "isaac", "mujoco", "plot", "viser"))
    p.add_argument("--out-dir", default="logs/sim2sim", help="Where cases + traces + PNGs live.")
    p.add_argument("--case", default="flat", choices=_CASES)
    p.add_argument("--lib", default="clips/g1_mb_v1_nodistract.h5", help="[prep] source library.")
    p.add_argument("--robot", default="g1_29j", help="MuJoCo robot name (scene prefix).")
    p.add_argument("--checkpoint", default=None, help="[isaac] RL policy .pt (absolute).")
    p.add_argument("--fm_checkpoint", default=None, help="[isaac] FM .pt (absolute).")
    p.add_argument("--run-name", default=None, help="[mujoco] RL run dir NAME holding exported/.")
    p.add_argument("--fm_update_every", type=int, default=12, help="[mujoco] FM re-plan cadence.")
    p.add_argument("--gaussian_noise", action="store_true",
                   help="Sample the FM stochastically on BOTH sides (training-matched). The "
                        "default instead runs both sides zero-noise, i.e. deterministic given "
                        "the conditioning, so the traces differ only by simulator dynamics "
                        "rather than by an RNG draw per re-plan. Whichever is chosen, this flag "
                        "sets Isaac and MuJoCo together -- their own defaults disagree.")
    p.add_argument("--port", type=int, default=8091, help="[viser] server port.")
    p.add_argument("--urdf", default="transfer/obelisk/g1_model/urdf/g1_29j.urdf",
                   help="[viser] URDF to pose (29-DOF).")
    p.add_argument("--clip_handoff_time", type=float, default=0.0,
                   help="[mujoco] sim time to hand the reference from the clip to the FM. "
                        "Defaults to 0 = the FM drives from step 0, matching the Isaac side: "
                        "both sample the generator with its learned null-history embedding while "
                        "the N-step window fills (--null_history_warmup, on by default). Before "
                        "that existed, 0 asked the FM to predict from an empty buffer and MuJoCo "
                        "reported 'Nan, Inf or huge value in CTRL ... Time = 0.0000', so this had "
                        "to be >= N steps; raise it again only to deliberately give MuJoCo a clip "
                        "lead-in, which Isaac will NOT have.")
    args = p.parse_args()
    if args.stage == "isaac" and not args.checkpoint:
        raise SystemExit("--stage isaac needs --checkpoint")
    if args.stage == "mujoco" and not args.run_name:
        raise SystemExit("--stage mujoco needs --run-name")
    return {"prep": stage_prep, "isaac": stage_isaac, "mujoco": stage_mujoco,
            "plot": stage_plot, "viser": stage_viser}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
