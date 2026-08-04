#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Scrub MANY Isaac-vs-MuJoCo clip rollouts side by side, one clip at a time.

``compare_sim2sim.py --stage viser`` replays ONE clip, which is the wrong shape for the
question a sweep raises: ``clip_ab_sweep.py`` reports that N clips diverge, and the next
thing to do is watch them and look for the pattern. Thirty separate viser sessions is not
watching them.

So this loads a whole set and puts a clip PICKER in the GUI. Both robots, the depth images
each policy actually saw, and MuJoCo's collision geometry, all switchable without a restart.

**Where the data comes from.** Isaac Lab's side is ONE ``debug_episodes.py`` bundle whose
envs were pinned to the clips (``mdp.events.resets.resolve_pinned_clips``), so env ``i`` is
clip ``ids[i]`` and all N traces come from a single GPU run rather than N launches. MuJoCo's
side is the per-clip ``sim_log.csv`` a ``clip_ab_sweep.py`` run already wrote. Nothing here
runs a simulator.

Usage::

    PYTHONPATH=source/robot_rl:source/motion_gen:. python scripts/rsl_rl/view_clip_ab_viser.py \\
        --run-dir logs/clip_ab/run100_19999 \\
        --bundle logs/clip_ab/run100_19999/isaac_diverging.npz \\
        --clips  logs/clip_ab/run100_19999/diverging_clips.json

``--clips`` must be the SAME list, in the same order, that the bundle's envs were pinned to;
it is what maps bundle episode ``i`` to a MuJoCo directory. Getting it wrong pairs two
different clips and the replay looks like a huge sim2sim gap, so the count is checked and the
mapping is printed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import sys
import time

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts" / "rsl_rl"))


def _compare_module():
    """Import ``compare_sim2sim`` for its trace loaders and viser helpers.

    By path rather than as a package: ``scripts/`` is not one, and duplicating the loaders
    is how the two viewers would drift apart on the schema.
    """
    path = _REPO / "scripts" / "rsl_rl" / "compare_sim2sim.py"
    spec = importlib.util.spec_from_file_location("compare_sim2sim", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["compare_sim2sim"] = mod
    spec.loader.exec_module(mod)
    return mod


def _rezero(iso: dict, mjc: dict) -> None:
    """Slide both traces so each STARTS at the origin, in place.

    The two simulators do not share a world frame: Isaac Lab spawns the clip on a tile
    somewhere in a large multi-tile grid (measured 12-15 m out), while the generated MuJoCo
    scene is authored in clip coordinates starting at (0, 0). Drawn raw, the robots sit far
    apart and the y-offset slider then moves MuJoCo relative to ITS own origin rather than
    relative to Isaac -- so there is no slider value that overlays them.

    Subtracting each trace's own step-0 position makes the comparison what it should be:
    both start together and any separation afterwards is behaviour. Only x/y are shifted --
    z is the height above the floor and both already agree on it (0.78 m), so zeroing it
    would sink the robots.
    """
    for tr in (iso, mjc):
        off = np.asarray(tr["root_pos"][0], dtype=float).copy()
        off[2] = 0.0
        tr["root_pos"] = np.asarray(tr["root_pos"], dtype=float) - off
        # Anything else drawn in the same frame has to move with it.
        for key in ("ref_pos", "body_pos"):
            v = tr.get(key)
            if v is not None:
                tr[key] = np.asarray(v, dtype=float) - off


def load_pairs(args, cs) -> list[dict]:
    """Load every (isaac, mujoco) trace pair, in clip order.

    Args:
        args: Parsed CLI namespace.
        cs: The ``compare_sim2sim`` module (for its loaders).

    Returns:
        One dict per clip with both traces and the shared step count. Clips whose MuJoCo run
        is missing are skipped with a note rather than silently dropped.
    """
    ids = json.loads(pathlib.Path(args.clips).read_text())
    bundle = np.load(args.bundle, allow_pickle=True)
    meta = json.loads(str(bundle["meta_json"]))
    n_ep = int(meta.get("num_captured", len(ids)))
    if n_ep < len(ids):
        raise SystemExit(
            f"bundle has {n_ep} episodes but --clips lists {len(ids)}. They must correspond "
            "one-to-one: episode i is the env pinned to clips[i]."
        )
    out = []
    for i, cid in enumerate(ids):
        csv_path = pathlib.Path(args.run_dir) / "mj" / f"clip_{cid:05d}" / "sim_log.csv"
        if not csv_path.is_file():
            print(f"[skip] clip {cid}: no {csv_path}")
            continue
        iso = cs._load_isaac(args.bundle, ep=i)
        mjc = cs._load_mujoco(str(csv_path))
        _rezero(iso, mjc)
        n = min(len(iso["t"]), len(mjc["t"]))
        out.append({"clip": int(cid), "iso": iso, "mjc": mjc, "n": n,
                    "mjc_q": cs._rpy_to_wxyz(mjc["root_rpy"])})
        print(f"[load] clip {cid:5d}: isaac {len(iso['t'])} steps, mujoco {len(mjc['t'])} "
              f"-> {n} shared")
    if not out:
        raise SystemExit("no clip pairs loaded")
    return out


def main() -> int:
    """Serve the multi-clip comparison."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True,
                   help="clip_ab_sweep run dir (supplies mj/clip_XXXXX/sim_log.csv).")
    p.add_argument("--bundle", required=True,
                   help="debug_episodes .npz whose envs were PINNED to --clips.")
    p.add_argument("--clips", required=True,
                   help="JSON list of clip ids, in the bundle's env order.")
    p.add_argument("--run-name", default="clip_ab_run100_19999",
                   help="Run dir holding exported/policy_parameters.yaml (the depth-camera "
                        "contract, which is what gives the images their pixel grid).")
    p.add_argument("--robot", default="g1_29j", help="MuJoCo model, for the collision geoms.")
    p.add_argument("--scene", default="basic_scene", help="MuJoCo scene the runs used.")
    p.add_argument("--urdf", default="transfer/obelisk/g1_model/urdf/g1_29j.urdf")
    p.add_argument("--port", type=int, default=8091)
    args = p.parse_args()

    cs = _compare_module()
    import viser
    from viser.extras import ViserUrdf

    from robot_rl.flow_matching.viz.robot import load_urdf, recording_to_urdf_perm, xyzw_to_wxyz

    pairs = load_pairs(args, cs)
    by_clip = {d["clip"]: d for d in pairs}
    labels = [f"clip {d['clip']}  ({d['n']} steps)" for d in pairs]

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    urdf = load_urdf(str(_REPO / args.urdf) if not os.path.isabs(args.urdf) else args.urdf)
    r_iso = ViserUrdf(server, urdf, root_node_name="/isaac")
    r_mjc = ViserUrdf(server, urdf, root_node_name="/mujoco")
    iso_perm = recording_to_urdf_perm(pairs[0]["iso"]["joint_names"], urdf)
    mjc_perm = (recording_to_urdf_perm(pairs[0]["mjc"]["jp_names"], urdf)
                if pairs[0]["mjc"]["joint_pos"] is not None else None)
    server.scene.add_grid("/grid", width=20.0, height=20.0)

    gui = server.gui
    pick = gui.add_dropdown("clip", options=labels, initial_value=labels[0])
    step = gui.add_slider("step", min=0, max=pairs[0]["n"] - 1, step=1, initial_value=0)
    playing = gui.add_checkbox("play", initial_value=False)
    speed = gui.add_slider("speed x", min=0.1, max=4.0, step=0.1, initial_value=1.0)
    # 0 overlays the two robots exactly (both traces are re-zeroed to the origin); raise it
    # to separate them for side-by-side gait comparison.
    y_off = gui.add_slider("MuJoCo y-offset [m]", min=0.0, max=3.0, step=0.1, initial_value=0.0)
    # In a COLLAPSIBLE folder, and behind a toggle: a markdown block whose text changes every
    # frame re-lays-out the whole panel, and with proportional digits the line widths move too,
    # so the window visibly jitters while scrubbing. Fixed-width formatting inside a fenced
    # code block (monospace, constant line length, constant line COUNT) removes the reflow;
    # the folder and the switch are for when it is simply not wanted.
    with gui.add_folder("readout", expand_by_default=True):
        show_readout = gui.add_checkbox("live readout", initial_value=True)
        readout = gui.add_markdown("")

    # Depth panels, one per (sim, camera). Sized from the artifact contract, zoomed because
    # 30x26 is unreadable at 1:1 (see compare_sim2sim._depth_to_rgb).
    layout = cs._depth_camera_layout(argparse.Namespace(run_name=args.run_name))
    depth_handles: list = []
    show_depth = depth_zoom = None
    if layout:
        with gui.add_folder("depth (as the policy sees it)"):
            show_depth = gui.add_checkbox("update depth panels", initial_value=True)
            depth_zoom = gui.add_slider("depth zoom", min=2, max=16, step=1, initial_value=8)
            for ci, (cam, hgt, wid) in enumerate(layout):
                for sim in ("isaac", "mujoco"):
                    h = gui.add_image(np.zeros((hgt * 8, wid * 8, 3), np.uint8),
                                      label=f"{sim} {cam}")
                    depth_handles.append((h, sim, ci, hgt, wid))

    coll = cs._mujoco_collision_geoms(
        argparse.Namespace(run_name=args.run_name, robot=args.robot), {"scene": args.scene})
    show_coll = (gui.add_checkbox(f"MuJoCo collision geoms ({len(coll['names'])})",
                                  initial_value=False) if coll else None)
    coll_handles: dict[str, object] = {}

    state = {"cur": pairs[0]}

    def pose(k: int) -> None:
        """Pose both robots at step ``k`` of the selected clip."""
        d = state["cur"]
        iso, mjc, n = d["iso"], d["mjc"], d["n"]
        k = max(0, min(k, n - 1))
        r_iso.update_cfg(iso["joint_pos"][k][iso_perm])
        server.scene.add_frame("/isaac", show_axes=False,
                               position=tuple(iso["root_pos"][k]),
                               wxyz=tuple(xyzw_to_wxyz(iso["root_quat"][k][None, :])[0]))
        if mjc_perm is not None and mjc["joint_pos"] is not None:
            r_mjc.update_cfg(mjc["joint_pos"][k][mjc_perm])
        pmj = np.array(mjc["root_pos"][k], dtype=float)
        pmj[1] += y_off.value
        server.scene.add_frame("/mujoco", show_axes=False, position=tuple(pmj),
                               wxyz=tuple(d["mjc_q"][k]))

        if show_depth is not None and show_depth.value:
            for h, sim, ci, hgt, wid in depth_handles:
                src = (iso if sim == "isaac" else mjc)["depth"]
                if src is None or k >= len(src) or ci >= src.shape[1]:
                    continue
                h.image = cs._depth_to_rgb(src[k, ci], hgt, wid, int(depth_zoom.value))

        if show_coll is not None:
            cs._draw_collision(coll, coll_handles, server, mjc, d["mjc_q"], k,
                               y_off.value, bool(show_coll.value))

        # MuJoCo's own fall test is height-only at 0.2 m; Isaac terminates at 0.5 m or 70
        # degrees of tilt. Show the ISAAC predicate, since that is what "this clip failed"
        # meant in the sweep -- otherwise the robot reads as upright while lying down.
        z, roll, pitch = mjc["root_pos"][k, 2], mjc["root_rpy"][k, 0], mjc["root_rpy"][k, 1]
        down = z < 0.5 or abs(roll) > np.radians(70) or abs(pitch) > np.radians(70)
        if show_readout.value:
            row = lambda lbl, v: (f"{lbl:<8s}" + "".join(f"{x:+8.2f}" for x in v))
            # Every field padded to a fixed width and the DOWN line always present, so
            # neither the line lengths nor the line count change between frames.
            readout.content = (
                "```\n"
                f"clip {d['clip']:<6d} step {k:>4d}/{n - 1:<4d} t {iso['t'][k]:6.2f}s\n"
                f"{'':8s}{'vx':>8s}{'vy':>8s}{'wz':>8s}\n"
                f"{row('cmd', iso['cmd'][k][:3])}\n"
                f"{row('isaac', iso['ach'][k][:3])}\n"
                f"{row('mujoco', mjc['ach'][k][:3])}\n"
                f"root z  {iso['root_pos'][k, 2]:8.3f}{z:8.3f}  m\n"
                f"tilt    {np.degrees(roll):+8.0f}{np.degrees(pitch):+8.0f}  deg\n"
                f"{'MUJOCO DOWN (Isaac rule)' if down else 'standing':<24s}\n"
                "```")

    def select(_evt=None) -> None:
        """Switch clips: re-range the slider, rewind, redraw."""
        d = by_clip[int(pick.value.split()[1])]
        state["cur"] = d
        step.max = d["n"] - 1
        step.value = 0
        pose(0)

    pick.on_update(select)
    for ctl in (step, y_off, show_coll, show_depth, depth_zoom, show_readout):
        if ctl is not None:
            ctl.on_update(lambda _e: pose(int(step.value)))

    pose(0)
    print(f"[viser] {len(pairs)} clips on http://localhost:{args.port}  "
          "(blue = Isaac, orange = MuJoCo)")
    dt = float(pairs[0]["iso"]["t"][1] - pairs[0]["iso"]["t"][0]) if pairs[0]["n"] > 1 else 0.02
    while True:
        if playing.value:
            nxt = int(step.value) + 1
            step.value = 0 if nxt > state["cur"]["n"] - 1 else nxt
        time.sleep(dt / max(speed.value, 1e-3))


if __name__ == "__main__":
    sys.exit(main())
