#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Scrub the teacher and SONIC side by side in viser, on the same clip and terrain.

Stage 4 of ``compare_trackers.py``; needs no Isaac, so you can re-open it without holding the GPU.
Loads every bundle the rollout stage produced and draws, for the selected episode:

* one robot per policy, each posed from **its own** recorded joints through **its own** URDF (the
  two robots are different assets -- ours is the box-merged USD's URDF, SONIC's is its capsule
  URDF -- so each needs its own joint permutation, taken from the bundle's ``joint_names``);
* a translucent reference ghost posed from the recorded reference (``ref_joint_pos`` plus
  ``ref_root_pos``/``ref_root_quat``), which is what both policies were asked to track;
* the terrain tile, rebuilt from the clip library's manifest carried in the bundle;
* pelvis paths for each policy plus the reference, so divergence is visible in space.

The ``policy y-offset`` slider is the thing to reach for first: at 0 the robots are exactly
superimposed on the shared reference, which is how you read tracking error directly; increase it
to separate them and watch gait.

Usage::

    PYTHONPATH=source/robot_rl:source/motion_gen \\
      python scripts/rsl_rl/view_tracker_comparison.py --out-dir logs/tracker_compare
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

from robot_rl.flow_matching.viz.robot import load_urdf, recording_to_urdf_perm, xyzw_to_wxyz
from view_clip_viser import _tube_mesh
from view_debug_episodes import _terrain_mesh

# One URDF per policy: the comparison runs two different robots on purpose.
_URDF = {
    "teacher": "transfer/obelisk/g1_model/urdf/g1_29j.urdf",
    "sonic": "robot_assets/g1_sonic/g1_sonic_pelvis_link.urdf",
}
_COLOR = {"teacher": (0.20, 0.45, 1.00), "sonic": (0.95, 0.45, 0.10)}
_REF_COLOR = (0.35, 0.85, 0.45)


def _load_bundles(out_dir: str) -> dict[tuple[str, str, int], tuple[dict, dict]]:
    """Load every ``<policy>_<reference>_pick<k>.npz`` in ``out_dir``."""
    found = {}
    for path in sorted(glob.glob(os.path.join(out_dir, "*_pick*.npz"))):
        base = os.path.basename(path)[: -len(".npz")]
        try:
            policy, reference, pick = base.split("_")
            key = (policy, reference, int(pick.replace("pick", "")))
        except ValueError:
            print(f"[viser] skipping unrecognised bundle name {base}")
            continue
        d = np.load(path, allow_pickle=True)
        arrays = {k: d[k] for k in d.files if k != "meta_json"}
        found[key] = (arrays, json.loads(str(d["meta_json"])))
    return found


def main() -> int:
    """Serve the comparison in viser."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="logs/tracker_compare")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--reference", default=None, choices=(None, "clip", "fm"),
                   help="Show only one reference source (default: the first available).")
    p.add_argument("--ribbon-radius", type=float, default=0.012)
    args = p.parse_args()

    bundles = _load_bundles(args.out_dir)
    if not bundles:
        raise SystemExit(f"[viser] no bundles in {args.out_dir}; run "
                         "'compare_trackers.py --stage rollout' first.")
    refs = sorted({k[1] for k in bundles})
    reference = args.reference or refs[0]
    picks = sorted({k[2] for k in bundles if k[1] == reference})
    policies = sorted({k[0] for k in bundles if k[1] == reference})
    print(f"[viser] {args.out_dir}: references={refs} -> showing '{reference}'; "
          f"policies={policies}; picks={picks}")

    import viser
    from viser.extras import ViserUrdf

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")

    # One ViserUrdf per policy plus one for the reference ghost. Loaded once; re-posed per frame.
    urdfs, robots, perms = {}, {}, {}
    for policy in policies:
        path = _URDF[policy]
        if not os.path.isfile(path):
            print(f"[viser] WARNING: {policy} URDF {path} not found; skipping that robot")
            continue
        urdfs[policy] = load_urdf(path)
        robots[policy] = ViserUrdf(server, urdfs[policy], root_node_name=f"/{policy}")
    if not robots:
        raise SystemExit("[viser] no policy URDFs could be loaded.")
    ghost_policy = policies[0]
    ref_robot = ViserUrdf(server, urdfs[ghost_policy], root_node_name="/reference")

    gui = server.gui
    pick_sel = gui.add_slider("clip pick", min=min(picks), max=max(picks), step=1,
                              initial_value=picks[0])
    n_ep = min(int(m["num_captured"]) for (pol, r, k), (_, m) in bundles.items() if r == reference)
    ep_sel = gui.add_slider("episode", min=0, max=max(n_ep - 1, 0), step=1, initial_value=0)
    n_steps = min(b["root_pos"].shape[0] for (pol, r, k), (b, _) in bundles.items()
                  if r == reference)
    step_sel = gui.add_slider("step", min=0, max=n_steps - 1, step=1, initial_value=0)
    playing = gui.add_checkbox("play", initial_value=False)
    speed = gui.add_slider("speed x", min=0.1, max=4.0, step=0.1, initial_value=1.0)
    y_off = gui.add_slider("policy y-offset [m]", min=0.0, max=3.0, step=0.05, initial_value=0.0)
    show_ref = gui.add_checkbox("reference ghost", initial_value=True)
    show_paths = gui.add_checkbox("pelvis paths", initial_value=True)
    vis = {pol: gui.add_checkbox(f"show {pol}", initial_value=True) for pol in robots}
    readout = gui.add_markdown("")

    plots: dict[str, object] = {}
    tabs = gui.add_tab_group()

    state = {"nodes": [], "terrain": None, "key": None}

    def _bundles_for(pick: int) -> dict[str, tuple[dict, dict]]:
        """Bundles for the selected reference and pick, keyed by policy."""
        return {pol: bundles[(pol, reference, pick)] for pol in robots
                if (pol, reference, pick) in bundles}

    def rebuild(pick: int, ep: int) -> None:
        """Swap the terrain, the paths and the embedded plots for a new episode."""
        for h in state["nodes"]:
            h.remove()
        state["nodes"] = []
        if state["terrain"] is not None:
            state["terrain"].remove()
            state["terrain"] = None

        active = _bundles_for(pick)
        if not active:
            return
        b0, m0 = next(iter(active.values()))
        tm = _terrain_mesh(m0.get("terrain_manifest", []) or [], int(b0["terrain_id"][ep]))
        if tm is not None:
            v, f = tm
            v = v + b0["cell_corner"][ep][None, :]
            state["terrain"] = server.scene.add_mesh_simple(
                "/terrain", vertices=v, faces=f, color=(0.55, 0.55, 0.58))
        for pol, (b, _m) in active.items():
            end = _episode_end(b, ep)
            vv, ff = _tube_mesh(b["root_pos"][:end, ep], args.ribbon_radius * 0.8)
            state["nodes"].append(server.scene.add_mesh_simple(
                f"/paths/{pol}", vertices=vv, faces=ff, color=_COLOR[pol],
                visible=show_paths.value))
        vv, ff = _tube_mesh(b0["ref_root_pos"][:_episode_end(b0, ep), ep], args.ribbon_radius)
        state["nodes"].append(server.scene.add_mesh_simple(
            "/paths/reference", vertices=vv, faces=ff, color=_REF_COLOR,
            visible=show_paths.value))

        for name in list(plots):
            plots[name].remove()
            plots.pop(name)
        png = os.path.join(args.out_dir, f"episode{ep:02d}_pick{pick}.png")
        if os.path.isfile(png):
            try:
                import imageio.v3 as iio
                with tabs.add_tab(f"ep{ep}"):
                    plots["ep"] = gui.add_image(np.asarray(iio.imread(png))[..., :3])
            except Exception as exc:                                    # noqa: BLE001
                print(f"[viser] could not embed {png}: {exc}")
        state["key"] = (pick, ep)

    def _episode_end(b: dict, ep: int) -> int:
        """First step after the episode's first termination (or the whole capture)."""
        done = b["done"][:, ep].astype(bool)
        return int(np.argmax(done)) + 1 if done.any() else b["done"].shape[0]

    def pose(pick: int, ep: int, k: int) -> None:
        """Pose every robot (and the ghost) at step ``k``."""
        active = _bundles_for(pick)
        if not active:
            return
        lines = []
        for i, (pol, (b, m)) in enumerate(sorted(active.items())):
            end = _episode_end(b, ep)
            kk = min(k, end - 1)
            if pol not in perms:
                perms[pol] = recording_to_urdf_perm(list(m["joint_names"]), urdfs[pol])
            robots[pol].update_cfg(b["joint_pos"][kk, ep][perms[pol]])
            pos = b["root_pos"][kk, ep].copy()
            pos[1] += (i + 1) * y_off.value
            server.scene.add_frame(f"/{pol}", show_axes=False, position=tuple(pos),
                                   wxyz=tuple(xyzw_to_wxyz(b["root_quat"][kk, ep][None, :])[0]),
                                   visible=vis[pol].value)
            err = float(np.linalg.norm(b["ref_pos"][kk, ep, 0] - b["body_pos"][kk, ep, 0]))
            lines.append(f"**{pol}** step {kk}/{end - 1}  pelvis err **{err:.3f} m**  "
                         f"vx {b['ach_vel'][kk, ep, 0]:+.2f}  z {pos[2]:.2f}"
                         + ("  *(ended)*" if kk < k else ""))

        b0, m0 = next(iter(sorted(active.items())))[1]
        kk0 = min(k, _episode_end(b0, ep) - 1)
        ref_key = f"ref:{ghost_policy}"
        if ref_key not in perms:
            perms[ref_key] = recording_to_urdf_perm(list(m0["ref_joint_names"]),
                                                    urdfs[ghost_policy])
        ref_robot.update_cfg(b0["ref_joint_pos"][kk0, ep][perms[ref_key]])
        server.scene.add_frame("/reference", show_axes=False,
                               position=tuple(b0["ref_root_pos"][kk0, ep]),
                               wxyz=tuple(xyzw_to_wxyz(b0["ref_root_quat"][kk0, ep][None, :])[0]),
                               visible=show_ref.value)
        readout.content = (f"reference **{reference}**, tile **{int(b0['terrain_id'][ep])}**  \n"
                           + "  \n".join(lines))

    def refresh(_evt=None) -> None:
        """Rebuild if the episode changed, then pose."""
        pick, ep, k = int(pick_sel.value), int(ep_sel.value), int(step_sel.value)
        if state["key"] != (pick, ep):
            rebuild(pick, ep)
        pose(pick, ep, k)

    for handle in (pick_sel, ep_sel, step_sel, y_off, show_ref, show_paths, *vis.values()):
        handle.on_update(refresh)

    refresh()
    print(f"[viser] http://localhost:{args.port}")
    dt = float(next(iter(bundles.values()))[1]["dt"])
    while True:
        if playing.value:
            nxt = int(step_sel.value) + 1
            step_sel.value = 0 if nxt > n_steps - 1 else nxt
        time.sleep(dt / max(speed.value, 1e-3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
