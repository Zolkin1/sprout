# Copyright 2026 Zachary Olkin. All rights reserved.

"""The viewer half of :mod:`fm_sample_explorer`: step re-plans, see what was proposed.

Imports no MuJoCo and no onnxruntime -- everything it draws was recorded headless, for the
reason stated in the recorder's docstring (attaching the viewer perturbs ``mjData`` enough to
change the trajectory, so a live stepper would show a rollout whose candidates were never the
ones scored).

Four views of one re-plan, and each answers a different question:

* **3-D scene** -- the terrain, the reference path, and every candidate's actual path, red
  where it eventually fell and teal where it survived.  Says WHERE the candidates diverge.
* **PCA of the emitted windows** -- the ``(F, D)`` chunks projected to 2-D.  Says whether the
  survivors and the fallers are separable in what the generator PROPOSED, which is precisely
  the question every chunk-level method needs answered yes.
* **Root height** and **feet height**, per candidate, over the chunk.  Says what the proposal
  physically is -- a step, a stumble, a stall -- where the PCA only says the proposals differ.

The PCA is fit on the candidates of the CURRENTLY SELECTED re-plan only.  Fitting it once
across every re-plan would let the leading component be "how far along the episode is", which
is the largest variance in the set and the least interesting: the question is whether
candidates differ AT ONE STATE.
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET

import numpy as np


def _window_blocks(d: int, nj: int = 29) -> dict[str, slice]:
    """Where each block sits in a window row.

    The shipped generator packs ``base_pose(9) + base_vel(6) + joint_pos(nj)``. Derived from
    the recorded width rather than hardcoded, so a differently-packed checkpoint is refused
    loudly instead of being sliced into nonsense.
    """
    if d != 9 + 6 + nj:
        raise SystemExit(
            f"window width {d} is not base_pose(9)+base_vel(6)+joint_pos({nj}) = {9+6+nj}. "
            "This viewer's block offsets would be wrong; state the layout explicitly.")
    return {"base_pose": slice(0, 9), "base_vel": slice(9, 15), "joint_pos": slice(15, 15 + nj)}


def _pca2(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project rows of ``x`` to 2-D by PCA, returning the scores and the explained fraction.

    Plain SVD on the centred matrix -- with a handful of candidates there is nothing for a
    fancier decomposition to buy, and this keeps the viewer dependency-free.
    """
    xc = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(xc, full_matrices=False)
    var = s ** 2
    frac = var[:2].sum() / var.sum() if var.sum() > 0 else 0.0
    return (u[:, :2] * s[:2]), np.asarray(frac)


def _panel_png(windows_sel, failed_sel, ks, blocks, dt: float = 0.02) -> np.ndarray:
    """Three real plots for one re-plan: PCA scatter, root height, and mean foot height.

    Rendered to an RGB array rather than served as an interactive figure, because the point
    is a glanceable panel beside the 3-D scene and matplotlib already carries this repo's
    paper style.

    The heights come from the window's own blocks: ``base_pose`` node 2 is the root z, and the
    foot proxy is the mean of the two ankle joints' angles -- NOT forward kinematics. A true
    foot height needs FK through the URDF; this is a shape cue and is labelled as such so it
    is not read as a metric height.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    red, teal = "#CC3311", "#009988"
    cols = [red if f else teal for f in failed_sel]

    flat = windows_sel.reshape(len(windows_sel), -1)
    scores, frac = _pca2(flat)
    for n, (x, y) in enumerate(scores):
        axes[0].scatter([x], [y], c=cols[n], s=160, edgecolors="black", zorder=3)
        axes[0].annotate(str(int(ks[n])), (x, y), fontsize=11,
                         xytext=(6, 4), textcoords="offset points")
    axes[0].set_xlabel(r"PC1")
    axes[0].set_ylabel(r"PC2")
    axes[0].set_title(rf"emitted windows, PCA ({float(frac):.0%} var)", fontsize=15)

    t = np.arange(windows_sel.shape[1]) * dt
    bp = blocks["base_pose"]
    for n in range(len(windows_sel)):
        axes[1].plot(t, windows_sel[n, :, bp.start + 2], color=cols[n], alpha=0.85, lw=2)
    axes[1].set_xlabel(r"time into chunk (s)")
    axes[1].set_ylabel(r"root $z$ (window frame, m)")
    axes[1].set_title(r"root height", fontsize=15)

    jp = blocks["joint_pos"]
    # Ankle pitch sits at policy-order indices 13/14 within the joint block.
    ankles = [jp.start + 13, jp.start + 14]
    for n in range(len(windows_sel)):
        axes[2].plot(t, windows_sel[n][:, ankles].mean(axis=-1), color=cols[n], alpha=0.85, lw=2)
    axes[2].set_xlabel(r"time into chunk (s)")
    axes[2].set_ylabel(r"mean ankle pitch (rad)")
    axes[2].set_title(r"ankle pitch (foot PROXY, not FK)", fontsize=15)

    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def stage_view(args) -> int:
    """Serve the viser app."""
    import viser

    path = os.path.join(args.out, "samples.npz")
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    trajs, off = [], 0
    for n in d["traj_len"]:
        trajs.append(d["traj_flat"][off:off + n])
        off += n
    replan, failed, windows = d["replan"], d["failed"], d["windows"]
    picks = sorted(set(int(r) for r in replan))
    K = int(meta["k"])
    print(f"[view] {path}: {len(picks)} re-plan(s), K={K}, reference "
          f"{meta['ref_verdict']}/{meta['ref_failure']} over {meta['n_chunks']} chunks")

    server = viser.ViserServer(port=int(args.port))
    server.scene.world_axes.visible = True

    # ---- terrain, drawn once ---------------------------------------------------------------
    n_drawn = 0
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mjeval"))
        import terrains
        course = terrains.isaac_tile_course(meta["clip_lib"], int(meta["tile"]),
                                            speeds=(float(meta["speed"]),))
        # `Course.geoms` are raw MJCF `<geom .../>` STRINGS, not structured boxes -- the
        # course's job is to be written into a scene file, not to be introspected. Parsing
        # them here beats re-deriving the geometry from the manifest, which would be a second
        # producer of the same thing and could drift from what the rollout actually ran on.
        for i, g in enumerate(course.geoms):
            el = ET.fromstring(g.strip())
            if el.get("type") != "box":
                continue
            size = [float(v) for v in el.get("size", "").split()]
            pos = [float(v) for v in el.get("pos", "0 0 0").split()]
            if len(size) != 3 or len(pos) != 3:
                continue
            # MJCF box `size` is the HALF-extent; viser takes full dimensions.
            server.scene.add_box(f"/terrain/{i}", dimensions=tuple(2.0 * v for v in size),
                                 position=tuple(pos), color=(150, 150, 150))
            n_drawn += 1
        print(f"[view] drew {n_drawn} of {len(course.geoms)} terrain geoms")
    except Exception as exc:                                              # noqa: BLE001
        # The scene is a convenience; the trajectories carry the result. Say so rather than
        # dying, so a viewer still opens on a box without the clip library.
        print(f"[view] terrain not drawn ({type(exc).__name__}: {exc}); paths only.")
    if n_drawn == 0:
        print("[view] NO terrain geometry drawn -- the paths are floating in an empty scene. "
              "Read heights against the reference path, not against the grid.")

    # ---- the robot, standing where the candidates diverge --------------------------------
    urdf_vis, urdf_joint_names = None, []
    if "qpos" in d.files:
        try:
            import yourdfpy
            from viser.extras import ViserUrdf
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            up = os.path.join(root, "robot_assets", "g1_perceptive", "g1_perceptive.urdf")
            urdf = yourdfpy.URDF.load(up, build_scene_graph=True, load_meshes=True)
            urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")
            urdf_joint_names = [j.name for j in urdf.robot.joints if j.type != "fixed"]
            print(f"[view] robot loaded ({len(urdf_joint_names)} actuated joints)")
        except Exception as exc:                                          # noqa: BLE001
            print(f"[view] robot NOT drawn ({type(exc).__name__}: {exc}); paths only.")
    else:
        print("[view] this recording predates qpos capture -- no robot. Re-record to get it.")

    gui = server.gui
    idx = gui.add_slider("re-plan", min=0, max=len(picks) - 1, step=1, initial_value=0)
    lbl = gui.add_markdown("")
    prev_btn = gui.add_button("< prev")
    next_btn = gui.add_button("next >")
    show_paths = gui.add_checkbox("candidate paths", initial_value=True)
    plot = gui.add_image(np.zeros((10, 10, 3), np.uint8), label="per-candidate plots")

    server.scene.add_spline_catmull_rom(
        "/reference", positions=d["ref_traj"].astype(np.float32),
        color=(60, 60, 60), line_width=3.0)

    def render(i: int) -> None:
        """Draw one re-plan's candidates."""
        pick = picks[int(i)]
        sel = np.flatnonzero(replan == pick)
        for h in list(getattr(render, "_handles", [])):
            h.remove()
        handles = []
        if show_paths.value:
            for j in sel:
                col = (200, 60, 40) if failed[j] else (30, 160, 140)
                handles.append(server.scene.add_spline_catmull_rom(
                    f"/cand/{pick}/{int(d['k'][j])}",
                    positions=trajs[j].astype(np.float32), color=col, line_width=2.0))
        render._handles = handles

        n_fail = int(failed[sel].sum())
        blocks = _window_blocks(windows.shape[-1])
        plot.image = _panel_png(windows[sel], failed[sel], d["k"][sel], blocks)
        if urdf_vis is not None and "qpos" in d.files:
            q = d["qpos"][sel[0]]
            # MuJoCo qpos is [pos(3), quat_wxyz(4), joints]; viser wants wxyz too, so the
            # slice transfers directly. All candidates share the branch state by construction,
            # so any of them poses the robot identically -- sel[0] is not a choice.
            server.scene.add_frame("/robot", position=tuple(float(v) for v in q[:3]),
                                   wxyz=tuple(float(v) for v in q[3:7]),
                                   show_axes=False)
            cfg = np.asarray(q[7:7 + len(urdf_joint_names)], dtype=np.float64)
            if cfg.size == len(urdf_joint_names):
                urdf_vis.update_cfg(cfg)
        w = windows[sel].reshape(len(sel), -1)
        scores, frac = _pca2(w)
        rows = "\n".join(
            f"| {int(d['k'][j])} | {'FELL' if failed[j] else 'survived'} | "
            f"{int(d['steps'][j])} | {float(d['progress'][j]):.2f} |" for j in sel)
        lbl.content = (
            f"### re-plan {pick} of {meta['n_chunks']}\n"
            f"**{len(sel) - n_fail}/{len(sel)} candidates survive**  "
            f"(reference ended: {meta['ref_verdict']})\n\n"
            f"PCA of emitted windows explains {float(frac):.0%} of variance; "
            f"scores (PC1, PC2):\n\n"
            + "\n".join(f"- k={int(d['k'][j])}: ({scores[n,0]:+.2f}, {scores[n,1]:+.2f})"
                        f"  {'FELL' if failed[j] else 'ok'}"
                        for n, j in enumerate(sel))
            + "\n\n| k | outcome | steps | progress (m) |\n|---|---|---|---|\n" + rows)

    idx.on_update(lambda _: render(idx.value))
    show_paths.on_update(lambda _: render(idx.value))
    prev_btn.on_click(lambda _: setattr(idx, "value", max(0, idx.value - 1)))
    next_btn.on_click(lambda _: setattr(idx, "value", min(len(picks) - 1, idx.value + 1)))
    render(0)

    print(f"[view] http://localhost:{args.port}   (ctrl-c to stop)")
    import time
    while True:
        time.sleep(1.0)
