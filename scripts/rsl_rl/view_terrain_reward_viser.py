# Copyright 2026 Zachary Olkin. All rights reserved.

"""See what the terrain-consistency reward sees: the plan, the terrain, and the COLLIDER POINTS.

The reward is a geometric claim about a predicted plan, so the way to trust it is to look at
the geometry it is made of.  Everything drawn here comes from the same
``motion_gen.plan_geometry`` / ``motion_gen.terrain_consistency`` producers the reward uses --
this is not a second implementation.

**Every layer is a separately named, separately toggleable scene node**, because the first
version of this tool drew red and green point clouds with no way to tell which was which.
The scene tree is the legend; the GUI repeats it in words.

What is on screen, from the ground up:

* ``/terrain`` -- the tile, rebuilt from the clip library's ``terrain_manifest`` by the same
  ``view_clip_viser._terrain_mesh`` four other viewers use.
* ``/plan/root_path`` -- the plan's PELVIS, one point per node (64 of them).  Orange.
* ``/plan/{left,right}_foot/{stance,buffer,swing}`` -- **foot SOLE points, every node**: the
  56 samples on each foot's bottom face, split into three clouds by what the velocity
  heuristic says.  This is the contact flag the reward gates on, drawn.
* ``/colliders`` -- **the full collider point set at ONE node** (~1886 surface samples,
  radius 0), coloured by signed distance to the terrain, penetrating points in red.  This is
  the panel that answers "what geometry is the reward actually using".
* ``/robot`` -- the robot at that node.  Visual meshes by default; the URDF's **collision**
  primitives can be shown instead, which is what the collider points are sampled from, so
  the two should coincide exactly.

Run (needs no simulator)::

    PYTHONPATH=source/robot_rl:source/motion_gen python scripts/rsl_rl/view_terrain_reward_viser.py \\
        --rollout logs/box2/rollouts/train_v0.50/rollout_<teacher> --port 8080

``--check N`` drives the whole draw path headlessly and exits, as a regression test.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import torch
import viser
import viser.uplot as uplot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from motion_gen import features as MGF                                      # noqa: E402
from motion_gen.joint_names import G1_JOINT_NAMES                           # noqa: E402
from motion_gen.plan_geometry import (                                      # noqa: E402
    CONTACT_DILATE_NODES,
    CONTACT_MAX_GAP_NODES,
    DEFAULT_COLLIDER_SPACING,
    CONTACT_SPEED_THRESH,
    FOOT_BODIES,
    PlanColliderModel,
    plan_collider_points,
    plan_contact,
)
from motion_gen.terrain_consistency import (                                # noqa: E402
    TerrainConsistencySpec,
    terrain_consistency,
)
import terrain_reward_io as trio                                            # noqa: E402
from view_clip_viser import _terrain_mesh                                   # noqa: E402

#: Colour of the plan's pelvis path.  Matches ``_DEBUG_FRAME_COLOR_MAP["pelvis_link"]`` so the
#: Isaac ribbon and this viewer name the same thing the same colour.
ROOT_RGB: tuple[int, int, int] = (242, 140, 26)

#: Per-side foot colours, also mirrored from ``_DEBUG_FRAME_COLOR_MAP``.
SIDE_RGB: dict[str, tuple[int, int, int]] = {
    "left_ankle_roll_link": (51, 115, 255),
    "right_ankle_roll_link": (230, 51, 64),
}

#: Contact-state colours.  ``stance`` matches ``fm_tracking_cmd.DEBUG_CONTACT_COLOR``.
#:
#: ``filled`` is a stance node the gap-closing filter ADDED (the velocity test fragmented one
#: stance phase).  It is graded exactly like a detected stance, so it gets a colour close to
#: stance's -- but a distinct one, because whether the filter is finding real stance interiors
#: or swallowing real swings is a judgement the viewer has to be able to support.
STATE_RGB: dict[str, tuple[int, int, int]] = {
    "stance": (38, 217, 89),
    "filled": (120, 235, 175),
    "buffer": (235, 200, 60),
    "swing": (130, 140, 160),
}

#: Point size (m) per contact state, so the state is readable even when colouring by SIDE.
STATE_SIZE: dict[str, float] = {"stance": 0.016, "filled": 0.014, "buffer": 0.011,
                                "swing": 0.007}

#: Colour for a collider point the reward actually CHARGES for penetrating.
CHARGED_RGB: tuple[int, int, int] = (235, 45, 35)

#: Colour for a point inside the terrain that the reward deliberately does NOT charge.
#:
#: Two reasons a point can be inside and uncharged, and they must be visible as such or the
#: viewer over-states the penalty: the 5 mm tolerance, and the +-2 node contact buffer, which
#: exempts a FOOT's own points near a detected stance. Without this colour the red cloud is
#: not what the reward reads -- measured on one window: 224 points inside, none charged.
EXEMPT_RGB: tuple[int, int, int] = (250, 170, 40)

TERRAIN_RGB: tuple[float, float, float] = (0.55, 0.55, 0.58)

LEGEND = """
### what you are looking at

**orange, big** — `plan/root_path`: the plan's **pelvis**, one point per node (the whole 64-node plan).

**foot sole points** — `plan/left_foot/*` and `plan/right_foot/*`: the 56 samples on each
foot's **bottom face**, at *every* node. Split by the velocity heuristic:

| | |
|---|---|
| **green, big** | `stance` — slowest sole point below the threshold |
| **pale green** | `filled` — a gap of ≤3 nodes *between* two stances, closed into them. Graded exactly like `stance`; the velocity test fragments one stance phase and this puts it back together |
| **yellow, mid** | `buffer` — within ±2 nodes of a stance. NOT graded as contact, only exempt from the penetration penalty |
| **grey, small** | `swing` — airborne |

Switch *colour feet by* to **side** to get blue = left, red = right instead (size still shows
the state).

**`colliders`** — the **full collider set at the one selected node** (~1886 surface samples,
radius 0), coloured by signed distance: dark = touching, bright = far. Inside the terrain
splits two ways, and the difference is what the reward actually charges:

| | |
|---|---|
| **red** | inside AND **charged** — past the 5 mm tolerance and not exempt |
| **amber** | inside but **not charged** — within the tolerance, or a foot's own point inside the ±2-node contact buffer |

Turn on *collision meshes* to see the primitives these are sampled from.

**`robot`** — the robot at the selected node.
"""


def _viridis(x: np.ndarray) -> np.ndarray:
    """Map ``x`` in ``[0, 1]`` to viridis RGB uint8.

    Args:
        x: Values in ``[0, 1]``.

    Returns:
        ``(N, 3)`` uint8 colours.
    """
    import matplotlib                                                       # noqa: PLC0415
    cmap = matplotlib.colormaps["viridis"]
    return (np.asarray(cmap(np.clip(x, 0.0, 1.0)))[:, :3] * 255).astype(np.uint8)


_EMPTY = np.zeros((1, 3), np.float32)

#: Scale (in reward units) below which the per-node plot's asinh axis is effectively linear.
#:
#: Both terms are exactly 0 on most nodes and heavy-tailed on the rest, so neither a linear
#: axis (the tail flattens everything else) nor a log one (zeros cannot be drawn) works.
#: asinh is linear inside this scale and logarithmic outside it. 0.01 puts the interesting
#: band -- a weighted term of order 0.01 to 1 -- across the visible range.
_ASINH_SCALE: float = 0.01


def _load_urdf_with_collision(urdf_path: str):
    """Load the URDF with BOTH scenes, so the collision primitives can be shown.

    ``viz.robot.load_urdf`` builds only the visual scene, which is right for every other
    viewer -- but here the collision geometry is the point: the collider samples come from
    exactly those primitives, so being able to overlay the two is how you confirm the
    sampling sits on the robot rather than near it.  Loaded here rather than by widening the
    shared helper, whose docstring turns on the visual scene being shared between many
    ``ViserUrdf`` instances.

    Args:
        urdf_path: The URDF to load.

    Returns:
        A ``yourdfpy.URDF`` with a collision scene built.
    """
    import os as _os                                                        # noqa: PLC0415
    from functools import partial                                           # noqa: PLC0415

    import yourdfpy                                                         # noqa: PLC0415
    return yourdfpy.URDF.load(
        urdf_path,
        build_scene_graph=True,
        build_collision_scene_graph=True,
        load_meshes=True,
        load_collision_meshes=True,
        filename_handler=partial(yourdfpy.filename_handler_magic,
                                 dir=_os.path.dirname(urdf_path)),
    )


class TerrainRewardScene:
    """The viser scene: persistent nodes, rebuilt per window and re-posed per node."""

    def __init__(self, server: viser.ViserServer, model: PlanColliderModel,
                 table: trio.TileTable, manifest: list[dict], urdf_path: str,
                 spec: TerrainConsistencySpec, joint_names: list[str]):
        """Create every scene node and the GUI.

        Args:
            server: The viser server.
            model: The collider model the reward uses.
            table: The tile table (tile geometry and routes).
            manifest: The library's raw ``terrain_manifest``, for the mesh rebuild.
            urdf_path: The robot URDF.
            spec: The reward constants.
            joint_names: The joint order the recorded plan is in.
        """
        self.server = server
        self.model = model
        self.table = table
        self.manifest = manifest
        self.spec = spec
        self._terrain = None

        self._root = server.scene.add_point_cloud(
            "/plan/root_path", points=_EMPTY, colors=np.array([ROOT_RGB], np.uint8),
            point_size=0.026, point_shape="circle")
        # One cloud per (foot, state): the scene tree then NAMES what each colour is, and
        # each can be isolated. Six nodes is worth it -- the first version of this drew one
        # cloud and was unreadable.
        self._soles: dict[tuple[str, str], viser.PointCloudHandle] = {}
        for foot in FOOT_BODIES:
            short = "left_foot" if foot.startswith("left") else "right_foot"
            for state in ("stance", "filled", "buffer", "swing"):
                self._soles[(foot, state)] = server.scene.add_point_cloud(
                    f"/plan/{short}/{state}", points=_EMPTY,
                    colors=np.array([STATE_RGB[state]], np.uint8),
                    point_size=STATE_SIZE[state], point_shape="circle")
        self._colliders = server.scene.add_point_cloud(
            "/colliders", points=_EMPTY, colors=np.array([STATE_RGB["swing"]], np.uint8),
            point_size=0.009, point_shape="circle")

        from robot_rl.flow_matching.viz.robot import (                      # noqa: PLC0415
            recording_to_urdf_perm, xyzw_to_wxyz)
        from viser.extras import ViserUrdf                                  # noqa: PLC0415
        self._xyzw_to_wxyz = xyzw_to_wxyz
        urdf = _load_urdf_with_collision(urdf_path)
        self._perm = recording_to_urdf_perm(list(joint_names), urdf)
        # ORDER MATTERS: the frame must exist BEFORE the ViserUrdf that parents to it, and the
        # HANDLE must be kept -- posing the robot is setting this frame's position/wxyz. The
        # first version re-added the frame every draw, which detached the meshes and left the
        # robot invisible at the origin.
        # A visible axis triad at the plan's anchor: with the scene hundreds of metres from
        # the origin, having one unmistakable landmark makes "the camera is aimed wrong"
        # distinguishable from "nothing was drawn".
        self._anchor_axes = server.scene.add_frame("/plan/anchor", show_axes=True,
                                                    axes_length=0.4, axes_radius=0.012)
        self._robot_frame = server.scene.add_frame("/robot", show_axes=False)
        self._robot = ViserUrdf(server, urdf, root_node_name="/robot",
                                load_collision_meshes=True)
        self._robot.show_visual = True
        self._robot.show_collision = False

        with server.gui.add_folder("legend", expand_by_default=True):
            server.gui.add_markdown(LEGEND)
        with server.gui.add_folder("window"):
            self.g_window = server.gui.add_slider("index", 0, 1, 1, 0)
            self.g_info = server.gui.add_text("plan", initial_value="", disabled=True)
            self.g_score = server.gui.add_text("score", initial_value="", disabled=True)
        with server.gui.add_folder("node"):
            self.g_node = server.gui.add_slider("node", 0, 1, 1, 0)
            self.g_node_info = server.gui.add_text("at node", initial_value="", disabled=True)
        with server.gui.add_folder("reward", expand_by_default=True):
            self.g_total = server.gui.add_text("total reward", initial_value="",
                                                disabled=True)
            self.g_terms = server.gui.add_text("terms", initial_value="", disabled=True)
            # WHICH LINK is being charged. The penetration weights are per body (feet 1x,
            # everything else 10x, because nothing but a foot is supposed to touch), so the
            # scalar alone cannot say whether a penalty is a benign stance foot or a shin
            # through a riser -- which is the distinction the weights exist to draw.
            self.g_bodies = server.gui.add_text("penetration by body", initial_value="",
                                                 disabled=True)
            # Live weights, seeded from the DECLARED reference balance. Exposed because the
            # two terms' natural scales differ by ~220x, so how they combine is the one thing
            # about this reward you cannot read off a single number.
            self.g_w_contact = server.gui.add_number(
                "weight: contact", initial_value=float(spec.contact_weight), min=0.0,
                step=1.0)
            self.g_w_pen = server.gui.add_number(
                "weight: penetration", initial_value=float(spec.penetration_weight),
                min=0.0, step=0.01)
            self.g_plot_log = server.gui.add_checkbox("plot: compress the tail", True)
            self.g_plot = server.gui.add_uplot(
                data=(np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)),
                series=(uplot.Series(label="node"),
                        uplot.Series(label="total", stroke="#f28c1a", width=2.0),
                        uplot.Series(label="contact", stroke="#26d959", width=1.5),
                        uplot.Series(label="penetration", stroke="#eb2d23", width=1.5)),
                # uPlot's x scale defaults to time=True, which formats the x series as Unix
                # timestamps -- so plan nodes 0..63 came out as clock times a minute apart.
                # The x axis here is a node INDEX, so the scale has to say so.
                scales={"x": uplot.Scale(time=False)},
                axes=(uplot.Axis(scale="x", label="plan node"),
                      uplot.Axis(scale="y", label="weighted penalty")),
                title="weighted penalty per node", aspect=1.5,
                legend=uplot.Legend(show=True))
        with server.gui.add_folder("view"):
            self.g_recenter = server.gui.add_button("recentre on the plan")
        with server.gui.add_folder("layers"):
            self.g_root = server.gui.add_checkbox("plan root path", True)
            self.g_sole = server.gui.add_checkbox("foot sole points (all nodes)", True)
            self.g_coll = server.gui.add_checkbox("collider points (selected node)", True)
            self.g_visual = server.gui.add_checkbox("robot visual meshes", True)
            self.g_collision = server.gui.add_checkbox("robot collision meshes", False)
        with server.gui.add_folder("colouring"):
            self.g_feet_by = server.gui.add_dropdown(
                "colour feet by", ("contact state", "side"), initial_value="contact state")
            self.g_coll_by = server.gui.add_dropdown(
                "colour colliders by",
                ("signed distance", "all penetration", "charged penetration only"),
                initial_value="signed distance")
            self.g_sd_range = server.gui.add_slider("sd colour range (m)", 0.05, 1.0, 0.05,
                                                     0.30)

    # -- per-window ------------------------------------------------------------------------
    def load_window(self, rollout: str, row: int, episode: dict, thresh: float,
                    dilate: int, blocks: tuple[str, ...], max_gap: int = 0) -> None:
        """Compute one recorded re-plan and rebuild the per-window layers.

        Args:
            rollout: The rollout ``.h5``.
            row: Window row index.
            episode: The per-episode index.
            thresh: Contact speed threshold (m/s).
            dilate: Contact dilation radius (nodes).
            blocks: The block selection the plan was packed with.
            max_gap: Longest non-contact run closed into the stance around it.
        """
        with h5py.File(rollout, "r") as f:
            plan = torch.as_tensor(f["fm_plan"][row : row + 1], dtype=torch.float64)
            a_pos = torch.as_tensor(f["anchor_pos"][row : row + 1], dtype=torch.float64)
            a_quat = torch.as_tensor(f["anchor_quat"][row : row + 1], dtype=torch.float64)
            uid, rep = int(f["episode_uid"][row]), int(f["replan_index"][row])

        er = episode["_uid_to_row"][uid]
        tid = int(episode["terrain_id"][er])
        corner = trio.corner_from_goal(self.table, np.array([tid]),
                                       episode["goal_pos_w"][er][None, :])

        bp, br, q, bq = trio.plan_to_world(plan, a_pos, a_quat, len(G1_JOINT_NAMES), 4, blocks)
        self.base_pos = bp.double()
        self.base_quat_xyzw = bq.double()
        self.joint_pos = q.double()
        self.pts = plan_collider_points(self.base_pos, br.double(), self.joint_pos, self.model)
        self.contact = plan_contact(self.pts, self.model, dt=0.02, thresh=thresh,
                                    dilate=dilate, max_gap=max_gap)
        terrain = trio.terrain_for_rows(self.table, np.array([tid]), corner)
        self.out = terrain_consistency(self.pts, self.contact, terrain, self.model, self.spec)
        self.sd = terrain.sdf(self.pts.reshape(1, -1, 3)).reshape(self.pts.shape[:3])[0]
        # What the reward SEES, reproduced exactly: the collider surface (sd - radius), the
        # tolerance, and the per-foot exemption. Anything less and the red cloud is a
        # different statement from the penalty.
        self.sd_eff = self.sd - self.model.radii.to(self.sd.dtype)
        self.depth = (-self.sd_eff - self.spec.penetration_tol_m).clamp_min(0.0)
        f_nodes, n_pts = self.sd.shape
        self.exempt = torch.zeros(f_nodes, n_pts, dtype=torch.bool)
        buffered = self.contact["contact_buffered"][0]
        for j, body_i in enumerate(self.model.foot_index.tolist()):
            own = self.model.body_index == body_i
            self.exempt |= buffered[:, j : j + 1] & own[None, :]
        self.corner = corner[0]

        self._rebuild_terrain(tid)
        self.g_node.max = int(self.pts.shape[1]) - 1
        fell = int(episode["first_fall_step"][er]) >= 0
        o = self.out
        self.g_info.value = (f"episode {uid}, re-plan {rep}, tile {tid} "
                             f"({'FELL' if fell else 'survived'}"
                             f"{', reached goal' if episode['reached_goal'][er] else ''})")
        n_raw = int(self.contact["contact_raw"][0].sum())
        n_fill = int(self.contact["contact_filled"][0].sum())
        self.g_score.value = (
            f"contact {o['contact_reward'][0]:.3f} | vol {o['penetration_volume_m3'][0]:.2e} m3"
            f" | max pen {o['max_penetration_m'][0] * 100:.1f} cm"
            f" | coverage {o['contact_coverage'][0]:.2f}"
            f" ({n_raw} detected + {n_fill} gap-filled foot-nodes)")
        self._anchor_axes.position = self.base_pos[0, 0].numpy()
        self._anchor_axes.wxyz = self._xyzw_to_wxyz(self.base_quat_xyzw[0, 0].numpy())
        self._draw_soles()
        self._draw_root()
        self.draw_reward()

    def draw_reward(self) -> None:
        """The reward panel: the weighted total, the breakdown, and the per-node series.

        The weights are read from the GUI rather than the spec, so the sliders are live -- but
        the per-node series and the scalar come from the SAME arrays
        (``node_contact_penalty`` / ``node_penetration_m``), so the plot and the number beside
        it cannot disagree.
        """
        w_c = float(self.g_w_contact.value)
        w_p = float(self.g_w_pen.value)
        node_c = self.out["node_contact_penalty"][0].numpy().astype(np.float64)
        node_p = self.out["node_penetration_m"][0].numpy().astype(np.float64)
        wc, wp = w_c * node_c, w_p * node_p
        total = wc + wp

        # A reward is the negated penalty; the terms are reported as the penalties they are.
        self.g_total.value = f"{-total.mean():+.4f}   (penalty {total.mean():.4f})"
        self.g_terms.value = (
            f"contact {node_c.mean():.5f} m x{w_c:g} = {wc.mean():.4f}  |  "
            f"penetration {node_p.mean():.4f} m x{w_p:g} = {wp.mean():.4f}")

        by_body = self.out["penetration_by_body"][0].numpy()
        ranked = sorted(zip(self.model.body_order, by_body), key=lambda kv: -kv[1])
        hot = [f"{b.replace('_link', '')} {v:.3f}" for b, v in ranked if v > 1e-9][:5]
        self.g_bodies.value = "  ".join(hot) if hot else "nothing penetrating"

        # asinh, not log: both terms are exactly ZERO on most nodes, which a log axis cannot
        # show at all. asinh is linear near 0 and logarithmic in the tail, so a clean plan
        # reads as a flat line on the axis instead of vanishing.
        def squash(x: np.ndarray) -> np.ndarray:
            """Compress the heavy tail while keeping exact zeros visible."""
            return np.arcsinh(x / _ASINH_SCALE) if self.g_plot_log.value else x

        nodes = np.arange(len(total), dtype=np.float64)
        self.g_plot.data = (nodes, squash(total), squash(wc), squash(wp))

    def _rebuild_terrain(self, tid: int) -> None:
        """Draw the tile mesh, translated onto its world corner."""
        if self._terrain is not None:
            self._terrain.remove()
            self._terrain = None
        got = _terrain_mesh(self.manifest, tid)
        if got is None:
            print(f"[WARN] no mesh for tile {tid}; drawing the plan without terrain")
            return
        verts, faces = got
        self._terrain = self.server.scene.add_mesh_simple(
            "/terrain", vertices=(verts + self.corner[None, :]).astype(np.float32),
            faces=faces.astype(np.uint32), color=TERRAIN_RGB, flat_shading=True)

    def _draw_root(self) -> None:
        """The plan's pelvis path, one point per node."""
        self._root.points = self.base_pos[0].numpy().astype(np.float32)
        self._root.colors = np.tile(np.array(ROOT_RGB, np.uint8),
                                    (self.base_pos.shape[1], 1))

    def _draw_soles(self) -> None:
        """Split every node's sole points into the three contact-state clouds."""
        raw = self.contact["contact_raw"][0].numpy()                  # (F, 2)
        fill = self.contact["contact_filled"][0].numpy()
        buf = self.contact["contact_buffered"][0].numpy()
        by_side = self.g_feet_by.value == "side"
        for j, foot in enumerate(FOOT_BODIES):
            pts = self.pts[0][:, self.model.sole_index[j]].numpy()   # (F, ns, 3)
            # Precedence: a detected stance, else a filled gap, else the buffer, else swing.
            state = np.where(raw[:, j], 0,
                             np.where(fill[:, j], 1, np.where(buf[:, j], 2, 3)))
            for code, name in enumerate(("stance", "filled", "buffer", "swing")):
                sel = state == code
                handle = self._soles[(foot, name)]
                if not sel.any():
                    handle.points = _EMPTY
                    handle.colors = np.array([STATE_RGB[name]], np.uint8)
                    continue
                p = pts[sel].reshape(-1, 3).astype(np.float32)
                rgb = SIDE_RGB[foot] if by_side else STATE_RGB[name]
                handle.points = p
                handle.colors = np.tile(np.array(rgb, np.uint8), (len(p), 1))

    # -- per-node --------------------------------------------------------------------------
    def set_node(self, node: int) -> None:
        """Draw the collider cloud and pose the robot at one plan node.

        Args:
            node: Plan node index.
        """
        node = int(np.clip(node, 0, self.pts.shape[1] - 1))
        p = self.pts[0, node].numpy().astype(np.float32)
        sd = self.sd[node].numpy()
        inside = self.sd_eff[node].numpy() < 0.0
        charged = (self.depth[node].numpy() > 0.0) & ~self.exempt[node].numpy()
        uncharged_inside = inside & ~charged

        mode = self.g_coll_by.value
        if mode == "charged penetration only":
            keep = charged
            rgb = CHARGED_RGB
        elif mode == "all penetration":
            keep = inside
            rgb = None
        else:
            keep = None

        if keep is not None:
            if not keep.any():
                self._colliders.points = _EMPTY
                self._colliders.colors = np.array([CHARGED_RGB], np.uint8)
            else:
                col = np.tile(np.array(CHARGED_RGB, np.uint8), (int(keep.sum()), 1))
                if rgb is None:
                    col[uncharged_inside[keep]] = EXEMPT_RGB
                self._colliders.points = p[keep]
                self._colliders.colors = col
        else:
            rng = float(self.g_sd_range.value)
            col = _viridis(np.clip(sd, 0.0, rng) / rng)
            col[uncharged_inside] = EXEMPT_RGB
            col[charged] = CHARGED_RGB
            self._colliders.points = p
            self._colliders.colors = col

        self._robot_frame.position = self.base_pos[0, node].numpy()
        self._robot_frame.wxyz = self._xyzw_to_wxyz(self.base_quat_xyzw[0, node].numpy())
        self._robot.update_cfg(self.joint_pos[0, node].numpy()[self._perm])

        ex = self.contact["contact_buffered"][0, node].numpy()
        deepest_charged = float(self.depth[node].numpy()[charged].max()) if charged.any() \
            else 0.0
        w_c = float(self.g_w_contact.value)
        w_p = float(self.g_w_pen.value)
        n_c = float(self.out["node_contact_penalty"][0, node])
        n_p = float(self.out["node_penetration_m"][0, node])
        self.g_node_info.value = (
            f"node {node}/{self.pts.shape[1] - 1}: {int(inside.sum())}/{len(sd)} inside, "
            f"{int(charged.sum())} CHARGED ({int(uncharged_inside.sum())} exempt); "
            f"deepest charged {deepest_charged * 100:.1f} cm, "
            f"deepest overall {max(0.0, float(-sd.min())) * 100:.1f} cm; "
            f"feet exempt: {'L' if ex[0] else '-'}{'R' if ex[1] else '-'}; "
            f"penalty {w_c * n_c + w_p * n_p:.4f} "
            f"(contact {w_c * n_c:.4f} + pen {w_p * n_p:.4f})")

    def aim_camera(self, clients=None) -> None:
        """Point every client's camera at the plan.

        **Load-bearing, not a nicety.** The terrain is assembled as a long row of tiles, so a
        tile's world corner is hundreds of metres from the origin (measured: x from -227 to
        +217 on this library). viser starts every camera at the origin looking at the origin,
        so a scene drawn in true world coordinates is entirely off-screen -- the GUI fills in
        and the 3-D view is empty, which looks exactly like nothing having been drawn.

        World coordinates are kept rather than re-centring the scene, so every position the
        readouts show still matches the rollout.

        Args:
            clients: Clients to aim, or ``None`` for all currently connected.
        """
        if not hasattr(self, "base_pos"):
            return
        target = self.base_pos[0, 0].numpy()
        if clients is None:
            clients = list(self.server.get_clients().values())
        for client in clients:
            client.camera.position = target + np.array([-3.5, -3.5, 2.5])
            client.camera.look_at = target

    def apply_visibility(self) -> None:
        """Honour the layer checkboxes."""
        self._root.visible = bool(self.g_root.value)
        for handle in self._soles.values():
            handle.visible = bool(self.g_sole.value)
        self._colliders.visible = bool(self.g_coll.value)
        self._robot.show_visual = bool(self.g_visual.value)
        self._robot.show_collision = bool(self.g_collision.value)


def _episode_index(path: str) -> dict:
    """The per-episode columns this viewer joins against.

    Args:
        path: The ``*.episodes.h5``.

    Returns:
        Mapping of column name -> array, plus ``goal_pos_w`` and ``_uid_to_row``.
    """
    with h5py.File(path, "r") as f:
        out = {k: f[k][:] for k in ("episode_uid", "terrain_id", "reached_goal",
                                    "first_fall_step")}
        out["goal_pos_w"] = np.stack([f["goal_pos_w_x"][:], f["goal_pos_w_y"][:],
                                      f["goal_pos_w_z"][:]], axis=-1)
    out["_uid_to_row"] = {int(u): i for i, u in enumerate(out["episode_uid"])}
    return out


def _pick_rows(rollout: str, n: int, seed: int, worst: bool, model: PlanColliderModel,
               episode: dict, table: trio.TileTable, blocks: tuple[str, ...],
               thresh: float, dilate: int, max_gap: int) -> np.ndarray:
    """Choose which windows are selectable.

    Args:
        rollout: The rollout ``.h5``.
        n: How many windows to offer.
        seed: RNG seed for the random draw.
        worst: Rank a larger random sample by max penetration and keep the deepest ``n``.
            The interesting plans are rare -- three quarters of them penetrate nothing -- so a
            plain random draw mostly shows plans with nothing wrong.
        model / episode / table / blocks / thresh / dilate: As for the grading path.

    Returns:
        Sorted row indices.
    """
    total = trio.rollout_meta(rollout)["_num_written"]
    rng = np.random.default_rng(seed)
    if not worst:
        return np.sort(rng.choice(total, size=min(n, total), replace=False))

    pool = np.sort(rng.choice(total, size=min(20 * n, total), replace=False))
    depth = np.empty(len(pool))
    with h5py.File(rollout, "r") as f:
        step = max(1, 4_000_000 // max(model.num_points * 64, 1))
        for lo in range(0, len(pool), step):
            idx = pool[lo : lo + step]
            plan = torch.as_tensor(f["fm_plan"][idx], dtype=torch.float64)
            ap = torch.as_tensor(f["anchor_pos"][idx], dtype=torch.float64)
            aq = torch.as_tensor(f["anchor_quat"][idx], dtype=torch.float64)
            er = np.array([episode["_uid_to_row"][int(u)] for u in f["episode_uid"][idx]])
            tid = episode["terrain_id"][er]
            corner = trio.corner_from_goal(table, tid, episode["goal_pos_w"][er])
            bp, br, q, _bq = trio.plan_to_world(plan, ap, aq, len(G1_JOINT_NAMES), 4, blocks)
            pts = plan_collider_points(bp.double(), br.double(), q.double(), model)
            out = terrain_consistency(pts, plan_contact(pts, model, dt=0.02, thresh=thresh,
                                                        dilate=dilate, max_gap=max_gap),
                                      trio.terrain_for_rows(table, tid, corner), model,
                                      TerrainConsistencySpec())
            depth[lo : lo + len(idx)] = out["max_penetration_m"].numpy()
    order = np.argsort(-depth)[: min(n, len(pool))]
    print(f"--worst: deepest penetration in the offered windows "
          f"{depth[order].max() * 100:.1f} cm, shallowest {depth[order].min() * 100:.1f} cm")
    return np.sort(pool[order])


def build_parser() -> argparse.ArgumentParser:
    """The CLI."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollout", required=True, help="rollout stem, WITHOUT .h5")
    p.add_argument("--clip", default="", help="clip library; default: the rollout's own")
    p.add_argument("--rows", type=int, default=64, help="how many windows to offer")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--spacing", type=float, default=DEFAULT_COLLIDER_SPACING,
                   help="collider surface-sample spacing (m); 0 for extreme points")
    p.add_argument("--thresh", type=float, default=CONTACT_SPEED_THRESH,
                   help="contact speed threshold (m/s); defaults to the reward's own")
    p.add_argument("--dilate", type=int, default=CONTACT_DILATE_NODES,
                   help="contact dilation radius in nodes; defaults to the reward's own")
    p.add_argument("--max_gap", type=int, default=CONTACT_MAX_GAP_NODES,
                   help="close runs of non-contact this long between two stances; "
                        "0 disables the filter")
    p.add_argument("--worst", action="store_true",
                   help="offer the DEEPEST-penetrating windows instead of random ones")
    p.add_argument("--check", type=int, default=0,
                   help="load this many windows, exercise the node slider, print a summary "
                        "and EXIT -- the whole draw path without a browser")
    p.add_argument("--isaac_joint_names", default="")
    p.add_argument("--urdf", default="")
    p.add_argument("--port", type=int, default=8080)
    return p


def main() -> None:
    """Serve the scene."""
    args = build_parser().parse_args()
    rollout = args.rollout + ".h5"
    spacing = args.spacing if args.spacing and args.spacing > 0 else None

    meta = trio.rollout_meta(rollout)
    clip = args.clip or meta["clip_library_path"]
    clip = clip if os.path.isabs(clip) else str(MGF._REPO_ROOT / clip)
    table = trio.load_tile_table(clip)
    with h5py.File(clip, "r") as f:
        raw = f.attrs["terrain_manifest"]
        manifest = json.loads(raw if isinstance(raw, str) else raw.decode())
    blocks = tuple(meta.get("reference_blocks") or trio.DEFAULT_BLOCKS)

    names = trio.isaac_joint_names(rollout, args.isaac_joint_names)
    if list(names) == list(G1_JOINT_NAMES):
        raise SystemExit("resolved the CLIP joint order for a PLAN; refusing -- see "
                         "inspect_terrain_reward._plan_model")
    model = PlanColliderModel(MGF.DEFAULT_MJCF, list(names), dtype=torch.float64,
                              spacing=spacing)
    print(f"colliders: {model.num_points} points over {len(model.body_order)} bodies, "
          f"{model.total_area_m2:.4f} m^2, {model.sole_index.shape[1]} sole points per foot")

    episode = _episode_index(args.rollout + ".episodes.h5")
    rows = _pick_rows(rollout, args.rows, args.seed, args.worst, model, episode, table,
                      blocks, args.thresh, args.dilate, args.max_gap)

    server = viser.ViserServer(port=args.port)
    urdf_path = args.urdf or str(MGF._REPO_ROOT / "robot_assets" / "g1_perceptive"
                                 / "g1_perceptive.urdf")
    scene = TerrainRewardScene(server, model, table, manifest, urdf_path,
                               TerrainConsistencySpec(), list(names))
    scene.g_window.max = len(rows) - 1

    def show(i: int) -> None:
        """Load and draw window ``i`` of the offered list, and re-aim the camera."""
        scene.load_window(rollout, int(rows[int(i)]), episode, args.thresh, args.dilate,
                          blocks, args.max_gap)
        scene.set_node(scene.g_node.value)
        scene.apply_visibility()
        scene.aim_camera()

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        """A client that connects after the scene is built still needs aiming."""
        scene.aim_camera([client])

    @scene.g_recenter.on_click
    def _(_ev) -> None:
        scene.aim_camera()

    @scene.g_window.on_update
    def _(_ev) -> None:
        show(scene.g_window.value)

    @scene.g_node.on_update
    def _(_ev) -> None:
        scene.set_node(scene.g_node.value)

    for handle in (scene.g_root, scene.g_sole, scene.g_coll, scene.g_visual,
                   scene.g_collision):
        @handle.on_update
        def _(_ev) -> None:
            scene.apply_visibility()

    @scene.g_feet_by.on_update
    def _(_ev) -> None:
        scene._draw_soles()

    for handle in (scene.g_w_contact, scene.g_w_pen, scene.g_plot_log):
        @handle.on_update
        def _(_ev) -> None:
            scene.draw_reward()
            scene.set_node(scene.g_node.value)

    for handle in (scene.g_coll_by, scene.g_sd_range):
        @handle.on_update
        def _(_ev) -> None:
            scene.set_node(scene.g_node.value)

    if args.check:
        for i in range(min(int(args.check), len(rows))):
            show(i)
            f_nodes = scene.pts.shape[1]
            for node in (0, f_nodes // 2, f_nodes - 1):
                scene.set_node(node)
            scene._draw_soles()
            n_sole = sum(len(h.points) for h in scene._soles.values())
            scene.draw_reward()
            print(f"  window {i} (row {rows[i]}): {scene.g_info.value}")
            print(f"    {scene.g_score.value}")
            print(f"    reward {scene.g_total.value}")
            print(f"    {scene.g_terms.value}")
            print(f"    {scene.g_node_info.value}")
            print(f"    root pts {len(scene._root.points)}, sole pts {n_sole}, "
                  f"collider pts {len(scene._colliders.points)}, "
                  f"robot at {np.round(scene._robot_frame.position, 3).tolist()}")
        print(f"OK: {min(int(args.check), len(rows))} windows, {scene.pts.shape[1]} nodes, "
              f"{model.num_points} collider points")
        return

    show(0)
    print(f"serving on http://localhost:{args.port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
