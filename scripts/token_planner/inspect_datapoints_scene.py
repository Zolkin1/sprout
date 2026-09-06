# Copyright 2026 Zachary Olkin. All rights reserved.

"""The viser half of ``inspect_datapoints.py``: one datapoint, every arm, in 3-D.

Split out for length rather than for dependency -- it runs in the same process, so
``robot_rl`` is already imported and there is nothing to gain by restating
:mod:`robot_rl.flow_matching.viz.robot`'s primitives here. Those are imported: they carry
the URDF path declared ONCE (five viewers used to hold their own copy and three had drifted)
and the joint permutation, which is load-bearing -- the recording's Isaac interleaved joint
order differs from the URDF's per-limb order in 27 of 29 positions, and getting it wrong
draws a plausible, entirely wrong skeleton.

What it does NOT reuse is ``RobotVisual``: that is keyed by recorded ENV, cycles five colours
and has no notion of feet, where this needs one visual per ARM, a distinct colour for each of
a dozen, and the FK'd ankle paths that are the whole point of drawing a trajectory rather
than tabulating it.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import viser
from viser.extras import ViserUrdf

from motion_gen.features import _REPO_ROOT

from robot_rl.flow_matching.viz.playback import PlaybackControls
from robot_rl.flow_matching.viz.robot import (
    DEFAULT_URDF,
    load_urdf,
    recording_to_urdf_perm,
    xyzw_to_wxyz,
)

_PLACEHOLDER_SEGMENTS = np.zeros((1, 2, 3), dtype=np.float32)
_PLACEHOLDER_POINTS = np.zeros((1, 3), dtype=np.float32)

#: Ghosts fade from near-solid at the scrubbed node to faint at the horizon.
_GHOST_ALPHA_NEAR, _GHOST_ALPHA_FAR = 0.50, 0.12


def _safe(name: str) -> str:
    """A viser scene-node segment. ``/`` in an arm name would nest it under another node."""
    return name.replace("/", "__")


def _polyline(points: np.ndarray) -> np.ndarray:
    """``(T, 3)`` path -> ``(T-1, 2, 3)`` line segments."""
    if points.shape[0] < 2:
        return _PLACEHOLDER_SEGMENTS
    return np.stack([points[:-1], points[1:]], axis=1).astype(np.float32)


class ArmVisual:
    """Body, base ribbon, foot ribbons and ghosts for one arm."""

    def __init__(self, server: viser.ViserServer, urdf, name: str,
                 color: tuple[float, float, float], joint_perm: np.ndarray,
                 max_ghosts: int = 8):
        """Build the scene nodes for one arm; ghosts are created lazily.

        Args:
            server: The viser server.
            urdf: The shared loaded URDF (see :func:`load_urdf`).
            name: Arm name; ``/`` is escaped for the node path.
            color: RGB in 0-1.
            joint_perm: Recording-order -> URDF-actuated-order index array.
            max_ghosts: Ghost slot pool size.
        """
        self._server, self._urdf, self._perm = server, urdf, joint_perm
        self._color = color
        self._max_ghosts = max_ghosts
        self._ghosts: list[tuple[Any, ViserUrdf]] | None = None
        self._pt: dict[str, np.ndarray] | None = None
        self.show_body = True
        self.show_curve = True
        self.show_feet = True
        self.show_foot_nodes = True
        self.show_ghosts = False
        self.visible = True

        root = f"/arm/{_safe(name)}"
        self._base = server.scene.add_frame(f"{root}/body", show_axes=False)
        self._body = ViserUrdf(server, urdf, root_node_name=f"{root}/body",
                               mesh_color_override=(*color, 1.0))
        self._curve = server.scene.add_line_segments(
            f"{root}/curve", points=_PLACEHOLDER_SEGMENTS, colors=color,
            line_width=4.0, visible=False)
        # Same width as the base ribbon: the feet are the channel a stair failure shows up
        # in first, so a thinner line there would be exactly backwards.
        self._feet = server.scene.add_line_segments(
            f"{root}/feet", points=_PLACEHOLDER_SEGMENTS, colors=color,
            line_width=4.0, visible=False)
        # One dot per predicted node on each foot: the ribbon says WHERE the foot goes, the
        # dot spacing says how fast -- bunched dots are a stance phase, spread ones a swing.
        self._foot_pts = server.scene.add_point_cloud(
            f"{root}/foot_pts", points=_PLACEHOLDER_POINTS, colors=color,
            point_size=0.008, point_shape="circle", visible=False)
        self._ghost_root = f"{root}/ghosts"

    # -- geometry -------------------------------------------------------
    def set_point(self, pt: dict[str, np.ndarray]) -> None:
        """Load one datapoint's trajectory for this arm (``base_pos`` / ``base_quat`` /
        ``joint_pos`` / ``feet``)."""
        self._pt = pt
        self._curve.points = _polyline(pt["base_pos"])
        feet = pt["feet"]                                        # (H, K, 3)
        segs = [_polyline(feet[:, k]) for k in range(feet.shape[1])]
        self._feet.points = np.concatenate(segs, axis=0).astype(np.float32)
        self._foot_pts.points = feet.reshape(-1, 3).astype(np.float32)

    def set_frame(self, k: int) -> None:
        """Pose the solid body (and any ghosts) at chunk position ``k``."""
        if self._pt is None:
            return
        h = self._pt["base_pos"].shape[0]
        k = int(np.clip(k, 0, h - 1))
        self._base.position = self._pt["base_pos"][k]
        self._base.wxyz = xyzw_to_wxyz(self._pt["base_quat"][k])
        self._body.update_cfg(self._pt["joint_pos"][k][self._perm])
        if self.show_ghosts and self.visible:
            self._pose_ghosts(k, h)
        elif self._ghosts is not None:
            for frame, _ in self._ghosts:
                frame.visible = False
        self._apply()

    def _pose_ghosts(self, k: int, h: int) -> None:
        """Spread the ghost pool over ``[k, H)``, fading with look-ahead."""
        if self._ghosts is None:
            self._ghosts = []
            for s in range(self._max_ghosts):
                frac = s / max(self._max_ghosts - 1, 1)
                alpha = _GHOST_ALPHA_NEAR + (_GHOST_ALPHA_FAR - _GHOST_ALPHA_NEAR) * frac
                frame = self._server.scene.add_frame(f"{self._ghost_root}/{s}",
                                                     show_axes=False, visible=False)
                self._ghosts.append((frame, ViserUrdf(
                    self._server, self._urdf, root_node_name=f"{self._ghost_root}/{s}",
                    mesh_color_override=(*self._color, alpha))))
        idx = np.round(np.linspace(k, h - 1, self._max_ghosts)).astype(int)
        for s, (frame, ghost) in enumerate(self._ghosts):
            j = int(idx[s])
            frame.position = self._pt["base_pos"][j]
            frame.wxyz = xyzw_to_wxyz(self._pt["base_quat"][j])
            ghost.update_cfg(self._pt["joint_pos"][j][self._perm])
            frame.visible = True

    # -- visibility -----------------------------------------------------
    def _apply(self) -> None:
        """Push the toggle state onto the handles."""
        on = self.visible
        self._base.visible = on and self.show_body
        self._curve.visible = on and self.show_curve
        self._feet.visible = on and self.show_feet
        self._foot_pts.visible = on and self.show_feet and self.show_foot_nodes
        if self._ghosts is not None and not (on and self.show_ghosts):
            for frame, _ in self._ghosts:
                frame.visible = False

    def set_visible(self, on: bool) -> None:
        """Toggle the whole arm."""
        self.visible = on
        self._apply()


def _cloud_colors(valid: np.ndarray, cam: np.ndarray, mode: str) -> np.ndarray:
    """Per-point RGB for the depth cloud.

    Args:
        valid: ``(P,)`` validity plane, thresholded.
        cam: ``(P,)`` camera index.
        mode: ``"validity"``, ``"camera"`` or ``"height"``.

    Returns:
        ``(P, 3)`` uint8.
    """
    p = valid.shape[0]
    out = np.zeros((p, 3), np.uint8)
    if mode == "camera":
        out[cam == 0] = (90, 190, 255)
        out[cam == 1] = (255, 170, 90)
    else:
        out[:] = (110, 200, 255)
    out[~valid] = (200, 60, 90)
    return out


def serve(payload: dict[str, Any], port: int = 8080, smoke: bool = False) -> int:
    """Serve the scene. Blocks, unless ``smoke``.

    Args:
        payload: Built by ``inspect_datapoints._report``.
        port: Requested port.
        smoke: Drive every dropdown and toggle once, then return instead of blocking. The
            callbacks are the only place a per-datapoint shape mismatch can surface, and a
            headless run never reaches them otherwise.

    Returns:
        Exit code (only on a clean shutdown; the playback loop otherwise blocks forever).
    """
    urdf_path = payload.get("urdf") or str(_REPO_ROOT / DEFAULT_URDF)
    if not os.path.exists(urdf_path):
        raise SystemExit(f"URDF not found: {urdf_path}")
    urdf = load_urdf(urdf_path)
    perm = recording_to_urdf_perm(list(payload["joint_names"]), urdf)

    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", width=14.0, height=14.0, cell_size=0.5)
    print(f"[inspect] viser on http://localhost:{server.get_port()}")

    arms = payload["arms"]
    visuals = {a["name"]: ArmVisual(server, urdf, a["name"], tuple(a["color"]), perm)
               for a in arms}
    cloud = server.scene.add_point_cloud(
        "/extero/cloud", points=_PLACEHOLDER_POINTS, colors=(0.4, 0.8, 1.0),
        point_size=0.01, point_shape="circle", visible=True)
    anchor = server.scene.add_frame("/anchor", show_axes=True, axes_length=0.3,
                                    axes_radius=0.008)

    points = payload["points"]
    labels = [p["label"] for p in points]

    with server.gui.add_folder("Datapoint"):
        picker = server.gui.add_dropdown("window", options=labels, initial_value=labels[0])
        recentre = server.gui.add_button("reset camera",
                                         hint="Every datapoint is drawn at the origin, so "
                                              "this always finds the robot.")
        head = server.gui.add_markdown(payload["artifacts"])
        table = server.gui.add_markdown("")

    with server.gui.add_folder("Exteroception"):
        show_cloud = server.gui.add_checkbox("depth cloud", True,
                                             hint="The xyz channels of the row the generator "
                                                  "was conditioned on, lifted to world.")
        cloud_mode = server.gui.add_dropdown("colour by",
                                            options=["validity", "camera"],
                                            initial_value="validity")
        hide_invalid = server.gui.add_checkbox("drop invalid pixels", False)
        cloud_size = server.gui.add_slider("point size", min=0.002, max=0.05, step=0.002,
                                           initial_value=0.01)

    toggles: dict[str, dict[str, Any]] = {}
    with server.gui.add_folder("Arms"):
        all_bodies = server.gui.add_checkbox("show every body", False,
                                             hint="Twelve solid robots overlap; off by "
                                                  "default, the ribbons carry the shape.")
        ghosts_for = server.gui.add_dropdown(
            "ghosts for", options=["(none)"] + [a["name"] for a in arms],
            initial_value="(none)",
            hint="Ghost robots along the horizon, for one arm at a time.")
        foot_nodes = server.gui.add_checkbox("foot nodes", True,
                                             hint="One dot per predicted node on each foot. "
                                                  "Their SPACING is the foot speed; the "
                                                  "ribbon alone does not show it.")
        for a in arms:
            with server.gui.add_folder(a["name"], expand_by_default=False):
                server.gui.add_markdown(a["note"])
                toggles[a["name"]] = {
                    "on": server.gui.add_checkbox("show", not a["hidden"]),
                    "body": server.gui.add_checkbox("body", False),
                    "curve": server.gui.add_checkbox("base path", True),
                    "feet": server.gui.add_checkbox("feet", True),
                }

    # A real swatch per arm rather than a colour NAME: half the palette is a shade of pink or
    # orange on purpose (the sampled draws and the token-forcing sweep are families), which is
    # legible in the scene and unnameable in text.
    with server.gui.add_folder("Legend", expand_by_default=True):
        server.gui.add_markdown(
            "Every arm is `history = clip`. Solid robot = the scrubbed node; ribbon = the "
            "whole 64-node plan; dots = per-node foot positions.")
        for a in arms:
            server.gui.add_rgb(
                a["name"], tuple(int(round(255 * c)) for c in a["color"]),
                disabled=True, hint=a["note"])

    play = PlaybackControls(server, int(payload["chunk_len"]), float(payload["dt"]))

    with server.gui.add_folder("Consistency", expand_by_default=False):
        server.gui.add_markdown("```\n" + payload["checks"] + "\n```")

    state = {"i": 0}

    def _look(client: viser.ClientHandle) -> None:
        """Point one client at the robot. Every datapoint is drawn at the origin, so this is
        the same view for all of them -- which is the point of recentring them."""
        client.camera.position = (3.0, 3.0, 2.0)
        client.camera.look_at = (0.0, 0.0, 0.8)

    def _load(i: int) -> None:
        """Swap in datapoint ``i``: every arm's trajectory, the cloud and the anchor."""
        state["i"] = i
        pt = points[i]
        for name, vis in visuals.items():
            vis.set_point(pt["arms"][name])
        pts, valid, cam = pt["cloud"], pt["cloud_valid"], pt["cloud_cam"]
        keep = valid if hide_invalid.value else np.ones_like(valid)
        finite = np.isfinite(pts).all(axis=-1) & keep
        cloud.points = (pts[finite] if finite.any() else _PLACEHOLDER_POINTS)
        cloud.colors = (_cloud_colors(valid[finite], cam[finite], cloud_mode.value)
                        if finite.any() else np.zeros((1, 3), np.uint8))
        anchor.position = pt["anchor_pos"]
        anchor.wxyz = xyzw_to_wxyz(pt["anchor_quat"])
        table.content = "```\n" + pt["table"] + "\n```"

    def _refresh(_evt: Any = None) -> None:
        """One dispatch for every handle: re-read the toggles and re-pose the frame."""
        i = labels.index(picker.value)
        if i != state["i"]:
            _load(i)
        cloud.visible = bool(show_cloud.value)
        cloud.point_size = float(cloud_size.value)
        for name, vis in visuals.items():
            t = toggles[name]
            vis.show_body = bool(t["body"].value) or bool(all_bodies.value)
            vis.show_curve = bool(t["curve"].value)
            vis.show_feet = bool(t["feet"].value)
            vis.show_foot_nodes = bool(foot_nodes.value)
            vis.show_ghosts = ghosts_for.value == name
            vis.set_visible(bool(t["on"].value))
        for vis in visuals.values():
            vis.set_frame(int(play.frame_slider.value))

    def _reload(_evt: Any = None) -> None:
        """Handles that change the cloud itself, not just visibility."""
        _load(labels.index(picker.value))
        _refresh()

    @recentre.on_click
    def _(_evt: Any) -> None:
        """Put every connected client back on the robot."""
        for client in server.get_clients().values():
            _look(client)

    play.on_frame(lambda _k: _refresh())
    for handle in (picker, show_cloud, cloud_size, all_bodies, ghosts_for, foot_nodes):
        handle.on_update(_refresh)
    for handle in (cloud_mode, hide_invalid):
        handle.on_update(_reload)
    for t in toggles.values():
        for handle in t.values():
            handle.on_update(_refresh)

    _load(0)
    _refresh()

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        """Point a new client at the robot."""
        _look(client)

    if smoke:
        for label in labels:
            picker.value = label
            for k in (0, int(payload["chunk_len"]) // 2, int(payload["chunk_len"]) - 1):
                play.frame_slider.value = k
        for handle in (show_cloud, hide_invalid, all_bodies, foot_nodes):
            handle.value = not handle.value
            handle.value = not handle.value
        for mode in cloud_mode.options:
            cloud_mode.value = mode
        for name in [a["name"] for a in arms]:
            ghosts_for.value = name
        ghosts_for.value = "(none)"
        for t in toggles.values():
            for handle in t.values():
                handle.value = not handle.value
                handle.value = not handle.value
        print(f"[inspect] scene smoke test passed: {len(labels)} datapoints x "
              f"{len(arms)} arms, every toggle exercised.")
        server.stop()
        return 0

    play.run()
    return 0
