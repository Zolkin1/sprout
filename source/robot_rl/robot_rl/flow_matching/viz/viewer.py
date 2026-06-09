# Copyright 2026 Zachary Olkin. All rights reserved.

"""Viser viewer for flow-matching visualization recordings.

Renders the recorded terrain, every recorded robot, and per-robot prediction
overlays (ghost robots with fading alpha, predicted-root curve, height-scan
points) with playback controls. No IsaacLab imports anywhere in this module.
"""

from __future__ import annotations

import threading

import numpy as np
import trimesh
import viser

from .playback import PlaybackControls
from .recording import VizRecordingMeta
from .robot import RobotVisual, load_urdf, recording_to_urdf_perm

_TERRAIN_FALLBACK_COLOR = (0.55, 0.55, 0.55)


class FlowMatchingViewer:
    """Interactive viser scene for one recording."""

    def __init__(
        self,
        data: dict[str, np.ndarray],
        meta: VizRecordingMeta,
        urdf_path: str,
        max_ghosts: int = 10,
        initial_ghosts: int = 6,
        point_size: float = 0.02,
        port: int = 8080,
    ):
        """Build the full scene and GUI; call :meth:`run` to start playback.

        Args:
            data: Recording arrays from ``load_viz_recording``.
            meta: Recording metadata.
            urdf_path: Robot URDF to render (recorded path or CLI override).
            max_ghosts: Ghost slot-pool size per robot (slider maximum).
            initial_ghosts: Initially active ghost count.
            point_size: Height-scan point size in meters.
            port: Viser server port.
        """
        self._data = data
        self._meta = meta
        self._lock = threading.Lock()  # playback thread vs viser GUI thread pool

        self.server = viser.ViserServer(port=port)
        self.server.scene.set_up_direction("+z")
        self._add_terrain()

        urdf = load_urdf(urdf_path)
        joint_perm = recording_to_urdf_perm(meta.joint_names, urdf)
        self._robots = [
            RobotVisual(
                self.server, urdf, env_index=i, joint_perm=joint_perm,
                future_len=meta.future_len, max_ghosts=max_ghosts,
                num_ghosts=initial_ghosts, point_size=point_size,
            )
            for i in range(meta.num_envs)
        ]

        self._playback = PlaybackControls(self.server, num_frames=meta.num_steps, dt=meta.dt)
        with self.server.gui.add_folder("Prediction"):
            ghost_slider = self.server.gui.add_slider(
                "Ghosts", min=0, max=max_ghosts, step=1, initial_value=initial_ghosts,
                hint="Ghost robots shown along the predicted trajectory.",
            )
            alpha_slider = self.server.gui.add_slider(
                "Ghost alpha", min=0.05, max=2.0, step=0.05, initial_value=1.0,
                hint="Scales the ghosts' near-to-far alpha ramp (1.0 = default, >1 more solid).",
            )
        with self.server.gui.add_folder("Robots"):
            visible_checkboxes = []
            pred_checkboxes = []
            for i in range(meta.num_envs):
                visible_checkboxes.append(
                    self.server.gui.add_checkbox(f"Robot {i} visible", initial_value=True)
                )
                pred_checkboxes.append(
                    self.server.gui.add_checkbox(f"Robot {i} prediction", initial_value=(i == 0))
                )

        for i in range(meta.num_envs):
            self._robots[i].set_overlay_enabled(pred_checkboxes[i].value)

            @visible_checkboxes[i].on_update
            def _(_, robot=self._robots[i], cb=visible_checkboxes[i]) -> None:
                with self._lock:
                    robot.set_visible(cb.value)
                self.set_frame(int(self._playback.frame_slider.value))

            @pred_checkboxes[i].on_update
            def _(_, robot=self._robots[i], cb=pred_checkboxes[i]) -> None:
                with self._lock:
                    robot.set_overlay_enabled(cb.value)
                self.set_frame(int(self._playback.frame_slider.value))

        @ghost_slider.on_update
        def _(_) -> None:
            with self._lock:
                for robot in self._robots:
                    robot.set_num_ghosts(int(ghost_slider.value))
            self.set_frame(int(self._playback.frame_slider.value))

        @alpha_slider.on_update
        def _(_) -> None:
            with self._lock:
                for robot in self._robots:
                    robot.set_ghost_alpha(float(alpha_slider.value))

        self._set_initial_camera()
        self._playback.on_frame(self.set_frame)
        self.set_frame(0)

    def _add_terrain(self) -> None:
        """Add the recorded terrain mesh (vertex-colored when available)."""
        vertices = self._data["terrain_vertices"]
        faces = self._data["terrain_faces"]
        colors = self._data.get("terrain_vertex_colors")
        if colors is not None:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors, process=False)
            self.server.scene.add_mesh_trimesh("/terrain", mesh)
        else:
            self.server.scene.add_mesh_simple(
                "/terrain", vertices, faces.astype(np.int64), color=_TERRAIN_FALLBACK_COLOR, side="double"
            )

    def _set_initial_camera(self) -> None:
        """Point new clients' cameras at robot 0's starting pose."""
        target = self._data["root_pos"][0, 0]

        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            client.camera.position = target + np.array([2.5, 2.5, 1.5])
            client.camera.look_at = target

    def set_frame(self, i: int) -> None:
        """Render recorded frame ``i`` (single code path for play/step/drag/GUI)."""
        data = self._data
        with self._lock:
            for r, robot in enumerate(self._robots):
                if not robot.visible:
                    continue
                robot.set_state(data["root_pos"][i, r], data["root_quat_xyzw"][i, r], data["joint_pos"][i, r])
                if not robot.enabled:
                    continue
                k = int(data["pred_index"][i, r])
                if k < 0:
                    robot.clear_prediction()
                else:
                    robot.set_prediction(
                        data["root_pos"][i, r],
                        data["pred_root_pos"][k],
                        data["pred_root_quat_xyzw"][k],
                        data["pred_joint_pos"][k],
                    )
                robot.set_scan(data["scan_hits"][i, r] if "scan_hits" in data else None)

    def run(self) -> None:
        """Block forever, driving playback (Ctrl-C to exit)."""
        self._playback.run()
