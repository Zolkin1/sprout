#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Viser scene for ``fm_debug.py --stage view``: clip vs generator vs what the policy saw.

Loaded by :func:`fm_debug.stage_view`; never imports Isaac Lab, so it runs while a rollout
holds the GPU. Everything it draws was recorded by ``DataLogger(log_world=True)``.

The layers, and what each one answers:

``robot``
    The measured robot, posed from its logged joints. Ground truth.
``clip``
    The pre-made clip reference as a translucent ghost. What the baseline tracks.
``fm``
    The generator's reference as a ghost, plus its predicted F-node horizon as a line with
    fading ghosts along it. What the generator says.
``window``
    The reference-window OBSERVATION lifted back into world coordinates -- the 11 base
    samples the policy's encoder was fed, at the strided look-ahead, plus the end-effector
    samples. Two variants: ``clean`` (recomputed) and ``observed`` (reconstructed from the
    corrupted observation vector, so it carries the noise and the +-20 clip; identical to
    ``clean`` unless the rollout ran with ``--corrupt_obs``).

**The load-bearing comparison is ``fm`` against ``window``.** The window is the FM
prediction after the whole conversion chain -- world lift, base-relative differencing, the
observation packing -- and back again. If the two do not coincide, the conversion is wrong
and the generator is not the problem. The ``observed`` variant additionally shows how much
of the reference the training-time noise eats.

**Two different "clip base" quantities, and they are meant to differ.** The ghosts are posed
from the window's sample 0 -- the ANCHORED reference, re-datumed onto the robot every step,
which is what the policy observes and what the CLF grades the links against. The static
orange/green tubes are ``y_des``'s base channels, which on the clip path
``MotionSequenceCommand._write_global_base_des`` re-points at the GLOBALLY-placed clip so
the reward and the drift termination share a datum. The gap between the ghost and the tube
IS the accumulated global drift; reading it as a reference error is the mistake this note
exists to prevent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robot_rl.flow_matching.viz.robot import (
    RobotVisual,
    load_urdf,
    recording_to_urdf_perm,
    xyzw_to_wxyz,
)
from view_clip_viser import _DEFAULT_URDF, _terrain_mesh, _tube_mesh

_COLORS: dict[str, tuple[float, float, float]] = {
    "clip": (0.95, 0.55, 0.10),      # orange -- the pre-made reference
    "fm": (0.15, 0.85, 0.45),        # green  -- the generator
    "window": (0.95, 0.85, 0.20),    # yellow -- what the policy observed
    "actual": (0.65, 0.65, 0.68),    # grey   -- the measured robot
}
_GHOST_ALPHA = 0.45
_MODE_ORDER = ("clip", "fm")


# ---------------------------------------------------------------------------
# Bundle indexing
# ---------------------------------------------------------------------------

@dataclass
class Rollout:
    """One recorded rollout, pre-indexed for fast per-frame reads."""

    mode: str
    data: dict[str, np.ndarray]
    meta: dict
    env_of_case: dict[str, int] = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        """Recorded step count."""
        return int(self.data["root_pos_w"].shape[0])

    def get(self, key: str, step: int, env: int):
        """``data[key][step, env]`` if present, else ``None``."""
        arr = self.data.get(key)
        return None if arr is None else arr[step, env]


def _base_cols(pos_names: list[str], base_frame: str) -> tuple[np.ndarray, np.ndarray] | None:
    """``(position columns, xyzw orientation columns)`` of the base frame within ``y_des``."""
    try:
        pos = [pos_names.index(f"{base_frame}:pos_{a}") for a in "xyz"]
        ori = [pos_names.index(f"{base_frame}:ori_{a}") for a in ("x", "y", "z", "w")]
    except ValueError:
        return None
    return np.asarray(pos, dtype=np.int64), np.asarray(ori, dtype=np.int64)


def _index_rollouts(bundles: dict[str, tuple[dict, dict]]) -> tuple[list[Rollout], list[dict]]:
    """Wrap each bundle in a :class:`Rollout` and resolve its case -> env mapping.

    Returns:
        ``(rollouts, cases)`` where ``cases`` is the case list of the first bundle (the
        rollouts share a clip library, so the case set is common).
    """
    rollouts: list[Rollout] = []
    cases: list[dict] = []
    for mode in _MODE_ORDER:
        if mode not in bundles:
            continue
        data, meta = bundles[mode]
        roll = Rollout(mode=mode, data=data, meta=meta)
        env_tid = [int(t) for t in (meta.get("env_terrain_id") or [])]
        for case in meta.get("cases", []):
            match = [i for i, t in enumerate(env_tid) if t == int(case["terrain_id"])]
            if match:
                roll.env_of_case[case["name"]] = match[0]
        rollouts.append(roll)
        if not cases:
            cases = list(meta.get("cases", []))
    if not rollouts:
        raise SystemExit("no usable rollouts in the bundles.")
    return rollouts, cases


# ---------------------------------------------------------------------------
# Depth images
# ---------------------------------------------------------------------------

def _depth_image(flat: np.ndarray, spec: dict) -> np.ndarray:
    """Colour-map one camera's flat depth reading into an RGB image.

    The recorded values are the NORMALISED observation (``(d - offset) * scale``), so the
    colour scale is fixed to the term's own clip band -- frames are then comparable over
    time and across cameras rather than auto-scaling to each frame's content.
    """
    from matplotlib import cm

    h, w = int(spec.get("height", 0)), int(spec.get("width", 0))
    if h * w != flat.size:
        side = int(np.sqrt(flat.size))
        h, w = side, max(1, flat.size // max(side, 1))
    img = np.asarray(flat, dtype=np.float64).reshape(h, w)
    lo, hi = (spec.get("clip") or [-2.0, 2.0])
    norm = np.clip((img - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    rgb = cm.get_cmap("viridis")(np.where(np.isfinite(norm), norm, 0.0))[..., :3]
    rgb[~np.isfinite(img)] = (1.0, 0.0, 1.0)                 # magenta = no reading
    scale = max(1, 220 // max(w, 1))
    rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    return (255.0 * rgb).astype(np.uint8)


# ---------------------------------------------------------------------------
# The scene
# ---------------------------------------------------------------------------

class FmDebugScene:
    """Everything for one (case, rollout-set) view, rebuilt when the case changes."""

    def __init__(self, server, urdf, rollouts: list[Rollout], cases: list[dict],
                 ribbon_radius: float = 0.012) -> None:
        """Build the static GUI + scene nodes; the case-specific ones come from :meth:`rebuild`."""
        self.server = server
        self.urdf = urdf
        self.rollouts = rollouts
        self.cases = cases
        self.ribbon_radius = ribbon_radius
        meta = rollouts[0].meta
        self.joint_names = list(meta.get("joint_names", []))
        self.pos_names = list(meta.get("pos_names", []))
        self.base_frame = str(meta.get("base_frame_name", "pelvis_link"))
        self.perm = recording_to_urdf_perm(self.joint_names, urdf)
        self.base_cols = _base_cols(self.pos_names, self.base_frame)
        self.depth_spec = list(meta.get("depth_camera_spec") or [])

        # One RobotVisual per rollout: the measured body plus that rollout's FM horizon
        # (the curve + fading ghosts it was written for).
        self.robots = {
            r.mode: RobotVisual(server, urdf, env_index=i, joint_perm=self.perm,
                                future_len=int(meta.get("fm_horizon", 1) or 1),
                                max_ghosts=8, num_ghosts=3, point_size=0.018)
            for i, r in enumerate(rollouts)
        }
        for visual in self.robots.values():
            visual.set_overlay_enabled(True)

        # Reference ghosts, one per source per rollout, and the window polylines.
        self.ghosts: dict[tuple[str, str], Any] = {}
        self.lines: dict[str, Any] = {}
        self.points: dict[str, Any] = {}
        for r in self.rollouts:
            for src in ("clip", "fm"):
                self.ghosts[(r.mode, src)] = self._make_ghost(f"/ref/{r.mode}/{src}", _COLORS[src])
            self.lines[f"{r.mode}:window"] = server.scene.add_line_segments(
                f"/window/{r.mode}/base", points=np.zeros((1, 2, 3), np.float32),
                colors=_to_u8(_COLORS["window"]), line_width=4.0, visible=False)
            self.points[f"{r.mode}:window_ee"] = server.scene.add_point_cloud(
                f"/window/{r.mode}/ee", points=np.zeros((1, 3), np.float32),
                colors=_to_u8(_COLORS["window"]), point_size=0.022,
                point_shape="circle", visible=False)

        self.state: dict[str, Any] = {"case": None, "static": []}

    def _make_ghost(self, name: str, color: tuple[float, float, float]):
        """A translucent URDF ghost with its own root frame."""
        from viser.extras import ViserUrdf

        frame = self.server.scene.add_frame(name, show_axes=False, visible=False)
        body = ViserUrdf(self.server, self.urdf, root_node_name=name,
                         mesh_color_override=(*color, _GHOST_ALPHA))
        return frame, body

    # -- static (per case) --------------------------------------------------
    def rebuild(self, case_name: str) -> None:
        """Swap the terrain and the whole-episode paths to ``case_name``."""
        for handle in self.state["static"]:
            handle.remove()
        self.state["static"] = []
        self.state["case"] = case_name
        case = next((c for c in self.cases if c["name"] == case_name), None)
        if case is None:
            return

        meta = self.rollouts[0].meta
        mesh = _terrain_mesh(meta.get("terrain_manifest") or [], int(case["terrain_id"]))
        env0 = self.rollouts[0].env_of_case.get(case_name)
        if mesh is not None and env0 is not None:
            corner = np.asarray((meta.get("env_cell_corner") or [[0, 0, 0]])[env0], dtype=np.float64)
            verts, faces = mesh
            self.state["static"].append(self.server.scene.add_mesh_simple(
                "/terrain", vertices=verts + corner[None, :], faces=faces,
                color=(0.55, 0.55, 0.58)))

        # Whole-episode paths: where the robot went, and where each source's ``y_des`` base
        # said it should be (GLOBAL on the clip path -- see the module docstring).
        for r in self.rollouts:
            env = r.env_of_case.get(case_name)
            if env is None:
                continue
            for key, tag, color, radius in (
                ("root_pos_w", "actual", _COLORS["actual"], self.ribbon_radius * 0.8),
                ("y_des_clip", "clip", _COLORS["clip"], self.ribbon_radius),
                ("y_des_fm", "fm", _COLORS["fm"], self.ribbon_radius),
            ):
                path = self._base_path(r, key, env)
                if path is None or len(path) < 2 or not np.isfinite(path).all():
                    continue
                verts, faces = _tube_mesh(path, radius)
                self.state["static"].append(self.server.scene.add_mesh_simple(
                    f"/paths/{r.mode}/{tag}", vertices=verts, faces=faces, color=color))

    def _base_path(self, roll: Rollout, key: str, env: int) -> np.ndarray | None:
        """The base position over the whole episode, from a world array or a ``y_des``."""
        arr = roll.data.get(key)
        if arr is None:
            return None
        if key == "root_pos_w":
            return np.asarray(arr[:, env, :3], dtype=np.float64)
        if self.base_cols is None:
            return None
        return np.asarray(arr[:, env][:, self.base_cols[0]], dtype=np.float64)

    # -- per frame ----------------------------------------------------------
    def set_frame(self, case_name: str, step: int, layers: dict[str, bool],
                  window_variant: str) -> None:
        """Pose everything for one recorded step."""
        for r in self.rollouts:
            env = r.env_of_case.get(case_name)
            visual = self.robots[r.mode]
            show = layers.get(f"rollout:{r.mode}", True) and env is not None
            visual.set_visible(show)
            for src in ("clip", "fm"):
                frame, _ = self.ghosts[(r.mode, src)]
                frame.visible = False
            self.lines[f"{r.mode}:window"].visible = False
            self.points[f"{r.mode}:window_ee"].visible = False
            if not show:
                continue
            k = min(step, r.num_steps - 1)

            visual.set_state(r.data["root_pos_w"][k, env],
                             r.data["root_quat_w"][k, env],
                             r.data["joint_pos"][k, env])
            visual.set_scan(r.get("scan_hits_w", k, env) if layers.get("scan") else None)

            for src in ("clip", "fm"):
                if layers.get(src):
                    self._pose_ghost(r, src, k, env)

            if layers.get("horizon") and "fm_pred_root_pos" in r.data:
                visual.set_prediction(
                    r.data["root_pos_w"][k, env],
                    r.data["fm_pred_root_pos"][k, env],
                    r.data["fm_pred_root_quat"][k, env],
                    r.data["fm_pred_joint_pos"][k, env],
                )
            else:
                visual.clear_prediction()

            if layers.get("window"):
                self._draw_window(r, k, env, window_variant)

    def _pose_ghost(self, roll: Rollout, src: str, step: int, env: int) -> None:
        """Pose the ``src`` reference ghost at the ANCHORED current-step reference.

        From the look-ahead window's sample 0 rather than ``y_des``: on the clip path
        ``y_des``'s base channels hold the GLOBALLY-placed clip (see the module docstring),
        so a ghost posed from them would sit where the robot *should have been by now*
        rather than on the reference the policy is currently observing. The global path is
        still drawn, as the static tube.
        """
        pos = roll.data.get(f"refwin_{src}_base_pos")
        quat = roll.data.get(f"refwin_{src}_base_quat")
        joints = roll.data.get(f"refwin_{src}_joint_pos")
        if pos is None or quat is None or joints is None:
            return
        p, q, j = pos[step, env, 0], quat[step, env, 0], joints[step, env, 0]
        if not (np.isfinite(p).all() and np.isfinite(q).all() and np.isfinite(j).all()):
            # strict_fm_only NaNs everything the generator does not write; a channel this
            # ghost needs being NaN means it cannot be posed, not that it is at the origin.
            return
        frame, body = self.ghosts[(roll.mode, src)]
        frame.position = np.asarray(p, dtype=np.float32)
        frame.wxyz = xyzw_to_wxyz(np.asarray(q, dtype=np.float32)[None, :])[0]
        body.update_cfg(np.asarray(j, dtype=np.float64)[self.perm])
        frame.visible = True

    def _draw_window(self, roll: Rollout, step: int, env: int, variant: str) -> None:
        """Draw the reference window's base samples as a polyline and its ee samples as dots.

        ``variant`` selects ``clean`` (recomputed from the command) or ``observed``
        (reconstructed from the corrupted observation the policy actually received).
        """
        tag = "obs" if variant == "clean" else "noisy"
        base = roll.data.get(f"refwin_{tag}_base_pos")
        if base is None:
            return
        pts = np.asarray(base[step, env], dtype=np.float32)      # [K, 3]
        if len(pts) < 2 or not np.isfinite(pts).all():
            return
        line = self.lines[f"{roll.mode}:window"]
        line.points = np.stack([pts[:-1], pts[1:]], axis=1)
        line.visible = True

        ee = roll.data.get(f"refwin_{tag}_ee_pos")
        if ee is not None:
            flat = np.asarray(ee[step, env], dtype=np.float32).reshape(-1, 3)
            flat = flat[np.isfinite(flat).all(axis=-1)]
            if flat.size:
                cloud = self.points[f"{roll.mode}:window_ee"]
                cloud.points = flat
                cloud.visible = True


def _to_u8(color: tuple[float, float, float]) -> tuple[int, int, int]:
    """0-1 RGB to viser's 0-255 tuple."""
    return tuple(int(round(255 * c)) for c in color)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def serve(bundles: dict[str, tuple[dict, dict]], urdf_path: str | None = None,
          port: int = 8080) -> int:
    """Serve the recorded rollouts. Blocks until interrupted.

    Args:
        bundles: ``{mode: (data, metadata)}`` from ``fm_debug.load_bundle``.
        urdf_path: G1 URDF; defaults to the repo's 21-joint model.
        port: viser port.

    Returns:
        Process exit code.
    """
    import viser

    rollouts, cases = _index_rollouts(bundles)
    meta = rollouts[0].meta
    dt = float(meta.get("dt") or 0.02)
    num_steps = min(r.num_steps for r in rollouts)
    print(f"[view] {len(rollouts)} rollout(s) {[r.mode for r in rollouts]}, "
          f"{len(cases)} case(s) {[c['name'] for c in cases]}, {num_steps} steps, dt={dt}")

    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("+z")
    urdf = load_urdf(urdf_path or _DEFAULT_URDF)
    scene = FmDebugScene(server, urdf, rollouts, cases)

    gui = server.gui
    case_names = [c["name"] for c in cases] or ["(none)"]
    case_sel = gui.add_dropdown("case", options=case_names, initial_value=case_names[0])
    window_sel = gui.add_dropdown(
        "reference window", options=["clean", "observed"], initial_value="observed")
    step_slider = gui.add_slider("step", min=0, max=max(num_steps - 1, 0), step=1, initial_value=0)
    playing = gui.add_checkbox("play", initial_value=False)
    speed = gui.add_slider("speed x", min=0.1, max=4.0, step=0.1, initial_value=1.0)

    toggles: dict[str, Any] = {}
    with gui.add_folder("layers"):
        toggles["clip"] = gui.add_checkbox("clip reference (ghost)", initial_value=True)
        toggles["fm"] = gui.add_checkbox("FM reference (ghost)", initial_value=True)
        toggles["horizon"] = gui.add_checkbox("FM horizon", initial_value=True)
        toggles["window"] = gui.add_checkbox("reference window (observed)", initial_value=True)
        toggles["scan"] = gui.add_checkbox("height-scan dots", initial_value=False)
        for r in rollouts:
            toggles[f"rollout:{r.mode}"] = gui.add_checkbox(
                f"rollout: {r.mode}-driven", initial_value=True)
    ghosts = gui.add_slider("horizon ghosts", min=0, max=8, step=1, initial_value=3)
    readout = gui.add_markdown("")

    depth_views: dict[str, Any] = {}
    if scene.depth_spec and any("depth_raw" in r.data for r in rollouts):
        with gui.add_folder("depth cameras"):
            for i, spec in enumerate(scene.depth_spec):
                depth_views[str(i)] = gui.add_image(
                    np.zeros((64, 64, 3), np.uint8),
                    label=f"{spec.get('obs_term', f'camera {i}')} (magenta = no reading)")

    def _layers() -> dict[str, bool]:
        """Current layer visibility, as the scene wants it."""
        return {k: bool(v.value) for k, v in toggles.items()}

    def _refresh(rebuild: bool = False) -> None:
        """Re-pose everything for the current case + step."""
        case = str(case_sel.value)
        if rebuild or scene.state["case"] != case:
            scene.rebuild(case)
        step = int(step_slider.value)
        for visual in scene.robots.values():
            visual.set_num_ghosts(int(ghosts.value))
        scene.set_frame(case, step, _layers(), str(window_sel.value))
        _update_readout(readout, depth_views, scene, rollouts, case, step, dt)

    for handle in (case_sel, window_sel, step_slider, ghosts, *toggles.values()):
        handle.on_update(lambda _evt: _refresh())

    _refresh(rebuild=True)
    # The BOUND port, not the requested one: viser silently increments past a port that is
    # already taken, so printing --port would send you to somebody else's server.
    bound = server.get_port()
    if bound != port:
        print(f"[view] port {port} was busy; viser took {bound} instead.")
    print(f"[view] viser on http://localhost:{bound}")
    try:
        while True:
            if playing.value:
                nxt = int(step_slider.value) + 1
                step_slider.value = 0 if nxt > num_steps - 1 else nxt
            time.sleep(dt / max(float(speed.value), 1e-3))
    except KeyboardInterrupt:
        print("\n[view] stopped.")
    return 0


def _update_readout(readout, depth_views, scene: FmDebugScene, rollouts: list[Rollout],
                    case: str, step: int, dt: float) -> None:
    """Write the per-step numbers next to the 3-D view, and refresh the depth images.

    The numbers are the ones you would otherwise go back to the plots for: how far each
    source's ANCHORED reference is from the robot right now, and how far the two sources
    are from each other. All three read the window's sample 0, so ``fm - clip`` is a
    like-for-like reference difference and not the accumulated global drift that
    ``y_des``'s base channels would report on the clip path.
    """
    lines = [f"**case {case}** step {step}  t={step * dt:.2f} s"]
    for r in rollouts:
        env = r.env_of_case.get(case)
        if env is None:
            continue
        k = min(step, r.num_steps - 1)
        root = r.data["root_pos_w"][k, env]
        parts = [f"env {env}"]
        anchored = {}
        for src in ("clip", "fm"):
            arr = r.data.get(f"refwin_{src}_base_pos")
            if arr is None or not np.isfinite(arr[k, env, 0]).all():
                continue
            anchored[src] = arr[k, env, 0]
            parts.append(f"|{src} - robot| = {np.linalg.norm(anchored[src] - root):.3f} m")
        if len(anchored) == 2:
            parts.append(f"|fm - clip| = {np.linalg.norm(anchored['fm'] - anchored['clip']):.3f} m")
        done = r.data.get("dones")
        if done is not None and bool(done[:k + 1, env].any()):
            parts.append("**episode already ended**")
        lines.append(f"- `{r.mode}`: " + ",  ".join(parts))

        if depth_views and "depth_raw" in r.data and r.mode == rollouts[0].mode:
            depth = r.data["depth_raw"][k, env]
            for i, spec in enumerate(scene.depth_spec):
                if str(i) in depth_views and i < depth.shape[0]:
                    depth_views[str(i)].image = _depth_image(depth[i], spec)
    readout.content = "  \n".join(lines)
