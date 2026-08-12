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

import os
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
import fm_resample as _fm_resample
from view_clip_viser import _DEFAULT_URDF, _terrain_mesh, _tube_mesh

#: Per-foot colours for the draws' FK'd foot paths, in ``fm_frame_names`` order (left, right).
_FOOT_COLORS: tuple[tuple[float, float, float], ...] = (
    (0.20, 0.65, 1.00),   # blue  -- left foot
    (1.00, 0.35, 0.35),   # red   -- right foot
)

#: Left/right foot colours for the CLIP's horizon path -- warm, so they read as the clip
#: family next to the FM feet's blue/red rather than as two more draws.
_CLIP_FOOT_COLORS: tuple[tuple[float, float, float], ...] = (
    (1.00, 0.78, 0.25),   # amber       -- left foot
    (0.90, 0.45, 0.08),   # dark orange -- right foot
)

#: Base line widths for the draw paths, scaled live by the viewer's "line width x" slider.
#: The feet are drawn THICKER than the base ribbon: they are the part being read against the
#: tread edges, and at F=62 nodes a thin polyline disappears against terrain.
_DRAW_WIDTH = {"base_chosen": 5.0, "base_reject": 2.0,
               "foot_chosen": 8.0, "foot_reject": 3.0}

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
        # The generator's own conditioning camera contract, which is NOT `depth_camera_spec`
        # when it samples the pair on its own grid -- and it is the one whose images actually
        # conditioned the prediction being drawn.
        self.fm_extero_spec = list(meta.get("fm_extero_spec") or [])
        self.meta = meta
        # Draw ribbons are rebuilt per frame (their count changes with the slider), so they
        # are tracked for removal rather than reused like the fixed-purpose nodes above.
        self._draw_nodes: list[Any] = []
        # FK for the draws' feet: the window predicts joints, not ee positions, so a foot
        # path has to be derived. Built once (the parse is the slow part) and only when the
        # bundle actually carries draws.
        self.fm_frame_names = list(meta.get("fm_frame_names", []))
        self._clip_horizon_nodes: list[Any] = []
        self._trail_nodes: list[Any] = []
        # ONE assembly per rollout, shared with the keyboard browser
        # (``fm_feet_browser.py``): it owns the FK chain, the y_des column resolution, the
        # served-trace cache and the terrain profile. Building those here as well would mean
        # two FK parses and, worse, two implementations that can disagree about which array is
        # the history.
        from fm_feet_data import FeetPlotSource
        self._feet_src = {
            r.mode: FeetPlotSource(r.data, r.meta, mode=r.mode, depth_image_fn=_depth_image)
            for r in rollouts
        }
        first = next(iter(self._feet_src.values()))
        self._draw_fk = first.fk
        # Which of the ee frames are FEET -- the wrists are in the same block and would
        # quadruple the line count for nothing here.
        self._foot_slots = first.foot_slots
        # Columns of y_des_clip for the clip's own path. The FM's feet come from FK, but the
        # clip states its feet directly, so these are read not derived.
        self._clip_cols = {"base": first.cols["base"], "feet": first.cols["feet"]}

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

    def set_draws(self, roll: Rollout, step: int, env: int, num_shown: int,
                  show_feet: bool = True, width_scale: float = 1.0) -> None:
        """Draw the alternative windows this re-plan sampled -- chosen bright, rejects dim.

        Only meaningful on a bundle captured with ``--num_draws > 1``; a no-op otherwise. The
        rejects are what the oracle had to choose between, so their spread IS the headroom any
        selection rule could exploit: if they lie on top of each other, no rule helps.

        Sorted by score so ``num_shown`` keeps the best, and every draw of a re-plan shares the
        plan anchor, so the fan-out is a genuine comparison at one robot pose.
        """
        for handle in self._draw_nodes:
            handle.remove()
        self._draw_nodes = []
        draws = roll.get("fm_draws", step, env)
        if draws is None or num_shown <= 0:
            return
        scores = roll.get("fm_scores", step, env)
        chosen = roll.get("fm_chosen", step, env)
        order = np.argsort(np.asarray(scores)) if scores is not None else np.arange(len(draws))
        keep = list(order[:max(1, int(num_shown))])
        try:
            paths = _fm_resample.draws_to_world(
                np.asarray(draws)[keep], roll.get("fm_cond_anchor_pos", step, env),
                roll.get("fm_cond_anchor_quat", step, env),
                len(self.joint_names) or 29, len(self.fm_frame_names) or 4,
                blocks=self.meta.get("fm_reference_blocks"),
                fk=self._draw_fk if show_feet else None)
        except Exception as exc:  # a bundle whose layout cannot be inferred
            print(f"[view] could not lift the draws to world: {exc}")
            return
        feet = paths.get("ee")
        scale = max(float(width_scale), 0.1)
        for rank, (j, path) in enumerate(zip(keep, paths["base"])):
            is_chosen = chosen is not None and int(j) == int(chosen)
            key = "chosen" if is_chosen else "reject"
            self._add_path(f"/draws/{rank}", path,
                           (0.05, 1.0, 0.35) if is_chosen else (0.55, 0.55, 0.62),
                           _DRAW_WIDTH[f"base_{key}"] * scale)
            if feet is None:
                continue
            # One line per foot per drawn window. Only the chosen draw's feet get their own
            # colours; the rejects' stay grey, or 16 draws x 2 feet reads as noise.
            for slot_i, slot in enumerate(self._foot_slots):
                color = (_FOOT_COLORS[slot_i % len(_FOOT_COLORS)] if is_chosen
                         else (0.45, 0.45, 0.50))
                # A SIBLING of the base path, not a child: removing a parent takes its
                # subtree with it, so a nested name is already gone by the time the loop
                # reaches it and viser warns about the double removal.
                self._add_path(f"/draws/{rank}_foot{slot}", feet[rank, :, slot], color,
                               _DRAW_WIDTH[f"foot_{key}"] * scale)

    def terrain_profile_world(self, case_name: str) -> list[tuple[float, float, float]]:
        """The case tile's top-surface step profile as world ``[(x0, x1, top)]``.

        Overlay boxes are tile-local and the placement is a pure translation onto the tile
        corner (``clip_frame_origin``), so world x is just ``corner_x + local_x``. Projected to
        x-z: every box contributes its own span, which is what makes a tread flush on a
        platform show as two segments at the same height rather than being lost.
        """
        env = next((r.env_of_case.get(case_name) for r in self.rollouts
                    if r.env_of_case.get(case_name) is not None), None)
        if env is None:
            return []
        tids = self.meta.get("env_terrain_id") or []
        corners = self.meta.get("env_cell_corner") or []
        if env >= len(tids) or env >= len(corners):
            return []
        tid = int(np.asarray(tids).reshape(-1)[env])
        corner = np.asarray(corners)[env]
        entry = next((m for m in (self.meta.get("terrain_manifest") or [])
                      if int(m.get("terrain_id", -999)) == tid), None)
        if entry is None or "overlays" not in entry:
            return []
        segs = []
        for o in entry["overlays"]:
            cx, sx = float(o["pos_xy"][0]), float(o["size_xyz"][0])
            top = float(o["base_z"]) + float(o["size_xyz"][2])
            segs.append((float(corner[0]) + cx - sx / 2.0,
                         float(corner[0]) + cx + sx / 2.0,
                         float(corner[2]) + top))
        return sorted(segs)

    def feet_plot_inputs(self, roll: Rollout, step: int, env: int, case_name: str,
                         tail_steps: int = 50) -> dict | None:
        """Inputs for the x-z figure, via the shared assembly.

        Delegates to :class:`fm_feet_data.FeetPlotSource` so the viser button and the keyboard
        browser (``fm_feet_browser.py``) draw from ONE implementation -- two copies of this
        reasoning is how their pictures would come to disagree about which array is the history.
        """
        src = self._feet_src.get(roll.mode)
        if src is None:
            return None
        row = src.inputs(step, env, tail_steps=tail_steps)
        if row is not None:
            row["row_label"] = f"{roll.mode} / history={src.history_source}"
        return row

    def served_trace(self, roll: Rollout, env: int) -> dict[str, np.ndarray] | None:
        """Root + feet of the reference the robot ACTUALLY tracked, per step.

        Delegates to the shared assembly (see ``self._feet_src``), which reads it from
        ``y_des_fm`` -- the applied desired output, already in world -- and FKs the feet
        because the window predicts no ``ee_pos``.
        """
        src = self._feet_src.get(roll.mode)
        return None if src is None else src.served_trace(env)

    def set_served_trail(self, roll: Rollout, step: int, env: int, show_feet: bool = True,
                         width_scale: float = 1.0) -> None:
        """Trail the served reference from the start of the rollout up to ``step``."""
        for handle in self._trail_nodes:
            handle.remove()
        self._trail_nodes = []
        trace = self.served_trace(roll, env) if step > 0 else None
        if trace is None:
            return
        scale = max(float(width_scale), 0.1)
        end = min(step + 1, trace["base"].shape[0])
        self._add_path("/served/base", trace["base"][:end], _COLORS["fm"],
                       _DRAW_WIDTH["base_chosen"] * scale, self._trail_nodes)
        if not show_feet or "feet" not in trace:
            return
        for i in range(trace["feet"].shape[1]):
            pts = trace["feet"][:end, i]
            if np.isfinite(pts).all():
                self._add_path(f"/served/foot{i}", pts,
                               _FOOT_COLORS[i % len(_FOOT_COLORS)],
                               _DRAW_WIDTH["foot_chosen"] * scale, self._trail_nodes)

    def set_clip_horizon(self, roll: Rollout, step: int, env: int, horizon: int,
                         show_feet: bool = True, width_scale: float = 1.0) -> None:
        """Draw the PRE-MADE clip over the same nodes the FM draws span.

        The comparison target the oracle scored against, and the thing the draws should be
        read against: node *k* of a prediction made at step *t* corresponds to clip step
        *t + k*, because the clip advances one step per control step. So the clip's own path
        over that horizon is just ``y_des_clip[t : t+F]`` -- no re-anchoring, no window
        round-trip, and already in world coordinates (verified identical to
        ``refwin_clip_base_pos`` sample 0 to 0.0000 m).

        Distinct from the two clip layers that already exist: the clip GHOST is a posed robot
        at the current step, and the clip TUBE is the whole episode. This is the horizon
        slice, which is what makes it comparable to a draw.
        """
        for handle in self._clip_horizon_nodes:
            handle.remove()
        self._clip_horizon_nodes = []
        arr = roll.data.get("y_des_clip")
        if arr is None or horizon <= 1 or not self._clip_cols["base"]:
            return
        full = np.asarray(arr[:, env])                                # [T, P]
        scale = max(float(width_scale), 0.1)
        cut = int(np.clip(step, 1, full.shape[0] - 1))
        far = int(min(cut + int(horizon), full.shape[0]))
        # Three tiers of the SAME curve, since a line's colour is per-segment and there is no
        # per-vertex alpha: what is already behind the robot, the F nodes a draw covers (the
        # comparison window), and the remainder of the clip.
        tiers = (("past", slice(0, cut + 1), 0.30, 0.7),
                 ("horizon", slice(cut, far), 1.00, 1.0),
                 ("ahead", slice(max(far - 1, 0), full.shape[0]), 0.55, 0.7))
        curves = [("base", self._clip_cols["base"], _COLORS["clip"],
                   _DRAW_WIDTH["base_chosen"])]
        if show_feet:
            curves += [(f"foot{i}", cols, _CLIP_FOOT_COLORS[i % len(_CLIP_FOOT_COLORS)],
                        _DRAW_WIDTH["foot_chosen"])
                       for i, cols in enumerate(self._clip_cols["feet"])]
        for cname, cols, color, base_w in curves:
            for tname, sl, dim, wmul in tiers:
                pts = full[sl, :][:, cols]
                if pts.shape[0] < 2 or not np.isfinite(pts).all():
                    continue
                # Fading toward the background rather than toward black: the terrain is dark,
                # so a scaled-to-black "past" would read as a solid line, not a faded one.
                faded = tuple(c * dim + (1.0 - dim) * 0.14 for c in color)
                self._add_path(f"/clip_path/{cname}_{tname}", pts, faded,
                               base_w * wmul * scale, self._clip_horizon_nodes)

    def _add_path(self, name: str, points: np.ndarray, color: tuple[float, float, float],
                  width: float, into: list | None = None) -> None:
        """Add one polyline through ``points`` and remember it for removal."""
        pts = np.asarray(points, dtype=np.float32)
        if pts.shape[0] < 2:
            return
        seg = np.stack([pts[:-1], pts[1:]], axis=1)
        (self._draw_nodes if into is None else into).append(self.server.scene.add_line_segments(
            name, points=seg,
            colors=np.tile(np.array(color, np.float32), (seg.shape[0], 2, 1)),
            line_width=width))

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
          port: int = 8080, out_dir: str = ".") -> int:
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
        # Kept out of `toggles` because that dict is handed to set_frame as the scene's layer
        # map; this one drives set_clip_horizon instead.
        clip_horizon_cb = gui.add_checkbox("clip path (root + feet, past faded)",
                                           initial_value=True)
        served_trail_cb = gui.add_checkbox("served FM trail (root + feet)",
                                          initial_value=True)
        for r in rollouts:
            toggles[f"rollout:{r.mode}"] = gui.add_checkbox(
                f"rollout: {r.mode}-driven", initial_value=True)
    ghosts = gui.add_slider("horizon ghosts", min=0, max=8, step=1, initial_value=3)

    # --- Multi-sample navigation. A re-plan is where the generator actually inferred; the
    # steps between it serve that one plan, so stepping frame-by-frame mostly shows the robot
    # walking through a fixed prediction. These jump between the inferences themselves.
    n_draws = 0
    for r in rollouts:
        arr = r.data.get("fm_draws")
        if arr is not None:
            n_draws = max(n_draws, int(np.asarray(arr).shape[2]))
    draws_shown = draw_feet = None
    if n_draws:
        with gui.add_folder(f"draws ({n_draws} per re-plan)"):
            draws_shown = gui.add_slider("show best N", min=0, max=n_draws, step=1,
                                         initial_value=min(4, n_draws))
            draw_feet = gui.add_checkbox("feet (FK of predicted joints)", initial_value=True)
            draw_width = gui.add_slider("line width x", min=0.5, max=4.0, step=0.1,
                                        initial_value=1.0)
            prev_replan = gui.add_button("<- re-plan")
            next_replan = gui.add_button("re-plan ->")
            feet_plot_btn = gui.add_button("plot feet x-z here")
    readout = gui.add_markdown("")

    # Horizon the clip slice spans: the generator's own F, so the clip path covers exactly
    # the nodes a draw does.
    horizon = int(meta.get("fm_horizon") or 0)
    if not horizon:
        for r in rollouts:
            arr = r.data.get("fm_pred_window")
            if arr is not None:
                horizon = int(np.asarray(arr).shape[2])
                break

    def _replans() -> np.ndarray:
        """Re-plan steps of the first rollout that recorded the conditioning."""
        for r in rollouts:
            if "fm_cond_hist_nodes" in r.data:
                try:
                    return _fm_resample.replan_steps(r.data, 0)
                except SystemExit:
                    continue
        return np.arange(num_steps)

    depth_views: dict[str, Any] = {}
    if scene.depth_spec and any("depth_raw" in r.data for r in rollouts):
        with gui.add_folder("depth cameras (policy)"):
            for i, spec in enumerate(scene.depth_spec):
                depth_views[str(i)] = gui.add_image(
                    np.zeros((64, 64, 3), np.uint8),
                    label=f"{spec.get('obs_term', f'camera {i}')} (magenta = no reading)")
    # What the GENERATOR was conditioned on. Same cameras as above whenever it shares the
    # policy's pair, a different sampling when it does not -- and only this one is the input
    # to the prediction on screen.
    fm_depth_views: dict[str, Any] = {}
    if scene.fm_extero_spec and any("fm_cond_extero" in r.data for r in rollouts):
        with gui.add_folder("depth cameras (generator input)"):
            for i, spec in enumerate(scene.fm_extero_spec):
                fm_depth_views[str(i)] = gui.add_image(
                    np.zeros((64, 64, 3), np.uint8),
                    label=f"FM {spec.get('obs_term', f'camera {i}')} "
                          f"{spec.get('width')}x{spec.get('height')}")

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
        width = float(draw_width.value) if draws_shown is not None else 1.0
        feet_on = bool(draw_feet.value) if draws_shown is not None else True
        if draws_shown is not None:
            for r in rollouts:
                env = r.env_of_case.get(case)
                if "fm_draws" in r.data and env is not None:
                    scene.set_draws(r, step, env, int(draws_shown.value),
                                    show_feet=feet_on, width_scale=width)
                    break
        # The whole clip with the traversed part faded, and the trail of FM output actually
        # served -- together they show where the two have parted company so far.
        for r in rollouts:
            env = r.env_of_case.get(case)
            if env is None:
                continue
            scene.set_clip_horizon(r, step, env, horizon if clip_horizon_cb.value else 0,
                                   show_feet=feet_on, width_scale=width)
            scene.set_served_trail(r, step if served_trail_cb.value else 0, env,
                                   show_feet=feet_on, width_scale=width)
            break
        _update_readout(readout, depth_views, scene, rollouts, case, step, dt,
                        fm_depth_views=fm_depth_views)

    def _jump(delta: int) -> None:
        """Move to the previous/next re-plan step."""
        marks = _replans()
        here = int(step_slider.value)
        ahead = marks[marks > here]
        behind = marks[marks < here]
        if delta > 0 and ahead.size:
            step_slider.value = int(ahead[0])
        elif delta < 0 and behind.size:
            step_slider.value = int(behind[-1])

    feet_plot_view: dict[str, Any] = {}

    def _feet_plot(_evt=None) -> None:
        """Render the x-z feet figure for the current step into a panel and a PNG."""
        import fm_feet_plot

        case = str(case_sel.value)
        step = int(step_slider.value)
        for r in rollouts:
            env = r.env_of_case.get(case)
            if env is None or "fm_draws" not in r.data:
                continue
            kw = scene.feet_plot_inputs(r, step, env, case)
            if kw is None:
                print("[view] no draws at this step; nothing to plot.")
                return
            fig = fm_feet_plot.feet_xz_figure(**kw, title_extra=f"  |  {case} / {r.mode}")
            if "panel" not in feet_plot_view:
                with gui.add_folder("feet x-z"):
                    feet_plot_view["panel"] = gui.add_image(
                        fm_feet_plot.figure_to_rgb(fig), label="feet x-z (latest)")
            else:
                feet_plot_view["panel"].image = fm_feet_plot.figure_to_rgb(fig)
            path = os.path.join(out_dir, f"feet_xz_{r.mode}_{case}_step{step:04d}.png")
            fig.savefig(path, dpi=140, bbox_inches="tight")
            import matplotlib.pyplot as _plt
            _plt.close(fig)
            print(f"[view] wrote {path}")
            return
        print("[view] this bundle carries no draws (capture with --num_draws > 1).")

    handles = [case_sel, window_sel, step_slider, ghosts, clip_horizon_cb, served_trail_cb,
               *toggles.values()]
    if draws_shown is not None:
        handles += [draws_shown, draw_feet, draw_width]
        next_replan.on_click(lambda _evt: _jump(+1))
        prev_replan.on_click(lambda _evt: _jump(-1))
        feet_plot_btn.on_click(_feet_plot)
    for handle in handles:
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
                    case: str, step: int, dt: float, fm_depth_views=None) -> None:
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
        if fm_depth_views and "fm_cond_extero" in r.data and r.mode == rollouts[0].mode:
            fm_depth = r.data["fm_cond_extero"][k, env]
            for i, spec in enumerate(scene.fm_extero_spec):
                if str(i) in fm_depth_views and i < fm_depth.shape[0]:
                    fm_depth_views[str(i)].image = _depth_image(fm_depth[i], spec)

        # Best-of-N: the spread across draws is what bounds any selection rule, and the
        # chosen draw's own score says whether a good window was available at all.
        scores = r.data.get("fm_scores")
        if scores is not None:
            s = np.asarray(scores[k, env])
            chosen = int(np.asarray(r.data["fm_chosen"][k, env]))
            lines.append(f"  - draws: chose #{chosen} of {s.size}, score {s[chosen]:.4f}  "
                         f"(best {s.min():.4f}, worst {s.max():.4f}, "
                         f"spread {s.max() - s.min():.4f})")
    readout.content = "  \n".join(lines)
