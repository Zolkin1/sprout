#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.
"""Offscreen MuJoCo rendering for the paper / website videos.

The rendering half of :mod:`record_videos`, kept separate because the two halves have
different failure modes and different iteration speeds.  Rolling an episode is physics and
ONNX and costs a minute; rendering it is pixels and costs seconds, and the whole point of
splitting them is that the look can be re-tuned as many times as it takes **without ever
re-running the policy** -- so the footage keeps describing the episode that was measured.

**Nothing here touches physics.**  A trace is a table of ``qpos`` rows that already happened;
this module loads a SECOND copy of the scene, restyles it freely, writes those rows in and
calls ``mj_forward``.  That is why the themes may recolour, fade or relight anything they
like: no number downstream depends on how the frame looked.

Three things the look needs and the authored scenes do not give:

* **A theme.**  The generated scenes all share one ``<visual>`` + ``<asset>`` block (they come
  out of ``generate_stairs_scene.scene_xml``), which is a black skybox over a dark blue
  checker -- legible in a debug viewer and murky on a projector.  :func:`themed_scene`
  swaps that one block for another, by exact string replacement, and raises if the block is
  not the one it expects rather than silently rendering the old look.
* **Walls you can see through.**  ``varied_updown`` tiles 13 and 23 are 1.75 m-wide corridors
  between two SOLID 1.77 m walls -- the robot is completely hidden from any side view.
  :func:`fade_walls` finds them GEOMETRICALLY (tall, thin, world-fixed) rather than by name,
  because on a tile scene they are called ``tile_slab11`` / ``tile_slab12`` and nothing in the
  name says wall.
* **A velocity panel.**  :class:`VelocityPanel` draws commanded against achieved with the
  feature spans shaded, which is the paper's "tracking is accurate except across a
  traversal" claim as a picture.  The shading comes from the compiled model's own geometry
  (:func:`terrain_feature_spans`), so it cannot disagree with the terrain on screen.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field

import mujoco
import numpy as np

# --------------------------------------------------------------------------------------------
# Themes
# --------------------------------------------------------------------------------------------

#: The block every generated scene shares, verbatim, from ``generate_stairs_scene.scene_xml``.
#: Matched as a REGION between two anchors rather than compared byte for byte, so a scene that
#: adds a line inside it still themes -- but the anchors must both be present, and a scene
#: missing them is refused instead of being rendered in the old look.
_BLOCK_OPEN = "    <visual>"
_BLOCK_CLOSE = "    </asset>"

#: ``asis`` is spelled out rather than left as "do nothing" so that every theme goes through
#: the same substitution and a themed scene is always a file this module wrote -- which keeps
#: the committed scenes read-only on the video path.
#:
#: Every theme raises ``offwidth``/``offheight`` to 3840x2160.  The authored scenes declare
#: 1920x1080, which is the OFFSCREEN FRAMEBUFFER and therefore a hard ceiling on the render:
#: :func:`render_trace` refuses a frame larger than it.  It costs nothing until a frame that
#: size is actually asked for.
THEMES: dict[str, str] = {
    "asis": """    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0.9 0.9 0.9"/>
        <rgba haze="0.15 0.25 0.35 1"/>
        <global azimuth="-140" elevation="-20" offwidth="3840" offheight="2160"/>
    </visual>

    <asset>
        <texture type="skybox" builtin="flat" rgb1="0 0 0" rgb2="0 0 0" width="512" height="3072"/>
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
        <material name="box_material" shininess="0.1" specular="0.5" roughness="0.3" rgba="0.75 0.6 0.42 1"/>
    </asset>""",
    # Daylight studio: pale sky, low-contrast grey floor grid, terrain in a desaturated slate
    # blue that separates from both the floor and the G1's own white/black shells.
    "bright": """    <visual>
        <headlight diffuse="0.42 0.42 0.44" ambient="0.42 0.43 0.45" specular="0.18 0.18 0.18"/>
        <rgba haze="0.90 0.93 0.96 1"/>
        <map znear="0.02" zfar="80" shadowclip="4" shadowscale="1.0" haze="0.15"/>
        <quality shadowsize="8192" offsamples="8"/>
        <global azimuth="20" elevation="-15" offwidth="3840" offheight="2160"/>
    </visual>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.60 0.72 0.84" rgb2="0.95 0.96 0.97" width="512" height="3072"/>
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.80 0.82 0.84" rgb2="0.73 0.76 0.79" markrgb="0.95 0.96 0.97" width="300" height="300"/>
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="4 4" reflectance="0.04"/>
        <material name="box_material" shininess="0.30" specular="0.28" roughness="0.55" rgba="0.36 0.50 0.64 1"/>
    </asset>""",
    # No floor texture at all: the checker reads as motion when the camera dollies, which on a
    # 15 m lane is most of the clip.  Warmer terrain, for the shots where the robot is dark.
    "studio": """    <visual>
        <headlight diffuse="0.40 0.40 0.42" ambient="0.46 0.46 0.48" specular="0.20 0.20 0.20"/>
        <rgba haze="0.93 0.93 0.94 1"/>
        <map znear="0.02" zfar="80" shadowclip="4" shadowscale="1.0" haze="0.10"/>
        <quality shadowsize="8192" offsamples="8"/>
        <global azimuth="20" elevation="-15" offwidth="3840" offheight="2160"/>
    </visual>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.78 0.82 0.86" rgb2="0.97 0.97 0.98" width="512" height="3072"/>
        <texture type="2d" name="groundplane" builtin="flat" rgb1="0.86 0.87 0.88" rgb2="0.86 0.87 0.88" width="8" height="8"/>
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="1 1" reflectance="0.06"/>
        <material name="box_material" shininess="0.25" specular="0.30" roughness="0.55" rgba="0.80 0.58 0.36 1"/>
    </asset>""",
}


def themed_scene(src_xml: str, dst_xml: str, theme: str) -> str:
    """Write a restyled copy of a scene and return its path.

    Args:
        src_xml: The scene the episode was ROLLED on.  Read only.
        dst_xml: Where to write the restyled copy.
        theme: A key of :data:`THEMES`.

    Returns:
        ``dst_xml``.

    Raises:
        KeyError: Unknown theme.
        ValueError: If the source scene does not carry the shared ``<visual>``/``<asset>``
            block.  Refused rather than passed through: a scene that themes silently to
            nothing would render in the old look and look like the theme did not work.
    """
    block = THEMES[theme]
    with open(src_xml, encoding="utf-8") as fh:
        xml = fh.read()
    i = xml.find(_BLOCK_OPEN)
    j = xml.find(_BLOCK_CLOSE)
    if i < 0 or j < 0 or j < i:
        raise ValueError(
            f"{src_xml} carries no '{_BLOCK_OPEN.strip()} ... {_BLOCK_CLOSE.strip()}' region, "
            "so there is nothing for the theme to replace. Every scene emitted by "
            "generate_stairs_scene.scene_xml has one.")
    out = xml[:i] + block + xml[j + len(_BLOCK_CLOSE):]
    # The include is relative to the scene file, so a themed copy has to sit beside the
    # original; `record_videos` puts it in robots/g1/ for exactly that reason.
    os.makedirs(os.path.dirname(dst_xml), exist_ok=True)
    with open(dst_xml, "w", encoding="utf-8") as fh:
        fh.write(out)
    return dst_xml


# --------------------------------------------------------------------------------------------
# Per-geom restyling (mjModel edits, after load)
# --------------------------------------------------------------------------------------------

#: A geom is WALL-LIKE when it stands this far above the scene's lowest surface ...
WALL_MIN_TOP_M: float = 1.10
#: ... its smaller horizontal half-extent is under this, i.e. it is a slab, not a platform ...
WALL_MAX_THIN_M: float = 0.25
#: ... and it is at least this many times longer than it is thick.
#:
#: **The aspect test is what keeps a long flight opaque.**  Height and thinness alone caught
#: the upper treads of ``terrain_scene``'s 20-riser ``tall`` staircase -- a 0.32 m tread IS
#: thin, and past riser 8 it IS more than 1.10 m up -- and rendered the top half of the
#: flight see-through.  A tread is 4.8x longer than thick; the stairwell walls it has to be
#: told apart from are 26-53x.
WALL_MIN_ASPECT: float = 8.0


#: Per-theme ``(diffuse, specular)`` for the scene's own lights.  These live in
#: ``<worldbody>``, not in the block :func:`themed_scene` replaces, and MuJoCo's default
#: directional light is bright enough that -- stacked on the headlight -- every UP-facing
#: face saturates: a 0.70 grey walkway rendered as white and the horizon disappeared.  Set
#: here, after load, for the same reason the geom colours are.
LIGHTS: dict[str, tuple[float, float]] = {
    "asis": (0.7, 0.3),
    "bright": (0.42, 0.10),
    "studio": (0.38, 0.08),
}


def apply_lighting(model, theme: str) -> None:
    """Set every scene light's diffuse and specular for a theme.

    Args:
        model: The compiled render model, edited in place.
        theme: A key of :data:`LIGHTS`.
    """
    diff, spec = LIGHTS[theme]
    for i in range(model.nlight):
        model.light_diffuse[i] = (diff, diff, diff)
        model.light_specular[i] = (spec, spec, spec)


def world_boxes(model) -> list[int]:
    """Ids of the world-fixed box geoms, i.e. the terrain and its decoration."""
    return [i for i in range(model.ngeom)
            if int(model.geom_bodyid[i]) == 0
            and int(model.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_BOX)]


def scene_floor_z(model) -> float:
    """The lowest world surface in the scene (m).

    The datum every other height in this module is measured from.  Takes each world-fixed
    geom's TOP face, which for the ground plane is its own ``pos_z`` and for a ground SLAB is
    its upper surface -- the two conventions the tile scenes mix, since a tile that authors
    its own footprint gets no plane at all.

    Args:
        model: The compiled render model.

    Returns:
        World z of the lowest surface (m), or 0.0 in a scene with no world geometry.
    """
    tops = []
    for i in range(model.ngeom):
        if int(model.geom_bodyid[i]) != 0:
            continue
        if int(model.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            tops.append(float(model.geom_pos[i][2]))
        else:
            tops.append(float(model.geom_pos[i][2] + model.geom_size[i][2]))
    return float(min(tops)) if tops else 0.0


def wall_geoms(model) -> list[int]:
    """Ids of the wall-like world geoms: tall, thin and world-fixed.

    Found by GEOMETRY, not by name.  On an ``isaac_tile_course`` the stairwell walls are
    emitted as ``tile_slab11`` / ``tile_slab12`` -- nothing in the name says wall, and a
    name-matched rule would miss them and leave the robot invisible behind a solid slab.

    Tall, thin AND long: see :data:`WALL_MIN_ASPECT` for the tread that the first two alone
    mistook for a wall.

    Args:
        model: The compiled render model.

    Returns:
        Geom ids, possibly empty.
    """
    floor_z = scene_floor_z(model)
    out = []
    for i in world_boxes(model):
        top = float(model.geom_pos[i][2] + model.geom_size[i][2])
        hx, hy = float(model.geom_size[i][0]), float(model.geom_size[i][1])
        thin, long_ = min(hx, hy), max(hx, hy)
        if (top - floor_z >= WALL_MIN_TOP_M and thin <= WALL_MAX_THIN_M
                and long_ >= WALL_MIN_ASPECT * thin):
            out.append(i)
    return out


def _detach_material(model, gid: int) -> None:
    """Point a geom at its own ``geom_rgba`` instead of its material.

    MuJoCo takes the colour from the MATERIAL whenever one is assigned, so writing
    ``geom_rgba`` without this is a no-op that looks like the colour ramp silently failed.
    """
    model.geom_matid[gid] = -1


def fade_walls(model, alpha: float) -> int:
    """Make the wall-like geoms translucent so the robot inside a corridor is visible.

    Args:
        model: The compiled render model, edited in place.
        alpha: Opacity in ``[0, 1]``; 1.0 leaves them alone, 0.0 hides them.

    Returns:
        How many geoms were faded.
    """
    ids = wall_geoms(model)
    if alpha >= 1.0:
        return 0
    for gid in ids:
        rgba = np.array(model.mat_rgba[int(model.geom_matid[gid])]
                        if int(model.geom_matid[gid]) >= 0 else model.geom_rgba[gid])
        _detach_material(model, gid)
        model.geom_rgba[gid] = (*rgba[:3], float(alpha))
    return len(ids)


#: Light -> saturated ramp for :func:`shade_terrain`, in the order a feature rises.  It does
#: NOT start at white: the lowest riser sat within a few percent of the pale ground and the
#: first step of a flight was invisible, which is the one this stack's failures happen on.
_HEIGHT_RAMP: tuple[tuple[float, float, float], ...] = (
    (0.66, 0.74, 0.81), (0.47, 0.61, 0.75), (0.31, 0.48, 0.66),
    (0.19, 0.35, 0.54), (0.11, 0.23, 0.39),
)

#: Ground slabs get their own neutral grey rather than the bottom of the ramp.  A tile that
#: authors its own footprint has NO ground plane -- the slab IS the floor -- and shading it as
#: "the shortest feature" put a near-white walkway under a pale sky with no horizon between
#: them.
_GROUND_RGB: tuple[float, float, float] = (0.70, 0.72, 0.745)

#: A world geom whose top is within this of the scene's lowest surface is ground, not a
#: feature.  The same test :func:`terrain_feature_spans` uses, so the colour and the shaded
#: band on the velocity panel agree about what counts as terrain.
GROUND_EPS_M: float = 0.02


def _ramp(frac: float) -> tuple[float, float, float]:
    """Linear interpolation along :data:`_HEIGHT_RAMP` at ``frac`` in ``[0, 1]``."""
    f = min(max(float(frac), 0.0), 1.0) * (len(_HEIGHT_RAMP) - 1)
    lo = int(math.floor(f))
    hi = min(lo + 1, len(_HEIGHT_RAMP) - 1)
    w = f - lo
    return tuple(float(_HEIGHT_RAMP[lo][k] * (1 - w) + _HEIGHT_RAMP[hi][k] * w) for k in range(3))


def shade_terrain(model, mode: str) -> int:
    """Recolour the terrain so a staircase reads as a staircase.

    One flat material over a twelve-step flight gives a single silhouette in which the treads
    are invisible at video bitrates; a per-step colour is the cheapest fix and costs nothing
    but ``geom_rgba``.

    Args:
        model: The compiled render model, edited in place.
        mode: ``"off"`` leaves the material alone; ``"height"`` ramps each feature's colour
            with its top face's height above the scene floor, which also reads on a box
            ladder; ``"alternate"`` gives adjacent features two shades, which reads better on
            a flight whose risers are all the same.

    Returns:
        How many geoms were recoloured, ground slabs included.

    Raises:
        ValueError: On an unknown mode.
    """
    if mode == "off":
        return 0
    if mode not in ("height", "alternate"):
        raise ValueError(f"terrain_shade must be off / height / alternate, got {mode!r}")
    floor_z = scene_floor_z(model)
    walls = set(wall_geoms(model))
    boxes = [i for i in world_boxes(model) if i not in walls]
    if not boxes:
        return 0
    tops = {i: float(model.geom_pos[i][2] + model.geom_size[i][2]) for i in boxes}
    ground = [i for i in boxes if tops[i] - floor_z < GROUND_EPS_M]
    feats = [i for i in boxes if i not in set(ground)]
    rise = (max(tops[i] for i in feats) - floor_z) if feats else 0.0
    for gid in ground:
        _detach_material(model, gid)
        model.geom_rgba[gid] = (*_GROUND_RGB, float(model.geom_rgba[gid][3]))
    for n, gid in enumerate(sorted(feats, key=lambda i: float(model.geom_pos[i][0]))):
        if mode == "height":
            frac = 0.0 if rise <= 1e-6 else (tops[gid] - floor_z) / rise
            rgb = _ramp(frac)
        else:
            rgb = _HEIGHT_RAMP[1] if n % 2 else _HEIGHT_RAMP[3]
        _detach_material(model, gid)
        model.geom_rgba[gid] = (*rgb, float(model.geom_rgba[gid][3]))
    return len(boxes)


def terrain_feature_spans(model, *, merge_gap_m: float = 0.35,
                          pad_m: float = 0.0) -> list[tuple[float, float]]:
    """Downrange ``(x_lo, x_hi)`` intervals occupied by RAISED terrain, merged.

    This is what the velocity panel shades, and it is read off the compiled model rather than
    restated from the course, so the band on the plot and the geometry in the frame are the
    same object.  Walls are excluded: they are not something the robot traverses.

    Args:
        model: The compiled render model.
        merge_gap_m: Features closer than this are one span -- which is what turns twelve
            stair treads into one staircase.
        pad_m: Widen every merged span by this much at each end.  0 by default, so a shaded
            interval means exactly "the root's x is over the feature" and nothing looser.

    Returns:
        Disjoint intervals, ascending.
    """
    floor_z = scene_floor_z(model)
    walls = set(wall_geoms(model))
    raw = []
    for i in world_boxes(model):
        if i in walls:
            continue
        top = float(model.geom_pos[i][2] + model.geom_size[i][2])
        if top - floor_z < GROUND_EPS_M:             # a ground slab, not a feature
            continue
        cx, sx = float(model.geom_pos[i][0]), float(model.geom_size[i][0])
        raw.append((cx - sx, cx + sx))
    if not raw:
        return []
    raw.sort()
    out = [list(raw[0])]
    for lo, hi in raw[1:]:
        if lo - out[-1][1] <= merge_gap_m:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [(lo - pad_m, hi + pad_m) for lo, hi in out]


# --------------------------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------------------------

#: ``azimuth`` is the compass bearing of the camera's FORWARD direction in the xy plane, so
#: 0 puts the camera behind the robot looking downrange and 90 puts it off the -y side looking
#: across the lane.  ``elevation`` is negative looking down.
CAMERAS: dict[str, dict[str, float]] = {
    "chase":    {"azimuth": 18.0, "elevation": -16.0, "distance": 5.0, "height": 0.45},
    "quarter":  {"azimuth": 52.0, "elevation": -15.0, "distance": 4.3, "height": 0.40},
    "side":     {"azimuth": 90.0, "elevation": -8.0,  "distance": 4.6, "height": 0.35},
    "front":    {"azimuth": 170.0, "elevation": -12.0, "distance": 4.4, "height": 0.35},
    "wide":     {"azimuth": 52.0, "elevation": -22.0, "distance": 8.5, "height": 0.80},
}


@dataclass
class Follower:
    """A look-at point that follows the root, per-axis smoothed.

    The pelvis bobs a few centimetres every step and swings laterally on a box; a look-at
    pinned rigidly to it transfers all of that to the frame and the clip is unwatchable.
    Downrange is followed HARD (the robot must not drift in frame) while lateral and vertical
    are low-passed, which is the split a camera operator makes by hand.

    Attributes:
        tau: Per-axis smoothing time constant (s).  0 tracks exactly.
        height: Metres added to the followed z, so the frame is centred on the torso rather
            than the pelvis.
    """

    tau: tuple[float, float, float] = (0.0, 0.7, 0.9)
    height: float = 0.35
    _state: np.ndarray | None = field(default=None, repr=False)

    def __call__(self, root_xyz: np.ndarray, dt: float) -> np.ndarray:
        """Advance the filter one frame and return the look-at point.

        Args:
            root_xyz: The root's world position this frame (m).
            dt: Seconds since the previous frame.

        Returns:
            ``(3,)`` look-at point.
        """
        target = np.asarray(root_xyz, dtype=np.float64).copy()
        target[2] += self.height
        if self._state is None:
            self._state = target.copy()
            return self._state.copy()
        for k in range(3):
            t = float(self.tau[k])
            a = 1.0 if t <= 0.0 else 1.0 - math.exp(-dt / t)
            self._state[k] += a * (target[k] - self._state[k])
        return self._state.copy()


def make_camera(preset: str, overrides: dict | None = None) -> tuple[mujoco.MjvCamera, Follower]:
    """Build the free camera and its follower from a preset plus overrides.

    Args:
        preset: A key of :data:`CAMERAS`.
        overrides: Any of ``azimuth`` / ``elevation`` / ``distance`` / ``height``, or ``None``.

    Returns:
        ``(camera, follower)``.  The camera's ``lookat`` is rewritten every frame.
    """
    spec = dict(CAMERAS[preset])
    spec.update({k: float(v) for k, v in (overrides or {}).items() if v is not None})
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = spec["azimuth"]
    cam.elevation = spec["elevation"]
    cam.distance = spec["distance"]
    return cam, Follower(height=spec["height"])


# --------------------------------------------------------------------------------------------
# The velocity panel
# --------------------------------------------------------------------------------------------

#: Trace column order, mirroring ``vel_tracking_sweep.TRACE_COLUMNS`` -- imported by name
#: there and restated here only as the default, so a trace written by something else can say
#: what its columns are.
TRACE_COLUMNS: tuple[str, ...] = ("t", "x", "y", "z", "vx", "vy", "wz",
                                  "cmd_vx", "cmd_vy", "cmd_wz")

#: Achieved / commanded column pair and axis label, per channel.
PANEL_CHANNELS: dict[str, tuple[str, str, str]] = {
    "vx": ("vx", "cmd_vx", r"$v_x$  [m/s]"),
    "vy": ("vy", "cmd_vy", r"$v_y$  [m/s]"),
    "wz": ("wz", "cmd_wz", r"$\omega_z$  [rad/s]"),
}


class VelocityPanel:
    """Commanded against achieved velocity, redrawn per frame by blitting.

    A full matplotlib redraw is ~30 ms a frame, which on a 25 s clip is most of the render
    time; the static half (axes, the commanded line, the shaded feature spans) is drawn once
    into a background and only the growing achieved line and the time cursor are redrawn.

    **The shaded bands are terrain, not time.**  They are the intervals during which the
    root's x lies over a raised feature, taken from :func:`terrain_feature_spans`, so the
    claim they support -- "the command is tracked except across a traversal" -- is read off
    the same geometry the viewer is looking at.
    """

    def __init__(self, trace: np.ndarray, channels: tuple[str, ...], width_px: int,
                 height_px: int, *, spans: list[tuple[float, float]] | None = None,
                 dpi: float = 100.0, columns: tuple[str, ...] = TRACE_COLUMNS,
                 face: str = "#FFFFFF", alpha: float = 1.0, smooth_s: float = 0.0) -> None:
        """Draw the static background once.

        Args:
            trace: ``(steps, len(columns))`` array.
            channels: Which of :data:`PANEL_CHANNELS` to stack, top to bottom.
            width_px / height_px: Panel size in pixels.
            spans: Downrange feature intervals to shade, or ``None`` for none.
            dpi: Figure dpi; the figure size is derived so the pixel size is exact, and
                the type scales with it -- pass ``100 * width / 1280`` to keep a panel's
                labels the same share of the frame at any render size.
            columns: Column names of ``trace``.
            face: Panel background colour.
            alpha: Panel background opacity, for the inset placement.
            smooth_s: Also draw a centred moving average of the achieved channel over this
                window (s), and make IT the heavy line.  The pelvis velocity carries a full
                per-stride ripple -- it swings a metre per second inside one step at 1.5 m/s
                -- so the raw line is honest and unreadable at video size.  The raw line
                stays drawn, thin, so the average is visibly an average and not a claim.
                0 draws the raw line alone.

        Raises:
            ValueError: On an unknown channel or a trace with no rows.
        """
        import matplotlib                                              # noqa: PLC0415
        matplotlib.use("Agg")
        from matplotlib.backends.backend_agg import FigureCanvasAgg    # noqa: PLC0415
        from matplotlib.figure import Figure                           # noqa: PLC0415

        bad = [c for c in channels if c not in PANEL_CHANNELS]
        if bad:
            raise ValueError(f"unknown velocity channel(s) {bad}; have {sorted(PANEL_CHANNELS)}")
        if trace.shape[0] == 0:
            raise ValueError("the trace has no rows, so there is no velocity to draw")
        col = {n: k for k, n in enumerate(columns)}
        self._t = np.asarray(trace[:, col["t"]], dtype=np.float64)
        self._t = self._t - self._t[0]
        x = np.asarray(trace[:, col["x"]], dtype=np.float64)

        # Font sizes are in POINTS, so matplotlib scales them with dpi on its own; the
        # margins below are not, and are written as a dpi-100 pixel count times `scale`.
        # Without that, doubling the frame halved the axis labels' share of it.  An inset
        # panel is under half the frame wide, where the full-size legend and the shading note
        # ran off the end of the axes: type shrinks and the note is dropped, not clipped.
        scale = dpi / 100.0
        narrow = width_px / scale < 700
        fs = 7.0 if narrow else 9.0

        self._fig = Figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
        self._canvas = FigureCanvasAgg(self._fig)
        self._fig.patch.set_facecolor(face)
        self._fig.patch.set_alpha(alpha)
        axes = self._fig.subplots(len(channels), 1, sharex=True, squeeze=False)[:, 0]
        # Margins in PIXELS, not in figure fractions.  A fractional bottom margin that fits a
        # two-channel panel clips the x-axis label off a one-channel one, which is half the
        # height -- and the label went missing exactly that way.
        self._fig.subplots_adjust(left=66 * scale / width_px,
                                  right=1 - 10 * scale / width_px,
                                  top=1 - 10 * scale / height_px,
                                  bottom=42 * scale / height_px, hspace=0.18)

        # Contiguous runs of "the root is over a raised feature", in TIME.
        on = np.zeros(x.shape, dtype=bool)
        for lo, hi in (spans or []):
            on |= (x >= lo) & (x <= hi)
        runs = _true_runs(on)

        dt = float(np.median(np.diff(self._t))) if self._t.size > 1 else 0.02
        win = max(1, int(round(float(smooth_s) / max(dt, 1e-6))))
        self._smooth_s = float(smooth_s) if win > 1 else 0.0

        self._ach, self._raw, self._cursor = [], [], []
        self._series: list[tuple[np.ndarray, np.ndarray | None]] = []
        for ax, name in zip(axes, channels):
            ach_c, cmd_c, label = PANEL_CHANNELS[name]
            a = np.asarray(trace[:, col[ach_c]], dtype=np.float64)
            c = np.asarray(trace[:, col[cmd_c]], dtype=np.float64)
            sm = _moving_average(a, win) if self._smooth_s else None
            self._series.append((a, sm))
            for i0, i1 in runs:
                ax.axvspan(self._t[i0], self._t[min(i1, len(self._t) - 1)],
                           color="#CC3311", alpha=0.10, lw=0, zorder=0)
            ax.plot(self._t, c, color="#333333", lw=2.0, ls="--", zorder=2, label="commanded")
            ax.plot(self._t, a if sm is None else sm, color="#BBBBBB", lw=1.2, zorder=1)
            if sm is not None:
                self._raw.append(ax.plot([], [], color="#7FC4E8", lw=1.0, zorder=3,
                                         label="achieved")[0])
                self._ach.append(ax.plot([], [], color="#0077BB", lw=2.6, zorder=4,
                                         label=f"achieved ({self._smooth_s:g} s mean)")[0])
            else:
                self._raw.append(None)
                self._ach.append(ax.plot([], [], color="#0077BB", lw=2.4, zorder=4,
                                         label="achieved")[0])
            self._cursor.append(ax.axvline(self._t[0], color="#CC3311", lw=1.4, zorder=5))
            lo = float(min(a.min(), c.min())); hi = float(max(a.max(), c.max()))
            pad = 0.12 * max(hi - lo, 0.5)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlim(self._t[0], self._t[-1])
            ax.set_ylabel(label, fontsize=fs + 1)
            ax.grid(True, alpha=0.25, lw=0.6)
            ax.tick_params(labelsize=fs)
            ax.set_facecolor(face)
            for s in ax.spines.values():
                s.set_alpha(0.4)
        axes[-1].set_xlabel("time [s]", fontsize=fs + 1)
        n_leg = (2 if self._smooth_s else 1) + 1
        axes[0].legend(loc="upper left", ncol=1 if narrow else n_leg, fontsize=fs,
                       framealpha=0.85, handlelength=1.4, borderpad=0.25,
                       columnspacing=1.0, labelspacing=0.25)
        if runs and not narrow:
            axes[0].text(0.995, 0.94, "shaded: root over a terrain feature",
                         transform=axes[0].transAxes, ha="right", va="top", fontsize=8.5,
                         color="#CC3311")

        self._canvas.draw()
        self._bg = self._canvas.copy_from_bbox(self._fig.bbox)
        self._axes = axes
        self._trace = trace
        self._cols = col
        self._channels = channels
        self.size = (int(self._canvas.get_width_height()[0]),
                     int(self._canvas.get_width_height()[1]))

    def frame(self, k: int) -> np.ndarray:
        """The panel as of trace row ``k``.

        Args:
            k: Row index into the trace.

        Returns:
            ``(H, W, 4)`` uint8 RGBA.
        """
        k = int(min(max(k, 0), len(self._t) - 1))
        self._canvas.restore_region(self._bg)
        for ax, line, raw, cur, (a, sm) in zip(self._axes, self._ach, self._raw,
                                               self._cursor, self._series):
            if raw is not None:
                raw.set_data(self._t[:k + 1], a[:k + 1])
                ax.draw_artist(raw)
            line.set_data(self._t[:k + 1], (a if sm is None else sm)[:k + 1])
            cur.set_xdata([self._t[k], self._t[k]])
            ax.draw_artist(line)
            ax.draw_artist(cur)
        self._canvas.blit(self._fig.bbox)
        return np.asarray(self._canvas.buffer_rgba()).copy()


def _moving_average(y: np.ndarray, win: int) -> np.ndarray:
    """Centred moving average of ``y`` over ``win`` samples, edges included.

    Edge-corrected rather than padded: dividing by the number of samples that actually fell
    in the window keeps the first and last points at the data's own level instead of decaying
    them toward zero, which on a velocity trace would read as the robot stopping.

    Args:
        y: The series.
        win: Window length in samples; ``<= 1`` returns a copy.

    Returns:
        The smoothed series, same length.
    """
    if win <= 1:
        return np.asarray(y, dtype=np.float64).copy()
    k = np.ones(int(win), dtype=np.float64)
    num = np.convolve(np.asarray(y, dtype=np.float64), k, mode="same")
    den = np.convolve(np.ones_like(y, dtype=np.float64), k, mode="same")
    return num / den


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``[start, end]`` index runs where ``mask`` is True."""
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return []
    breaks = np.nonzero(np.diff(idx) > 1)[0]
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


# --------------------------------------------------------------------------------------------
# Captions
# --------------------------------------------------------------------------------------------

def _font(size: int):
    """A TrueType font at ``size`` px, from matplotlib's bundled DejaVu Sans."""
    from matplotlib import font_manager                               # noqa: PLC0415
    from PIL import ImageFont                                         # noqa: PLC0415
    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)


def draw_caption(rgb: np.ndarray, lines: list[str], *, corner: str = "tl",
                 size: int = 30, pad: int = 18) -> np.ndarray:
    """Burn a few lines of text into a frame, on a translucent plate.

    Args:
        rgb: ``(H, W, 3)`` uint8 frame.
        lines: Text lines; empty leaves the frame untouched.
        corner: ``tl`` / ``tr`` / ``bl`` / ``br``.
        size: Font size in px.
        pad: Margin from the frame edge, in px.

    Returns:
        A new ``(H, W, 3)`` uint8 frame.
    """
    if not lines:
        return rgb
    from PIL import Image, ImageDraw                                  # noqa: PLC0415

    img = Image.fromarray(rgb).convert("RGBA")
    plate = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(plate)
    font = _font(size)
    w = max(int(d.textlength(s, font=font)) for s in lines)
    h = int(1.35 * size * len(lines))
    x = pad if corner in ("tl", "bl") else img.size[0] - w - 3 * pad
    y = pad if corner in ("tl", "tr") else img.size[1] - h - 2 * pad
    d.rounded_rectangle([x, y, x + w + 2 * pad, y + h + pad], radius=10,
                        fill=(255, 255, 255, 190))
    for n, s in enumerate(lines):
        d.text((x + pad, y + int(pad * 0.5) + int(1.35 * size * n)), s,
               font=font, fill=(20, 24, 30, 255))
    return np.asarray(Image.alpha_composite(img, plate).convert("RGB"))


# --------------------------------------------------------------------------------------------
# The render loop
# --------------------------------------------------------------------------------------------

def _even(n: int) -> int:
    """Round down to an even number; h264 with yuv420p refuses odd dimensions."""
    return int(n) - (int(n) % 2)


def _paste(base: np.ndarray, rgba: np.ndarray, x: int, y: int) -> np.ndarray:
    """Alpha-composite an RGBA image onto an RGB frame at ``(x, y)``, clipped to the frame."""
    h, w = rgba.shape[:2]
    y0, x0 = max(0, y), max(0, x)
    y1, x1 = min(base.shape[0], y + h), min(base.shape[1], x + w)
    if y1 <= y0 or x1 <= x0:
        return base
    src = rgba[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    a = src[..., 3:4] / 255.0
    base[y0:y1, x0:x1] = (src[..., :3] * a
                          + base[y0:y1, x0:x1].astype(np.float32) * (1.0 - a)).astype(np.uint8)
    return base


@dataclass
class RenderSpec:
    """Everything the render loop needs that is not the trace itself.

    Attributes:
        width / height: Size of the 3-D view, in pixels.  The panel is added BELOW this in
            ``bottom`` placement, so the file is taller than ``height`` there.  Capped by the
            scene's offscreen framebuffer, which the themes set to 3840x2160.
        fps: Output frame rate.
        rt: Sim seconds per output second.  0.5 is half speed.
        camera / cam_overrides: See :func:`make_camera`.
        theme / terrain_shade / wall_alpha: See the restyling helpers above.
        geom_groups: Which MuJoCo geom groups to draw.  The G1's visual meshes are group 1 and
            the terrain is group 2; group 0 is the collision primitives, which are boxes where
            the visual is a foot and have no business in a video.
        vel_plot: ``off`` / ``bottom`` / ``inset``.
        vel_channels: Which velocity channels the panel stacks.
        caption: Lines burnt into the top-left corner; empty for a clean frame.
        shadows / reflections: Renderer flags.
        vel_smooth_s: Moving-average window for the panel's achieved line (s); 0 draws the
            raw per-step velocity alone.  See :class:`VelocityPanel`.
        follow_yaw: Rotate the camera with the robot's heading, so ``camera``'s azimuth
            becomes an offset FROM the robot rather than a world bearing.  A fixed azimuth is
            right for a lane, where the robot goes one way; on a clip with a turn in it the
            robot swings through the frame and ends up walking away from the camera.  The
            heading is low-passed by ``follow_yaw_tau`` -- the raw yaw carries the gait's own
            oscillation, which a camera locked to it would put into every frame.
        follow_yaw_tau: Smoothing time constant for that rotation (s).
        span_pad_m: Widen every shaded feature span by this much at each end.  0 means a
            shaded interval is exactly "the root's x is over the feature"; anything larger is
            a readability choice and not a measurement, which is why it is 0 by default even
            though the pair gives up speed on the APPROACH to a feature as well as on it.
        trim_s: ``(head, tail)`` seconds cut from the start and the end of the trace.
        hold_end_s: Seconds of the final frame held at the end, so a looping clip lands.
    """

    width: int = 1920
    height: int = 1080
    fps: int = 30
    rt: float = 1.0
    camera: str = "quarter"
    cam_overrides: dict = field(default_factory=dict)
    theme: str = "bright"
    terrain_shade: str = "height"
    wall_alpha: float = 0.30
    geom_groups: tuple[int, ...] = (1, 2)
    vel_plot: str = "off"
    vel_channels: tuple[str, ...] = ("vx",)
    vel_smooth_s: float = 0.0
    follow_yaw: bool = False
    follow_yaw_tau: float = 1.2
    span_pad_m: float = 0.0
    caption: tuple[str, ...] = ()
    shadows: bool = True
    reflections: bool = False
    trim_s: tuple[float, float] = (0.0, 0.0)
    hold_end_s: float = 0.6


def render_trace(scene_xml: str, qpos: np.ndarray, time_s: np.ndarray, trace: np.ndarray,
                 out_mp4: str, spec: RenderSpec, *,
                 columns: tuple[str, ...] = TRACE_COLUMNS) -> dict:
    """Render one recorded episode to an mp4.

    Args:
        scene_xml: The scene the episode was rolled on.  A THEMED COPY is written beside it
            and loaded; the original is never modified.
        qpos: ``(steps, nq)`` recorded poses.
        time_s: ``(steps,)`` sim times.
        trace: ``(steps, len(columns))`` velocity trace, for the panel.  May be empty when
            ``spec.vel_plot == "off"``.
        out_mp4: Output path.
        spec: The look.
        columns: Column names of ``trace``.

    Returns:
        A dict of what was rendered: frame count, duration, the themed scene path and the
        feature spans, for the shot manifest.

    Raises:
        ValueError: If the trace and the poses disagree on how many steps there were, or the
            scene's offscreen buffer is smaller than the requested frame.
    """
    import imageio.v2 as iio                                          # noqa: PLC0415

    if trace.size and trace.shape[0] != qpos.shape[0]:
        raise ValueError(f"the trace has {trace.shape[0]} rows and the poses {qpos.shape[0]}; "
                         "they are recorded at the same hook and must agree")
    themed = themed_scene(scene_xml, _themed_path(scene_xml, spec.theme), spec.theme)
    model = mujoco.MjModel.from_xml_path(themed)
    if model.vis.global_.offwidth < spec.width or model.vis.global_.offheight < spec.height:
        raise ValueError(
            f"the scene's offscreen buffer is {model.vis.global_.offwidth}x"
            f"{model.vis.global_.offheight}, smaller than the requested "
            f"{spec.width}x{spec.height}; widen <global offwidth/offheight> in the theme")
    apply_lighting(model, spec.theme)
    n_shade = shade_terrain(model, spec.terrain_shade)
    n_wall = fade_walls(model, spec.wall_alpha)
    spans = terrain_feature_spans(model, pad_m=spec.span_pad_m)
    data = mujoco.MjData(model)

    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    for g in spec.geom_groups:
        opt.geomgroup[int(g)] = 1

    # Trim in SIM time, then choose output frames by sim time so --rt is exact rather than a
    # rounded stride: at 50 Hz control and 30 fps a stride would be 1.67 and quantising it
    # makes the playback speed wrong by 20%.
    dt = float(np.median(np.diff(time_s))) if time_s.size > 1 else 0.02
    t0 = float(time_s[0]) + float(spec.trim_s[0])
    t1 = float(time_s[-1]) - float(spec.trim_s[1])
    if t1 <= t0:
        raise ValueError(f"--trim leaves nothing: {t0:.2f} s .. {t1:.2f} s")
    n_out = max(1, int(math.floor((t1 - t0) / (spec.rt / spec.fps))) + 1)
    frame_t = t0 + np.arange(n_out) * (spec.rt / spec.fps)
    frame_k = np.clip(np.searchsorted(time_s, frame_t), 0, len(time_s) - 1)

    panel = None
    if spec.vel_plot != "off":
        if not trace.size:
            raise ValueError("--vel_plot needs a trace, and this recording has none")
        # One generous panel plus a thinner strip per extra channel: three channels stacked
        # at the single-channel height would be taller than the 3-D view they annotate.
        n_extra = max(0, len(spec.vel_channels) - 1)
        pw = spec.width if spec.vel_plot == "bottom" else int(0.44 * spec.width)
        base, step = ((0.26, 0.17) if spec.vel_plot == "bottom" else (0.20, 0.13))
        ph = int((base + step * n_extra) * spec.height)
        panel = VelocityPanel(trace, spec.vel_channels, _even(pw), _even(ph), spans=spans,
                              dpi=100.0 * spec.width / 1280.0,
                              alpha=1.0 if spec.vel_plot == "bottom" else 0.88,
                              smooth_s=spec.vel_smooth_s)

    out_w = _even(spec.width)
    out_h = _even(spec.height + (panel.size[1] if spec.vel_plot == "bottom" else 0))
    cam, follow = make_camera(spec.camera, spec.cam_overrides)

    os.makedirs(os.path.dirname(os.path.abspath(out_mp4)) or ".", exist_ok=True)
    renderer = mujoco.Renderer(model, height=_even(spec.height), width=out_w)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = int(bool(spec.shadows))
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = int(bool(spec.reflections))
    writer = iio.get_writer(out_mp4, fps=spec.fps, codec="libx264", quality=9,
                            pixelformat="yuv420p", macro_block_size=1)
    base_az, yaw_s = float(cam.azimuth), None
    try:
        prev_t = None
        for n, k in enumerate(frame_k):
            data.qpos[:] = qpos[k]
            mujoco.mj_forward(model, data)
            step = (frame_t[n] - prev_t) if prev_t is not None else dt
            prev_t = frame_t[n]
            cam.lookat[:] = follow(qpos[k, :3], max(step, 1e-3))
            if spec.follow_yaw:
                w, qx, qy, qz = (float(v) for v in qpos[k, 3:7])
                yaw = math.degrees(math.atan2(2.0 * (w * qz + qx * qy),
                                              1.0 - 2.0 * (qy * qy + qz * qz)))
                if yaw_s is None:
                    yaw_s = yaw
                else:
                    # Unwrap against the filter state, or a pass through +-180 deg would spin
                    # the camera the long way round in a single frame.
                    yaw += 360.0 * round((yaw_s - yaw) / 360.0)
                    a = 1.0 - math.exp(-max(step, 1e-3) / max(spec.follow_yaw_tau, 1e-3))
                    yaw_s += a * (yaw - yaw_s)
                cam.azimuth = base_az + yaw_s
            renderer.update_scene(data, camera=cam, scene_option=opt)
            frame = renderer.render()
            if spec.caption:
                frame = draw_caption(frame, list(spec.caption),
                                     size=int(round(30 * spec.width / 1280)),
                                     pad=int(round(18 * spec.width / 1280)))
            if panel is not None and spec.vel_plot == "bottom":
                out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                out[:frame.shape[0]] = frame
                out[:] = _paste(out, panel.frame(int(k)), 0, frame.shape[0])
                frame = out
            elif panel is not None:
                frame = _paste(frame.copy(), panel.frame(int(k)),
                               out_w - panel.size[0] - 24,
                               spec.height - panel.size[1] - 24)
            writer.append_data(frame)
            if n == len(frame_k) - 1:
                for _ in range(int(round(spec.hold_end_s * spec.fps))):
                    writer.append_data(frame)
    finally:
        writer.close()
        renderer.close()
    return {"frames": int(n_out), "seconds": float(n_out / spec.fps),
            "sim_seconds": float(t1 - t0), "themed_scene": themed,
            "feature_spans": [[float(a), float(b)] for a, b in spans],
            "geoms_shaded": int(n_shade), "walls_faded": int(n_wall),
            "out": os.path.abspath(out_mp4)}


def _themed_path(scene_xml: str, theme: str) -> str:
    """Where the themed copy of a scene goes: beside it, with the theme in the name.

    Beside it rather than in the shot directory because the scene ``<include>``s the robot
    model by a RELATIVE path, so a copy anywhere else fails to compile.
    """
    d, base = os.path.split(scene_xml)
    stem = re.sub(r"\.xml$", "", base)
    return os.path.join(d, f"{stem}__view_{theme}.xml")
