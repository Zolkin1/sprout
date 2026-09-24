# Copyright 2026 Zachary Olkin. All rights reserved.

"""Per-robot viser scene nodes: URDF body, prediction ghosts, curve, height scan.

No IsaacLab imports here; everything renders from numpy arrays loaded out of a
recording (see ``recording.py``).
"""

from __future__ import annotations

import os
from functools import partial

import numpy as np
import viser
import yourdfpy
from viser.extras import ViserUrdf

#: The robot every stage of the pipeline now uses, and therefore the one every viewer should
#: draw by default. It is the trained robot: mode_11 motors, the 2.64 kg compute backpack,
#: passive rubber hands, and the depth mounts on their as-built CAD pose.
#:
#: Declared ONCE. Five viewers used to carry their own copy and three of them had drifted --
#: ``view_clip_viser`` was still defaulting to a 21-DoF model, which cannot even be permuted
#: onto a 29-DoF recording. A default in five places is a default that rots in four of them.
#:
#: Repo-relative; callers resolve it against their own root.
DEFAULT_URDF: str = "robot_assets/g1_perceptive/g1_perceptive.urdf"

# Per-robot overlay colors (RGB 0-1), cycled by env index. Robot 0 keeps the
# orange the old in-sim visualization used for the root line.
_ROBOT_COLORS: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.55, 0.1),   # orange
    (0.2, 0.4, 1.0),    # blue
    (0.2, 1.0, 0.4),    # green
    (1.0, 0.2, 0.9),    # magenta
    (1.0, 0.85, 0.2),   # yellow
)
_SCAN_COLOR: tuple[float, float, float] = (0.3, 0.8, 1.0)
# Ghost alpha ramp: near future fairly solid, far future faint.
_GHOST_ALPHA_NEAR = 0.55
_GHOST_ALPHA_FAR = 0.12

# Placeholder geometry for handles that are created hidden and filled on the
# first frame update (viser requires non-empty arrays at creation).
_PLACEHOLDER_SEGMENTS = np.zeros((1, 2, 3), dtype=np.float32)
_PLACEHOLDER_POINTS = np.zeros((1, 3), dtype=np.float32)


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Reorder quaternion components from this repo's (x, y, z, w) to viser's (w, x, y, z)."""
    return q[..., [3, 0, 1, 2]]


def _disambiguate_geometry_names(robot: yourdfpy.Robot) -> int:
    """Rename every ``<visual>``/``<collision>`` whose ``name`` is not usable as a scene-graph
    node, in place, and return how many were renamed.

    A URDF may name a visual anything it likes, including the name of its own link -- and
    ``make_perceptive_urdf.py`` does exactly that for the two camera mounts and the backpack
    shell. That is legal URDF and correct everywhere else in the stack (Isaac and MuJoCo both
    place those meshes right); it breaks ONLY the yourdfpy -> viser path, because the two
    disagree about what the name means:

    * yourdfpy passes the visual's name to ``trimesh.Scene.add_geometry`` as the GEOMETRY key,
      and trimesh then picks a NODE name that is unique among existing nodes -- so a visual
      named ``backpack_shell_link`` becomes geometry ``backpack_shell_link`` hanging off node
      ``backpack_shell_link_1``, because the link already owns the bare name;
    * ``ViserUrdf`` iterates ``scene.geometry.items()`` and uses each key as if it were the
      node name, so it looks up the LINK's parent (the torso) instead of the mesh's, and bakes
      the fixed-joint transform into the vertices of a mesh that is then parented under that
      same joint frame.

    The mesh is thus posed by its mount transform TWICE and hangs off the robot by it. There
    is no shape or count error anywhere: all 39 meshes load, the joint tree is right, and the
    only symptom is three parts floating in space.

    The fix belongs here rather than in the URDF: ``robot_assets/g1_perceptive/g1_perceptive.urdf``
    is fingerprinted into every exported ``tracker.onnx``/``fm.onnx`` (see ``motion_gen.robot_asset``),
    so editing an attribute no simulator reads would invalidate every checkpoint on disk to fix
    a rendering bug.

    Args:
        robot: A parsed ``yourdfpy.Robot`` (mutated in place).

    Returns:
        Number of renamed geometry entries.
    """
    link_names = {link.name for link in robot.links}
    taken: set[str] = set()
    renamed = 0
    for link in robot.links:
        for kind, geoms in (("visual", link.visuals), ("collision", link.collisions)):
            for i, g in enumerate(geoms):
                # A ``None`` name is the common case and is SAFE: trimesh then derives the key
                # from the mesh file (``pelvis.STL``), which cannot collide with a link name.
                # Leave it alone -- those names are what the viser scene tree shows.
                if g.name is None or (g.name not in link_names and g.name not in taken):
                    if g.name is not None:
                        taken.add(g.name)
                    continue
                base = f"{link.name}_{kind}_{i}"
                new, k = base, 0
                while new in link_names or new in taken:
                    k += 1
                    new = f"{base}_{k}"
                g.name, renamed = new, renamed + 1
                taken.add(new)
    return renamed


def _assert_every_mesh_is_its_own_node(scene, urdf_path: str, which: str) -> None:
    """Raise unless every geometry key in ``scene`` is also the graph node that holds it.

    This is the invariant :class:`ViserUrdf` assumes and never checks; when it fails the
    meshes still render, just in the wrong place (see :func:`_disambiguate_geometry_names`).
    A silent misplacement is exactly the failure this repo keeps paying for, so assert it.

    Args:
        scene: ``urdf.scene`` or ``urdf.collision_scene``.
        urdf_path: Path, for the error message.
        which: ``"visual"`` or ``"collision"``, for the error message.

    Raises:
        ValueError: If any geometry key is not a geometry node of the same name.
    """
    geom, nodes = set(scene.geometry), set(scene.graph.nodes_geometry)
    if geom != nodes:
        raise ValueError(
            f"{urdf_path}: the {which} scene graph names a mesh differently from the node that "
            f"holds it, which ViserUrdf would silently render in the wrong place.\n"
            f"  geometry keys with no matching node: {sorted(geom - nodes)}\n"
            f"  mesh nodes with no matching key    : {sorted(nodes - geom)}\n"
            "  (_disambiguate_geometry_names was supposed to have prevented this)"
        )


def load_urdf(urdf_path: str, with_collision: bool = False) -> yourdfpy.URDF:
    """Load a URDF with meshes resolved relative to the URDF's directory.

    The returned instance is meant to be loaded ONCE and shared by every
    :class:`ViserUrdf` (main robots and all ghosts): ViserUrdf copies each mesh
    at construction, and ``update_cfg`` mutates the shared scene graph but
    reads the transforms back synchronously, so sharing is safe as long as all
    updates are serialized (the viewer holds a lock around frame updates).
    Relies on viser 1.0.x internals; fall back to one ``load_urdf`` per
    ViserUrdf if an upgrade breaks this.

    The URDF is parsed twice on purpose: :func:`_disambiguate_geometry_names` has to run
    before the scene graph exists, and re-parsing the XML costs nothing next to loading the
    meshes (which the first pass skips).

    Args:
        urdf_path: Path to the URDF.
        with_collision: Also build and load the COLLISION scene. Off by default -- only
            ``view_terrain_reward_viser`` overlays the collision primitives, and loading them
            doubles the mesh work for every other viewer.

    Returns:
        The loaded URDF, with its visual (and optionally collision) scene built.
    """
    handler = partial(yourdfpy.filename_handler_magic, dir=os.path.dirname(urdf_path))
    parsed = yourdfpy.URDF.load(
        urdf_path, build_scene_graph=False, load_meshes=False, filename_handler=handler,
    )
    renamed = _disambiguate_geometry_names(parsed.robot)
    if renamed:
        print(f"[urdf] {renamed} visual/collision name(s) collided with a link name and were "
              f"renamed for rendering ({os.path.basename(urdf_path)}); the file on disk is untouched.")
    urdf = yourdfpy.URDF(
        robot=parsed.robot,
        build_scene_graph=True,
        load_meshes=True,
        build_collision_scene_graph=with_collision,
        load_collision_meshes=with_collision,
        filename_handler=handler,
    )
    _assert_every_mesh_is_its_own_node(urdf.scene, urdf_path, "visual")
    if with_collision:
        _assert_every_mesh_is_its_own_node(urdf.collision_scene, urdf_path, "collision")
    return urdf


def recording_to_urdf_perm(recording_joint_names: list[str], urdf: yourdfpy.URDF) -> np.ndarray:
    """Map the recording's joint ordering onto the URDF's actuated-joint ordering.

    Args:
        recording_joint_names: Joint names in the recording's ``joint_pos`` order.
        urdf: The loaded URDF.

    Returns:
        Index array ``perm`` such that ``joint_pos[perm]`` is in
        ``urdf.actuated_joint_names`` order (what ``ViserUrdf.update_cfg`` expects).

    Raises:
        ValueError: If any actuated URDF joint is missing from the recording.
    """
    actuated = list(urdf.actuated_joint_names)
    # The reverse direction is NOT an error -- a URDF may legitimately pose a subset -- but it is
    # almost always a mistake, and a silent one: the extra joints simply stay at their default, so a
    # 29-DOF motion rendered on a 21-DOF URDF looks stiff rather than broken.  Say so.
    extra = [n for n in recording_joint_names if n not in actuated]
    if extra:
        print(f"[urdf] WARNING: {len(extra)} recorded joint(s) are not actuated by this URDF and will "
              f"be frozen at their default: {', '.join(extra[:8])}"
              f"{' ...' if len(extra) > 8 else ''}")
    missing = [n for n in actuated if n not in recording_joint_names]
    if missing:
        raise ValueError(
            "URDF actuated joints missing from the recording.\n"
            f"  missing            : {missing}\n"
            f"  recording joints   : {recording_joint_names}\n"
            f"  URDF actuated joints: {actuated}"
        )
    return np.array([recording_joint_names.index(n) for n in actuated], dtype=np.int64)


class RobotVisual:
    """All viser scene nodes for one recorded env.

    Owns the solid robot body, and the prediction overlays: ghost robots along
    the predicted trajectory, the predicted-root curve, and the height-scan
    point cloud. Overlay state (enabled / ghost count) only flips visibility
    and slot selection; the viewer re-applies the current frame afterwards, so
    nothing is cached here.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        urdf: yourdfpy.URDF,
        env_index: int,
        joint_perm: np.ndarray,
        future_len: int,
        max_ghosts: int,
        num_ghosts: int,
        point_size: float,
    ):
        """Create the body nodes immediately; ghosts are built lazily on first enable.

        Args:
            server: The viser server.
            urdf: Shared loaded URDF (see :func:`load_urdf`).
            env_index: Recorded env index (names the scene nodes, picks the color).
            joint_perm: Recording-order -> URDF-actuated-order index array.
            future_len: Prediction horizon F.
            max_ghosts: Size of the ghost slot pool.
            num_ghosts: Initially active ghost count.
            point_size: Height-scan point size in meters.
        """
        self._server = server
        self._urdf = urdf
        self._joint_perm = joint_perm
        self._future_len = future_len
        self._max_ghosts = max_ghosts
        self._num_ghosts = num_ghosts
        self._color = _ROBOT_COLORS[env_index % len(_ROBOT_COLORS)]
        self._ghost_prefix = f"/overlays/{env_index}/ghosts"
        self.enabled = False
        self.visible = True
        self._alpha_scale = 1.0

        self._base = server.scene.add_frame(f"/robots/{env_index}", show_axes=False)
        self._body = ViserUrdf(server, urdf, root_node_name=f"/robots/{env_index}")

        self._pred_line = server.scene.add_line_segments(
            f"/overlays/{env_index}/pred_line",
            points=_PLACEHOLDER_SEGMENTS,
            colors=self._color,
            line_width=3.0,
            visible=False,
        )
        self._scan = server.scene.add_point_cloud(
            f"/overlays/{env_index}/scan",
            points=_PLACEHOLDER_POINTS,
            colors=_SCAN_COLOR,
            point_size=point_size,
            point_shape="circle",
            visible=False,
        )
        # Ghost slot pool, built lazily: slot s always shows the FIXED future
        # fraction (s + 1) / max_ghosts with a matching fixed alpha, so the
        # ghost-count slider only flips slot visibility (no geometry re-upload).
        self._ghosts: list[tuple[viser.FrameHandle, ViserUrdf]] | None = None

    def _slot_base_alpha(self, slot: int) -> float:
        """Unscaled alpha for a ghost slot (near future solid, far future faint)."""
        frac = slot / max(self._max_ghosts - 1, 1)
        return _GHOST_ALPHA_NEAR + (_GHOST_ALPHA_FAR - _GHOST_ALPHA_NEAR) * frac

    def _build_ghosts(self) -> None:
        """Create the ghost slot pool (one ViserUrdf per slot, alpha by fraction)."""
        self._ghosts = []
        for s in range(self._max_ghosts):
            alpha = min(1.0, self._slot_base_alpha(s) * self._alpha_scale)
            frame = self._server.scene.add_frame(f"{self._ghost_prefix}/{s}", show_axes=False, visible=False)
            ghost = ViserUrdf(
                self._server,
                self._urdf,
                root_node_name=f"{self._ghost_prefix}/{s}",
                mesh_color_override=(*self._color, alpha),
            )
            self._ghosts.append((frame, ghost))

    def _active_slots(self) -> list[int]:
        """Ghost slots shown for the current count, spread across the pool."""
        if self._num_ghosts <= 0:
            return []
        slots = np.round(np.linspace(0, self._max_ghosts - 1, self._num_ghosts)).astype(int)
        return sorted(set(slots.tolist()))

    def _slot_future_index(self, slot: int) -> int:
        """Future step shown by a ghost slot (its fixed fraction of the horizon)."""
        return round((slot + 1) / self._max_ghosts * (self._future_len - 1))

    def set_state(self, root_pos: np.ndarray, root_quat_xyzw: np.ndarray, joint_pos: np.ndarray) -> None:
        """Pose the solid robot body for the current frame."""
        self._base.position = root_pos
        self._base.wxyz = xyzw_to_wxyz(root_quat_xyzw)
        self._body.update_cfg(joint_pos[self._joint_perm])

    def set_prediction(
        self,
        current_root_pos: np.ndarray,
        pred_root_pos: np.ndarray,
        pred_root_quat_xyzw: np.ndarray,
        pred_joint_pos: np.ndarray,
    ) -> None:
        """Show the predicted trajectory: root curve + ghost robots.

        Args:
            current_root_pos: ``(3,)`` current root position (the curve emanates
                from the robot).
            pred_root_pos: ``(F, 3)`` predicted world root positions.
            pred_root_quat_xyzw: ``(F, 4)`` predicted world root quaternions.
            pred_joint_pos: ``(F, nj)`` predicted joint positions (recording order).
        """
        pts = np.concatenate([current_root_pos[None], pred_root_pos], axis=0)
        self._pred_line.points = np.stack([pts[:-1], pts[1:]], axis=1)
        self._pred_line.visible = True

        if self._ghosts is None:
            self._build_ghosts()
        active = set(self._active_slots())
        for s, (frame, ghost) in enumerate(self._ghosts):
            if s in active:
                k = self._slot_future_index(s)
                frame.position = pred_root_pos[k]
                frame.wxyz = xyzw_to_wxyz(pred_root_quat_xyzw[k])
                ghost.update_cfg(pred_joint_pos[k][self._joint_perm])
                frame.visible = True
            else:
                frame.visible = False

    def clear_prediction(self) -> None:
        """Hide the curve and ghosts (no valid prediction at this frame)."""
        self._pred_line.visible = False
        if self._ghosts is not None:
            for frame, _ in self._ghosts:
                frame.visible = False

    def set_scan(self, hits_w: np.ndarray | None) -> None:
        """Show the height-scan hit points (``(R, 3)`` world frame, may contain inf)."""
        if hits_w is None:
            self._scan.visible = False
            return
        finite = hits_w[np.isfinite(hits_w).all(axis=-1)]
        if finite.shape[0] == 0:
            self._scan.visible = False
            return
        self._scan.points = finite.astype(np.float32)
        self._scan.visible = True

    def set_overlay_enabled(self, enabled: bool) -> None:
        """Toggle the prediction overlays; hides them immediately when disabled."""
        self.enabled = enabled
        if not enabled:
            self.clear_prediction()
            self._scan.visible = False

    def set_visible(self, visible: bool) -> None:
        """Toggle the entire robot: body AND prediction overlays.

        When re-shown, the overlays come back on the next frame update (the
        viewer re-applies the current frame after toggling).
        """
        self.visible = visible
        self._base.visible = visible
        if not visible:
            self.clear_prediction()
            self._scan.visible = False

    def set_num_ghosts(self, count: int) -> None:
        """Change the active ghost count (the viewer re-applies the frame after)."""
        self._num_ghosts = count

    def set_ghost_alpha(self, scale: float) -> None:
        """Scale every ghost's alpha by ``scale`` (1.0 = the built-in near/far ramp,
        >1 more solid; effective alphas are clamped to 1.0).

        Updates the live mesh opacities through ``ViserUrdf._meshes`` (the
        RGBA-override meshes are ``add_mesh_simple`` handles whose ``opacity``
        is a settable prop) — same viser-internals reliance as the shared-URDF
        trick in :func:`load_urdf`.
        """
        self._alpha_scale = scale
        if self._ghosts is None:
            return  # built lazily later with the new scale
        for s, (_, ghost) in enumerate(self._ghosts):
            alpha = min(1.0, self._slot_base_alpha(s) * scale)
            for mesh in ghost._meshes:
                mesh.opacity = alpha
