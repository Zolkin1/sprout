#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Augment a recording by DISPLACING THE ROBOT, so the generator has to localise terrain.

The generator was measured not to register terrain at all: displace the robot downrange and
its predicted stair segment slides along with it (0.098 / 0.168 m for a 0.10 / 0.20 m shift,
where the anchor mask requires 0.000 -- ``fm_registration_probe.py``). The cause is in the
data, not the model. Every recorded window has the robot sitting exactly on its clip, so
depth and the correct answer are perfectly correlated and depth can be ignored outright.

This breaks that correlation offline. For each source window it re-places the robot at its
recorded pose PLUS a sampled ``(downrange, lateral, yaw)`` offset, re-casts the depth cameras
from there, and writes a new window. No physics, no controller, no rollout -- Isaac is used
only as a renderer.

What changes, and what deliberately does not:

* ``depth`` -- re-rendered from the displaced pose.
* ``height_scan`` -- re-cast from the displaced pose, when the recording carries one. These two
  are the ONLY channels carrying the offset. The scan was previously copied through
  BIT-IDENTICAL while the anchor moved (measured: 0.18 m mean anchor shift, 40/40 windows
  bit-equal), so a scan-conditioned model saw a sensor that contradicted its own target -- this
  script's premise, inverted. Which sensor is derived from the RECORDED ray count, not named,
  since the generator may scan on its own grid.
* ``history_state[H-1]`` root position + yaw -- the SNAP ANCHOR, the frame the dataset
  anchors ``ref_future`` against (``world_to_window``'s anchor is YAW-ONLY, so roll/pitch
  there are never read). Displacing it is what turns the unchanged world-frame clip into a
  CORRECTION back onto the terrain. No target arithmetic, so no sign to get backwards.
* ``ref_future`` / ``cmd_future`` -- untouched. They are stored in WORLD; the clip does not
  move just because the robot did.
* the HISTORY (rows ``H-1-n .. H-2`` and all of ``history_ref``) -- untouched, and this is
  load-bearing rather than incidental. The history is the generator's own previous output
  expressed relative to its own frame, so it is invariant to a rigid displacement anyway --
  and if the offset DID leak into it, the model could recover the offset without ever
  consulting depth, which is precisely the shortcut this augmentation exists to remove.

Terrain must be IDENTICAL to the recording: the recorded poses are absolute world, so the
tiles have to be where they were. The library is taken from the recording's own
``clip_library_path`` and its sha256 is verified, and ``--num_envs`` defaults to the
recording's, because env origins enter the tile layout.

    # 1. prove the pipeline: re-render at ZERO offset and diff against the recorded depth
    python scripts/rsl_rl/augment_registration.py --headless --data <rec>.h5 --check_only

    # 2. then augment
    python scripts/rsl_rl/augment_registration.py --headless --data <rec>.h5 \\
        --out <aug>.h5 --num_augments 2 --downrange 0.15 --lateral 0.20 --yaw 0.15
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="G1-teacher-clf-terrain-depth")
parser.add_argument("--data", required=True, help="Recording to augment (needs clip_id).")
parser.add_argument("--out", default=None, help="Output .h5 (omit with --check_only).")
parser.add_argument("--num_augments", type=int, default=3,
                    help="New points generated per SOURCE point. Default 3 to match the MVP "
                         "pair, recovered arithmetically because neither file records the "
                         "flag: its augmented_20260812.h5 holds 277878 windows against a "
                         "200000-window source, and 277878 = 3 x 92626 = 3 x 46.31%% of the "
                         "source. That 46.31%% is the TERRAIN-IN-HORIZON survival rate, which "
                         "a comparable recording measures at 45.2%% -- the two agree to about "
                         "a point. Note 277878 also factors as 2 x 138939 (69.47%%), which is "
                         "what you get by mistaking the filter for 'not flat'; the filter is "
                         "stricter than that, so 3 is the reading the measurement supports.")
parser.add_argument("--downrange", type=float, default=0.15,
                    help="Symmetric +- bound (m) along the terrain's traversal axis. Keep "
                         "SMALL: past roughly a stride the correct behaviour stops being "
                         "'the same trajectory, corrected' and becomes 'take another step "
                         "first', so a large bound teaches the terrain motion to start early.")
parser.add_argument("--lateral", type=float, default=0.20,
                    help="Symmetric +- bound (m) across it. Tolerates more than downrange -- "
                         "the anchor mask leaves lateral FREE on terrain segments.")
parser.add_argument("--yaw", type=float, default=0.15,
                    help="Symmetric +- bound (rad) on heading. The mask PINS yaw on terrain, "
                         "so a yaw offset is a correction the generator must undo.")
parser.add_argument("--max_source", type=int, default=0,
                    help="Cap source windows processed (0 = all), sampled EVENLY across the "
                         "recording rather than as a prefix. Augmented output is this "
                         "times --num_augments.")
parser.add_argument("--batch", type=int, default=0,
                    help="Windows per render pass (0 = use num_envs).")
parser.add_argument("--num_envs", type=int, default=0,
                    help="0 = the recording's own num_envs, which is what reproduces the "
                         "tile layout the recorded world poses live in.")
parser.add_argument("--check_only", action="store_true",
                    help="Re-render at ZERO offset and report the diff against the recorded "
                         "depth. Proves placement, tile alignment and sensor refresh before "
                         "any augmented number is trusted. Writes nothing.")
parser.add_argument("--no_depth_dr", action="store_true",
                    help="Disable depth sensor DR (mount randomisation + noise) for this run. "
                         "REQUIRED to interpret --check_only: the recording carries "
                         "extero_corrupted=True, so a fresh render draws different noise and "
                         "the delta-0 diff can never be zero with DR live.")
parser.add_argument("--quat_wxyz", action="store_true",
                    help="Write the root quaternion permuted to wxyz. A/B for the convention: "
                         "the recording states xyzw, the sim API's expectation is what this "
                         "settles empirically against a DR-free delta-0 check.")
parser.add_argument("--clutter", default=None, metavar="SIDECAR.json",
                    help="Re-render against a visual-clutter variant "
                         "(scripts/rsl_rl/author_clutter.py) instead of displacing the robot. "
                         "Implies --visual_only. The sidecar is merged into the SAME library "
                         "the recording names, and adds only ``decor`` overlays, so the "
                         "ground mesh and every tile corner are unchanged and the recorded "
                         "world poses stay exactly as valid as they were -- which is the "
                         "whole reason a recording can be re-rendered at all.")
parser.add_argument("--visual_only", action="store_true",
                    help="Change ONLY what the cameras see: re-render every window at its own "
                         "recorded pose, with no displacement and no reference arithmetic. "
                         "The output differs from the input in depth alone, which is what "
                         "makes the pair a controlled A/B rather than two different datasets.")
parser.add_argument("--augment_flat", action="store_true",
                    help="Also augment windows with NO terrain in their horizon. OFF by "
                         "default and it should stay off: a flat segment is anchor-mask 35 "
                         "(downrange/lateral/yaw all FREE), i.e. ego-anchored, so under a "
                         "displaced robot the correct reference MOVES WITH IT. Holding the "
                         "world clip fixed there would teach the generator to steer back to "
                         "an arbitrary world position on flat ground -- the opposite of what "
                         "the mask says, and worse than not augmenting at all.")
parser.add_argument("--env_seed", type=int, default=42,
                    help="Env seed; MUST equal the recording's --seed or the terrain tiles "
                         "are laid out differently and absolute world poses are meaningless.")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

if args_cli.clutter:
    args_cli.visual_only = True
if args_cli.visual_only:
    if args_cli.check_only:
        raise SystemExit("--visual_only and --check_only are the same pass with and without a "
                         "writer; pick one. --check_only reports the delta and writes "
                         "nothing, --visual_only writes the re-rendered recording.")
    # The displacement bounds DEFAULT to non-zero, so the test is whether they were asked
    # for, not whether they are set. Inheriting a default is not intent; typing the flag is.
    _asked = [n for n in ("downrange", "lateral", "yaw") if f"--{n}" in sys.argv]
    if _asked:
        raise SystemExit(
            f"--visual_only was given {_asked} explicitly. Displacing the robot AND changing "
            "the scene is a legitimate pass, but the two together make an augmented pair that "
            "differs in two variables, so neither can be attributed -- ask for it in a "
            "separate run rather than by combining flags here.")
    args_cli.downrange = args_cli.lateral = args_cli.yaw = 0.0

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from isaaclab.managers import SceneEntityCfg  # noqa: E402

# After the SimulationApp, or the per-link contact sensors find no bodies (same as play.py).
import robot_rl.sensors._recursive_contact_sensor_impl  # noqa: F401,E402
import robot_rl.tasks  # noqa: F401,E402

# Module scope, NOT inside the render loop. A loop body is not a scope, so a
# ``from ... import mdp`` down there makes ``mdp`` a local of main() for the WHOLE function --
# and the sensor-scoping added later reads it ~300 lines earlier, which is an
# UnboundLocalError on the first call. The script has not been runnable since.
from robot_rl.tasks.manager_based.robot_rl import mdp  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from motion_gen.clips.clip_io import load_clip_library  # noqa: E402
from recorded_data_io import (  # noqa: E402
    RecordedWindowWriter,
    load_recorded_windows,
)


def term_slice(names, dims, target: str) -> slice:
    """Column slice of a named term inside a concatenated node row."""
    off = 0
    for n, d in zip(names, dims):
        if n == target:
            return slice(off, off + d)
        off += d
    raise KeyError(f"{target!r} not in {names}")


def yaw_of(quat_xyzw: np.ndarray) -> np.ndarray:
    """Heading (rad) from an xyzw quaternion array ``[..., 4]``."""
    x, y, z, w = (quat_xyzw[..., i] for i in range(4))
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_to_quat(yaw: np.ndarray) -> np.ndarray:
    """xyzw quaternion for a pure-yaw rotation."""
    out = np.zeros(yaw.shape + (4,), dtype=np.float32)
    out[..., 2] = np.sin(yaw / 2.0)
    out[..., 3] = np.cos(yaw / 2.0)
    return out


def quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of xyzw quaternions, ``a * b``."""
    ax, ay, az, aw = (a[..., i] for i in range(4))
    bx, by, bz, bw = (b[..., i] for i in range(4))
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


#: Bit index of each anchor DOF, mirroring ``motion_gen.world_lift._ANCHOR_DOF_NAMES``.
#: bit SET = that DOF is FREE, i.e. the clip follows the robot along it.
_DOF_DOWNRANGE, _DOF_LATERAL, _DOF_YAW = 0, 1, 5


def apply_free_axis_shift(
    ref: np.ndarray, node_mask: np.ndarray, robot_pos: np.ndarray, terrain_yaw: np.ndarray,
    d_dr: np.ndarray, d_lat: np.ndarray, d_yaw: np.ndarray,
) -> np.ndarray:
    """Move each reference node by the part of the robot's displacement its mask leaves FREE.

    This is what makes the augmentation teach the MASK rather than "everything is world-fixed".
    A node whose downrange/lateral/yaw are all free (mask 35, flat walking) receives the WHOLE
    displacement, so after re-anchoring its target is unchanged -- displacing the robot on flat
    ground is a genuine no-op, exactly as the ego-anchored semantics say. A stairs/box node
    (mask 2) is free only in lateral, so it follows the robot sideways but stays put downrange
    and in yaw, which is precisely the correction the generator has to learn. A transition
    window therefore gets the full shift across its flat approach and a lateral-only shift
    across its terrain, in ONE window.

    Args:
        ref: ``(n, F, D)`` reference nodes, WORLD frame, ``[pos3, quat4(xyzw), lin3, ang3, ...]``.
        node_mask: ``(n, F)`` uint8 anchor mask per node (bit SET = FREE).
        robot_pos: ``(n, 3)`` the robot's ORIGINAL world position (the yaw rotation centre).
        terrain_yaw: ``(n,)`` heading of each window's terrain frame; the mask names its
            downrange/lateral axes in THIS frame, not the world's.
        d_dr, d_lat, d_yaw: ``(n,)`` the displacement applied to the robot.

    Returns:
        ``(n, F, D)`` with node positions, orientations and velocities moved.
    """
    out = ref.copy()
    free_dr = ((node_mask >> _DOF_DOWNRANGE) & 1).astype(np.float64)      # (n, F)
    free_lat = ((node_mask >> _DOF_LATERAL) & 1).astype(np.float64)
    free_yaw = ((node_mask >> _DOF_YAW) & 1).astype(np.float64)

    dr_n = d_dr[:, None] * free_dr                                        # (n, F)
    lat_n = d_lat[:, None] * free_lat
    yaw_n = d_yaw[:, None] * free_yaw

    ty = terrain_yaw[:, None]
    dx = dr_n * np.cos(ty) - lat_n * np.sin(ty)                           # terrain -> world
    dy = dr_n * np.sin(ty) + lat_n * np.cos(ty)

    # Yaw rotates the node about the ROBOT, so a free-yaw node keeps its pose relative to it.
    c, sn = np.cos(yaw_n), np.sin(yaw_n)
    rel = out[..., 0:2] - robot_pos[:, None, 0:2]
    out[..., 0] = robot_pos[:, None, 0] + rel[..., 0] * c - rel[..., 1] * sn + dx
    out[..., 1] = robot_pos[:, None, 1] + rel[..., 0] * sn + rel[..., 1] * c + dy

    qy = np.zeros(node_mask.shape + (4,))                                 # xyzw yaw quaternion
    qy[..., 2] = np.sin(yaw_n / 2.0)
    qy[..., 3] = np.cos(yaw_n / 2.0)
    out[..., 3:7] = quat_mul_xyzw(qy, out[..., 3:7])
    for a, b in ((7, 10), (10, 13)):                                      # lin + ang velocity
        vx, vy = out[..., a], out[..., a + 1]
        out[..., a] = vx * c - vy * sn
        out[..., a + 1] = vx * sn + vy * c
    return out


def main() -> int:
    """Re-render depth at displaced robot poses and write the augmented recording."""
    import h5py

    rng = np.random.default_rng(args_cli.seed)
    data, meta = load_recorded_windows(args_cli.data)
    W = int(data["history_state"].shape[0])
    H = int(data["history_state"].shape[1])

    lib = meta.clip_library_path
    if not lib:
        raise SystemExit("recording carries no clip_library_path; re-record with the current "
                         "record_data.py.")
    if meta.clip_library_sha256:
        import hashlib
        got = hashlib.sha256(open(lib, "rb").read()).hexdigest()
        if got != meta.clip_library_sha256:
            raise SystemExit(
                f"clip library '{lib}' is sha256 {got[:12]} but the recording was made "
                f"against {meta.clip_library_sha256[:12]}. The recorded poses are ABSOLUTE "
                "world, so a different library puts the tiles somewhere else and every "
                "re-rendered depth image would be of the wrong geometry.")

    n_env = args_cli.num_envs or int(meta.num_envs)
    batch = args_cli.batch or n_env
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=n_env)
    # The env SEED must match the recording's. ClipTerrainImporter's tile->world layout is
    # built at env creation, so an unseeded run lays the tiles out differently and the
    # recording's ABSOLUTE world poses then land on different geometry entirely.
    env_cfg.seed = int(args_cli.env_seed)
    env_cfg.commands.traj_ref.path = lib
    env_cfg.commands.traj_ref.clip_device = "cpu"
    terrain = getattr(getattr(env_cfg, "scene", None), "terrain", None)
    if terrain is not None and hasattr(terrain, "clip_library_path"):
        terrain.clip_library_path = lib
        if args_cli.clutter:
            # The importer merges the sidecar itself, and verifies it was authored against
            # this library. Nothing about the ground changes, so the sha check above -- which
            # exists because the recorded poses are absolute world -- still holds and still
            # means what it says.
            terrain.decor_overlay_path = args_cli.clutter
    if args_cli.no_depth_dr:
        # DERIVED from the event functions rather than a name list. An earlier hardcoded guess
        # ("randomize_depth_mount_*") matched nothing, so camera EXTRINSICS/INTRINSICS
        # randomization stayed live through every "DR off" run -- a per-episode camera pose
        # draw, which is why the re-render came out vertically shifted while the scene
        # structure matched. A list cannot go stale if there is no list.
        killed = []
        for _name, _term in list(vars(env_cfg.events).items()):
            _func = getattr(_term, "func", None)
            if getattr(_func, "__module__", "").endswith("mdp.events.depth"):
                setattr(env_cfg.events, _name, None)
                killed.append(_name)
        if not killed:
            raise SystemExit("--no_depth_dr: this task declares no depth DR events at all; "
                             "either it is not a depth task or the events moved module.")
        print(f"[augment] depth DR OFF: {killed}")
    # A re-render must reproduce the SOURCE exactly, so the resolution comes from the
    # recording's own camera contract -- not from a flag, and not from whatever the task
    # happens to default to. When the recording was made at the generator's own resolution
    # (record_data.py --fm_depth_resolution), that means building the same second pair here.
    _rec_cams = list(getattr(meta, "depth_camera_spec", None) or [])
    if _rec_cams:
        _res = {(int(c["width"]), int(c["height"])) for c in _rec_cams}
        if len(_res) != 1:
            raise SystemExit(f"the recording's cameras sample at {sorted(_res)}; one "
                             "recording's cameras share one resolution.")
        from robot_rl.tasks.manager_based.robot_rl.g1.g1_depth_env_cfg import (
            add_fm_depth_pair,
        )
        add_fm_depth_pair(env_cfg, *_res.pop())

    # Same for the generator's own dense scanner, and for the same reason: the recording was
    # made on a task PLUS `record_data.py --fm_height_scan`, so the bare task carries only the
    # policy's 176-ray scanner and a re-cast would be of the wrong grid. DERIVED from the
    # recording's own contract rather than re-declared, so the two cannot disagree by a typo.
    _rec_scan = list(getattr(meta, "height_scan_spec", None) or [])
    if _rec_scan:
        from robot_rl.flow_matching.online import fm_scan_geometry
        _geom = fm_scan_geometry(_rec_scan)
        if _geom is not None:
            from robot_rl.tasks.manager_based.robot_rl.g1.g1_fm_scan_env_cfg import (
                add_fm_height_scan,
            )
            print(f"[augment] rebuilding the recording's scanner: {_geom['size'][0]}x"
                  f"{_geom['size'][1]} m @ {_geom['resolution']} m "
                  f"({_geom['height']}x{_geom['width']})")
            add_fm_height_scan(env_cfg, size=tuple(_geom["size"]),
                               resolution=_geom["resolution"],
                               offset_xy=_geom["offset_xy"])

    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    robot = u.scene["robot"]
    def to_torch(x):
        """Sim data view (torch / warp / numpy) as a torch tensor."""
        if isinstance(x, torch.Tensor):
            return x
        try:
            import warp as wp
            return wp.to_torch(x)
        except Exception:
            return torch.as_tensor(np.asarray(x), device=u.device)

    u.reset()

    # WHICH cameras to re-render, derived from the observation terms rather than named. The
    # recorded ``depth`` array came from whichever pair the recorder was scoped to, so this
    # has to be the same scoping -- re-rendering the policy's cameras into a recording made
    # from the generator's would be silently wrong (or, if the widths differ, loudly wrong at
    # the writer, which is the better half of the outcome).
    _render_groups = mdp.fm_depth_groups(u)
    _render_sensors = [name for name, _ in mdp.find_depth_obs_terms(u, _render_groups)]
    if not _render_sensors:
        raise SystemExit("no depth observation terms on this task; there is nothing to "
                         "re-render. --task must be a depth task.")
    _rec_spec = list(getattr(meta, "depth_camera_spec", None) or [])
    if _rec_spec:
        _live = mdp.depth_camera_specs(u, _render_groups)
        _live_res = [(int(c["width"]), int(c["height"])) for c in _live]
        _rec_res = [(int(c["width"]), int(c["height"])) for c in _rec_spec]
        if _live_res != _rec_res:
            raise SystemExit(
                f"this task samples depth at {_live_res} but the recording was made at "
                f"{_rec_res}. A re-render must reproduce the source EXACTLY -- pass "
                "--fm_depth_resolution to record_data.py's resolution, or point --task at "
                "the task the recording came from."
            )
    print(f"[augment] re-rendering {_render_sensors} "
          f"(obs group(s): {_render_groups or 'all'})")

    # The HEIGHT SCAN has to be re-cast too. Every per-window array is otherwise copied
    # straight from the un-displaced source (see the ``out = {...}`` comprehension below), so
    # before this the scan was carried through BIT-IDENTICAL while the anchor moved by ~0.18 m
    # -- the sensor said "the riser is here" and the target said "correct back to there". That
    # is the very shortcut this script exists to remove, inverted, and it is silently worse
    # than not augmenting at all. Harmless for a depth-conditioned FM, fatal for a
    # scan-conditioned one.
    #
    # WHICH sensor, DERIVED from the recorded width rather than named: the generator may scan
    # on its own grid, in which case the recorded ``height_scan`` came from THAT sensor and
    # re-casting the policy's would produce a correctly-shaped array of the wrong geometry.
    _scan_sensor = None
    if "height_scan" in data:
        _rec_k = int(data["height_scan"].shape[1])
        _cands = {}
        for _sname in u.scene.sensors:
            if _sname in _render_sensors:
                continue          # a depth camera, already handled above
            try:
                _cands[_sname] = int(mdp.height_scan(u, SceneEntityCfg(_sname)).shape[1])
            except Exception:
                continue          # not a height-scanning RayCaster
        _match = [n for n, k in _cands.items() if k == _rec_k]
        if not _match:
            raise SystemExit(
                f"the recording carries a {_rec_k}-ray 'height_scan' but this task's scanners "
                f"cast {_cands or 'nothing'}. A re-cast must reproduce the source EXACTLY, so "
                "point --task at the task the recording came from (or pass the same "
                "--fm_height_scan geometry it was recorded with).")
        if len(_match) > 1:
            raise SystemExit(
                f"{_match} all cast {_rec_k} rays, so the recorded scan cannot be attributed "
                "to one of them. Give them distinct grids, or drop the ambiguous sensor.")
        _scan_sensor = _match[0]
        _scan_cfg = SceneEntityCfg(_scan_sensor)
        # The recording's own geometry must match what this env casts. Same doctrine as the
        # depth resolution check above: a re-cast has to reproduce the SOURCE, and a scan of
        # the right ray count over the wrong extent is invisible to every width assertion.
        _rec_scan_spec = list(getattr(meta, "height_scan_spec", None) or [])
        if _rec_scan_spec:
            _live_scan = mdp.height_scan_specs(u, _scan_sensor)
            _keys = ("resolution", "size", "ordering", "height", "width", "offset_pos",
                     "ray_alignment", "sensor_offset")
            _want = {k: _rec_scan_spec[0].get(k) for k in _keys if k in _rec_scan_spec[0]}
            _got = {k: _live_scan[0].get(k) for k in _keys if k in _rec_scan_spec[0]}
            if _want != _got:
                raise SystemExit(
                    f"the recording's scan geometry differs from this task's:\n"
                    f"  recording: {_want}\n  task     : {_got}\n"
                    "A re-cast must reproduce the source EXACTLY -- pass the same "
                    "--fm_height_scan geometry record_data.py used, or point --task at the "
                    "task the recording came from.")
        print(f"[augment] re-casting height scan from {_scan_sensor!r} ({_rec_k} rays)")
    else:
        print("[augment] recording carries no 'height_scan'; nothing to re-cast")

    # Joint ORDER: the recording's joint_pos columns are meaningless unless they line up with
    # this articulation's. A silent mismatch would scramble the posture, which moves the
    # torso-mounted cameras as well as the self-scan -- i.e. it would corrupt exactly what we
    # are trying to render.
    rec_joints = list(meta.state_joint_names)
    if rec_joints != list(robot.joint_names):
        raise SystemExit(
            "joint order mismatch between the recording and the robot.\n"
            f"  recording: {rec_joints[:4]} ... ({len(rec_joints)})\n"
            f"  robot    : {list(robot.joint_names)[:4]} ... ({len(robot.joint_names)})")

    # clip_id -> terrain_id -> the envs sitting on that tile. This is the mapping the RL's own
    # reset uses, so it is exact; picking a tile by PROXIMITY to the recorded pose would be a
    # guess that fails whenever the robot is far along its tile.
    terr = u.scene.terrain
    env_tid = to_torch(terr.env_terrain_id).detach().cpu().numpy().reshape(-1)
    env_org = to_torch(terr.env_origins).detach().cpu().numpy().reshape(-1, 3)
    with h5py.File(lib, "r") as _lf:
        clip_tid = _lf["terrain_id"][:]
    env_tyaw = to_torch(terr.env_terrain_yaw).detach().cpu().numpy().reshape(-1)
    _libm = load_clip_library(lib, device="cpu",
                              keys=["anchor_mask", "anchor_mask_t", "clip_len"])
    amt_np = _libm.anchor_mask_per_step().numpy()          # (num_clips, T) uint8, 255 resolved
    clen_np = _libm.clip_len_or_full().numpy()
    envs_for_tid: dict[int, np.ndarray] = {
        int(t): np.flatnonzero(env_tid == int(t)) for t in np.unique(env_tid)
    }
    print(f"[augment] {len(envs_for_tid)} distinct terrain ids across {len(env_tid)} envs")

    names, dims = list(meta.state_term_names), list(meta.state_term_dims)
    s_pos, s_quat = term_slice(names, dims, "root_pos_w"), term_slice(names, dims, "root_quat_w")
    s_jp = term_slice(names, dims, "joint_pos")

    # SPREAD, never a prefix. The recording is written env-batch by env-batch, so the first
    # N windows are the OPENINGS of episodes -- taking a prefix samples one narrow band of
    # clip_step and silently answers a different question than the one asked (it read the
    # first 36 steps of a 2-6 s clip here, which made every downrange obstacle look far away).
    src = (np.arange(W) if not args_cli.max_source or args_cli.max_source >= W
           else np.linspace(0, W - 1, args_cli.max_source).astype(np.int64))
    if not args_cli.check_only and not args_cli.visual_only and not args_cli.augment_flat:
        # Keep only windows whose horizon CONTAINS terrain (>=1 anchor-mask-pinned node):
        # terrain-only segments and transitions. Pure-flat windows are ego-anchored and must
        # not be given a world-fixed correction -- see --augment_flat. A transition window
        # keeps its flat approach, which is wanted: correcting during the run-up to the
        # terrain is exactly the behaviour that has to be learned.
        from motion_gen.world_lift import anchor_mask_is_pinned
        _lib = load_clip_library(lib, device="cpu",
                                 keys=["anchor_mask", "anchor_mask_t", "clip_len"])
        _pin = anchor_mask_is_pinned(_lib.anchor_mask_per_step()).numpy()
        _clen = _lib.clip_len_or_full().numpy()
        _cid = np.asarray(data["clip_id"]); _cst = np.asarray(data["clip_step"])
        _F = int(data["ref_future"].shape[1])
        keep_src = np.array([
            bool(_pin[_cid[i], _cst[i]:min(_cst[i] + _F, _clen[_cid[i]])].any()) for i in src
        ])
        n_before = len(src)
        src = src[keep_src]
        print(f"[augment] terrain filter: {len(src)}/{n_before} source windows have terrain in "
              f"their horizon ({100.0 * len(src) / max(n_before, 1):.1f}%); pure-flat windows "
              "are skipped (they are ego-anchored and must move WITH the robot).")
        if len(src) == 0:
            raise SystemExit("no windows contain terrain; nothing to augment.")
    reps = 1 if (args_cli.check_only or args_cli.visual_only) else int(args_cli.num_augments)
    hs_all = data["history_state"]

    # CAMERA LATENCY. A recording made after 2026-08-21 carries the last few depth frames
    # and the state each was captured with, and the FM samples one of them per window. They
    # cannot be copied through: they would be the UN-displaced imagery, and since the shipped
    # latency distribution gives lag 0 zero weight, training would see only those and the
    # whole augmentation would be silently undone. Each is re-rendered from its own recorded
    # pose plus the SAME rigid displacement.
    _has_lag = "depth_lag" in data and "depth_lag_state" in data
    if "depth_lag" in data and "depth_lag_state" not in data:
        raise SystemExit(
            "recording has 'depth_lag' but no 'depth_lag_state', so a lagged frame has no "
            "capture pose to re-render from. Re-record with the current record_data.py.")
    _n_lag = int(data["depth_lag"].shape[1]) if _has_lag else 0
    print(f"[augment] lagged depth frames: {_n_lag if _has_lag else 'none in this recording'}"
          + (" (each re-rendered at its own displaced pose)" if _has_lag else ""))

    writer = None
    if not args_cli.check_only:
        if not args_cli.out:
            raise SystemExit("--out is required unless --check_only.")
        depth_shape = (int(data["depth"].shape[1]), int(data["depth"].shape[2]))
        writer = RecordedWindowWriter(
            args_cli.out, len(src) * reps, H, int(hs_all.shape[2]),
            int(data["ref_future"].shape[1]),
            int(data["height_scan"].shape[1]) if "height_scan" in data else 0,
            int(data["cmd_future"].shape[2]) if "cmd_future" in data else 0,
            meta, depth_shape=depth_shape, depth_lag_steps=max(_n_lag - 1, 0),
            tag_distractors=True, tag_families=True,
            tag_clip_index=True, record_history_ref=True,
            # Inherit the SOURCE's storage dtypes. Without this a float16 recording's scan
            # comes back float32 in the augmented half -- doubling it on disk and in the
            # dataset, and making the two halves of one training set disagree about their
            # own precision after `train.py` concatenates them.
            dtypes={k: str(v.dtype) for k, v in data.items()
                    if isinstance(v, np.ndarray) and v.dtype != np.float32},
        )

    diffs = []
    scan_diffs = []
    scan_miss_mismatch = []
    scan_miss_frac = []
    samples = []
    perwin = []
    allren = []
    allsrc = []
    n_done = 0
    # Schedule passes so NO window is dropped. Only one window per env can be placed per
    # render pass (they share the env's robot), and after the terrain filter the surviving
    # windows are NOT evenly spread across envs -- so slicing the list into fixed batches
    # leaves duplicate envs in a batch, which the placement loop then skips. That silently
    # lost 51% of the output on the first full run. Bucket by env instead and take the k-th
    # entry of every env's queue as pass k: every window is placed exactly once.
    by_env: dict[int, list[int]] = {}
    src_env_all = np.asarray(data["env_id"]).astype(int)
    for _rep in range(reps):
        for i in src:
            by_env.setdefault(int(src_env_all[i]), []).append(int(i))
    n_pass = max((len(v) for v in by_env.values()), default=0)
    passes = [[q[k] for q in by_env.values() if k < len(q)] for k in range(n_pass)]
    print(f"[augment] {sum(len(p) for p in passes)} placements over {n_pass} passes "
          f"(<= {len(by_env)} envs each; bucketed by env so none are skipped)")
    for _pi, _plist in enumerate(passes):
        for lo in range(0, len(_plist), batch):
            idx = np.asarray(_plist[lo:lo + batch])
            k = len(idx)
            snap = hs_all[idx, H - 1, :].astype(np.float64)
            pos = snap[:, s_pos].copy()
            quat = snap[:, s_quat].copy()
            jp = snap[:, s_jp].copy()

            if args_cli.check_only or args_cli.visual_only:
                # Exactly zero, and never drawn: under --visual_only the whole point is that
                # the robot is where the recording put it, so the pair differs in depth
                # alone. Drawing and then adding zero would still run the reference
                # arithmetic below, which is an f64 round trip that need not be bit-exact.
                d_dr = d_lat = d_yaw = np.zeros(k)
            else:
                d_dr = rng.uniform(-args_cli.downrange, args_cli.downrange, k)
                d_lat = rng.uniform(-args_cli.lateral, args_cli.lateral, k)
                d_yaw = rng.uniform(-args_cli.yaw, args_cli.yaw, k)
            # Offsets are in the ROBOT's own heading frame -- on the axis-aligned tiles built
            # today this equals the terrain frame the anchor mask names its DOFs in, and it
            # stays meaningful if a tile is ever rotated.
            hd = yaw_of(quat)
            new_pos = pos.copy()
            new_pos[:, 0] += d_dr * np.cos(hd) - d_lat * np.sin(hd)
            new_pos[:, 1] += d_dr * np.sin(hd) + d_lat * np.cos(hd)
            new_quat = quat_mul_xyzw(yaw_to_quat(d_yaw).astype(np.float64), quat)

            # Place: full pose (the cameras are torso-mounted, so roll/pitch matter to the
            # render) and the recorded joint pose. Envs beyond k keep whatever they had; only
            # the first k rows are read back.
            # Route each window to an env whose tile has the SAME terrain_id, keeping the
            # pose TILE-LOCAL. Tiles of one terrain_id are geometrically identical, so which
            # instance we land on cannot change the depth image -- only the terrain_id can.
            # The source env is RECORDED, so there is no instance to infer. clip_id ->
            # terrain_id is still checked below, as a guard that the layout really matches.
            src_envs = np.asarray(data["env_id"])[idx].astype(int)
            tids = clip_tid[np.asarray(data["clip_id"])[idx]]
            slots, keep = [], []
            used_envs: set[int] = set()
            for j, t in enumerate(tids):
                e = int(src_envs[j])
                if e < 0 or e >= len(env_tid):
                    continue
                if int(env_tid[e]) != int(t):
                    raise SystemExit(
                        f"env {e} sits on terrain {int(env_tid[e])} but window {j}'s clip "
                        f"wants terrain {int(t)}. The tile layout does not match the "
                        "recording -- check that --env_seed and --num_envs equal the "
                        "recording's, or the recorded world poses are meaningless here.")
                if e in used_envs:
                    continue                      # safety net; the schedule already prevents it
                used_envs.add(e)
                slots.append((e, e))              # place back into the very env it came from
                keep.append(j)
            if not keep:
                continue
            keep = np.asarray(keep)
            tgt_env = np.array([a for a, _ in slots])
            src_env = np.array([b for _, b in slots])
            local = new_pos[keep] - env_org[src_env]
            place_pos = local + env_org[tgt_env]

            # ONE env row set for every pose this window renders at: the snap and, when the
            # recording carries them, each lagged camera frame.
            _root0 = to_torch(robot.data.root_state_w)
            ti = torch.as_tensor(tgt_env, device=_root0.device, dtype=torch.long)

            def _render_at(_pp, _qq, _jj, *, first=False):
                """Place the robot at one pose and read the sensors. Returns ``(depth, scan)``.

                Lifted verbatim out of the loop so the lagged camera frames can be rendered
                from their OWN poses without duplicating it. Every line of the settle-and-read
                sequence below is load-bearing and was arrived at by measurement -- the step
                count, re-asserting the state before each step, and reading twice keeping the
                second. Do not condense it.

                Args:
                    _pp: ``[n, 3]`` world position to place the root at.
                    _qq: ``[n, 4]`` root orientation, xyzw (the recording's convention).
                    _jj: ``[n, nj]`` joint positions.
                    first: Run the one-time quat/pose diagnostics. Only the snap pose should,
                        since they compare against the recorded snap.

                Returns:
                    ``(depth [n, C, HW], scan [n, K] or None)``.
                """
                root = to_torch(robot.data.root_state_w).clone()
                root[ti, 0:3] = torch.as_tensor(_pp, device=root.device, dtype=root.dtype)
                # CONVENTION: the recording states quats as xyzw (meta.state_quat_format), while
                # `root_state_w` carries them wxyz. Writing one into the other silently yields a
                # ~180 deg rotation for a near-upright robot -- the camera points elsewhere and the
                # body appears mirrored front-to-back, which is exactly what the first visual diff
                # showed. Converted here, and ROUND-TRIPPED below so the convention can never be
                # guessed wrong again.
                q_write = (_qq[:, [3, 0, 1, 2]] if args_cli.quat_wxyz
                           else _qq)
                root[ti, 3:7] = torch.as_tensor(q_write, device=root.device, dtype=root.dtype)
                root[ti, 7:] = 0.0
                robot.write_root_state_to_sim(root)
                if first and not hasattr(main, "_quat_checked"):
                    # SEMANTIC check, not a round trip: does the robot now HOLD the recorded
                    # orientation? Comparing what we wrote against what we read back proves only
                    # that the API stores what it is given -- it round-trips under ANY convention
                    # and so cannot catch a convention error at all.
                    back = to_torch(robot.data.root_quat_w)[ti].detach().cpu().numpy()
                    rec_q = _qq
                    d_same = np.abs(np.abs((back * rec_q).sum(-1)) - 1.0).max()
                    d_conv = np.abs(np.abs((back * rec_q[:, [3, 0, 1, 2]]).sum(-1)) - 1.0).max()
                    print(f"[augment] quat check: |read . recorded_xyzw|-1 = {d_same:.5f}, "
                          f"vs wxyz-permuted = {d_conv:.5f}  (the SMALLER names the convention "
                          f"root_quat_w reports in)")
                    main._quat_checked = True
                jpos = to_torch(robot.data.joint_pos).clone()
                jpos[ti] = torch.as_tensor(_jj, device=jpos.device, dtype=jpos.dtype)
                robot.write_joint_position_to_sim(jpos)
                robot.write_joint_velocity_to_sim(torch.zeros_like(jpos))
                # Target the written posture so the PD does not pull away during the step.
                # MEASURED: this changes nothing -- the depth output is bit-identical with and
                # without it and the drift stays at 3.6 mrad median / 47 mrad max -- so the
                # residual joint motion is the solver settling contacts, not the actuators.
                # Kept because holding the pose we asked for is the correct intent regardless.
                robot.set_joint_position_target(jpos)
                # Step ENOUGH times for the ray-casters' update_period to elapse, re-asserting the
                # pose before each step so physics cannot drift it.
                #
                # One step per batch was the bug behind the bimodal delta-0 result: the casters run
                # at update_period 0.02 against a smaller physics_dt, so a single step ticks them
                # only every other batch. Since windows are written grouped by snap event and the
                # batch size equals num_envs, each batch IS one snap index -- which made the
                # good/bad split alternate exactly with window-position-within-episode (pos 0: 1%
                # good, pos 1: 95%, pos 2: 1%, pos 3: 96%). Half the frames were the PREVIOUS
                # batch's image, which no amount of placement accuracy could fix.
                _period = float(getattr(u.scene.sensors["depth_camera_near"].cfg, "update_period",
                                        0.0) or 0.0)
                # x2: the casters lag one PASS behind this loop regardless of how much sim time
                # elapses inside it, so a single pass reads the previous batch's image for about
                # half the windows. Doubling the cycle costs one extra step and removes the lag.
                _n_steps = 2 * max(2, int(round(_period / float(u.physics_dt))) + 1)
                for _ in range(_n_steps):
                    robot.write_root_state_to_sim(root)
                    robot.write_joint_position_to_sim(jpos)
                    robot.write_joint_velocity_to_sim(torch.zeros_like(jpos))
                    u.sim.step(render=True)
                u.scene.update(u.physics_dt)

                if first and not hasattr(main, "_pose_checked"):
                    # Did the robot actually END UP where we put it? Compares the ACHIEVED root
                    # pose against the recorded one, so a depth mismatch can be attributed to the
                    # placement or exonerated from it. Saved, not printed: IsaacSim eats stdout.
                    aq = to_torch(robot.data.root_quat_w)[ti].detach().cpu().numpy()
                    ap = to_torch(robot.data.root_pos_w)[ti].detach().cpu().numpy()
                    aj = to_torch(robot.data.joint_pos)[ti].detach().cpu().numpy()
                    lk = to_torch(robot.data.root_link_pos_w)[ti].detach().cpu().numpy()
                    cm = to_torch(robot.data.root_com_pos_w)[ti].detach().cpu().numpy()
                    np.savez((args_cli.out or args_cli.data) + ".pose_check.npz",
                             achieved_quat=aq, wrote_quat=q_write, achieved_pos=ap,
                             wrote_pos=_pp, achieved_jp=aj, wrote_jp=_jj,
                             link_pos=lk, com_pos=cm)
                    main._pose_checked = True
                # FORCE the ray-casters to re-cast. They carry update_period=0.02 and are normally
                # driven by env.step()'s sensor pass; a bare sim.step() leaves them believing they
                # are already current, so the read returns STALE data -- which looked exactly like
                # a placement bug (near-uniform error, max == the far camera's max range).
                # Read the sensors TWICE, keep the SECOND. The casters lag one READ CYCLE:
                # measured, 89% of mismatched frames were bit-equal to the SAME ENV's PREVIOUS
                # placement -- the first read returns the prior batch's image. Extra sim steps do
                # not help (the lag is in the read, not in elapsed time); a second update+read
                # does. This alone was the whole bimodal ~50% delta-0 failure.
                for _read in range(2):
                    u.scene.update(u.physics_dt)
                    # DERIVED from the observation terms, not a hardcoded pair of names: the
                    # generator may sample the same lenses on its own grid, in which case the
                    # recorded ``depth`` came from THAT pair and re-rendering the policy's would
                    # silently produce the wrong array (or the wrong width).
                    for _sname in _render_sensors + ([_scan_sensor] if _scan_sensor else []):
                        u.scene.sensors[_sname].update(u.physics_dt, force_recompute=True)
                    # Same producer the recorder used, so this is the OBSERVATION value,
                    # normalized identically -- anything else would not be like for like.
                    _dep = to_torch(mdp.depth_obs_stack(u, _render_groups))[ti].detach().cpu().numpy()
                    # The scan goes through the RAW obs function, which is exactly how the recorder
                    # reads it (record_data.py calls mdp.height_scan directly) -- so no term noise
                    # and no term clip, on both sides. Same one-read-cycle lag as the cameras, so
                    # it relies on the same "keep the SECOND read" above.
                    _scn = (to_torch(mdp.height_scan(u, _scan_cfg))[ti].detach().cpu().numpy()
                           if _scan_sensor is not None else None)

                return _dep, _scn

            dep, scn = _render_at(place_pos, new_quat[keep], jp[keep], first=True)

            if args_cli.check_only:
                rec = data["depth"][idx[keep]]
                diffs.append(np.abs(dep - rec))
                perwin.append(np.stack([idx[keep],
                                        np.abs(dep - rec).mean(axis=(1, 2))], axis=1))
                allren.append(dep.astype(np.float32))
                allsrc.append(idx[keep])
                if scn is not None:
                    # The scan's delta-0 should be TIGHTER than depth's: it carries no DR and
                    # no noise, so the only residual is float replay of the same cast plus the
                    # recording's storage quantum. A non-zero scan diff at zero offset
                    # therefore means the placement or the tile layout is wrong, not that a
                    # sensor was jittered.
                    #
                    # A MISSED ray is -inf on both sides, and inf - inf is NaN -- which
                    # poisons the mean and reports nothing. Compare the finite rays, and score
                    # the miss PATTERN separately: two casts of the same geometry must miss on
                    # the same rays, and that agreement is itself a placement check.
                    rec_scan = data["height_scan"][idx[keep]].astype(np.float64)
                    fin_new, fin_rec = np.isfinite(scn), np.isfinite(rec_scan)
                    both = fin_new & fin_rec
                    scan_diffs.append(np.abs(scn[both] - rec_scan[both]))
                    scan_miss_mismatch.append(float((fin_new != fin_rec).mean()))
                    scan_miss_frac.append(float((~fin_rec).mean()))
                if len(samples) < 6:      # keep a few raw pairs for visual inspection
                    for j in range(min(6 - len(samples), dep.shape[0])):
                        samples.append((rec[j].copy(), dep[j].copy()))
            else:
                sel_idx = idx[keep]
                # depth_lag* are EXCLUDED here and set explicitly below. Copying them is
                # the failure this whole branch exists to prevent, so it must not be the
                # default that a future key silently inherits.
                out = {kk: np.asarray(v[sel_idx]).copy() for kk, v in data.items()
                       if isinstance(v, np.ndarray) and v.shape[:1] == (W,)
                       and kk not in ("depth_lag", "depth_lag_state")}
                out["depth"] = dep.astype(np.float32)
                if _has_lag:
                    # Lag 0 IS the snap frame, so it is assigned rather than re-rendered --
                    # the identity `depth_lag[:, 0] == depth` then cannot drift apart.
                    _ls = np.asarray(data["depth_lag_state"][sel_idx])
                    _out_lag = np.empty((len(sel_idx), _n_lag) + dep.shape[1:], dtype=np.float32)
                    _out_lag[:, 0] = dep.astype(np.float32)
                    _new_ls = _ls.copy()
                    _new_ls[:, 0, s_pos] = place_pos.astype(_ls.dtype)
                    _new_ls[:, 0, s_quat] = new_quat[keep].astype(_ls.dtype)
                    # ONE rigid offset for the whole window, resolved in the SNAP's heading
                    # frame -- the counterfactual is "the robot had been displaced for this
                    # window", not "each frame was displaced in its own frame".
                    _c, _sn = np.cos(hd[keep]), np.sin(hd[keep])
                    _ddr, _dlat, _dyaw = d_dr[keep], d_lat[keep], d_yaw[keep]
                    for _k in range(1, _n_lag):
                        _lp = _ls[:, _k, s_pos].astype(np.float64).copy()
                        _lp[:, 0] += _ddr * _c - _dlat * _sn
                        _lp[:, 1] += _ddr * _sn + _dlat * _c
                        _lq = quat_mul_xyzw(yaw_to_quat(_dyaw).astype(np.float64),
                                            _ls[:, _k, s_quat].astype(np.float64))
                        _lp = (_lp - env_org[src_env]) + env_org[tgt_env]
                        _dk, _ = _render_at(_lp, _lq, _ls[:, _k, s_jp].astype(np.float64))
                        _out_lag[:, _k] = _dk.astype(np.float32)
                        _new_ls[:, _k, s_pos] = _lp.astype(_ls.dtype)
                        _new_ls[:, _k, s_quat] = _lq.astype(_ls.dtype)
                    out["depth_lag"] = _out_lag
                    out["depth_lag_state"] = _new_ls
                # Overwrite, do NOT inherit: the comprehension above just copied the
                # un-displaced scan. Keep the source dtype so an f16 recording round-trips.
                if scn is not None:
                    out["height_scan"] = scn.astype(data["height_scan"].dtype)
                # Move each reference node by the FREE part of the displacement, per its own
                # anchor mask. Without this the target says "every axis is world-fixed" and the
                # generator would learn to correct even along axes the mask leaves free.
                _F = out["ref_future"].shape[1]
                _c = np.asarray(data["clip_id"])[sel_idx]
                _s0 = np.asarray(data["clip_step"])[sel_idx]
                _steps = np.minimum(_s0[:, None] + np.arange(_F)[None, :],
                                    (clen_np[_c] - 1)[:, None])
                _nm = amt_np[_c[:, None], _steps]                       # (n, F) per-node mask
                out["ref_future"] = apply_free_axis_shift(
                    out["ref_future"].astype(np.float64), _nm, pos[keep],
                    env_tyaw[tgt_env], d_dr[keep], d_lat[keep], d_yaw[keep],
                ).astype(np.float32)
                # history_ref's LAST row is the applied reference AT THE SNAP, which is by
                # definition ref_future node 0 -- the dataset asserts the identity. Move it by
                # the same node-0 shift so the assertion still means something. It cannot leak
                # the offset into the model: the history block is rows H-1-n .. H-2, which
                # EXCLUDES this row, and it feeds nothing but that check.
                out["history_ref"][:, H - 1, :] = apply_free_axis_shift(
                    out["history_ref"][:, H - 1:H, :].astype(np.float64), _nm[:, :1],
                    pos[keep], env_tyaw[tgt_env], d_dr[keep], d_lat[keep], d_yaw[keep],
                )[:, 0, :].astype(np.float32)
                # ONLY the snap anchor moves: row H-1's position + yaw. History rows and
                # history_ref stay put so the offset cannot leak in through them.
                out["history_state"][:, H - 1, s_pos] = place_pos.astype(np.float32)
                out["history_state"][:, H - 1, s_quat] = new_quat[keep].astype(np.float32)
                writer.write_batch(out)
            n_done += len(keep)
            if n_done % (batch * 8) < batch:
                print(f"[augment] {n_done}/{len(src) * reps}")

    if args_cli.check_only:
        d = np.concatenate([x.reshape(-1) for x in diffs])
        # To a FILE, not just stdout: IsaacSim's exit path discards buffered Python stdout,
        # so a printed result routinely vanishes from a redirected log even on a clean run.
        stats = dict(mean=float(d.mean()), p50=float(np.median(d)),
                     p99=float(np.percentile(d, 99)), max=float(d.max()),
                     frac_gt_1mm=float((d > 1e-3).mean()), n_windows=int(n_done))
        cpath = (args_cli.out or os.path.splitext(args_cli.data)[0]) + ".delta0_check.npz"
        if allren:
            np.savez_compressed((args_cli.out or args_cli.data) + ".renders.npz",
                                ren=np.concatenate(allren), src=np.concatenate(allsrc))
        if perwin:
            stats["perwin"] = np.concatenate(perwin)      # (n, 2): source index, mean |err|
        if samples:
            stats["rec"] = np.stack([a for a, _ in samples])
            stats["ren"] = np.stack([b for _, b in samples])
        sd = np.concatenate([x.reshape(-1) for x in scan_diffs]) if scan_diffs else None
        if sd is not None and sd.size:
            miss_mm = float(np.mean(scan_miss_mismatch))
            stats.update(scan_mean=float(sd.mean()), scan_p50=float(np.median(sd)),
                         scan_p99=float(np.percentile(sd, 99)), scan_max=float(sd.max()),
                         scan_frac_gt_1mm=float((sd > 1e-3).mean()),
                         scan_miss_mismatch=miss_mm,
                         scan_miss_frac=float(np.mean(scan_miss_frac)))
        elif sd is not None:
            sd = None
        np.savez(cpath, **stats)
        print(f"[augment] wrote {cpath}")
        print(f"\n=== delta-0 ROUND TRIP over {n_done} windows ===")
        print(f"  |re-rendered - recorded| depth:  mean {d.mean():.5f}  p50 "
              f"{np.median(d):.5f}  p99 {np.percentile(d, 99):.5f}  max {d.max():.5f}")
        print(f"  fraction of pixels > 1e-3: {100.0 * (d > 1e-3).mean():.2f}%")
        if sd is not None:
            print(f"  |re-cast - recorded|   scan :  mean {sd.mean():.6f}  p50 "
                  f"{np.median(sd):.6f}  p99 {np.percentile(sd, 99):.6f}  max {sd.max():.6f}"
                  "   [FINITE rays only]")
            print(f"  fraction of rays > 1e-3:   {100.0 * (sd > 1e-3).mean():.3f}%  "
                  f"(the recording stores {data['height_scan'].dtype}; its quantum near 1 m "
                  "is ~1e-3, so this is the floor, not an error)")
            print(f"  missed-ray PATTERN disagreement: "
                  f"{100.0 * float(np.mean(scan_miss_mismatch)):.4f}% of rays "
                  f"(recording misses {100.0 * float(np.mean(scan_miss_frac)):.2f}%) -- two "
                  "casts of the same geometry must miss on the SAME rays")
            print("  (the scan carries no DR and no noise, so its delta-0 should be TIGHTER "
                  "than depth's -- a non-zero scan diff means placement, not jitter)")
        else:
            print("  scan: NOT CHECKED (recording carries no 'height_scan')")
        print("  (near-zero => placement, tile alignment and sensor refresh are all correct)")
    else:
        writer.close()
        print(f"[augment] wrote {args_cli.out}: {n_done} windows "
              f"({len(src)} source x {reps})")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    sys.exit(code)
