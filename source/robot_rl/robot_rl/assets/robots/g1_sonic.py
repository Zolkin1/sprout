# Copyright 2026 Zachary Olkin. All rights reserved.

"""SONIC's own Unitree G1, for running the SONIC baseline on the body it was trained on.

Transcribed from ``gear_sonic/envs/manager_env/robots/g1.py::G1_CYLINDER_MODEL_12_DEX_CFG`` in
the NVlabs/GR00T-WholeBodyControl checkout (itself adapted from BeyondMimic, which is also where
this repo's :mod:`.g1_21j` gains come from -- the armature constants and the
``stiffness = armature * (2*pi*10)^2`` / ``damping = 2*zeta*armature*omega`` formula are shared).
It is transcribed rather than imported because ``gear_sonic`` targets IsaacLab 2.3 and pins
``numpy==1.26.4``, which cannot coexist with this repo's IsaacLab 3.0-beta checkout.

Where it differs from our own G1 -- all of which is deliberate, because SONIC's action scale is
derived from these gains and mixing the two would silently change its effective action range:

===============================  =====================  =====================
quantity                         SONIC                  ours (:mod:`.g1_21j`)
===============================  =====================  =====================
hip_pitch class / effort         7520_22 / 139 N*m      7520_14 / 88 N*m
waist roll+pitch class           2 x 5020 / 50 N*m      7520_14 / 50 N*m
wrist pitch+yaw class            4010 / 5 N*m           5020 / 5 N*m
action scale factor              ``0.25 * e / k``       ``0.125 * e / k``
default pose                     hip -0.312, knee       hip -0.25, knee 0.46,
                                 0.669, ankle -0.363,   ankle -0.25, shoulder
                                 elbow 0.6              pitch 0.31, elbow 0.8
geometry                         capsule-ized URDF      box-merged USD
===============================  =====================  =====================

The asset is spawned by converting SONIC's URDF at load time (same as upstream) rather than from
a USD: every ``.usd`` in their checkout is a git-LFS pointer, and the URDF plus its meshes are
what ``git lfs pull`` actually materializes.
"""

from __future__ import annotations

import os
import pathlib

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# --- BeyondMimic actuator classes, verbatim from gear_sonic/envs/manager_env/robots/g1.py ---
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

_CLONE_NAME = "GR00T-WholeBodyControl"
_ASSET_TAIL = pathlib.Path("gear_sonic/data/assets/robot_description")
_URDF_TAIL = _ASSET_TAIL / "urdf/g1/main.urdf"
_PROBE_MESH_TAIL = _ASSET_TAIL / "meshes/g1/pelvis.STL"

# SONIC's 29-DOF order: IsaacSim's breadth-first ordering of its URDF, from
# ``gear_sonic/envs/env_utils/joint_utils.py::G1_ISAACLab_ORDER``. This repo's own 29-DOF USD
# resolves to the same order, which is why no permutation is needed between the two robots'
# action and observation vectors -- asserted at env build time, never assumed.
G1_SONIC_ISAAC_JOINT_ORDER: tuple[str, ...] = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)


def find_groot_root() -> pathlib.Path:
    """Absolute path to the GR00T-WholeBodyControl clone.

    The clone is a sibling of this repo, so a fixed number of ``..`` hops breaks inside a git
    worktree (``<repo>/.claude/worktrees/<branch>``). Honour ``$GROOT_WBC_ROOT`` first, then walk
    up looking for the clone -- the same approach as
    ``scripts/rsl_rl/generate_mb_library.py::_resolve_mb_root``, which already resolves this
    clone for the Motion Bricks stage.

    Returns:
        The clone's root directory.

    Raises:
        FileNotFoundError: If no clone can be found, naming the override.
    """
    env = os.environ.get("GROOT_WBC_ROOT")
    if env:
        p = pathlib.Path(env).expanduser().resolve()
        if not (p / _URDF_TAIL).is_file():
            raise FileNotFoundError(
                f"$GROOT_WBC_ROOT={p} does not contain {_URDF_TAIL}. Point it at the "
                f"{_CLONE_NAME} clone."
            )
        return p
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / _CLONE_NAME
        if (cand / _URDF_TAIL).is_file():
            return cand.resolve()
    raise FileNotFoundError(
        f"cannot find the {_CLONE_NAME} clone above {here}. Clone it as a sibling of this repo "
        "or set $GROOT_WBC_ROOT. The G1 meshes are git-LFS: after cloning run "
        f"'git lfs pull --include=\"{_ASSET_TAIL}/**\"'."
    )


def sonic_urdf_path() -> str:
    """Absolute path to SONIC's G1 URDF, checking that its LFS meshes are materialized.

    Raises:
        FileNotFoundError: If the meshes are still LFS pointer stubs, which would otherwise
            surface as an opaque failure inside the USD importer.
    """
    root = find_groot_root()
    probe = root / _PROBE_MESH_TAIL
    if probe.is_file() and probe.stat().st_size < 1024:
        raise FileNotFoundError(
            f"{probe} is a {probe.stat().st_size}-byte git-LFS pointer, not a mesh. Run "
            f"'git lfs pull --include=\"{_ASSET_TAIL}/**\"' in {root}."
        )
    return str(root / _URDF_TAIL)


def _find_project_root() -> pathlib.Path:
    """Walk up until a directory containing ``robot_assets`` is found (as :mod:`.g1_21j` does)."""
    path = pathlib.Path(__file__).resolve().parent
    while path != path.parent:
        if (path / "robot_assets").is_dir():
            return path
        path = path.parent
    raise FileNotFoundError("could not find the project root containing 'robot_assets'")


def sonic_patched_urdf_path(force: bool = False) -> str:
    """Path to a generated copy of SONIC's URDF whose root link is renamed ``pelvis_link``.

    SONIC calls its root link ``pelvis``; every other link name it uses matches ours
    (``*_ankle_roll_link``, ``torso_link``, ``*_wrist_yaw_link``). That single name is load-bearing
    on our side: the height-scanner prim path, the nested contact-sensor prim paths, the CLF base
    frame and the clip library's ``ref_frame`` all say ``pelvis_link``. Renaming the link is a
    pure identifier change -- no geometry, inertia or kinematics is touched -- and it lets the
    SONIC env stay a thin subclass of the teacher env, which is what keeps the comparison honest:
    identical terrain, commands, terminations and episode length, with only the robot and the
    policy differing.

    The copy also rewrites ``package://robot_description/`` mesh references to absolute paths in
    the clone, because it no longer sits beside the meshes.

    Generated under ``robot_assets/g1_sonic/`` (self-ignoring: it embeds machine-specific absolute
    paths, so it is regenerated per checkout rather than committed).

    Args:
        force: Rewrite even if the cached copy is newer than the source.

    Returns:
        Absolute path to the patched URDF.
    """
    src = pathlib.Path(sonic_urdf_path())
    out_dir = _find_project_root() / "robot_assets" / "g1_sonic"
    out = out_dir / "g1_sonic_pelvis_link.urdf"
    if not force and out.is_file() and out.stat().st_mtime >= src.stat().st_mtime:
        return str(out)
    mesh_root = src.parents[2]                                    # .../robot_description
    text = src.read_text()
    # Exact-match the quoted name so ``pelvis_contour_link`` is left alone.
    patched = (
        text.replace('<link name="pelvis">', '<link name="pelvis_link">')
        .replace('<parent link="pelvis" />', '<parent link="pelvis_link" />')
        .replace('<parent link="pelvis"/>', '<parent link="pelvis_link"/>')
        .replace("package://robot_description/", f"file://{mesh_root}/")
    )
    if '"pelvis"' in patched:
        raise RuntimeError(
            f"{src} still references the bare link name 'pelvis' after patching: "
            f"{[ln for ln in patched.splitlines() if chr(34) + 'pelvis' + chr(34) in ln][:3]}. "
            "The upstream URDF's formatting changed; update the substitutions in "
            "robot_rl.assets.robots.g1_sonic.sonic_patched_urdf_path."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ".gitignore").write_text("# generated from the GR00T clone; machine-specific\n*\n")
    out.write_text(patched)
    print(f"[INFO] wrote SONIC G1 URDF with root link renamed pelvis -> pelvis_link: {out}")
    return str(out)


_SONIC_INIT_STATE = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.76),
    joint_pos={
        ".*_hip_pitch_joint": -0.312,
        ".*_knee_joint": 0.669,
        ".*_ankle_pitch_joint": -0.363,
        ".*_elbow_joint": 0.6,
        "left_shoulder_roll_joint": 0.2,
        "left_shoulder_pitch_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_pitch_joint": 0.2,
    },
    joint_vel={".*": 0.0},
)

_SONIC_ACTUATORS: dict[str, ImplicitActuatorCfg] = {
    "legs": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_hip_yaw_joint",
            ".*_hip_roll_joint",
            ".*_hip_pitch_joint",
            ".*_knee_joint",
        ],
        effort_limit_sim={
            ".*_hip_yaw_joint": 88.0,
            ".*_hip_roll_joint": 139.0,
            ".*_hip_pitch_joint": 139.0,
            ".*_knee_joint": 139.0,
        },
        velocity_limit_sim={
            ".*_hip_yaw_joint": 32.0,
            ".*_hip_roll_joint": 20.0,
            ".*_hip_pitch_joint": 20.0,
            ".*_knee_joint": 20.0,
        },
        stiffness={
            ".*_hip_pitch_joint": STIFFNESS_7520_22,
            ".*_hip_roll_joint": STIFFNESS_7520_22,
            ".*_hip_yaw_joint": STIFFNESS_7520_14,
            ".*_knee_joint": STIFFNESS_7520_22,
        },
        damping={
            ".*_hip_pitch_joint": DAMPING_7520_22,
            ".*_hip_roll_joint": DAMPING_7520_22,
            ".*_hip_yaw_joint": DAMPING_7520_14,
            ".*_knee_joint": DAMPING_7520_22,
        },
        armature={
            ".*_hip_pitch_joint": ARMATURE_7520_22,
            ".*_hip_roll_joint": ARMATURE_7520_22,
            ".*_hip_yaw_joint": ARMATURE_7520_14,
            ".*_knee_joint": ARMATURE_7520_22,
        },
    ),
    "feet": ImplicitActuatorCfg(
        joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
        effort_limit_sim=50.0,
        velocity_limit_sim=37.0,
        stiffness=2.0 * STIFFNESS_5020,
        damping=2.0 * DAMPING_5020,
        armature=2.0 * ARMATURE_5020,
    ),
    "waist": ImplicitActuatorCfg(
        joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
        effort_limit_sim=50.0,
        velocity_limit_sim=37.0,
        stiffness=2.0 * STIFFNESS_5020,
        damping=2.0 * DAMPING_5020,
        armature=2.0 * ARMATURE_5020,
    ),
    "waist_yaw": ImplicitActuatorCfg(
        joint_names_expr=["waist_yaw_joint"],
        effort_limit_sim=88.0,
        velocity_limit_sim=32.0,
        stiffness=STIFFNESS_7520_14,
        damping=DAMPING_7520_14,
        armature=ARMATURE_7520_14,
    ),
    "arms": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_joint",
            ".*_wrist_roll_joint",
            ".*_wrist_pitch_joint",
            ".*_wrist_yaw_joint",
        ],
        effort_limit_sim={
            ".*_shoulder_pitch_joint": 25.0,
            ".*_shoulder_roll_joint": 25.0,
            ".*_shoulder_yaw_joint": 25.0,
            ".*_elbow_joint": 25.0,
            ".*_wrist_roll_joint": 25.0,
            ".*_wrist_pitch_joint": 5.0,
            ".*_wrist_yaw_joint": 5.0,
        },
        velocity_limit_sim={
            ".*_shoulder_pitch_joint": 37.0,
            ".*_shoulder_roll_joint": 37.0,
            ".*_shoulder_yaw_joint": 37.0,
            ".*_elbow_joint": 37.0,
            ".*_wrist_roll_joint": 37.0,
            ".*_wrist_pitch_joint": 22.0,
            ".*_wrist_yaw_joint": 22.0,
        },
        stiffness={
            ".*_shoulder_pitch_joint": STIFFNESS_5020,
            ".*_shoulder_roll_joint": STIFFNESS_5020,
            ".*_shoulder_yaw_joint": STIFFNESS_5020,
            ".*_elbow_joint": STIFFNESS_5020,
            ".*_wrist_roll_joint": STIFFNESS_5020,
            ".*_wrist_pitch_joint": STIFFNESS_4010,
            ".*_wrist_yaw_joint": STIFFNESS_4010,
        },
        damping={
            ".*_shoulder_pitch_joint": DAMPING_5020,
            ".*_shoulder_roll_joint": DAMPING_5020,
            ".*_shoulder_yaw_joint": DAMPING_5020,
            ".*_elbow_joint": DAMPING_5020,
            ".*_wrist_roll_joint": DAMPING_5020,
            ".*_wrist_pitch_joint": DAMPING_4010,
            ".*_wrist_yaw_joint": DAMPING_4010,
        },
        armature={
            ".*_shoulder_pitch_joint": ARMATURE_5020,
            ".*_shoulder_roll_joint": ARMATURE_5020,
            ".*_shoulder_yaw_joint": ARMATURE_5020,
            ".*_elbow_joint": ARMATURE_5020,
            ".*_wrist_roll_joint": ARMATURE_5020,
            ".*_wrist_pitch_joint": ARMATURE_4010,
            ".*_wrist_yaw_joint": ARMATURE_4010,
        },
    ),
}


def _deactivate_world_anchor_joints(usd_path: str) -> list[str]:
    """Deactivate any joint in ``usd_path`` that pins the root link to the world.

    IsaacLab 3.0's URDF importer emits a ``PhysicsFixedJoint`` named ``root_joint`` between the
    articulation prim and the root link **even when ``fix_base=False``** (verified in the
    converted ``payloads/Physics/physics.usda``, with no disabling override anywhere). It is
    enabled, so the pelvis is bolted to the articulation prim's frame.

    That is fatal here rather than merely wrong: the clip reset teleports the root tens of metres
    onto its terrain tile, the constraint yanks it back, and the solver diverges inside a single
    control step -- measured at 5 -> 570 -> 84000 rad/s joint velocity, with every episode dying
    instantly on the base-drift termination. Deactivating the joint restores proper floating-base
    behaviour (measured: root stays where it is written and settles under gravity, joint velocity
    0).

    ``ArticulationRootPropertiesCfg.fix_root_link`` cannot do this: IsaacLab finds the
    world-anchoring joint via ``find_global_fixed_joint_prim``, which only recognises a joint with
    one of ``body0``/``body1`` **empty**, and this one has both targets set.

    Args:
        usd_path: The converted USD to patch in place.

    Returns:
        Paths of the joints that were deactivated.
    """
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    hit = []
    for prim in Usd.PrimRange(root):
        joint = UsdPhysics.Joint(prim)
        if not joint:
            continue
        has_b0 = joint.GetBody0Rel().GetTargets() != []
        has_b1 = joint.GetBody1Rel().GetTargets() != []
        # Either shape of world anchor: IsaacLab's own (one empty target) or the importer's
        # ``root_joint`` (both targets set, body0 being the non-rigid articulation prim).
        if prim.GetName() == "root_joint" or not (has_b0 and has_b1):
            prim.SetActive(False)
            hit.append(str(prim.GetPath()))
    if hit:
        stage.Save()
    return hit


def sonic_usd_path(force: bool = False) -> str:
    """Convert SONIC's URDF to a USD under ``robot_assets/g1_sonic/`` and patch it.

    Converting once into the repo (rather than letting the spawner reconvert into a fresh
    temporary directory on every run) is what makes the world-anchor patch above possible: the
    fix has to be applied to the asset before any environment clones it.

    Args:
        force: Reconvert even if the cached USD is newer than the URDF.

    Returns:
        Absolute path to the patched USD.
    """
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    urdf = pathlib.Path(sonic_patched_urdf_path(force=force))
    out_dir = _find_project_root() / "robot_assets" / "g1_sonic"
    out_dir.mkdir(parents=True, exist_ok=True)
    # IsaacLab's converter caches on a hash of this cfg, so an unchanged URDF is not reconverted.
    converter = UrdfConverter(
        UrdfConverterCfg(
            asset_path=str(urdf),
            usd_dir=str(out_dir),
            force_usd_conversion=force,
            fix_base=False,
            merge_fixed_joints=True,
            # The actuator cfgs own the gains, exactly as upstream does.
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
            ),
        )
    )
    # Idempotent: once the joint is deactivated, ``Usd.PrimRange`` no longer traverses it, so a
    # cached USD re-reports "none" and is left untouched.
    dropped = _deactivate_world_anchor_joints(converter.usd_path)
    if dropped:
        print(f"[INFO] SONIC G1 USD {converter.usd_path}: deactivated world-anchor joint(s) "
              f"{dropped}")
    return converter.usd_path


def g1_sonic_cfg(self_collisions: bool = False) -> ArticulationCfg:
    """SONIC's Unitree G1, as the released checkpoint was trained on it.

    A factory rather than a module constant because it has to resolve the URDF in the GR00T
    clone: doing that at import time would make ``import robot_rl`` fail on any machine without
    the clone, and this package is imported eagerly by the task registry.

    Args:
        self_collisions: Whether to enable self-collisions between the robot's own links.
            Upstream sets this True, but it does so on a **capsule-ized** robot, and
            ``replace_cylinders_with_capsules`` was removed from the URDF importer in IsaacLab
            3.0 ("no longer supported by the URDF importer 3.0"). Without it the collider for
            each link is the convex hull of its visual mesh, and neighbouring hulls overlap
            across every joint -- with self-collisions on, the depenetration impulse launches the
            robot in the first physics step, so it cannot even stand. Defaulting to False is the
            closest behaviour we can reproduce; it is a documented delta from SONIC's training
            setup, not an accident. (SONIC's locomotion does not rely on self-contact; the
            terminations it is graded against here are base drift, not self-collision.)

    Returns:
        The articulation config, spawning by URDF->USD conversion at scene build time.

    Raises:
        FileNotFoundError: If the clone or its LFS meshes are missing.
    """
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=sonic_usd_path(),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                # The world-anchoring ``root_joint`` the importer emits is deactivated in the USD
                # itself by :func:`_deactivate_world_anchor_joints`; ``fix_root_link`` cannot
                # reach it (see that function's docstring for why).
                enabled_self_collisions=self_collisions,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=_SONIC_INIT_STATE,
        soft_joint_pos_limit_factor=0.9,
        actuators=dict(_SONIC_ACTUATORS),
    )


def _action_scale(actuators: dict[str, ImplicitActuatorCfg], factor: float) -> dict[str, float]:
    """Per-joint action scale ``factor * effort_limit_sim / stiffness``, keyed by joint regex.

    Mirrors upstream's ``G1_MODEL_12_ACTION_SCALE`` loop (``robots/g1.py:360``) and this repo's
    own :data:`~robot_rl.assets.robots.g1_21j.G1_ACTION_SCALE`.
    """
    scale: dict[str, float] = {}
    for a in actuators.values():
        e, s, names = a.effort_limit_sim, a.stiffness, a.joint_names_expr
        if not isinstance(e, dict):
            e = dict.fromkeys(names, e)
        if not isinstance(s, dict):
            s = dict.fromkeys(names, s)
        for n in names:
            if n in e and n in s and s[n]:
                scale[n] = factor * e[n] / s[n]
    return scale


# SONIC uses 0.25 where we use 0.125, so its actions cover twice the joint range per unit.
G1_SONIC_ACTION_SCALE = _action_scale(_SONIC_ACTUATORS, 0.25)
