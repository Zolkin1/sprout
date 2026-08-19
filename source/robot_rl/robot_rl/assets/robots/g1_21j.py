# Copyright 2026 Zachary Olkin. All rights reserved.

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# From beyond mimic
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

import pathlib


def _find_project_root() -> pathlib.Path:
    """Walk up from this file until we find a directory containing 'robot_assets'."""
    path = pathlib.Path(__file__).resolve().parent
    while path != path.parent:
        if (path / "robot_assets").is_dir():
            return path
        path = path.parent
    raise FileNotFoundError("Could not find project root containing 'robot_assets'")


ROBOT_ASSETS = str(_find_project_root() / "robot_assets" / "g1")
# TODO: Fix warnings about waist_roll_link and yaw_link inertia and mass
G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{ROBOT_ASSETS}/g1_21j/g1_21j_box_merged/g1_21j_box_merged.usda",
        # 29 DOF (waist roll/pitch + wrists unlocked), hands as FIXED-JOINT LINKS.
        # The "_hands" variant exists because the stock URDF hangs the hand meshes off the
        # wrist as bare <visual> elements: the importer then gives them a non-identity
        # mesh->body offset, and MultiMeshRayCaster does not intersect meshes on that path,
        # so the depth cameras saw every limb except the hands.  Regenerate both with
        # robot_assets/g1/g1_21j/{make_hand_links_urdf,urdf_to_usd_conversion}.py.
        # Joint count and ORDER are identical to the pre-hands asset (verified), so
        # checkpoints trained on it load unchanged; only body_names grows (30 -> 46, the
        # old list is an exact prefix).
        usd_path=f"{ROBOT_ASSETS}/g1_21j/g1_21j_box_merged_29j_hands/g1_21j_box_merged_29j_hands.usda",
        # usd_path=f"{ROBOT_ASSETS}/g1_21j/g1_21j_merged/g1_21j_merged.usda",
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
            enabled_self_collisions=True,
            solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Standing
        pos=(0.0, 0.0, 0.785),

        joint_pos={
            # Standing
            ".*_hip_pitch_joint": -0.25,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_yaw_joint": 0.0,
            ".*_knee_joint": 0.46,
            ".*_ankle_pitch_joint": -0.25,
            ".*_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_shoulder_pitch_joint": 0.31,
            "left_shoulder_roll_joint": 0.24,
            "right_shoulder_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.31,
            "right_shoulder_roll_joint": -0.24,
            ".*_elbow_joint": 0.8,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            # HIP PITCH is the 22.5 gear (mode_11), not the 14.3 -- effort, velocity AND the
            # gain class move together.  Half the change is a trap: G1_ACTION_SCALE is
            # 0.125 * effort / stiffness, so raising the effort limit alone would INFLATE the
            # hip-pitch action range 1.58x (0.2738 -> 0.4324) instead of shrinking it to
            # 0.1753.  Every field below is the mode_11 URDF's, checked against that file by
            # tests/test_actuator_table.py.
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
                ".*_hip_yaw_joint": STIFFNESS_7520_14,
                ".*_hip_roll_joint": STIFFNESS_7520_22,
                ".*_hip_pitch_joint": STIFFNESS_7520_22,
                ".*_knee_joint": STIFFNESS_7520_22,
            },
            damping={
                ".*_hip_yaw_joint": DAMPING_7520_14,
                ".*_hip_roll_joint": DAMPING_7520_22,
                ".*_hip_pitch_joint": DAMPING_7520_22,
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
            effort_limit_sim=35.0,
            velocity_limit_sim=30.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "waist": ImplicitActuatorCfg(
            effort_limit_sim=88.0,
            velocity_limit_sim=32.0,
            joint_names_expr=["waist_yaw_joint"],
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
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_5020,
                ".*_shoulder_roll_joint": STIFFNESS_5020,
                ".*_shoulder_yaw_joint": STIFFNESS_5020,
                ".*_elbow_joint": STIFFNESS_5020,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_5020,
                ".*_shoulder_roll_joint": DAMPING_5020,
                ".*_shoulder_yaw_joint": DAMPING_5020,
                ".*_elbow_joint": DAMPING_5020,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_5020,
                ".*_shoulder_roll_joint": ARMATURE_5020,
                ".*_shoulder_yaw_joint": ARMATURE_5020,
                ".*_elbow_joint": ARMATURE_5020,
            },
        ),
        # Newly-unlocked DOFs.  waist roll/pitch: waist_yaw's 7520_14 class at the URDF's effort.
        # wrists: the arms' 5020 class (roll 25 N*m, pitch/yaw 5 N*m per the model).  Sensible defaults
        # from the sibling joints -- expected to be tuned before the real run.
        "waist_rp": ImplicitActuatorCfg(
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            effort_limit_sim=35.0,
            velocity_limit_sim=30.0,
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
        ),
        "wrists": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_roll_joint", ".*_wrist_pitch_joint", ".*_wrist_yaw_joint"],
            effort_limit_sim={
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            # Roll is an arm-class motor; pitch and yaw are the light 5 N*m ones and turn
            # slower.  One scalar here would have quietly given them roll's 37 rad/s.
            velocity_limit_sim={
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness=STIFFNESS_5020,
            damping=DAMPING_5020,
            armature=ARMATURE_5020,
        ),
    },
)
"""Configuration for the Unitree G1 Humanoid robot."""

G1_ACTION_SCALE = {}
for a in G1_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            G1_ACTION_SCALE[n] = 0.125 * e[n] / s[n]

def format_actuator_table(cfg: ArticulationCfg | None = None) -> str:
    """The resolved motor table, one line per joint pattern, for printing at run start.

    ``G1_ACTION_SCALE`` is DERIVED -- ``0.125 * effort / stiffness`` -- so a motor swap
    silently rescales the policy's action range, and the DIRECTION depends on whether the
    gain class moved with the effort limit. Putting hip pitch on the 22.5 gear shrinks it
    0.2738 -> 0.1753; changing only the effort limit would have grown it to 0.4324. Nothing
    at runtime shows either number, which is why this exists to be printed rather than
    trusted.

    Args:
        cfg: Articulation to read; :data:`G1_CFG` when omitted.

    Returns:
        A formatted multi-line table.
    """
    cfg = G1_CFG if cfg is None else cfg
    out = [f"  {'group':8s} {'joint':30s} {'effort':>8s} {'vel':>6s} {'kp':>9s} "
           f"{'kd':>8s} {'armature':>9s} {'act.scale':>10s}"]
    for group, act in cfg.actuators.items():
        for pat in act.joint_names_expr:
            def field(name: str, _act=act, _pat=pat):
                v = getattr(_act, name)
                return v.get(_pat) if isinstance(v, dict) else v
            eff, vel = field("effort_limit_sim"), field("velocity_limit_sim")
            stf, dmp, arm = field("stiffness"), field("damping"), field("armature")
            scale = "-" if not (eff and stf) else f"{0.125 * eff / stf:.4f}"
            out.append(f"  {group:8s} {pat:30s} {eff:8.1f} {vel:6.1f} {stf:9.4f} "
                       f"{dmp:8.4f} {arm:9.6f} {scale:>10s}")
    return "\n".join(out)


G1_MINIMAL_CFG = G1_CFG.copy()


