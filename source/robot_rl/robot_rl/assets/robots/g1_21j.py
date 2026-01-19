import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

ROBOT_ASSETS = "robot_assets/g1"
# TODO: Fix warnings about waist_roll_link and yaw_link inertia and mass
G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ROBOT_ASSETS}/g1_21j_self_col.usd",
        # usd_path=f"{ROBOT_ASSETS}/g1_21j_urdf_v4_min_contacts.usd",
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
            # enabled_self_collisions=False,
            solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Bend up
        # pos=(0.0, 0.0, 0.62), #0.63),
        # rot=(0.73, 0.0, 0.67, 0.0),

        # Standing
        pos=(0.0, 0.0, 0.785),

        joint_pos={
            # Bend up
            # ".*_hip_pitch_joint": -2.0,
            # ".*_hip_roll_joint": 0.0,
            # ".*_hip_yaw_joint": 0.0,
            # ".*_knee_joint": 1.0,
            # ".*_ankle_pitch_joint": -0.48,
            # ".*_ankle_roll_joint": 0.0,
            # "waist_yaw_joint": 0.0,
            # "left_shoulder_yaw_joint": 0.0,
            # "left_shoulder_pitch_joint": 0.1,
            # "left_shoulder_roll_joint": 0.5,
            # "right_shoulder_yaw_joint": 0.0,
            # "right_shoulder_pitch_joint": 0.1,
            # "right_shoulder_roll_joint": -0.5,
            # ".*_elbow_joint": 0.7,  # 1.39,

            # Standing
            ".*_hip_pitch_joint": -0.25,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_yaw_joint": 0.0,
            ".*_knee_joint": 0.46,
            ".*_ankle_pitch_joint": -0.25,
            ".*_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_shoulder_pitch_joint": 0.07,
            "left_shoulder_roll_joint": 0.24,
            "right_shoulder_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.07,
            "right_shoulder_roll_joint": -0.24,
            ".*_elbow_joint": 0.1, #1.39,

            # ".*_wrist_roll_joint": 0.,
            # ".*_wrist_pitch_joint": 0.,
            # ".*_wrist_yaw_joint": 0.,
            # "left_one_joint": 1.0,
            # "right_one_joint": -1.0,
            # "left_two_joint": 0.52,
            # "right_two_joint": -0.52,
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
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_yaw_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_joint": 150.0,
            },
            damping={
                ".*_hip_yaw_joint": 2.0,
                ".*_hip_roll_joint": 2.0,
                ".*_hip_pitch_joint": 2.0,
                ".*_knee_joint": 4.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=50,
            velocity_limit_sim=37.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
        ),
        "waist": ImplicitActuatorCfg(
            effort_limit_sim=88,
            velocity_limit_sim=37.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=100.0,
            damping=2.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim=300,
            velocity_limit_sim=100.0,
            stiffness={
                ".*_shoulder_pitch_joint": 100.0,
                ".*_shoulder_roll_joint": 100.0,
                ".*_shoulder_yaw_joint": 50.0,
                ".*_elbow_joint": 50.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 2.0,
                ".*_shoulder_roll_joint": 2.0,
                ".*_shoulder_yaw_joint": 2.0,
                ".*_elbow_joint": 2.0,
            },
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_.*": 0.01,
            },
        ),
    },
)
"""Configuration for the Unitree G1 Humanoid robot."""


G1_MINIMAL_CFG = G1_CFG.copy()
# G1_MINIMAL_CFG.spawn.usd_path = f"{G1_CUSTOM_DIR}/Robots/Unitree/G1/g1_23dof_minimal.usda"
"""Configuration for the Unitree G1 Humanoid robot with fewer collision meshes.

This configuration removes most collision meshes to speed up simulation.
"""



""" G1 Delayed Actuator Cfg"""
G1_DELAYED_ACTUATOR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ROBOT_ASSETS}/g1_21j_urdf_v3_min_contacts.usd",
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
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.785),  # TODO: Consider setting this higher if I init issues
        joint_pos={
            ".*_hip_pitch_joint": -0.25,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_yaw_joint": 0.0,
            ".*_knee_joint": 0.46,
            ".*_ankle_pitch_joint": -0.25,
            ".*_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_shoulder_pitch_joint": 0.07,
            "left_shoulder_roll_joint": 0.24,
            "right_shoulder_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.07,
            "right_shoulder_roll_joint": -0.24,
            ".*_elbow_joint": 1.39,
            # ".*_wrist_roll_joint": 0.,
            # ".*_wrist_pitch_joint": 0.,
            # ".*_wrist_yaw_joint": 0.,
            # "left_one_joint": 1.0,
            # "right_one_joint": -1.0,
            # "left_two_joint": 0.52,
            # "right_two_joint": -0.52,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_yaw_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_joint": 150.0,
            },
            damping={
                ".*_hip_yaw_joint": 2.0,
                ".*_hip_roll_joint": 2.0,
                ".*_hip_pitch_joint": 2.0,
                ".*_knee_joint": 4.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
            },
            min_delay=0,
            max_delay=1,
        ),
        "feet": DelayedPDActuatorCfg(
            effort_limit_sim=50,
            velocity_limit_sim=37.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
            min_delay=0,
            max_delay=1,
        ),
        "waist": DelayedPDActuatorCfg(
            effort_limit_sim=88,
            velocity_limit_sim=37.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=100.0,
            damping=2.0,
            armature=0.01,
            min_delay=0,
            max_delay=1,
        ),
        "arms": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim=300,
            velocity_limit_sim=100.0,
            stiffness={
                ".*_shoulder_pitch_joint": 100.0,
                ".*_shoulder_roll_joint": 100.0,
                ".*_shoulder_yaw_joint": 50.0,
                ".*_elbow_joint": 50.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 2.0,
                ".*_shoulder_roll_joint": 2.0,
                ".*_shoulder_yaw_joint": 2.0,
                ".*_elbow_joint": 2.0,
            },
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_.*": 0.01,
            },
            min_delay=0,
            max_delay=1,
        ),
    },
)

