# Copyright 2026 Zachary Olkin. All rights reserved.

"""Shared G1 CLF-tracking scaffolding: actions, scene, contact sensors, dynamics
randomization, and the CLF / penalty rewards.

This is an abstract base -- it declares no observations, commands or terminations,
because the concrete env (:mod:`g1_teacher_env_cfg`) supplies all three.  It exists to
hold the parts that are genuinely independent of *what* is being tracked: the joint
position action, the robot + contact-sensor + light scene, the domain randomization
event set, and the CLF reward terms.
"""

from dataclasses import MISSING

##
# Isaac Lab
##
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab_tasks.utils import PresetCfg
from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg
from isaaclab_physx.sensors import ContactSensorCfg as PhysXContactSensorCfg

from isaaclab.envs import ManagerBasedRLEnvCfg

##
# Robot RL
##
from robot_rl.tasks.manager_based.robot_rl import mdp
from robot_rl.assets.robots import G1_ACTION_SCALE
from robot_rl.assets.robots.g1_21j import G1_CFG  # isort: skip


##
# Actions
##
@configclass
class G1ClfTrackingActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=G1_ACTION_SCALE, use_default_offset=True)


@configclass
class ClfTrackingBaseEnvContactSensorCfg(PresetCfg):

    def __init__(self, prim_path):
        self.default = PhysXContactSensorCfg(prim_path=prim_path, history_length=3,
                                        track_air_time=True)
        self.newton = NewtonContactSensorCfg(prim_path=prim_path, history_length=3,
                                        track_air_time=True)
        self.physx = self.default

@configclass
class G1ClfTrackingSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # Ground terrain. A placeholder: every concrete scene replaces this outright -- the
    # teacher swaps in a ClipTerrainImporterCfg that meshes the clip library itself (see
    # g1_teacher_terrain_env_cfg.G1TeacherTerrainSceneCfg). Only the physics material is
    # read from here (G1ClfTrackingEnvCfg.__post_init__ copies it onto sim).
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            # MAX, not multiply: this material never sets ``restitution``, so it is 0.0, and
            # under "multiply" the effective coefficient was 0.0 * foot = 0.0 no matter what
            # randomize_ground_contact_friction drew -- the restitution DR was dead. PhysX
            # resolves two disagreeing modes by enum precedence (average < min < multiply <
            # max), so "max" here wins over whatever the robot's shapes carry, and
            # max(0.0, foot) == foot makes the drawn range the EFFECTIVE range.
            # There is no "add" mode; the four are average / min / multiply / max.
            restitution_combine_mode="max",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # robots
    robot: ArticulationCfg = MISSING

    left_thigh_contact = ClfTrackingBaseEnvContactSensorCfg("{ENV_REGEX_NS}/Robot/Geometry/pelvis_link/left_hip_pitch_link/.*")
    right_thigh_contact = ClfTrackingBaseEnvContactSensorCfg("{ENV_REGEX_NS}/Robot/Geometry/pelvis_link/right_hip_pitch_link/.*")
    # Torso == the ``torso_link`` rigid body (the torso box collider).  On the 29-DOF robot the waist is
    # UN-merged (pelvis -> waist_yaw_link -> waist_roll_link -> torso_link), so the torso and the whole
    # arm chain nest UNDER torso_link -- on the old 21-DOF merged model they sat directly on
    # waist_yaw_link.  Match the body itself (no trailing ``/.*``) so a faceplant loads the torso, not its
    # arm descendants.
    #
    # This sensor was DEAD until 2026-08-19 and said nothing about it: the box it names sat on
    # waist_yaw_link, so torso_link had no collider and nothing could ever report contact here.  The box
    # now lives on torso_link (see the comment on it in g1_21j_29dof.urdf), which also means it finally
    # rotates with the unlocked waist instead of staying upright under a 30-degree lean.
    torso_contact = ClfTrackingBaseEnvContactSensorCfg("{ENV_REGEX_NS}/Robot/Geometry/pelvis_link/waist_yaw_link/waist_roll_link/torso_link")
    # The compute backpack, which torso_contact does NOT cover: that path matches the torso
    # body exactly, with no trailing ``/.*``, deliberately so a faceplant loads the torso and
    # not its arm descendants.  The pack is a child of torso_link and 2.64 kg standing ~0.2 m
    # behind it, so a backwards fall lands on this before anything else.
    backpack_contact = ClfTrackingBaseEnvContactSensorCfg("{ENV_REGEX_NS}/Robot/Geometry/pelvis_link/waist_yaw_link/waist_roll_link/torso_link/backpack_shell_link")
    left_elbow_contact = ClfTrackingBaseEnvContactSensorCfg("{ENV_REGEX_NS}/Robot/Geometry/pelvis_link/waist_yaw_link/waist_roll_link/torso_link/left_shoulder_pitch_link/left_shoulder_roll_link/left_shoulder_yaw_link/.*")
    right_elbow_contact = ClfTrackingBaseEnvContactSensorCfg("{ENV_REGEX_NS}/Robot/Geometry/pelvis_link/waist_yaw_link/waist_roll_link/torso_link/right_shoulder_pitch_link/right_shoulder_roll_link/right_shoulder_yaw_link/.*")
    left_foot_contact = ClfTrackingBaseEnvContactSensorCfg("{ENV_REGEX_NS}/Robot/Geometry/pelvis_link/left_hip_pitch_link/left_hip_roll_link/left_hip_yaw_link/left_knee_link/left_ankle_pitch_link/.*")
    right_foot_contact = ClfTrackingBaseEnvContactSensorCfg("{ENV_REGEX_NS}/Robot/Geometry/pelvis_link/right_hip_pitch_link/right_hip_roll_link/right_hip_yaw_link/right_knee_link/right_ankle_pitch_link/.*")
    # Shin == the ``knee_link`` rigid body (the shank between the knee joint
    # and the ankle).  Match the link itself (no trailing ``/.*``) so the
    # sensor sees only the shin and not its foot descendants — used for a
    # dedicated shin-contact termination on the stairs.
    left_shin_contact = ClfTrackingBaseEnvContactSensorCfg("{ENV_REGEX_NS}/Robot/Geometry/pelvis_link/left_hip_pitch_link/left_hip_roll_link/left_hip_yaw_link/left_knee_link")
    right_shin_contact = ClfTrackingBaseEnvContactSensorCfg("{ENV_REGEX_NS}/Robot/Geometry/pelvis_link/right_hip_pitch_link/right_hip_roll_link/right_hip_yaw_link/right_knee_link")

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class G1ClfTrackingEventsCfg:
    """Domain randomization. The reference reset is added by the concrete env."""

    # As of Isaac v3.0.0 beta there is no way to randomize dynamic and viscous friction that I can see.
    # Randomize joint friction
    joint_friction_params = EventTerm(
        func=mdp.randomize_joint_parameters_multi_friction,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "static_friction_distribution_params": (0.3, 1.6),
                "dynamic_friction_distribution_params": (0.3, 1.2),
                "viscous_friction_distribution_params": (0.01, 0.1),
                "operation": "add"},
    )

    # Randomize armature
    joint_armature_params = EventTerm(
        func=mdp.randomize_joint_parameters_multi_friction,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "armature_distribution_params": (0.95, 1.05),
                "operation": "scale"},
    )

    # PD Gain randomization
    gain_randomization = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform"
        },
    )

    # Adjust torso mass. Named for what it does: it scales the TORSO link, never the pelvis
    # (the floating base), which the old name "add_base_mass" claimed. The base's own mass is
    # still not randomized -- SONIC scales EVERY body at (0.8, 1.2) on reset instead.
    add_torso_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            # 29-DOF: torso is its own link.  The two backpack masses are included because
            # they are 2.64 kg of payload -- the single largest thing bolted to this robot --
            # and scoping this to torso_link alone gave the biggest new mass NO
            # randomisation at all.  Their real values are also the least certain numbers on
            # the robot (weighed, with "everything else" lumped into the Thor), so they are
            # exactly what a mass DR is for.  backpack_shell_link is the mesh and carries a
            # token mass; scaling it would randomise nothing.
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["torso_link", "backpack_thor_link", "backpack_zed_link"]),
            "mass_distribution_params": (0.85, 1.15),  # (-5.0, 5.0),
            "operation": "scale",
        },
    )

    # Randomize calibration errors
    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    randomize_ground_contact_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            # Now actually reaches physics -- see the terrain's restitution_combine_mode.
            # 0.5 matches SONIC's range (whose own is inert for the same multiply-by-zero
            # reason, so the number is borrowed, not the behaviour).
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
            "make_consistent": True,  # ensures dynamic friction <= static friction
        },
    )

    # On the TORSO, not the pelvis: on the 29-DOF robot the waist is un-merged
    # (pelvis -> waist_yaw -> waist_roll -> torso_link) and the pelvis carries very little of
    # the mass, so perturbing its CoM barely moves the whole-body CoM. The torso is where the
    # mass is, which is also where add_torso_mass scales -- the two now agree on a body.
    # SONIC randomizes the same link, at x (-0.025, 0.025), y/z (-0.05, 0.05); ours keeps the
    # ranges it had, so z is still the pelvis-era (-0.01, 0.01) and is the tight one.
    torso_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # The backpack's CoM, separately from the torso's because the uncertainty is a different
    # shape. The torso's is a manufacturing/loading spread about a well-known centroid; the
    # pack's is where a bolt-on assembly actually ended up, plus the fact that its 1.940 kg
    # lumps the Thor together with a mounting plate and cabling whose spread is not modelled
    # (the inertia tensor is the Thor BLOCK alone). +/-2 cm is roughly the plate's own depth.
    backpack_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["backpack_thor_link", "backpack_zed_link"]),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.75, 0.75), "y": (-0.75, 0.75)}},
    )


@configclass
class G1ClfTrackingRewardCfg:
    """CLF tracking + regularization penalties. Task-specific tracking rewards (e.g.
    commanded-velocity terms) are added by the concrete env, which knows the command."""

    ##
    # Joint limits
    ##
    # Penalize ankle and knee joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
    )

    ##
    # Torque
    ##
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)

    torque_lims = RewTerm(
        func=mdp.torque_limits,
        weight=-1.0,
    )

    ##
    # Undesired Contacts
    ##
    undesired_contacts = RewTerm(
        func=mdp.multiple_undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfgs": [SceneEntityCfg("left_thigh_contact",),
                            SceneEntityCfg("right_thigh_contact",),
                            SceneEntityCfg("torso_contact",),
                            SceneEntityCfg("left_elbow_contact",),
                            SceneEntityCfg("right_elbow_contact",),],
            "threshold": 1.0,
        },
    )

    ##
    # Action rate
    ##
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    ##
    # CLF Tracking
    ##
    clf_reward = RewTerm(
        func=mdp.clf_reward,
        weight=10.0,
        params={
            "command_name": "traj_ref",
            "max_eta_err": 6.0,
        }
    )

    ##
    # CLF Stability
    ##
    clf_decreasing_condition = RewTerm(
        func=mdp.clf_decreasing_condition,
        weight=-5.0,
        params={
            "command_name": "traj_ref",
            "alpha": 0.02,
            "eta_max": 6,
            "eta_dot_max": 12,
        }
    )


##
# Normal Config
##
@configclass
class G1ClfTrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Abstract base: scene / actions / events / rewards only.

    ``observations``, ``commands`` and ``terminations`` are deliberately left to the
    concrete env, and no curriculum manager is created (no task here uses one).
    """

    # Scene settings
    scene: G1ClfTrackingSceneCfg = G1ClfTrackingSceneCfg(num_envs=4096, env_spacing=2.5)

    actions: G1ClfTrackingActionsCfg = G1ClfTrackingActionsCfg()

    rewards: G1ClfTrackingRewardCfg = G1ClfTrackingRewardCfg()

    events: G1ClfTrackingEventsCfg = G1ClfTrackingEventsCfg()

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        ##
        # Scene
        ##
        self.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physics = PhysxCfg(
            gpu_max_rigid_patch_count=10 * 2**15,
            gpu_total_aggregate_pairs_capacity=2**22,
        )
