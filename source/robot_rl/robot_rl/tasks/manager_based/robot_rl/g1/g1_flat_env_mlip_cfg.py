from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from robot_rl.tasks.manager_based.robot_rl import mdp
from robot_rl.tasks.manager_based.robot_rl.terrains.rough import (
    ROUGH_SLOPED_FOR_FLAT_HZD_CFG,
)

from .g1_rough_env_mlip_cfg import G1RoughMlipEnvCfg


##
# Environment configuration
##
@configclass
class G1FlatMlipEnvCfg(G1RoughMlipEnvCfg):
    """Configuration for the G1 Flat environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()


        # self.rewards.clf_decreasing_condition = None


        self.rewards.action_rate_l2.weight = -0.01

        self.scene.terrain.terrain_generator = ROUGH_SLOPED_FOR_FLAT_HZD_CFG
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no curriculum
        self.curriculum.terrain_levels = None
        self.curriculum.clf_curriculum = None



        
        
@configclass
class G1_custom_mlip_clf(G1FlatMlipEnvCfg):
    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        # both front and back 1.14
        # just front: 0.616
        self.events.add_plate_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
                "mass_distribution_params": (0.616, 0.616),
                "operation": "add",
            },
        )


class G1FlatMlipEnvCfg_PLAY(G1FlatMlipEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        # self.events.push_robot = None
        self.events.push_robot.interval_range_s = (5.0, 5.0)
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "yaw": (0, 0),
        }  # (-3.14, 3.14)},


