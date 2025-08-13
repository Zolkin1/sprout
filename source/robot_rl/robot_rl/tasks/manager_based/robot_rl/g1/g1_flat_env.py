from isaaclab.utils import configclass

from .g1_rough_env_cfg import G1RoughEnvCfg
from robot_rl.tasks.manager_based.robot_rl.terrains.rough import ROUGH_SLOPED_FOR_FLAT_HZD_CFG
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from robot_rl.tasks.manager_based.robot_rl import mdp

##
# Environment configuration
##
@configclass
class G1FlatEnvCfg(G1RoughEnvCfg):
    """Configuration for the G1 Flat environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.events.add_plate_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
                "mass_distribution_params": (0.616,0.616),
                "operation": "add",
            }
        )

        ##
        # Scene
        ##
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # self.scene.terrain.terrain_generator = ROUGH_SLOPED_FOR_FLAT_HZD_CFG
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # TODO: Consider changing weights/params from the rough env
