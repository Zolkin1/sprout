from isaaclab.utils import configclass

from .g1_rough_env_cfg import G1RoughEnvCfg


##
# Environment configuration
##
@configclass
class G1FlatEnvCfg(G1RoughEnvCfg):
    """Configuration for the G1 Flat environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        ##
        # Scene
        ##
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # TODO: Consider changing weights/params from the rough env
