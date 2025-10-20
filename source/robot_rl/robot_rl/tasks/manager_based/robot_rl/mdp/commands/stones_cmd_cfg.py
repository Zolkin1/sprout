import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
from dataclasses import MISSING

from .stones_cmd import StonesCommandTerm

from robot_rl.tasks.manager_based.robot_rl.constants import STONES

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
@configclass
class StonesCommandCfg(CommandTermCfg):
    """
    Configuration for the StonesCommandTerm.
    """

    class_type: type = StonesCommandTerm
    asset_name: str = "robot"
    debug_vis: bool = True

    output_command_name: str = MISSING

    resampling_time_range: tuple[float, float] = (5.0, 15.0)  # Resampling time range in seconds

    # Visualization configurations
    nextstone_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/stone",
        markers={
            "nextstone": sim_utils.CuboidCfg(
                size=(STONES.stone_x, STONES.stone_y/2.0, STONES.stone_z), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
            )
        },
    )
    nextnextstone_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/nextstone",
        markers={
            "nextnextstone": sim_utils.CuboidCfg(
                size=(STONES.stone_x, STONES.stone_y/2.0, STONES.stone_z), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
            )
        },
    )
    
    originframe_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/originframe",
        markers={
            "originframe": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
            )}
    )
