import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def reset_on_reference(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        command_name: str,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    """
    This event will reset the robot somewhere on the reference trajectory.

    First we will sample a random time in the reference trajectory.
    Then we grab the desired reference at that time. We can either set this to the robot state, or add some
        randomization on it.
    The random sample needs to go through the trajectory reference command so that the command knows what the
        current phase of the commanded trajectory is.
    """