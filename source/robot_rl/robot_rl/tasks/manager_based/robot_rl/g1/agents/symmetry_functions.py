from typing import Tuple
import torch
import tensordict

def symmetric_data_augmentation(env, obs: tensordict.TensorDict, actions: torch.Tensor) -> Tuple[tensordict.TensorDict, torch.Tensor]:
    """
    Augment the data for the RSL RL data augmentation.

    obs: Tensor of shape [batch, num_obs]
    actions: Tensor of shape [batch, num_actions]
    env: RL vec env

    Flip the observation and actions.
    """

    # Can pull the remapping matrix R from the trajectory manager from the command from the env
    obs
    ##
    # Obs
    ##
    # Flip base ang vel yaw rate

    # Flip projected gravity?

    # Flip joints positions

    # Flip joint vels

    # Flip action obs

    # Flip ref_traj, act_traj

    # Flip ref_traj_vel and act_traj_vel

    # Flip root_quat

    # Flip base lin vel

    ##
    # Actions
    ##
    # Flip position actions


# TODO: Make one for the walking that also flips the phasing variable