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

    if obs is not None:
        device = obs.device

        batch_size = obs.batch_size[0]

        obs_aug = obs.repeat(2)

        # Original observations
        obs_aug["policy"][:batch_size] = obs["policy"][:batch_size]

        for group in ["policy, critic"]:
            obs_idx = 0
            for i, name in enumerate(env.unwrapped.observation_manager.active_terms[group]):
                obs_size = 0
                if name == "base_ang_vel":
                    obs_size = env.unwrapped.observation_manager.group_obs_term_dim[group][i].item()

                    obs_aug[group][batch_size:][obs_idx:obs_idx + obs_size] = (
                        obs[group][:][obs_idx:obs_idx + obs_size] * torch.tensor([-1, 1, -1], device=device))
                elif name == "base_lin_vel":
                    obs_size = env.unwrapped.observation_manager.group_obs_term_dim[group][i].item()

                    obs_aug[group][batch_size:][obs_idx:obs_idx + obs_size] = (
                        obs[group][:][obs_idx:obs_idx + obs_size] * torch.tensor([1, -1, 1], device=device)
                    )
                elif name == "projected_gravity":
                    obs_size = env.unwrapped.observation_manager.group_obs_term_dim[group][i].item()

                    obs_aug[group][batch_size:][obs_idx:obs_idx + obs_size] = (
                        obs[group][:][obs_idx:obs_idx + obs_size] * torch.tensor([1, -1, 1], device=device)
                    )
                elif name == "velocity_commands":
                    obs_size = env.unwrapped.observation_manager.group_obs_term_dim[group][i].item()

                    obs_aug[group][batch_size:][obs_idx:obs_idx + obs_size] = (
                        obs[group][:][obs_idx:obs_idx + obs_size] * torch.tensor([1, -1, -1], device=device)
                    )
                elif name == "joint_pos" or name == "joint_vel" or name == "actions":
                    obs_size = env.unwrapped.observation_manager.group_obs_term_dim[group][i].item()
                    obs_aug[group][batch_size:][obs_idx:obs_idx + obs_size] = (
                        _switch_g1_joints(obs[group][:][obs_idx:obs_idx + obs_size])
                    )
                elif name == "sin_phase" or name == "cos_phase":
                    obs_size = env.unwrapped.observation_manager.group_obs_term_dim[group][i].item()
                    obs_aug[group][batch_size:][obs_idx:obs_idx + obs_size] = obs[group][:][obs_idx:obs_idx + obs_size]
                elif name == "ref_traj" or name == "act_traj" or name == "ref_traj_vel" or name == "act_traj_vel":
                    obs_size = env.unwrapped.observation_manager.group_obs_term_dim[group][i].item()
                    obs_aug[group][batch_size:][obs_idx:obs_idx + obs_size] = (
                        cmd.get_symmetric_traj(obs[group][:][obs_idx:obs_idx + obs_size])
                    )
                elif name == "root_quat":
                    obs_size = env.unwrapped.observation_manager.group_obs_term_dim[group][i].item()
                    obs_aug[group][batch_size:][obs_idx:obs_idx + obs_size] = (
                        obs[group][:][obs_idx:obs_idx + obs_size] * torch.tensor([1, 1, -1, -1], device=device)
                    )
                elif name == "contact_state":
                    obs_size = env.unwrapped.observation_manager.group_obs_term_dim[group][i].item()
                    obs_aug[group][batch_size:][obs_idx:obs_idx + obs_size] = (
                        cmd.get_symmetric_contacts(obs[group][:][obs_idx:obs_idx + obs_size])
                    )

                # TODO: Add height map support

                obs_idx += obs_size
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]

        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)

        # Original actions
        actions_aug[:batch_size] = actions

        actions_aug[batch_size:] = _switch_g1_joints(actions)
    else:
        actions_aug = None

    return (obs_aug, actions_aug)

# TODO: Make one for the walking that also flips the phasing variable

def _switch_g1_joints(joints: torch.Tensor) -> torch.Tensor:
    """
    Reflection the joint values about the sagittal plane.

    IsaacSim ordering:
    [
        left_hip_pitch_joint, right_hip_pitch_joint, waist_yaw_joint,
        left_hip_roll_joint, right_hip_roll_joint, left_shoulder_pitch_joint,
        right_shoulder_pitch_joint, left_hip_yaw_joint, right_hip_yaw_joint,
        left_shoulder_roll_joint, right_shoulder_roll_joint, left_knee_joint,
        right_knee_joint, left_shoulder_yaw_joint, right_shoulder_yaw_joint,
        left_ankle_pitch_joint, right_ankle_pitch_joint, left_elbow_joint,
        right_elbow_joint, left_ankle_roll_joint, right_ankle_roll_joint
    ]

    Map all left -> right and right -> left
    Negate all roll and yaw joints.

    Left legs:
    [0, 3, 7, 11, 15, 19]

    Right Legs:
    [1, 4, 8, 12, 16, 20]

    Left Arms:
    []

    """
    joints_switched = torch.zeros_like(joints)

    joints_switched[:, :2] = joints[:, :2]

