import numpy as np
import torch

from sensor_msgs.msg import Joy

from .policy import RLPolicy


class BehaviorManager:
    """
    Manages which behavior is in use.
    """
    def __init__(self,
                 behavior_names: list[str],
                 behavior_buttons: list[int],
                 init_behavior: str,
                 hf_repo_ids: list[str],
                 hf_policy_folders: list[str],):
        """
        Initialize the behavior manager.

        TODO: Need to take in a list of valid behavior transitions.
        TODO: Add support for one policy with multiple behaviors.
        """

        self.behavior_names = behavior_names
        self.behavior_buttons = behavior_buttons

        self.policies = []
        for repo_id, policy_folder in zip(hf_repo_ids, hf_policy_folders):
            self.policies.append(RLPolicy(repo_id, policy_folder))

        self.active_behavior = init_behavior

        self.last_behavior_switch = 0.0

    def check_behavior_switch(self, joy_msg: Joy, time) -> str:
        if (time - self.last_behavior_switch) > 0.1:
            self.last_behavior_switch = time
            for i, button in enumerate(self.behavior_buttons):
                if joy_msg.buttons[button] == 1:
                    # TODO: Should also verify the validity of the transition
                    self.active_behavior = self.behavior_names[i]

        return self.active_behavior

    def get_active_behavior(self):
        return self.active_behavior

    def get_active_policy(self):
        return self.policies[self.get_active_policy_idx()]

    def get_active_policy_idx(self):
        return self.behavior_names.index(self.active_behavior)

    def create_obs(self,
                   qfb: np.ndarray,
                   vfb_ang: np.ndarray,
                   qjoints: np.ndarray,
                   vjoints: np.ndarray,
                   time: float,
                   cmd_vel: np.ndarray,
                   joint_names: list[str],
                   ):
        policy_idx = self.get_active_policy_idx()
        return self.policies[policy_idx].create_obs(qfb, vfb_ang, qjoints, vjoints, time - self.last_behavior_switch, cmd_vel, joint_names)

    def get_action(self, obs: torch.Tensor, joint_names_out: list[str]) -> np.ndarray:
        policy_idx = self.get_active_policy_idx()
        return self.policies[policy_idx].get_action(obs, joint_names_out)
