from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.observations import generated_commands  # noqa: F401, F403


def sin_phase(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    period = generated_commands(env, command_name).clone()

    phase = 2 * torch.pi * (env.sim.current_time / period)
    sphase = torch.sin(phase).unsqueeze(-1)

    return sphase


def cos_phase(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    period = env.command_manager.get_command(command_name).clone()

    phase = 2 * torch.pi * (env.sim.current_time / period)
    cphase = torch.cos(phase).unsqueeze(-1)

    return cphase


# TODO: Remove
def is_ground_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    sp = sin_phase(env, period)
    cp = cos_phase(env, period)

    return torch.tensor([(sp < 0.0), (cp < 0.0)])
