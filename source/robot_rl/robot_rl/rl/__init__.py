# Copyright 2026 Zachary Olkin. All rights reserved.

"""In-repo rsl-rl extensions: the reference-encoder actor and its auxiliary-loss PPO.

rsl-rl resolves ``class_name`` through :func:`rsl_rl.utils.resolve_callable`, which accepts
``"package.module:Class"``.  Everything here is therefore reachable from a runner config
without patching the installed wheel -- which matters because ``env_isaaclab/`` is
gitignored and one ``uv sync`` away from being erased.
"""

from .aux_ppo import AuxPPO
from .ref_encoder_model import RefEncoderMLPModel

__all__ = ["AuxPPO", "RefEncoderMLPModel"]
