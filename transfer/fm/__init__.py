# Copyright 2026 Zachary Olkin. All rights reserved.

"""Vendored, isaaclab-free flow-matching inference package for deployment.

A standalone pure-torch copy of the generator's inference stack
(``robot_rl.flow_matching``) used by the sim2sim runner and (later) the hardware
controller. It imports **no** isaaclab or robot_rl: the frame/quaternion math
lives in :mod:`quat_math` (parity-tested against ``isaaclab.utils.math`` in
``tests/test_quat_math_parity.py``).

Keep these modules in lock-step with ``source/robot_rl/robot_rl/flow_matching``;
the only edit versus the originals is the math import (isaaclab -> quat_math) in
``transforms.py``.
"""

from .config import FlowMatchingConfig
from .model import FlowMatchingTransformer
from .normalizer import Normalizer
from .online import (
    BatchedHistory,
    FmOnlineContext,
    build_cmd_tokens,
    build_height_token,
    build_history_tokens,
    load_fm_context,
    sample_world,
)

__all__ = [
    "FlowMatchingConfig",
    "FlowMatchingTransformer",
    "Normalizer",
    "BatchedHistory",
    "FmOnlineContext",
    "build_cmd_tokens",
    "build_height_token",
    "build_history_tokens",
    "load_fm_context",
    "sample_world",
]
