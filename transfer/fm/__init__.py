# Copyright 2026 Zachary Olkin. All rights reserved.

"""Vendored, isaaclab-free flow-matching inference package for deployment.

A standalone pure-torch copy of the generator's inference stack
(``robot_rl.flow_matching``) used by the sim2sim runner and (later) the hardware
controller. It imports **no** isaaclab or robot_rl: the frame/quaternion math
lives in :mod:`quat_math` (parity-tested against ``isaaclab.utils.math`` in
``tests/test_quat_math_parity.py``).

Keep these modules in lock-step with ``source/robot_rl/robot_rl/flow_matching``;
the only edits versus the originals are the math import (isaaclab -> quat_math)
and ``online.py``'s asymmetric history/target frame support.
"""

from .config import FlowMatchingConfig
from .model import FlowMatchingTransformer
from .normalizer import Normalizer
from .online import (
    BatchedHistory,
    FmOnlineContext,
    build_history_token,
    load_fm_context,
    normalize_future_cmd,
    sample_world,
)

__all__ = [
    "FlowMatchingConfig",
    "FlowMatchingTransformer",
    "Normalizer",
    "BatchedHistory",
    "FmOnlineContext",
    "build_history_token",
    "load_fm_context",
    "normalize_future_cmd",
    "sample_world",
]
