# Copyright 2026 Zachary Olkin. All rights reserved.

"""Vendored, isaaclab-free flow-matching inference package (torch oracle / diagnostics).

A standalone pure-torch copy of the generator's inference stack
(``robot_rl.flow_matching``). It imports **no** isaaclab or robot_rl: the frame/quaternion
math lives in :mod:`quat_math` (parity-tested against ``isaaclab.utils.math`` in
``tests/test_quat_math_parity.py``).

Two consumers remain, both dev-side: the numpy deploy ports are parity-tested against
this stack (``transfer/deploy/tests/test_deploy_parity.py``), and
``scripts/flow_matching/fm_test_bench.py`` sweeps checkpoints with it without needing
isaaclab. **Deployment does not use it** -- sim2sim and hardware run the ONNX artifacts
through ``transfer/deploy``.

Keep these modules in lock-step with ``source/robot_rl/robot_rl/flow_matching``;
the only edit versus the originals is the math import (isaaclab -> quat_math) in
``transforms.py``.

TODO(depth): this copy is HEIGHT-SCAN ONLY. It predates the exteroception selector, so it
lacks ``extero_source`` / ``extero_tokens`` / ``DEPTH_CLIP`` / ``sanitize_extero`` and
cannot load a depth-conditioned checkpoint (``model.height_in`` would see the wrong token
count). Bring it forward from ``robot_rl.flow_matching`` before pointing the FM test bench
at a depth FM.
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
