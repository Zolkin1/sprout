# Copyright 2026 Zachary Olkin. All rights reserved.

"""NVIDIA's SONIC whole-body motion tracker, run as a comparison baseline.

Upstream: https://github.com/NVlabs/GR00T-WholeBodyControl (``gear_sonic``). We drive the
released ONNX encoder/decoder pair from HuggingFace ``nvidia/GEAR-SONIC``; the training stack is
never imported (its pins conflict with this repo's IsaacLab checkout).

Layout:

* :mod:`.spec` -- the observation contract: names -> widths -> offsets, validated against the graphs
* :mod:`.obs` -- numpy assembly of both input vectors, ported from the C++ deploy reference
* :mod:`.policy` -- :class:`~robot_rl.baselines.sonic.policy.SonicPolicy`, onnxruntime only
* :mod:`.env_bridge` -- the one torch/isaaclab module: env + command term -> obs -> actions

Only SONIC's ``g1`` encoder mode is implemented: it consumes the reference joint position and
velocity window plus the reference-vs-robot root orientation, which is exactly what this repo's
clip and flow-matching command terms already produce. The teleop and SMPL modes need VR or SMPL
inputs we have no source for.
"""

from .obs import MotionReference, RobotState
from .policy import SonicPolicy
from .spec import SonicSpec, load_spec

__all__ = ["MotionReference", "RobotState", "SonicPolicy", "SonicSpec", "load_spec"]
