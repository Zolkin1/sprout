# Copyright 2026 Zachary Olkin. All rights reserved.

"""Torch-free deployment runtime (numpy + onnxruntime) for the generator+tracker pair.

The ONLY runtime for the MuJoCo sim2sim runner and the hardware (obelisk) controller.
Imports **no** torch / isaaclab / robot_rl: quaternion + transform math live in
:mod:`quat_math_np` / :mod:`transforms_np` (parity-tested against the isaaclab-free
``transfer/torch_math`` oracle and ``robot_rl.flow_matching.transforms`` respectively),
depth-camera geometry in :mod:`depth_obs`, and the two networks run via onnxruntime.
"""
