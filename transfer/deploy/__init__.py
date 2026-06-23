# Copyright 2026 Zachary Olkin. All rights reserved.

"""Torch-free deployment runtime (numpy + onnxruntime) for the generator+tracker pair.

Shared by the MuJoCo sim2sim runner and the hardware (obelisk) controller. Imports
**no** torch / isaaclab / robot_rl: quaternion + transform math live in
:mod:`quat_math_np` / :mod:`transforms_np` (parity-tested against the torch
``transfer/fm`` oracle), and the two networks run via onnxruntime.
"""
