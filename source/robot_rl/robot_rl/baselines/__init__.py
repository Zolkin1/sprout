# Copyright 2026 Zachary Olkin. All rights reserved.

"""Third-party policies run as comparison baselines against our trained teacher.

Each baseline lives in its own subpackage and is loaded through an inference-only artifact
(ONNX), never by importing the upstream training stack -- their dependency pins conflict with
this repo's IsaacLab checkout. See :mod:`robot_rl.baselines.sonic`.
"""
