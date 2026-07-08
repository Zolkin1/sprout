# Copyright 2026 Zachary Olkin. All rights reserved.

"""IsaacLab-free offline motion/clip generation for the G1 RL pipeline.

This package holds the reference-trajectory engine (trajectory/library/multiskill
managers), the world-frame lift math, clip I/O + fingerprints, forward kinematics,
the offline clip stitcher, and (in :mod:`motion_gen.motion_matching`) the
motion-matching transition generator.

It deliberately depends only on ``torch``/``numpy``/``h5py``/``pyyaml``/
``tensordict``/``huggingface_hub`` -- **not** on IsaacLab or Isaac Sim -- so offline
clip generation runs without the simulator.  The RL package (``robot_rl``) depends on
this package, never the other way around.

The one exception is the terrain clip path, which still reaches into ``robot_rl``'s
terrain-block geometry via lazy imports; flat generation and all of the engine import
cleanly with no IsaacLab present.
"""

__version__ = "0.1.0"
