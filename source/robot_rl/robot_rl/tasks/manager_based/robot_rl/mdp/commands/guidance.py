# Copyright 2026 Zachary Olkin. All rights reserved.

"""Terrain velocity-guidance helpers (re-export shim).

The pure math moved to the IsaacLab-free :mod:`motion_gen.guidance_math` package so
the offline clip stitcher can use ``goal_pd_commands`` without importing any RL code.
This module re-exports the functions so the online velocity command and the existing
tests keep importing them from here.
"""

from __future__ import annotations

from motion_gen.guidance_math import (  # noqa: F401
    _wrap_to_pi,
    goal_pd_commands,
    guidance_pd_commands,
)
