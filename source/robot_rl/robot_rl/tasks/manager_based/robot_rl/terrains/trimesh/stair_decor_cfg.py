# Copyright 2026 Zachary Olkin. All rights reserved.

"""Explicit, resolved stair decoration config.

A staircase can carry per-side decoration (a solid sidewall or a run of vertical
railing posts), floating (thin, riser-less) treads, and a rounded leading nosing.
During **online** terrain generation these are sampled from probabilities on the
stair cfg (``float_prob`` / ``wall_prob`` / ``pole_prob`` …).  For the **offline
clip** pipeline the variant must instead be *resolved once* by the generator and
recorded verbatim in the clip manifest, because the RL importer and the viewers
rebuild the mesh from the manifest without sharing the generator's RNG state.

:class:`StairDecorCfg` is that resolved, deterministic description: when it is set
on a :class:`StairBlockCfg` the tread builder renders it exactly and draws no
random numbers.  It lives in its own module (imported by both ``trimesh/stair.py``
and ``blocks/stair.py``) to avoid the ``stair.py`` <-> ``stair_cfg.py`` import
cycle.
"""

from __future__ import annotations

from typing import Literal

from isaaclab.utils import configclass


@configclass
class StairSideDecorCfg:
    """Decoration on one side (``left`` / ``right``) of a staircase."""

    kind: Literal["none", "wall", "posts"] = "none"
    """``"none"`` -- bare edge; ``"wall"`` -- one solid full-length sidewall;
    ``"posts"`` -- a run of square-section vertical railing posts marched along x."""

    thickness: float = 0.1
    """Wall thickness in y, or post cross-section side length (m)."""

    height: float = 1.5
    """Height above the highest tread top (m)."""

    spacing: float = 0.5
    """Gap between consecutive posts along x (m). Read only for ``"posts"``."""


@configclass
class StairDecorCfg:
    """Fully resolved (deterministic) decoration for one staircase span.

    Mirrors the ``decor`` dict recorded per stair band in the clip manifest.  All
    fields are explicit numbers (no probabilities): given the same ``StairDecorCfg``
    the tread builder produces byte-identical geometry every time.
    """

    float_thickness: float = 0.0
    """Thickness of floating (riser-less) treads (m). ``0.0`` => solid steps."""

    nosing_radius: float = 0.0
    """Radius of the rounded leading nosing on each tread (m). ``0.0`` => square
    edge. Constrained to ``<= min(step_height, step_depth) / 2`` by the builder."""

    nosing_segments: int = 6
    """Arc tessellation of the rounded nosing (number of segments). Recorded in the
    manifest so every renderer samples the arc identically."""

    left: StairSideDecorCfg = StairSideDecorCfg()
    """Decoration on the ``-y`` side of the staircase."""

    right: StairSideDecorCfg = StairSideDecorCfg()
    """Decoration on the ``+y`` side of the staircase."""
