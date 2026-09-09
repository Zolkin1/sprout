# Copyright 2026 Zachary Olkin. All rights reserved.

"""Terrain geometry for offline clip generation -- the authoritative, sim-free source.

The generators EMIT the geometry they place clips on, and the RL side imports the same
numbers, so nothing is recomputed (or allowed to drift) between generation and training:

* :mod:`stair_geom` -- stair tread / riser arithmetic (also read by the IsaacLab
  ``terrains.trimesh.stair`` block builder).
* :mod:`box_clip` -- overlay-box construction for the stair / jump families, incl. the
  ``variant="floating"`` riser-less treads.
* :mod:`terrain_place` -- placing a clip onto a tile (SE(2)+z alignment).
* :mod:`terrain_sampler` / :mod:`terrain_variants` -- tile sampling and per-tile
  geometry variation.
* :mod:`distractor` -- occupancy bookkeeping for distractor boxes the robot must ignore.
* :mod:`freeform` -- sections authored for their own sake, with no clip behind them: the
  FM-driven fine-tune predicts its reference from the cameras, so the ground no longer has to
  be built under a recorded motion.  Each tile carries the ``route`` the RL steering follows.
* :mod:`box2` / :mod:`step_courses` -- FIXED free-form courses, built on :mod:`freeform`'s
  helpers.  Where the families above present a DISTRIBUTION, these present one obstacle that
  can be overfitted to, re-evaluated on and compared against the MuJoCo eval: the two-box
  course, a single wide step (a curb), and a staircase walked up and back down.  Two of the
  three mirror an mjeval course and are held to it by a test rather than by an import, since
  this package stays free of ``transfer`` as it stays free of IsaacLab.
"""
