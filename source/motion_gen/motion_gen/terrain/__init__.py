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
"""
