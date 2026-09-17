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
* :mod:`freeform` -- the SHARED FOUNDATION every course below is built on: the ``route`` block
  the RL steering follows, spawn areas, the goal geometry and the manifest validator.  It also
  holds three DEPRECATED randomised section families (``flat`` / ``boxes`` / ``stairs``) that
  nothing was ever fine-tuned on; see its own docstring for which half is which.
* :mod:`box2` / :mod:`step_courses` / :mod:`varied_courses` -- the courses in use.  Where a
  randomised family presents a DISTRIBUTION, these present geometry that can be overfitted to,
  re-evaluated on and compared against the MuJoCo eval:

  - :mod:`box2` -- two 0.40 m boxes 3 m apart, the worst course in the mjeval catalogue and
    the only one that fails at every speed.  Mirrors an mjeval course.
  - :mod:`step_courses` -- a staircase walked up into a descent.  Mirrors an mjeval course.
    (Its CURB course was removed on 2026-09-09; nothing was fine-tuned on it.)
  - :mod:`varied_courses` -- 30 box + 30 up-into-down tiles in ONE library, where a tile is
    GEOMETRY rather than an initial condition: the spawn areas and a continuous speed band
    carry the variation a fixed course puts on separate tiles.

  The two mirrored courses are held to their mjeval twins by a TEST rather than by an import,
  since this package stays free of ``transfer`` as it stays free of IsaacLab.
"""
