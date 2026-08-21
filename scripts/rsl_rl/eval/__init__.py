# Copyright 2026 Zachary Olkin. All rights reserved.

"""The terrain evaluation harness.

Two entry points, sharing one on-disk manifest so no argument is ever retyped between
stages and no two stages can disagree about what was run:

* :mod:`evaluate` -- ``--stage plan | cell | grid``.  Selects the tile/clip set, builds a
  self-contained subset clip library, then rolls the policy over it one CELL at a time (a
  cell is one ``reference_mode`` x ``dr_preset`` x ``fm_label``).  Only the ``cell`` stage
  imports IsaacLab.
* :mod:`report` -- ``--stage reduce | plot``.  Turns one or several run directories into
  statistics and figures.  Imports no IsaacLab, and ``reduce`` imports no matplotlib.

The supporting modules are split so the parts that need neither a simulator nor a display
can be unit-tested on their own: :mod:`sampler` (selection + scheduling), :mod:`schema`
(the artifact contract), :mod:`dr_presets` (the randomization table), :mod:`stats` (the
estimators) and :mod:`figures` (the plots).
"""
