# Copyright 2026 Zachary Olkin. All rights reserved.

"""How far ahead of a terrain feature a clip stops tracking the command and starts transitioning.

ONE constant, shared by the two consumers that must agree about it:

* the CLIP GENERATOR -- where the approach stops being steady locomotion and the in-betweening into
  the terrain entry begins;
* the VISUAL-CLUTTER pass (``terrain.clutter``) -- an interactable-looking decoy is vetoed if the
  robot's path enters its zone at all, because walking past a staircase is a lesson and walking into
  its approach zone without latching is the opposite one.

``clutter.py`` declared its own copy first, with a docstring that said exactly what this module is:
"One declared number... the same distance at which a clip should stop tracking the steady-state
command and start following the terrain motion. Today the clips have no such contract (the latch is
abrupt and its extent varies by an order of magnitude between families), so this is the definition
arriving before the thing it will govern." This is that thing arriving.

It lives in ``motionbricks/`` rather than ``terrain/`` because ``mb_env.load_schema_module`` resolves
modules by path out of this directory, which is how the Motion Bricks conda env -- a separate process
that cannot import the package normally -- reads the generation contracts. numpy + stdlib only, no
intra-package imports, same as its siblings.

WHY 2.0 m, and why it is hard-coded
-----------------------------------
Hard-coded on purpose. Deriving it from the camera would make it a function of the approach speed,
the terrain's entry speed and the acceleration bound, which is a lot of machinery for a number best
re-evaluated once a generator has been trained against one.

What the camera contract does say, computed from the ``_FAR`` spec in ``g1_depth_env_cfg.py``
(focal 4.0 mm, sampled 70 x 42 deg, 30 x 26 px -> ``fx = 21.42``, ``fy = 33.87`` px, mounted
1.116 m above the walking surface):

* the NEAR camera (100 x 70 deg at 65 deg down) reaches ~1.5 m of ground and no further, so the far
  camera is the ONLY sensor that sees an approaching feature at all;
* at 2 m the far camera puts a 4-step 0.13 m flight at **7.4 px** and a 0.45 m box face at **6.4 px**
  vertically -- comfortably resolved, which is the actual requirement;
* the FM's predicted window is 62 nodes ~ 1.24 s, which reaches 1.74 m at 1.4 m/s. So at 2 m the
  window is about the size of the zone: the window the generator must predict is essentially
  entirely INSIDE it, and "start transitioning" has to come from the depth image and the command
  rather than from seeing the terrain inside the window. That is the contract we want it to learn.

And one thing worth writing down, because it bounds what any zone can buy:

* **what distinguishes a staircase from a box is never resolved.** Tread structure subtends ~1.7 px
  per riser at 2 m and less further out, while both features present a well-resolved vertical face.
  A single 0.13 m riser peaks at 1.97 px at any range -- which is the reason
  ``mb_plan_library``'s ``single_step_frac`` is 0. That is a statement about the sensor, not about
  the model, and it is a measured mechanism for the stairs-read-as-box confusion the eval harness
  put at 0.174 closed-loop against a 0.103 control.

If it ever wants sizing by measurement instead of assertion, the instrument exists:
``make_terrain_probe_clips.py`` + ``fm_terrain_probe.py`` build "one flat clip, N tiles, N sceneries,
cameras the only variable", so authoring the same staircase at 1.0/1.5/2.0/3.0/4.0/5.0 m and reading
the climb-plan rate from ``eval/terrain_class.py`` gives the saturation distance directly.
"""

from __future__ import annotations

#: Distance (m) ahead of a terrain feature at which a clip stops tracking the steady command and
#: starts transitioning into the terrain motion -- and, for the clutter pass, the radius within
#: which an interactable-looking decoy may not be placed. See the module docstring for why 2.0.
APPROACH_ZONE_M: float = 2.0
