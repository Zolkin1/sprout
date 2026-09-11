# Copyright 2026 Zachary Olkin. All rights reserved.

"""How much walkable surface a clip RESERVES next to a terrain feature.

ONE constant, shared by the two consumers that must agree about it:

* the CLIP GENERATOR -- pass A adds it to the hold that adjoins the terrain, so the in-betweening
  into the terrain entry always has surface to happen on;
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

IT IS A PLANNING QUANTITY, NOT A TRANSIENT LENGTH
-------------------------------------------------
This is the one thing to get right about it, and it was got wrong once. The zone is how much surface
is RESERVED beside the terrain; it is not how long the in-betweening into the terrain lasts. That
duration is Motion Bricks' own choice, bounded only by the velocity change (``mb_env.hop_tokens``),
and whatever the transient does not use is covered by the exact loop in front of it, at a constant
commanded speed. The transition takes as long as it naturally takes and no longer.

Forcing the transient to SPAN the zone was tried and reverted, and the measurements say why:

* a clip commanded 0.2 m/s entering a 0.30 m/s stair traversal has |dv| = 0.1 m/s and essentially
  nothing to transition. Making it cover 2 m bought 1.87 s of coasting against a pinned pose --
  exactly the "extra frames are spent coasting rather than ramping" failure the ``slack`` bound
  exists to prevent;
* it drove 24 of 24 approaches to the model's 16-token ceiling (against 2 of 24 when free), so the
  latch came out at 1.87 s x speed -- still proportional to speed, which was the very thing the
  floor was meant to remove;
* it made the entry SEAM worse: pooled seam-accel ratio p50 2.42 / p95 6.77 forced, against
  1.71 / 5.17 free. A transient given only the time it needs converges cleanly; one made to coast
  arrives with a mismatched gait phase.

So the latch distance is a deterministic function of the deceleration -- 0.10 m at 0.2 m/s where
there is nothing to shed, 4.29 m at 3.3 m/s where there is. That is not an unlearnable spread: both
the current speed and the distance to the terrain are observable, so where the transition begins is
predictable from what the generator can see. A FIXED latch would be the arbitrary one.

Where a deceleration genuinely exceeds one horizon (3.5 -> 0.1 m/s wants 2.21 s against a 1.867 s
ceiling), the answer is the lead-up's own loop/transient alternation shedding speed on the FLAT
first, so the terrain transient bridges only what is left.

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
  A single 0.13 m riser peaks at 1.97 px at any range. That is a statement about the sensor, not
  about the model, and it is a measured mechanism for the stairs-read-as-box confusion the eval
  harness put at 0.174 closed-loop against a 0.103 control.

  **This used to be the reason ``single_step_frac`` was 0 and ``n_steps_range`` started at 3, and
  that conclusion was REVERSED (2026-09-09).** The sensor argument is unchanged and still correct;
  what changed is what follows from it. A curb and a two-riser step-up are real deployment geometry,
  and if the robot cannot distinguish one from a low box then it has to survive the ambiguity rather
  than never meet it -- a library that omits them does not remove the confusion, it removes the
  training signal for it. ``n_steps_range`` is now ``[2, 7]``.

  **A ONE-riser curb is still absent, and for a different reason than the camera.** It is not
  authorable from a periodic stair reference at all: the entry gate anchors the stance foot onto
  tread 1 while ``exit_surf`` is ``upper_z = n_steps * riser``, so at ``n_steps == 1`` the entry and
  exit are the same surface and the traversal has zero length -- and a climbing stride never settles
  on the tread it just reached, so no gating change reaches it. It needs a step-up-and-settle solve.
  The gap is real: the smallest solved ``jump_on`` box is 0.25 m, a curb is 0.13-0.17 m.

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
