# Copyright 2026 Zachary Olkin. All rights reserved.

"""DEPRECATED: motion matching over a cached candidate graph -- the clip generator that
preceded the Motion Bricks pipeline.

Retained for ablation studies only; nothing on the live pipeline imports this package.  See
``README.md`` in this directory for what each module does and how to run the matchers.

The geometry / ingest / clip-writing modules that used to live here are NOT deprecated and
moved out: :mod:`motion_gen.terrain`, :mod:`motion_gen.ingest`,
:mod:`motion_gen.clip_output`, :mod:`motion_gen.gen_parallel`, :mod:`motion_gen.playback`,
and :mod:`motion_gen.joint_names`.
"""
