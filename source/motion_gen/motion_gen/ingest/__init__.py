# Copyright 2026 Zachary Olkin. All rights reserved.

"""Readers that turn a reference-motion library on disk into tensors.

:mod:`clean_ingest` is the live one: ``scripts/rsl_rl/export_stride_bank.py`` (stage 0 of
the motionbricks pipeline) uses it to pull steady strides out of the HF skill library.
:mod:`terrain_ingest` reads the terrain skills the same way.  :mod:`motion_db` defines the
common ``MotionSource`` contract, :mod:`ingest_common` the shared resampling helpers, and
:mod:`bones_ingest` reads the raw bones-seed CSV/HDF5 (its feature-cache CLI belongs to the
deprecated matcher, but ``clean_ingest`` reuses its joint-remap and resample helpers).
"""
