# Copyright 2026 Zachary Olkin. All rights reserved.

"""Ballast is sliced to the parent's OWN cameras, or a near-only fine-tune trains on noise.

``train.py`` applies ``apply_camera_subset`` once, to the training set, and that function's
docstring explains why once is enough there: every consumer derives the sensor count from the
DATA's shape, so slicing ``data`` and ``meta`` in one place leaves the rest alone.

The ballast path loads its OWN recording, so "once" does not cover it. Without its own call, a
near-only parent (``extero_sensor_count=1``) is handed the recording's two-camera rows, because
``PretrainWindowAdapter`` takes ``extero_shape`` from the recording's attributes rather than from
the model. Measured on the v6g pair: the model wants 1x3900 and the recording holds (N, 2, 780).

That matters because ``multi.yaml`` is the one course carrying ballast (2000 chunks), so it is
exactly the arm a near-only-vs-two-camera comparison would want.
"""

from __future__ import annotations

import ast
import pathlib

_BALLAST = (pathlib.Path(__file__).resolve().parents[3]
            / "scripts/flow_matching/ballast.py")


def test_the_ballast_builder_applies_the_camera_subset() -> None:
    """``build_ballast_dataset`` slices the recording to the checkpoint's cameras."""
    src = _BALLAST.read_text()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "build_ballast_dataset")
    body = ast.get_source_segment(src, fn) or ""
    assert "apply_camera_subset" in body, (
        "build_ballast_dataset must slice the recording to the parent's own cameras; without "
        "it a near-only generator is fed two-camera rows and the ballast trains it on a "
        "sensor geometry it does not have")
    assert "extero_keep_cameras" in body, (
        "the subset must come from the CHECKPOINT's config, so it follows the parent "
        "automatically -- a two-camera parent carries None and the call is a no-op")


def test_the_subset_is_applied_before_the_dataset_is_built() -> None:
    """The slice happens between loading the windows and constructing the dataset.

    Order is the whole point: ``FutureStateDataset`` reads ``cfg.extero_xyz_offset`` per sensor
    found in the data, so an unsliced two-camera array against a one-camera config raises
    ``IndexError: list index out of range`` from deep inside the dataset -- which reads as a
    config bug rather than a missing slice.
    """
    src = _BALLAST.read_text()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "build_ballast_dataset")
    body = ast.get_source_segment(src, fn) or ""
    i_load = body.find("load_recorded_windows(")
    i_sub = body.find("apply_camera_subset(")
    i_ds = body.find("FutureStateDataset(")
    assert -1 not in (i_load, i_sub, i_ds), "expected load -> subset -> dataset in this order"
    assert i_load < i_sub < i_ds, (
        "apply_camera_subset must sit between load_recorded_windows and FutureStateDataset; "
        f"found load@{i_load} subset@{i_sub} dataset@{i_ds}")
