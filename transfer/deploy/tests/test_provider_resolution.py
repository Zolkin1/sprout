# Copyright 2026 Zachary Olkin. All rights reserved.

"""Which execution provider a request resolves to, and which requests may fail.

Three stances share one function, and the difference between them is the point:

* ``None`` -- CPU, unconditionally. The DEPLOYMENT default: on hardware the provider is
  part of what a run claims about itself, so it is stated, never discovered. A GPU that
  appears because a wheel happened to be installed is exactly what this forbids.
* a NAMED provider -- taken, or refused loudly. Asking for CUDA on a CPU-only wheel is a
  mistake worth stopping the run for, since the alternative is silently missing the
  control period by 5x.
* ``"auto"`` -- the GPU when there is one, else the CPU, and never an error. The
  SIMULATION default (``gen_tracker_runner.py``), where the generator costs ~100 ms on a
  CPU and ~18 ms on a GPU, and "this box has no CUDA" is an answer rather than a fault.

``ort.get_available_providers`` reports what the WHEEL was built with, so it differs
between machines; every assertion here is written against what it returns rather than
against a fixed provider list.
"""

from __future__ import annotations

import os
import sys

import onnxruntime as ort
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "transfer"))

from deploy.tracker_session import resolve_providers  # noqa: E402

_CUDA = "CUDAExecutionProvider"
_CPU = "CPUExecutionProvider"


def test_none_is_cpu_only_whatever_the_wheel_offers():
    """The deployment default cannot be moved by installing a GPU wheel."""
    assert resolve_providers(None) == [_CPU]


def test_auto_takes_the_gpu_when_there_is_one_and_never_raises():
    """``auto`` resolves against what is installed, and has no failure mode."""
    got = resolve_providers(["auto"])
    if _CUDA in ort.get_available_providers():
        assert got == [_CUDA, _CPU]
    else:
        assert got == [_CPU]


def test_auto_is_case_insensitive_and_composes_with_named_providers():
    """It is one element of the list, not a whole-list mode."""
    assert resolve_providers(["AUTO"]) == resolve_providers(["auto"])
    assert resolve_providers(["auto", "cpu"])[-1] == _CPU


def test_cpu_is_always_the_last_resort():
    """Every resolution ends in CPU, so an op the accelerator lacks still runs."""
    for request in (["auto"], ["cpu"], None):
        assert resolve_providers(request)[-1] == _CPU


def test_a_named_provider_the_wheel_lacks_still_raises():
    """The loud failure ``auto`` opts out of is still there for a stated provider."""
    with pytest.raises(ValueError, match="onnxruntime has no"):
        resolve_providers(["NoSuchExecutionProvider"])
