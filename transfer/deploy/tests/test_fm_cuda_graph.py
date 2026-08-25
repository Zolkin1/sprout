# Copyright 2026 Zachary Olkin. All rights reserved.

"""CUDA Graph capture for the generator: that it is faster, and that it is not STALE.

Capture is what makes the generator fit its control period on the Jetson Thor (19.3 -> 14-15
ms per re-plan against a 20 ms tick), and it is only reachable because ``fm.onnx`` is now
exported at a fixed batch -- onnxruntime refuses capture outright while any node sits on the
host, which a symbolic batch axis guarantees.

The reason this file exists rather than a comment: a captured graph holds raw DEVICE
POINTERS, and every wrong way of feeding it fails **silently**. The measurements that
established that are encoded below as tests, because the failure mode -- "the plan stops
responding to the cameras" -- is indistinguishable on a walking robot from a generator whose
conditioning has not moved.

Needs a real ``fm.onnx`` and a CUDA-capable onnxruntime, so it skips cleanly without either:
point it at a run directory with ``G1_POLICY_DIR``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

ort = pytest.importorskip("onnxruntime", exc_type=ImportError)

from deploy.fm_session import load_fm_context, sample_window  # noqa: E402

_DEFAULT_POLICY_DIR = os.path.join(
    _HERE, "..", "..", "obelisk", "g1_control", "resource", "hf_cache",
    "zolkin_robot_rl", "policies", "2026-08-04_21-55-17_yaw_k12_v1")


def _fm_path() -> str:
    """The ``fm.onnx`` under test, or skip."""
    d = os.environ.get("G1_POLICY_DIR", _DEFAULT_POLICY_DIR)
    path = os.path.abspath(os.path.join(d, "fm.onnx"))
    if not os.path.exists(path):
        pytest.skip(f"no fm.onnx at {path}; set G1_POLICY_DIR")
    return path


def _require_cuda() -> None:
    """Skip unless this onnxruntime can actually place nodes on a GPU."""
    preload = getattr(ort, "preload_dlls", None)
    if preload is not None:
        try:
            preload()
        except Exception:                                              # noqa: BLE001
            pass
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("no CUDAExecutionProvider in this onnxruntime build")


def _conditioning(ctx, seed: int) -> dict:
    """A full feed for ``ctx``, dependent on ``seed`` so two of them differ."""
    rng = np.random.default_rng(seed)
    return {
        "history": rng.standard_normal((1, ctx.N, ctx.D)),
        "extero": rng.standard_normal((1, ctx.extero_flat_dim)),
        "cmd_tokens": rng.standard_normal((1, ctx.num_cmd_tokens, ctx.cmd_dim)),
        "noise": np.zeros((1, ctx.F, ctx.D)),
        "history_null": np.zeros((1,), dtype=bool),
    }


def _plan(ctx, feed: dict) -> np.ndarray:
    """One window from ``ctx`` for ``feed``."""
    return sample_window(ctx, feed["history"], feed["extero"], feed["cmd_tokens"],
                         feed["noise"], history_null=feed["history_null"])


def test_capture_is_exact_and_tracks_its_input():
    """A replay must equal the plain session AND follow its conditioning.

    Two assertions, and the second is the load-bearing one. Exactness says capture did not
    change the arithmetic -- it is the same kernels in the same order, submitted as one unit.
    Tracking says the per-call host-to-device copy is actually reaching the buffers the
    capture recorded; without it the replay returns the FIRST plan forever, bit-identical,
    with no error raised anywhere.
    """
    _require_cuda()
    path = _fm_path()
    plain = load_fm_context(path, providers=["cuda"])
    captured = load_fm_context(path, providers=["cuda"], cuda_graph=True)
    assert captured.binding is not None, "capture did not bind device buffers"

    a, b = _conditioning(plain, 1), _conditioning(plain, 2)
    want_a, want_b = _plan(plain, a), _plan(plain, b)
    # The premise: the two conditionings give genuinely different plans, so "replayed the old
    # answer" and "computed the new one" are distinguishable at all.
    assert np.abs(want_a - want_b).max() > 1e-3

    got_a, got_b = _plan(captured, a), _plan(captured, b)
    assert np.array_equal(got_a, want_a), "capture changed the result"
    assert np.array_equal(got_b, want_b), "the replay did not compute its own conditioning"
    assert np.abs(got_b - want_a).max() > 1e-3, (
        "the replay reproduced the FIRST plan: the host-to-device copy is not reaching the "
        "captured device buffers, so every later plan would be stale and silent")

    # Order must not matter either -- re-running the first conditioning after the second must
    # give the first answer again, not whatever was last in the buffers.
    assert np.array_equal(_plan(captured, a), want_a)


def test_a_capture_session_cannot_be_driven_by_plain_run():
    """``sess.run()`` on a capture-enabled session is the trap, so it is pinned here.

    It does not raise. It returns an EMPTY output list -- once the capture has been recorded,
    which ``enable_cuda_graph`` does at load time, every plain ``run()`` yields nothing at all.
    So a caller reaching for ``[0]`` gets an IndexError far from the cause, and a caller
    tolerant of a short list gets no plan and no error. This is why the capture path goes
    through the binding, and why ``make_session(cuda_graph=True)`` documents the session as
    usable only that way.
    """
    _require_cuda()
    ctx = load_fm_context(_fm_path(), providers=["cuda"], cuda_graph=True)
    feed = _conditioning(ctx, 3)
    raw = {k: np.asarray(v, dtype=np.float32) for k, v in feed.items() if k != "history_null"}
    raw["history_null"] = np.zeros((1, 1), dtype=np.float32)
    for attempt in range(2):
        assert len(ctx.sess.run(["trajectory"], raw)) == 0, (
            f"plain run() returned an output on attempt {attempt}. onnxruntime used to return "
            "nothing here once the graph was captured; if that changed, the binding path is no "
            "longer the only safe one and this test should say so rather than be deleted.")
    # ...while the bound path keeps working on the same session, so the emptiness above is a
    # property of how it was CALLED, not a session that has been broken.
    assert np.isfinite(_plan(ctx, feed)).all()


def test_capture_is_refused_without_cuda():
    """Asked for on the CPU, this is a contradiction rather than a slow mode -- so it raises."""
    with pytest.raises(ValueError, match="nothing to capture on the CPU"):
        load_fm_context(_fm_path(), providers=None, cuda_graph=True)
