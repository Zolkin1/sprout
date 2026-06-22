# Copyright 2026 Zachary Olkin. All rights reserved.

"""Parity test: vendored ``transfer/fm/quat_math.py`` vs ``isaaclab.utils.math``.

Dev-only (imports isaaclab); the deployed sim2sim code never imports isaaclab.
Run directly (``python transfer/fm/tests/test_quat_math_parity.py``) or via pytest.
Guarantees the vendored xyzw math reproduces the repo's isaaclab math bit-for-bit
(up to fp tolerance and the ``q`` vs ``-q`` sign ambiguity for matrix->quat).
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quat_math import (  # noqa: E402
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_box_minus,
    quat_conjugate,
    quat_from_matrix,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)

import isaaclab.utils.math as lab  # noqa: E402


def _rand_quat(n: int, gen: torch.Generator) -> torch.Tensor:
    """Random unit xyzw quaternions, shape ``(n, 4)``."""
    q = torch.randn(n, 4, generator=gen, dtype=torch.float64)
    return q / q.norm(dim=-1, keepdim=True)


def _assert_close(name: str, a: torch.Tensor, b: torch.Tensor, atol: float = 1e-9) -> None:
    err = (a - b).abs().max().item()
    assert err < atol, f"{name}: max abs err {err:.3e} exceeds {atol:.1e}"
    print(f"  ok  {name:34s} max_err={err:.2e}")


def test_quat_math_parity() -> None:
    """All vendored functions match isaaclab.utils.math on random inputs."""
    gen = torch.Generator().manual_seed(0)
    n = 4096
    q1 = _rand_quat(n, gen)
    q2 = _rand_quat(n, gen)
    v = torch.randn(n, 3, generator=gen, dtype=torch.float64)
    p1 = torch.randn(n, 3, generator=gen, dtype=torch.float64)
    p2 = torch.randn(n, 3, generator=gen, dtype=torch.float64)

    _assert_close("quat_mul", quat_mul(q1, q2), lab.quat_mul(q1, q2))
    _assert_close("quat_conjugate", quat_conjugate(q1), lab.quat_conjugate(q1))
    _assert_close("quat_inv", quat_inv(q1), lab.quat_inv(q1))
    _assert_close("quat_apply", quat_apply(q1, v), lab.quat_apply(q1, v))
    _assert_close("quat_apply_inverse", quat_apply_inverse(q1, v), lab.quat_apply_inverse(q1, v))
    _assert_close("matrix_from_quat", matrix_from_quat(q1), lab.matrix_from_quat(q1))
    _assert_close("quat_box_minus", quat_box_minus(q1, q2), lab.quat_box_minus(q1, q2), atol=1e-8)

    mine_p, mine_q = combine_frame_transforms(p1, q1, p2, q2)
    lab_p, lab_q = lab.combine_frame_transforms(p1, q1, p2, q2)
    _assert_close("combine_frame_transforms.p", mine_p, lab_p)
    _assert_close("combine_frame_transforms.q", mine_q, lab_q)

    mine_p, mine_q = subtract_frame_transforms(p1, q1, p2, q2)
    lab_p, lab_q = lab.subtract_frame_transforms(p1, q1, p2, q2)
    _assert_close("subtract_frame_transforms.p", mine_p, lab_p)
    _assert_close("subtract_frame_transforms.q", mine_q, lab_q)

    # matrix->quat: compare as rotations (q and -q are the same rotation), so
    # check the recovered rotation matrix rather than raw components.
    R = matrix_from_quat(q1)
    _assert_close("quat_from_matrix(roundtrip R)", matrix_from_quat(quat_from_matrix(R)), R, atol=1e-8)
    _assert_close(
        "quat_from_matrix vs lab (as rotation)",
        matrix_from_quat(quat_from_matrix(R)),
        matrix_from_quat(lab.quat_from_matrix(R)),
        atol=1e-8,
    )
    print("ALL PARITY CHECKS PASSED")


if __name__ == "__main__":
    test_quat_math_parity()
