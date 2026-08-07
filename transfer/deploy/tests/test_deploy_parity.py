# Copyright 2026 Zachary Olkin. All rights reserved.

"""Parity: numpy deploy core (quat_math_np / transforms_np) vs the torch oracle.

Dev-only (imports torch). Asserts the numpy ports reproduce their torch counterparts
numerically, so the torch-free deployment runtime stays faithful to the validated path.

The two oracles differ on purpose:

* quaternion math -> ``transfer/torch_math/quat_math`` (isaaclab-free, so this half runs
  in a torch-only env), itself pinned against ``isaaclab.utils.math`` in that package's
  own parity test;
* frame transforms -> ``robot_rl.flow_matching.transforms``, the REAL implementation the
  training side uses. It pulls in isaaclab, so it is behind ``importorskip`` and this half
  skips where IsaacSim is not installed. Previously this came from a vendored copy in
  ``transfer/fm``; testing against the original removes the lock-step drift risk.

Run: ``python transfer/deploy/tests/test_deploy_parity.py``
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

_TRANSFER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _TRANSFER)

from deploy import quat_math_np as qn  # noqa: E402
from deploy import transforms_np as tn  # noqa: E402
from deploy import window_np as wn  # noqa: E402
from torch_math import quat_math as qt  # noqa: E402


def _rand_quat(n, g):
    q = torch.randn(n, 4, generator=g, dtype=torch.float64)
    return q / q.norm(dim=-1, keepdim=True)


def _close(name, a_np, b_torch, atol=1e-10):
    err = float(np.abs(a_np - b_torch.detach().numpy()).max())
    assert err < atol, f"{name}: max abs err {err:.3e} >= {atol:.1e}"
    print(f"  ok  {name:34s} max_err={err:.2e}")


def test_deploy_parity_quat_math() -> None:
    """The numpy quaternion ops match ``torch_math.quat_math`` on random inputs.

    Isaaclab-free on both sides, so this runs in a torch-only environment.
    """
    g = torch.Generator().manual_seed(0)
    n = 512
    q1, q2 = _rand_quat(n, g), _rand_quat(n, g)
    v = torch.randn(n, 3, generator=g, dtype=torch.float64)
    p1, p2 = torch.randn(n, 3, generator=g, dtype=torch.float64), torch.randn(n, 3, generator=g, dtype=torch.float64)
    np_ = lambda t: t.detach().numpy()

    # --- quat_math ---
    _close("quat_mul", qn.quat_mul(np_(q1), np_(q2)), qt.quat_mul(q1, q2))
    _close("quat_inv", qn.quat_inv(np_(q1)), qt.quat_inv(q1))
    _close("quat_apply", qn.quat_apply(np_(q1), np_(v)), qt.quat_apply(q1, v))
    _close("matrix_from_quat", qn.matrix_from_quat(np_(q1)), qt.matrix_from_quat(q1))
    _close("quat_box_minus", qn.quat_box_minus(np_(q1), np_(q2)), qt.quat_box_minus(q1, q2), atol=1e-8)
    mp_np, mq_np = qn.combine_frame_transforms(np_(p1), np_(q1), np_(p2), np_(q2))
    mp_t, mq_t = qt.combine_frame_transforms(p1, q1, p2, q2)
    _close("combine.p", mp_np, mp_t); _close("combine.q", mq_np, mq_t)
    sp_np, sq_np = qn.subtract_frame_transforms(np_(p1), np_(q1), np_(p2), np_(q2))
    sp_t, sq_t = qt.subtract_frame_transforms(p1, q1, p2, q2)
    _close("subtract.p", sp_np, sp_t); _close("subtract.q", sq_np, sq_t)
    # matrix->quat compared as rotation (q vs -q ambiguity)
    R = qt.matrix_from_quat(q1)
    _close("quat_from_matrix(asR)", qn.matrix_from_quat(qn.quat_from_matrix(np_(R))),
           qt.matrix_from_quat(qt.quat_from_matrix(R)), atol=1e-8)
    print("QUAT-MATH PARITY CHECKS PASSED")


def test_deploy_parity_transforms() -> None:
    """``transforms_np`` matches the REAL ``robot_rl.flow_matching.transforms``.

    Skipped where isaaclab / IsaacSim is not installed (the training-side module imports
    ``isaaclab.utils.math``).
    """
    # ``exc_type`` is required to skip (rather than error) when the module is FOUND but its
    # own ``import isaaclab`` fails -- which is exactly the no-IsaacSim case. Without it
    # pytest >= 9.1 turns this into a hard error.
    tt = pytest.importorskip(
        "robot_rl.flow_matching.transforms",
        reason="needs the isaaclab env for the training-side transforms oracle",
        exc_type=ImportError,
    )
    g = torch.Generator().manual_seed(0)
    n, nj, K = 512, 21, 4
    np_ = lambda t: t.detach().numpy()

    # --- transforms: relative_state round-trips through state_to_absolute ---
    pw, qw = torch.randn(n, 3, generator=g, dtype=torch.float64), _rand_quat(n, g)
    vw, ww = torch.randn(n, 3, generator=g, dtype=torch.float64), torch.randn(n, 3, generator=g, dtype=torch.float64)
    jp, jv = torch.randn(n, nj, generator=g, dtype=torch.float64), torch.randn(n, nj, generator=g, dtype=torch.float64)
    p_ref, q_ref = torch.randn(n, 3, generator=g, dtype=torch.float64), _rand_quat(n, g)
    # Extra frames are position-only in the state vector (see flow_matching.transforms).
    fp = torch.randn(n, K, 3, generator=g, dtype=torch.float64)

    rel_t = tt.relative_state(pw, qw, vw, ww, jp, jv, p_ref, q_ref, fp)
    rel_np = tn.relative_state(np_(pw), np_(qw), np_(vw), np_(ww), np_(jp), np_(jv), np_(p_ref), np_(q_ref),
                               np_(fp))
    _close("relative_state", rel_np, rel_t, atol=1e-9)
    assert rel_t.shape[-1] == tn.target_state_dim(nj, K) == 2 * nj + 15 + 3 * K

    abs_t = tt.state_to_absolute(rel_t, p_ref, q_ref, nj, num_extra_frames=K)
    abs_np = tn.state_to_absolute(rel_np, np_(p_ref), np_(q_ref), nj, num_extra_frames=K)
    for key in ("root_pos_w", "root_link_lin_vel_w", "frame_pos_w", "joint_pos"):
        _close(f"state_to_absolute[{key}]", abs_np[key], abs_t[key], atol=1e-8)
    # quat fields compared as rotation
    _close("state_to_absolute[root_quat asR]",
           qn.matrix_from_quat(abs_np["root_quat_w"]), qt.matrix_from_quat(abs_t["root_quat_w"]), atol=1e-7)
    print("TRANSFORMS PARITY CHECKS PASSED")


def test_deploy_parity_window():
    """``window_np.world_to_window`` reproduces ``flow_matching.window.world_to_window``.

    The reference window is the representation the policy observes AND the one the
    generator predicts, so a divergence between the two implementations of it is a
    sim2real gap that no shape check can see -- both sides produce a correctly-sized
    observation either way.  The ``base_pose`` block is the sharp edge: both halves are the
    reference pose in the anchor's frame, and the orientation half is a 6D rotation whose
    two columns have to come out in the same order on both sides.
    """
    wt = pytest.importorskip(
        "robot_rl.flow_matching.window",
        reason="needs the isaaclab env for the training-side window oracle",
        exc_type=ImportError,
    )
    g = torch.Generator().manual_seed(1)
    n, k, nj, K = 6, 11, 21, 4
    np_ = lambda t: t.detach().numpy()

    act_p = torch.randn(n, 1, 3, generator=g, dtype=torch.float64)
    act_q = _rand_quat(n, g)[:, None, :]
    ref_p = torch.randn(n, k, 3, generator=g, dtype=torch.float64)
    ref_q = _rand_quat(n * k, g).view(n, k, 4)
    ref_l = torch.randn(n, k, 3, generator=g, dtype=torch.float64)
    ref_a = torch.randn(n, k, 3, generator=g, dtype=torch.float64)
    jp = torch.randn(n, k, nj, generator=g, dtype=torch.float64)
    jv = torch.randn(n, k, nj, generator=g, dtype=torch.float64)
    ee = torch.randn(n, k, K, 3, generator=g, dtype=torch.float64)

    win_t = wt.world_to_window(act_p, act_q, ref_p, ref_q, ref_l, ref_a, jp, jv, ee)
    win_np = wn.world_to_window(np_(act_p), np_(act_q), np_(ref_p), np_(ref_q), np_(ref_l),
                                np_(ref_a), np_(jp), np_(jv), np_(ee))
    _close("world_to_window", win_np, win_t, atol=1e-9)

    # The layout tables must agree too -- the numpy side restates them, and a silent
    # disagreement would slice the right width out of the wrong columns.
    assert wn.window_blocks(nj, K) == [(n_, w) for n_, w in wt.window_blocks(nj, K)]
    assert wn.window_dim(nj, K) == wt.window_dim(nj, K) == win_t.shape[-1]
    assert wn.window_block_slices(nj, K) == wt.window_block_slices(nj, K)

    # A teacher may pack a SUBSET, and both sides have to drop the same columns -- the
    # deploy builds the clip-path window from its tracker's list, so a port that ignored
    # the selection would hand the policy a differently-shaped (or differently-ordered)
    # vector than the one it trained on.
    slim = ("base_pose", "base_vel", "joint_pos")
    slim_t = wt.world_to_window(act_p, act_q, ref_p, ref_q, ref_l, ref_a, jp, jv, ee,
                                blocks=slim)
    slim_np = wn.world_to_window(np_(act_p), np_(act_q), np_(ref_p), np_(ref_q), np_(ref_l),
                                 np_(ref_a), np_(jp), np_(jv), np_(ee), blocks=slim)
    _close("world_to_window[slim]", slim_np, slim_t, atol=1e-9)
    assert wn.window_blocks(nj, K, slim) == [(n_, w) for n_, w in wt.window_blocks(nj, K, slim)]
    assert wn.window_dim(nj, K, slim) == wt.window_dim(nj, K, slim) == slim_t.shape[-1]
    assert wn.window_block_slices(nj, K, slim) == wt.window_block_slices(nj, K, slim)
    # Naming the blocks out of order still packs them in canonical order, on both sides.
    assert (wn.resolve_reference_blocks(("joint_pos", "base_pose", "base_vel"))
            == wt.resolve_reference_blocks(("joint_pos", "base_pose", "base_vel")) == slim)
    print("WINDOW PARITY CHECKS PASSED")


if __name__ == "__main__":
    test_deploy_parity_quat_math()
    test_deploy_parity_transforms()
    test_deploy_parity_window()
    print("ALL DEPLOY-CORE PARITY CHECKS PASSED")
