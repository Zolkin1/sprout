# Copyright 2026 Zachary Olkin. All rights reserved.

"""The MuJoCo critic port: does it reproduce the fit, and can this test SEE it if it does not.

Two halves, and the second is what makes the first mean anything.

**Reproduction** -- ``critic.FailureCritic.p_fail`` against ``fit_value``'s own batcher and
model on the same real windows.  Measured 2.1e-07 mean / 1.1e-06 max, which is fp32
accumulation order between a batched and a single-sample forward.

**Resolving power** -- each named sabotage must MOVE ``p_fail`` by more than a stated margin.
A reproduction test with no sabotage arm is exactly the instrument the Isaac best-of-K harness
turned out to be: its same-noise control matched its real signal, so it had no power to
distinguish them and its conclusion was unsupported.  A margin that fails to bind is itself
the finding -- it says this test cannot see that class of error, and something structural has
to catch it instead.

**Two arms are documented as BLIND, and they are findings about the CRITIC rather than
failures of the port** (measured over 64 windows sampled across the whole recording)::

    scan zeroed              0.0704      uses the scan's CONTENT
    scan folded (16,11).T    0.0049      but barely its spatial LAYOUT   (14x less)
    history zeroed           0.0577      uses the history's CONTENT
    history rolled by one    0.0024      but barely its temporal ORDER   (24x less)

So a transposed scan fold and a mis-ordered history ring are both invisible to a p_fail delta.
They are caught by construction instead: :data:`critic.SCAN_HW` is ``(11, 16)`` and NOT square,
so a transposed fold raises on shape, and ``ProprioRing``/``BatchedHistory`` are both
newest-last by their own tests.  Stated here so nobody reads a green suite as covering them.

**Sampling matters and got this wrong once.**  On the FIRST 64 windows of the recording every
sabotage involving history reads exactly 0.0000 -- those are the opening re-plans of the first
episodes, ``history_null`` is ~1.0, the ring is all zeros and rolling zeros changes nothing.
Sample across the file (``history_null`` 0.031) or the suite silently tests nothing.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# transfer/sim and transfer/ must BOTH be on the path, the same bootstrap every other test
# in this directory uses: pytest imports transfer/sim/__init__.py while collecting here, and
# that file does `from sim.robot import Robot` -- a bare-name import that only resolves with
# transfer/ on the path. Without them collection fails with "No module named 'sim'", which
# names neither this file nor the real cause. Derived from __file__ rather than counting `..`
# hops from the repo root, so moving this file cannot silently break it.
_SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(os.path.dirname(_SIM))
for _p in (_SIM, os.path.dirname(_SIM), os.path.join(_SIM, "mjeval"),
           os.path.join(_ROOT, "source", "robot_rl"),
           os.path.join(_ROOT, "source", "motion_gen")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CKPT = os.path.join(_ROOT, "logs/value/v2/V_priv.fold0.pt")
ROLLOUT = os.path.join(
    _ROOT, "logs/rollouts/v2/rollout_2026-08-20_16-36-10_mode11_decor_plan_v1.h5")
N_WINDOWS = 64

pytestmark = pytest.mark.skipif(
    not (os.path.exists(CKPT) and os.path.exists(ROLLOUT)),
    reason="needs the v2 critic fit and its rollout; both are large local artifacts")


@pytest.fixture(scope="module")
def fitted():
    """The critic, a representative window sample, and the baseline ``p_fail``.

    Returns:
        ``(FailureCritic, windows dict, baseline p_fail array)``.
    """
    import h5py

    import critic as C

    fc = C.FailureCritic(CKPT, device="cpu")
    cols = ("history", "extero", "cmd_future", "history_null", "height_scan", "proprio",
            "goal_downrange_m", "goal_lateral_m", "goal_height_err_m", "base_twist_yaw")
    with h5py.File(ROLLOUT, "r") as f:
        idx = np.sort(np.random.default_rng(7).choice(
            f["history"].shape[0], N_WINDOWS, replace=False))
        W = {k: f[k][idx] for k in cols}
    # The sample must span the recording, not its opening. See the module docstring.
    assert W["history_null"].mean() < 0.25, (
        f"history_null is {W['history_null'].mean():.2f} in this sample -- these are opening "
        "re-plans with an empty ring, and every history sabotage reads 0.0 against them")
    base = np.array([fc.p_fail(_inputs(C, W, i)) for i in range(N_WINDOWS)])
    return fc, W, base


def _inputs(C, W, i, **override):
    """One :class:`critic.CriticInputs` from row ``i``, with optional field overrides.

    Args:
        C: The ``critic`` module.
        W: The window column dict.
        i: Row index.
        **override: Fields to replace, i.e. the sabotage.

    Returns:
        The inputs.
    """
    d = dict(
        history=W["history"][i], extero=W["extero"][i], cmd_future=W["cmd_future"][i],
        history_null=bool(W["history_null"][i]),
        goal=np.array([W["goal_downrange_m"][i], W["goal_lateral_m"][i],
                       W["goal_height_err_m"][i]]),
        twist=W["base_twist_yaw"][i],
        scan=np.asarray(W["height_scan"][i], np.float32).reshape(*C.SCAN_HW),
        proprio=W["proprio"][i])
    d.update(override)
    return C.CriticInputs(**d)


def test_reproduces_the_fit(fitted):
    """``FailureCritic`` must agree with ``fit_value``'s own pipeline on real windows."""
    import importlib.util

    fc, W, base = fitted
    spec = importlib.util.spec_from_file_location(
        "fv", os.path.join(_ROOT, "scripts/flow_matching/fit_value.py"))
    fv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fv)
    theirs = fv._predict(fc.model, fc.ctx, W, np.arange(N_WINDOWS), True, "cpu",
                         tuple(fc.scan_hw), 64)
    d = np.abs(base - theirs)
    assert d.max() < 1e-4, f"port disagrees with fit_value: max |dp| = {d.max():.3e}"


def test_the_no_goal_arm_drops_the_SAME_columns_on_both_sides(fitted):
    """The port and the fit must remove the goal identically, columns included.

    Only the assembled context is compared, not a ``p_fail``: this checkpoint's ``context_in``
    is 9 wide, so there is no forward pass to make. That is the point -- the hazard is not a
    shape error, it is one side dropping the goal from the FRONT and the other from the back,
    which leaves both widths at 6 and slides the twist into the goal's columns. Every number
    stays finite and plausible, and only a column-wise comparison can see it.
    """
    import importlib.util

    import torch

    import critic as C

    fc, W, _ = fitted
    spec = importlib.util.spec_from_file_location(
        "fv", os.path.join(_ROOT, "scripts/flow_matching/fit_value.py"))
    fv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fv)
    was = fc.goal_context
    try:
        fc.goal_context = False
        mine = torch.cat([fc._inputs(_inputs(C, W, i)).context
                          for i in range(N_WINDOWS)], dim=0)
    finally:
        fc.goal_context = was
    theirs = fv._batch(fc.ctx, W, np.arange(N_WINDOWS), True, "cpu",
                       tuple(fc.scan_hw), goal_context=False).context
    assert mine.shape == theirs.shape == (N_WINDOWS, 6), tuple(mine.shape)
    # And the surviving columns are the full twist, unmoved from where the with-goal arm
    # puts it: the goal is PREPENDED, so it owns 0:3 and nothing else shifts.
    with_goal = fv._batch(fc.ctx, W, np.arange(N_WINDOWS), True, "cpu",
                          tuple(fc.scan_hw), goal_context=True).context
    assert torch.allclose(mine, theirs, atol=0, rtol=0)
    assert torch.allclose(theirs, with_goal[:, 3:], atol=0, rtol=0)


def test_a_goal_context_stamp_that_contradicts_context_dim_is_REFUSED(tmp_path):
    """``goal_context: False`` on a 9-wide checkpoint must not load.

    ``load_state_dict`` cannot catch this: the checkpoint and the model are both built from
    ``context_dim``, so they agree with each other and disagree only with the stamp -- and the
    stamp is what decides which columns the batch carries.
    """
    import torch

    import critic as C

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    assert int(ck["context_dim"]) == 9 and bool(ck["privileged"])
    ck["goal_context"] = False
    bad = tmp_path / "V_priv_bad_stamp.pt"
    torch.save(ck, bad)
    with pytest.raises(ValueError, match="context_dim"):
        C.FailureCritic(str(bad), device="cpu")


#: ``(sabotage, measured mean |dp|, margin)`` over this module's own 64-window sample.
#:
#: **Margins are half the measured value, and that is the whole rule.**  Two blind arms
#: measure 0.0024 and 0.0049 (see the module docstring), so every margin here sits at least
#: 5x above that floor while leaving 2x of headroom below what the arm actually achieves --
#: which is what makes this a regression test rather than a re-statement of one run.  A
#: number that has to be LOWERED to stay green is a real change in what the critic reads,
#: not a flaky threshold: re-measure before touching it.
_SABOTAGE = [
    ("extero_double_normalized", 0.0915, 0.045),
    ("extero_channel_fold_transposed", 0.1527, 0.075),
    ("goal_downrange_negated", 0.1704, 0.085),
    ("proprio_joints_permuted", 0.0853, 0.042),
    ("proprio_zeroed", 0.1118, 0.055),
    ("scan_zeroed", 0.0704, 0.035),
    ("history_zeroed", 0.0577, 0.028),
    ("twist_zeroed", 0.1292, 0.064),
]


@pytest.mark.parametrize("name,measured,margin",
                         _SABOTAGE, ids=[s[0] for s in _SABOTAGE])
def test_the_suite_can_see_this_error(fitted, name, measured, margin):
    """Each sabotage must move ``p_fail``, or this suite cannot detect that error class."""
    stat = "mean"
    import torch

    import critic as C

    fc, W, base = fitted
    n29 = 29

    def sab(i):
        if name == "extero_double_normalized":
            e = fc.ctx.extero_norm.normalize(
                torch.as_tensor(np.asarray(W["extero"][i], np.float32).reshape(1, -1)))
            return {"extero": e.numpy().reshape(W["extero"][i].shape)}
        if name == "extero_channel_fold_transposed":
            e = np.asarray(W["extero"][i], np.float32)
            n_s, n_k = e.shape
            return {"extero": e.reshape(n_s, 5, n_k // 5).transpose(0, 2, 1).reshape(n_s, n_k)}
        if name == "goal_downrange_negated":
            return {"goal": np.array([-W["goal_downrange_m"][i], W["goal_lateral_m"][i],
                                      W["goal_height_err_m"][i]])}
        if name == "proprio_joints_permuted":
            perm = np.r_[np.random.default_rng(i).permutation(n29), np.arange(n29, 64)]
            return {"proprio": W["proprio"][i][:, perm]}
        if name == "proprio_zeroed":
            return {"proprio": np.zeros_like(W["proprio"][i])}
        if name == "scan_zeroed":
            return {"scan": np.zeros(C.SCAN_HW, np.float32)}
        if name == "history_zeroed":
            return {"history": np.zeros_like(W["history"][i])}
        if name == "twist_zeroed":
            return {"twist": np.zeros(6, np.float32)}
        raise AssertionError(name)

    p = np.array([fc.p_fail(_inputs(C, W, i, **sab(i))) for i in range(N_WINDOWS)])
    d = np.abs(p - base)
    v = d.mean() if stat == "mean" else d.max()
    assert v > margin, (
        f"{name}: {stat} |dp| = {v:.4f} <= {margin} (measured {measured:.4f} when this margin "
        "was set). This suite CANNOT see that error class, so the reproduction test above "
        "says nothing about it -- catch it structurally or record it as blind, the way the "
        "scan-transpose and history-roll arms are. Do not simply lower the margin: the blind "
        "floor is 0.005, so a fall to that level means the critic stopped reading the input.")


def test_the_scan_fold_is_not_square_so_a_transpose_cannot_be_silent():
    """The one structural guard covering an arm the sabotage suite is measurably blind to.

    A transposed scan fold moves ``p_fail`` by 0.0049 -- 14x less than zeroing the scan -- so
    no delta test can catch it.  What makes it safe is that ``(11, 16)`` is not square, so the
    wrong fold raises on shape instead.  If the grid ever becomes square this guard is gone
    and the error becomes genuinely silent, which is what this test is here to announce.
    """
    import critic as C

    h, w = C.SCAN_HW
    assert h != w, (
        f"SCAN_HW is {C.SCAN_HW}, which is SQUARE. A transposed fold is now shape-silent and "
        "the sabotage suite is measurably blind to it (0.0049 against a 0.05 margin). Add a "
        "positive orientation check -- see transfer/sim/tests/test_height_scan_grid.py, which "
        "pins the dense 121x121 grid against the coarse cast for exactly this reason.")
    assert h * w == 176, f"the coarse scan is 176 rays, {C.SCAN_HW} is {h * w}"

#: ``(name, root_quat_xyzw, expected gravity_b, expected yaw-frame twist of world [1,0,0])``.
#:
#: Canonical rotations, where the right answer is fixed by inspection rather than by another
#: implementation.  This exists because the first version of these helpers hand-rolled the
#: quaternion algebra and was WRONG -- finite, plausible, and off by a whole rotation.  They
#: now delegate to ``deploy.quat_math_np``, the tree's parity-tested xyzw port, and this pins
#: the two properties that make the delegation right:
#:
#: * a pure YAW must not tilt gravity (it is a rotation about the gravity axis), and
#: * ``yaw_quat`` of a pure roll or pitch is the identity, so the twist is unchanged there.
#:
#: Both would break under a wxyz/xyzw swap, which is the specific bug this replaced.
_ROTATIONS = [
    ("identity", [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]),
    ("yaw_90", [0.0, 0.0, 0.70710678, 0.70710678], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]),
    # R^T @ [0,0,-1] is minus the third ROW of R, which for a +90 deg roll is [0, 1, 0].
    ("roll_90", [0.70710678, 0.0, 0.0, 0.70710678], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]),
    ("pitch_90", [0.0, 0.70710678, 0.0, 0.70710678], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
]


@pytest.mark.parametrize("name,quat,grav,twist", _ROTATIONS, ids=[r[0] for r in _ROTATIONS])
def test_frame_helpers_on_canonical_rotations(name, quat, grav, twist):
    """``projected_gravity_b`` / ``twist_yaw`` must be right on rotations we can check by hand."""
    import critic as C

    q = np.array(quat)
    assert np.allclose(C.projected_gravity_b(q), grav, atol=1e-6), (
        f"{name}: gravity_b = {C.projected_gravity_b(q)}, expected {grav}. The runtime's "
        "root_quat is XYZW (build_state stores wxyz_to_xyzw(qpos[3:7])); a w-first read here "
        "is finite and plausible and rotates the whole frame.")
    got = C.twist_yaw(q, np.array([1.0, 0.0, 0.0]), np.zeros(3))
    assert np.allclose(got[:3], twist, atol=1e-6), f"{name}: twist = {got[:3]}, expected {twist}"
    assert np.allclose(got[3:], 0.0, atol=1e-6), "zero angular velocity must stay zero"
