# Copyright 2026 Zachary Olkin. All rights reserved.

"""The two copies of the tracking-error contract must not drift.

``transfer.deploy.tracking_error`` (numpy, torch-free, runs in ``env_transfer``) and
``robot_rl...mdp.tracking_error`` (torch, folds every control step over up to 1024 envs) exist
separately because ``robot_rl`` must not import ``transfer`` and a numpy round-trip inside the
IsaacLab loop would be a device sync per step.  That is the same "duplicate and pin with a test"
rule ``transfer/sim/robots/g1/make_g1_mjcf.py`` states for the actuator table.

What is pinned, and each check matters on its own:

* the **constants** -- the bootstrap spec field-for-field, and the channel layout.  A row is
  written to disk by one side and read by the other, so a reordered channel is a plausible
  number in the wrong column rather than an error;
* the **functions** -- error row, violation, per-step reward, chunk reward and the normalized
  return, on random states, to float tolerance;
* the two **semantic** properties the design rests on, checked on both implementations rather
  than argued: a pure yaw offset is never a violation, and a successful episode's return does
  not depend on how long it took.

``robot_rl`` needs no simulator to import (``mdp`` pulls 152 ``isaaclab.*`` modules and zero
``isaacsim``/``omni``/``pxr``/``carb``), but it does need ``env_isaaclab``, so the torch half is
``importorskip``-guarded the way ``test_fk_parity.py`` guards ``motion_gen``.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     os.pardir, os.pardir, os.pardir))
for _p in (_ROOT, os.path.join(_ROOT, "source", "robot_rl"),
           os.path.join(_ROOT, "source", "motion_gen")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transfer.deploy import tracking_error as np_te  # noqa: E402

torch = pytest.importorskip("torch")
t_te = pytest.importorskip("robot_rl.tasks.manager_based.robot_rl.mdp.tracking_error")

#: Joint count of the g1_perceptive robot. Only the row width depends on it.
_NJ = 29

#: Tolerance. Both sides run float64 over the same algebra, but `acos` near +-1 and the
#: quaternion normalisation differ in the last ulp, so this is not exact-equality territory.
_ATOL = 1e-12


def _random_states(n: int, seed: int) -> dict[str, np.ndarray]:
    """A batch of reference/actual states, with quaternions on the unit sphere.

    Args:
        n: Batch size.
        seed: RNG seed.

    Returns:
        The ten arrays :func:`tracking_error` takes, keyed by argument name.
    """
    rng = np.random.default_rng(seed)

    def quat(k: int) -> np.ndarray:
        """``[k, 4]`` unit xyzw quaternions."""
        q = rng.normal(size=(k, 4))
        return q / np.linalg.norm(q, axis=-1, keepdims=True)

    return {
        "ref_pos_w": rng.normal(size=(n, 3)),
        "ref_quat_w": quat(n),
        "ref_lin_vel_w": rng.normal(size=(n, 3)),
        "ref_ang_vel_w": rng.normal(size=(n, 3)),
        "ref_joint_pos": rng.normal(size=(n, _NJ)),
        "act_pos_w": rng.normal(size=(n, 3)),
        "act_quat_w": quat(n),
        "act_lin_vel_w": rng.normal(size=(n, 3)),
        "act_ang_vel_w": rng.normal(size=(n, 3)),
        "act_joint_pos": rng.normal(size=(n, _NJ)),
    }


def _as_torch(state: dict[str, np.ndarray]) -> dict[str, "torch.Tensor"]:
    """The same batch as float64 tensors."""
    return {k: torch.tensor(v, dtype=torch.float64) for k, v in state.items()}


def test_the_bootstrap_specs_are_identical():
    """Field for field, including provenance -- one side moving a default is the bug."""
    assert np_te.BOOTSTRAP_SPEC.to_dict() == t_te.BOOTSTRAP_SPEC.to_dict()


def test_the_channel_layout_is_identical():
    """A reordered channel is a plausible number in the wrong column, never an error."""
    assert np_te.ERROR_CHANNEL_NAMES == t_te.ERROR_CHANNEL_NAMES
    assert np_te.NUM_FIXED_CHANNELS == t_te.NUM_FIXED_CHANNELS
    assert np_te.error_channel_names(_NJ) == t_te.error_channel_names(_NJ)


def test_the_error_rows_agree():
    """The whole row, on random states, both frames and both angle channels."""
    state = _random_states(64, seed=7)
    a = np_te.tracking_error(**state)
    b = t_te.tracking_error(**_as_torch(state)).numpy()
    assert a.shape == (64, np_te.NUM_FIXED_CHANNELS + _NJ)
    assert np.abs(a - b).max() < _ATOL, f"max |diff| = {np.abs(a - b).max():.3e}"


def test_the_reward_and_the_violation_agree():
    """Both readings of the row, under a non-default spec so no constant is load-bearing."""
    spec_kw = dict(max_lateral_m=0.25, max_height_m=0.18, max_tilt_rad=math.radians(22.0),
                   sigma_pos_m=0.07, sigma_ori_rad=math.radians(9.0), sigma_lin_vel=0.4,
                   sigma_ang_vel=1.3, sigma_joint_rad=0.11,
                   weights=(0.3, 0.2, 0.15, 0.35), gamma=0.9,
                   provenance={"bootstrap": False, "test": True})
    np_spec, t_spec = np_te.TrackingErrorSpec(**spec_kw), t_te.TrackingErrorSpec(**spec_kw)

    state = _random_states(64, seed=11)
    # Scaled down so a good fraction of the batch lands on EACH side of the bounds -- random
    # normal states violate essentially always, which would pass with the test inverted.
    err_np = np_te.tracking_error(**state) * 0.25
    err_t = torch.tensor(err_np, dtype=torch.float64)

    v_np, v_t = np_te.violates(err_np, np_spec), t_te.violates(err_t, t_spec).numpy()
    assert 0 < int(v_np.sum()) < v_np.size, "degenerate batch: every row on one side of the bound"
    assert np.array_equal(v_np, v_t)

    assert np_te.VIOLATION_CHECK_NAMES == t_te.VIOLATION_CHECK_NAMES
    for check in np_te.VIOLATION_CHECK_NAMES:
        assert np.array_equal(np_te.violation_checks(err_np, np_spec)[check],
                              t_te.violation_checks(err_t, t_spec)[check].numpy())

    r_np = np_te.step_reward(err_np, np_spec)
    r_t = t_te.step_reward(err_t, t_spec).numpy()
    assert np.all((r_np > 0.0) & (r_np <= 1.0))
    assert np.abs(r_np - r_t).max() < _ATOL

    chunk_np = np_te.chunk_reward(err_np.reshape(8, 8, -1), np_spec)
    chunk_t = t_te.chunk_reward(err_t.reshape(8, 8, -1), t_spec).numpy()
    assert np.abs(chunk_np - chunk_t).max() < _ATOL

    rewards = np.abs(np.sin(np.arange(37) * 0.7))
    for failed in (True, False):
        assert np_te.normalized_return(rewards, np_spec, failed=failed) == pytest.approx(
            t_te.normalized_return(rewards, t_spec, failed=failed), abs=_ATOL)


def test_a_pure_yaw_offset_is_never_a_violation():
    """Yaw is graded by the reward and can never end an episode, on BOTH sides.

    The anchor divides heading out in full, so yaw error is purely chunk-local drift -- the one
    check a soft term handles better than a hard bound. It is also the property most likely to be
    undone by someone "tidying" `violates` into a single geodesic angle, which is why it is
    asserted rather than left to the docstring.
    """
    zero, identity = np.zeros((1, 3)), np.array([[0.0, 0.0, 0.0, 1.0]])
    joints = np.zeros((1, _NJ))
    for deg in (30.0, 90.0, 175.0):
        half = math.radians(deg) / 2.0
        yawed = np.array([[0.0, 0.0, math.sin(half), math.cos(half)]])
        args = dict(ref_pos_w=zero, ref_quat_w=identity, ref_lin_vel_w=zero,
                    ref_ang_vel_w=zero, ref_joint_pos=joints, act_pos_w=zero,
                    act_quat_w=yawed, act_lin_vel_w=zero, act_ang_vel_w=zero,
                    act_joint_pos=joints)
        err = np_te.tracking_error(**args)
        assert err[0, 3] == pytest.approx(0.0, abs=1e-12), "a pure yaw moved the TILT channel"
        assert abs(err[0, 4]) == pytest.approx(math.radians(deg) if deg < 180 else 0.0, abs=1e-9)
        assert not bool(np_te.violates(err)[0])
        assert not bool(t_te.violates(t_te.tracking_error(**_as_torch(args)))[0])
        # ...but it IS penalised, or the reward would be blind to heading.
        assert float(np_te.step_reward(err)[0]) < 1.0


def test_a_pure_tilt_beyond_the_bound_is_a_violation():
    """The other half of the same statement: the cone bites, and only on the tilt check."""
    zero, identity = np.zeros((1, 3)), np.array([[0.0, 0.0, 0.0, 1.0]])
    joints = np.zeros((1, _NJ))
    half = math.radians(40.0) / 2.0
    pitched = np.array([[0.0, math.sin(half), 0.0, math.cos(half)]])
    args = dict(ref_pos_w=zero, ref_quat_w=identity, ref_lin_vel_w=zero, ref_ang_vel_w=zero,
                ref_joint_pos=joints, act_pos_w=zero, act_quat_w=pitched, act_lin_vel_w=zero,
                act_ang_vel_w=zero, act_joint_pos=joints)
    err = np_te.tracking_error(**args)
    assert err[0, 3] == pytest.approx(math.radians(40.0), abs=1e-9)
    checks = np_te.violation_checks(err)
    assert bool(checks["tilt"][0])
    assert not bool(checks["lateral"][0]) and not bool(checks["height"][0])
    assert bool(t_te.violates(t_te.tracking_error(**_as_torch(args)))[0])


def test_a_downrange_excursion_is_never_a_violation():
    """Downrange is graded and never fatal, however large -- on BOTH sides.

    The generator moves the reference along the direction of travel to MATCH THE TERRAIN: a
    re-plan that puts the reference on the next tread jumps it forward, and the tracker cannot
    then be anything but behind it. Bounding that would end episodes for the one thing the
    generator is supposed to do on stairs. Measured on tile 20, downrange excursions reach
    0.30-0.45 m on episodes whose lateral and height never pass 0.09 m.

    Asserted rather than left to the docstring because "the position box" is a natural thing
    to re-lump into a single per-axis test, and doing so would silently restore the bound.
    """
    identity, joints = np.array([[0.0, 0.0, 0.0, 1.0]]), np.zeros((1, _NJ))
    zero = np.zeros((1, 3))
    for downrange in (0.3, 1.0, 5.0):
        args = dict(ref_pos_w=zero, ref_quat_w=identity, ref_lin_vel_w=zero, ref_ang_vel_w=zero,
                    ref_joint_pos=joints, act_pos_w=np.array([[downrange, 0.0, 0.0]]),
                    act_quat_w=identity, act_lin_vel_w=zero, act_ang_vel_w=zero,
                    act_joint_pos=joints)
        err = np_te.tracking_error(**args)
        assert err[0, 0] == pytest.approx(downrange, abs=1e-12)
        assert not bool(np_te.violates(err)[0]), f"{downrange} m downrange ended an episode"
        assert not bool(t_te.violates(t_te.tracking_error(**_as_torch(args)))[0])
        # ...but it IS penalised, or the reward would be blind to it.
        assert float(np_te.step_reward(err)[0]) < 1.0


def test_lateral_and_height_are_bounded_independently():
    """The two surviving position checks fire separately and are named separately."""
    identity, joints = np.array([[0.0, 0.0, 0.0, 1.0]]), np.zeros((1, _NJ))
    zero = np.zeros((1, 3))
    spec = np_te.TrackingErrorSpec(max_lateral_m=0.2, max_height_m=0.1)
    t_spec = t_te.TrackingErrorSpec(max_lateral_m=0.2, max_height_m=0.1)
    for offset, fires, quiet in (([0.0, 0.3, 0.0], "lateral", "height"),
                                 ([0.0, 0.0, 0.15], "height", "lateral")):
        args = dict(ref_pos_w=zero, ref_quat_w=identity, ref_lin_vel_w=zero,
                    ref_ang_vel_w=zero, ref_joint_pos=joints,
                    act_pos_w=np.array([offset]), act_quat_w=identity, act_lin_vel_w=zero,
                    act_ang_vel_w=zero, act_joint_pos=joints)
        err = np_te.tracking_error(**args)
        checks = np_te.violation_checks(err, spec)
        assert bool(checks[fires][0]) and not bool(checks[quiet][0])
        assert bool(t_te.violates(t_te.tracking_error(**_as_torch(args)), t_spec)[0])


def test_the_position_error_is_in_the_references_yaw_frame():
    """Displacement is named downrange/lateral/height against the REFERENCE's heading.

    Not world axes: a robot 0.25 m along a reference that is facing world +y is 0.25 m
    DOWNRANGE, and reading that as lateral would put the wrong axis in every report and let a
    per-axis box be tightened on the wrong one.
    """
    half = math.radians(90.0) / 2.0
    yawed = np.array([[0.0, 0.0, math.sin(half), math.cos(half)]])
    zero, joints = np.zeros((1, 3)), np.zeros((1, _NJ))
    args = dict(ref_pos_w=zero, ref_quat_w=yawed, ref_lin_vel_w=zero, ref_ang_vel_w=zero,
                ref_joint_pos=joints, act_pos_w=np.array([[0.0, 0.25, 0.0]]), act_quat_w=yawed,
                act_lin_vel_w=zero, act_ang_vel_w=zero, act_joint_pos=joints)
    err = np_te.tracking_error(**args)
    assert err[0, 0] == pytest.approx(0.25, abs=1e-12)      # downrange
    assert err[0, 1] == pytest.approx(0.0, abs=1e-12)       # lateral
    assert np.abs(err - t_te.tracking_error(**_as_torch(args)).numpy()).max() < _ATOL


def test_a_successful_return_does_not_depend_on_episode_length():
    """Flawless tracking scores 1.0 whether it took 1 chunk or 200, on BOTH sides.

    An absorbing tail on a NON-failure terminal would make a fast arrival score 0.46 against a
    slow one's 0.95 -- a statement about episode length wearing the label "tracking quality",
    and one the critic would happily learn.
    """
    for length in (1, 12, 60, 200):
        ones = np.ones(length)
        assert np_te.normalized_return(ones, failed=False) == pytest.approx(1.0, abs=1e-12)
        assert t_te.normalized_return(ones, failed=False) == pytest.approx(1.0, abs=1e-12)
        # A failure at the same point is worth strictly less, and less the earlier it is.
        assert np_te.normalized_return(ones, failed=True) < 1.0

    early = np_te.normalized_return(np.ones(4), failed=True)
    late = np_te.normalized_return(np.ones(40), failed=True)
    assert early < late, "failing early must be worth less than failing late"


def test_the_spec_refuses_what_it_cannot_mean():
    """Both sides reject the same malformed specs, including the load-bearing weight sum."""
    for mod in (np_te, t_te):
        with pytest.raises(ValueError, match="sum to 1"):
            mod.TrackingErrorSpec(weights=(0.5, 0.5, 0.5, 0.5))
        with pytest.raises(ValueError, match="must be positive"):
            mod.TrackingErrorSpec(max_lateral_m=-0.2)
        with pytest.raises(ValueError, match="must be positive"):
            mod.TrackingErrorSpec(sigma_pos_m=0.0)
        with pytest.raises(ValueError, match="gamma"):
            mod.TrackingErrorSpec(gamma=1.0)
        with pytest.raises(ValueError, match="unknown keys"):
            mod.TrackingErrorSpec.from_dict({"sigma_pos_m": 0.1, "sigma_position_m": 0.1})
        # The bound that was REMOVED must not be silently accepted as a field again -- a spec
        # carrying a downrange bound that nothing reads is exactly the trap this avoids.
        with pytest.raises(ValueError, match="unknown keys"):
            mod.TrackingErrorSpec.from_dict({"max_pos_box_m": [0.2, 0.2, 0.2]})


def test_the_spec_round_trips_through_a_dict():
    """`to_dict` / `from_dict` is how a spec reaches a checkpoint and a manifest."""
    for mod in (np_te, t_te):
        spec = mod.TrackingErrorSpec(max_lateral_m=0.1, max_height_m=0.3, gamma=0.9,
                                     provenance={"bootstrap": False, "n_episodes": 12})
        assert mod.TrackingErrorSpec.from_dict(spec.to_dict()) == spec
        assert not spec.is_bootstrap
        assert mod.BOOTSTRAP_SPEC.is_bootstrap
