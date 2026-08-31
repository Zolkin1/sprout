# Copyright 2026 Zachary Olkin. All rights reserved.

"""The physical randomization the Isaac rollouts carried, reproduced in MuJoCo.

``V_priv`` was fitted on ``logs/rollouts/v2/...``, and mjeval runs one nominal robot, so the
critic reads a robot it never saw.  This closes that gap -- as far as it can be closed.

**What the recording ACTUALLY had, which is less than it looks.**  ``record_rollouts.py``
never imports ``dr_presets``; the ``dr_presets: none,train`` in that directory's
``manifest.json`` belongs to a separate ``evaluate.py --stage plan`` that shares the folder.
So there was no preset, and:

* **seven physical terms were active, all at STARTUP** -- drawn once per env, never re-drawn.
  The recording is 128 fixed robots across all 342,711 windows;
* ``push_robot`` is ``None`` and ``base_external_force_torque`` has a zero range, so there
  were **no pushes**;
* ``enable_corruption=False`` on both observation groups, so there was **no observation noise
  and no reference-window noise**;
* the depth noiser is never constructed (its only builder event is nulled), so there was **no
  depth noise**;
* dynamic and viscous joint friction are **dead code**, commented out behind a TODO, so the
  ranges the config declares for them never reached the sim.

That is why this module randomizes physics only.  There is nothing time-varying to match.

**Per EPISODE here, per ENV there.**  MuJoCo has no persistent env to carry a startup draw, so
the population is reproduced by drawing per episode.  That matches the DISTRIBUTION, not any
particular robot -- and it cannot match a particular one, because the drawn values were never
recorded (the rollout carries ``env_id`` but no DR columns).

**Ground friction is deliberately absent.**  Isaac's ``randomize_ground_contact_friction`` was
nulled by the tracker play trim, so it is not part of the target.  That is fortunate: MuJoCo
combines contact friction as the elementwise MAX of the two geoms here (``npair=0``,
``geom_priority`` all 0), so a foot-scoped draw below the terrain's 1.0 would be silently
discarded, and porting Isaac's foot scoping literally could only ever RAISE friction.

**A semantic caveat that no amount of care removes.**  Isaac's joint friction coefficient and
MuJoCo's ``dof_frictionloss`` are similar in spirit -- both resist motion at the joint -- but
they are not the same parameter, and the ranges are not known to be commensurate.  The other
six terms (armature, gains, mass, CoM, joint-default offset) are dimensionally identical
across the two engines.  Treat the friction term as approximate and the rest as matched.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

#: The seven live terms, with Isaac's own ranges and operations.
#: Sources: ``g1_clf_tracking_base.py:144-271`` for the terms, ``physical_randomization.py``
#: for the ops. Every distribution is UNIFORM -- there is no log-uniform or gaussian anywhere
#: in the target, so a "distribution match" here is a range match.
ISAAC_DR = {
    "joint_friction": {"range": (0.3, 1.6), "op": "add", "target": "dof_frictionloss"},
    "joint_armature": {"range": (0.95, 1.05), "op": "scale", "target": "dof_armature"},
    "stiffness": {"range": (0.9, 1.1), "op": "scale", "target": "actuator_kp"},
    "damping": {"range": (0.9, 1.1), "op": "scale", "target": "actuator_kd"},
    "torso_mass": {"range": (0.85, 1.15), "op": "scale", "target": "body_mass"},
    "joint_default_pos": {"range": (-0.01, 0.01), "op": "add", "target": "default_angles"},
    "torso_com_xy": {"range": (-0.05, 0.05), "op": "set", "target": "body_ipos"},
    "torso_com_z": {"range": (-0.01, 0.01), "op": "set", "target": "body_ipos"},
    "backpack_com": {"range": (-0.02, 0.02), "op": "set", "target": "body_ipos"},
}

#: Bodies whose mass Isaac scaled together.
MASS_BODIES = ("torso_link", "backpack_thor_link", "backpack_zed_link")


@dataclass
class DrDraw:
    """One episode's parameter draw, in the units the sim takes them.

    Attributes:
        joint_friction: ``(29,)`` per-joint friction loss.
        joint_armature: ``(29,)`` multipliers on the nominal armature.
        stiffness / damping: ``(29,)`` multipliers on the POLICY's exported gains.
        mass_scale: One multiplier shared by :data:`MASS_BODIES`, as Isaac applied it.
        joint_default_offset: ``(29,)`` radians added to the action offset.
        torso_com / backpack_com: ``(3,)`` CoM offsets, metres.
        seed: What produced this draw.
    """

    joint_friction: np.ndarray
    joint_armature: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    mass_scale: float
    joint_default_offset: np.ndarray
    torso_com: np.ndarray
    backpack_com: np.ndarray
    seed: int

    def to_json(self) -> str:
        """Serialise for the record.

        Returns:
            A JSON string.
        """
        return json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                           for k, v in self.__dict__.items()})


def draw(rng: np.random.Generator, n_joints: int, seed: int) -> DrDraw:
    """Draw one episode's parameters from Isaac's ranges.

    Args:
        rng: Source of randomness.
        n_joints: Actuated joint count (29 on this robot).
        seed: Recorded on the draw so it can be reproduced.

    Returns:
        The draw.
    """
    u = lambda k, n=None: rng.uniform(*ISAAC_DR[k]["range"], size=n)  # noqa: E731
    return DrDraw(
        joint_friction=u("joint_friction", n_joints),
        joint_armature=u("joint_armature", n_joints),
        stiffness=u("stiffness", n_joints),
        damping=u("damping", n_joints),
        # ONE scale for all three bodies: Isaac's term lists them together, so drawing per
        # body would randomize their mass RATIO, which its draw does not.
        mass_scale=float(u("torso_mass")),
        joint_default_offset=u("joint_default_pos", n_joints),
        torso_com=np.array([*u("torso_com_xy", 2), *u("torso_com_z", 1)]),
        backpack_com=u("backpack_com", 3),
        seed=int(seed))


class MujocoDR:
    """Applies a :class:`DrDraw` to a MuJoCo model, re-basing from nominal every time.

    **``mj_resetData`` does not touch ``mjModel``**, so DR persists across episodes in this
    harness.  Every field is therefore restored from a pristine copy captured at construction
    before the new draw is applied -- without that, a per-episode ``scale`` draw compounds and
    100 episodes of ``U(0.9, 1.1)`` is a log-random walk rather than a re-draw.

    Args:
        robot: The ``transfer.sim.robot.Robot``, already loaded.
        policy: The ``GenTrackerPolicy``, for the nominal gains and joint order.
    """

    def __init__(self, robot, policy) -> None:
        import mujoco  # noqa: PLC0415

        self._mj = mujoco
        m = robot.mj_model
        self.robot, self.policy = robot, policy
        self.joint_names = list(policy.get_joint_names())
        # Joint -> DOF. `dof_armature` is indexed by DOF and nv (35) exceeds nu (29) by the
        # free joint's six, so indexing it with a joint id silently perturbs the wrong joint.
        self.dof = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                             for n in self.joint_names], dtype=np.int64)
        self.mass_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
                         for b in MASS_BODIES]
        self.torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.pack_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
                         for b in ("backpack_thor_link", "backpack_zed_link")]
        if min([*self.mass_ids, self.torso_id, *self.pack_ids]) < 0:
            raise ValueError(f"model is missing one of {MASS_BODIES}; this is not the "
                             "perceptive robot and the CoM/mass terms would hit body -1")
        self.nominal = {
            "dof_armature": np.array(m.dof_armature, copy=True),
            "dof_frictionloss": np.array(m.dof_frictionloss, copy=True),
            "body_mass": np.array(m.body_mass, copy=True),
            "body_inertia": np.array(m.body_inertia, copy=True),
            "body_ipos": np.array(m.body_ipos, copy=True),
        }
        self.nominal_kp = np.array(policy.get_kp(), dtype=np.float64, copy=True)
        self.nominal_kd = np.array(policy.get_kd(), dtype=np.float64, copy=True)

    def restore_nominal(self) -> None:
        """Put every randomized field back to its pristine value."""
        m = self.robot.mj_model
        for k, v in self.nominal.items():
            getattr(m, k)[:] = v
        self._mj.mj_setConst(m, self.robot.mj_data)

    def apply(self, d: DrDraw) -> None:
        """Apply one draw, from nominal.

        Args:
            d: The draw.
        """
        m, data = self.robot.mj_model, self.robot.mj_data
        self.restore_nominal()
        m.dof_frictionloss[self.dof] = self.nominal["dof_frictionloss"][self.dof] + d.joint_friction
        m.dof_armature[self.dof] = self.nominal["dof_armature"][self.dof] * d.joint_armature
        for b in self.mass_ids:
            m.body_mass[b] = self.nominal["body_mass"][b] * d.mass_scale
            # Inertia scales WITH mass. MuJoCo does not do this for you: scaling body_mass
            # alone leaves the original inertia tensor, which is a different (unphysical)
            # body rather than a heavier one.
            m.body_inertia[b] = self.nominal["body_inertia"][b] * d.mass_scale
        m.body_ipos[self.torso_id] = self.nominal["body_ipos"][self.torso_id] + d.torso_com
        for b in self.pack_ids:
            m.body_ipos[b] = self.nominal["body_ipos"][b] + d.backpack_com
        # body_subtreemass stays stale until this runs; it is what the cost-of-transport
        # denominator reads.
        self._mj.mj_setConst(m, data)
        # Gains LAST and from the policy's own exported values, because `apply_policy_gains`
        # writes them ABSOLUTELY -- running it after this would discard the draw entirely,
        # and running the draw without re-basing would compound it across episodes.
        self.robot.apply_policy_gains(self.nominal_kp * d.stiffness,
                                      self.nominal_kd * d.damping, self.joint_names)

    def total_mass(self) -> float:
        """Current total mass, for refreshing ``CourseRig.total_mass`` after a draw.

        That value is captured per COURSE but read per EPISODE as the cost-of-transport
        denominator, so a per-episode mass draw makes it stale.

        Returns:
            Sum of ``body_mass``.
        """
        return float(np.sum(self.robot.mj_model.body_mass))
