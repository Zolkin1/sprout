# Copyright 2026 Zachary Olkin. All rights reserved.

"""The reference-window state representation -- the one contract three subsystems share.

A *window node* is one timestep of reference, expressed relative to a measured robot pose
(the **anchor**). Per node, in packing order::

    [base_pose(9), base_vel(6), joint_pos(nj), joint_vel(nj), ee_pos(3 * K)]

with

* ``base_pose[0:3]`` -- ``R_yawT (p_ref - p_act)``, the reference base position relative to
  the anchor, in the anchor's YAW (heading) frame;
* ``base_pose[3:9]`` -- ``R_yawT R_ref`` in the 6D rotation representation (its first two
  COLUMNS), i.e. the reference base orientation expressed in the anchor's yaw frame;
* ``base_vel`` -- the reference base linear+angular velocity rotated into the yaw frame;
* ``joint_pos`` / ``joint_vel`` -- frame-invariant, in the **robot's** joint order;
* ``ee_pos`` -- each end-effector position relative to **that node's own reference base**
  (posture, free of tracking error), 3 per frame. This one uses the reference's FULL
  rotation, not a yaw frame: it is posture in the reference's own body frame, which is what
  posture means.

Both halves of ``base_pose`` are the same thing measured the same way: *the reference pose,
in the anchor's yaw frame*. They used to disagree -- position was ``ref - actual`` in the
robot frame while orientation was ``actual - ref`` in the reference frame, two vectors
pointing opposite ways packed as one error vector -- which is what
:data:`BASE_POSE_CONVENTION` records and refuses to mix.

.. admonition:: Why the anchor is YAW-ONLY

    ``R_yaw`` is ``yaw_quat(R_act)``: the anchor supplies heading, never roll or pitch. So a
    leaning robot does not rotate the frame its reference is observed in, and the reference's
    height above the terrain reads the same however the robot is tilted. Three reasons:

    * **The generator cannot observe its own tilt.** Its conditioning is the history, the
      depth and the command -- no proprioception, no ``projected_gravity``. Under a
      full-rotation anchor, part of its regression target would depend on a quantity it has
      no access to: irreducible noise in the target rather than a hard prediction.
    * **Terrain is gravity-defined.** Under a tilted frame the same physical clearance maps
      to a different ``z``, so both the policy and the generator would have to learn to undo
      the tilt purely to keep the reference off the floor.
    * **The rest of the pipeline is already yaw-only** -- ``motion_gen.world_lift`` is a
      yaw-only lift and the inverse of ``base_trajectory_cmd._align_yaw``; the height scanner
      is ``ray_alignment="yaw"``. This block was the outlier.

    Heading invariance survives -- yaw is still divided out -- so no world coordinate reaches
    the policy. The reference's OWN roll and pitch are still carried in full by
    ``base_pose[3:9]``: the yaw frame drops the anchor's tilt, never the reference's. That is
    the one semantic shift: the block means "the reference's orientation in my heading frame"
    (so, its absolute tilt against gravity) rather than "relative to my body".

.. admonition:: Why 6D rather than a log map

    ``base_pose[3:6]`` was a ``quat_box_minus`` rotation vector, whose magnitude saturates
    at ``pi``: a reference rotating more than ``pi`` rad relative to the anchor over the
    horizon **wrapped**, putting a jump discontinuity in both the generator's regression
    target and the policy's observation. The 6D representation has no branch cut, so the
    whole class of wrap defect is gone -- along with the dataset filter that existed to drop
    the affected training windows.

    The cost is that a raw 9-vector is not necessarily a rotation. Nothing here assumes it
    is: :func:`window_to_world` Gram-Schmidts on the way out (``rot6d_to_matrix``), and
    :func:`reanchor_window` is exactly linear in the two columns, so it neither needs nor
    silently applies a projection. The generator projects its own output once, where it is
    produced.

Everything here is relative, so no world coordinate survives into a window. The anchor is
divided out on the way in and only reappears if you deliberately call
:func:`window_to_world`.

Three producers share this module so the representation cannot drift between them:

1. ``mdp.observations.reference_window`` -- the policy observation, anchored on the robot now;
2. ``flow_matching.dataset`` -- the FM's training target *and* its autoregressive history,
   built offline from recorded world-frame arrays;
3. ``transfer.deploy.window_np`` -- the torch-free port, parity-tested against this one.

Because the FM predicts in exactly this representation, its output *is* the observation: no
world lift, in sim or at deploy.

All quaternions are ``(x, y, z, w)``, matching ``robot.data`` and ``isaaclab.utils.math``.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from isaaclab.utils.math import (
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_mul,
    yaw_quat,
)

from .transforms import (
    matrix_to_rot6d,
    quat_to_rot6d,
    rot6d_to_matrix,
    rot6d_to_quat,
    so3_exp,
    so3_log,
)

#: Content and frame of the ``base_pose`` block, carried into the exported tracker
#: parameters and checked at deploy load.
#:
#: The width change from 6 to 9 is caught on its own by every layout check
#: (``fm_session._check_state_layout``, ``deploy.gen_tracker._build_window``), so what this
#: string really guards is the SIGN: flipping the orientation half was invisible in the
#: observation's shape, and a policy trained under either convention loads clean against the
#: other and walks sideways. Bump it in the same commit as the maths.
BASE_POSE_CONVENTION = "pose_in_anchor_yaw_frame_pos_ref_minus_actual_ori_rot6d"

#: Which pose the ``base_pose`` block's POSITION is measured from, per ``window_anchor``.
#:
#: Two conventions, identical in width and in every other block:
#:
#: * ``"robot_now"`` -- position is ``p_ref - p_actual(t)``, the robot as it is THIS step. A
#:   feedback term: it is the tracking error, and it is what every run before 2026-08-20 was
#:   trained on.
#: * ``"plan"`` -- position is ``p_ref - p_plan``, measured from the robot's pose at the last
#:   re-plan. A feedforward term: where the reference wants to be relative to where the plan
#:   was made, which is a pure function of the reference.
#:
#: **The point of "plan" is that it removes odometry from the deploy path.** The generator
#: already predicts in the plan frame -- ``dataset.py`` anchors both its conditioning history
#: and its target on the measured root at the snap -- so under ``"plan"`` its output IS the
#: observation and ``gen_tracker._serve_nodes`` gathers it directly. Under ``"robot_now"`` the
#: runtime has to convert, and the only way to do that is to read the robot's measured world
#: position every control step (``reanchor_window``), which on hardware means real odometry.
#:
#: The generator itself is INVARIANT to the choice: its history feedback is the raw prediction
#: (``gen_tracker`` stores ``_pred``, never the re-anchored serve), so one ``fm.onnx`` is
#: correct with a tracker of either convention. What must not be mixed is a TRACKER and the
#: runtime arm serving it, which is what the stamped string guards.
BASE_POSE_CONVENTIONS: dict[str, str] = {
    "robot_now": "pose_in_anchor_yaw_frame_pos_ref_minus_actual_ori_rot6d",
    "plan": "pose_in_plan_yaw_frame_pos_ref_minus_plan_ori_rot6d",
}

#: The default, and it must stay ``"robot_now"``.
#:
#: Runs trained before the flag existed wrote ``base_pose_convention`` into their ``params/``
#: from the bare module constant, and ``provenance.check_run_conventions`` refuses a mismatch.
#: Their saved config carries no ``window_anchor`` field, so it resolves to this default --
#: which therefore has to reproduce the old string exactly, or those runs stop being
#: replayable.
DEFAULT_WINDOW_ANCHOR = "robot_now"


def base_pose_convention(window_anchor: str = DEFAULT_WINDOW_ANCHOR) -> str:
    """The convention string for one ``window_anchor``.

    Args:
        window_anchor: ``"robot_now"`` or ``"plan"``; see :data:`BASE_POSE_CONVENTIONS`.

    Returns:
        The stamped convention string.

    Raises:
        ValueError: On an unknown anchor. Never defaults -- a silent fallback here is a policy
            that loads clean and steers wrong, which is the failure this string exists for.
    """
    try:
        return BASE_POSE_CONVENTIONS[window_anchor]
    except KeyError:
        raise ValueError(
            f"unknown window_anchor {window_anchor!r}; expected one of "
            f"{sorted(BASE_POSE_CONVENTIONS)}"
        ) from None


#: Per-block uniform-noise HALF-WIDTHS over one window node, in RAW units.
#:
#: ONE table, used in two places that model the SAME physical quantity -- the generator's
#: prediction error:
#:
#: * ``mdp.reference_window``'s observation noise, which corrupts what the policy sees so
#:   it does not learn to trust a perfect reference;
#: * the FM's own autoregressive HISTORY noise (``flow_matching.dataset``), which corrupts
#:   the past nodes it conditions on.
#:
#: They were separate tables until the two sides came to speak the same representation, and
#: they had drifted by up to 10x (joint_pos 0.1 vs 0.01). Keeping them separate is not a
#: tuning knob, it is a bug waiting to happen: at deploy the history IS the model's own past
#: output, so the error corrupting it is by definition the error the observation noise
#: stands in for. Scale the FM side as a whole with ``FlowMatchingConfig.noise_scale`` if you
#: want to ablate the magnitude; do not fork the shape.
#:
#: One deliberate conservatism: this is a blanket over ALL window offsets, while the history
#: only ever carries node 0, whose error is the smallest of the horizon. So the history is
#: noised slightly harder than reality -- the safe direction.
#:
#: MEASURED, and the reason the numbers below are not obvious: the generator's error GROWS
#: 3-10x from node 0 to node 50, so one number per block cannot be right at both ends. Run
#: ``scripts/flow_matching/fm_error_by_offset.py`` after an FM retrain to re-check; against the
#: 100-epoch XL generator (teacher-forced, so a LOWER bound on the closed-loop error these
#: stand in for) the per-offset MEAN errors were base_pos ``[0.043, 0.028, 0.032]`` m,
#: base_ori ``0.036``, base_lin_vel ``0.122``, base_ang_vel ``0.237``, joint_pos ``0.076``.
#:
#: Position-like entries are per-axis ``(x, y, z)``. The vertical axis is NOT tighter -- that
#: assumption is what left ``base_pos`` z at ``0.01`` (0.18x of the measured error, by far the
#: worst entry in the table) while the generator's z error is slightly LARGER than its y error.
#: z is also where terrain lives, so under-noising it taught the policy to trust reference
#: height it will not get. Now ``0.03``: 3x better, still ~0.5x of a lower bound, and
#: deliberately short of full calibration because over-noising z costs stair clearance.
#: base_ori and the velocity entries are left alone until there is a YAW-FRAME checkpoint to
#: measure -- the representation change alters what base_ori even means. ``joint_pos`` measures
#: 0.76x and is representation-independent, so it is a live candidate. ``joint_vel`` and
#: ``ee_pos`` are unmeasured AND unobserved (see ``REFERENCE_WINDOW_BLOCKS``), hence inert.
REFERENCE_WINDOW_NOISE: dict[str, float | list[float]] = {
    "base_pos": [0.06, 0.06, 0.03],   # reference base position error, metres
    # Perturbation of each of the 6 rot6d components. DIMENSIONLESS, not radians -- but the
    # number carries over from when it was: the two columns are unit vectors, so displacing
    # a component by eps tilts the frame by ~eps rad for small eps. Kept at the old 0.03 so
    # the level the policy and the generator's history see is unchanged in practice.
    "base_ori": 0.03,
    "base_lin_vel": 0.25,             # m/s
    "base_ang_vel": 0.25,             # rad/s
    "joint_pos": 0.1,                 # rad
    "joint_vel": 0.5,                 # rad/s
    "ee_pos": [0.05, 0.05, 0.01],     # ee position rel. reference base, metres
}


#: Per-block RAW-unit magnitude of the adjuster's residual at a unit action.
#:
#: The RL adjuster emits a dimensionless action, squashed to ``[-1, 1]``, and this table is
#: what turns one node of it into physical units:
#: ``delta_raw[c] = tanh(a[c]) * RESIDUAL_SCALE[c]``. It sits beside
#: :data:`REFERENCE_WINDOW_NOISE` on purpose -- both tables say "how wrong is the generator's
#: prediction, per block", so they answer to one measurement and cannot drift apart the way
#: the observation and history noise tables did.
#:
#: **Two of these entries are safety bounds and the rest are only authority scales.** The
#: failure termination (``mdp.tracking_error.violates``) is measured against the reference the
#: adjuster controls, so the residual's reach on ``base_pose`` is what decides whether the
#: policy can move its own grader. ``base_pos`` and the rotation half are therefore bounded;
#: the velocity and joint channels are limited by the residual L2 reward instead, because
#: neither can defeat a check (a velocity residual cannot flatten a walking speed profile
#: within the reward's 0.5 m/s kernel, and no check reads joint angles at all).
#:
#: The numbers, and where each comes from:
#:
#: * ``base_pos`` -- ``0.20`` m, chosen to EQUAL ``TrackingErrorSpec.max_lateral_m`` and
#:   ``max_height_m``. Deliberately loose to start: the point of v1 is to measure how much
#:   authority the adjuster actually uses before tightening. The consequence is real and is
#:   why the diagnostics in ``g1_adjuster_env_cfg`` are not optional -- at this cap the
#:   residual can move the graded node by the entire violation threshold.
#: * ``base_ori`` -- a PARAMETERISATION, not the bound. The real limit is
#:   :data:`RESIDUAL_MAX_TILT_RAD`, enforced geometrically by
#:   :func:`clamp_window_rotation_residual`, because at 15 deg the small-angle
#:   component-to-radian equivalence this table's ``base_ori`` note relies on has broken
#:   down (measured: a 0.020 component displacement tilts the frame ~1.6 deg, so the
#:   relationship is only locally linear). ``0.20`` makes the bound reachable at a unit
#:   action without overshooting it far enough to make the clamp the whole mechanism.
#: * ``base_lin_vel`` -- ``0.50`` m/s, ~2.2x the generator's measured far-node error
#:   (0.122 mean over the observed offsets, ~1.85x that at node 61).
#: * ``base_ang_vel`` -- ``0.50`` rad/s, only ~1.1x its far-node error, and deliberately the
#:   least generous entry relative to the error it corrects. The measured fraction of
#:   window-error energy in ``{const, const+ramp}`` is 0.04/0.11 for ``ang_vel`` against an
#:   i.i.d. baseline of 0.077/0.154 -- i.e. SUB-white. There is no structured error there for
#:   a residual to remove, and authority it cannot use is authority it can only misuse.
#: * ``joint_pos`` -- ``0.30`` rad, ~2.1x its far-node error (0.076 mean).
#: * ``joint_vel`` / ``ee_pos`` -- ``0.0``. Neither is predicted by any shipped generator
#:   (they are absent from ``reference_blocks``), so they never reach the layout. Zero rather
#:   than a guess means that if a future selection does include them the residual is INERT
#:   until someone calibrates it, instead of silently free on an unmeasured channel.
REFERENCE_WINDOW_RESIDUAL_SCALE: dict[str, float | list[float]] = {
    "base_pos": [0.20, 0.20, 0.20],   # metres -- a BOUND; see above
    "base_ori": 0.20,                 # rot6d components; the bound is RESIDUAL_MAX_TILT_RAD
    "base_lin_vel": 0.50,             # m/s
    "base_ang_vel": 0.50,             # rad/s
    "joint_pos": 0.30,                # rad
    "joint_vel": 0.0,                 # unpredicted -- inert until calibrated
    "ee_pos": 0.0,                    # unpredicted -- inert until calibrated
}

#: Hard cap on the ANGLE the residual may rotate one reference node by, in radians.
#:
#: Half of ``TrackingErrorSpec.max_tilt_rad`` (30 deg), so the tilt violation keeps 2x of
#: margin against the residual where ``base_pos`` currently keeps 1x.
#:
#: Enforced as an angle rather than as a box on the six rot6d components because the two are
#: not the same thing at this magnitude, and the difference is unbounded in the wrong
#: direction: the components are two unit columns, so a per-component box constrains the
#: frame's tilt only to first order. :func:`clamp_window_rotation_residual` measures the
#: realised rotation and shrinks it, which is exact at any angle.
RESIDUAL_MAX_TILT_RAD: float = math.radians(15.0)


#: Blocks a window node packs, in order, when none are selected explicitly.
#:
#: A block can be dropped to ask what the policy actually needs from the reference. The
#: selection threads through the observation term, the noise / scale builders, AND the FM's
#: dataset, losses and export -- everything reads this one list, so a subset cannot land in
#: one place and not another. Dropping a block also skips computing it, which is not free:
#: the observation manager is the largest phase of the env step, and the generator's output
#: space shrinks with it.
REFERENCE_WINDOW_BLOCK_NAMES: tuple[str, ...] = (
    "base_pose", "base_vel", "joint_pos", "joint_vel", "ee_pos",
)

#: Layout block -> the per-block keys in :data:`REFERENCE_WINDOW_NOISE` (and the
#: observation's SCALE table) that expand into it. ``base_pose`` and ``base_vel`` are each
#: two halves in different units, which is why those tables key more finely than the layout
#: does; ``base_pos`` and both ``base_vel`` halves are 3 channels, ``base_ori`` is 6.
REFERENCE_BLOCK_KEYS: dict[str, tuple[str, ...]] = {
    "base_pose": ("base_pos", "base_ori"),
    "base_vel": ("base_lin_vel", "base_ang_vel"),
    "joint_pos": ("joint_pos",),
    "joint_vel": ("joint_vel",),
    "ee_pos": ("ee_pos",),
}


def resolve_reference_blocks(blocks: Sequence[str] | None) -> tuple[str, ...]:
    """Validate a block selection and put it in canonical packing order.

    Args:
        blocks: Requested block names, or ``None`` for
            :data:`REFERENCE_WINDOW_BLOCK_NAMES`.

    Returns:
        The selection, ordered as a window node packs it.

    Raises:
        ValueError: On an unknown name or an empty selection.
    """
    if blocks is None:
        return REFERENCE_WINDOW_BLOCK_NAMES
    wanted = set(blocks)
    unknown = wanted - set(REFERENCE_WINDOW_BLOCK_NAMES)
    if unknown:
        raise ValueError(
            f"unknown reference-window block(s) {sorted(unknown)}; "
            f"expected a subset of {list(REFERENCE_WINDOW_BLOCK_NAMES)}"
        )
    if not wanted:
        raise ValueError("a reference window needs at least one block")
    return tuple(b for b in REFERENCE_WINDOW_BLOCK_NAMES if b in wanted)


def window_blocks(
    num_joints: int, num_ee_frames: int, blocks: Sequence[str] | None = None
) -> list[tuple[str, int]]:
    """Per-node block layout as ``[(name, width), ...]`` in packing order.

    The single source of truth for how one window node is packed. Everything that slices a
    window derives its offsets from here rather than restating them.

    Args:
        num_joints: Robot joint count ``nj``.
        num_ee_frames: End-effector frames the clip library exposes ``K``.
        blocks: Subset to include; defaults to all of them.

    Returns:
        ``[(name, width), ...]`` for the selection.
    """
    widths = {
        "base_pose": 9,   # position(3) + the 6D rotation representation(6)
        "base_vel": 6,
        "joint_pos": num_joints,
        "joint_vel": num_joints,
        "ee_pos": 3 * num_ee_frames,
    }
    return [(b, widths[b]) for b in resolve_reference_blocks(blocks)]


def window_dim(
    num_joints: int, num_ee_frames: int, blocks: Sequence[str] | None = None
) -> int:
    """Width ``D`` of one window node for the given selection."""
    return sum(w for _, w in window_blocks(num_joints, num_ee_frames, blocks))


def window_block_slices(
    num_joints: int, num_ee_frames: int, blocks: Sequence[str] | None = None
) -> dict[str, slice]:
    """Column slice of each selected block within one window node."""
    out: dict[str, slice] = {}
    offset = 0
    for name, width in window_blocks(num_joints, num_ee_frames, blocks):
        out[name] = slice(offset, offset + width)
        offset += width
    return out


#: Node content a MEASURED history row cannot fill on a ROBOT, and therefore takes from the
#: node the runtime last SERVED instead.
#:
#: The G1 has no odometry and no base linear-velocity estimate, so a history row built from
#: the robot's own state can supply its orientation (IMU roll/pitch, plus the heading change
#: since the plan anchor, which is at most ``fm_update_every`` steps of gyro), its angular
#: velocity and its joints -- but not where the base IS or how fast it is travelling. Those
#: two come from the prediction, which is what the closed loop would have used anyway.
#:
#: Named at HALF-BLOCK granularity because that is where the answer changes: ``base_pose``
#: is one block in the layout but its position half is unmeasurable while its rotation half
#: is not, and reporting or serving them together would hide that.
#:
#: TODO(measured-history-odometry): leg odometry gives the base displacement over the few
#: steps since the plan anchor, which is all the ``plan`` convention needs, and a base
#: velocity estimate would close the other one. Revisit both entries then -- and note the
#: gap probe measures these two as the SMALLEST of the five, so the expected gain is small.
MEASURED_HISTORY_FALLBACK: tuple[str, ...] = ("base_pos", "base_lin_vel")

#: Where each :data:`MEASURED_HISTORY_FALLBACK` name sits inside its parent block.
_FALLBACK_HALVES: dict[str, tuple[str, int, int]] = {
    "base_pos": ("base_pose", 0, 3),        # the position half of base_pose
    "base_lin_vel": ("base_vel", 0, 3),     # the linear half of base_vel
}


#: ``history_from`` serving arm -> the ``history_source`` a checkpoint must declare for it.
#: ``"clip"`` maps to ``"reference"``: teacher forcing feeds the clip's node, which IS an
#: applied reference, so it is the same KIND of row the reference-trained model expects.
HISTORY_ARM_SOURCE: dict[str, str] = {
    "generator": "reference",
    "clip": "reference",
    "measured": "measured",
}


def assert_history_source_matches(history_from: str, checkpoint_source: str | None,
                                  where: str = "") -> None:
    """Refuse a serving arm the checkpoint was not fitted for.

    The two history kinds produce rows of IDENTICAL width, so a mismatch loads clean, runs,
    and is merely wrong -- the same failure shape as ``window_anchor`` and
    ``attention_mode``. Nothing downstream can catch it, which is why this is a hard refusal
    at load rather than a warning.

    A ``None`` checkpoint source means the artifact predates the flag, which is
    definitionally ``"reference"`` -- resolved by the caller's config loader, not silently
    permitted here.

    Args:
        history_from: The runtime's serving arm.
        checkpoint_source: What the checkpoint says it trained on.
        where: Optional context for the message (an artifact path).

    Raises:
        ValueError: On an unknown arm, or an arm the checkpoint does not match.
    """
    want = HISTORY_ARM_SOURCE.get(history_from)
    if want is None:
        raise ValueError(
            f"history_from={history_from!r} is not one of {sorted(HISTORY_ARM_SOURCE)}.")
    got = str(checkpoint_source or "reference")
    if got != want:
        raise ValueError(
            f"history_from={history_from!r} needs a generator trained with "
            f"history_source={want!r}, but {where or 'this checkpoint'} declares "
            f"history_source={got!r}.\n"
            "The two produce history rows of the SAME WIDTH, so this would load and run and "
            "simply feed the model a kind of history it was never fitted on. Either serve "
            f"the arm it was trained for, or point at a {want!r}-trained generator.")


def measured_history_fallback_spans(
    num_joints: int, num_ee_frames: int, blocks: Sequence[str] | None = None
) -> tuple[tuple[int, int], ...]:
    """Column spans of :data:`MEASURED_HISTORY_FALLBACK` within one window node.

    Returned as spans rather than as a mask so the torch and numpy runtimes can each build
    their own without this module having to know which array library is asking -- the same
    reason :func:`window_block_slices` returns slices rather than indices.

    A fallback whose parent block is not in the selection contributes nothing: a checkpoint
    that does not carry ``base_vel`` has no linear-velocity half to fall back on.

    Args:
        num_joints: Robot joint count ``nj``.
        num_ee_frames: End-effector frames ``K``.
        blocks: Subset to include; defaults to all of them.

    Returns:
        ``((start, stop), ...)`` in packing order.
    """
    slices = window_block_slices(num_joints, num_ee_frames, blocks)
    out = []
    for name in MEASURED_HISTORY_FALLBACK:
        parent, lo, hi = _FALLBACK_HALVES[name]
        if parent in slices:
            base = slices[parent].start
            out.append((base + lo, base + hi))
    return tuple(out)


def expand_window_blocks(
    num_joints: int,
    num_ee_frames: int,
    per_block: dict[str, float | list[float]],
    blocks: Sequence[str] | None = None,
) -> list[float]:
    """Expand per-block values to one value per channel of ONE window node.

    Shared by everything needing a per-channel vector over the layout -- the observation's
    noise and scale, the FM's history noise -- so a layout change or a block SELECTION
    cannot land in one and not the others.

    Args:
        num_joints: Robot joint count ``nj``.
        num_ee_frames: End-effector frames ``K``.
        per_block: Value per :data:`REFERENCE_BLOCK_KEYS` key; position-like keys accept a
            scalar or an ``(x, y, z)`` triple.
        blocks: Subset to include; defaults to all of them.

    Returns:
        ``window_dim(nj, K, blocks)`` values, in packing order.

    Raises:
        ValueError: If the built width disagrees with the layout.
    """
    def axes(key: str, n: int = 3) -> list[float]:
        """The block's value as ``n`` per-channel entries.

        A sequence is taken verbatim (and must already be the right length); a scalar is
        repeated. ``n`` defaults to 3 because every key but ``base_ori`` is a 3-vector.
        """
        value = per_block[key]
        return [float(v) for v in value] if isinstance(value, (tuple, list)) else [float(value)] * n

    builders = {
        # base_ori is the 6D rotation representation: 6 channels, one value each.
        "base_pose": lambda: axes("base_pos") + axes("base_ori", 6),
        "base_vel": lambda: axes("base_lin_vel") + axes("base_ang_vel"),
        "joint_pos": lambda: [float(per_block["joint_pos"])] * num_joints,
        "joint_vel": lambda: [float(per_block["joint_vel"])] * num_joints,
        "ee_pos": lambda: axes("ee_pos") * num_ee_frames,
    }
    selected = resolve_reference_blocks(blocks)
    out: list[float] = []
    for name in selected:
        out += builders[name]()
    expected = window_dim(num_joints, num_ee_frames, selected)
    if len(out) != expected:
        raise ValueError(
            f"Built {len(out)} channels, expected {expected} (blocks={list(selected)}, "
            f"nj={num_joints}, ee={num_ee_frames}). The per-block expansion has drifted "
            "from window_blocks."
        )
    return out


def split_window(
    window: torch.Tensor, num_joints: int, num_ee_frames: int,
    blocks: Sequence[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Split a ``[..., D]`` window into its named blocks (views, not copies).

    Args:
        window: Window nodes, last axis ``D``.
        num_joints: Robot joint count ``nj``.
        num_ee_frames: End-effector frames ``K``.

    Returns:
        Block name -> ``[..., width]`` view.

    Raises:
        ValueError: If ``window``'s last axis disagrees with the layout.
    """
    expected = window_dim(num_joints, num_ee_frames, blocks)
    if window.shape[-1] != expected:
        raise ValueError(
            f"window is {window.shape[-1]} wide but the layout implies {expected} "
            f"(blocks={list(resolve_reference_blocks(blocks))}, nj={num_joints}, "
            f"ee={num_ee_frames})."
        )
    return {
        name: window[..., slc]
        for name, slc in window_block_slices(num_joints, num_ee_frames, blocks).items()
    }


def project_window_rotations(
    window: torch.Tensor,
    num_joints: int,
    num_ee_frames: int,
    blocks: Sequence[str] | None = None,
) -> torch.Tensor:
    """Re-orthonormalize ``base_pose``'s 6D rotation block of raw generator output.

    A network emitting 6 free numbers produces something only approximately a rotation.
    This projects it back -- Gram-Schmidt, then read the two columns off again -- so the
    served reference is a real pose and the autoregressive history stays on the manifold it
    was trained on.

    Preferred over an orthonormality PENALTY: the penalty trades prediction accuracy
    against a constraint that can be satisfied exactly and for free after the fact, and it
    would still leave the served value slightly off. Applied once, where the prediction is
    produced -- :func:`flow_matching.online.sample_window` in simulation and the exported
    ONNX module at deploy -- so nothing downstream (including the numpy port) has to know.

    A no-op when ``base_pose`` is not in the selection, and safe on already-orthonormal
    input.

    Args:
        window: ``[..., D]`` raw window nodes in physical units.
        num_joints: Robot joint count ``nj``.
        num_ee_frames: End-effector frames ``K``.
        blocks: The selection ``window`` is packed with; defaults to all of them.

    Returns:
        ``[..., D]``, same layout, with the rotation block on SO(3).
    """
    slices = window_block_slices(num_joints, num_ee_frames, blocks)
    if "base_pose" not in slices:
        return window
    start = slices["base_pose"].start + 3
    projected = matrix_to_rot6d(rot6d_to_matrix(window[..., start:start + 6]))
    return torch.cat([window[..., :start], projected, window[..., start + 6:]], dim=-1)


def residual_scale_vector(
    num_joints: int,
    num_ee_frames: int,
    blocks: Sequence[str] | None = None,
    scales: dict[str, float | list[float]] | None = None,
) -> list[float]:
    """Per-channel RAW-unit scale of the adjuster's residual over ONE window node.

    Expanded through :func:`expand_window_blocks`, the same helper the observation's noise and
    scale vectors go through, so the residual's authority cannot come to disagree with the
    layout or with a block SELECTION.

    Args:
        num_joints: Robot joint count ``nj``.
        num_ee_frames: End-effector frames ``K``.
        blocks: The selection the window is packed with; defaults to all of them.
        scales: Per-block magnitudes; defaults to
            :data:`REFERENCE_WINDOW_RESIDUAL_SCALE`.

    Returns:
        ``window_dim(nj, K, blocks)`` values, in packing order.
    """
    table = REFERENCE_WINDOW_RESIDUAL_SCALE if scales is None else scales
    return expand_window_blocks(num_joints, num_ee_frames, table, blocks)


def clamp_window_rotation_residual(
    adjusted: torch.Tensor,
    original: torch.Tensor,
    num_joints: int,
    num_ee_frames: int,
    blocks: Sequence[str] | None = None,
    max_tilt_rad: float = RESIDUAL_MAX_TILT_RAD,
) -> torch.Tensor:
    """Limit how far a residual rotated each node's ``base_pose``, as an ANGLE.

    Bounding the six rot6d components individually does NOT bound the frame's tilt: the block
    is two unit columns, so a per-component box controls the realised rotation only to first
    order, and the residual's whole reason for existing is to be large enough to matter.  So
    this measures the rotation the residual actually produced -- ``Log(R_new R_old^T)`` -- and
    where its magnitude exceeds ``max_tilt_rad``, shrinks the rotation vector to land exactly
    on the bound.  Scaling in the Lie algebra keeps the axis and moves only the angle, which
    is the geodesic interpolation toward the original.

    Only ever SHRINKS: a node already inside the bound is returned bit-identical, so this is
    safe to apply unconditionally and is idempotent.

    Expects both arguments already on SO(3) -- call :func:`project_window_rotations` on the
    adjusted window first, since adding six free numbers to an orthonormal pair leaves the
    manifold and ``so3_log`` needs a real rotation.

    A no-op when ``base_pose`` is not in the selection.

    Args:
        adjusted: ``[..., D]`` window nodes AFTER the residual was added and projected.
        original: ``[..., D]`` the same nodes before it was added.
        num_joints: Robot joint count ``nj``.
        num_ee_frames: End-effector frames ``K``.
        blocks: The selection both windows are packed with; defaults to all of them.
        max_tilt_rad: The bound; defaults to :data:`RESIDUAL_MAX_TILT_RAD`.

    Returns:
        ``[..., D]``, same layout, with every node's rotation within ``max_tilt_rad`` of its
        original.
    """
    slices = window_block_slices(num_joints, num_ee_frames, blocks)
    if "base_pose" not in slices:
        return adjusted
    start = slices["base_pose"].start + 3
    r_new = rot6d_to_matrix(adjusted[..., start:start + 6])
    r_old = rot6d_to_matrix(original[..., start:start + 6])

    delta = so3_log(r_new @ r_old.transpose(-1, -2))               # (..., 3), angle in [0, pi]
    angle = torch.linalg.norm(delta, dim=-1, keepdim=True)
    # clamp_min on the divisor keeps the identity rotation (angle exactly 0) from producing a
    # nan factor; clamp_max(1.0) means an inside node's rotation vector is reproduced exactly.
    factor = (float(max_tilt_rad) / angle.clamp_min(1e-12)).clamp_max(1.0)
    shrunk = matrix_to_rot6d(so3_exp(delta * factor) @ r_old)
    # Select rather than rely on ``factor == 1``: the log/exp round trip is exact in exact
    # arithmetic but costs ~1e-7 in float32, and this function is applied every re-plan to a
    # window where almost every node is already inside the bound. ``where`` makes "unchanged
    # inside the bound" literally true, which is what lets the caller apply it blindly.
    #
    # The tolerance is what makes it IDEMPOTENT rather than merely convergent. A node the
    # clamp bound sits at exactly ``max_tilt_rad``, and re-measuring that rotation in float32
    # can read a few ulps ABOVE it -- so a bare ``<=`` would shrink it again by ~1e-7 on every
    # re-application. 1e-6 relative widens the effective bound by 15 nano-degrees, which is
    # immaterial next to the 15 deg it is bounding.
    inside = angle <= float(max_tilt_rad) * (1.0 + 1e-6)           # (..., 1), broadcasts over 6
    limited = torch.where(inside, adjusted[..., start:start + 6], shrunk)
    return torch.cat([adjusted[..., :start], limited, adjusted[..., start + 6:]], dim=-1)


def _broadcast_anchor(anchor: torch.Tensor, node_shape: torch.Size) -> torch.Tensor:
    """Expand an anchor to one entry per node and flatten to ``(-1, width)``.

    ``isaaclab.utils.math`` reshapes both of its arguments to 2-D and views the result back
    to the *vector's* shape, so it does not broadcast -- the anchor has to be materialised
    per node before it goes in. Accepts an anchor carrying the node axis already, or one
    with that axis of size 1 (or absent) to be repeated across nodes.

    Args:
        anchor: ``[..., width]``, broadcastable against ``node_shape + (width,)``.
        node_shape: Leading shape of the node-indexed inputs, e.g. ``(N, K)``.

    Returns:
        ``(prod(node_shape), width)``.
    """
    width = anchor.shape[-1]
    return torch.broadcast_to(anchor, tuple(node_shape) + (width,)).reshape(-1, width)


def world_to_window(
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    ref_pos_w: torch.Tensor,
    ref_quat_w: torch.Tensor,
    ref_lin_w: torch.Tensor,
    ref_ang_w: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    ee_pos_w: torch.Tensor,
    blocks: Sequence[str] | None = None,
) -> torch.Tensor:
    """Build window nodes from world-frame reference state and a measured anchor.

    The forward map, shared by the live observation, the offline dataset build and (in its
    numpy port) the deploy runtime. Node-indexed arguments share a leading shape
    ``B = (..., K)``; the anchor may carry that shape, carry it with the node axis of size
    ``1``, or omit the node axis entirely -- it is broadcast either way. That is what lets
    one call serve both the FM's target (one anchor over ``F`` future nodes) and its
    autoregressive history (a *per-row* anchor over ``N`` rows).

    Args:
        root_pos_w: ``[..., 3]`` anchor position (the measured robot base), world frame.
        root_quat_w: ``[..., 4]`` anchor orientation (xyzw), world frame.
        ref_pos_w: ``[..., K, 3]`` reference base position per node, world frame.
        ref_quat_w: ``[..., K, 4]`` reference base orientation per node (xyzw), world frame.
        ref_lin_w: ``[..., K, 3]`` reference base linear velocity per node, world frame.
        ref_ang_w: ``[..., K, 3]`` reference base angular velocity per node, world frame.
        joint_pos: ``[..., K, nj]`` reference joint positions, **robot joint order**.
        joint_vel: ``[..., K, nj]`` reference joint velocities, robot joint order.
        ee_pos_w: ``[..., K, E, 3]`` reference end-effector positions, world frame.

        blocks: Layout blocks to emit; defaults to all of them. A dropped block is not
            computed at all.

    Returns:
        ``[..., K, D]`` window nodes, ``D = window_dim(nj, E, blocks)``.
    """
    selected = resolve_reference_blocks(blocks)
    node_shape = ref_pos_w.shape[:-1]                        # (..., K)
    flat = math.prod(node_shape)

    act_p = _broadcast_anchor(root_pos_w, node_shape)         # (flat, 3)
    act_q = _broadcast_anchor(root_quat_w, node_shape)        # (flat, 4)
    # The anchor is YAW-ONLY: the base blocks are observed in the anchor's heading frame, not
    # its full body frame, so the robot's roll and pitch never enter. See the module
    # docstring for why. Computed once -- both base blocks use it.
    act_yaw = yaw_quat(act_q)                                 # (flat, 4)
    ref_p = ref_pos_w.reshape(flat, 3)
    ref_q = ref_quat_w.reshape(flat, 4)

    def base_pose() -> torch.Tensor:
        """The reference base pose in the anchor's YAW frame: ``[pos(3), rot6d(6)]``.

        Both halves measure the same thing the same way -- position ``ref - actual`` rotated
        into the anchor's heading frame, and the reference orientation ``R_yawT R_ref`` as
        its first two columns.
        """
        return torch.cat([
            quat_apply_inverse(act_yaw, ref_p - act_p),
            quat_to_rot6d(quat_mul(quat_conjugate(act_yaw), ref_q)),   # R_yawT R_ref
        ], dim=-1)

    def base_vel() -> torch.Tensor:
        """Reference base linear+angular velocity rotated into the anchor's yaw frame."""
        return torch.cat([
            quat_apply_inverse(act_yaw, ref_lin_w.reshape(flat, 3)),
            quat_apply_inverse(act_yaw, ref_ang_w.reshape(flat, 3)),
        ], dim=-1)

    def ee_pos() -> torch.Tensor:
        """EE positions relative to THAT node's own reference base.

        Posture, free of the tracking error ``base_pose`` already carries.
        """
        num_ee = ee_pos_w.shape[-2]
        ee_flat = ee_pos_w.reshape(flat, num_ee, 3)
        ee_q = ref_q[:, None, :].expand(flat, num_ee, 4).reshape(-1, 4)
        return quat_apply_inverse(
            ee_q, (ee_flat - ref_p[:, None, :]).reshape(-1, 3)
        ).reshape(flat, num_ee * 3)

    # Each block is built ONLY if selected -- these are the largest phase of the env step
    # on the observation side, and a dropped block should cost nothing.
    built = {
        "base_pose": base_pose,
        "base_vel": base_vel,
        "joint_pos": lambda: joint_pos.reshape(flat, -1),
        "joint_vel": lambda: joint_vel.reshape(flat, -1),
        "ee_pos": ee_pos,
    }
    node = torch.cat([built[b]() for b in selected], dim=-1)
    return node.reshape(tuple(node_shape) + (node.shape[-1],))


def reanchor_window(
    window: torch.Tensor,
    old_pos_w: torch.Tensor,
    old_quat_w: torch.Tensor,
    new_pos_w: torch.Tensor,
    new_quat_w: torch.Tensor,
    num_joints: int,
    num_ee_frames: int,
    blocks: Sequence[str] | None = None,
) -> torch.Tensor:
    """Re-express window nodes against a DIFFERENT anchor, block by block.

    A window is relative to one measured robot pose; when the robot has moved on and the
    same nodes must be served against where it is now, only the blocks that reference the
    anchor need touching:

    * ``base_pose`` -- position goes up to the world through the old anchor and back down
      through the new; the orientation half is a pure rotation of its two columns by the
      anchor delta, so it never leaves the 6D representation;
    * ``base_vel`` -- a pure rotation ``R_newT R_old``, no position involved;
    * ``joint_pos`` / ``joint_vel`` -- frame-invariant, copied;
    * ``ee_pos`` -- relative to *that node's own* reference base, so anchor-invariant too,
      copied.

    Doing it per block rather than lifting the whole node to the world and back is what
    makes this work for a block SELECTION at all: the world round trip needs every block
    to be present on the way back, and a window that drops ``joint_vel`` or ``ee_pos``
    cannot supply them.

    Args:
        window: ``[..., K, D]`` window nodes anchored on the old pose.
        old_pos_w / old_quat_w: ``[..., 3]`` / ``[..., 4]`` anchor the nodes are in.
        new_pos_w / new_quat_w: the anchor to express them against.
        num_joints: Robot joint count ``nj``.
        num_ee_frames: End-effector frames ``K``.
        blocks: The selection ``window`` is packed with; defaults to all of them.

    Returns:
        ``[..., K, D]`` nodes in the new anchor's frame, same layout.
    """
    selected = resolve_reference_blocks(blocks)
    parts = split_window(window, num_joints, num_ee_frames, selected)
    node_shape = window.shape[:-1]
    flat = math.prod(node_shape)
    old_p = _broadcast_anchor(old_pos_w, node_shape)
    old_q = yaw_quat(_broadcast_anchor(old_quat_w, node_shape))
    new_p = _broadcast_anchor(new_pos_w, node_shape)
    new_q = yaw_quat(_broadcast_anchor(new_quat_w, node_shape))

    out: list[torch.Tensor] = []
    for name in selected:
        block = parts[name]
        if name == "base_pose":
            # Both anchors are yaw-only (see above), so every rotation here is a heading.
            #
            # Position: up to the world through the old anchor, back down through the new.
            #
            # Orientation: no round trip at all. The block holds the two leading COLUMNS of
            # R_yaw(old)T R_ref, and R_yaw(new)T R_ref = (R_yaw(new)T R_yaw(old)) (R_yaw(old)T
            # R_ref), so each column is simply rotated by the anchor delta -- which is now a
            # PURE YAW rotation. Exactly linear, which is why this neither needs Gram-Schmidt
            # nor quietly projects a raw generator output onto SO(3) behind the caller's back.
            ref_p = old_p + quat_apply(old_q, block[..., 0:3].reshape(flat, 3))
            q_delta = quat_mul(quat_conjugate(new_q), old_q)          # yaw-only delta
            out.append(torch.cat([
                quat_apply_inverse(new_q, ref_p - new_p),
                quat_apply(q_delta, block[..., 3:6].reshape(flat, 3)),
                quat_apply(q_delta, block[..., 6:9].reshape(flat, 3)),
            ], dim=-1))
        elif name == "base_vel":
            out.append(torch.cat([
                quat_apply_inverse(new_q, quat_apply(old_q, block[..., 0:3].reshape(flat, 3))),
                quat_apply_inverse(new_q, quat_apply(old_q, block[..., 3:6].reshape(flat, 3))),
            ], dim=-1))
        else:
            out.append(block.reshape(flat, -1))
    node = torch.cat(out, dim=-1)
    return node.reshape(tuple(node_shape) + (node.shape[-1],))


def window_to_world(
    window: torch.Tensor,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    num_joints: int,
    num_ee_frames: int,
    blocks: Sequence[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Exact inverse of :func:`world_to_window`: put the window back in the world.

    A window is entirely relative by design -- that is what keeps world coordinates away
    from the policy -- which also makes it impossible to eyeball. Given the anchor the
    forward direction divided out, this reconstructs the world-frame reference it encodes,
    so a viewer can draw *what the policy actually saw*, and so the FM tracker can feed a
    world-frame CLF and debug markers in simulation. Nothing on the deploy path calls it.

    Only the SELECTED blocks come back. A caller that dropped a block gets it absent from
    the result rather than zero-filled -- the ``y_des`` writer and the debug ribbon both
    branch on presence, and a plausible zero would be silently drawn and graded.

    Feed a CLEAN window, not a corrupted observation: noise, ``clip`` and ``scale`` are
    applied by the observation manager AFTER this layout and are not inverted here.

    Args:
        window: ``[..., K, D]`` window nodes.
        root_pos_w: ``[..., 3]`` anchor position, broadcast over nodes.
        root_quat_w: ``[..., 4]`` anchor orientation (xyzw).
        num_joints: Robot joint count ``nj``.
        num_ee_frames: End-effector frames ``K``.
        blocks: The selection ``window`` was packed with; defaults to all of them.

    Returns:
        Mapping with whichever of ``base_pos_w`` / ``base_quat_w`` / ``base_lin_vel_w`` /
        ``base_ang_vel_w`` / ``joint_pos`` / ``joint_vel`` / ``ee_pos_w`` the selection
        supports.

    Raises:
        ValueError: If ``ee_pos`` is selected without ``base_pose`` (it lifts through that
            node's own reference base, which the base block carries).

    Note:
        ``base_quat_w`` comes back through ``quat_from_matrix``, which fixes the sign, so it
        may be the negation of the quaternion that went in. That is the same rotation;
        compare rotations, not raw components.
    """
    selected = resolve_reference_blocks(blocks)
    parts = split_window(window, num_joints, num_ee_frames, selected)
    node_shape = window.shape[:-1]
    flat = math.prod(node_shape)
    act_p = _broadcast_anchor(root_pos_w, node_shape)
    # Yaw-only, matching the forward map: the base blocks were observed in the anchor's
    # heading frame, so that is the frame they have to be lifted out of.
    act_q = yaw_quat(_broadcast_anchor(root_quat_w, node_shape))
    shape3 = tuple(node_shape) + (3,)
    out: dict[str, torch.Tensor] = {}

    pos_w = quat_w = None
    if "base_pose" in parts:
        pos_err_b = parts["base_pose"][..., 0:3].reshape(flat, 3)
        ori6d_b = parts["base_pose"][..., 3:9].reshape(flat, 6)
        pos_w = act_p + quat_apply(act_q, pos_err_b)
        # Forward: the block is R_yawT R_ref, so R_ref = R_yaw (R_yawT R_ref). This is the
        # one place a rotation MATRIX has to come back out of the 6D pair, so it is also the
        # one place Gram-Schmidt runs (inside rot6d_to_quat) -- a raw generator output is
        # only approximately orthonormal, and a quaternion has nowhere to put the residual.
        #
        # Note the recovered quaternion carries the reference's FULL orientation, roll and
        # pitch included: the yaw frame drops the ANCHOR's tilt, never the reference's.
        quat_w = quat_mul(act_q, rot6d_to_quat(ori6d_b))
        out["base_pos_w"] = pos_w.reshape(shape3)
        out["base_quat_w"] = quat_w.reshape(tuple(node_shape) + (4,))
    if "base_vel" in parts:
        out["base_lin_vel_w"] = quat_apply(
            act_q, parts["base_vel"][..., 0:3].reshape(flat, 3)).reshape(shape3)
        out["base_ang_vel_w"] = quat_apply(
            act_q, parts["base_vel"][..., 3:6].reshape(flat, 3)).reshape(shape3)
    if "joint_pos" in parts:
        out["joint_pos"] = parts["joint_pos"].clone()
    if "joint_vel" in parts:
        out["joint_vel"] = parts["joint_vel"].clone()
    if "ee_pos" in parts:
        if pos_w is None:
            raise ValueError(
                "ee_pos lifts through THAT node's own reference base, so window_to_world "
                "cannot place it unless base_pose is in the same selection."
            )
        ee_rel = parts["ee_pos"].reshape(flat, num_ee_frames, 3)
        ee_q = quat_w[:, None, :].expand(flat, num_ee_frames, 4).reshape(-1, 4)
        out["ee_pos_w"] = (pos_w[:, None, :] + quat_apply(
            ee_q, ee_rel.reshape(-1, 3)).reshape(flat, num_ee_frames, 3)
        ).reshape(tuple(node_shape) + (num_ee_frames, 3))
    return out
