# Copyright 2026 Zachary Olkin. All rights reserved.

"""Run one ARM of the MuJoCo terrain eval: every authored course x every commanded speed.

An arm is one tracker.  The generator is held fixed across arms (``evaluate.py`` states which),
so a paired difference between two ``episodes.h5`` files is a statement about the tracker and
nothing else.  Closed loop throughout -- the generator conditions on its own served history and
on depth cast from wherever the robot actually went.  There is no clip, no teacher forcing and
no domain randomisation anywhere in this file.

**Three nested scopes, and that nesting is the point of the module.**  The costs are measured,
and they are three orders of magnitude apart, so what belongs where is not a matter of taste:

1. **Per PROCESS (~10 s):** load the ONNX pair and derive the generator's camera lag.  Done
   once in :func:`load_arm`; a per-episode reload would cost half an hour over a 200-episode
   sweep and would prove nothing, the graphs being immutable.
2. **Per COURSE (~50 ms):** :class:`CourseRig` -- the MuJoCo model, its depth casters, the
   actuator ids behind the policy's joint order, the effort limits and the control decimation.
   Every one of these is derived FROM the model, which is why they cannot outlive it.
3. **Per EPISODE (~0 ms):** :func:`run_episode` resets state in place.  It calls
   ``mujoco.mj_resetData``, never ``Robot.reset_robot``: that method re-parses the XML into a
   NEW ``mj_model``/``mj_data`` pair, which silently invalidates every scope-2 object -- the
   depth cameras hold body ids into the old model, ``act_ids`` and ``effort_limits`` index it,
   and the policy's PD gains were written into it.  Nothing would raise; the cameras would cast
   against a stale model and the servos would revert to the MJCF's gains, which on this robot
   are 2.46x too soft at both hip pitches.

**HEADLESS ONLY, and there is deliberately no way to ask for a viewer.**  ``viewer.sync()``
perturbs ``mjData`` enough to change the trajectory: one tile measured here survived 16 s
headless and fell reproducibly at 6.04 s with the viewer attached.  A harness whose verdicts
depend on whether anyone was watching is not measuring the policy, so the option is absent
rather than defaulted off.  Watch a single cell in ``gen_tracker_runner.py`` instead.

**The generator's depth stays LAGGED** (:class:`gen_tracker_runner.FmDepthLag`, derived from
the graph's own ``depth_latency_probs``): the checkpoint was fitted with tens of milliseconds
between capture and inference, and a fresh frame is off-distribution in exactly that variable.
Only the generator is delayed; the tracker's own ``depth`` observation stays live, because it
was trained on fresh frames.
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR = os.path.dirname(_HERE)
# This directory (siblings by bare name, the way `accumulate` and `schema` reach `terrains`),
# `transfer/sim` (the MuJoCo halves), and `transfer` (the torch-free deploy runtime).
for _d in (_HERE, _SIM_DIR, os.path.dirname(_SIM_DIR)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from deploy.fm_session import default_depth_lag_steps  # noqa: E402
from deploy.gen_tracker import GenTrackerPolicy  # noqa: E402
from deploy.tracker_session import convert_joint_order  # noqa: E402

from depth_camera import MujocoDepthCameras  # noqa: E402
from gen_tracker_runner import FmDepthLag, build_state, load_standing_pose  # noqa: E402
from robot import Robot  # noqa: E402

import accumulate  # noqa: E402
import schema  # noqa: E402
import terrains  # noqa: E402

#: ``<repo>/transfer/sim/mjeval`` -> ``<repo>``, verified against a marker rather than trusted.
#: The same idiom ``report.py`` uses: a plausible-but-wrong root surfaces much later, as a
#: missing artifact at a path nobody typed.
REPO_ROOT: str = os.path.dirname(os.path.dirname(_SIM_DIR))
if not os.path.isfile(os.path.join(REPO_ROOT, "scripts", "rsl_rl", "eval", "terrain_class.py")):
    raise RuntimeError(f"{__file__} derived REPO_ROOT={REPO_ROOT}, which has no "
                       "scripts/rsl_rl/eval/terrain_class.py; fix the derivation if this "
                       "module moved rather than adding a '..' hop.")

#: The solved standing pose every episode spawns from -- the same default
#: ``gen_tracker_runner.main`` uses, so a cell of this grid and a hand-run of that CLI start
#: the robot in the same configuration.
DEFAULT_STANDING_CONFIG: str = os.path.join(
    REPO_ROOT, "hf", "trajectories", "retargeted", "2026-06-16_16-30-26_merged",
    "standing", "standing_config_solution.yaml")

#: Stored in the ``seed`` column.  Under ``zero_noise`` the flow integrates from ``x0 = 0``
#: rather than from a draw, so there is no sampler seed to record and -1 says so rather than
#: implying that 0 was one.
ZERO_NOISE_SEED: int = -1

#: Default clip on the steering controller's yaw-rate output (rad/s).  0.68 is the largest
#: ``|wz|`` the training library ever commands (on the box family); the stair families are
#: commanded exactly 0.0, so anything non-zero there is extrapolation whatever the clip.
STEER_WZ_MAX: float = 0.68

#: The zero command held at the start of every episode; a module constant so the hold does not
#: allocate an array per step.
_ZERO_CMD: np.ndarray = np.zeros(3)

#: Multiplier turning ``(terrain_idx, speed_idx)`` into a unique ``episode_id``.  Bigger than
#: any course's ladder, so the id is derived rather than counted and a shard produces the same
#: number for a cell whichever shard ran it -- which is what lets merged shard tables be
#: concatenated without renumbering.
EPISODE_ID_STRIDE: int = 100


@dataclass(frozen=True)
class ArmRun:
    """One arm's configuration: which tracker, which generator, and where the table goes.

    Attributes:
        arm: The ``arm`` column's value; 0 = baseline, 1 = fine-tuned.
        label: Short name, used for the progress lines and the output directory.
        tracker_onnx: Absolute path to this arm's ``tracker.onnx``.
        fm_onnx: Absolute path to the generator.  The SAME file for every arm -- that is what
            makes the difference between two tables attributable to the tracker.
        out_dir: Directory to write ``episodes.h5`` into.  ``report.py`` treats this directory
            as a run, so its ``manifest.json`` belongs here too.
        robot_name: MuJoCo robot prefix; the scenes are emitted as ``<robot>_<course>.xml``.
        providers: onnxruntime execution providers, as ``resolve_providers`` takes them.
        fall_height: Metres of pelvis clearance above the scene's LOWEST terrain surface below
            which the episode is a fall.  On a descending course that datum is the bottom of
            the flight, so up top only the tilt cone is guarding -- see
            :attr:`terrains.Course.fall_datum_z`.
        fall_tilt_deg: Roll/pitch bound from upright, degrees.
        zero_noise: Integrate the flow from ``x0 = 0``.  True is what the hardware controller
            runs and what makes a repeat at fixed ``(course, speed)`` bit-identical, which is
            why the sweep has one trial per cell.  Set False to put the generator back on its
            TRAINING distribution -- an ordinary Gaussian draw -- which is the only way to ask
            what the flow's own sampling variance does to an outcome.  Then ``fm_seed``
            selects which draw, and repeats at one cell stop being redundant.
        fm_seed: Seed for the generator's noise draws.  Only meaningful when ``zero_noise`` is
            False; recorded in the table's ``seed`` column either way.
        spawn_z_offset: Metres the spawn pelvis is lifted above the standing config's value.
            The optimizer's pelvis frame sits ~3 cm below MuJoCo's foot-contact height, so
            without it every episode opens with the feet through the floor.
        standing_config: The ``*_config_solution.yaml`` to spawn from.
        fm_depth_lag_steps: Override the generator's camera lag, in control steps.  ``None``
            derives it from the graph's own ``depth_latency_probs``, which is what a
            checkpoint's own training distribution says and therefore the default.
        shards: How many processes the catalogue is split across.  Each takes the courses
            whose global index is congruent to :attr:`shard`, so `terrain_idx` stays global and
            the shard tables concatenate.  Sharding cannot change a result: 13 runs of one
            episode at 1-, 4- and 8-way concurrency were bit-identical.
        steer_kp: P gain on heading error toward the goal (1/s); ``0`` disables steering and
            holds the constant command the training clips used.  See :func:`steer_yaw_rate` --
            non-zero is OFF-DISTRIBUTION on the stair courses.
        steer_wz_max: Clip on the steering yaw rate (rad/s).
        cmd_hold_s: Seconds of zero command before the step to the commanded speed; see
            :data:`terrains.CMD_HOLD_S` for the measurement that motivates it.
        shard: Which residue class this process runs, in ``0 .. shards - 1``.
        terrain_ref: A ``terrain_class.py`` reference set.  Empty means no plan classification,
            and the four ``n_plans*`` columns are then legitimately zero rather than missing.
        max_courses: Truncate the catalogue, for a smoke run.  ``0`` runs all of it.
        max_speeds: Truncate each course's ladder, for a smoke run.  ``0`` runs all of it.
        attrs: Provenance from the caller, stamped onto ``episodes.h5`` verbatim.  What the
            RUNNER resolved (dt, cadence, the lag it actually applied) is added on top, so the
            file records the resolved configuration and not merely the requested one.
    """

    arm: int
    label: str
    tracker_onnx: str
    fm_onnx: str
    out_dir: str
    robot_name: str = "g1_perceptive"
    providers: tuple[str, ...] = ("auto",)
    fall_height: float = 0.2
    fall_tilt_deg: float = 70.0
    spawn_z_offset: float = 0.03
    standing_config: str = DEFAULT_STANDING_CONFIG
    fm_depth_lag_steps: int | None = None
    terrain_ref: str = ""
    steer_kp: float = 0.0
    steer_wz_max: float = STEER_WZ_MAX
    cmd_hold_s: float = terrains.CMD_HOLD_S
    zero_noise: bool = True
    fm_seed: int = 0
    shards: int = 1
    shard: int = 0
    max_courses: int = 0
    max_speeds: int = 0
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Spawn:
    """The standing state every episode restarts from, in the orders each consumer needs.

    Attributes:
        joint_pos_isaac: Standing joint angles in the policy's joint order.  Kept because the
            MuJoCo order is a property of the model, which is scope 2, while the solved pose
            is not.
        base_pos: Pelvis world position, already lifted by ``spawn_z_offset``.
        base_quat_wxyz: Pelvis orientation in MuJoCo's ``(w, x, y, z)`` order.
        base_lin_vel_w / base_ang_vel_w / joint_vel_isaac: The velocity half, or ``None`` for
            the standing spawn, which is at rest.  A CLIP spawn is not: clip 300 starts at
            0.298 m/s with joint rates to 1.09 rad/s, i.e. mid-gait.  Isaac writes all four
            buffers at reset (``resets._commit_clip_reset``), so dropping these would place
            the robot at the right pose and dead still -- a different initial condition that
            looks perfectly correct in a viewer.
    """

    joint_pos_isaac: np.ndarray
    base_pos: np.ndarray
    base_quat_wxyz: np.ndarray
    base_lin_vel_w: np.ndarray | None = None
    base_ang_vel_w: np.ndarray | None = None
    joint_vel_isaac: np.ndarray | None = None


def clip_spawn(course, policy_joint_names, *, spawn_z_offset: float = 0.005) -> Spawn:
    """The initial condition of an ``isaac_clip_course``: the clip's own reference at step 0.

    Isaac starts a tracker episode ON the clip -- ``resets._commit_clip_reset`` writes root
    pose, root velocity, joint position and joint velocity from
    ``reference_world_state_at(clip_id, start_step, ...)``.  mjeval's default spawn is a
    STANDING pose at rest, which on stairs is the difference between arriving in phase with
    the treads and arriving from a standstill, and is the leading suspect for why an
    Isaac-matched budget looked infeasible.

    **The joint order is converted by NAME, and that is not optional.**  The clip's ``y_des``
    columns are in the ROBOT's per-limb order (``left_hip_pitch, left_hip_roll, left_hip_yaw,
    ...``) while ``Spawn.joint_pos_isaac`` is in the POLICY's interleaved order
    (``left_hip_pitch, right_hip_pitch, waist_yaw, ...``).  They agree in 2 of 29 positions.
    A positional copy is finite, plausible, and puts every joint at another joint's angle.

    **No yaw rotation is applied**, and that is a measured fact rather than an omission:
    Isaac's ``lift_base_world_state`` composes ``quat_apply(anchor_inv_yaw(world_quat), ...)``
    onto the clip's tile-local pose, but ``goal_frame_yaw`` is exactly 0.0 on all 15129
    episodes of the v2 rollout, so that rotation is the identity and tile-local differs from
    world by a pure translation -- which ``isaac_clip_course`` has already applied.  A library
    that ever authors turned tiles breaks this; ``goal_anchor.assert_yaw_free`` is the guard.

    Args:
        course: A course from ``terrains.isaac_clip_course``.
        policy_joint_names: ``policy.get_joint_names()`` -- the order ``Spawn`` is read in.
        spawn_z_offset: Lift applied to the pelvis, matching Isaac's own reset (0.005 m).
            Isaac added it because every episode otherwise opened 1 cm underground and took a
            PhysX ejection kick; it is carried here so the two start from the same height.

    Returns:
        The spawn, velocities included.

    Raises:
        ValueError: If the course was not built by ``isaac_clip_course`` (no clip params), or
            if the clip and policy joint NAME SETS differ, which would make the conversion
            silently drop or invent a joint.
    """
    p = course.params
    if "clip_start_joint_pos" not in p:
        raise ValueError(
            f"course {course.name!r} carries no clip start state; clip_spawn needs a course "
            "built by terrains.isaac_clip_course, not isaac_tile_course.")
    clip_names = list(p["clip_joint_names"])
    if set(clip_names) != set(policy_joint_names):
        missing = sorted(set(policy_joint_names) - set(clip_names))
        extra = sorted(set(clip_names) - set(policy_joint_names))
        raise ValueError(
            f"clip and policy joint NAME SETS differ - missing from clip: {missing}; "
            f"not in policy: {extra}. The by-name conversion would drop or invent a joint.")
    pos = np.asarray(p["clip_start_pos"], dtype=np.float64).copy()
    pos[2] += float(spawn_z_offset)
    return Spawn(
        joint_pos_isaac=convert_joint_order(
            np.asarray(p["clip_start_joint_pos"], dtype=np.float64),
            clip_names, list(policy_joint_names)),
        base_pos=pos,
        base_quat_wxyz=np.asarray(p["clip_start_quat_wxyz"], dtype=np.float64),
        base_lin_vel_w=np.asarray(p["clip_start_lin_vel"], dtype=np.float64),
        base_ang_vel_w=np.asarray(p["clip_start_ang_vel"], dtype=np.float64),
        joint_vel_isaac=convert_joint_order(
            np.asarray(p["clip_start_joint_vel"], dtype=np.float64),
            clip_names, list(policy_joint_names)))


class CourseRig:
    """One course's MuJoCo model and everything derived from it.

    Built once per course and reused by all ten of its episodes.  Every attribute here indexes
    or holds a reference into ``mj_model``, which is why :func:`run_episode` resets the DATA in
    place and never rebuilds the model.
    """

    def __init__(self, course: terrains.Course, policy: GenTrackerPolicy, cfg: ArmRun,
                 spawn: Spawn) -> None:
        """Load the course's scene and resolve the ids the episode loop needs.

        Args:
            course: The authored course; supplies the scene name and the fall datum's context.
            policy: The loaded pair, for the joint order, the trained PD gains and the camera
                contracts.
            cfg: The arm being run.
            spawn: The process-level standing pose, converted here into THIS model's joint
                order.  The order is a property of the model and the pose is not, which is why
                the two are resolved in different scopes.
        """
        self.course = course
        # Emitted immediately before it is loaded, rather than depending on a preceding
        # `--stage scenes`: the write is deterministic and costs microseconds, and an ordering
        # dependency between two commands is the trap this repo keeps paying for.
        terrains.write_scene(course, cfg.robot_name)
        self.robot = Robot(
            robot_name=cfg.robot_name, scene_name=course.name,
            joystick_scaling=np.ones(3), input_function=lambda _t: np.zeros(3),
            fall_height=cfg.fall_height,
            fall_tilt_rad=float(np.deg2rad(cfg.fall_tilt_deg)),
        )
        model = self.robot.mj_model
        # apply_policy_gains, NOT set_pd_gains_from_policy: the latter prints one line per
        # joint (29 per course, 580 per arm) and never populates `_policy_gains`, so it is the
        # weaker of the two for exactly the situation this loop is in.
        self.robot.apply_policy_gains(policy.get_kp(), policy.get_kd(), policy.get_joint_names())

        isaac = policy.get_joint_names()
        self.hold_target = convert_joint_order(spawn.joint_pos_isaac, isaac,
                                               self.robot.joint_names)
        # Same conversion, same reason: the two orders differ in 27 of 29 positions, so a
        # positionally-copied velocity vector is finite, plausible and pairs each joint's
        # rate with a different joint.
        self.hold_joint_vel = (None if spawn.joint_vel_isaac is None else
                               convert_joint_order(spawn.joint_vel_isaac, isaac,
                                                   self.robot.joint_names))
        self.act_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{n}_pos") for n in isaac],
            dtype=np.int64)
        if int(self.act_ids.min()) < 0:
            missing = [n for n, a in zip(isaac, self.act_ids) if a < 0]
            raise ValueError(f"{course.name}: no position actuator for {missing}")
        # From the MJCF's own `jnt_actfrcrange`, because `policy_parameters.yaml` carries no
        # effort limits -- the same resolution `gen_tracker_runner.run` does, so the torque
        # utilisation column means the same thing in both.  Resolved through an explicit id
        # check: a missing joint returns -1, which indexes the LAST row of the table and would
        # silently give one joint another's limit.
        jnt_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in isaac]
        if min(jnt_ids) < 0:
            raise ValueError(f"{course.name}: model has no joint(s) "
                             f"{[n for n, j in zip(isaac, jnt_ids) if j < 0]}")
        self.effort_limits = np.array(
            [float(np.max(np.abs(model.jnt_actfrcrange[j]))) for j in jnt_ids],
            dtype=np.float64)
        self.decim = int(round(policy.get_dt() / model.opt.timestep))
        self.total_mass = float(np.sum(model.body_mass))

        self.depth_cams: MujocoDepthCameras | None = None
        spec = policy.get_depth_camera_spec()
        if spec:
            self.depth_cams = MujocoDepthCameras(model, spec)
        self.fm_depth_cams: MujocoDepthCameras | None = None
        if policy.depth_sampling_is_split():
            if self.depth_cams is None:
                raise ValueError(f"{course.name}: the generator samples the cameras on its own "
                                 "grid, but the tracker declares none to sample")
            self.fm_depth_cams = MujocoDepthCameras(model, policy.get_fm_depth_camera_spec())
            # One physical camera pair sampled on two grids: the mount is shared even though
            # the per-frame draws cannot be (differing pixel counts).  No noise is enabled
            # here at all, so this only keeps the two casts geometrically identical.
            self.fm_depth_cams.mirror_extrinsics_from(self.depth_cams)


def load_arm(cfg: ArmRun) -> tuple[GenTrackerPolicy, FmDepthLag, Spawn]:
    """Scope 1: load the ONNX pair once, and everything else that outlives a course.

    Args:
        cfg: The arm to load.

    Returns:
        ``(policy, fm_lag, spawn)``.

    Raises:
        FileNotFoundError: If either artifact is absent.  Named here rather than inside
            onnxruntime, which reports a missing file as a protobuf parse failure.
    """
    for role, path in (("tracker", cfg.tracker_onnx), ("generator", cfg.fm_onnx)):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"arm {cfg.label!r}: no {role} at {path}")
    policy = GenTrackerPolicy(
        cfg.tracker_onnx, cfg.fm_onnx,
        zero_noise=bool(cfg.zero_noise),
        seed=int(cfg.fm_seed),
        providers=list(cfg.providers) or None,
        # One generator against two trackers exported at different times, which is the A/B's
        # whole premise.  The field that can disagree is `mjcf_sha256`, and it reaches nothing
        # in this loop: the generator never reads the MJCF (its FK loss used it offline, at
        # dataset build) and `apply_policy_gains` overwrites MuJoCo's actuator gains with the
        # policy's -- the same measured argument `stair_probe._MJCF_ONLY` makes.  This is SIM
        # ONLY; nothing on the robot path sets it.
        allow_robot_asset_mismatch=True,
    )
    policy.load()
    print(f"[INFO] onnxruntime providers: {policy.active_providers()}")

    probs = getattr(policy.ctx, "depth_latency_probs", None)
    if cfg.fm_depth_lag_steps is not None:
        fm_lag = FmDepthLag(cfg.fm_depth_lag_steps, "--fm_depth_lag_steps")
    else:
        fm_lag = FmDepthLag(default_depth_lag_steps(probs),
                            f"derived from fm.onnx depth_latency_probs={probs}")

    isaac = policy.get_joint_names()
    jp, base_pos, base_quat = load_standing_pose(
        cfg.standing_config, isaac,
        fallback=dict(zip(isaac, np.asarray(policy._trk.default_joint_angles).ravel())))
    base_pos = base_pos.copy()
    base_pos[2] += cfg.spawn_z_offset
    return policy, fm_lag, Spawn(jp, base_pos, base_quat)


class PlanClassifier:
    """Labels each re-plan ``flat`` / ``jump`` / ``stairs``, in the generator's own output space.

    A thin wrapper over ``scripts/rsl_rl/eval/terrain_class.TerrainClassifier`` -- numpy-only,
    and the one thing this package borrows from the IsaacLab harness -- so the class NAMES are
    resolved once at construction rather than by an import inside the control loop.

    **The label is a POPULATION statistic and not a per-plan verdict.**  The classifier was
    validated against real clip windows (control accuracy 0.859) and is then applied to
    generator output, where within matched step bands a flagged plan is no more divergent from
    the clip than an unflagged one.  So ``n_plans_jump / n_plans`` is readable against a
    control -- its own stairs->jump confusion is ~0.10 on these families -- and "this plan was
    a misread" is not a claim it supports.
    """

    def __init__(self, path: str) -> None:
        """Load a reference set built by ``terrain_class.py --windows ... --out ...``.

        Args:
            path: The reference ``.npz``.
        """
        eval_dir = os.path.join(REPO_ROOT, "scripts", "rsl_rl", "eval")
        if eval_dir not in sys.path:
            sys.path.append(eval_dir)
        import terrain_class  # noqa: PLC0415

        self._clf = terrain_class.TerrainClassifier.load(path)
        self._names = terrain_class.COARSE_CLASSES

    def label(self, window: np.ndarray) -> str:
        """Classify one plan.

        Args:
            window: The live plan ``[F, D]`` in its own anchor frame, as
                ``GenTrackerPolicy.predicted_window`` returns it.  The features are
                differenced from node 0, so the anchor cancels and a runtime plan is
                comparable to a recorded clip window taken anywhere.

        Returns:
            One of ``flat`` / ``jump`` / ``stairs``.
        """
        return self._names[int(self._clf.classify(window[None])[0])]


def load_classifier(path: str) -> PlanClassifier | None:
    """The plan classifier, or ``None`` when no reference set was given.

    Without one the four ``n_plans*`` columns are zero, which is a true statement -- no plan
    was classified -- rather than a missing measurement.

    Args:
        path: A ``terrain_class.py --out`` reference ``.npz``, or ``""``.

    Returns:
        A :class:`PlanClassifier`, or ``None``.
    """
    if not path:
        return None
    clf = PlanClassifier(path)
    print(f"[INFO] plan classifier from {path}. Read its rate AGAINST a control, never as an "
          "absolute.")
    return clf


def _reset_episode_state(policy: GenTrackerPolicy, fm_lag: FmDepthLag) -> None:
    """Clear every buffer that would otherwise carry the previous episode into this one.

    Two leaks, both silent, both the same shape -- a ring buffer that ``policy.reset()`` does
    not own:

    * ``policy._trk.action_isaac`` is the tracker's last raw output, and ``reset()`` clears the
      observation history but NOT this.  The next ``_push_history`` fills the WHOLE ``actions``
      history window with its current value, so without this line every episode after the first
      opens with N steps of the previous episode's final action presented as its own past.
      IsaacLab zeroes this buffer on an env reset, which is what the tracker was trained
      against.  Rebound rather than zeroed in place, matching ``TrackerSession.__init__``:
      ``get_action`` rebinds the attribute to an onnxruntime output every step, and an
      in-place write would depend on that buffer being writable.
    * ``FmDepthLag``'s ring holds the last ``steps + 1`` depth frames with their poses.  Left
      alone, the first frames the generator sees on a new course are the last frames of the
      previous one -- a different piece of terrain, at full confidence.

    Args:
        policy: The loaded pair.
        fm_lag: The generator's camera-lag ring.
    """
    policy.reset()
    policy._trk.action_isaac = np.zeros_like(policy._trk.action_isaac)
    fm_lag._buf.clear()


def steer_yaw_rate(course: terrains.Course, x: float, y: float, yaw: float,
                   kp: float, wz_max: float) -> float:
    """Proportional yaw rate that points the robot at the goal.

    Nothing in this stack steers: the command is a body-frame velocity and the generator plans
    from depth, so a few degrees of accumulated heading error over six metres of course becomes
    a lateral miss.  This closes that one loop -- bearing to the goal, P on the heading error --
    so a course is scored on whether the robot could traverse it rather than on whether it
    happened to stay on the centreline.

    **It is off-distribution on stairs, and that is not a detail.**  Measured over the training
    library, the two stair families (3500 clips) carry ``|wz| max = 0.0000`` and
    ``|vy| max = 0.0000`` -- the generator has never once been commanded a turn on a staircase.
    ``flat`` and ``box`` do turn (up to 0.80 and 0.68 rad/s), so steering is in distribution
    there.  A stair number measured with this on is a different measurement, not a cleaner one,
    which is why it is opt-in and why ``evaluate.py`` prints the fact when it is enabled.

    Args:
        course: The course, for its goal position.
        x: Root world x (m).
        y: Root world y (m).
        yaw: Root heading (rad).
        kp: Proportional gain (1/s).
        wz_max: Symmetric clip on the output (rad/s).

    Returns:
        The commanded yaw rate (rad/s); 0.0 once the root is inside the goal band, so the
        controller stops fighting for the centreline after the course is already cleared.
    """
    gx, gy = course.target_xy
    if course.in_goal(x, y):
        return 0.0
    bearing = math.atan2(gy - y, gx - x)
    # Wrapped to (-pi, pi] so a heading either side of due-north-of-goal turns the short way.
    err = math.atan2(math.sin(bearing - yaw), math.cos(bearing - yaw))
    return float(np.clip(kp * err, -wz_max, wz_max))


def run_episode(rig: CourseRig, policy: GenTrackerPolicy, fm_lag: FmDepthLag, spawn: Spawn,
                cmd: np.ndarray, classifier: PlanClassifier | None,
                recorder=None, steer_kp: float = 0.0,
                steer_wz_max: float = STEER_WZ_MAX,
                cfg_hold: float = terrains.CMD_HOLD_S,
                probe=None) -> tuple[str, str, Any]:
    """One deterministic trial: spawn, drive at ``cmd``, and stop on a fall, the target or time.

    **Three orderings inside the loop are load-bearing.**  The fall test runs BEFORE the target
    test, which is what makes the three verdicts mutually exclusive -- a robot that face-plants
    across the line is a fall, and ``schema.derive_success`` relies on it.  The accumulator
    folds BEFORE the physics advance, so every folded frame is one the policy actually acted on
    and the terminal frame is left for ``fold_terminal``; omitting that terminal fold in the
    Isaac harness made 289 of 469 fallen episodes read as upright.  And the camera lag is
    applied LAST, after both depth casts, which is what ``build_state`` does internally and why
    it is passed in rather than wrapped around the call.

    Args:
        rig: The course's model and ids.
        policy: The loaded pair.
        fm_lag: The generator's camera-lag ring.
        spawn: The standing state to restart from.
        cmd: The commanded ``(vx, vy, wz)``, held constant for the whole episode -- which is
            how every terrain clip in the training library was generated.
        classifier: A :class:`PlanClassifier`, or ``None`` to leave every plan unlabelled.
        recorder: Optional ``(robot, policy) -> None``, called once per control step at the
            fold -- so a captured frame is one the policy acted on, not the state after an
            advance it never saw.  ``None`` for the sweep, which therefore pays nothing for it.
            ``view_cell.py`` passes one to capture a replayable trajectory; because the episode
            is deterministic, the captured re-run reproduces its table row exactly, which is
            what makes the replay THE episode rather than a similar one.
        probe: Optional ``(robot, policy, state, is_replan) -> None``, called once per control
            step at the same point as ``recorder`` but ALSO handed the state dict the policy
            was just given, and told whether this step carried a re-plan.  That is what the
            failure critic needs and ``recorder`` cannot supply: its inputs are the
            generator's conditioning, which lives in ``state``, and its proprioception ring
            runs at the CONTROL rate while it is only read at a re-plan.  Additive rather than
            a widened ``recorder`` so ``view_cell.py`` and ``fm_sample_explorer.py`` are
            untouched.  ``None`` by default and therefore free.

    Returns:
        ``(verdict, failure_code, accumulator)``, the first two being
        :data:`terrains.VERDICT_CODES` / :data:`terrains.FAILURE_CODES` keys.
    """
    robot, course = rig.robot, rig.course
    # Zeroes qpos, qvel, contacts, ctrl AND mj_data.time, leaving mj_model -- and therefore the
    # cameras, the actuator ids and the applied PD gains -- untouched.
    mujoco.mj_resetData(robot.mj_model, robot.mj_data)
    robot.set_base_and_joints(spawn.base_pos, spawn.base_quat_wxyz, rig.hold_target,
                              base_lin_vel_w=spawn.base_lin_vel_w,
                              base_ang_vel_w=spawn.base_ang_vel_w,
                              joint_vel_mujoco=rig.hold_joint_vel)
    _reset_episode_state(policy, fm_lag)

    acc = accumulate.EpisodeAccumulator(course, cmd, policy.get_dt(), rig.act_ids,
                                        rig.effort_limits, rig.total_mass)
    # The command is rebuilt per step only when steering; otherwise it is the constant the
    # training clips hold, and `live_cmd is cmd` keeps that path allocation-free.
    live_cmd = cmd
    hold_until = float(cfg_hold)
    verdict, failure = "timeout", ""
    last_plan_id = -1
    try:
        while robot.mj_data.time <= course.budget_s:
            # The opening hold: stand at a zero command, then STEP to the real one. Not scored,
            # because a commanded zero met by a standing robot is a near-perfect tracking
            # number that says nothing -- see accumulate.fold's `scored`.
            holding = robot.mj_data.time < hold_until
            if holding:
                live_cmd = _ZERO_CMD
            elif steer_kp > 0.0:
                qw, qx, qy, qz = (float(v) for v in robot.mj_data.qpos[3:7])
                yaw = math.atan2(2.0 * (qw * qz + qx * qy),
                                 1.0 - 2.0 * (qy * qy + qz * qz))
                live_cmd = np.array([cmd[0], cmd[1],
                                     steer_yaw_rate(course, float(robot.mj_data.qpos[0]),
                                                    float(robot.mj_data.qpos[1]), yaw,
                                                    steer_kp, steer_wz_max)])
            else:
                live_cmd = cmd
            state = build_state(robot, policy, live_cmd, rig.depth_cams, want_depth_hits=False,
                                fm_depth_cams=rig.fm_depth_cams, fm_lag=fm_lag)
            obs = policy.create_obs(state)
            robot.apply_action(policy.get_action(obs, robot.joint_names))
            if robot.failed():
                verdict, failure = "fall", robot.failure_reason()
                break
            plan_class = None
            plan = policy.last_plan()
            # The edge is detected unconditionally now. It used to be computed INSIDE the
            # `classifier is not None` branch, so `last_plan_id` never advanced without a
            # classifier and any second consumer of the edge would have seen every step as a
            # re-plan -- silently, since the flag is plausible either way.
            is_replan = plan is not None and int(plan["plan_id"]) != last_plan_id
            if is_replan:
                last_plan_id = int(plan["plan_id"])
                if classifier is not None:
                    window = policy.predicted_window()
                    if window is not None:
                        plan_class = classifier.label(window)
            acc.fold(robot, policy, plan_class, cmd=live_cmd, scored=not holding)
            # Fired at the same point as the fold, so a captured frame is one the policy
            # actually acted on rather than the state after an advance it never saw.  `None`
            # by default and therefore free: the sweep pays nothing for the viewer's hook.
            if recorder is not None:
                recorder(robot, policy)
            if probe is not None:
                probe(robot, policy, state, is_replan)
            for _ in range(rig.decim):
                robot.step()
            # The traversal gate is the second half of "got through": these lanes are finite in
            # y on an infinite floor, so the circle alone can be entered by walking beside the
            # course.
            if (course.in_goal(float(robot.mj_data.qpos[0]), float(robot.mj_data.qpos[1]))
                    and acc.traversed):
                verdict = "success"
                break
    except KeyboardInterrupt:
        verdict, failure = "interrupted", ""
    acc.fold_terminal(robot)
    return verdict, failure, acc


def run_arm(cfg: ArmRun) -> str:
    """Run every cell of one arm and write its ``episodes.h5``.

    Args:
        cfg: The arm to run.

    Returns:
        The path written.  On an interrupt that is ``episodes_partial.h5`` instead, because the
        resume in ``evaluate.py`` keys on ``episodes.h5`` existing -- a truncated table under
        the finished name would be skipped forever and silently reported as a complete arm.
    """
    policy, fm_lag, spawn = load_arm(cfg)
    classifier = load_classifier(cfg.terrain_ref)
    courses = terrains.catalogue()
    if cfg.max_courses:
        courses = courses[:cfg.max_courses]

    # The catalogue is enumerated in FULL and the shard skips what is not its own, so
    # `terrain_idx` stays the global catalogue index in every shard.  Slicing the list instead
    # would renumber the courses per shard, and the merged table would then carry several
    # different courses under one `terrain_idx` -- which the report clusters on, so the
    # bootstrap would resample a cluster that does not exist.
    mine = [i for i in range(len(courses)) if i % cfg.shards == cfg.shard]

    table = schema.EpisodeTable()
    interrupted = False
    wall0 = time.time()
    for terrain_idx in mine:
        course = courses[terrain_idx]
        rig = CourseRig(course, policy, cfg, spawn)
        speeds = course.speeds[:cfg.max_speeds] if cfg.max_speeds else course.speeds
        for speed_idx, speed in enumerate(speeds):
            cmd = np.array([float(speed), 0.0, 0.0])
            t0 = time.time()
            verdict, failure, acc = run_episode(rig, policy, fm_lag, spawn, cmd,
                                                classifier, steer_kp=cfg.steer_kp,
                                                steer_wz_max=cfg.steer_wz_max,
                                                cfg_hold=cfg.cmd_hold_s)
            # Derived rather than counted, so an episode's id is the same number whichever
            # shard ran it and a merged table has no duplicates to renumber.
            row = acc.row(episode_id=terrain_idx * EPISODE_ID_STRIDE + speed_idx,
                          terrain_idx=terrain_idx, speed_idx=speed_idx,
                          arm=cfg.arm, seed=ZERO_NOISE_SEED, verdict=verdict,
                          failure_code=failure)
            table.append(row)
            tag = f"{cfg.label}" if cfg.shards == 1 else f"{cfg.label} s{cfg.shard}"
            print(f"[{tag}] {terrain_idx:02d}/{len(courses)} {course.name:<26} "
                  f"v={speed:5.3f}  {verdict:<11}{failure:<7} "
                  f"t={row['steps_survived'] * policy.get_dt():6.2f}s "
                  f"x={row['progress_x_m']:5.2f}m  ({time.time() - t0:.0f}s wall)", flush=True)
            if verdict == "interrupted":
                interrupted = True
                break
        del rig  # the next course's model replaces it; hold one MuJoCo scene at a time
        if interrupted:
            break

    attrs = dict(cfg.attrs)
    attrs.update({
        "arm": cfg.arm, "arm_label": cfg.label,
        "tracker_onnx": cfg.tracker_onnx, "fm_onnx": cfg.fm_onnx,
        "dt": policy.get_dt(), "fm_update_every": policy.fm_update_every,
        "fm_depth_lag_steps": fm_lag.steps,
        "zero_noise": True, "domain_randomization": "none",
        "steer_kp": cfg.steer_kp, "steer_wz_max": cfg.steer_wz_max,
        "cmd_hold_s": cfg.cmd_hold_s,
        "goal_half_width_m": [c.goal_half_width_m for c in terrains.catalogue()],
        "goal_depth_m": terrains.GOAL_DEPTH,
        "depth_noise": False, "providers": list(cfg.providers),
        "n_courses": len(courses), "n_episodes": len(table),
        "shards": cfg.shards, "shard": cfg.shard,
        "terrain_ref": cfg.terrain_ref, "interrupted": interrupted,
        "wall_seconds": round(time.time() - wall0, 1),
    })
    if cfg.shards > 1:
        name = (f"episodes_shard{cfg.shard}of{cfg.shards}"
                f"{'_partial' if interrupted else ''}.h5")
    else:
        name = "episodes_partial.h5" if interrupted else "episodes.h5"
    path = os.path.join(cfg.out_dir, name)
    table.write(path, attrs)
    stale = os.path.join(cfg.out_dir, "episodes_partial.h5")
    if not interrupted and os.path.isfile(stale):
        # A completed arm leaves no partial behind: two tables in one directory is a question
        # about which one the report read, and the answer would be invisible in the figures.
        os.remove(stale)
    print(f"[{cfg.label}] wrote {len(table)} episodes to {path}"
          + (" -- INTERRUPTED, this arm is incomplete" if interrupted else ""))
    return path


def merge_shards(out_dir: str) -> str:
    """Concatenate an arm's shard tables into the single ``episodes.h5`` the report reads.

    Sharding is a scheduling decision and must leave no trace in the artifact, so this writes
    exactly the file a one-process run would have written.  Rows are sorted by ``episode_id``
    (which is derived from ``(terrain_idx, speed_idx)``, not counted) so the merged order is
    the catalogue's order regardless of how the work was split.

    Args:
        out_dir: The arm directory holding ``episodes_shard{i}of{n}.h5`` files.

    Returns:
        The path to the merged ``episodes.h5``.

    Raises:
        FileNotFoundError: When no shard tables are present.
        ValueError: When the shards disagree on how many there should be, when one is missing,
            or when two carry the same ``episode_id`` -- each of which would otherwise produce a
            quietly incomplete table that still reduces and still plots.
    """
    import glob  # noqa: PLC0415 - only the merge path enumerates shard files

    paths = sorted(glob.glob(os.path.join(out_dir, "episodes_shard*of*.h5")))
    if not paths:
        raise FileNotFoundError(f"no shard tables in {out_dir}")
    tables = [schema.read_table(p) for p in paths]
    totals = {int(a["shards"]) for _, a in tables}
    if len(totals) != 1:
        raise ValueError(f"{out_dir}: shards disagree on the split: {sorted(totals)}")
    total = totals.pop()
    have = sorted(int(a["shard"]) for _, a in tables)
    if have != list(range(total)):
        raise ValueError(f"{out_dir}: expected shards 0..{total - 1}, found {have}. "
                         "Re-run the missing one rather than reducing a partial grid.")

    merged = {name: np.concatenate([c[name] for c, _ in tables])
              for name, _ in schema.MJ_EPISODE_COLUMNS}
    ids = merged["episode_id"]
    if len(set(ids.tolist())) != len(ids):
        raise ValueError(f"{out_dir}: duplicate episode_id across shards -- the split "
                         "overlapped, so some cells ran twice and others not at all.")
    order = np.argsort(ids, kind="stable")

    table = schema.EpisodeTable()
    for i in order:
        table.append({name: merged[name][i].item() for name, _ in schema.MJ_EPISODE_COLUMNS})

    attrs = dict(tables[0][1])
    for key in ("schema_version", "schema_minor", "columns", "units", "class_names",
                "verdict_names", "failure_names", "written_at", "shard"):
        attrs.pop(key, None)
    attrs["shards"] = total
    attrs["n_episodes"] = len(table)
    attrs["wall_seconds"] = round(max(float(a.get("wall_seconds", 0.0)) for _, a in tables), 1)
    attrs["merged_from"] = [os.path.basename(p) for p in paths]
    path = os.path.join(out_dir, "episodes.h5")
    table.write(path, attrs)
    print(f"[merge] {len(paths)} shards -> {path} ({len(table)} episodes)")
    return path
