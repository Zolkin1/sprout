# Copyright 2026 Zachary Olkin. All rights reserved.

"""Configuration for the robot-state-conditioned flow-matching tracking command."""

from isaaclab.utils import configclass

from .motion_sequence_cmd_cfg import MotionSequenceCommandCfg


@configclass
class FmTrackingCommandCfg(MotionSequenceCommandCfg):
    """Config for :class:`FmTrackingCommand` (closed-loop, robot-state-conditioned FM).

    The FM conditions on the **live robot state** history (N most-recent measured steps,
    in the FM's xyz+yaw frame), an exteroception token, and the future velocity command;
    it predicts the F-step ego-anchored clip future, re-anchored on the robot each
    re-plan. The CLF tracking error is computed in the world frame (inherited).

    The clip library supplies **only** the reset spawn pose (and the terrain meshed from
    it) and the ``target_vel`` command -- there is no clip fallback for the reference
    itself, not even during warmup. See :attr:`strict_fm_only`.

    On the free-form terrain sections it supplies neither: :attr:`freeform_command` generates
    the velocity command instead, and the reset spawns off the manifest, so that library needs
    no clips at all.
    """

    fm_depth_lag_steps: int | None = None
    """Synthetic camera lag for the GENERATOR, in control steps of 20 ms.

    The generator trains on depth frames 20-60 ms old (compute plus the ROS hop), so feeding
    it the live frame here is off-distribution in exactly the variable that model was fitted
    to. ``None`` derives the lag from the loaded checkpoint's own
    ``depth_latency_probs`` (rounded mean: the shipped distribution gives 2 steps = 40 ms);
    an int forces it; ``0`` disables it.

    **Only the generator is delayed.** The teacher's own depth observation stays live -- it
    was trained on fresh frames, and delaying that is a separate question.

    COST: a non-zero lag makes the generator's exteroception read EVERY control step instead
    of once per re-plan, because a 2-step-old frame has to have been captured to be served.
    That is the unprojection running 12x more often on the ``fm_snap_every=12`` cadence. At
    the default of 0 nothing extra runs at all, so a checkpoint without a latency
    distribution costs exactly what it did before.
    """

    class_type: type | str = (
        "robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking"
        ".fm_tracking_cmd:FmTrackingCommand"
    )

    fm_checkpoint: str = ""
    """Path to the trained robot-state-conditioned flow-matching ``.pt`` checkpoint
    (``target_source=ego_anchored_clip_future``). Required, except with
    :attr:`reference_source` ``= "clip"``, where an empty value means "no generator at all"."""

    reference_source: str = "fm"
    """Which reference the policy actually TRACKS: ``"fm"`` or ``"clip"``.

    Both are computed every step regardless -- the command caches the generator's
    ``y_des``/``dy_des`` and the clip's side by side, and
    :meth:`~.fm_tracking_cmd.FmTrackingCommand.as_clip_reference` /
    :meth:`~.fm_tracking_cmd.FmTrackingCommand.as_fm_reference` serve the other one to a
    caller that asks. This field only decides which of the two is applied.

    ``"clip"`` exists for ONE purpose: the FM-vs-clip diagnostic
    (``scripts/rsl_rl/fm_debug.py``). Driving from the clip and running the generator
    PASSIVELY means the two references are evaluated at *exactly the same robot states*, so
    a difference between them is the generator's, not a consequence of the robot having
    already diverged. The FM-driven run answers "how does it behave"; this one answers
    "what would it have said here".

    Pair ``"clip"`` with the TEACHER's anchoring -- otherwise the clip reference is not the
    one the policy was trained against and the baseline is not a baseline. That means
    ``ego_anchor_fraction=1.0`` (the task defaults it to 0.0, the deploy value) and leaving
    ``ego_anchor_every`` at its cfg default, which follows ``REPLAN_PERIOD_K``. This used to
    say "pin it to 1", correct while the teacher also trained at 1; pinning a cadence here is
    how that went stale, so match the teacher rather than a number. ``fm_debug.py`` and
    ``fm_survival_sweep.py``'s ``clip_control`` both do this.

    With ``"clip"`` and an empty :attr:`fm_checkpoint` the generator is not loaded at all --
    the pure clip control, useful to rule out any FM-induced cost or nondeterminism."""

    fm_update_every: int = 1
    """Re-run flow-matching inference every ``K`` control steps (the receding-horizon
    re-plan cadence); the applied index advances through the F-step window in between.
    ``1`` = re-plan every step, so the cursor never advances and the reference is always
    node 0 of a fresh prediction. Must be ``>= 1``.

    This need NOT match the recording's ``fm_snap_every``; the agreement check that used to
    enforce it was removed. Note what that means in each direction:

    * FASTER than the recording (``K < fm_snap_every``) is horizon-SAFE -- the required
      horizon is ``max_offset + K``, so a smaller K needs less of it -- and the query
      states are a superset of the snapped ones rather than different in kind, since the
      teacher re-anchors every step regardless.
    * SLOWER (``K > fm_snap_every``) is the direction that bites: the cursor runs further
      into the window between re-plans, and ``_check_horizon`` is now the only thing
      standing between that and a silently frozen reference tail.

    Cost scales directly with this: the generator is ~99% of per-step inference, so K=1
    makes every control tick pay the full FM cost instead of one tick in K."""

    fm_num_sampling_steps: int = 0
    """Override the checkpoint's Euler sampling-step count at inference (``0`` = trained).

    ``0`` rather than ``None`` because IsaacLab's configclass infers a field's type from its
    DEFAULT VALUE, so a ``None`` default makes hydra reject every int:
    ``Incorrect type under namespace /commands/traj_ref/fm_num_sampling_steps. Expected:
    <class 'NoneType'>, Received: <class 'int'>``. The knob was therefore documented but
    impossible to set from the command line, which is worse than not having it. Same trap as
    the noise tensors that cannot live in a cfg at all.

    Raising it reduces the flow's DISCRETISATION error without retraining, so it separates
    "the sampler is under-resolved" from "the model is wrong". Costs latency in proportion --
    the generator is ~99% of per-step inference."""

    fm_device: str | None = None
    """Device to load the flow-matching model on (``None`` uses the env device)."""

    strict_fm_only: bool = True
    """Poison every ``y_des``/``dy_des`` channel the FM did not write with NaN, each step.

    The barrier between the generator path and the offline clips. The FM writes the base
    frame, its end-effector frames and the joints; everything else in the output vector
    (the COM block, the ``@root`` body-link blocks, any frame the checkpoint does not
    predict) is clip-derived and must not reach a reward, an observation or a
    termination here. Filling those with NaN turns an accidental read into an immediate,
    obvious failure instead of a plausible-looking number from the wrong source.

    Turn it off only to debug what a clip-sourced channel *would* have contained."""

    seed_history_from_clip: str = "auto"
    """When to fill a resetting env's generator history from the CLIP's preceding steps.

    The generator's first inference of an episode otherwise runs on the learned
    ``null_history`` embedding, which is phase-agnostic by construction and -- because the
    recordings it was fitted on start on flat -- effectively a FLAT-GROUND prior. Both are
    fine when the episode opens on flat, which is how the robot starts on hardware. Neither is
    fine when it opens on terrain.

    * ``"auto"`` (default) -- seed exactly when ``history_from="clip"``. Teacher forcing is
      meant to hold the prediction in phase with the clip, and it cannot do that on the first
      inference unless the history is there.
    * ``"on_terrain_start"`` -- seed only the envs whose start lands at or after the clip's
      ``terrain_start_step``, i.e. the ones with no flat lead-in to open on. Everything else
      keeps the null embedding, which is what deploy gets. This is the setting that lets the
      ``*_pure`` stair families be trained on at all without giving every other episode an
      opening it will never see on a robot.
    * ``"always"`` / ``"never"`` -- unconditional.

    Seeding needs ``N`` real steps before the start; :attr:`FmTrackingCommand.min_start_step`
    reserves them, and the clip-step gather clamps only at the TOP, so without that reserve a
    start inside it would read the clip's END as its history."""

    history_from: str = "generator"
    """Where the FM's ``N`` autoregressive context tokens come from. A DIAGNOSTIC switch.

    * ``"generator"`` (default, and the only deployable one) -- the nodes this command
      served at the previous ``N`` steps, i.e. the model's own output. This is the real
      closed loop.
    * ``"clip"`` -- the nodes the CLIP reference would have produced at those same steps,
      anchored the same way. Teacher forcing: the generator still predicts the future and
      the tracker still follows it, but the context it conditions on is always correct.

    This is the experiment that separates the two failure modes that look identical from
    the outside. The model is trained on clip history and deployed on its own, so if the
    rollout is bad you cannot tell whether the MODEL is weak or whether the LOOP diverges
    (exposure bias). Run both:

    * ``"clip"`` good, ``"generator"`` bad -> the loop. Attack the exposure bias: more
      history-token noise (``FlowMatchingConfig.noise_scale``), scheduled sampling, or a
      shorter re-plan horizon.
    * both bad -> the model. More data/epochs, or the conditioning is missing something.
    * ``"clip"`` also bad in a way ``val_flow_mse`` did not predict -> the offline metric
      is not measuring what the tracker needs.

    Not available at deploy: there is no clip on the robot. ``transfer/deploy`` implements
    the generator path only."""

    fm_guidance_scale: float = 1.0
    """Classifier-free guidance weight ``w`` over the FM's HISTORY conditioning.

    ``1.0`` = off (plain conditional sampling). ``w > 1`` integrates
    ``v_null + w (v_cond - v_null)``, extrapolating along the direction the history
    contributes, so the sample adheres to it more strongly. Needs no retrain: the
    checkpoint was trained with the history dropped on ``null_cond_prob`` of samples, so
    ``v_null`` is a learned branch.

    Why it is interesting for THIS model: the history is the previous OUTPUT, so
    strengthening history adherence is strengthening the coupling between consecutive
    predictions -- i.e. temporal coherence, which is the same axis as "the trajectory jumps
    between re-plans". Conditioning adherence and temporal coherence are one knob here in a
    way they are not for image models.

    Expect it to help ``history_from="clip"`` more than ``"generator"``, and possibly to
    HURT the latter past some ``w``: guidance makes the model trust the history, and in
    closed loop that history is its own drifting output. Guidance also samples off the data
    distribution, which matters more here than for images because the output has to be
    dynamically feasible for the tracker to follow. Sweep against TRACKING, not flow MSE.

    Costs one extra forward per integration step, batched as a single 2x-batch call.
    NOT yet in the ONNX export -- ``export_fm_onnx`` folds the sampler into the graph, so a
    guided deploy needs the guided loop baked in there too."""

    fm_zero_noise: bool = False
    """Integrate the flow from ``x0 = 0`` instead of Gaussian noise, so the prediction is a
    deterministic function of the conditioning.

    Worth pairing with ``history_from`` when the complaint is that the prediction JUMPS
    between consecutive inferences: at ``fm_update_every=1`` the generator re-samples every
    control step, so with Gaussian noise some of that jump is just the fresh draw rather
    than the conditioning changing.

    Default ``False`` = training-matched stochastic sampling; leave it there for training and
    for any run meant to represent real behaviour. Set it only for A-B diagnostics where the
    per-call noise draw would mask the effect being measured -- notably sim2sim, whose MuJoCo
    counterpart (``GenTrackerPolicy(zero_noise=...)``, exposed as ``--zero_noise`` /
    ``--gaussian_noise``) defaults it ON, so a default Isaac run and a default MuJoCo run are
    NOT comparing like with like. ``x0 = 0`` is off the training distribution."""

    # --- Free-form velocity command (the terrain sections of the FM fine-tune) ---
    #
    # Every field below is inert while ``freeform_command`` is False, which is the default, so
    # the five clip tasks keep the library's ``target_vel`` untouched.

    freeform_command: bool = False
    """Generate the velocity command HERE instead of reading the clip's ``target_vel``.

    The clip-driven tasks take the command from the library because the clip WAS generated
    from it -- reference and command are one artefact. The free-form terrain sections have no
    clips: the reference is whatever the generator predicts from depth, so a command has to be
    manufactured, and it is manufactured the way the deployed joystick supplies one -- a
    forward speed plus a yaw rate that keeps the robot pointed down the section.

    Turning this on replaces all four command-velocity accessors
    (:meth:`~.fm_tracking_cmd.FmTrackingCommand.get_command_velocity` and the three
    look-ahead forms), which is every path the command reaches: the FM's own conditioning
    tokens, the policy's ``command_window`` observation, the recorder's ``target_vel`` column
    and the eval accumulator's tracking target. It also makes the command self-sufficient of a
    library ``target_vel`` -- ``command_velocity_available`` reports True regardless -- which
    is what lets a manifest-only terrain ``.h5`` be passed as ``--clip``.

    **The command is HELD across every offset**, where the clip path serves a per-offset
    future. See :attr:`freeform_speed_min` for why that is a train/deploy mismatch being
    removed rather than a simplification.
    """

    freeform_speed_min: float = 0.6
    """Low end of the per-episode forward-speed draw (m/s), inclusive.

    One constant per EPISODE, drawn uniformly in ``[min, max]`` at reset and held for the
    whole rollout -- which is how every terrain clip in the training library was generated
    (one commanded velocity per clip, no ramp, from frame 0). ``vy`` is always exactly 0.0:
    the hardware controller ships ``v_y_scale: 0.0`` and the library measures
    ``|vy| max = 0.0000`` across all 3500 stair clips, so a lateral channel would train
    something the deployed pair cannot ask for.

    The default band is the trained stair grid's endpoints (``transfer/sim/mjeval/terrains``
    ``STAIR_SPEEDS``: a uniform 0.600 .. 2.600 over 3500 clips). Restated rather than imported
    because ``robot_rl`` must not import the MuJoCo half.

    **The whole (vx, 0, wz) triple is broadcast across every window offset and across the
    generator's F-step horizon.** The clip path instead serves ``target_vel`` sampled at each
    offset, which is a future no deployed runtime has: both ``transfer/deploy/gen_tracker.py``
    and the C++ runtime hold ONE command across the offsets and over F ("the future command is
    not knowable; hold the one from request time"). So holding it here removes a train/deploy
    mismatch for free -- but it does change what the tracker's reference-group encoder sees,
    because ``command_window`` goes from 13 distinct rows to 13 copies of one.
    """

    freeform_speed_max: float = 2.6
    """High end of the per-episode forward-speed draw (m/s), inclusive.

    See :attr:`freeform_speed_min`. Must be ``>=`` it; equal pins every episode to one speed.
    """

    freeform_cmd_hold_s: float = 0.5
    """Seconds of ZERO command at the start of every episode, before the step to the draw.

    The robot spawns from a standing pose at rest and would otherwise be told the full speed
    on the very first control step. A HOLD then a STEP, not a ramp: a ramp is its own
    off-distribution shape, while standing-to-walk transitions at a command that starts at 0.0
    are something the flat family actually contains. The number is
    ``transfer/sim/mjeval/terrains.CMD_HOLD_S``, where it was measured: on the first
    400-episode MuJoCo grid 59% of the baseline's falls (94 of 159) never reached the terrain
    at all, and in the fast half of the ladder the median fall was 0.40 m from spawn.

    Rounded to whole control steps against ``env.step_dt``; ``0`` restores the instant-full-
    speed opening.
    """

    freeform_steer_kp: float = 1.0
    """Proportional gain (1/s) from heading error to commanded yaw rate.

    ``0`` disables steering and leaves ``wz = 0`` -- the constant ``(vx, 0, 0)`` the training
    clips carry -- while everything else about the free-form command stays on. See
    :func:`~.fm_tracking_cmd.freeform_yaw_rate` for the law, the cross-track gain it implies
    with :attr:`freeform_lookahead_m`, and the distribution change it is.
    """

    freeform_lookahead_m: float = 5.0
    """Distance along the section centreline to the point the robot is aimed at (m).

    Must be ``> 0``. A pure P on BEARING has effective cross-track gain ``kp / L``, so this
    and :attr:`freeform_steer_kp` are not independent: at the default pair that is 0.2 1/s,
    and 5 m of look-ahead across a 1.219 m flight is a weak restoring gain. ``L`` is therefore
    the first thing to tune if lateral drift persists -- shortening it stiffens the loop
    without touching ``kp``.
    """

    freeform_wz_max: float = 0.68
    """Symmetric clip on the steering yaw rate (rad/s).

    0.68 is the largest ``|wz|`` the training library ever commands, on the box family
    (``transfer/sim/mjeval/runner.STEER_WZ_MAX``, restated because ``robot_rl`` cannot import
    the MuJoCo half). It is a bound on how far outside the commanded distribution the
    generator is pushed, not a physical limit.
    """

    # --- Debug visualization ---
    debug_vis_show_fm_prediction: bool = True
    """Draw the FM's predicted world-frame trajectory as a ribbon of spheres (one color per
    frame). Overlay it on the yellow observation-window ribbon (``debug_vis_obs_window``) to
    check that what the policy observes is what the generator predicted."""

    debug_vis_fm_marker_radius: float = 0.02
    """Sphere radius (m) for the FM-prediction ribbon points."""
