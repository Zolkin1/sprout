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
    * ``"measured"`` -- the ROBOT'S OWN state at those steps, as a window node in the same
      frame. Where the other two arms both say "the reference applied at that step", this
      one says "where the robot actually got to", which is a different kind of quantity and
      not one the model was fitted on -- see :data:`MEASURED_HISTORY_FALLBACK` for the two
      blocks a robot cannot supply, and ``scripts/flow_matching/measured_history_gap.py``
      for how far off-distribution the rest of the row lands.

    ``"measured"`` sits between the other two in how grounded the context is: teacher
    forcing is correct-by-construction and unavailable on a robot, the closed loop is the
    model's own drifting output, and this is real but off-distribution. Unlike ``"clip"`` it
    IS deployable in principle -- under ``window_anchor="plan"`` it needs the base
    displacement over at most ``fm_update_every`` control steps, not global odometry.

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

    ``"clip"`` is not available at deploy -- there is no clip on the robot.
    ``"generator"`` and ``"measured"`` both are, and ``transfer/deploy`` implements both."""

    allow_history_source_mismatch: bool = False
    """Serve an arm the loaded checkpoint was NOT fitted for.

    Normally refused: the two history kinds produce rows of identical width, so a mismatch
    loads clean and merely feeds the model a quantity it has never seen
    (:func:`~robot_rl.flow_matching.window.assert_history_source_matches`).

    The one legitimate use is FINE-TUNING onto the other arm -- recording rollouts under
    ``history_from="measured"`` from a ``history_source="reference"`` generator is
    deliberately off-distribution, because those rollouts are the training set that moves it
    onto the new conditioning. Anything MEASURED under a mismatch is a statement about a
    model being fed the wrong kind of history, which is occasionally the question and never
    the default.

    SIM ONLY by construction: nothing on the deploy path sets it, and the numpy runtime has
    no equivalent -- ``GenTrackerPolicy`` refuses unconditionally, so a mismatched pair
    cannot reach a robot."""

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

    **That is the DEFAULT's provenance, not a statement about the library in use, and it has
    already misled once.** ``STAIR_SPEEDS`` describes the 3500-clip library v6g replaced.
    Measured on ``g1_mb_v6g_decor.h5`` (10000 clips): commanded vx spans -1.000 .. 2.500, the
    three terrain families floor at 0.200, flat reaches -1.000 because it includes backwards
    walking, and 55.5% of clips dip below 0.6 somewhere. A band below 0.6 is therefore well
    inside what the tracker and the generator were trained on -- read ``target_vel`` off the
    library rather than inferring a floor from this number.

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

    freeform_approach_kp: float = 0.0
    """Proportional gain (1/s) from RANGE-TO-GOAL to commanded forward speed. ``0`` is OFF.

    **Off by default, and off is the exact behaviour every existing config was measured
    under**: at ``0`` the commanded ``vx`` is the per-episode draw held for the whole episode,
    unchanged, and none of the goal state below is even read. Only a course that opts in pays
    for it. ``test_freeform_approach_defaults_off.py`` pins that across every registered task.

    Switched on, the forward channel joins the yaw channel on one controller:
    ``vx = clamp(kp * (range - stop_radius) - kd * range_rate, 0, drawn_speed)``.

    **The per-episode draw becomes a CEILING rather than the command**, which is what keeps
    :attr:`freeform_speed_min` / :attr:`freeform_speed_max` meaning what they meant: far from
    the goal ``kp * range`` saturates the clamp, so the robot cruises at its drawn speed
    exactly as before, and the gain only bites on the approach. At the default 1.5 m stop
    radius a gain of 0.6 starts decelerating a 2.5 m/s episode about 5.7 m out.

    The floor at 0 matters: a P controller past its setpoint would command BACKWARDS, and
    reversing is both off-distribution for the generator and not what "stop" means.
    """

    freeform_approach_kd: float = 0.0
    """Damping gain (dimensionless) on the CLOSING RATE, the D half of the approach controller.

    Subtracted from the P term, so it brakes when the robot is closing fast. The rate is the
    root's planar velocity projected onto the goal direction -- a real derivative off the
    state, not a finite difference of the range, so it carries no sampling noise and needs no
    filter. ``0`` leaves a pure P, which is stable here because the tracker is itself a closed
    loop on the commanded velocity; raise it only if the approach visibly overshoots.

    Ignored entirely when :attr:`freeform_approach_kp` is ``0``.
    """

    freeform_approach_min_speed: float = 0.0
    """FLOOR on the commanded forward speed while the goal is still ahead (m/s). ``0`` is off.

    Without it the P law decays continuously and the command spends the last couple of metres
    in the 0.05-0.3 m/s band -- which is both where the robot stops walking for reasons no
    reward can see, and a band the generator has barely trained on (5.8% of its command rows
    are ``|vx| < 0.1``; ~7% fall in 0.1-0.3). The robot then settles wherever its own deadband
    put it, SHORT of the goal, and is scored as not having arrived: observed directly, robots
    coming to rest just outside the goal region while otherwise behaving correctly.

    With a floor the command is a RAMP then a STEP -- cruise, decelerate to the floor, hold it
    to the stop radius, then zero -- so every commanded value is either 0 or inside the trained
    band, and the robot is still walking when it crosses into the region it has to hold.
    :attr:`freeform_cmd_hold_s` makes the same ramp-vs-step argument at the other end of the
    episode.

    Capped per env by that episode's own drawn speed, so a slow draw is never sped up and
    :attr:`freeform_speed_min` keeps defining the band. Not applied past the goal -- there the
    floor would drive the robot further away, into whatever is behind it.

    Ignored entirely when :attr:`freeform_approach_kp` is ``0``.
    """

    freeform_stop_radius_m: float = 0.0
    """Range within which the WHOLE command triple is zeroed (m). ``0`` disables the stop.

    Inside it the robot is commanded to stand: ``vx = 0`` and ``wz = 0``. Zeroing yaw too is
    not tidiness -- :func:`~.fm_tracking_cmd.freeform_yaw_rate` aims at a point ``lookahead``
    metres down the centreline, and once the robot is AT the goal that bearing is measuring
    the last of the approach error at ever-shorter baseline, so a robot trying to hold station
    would be commanded to keep turning. It mirrors the opening hold, which zeroes the triple
    for the same reason: a yaw rate on a standing robot is as much a step input as a speed is.

    Subtracted inside the P term as well, so the commanded speed reaches 0 AT the radius
    rather than stepping to it -- the two are one continuous law, not a ramp plus a cliff.

    Ignored entirely when :attr:`freeform_approach_kp` is ``0``.
    """

    freeform_wz_max: float = 0.68
    """Symmetric clip on the steering yaw rate (rad/s).

    0.68 is the largest ``|wz|`` the training library ever commands, on the box family
    (``transfer/sim/mjeval/runner.STEER_WZ_MAX``, restated because ``robot_rl`` cannot import
    the MuJoCo half). It is a bound on how far outside the commanded distribution the
    generator is pushed, not a physical limit.
    """

    # --- Debug visualization ---
    fm_perturb_fraction: float = 1.0
    """Fraction of envs ELIGIBLE for the diagnostic perturbations, drawn per episode.

    An outer gate over :attr:`fm_upper_camera_noise_fraction` and
    :attr:`fm_history_null_fraction`: an env outside it is pristine no matter what those say.

    **Why a gate rather than just halving the two fractions.** Independent draws at 0.5 each
    leave only ``0.5 * 0.5 = 25%`` of envs untouched, and which envs those are moves with every
    draw.  A gate guarantees a clean control population of a stated size, in the SAME run and
    therefore on the same terrain, tiles and command distribution -- which is what lets the
    recording be split by arm afterwards instead of compared across runs.

    ``1.0`` (default) reproduces the ungated behaviour: every env is eligible.

    **MEASURED WORTH IT (2026-09-08).**  The AWR arm trained on data collected with the
    upper-camera perturbation moved Isaac box2 success ``0.1421 -> 0.2650`` (+0.125, 30 of 30
    matched cells) against ``+0.050`` for the unperturbed v1 arms, with a correspondingly
    wider advantage (sd 0.117 vs 0.088) -- the perturbation's job is to widen the state and
    plan distribution, and it did.  **Confounded**: that run also changed course (v2 adds 3 m
    of run-out past the second box), so this is not an isolated attribution.  See
    ``scripts/flow_matching/finetune_generator.py``'s module docstring.

    **The perturbations are a BEHAVIOUR POLICY, not a data corruption.** They are applied to
    what the generator is CONDITIONED on, and the recorder stores the true history and the true
    camera regardless -- so the critic and the fine-tune learn from the measurement that
    actually occurred, and the perturbation's only job is to widen the distribution of states
    and plans the run visits.  That split is load-bearing and is pinned by
    ``test_the_recording_stores_the_TRUE_conditioning``.
    """

    fm_upper_camera_noise_fraction: float = 0.0
    """Fraction of envs whose UPPER (far) depth camera is replaced with garbage, re-drawn per
    episode.

    The upper camera is the one that previews terrain AHEAD -- mounted at z = 0.285 m and
    pitched 15 deg down, with its boresight meeting the ground 4.25 m out, against the near
    camera's 0.51 m.  So this is the knob that asks whether a plan's MODE depends on forward
    terrain preview: if the generator still commits to the same box dismount with the far
    camera destroyed, the decision is not coming from looking ahead.

    **Which sensor is "upper" is DERIVED from the checkpoint's own ``extero_spec``** (the
    highest ``offset_pos[2]``), never from a hardcoded index.  The obs group's term order is
    what fixes the sensor axis today, and a reordering there would silently corrupt the near
    camera instead while the flag went on claiming otherwise.

    See :attr:`fm_upper_camera_noise_mode` for what "garbage" means -- the choice matters more
    than the fraction does.
    """

    fm_upper_camera_noise_mode: str = "gaussian"
    """How :attr:`fm_upper_camera_noise_fraction` corrupts the camera: ``"gaussian"`` or
    ``"dead"``.  **These test different things and only one of them is in distribution.**

    ``"dead"`` is what a failed or fully-occluded camera looks like: every pixel at the far end
    of the depth band, xyz zeroed, and the VALIDITY channel set to 0.  The model was trained
    with 6.4-8.9% of pixels carrying no reading and has a validity channel for exactly this, so
    a whole camera reading "no data" is an extrapolation along a direction it understands.

    ``"gaussian"`` replaces every channel -- validity included -- with unit-variance noise.
    Nothing in training looks like this: the model is handed an input asserting, incoherently,
    that it has valid readings of random geometry.  It is the harsher test of "can this be
    ignored", but a behaviour change under it does not distinguish "the plan needed that
    camera" from "the model was pushed off distribution", so read a result from it narrowly.
    """

    fm_upper_camera_noise_cadence: str = "episode"
    """How often :attr:`fm_upper_camera_noise_fraction` is re-drawn: ``"episode"`` or
    ``"replan"``.  **They ask different questions.**

    ``"episode"`` (default) is a BROKEN camera: drawn once at reset and held, so a robot is
    either on the arm for a whole traverse or not at all.  That is the right shape for "can
    this policy work without forward preview" and for domain randomisation, and it is the only
    cadence under which an episode's outcome is attributable to the arm.

    ``"replan"`` is intermittent DROPOUT: re-drawn for every env that re-plans, so the camera
    can be fine on approach and dead at the moment the plan commits to a dismount.  That is
    the sharper probe of whether a particular DECISION uses the preview, and it avoids a
    confound the per-episode cadence has: training corrupted 20% of windows independently
    (``null_cond_prob``), never a whole episode, so a failure late in a fully-corrupted episode
    may be compounding state drift rather than the camera mattering at that moment.

    Under ``"replan"`` the ribbon's root colour changes as the arm does, so each drawn plan is
    coloured by the conditioning it was actually sampled under.
    """

    fm_history_null_fraction: float = 0.0
    """Fraction of envs served the model's learned NULL-HISTORY embedding instead of their
    own history, re-drawn on :attr:`fm_history_null_cadence`.

    **What this is for: breaking a committed MODE.** The generator is autoregressive on its
    own past output, so once it reads a box top as a descending staircase it keeps being
    conditioned on a plan that says "stairs down" and the reading persists.  History-nulling
    removes that conditioning, so the mode is re-decided from exteroception and the command
    alone at every re-plan of an affected env.

    **It is IN DISTRIBUTION, not a hack.** Training nulls the history on ``null_prob`` of its
    windows (0.3 by default) and the model carries a learned ``null_history`` embedding for
    exactly that case -- the same embedding that already covers the first steps after a reset,
    where no history exists yet (see :meth:`BatchedHistory.null_mask`).  What is new here is
    only WHICH envs get it and for how long.

    **Drawn at random per env, never strided.** Envs map to tiles through a structured grid
    lookup, so taking every n-th env would risk putting every history-nulled env on the same
    tile and confounding the comparison with terrain.

    ``0.0`` (default) leaves the serve path exactly as it was: the mask is then whatever
    ``BatchedHistory.null_mask`` says, i.e. post-reset underfill only.
    """

    fm_history_null_cadence: str = "episode"
    """How often :attr:`fm_history_null_fraction` is re-drawn: ``"episode"`` or ``"replan"``.
    Mirrors :attr:`fm_upper_camera_noise_cadence` field for field, so the two arms can be
    configured the same way and a run can state one cadence for both.

    ``"episode"`` (default) draws once at reset and holds, so one robot plans without history
    from spawn to fall.  That is the cadence under which an EPISODE's outcome is attributable
    to the arm, and it is what a recording split by arm wants.

    ``"replan"`` re-draws for every env about to sample, so the conditioning changes mid
    episode.  Two reasons to want it, and the first is the stronger:

    * **It is what TRAINING did.** The model nulls history on ``null_prob`` of its WINDOWS
      (0.3 by default), independently, never for a whole episode -- so per-re-plan is the
      cadence the learned ``null_history`` embedding was actually fitted under.  Under
      ``"episode"`` a failure late in a fully-nulled episode may be compounding state drift
      rather than the missing history mattering at that moment; this removes that confound,
      exactly as the camera arm's own ``"replan"`` does.
    * **It is far easier to SEE.** The arm is gated by
      :attr:`fm_perturb_fraction`, so at a small ``--num_envs`` a per-episode draw leaves
      most resets with no affected env at all and nothing to look at for a whole episode.

    The cost, and it is real: an episode is then a mixture, so no episode-level outcome can be
    attributed to the arm.  Use ``"episode"`` for anything being measured and ``"replan"`` for
    anything being watched or compared per PLAN.

    Under either cadence the ribbon colours each drawn plan by the conditioning it was
    actually sampled under, because the draw lands before the plan does (see
    ``_snapshot_and_infer``); under ``"replan"`` an env on the history arm therefore alternates
    between the two arm colours as the camera arm re-rolls beside it.
    """

    fm_speed_mismatch_fraction: float = 0.0
    """Fraction of envs whose CONDITIONING command speed is offset from the one the robot is
    actually being asked for, re-drawn on :attr:`fm_speed_mismatch_cadence` and held for
    :attr:`fm_speed_mismatch_replans`.

    **What this is for: the approach speed is a MODE choice, and the generator only ever sees
    one.** On the free-form path the episode draws a single speed and holds it, so the command
    channel is constant for the whole run and the generator has never had to plan a box
    approach at a speed other than the one it is travelling at. That makes the command a
    near-redundant input, and it is the input a deployed joystick moves most.

    **It is a CONDITIONING corruption, not a new command.** The override reaches
    ``fut_cmd`` inside :meth:`_run_fm` and nothing else, so the robot is still ASKED for the
    nominal speed: the tracking rewards, the ``command_window`` observation, the eval's
    tracking target and the recorded ``cmd_future`` all keep the true value. The right reading
    is the dead upper camera's, not a curriculum's -- an input the generator must stay robust
    to, and one the progress reward deliberately scores against the NOMINAL speed so a
    generator that follows the corrupted number is penalised for it.

    **Drawn at random per env, never strided**, for the same reason as the other two arms:
    envs map to tiles through a structured grid lookup, so every n-th env risks putting the
    whole arm on one tile and confounding it with terrain.

    ``0.0`` (default) leaves the conditioning exactly as it was.
    """

    fm_speed_mismatch_range_mps: float = 0.75
    """Half-width of the uniform OFFSET applied to the conditioning speed, in m/s.

    An OFFSET rather than an absolute speed, because :meth:`reset` redraws the episode's
    nominal ``_ff_speed`` AFTER the arm's own draw -- an absolute would be stale by exactly
    one episode, and stale in a way that still looks like a plausible speed.

    0.75 against the free-form band's own 0.5--2.2 m/s: large enough that the conditioning
    names a different gait regime than the body is in, small enough that the offset speed is
    still inside the range the generator was trained on. A corruption that pushed the command
    outside the training band would test extrapolation instead of robustness.
    """

    fm_speed_mismatch_replans: tuple[int, int] = (2, 3)
    """Inclusive range of consecutive re-plans one drawn offset is HELD for.

    A mismatch has to outlast a single plan to matter: the generator re-plans every 0.24 s and
    a one-window corruption is corrected before the body has moved. Holding it for 2--3
    re-plans (0.48--0.72 s) covers most of an approach stride while still letting several
    draws land in one episode.

    The countdown decrements per re-plan and a fresh activation is rolled ONLY where it has
    reached zero, so an active hold is never re-rolled mid-persistence -- which is what makes
    the hold length mean what it says.
    """

    fm_speed_mismatch_cadence: str = "episode"
    """How often :attr:`fm_speed_mismatch_fraction` is re-drawn: ``"episode"`` or ``"replan"``.

    Mirrors :attr:`fm_history_null_cadence` and :attr:`fm_upper_camera_noise_cadence` field
    for field, with the same trade-off: ``"episode"`` makes an episode's outcome attributable
    to the arm and is what a recording being MEASURED wants, while ``"replan"`` re-rolls the
    activation whenever a hold expires and is easier to WATCH at small env counts.

    Note the hold in :attr:`fm_speed_mismatch_replans` applies under BOTH cadences -- under
    ``"episode"`` it decides how long each activation within an eligible episode lasts, and
    under ``"replan"`` it additionally bounds how often the env can re-roll.
    """

    debug_vis_goal_disc: bool = False
    """Draw the DWELL disc the robot has to stop inside, one translucent cylinder per env.

    Off by default, like every other course-specific marker, and it draws nothing unless the
    task also wires :class:`~...terminations.terminations.freeform_goal_dwell` -- the radius is
    read back OFF THAT TERM rather than restated here, so the picture cannot disagree with the
    rule it is a picture of. A cfg that asks for the disc with no dwell term WARNS rather than
    drawing nothing silently: an absent marker reads exactly like a region correctly placed
    off-screen, which is the one reading a goal visualiser must never allow --
    ``MotionSequenceCommand._load_goal_spec`` records the same hazard for the gate.

    Needs the master ``debug_vis`` on and ``--viz kit``, like the rest.
    """

    debug_vis_show_fm_prediction: bool = True
    """Draw the FM's predicted world-frame trajectory as a ribbon of spheres (one color per
    frame). Overlay it on the yellow observation-window ribbon (``debug_vis_obs_window``) to
    check that what the policy observes is what the generator predicted."""

    debug_vis_fm_marker_radius: float = 0.02
    """Sphere radius (m) for the FM-prediction ribbon points."""
