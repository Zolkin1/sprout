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
    """

    stash_conditioning: bool = False
    """Keep each plan's CONDITIONING, as built, for an observation term to read.

    Off by default because it is dominated by a depth row per env and only the RL adjuster
    reads it -- its network re-encodes this conditioning to borrow the generator's frozen
    prefix, and the borrow is only exact if it gets the tensors that actually produced the
    plan (rebuilding them at observation time would read a different instant and a different
    camera lag).

    **A cfg flag rather than a method call on the env, and the ordering is why.**
    ``ObservationManager._prepare_terms`` invokes every term ONCE at construction to discover
    its shape, and the observation manager is built inside ``load_managers`` -- so an env that
    enabled this after ``super().__init__()`` would have the term raise during that probe.
    The command manager is constructed FIRST (``ManagerBasedRLEnv.load_managers`` builds it
    before delegating to the base), so a flag read in ``_post_init`` is early enough and a
    method call from the env is not.
    """

    stash_flow: bool = False
    """Keep each plan's final Euler-step velocity field, for an observation term to read.

    Same trade and the same ordering reason as :attr:`stash_conditioning`. The adjuster reads
    it beside the window it corrects, so each of its query tokens carries both what the
    generator predicted at that node and which way the flow was still moving there.
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

    generator_kind: str = "flow_matching"
    """WHICH KIND of generator :attr:`fm_checkpoint` is -- ``"flow_matching"`` or
    ``"token_planner"``.

    Deliberately NOT a third value of :attr:`reference_source`. That field answers "is the
    applied reference the clip or a generated one?", which is a different question from "which
    generator produced it"; folding them together would make every consumer that tests
    ``reference_source == "fm"`` -- the eval runner, ``fm_debug.py``, the sweeps, the data
    logger -- silently stop recognising a generated arm.

    The two kinds are served through the same interface: a context carrying the same fields
    and a ``sample_window`` with the same signature, returning ``(V, F, D)`` raw window nodes.
    Everything in this command downstream of the load is written against that interface and
    does not branch on the kind. What DOES differ is refused rather than faked -- a token
    planner has no ``x0`` noise and no flow field, so :attr:`fm_zero_noise`,
    :attr:`fm_guidance_scale`, :attr:`fm_num_sampling_steps` and the adjuster's ``plan_flow``
    do not apply to it and it says so.
    """

    planner_greedy: bool = True
    """Take the argmax at each decode step instead of sampling. ``generator_kind="token_planner"`` only.

    **Its own field, rather than a re-use of :attr:`fm_zero_noise`.** Both do mean "make the
    plan a deterministic function of the conditioning", and an earlier version mapped one onto
    the other -- but ``fm_zero_noise`` is named for a mechanism a token planner does not have
    (there is no ``x0`` to integrate from; ``sample_window`` REFUSES a noise tensor), so the
    flag would have meant something different depending on which artifact was loaded, with
    nothing on screen to say which. That is the failure this file already records for
    ``window_anchor``, ``attention_mode`` and ``history_source``.

    It defaults TRUE where ``fm_zero_noise`` defaults False, and deliberately: the deployed
    controller runs the deterministic arm, and the difference here is not small. Measured on
    the epoch-20 checkpoint, sampling changes **98% of the 16 tokens** against greedy and
    moves base position by 0.169 m sd over 20 draws -- so a default of "sample" would have
    made every first look at a planner a look at one arbitrary draw.
    """

    planner_force_tokens: int = 0
    """Teacher-force this many LEADING decode tokens from the clip. ``token_planner`` only.

    ``0`` is off. ``n > 0`` encodes the clip's own future window through the frozen
    tokenizer at each re-plan and pins the first ``n`` of the plan's ``N`` tokens to the
    result, letting the model decode the rest conditioned on them; ``-1`` means all of them.

    **This is a strictly finer instrument than ``history_from="clip"``, and they compose.**
    The history arm forces the CONDITIONING and leaves the plan free, which is the only
    teacher forcing a flow generator can express (``sample_window`` there refuses this
    argument by name). Here the plan itself can be pinned chunk by chunk, so the two ends of
    the loop separate: with the history forced and ``n = 0`` a bad plan is the model's own;
    sweeping ``n`` from ``0`` to ``N`` walks from the model's plan to the clip's and says how
    much of the error each token carries.

    ``-1`` is the useful control at both ends: it should reproduce the TOKENIZER's
    reconstruction of the clip and nothing better, so a run that still fails there is failing
    for a reason no amount of planner training can fix.
    """

    planner_temperature: float = 1.0
    """Softmax temperature for the categorical draw. Ignored when :attr:`planner_greedy`."""

    planner_seed: int | None = None
    """Seed for the sampling RNG, or ``None`` to draw off the global one.

    Set it for anything whose result has to be replayable. The flow generator's equivalent
    handle is ``evaluate.py --fm_seed``, which reaches the model as an injected ``noise``
    tensor -- a path a planner refuses by design, so without this field a stochastic planner
    run would have had no reproducibility handle at all.
    """

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

    seed_history_from_stand: bool = False
    """Seed a resetting env's generator history with the ROBOT'S OWN spawn state, repeated.

    The fix for the first inference of an episode, and it exists because the alternatives do
    not cover a standing spawn:

    * the ``null_history`` embedding is phase-AGNOSTIC, fitted on 20% of windows nulled at
      random from a corpus that is overwhelmingly walking -- so it has no way to express
      "both feet are planted and the robot is still", and the opening plan comes out
      mid-gait, feet off the ground, drifting in whichever direction the prior leans;
    * :attr:`seed_history_from_clip` reads the clip's PRECEDING steps, of which there are
      none at ``start_step = 0``, and which would describe a mid-stride robot anyway when
      :attr:`stand_start` has just placed a standing one.

    What this writes is true rather than fabricated: the robot IS standing at that pose, and
    under ``history_from="measured"`` the ring holds achieved states, so N copies of the spawn
    state say exactly "it has been standing here". That is the distinction from the warm-up
    paths this repo deleted, which filled the ring with ZEROS -- a state no episode ever
    started from.

    Pairs with :attr:`stand_start` and :attr:`cmd_hold_s`: standing pose, standing history,
    zero command. All three agree, and the generator is asked for the one thing it can then
    unambiguously produce.
    """

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

    # --- Debug visualization ---
    debug_vis_show_fm_prediction: bool = True
    """Draw the FM's predicted world-frame trajectory as a ribbon of spheres (one color per
    frame). Overlay it on the yellow observation-window ribbon (``debug_vis_obs_window``) to
    check that what the policy observes is what the generator predicted."""

    debug_vis_fm_marker_radius: float = 0.02
    """Sphere radius (m) for the FM-prediction ribbon points."""
