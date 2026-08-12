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

    # --- Best-of-N oracle selection (diagnostic) ---
    fm_num_draws: int = 1
    """Sample this many independent windows per re-plan and SERVE the one closest to the clip.

    ``1`` (default) = one draw, i.e. no selection: the sampled window is served as-is, which
    with ``fm_zero_noise`` is the DEPLOYABLE configuration. It is still recorded as a
    one-entry draw set with its clip score (diagnostic only, it changes nothing about what is
    served), so a deploy-representative capture is readable by the same figures and browser as
    an oracle capture and comparable through the same number. Above that this is an **ORACLE**: the
    score consults the clip reference, which does not exist on hardware, so nothing here is
    deployable. What it measures is the model's SUPPORT -- whether a good window is in the
    distribution at all -- which is the upper bound on what any deployable selection rule
    (draw consensus, a learned critic) could ever buy. It exists because the sampling spread
    at fixed conditioning was measured to be the same order as the total prediction error.

    The selected draw becomes the served window, so it also feeds the autoregressive history
    (see ``history_from``): the loop continues from the chosen draw, which makes this
    "oracle-guided closed loop" -- a third case between teacher forcing and the real loop.

    Costs N forwards per re-plan, batched as one N-times-larger call. The generator is ~99%
    of per-step inference, so wall-clock scales with N once the batch stops being
    launch-bound -- fine at a handful of envs, not something to leave on for a sweep.

    Requires Gaussian noise: with ``fm_zero_noise`` every draw is identical, so the
    combination is refused rather than silently selecting among N copies."""

    fm_diag_draws: int = 0
    """Sample this many EXTRA windows per re-plan for diagnostics only. Never served.

    The point: keep the physics on the deployable sample -- one draw, and with
    ``fm_zero_noise`` the deterministic one -- while still recording what the sampling
    distribution looks like around it. So the x-z figures and the viser draws layer show the
    spread, and the rollout they are annotating is still the one you would deploy.

    Distinct from ``fm_num_draws``, which SELECTS: that one changes what is served (an oracle,
    since it scores against the clip) and is therefore incompatible with ``fm_zero_noise``.
    This one changes nothing about the rollout, so it composes with either serving mode.

    Recorded as draw 0 = the SERVED window, draws 1..N = the diagnostic samples, and
    ``fm_chosen`` = 0. Every entry is scored against the clip, so the plots highlight what was
    actually served and grey out what merely could have been sampled.

    Ignored when ``fm_num_draws > 1`` (the selection path already records its whole candidate
    set). Costs N extra forwards per re-plan, batched -- the same price as selection, for
    information instead of behaviour."""

    fm_select_w_base_pos: float = 1.0
    fm_select_w_base_ori: float = 0.0
    fm_select_w_base_lin_vel: float = 0.0
    fm_select_w_base_ang_vel: float = 0.0
    fm_select_w_joint_pos: float = 0.0
    fm_select_w_joint_vel: float = 0.0
    fm_select_w_ee_pos: float = 0.0
    """Per-block weights for the draw score (``fm_num_draws > 1``).

    The score is ``sum_key w_key * RMS_key``, where ``RMS_key`` is the root-mean-square
    difference between the draw and the clip window over that key's channels and all F
    nodes, in RAW units (metres, m/s, rad). Weighting per key rather than per LAYOUT block
    because ``base_pose`` and ``base_vel`` are each two halves in different units -- a single
    RMS over ``base_pose`` would add metres to a 6D rotation. Keys are
    :data:`~robot_rl.flow_matching.window.REFERENCE_BLOCK_KEYS`' values, and the expansion to
    channels goes through ``expand_window_blocks``, the same helper the noise and scale tables
    use, so a layout change cannot misalign the weights here alone.

    The default is ``base_pos`` only: one interpretable unit (metres of base-position error,
    summed over the horizon), and the quantity closest to the failure this was built for --
    where the base ends up relative to a stair edge. Everything else defaults to 0 rather
    than to a guessed relative scale. Turn a key on once you have measured its magnitude:
    with ``fm_select_log_every`` set, every key's RMS is printed whether or not it is
    weighted, so one run tells you what a defensible weight would be.

    Separate float fields rather than one dict because IsaacLab's configclass infers a
    field's type from its default: a ``None`` default makes hydra reject every value (the
    trap already documented on ``fm_num_sampling_steps``), and a mutable dict default cannot
    be overridden key-wise from the command line under struct mode. Weighting a key the
    checkpoint does not predict is an error, not a no-op."""

    fm_select_log_every: int = 0
    """Print the draw-selection summary every this many re-plans (``0`` = never).

    One line: the chosen draw's index, the best/worst/spread of the score across the N draws,
    and every key's RMS for the chosen draw. The SPREAD is the number that says how much
    headroom selection has -- if best and worst are close, no selection rule can buy
    anything and the model itself is the problem."""

    # --- Debug visualization ---
    debug_vis_show_fm_prediction: bool = True
    """Draw the FM's predicted world-frame trajectory as a ribbon of spheres (one color per
    frame). Overlay it on the yellow observation-window ribbon (``debug_vis_obs_window``) to
    check that what the policy observes is what the generator predicted."""

    debug_vis_fm_marker_radius: float = 0.02
    """Sphere radius (m) for the FM-prediction ribbon points."""

    fm_viz_num_draws: int = 0
    """Draw this many of the N sampled windows, INCLUDING the chosen one (``0`` = chosen only).

    The chosen draw keeps the full-opacity ribbon; the next ``fm_viz_num_draws - 1`` by score
    are drawn faded, so the picture shows the spread the selection had to work with and which
    of it was picked. Capped at ``fm_num_draws``.

    Fading is per-PROTOTYPE, not per-instance: ``VisualizationMarkers.visualize`` takes only
    translations / orientations / scales / marker_indices, with no per-point colour, so each
    ribbon frame gets a second low-opacity prototype. That doubles the prototype count and
    multiplies the sphere count by the number drawn -- at F=62 and 5 frames a draw is ~310
    spheres, so keep this small (4 is legible; 16 is a cloud)."""

    debug_vis_fm_reject_opacity: float = 0.25
    """Opacity of the non-chosen draws' ribbons (see ``fm_viz_num_draws``).

    ``PreviewSurfaceCfg.opacity`` affects interactive rendering only, which is exactly this
    use -- it will not show up in a recorded video pass."""

    debug_vis_fm_reject_scale: float = 0.6
    """Radius multiplier for the non-chosen draws' spheres, on top of the opacity.

    Applied through the per-instance ``scales`` argument, which IS available even though
    colour is not -- so shrinking needs no extra prototypes. Transparency alone reads poorly
    against terrain at F=62; smaller and fainter together separates cleanly."""
