# Copyright 2026 Zachary Olkin. All rights reserved.

"""The adjuster task: train a residual on the generator's plan, tracker frozen.

Derives from the depth TEACHER (:class:`G1TeacherTerrainDepthEnvCfg`), which is where the
scene, the cameras, the depth observation terms and the camera randomisation all live.

Not from the FM tracker PLAY task, though that is the other closed-loop cfg in the tree.
Inheriting it was tried and reverted: this class replaces its observations, its terminations
and its rewards outright, so the only things it actually contributed were the FM command cfg
-- imported directly below, which is reuse without the coupling -- and one
``tracking_mode="joint"`` line, restated here with its reason. Deriving from a task nobody
runs, to borrow one line, is a dependency that only creates ways to be surprised.

Not from the tracker fine-tune either, and that one would have been wrong twice over: it pins
``reference_source="clip"`` as its stated safety property, and its rewards grade the tracker's
joint action, which the adjuster does not emit.

**What it adds**: the FM command, two actor observation groups, a widened critic group, four
terminations, five rewards, and the env-level fields that let a frozen tracker run inside the
step.

## The tracker's own rewards do NOT carry over

Not one of ``G1ClfTrackingRewardCfg``'s seven terms. ``dof_torques_l2``, ``torque_limits``,
``dof_pos_limits`` and ``action_rate_l2`` all grade a joint action the adjuster does not emit
-- and ``action_rate_l2``, reading the action manager, would have silently become the
residual's rate over thousands of dimensions. ``clf_reward`` and
``clf_decreasing_condition`` grade tracking, which the dense term here already does, through a
measured kernel rather than a CLF normalisation. ``undesired_contacts`` is the one arguable
loss; the failure termination covers it, and it goes back only if the adjuster is seen to
learn to scrape.

## Why the goal is a truncation and the deadline a termination

Only a truncation collects ``gamma * V(s)``, so making arrival the only truncation IS the
mechanism for rewarding it: with a positive dense reward, V is positive everywhere, and
arriving is worth more than running the clock out even though neither carries a bonus. It
costs no new constant and introduces no dependence on how long a clip happens to be.

That needs the deadline to be OBSERVABLE, or the critic is asked to predict a cut it has no
evidence for -- the classic time-limit bootstrapping error. ``goal_geometry``'s fourth channel
is that evidence. If it is ever dropped from the critic group, ``clip_end`` has to go back to
``time_out=True``.
"""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab.sensors import RayCasterCfg, patterns
from robot_rl.tasks.manager_based.robot_rl import mdp
from robot_rl.tasks.manager_based.robot_rl.mdp.sensors.ray_fan import RayFanPatternCfg
from robot_rl.tasks.manager_based.robot_rl.envs import AdjusterEnvCfg

from .g1_teacher_env_cfg import (
    PROPRIO_HISTORY_LENGTH,
    _REF_PARAMS,
    _REF_SCALE,
    _REF_WINDOW_PARAMS,
)
from .g1_depth_env_cfg import (
    G1TeacherTerrainDepthEnvCfg,
    G1TeacherTerrainDepthObservationsCfg,
)
# The MDP the off-policy fine-tune ladder is measured on, composed wholesale by the free-form
# adjuster arm at the bottom of this file.
from .g1_finetune_env_cfg import G1FinetuneFreeformEnvCfg
# The closed-loop FM command, imported rather than inherited. It is the one genuinely shared
# piece: ~40 lines of reasoned config (the "plan" anchor, the CLF weights, no default clip or
# checkpoint, the CPU clip device) that must not be restated and drift.
from .g1_tracker_env_cfg import G1FmTrackerCommandsCfg

#: Seconds of slack past the clip's last usable step before the deadline fires.
#:
#: Without it the budget is IDENTICALLY the clip's own remaining duration -- the cursor is
#: ``start_step + episode_length_buf`` -- so arriving is a race and hurrying is the reliable
#: strategy. Commanded speed is an established degradation vector on stairs, so that is a real
#: failure mode and not a theoretical one.
#:
#: 3.0 s is +36% on the median clip and tolerates being ~3.6 m behind schedule at 1.2 m/s,
#: against a measured clip-arm 0.99-quantile downrange shortfall of 0.115-0.235 m. So the
#: slack is 15-30x the demonstrated lag: tracking well is comfortably the way to arrive, and
#: sprinting buys nothing.
CONTROL_DT: float = 0.02
"""The tracker's control period. Only used to express a step count as a fraction."""

GOAL_OFFSET_SCALE: tuple[float, float, float] = (1.0 / 10.0, 1.0 / 1.5, 1.0 / 0.6)
"""Reciprocals of a characteristic (downrange, lateral, height) size, in metres.

The critic group is deliberately un-normalized, and every term in it is O(1) by construction
-- except this one was not. Measured over 60 agent steps at 32 envs: downrange reached 14.1 m,
giving `goal_geometry` a std of 2.105 against <= 0.93 for every other term. The three numbers
are the tile's downrange extent, the fitted goal HALF-WIDTH and its height TOLERANCE, so 1.0
reads as "about one goal-region away" -- the scale `mdp.in_goal` thresholds on. Declared
rather than fitted, like the extero contract normalization, so it means the same thing in
every run.
"""

STEER_KP: float = 1.5
"""Gain of the yaw controller that holds the robot on the goal bearing.

Nothing else in this stack steers -- the command is a body-frame velocity and the generator
plans from depth -- so accumulated heading error over several metres becomes a lateral miss,
and on a thin flight of stairs a lateral miss is a fall. Closing this one loop is the largest
single effect measured on this pair, larger than any conditioning change tried.

OFF-DISTRIBUTION on the stair families by construction: their clips carry |wz| = 0 across the
whole library, so the generator has never been asked to turn there. That is the point -- the
adjuster is being trained on a command distribution the generator cannot already serve -- but
it also means results here are not comparable to unsteered ones.
"""

STAND_SETTLE_S: float = 0.3
"""Seconds of zero command at the start of an episode, to stand and settle.

0.3 s is 15 control steps, so it spans the first re-plan interval (12 steps at
``fm_update_every=12``) with margin: the opening plan AND the one after it are drawn while
the command is still zero, so the robot is standing before it is asked to move.
"""

#: Seconds of grace past the clip's last usable step. NOT restated here: the goal anchor sits
#: in the clip's final 1.5 s, past that step, so the slack is what makes the region reachable
#: at all -- one decision, one constant, shared with the eval harness's added `clip_end`.
GOAL_SLACK_S: float = mdp.GOAL_DEADLINE_SLACK_S

#: Episode budget in seconds. The longest clip's last usable step is 948 control steps; plus
#: the slack that is 1098, so 24.0 s (1200 steps) leaves ``time_out`` a genuine backstop rather
#: than the binding constraint on the 20 s flat clips.
EPISODE_LENGTH_S: float = 24.0

#: Episode budget for the FREE-FORM arm, seconds.
#:
#: 40.0, and it is not a choice: it is what `run_finetune_loop.sh` passes at collect
#: (`env.episode_length_s=40.0`), so the off-policy arms' episodes are this long and the PPO
#: arm's must be too. On this course the budget is genuinely non-binding on the slow x far
#: cells, which is the property that keeps a time-out from being scored as a policy failure.
FREEFORM_EPISODE_LENGTH_S: float = 40.0

#: Envs the free-form arm trains on. What `configs/finetune/boxes.yaml` declares and what the
#: off-policy collect runs, so the two arms see the same throughput per iteration. NOT part of
#: the MDP; at EVAL it is, because it sets the censored fraction.
FREEFORM_NUM_ENVS: int = 2048

#: Knots along the plan's node axis for the free-form arm: ONE PER NODE, so the action is the
#: full 62 x 44 = 2728-dim residual. `knot_basis(F, F)` is exactly the identity (verified,
#: `max|B - I| = 0.0`), so this is a config field rather than a second code path -- and 13,
#: the clip arm's value, is the documented fallback if the mean cannot move at full width.
FREEFORM_NUM_KNOTS: int = 62

#: Reciprocals of a characteristic (downrange, lateral, height) size on the BOXES course, m.
#:
#: The clip families' numbers do not transfer, so they are restated rather than reused:
#:
#: * downrange 10 m -- a boxes tile's goal sits `last_box + RUNOUT_M(3.0) + TARGET_LEAD_M(1.5)
#:   - GOAL_DEPTH_M/2(1.0)` from its start, which over the drawn geometry of
#:   `configs/finetune/boxes.yaml` is ~8-14 m;
#: * lateral 0.75 m -- the route's own `goal_half_width`, floored at `GOAL_HALF_WIDTH_FLOOR_M`,
#:   which is what `logs/ft_boxes/goal_regions.json` declares and what the termination passes
#:   as `min_half_width`;
#: * height 0.5 m -- the `height_tol_m` both launchers pass.
#:
#: So 1.0 reads "about one goal-region away", the scale the termination thresholds on. Declared
#: rather than fitted, so it means the same thing in every run.
FREEFORM_GOAL_OFFSET_SCALE: tuple[float, float, float] = (1.0 / 10.0, 1.0 / 0.75, 1.0 / 0.5)


@configclass
class G1AdjusterObservationsCfg(G1TeacherTerrainDepthObservationsCfg):
    """Five groups: two the frozen tracker reads, two the adjuster reads, one the critic reads.

    **Extends the DEPTH observations rather than re-declaring the groups**, and that is not
    cosmetic: the depth variant's ``PolicyCfg`` is what nulls the height scan and adds the two
    cameras. An earlier version built ``policy`` from the plain teacher's group and the frozen
    tracker refused to load against it -- 1106 wide against the 2490 its trunk expects, with
    ``depth_near`` and ``depth_far`` missing and a ``height_scan`` the tracker had never seen.
    That is precisely the failure ``FrozenTracker``'s per-term oracle exists to name, and it
    named it; without that check it would have surfaced as a ``load_state_dict`` shape error
    about ``mlp.0.weight``.

    So ``policy`` and ``reference`` are inherited untouched in WIDTH and term list -- the
    tracker is frozen and its observation contract is fixed by its checkpoint -- but
    ``reference``'s NOISE is switched off here, which changes no shape and is measured.

    The teacher hardens its tracker against ``REFERENCE_WINDOW_NOISE`` (base position
    +-0.06/0.06/0.03 m, joints +-0.1 rad, velocities +-0.25) because at deploy the generator
    predicts the reference and its output is not clean. That is right for TRAINING THE
    TRACKER. It is wrong here, because the adjuster's whole job is to write a correction into
    the same channels, and it has to arrive through the tracker's encoder to change anything:
    on v6 the learned correction reached ~0.006 m of base position after 840 iterations, an
    order of magnitude UNDER the +-0.06 m of i.i.d. jitter being added to the very channel it
    was writing into. The tracker is also already near-indifferent to i.i.d. reference jitter
    -- the eval measured all DR on costing at most 2.7 points -- so this noise buys robustness
    the tracker has and costs the adjuster its signal.

    Note this corrupts only the tracker's VIEW; the reward grades the served plan, which the
    residual changes directly. The two are different channels, and it is the view that decides
    behaviour.

    ``fm_cond`` and ``fm_plan`` are the adjuster's, and they are observations rather than
    something the policy recomputes because PPO's rollout runs under ``inference_mode``:
    nothing built there can enter an autograd graph later, so ``update`` re-runs the actor from
    stored observations. Both are noise-free (``enable_corruption=False``): they are the
    generator's own state, not a sensor, and the plan arrives in a DECLARED normalized space
    that an empirical normalizer would demote to a dataset statistic.
    """

    #: Whether the ENV silences the inherited reference-window noise. Not done here, because
    #: the two adjuster tasks answer it differently and an observation group is the wrong place
    #: to record an MDP decision: the CLIP-terrain arm silences it (the argument above), while
    #: the free-form arm keeps it, because there the noise is part of a transition kernel the
    #: off-policy arm being compared against also runs under. See each env cfg's
    #: ``__post_init__``.

    @configclass
    class FmCondCfg(ObsGroup):
        """The generator's conditioning for the live plan, as it was BUILT.

        History tokens + exteroception tokens + command tokens, flattened. The adjuster
        re-encodes these to borrow the generator's frozen prefix, so they have to be the
        tensors that produced the plan -- read from the command's stash rather than rebuilt,
        because a rebuild would sample a different instant and a different camera lag.
        """

        fm_cond: ObsTerm = ObsTerm(func=mdp.fm_plan_conditioning,
                                   params={"command_name": "traj_ref"})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class FmPlanCfg(ObsGroup):
        """The live plan and its final-step flow field, standardized.

        Two views of each node: what the generator predicted, and which way the flow was still
        moving there -- near ``tau = 1`` a statement about how far the drawn sample sits from
        its conditional mode, which is the quantity 20-35% of outcomes turn on.
        """

        fm_plan: ObsTerm = ObsTerm(func=mdp.fm_plan_window,
                                   params={"command_name": "traj_ref"})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """DECLARED, not inherited-and-nulled, because almost every term changed.

        Four principles, each of which removed channels rather than added them:

        1. **It is V(s), not Q(s, a).** ``window_residual`` -- 2728 channels, 60% of the old
           group -- WAS the action. A critic that sees the sampled residual is fitting an
           action-value while PPO's advantage arithmetic assumes a state-value, so the
           baseline absorbs variance it should be exposing.
        2. **Sample count is small.** One transition per re-plan, not per control step, so
           this group trains on ~1/12 the data a per-step critic would. 4519 channels against
           that is a width the data cannot support; this is ~1635.
        3. **The terrain representation should reach as far as the PLAN does.** The 176-cell
           uniform grid sees ~1 m while the chunk covers 1.24 s of travel. Replaced by a
           forward ray fan to 3.5 m plus a dense patch under the feet -- 317 channels for
           3.5x the reach.
        4. **Absolute yaw is noise.** ``root_quat`` is dropped; ``projected_gravity`` already
           carries roll and pitch, and nothing here should depend on which way the tile faces.

        The proprioception keeps its DENSE 10-step history: the critic wants the current state
        and how it got there. The decimated layout in this stack is the FUTURE window
        (``REFERENCE_OFFSETS``), which ``reference_window`` keeps unchanged.
        """

        # -- current state + history, dense, exactly the actor's frames ------------------
        base_lin_vel: ObsTerm = ObsTerm(func=mdp.base_lin_vel, scale=1.0,
                               history_length=PROPRIO_HISTORY_LENGTH)
        base_ang_vel: ObsTerm = ObsTerm(func=mdp.base_ang_vel, scale=1.0,
                               history_length=PROPRIO_HISTORY_LENGTH)
        projected_gravity: ObsTerm = ObsTerm(func=mdp.projected_gravity,
                                    history_length=PROPRIO_HISTORY_LENGTH)
        joint_pos: ObsTerm = ObsTerm(func=mdp.joint_pos_rel,
                            history_length=PROPRIO_HISTORY_LENGTH)
        joint_vel: ObsTerm = ObsTerm(func=mdp.joint_vel_rel, scale=0.05,
                            history_length=PROPRIO_HISTORY_LENGTH)
        # LAST FRAME ONLY. These are the frozen tracker's joint targets, not the adjuster's
        # action, so they are state -- but ten frames of them is 290 channels restating what
        # joint_pos already carries.
        actions: ObsTerm = ObsTerm(func=mdp.last_action, clip=(-20.0, 20.0))

        # -- the plan being corrected, and the command ----------------------------------
        reference_window: ObsTerm = ObsTerm(func=mdp.reference_window, params=_REF_WINDOW_PARAMS,
                                   clip=(-20.0, 20.0), scale=_REF_SCALE)
        command_window: ObsTerm = ObsTerm(func=mdp.command_window, params=_REF_PARAMS)

        # -- terrain, privileged, in the base YAW frame ---------------------------------
        terrain_rays: ObsTerm = ObsTerm(func=mdp.terrain_rays)
        terrain_base_patch: ObsTerm = ObsTerm(func=mdp.terrain_base_patch)

        # -- the goal ---------------------------------------------------------------------
        # KEPT, and it is not optional: arrival is a bootstrapped truncation here, so a critic
        # that cannot see the goal or the deadline is asked to predict a cut it has no
        # evidence for. Four channels.
        goal_geometry: ObsTerm = ObsTerm(
            func=mdp.goal_geometry,
            params={"command_name": "traj_ref", "slack_s": GOAL_SLACK_S,
                    "step_scale": 1.0 / (EPISODE_LENGTH_S / CONTROL_DT),
                    "offset_scale": GOAL_OFFSET_SCALE},
        )

        def __post_init__(self) -> None:
            """Clean inputs: this group is privileged and never deployed."""
            self.enable_corruption = False
            self.concatenate_terms = True

    fm_cond: FmCondCfg = FmCondCfg()
    fm_plan: FmPlanCfg = FmPlanCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class G1AdjusterTerminationCfg:
    """Four endings, and which flag each carries is the reward design.

    ``time_out=True`` bootstraps ``gamma * V(s)`` onto the return; ``False`` does not. So:

    * ``goal_reached`` -- TRUNCATION. The only ending that collects the future value, which is
      what makes arriving worth more than running the clock out with no bonus anywhere.
    * ``clip_end`` / ``time_out`` -- TERMINATIONS, though they are time limits. Legitimate only
      because the critic can SEE the deadline (``goal_geometry``); without that channel this is
      the classic time-limit bootstrapping error.
    * ``tracking_violation`` -- TERMINATION. A failure, so no bootstrap.

    ``mdp.base_fallen`` is deliberately absent. The reference-relative check replaces it, and
    the swap is measured rather than argued: at the adopted bounds 13 of 13 falls also
    violated, 0 of 37 loose successes were demoted, and the violation LEADS the fall by a
    median 0.20 s -- so it catches every fall the absolute test did, earlier, and stays correct
    on multi-level terrain where an absolute height datum does not.
    """

    time_out: DoneTerm = DoneTerm(func=mdp.time_out, time_out=False)
    clip_end: DoneTerm = DoneTerm(
        func=mdp.clip_end_reached,
        params={"command_name": "traj_ref", "reserve_lookahead": True,
                "slack_s": GOAL_SLACK_S},
        time_out=False,
    )
    goal_reached: DoneTerm = DoneTerm(
        func=mdp.GoalReached,
        params={"command_name": "traj_ref", "spec_path": ""},
        # NOT a truncation any more. With `time_out=True` rsl_rl bootstraps `gamma * V(s)`
        # at arrival, and V there is a self-referential fixed point whose scale is set by
        # gamma; `G1AdjusterRewardCfg.goal_bonus` states the arrival value directly instead.
        # Every ending on this task is now a true termination, so nothing is bootstrapped.
        time_out=False,
    )
    tracking_violation: DoneTerm = DoneTerm(
        func=mdp.TrackingViolation,
        params={"command_name": "traj_ref", "spec_path": ""},
        time_out=False,
    )


@configclass
class G1AdjusterRewardCfg:
    """One scalar per 0.24 s decision: dense trackability, two residual penalties, two endings.

    The env folds the dense term at the CONTROL rate and reduces it to the interval MEAN, so
    the policy sees exactly one number per action -- which is all it can act on, since the
    robot rolls forward through the interval whatever it does. A mean rather than a sum because
    ``step_reward`` is bounded in ``(0, 1]`` by construction and a mean keeps that, where a sum
    would scale with the cadence.

    The weights are a STARTING POINT and the first run's job is to calibrate them from the
    ``Episode_Reward/*`` logs, not to confirm them. What the relative sizes encode: a full
    interval of perfect tracking is worth 1.0, a failure costs 10 intervals of it, a timeout
    costs 2, and the residual penalties are set so the nominal authority (~0.13 sigma RMS,
    mean-square ~0.017) costs a few percent of a good interval rather than dominating it.
    """

    trackability: RewTerm = RewTerm(func=mdp.served_track_reward, weight=1.0)
    # -0.5 and -0.25, down 4x from -2.0/-1.0. MEASURED on the first steered run: the two
    # penalties together came to -0.305 against a trackability signal of +0.306 -- they
    # cancelled the ENTIRE dense reward, leaving only the sparse failure term, so the
    # optimum of the reward as written was "emit no residual at all". That was true at the
    # initial sigma too (~83%), so it is a mis-scaling and not just the sigma inflation.
    #
    # The residual is already bounded HARD -- 0.2 m on position, 15 deg on orientation, in
    # `apply_plan_residual` -- so this L2 was only ever meant as a soft anchor on top of a
    # real cap. At a quarter weight it is ~21% of the trackability signal: still a pull
    # toward small corrections, no longer the dominant term.
    #
    # These scale as sigma^2, so they are coupled to `init_std`: raising it 0.15 -> 0.30
    # multiplies them by ~3.3 and would re-create the cancellation. Move the two together.
    # MEASURED again on v6, and the weight came down a second time. `residual_l2` charges
    # `E[r^2] = mean^2 + var`, and over 840 iterations the sampled residual stayed ~96%
    # VARIANCE (see `residual_rate_l2` below for how that is read off) -- so all but a few
    # percent of this penalty is a tax on the policy's own exploration noise, which supplies
    # no gradient toward a better mean and, with `entropy_coef = 0`, is the only pressure
    # left on sigma. It grew from -0.043 to -0.071 while trackability fell 0.460 -> 0.416 and
    # `goal_reached` moved +0.036, i.e. the run spent its reward budget on the tax rather
    # than the task. The residual is already bounded HARD in `apply_plan_residual`, so this
    # is a soft anchor on top of a real cap and does not need to be large.
    residual_l2: RewTerm = RewTerm(
        func=mdp.plan_residual_l2, weight=-0.15, params={"command_name": "traj_ref"}
    )
    # OFF, and it is a measured duplicate rather than a judgement call. For a residual that
    # is i.i.d. across re-plans, `E[(dr)^2] = 2 Var`, so at the old weights (-0.5 level,
    # -0.25 rate) the two logged rewards are EQUAL -- and over the whole of v6 the ratio sat
    # at 0.964-0.981, never leaving that regime. It was therefore charging the same
    # exploration noise `residual_l2` already charges, for half the penalty budget and
    # essentially no independent gradient. It only starts to say something different once
    # the learned mean becomes temporally structured, which is the regime it was written for
    # and which no run has reached.
    #
    # Kept at weight 0 rather than deleted: at 0 the TERM still runs, so
    # `Episode_Reward/residual_rate_l2` keeps reporting 0 and the ratio above is no longer
    # recoverable from it -- which is why `Diagnostics/residual_*` in `AdjusterEnv.step`
    # now computes the mean/variance split directly. Re-enable by giving it a weight.
    residual_rate_l2: RewTerm = RewTerm(
        func=mdp.plan_residual_rate_l2, weight=0.0, params={"command_name": "traj_ref"}
    )
    # EXPLICIT, replacing the bootstrap. `goal_reached` used to be a truncation, so rsl_rl
    # added `gamma * V(s_goal)` -- but V at a goal-adjacent state is defined only through
    # ITSELF, with fixed point `r / (1 - gamma)`, so the arrival value was ~100 decisions of
    # trackability at gamma 0.99 and its scale was an artefact of gamma rather than a
    # statement about the task. A terminal reward says what arriving is worth.
    #
    # Sized from the measured per-decision trackability (~0.45-0.50): 8.0 is ~16-18 decisions
    # of it, so arriving beats dawdling to the deadline (which at gamma 0.96 accumulates
    # ~0.45 * (1 - 0.96^20) / 0.04 ~= 6.1 and then takes the timeout penalty with no
    # bootstrap) without swamping the dense signal that shapes HOW it arrives.
    goal_bonus: RewTerm = RewTerm(
        func=mdp.episode_ended_by, weight=8.0,
        params={"term_keys": ["goal_reached"]},
    )
    failed: RewTerm = RewTerm(
        func=mdp.episode_ended_by, weight=-10.0,
        params={"term_keys": ["tracking_violation"]},
    )
    ran_out_of_time: RewTerm = RewTerm(
        func=mdp.episode_ended_by, weight=-2.0,
        params={"term_keys": ["clip_end", "time_out"]},
    )


@configclass
class G1AdjusterEnvCfg(G1TeacherTerrainDepthEnvCfg, AdjusterEnvCfg):
    """The adjuster task.

    Multiple inheritance rather than duplication: the depth teacher supplies the scene, the
    cameras and the depth observation terms, while :class:`AdjusterEnvCfg` supplies the four
    fields the env CLASS reads (the frozen tracker's run, the two spec paths and the command
    name). Neither knows about the other.

    Three paths have no defaults and every one of them is refused rather than guessed:
    ``--clip`` (the library and its terrain), ``--fm_checkpoint`` (the generator),
    ``--tracker_run`` (the policy that drives the robot), plus the two fitted specs. A default
    path to an artifact is a thing this repo has watched rot silently more than once.
    """

    # DECLARED at class level, because the base chain MUTATES both: the depth teacher edits
    # observation terms by name (it nulls the height scan and installs the generator's own
    # depth group), and assigning a fresh group set afterwards would discard that.
    commands: G1FmTrackerCommandsCfg = G1FmTrackerCommandsCfg()
    observations: G1AdjusterObservationsCfg = G1AdjusterObservationsCfg()
    terminations: G1AdjusterTerminationCfg = G1AdjusterTerminationCfg()

    # `rewards` is deliberately NOT declared here, and the reason is a real failure: the
    # teacher's own `__post_init__` reaches into `self.rewards.undesired_contacts` by NAME to
    # override its declared weight, so a reward cfg without that term makes the base chain
    # raise before this class's post-init ever runs. Leaving it inherited means the teacher
    # pokes the teacher's set, the tracker base then clears it, and the adjuster's is installed
    # below -- each stage operating on a set that has what it expects.

    #: The fitted goal regions, wired into the goal termination by ``__post_init__``.
    goal_regions_path: str = ""

    def __post_init__(self) -> None:
        """Run the depth teacher's setup, then install the adjuster's own MDP."""
        super().__post_init__()
        self.rewards = G1AdjusterRewardCfg()

        # Silence the tracker's reference-window jitter. Moved here from the observations cfg
        # because it is a statement about THIS task's MDP: the learned correction reached
        # ~0.006 m of base position after 840 iterations, an order of magnitude under the
        # +-0.06 m being added to the channel it writes into. The free-form arm deliberately
        # does NOT do this -- see `G1AdjusterFreeformEnvCfg`.
        self.observations.reference.reference_window.noise = None

        # JOINT tracking mode. The depth teacher is a BODY-tracking env and
        # ``FmTrackingCommand`` refuses body mode: the body CLF grades 13 ``@root`` links the
        # generator does not predict, so under ``strict_fm_only`` every one of them is NaN.
        # Nothing is lost -- no reward here reads the CLF, and the actor's groups are
        # mode-independent -- so a body-mode depth tracker still loads and behaves as trained.
        # Safe after ``super()``: ``_apply_tracking_mode`` is a validation only.
        self.commands.traj_ref.tracking_mode = "joint"

        # The episode has to outlast the clip plus the slack, or `time_out` becomes the
        # binding deadline and the slack buys nothing.
        self.episode_length_s = EPISODE_LENGTH_S

        # MEASURED history, which is a decision about what the adjuster can see.
        #
        # The generator's prefix IS the adjuster's state encoder, so what the prefix encodes is
        # what the policy can condition on. Under `generator` the history rows are the model's
        # own past output -- a plan, not a robot -- and an adjuster asked to correct for where
        # the robot actually IS would have no channel carrying it. Under `measured` 38 of the
        # 44 dims are the robot's achieved state (only base position and linear velocity fall
        # back to the prediction, because no robot can supply them).
        #
        # A second, quieter benefit: the residual then does NOT compound into the generator's
        # conditioning, so the adjuster's effect stays attributable rather than entangled with
        # its own history.
        #
        # This must MATCH the checkpoint's own `history_source`; `assert_history_source_matches`
        # refuses a disagreement, which is what makes a reference-trained generator a loud
        # error here rather than a silent arm swap. Override on the CLI for that control arm:
        # `env.commands.traj_ref.history_from=generator`.
        self.commands.traj_ref.history_from = "measured"

        # EVERY episode opens at the clip's own initial condition. Two reasons, and the
        # first is about the generator rather than about the task:
        #
        # 1. The FM's first inference of an episode runs under the learned NULL-HISTORY
        #    embedding -- there is no history yet -- and that embedding is a far better
        #    prior on FLAT ground than on terrain. A random start phase drops the robot
        #    mid-motion, so its opening plan is regularly a null-history inference of a
        #    stair or a box, which is the generator's weakest configuration. Clip step 0 is
        #    flat by construction.
        # 2. It removes the spawn-inside-goal artifact outright. The goal anchor sits at
        #    `clip_len - 51` while `sample_start_step` only guaranteed `clip_len - start >=
        #    100` control steps, so the latest legal start left ~1.0 s of clip travel to
        #    reach it against a 1.0 m `approach_margin_m`. MEASURED over 493 episodes:
        #    31.3% of goal-reached episodes ended within ONE agent step -- a third of the
        #    positive signal collected for spawning, and since arrival is the BOOTSTRAPPED
        #    truncation it taught the critic that spawn-like states carry a terminal value.
        #    Raising `random_start_min_remaining_s` to 4.0 took that to 4.6%; starting at 0
        #    takes it to none, and costs no clip coverage (every clip is >= 201 steps).
        #
        # The cost is start-state diversity, which the reference-window noise and the FM's
        # own sampling noise already supply in the dimension that matters here.
        # TODO(adjuster-start-phase): revisit once the residual trains -- a curriculum that
        # re-introduces mid-clip starts would broaden the state distribution, and by then
        # the null-history-on-terrain weakness may be one the adjuster can correct FOR.
        self.commands.traj_ref.random_start_phase = False

        # STAND, SETTLE, THEN GO. The clip's first frame is a running motion, and the
        # generator's first inference -- made under the null-history embedding, its weakest
        # configuration -- would be made from it. No robot ever starts that way. So spawn
        # standing and hold the command at zero long enough to cover the first re-plan, which
        # lets the tracker settle before the first plan it must follow means anything.
        #
        # TODO(adjuster-clip-free-start): once these two are on, the ONLY thing the clip still
        # supplies at reset is the velocity command and a place to stand. That is an argument
        # for driving this task from a free-form command instead of a clip library -- see the
        # clip-free terrain work on worktree-fm-reward-finetune. Not done here: the spawn pose
        # still has to come from somewhere, and the clip is what knows where a tile's
        # traversal begins.
        self.commands.traj_ref.stand_start = True
        self.commands.traj_ref.cmd_hold_s = STAND_SETTLE_S
        # ...and tell the generator the robot has been standing, rather than leaving it the
        # phase-agnostic null embedding, which cannot express "both feet planted" and
        # produces a mid-gait opening plan with the feet off the ground. Chosen by eye on the
        # play viz -- the opening plan looks right with it on.
        #
        # TODO(adjuster-zero-command-drift): the seed fixes the POSE at the opening but
        # exposes a second, separate defect it cannot fix. MEASURED on the bare generator,
        # predicted forward travel over a 62-node chunk under a zero command:
        #
        #     null history (no seed)      +0.004 m      <- looks fine, but only because the
        #     standing seed              -0.100 m          null embedding is phase-agnostic
        #     full history, mid-episode  -0.094 m      <- the SAME bias, all episode long
        #     normal command             +0.909 m      <- never backwards, 0 of 1152
        #
        # So the generator drifts ~9 cm per chunk BACKWARDS under a zero command whenever it
        # has real history, at any point in the episode; the seed simply makes the opening
        # consistent with the rest instead of masking it. Likely cause, from a second
        # direction: stand commands are 16.4% of the FLAT clips and 0.00% of every terrain
        # family, so "hold still with stairs in view" is a combination the checkpoint was
        # never trained on -- which is why the same thing works cleanly in MuJoCo, where the
        # stand happens on a purpose-built flat lane with no terrain in the cameras.
        #
        # Not fixable from this side: it is the frozen generator's behaviour. The routes are
        # (a) let the adjuster correct it -- systematic and identifiable, the best case for a
        # residual, though 0.2 m of authority against it; or (b) put stands-on-terrain into
        # the training distribution, which the clip-free redesign would make natural.
        self.commands.traj_ref.seed_history_from_stand = True

        # Steering, on the ENV cfg so the play variant inherits it: the two must agree, or
        # what is watched is not what is trained.
        self.steer_kp = STEER_KP

        # The two privileged terrain sensors, added to the inherited scene. Both are
        # `ray_alignment="yaw"`, which is what puts them in the base's YAW frame: under the
        # full orientation the whole pattern would pitch with the torso and a lean would read
        # as terrain moving. Both cast against the importer's mesh at /World/ground.
        self.scene.terrain_base_patch = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Geometry/pelvis_link",
            # 0.2 m AHEAD, and high enough that a ray starting inside a riser still casts.
            offset=RayCasterCfg.OffsetCfg(pos=(0.2, 0.0, 20.0)),
            ray_alignment="yaw",
            # 1.2 x 0.8 m at 0.1 m. IsaacLab's grid is endpoint-INCLUSIVE, so this is 13 x 9
            # = 117 rays, not 12 x 8 = 96 -- the same convention that makes the stock
            # [1.5, 1.0] scan 176 rather than 150.
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.2, 0.8]),
            mesh_prim_paths=["/World/ground"],
            debug_vis=False,
        )
        self.scene.terrain_rays = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Geometry/pelvis_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=RayFanPatternCfg(),
            mesh_prim_paths=["/World/ground"],
            debug_vis=False,
        )

        # Capture the generator's conditioning and its final-step flow field: the adjuster's
        # two actor observation groups ARE those. Set on the command cfg rather than called on
        # the env, because the observation manager probes every term's shape while the managers
        # are still being built -- see FmTrackingCommandCfg.stash_conditioning.
        self.commands.traj_ref.stash_conditioning = True
        self.commands.traj_ref.stash_flow = True

        # The spec paths are NOT fanned out to the termination terms here, deliberately. Each
        # term falls back to the env cfg's field when its own `spec_path` param is empty, so
        # there is still exactly one statement of each path -- and no ordering hazard: hydra
        # applies CLI overrides to the already-composed cfg, so a fan-out in this method would
        # copy the DEFAULT and a `--tracking_spec` flag would never reach the term. Found by
        # `test_each_spec_path_is_stated_once_and_fanned_out`.


@configclass
class G1AdjusterEnvCfgPlay(G1AdjusterEnvCfg):
    """Watch the adjuster: few envs, both terrain sensors DRAWN, no randomisation.

    Exists to make the ray casts visible. ``debug_vis`` on a ``RayCaster`` renders one marker
    per hit, so at training env counts it is 300k markers and unusable; here it is a handful.

    Observation corruption is off so what is drawn is what the policy is fed, and the episode
    still starts at the clip's initial condition, so a staircase is approached the way
    training approaches it.
    """

    def __post_init__(self) -> None:
        """Shrink the scene and turn the terrain sensors' markers on."""
        super().__post_init__()
        # ONE replica row, so the library's tiles appear ONCE each. `num_rows` is an env-
        # CAPACITY knob -- rows are pure replicas of the whole tile set -- so the training
        # value of 10 turns an 8-tile library into 80 tiles stretching off into the distance.
        # Every other play cfg here does the same (see G1TeacherTerrainEnvCfgPlay).
        self.scene.terrain.num_rows = 1
        # One env per tile, so each terrain is shown exactly once and nothing is hidden
        # behind a robot standing on top of it.
        self.scene.num_envs = 8
        # NO DECORATION. Walls ring the RENDERED FLOOR rather than the manifest tile, and at
        # a small tile count the floors are close enough together that one tile's walls stand
        # on its neighbour -- which is confusing to look at and hides the ray casts this cfg
        # exists to show. All-or-nothing is the only granularity available: the wall knobs
        # (`wall_frac`, `wall_clearance_m`, ...) live in `DecorateConfig`, which ran OFFLINE
        # when the library was decorated, so they are baked into the .h5.
        #
        # WHAT THIS CHANGES, and it is not nothing: decoration is invisible to physics and to
        # the terrain sensors here (both cast against /World/ground alone), but the DEPTH
        # CAMERAS do see it, and they feed the generator. So the plans drawn in this view are
        # from a slightly different conditioning than training's. Fine for inspecting sensor
        # geometry; not a configuration to draw behavioural conclusions from. Turn it back on
        # with `env.scene.terrain.render_decor=True`.
        self.scene.terrain.render_decor = False
        self.observations.policy.enable_corruption = False
        # The command term's own ribbons: the FM's predicted world trajectory (which is what
        # the adjuster is correcting) and the goal region it is aiming at. `debug_vis` is the
        # master switch; the two sub-flags below select what it draws.
        self.commands.traj_ref.debug_vis = True
        self.commands.traj_ref.debug_vis_show_fm_prediction = True
        # A PATH, not a bool: the regions are fitted per family and live in the same JSON the
        # GoalReached termination reads, so the drawing and the termination cannot disagree
        # about where the goal is.
        self.commands.traj_ref.debug_vis_goal_regions = self.goal_regions_path
        # THE POINT of this cfg. Both are False in training: markers are per-hit, so 317 of
        # them per env would be 300k+ at 1024 envs.
        self.scene.terrain_rays.debug_vis = True
        self.scene.terrain_base_patch.debug_vis = True


##
# ==========================================================================================
# THE FREE-FORM ARM: the same adjuster, on the MDP the off-policy fine-tune is measured on
# ==========================================================================================
##
#
# Everything above trains on the CLIP terrain teacher, with a tracking-quality reward and
# clip-derived terminations. That is a different MDP from the one `awr` / `fbc` / `arrived` /
# `bc` are graded on, so a PPO number measured there answers a different question than the
# ladder does.
#
# This subclass exists so the two can be compared. It composes `G1FinetuneFreeformEnvCfg` --
# the EXACT task `configs/finetune/boxes.yaml` names -- with the four env-class fields a frozen
# tracker needs, and installs the online twin of the offline progress reward. What is left
# different is the ACTION SPACE, which is the point of the comparison: the off-policy arms
# re-fit the generator's own weights, so they can only move the reference inside its sampling
# support; this one adds a bounded residual to its output, which can leave that support.
#
# ONE deliberate departure, recorded here because it is the only one: the three generator-side
# corruptions `run_finetune_loop.sh` applies at COLLECT time -- `fm_perturb_fraction`,
# `fm_upper_camera_noise_fraction` and `fm_speed_mismatch_fraction` -- are left at their cfg
# defaults of 0. The first exists to manufacture the behavioural spread AWR reweights, which is
# a job PPO's own Gaussian does. The teacher's world DR (`events`: actuator gains, mass,
# friction, pushes, camera extrinsics, depth noise) and the tracker's observation corruption are
# UNTOUCHED and identical to the collect. None of the three affects scoring: they are
# collect-only overrides and `run_quick_eval.sh` applies none of them.


@configclass
class G1ProgressRewardCfg:
    """The offline progress reward, computed live, and nothing else.

    Three terms at weight 1.0 each, so ``Episode_Reward/*`` reports exactly the three columns
    the offline sidecar stores and a training log can be diffed against a graded rollout
    directly. The weights are NOT free parameters here -- they are 1.0 because the reward being
    reproduced has no weights, and moving one would make this a different reward from the one
    the arms being compared against were graded by.

    Every term the clip-terrain arm carries is gone, and each for a reason rather than by
    omission:

    * ``trackability`` graded how well the tracker followed the plan. The off-policy arms are
      not graded on that at all -- their reward is geometric plus progress -- so keeping it
      would be the largest single difference between the two MDPs.
    * ``residual_l2`` / ``residual_rate_l2`` are a penalty on the ACTION. The off-policy arms
      have no action to penalise, and the residual is already bounded hard in
      ``apply_plan_residual`` (0.20 m position, 15 deg tilt), so this was a soft anchor on top
      of a real cap rather than part of the task.
    * ``goal_bonus`` / ``failed`` / ``ran_out_of_time`` are replaced by the progress reward's
      own terminal term, which is gated by the episode's worst chunk -- so arrival is paid for
      only in proportion to how cleanly the whole traverse was planned. ``ran_out_of_time`` has
      no counterpart at all: the offline reward pays nothing for running the clock out, and its
      tail convention continues at the last ongoing reward rather than charging a penalty.
    """

    progress_terrain: RewTerm = RewTerm(
        func=mdp.progress_terrain, weight=1.0, params={"command_name": "traj_ref"})
    progress_velocity: RewTerm = RewTerm(
        func=mdp.progress_velocity, weight=1.0, params={"command_name": "traj_ref"})
    progress_success: RewTerm = RewTerm(
        func=mdp.progress_success, weight=1.0, params={"command_name": "traj_ref"})


@configclass
class G1AdjusterFreeformObservationsCfg(G1AdjusterObservationsCfg):
    """The adjuster's groups, with the free-form task's global-drift channel added back.

    ``G1AdjusterObservationsCfg`` dropped ``tracking_error_global`` when it narrowed the critic
    (it was a clip-datum channel, and the adjuster had no clip to measure against). The
    free-form base REQUIRES it -- ``G1FinetuneFreeformEnvCfg.__post_init__`` re-points its
    ``datum`` by name and is an ``AttributeError`` against a group that lacks it.

    Adding it back is not merely appeasing that line, and it is why this is a subclass rather
    than a change to the parent. Under ``history_from="generator"`` -- which this arm is forced
    into, the parent generator being a reference-history checkpoint -- the history rows in
    ``fm_cond`` are the model's own past output, so the ACTOR has no channel carrying how far
    the tracker has fallen behind the plan. Plan-anchor drift is exactly that quantity, it
    re-zeros at every re-plan so it is bounded by ``K * dt * speed`` and stays O(1), and giving
    it to the critic is the cheapest recovery of what the history arm costs. Six channels.

    A subclass so the CLIP-terrain arm's critic width is untouched: its checkpoints resume.
    """

    @configclass
    class CriticCfg(G1AdjusterObservationsCfg.CriticCfg):
        """The adjuster's critic group plus the plan-anchor drift."""

        #: Declared with the free-form datum already set, so the base's re-point is a no-op
        #: restatement rather than the only thing that makes it correct.
        tracking_error_global: ObsTerm = ObsTerm(
            func=mdp.tracking_error_global,
            params={"command_name": "traj_ref", "datum": "plan_anchor"})

    critic: CriticCfg = CriticCfg()


@configclass
class G1AdjusterFreeformEnvCfg(G1FinetuneFreeformEnvCfg, AdjusterEnvCfg):
    """The adjuster on the free-form fine-tune MDP -- the PPO arm of the algorithm ladder.

    Base order is load-bearing. ``G1FinetuneFreeformEnvCfg`` supplies the whole MDP (the
    manifest-only terrain, the section spawn, the steered free-form command, the five
    terminations, the world DR); ``AdjusterEnvCfg`` supplies only the four fields
    :class:`AdjusterEnv` reads off the cfg and defines no ``__post_init__``, so it goes last and
    contributes nothing to the composition order.

    **Almost nothing is declared at class level, and that is deliberate.** ``commands``,
    ``terminations``, ``events`` and ``rewards`` are all INHERITED: declaring a second
    ``commands`` (as the clip-terrain arm does, importing the tracker's) would replace the
    object ``G1FinetuneFreeformEnvCfg.__post_init__`` mutates, and ``reference_source``,
    ``seed_history_from_clip`` and ``random_start_before_terrain_start`` would all silently
    revert to the clip task's values. ``rewards`` is installed in ``__post_init__`` rather than
    declared for the reason the clip arm documents: the teacher's own ``__post_init__`` reaches
    ``self.rewards.undesired_contacts`` by name while it runs.

    **What the clip-terrain arm sets and this one must NOT**, each checked against a
    constructed cfg rather than read off the class:

    ======================================  ==========================================
    the clip arm does                        why it is wrong here
    ======================================  ==========================================
    ``history_from = "measured"``            FATAL, not a preference. The ladder's parent
                                             is a ``reference``-history checkpoint and
                                             ``assert_history_source_matches`` refuses the
                                             pair at generator load.
    ``stand_start`` / ``cmd_hold_s``         Read by the CLIP reset, which is ``None`` here.
                                             ``freeform_cmd_hold_s`` (0.5 s) is the real hold
                                             and the section spawn already stands the robot up.
    ``seed_history_from_stand``              NOT inert -- ``_snapshot_and_infer`` seeds
                                             regardless of the command source, so it would
                                             change the opening plan. The off-policy collect
                                             opens on the learned null-history embedding.
    ``tracking_mode`` / ``random_start_phase``  Already set by the free-form base, with reasons.
    ``steer_kp = 1.5``                       The free-form command steers itself
                                             (``freeform_steer_kp``). The adjuster's installer
                                             also scans for a ``mdp.GoalReached`` INSTANCE,
                                             which this task does not have -- it uses the
                                             ``freeform_goal_reached`` function -- so leaving
                                             it on raises at construction.
    ``episode_length_s = 24.0``              The collect runs 40 s.
    silencing ``reference_window`` noise     It is part of the transition kernel both arms run
                                             under, so removing it for one makes the MDPs differ.
    ======================================  ==========================================
    """

    #: The critic keeps the adjuster's widened group plus the plan-anchor drift; the two actor
    #: groups and the tracker's two come with it. Declared because the base chain MUTATES
    #: observation terms by name.
    observations: G1AdjusterFreeformObservationsCfg = G1AdjusterFreeformObservationsCfg()

    #: Fitted goal regions. Unused by the terminations on this task -- the free-form goal is
    #: declared by the manifest route, not fitted from clips -- and kept only so
    #: ``--goal_regions`` composes and the debug viz can draw them.
    goal_regions_path: str = ""

    def __post_init__(self) -> None:
        """Compose the free-form fine-tune, then make it an adjuster task."""
        super().__post_init__()
        self.rewards = G1ProgressRewardCfg()

        # The collect's budget. The free-form base leaves the teacher's 20 s, which would cut
        # the slow x far cells before the goal band could fire and convert an arrival into a
        # time-out.
        self.episode_length_s = FREEFORM_EPISODE_LENGTH_S
        # ...and the collect's env count, which is where it actually binds (`--num_envs` still
        # wins). It is not part of the MDP -- it changes throughput, and at EVAL it sets the
        # censored fraction, which is why `run_quick_eval.sh` insists the arms match there.
        self.scene.num_envs = FREEFORM_NUM_ENVS

        # FULL-WIDTH action: one knot per plan node, so `knot_basis` is the identity and the
        # residual is the whole 62 x 44 window. Must match the actor's `num_knots`.
        self.num_knots = FREEFORM_NUM_KNOTS

        # OFF, and it must be set rather than left: `AdjusterEnv._install_steering` refuses a
        # non-zero gain unless a `mdp.GoalReached` term is active, and this task steers through
        # the command's own pure-pursuit controller instead.
        self.steer_kp = 0.0

        # The generator-side collect corruptions, OFF and stated rather than left at their
        # defaults. The two inner fractions already default to 0, so this gate changes no
        # behaviour -- but it defaults to 1.0, which READS as "perturbation on", and this is the
        # one place a reader looks to find out whether this arm carries them. It does not: see
        # the block comment above this class for why the perturbation arm in particular is the
        # off-policy loop's exploration device rather than part of the task.
        self.commands.traj_ref.fm_perturb_fraction = 0.0

        # The generator's conditioning and its final-step flow field ARE the actor's two
        # observation groups. Set on the command cfg rather than called on the env, because the
        # observation manager probes every term's shape while the managers are still building.
        self.commands.traj_ref.stash_conditioning = True
        self.commands.traj_ref.stash_flow = True

        # The two privileged terrain sensors the critic reads. `ray_alignment="yaw"` puts both
        # in the base's YAW frame: under the full orientation the pattern would pitch with the
        # torso and a lean would read as terrain moving.
        self.scene.terrain_base_patch = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Geometry/pelvis_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.2, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.2, 0.8]),
            mesh_prim_paths=["/World/ground"],
            debug_vis=False,
        )
        self.scene.terrain_rays = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Geometry/pelvis_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=RayFanPatternCfg(),
            mesh_prim_paths=["/World/ground"],
            debug_vis=False,
        )

        # The critic's goal channel, rescaled for THIS course. `mdp.goal_geometry` already
        # dispatches clip-free -- `goal_anchor_world` branches on `cmd.has_clips` to
        # `freeform_goal_anchor_world`, and `last_usable_step` has its own clip-free branch --
        # so only the three scalars change, and `slack_s` goes to 0 because `clip_end` is
        # disabled here and `time_out` is the only deadline.
        self.observations.critic.goal_geometry.params = {
            "command_name": "traj_ref",
            "slack_s": 0.0,
            "step_scale": 1.0 / (FREEFORM_EPISODE_LENGTH_S / CONTROL_DT),
            "offset_scale": FREEFORM_GOAL_OFFSET_SCALE,
        }


@configclass
class G1AdjusterFreeformEnvCfgPlay(G1AdjusterFreeformEnvCfg):
    """Watch the free-form adjuster: few envs, the plan DRAWN, the randomisation off.

    Everything that decides BEHAVIOUR is inherited unchanged -- the reward set is irrelevant
    here, but the command, the terminations, the speed band and both frozen networks are the
    ones the policy was trained under, so what is on screen is what was scored.

    ``render_decor`` STAYS ON, unlike the clip-terrain play cfg which turns it off for speed.
    The generator was fitted on decorated scenes and misreads a bare vertical surface as a box,
    so watching it without decoration is watching a different input distribution.
    """

    def __post_init__(self) -> None:
        """Compose the training task, then make it watchable."""
        super().__post_init__()
        self.scene.num_envs = 4
        # On the TERRAIN, not the scene: the scene cfg treats an unknown attribute as an asset
        # and refuses it by name. One replica row, so each tile of the library appears once.
        self.scene.terrain.num_rows = 1
        # DR off: the point is to see what the policy does, not what a disturbance did. The
        # tracker's observation corruption goes with it -- on a 4-env view a noise draw is a
        # confound you cannot average out.
        for group in ("policy", "reference"):
            getattr(self.observations, group).enable_corruption = False
        # The generator's served plan, the goal band, and the two privileged terrain sensors.
        self.commands.traj_ref.debug_vis = True
        self.scene.terrain_rays.debug_vis = True
        self.scene.terrain_base_patch.debug_vis = True
