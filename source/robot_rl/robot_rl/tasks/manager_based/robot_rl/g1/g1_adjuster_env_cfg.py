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

from robot_rl.tasks.manager_based.robot_rl import mdp
from robot_rl.tasks.manager_based.robot_rl.envs import AdjusterEnvCfg

from .g1_depth_env_cfg import G1TeacherTerrainDepthEnvCfg
from .g1_teacher_env_cfg import G1TeacherObservationsCfg
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
GOAL_SLACK_S: float = 3.0

#: Episode budget in seconds. The longest clip's last usable step is 948 control steps; plus
#: the slack that is 1098, so 24.0 s (1200 steps) leaves ``time_out`` a genuine backstop rather
#: than the binding constraint on the 20 s flat clips.
EPISODE_LENGTH_S: float = 24.0


@configclass
class G1AdjusterObservationsCfg:
    """Five groups: two the frozen tracker reads, two the adjuster reads, one the critic reads.

    The first two are the TRACKER's, unchanged from the teacher -- it is a frozen policy and
    its observation contract is fixed by its checkpoint, which ``FrozenTracker`` asserts.

    ``fm_cond`` and ``fm_plan`` are the adjuster's, and they are observations rather than
    something the policy recomputes because PPO's rollout runs under ``inference_mode``:
    nothing built there can enter an autograd graph later, so ``update`` re-runs the actor from
    stored observations. Both are noise-free (``enable_corruption=False``): they are the
    generator's own state, not a sensor, and the plan arrives in a DECLARED normalized space
    that an empirical normalizer would demote to a dataset statistic.
    """

    policy: G1TeacherObservationsCfg.PolicyCfg = G1TeacherObservationsCfg.PolicyCfg()
    reference: G1TeacherObservationsCfg.ReferenceCfg = G1TeacherObservationsCfg.ReferenceCfg()

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
    class CriticCfg(G1TeacherObservationsCfg.CriticCfg):
        """The teacher's privileged group, plus what a residual policy's critic needs.

        Both additions are appended LAST so a resumed critic can be zero-padded rather than
        permuted -- the mechanism ``scripts/rsl_rl/pad_critic_for_finetune.py`` exists for.

        ``window_residual`` is not optional. The group already carries the reference, which
        says how wrong the served plan is; the residual says how much of that the ADJUSTER
        caused. Without the second, the critic cannot separate "the generator was wrong" from
        "the adjuster made it worse" and eats the difference as variance -- measured to turn
        advantages into noise, inflate entropy and stall a run past a thousand iterations.

        ``goal_geometry`` is what makes the bootstrapped goal truncation well posed; see the
        module docstring.
        """

        window_residual: ObsTerm = ObsTerm(func=mdp.window_residual,
                                           params={"command_name": "traj_ref"})
        goal_geometry: ObsTerm = ObsTerm(func=mdp.goal_geometry,
                                         params={"command_name": "traj_ref"})

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
        time_out=True,
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
    residual_l2: RewTerm = RewTerm(
        func=mdp.plan_residual_l2, weight=-2.0, params={"command_name": "traj_ref"}
    )
    residual_rate_l2: RewTerm = RewTerm(
        func=mdp.plan_residual_rate_l2, weight=-1.0, params={"command_name": "traj_ref"}
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

        # The spec paths are NOT fanned out to the termination terms here, deliberately. Each
        # term falls back to the env cfg's field when its own `spec_path` param is empty, so
        # there is still exactly one statement of each path -- and no ordering hazard: hydra
        # applies CLI overrides to the already-composed cfg, so a fan-out in this method would
        # copy the DEFAULT and a `--tracking_spec` flag would never reach the term. Found by
        # `test_each_spec_path_is_stated_once_and_fanned_out`.
