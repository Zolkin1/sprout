# Copyright 2026 Zachary Olkin. All rights reserved.

"""A closed-loop rollout's two tables: per-re-plan WINDOWS and per-episode OUTCOMES.

``record_rollouts.py`` streams one window per generator re-plan.  Fitting a value function on
those windows needs the thing a window cannot carry: **how the episode it belongs to ended**,
which is not known until it does.  So a rollout writes two files, and this module owns both.

**Windows are stored in the GENERATOR's own space, not the clip recorder's.**  That is the
one structural decision here and it was reached by trying the other way first.
``record_data.py`` writes 22 WORLD-frame state terms per node -- root pose, twist, joints, and
four end-effector frames -- because a clip carries all of them and the dataset rebuilds a
training window from them.  A generator predicts three BLOCKS
(``base_pose, base_vel, joint_pos``) and no end effectors at all, so on an FM-driven rollout
that format cannot be filled: ``get_future_world_frames`` refuses outright, and every array
that survives is a lossy re-encoding of something the command already holds exactly.

What it holds, at the instant of a re-plan, needing no reconstruction:

===============================  ================================  ====================
quantity                         source                            shape
===============================  ================================  ====================
history tokens                   ``_history.stack_padded("node")``  ``(N, 4, D)``
exteroception actually served     ``_lagged_extero()``               ``(N, T, K)``
command over the horizon          ``get_command_velocity_horizon``   ``(N, F, 3)``
the emitted chunk                 ``_pred_window``                   ``(N, F, D)``
the plan anchor                   ``_anchor_pos`` / ``_anchor_quat``  ``(N, 3)`` / ``(N, 4)``
===============================  ================================  ====================

Those five ARE the model's input and output.  Storing them is a record of what happened;
storing the clip format would be a record of what a clip would have looked like.

**The join key is ``episode_uid``, assigned at RESET**, and that is the whole reason this
module exists rather than a few lines in the recorder:

* ``collect_windows`` already keeps an ``episode_id``, but it is a per-ENV counter that starts
  at the env index and steps by ``num_envs``; it says which episode of that env this was.
* the eval accumulator's ``episode_id`` is assigned at HARVEST, from a global counter, in
  whatever order envs happen to finish.

Neither is the other, and joining a window table on one against an outcome table on the other
lines up different episodes -- silently, and only for envs that have reset a different number
of times, which is every env after the first few hundred steps.  One counter, written into
both files, removes the class of bug rather than documenting it.

**Outcomes are folded by** :class:`eval.accumulate.EpisodeAccumulator`, unchanged.  That is
deliberate reuse and not convenience: it is where ``fell`` and the four goal columns are
DEFINED, and a second implementation of "did this episode arrive" would be free to disagree
with the one the eval harness reports.  What this module adds is the identity columns the eval
gets from its plan (which a rollout has no equivalent of) and a writer sized to what a critic
reads, rather than the eval's full 68-column contract.

**The harvest has to happen inside the reset event.**  By the time ``env.step`` returns, the
terminal state is gone -- the buffers already describe the next episode's spawn.  So this
wraps ``reset_on_clip`` the way ``eval.runner.EvalResetHook`` does, for the same reason.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (os.path.join(_HERE, "eval"), _HERE):
    if _d not in sys.path:
        sys.path.insert(0, _d)

#: Bump on any column removal, rename or change of meaning.
# 2: the per-chunk tracking-error block (`track_err`) and the strict violation columns.
SCHEMA_VERSION: int = 4

#: Schema versions the readers accept.  ONE set, because ``schema_version`` is a single
#: number stamped on both tables -- so a reader that accepts only the newest orphans the
#: episode table of an older recording as surely as the window table.
#:
#: Bumping :data:`SCHEMA_VERSION` alone makes every recording already on disk unreadable, which
#: is what happened when the perturbation columns landed: the 32k-episode v1 set the critic was
#: fitted on stopped loading.  So the reader states which versions it can read, and what it
#: substitutes for each one's missing columns.
READABLE_SCHEMAS: frozenset[int] = frozenset((2, 3, 4))

#: Columns added after schema 2, with the value to fill for an older file.  Keyed by column
#: rather than by version because the reader's job is "substitute for what is missing", and a
#: schema-2 file is missing everything a schema-3 one is missing plus more.
#:
#: **This is not the "absent is not zero" rule being broken -- it is the case that rule exists
#: to distinguish.** `penetration_volume_m3` is absent when the collider model carries no
#: areas, where a zero would assert a geometrically perfect plan; here the perturbation arms
#: did not EXIST when a schema-2 file was written, so "this row was not perturbed" is the
#: accurate statement about it rather than a guess standing in for one.
WINDOW_SCHEMA_BACKFILL: dict[str, tuple[int, str]] = {
    "perturb_null_history": (0, "u1"),
    "perturb_noisy_upper_cam": (0, "u1"),
    # Schema 4: the speed-mismatch arm. The OFFSET backfills to 0.0, which is the accurate
    # statement about a file written before the arm existed -- and it is an offset rather
    # than an absolute speed precisely so that 0 means "no mismatch" instead of "commanded
    # 0 m/s", which is a real and very different thing on this task.
    "perturb_speed_mismatch": (0, "u1"),
    "perturb_speed_offset_mps": (0, "f4"),
}

_F = "f4"
_I = "i4"
_L = "i8"
_U8 = "u1"

#: ``(name, dtype)`` per episode, in write order.
#:
#: A SUBSET of the eval's 68, chosen for what fitting a critic and computing a terminal reward
#: actually read.  The eval's tracking and effort metrics are dropped: they describe how well
#: a reference was followed, which is not what this table is for, and every one of them would
#: have to be filled or explained.
ROLLOUT_EPISODE_COLUMNS: tuple[tuple[str, str], ...] = (
    # --- identity / join ---------------------------------------------------------------
    ("episode_uid", _L),          # THE join key; matches the window table's column
    ("env_id", _I),
    ("clip_id", _I),              # index into the library the rollout ran
    ("terrain_id", _I),           # manifest tile id; -1 for flat
    ("family_code", _U8),         # indexes the file's own `family_names` attr
    ("start_step", _I),           # episodes may open mid-clip
    ("last_usable_step", _I),     # what clip_end truncates at, lookahead reserve included
    ("n_replans", _I),            # windows this episode contributed; the n-step join's bound
    # --- outcome -----------------------------------------------------------------------
    ("steps_survived", _I),
    ("terminated", _U8),          # a FAILURE term fired (not a truncation)
    ("truncated", _U8),           # clip_end / time_out
    ("censored", _U8),            # still running when the rollout hit its ceiling
    ("first_fall_step", _I),      # -1 = never crossed the fall predicate
    # The STRICT, reference-relative checks. `first_fall_step` above is reference-FREE and
    # says nothing about a robot that stumbles down a flight and barely catches itself;
    # `terminal_reward(strict=True)` reads these.
    ("first_violation_step", _I), # -1 = never; the union of the three below
    ("first_lateral_step", _I),
    ("first_height_step", _I),
    ("first_tilt_step", _I),
    ("track_reward_mean", _F),    # NaN when the checks were not turned on
    ("min_base_height_w", _F),
    ("max_tilt_rad", _F),
    # --- the SPATIAL goal (mdp.goal_region) --------------------------------------------
    ("reached_goal", _U8),
    ("first_goal_step", _I),
    ("min_dist_to_goal_m", _F),
    ("goal_downrange_m", _F),
    ("goal_lateral_m", _F),
    ("goal_height_err_m", _F),
    ("max_goal_downrange_m", _F),
    # The region's own placement, so a viewer or a plot can DRAW it without re-running the
    # sim. The anchor is the globally-placed clip's terminal base pose and the yaw is the
    # terrain's authored heading; both are episode constants, and neither is recoverable
    # from the offsets alone.
    ("goal_pos_w_x", _F), ("goal_pos_w_y", _F), ("goal_pos_w_z", _F),
    ("goal_frame_yaw", _F),
    # The base was ALREADY inside the region on the episode's first folded step, so there was
    # no task to do and the success is free. Recorded rather than silently dropped: it is a
    # property of where the episode opened, and a fit that ignores it is training a critic on
    # a large population of states labelled "success" that required nothing.
    #
    # A large fraction of arrivals were this whenever the clip's own random start phase was
    # left on, because that phase opens episodes at or past the clip's own endpoint.
    # `--start_step 0` is the default for exactly that reason.
    ("spawned_in_goal", _U8),
    # How far the CLIP itself travels from where this episode opened to the goal. Negative
    # means the goal is BEHIND the spawn, which a forward half-plane cannot express -- see
    # `mdp.goal_region.goal_travel_m`. Recorded per episode so a library nobody has measured
    # cannot hide the same defect the shipped one did.
    ("goal_travel_m", _F),
)

ROLLOUT_COLUMN_NAMES: tuple[str, ...] = tuple(n for n, _ in ROLLOUT_EPISODE_COLUMNS)


#: One row per RE-PLAN of the terrain-consistency grade, written when
#: ``record_rollouts.py --terrain_reward`` is on.
#:
#: **The columns are the offline sidecar's, deliberately.**  ``inspect_terrain_reward.py
#: --stage reward`` writes ``contact_penalty`` / ``penetration_sum_m`` per window under the
#: same ``(episode_uid, replan_index)`` key, and the critic regresses on
#: ``contact_penalty * contact_weight + penetration_sum_m * penetration_weight`` built from
#: exactly those two.  Recording the same names under the same key is what makes the live and
#: offline numbers joinable and diffable rather than merely similar -- which is the whole
#: point of :mod:`test_terrain_reward_parity`.
#:
#: **The RAW terms are stored, never the penalty and never a reward**, matching this module's
#: existing rule for the tracking error.  ``tau``, the per-body penetration weights and the
#: contact/penetration mix all stay offline knobs, so a re-weighting costs an array op
#: instead of a re-capture.
ROLLOUT_CHUNK_COLUMNS: tuple[tuple[str, str], ...] = (
    # --- identity / join; the SAME key the window table and the sidecar use ---------------
    ("episode_uid", _L),
    ("replan_index", _I),
    ("env_id", _I),
    # --- THE two terms the critic is fitted on -------------------------------------------
    ("contact_penalty", _F),
    ("penetration_sum_m", _F),
    # --- the served-window twins (first `served_nodes` nodes) ----------------------------
    # Recorded but NOT what the critic uses: the mode error lives in the tail of the plan,
    # not in the 12 nodes that get executed.  Here so that choice stays measurable.
    ("contact_penalty_served", _F),
    ("penetration_sum_m_served", _F),
    # --- alternative reductions and the vacuity guards ------------------------------------
    ("penetration_volume_m3", _F),   # area-weighted, density-invariant
    ("penetration_mean", _F),
    ("max_penetration_m", _F),
    ("contact_reward", _F),
    ("clearance", _F),
    ("support_fraction", _F),        # a plan that never plants a foot is charged nothing
    ("contact_coverage", _F),
    ("exempt_fraction", _F),
    # --- WHERE ON THE COURSE this re-plan happened --------------------------------------
    #
    # Signed downrange to the goal anchor in the TERRAIN frame at the instant of the re-plan
    # -- the same quantity, from the same function, the window table records per window.
    #
    # **This is what makes the penalty comparison position-matched rather than
    # phase-matched.** The penalty is a per-re-plan average, and a better policy survives
    # longer, so its re-plans come from further down the course where the geometry is harder;
    # an unmatched average charges it for getting further. Capping on `replan_index` corrects
    # most of that (pooled +0.39 -> -0.07 on the 30-cell grid) but matches TIME, not
    # POSITION: a policy that tracks better covers more ground per second, so residual bias
    # survives in its favour. Binning on this column compares plans made at the same PLACE.
    ("goal_downrange_m", _F),
)

ROLLOUT_CHUNK_NAMES: tuple[str, ...] = tuple(n for n, _ in ROLLOUT_CHUNK_COLUMNS)

#: Columns of :data:`ROLLOUT_CHUNK_COLUMNS` the recorder supplies ITSELF rather than reading
#: from the grader: the three join keys and the course position.
CHUNK_NON_TERM_COLUMNS: frozenset[str] = frozenset(
    ("episode_uid", "replan_index", "env_id", "goal_downrange_m"))

#: The :func:`~motion_gen.terrain_consistency.terrain_consistency` keys the chunk table reads.
#:
#: Derived by SUBTRACTION, so a new column is excluded by naming it above rather than by
#: remembering to. Getting this wrong is a KeyError on the first re-plan of a run --
#: `_terrain_chunk_batch` looks every one of these up in the grader's output -- which is loud,
#: but only after a simulator launch.
TERRAIN_CHUNK_TERMS: tuple[str, ...] = tuple(
    n for n in ROLLOUT_CHUNK_NAMES if n not in CHUNK_NON_TERM_COLUMNS)


# TODO(jump-off-front): this family fails far more than any other and it is not understood.
#
# Established, from the per-window goal offsets (clean data, independent of the contaminated
# tilt column -- see TODO(terminal-tilt)):
#   * almost all of its failures bind on DOWNRANGE at their closest approach, not on lateral;
#   * lateral offsets at closest approach sit comfortably INSIDE the fitted half-width, so a
#     yaw/width hypothesis is not supported;
#   * so the robot is not getting there, rather than getting there and being missed.
#
# NOT established: whether those failures are falls or stalls. That split reads
# `first_fall_step`, which TODO(terminal-tilt) shows can be set spuriously.
#
# Hypotheses worth separating:
#   (a) real dismount failures -- the generator genuinely cannot do the forward drop;
#   (b) the 70 deg TILT limb sitting inside the family's own envelope. This is the same shape
#       of defect the HEIGHT limb already had here: `base_fallen`'s old 0.50 m sat inside
#       jump_off_front's own reference envelope, which is why it was lowered. A forward
#       dismount pitches hard, so the tilt limb deserves the identical check -- and it cannot
#       be run until the tilt column is trustworthy;
#   (c) the goal, anchored at the clip's terminal position, sitting past a drop the robot
#       cannot reach -- an unreachable region rather than an unachievable motion.
#
# Cheapest next step: draw a dozen of them in Isaac with the goal region visible
# (`--viz kit`, `env.commands.traj_ref.debug_vis_goal_regions=...`) and watch where they
# stop. That is how the goal-region defects were found, and it costs minutes.
#
# Numbers deliberately not inlined: they are a property of one pair on one library and would
# go stale faster than this comment. Re-derive from a rollout's own `episodes.h5`.
# NOTE: moved here from record_rollouts.py so a second consumer can use it. That
# module parses argv at import time and therefore cannot be imported, which had
# already forced one test to AST-extract this class rather than import it.
class ProprioRing:
    """The last ``H`` control steps of proprioception, newest last.

    Kept at the CONTROL rate and not at the re-plan rate. A fall develops over tens of
    milliseconds; sampled every 0.24 s the onset is simply not in the data, and the onset is
    what the lead-time gate is about.

    Maintained here rather than read out of the observation manager's history-stacked terms,
    whose flattening to ``(N, history*dim)`` may be (T, D) or (D, T) -- an IsaacLab internal.
    A transposed history is shape-silent and partly fittable, which is the worst combination.
    """

    def __init__(self, n_envs: int, device, history: int, dim: int) -> None:
        """Allocate the ring.

        Args:
            n_envs: Environment count.
            device: Torch device.
            history: Rows kept per env.
            dim: Row width; must equal :data:`rollout_outcomes.PROPRIO_DIM`.
        """
        self.buf = torch.zeros(n_envs, history, dim, device=device)
        self.history = int(history)

    def push(self, rd) -> None:
        """Roll in this step's row for every env.

        Args:
            rd: The accumulator's converting view of ``ArticulationData`` (several of those
                buffers are warp arrays, which do not accept a torch-style index).
        """
        row = torch.cat([rd.joint_pos, rd.joint_vel, rd.root_ang_vel_b,
                         rd.projected_gravity_b], dim=-1)
        self.buf = torch.roll(self.buf, shifts=-1, dims=1)
        self.buf[:, -1] = row

    def invalidate(self, dones) -> None:
        """Zero the ring for envs that just reset, so no window spans two episodes."""
        m = dones.to(torch.bool)
        if bool(m.any()):
            self.buf[m] = 0.0


#: The bound fields of a stamped ``tracking_spec`` that decide what the STRICT columns MEAN.
#: The kernel widths and weights shape a reward, not a latch, so they are not compared.
TRACKING_BOUND_FIELDS: tuple[str, ...] = ("max_lateral_m", "max_height_m", "max_tilt_rad")


def read_tracking_spec(path: str) -> dict | None:
    """The ``tracking_spec`` an episode table was written under, or ``None``.

    ``record_rollouts`` stamps it for a stated reason -- ``first_violation_step`` and
    ``track_reward_mean`` are only interpretable against the bounds that produced them -- and
    for a long time nothing read it back.

    Args:
        path: An ``*.episodes.h5``.

    Returns:
        The spec dict, or ``None`` on a table written before it was stamped.
    """
    import json                                                            # noqa: PLC0415

    import h5py                                                           # noqa: PLC0415

    with h5py.File(path, "r") as f:
        raw = f.attrs.get("tracking_spec")
    if raw is None:
        return None
    return json.loads(raw) if isinstance(raw, (str, bytes)) else dict(raw)


def assert_tracking_bounds_agree(paths: dict[str, str]) -> dict:
    """Refuse to compare arms whose STRICT columns were latched at different bounds.

    **This is the guard that was missing when the height bound moved 0.20 -> 0.12.**  Two arms
    recorded either side of that change carry ``first_violation_step`` latched at different
    thresholds, so any rule that reads it -- and ``strict`` is the default everywhere -- puts a
    40% bound tightening into the reported delta and attributes it to the policy.  Nothing in
    the numbers looks wrong: both arms have the column, both are populated, both are plausible.

    Args:
        paths: ``{arm label: an *.episodes.h5 from that arm}``.

    Returns:
        The agreed bounds, or ``{}`` when no arm stamped a spec.

    Raises:
        SystemExit: If two arms disagree on any of :data:`TRACKING_BOUND_FIELDS`.
    """
    seen: dict[tuple, list[str]] = {}
    for label, p in paths.items():
        spec = read_tracking_spec(p)
        if spec is None:
            continue
        key = tuple(round(float(spec.get(f, float("nan"))), 9)
                    for f in TRACKING_BOUND_FIELDS)
        seen.setdefault(key, []).append(label)
    if len(seen) <= 1:
        return ({f: v for f, v in zip(TRACKING_BOUND_FIELDS, next(iter(seen)))}
                if seen else {})
    lines = [f"      {dict(zip(TRACKING_BOUND_FIELDS, k))}  <- {', '.join(v)}"
             for k, v in seen.items()]
    raise SystemExit(
        "REFUSING: these arms were recorded under DIFFERENT strict tracking bounds, so their "
        "`first_violation_step` columns do not mean the same thing:\n"
        + "\n".join(lines)
        + "\n  Every success rule except `--metric reached_goal --failure_rule loose` reads "
          "that column, so a delta across this boundary contains the bound change. Either "
          "re-record the older arm under the current bounds, or reduce both with "
          "`--metric reached_goal --failure_rule loose`, which reads only `reached_goal`.")


def first_bad_step(cols: dict[str, np.ndarray], *, strict: bool = True) -> np.ndarray:
    """The step an episode FAILED on: the fall crossing or the strict violation, first wins.

    One implementation, because two consumers need it for different purposes and they must not
    be able to disagree about when an episode ended: :func:`terminal_reward` scores the outcome
    from it, and ``fit_value``'s drop-after-latch cuts the training set on it.  A window kept
    past the step its own label was decided by is a state that termination never produces.

    ``-1`` means "never" on BOTH limbs, so they are combined by a min over the non-negative
    entries.  A plain ``np.minimum`` returns -1 the moment either limb never fired, which reads
    as "failed at step -1" and is the whole reason this is a named function.

    Args:
        cols: Episode columns, needing ``first_fall_step`` and -- for the strict rule --
            ``first_violation_step``.
        strict: Include the tracking violation.  ``False`` gives the LOOSE rule, so a number
            published before the strict checks existed stays reproducible from the same table.

    Returns:
        ``(n_episodes,)`` int64 step indices, ``-1`` where the episode never failed.
    """
    ff = np.asarray(cols["first_fall_step"], dtype=np.int64)
    if not strict or "first_violation_step" not in cols:
        return ff
    fv = np.asarray(cols["first_violation_step"], dtype=np.int64)
    big = np.iinfo(np.int64).max
    out = np.minimum(np.where(ff < 0, big, ff), np.where(fv < 0, big, fv))
    return np.where(out == big, -1, out)


def terminal_reward(cols: dict[str, np.ndarray], *, strict: bool = True) -> np.ndarray:
    """The sparse terminal reward, stated once so every consumer reads the same rule.

    Under the MDP this rollout serves: gamma = 1, reward zero everywhere but the terminal
    step, and ``V(s) = -P(fail | s)``.  So::

        r_T =  0   the base entered the goal region before it ever FAILED
        r_T = -1   otherwise -- failed first, or survived without arriving

    **"Failed" is the fall crossing OR the strict tracking violation, whichever came first**
    (``first_violation_step``; see ``transfer/deploy/tracking_error.py``).  The fall predicate
    alone is reference-free -- pelvis low, or far from world upright -- and says nothing about
    a robot that stumbles down a flight and barely catches itself, which is exactly the
    behaviour a critic fitted on this reward must not be taught to like.  ``strict=False``
    reproduces the fall-only rule, and a table written before the tracking columns existed
    takes that path automatically, so every number published under the old rule stays
    reproducible from the same data.

    The second clause is the load-bearing one.  If merely staying upright scored 0, standing
    still would score identically to climbing the staircase with none of the risk, and an
    operator that reshapes a generator toward high-scoring outcomes finds that immediately.
    Arrival is what makes the reward un-gameable by waiting.

    **Censored episodes are excluded by the caller, not scored here.**  An episode cut when
    the rollout hit its step ceiling has no outcome -- it neither fell nor failed to arrive --
    and scoring it -1 would teach the critic that whatever was happening at the ceiling
    predicts failure, which is a property of the harness.

    Args:
        cols: Columns from :func:`read_rollout_episodes`.
        strict: Count a tracking violation as a failure. Ignored when the table carries no
            ``first_violation_step``.

    Returns:
        Float array in ``{-1.0, 0.0}``, one per episode.
    """
    arrived = np.asarray(cols["reached_goal"]).astype(bool)
    fg = np.asarray(cols["first_goal_step"])
    # WHICHEVER CAME FIRST. An episode that reaches the goal and then fails in the run-out
    # did the task; one that fails and is later dragged across the line by a reset did not.
    # This is `mjeval`'s convention -- it tests the failure predicate before the target every
    # step and breaks on either -- and it is worth 17 points here: measured on a 298-episode
    # rollout, 51 episodes arrived and then fell, which a plain `arrived AND not terminated`
    # scored as failures. That is larger than any policy difference this is meant to resolve.
    ff = first_bad_step(cols, strict=strict)
    failed_first = (ff >= 0) & ((fg < 0) | (ff < fg))
    return np.where(arrived & ~failed_first, 0.0, -1.0).astype(np.float32)


class RolloutOutcomeTable:
    """Append-only per-episode accumulator, written once at the end.

    Held in memory: a long rollout is tens of thousands of episodes of 22 small columns, a
    few megabytes, so incremental appends would buy nothing and cost a torn file on an
    interrupt.
    """

    def __init__(self, family_names: list[str] | None = None) -> None:
        """Start an empty table.

        Args:
            family_names: ``family_names[code]`` for the ``family_code`` column, stored on
                the file so a reader never has to guess what a code meant.
        """
        self._rows: list[dict] = []
        self.family_names = list(family_names or ["unknown"])

    def append_batch(self, batch: dict[str, np.ndarray]) -> int:
        """Add one episode per element of ``batch``'s arrays.

        Args:
            batch: A dict whose keys are exactly :data:`ROLLOUT_COLUMN_NAMES`, each an array
                of the same length.

        Returns:
            How many episodes were added.

        Raises:
            ValueError: On a missing or unexpected key, naming both sets, or on ragged
                lengths.  Validated here rather than at write time so the failure points at
                the batch that produced it rather than at the end of a multi-hour rollout.
        """
        want, got = set(ROLLOUT_COLUMN_NAMES), set(batch)
        if want != got:
            raise ValueError(
                f"rollout episode batch does not match the schema: missing "
                f"{sorted(want - got)}, unexpected {sorted(got - want)}")
        lengths = {len(np.atleast_1d(v)) for v in batch.values()}
        if len(lengths) != 1:
            raise ValueError(f"ragged rollout episode batch: column lengths {lengths}")
        n = lengths.pop()
        for i in range(n):
            self._rows.append({k: np.atleast_1d(batch[k])[i] for k in ROLLOUT_COLUMN_NAMES})
        return n

    def __len__(self) -> int:
        """Episodes appended so far."""
        return len(self._rows)

    def write(self, path: str, attrs: dict | None = None,
              chunks: "ChunkRewardTable | None" = None) -> None:
        """Write the table plus its provenance.

        Args:
            path: Destination ``.h5``.  Parent directories are created.
            attrs: Run provenance; every value is JSON-encoded so nested dicts survive.
            chunks: An optional per-re-plan terrain-reward table, written into a ``chunks``
                GROUP of the same file.  One artifact rather than two, because the chunk rows
                join to these episodes and a pair of files can be separated.
        """
        import h5py

        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with h5py.File(path, "w") as f:
            for name, dtype in ROLLOUT_EPISODE_COLUMNS:
                f.create_dataset(name, data=np.asarray([r[name] for r in self._rows],
                                                       dtype=dtype))
            f.attrs["schema_version"] = SCHEMA_VERSION
            f.attrs["columns"] = json.dumps(list(ROLLOUT_COLUMN_NAMES))
            f.attrs["family_names"] = json.dumps(self.family_names)
            for key, value in (attrs or {}).items():
                f.attrs[key] = json.dumps(value, default=str)
            if chunks is not None:
                chunks.write_into(f)


class ChunkRewardTable:
    """Append-only per-RE-PLAN terrain-consistency table, written beside the episodes.

    One row per re-plan, keyed ``(episode_uid, replan_index)`` -- the same key the window
    table stamps and the same key ``inspect_terrain_reward.py --stage reward`` writes its
    sidecar under.  That is what lets the live and offline grades be joined and differenced
    rather than merely compared in aggregate.

    Held in memory like :class:`RolloutOutcomeTable`: a 4000-episode arm is ~150k rows of 15
    small columns, under 10 MB.
    """

    def __init__(self, spec_meta: dict | None = None) -> None:
        """Start an empty table.

        Args:
            spec_meta: The :class:`~motion_gen.terrain_consistency.TerrainConsistencySpec`
                and contact settings the rows were graded under, stamped onto the group.  The
                offline re-grade READS this rather than restating its own defaults, so the
                two sides cannot drift into grading different things while reporting the same
                column names.
        """
        self._cols: dict[str, list[np.ndarray]] = {n: [] for n in ROLLOUT_CHUNK_NAMES}
        self._n = 0
        self.spec_meta = dict(spec_meta or {})

    def append_batch(self, batch: dict[str, np.ndarray]) -> int:
        """Add one row per element of ``batch``'s arrays.

        Args:
            batch: A dict whose keys are exactly :data:`ROLLOUT_CHUNK_NAMES`.

        Returns:
            How many rows were added.

        Raises:
            ValueError: On a missing or unexpected key, or on ragged lengths.  A missing
                ``penetration_volume_m3`` is the expected way to get here: it is ABSENT (not
                zero) when the collider model carries no areas, i.e. when the model was built
                in extreme-point mode instead of by surface sampling.
        """
        want, got = set(ROLLOUT_CHUNK_NAMES), set(batch)
        if want != got:
            raise ValueError(
                f"terrain chunk batch does not match the schema: missing {sorted(want - got)}"
                f", unexpected {sorted(got - want)}. A missing `penetration_volume_m3` means "
                "the collider model has no areas -- build it with a surface `spacing` rather "
                "than in extreme-point mode.")
        lengths = {len(np.atleast_1d(v)) for v in batch.values()}
        if len(lengths) != 1:
            raise ValueError(f"ragged terrain chunk batch: column lengths {lengths}")
        n = lengths.pop()
        for name, dtype in ROLLOUT_CHUNK_COLUMNS:
            self._cols[name].append(np.asarray(batch[name], dtype=dtype).reshape(-1))
        self._n += n
        return n

    def __len__(self) -> int:
        """Rows appended so far."""
        return self._n

    def write_into(self, f) -> None:
        """Write the rows as a ``chunks`` group of an open episodes file.

        Args:
            f: An open, writable :class:`h5py.File`.
        """
        g = f.create_group("chunks")
        for name, dtype in ROLLOUT_CHUNK_COLUMNS:
            parts = self._cols[name]
            data = (np.concatenate(parts) if parts else np.zeros(0, dtype=dtype))
            g.create_dataset(name, data=data.astype(dtype))
        g.attrs["columns"] = json.dumps(list(ROLLOUT_CHUNK_NAMES))
        g.attrs["n_rows"] = int(self._n)
        g.attrs["spec"] = json.dumps(self.spec_meta, sort_keys=True, default=str)


def read_rollout_chunks(path: str) -> tuple[dict[str, np.ndarray], dict]:
    """Read the ``chunks`` group written by :meth:`ChunkRewardTable.write_into`.

    Args:
        path: The episodes ``.h5``.

    Returns:
        ``(columns, spec)``; ``({}, {})`` when the file carries no chunk group (which is what
        a run without ``--terrain_reward`` produces).
    """
    import h5py

    with h5py.File(path, "r") as f:
        if "chunks" not in f:
            return {}, {}
        g = f["chunks"]
        cols = {name: g[name][:] for name in ROLLOUT_CHUNK_NAMES if name in g}
        spec = json.loads(g.attrs.get("spec", "{}"))
    return cols, spec


def read_rollout_episodes(path: str) -> tuple[dict[str, np.ndarray], dict]:
    """Read a table written by :meth:`RolloutOutcomeTable.write`.

    Args:
        path: The ``.h5``.

    Returns:
        ``(columns, attrs)``, attrs JSON-decoded.

    Raises:
        ValueError: On an unknown ``schema_version``.  Reading a renamed column as its old
            self is worse than refusing.
    """
    import h5py

    cols: dict[str, np.ndarray] = {}
    attrs: dict = {}
    with h5py.File(str(path), "r") as f:
        version = int(f.attrs.get("schema_version", -1))
        # The episode table's COLUMNS are unchanged across schemas 2 and 3 -- the
        # perturbation arms went into the window table and the terrain grade into its own
        # `chunks` group -- so an older file needs no backfill here, only admitting.
        if version not in READABLE_SCHEMAS:
            raise ValueError(
                f"{path} is schema_version {version}; this reader knows "
                f"{sorted(READABLE_SCHEMAS)}. "
                "Re-record rather than reading it through the wrong contract.")
        for key in f.keys():
            # DATASETS ONLY. `--terrain_reward` puts the per-chunk grade in a `chunks` GROUP
            # inside this same file (`ChunkRewardTable.write_into`), and `f[key][...]` on a
            # group raises `TypeError: Accessing a group is done with bytes or str, not
            # <class 'ellipsis'>` -- so every recording graded live was unreadable through
            # this reader, and unmergeable, while `box2_ic_report` worked because it uses
            # h5py directly. The chunk table has its own reader (`read_rollout_chunks`);
            # this one returns the episode COLUMNS and must skip anything that is not one.
            if not isinstance(f[key], h5py.Dataset):
                continue
            cols[key] = f[key][...]
        for key, raw in f.attrs.items():
            if key == "schema_version":
                attrs[key] = int(raw)
                continue
            try:
                attrs[key] = json.loads(raw)
            except (TypeError, ValueError):
                attrs[key] = raw
    return cols, attrs


#: ``(name, dtype)`` per WINDOW, in write order.  Every one is the model's own input or
#: output at a re-plan; nothing here is reconstructed.
ROLLOUT_WINDOW_COLUMNS: tuple[tuple[str, str], ...] = (
    # --- join keys ---------------------------------------------------------------------
    ("episode_uid", _L),          # matches the outcome table's column
    ("replan_index", _I),         # from 0, within the episode
    ("env_id", _I),
    ("clip_id", _I),
    ("family_code", _U8),
    # --- the conditioning: state s_t ----------------------------------------------------
    ("history", "f4"),            # (N, num_history, D) -- the nodes it emitted before this
    ("history_null", _U8),        # 1 while the ring is still filling; the learned null branch
    ("extero", "f2"),             # (T, K) -- the reading ACTUALLY served, lag included
    ("cmd_future", "f4"),         # (F, cmd_dim) -- the command over the horizon
    ("depth_lag_used", _U8),      # lag served, in control steps
    # --- HOW THE GENERATOR WAS PERTURBED WHEN IT PRODUCED THIS CHUNK --------------------
    #
    # The two diagnostic arms are a BEHAVIOUR POLICY: the generator is conditioned on a dead
    # upper camera and/or the learned null-history embedding to widen the distribution of
    # states and plans a run visits, while `history` and `extero` above store the TRUE
    # measurement -- so the critic and the fine-tune learn from what actually happened.
    #
    # That makes these flags the only record that a row's ACTION came from conditioning other
    # than the state stored beside it.  Without them the perturbed and pristine populations are
    # indistinguishable on disk: the advantage could not be split by arm, and a fine-tune could
    # not weight, drop or balance them.  Two bytes a window against ~30 kB of conditioning.
    #
    # `history_null` above is a DIFFERENT thing and both are needed: it is the post-reset
    # transient, true of the recorded history, whereas this is the deliberate arm applied to
    # the model while the recorded history stayed complete.
    ("perturb_null_history", _U8),   # the learned null embedding replaced the real history
    ("perturb_noisy_upper_cam", _U8),  # the upper (far) camera was replaced with garbage
    ("perturb_speed_mismatch", _U8),   # the CONDITIONING command speed was offset
    # The OFFSET, in m/s, signed. Recorded rather than the absolute conditioning speed
    # because `cmd_future` already carries the TRUE command, so the pair is complete and a
    # zero here reads as "no mismatch" rather than as a commanded standstill.
    ("perturb_speed_offset_mps", _F),
    # --- the action a_t -----------------------------------------------------------------
    ("fm_plan", "f4"),            # (F, D) -- the emitted chunk, in window space
    # --- where it happened, for privileged critic features and diagnostics --------------
    ("anchor_pos", "f4"),         # (3,) plan anchor, world
    ("anchor_quat", "f4"),        # (4,) xyzw
    ("clip_step", _I),
    ("height_scan", "f2"),        # (K_scan,) privileged; empty when the task has no scanner
    # --- where the goal is, from HERE. Privileged, and load-bearing for the critic --------
    #
    # A value function fitted with a sparse terminal goal reward cannot estimate P(fail)
    # without knowing how far the goal is: two states that look identical to the generator
    # are worth different amounts depending on whether the terrain is one metre ahead or
    # eight. Omitting this makes V unpredictable BY CONSTRUCTION, and the resulting poor fit
    # looks like the method failing rather than like a missing input.
    #
    # Recorded per window rather than derived later because deriving it needs the clip's
    # world placement, which only exists inside the running sim.
    ("goal_downrange_m", "f4"),   # + = past the goal anchor, in the TERRAIN frame
    ("goal_lateral_m", "f4"),
    ("goal_height_err_m", "f4"),
    # --- proprioception, as per-timestep tokens for the critic ---------------------------
    #
    # `(H, PROPRIO_DIM)`, NEWEST LAST. Its own ring rather than the observation manager's
    # history-stacked terms: those flatten to `(N, history*dim)`, and whether that is (T, D)
    # or (D, T) is an IsaacLab internal. Getting it backwards is shape-silent and would hand
    # the critic a transposed history it could still partly fit.
    ("proprio", "f4"),
    # The measured base TWIST -- linear(3) + angular(3) -- in the ROBOT'S YAW FRAME,
    # `yaw_quat(root_quat_w)`. Six wide and in that frame on purpose: that is exactly the
    # layout and the frame of the plan's own `base_vel` block, so "what the base is doing"
    # and "what the plan asked for" are comparable without a transform. The body frame would
    # tilt with the torso and swing the forward component on every pitch; world would defeat
    # heading invariance.
    #
    # Only the LINEAR half is privileged -- it needs odometry. The angular half is
    # measurable: an IMU gives body-frame rates and the yaw frame is reachable from those
    # plus the IMU's own roll and pitch, the same argument that keeps the generator's depth
    # unprojection odometry-free. Carried together so the twist stays one object;
    # `V_onboard` takes the angular half and drops the linear one.
    ("base_twist_yaw", "f4"),
    # --- the per-step tracking ERROR of the PREVIOUS chunk -------------------------------
    #
    # `(K, NUM_FIXED_CHANNELS + nj)`, oldest first, as `transfer/deploy/tracking_error.py`
    # lays it out. What the critic's QUALITY head regresses a return over, and what the
    # strict failure checks are thresholded from.
    #
    # **`_prev` IS THE POINT, AND THE JOIN IS OFF BY ONE ROW.** A window is emitted at the
    # re-plan that PRODUCES its `fm_plan`, and that chunk's consequences have not happened
    # yet -- it goes on to drive the next K control steps. So the error carried here is the
    # one the PREVIOUS chunk produced, and the reward for row `i`'s action is row `i+1`'s
    # `track_err_prev`, joined within an episode on `replan_index`. That is the ordinary
    # `(s_t, a_t, r_t)` storage shift, and the column is NAMED for it because storing it
    # unnamed would be an off-by-one-chunk that fits, trains, and produces a plausible weak
    # critic rather than an error.
    #
    # The re-plan is applied inside `_update_command`, i.e. inside `env.step`, so by the time
    # the recorder folds this step the cursor is already on the NEW plan's node 0. The ring is
    # therefore K+1 deep and the newest row -- the one belonging to the new chunk -- is
    # dropped here, leaving exactly the K steps the previous chunk served.
    #
    # The ERROR VECTOR is stored, never a reward and never a boolean. Both bake a choice into
    # the file: the kernel widths and weights would fix the reward, the bounds would fix the
    # verdict. Storing rows keeps all of them an OFFLINE knob, so a threshold or a weight
    # chosen after this recording is a relabel rather than a re-record -- which is the whole
    # reason the checks are latched rather than terminating. It also makes the episode
    # table's `first_violation_step` recomputable under any spec: re-plans tile the episode,
    # so concatenating these blocks in `replan_index` order reconstructs the per-step error.
    #
    # Cost: ~2 kB/window at K=12, nj=29 against `extero`'s 15.6 kB, so about +13% on the file.
    ("track_err_prev", "f4"),
)

ROLLOUT_WINDOW_NAMES: tuple[str, ...] = tuple(n for n, _ in ROLLOUT_WINDOW_COLUMNS)

#: Per-window columns small enough to load in full just to DECIDE what else to load.  Fitting
#: a value function walks the join keys and the outcome; it touches an extero row only for
#: the windows it actually trains on.
ROLLOUT_WINDOW_INDEX: tuple[str, ...] = (
    "episode_uid", "replan_index", "env_id", "clip_id", "family_code", "history_null",
    "depth_lag_used", "clip_step",
    "goal_downrange_m", "goal_lateral_m", "goal_height_err_m",
)

#: Width of one proprioception row: joint_pos(29) + joint_vel(29) + base_ang_vel(3) +
#: projected_gravity(3). All measurable on the robot; none of it visible to the generator,
#: which conditions on its own plan, the camera row and the command.
PROPRIO_DIM: int = 64
#: Control steps of proprioception kept per window. 10 = 0.2 s, at the CONTROL rate rather
#: than the re-plan rate: a fall develops over tens of milliseconds, and sampling it every
#: 0.24 s would miss the onset -- which is the moment the lead-time gate is entirely about.
PROPRIO_HISTORY: int = 10


class TrackErrRing:
    """The tracking-error rows since the last re-plan, newest last.

    Sibling of :class:`ProprioRing` and deliberately the same shape of object, because it has
    the same three properties and each was got wrong somewhere before: kept at the CONTROL
    rate rather than the re-plan rate, zeroed on reset so no window spans two episodes, and
    read at the re-plan instant rather than reconstructed afterwards.

    Depth is ``fm_update_every + 1``, and the extra row is the whole reason this is not a
    plain ``ProprioRing``.  The re-plan is applied inside ``_update_command`` -- inside
    ``env.step`` -- so by the time the recorder folds the step at which a window is emitted,
    the cursor is already on the NEW plan's node 0 and that step's error belongs to the new
    chunk.  A ring of depth ``K`` would therefore hand out ``K-1`` steps of the previous chunk
    plus one of the new one: a mix belonging to neither action.  With ``K+1``, dropping the
    newest row leaves exactly the ``K`` steps the previous chunk served.

    See ``track_err_prev`` in :data:`ROLLOUT_WINDOW_COLUMNS` for the resulting one-row join
    shift, which is deliberate and is named for.

    Args:
        n_envs: Environment count.
        device: Where the buffer lives.
        history: Rows kept; pass ``fm_update_every + 1``.
        dim: Row width, ``NUM_FIXED_CHANNELS + num_joints``.
    """

    def __init__(self, n_envs: int, device, history: int, dim: int) -> None:
        self.buf = torch.zeros(n_envs, int(history), int(dim), device=device)
        self.history = int(history)

    def push(self, row: torch.Tensor | None) -> None:
        """Roll in this step's error row for every env.

        Args:
            row: ``[N, dim]``, or ``None`` before the generator's first plan -- in which case
                nothing is rolled, so the ring holds zeros rather than a stale row repeated.
        """
        if row is None:
            return
        self.buf = torch.roll(self.buf, shifts=-1, dims=1)
        self.buf[:, -1] = row

    def invalidate(self, dones: torch.Tensor) -> None:
        """Zero the ring for envs that just reset."""
        ids = torch.nonzero(dones.reshape(-1)).flatten()
        if ids.numel():
            self.buf[ids] = 0.0


class RolloutWindowWriter:
    """Streams per-re-plan windows to a chunked ``.h5``.

    Pre-sized and written in place, like ``recorded_data_io.RecordedWindowWriter``: a rollout
    is hours long and the extero rows alone are tens of gigabytes, so nothing is buffered
    beyond one flush batch.

    ``extero`` and ``height_scan`` are **float16**.  Both are sensor readings whose useful
    precision is far coarser than f32 -- the depth row arrives already normalised and clipped
    to a band of +-5.575, so f16's ~3 decimal digits resolve it to well under a millimetre at
    the near plane -- and they are the whole file's size. Measured elsewhere in this repo:
    lzf holds ~39x on f16 sensor data where it collapses to ~1.7x on f32 with mantissa
    jitter.
    """

    def __init__(self, path: str, *, num_windows: int, num_history: int, state_dim: int,
                 future_len: int, cmd_dim: int, extero_shape: tuple[int, int],
                 proprio_hw: tuple[int, int], scan_dim: int = 0,
                 track_hw: tuple[int, int] | None = None, attrs: dict | None = None,
                 compression: str | None = "lzf") -> None:
        """Create the file and pre-size every dataset.

        Args:
            path: Destination ``.h5``.
            num_windows: Upper bound on windows; the file records how many were written.
            num_history: ``N``, history tokens per window.
            state_dim: ``D``, the window node width.
            future_len: ``F``, nodes the generator predicts.
            cmd_dim: Command width.
            extero_shape: ``(T, K)`` per reading, from the generator's own contract.
            proprio_hw: ``(H, PROPRIO_DIM)`` proprioception ring, newest last.
            scan_dim: Privileged height-scan width, or 0 for none.
            track_hw: ``(fm_update_every, NUM_FIXED_CHANNELS + nj)`` -- the WRITTEN block's
                shape, one row per step of the previous chunk. ``None`` omits the column
                rather than zero-filling it, because zeros read as "tracked perfectly".
            attrs: Provenance, JSON-encoded onto the root.
            compression: h5py compression for every dataset.
        """
        import h5py

        self.path = str(path)
        self.num_windows = int(num_windows)
        shapes = {
            "episode_uid": (num_windows,), "replan_index": (num_windows,),
            "env_id": (num_windows,), "clip_id": (num_windows,),
            "family_code": (num_windows,), "history_null": (num_windows,),
            "depth_lag_used": (num_windows,), "clip_step": (num_windows,),
            # The behaviour-policy arms. Declared HERE as well as in
            # ROLLOUT_WINDOW_COLUMNS because `shapes` is what actually gets allocated -- a
            # column present only in the tuple is created by nothing and dropped in silence.
            "perturb_null_history": (num_windows,),
            "perturb_noisy_upper_cam": (num_windows,),
            "perturb_speed_mismatch": (num_windows,),
            "perturb_speed_offset_mps": (num_windows,),
            "history": (num_windows, int(num_history), int(state_dim)),
            "extero": (num_windows, int(extero_shape[0]), int(extero_shape[1])),
            "cmd_future": (num_windows, int(future_len), int(cmd_dim)),
            "fm_plan": (num_windows, int(future_len), int(state_dim)),
            "anchor_pos": (num_windows, 3), "anchor_quat": (num_windows, 4),
            "goal_downrange_m": (num_windows,), "goal_lateral_m": (num_windows,),
            "goal_height_err_m": (num_windows,),
            "proprio": (num_windows, int(proprio_hw[0]), int(proprio_hw[1])),
            "base_twist_yaw": (num_windows, 6),
        }
        if int(scan_dim) > 0:
            shapes["height_scan"] = (num_windows, int(scan_dim))
        if track_hw is not None:
            shapes["track_err_prev"] = (num_windows, int(track_hw[0]), int(track_hw[1]))
        dtypes = dict(ROLLOUT_WINDOW_COLUMNS)
        self._keys = list(shapes)
        self._f = h5py.File(self.path, "w")
        for key, shape in shapes.items():
            self._f.create_dataset(key, shape=shape, dtype=dtypes[key],
                                   chunks=_chunk(shape, dtypes[key]),
                                   compression=compression)
        self._f.attrs["schema_version"] = SCHEMA_VERSION
        self._f.attrs["columns"] = json.dumps(self._keys)
        for k, v in (attrs or {}).items():
            self._f.attrs[k] = json.dumps(v, default=str)
        self._f.attrs["num_windows"] = int(num_windows)
        self._written = 0

    @property
    def written(self) -> int:
        """Windows written so far."""
        return self._written

    def write_batch(self, batch: dict[str, np.ndarray]) -> int:
        """Append a batch, truncating at ``num_windows``.

        Args:
            batch: One array per declared key, all the same leading length.

        Returns:
            How many rows were actually written (0 once the file is full).

        Raises:
            KeyError: On a missing declared key -- a silently absent column would read back
                as zeros, which for ``fm_plan`` is a valid-looking window.
        """
        missing = [k for k in self._keys if k not in batch]
        if missing:
            raise KeyError(f"rollout window batch is missing {missing}")
        n = len(np.atleast_1d(batch[self._keys[0]]))
        take = min(n, self.num_windows - self._written)
        if take <= 0:
            return 0
        lo, hi = self._written, self._written + take
        for key in self._keys:
            self._f[key][lo:hi] = np.asarray(batch[key])[:take]
        self._written = hi
        return take

    def close(self) -> None:
        """Record how many windows were written and close."""
        if self._f is not None:
            self._f.attrs["num_windows_written"] = int(self._written)
            self._f.close()
            self._f = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *exc):
        """Close on the way out, including on an exception."""
        self.close()
        return False


def _chunk(shape: tuple[int, ...], dtype: str, target_bytes: int = 4 << 20) -> tuple[int, ...]:
    """A chunk covering whole rows, ~``target_bytes`` at a time.

    Args:
        shape: The dataset shape.
        dtype: Its dtype string.
        target_bytes: Chunk size to aim for.

    Returns:
        The chunk shape.
    """
    row = int(np.prod(shape[1:])) * int(np.dtype(dtype).itemsize) if len(shape) > 1 else \
        int(np.dtype(dtype).itemsize)
    rows = max(1, min(int(shape[0]), max(1, target_bytes // max(row, 1)), 8192))
    return (rows,) + tuple(int(s) for s in shape[1:])


def read_rollout_windows(path: str, columns: list[str] | None = None
                         ) -> tuple[dict[str, np.ndarray], dict]:
    """Read a window table, optionally only some columns.

    Args:
        path: The ``.h5``.
        columns: Columns to load; ``None`` loads all. Pass
            :data:`ROLLOUT_WINDOW_INDEX` to walk the join keys without touching an image.

    Returns:
        ``(data, attrs)``, sliced to the number of windows actually written.

    Raises:
        ValueError: On an unknown ``schema_version``, or a requested column that is absent.
    """
    import h5py

    data: dict[str, np.ndarray] = {}
    attrs: dict = {}
    with h5py.File(str(path), "r") as f:
        version = int(f.attrs.get("schema_version", -1))
        if version not in READABLE_SCHEMAS:
            raise ValueError(
                f"{path} is schema_version {version}; this reader knows "
                f"{sorted(READABLE_SCHEMAS)}.")
        n = int(f.attrs.get("num_windows_written", f.attrs.get("num_windows", 0)))
        want = list(columns) if columns is not None else [k for k in f.keys()]
        # A column added after this file's schema is BACKFILLED with the value that is true of
        # a file written before the feature existed; anything else absent is still a refusal.
        fill = {c: WINDOW_SCHEMA_BACKFILL[c] for c in want
                if c not in f and c in WINDOW_SCHEMA_BACKFILL}
        absent = [c for c in want if c not in f and c not in fill]
        if absent:
            raise ValueError(f"{path} has no column(s) {absent}")
        for key in want:
            if key in fill:
                value, dtype = fill[key]
                data[key] = np.full(n, value, dtype=dtype)
                continue
            data[key] = f[key][:n]
        for key, raw in f.attrs.items():
            if key in ("schema_version", "num_windows", "num_windows_written"):
                attrs[key] = int(raw)
                continue
            try:
                attrs[key] = json.loads(raw)
            except (TypeError, ValueError):
                attrs[key] = raw
    return data, attrs


class RolloutEpisodeTracker:
    """Owns the episode identity, folds outcomes, and harvests inside the clip reset.

    Wraps ``events.reset_on_clip``'s function, exactly as ``eval.runner.EvalResetHook`` does
    and for the same reason: the state of the episode that just ended is readable there and
    nowhere else.  By the time ``env.step`` returns, the buffers describe the next spawn.

    It also owns the two counters the window table joins on -- ``episode_uid`` (per episode,
    global) and ``replan_index`` (per re-plan, within an episode).  Both live here rather than
    in the recorder because they must be reset together and at the same instant: an episode
    id bumped without its index, or the reverse, produces a table that joins to the wrong
    rows for exactly one episode per env and reads correctly everywhere else.
    """

    def __init__(self, base_func, *, env, accum, table, family_codes=None,
                 goal_spec=None) -> None:
        """Install over a reset function.

        Args:
            base_func: The original ``reset_on_clip`` function.
            env: The unwrapped env.
            accum: An ``eval.accumulate.EpisodeAccumulator``.
            table: A :class:`RolloutOutcomeTable`.
            family_codes: Optional ``[num_clips]`` long tensor of family codes.
            goal_spec: Optional ``mdp.goal_region.GoalRegionSpec``, already attached to
                ``accum``; used here only to re-resolve per-family tolerances on reset.
        """
        self.base_func = base_func
        self.env = env
        self.accum = accum
        self.table = table
        self.family_codes = family_codes
        self.goal_spec = goal_spec
        n = int(env.num_envs)
        dev = env.device
        self.episode_uid = torch.arange(n, dtype=torch.long, device=dev)
        self.replan_index = torch.zeros(n, dtype=torch.long, device=dev)
        self._next_uid = int(n)
        self.armed = False
        self.harvested = 0
        #: Episodes each env has completed NATURALLY (the end-of-run flush does not count).
        #: A rollout that stops on a total episode count stops as soon as the fastest
        #: episodes have finished, which under a per-episode speed draw selects the shortest
        #: ones and discards every slow episode still in flight. Stopping instead once EVERY
        #: env has completed the same number makes the collected set a complete design over
        #: (env, round) whose membership does not depend on how long an episode took.
        self.completed_per_env = np.zeros(n, dtype=np.int64)
        self._term_names: list[str] = []

    # ------------------------------------------------------------------
    def on_replan(self, idx: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Claim ``(episode_uid, replan_index)`` for the envs emitting a window now.

        Args:
            idx: ``[k]`` env indices whose re-plan produced a window this step.

        Returns:
            ``(uid[k], replan_index[k])`` as numpy, BEFORE the index is bumped -- so the
            first window of an episode is index 0.
        """
        uid = self.episode_uid[idx].detach().to("cpu").numpy().astype(np.int64)
        ridx = self.replan_index[idx].detach().to("cpu").numpy().astype(np.int32)
        self.replan_index[idx] += 1
        return uid, ridx

    def fold(self, dones) -> None:
        """Fold one step, excluding envs that just reset.

        Args:
            dones: The step's done flags.
        """
        self.accum.fold(~dones.to(torch.bool))

    # ------------------------------------------------------------------
    def __call__(self, env, env_ids, **kwargs):
        """Harvest the ending episodes, then delegate to the real clip reset."""
        if self.armed and env_ids is not None and len(env_ids) > 0:
            self._harvest(env, env_ids)
        self.armed = True
        out = self.base_func(env, env_ids, **kwargs)
        # AFTER the reset commits the new clip: the goal anchor is that clip's own terminal
        # pose and its tolerances are that clip's family's, so resolving either beforehand
        # describes the PREVIOUS episode -- a plausible region on the wrong tile.
        self._refresh_goal(env, env_ids)
        return out

    def _refresh_goal(self, env, env_ids) -> None:
        """Re-resolve the goal anchor and per-family tolerances for the envs that reset."""
        if getattr(self.accum, "_gr", None) is None:
            return
        if self.goal_spec is not None and self.family_codes is not None:
            cmd = env.command_manager.get_term("traj_ref")
            ids = (torch.arange(int(env.num_envs), device=cmd.clip_id.device)
                   if env_ids is None else env_ids)
            codes = self.family_codes[cmd.clip_id.to(self.family_codes.device)]
            names = [self.table.family_names[int(c)]
                     for c in codes[ids.to(codes.device)].tolist()]
            self.accum.set_goal_families(_normalized(names), env_ids=ids)
        self.accum.refresh_goal_anchor(env_ids)

    def flush(self, env, env_ids) -> None:
        """Harvest the episodes still in flight, marked censored.

        Called once at the end of a rollout.  Those episodes did not finish, so their
        outcome is not "failed to arrive" -- they have none, and
        :func:`terminal_reward`'s caller drops them.

        Args:
            env: The unwrapped env.
            env_ids: The envs still running.
        """
        if env_ids is not None and len(env_ids) > 0:
            self._harvest(env, env_ids, censored=True)

    # ------------------------------------------------------------------
    def _harvest(self, env, env_ids, censored: bool = False) -> None:
        """Emit one row per ended episode, zero those accumulators, and issue new uids."""
        ids = env_ids if torch.is_tensor(env_ids) else torch.as_tensor(env_ids)
        ids = ids.to(self.accum.device).long()
        ids_np = ids.detach().to("cpu").numpy().astype(np.int64)
        cmd = env.command_manager.get_term("traj_ref")
        tm = env.termination_manager
        if not self._term_names:
            self._term_names = list(tm.active_terms)

        # The terminal frame, before `snapshot` reduces. Without it the one frame that ENDED
        # the episode -- for a fall, the frame that crossed the threshold -- is the one frame
        # missing from every extreme.
        from robot_rl.tasks.manager_based.robot_rl.mdp import goal_region  # noqa: PLC0415

        self.accum.fold_terminal(ids)
        red = self.accum.snapshot(ids)
        gp = self.accum._goal_pos_w[ids].detach().to("cpu").numpy()
        travel = goal_region.goal_travel_m(cmd).detach().to("cpu").numpy()[ids_np]

        truncated = tm.time_outs.detach().to("cpu").numpy().astype(bool)[ids_np]
        terminated = tm.terminated.detach().to("cpu").numpy().astype(bool)[ids_np]

        lu = goal_region.last_usable_step(cmd).detach().to("cpu").numpy()[ids_np]
        clip_ids = cmd.clip_id.detach().to("cpu").numpy().astype(np.int64)[ids_np]
        lib = cmd.manager.lib
        tid = getattr(lib, "terrain_id", None)
        if tid is not None and tid.numel():
            terrain = tid.detach().to("cpu").numpy()[clip_ids].astype(np.int64)
        else:
            # A CLIP-FREE library has no per-clip tile table, but the env is still standing on
            # a tile and the importer knows which. Without this the column is -1 on every row
            # and the tile an episode ran on -- which on the free-form terrain IS its initial
            # condition -- is unrecoverable from the table.
            env_tid = getattr(getattr(env.scene, "terrain", None), "env_terrain_id", None)
            terrain = (env_tid.detach().to("cpu").numpy()[ids_np].astype(np.int64)
                       if env_tid is not None else np.full(ids_np.size, -1))
        fam = (self.family_codes.detach().to("cpu").numpy()[clip_ids]
               if self.family_codes is not None else np.zeros(ids_np.size))

        self.table.append_batch({
            "episode_uid": self.episode_uid[ids].detach().to("cpu").numpy(),
            "env_id": ids_np,
            "clip_id": clip_ids,
            "terrain_id": terrain,
            "family_code": fam,
            "start_step": cmd.start_step.detach().to("cpu").numpy()[ids_np],
            "last_usable_step": lu,
            "n_replans": self.replan_index[ids].detach().to("cpu").numpy(),
            "steps_survived": red["steps_survived"],
            "terminated": terminated.astype(np.uint8),
            "truncated": truncated.astype(np.uint8),
            "censored": np.full(ids_np.size, int(bool(censored)), dtype=np.uint8),
            "first_fall_step": red["first_fall_step"],
            "first_violation_step": red["first_violation_step"],
            "first_lateral_step": red["first_lateral_step"],
            "first_height_step": red["first_height_step"],
            "first_tilt_step": red["first_tilt_step"],
            "track_reward_mean": red["track_reward_mean"],
            "min_base_height_w": red["min_base_height_w"],
            "max_tilt_rad": red["max_tilt_rad"],
            "reached_goal": red["reached_goal"],
            "first_goal_step": red["first_goal_step"],
            "min_dist_to_goal_m": red["min_dist_to_goal_m"],
            "goal_downrange_m": red["goal_downrange_m"],
            "goal_lateral_m": red["goal_lateral_m"],
            "goal_height_err_m": red["goal_height_err_m"],
            "max_goal_downrange_m": red["max_goal_downrange_m"],
            "goal_pos_w_x": gp[:, 0], "goal_pos_w_y": gp[:, 1], "goal_pos_w_z": gp[:, 2],
            "goal_frame_yaw": self.accum._goal_yaw[ids].detach().to("cpu").numpy(),
            # `first_goal_step == 1` is the first FOLDED step: `EpisodeAccumulator.fold`
            # increments its counter before the goal fold runs, so arrival is 1-based and 0
            # means "never".
            "spawned_in_goal": (red["first_goal_step"] == 1).astype(np.uint8),
            "goal_travel_m": travel,
        })
        self.harvested += ids_np.size
        if not censored:
            np.add.at(self.completed_per_env, ids_np, 1)
        self.accum.reset_envs(ids)
        self.reissue(ids)

    def rounds_complete(self) -> int:
        """How many episodes EVERY env has finished -- the number of complete rounds.

        Returns:
            ``min`` over envs of the naturally-completed episode count.
        """
        return int(self.completed_per_env.min()) if self.completed_per_env.size else 0

    def reissue(self, ids: torch.Tensor) -> None:
        """Give ``ids`` fresh identities: a new uid each, and a re-plan index back at 0.

        Split out of :meth:`_harvest` so the invariant is testable without a simulator.
        BOTH counters, in one call, always: a uid bumped without its index makes the first
        windows of the new episode claim indices continuing the previous one, and an index
        reset without a uid makes two episodes share a key. Either way the tables still join
        -- to the wrong rows, for exactly the envs that reset, which is the subset most
        likely to be interesting.

        Args:
            ids: ``[k]`` env indices that just ended an episode.
        """
        k = int(ids.numel())
        if k == 0:
            return
        self.episode_uid[ids] = torch.arange(self._next_uid, self._next_uid + k,
                                             dtype=torch.long, device=self.episode_uid.device)
        self._next_uid += k
        self.replan_index[ids] = 0


def _normalized(names: list[str]) -> list[str]:
    """Strip the ``family_`` prefix and any ``.h5`` suffix, as the eval's sampler does.

    The clip library spells a family ``family_stairs_up`` (and, for a composed library,
    ``family_stairs_up.h5``); ``goal_regions.json`` is keyed the way
    ``eval.sampler.normalize_family`` returns it.  Deliberately re-implemented in one line
    rather than importing the eval's sampler, which would pull its whole clip-index module in
    for a string strip.

    Args:
        names: Raw family names.

    Returns:
        Normalised names.
    """
    out = []
    for n in names:
        n = str(n)
        n = n[:-3] if n.endswith(".h5") else n
        out.append(n[len("family_"):] if n.startswith("family_") else n)
    return out
