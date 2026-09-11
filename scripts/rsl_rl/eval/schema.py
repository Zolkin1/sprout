# Copyright 2026 Zachary Olkin. All rights reserved.

"""The eval artifact contract: columns, derived arithmetic, and the table writer/reader.

Shared by the run side (which fills it) and the reduce side (which reads it), so a column
name is spelled exactly once.  Pure numpy + h5py: no torch, no IsaacLab, no matplotlib.

Design rules the column list follows, each of which exists because the obvious alternative
produces a wrong statistic:

* **Store primitives, derive the rest.**  ``success`` is a contested definition (the teacher
  task fails on clip-graded tracking error, the tracker task only on falling), so the table
  stores which terms fired and how long the episode lasted, and the reducer derives success
  from a *named* definition it prints.
* **No ``time_to_failure_s`` column.**  It is ``steps_survived * dt`` for failures and
  undefined for successes, and the first ``nanmean`` over such a column silently conditions
  on failing -- which is the bias that made a policy surviving only its three easiest envs
  look like the best tracker in the repo's own survival sweep.  Stored instead as
  ``steps_survived`` plus ``event_observed``, the standard survival pair.
* **No ``survived_fraction_*`` column.**  The denominator is ambiguous (``clip_len``,
  ``clip_len - 1``, or the reserved-lookahead end) and they differ by 5-15%, so a perfect
  episode can read 0.95 and look like a near-miss.  The reducer forms fractions from
  primitives with a named denominator.
* **MSE, not RMSE.**  Means of per-episode RMSE are biased low by Jensen, and
  ``vel_rmse_lin = sqrt(mse_vx + mse_vy)`` needs the squares.  This matches
  ``play_plots``/``clip_ab_sweep``, which square-root only at the join.
* **Enum codes plus a name table**, following the library's own
  ``terrain_type_code``/``terrain_type_names`` idiom, so the file carries no variable-length
  strings and every consumer resolves names the same way.
"""

from __future__ import annotations

import json

import numpy as np

#: Bump on any column removal, rename or change of meaning.  The grid refuses to reuse a
#: cell written under a different major, and the reducer refuses a file it does not know.
SCHEMA_VERSION: int = 2
#: Bump on a purely additive column.  A reader may ignore columns it does not know.
SCHEMA_MINOR: int = 1

_F = "f4"
_I = "i4"
_U8 = "u1"
_U16 = "u2"

#: ``(name, dtype)`` for every per-episode column, in write order.
#:
#: Grouped by what answers which question; the groups are not stored separately.
EPISODE_COLUMNS: tuple[tuple[str, str], ...] = (
    # --- identity: what ran ------------------------------------------------------------
    ("episode_id", _I),          # unique within the cell
    ("env_id", _I),
    ("clip_id", _I),             # index into the SUBSET library == the plan's clip list
    ("source_clip_id", _I),      # index into the SOURCE library, for joins back
    ("terrain_id", _I),          # manifest tile id; -1 for flat
    ("scene_tile_id", _I),       # the physical tile column the env stood on. Distinct per
                                 # flat tile, unlike terrain_id, which would collapse every
                                 # flat clip into ONE bootstrap cluster.
    ("family_code", _U8),
    ("type_code", _U8),
    ("repeat_idx", _I),          # which of the R draws this was
    ("draw_id", _I),             # comparable across cells at fixed dr_preset; -1 = unknown
    # --- extent: how long it could have lasted -----------------------------------------
    ("clip_len", _I),
    ("start_step", _I),
    ("last_usable_step", _I),    # what clip_end_reached truncates at, lookahead included
    ("steps_survived", _I),
    # --- outcome -----------------------------------------------------------------------
    ("termination_bits", _U16),  # one bit per active term; several can fire on one step
    ("termination_step", _I),
    ("event_observed", _U8),     # 1 = a failure was observed, 0 = censored (survived/cut)
    ("admin_censored", _U8),     # still running when the rollout hit its step ceiling
    ("first_fall_step", _I),     # -1 = never crossed the fall predicate
    # --- the STRICT tracking checks (transfer/deploy/tracking_error.py) -----------------
    #
    # The fall predicate above is reference-FREE: pelvis low, or far from world upright. It
    # says nothing about a robot that stumbles down a flight and barely catches itself, which
    # is exactly the behaviour a selector must not be taught to like. These columns carry the
    # reference-relative checks -- lateral, height and tilt against the SERVED node -- and are
    # all -1 when `attach_tracking` was never called, which `report.py` distinguishes from
    # "measured and never fired" rather than reading as a clean run.
    #
    # LATCHED, NEVER TERMINATING: the episode runs on, so one rollout yields both the strict
    # and the loose verdict and a threshold change afterwards is an offline relabel.
    ("first_violation_step", _I),  # -1 = never; the union of the three below
    ("first_lateral_step", _I),
    ("first_height_step", _I),
    ("first_tilt_step", _I),
    ("track_reward_mean", _F),     # NaN without attach_tracking
    # --- terrain -----------------------------------------------------------------------
    ("has_terrain_span", _U8),
    ("terrain_start_step", _I),  # -1 when there is no pinned span at all
    ("terrain_end_step", _I),
    ("terrain_grace_steps", _I), # <= 0 means the +2 s clause cannot bind for this clip
    ("terrain_end_reached", _U8),
    ("terrain_completed", _U8),
    ("terrain_progress", _F),    # NaN without a pinned span
    # --- the SPATIAL goal (mdp.goal_region) --------------------------------------------
    #
    # Every column ABOVE this block is a clip-CURSOR condition: `steps_survived` against a
    # step index. None of them can tell a robot that climbed the staircase from one that
    # stood still while the cursor ran out, so none of them can carry a success rule for a
    # method that reshapes the generator toward "outcomes that score well" -- standing still
    # would be a trivially reachable optimum. These are the columns that answer "did it
    # physically get there".
    #
    # `reached_goal` is evaluated per STEP against the run's own `goal_regions.json` (an
    # episode that arrives and then wanders back out still arrived). The three raw terminal
    # offsets are stored BESIDE it so a recalibration of the tolerances can be checked, and
    # so `build_goal_regions.py` has something to fit -- that is the whole reason the
    # tolerances are an artifact and not constants.
    ("reached_goal", _U8),
    ("first_goal_step", _I),      # -1 = never entered
    ("min_dist_to_goal_m", _F),   # 0 inside; separates a near-miss timeout from a stall
    ("goal_downrange_m", _F),     # terminal, terrain frame; + = past the clip's own endpoint
    ("goal_lateral_m", _F),       # terminal, terrain frame
    ("goal_height_err_m", _F),    # terminal; + = above. The limb that catches a robot that
                                  # walked off an elevated tile onto the floor downrange of it
    ("max_goal_downrange_m", _F), # furthest downrange reached, the progress analogue
    # --- commanded-velocity tracking ---------------------------------------------------
    ("vel_mse_vx", _F), ("vel_mse_vy", _F), ("vel_mse_wz", _F),
    ("vel_maxabs_vx", _F), ("vel_maxabs_vy", _F), ("vel_maxabs_wz", _F),
    ("cmd_speed_mean", _F),      # without it a velocity error is uninterpretable
    ("n_steps_scored", _I),
    # --- reference tracking ------------------------------------------------------------
    ("ref_base_pos_mse", _F),
    ("ref_base_ori_mse_rad", _F),
    ("ref_base_ori_max_rad", _F),
    ("ref_joint_pos_mse", _F),
    ("ref_ee_pos_mse", _F),
    ("ref_base_lin_vel_mse", _F),
    ("ref_base_ang_vel_mse", _F),
    ("ref_joint_vel_mse", _F),
    ("ref_finite_frac", _F),     # strict_fm_only NaNs whole blocks; make that visible
    ("clf_v_mean", _F),
    # --- robot state -------------------------------------------------------------------
    ("min_base_height_w", _F),
    ("max_tilt_rad", _F),
    ("end_base_height_w", _F),
    ("end_tilt_rad", _F),
    # --- effort / smoothness -----------------------------------------------------------
    ("action_rate_mean", _F),
    ("action_rate_max", _F),
    ("torque_util_max", _F),     # matches gen_tracker_metrics' torques.max_utilization
    ("torque_rms", _F),
    ("mech_work_j", _F),
    ("distance_m", _F),
    ("cost_of_transport", _F),
    ("foot_slip_dist_m", _F),
    ("foot_contact_frac", _F),
    # --- what KIND of motion the generator planned (terrain_class.py) -------------------
    #
    # One classification per RE-PLAN, counted over the episode. Zero on the clip arm, which
    # loads no generator. Read as a RATE against the open-loop control measured over 5
    # by-clip splits -- stairs->jump 0.059 [0.047, 0.071] -- not as an absolute: the
    # classifier's own confusion is most of any small number here.
    ("n_plans", _I),             # re-plans classified; the denominator
    ("n_plans_flat", _I),
    ("n_plans_jump", _I),
    ("n_plans_stairs", _I),
)

EPISODE_COLUMN_NAMES: tuple[str, ...] = tuple(n for n, _ in EPISODE_COLUMNS)

#: Units, for the report's definitions block.  Anything absent is dimensionless.
COLUMN_UNITS: dict[str, str] = {
    "vel_mse_vx": "(m/s)^2", "vel_mse_vy": "(m/s)^2", "vel_mse_wz": "(rad/s)^2",
    "vel_maxabs_vx": "m/s", "vel_maxabs_vy": "m/s", "vel_maxabs_wz": "rad/s",
    "cmd_speed_mean": "m/s",
    "ref_base_pos_mse": "m^2", "ref_base_ori_mse_rad": "rad^2",
    "ref_base_ori_max_rad": "rad", "ref_joint_pos_mse": "rad^2",
    "ref_ee_pos_mse": "m^2", "ref_base_lin_vel_mse": "(m/s)^2",
    "ref_base_ang_vel_mse": "(rad/s)^2", "ref_joint_vel_mse": "(rad/s)^2",
    "min_base_height_w": "m", "end_base_height_w": "m",
    "max_tilt_rad": "rad", "end_tilt_rad": "rad",
    "torque_rms": "N*m", "mech_work_j": "J", "distance_m": "m",
    "foot_slip_dist_m": "m",
    "min_dist_to_goal_m": "m", "goal_downrange_m": "m", "goal_lateral_m": "m",
    "goal_height_err_m": "m", "max_goal_downrange_m": "m",
}

#: The fall predicate applied to the recorded extremes.
#:
#: The height limb matches ``mdp.base_fallen`` exactly. The tilt limb does NOT: ``base_fallen``
#: checks ``|roll|`` and ``|pitch|`` SEPARATELY against 70 deg, while the stored
#: ``max_tilt_rad`` is the total angle from upright, so a pose at roll 50 / pitch 50 is ~70 deg
#: total with neither axis over its limit. The derived flag therefore fires slightly EARLIER
#: than the termination -- measured at 0.054 against 0.045 on the same rollout.
#:
#: The direction is the safe one: derived falls are a SUPERSET of the termination's, so the
#: column never misses a real fall. Quote the ``base_fallen`` share when the exact repo
#: predicate is wanted. Making them identical would mean storing max |roll| and max |pitch|
#: separately rather than one total-tilt column.
#:
#: MuJoCo uses a third rule (``qpos[2] < 0.2``, patched with ``|roll| > 1.0`` in
#: ``compare_sim2sim``), which is why the table stores the extremes rather than only a flag.
FALL_MIN_HEIGHT_M: float = 0.5
FALL_TILT_RAD: float = np.deg2rad(70.0)

#: Seconds the robot must survive PAST the terrain to count as having completed it.
TERRAIN_POST_SECONDS: float = 2.0


def cell_key(reference_mode: str, dr_preset: str, fm_label: str = "-") -> str:
    """Filesystem-safe key for one grid cell.

    Args:
        reference_mode: ``clip``, ``fm``, ``fm_tf`` (the generator teacher-forced -- same
            generator as ``fm``, history from the clip) or ``fm_meas`` (same generator,
            history from the robot's own achieved state). See ``evaluate.REFERENCE_MODES``.
        dr_preset: ``none``, ``train``, ``ood``, ...
        fm_label: The generator's label; ``-`` for the clip arm.

    Returns:
        ``<reference_mode>__<dr_preset>__<fm_label>``.
    """
    return f"{reference_mode}__{dr_preset}__{fm_label or '-'}"


def parse_cell_key(key: str) -> tuple[str, str, str]:
    """Inverse of :func:`cell_key`.

    Raises:
        ValueError: If ``key`` does not have exactly three ``__``-separated parts.
    """
    parts = key.split("__")
    if len(parts) != 3:
        raise ValueError(
            f"cell key {key!r} must be '<reference_mode>__<dr_preset>__<fm_label>'"
        )
    return parts[0], parts[1], parts[2]


def last_usable_step(clip_len: int, max_offset: int) -> int:
    """The step a SURVIVING episode ends on, lookahead reserve included.

    ``clip_end_reached(reserve_lookahead=True)`` fires at ``clip_len - 1 - max_offset``, so
    the achievable maximum is that, not ``clip_len - 1``.  Every fraction and every terrain
    gate has to use this as its denominator or a perfect episode reads as a near-miss.

    Args:
        clip_len: The clip's real frame count before repeat-last padding.
        max_offset: ``max(REFERENCE_OFFSETS)`` -- the observation window's furthest sample.

    Returns:
        The last usable step index, never below 0 (mirroring the termination's own
        ``clamp_min(0)`` for clips shorter than the window).
    """
    return max(int(clip_len) - 1 - int(max_offset), 0)


def terrain_columns(
    steps_survived: np.ndarray,
    *,
    terrain_start_step: np.ndarray,
    terrain_end_step: np.ndarray,
    last_usable: np.ndarray,
    dt: float,
    post_seconds: float = TERRAIN_POST_SECONDS,
) -> dict[str, np.ndarray]:
    """The three terrain outcomes, plus the grace window each clip actually had.

    Why three and not one: the binary "got through the terrain" is **identical to success**
    on four of the nine families, for two opposite reasons.  ``flat`` has no terrain-pinned
    frame at all, so ``terrain_end_step`` falls back to ``clip_len - 1``; the ``*_pure`` and
    ``distractor`` families are pinned all the way to their last frame, because their clips
    end *on* the terrain.  Only the five families whose definition includes an exit segment
    ("walk in, climb, walk off the top platform") have a run-out for the metric to measure.
    So:

    * ``terrain_end_reached`` -- survived past the pinned span.  Informative on those five.
    * ``terrain_completed`` -- also survived the grace window.  Mostly equals success.
    * ``terrain_progress`` -- how far ALONG the pinned span the robot got before stopping.
      Informative on the four where the binaries are no-ops, and the one a paper wants.

    ``terrain_progress`` is denominated by the REACHABLE span, ``min(terrain_end,
    last_usable) - terrain_start``, not the full span.  With the full span a successful
    episode on a family that ends on the terrain scores 0.83-0.91 rather than 1.0, because
    the episode stops at ``last_usable`` which is before ``terrain_end`` -- so pooling an
    ECDF over clips would smear the success point-mass across [0.83, 1.0] and read as "a
    spread of near-completions" when it is only the censoring boundary moving.

    Args:
        steps_survived: ``[n]`` steps completed per episode.
        terrain_start_step: ``[n]`` first terrain-pinned frame; ``-1`` for none.
        terrain_end_step: ``[n]`` last terrain-pinned frame.
        last_usable: ``[n]`` from :func:`last_usable_step`.
        dt: Control step, seconds.
        post_seconds: Grace window after the terrain.

    Returns:
        ``has_terrain_span``, ``terrain_grace_steps``, ``terrain_end_reached``,
        ``terrain_completed``, ``terrain_progress``.
    """
    steps = np.asarray(steps_survived, dtype=np.int64)
    t0 = np.asarray(terrain_start_step, dtype=np.int64)
    t1 = np.asarray(terrain_end_step, dtype=np.int64)
    lu = np.asarray(last_usable, dtype=np.int64)
    post = int(round(float(post_seconds) / float(dt)))

    has_span = t0 >= 0
    grace = np.minimum(post, lu - t1)
    end_reached = steps >= t1
    completed = steps >= np.minimum(t1 + post, lu)

    reach_end = np.minimum(t1, lu)
    span = np.maximum(reach_end - t0, 1)
    prog = (np.minimum(steps, reach_end) - t0) / span
    prog = np.clip(prog, 0.0, 1.0).astype(np.float32)
    prog = np.where(has_span, prog, np.float32("nan"))

    return {
        "has_terrain_span": has_span.astype(np.uint8),
        "terrain_grace_steps": grace.astype(np.int32),
        "terrain_end_reached": end_reached.astype(np.uint8),
        "terrain_completed": completed.astype(np.uint8),
        "terrain_progress": prog,
    }


def completion_is_informative(terrain_end_step: np.ndarray, last_usable: np.ndarray) -> np.ndarray:
    """Whether ``terrain_completed`` says anything beyond ``success``, per episode.

    False when the pinned span runs to (or past) the last usable step: the completion test
    then reduces to "reached the end of the episode", which IS success.  The reducer prints
    a dash rather than a number for a family where this is false everywhere -- a rate that
    is silently the success rate reads, in a table, as an independent result.
    """
    return (np.asarray(terrain_end_step, dtype=np.int64)
            < np.asarray(last_usable, dtype=np.int64))


def fell(min_base_height_w: np.ndarray, max_tilt_rad: np.ndarray,
         *, min_height: float = FALL_MIN_HEIGHT_M,
         tilt_rad: float = FALL_TILT_RAD) -> np.ndarray:
    """Derived fall flag at a stated threshold.

    Derived rather than stored, and deliberately NOT added as a termination term: the
    teacher task has no fall detector, and adding one would change *when* episodes end (a
    fall could pre-empt the clip-graded terminations) and make runs incomparable.  Recording
    the height and tilt extremes instead lets both this predicate and MuJoCo's be applied
    offline to the same rollout.
    """
    return ((np.asarray(min_base_height_w) < min_height)
            | (np.asarray(max_tilt_rad) > tilt_rad))


def first_bad_step(cols: dict, *, strict: bool = True) -> np.ndarray:
    """Step of the first FAILURE event, or -1 -- the one input the success rule needs.

    Whichever of the fall crossing and the strict tracking violation happened first.  ``-1``
    on either limb means "never", so this is a min over the non-negative entries and not a
    plain ``np.minimum``, which would return -1 the moment either limb never fired.

    Args:
        cols: Columns from a table read.
        strict: Include the tracking violation.  ``False`` reproduces the LOOSE rule -- the
            fall crossing alone -- so both verdicts come off one rollout and every number
            published before the strict checks existed stays reproducible from the same data.

    Returns:
        ``[N]`` int64.
    """
    fall = np.asarray(cols["first_fall_step"], dtype=np.int64)
    if not strict or "first_violation_step" not in cols:
        return fall
    viol = np.asarray(cols["first_violation_step"], dtype=np.int64)
    big = np.iinfo(np.int64).max
    out = np.minimum(np.where(fall < 0, big, fall), np.where(viol < 0, big, viol))
    return np.where(out == big, -1, out).astype(np.int64)


def arrived_before_failing(cols: dict, *, strict: bool = True) -> np.ndarray:
    """**The success rule, stated once for the whole tree.**

    An episode succeeds when its base entered the goal region BEFORE it first failed.  Falling,
    violating the tracking checks, and timing out are all failures; arriving and then falling in
    the run-out is a success, because the task was done.

    This is ``rollout_outcomes.terminal_reward``'s rule, and it is now also ``report.py``'s.
    The two used to disagree -- ``report.py`` failed any episode whose fall TERMINATION bit was
    set regardless of ordering, which scored arrive-then-fall as a failure on 51 of 298 episodes
    of one measured rollout, while ``terminal_reward`` scored it a success. A harness cannot
    compare two arms under two success rules, and this stack had thirty.

    Censoring is NOT handled here: an admin-censored episode has no outcome at all, and callers
    drop it rather than scoring it either way.

    Args:
        cols: Columns from a table read; needs ``reached_goal``, ``first_goal_step`` and
            ``first_fall_step`` (plus ``first_violation_step`` when ``strict``).
        strict: Count a tracking violation as a failure.  ``False`` is the loose rule.

    Returns:
        ``[N]`` bool.
    """
    arrived = np.asarray(cols["reached_goal"]).astype(bool)
    goal = np.asarray(cols["first_goal_step"], dtype=np.int64)
    bad = first_bad_step(cols, strict=strict)
    return arrived & ((bad < 0) | (goal >= 0) & (goal < bad))


#: HDF5's own chunk cache defaults to 1 MiB, so a chunk far above that is read straight
#: through it on every access.  Same reasoning (and the same number) as
#: ``recorded_data_io._CHUNK_TARGET_BYTES``.
_CHUNK_TARGET_BYTES: int = 4 << 20
#: Row cap per chunk regardless of width, so a 1-byte column does not get a 4M-row chunk.
_CHUNK_MAX_ROWS: int = 8192


def _chunk_rows(itemsize: int, capacity: int) -> int:
    """Rows per chunk for a 1-D column of ``itemsize`` bytes."""
    by_bytes = max(1, _CHUNK_TARGET_BYTES // max(1, int(itemsize)))
    return int(max(1, min(_CHUNK_MAX_ROWS, by_bytes, max(1, capacity))))


class EpisodeTableWriter:
    """Streaming writer for the per-episode table: one 1-D dataset per column.

    Columnar rather than a compound dtype so the reduce side can read three columns out of
    fifty-seven without touching the rest, which is what keeps the plot stage cheap.

    Datasets are resizable and the row count is stamped at :meth:`close`, so a killed
    rollout leaves a readable, correctly-trimmed file rather than a tail of zeros that would
    read as a pile of instant failures.  The cell manifest rides along as per-field JSON root
    attributes, making the file self-describing without its sidecar -- the same contract
    ``recorded_data_io.RecordedWindowWriter`` uses.
    """

    def __init__(self, path: str, capacity: int, meta: dict | None = None,
                 compression: str | None = "lzf") -> None:
        """Open the file and create every column.

        Args:
            path: Output ``.h5``.  Overwritten.
            capacity: Expected row count.  Only sizes the chunks; the datasets grow past it
                if needed (administratively censored episodes can push past the estimate).
            meta: Cell manifest, stored as per-field JSON root attrs.
            compression: h5py filter; ``lzf`` is fast and these columns compress well.
        """
        import h5py  # noqa: PLC0415

        self.path = str(path)
        self._n = 0
        self._f = h5py.File(self.path, "w")
        for name, dtype in EPISODE_COLUMNS:
            dt = np.dtype(dtype)
            self._f.create_dataset(
                name, shape=(0,), maxshape=(None,), dtype=dt,
                chunks=(_chunk_rows(dt.itemsize, capacity),), compression=compression,
            )
        self._f.attrs["schema_version"] = SCHEMA_VERSION
        self._f.attrs["schema_minor"] = SCHEMA_MINOR
        self._f.attrs["columns"] = json.dumps(list(EPISODE_COLUMN_NAMES))
        for key, value in (meta or {}).items():
            self.set_attr(key, value)

    def set_attr(self, name: str, value) -> None:
        """Store one manifest field as a JSON root attribute."""
        self._f.attrs[name] = json.dumps(value, default=str)

    def append(self, rows: dict[str, np.ndarray]) -> int:
        """Append a batch of episodes.

        Args:
            rows: One entry per column in :data:`EPISODE_COLUMNS`, all the same length.

        Returns:
            The number of rows appended.

        Raises:
            KeyError: If a column is missing or unknown -- every column must be supplied on
                every call, so a column added to the schema and forgotten here fails loudly
                instead of writing zeros.
            ValueError: If the supplied columns have differing lengths.
        """
        missing = [n for n in EPISODE_COLUMN_NAMES if n not in rows]
        extra = [n for n in rows if n not in set(EPISODE_COLUMN_NAMES)]
        if missing or extra:
            raise KeyError(
                f"episode batch has missing={missing} unknown={extra}; every column in "
                "EPISODE_COLUMNS must be supplied on every append"
            )
        lengths = {n: int(np.asarray(v).shape[0]) for n, v in rows.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"episode batch columns have differing lengths: {lengths}")
        k = next(iter(lengths.values()))
        if k == 0:
            return 0
        for name, dtype in EPISODE_COLUMNS:
            ds = self._f[name]
            ds.resize((self._n + k,))
            ds[self._n:self._n + k] = np.asarray(rows[name], dtype=np.dtype(dtype))
        self._n += k
        return k

    @property
    def num_written(self) -> int:
        """Rows written so far."""
        return self._n

    def close(self) -> None:
        """Stamp the row count and close."""
        if self._f is not None:
            self._f.attrs["num_episodes_written"] = self._n
            self._f.close()
            self._f = None

    def __enter__(self) -> "EpisodeTableWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_episode_table(path: str, columns: list[str] | None = None
                       ) -> tuple[dict[str, np.ndarray], dict]:
    """Read an episode table and its manifest.

    Args:
        path: The ``episodes.h5`` to read.
        columns: Columns to load; ``None`` loads all of them.

    Returns:
        ``(data, meta)``.  ``meta`` holds the JSON-decoded root attrs plus
        ``num_episodes_written``.

    Raises:
        ValueError: If the file's ``schema_version`` differs from :data:`SCHEMA_VERSION`
            (a silent read under the wrong contract is worse than a refusal), or if a
            requested column is absent.
    """
    import h5py  # noqa: PLC0415

    with h5py.File(str(path), "r") as f:
        version = int(f.attrs.get("schema_version", -1))
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"{path} was written under schema_version {version}, but this code "
                f"implements {SCHEMA_VERSION}. Columns may have changed meaning; re-run the "
                "cell or check out the commit that wrote it."
            )
        n = int(f.attrs.get("num_episodes_written", 0))
        want = list(columns) if columns is not None else list(EPISODE_COLUMN_NAMES)
        absent = [c for c in want if c not in f]
        if absent:
            raise ValueError(f"{path} has no column(s) {absent}")
        data = {c: f[c][:n] for c in want}
        meta: dict = {}
        for key, raw in f.attrs.items():
            if key in ("schema_version", "schema_minor", "num_episodes_written"):
                continue
            try:
                meta[key] = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
            except (json.JSONDecodeError, TypeError):
                meta[key] = raw
        meta["schema_version"] = version
        meta["num_episodes_written"] = n
    return data, meta
