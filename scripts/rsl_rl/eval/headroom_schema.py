# Copyright 2026 Zachary Olkin. All rights reserved.

"""Record layout for the state-level best-of-K headroom test.

One row per ``(state, candidate)`` -- exploded rather than nested, because every table in
``headroom_report`` is a groupby over states and the K-wide form would have to be melted
first anyway.  At the planned budget (10k states x K=16) that is 160k rows of ~40 columns,
which is small enough that the raw file is the artifact and every statistic is derived from
it.

**``graded_score`` is deliberately NOT a column here.**  Its composition needs normalizers
(what counts as a large base-position error?) and picking those before seeing the data would
bake a guess into the raw record.  The COMPONENTS are stored; ``headroom_report`` divides
each by its own median over the run, sums, and records the divisors.  So the weighting is a
reporting decision that can be redone against the same file, which is exactly what §7.1 of
the spec asks for.

**The parent's own rollout is a row too**, at ``candidate_idx = PARENT_ROW`` with
``is_parent_rollout = 1``.  Candidate 0 is handed the parent's own flow-matching noise draw
and the same continuation seed, so that row and the parent row are the SAME computation up
to the state restore -- their difference is the harness noise floor, measured at every
branched state on the real state distribution instead of in a synthetic determinism test.
Filter it out (``is_parent_rollout == 0``) for every best-of-K statistic.
"""

from __future__ import annotations

import numpy as np

#: Bumped whenever a column is added, removed or re-meaning-ed.  ``headroom.py`` refuses a
#: run directory written by a different one rather than reading columns positionally.
SCHEMA_VERSION: int = 1

#: ``candidate_idx`` sentinel for the parent env's own un-restored continuation.
PARENT_ROW: int = 255

#: Phase of the clip the branched state sits in.  Derived from the clip step against the
#: clip's pinned terrain span, so it is a property of WHERE the robot is, not of how it got
#: there -- which is what lets a mid-feature reset and a survived approach share a bucket.
PHASES: tuple[str, ...] = ("no_terrain", "approach", "early_feature", "mid_feature", "exit")
PHASE_CODES: dict[str, int] = {p: i for i, p in enumerate(PHASES)}

#: How the parent episode's start step was drawn.  Both are used (spec §5); the label is
#: recorded so results can be stratified rather than pooled.
RESET_SOURCES: tuple[str, ...] = ("sampled", "pinned")
RESET_SOURCE_CODES: dict[str, int] = {s: i for i, s in enumerate(RESET_SOURCES)}

#: Which commit rule drove the outer rollout.
#:
#: * ``random``   -- the parent runs free and is NEVER restored; branch stride = the whole
#:   window.  Statistically identical to deployment, so its states are the deployed
#:   distribution and its Table 4 ties out against ``report.py``'s ``fm`` cell.
#: * ``argmax``   -- commit the best-scoring candidate's one-chunk-in state, every chunk.  A
#:   deployable oracle; its episode success upper-bounds one round of sharpening.
#: * ``random_k`` -- commit a UNIFORMLY RANDOM candidate, every chunk.  The control for
#:   ``argmax``: same restore cadence, same number of restores, same everything except which
#:   candidate wins.  Without it an ``argmax`` episode-success number cannot be separated
#:   from the cost of restoring the parent 31 times an episode -- which the same-noise
#:   control measured as the whole of the per-window best-of-K rise.
#: ``random_k`` is APPENDED, never inserted: the codes are stored in every existing record
#: file, so reordering this tuple would silently re-label already-written runs.
COMMIT_MODES: tuple[str, ...] = ("random", "argmax", "random_k")
COMMIT_MODE_CODES: dict[str, int] = {m: i for i, m in enumerate(COMMIT_MODES)}

#: The per-candidate error channels, in the order ``accumulate.channel_groups`` names them.
#: Each is an RMS over the scored window, against the branch's OWN generator reference --
#: never against a clip and never against a nominal stance (spec §4.2).  ``base_ori`` is a
#: geodesic angle, not a component-wise quaternion difference.
ERROR_CHANNELS: tuple[str, ...] = (
    "base_pos", "base_ori", "joint_pos", "ee_pos", "base_lin_vel", "base_ang_vel",
)

#: Cost channels that are not tracking errors.  Kept separate because they are bounded
#: fractions or absolute physical quantities and their medians are not comparable to a
#: tracking RMS.
COST_CHANNELS: tuple[str, ...] = (
    "undesired_contact_frac", "action_rate", "torque_util",
)

STATE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("state_id", "int64"),
    ("episode_id", "int64"),
    ("group_id", "int32"),
    ("branch_step", "int32"),
    ("t_step", "int32"),
    ("clip_step", "int32"),
    ("clip_id", "int32"),
    ("source_clip_id", "int32"),
    ("family_code", "uint8"),
    ("terrain_id", "int32"),
    ("scene_tile_id", "int32"),
    ("start_step", "int32"),
    ("terrain_start_step", "int32"),
    ("terrain_end_step", "int32"),
    ("last_usable_step", "int32"),
    ("reset_source", "uint8"),
    ("phase_code", "uint8"),
    ("commit_mode_code", "uint8"),
    ("state_seed", "int64"),
)

CANDIDATE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("candidate_idx", "uint8"),
    ("is_parent_rollout", "uint8"),
    ("survived", "uint8"),
    ("truncated", "uint8"),
    ("window_steps", "int32"),
    ("steps_before_term", "int32"),
    ("time_to_term_s", "float32"),
    ("termination_bits", "uint16"),
    ("terminal_airborne", "uint8"),
    ("min_height_margin_m", "float32"),
    ("max_tilt_rad", "float32"),
    ("foot_contact_frac", "float32"),
)

#: The full column list: state identity, then the candidate outcome, then the error and cost
#: channels expanded from the two tuples above.  Assembled rather than written out so a new
#: channel cannot be added to one place and forgotten in the other.
COLUMNS: tuple[tuple[str, str], ...] = (
    STATE_COLUMNS
    + CANDIDATE_COLUMNS
    + tuple((f"err_{c}", "float32") for c in ERROR_CHANNELS)
    + tuple((c, "float32") for c in COST_CHANNELS)
)

COLUMN_NAMES: tuple[str, ...] = tuple(n for n, _ in COLUMNS)


def phase_code(clip_step, terrain_start_step, terrain_end_step) -> np.ndarray:
    """Label where in the clip's terrain span a state sits.

    Split at the MIDPOINT of the pinned span rather than into thirds: on the shortest
    families the span is a handful of frames and a three-way split puts single frames in
    their own bucket, which reads in a table as a phase with n=3.

    Args:
        clip_step: Absolute clip step of the branched state (``start_step + t_step``).
        terrain_start_step: First pinned terrain frame, or negative for a flat clip.
        terrain_end_step: Last pinned terrain frame.

    Returns:
        ``uint8`` codes into :data:`PHASES`.
    """
    s = np.asarray(clip_step, dtype=np.int64)
    t0 = np.asarray(terrain_start_step, dtype=np.int64)
    t1 = np.asarray(terrain_end_step, dtype=np.int64)
    mid = t0 + np.maximum(t1 - t0, 0) // 2
    out = np.full(s.shape, PHASE_CODES["approach"], dtype=np.uint8)
    out = np.where(s >= t0, PHASE_CODES["early_feature"], out).astype(np.uint8)
    out = np.where(s > mid, PHASE_CODES["mid_feature"], out).astype(np.uint8)
    out = np.where(s > t1, PHASE_CODES["exit"], out).astype(np.uint8)
    # A flat clip has no span at all (terrain_start_step < 0), and the comparisons above
    # would have put every one of its states in "exit". Applied last so it wins.
    return np.where(t0 < 0, PHASE_CODES["no_terrain"], out).astype(np.uint8)


class RecordWriter:
    """Streaming columnar HDF5 writer for the ``(state, candidate)`` table.

    One 1-D dataset per column, resizable, so the reduce side can read three columns out of
    forty.  Same shape as ``schema.EpisodeTableWriter`` in the episode harness, but
    resizable rather than pre-sized: the state count is not known up front here (it depends
    on how long each episode survives), and a capacity guess that ran short would truncate
    the run.
    """

    def __init__(self, path: str, *, meta: dict) -> None:
        """Create the file and its datasets.

        Args:
            path: Output ``.h5`` path.
            meta: JSON-serialisable run metadata, stored as a single attribute.
        """
        import json  # noqa: PLC0415

        import h5py  # noqa: PLC0415

        self.path = str(path)
        self._f = h5py.File(self.path, "w")
        self._f.attrs["schema_version"] = SCHEMA_VERSION
        self._f.attrs["meta"] = json.dumps(meta, default=str)
        self._f.attrs["phases"] = list(PHASES)
        self._f.attrs["commit_modes"] = list(COMMIT_MODES)
        self._f.attrs["reset_sources"] = list(RESET_SOURCES)
        for name, dtype in COLUMNS:
            self._f.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype,
                                   chunks=(4096,), compression="lzf")
        self.num_written = 0

    def append(self, rows: dict[str, np.ndarray]) -> None:
        """Append one batch.

        Args:
            rows: Column name -> 1-D array, every column of :data:`COLUMNS` present and all
                the same length.

        Raises:
            KeyError: If a column is missing, naming it -- a zero-fill here would read
                downstream as a measurement.
        """
        missing = [n for n in COLUMN_NAMES if n not in rows]
        if missing:
            raise KeyError(f"RecordWriter.append is missing columns: {missing}")
        n = int(len(rows[COLUMN_NAMES[0]]))
        if n == 0:
            return
        for name, _ in COLUMNS:
            dset = self._f[name]
            dset.resize((self.num_written + n,))
            dset[self.num_written:] = np.asarray(rows[name])
        self.num_written += n

    def close(self) -> None:
        """Flush and close."""
        self._f.attrs["num_rows"] = self.num_written
        self._f.close()


def read_records(path: str, columns: list[str] | None = None) -> dict[str, np.ndarray]:
    """Read the table back, optionally a subset of columns.

    Args:
        path: The ``.h5`` written by :class:`RecordWriter`.
        columns: Column names to read, or ``None`` for all of them.

    Returns:
        Column name -> array, plus ``"meta"`` holding the parsed run metadata.

    Raises:
        SystemExit: If the file was written by a different schema version.
    """
    import json  # noqa: PLC0415

    import h5py  # noqa: PLC0415

    with h5py.File(path, "r") as f:
        ver = int(f.attrs.get("schema_version", -1))
        if ver != SCHEMA_VERSION:
            raise SystemExit(
                f"{path} is headroom schema_version {ver}, but this code implements "
                f"{SCHEMA_VERSION}. Re-run --stage run."
            )
        want = list(columns) if columns else list(COLUMN_NAMES)
        out = {n: f[n][:] for n in want}
        out["meta"] = json.loads(f.attrs["meta"])
    return out
