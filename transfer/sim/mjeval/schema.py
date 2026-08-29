# Copyright 2026 Zachary Olkin. All rights reserved.

"""The MuJoCo eval artifact contract: columns, the h5 writer/reader, and the success rule.

Shared by the run side (which fills it) and the report side (which reads it), so a column name
is spelled exactly once.  Pure numpy + h5py: no torch, no IsaacLab, no matplotlib, no mujoco.

The design rules are ``scripts/rsl_rl/eval/schema.py``'s, and they are repeated here because
each one exists because the obvious alternative produces a wrong statistic:

* **Store primitives, derive the rest.**  ``success`` is a contested definition, so the table
  stores ``reached_target`` and ``traversed`` separately and the reducer derives success from a
  NAMED rule it prints (:func:`derive_success`).  ``event_observed`` is likewise not a column --
  ``report.py`` forms it from the verdict, so there is one place that decides what counts as an
  event.
* **No ``time_to_failure_s``.**  It is ``steps_survived * dt`` for failures and undefined for
  successes, and the first ``nanmean`` over such a column silently conditions on failing --
  the bias that once made a policy surviving only its three easiest envs look like the best
  tracker in the repo.  Stored instead as ``steps_survived`` plus a derived event flag, the
  standard survival pair.
* **MSE, not RMSE.**  Means of per-episode RMSE are biased low by Jensen, and a combined
  ``vel_rmse_lin = sqrt(mse_vx + mse_vy)`` needs the squares.  The root is taken at the join.
* **Enum codes plus a name table**, so the file carries no variable-length strings and every
  consumer resolves names the same way.

**Five Isaac columns are deliberately absent, because here they are derivable rather than
measured.**  Saying so matters: their absence is a property of this design, not an omission.
``cmd_speed_mean`` equals ``cmd_vx`` (the command is a per-episode constant, where Isaac's
varied within a clip); ``n_steps_scored`` equals ``steps_survived`` (every step is scored,
there being no clip to run out of); ``episode_time_s`` is ``steps_survived * dt``; and
``budget_s``, ``target_x/y``, the goal half-width and ``course_length_m`` are properties of the
COURSE, carried once in the run manifest and joined on ``terrain_idx`` rather than repeated on
all ten of its rows.

**The verdict is three-way and mutually exclusive**, because the runner checks
``robot.failed()`` before the target test:

* ``success`` -- the root entered the goal rectangle AND the traversal gate passed.
* ``fall``    -- the height or tilt predicate fired first.  This is the only EVENT; success and
                 timeout are right-CENSORED, which is what makes Kaplan-Meier valid on
                 ``progress_frac``.
* ``timeout`` -- neither happened before the course's budget elapsed.  A robot that drifts wide
                 and misses the course entirely lands here rather than in ``fall``, which is
                 the distinction the whole verdict split exists to make.
"""

from __future__ import annotations

import datetime
import json
import os
import sys

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# Both this directory (so `import terrains` resolves whether this module is loaded as
# `mjeval.schema` or bare `schema`) and its parent (the bootstrap `terrains.py` itself needs,
# to reach `generate_stairs_scene`).
for _d in (_HERE, os.path.dirname(_HERE)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import terrains  # noqa: E402

#: Bump on any column removal, rename or change of meaning.  The reducer refuses a file it
#: does not know rather than reading a renamed column as its old self.
SCHEMA_VERSION: int = 1
#: Bump on a purely additive column.  A reader may ignore columns it does not know.
SCHEMA_MINOR: int = 0

_F = "f4"
_I = "i4"
_U8 = "u1"

#: ``(name, dtype)`` for every per-episode column, in write order.  Grouped by what answers
#: which question; the groups are not stored separately.  Spellings match
#: ``scripts/rsl_rl/eval/schema.py`` wherever the quantity is the same, so the two harnesses'
#: tables stay diffable rather than merely similar.
MJ_EPISODE_COLUMNS: tuple[tuple[str, str], ...] = (
    # --- identity: what ran ------------------------------------------------------------
    ("episode_id", _I),
    ("arm", _U8),                 # 0 = baseline pair, 1 = fine-tuned pair
    ("seed", _I),                 # FM sampler seed; -1 under zero_noise (the deployed sampler)
    ("terrain_idx", _I),          # 0..19, index into terrains.catalogue()
    ("terrain_class", _U8),       # terrains.CLASS_CODES
    ("variant", _U8),             # index within the class
    ("speed_idx", _U8),           # index into the course's own ladder, so the grid joins
                                  # without a float compare
    ("cmd_vx", _F), ("cmd_vy", _F), ("cmd_wz", _F),
    # --- outcome -----------------------------------------------------------------------
    ("steps_survived", _I),
    ("verdict", _U8),             # terrains.VERDICT_CODES
    ("failure_code", _U8),        # terrains.FAILURE_CODES -- which predicate fired
    ("first_fall_step", _I),      # -1 = never crossed the fall predicate
    # --- target ------------------------------------------------------------------------
    ("reached_target", _U8),
    ("time_to_target_s", _F),     # NaN if never
    ("traversed", _U8),           # cleared/dropped the flight, i.e. did not walk around it
    ("min_dist_to_target_m", _F), # separates a near-miss timeout from never leaving the spawn
    # --- course progress (the terrain_progress replacement) ----------------------------
    ("progress_x_m", _F),         # max downrange x reached
    ("progress_frac", _F),        # progress_x_m / course_length_m
    ("distance_m", _F),           # planar path length actually walked
    # --- commanded-velocity tracking ---------------------------------------------------
    # The headline three are YAW-frame; the two `_body` are full-body-frame, kept only so the
    # numbers stay comparable with Isaac's `accumulate.fold`, which uses `root_lin_vel_b`.
    ("vel_mse_vx", _F), ("vel_mse_vy", _F), ("vel_mse_wz", _F),
    ("vel_mse_vx_body", _F), ("vel_mse_vy_body", _F),
    ("vel_mse_vx_pre", _F),       # before the first riser / box face -- the honest tracking
    ("vel_mse_vx_post", _F),      # on the course, where deviation is correct behaviour
    # --- robot state -------------------------------------------------------------------
    ("min_base_height_w", _F),
    ("max_base_height_w", _F),    # what the traversal gate is read against
    ("max_tilt_rad", _F),
    ("end_base_height_w", _F),
    ("end_tilt_rad", _F),
    # --- effort / smoothness -----------------------------------------------------------
    ("action_rate_mean", _F), ("action_rate_max", _F),
    ("torque_util_max", _F), ("torque_rms", _F),
    ("mech_work_j", _F), ("cost_of_transport", _F),
    # --- what the generator PLANNED (terrain_class.py, Isaac names verbatim) -----------
    ("n_plans", _I), ("n_plans_flat", _I), ("n_plans_jump", _I), ("n_plans_stairs", _I),
    # --- timing ------------------------------------------------------------------------
    ("fm_ms_mean", _F), ("trk_ms_mean", _F),
)

#: Column names alone, in write order.
COLUMN_NAMES: tuple[str, ...] = tuple(name for name, _ in MJ_EPISODE_COLUMNS)

#: Units for every dimensioned column, so a reader never has to guess whether a velocity
#: error is an MSE or an RMSE.
UNITS: dict[str, str] = {
    "cmd_vx": "m/s", "cmd_vy": "m/s", "cmd_wz": "rad/s",
    "time_to_target_s": "s", "min_dist_to_target_m": "m",
    "progress_x_m": "m", "progress_frac": "-", "distance_m": "m",
    "vel_mse_vx": "(m/s)^2", "vel_mse_vy": "(m/s)^2", "vel_mse_wz": "(rad/s)^2",
    "vel_mse_vx_body": "(m/s)^2", "vel_mse_vy_body": "(m/s)^2",
    "vel_mse_vx_pre": "(m/s)^2", "vel_mse_vx_post": "(m/s)^2",
    "min_base_height_w": "m", "max_base_height_w": "m",
    "end_base_height_w": "m", "max_tilt_rad": "rad", "end_tilt_rad": "rad",
    "action_rate_mean": "1/s", "action_rate_max": "1/s",
    "torque_util_max": "-", "torque_rms": "N.m", "mech_work_j": "J",
    "cost_of_transport": "-", "fm_ms_mean": "ms", "trk_ms_mean": "ms",
}


def _name_table(codes: dict[str, int]) -> list[str]:
    """Invert a ``{name: code}`` map into a positional list for storage on the file.

    Args:
        codes: The enum, e.g. :data:`terrains.VERDICT_CODES`.

    Returns:
        ``out[code] == name`` for every code.

    Raises:
        ValueError: When the codes are not exactly ``0..n-1``.  A gap would make the
            positional list lie about which name a stored code means.
    """
    vals = sorted(codes.values())
    if vals != list(range(len(vals))):
        raise ValueError(f"enum codes must be contiguous from 0, got {sorted(codes.items())}")
    out = [""] * len(vals)
    for name, code in codes.items():
        out[code] = name
    return out


class EpisodeTable:
    """Append-only accumulator for episode rows, written once at the end of a shard.

    Held in memory because the whole grid is 200 rows per arm of 45 float/int columns -- tens
    of kilobytes -- so incremental h5 appends would buy nothing and cost a partially written
    file on an interrupt.
    """

    def __init__(self) -> None:
        """Start an empty table."""
        self._rows: list[dict] = []

    def append(self, row: dict) -> None:
        """Add one episode.

        Args:
            row: A dict whose keys are exactly :data:`COLUMN_NAMES`.

        Raises:
            ValueError: When a key is missing or unexpected, naming both sets.  Validating
                here rather than at write time means the failure points at the accumulator
                that produced the row rather than at the end of a half-hour sweep.
        """
        want, got = set(COLUMN_NAMES), set(row)
        if want != got:
            raise ValueError(
                f"episode row does not match the schema: missing {sorted(want - got)}, "
                f"unexpected {sorted(got - want)}")
        self._rows.append(dict(row))

    def __len__(self) -> int:
        """Number of episodes appended so far."""
        return len(self._rows)

    def write(self, path: str, attrs: dict) -> None:
        """Write the table plus its provenance to an HDF5 file.

        Args:
            path: Destination ``.h5``.  Parent directories are created.
            attrs: Run provenance (artifact paths, hashes, git state, resolved config).  Every
                value is JSON-encoded, so nested dicts survive the round trip.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with h5py.File(path, "w") as f:
            for name, dtype in MJ_EPISODE_COLUMNS:
                vals = [r[name] for r in self._rows]
                f.create_dataset(name, data=np.asarray(vals, dtype=dtype))
            f.attrs["schema_version"] = SCHEMA_VERSION
            f.attrs["schema_minor"] = SCHEMA_MINOR
            f.attrs["columns"] = json.dumps(list(COLUMN_NAMES))
            f.attrs["units"] = json.dumps(UNITS)
            f.attrs["class_names"] = json.dumps(_name_table(terrains.CLASS_CODES))
            f.attrs["verdict_names"] = json.dumps(_name_table(terrains.VERDICT_CODES))
            f.attrs["failure_names"] = json.dumps(_name_table(terrains.FAILURE_CODES))
            f.attrs["written_at"] = datetime.datetime.now(datetime.UTC).isoformat()
            for key, value in (attrs or {}).items():
                f.attrs[key] = json.dumps(value, default=str)


def read_table(path: str) -> tuple[dict[str, np.ndarray], dict]:
    """Read an episode table back.

    Args:
        path: The ``.h5`` written by :meth:`EpisodeTable.write`.

    Returns:
        ``(columns, attrs)``.  Every attribute that was JSON-encoded comes back decoded; the
        two integer version attributes come back as ints.

    Raises:
        ValueError: On a ``schema_version`` this module does not know.  A silently-read file
            from a future major would mean reading a renamed column as its old self.
    """
    cols: dict[str, np.ndarray] = {}
    attrs: dict = {}
    with h5py.File(path, "r") as f:
        version = int(f.attrs.get("schema_version", -1))
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"{path} is schema_version {version}; this reader knows {SCHEMA_VERSION}. "
                "Re-run the sweep rather than reading it through the wrong contract.")
        for key in f.keys():
            cols[key] = f[key][...]
        for key, raw in f.attrs.items():
            if key in ("schema_version", "schema_minor"):
                attrs[key] = int(raw)
                continue
            try:
                attrs[key] = json.loads(raw)
            except (TypeError, ValueError):
                attrs[key] = raw
    return cols, attrs


def derive_success(cols: dict[str, np.ndarray]) -> np.ndarray:
    """The success rule, stated once so the report can print it verbatim.

    **An episode succeeds when its root entered the goal rectangle AND the traversal gate
    passed.**  Both halves are needed: the courses are finite in lateral extent while the floor
    is not, so a robot that drifts wide reaches the target without ever touching the terrain,
    and the gate -- 80% of the flight height, in the course's own direction -- is what excludes
    it.  The runner tests ``robot.failed()`` before the target every step, so an episode that
    face-plants across the line is a fall and is already excluded by ``verdict``.

    Args:
        cols: Columns from :func:`read_table`.

    Returns:
        Float array, 1.0 where the episode succeeded and 0.0 elsewhere.  Float rather than
        bool because every downstream estimator averages it.
    """
    verdict = np.asarray(cols["verdict"])
    ok = (verdict == terrains.VERDICT_CODES["success"])
    ok &= np.asarray(cols["reached_target"]).astype(bool)
    ok &= np.asarray(cols["traversed"]).astype(bool)
    return ok.astype(float)
