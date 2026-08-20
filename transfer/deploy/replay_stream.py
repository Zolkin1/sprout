# Copyright 2026 Zachary Olkin. All rights reserved.

"""A per-tick recording of one runtime's inputs and outputs, for cross-language replay.

WHY THIS EXISTS. The async C++ obelisk stack tracks worse than the synchronous python
sim2sim on the same artifact pair, and two very different causes produce that: the C++
computing something different, or the C++ being fed something different (staleness, jitter).
Nothing separated them. This records every tick's INPUTS from a python run so the C++
runtime can be handed the identical stream offline -- no MuJoCo, no threads, no ROS -- and
its outputs diffed against python's. A divergence is then a code difference, because
everything else was held fixed by construction.

WHAT "IDENTICAL" CAN MEAN HERE, which is less than it sounds:

* Our own arithmetic -- the observation assembly, the re-anchoring, the depth row -- can be
  compared EXACTLY, with the network outputs held fixed.
* Anything downstream of a network cannot. Python runs onnxruntime 1.22 and the container's
  C++ runs 1.24.4, and different builds of one graph do not agree bit for bit. So the
  ``obs_checksum`` (blake2b over the float32 bytes) is a fast identical/not-identical
  signal, never the acceptance test.

WHAT IS NOT CAPTURED, deliberately: the generator's own plan. It is the runtime's output,
not its input, and feeding it back would make the comparison vacuous on exactly the
component most likely to differ. It IS recorded as an output, per tick, so a divergence can
be localised to the plan rather than to what was done with it.

THE ONE AUTOREGRESSIVE LEAK, and the reason ``prev_action`` is here. The ``actions``
observation term is the runtime's OWN previous network output
(``policy_runtime.cc``: ``tracker_->last_action()``). A replay fed only the sensor readings
is therefore autoregressive through that one term, however teacher-forced it looks -- and
after the first divergence every later tick is a consequence rather than an independent
sample. Recording the previous action lets the C++ side be driven fully teacher-forced, in
which case each tick IS an independent test and the comparator reports a distribution
rather than a first failure.

Format: the same manifest + raw float64 blobs the parity fixtures use
(:mod:`deploy.fixture_io`), so the C++ reads it with the reader it already has.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from .fixture_io import FixtureWriter

#: One case name; the stream is a single case whose blobs are ``[ticks, ...]``.
CASE = "replay"

#: What the C++ side calls this comparison.
FN = "policy_runtime_step"

#: Tolerances, and they are NOT one number. Split because the mechanisms differ: our own
#: arithmetic should be exact to rounding, while a value that has passed through a network is
#: bounded by the two onnxruntime builds' disagreement and nothing tighter.
#:
#: `network` is deliberately loose. It is not a claim that 1e-4 is fine on an action -- it is
#: the largest disagreement two ORT builds were observed to produce, and a number that must be
#: RE-MEASURED rather than trusted, which is why the comparator prints the distribution it
#: actually saw instead of only a pass/fail.
ATOL_EXACT = 1e-12
ATOL_NETWORK = 1e-4

#: Blobs recorded per tick. Names are the C++ HardwareReadings fields where they correspond,
#: so the replay tool's mapping is one obvious line each rather than a lookup table.
INPUT_FIELDS = (
    "root_pos",          # (3,)
    "base_quat_xyzw",    # (4,)
    "base_ang_vel_b",    # (3,) root-link frame, what the gyro measures
    # (3,) WORLD frame. Not a policy input -- no artifact observes base linear velocity
    # -- but the achieved-velocity comparison against a robot needs it, and the robot
    # side has it from the estimator.
    "base_lin_vel_w",
    "joint_pos",         # (nj,) POLICY order
    "joint_vel",         # (nj,)
    "command",           # (3,)
    "depth_raw",         # (cams * pixels,) METRES, pre-normalisation
    "fm_camera_pos",     # (cams * 3,) in the root's yaw-aligned frame
    "fm_camera_rot",     # (cams * 9,) row-major
    "height_scan",       # (rays,) or empty
    "prev_action",       # (nj,) the previous tick's RAW network output -- see the module docs
)

#: Blobs recorded per tick as the oracle's answer.
OUTPUT_FIELDS = (
    "obs",               # (num_obs,) the assembled observation
    "action",            # (nj,) raw network output, policy order
    "window",            # (offsets * node_dim,) the served reference window
    # blake2b-64, split into two 32-bit halves. NOT one float64: the digest uses all 64 bits
    # and a float64 represents integers exactly only to 2**53, so storing it whole silently
    # rounded it and 299 of 300 ticks then "mismatched" against their own source. Each half is
    # < 2**32 and therefore exact.
    "obs_checksum_hi",   # (1,) digest >> 32
    "obs_checksum_lo",   # (1,) digest & 0xffffffff
    "ctrl_step",         # (1,)
    "replanned",         # (1,) 1.0 on a tick the generator re-inferred
    # The `history_null` the generator was ASKED with, or -1 before any plan exists. Recorded
    # because a mismatch here is invisible in every shape and changes the plan completely: the
    # graph either reads the history tokens or swaps them for the learned null embedding, and
    # the two are different functions. A disagreement confined to the first N ticks, decaying
    # as real history fills in, is exactly what it would look like.
    "fm_history_null",
    # The generator's conditioning, per tick: the flat exteroception row and the held future
    # command. Pure arithmetic on both sides -- the encoder lives inside the graph -- so these
    # should be EXACT, and with history_null on they plus the noise are the whole input.
    "fm_extero",
    "fm_future_cmd",
    # The history the generator was conditioned on: N nodes it previously SERVED, and the
    # commands beside them. These are the runtime's own past output, so they diverge as soon
    # as the two runtimes' plans do -- which is exactly why they are here. With the extero row
    # and history_null agreeing, the first tick where the HISTORY disagrees separates "the
    # conditioning differs" from "the inference differs", and there is nothing else left.
    "fm_history_nodes",
    "fm_history_cmd"
)


class ReplayWriter:
    """Accumulates per-tick rows and writes them as one fixture case.

    Rows are held in memory: a 16 s run at 50 Hz is 800 ticks, and the widest field is the
    depth row (2 cameras x 780 px), so the whole stream is a few tens of MB. Streaming to
    disk would buy nothing and cost the ability to write the shapes in the manifest.

    Args:
        out_dir: Where ``manifest.json`` and the blobs go.
        num_joints: Joint count, for the manifest's structural attrs.
        artifacts: The tracker/generator paths the run used, recorded so a replay cannot be
            compared against a stream from a DIFFERENT pair without it being visible.
    """

    def __init__(self, out_dir: str, num_joints: int, artifacts: dict[str, str]) -> None:
        self.out_dir = out_dir
        self.num_joints = int(num_joints)
        self.artifacts = dict(artifacts)
        self._rows: list[dict[str, np.ndarray]] = []

    def append(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        """Record one tick.

        Args:
            inputs: Every name in :data:`INPUT_FIELDS`; a missing one is recorded empty, so
                a scan-less or camera-less policy needs no special case.
            outputs: Every name in :data:`OUTPUT_FIELDS`.

        Raises:
            KeyError: If an output is missing. Inputs may be absent (a policy that reads no
                height scan), but an absent OUTPUT means the oracle did not answer, and
                recording a zero for it would compare the C++ against nothing.
        """
        row: dict[str, np.ndarray] = {}
        for name in INPUT_FIELDS:
            v = inputs.get(name)
            row[name] = (np.zeros(0) if v is None
                         else np.asarray(v, dtype=np.float64).ravel())
        for name in OUTPUT_FIELDS:
            if name not in outputs:
                raise KeyError(f"replay row is missing the output '{name}'")
            row[name] = np.atleast_1d(np.asarray(outputs[name], dtype=np.float64)).ravel()
        self._rows.append(row)

    def write(self, extra_attrs: dict[str, Any] | None = None) -> str:
        """Stack the rows and write the fixture set.

        Returns:
            Path to the manifest.

        Raises:
            ValueError: If no ticks were recorded, or a field's width changed mid-run. A
                width that changes is a contract change mid-stream and there is no sane way
                to stack it; failing here beats a ragged array quietly becoming dtype=object.
        """
        if not self._rows:
            raise ValueError("no ticks recorded; nothing to compare against")
        stacked: dict[str, np.ndarray] = {}
        for name in INPUT_FIELDS + OUTPUT_FIELDS:
            widths = {r[name].size for r in self._rows}
            if len(widths) != 1:
                raise ValueError(
                    f"field '{name}' changed width mid-run ({sorted(widths)}); the contract "
                    "cannot change inside one stream")
            stacked[name] = np.stack([r[name] for r in self._rows])

        w = FixtureWriter(self.out_dir, "transfer/deploy/replay_stream.py")
        w.add(
            CASE, FN,
            inputs={k: stacked[k] for k in INPUT_FIELDS},
            outputs={k: stacked[k] for k in OUTPUT_FIELDS},
            atol=ATOL_EXACT,
            compare="per_tick",
            attrs={
                "ticks": len(self._rows),
                "num_joints": self.num_joints,
                "atol_exact": ATOL_EXACT,
                "atol_network": ATOL_NETWORK,
                **(extra_attrs or {}),
            },
        )
        return w.write_manifest(seed=0, extra={"artifacts": self.artifacts})


def load(out_dir: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read a stream back, for the comparator and for the self-parity check.

    Args:
        out_dir: Directory holding ``manifest.json``.

    Returns:
        ``(arrays, manifest)`` where arrays maps every field name to ``[ticks, width]``.

    Raises:
        ValueError: If a blob's size disagrees with its manifest shape -- a truncated or
            half-regenerated stream, which must not be silently compared.
    """
    with open(os.path.join(out_dir, "manifest.json")) as fh:
        manifest = json.load(fh)
    case = next(c for c in manifest["cases"] if c["case"] == CASE)
    arrays: dict[str, np.ndarray] = {}
    for entry in case["inputs"] + case["outputs"]:
        shape = tuple(entry["shape"])
        raw = np.fromfile(os.path.join(out_dir, entry["file"]), dtype=np.float64)
        # prod(()) is 1, which is right for a scalar blob; an empty FIELD is shape [ticks, 0]
        # and prods to 0, which is also right. Written out rather than as a conditional
        # expression, whose precedence made the scalar case read backwards.
        expected = int(np.prod(shape)) if shape else 1
        if raw.size != expected:
            raise ValueError(
                f"{entry['file']} holds {raw.size} values but the manifest says {shape} "
                f"({expected}); the stream is truncated or half-regenerated")
        arrays[entry["name"]] = raw.reshape(shape)
    return arrays, manifest


def split_checksum(value: int) -> tuple[float, float]:
    """A 64-bit checksum as two float64-exact halves.

    Args:
        value: The blake2b-64 digest.

    Returns:
        ``(hi, lo)``, each an integer below 2**32 held exactly by a float64.
    """
    v = int(value) & 0xFFFFFFFFFFFFFFFF
    return float(v >> 32), float(v & 0xFFFFFFFF)


def join_checksum(hi: float, lo: float) -> int:
    """Reassemble what :func:`split_checksum` wrote.

    Args:
        hi: High 32 bits, as recorded.
        lo: Low 32 bits.

    Returns:
        The 64-bit checksum.
    """
    return (int(round(hi)) << 32) | int(round(lo))
