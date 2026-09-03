#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""How far a MEASURED history row sits from the reference row the generator eats.

The generator's ``N`` history tokens are reference-window nodes.  Today two producers fill
them -- the node this runtime SERVED (closed loop) or the CLIP's node (teacher forcing) --
and both are "the reference applied at that step".  A third arm feeds the robot's OWN
measured state instead, which changes the KIND of quantity in most of the row.  This script
measures by how much, before anything is run in a simulator, in the units the model actually
sees: the state normalizer's per-dimension standard deviations.

**Read the headline as a fraction of a training standard deviation.**  1.0 means a measured
row differs from a reference row by a full sigma of the block's own training spread -- fully
off-distribution.  0.1 means a 10% perturbation of a signal the model was fitted on.

Two sources, one output schema, because the two answer different questions and the numbers
have to be comparable:

``--rollout`` (``record_rollouts.py`` output)
    SERVED node vs measured, under the CLOSED LOOP -- the distribution the eval arm actually
    runs in.  Only ``joint_pos`` is recoverable: the file carries the served ``history`` ring
    and the measured ``ProprioRing``, but no measured root pose per history row.

``--recording`` (``record_data.py`` windowed ``.h5``, the FM's own training set)
    APPLIED REFERENCE vs measured, at the SNAP ROW -- how far the measured row is from what
    the model was fitted on.  Covers every block, because ``record_data`` substitutes the plan
    anchor into history rows ``0..H-2`` but leaves row ``H-1`` fully measured.

.. admonition:: Why the snap row answers the question, for every block

    A history row is observed in the PLAN's yaw frame, not the measured one, so the snap row
    is not literally a history row.  It does not need to be, because the reported statistic
    is a DIFFERENCE between two nodes built against the SAME anchor:

    * the anchor's POSITION appears in both and cancels;
    * the anchor's YAW is a rotation, applied identically to both, so it preserves the
      difference's norm -- and every block's normalizer entries are equal within the block
      (``0.4, 0.4, 0.4`` for ``base_pos``, ``1`` throughout ``base_rot6d``, and so on), so
      normalizing does not break that.

    So the per-block gap NORM -- which is what this script reports -- is the same under any
    anchor, and the snap row's measured pose serves as well as the plan's would.

    What is NOT anchor-free is the per-COMPONENT split of the base blocks: a yaw rotation
    mixes ``x`` into ``y``.  Only ``joint_pos``, which never had a frame, has a meaningful
    per-dimension breakdown here, and it is the only block for which one is reported.

    ``--self_check`` asserts both halves rather than leaving them as an argument: the gaps
    are recomputed under a second, deliberately wrong anchor; the norms must agree to
    floating-point, and the base blocks' component split must MOVE (or the "wrong" anchor is
    not being applied and the invariance the norms just passed is vacuous).

``base_pos`` and ``base_lin_vel`` are the two blocks a robot cannot measure (no odometry, no
base linear-velocity estimate), so the arm takes them from the last prediction.  They are
reported anyway: ``base_pos``'s gap is exactly the base position tracking error, which is the
number that says what measuring them would be worth if odometry ever arrives.

No IsaacLab and no torch beyond reading the checkpoint's normalizer; ``deploy.window_np`` is
the pure-numpy port of the same representation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
#: ``<repo>/scripts/flow_matching`` -> ``<repo>``.  Derived and marker-checked rather than
#: counted in ``..`` hops, which has silently broken when a module moved.
REPO_ROOT: str = os.path.dirname(os.path.dirname(_HERE))
if not os.path.isdir(os.path.join(REPO_ROOT, "transfer", "deploy")):
    raise SystemExit(f"[ERROR] {REPO_ROOT} is not the repo root (no transfer/deploy)")
sys.path.insert(0, os.path.join(REPO_ROOT, "transfer"))

from deploy.quat_math_np import quat_apply_inverse, quat_conjugate, quat_mul, yaw_quat  # noqa: E402
from deploy.window_np import quat_to_rot6d  # noqa: E402

#: Block name -> ``(start, stop)`` in a ``base_pose + base_vel + joint_pos`` node, with
#: ``joint_pos`` left open-ended (the joint count comes from the artifact).
#:
#: Split FINER than the node's own blocks on purpose: ``base_pose`` is one block in the
#: layout but its two halves have opposite answers to "can a robot measure this", and
#: reporting them together would average a measurable rotation into an unmeasurable
#: position.  Same for ``base_vel``.
BLOCK_SPANS: dict[str, tuple[int, int]] = {
    "base_pos": (0, 3),
    "base_rot6d": (3, 9),
    "base_lin_vel": (9, 12),
    "base_ang_vel": (12, 15),
    "joint_pos": (15, -1),
}

#: Which blocks a measured history row actually fills on a ROBOT.  The other two are taken
#: from the node the runtime last SERVED, because the G1 has no odometry and no base
#: linear-velocity estimate.  Mirrors ``MEASURED_HISTORY_FALLBACK`` on the runtime side.
MEASURED_BLOCKS: tuple[str, ...] = ("base_rot6d", "base_ang_vel", "joint_pos")

#: Blocks expressed in the ANCHOR's yaw frame, whose per-dimension split therefore depends
#: on which anchor was used.  Their norms do not (see the module docstring), so they are
#: reported and their component breakdown is not.
YAW_FRAME_BLOCKS: frozenset[str] = frozenset(
    {"base_pos", "base_rot6d", "base_lin_vel", "base_ang_vel"})

#: Physical unit each block's gap is additionally reported in, for a reader who does not
#: think in sigmas.
BLOCK_UNITS: dict[str, str] = {
    "base_pos": "m",
    "base_rot6d": "deg",
    "base_lin_vel": "m/s",
    "base_ang_vel": "rad/s",
    "joint_pos": "rad",
}


def dim_names_for(block: str, joint_names: list[str]) -> list[str]:
    """Names for one block's dimensions, where a per-dimension split means anything.

    ``[]`` for every base block, which suppresses their per-dimension breakdown: those are
    expressed in the anchor's yaw frame, and this script's anchor is not the one a history
    row actually uses, so a yaw rotation is free to move error between ``x`` and ``y``.  The
    block's NORM survives that (see the module docstring) and is what gets reported.
    ``joint_pos`` has no frame at all, so its per-joint split is a fact about the robot.

    Args:
        block: Block name from :data:`BLOCK_SPANS`.
        joint_names: The artifact's joint order, used for ``joint_pos``.

    Returns:
        One name per dimension, or ``[]`` where the split is not meaningful.
    """
    return [] if block in YAW_FRAME_BLOCKS else list(joint_names)


def sha256_prefix(path: str, n: int = 12) -> str:
    """First ``n`` hex chars of a file's SHA-256, read in chunks.

    Args:
        path: File to hash.
        n: Characters to keep.

    Returns:
        The truncated digest, or ``""`` if the file cannot be read.
    """
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        return h.hexdigest()[:n]
    except OSError:
        return ""


def load_state_normalizer(
    ckpt_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, dict]:
    """Read the FM checkpoint's state normalizer and the config fields this script needs.

    The normalizer is what makes the gap a MODEL statement rather than a physical one: the
    graph folds ``(x - mean) / std`` in, so a difference of one ``std`` is a difference of
    one unit at the model's input whatever the block's physical scale.

    Note several dimensions carry an identity mask (``std = 1``) -- the 6D rotation block is
    left raw on purpose, since standardizing it would break its near-unit-scale structure --
    so for those "std units" and raw units coincide.  That is correct, not a gap in the
    measurement: the model really does see them raw.

    Args:
        ckpt_path: Path to an FM ``.pt``.

    Returns:
        ``(mean, std, standardize_mask, config, meta)``, the first three as ``(D,)`` float64.
        ``meta`` is the recording provenance the checkpoint carries, which is where the
        node's own joint order lives.

    Raises:
        SystemExit: If the checkpoint has no ``state_normalizer``.
    """
    import torch  # noqa: PLC0415  -- the only torch use, and only to unpickle

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sn = ckpt.get("state_normalizer")
    if sn is None:
        raise SystemExit(f"[ERROR] {ckpt_path} has no 'state_normalizer'")
    cfg = ckpt.get("config") or ckpt.get("cfg") or {}
    cfg = cfg if isinstance(cfg, dict) else vars(cfg)
    return (sn["mean"].numpy().astype(np.float64),
            sn["std"].numpy().astype(np.float64),
            sn["standardize_mask"].numpy().astype(bool),
            cfg, dict(ckpt.get("meta") or {}))


def resolve_spans(node_dim: int, n_joints: int) -> dict[str, tuple[int, int]]:
    """Close :data:`BLOCK_SPANS`' open-ended ``joint_pos`` against the real widths.

    Refuses rather than truncating a node that is not ``base_pose + base_vel + joint_pos``:
    a wider node means another block is present and every span after ``base_pos`` would be
    describing the wrong dimensions -- silently, since the slices would still be in range.

    Args:
        node_dim: Width of one window node.
        n_joints: Joint count from the recording's own term table.

    Returns:
        Block name -> ``(start, stop)``.

    Raises:
        SystemExit: If ``node_dim != 15 + n_joints``.
    """
    if node_dim != 15 + n_joints:
        raise SystemExit(
            f"[ERROR] node width {node_dim} != 15 + {n_joints} joints. This probe assumes "
            "reference_blocks == ['base_pose', 'base_vel', 'joint_pos']; a node carrying "
            "joint_vel or ee_pos needs its spans extended before the numbers mean anything."
        )
    return {k: (a, node_dim if b < 0 else b) for k, (a, b) in BLOCK_SPANS.items()}


def _json_attr(attrs: dict, key: str, default=None):
    """Read an HDF5 attribute that may be a JSON string or an already-decoded sequence.

    Args:
        attrs: The attribute mapping.
        key: Attribute name.
        default: Returned when the key is absent.

    Returns:
        The decoded value.
    """
    if key not in attrs:
        return default
    v = attrs[key]
    if isinstance(v, bytes):
        v = v.decode()
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v          # a plain string attribute, written before the JSON convention
    return list(v) if hasattr(v, "__len__") else v


def family_table(path: str, attrs: dict) -> list[str]:
    """The family-code -> name table belonging to THIS file's clip library.

    Taken from the file, or from its ``.episodes.h5`` sibling for a rollout (whose window
    table does not carry one), and from nowhere else.  In particular NOT from the FM
    checkpoint: a rollout is generally run against a different library than the generator
    trained on -- here ``eval_clips.h5`` against ``g1_mb_v5_body_mode11_decor.h5`` -- and the
    two tables assign different names to the same integer.  A mislabelled family is a wrong
    conclusion that looks entirely reasonable.

    Args:
        path: The source file, used to find the sibling.
        attrs: Its attributes.

    Returns:
        The name table, or ``[]`` when none can be found (callers then print bare codes).
    """
    names = _json_attr(attrs, "family_names")
    if names:
        return [str(n) for n in names]
    sib = os.path.splitext(path)[0] + ".episodes.h5"
    if os.path.exists(sib):
        try:
            with h5py.File(sib, "r") as g:
                return [str(n) for n in (_json_attr(dict(g.attrs), "family_names") or [])]
        except OSError:
            return []
    return []


def block_stats(gap_std: np.ndarray, gap_phys: np.ndarray, unit: str,
                dim_names: list[str] | None = None) -> dict:
    """Summarize one block's per-window gap.

    The headline is the per-DIMENSION RMS in std units, so blocks of different widths are
    directly comparable: a 29-wide block whose every dimension is off by 0.1 sigma scores
    0.1, not ``0.1 * sqrt(29)``.

    Args:
        gap_std: ``(W, d)`` per-dimension difference in normalizer-std units.
        gap_phys: ``(W,)`` the same gap in a physical unit (already reduced).
        unit: Name of that physical unit, for the report.
        dim_names: Optional per-dimension names, so the worst offenders are reported by
            name rather than as an index into a block whose offset the reader has to
            work out.

    Returns:
        A JSON-ready dict of percentiles and the per-dimension worst case.
    """
    per_win = np.sqrt(np.mean(gap_std ** 2, axis=1))          # (W,) RMS over dims
    per_dim = np.sqrt(np.mean(gap_std ** 2, axis=0))          # (d,) RMS over windows
    pct = lambda a, q: float(np.percentile(a, q))             # noqa: E731
    order = np.argsort(per_dim)[::-1][:5]
    # Omitted rather than caveated where a yaw rotation mixes the components: a "worst
    # dimension" that depends on which anchor happened to be used is a fact about the anchor.
    worst = ([] if not dim_names else
             [{"index": int(i), "name": dim_names[i], "rms": float(per_dim[i])} for i in order])
    return {
        "n_dims": int(gap_std.shape[1]),
        "std_units": {"p50": pct(per_win, 50), "p90": pct(per_win, 90),
                      "p99": pct(per_win, 99), "rms": float(np.sqrt(np.mean(per_win ** 2))),
                      "worst_dims": worst},
        "physical": {"unit": unit, "p50": pct(gap_phys, 50), "p90": pct(gap_phys, 90),
                     "p99": pct(gap_phys, 99)},
    }


def geodesic_deg(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    """Angle between two orientations, in degrees.

    The interpretable form of a ``base_rot6d`` gap: a 6-vector difference has no physical
    reading, while "the measured base is N degrees from the reference base" does.

    Args:
        q_a: ``(W, 4)`` xyzw.
        q_b: ``(W, 4)`` xyzw.

    Returns:
        ``(W,)`` degrees in ``[0, 180]``.
    """
    rel = quat_mul(quat_conjugate(q_a), q_b)
    # |w| -> the angle of the SHORTER rotation, so a double-cover sign flip cannot read as
    # a ~360 degree error on an otherwise identical orientation.
    return np.degrees(2.0 * np.arccos(np.clip(np.abs(rel[..., 3]), 0.0, 1.0)))


def nodes_from_rows(rows: np.ndarray, anchor_p: np.ndarray, anchor_q: np.ndarray,
                    slc: dict[str, tuple[int, int]]) -> np.ndarray:
    """Build window nodes from raw recorded state rows against a given anchor.

    Args:
        rows: ``(W, state_dim)`` rows in ``record_data``'s ``state_term_names`` layout.
        anchor_p: ``(W, 3)`` anchor position, world frame.
        anchor_q: ``(W, 4)`` anchor orientation, xyzw world frame.
        slc: Term name -> ``(start, stop)`` from the recording's own term table.

    Returns:
        ``(W, D)`` window nodes, ``base_pose + base_vel + joint_pos``.
    """
    g = lambda name: rows[:, slc[name][0]:slc[name][1]]        # noqa: E731
    yaw = yaw_quat(anchor_q)
    return np.concatenate([
        quat_apply_inverse(yaw, g("root_pos_w") - anchor_p),
        quat_to_rot6d(quat_mul(quat_conjugate(yaw), g("root_quat_w"))),
        quat_apply_inverse(yaw, g("root_link_lin_vel_w")),
        quat_apply_inverse(yaw, g("root_link_ang_vel_w")),
        g("joint_pos"),
    ], axis=-1)


def term_slices(names: list[str], dims: list[int]) -> dict[str, tuple[int, int]]:
    """Name -> ``(start, stop)`` from a recording's parallel term-name/width attributes.

    Args:
        names: ``state_term_names``.
        dims: ``state_term_dims``.

    Returns:
        The slice table.
    """
    out, off = {}, 0
    for n, d in zip(names, dims):
        out[n] = (off, off + int(d))
        off += int(d)
    return out


# ----------------------------------------------------------------------------------------
# Source: the training recording (record_data.py windowed .h5)
# ----------------------------------------------------------------------------------------
def gap_from_recording(path: str, mean: np.ndarray, std: np.ndarray, max_windows: int,
                       self_check: bool, rng: np.random.Generator) -> dict:
    """Measured vs APPLIED REFERENCE at the snap row of an FM training recording.

    Args:
        path: The windowed ``.h5``.
        mean: ``(D,)`` normalizer mean (unused for a difference, kept for symmetry).
        std: ``(D,)`` normalizer std -- what turns a physical gap into a model one.
        max_windows: Cap on windows read.
        self_check: Re-measure the anchor-invariant blocks under a wrong anchor and assert
            they agree.
        rng: Source of the self-check's deliberately wrong anchor.

    Returns:
        The report dict.
    """
    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
        # ``record_data`` writes its scalar attributes as JSON, so this one arrives as the
        # five characters `"plan"` rather than the four of plan. A bare string compare reads
        # every current recording as the legacy per-step format and refuses it.
        anchor = _json_attr(attrs, "history_anchor", "")
        if str(anchor) != "plan":
            raise SystemExit(
                f"[ERROR] history_anchor={anchor!r}, not 'plan'. The snap-row argument "
                "assumes the plan-anchored recorder; a per-step recording has a different "
                "row layout entirely."
            )
        names = _json_attr(attrs, "state_term_names")
        dims = _json_attr(attrs, "state_term_dims")
        slc = term_slices([str(n) for n in names], list(dims))
        n_joints = slc["joint_pos"][1] - slc["joint_pos"][0]
        written = int(attrs.get("num_windows_written", f["history_state"].shape[0]))
        take = min(written, int(max_windows))
        h = int(f["history_state"].shape[1])
        # The SNAP row.  Rows 0..H-2 have their measured root pose overwritten with the plan
        # anchor by the recorder; this one is left fully measured, which is the whole reason
        # the comparison is possible without a re-record.
        meas = np.asarray(f["history_state"][:take, h - 1, :], dtype=np.float64)
        ref = np.asarray(f["history_ref"][:take, h - 1, :], dtype=np.float64)
        fam = (np.asarray(f["clip_family"][:take]) if "clip_family" in f
               else np.zeros(take, dtype=np.uint8))
        fam_names = family_table(path, attrs)
        joint_names = [str(n) for n in (_json_attr(attrs, "state_joint_names") or [])]

    anchor_p, anchor_q = meas[:, slc["root_pos_w"][0]:slc["root_pos_w"][1]], \
        meas[:, slc["root_quat_w"][0]:slc["root_quat_w"][1]]
    node_m = nodes_from_rows(meas, anchor_p, anchor_q, slc)
    node_r = nodes_from_rows(ref, anchor_p, anchor_q, slc)
    spans = resolve_spans(node_m.shape[1], n_joints)

    phys = {
        "base_pos": np.linalg.norm(node_r[:, 0:3] - node_m[:, 0:3], axis=1),
        "base_rot6d": geodesic_deg(anchor_q, ref[:, slc["root_quat_w"][0]:slc["root_quat_w"][1]]),
        "base_lin_vel": np.linalg.norm(node_r[:, 9:12] - node_m[:, 9:12], axis=1),
        "base_ang_vel": np.linalg.norm(node_r[:, 12:15] - node_m[:, 12:15], axis=1),
        "joint_pos": np.sqrt(np.mean((node_r[:, 15:] - node_m[:, 15:]) ** 2, axis=1)),
    }

    checks = {}
    if self_check:
        # A deliberately WRONG anchor: a different yaw and a metre of translation.  The
        # anchor-invariant blocks must be unmoved by it; the other two must move, or the
        # check is passing because nothing depends on the anchor at all.
        ang = rng.uniform(-np.pi, np.pi, size=(take, 1))
        bad_q = np.concatenate([np.zeros((take, 2)), np.sin(ang / 2), np.cos(ang / 2)], axis=1)
        bad_p = anchor_p + rng.normal(scale=1.0, size=anchor_p.shape)
        bm = nodes_from_rows(meas, bad_p, bad_q, slc)
        br = nodes_from_rows(ref, bad_p, bad_q, slc)
        for name, (a, b) in spans.items():
            # The reported statistic: per-window RMS over the block's dims, in std units.
            n0 = np.sqrt(np.mean(((node_r[:, a:b] - node_m[:, a:b]) / std[a:b]) ** 2, axis=1))
            n1 = np.sqrt(np.mean(((br[:, a:b] - bm[:, a:b]) / std[a:b]) ** 2, axis=1))
            # The per-COMPONENT split, which a yaw rotation is free to mix.
            c0 = np.sqrt(np.mean(((node_r[:, a:b] - node_m[:, a:b]) / std[a:b]) ** 2, axis=0))
            c1 = np.sqrt(np.mean(((br[:, a:b] - bm[:, a:b]) / std[a:b]) ** 2, axis=0))
            # Two tolerances, two orders of magnitude apart, and the SEPARATION is what makes
            # the pair of assertions mean anything.  The recorded rows are float32, so a
            # quaternion round trip moves an invariant by ~1e-7 -- tightening past that fails
            # on correct data.  A random yaw, by contrast, mixes x into y wholesale, so a
            # genuine component move is O(0.1); anything between the two bands would be
            # neither.
            checks[name] = {
                "norm_max_abs_diff": float(np.abs(n0 - n1).max()),
                "norm_invariant": bool(np.allclose(n0, n1, rtol=1e-5, atol=1e-5)),
                "components_moved": bool(not np.allclose(c0, c1, rtol=1e-3, atol=1e-3)),
            }
            if not checks[name]["norm_invariant"]:
                raise SystemExit(
                    f"[ERROR] self_check: {name}'s gap norm is not anchor-invariant (max "
                    f"|delta| {checks[name]['norm_max_abs_diff']:.3g}). The snap-row argument "
                    "does not hold and these numbers do not describe a history row."
                )
        # The wrong anchor must actually bite somewhere, or every norm above passed for the
        # trivial reason that nothing was applied.  The base blocks are where it shows: a
        # yaw rotation mixes their components while leaving the norm alone.
        if not any(checks[n]["components_moved"] for n in ("base_pos", "base_ang_vel")):
            raise SystemExit(
                "[ERROR] self_check: the deliberately wrong anchor moved no component of any "
                "base block, so it is not being applied and the norm invariance it just "
                "'confirmed' is vacuous."
            )
        if checks["joint_pos"]["components_moved"]:
            raise SystemExit(
                "[ERROR] self_check: joint_pos moved under a different anchor. It is a "
                "frame-free block and cannot; the node builder is mixing blocks."
            )

    return _assemble(node_m, node_r, spans, phys, std, fam, fam_names, joint_names,
                     comparison="measured_vs_applied_reference", extra={"self_check": checks})


# ----------------------------------------------------------------------------------------
# Source: a closed-loop rollout (record_rollouts.py .h5)
# ----------------------------------------------------------------------------------------
def gap_from_rollout(path: str, std: np.ndarray, max_windows: int, sweep: bool,
                     joint_names: list[str], ckpt_path: str) -> dict:
    """Measured vs SERVED ``joint_pos``, under the closed loop, with a row-offset sweep.

    Only ``joint_pos`` is recoverable here -- the file has the served ``history`` ring and
    the measured ``ProprioRing``, but no measured root pose per history row.

    The sweep is the point as much as the number is.  ``history`` is pushed inside
    ``env.step`` and holds the node served at ``t-1``; ``proprio`` is pushed after the step
    and holds the state at ``t``.  Reading that off the code gives an expected lag, but a
    silent off-by-one here would bias every gap upward by one control step of joint motion,
    so the offset is MEASURED: the true alignment is the argmin, and a flat sweep means the
    two rings are not related the way this assumes.

    Args:
        path: The rollout ``.h5``.
        std: ``(D,)`` normalizer std.
        max_windows: Cap on windows read.
        sweep: Run the offset sweep (else use the argmin-free nominal lag).
        joint_names: The node's joint order, for naming the worst dimensions.
        ckpt_path: The checkpoint the normalizer came from, checked against the one the
            rollout actually ran.

    Returns:
        The report dict.
    """
    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
        # The served nodes in this file were produced by ONE generator, and the gap is
        # divided by ONE generator's standard deviations. If they are not the same run, the
        # number is silently rescaled -- every std in the joint block differs run to run,
        # since that block is fitted rather than declared.
        ran = str(attrs.get("fm_checkpoint", "")).strip('"')
        if ran and os.path.basename(os.path.dirname(ran)) != \
                os.path.basename(os.path.dirname(os.path.abspath(ckpt_path))):
            raise SystemExit(
                f"[ERROR] this rollout was driven by {ran}, but the normalizer comes from "
                f"{ckpt_path}. The served history rows are that generator's output and the "
                "gap has to be measured in that generator's units; pass its checkpoint."
            )
        written = int(attrs.get("num_windows_written", f["history"].shape[0]))
        take = min(written, int(max_windows))
        hist = np.asarray(f["history"][:take], dtype=np.float64)        # (W, N, D)
        prop = np.asarray(f["proprio"][:take], dtype=np.float64)        # (W, H, P)
        null = np.asarray(f["history_null"][:take]).astype(bool)
        fam = (np.asarray(f["family_code"][:take]) if "family_code" in f
               else np.zeros(take, dtype=np.uint8))
        layout = _json_attr(attrs, "proprio_layout", [])
        fam_names = family_table(path, attrs)

    if "joint_pos(29)" not in [str(x) for x in layout]:
        raise SystemExit(f"[ERROR] proprio_layout {layout} does not start with joint_pos(29)")
    n_joints = 29
    _, n_hist, node_d = hist.shape
    spans = resolve_spans(node_d, n_joints)
    a, b = spans["joint_pos"]

    # Drop null-masked windows: those history rows are ZERO placeholders the model is told to
    # ignore, and scoring them would report the size of a zero row rather than a gap.
    keep = ~null
    if not keep.any():
        raise SystemExit("[ERROR] every window is history_null; nothing to compare")
    hist, prop, fam = hist[keep], prop[keep], fam[keep]
    served = hist[:, :, a:b]                                            # (W, N, nj)
    meas_all = prop[:, :, 0:n_joints]                                   # (W, H, nj)
    n_prop = meas_all.shape[1]

    # Offset sweep: history row `i` (oldest..newest) against proprio row `i + off`.
    sweep_rows = []
    lo = 0
    hi = n_prop - n_hist
    for off in (range(lo, hi + 1) if sweep else [hi - 1]):
        m = meas_all[:, off:off + n_hist, :]
        g = np.sqrt(np.mean(((served - m) / std[a:b]) ** 2, axis=(1, 2)))
        sweep_rows.append({"proprio_offset": int(off), "std_units_p50": float(np.median(g)),
                           "std_units_rms": float(np.sqrt(np.mean(g ** 2)))})
    curve = [r["std_units_p50"] for r in sweep_rows]
    k = int(np.argmin(curve))
    off = sweep_rows[k]["proprio_offset"]
    # The test is that the minimum is INTERIOR, not that it is deep.  A one-control-step
    # (20 ms) shift moves the joints very little next to the tracking error itself, so the
    # curve is legitimately shallow -- demanding a margin would fail on correct data.  What
    # a real misalignment looks like is a minimum pinned to an endpoint: the sweep never
    # bracketed an optimum and the true lag is outside the range the two ring lengths allow.
    if sweep and len(curve) > 2 and k in (0, len(curve) - 1):
        raise SystemExit(
            f"[ERROR] the row-offset sweep minimises at its {'first' if k == 0 else 'last'} "
            f"offset ({[round(c, 4) for c in curve]}), so it never bracketed a minimum. The "
            "history and proprio rings are not aligned the way this script assumes, and the "
            "gap it would report is an alignment artifact."
        )

    m = meas_all[:, off:off + n_hist, :]
    # Flatten the N history rows into the window axis: each row is its own sample of "a
    # measured row against the served row for the same step".
    gap_std = ((served - m) / std[a:b]).reshape(-1, n_joints)
    gap_phys = np.sqrt(np.mean((served - m) ** 2, axis=2)).reshape(-1)
    fam_rep = np.repeat(fam, n_hist)

    out = {
        "comparison": "measured_vs_served_generator_node",
        "n_samples": int(gap_std.shape[0]),
        "n_windows": int(served.shape[0]),
        "dropped_history_null": int((~keep).sum()),
        "row_alignment": {"proprio_offset": off, "sweep": sweep_rows},
        "blocks": {"joint_pos": block_stats(gap_std, gap_phys, BLOCK_UNITS["joint_pos"],
                                            joint_names)},
        "blocks_unavailable": ["base_pos", "base_rot6d", "base_lin_vel", "base_ang_vel"],
    }
    out["by_family"] = {}
    for code in sorted(set(int(c) for c in fam_rep.tolist())):
        sel = fam_rep == code
        if sel.sum() < 100:
            continue
        nm = str(fam_names[code]) if code < len(fam_names) else f"code_{code}"
        out["by_family"][nm] = {
            "n": int(sel.sum()),
            "joint_pos": block_stats(gap_std[sel], gap_phys[sel], BLOCK_UNITS["joint_pos"],
                                     joint_names),
        }
    return out


def _assemble(node_m: np.ndarray, node_r: np.ndarray, spans: dict, phys: dict,
              std: np.ndarray, fam: np.ndarray, fam_names: list, joint_names: list,
              comparison: str, extra: dict) -> dict:
    """Reduce a measured/reference node pair into the report schema.

    Args:
        node_m: ``(W, D)`` measured nodes.
        node_r: ``(W, D)`` reference nodes.
        spans: Block name -> ``(start, stop)``.
        phys: Block name -> ``(W,)`` gap in physical units.
        std: ``(D,)`` normalizer std.
        fam: ``(W,)`` family codes.
        fam_names: Family-code -> name table.
        joint_names: The node's joint order, for naming the worst dimensions.
        comparison: What the two sides are, for the report.
        extra: Merged into the top level.

    Returns:
        The report dict.
    """
    out = {"comparison": comparison, "n_windows": int(node_m.shape[0]), "blocks": {}, **extra}
    for name, (a, b) in spans.items():
        out["blocks"][name] = block_stats(
            (node_r[:, a:b] - node_m[:, a:b]) / std[a:b], phys[name], BLOCK_UNITS[name],
            dim_names_for(name, joint_names))
        out["blocks"][name]["role"] = ("measured" if name in MEASURED_BLOCKS
                                       else "fallback_to_served_node")
    out["by_family"] = {}
    for code in sorted(set(int(c) for c in fam.tolist())):
        sel = fam == code
        if sel.sum() < 100:
            continue
        nm = (str(fam_names[code]) if code < len(fam_names) else f"code_{code}")
        out["by_family"][nm] = {"n": int(sel.sum())} | {
            name: block_stats((node_r[sel, a:b] - node_m[sel, a:b]) / std[a:b],
                              phys[name][sel], BLOCK_UNITS[name],
                              dim_names_for(name, joint_names))
            for name, (a, b) in spans.items()
        }
    return out


def print_report(rep: dict) -> None:
    """Print the human-readable summary.

    Args:
        rep: The report dict.
    """
    print(f"\n  comparison: {rep['comparison']}   windows: {rep['n_windows']}")
    if "row_alignment" in rep:
        sw = rep["row_alignment"]["sweep"]
        print(f"  row alignment: proprio_offset={rep['row_alignment']['proprio_offset']} "
              f"(sweep p50 " + ", ".join(f"{r['proprio_offset']}:{r['std_units_p50']:.3f}"
                                         for r in sw) + ")")
    print(f"\n  {'block':<14} {'role':<24} {'d':>3}  "
          f"{'p50':>7} {'p90':>7} {'p99':>7}   physical p50/p90")
    for name, s in rep["blocks"].items():
        u = s["physical"]
        print(f"  {name:<14} {s.get('role', '-'):<24} {s['n_dims']:>3}  "
              f"{s['std_units']['p50']:7.3f} {s['std_units']['p90']:7.3f} "
              f"{s['std_units']['p99']:7.3f}   "
              f"{u['p50']:.4g} / {u['p90']:.4g} {u['unit']}")
    if rep.get("blocks_unavailable"):
        print(f"  (not recoverable from this source: {', '.join(rep['blocks_unavailable'])})")

    for name, s in rep["blocks"].items():
        worst = s["std_units"].get("worst_dims") or []
        if worst:
            print(f"\n  {name} worst dimensions (RMS sigma): "
                  + ", ".join(f"{w['name']} {w['rms']:.2f}" for w in worst))

    if rep.get("by_family"):
        print(f"\n  {'family':<24} {'n':>8}  " +
              "  ".join(f"{n:>12}" for n in rep["blocks"]))
        for fam, v in rep["by_family"].items():
            cells = "  ".join(f"{v[n]['std_units']['p50']:12.3f}" if n in v else f"{'-':>12}"
                              for n in rep["blocks"])
            print(f"  {fam:<24} {v['n']:>8}  {cells}")

    meas = [s["std_units"]["p50"] for n, s in rep["blocks"].items() if n in MEASURED_BLOCKS]
    if meas:
        print(f"\n  headline: the blocks a robot CAN measure move "
              f"{min(meas):.3f}-{max(meas):.3f} sigma at the median.")


def main() -> None:
    """Parse arguments, run the selected source, print and write the report."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--rollout", help="record_rollouts.py .h5 (closed loop, joint_pos only)")
    src.add_argument("--recording", help="record_data.py windowed .h5 (training set, all blocks)")
    p.add_argument("--fm_checkpoint", required=True,
                   help="FM .pt whose state_normalizer defines the units")
    p.add_argument("--out", default=None, help="write the report JSON here")
    p.add_argument("--max_windows", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_self_check", action="store_true",
                   help="skip the anchor-invariance assertion (recording mode)")
    p.add_argument("--no_sweep", action="store_true",
                   help="skip the row-offset sweep (rollout mode)")
    args = p.parse_args()

    mean, std, mask, cfg, meta = load_state_normalizer(args.fm_checkpoint)
    ckpt_joints = [str(n) for n in _json_attr(meta, "state_joint_names", []) or []]
    path = args.rollout or args.recording
    print(f"[gap] source     {path}")
    print(f"[gap] normalizer {args.fm_checkpoint}")
    print(f"[gap]            blocks={cfg.get('reference_blocks')} N={cfg.get('num_history_states')} "
          f"D={len(std)}  identity dims={int((~mask).sum())}")

    if args.rollout:
        # The rollout's window table carries no joint order, so it comes from the checkpoint
        # -- which is sound here and only here: the served rows in that file ARE this
        # generator's output, and the guard inside refuses a checkpoint that did not drive it.
        rep = gap_from_rollout(args.rollout, std, args.max_windows, not args.no_sweep,
                               ckpt_joints, args.fm_checkpoint)
    else:
        rep = gap_from_recording(args.recording, mean, std, args.max_windows,
                                 not args.no_self_check, np.random.default_rng(args.seed))
    rep["source_path"] = path
    rep["source_sha256"] = sha256_prefix(path)
    rep["fm_checkpoint"] = args.fm_checkpoint
    rep["fm_checkpoint_sha256"] = sha256_prefix(args.fm_checkpoint)
    rep["reference_blocks"] = list(cfg.get("reference_blocks") or [])
    print_report(rep)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(rep, fh, indent=2)
        print(f"\n[gap] -> {args.out}")


if __name__ == "__main__":
    main()
