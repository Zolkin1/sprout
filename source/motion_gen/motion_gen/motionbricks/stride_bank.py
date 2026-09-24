# Copyright 2026 Zachary Olkin. All rights reserved.

"""The optimized-stride bank: one definition of the format, plus the per-period geometry.

The bank (written by ``scripts/rsl_rl/export_stride_bank.py``) is the shared artefact between the two
halves of Motion Bricks clip generation, and BOTH halves must agree on which stride index means what:

* the Motion Bricks side (``mb_env.StrideTargeter``, in the ``motionbricks`` conda env) pins a stride's
  pose as the model's target keyframe;
* the ``motion_gen`` side replays the SAME stride exactly, looped, as the clip's steady-state motion.

If those two read different sources they can drift silently, which is why the loop is built from the
bank rather than re-ingested from the trajectory library.

Numpy only and free of intra-package imports, so the Motion Bricks env can load it by path via
``mb_env.load_schema_module`` -- the same contract as ``raw_schema`` / ``terrain_plan`` /
``distractor_plan`` / ``command_schedule``.

**Periodicity.** Each stride is one CLOSED cycle whose last frame duplicates the first IN POSE (the
traj-opt closure), so the distinct period is ``P = F - 1`` frames and the wrap point is frame ``P``.
Measured over all 251 strides of the 50 Hz bank, that closure is bit-exact: joint
``max|frame[F-1] - frame[0]| = 0`` and the net per-period yaw matches the commanded ``wz * P * dt`` to
1.2e-7 rad.  So a looped replay needs no blending at the period seam -- see
:func:`motion_gen.terrain.box_clip.loop_frames`.

**Per-period displacement is DERIVED, never stored.**  It is a pure function of the stored ``qpos``, so
recomputing it on load cannot drift from the motion, whereas a cached copy could.  It also means banks
written before this module still load.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

# qpos layout the bank stores, matching what Motion Bricks' converter expects.
QPOS_ROOT_POS = slice(0, 3)
QPOS_ROOT_QUAT = slice(3, 7)          # wxyz
QPOS_JOINTS = slice(7, 36)


def yaw_of(quat_wxyz: np.ndarray) -> np.ndarray:
    """Heading angle (rad) of ``[..., 4]`` wxyz quaternions, about +z."""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def period_net_se2(qpos: np.ndarray) -> np.ndarray:
    """Per-period displacement of one stride, IN THE BODY FRAME OF ITS FIRST FRAME.

    Returns ``[dx, dy, dz, dyaw]``: how far the root travels forward / left / up and how far it turns
    over one full cycle (frame 0 -> frame ``F-1``).  Body frame, so it is directly comparable to the
    manifest's ``steady_vel = (vel_x, vel_y, vel_yaw)`` and independent of however the stride happens to
    be oriented in the source library.

    ``dyaw`` is wrapped to ``[-pi, pi]``.  Safe for this bank -- the largest |wz| is 0.8 rad/s over a
    2.14 s period, i.e. 1.7 rad -- but a stride turning more than half a turn per cycle would alias.
    """
    q = np.asarray(qpos, dtype=np.float64)
    p0, p1 = q[0, QPOS_ROOT_POS], q[-1, QPOS_ROOT_POS]
    yaw0, yaw1 = float(yaw_of(q[0, QPOS_ROOT_QUAT])), float(yaw_of(q[-1, QPOS_ROOT_QUAT]))
    d = p1 - p0
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    dyaw = (yaw1 - yaw0 + np.pi) % (2.0 * np.pi) - np.pi
    return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2], dyaw], dtype=np.float32)


def twist_from_net(net_se2: np.ndarray, period_s) -> np.ndarray:
    """Constant body twist ``(vx, vy, wz)`` that produces per-period displacement ``net_se2`` in ``T``.

    The SE(2) logarithm, NOT ``net / T``.  A turning stride's frame rotates as it goes, so the straight
    body-frame displacement from start to end is the arc's CHORD and is shorter than the distance
    actually travelled -- dividing it by the period under-reports forward speed by ``sinc(theta/2)``.
    Small on any one stride, but it is exactly the quantity the RL is graded against: measured on the
    real 251-stride bank, the naive form disagrees with the manifest command by p95 0.096 m/s across the
    144 strides that turn, while this form agrees to the solver's own residual.

    Given ``theta = dyaw`` and ``V(theta) = [[A, -B], [B, A]]`` with ``A = sin(theta)/theta`` and
    ``B = (1-cos(theta))/theta``, the net satisfies ``[dx,dy] = V @ [vx,vy] * T``; this inverts it.
    Degrades continuously to ``net / T`` as ``theta -> 0``.

    Args:
        net_se2: ``[N,4]`` or ``[4]`` body-frame ``(dx, dy, dz, dyaw)`` per period.
        period_s: ``[N]`` or scalar period duration (s).

    Returns ``[N,3]`` (or ``[3]``) float32 ``(vx, vy, wz)``.
    """
    net = np.atleast_2d(np.asarray(net_se2, dtype=np.float64))
    t = np.atleast_1d(np.asarray(period_s, dtype=np.float64))
    dx, dy, th = net[:, 0], net[:, 1], net[:, 3]
    small = np.abs(th) < 1e-8
    safe = np.where(small, 1.0, th)                        # avoid 0/0 in the unused branch
    a = np.where(small, 1.0, np.sin(safe) / safe)
    b = np.where(small, 0.0, (1.0 - np.cos(safe)) / safe)
    den = np.maximum(a * a + b * b, 1e-12)
    vx = (a * dx + b * dy) / den / t
    vy = (-b * dx + a * dy) / den / t
    out = np.stack([vx, vy, th / t], axis=-1).astype(np.float32)
    return out[0] if np.ndim(net_se2) == 1 else out


def sample_frames(qpos: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Interpolate a stride at FRACTIONAL frame positions ``taps``, wrapping through the closure frame.

    Shared by both halves of generation, which need windows at fractional positions for different
    reasons: Motion Bricks runs at 30 Hz while the bank is stored at the 50 Hz control period, so a
    4-frame target window is spaced 5/3 bank frames apart; and a transient's context is the outgoing
    loop's tail at that same spacing.  Keeping one routine means the pose Motion Bricks is aimed at and
    the pose the assembled clip actually contains are interpolated identically.

    Positions wrap modulo the period, so a window may straddle the seam.  Quaternions get a
    hemisphere fix before interpolation and are renormalized after.
    """
    s = np.asarray(qpos)
    p = s.shape[0] - 1                                     # distinct period; frame p == frame 0
    taps = np.asarray(taps, dtype=np.float64) % p
    lo = np.floor(taps)
    i0 = lo.astype(np.int64) % p
    i1 = (i0 + 1) % p
    w = (taps - lo)[:, None].astype(np.float32)
    out = s[i0] * (1.0 - w) + s[i1] * w
    q0, q1 = s[i0, QPOS_ROOT_QUAT], s[i1, QPOS_ROOT_QUAT]
    q1 = np.where((q0 * q1).sum(-1, keepdims=True) < 0.0, -q1, q1)     # kill double-cover sign flips
    q = q0 * (1.0 - w) + q1 * w
    out[:, QPOS_ROOT_QUAT] = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    return np.ascontiguousarray(out, dtype=np.float32)


def phase_window(qpos: np.ndarray, phase: float, n: int, step: float) -> np.ndarray:
    """``[n, 36]``: ``n`` samples ``step`` bank-frames apart, starting at bank-frame ``phase``.

    ``step`` is ``mb_dt / bank_dt`` for a window handed to Motion Bricks (its frames are ``mb_dt`` apart
    in time, not one bank frame apart).  Use a negative offset for a TRAILING window, e.g. the outgoing
    loop's last ``n`` frames are ``phase_window(s, phase_out - (n - 1) * step, n, step)``.
    """
    return sample_frames(qpos, phase + np.arange(int(n)) * float(step))


def closure_residual(qpos: np.ndarray) -> tuple[float, float]:
    """``(joint_gap, height_gap)`` between frame ``F-1`` and frame 0 -- 0 for a properly closed cycle.

    A non-zero joint gap means the stride does NOT close, so looping it would step at every period seam
    (and the loop would need blending).  Reported by the exporter so a bad stride is visible up front
    rather than as a mystery artefact in a trained policy.
    """
    q = np.asarray(qpos, dtype=np.float64)
    return (float(np.abs(q[-1, QPOS_JOINTS] - q[0, QPOS_JOINTS]).max()),
            float(abs(q[-1, 2] - q[0, 2])))


@dataclass
class StrideBank:
    """A loaded stride bank.  ``strides[i]`` is ``qpos[F_i, 36]``; every other array is indexed by ``i``."""

    strides: list                  # list of float32 [F_i, 36]
    steady_vel: np.ndarray         # [N,3] COMMANDED (vx, vy, wz) from the traj-opt manifest
    periodic: np.ndarray           # [N] uint8
    net_se2: np.ndarray            # [N,4] MEASURED per-period (dx, dy, dz, dyaw), body frame
    dt: float                      # bank sampling period (s) -- the RL control period, not MB's fps
    names: list                    # [N] str
    meta: dict
    # The SOLVER's own per-foot contact schedule, [N] of uint8 [F_i, 2], or None per stride for a bank
    # written before it was carried.  Last, with a default, so old call sites keep working; padded in
    # __post_init__ so ``bank.contact[i]`` is always safe to index.
    contact: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.contact) < len(self.strides):
            self.contact = list(self.contact) + [None] * (len(self.strides) - len(self.contact))

    @property
    def period_frames(self) -> np.ndarray:
        """[N] distinct period ``P = F - 1`` (the closure frame is the wrap point, not a new frame)."""
        return np.array([s.shape[0] - 1 for s in self.strides], dtype=np.int64)

    @property
    def period_s(self) -> np.ndarray:
        """[N] period duration in seconds."""
        return self.period_frames * self.dt

    @property
    def achieved_vel(self) -> np.ndarray:
        """[N,3] the constant body twist ``(vx, vy, wz)`` a looped replay ACTUALLY tracks.

        This -- not ``steady_vel`` -- is what a clip built from this stride should report as its
        velocity: it is derived from the motion, so it cannot drift from what the replay actually does,
        whereas the manifest is an input to the solver.  On the real bank the two agree to p95 0.0003
        m/s, so this is a correctness guarantee rather than a correction.
        """
        return twist_from_net(self.net_se2, self.period_s)

    def __len__(self) -> int:
        return len(self.strides)


def load_stride_bank(path: str) -> StrideBank:
    """Read a stride-bank ``.npz`` written by ``export_stride_bank.py``.

    Entries have different cycle lengths, so the file stores one flat ``qpos_all`` plus per-entry
    offsets (pickle-free).  ``net_se2`` is derived here rather than read, so it cannot disagree with the
    motion it describes.
    """
    d = np.load(path, allow_pickle=False)
    qpos_all, offsets = d["qpos_all"], d["offsets"]
    strides = [np.ascontiguousarray(qpos_all[offsets[i]:offsets[i + 1]])
               for i in range(len(offsets) - 1)]
    meta = json.loads(str(d["meta"]))
    n = len(strides)
    periodic = d["periodic"] if "periodic" in d else np.ones(n, dtype=np.uint8)
    net = np.stack([period_net_se2(s) for s in strides], axis=0) if n else np.zeros((0, 4), np.float32)
    # Older banks predate the solver contact schedule; absent is represented as None per stride so a
    # caller has to decide what to do rather than silently getting a zero-filled "never in contact".
    ca = d["contact_all"] if "contact_all" in d else None
    contact = ([np.ascontiguousarray(ca[offsets[i]:offsets[i + 1]]) for i in range(n)]
               if ca is not None else [None] * n)
    return StrideBank(
        strides=strides,
        steady_vel=d["steady_vel"].astype(np.float32),
        periodic=np.asarray(periodic, dtype=np.uint8),
        net_se2=net,
        dt=float(meta["dt"]),
        names=list(meta.get("names", [f"stride_{i}" for i in range(n)])),
        meta=meta,
        contact=contact,
    )


def save_stride_bank(path: str, strides: list, steady_vel: np.ndarray, periodic: np.ndarray,
                     meta: dict, contact: list | None = None) -> None:
    """Write a stride bank.  ``strides`` is a list of ``qpos[F_i, 36]`` float32 arrays.

    ``contact`` is the optional per-stride ``[F_i, 2]`` solver contact schedule, stored on the SAME
    offsets as the poses so a loop replay can slice both identically.
    """
    for i, s in enumerate(strides):
        if s.ndim != 2 or s.shape[1] != 36:
            raise ValueError(f"stride {i}: expected qpos [F,36], got {s.shape}")
    offsets = np.zeros(len(strides) + 1, dtype=np.int64)
    np.cumsum([s.shape[0] for s in strides], out=offsets[1:])
    np.savez_compressed(
        path,
        qpos_all=np.concatenate(strides, axis=0).astype(np.float32),
        offsets=offsets,
        steady_vel=np.asarray(steady_vel, dtype=np.float32),
        periodic=np.asarray(periodic, dtype=np.uint8),
        meta=np.array(json.dumps(meta)),
        **({"contact_all": np.concatenate(contact, axis=0).astype(np.uint8)}
           if contact is not None and all(c is not None for c in contact) else {}),
    )
