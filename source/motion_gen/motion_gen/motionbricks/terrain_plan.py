# Copyright 2026 Zachary Olkin. All rights reserved.

"""The TERRAIN PLAN -- the second Stage-2 <-> Stage-1 contract, alongside :mod:`.raw_schema`.

A terrain clip needs information from both environments: the traj-opt terrain reference and the authored
tile geometry live behind ``motion_gen`` (``env_isaaclab``), while the approach / exit motion is generated
by Motion Bricks (its own conda env).  So the terrain half is computed FIRST and written out as a plan:

1. **Stage 2a** (``env_isaaclab``): pick a reference, author the tile, place the replay on it, filter the
   boundary frames -> :class:`TerrainPlan`.
2. **Stage 1** (Motion Bricks env): read the plan; generate an approach that ends at ``entry_frame`` and
   an exit that starts at ``exit_frame`` -> two :class:`~.raw_schema.RawMotion` files.
3. **Stage 2b** (``env_isaaclab``): read the plan + both raw motions, splice, write the clip ``.h5``.

Because the plan carries the PLACED replay itself (not a recipe for re-deriving it), Stage 2b never
recomputes the placement -- there is exactly one terrain segment and both stages see the same frames.

Like :mod:`.raw_schema` this module is numpy-only with no intra-package imports, so the Motion Bricks env
can load it straight from its source file (see ``scripts/rsl_rl/mb_env.py``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

_QPOS_DIM = 36


def _slerp_wxyz(q0: np.ndarray, q1: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Shortest-arc quaternion interpolation between ``q0[N,4]`` and ``q1[N,4]`` at weights ``w[N]``."""
    q1 = np.where((q0 * q1).sum(-1, keepdims=True) < 0.0, -q1, q1)
    dot = np.clip((q0 * q1).sum(-1), -1.0, 1.0)
    theta = np.arccos(dot)
    sin_t = np.sin(theta)
    close = sin_t < 1e-4
    safe = np.where(close, 1.0, sin_t)
    a = (np.sin((1.0 - w) * theta) / safe)[:, None]
    b = (np.sin(w * theta) / safe)[:, None]
    out = np.where(close[:, None], q0 * (1.0 - w)[:, None] + q1 * w[:, None], a * q0 + b * q1)
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def sample_qpos(qpos: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Sample MuJoCo ``qpos[F,36]`` at FRACTIONAL frame indices ``idx[N]`` (clamped to the clip).

    Root position and joints interpolate linearly, the root quaternion spherically.  Used to read a
    terrain window at the Motion Bricks frame rate out of a plan stored at the RL control dt.
    """
    qpos = np.asarray(qpos, dtype=np.float64)
    f = np.clip(np.asarray(idx, dtype=np.float64), 0.0, qpos.shape[0] - 1.0)
    i0 = np.floor(f).astype(np.int64)
    i1 = np.minimum(i0 + 1, qpos.shape[0] - 1)
    w = f - i0
    out = np.empty((f.shape[0], _QPOS_DIM), dtype=np.float32)
    lin = np.concatenate([np.arange(0, 3), np.arange(7, _QPOS_DIM)])
    out[:, lin] = (qpos[i0][:, lin] * (1.0 - w)[:, None] + qpos[i1][:, lin] * w[:, None])
    out[:, 3:7] = _slerp_wxyz(qpos[i0][:, 3:7], qpos[i1][:, 3:7], w)
    return out


@dataclass
class TerrainPlan:
    """One planned terrain tile traversal: the placed replay plus everything the MB side needs around it."""

    skill: str                        # terrain family / library skill (see .terrain_families)
    dt: float                         # sampling period of ``terrain_qpos`` (the RL control dt)
    terrain_qpos: np.ndarray          # [F, 36] placed replay in TILE frame: root_pos, root_quat_wxyz, joints
    terrain_contact: np.ndarray       # [F, 2] uint8 per-foot in-contact flags of the replay
    entry_frame: int                  # replay frame the MB approach hands off to
    exit_frame: int                   # replay frame the MB exit picks up from
    approach_surf_z: float            # walkable surface height the approach runs on (tile frame)
    exit_surf_z: float                # walkable surface height the exit runs on (tile frame)
    approach_vx: float                # steady forward command for the approach (m/s, unsigned)
    exit_vx: float                    # steady command for the exit (m/s, SIGNED along tile +x)
    entry_vx: float                   # replay's forward speed AT ``entry_frame`` (approach ramp target)
    exit_speed: float                 # replay's signed speed AT ``exit_frame`` (exit ramp start)
    approach_mode: str                # "walk" | "stand"
    exit_mode: str                    # "walk" | "stand"
    approach_s: float                 # requested approach duration (s)
    exit_s: float                     # requested exit duration (s)
    # Candidate frames the boundary gates accepted, kept so an alternative can be tried without re-deriving.
    # NOTE the two index spaces: ``exit_candidates`` index ``terrain_qpos`` (same space as ``exit_frame``),
    # but ``entry_candidates`` index the SOURCE PERIOD of the pre-placement traversal -- choosing one
    # re-anchors the whole replay onto the tile (the chosen frame becomes ``entry_frame`` = 0), so they
    # cannot both live in the final frame space.  ``meta["entry_source_frame"]`` records which one was used.
    entry_candidates: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    exit_candidates: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    meta: dict = field(default_factory=dict)   # tile geometry (``boxes``), reference name, gate diagnostics

    def __post_init__(self) -> None:
        """Validate shapes and clamp the boundary frames into the replay."""
        self.terrain_qpos = np.asarray(self.terrain_qpos, dtype=np.float32)
        self.terrain_contact = np.asarray(self.terrain_contact, dtype=np.uint8)
        f = self.terrain_qpos.shape[0]
        if self.terrain_qpos.shape != (f, _QPOS_DIM):
            raise ValueError(f"terrain_qpos must be [F, {_QPOS_DIM}], got {self.terrain_qpos.shape}")
        if self.terrain_contact.shape != (f, 2):
            raise ValueError(f"terrain_contact must be [F, 2] matching F={f}, got {self.terrain_contact.shape}")
        if not (0 <= int(self.entry_frame) < f) or not (0 <= int(self.exit_frame) < f):
            raise ValueError(f"boundary frames ({self.entry_frame}, {self.exit_frame}) outside the replay F={f}")
        if int(self.exit_frame) <= int(self.entry_frame):
            raise ValueError(f"exit_frame {self.exit_frame} must be after entry_frame {self.entry_frame}")
        self.entry_frame, self.exit_frame = int(self.entry_frame), int(self.exit_frame)
        self.entry_candidates = np.asarray(self.entry_candidates, dtype=np.int64).reshape(-1)
        self.exit_candidates = np.asarray(self.exit_candidates, dtype=np.int64).reshape(-1)

    # ---- geometry / bookkeeping ----------------------------------------------------------------
    @property
    def n_frames(self) -> int:
        """Number of frames in the placed replay (before it is cut to the boundary frames)."""
        return int(self.terrain_qpos.shape[0])

    @property
    def replay(self) -> np.ndarray:
        """The replay CUT to its boundary frames -- the segment that actually enters the clip ``[n, 36]``."""
        return self.terrain_qpos[self.entry_frame:self.exit_frame + 1]

    @property
    def replay_contact(self) -> np.ndarray:
        """Per-foot contacts of :attr:`replay` ``[n, 2]``."""
        return self.terrain_contact[self.entry_frame:self.exit_frame + 1]

    @property
    def boxes(self) -> list[dict]:
        """The authored tile geometry (``{pos, size, yaw}`` full extents, tile frame)."""
        return list(self.meta.get("boxes", []))

    # ---- windows for the Motion Bricks side ---------------------------------------------------
    def approach_pin_window(self, t_offset: float, n: int, dt_mb: float) -> np.ndarray:
        """Target-pose window ``[n, 36]`` for an approach that is ``-t_offset`` seconds from the entry.

        The window is the replay itself, read at the Motion Bricks frame spacing ``dt_mb`` starting
        ``t_offset`` seconds from :attr:`entry_frame` (negative = before the entry, so the approach is still
        in-betweening toward it; positive = past it, i.e. the terrain motion the approach is about to hand
        over to).  Pinning the terrain's FUTURE as the target -- rather than freezing the entry pose -- is
        what keeps the last approach frames moving at the entry's velocity instead of settling onto it.

        Root xy is zeroed (the planar target comes from the exact-velocity pin, which overwrites it) and
        root z is made RELATIVE to :attr:`approach_surf_z`, because the MB approach is generated on its own
        flat ground and only grounded onto the real surface when the clip is spliced.
        """
        idx = self.entry_frame + (float(t_offset) + np.arange(int(n)) * float(dt_mb)) / float(self.dt)
        win = sample_qpos(self.terrain_qpos, idx)
        win[:, 0:2] = 0.0
        win[:, 2] -= float(self.approach_surf_z)
        return win

    def exit_seed_window(self, n: int, dt_mb: float) -> np.ndarray:
        """Context (history) window ``[n, 36]`` ENDING at :attr:`exit_frame`, in TILE coordinates.

        Seeding Motion Bricks with this window makes the generated walk-away start ON the terrain -- its
        history IS the replay's last frames -- so the exit is continuous with the replay by construction
        rather than by a seam ease.  Sampled at the MB frame spacing ``dt_mb`` (NOT the plan's dt) so the
        history carries the true velocity: a window of consecutive 50 Hz frames fed to a 30 Hz model reads
        as a robot moving at 60% speed.
        """
        idx = self.exit_frame - (int(n) - 1 - np.arange(int(n))) * float(dt_mb) / float(self.dt)
        return sample_qpos(self.terrain_qpos, idx)


def save_terrain_plan(path: str, plan: TerrainPlan) -> None:
    """Write a :class:`TerrainPlan` to ``path`` as a compressed, pickle-free ``.npz``."""
    scalars = {
        "skill": plan.skill, "dt": plan.dt,
        "entry_frame": plan.entry_frame, "exit_frame": plan.exit_frame,
        "approach_surf_z": plan.approach_surf_z, "exit_surf_z": plan.exit_surf_z,
        "approach_vx": plan.approach_vx, "exit_vx": plan.exit_vx,
        "entry_vx": plan.entry_vx, "exit_speed": plan.exit_speed,
        "approach_mode": plan.approach_mode, "exit_mode": plan.exit_mode,
        "approach_s": plan.approach_s, "exit_s": plan.exit_s,
        "meta": plan.meta,
    }
    np.savez_compressed(
        path,
        terrain_qpos=plan.terrain_qpos, terrain_contact=plan.terrain_contact,
        entry_candidates=plan.entry_candidates, exit_candidates=plan.exit_candidates,
        scalars=np.array(json.dumps(scalars)),
    )


def load_terrain_plan(path: str) -> TerrainPlan:
    """Load a :class:`TerrainPlan` previously written by :func:`save_terrain_plan`."""
    with np.load(path, allow_pickle=False) as d:
        s = json.loads(str(d["scalars"]))
        return TerrainPlan(
            skill=s["skill"], dt=float(s["dt"]),
            terrain_qpos=d["terrain_qpos"], terrain_contact=d["terrain_contact"],
            entry_frame=int(s["entry_frame"]), exit_frame=int(s["exit_frame"]),
            approach_surf_z=float(s["approach_surf_z"]), exit_surf_z=float(s["exit_surf_z"]),
            approach_vx=float(s["approach_vx"]), exit_vx=float(s["exit_vx"]),
            entry_vx=float(s["entry_vx"]), exit_speed=float(s["exit_speed"]),
            approach_mode=s["approach_mode"], exit_mode=s["exit_mode"],
            approach_s=float(s["approach_s"]), exit_s=float(s["exit_s"]),
            entry_candidates=d["entry_candidates"], exit_candidates=d["exit_candidates"],
            meta=s.get("meta", {}),
        )
