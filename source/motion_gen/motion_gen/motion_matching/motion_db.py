# Copyright 2026 Zachary Olkin. All rights reserved.

"""The motion-matching database: per-frame states + descriptors + nearest-neighbor search.

A :class:`MotionSource` is one contiguous motion (a bones-seed clip or a traj-opt
stride, densely sampled at ``dt``) given as world-frame per-frame robot state.  The
:class:`MotionDB` flattens every source's frames into one table, computes the
contact-free descriptors (:mod:`.features`), z-scores + weights them, and searches by
weighted-L2 plus an optional velocity-bias term that pulls matches toward a commanded
future velocity.

Search is exact brute force (a chunked ``cdist``); at ~1e5-1e6 frames on GPU this is
trivially fast and avoids the approximate-index complexity, and -- unlike a prebuilt
KD-tree -- it composes cleanly with the per-query velocity-bias reweighting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import features as F

# source_kind codes
BONES = 0
TRAJOPT = 1


@dataclass
class MotionSource:
    """One contiguous motion in world-frame per-frame state (all tensors ``[Fi, ...]``)."""

    joint_pos: torch.Tensor       # [Fi, 21]
    joint_vel: torch.Tensor       # [Fi, 21]
    root_pos: torch.Tensor        # [Fi, 3]  world
    root_quat: torch.Tensor       # [Fi, 4]  wxyz, world
    root_lin_vel: torch.Tensor    # [Fi, 3]  world
    root_ang_vel: torch.Tensor    # [Fi, 3]  world
    source_kind: int              # BONES | TRAJOPT
    skill_id: int = -1            # traj-opt skill index; -1 for bones/unknown
    stride_idx: torch.Tensor | None = None  # [Fi] traj-opt global stride idx (handoff), else None
    phase: torch.Tensor | None = None       # [Fi] traj-opt phase in [0,1] (handoff), else None
    wrap: str = "episodic"                   # "periodic" (full-cycle strides) | "episodic" (recordings)
    name: str = ""

    @property
    def n(self) -> int:
        return int(self.joint_pos.shape[0])


def _yaw_planar_vel(root_quat, root_lin_vel, root_ang_vel):
    """Per-frame ``(vx, vy, wz)`` in the root yaw-aligned frame (command convention)."""
    yq_inv = F.quat_inv(F.yaw_quat(root_quat))
    lin = F.quat_apply(yq_inv, root_lin_vel)
    ang = F.quat_apply(yq_inv, root_ang_vel)
    return torch.stack([lin[:, 0], lin[:, 1], ang[:, 2]], dim=-1)   # [Fi, 3]


class MotionDB:
    """Flattened per-frame table + descriptors + brute-force NN search."""

    def __init__(self, cfg: F.FeatureConfig, device: str | torch.device = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        # Per-frame state (filled by build/load)
        self.joint_pos: torch.Tensor
        self.joint_vel: torch.Tensor
        self.root_pos: torch.Tensor
        self.root_quat: torch.Tensor
        self.root_lin_vel: torch.Tensor
        self.root_ang_vel: torch.Tensor
        # Per-frame metadata
        self.owner: torch.Tensor          # [N] source index
        self.local: torch.Tensor          # [N] frame index within its source
        self.valid: torch.Tensor          # [N] bool: full offset window available
        self.source_kind: torch.Tensor    # [N]
        self.skill_id: torch.Tensor       # [N]
        self.stride_idx: torch.Tensor     # [N] (-1 where n/a)
        self.phase: torch.Tensor          # [N] (nan where n/a)
        self.future_vel: torch.Tensor     # [N, 3] near-future (vx,vy,wz) yaw-aligned
        # Descriptors + normalization
        self.desc_norm: torch.Tensor      # [N, D] z-scored + weighted (search space)
        self.mean: torch.Tensor           # [D]
        self.std: torch.Tensor            # [D]
        self.weight: torch.Tensor         # [D]
        self.source_names: list[str] = []

    # ---- build -------------------------------------------------------------
    def build(self, sources: list[MotionSource], fk) -> "MotionDB":
        cfg = self.cfg
        dev = self.device
        offs = cfg.offset_frames()
        fut_local = [i for i, o in enumerate(offs) if o > 0]   # future offset positions
        raw_desc, states, meta, fvel = [], [], [], []
        for si, s in enumerate(sources):
            jp = s.joint_pos.to(dev); jv = s.joint_vel.to(dev)
            rq = s.root_quat.to(dev); rl = s.root_lin_vel.to(dev); ra = s.root_ang_vel.to(dev)
            Fi = s.n
            feat = F.compute_frame_features(jp, jv, rq, rl, ra, fk, cfg)   # [Fi, pfd]
            desc = F.assemble_descriptors(feat, cfg, wrap=s.wrap)          # [Fi, D]
            # near-future planar velocity = mean over the future offset frames (same wrap)
            planar = _yaw_planar_vel(rq, rl, ra)                          # [Fi, 3]
            idx = F.offset_index(Fi, offs, s.wrap, device=dev)            # [Fi, O]
            planar_off = planar[idx.reshape(-1)].reshape(Fi, len(offs), 3)
            fv = planar_off[:, fut_local].mean(dim=1) if fut_local else planar   # [Fi, 3]

            raw_desc.append(desc)
            fvel.append(fv)
            states.append((jp, jv, s.root_pos.to(dev), rq, rl, ra))
            stride = (s.stride_idx.to(dev) if s.stride_idx is not None
                      else torch.full((Fi,), -1, dtype=torch.long, device=dev))
            phase = (s.phase.to(dev) if s.phase is not None
                     else torch.full((Fi,), float("nan"), device=dev))
            meta.append((
                torch.full((Fi,), si, dtype=torch.long, device=dev),
                torch.arange(Fi, dtype=torch.long, device=dev),
                torch.full((Fi,), s.source_kind, dtype=torch.long, device=dev),
                torch.full((Fi,), s.skill_id, dtype=torch.long, device=dev),
                stride, phase,
            ))
            self.source_names.append(s.name or f"src{si}")

        self.joint_pos = torch.cat([s[0] for s in states])
        self.joint_vel = torch.cat([s[1] for s in states])
        self.root_pos = torch.cat([s[2] for s in states])
        self.root_quat = torch.cat([s[3] for s in states])
        self.root_lin_vel = torch.cat([s[4] for s in states])
        self.root_ang_vel = torch.cat([s[5] for s in states])
        self.owner = torch.cat([m[0] for m in meta])
        self.local = torch.cat([m[1] for m in meta])
        self.source_kind = torch.cat([m[2] for m in meta])
        self.skill_id = torch.cat([m[3] for m in meta])
        self.stride_idx = torch.cat([m[4] for m in meta])
        self.phase = torch.cat([m[5] for m in meta])
        self.future_vel = torch.cat(fvel)
        raw = torch.cat(raw_desc)                                        # [N, D]
        # Every frame gets a full-window descriptor (edges wrapped/clamped per source),
        # so all frames are searchable.
        self.valid = torch.ones(raw.shape[0], dtype=torch.bool, device=dev)

        # z-score across all frames, then per-group weight (tiled across offsets)
        mean = raw.mean(dim=0)
        std = raw.std(dim=0).clamp_min(1e-6)
        w1 = F.group_weight_vector(cfg, device=dev)                      # [pfd]
        weight = w1.repeat(len(offs))                                    # [D]
        self.mean, self.std, self.weight = mean, std, weight
        self.desc_raw = raw                                              # [N, D] pre-normalization
        self.desc_norm = ((raw - mean) / std) * weight
        return self

    # ---- search ------------------------------------------------------------
    def _normalize(self, raw_query: torch.Tensor) -> torch.Tensor:
        return ((raw_query.to(self.device) - self.mean) / self.std) * self.weight

    def nn_search(
        self,
        raw_query: torch.Tensor,          # [Q, D] raw descriptor(s)
        vel_bias: torch.Tensor | None = None,   # [Q, 3] target (vx,vy,wz), yaw frame
        w_vel: float = 0.0,
        candidate_mask: torch.Tensor | None = None,  # [N] bool: allowed frames
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Nearest frame per query. Returns ``(global_idx[Q], cost[Q], per_query_min)``.

        Only ``valid`` frames (and ``candidate_mask`` if given) are eligible.
        """
        # Exact Euclidean mode: the default matmul expansion loses precision and gives
        # a spurious ~1e-2 distance for identical vectors (breaks self-match / near-ties).
        mm = "donot_use_mm_for_euclid_dist"
        q = self._normalize(raw_query)                                   # [Q, D]
        cost = torch.cdist(q, self.desc_norm, compute_mode=mm)           # [Q, N]
        if vel_bias is not None and w_vel > 0.0:
            cost = cost + w_vel * torch.cdist(vel_bias.to(self.device), self.future_vel, compute_mode=mm)
        elig = self.valid.clone()
        if candidate_mask is not None:
            elig = elig & candidate_mask.to(self.device)
        cost = cost.masked_fill(~elig.unsqueeze(0), float("inf"))
        best_cost, best_idx = cost.min(dim=1)
        return best_idx, best_cost, best_cost

    # ---- persistence -------------------------------------------------------
    def save(self, path: str) -> None:
        d = dict(
            joint_pos=self.joint_pos.cpu().numpy(), joint_vel=self.joint_vel.cpu().numpy(),
            root_pos=self.root_pos.cpu().numpy(), root_quat=self.root_quat.cpu().numpy(),
            root_lin_vel=self.root_lin_vel.cpu().numpy(), root_ang_vel=self.root_ang_vel.cpu().numpy(),
            owner=self.owner.cpu().numpy(), local=self.local.cpu().numpy(),
            valid=self.valid.cpu().numpy(), source_kind=self.source_kind.cpu().numpy(),
            skill_id=self.skill_id.cpu().numpy(), stride_idx=self.stride_idx.cpu().numpy(),
            phase=self.phase.cpu().numpy(), future_vel=self.future_vel.cpu().numpy(),
            desc_raw=self.desc_raw.cpu().numpy(),
            desc_norm=self.desc_norm.cpu().numpy(), mean=self.mean.cpu().numpy(),
            std=self.std.cpu().numpy(), weight=self.weight.cpu().numpy(),
            source_names=np.array(self.source_names, dtype=object),
        )
        np.savez(path, **d)

    @classmethod
    def load(cls, path: str, cfg: F.FeatureConfig, device="cpu") -> "MotionDB":
        z = np.load(path, allow_pickle=True)
        db = cls(cfg, device)
        for k in ("joint_pos", "joint_vel", "root_pos", "root_quat", "root_lin_vel",
                  "root_ang_vel", "future_vel", "desc_raw", "desc_norm", "mean", "std", "weight"):
            setattr(db, k, torch.as_tensor(z[k], dtype=torch.float32, device=device))
        for k in ("owner", "local", "source_kind", "skill_id", "stride_idx"):
            setattr(db, k, torch.as_tensor(z[k], dtype=torch.long, device=device))
        db.valid = torch.as_tensor(z["valid"], dtype=torch.bool, device=device)
        db.phase = torch.as_tensor(z["phase"], dtype=torch.float32, device=device)
        db.source_names = list(z["source_names"])
        return db
