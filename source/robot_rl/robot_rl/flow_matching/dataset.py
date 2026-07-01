# Copyright 2026 Zachary Olkin. All rights reserved.

"""Turn recorded teacher rollouts into robot-state-conditioned FM training items.

The dataset operates on already-loaded numpy arrays plus a ``meta`` object (duck
-typed; only its layout attributes are read), so it has no dependency on the
recording scripts and is unit-testable with synthetic data. The training script
loads the ``.npz`` via ``recorded_data_io.load_recorded_data`` and passes
``(data, meta)`` here.

Paradigm (``target_source="ego_anchored_clip_future"``): each training item is
sampled at a **snap (re-anchor) event** of a recorded teacher rollout. It pairs

* ``history``: the ``N`` most-recent REAL robot-state steps ending at the snap, in
  the xyz+yaw ground frame anchored on the robot base at the snap (phase only);
* ``height``: the robot root z above ground at the snap (a heightmap stand-in);
* ``cmd_tokens``: the strided FUTURE velocity command over the F-step window;
* ``state`` (the target): the ``F``-node clip future **ego-anchored onto the robot**
  at the snap (recorded ``ref_future``), re-expressed in the SAME robot xyz+yaw
  frame -- node 0 offset ≈ 0 by construction.

One state normalizer (history + target share it), a command normalizer, and a 1-D
height normalizer. Snap events are exactly the FM's deploy re-inference points, so
the training anchors match the teacher's actual rollout.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import FlowMatchingConfig
from .normalizer import Normalizer
from .transforms import (
    FRAME_BLOCK_DIM,
    frame_block_start,
    orientation_slices,
    relative_state,
    target_state_dim,
)


def resolve_term_slice(term_names: list[str], term_dims: list[int], target: str) -> tuple[int, int]:
    """Return ``(start, end)`` of a named term within a concatenated vector.

    Args:
        term_names: Ordered term names.
        term_dims: Per-term widths (same order).
        target: The term to locate.

    Raises:
        KeyError: If ``target`` is not present.
    """
    offset = 0
    for name, dim in zip(term_names, term_dims):
        if name == target:
            return offset, offset + dim
        offset += dim
    raise KeyError(f"term {target!r} not found in {term_names}")


def _stack_frame_tensors(
    split_fn, frame_names: list[str], rows: int
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Gather extra-frame world tensors and flatten to ``(rows, K, ·)`` for relative_state.

    Args:
        split_fn: ``name -> (..., dim)`` slicer over the windowed state (the local
            ``_split``), where ``...`` flattens to ``rows``.
        frame_names: Ordered extra body-frame names; ``[]`` returns all ``None``.
        rows: Flattened leading dimension.

    Returns:
        ``(frame_pos_w, frame_quat_w, frame_lin_vel_w, frame_ang_vel_w)`` each
        ``(rows, K, ·)``, or all ``None`` when ``frame_names`` is empty.
    """
    if not frame_names:
        return None, None, None, None
    k = len(frame_names)
    fp = torch.stack([split_fn(f"{f}:pos_w") for f in frame_names], dim=-2).reshape(rows, k, 3)
    fq = torch.stack([split_fn(f"{f}:quat_w") for f in frame_names], dim=-2).reshape(rows, k, 4)
    fl = torch.stack([split_fn(f"{f}:lin_vel_w") for f in frame_names], dim=-2).reshape(rows, k, 3)
    fa = torch.stack([split_fn(f"{f}:ang_vel_w") for f in frame_names], dim=-2).reshape(rows, k, 3)
    return fp, fq, fl, fa


def snap_windows(
    episode_id: np.ndarray,
    step_in_episode: np.ndarray,
    done: np.ndarray,
    reanchor_active: np.ndarray,
    num_history: int,
    stride: int = 1,
) -> list[np.ndarray]:
    """History-row windows ending at each snap (re-anchor) event.

    Returns one int array of length ``num_history`` per usable snap: the ``N``
    consecutive rows of one episode ending at a row where ``reanchor_active`` is set
    (the FM's deploy re-inference points). A snap is usable iff it has ``N-1`` prior
    consecutive steps in the same episode with no ``done`` before the snap.

    Args:
        episode_id / step_in_episode / done / reanchor_active: Per-pair flat arrays.
        num_history: ``N`` history steps (the snap is the last of the ``N``).
        stride: Keep every ``stride``-th usable snap (per episode).

    Returns:
        List of int arrays, each of length ``num_history`` (last entry = the snap row).
    """
    order = np.lexsort((step_in_episode, episode_id))
    ep_sorted = episode_id[order]
    step_sorted = step_in_episode[order]
    done_sorted = done[order]
    snap_sorted = reanchor_active[order].astype(bool)

    out: list[np.ndarray] = []
    n = len(order)
    start = 0
    while start < n:
        end = start
        while end + 1 < n and ep_sorted[end + 1] == ep_sorted[start]:
            end += 1
        ep_rows = order[start : end + 1]
        ep_steps = step_sorted[start : end + 1]
        ep_done = done_sorted[start : end + 1]
        ep_snap = snap_sorted[start : end + 1]
        length = len(ep_rows)
        kept = 0
        for j in range(num_history - 1, length):
            if not ep_snap[j]:
                continue
            sl = slice(j - (num_history - 1), j + 1)
            if not np.all(np.diff(ep_steps[sl]) == 1):
                continue
            if np.any(ep_done[sl][:-1] != 0):  # no reset within the history window
                continue
            if kept % stride == 0:
                out.append(ep_rows[sl].copy())
            kept += 1
        start = end + 1
    return out


class FutureStateDataset(Dataset):
    """Robot-state-conditioned FM items: (history, height, cmd) -> ego-anchored clip future."""

    def __init__(
        self,
        data: dict[str, np.ndarray],
        meta: Any,
        cfg: FlowMatchingConfig,
        windows: list[np.ndarray],
        state_normalizer: Normalizer | None = None,
        cmd_normalizer: Normalizer | None = None,
        height_normalizer: Normalizer | None = None,
        add_noise: bool = False,
    ):
        """Build and (optionally) fit normalizers over the provided snap windows.

        Args:
            data: Recorded arrays (``state``, ``ref_future``, ``cmd_future``).
            meta: Layout metadata (``state_term_names/dims``, ``state_frame_names``,
                ``ref_future_term_names/dims``, ``ref_future_len``).
            cfg: Config; ``state_dim``/``cmd_dim``/``history_dim`` written back here.
            windows: History-row windows (length ``N``) from :func:`snap_windows`.
            state_normalizer / cmd_normalizer / height_normalizer: reused on the val
                split; fit on the train split when ``None``.
            add_noise: Add resampled observation noise to the history tokens in
                ``__getitem__`` (train split only); gated by ``cfg.obs_noise``.
        """
        self.cfg = cfg
        f = cfg.future_len
        n = cfg.num_history_states
        self.N = n
        if n < 1:
            raise ValueError(f"num_history_states={n} must be >= 1.")
        if cfg.target_source != "ego_anchored_clip_future":
            raise NotImplementedError(
                f"FutureStateDataset on this branch supports only "
                f"target_source='ego_anchored_clip_future' (got {cfg.target_source!r}); "
                "the robot-state-conditioned FM trains on recorded ego-anchored clip futures."
            )
        if data.get("ref_future") is None:
            raise ValueError(
                "target_source='ego_anchored_clip_future' requires a 'ref_future' array; "
                "re-record with scripts/rsl_rl/record_data.py (ego_anchor_fraction=1.0)."
            )
        if data.get("height_scan") is None:
            raise ValueError(
                "the height-scan FM requires a 'height_scan' array; re-record with a "
                "height_scanner sensor active (e.g. --task G1-teacher-clf-record-play)."
            )
        if len(windows) == 0:
            raise ValueError("No valid snap windows; check reanchor_active vs num_history_states.")

        # -- resolve state layout (the node layout of ref_future == state layout) --
        s_names, s_dims = list(meta.state_term_names), list(meta.state_term_dims)
        s_slc = {name: resolve_term_slice(s_names, s_dims, name) for name in s_names}
        nj = s_dims[s_names.index("joint_pos")]
        self.num_joints = nj
        recorded_frames = list(getattr(meta, "state_frame_names", []) or [])
        self.target_frame_names = list(recorded_frames)
        self.num_extra_frames = len(self.target_frame_names)
        self.state_dim = target_state_dim(nj, self.num_extra_frames)
        cfg.state_dim = self.state_dim
        cfg.history_dim = self.state_dim

        rf = data["ref_future"]
        if rf.shape[1] != f:
            raise ValueError(
                f"ref_future has F={rf.shape[1]} nodes but cfg.future_len={f}; "
                "re-record with --ref_future_len matching future_len."
            )

        # -- command (strided future velocity profile from cmd_future) --
        self.cmd_array = cfg.cmd_array
        cmd_future = data.get("cmd_future")
        if self.cmd_array is not None and cmd_future is None:
            raise ValueError("cmd_array set but the recording has no 'cmd_future' array.")
        cmd_dim = int(cmd_future.shape[-1]) if (self.cmd_array is not None and cmd_future is not None) else 0
        self.cmd_dim = cmd_dim
        cfg.cmd_dim = cmd_dim
        self.cmd_steps = list(range(0, f, cfg.cmd_stride))
        self.num_cmd_tokens = len(self.cmd_steps) if cmd_dim > 0 else 0

        idx_hist = np.stack(windows).astype(np.int64)  # (Nw, N) history rows; last = snap row
        snap_rows = idx_hist[:, -1]                    # (Nw,)
        n_win = idx_hist.shape[0]

        state_t = torch.from_numpy(np.ascontiguousarray(data["state"])).float()
        rf_t = torch.from_numpy(np.ascontiguousarray(rf)).float()  # (P, F, node)

        # Anchor = robot base at the snap row (xyz+yaw, applied inside relative_state).
        root_pos_slc, root_quat_slc = s_slc["root_pos_w"], s_slc["root_quat_w"]
        p_anchor = state_t[snap_rows, root_pos_slc[0]:root_pos_slc[1]]    # (Nw, 3)
        q_anchor = state_t[snap_rows, root_quat_slc[0]:root_quat_slc[1]]  # (Nw, 4)

        # -- history: N real robot-state rows ending at the snap, in the anchor frame --
        hw = state_t[idx_hist]  # (Nw, N, state_raw)

        def _split_hist(name: str) -> torch.Tensor:
            a, b = s_slc[name]
            return hw[..., a:b]

        m_h = n_win * n
        pr_h = p_anchor[:, None, :].expand(n_win, n, 3).reshape(m_h, 3)
        qr_h = q_anchor[:, None, :].expand(n_win, n, 4).reshape(m_h, 4)
        fp, fq, fl, fa = _stack_frame_tensors(_split_hist, self.target_frame_names, m_h)
        history = relative_state(
            _split_hist("root_pos_w").reshape(m_h, 3), _split_hist("root_quat_w").reshape(m_h, 4),
            _split_hist("root_link_lin_vel_w").reshape(m_h, 3), _split_hist("root_link_ang_vel_w").reshape(m_h, 3),
            _split_hist("joint_pos").reshape(m_h, nj), _split_hist("joint_vel").reshape(m_h, nj),
            pr_h, qr_h, frame_pos_w=fp, frame_quat_w=fq, frame_lin_vel_w=fl, frame_ang_vel_w=fa,
        ).reshape(n_win, n, self.state_dim).contiguous()

        # -- target: the F-node ego-anchored clip future at the snap, in the same frame --
        tw = rf_t[snap_rows]  # (Nw, F, node)

        def _split_tgt(name: str) -> torch.Tensor:
            a, b = s_slc[name]
            return tw[..., a:b]

        m_t = n_win * f
        pr_t = p_anchor[:, None, :].expand(n_win, f, 3).reshape(m_t, 3)
        qr_t = q_anchor[:, None, :].expand(n_win, f, 4).reshape(m_t, 4)
        fpt, fqt, flt, fat = _stack_frame_tensors(_split_tgt, self.target_frame_names, m_t)
        target = relative_state(
            _split_tgt("root_pos_w").reshape(m_t, 3), _split_tgt("root_quat_w").reshape(m_t, 4),
            _split_tgt("root_link_lin_vel_w").reshape(m_t, 3), _split_tgt("root_link_ang_vel_w").reshape(m_t, 3),
            _split_tgt("joint_pos").reshape(m_t, nj), _split_tgt("joint_vel").reshape(m_t, nj),
            pr_t, qr_t, frame_pos_w=fpt, frame_quat_w=fqt, frame_lin_vel_w=flt, frame_ang_vel_w=fat,
        ).reshape(n_win, f, self.state_dim).contiguous()

        # node-0 invariant: the clip ego-anchored at the snap sits on the robot root,
        # so target node 0's root position ≈ 0 in the robot frame. Warn if it drifts.
        rb = frame_block_start(nj, 0)
        node0_off = target[:, 0, rb:rb + 3].norm(dim=-1).mean().item()
        if node0_off > 0.15:
            print(f"[fm-data][WARN] target node-0 root offset mean={node0_off:.3f} m (> 0.15); "
                  "the recorded ref_future may not be ego-anchored onto the robot at the snap.")

        # Height token = the recorded height scan at the snap row (per-ray ground
        # clearance, body-relative). Replaces the old root-z scalar. K resolved here.
        hs = torch.from_numpy(np.ascontiguousarray(data["height_scan"])).float()  # (P, K)
        self.height = hs[snap_rows].contiguous()  # (Nw, K)
        self.height_dim = self.height.shape[1]
        cfg.height_dim = self.height_dim

        # Reference pose per window (the snap-step robot base), raw for eval/denorm.
        self.p_ref = p_anchor.contiguous()
        self.q_ref = q_anchor.contiguous()

        # Strided future command tokens.
        if cmd_dim > 0:
            cf = torch.from_numpy(np.ascontiguousarray(cmd_future)).float()[snap_rows]  # (Nw, F, cmd)
            cmd_strided = cf[:, self.cmd_steps, :].contiguous()
        else:
            cmd_strided = None

        # -- normalizers (fit on train; ori6d blocks left raw) --
        state_mask = torch.ones(self.state_dim, dtype=torch.bool)
        for o0, o1 in orientation_slices(nj, self.num_extra_frames):
            state_mask[o0:o1] = False
        if state_normalizer is None:
            state_normalizer = Normalizer.fit(target.reshape(-1, self.state_dim), state_mask)
        self.state_normalizer = state_normalizer

        if height_normalizer is None:
            height_normalizer = Normalizer.fit(
                self.height, torch.ones(self.height_dim, dtype=torch.bool),
                eps=float(cfg.height_norm_eps),
            )
        self.height_normalizer = height_normalizer

        if cmd_dim > 0:
            if cmd_normalizer is None:
                cmd_normalizer = Normalizer.fit(
                    cmd_strided.reshape(-1, cmd_dim), torch.ones(cmd_dim, dtype=torch.bool)
                )
            self.cmd_normalizer = cmd_normalizer
            self.cmd_tokens = cmd_normalizer.normalize(cmd_strided).contiguous()
        else:
            self.cmd_normalizer = None
            self.cmd_tokens = None

        # Pre-normalize once.
        self.state = state_normalizer.normalize(target)           # (Nw, F, D) target
        self.history = state_normalizer.normalize(history)        # (Nw, N, D) conditioning
        self.height_norm_tokens = height_normalizer.normalize(self.height)  # (Nw, 1)

        # -- observation noise (train-time augmentation; conditioning tokens only) --
        self.add_noise = bool(add_noise and cfg.obs_noise)
        self.state_noise = (
            self._state_noise_std(state_normalizer) if self.add_noise else None
        )
        # Height-scan noise (normalized space = raw m / std), per ray. The height token is a
        # real (noisy) sensor at deploy, so it is noised in training like the history tokens.
        self.height_noise = (
            (float(cfg.noise_height_scan) * float(cfg.noise_scale)) / height_normalizer.std
            if self.add_noise else None
        )

    def _state_noise_std(self, state_normalizer: Normalizer) -> torch.Tensor:
        """Per-channel uniform noise half-width (normalized space) for a history token.

        Maps the config's physical per-group amplitudes onto the state layout
        ``[joint_pos | joint_vel | frame blocks x (1+K)]``, scales by ``cfg.noise_scale``,
        then divides by the normalizer std (ori6d channels have std==1, so their noise
        stays in raw 6D units).
        """
        cfg = self.cfg
        nj = self.num_joints
        amp = torch.zeros(self.state_dim, dtype=self.state.dtype)
        amp[0:nj] = cfg.noise_joint_pos
        amp[nj : 2 * nj] = cfg.noise_joint_vel
        for fi in range(1 + self.num_extra_frames):
            b = frame_block_start(nj, fi)
            amp[b : b + 3] = cfg.noise_pos              # pos(3)
            amp[b + 3 : b + 6] = cfg.noise_lin_vel      # lin_vel(3)
            amp[b + 6 : b + 12] = cfg.noise_ori         # ori6d(6)
            amp[b + 12 : b + FRAME_BLOCK_DIM] = cfg.noise_ang_vel  # ang_vel(3)
        return (amp * cfg.noise_scale) / state_normalizer.std

    def __len__(self) -> int:
        return self.state.shape[0]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        history = self.history[i]
        height = self.height_norm_tokens[i]
        if self.add_noise:
            history = history + (torch.rand(self.N, self.state_dim) * 2.0 - 1.0) * self.state_noise
            height = height + (torch.rand(self.height_dim) * 2.0 - 1.0) * self.height_noise
        item = {
            "state": self.state[i],          # (F, D) ego-anchored clip future target
            "history": history,              # (N, D) real-robot phase conditioning
            "height": height,                # (K,) height-scan token (noised in train)
            "p_ref": self.p_ref[i],
            "q_ref": self.q_ref[i],
        }
        if self.cmd_tokens is not None:
            item["cmd_tokens"] = self.cmd_tokens[i]
        return item


def build_train_val(
    data: dict[str, np.ndarray], meta: Any, cfg: FlowMatchingConfig
) -> tuple[FutureStateDataset, FutureStateDataset | None]:
    """Build train/val datasets, splitting by ``episode_id`` to avoid leakage.

    Uses snap-event windows (the robot-state-conditioned FM paradigm). Normalizers are
    fit on the train split and shared with val. ``train_ds.val_episode_ids`` records the
    held-out episode ids so the test bench can label train vs eval. If the val split has
    no windows, ``None`` is returned for it.

    Args:
        data: Recorded arrays (must include ``reanchor_active`` + ``ref_future``).
        meta: Layout metadata.
        cfg: Config.

    Returns:
        ``(train_ds, val_ds_or_None)``.
    """
    reanchor = data.get("reanchor_active")
    if reanchor is None:
        raise ValueError(
            "Recording has no 'reanchor_active' flag; the robot-state-conditioned FM trains "
            "on snap (re-anchor) events. Re-record with ego_anchor_fraction=1.0."
        )
    windows = snap_windows(
        data["episode_id"], data["step_in_episode"], data["done"],
        reanchor, cfg.num_history_states, cfg.stride,
    )
    if len(windows) == 0:
        raise ValueError(
            "No valid snap windows; every snap lacks N prior consecutive steps. "
            "Check ego_anchor_every >= num_history_states and the recording length."
        )

    ep_of_window = np.array([data["episode_id"][w[-1]] for w in windows])  # episode of the snap
    unique_eps = np.unique(ep_of_window)
    rng = np.random.default_rng(cfg.seed)
    n_val = int(round(cfg.val_fraction * len(unique_eps)))
    val_eps = set(rng.choice(unique_eps, size=n_val, replace=False).tolist()) if n_val > 0 else set()

    train_windows = [w for w, e in zip(windows, ep_of_window) if e not in val_eps]
    val_windows = [w for w, e in zip(windows, ep_of_window) if e in val_eps]
    print(f"[fm-data] {len(windows)} snap windows ({len(train_windows)} train / "
          f"{len(val_windows)} val) over {len(unique_eps)} episodes.")

    train_ds = FutureStateDataset(data, meta, cfg, train_windows, add_noise=True)
    train_ds.val_episode_ids = sorted(int(e) for e in val_eps)  # for the bench's eval split
    val_ds = None
    if len(val_windows) > 0:
        val_ds = FutureStateDataset(
            data, meta, cfg, val_windows,
            state_normalizer=train_ds.state_normalizer,
            cmd_normalizer=train_ds.cmd_normalizer,
            height_normalizer=train_ds.height_normalizer,
        )
    return train_ds, val_ds
