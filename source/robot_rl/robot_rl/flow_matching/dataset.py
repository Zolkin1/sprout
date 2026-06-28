# Copyright 2026 Zachary Olkin. All rights reserved.

"""Turn recorded rollouts into fixed-length state windows for flow matching.

The dataset operates on already-loaded numpy arrays plus a ``meta`` object (duck
-typed; only its layout attributes are read), so it has no dependency on the
recording scripts and is unit-testable with synthetic data. The training script
is responsible for loading the ``.npz`` via ``recorded_data_io.load_recorded_data``
and passing ``(data, meta)`` here.

Each window is ``future_len`` (F) consecutive recorded steps in one relative-frame
representation, anchored at the most recent history step (index ``num_history_states
- 1``). The first ``num_history_states`` (N) steps are the recent past used as clean
conditioning tokens; the rest are the predicted future. The commanded-velocity
profile over the window is sampled every ``cmd_stride`` steps into conditioning
command tokens. There is one state normalizer (history and target share the
representation) and a separate command normalizer.
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
        rows: Flattened leading dimension (e.g. ``n_win * window_len``).

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


def episode_windows(
    episode_id: np.ndarray,
    step_in_episode: np.ndarray,
    done: np.ndarray,
    window_len: int,
    stride: int,
) -> list[np.ndarray]:
    """Build valid window index arrays from flat per-pair bookkeeping.

    A window is valid iff all its rows share one ``episode_id``, the
    ``step_in_episode`` values are strictly consecutive, and no row before the
    last is a ``done`` (resets are never crossed). Returns row indices into the
    original flat arrays.

    Args:
        episode_id: Per-pair episode id, shape ``(N,)``.
        step_in_episode: Per-pair step index within its episode, shape ``(N,)``.
        done: Per-pair episode-end flag, shape ``(N,)``.
        window_len: Window length ``F``.
        stride: Sliding stride.

    Returns:
        List of int arrays, each of length ``window_len``.
    """
    order = np.lexsort((step_in_episode, episode_id))  # primary: episode_id, secondary: step
    ep_sorted = episode_id[order]
    step_sorted = step_in_episode[order]
    done_sorted = done[order]

    windows: list[np.ndarray] = []
    n = len(order)
    start = 0
    while start < n:
        end = start
        while end + 1 < n and ep_sorted[end + 1] == ep_sorted[start]:
            end += 1
        ep_rows = order[start : end + 1]
        ep_steps = step_sorted[start : end + 1]
        ep_done = done_sorted[start : end + 1]
        length = len(ep_rows)
        for s in range(0, length - window_len + 1, stride):
            sl = slice(s, s + window_len)
            if not np.all(np.diff(ep_steps[sl]) == 1):
                continue
            if np.any(ep_done[sl][:-1] != 0):
                continue
            windows.append(ep_rows[sl].copy())
        start = end + 1
    return windows


def filter_windows_by_recovery(
    windows: list[np.ndarray],
    perturb_active: np.ndarray | None,
    num_history_states: int,
    mode: str,
) -> list[np.ndarray]:
    """Drop windows whose prediction region overlaps a perturbation (recovery mode).

    For perturbed recordings (``perturb_active`` flags steps where action-noise /
    pushes were active), ``"clean_prediction"`` keeps a window only when its
    prediction region (steps ``num_history_states..F``) is entirely clean; the
    history (steps ``0..num_history_states``) may be perturbed. This yields both
    fully-clean windows and bad-state-history -> clean-recovery windows, dropping
    only the "predict-the-perturbation" windows. ``"all"`` keeps everything.

    Args:
        windows: Valid window index arrays (each length ``F``), from
            :func:`episode_windows`.
        perturb_active: Per-pair perturbation flag (``1`` = active), or ``None``
            (recordings without it always fall back to keeping all windows).
        num_history_states: ``N`` -- the number of leading history tokens.
        mode: ``"all"`` or ``"clean_prediction"``.

    Returns:
        The filtered list of windows.

    Raises:
        ValueError: If ``mode`` is not recognized.
    """
    if mode == "all" or perturb_active is None:
        return windows
    if mode != "clean_prediction":
        raise ValueError(f"Unknown recovery_mode {mode!r}; expected 'clean_prediction' or 'all'.")
    if not windows:
        return windows
    # Vectorized over all windows at once (a per-window Python loop is too slow at the
    # ~10M-window scale of large recordings): keep windows whose prediction region
    # (steps num_history_states..F) has no active perturbation flag.
    pa = np.asarray(perturb_active).astype(bool)
    idx = np.stack(windows)  # (Nw, F)
    keep = ~pa[idx[:, num_history_states:]].any(axis=1)  # (Nw,)
    return [windows[i] for i in np.nonzero(keep)[0].tolist()]


class FutureStateDataset(Dataset):
    """Precomputed fixed-length state windows in the relative-frame representation."""

    def __init__(
        self,
        data: dict[str, np.ndarray],
        meta: Any,
        cfg: FlowMatchingConfig,
        windows: list[np.ndarray],
        state_normalizer: Normalizer | None = None,
        cmd_normalizer: Normalizer | None = None,
        add_noise: bool = False,
    ):
        """Build and (optionally) fit normalizers over the provided windows.

        Args:
            data: Recorded arrays (must include ``state``; ``ref_state`` for the
                reference target; the command array when ``cfg.cmd_array`` is set).
            meta: Layout metadata (reads ``state_term_names/dims``, ``state_frame_names``).
            cfg: Config; ``state_dim`` and ``cmd_dim`` are written back here.
            windows: Valid window index arrays (from :func:`episode_windows`).
            state_normalizer: If given, reused (val split); else fit on these windows.
            cmd_normalizer: If given, reused (val split); else fit on these windows.
            add_noise: Add resampled observation noise to the conditioning tokens
                (first N steps) in ``__getitem__`` (train split only). Gated by
                ``cfg.obs_noise``; the prediction target is never perturbed.
        """
        self.cfg = cfg
        f = cfg.future_len
        n = cfg.num_history_states
        self.N = n
        if n < 1 or n >= f:
            raise ValueError(f"num_history_states={n} must satisfy 1 <= N < future_len={f}.")
        if len(windows) == 0:
            raise ValueError("No valid windows; check episode lengths vs future_len.")

        # -- resolve state layout --
        s_names, s_dims = list(meta.state_term_names), list(meta.state_term_dims)
        s_slc = {name: resolve_term_slice(s_names, s_dims, name) for name in s_names}
        nj = s_dims[s_names.index("joint_pos")]
        self.num_joints = nj
        # Extra body frames recorded beyond root + joints (e.g. feet/hands for the
        # CLF). Each contributes a [pos, lin_vel, ori6d, ang_vel] block to the state.
        # History and target share this single (full) representation.
        recorded_frames = list(getattr(meta, "state_frame_names", []) or [])
        self.target_frame_names = list(recorded_frames)
        self.num_extra_frames = len(self.target_frame_names)
        self.state_dim = target_state_dim(nj, self.num_extra_frames)
        cfg.state_dim = self.state_dim

        # -- resolve the command input (strided commanded-velocity profile) --
        self.cmd_array = cfg.cmd_array
        if self.cmd_array is not None:
            cmd_np = data.get(self.cmd_array)
            if cmd_np is None:
                raise ValueError(
                    f"cmd_array={self.cmd_array!r} not in the recording; re-record with "
                    f"scripts/rsl_rl/record_data.py on a clip task that exposes it (e.g. target_vel)."
                )
            cmd_dim = int(cmd_np.shape[1])
        else:
            cmd_dim = 0
        self.cmd_dim = cmd_dim
        cfg.cmd_dim = cmd_dim
        self.cmd_steps = list(range(0, f, cfg.cmd_stride))
        self.num_cmd_tokens = len(self.cmd_steps) if cmd_dim > 0 else 0

        idx = np.stack(windows).astype(np.int64)  # (Nw, F)
        n_win = idx.shape[0]

        state_t = torch.from_numpy(np.ascontiguousarray(data["state"])).float()
        sw = state_t[idx]  # (Nw, F, state_raw)

        def _split(name: str) -> torch.Tensor:
            a, b = s_slc[name]
            return sw[..., a:b]

        p = _split("root_pos_w")              # (Nw, F, 3)
        q = _split("root_quat_w")             # (Nw, F, 4)
        v = _split("root_link_lin_vel_w")     # (Nw, F, 3)
        wv = _split("root_link_ang_vel_w")    # (Nw, F, 3)
        jp = _split("joint_pos")              # (Nw, F, nj)
        jv = _split("joint_vel")              # (Nw, F, nj)

        # Relative-frame anchor = most recent history step (index N-1), shared by the
        # actual-state conditioning and the prediction target.
        anchor = n - 1
        p_ref0 = p[:, anchor : anchor + 1, :]
        q_ref0 = q[:, anchor : anchor + 1, :]
        m = n_win * f
        pr = p_ref0.expand(n_win, f, 3).reshape(m, 3)
        qr = q_ref0.expand(n_win, f, 4).reshape(m, 4)

        fp, fq, fl, fa = _stack_frame_tensors(_split, self.target_frame_names, m)
        actual = relative_state(
            p.reshape(m, 3), q.reshape(m, 4), v.reshape(m, 3), wv.reshape(m, 3),
            jp.reshape(m, nj), jv.reshape(m, nj), pr, qr,
            frame_pos_w=fp, frame_quat_w=fq, frame_lin_vel_w=fl, frame_ang_vel_w=fa,
        ).reshape(n_win, f, self.state_dim).contiguous()  # conditioning source

        if cfg.target_source == "reference":
            target = self._reference_future(data, s_slc, idx, p, q, anchor, f, nj).contiguous()
        elif cfg.target_source == "actual":
            target = actual
        else:
            raise ValueError(
                f"Unknown target_source {cfg.target_source!r}; expected 'actual' or 'reference'."
            )

        # Reference pose per window (anchor step), kept raw for eval/denorm to world.
        self.p_ref = p[:, anchor, :].contiguous()  # (Nw, 3)
        self.q_ref = q[:, anchor, :].contiguous()  # (Nw, 4)

        # Strided command tokens over the window.
        if cmd_dim > 0:
            cmd_full = torch.from_numpy(np.ascontiguousarray(data[self.cmd_array])).float()[idx]  # (Nw,F,cmd)
            cmd_strided = cmd_full[:, self.cmd_steps, :].contiguous()  # (Nw, num_cmd_tokens, cmd)
        else:
            cmd_strided = None

        # -- normalizers (fit on train; every frame's ori6d block left raw) --
        state_mask = torch.ones(self.state_dim, dtype=torch.bool)
        for o0, o1 in orientation_slices(nj, self.num_extra_frames):
            state_mask[o0:o1] = False
        if state_normalizer is None:
            state_normalizer = Normalizer.fit(target.reshape(-1, self.state_dim), state_mask)
        self.state_normalizer = state_normalizer

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

        # Pre-normalize once. ``cond`` is the actual-state conditioning; for the
        # "actual" target it is the same tensor as the (target) ``state``.
        self.state = state_normalizer.normalize(target)
        self.cond = self.state if cfg.target_source == "actual" else state_normalizer.normalize(actual)

        # -- observation noise (train-time augmentation; conditioning tokens only) --
        self.add_noise = bool(add_noise and cfg.obs_noise)
        self.state_noise = (
            self._state_noise_std(state_normalizer) if self.add_noise else None
        )

    def _reference_future(
        self,
        data: dict[str, np.ndarray],
        s_slc: dict[str, tuple[int, int]],
        idx: np.ndarray,
        p_actual: torch.Tensor,
        q_actual: torch.Tensor,
        anchor: int,
        f: int,
        nj: int,
    ) -> torch.Tensor:
        """Build the prediction target from the recorded reference (``ref_state``).

        The reference window is expressed in the *same* relative frame as the
        actual-state conditioning: anchored at the actual root pose at the most
        recent history step (``p_actual``/``q_actual`` at index ``anchor``), so the
        conditioning is identical regardless of target source.

        Args:
            data: Recorded arrays; must include a non-``None`` ``ref_state``.
            s_slc: Per-term ``(start, end)`` slices within the state vector.
            idx: Window row indices, shape ``(Nw, F)``.
            p_actual: Actual root positions over the window, shape ``(Nw, F, 3)``.
            q_actual: Actual root quaternions over the window, shape ``(Nw, F, 4)``.
            anchor: Index of the anchor (most recent history) step.
            f: Window length ``F``.
            nj: Number of joints.

        Returns:
            Reference window in the relative-frame representation, shape ``(Nw, F, state_dim)``.

        Raises:
            ValueError: If the recording has no ``ref_state`` array.
        """
        ref_np = data.get("ref_state")
        if ref_np is None:
            raise ValueError(
                "target_source='reference' requires a 'ref_state' array in the recording; "
                "re-record with scripts/rsl_rl/record_data.py on a task that has a trajectory command."
            )
        ref_t = torch.from_numpy(np.ascontiguousarray(ref_np)).float()
        rw = ref_t[idx]  # (Nw, F, state_raw)
        n_win = rw.shape[0]

        def _split(name: str) -> torch.Tensor:
            a, b = s_slc[name]
            return rw[..., a:b]

        pf = _split("root_pos_w")
        qf = _split("root_quat_w")
        vf = _split("root_link_lin_vel_w")
        wf = _split("root_link_ang_vel_w")
        jpf = _split("joint_pos")
        jvf = _split("joint_vel")

        p_ref = p_actual[:, anchor : anchor + 1, :].expand(n_win, f, 3)
        q_ref = q_actual[:, anchor : anchor + 1, :].expand(n_win, f, 4)

        m = n_win * f
        fpos, fquat, flin, fang = _stack_frame_tensors(_split, self.target_frame_names, m)
        return relative_state(
            pf.reshape(m, 3), qf.reshape(m, 4), vf.reshape(m, 3), wf.reshape(m, 3),
            jpf.reshape(m, nj), jvf.reshape(m, nj),
            p_ref.reshape(m, 3), q_ref.reshape(m, 4),
            frame_pos_w=fpos, frame_quat_w=fquat, frame_lin_vel_w=flin, frame_ang_vel_w=fang,
        ).reshape(n_win, f, self.state_dim)

    def _state_noise_std(self, state_normalizer: Normalizer) -> torch.Tensor:
        """Per-channel uniform noise half-width (normalized space) for a state token.

        Maps the config's physical per-group amplitudes onto the state layout
        ``[joint_pos | joint_vel | frame blocks x (1+K)]`` (each frame block =
        ``[pos(3), lin_vel(3), ori6d(6), ang_vel(3)]``, frame 0 the root), scales by
        ``cfg.noise_scale``, then divides by the normalizer std so the amplitude is in
        the same normalized space as ``self.state`` (ori6d channels have ``std == 1``,
        so their noise stays in raw 6D units).

        Args:
            state_normalizer: The fitted state normalizer.

        Returns:
            Non-negative half-widths, shape ``(state_dim,)``.
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
        state = self.state[i]
        cond = self.cond[i]
        if self.add_noise:
            noise = (torch.rand(self.N, self.state_dim) * 2.0 - 1.0) * self.state_noise
            cond = cond.clone()
            cond[: self.N] = cond[: self.N] + noise
        item = {
            "state": state,
            "cond": cond,
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

    Normalizers are fit on the train split and shared with val. If the val split
    has no valid windows, ``None`` is returned for it.

    Args:
        data: Recorded arrays.
        meta: Layout metadata.
        cfg: Config.

    Returns:
        ``(train_ds, val_ds_or_None)``.
    """
    window_len = cfg.future_len
    windows = episode_windows(
        data["episode_id"], data["step_in_episode"], data["done"], window_len, cfg.stride
    )
    if len(windows) == 0:
        raise ValueError("No valid windows produced from the dataset.")

    # Recovery-aware filtering: keep windows whose prediction region is clean when
    # the recording carries a per-step perturbation flag (record_data.py --perturb).
    recovery_mode = getattr(cfg, "recovery_mode", "all")
    perturb_active = data.get("perturb_active")
    if recovery_mode != "all" and perturb_active is None:
        print(f"[fm-data] recovery_mode={recovery_mode!r} but recording has no 'perturb_active'; "
              "using all windows.")
    elif recovery_mode != "all":
        n_before = len(windows)
        windows = filter_windows_by_recovery(
            windows, perturb_active, cfg.num_history_states, recovery_mode
        )
        print(f"[fm-data] recovery_mode={recovery_mode!r}: kept {len(windows)}/{n_before} windows "
              f"(dropped {n_before - len(windows)} with a perturbed prediction region).")
        if len(windows) == 0:
            raise ValueError(
                f"recovery_mode={recovery_mode!r} dropped all windows; the prediction region of "
                "every window overlaps a perturbation. Lower the perturbation duty cycle or use "
                "--recovery-mode all."
            )

    ep_of_window = np.array([data["episode_id"][w[0]] for w in windows])
    unique_eps = np.unique(ep_of_window)
    rng = np.random.default_rng(cfg.seed)
    n_val = int(round(cfg.val_fraction * len(unique_eps)))
    val_eps = set(rng.choice(unique_eps, size=n_val, replace=False).tolist()) if n_val > 0 else set()

    train_windows = [w for w, e in zip(windows, ep_of_window) if e not in val_eps]
    val_windows = [w for w, e in zip(windows, ep_of_window) if e in val_eps]

    # Observation noise augments the training split only; val keeps clean inputs.
    train_ds = FutureStateDataset(data, meta, cfg, train_windows, add_noise=True)
    val_ds = None
    if len(val_windows) > 0:
        val_ds = FutureStateDataset(
            data, meta, cfg, val_windows,
            state_normalizer=train_ds.state_normalizer,
            cmd_normalizer=train_ds.cmd_normalizer,
        )
    return train_ds, val_ds
