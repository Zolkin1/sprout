# Copyright 2026 Zachary Olkin. All rights reserved.

"""Turn recorded rollouts into (history, future) windows for flow matching.

The dataset operates on already-loaded numpy arrays plus a ``meta`` object (duck
-typed; only its layout attributes are read), so it has no dependency on the
recording scripts and is unit-testable with synthetic data. The training script
is responsible for loading the ``.npz`` via ``recorded_data_io.load_recorded_data``
and passing ``(data, meta)`` here.

History token per past step = [relative-state] + [velocity_command] + [height_scan].
Future target per step = relative-state. Root quantities are expressed relative to
the reference frame at the last history step.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import FlowMatchingConfig
from .normalizer import Normalizer
from .transforms import orientation_slices, relative_state, target_state_dim


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


# TODO(partial-history): only full H+F windows are trained on, so the first
# H-1 steps after every reset produce no training data and no predictions at
# inference (the viz shows nothing for the first H steps of each episode).
# Consider training on partial histories so the model can predict from step 0.
# Candidate approaches (roughly in order of effort):
#   1. Repeat-first-state ("edge") padding: pad missing history by repeating
#      the episode's first state (what Diffusion Policy / ACT do). Keeps every
#      padded step a geometrically valid state (relative frames, ori6d) with no
#      architecture change. NOTE: naive zero-padding is a trap here — post-
#      normalization zeros look like the mean state, and all-zeros is not a
#      valid ori6d rotation.
#   2. Add a per-step 0/1 validity flag to the history token so the model can
#      tell padding from "standing still for H steps".
#   3. Randomize the history truncation length during training (history
#      dropout) so the model is robust to any context length, not just the
#      lengths that occur right after resets.
#   4. The principled version: pad with anything + a key_padding_mask in
#      FlowMatchingTransformer.encode so attention never reads padded steps.
# Whatever padding rule is trained with must be mirrored exactly in the
# inference-side history buffers (scripts/flow_matching/record_viz.py).
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
        window_len: Window length ``H + F``.
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


class FutureStateDataset(Dataset):
    """Precomputed (history, future) windows in the relative-frame representation."""

    def __init__(
        self,
        data: dict[str, np.ndarray],
        meta: Any,
        cfg: FlowMatchingConfig,
        windows: list[np.ndarray],
        target_normalizer: Normalizer | None = None,
        history_normalizer: Normalizer | None = None,
    ):
        """Build and (optionally) fit normalizers over the provided windows.

        Args:
            data: Recorded arrays (must include ``obs`` and ``state``).
            meta: Layout metadata (reads ``obs_term_names/dims``, ``state_term_names/dims``).
            cfg: Config; ``state_dim`` and ``history_in_dim`` are written back here.
            windows: Valid window index arrays (from :func:`episode_windows`).
            target_normalizer: If given, reused (val split); else fit on these windows.
            history_normalizer: If given, reused (val split); else fit on these windows.
        """
        self.cfg = cfg
        h, f = cfg.history_len, cfg.future_len
        window_len = h + f
        if len(windows) == 0:
            raise ValueError("No valid windows; check episode lengths vs history_len + future_len.")

        # -- resolve state layout --
        s_names, s_dims = list(meta.state_term_names), list(meta.state_term_dims)
        s_slc = {name: resolve_term_slice(s_names, s_dims, name) for name in s_names}
        nj = s_dims[s_names.index("joint_pos")]
        self.num_joints = nj
        # Extra body frames recorded beyond root + joints (e.g. feet/hands for the
        # CLF). Each contributes a [pos, lin_vel, ori6d, ang_vel] block to the state.
        self.state_frame_names = list(getattr(meta, "state_frame_names", []) or [])
        self.num_extra_frames = len(self.state_frame_names)
        self.state_dim = target_state_dim(nj, self.num_extra_frames)

        # -- resolve the command + perception inputs --
        # The command, when present, comes from a dedicated recorded array
        # (cfg.cmd_array, e.g. "target_vel") and conditions the *whole* window: past
        # steps in the encoder history token, future steps in the decoder. With no
        # cmd_array there is no command conditioning (cmd_dim == 0). The height scan is
        # optional -- flat-ground tasks have no "height_scan" term.
        try:
            self.scan_slice = resolve_term_slice(meta.obs_term_names, meta.obs_term_dims, "height_scan")
        except KeyError:
            self.scan_slice = (0, 0)
        scan_dim = self.scan_slice[1] - self.scan_slice[0]

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
            cmd_dim = 0  # no command conditioning

        # The command spans the whole window, so past (encoder) and future (decoder)
        # command widths are equal.
        self.future_cmd_dim = cmd_dim
        self.history_in_dim = self.state_dim + cmd_dim + scan_dim
        cfg.state_dim = self.state_dim
        cfg.history_in_dim = self.history_in_dim
        cfg.cmd_dim = self.future_cmd_dim

        idx = np.stack(windows).astype(np.int64)  # (Nw, W)
        n_win, _ = idx.shape

        state_t = torch.from_numpy(np.ascontiguousarray(data["state"])).float()
        sw = state_t[idx]  # (Nw, W, state_raw)

        # Command + scan, sliced *before* windowing (the full obs is ~obs_dim/used_dim
        # times wider, e.g. 349 vs 6, and OOMs the host on large overlapping windows).
        # past_cmd conditions the encoder; future_cmd (array source only) the decoder.
        if self.cmd_array is not None:
            cmd_full = torch.from_numpy(np.ascontiguousarray(data[self.cmd_array])).float()[idx]  # (Nw,W,cmd)
            past_cmd = cmd_full[:, :h, :]            # (Nw, H, cmd_dim) -> encoder
            future_cmd_raw = cmd_full[:, h:, :]      # (Nw, F, cmd_dim) -> decoder
        else:
            past_cmd = None
            future_cmd_raw = None
        if scan_dim > 0:
            scan_cols = data["obs"][:, self.scan_slice[0]:self.scan_slice[1]]
            scan = torch.from_numpy(np.ascontiguousarray(scan_cols)).float()[idx][:, :h, :]  # (Nw,H,scan)
        else:
            scan = None

        def _split(name: str) -> torch.Tensor:
            a, b = s_slc[name]
            return sw[..., a:b]

        p = _split("root_pos_w")        # (Nw, W, 3)
        q = _split("root_quat_w")       # (Nw, W, 4)
        v = _split("root_lin_vel_w")    # (Nw, W, 3)
        wv = _split("root_ang_vel_w")   # (Nw, W, 3)
        jp = _split("joint_pos")        # (Nw, W, nj)
        jv = _split("joint_vel")        # (Nw, W, nj)

        # Reference frame = last history step (index h-1), broadcast across the window.
        p_ref = p[:, h - 1 : h, :].expand(n_win, window_len, 3)
        q_ref = q[:, h - 1 : h, :].expand(n_win, window_len, 4)

        m = n_win * window_len
        fpos, fquat, flin, fang = _stack_frame_tensors(_split, self.state_frame_names, m)
        rel = relative_state(
            p.reshape(m, 3), q.reshape(m, 4), v.reshape(m, 3), wv.reshape(m, 3),
            jp.reshape(m, nj), jv.reshape(m, nj),
            p_ref.reshape(m, 3), q_ref.reshape(m, 4),
            frame_pos_w=fpos, frame_quat_w=fquat, frame_lin_vel_w=flin, frame_ang_vel_w=fang,
        ).reshape(n_win, window_len, self.state_dim)

        hist_state = rel[:, :h, :]                       # (Nw, H, state_dim)
        parts = [hist_state]
        if past_cmd is not None:
            parts.append(past_cmd)                       # command (past) -> encoder
        if scan is not None:
            parts.append(scan)
        history = torch.cat(parts, dim=-1).contiguous()  # (Nw, H, history_in_dim)

        # Future target: the actual recorded state, or the tracked reference --
        # both expressed in the same relative frame (anchored at the actual pose
        # at step h-1, so history conditioning is identical for either source).
        if cfg.target_source == "reference":
            future = self._reference_future(data, s_slc, idx, p, q, h, f, nj)  # (Nw, F, state_dim)
        elif cfg.target_source == "actual":
            future = rel[:, h:, :].contiguous()                     # (Nw, F, state_dim)
        else:
            raise ValueError(
                f"Unknown target_source {cfg.target_source!r}; expected 'actual' or 'reference'."
            )

        # Reference pose per window, kept raw for eval/denorm back to world frame.
        self.p_ref = p[:, h - 1, :].contiguous()  # (Nw, 3)
        self.q_ref = q[:, h - 1, :].contiguous()  # (Nw, 4)

        # -- normalizers (fit on train; every frame's ori6d block left raw) --
        target_mask = torch.ones(self.state_dim, dtype=torch.bool)
        for o0, o1 in orientation_slices(nj, self.num_extra_frames):
            target_mask[o0:o1] = False
        # Command + scan columns of the history token are standardized.
        history_mask = torch.cat(
            [target_mask, torch.ones(self.history_in_dim - self.state_dim, dtype=torch.bool)]
        )

        if target_normalizer is None:
            target_normalizer = Normalizer.fit(future.reshape(-1, self.state_dim), target_mask)
        if history_normalizer is None:
            history_normalizer = Normalizer.fit(history.reshape(-1, self.history_in_dim), history_mask)
        self.target_normalizer = target_normalizer
        self.history_normalizer = history_normalizer

        # Pre-normalize once (store normalized tensors; __getitem__ is a pure slice).
        self.history = history_normalizer.normalize(history)
        self.future = target_normalizer.normalize(future)

        # Future command (decoder conditioning), normalized with the *same* stats as
        # the past command in the history token (its command columns), so past and
        # future command live in one normalized space.
        if future_cmd_raw is not None:
            c0, c1 = self.state_dim, self.state_dim + self.future_cmd_dim
            cmd_mean = history_normalizer.mean[c0:c1]
            cmd_std = history_normalizer.std[c0:c1]
            self.future_cmd = ((future_cmd_raw - cmd_mean) / cmd_std).contiguous()
        else:
            self.future_cmd = None

    def _reference_future(
        self,
        data: dict[str, np.ndarray],
        s_slc: dict[str, tuple[int, int]],
        idx: np.ndarray,
        p_actual: torch.Tensor,
        q_actual: torch.Tensor,
        h: int,
        f: int,
        nj: int,
    ) -> torch.Tensor:
        """Build the future target from the recorded reference (``ref_state``).

        The reference future is expressed in the *same* relative frame as the
        actual-state target: anchored at the actual root pose at the last history
        step (``p_actual``/``q_actual`` at index ``h-1``), so the history
        conditioning is identical regardless of target source.

        Args:
            data: Recorded arrays; must include a non-``None`` ``ref_state``.
            s_slc: Per-term ``(start, end)`` slices within the state vector.
            idx: Window row indices, shape ``(Nw, W)``.
            p_actual: Actual root positions over the window, shape ``(Nw, W, 3)``.
            q_actual: Actual root quaternions over the window, shape ``(Nw, W, 4)``.
            h: History length.
            f: Future length.
            nj: Number of joints.

        Returns:
            Reference future in the relative-frame representation, shape ``(Nw, F, state_dim)``.

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
        rw = ref_t[idx][:, h:, :]  # (Nw, F, state_raw)
        n_win = rw.shape[0]

        def _split(name: str) -> torch.Tensor:
            a, b = s_slc[name]
            return rw[..., a:b]

        pf = _split("root_pos_w")       # (Nw, F, 3)
        qf = _split("root_quat_w")      # (Nw, F, 4)
        vf = _split("root_lin_vel_w")   # (Nw, F, 3)
        wf = _split("root_ang_vel_w")   # (Nw, F, 3)
        jpf = _split("joint_pos")       # (Nw, F, nj)
        jvf = _split("joint_vel")       # (Nw, F, nj)

        p_ref = p_actual[:, h - 1 : h, :].expand(n_win, f, 3)
        q_ref = q_actual[:, h - 1 : h, :].expand(n_win, f, 4)

        m = n_win * f
        fpos, fquat, flin, fang = _stack_frame_tensors(_split, self.state_frame_names, m)
        return relative_state(
            pf.reshape(m, 3), qf.reshape(m, 4), vf.reshape(m, 3), wf.reshape(m, 3),
            jpf.reshape(m, nj), jvf.reshape(m, nj),
            p_ref.reshape(m, 3), q_ref.reshape(m, 4),
            frame_pos_w=fpos, frame_quat_w=fquat, frame_lin_vel_w=flin, frame_ang_vel_w=fang,
        ).reshape(n_win, f, self.state_dim).contiguous()

    def __len__(self) -> int:
        return self.history.shape[0]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        item = {
            "history": self.history[i],
            "future": self.future[i],
            "p_ref": self.p_ref[i],
            "q_ref": self.q_ref[i],
        }
        if self.future_cmd is not None:
            item["future_cmd"] = self.future_cmd[i]
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
    window_len = cfg.history_len + cfg.future_len
    windows = episode_windows(
        data["episode_id"], data["step_in_episode"], data["done"], window_len, cfg.stride
    )
    if len(windows) == 0:
        raise ValueError("No valid windows produced from the dataset.")

    ep_of_window = np.array([data["episode_id"][w[0]] for w in windows])
    unique_eps = np.unique(ep_of_window)
    rng = np.random.default_rng(cfg.seed)
    n_val = int(round(cfg.val_fraction * len(unique_eps)))
    val_eps = set(rng.choice(unique_eps, size=n_val, replace=False).tolist()) if n_val > 0 else set()

    train_windows = [w for w, e in zip(windows, ep_of_window) if e not in val_eps]
    val_windows = [w for w, e in zip(windows, ep_of_window) if e in val_eps]

    train_ds = FutureStateDataset(data, meta, cfg, train_windows)
    val_ds = None
    if len(val_windows) > 0:
        val_ds = FutureStateDataset(
            data, meta, cfg, val_windows,
            target_normalizer=train_ds.target_normalizer,
            history_normalizer=train_ds.history_normalizer,
        )
    return train_ds, val_ds
