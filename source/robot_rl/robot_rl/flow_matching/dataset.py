# Copyright 2026 Zachary Olkin. All rights reserved.

"""Turn recorded teacher rollouts into autoregressive reference-window FM training items.

The dataset operates on already-loaded numpy arrays plus a ``meta`` object (duck
-typed; only its layout attributes are read), so it has no dependency on the
recording scripts and is unit-testable with synthetic data. The training script
loads the ``.npz`` via ``recorded_data_io.load_recorded_data`` and passes
``(data, meta)`` here.

Paradigm (``target_source="ego_anchored_clip_future"``): each training item is
sampled at a **snap (re-anchor) event** of a recorded teacher rollout. It pairs

* ``history``: the ``N`` reference nodes APPLIED at the ``N`` steps before the snap --
  what the tracker had just been tracking -- each expressed relative to the robot as it
  was at *that* step;
* ``height``: the exteroception tokens at the snap -- the pelvis height scan (one
  token) or the two depth cameras (one token each), per ``cfg.extero_source``;
* ``cmd_tokens``: the strided FUTURE velocity command over the F-step window;
* ``state`` (the target): the ``F``-node clip future **ego-anchored onto the robot** at
  the snap (recorded ``ref_future``), all in the snap robot's frame -- node 0's base
  offset is the live tracking error, ≈ 0 by construction.

Both blocks are the same thing (:mod:`~robot_rl.flow_matching.window` nodes -- the exact
vector ``mdp.reference_window`` feeds the policy), so history and target share one
normalizer, and at deploy the model conditions on nodes it emitted itself. The measured
robot enters only as the frame each block is expressed relative to, and it cancels.

Nothing here needs a re-record beyond the window geometry: ``record_data.py`` already
writes the applied-reference ring (``history_ref`` / per-step ``ref_state``) alongside the
measured one, column-for-column in the same layout.

The exposure-bias caveat is worth stating plainly: this history is the CLIP TEACHER's
applied reference, never the model's own output, so training and deploy see different
history distributions. ``cfg.null_cond_prob`` and the per-channel history noise
(:meth:`FutureStateDataset._state_noise_std`) are what stand between them.
"""

from __future__ import annotations

from typing import Any, Sequence


import numpy as np
import torch
from torch.utils.data import Dataset

from .config import FlowMatchingConfig
from .normalizer import Normalizer
from .transforms import sanitize_extero
from .window import (
    REFERENCE_WINDOW_NOISE,
    expand_window_blocks,
    resolve_reference_blocks,
    window_block_slices,
    window_dim,
    world_to_window,
)


#: Loss weight -> the window blocks the term relates. A term whose blocks are not all
#: selected has nothing to tie together, so :func:`resolve_block_selection` zeroes its
#: weight rather than letting it index columns that are not there.
_LOSS_TERM_BLOCKS: dict[str, tuple[str, ...]] = {
    "vel_consistency_weight": ("base_pose", "base_vel"),
    "accel_consistency_weight": ("base_vel",),
    "joint_vel_consistency_weight": ("joint_pos", "joint_vel"),
    "fk_pose_weight": ("joint_pos", "ee_pos"),
    "body_pose_root_weight": ("joint_pos",),
    "body_pose_world_weight": ("joint_pos", "base_pose"),
}


def resolve_block_selection(cfg: FlowMatchingConfig, meta: Any) -> tuple[str, ...]:
    """Settle which window blocks this model predicts, and disable what they cannot support.

    Order of authority: an explicit ``cfg.reference_blocks``, else the teacher that
    produced the recording (``meta.reference_blocks``), else all five -- which is what a
    recording predating the field necessarily was.

    The selection is written back onto ``cfg`` so the checkpoint records what was actually
    trained, and any consistency weight whose blocks are absent is zeroed here (with a
    printed line). Leaving such a weight live is the failure this whole plumbing exists to
    prevent: the term would index columns belonging to a different block and quietly
    regularize toward nonsense.

    Args:
        cfg: The training config; ``reference_blocks`` and the loss weights are updated.
        meta: Recording layout metadata (only ``reference_blocks`` is read).

    Returns:
        The selection in canonical packing order.
    """
    recorded = list(getattr(meta, "reference_blocks", None) or []) or None
    requested = list(cfg.reference_blocks or []) or None
    selected = resolve_reference_blocks(requested if requested is not None else recorded)
    if requested is not None and recorded is not None and tuple(recorded) != selected:
        # Legal -- the FM can be trained for a teacher other than the recorded one -- but
        # it is the setup where a width mismatch bites at play time, so say it out loud.
        print(f"[fm-data] reference_blocks={list(selected)} OVERRIDES the recording's "
              f"teacher, which observed {recorded}. The generator will only drive a "
              "policy whose reference_window packs the requested blocks.")
    elif recorded is None and requested is None:
        # Only on the FIRST resolution -- the val/eval datasets re-enter this with the
        # selection already written back, and repeating the note would read as a fresh
        # finding about a different recording.
        print(f"[fm-data] the recording states no reference_blocks (it predates the "
              f"field, so its teacher observed all five); using {list(selected)}.")
    cfg.reference_blocks = list(selected)

    for weight_name, needed in _LOSS_TERM_BLOCKS.items():
        missing = [b for b in needed if b not in selected]
        if missing and float(getattr(cfg, weight_name)) > 0.0:
            print(f"[fm-data] {weight_name}={getattr(cfg, weight_name)} -> 0.0: it ties "
                  f"{list(needed)} and this window has no {missing}.")
            setattr(cfg, weight_name, 0.0)
    return selected


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


def _stack_frame_positions(split_fn, frame_names: list[str]) -> torch.Tensor:
    """Gather extra-frame world POSITIONS as ``(..., K, 3)`` for :func:`world_to_window`.

    The recording carries each frame's full ``pos/quat/lin_vel/ang_vel``, but the window's
    ee block is position-only (see :mod:`~robot_rl.flow_matching.window`), so the other
    three are simply not read -- which is why widening the ee block back out would need no
    re-record.

    Args:
        split_fn: ``name -> (..., dim)`` slicer over the windowed rows.
        frame_names: Ordered extra body-frame names; ``[]`` gives a ``(..., 0, 3)`` array.

    Returns:
        ``frame_pos_w`` ``(..., K, 3)``.
    """
    if not frame_names:
        probe = split_fn("root_pos_w")
        return probe.new_zeros(probe.shape[:-1] + (0, 3))
    return torch.stack([split_fn(f"{f}:pos_w") for f in frame_names], dim=-2)


def _rows_to_window(
    split_fn,
    anchor_pos: torch.Tensor,
    anchor_quat: torch.Tensor,
    frame_names: list[str],
    blocks: Sequence[str],
) -> torch.Tensor:
    """Turn recorded world-frame reference rows into window nodes.

    The one place the recorded column layout is mapped onto the window representation, used
    for BOTH blocks the model sees: the ``F``-node future (one anchor for all nodes) and the
    ``N``-node autoregressive history (a per-row anchor). The only difference between the
    two calls is the shape of the anchor, which :func:`world_to_window` broadcasts.

    Args:
        split_fn: ``name -> (..., dim)`` slicer over the recorded rows, in the
            ``meta.state_term_names`` layout.
        anchor_pos: ``(..., 3)`` measured robot base position each node is expressed
            relative to.
        anchor_quat: ``(..., 4)`` measured robot base orientation (xyzw).
        frame_names: Ordered extra body-frame names.
        blocks: The block selection to pack.

    Returns:
        ``(..., D)`` window nodes.
    """
    # The recorded columns for a dropped block are read but never used, which costs one
    # slice of an already-materialised array -- ``world_to_window`` does not compute the
    # block itself. Reading them unconditionally keeps this mapping in one shape.
    return world_to_window(
        anchor_pos, anchor_quat,
        split_fn("root_pos_w"), split_fn("root_quat_w"),
        split_fn("root_link_lin_vel_w"), split_fn("root_link_ang_vel_w"),
        split_fn("joint_pos"), split_fn("joint_vel"),
        _stack_frame_positions(split_fn, frame_names),
        blocks=blocks,
    )


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
                ``ref_future_term_names/dims``, ``ref_future_len``, ``reference_blocks``).
            cfg: Config; ``reference_blocks``, ``state_dim``, ``cmd_dim``, ``history_dim``
                and any unsupportable loss weight are written back here.
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
        # The history rows are only ever read as ANCHOR poses, so a recording whose rows are
        # anchored per step has the same shape as one anchored on the plan and loads clean --
        # while meaning something the generator is never served. At deploy the history IS a
        # slice of the previous plan, in that plan's frame, with no transformation applied.
        anchor = str(getattr(meta, "history_anchor", "") or "")
        if anchor != "plan":
            raise ValueError(
                f"history_anchor={anchor or '<absent>'!r}, but the generator conditions on a "
                "PLAN-frame history (its own previous prediction, untransformed). A per-step "
                "anchored recording differences each reference against its own contemporaneous "
                "robot pose, which cancels the reference's motion: measured, its per-lag means "
                "are FLAT (0.0196/0.0197/0.0199/0.0199 m) and the rows sit 6.9 mm apart, "
                "against 0.1231->0.1657 m and 45 mm plan-anchored. Re-record with "
                "scripts/rsl_rl/record_data.py --num_windows (an absent field means the "
                "recording predates the change, so it is per-step)."
            )
        cfg.check_extero_source()
        if data.get(cfg.extero_source) is None:
            hint = (
                "a height_scanner sensor active (e.g. --task G1-teacher-clf-record-play)"
                if cfg.extero_source == "height_scan"
                else "the depth cameras active (--task G1-teacher-clf-terrain-depth-play); "
                     "note record_data.py stores 'depth' only when the task has them"
            )
            raise ValueError(
                f"extero_source={cfg.extero_source!r} requires a '{cfg.extero_source}' array "
                f"in the recording; re-record with {hint}. "
                f"Present keys: {sorted(k for k in data if data.get(k) is not None)}"
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
        # Which blocks a node packs -- the teacher's selection, resolved from the recording
        # unless the config names one. Everything downstream (D, the slices, the noise
        # table, the losses, the export) derives from this one call.
        self.blocks = resolve_block_selection(cfg, meta)
        self.state_dim = window_dim(nj, self.num_extra_frames, self.blocks)
        self.block_slices = window_block_slices(nj, self.num_extra_frames, self.blocks)
        cfg.state_dim = self.state_dim
        cfg.history_dim = self.state_dim

        rf = data["ref_future"]
        if rf.shape[1] < f:
            raise ValueError(
                f"ref_future has F={rf.shape[1]} nodes but cfg.future_len={f}; the recording "
                "is SHORTER than the horizon being trained. Re-record with "
                f"--ref_future_len >= {f}."
            )
        if rf.shape[1] > f:
            # Truncating is safe and deliberate: node k is the reference k steps after the
            # snap, so the first f of a longer recording ARE the f-node horizon. Recording
            # with headroom therefore lets one file train every F <= it -- which matters
            # because the horizon is set by the tracker's re-plan period
            # (F >= max_window_offset + fm_update_every) and that is a thing you sweep.
            print(f"[fm-data] ref_future has {rf.shape[1]} nodes; truncating to "
                  f"future_len={f}.")
            rf = rf[:, :f]

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

        # Two input formats produce the SAME raw tensors; everything downstream
        # (world_to_window, normalizers, striding, noise) is shared.
        #
        # BOTH blocks the model sees are the APPLIED REFERENCE -- the history is the
        # reference this generator emitted at the previous steps, the target is the
        # reference over the future window -- and the measured robot appears only as the
        # ANCHOR each block is expressed relative to. That is what makes the model
        # autoregressive over its own output at deploy: it pushes back the node it just
        # emitted and needs no state estimate.
        #
        # The two anchors differ, and the difference is the whole design:
        #   * TARGET: one anchor, the robot at the snap, shared by all F nodes -- so the
        #     future is one rigid plan in the robot's current frame, exactly what the
        #     policy's ``reference_window`` observation is.
        #   * HISTORY: a PER-ROW anchor, the robot as it was at that row's own step -- so
        #     each history token is literally the node this model emitted then. Re-anchoring
        #     them onto the robot NOW would instead need the egomotion since, which is the
        #     odometry the deploy path no longer carries.
        #
        # Row indexing: the recorded rings are pushed at one instant per control step
        # (record_data.collect_windows), and ``command_manager.compute()`` runs before
        # ``observation_manager.compute()`` inside ``env.step``, so ``history_ref[H-1]``,
        # ``history_state[H-1]`` and ``ref_future[0]`` are all the SAME step. Node 0 of the
        # target is therefore the CURRENT reference, and the N history rows are the N
        # BEFORE it: indices ``H-1-n .. H-2``. (Guard 3 below pins this.)
        #
        #   per-step (.npz): ``windows`` = length-(N+1) row-index arrays; ``[:-1]`` are the
        #     history rows, ``[-1]`` the snap.
        #   windowed (.h5): ``windows`` = a 1-D array of window indices into the pre-formed
        #     arrays; ``history_state``/``history_ref``/``ref_future`` are already snap-formed.
        windowed = "history_state" in data
        root_pos_slc, root_quat_slc = s_slc["root_pos_w"], s_slc["root_quat_w"]
        _p = lambda t: t[..., root_pos_slc[0]:root_pos_slc[1]]
        _q = lambda t: t[..., root_quat_slc[0]:root_quat_slc[1]]
        rf_t = torch.from_numpy(np.ascontiguousarray(rf)).float()  # (P|W, F, node)
        if windowed:
            sel = np.asarray(windows, dtype=np.int64)  # (Nw,) window selector
            n_win = int(sel.shape[0])
            hs_state = torch.from_numpy(np.ascontiguousarray(data["history_state"])).float()
            H = int(hs_state.shape[1])
            if n + 1 > H:
                raise ValueError(
                    f"num_history_states={n} needs history_len >= {n + 1} (N context rows "
                    f"plus the snap row they precede), but the recording has {H}; "
                    f"re-record with --history_len >= {n + 1}."
                )
            if data.get("history_ref") is None:
                raise ValueError(
                    "The autoregressive history is the APPLIED REFERENCE, which lives in the "
                    "'history_ref' array; this recording has none. Re-record with the current "
                    "record_data.py (it writes history_ref unconditionally)."
                )
            hr = torch.from_numpy(np.ascontiguousarray(data["history_ref"])).float()
            # The recorder writes the PLAN ANCHOR into rows 0..H-2, so these n rows all carry
            # the same pose -- the robot at the last re-plan. Row H-1 stays measured.
            hw_meas = hs_state[sel][:, H - 1 - n:H - 1, :]   # (Nw, n, node) anchors, lags n..1
            hw_ref = hr[sel][:, H - 1 - n:H - 1, :]          # (Nw, n, node) applied reference
            snap_meas = hs_state[sel][:, H - 1, :]           # (Nw, node) robot at the snap
            snap_ref = hr[sel][:, H - 1, :]                  # (Nw, node) reference at the snap
            tw = rf_t[sel]                                   # (Nw, F, node)
        else:
            idx_all = np.stack(windows).astype(np.int64)     # (Nw, N+1); last = snap
            idx_hist, snap_rows = idx_all[:, :-1], idx_all[:, -1]
            n_win = idx_all.shape[0]
            state_t = torch.from_numpy(np.ascontiguousarray(data["state"])).float()
            if data.get("ref_state") is None:
                raise ValueError(
                    "The autoregressive history is the APPLIED REFERENCE, which lives in the "
                    "per-step 'ref_state' array; this recording has none. Re-record with the "
                    "current record_data.py."
                )
            ref_t = torch.from_numpy(np.ascontiguousarray(data["ref_state"])).float()
            hw_meas = state_t[idx_hist]                      # (Nw, N, node)
            hw_ref = ref_t[idx_hist]                         # (Nw, N, node)
            snap_meas = state_t[snap_rows]                   # (Nw, node)
            snap_ref = ref_t[snap_rows]
            tw = rf_t[snap_rows]                             # (Nw, F, node)

        p_anchor, q_anchor = _p(snap_meas), _q(snap_meas)    # (Nw, 3) / (Nw, 4)

        # -- history: the emitted reference at lags n..1, all in the PLAN's frame (windowed;
        # the legacy per-step path still anchors each row on its own step) --
        history = _rows_to_window(
            lambda name: hw_ref[..., s_slc[name][0]:s_slc[name][1]],
            _p(hw_meas), _q(hw_meas), self.target_frame_names, self.blocks,
        ).contiguous()                                        # (Nw, n, D)

        # -- target: the F-node ego-anchored clip future, all in the snap robot's frame --
        target = _rows_to_window(
            lambda name: tw[..., s_slc[name][0]:s_slc[name][1]],
            p_anchor[:, None, :], q_anchor[:, None, :], self.target_frame_names, self.blocks,
        ).contiguous()                                        # (Nw, F, D)

        self._check_window_invariants(target, history, snap_ref, tw[:, 0], s_slc, f)

        # Exteroception tokens at the snap row. The recording carries either
        # ``height_scan`` (P|W, K) -- one token -- or ``depth`` (P|W, T, K) with one token
        # per camera, so the token count and per-token width both come from the array's
        # shape rather than being configured.
        hs = torch.from_numpy(np.ascontiguousarray(data[cfg.extero_source])).float()
        if hs.ndim == 2:
            hs = hs.unsqueeze(1)                                   # (P|W, 1, K)
        n_tokens, token_dim = int(hs.shape[1]), int(hs.shape[2])
        # Sanitize before the normalizer fit so a non-finite entry cannot poison the
        # per-dimension mean and yield a NaN loss (matches the deploy token builder).
        hs = sanitize_extero(hs, cfg.extero_source)
        # Flattened to (rows, T*K): the normalizer, the noise and __getitem__ all work on
        # the flat width, and the model reshapes back to (B, T, K) internally.
        hs = hs.reshape(hs.shape[0], n_tokens * token_dim)
        self.height = (hs[sel] if windowed else hs[snap_rows]).contiguous()  # (Nw, T*K)
        self.height_dim = token_dim
        self.extero_tokens = n_tokens
        cfg.height_dim = token_dim
        cfg.extero_tokens = n_tokens
        # Contract of the sensor behind those tokens, carried from the recording so the
        # trained checkpoint (and the ONNX exported from it) states which cameras it
        # conditions on. Only depth has one; the height scan's geometry travels in the
        # tracker's ``height_scan_spec``.
        self.extero_spec = self._resolve_extero_spec(cfg, meta, n_tokens, token_dim)

        # Reference pose per window (the snap-step robot base), raw for eval/denorm.
        self.p_ref = p_anchor.contiguous()
        self.q_ref = q_anchor.contiguous()

        # Strided future command tokens.
        if cmd_dim > 0:
            cf = torch.from_numpy(np.ascontiguousarray(cmd_future)).float()  # (P|W, F, cmd)
            cf = cf[sel] if windowed else cf[snap_rows]                       # (Nw, F, cmd)
            cmd_strided = cf[:, self.cmd_steps, :].contiguous()
        else:
            cmd_strided = None

        # -- normalizers (fit on train) --
        # base_pose's orientation half is EXEMPT from standardization -- the reason the
        # normalizer carries a mask at all (see ``normalizer`` module docstring). Its six
        # channels are two orthonormal columns: already O(1) and unit-scaled, and a
        # per-channel affine would shear them off the manifold, so the model would be
        # regressing towards targets that are no longer rotations and the Gram-Schmidt on
        # the way out would silently absorb the difference. Everything else is standardized.
        #
        # Note the fit is over all (Nw * F) nodes, so base_pose's position std is set by the
        # FAR nodes -- node 0's near-zero residual ends up squashed, and node 0 is the one
        # the tracker's CLF reads. Unchanged by the representation change, but worth
        # revisiting with per-node loss weighting.
        state_mask = torch.ones(self.state_dim, dtype=torch.bool)
        if "base_pose" in self.block_slices:
            bp = self.block_slices["base_pose"].start
            state_mask[bp + 3:bp + 9] = False
        if state_normalizer is None:
            state_normalizer = Normalizer.fit(target.reshape(-1, self.state_dim), state_mask)
        self.state_normalizer = state_normalizer

        if height_normalizer is None:
            height_normalizer = Normalizer.fit(
                self.height, torch.ones(self.height.shape[1], dtype=torch.bool),
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
        self.height_norm_tokens = height_normalizer.normalize(self.height)  # (Nw, T*K)

        # -- observation noise (train-time augmentation; conditioning tokens only) --
        self.add_noise = bool(add_noise and cfg.obs_noise)
        self.state_noise = (
            self._state_noise_std(state_normalizer) if self.add_noise else None
        )
        # Height-scan noise (normalized space = raw m / std), per ray. The height token is a
        # real (noisy) sensor at deploy, so it is noised in training like the history tokens.
        self.height_noise = (
            (float(cfg.noise_extero) * float(cfg.noise_scale)) / height_normalizer.std
            if self.add_noise else None
        )

    def _check_window_invariants(
        self,
        target: torch.Tensor,
        history: torch.Tensor,
        snap_ref: torch.Tensor,
        future_node0: torch.Tensor,
        s_slc: dict[str, tuple[int, int]],
        f: int,
    ) -> None:
        """Fail on the recording defects that would otherwise train a plausible wrong model.

        Every one of these is silent: the arrays are the right shape and the loss converges
        either way, and the damage only shows up as a generator that steers oddly.

        Guards 1, 2 and 4 read a block out of the packed window, so they are skipped when
        the selection has dropped it. Guard 3 works on the RAW recorded rows and always
        runs -- it is the one that pins the recorder's off-by-one, and it is worth having
        whatever the window is packing.

        Args:
            target: ``(Nw, F, D)`` future window nodes.
            history: ``(Nw, N, D)`` autoregressive history nodes.
            snap_ref: ``(Nw, node)`` recorded APPLIED reference at the snap step, raw world
                layout -- the cross-check for the row alignment.
            future_node0: ``(Nw, node)`` recorded ``ref_future`` node 0, same raw layout.
            s_slc: Recorded term name -> ``(start, end)`` column slice.
            f: ``cfg.future_len``.

        Raises:
            ValueError: On a violated invariant.
        """
        if "base_pose" in self.block_slices:
            bp = self.block_slices["base_pose"]
            pos0 = target[:, 0, bp.start:bp.start + 3]
            ori6d = target[..., bp.start + 3:bp.start + 9]

            # 1. Node-0 ego anchor. The clip anchored at the snap sits on the robot, so
            #    node 0's base_pose position is the (small) live tracking error. The
            #    paradigm assumes it: the policy reads node 0 as "where the reference is
            #    right now".
            node0_off = pos0.norm(dim=-1).mean().item()
            if node0_off > 0.15:
                raise ValueError(
                    f"target node-0 base position offset mean={node0_off:.3f} m (> 0.15): the "
                    "recorded ref_future is not ego-anchored onto the robot at the snap. "
                    "Re-record with ego_anchor_fraction=1.0 on the teacher command."
                )

            # 2. The orientation half really is a rotation. The TARGET is built from
            #    recorded unit quaternions, so its two 6D columns must be orthonormal to
            #    float precision -- the model's output need not be, but the thing it
            #    regresses towards does. This replaces the old pi-branch-cut guard, which
            #    the 6D representation makes unfireable: what it protected against was a
            #    log map wrapping, and there is no longer a log map. A packing bug (wrong
            #    slice, transposed columns, a stale 6-wide recording read as 9) lands here
            #    instead of surviving as a plausible-looking target.
            c0, c1 = ori6d[..., 0:3], ori6d[..., 3:6]
            worst = max(
                (c0.norm(dim=-1) - 1.0).abs().max().item(),
                (c1.norm(dim=-1) - 1.0).abs().max().item(),
                (c0 * c1).sum(dim=-1).abs().max().item(),
            )
            if worst > 1e-3:
                raise ValueError(
                    f"target base_pose orientation is not a rotation: worst deviation from "
                    f"two orthonormal columns is {worst:.2e} (> 1e-3). The 6D block is "
                    "built from recorded unit quaternions, so this is a packing bug, not "
                    "data -- check the base_pose slice and the recording's width."
                )

        # 3. History/target row alignment -- the highest-value guard here. ``history_ref``
        #    and ``ref_future`` come from different accessors (the per-step world cache vs
        #    the look-ahead gather), and the recorder pushes its rings one step per control
        #    step, so an off-by-one is entirely possible and entirely invisible. They must
        #    describe the SAME step: history row H-1 (which the history block excludes) is
        #    the reference at the snap, and so is ref_future node 0. Comparing them also
        #    catches history_ref carrying the GLOBALLY-placed clip base rather than the
        #    anchored one, and a joint-order scramble between the two accessors.
        a, b = s_slc["root_pos_w"]
        jp_a, jp_b = s_slc["joint_pos"]
        d_pos = (snap_ref[:, a:b] - future_node0[:, a:b]).norm(dim=-1).max().item()
        d_joint = (snap_ref[:, jp_a:jp_b] - future_node0[:, jp_a:jp_b]).abs().max().item()
        if d_pos > 1e-3 or d_joint > 1e-3:
            raise ValueError(
                "history_ref[H-1] != ref_future[0]: the recorded applied-reference ring and "
                f"the look-ahead future disagree at the snap step (base {d_pos:.4f} m, joints "
                f"{d_joint:.4f} rad). They must be the same step -- the history block is the N "
                "rows BEFORE node 0. Either the recorder's ring is off by one against the "
                "look-ahead, or history_ref is carrying the globally-placed clip base instead "
                "of the anchored one."
            )

        # 4. The history's last row is ONE control step before node 0, so the gap between
        #    them must look like one step. Calibrated against the gap between consecutive
        #    TARGET nodes -- which is one step by construction -- rather than an absolute
        #    threshold: the raw number is dominated by how fast the clip's joints move (a
        #    18 rad/s joint covers 0.36 rad in a step all by itself), so any fixed tolerance
        #    is either loose on slow clips or wrong on fast ones. An off-by-k history shows
        #    up as a gap ~k times the one-step gap, which this catches at any joint speed.
        if "joint_pos" not in self.block_slices:
            return
        jp = self.block_slices["joint_pos"]
        one_step = (target[:, 1, jp] - target[:, 0, jp]).abs().mean().item()
        hist_gap = (history[:, -1, jp] - target[:, 0, jp]).abs().mean().item()
        if hist_gap > 3.0 * one_step + 1e-4:
            raise ValueError(
                f"history[-1] is {hist_gap / max(one_step, 1e-9):.1f} control steps away from "
                f"target[0] in joint_pos (gap {hist_gap:.4f} rad vs {one_step:.4f} rad per "
                "step): the autoregressive history rows are misaligned against the future "
                "window. The history block is the N rows BEFORE node 0."
            )

    @staticmethod
    def _resolve_extero_spec(
        cfg: FlowMatchingConfig, meta: Any, n_tokens: int, token_dim: int
    ) -> list[dict] | None:
        """The sensor contract behind the exteroception tokens, validated against them.

        Only the depth source has one: it is the deployable signal, and reproducing it
        off-sim needs the camera geometry, whereas the height scan's grid already travels
        in the tracker's ``height_scan_spec``.

        Args:
            cfg: The training config (its ``extero_source`` selects the sensor).
            meta: Recording layout metadata.
            n_tokens: Token count found in the recorded array.
            token_dim: Per-token width found in the recorded array.

        Returns:
            The per-camera contract, or ``None`` for the height scan.

        Raises:
            ValueError: If a depth recording carries no camera contract, or if the
                contract disagrees with the shape of the recorded array.
        """
        if cfg.extero_source != "depth":
            return None
        spec = getattr(meta, "depth_camera_spec", None)
        if not spec:
            raise ValueError(
                "extero_source='depth' requires the recording's 'depth_camera_spec' "
                "(the camera geometry the FM must state at deploy); this recording has "
                "none. Re-record with scripts/rsl_rl/record_data.py on a depth task."
            )
        if len(spec) != n_tokens:
            raise ValueError(
                f"depth_camera_spec lists {len(spec)} cameras but the recorded depth has "
                f"{n_tokens} token(s)."
            )
        for cam in spec:
            if int(cam["width"]) * int(cam["height"]) != token_dim:
                raise ValueError(
                    f"camera '{cam.get('sensor')}' is {cam['width']}x{cam['height']} "
                    f"= {int(cam['width']) * int(cam['height'])} pixels but the recorded "
                    f"depth token is {token_dim} wide."
                )
        return [dict(cam) for cam in spec]

    def _state_noise_std(self, state_normalizer: Normalizer) -> torch.Tensor:
        """Per-channel uniform noise half-width (normalized space) for a history token.

        Expands :data:`~robot_rl.flow_matching.window.REFERENCE_WINDOW_NOISE` over the
        window layout, scales by ``cfg.noise_scale``, then divides by the normalizer std.

        It is the SAME table the policy's ``reference_window`` observation is corrupted
        with, deliberately: both model the generator's prediction error, and at deploy this
        model's history is its own past output, so that error is exactly what corrupts it.
        This noise is also the main thing standing between the training history (the clip
        teacher's applied reference) and the deploy history (the model's own) -- the
        exposure-bias gap -- so it matters more here than it did when the history was
        measured state.
        """
        amp = torch.tensor(
            expand_window_blocks(self.num_joints, self.num_extra_frames,
                                 REFERENCE_WINDOW_NOISE, self.blocks),
            dtype=self.state.dtype,
        )
        return (amp * self.cfg.noise_scale) / state_normalizer.std

    def __len__(self) -> int:
        return self.state.shape[0]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        history = self.history[i]
        height = self.height_norm_tokens[i]
        if self.add_noise:
            history = history + (torch.rand(self.N, self.state_dim) * 2.0 - 1.0) * self.state_noise
            height = height + (torch.rand(self.height.shape[1]) * 2.0 - 1.0) * self.height_noise
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
    # Pre-formed windowed (.h5) recordings skip the snap_windows pass entirely.
    if "history_state" in data:
        return _build_train_val_windowed(data, meta, cfg)
    reanchor = data.get("reanchor_active")
    if reanchor is None:
        raise ValueError(
            "Recording has no 'reanchor_active' flag; the robot-state-conditioned FM trains "
            "on snap (re-anchor) events. Re-record with ego_anchor_fraction=1.0."
        )
    # N+1 rows: the N autoregressive context rows plus the snap row they precede (the
    # snap is node 0 of the future, so it is NOT part of the history block).
    windows = snap_windows(
        data["episode_id"], data["step_in_episode"], data["done"],
        reanchor, cfg.num_history_states + 1, cfg.stride,
    )
    if len(windows) == 0:
        raise ValueError(
            "No valid snap windows; every snap lacks N+1 prior consecutive steps. "
            "Check ego_anchor_every > num_history_states and the recording length."
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


def _distractor_keep_mask(
    data: dict[str, np.ndarray], cfg: FlowMatchingConfig, num_windows: int
) -> np.ndarray | None:
    """Boolean keep-mask dropping DISTRACTOR-family windows, or ``None`` to keep everything.

    Distractor clips are flat walking routed around obstacle boxes. They are ego-anchored like
    flat clips (so they DO produce honest snap windows), but whether they help the generator is
    an open question -- hence the opt-out. ``record_data.py`` tags every window with
    ``is_distractor``, so this is a train-time switch needing no re-record.

    Args:
        data: The loaded recording arrays.
        cfg: Config; ``skip_distractors`` enables the filter.
        num_windows: ``W``, for validating the tag's length.

    Returns:
        ``(W,)`` bool array (True = keep), or ``None`` when the filter is off / a no-op.

    Raises:
        ValueError: If the filter is requested but the recording carries no tag, or the tag
            would drop every window.
    """
    if not getattr(cfg, "skip_distractors", False):
        return None
    tag = data.get("is_distractor")
    if tag is None:
        raise ValueError(
            "--skip_distractors requested but this recording has no 'is_distractor' tag (it "
            "predates the field). Either re-record with the current record_data.py, or drop the "
            "flag -- silently training on the distractors would misreport what was excluded."
        )
    tag = np.asarray(tag).reshape(-1).astype(bool)
    if tag.shape[0] != num_windows:
        raise ValueError(
            f"'is_distractor' has {tag.shape[0]} entries but there are {num_windows} windows."
        )
    n_drop = int(tag.sum())
    if n_drop == 0:
        print("[fm-data] --skip_distractors: no distractor windows in this recording (no-op).")
        return None
    if n_drop == num_windows:
        raise ValueError(
            f"--skip_distractors would drop ALL {num_windows} windows (the whole recording is "
            f"distractors); nothing left to train on."
        )
    print(f"[fm-data] --skip_distractors: dropping {n_drop}/{num_windows} distractor windows "
          f"({100.0 * n_drop / num_windows:.1f}%).")
    return ~tag


def _family_keep_mask(
    data: dict[str, np.ndarray], meta: Any, cfg: FlowMatchingConfig, num_windows: int
) -> np.ndarray | None:
    """Boolean keep-mask dropping the ``cfg.skip_families`` clip families, or ``None``.

    The general form of :func:`_distractor_keep_mask`. Reads the per-window ``clip_family``
    code and ``meta.family_names``, both written at record time, so any family is a
    train-time choice.

    Args:
        data: The loaded recording arrays.
        meta: Layout metadata (``family_names`` is read).
        cfg: Config; ``skip_families`` names what to drop.
        num_windows: ``W``, for validating the tag's length.

    Returns:
        ``(W,)`` bool array (True = keep), or ``None`` when nothing is requested.

    Raises:
        ValueError: If the recording predates the tag, a named family is absent from it, or
            the filter would drop every window. Each of those would otherwise look like a
            successful no-op.
    """
    wanted = [f for f in (cfg.skip_families or []) if f]
    if not wanted:
        return None
    codes = data.get("clip_family")
    names = list(getattr(meta, "family_names", []) or [])
    if codes is None or not names:
        raise ValueError(
            f"--skip_families {wanted} needs the per-window 'clip_family' tag and the "
            "recording's family_names; this recording predates them. Re-record with the "
            "current record_data.py (it writes both unconditionally), or drop the flag -- "
            "silently training on those families would misreport what was excluded."
        )
    unknown = [f for f in wanted if f not in names]
    if unknown:
        raise ValueError(
            f"--skip_families named {unknown}, which this recording does not contain. Its "
            f"families are {names}. A typo here would silently drop nothing."
        )
    codes = np.asarray(codes).reshape(-1).astype(np.int64)
    if codes.shape[0] != num_windows:
        raise ValueError(
            f"'clip_family' has {codes.shape[0]} entries but there are {num_windows} windows."
        )
    drop = np.isin(codes, [names.index(f) for f in wanted])
    n_drop = int(drop.sum())
    if n_drop == num_windows:
        raise ValueError(
            f"--skip_families {wanted} would drop ALL {num_windows} windows; nothing left."
        )
    if n_drop == 0:
        print(f"[fm-data] --skip_families {wanted}: no such windows in this recording (no-op).")
        return None
    print(f"[fm-data] --skip_families {wanted}: dropping {n_drop}/{num_windows} windows "
          f"({100.0 * n_drop / num_windows:.1f}%).")
    return ~drop


def _build_train_val_windowed(
    data: dict[str, np.ndarray], meta: Any, cfg: FlowMatchingConfig
) -> tuple[FutureStateDataset, FutureStateDataset | None]:
    """Train/val split for the pre-formed windowed (``.h5``) format.

    The snap windows are already formed at record time (:class:`RecordedWindowWriter`),
    so there is no :func:`snap_windows` pass: split the ``W`` windows by their per-window
    snap ``episode_id`` (leakage-free, same RNG/``val_fraction`` as the per-step path) and
    pass a 1-D window-index selector to :class:`FutureStateDataset`. ``cfg.stride`` applies
    as a global train-time subsample of each split (a no-op at the default ``stride=1``).
    """
    W = int(data["history_state"].shape[0])
    ep = np.asarray(data["episode_id"]).reshape(-1)  # (W,) snap-row episode id per window
    keep = _distractor_keep_mask(data, cfg, W)
    fam_keep = _family_keep_mask(data, meta, cfg, W)
    keep = fam_keep if keep is None else (keep if fam_keep is None else (keep & fam_keep))
    if keep is not None:
        ep = ep[keep]
        data = {**data, **{k: v[keep] for k, v in data.items()
                           if isinstance(v, np.ndarray) and v.shape[:1] == (W,)}}
        W = int(ep.shape[0])
    unique_eps = np.unique(ep)
    rng = np.random.default_rng(cfg.seed)
    n_val = int(round(cfg.val_fraction * len(unique_eps)))
    val_eps = set(rng.choice(unique_eps, size=n_val, replace=False).tolist()) if n_val > 0 else set()

    all_idx = np.arange(W, dtype=np.int64)
    in_val = np.isin(ep, list(val_eps)) if val_eps else np.zeros(W, dtype=bool)
    stride = max(int(cfg.stride), 1)
    train_idx = all_idx[~in_val][::stride]
    val_idx = all_idx[in_val][::stride]
    print(f"[fm-data] {W} snap windows ({train_idx.size} train / {val_idx.size} val) "
          f"over {len(unique_eps)} episodes [windowed .h5].")

    train_ds = FutureStateDataset(data, meta, cfg, train_idx, add_noise=True)
    train_ds.val_episode_ids = sorted(int(e) for e in val_eps)  # for the bench's eval split
    val_ds = None
    if val_idx.size > 0:
        val_ds = FutureStateDataset(
            data, meta, cfg, val_idx,
            state_normalizer=train_ds.state_normalizer,
            cmd_normalizer=train_ds.cmd_normalizer,
            height_normalizer=train_ds.height_normalizer,
        )
    return train_ds, val_ds
