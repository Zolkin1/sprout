# Copyright 2026 Zachary Olkin. All rights reserved.

"""The fine-tune dataset: the generator's OWN rollout, replayed as (conditioning, chunk, label).

``scripts/rsl_rl/record_rollouts.py`` writes what the generator was conditioned on at every
re-plan and the chunk it emitted, all in PHYSICAL units -- the same space
``online.build_*_tokens`` and ``online.sample_window`` work in.  So a fine-tune re-uses the
pretrained checkpoint's normalizers rather than fitting new ones, which is not an
optimisation but a requirement: a refitted normalizer changes what every weight means and
nothing would transfer.

The target is the emitted chunk itself.  This is advantage-conditioned self-distillation --
the model is taught to reproduce its own actions, told for each one whether it worked -- so
the fine-tune has no access to a "better" action and cannot invent one.  All the label does
is decide which parts of the model's own conditional get amplified when it is pinned
positive at inference.

Two things this deliberately does NOT carry, both from ``FutureStateDataset``:

* **the anchor mask.**  It weights the loss towards axes a CLIP pins, and a generator's own
  output pins nothing -- there is no ground truth here to be off-axis from.  The fine-tune
  therefore runs at ``pinned_axis_weight = 1.0``, where :func:`flow.flow_matching_loss`
  ignores the mask entirely.
* **history noise.**  ``FutureStateDataset`` perturbs the history to model the generator's
  own prediction error feeding back at deploy.  These histories ARE that feedback, recorded
  from a closed loop -- adding the model of a thing on top of the thing itself would
  double-count it.

The extero rows are the bulk of the file (2 x 3900 float16 per window, ~8 GB over a 15k
episode rollout), so they are read from the ``.h5`` per item rather than held in RAM.  Every
other column is small enough to load once.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .advantage import LABEL_NAMES, LABEL_NULL


class _CpuContext:
    """The handful of context fields the token builders read, on the CPU.

    The dataset cannot hold the loaded :class:`FmOnlineContext` itself: its normalizers live
    on the training device, and its model is a CUDA module that a forked DataLoader worker
    must not inherit.  So the normalizers are CLONED to the CPU here -- values from the
    pretrained checkpoint, unchanged, which is the requirement -- and the model is left
    behind.  ``online.build_*_tokens`` read only these attributes and are reused as they are.
    """

    def __init__(self, ctx):
        """Copy the token-building fields off a loaded context.

        Args:
            ctx: A loaded :class:`~robot_rl.flow_matching.online.FmOnlineContext`.
        """
        import copy

        self.N, self.F, self.D = int(ctx.N), int(ctx.F), int(ctx.D)
        # Deep-copied FIRST: `Normalizer.to` moves in place, so calling it on the context's
        # own normalizer would leave the loaded generator with CPU statistics and a CUDA
        # model -- which fails on the next `sample_window`, far from here.
        self.state_norm = copy.deepcopy(ctx.state_norm).to("cpu")
        self.extero_norm = copy.deepcopy(ctx.extero_norm).to("cpu")
        self.cmd_norm = (None if ctx.cmd_norm is None
                         else copy.deepcopy(ctx.cmd_norm).to("cpu"))
        self.cmd_dim = int(ctx.cmd_dim)
        self.cmd_steps = list(ctx.cmd_steps)
        self.extero_source = str(ctx.extero_source)
        self.extero_clip = ctx.extero_clip


class RolloutWindowDataset(Dataset):
    """One recorded re-plan per item, in the layout :func:`flow.flow_matching_loss` takes.

    Args:
        path: The rollout window ``.h5``.
        rows: Row indices into that file to use (the train or val split).
        ctx: A loaded :class:`~robot_rl.flow_matching.online.FmOnlineContext` -- the source
            of the normalizers, the command stride and the extero band.  Everything the
            model's own checkpoint already states.
        label: ``[W]`` int8 advantage label per row of the FILE (not of ``rows``), or
            ``None`` for an unconditioned fine-tune.
        null_prob: Probability of overriding the label with ``null`` per access, so
            classifier-free guidance over the label is available at inference.
        weight: ``[W]`` non-negative per-FILE-row loss weight (AWR), or ``None`` for a
            uniform one.  Carried alongside ``label`` rather than instead of it because the
            two are independent choices: a weighted run uses the plain architecture, so it
            passes ``weight`` and NO ``label``.
        seed: RNG seed for the null dropout.

    Raises:
        ValueError: If the recording's window geometry disagrees with the checkpoint's.  A
            mismatched ``F`` or ``D`` would reshape silently into a plausible-looking window.
    """

    def __init__(self, path: str, rows: np.ndarray, ctx, label: np.ndarray | None = None,
                 null_prob: float = 0.3, seed: int = 0,
                 weight: np.ndarray | None = None):
        import h5py

        self.path = str(path)
        self.rows = np.asarray(rows, dtype=np.int64)
        # A CPU copy, so a worker process neither touches CUDA nor inherits the model.
        self.ctx = _CpuContext(ctx)
        self.null_prob = float(null_prob)
        self._rng = np.random.default_rng(seed)
        self._h5 = None  # opened lazily, per worker process

        with h5py.File(self.path, "r") as f:
            n = int(f.attrs.get("num_windows_written", f.attrs.get("num_windows", 0)))
            f_len, d = f["fm_plan"].shape[1:]
            n_hist, d_h = f["history"].shape[1:]
            self.extero_shape = tuple(int(v) for v in f["extero"].shape[1:])
            self.cmd_dim = int(f["cmd_future"].shape[-1])
            # Small columns, held in RAM: the join keys and the null-history flag.
            self.history_null = f["history_null"][:n].astype(bool)
            self.family_code = f["family_code"][:n]
            self.episode_uid = f["episode_uid"][:n]
            self.replan_index = f["replan_index"][:n]
        if (int(f_len), int(d)) != (ctx.F, ctx.D):
            raise ValueError(
                f"{path} holds ({f_len}, {d}) chunks but the checkpoint's window is "
                f"({ctx.F}, {ctx.D}). One of the two is not the generator that made this "
                "recording.")
        if (int(n_hist), int(d_h)) != (ctx.N, ctx.D):
            raise ValueError(
                f"{path} holds ({n_hist}, {d_h}) history but the checkpoint wants "
                f"({ctx.N}, {ctx.D}).")
        if self.rows.size and int(self.rows.max()) >= n:
            raise ValueError(f"row {int(self.rows.max())} is past the {n} written windows")
        self.label = None if label is None else np.asarray(label, dtype=np.int64)
        if self.label is not None and self.label.shape[0] != n:
            raise ValueError(
                f"the label sidecar has {self.label.shape[0]} entries against {n} windows. "
                "It must be one per FILE row -- joined on (episode_uid, replan_index) by "
                "the caller, not sliced to a split.")
        # float64, NOT float32. The weights come out of an `exp` shifted so the max is 1.0,
        # so at a sharp beta most of them are tiny -- and in float32 they round to EXACTLY
        # zero (10.1% of rows at beta 0.05, 96.9% at 0.02), which silently drops those rows
        # and makes ~1.7% of batches all-zero, i.e. a no-op step. Nothing downstream can see
        # it: `flow_mse` is unweighted, and the reported ESS is computed in float64 upstream.
        self.weight = None if weight is None else np.asarray(weight, dtype=np.float64)
        if self.weight is not None:
            if self.weight.shape[0] != n:
                raise ValueError(
                    f"the weight array has {self.weight.shape[0]} entries against {n} "
                    "windows. Like the label it must be one per FILE row.")
            bad = ~np.isfinite(self.weight[self.rows]) | (self.weight[self.rows] < 0)
            if bad.any():
                raise ValueError(
                    f"{int(bad.sum())} of this split's weights are negative or non-finite. "
                    "A NaN would poison the whole batch mean and a negative weight would "
                    "train the model AWAY from a chunk, which is not what AWR does.")

    def __len__(self) -> int:
        """Number of windows in this split."""
        return int(self.rows.size)

    def _file(self):
        """The open ``h5py.File``, opened on first use in whichever process reads it.

        A handle opened in the parent and inherited by a DataLoader worker is not safe to
        read from concurrently; opening per process is the supported pattern.
        """
        if self._h5 is None:
            import h5py
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        """Return one window, normalized into the model's space.

        Args:
            i: Index into this split.

        Returns:
            ``state`` / ``history`` / ``extero`` / ``cmd_tokens`` / ``history_null`` and,
            when labelled, ``advantage``.
        """
        from . import online  # local: online imports this module's siblings

        r = int(self.rows[i])
        f = self._file()
        plan = torch.from_numpy(np.asarray(f["fm_plan"][r], dtype=np.float32))[None]
        hist = torch.from_numpy(np.asarray(f["history"][r], dtype=np.float32))[None]
        ext = torch.from_numpy(np.asarray(f["extero"][r], dtype=np.float32))[None]
        cmd = torch.from_numpy(np.asarray(f["cmd_future"][r], dtype=np.float32))[None]

        item = {
            # The SERVED window, normalized back into the space the flow denoises in. The
            # served value went through `project_window_rotations`, so this is the chunk the
            # tracker actually followed rather than the raw network output.
            "state": self.ctx.state_norm.normalize(plan)[0],
            "history": online.build_history_tokens(self.ctx, hist)[0],
            "extero": online.build_extero_tokens(self.ctx, ext)[0],
            "history_null": torch.tensor(bool(self.history_null[r])),
        }
        tokens = online.build_cmd_tokens(self.ctx, cmd)
        if tokens is not None:
            item["cmd_tokens"] = tokens[0]
        if self.label is not None:
            lab = int(self.label[r])
            if self.null_prob > 0.0 and self._rng.random() < self.null_prob:
                lab = LABEL_NULL
            item["advantage"] = torch.tensor(lab, dtype=torch.long)
        if self.weight is not None:
            item["weight"] = torch.tensor(float(self.weight[r]), dtype=torch.float64)
        return item

    def label_counts(self) -> dict[str, int]:
        """Count this split's labels by name, for the run banner."""
        if self.label is None:
            return {}
        lab = self.label[self.rows]
        return {name: int((lab == i).sum()) for i, name in enumerate(LABEL_NAMES)}


def join_labels(sidecar: dict[str, np.ndarray], episode_uid: np.ndarray,
                replan_index: np.ndarray) -> np.ndarray:
    """Expand a label sidecar onto every row of the window file, on ``(uid, replan)``.

    The sidecar is shorter than the file: ``fit_value.py`` drops censored episodes, which
    have no outcome to score.  Joining POSITIONALLY would therefore shift every label by the
    number of censored windows before it -- silently, and into a perfectly valid-looking
    array.  Rows with no entry come back ``null``, which is the correct reading of "this
    chunk has no advantage" and not a filler value.

    Args:
        sidecar: Must carry ``label``, ``episode_uid`` and ``replan_index``.
        episode_uid: ``[W]`` the window file's own episode ids.
        replan_index: ``[W]`` the window file's own re-plan indices.

    Returns:
        ``[W]`` int64 labels, ``null`` where the sidecar has no row.

    Raises:
        ValueError: If the sidecar's own key pair has duplicates -- then the join is not a
            function and whichever row won would be arbitrary.
    """
    order, pos, hit = _join_positions(sidecar, episode_uid, replan_index)
    out = np.full(episode_uid.shape[0], LABEL_NULL, dtype=np.int64)
    out[hit] = sidecar["label"].astype(np.int64)[order][pos[hit]]
    return out


def _join_positions(sidecar: dict[str, np.ndarray], episode_uid: np.ndarray,
                    replan_index: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve ``(uid, replan)`` from a sidecar onto a window file's rows.

    Split out so the label join and the VALUE join cannot drift apart: they must agree row
    for row, and two copies of a searchsorted are two chances to key them differently.

    Args:
        sidecar: Must carry ``episode_uid`` and ``replan_index``.
        episode_uid: ``[W]`` the window file's own episode ids.
        replan_index: ``[W]`` the window file's own re-plan indices.

    Returns:
        ``(order, pos, hit)`` -- the sidecar's sort permutation, the position each file row
        resolves to within it, and which rows actually matched.

    Raises:
        ValueError: If the sidecar's own key pair has duplicates -- then the join is not a
            function and whichever row won would be arbitrary.
    """
    key_side = (sidecar["episode_uid"].astype(np.int64) << np.int64(20)) \
        + sidecar["replan_index"].astype(np.int64)
    if np.unique(key_side).size != key_side.size:
        raise ValueError("the label sidecar has duplicate (episode_uid, replan_index) keys")
    key_file = (episode_uid.astype(np.int64) << np.int64(20)) \
        + replan_index.astype(np.int64)
    order = np.argsort(key_side)
    pos = np.searchsorted(key_side[order], key_file)
    pos = np.clip(pos, 0, max(key_side.size - 1, 0))
    hit = key_side[order][pos] == key_file if key_side.size else np.zeros_like(key_file, bool)
    return order, pos, hit


def join_values(sidecar: dict[str, np.ndarray], key: str, episode_uid: np.ndarray,
                replan_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Expand a FLOAT sidecar column onto every row of the window file, on ``(uid, replan)``.

    The float twin of :func:`join_labels`, and it returns the hit mask rather than a filler.
    A label has ``null`` to mean "no entry"; a float has no such value, and ``0.0`` is a
    perfectly ordinary advantage -- so a miss filled with zero would read as a chunk the
    critic scored exactly neutral, which under AWR is a weight of ``exp(0) = 1``, i.e. an
    unlabelled row silently carrying average influence over the update.  Misses come back
    ``NaN`` AND flagged, so neither a downstream mean nor a downstream weight can absorb one
    quietly.

    Args:
        sidecar: Must carry ``key``, ``episode_uid`` and ``replan_index``.
        key: The column to join, e.g. ``"advantage"``.
        episode_uid: ``[W]`` the window file's own episode ids.
        replan_index: ``[W]`` the window file's own re-plan indices.

    Returns:
        ``(values, hit)`` -- ``[W]`` float64 with ``NaN`` where the sidecar has no row, and
        ``[W]`` bool marking the rows it did have.

    Raises:
        KeyError: If ``key`` is not in the sidecar.
        ValueError: If the sidecar's key pair has duplicates.
    """
    if key not in sidecar:
        raise KeyError(f"the sidecar carries no {key!r} column; it has "
                       f"{sorted(sidecar)}")
    order, pos, hit = _join_positions(sidecar, episode_uid, replan_index)
    out = np.full(episode_uid.shape[0], np.nan, dtype=np.float64)
    out[hit] = np.asarray(sidecar[key], dtype=np.float64)[order][pos[hit]]
    return out, hit
