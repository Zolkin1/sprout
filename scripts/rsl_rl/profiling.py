# Copyright 2026 Zachary Olkin. All rights reserved.

"""Lightweight, opt-in profiling harness for the RSL-RL train / play scripts.

Two complementary mechanisms, both off unless a ``--profile`` flag turns them
on (importing this module has no side effects):

- :class:`PhaseTimer` — names spans of work and times them with CUDA events
  (falling back to ``time.perf_counter`` on CPU).  It also emits NVTX ranges so
  the *same* phase names line up on a PyTorch-profiler / Nsight Systems
  timeline.  ``report()`` prints a per-phase table (mean ms/frame and % of the
  frame), splitting a designated set of top-level phases that partition the
  frame from nested "detail" phases shown for information only.
- :class:`TorchProfilerSession` — a thin wrapper over ``torch.profiler`` that
  prints a kernel/op table and dumps a Chrome trace for a bounded run.

Helpers :func:`instrument_env_managers` and :func:`instrument_frame_step` wrap
manager ``compute`` methods (and the trajectory command term) so we get a
per-manager breakdown without editing the Isaac Lab environment.

CUDA-event note: events are asynchronous, so elapsed times are only read after a
single ``torch.cuda.synchronize()`` inside ``report()``.  Until then we just
stash ``(start, end)`` event pairs.
"""

from __future__ import annotations

import functools
import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterable

import torch


class PhaseTimer:
    """Accumulate named-phase timings across many frames and report a breakdown.

    Args:
        device: Torch device (or string).  Determines whether timing uses CUDA
            events (GPU) or ``time.perf_counter`` (CPU).
        warmup: Number of leading frames to skip before recording, so one-time
            lazy initialisation (first-step table builds, kernel autotuning)
            doesn't skew the means.  Applied again after every :meth:`reset`.
        max_record_frames: Safety cap — stop appending records past this many
            recorded frames so a long run can't grow memory without bound.
            ``report()`` still works on whatever was collected.
    """

    def __init__(
        self,
        device: str | torch.device,
        warmup: int = 2,
        max_record_frames: int = 5000,
        sync: bool = False,
    ) -> None:
        self.is_cuda = torch.device(device).type == "cuda"
        # In sync mode, drain the whole device at each phase boundary and time
        # with the wall clock. Slower (serializes the pipeline) but gives true,
        # correctly-nested per-phase costs — including Warp (physics/sensor)
        # work that lives on a different CUDA stream than Torch's events, which
        # the default async CUDA-event timing can mis-attribute to whichever
        # phase first forces a cross-stream sync.
        self.sync = sync and self.is_cuda
        self.warmup = warmup
        self.max_record_frames = max_record_frames
        self._frames = 0
        self._records: dict[str, list] = {}

    def _recording(self) -> bool:
        """Whether the current frame is inside the active recording window."""
        return self.warmup <= self._frames < self.warmup + self.max_record_frames

    @contextmanager
    def phase(self, name: str):
        """Time the wrapped block under ``name`` (no-op outside the window)."""
        if not self._recording():
            yield
            return
        if self.sync:
            # Drain the whole device at both boundaries → true wall-clock cost,
            # correctly nested across the Torch/Warp stream split.
            torch.cuda.nvtx.range_push(name)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                yield
            finally:
                torch.cuda.synchronize()
                self._records.setdefault(name, []).append((time.perf_counter() - t0) * 1e3)
                torch.cuda.nvtx.range_pop()
        elif self.is_cuda:
            torch.cuda.nvtx.range_push(name)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self._records.setdefault(name, []).append((start, end))
                torch.cuda.nvtx.range_pop()
        else:
            t0 = time.perf_counter()
            try:
                yield
            finally:
                self._records.setdefault(name, []).append((time.perf_counter() - t0) * 1e3)

    def tick(self) -> None:
        """Advance the frame counter — call exactly once per frame/step."""
        self._frames += 1

    def reset(self) -> None:
        """Clear all records and the frame counter (start a fresh window)."""
        self._frames = 0
        self._records = {}

    def _total_ms(self, recs: list) -> float:
        """Sum the recorded durations (ms) for one phase.

        Records are CUDA event pairs (async event mode) or plain float ms
        (CPU / sync mode); auto-detect by element type.
        """
        if recs and isinstance(recs[0], tuple):
            return float(sum(s.elapsed_time(e) for s, e in recs))
        return float(sum(recs))

    def report(
        self,
        total_key: str = "frame_total",
        top_level: Iterable[str] = (),
        reset: bool = False,
        title: str = "PROFILE",
        out_path: str | None = None,
    ) -> None:
        """Print a per-phase breakdown table (and optionally append it to a file).

        Args:
            total_key: Phase whose total is the 100% reference for the frame
                (typically a phase wrapping the whole step / loop body).
            top_level: Phases that *partition* the frame (non-overlapping).
                Their sum is subtracted from ``total_key`` to print an
                ``unaccounted`` row (≈ physics + render + Python glue).  Every
                other recorded phase is shown in a "detail" section as % of the
                frame but NOT subtracted (they nest inside a top-level phase).
            reset: Clear records after reporting (for rolling windows in play).
            title: Header label.
            out_path: If set, append the rendered table to this file as well as
                printing it — so a snapshot survives even a hard ``kill`` that
                skips the ``finally`` block.  Best-effort (write errors warn,
                never crash the run).
        """
        if not self._records:
            if reset:
                self.reset()
            return
        if self.is_cuda:
            torch.cuda.synchronize()

        totals = {n: self._total_ms(r) for n, r in self._records.items()}
        counts = {n: len(r) for n, r in self._records.items()}
        frames = counts.get(total_key) or max(counts.values())
        ref = totals.get(total_key) or sum(totals.get(k, 0.0) for k in top_level) or 1.0
        top_set = [k for k in top_level if k in totals]

        def _fmt(name: str) -> str:
            tot = totals[name]
            return (
                f"  {name:<28s} {tot / frames:8.3f} ms/frame "
                f"{100.0 * tot / ref:6.1f}%   (n={counts[name]})"
            )

        lines = [
            "",
            f"==================== {title} ====================",
            f"recorded frames: {frames}   device: {'cuda' if self.is_cuda else 'cpu'}",
            f"frame mean: {ref / frames:.3f} ms/frame  (~{1000.0 * frames / ref:.1f} FPS)",
            "-- top-level (partition of frame) --",
        ]
        for name in sorted(top_set, key=lambda n: totals[n], reverse=True):
            lines.append(_fmt(name))
        if top_set:
            unaccounted = ref - sum(totals[k] for k in top_set)
            lines.append(
                f"  {'unaccounted (physics/render)':<28s} {unaccounted / frames:8.3f} ms/frame "
                f"{100.0 * unaccounted / ref:6.1f}%"
            )
        detail = [n for n in totals if n not in top_set and n != total_key]
        if detail:
            lines.append("-- detail (nested; % of frame) --")
            for name in sorted(detail, key=lambda n: totals[n], reverse=True):
                lines.append(_fmt(name))
        lines.append("=" * (42 + len(title)))
        text = "\n".join(lines)
        print(text, flush=True)

        if out_path is not None:
            try:
                os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                with open(out_path, "a") as fh:
                    fh.write(text + "\n")
            except Exception as err:  # noqa: BLE001 - reporting must not crash the run
                print(f"[profiling] could not write report to {out_path}: {err}", flush=True)

        if reset:
            self.reset()


class TorchProfilerSession:
    """Context manager wrapping ``torch.profiler`` for a bounded deep-dive run.

    On exit, prints the top ops by self-CUDA (or self-CPU) time and writes a
    Chrome trace to ``log_dir``.  PhysX / Warp kernels are under-attributed by
    the PyTorch profiler — use the PhaseTimer ``unaccounted`` row (or Nsight
    Systems with the NVTX ranges this module emits) to account for those.
    """

    def __init__(
        self,
        log_dir: str | None,
        enabled: bool = True,
        record_shapes: bool = True,
        with_stack: bool = False,
        profile_memory: bool = False,
        row_limit: int = 30,
    ) -> None:
        self.log_dir = log_dir
        self.enabled = enabled
        self.record_shapes = record_shapes
        self.with_stack = with_stack
        self.profile_memory = profile_memory
        self.row_limit = row_limit
        self._prof = None

    def __enter__(self) -> "TorchProfilerSession":
        if not self.enabled:
            return self
        from torch.profiler import ProfilerActivity, profile

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        self._prof = profile(
            activities=activities,
            record_shapes=self.record_shapes,
            with_stack=self.with_stack,
            profile_memory=self.profile_memory,
        )
        self._prof.__enter__()
        print("[profiling] torch.profiler active for this run.", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self.enabled or self._prof is None:
            return False
        self._prof.__exit__(exc_type, exc, tb)
        sort_key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
        try:
            print(self._prof.key_averages().table(sort_by=sort_key, row_limit=self.row_limit), flush=True)
        except Exception as err:  # noqa: BLE001 - reporting must not crash the run
            print(f"[profiling] could not print key_averages table: {err}", flush=True)
        if self.log_dir is not None:
            path = os.path.join(self.log_dir, "profile_trace.json")
            try:
                os.makedirs(self.log_dir, exist_ok=True)
                self._prof.export_chrome_trace(path)
                print(f"[profiling] chrome trace written to {path}", flush=True)
            except Exception as err:  # noqa: BLE001
                print(f"[profiling] could not export chrome trace: {err}", flush=True)
        return False


def wrap_method(obj: Any, attr: str, timer: PhaseTimer, label: str) -> Callable[[], None] | None:
    """Wrap ``obj.attr`` so each call is timed under ``label``.

    Returns an undo callable that restores the original, or ``None`` if the
    attribute is missing or already wrapped.
    """
    if obj is None or not hasattr(obj, attr):
        return None
    orig = getattr(obj, attr)
    if getattr(orig, "_phase_wrapped", False):
        return None

    @functools.wraps(orig)
    def wrapped(*args, **kwargs):
        with timer.phase(label):
            return orig(*args, **kwargs)

    wrapped._phase_wrapped = True  # type: ignore[attr-defined]

    setattr(obj, attr, wrapped)

    def undo() -> None:
        # The original was a bound method / attr; restoring it as an instance
        # attribute shadows the wrapper with the original behaviour.
        setattr(obj, attr, orig)

    return undo


def instrument_env_managers(
    env: Any,
    timer: PhaseTimer,
    include_debug_vis: bool = False,
    include_command_detail: bool = True,
) -> list[Callable[[], None]]:
    """Wrap the manager ``compute`` methods so each shows up as a phase.

    Gives a per-manager breakdown (observations / rewards / terminations /
    commands / events) plus, for the trajectory command term, a drill-down of
    its per-step ``_update_command`` work — all without editing Isaac Lab.

    Args:
        env: The (possibly wrapped) RL env; ``env.unwrapped`` must expose the
            manager attributes.
        timer: The :class:`PhaseTimer` to record into.
        include_debug_vis: Also wrap the trajectory term's debug-vis callback
            (useful in play, where it draws markers every frame).
        include_command_detail: Split the ``command`` cost into the trajectory
            command's sub-steps — selection (``traj.pre_update``), measured
            outputs, desired outputs (Bezier ``get_output`` + transform), the
            CLF, and the velocity command. These are *detail* phases (nested
            inside ``command``), shown as % of frame but not subtracted. Some
            (e.g. ``traj.desired``) also fire on the reset path, so their
            per-frame mean folds in occasional reset-time calls.

    Returns:
        A list of undo callables (call each to restore the originals).
    """
    base = getattr(env, "unwrapped", env)
    undos: list[Callable[[], None]] = []

    def _w(obj: Any, attr: str, label: str) -> None:
        undo = wrap_method(obj, attr, timer, label)
        if undo is not None:
            undos.append(undo)

    _w(getattr(base, "observation_manager", None), "compute", "obs")
    _w(getattr(base, "reward_manager", None), "compute", "reward")
    _w(getattr(base, "termination_manager", None), "compute", "termination")
    _w(getattr(base, "command_manager", None), "compute", "command")
    _w(getattr(base, "event_manager", None), "apply", "events")

    cmd_mgr = getattr(base, "command_manager", None)

    def _get_term(name: str) -> Any:
        if cmd_mgr is None:
            return None
        try:
            return cmd_mgr.get_term(name)
        except Exception:  # noqa: BLE001 - task may not have this term
            return None

    term = _get_term("traj_ref")
    if term is not None:
        _w(term, "_pre_update_phase", "traj.pre_update")
        if include_debug_vis:
            _w(term, "_debug_vis_callback", "traj.debug_vis")
        if include_command_detail:
            # Sub-steps of BaseTrajectoryCommand._update_command.
            _w(term, "get_measured_outputs", "traj.measured")
            _w(term, "get_desired_outputs", "traj.desired")
            _w(term, "_transform_desired_outputs", "traj.transform")
            _w(getattr(term, "manager", None), "get_output", "traj.get_output")
            _w(getattr(term, "clf", None), "compute_vdot", "traj.clf")
            # One level inside get_measured_outputs — to see whether its cost is
            # the contact-sensor read, the alignment math, or the (unavoidable)
            # physics body-state fetch (the remainder of traj.measured).
            _w(term, "get_contact_state", "traj.contact_state")
            _w(term, "compute_measured_output", "traj.compute_measured")

    if include_command_detail:
        vel_term = _get_term("base_velocity")
        if vel_term is not None:
            _w(vel_term, "_update_command", "vel_cmd")

    return undos


def instrument_frame_step(
    env: Any,
    timer: PhaseTimer,
    attr: str = "step",
    report_every: int | None = None,
    report_kwargs: dict | None = None,
) -> Callable[[], None] | None:
    """Wrap ``env.step`` to time the whole step as ``frame_total`` and tick once.

    Use in training, where the rollout loop lives inside ``runner.learn`` and we
    can't instrument the loop body directly.  In play, time the phases inline
    instead.

    Args:
        env: The env whose ``step`` (or ``attr``) to wrap.
        timer: The :class:`PhaseTimer` to record into.
        report_every: If set (>0), print a cumulative snapshot every this many
            frames during the run — gives live visibility and means a snapshot
            is already on disk (when ``report_kwargs['out_path']`` is set) even
            if the run is hard-killed before the ``finally`` report.
        report_kwargs: kwargs forwarded to :meth:`PhaseTimer.report` for the
            periodic snapshot (``total_key``, ``top_level``, ``title``,
            ``out_path``); ``reset`` is forced ``False`` so snapshots stay
            cumulative.

    Returns:
        An undo callable, or ``None`` if the attribute is missing/already wrapped.
    """
    if not hasattr(env, attr):
        return None
    orig = getattr(env, attr)
    if getattr(orig, "_phase_wrapped", False):
        return None
    rkw = dict(report_kwargs or {})
    rkw.pop("reset", None)

    @functools.wraps(orig)
    def wrapped(*args, **kwargs):
        with timer.phase("frame_total"):
            out = orig(*args, **kwargs)
        timer.tick()
        if report_every and timer._frames % report_every == 0:
            timer.report(reset=False, **rkw)
        return out

    wrapped._phase_wrapped = True  # type: ignore[attr-defined]
    setattr(env, attr, wrapped)

    def undo() -> None:
        setattr(env, attr, orig)

    return undo
