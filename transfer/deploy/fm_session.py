# Copyright 2026 Zachary Olkin. All rights reserved.

"""onnxruntime flow-matching generator session (numpy glue + ORT net).

Numpy + onnxruntime counterpart of ``robot_rl.flow_matching.online``. The ONNX
(``fm.onnx``, exported by ``scripts/flow_matching/export_fm_onnx.py``) folds in the
normalizers and the unrolled Euler sampler, so this assembles the raw conditioning
tokens, runs one ``session.run`` for the whole sampler, and returns the predicted
window. There is no inpainting / known mask -- conditioning is separate tokens. No torch.

There is also no geometry left here, which is the point. The generator predicts
reference-window nodes directly (see ``robot_rl.flow_matching.window``), so:

* the ``N`` history tokens are the nodes this runtime last served -- its own output --
  and go in unchanged;
* the ``F`` predicted nodes go straight into the tracker's ``reference_window``
  observation, with no lift.

The previous version built the history from measured world-frame root poses and lifted
the prediction back out through them, which meant the generator could not run without a
base-pose estimate. Nothing on this path needs one now.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import onnxruntime as ort

from .depth_obs import sanitize_extero_np
from .scan_obs import parse_extero_spec
from .tracker_session import make_session
from .window_np import REFERENCE_WINDOW_BLOCK_NAMES, window_blocks, window_dim


@dataclass
class FmContext:
    """Loaded ``fm.onnx`` session + metadata (mirrors ``flow_matching.online`` FmOnlineContext)."""

    sess: ort.InferenceSession
    N: int
    F: int
    D: int
    nj: int
    target_num_extra_frames: int
    num_steps: int
    cmd_dim: int
    cmd_steps: list[int]
    num_cmd_tokens: int
    joint_names: list[str]
    target_frame_names: list[str]
    extero_sensor_dim: int
    reference_blocks: tuple[str, ...] = REFERENCE_WINDOW_BLOCK_NAMES
    """Which reference-window blocks this graph PREDICTS, in packing order.

    Must equal the tracker's ``reference_window_spec['blocks']`` -- the prediction is fed
    to it verbatim. ``GenTrackerPolicy.load`` is where the two meet."""
    extero_channels: int = 1
    """Planes per pixel in one sensor row (1 = depth alone, 4 = + xyz, 5 = + validity).

    ``extero_sensor_dim`` counts them, being the row width the graph receives, so anything
    comparing that width against an image shape divides by this first.
    """

    extero_source: str = "height_scan"
    """Which sensor feeds the exteroception tokens ("height_scan" or "depth")."""
    extero_sensor_count: int = 1
    """Number of exteroception SENSOR readings: 1 for the scan, one per camera for depth."""
    extero_spec: list[dict] = field(default_factory=list)
    """Camera contract behind those readings (depth only); see ``depth_obs.py``."""
    extero_encoder: str = "mlp"
    """Which embedding the graph bakes in: ``"mlp"`` (one token per reading) or ``"cnn"``.

    Not needed to RUN the graph -- the embedding is inside it -- but it says whether
    :attr:`extero_image_hw` is load-bearing."""
    extero_image_hw: list[tuple[int, int]] = field(default_factory=list)
    """Per-camera ``(height, width)`` the graph folds each flat row into (CNN only).

    The reason this travels at all: ``extero_sensor_dim`` cannot distinguish a 30x26 camera
    from a 26x30 one, and under a convolution a transposed camera is a SCRAMBLED image
    rather than a reordered vector -- every tensor the right size, every kernel reading the
    wrong neighbourhood. Empty for a flat-MLP graph (where pixel order genuinely does not
    matter) or one exported before the field existed."""
    extero_clip: float | None = None
    """Clip band this graph was TRAINED under, or ``None`` for the per-source default.

    Travels in ``fm_meta`` because a live reading clamped harder (or looser) than the
    training data is silently a differently-bounded sensor. Absent in a graph exported
    before the field, where the default is what it used.
    """
    extero_tokens_per_sensor: int = 1
    """Transformer tokens one reading becomes (the CNN grid's cell count; 1 for the MLP)."""
    fm_snap_every: int | None = None
    """Re-plan period (control steps) the model's training targets were cut at.

    ``GenTrackerPolicy`` must use the same ``fm_update_every``: the horizon ``F`` is sized
    from this period. ``None`` for a graph exported before the field existed."""

    @property
    def extero_flat_dim(self) -> int:
        """Flat width of the exteroception input, ``extero_sensor_count * extero_sensor_dim``."""
        return self.extero_sensor_count * self.extero_sensor_dim
    has_history_null: bool = False
    """Whether the graph accepts the ``history_null`` input (learned null-history embedding).

    False for an ``fm.onnx`` exported before that input existed; the caller must then fall back
    to withholding the generator until its history window fills, and cannot match the RL side's
    step-0 behaviour. Re-export with ``export_fm_onnx.py`` to enable it."""

    @property
    def frame_names(self) -> list[str]:
        """Predicted (target) extra-frame order -- what ``ee_ref`` consumes."""
        return self.target_frame_names

    @property
    def num_extra_frames(self) -> int:
        """Alias for the predicted (target) extra-frame count."""
        return self.target_num_extra_frames


_WARNED_NO_NULL = False


def _check_state_layout(onnx_path: str, meta: dict) -> tuple[str, ...]:
    """Validate the graph's declared state layout and return the blocks it predicts.

    The generator predicts window nodes DIRECTLY into the tracker's observation, so a
    layout disagreement is not a bad lift any more -- it is a scrambled observation, fed to
    a policy that will happily act on it. Nothing observable in the graph reveals it: ``D``,
    ``nj`` and ``len(target_frame_names)`` can coincide between layouts, and the previous
    (root-block / ee-block) layout is exactly 3 channels wider for every ``(nj, K)``, so an
    old graph is close enough to look plausible.

    What is checked here is that the declaration is INTERNALLY coherent -- known block
    names, canonical order, widths and ``D`` all agreeing with ``(nj, K)``. Which selection
    is the RIGHT one is not knowable from the generator alone: it has to equal the
    tracker's, and that comparison happens in ``GenTrackerPolicy.load`` where both
    artifacts are in hand.

    Args:
        onnx_path: Path of the graph being loaded, for the error message.
        meta: The parsed ``fm_meta`` mapping.

    Returns:
        The declared block names, in packing order.

    Raises:
        ValueError: If the declared layout is missing, unknown, or inconsistent with ``D``.
    """
    nj, K, d = int(meta["nj"]), len(meta["target_frame_names"]), int(meta["D"])
    declared = meta.get("window_blocks")
    if declared is None:
        raise ValueError(
            f"{onnx_path}: fm_meta has no 'window_blocks'. This graph predates the "
            "reference-window representation -- it emits the old relative-state vector "
            f"(2*nj + 15 + 3*K = {2 * nj + 15 + 3 * K} wide) rather than window nodes "
            f"({window_dim(nj, K)} wide), and the two are not interchangeable. "
            "Re-record, retrain and re-export."
        )
    names = tuple(str(n) for n, _ in declared)
    widths = [int(w) for _, w in declared]
    expected = window_blocks(nj, K, names)   # raises on an unknown name
    if [n for n, _ in expected] != list(names):
        raise ValueError(
            f"{onnx_path}: fm_meta window_blocks are out of packing order: {list(names)} "
            f"should be {[n for n, _ in expected]}. A window is a concatenation in a fixed "
            "order, so this is a different vector, not the same one relabelled."
        )
    if [w for _, w in expected] != widths:
        raise ValueError(
            f"{onnx_path}: fm_meta window_blocks widths {widths} != this runtime's "
            f"{[w for _, w in expected]} for nj={nj}, K={K}. Re-export the FM checkpoint."
        )
    if d != sum(widths):
        raise ValueError(
            f"{onnx_path}: fm_meta D={d} but its own window_blocks sum to {sum(widths)}."
        )
    return names


def load_fm_context(onnx_path: str, num_threads: int = 0, providers=None) -> FmContext:
    """Load ``fm.onnx`` and parse its ``fm_meta`` metadata into an :class:`FmContext`."""
    sess = make_session(onnx_path, num_threads, providers)
    raw = sess.get_modelmeta().custom_metadata_map
    if "fm_meta" not in raw:
        raise ValueError(f"{onnx_path} has no 'fm_meta' metadata; re-export with metadata.")
    m = json.loads(raw["fm_meta"])
    source = str(m.get("extero_source", "height_scan"))
    # A graph exported before the exteroception fields were renamed is REFUSED rather than
    # read through a default. Both keys used to be read with ``.get(..., 1)``, which on a
    # stale graph silently described a one-sensor, one-pixel camera -- the deploy then
    # failed later, in the camera-contract check, with a message about pixel counts. The
    # asymmetry with the .pt loader (which does shim the old names) is deliberate: fixing
    # this is one re-export, while a checkpoint would be a retrain.
    stale = [old for old, new in (("height_dim", "extero_sensor_dim"),
                                  ("extero_tokens", "extero_sensor_count"))
             if new not in m and old in m]
    if stale:
        raise ValueError(
            f"{onnx_path}: fm_meta carries the pre-rename exteroception keys {stale} and not "
            "their replacements (extero_sensor_dim / extero_sensor_count). Re-export with "
            "scripts/flow_matching/export_fm_onnx.py -- the checkpoint itself still loads."
        )
    # Same rule for a DEPTH graph that does not state its clip band. `None` used to mean "use
    # the per-source default", which was unambiguous only while the two tables agreed; since the
    # depth band widened, robot_rl resolves 5.575 and EXTERO_CLIP_NP below resolves 2.0, so a
    # None here silently ran a generator 3.5 of a band tighter than it trained. Nothing in the
    # graph says which era it came from, so it is stated or it is refused -- and unlike the
    # checkpoint, fixing a graph is one re-export.
    if source == "depth" and m.get("extero_clip") is None:
        raise ValueError(
            f"{onnx_path}: fm_meta states no extero_clip for a DEPTH generator. That is "
            "ambiguous rather than absent -- the training and deploy per-source defaults are "
            "+-5.575 and +-2.0 respectively, so the graph would run a different band than it "
            "trained on. Re-export with scripts/flow_matching/export_fm_onnx.py from a "
            "checkpoint whose config states extero_clip (the exporter now refuses one that "
            "does not)."
        )
    # The old history_source refusal is gone: the deploy runtime CAN produce this history,
    # because the history is the runtime's own last output rather than a measured state it
    # would have to estimate.
    blocks = _check_state_layout(onnx_path, m)
    return FmContext(
        sess=sess, N=m["N"], F=m["F"], D=m["D"], nj=m["nj"],
        reference_blocks=blocks,
        target_num_extra_frames=len(m["target_frame_names"]),
        num_steps=m["num_steps"], cmd_dim=m["cmd_dim"],
        cmd_steps=list(m["cmd_steps"]), num_cmd_tokens=m["num_cmd_tokens"],
        joint_names=list(m["joint_names"]), target_frame_names=list(m["target_frame_names"]),
        extero_sensor_dim=int(m["extero_sensor_dim"]),
        extero_source=source, extero_sensor_count=int(m["extero_sensor_count"]),
        # Parsed by the contract's OWN source: the same field holds a camera pair for one
        # checkpoint and a ray grid for another. It used to go through the depth parser
        # unconditionally, so a scan-conditioned graph was refused at load for "missing
        # field(s) ['fx', 'fy', 'cx', 'cy', ...]" -- camera fields a height scanner has no
        # reason to carry, naming a defect that was not there.
        extero_spec=parse_extero_spec(m.get("extero_spec"), source,
                                      source=f"{onnx_path} fm_meta"),
        # ``.get`` here, unlike the two above: these describe the EMBEDDING, and a graph
        # exported before it was selectable is unambiguously the flat one. The defaults are
        # therefore correct for such a graph rather than a guess -- which is exactly the
        # test the two renamed keys fail (they would describe a camera).
        extero_channels=int(m.get("extero_channels", 1)),
        extero_encoder=str(m.get("extero_encoder", "mlp")),
        extero_image_hw=[(int(h), int(w)) for h, w in (m.get("extero_image_hw") or [])],
        extero_tokens_per_sensor=int(m.get("extero_tokens_per_sensor", 1)),
        extero_clip=(None if m.get("extero_clip") is None else float(m["extero_clip"])),
        fm_snap_every=(int(m["fm_snap_every"]) if m.get("fm_snap_every") else None),
        # Trust the graph itself over the metadata flag: an input either exists or it does not.
        has_history_null=any(i.name == "history_null" for i in sess.get_inputs()),
    )


def build_history_tokens(ctx: FmContext, nodes: np.ndarray) -> np.ndarray:
    """The ``N`` autoregressive context tokens ``(V, N, D)``, raw.

    A shape check and nothing else. The history is the ``N`` reference-window nodes this
    runtime last served -- its own output -- already expressed relative to the robot at the
    step each was served, so there is nothing to transform. The ``fm.onnx`` graph folds in
    the normalization.

    That is what took odometry off the generator's critical path: the previous version
    built these from measured world-frame root poses, which hardware only has via FAST-LIO.

    Args:
        ctx: Loaded FM context.
        nodes: Raw window nodes ``(V, N, D)``, oldest first.

    Returns:
        ``(V, N, D)`` float64.

    Raises:
        ValueError: If ``nodes`` does not match the graph's ``(N, D)``.
    """
    arr = np.asarray(nodes, dtype=np.float64)
    if arr.shape[1:] != (ctx.N, ctx.D):
        raise ValueError(
            f"history must be (V, {ctx.N}, {ctx.D}) window nodes for this graph, "
            f"got {arr.shape}."
        )
    return arr


def build_extero_tokens(ctx: FmContext, values: np.ndarray) -> np.ndarray:
    """Build the **raw** exteroception tokens ``(V, extero_sensor_count * extero_sensor_dim)``.

    Source-agnostic, mirroring ``flow_matching.online.build_extero_tokens``: the context
    knows which sensor the checkpoint was trained on, so the caller passes whichever
    reading that names -- the live per-ray height scan (body-relative, yaw-aligned) or the
    per-camera normalised depth rows -- and the clip band follows from
    ``ctx.extero_source``. Both are already in the FM's xyz+yaw ego frame, so there is no
    re-anchoring. The ``fm.onnx`` graph folds in the normalization, so this only shapes
    and sanitises.

    Args:
        ctx: Loaded FM context.
        values: Raw reading, shape ``(V, T, dim)``, ``(V, T*dim)`` or ``(T*dim,)``.

    Returns:
        Raw tokens flattened to ``(V, T*dim)`` -- the graph reshapes internally.

    Raises:
        ValueError: If the flat width does not match the model's exteroception input.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    flat = arr.reshape(arr.shape[0], -1)
    if flat.shape[-1] != ctx.extero_flat_dim:
        raise ValueError(
            f"exteroception width {flat.shape[-1]} != fm extero_flat_dim {ctx.extero_flat_dim} "
            f"({ctx.extero_sensor_count} x {ctx.extero_sensor_dim}, source '{ctx.extero_source}')."
        )
    # Missed rays / bad sensor reads arrive as +/-inf or NaN; the fm.onnx graph folds in
    # the normalization, so a non-finite value here would poison it. The band is
    # per-source: metres for the scan, normalised units for depth.
    return sanitize_extero_np(flat, ctx.extero_source, clip=ctx.extero_clip)


def build_cmd_tokens(ctx: FmContext, future_cmd: np.ndarray) -> np.ndarray:
    """Stride the per-step commanded-velocity profile into **raw** command tokens.

    Args:
        ctx: Loaded FM context.
        future_cmd: Raw command profile over the window ``(V, F, cmd_dim)``.

    Returns:
        Raw strided command tokens ``(V, num_cmd_tokens, cmd_dim)`` (the ``fm.onnx``
        graph folds in the command normalization).
    """
    return future_cmd[:, ctx.cmd_steps, :]


def sample_window(ctx: FmContext, history, extero, cmd_tokens, noise, history_null=None) -> np.ndarray:
    """Sample the ``F``-node reference window. No world lift -- the window IS the output.

    Node 0 is the current step and every node is already relative to the robot at this
    re-plan, in the layout the tracker's ``reference_window`` observation expects. The
    caller slices the offsets it needs straight into the observation, and pushes node 0
    back as the next step's history.

    Args:
        ctx: Loaded FM context.
        history: Raw context tokens ``(V, N, D)`` from :func:`build_history_tokens`.
        extero: Raw exteroception tokens ``(V, T*extero_sensor_dim)``.
        cmd_tokens: Raw strided command tokens ``(V, num_cmd_tokens, cmd_dim)``.
        noise: Sampling noise ``(V, F, D)`` (caller-supplied for determinism).
        history_null: Optional ``(V,)`` bool/float; where set, the graph swaps the history
            content for the learned ``null_history`` embedding, so the generator can be
            sampled before its history window has filled. Silently ignored (with a one-time
            warning) on a graph exported without the ``history_null`` input.

    Returns:
        Denormalized window nodes ``(V, F, D)``.
    """
    feed = {
        "history": history.astype(np.float32),
        "extero": extero.astype(np.float32),
        "cmd_tokens": cmd_tokens.astype(np.float32),
        "noise": noise.astype(np.float32),
    }
    if ctx.has_history_null:
        nz = (np.zeros((history.shape[0], 1), dtype=np.float32) if history_null is None
              else np.asarray(history_null, dtype=np.float32).reshape(-1, 1))
        feed["history_null"] = nz
    elif history_null is not None and np.any(history_null):
        global _WARNED_NO_NULL
        if not _WARNED_NO_NULL:
            print("[WARN fm_session] This fm.onnx has no 'history_null' input, so the learned "
                  "null-history embedding cannot be used; the generator will condition on a "
                  "not-yet-full history window. Re-export with export_fm_onnx.py.")
            _WARNED_NO_NULL = True
    return ctx.sess.run(["trajectory"], feed)[0].astype(np.float64)  # (V, F, D)
