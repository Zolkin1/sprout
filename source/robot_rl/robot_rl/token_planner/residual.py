# Copyright 2026 Zachary Olkin. All rights reserved.

"""The FM's own error as a tokenizable signal: layout, normalization and the cache.

The generator is the limit of this stack (``fm/train 0.468`` against ``clip/train 0.930``),
and teacher forcing buys almost nothing once the jump families are removed -- so the gap is
prediction ACCURACY, not closed-loop drift. That is what a residual adjuster attacks: keep the
flow-matching generator as the planner of record and train a token model to correct it, because
a categorical policy has an exact per-token log-probability and entropy and can therefore be
fine-tuned with PPO without a reparameterisation trick.

This module owns the offline half of that: turning ``(plan P, clip target gt)`` pairs into a
residual ``delta``, normalizing it, and persisting it in the shape the tokenizer already eats.

.. admonition:: The composition rule is NOT restated here

    ``flow_matching.window.compose_window_residual`` is the single rule, shared with
    ``FmTrackingCommand.apply_plan_residual`` which serves it online. A residual fitted against
    a different composition would mean something else the moment it reached the robot, and
    nothing about the shapes would say so.

Three things that look like details and are not:

* **The residual node layout is IDENTICAL to the absolute one** -- 44 wide, same blocks, same
  order. So ``window_blocks``, ``metrics.channel_group_labels``, ``ReconAccumulator`` and
  ``train_tokenizer.channel_weights`` are reused untouched, and a residual RMSE is directly
  comparable to the generator's own held-out error and to the absolute tokenizer's floor.
* **The residual normalizer standardizes EVERY channel, including the rot6d block.** The state
  normalizer deliberately leaves that block raw to keep it Gram-Schmidt-able, but a DIFFERENCE
  of two near-orthonormal frames is not a rotation -- and left raw it carries ~0.02 magnitude
  against ``base_pos``'s ~0.09, which is the loss ignoring it.
* **The mean is forced to zero.** A zero residual must normalize to zero, so "leave the
  generator alone" is representable at the origin -- which is what an adjuster's ``ZERO_CODE``
  and its ``alpha = 0`` initialisation both depend on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from robot_rl.flow_matching.normalizer import Normalizer

from .data import ChunkData, cache_key

#: Residual cache format version. Bump on any change to what a build persists; an older cache
#: is refused rather than reinterpreted.
RESIDUAL_CACHE_VERSION: int = 1

#: Payload tag distinguishing the two kinds of chunk cache. Written into the envelope rather
#: than into :func:`~robot_rl.token_planner.data.cache_key` so that adding it does not
#: invalidate every absolute cache already on disk: an untagged file predates the residual arm
#: and is an absolute build, which is exactly what the default says.
ABSOLUTE_ARM: str = "absolute"
RESIDUAL_ARM: str = "residual"


@dataclass
class ResidualChunkData(ChunkData):
    """A :class:`~robot_rl.token_planner.data.ChunkData` whose chunks are FM residuals.

    ``train_nodes`` / ``val_nodes`` hold the NORMALIZED residual, and
    ``state_normalizer_state`` is the RESIDUAL normalizer -- so every consumer that converts a
    reconstruction error to physical units through it keeps working and reports metres of
    residual error, which is the number the generator's own held-out RMSE is quoted in.

    Attributes:
        train_plan / val_plan: ``(Nw, H, D)`` the generator's plans ``P``, in RAW units and
            already rotation-projected, i.e. exactly what ``online.sample_window`` served.
            Stored for every row, not just val: the plan-space gate needs them, and the
            adjuster's plan encoder consumes them as conditioning.
        absolute_normalizer_state: the generator's own STATE normalizer, kept so a consumer
            can put ``train_plan`` back into the space the generator predicts in without
            re-reading the checkpoint.
        fm_provenance: which generator produced the plans, how it was sampled, and which rows
            were judged honest. Part of the cache key.
    """

    train_plan: torch.Tensor | None = None
    val_plan: torch.Tensor | None = None
    #: The generator's CONDITIONING for each window -- normalized history rows, the
    #: normalized exteroception row, and the strided command tokens -- stored so the adjuster
    #: stage is a function of this file alone.
    #:
    #: It could be re-derived from the recording, and deliberately is not: the adjuster has to
    #: see the same conditioning the generator saw for the plan it is correcting, and a
    #: re-derivation joins two independently-built window sets by INDEX. That join is right
    #: until a filter, a lag draw or a row order changes, at which point it silently pairs a
    #: plan with someone else's depth -- which looks exactly like a model that will not learn.
    train_history: torch.Tensor | None = None
    val_history: torch.Tensor | None = None
    train_extero: torch.Tensor | None = None
    val_extero: torch.Tensor | None = None
    train_cmd: torch.Tensor | None = None
    val_cmd: torch.Tensor | None = None
    absolute_normalizer_state: dict[str, torch.Tensor] = field(default_factory=dict)
    fm_provenance: dict[str, Any] = field(default_factory=dict)


def residual_cache_key(
    provenance: dict[str, Any],
    chunk_len: int,
    reference_blocks: Sequence[str],
    fm_provenance: dict[str, Any],
) -> dict[str, Any]:
    """The fields a residual cache must agree on to be reusable.

    The absolute key plus everything about the generator that changes what a residual IS.
    ``history_source`` is in there because the same recording under a different history arm
    conditions the generator differently and therefore yields a different error; ``draw``
    because a Gaussian sample and the deployed ``x0 = 0`` are different plans; ``honesty``
    because a cache built from rows the generator trained on is a different dataset with the
    same shape.

    Args:
        provenance: The build's :attr:`ChunkData.provenance`.
        chunk_len: Nodes per chunk.
        reference_blocks: The resolved block selection.
        fm_provenance: See :attr:`ResidualChunkData.fm_provenance`.

    Returns:
        The key dict, saved beside the tensors and compared on load.
    """
    key = cache_key(provenance, chunk_len, reference_blocks)
    key["residual_version"] = RESIDUAL_CACHE_VERSION
    for name in ("fm_run", "fm_sha256", "draw", "history_source", "honesty", "fm_epoch"):
        key[f"fm_{name}" if not name.startswith("fm_") else name] = fm_provenance.get(name)
    return key


def save_residual_cache(path: str, chunks: ResidualChunkData) -> None:
    """Write a residual build to ``path``.

    Args:
        path: Destination ``.pt``.
        chunks: The build to persist.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {
        "arm": RESIDUAL_ARM,
        "key": residual_cache_key(
            chunks.provenance, chunks.chunk_len, chunks.reference_blocks, chunks.fm_provenance
        ),
        "chunks": {k: v for k, v in vars(chunks).items()},
    }
    torch.save(payload, path)


def cache_arm(payload: dict[str, Any]) -> str:
    """Which kind of chunk cache a loaded payload is.

    Args:
        payload: A ``torch.load`` of a cache file.

    Returns:
        :data:`RESIDUAL_ARM` or :data:`ABSOLUTE_ARM`. An untagged payload predates the
        residual arm and is absolute -- the one case where a missing value has a single
        correct reading, because no residual cache was ever written without the tag.
    """
    return str(payload.get("arm", ABSOLUTE_ARM))


def load_residual_cache(
    path: str, key: dict[str, Any], payload: dict[str, Any] | None = None
) -> ResidualChunkData:
    """Read a residual cache, refusing an absolute one and refusing a key mismatch.

    Args:
        path: A file written by :func:`save_residual_cache`.
        key: The key the caller requires, from :func:`residual_cache_key`.
        payload: An already-loaded ``torch.load`` of ``path``, to avoid a second full read.

    Returns:
        The cached :class:`ResidualChunkData`.

    Raises:
        ValueError: If the file is an absolute cache, or on a key mismatch naming every
            field that differs.
    """
    if payload is None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    arm = cache_arm(payload)
    if arm != RESIDUAL_ARM:
        raise ValueError(
            f"{path} is an {arm!r} chunk cache, but a RESIDUAL build was asked for. The two "
            "hold tensors of identical shape in different spaces -- absolute chunks are the "
            "clip's own future, residual chunks are the generator's error against it -- so "
            "training on the wrong one would run to completion and report plausible numbers "
            "about the wrong quantity."
        )
    have = dict(payload.get("key", {}))
    diff = {k: (have.get(k), v) for k, v in key.items() if have.get(k) != v}
    if diff:
        lines = "\n".join(f"    {k}: cached {a!r} != requested {b!r}"
                          for k, (a, b) in sorted(diff.items()))
        raise ValueError(
            f"residual cache {path} does not match this build:\n{lines}\n"
            "Delete it or point --cache somewhere else -- it is not rebuilt in place."
        )
    return ResidualChunkData(**payload["chunks"])


def fit_residual_normalizer(delta: torch.Tensor, eps: float = 1e-6) -> Normalizer:
    """Fit the residual's per-channel scale, with the mean pinned to ZERO.

    Every channel is standardized, unlike the state normalizer which leaves the rot6d block
    raw -- see this module's docstring for why that distinction does not carry over to a
    difference of two frames.

    The mean is not fitted. A residual's zero means "the generator was right", and that has to
    survive normalization for an adjuster to be able to say it: ``ZERO_CODE``, the ``alpha =
    0`` identity initialisation and the injected all-zero chunks all read the origin as "no
    correction". A fitted mean would move that point to wherever this particular dataset's
    average error happened to sit. The fitted mean is still worth REPORTING -- it is the
    generator's systematic bias -- which is why callers print it beside the scale.

    Args:
        delta: ``(N, ..., D)`` residuals in raw units; every axis but the last is pooled.
        eps: Lower bound on the per-channel std, so a dead channel cannot divide by zero.

    Returns:
        A :class:`Normalizer` with ``mean = 0``, fitted ``std``, and an all-True mask.
    """
    flat = delta.reshape(-1, delta.shape[-1]).double()
    std = flat.std(dim=0).clamp_min(eps).to(torch.float32)
    mean = torch.zeros_like(std)
    return Normalizer(mean, std, torch.ones_like(std, dtype=torch.bool))


def declared_residual_normalizer(
    num_joints: int, num_ee_frames: int, blocks: Sequence[str] | None = None
) -> Normalizer:
    """The residual normalizer as a DECLARED band rather than a dataset statistic.

    Uses ``window.residual_scale_vector`` -- the per-channel authority the RL adjuster is
    already given (0.20 m of base position, 0.30 rad of joint angle, a 15 deg tilt) -- so one
    normalized unit means "the whole correction the adjuster is allowed to make" and the same
    number means the same thing on every dataset. The cost is resolution: the measured
    residual is a fraction of that band, so a fitted scale packs the FSQ lattice better.

    Args:
        num_joints: Robot joint count ``nj``.
        num_ee_frames: End-effector frames ``K``.
        blocks: The selection the window is packed with.

    Returns:
        A :class:`Normalizer` with ``mean = 0``, the declared ``std``, and an all-True mask.

    Raises:
        ValueError: If any selected channel declares a scale of zero -- ``joint_vel`` and
            ``ee_pos`` are inert in that table, so a selection containing them has no declared
            band and must use a fitted one rather than divide by zero.
    """
    from robot_rl.flow_matching.window import residual_scale_vector  # noqa: PLC0415

    scale = torch.tensor(
        residual_scale_vector(num_joints, num_ee_frames, blocks), dtype=torch.float32
    )
    if bool((scale <= 0.0).any()):
        dead = torch.nonzero(scale <= 0.0, as_tuple=False).flatten().tolist()
        raise ValueError(
            f"the declared residual band is zero on channel(s) {dead[:8]}"
            f"{'...' if len(dead) > 8 else ''}: REFERENCE_WINDOW_RESIDUAL_SCALE leaves "
            "joint_vel and ee_pos inert because the adjuster was never calibrated on them. "
            "Use the fitted scale for a selection that packs them."
        )
    return Normalizer(torch.zeros_like(scale), scale, torch.ones_like(scale, dtype=torch.bool))


@torch.no_grad()
def find_zero_code(
    tokenizer: torch.nn.Module,
    normalizer: Normalizer,
    labels: torch.Tensor,
    group_names: Sequence[str],
    batch: int = 2048,
) -> tuple[int, dict[str, float], float]:
    """The single token id whose constant chunk decodes closest to "no correction".

    An adjuster has to be able to say "the generator was right here", and under a shared
    codebook that means one id whose decode is ~zero. Nothing pins a particular id to it --
    FSQ is parameter-free and its lattice has no exact centre on an even-levelled axis -- so
    the id is FOUND after training and RECORDED, and a build that injected no all-zero chunks
    will simply not have a good one, which is what the returned norm is for.

    Every id is scored as a whole chunk of that id repeated, because that is how the adjuster
    would emit "no correction anywhere": a decode is not per-token, so the norm of a
    single-token chunk says nothing about a constant one.

    Args:
        tokenizer: A trained ``TrajectoryTokenizer``.
        normalizer: The residual normalizer, to report the decode in PHYSICAL units --
            "2 mm of base position" is the gate, and a normalized number cannot be compared
            against it.
        labels: ``(D,)`` channel group labels from ``metrics.channel_group_labels``.
        group_names: Group names those labels index.
        batch: Ids scored per decoder pass.

    Returns:
        ``(zero_code, per_group_abs_max, overall_rms)`` -- the id, the largest absolute
        physical value the decode reaches in each channel group, and the decode's overall
        physical RMS. The per-group maxima are what the gate reads.
    """
    device = next(tokenizer.parameters()).device
    vocab, n_tok = int(tokenizer.vocab_size), int(tokenizer.num_tokens)
    std = normalizer.std.to(device)
    lab = labels.to(device)
    best_id, best_rms, best_group = -1, float("inf"), {}
    for start in range(0, vocab, batch):
        ids = torch.arange(start, min(start + batch, vocab), device=device, dtype=torch.long)
        chunk = ids[:, None].expand(-1, n_tok).contiguous()          # (b, N) constant chunks
        # Physical units: the decode is normalized residual, and the gate is in metres.
        phys = tokenizer.decode_ids(chunk).float() * std             # (b, H, D)
        rms = phys.pow(2).mean(dim=(1, 2)).sqrt()                    # (b,)
        j = int(rms.argmin())
        if float(rms[j]) < best_rms:
            best_rms = float(rms[j])
            best_id = int(ids[j])
            amax = phys[j].abs().amax(dim=0)                         # (D,)
            best_group = {
                name: float(amax[lab == g].max()) if bool((lab == g).any()) else 0.0
                for g, name in enumerate(group_names)
            }
    return best_id, best_group, best_rms


@torch.no_grad()
def plan_space_report(
    plan: torch.Tensor,
    target: torch.Tensor,
    residual_hat: torch.Tensor,
    num_joints: int,
    num_ee_frames: int,
    blocks: Sequence[str] | None,
    family: torch.Tensor,
    family_names: Sequence[str],
    vocab_size: int,
    alpha: float = 1.0,
) -> dict[str, Any]:
    """Score ``P`` and ``P (+) delta_hat`` against the clip, in PLAN space and metres.

    The gate this stage turns on, and the reason ``P`` is stored for every row. Scoring the
    residual's own reconstruction error would be equivalent on every LINEAR channel -- since
    ``(P + d_hat) - gt == d_hat - d`` -- but not on the rotation half, where the composition
    projects onto SO(3) and bounds the tilt. Those are exactly the two operations a served
    correction goes through, so the number that decides whether the correction helps has to be
    measured on their far side.

    Both arms go through one :class:`~robot_rl.token_planner.metrics.ReconAccumulator` with an
    identity normalizer, so the two columns share a reduction definition rather than each
    computing an RMSE its own way.

    Args:
        plan: ``(Nw, H, D)`` the generator's plans, raw units.
        target: ``(Nw, H, D)`` the clip's own future, raw units.
        residual_hat: ``(Nw, H, D)`` the tokenizer's reconstruction of the residual, raw units.
        num_joints: Robot joint count ``nj``.
        num_ee_frames: End-effector frames ``K``.
        blocks: The block selection all three are packed with.
        family: ``(Nw,)`` family codes.
        family_names: Names those codes index.
        vocab_size: Token vocabulary, for the accumulator's histogram (unused here).
        alpha: Gain on the reconstructed residual, passed to the composition.

    Returns:
        ``{"plan": <report>, "plan_plus_residual": <report>}``, each a
        :meth:`ReconAccumulator.report`, in metres / radians.
    """
    from robot_rl.flow_matching.window import compose_window_residual  # noqa: PLC0415

    from .metrics import ReconAccumulator, channel_group_labels  # noqa: PLC0415

    names, labels = channel_group_labels(num_joints, num_ee_frames, blocks)
    ident = torch.ones(labels.shape[0])
    adjusted = compose_window_residual(
        plan, residual_hat, num_joints, num_ee_frames, blocks, alpha=alpha
    )
    out: dict[str, Any] = {}
    for arm, pred in (("plan", plan), ("plan_plus_residual", adjusted)):
        acc = ReconAccumulator(
            group_names=names, labels=labels, std=ident, mean=torch.zeros_like(ident),
            chunk_len=int(plan.shape[1]), vocab_size=int(vocab_size),
            num_families=len(family_names), family_names=list(family_names),
        )
        # Raw units in, identity std: the accumulator's "physical" conversion is a no-op and
        # its RMSEs are already metres -- the same units the generator's own held-out error
        # and the absolute tokenizer's floor are quoted in.
        acc.update(target, pred, torch.zeros(plan.shape[0], 1, dtype=torch.long), family)
        out[arm] = acc.report()
    return out
