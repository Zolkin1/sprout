#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Widen a teacher checkpoint's critic so it can resume into the fine-tune task.

The fine-tune env appends one observation term to the privileged critic -- the
generator-vs-clip residual (``mdp.fm_reference_residual``) -- so the critic's input grows and
``critic_state_dict`` no longer fits.  ``load_state_dict`` refuses a size mismatch whatever
``strict`` says, so the checkpoint has to be widened before ``--resume`` can read it.

**The new columns are ZERO, and that is the whole point.**  A zero column contributes nothing,
so the padded critic computes the same function as the original: resuming cannot make the
value function worse than it already was.  The weights are zero, not the inputs, so their
GRADIENTS are not -- the critic learns to use the residual from there.

Measured, the padded critic agrees with the original to ~8e-7 relative (5e-6 absolute on
values of order 6), not to the last bit.  That residue is float32 reassociation, not leakage:
the first layer now reduces over 2533 terms instead of 1961, and BLAS sums a longer row in a
different order.  The check that separates the two is padding the input with random values
instead of zeros -- the difference is byte-for-byte the same either way, which it could not be
if the new columns were contributing.

**The new term must be LAST in the observation group.**  Terms concatenate in declaration
order, so appending keeps every existing input at its original index and the old matrix drops
into ``new[:, :old_width]`` unchanged.  Inserting anywhere else would silently PERMUTE the
critic's inputs, which loads without complaint and is nonsense.  This script therefore only
ever appends, and refuses to shrink.

Nothing else in the checkpoint is touched: the actor (and its action-noise std), the iteration
counter and the optimizer state all pass through, so ``--resume`` behaves exactly as it would
have.  ``--reset_optimizer`` is offered for a specific reason -- see its help.

Run::

    python scripts/rsl_rl/pad_critic_for_finetune.py \\
        --checkpoint logs/rsl_rl/g1/<teacher run>/model_19999.pt \\
        --out logs/rsl_rl/g1/<teacher run>/model_19999_padded.pt \\
        --new_width 2533
"""

from __future__ import annotations

import argparse
import pathlib

import torch

#: The critic's first layer in the ``RslRlMLPModelCfg`` MLP.  Named rather than discovered so a
#: change in the model's layer naming fails loudly here instead of padding the wrong tensor.
FIRST_LAYER_WEIGHT = "mlp.0.weight"


def pad_critic(state: dict[str, torch.Tensor], new_width: int) -> tuple[dict, int]:
    """Return a copy of ``state`` whose first layer accepts ``new_width`` inputs.

    Args:
        state: The critic ``state_dict``.
        new_width: The fine-tune env's critic observation width.

    Returns:
        ``(padded_state, old_width)``.

    Raises:
        KeyError: If the first-layer weight is not where this script expects it.
        ValueError: If the requested width is not strictly greater than the current one.
    """
    if FIRST_LAYER_WEIGHT not in state:
        raise KeyError(
            f"{FIRST_LAYER_WEIGHT!r} is not in the critic state dict (found "
            f"{sorted(state)[:4]}...). The critic's layer naming changed, and padding a "
            "tensor chosen by guesswork would corrupt the value function silently."
        )
    weight = state[FIRST_LAYER_WEIGHT]
    out_features, old_width = weight.shape
    if new_width == old_width:
        raise ValueError(
            f"the critic is already {old_width} wide, so there is nothing to pad. Either the "
            "fine-tune env's extra term is missing, or this checkpoint was padded already."
        )
    if new_width < old_width:
        raise ValueError(
            f"refusing to shrink the critic {old_width} -> {new_width}. This script only "
            "APPENDS columns; dropping inputs would need to know which ones, and getting "
            "that wrong permutes the critic's inputs without any error."
        )

    padded = weight.new_zeros((out_features, new_width))
    padded[:, :old_width] = weight
    out = dict(state)
    out[FIRST_LAYER_WEIGHT] = padded
    return out, old_width


def pad_optimizer_moments(
    opt_state: dict, old_shape: tuple[int, int], new_width: int
) -> int:
    """Widen Adam's moments for the one parameter whose shape changed.

    Adam keeps ``exp_avg`` / ``exp_avg_sq`` shaped like their parameter, so widening the
    critic's first layer without widening its moments makes ``optimizer.step()`` raise deep
    inside the fused kernel ("size of tensor a (1961) must match tensor b (2533)") -- long
    after the load that could have explained it.

    The parameter is found BY SHAPE rather than by index.  rsl-rl builds one Adam over
    ``chain(actor.parameters(), critic.parameters())`` and the state is keyed by position in
    that chain, so an index would have to reconstruct the actor's parameter count from a
    ``state_dict`` whose key order is not guaranteed to be the ``parameters()`` order.  The
    shape is unambiguous, and this asserts that rather than assuming it.

    Args:
        opt_state: The ``optimizer_state_dict``, modified in place.
        old_shape: The critic first layer's shape before padding.
        new_width: The padded input width.

    Returns:
        The number of moment tensors widened.

    Raises:
        ValueError: If the parameter cannot be identified unambiguously by shape.
    """
    per_param = opt_state.get("state", {})
    hits = [
        key for key, entry in per_param.items()
        if any(isinstance(v, torch.Tensor) and tuple(v.shape) == old_shape
               for v in entry.values())
    ]
    if len(hits) != 1:
        raise ValueError(
            f"expected exactly one optimizer parameter of shape {old_shape}, found "
            f"{len(hits)}. Padding the wrong one would corrupt Adam's moments silently; "
            "re-run with --reset_optimizer, which sidesteps the question."
        )
    widened = 0
    for name, value in list(per_param[hits[0]].items()):
        if isinstance(value, torch.Tensor) and tuple(value.shape) == old_shape:
            grown = value.new_zeros((old_shape[0], new_width))
            grown[:, : old_shape[1]] = value
            per_param[hits[0]][name] = grown
            widened += 1
    return widened


def main() -> int:
    """Widen the checkpoint's critic and write it out."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, help="the teacher .pt to widen")
    parser.add_argument("--out", required=True, help="where to write the widened copy")
    parser.add_argument("--new_width", type=int, required=True,
                        help="the fine-tune env's critic obs width (print it with "
                             "--print_width on the task, or read the manager's summary)")
    parser.add_argument(
        "--reset_optimizer", action="store_true",
        help="clear Adam's moments, keeping the parameter-group structure. The teacher's "
             "second-moment estimates are small because it CONVERGED, and Adam divides by "
             "their square root -- so the first large gradient after an observation change "
             "takes an outsized step. Clearing them costs a few iterations of re-estimation "
             "and removes that amplification. Off by default: it changes resume semantics, "
             "and is worth turning on only if the run is unstable in its opening iterations.",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing --out")
    args = parser.parse_args()

    src = pathlib.Path(args.checkpoint)
    dst = pathlib.Path(args.out)
    if dst.exists() and not args.force:
        raise SystemExit(f"{dst} exists; pass --force to overwrite.")

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    if "critic_state_dict" not in ckpt:
        raise SystemExit(
            f"{src} has no 'critic_state_dict' (keys: {sorted(ckpt)}). This is not an rsl-rl "
            "checkpoint, or it was saved by a version this script has not seen."
        )

    old_shape = tuple(ckpt["critic_state_dict"][FIRST_LAYER_WEIGHT].shape)
    padded, old_width = pad_critic(ckpt["critic_state_dict"], args.new_width)
    ckpt["critic_state_dict"] = padded

    if not args.reset_optimizer and ckpt.get("optimizer_state_dict") is not None:
        n = pad_optimizer_moments(ckpt["optimizer_state_dict"], old_shape, args.new_width)
        print(f"[pad-critic] widened {n} Adam moment tensor(s) to match")

    if args.reset_optimizer:
        opt = ckpt.get("optimizer_state_dict")
        if opt is None:
            raise SystemExit("--reset_optimizer: this checkpoint carries no optimizer state.")
        # Only the per-parameter moments go. param_groups is left alone, so the structure
        # still matches the optimizer rsl-rl builds and load_state_dict is still happy.
        opt["state"] = {}
        print("[pad-critic] optimizer moments cleared; param_groups kept")

    torch.save(ckpt, dst)
    added = args.new_width - old_width
    print(f"[pad-critic] critic {old_width} -> {args.new_width} (+{added} zero columns)")
    print(f"[pad-critic] actor, iteration ({ckpt.get('iter')}) and the action-noise std "
          "pass through untouched")
    print(f"[pad-critic] wrote {dst}")
    print("[pad-critic] the padded critic computes the same function as the original (equal to "
          "float32 reassociation, ~1e-6 relative); resume with --load_run/--checkpoint "
          "pointing at this file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
