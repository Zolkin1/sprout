#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Re-run the flow-matching generator on a captured rollout's conditioning -- no simulator.

An ``fm_debug`` bundle records the RAW inputs ``sample_window`` consumes at every re-plan
(history nodes, exteroception, the command horizon, the null-history mask) plus the plan anchor.
With the checkpoint that is everything: this module rebuilds the conditioning, re-normalises it
through the same helpers the online path uses, and samples as many windows as asked. So the
questions that need a new sample count, more Euler steps, guidance, or zero noise cost seconds
instead of a GPU rollout, and the answers are on exactly the states the rollout visited.

Two consumers: ``fm_debug --stage view`` (the stepper's "resample here" button) and the CLI
below, whose ``--check`` mode is the load-bearing test -- an offline **zero-noise** sample must
equal the online one bit-for-bit, which is the only way to know the recorded conditioning is
complete and correctly interpreted. A missing or mis-normalised input still produces
plausible windows; only the exact match rules that out.

Example::

    # prove the bundle's conditioning reproduces the rollout, then resample
    python scripts/rsl_rl/fm_resample.py --out logs/fm_debug/<run> --mode fm --check
    python scripts/rsl_rl/fm_resample.py --out logs/fm_debug/<run> --mode fm \
        --step 180 --env 0 --num_draws 32 --save draws.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for _p in (os.path.join(_REPO, "source", "robot_rl"), os.path.join(_REPO, "source", "motion_gen"),
           _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: Bundle keys the replay needs. Absence of any one of them means the bundle predates the
#: conditioning capture and cannot be replayed -- reported as such rather than worked around,
#: since a partial replay would silently condition on zeros.
REQUIRED_KEYS = ("fm_cond_hist_nodes", "fm_cond_extero", "fm_cond_cmd_future",
                 "fm_cond_hist_null", "fm_cond_anchor_pos", "fm_cond_anchor_quat")


def check_replayable(data: dict) -> None:
    """Raise unless the bundle carries the whole conditioning block.

    Raises:
        SystemExit: Naming the missing arrays and how to get them.
    """
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise SystemExit(
            f"this bundle cannot be replayed offline: missing {missing}. It was captured "
            "before the generator conditioning was logged -- re-run `fm_debug --stage run` "
            "(any --num_draws) with the current data_logger."
        )


def conditioning_at(data: dict, step: int, env: int) -> dict[str, np.ndarray]:
    """The generator's conditioning for one ``(step, env)``, as it was at that step.

    Note the recorded conditioning is the plan CURRENTLY BEING SERVED, so it is constant across
    the ``fm_update_every`` steps between re-plans; any step inside a plan therefore replays
    that plan. Use :func:`replan_steps` to land on the re-plan itself.
    """
    check_replayable(data)
    out = {k[len("fm_cond_"):]: np.asarray(data[k][step, env]) for k in REQUIRED_KEYS}
    if "fm_pred_window" in data:
        out["served_window"] = np.asarray(data["fm_pred_window"][step, env])
    if "fm_draws" in data:
        out["draws"] = np.asarray(data["fm_draws"][step, env])
        out["scores"] = np.asarray(data["fm_scores"][step, env])
        out["chosen"] = int(np.asarray(data["fm_chosen"][step, env]))
    return out


def replan_steps(data: dict, env: int = 0) -> np.ndarray:
    """Steps at which the generator re-planned.

    Derived from the CONDITIONING changing rather than from the serving cursor: the
    conditioning is overwritten only at an inference, so a change is a re-plan by definition,
    and it is present in exactly the bundles this module can replay. The cursor
    (``fm_pred_idx``, non-increase = re-plan) is the equivalent signal where available, but it
    went unlogged for a long time -- the logger looked for a public attribute the command does
    not have -- so old bundles have the conditioning and not the cursor.
    """
    check_replayable(data)
    hist = np.asarray(data["fm_cond_hist_nodes"])[:, env]
    changed = np.any(hist[1:] != hist[:-1], axis=tuple(range(1, hist.ndim)))
    return np.nonzero(np.concatenate([[True], changed]))[0]


def infer_blocks(node_dim: int, num_joints: int, num_ee: int) -> tuple[str, ...]:
    """Which window blocks a node of width ``node_dim`` is packed with.

    The bundle records the generator's windows but not its block selection, and the selection
    is what says where ``base_pose`` ends and ``joint_pos`` begins. Recovered by searching the
    canonical packing order for the subset whose width matches -- unambiguous in practice
    (the widths are 9 / 6 / nj / nj / 3K) and it fails loudly rather than mis-slicing.

    Raises:
        ValueError: If no subset matches, or if more than one does.
    """
    from itertools import combinations

    from robot_rl.flow_matching.window import REFERENCE_WINDOW_BLOCK_NAMES, window_dim

    hits = []
    for r in range(1, len(REFERENCE_WINDOW_BLOCK_NAMES) + 1):
        for combo in combinations(REFERENCE_WINDOW_BLOCK_NAMES, r):
            if window_dim(num_joints, num_ee, combo) == node_dim:
                hits.append(combo)
    if len(hits) != 1:
        raise ValueError(
            f"cannot infer the block selection for node_dim={node_dim} with nj={num_joints}, "
            f"K={num_ee}: {len(hits)} candidates {hits}. Pass the blocks explicitly.")
    return hits[0]


def make_fk_chain(joint_names: list[str], frame_names: list[str],
                  base_link: str = "pelvis_link"):
    """Batched FK from the base to ``frame_names``, or ``None`` if it cannot be built.

    The generator's window carries ``joint_pos`` but usually not ``ee_pos``, so a foot
    trajectory has to be derived from the predicted joints. Same chain the live Isaac ribbon
    uses (``FmTrackingCommand._build_fk_chain``), so the offline drawing and the in-sim one
    agree. Diagnostic only: nothing downstream consumes FK output as if it were predicted.
    """
    try:
        from motion_gen.features import DEFAULT_MJCF
        from motion_gen.fk import FKChain

        return FKChain(DEFAULT_MJCF, list(joint_names), list(frame_names), base_link=base_link)
    except Exception as exc:
        print(f"[fm_resample] could not build the FK chain ({exc}); feet will not be drawn.")
        return None


def draws_to_world(draws: np.ndarray, anchor_pos: np.ndarray, anchor_quat: np.ndarray,
                   num_joints: int, num_ee: int, blocks=None, fk=None) -> dict[str, np.ndarray]:
    """World-frame paths of ``[n, F, D]`` windows.

    Every draw of one re-plan shares that plan's anchor, which is what makes them comparable:
    they are alternative futures from one robot pose, not from different ones.

    Args:
        draws: ``[n, F, D]`` windows in the model's layout.
        anchor_pos / anchor_quat: The plan's anchor (``[3]`` / ``[4]``, xyzw).
        num_joints: ``nj`` of the window layout.
        num_ee: ``K`` extra frames of the window layout.
        blocks: The block selection; inferred from the width when omitted (which is ambiguous
            for a joints-only tail, so pass it when the bundle records it).
        fk: An :func:`make_fk_chain` result. Given one, feet/ee paths come from FK of the
            predicted joints; without one only the base path is returned.

    Returns:
        ``{"base": [n, F, 3]}`` plus ``{"ee": [n, F, K, 3]}`` when ``fk`` is supplied and the
        window carries joints.
    """
    import torch
    from isaaclab.utils.math import quat_apply

    from robot_rl.flow_matching.window import window_to_world

    d = torch.as_tensor(np.asarray(draws), dtype=torch.float32)
    n, f = d.shape[0], d.shape[1]
    # The anchor needs the window's own singleton horizon axis -- ``[n, 1, 3]``, not
    # ``[n, 3]`` -- because window_to_world broadcasts it across the F nodes. This is the
    # shape the online caller passes (``self._anchor_pos[:, None, :]``).
    pos = torch.as_tensor(np.asarray(anchor_pos), dtype=torch.float32).view(1, 1, 3).expand(n, 1, 3)
    quat = torch.as_tensor(np.asarray(anchor_quat), dtype=torch.float32).view(1, 1, 4).expand(n, 1, 4)
    if blocks is None:
        blocks = infer_blocks(d.shape[-1], num_joints, num_ee)
    world = window_to_world(d, pos, quat, num_joints, num_ee, blocks)
    out = {"base": world["base_pos_w"].detach().cpu().numpy()}
    if fk is not None and "joint_pos" in world:
        # FK gives base-frame offsets; rotate each by that node's own predicted base
        # orientation before adding its base position -- the same composition
        # ``FmTrackingCommand._fk_ee_positions`` does, so both drawings agree.
        q = world["joint_pos"].reshape(-1, world["joint_pos"].shape[-1])
        rel, _rot, _jv, _jw = fk.pose_and_jacobian(q)
        k = rel.shape[1]
        rel = rel.reshape(n, f, k, 3)
        base_q = world["base_quat_w"][:, :, None, :].expand(n, f, k, 4).reshape(-1, 4)
        ee = (world["base_pos_w"][:, :, None, :]
              + quat_apply(base_q, rel.reshape(-1, 3)).reshape(n, f, k, 3))
        out["ee"] = ee.detach().cpu().numpy()
    return out


def load_context(checkpoint: str, device: str, num_sampling_steps: int | None = None,
                 guidance_scale: float = 1.0, zero_noise: bool = False):
    """Load the FM once for repeated sampling, wholly on ``device``.

    Not the load-on-CPU-then-move-the-model pattern the offline probes use: those normalise
    inside a Dataset on CPU and move whole batches, whereas this normalises the conditioning
    and denormalises the output through ``ctx``'s own normalizers, so the stats have to live
    where the tensors do. Splitting them raises mid-``normalize``.
    """
    from robot_rl.flow_matching.online import load_fm_context

    return load_fm_context(checkpoint, device, num_sampling_steps=num_sampling_steps,
                           zero_noise=zero_noise, guidance_scale=guidance_scale)


def sample(ctx, cond: dict[str, np.ndarray], num_draws: int, seed: int | None = None):
    """Draw ``num_draws`` windows from one conditioning -- ``[num_draws, F, D]`` torch tensor.

    One batched call, the conditioning repeated per draw, exactly as the online
    ``_sample_best_of_n`` does: the sampler draws its noise per batch row, so repeated
    conditioning gives independent draws.

    Args:
        ctx: From :func:`load_context`.
        cond: One entry of :func:`conditioning_at`.
        num_draws: How many windows to sample.
        seed: Seeds the global torch RNG first, since the sampler takes its noise from there.
            Fixing it makes a resample reproducible; it does NOT reproduce a captured draw,
            whose noise came from a different RNG stream (use ``zero_noise`` for that).

    Returns:
        ``[num_draws, F, D]`` denormalized windows, in the same layout as the served window.
    """
    import torch

    from robot_rl.flow_matching.online import (
        build_cmd_tokens,
        build_extero_tokens,
        build_history_tokens,
        sample_window,
    )

    if seed is not None:
        torch.manual_seed(int(seed))
    dev = ctx.model.parameters().__next__().device

    def rep(x: np.ndarray) -> torch.Tensor:
        """One row per draw of a single-env input."""
        t = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)[None]
        return t.repeat_interleave(num_draws, dim=0)

    history = build_history_tokens(ctx, rep(cond["hist_nodes"]))
    extero = build_extero_tokens(ctx, rep(cond["extero"]))
    cmd_tokens = build_cmd_tokens(ctx, rep(cond["cmd_future"]))
    null = torch.as_tensor(bool(cond["hist_null"]), device=dev).view(1).repeat(num_draws)
    return sample_window(ctx, history, extero, cmd_tokens, history_null_mask=null)


def check(data: dict, meta: dict, checkpoint: str, device: str, env: int, limit: int) -> int:
    """Assert an offline zero-noise sample reproduces the online window, at every re-plan.

    Only valid on a bundle captured with ``fm_zero_noise`` (and therefore one draw): with
    Gaussian noise the online window depends on RNG state this cannot recover, and the
    comparison would fail for a reason that says nothing about the conditioning.

    Returns:
        Process exit status: 0 when every checked re-plan matches.
    """
    import torch

    if "fm_pred_window" not in data:
        raise SystemExit("bundle has no 'fm_pred_window', so there is nothing to check against.")
    ctx = load_context(checkpoint, device, zero_noise=True)
    steps = replan_steps(data, env)[:limit]
    print(f"[fm_resample] checking {steps.size} re-plans (env {env}) against an offline "
          f"zero-noise sample, {ctx.num_steps} Euler steps")
    worst = 0.0
    bad = 0
    for s in steps:
        cond = conditioning_at(data, int(s), env)
        got = sample(ctx, cond, 1)[0].detach().cpu().numpy()
        want = cond["served_window"]
        diff = float(np.abs(got - want).max())
        worst = max(worst, diff)
        if diff > 1e-4:
            bad += 1
            print(f"  step {int(s):5d}: max|offline - online| = {diff:.3e}   MISMATCH")
    print(f"[fm_resample] worst max|diff| over {steps.size} re-plans: {worst:.3e}  "
          f"({'OK' if bad == 0 else f'{bad} MISMATCHES'})")
    if bad:
        print("  A mismatch means the recorded conditioning is incomplete or re-normalised "
              "differently than online -- not that the model is wrong. Compare the four "
              "fm_cond_* arrays against what _run_fm passes to sample_window.")
    return 1 if bad else 0


def main() -> int:
    """Replay or verify a bundle's generator conditioning."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="fm_debug run directory holding <mode>.npz.")
    p.add_argument("--mode", default="fm", help="Which bundle (clip|fm).")
    p.add_argument("--checkpoint", default=None,
                   help="FM checkpoint (default: the one recorded in the bundle's metadata).")
    p.add_argument("--device", default="cuda:0", help="Device to sample on.")
    p.add_argument("--env", type=int, default=0, help="Env index within the bundle.")
    p.add_argument("--step", type=int, default=None,
                   help="Rollout step to resample at (default: the first re-plan).")
    p.add_argument("--num_draws", type=int, default=16, help="Draws to take.")
    p.add_argument("--num_sampling_steps", type=int, default=None,
                   help="Override the checkpoint's Euler step count.")
    p.add_argument("--guidance_scale", type=float, default=1.0,
                   help="Classifier-free guidance over the history (1.0 = off).")
    p.add_argument("--zero_noise", action="store_true", help="Integrate from x0 = 0.")
    p.add_argument("--seed", type=int, default=None, help="Seed the sampler's RNG.")
    p.add_argument("--check", action="store_true",
                   help="Verify the conditioning reproduces the online window (needs a bundle "
                        "captured with --fm_zero_noise).")
    p.add_argument("--check_limit", type=int, default=8, help="Re-plans to check.")
    p.add_argument("--save", default=None, help="Write the draws to this .npz.")
    args = p.parse_args()

    from fm_debug import load_bundle

    data, meta = load_bundle(args.out, args.mode)
    ckpt = args.checkpoint or meta.get("fm_checkpoint")
    if not ckpt:
        raise SystemExit("no --checkpoint and the bundle records no fm_checkpoint.")
    if args.check:
        return check(data, meta, ckpt, args.device, args.env, args.check_limit)

    step = args.step if args.step is not None else int(replan_steps(data, args.env)[0])
    cond = conditioning_at(data, step, args.env)
    ctx = load_context(ckpt, args.device, args.num_sampling_steps, args.guidance_scale,
                       args.zero_noise)
    draws = sample(ctx, cond, args.num_draws, args.seed).detach().cpu().numpy()
    print(f"[fm_resample] step {step} env {args.env}: {draws.shape[0]} draws of "
          f"{draws.shape[1]}x{draws.shape[2]}, {ctx.num_steps} Euler steps, "
          f"zero_noise={ctx.zero_noise}, guidance={ctx.guidance_scale}")
    spread = float(np.abs(draws - draws.mean(axis=0, keepdims=True)).mean())
    print(f"  mean |draw - mean(draws)| = {spread:.4f} (raw window units)")
    if "served_window" in cond:
        d = np.abs(draws - cond["served_window"][None]).reshape(draws.shape[0], -1).max(axis=1)
        print(f"  max|draw - served| per draw: min {d.min():.4f}  max {d.max():.4f}")
    if args.save:
        np.savez_compressed(args.save, draws=draws, step=step, env=args.env,
                            **{k: v for k, v in cond.items()})
        print(f"[fm_resample] wrote {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
