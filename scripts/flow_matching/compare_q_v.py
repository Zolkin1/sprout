# Copyright 2026 Zachary Olkin. All rights reserved.

"""Is the ACTION worth anything?  ``Q(s, a)`` against ``V(s)`` on identical ranking pairs.

**DEPRECATED 2026-09-10, and currently BROKEN. Not a warning about style -- it does not run.**

``Q(s, a)`` borrows its token assembly from :class:`~robot_rl.flow_matching.value.FlowValueModel`
(``FlowQModel.value_tokens`` calls ``super().value_tokens``), and that critic was cut down to the
architecture the paper describes: ONE readout token, ONE scalar value head, and no scan or proprio
token streams. Q therefore lost the exteroception it was fitted with, and ``ValueInputs`` no longer
accepts ``scan=`` / ``proprio=``.

Left broken deliberately rather than quietly rewired, because reviving it is a RESEARCH decision
and not a mechanical fix: Q would need its own scan and proprio streams, inserted before its action
tokens so its own token order is preserved, and every fitted ``Q*.pt`` predates that and would need
refitting anyway. Doing it silently would re-specify a second experiment as a side effect of a
documentation goal.

To revive: give ``FlowQModel`` its own scan/proprio encoders and type embeddings (it already owns
its head, for the same reason), then refit. The best-of-K result this line produced is recorded in
``oracle-gain-ceiling`` -- one oracle decision was worth +0.0099 of success, which is the number
that should decide whether reviving it is worth the GPU time.

``fit_q`` reports a within-state ranking accuracy, and on its own that number cannot be read.
``V`` is not at chance on the same task: the two episodes in a ``(clip, replan)`` group have
diverged a little even though they share terrain, command and start, so a state-only critic
can rank them from that divergence alone.  The ACTION's contribution is the GAP, and only a
paired comparison on the same pairs measures it.

Both models are scored over the same held-out windows, the pairs are formed once, and the two
accuracies are computed against that one set -- so the comparison is not confounded by a
different pair population, a different split, or a different label.

Reported per re-plan DEPTH as well as pooled, because the near-replicate assumption weakens
with depth: two episodes on one clip are nearly the same state at re-plan 1 and quite
different by re-plan 30.  If the gap is real it should be largest where the states are closest,
and a gap that GROWS with depth is more likely the two models disagreeing about a state than
about an action.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_ROOT, "source", "robot_rl"), os.path.join(_ROOT, "scripts", "rsl_rl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fit_q  # noqa: E402
from robot_rl.flow_matching import online, qvalue, value  # noqa: E402


def _v_batch(ctx, W, idx, device, scan_hw):
    """The state half only, for scoring ``V``.

    Args:
        ctx: FM context.
        W: Window columns.
        idx: Rows.
        device: Torch device.
        scan_hw: Scan grid.

    Returns:
        A :class:`ValueInputs`.
    """
    return fit_q._batch(ctx, W, idx, device, scan_hw, torch.as_tensor(
        ctx.cmd_steps, dtype=torch.long, device=device)).state


def main() -> int:
    """Score both models on one pair set and report the gap."""
    raise SystemExit(
        "DEPRECATED and BROKEN: the Q(s, a) line borrows its tokens from FlowValueModel, "
        "which was cut to one readout, one scalar head and NO scan/proprio streams for the "
        "paper. Q lost the exteroception it was fitted with, and ValueInputs no longer takes "
        "scan=/proprio=. Reviving it means giving FlowQModel its own scan/proprio streams and "
        "refitting -- a research decision, not a mechanical fix. See the module docstring.")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollout", required=True)
    p.add_argument("--q_ckpt", required=True)
    p.add_argument("--v_ckpt", required=True, help="A fit_value checkpoint, e.g. V_priv.fold0.pt")
    p.add_argument("--max_windows", type=int, default=120000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="")
    args = p.parse_args()

    W, meta, lab = fit_q._load(args.rollout, args.max_windows)
    scan_hw = tuple(meta["attrs"]["height_scan_hw"])
    qk = torch.load(args.q_ckpt, map_location="cpu", weights_only=False)
    ctx = online.load_fm_context(qk["fm_checkpoint"], device=args.device)
    steps = torch.as_tensor(ctx.cmd_steps, dtype=torch.long, device=args.device)

    q = qvalue.FlowQModel(copy.deepcopy(ctx.model), action_dim=int(qk["action_dim"]),
                          action_steps=int(qk["action_steps"]), scan_hw=tuple(qk["scan_hw"]),
                          proprio_dim=int(qk["proprio_hw"][-1]),
                          proprio_steps=int(qk["proprio_hw"][0]),
                          context_dim=int(qk["context_dim"])).to(args.device)
    q.load_state_dict(qk["model"])
    q.eval()

    vk = torch.load(args.v_ckpt, map_location="cpu", weights_only=False)
    if os.path.abspath(str(vk["fm_checkpoint"])) != os.path.abspath(str(qk["fm_checkpoint"])):
        print(f"[warn] Q and V borrow DIFFERENT generator encoders:\n"
              f"       Q {qk['fm_checkpoint']}\n       V {vk['fm_checkpoint']}\n"
              "       The gap below then mixes 'the action helps' with 'a different trunk', "
              "which is not what this measures.")
    v = value.FlowValueModel(copy.deepcopy(ctx.model), scan_hw=tuple(vk["scan_hw"]),
                             proprio_dim=int(vk["proprio_hw"][-1]),
                             proprio_steps=int(vk["proprio_hw"][0]),
                             context_dim=int(vk["context_dim"])).to(args.device)
    v.load_state_dict(vk["model"])
    v.eval()

    n = W["episode_uid"].size
    idx = np.arange(n)
    pq = np.zeros(n)
    pv = np.zeros(n)
    with torch.no_grad():
        for i in range(0, n, args.batch):
            j = idx[i:i + args.batch]
            b = fit_q._batch(ctx, W, j, args.device, scan_hw, steps)
            pq[i:i + j.size] = q.p_fail(b).cpu().numpy()
            # V's side is its VALUE head, negated. It used to be `v.p_fail`, a survival
            # probability on the same scale as Q's -- but that head is gone from
            # FlowValueModel, and this comparison does not need the scales to match: both
            # sides go through `within_state_ranking`, which reads the ORDER alone. The sign
            # does matter, and flips: a higher P(fail) predicts failure, while a higher
            # expected RETURN predicts success.
            pv[i:i + j.size] = -v.quality(b.state).cpu().numpy()

    f = lab["failed"]
    rq = fit_q.within_state_ranking(pq, f, W["clip_id"], W["replan_index"])
    rv = fit_q.within_state_ranking(pv, f, W["clip_id"], W["replan_index"])
    print(f"\nscored {n:,} windows -> {rq['n_pairs']:,} ranking pairs "
          f"in {rq['n_groups']:,} groups")
    print(f"\n  {'model':>6} {'within-state rank acc':>23}")
    print(f"  {'V(s)':>6} {rv['acc']:>23.4f}")
    print(f"  {'Q(s,a)':>6} {rq['acc']:>23.4f}")
    print(f"  {'GAP':>6} {rq['acc'] - rv['acc']:>+23.4f}   <- what the ACTION is worth")
    print("\n  0.5 is chance. V is NOT expected to sit at 0.5: the two episodes in a group "
          "have\n  diverged slightly, so a state-only critic can rank them from that alone.")
    print(f"\n  {'replan depth':>13} {'pairs':>9} {'V':>8} {'Q':>8} {'gap':>8}")
    for d in sorted(set(rq["by_depth"]) & set(rv["by_depth"])):
        aq, nq = rq["by_depth"][d]
        av, _ = rv["by_depth"][d]
        print(f"  {f'{d}-{d + 4}':>13} {nq:>9,} {av:>8.3f} {aq:>8.3f} {aq - av:>+8.3f}")
    print("\n  Read the DEPTH table: the near-replicate assumption is tightest at low depth, "
          "so a\n  real action effect should be clearest there. A gap that GROWS with depth is "
          "more\n  likely the two models disagreeing about a STATE than about an action.")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"v": rv, "q": rq, "gap": rq["acc"] - rv["acc"],
                       "q_ckpt": args.q_ckpt, "v_ckpt": args.v_ckpt}, fh, indent=1, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
