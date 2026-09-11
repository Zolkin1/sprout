# Copyright 2026 Zachary Olkin. All rights reserved.

"""Score a FITTED critic on a rollout stem it never saw.

``fit_value.py`` holds out episodes and one TILE from the stem it is given, which is a
within-dataset split.  A genuinely held-out initial condition lives in its own recording -- the
box2 grid puts ``(2.25 m, 1.00 m/s)`` in its own library so a training run cannot reach it -- and
nothing reads a fitted checkpoint back against a separate stem.  This does.

**It reuses ``fit_value``'s own loader and predictors rather than reimplementing them.**  The
window table, the drop-after-latch filter, the label and the quality target are all decisions
that have to match the fit exactly or the numbers are not comparable, and a second
implementation of them is a second place for them to drift.  ``fit_value`` is a script rather
than a module, so it is imported by path.

**The checkpoint's own stamps decide how the model is rebuilt.**  ``privileged`` and
``goal_context`` differ only in the context token's WIDTH, and ``quality_bins`` only in the
head's -- so a mismatch is a shape error at best and a silent reinterpretation at worst.  All
three are read off the file, never passed in.

Usage::

    python scripts/flow_matching/score_rollouts.py \\
        logs/box2/critic_v1/V_priv.pt logs/box2/rollouts/holdout_v1.00/rollout_<trk> \\
        --fm_checkpoint <fm.pt>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import sys

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for _pkg in ("motion_gen", "robot_rl"):
    _p = str(_REPO_ROOT / "source" / _pkg)
    if _p not in sys.path:
        sys.path.insert(0, _p)
for _d in (str(_REPO_ROOT / "scripts" / "rsl_rl"), str(_HERE)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from robot_rl.flow_matching import value, value_gates  # noqa: E402
from robot_rl.flow_matching.online import load_fm_context  # noqa: E402


def _load_fit_value():
    """Import ``fit_value`` by path -- ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location(
        "fit_value", str(_HERE / "fit_value.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rebuild(ckpt: dict, ctx, device: str):
    """Reconstruct the fitted model from a checkpoint's own stamps.

    Args:
        ckpt: A ``fit_value`` checkpoint dict.
        ctx: The generator context supplying the frozen encoder.
        device: Torch device.

    Returns:
        The model, in eval mode with the checkpoint's weights loaded strictly.

    Raises:
        SystemExit: If the checkpoint carries no ``context_dim`` -- it predates the stamp and
            the width cannot be inferred without guessing which arm produced it.
    """
    if "context_dim" not in ckpt:
        raise SystemExit(
            "this checkpoint carries no `context_dim`, so the context token's width is a "
            "guess: the privileged and goal-context arms differ only in it. Re-fit rather "
            "than assume.")
    import copy

    priv = bool(ckpt.get("privileged", True))
    # `quality_bins` from the stamp, falling back to the support tensor's own length -- the
    # same preference order mjeval's reader uses, and for its reason: the stamp also carries
    # the spec, which no shape can recover.
    bins = int(ckpt.get("quality_bins", 0)) or int(
        ckpt["model"]["quality_support"].numel() if "quality_support" in ckpt["model"] else 0)
    # Constructed exactly as `fit_value._fit` constructs it, including that the scan and the
    # proprioception tokens exist only on the privileged arm: a mismatch there is a shape
    # error under a strict load, which is the good case.
    model = value.FlowValueModel(
        copy.deepcopy(ctx.model),
        scan_hw=(tuple(ckpt["scan_hw"]) if priv else None),
        proprio_dim=(int(ckpt["proprio_hw"][-1]) if priv else 0),
        proprio_steps=(int(ckpt["proprio_hw"][0]) if priv else 0),
        context_dim=int(ckpt["context_dim"]),
        quality_bins=bins,
    ).to(device)
    # The stamp against the width the flags imply. `load_state_dict` compares the checkpoint
    # to the MODEL and both come from `context_dim`, so it cannot see a stamp that disagrees
    # with them -- and the failure is not a shape error but a linear layer reading the twist
    # out of the goal's columns.
    # `worst_penalty_context` joins `goal_context` here: A checkpoint fitted with `--worst_context` stamps `worst_penalty_context` and is one block WIDER, so a check that omits the flag rejects a perfectly good critic (and, if the widths ever coincided, would accept a wrong one).
    want = value.context_width(
        privileged=priv,
        goal_context=bool(ckpt.get("goal_context", True)),
        worst_penalty_context=bool(ckpt.get("worst_penalty_context", False)))
    if int(ckpt["context_dim"]) != want:
        raise SystemExit(
            f"context_dim {ckpt['context_dim']} but privileged={priv} / "
            f"goal_context={ckpt.get('goal_context', True)} imply {want}. The token would be "
            "read with its fields offset, which is not a shape error.")
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main() -> int:
    """Score one checkpoint against one stem and print the gates."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="a fit_value .pt (V_priv.pt / V_onboard.pt)")
    p.add_argument("stem", help="rollout stem to score (without .h5)")
    p.add_argument("--fm_checkpoint", default="",
                   help="generator .pt supplying the frozen encoder; defaults to the one the "
                        "checkpoint names")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max_windows", type=int, default=0)
    p.add_argument("--reward_npz", default="",
                   help="The terrain sidecar the critic was fitted against. Required only "
                        "for a critic stamped `worst_penalty_context`, whose context token "
                        "carries a block derived from that file's `penalty` column.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--json", default="", help="also write the gates here")
    a = p.parse_args()

    fv = _load_fit_value()
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    fm_ckpt = a.fm_checkpoint or ck.get("fm_checkpoint", "")
    if not fm_ckpt or not os.path.isfile(fm_ckpt):
        raise SystemExit(f"generator checkpoint {fm_ckpt!r} not found; pass --fm_checkpoint. "
                         "It supplies the frozen encoder and all three normalizers, and a "
                         "wrong one loads with correct shapes and is silent.")
    # The generator IDENTIFIES the encoder; a path only says where it was. Warn rather than
    # refuse: scoring on another machine is legitimate, being wrong about which model is not.
    want = ck.get("fm_checkpoint_sha256", "")
    if want and fv._sha256_file(fm_ckpt) != want:
        print(f"[score] !! the generator at {fm_ckpt} does not match the sha256 this critic "
              "was fitted against. The encoder and every normalizer come from it, so the "
              "numbers below are not comparable to the fit's.")

    priv = bool(ck.get("privileged", True))
    print(f"[score] {os.path.basename(a.checkpoint)}: variant {ck.get('variant','?')}, "
          f"privileged={priv}, goal_context={ck.get('goal_context', True)}")
    # Scored under the rule the critic was FITTED under, read off its stamp. Scoring a
    # loose-rule head against strict labels (or the reverse) compares two different
    # quantities and both are plausible numbers; an unstamped checkpoint predates the flag,
    # when strict was the only behaviour.
    rule = str(ck.get("failure_rule", "strict"))
    print(f"[score] failure rule from the checkpoint: {rule}"
          + ("" if "failure_rule" in ck else "  (unstamped -- predates the flag, so strict)"))
    # Same reasoning as `predict_value`: a `--worst_context` critic cannot be scored without
    # the sidecar that block is derived from.
    wctx = bool(ck.get("worst_penalty_context", False))
    if wctx and not a.reward_npz:
        raise SystemExit(
            "this critic was fitted with --worst_context, so scoring it needs the same "
            "terrain sidecar to rebuild that context block: pass --reward_npz.")
    W, E, lab_all = fv._load(a.stem, a.max_windows, strict_failure=(rule == "strict"),
                             reward_npz=a.reward_npz, worst_context=wctx)
    ctx = load_fm_context(fm_ckpt, device=a.device)
    model = rebuild(ck, ctx, a.device)

    idx = np.arange(len(W["episode_uid"]))
    p_fail = fv._predict(model, ctx, W, idx, priv, a.device, tuple(ck["scan_hw"]), a.batch,
                         worst_so_far=lab_all.get("worst_so_far"))
    gates = fv._gates(p_fail, lab_all, W, idx)
    q_pred, q_tgt = fv._predict_quality(model, ctx, W, idx, priv, a.device,
                                        tuple(ck["scan_hw"]), a.batch,
                                        worst_so_far=lab_all.get("worst_so_far"))

    n_ep = len(np.unique(W["episode_uid"]))
    n_fail = int(lab_all[idx].sum())
    print(f"[score] {len(idx)} windows over {n_ep} episodes; "
          f"{n_fail} failed ({n_fail / max(len(idx), 1):.1%} of windows)")
    print(f"[score] AUC {gates.get('auc', float('nan')):.4f}   "
          f"ECE {gates.get('ece', float('nan')):.4f}")
    if n_fail in (0, len(idx)):
        print("[score] !! ONE CLASS ONLY -- the AUC above is undefined, not merely weak.")
    ok = np.isfinite(q_pred) & np.isfinite(q_tgt)
    if ok.sum() > 2:
        rho = value_gates.spearman(q_pred[ok], q_tgt[ok])
        rng = np.random.default_rng(0)
        shuf = value_gates.spearman(q_pred[ok], rng.permutation(q_tgt[ok]))
        print(f"[score] quality: rho {rho:+.4f} (shuffled {shuf:+.4f}) over {int(ok.sum())} "
              f"windows; pred_sd {q_pred[ok].std():.4f} vs target_sd {q_tgt[ok].std():.4f}")
        gates["quality_rho"] = float(rho)
        gates["quality_rho_shuffled"] = float(shuf)
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)) or ".", exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(fv._jsonable(gates), f, indent=2)
        print(f"[score] wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
