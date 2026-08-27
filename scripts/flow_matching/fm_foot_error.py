# Copyright 2026 Zachary Olkin. All rights reserved.

"""Where the generator's motion differs from the clip, in the places that decide a fall.

``fm_error_by_offset.py`` measures the error in the window's own channels. Those are
``base_pose`` (9), ``base_vel`` (6) and ``joint_pos`` (29) -- **there is no end-effector
block**, so a foot-placement error has never been measured anywhere. On a staircase that is
the quantity that matters: a root that is 5 cm off is recoverable, a swing foot that is 5 cm
off catches a riser.

So this FKs both the prediction and the clip target from the same model and compares the
BODIES: the two ankles and the root. The difference is reported per family and per look-ahead
offset, and split into the components a perturbation could actually model:

* **const** -- a rigid offset shared by the whole horizon;
* **ramp**  -- a drift growing linearly with look-ahead;
* **residual** -- what is left, which is the part no smooth perturbation reproduces.

That split is the point. ``REFERENCE_WINDOW_NOISE`` is i.i.d. per element, and the tracker is
already near-indifferent to i.i.d. jitter, so the question is not how BIG the generator's
error is but what SHAPE it has -- and whether the shape is one the tracker could be trained
against.

**Phase is controlled by construction, not by a flag.** Forwarding a checkpoint over recorded
windows conditions it on the RECORDED history, so the prediction starts from the same state
the clip does. That is teacher forcing: the phase cannot drift, and what remains is the error
the generator makes in one shot rather than the compounding of a closed loop.

    python scripts/flow_matching/fm_foot_error.py \\
        --checkpoint logs/flow_matching/<run>/best.pt \\
        --data <recording>.h5 --num_windows 3000 --out fm_foot_error.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, os.path.join(_REPO, "source", "robot_rl"))
sys.path.insert(0, os.path.join(_REPO, "source", "motion_gen"))
sys.path.insert(0, os.path.join(_REPO, "scripts", "rsl_rl"))

from motion_gen.fk import FKChain  # noqa: E402
from recorded_data_io import load_recorded_windows  # noqa: E402
from robot_rl.flow_matching.config import FlowMatchingConfig  # noqa: E402
from robot_rl.flow_matching.dataset import FutureStateDataset  # noqa: E402
from robot_rl.flow_matching.flow import sample_trajectory  # noqa: E402
from robot_rl.flow_matching.online import load_fm_context  # noqa: E402

#: The bodies compared. The ankles are where a fall starts; the root is the reference the
#: existing noise table already covers, kept so the two are on one scale.
BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")


def decompose(err: np.ndarray) -> dict:
    """Split an error curve over look-ahead into const, ramp and residual energy.

    Args:
        err: ``(N, F)`` signed error against look-ahead, per window.

    Returns:
        Fractions of total energy in each component, and the fitted magnitudes.
    """
    n, f = err.shape
    t = np.linspace(0.0, 1.0, f)
    A = np.stack([np.ones(f), t], axis=1)                    # const, ramp
    coef, *_ = np.linalg.lstsq(A, err.T, rcond=None)          # (2, N)
    fit = (A @ coef).T
    tot = float((err ** 2).sum())
    e_const = float(((np.ones(f)[None, :] * coef[0][:, None]) ** 2).sum())
    e_fit = float((fit ** 2).sum())
    return {
        "const_frac": e_const / max(tot, 1e-12),
        "const_plus_ramp_frac": e_fit / max(tot, 1e-12),
        "residual_frac": 1.0 - e_fit / max(tot, 1e-12),
        "const_median_m": float(np.median(np.abs(coef[0]))),
        "ramp_median_m": float(np.median(np.abs(coef[1]))),
    }


def rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    """``(..., 6)`` rot6d -> ``(..., 3, 3)``. The 6 are the first two COLUMNS of R.

    Gram-Schmidt: normalise the first column, orthogonalise the second against it, and take
    the third as their cross product. This is the inverse of how the window stores the base
    orientation, so a window round-trips through it exactly.
    """
    a, b = r6[..., 0:3], r6[..., 3:6]
    c0 = a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9, None)
    b = b - (c0 * b).sum(-1, keepdims=True) * c0
    c1 = b / np.clip(np.linalg.norm(b, axis=-1, keepdims=True), 1e-9, None)
    c2 = np.cross(c0, c1)
    return np.stack([c0, c1, c2], axis=-1)


def bodies_from_window(win: np.ndarray, ctx, chain: FKChain,
                       blocks: dict) -> np.ndarray:
    """World-ish body positions implied by a window, via FK from its own root pose.

    The window carries the root pose and the joint angles; the feet follow. Both prediction
    and target go through the SAME chain, so anything the model gets wrong about the robot
    cancels and what remains is the difference between the two motions.

    Args:
        win: ``(N, F, D)`` denormalized window nodes.
        ctx: The loaded FM context (for the joint order).
        chain: FK chain over ``BODIES``, rooted at the pelvis.
        blocks: ``block_slices`` from the dataset.

    Returns:
        ``(N, F, B, 3)`` body positions in the window's own frame.
    """
    n, f, _ = win.shape
    bp = blocks["base_pose"]
    jp = blocks["joint_pos"]
    root = win[:, :, bp.start:bp.start + 3]                      # (N, F, 3)
    rot = rot6d_to_matrix(win[:, :, bp.start + 3:bp.start + 9])   # (N, F, 3, 3)
    q = torch.as_tensor(win[:, :, jp].reshape(n * f, -1), dtype=torch.float32)
    pos, _rot, _jv, _jw = chain.pose_and_jacobian(q)             # (N*F, B, 3) pelvis frame
    pos = pos.numpy().reshape(n, f, -1, 3)
    # Full rigid placement: rotate the pelvis-frame FK by the window's own base orientation,
    # then translate. A tilted base really does swing the feet, so leaving the rotation out
    # under-reports the foot error -- and the orientation is right there in the window.
    return np.einsum("nfij,nfbj->nfbi", rot, pos) + root[:, :, None, :]


@torch.no_grad()
def main() -> int:
    """Measure and report the generator's body-position error against the clip."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True, help="Recording to forward over.")
    p.add_argument("--num_windows", type=int, default=3000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--dump_traj", type=int, default=0, metavar="N",
                   help="Also save N whole predicted/target trajectories, for eyeballing "
                        "the metrics against something concrete.")
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data, meta = load_recorded_windows(a.data)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = FlowMatchingConfig.from_dict(ck["config"])
    cfg.transition_fraction = None
    if not os.path.exists(str(cfg.fk_model_path)):
        from motion_gen.features import DEFAULT_MJCF
        print(f"[foot-err] fk_model_path is not on this box; using {DEFAULT_MJCF}")
        cfg.fk_model_path = DEFAULT_MJCF
    ctx = load_fm_context(a.checkpoint, "cpu")
    model = ctx.model.to(dev).eval()

    W = data["history_state"].shape[0]
    sel = np.linspace(0, W - 1, min(a.num_windows, W)).astype(np.int64)
    keys = [k for k, v in data.items() if isinstance(v, np.ndarray) and v.shape[:1] == (W,)]
    sub = {k: data[k][sel] for k in keys}
    ds = FutureStateDataset(sub, meta, cfg, np.arange(len(sel)),
                            state_normalizer=ctx.state_norm,
                            cmd_normalizer=getattr(ctx, "cmd_norm", None),
                            extero_normalizer=getattr(ctx, "extero_norm", None))
    chain = FKChain(cfg.fk_model_path, list(ctx.joint_names), list(BODIES),
                    base_link="pelvis_link", dtype=torch.float32)

    gen = torch.Generator(device="cpu").manual_seed(a.seed)
    preds, targs = [], []
    for lo in range(0, len(ds), a.batch):
        hi = min(lo + a.batch, len(ds))
        items = [ds[k] for k in range(lo, hi)]
        st = lambda key: torch.stack([it[key] for it in items]).to(dev)  # noqa: E731
        noise = torch.randn(hi - lo, cfg.future_len, cfg.state_dim, generator=gen).to(dev)
        out = sample_trajectory(model, st("history"), st("extero"),
                                st("cmd_tokens") if "cmd_tokens" in items[0] else None,
                                cfg.future_len, cfg.state_dim, cfg.num_sampling_steps,
                                noise=noise)
        preds.append(ctx.state_norm.denormalize(out.cpu()).numpy())
        targs.append(ctx.state_norm.denormalize(
            torch.stack([it["state"] for it in items])).numpy())
    pred = np.concatenate(preds)
    targ = np.concatenate(targs)

    bp = ds.block_slices
    P = bodies_from_window(pred, ctx, chain, bp)
    T = bodies_from_window(targ, ctx, chain, bp)
    root_err = np.linalg.norm(pred[:, :, bp["base_pose"].start:bp["base_pose"].start + 3]
                              - targ[:, :, bp["base_pose"].start:bp["base_pose"].start + 3],
                              axis=-1)                                    # (N, F)
    foot_err = np.linalg.norm(P - T, axis=-1)                             # (N, F, B)
    # WHERE the foot error comes from, which decides what a perturbation has to act on.
    # bodies_from_window is FK(joints) + root, so subtracting the two FK terms leaves the
    # LIMB component -- the part the joint angles got wrong -- and the root term is the part
    # the feet inherit rigidly from the base. If the limb component is small, perturbing the
    # root alone reproduces the error; if it is not, a root-only perturbation cannot.
    # The limb term stays in the PELVIS frame, unrotated: it is the pure joint-angle
    # contribution, and rotating it would put the base's orientation error back in.
    qp = torch.as_tensor(pred[:, :, bp["joint_pos"]].reshape(-1, pred.shape[-1] and
                                                             len(ctx.joint_names)),
                         dtype=torch.float32)
    qt = torch.as_tensor(targ[:, :, bp["joint_pos"]].reshape(-1, len(ctx.joint_names)),
                         dtype=torch.float32)
    fp = chain.pose_and_jacobian(qp)[0].numpy().reshape(P.shape)
    ft = chain.pose_and_jacobian(qt)[0].numpy().reshape(T.shape)
    limb_err = np.linalg.norm(fp - ft, axis=-1)                           # (N, F, B)
    # JOINT space, in radians. The limb term above is the foot POSITION error with the base
    # removed -- still metres, still end-effector space. Whether the JOINT ANGLES have the
    # same shape is a separate question, and the two need not agree: a constant-plus-ramp in
    # joint space only maps to one in foot space where the Jacobian is roughly constant, and
    # it is not across a stride. This is what decides whether a perturbation can be applied
    # to the angles directly or has to be applied in task space and mapped back.
    jerr = pred[:, :, bp["joint_pos"]] - targ[:, :, bp["joint_pos"]]      # (N, F, nj) rad
    jn = np.linalg.norm(jerr, axis=-1)                                    # (N, F) rad

    fam_names = json.loads(meta_attr(meta, "family_names", "[]"))
    fam = np.asarray(sub.get("clip_family", np.zeros(len(sel), np.int64)))
    res = {"checkpoint": a.checkpoint, "data": a.data, "n_windows": int(len(sel)),
           "future_len": int(cfg.future_len), "bodies": list(BODIES), "per_family": {}}

    print(f"\n=== generator vs clip, {len(sel)} windows, teacher-forced by construction ===")
    print("%-22s %7s %9s %9s %9s %9s" % ("family", "n", "root p50", "foot p50", "foot p90",
                                         "foot@far"))
    for i, name in enumerate(fam_names):
        m = fam == i
        if m.sum() < 30:
            continue
        fe = foot_err[m].reshape(int(m.sum()), foot_err.shape[1], -1).max(axis=-1)
        re_ = root_err[m]
        row = dict(n=int(m.sum()),
                   root_p50=float(np.median(re_)), foot_p50=float(np.median(fe)),
                   foot_p90=float(np.percentile(fe, 90)),
                   foot_final=float(np.median(fe[:, -1])),
                   foot_first=float(np.median(fe[:, 0])),
                   limb_p50=float(np.median(limb_err[m].max(axis=-1))),
                   root=decompose(re_), foot=decompose(fe),
                   limb=decompose(limb_err[m].max(axis=-1)),
                   joint_p50=float(np.median(jn[m])),
                   joint=decompose(jn[m]))
        res["per_family"][name] = row
        print("%-22s %7d %9.4f %9.4f %9.4f %9.4f"
              % (name.replace("family_", ""), row["n"], row["root_p50"], row["foot_p50"],
                 row["foot_p90"], row["foot_final"]))

    print("\n--- WHERE the foot error comes from ---")
    print("%-22s %10s %10s %10s   %s" % ("family", "root", "limb", "foot", "limb/foot"))
    for name, row in res["per_family"].items():
        print("%-22s %10.4f %10.4f %10.4f   %.2f"
              % (name.replace("family_", ""), row["root_p50"], row["limb_p50"],
                 row["foot_p50"], row["limb_p50"] / max(row["foot_p50"], 1e-9)))

    # PER JOINT, and whether one scale vector would do. A norm over 29 joints says how big
    # the error is and nothing about its distribution: if it sits in the swing leg, a uniform
    # perturbation with the right total is still the wrong perturbation.
    jn_names = list(ctx.joint_names)
    prof = {}
    for i, name in enumerate(fam_names):
        m = fam == i
        if m.sum() < 30:
            continue
        prof[name] = np.median(np.abs(jerr[m]), axis=(0, 1))            # (nj,) rad
    res["per_joint"] = {k: v.tolist() for k, v in prof.items()}
    res["joint_names"] = jn_names

    keep_fams = [k for k in prof if "pure" not in k and "distractor" not in k]
    print("\n--- PER JOINT: median |dq| (rad), the eight largest on stairs_up ---")
    ref = prof.get("family_stairs_up", next(iter(prof.values())))
    order = np.argsort(-ref)[:8]
    hdr = "".join(f"{k.replace('family_','')[:9]:>10s}" for k in keep_fams)
    print(f"  {'joint':<26s}{hdr}")
    for j in order:
        row = "".join(f"{prof[k][j]:10.4f}" for k in keep_fams)
        print(f"  {jn_names[j].replace('_joint',''):<26s}{row}")
    tot = "".join(f"{prof[k].sum():10.4f}" for k in keep_fams)
    print(f"  {'-- sum over all 29 --':<26s}{tot}")

    # IS THE SHAPE STABLE? Two questions with different answers: does the profile look the
    # same from family to family, and does it look the same from window to window inside one.
    def unit(v):
        return v / max(float(np.linalg.norm(v)), 1e-12)
    print("\n--- is the per-joint SHAPE the same? (cosine similarity of the profile) ---")
    base = unit(prof["family_stairs_up"]) if "family_stairs_up" in prof else unit(ref)
    for k in prof:
        print(f"  vs stairs_up: {k.replace('family_',''):<20s} {float(unit(prof[k]) @ base):.3f}")
    m = fam == fam_names.index("family_stairs_up") if "family_stairs_up" in fam_names else None
    if m is not None and m.sum() > 30:
        per_win = np.abs(jerr[m]).mean(axis=1)                            # (n, nj)
        u = per_win / np.clip(np.linalg.norm(per_win, axis=1, keepdims=True), 1e-12, None)
        cos = u @ unit(prof["family_stairs_up"])
        print(f"  within stairs_up, window-to-family-mean: p10 {np.percentile(cos,10):.2f} "
              f"p50 {np.median(cos):.2f} p90 {np.percentile(cos,90):.2f}")

    print("\n--- JOINT space (rad) vs END-EFFECTOR space (m): same shape? ---")
    print("%-20s %-24s %-24s" % ("", "foot / EE  (m)", "joints  (rad)"))
    print("%-20s %8s %8s %8s %8s %8s %8s" % ("family", "p50", "+ramp", "resid",
                                             "p50", "+ramp", "resid"))
    for name, row in res["per_family"].items():
        print("%-20s %8.4f %8.2f %8.2f %8.4f %8.2f %8.2f"
              % (name.replace("family_", ""), row["limb_p50"],
                 row["limb"]["const_plus_ramp_frac"], row["limb"]["residual_frac"],
                 row["joint_p50"], row["joint"]["const_plus_ramp_frac"],
                 row["joint"]["residual_frac"]))

    print("\n--- SHAPE of the error (what a perturbation would have to reproduce) ---")
    print("%-22s %11s %11s %11s %11s" % ("family", "foot const", "+ramp", "residual",
                                         "growth 0->F"))
    for name, row in res["per_family"].items():
        g = row["foot_final"] / max(row["foot_first"], 1e-9)
        print("%-22s %10.2f %11.2f %11.2f %10.1fx"
              % (name.replace("family_", ""), row["foot"]["const_frac"],
                 row["foot"]["const_plus_ramp_frac"], row["foot"]["residual_frac"], g))

    if a.dump_traj:
        k = np.linspace(0, len(sel) - 1, a.dump_traj).astype(int)
        res["traj"] = {"pred_bodies": P[k].tolist(), "targ_bodies": T[k].tolist(),
                       "rows": k.tolist(), "family": fam[k].tolist()}
    out = a.out or os.path.join(os.path.dirname(a.checkpoint), "fm_foot_error.json")
    with open(out, "w") as fh:
        json.dump(res, fh)
    print(f"\n[foot-err] wrote {out}")
    return 0


def meta_attr(meta, name: str, default: str) -> str:
    """Read one attribute off the recording metadata, JSON-encoded."""
    v = getattr(meta, name, None)
    return json.dumps(v) if v is not None else default


if __name__ == "__main__":
    raise SystemExit(main())
