# Copyright 2026 Zachary Olkin. All rights reserved.

"""Look at what the advantage label calls a GOOD chunk and what it calls a BAD one.

The label is a number per re-plan and a number cannot say what it preferred.  This draws the
chunks themselves -- the generator's own emitted windows -- in viser, green for high advantage
and red for low.

**Matched on V(s), and that is the whole design.**  Taking the global top and bottom by
advantage shows EASY STATES against HARD ONES, not good chunks against bad ones: a
Monte-Carlo advantage is larger wherever the outcome was still open, so an unmatched pair is
mostly a picture of episode difficulty.  That is the same confound ``label_advantage``'s G2
gate exists to control for -- *within narrow bands of V(s_t), do positive-labelled chunks fail
less than negative ones?* -- so the pairing here uses the same control: both members of a pair
come from one narrow ``p_fail`` band and one stratum, and only then are they sorted by
advantage.  Whatever visibly differs is then the CHUNK.

Two frames, because they answer different questions:

* ``--align anchor`` (default) -- every chunk drawn from a common origin, in its own plan
  frame.  This is the "what does a good chunk LOOK like" view: the shapes are directly
  comparable because the pose they were planned from has been divided out.
* ``--align world`` -- true world positions on the tile, with the terrain drawn.  This is the
  "WHERE on the course do good and bad chunks happen" view.

Run after ``label_advantage.py``::

    python scripts/rsl_rl/view_advantage_chunks.py \\
        --rollout logs/box2/merged/box2_train \\
        --labels logs/box2/critic_v2_loose/adv_p70_w0.npz \\
        --predictions logs/box2/critic_v2_loose/predictions.npz --k 12
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for _p in ("source/motion_gen", "source/robot_rl", "transfer"):
    _d = str(_REPO_ROOT / _p)
    if _d not in sys.path:
        sys.path.insert(0, _d)
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from deploy.window_np import base_pose_block_to_world, window_block_slices  # noqa: E402

#: Green (high advantage) and red (low), as viser RGB.
HIGH_RGB = (0.15, 0.80, 0.35)
LOW_RGB = (0.85, 0.20, 0.22)

#: Joints in the shipped window layout, and no end-effector block: ``state_dim`` is
#: ``base_pose(9) + base_vel(6) + joint_pos(29)``.  Derived from the width rather than assumed,
#: so a differently packed recording is refused instead of sliced into nonsense.
_N_JOINTS = 29


def block_slices(state_dim: int):
    """Where each block sits in a window row.

    Args:
        state_dim: The row width.

    Returns:
        The block-name -> slice mapping.

    Raises:
        SystemExit: If the width is not the shipped layout, whose offsets this assumes.
    """
    if state_dim != 9 + 6 + _N_JOINTS:
        raise SystemExit(
            f"window width {state_dim} is not base_pose(9)+base_vel(6)+joint_pos({_N_JOINTS})"
            f" = {9 + 6 + _N_JOINTS}. The block offsets would be wrong; state the layout.")
    return window_block_slices(_N_JOINTS, 0, ("base_pose", "base_vel", "joint_pos"))


def join_on_window(sidecar: dict, uid: np.ndarray, ridx: np.ndarray) -> np.ndarray:
    """Expand a sidecar's per-row values onto the window table, on ``(uid, replan_index)``.

    Positional joining is wrong and silently so: ``fit_value`` drops censored episodes and
    every window at or after the first bad step, so the sidecar is SHORTER than the window
    file and a positional read shifts every value by the number of dropped rows before it.

    Args:
        sidecar: Carries ``episode_uid``, ``replan_index`` and ``advantage``.
        uid: ``[W]`` the window file's episode ids.
        ridx: ``[W]`` its re-plan indices.

    Returns:
        ``[W]`` advantage, NaN where the sidecar has no entry for that window.
    """
    key = (sidecar["episode_uid"].astype(np.int64) << np.int64(20)) + \
        sidecar["replan_index"].astype(np.int64)
    want = (uid.astype(np.int64) << np.int64(20)) + ridx.astype(np.int64)
    order = np.argsort(key)
    pos = np.searchsorted(key[order], want)
    pos = np.clip(pos, 0, key.size - 1)
    hit = key[order][pos] == want
    out = np.full(want.size, np.nan, dtype=np.float64)
    out[hit] = sidecar["advantage"].astype(np.float64)[order][pos[hit]]
    return out


def matched_pairs(adv: np.ndarray, v: np.ndarray, strata: np.ndarray, k: int,
                  band: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Pick ``k`` high- and ``k`` low-advantage windows from matched V(s) bands.

    Args:
        adv: ``[W]`` advantage, NaN where unlabelled.
        v: ``[W]`` the critic's ``p_fail`` -- the difficulty being controlled for.
        strata: ``[W]`` stratum id; a pair never crosses one.
        k: Pairs to return.
        band: Half-width of the ``p_fail`` window a pair must share.
        rng: Draw source.

    Returns:
        ``(high_idx, low_idx)``, aligned so element ``i`` of each is one matched pair.

    Raises:
        SystemExit: If no pair can be formed -- which is itself the finding, and must not be
            reported as an empty picture.
    """
    ok = np.isfinite(adv) & np.isfinite(v)
    hi_idx, lo_idx = [], []
    used: set[int] = set()
    # MANY pairs per stratum, not one. Restricting to a single phase leaves exactly one
    # stratum, and a one-pair-per-stratum loop would then return a single pair whatever --k
    # said -- the selection silently ignoring the argument that asked for it.
    for s in rng.permutation(np.unique(strata[ok])):
        m = ok & (strata == s)
        if m.sum() < 2:
            continue
        idx = np.flatnonzero(m)
        order = idx[np.argsort(adv[idx])]
        quarter = max(1, order.size // 4)
        lo_pool, hi_pool = order[:quarter], order[-quarter:]
        for h in rng.permutation(hi_pool):
            if len(hi_idx) >= k:
                break
            near = [int(x) for x in lo_pool
                    if abs(v[x] - v[h]) <= band and int(x) not in used]
            if near:
                lo = int(rng.choice(near))
                hi_idx.append(int(h)); lo_idx.append(lo)
                used.add(int(h)); used.add(lo)
        if len(hi_idx) >= k:
            break
    if not hi_idx:
        raise SystemExit(
            f"no high/low pair shares a p_fail band of +-{band}. Either the advantage is "
            "perfectly aligned with difficulty (which is what G2 tests for and would mean the "
            "label carries no chunk information), or --band is too tight. Widen it, or read "
            "the G2 gate instead -- an empty viewer is a result, not a bug.")
    return np.array(hi_idx), np.array(lo_idx)


def main() -> int:
    """Select matched chunks and serve them in viser."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollout", required=True, help="rollout stem (without .h5)")
    p.add_argument("--labels", required=True, help="advantage .npz from label_advantage.py")
    p.add_argument("--predictions", default="",
                   help="predictions.npz, for the p_fail used to MATCH pairs. Without it the "
                        "pairing cannot control for difficulty and is refused.")
    p.add_argument("--k", type=int, default=12, help="matched pairs to draw")
    p.add_argument("--band", type=float, default=0.05,
                   help="half-width of the p_fail band a matched pair must share")
    p.add_argument("--speed", type=float, default=None,
                   help="Restrict to one COMMANDED forward speed (m/s), read from the "
                        "window's own cmd_future. The recording pins one speed per run, so "
                        "this selects a run. Omitted = every speed, which mixes six "
                        "conditions that differ by 12x in success rate.")
    p.add_argument("--speed_tol", type=float, default=0.02)
    p.add_argument("--phase", type=int, default=None, choices=(0, 1, 2, 3, 4),
                   help="Restrict to one fifth of episode PROGRESS (0 = the opening, 4 = the "
                        "end). The same five bins the label stratified on. On this course the "
                        "boxes sit in the middle, so 1-2 is the approach and the decision.")
    p.add_argument("--list", action="store_true",
                   help="print the speeds and phases actually present, then exit")
    p.add_argument("--align", choices=("anchor", "world"), default="anchor")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    import h5py

    with h5py.File(a.rollout + ".h5", "r") as f:
        n = f["episode_uid"].shape[0]
        uid, ridx = f["episode_uid"][:], f["replan_index"][:]
        plan = f["fm_plan"]
        state_dim = int(plan.shape[-1])
        sl = block_slices(state_dim)["base_pose"]
        anchor_pos, anchor_quat = f["anchor_pos"][:], f["anchor_quat"][:]
        side = dict(np.load(a.labels))
        adv = join_on_window(side, uid, ridx)
        if not a.predictions:
            raise SystemExit(
                "--predictions is required: without p_fail the high/low pairs cannot be "
                "matched on difficulty, and an unmatched pair shows easy states against hard "
                "ones rather than good chunks against bad ones.")
        pr = np.load(a.predictions)
        v = join_on_window({"episode_uid": pr["episode_uid"],
                            "replan_index": pr["replan_index"],
                            # fit_value writes the failure head as `p_<variant>`.
                            "advantage": pr["p_V_priv"]}, uid, ridx)
        # Stratify the same way the label did: progress within the episode, so a pair is not
        # an episode opening against an episode end.
        _, inv = np.unique(uid, return_inverse=True)
        last = np.zeros(n, dtype=np.int64)
        np.maximum.at(last, inv, ridx)
        prog = np.where(last[inv] > 0, ridx / np.maximum(last[inv], 1), 0.0)
        strata = np.clip((prog * 5).astype(np.int64), 0, 4)
        # The commanded speed rides in the window's own cmd_future, node 0, x component.
        speed = f["cmd_future"][:, 0, 0].astype(np.float64)
        if a.list:
            lab = np.isfinite(adv)
            print("  commanded speeds present (labelled windows):")
            for sp in np.unique(np.round(speed[lab], 2)):
                m = lab & (np.abs(speed - sp) <= a.speed_tol)
                per = "  ".join(f"p{b}:{int((m & (strata == b)).sum())}" for b in range(5))
                print(f"    {sp:5.2f} m/s   n={int(m.sum()):7d}   {per}")
            return 0
        # Restrict BEFORE matching, so the pair is drawn from the condition asked for rather
        # than merely reported under it.
        keep = np.ones(n, dtype=bool)
        if a.speed is not None:
            keep &= np.abs(speed - a.speed) <= a.speed_tol
        if a.phase is not None:
            keep &= strata == a.phase
        if not keep.any():
            raise SystemExit(f"no windows at speed={a.speed} phase={a.phase}; --list shows "
                             "what the recording actually holds.")
        adv = np.where(keep, adv, np.nan)
        where = (f"speed {a.speed} m/s" if a.speed is not None else "all speeds") + \
                (f", phase {a.phase}/4" if a.phase is not None else ", all phases")
        print(f"[view] restricted to {where}: "
              f"{int((keep & np.isfinite(adv)).sum())} labelled window(s)")
        hi, lo = matched_pairs(adv, v, strata, a.k, a.band,
                               np.random.default_rng(a.seed))
        rows = np.concatenate([hi, lo])
        blocks = {int(i): plan[int(i)][:, sl].astype(np.float64) for i in rows}

    print(f"[view] {len(hi)} matched pair(s); "
          f"high adv {adv[hi].mean():+.4f} vs low {adv[lo].mean():+.4f}, "
          f"p_fail {v[hi].mean():.3f} vs {v[lo].mean():.3f} (matched to +-{a.band})")

    import viser

    server = viser.ViserServer(port=a.port)
    ident = np.array([0.0, 0.0, 0.0, 1.0])
    for name, idxs, rgb in (("high", hi, HIGH_RGB), ("low", lo, LOW_RGB)):
        for j, i in enumerate(idxs):
            if a.align == "anchor":
                pos, _ = base_pose_block_to_world(blocks[int(i)], np.zeros(3), ident)
            else:
                pos, _ = base_pose_block_to_world(blocks[int(i)], anchor_pos[int(i)],
                                                  anchor_quat[int(i)])
            server.scene.add_spline_catmull_rom(
                f"/{name}/{j}", positions=pos.astype(np.float32),
                color=rgb, line_width=3.0)
    print(f"[view] serving on http://localhost:{a.port}  "
          f"(green = high advantage, red = low, matched on p_fail; "
          f"align={a.align})")
    while True:
        import time
        time.sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(main())
