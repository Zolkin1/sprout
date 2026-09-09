# Copyright 2026 Zachary Olkin. All rights reserved.
"""What terrain penalty does a CLEAN box dismount actually score?

The gate that gives the progress reward its velocity and success credit is
``sin((pi/2) R^s)`` with ``R = exp(-penalty/tau)``, and ``s = 4`` -- so the gate is a
statement about how large a penalty is "bad".  That statement has never been checked against
a motion known to be good.  It matters because of a measured asymmetry: over 352 successful
box2 episodes the BEST worst-chunk penalty was 0.673 and the median 1.844, which at ``s = 4``
gates to 0.033 and ~1e-5.  Two readings fit that equally well:

1. the policy never dismounts cleanly, and the gate is correctly refusing to pay for luck; or
2. the GRADER charges a clean dismount ~0.7 anyway, in which case the gate can never pay for
   a good dismount and the velocity/success terms are dead weight on this terrain.

They are distinguished by grading a dismount that is clean BY CONSTRUCTION.  The clip
library's authored reference motions are that: they are the motions the teacher was trained
to track, generated offline against their own tile geometry, and they are what "correct" means
everywhere else in this stack.  So this script grades library windows -- not rollout plans --
through the SAME ``terrain_consistency`` call, the same collider model and the same spec the
reward uses, and reports the penalty distribution plus the gate it implies.

**It reads the CLIP's own joint order, not the policy's.** The two differ on 27 of the G1's 29
joints and FK'ing under the wrong one produces a plausible wrong skeleton -- measured
elsewhere in this stack as moving median foot speed 0.19 -> 0.72 m/s and contact coverage
43% -> 3%, i.e. it looks exactly like a real finding.  A library's ``y_des`` is in the clip
order by definition, so that is what is used, and the resolved order is printed.

Usage::

    python scripts/rsl_rl/calibrate_clean_dismount.py \\
        --clip clips/g1_mb_v5_body_mode11_decor.h5 \\
        --family family_jump_off_front --gate_sharpness 4.0
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import h5py
import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from motion_gen import features as MGF                                    # noqa: E402
from motion_gen.joint_names import G1_JOINT_NAMES                         # noqa: E402
from motion_gen.math_utils import quat_apply                              # noqa: E402
from motion_gen.plan_geometry import (                                    # noqa: E402
    CONTACT_SPEED_THRESH,
    PlanColliderModel,
    plan_collider_points,
    plan_contact,
)
from motion_gen.terrain_consistency import (                              # noqa: E402
    TerrainConsistencySpec,
    sdf_chunk_nodes,
    terrain_consistency,
)

import terrain_reward_io as trio                                          # noqa: E402


def quat_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Rotation matrices from WXYZ quaternions, built by rotating the basis vectors.

    Written as three :func:`~motion_gen.math_utils.quat_apply` calls rather than as the
    closed-form matrix on purpose: the algebra has been hand-rolled wrong in this repo
    before, and the error was finite and plausible. Applying the rotation to ``e_i`` gives
    column ``i`` by definition, so this cannot be subtly transposed -- and if it were, the
    columns would not be orthonormal, which the caller asserts.

    Args:
        quat: ``(..., 4)`` WXYZ, the order a clip library writes.

    Returns:
        ``(..., 3, 3)`` with the rotated basis vectors as COLUMNS, i.e. ``R @ v`` maps a
        body-frame vector into the world.
    """
    lead = quat.shape[:-1]
    q = quat.reshape(-1, 4)
    cols = []
    for i in range(3):
        e = torch.zeros(q.shape[0], 3, dtype=q.dtype, device=q.device)
        e[:, i] = 1.0
        cols.append(quat_apply(q, e))
    return torch.stack(cols, dim=-1).reshape(*lead, 3, 3)


def gate(penalty: np.ndarray, sharpness: float, tau: float = 1.0) -> np.ndarray:
    """The reward's terrain-consistency gate, restated exactly as the grader builds it.

    Args:
        penalty: Per-window total terrain penalty.
        sharpness: ``s``. Multiplies the penalty before gating, so ``s = 4`` gates as though
            every penalty were four times larger.
        tau: The sidecar's declared temperature, 1.0, shared with the existing reward.

    Returns:
        The gate in ``[0, 1]``.
    """
    r = np.clip(np.exp(-np.asarray(penalty, dtype=np.float64) / tau), 0.0, 1.0)
    return np.sin(0.5 * np.pi * np.power(r, float(sharpness)))


def _families(lib: h5py.File) -> dict[str, np.ndarray]:
    """The clip-family mask, derived from the TILE the clip was generated on.

    A clip library carries no ``family_code`` -- that column is written by the RECORDER, from
    this same join.  What the library has is ``terrain_id`` per clip and a ``terrain_manifest``
    whose tiles each name their generating family, so the mask is the join of the two.  Doing
    it here rather than requiring a recording is what lets this run against the authored
    motions directly, which is the whole point: a recording's windows are the generator's
    plans, and those are the thing being calibrated AGAINST.

    Args:
        lib: The open library.

    Returns:
        One per-clip bool mask per family name (``family_jump_off_front.h5`` is keyed both
        with and without the ``.h5``, since the manifest spells it with one and the FM's
        ``skip_families`` without).

    Raises:
        SystemExit: If the library has no terrain manifest, which a flat-only library
            legitimately does not -- and then there is no box to dismount from either.
    """
    if "terrain_manifest" not in lib.attrs:
        raise SystemExit(
            "this library carries no terrain_manifest, so no clip can be attributed to a "
            "family -- and a flat library has no box to dismount from. Use a terrain library.")
    man = json.loads(lib.attrs["terrain_manifest"])
    tiles = man if isinstance(man, list) else man.get("tiles", [])
    fam_of: dict[int, str] = {}
    for t in tiles:
        fam_of[int(t.get("terrain_id", -1))] = str(t.get("family", "?"))
    tid = lib["terrain_id"][...]
    out: dict[str, np.ndarray] = {}
    for name in sorted(set(fam_of.values())):
        ids = np.array([k for k, v in fam_of.items() if v == name], dtype=np.int64)
        m = np.isin(tid, ids)
        out[name] = m
        if name.endswith(".h5"):
            out[name[:-3]] = m
    return out


def _windows_from_clips(lib: h5py.File, sel: np.ndarray, *, nodes: int, stride: int,
                        max_windows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Slice each selected clip's ``y_des`` into fixed-length windows.

    The reward grades a PLAN, which is ``nodes`` future reference nodes, so a clip has to be
    cut the same way for the two numbers to be comparable at all.

    Args:
        lib: The open library.
        sel: Clip indices to use.
        nodes: Nodes per window, matching the generator's future length.
        stride: Nodes between window starts.
        max_windows: Cap, sampled without replacement if exceeded.
        seed: RNG seed for that sample.

    Returns:
        ``(rows, starts)`` -- the clip index and the node offset of each window.
    """
    lens = lib["clip_len"][...] if "clip_len" in lib else None
    rows_l, start_l, short = [], [], 0
    for c in sel:
        n = int(lens[c]) if lens is not None else int(lib["y_des"].shape[1])
        # **A clip shorter than the window is SKIPPED, not clamped to start 0.** `y_des` is
        # padded by repeating the last frame, so a window reaching past `clip_len` grades a
        # FROZEN POSE -- which is geometrically immaculate and would bias this script's whole
        # conclusion ("authored motions score ~0") toward the answer it is testing. On the
        # shipped library no jump family has a clip under 219 nodes, so this drops nothing
        # today; it exists so a shorter library cannot quietly flatter the measurement.
        if n < nodes:
            short += 1
            continue
        for s0 in range(0, n - nodes + 1, stride):
            rows_l.append(int(c)); start_l.append(int(s0))
    if short:
        print(f"  [calib] skipped {short}/{len(sel)} clips shorter than {nodes} nodes "
              "(their window would reach into the repeated-last-frame padding)")
    rows = np.asarray(rows_l, dtype=np.int64)
    starts = np.asarray(start_l, dtype=np.int64)
    if rows.size > max_windows:
        keep = np.random.default_rng(seed).choice(rows.size, size=max_windows, replace=False)
        keep.sort()
        rows, starts = rows[keep], starts[keep]
    return rows, starts


def main() -> None:
    """Grade authored dismount windows and print the penalty and the gate it implies."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", required=True, help="The clip library to read.")
    p.add_argument("--family", default="family_jump_off_front",
                   help="Which family counts as a clean dismount. The FRONT dismount, not "
                        "`family_jump_off`, which jumps BACKWARDS and is excluded from FM "
                        "training because no forward camera can see it.")
    p.add_argument("--compare_family", default="family_jump_on",
                   help="A second family for contrast, so the dismount number is read "
                        "against another authored motion rather than as an absolute.")
    p.add_argument("--nodes", type=int, default=62,
                   help="Nodes per window, matching the generator's future length.")
    p.add_argument("--stride", type=int, default=12,
                   help="Nodes between window starts; K=12 is one re-plan.")
    p.add_argument("--max_windows", type=int, default=4000)
    p.add_argument("--max_clips", type=int, default=200)
    p.add_argument("--gate_sharpness", type=float, default=4.0,
                   help="`s`, the live value. Reported at 1/2/4 regardless, since the choice "
                        "of s is exactly what this measurement informs.")
    p.add_argument("--spacing", type=float, default=None,
                   help="Collider surface-sample spacing (m); None = extreme points.")
    p.add_argument("--served_nodes", type=int, default=12)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="", help="Optional JSON output path.")
    args = p.parse_args()

    path = args.clip if os.path.isabs(args.clip) else str(MGF._REPO_ROOT / args.clip)
    spec = TerrainConsistencySpec(served_nodes=int(args.served_nodes))
    # float64 and the CLIP joint order, matching `_plan_model` in every respect except the
    # order. `_plan_model` REFUSES the clip order, because a recorded PLAN is in the policy's
    # and the two differ on 27 of the G1's 29 joints; here the input is a library's `y_des`,
    # which is in the clip order by definition, so the correct order is the one that function
    # rejects. Stated because the asymmetry looks like an oversight and is not.
    model = PlanColliderModel(MGF.DEFAULT_MJCF, list(G1_JOINT_NAMES), dtype=torch.float64,
                              spacing=args.spacing).to(args.device)
    print(f"[calib] collider model: {model.num_points} points over "
          f"{len(model.body_order)} bodies, {model.total_area_m2:.4f} m^2 "
          f"({'surface samples' if args.spacing else 'extreme points'}), "
          f"{model.sole_index.shape[1]} sole points per foot")
    print(f"[calib] joint order: CLIP order ({len(G1_JOINT_NAMES)} joints), "
          f"first three {G1_JOINT_NAMES[:3]}")

    table = trio.load_tile_table(path)
    report: dict = {"clip": args.clip, "gate_sharpness": float(args.gate_sharpness),
                    "families": {}}
    with h5py.File(path, "r") as lib:
        fams = _families(lib)
        if not fams:
            raise SystemExit(
                f"{args.clip} carries no family table, so a dismount cannot be selected. A "
                "flat-only library legitimately has none; use a terrain library.")
        ttc = lib["terrain_type_code"][...]
        tid = lib["terrain_id"][...]
        for fam in (args.family, args.compare_family):
            if fam not in fams:
                print(f"[calib] SKIP {fam}: absent. Present: {sorted(fams)}")
                continue
            # Terrain clips only: a flat clip has no box to dismount from, and its penalty
            # would be a floor rather than a dismount reading.
            sel = np.flatnonzero(fams[fam] & (ttc != 0))[:int(args.max_clips)]
            if sel.size == 0:
                print(f"[calib] SKIP {fam}: no terrain clips in it")
                continue
            rows, starts = _windows_from_clips(lib, sel, nodes=int(args.nodes),
                                               stride=int(args.stride),
                                               max_windows=int(args.max_windows),
                                               seed=int(args.seed))
            if rows.size == 0:
                print(f"[calib] SKIP {fam}: no windows of {args.nodes} nodes fit")
                continue
            # `y_des` is (clips, T, D) over NAMED channels. Read the base pose and the
            # joint angles by name -- the clip's own order, which is `model`'s here.
            pn = json.loads(lib.attrs["pos_output_names"])
            i_pos = [pn.index(f"pelvis_link:pos_{a}") for a in "xyz"]
            i_ori = [pn.index(f"pelvis_link:ori_{a}") for a in "wxyz"]
            i_q = [pn.index(f"joint:{j}") for j in G1_JOINT_NAMES]
            y = np.stack([lib["y_des"][int(c), int(s0):int(s0) + int(args.nodes), :]
                          for c, s0 in zip(rows, starts)], axis=0)
            yt = torch.as_tensor(y, dtype=torch.float64, device=args.device)
            base_pos = yt[..., i_pos]
            quat = yt[..., i_ori]                                   # WXYZ, as the clip writes
            base_rot = quat_to_matrix(quat)
            eye = base_rot[0, 0] @ base_rot[0, 0].T
            if not torch.allclose(eye, torch.eye(3, dtype=eye.dtype, device=eye.device),
                                  atol=1e-6):
                raise SystemExit(f"base_rot is not orthonormal: R R^T =\n{eye}")
            q = yt[..., i_q]
            pts = plan_collider_points(base_pos, base_rot, q, model)
            contact = plan_contact(pts, model, dt=float(lib.attrs["dt"]))
            trows = torch.as_tensor([table.row_of(int(tid[int(c)])) for c in rows],
                                    dtype=torch.long)
            from motion_gen.terrain.box_sdf import BoxUnionTerrain        # noqa: PLC0415
            terrain = BoxUnionTerrain(table.boxes, table.mask).index(trows).to(args.device)
            got = terrain_consistency(pts, contact, terrain, model, spec,
                                      chunk_nodes=sdf_chunk_nodes(
                                          pts.shape[0], int(pts.shape[2]),
                                          int(table.boxes.shape[1])))
            pen = got["total_penalty"].detach().cpu().numpy().astype(np.float64)
            # THE SOUNDNESS CHECK, and it is not optional: a terrain the grader cannot see
            # would also score ~0 total penalty, so a clean number is ambiguous on its own.
            # `contact_penalty` is what disambiguates -- it charges every CONTACTING body by
            # its distance to the nearest surface, so a plan floating away from the boxes
            # scores LARGE there while penetration stays 0. Near-zero contact penalty with a
            # real support fraction is the placement being right.
            def _np(k: str) -> np.ndarray:
                return got[k].detach().cpu().numpy().astype(np.float64)
            cpen, ppen = _np("contact_penalty"), _np("penetration_sum_m")
            supp, cov = _np("support_fraction"), _np("contact_coverage")
            print(f"  [check] contact_penalty median {np.median(cpen):.5f} "
                  f"p95 {np.quantile(cpen, 0.95):.5f}   "
                  f"penetration_sum_m median {np.median(ppen):.5f} "
                  f"p95 {np.quantile(ppen, 0.95):.5f}")
            print(f"  [check] support_fraction median {np.median(supp):.3f}   "
                  f"contact_coverage median {np.median(cov):.3f}"
                  + ("   <-- NO CONTACT: the placement or the contact flag is wrong, and a "
                     "0 penalty here means nothing" if np.median(cov) <= 0.0 else ""))
            q = np.quantile(pen, [0.05, 0.25, 0.5, 0.75, 0.95])
            ent = {"n_clips": int(sel.size), "n_windows": int(pen.size),
                   "penalty": {"p05": q[0], "p25": q[1], "median": q[2],
                               "p75": q[3], "p95": q[4], "min": float(pen.min()),
                               "mean": float(pen.mean())},
                   "contact_penalty_median": float(np.median(cpen)),
                   "penetration_sum_m_median": float(np.median(ppen)),
                   "support_fraction_median": float(np.median(supp)),
                   "contact_coverage_median": float(np.median(cov)),
                   "gate": {f"s{s:g}": float(np.median(gate(pen, s)))
                            for s in (1.0, 2.0, 4.0, float(args.gate_sharpness))},
                   "best_window": {"penalty": float(pen.min()),
                                   f"gate_s{args.gate_sharpness:g}":
                                       float(gate(np.array([pen.min()]),
                                                  args.gate_sharpness)[0])}}
            report["families"][fam] = ent
            print(f"\n=== {fam}: {sel.size} clips, {pen.size} windows ===")
            print(f"  penalty   min {pen.min():.3f}  p05 {q[0]:.3f}  median {q[2]:.3f}  "
                  f"p95 {q[4]:.3f}")
            print(f"  gate      s=1 {np.median(gate(pen, 1.0)):.4f}   "
                  f"s=2 {np.median(gate(pen, 2.0)):.4f}   "
                  f"s=4 {np.median(gate(pen, 4.0)):.4f}   (at the MEDIAN penalty)")
            print(f"  BEST window: penalty {pen.min():.3f} -> "
                  f"gate {gate(np.array([pen.min()]), args.gate_sharpness)[0]:.4f} "
                  f"at s={args.gate_sharpness:g}")

    # The reading. 0.673 is the best worst-chunk penalty over 352 successful box2 episodes.
    ref = 0.673
    print(f"\n=== the comparison this exists for ===")
    print(f"  best POLICY worst-chunk penalty on box2 (352 successes): {ref:.3f} "
          f"-> gate {gate(np.array([ref]), args.gate_sharpness)[0]:.4f}")
    for fam, ent in report["families"].items():
        m = ent["penalty"]["median"]
        verdict = ("the grader charges an AUTHORED motion as much as the policy -- the gate "
                   "cannot pay for a clean dismount"
                   if m >= ref else
                   "an authored motion scores CLEANER than anything the policy managed -- "
                   "the gate has room and the policy is the limit")
        print(f"  {fam}: authored median {m:.3f} vs policy best {ref:.3f} -> {verdict}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[calib] wrote {args.out}")


if __name__ == "__main__":
    main()
