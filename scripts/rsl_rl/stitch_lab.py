#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Pick a terrain hand-off frame by hand and see the clip it actually produces.

The seam between the flat approach and the terrain replay is chosen by a gate
(:func:`terrain_families.boundary_candidates`) that tests SOLE HEIGHT and swing clearance and says
nothing about foot ORIENTATION -- so two poses can both pass while their toes sit 30 mm apart in z,
and the ``ease`` blend then interpolates between them with no knowledge of the ground. That is a
thing you have to LOOK at, and until now the only way to look was to build ten thousand clips and
open one.

This is the small loop. Stage ``sweep`` plans the same tile once per candidate hand-off frame,
in-process, and records what each one would hand off (that is seconds, not a build). Stage ``view``
scrubs them in viser: the tile, the robot at the hand-off, the gate's verdict, and the foot geometry
the gate does not look at. ``Generate`` then runs THE REAL PIPELINE for the frame you picked --
pass 1, Motion Bricks, pass 2 -- so what you see is the clip, not a preview of one.

**You may pick a frame the gate REJECTED.** That is deliberate and is most of the value: the gate's
own candidates are often a single frame (measured on a 2-riser tile: `entry_candidates == [17]`),
so "why not this one" is the question worth answering, and the answer is a rendered seam.

**The two candidate arrays live in different index spaces**, which is the one thing to keep straight:
``entry_candidates`` index the SOURCE PERIOD of the reference before placement (choosing one
re-anchors the whole replay so that frame becomes frame 0), while ``exit_candidates`` index the
PLACED replay. This tool sweeps the former; the slider is a source-period frame.

Run::

    # 1. sweep every hand-off frame for one tile spec (~1 s per frame, no GPU)
    python scripts/rsl_rl/stitch_lab.py --stage sweep --out /tmp/stitch \\
        --skill stairs_up --riser 0.15 --tread 0.25 --n-steps 2 --terrain-speed 0.69 \\
        --source hf/trajectories/retargeted/2026-06-13_15-10-58_merged

    # 2. look, pick, generate
    python scripts/rsl_rl/stitch_lab.py --stage view --out /tmp/stitch \\
        --stride-bank mb_raw/lib_v6f/stride_bank.npz \\
        --mb-root /home/zolkin/AmberLab/Project-Isaac-RL/GR00T-WholeBodyControl/motionbricks
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

from motion_gen.joint_names import G1_JOINT_NAMES
from motion_gen.terrain.footprint import FOOT_BOX_CENTER, FOOT_BOX_HALF

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))


# --------------------------------------------------------------------------------- geometry


def tile_boxes(meta: dict) -> list[dict]:
    """The plan's own tile boxes as ``{pos_xy, size_xyz, base_z}`` dicts.

    ``meta["boxes"]`` is centre + FULL extents (``{pos: [x,y,z], size: [sx,sy,sz]}``); every viewer
    in this repo consumes the overlay schema instead, so convert once here rather than teaching the
    renderers a second shape.
    """
    out = []
    for b in meta.get("boxes", []):
        (px, py, pz), (sx, sy, sz) = b["pos"], b["size"]
        out.append({"pos_xy": [float(px), float(py)], "size_xyz": [float(sx), float(sy), float(sz)],
                    "base_z": float(pz) - float(sz) / 2.0})
    return out


def surface_under(boxes: list[dict], x: float, y: float) -> float:
    """Height of the highest box top covering ``(x, y)``, or 0.0 for the tile floor."""
    z = 0.0
    for b in boxes:
        px, py = b["pos_xy"]
        sx, sy, sz = b["size_xyz"]
        if abs(x - px) <= sx / 2 + 1e-9 and abs(y - py) <= sy / 2 + 1e-9:
            z = max(z, b["base_z"] + sz)
    return z


ANKLES = ("left_ankle_roll_link", "right_ankle_roll_link")


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    """``(w, x, y, z)`` -> a 3x3 rotation matrix."""
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _corners(pos_w: np.ndarray, R_w: np.ndarray) -> np.ndarray:
    """The foot box's 8 world corners from a world ankle pose, ``(8, 3)``.

    Built from the rotation MATRIX rather than round-tripping through a quaternion, but from the
    same `FOOT_BOX_CENTER` / `FOOT_BOX_HALF` constants `terrain.footprint` uses -- one foot model,
    so this tool and the tile sizing cannot disagree about where the foot is.
    """
    cx, cy, cz = FOOT_BOX_CENTER
    hx, hy, hz = FOOT_BOX_HALF
    local = np.array([[cx + sx * hx, cy + sy * hy, cz + sz * hz]
                      for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])
    return pos_w[None, :] + local @ R_w.T


def foot_report(qpos_row: np.ndarray, boxes: list[dict], fk) -> dict:
    """What the gate does NOT look at, for one MuJoCo-layout ``qpos`` row.

    Returns each foot's lowest corner height, its PITCH, and the deepest corner relative to the
    surface under that corner. The gate only ever compares a SOLE HEIGHT against a tread within
    ``contact_tol`` and a swing foot against a clearance band -- nothing in it constrains foot
    orientation, which is why two frames it both accepts can hand off very differently.
    """
    root_p, root_q, joints = qpos_row[0:3], qpos_row[3:7], qpos_row[7:]
    R_root = _quat_to_R(root_q)
    pos_b, rot_b = fk(joints)                     # base-frame (K, 3) and (K, 3, 3)
    out: dict = {}
    worst = 9.9
    for i, side in enumerate(("left", "right")):
        p_w = root_p + R_root @ pos_b[i]
        R_w = R_root @ rot_b[i]
        c = _corners(p_w, R_w)
        clr = min(float(c[k, 2]) - surface_under(boxes, float(c[k, 0]), float(c[k, 1]))
                  for k in range(8))
        out[side] = {"sole_z": float(c[:, 2].min()),
                     "pitch_deg": float(np.degrees(np.arcsin(np.clip(-R_w[2, 0], -1.0, 1.0)))),
                     "clearance": clr}
        worst = min(worst, clr)
    out["worst_clearance"] = worst
    return out


# --------------------------------------------------------------------------------- stage: sweep


def stage_sweep(args) -> int:
    """Plan the tile once per source-period frame and record what each hand-off would give."""
    sys.path.insert(0, _HERE)
    import torch
    from mb_terrain_clips import ingest_family_sources, make_builder, plan_from_spec
    from motion_gen.motionbricks.terrain_plan import save_terrain_plan
    from motion_gen.features import FKChain, DEFAULT_MJCF

    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    sources = ingest_family_sources(args.skill, args, dev)
    builder = make_builder(args, dev)
    mjcf = DEFAULT_MJCF if os.path.isabs(DEFAULT_MJCF) else os.path.join(_REPO, DEFAULT_MJCF)
    chain = FKChain(mjcf, list(G1_JOINT_NAMES), list(ANKLES), base_link="pelvis_link")

    def fk(joints: np.ndarray):
        """Base-frame ankle ``(pos[K,3], rot[K,3,3])`` for one joint vector."""
        jt = torch.as_tensor(np.asarray(joints, dtype=np.float32)[None, :])
        pos, rot, _, _ = chain.pose_and_jacobian(jt)
        return pos[0].numpy(), rot[0].numpy()

    # One unforced plan first: it tells us the source period and the gate's own candidates.
    base = plan_from_spec(sources, args, dev, builder)
    meta = base.meta if isinstance(base.meta, dict) else json.loads(base.meta)
    period = int(meta.get("source_period", 0)) or len(base.terrain_qpos)
    cands = [int(v) for v in np.asarray(base.entry_candidates).tolist()]
    print(f"[stitch] source period {period} frames; gate candidates {cands}")

    rows = []
    for k in range(period):
        args.entry_frame = k
        try:
            plan = plan_from_spec(sources, args, dev, builder)
        except SystemExit as e:                       # a forced frame the pipeline cannot place
            rows.append({"k": k, "ok": False, "why": str(e), "is_candidate": k in cands})
            print(f"[stitch]   k={k:3d}  REFUSED: {e}")
            continue
        path = os.path.join(args.out, f"plan_k{k:03d}.npz")
        save_terrain_plan(path, plan)
        m = plan.meta if isinstance(plan.meta, dict) else json.loads(plan.meta)
        boxes = tile_boxes(m)
        q = np.asarray(plan.terrain_qpos)
        rep = foot_report(q[0], boxes, fk)             # frame 0 IS the hand-off pose
        worst_all = min(foot_report(q[t], boxes, fk)["worst_clearance"]
                        for t in range(0, len(q), max(1, len(q) // 24)))
        rows.append({"k": k, "ok": True, "is_candidate": k in cands, "plan": path,
                     "n_frames": int(len(q)), "entry": rep, "replay_worst": worst_all,
                     "exit_frame": int(m.get("exit_frame", -1)) if "exit_frame" in m else None})
        flag = "cand" if k in cands else "    "
        print(f"[stitch]   k={k:3d} {flag}  hand-off clearance {rep['worst_clearance']*1000:+7.1f} mm"
              f"  pitch L/R {rep['left']['pitch_deg']:+6.1f}/{rep['right']['pitch_deg']:+6.1f} deg"
              f"  replay worst {worst_all*1000:+7.1f} mm")

    meta0 = base.meta if isinstance(base.meta, dict) else json.loads(base.meta)
    with open(os.path.join(args.out, "sweep.json"), "w") as f:
        json.dump({"skill": args.skill, "period": period, "candidates": cands,
                   "boxes": tile_boxes(meta0), "rows": rows,
                   "spec": {k: getattr(args, k) for k in
                            ("riser", "tread", "n_steps", "width", "terrain_speed",
                             "approach_vx", "exit_vx", "approach_dist", "exit_dist",
                             "source", "library", "skill")}}, f, indent=1)
    ok = sum(1 for r in rows if r["ok"])
    print(f"[stitch] wrote {args.out}/sweep.json -- {ok}/{period} frames plannable, "
          f"{len(cands)} of them gate candidates")
    return 0


# --------------------------------------------------------------------------------- generation


def generate(out_dir: str, k: int, spec: dict, args, tokens=0,
             free_root: bool = False, exit_frame: int | None = None) -> str | None:
    """Run THE REAL pipeline for hand-off frame ``k`` -> a one-clip ``.h5``, or None on failure.

    Three subprocesses, exactly as a library build does them: pass 1 (this env) re-plans with the
    frame forced, Motion Bricks (the ``motionbricks`` conda env) generates the two raw segments,
    pass 2 (this env) splices them around the replay. Nothing here reimplements any of it -- that
    is the whole point, since a preview that agreed with the pipeline only approximately would be
    worse than no preview.
    """
    # The horizon is part of the identity: the whole point is comparing the SAME hand-off at
    # several token counts, so they must not overwrite each other.
    _tag = (f"_t{tokens[0]:02d}-{tokens[1]:02d}" if isinstance(tokens, (tuple, list))
            else (f"_t{tokens:02d}" if tokens else ""))
    work = os.path.join(out_dir, f"gen_k{k:03d}" + _tag
                        + ("_free" if free_root else "")
                        + (f"_x{exit_frame:03d}" if exit_frame is not None else ""))
    os.makedirs(work, exist_ok=True)
    plan = os.path.join(work, "plan.npz")
    clip = os.path.join(work, "clip.h5")
    py = sys.executable
    env = dict(os.environ, PYTHONPATH=f"{_REPO}/source/robot_rl:{_REPO}/source/motion_gen")
    tc = os.path.join(_HERE, "mb_terrain_clips.py")

    common = [py, tc, "--skill", spec["skill"], "--source", spec["source"],
              "--library", spec["library"], "--riser", str(spec["riser"]),
              "--tread", str(spec["tread"]), "--n-steps", str(spec["n_steps"]),
              "--width", str(spec["width"]), "--terrain-speed", str(spec["terrain_speed"]),
              "--approach-vx", str(spec["approach_vx"]), "--exit-vx", str(spec["exit_vx"]),
              "--approach-dist", str(spec["approach_dist"]), "--exit-dist", str(spec["exit_dist"])]
    steps = [
        ("pass 1", common + ["--entry-frame", str(k), "--dump-plan", plan]
                  + ([] if exit_frame is None else ["--exit-frame-override", str(exit_frame)])),
        ("motion bricks", ["conda", "run", "--no-capture-output", "-n", args.conda_env, "python",
                           "-u", os.path.join(_HERE, "mb_terrain_segments.py"),
                           "--mb-root", args.mb_root, "--stride-bank", os.path.abspath(args.stride_bank),
                           "--plan", os.path.abspath(plan), "--out-dir", os.path.abspath(work),
                           "--side", "both", "--gen-attempts", "6", "--pin-future", "auto"]
                          + (["--token-band", str(tokens[0]), str(tokens[1])]
                             if isinstance(tokens, (tuple, list))
                             else (["--token-band", str(tokens), str(tokens)] if tokens else []))
                          + (["--free-root"] if free_root else [])),
        ("pass 2", common + ["--plan", plan, "--approach-raw", os.path.join(work, "approach.npz"),
                             "--exit-raw", os.path.join(work, "exit.npz"),
                             "--stride-bank", args.stride_bank, "--out", clip]),
    ]
    for name, cmd in steps:
        print(f"[stitch] {name} ...", flush=True)
        r = subprocess.run(cmd, cwd=_REPO, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            tail = (r.stderr or r.stdout).strip().splitlines()[-6:]
            print(f"[stitch] {name} FAILED:\n   " + "\n   ".join(tail))
            return None
    return clip


# --------------------------------------------------------------------------------- stage: view


def _clip_poses(path: str):
    """``(pos[T,3], wxyz[T,4], joints[T,29], overlays, anchor_mask[T])`` from a one-clip ``.h5``.

    The clip's own manifest is used, NOT the plan's boxes: the spliced clip is authored in the
    TILE's frame with its footprint margin and approach platform, so its overlays sit at x 4.95-6.45
    where the plan's sit at 0.5-2.0. Drawing the plan's tile under the clip's motion would put the
    robot metres off the stairs and look like a placement bug that is not there.
    Orientation is stored xyzw and must be flipped for viser.
    """
    import h5py
    from robot_rl.flow_matching.viz.robot import xyzw_to_wxyz
    with h5py.File(path, "r") as f:
        names = json.loads(f.attrs["pos_output_names"])
        m = int(f["clip_len"][0])
        y = np.asarray(f["y_des"][0, :m, :])
        am = np.asarray(f["anchor_mask_t"][0, :m])
        si = np.asarray(f["stride_idx"][0, :m])
        man = json.loads(f.attrs["terrain_manifest"])
    pos = y[:, [names.index(f"pelvis_link:pos_{c}") for c in "xyz"]]
    wxyz = xyzw_to_wxyz(y[:, [names.index(f"pelvis_link:ori_{c}") for c in "xyzw"]])
    jidx = [i for i, n in enumerate(names) if n.startswith("joint:")]
    jnames = [names[i].split(":", 1)[1] for i in jidx]
    overlays = man[0].get("overlays", []) if man else []
    ank = {sd: y[:, [names.index(f"{sd}_ankle_roll_link:pos_{c}") for c in "xyz"]]
           for sd in ("left", "right")}
    return pos, wxyz, y[:, jidx], jnames, overlays, am, si, ank


def segment_provenance(clip_path: str) -> dict:
    """The APPROACH segment's own record of the token band it was given, beside its clip.

    Read rather than recomputed: `hop_tokens` needs `mb.nfpt` / `mb.min_tokens` / `mb.max_tokens`,
    which live on the Motion Bricks inferencer and exist only inside the conda env. The generator
    writes what it actually used, so this is the measurement and not a reconstruction of one.
    """
    path = os.path.join(os.path.dirname(clip_path), "approach.npz")
    if not os.path.exists(path):
        return {}
    d = np.load(path, allow_pickle=True)
    if "meta" not in d.files:
        return {}
    return (json.loads(str(d["meta"].item())) or {}).get("provenance", {})


def phase_spans(am: np.ndarray, si: np.ndarray) -> dict:
    """Frame spans of the four things a terrain clip is made of, in clip order."""
    n = len(am)
    ter = np.flatnonzero(am == 2)
    a, b = (int(ter[0]), int(ter[-1])) if len(ter) else (n, n)
    before, after = np.arange(n) < a, np.arange(n) > b
    tr_in = np.flatnonzero(before & (si < 0))
    tr_out = np.flatnonzero(after & (si < 0))
    return {"loops_in": np.flatnonzero(before & (si >= 0)), "transient_in": tr_in,
            "replay": np.arange(a, b + 1), "transient_out": tr_out,
            "loops_out": np.flatnonzero(after & (si >= 0)), "seam": a}


def seam_mechanics(sp: dict, prov: dict, spec: dict, bank_period: int | None) -> str:
    """Why the transient is the length it is, and whether it could have taken another step.

    The transient is the ONLY place the approach can reconcile itself with the hand-off pose, and
    `hop_tokens` sizes it from the VELOCITY change alone -- `t_min = max(floor_s, |dv|/a_max)`, then
    a `slack` ceiling. When the speed change is small the floor is what bites, so the window is short
    regardless of how much reconciling the pose actually needs. If the hand-off wants the other foot
    forward, a half gait cycle has to happen inside that window or the foot gets dragged there.
    """
    dt = 0.02
    n_tr = len(sp["transient_in"])
    lines = [f"**transient** frames {sp['transient_in'][0] if n_tr else '-'}"
             f"–{sp['transient_in'][-1] if n_tr else '-'} "
             f"({n_tr} frames, {n_tr * dt:.2f} s) → seam at **{sp['seam']}**"]
    if prov:
        nmin, nmax = prov.get("n_min"), prov.get("n_max")
        tok, mx = prov.get("tokens"), prov.get("max_tokens")
        nfpt, fps = prov.get("nfpt"), prov.get("fps", 30.0)
        choice = "**no choice at all**" if nmin == nmax else f"a {nmax - nmin}-token window"
        lines.append(f"**token band** `[{nmin}, {nmax}]`, took {tok}"
                     + (f" of a {mx} ceiling" if mx else " (ceiling not recorded — regenerate)")
                     + f" — {choice}")
        if mx and nfpt:
            head = (mx - (nmax or 0)) * nfpt / float(fps)
            lines.append(f"**headroom to the ceiling** {head:.2f} s — "
                         + ("unused" if head > 0.05 else "none, the ceiling is binding"))
        lines.append(f"sized by `t_min = max(floor {prov.get('min_transient_s')} s, "
                     f"|dv|/a_max)`, slack {prov.get('hop_slack')}")
    if bank_period:
        half = bank_period * dt / 2.0
        verdict = ("it **can** fit one" if n_tr * dt >= half
                   else f"it **cannot** — short by {half - n_tr * dt:.2f} s")
        lines.append(f"**a half gait step** at this speed costs {bank_period // 2} frames "
                     f"({half:.2f} s); the transient has {n_tr} ({n_tr * dt:.2f} s), so {verdict}")
    lines.append("*Grey = exact loops, red = the transient, green = the replay. Watch the ankle "
                 "paths: a foot that doubles back is the reconciliation happening inside a window "
                 "too short for a step.*")
    return "\n\n".join(lines)


def stage_view(args) -> int:
    """Scrub the hand-off frames, generate the one you pick, and see the CLIP in the same scene."""
    import viser
    from viser.extras import ViserUrdf
    import trimesh
    sys.path.insert(0, _HERE)
    from view_terrain_probe_viser import box_mesh
    from robot_rl.flow_matching.viz.robot import DEFAULT_URDF, load_urdf, recording_to_urdf_perm

    with open(os.path.join(args.out, "sweep.json")) as f:
        sweep = json.load(f)
    rows = {int(r["k"]): r for r in sweep["rows"]}
    cands = set(sweep["candidates"])

    urdf = load_urdf(os.path.join(_REPO, DEFAULT_URDF))
    perm_qpos = recording_to_urdf_perm(list(G1_JOINT_NAMES), urdf)

    server = viser.ViserServer(port=args.port)
    root = server.scene.add_frame("/robot", show_axes=False)
    vurdf = ViserUrdf(server, urdf, root_node_name="/robot")

    gui = server.gui
    plannable = sorted(k for k, r in rows.items() if r["ok"])
    k_sl = gui.add_slider("Hand-off frame k", min=min(plannable), max=max(plannable), step=1,
                          initial_value=(sorted(cands)[0] if cands else plannable[0]))
    src = gui.add_dropdown("Showing", ("terrain replay", "generated clip"),
                           initial_value="terrain replay")
    t_sl = gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
    seam_btn = gui.add_button("Jump to the entry seam")
    info = gui.add_markdown("")
    tok_sl = gui.add_slider("Horizon (tokens; 0 = pipeline default)", min=0, max=16, step=1,
                            initial_value=0)
    free_cb = gui.add_checkbox("Free root (MB owns timing + root motion)", False)
    gen_btn = gui.add_button("Generate this clip (real pipeline)")
    gen_out = gui.add_markdown("")

    # `poses` is whatever is currently drawn; `perm` differs between the two sources because a
    # clip stores its own joint order while a plan's qpos is G1_JOINT_NAMES.
    st: dict = {"poses": None, "perm": perm_qpos, "seam": 0, "clip": {}, "mesh": None,
                "paths": []}

    # The gait period of the stride nearest this clip's approach speed -- what a half step COSTS.
    bank_period = None
    try:
        bk = np.load(args.stride_bank, allow_pickle=True)
        sv, off = bk["steady_vel"], np.asarray(bk["offsets"])
        v = abs(float(sweep["spec"]["approach_vx"]))
        i = int(np.argmin(np.abs(sv[:, 0] - v) + 3 * np.abs(sv[:, 1]) + 3 * np.abs(sv[:, 2])))
        bank_period = int(off[i + 1] - off[i])
    except Exception as e:                      # a missing bank costs the half-step line, nothing else
        print(f"[stitch] no stride bank ({e}); the half-step comparison will be omitted")

    def draw_paths(pelvis, ank, sp) -> None:
        """Pelvis + both ankle tracks, coloured by which piece of the clip made them.

        The ankle tracks are the point: a foot doubling back on itself is the transient
        reconciling a hand-off it does not have room for, and it is invisible in the pelvis path.
        """
        for h in st["paths"]:
            h.remove()
        st["paths"] = []
        colours = {"loops_in": (150, 150, 150), "transient_in": (220, 60, 60),
                   "replay": (60, 180, 90), "transient_out": (220, 140, 60),
                   "loops_out": (150, 150, 150)}
        for name, track in (("pelvis", pelvis), ("left", ank["left"]), ("right", ank["right"])):
            for phase, colour in colours.items():
                idx = sp[phase]
                if len(idx) == 0:
                    continue
                st["paths"].append(server.scene.add_point_cloud(
                    f"/path/{name}/{phase}", points=np.asarray(track)[idx].astype(np.float32),
                    colors=np.tile(np.array(colour, np.uint8), (len(idx), 1)),
                    point_size=0.012 if name == "pelvis" else 0.008))

    def set_tile(boxes: list[dict]) -> None:
        if st["mesh"] is not None:
            st["mesh"].remove()
        st["mesh"] = (server.scene.add_mesh_trimesh(
            "/tile", trimesh.util.concatenate([box_mesh(b) for b in boxes])) if boxes else None)

    def draw(t: int) -> None:
        p = st["poses"]
        if p is None:
            return
        pos, wxyz, cfg = p
        t = int(np.clip(t, 0, len(pos) - 1))
        root.position = tuple(float(v) for v in pos[t])
        root.wxyz = tuple(float(v) for v in wxyz[t])
        vurdf.update_cfg(np.asarray(cfg[t])[st["perm"]])

    def clip_path(k: int, tokens: int | None = None) -> str:
        t = int(tok_sl.value) if tokens is None else int(tokens)
        return os.path.join(args.out, f"gen_k{k:03d}" + (f"_t{t:02d}" if t else "")
                            + ("_free" if free_cb.value else ""), "clip.h5")

    def show_replay(k: int) -> None:
        r = rows.get(int(k))
        if r is None or not r["ok"]:
            info.content = f"### k = {k}\n\n**REFUSED** — {r['why'] if r else 'not swept'}"
            return
        q = np.asarray(np.load(r["plan"], allow_pickle=True)["terrain_qpos"])
        st["poses"] = (q[:, 0:3], q[:, 3:7], q[:, 7:])
        st["perm"], st["seam"] = perm_qpos, 0
        for h in st["paths"]:
            h.remove()
        st["paths"] = []
        set_tile(sweep["boxes"])
        t_sl.max = len(q) - 1
        t_sl.value = 0
        e = r["entry"]
        verdict = "**GATE CANDIDATE**" if int(k) in cands else "**NOT a gate candidate** (unvetted)"
        have = " · *generated clip available — switch `Showing`*" if os.path.exists(clip_path(k)) else ""
        info.content = (
            f"### k = {k} — terrain replay — {verdict}{have}\n\n"
            f"| | left | right |\n|---|---|---|\n"
            f"| lowest corner z | {e['left']['sole_z']:.3f} | {e['right']['sole_z']:.3f} |\n"
            f"| foot pitch | {e['left']['pitch_deg']:+.1f}° | {e['right']['pitch_deg']:+.1f}° |\n"
            f"| corner clearance | {e['left']['clearance']*1000:+.1f} mm "
            f"| {e['right']['clearance']*1000:+.1f} mm |\n\n"
            f"hand-off worst **{e['worst_clearance']*1000:+.1f} mm** · "
            f"replay worst **{r['replay_worst']*1000:+.1f} mm** · {r['n_frames']} frames\n\n"
            f"*Pitch and clearance are what the gate does NOT test.*")
        draw(0)

    def show_clip(k: int) -> None:
        path = clip_path(k)
        if not os.path.exists(path):
            info.content = f"### k = {k}\n\nNo generated clip yet — press **Generate**."
            return
        key = (k, int(tok_sl.value), bool(free_cb.value))
        if key not in st["clip"]:
            pos, wxyz, joints, jnames, overlays, am, si, ank = _clip_poses(path)
            st["clip"][key] = (pos, wxyz, joints, recording_to_urdf_perm(jnames, urdf),
                               overlays, am, si, ank, segment_provenance(path))
        pos, wxyz, joints, perm, overlays, am, si, ank, prov = st["clip"][key]
        st["poses"], st["perm"] = (pos, wxyz, joints), perm
        set_tile(overlays)
        t_sl.max = len(pos) - 1
        sp = phase_spans(am, si)
        st["seam"] = int(sp["seam"])
        t_sl.value = st["seam"]
        draw_paths(pos, ank, sp)
        forced = int(tok_sl.value)
        info.content = (f"### k = {k}"
                        + (f", horizon FORCED to {forced} tokens" if forced else ", default horizon")
                        + (" · FREE ROOT" if free_cb.value else "")
                        + f" — GENERATED CLIP, {len(pos)} frames\n\n"
                        + seam_mechanics(sp, prov, sweep["spec"], bank_period)
                        + f"\n\n`{path}`")
        draw(t_sl.value)

    def refresh() -> None:
        (show_clip if src.value == "generated clip" else show_replay)(int(k_sl.value))

    @k_sl.on_update
    def _(_e) -> None:
        refresh()

    @src.on_update
    def _(_e) -> None:
        refresh()

    @t_sl.on_update
    def _(_e) -> None:
        draw(int(t_sl.value))

    @tok_sl.on_update
    def _(_e) -> None:
        if src.value == "generated clip":
            refresh()

    @free_cb.on_update
    def _(_e) -> None:
        if src.value == "generated clip":
            refresh()

    @seam_btn.on_click
    def _(_e) -> None:
        t_sl.value = st["seam"]

    @gen_btn.on_click
    def _(_e) -> None:
        k = int(k_sl.value)
        gen_btn.disabled = True
        gen_out.content = f"generating k = {k} … (pass 1 → Motion Bricks → pass 2, ~30 s)"
        try:
            path = generate(args.out, k, sweep["spec"], args, tokens=int(tok_sl.value),
                            free_root=bool(free_cb.value))
        finally:
            gen_btn.disabled = False
        if not path:
            gen_out.content = "**generation FAILED** — see the console"
            return
        st["clip"].pop((k, int(tok_sl.value), bool(free_cb.value)), None)
        gen_out.content = f"**done** — `{path}`"
        src.value = "generated clip"            # fires on_update -> show_clip
        refresh()

    refresh()
    print(f"[stitch] viser on http://localhost:{args.port}  "
          f"({len(plannable)} plannable frames, {len(cands)} gate candidates)", flush=True)
    import time
    while True:
        time.sleep(1.0)


# --------------------------------------------------------------------------------- cli


def main() -> int:
    """Parse args and run a stage."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=["sweep", "view"])
    p.add_argument("--out", required=True, help="Working directory for plans and generated clips.")
    p.add_argument("--port", type=int, default=8080)
    # sweep: the tile spec, mirroring mb_terrain_clips.py's own flags
    p.add_argument("--skill", default="stairs_up")
    p.add_argument("--source", default="hf/trajectories/retargeted/2026-06-13_15-10-58_merged")
    p.add_argument("--library", default="trajectories/skills/library_v2")
    p.add_argument("--hf-repo", default="zolkin/robot_rl")
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--device", default="cpu")
    p.add_argument("--riser", type=float, default=0.15)
    p.add_argument("--tread", type=float, default=0.25)
    p.add_argument("--n-steps", type=int, default=2)
    p.add_argument("--width", type=float, default=1.4)
    p.add_argument("--nav-width", type=float, default=None)
    p.add_argument("--terrain-speed", type=float, default=0.69)
    p.add_argument("--approach-vx", type=float, default=0.8)
    p.add_argument("--exit-vx", type=float, default=0.8)
    p.add_argument("--approach-dist", type=float, default=4.0)
    p.add_argument("--exit-dist", type=float, default=4.0)
    # generation
    p.add_argument("--stride-bank", default="mb_raw/lib_v6f/stride_bank.npz")
    p.add_argument("--mb-root",
                   default="/home/zolkin/AmberLab/Project-Isaac-RL/GR00T-WholeBodyControl/motionbricks",
                   help="ABSOLUTE: the default is relative in the batch driver and wrong from a worktree.")
    p.add_argument("--conda-env", default="motionbricks")
    args = p.parse_args()

    # mb_terrain_clips reads a wider namespace than this CLI declares; fill its defaults once here
    # rather than restating each flag, so a knob added there does not silently read as missing.
    for k, v in {"entry_pick": "first", "exit_pick": "first", "contact_tol": 0.02,
                 "swing_clear": None, "swing_band": None, "terrain_id": 0,
                 "footprint_margin": 1.0, "has_floor": True, "stair_float_thickness": 0.0,
                 "box_height": 0.45, "box_platform_buffer": 1.0, "box_nav_width": 0.0,
                 "entry_frame": None, "exit_frame_override": None}.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    return stage_sweep(args) if args.stage == "sweep" else stage_view(args)


if __name__ == "__main__":
    raise SystemExit(main())
