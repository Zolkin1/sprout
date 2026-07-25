# Copyright 2026 Zachary Olkin. All rights reserved.

"""Shared world-construction ("playback") for motion matching.

Both the greedy (vanilla) and the offline beam-search matchers select a *path* of DB
frames; this module turns that path into a continuous world trajectory.  Playing a frame
drives the world root by the clip's per-step local root delta (rotated into a running
world yaw) and emits the clip's joints / root orientation.  At each **cut** (a step flagged
``jumped``) the entering clip is re-anchored SE(2)+z onto the running root and an
inertialization offset (joints / orientation / root velocity / height) is captured and
decayed to zero over ``blend_time_s`` so nothing pops.  Root height is taken from the
clip's ABSOLUTE pelvis height (not integrated), so the feet cannot drift into the ground.

Keeping this in ONE place means the two matchers are compared on identical playback -- only
their frame *selection* differs.  ``play_frame_path`` is a pure function of a precomputed
``(frame_path, jumped)`` and a duck-typed ``db``; the selection loops live in the matchers.
"""

from __future__ import annotations

import math

import torch

from ..math_utils import quat_inv, quat_mul


def smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def nlerp(q0: torch.Tensor, q1: torch.Tensor, t: float) -> torch.Tensor:
    """Normalized lerp between wxyz quats (hemisphere-fixed); ~slerp for small angles."""
    if float((q0 * q1).sum()) < 0:
        q1 = -q1
    q = (1.0 - t) * q0 + t * q1
    return q / q.norm().clamp_min(1e-9)


def inertialize_seam(root_pos, root_quat, joints, seam, blend_n, *, anchored_after):
    """Ease the FREE side of a spliced seam toward the ANCHORED side over ``blend_n`` frames -- the SAME
    inertialization :func:`play_frame_path` applies at a cut (root pos + joints decayed additively, root
    quat by :func:`nlerp`), so a terrain-clip transition eases exactly like a normal flat clip transition.

    The offline terrain splice joins a played terrain segment (the "anchored" side, kept pristine on its
    treads) to a DP-generated flat segment (the "free" side).  Rather than a rigid SE(2)+z weld -- which
    pins the whole free segment to the anchored boundary's (possibly crouched) height and drops its feet
    off its own ground -- each segment keeps its OWN ground-referenced height and this decays the seam
    discontinuity, so the free side eases in over the blend while staying grounded away from the seam.

    Args:
        root_pos/root_quat/joints: the concatenated clip ``[T, ...]`` (modified copies returned).
        seam: index of the ANCHORED segment's boundary -- its FIRST frame when ``anchored_after`` (the
            anchored segment starts at ``seam``; free frames are ``< seam``), else its LAST frame is
            ``seam-1`` (anchored segment ends there; free frames are ``>= seam``).
        blend_n: blend length in frames.
        anchored_after: ``True`` if the anchored (terrain) segment is AFTER the seam.
    """
    root_pos, root_quat, joints = root_pos.clone(), root_quat.clone(), joints.clone()
    T = root_pos.shape[0]
    if anchored_after:                        # anchored starts at `seam`; ease the free TAIL (< seam)
        ref = seam
        idxs = [(seam - 1 - j, j) for j in range(min(int(blend_n), seam))]
    else:                                     # anchored ends at `seam-1`; ease the free HEAD (>= seam)
        ref = seam - 1
        idxs = [(seam + j, j) for j in range(min(int(blend_n), T - seam))]
    off_p = root_pos[ref] - root_pos[idxs[0][0]] if idxs else None
    off_j = joints[ref] - joints[idxs[0][0]] if idxs else None
    off_q = (quat_mul(root_quat[ref].unsqueeze(0), quat_inv(root_quat[idxs[0][0]].unsqueeze(0)))[0]
             if idxs else None)
    for f, j in idxs:
        d = smoothstep(1.0 - j / float(blend_n))
        root_pos[f] = root_pos[f] + off_p * d
        joints[f] = joints[f] + off_j * d
        tgt_q = quat_mul(off_q.unsqueeze(0), root_quat[f].unsqueeze(0))[0]
        root_quat[f] = nlerp(root_quat[f], tgt_q, d)
    return root_pos, root_quat, joints


def play_frame_path(frame_path, jumped, db, cfg, *, debug: bool = False):
    """Turn a selected DB-frame path into a world trajectory.

    Args:
        frame_path: length-``T`` global DB frame indices (ints / a 1-D LongTensor), one per
            output step.
        jumped: length-``T`` bools; ``True`` marks a cut (re-anchor + inertialize).
            ``jumped[0]`` must be ``False`` -- step 0 establishes the initial anchor.
        db: object exposing ``device``, ``yaw_all``, ``root_pos``, ``joints``, ``owner``,
            ``src_start`` and the methods ``_wquat(g_yaw, gi)`` / ``_local_delta(gi)``.  For
            periodic (clean) clips it also exposes ``src_len`` and ``is_periodic_frame`` so the
            loop seam is handled (a continue that jumps backward to the clip start).
        cfg: carries ``dt`` and ``blend_time_s``.
        debug: also return the vanilla-format trace.

    Returns:
        ``(root_pos[T,3], root_quat[T,4] wxyz, joints[T,nj], dbg)``.  ``dbg`` is ``None``
        unless ``debug`` (else a list of ``(step, jumped, owner, local, base_spd, out_spd)``
        tuples, identical to the vanilla ``self.dbg`` trace).
    """
    dev = db.device
    blend_n = max(1, int(round(cfg.blend_time_s / cfg.dt)))
    T = len(frame_path)
    dbg = [] if debug else None

    def rz(g, v):   # rotate xy by yaw g, keep z
        c, s = math.cos(g), math.sin(g)
        return torch.stack([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])

    g0 = int(frame_path[0])
    g_yaw = -float(db.yaw_all[g0])                      # start facing +x
    world_pos = torch.tensor([0.0, 0.0, float(db.root_pos[g0, 2])], device=dev)
    world_yaw = 0.0
    # last emitted output (for capturing the inertialization offset at a cut)
    prev_j = db.joints[g0].clone()
    prev_q = db._wquat(g_yaw, g0)
    prev_v = rz(g_yaw, db._local_delta(g0))
    off_j = torch.zeros_like(prev_j); off_v = torch.zeros(3, device=dev)
    off_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
    off_z = 0.0                                         # root-height (z) inertialization offset
    prev_wz = float(db.root_pos[g0, 2])                 # last emitted world z
    blend_k = blend_n                                   # not blending initially
    Wp, Wq, Wj = [], [], []
    _is_per = getattr(db, "is_periodic_frame", None)
    for step in range(T):
        g_abs = int(frame_path[step])
        prev_abs = int(frame_path[step - 1]) if step > 0 else g_abs
        if step > 0 and bool(jumped[step]):
            # capture offset (old output - new clip), re-anchor yaw, start the decay
            g_yaw = world_yaw - float(db.yaw_all[g_abs])
            new_q = db._wquat(g_yaw, g_abs)
            off_j = prev_j - db.joints[g_abs]
            off_v = prev_v - rz(g_yaw, db._local_delta(g_abs))
            off_q = quat_mul(prev_q.unsqueeze(0), quat_inv(new_q.unsqueeze(0)))[0]
            off_z = prev_wz - float(db.root_pos[g_abs, 2])
            blend_k = 0
        elif (step > 0 and _is_per is not None and g_abs != prev_abs + 1
              and bool(_is_per[g_abs])):
            # PERIODIC LOOP SEAM (a continue that wrapped backward to the clip start): advance
            # the running yaw by the clip's net-per-period yaw so heading continues (for turns);
            # the pose is periodic-continuous so NO inertialization and NOT a cut (jumped stays
            # False).  Position keeps advancing via the clip's per-step delta below.
            o = int(db.owner[g_abs]); first = int(db.src_start[o])
            last = first + int(db.src_len[o]) - 1
            g_yaw += float(db.yaw_all[last]) - float(db.yaw_all[first])
        base_q = db._wquat(g_yaw, g_abs)
        base_v = rz(g_yaw, db._local_delta(g_abs))
        base_j = db.joints[g_abs]
        base_z = float(db.root_pos[g_abs, 2])           # absolute (ground-referenced) height
        d = smoothstep(1.0 - blend_k / blend_n) if blend_k < blend_n else 0.0
        if blend_k < blend_n:                           # inertialize: decay the offset 1 -> 0
            out_j = base_j + off_j * d
            out_v = base_v + off_v * d
            out_q = nlerp(base_q, quat_mul(off_q.unsqueeze(0), base_q.unsqueeze(0))[0], d)
            blend_k += 1
        else:
            out_j, out_v, out_q = base_j, base_v, base_q
        # xy: velocity-integrated (command-relative, inertialized); z: the clip's ABSOLUTE
        # pelvis height (+ inertialized offset) so the root cannot drift into the ground.
        xy = world_pos[:2] + out_v[:2]
        world_z = base_z + off_z * d
        world_pos = torch.stack([xy[0], xy[1], torch.tensor(world_z, device=dev, dtype=xy.dtype)])
        world_yaw = g_yaw + float(db.yaw_all[g_abs])
        prev_wz = world_z
        Wp.append(world_pos.clone()); Wq.append(out_q); Wj.append(out_j)
        if dbg is not None:
            o = int(db.owner[g_abs]); l = g_abs - int(db.src_start[o])
            dbg.append((step, bool(jumped[step]), o, l,
                        float(base_v.norm()) / cfg.dt, float(out_v.norm()) / cfg.dt))
        prev_j, prev_q, prev_v = out_j, out_q, out_v
    return torch.stack(Wp), torch.stack(Wq), torch.stack(Wj), dbg
