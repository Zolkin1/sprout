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

from .math_utils import quat_inv, quat_mul


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
    off = seam_offset(root_pos, root_quat, joints, seam, anchored_after=anchored_after)
    return apply_seam_offset(root_pos, root_quat, joints, seam, off, blend_n,
                             anchored_after=anchored_after)


def seam_offset(root_pos, root_quat, joints, seam, *, anchored_after, welded=False):
    """The RESIDUAL at a seam -- the mismatch that should not be there, with legitimate travel removed.

    Split out of :func:`inertialize_seam` so a clip with MANY seams can capture every offset from the
    UNMODIFIED concatenation before any of them is applied.  Capturing lazily, seam by seam, makes the
    result depend on call order whenever two seams are closer than ``2 * blend_n``: the second capture
    reads frames the first ease already moved, and its window reaches back over the first seam, reopening
    it.  Harmless for a 3-segment terrain clip whose replay is long, wrong for
    ``[loop][transient][loop][transient]...``.

    **The root-position part subtracts one frame of the anchored side's own travel.**  Taking the raw
    difference between the two boundary frames treats a frame of legitimate forward motion as if it were
    a discontinuity, and easing it lands the free boundary frame exactly ON the anchored one -- the root
    stops dead for a frame.  That is what makes a spliced terrain clip report a suspiciously perfect
    ``seam_speed`` of 0.000 while actually carrying a velocity dip at every splice.

    ``welded=True`` says the placement ALREADY made the position continuous (a chained
    ``align_and_concat`` join does), in which case the correct positional residual is exactly zero and is
    returned as such.  Subtracting travel is not good enough there: it computes one step minus the NEXT
    step, which cancels only if the gait's step length is constant, and a real gait's is not -- measured,
    that injected enough noise to take the flat per-seam ratio p95 from 0.51 to 4.61.  After a GROUNDED
    placement the surface residual is real, and there the travel subtraction is what stops a frame of
    legitimate motion being eased away.

    Returns ``(off_p, off_q, off_j)``, or ``None`` if the seam has no free frames to ease.
    """
    n = root_pos.shape[0]
    ref = seam if anchored_after else seam - 1
    free = seam - 1 if anchored_after else seam
    if not (0 <= ref < n and 0 <= free < n):
        return None
    if welded:
        off_p = torch.zeros_like(root_pos[ref])
    else:
        off_p = root_pos[ref] - root_pos[free]
        # One frame of the ANCHORED side's motion, in the direction that carries it toward the seam.
        nxt = seam + 1 if anchored_after else seam - 2
        if 0 <= nxt < n:
            travel = (root_pos[nxt] - root_pos[ref]) if anchored_after else (root_pos[ref] - root_pos[nxt])
            off_p = off_p - travel
    return (off_p,
            quat_mul(root_quat[ref].unsqueeze(0), quat_inv(root_quat[free].unsqueeze(0)))[0],
            joints[ref] - joints[free])


def apply_seam_offset(root_pos, root_quat, joints, seam, off, blend_n, *, anchored_after):
    """Decay a captured seam ``off`` into the FREE side over ``blend_n`` frames (returns modified copies).

    The correction is full at the free boundary frame and fades to zero ``blend_n`` frames away, so the
    free side eases in while staying on its own ground further out.  Because :func:`seam_offset` has
    already removed one frame of legitimate travel, "full" here means the free frame lands where the
    anchored side's motion says it should -- not on top of it.
    """
    root_pos, root_quat, joints = root_pos.clone(), root_quat.clone(), joints.clone()
    if off is None or int(blend_n) < 1:
        return root_pos, root_quat, joints
    off_p, off_q, off_j = off
    t = root_pos.shape[0]
    if anchored_after:                        # anchored starts at `seam`; ease the free TAIL (< seam)
        idxs = [(seam - 1 - j, j) for j in range(min(int(blend_n), seam))]
    else:                                     # anchored ends at `seam-1`; ease the free HEAD (>= seam)
        idxs = [(seam + j, j) for j in range(min(int(blend_n), t - seam))]
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
