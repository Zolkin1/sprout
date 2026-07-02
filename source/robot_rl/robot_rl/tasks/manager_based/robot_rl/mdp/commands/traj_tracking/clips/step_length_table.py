# Copyright 2026 Zachary Olkin. All rights reserved.

"""Per-trajectory forward swing-reach table (shared by online + offline).

The terrain-approach planner picks the walk trajectory whose step lands the
swing foot at the stair stance reference; that decision needs, per trajectory,
the forward (local +x) distance the swing foot travels over one stride.  The
online command (:class:`BatchedMultiSkillCommand`) and the offline clip
generator (:class:`OfflineClipGenerator`) both need this identical quantity,
so it lives here as one function rather than being duplicated.

The table is built from Bezier evaluation just before the first contact gate,
matching the runtime next-gate prediction (before the world transform), so the
prediction and the table stay consistent.
"""

from __future__ import annotations

import torch


def expand_ref_frames(frames: list[str]) -> list[str]:
    """Deduplicate ``frames`` and add each one's left/right mirror.

    Mirrors ``BaseTrajectoryCommand._parse_ref_frames`` (minus the robot
    body-name lookup, which needs a live env): half-periodic trajectories
    reflect ``right_*`` <-> ``left_*`` for their reflected half, so
    :meth:`MultiSkillManager.build_ref_frame_map` needs both feet present in
    the ref-frame list.  Offline callers pass ``manager.ref_frames`` (which may
    carry only the ``right`` foot, duplicated per domain).

    Args:
        frames: Per-domain reference frame names (may repeat / be one-sided).

    Returns:
        Ordered unique frame names with each foot's mirror included.
    """
    out: list[str] = []
    for f in frames:
        if f not in out:
            out.append(f)
        opp = None
        if f.startswith("right"):
            opp = "left" + f[len("right"):]
        elif f.startswith("left"):
            opp = "right" + f[len("left"):]
        if opp is not None and opp not in out:
            out.append(opp)
    return out


def build_ref_frame_to_pos_xy_idx(
    manager, ref_frames: list[str], device: str | torch.device
) -> torch.Tensor:
    """Map each ``ref_frames`` entry to its ``(pos_x, pos_y)`` columns.

    The columns index ``manager.pos_output_names`` — the same ordering
    ``manager.data["coeffs_pos"]`` (and hence ``_compute_bezier_batched``)
    produces (``order_outputs`` permutes both together).  ``ref_frames`` must be
    the SAME (expanded, both-feet) list passed to
    :meth:`MultiSkillManager.build_ref_frame_map`, so ``swing_ref`` indices from
    ``_ref_frame_map`` line up with this table.

    Args:
        manager: A loaded :class:`MultiSkillManager`.
        ref_frames: Expanded ref-frame list (see :func:`expand_ref_frames`).
        device: Torch device for the returned index tensor.

    Returns:
        ``[len(ref_frames), 2]`` long tensor of ``(pos_x, pos_y)`` columns.

    Raises:
        ValueError: A ref frame is missing ``pos_x`` or ``pos_y`` outputs.
    """
    names = list(manager.pos_output_names)
    rows: list[list[int]] = []
    for frame in ref_frames:
        try:
            rows.append([names.index(f"{frame}:pos_x"), names.index(f"{frame}:pos_y")])
        except ValueError as e:
            raise ValueError(
                f"Ref frame '{frame}' is missing pos_x or pos_y in the "
                "trajectory position outputs; step-length prediction needs both."
            ) from e
    return torch.tensor(rows, dtype=torch.long, device=device)


def traj_step_length_table(
    manager,
    ref_frame_to_pos_xy_idx: torch.Tensor,
    device: str | torch.device,
) -> torch.Tensor:
    """Forward swing reach per trajectory (``inf`` for gate-less trajectories).

    For each trajectory with contact gates, evaluate its position output just
    before the *first* gate (``gate_phi - EPS``) and read the swing foot's
    local-frame x — the forward distance the swing foot lands ahead of the
    stance-foot anchor over one stride.  Trajectories without gates (perpetual,
    e.g. standing) get ``inf`` so they are never chosen as a slow-down step.

    Args:
        manager: A loaded :class:`MultiSkillManager` (needs ``_num_gates_per_traj``,
            ``_gate_phi_table``, ``_ref_frame_map`` (call
            :meth:`MultiSkillManager.build_ref_frame_map` first), ``_EPS_PHI``,
            ``_get_domain_indices``, ``_compute_normalized_tau``,
            ``_compute_bezier_batched``, and ``data["coeffs_pos"|"domain_times"]``).
        ref_frame_to_pos_xy_idx: ``[num_ref_frames, 2]`` mapping each entry of
            ``manager.ref_frames`` to its ``(pos_x, pos_y)`` column in
            ``manager.pos_output_names`` — the same ordering
            ``manager.data["coeffs_pos"]`` (hence ``_compute_bezier_batched``)
            produces.
        device: Torch device for the returned table.

    Returns:
        ``[num_trajectories]`` forward swing reach (local +x), ``inf`` where the
        trajectory has no contact gates.
    """
    eps = manager._EPS_PHI
    T = manager.num_trajectories
    step_len = torch.full((T,), float("inf"), device=device)

    t_ids = torch.where(manager._num_gates_per_traj > 0)[0]
    if t_ids.numel() == 0:
        return step_len

    # First gate (index 0) = end of the first stride.
    gate_phi = manager._gate_phi_table[t_ids, 0]                 # [G]
    pre_phi = gate_phi - eps
    # Post-gate ref frame = the foot that lands at the gate (wrap so a phi=1.0
    # gate reads from the next period's start).
    post_phi = torch.remainder(gate_phi + eps, 1.0)
    post_dom = manager._get_domain_indices(post_phi, t_ids)
    swing_ref = manager._ref_frame_map[t_ids, post_dom]         # [G] -> ref_frames idx

    # Evaluate the position output at ``pre_phi`` per trajectory.
    pre_dom = manager._get_domain_indices(pre_phi, t_ids)
    tau = manager._compute_normalized_tau(pre_phi, t_ids, pre_dom)
    coeffs = manager.data["coeffs_pos"][t_ids, pre_dom]         # [G, P, K+1]
    T_dom = manager.data["domain_times"][t_ids, pre_dom]
    pos = manager._compute_bezier_batched(tau, coeffs, T_dom, derivative=False)  # [G, P]

    x_cols = ref_frame_to_pos_xy_idx[swing_ref, 0]             # [G] swing pos_x col
    local_x = pos.gather(1, x_cols.unsqueeze(1)).squeeze(1)    # [G]
    step_len[t_ids] = local_x
    return step_len
