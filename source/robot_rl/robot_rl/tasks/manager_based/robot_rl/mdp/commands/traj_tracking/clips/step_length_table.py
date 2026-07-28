# Copyright 2026 Zachary Olkin. All rights reserved.

"""Per-trajectory forward swing-reach table (for the offline clip generator).

The terrain-approach planner picks the walk trajectory whose step lands the
swing foot at the stair stance reference; that decision needs, per trajectory,
the forward (local +x) distance the swing foot travels over one stride.  The
offline clip generator (:class:`OfflineClipGenerator`) uses this when stitching
a flat approach onto a staircase.

The table is built from Bezier evaluation just before the first contact gate,
so the swing reach matches the trajectory's actual per-stride step length.
"""

from __future__ import annotations

import torch

from ..sagittal_reflector import swap_left_right


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


def _landing_foot_pos_x_cols(
    manager, t_ids: torch.Tensor, post_dom: torch.Tensor, device: str | torch.device,
) -> torch.Tensor:
    """``pos_x`` output column of the foot that LANDS at each gate.

    The landing foot is the contact body of the post-gate domain (the foot
    coming into contact at the strike), taken from
    ``_contact_bodies_per_domain`` — the physically correct swing-foot
    resolution.  This deliberately does NOT use ``_ref_frame_map``: the bezier
    reference frame is the right ankle for *every* domain in the current
    library, so ``_ref_frame_map`` would always select the right foot, and with
    the two-gate full-periodic change (gate 0 = the mid-cycle LEFT strike) that
    would measure the wrong foot at the wrong phase.  For the reflected half of a
    half-periodic cycle the stored bodies are sagittally mirrored, so the body
    name is swapped back (matching ``_precompute_contact_table``).

    Args:
        manager: A loaded :class:`MultiSkillManager`.
        t_ids: ``[G]`` global trajectory indices (gated trajectories).
        post_dom: ``[G]`` expanded-domain index just after each gate.
        device: Torch device for the returned tensor.

    Returns:
        ``[G]`` long tensor of ``pos_x`` columns into ``manager.pos_output_names``.

    Raises:
        ValueError: A post-gate domain has no contact body (cannot resolve the
            landing foot) or the foot lacks a ``pos_x`` output.
    """
    names = manager.pos_output_names
    nd_tbl = manager.data["num_original_domains"]
    cols = torch.empty(t_ids.numel(), dtype=torch.long, device=device)
    for i, (t, pd) in enumerate(zip(t_ids.tolist(), post_dom.tolist())):
        nd = int(nd_tbl[t].item())
        bodies = manager._contact_bodies_per_domain[t][pd % nd]
        if not bodies:
            raise ValueError(
                f"trajectory {t}: post-gate domain {pd} has no contact body, so "
                "the landing foot for the step-length table cannot be resolved. "
                "Gates must sit at a foot strike (a domain that comes into contact)."
            )
        body = bodies[0]
        if pd >= nd:  # reflected half of a half-periodic cycle: feet are swapped
            body = swap_left_right(body)
        try:
            cols[i] = names.index(f"{body}:pos_x")
        except ValueError as e:
            raise ValueError(
                f"landing foot '{body}' is missing a pos_x output; the "
                "step-length table needs the swing foot's forward position."
            ) from e
    return cols


def traj_step_length_table(
    manager,
    device: str | torch.device,
) -> torch.Tensor:
    """Forward swing reach per trajectory (``inf`` for gate-less trajectories).

    For each trajectory with contact gates, evaluate its position output just
    before the *first* gate (``gate_phi - EPS``) and read the LANDING foot's
    local-frame x — the forward distance the swing foot lands ahead of the
    stance-foot anchor over the first stride.  The landing foot is resolved from
    the contact schedule (:func:`_landing_foot_pos_x_cols`), not the bezier
    reference frame.  After the two-gate full-periodic change (gate 0 = the
    mid-cycle strike), this is the single first-stride reach the terrain-approach
    planner needs; before it (or for episodic / no-mid fallbacks, gate 0 =
    phi 1.0) it is the reach to the cycle-wrap strike.  Trajectories without
    gates (perpetual, e.g. standing) get ``inf`` so they are never chosen as a
    slow-down step.

    Args:
        manager: A loaded :class:`MultiSkillManager` (needs ``_num_gates_per_traj``,
            ``_gate_phi_table``, ``_EPS_PHI``, ``_get_domain_indices``,
            ``_compute_normalized_tau``, ``_compute_bezier_batched``,
            ``_contact_bodies_per_domain``, ``pos_output_names`` and
            ``data["coeffs_pos"|"domain_times"|"num_original_domains"]``).
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

    # First gate (index 0) = end of the first stride (the mid-cycle strike for a
    # full-periodic gait; phi=1.0 for episodic / no-mid fallbacks).
    gate_phi = manager._gate_phi_table[t_ids, 0]                 # [G]
    pre_phi = gate_phi - eps
    # Post-gate domain = the domain the landing foot comes into contact in (wrap
    # so a phi=1.0 gate reads from the next period's start).
    post_phi = torch.remainder(gate_phi + eps, 1.0)
    post_dom = manager._get_domain_indices(post_phi, t_ids)
    x_cols = _landing_foot_pos_x_cols(manager, t_ids, post_dom, device)  # [G]

    # Evaluate the position output at ``pre_phi`` per trajectory.
    pre_dom = manager._get_domain_indices(pre_phi, t_ids)
    tau = manager._compute_normalized_tau(pre_phi, t_ids, pre_dom)
    coeffs = manager.data["coeffs_pos"][t_ids, pre_dom]         # [G, P, K+1]
    T_dom = manager.data["domain_times"][t_ids, pre_dom]
    pos = manager._compute_bezier_batched(tau, coeffs, T_dom, derivative=False)  # [G, P]

    local_x = pos.gather(1, x_cols.unsqueeze(1)).squeeze(1)    # [G]
    step_len[t_ids] = local_x
    return step_len
