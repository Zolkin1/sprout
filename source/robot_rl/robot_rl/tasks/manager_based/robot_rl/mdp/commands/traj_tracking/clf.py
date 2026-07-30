# Copyright 2026 Zachary Olkin. All rights reserved.

from typing import Dict
import torch
import numpy as np
from scipy.linalg import solve_continuous_are
from isaaclab.utils.math import quat_box_minus

from motion_gen.joint_names import is_root_frame

class CLF:
    """
    Continuous-time Control Lyapunov Function (CLF) evaluator for relative-degree-2 outputs.
    Uses user-provided LIP dynamics (A_lip, B_lip) and augments with double-integrator channels
    for additional outputs. Solves the continuous-time ARE once via SciPy, caches P and LQR gain K
    on the specified torch device for efficient V, V_dot, and control-law evaluation.

    The command's ``y``/``dy`` vectors may carry MORE channels than the CLF grades:
    ``tracked_vel_names`` selects the subset that forms eta.  That is what makes the
    joint-vs-body tracking A/B a config switch on one clip library -- the clip always stores
    both the ``joint:*`` and the root-frame body channels, and only the selection changes.
    """
    def __init__(
        self,
        sim_dt: float,
        batch_size: int,
        ordered_vel_output_names: list[str],
        ordered_pos_output_names: list[str],
        device: torch.device = None,
        Q_weights: Dict = None,
        R_weights: Dict = None,
        tracked_vel_names: list[str] | None = None,
        # domain_scalar: list[float]|None = None
    ):
        """Build P over the tracked channel subset and cache the gathers into the full vectors.

        Args:
            sim_dt: Simulation timestep.
            batch_size: Number of environments.
            ordered_vel_output_names: FULL velocity channel names of the command's ``dy``.
            ordered_pos_output_names: FULL position channel names of the command's ``y``.
            device: Torch device.
            Q_weights: Per-channel ``[q_pos, q_vel]``, keyed by velocity channel name.
            R_weights: Per-channel ``[r]``, keyed by velocity channel name.
            tracked_vel_names: Velocity channels that form eta; ``None`` tracks all of them.
                Order is normalised to ``ordered_vel_output_names`` order regardless of how it
                is passed, which is what keeps the pos/vel channel correspondence that
                :meth:`compute_y_err` relies on.
        """
        # Initialize device and basic parameters
        self.device = device
        self.sim_dt = sim_dt

        # --- Tracked-channel subset -------------------------------------------------
        full_pos = list(ordered_pos_output_names)
        full_vel = list(ordered_vel_output_names)
        if tracked_vel_names is None:
            tracked_pos, tracked_vel = full_pos, full_vel
        else:
            want = set(tracked_vel_names)
            unknown = sorted(want - set(full_vel))
            if unknown:
                raise ValueError(f"tracked_vel_names not in the command's outputs: {unknown}")
            # Filter the FULL lists to preserve their relative order.  A position list also
            # keeps each tracked frame's ``ori_w`` (present in y, absent from dy).
            tracked_vel = [n for n in full_vel if n in want]
            tracked_pos = [
                n for n in full_pos
                if n in want or (n.endswith(":ori_w") and f"{n[:-1]}x" in want)
            ]
        self._subset = (tracked_pos != full_pos) or (tracked_vel != full_vel)
        self.pos_gather = torch.tensor([full_pos.index(n) for n in tracked_pos],
                                       dtype=torch.long, device=self.device)
        self.vel_gather = torch.tensor([full_vel.index(n) for n in tracked_vel],
                                       dtype=torch.long, device=self.device)

        # From here on, "ordered_*_output_names" means the TRACKED set: P, the subgroup index
        # maps and the quaternion maps are all built over it.
        ordered_pos_output_names = tracked_pos
        ordered_vel_output_names = tracked_vel
        self.n_outputs = 2*len(ordered_vel_output_names)
        self.ordered_pos_output_names = ordered_pos_output_names
        self.ordered_vel_output_names = ordered_vel_output_names

        # Set up default Q, R if not provided
        # Q_weights should be length = n_states, R_weights length = n_inputs
        # self.num_domain = num_domain

        n_states = self.n_outputs
        self.n_inputs = int(self.n_outputs/2)
        if Q_weights is None:
            Q = np.ones(n_states)
        else:
            # Create the matrix using the dict
            Q = np.ones(n_states)
            for i, name in enumerate(ordered_vel_output_names):
                Q[2*i] = Q_weights[name][0]
                Q[2*i + 1] = Q_weights[name][1]
        if R_weights is None:
            R = 0.1 * np.ones(self.n_inputs)
        else:
            R = np.ones(self.n_inputs)
            for i, name in enumerate(ordered_vel_output_names):
                R[i] = R_weights[name][0]


    # if self.num_domain == 1:

        Q_np = np.diag(Q)
        R_np = np.diag(R)

        # Solve for P and LQR gain K in NumPy
        P_np = self._compute_PK_np(Q_np,R_np)

        # Cache as torch tensors
        self.P = torch.from_numpy(P_np).to(self.device).to(torch.float32)
        self.lambda_max = torch.linalg.eigvalsh(self.P)[-1]
        self.norm_P = torch.linalg.norm(self.P, ord=2)


        # else:
        #     P = []
        #     lambda_max = []
        #     norm_P = []
        #     for i in range(self.num_domain):
        #         Q_np = np.diag(Q)*domain_scalar[i*2]
        #         R_np = np.diag(R)*domain_scalar[i*2+1]
        #
        #         # Solve for P and LQR gain K in NumPy
        #         P_np = self._compute_PK_np(Q_np,R_np)
        #
        #         # Cache as torch tensors
        #         P.append(torch.from_numpy(P_np).to(self.device).to(torch.float32))
        #         lambda_max.append(torch.linalg.eigvalsh(self.P)[-1])
        #         norm_P.append(torch.linalg.norm(self.P, ord=2))
        #     self.P = P

        # For ablations
        self.P_ID = torch.from_numpy(np.eye(self.n_outputs)).to(self.device).to(torch.float32)

        # Build eta-state indices for each subgroup.
        # For output i, eta has position at 2*i and velocity at 2*i+1.
        subgroup_eta_indices: dict[str, list[int]] = {
            "pelvis_pos": [],
            "pelvis_lin_vel": [],
            "pelvis_ori": [],
            "pelvis_ang_vel": [],
            "joint_pos": [],
            "joint_vel": [],
            "other_body_pos": [],
            "other_body_lin_vel": [],
            "other_body_ori": [],
            "other_body_ang_vel": [],
            # Root-frame body links (``<body>@root:*``) are their own group so the logged
            # error split separates local limb tracking from the world-frame skeleton.
            "root_body_pos": [],
            "root_body_lin_vel": [],
            "root_body_ori": [],
            "root_body_ang_vel": [],
        }
        for i, name in enumerate(ordered_vel_output_names):
            if name.startswith("pelvis_link:pos_"):
                pos_key = "pelvis_pos"
                vel_key = "pelvis_lin_vel"
            elif name.startswith("pelvis_link:ori_"):
                pos_key = "pelvis_ori"
                vel_key = "pelvis_ang_vel"
            elif name.startswith("joint:"):
                pos_key = "joint_pos"
                vel_key = "joint_vel"
            elif is_root_frame(name) and ":pos_" in name:
                pos_key = "root_body_pos"
                vel_key = "root_body_lin_vel"
            elif is_root_frame(name) and ":ori_" in name:
                pos_key = "root_body_ori"
                vel_key = "root_body_ang_vel"
            elif ":pos_" in name:
                pos_key = "other_body_pos"
                vel_key = "other_body_lin_vel"
            elif ":ori_" in name:
                pos_key = "other_body_ori"
                vel_key = "other_body_ang_vel"
            else:
                continue
            subgroup_eta_indices[pos_key].append(2 * i)
            subgroup_eta_indices[vel_key].append(2 * i + 1)

        self.subgroup_indices: dict[str, torch.Tensor] = {
            k: torch.tensor(v, dtype=torch.long, device=self.device)
            for k, v in subgroup_eta_indices.items()
        }
        self.v_subgroups: dict[str, torch.Tensor] = {}

        # Build mapping from body name to [ori_x_idx, ori_y_idx, ori_z_idx, ori_w_idx] in y tensor
        self.ori_body_indices_manifold: dict[str, list[int]] = {}
        self.ori_body_indices_tangent: dict[str, list[int]] = {}

        self.not_quat_mask_manifold = torch.ones(len(ordered_pos_output_names), dtype=torch.bool, device=self.device)
        self.not_quat_mask_tangent = torch.ones(len(ordered_vel_output_names), dtype=torch.bool, device=self.device)

        for i, name in enumerate(ordered_pos_output_names):
            for ori_axis in [":ori_x", ":ori_y", ":ori_z", ":ori_w"]:
                if ori_axis in name:
                    body_name = name.split(ori_axis)[0]

                    if body_name in self.ori_body_indices_manifold:
                        self.ori_body_indices_manifold[body_name].append(i)
                    else:
                        self.ori_body_indices_manifold[body_name] = [i]
                    self.not_quat_mask_manifold[i] = False

        for i, name in enumerate(ordered_vel_output_names):
            for ori_axis in [":ori_x", ":ori_y", ":ori_z"]:
                if ori_axis in name:
                    body_name = name.split(ori_axis)[0]

                    self.not_quat_mask_tangent[i] = False
                    if body_name in self.ori_body_indices_tangent:
                        self.ori_body_indices_tangent[body_name].append(i)
                    else:
                        self.ori_body_indices_tangent[body_name] = [i]

        # Stack the per-body quaternion index blocks so compute_y_err runs ONE batched
        # quat_box_minus instead of one kernel launch per tracked body.  Body order is shared
        # between the two so the manifold rows and the tangent rows line up.
        self._ori_bodies = list(self.ori_body_indices_manifold)
        if self._ori_bodies:
            self._ori_manifold_idx = torch.tensor(
                [self.ori_body_indices_manifold[b] for b in self._ori_bodies],
                dtype=torch.long, device=self.device)                       # [B, 4]
            self._ori_tangent_idx = torch.tensor(
                [self.ori_body_indices_tangent[b] for b in self._ori_bodies],
                dtype=torch.long, device=self.device)                       # [B, 3]
        else:
            self._ori_manifold_idx = torch.zeros((0, 4), dtype=torch.long, device=self.device)
            self._ori_tangent_idx = torch.zeros((0, 3), dtype=torch.long, device=self.device)

        self.v_buffer = torch.zeros((batch_size, 3), device=self.device)
        self.step_count = 0

    def _select(self, y_act, y_nom, dy_act, dy_nom):
        """Gather the tracked channel subset out of the command's full ``y``/``dy`` vectors.

        Args:
            y_act, y_nom: ``[B, P_full]`` measured / desired positions.
            dy_act, dy_nom: ``[B, V_full]`` measured / desired velocities.

        Returns:
            The same four tensors restricted to the tracked channels (unchanged when the CLF
            tracks everything, so the common path stays copy-free).
        """
        if not self._subset:
            return y_act, y_nom, dy_act, dy_nom
        return (y_act[:, self.pos_gather], y_nom[:, self.pos_gather],
                dy_act[:, self.vel_gather], dy_nom[:, self.vel_gather])


    def _compute_PK_np(self, Q_np, R_np) -> tuple[np.ndarray, np.ndarray]:
        """
        Construct a pure double integrator system for all outputs,
        and solve for the LQR gain K and Lyapunov matrix P.
        """

        # Assume each output has a double integrator model:
        #   [ẋ] = [0 1][x] + [0] u
        #         [0 0]      [1]


        # 1) Build block-diagonal A and B matrices (double integrators)
        A_blk = np.array([[0.0, 1.0], [0.0, 0.0]])  # (2x2)
        B_blk = np.array([[0.0], [1.0]])           # (2x1)

        A_full = np.kron(np.eye(self.n_inputs), A_blk)   # (2n x 2n)    # n_inputs is totaly number of double integrator systems
        B_full = np.kron(np.eye(self.n_inputs), B_blk)   # (2n x n)


        # 2) Solve CARE: A^T P + P A - P B R^{-1} B^T P + Q = 0
        # TODO(future): switch to the discrete-time ARE (scipy.solve_discrete_are)
        # for self-consistency with the discrete finite-difference vdot
        # (V_k - V_{k-1}) and the per-step CLF decrease condition. Would use the
        # ZOH-discretized double integrator A_d=[[1,h],[0,1]], B_d=[[h^2/2],[h]]
        # with h=sim_dt (and cost-equivalent Q*h, R*h to preserve P's scale). The
        # position/velocity weight ratio p_pos/p_vel = sqrt(q_p/r) is unchanged for
        # fine h, so Q/R tuning carries over; the discretization only bites at
        # coarse h or very large q_p/r. See scripts/rsl_rl/clf_weight_check.py.
        P = solve_continuous_are(A_full, B_full, Q_np, R_np)

        return P


    def compute_v(
        self,
        y_act: torch.Tensor,
        y_nom: torch.Tensor,
        dy_act: torch.Tensor,
        dy_nom: torch.Tensor,
        domain_idx: int = 0,
    ) -> torch.Tensor:
        """
        Evaluate V = (y_act - y_nom)^T P (y_act - y_nom).

        Inputs are the command's FULL ``y``/``dy`` vectors; only the tracked channel subset
        (see ``tracked_vel_names``) enters eta.
        """
        y_act, y_nom, dy_act, dy_nom = self._select(y_act, y_nom, dy_act, dy_nom)

        y_err = self.compute_y_err(y_act, y_nom)

        dy_err = dy_act - dy_nom
        batch_size = y_act.shape[0]
        eta = torch.zeros(batch_size,self.n_outputs, device=y_act.device)
        eta[:,0::2] = y_err      # even indices: positions
        eta[:,1::2] = dy_err     # odd indices: velocities

        # Compute per-subgroup V contributions (self-contribution, excluding cross-terms)
        self.v_subgroups = {}
        for name, idx in self.subgroup_indices.items():
            if idx.numel() > 0:
                eta_sub = eta[:, idx].contiguous()
                P_sub = self.P_ID[idx][:, idx].contiguous()     # Use identity P for these to just compute errors, not Lyap values
                self.v_subgroups[name] = (torch.matmul(eta_sub, P_sub) * eta_sub).sum(dim=-1)

        V = torch.einsum('bi,ij,bj->b', eta, self.P, eta)

        self.v_buffer[:, 2] = self.v_buffer[:, 1]
        self.v_buffer[:, 1] = self.v_buffer[:, 0]
        # We detach() so that backprop does not try to flow through the history buffer.
        self.v_buffer[:, 0] = V.detach()

        self.step_count += 1
        return V

    def compute_vdot(
        self,
        y_act: torch.Tensor,
        y_nom: torch.Tensor,
        dy_act: torch.Tensor,
        dy_nom: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the per-step Lyapunov DECREASE, returns (delta_v, V_curr).

        NOTE the units: this is the raw difference ``V_k - V_{k-1}`` over ONE CONTROL step,
        NOT a time derivative -- there is deliberately no division by dt (which is why
        ``self.sim_dt`` goes unused). Consumers are written to match: the ``max_violation``
        normalizer in ``mdp.rewards.clf_decreasing_condition`` carries its own ``dt``
        factor, and its ``alpha * v`` term is likewise un-scaled, so the effective
        continuous-time decay rate there is ``alpha / dt``, not ``alpha``. Anything that
        starts dividing by dt here has to move those two with it, and any tuned Q/R
        weights need re-checking.

        ``self.sim_dt`` is the SIM step (0.005 s), while this is called once per CONTROL
        step (``sim.dt * decimation``, 0.02 s) -- so it is not the right dt to divide by
        either. See the discrete-ARE TODO in ``_compute_PK_np``: ``P`` solves the
        CONTINUOUS ARE while this difference is discrete, which is the same
        continuous-vs-discrete seam.
        """
        v_curr = self.compute_v(y_act, y_nom, dy_act, dy_nom,)# yaw_idx)
       
        B = v_curr.shape[0]

        if self.step_count == 0:
            vdot_raw = torch.zeros((B,), device=self.device)
        else:
            # Discrete time CLF computation
            V_k = self.v_buffer[:, 0]
            V_k1 = self.v_buffer[:, 1]
            vdot_raw = (V_k - V_k1)

        # Clamp unreasonably large vdot values (e.g., after resets)
        vdot_raw = torch.where(torch.abs(vdot_raw) > 10000, torch.zeros_like(vdot_raw), vdot_raw)

        return vdot_raw, v_curr


    def compute_y_err(self, y_act, y_des):
        """
        Compute the error between the measured values and the desired values.

        Must respect the quaternion orientation by using the box minus operator.

        Args:
            y_act: ``[B, P]`` measured positions over the TRACKED channels.
            y_des: ``[B, P]`` desired positions over the TRACKED channels.

        Returns:
            ``[B, V]`` error in the tangent (velocity) channel layout.
        """
        b = y_act.shape[0]
        y_err = torch.zeros((b, len(self.ordered_vel_output_names)), device=self.device)

        # Subtract all the non-quaternions normally
        y_err[:, self.not_quat_mask_tangent] = y_act[:, self.not_quat_mask_manifold] - y_des[:, self.not_quat_mask_manifold]

        # Subtract quats with the box-minus operator.  ONE batched call over every tracked
        # orientation frame -- body tracking pushes this from 5 frames to 18, and a Python
        # loop would be 18 kernel launches on the critical path of every step.
        nb = self._ori_manifold_idx.shape[0]
        if nb:
            act_q = y_act[:, self._ori_manifold_idx.reshape(-1)].reshape(b * nb, 4)
            des_q = y_des[:, self._ori_manifold_idx.reshape(-1)].reshape(b * nb, 4)
            box = quat_box_minus(act_q, des_q).reshape(b, nb * 3)
            y_err[:, self._ori_tangent_idx.reshape(-1)] = box

        return y_err