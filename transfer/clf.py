"""
clf_qp_controller.py
Minimal Control-Lyapunov-Function QP for HZD/virtual-constraint walking
"""

from dataclasses import dataclass
from collections.abc import Sequence

import casadi as ca
import numpy as np
import pinocchio as pin


@dataclass
class CLFQPParams:
    kp: float = 100.0  # proportional gain for desired dyn
    kd: float = 2.0 * np.sqrt(100.0)
    alpha: float = 20.0  # CLF decay rate  (dotV <= -alpha V)
    u_min: Sequence[float] = None
    u_max: Sequence[float] = None


class CLFQPController:
    """
    Builds V(z) = zᵀ P z from linear error dynamics and solves
        min_u ||u - u_ff||²
        s.t.  dot V <= -c V
              u_min ≤ u ≤ u_max
    """

    def __init__(
        self,
        model: pin.Model,
        virtual_constraint_fun,
        phase_variable_fun,
        params: CLFQPParams = CLFQPParams(),
    ):
        """
        virtual_constraint_fun: (q, τ) -> y  (Rⁿ × R → Rᵖ)
        phase_variable_fun    : (q)    -> τ  (Rⁿ → R)   monotonic in step
        """
        self.model, self.params = model, params
        self.vc_fun = virtual_constraint_fun
        self.phase_fun = phase_variable_fun

        self.nv = model.nv
        self.np = model.nv  # full torque input assumed; edit for under-actuation

        # ----------------------------------------------------------------------
        # Pre-compute Lyapunov matrix P from desired linear error dynamics
        A = np.block([
            [np.zeros((params.kp.shape[0], params.kp.shape[0])), np.eye(params.kp.shape[0])],
            [-np.diag(params.kp), -np.diag(params.kd)],
        ])
        Q = np.eye(A.shape[0])
        # solve Aᵀ P + P A = -Q
        from scipy.linalg import solve_continuous_lyapunov

        self.P = solve_continuous_lyapunov(A.T, -Q)
        # ----------------------------------------------------------------------

        # Build CasADi symbols *once* (faster at run-time)
        self._build_casadi_controller()

    # --------------------------------------------------------------------------
    # Public interface
    # --------------------------------------------------------------------------

    def compute_control(self, q, dq):
        """Return torque vector u (np.ndarray, shape=(nv,))."""
        x = np.concatenate([q, dq])
        sol = self.qp_solver(x0=x)
        return np.array(sol["x"]).squeeze()[: self.nv]

    # --------------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------------

    def _build_casadi_controller(self):
        """Generate symbolic CLF-QP and wrap in a callable CasADi function."""
        nv = self.nv
        # ----- symbols ---------------------------------------------------------
        q_sym = ca.SX.sym("q", nv)
        dq_sym = ca.SX.sym("dq", nv)
        u_sym = ca.SX.sym("u", nv)

        # phase & output --------------------------------------------------------
        tau = self.phase_fun(q_sym)  # scalar phase variable
        y = self.vc_fun(q_sym, tau)  # ℝᵖ outputs
        dy = ca.jacobian(y, q_sym) @ dq_sym

        z = ca.vertcat(y, dy)  # error state

        # CLF -------------------------------------------------------------------
        V = ca.mtimes([z.T, self.P, z])  # zᵀ P z
        dVdq = ca.gradient(V, q_sym)
        dVddq = ca.gradient(V, dq_sym)

        # ----- dynamics --------------------------------------------------------
        # USER-SUPPLIED call: replace with your own inverse dynamics wrapper.
        # ----------------------------------------------------------------------
        #   pin.computeABCD returns   M(q), h(q,dq), S (actuator map)
        # ----------------------------------------------------------------------
        M = pin.crba(self.model, pin.Data(self.model), q_sym)
        h = pin.nonLinearEffects(self.model, pin.Data(self.model), q_sym, dq_sym)

        # affine dynamics  ddq = M⁻¹ (u - h)
        Minv = ca.inv(M)

        LfV = ca.dot(dVdq, dq_sym) + ca.dot(dVddq, Minv @ (-h))
        LgV = ca.mtimes([dVddq, Minv])  # row vector × matrix

        # ----- QP cost ---------------------------------------------------------
        u_ff = ca.SX.zeros((nv, 1))  # feed-forward, use 0 or precomp.
        obj = ca.sumsqr(u_sym - u_ff)

        # QP constraint:  LfV + LgV u + c V <= 0
        clf_con = LfV + ca.mtimes(LgV, u_sym) + self.params.alpha * V

        # bounds
        u_min = self.params.u_min if self.params.u_min is not None else [-1e6] * nv
        u_max = self.params.u_max if self.params.u_max is not None else [1e6] * nv

        nlp = {"x": u_sym, "f": obj, "g": clf_con}
        opts = {"lbg": -ca.inf, "ubg": 0.0, "lbx": u_min, "ubx": u_max}
        self.qp_solver = ca.qpsol("solver", "qpoases", nlp, opts)


# ------------------------------------------------------------------------------
# EXAMPLE STUBS (replace with your robot-specific code)
# ------------------------------------------------------------------------------


def time_phase_variable(q, t, T):
    return t / T


def joint_virtual_constraint(q, tau):
    """
    Simple 1-DoF output: knee angle follows a Bézier curve of tau.
    For p>1 just stack outputs with ca.vertcat().
    """
    # load from bezier or something?
    # q_desired = q  # This was unused
    # return q[knee_idx] - B  # knee_idx and B were undefined
    # For now, return a simple example - replace with actual implementation
    return q[0]  # Return first joint as example


# ------------------------------------------------------------------------------
# USAGE
# ------------------------------------------------------------------------------


if __name__ == "__main__":
    urdf_path = "path/to/your_robot.urdf"
    model = pin.buildModelFromUrdf(urdf_path)
    ctrl = CLFQPController(
        model,
        virtual_constraint_fun=joint_virtual_constraint,
        phase_variable_fun=time_phase_variable,
        params=CLFQPParams(kp=np.array([100.0]), kd=np.array([20.0]), alpha=10.0, u_min=[-100], u_max=[100]),
    )
    # dummy state
    q0 = pin.neutral(model)
    dq0 = np.zeros(model.nv)
    u = ctrl.compute_control(q0, dq0)
    # store the trajectory and visualize it in meshcat, plot it, etc.
    print("Torque command:", u)
