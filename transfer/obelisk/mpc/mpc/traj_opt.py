import numpy as np
import pandas as pd
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def dynamics_constraint(f, z, v):
    """
    Creates dynamics constraints
    :param f: dynamics function
    :param z: state variable, N+1 x n
    :param v: input variable, N x m
    :return: dynamics constraints, 1 x N*n
    """
    g = []
    for k in range(v.shape[0]):
        g.append(f(z[k, :].T, v[k, :].T).T - z[k + 1, :])
    g = ca.horzcat(*g)
    g_lb = ca.DM(*g.shape)
    g_ub = ca.DM(*g.shape)
    return g, g_lb, g_ub


def quadratic_objective(x, Q, goal=None):
    """
    Returns a quadratic objective function of the form sum_0^N x_k^T @ Q @ x_k
    :param x: state variable, N+1 x n
    :param Q: Quadratic cost matrix
    :param goal: desired state, 1 x n
    :return: scalar quadratic objective function
    """
    if goal is None:
        dist = x
    else:
        if goal.shape == x.shape:
            dist = x - goal
        else:
            dist = x - ca.repmat(goal, x.shape[0], 1)
    return ca.sum1(ca.sum2((dist @ Q) * dist))


def ellipse_barrier_value(z, obs_cx, obs_cy, obs_rx, obs_ry, obs_yaw):
    """
    Computes the ellipse barrier function value h(z).

    h(z) = (x_local/rx)^2 + (y_local/ry)^2 - 1
    h(z) > 0 means outside ellipse (safe)
    h(z) < 0 means inside ellipse (unsafe)

    :param z: state (n,), uses z[0] as x, z[1] as y
    :param obs_cx: ellipse center x (scalar)
    :param obs_cy: ellipse center y (scalar)
    :param obs_rx: semi-axis in local x direction (scalar)
    :param obs_ry: semi-axis in local y direction (scalar)
    :param obs_yaw: rotation angle of ellipse in radians (scalar)
    :return: barrier value h (scalar)
    """
    # Displacement from ellipse center
    dx = z[0] - obs_cx
    dy = z[1] - obs_cy

    # Rotate to ellipse-local frame
    cos_yaw = ca.cos(obs_yaw)
    sin_yaw = ca.sin(obs_yaw)
    x_local = dx * cos_yaw + dy * sin_yaw
    y_local = -dx * sin_yaw + dy * cos_yaw

    # Barrier function: (x_local/rx)^2 + (y_local/ry)^2 - 1
    h = (x_local / obs_rx)**2 + (y_local / obs_ry)**2 - 1
    return h


def single_obstacle_constraint_k(z, obs_cx, obs_cy, obs_rx, obs_ry, obs_yaw):
    """
    Ellipse avoidance constraint for a single timestep.
    Constraint: h(z) >= 0

    :param z: state at timestep k (n,)
    :param obs_cx: ellipse center x (scalar)
    :param obs_cy: ellipse center y (scalar)
    :param obs_rx: semi-axis in local x direction (scalar)
    :param obs_ry: semi-axis in local y direction (scalar)
    :param obs_yaw: rotation angle of ellipse in radians (scalar)
    :return: constraint g, lower bound, upper bound
    """
    g = ellipse_barrier_value(z, obs_cx, obs_cy, obs_rx, obs_ry, obs_yaw)
    return g, ca.DM([0]), ca.DM.inf()


def single_obstacle_constraint(z, obs_cx, obs_cy, obs_rx, obs_ry, obs_yaw):
    """
    Applies ellipse constraint across all timesteps for a single obstacle.
    """
    g = []
    g_lb = []
    g_ub = []
    for k in range(z.shape[0]):
        g_k, g_lb_k, g_ub_k = single_obstacle_constraint_k(
            z[k, :], obs_cx, obs_cy, obs_rx, obs_ry, obs_yaw
        )
        g.append(g_k)
        g_lb.append(g_lb_k)
        g_ub.append(g_ub_k)
    return ca.horzcat(*g), ca.horzcat(*g_lb), ca.horzcat(*g_ub)


def obstacle_constraints(z, obs_cx, obs_cy, obs_rx, obs_ry, obs_yaw):
    """
    Constructs elliptical obstacle constraints for all obstacles.

    :param z: state variable, N+1 x n
    :param obs_cx: center x coordinates, K x 1
    :param obs_cy: center y coordinates, K x 1
    :param obs_rx: semi-axis in local x, K x 1
    :param obs_ry: semi-axis in local y, K x 1
    :param obs_yaw: rotation angle in radians, K x 1
    """
    g = []
    g_lb = []
    g_ub = []
    for i in range(obs_rx.shape[0]):
        g_i, g_lb_i, g_ub_i = single_obstacle_constraint(
            z, obs_cx[i, :], obs_cy[i, :], obs_rx[i, :], obs_ry[i, :], obs_yaw[i, :]
        )
        g.append(g_i)
        g_lb.append(g_lb_i)
        g_ub.append(g_ub_i)
    return ca.horzcat(*g), ca.horzcat(*g_lb), ca.horzcat(*g_ub)


# =============================================================================
# Control Barrier Function (CBF) Constraints
# =============================================================================

def single_cbf_constraint_k(z_k, z_kplus1, obs_cx, obs_cy, obs_rx, obs_ry, obs_yaw, alpha, delta):
    """
    Discrete-time CBF constraint for a single timestep transition.

    CBF constraint: h(z[k+1]) >= (1 - alpha) * h(z[k]) + delta
    Rearranged as: h(z[k+1]) - (1 - alpha) * h(z[k]) - delta >= 0

    :param z_k: state at timestep k (n,)
    :param z_kplus1: state at timestep k+1 (n,)
    :param obs_cx: ellipse center x (scalar)
    :param obs_cy: ellipse center y (scalar)
    :param obs_rx: semi-axis in local x direction (scalar)
    :param obs_ry: semi-axis in local y direction (scalar)
    :param obs_yaw: rotation angle of ellipse in radians (scalar)
    :param alpha: CBF decay rate (0 < alpha < 1)
    :param delta: safety margin (delta >= 0)
    :return: constraint g, lower bound, upper bound
    """
    h_k = ellipse_barrier_value(z_k, obs_cx, obs_cy, obs_rx, obs_ry, obs_yaw)
    h_kplus1 = ellipse_barrier_value(z_kplus1, obs_cx, obs_cy, obs_rx, obs_ry, obs_yaw)

    g = h_kplus1 - (1 - alpha) * h_k - delta
    return g, ca.DM([0]), ca.DM.inf()


def single_cbf_constraint(z, obs_cx, obs_cy, obs_rx, obs_ry, obs_yaw, alpha, delta):
    """
    Applies CBF constraint across all timestep transitions for a single obstacle.
    Note: N transitions (k=0 to N-1), not N+1 like hard constraints.

    :param z: state trajectory, (N+1) x n
    :param alpha: CBF decay rate (0 < alpha < 1)
    :param delta: safety margin (delta >= 0)
    """
    g = []
    g_lb = []
    g_ub = []
    for k in range(z.shape[0] - 1):  # N transitions
        g_k, g_lb_k, g_ub_k = single_cbf_constraint_k(
            z[k, :], z[k + 1, :], obs_cx, obs_cy, obs_rx, obs_ry, obs_yaw, alpha, delta
        )
        g.append(g_k)
        g_lb.append(g_lb_k)
        g_ub.append(g_ub_k)
    return ca.horzcat(*g), ca.horzcat(*g_lb), ca.horzcat(*g_ub)


def cbf_constraints(z, obs_cx, obs_cy, obs_rx, obs_ry, obs_yaw, alpha, delta):
    """
    Constructs CBF constraints for all obstacles.

    :param z: state variable, (N+1) x n
    :param obs_cx: center x coordinates, K x 1
    :param obs_cy: center y coordinates, K x 1
    :param obs_rx: semi-axis in local x, K x 1
    :param obs_ry: semi-axis in local y, K x 1
    :param obs_yaw: rotation angle in radians, K x 1
    :param alpha: CBF decay rate (0 < alpha < 1)
    :param delta: safety margin (delta >= 0)
    """
    g = []
    g_lb = []
    g_ub = []
    for i in range(obs_rx.shape[0]):
        g_i, g_lb_i, g_ub_i = single_cbf_constraint(
            z, obs_cx[i, :], obs_cy[i, :], obs_rx[i, :], obs_ry[i, :], obs_yaw[i, :],
            alpha, delta
        )
        g.append(g_i)
        g_lb.append(g_lb_i)
        g_ub.append(g_ub_i)
    return ca.horzcat(*g), ca.horzcat(*g_lb), ca.horzcat(*g_ub)


def initial_condition_equality_constraint(z, z0):
    dist = z[0, :z0.shape[1]] - z0
    return dist, ca.DM(*dist.shape), ca.DM(*dist.shape)


def velocity_norm_constraint(v, v_norm_max):
    """
    Constrains the 2-norm of [v_par, v_perp] at each timestep.
    v[k,0]^2 + v[k,1]^2 <= v_norm_max^2

    :param v: input variable, N x m
    :param v_norm_max: maximum velocity 2-norm
    :return: constraints g, g_lb, g_ub
    """
    g = []
    g_lb = []
    g_ub = []
    for k in range(v.shape[0]):
        g.append(v[k, 0]**2 + v[k, 1]**2)
        g_lb.append(ca.DM([0]))
        g_ub.append(ca.DM([v_norm_max**2]))
    return ca.horzcat(*g), ca.horzcat(*g_lb), ca.horzcat(*g_ub)


def input_rate_constraint(v, v_prev, dv_max):
    """Constrains the rate of change of inputs between consecutive timesteps.

    Enforces -dv_max[j] <= v[k,j] - v[k-1,j] <= dv_max[j] for each component j.
    At k=0, the constraint is against v_prev (parameter from previous solve).

    :param v: input variable, N x m
    :param v_prev: previous solve's first input, 1 x m (CasADi parameter)
    :param dv_max: per-component rate limits, (m,) numpy array
    :return: constraints g, g_lb, g_ub
    """
    g = []
    g_lb = []
    g_ub = []
    m = v.shape[1]
    dv_max_dm = ca.DM(dv_max)

    for k in range(v.shape[0]):
        if k == 0:
            diff = v[0, :] - v_prev
        else:
            diff = v[k, :] - v[k - 1, :]
        for j in range(m):
            g.append(diff[j])
            g_lb.append(ca.DM([-float(dv_max_dm[j])]))
            g_ub.append(ca.DM([float(dv_max_dm[j])]))

    return ca.horzcat(*g), ca.horzcat(*g_lb), ca.horzcat(*g_ub)


def setup_trajopt_solver(pm, N, Nobs, has_rate_limit=False):
    """Set up decision variables, bounds, and parameters for trajectory optimization.

    :param pm: planning model with dynamics, bounds, etc.
    :param N: planning horizon
    :param Nobs: number of obstacles
    :param has_rate_limit: if True, adds p_v_prev parameter for rate limiting
    :return: tuple of decision variables, bounds, and parameters
    """
    # Set up state/input bounds
    z_min = ca.DM(pm.z_min)
    z_max = ca.DM(pm.z_max)
    v_min = ca.DM(pm.v_min)
    v_max = ca.DM(pm.v_max)
    z_lb = ca.repmat(z_min.T, N + 1, 1)
    z_ub = ca.repmat(z_max.T, N + 1, 1)
    v_lb = ca.repmat(v_min.T, N, 1)
    v_ub = ca.repmat(v_max.T, N, 1)

    # Parameters: initial condition, final condition
    p_z0 = ca.MX.sym("p_z0", 1, pm.n)  # Initial projection Pz(x0) state
    p_zf = ca.MX.sym("p_zf", 1, pm.n)  # Goal state
    p_obs_cx = ca.MX.sym("p_obs_cx", Nobs, 1)  # obstacle center x
    p_obs_cy = ca.MX.sym("p_obs_cy", Nobs, 1)  # obstacle center y
    p_obs_rx = ca.MX.sym("p_obs_rx", Nobs, 1)  # semi-axis in local x
    p_obs_ry = ca.MX.sym("p_obs_ry", Nobs, 1)  # semi-axis in local y
    p_obs_yaw = ca.MX.sym("p_obs_yaw", Nobs, 1)  # rotation angle (radians)

    # Previous solve's first input (for rate limiting)
    p_v_prev = ca.MX.sym("p_v_prev", 1, pm.m) if has_rate_limit else None

    # Make decision variables (2D double integrator)
    z = ca.MX.sym("z", N + 1, pm.n)
    v = ca.MX.sym("v", N, pm.m)

    return z, v, z_lb, z_ub, v_lb, v_ub, p_z0, p_zf, p_obs_cx, p_obs_cy, p_obs_rx, p_obs_ry, p_obs_yaw, p_v_prev


def trajopt_solver(pm, N, Q, R, Nobs, Qf=None, max_iter=1000, debug_filename=None, v_norm_max=None):
    z, v, z_lb, z_ub, v_lb, v_ub, p_z0, p_zf, p_obs_cx, p_obs_cy, p_obs_rx, p_obs_ry, p_obs_yaw, _ = setup_trajopt_solver(pm, N, Nobs)

    if Qf is None:
        Qf = Q
    Q = ca.DM(Q)
    Qf = ca.DM(Qf)

    # Define NLP
    obj = quadratic_objective(z[:-1, :], Q, goal=p_zf) + quadratic_objective(v, R) + quadratic_objective(z[-1, :], Qf, goal=p_zf)
    g_dyn, g_lb_dyn, g_ub_dyn = dynamics_constraint(pm.f, z, v)
    g_obs, g_lb_obs, g_ub_obs = obstacle_constraints(z, p_obs_cx, p_obs_cy, p_obs_rx, p_obs_ry, p_obs_yaw)
    g_ic, g_lb_ic, g_ub_ic = initial_condition_equality_constraint(z, p_z0)

    g = ca.horzcat(g_dyn, g_obs, g_ic)
    g_lb = ca.horzcat(g_lb_dyn, g_lb_obs, g_lb_ic)
    g_ub = ca.horzcat(g_ub_dyn, g_ub_obs, g_ub_ic)

    if v_norm_max is not None:
        g_vnorm, g_lb_vnorm, g_ub_vnorm = velocity_norm_constraint(v, v_norm_max)
        g = ca.horzcat(g, g_vnorm)
        g_lb = ca.horzcat(g_lb, g_lb_vnorm)
        g_ub = ca.horzcat(g_ub, g_ub_vnorm)

    g = g.T
    g_lb = g_lb.T
    g_ub = g_ub.T

    # Generate solver
    x_nlp = ca.vertcat(
        ca.reshape(z, (N + 1) * pm.n, 1),
        ca.reshape(v, N * pm.m, 1),
    )
    lbx = ca.vertcat(
        ca.reshape(z_lb, (N + 1) * pm.n, 1),
        ca.reshape(v_lb, N * pm.m, 1),
    )
    ubx = ca.vertcat(
        ca.reshape(z_ub, (N + 1) * pm.n, 1),
        ca.reshape(v_ub, N * pm.m, 1),
    )
    p_nlp = ca.vertcat(p_z0.T, p_zf.T, p_obs_cx, p_obs_cy, p_obs_rx, p_obs_ry, p_obs_yaw)

    x_cols, g_cols, p_cols = generate_col_names(pm, N, Nobs, x_nlp, g, p_nlp, has_vnorm=v_norm_max is not None)
    nlp_dict = {
        "x": x_nlp,
        "f": obj,
        "g": g,
        "p": p_nlp
    }
    nlp_opts = {
        "ipopt.linear_solver": "mumps",
        "ipopt.sb": "yes",
        "ipopt.max_iter": max_iter,
        "ipopt.tol": 1e-4,
        "ipopt.print_level": 0,
        "print_time": False,
    }

    if debug_filename is not None:
        nlp_opts['iteration_callback'] = SolverCallback('iter_callback', debug_filename, x_cols, g_cols, p_cols, {})

    nlp_solver = ca.nlpsol("trajectory_generator", "ipopt", nlp_dict, nlp_opts)

    solver = {"solver": nlp_solver, "lbg": g_lb, "ubg": g_ub, "lbx": lbx, "ubx": ubx, "g_cols": g_cols, "x_cols": x_cols, "p_cols": p_cols, "callback": nlp_opts.get('iteration_callback')}

    return solver, nlp_dict, nlp_opts


def cbf_trajopt_solver(pm, N, Q, R, Nobs, alpha, delta, Qf=None, max_iter=1000, debug_filename=None, v_norm_max=None, dv_max=None):
    """Trajectory optimization solver with CBF obstacle constraints.

    :param pm: planning model with dynamics, bounds, etc.
    :param N: planning horizon
    :param Q: state cost matrix
    :param R: input cost matrix
    :param Nobs: number of obstacles
    :param alpha: CBF decay rate (0 < alpha < 1)
    :param delta: safety margin (delta >= 0)
    :param Qf: terminal state cost matrix (defaults to Q)
    :param max_iter: max IPOPT iterations
    :param debug_filename: optional file for debug output
    :param v_norm_max: if set, constrains velocity 2-norm
    :param dv_max: per-component input rate limits (m,), None to disable
    """
    has_rate = dv_max is not None
    z, v, z_lb, z_ub, v_lb, v_ub, p_z0, p_zf, p_obs_cx, p_obs_cy, p_obs_rx, p_obs_ry, p_obs_yaw, p_v_prev = setup_trajopt_solver(pm, N, Nobs, has_rate_limit=has_rate)

    if Qf is None:
        Qf = Q
    Q = ca.DM(Q)
    Qf = ca.DM(Qf)

    # Define NLP
    obj = quadratic_objective(z[:-1, :], Q, goal=p_zf) + quadratic_objective(v, R) + quadratic_objective(z[-1, :], Qf, goal=p_zf)
    g_dyn, g_lb_dyn, g_ub_dyn = dynamics_constraint(pm.f, z, v)
    g_cbf, g_lb_cbf, g_ub_cbf = cbf_constraints(z, p_obs_cx, p_obs_cy, p_obs_rx, p_obs_ry, p_obs_yaw, alpha, delta)
    g_ic, g_lb_ic, g_ub_ic = initial_condition_equality_constraint(z, p_z0)

    g = ca.horzcat(g_dyn, g_cbf, g_ic)
    g_lb = ca.horzcat(g_lb_dyn, g_lb_cbf, g_lb_ic)
    g_ub = ca.horzcat(g_ub_dyn, g_ub_cbf, g_ub_ic)

    if v_norm_max is not None:
        g_vnorm, g_lb_vnorm, g_ub_vnorm = velocity_norm_constraint(v, v_norm_max)
        g = ca.horzcat(g, g_vnorm)
        g_lb = ca.horzcat(g_lb, g_lb_vnorm)
        g_ub = ca.horzcat(g_ub, g_ub_vnorm)

    if has_rate:
        g_rate, g_lb_rate, g_ub_rate = input_rate_constraint(v, p_v_prev, dv_max)
        g = ca.horzcat(g, g_rate)
        g_lb = ca.horzcat(g_lb, g_lb_rate)
        g_ub = ca.horzcat(g_ub, g_ub_rate)

    g = g.T
    g_lb = g_lb.T
    g_ub = g_ub.T

    # Generate solver
    x_nlp = ca.vertcat(
        ca.reshape(z, (N + 1) * pm.n, 1),
        ca.reshape(v, N * pm.m, 1),
    )
    lbx = ca.vertcat(
        ca.reshape(z_lb, (N + 1) * pm.n, 1),
        ca.reshape(v_lb, N * pm.m, 1),
    )
    ubx = ca.vertcat(
        ca.reshape(z_ub, (N + 1) * pm.n, 1),
        ca.reshape(v_ub, N * pm.m, 1),
    )
    p_nlp = ca.vertcat(p_z0.T, p_zf.T, p_obs_cx, p_obs_cy, p_obs_rx, p_obs_ry, p_obs_yaw)
    if has_rate:
        p_nlp = ca.vertcat(p_nlp, p_v_prev.T)

    x_cols, g_cols, p_cols = generate_cbf_col_names(pm, N, Nobs, x_nlp, g, p_nlp, has_vnorm=v_norm_max is not None, has_rate_limit=has_rate)
    nlp_dict = {
        "x": x_nlp,
        "f": obj,
        "g": g,
        "p": p_nlp
    }
    nlp_opts = {
        "ipopt.linear_solver": "mumps",
        "ipopt.sb": "yes",
        "ipopt.max_iter": max_iter,
        "ipopt.tol": 1e-4,
        "ipopt.print_level": 0,
        "print_time": False,
    }

    if debug_filename is not None:
        nlp_opts['iteration_callback'] = SolverCallback('iter_callback', debug_filename, x_cols, g_cols, p_cols, {})

    nlp_solver = ca.nlpsol("cbf_trajectory_generator", "ipopt", nlp_dict, nlp_opts)

    solver = {"solver": nlp_solver, "lbg": g_lb, "ubg": g_ub, "lbx": lbx, "ubx": ubx, "g_cols": g_cols, "x_cols": x_cols, "p_cols": p_cols, "callback": nlp_opts.get('iteration_callback')}

    return solver, nlp_dict, nlp_opts


def generate_cbf_col_names(pm, N, Nobs, x, g, p, has_vnorm=False, has_rate_limit=False):
    """Generate column names for CBF solver (N constraints per obstacle, not N+1)."""
    z_str = np.array(["z"] * ((N + 1) * pm.n), dtype='U8').reshape((N + 1, pm.n))
    v_str = np.array(["v"] * (N * pm.m), dtype='U8').reshape((N, pm.m))
    for r in range(z_str.shape[0]):
        for c in range(z_str.shape[1]):
            z_str[r, c] = z_str[r, c] + f"_{r}_{c}"
    for r in range(v_str.shape[0]):
        for c in range(v_str.shape[1]):
            v_str[r, c] = v_str[r, c] + f"_{r}_{c}"
    x_cols = list(np.vstack((
        np.reshape(z_str, ((N + 1) * pm.n, 1)),
        np.reshape(v_str, (N * pm.m, 1))
    )).squeeze())

    g_dyn_cols = []
    for k in range(N):
        g_dyn_cols.extend(["dyn_" + st + f"_{k}" for st in pm.state_names])
    # CBF has N constraints per obstacle (transitions), not N+1 (timesteps)
    g_cbf_cols = []
    for i in range(Nobs):
        g_cbf_cols.extend([f"cbf_{i}_{k}" for k in range(N)])
    g_cols = g_dyn_cols + g_cbf_cols + ["ic_" + st for st in pm.state_names]

    if has_vnorm:
        g_cols += [f"vnorm_{k}" for k in range(N)]

    if has_rate_limit:
        for k in range(N):
            g_cols += [f"rate_{k}_{j}" for j in range(pm.m)]

    obs_cx_lst = [f'obs_{i}_cx' for i in range(Nobs)]
    obs_cy_lst = [f'obs_{i}_cy' for i in range(Nobs)]
    obs_rx_lst = [f'obs_{i}_rx' for i in range(Nobs)]
    obs_ry_lst = [f'obs_{i}_ry' for i in range(Nobs)]
    obs_yaw_lst = [f'obs_{i}_yaw' for i in range(Nobs)]
    p_cols = [f'z_ic_{i}' for i in range(pm.n)] + [f'z_g_{i}' for i in range(pm.n)] + \
             obs_cx_lst + obs_cy_lst + obs_rx_lst + obs_ry_lst + obs_yaw_lst

    if has_rate_limit:
        p_cols += [f'v_prev_{j}' for j in range(pm.m)]

    assert len(x_cols) == x.numel() and len(g_cols) == g.numel() and len(p_cols) == p.numel()
    return x_cols, g_cols, p_cols


def generate_col_names(pm, N, Nobs, x, g, p, H_rev=0, has_vnorm=False):
    z_str = np.array(["z"] * ((N + 1) * pm.n), dtype='U8').reshape((N + 1, pm.n))
    v_str = np.array(["v"] * (N * pm.m), dtype='U8').reshape((N, pm.m))
    for r in range(z_str.shape[0]):
        for c in range(z_str.shape[1]):
            z_str[r, c] = z_str[r, c] + f"_{r}_{c}"
    for r in range(v_str.shape[0]):
        for c in range(v_str.shape[1]):
            v_str[r, c] = v_str[r, c] + f"_{r}_{c}"
    x_cols = list(np.vstack((
        np.reshape(z_str, ((N + 1) * pm.n, 1)),
        np.reshape(v_str, (N * pm.m, 1))
    )).squeeze())

    g_dyn_cols = []
    for k in range(N):
        g_dyn_cols.extend(["dyn_" + st + f"_{k}" for st in pm.state_names])
    g_obs_cols = []
    for i in range(Nobs):
        g_obs_cols.extend([f"obs_{i}_{k}" for k in range(N + 1)])
    g_tube_dyn = [f"tube_{k}" for k in range(N)]
    g_cols = g_dyn_cols + g_obs_cols + ["ic_" + st for st in pm.state_names]

    if has_vnorm:
        g_cols += [f"vnorm_{k}" for k in range(N)]

    if not len(g_cols) == g.numel():
        g_cols.extend(g_tube_dyn)

    obs_cx_lst = [f'obs_{i}_cx' for i in range(Nobs)]
    obs_cy_lst = [f'obs_{i}_cy' for i in range(Nobs)]
    obs_rx_lst = [f'obs_{i}_rx' for i in range(Nobs)]
    obs_ry_lst = [f'obs_{i}_ry' for i in range(Nobs)]
    obs_yaw_lst = [f'obs_{i}_yaw' for i in range(Nobs)]
    p_cols = [f'z_ic_{i}' for i in range(pm.n)] + [f'z_g_{i}' for i in range(pm.n)] + \
             obs_cx_lst + obs_cy_lst + obs_rx_lst + obs_ry_lst + obs_yaw_lst

    if not len(p_cols) == p.numel():
        e_cols = [f"e_{i}" for i in range(H_rev)]
        v_prev_str = np.array(["v_prev"] * (H_rev * pm.m), dtype='U13').reshape((H_rev, pm.m))
        for r in range(v_prev_str.shape[0]):
            for c in range(v_prev_str.shape[1]):
                v_prev_str[r, c] = v_prev_str[r, c] + f"_{r}_{c}"
        p_cols += e_cols + list(np.reshape(v_prev_str, (-1, 1)).squeeze())
    assert len(x_cols) == x.numel() and len(g_cols) == g.numel() and len(p_cols) == p.numel()
    return x_cols, g_cols, p_cols


def init_params(z0, zf, obs, v_prev=None):
    """Packs initial condition, goal, and ellipse obstacle parameters into a single vector.

    :param z0: initial state (n,)
    :param zf: goal state (n,)
    :param obs: obstacle dictionary with keys 'cx', 'cy', 'rx', 'ry', 'yaw'
    :param v_prev: previous solve's first input (m,), None if rate limiting disabled
    :return: parameter vector
    """
    parts = [
        z0[:, None],
        zf[:, None],
        obs['cx'][:, None],
        obs['cy'][:, None],
        obs['rx'][:, None],
        obs['ry'][:, None],
        obs['yaw'][:, None]
    ]
    if v_prev is not None:
        parts.append(v_prev[:, None])
    return np.vstack(parts)


def init_decision_var(z, v):
    N = v.shape[0]
    n = z.shape[1]
    m = v.shape[1]
    x_init = np.vstack([
        np.reshape(z, ((N + 1) * n, 1), order='F'),
        np.reshape(v, (N * m, 1), order='F')
    ])
    return x_init


def extract_solution(sol, N, n, m):
    z_ind = (N + 1) * n
    v_ind = N * m
    z_sol = np.array(sol["x"][:z_ind, :]).reshape((N + 1, n), order='F')
    v_sol = np.array(sol["x"][z_ind:z_ind + v_ind, :]).reshape((N, m), order='F')
    return z_sol, v_sol


def plot_problem(ax, obs, z0, zf):
    """
    Plots elliptical obstacles, start, and goal positions.
    """
    for i in range(len(obs["rx"])):
        cx = obs['cx'][i]
        cy = obs['cy'][i]
        rx = obs['rx'][i]
        ry = obs['ry'][i]
        yaw_deg = np.degrees(obs['yaw'][i])  # matplotlib uses degrees

        # Ellipse takes width/height (full axes, not semi-axes)
        ellipse = Ellipse(
            (cx, cy),
            width=2 * rx,
            height=2 * ry,
            angle=yaw_deg,
            color='r',
            alpha=0.5
        )
        ax.add_patch(ellipse)
    plt.plot(z0[0], z0[1], 'rx')
    plt.plot(zf[0], zf[1], 'go')


def compute_constraint_violation(solver, g):
    g = np.array(g)
    ubg = np.array(solver["ubg"])
    lbg = np.array(solver["lbg"])
    viol = np.maximum(np.maximum(g - ubg, 0), np.maximum(lbg - g, 0))
    return viol


def segment_constraint_violation(g_viol, g_col):
    g_dyn_viol = g_viol[[j for j, s in enumerate(g_col) if "dyn" in s]]
    g_seg = {"Dynamics": g_dyn_viol}
    i = 0
    while i >= 0:
        idx = [j for j, s in enumerate(g_col) if f"obs_{i}" in s]
        if idx:
            g_obs_viol = g_viol[idx]
            g_seg[f"Obstacle {i}"] = g_obs_viol
            i += 1
        else:
            i = -1
    ic_viol = g_viol[[j for j, s in enumerate(g_col) if "ic" in s]]
    g_seg["Initial Condition"] = ic_viol

    tube_idx = [j for j, s in enumerate(g_col) if "tube" in s]
    if tube_idx:
        g_seg["Tube Dynamics"] = g_viol[tube_idx]

    return g_seg


def get_warm_start(warm_start, start, goal, N, planning_model, obs=None, Q=None, R=None, nominal_ws='interpolate'):
    if warm_start == 'start':
        v_init = np.zeros((N, planning_model.m))
        z_init = np.repeat(start[None, :], N + 1, 0)
    elif warm_start == 'goal':
        v_init = np.zeros((N, planning_model.m))
        z_init = np.repeat(goal[None, :], N + 1, 0)
    elif warm_start == 'interpolate':
        z_init = np.outer(np.linspace(0, 1, N + 1), (goal - start)) + start
        v_init = np.diff(z_init, axis=0) / planning_model.dt
    elif warm_start == 'nominal':
        assert obs is not None and Q is not None and R is not None
        sol, solver = solve_nominal(start, goal, obs, planning_model, N, Q, R, warm_start=nominal_ws)
        z_init, v_init = extract_solution(sol, N, planning_model.n, planning_model.m)
    else:
        raise ValueError(f'Warm start {warm_start} not implemented. Must be ic, goal, interpolate, or nominal')

    return z_init, v_init


def solve_nominal(start, goal, obs, planning_model, N, Q, R, warm_start='start', debug_filename=None, v_norm_max=None):
    solver, nlp_dict, nlp_opts = trajopt_solver(planning_model, N, Q, R, len(obs["rx"]), debug_filename=debug_filename, v_norm_max=v_norm_max)

    z_init, v_init = get_warm_start(warm_start, start, goal, N, planning_model)

    params = init_params(start, goal, obs)
    x_init = init_decision_var(z_init, v_init)

    sol = solver["solver"](x0=x_init, p=params, lbg=solver["lbg"], ubg=solver["ubg"], lbx=solver["lbx"],
                           ubx=solver["ubx"])

    if 'iteration_callback' in nlp_opts.keys():
        nlp_opts['iteration_callback'].write_data(solver, params)
    return sol, solver


def solve_nominal_cbf(start, goal, obs, planning_model, N, Q, R, alpha, delta, warm_start='start', debug_filename=None, v_norm_max=None, dv_max=None):
    """Solve trajectory optimization with CBF obstacle constraints.

    :param start: initial state (n,)
    :param goal: goal state (n,)
    :param obs: obstacle dictionary with keys 'cx', 'cy', 'rx', 'ry', 'yaw'
    :param planning_model: planning model with dynamics, bounds, etc.
    :param N: planning horizon
    :param Q: state cost matrix
    :param R: input cost matrix
    :param alpha: CBF decay rate (0 < alpha < 1)
    :param delta: safety margin (delta >= 0)
    :param warm_start: initialization method ('start', 'goal', 'interpolate')
    :param debug_filename: optional file for debug output
    :param v_norm_max: if set, constrains velocity 2-norm
    :param dv_max: per-component input rate limits (m,), None to disable
    """
    solver, nlp_dict, nlp_opts = cbf_trajopt_solver(
        planning_model, N, Q, R, len(obs["rx"]), alpha, delta, debug_filename=debug_filename, v_norm_max=v_norm_max, dv_max=dv_max
    )

    z_init, v_init = get_warm_start(warm_start, start, goal, N, planning_model)

    v_prev = np.zeros(planning_model.m) if dv_max is not None else None
    params = init_params(start, goal, obs, v_prev=v_prev)
    x_init = init_decision_var(z_init, v_init)

    sol = solver["solver"](x0=x_init, p=params, lbg=solver["lbg"], ubg=solver["ubg"], lbx=solver["lbx"],
                           ubx=solver["ubx"])

    if 'iteration_callback' in nlp_opts.keys():
        nlp_opts['iteration_callback'].write_data(solver, params)
    return sol, solver


class SolverCallback(ca.Callback):

    def __init__(self, name, debug_filename, x_cols, g_cols, p_cols, opts):
        ca.Callback.__init__(self)

        self.filename = debug_filename
        self.cols = ["iter"] + x_cols + g_cols
        self.g_cols = g_cols
        self.x_cols = x_cols
        self.p_cols = p_cols
        self.df = pd.DataFrame(columns=self.cols)
        self.it = 0

        self.nx = len(x_cols)
        self.ng = len(g_cols)
        self.np = len(p_cols)

        # Initialize internal objects
        self.construct(name, opts)

    def get_n_in(self):
        return ca.nlpsol_n_out()

    def get_n_out(self):
        return 1

    def get_name_in(self, i):
        return ca.nlpsol_out(i)

    def get_name_out(self, i):
        return "ret"

    def get_sparsity_in(self, i):
        n = ca.nlpsol_out(i)
        if n == 'f':
            return ca.Sparsity.scalar()
        elif n in ('x', 'lam_x'):
            return ca.Sparsity.dense(self.nx)
        elif n in ('g', 'lam_g'):
            return ca.Sparsity.dense(self.ng)
        else:
            return ca.Sparsity(0, 0)

    def eval(self, arg):
        # Create dictionary
        darg = {}
        for (i, s) in enumerate(ca.nlpsol_out()):
            darg[s] = arg[i]

        x = darg['x']
        g = darg['g']

        new_row_df = pd.DataFrame([np.concatenate((np.array([[self.it]]), np.array(x), np.array(g)), axis=0).squeeze()], columns=self.cols)
        self.df = pd.concat([self.df, new_row_df], ignore_index=True)

        self.it += 1

        return [0]

    def write_data(self, solver, params):
        new_cols = ["lb_" + s for s in self.g_cols] + ["ub_" + s for s in self.g_cols] + ["lb_" + s for s in self.x_cols] + ["ub_" + s for s in self.x_cols]
        new_df = pd.DataFrame(np.zeros((self.df.shape[0], len(new_cols))), columns=new_cols)
        self.df = pd.concat([self.df, new_df], axis=1)
        self.df.loc[0, ["lb_" + s for s in self.g_cols]] = np.array(solver["lbg"]).squeeze()
        self.df.loc[0, ["ub_" + s for s in self.g_cols]] = np.array(solver["ubg"]).squeeze()
        self.df.loc[0, ["lb_" + s for s in self.x_cols]] = np.array(solver["lbx"]).squeeze()
        self.df.loc[0, ["ub_" + s for s in self.x_cols]] = np.array(solver["ubx"]).squeeze()
        self.df.loc[0, self.p_cols] = params.squeeze()

        self.df.to_csv(self.filename, index=False)