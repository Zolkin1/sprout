import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import matplotlib.cm as cm


class CasadiRomDynamics(ABC):
    """
    Abstract class for Reduced order Model Dynamics
    """
    n: int  # Dimension of state
    m: int  # Dimension of input
    state_names: list
    input_names: list

    def __init__(self, dt, z_min, z_max, v_min, v_max):
        """
        Common constructor functionality
        :param dt: time discretization
        :param z_min: lower state bound
        :param z_max: upper state bound
        :param v_min: lower input bound
        :param v_max: upper input bound
        :param backend: 'casadi' for when using dynamics for a casadi optimization program,
               'numpy' for use with numpy arrays
        """
        self.dt = dt
        self.v_min = v_min
        self.v_max = v_max
        self.z_min = z_min
        self.z_max = z_max
        self.state_names = []
        self.input_names = []

    @abstractmethod
    def f(self, z, v):
        """
        Dynamics function
        :param z: current state
        :param v: input
        :return: next state
        """
        raise NotImplementedError

    @staticmethod
    def plot_spacial(ax, xt, c=None):
        """
        Plots the x, y spatial trajectory on the given axes with a color gradient to indicate time series.
        :param ax: axes on which to plot
        :param xt: state trajectory
        :param c: color/line type
        """
        N = xt.shape[0]
        colors = cm.viridis(np.linspace(0, 1, N))  # Use the 'viridis' colormap

        # Plot segments with color gradient
        if c is None:
            for i in range(N - 1):
                ax.plot(xt[i:i + 2, 0], xt[i:i + 2, 1], color=colors[i])
            scatter = ax.scatter(xt[:, 0], xt[:, 1], c=np.linspace(0, 1, N), cmap='viridis', s=10,
                                 edgecolor='none')  # Plot points for better visibility

            # Add color bar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Time')
        else:
            ax.plot(xt[:, 0], xt[:, 1], c)

    def plot_ts(self, axs, xt, ut):
        """
        Plots states and inputs over time
        :param axs: size 2 array of axes on which to plot states (first) and inputs (second)
        :param xt: state trajectory
        :param ut: input trajectory
        """
        N = xt.shape[0]
        ts = np.linspace(0, N * self.dt, N)
        axs[0].plot(ts, xt)
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('State')

        axs[1].plot(ts[:-1], ut)
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Input')


class CasadiUnicycle(CasadiRomDynamics):
    n = 3   # [x, y, theta]
    m = 3   # [v_par, v_perp, omega]

    def __init__(self, dt, z_min, z_max, v_min, v_max):
        super().__init__(dt, z_min, z_max, v_min, v_max)
        self.dt = dt
        self.state_names = ['x', 'y', 'theta']
        self.input_names = ['vpar', 'vperp', 'omega']

    def f(self, x, u):
        xm, ym, thm = x[0], x[1], x[2]
        vpar, vperp, w = u[0], u[1], u[2]
        xp = xm + self.dt * (vpar * ca.cos(thm) - vperp * ca.sin(thm))
        yp = ym + self.dt * (vpar * ca.sin(thm) + vperp * ca.cos(thm))
        thp = thm + self.dt * w
        return ca.vertcat(xp, yp, thp)

    def plot_ts(self, axs, xt, ut):
        super().plot_ts(axs, xt, ut)
        axs[0].legend(self.state_names)
        axs[1].legend(self.input_names)