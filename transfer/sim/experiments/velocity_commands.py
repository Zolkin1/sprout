import numpy as np

def step_to_max(sim_time):
    """Pass a given input as a function of time."""
    if sim_time > 1.:
        return np.array([0.75,0.0,0.0])
    else:
        return np.array([0.0,0.0,0.0])

def smooth_ramp(sim_time):
    """Compute a ramp input over a few seconds up to max speed."""
    RAMP_TIME = 2.0

    MAX_SPEED = 0.75

    slope = MAX_SPEED / RAMP_TIME

    return np.array([min(slope * sim_time, MAX_SPEED),0.0,0.0])

def speed_steps(sim_time):
    # time_steps = np.array([3, 3, 3, 3, 6, 6])
    # speeds = np.array([0, 0.25, -0.25, 0.5, 0.75, -0.5])
    time_steps = np.array([3, 3, 6, 6, 6])
    speeds = np.array([0, 0.5, 0.75, -0.75, 0.75])

    # Compute start times of each interval
    start_times = np.cumsum(np.insert(time_steps[:-1], 0, 0))

    def _get_velocity_at_time(t):
        # Ensure t is within bounds
        if t < 0 or t >= np.sum(time_steps):
            raise ValueError("Time t is out of bounds.")

        # Find the index of the bin t belongs to
        idx = np.searchsorted(start_times, t, side='right') - 1
        return speeds[idx]

    return np.array([_get_velocity_at_time(sim_time), 0, 0])