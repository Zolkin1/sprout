# Copyright 2026 Zachary Olkin. All rights reserved.

import numpy as np

#: Commanded forward speed for :func:`constant_forward`, in m/s. Named rather than inline
#: because this profile exists to be COMPARED across runs and across simulators -- if the
#: number is buried in the function body, two runs can differ by it without anything saying so.
CONSTANT_FORWARD_SPEED = 0.8


def constant_forward(sim_time):
    """Full forward speed from t=0. No ramp, no lateral, no yaw.

    Harsher than :func:`smooth_ramp` on purpose: the robot spawns standing and is asked for
    the whole command on the first tick, so the generator's very first plan is conditioned on
    a speed the body has not got yet. `--cmd_max_acc` is 0 by default, so nothing smooths it.

    Args:
        sim_time: Simulation time, unused -- the command is constant by construction.

    Returns:
        ``(vx, vy, wz)``.
    """
    del sim_time
    return np.array([CONSTANT_FORWARD_SPEED, 0.0, 0.0])


#: Commanded BACKWARD speed for :func:`constant_backward`, in m/s. Negative by construction.
#: Obelisk's joystick tops out at v_x_scale = 0.8, so this matches what a full-stick reverse
#: actually asks for there -- the point of the profile is to compare the two stacks, and a
#: different magnitude would be one more reason they could differ.
CONSTANT_BACKWARD_SPEED = -0.8


def constant_backward(sim_time):
    """Full REVERSE speed from t=0. No ramp, no lateral, no yaw.

    Exists because walking backwards is the one gait the obelisk stack stumbles on while the
    sim2sim runner does not, and reproducing that needs a scripted command: the comparison has
    to be repeatable, and a human on a stick is not.

    Note the robot cannot SEE where it is going in this profile. Both depth cameras look
    forward and down, so in reverse the generator is conditioned on terrain the robot is
    walking away from. That is true of both simulators equally, which is what makes it a fair
    comparison rather than a confound.

    Args:
        sim_time: Simulation time, unused -- the command is constant by construction.

    Returns:
        ``(vx, vy, wz)``.
    """
    del sim_time
    return np.array([CONSTANT_BACKWARD_SPEED, 0.0, 0.0])


def step_to_max(sim_time):
    """Pass a given input as a function of time."""
    if sim_time > 1.:
        return np.array([0.75,0.0,0.0])
    else:
        return np.array([0.0,0.0,0.0])

def smooth_ramp_running(sim_time):
    """Compute a ramp input over a few seconds up to max speed."""
    RAMP_TIME = 3.0 #1.0

    MAX_SPEED = 1.4 #3.8 #3.2 #2.6 #2.0 #1.4

    slope = MAX_SPEED / RAMP_TIME

    y_vel = 0.0
    if sim_time > 6.0 and sim_time < 8.0:
        y_vel = -0.75
    elif sim_time > 8.0 and sim_time < 10.0:
        y_vel = 0.75

    yaw_vel = 0.0
    if sim_time > 10.0 and sim_time < 12.0:
        yaw_vel = -0.75
    elif sim_time > 12.0 and sim_time < 14.0:
        yaw_vel = 0.75

    return np.array([min(slope * sim_time, MAX_SPEED),y_vel,yaw_vel])

def smooth_ramp(sim_time):
    """Compute a ramp input over a few seconds up to max speed."""
    RAMP_TIME = 2.0

    MAX_SPEED = 0.75

    slope = MAX_SPEED / RAMP_TIME

    return np.array([min(slope * sim_time, MAX_SPEED),0.0,0.0])

def speed_steps_running(sim_time):
    # time_steps = np.array([3, 3, 3, 3, 6, 6])
    # speeds = np.array([0, 0.25, -0.25, 0.5, 0.75, -0.5])
    time_steps = np.array([3, 6, 6, 6])
    speeds = np.array([0, 2.0, 2.5, 1.6])

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




def ramped_speed_steps(sim_time):
    time_steps = np.array([3, 3, 3, 3])
    speeds = np.array([0.0, 0.5, -0.5, 0.75])


    RAMP_TIME = 1.0  # Duration over which to ramp between speeds

    # Compute start times of each interval
    start_times = np.cumsum(np.insert(time_steps[:-1], 0, 0))
    end_times = np.cumsum(time_steps)

    total_time = end_times[-1]
    if sim_time < 0:
        return np.array([speeds[0], 0.0, 0.0])
    elif sim_time >= total_time:
        return np.array([speeds[-1], 0.0, 0.0])

    # Find current interval index
    idx = np.searchsorted(start_times, sim_time, side='right') - 1

    t0 = start_times[idx]
    t1 = end_times[idx]
    v0 = speeds[idx]
    v1 = speeds[idx + 1] if idx + 1 < len(speeds) else speeds[idx]

    # Ramp forward only within the last RAMP_TIME seconds of the interval
    ramp_start = t1 - RAMP_TIME
    if sim_time < ramp_start or RAMP_TIME <= 0:
        vx = v0
    else:
        alpha = (sim_time - ramp_start) / RAMP_TIME
        vx = (1 - alpha) * v0 + alpha * v1

    return np.array([vx, 0.0, 0.0])


# ---------------------------------------------------------------------------------------------
# A flat-ground locomotion demo
# ---------------------------------------------------------------------------------------------

#: The FLAT half of the training library's own command envelope, measured over the 3000
#: ``terrain_type_code == 0`` clips of ``g1_mb_v6g.h5`` (the library the deployed tracker
#: trained on): ``vx [-1.00, +2.50]``, ``vy [-0.59, +0.60]``, ``wz [-0.79, +0.78]``.
#:
#: :func:`flat_locomotion_demo` stays inside it on purpose.  The two stair families carry
#: ``|vy| max = |wz| max = 0.0000`` exactly, so a profile with a side-step or a turn in it is
#: off-distribution on a staircase and in-distribution on flat -- which is why this one is a
#: FLAT demo and says so in its name.
FLAT_ENVELOPE: dict[str, tuple[float, float]] = {
    "vx": (-1.00, 2.50), "vy": (-0.59, 0.60), "wz": (-0.79, 0.78),
}

#: Peak commands for :func:`flat_locomotion_demo`, each inside :data:`FLAT_ENVELOPE` with
#: margin.  Named rather than inline for the same reason as ``CONSTANT_FORWARD_SPEED``: the
#: profile exists to be re-recorded, and two cuts should not differ by a number nobody named.
DEMO_SIDE_VY: float = 0.50
DEMO_RUN_VX: float = 2.20
DEMO_RUN_WZ: float = 0.55
DEMO_WALK_VX: float = 0.80

#: Segment boundaries in seconds, measured from the END of the opening hold.
DEMO_MARKS: tuple[float, ...] = (1.0, 4.0, 7.0, 8.5, 10.5, 15.5, 17.5, 21.5, 23.0, 25.0)

#: Total length of the profile (s); a course carrying it needs at least this much budget.
DEMO_DURATION_S: float = DEMO_MARKS[-1]


def _ramp(t: float, t0: float, t1: float, a: float, b: float) -> float:
    """Smoothstep from ``a`` at ``t0`` to ``b`` at ``t1``, clamped outside.

    Smooth rather than linear because the transitions ARE the demo: a step changes the command
    faster than the body can follow, and what the clip would then show is the tracker catching
    up rather than the gait changing.

    Args:
        t: Time (s).
        t0 / t1: Start and end of the transition (s).
        a / b: Values at ``t0`` and ``t1``.

    Returns:
        The interpolated value.
    """
    if t <= t0:
        return a
    if t >= t1:
        return b
    u = (t - t0) / (t1 - t0)
    return a + (b - a) * u * u * (3.0 - 2.0 * u)


def flat_locomotion_demo(sim_time):
    """Stand, side-step both ways, break into a turning run, settle to a walk, stop.

    The flat-ground counterpart to the terrain clips: one continuous take through every gait
    the pair has on level ground, in an order that shows each transition rather than each
    gait on its own.  Segment by segment, with :data:`DEMO_MARKS` as the boundaries:

    ===============  ==========================================================
    0.0 - 1.0 s      stand
    1.0 - 4.0 s      pure LATERAL, +y (no forward, no yaw)
    4.0 - 7.0 s      pure lateral, -y -- back across its own track
    7.0 - 8.5 s      lateral out, forward in
    8.5 - 15.5 s     RUN at ``DEMO_RUN_VX`` while turning at ``DEMO_RUN_WZ``
    15.5 - 17.5 s    yaw out, speed down
    17.5 - 21.5 s    WALK at ``DEMO_WALK_VX``
    21.5 - 23.0 s    decelerate
    23.0 - 25.0 s    stand
    ===============  ==========================================================

    Every value sits inside :data:`FLAT_ENVELOPE`, the library's own flat-clip range, so the
    demo asks for nothing the generator was not trained to be asked for.

    Args:
        sim_time: Seconds since the command started (i.e. since the end of the opening hold).

    Returns:
        ``(vx, vy, wz)``.
    """
    t = float(sim_time)
    m = DEMO_MARKS
    # Lateral: out to +y, across to -y, then back to zero as the run starts.
    if t < m[1]:
        vy = _ramp(t, m[0], m[0] + 0.6, 0.0, DEMO_SIDE_VY)
    elif t < m[2]:
        vy = _ramp(t, m[1], m[1] + 0.8, DEMO_SIDE_VY, -DEMO_SIDE_VY)
    else:
        vy = _ramp(t, m[2], m[3], -DEMO_SIDE_VY, 0.0)
    # Forward: nothing until the lateral section ends, then run, then walk, then stop.
    if t < m[3]:
        vx = 0.0
    elif t < m[5]:
        vx = _ramp(t, m[3], m[4], 0.0, DEMO_RUN_VX)
    elif t < m[7]:
        vx = _ramp(t, m[5], m[6], DEMO_RUN_VX, DEMO_WALK_VX)
    else:
        vx = _ramp(t, m[7], m[8], DEMO_WALK_VX, 0.0)
    # Yaw: only while running, so the turn is unambiguously part of the run.
    wz = (_ramp(t, m[4] - 0.5, m[4] + 0.5, 0.0, DEMO_RUN_WZ) if t < m[5]
          else _ramp(t, m[5], m[6], DEMO_RUN_WZ, 0.0))
    return np.array([vx, vy, wz])
