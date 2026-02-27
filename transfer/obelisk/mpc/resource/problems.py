import numpy as np

start = np.array([0., 0., 0.])

# Obstacle format:
#   'cx', 'cy' = center coordinates
#   'rx', 'ry' = semi-axis lengths in local x/y (before rotation)
#   'yaw' = rotation angle in radians
problem_dict = {
    "gap": {
        "start": start,
        "goal": np.array([6., 6., 0.]),
        "obs": {
            'cx': np.array([4., 3.]),
            'cy': np.array([0., 6.]),
            'rx': np.array([2., 2.]),
            'ry': np.array([3., 1.]),
            'yaw': np.array([0.2, -0.2])
        },
    },
    "right": {
        "start": start,
        "goal": np.array([30.0, 15.0, 0.0]),
        "obs": {
            'cx': np.array([20., 15.0]),
            'cy': np.array([10., -4.0]),
            'rx': np.array([5.0, 5.0]),
            'ry': np.array([5.0, 5.0]),
            'yaw': np.array([0., 0.])
            # 'cx': np.zeros((0,)),
            # 'cy': np.zeros((0,)),
            # 'rx': np.zeros((0,)),
            # 'ry': np.zeros((0,)),
            # 'yaw': np.zeros((0,))
        },
    },
    "right_wide": {
        "start": start,
        "goal": np.array([2, 0, 0]),
        "obs": {
            'cx': np.array([1., 1.25]),
            'cy': np.array([1., -1.25]),
            'rx': np.array([0.5, 0.5]),
            'ry': np.array([0.5, 0.5]),
            'yaw': np.array([0., 0.])
        },
    },
    "straight2": {
        "start": start,
        "goal": np.array([2., 0., 0.]),
        "obs": {
            'cx': np.zeros((0,)),
            'cy': np.zeros((0,)),
            'rx': np.zeros((0,)),
            'ry': np.zeros((0,)),
            'yaw': np.zeros((0,))
        },
    },
    "straight4": {
        "start": start,
        "goal": np.array([4., 0., 0.]),
        "obs": {
            'cx': np.zeros((0,)),
            'cy': np.zeros((0,)),
            'rx': np.zeros((0,)),
            'ry': np.zeros((0,)),
            'yaw': np.zeros((0,))
        },
    }
}