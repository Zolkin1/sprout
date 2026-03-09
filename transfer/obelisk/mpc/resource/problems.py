import numpy as np

start = np.array([0., 0., 0.])

# Obstacle format:
#   'cx', 'cy' = center coordinates
#   'rx', 'ry' = semi-axis lengths in local x/y (before rotation)
#   'yaw' = rotation angle in radians
problem_dict = {
    "walk_forward": {
        "start": start,
        "goal": np.array([4., 2., 0.]),
        "obs": {
            'cx': np.array([2.0,]),
            'cy': np.array([0.5,]),
            'rx': np.array([1.0,]),
            'ry': np.array([1.0,]),
            'yaw': np.array([0.0,])
        },
    },
    "right": {
        "start": start,
        "goal": np.array([4.0, 0.0, 0.0]), #np.array([30.0, 15.0, 0.0]),
        "obs": {
            # 'cx': np.array([20., 15.0]),
            # 'cy': np.array([10., -4.0]),
            # 'rx': np.array([5.0, 5.0]),
            # 'ry': np.array([5.0, 5.0]),
            # 'yaw': np.array([0., 0.])
            'cx': np.zeros((0,)),
            'cy': np.zeros((0,)),
            'rx': np.zeros((0,)),
            'ry': np.zeros((0,)),
            'yaw': np.zeros((0,))
        },
    },
    "breezeway": {
        "start": start,
        "goal": np.array([27.7, 1.75, 0.0]),
        "obs": {
          'cx': np.array([2.8, 10.7, 20.7, 13.0, 13.0]),
          'cy': np.array([2.55, -0.45, 3.95, -2.25, 5.75]),
          'rx': np.array([0.5, 1.8, 1.8, 15, 15,]),
          'ry': np.array([0.5, 2.9, 2.9, 1.5, 1.5]),
          'yaw': np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        },
    },
    "breezeway_v2": {
        "start": start,
        "goal": np.array([27.7, 0.0, 0.0]),
        "obs": {
          'cx': np.array([2.8, 10.7, 14.5, 13.0, 13.0]),
          'cy': np.array([2.55, -0.45, 1.55, -2.25, 5.75]),
          'rx': np.array([0.5, 1.3, 2.4, 15, 15,]),
          'ry': np.array([0.5, 2.4, 1.3, 1.0, 1.0]),
          'yaw': np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
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
