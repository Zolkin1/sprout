# Copyright 2026 Zachary Olkin. All rights reserved.

"""Flow-matching transformer for predicting future robot states.

A conditional flow-matching (rectified-flow) encoder-decoder transformer that,
given a history window (relative-frame robot state + velocity command + perceptive
input), generates a future trajectory of robot states in the current root frame.
"""

from .config import FlowMatchingConfig
from .dataset import FutureStateDataset, build_train_val, episode_windows, resolve_term_slice
from .flow import flow_matching_loss, sample_flow_batch, sample_trajectory
from .model import FlowMatchingTransformer
from .normalizer import Normalizer
from .transforms import (
    matrix_to_rot6d,
    orientation_slice,
    quat_to_rot6d,
    relative_state,
    relative_state_from_window,
    rot6d_to_matrix,
    rot6d_to_quat,
    state_to_absolute,
    target_state_dim,
)

__all__ = [
    "FlowMatchingConfig",
    "FlowMatchingTransformer",
    "FutureStateDataset",
    "Normalizer",
    "build_train_val",
    "episode_windows",
    "resolve_term_slice",
    "flow_matching_loss",
    "sample_flow_batch",
    "sample_trajectory",
    "matrix_to_rot6d",
    "rot6d_to_matrix",
    "quat_to_rot6d",
    "rot6d_to_quat",
    "relative_state",
    "relative_state_from_window",
    "state_to_absolute",
    "orientation_slice",
    "target_state_dim",
]
