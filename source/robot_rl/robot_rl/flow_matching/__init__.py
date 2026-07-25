# Copyright 2026 Zachary Olkin. All rights reserved.

"""Flow-matching transformer for predicting future robot states.

A conditional flow-matching (rectified-flow) PARC-style single transformer that
denoises a window of relative-frame state tokens conditioned on the most-recent
measured states (dropped in as clean tokens, inpainting) and a strided
commanded-velocity profile, generating a future trajectory in the current root frame.
"""

from .config import FlowMatchingConfig
from .dataset import (
    FutureStateDataset,
    build_train_val,
    resolve_term_slice,
    snap_windows,
)
from motion_gen.fk import FKChain
from .flow import (
    fk_consistency_terms,
    flow_matching_loss,
    frame_consistency_terms,
    joint_vel_consistency_term,
    sample_flow_batch,
    sample_trajectory,
    sample_trajectory_grad,
    terrain_penetration_term,
)
from .model import FlowMatchingTransformer
from .normalizer import Normalizer
from .terrain import FlatTerrain, box_sole_points
from .transforms import (
    FRAME_BLOCK_DIM,
    frame_block_start,
    matrix_from_quat,
    matrix_to_rot6d,
    orientation_slice,
    orientation_slices,
    quat_to_rot6d,
    relative_state,
    relative_state_from_window,
    rot6d_to_matrix,
    rot6d_to_quat,
    so3_log,
    state_to_absolute,
    yaw_quat,
    target_state_dim,
)

__all__ = [
    "FlowMatchingConfig",
    "FlowMatchingTransformer",
    "FutureStateDataset",
    "snap_windows",
    "Normalizer",
    "build_train_val",
    "resolve_term_slice",
    "FKChain",
    "FlatTerrain",
    "box_sole_points",
    "flow_matching_loss",
    "frame_consistency_terms",
    "joint_vel_consistency_term",
    "fk_consistency_terms",
    "terrain_penetration_term",
    "sample_flow_batch",
    "sample_trajectory",
    "sample_trajectory_grad",
    "matrix_from_quat",
    "matrix_to_rot6d",
    "rot6d_to_matrix",
    "quat_to_rot6d",
    "rot6d_to_quat",
    "so3_log",
    "yaw_quat",
    "relative_state",
    "relative_state_from_window",
    "state_to_absolute",
    "orientation_slice",
    "orientation_slices",
    "frame_block_start",
    "FRAME_BLOCK_DIM",
    "target_state_dim",
]
