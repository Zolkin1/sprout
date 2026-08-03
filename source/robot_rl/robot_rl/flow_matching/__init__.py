# Copyright 2026 Zachary Olkin. All rights reserved.

"""Flow-matching transformer for predicting future robot states.

A conditional flow-matching (rectified-flow) PARC-style single transformer that
denoises a window of relative-frame state tokens conditioned on the most-recent
measured states (dropped in as clean tokens, inpainting) and a strided
commanded-velocity profile, generating a future trajectory in the current root frame.
"""

from .config import EXTERO_SOURCES, FlowMatchingConfig
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
)
from .model import FlowMatchingTransformer
from .normalizer import Normalizer
from .transforms import (
    EE_BLOCK_DIM,
    ROOT_BLOCK_DIM,
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
    so3_exp,
    so3_log,
    state_to_absolute,
    yaw_quat,
    target_state_dim,
)
from .window import (
    BASE_POSE_CONVENTION,
    REFERENCE_BLOCK_KEYS,
    REFERENCE_WINDOW_BLOCK_NAMES,
    REFERENCE_WINDOW_NOISE,
    expand_window_blocks,
    reanchor_window,
    resolve_reference_blocks,
    split_window,
    window_block_slices,
    window_blocks,
    window_dim,
    window_to_world,
    world_to_window,
)

__all__ = [
    "FlowMatchingConfig",
    "EXTERO_SOURCES",
    "FlowMatchingTransformer",
    "FutureStateDataset",
    "snap_windows",
    "Normalizer",
    "build_train_val",
    "resolve_term_slice",
    "FKChain",
    "flow_matching_loss",
    "frame_consistency_terms",
    "joint_vel_consistency_term",
    "fk_consistency_terms",
    "sample_flow_batch",
    "sample_trajectory",
    "sample_trajectory_grad",
    "matrix_from_quat",
    "matrix_to_rot6d",
    "rot6d_to_matrix",
    "quat_to_rot6d",
    "rot6d_to_quat",
    "so3_exp",
    "so3_log",
    "yaw_quat",
    "relative_state",
    "relative_state_from_window",
    "state_to_absolute",
    "orientation_slice",
    "orientation_slices",
    "frame_block_start",
    "ROOT_BLOCK_DIM",
    "EE_BLOCK_DIM",
    "target_state_dim",
    "world_to_window",
    "BASE_POSE_CONVENTION",
    "REFERENCE_BLOCK_KEYS",
    "REFERENCE_WINDOW_BLOCK_NAMES",
    "REFERENCE_WINDOW_NOISE",
    "expand_window_blocks",
    "reanchor_window",
    "resolve_reference_blocks",
    "window_to_world",
    "window_blocks",
    "window_block_slices",
    "window_dim",
    "split_window",
]
