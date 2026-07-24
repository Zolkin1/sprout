# Copyright 2026 Zachary Olkin. All rights reserved.

"""Motion Bricks MuJoCo ``qpos`` layout + joint reindex to the repo's G1 joint set.

Motion Bricks emits a 36-D MuJoCo ``qpos`` (``assets/skeletons/g1/g1.xml``):
``[root_pos(3), root_quat_wxyz(4), joints(29)]`` (Z-up; the agent reorders the root quaternion to
wxyz before we ever see it). The 29 joints of that skeleton happen to be in the *same order* as this
repo's :data:`motion_gen.motion_matching.features.G1_JOINT_NAMES`, but we never rely on that -- the
raw file carries its own ``joint_names`` and we reindex *by name* so a future skeleton change surfaces
as a clear error (or a correct permutation) instead of a silent misalignment.
"""

from __future__ import annotations

import numpy as np

# ``qpos`` slices for the Motion Bricks G1 MuJoCo model.
MB_QPOS_ROOT_POS = slice(0, 3)     # world XYZ (Z-up)
MB_QPOS_ROOT_QUAT = slice(3, 7)    # wxyz (already reordered by the agent)
MB_QPOS_JOINTS = slice(7, 36)      # 29 hinge joint angles


def split_qpos(qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split ``[T, 36]`` Motion Bricks qpos into ``(root_pos[T,3], root_quat_wxyz[T,4], joints[T,29])``."""
    qpos = np.asarray(qpos)
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"expected qpos [T, 36], got {qpos.shape}")
    return (
        qpos[:, MB_QPOS_ROOT_POS],
        qpos[:, MB_QPOS_ROOT_QUAT],
        qpos[:, MB_QPOS_JOINTS],
    )


def reindex_to(src_names: list[str], dst_names: list[str] | None = None) -> np.ndarray:
    """Return int indices ``idx`` such that ``src_joints[:, idx]`` is in ``dst_names`` order.

    ``dst_names`` defaults to the repo's :data:`~motion_gen.motion_matching.features.G1_JOINT_NAMES`.
    Raises ``ValueError`` if any destination joint is absent from ``src_names`` (a genuine skeleton
    mismatch we want to fail loudly on, not paper over).
    """
    if dst_names is None:
        from ..motion_matching.features import G1_JOINT_NAMES

        dst_names = G1_JOINT_NAMES
    src_index = {name: i for i, name in enumerate(src_names)}
    missing = [n for n in dst_names if n not in src_index]
    if missing:
        raise ValueError(f"source joints missing {len(missing)} of the target set, e.g. {missing[:5]}")
    return np.asarray([src_index[n] for n in dst_names], dtype=np.int64)
