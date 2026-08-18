# Copyright 2026 Zachary Olkin. All rights reserved.

"""The 5-channel exteroception ROW, as the deploy runtime assembles it.

``test_depth_unproject`` covers the geometry -- where a pixel lands, and that the numpy and
torch unprojections agree. What it does not cover is the layer above: turning a pair of camera
images into the flat rows the generator eats. That assembly decides the CHANNEL ORDER, which
plane is which, and how validity is found, and every one of those is silent when wrong -- the
shapes stay right, the numbers stay plausible, and a convolution just reads a scrambled image.

It was also, until now, untestable: the assembly lived inside ``GenTrackerPolicy``, which
needs two ONNX artifacts to construct. That is why this seam had no coverage while the
training side had plenty, and it is the seam a train/deploy skew hides in -- the clip-band
mismatch (trained at +-5.575, deployed at +-2.0) lived exactly here.

So the parity test at the bottom is the point of the file: the same contract, image and pose
through the DEPLOY builder and through the TRAINING one must give the same row, bit for bit
modulo dtype.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from deploy.depth_obs import (  # noqa: E402
    depth_xyz_rows_np,
    invalid_depth_mask,
    normalize_depth_np,
)

_H, _W = 4, 5
_K = _H * _W


def _cam(sensor: str = "near", pitch_sin: float = 0.5373, pitch_cos: float = 0.84339) -> dict:
    """One camera contract entry, pitched down like the real pair."""
    return {
        "obs_term": sensor, "sensor": sensor, "mount_body": "torso_link",
        "offset_pos": [0.11, 0.0, 0.10],
        "offset_quat_xyzw": [0.0, pitch_sin, 0.0, pitch_cos],
        "width": _W, "height": _H, "fx": 2.5, "fy": 2.2, "cx": _W / 2, "cy": _H / 2,
        "max_distance": 8.0, "min_dist": 0.10, "max_dist": 7.9,
        "offset": 0.75, "scale": 0.5, "min_clip_val": 0.0, "max_clip_val": 0.0,
    }


def _pose(z: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """A camera pose in the yaw frame: forward of and above the root, looking level."""
    return np.array([0.15, 0.0, z]), np.eye(3)


def _image(n_cam: int = 2, seed: int = 0) -> np.ndarray:
    """Normalised depth for ``n_cam`` cameras, with a few genuine misses mixed in."""
    rng = np.random.default_rng(seed)
    metres = rng.uniform(0.5, 5.0, size=(n_cam, _K))
    metres[:, ::7] = np.inf                          # rays that hit nothing
    return np.stack([normalize_depth_np(metres[i], **{
        k: _cam()[k] for k in ("offset", "scale", "min_dist", "max_dist",
                               "min_clip_val", "max_clip_val")}) for i in range(n_cam)])


def test_the_row_is_channel_major_and_keeps_the_depth_plane():
    """Channel 0 must be the image itself, untouched, with xyz after it -- not interleaved.

    ``images_from_flat`` folds a row as ``(C, H, W)``. An interleaved row folds into a
    scrambled image with every shape still correct, so this is the assertion standing between
    a convolution and nonsense.
    """
    spec, image = [_cam("near"), _cam("far")], _image()
    rows = depth_xyz_rows_np(spec, image, [_pose(), _pose(0.5)], channels=4)

    assert rows.shape == (2, 4 * _K)
    folded = rows.reshape(2, 4, _K)
    assert np.array_equal(folded[:, 0], image)              # depth plane preserved exactly
    assert np.isfinite(rows).all()
    assert not np.allclose(folded[:, 0, :], folded[:, 1, :])   # xyz is not a copy of depth


def test_the_fifth_channel_is_validity_found_by_range():
    """Validity comes from the RANGE test, not from equality with the sentinel.

    ``DepthNoiser`` runs after normalisation and its 3x3 blur leaves a geometric miss NEAR the
    sentinel rather than on it, so equality would call blurred misses valid and unproject them
    onto the robot itself. The same asymmetry the height scan does not have.
    """
    spec, image = [_cam("near")], _image(n_cam=1)
    rows = depth_xyz_rows_np(spec, image, [_pose()], channels=5)

    assert rows.shape == (1, 5 * _K)
    validity = rows.reshape(1, 5, _K)[0, 4]
    assert set(np.unique(validity)) <= {0.0, 1.0}
    np.testing.assert_array_equal(validity == 0.0, invalid_depth_mask(spec[0], image[0]))
    assert 0 < validity.sum() < _K, "the fixture must carry both real readings and misses"


def test_a_pose_per_camera_is_required():
    """A missing pose is a silently misplaced cloud, so it is refused rather than broadcast."""
    with pytest.raises(ValueError, match="one pose per camera"):
        depth_xyz_rows_np([_cam("near"), _cam("far")], _image(), [_pose()], channels=4)


def test_only_four_or_five_channels_are_assembled():
    """1 channel needs no widening and anything else is not a layout this code emits."""
    for bad in (1, 3, 6):
        with pytest.raises(ValueError, match="4 .* or 5"):
            depth_xyz_rows_np([_cam()], _image(n_cam=1), [_pose()], channels=bad)


def test_the_deploy_row_matches_the_training_row():
    """THE parity test: the deploy builder and the dataset's must produce the same row.

    One is numpy in ``transfer/deploy``, the other torch in ``robot_rl.flow_matching.dataset``,
    and a training run uses the second while MuJoCo and hardware use the first. They are
    separate implementations on purpose (``robot_rl`` must not import ``transfer``), which is
    exactly why they have to be compared rather than assumed.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("isaaclab")
    from robot_rl.flow_matching.dataset import FutureStateDataset as DS  # noqa: PLC0415

    spec = [_cam("near"), _cam("far")]
    image = _image()
    poses = [_pose(), _pose(0.5)]

    deploy = depth_xyz_rows_np(spec, image, poses, channels=5)

    torch_poses = [(torch.as_tensor(p, dtype=torch.float64)[None, :],
                    torch.as_tensor(r, dtype=torch.float64)[None, :, :]) for p, r in poses]
    training = DS._depth_xyz_rows(
        torch.as_tensor(image.reshape(1, -1), dtype=torch.float64),
        spec, torch_poses, n_sensors=len(spec), sensor_dim=_K, channels=5,
    )

    np.testing.assert_allclose(
        training.numpy().reshape(len(spec), 5 * _K), deploy, atol=1e-9,
        err_msg="the deploy and training row builders disagree -- a model would be trained on "
                "one layout and served the other",
    )
