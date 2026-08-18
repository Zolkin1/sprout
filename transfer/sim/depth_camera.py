# Copyright 2026 Zachary Olkin. All rights reserved.

"""MuJoCo re-implementation of the self-scanning IsaacLab depth cameras.

The training-side sensor is a ``MultiMeshRayCasterCamera``: a pinhole grid of rays cast
from the camera origin against the terrain AND every one of the robot's own link meshes,
reported as ``distance_to_image_plane`` (perpendicular z-depth, not ray length).  This
reproduces it with :func:`mujoco.mj_multiRay` -- all of a pinhole camera's rays share one
origin, which is exactly that function's signature -- and then applies the same
normalisation the observation term does.

Everything geometric comes from the camera contract embedded in the ONNX artifacts
(``mdp.depth_camera_specs``, see ``transfer/deploy/depth_obs.py``), so this file holds no
camera numbers of its own.  What it does hold is the MuJoCo-side convention:

* ``self_scan_group`` (1) -- the robot's VISUAL mesh geoms, one per link, the same STL
  geometry IsaacLab ray-casts.  ``mj_ray`` ignores ``contype``/``conaffinity``, so these
  visual-only geoms are hit.  Group 0 (collision) is deliberately excluded: the foot
  there is a single box, not the visual foot.
* ``terrain_group`` (2) -- the floor and stair geoms, matching
  :meth:`Robot.compute_height_scan` and ``generate_stairs_scene.py``.

Group 1 also holds the hand/palm/finger visual meshes.  IsaacLab casts those too, so the
two agree: ``compare_depth_sim2sim.py`` reports the hand pixels within 0.005 m.  This used
to be a standing ~60/780-pixel delta, because IsaacLab's ``SELF_SCAN_LINKS`` could not see
them -- the stock URDF hung the hand meshes off the wrist as bare ``<visual>`` elements,
which the importer turned into Xforms with a non-identity mesh->body offset that
``MultiMeshRayCaster`` silently fails to intersect.  The fix was to make them real
fixed-joint links (``robot_assets/g1/g1_21j/make_hand_links_urdf.py``), so nothing has to
be filtered out here.
"""

from __future__ import annotations

import os
import sys

import mujoco
import numpy as np

_TRANSFER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TRANSFER not in sys.path:
    sys.path.insert(0, _TRANSFER)

from deploy.depth_obs import (  # noqa: E402
    invalid_depth_value,
    normalize_depth_np,
    pinhole_ray_directions,
)
from deploy.quat_math_np import matrix_from_quat  # noqa: E402

#: MuJoCo geom group holding the robot's per-link visual meshes (self-occlusion).
SELF_SCAN_GROUP = 1
#: MuJoCo geom group holding the terrain (floor + stair boxes).
TERRAIN_GROUP = 2

# ---------------------------------------------------------------------------
# Training-distribution sensor noise (OFF by default; see DepthNoiserNp)
# ---------------------------------------------------------------------------
#
# Numpy counterparts of ``mdp/sensors/noise.py`` and the reset extrinsics jitter in
# ``g1_depth_env_cfg.py``. These are TRAINING-distribution parameters, not part of the
# deployment contract: the export runs on a ``-play`` task where every depth DR event is
# disabled, so there is nothing to embed and the values are restated here. Keep them in
# step with ``_make_depth_noise()`` / ``_EXTRINSICS_*_RANGE``.

#: Mount position tolerance, per axis in metres (mount body frame).
EXTRINSICS_POS_RANGE = (0.01, 0.01, 0.01)
#: Mount orientation tolerance, per axis in radians (about the camera's own axes).
EXTRINSICS_ROT_RANGE = (0.03, 0.03, 0.03)
#: Per-pixel additive noise half-width, in normalised units (= 0.03 m of range).
NOISE_UNIFORM = 0.015
#: Peak of the per-episode plane bias, in normalised units (= 0.1 m).
NOISE_MAX_PLANE_DEVIATION = 0.05
#: Upper bound of the per-episode Gaussian blur sigma, in pixels.
NOISE_SIGMA_MAX = 0.33
#: Per-pixel, per-frame dropout probability.
NOISE_DROPOUT_PROB = 0.02
#: Raw depth (metres) a dropped pixel is written as, before normalisation.
NOISE_DROPOUT_VAL = 0.0


def quat_xyzw_to_matrix(quat_xyzw) -> np.ndarray:
    """Rotation matrix from an ``(x, y, z, w)`` quaternion.

    The camera contract stores the mount rotation in xyzw (IsaacLab's ``OffsetCfg.rot``).
    Delegates to the deploy runtime's own quaternion math, which is parity-tested against
    ``isaaclab.utils.math``, so the cameras cannot drift from the convention the rest of
    the deployment uses.

    Args:
        quat_xyzw: Quaternion as ``(x, y, z, w)``; need not be normalised.

    Returns:
        The ``(3, 3)`` rotation matrix.
    """
    q = np.asarray(quat_xyzw, dtype=np.float64)
    return matrix_from_quat(q / np.linalg.norm(q))


def _euler_xyz_to_matrix(euler) -> np.ndarray:
    """Rotation matrix from XYZ Euler angles, matching ``quat_from_euler_xyz``.

    Args:
        euler: Roll, pitch, yaw in radians.

    Returns:
        The ``(3, 3)`` rotation matrix.
    """
    quat_wxyz = np.zeros(4, dtype=np.float64)
    mujoco.mju_euler2Quat(quat_wxyz, np.asarray(euler, dtype=np.float64), "xyz")
    return matrix_from_quat(np.roll(quat_wxyz, -1))   # wxyz -> xyzw


class _Camera:
    """One camera's immutable geometry plus its per-frame scratch buffers."""

    def __init__(self, mj_model, spec: dict):
        """Resolve a contract entry against the model and preallocate the cast buffers.

        Args:
            mj_model: The loaded MuJoCo model.
            spec: One camera entry from the depth contract.

        Raises:
            ValueError: If the contract's mount body is not in the model.
        """
        self.obs_term = str(spec["obs_term"])
        self.sensor = str(spec.get("sensor", self.obs_term))
        self.width, self.height = int(spec["width"]), int(spec["height"])
        self.max_distance = float(spec["max_distance"])
        self.norm = {
            key: float(spec[key])
            for key in ("offset", "scale", "min_dist", "max_dist", "min_clip_val", "max_clip_val")
        }

        self.body_name = str(spec["mount_body"])
        self.body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, self.body_name)
        if self.body_id < 0:
            raise ValueError(
                f"camera '{self.sensor}' mounts on body '{self.body_name}', which this "
                "MuJoCo model does not have."
            )
        self.invalid_value = invalid_depth_value(spec)
        self.offset_pos = np.asarray(spec["offset_pos"], dtype=np.float64)
        self.offset_rot = quat_xyzw_to_matrix(spec["offset_quat_xyzw"])
        # Kept so extrinsics jitter perturbs the nominal mount rather than compounding.
        self.nominal_offset_pos = self.offset_pos.copy()
        self.nominal_offset_rot = self.offset_rot.copy()

        self.dirs_cam = pinhole_ray_directions(
            self.width, self.height,
            float(spec["fx"]), float(spec["fy"]), float(spec["cx"]), float(spec["cy"]),
        )
        # cos(angle off the optical axis) per pixel: the ray-length -> z-depth projection,
        # since the camera-frame x component of a unit direction IS that cosine.
        self.cos_theta = self.dirs_cam[:, 0].copy()

        self.num_rays = self.dirs_cam.shape[0]
        self._dirs_w = np.zeros((self.num_rays, 3), dtype=np.float64)
        self._dist = np.zeros(self.num_rays, dtype=np.float64)
        self._geomid = np.zeros(self.num_rays, dtype=np.int32)
        # Filled by MujocoDepthCameras.compute: the hit geom per ray and which rays
        # resolved. Callers that need hit provenance (a per-body census, a hand mask) read
        # these instead of re-casting, so there is one cast convention.
        self.hit = np.zeros(self.num_rays, dtype=bool)

    @property
    def geom_ids(self) -> np.ndarray:
        """Per-ray hit geom id from the last cast (meaningless where ``hit`` is False)."""
        return self._geomid

    def pose(self, mj_data) -> tuple[np.ndarray, np.ndarray]:
        """World position and orientation of the camera at the current state.

        Args:
            mj_data: The MuJoCo data whose forward kinematics are current.

        Returns:
            ``(position (3,), rotation (3, 3))`` -- x = optical axis, y = left, z = up.
        """
        body_rot = mj_data.xmat[self.body_id].reshape(3, 3)
        return body_rot @ self.offset_pos + mj_data.xpos[self.body_id], body_rot @ self.offset_rot


class MujocoDepthCameras:
    """The policy's depth cameras, cast against a MuJoCo scene.

    Built once from the camera contract carried in the ONNX artifacts, then called every
    control step.  Both IsaacLab cameras update at the control rate (50 Hz), so there is
    no staleness or latency model to reproduce -- one cast per step is exact.
    """

    def __init__(
        self,
        mj_model,
        spec: list[dict],
        self_scan_group: int = SELF_SCAN_GROUP,
        terrain_group: int = TERRAIN_GROUP,
    ):
        """Resolve every camera in the contract against the model.

        Args:
            mj_model: The loaded MuJoCo model.
            spec: The depth-camera contract (``policy.get_depth_camera_spec()``).
            self_scan_group: Geom group of the robot's visual meshes, so the robot
                occludes itself as it does in training.
            terrain_group: Geom group of the terrain.

        Raises:
            ValueError: If the contract is empty, or names a body the model lacks.
        """
        if not spec:
            raise ValueError("MujocoDepthCameras needs a non-empty depth-camera contract.")
        self.cameras = [_Camera(mj_model, cam) for cam in spec]
        self.obs_term_names = [cam.obs_term for cam in self.cameras]
        self.noiser: DepthNoiserNp | None = None
        self._model = mj_model
        # A 6-byte mask over geom groups: terrain (so the ground is visible) + the robot's
        # visual meshes (so it occludes itself). bodyexclude=-1 excludes nothing; the
        # robot's COLLISION geoms are skipped by living in group 0.
        self._geomgroup = np.zeros(6, dtype=np.uint8)
        self._geomgroup[int(terrain_group)] = 1
        self._geomgroup[int(self_scan_group)] = 1
        # Geoms in the cast groups, for the cutoff computation in :meth:`_cutoff_for`.
        self._cast_geoms = np.array(
            [g for g in range(mj_model.ngeom) if self._geomgroup[mj_model.geom_group[g]]],
            dtype=np.int32,
        )
        self._num_rays = self.cameras[0].num_rays
        if any(cam.num_rays != self._num_rays for cam in self.cameras):
            raise ValueError(
                "depth cameras have differing pixel counts "
                f"({[cam.num_rays for cam in self.cameras]}); compute() returns them as one "
                "array (and the FM stacks them on one token axis), so they must match."
            )

    def _cutoff_for(self, mj_data, pos: np.ndarray) -> float:
        """A ``mj_multiRay`` cutoff that provably culls NOTHING in the cast groups.

        THIS EXISTS BECAUSE OF A REAL BUG, so it must not be "optimised" back:
        ``cutoff`` culls a candidate geom when ``dist(ray_origin, geom_origin) - rbound``
        exceeds it -- a distance to the geom's ORIGIN, not to its surface. **A plane has
        ``rbound = 0`` and the ground plane's origin is the world origin**, so passing the
        camera's ``max_distance`` (8 m near / 12 m far) silently discarded the entire floor
        once the robot walked that far from spawn. Symptom: every ground pixel in the near
        camera reads "no reading" while self-occlusion pixels stay perfect, so the policy is
        handed a depth image with no terrain in it -- and the near camera loses it first,
        because its cutoff is the smaller one. Measured: ground hits vanish at |x| >= 8.5 m
        (near) and >= 13 m (far).

        ``max_distance`` is enforced AFTER the cast, by rejecting ``dist > max_distance``
        (see :meth:`compute`), so the cutoff was never what made the range correct -- it was
        only ever a speed optimisation, and it buys about 1%.

        Bound: culling needs ``dist_i - rbound_i > cutoff``; with ``rbound_i >= 0`` and
        ``cutoff >= max_i dist_i``, that can never hold for any geom type, plane included.

        Args:
            mj_data: MuJoCo data with current forward kinematics.
            pos: The ray origin (camera position), world frame.

        Returns:
            The cutoff to pass to ``mj_multiRay``.
        """
        if self._cast_geoms.size == 0:
            return float(np.max([c.max_distance for c in self.cameras]))
        d = np.linalg.norm(mj_data.geom_xpos[self._cast_geoms] - pos, axis=1).max()
        return float(d) + float(np.max([c.max_distance for c in self.cameras]))

    def enable_noise(self, rng: np.random.Generator) -> "DepthNoiserNp":
        """Attach the training sensor-noise model to these cameras.

        Constructed here rather than passed in because the noiser needs the resolved
        per-camera geometry.

        Args:
            rng: Random generator for the per-episode and per-frame draws.

        Returns:
            The attached noiser (so the caller can :meth:`DepthNoiserNp.reset` it).
        """
        self.noiser = DepthNoiserNp(self.cameras, rng=rng)
        return self.noiser

    def jitter_extrinsics(
        self,
        rng: np.random.Generator,
        pos_range=EXTRINSICS_POS_RANGE,
        rot_range=EXTRINSICS_ROT_RANGE,
    ) -> None:
        """Perturb every camera's mount pose, as the training reset event does.

        Mount tolerance and calibration error: a real camera is never exactly at its
        nominal offset, and the policy was trained across this spread.  Perturbs around
        the NOMINAL pose rather than the last sample, so repeated calls cannot random-walk
        the mount away (mirroring ``mdp.events.depth.randomize_camera_extrinsics``).  The
        rotation is applied on the right, i.e. about the camera's own axes.

        Args:
            rng: Random generator.
            pos_range: Per-axis position half-width in metres, in the mount body frame.
            rot_range: Per-axis rotation half-width in radians (XYZ Euler).
        """
        for cam in self.cameras:
            cam.offset_pos = cam.nominal_offset_pos + rng.uniform(-1.0, 1.0, 3) * np.asarray(
                pos_range, dtype=np.float64
            )
            euler = rng.uniform(-1.0, 1.0, 3) * np.asarray(rot_range, dtype=np.float64)
            cam.offset_rot = cam.nominal_offset_rot @ _euler_xyz_to_matrix(euler)

    def mirror_extrinsics_from(self, other: "MujocoDepthCameras") -> None:
        """Adopt another instance's sampled mount poses, camera by camera.

        For the case where the generator samples the same physical lenses on its own grid:
        one camera cannot be in two places, so the two casters must share the mount
        perturbation. Mirrors ``mdp.events.depth.randomize_camera_extrinsics`` with a
        ``sensor_cfgs`` list on the IsaacLab side, so sim2sim reproduces the same coupling.

        Only the mount is copied -- the per-frame noise draws stay independent, since they
        cannot be shared across differing pixel counts anyway.

        Args:
            other: The caster whose sampled mounts to adopt (typically the tracker's).

        Raises:
            ValueError: On a camera-count mismatch, or if a corresponding pair does not
                share a NOMINAL mount -- which would mean they are not one physical camera
                and must not share a perturbation.
        """
        if len(self.cameras) != len(other.cameras):
            raise ValueError(
                f"cannot mirror mounts: {len(self.cameras)} camera(s) here against "
                f"{len(other.cameras)} there."
            )
        for mine, theirs in zip(self.cameras, other.cameras):
            if not (np.allclose(mine.nominal_offset_pos, theirs.nominal_offset_pos, atol=1e-9)
                    and np.allclose(mine.nominal_offset_rot, theirs.nominal_offset_rot,
                                    atol=1e-9)):
                raise ValueError(
                    f"'{mine.sensor}' and '{theirs.sensor}' do not share a nominal mount, so they "
                    "are not two samplings of one physical camera."
                )
            mine.offset_pos = theirs.offset_pos.copy()
            mine.offset_rot = theirs.offset_rot.copy()

    def compute(self, mj_data, want_hits: bool = False):
        """Cast every camera and return the normalised images (and optionally hit points).

        Args:
            mj_data: MuJoCo data with current forward kinematics (post ``mj_step`` /
                ``mj_forward``).
            want_hits: Also return the world-space hit point per pixel. Off by default:
                the gather that builds it costs ~50 us/step and only the viewer wants it.

        Returns:
            ``images (T, H*W)``, or ``(images, hit_points_world (T, H*W, 3))`` when
            ``want_hits``.  Images are normalised exactly as the training observation term
            normalises them, row-major with ``v = 0`` the top row.  Hit points are ``nan``
            where the ray missed.  Each camera's per-ray geom ids stay in
            :attr:`_Camera.geom_ids` for callers that need hit provenance.
        """
        images = np.empty((len(self.cameras), self._num_rays), dtype=np.float64)
        hits = (np.full((len(self.cameras), self._num_rays, 3), np.nan, dtype=np.float64)
                if want_hits else None)
        for i, cam in enumerate(self.cameras):
            pos, rot = cam.pose(mj_data)
            np.matmul(cam.dirs_cam, rot.T, out=cam._dirs_w)
            mujoco.mj_multiRay(
                self._model, mj_data, pos, cam._dirs_w.ravel(), self._geomgroup,
                1,          # flg_static: the terrain lives on the static worldbody
                -1,         # bodyexclude: nothing (self-occlusion is the point)
                cam._geomid, cam._dist, None, cam.num_rays,
                # NOT max_distance -- that silently culls the ground plane once the robot
                # walks that far from the world origin. See :meth:`_cutoff_for`.
                self._cutoff_for(mj_data, pos),
            )
            # The cast does not bound the ray, so a hit beyond the camera's range can be
            # reported; IsaacLab cuts the ray itself at max_distance, so reject those
            # explicitly. THIS is what enforces the range. A miss reports exactly
            # max_distance, matching depth_clipping_behavior="max" -- the sentinel the
            # normalisation treats as "no reading".
            hit = (cam._dist >= 0.0) & (cam._dist <= cam.max_distance)
            depth_z = np.where(
                hit, np.minimum(cam._dist * cam.cos_theta, cam.max_distance), cam.max_distance
            )
            images[i] = normalize_depth_np(depth_z, **cam.norm)
            cam.hit = hit
            if want_hits:
                hits[i][hit] = pos + cam._dist[hit, None] * cam._dirs_w[hit]
        if self.noiser is not None:
            images = self.noiser(images)
        return (images, hits) if want_hits else images


class DepthNoiserNp:
    """Per-episode + per-frame depth corruption, in normalised units.

    Applies the training noise in the same ORDER as
    ``robot_rl...mdp.sensors.noise.DepthNoiser``: per-episode plane bias, per-pixel
    uniform noise, a 3x3 Gaussian blur (zero-padded, so the 1-pixel border is pulled
    toward 0), then dropout.  The blur sigma and the plane are drawn once per
    :meth:`reset`, matching the ``resample_depth_noise`` reset event.
    """

    def __init__(self, cameras: list[_Camera], rng: np.random.Generator | None = None):
        """Precompute the per-camera pixel grids the plane bias and blur need.

        Args:
            cameras: The cameras whose images will be corrupted (one noise state each,
                as in training, where every camera gets its own noiser).
            rng: Random generator (a fresh default when omitted).
        """
        self._rng = rng if rng is not None else np.random.default_rng()
        # One record per camera, so the per-episode state cannot fall out of alignment
        # with the geometry it belongs to.
        self._cams = []
        for cam in cameras:
            ys = np.linspace(-1.0, 1.0, cam.height)
            xs = np.linspace(-1.0, 1.0, cam.width)
            grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
            self._cams.append({
                "shape": (cam.height, cam.width),
                "grid_x": grid_x.reshape(-1),
                "grid_y": grid_y.reshape(-1),
                "dropout": (NOISE_DROPOUT_VAL - cam.norm["offset"]) * cam.norm["scale"],
                "plane": None,
                "kernel": None,
            })
        self.reset()

    def reset(self) -> None:
        """Re-draw the per-episode plane bias and blur kernel for every camera."""
        # Squared distances of the 3x3 taps from the centre.
        offsets = np.array([2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0])
        for cam in self._cams:
            coeffs = self._rng.standard_normal(3)
            # Scale so the plane's peak over the [-1, 1] image square is the drawn
            # deviation: max |a x + b y + c| = |a| + |b| + |c| there.
            peak = self._rng.uniform(0.0, NOISE_MAX_PLANE_DEVIATION)
            coeffs *= peak / max(np.abs(coeffs).sum(), 1e-12)
            cam["plane"] = coeffs[0] * cam["grid_x"] + coeffs[1] * cam["grid_y"] + coeffs[2]

            sigma = abs(self._rng.uniform(0.0, NOISE_SIGMA_MAX))
            kernel = np.exp(-offsets / (2.0 * max(sigma ** 2, 1e-12)))
            cam["kernel"] = (kernel / kernel.sum()).reshape(3, 3)

    def __call__(self, images: np.ndarray) -> np.ndarray:
        """Corrupt normalised images ``(T, H*W)`` in the training order.

        Args:
            images: Normalised depth, one row per camera.

        Returns:
            The corrupted images, same shape.
        """
        out = np.empty_like(images, dtype=np.float64)
        for i, cam in enumerate(self._cams):
            row = images[i] + cam["plane"]
            row += self._rng.uniform(-NOISE_UNIFORM, NOISE_UNIFORM, row.shape)
            row = _blur3x3(row.reshape(cam["shape"]), cam["kernel"]).reshape(-1)
            dropped = self._rng.random(row.shape) < NOISE_DROPOUT_PROB
            row[dropped] = cam["dropout"]
            out[i] = row
        return out


def _blur3x3(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Zero-padded 3x3 correlation, matching the training blur's ``padding=1``.

    Args:
        image: A single ``(H, W)`` image.
        kernel: The normalised ``(3, 3)`` kernel.

    Returns:
        The blurred image, same shape.
    """
    padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2), dtype=np.float64)
    padded[1:-1, 1:-1] = image
    out = np.zeros_like(image)
    for dy in range(3):
        for dx in range(3):
            out += kernel[dy, dx] * padded[dy:dy + image.shape[0], dx:dx + image.shape[1]]
    return out
