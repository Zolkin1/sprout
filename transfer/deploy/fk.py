# Copyright 2026 Zachary Olkin. All rights reserved.

"""Forward kinematics of predicted window nodes, for the viewers.

The slim reference window predicts ``base_pose`` and ``joint_pos`` but no ``ee_pos``, so the
target frames are not served -- they are IMPLIED. Recovering them means putting the predicted
base pose and predicted joints on a scratch model and reading the body positions back, which
is what this does.

WHY PINOCCHIO, AND WHY THE ROS URDF. This was MuJoCo on
``transfer/sim/robots/g1/g1_29j.xml`` -- ``FlowMatchingConfig.fk_model_path``, the model the
generator's own FK consistency loss is defined against -- on the argument that FK'ing anything
else would make the drawn feet a SECOND OPINION about where the feet are, which a display
built to expose a discrepancy must not be able to manufacture.

Measured, the two models are not two opinions. Over 200 random configurations the MJCF and the
URDF agree on the four frames the ribbon draws to 4.6e-7 m at the wrists and 5.7e-9 m at the
ankles -- accumulated rounding in the MJCF's literals, five orders of magnitude below a marker
you could see. ``tests/test_fk_parity.py`` pins that, with MuJoCo as a test-only oracle in the
same way ``torch_math`` is for the numpy ports.

What moving buys is one kinematics library on the robot instead of two: the C++ controller
FKs the cameras with pinocchio every tick (``g1_deploy::CameraPoseFk``), so the alternative
was shipping MuJoCo into the container for a drawing tool alone.

WHY IT LIVES HERE RATHER THAN IN transfer/sim. Two consumers -- the MuJoCo sim2sim runner's
viewer ribbon and the ROS marker node -- and only ``transfer/deploy`` is on both their import
paths.

``import pinocchio`` is deliberately INSIDE this module and nothing on the on-robot python
import graph (``deploy.hardware``, ``deploy.gen_tracker``, ``deploy.tracker_session``) reaches
it. That path stays numpy + onnxruntime; this is a drawing tool.
"""

from __future__ import annotations

import pathlib

import numpy as np

from .quat_math_np import quat_apply

#: The robot's ROS URDF -- the same model the C++ controller's camera FK is configured with by
#: default, so a bring-up has exactly one kinematic description on it. Derived from this
#: package's own location rather than counted in ``..`` hops, so moving the package does not
#: silently produce a path that no longer exists.
DEFAULT_FK_MODEL = str(
    pathlib.Path(__file__).resolve().parent.parent
    / "obelisk"
    / "g1_model"
    / "urdf"
    / "g1_29j.urdf"
)


class WindowFK:
    """FK of predicted window nodes, against the robot's URDF.

    One scratch ``pinocchio.Data`` on a privately built model, so evaluating a hypothetical
    pose can never disturb anything else. ``framesForwardKinematics`` is the whole cost -- no
    dynamics, no collision, no contact.

    The model is built FIXED-BASE and the predicted base pose applied afterwards. A floating
    joint would add seven coordinates that the window's ``base_pose`` block already carries in
    its own convention, and converting into pinocchio's would be a second place to get the
    quaternion order wrong.
    """

    def __init__(
        self,
        model_path: str,
        frame_names: list[str],
        isaac_joint_names: list[str],
    ) -> None:
        """Build the model and resolve the frames and the joint permutation once.

        Args:
            model_path: URDF to FK against. ``DEFAULT_FK_MODEL`` unless you know otherwise.
            frame_names: Bodies to report, in the order results should come back. Typically
                ``tracker.onnx``'s ``ee_frame_names``. A name absent from the model is
                tolerated and reported as NaN rather than raising, so one renamed frame does
                not cost the whole drawing.
            isaac_joint_names: The policy's joint order, which is what ``joint_pos`` blocks
                and ``predicted_joint_pos`` are in. The URDF's own order is per limb and
                differs on 27 of the G1's 29, so the permutation is resolved here, once, BY
                NAME -- getting it wrong yields a plausible, wrong skeleton.

        Raises:
            ImportError: If pinocchio is not installed.
            ValueError: If the model's joint count does not match ``isaac_joint_names``, or a
                named joint is absent from it.
        """
        import pinocchio as pin  # noqa: PLC0415  -- see the module docstring

        self._pin = pin
        self.model = pin.buildModelFromUrdf(model_path)
        self._data = self.model.createData()
        self.model_path = model_path
        self.frame_names = list(frame_names)
        self.isaac_joint_names = list(isaac_joint_names)

        # njoints counts the universe joint, which is not one.
        nj = self.model.njoints - 1
        if nj != len(isaac_joint_names):
            raise ValueError(
                f"{model_path} has {nj} actuated joints but the policy names "
                f"{len(isaac_joint_names)}; these must be the same robot"
            )
        missing_joints = [n for n in isaac_joint_names if not self.model.existJointName(n)]
        if missing_joints:
            raise ValueError(
                f"{model_path} joint names do not match the policy's: {missing_joints} missing"
            )
        self._q_index = [
            self.model.joints[self.model.getJointId(n)].idx_q for n in isaac_joint_names
        ]

        self._frame_ids = [
            self.model.getFrameId(n) if self.model.existFrame(n) else -1
            for n in self.frame_names
        ]
        self.missing = [n for n, i in zip(self.frame_names, self._frame_ids) if i < 0]
        # The frame the fixed-base FK is relative to: named by the MODEL, since a URDF's root
        # link is whatever the file puts first (here ``pelvis``, which the MJCF calls
        # ``pelvis_link`` -- the same body under two names, and the reason a hardcoded string
        # would be wrong).
        self._root_id = self.model.getFrameId(self.model.frames[1].name)

    @property
    def resolved(self) -> bool:
        """True if at least one requested frame exists in the model."""
        return any(i >= 0 for i in self._frame_ids)

    def frame_positions(
        self, ref_pos_w: np.ndarray, ref_quat_w: np.ndarray, joint_pos_isaac: np.ndarray
    ) -> np.ndarray:
        """World positions of the target frames, one row per node.

        Args:
            ref_pos_w: ``[N, 3]`` predicted base positions in world (from
                :func:`window_np.base_pose_block_to_world`).
            ref_quat_w: ``[N, 4]`` xyzw predicted base orientations in world.
            joint_pos_isaac: ``[N, nj]`` predicted joint angles in the POLICY's order.

        Returns:
            ``[N, len(frame_names), 3]`` world positions. Rows for frames absent from the
            model are NaN.
        """
        ref_pos_w = np.atleast_2d(np.asarray(ref_pos_w, dtype=np.float64))
        ref_quat_w = np.atleast_2d(np.asarray(ref_quat_w, dtype=np.float64))
        joint_pos_isaac = np.atleast_2d(np.asarray(joint_pos_isaac, dtype=np.float64))
        n = ref_pos_w.shape[0]
        out = np.full((n, len(self._frame_ids), 3), np.nan)
        if not self.resolved:
            return out
        q = self._pin.neutral(self.model)
        for row in range(n):
            for slot, angle in zip(self._q_index, joint_pos_isaac[row]):
                q[slot] = angle
            self._pin.framesForwardKinematics(self.model, self._data, q)
            root_inv = self._data.oMf[self._root_id].inverse()
            for f, fid in enumerate(self._frame_ids):
                if fid < 0:
                    continue
                # The predicted base pose carries it from the root frame into the world. Same
                # composition the MuJoCo version got for free from the free joint.
                local = (root_inv * self._data.oMf[fid]).translation
                out[row, f] = ref_pos_w[row] + quat_apply(ref_quat_w[row], local)
        return out
