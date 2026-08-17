# Copyright 2026 Zachary Olkin. All rights reserved.

"""Forward kinematics of predicted window nodes, on the model the FM was trained against.

The slim reference window predicts ``base_pose`` and ``joint_pos`` but no ``ee_pos``, so the
target frames are not served -- they are IMPLIED. Recovering them means putting the predicted
base pose and predicted joints on a scratch model and reading the body positions back, which
is what this does.

WHY MUJOCO, AND WHY THIS PARTICULAR FILE. ``transfer/sim/robots/g1/g1_29j.xml`` is
``FlowMatchingConfig.fk_model_path`` -- the model the generator's own FK consistency loss is
defined against. FK'ing anything else (the ROS URDF under ``transfer/obelisk/g1_model``, say,
which has different link origins and 40 extra fixed joints) would make the drawn feet a
*second opinion* about where the feet are, and a viewer whose job is to expose a discrepancy
must not be able to manufacture one. Using this model makes disagreement impossible by
construction: the frames are exactly where the predicted joints put them.

WHY IT LIVES HERE RATHER THAN IN transfer/sim. Two consumers -- the MuJoCo sim2sim runner's
viewer ribbon and the ROS marker node -- and only ``transfer/deploy`` is on both their import
paths.

``import mujoco`` is deliberately INSIDE this module and nothing on the on-robot import graph
(``deploy.hardware``, ``deploy.gen_tracker``, ``deploy.tracker_session``) reaches it. The
control path stays numpy + onnxruntime; this is a drawing tool.
"""

from __future__ import annotations

import pathlib

import numpy as np

from .gen_tracker import xyzw_to_wxyz
from .tracker_session import convert_joint_order

#: The MJCF ``FlowMatchingConfig.fk_model_path`` names -- the model the generator's FK
#: consistency loss is defined against, and therefore the only correct one to FK its output
#: with. Derived from this package's own location rather than counted in ``..`` hops, so
#: moving the package does not silently produce a path that no longer exists.
DEFAULT_FK_MODEL = str(
    pathlib.Path(__file__).resolve().parent.parent / "sim" / "robots" / "g1" / "g1_29j.xml"
)


class WindowFK:
    """FK of predicted window nodes against the generator's own FK model.

    One scratch ``MjData`` on a privately loaded model, so evaluating a hypothetical pose can
    never disturb a live simulation's state. ``mj_kinematics`` is the whole cost -- no
    dynamics, no collision, no contact -- which is ~30 us per node on a 29-DoF G1.
    """

    def __init__(
        self,
        model_path: str,
        frame_names: list[str],
        isaac_joint_names: list[str],
    ) -> None:
        """Load the model and resolve the frames and the joint permutation once.

        Args:
            model_path: MJCF to FK against. Must be the one the generator's FK loss used --
                ``transfer/sim/robots/g1/g1_29j.xml`` unless you know otherwise.
            frame_names: Bodies to report, in the order results should come back. Typically
                ``tracker.onnx``'s ``ee_frame_names``. A name absent from the model is
                tolerated and reported as NaN rather than raising, so one renamed frame does
                not cost the whole drawing.
            isaac_joint_names: The policy's joint order, which is what ``joint_pos`` blocks
                and ``predicted_joint_pos`` are in. The MJCF's own order is read from the
                model and the permutation resolved here, once.

        Raises:
            ImportError: If mujoco is not installed.
            ValueError: If the model's joint count does not match ``isaac_joint_names``.
        """
        import mujoco  # noqa: PLC0415  -- see the module docstring

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self._data = mujoco.MjData(self.model)
        self.frame_names = list(frame_names)
        self._body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in self.frame_names
        ]
        self.missing = [n for n, i in zip(self.frame_names, self._body_ids) if i < 0]

        # qpos is [pos(3), quat(4, WXYZ), joints(nj)] for a floating base, so the joint count
        # is nq - 7. Checked rather than assumed: a mismatch here would silently write joints
        # into the wrong slots and produce a plausible, wrong skeleton.
        nj = self.model.nq - 7
        if nj != len(isaac_joint_names):
            raise ValueError(
                f"{model_path} has {nj} actuated joints but the policy names "
                f"{len(isaac_joint_names)}; these must be the same robot"
            )
        self.isaac_joint_names = list(isaac_joint_names)
        self.mj_joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in range(self.model.njnt)
        ]
        # Drop the free joint; what remains is qpos[7:] in order.
        self.mj_joint_names = [n for n in self.mj_joint_names if n in set(self.isaac_joint_names)]
        if len(self.mj_joint_names) != nj:
            raise ValueError(
                f"{model_path} joint names do not match the policy's: "
                f"{sorted(set(self.isaac_joint_names) - set(self.mj_joint_names))} missing"
            )

    @property
    def resolved(self) -> bool:
        """True if at least one requested frame exists in the model."""
        return any(i >= 0 for i in self._body_ids)

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
        out = np.full((n, len(self._body_ids), 3), np.nan)
        if not self.resolved:
            return out
        for row in range(n):
            self._data.qpos[0:3] = ref_pos_w[row]
            self._data.qpos[3:7] = xyzw_to_wxyz(ref_quat_w[row])
            self._data.qpos[7:] = convert_joint_order(
                joint_pos_isaac[row], self.isaac_joint_names, self.mj_joint_names
            )
            self._mj.mj_kinematics(self.model, self._data)
            for f, bid in enumerate(self._body_ids):
                if bid >= 0:
                    out[row, f] = self._data.xpos[bid]
        return out
