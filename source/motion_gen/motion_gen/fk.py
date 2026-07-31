# Copyright 2026 Zachary Olkin. All rights reserved.

"""Differentiable, batched forward kinematics for the FK consistency loss.

A pure-torch FK chain parsed once from a MuJoCo MJCF (``g1_29j.xml``, see
``features.DEFAULT_MJCF``), so it needs **no** external kinematics dependency and
can be validated exactly against ``mujoco.mj_kinematics``. It runs on the model's
device (GPU), is differentiable in the joint angles, and is batched over arbitrary
leading dims — so it can be evaluated inside the training loss with gradients flowing
back to the predicted joint positions/velocities.

Scope/assumptions (true for the G1 21-DoF model): every actuated joint is a single
``hinge`` with anchor ``pos="0 0 0"``; bodies are placed by ``pos`` (+ optional
``quat`` in wxyz); the only ``free`` joint is the floating base (skipped — FK is in
the base/pelvis frame). Anything else (slide joints, multiple joints per body,
``euler``/``axisangle`` body orientations) raises, so a silent wrong-FK can't happen.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import torch


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix ``(3, 3)`` from a wxyz quaternion (MuJoCo convention)."""
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _vec(s: str | None, default: tuple[float, ...]) -> np.ndarray:
    """Parse a whitespace-separated float vector attribute, or a default."""
    if s is None:
        return np.array(default, dtype=np.float64)
    return np.array([float(v) for v in s.split()], dtype=np.float64)


def _euler_to_matrix(e: np.ndarray) -> np.ndarray:
    """MuJoCo intrinsic-xyz euler (radian) -> rotation matrix ``R = Rx·Ry·Rz``."""
    cx, cy, cz = np.cos(e)
    sx, sy, sz = np.sin(e)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rx @ ry @ rz


def _axisangle_to_matrix(v: np.ndarray) -> np.ndarray:
    """MuJoCo axisangle ``[x y z angle]`` (radian) -> rotation matrix (Rodrigues)."""
    axis, ang = v[:3], v[3]
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=np.float64)
    return np.eye(3) + np.sin(ang) * k + (1.0 - np.cos(ang)) * (k @ k)


def _body_rotation(body: ET.Element) -> np.ndarray | None:
    """Static rotation of an MJCF body from quat/euler/axisangle, or ``None`` if the
    orientation uses an unsupported spec (``xyaxes``/``zaxis``) -- raised only later if
    the body is actually on a requested FK chain."""
    if body.get("quat") is not None:
        return _quat_wxyz_to_matrix(_vec(body.get("quat"), (1.0, 0.0, 0.0, 0.0)))
    if body.get("euler") is not None:
        return _euler_to_matrix(_vec(body.get("euler"), (0.0, 0.0, 0.0)))
    if body.get("axisangle") is not None:
        return _axisangle_to_matrix(_vec(body.get("axisangle"), (0.0, 0.0, 1.0, 0.0)))
    if body.get("xyaxes") is not None or body.get("zaxis") is not None:
        return None  # unsupported; only an error if this body lies on an FK chain
    return np.eye(3, dtype=np.float64)


class _Body:
    """One MJCF body: static transform relative to its parent + its (optional) hinge."""

    def __init__(self, name: str, parent: str | None, pos: np.ndarray, rot: np.ndarray,
                 joint_name: str | None, axis: np.ndarray | None, jpos: np.ndarray | None):
        self.name = name
        self.parent = parent
        self.pos = pos          # (3,) static translation in the parent frame
        self.rot = rot          # (3, 3) static rotation in the parent frame
        self.joint_name = joint_name  # actuated hinge name, or None (fixed body)
        self.axis = axis        # (3,) hinge axis in the body frame, or None
        self.jpos = jpos        # (3,) hinge anchor in the body frame, or None


def _parse_mjcf(path: str) -> dict[str, _Body]:
    """Parse the MJCF ``<worldbody>`` into a ``name -> _Body`` tree.

    Raises:
        NotImplementedError: For body/joint features the FK does not model.
        ValueError: On a malformed tree.
    """
    root = ET.parse(path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{path}: no <worldbody>.")
    bodies: dict[str, _Body] = {}

    def walk(elem: ET.Element, parent: str | None) -> None:
        for body in elem.findall("body"):
            name = body.get("name")
            if name is None:
                raise ValueError(f"{path}: <body> without a name under {parent!r}.")
            pos = _vec(body.get("pos"), (0.0, 0.0, 0.0))
            rot = _body_rotation(body)  # may be None (unsupported spec; off-chain bodies are fine)
            joints = body.findall("joint")
            jn = ax = jp = None
            hinges = [j for j in joints if (j.get("type") or "hinge") != "free"]
            if any((j.get("type") or "hinge") == "free" for j in joints):
                pass  # floating base — no actuated DoF on this body
            if len(hinges) > 1:
                raise NotImplementedError(f"{path}: body {name!r} has >1 actuated joint.")
            if hinges:
                j = hinges[0]
                jtype = j.get("type") or "hinge"
                if jtype not in ("hinge", "revolute"):
                    raise NotImplementedError(f"{path}: joint {j.get('name')!r} type={jtype!r} unsupported.")
                jn = j.get("name")
                ax = _vec(j.get("axis"), (0.0, 0.0, 1.0))
                jp = _vec(j.get("pos"), (0.0, 0.0, 0.0))
            bodies[name] = _Body(name, parent, pos, rot, jn, ax, jp)
            walk(body, name)

    walk(worldbody, None)
    return bodies


class FKChain:
    """Batched, differentiable FK from a base link to a set of end-effector frames.

    Build once (CPU parse), then :meth:`to` the model device. :meth:`pose_and_jacobian`
    maps joint angles to each EE's base-frame pose and geometric Jacobian.
    """

    def __init__(
        self,
        mjcf_path: str,
        joint_names: list[str],
        frame_names: list[str],
        base_link: str = "pelvis_link",
        dtype: torch.dtype = torch.float32,
    ):
        """Parse ``mjcf_path`` and pre-resolve the base->frame chains.

        Args:
            mjcf_path: MuJoCo XML defining the kinematics (e.g. ``g1_21j.xml``).
            joint_names: FM joint order (the columns of the predicted ``joint_pos``).
            frame_names: End-effector body names to compute FK for (``target_frame_names``).
            base_link: Root body the FK is expressed relative to (the FM root frame).
            dtype: Working dtype for the constant transforms.
        """
        bodies = _parse_mjcf(mjcf_path)
        if base_link not in bodies:
            raise ValueError(f"{mjcf_path}: base link {base_link!r} not found.")
        self.nj = len(joint_names)
        self.frame_names = list(frame_names)
        self._dtype = dtype
        self._jname_to_idx = {n: i for i, n in enumerate(joint_names)}

        # Per frame: the ordered list of bodies from (base child) down to the EE, each a
        # (R_static, t_static, joint-axis|None, joint-anchor|None, q_index|None) step.
        self._chains: list[list[dict]] = []
        self._n_matched = 0
        for f in frame_names:
            if f not in bodies:
                raise ValueError(f"{mjcf_path}: end-effector frame {f!r} not a body.")
            path: list[_Body] = []
            cur = bodies[f]
            while cur is not None and cur.name != base_link:
                path.append(cur)
                cur = bodies.get(cur.parent) if cur.parent is not None else None
            if cur is None or cur.name != base_link:
                raise ValueError(f"{mjcf_path}: {f!r} is not a descendant of {base_link!r}.")
            path.reverse()  # base-child ... EE
            steps = []
            for b in path:
                if b.rot is None:
                    raise NotImplementedError(
                        f"body {b.name!r} on the {f!r} chain uses an unsupported orientation "
                        f"spec (xyaxes/zaxis); add support in _body_rotation."
                    )
                # A body's joint is an FM DoF only if its name is in the FM joint set.
                # MJCF joints absent from the FM set (e.g. the 3-DoF wrists in this
                # 27-DoF model that the 21-DoF FM does not predict) are held FIXED at 0
                # -- a fixed body with only its static transform -- matching the bench,
                # which leaves those qpos at 0.
                is_fm = b.joint_name is not None and b.joint_name in self._jname_to_idx
                if is_fm:
                    self._n_matched += 1
                axis = b.axis if is_fm else None
                jpos = b.jpos if is_fm else None
                steps.append({
                    "R": torch.tensor(b.rot, dtype=dtype),
                    "t": torch.tensor(b.pos, dtype=dtype),
                    "axis": None if axis is None else torch.tensor(axis / (np.linalg.norm(axis) + 1e-12), dtype=dtype),
                    "jpos": None if jpos is None else torch.tensor(jpos, dtype=dtype),
                    "q_idx": self._jname_to_idx[b.joint_name] if is_fm else None,
                })
            self._chains.append(steps)
        if self._n_matched == 0:
            raise ValueError(
                f"No FM joints matched any {base_link!r}->frame chain in {mjcf_path}; "
                f"check that the FM joint names match the MJCF joint names."
            )
        self.device = torch.device("cpu")

    def to(self, device: torch.device | str) -> "FKChain":
        """Move every constant transform to ``device`` (in place); returns self."""
        self.device = torch.device(device)
        for steps in self._chains:
            for s in steps:
                s["R"] = s["R"].to(self.device)
                s["t"] = s["t"].to(self.device)
                if s["axis"] is not None:
                    s["axis"] = s["axis"].to(self.device)
                    s["jpos"] = s["jpos"].to(self.device)
        return self

    @staticmethod
    def _axis_rot(axis: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Rodrigues rotation about a fixed unit ``axis`` by per-sample angles ``q (M,)``."""
        kx, ky, kz = axis
        zero = torch.zeros_like(kx)
        kmat = torch.stack([
            torch.stack([zero, -kz, ky]),
            torch.stack([kz, zero, -kx]),
            torch.stack([-ky, kx, zero]),
        ])  # (3, 3) skew of the axis
        s = torch.sin(q)[:, None, None]
        c = torch.cos(q)[:, None, None]
        eye = torch.eye(3, dtype=axis.dtype, device=axis.device)
        return eye + s * kmat + (1.0 - c) * (kmat @ kmat)  # (M, 3, 3)

    def pose_and_jacobian(
        self, joint_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward kinematics + geometric Jacobian for every EE frame.

        Args:
            joint_pos: Joint angles in the FM joint order, shape ``(M, nj)``.

        Returns:
            ``(pos, rot, jv, jw)``: EE positions ``(M, K, 3)`` and rotation matrices
            ``(M, K, 3, 3)`` in the base frame, and the base-frame geometric Jacobians
            ``jv``/``jw`` ``(M, K, 3, nj)`` mapping joint velocities to EE linear /
            angular velocity.
        """
        m = joint_pos.shape[0]
        dev, dt = joint_pos.device, joint_pos.dtype
        pos_out, rot_out, jv_out, jw_out = [], [], [], []
        for steps in self._chains:
            r_cum = torch.eye(3, dtype=dt, device=dev).expand(m, 3, 3).contiguous()
            t_cum = torch.zeros(m, 3, dtype=dt, device=dev)
            axes_b: list[torch.Tensor] = []      # base-frame joint axes (M, 3)
            anchors_b: list[torch.Tensor] = []   # base-frame joint anchors (M, 3)
            q_idxs: list[int] = []
            for s in steps:
                # Apply the body static transform -> "pre-joint" cumulative frame.
                r_pre = r_cum @ s["R"]
                t_pre = torch.einsum("mij,j->mi", r_cum, s["t"]) + t_cum
                if s["q_idx"] is not None:
                    axis_b = torch.einsum("mij,j->mi", r_pre, s["axis"])             # (M, 3)
                    anchor_b = torch.einsum("mij,j->mi", r_pre, s["jpos"]) + t_pre   # (M, 3)
                    axes_b.append(axis_b)
                    anchors_b.append(anchor_b)
                    q_idxs.append(s["q_idx"])
                    rj = self._axis_rot(s["axis"], joint_pos[:, s["q_idx"]])         # (M, 3, 3)
                    # anchor jpos is the rotation center; with the rotation applied:
                    t_cum = t_pre + torch.einsum("mij,j->mi", r_pre, s["jpos"]) \
                        - torch.einsum("mij,mjk,k->mi", r_pre, rj, s["jpos"])
                    r_cum = r_pre @ rj
                else:
                    r_cum, t_cum = r_pre, t_pre
            p_ee = t_cum  # (M, 3)
            jv = torch.zeros(m, 3, self.nj, dtype=dt, device=dev)
            jw = torch.zeros(m, 3, self.nj, dtype=dt, device=dev)
            for axis_b, anchor_b, qi in zip(axes_b, anchors_b, q_idxs):
                jv[:, :, qi] = torch.linalg.cross(axis_b, p_ee - anchor_b)
                jw[:, :, qi] = axis_b
            pos_out.append(p_ee)
            rot_out.append(r_cum)
            jv_out.append(jv)
            jw_out.append(jw)
        return (
            torch.stack(pos_out, dim=1),   # (M, K, 3)
            torch.stack(rot_out, dim=1),   # (M, K, 3, 3)
            torch.stack(jv_out, dim=1),    # (M, K, 3, nj)
            torch.stack(jw_out, dim=1),    # (M, K, 3, nj)
        )
