# Copyright 2026 Zachary Olkin. All rights reserved.

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from transfer.sim.rl_policy import RLPolicy

import mujoco
import numpy as np
import pygame
from scipy.spatial.transform import Rotation


class Robot:
    def __init__(self, robot_name: str, scene_name: str, joystick_scaling: np.ndarray,
                 input_function: Callable[[float], np.array] = None, use_pd: bool =False, gains: Dict[str, float]=None, rng=None):
        """Initialize the robot with its model and data."""
        # ``g1_perceptive`` is the CURRENT robot and the default everywhere: mode_11 motors,
        # passive rubber hands, a compute backpack and two camera mounts, generated from the
        # URDF by ``make_g1_mjcf.py``. Its scenes are ``g1_perceptive_{basic,terrain,tile*}
        # _scene.xml``. ``g1_29j`` is the pre-mode_11 29-DOF model, kept because it is still
        # the FK reference in some older provenance -- its SCENES are gone, so naming it here
        # gets a missing-file error rather than a silently different robot.
        if robot_name not in ("g1_21j", "g1_21j_M4", "g1_21j_compute", "g1_29j",
                              "g1_perceptive"):
            raise ValueError(
                f"Invalid robot name {robot_name!r}! Supported: g1_21j, g1_21j_M4, "
                f"g1_21j_compute, g1_29j, g1_perceptive."
            )

        self.robot_name = robot_name
        self.scene_name = scene_name
        self.mj_model, self.mj_data = self._get_model_data()
        self.commanded_vel = np.zeros(3)  # Store commanded velocity
        self.input_function = input_function
        self.rng = rng
        self.joystick_scaling = joystick_scaling
        self.joint_names = self.get_joint_names()
        self.gains = gains
        print(f"Mujoco joint names: {self.joint_names}")

        body_name = "torso_link"
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        self.torso_ipos = self.mj_model.body_ipos[body_id]

        self.use_pd = use_pd
        self._policy_gains: tuple | None = None   # set by apply_policy_gains

        if self.input_function is None:
            # Initialize joystick
            pygame.init()
            pygame.joystick.init()
            joystick_count = pygame.joystick.get_count()
            if joystick_count < 1:
                print("No joystick detected, using initial command from config instead.")
                self.joystick = None
            else:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                print(f"Using controller: {self.joystick.get_name()}")

    def _get_model_data(self):
        """Create the mj model and data from the given robot."""
        file_name = f"{self.robot_name}_{self.scene_name}.xml"
        relative_path = f"robots/g1/{file_name}"
        path = os.path.join(os.getcwd(), relative_path)
        print(f"Trying to load the xml at {path}")
        mj_model = mujoco.MjModel.from_xml_path(path)
        mj_data = mujoco.MjData(mj_model)

        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)

        return mj_model, mj_data

    def reset_robot(self):
        """Resets the robot (re-applying any policy gains, since the model is rebuilt)."""
        self.mj_model, self.mj_data = self._get_model_data()
        if self._policy_gains is not None:
            self.apply_policy_gains(*self._policy_gains)

    def add_base_mass(self, added_mass):
        """Add mass to the robot base."""
        body_name = "torso_link"
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        self.mj_model.body_mass[body_id] += added_mass

        print(f"Adjusting the mass of the {body_name} by adding {added_mass}.")

    def apply_force_disturbance(self, force_disturbance):
        """Apply the force_disturbance to the robot base."""
        body_name = "torso_link"
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        self.mj_data.xfrc_applied[body_id] = force_disturbance

    def randomize_torso_mass_pos(self, max_movement: np.array):
        """Shift where the torso mass is randomly."""
        body_name = "torso_link"
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        if self.rng is not None:
            rand_vec = self.rng.uniform(low=-max_movement, high=max_movement)
        else:
            rand_vec = np.random.uniform(low=-max_movement, high=max_movement)

        self.mj_model.body_ipos[body_id] += rand_vec

        print(f"Adjusting ipos of torso by: {rand_vec}.")

        return rand_vec

    def reset_torso_mass_pos(self):
        body_name = "torso_link"
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        self.mj_model.body_ipos[body_id] += self.torso_ipos

    def scale_torso_mass(self, scale: float) -> None:
        """Scale the torso mass by the given factor.

        Args:
            scale: Multiplicative factor for the torso mass.
        """
        body_name = "torso_link"
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        self.mj_model.body_mass[body_id] *= scale
        print(f"Scaling torso mass by {scale}. New mass: {self.mj_model.body_mass[body_id]}")

    def set_torso_mass_pos_offset(self, offset: np.ndarray) -> None:
        """Set the torso COM position offset to an exact value.

        Args:
            offset: 3D offset vector [x, y, z] in meters.
        """
        body_name = "torso_link"
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        self.mj_model.body_ipos[body_id] = self.torso_ipos + offset
        print(f"Setting torso COM offset to: {offset}")

    def scale_kd_gains(self, scale: float) -> None:
        """Scale all actuator kd gains by the given factor.

        Args:
            scale: Multiplicative factor for kd gains.
        """
        for i in range(self.mj_model.nu):
            self.mj_model.actuator_biasprm[i, 2] *= scale
        print(f"Scaling all kd gains by {scale}")

    def get_joystick_command(self, policy):
        """Get velocity commands from joystick."""
        des_vel = np.zeros(3)
        if self.joystick is not None:
            for event in pygame.event.get():
                pass
            # Left stick: control vx, vy (2D plane), right stick X-axis: vyaw
            vy = -(self.joystick_scaling[0]*self.joystick.get_axis(0)) * policy.get_max_vy()
            vx = -(self.joystick_scaling[0]*self.joystick.get_axis(1)) * policy.get_max_vx()
            vyaw = -(self.joystick_scaling[0]*self.joystick.get_axis(3)) * policy.get_max_vyaw()

            des_vel[0] = vx
            des_vel[1] = vy
            des_vel[2] = vyaw
        else:
            des_vel = np.array([0.5, 0.0, 0.0])
        self.commanded_vel = des_vel  # Store the commanded velocity
        # print(f"Commanded velocity: {des_vel}")
        return des_vel

    def create_observation(self, policy, height_map=None, sensor_pos=None):
        """Create observation for the policy."""
        qpos = self.mj_data.qpos
        qvel = self.mj_data.qvel
        sim_time = self.mj_data.time
        # pg = self.get_projected_gravity(qpos[3:7])
        if self.input_function is None:
            self.commanded_vel = self.get_joystick_command(policy)
        else:
            self.commanded_vel = self.input_function(sim_time)

        if self.use_pd:
            self.pd_control(qpos, qvel)

        obs = policy.create_obs(
            qpos[:7],
            qvel[3:6],
            qpos[7:],
            qvel[6:],
            sim_time,
            self.commanded_vel,
            self.joint_names
        )

        # Log what the policy actually saw (clipped, ramped if max_acc is set).
        self.commanded_vel = policy.vel_target_b.copy()

        return obs

    def get_log_data(self, policy, obs, action):
        """Get data to be logged."""
        torques = []
        for j in range(self.mj_model.nu):
            torques.append(self.mj_data.actuator_force[j])

        left_ankle_pos = self.mj_data.sensor("left_ankle_pos").data
        right_ankle_pos = self.mj_data.sensor("right_ankle_pos").data

        log = [
            self.mj_data.time,
            *self.mj_data.qpos.tolist(),
            *self.get_local_vel().tolist(),
            *self.mj_data.qvel[3:].tolist(),
            *obs[0, :].numpy().tolist(),
            *action.tolist(),
            *torques,
            *left_ankle_pos.tolist(),
            *right_ankle_pos.tolist(),
            *self.commanded_vel.tolist(),  # Add commanded velocity to log data
        ]
        return log

    def get_local_vel(self):
        """Convert the global floating base velocity to the local frame."""
        # Convert to scipy format [x, y, z, w]
        q_scipy = np.array([self.mj_data.qpos[4], self.mj_data.qpos[5], self.mj_data.qpos[6], self.mj_data.qpos[3]])

        # Create rotation object
        r = Rotation.from_quat(q_scipy)

        # Inverse rotate to get local velocity
        v_local = r.inv().apply(self.mj_data.qvel[:3])

        return v_local

    def get_base_world_velocity(self, body_name: str = "pelvis_link") -> tuple[np.ndarray, np.ndarray]:
        """World-frame linear and angular velocity of a base LINK frame origin.

        Returns the velocity of ``body_name``'s frame origin (``xpos``) in the world
        frame, matching the flow-matching generator's training convention (Isaac Lab
        ``root_link_{lin,ang}_vel_w`` is the link-frame-origin twist). Computed from
        the body Jacobian (``mj_jacBody`` is evaluated at the frame origin):
        ``v = Jp @ qvel``, ``omega = Jr @ qvel``.

        NOTE: ``mj_objectVelocity`` reports the twist at the body **CoM**, which
        differs from the frame origin by ``omega x r_com`` as soon as the body
        rotates -- a silent error that is zero at standstill but grows with angular
        velocity (and corrupts the FM's velocity input during walking/running).

        Args:
            body_name: MuJoCo body whose link-frame velocity to read (the FM/CLF base).

        Returns:
            ``(lin_vel_w, ang_vel_w)`` each a length-3 numpy array in the world frame.
        """
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"Body '{body_name}' not found in MuJoCo model.")
        jacp = np.zeros((3, self.mj_model.nv))
        jacr = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jacBody(self.mj_model, self.mj_data, jacp, jacr, body_id)
        lin_vel_w = jacp @ self.mj_data.qvel
        ang_vel_w = jacr @ self.mj_data.qvel
        return lin_vel_w, ang_vel_w

    def get_body_world_states(self, body_names: list[str]) -> np.ndarray:
        """World-frame pose + link twist for a list of bodies (for the FM history).

        Only needed when the loaded generator's history token carries extra frames
        (legacy symmetric checkpoints). The deployment-design generator consumes no
        end-effector frames, so this is unused there.

        Args:
            body_names: MuJoCo body names, in the FM's frame order.

        Returns:
            Array ``[K, 13]`` per frame ``[pos_w(3), quat_w(4, xyzw), lin_vel_w(3),
            ang_vel_w(3)]`` (link-frame velocity in the world frame).
        """
        out = np.zeros((len(body_names), 13))
        jacp = np.zeros((3, self.mj_model.nv))
        jacr = np.zeros((3, self.mj_model.nv))
        for i, name in enumerate(body_names):
            bid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid == -1:
                raise ValueError(f"Body '{name}' not found in MuJoCo model.")
            out[i, 0:3] = self.mj_data.xpos[bid]
            wxyz = self.mj_data.xquat[bid]
            out[i, 3:7] = [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]  # wxyz -> xyzw
            # Frame-origin twist in world (matches Isaac body_link_vel_w); see
            # get_base_world_velocity for why mj_objectVelocity (CoM) is wrong.
            mujoco.mj_jacBody(self.mj_model, self.mj_data, jacp, jacr, bid)
            out[i, 7:10] = jacp @ self.mj_data.qvel   # linear (world, frame origin)
            out[i, 10:13] = jacr @ self.mj_data.qvel  # angular (world)
        return out

    def compute_height_scan(
        self, base_pos: np.ndarray, base_quat_xyzw: np.ndarray,
        resolution: float = 0.1, size=(1.5, 1.0), offset_fwd: float = 0.75,
        sensor_offset: float = 0.5, ray_start_h: float = 20.0, terrain_group: int = 2,
        offset_lat: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Replicate IsaacLab ``mdp.height_scan`` via a MuJoCo grid raycast.

        Casts a yaw-aligned grid of straight-down rays under the pelvis against the
        terrain (geom group ``terrain_group``) and returns per-ray ground clearance
        matching IsaacLab: ``clearance = base_z - hit_z - sensor_offset`` (verified
        empirically: recorded ``root_z - scan == 0.5`` exactly). Ray order is y-outer,
        x-inner (IsaacLab ``GridPatternCfg`` "xy" ordering); grid is centred
        ``offset_fwd`` m ahead of the pelvis. Defaults match ``G1TeacherSceneCfg``'s
        ``height_scanner`` (0.1 m over 1.5x1.0 -> 16x11 = 176 rays).

        Args:
            base_pos: Pelvis world position ``(3,)``.
            base_quat_xyzw: Pelvis world quaternion ``(x, y, z, w)`` (yaw is used).
            resolution/size/offset_fwd/sensor_offset: match the trained height_scanner.
            ray_start_h: Height above the pelvis to start each (downward) ray.
            terrain_group: MuJoCo geom group of the raycast-able terrain. The robot must
                NOT be in this group, so rays hit the ground and skip the robot's own body.
            offset_lat: Grid centre LEFT of the pelvis (m), i.e. ``offset_pos[1]`` of the
                scanner cfg. Zero for every scanner shipped so far -- taken as an argument
                rather than assumed, because a silently dropped lateral mount would shift
                the whole map sideways and still look like a plausible reading.

        Returns:
            ``(clearance (K,), hit_points_world (K, 3))``; no-hit rays -> ``nan``.
        """
        hx, hy = float(size[0]) / 2.0, float(size[1]) / 2.0
        xs = np.arange(-hx, hx + 1e-9, resolution)   # 16 for size 1.5 @ 0.1
        ys = np.arange(-hy, hy + 1e-9, resolution)   # 11 for size 1.0 @ 0.1
        qx, qy, qz, qw = (float(v) for v in base_quat_xyzw)
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        c, s = np.cos(yaw), np.sin(yaw)
        bx, by, bz = float(base_pos[0]), float(base_pos[1]), float(base_pos[2])
        geomgroup = np.zeros(6, dtype=np.uint8)
        geomgroup[int(terrain_group)] = 1           # hit ONLY the terrain group (skip the robot)
        geomid = np.array([-1], dtype=np.int32)
        vec = np.array([0.0, 0.0, -1.0])            # straight down (IsaacLab ray direction)
        clearance = np.full(xs.size * ys.size, np.nan)
        hits = np.full((xs.size * ys.size, 3), np.nan)
        k = 0
        for gy in ys:                               # y outer
            for gx in xs:                           # x inner (x varies fastest)
                lx, ly = offset_fwd + gx, offset_lat + gy   # mount-offset grid, pelvis frame
                wx = bx + c * lx - s * ly           # yaw-rotate into world
                wy = by + s * lx + c * ly
                pnt = np.array([wx, wy, bz + ray_start_h])
                dist = mujoco.mj_ray(self.mj_model, self.mj_data, pnt, vec, geomgroup, 1, -1, geomid)
                if dist >= 0.0:
                    hit_z = pnt[2] - dist
                    clearance[k] = bz - hit_z - sensor_offset
                    hits[k] = (wx, wy, hit_z)
                k += 1
        return clearance, hits

    def pd_control(self, qpos: np.ndarray, qvel: np.ndarray):
        """Compute the PD control actions to keep the robot moving straight."""
        kp_y = self.gains["kp_y"]
        kd_y = self.gains["kd_y"]
        y_vel = np.sign(self.commanded_vel[0]) * np.clip(-kp_y * qpos[1] + -kd_y * qvel[1], -0.5, 0.5)
        self.commanded_vel[1] = y_vel

        kp_yaw = self.gains["kp_yaw"]
        kd_yaw = self.gains["kd_yaw"]
        siny_cosp = 2 * (qpos[3] * qpos[6] + qpos[4] * qpos[5])
        cosy_cosp = 1 - 2 * (qpos[5] * qpos[5] + qpos[6] * qpos[6])
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        angular_vel = np.sign(self.commanded_vel[0]) * np.clip(-kp_yaw * yaw + -kd_yaw * qvel[5], -0.5, 0.5)
        self.commanded_vel[2] = angular_vel

        # print(f"Commanded velocity: {self.commanded_vel}, y pos: {qpos[1]}, y vel: {qvel[1]}, yaw: {yaw}")

    def apply_policy_gains(self, kp, kd, joint_names) -> int:
        """Overwrite the MJCF position-servo gains with the POLICY's trained kp/kd.

        The policy trains against Isaac Lab actuator gains and exports them in
        ``policy_parameters.yaml``; the MJCF carries its own hand-written numbers, which drift
        from the asset as soon as anyone retunes training (measured on g1_29j: hips 100 vs a
        trained 40.18, shoulders 100 vs 14.25, knees 150 vs 99.10 -- 2-7x too stiff). Applying
        the exported values makes the deployed impedance follow the policy automatically, so
        changing gains in training propagates here with no XML edit.

        For a MuJoCo ``position`` actuator the servo is ``gainprm[0] = kp``,
        ``biasprm[1] = -kp``, ``biasprm[2] = -kv``; all three must move together.

        Args:
            kp: Per-joint stiffness in ``joint_names`` order.
            kd: Per-joint damping in ``joint_names`` order.
            joint_names: Joint order the gains are given in (the policy's Isaac order).

        Returns:
            Number of actuators updated.
        """
        if kp is None or kd is None:
            print("[WARN] policy exposes no kp/kd; keeping the MJCF gains.")
            return 0
        # Remembered so ``reset_robot`` (which rebuilds the model from XML) re-applies them
        # instead of silently reverting to the hand-written values.
        self._policy_gains = (kp, kd, joint_names)
        kp_by = dict(zip(joint_names, np.asarray(kp, dtype=float).ravel()))
        kd_by = dict(zip(joint_names, np.asarray(kd, dtype=float).ravel()))
        n, missing = 0, []
        for a in range(self.mj_model.nu):
            if self.mj_model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
                continue
            jid = int(self.mj_model.actuator_trnid[a, 0])
            jname = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if jname not in kp_by:
                missing.append(jname)
                continue
            self.mj_model.actuator_gainprm[a, 0] = kp_by[jname]
            self.mj_model.actuator_biasprm[a, 1] = -kp_by[jname]
            self.mj_model.actuator_biasprm[a, 2] = -kd_by[jname]
            n += 1
        if missing:
            print(f"[WARN] no policy gains for {len(missing)} actuator(s) {missing[:6]}; "
                  f"those keep their MJCF values.")
        print(f"[INFO] applied policy kp/kd to {n}/{self.mj_model.nu} actuators.")
        return n

    def apply_action(self, action):
        """Apply control action to the robot."""
        self.mj_data.ctrl[: len(action)] = action

    def step(self):
        """Step the robot simulation."""
        mujoco.mj_step(self.mj_model, self.mj_data)

    def failed(self):
        return self.mj_data.qpos[2] < 0.2

    def get_joint_names(self) -> list[str]:
        """Extract joint names from qpos in MuJoCo order.

        Returns:
            List of joint names corresponding to qpos[7:] (excluding floating base)
        """
        joint_names = []
        for i in range(7, self.mj_model.nq):
            joint_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i - 6)
            if joint_name:
                joint_names.append(joint_name)
        return joint_names

    def set_initial_condition(self, policy: "RLPolicy") -> None:
        """Set the robot state from the valid IC in the policy YAML.

        The IC vectors are [base, joints_in_isaac_order]. Joint values are
        reordered to MuJoCo order before writing to qpos/qvel.

        Args:
            policy: The RLPolicy object containing the valid IC data.
        """
        ic_pos = policy.get_valid_ic_pos()
        ic_vel = policy.get_valid_ic_vel()
        if ic_pos is None or ic_vel is None:
            print("[WARNING] No valid IC found in policy params, using XML default.")
            return

        isaac_joint_names = policy.get_joint_names()

        # qpos: [base_pos(3), base_quat(4), joint_pos(N)]
        # Isaac stores quat as (x, y, z, w); MuJoCo expects (w, x, y, z).
        self.mj_data.qpos[:3] = ic_pos[:3]
        self.mj_data.qpos[3] = ic_pos[6]
        self.mj_data.qpos[4:7] = ic_pos[3:6]
        joint_pos_isaac = ic_pos[7:]
        self.mj_data.qpos[7:] = policy.convert_joint_order(
            joint_pos_isaac, isaac_joint_names, self.joint_names
        )

        # qvel: [base_lin_vel(3), base_ang_vel(3), joint_vel(N)]
        self.mj_data.qvel[:6] = ic_vel[:6]
        joint_vel_isaac = ic_vel[6:]
        self.mj_data.qvel[6:] = policy.convert_joint_order(
            joint_vel_isaac, isaac_joint_names, self.joint_names
        )

        # Forward kinematics to update derived quantities
        mujoco.mj_forward(self.mj_model, self.mj_data)
        print("[INFO] Set initial condition from policy YAML.")

    def set_base_and_joints(self, base_pos: np.ndarray, base_quat_wxyz: np.ndarray,
                            joint_pos_mujoco: np.ndarray,
                            base_lin_vel_w: np.ndarray | None = None,
                            base_ang_vel_w: np.ndarray | None = None,
                            joint_vel_mujoco: np.ndarray | None = None) -> None:
        """Set the floating-base pose + joint positions (MuJoCo order), and velocities.

        Any velocity left as ``None`` is zeroed (the standing-spawn behaviour). Both velocity
        arguments are taken in the WORLD frame, matching IsaacLab's
        ``write_root_link_velocity_to_sim``; the angular one is rotated into the body frame on
        the way in, because a MuJoCo free joint stores ``qvel[:3]`` in the world frame but
        ``qvel[3:6]`` in the body frame.

        Args:
            base_pos: Base (pelvis) world position ``[x, y, z]``.
            base_quat_wxyz: Base orientation quaternion in MuJoCo ``(w, x, y, z)`` order.
            joint_pos_mujoco: Joint positions in MuJoCo joint order (``qpos[7:]``).
            base_lin_vel_w: World-frame base linear velocity ``[vx, vy, vz]``, or ``None``.
            base_ang_vel_w: World-frame base angular velocity ``[wx, wy, wz]``, or ``None``.
            joint_vel_mujoco: Joint velocities in MuJoCo joint order (``qvel[6:]``), or ``None``.
        """
        self.mj_data.qpos[:3] = base_pos
        self.mj_data.qpos[3:7] = base_quat_wxyz
        self.mj_data.qpos[7:] = joint_pos_mujoco
        self.mj_data.qvel[:] = 0.0
        if base_lin_vel_w is not None:
            self.mj_data.qvel[:3] = base_lin_vel_w
        if base_ang_vel_w is not None:
            rot = Rotation.from_quat(np.asarray(base_quat_wxyz)[[1, 2, 3, 0]])  # -> xyzw
            self.mj_data.qvel[3:6] = rot.inv().apply(np.asarray(base_ang_vel_w))
        if joint_vel_mujoco is not None:
            self.mj_data.qvel[6:] = joint_vel_mujoco
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def scale_gains(self, kp_mult: float, kv_mult: float, name_filter: list[str] | None = None) -> None:
        """Scale position-actuator kp (and kv damping) gains in place.

        Used for a stiff, well-damped standing hold (fixed-joint PD cannot balance the
        robot at the XML gains); the RL gains set at handoff overwrite these absolutely.

        Args:
            kp_mult: Multiplier for the proportional gain (``gainprm[0]`` and ``-biasprm[1]``).
            kv_mult: Multiplier for the derivative/damping gain (``-biasprm[2]``).
            name_filter: If given, only scale actuators whose name contains one of these
                substrings (e.g. ``["hip", "knee", "ankle"]`` for the legs).
        """
        for a in range(self.mj_model.nu):
            name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            if name_filter is not None and not any(s in name for s in name_filter):
                continue
            self.mj_model.actuator_gainprm[a, 0] *= kp_mult
            self.mj_model.actuator_biasprm[a, 1] *= kp_mult
            self.mj_model.actuator_biasprm[a, 2] *= kv_mult

    def set_pd_gains_from_policy(self, policy) -> None:
        """Set PD gains on MuJoCo actuators from the RLPolicy gains.

        Args:
            policy: The RLPolicy object containing kp and kd gains in Isaac order.

        Raises:
            ValueError: If an actuator for a joint is not found in the MuJoCo model.
        """
        kp_isaac = policy.get_kp()
        kd_isaac = policy.get_kd()
        isaac_joint_names = policy.get_joint_names()

        for i, joint_name in enumerate(isaac_joint_names):
            actuator_name = f"{joint_name}_pos"
            actuator_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )

            if actuator_id == -1:
                raise ValueError(
                    f"Actuator '{actuator_name}' for joint '{joint_name}' not found in MuJoCo model"
                )

            kp = kp_isaac[i]
            kd = kd_isaac[i]

            # For position actuators: force = kp*(ctrl-pos) - kd*vel
            self.mj_model.actuator_gainprm[actuator_id, 0] = kp
            self.mj_model.actuator_biasprm[actuator_id, 1] = -kp
            self.mj_model.actuator_biasprm[actuator_id, 2] = -kd

            print(f"Set gains for '{joint_name}': kp={kp}, kd={kd}")
