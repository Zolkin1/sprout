import os
from collections.abc import Callable

import mujoco
import numpy as np
import pygame
from scipy.spatial.transform import Rotation


class Robot:
    def __init__(self, robot_name: str, scene_name: str, input_function: Callable[[float], np.array] = None, rng=None):
        """Initialize the robot with its model and data."""
        if robot_name != "g1_21j" and robot_name != "g1_21j_M4" and robot_name != "g1_21j_compute" and robot_name != "g1_21j_compute_mlip" and robot_name != "g1_21j_compute_stones":
            raise ValueError("Invalid robot name! Only support g1_21j for now.")

        self.robot_name = robot_name
        self.scene_name = scene_name
        self.mj_model, self.mj_data = self._get_model_data()
        self.commanded_vel = np.zeros(3)  # Store commanded velocity
        self.input_function = input_function
        self.rng = rng

        body_name = "torso_link"
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        self.torso_ipos = self.mj_model.body_ipos[body_id]

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
        relative_path = f"transfer/sim/robots/g1/{file_name}"
        path = os.path.join(os.getcwd(), relative_path)
        print(f"Trying to load the xml at {path}")
        mj_model = mujoco.MjModel.from_xml_path(path)
        mj_data = mujoco.MjData(mj_model)

        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)

        return mj_model, mj_data

    def reset_robot(self):
        """Resets the robot."""
        self.mj_model, self.mj_data = self._get_model_data()

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

    def get_projected_gravity(self, quat):
        """Calculate projected gravity from quaternion."""
        qw, qx, qy, qz = quat
        pg = np.zeros(3)
        pg[0] = 2 * (-qz * qx + qw * qy)
        pg[1] = -2 * (qz * qy + qw * qx)
        pg[2] = 1 - 2 * (qw * qw + qz * qz)
        return pg

    def get_joystick_command(self):
        """Get velocity commands from joystick."""
        des_vel = np.zeros(3)
        if self.joystick is not None:
            for event in pygame.event.get():
                pass
            # Left stick: control vx, vy (2D plane), right stick X-axis: vyaw
            vy = -(self.joystick.get_axis(0))
            vx = -(self.joystick.get_axis(1))
            vyaw = -(self.joystick.get_axis(3)) * 1

            # Clip or zero out small values
            if abs(vx) < 0.1:
                vx = 0
            else:
                vx = np.clip(vx, -0.75, 0.75)
            if abs(vy) < 0.1:
                vy = 0
            else:
                vy = np.clip(vy, -0.0, 0.0)
            if abs(vyaw) < 0.1:
                vyaw = 0
            else:
                vyaw = np.clip(vyaw, -3.14, 3.14)
            des_vel[0] = vx
            des_vel[1] = vy
            des_vel[2] = vyaw
        else:
            des_vel = np.array([0.6, 0.0, 0.0])
        self.commanded_vel = des_vel  # Store the commanded velocity
        print(f"Commanded velocity: {des_vel}")
        return des_vel

    def create_observation(self, policy, height_map=None, sensor_pos=None):
        """Create observation for the policy."""
        qpos = self.mj_data.qpos
        qvel = self.mj_data.qvel
        print(f"base vel: {qvel[0:3]}")
        sim_time = self.mj_data.time
        pg = self.get_projected_gravity(qpos[3:7])
        if self.input_function is None:
            self.commanded_vel = self.get_joystick_command()
        else:
            self.commanded_vel = self.input_function(sim_time)

        # Apply a PD controller on the y axis through the heading
        kp = 0.0
        kd = 0.0
        angular_vel = np.sign(self.commanded_vel[0]) * max(min(-kp * qpos[1] + -kd * qvel[1], 1), -1)
        self.commanded_vel[2] = np.clip(angular_vel, -0.5, 0.5)

        return policy.create_obs(
            qpos[7:],
            qvel[3:6],
            qvel[6:],
            sim_time,
            pg,
            self.commanded_vel,
            height_map=height_map,
            sensor_pos=sensor_pos,
        )

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

    def apply_action(self, action):
        """Apply control action to the robot."""
        self.mj_data.ctrl[: len(action)] = action

    def step(self):
        """Step the robot simulation."""
        mujoco.mj_step(self.mj_model, self.mj_data)

    def failed(self):
        return self.mj_data.qpos[2] < 0.2
