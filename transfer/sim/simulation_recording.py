import csv
import math
import os
from collections.abc import Callable
from datetime import datetime

import mujoco
import mujoco.viewer
import numpy as np
import yaml
import mediapy as media  # pip install mediapy
from .robot import Robot


def log_row_to_csv(filename, data):
    """
    Appends a single row of data to an existing CSV file.

    Args:
      filename (str): The path to the CSV file.
      data_row (list): A list of data points for the row.
    """
    try:
        # Create the file if it doesn't exist
        if not os.path.exists(filename):
            print(f"Creating new log file: {filename}")
            with open(filename, "w", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(data)
        else:
            # Append to existing file
            with open(filename, "a", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(data)
                # Force write to disk
                csvfile.flush()
                os.fsync(csvfile.fileno())
    except Exception as e:
        print(f"Error appending row to {filename}: {e}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"File exists: {os.path.exists(filename)}")
        print(f"File permissions: {oct(os.stat(filename).st_mode)[-3:] if os.path.exists(filename) else 'N/A'}")


class Simulation:
    def __init__(
        self,
        policy,
        robot: Robot,
        log: bool = False,
        log_dir: str = None,
        use_height_sensor: bool = True,
        tracking_body_name: str = "",
        record_video: bool = False,
        video_width: int = 5120,
        video_height: int = 1440,
        video_fps: int = 60,
        camera_config: dict = None,
    ):
        """Initialize the simulation.

        Args:
            policy: The policy to use for control
            robot: Robot instance
            log: Whether to log data
            log_dir: Directory to save logs
            use_height_sensor: Whether to use height sensor (default: False)
            tracking_body_name: Name of body to track with camera
            record_video: Whether to record video (default: False)
            video_width: Video width in pixels (default: 1280)
            video_height: Video height in pixels (default: 720)
            video_fps: Video frames per second (default: 30)
            camera_config: Dictionary with camera settings (default: None)
                Options:
                - 'lookat': [x, y, z] - point camera looks at
                - 'distance': float - distance from lookat point
                - 'azimuth': float - rotation around z-axis (degrees)
                - 'elevation': float - rotation above xy-plane (degrees)
                - 'type': 'fixed' or 'tracking' - camera type
                - 'body_name': str - body to track (for tracking cameras)
        """
        self.policy = policy
        self.robot = robot
        self.log = log
        self.log_dir = log_dir
        self.log_file = None
        self.new_log_folder = ""
        self.use_height_sensor = use_height_sensor

        # Video recording parameters
        self.record_video = record_video
        self.video_width = video_width
        self.video_height = video_height
        self.video_fps = video_fps
        self.video_frames = []
        self.video_path = None
        self.camera_config = camera_config or {}
        
        # Check and adjust framebuffer size if recording video
        if self.record_video:
            self._check_framebuffer_size()

        # Setup simulation parameters
        self.sim_steps_per_policy_update = int(policy.dt / robot.mj_model.opt.timestep)
        self.sim_loop_rate = self.sim_steps_per_policy_update * robot.mj_model.opt.timestep
        self.viewer_rate = math.ceil((1 / 50) / robot.mj_model.opt.timestep)

        # Tracking body
        self.tracking_body_name = tracking_body_name

        # Setup logging if enabled
        if self.log or self.record_video:
            if self.log_dir is None:
                self.log_dir = "./logs"  # Default log directory
                print(f"No log_dir specified, using default: {self.log_dir}")
            self._setup_logging()

    def _get_arrow_rotation_matrix(self, direction: np.ndarray) -> np.ndarray:
        """Create rotation matrix to orient arrow along direction vector."""
        direction = direction / (np.linalg.norm(direction) + 1e-8)

        # Default arrow points along z-axis, rotate to point along direction
        z_axis = np.array([0, 0, 1])

        if np.allclose(direction, z_axis):
            return np.eye(3)
        elif np.allclose(direction, -z_axis):
            return np.diag([1, -1, -1])

        # Rodrigues' rotation formula
        v = np.cross(z_axis, direction)
        s = np.linalg.norm(v)
        c = np.dot(z_axis, direction)

        vx = np.array([[0, -v[2], v[1]], 
                       [v[2], 0, -v[0]], 
                       [-v[1], v[0], 0]])

        rotation_matrix = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
        return rotation_matrix
    def _setup_logging(self):
        """Setup logging directory and files."""
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d-%H-%M-%S")
        self.new_log_folder = os.path.join(self.log_dir, timestamp_str)
        try:
            os.makedirs(self.new_log_folder, exist_ok=True)
            print(f"Successfully created folder: {self.new_log_folder}")
        except OSError as e:
            print(f"Error creating folder {self.new_log_folder}: {e}")

        print(f"Saving logs to {self.new_log_folder}.")
        
        if self.log:
            self.log_file = os.path.join(self.new_log_folder, "sim_log.csv")

        if self.record_video:
            self.video_path = os.path.join(self.new_log_folder, "simulation.mp4")

        # Save simulation configuration
        data_structure = [
            {"name": "time", "length": 1},
            {"name": "qpos", "length": self.robot.mj_data.qpos.shape[0]},
            {"name": "qvel", "length": self.robot.mj_data.qvel.shape[0]},
            {"name": "obs", "length": self.policy.get_num_obs()},
            {"name": "action", "length": self.policy.get_num_actions()},
            {"name": "torque", "length": self.robot.mj_model.nu},
            {"name": "left_ankle_pos", "length": 3},
            {"name": "right_ankle_pos", "length": 3},
            {"name": "commanded_vel", "length": 3},
        ]

        sim_config = {
            "simulator": "mujoco",
            "robot": self.robot.robot_name,
            "policy": self.policy.get_chkpt_path(),
            "policy_dt": self.policy.dt,
            "use_height_sensor": self.use_height_sensor,
            "record_video": self.record_video,
            "video_fps": self.video_fps if self.record_video else None,
            "data_structure": data_structure,
        }

        config_path = os.path.join(self.new_log_folder, "sim_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(sim_config, f)

    def _check_framebuffer_size(self):
        """Check and adjust framebuffer size for video recording."""
        buffer_width = self.robot.mj_model.vis.global_.offwidth
        buffer_height = self.robot.mj_model.vis.global_.offheight
        
        if self.video_width > buffer_width or self.video_height > buffer_height:
            print(f"Warning: Requested video resolution ({self.video_width}x{self.video_height}) "
                  f"exceeds framebuffer size ({buffer_width}x{buffer_height})")
            print(f"Adjusting framebuffer size to {self.video_width}x{self.video_height}")
            
            # Update the model's framebuffer size
            self.robot.mj_model.vis.global_.offwidth = max(self.video_width, buffer_width)
            self.robot.mj_model.vis.global_.offheight = max(self.video_height, buffer_height)

    def _save_video(self):
        """Save recorded video frames to file."""
        if self.video_frames and self.video_path:
            print(f"Saving video with {len(self.video_frames)} frames to {self.video_path}")
            media.write_video(self.video_path, self.video_frames, fps=self.video_fps)
            print(f"Video saved successfully to {self.video_path}")
            # Clear frames to free memory
            self.video_frames = []

    def _setup_camera(self, camera=None):
        """Setup camera with custom configuration.
        
        Args:
            camera: MjvCamera object to configure (creates new one if None)
        
        Returns:
            Configured MjvCamera object
        """
        if camera is None:
            camera = mujoco.MjvCamera()
        
        # Set camera type
        cam_type = self.camera_config.get('type', 'fixed')
        body_name = self.camera_config.get('body_name', self.tracking_body_name)
        
        if cam_type == 'tracking' and body_name:
            camera.trackbodyid = mujoco.mj_name2id(
                self.robot.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        else:
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        
        # Set lookat point
        if 'lookat' in self.camera_config:
            camera.lookat[:] = self.camera_config['lookat']
        
        # Set distance
        if 'distance' in self.camera_config:
            camera.distance = self.camera_config['distance']
        
        # Set azimuth (rotation around z-axis)
        if 'azimuth' in self.camera_config:
            camera.azimuth = self.camera_config['azimuth']
        
        # Set elevation (angle above xy-plane)
        if 'elevation' in self.camera_config:
            camera.elevation = self.camera_config['elevation']
        
        return camera

    def get_logging_folder(self):
        return self.new_log_folder

    def run_headless(
        self,
        total_time: float,
        force_disturbance: Callable[[float], np.array] = None,
    ):
        """Run the simulation without a viewer."""
        print(
            f"Starting mujoco simulation with robot {self.robot.robot_name}.\n"
            f"Policy dt set to {self.policy.dt} s ({self.sim_steps_per_policy_update} steps per policy update.)\n"
            f"Simulation dt set to {self.robot.mj_model.opt.timestep} s. Sim loop rate set to {self.sim_loop_rate} s.\n"
            f"Height sensor enabled: {self.use_height_sensor}\n"
            f"Video recording enabled: {self.record_video}\n"
        )

        # Setup offscreen renderer for video
        if self.record_video:
            renderer = mujoco.Renderer(self.robot.mj_model, self.video_height, self.video_width)
            frame_interval = int(1.0 / (self.video_fps * self.robot.mj_model.opt.timestep))
            
            # Setup camera with custom configuration
            camera = self._setup_camera()
            print("Note: Height map visualization (red dots) will not appear in recorded video.")

        # Setup height sensor if enabled
        if self.use_height_sensor:
            grid_size = (1.5, 1.5)
            x_y_num_rays = (25, 25)
            print(f"Height sensor configured: grid_size={grid_size}, rays={x_y_num_rays}")

        if total_time < 0:
            raise ValueError("Headless simulation must have a positive total time specified!")

        success = True
        step_count = 0

        try:
            while self.robot.mj_data.time < total_time:
                # Get observation with height map data (if enabled) and compute action
                if self.use_height_sensor:
                    # Ray cast to get height map
                    height_map = self._ray_cast_sensor(
                        self.robot.mj_model, self.robot.mj_data, "height_sensor_site", grid_size, x_y_num_rays
                    )
                    site_id = mujoco.mj_name2id(self.robot.mj_model, mujoco.mjtObj.mjOBJ_SITE, "height_sensor_site")
                    sensor_pos = self.robot.mj_data.site_xpos[site_id]

                    # Create observation WITH height map data for policy
                    obs = self.robot.create_observation(self.policy, height_map=height_map, sensor_pos=sensor_pos)
                else:
                    # Create observation WITHOUT height map data
                    obs = self.robot.create_observation(self.policy)
                
                # Policy uses the observation (including height map if enabled)
                action = self.policy.get_action(obs)
                self.robot.apply_action(action)

                if self.robot.failed():
                    success = False
                    break

                # Step the simulator
                for i in range(self.sim_steps_per_policy_update):
                    if force_disturbance is not None:
                        self.robot.apply_force_disturbance(force_disturbance(self.robot.mj_data.time))

                    # Step the sim
                    self.robot.step()
                    step_count += 1

                    # Record video frame at specified fps
                    if self.record_video and step_count % frame_interval == 0:
                        renderer.update_scene(self.robot.mj_data, camera=camera)
                        frame = renderer.render()
                        self.video_frames.append(frame)

                    # Only log at viewer_rate intervals
                    if i % self.viewer_rate == 0:
                        if self.log:
                            log_data = self.robot.get_log_data(self.policy, obs, action)
                            log_row_to_csv(self.log_file, log_data)

        finally:
            # Save video when simulation ends
            if self.record_video:
                self._save_video()
                renderer.close()

        return success

    def run(
        self,
        total_time: float,
        force_disturbance: Callable[[float], np.array] = None,
    ):
        """Run the simulation."""
        print(
            f"Starting mujoco simulation with robot {self.robot.robot_name}.\n"
            f"Policy dt set to {self.policy.dt} s ({self.sim_steps_per_policy_update} steps per policy update.)\n"
            f"Simulation dt set to {self.robot.mj_model.opt.timestep} s. Sim loop rate set to {self.sim_loop_rate} s.\n"
            f"Height sensor enabled: {self.use_height_sensor}\n"
            f"Video recording enabled: {self.record_video}\n"
        )
        
        if self.record_video and self.use_height_sensor:
            print("Note: Height map visualization (red dots) will appear in viewer but not in recorded video.")
            print("      Height map DATA is still passed to the policy for control.")

        success = True

        # Setup offscreen renderer for video if needed
        if self.record_video:
            renderer = mujoco.Renderer(self.robot.mj_model, self.video_height, self.video_width)
            frame_interval = int(1.0 / (self.video_fps * self.robot.mj_model.opt.timestep))
            step_count = 0

        try:
            with mujoco.viewer.launch_passive(self.robot.mj_model, self.robot.mj_data) as viewer:
                # Setup camera with custom configuration
                if self.tracking_body_name != "" or self.camera_config:
                    self._setup_camera(viewer.cam)
                    if self.tracking_body_name:
                        print(f"Camera tracking body: {self.tracking_body_name}")
                    if self.camera_config:
                        print(f"Camera config applied: {self.camera_config}")

                # Setup height sensor visualization if enabled
                if self.use_height_sensor:
                    grid_size = (1.0, 1.0)
                    x_y_num_rays = (11, 11)
                    height_map = self._ray_cast_sensor(
                        self.robot.mj_model, self.robot.mj_data, "height_sensor_site", grid_size, x_y_num_rays
                    )
                    # Add custom debug spheres
                    for ii, pos in enumerate(height_map.reshape(-1, 3)):
                        mujoco.mjv_initGeom(
                            viewer.user_scn.geoms[ii],
                            type=mujoco.mjtGeom.mjGEOM_SPHERE,
                            size=np.array([0.02, 0, 0]),
                            pos=pos,
                            mat=np.eye(3).flatten(),
                            rgba=np.array([1, 0, 0, 1]),
                        )
                        viewer.user_scn.ngeom += 1

                while viewer.is_running():
                    if total_time > 0 and self.robot.mj_data.time > total_time:
                        break
                    
                    # Get observation with height map data (if enabled) and compute action
                    if self.use_height_sensor:
                        # Ray cast to get height map
                        height_map = self._ray_cast_sensor(
                            self.robot.mj_model, self.robot.mj_data, "height_sensor_site", grid_size, x_y_num_rays
                        )
                        site_id = mujoco.mj_name2id(self.robot.mj_model, mujoco.mjtObj.mjOBJ_SITE, "height_sensor_site")
                        sensor_pos = self.robot.mj_data.site_xpos[site_id]

                        # Create observation WITH height map data for policy
                        obs = self.robot.create_observation(self.policy, height_map=height_map, sensor_pos=sensor_pos)
                    else:
                        # Create observation WITHOUT height map data
                        obs = self.robot.create_observation(self.policy)
                    
                    # Policy uses the observation (including height map if enabled)
                    action = self.policy.get_action(obs)
                    self.robot.apply_action(action)

                    if self.robot.failed():
                        success = False
                        break

                    # Step the simulator
                    for i in range(self.sim_steps_per_policy_update):
                        # Update scene
                        scene = mujoco.MjvScene(self.robot.mj_model, maxgeom=1000)
                        opt = mujoco.MjvOption()
                        mujoco.mjv_updateScene(
                            self.robot.mj_model,
                            self.robot.mj_data,
                            opt,
                            None,
                            viewer.cam,
                            mujoco.mjtCatBit.mjCAT_ALL,
                            scene,
                        )

                        # Update height sensor visualization if enabled
                        if self.use_height_sensor:
                            height_map = self._ray_cast_sensor(
                                self.robot.mj_model, self.robot.mj_data, "height_sensor_site", grid_size, x_y_num_rays
                            )
                            for ii, pos in enumerate(height_map.reshape(-1, 3)):
                                viewer.user_scn.geoms[ii].pos = pos

                        if force_disturbance is not None:
                            force_vector = force_disturbance(self.robot.mj_data.time)
                            self.robot.apply_force_disturbance(force_vector)

                            # Visualize the force as an arrow
                            if np.linalg.norm(force_vector[:3]) > 0.01:  # Only show if force is non-zero
                                # Get the body position where force is applied
                                body_id = mujoco.mj_name2id(
                                    self.robot.mj_model, 
                                    mujoco.mjtObj.mjOBJ_BODY, 
                                    "torso_link"  # or whatever body you're applying force to
                                )
                                body_pos = self.robot.mj_data.xpos[body_id]

                                # Scale force for visualization (adjust scale_factor as needed)
                                scale_factor = 0.01  # Adjust this to make arrow visible
                                force_direction = force_vector[:3] * scale_factor
                                arrow_end = body_pos + force_direction

                                # Add arrow to viewer (reuse existing geom or add new one)
                                geom_idx = viewer.user_scn.ngeom
                                if geom_idx < viewer.user_scn.maxgeom:
                                    mujoco.mjv_initGeom(
                                        viewer.user_scn.geoms[geom_idx],
                                        type=mujoco.mjtGeom.mjGEOM_ARROW,
                                        size=np.array([0.02, 0.02, np.linalg.norm(force_direction)]),
                                        pos=body_pos,
                                        mat=self._get_arrow_rotation_matrix(force_direction).flatten(),
                                        rgba=np.array([1.0, 1.0, 0.0, 1.0])  # Yellow arrow
                                    )
                                    viewer.user_scn.ngeom += 1

                        # Step the sim
                        self.robot.step()

                        # Record video frame
                        if self.record_video:
                            step_count += 1
                            if step_count % frame_interval == 0:
                                # Render directly from the renderer with the current camera
                                renderer.update_scene(self.robot.mj_data, camera=viewer.cam)
                                frame = renderer.render()
                                self.video_frames.append(frame)

                        # Only log and sync viewer at viewer_rate intervals
                        if i % self.viewer_rate == 0:
                            if self.log:
                                log_data = self.robot.get_log_data(self.policy, obs, action)
                                log_row_to_csv(self.log_file, log_data)
                            viewer.sync()

        finally:
            # Save video when simulation ends
            if self.record_video:
                self._save_video()
                renderer.close()

        return success

    def _ray_cast_sensor(self, model, data, site_name, size, x_y_num_rays, sen_offset=0):
        """Using a grid pattern, create a height map using ray casting."""
        ray_pos_shape = x_y_num_rays
        ray_pos_shape = ray_pos_shape + (3,)
        ray_pos = np.zeros(ray_pos_shape)

        # Get the site location
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        site_pos = data.site_xpos[site_id].copy()

        # Add to the global z
        site_pos[2] = site_pos[2] + 10

        site_pos[0] = site_pos[0] - size[0] / 2.0
        site_pos[1] = site_pos[1] - size[1] / 2.0

        # Ray information
        direction = np.zeros(3)
        direction[2] = -1
        geom_group = np.zeros(6, dtype=np.int32)
        geom_group[2] = 1  # Only include group 2

        # Ray spacing
        spacing = np.zeros(3)
        spacing[0] = size[0] / (x_y_num_rays[0] - 1)
        spacing[1] = size[1] / (x_y_num_rays[1] - 1)

        # Loop through the rays
        for xray in range(x_y_num_rays[0]):
            for yray in range(x_y_num_rays[1]):
                geom_id = np.zeros(1, dtype=np.int32)
                offset = spacing.copy()
                offset[0] = spacing[0] * xray
                offset[1] = spacing[1] * yray

                ray_origin = offset + site_pos
                ray_pos[xray, yray, 2] = -mujoco.mj_ray(
                    model, data, ray_origin.astype(np.float64), direction.astype(np.float64), geom_group, 1, -1, geom_id
                )

                ray_pos[xray, yray, :] = ray_origin + ray_pos[xray, yray, :]

        return ray_pos