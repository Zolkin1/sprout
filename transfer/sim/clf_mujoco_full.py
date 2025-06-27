"""
Full CLF MuJoCo Simulation with G1 Solution Trajectories

This simulation uses the full CLF controller from transfer/clf.py with Pinocchio
integration and reference trajectories loaded from g1_solution.yaml.
"""

import os
import time
import math
import yaml
import numpy as np
import mujoco
import mujoco.viewer
from datetime import datetime
import csv

try:
    import pinocchio as pin
    from transfer.clf import CLFQPController, CLFQPParams
    PINOCCHIO_AVAILABLE = True
except ImportError:
    print("Warning: Pinocchio not available. Using simplified PD controller.")
    PINOCCHIO_AVAILABLE = False

from transfer.sim.robot import Robot


class BezierTrajectoryGenerator:
    """Handles Bezier curve evaluation for reference trajectories."""
    
    def __init__(self, yaml_path: str):
        """Load trajectory data from YAML file."""
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
        
        # Extract data
        self.bezier_coeffs = np.array(data['bezier_coeffs'])
        self.joint_order = data['joint_order']
        self.spline_order = data['spline_order']
        self.T = data['T'][0] if isinstance(data['T'], list) else data['T']
        
        # Reshape coefficients: [num_joints, num_control_points]
        num_control_points = self.spline_order + 1
        num_joints = len(self.joint_order)
        self.bezier_coeffs_reshaped = self.bezier_coeffs.reshape(num_joints, num_control_points)
        
        print(f"Loaded trajectory with {num_joints} joints, {num_control_points} control points")
        print(f"Step period: {self.T} seconds")
    
    def evaluate_bezier(self, tau: float, order: int = 0) -> np.ndarray:
        """
        Evaluate Bezier curve at phase variable tau.
        
        Args:
            tau: Phase variable in [0, 1]
            order: 0 for position, 1 for velocity
            
        Returns:
            Joint positions/velocities as numpy array
        """
        tau = np.clip(tau, 0.0, 1.0)
        n_joints, n_control_points = self.bezier_coeffs_reshaped.shape
        degree = n_control_points - 1
        
        if order == 0:
            # Position evaluation
            result = np.zeros(n_joints)
            for i in range(n_control_points):
                binomial_coeff = math.comb(degree, i)
                bernstein = binomial_coeff * (tau ** i) * ((1 - tau) ** (degree - i))
                result += bernstein * self.bezier_coeffs_reshaped[:, i]
            return result
        
        elif order == 1:
            # Velocity evaluation
            result = np.zeros(n_joints)
            for i in range(degree):
                binomial_coeff = math.comb(degree - 1, i)
                bernstein = binomial_coeff * (tau ** i) * ((1 - tau) ** (degree - 1 - i))
                diff = self.bezier_coeffs_reshaped[:, i + 1] - self.bezier_coeffs_reshaped[:, i]
                result += degree * bernstein * diff
            return result / self.T
        
        else:
            raise ValueError("order must be 0 or 1")


class CLFVirtualConstraint:
    """Virtual constraint function for CLF controller."""
    
    def __init__(self, trajectory_generator: BezierTrajectoryGenerator, joint_indices: list):
        self.traj_gen = trajectory_generator
        self.joint_indices = joint_indices
    
    def __call__(self, q, tau):
        """Compute virtual constraint output y = q - q_desired(tau)."""
        q_desired = self.traj_gen.evaluate_bezier(tau, order=0)
        return q[self.joint_indices] - q_desired[self.joint_indices]


class CLFPhaseVariable:
    """Phase variable function for CLF controller."""
    
    def __init__(self, step_period: float):
        self.step_period = step_period
    
    def __call__(self, q, t=None):
        """Compute phase variable tau from time."""
        if t is None:
            # Use current simulation time
            return 0.0  # Placeholder - will be set in simulation loop
        return (t % self.step_period) / self.step_period


class FullCLFMujocoSimulation:
    """Full MuJoCo simulation using CLF controller with G1 trajectories."""
    
    def __init__(self, robot_name: str = "g1_21j", scene_name: str = "basic_scene", 
                 yaml_path: str = "source/robot_rl/robot_rl/assets/robots/g1_solution.yaml",
                 urdf_path: str = "transfer/sim/robots/g1/g1_21j.urdf",
                 log: bool = True, log_dir: str = "logs"):
        """Initialize the full CLF MuJoCo simulation."""
        self.robot_name = robot_name
        self.scene_name = scene_name
        self.log = log
        self.log_dir = log_dir
        self.log_file = None
        
        # Load robot
        self.robot = Robot(robot_name, scene_name)
        self.mj_model = self.robot.mj_model
        self.mj_data = self.robot.mj_data
        
        # Load trajectory data
        self.traj_gen = BezierTrajectoryGenerator(yaml_path)
        
        # Setup joint mapping (assuming joint order matches)
        self.joint_indices = list(range(self.mj_model.nu))  # All actuated joints
        
        # Create virtual constraint and phase variable functions
        self.virtual_constraint = CLFVirtualConstraint(self.traj_gen, self.joint_indices)
        self.phase_variable = CLFPhaseVariable(self.traj_gen.T)
        
        # Create CLF controller
        self._create_clf_controller(urdf_path)
        
        # Simulation parameters
        self.dt = self.mj_model.opt.timestep
        self.sim_steps_per_control = 1  # Control at simulation rate
        
        # Setup logging
        if self.log:
            self._setup_logging()
        
        # Initialize state
        self.current_time = 0.0
        self.step_count = 0
    
    def _create_clf_controller(self, urdf_path: str):
        """Create the CLF controller with Pinocchio integration."""
        if not PINOCCHIO_AVAILABLE:
            print("Using simplified PD controller (Pinocchio not available)")
            self.kp = 100.0 * np.ones(self.mj_model.nu)
            self.kd = 20.0 * np.ones(self.mj_model.nu)
            self.use_full_clf = False
            return
        
        try:
            # Load URDF into Pinocchio
            if not os.path.exists(urdf_path):
                print(f"URDF file not found: {urdf_path}")
                print("Using simplified PD controller")
                self.kp = 100.0 * np.ones(self.mj_model.nu)
                self.kd = 20.0 * np.ones(self.mj_model.nu)
                self.use_full_clf = False
                return
            
            print(f"Loading Pinocchio model from: {urdf_path}")
            self.pin_model = pin.buildModelFromUrdf(urdf_path)
            self.pin_data = pin.Data(self.pin_model)
            
            # Create CLF parameters
            num_outputs = len(self.joint_indices)
            clf_params = CLFQPParams(
                kp=np.array([100.0] * num_outputs),
                kd=np.array([20.0] * num_outputs),
                alpha=10.0,
                u_min=[-100.0] * self.mj_model.nu,
                u_max=[100.0] * self.mj_model.nu
            )
            
            # Create CLF controller
            self.clf_controller = CLFQPController(
                self.pin_model,
                virtual_constraint_fun=self.virtual_constraint,
                phase_variable_fun=self.phase_variable,
                params=clf_params
            )
            
            self.use_full_clf = True
            print("Full CLF controller created successfully")
            
        except Exception as e:
            print(f"Error creating CLF controller: {e}")
            print("Falling back to simplified PD controller")
            self.kp = 100.0 * np.ones(self.mj_model.nu)
            self.kd = 20.0 * np.ones(self.mj_model.nu)
            self.use_full_clf = False
    
    def _setup_logging(self):
        """Setup logging directory and files."""
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d-%H-%M-%S")
        new_folder_path = os.path.join(self.log_dir, f"full_clf_sim_{timestamp_str}")
        try:
            os.makedirs(new_folder_path, exist_ok=True)
            print(f"Created log directory: {new_folder_path}")
        except OSError as e:
            print(f"Error creating log directory {new_folder_path}: {e}")
        
        self.log_file = os.path.join(new_folder_path, "full_clf_sim_log.csv")
        
        # Write header
        header = ["time", "step_count", "phase_var", "controller_type"]
        header.extend([f"q_{i}" for i in range(self.mj_model.nq)])
        header.extend([f"dq_{i}" for i in range(self.mj_model.nv)])
        header.extend([f"q_des_{i}" for i in range(self.mj_model.nu)])
        header.extend([f"dq_des_{i}" for i in range(self.mj_model.nu)])
        header.extend([f"torque_{i}" for i in range(self.mj_model.nu)])
        
        with open(self.log_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(header)
    
    def compute_control(self, q, dq, tau):
        """Compute control torques using CLF controller."""
        # Get desired positions and velocities
        q_des = self.traj_gen.evaluate_bezier(tau, order=0)
        dq_des = self.traj_gen.evaluate_bezier(tau, order=1)
        
        if self.use_full_clf:
            try:
                # Use full CLF controller
                # Extract joint positions and velocities (skip base)
                q_joints = q[7:]  # Skip base position/orientation
                dq_joints = dq[6:]  # Skip base velocity
                
                # Compute CLF control
                torque = self.clf_controller.compute_control(q_joints, dq_joints)
                
                # Ensure torque is the right shape
                if len(torque) != self.mj_model.nu:
                    print(f"Warning: Torque shape mismatch. Expected {self.mj_model.nu}, got {len(torque)}")
                    torque = np.zeros(self.mj_model.nu)
                
                controller_type = "CLF"
                
            except Exception as e:
                print(f"CLF controller failed: {e}")
                print("Falling back to PD controller")
                # Fall back to PD controller
                q_error = q[7:] - q_des
                dq_error = dq[6:] - dq_des
                torque = -self.kp * q_error - self.kd * dq_error
                controller_type = "PD_FALLBACK"
        else:
            # Use simple PD controller
            q_error = q[7:] - q_des  # Skip base position/orientation
            dq_error = dq[6:] - dq_des  # Skip base velocity
            
            # Apply gains
            torque = -self.kp * q_error - self.kd * dq_error
            controller_type = "PD"
        
        # Apply torque limits
        torque = np.clip(torque, -100.0, 100.0)
        
        return torque, q_des, dq_des, controller_type
    
    def log_data(self, tau, q_des, dq_des, torque, controller_type):
        """Log simulation data."""
        if not self.log:
            return
        
        log_data = [
            self.current_time,
            self.step_count,
            tau,
            controller_type
        ]
        log_data.extend(self.mj_data.qpos.tolist())
        log_data.extend(self.mj_data.qvel.tolist())
        log_data.extend(q_des.tolist())
        log_data.extend(dq_des.tolist())
        log_data.extend(torque.tolist())
        
        with open(self.log_file, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(log_data)
    
    def run(self, max_steps: int = 10000, real_time: bool = True):
        """Run the full CLF MuJoCo simulation."""
        print("Starting Full CLF MuJoCo simulation")
        print(f"Robot: {self.robot_name}, Scene: {self.scene_name}")
        print(f"Step period: {self.traj_gen.T} seconds")
        print(f"Simulation dt: {self.dt} seconds")
        print(f"Max steps: {max_steps}")
        print(f"Using full CLF: {self.use_full_clf}")
        
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            start_time = time.time()
            
            while viewer.is_running() and self.step_count < max_steps:
                # Update phase variable
                tau = (self.current_time % self.traj_gen.T) / self.traj_gen.T
                
                # Get current state
                q = self.mj_data.qpos
                dq = self.mj_data.qvel
                
                # Compute control
                torque, q_des, dq_des, controller_type = self.compute_control(q, dq, tau)
                
                # Apply control
                self.mj_data.ctrl[:] = torque
                
                # Step simulation
                mujoco.mj_step(self.mj_model, self.mj_data)
                
                # Update time and counters
                self.current_time += self.dt
                self.step_count += 1
                
                # Log data
                self.log_data(tau, q_des, dq_des, torque, controller_type)
                
                # Sync viewer
                viewer.sync()
                
                # Real-time pacing
                if real_time:
                    elapsed = time.time() - start_time
                    expected = self.current_time
                    if elapsed < expected:
                        time.sleep(expected - elapsed)
                
                # Print progress
                if self.step_count % 1000 == 0:
                    print(f"Step {self.step_count}, Time: {self.current_time:.2f}s, Phase: {tau:.3f}, Controller: {controller_type}")
        
        print(f"Simulation completed. Total steps: {self.step_count}")
        if self.log:
            print(f"Log saved to: {self.log_file}")


def main():
    """Main function to run the full CLF simulation."""
    # Create and run simulation
    sim = FullCLFMujocoSimulation(
        robot_name="g1_21j",
        scene_name="basic_scene",
        yaml_path="source/robot_rl/robot_rl/assets/robots/g1_solution.yaml",
        urdf_path="transfer/sim/robots/g1/g1_21j.urdf",
        log=True,
        log_dir="logs"
    )
    
    # Run for 10 seconds
    max_steps = int(10.0 / sim.dt)
    sim.run(max_steps=max_steps, real_time=True)


if __name__ == "__main__":
    main() 