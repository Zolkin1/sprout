#!/usr/bin/env python3
"""
Example demonstrating efficient batched evaluation of gait library trajectories.

This example shows how the gait library system now precomputes control points
for all velocities in batched tensors and uses efficient batched Bezier evaluation.
"""

import torch
from pathlib import Path

# Add the source directory to the path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_config.gait_library_traj import (
    GaitLibraryEndEffectorConfig
)


def demonstrate_batched_evaluation():
    """Demonstrate the efficient batched evaluation approach."""
    
    # Configuration
    gait_library_path = "source/robot_rl/robot_rl/assets/robots/gait_library/"
    config_name = "single_support"
    
    # Use discretization mode: (min_vel, max_vel, step) in m/s
    # This will generate gaits at 0.1, 0.2, 0.3, 0.4, 0.5 m/s
    gait_velocity_ranges = (0.1, 0.5, 0.1)
    
    print("=== Gait Library Batched Evaluation Example ===")
    print(f"Gait library path: {gait_library_path}")
    print(f"Config name: {config_name}")
    print(f"Velocity range: {gait_velocity_ranges}")
    print()
    
    # Initialize gait library for end-effector trajectories
    gait_config = GaitLibraryEndEffectorConfig(
        gait_library_path=gait_library_path,
        gait_velocity_ranges=gait_velocity_ranges,
        config_name=config_name
    )
    
    # Show available gaits
    available_gaits = gait_config.get_available_gaits()
    available_speeds = gait_config.get_available_speeds()
    
    print("Available gaits:")
    for gait_name, speed_cms in zip(available_gaits, available_speeds):
        min_vel, max_vel = gait_config.gait_velocity_ranges[gait_name]
        print(f"  {gait_name}: {speed_cms} cm/s (range: {min_vel:.2f}-{max_vel:.2f} m/s)")
    print()
    
    # Show precomputed control point tensors
    print("Precomputed control point tensors:")
    print(f"  left_coeffs_batched:  {gait_config.left_coeffs_batched.shape}")
    print(f"  right_coeffs_batched: {gait_config.right_coeffs_batched.shape}")
    print(f"  Bezier degree: {gait_config.bez_deg}")
    print()
    
    # Simulate different desired velocities for multiple environments
    desired_velocities = torch.tensor([
        [0.05, 0.0],   # Very slow - should use first gait
        [0.15, 0.0],   # Medium slow - should use second gait  
        [0.25, 0.0],   # Medium - should use third gait
        [0.35, 0.0],   # Medium fast - should use fourth gait
        [0.45, 0.0],   # Fast - should use fifth gait
        [0.55, 0.0],   # Very fast - should use last gait (catch-all)
        [0.0, 0.2],    # Lateral movement - should use second gait
        [0.3, 0.3],    # Diagonal movement - should use fourth gait
    ], dtype=torch.float32)
    
    print("Desired velocities for each environment:")
    for i, vel in enumerate(desired_velocities):
        vel_mag = torch.norm(vel).item()
        print(f"  Env {i}: {vel.tolist()} (magnitude: {vel_mag:.2f} m/s)")
    print()
    
    # Select gaits based on velocity
    gait_indices = gait_config.select_gaits_by_velocity(desired_velocities)
    
    print("Gait selection results:")
    for i, (vel, gait_idx) in enumerate(zip(desired_velocities, gait_indices)):
        vel_mag = torch.norm(vel).item()
        selected_gait = available_gaits[gait_idx]
        min_vel, max_vel = gait_config.gait_velocity_ranges[selected_gait]
        print(f"  Env {i}: {vel_mag:.2f} m/s → {selected_gait} (range: {min_vel:.2f}-{max_vel:.2f} m/s)")
    print()
    
    # Demonstrate the batched evaluation process
    print("=== Batched Evaluation Process ===")
    print("1. Control points are precomputed for all velocities:")
    print(f"   - Shape: [num_vel={len(available_gaits)}, jt_dim, degree+1]")
    print("2. All velocities are evaluated simultaneously using batched Bezier:")
    print("   - Reshape to [num_vel*N, jt_dim, degree+1]")
    print("   - Single call to bezier_deg_batched() for all environments")
    print("3. Results are reshaped back to [num_vel, N, jt_dim]")
    print("4. Appropriate velocity is selected for each environment")
    print()
    
    # Show the velocity ranges for each gait
    print("Velocity ranges for gait selection:")
    for i, gait_name in enumerate(available_gaits):
        min_vel, max_vel = gait_config.gait_velocity_ranges[gait_name]
        if i == 0:
            print(f"  {gait_name}: 0.00 - {max_vel:.2f} m/s")
        elif i == len(available_gaits) - 1:
            print(f"  {gait_name}: {min_vel:.2f} - ∞ m/s (catch-all)")
        else:
            print(f"  {gait_name}: {min_vel:.2f} - {max_vel:.2f} m/s")
    print()
    
    print("=== Benefits of Batched Evaluation ===")
    print("✓ Ultra-efficient: Single batched evaluation for all velocities")
    print("✓ Memory efficient: Precomputed tensors, no runtime allocation")
    print("✓ GPU optimized: All operations are tensor-based")
    print("✓ Simple selection: Just index into precomputed results")
    print("✓ Scalable: Performance doesn't degrade with more velocities")


if __name__ == "__main__":
    demonstrate_batched_evaluation() 