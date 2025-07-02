import torch
from typing import List, Dict, Optional
from isaacsim.core.prims import XFormPrim
from isaacsim.core.simulation_manager import SimulationManager
from pxr import UsdPhysics
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_config.jt_traj import get_euler_from_quat


def find_link_path_in_hierarchy(base_path: str, link_chain: str) -> Optional[str]:
    """
    Traverse through the link hierarchy to find the proper prim path.
    
    Args:
        base_path: Base path to start searching from (e.g., "/World/envs/env_0/Robot")
        link_chain: Link chain specification (e.g., "left_ankle_roll_link/left_foot_middle")
    
    Returns:
        Full prim path if found, None otherwise
    """
    import omni.usd
    
    # Get the USD stage
    stage = omni.usd.get_context().get_stage()
    
    # Split the link chain into individual links
    links = link_chain.split('/')
    
    # Start from the base path
    current_path = base_path
    
    # Traverse through each link in the chain
    for link in links:
        # Try to find the link at the current path
        full_path = f"{current_path}/{link}"
        prim = stage.GetPrimAtPath(full_path)
        
        if prim.IsValid():
            current_path = full_path
        else:
            # If not found, try to find it as a child of the current path
            parent_prim = stage.GetPrimAtPath(current_path)
            if not parent_prim.IsValid():
                print(f"Warning: Parent prim not found at {current_path}")
                return None
            
            # Search through children
            found = False
            for child in parent_prim.GetChildren():
                if child.GetName() == link:
                    current_path = str(child.GetPath())
                    found = True
                    break
            
            if not found:
                # Try to find by partial name match
                for child in parent_prim.GetChildren():
                    if link in child.GetName():
                        current_path = str(child.GetPath())
                        found = True
                        break
                
                if not found:
                    print(f"Warning: Link '{link}' not found under {current_path}")
                    return None
    
    return current_path


def resolve_frame_path(frame_name: str, env_ns: str) -> Optional[str]:
    """
    Resolve a frame name to its full prim path, handling link chains.
    
    Args:
        frame_name: Frame name (can be a single link or a chain like "link1/link2/link3")
        env_ns: Environment namespace (e.g., "/World/envs/env_0/Robot")
    
    Returns:
        Full prim path if found, None otherwise
    """
    # If the frame name contains slashes, it's a link chain
    if '/' in frame_name:
        return find_link_path_in_hierarchy(env_ns, frame_name)
    else:
        # Single link, just append to the environment namespace
        return f"{env_ns}/{frame_name}"


class EndEffectorTracker:
    def __init__(self, constraint_specs: List[Dict], env_ns: str = "/World/envs/env_0/Robot"):
        self.env_ns = env_ns
        self.constraint_specs = constraint_specs
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        self.ee_views = {}  # key: frame name, value: prim view (XFormPrim or physics view)
        self._initialize_views()

    def _initialize_views(self):
        import omni.usd

        # Get the USD stage
        stage = omni.usd.get_context().get_stage()

        # Debug: Traverse all prims in the scene
        print("Available prims in scene:")
        for prim in stage.Traverse():
            print(f"  {prim.GetPath()}")

        for spec in self.constraint_specs:
            if "frame" not in spec:
                continue

            frame_name = spec["frame"]
            
            # Resolve the frame path using the new function
            full_path = resolve_frame_path(frame_name, self.env_ns)
            
            if full_path is None:
                print(f"Warning: Could not resolve frame path for '{frame_name}'")
                continue
            
            print(f"Resolved '{frame_name}' to '{full_path}'")
            
            # Try to find the prim
            prim = stage.GetPrimAtPath(full_path)
            
            if not prim.IsValid():
                print(f"Warning: Prim not found at {full_path}")
                continue

            # Create appropriate view based on prim type
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                view = self._physics_sim_view.create_articulation_view(full_path)
            elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
                view = self._physics_sim_view.create_rigid_body_view(full_path)
            else:
                view = XFormPrim(full_path, reset_xform_properties=False)
            
            self.ee_views[frame_name] = view
            print(f"Successfully created view for '{frame_name}' at '{full_path}'")

    def get_pose(self, frame_name: str):
        """Returns (position, euler_orientation) for a given EE frame."""
        if frame_name not in self.ee_views:
            raise ValueError(f"No view found for frame '{frame_name}'")
        
        view = self.ee_views[frame_name]

        if isinstance(view, XFormPrim):
            pos, quat = view.get_world_pose()
        else:  # physics views
            poses = view.get_transforms()
            pos, quat = poses[0][:3], poses[0][3:]

        pos = torch.tensor(pos)
        euler = get_euler_from_quat(torch.tensor(quat))
        return pos, euler

    def get_relabel_matrix(self, frame_name: str, is_orientation: bool) -> torch.Tensor:
        """Returns the relabel matrix for mirroring."""
        if is_orientation:
            mapping = [-1.0, 1.0, 1.0]  # roll flipped
        else:
            mapping = [1.0, -1.0, 1.0]  # y flipped
        return torch.diag(torch.tensor(mapping, dtype=torch.float32))

    def get_remapped_pose(self, frame_name: str, is_orientation: bool) -> torch.Tensor:
        pos, ori = self.get_pose(frame_name)
        raw = ori if is_orientation else pos
        remap = self.get_relabel_matrix(frame_name, is_orientation)
        return remap @ raw


class EndEffectorTrajectoryConfig:
    """Configuration class for end effector trajectories."""
    
    def __init__(self, yaml_path="source/robot_rl/robot_rl/assets/robots/single_support_config_solution_ee.yaml"):
        self.yaml_path = yaml_path
        self.constraint_specs = []
        self.bezier_coeffs = {}
        self.T = 0.0
        self.load_from_yaml()
    
    def load_from_yaml(self):
        """Load constraint specs and bezier coefficients from YAML file."""
        import yaml
        
        with open(self.yaml_path, 'r') as file:
            data = yaml.safe_load(file)
        
        # Load constraint specifications
        self.constraint_specs = data.get('constraint_specs', [])
        
        # Load bezier coefficients for each constraint
        self.bezier_coeffs = data.get('bezier_coeffs', {})
        
        # Load step period
        self.T = data.get('T', 0.5)
        
        return self
    
    def get_constraint_frames(self) -> List[str]:
        """Extract frame names from constraint specs."""
        frames = []
        for spec in self.constraint_specs:
            if "frame" in spec:
                frames.append(spec["frame"])
        return frames
    
    def get_bezier_coeffs_for_frame(self, frame_name: str, constraint_type: str, device: torch.device = None) -> torch.Tensor:
        """Get bezier coefficients for a specific frame and constraint type."""
        key = f"{frame_name}_{constraint_type}"
        if key in self.bezier_coeffs:
            coeffs = torch.tensor(self.bezier_coeffs[key], dtype=torch.float32)
            if device is not None:
                coeffs = coeffs.to(device)
            return coeffs
        else:
            # Return zero coefficients if not found
            zero_coeffs = torch.zeros(6, dtype=torch.float32)  # Assuming 5th degree bezier + 1
            if device is not None:
                zero_coeffs = zero_coeffs.to(device)
            return zero_coeffs
    
    def evaluate_bezier_trajectory(
        self, frame_name: str, constraint_type: str,
        phase_var: torch.Tensor, T: torch.Tensor,
        bezier_deg: int = 5
    ) -> torch.Tensor:
        """Evaluate bezier trajectory for a specific frame and constraint type."""
        from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_config.jt_traj import bezier_deg as bezier_eval
        
        # Get device from input tensors
        device = phase_var.device
        
        coeffs = self.get_bezier_coeffs_for_frame(frame_name, constraint_type, device)
        if coeffs.numel() == 0:
            return torch.zeros_like(phase_var)
        
        # Reshape coefficients for bezier evaluation
        # Assuming coeffs is [num_axes * (degree + 1)]
        num_axes = coeffs.shape[0] // (bezier_deg + 1)
        coeffs_reshaped = coeffs.view(num_axes, bezier_deg + 1)
        
        # Evaluate bezier curve
        result = bezier_eval(0, phase_var, T, coeffs_reshaped, bezier_deg)
        return result
