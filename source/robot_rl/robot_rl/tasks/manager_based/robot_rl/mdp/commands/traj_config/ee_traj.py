import torch
from typing import List, Dict
from isaacsim.core.prims import XFormPrim
from isaacsim.core.simulation_manager import SimulationManager
from pxr import UsdPhysics
import isaaclab.sim as sim_utils
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_config.jt_traj import get_euler_from_quat



        constraint_specs = [
            {"type": "ee_ori", "frame": "pelvis_link", "axes": [0, 1, 2]},
            {"type": "com_pos", "axes": [0, 1, 2]},  # Center of mass (optional, placeholder)

            {"type": "ee_pos", "frame": "left_ankle_roll_link/left_foot_middle", "axes": [0, 1, 2]},
            {"type": "ee_ori", "frame": "left_ankle_roll_link/left_foot_middle", "axes": [0, 1, 2]},

            {"type": "ee_pos", "frame": "left_elbow_link/left_wrist_roll_link/left_wrist_pitch_link/left_wrist_yaw_link/left_hand_palm_link", "axes": [0, 1, 2]},
            {"type": "ee_ori", "frame": "left_elbow_link/left_wrist_roll_link/left_wrist_pitch_link/left_wrist_yaw_link/left_hand_palm_link", "axes": [2]},

            
            {"type": "joint", "indices": [12]},  # This one is joint-specific; EE tracker won't handle it
        ]

        from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_config.ee_traj import EndEffectorTracker
        ee_tracker = EndEffectorTracker(constraint_specs,  self.env.scene.env_regex_ns)
        # from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_config.ee_traj import read_ee_pose
        # ee_pose = read_ee_pose("left_hand_palm_joint")
        import pdb; pdb.set_trace()


class EndEffectorTracker:
    def __init__(self, constraint_specs: List[Dict], env_ns: str = "/World/envs/env_0/Robot"):
        self.env_ns = env_ns
        self.constraint_specs = constraint_specs
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        self.ee_views = {}  # key: frame name, value: prim view (XFormPrim or physics view)
        self._initialize_views()

    def _initialize_views(self):
        
        from pxr import Usd
        import omni.usd

          # Get the USD stage
        stage = omni.usd.get_context().get_stage()

          # Traverse all prims in the scene
        for prim in stage.Traverse():
          print(prim.GetPath())  # full prim path, e.g., /World/envs/env_0/Robot/left_hand_palm_joint


        for spec in self.constraint_specs:
            if "frame" not in spec:
                continue

            frame_name = spec["frame"]
            full_path = f"{self.env_ns}/Robot/{frame_name}"
       
            prim = sim_utils.find_first_matching_prim(full_path)

            if prim is None:
                raise ValueError(f"Prim {full_path} not found.")

            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                view = self._physics_sim_view.create_articulation_view(full_path)
            elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
                view = self._physics_sim_view.create_rigid_body_view(full_path)
            else:
                view = XFormPrim(full_path, reset_xform_properties=False)
            self.ee_views[frame_name] = view

    def get_pose(self, frame_name: str):
        """Returns (position, euler_orientation) for a given EE frame."""
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
