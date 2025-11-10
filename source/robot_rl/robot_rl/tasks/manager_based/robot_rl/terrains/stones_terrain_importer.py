# stones_terrain_importer.py
from isaaclab.terrains import TerrainImporter
from robot_rl.tasks.manager_based.robot_rl.terrains.stones_terrain_generator import StonesTerrainGenerator
import torch
import isaaclab.sim as sim_utils
from pxr import UsdGeom, UsdPhysics
class StonesTerrainImporter(TerrainImporter):
    
    """Simplified importer for stones terrain."""
    env_terrain_infos: dict[str, torch.Tensor]
    
    def __init__(self, cfg):
        # check that the config is valid
        cfg.validate()
        # store inputs
        self.cfg = cfg
        self.device = sim_utils.SimulationContext.instance().device  # type: ignore
        
        # create buffers for the terrains
        self.terrain_prim_paths = list()
        self.terrain_origins = None
        self.env_origins = None  # assigned later when `configure_env_origins` is called
        # private variables
        self._terrain_flat_patches = dict()
        # auto-import the terrain based on the config
        if self.cfg.terrain_type == "generator":
            # check config is provided
            if self.cfg.terrain_generator is None:
                raise ValueError("Input terrain type is 'generator' but no value provided for 'terrain_generator'.")
            # generate the terrain
            terrain_generator = self.cfg.terrain_generator.class_type(
                cfg=self.cfg.terrain_generator, device=self.device
            )
            self.import_mesh("terrain_stones", terrain_generator.terrain_mesh)
            
            # configure the terrain origins based on the terrain generator
            self.configure_env_origins(terrain_generator.terrain_origins)
            # refer to the flat patches
            self._terrain_flat_patches = terrain_generator.flat_patches
            
            # self._configure_stepping_stones_visual()
            # self.import_ground_plane("terrain_flat")
            # self._configure_flat_collision()
            
            
        else:
            raise ValueError(f"Unknown terrain type '{self.cfg.terrain_type}', must be 'generator'.")

        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

        self.terrain_infos = getattr(terrain_generator, "terrain_infos", {})
        
        # Initialize env_terrain_infos as an actual dictionary
        self.env_terrain_infos: dict[str, torch.Tensor] = {}
        if self.cfg.terrain_generator.curriculum == False:
            raise RuntimeError("StonesTerrainImporter only supports curriculum learning currently.")
        #assuming _compute_env_origins_curriculum has been called in configure_env_origins
        self.configure_env_infos(self.terrain_infos)
        
    def _configure_flat_collision(self):
        """Configure the flat ground plane to be invisible but with collision."""
        stage = sim_utils.SimulationContext.instance().stage
        
        # Get the terrain_flat prim path
        flat_prim_path = f"{self.cfg.prim_path}/terrain_flat"
        flat_prim = stage.GetPrimAtPath(flat_prim_path)
        
        if flat_prim:
            # Make invisible (only for collision, not rendering)
            imageable = UsdGeom.Imageable(flat_prim)
            imageable.MakeInvisible()
            # imageable.MakeVisible()
            # Ensure collision is enabled (should be by default from import_ground_plane)
            # if not flat_prim.HasAPI(UsdPhysics.CollisionAPI):
            #     UsdPhysics.CollisionAPI.Apply(flat_prim)
            
            print(f"Configured flat collision plane at {flat_prim_path} (invisible, collision enabled)")
    
    def _configure_stepping_stones_visual(self):
        """Configure the stepping stones mesh to be visual only (no collision)."""
        stage = sim_utils.SimulationContext.instance().stage
        
        # Get the terrain_stones prim path
        stones_prim_path = f"{self.cfg.prim_path}/terrain_stones"
        stones_prim = stage.GetPrimAtPath(stones_prim_path)
        
        if stones_prim:
            # Disable collision on the stepping stones
            if stones_prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI(stones_prim)
                collision_api.GetCollisionEnabledAttr().Set(False)
            
            # Ensure it's visible for raycasting and rendering
            imageable = UsdGeom.Imageable(stones_prim)
            imageable.MakeVisible()
            
            
            print(f"Configured stepping stones at {stones_prim_path} (visible, collision disabled)")

    def configure_env_infos(self, terrain_infos):
        """Configure environment with terrain info."""
        self.terrain_infos_tensor = {k: torch.tensor(v, dtype=torch.float32, device=self.device) for k, v in terrain_infos.items()}
        self._compute_env_infos_curriculum(self.cfg.num_envs, self.terrain_infos_tensor)
        
    def _compute_env_infos_curriculum(self, num_envs: int, infos: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute environment infos for curriculum learning."""
        self.env_terrain_infos["rel_x"] = torch.zeros((num_envs,infos["rel_x"].shape[-1]), dtype=torch.float32, device=self.device)
        self.env_terrain_infos["rel_z"] = torch.zeros((num_envs,infos["rel_z"].shape[-1]), dtype=torch.float32, device=self.device)
        self.env_terrain_infos["start_stone_pos"] = torch.zeros((num_envs, 3), dtype=torch.float32, device=self.device)
        self.env_terrain_infos["stone_x"] = torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device)

        self.env_terrain_infos["rel_x"][:] = infos["rel_x"][self.terrain_levels, self.terrain_types]
        self.env_terrain_infos["rel_z"][:] = infos["rel_z"][self.terrain_levels, self.terrain_types]
        self.env_terrain_infos["start_stone_pos"][:] = infos["start_stone_pos"][self.terrain_levels, self.terrain_types]
        self.env_terrain_infos["stone_x"][:] = infos["stone_x"][self.terrain_levels, self.terrain_types]

        return infos
    
    def update_env_origins_and_infos(self, env_ids: torch.Tensor, move_up: torch.Tensor, move_down: torch.Tensor):
        """Update the environment origins based on the terrain levels."""
        # check if grid-like spawning
        if self.terrain_origins is None:
            return
        # update terrain level for the envs
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # robots that solve the last level are sent to a random one
        # the minimum level is zero
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0),
        )
        # update the env origins
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        self.env_terrain_infos["rel_x"][:] = self.terrain_infos_tensor["rel_x"][self.terrain_levels, self.terrain_types]
        self.env_terrain_infos["rel_z"][:] = self.terrain_infos_tensor["rel_z"][self.terrain_levels, self.terrain_types]
        self.env_terrain_infos["start_stone_pos"][:] = self.terrain_infos_tensor["start_stone_pos"][self.terrain_levels, self.terrain_types]
        self.env_terrain_infos["stone_x"][:] = self.terrain_infos_tensor["stone_x"][self.terrain_levels, self.terrain_types]
