# stones_terrain_importer.py
from isaaclab.terrains import TerrainImporter
from robot_rl.tasks.manager_based.robot_rl.terrains.stones_terrain_generator import StonesTerrainGenerator
import torch
import isaaclab.sim as sim_utils
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
            self.import_mesh("terrain", terrain_generator.terrain_mesh)
            # configure the terrain origins based on the terrain generator
            self.configure_env_origins(terrain_generator.terrain_origins)
            # refer to the flat patches
            self._terrain_flat_patches = terrain_generator.flat_patches
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

    def configure_env_infos(self, terrain_infos):
        """Configure environment with terrain info."""
        terrain_infos_tensor = {k: torch.tensor(v, dtype=torch.float32, device=self.device) for k, v in terrain_infos.items()}
        self._compute_env_infos_curriculum(self.cfg.num_envs, terrain_infos_tensor)
        
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
    