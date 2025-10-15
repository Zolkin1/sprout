# stones_terrain_importer.py
from isaaclab.terrains import TerrainImporter
from robot_rl.tasks.manager_based.robot_rl.terrains.stones_terrain_generator import StonesTerrainGenerator
import torch

class StonesTerrainImporter(TerrainImporter):
    
    """Simplified importer for stones terrain."""
    env_terrain_infos: dict[str, torch.Tensor]
    
    def __init__(self, cfg):
        # safely call base class init if it does useful setup
        super().__init__(cfg)
        self.cfg = cfg
        # Instantiate your generator directly
        self.terrain_generator = StonesTerrainGenerator(cfg.terrain_generator)
        self.terrain_mesh = self.terrain_generator.terrain_mesh
        self.terrain_origins = self.terrain_generator.terrain_origins
        # For Isaac Lab env origin setup
        self.configure_env_origins(self.terrain_origins)
        self.terrain_infos = getattr(self.terrain_generator, "terrain_infos", {})
        
        # Initialize env_terrain_infos as an actual dictionary
        self.env_terrain_infos: dict[str, torch.Tensor] = {}
        if self.cfg.terrain_generator.curriculum == False:
            raise RuntimeError("StonesTerrainImporter only supports curriculum learning currently.")

        self.configure_env_infos(self.terrain_infos)

    def configure_env_infos(self, terrain_infos):
        """Configure environment with terrain info."""
        terrain_infos_tensor = {k: torch.tensor(v, dtype=torch.float32, device=self.device) for k, v in terrain_infos.items()}
        self._compute_env_infos_curriculum(self.cfg.num_envs, terrain_infos_tensor)
        
    def _compute_env_infos_curriculum(self, num_envs: int, infos: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute environment infos for curriculum learning."""
        self.env_terrain_infos["rel_x"] = torch.zeros((num_envs,infos["rel_x"].shape[-1]), dtype=torch.float32, device=self.device)
        self.env_terrain_infos["rel_z"] = torch.zeros((num_envs,infos["rel_z"].shape[-1]), dtype=torch.float32, device=self.device)
        self.env_terrain_infos["start_platform_pos"] = torch.zeros((num_envs, 3), dtype=torch.float32, device=self.device)
        self.env_terrain_infos["origin"] = torch.zeros((num_envs, 3), dtype=torch.float32, device=self.device)

        self.env_terrain_infos["rel_x"][:] = infos["rel_x"][self.terrain_levels, self.terrain_types]
        self.env_terrain_infos["rel_z"][:] = infos["rel_z"][self.terrain_levels, self.terrain_types]
        self.env_terrain_infos["start_platform_pos"][:] = infos["start_platform_pos"][self.terrain_levels, self.terrain_types]
        self.env_terrain_infos["origin"][:] = infos["origin"][self.terrain_levels, self.terrain_types]

        return infos
    