# stones_terrain_importer.py
from isaaclab.terrains import TerrainImporter
from robot_rl.tasks.manager_based.robot_rl.terrains.stones_terrain_generator import StonesTerrainGenerator

class StonesTerrainImporter(TerrainImporter):
    """Simplified importer for stones terrain."""

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
        self.terrain_info = getattr(self.terrain_generator, "terrain_info", {})
