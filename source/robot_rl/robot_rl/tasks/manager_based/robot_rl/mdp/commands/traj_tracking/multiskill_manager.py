# Copyright 2026 Zachary Olkin. All rights reserved.

"""MultiSkillManager: batched trajectory management across multiple skills.

Replaces the loop-per-trajectory pattern in LibraryManager with fully batched
tensor operations. Supports multi-dimensional conditioning (velocity, terrain).
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict
from torch import Tensor

from .library_fingerprint import fingerprint_skill_folders
from .manager_base import ManagerBase
from .sagittal_reflector import swap_left_right
from .trajectory_manager import TrajectoryManager, TrajectoryType

# Bump whenever the on-disk schema of the merged-tensor cache changes — i.e.
# whenever the payload dict written by ``MultiSkillManager._save_library_cache``
# (the ``self.data`` keys, the per-domain metadata lists, or the tau helpers
# produced by ``_build_global_tensors``) changes shape or meaning.  A bump
# invalidates every existing cache file so stale layouts are never restored.
# v2: ConditionerData gained per-terrain-dim (lo, hi) bounds (range support).
_LIBRARY_CACHE_VERSION = 2


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ConditionerData:
    """Structured conditioning information parsed from a trajectory YAML.

    A terrain dimension (``terrain_width`` / ``terrain_height`` /
    ``terrain_length``) may be declared either as a single number or as a
    ``{min, max}`` *range* (with either bound optional).  A range marks the
    dimension a *don't-care* for this trajectory: it was optimized to work for
    any terrain value inside the range.  The ``terrain_*`` fields hold a
    representative scalar (the fixed value, or a finite bound / the midpoint of
    a range) used for the conditioning tensor and the terrain mask; the paired
    ``terrain_*_bounds`` ``(lo, hi)`` carry the full extent (``lo == hi`` for a
    fixed value; ``hi == +inf`` / ``lo == 0`` for an open range).  The bounds
    drive the range-aware terrain-dim selection filter and the offline
    generator's per-terrain dimension sampling.
    """

    vel_x: float = 0.0
    vel_y: float = 0.0
    vel_yaw: float = 0.0
    terrain: str = "flat"
    terrain_width: float = 0.0
    terrain_height: float = 0.0
    terrain_length: float = 0.0
    terrain_width_bounds: tuple[float, float] = (0.0, 0.0)
    terrain_height_bounds: tuple[float, float] = (0.0, 0.0)
    terrain_length_bounds: tuple[float, float] = (0.0, 0.0)

    def to_continuous_tensor(self) -> list[float]:
        """Return the continuous conditioning dimensions as a flat list."""
        return [
            self.vel_x, self.vel_y, self.vel_yaw,
            self.terrain_width, self.terrain_height, self.terrain_length,
        ]

    def terrain_bounds_lo(self) -> list[float]:
        """Lower bounds ``[width, height, length]`` of the terrain dims."""
        return [
            self.terrain_width_bounds[0],
            self.terrain_height_bounds[0],
            self.terrain_length_bounds[0],
        ]

    def terrain_bounds_hi(self) -> list[float]:
        """Upper bounds ``[width, height, length]`` (``+inf`` for open ranges)."""
        return [
            self.terrain_width_bounds[1],
            self.terrain_height_bounds[1],
            self.terrain_length_bounds[1],
        ]


@dataclass
class SkillData:
    """Metadata for a single skill (a group of related trajectories)."""

    name: str
    traj_indices: list[int] = field(default_factory=list)
    num_trajectories: int = 0
    has_terrain: bool = False
    conditioning_tensor: Tensor | None = None  # [num_traj, C] continuous dims


# ---------------------------------------------------------------------------
# Trajectory type int encoding (matches TrajectoryType enum order)
# ---------------------------------------------------------------------------

_TRAJ_TYPE_TO_INT = {
    TrajectoryType.HALF_PERIODIC: 0,
    TrajectoryType.FULL_PERIODIC: 1,
    TrajectoryType.EPISODIC: 2,
    TrajectoryType.PERPETUAL: 3,
}

_HALF_PERIODIC_INT = _TRAJ_TYPE_TO_INT[TrajectoryType.HALF_PERIODIC]
_FULL_PERIODIC_INT = _TRAJ_TYPE_TO_INT[TrajectoryType.FULL_PERIODIC]
_EPISODIC_INT = _TRAJ_TYPE_TO_INT[TrajectoryType.EPISODIC]
_PERPETUAL_INT = _TRAJ_TYPE_TO_INT[TrajectoryType.PERPETUAL]

# Terrain-dim matching tolerances (width, height, length — same ordering as
# the trajectory conditioner's terrain columns) for the per-env trajectory
# filter (see :meth:`MultiSkillManager._terrain_dim_allowed`).  Heights are
# exact in both the library and the terrain; lengths absorb the legacy stair
# generator's tile-to-cell depth rescale while staying far below the
# library's grid spacing.  Width is currently always unconstrained (NaN from
# the terrain query) so its tolerance is a placeholder.
_TERRAIN_DIM_TOLS = (0.010, 0.005, 0.010)


# ---------------------------------------------------------------------------
# MultiSkillManager
# ---------------------------------------------------------------------------
class MultiSkillManager(ManagerBase):
    """Manages trajectories across multiple skills with fully batched evaluation.

    All trajectory data is flattened into global tensors so that every method
    (get_output, get_contact_state, …) runs as a single batched operation
    over all environments — no per-trajectory Python loops at runtime.

    Skills are groups of trajectories that share a semantic label (e.g.
    "walking", "running", "stairs").  Within each skill, trajectory selection
    uses nearest-neighbour lookup on continuous conditioning dimensions
    (vel_x, vel_y, vel_yaw, terrain dims).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        path: str,
        device: str | torch.device,
        env: Any = None,
        conditioner_generator_name: str | None = None,
        hf_repo: str | None = None,
        use_cache: bool = True,
    ):
        """Load trajectory YAMLs from a top-level folder containing per-skill subfolders.

        The expected layout is::

            path/
            ├── walking/
            │   ├── walk_20.yaml
            │   └── walk_40.yaml
            ├── running/
            │   ├── run_160.yaml
            │   └── run_180.yaml
            └── ...

        Each immediate subdirectory of ``path`` becomes a skill, named after
        the subdirectory.  The velocity command's ``_skill_list`` may be a
        strict subset of the loaded skills — extras are loaded into the
        pool but never sampled (see :meth:`_lazy_init_skill_filter`).

        Args:
            path: Top-level folder containing one subfolder per skill.
                If ``hf_repo`` is set, this is relative to the HF repo root.
            device: Torch device.
            env: IsaacLab environment (needed for conditioner queries at
                runtime).  Can be ``None`` for offline / test usage.
            conditioner_generator_name: Name of the command term that provides
                the conditioning signal at runtime.
            hf_repo: Optional HuggingFace repo ID (e.g. ``"zolkin/robot_rl"``).
                When set, ``path`` is treated as a path within the repo and
                downloaded to a local ``hf/`` cache directory.
            use_cache: When True (default), cache the merged global tensors to
                disk keyed by a content hash of the library's YAML files, so an
                unchanged library skips the (~tens-of-seconds) YAML parse on the
                next launch.  See :meth:`_load_or_build_library`.
        """
        self.device = torch.device(device)
        self.env = env
        self.conditioner_generator_name = conditioner_generator_name

        # --- Discover skill folders from subdirectories --------------------
        skill_folders = self._discover_skill_folders(path, hf_repo)

        # --- Portable content fingerprint of the loaded library ------------
        # Machine-independent (relative layout + file contents only), unlike
        # the local tensor-cache key.  The owning command writes this back onto
        # its cfg so it is serialised into ``params/env.yaml``; at play / resume
        # it is compared against the trained-against value to catch a library
        # that has silently diverged from the one the policy trained on.
        self.library_fingerprint: dict = fingerprint_skill_folders(skill_folders)
        print(
            f"[MultiSkillManager] trajectory library '{path}' fingerprint "
            f"{self.library_fingerprint['short']} "
            f"({self.library_fingerprint['num_trajectories']} trajectories: "
            f"{self.library_fingerprint['skill_counts']})"
        )

        # --- Load cached merged tensors, or build them from the YAMLs ------
        # Parsing ~1.5k trajectory YAMLs into the merged global tensors costs
        # tens of seconds on every launch.  Cache the built result keyed by a
        # content hash of the library's YAML files so an unchanged library
        # loads near-instantly; any added / removed / edited YAML changes the
        # key and forces a rebuild.  Sets ``self.data`` and the metadata that
        # :meth:`_build_global_tensors` would, and returns the post-sort
        # ``(skill_labels, conditioner_datas)`` for the per-skill build below.
        skill_labels, conditioner_datas = self._load_or_build_library(
            skill_folders, device, use_cache,
        )

        # --- Build per-skill metadata --------------------------------------
        self._build_skill_data(skill_labels, conditioner_datas)

        # --- Skill filter state (lazy-init at first selection) -------------
        # ``_traj_skill_idx`` is the per-trajectory skill index (built by
        # ``_build_global_tensors`` at ``data["skill_idx"]``). Cached here as a
        # plain Tensor for the per-env skill mask in ``_select_trajectories``.
        # The vel→traj skill-id gather and the alignment check are deferred to
        # ``_lazy_init_skill_filter`` because the conditioner command term may
        # not exist on ``env.command_manager`` at manager-construction time.
        self._traj_skill_idx: Tensor = self.data["skill_idx"].to(self.device)
        self._vel_skill_to_traj_skill: Tensor | None = None
        self._skill_filter_ready: bool = False

        # Owner that exposes the per-env active ``skill_id`` tensor and the
        # ordered ``_skill_list`` for the skill-filter validation.  Set by
        # :meth:`set_skill_owner`, typically called from the owning
        # trajectory cmd's ``_post_init``.  Lazy-init runs against this
        # owner instead of the velocity (conditioner) term — the velocity
        # term is still used as the ``command`` source via
        # ``conditioner_generator_name``.
        self._skill_owner: Any = None
        # Optional terrain lookup ``fn(xy_w) -> (dims[N, 3], valid[N])``
        # returning the (width, height, length) of the terrain feature at
        # each world xy (NaN = dimension unconstrained).  Registered via
        # :meth:`set_terrain_dim_lookup`; ``None`` (default) disables the
        # terrain-dim selection filter (offline / flat-only / test usage).
        self._terrain_dims_at: Any = None
        self._terrain_dim_tols = torch.tensor(
            _TERRAIN_DIM_TOLS, dtype=torch.float32, device=self.device,
        )
        self._warned_dim_fallback: bool = False

        # --- Pre-compute binomial coefficients for Bezier evaluation -------
        self._binomial_coeffs: dict[int, Tensor] = {}
        for d in range(self.spline_order + 1):
            self._binomial_coeffs[d] = torch.tensor(
                [math.comb(d, i) for i in range(d + 1)],
                dtype=torch.float32, device=self.device,
            )

        # --- Cache for per-step trajectory assignment ----------------------
        self._cache_valid = False
        self._cached_global_indices: Tensor | None = None
        self._cached_env_ids: Tensor | None = None

        # --- Trajectory transition tracking --------------------------------
        # ``_traj_changed`` is populated by :meth:`_ensure_cache` on each
        # cache rebuild and read by the owning command term to re-arm the
        # contact gate.  Snapshot is committed by :meth:`commit_traj_state`,
        # not eagerly inside ``_ensure_cache`` — see that method's docstring.
        self._prev_global_indices: Tensor | None = None
        self._traj_changed: Tensor | None = None

        # --- Contact-gate metadata ----------------------------------------
        self._build_gate_tables()

        # --- Per-env phase + gate state -----------------------------------
        # Allocated lazily (we may not know num_envs at construction time
        # for offline / test usage).  ``phase`` lives in [0, 1]; advances
        # by ``step_dt / total_time`` each step (wrap for periodic, clamp
        # for episodic, perpetual stays at 0).  ``next_gate_idx`` is the
        # contact-gate currently armed per env, or -1 if none.
        # ``gate_rel_phi`` is signed phi-distance from the armed gate
        # (positive = past, negative = before), accumulated monotonically
        # — does NOT wrap with ``phase``.  Re-anchored on every gate
        # (re)arm by :meth:`_refresh_gate_rel_phi`.
        # ``gate_rel_phi_pre_snap`` captures ``gate_rel_phi`` immediately
        # after :meth:`update_phase` runs and *before* any contact-gate
        # snap mutates it.  It's the value the gate logic actually
        # decides on, so it's what the data-logger / play plots should
        # display to show "how early/late was the contact when the fire
        # happened."
        self.phase: Tensor | None = None
        self.next_gate_idx: Tensor | None = None
        self.gate_rel_phi: Tensor | None = None
        self.gate_rel_phi_pre_snap: Tensor | None = None

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _discover_skill_folders(
        path: str, hf_repo: str | None
    ) -> dict[str, Path]:
        """Discover per-skill subfolders under a top-level path.

        If ``hf_repo`` is set, downloads the folder from HuggingFace first.
        Each immediate subdirectory that contains YAML files becomes a skill.

        Args:
            path: Top-level folder (local or HF-relative).
            hf_repo: Optional HuggingFace repo ID.

        Returns:
            Dict mapping skill name to resolved local ``Path``.
        """
        if hf_repo is not None:
            resolved = MultiSkillManager._download_from_hf(hf_repo, path)
        else:
            resolved = Path(path)

        if not resolved.exists():
            raise FileNotFoundError(f"Skill root folder not found: {resolved}")
        if not resolved.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {resolved}")

        skill_folders: dict[str, Path] = {}
        for child in sorted(resolved.iterdir()):
            if not child.is_dir():
                continue
            yamls = list(child.glob("*.yaml")) + list(child.glob("*.yml"))
            if len(yamls) > 0:
                skill_folders[child.name] = child

        if len(skill_folders) == 0:
            # Fall back: maybe the path itself contains YAMLs (single-skill)
            yamls = list(resolved.glob("*.yaml")) + list(resolved.glob("*.yml"))
            if len(yamls) > 0:
                skill_folders["default"] = resolved
            else:
                raise ValueError(
                    f"No skill subfolders with YAML files found under: {resolved}"
                )

        return skill_folders

    @staticmethod
    def _download_from_hf(hf_repo: str, hf_path: str) -> Path:
        """Download a folder from HuggingFace to a local cache.

        Args:
            hf_repo: HuggingFace repo ID (e.g. ``"zolkin/robot_rl"``).
            hf_path: Path within the repo to download.

        Returns:
            Local ``Path`` to the downloaded folder.
        """
        root = os.path.abspath(os.getcwd())
        cache_dir = os.path.join(root, "hf")
        os.makedirs(cache_dir, exist_ok=True)

        local_folder = os.path.join(cache_dir, hf_path)

        # Check if already cached
        if os.path.isdir(local_folder):
            # Check for YAMLs in subdirectories or directly
            has_yamls = (
                list(Path(local_folder).glob("*.yaml"))
                + list(Path(local_folder).glob("*.yml"))
                + list(Path(local_folder).glob("**/*.yaml"))
            )
            if len(has_yamls) > 0:
                print(f"Using cached trajectories from {local_folder}")
                return Path(local_folder)

        try:
            from huggingface_hub import snapshot_download

            print(f"Downloading {hf_path} from {hf_repo}...")
            # Use multiple patterns: fnmatch doesn't support **
            # Match flat YAMLs and one level of skill subdirectories
            snapshot_download(
                repo_id=hf_repo,
                allow_patterns=[
                    f"{hf_path}/*",      # flat files
                    f"{hf_path}/*/*",    # files in skill subdirectories
                ],
                local_dir=cache_dir,
            )
            print(f"Successfully downloaded to {local_folder}")
            return Path(local_folder)

        except ImportError:
            raise RuntimeError(
                "huggingface_hub is required for downloading trajectories. "
                "Install with: pip install huggingface_hub"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download from HuggingFace: {e}"
            )

    @staticmethod
    def _find_yaml_files(folder_path: str) -> list[Path]:
        """Find all YAML files in a folder."""
        p = Path(folder_path)
        if not p.exists():
            raise FileNotFoundError(f"Skill folder not found: {folder_path}")
        if not p.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {folder_path}")
        yamls = sorted(p.glob("*.yaml")) + sorted(p.glob("*.yml"))
        if len(yamls) == 0:
            raise ValueError(f"No YAML trajectory files in: {folder_path}")
        return yamls

    @staticmethod
    def _parse_terrain_dim(raw: Any) -> tuple[float, tuple[float, float]]:
        """Parse one terrain dimension into ``(representative, (lo, hi))``.

        A plain number is a fixed value (``lo == hi == value``).  A
        ``{min, max}`` dict is a range (a *don't-care* dimension); an omitted
        ``min`` defaults to ``0.0`` and an omitted ``max`` to ``+inf``.  The
        representative scalar (for the conditioning tensor / terrain mask) is
        the midpoint of a finite range, the single finite bound of a half-open
        range, or the fixed value.
        """
        if isinstance(raw, dict):
            lo = float(raw.get("min", 0.0))
            hi = float(raw.get("max", float("inf")))
            if math.isfinite(lo) and math.isfinite(hi):
                rep = 0.5 * (lo + hi)
            elif math.isfinite(lo):
                rep = lo
            elif math.isfinite(hi):
                rep = hi
            else:
                rep = 0.0
            return rep, (lo, hi)
        v = float(raw)
        return v, (v, v)

    @staticmethod
    def _parse_conditioner(raw_conditioner: Any, skill_name: str) -> ConditionerData:
        """Parse a conditioner field from YAML into ConditionerData.

        Supports both old format (list of floats) and new format (dict).  Each
        terrain dimension in the dict form may itself be a single number or a
        ``{min, max}`` range (see :meth:`_parse_terrain_dim`).
        """
        if isinstance(raw_conditioner, list):
            # Old format: [min, max] — use midpoint as vel_x, zeros elsewhere
            mid = sum(raw_conditioner) / len(raw_conditioner)
            return ConditionerData(vel_x=mid)
        elif isinstance(raw_conditioner, dict):
            w_rep, w_bounds = MultiSkillManager._parse_terrain_dim(
                raw_conditioner.get("terrain_width", 0.0)
            )
            h_rep, h_bounds = MultiSkillManager._parse_terrain_dim(
                raw_conditioner.get("terrain_height", 0.0)
            )
            l_rep, l_bounds = MultiSkillManager._parse_terrain_dim(
                raw_conditioner.get("terrain_length", 0.0)
            )
            return ConditionerData(
                vel_x=raw_conditioner.get("vel_x", 0.0),
                vel_y=raw_conditioner.get("vel_y", 0.0),
                vel_yaw=raw_conditioner.get("vel_yaw", 0.0),
                terrain=raw_conditioner.get("terrain", "flat"),
                terrain_width=w_rep,
                terrain_height=h_rep,
                terrain_length=l_rep,
                terrain_width_bounds=w_bounds,
                terrain_height_bounds=h_bounds,
                terrain_length_bounds=l_bounds,
            )
        else:
            raise ValueError(
                f"Unrecognised conditioner format for skill '{skill_name}': "
                f"{type(raw_conditioner)}"
            )

    # ------------------------------------------------------------------
    # Library load / merged-tensor cache
    # ------------------------------------------------------------------

    def _load_or_build_library(
        self,
        skill_folders: dict[str, Path],
        device: str | torch.device,
        use_cache: bool,
    ) -> tuple[list[str], list[ConditionerData]]:
        """Populate the merged global tensors, from cache when possible.

        Returns ``(skill_labels, conditioner_datas)`` in post-sort global
        order for the downstream :meth:`_build_skill_data` call.  On a cache
        hit the YAML parse + :meth:`_build_global_tensors` are skipped
        entirely; on a miss (or with caching disabled) the full build runs and
        its result is written back to the cache.  All cache I/O is
        best-effort — any failure falls back to building from YAML.
        """
        cache_path: Path | None = None
        cache_key: str | None = None
        if use_cache:
            try:
                yaml_paths = self._gather_library_yaml_paths(skill_folders)
                cache_key = self._library_cache_key(yaml_paths)
                cache_path = self._library_cache_path(cache_key)
            except Exception as e:  # noqa: BLE001 - cache is best-effort
                print(
                    "[WARN] MultiSkillManager: could not compute library cache "
                    f"key ({e}); loading from YAML without caching."
                )
                cache_path = None
            if cache_path is not None:
                restored = self._try_load_library_cache(cache_path, cache_key)
                if restored is not None:
                    return restored

        skill_labels, conditioner_datas = self._build_from_yaml(skill_folders, device)
        if cache_path is not None:
            self._save_library_cache(
                cache_path, cache_key, skill_labels, conditioner_datas
            )
        return skill_labels, conditioner_datas

    def _build_from_yaml(
        self,
        skill_folders: dict[str, Path],
        device: str | torch.device,
    ) -> tuple[list[str], list[ConditionerData]]:
        """Parse every trajectory YAML and build the merged global tensors.

        Sets the same instance attributes the cache path restores
        (``trajectory_names`` / ``skill_labels``, the output-name & count
        metadata, ``num_trajectories``, and — via
        :meth:`_build_global_tensors` — ``self.data`` plus the per-domain and
        tau helpers).  Returns ``(skill_labels, conditioner_datas)`` in
        post-sort global order.
        """
        # --- Load all trajectories via TrajectoryManager (then discard) ----
        managers: list[TrajectoryManager] = []
        skill_labels: list[str] = []
        conditioner_datas: list[ConditionerData] = []

        for skill_name, folder_path in skill_folders.items():
            yaml_paths = self._find_yaml_files(str(folder_path))
            for yp in yaml_paths:
                mgr = TrajectoryManager(str(yp), None, device)
                managers.append(mgr)
                skill_labels.append(skill_name)
                conditioner_datas.append(
                    self._parse_conditioner(mgr.traj_data.conditioner, skill_name)
                )

        if len(managers) == 0:
            raise ValueError("No trajectory YAML files found in any skill folder.")

        # --- Sort trajectories by conditioner within each skill ------------
        # This matches LibraryManager's sorting convention and ensures
        # consistent indexing.
        combined = list(zip(managers, skill_labels, conditioner_datas))
        combined.sort(key=lambda x: (x[1], x[2].vel_x))
        managers = [c[0] for c in combined]
        skill_labels = [c[1] for c in combined]
        conditioner_datas = [c[2] for c in combined]

        # --- Capture per-trajectory names (for stats reporting) ------------
        # Stored in the same global order as ``self._global_conditioning``.
        self.trajectory_names: list[str] = [m.traj_data.name for m in managers]
        self.skill_labels: list[str] = list(skill_labels)

        # --- Verify compatibility across all trajectories ------------------
        ref = managers[0]
        self.pos_output_names = list(ref.traj_data.pos_output_names)
        self.vel_output_names = list(ref.traj_data.vel_output_names)
        self.num_pos_outputs = ref.traj_data.num_pos_outputs
        self.num_vel_outputs = ref.traj_data.num_vel_outputs
        self.spline_order = ref.traj_data.spline_order
        self.ref_frames = list(ref.traj_data.reference_frames)

        for i, mgr in enumerate(managers):
            td = mgr.traj_data
            if td.num_pos_outputs != self.num_pos_outputs:
                raise ValueError(
                    f"Trajectory {td.name} has {td.num_pos_outputs} pos outputs, "
                    f"expected {self.num_pos_outputs}."
                )
            if td.num_vel_outputs != self.num_vel_outputs:
                raise ValueError(
                    f"Trajectory {td.name} has {td.num_vel_outputs} vel outputs, "
                    f"expected {self.num_vel_outputs}."
                )
            if td.pos_output_names != self.pos_output_names:
                raise ValueError(
                    f"Trajectory {td.name} pos output names mismatch.\n"
                    f"Expected: {self.pos_output_names}\nGot: {td.pos_output_names}"
                )
            if td.vel_output_names != self.vel_output_names:
                raise ValueError(
                    f"Trajectory {td.name} vel output names mismatch.\n"
                    f"Expected: {self.vel_output_names}\nGot: {td.vel_output_names}"
                )

        # --- Build global flat tensors from managers -----------------------
        self.num_trajectories = len(managers)
        self._build_global_tensors(managers, skill_labels, conditioner_datas)
        return skill_labels, conditioner_datas

    @staticmethod
    def _gather_library_yaml_paths(skill_folders: dict[str, Path]) -> list[Path]:
        """All trajectory YAML paths across every skill folder, sorted.

        Mirrors the set the YAML build actually loads (via
        :meth:`_find_yaml_files`), so the cache key sees exactly the files
        that would be parsed.
        """
        paths: list[Path] = []
        for folder in skill_folders.values():
            paths.extend(MultiSkillManager._find_yaml_files(str(folder)))
        return sorted(paths)

    @staticmethod
    def _library_cache_key(yaml_paths: list[Path]) -> str:
        """Content hash of the library: per-file (path, size, mtime) + version.

        Any added / removed / edited / touched YAML changes the digest, so a
        stale cache can never match a mutated library.
        """
        h = hashlib.sha256()
        h.update(f"version={_LIBRARY_CACHE_VERSION};".encode())
        for p in yaml_paths:
            st = p.stat()
            h.update(str(p).encode())
            h.update(f":{st.st_size}:{st.st_mtime_ns};".encode())
        return h.hexdigest()

    @staticmethod
    def _library_cache_path(cache_key: str) -> Path:
        """Filesystem location of the content-addressed cache file.

        Defaults to ``~/.cache/robot_rl/msm_tensors/<key>.pt`` (cwd-independent,
        never inside the library or source tree).  Override the directory with
        the ``ROBOT_RL_TRAJ_CACHE_DIR`` environment variable (used by tests to
        avoid touching the user cache).
        """
        cache_dir = os.environ.get("ROBOT_RL_TRAJ_CACHE_DIR")
        base = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "robot_rl" / "msm_tensors"
        return base / f"{cache_key}.pt"

    def _try_load_library_cache(
        self, cache_path: Path, cache_key: str | None
    ) -> tuple[list[str], list[ConditionerData]] | None:
        """Restore the merged tensors from ``cache_path`` if valid.

        Returns ``(skill_labels, conditioner_datas)`` on success, or ``None``
        on any miss / mismatch / read error (caller then builds from YAML).
        """
        if not cache_path.exists():
            return None
        try:
            payload = torch.load(
                str(cache_path), map_location="cpu", weights_only=False
            )
        except Exception as e:  # noqa: BLE001 - corrupt / unreadable cache
            print(
                "[WARN] MultiSkillManager: failed to read tensor cache at "
                f"{cache_path} ({e}); rebuilding from YAML."
            )
            return None
        if (
            not isinstance(payload, dict)
            or payload.get("version") != _LIBRARY_CACHE_VERSION
            or payload.get("key") != cache_key
        ):
            return None
        try:
            return self._restore_from_cache_payload(payload)
        except Exception as e:  # noqa: BLE001 - schema drift / partial payload
            print(
                "[WARN] MultiSkillManager: tensor cache restore failed "
                f"({e}); rebuilding from YAML."
            )
            return None

    def _restore_from_cache_payload(
        self, payload: dict
    ) -> tuple[list[str], list[ConditionerData]]:
        """Reconstruct the merged-tensor state from a cache payload dict."""
        self.num_trajectories = int(payload["num_trajectories"])
        self.max_expanded_domains = int(payload["max_expanded_domains"])
        self.pos_output_names = list(payload["pos_output_names"])
        self.vel_output_names = list(payload["vel_output_names"])
        self.num_pos_outputs = int(payload["num_pos_outputs"])
        self.num_vel_outputs = int(payload["num_vel_outputs"])
        self.spline_order = int(payload["spline_order"])
        self.ref_frames = list(payload["ref_frames"])
        self.trajectory_names = list(payload["trajectory_names"])
        self.skill_labels = list(payload["skill_labels"])
        self._contact_bodies_per_domain = payload["contact_bodies_per_domain"]
        self._ref_frame_names_per_domain = payload["ref_frame_names_per_domain"]
        self._original_domain_times = payload["original_domain_times"].to(self.device)
        self._cumsum_T = payload["cumsum_T"].to(self.device)
        self._total_original_T = payload["total_original_T"].to(self.device)
        self.data = TensorDict(
            {k: v.to(self.device) for k, v in payload["data"].items()},
            batch_size=[self.num_trajectories],
            device=self.device,
        )

        conditioner_datas = list(payload["conditioner_datas"])
        # Cheap integrity guard — a torn or schema-drifted payload should miss.
        if (
            self.data["coeffs_pos"].shape[0] != self.num_trajectories
            or len(self.skill_labels) != self.num_trajectories
            or len(conditioner_datas) != self.num_trajectories
        ):
            raise ValueError("cache payload shape/length mismatch")
        return list(self.skill_labels), conditioner_datas

    def _save_library_cache(
        self,
        cache_path: Path,
        cache_key: str | None,
        skill_labels: list[str],
        conditioner_datas: list[ConditionerData],
    ) -> None:
        """Write the merged-tensor state to ``cache_path`` (best-effort)."""
        payload = {
            "version": _LIBRARY_CACHE_VERSION,
            "key": cache_key,
            "num_trajectories": self.num_trajectories,
            "max_expanded_domains": self.max_expanded_domains,
            "pos_output_names": list(self.pos_output_names),
            "vel_output_names": list(self.vel_output_names),
            "num_pos_outputs": self.num_pos_outputs,
            "num_vel_outputs": self.num_vel_outputs,
            "spline_order": self.spline_order,
            "ref_frames": list(self.ref_frames),
            "trajectory_names": list(self.trajectory_names),
            "skill_labels": list(skill_labels),
            "conditioner_datas": list(conditioner_datas),
            "data": {k: v.detach().cpu() for k, v in self.data.items()},
            "contact_bodies_per_domain": self._contact_bodies_per_domain,
            "ref_frame_names_per_domain": self._ref_frame_names_per_domain,
            "original_domain_times": self._original_domain_times.detach().cpu(),
            "cumsum_T": self._cumsum_T.detach().cpu(),
            "total_original_T": self._total_original_T.detach().cpu(),
        }
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_name(cache_path.name + ".tmp")
            torch.save(payload, str(tmp_path))
            tmp_path.replace(cache_path)
        except Exception as e:  # noqa: BLE001 - cache write is best-effort
            print(
                "[WARN] MultiSkillManager: failed to write tensor cache at "
                f"{cache_path} ({e}); continuing without it."
            )

    def _build_skill_data(
        self,
        skill_labels: list[str],
        conditioner_datas: list[ConditionerData],
    ) -> None:
        """Build per-skill metadata and conditioning tensors."""
        unique_skills = list(dict.fromkeys(skill_labels))
        self.skill_name_to_idx: dict[str, int] = {
            name: i for i, name in enumerate(unique_skills)
        }
        self.skills: list[SkillData] = []

        for skill_name in unique_skills:
            indices = [
                i for i, label in enumerate(skill_labels) if label == skill_name
            ]
            # Each skill must occupy a contiguous block of global trajectory
            # indices: the sort at construction time (by (skill, vel_x)) is
            # what makes the per-skill segmented selection in
            # ``_select_within_skills`` return the *same* global index the
            # legacy full-library masked ``argmin`` did.  Guard the invariant
            # here so any future change to the sort fails loudly instead of
            # silently corrupting trajectory assignment.
            if indices != list(range(indices[0], indices[0] + len(indices))):
                raise RuntimeError(
                    f"Skill '{skill_name}' trajectory indices are not "
                    f"contiguous ({indices[:3]}...{indices[-3:]}). "
                    "MultiSkillManager requires trajectories grouped by skill "
                    "(see the (skill, vel_x) sort in __init__)."
                )
            conds = [conditioner_datas[i] for i in indices]
            has_terrain = any(c.terrain != "flat" for c in conds)

            cond_tensor = torch.tensor(
                [c.to_continuous_tensor() for c in conds],
                dtype=torch.float32,
                device=self.device,
            )

            self.skills.append(
                SkillData(
                    name=skill_name,
                    traj_indices=indices,
                    num_trajectories=len(indices),
                    has_terrain=has_terrain,
                    conditioning_tensor=cond_tensor,
                )
            )

        # Per-skill contiguous-block bounds, indexed by the manager's (traj)
        # skill idx.  Used by ``_select_within_skills`` to slice each skill's
        # rows out of the global conditioning tensor and to map a per-group
        # local ``argmin`` back to a global trajectory index.  Kept as plain
        # Python ints (not tensors) so the per-step selection loop slices with
        # them without a host-device sync.
        self._skill_traj_offset: list[int] = [s.traj_indices[0] for s in self.skills]
        self._skill_traj_count: list[int] = [s.num_trajectories for s in self.skills]

        # Build global conditioning tensor across all trajectories.
        # Ordered by global trajectory index (0 .. T-1).
        all_conds = [cd.to_continuous_tensor() for cd in conditioner_datas]
        self._global_conditioning = torch.tensor(
            all_conds, dtype=torch.float32, device=self.device,
        )  # [T, C]

        # Per-trajectory terrain-dim bounds (width, height, length): a fixed
        # value has lo == hi; a range (a ``{min, max}`` conditioner) has
        # lo < hi (hi may be +inf for an open range).  Kept alongside
        # ``_global_conditioning`` (which stores only the representative scalar)
        # so the terrain-dim filter can do range containment and the offline
        # generator can sample a range dim's built value.
        self._terrain_dim_lo = torch.tensor(
            [cd.terrain_bounds_lo() for cd in conditioner_datas],
            dtype=torch.float32, device=self.device,
        )  # [T, 3]
        self._terrain_dim_hi = torch.tensor(
            [cd.terrain_bounds_hi() for cd in conditioner_datas],
            dtype=torch.float32, device=self.device,
        )  # [T, 3]

        # Terrain mask: True if trajectory has real terrain (positive dims).
        # Flat trajectories use -1 as a sentinel for terrain dimensions.
        terrain_dims = self._global_conditioning[:, 3:]  # [T, 3]
        self._terrain_mask = (terrain_dims > 0).any(dim=1)  # [T]

        # Per-skill terrain flag, indexed by the manager's (traj) skill idx.
        # Used to decide which envs need the terrain-dim filter in
        # ``_terrain_dim_allowed``.
        self._skill_has_terrain = torch.tensor(
            [s.has_terrain for s in self.skills], dtype=torch.bool, device=self.device,
        )  # [num_skills]
        # Python-list mirror for the per-skill selection loop, so deciding
        # whether a skill needs the terrain-dim filter costs no host-device
        # sync per skill.
        self._skill_has_terrain_list: list[bool] = [s.has_terrain for s in self.skills]

    # ------------------------------------------------------------------
    # Output ordering (called by TrajectoryCommand after construction)
    # ------------------------------------------------------------------

    def order_outputs(
        self,
        pos_output_names: list[str],
        vel_output_names: list[str],
    ) -> None:
        """Reorder outputs in the global coefficient tensors.

        This must be called before evaluation if the downstream consumer
        expects a different output ordering than the YAML default.

        Args:
            pos_output_names: Desired position output name order.
            vel_output_names: Desired velocity output name order.
        """
        # Build permutation indices for position outputs
        pos_perm = [self.pos_output_names.index(n) for n in pos_output_names]
        vel_perm = [self.vel_output_names.index(n) for n in vel_output_names]

        pos_perm_t = torch.tensor(pos_perm, dtype=torch.long, device=self.device)
        vel_perm_t = torch.tensor(vel_perm, dtype=torch.long, device=self.device)

        # Reorder coefficient tensors: [T, D, P, K+1] -> index dim 2
        self.data["coeffs_pos"] = self.data["coeffs_pos"][:, :, pos_perm_t, :]
        self.data["coeffs_vel"] = self.data["coeffs_vel"][:, :, vel_perm_t, :]

        self.pos_output_names = list(pos_output_names)
        self.vel_output_names = list(vel_output_names)
        self.num_pos_outputs = len(pos_output_names)
        self.num_vel_outputs = len(vel_output_names)

    # ------------------------------------------------------------------
    # Properties mirroring LibraryManager / TrajectoryManager interface
    # ------------------------------------------------------------------

    @property
    def get_pos_output_names(self) -> list[str]:
        """Get position output names (includes ori_w)."""
        return self.pos_output_names

    @property
    def get_vel_output_names(self) -> list[str]:
        """Get velocity output names (excludes ori_w)."""
        return self.vel_output_names

    def get_reference_frames(self) -> list[str]:
        """Get the reference frame names."""
        return self.ref_frames

    def get_num_pos_outputs(self) -> int:
        """Get number of position outputs (includes ori_w)."""
        return self.num_pos_outputs

    def get_num_vel_outputs(self) -> int:
        """Get number of velocity outputs (excludes ori_w)."""
        return self.num_vel_outputs

    def get_trajectory_type(self) -> TrajectoryType:
        """Get the trajectory type of the first trajectory.

        Note: in a multi-skill setting trajectory types may vary per env.
        This returns the most common type for compatibility.
        """
        types = self.data["traj_type"]
        mode_val = types.mode().values.item()
        for tt, iv in _TRAJ_TYPE_TO_INT.items():
            if iv == mode_val:
                return tt
        return TrajectoryType.HALF_PERIODIC

    # ------------------------------------------------------------------
    # Cache: per-step trajectory assignment
    # ------------------------------------------------------------------

    def invalidate_cache(self) -> None:
        """Invalidate the per-step cache.  Call once at the start of each step."""
        self._cache_valid = False

    def commit_traj_state(self, env_ids: Tensor | None = None) -> None:
        """Snapshot the current cached trajectory assignment as the
        "previous" state used by the next :meth:`_ensure_cache` call to
        detect transitions.

        Decoupled from :meth:`_ensure_cache` so external callers (reset
        events, observation queries, snap helpers, loggers) do not
        silently consume ``_traj_changed`` before the owning command's
        ``_pre_update_phase`` has read it.  The owning command term must
        call this exactly once per step, after gate re-arm has run.

        Args:
            env_ids: Optional subset; ``None`` commits the full batch.
        """
        if self._cached_global_indices is None:
            return
        if env_ids is None:
            self._prev_global_indices = self._cached_global_indices.clone()
        else:
            self._prev_global_indices[env_ids] = self._cached_global_indices[env_ids]

    def set_trajectory_indices(self, global_indices: Tensor) -> None:
        """Directly set the trajectory index for each environment.

        This bypasses the conditioner lookup and is useful for testing or
        when the caller already knows the assignment.

        Args:
            global_indices: Shape ``[N]`` tensor of global trajectory indices.
        """
        self._cached_global_indices = global_indices.to(self.device)
        self._cached_env_ids = None
        self._cache_valid = True

    def _ensure_cache(self, env_ids: Tensor | None = None) -> None:
        """Ensure the trajectory assignment cache is populated.

        Selects the best trajectory per environment across all skills based
        on the commanded velocity (and terrain, when applicable).  Also
        tracks per-env trajectory transitions between steps.

        Args:
            env_ids: Optional subset of environment indices.  ``None`` means
                the full batch.
        """
        if self._cache_valid and self._cached_env_ids is env_ids:
            return

        if self.env is not None and self.conditioner_generator_name:
            conditioner = self._get_conditioner_from_env(env_ids)
            new_indices = self._select_trajectories(conditioner, env_ids=env_ids)

            # --- Trajectory transition detection --------------------------
            # Compute ``_traj_changed`` against the previous snapshot, but
            # do NOT update ``_prev_global_indices`` here.  Many callers
            # (reset events, observation fns, snap helpers) invoke
            # ``_ensure_cache`` between two ``_pre_update_phase`` calls; if
            # they updated the prev array they would silently consume the
            # transition before the owning command saw it.  The owning
            # command term must call :meth:`commit_traj_state` exactly
            # once per step, after gate re-arm has read ``_traj_changed``.
            if self._prev_global_indices is not None:
                if env_ids is not None:
                    self._traj_changed = new_indices != self._prev_global_indices[env_ids]
                else:
                    self._traj_changed = new_indices != self._prev_global_indices
            else:
                # First-call init: prev array must be populated so
                # subsequent comparisons have a baseline.  This is the
                # only place ``_ensure_cache`` writes it.
                self._traj_changed = torch.zeros(
                    new_indices.shape[0], dtype=torch.bool, device=self.device,
                )
                if env_ids is not None:
                    N_total = self.env.num_envs
                    self._prev_global_indices = torch.zeros(
                        N_total, dtype=torch.long, device=self.device,
                    )
                    self._prev_global_indices[env_ids] = new_indices
                else:
                    self._prev_global_indices = new_indices.clone()

            self._cached_global_indices = new_indices
        else:
            raise RuntimeError(
                "Cannot auto-populate cache: no env or conditioner_generator_name. "
                "Call set_trajectory_indices() explicitly."
            )

        self._cached_env_ids = env_ids
        self._cache_valid = True

    def _get_conditioner_from_env(self, env_ids: Tensor | None) -> Tensor:
        """Query the environment command manager for conditioning values.

        Returns:
            Tensor of shape ``[N, C]`` where C is the number of continuous
            conditioning dimensions.
        """
        cond_term = self.env.command_manager.get_term(self.conditioner_generator_name)
        # cond_term.command is typically [num_envs, cmd_dim]
        raw = cond_term.command
        if env_ids is not None:
            raw = raw[env_ids]
        # Pad or truncate to 6 continuous dims expected by ConditionerData
        N = raw.shape[0]
        C = 6  # vel_x, vel_y, vel_yaw, terrain_w, terrain_h, terrain_l
        cond = torch.zeros(N, C, device=self.device)
        cond[:, :min(raw.shape[1], C)] = raw[:, :min(raw.shape[1], C)]
        return cond

    def _lazy_init_skill_filter(self) -> None:
        """One-time validation + skill-id mapping build.

        Resolves the conditioner term, requires it to be a
        :class:`MultiskillVelocityTrackingCommand`-shaped object exposing a
        per-env ``skill_id`` and an ordered ``_skill_list``. Validates that
        the velocity command's skills are a subset of the trajectory
        manager's skills, then caches a gather tensor that maps vel-cmd
        skill indices to trajectory-manager skill indices.  Extra skills
        in the manager (loaded subfolders that the env's terrain doesn't
        advertise) are allowed and remain unused — they live in the
        trajectory pool but never get sampled.

        Deferred to first selection because the conditioner term may not be
        registered on ``env.command_manager`` at manager construction time.
        """
        if self._skill_filter_ready:
            return

        if self._skill_owner is None:
            raise RuntimeError(
                "MultiSkillManager has no skill owner registered. Call "
                "set_skill_owner(...) from the trajectory command's "
                "_post_init before any trajectory selection runs."
            )
        owner = self._skill_owner
        for required in ("skill_id", "_skill_list"):
            if not hasattr(owner, required):
                raise TypeError(
                    f"MultiSkillManager's skill owner "
                    f"({type(owner).__name__}) must expose '{required}'."
                )

        owner_skills: list[str] = list(owner._skill_list)
        traj_skills = set(self.skill_name_to_idx.keys())
        missing_in_traj = sorted(set(owner_skills) - traj_skills)
        if missing_in_traj:
            raise KeyError(
                "Skill-owner declares skills the trajectory pool does not "
                f"contain: {missing_in_traj}. "
                f"Available trajectory skills: {sorted(traj_skills)}."
            )

        self._vel_skill_to_traj_skill = torch.tensor(
            [self.skill_name_to_idx[name] for name in owner_skills],
            dtype=torch.long, device=self.device,
        )
        self._skill_filter_ready = True

    def set_skill_owner(self, owner: Any) -> None:
        """Register the term that owns the per-env active ``skill_id``.

        The owner must expose:

        - ``skill_id`` — ``[num_envs]`` long tensor of the active skill
          per env, indexing into the owner's ``_skill_list``.
        - ``_skill_list`` — ordered list of skill names (= the active
          skill index space).

        Validation is deferred to :meth:`_lazy_init_skill_filter`; calling
        ``set_skill_owner`` only stashes the reference.
        """
        self._skill_owner = owner

    def set_terrain_dim_lookup(self, fn: Any) -> None:
        """Register the terrain's per-point terrain-dimension query.

        ``fn(xy_w) -> (dims, valid)`` where ``dims`` is ``[N, 3]`` holding
        the terrain feature's ``(width, height, length)`` at each world xy —
        same ordering as the trajectory conditioner's terrain columns — and
        ``valid`` is ``[N]`` bool (False where the cell has no terrain
        feature).  A NaN entry in ``dims`` marks that dimension as
        unconstrained (it never participates in matching).  Typically
        ``MetaCompositeTerrainImporter.terrain_dims_at``.  ``None`` (the
        default) disables the terrain-dim selection filter.
        """
        self._terrain_dims_at = fn

    def _terrain_dim_allowed(
        self, anchor_xy: Tensor, env_traj_skill_ids: Tensor
    ) -> Tensor | None:
        """Per-env trajectory mask from the terrain dimensions under the anchor.

        For envs whose skill has terrain trajectories AND whose anchor cell
        has a terrain feature, only trajectories whose conditioner
        ``(terrain_width, terrain_height, terrain_length)`` matches the
        feature's dims pass — per dimension, within ``_TERRAIN_DIM_TOLS``,
        with NaN terrain dims unconstrained.  All other envs pass every
        trajectory (the skill mask alone applies).

        The length tolerance absorbs the legacy stair generator's depth
        rescale (``step_depth = size_x / round(size_x / nominal)``, ≤ ~2 mm
        off nominal) while staying far below the library's 0.04 m grid step.

        Args:
            anchor_xy: ``[N, 2]`` world xy used to resolve each env's
                terrain feature.
            env_traj_skill_ids: ``[N]`` skill indices in the manager's
                (trajectory) skill index space.

        Returns:
            ``[N, T]`` bool mask, or ``None`` when no lookup is registered.
        """
        if self._terrain_dims_at is None:
            return None
        dims, valid = self._terrain_dims_at(anchor_xy)           # [N, 3], [N]
        needs_dims = self._skill_has_terrain[env_traj_skill_ids] & valid  # [N]
        # Range containment against each trajectory's terrain-dim bounds
        # (see ConditionerData): a fixed dim (lo == hi) reduces to
        # |feature - value| <= tol; a range dim matches any in-range feature.
        lo = self._terrain_dim_lo.unsqueeze(0)                       # [1, T, 3]
        hi = self._terrain_dim_hi.unsqueeze(0)                       # [1, T, 3]
        feat = dims.unsqueeze(1)                                     # [N, 1, 3]
        tols = self._terrain_dim_tols                                # [3]
        # NaN terrain dims compare False against any bound, so the explicit
        # isnan() OR is what marks them unconstrained.
        per_dim_ok = (
            ((feat >= lo - tols) & (feat <= hi + tols))
            | dims.isnan().unsqueeze(1)
        )                                                            # [N, T, 3]
        dim_ok = per_dim_ok.all(dim=-1)                              # [N, T]
        return torch.where(needs_dims.unsqueeze(1), dim_ok, torch.ones_like(dim_ok))

    def _apply_terrain_dim_filter(
        self, allowed: Tensor, anchor_xy: Tensor, env_traj_skill_ids: Tensor
    ) -> Tensor:
        """Intersect the skill mask with the terrain-dim mask, with fallback.

        Rows where the intersection is empty (the feature's dimensions
        aren't represented in the library) fall back to the skill-only mask
        — an all-masked row would otherwise make ``argmin`` silently return
        trajectory 0 of the wrong skill.  Warns once when that happens.

        Args:
            allowed: ``[N, T]`` skill-eligibility mask.
            anchor_xy: ``[N, 2]`` world xy terrain anchor per env.
            env_traj_skill_ids: ``[N]`` manager-space skill indices.

        Returns:
            ``[N, T]`` refined mask.
        """
        dim_mask = self._terrain_dim_allowed(anchor_xy, env_traj_skill_ids)
        if dim_mask is None:
            return allowed
        refined = allowed & dim_mask
        has_candidate = refined.any(dim=1)
        if not bool(has_candidate.all()) and not self._warned_dim_fallback:
            print(
                "[WARN] MultiSkillManager: some envs sit on a terrain feature "
                "whose (width, height, length) has no matching trajectory in "
                "the library — falling back to skill-only selection for those "
                "envs. Check that the terrain's feature dimensions (e.g. stair "
                "step_dim_options) were derived from this trajectory library."
            )
            self._warned_dim_fallback = True
        return torch.where(has_candidate.unsqueeze(1), refined, allowed)

    def _selection_anchor_xy(self, env_ids: Tensor | None) -> Tensor:
        """World xy used to resolve each env's terrain feature during selection.

        Uses the owner's ``ref_poses`` (latest in-contact stance reference,
        snapped to the stair tread for terrain-aware skills) — stable across
        per-step pelvis jitter and the same anchor the velocity command uses
        for its cell lookups.

        Args:
            env_ids: Optional env subset.

        Returns:
            ``[N, 2]`` world xy.
        """
        anchor_xy = self._skill_owner.ref_poses[:, :2]
        if env_ids is not None:
            anchor_xy = anchor_xy[env_ids]
        return anchor_xy

    def _select_within_skills(
        self,
        vel_cmd: Tensor,
        env_traj_skill_ids: Tensor,
        anchor_xy: Tensor | None,
    ) -> Tensor:
        """Nearest trajectory per env, evaluated within each env's skill block.

        Equivalent to a full-library ``cdist`` followed by a skill mask and a
        global ``argmin``, but the velocity distance is computed only against
        the env's own skill's contiguous trajectory block — so the per-step
        cost scales with the per-skill trajectory count, not the whole
        library.  Because each skill occupies a contiguous, ``vel_x``-sorted
        global-index range (guarded in :meth:`_build_skill_data`), a per-group
        local ``argmin`` (first-min) plus the block offset reproduces the
        global first-min index exactly.

        The optional terrain-dim filter (the legacy
        :meth:`_terrain_dim_allowed` / :meth:`_apply_terrain_dim_filter` pair)
        is folded in per group, materializing only ``[N_s, T_s, 3]`` instead
        of ``[N, T, 3]``.

        Args:
            vel_cmd: ``[N, 3]`` per-env velocity conditioner (vx, vy, vyaw).
            env_traj_skill_ids: ``[N]`` skill indices in the manager's
                (trajectory) skill index space.
            anchor_xy: ``[N, 2]`` world xy used to resolve each env's terrain
                feature for the terrain-dim filter, or ``None`` to disable the
                filter (no terrain lookup registered / offline usage).

        Returns:
            ``[N]`` global trajectory indices.
        """
        n = vel_cmd.shape[0]
        selected = torch.empty(n, dtype=torch.long, device=self.device)
        if n == 0:
            return selected

        use_terrain = self._terrain_dims_at is not None and anchor_xy is not None
        tols = self._terrain_dim_tols  # [3]

        for s in range(len(self.skills)):
            rows = (env_traj_skill_ids == s).nonzero(as_tuple=True)[0]  # [N_s]
            if rows.numel() == 0:
                continue
            off = self._skill_traj_offset[s]
            cnt = self._skill_traj_count[s]

            vel_block = self._global_conditioning[off:off + cnt, :3]   # [T_s, 3]
            d = torch.cdist(vel_cmd[rows], vel_block)                  # [N_s, T_s]

            # Terrain-dim filter (only for terrain-bearing skills, mirroring
            # the legacy ``needs_dims = _skill_has_terrain & valid`` gate).
            if use_terrain and self._skill_has_terrain_list[s]:
                dims, valid = self._terrain_dims_at(anchor_xy[rows])   # [N_s,3],[N_s]
                # Range containment against each trajectory's terrain-dim
                # bounds (fixed dims have lo == hi, reducing to the old
                # |feature - value| <= tol; range dims accept any in-range
                # feature).
                lo = self._terrain_dim_lo[off:off + cnt].unsqueeze(0)  # [1, T_s, 3]
                hi = self._terrain_dim_hi[off:off + cnt].unsqueeze(0)  # [1, T_s, 3]
                feat = dims.unsqueeze(1)                                # [N_s, 1, 3]
                # NaN terrain dims compare False against any bound, so the
                # explicit isnan() OR is what marks them unconstrained.
                per_dim_ok = (
                    ((feat >= lo - tols) & (feat <= hi + tols))
                    | dims.isnan().unsqueeze(1)
                )                                                       # [N_s, T_s, 3]
                dim_ok = per_dim_ok.all(dim=-1)                         # [N_s, T_s]
                # Envs whose anchor cell has no terrain feature (~valid) match
                # every trajectory in the skill.
                dim_ok = torch.where(
                    valid.unsqueeze(1), dim_ok, torch.ones_like(dim_ok)
                )
                # Empty-match fallback: rows where no trajectory in this skill
                # matches the feature dims fall back to skill-only velocity
                # selection (mirrors :meth:`_apply_terrain_dim_filter`).
                has_candidate = dim_ok.any(dim=1)                      # [N_s]
                if not bool(has_candidate.all()) and not self._warned_dim_fallback:
                    print(
                        "[WARN] MultiSkillManager: some envs sit on a terrain "
                        "feature whose (width, height, length) has no matching "
                        "trajectory in the library — falling back to skill-only "
                        "selection for those envs. Check that the terrain's "
                        "feature dimensions (e.g. stair step_dim_options) were "
                        "derived from this trajectory library."
                    )
                    self._warned_dim_fallback = True
                dim_ok = torch.where(
                    has_candidate.unsqueeze(1), dim_ok, torch.ones_like(dim_ok)
                )
                d = d.masked_fill(~dim_ok, float("inf"))

            selected[rows] = d.argmin(dim=1) + off

        return selected

    def _select_trajectories(
        self,
        conditioner: Tensor,
        env_ids: Tensor | None = None,
    ) -> Tensor:
        """Select the nearest trajectory per env, restricted to the env's skill.

        The active per-env ``skill_id`` comes from the registered skill
        owner (the trajectory cmd) — not the velocity cmd.  The velocity
        cmd's sampled bucket may be in a different skill while
        ``vel_target_b`` is mid-ramp; the owner's active skill follows
        wherever ``vel_target_b`` actually sits.

        Args:
            conditioner: ``(N, C)`` conditioning vectors. Only the first three
                dims (vel_x, vel_y, vel_yaw) participate in the distance;
                the skill mask plus the terrain-dim filter (when a terrain
                lookup is registered — see :meth:`set_terrain_dim_lookup`)
                replace the legacy terrain-dim distance.
            env_ids: Optional env subset matching ``conditioner``'s rows. When
                provided, the owner's ``skill_id`` is sliced the same way so
                row counts stay aligned (callers like ``reset_on_reference``
                pass an env subset).

        Returns:
            ``(N,)`` global trajectory indices.
        """
        self._lazy_init_skill_filter()

        owner_skill_ids: Tensor = self._skill_owner.skill_id  # (num_envs,)
        if env_ids is not None:
            owner_skill_ids = owner_skill_ids[env_ids]

        # Sanity guard: catches any future caller that slices the conditioner
        # without also passing the matching ``env_ids`` (or vice versa).
        if conditioner.shape[0] != owner_skill_ids.shape[0]:
            raise RuntimeError(
                f"conditioner ({conditioner.shape[0]}) and owner.skill_id "
                f"({owner_skill_ids.shape[0]}) must have matching env count."
            )

        env_traj_skill_ids = self._vel_skill_to_traj_skill[owner_skill_ids]  # (N,)

        # Velocity-nearest pick within each env's own skill block.  The skill
        # restriction and (when registered) the terrain-dim filter are folded
        # into the per-group computation — see :meth:`_select_within_skills`.
        anchor_xy = (
            self._selection_anchor_xy(env_ids)
            if self._terrain_dims_at is not None
            else None
        )
        selected = self._select_within_skills(
            conditioner[:, :3], env_traj_skill_ids, anchor_xy,
        )
        return selected

    def _select_trajectories_reference(
        self,
        conditioner: Tensor,
        env_ids: Tensor | None = None,
    ) -> Tensor:
        """Reference (pre-optimization) full-library selection — TEST ONLY.

        Computes the velocity-nearest, skill-masked, terrain-filtered
        selection over the *entire* library with a single ``[N, T]`` ``cdist``
        (and an ``[N, T, 3]`` terrain mask).  Retained as the permanent
        equivalence oracle for :meth:`_select_within_skills` — the
        ``TestSelectionEquivalence`` suite asserts the production path
        (:meth:`_select_trajectories`) returns byte-identical global indices,
        guarding the load-bearing ``_traj_changed`` contract against future
        drift.  Never called on the training hot path.
        """
        self._lazy_init_skill_filter()

        owner_skill_ids: Tensor = self._skill_owner.skill_id
        if env_ids is not None:
            owner_skill_ids = owner_skill_ids[env_ids]
        if conditioner.shape[0] != owner_skill_ids.shape[0]:
            raise RuntimeError(
                f"conditioner ({conditioner.shape[0]}) and owner.skill_id "
                f"({owner_skill_ids.shape[0]}) must have matching env count."
            )
        env_traj_skill_ids = self._vel_skill_to_traj_skill[owner_skill_ids]

        vel_cmd = conditioner[:, :3]
        vel_global = self._global_conditioning[:, :3]
        vel_dists = torch.cdist(vel_cmd, vel_global)

        allowed = self._traj_skill_idx.unsqueeze(0) == env_traj_skill_ids.unsqueeze(1)
        if self._terrain_dims_at is not None:
            allowed = self._apply_terrain_dim_filter(
                allowed, self._selection_anchor_xy(env_ids), env_traj_skill_ids,
            )
        vel_dists = vel_dists.masked_fill(~allowed, float("inf"))
        selected = vel_dists.argmin(dim=1)
        return selected

    def nearest_traj_in_skill(
        self, owner_skill_ids: Tensor, env_ids: Tensor,
        anchor_xy: Tensor | None = None,
    ) -> Tensor:
        """Nearest trajectory (by velocity) within an *arbitrary* skill.

        Unlike :meth:`_select_trajectories` (which picks within each env's
        *current* owner skill), this resolves the nearest trajectory in a
        caller-supplied skill.  Used by the terrain-approach planner to peek
        at the stair trajectory an env *would* commit to, so it can predict
        that trajectory's snapped stance reference.

        Args:
            owner_skill_ids: ``[K]`` skill indices in the owner's
                (terrain ``skill_list``) index space.
            env_ids: ``[K]`` env indices, aligned to ``owner_skill_ids``,
                used to read each env's velocity conditioner.
            anchor_xy: Optional ``[K, 2]`` world xy used to resolve the
                terrain feature for the terrain-dim filter.  Defaults to the
                owner's stance reference; the approach planner passes the
                just-landed stance foot instead (``ref_poses`` is stale at
                gate-fire time).

        Returns:
            ``[K]`` global trajectory indices.
        """
        self._lazy_init_skill_filter()
        conditioner = self._get_conditioner_from_env(env_ids)
        traj_skill_ids = self._vel_skill_to_traj_skill[owner_skill_ids]  # (K,)
        if self._terrain_dims_at is not None and anchor_xy is None:
            anchor_xy = self._selection_anchor_xy(env_ids)
        # Shares the per-skill segmented selection with
        # ``_select_trajectories``, but for a caller-supplied skill.
        return self._select_within_skills(
            conditioner[:, :3],
            traj_skill_ids,
            anchor_xy if self._terrain_dims_at is not None else None,
        )

    # ------------------------------------------------------------------
    # Core batched helpers
    # ------------------------------------------------------------------

    def _get_global_indices(self, env_ids: Tensor | None = None) -> Tensor:
        """Return the cached global trajectory indices, ensuring cache is valid."""
        self._ensure_cache(env_ids)
        return self._cached_global_indices

    def get_current_trajectory_indices(self, env_ids: Tensor | None = None) -> Tensor:
        """Public wrapper around the trajectory assignment cache.

        Resolves the current trajectory index per environment without
        computing any outputs.  Callers can use this to detect skill/
        trajectory changes between steps.

        Args:
            env_ids: Optional environment index subset.  ``None`` resolves
                the full batch.

        Returns:
            ``[N]`` global trajectory indices for the selected envs.
        """
        return self._get_global_indices(env_ids)

    def _get_domain_indices(self, phase: Tensor, traj_idx: Tensor) -> Tensor:
        """Batched domain lookup via searchsorted on per-trajectory boundaries.

        Args:
            phase: ``[N]`` phase per environment.
            traj_idx: ``[N]`` global trajectory index per environment.

        Returns:
            ``[N]`` domain indices (into the expanded domain dimension).
        """
        total = self.data["total_time"][traj_idx]  # [N]
        t = phase * total  # [N]

        # Gather per-trajectory boundaries: [N, D_max+1]
        boundaries = self.data["domain_boundaries"][traj_idx]  # [N, D_max+1]

        # searchsorted along dim=1
        domain_idx = torch.searchsorted(boundaries, t.unsqueeze(1), right=False).squeeze(1) - 1

        # Clamp to valid range per trajectory
        max_dom = self.data["expanded_domains"][traj_idx] - 1
        domain_idx = torch.clamp(domain_idx, min=torch.zeros_like(max_dom), max=max_dom)

        return domain_idx

    def _compute_normalized_tau(
        self, phase: Tensor, traj_idx: Tensor, domain_idx: Tensor
    ) -> Tensor:
        """Compute tau ∈ [0,1] within the current domain for each environment.

        Args:
            phase: ``[N]`` phase in [0, 1].
            traj_idx: ``[N]`` global trajectory indices.
            domain_idx: ``[N]`` expanded domain indices.

        Returns:
            ``[N]`` normalized tau values.
        """

        # For half-periodic: fold the second half onto the first
        # ([0.5, 1) → [0, 0.5)) and scale to [0, 1).
        is_half = self.data["traj_type"][traj_idx] == _HALF_PERIODIC_INT
        phase = torch.where(is_half & (phase >= 0.5), phase - 0.5, phase)
        phase = torch.where(is_half, phase * 2.0, phase)

        # Domain index within original (non-reflected) domains
        num_orig = self.data["num_original_domains"][traj_idx]  # [N]
        dom_in_half = domain_idx % num_orig

        # Relative start of domain within the half-period
        # cumsum_T[traj, dom] / total_original_T[traj]
        rel_prev = self._cumsum_T[traj_idx, dom_in_half] / self._total_original_T[traj_idx]

        # Duration of this domain
        T_dom = self.data["domain_times"][traj_idx, domain_idx]

        # Normalize
        tau_norm = (phase - rel_prev) * self._total_original_T[traj_idx] / T_dom
        return torch.clamp(tau_norm, 0.0, 1.0)

    def _compute_bezier_batched(
        self, phase: Tensor, ctrl_pts: Tensor, T_dom: Tensor, derivative: bool
    ) -> Tensor:
        """Batched Bernstein polynomial evaluation.

        Args:
            phase: ``[N]`` phase in [0, 1].
            ctrl_pts: ``[N, num_outputs, degree+1]`` control points.
            T_dom: ``[N]`` domain durations (for velocity scaling).
            derivative: If True, compute first derivative of the Bezier curve.

        Returns:
            ``[N, num_outputs]`` interpolated values.
        """
        degree = ctrl_pts.shape[-1] - 1

        if not derivative:
            coefs = self._binomial_coeffs[degree]  # [degree+1]
            i_vec = torch.arange(degree + 1, device=ctrl_pts.device)
            tau_pow = phase.unsqueeze(1) ** i_vec  # [N, degree+1]
            one_minus_pow = (1 - phase).unsqueeze(1) ** (degree - i_vec)
            weights = coefs * tau_pow * one_minus_pow  # [N, degree+1]
            return torch.einsum("nd,nod->no", weights, ctrl_pts)
        else:
            coefs = self._binomial_coeffs[degree - 1]  # [degree]
            i_vec = torch.arange(degree, device=ctrl_pts.device)
            tau_pow = phase.unsqueeze(1) ** i_vec
            one_minus_pow = (1 - phase).unsqueeze(1) ** (degree - 1 - i_vec)
            weights = degree * coefs * tau_pow * one_minus_pow
            cp_diff = ctrl_pts[:, :, 1:] - ctrl_pts[:, :, :-1]
            result = torch.einsum("nd,nod->no", weights, cp_diff)
            return result / T_dom.unsqueeze(1)

    # ------------------------------------------------------------------
    # ManagerBase interface — fully batched
    # ------------------------------------------------------------------

    def get_output(
        self, phase: Tensor, env_ids: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Compute position and velocity outputs for all environments.

        Args:
            phase: ``[N]`` phase in [0, 1] per environment.
            env_ids: Optional environment index subset.

        Returns:
            ``(pos_outputs, vel_outputs)`` each of shape ``[N, P]`` / ``[N, V]``.
        """
        traj_idx = self._get_global_indices(env_ids)
        domain_idx = self._get_domain_indices(phase, traj_idx)
        tau = self._compute_normalized_tau(phase, traj_idx, domain_idx)

        # Gather coefficients: [N, P/V, K+1]
        env_coeffs_pos = self.data["coeffs_pos"][traj_idx, domain_idx]
        env_coeffs_vel = self.data["coeffs_vel"][traj_idx, domain_idx]
        T_dom = self.data["domain_times"][traj_idx, domain_idx]

        pos_out = self._compute_bezier_batched(tau, env_coeffs_pos, T_dom, derivative=False)
        vel_out = self._compute_bezier_batched(tau, env_coeffs_vel, T_dom, derivative=False)

        return pos_out, vel_out

    def get_current_domains(
        self, phase: Tensor, env_ids: Tensor | None = None
    ) -> Tensor:
        """Return the domain index for each environment.

        Args:
            phase: ``[N]`` phase in [0, 1] per environment.
            env_ids: Optional environment index subset.

        Returns:
            ``[N]`` domain indices.
        """
        traj_idx = self._get_global_indices(env_ids)
        return self._get_domain_indices(phase, traj_idx)

    def get_domain_times(
        self, phase: Tensor, env_ids: Tensor | None = None
    ) -> Tensor:
        """Get the duration of each environment's current domain.

        Args:
            phase: ``[N]`` phase in [0, 1] per environment.
            env_ids: Optional environment index subset.

        Returns:
            ``[N]`` domain durations.
        """
        traj_idx = self._get_global_indices(env_ids)
        domain_idx = self._get_domain_indices(phase, traj_idx)
        return self.data["domain_times"][traj_idx, domain_idx]

    def get_num_domains(self, env_ids: Tensor | None = None) -> Tensor:
        """Get the expanded domain count for each environment.

        Args:
            env_ids: Optional environment index subset.

        Returns:
            ``[N]`` expanded domain counts.
        """
        traj_idx = self._get_global_indices(env_ids)
        return self.data["expanded_domains"][traj_idx]

    def get_total_time(self) -> Tensor:
        """Get the total time of the first trajectory (compatibility)."""
        return self.data["total_time"][0]

    def get_ref_frames_in_use(
        self, phase: Tensor, ref_frames: list[str], env_ids: Tensor | None = None
    ) -> Tensor:
        """Determine the active reference frame per environment.

        Args:
            phase: ``[N]`` phase in [0, 1] per environment.
            ref_frames: List of reference frame names.
            env_ids: Optional environment index subset.

        Returns:
            ``[N]`` indices into ``ref_frames``.
        """
        # Lazily build lookup table on first call
        if not hasattr(self, "_ref_frame_map") or self._ref_frame_key != tuple(ref_frames):
            self._precompute_ref_frame_map(ref_frames)

        traj_idx = self._get_global_indices(env_ids)
        domain_idx = self._get_domain_indices(phase, traj_idx)
        return self._ref_frame_map[traj_idx, domain_idx]

    def _precompute_ref_frame_map(self, ref_frames: list[str]) -> None:
        """Build a [T, D_max] lookup table mapping (traj, domain) → ref frame index."""
        T = self.num_trajectories
        D = self.max_expanded_domains
        table = torch.zeros(T, D, dtype=torch.long, device=self.device)

        # TODO: Double check this comment - is this right?
        # We need domain→frame mappings.  Since we discarded managers, we
        # rebuild from the stored data.  For now, use a simple approach:
        # all domains within a trajectory share the same set of reference frames
        # in alternating order matching the original domain data.
        #
        # This is populated during _build_global_tensors if we store it.
        # For robustness, store a ref_frame_per_domain tensor at load time.
        if hasattr(self, "_ref_frame_domain_map"):
            table = self._ref_frame_domain_map
        else:
            # Fallback: all zeros (first ref frame).  This will be overridden
            # once _build_ref_frame_map is called during loading.
            pass

        self._ref_frame_map = table
        self._ref_frame_key = tuple(ref_frames)

    def get_contact_state(
        self, phase: Tensor, contact_frames: list[str], env_ids: Tensor | None = None
    ) -> Tensor:
        """Return the contact state per environment and contact frame.

        Args:
            phase: ``[N]`` phase in  [0, 1] per environment.
            contact_frames: List of contact frame names.
            env_ids: Optional environment index subset.

        Returns:
            ``[N, num_contacts]`` binary contact states.
        """
        # Build the lookup once (and rebuild only if the contact-frame list
        # changes).  The table is derived purely from the static trajectory
        # contact layout + ``contact_frames``, so it never changes within a
        # run.  NOTE: the guard keys off ``_contact_table_key`` (set by
        # ``_precompute_contact_table``); a prior version checked
        # ``hasattr(self, "_contact_table")`` — an attribute that is never
        # assigned — so the O(num_trajectories) Python rebuild ran on *every*
        # call (twice per step), dominating per-step time as the library grew.
        if getattr(self, "_contact_table_key", None) != tuple(contact_frames):
            self._precompute_contact_table(contact_frames)

        traj_idx = self._get_global_indices(env_ids)
        domain_idx = self._get_domain_indices(phase, traj_idx)

        # For half-periodic trajectories, use reflected table in second half
        is_half = self.data["traj_type"][traj_idx] == _HALF_PERIODIC_INT
        in_second_half = is_half & (phase >= 0.5)

        result = self._contact_table_first[traj_idx, domain_idx]
        reflected = self._contact_table_second[traj_idx, domain_idx]
        return torch.where(in_second_half.unsqueeze(1), reflected, result)

    def _precompute_contact_table(self, contact_frames: list[str]) -> None:
        """Build contact lookup tables of shape [T, D_max, num_contacts]."""
        # This requires domain-level contact body information which we need
        # to store during loading.  Use the stored _contact_bodies_per_domain.
        T = self.num_trajectories
        D = self.max_expanded_domains
        C = len(contact_frames)

        table_first = torch.zeros(T, D, C, device=self.device)
        table_second = torch.zeros(T, D, C, device=self.device)

        if hasattr(self, "_contact_bodies_per_domain"):
            for ti in range(T):
                ed = self.data["expanded_domains"][ti].item()
                nd = self.data["num_original_domains"][ti].item()
                is_half = self.data["traj_type"][ti].item() == _HALF_PERIODIC_INT

                for di in range(ed):
                    bodies = self._contact_bodies_per_domain[ti][di % nd]
                    reflected_bodies = [swap_left_right(b) for b in bodies]
                    for ci, frame in enumerate(contact_frames):
                        if frame in bodies:
                            table_first[ti, di, ci] = 1.0
                        if is_half and frame in reflected_bodies:
                            table_second[ti, di, ci] = 1.0
                        elif not is_half and frame in bodies:
                            table_second[ti, di, ci] = 1.0

        self._contact_table_first = table_first
        self._contact_table_second = table_second
        self._contact_table_key = tuple(contact_frames)

    # ------------------------------------------------------------------
    # Tensor construction
    # ------------------------------------------------------------------

    def _build_global_tensors(
        self,
        managers: list[TrajectoryManager],
        skill_labels: list[str],
        conditioner_datas: list[ConditionerData],
    ) -> None:
        """Extract data from TrajectoryManagers into padded global tensors.

        This method also stores per-domain metadata (contact bodies, ref frames)
        needed for lazy table construction.
        """
        T = self.num_trajectories
        P = self.num_pos_outputs
        V = self.num_vel_outputs
        K = self.spline_order

        # Determine max expanded domain count
        expanded_counts = [mgr.expanded_num_domains for mgr in managers]
        D_max = max(expanded_counts)
        self.max_expanded_domains = D_max

        # Allocate tensors
        coeffs_pos = torch.zeros(T, D_max, P, K + 1, device=self.device)
        coeffs_vel = torch.zeros(T, D_max, V, K + 1, device=self.device)
        domain_times = torch.zeros(T, D_max, device=self.device)
        domain_boundaries = torch.zeros(T, D_max + 1, device=self.device)
        total_time = torch.zeros(T, device=self.device)
        expanded_domains = torch.zeros(T, dtype=torch.long, device=self.device)
        traj_type = torch.zeros(T, dtype=torch.long, device=self.device)
        skill_idx_tensor = torch.zeros(T, dtype=torch.long, device=self.device)
        num_original_domains = torch.zeros(T, dtype=torch.long, device=self.device)
        ref_frame_offset = torch.zeros(T, D_max, 3, device=self.device)
        origin_relative_to_stair_center = torch.zeros(T, 3, device=self.device)

        unique_skills = list(dict.fromkeys(skill_labels))
        skill_name_to_idx = {name: i for i, name in enumerate(unique_skills)}

        # Per-domain metadata (stored as Python lists for lazy table building)
        contact_bodies_per_domain: list[list[list[str]]] = []
        ref_frame_per_domain: list[list[str]] = []

        original_domain_times = torch.zeros(T, D_max, device=self.device)

        for i, mgr in enumerate(managers):
            ed = mgr.expanded_num_domains
            nd = mgr.num_domains
            expanded_domains[i] = ed
            num_original_domains[i] = nd
            traj_type[i] = _TRAJ_TYPE_TO_INT[mgr.traj_data.trajectory_type]
            skill_idx_tensor[i] = skill_name_to_idx[skill_labels[i]]

            # Coefficients
            coeffs_pos[i, :ed, :, :] = mgr._all_coeffs_pos
            coeffs_vel[i, :ed, :, :] = mgr._all_coeffs_vel

            # Domain times
            domain_times[i, :ed] = mgr._T_all
            original_domain_times[i, :nd] = mgr.T

            # Domain boundaries
            db = mgr.domain_boundaries
            domain_boundaries[i, : ed + 1] = db
            if ed + 1 < D_max + 1:
                domain_boundaries[i, ed + 1 :] = db[-1]

            # Total time
            if mgr.traj_data.trajectory_type == TrajectoryType.HALF_PERIODIC:
                total_time[i] = mgr.traj_data.total_time * 2
            else:
                total_time[i] = mgr.traj_data.total_time

            # Contact bodies per domain (original domains only)
            bodies_list: list[list[str]] = []
            frames_list: list[str] = []
            for di, domain_name in enumerate(mgr.traj_data.domain_order):
                dd = mgr.traj_data.domain_data[domain_name]
                bodies_list.append(list(dd.contact_bodies))
                frames_list.append(dd.bezier_frame)
                ref_frame_offset[i, di, :] = dd.ref_frame_offset
                if mgr.traj_data.trajectory_type == TrajectoryType.HALF_PERIODIC:
                    # Reflected half reuses the same offset.
                    ref_frame_offset[i, di + mgr.num_domains, :] = dd.ref_frame_offset
            # Fail loudly if no domain declares any contact bodies. The most
            # common cause is a YAML field-name typo (e.g. ``contact_frames``
            # instead of the expected ``contact_bodies``), or values that
            # don't match the scene's URDF link names — both of which
            # silently produce empty lists here, an all-zero contact table,
            # and a frozen ``ref_poses`` that's almost impossible to debug
            # from symptoms alone.
            if not any(bodies_list):
                traj_name = getattr(mgr.traj_data, "name", f"<index {i}>")
                raise ValueError(
                    f"Trajectory '{traj_name}' has no contact bodies declared "
                    f"in any of its {len(bodies_list)} domain(s). Check the "
                    f"trajectory YAML: per-domain field must be named "
                    f"'contact_bodies' and values must match scene URDF link "
                    f"names (got {bodies_list!r})."
                )

            contact_bodies_per_domain.append(bodies_list)
            ref_frame_per_domain.append(frames_list)

            # Per-trajectory stair-origin offset (top-level YAML field).
            origin_relative_to_stair_center[i] = mgr.traj_data.origin_relative_to_stair_center

        self.data = TensorDict(
            {
                "coeffs_pos": coeffs_pos,
                "coeffs_vel": coeffs_vel,
                "domain_times": domain_times,
                "domain_boundaries": domain_boundaries,
                "total_time": total_time,
                "expanded_domains": expanded_domains,
                "num_original_domains": num_original_domains,
                "traj_type": traj_type,
                "skill_idx": skill_idx_tensor,
                "ref_frame_offset": ref_frame_offset,
                "origin_relative_to_stair_center": origin_relative_to_stair_center,
            },
            batch_size=[T],
            device=self.device,
        )

        # Store metadata for lazy table building
        self._contact_bodies_per_domain = contact_bodies_per_domain
        self._ref_frame_names_per_domain = ref_frame_per_domain

        # Tau normalisation helpers
        self._original_domain_times = original_domain_times
        cumsum = torch.zeros(T, D_max, device=self.device)
        cumsum[:, 1:] = torch.cumsum(original_domain_times[:, :-1], dim=1)
        self._cumsum_T = cumsum
        self._total_original_T = original_domain_times.sum(dim=1)

    # ------------------------------------------------------------------
    # Placeholder stubs for future features
    # ------------------------------------------------------------------

    def transform_terrain_relative(
        self, positions: Tensor, env_ids: Tensor | None = None
    ) -> Tensor:
        """Transform positions to be relative to terrain geometry.

        Currently a no-op stub.  Future implementation will use
        ``env.scene.terrain`` to position terrain-dependent trajectories
        in the correct global frame.

        Args:
            positions: ``[N, ...]`` position tensor.
            env_ids: Optional environment indices.

        Returns:
            The input positions unchanged.
        """
        return positions

    # ------------------------------------------------------------------
    # Contact-gate tables
    # ------------------------------------------------------------------

    def _build_gate_tables(self) -> None:
        """Pre-compute contact-gate phi values and per-gate body name lists.

        Gate points per trajectory type:

        - Half-periodic: 2 gates (phi=0.5 entering reflected second half;
          phi=1.0 wrapping back to first half).
        - Full-periodic: 2 gates — the mid-cycle strike (gate 0: the swing
          foot's touchdown that begins the second stride, at a phi *derived
          from the contact schedule*, not hardcoded) and the cycle wrap
          (gate 1: phi=1.0, the domain-0 stance foot re-striking).  A
          full-periodic gait with no distinct mid-cycle strike (one stance
          set for the whole cycle) falls back to a single gate at phi=1.0.
        - Episodic: 1 gate at phi=1.0 (end of trajectory).
        - Perpetual: 0 gates.

        The expected-contact body names at each gate are taken from the
        first domain of the next period — for half-periodic phi=0.5 this is
        the sagittally reflected first domain.  The boolean mask in the
        command-term's contact-body ordering is materialised lazily via
        :meth:`set_gate_contact_layout`.
        """
        T = self.num_trajectories
        MAX_GATES = 2

        gate_phi = torch.zeros(T, MAX_GATES, device=self.device)
        gate_active = torch.zeros(T, MAX_GATES, dtype=torch.bool, device=self.device)
        # Phi-distance from the previous gate (wrapping for periodic types)
        # to this gate.  Used to scale the contact-gate early window in
        # the command term so a configured ``contact_gate_window_frac``
        # is interpreted as "fraction of the local domain", not raw phi.
        gate_pre_size = torch.ones(T, MAX_GATES, device=self.device)
        num_gates = torch.zeros(T, dtype=torch.long, device=self.device)
        gate_body_names: list[list[list[str]]] = [
            [[] for _ in range(MAX_GATES)] for _ in range(T)
        ]

        for ti in range(T):
            tt = self.data["traj_type"][ti].item()
            first_domain_bodies = list(self._contact_bodies_per_domain[ti][0])

            if tt == _HALF_PERIODIC_INT:
                gate_phi[ti, 0] = 0.5
                gate_active[ti, 0] = True
                gate_body_names[ti][0] = [swap_left_right(b) for b in first_domain_bodies]
                gate_pre_size[ti, 0] = 0.5

                gate_phi[ti, 1] = 1.0
                gate_active[ti, 1] = True
                gate_body_names[ti][1] = list(first_domain_bodies)
                gate_pre_size[ti, 1] = 0.5
                num_gates[ti] = 2
            elif tt == _FULL_PERIODIC_INT:
                # A full-periodic gait is TWO strides with TWO foot-strike
                # contacts per cycle.  Gate 0 is the mid-cycle strike (the swing
                # foot's touchdown that starts the second stride); gate 1 is the
                # cycle-wrap strike (phi=1.0).  Gating both restores per-stride
                # contact resync (parity with half-periodic).  The mid-cycle phi
                # is DERIVED from the contact schedule -- the first domain after
                # domain 0 whose contact set is non-empty and differs from the
                # domain-0 stance (the opposite foot coming into contact; robust
                # to intervening flight phases) -- not hardcoded to 0.5.
                nd = int(self.data["num_original_domains"][ti].item())
                stance0 = set(first_domain_bodies)
                mid_dom = next(
                    (
                        di for di in range(1, nd)
                        if self._contact_bodies_per_domain[ti][di]
                        and set(self._contact_bodies_per_domain[ti][di]) != stance0
                    ),
                    None,
                )
                if mid_dom is not None:
                    total_t = float(self.data["total_time"][ti].item())
                    phi_mid = float(self.data["domain_boundaries"][ti, mid_dom].item()) / total_t
                    gate_phi[ti, 0] = phi_mid
                    gate_active[ti, 0] = True
                    gate_body_names[ti][0] = list(self._contact_bodies_per_domain[ti][mid_dom])
                    gate_pre_size[ti, 0] = phi_mid

                    gate_phi[ti, 1] = 1.0
                    gate_active[ti, 1] = True
                    gate_body_names[ti][1] = list(first_domain_bodies)
                    gate_pre_size[ti, 1] = 1.0 - phi_mid
                    num_gates[ti] = 2
                else:
                    # No distinct mid-cycle strike (single stance set): fall back
                    # to the single cycle-wrap gate.
                    gate_phi[ti, 0] = 1.0
                    gate_active[ti, 0] = True
                    gate_body_names[ti][0] = list(first_domain_bodies)
                    gate_pre_size[ti, 0] = 1.0
                    num_gates[ti] = 1
            elif tt == _EPISODIC_INT:
                gate_phi[ti, 0] = 1.0
                gate_active[ti, 0] = True
                gate_body_names[ti][0] = list(first_domain_bodies)
                gate_pre_size[ti, 0] = 1.0
                num_gates[ti] = 1
            # Perpetual: no gates.

        self._gate_phi_table = gate_phi
        self._gate_active_table = gate_active
        self._gate_pre_size_table = gate_pre_size
        self._num_gates_per_traj = num_gates
        self._gate_body_names_per_gate = gate_body_names
        self._max_gates = MAX_GATES
        self._gate_contact_mask: Tensor | None = None

    def set_gate_contact_layout(self, contact_bodies: list[str]) -> None:
        """Materialise the [T, MAX_GATES, B] expected-contact mask in the
        ordering of *contact_bodies*.

        Args:
            contact_bodies: Ordered list of contact body names matching the
                command term's runtime contact tensor layout.
        """
        T = self.num_trajectories
        MAX_GATES = self._max_gates
        B = len(contact_bodies)

        mask = torch.zeros(T, MAX_GATES, B, dtype=torch.bool, device=self.device)
        body_to_idx = {n: i for i, n in enumerate(contact_bodies)}

        for ti in range(T):
            for gi in range(MAX_GATES):
                if not self._gate_active_table[ti, gi].item():
                    continue
                for body in self._gate_body_names_per_gate[ti][gi]:
                    if body in body_to_idx:
                        mask[ti, gi, body_to_idx[body]] = True

        self._gate_contact_mask = mask

    # ------------------------------------------------------------------
    # Per-env phase state and snap operations
    # ------------------------------------------------------------------

    def _ensure_phase_state(self, num_envs: int) -> None:
        """Lazily allocate ``phase``, ``next_gate_idx``, ``gate_rel_phi``,
        and ``gate_rel_phi_pre_snap``."""
        if (
            self.phase is not None
            and self.phase.shape[0] == num_envs
        ):
            return
        self.phase = torch.zeros(num_envs, device=self.device)
        self.next_gate_idx = -torch.ones(num_envs, dtype=torch.long, device=self.device)
        self.gate_rel_phi = torch.zeros(num_envs, device=self.device)
        self.gate_rel_phi_pre_snap = torch.zeros(num_envs, device=self.device)
        # Per-env wrap-aware phi delta written by :meth:`update_phase` for
        # the envs that actually advanced this tick; consumers (cross-fade
        # state, future analytics) read this instead of diffing
        # ``phase`` themselves.  Envs that didn't advance get 0.
        self.last_phase_delta = torch.zeros(num_envs, device=self.device)

    def _refresh_gate_rel_phi(self, env_ids: Tensor) -> None:
        """Recompute ``gate_rel_phi`` from current phase + armed gate.

        Sets ``gate_rel_phi[i] = phase[i] - gate_phi[next_gate_idx[i]]`` for
        envs with an active gate, or ``0`` for envs whose gate is ``-1``
        (perpetual / post-episodic).  Called after any operation that
        changes which gate is armed (reseed, advance, snap-with-advance).

        Args:
            env_ids: Env indices to refresh.
        """
        if env_ids.numel() == 0:
            return
        traj_idx = self._get_global_indices()[env_ids]
        gate_idx = self.next_gate_idx[env_ids]
        active = gate_idx >= 0
        safe_idx = torch.clamp(gate_idx, min=0)
        gate_phi = self._gate_phi_table[traj_idx, safe_idx]
        rel = self.phase[env_ids] - gate_phi
        self.gate_rel_phi[env_ids] = torch.where(active, rel, torch.zeros_like(rel))

    _EPS_PHI = 0.001
    """Small phi-space nudge used by the snap operations.

    Snap operations land phase at ``gate_phi ± _EPS_PHI`` so that the
    post-snap value falls strictly inside the intended (new vs. previous)
    domain rather than on the boundary, where ``searchsorted(...,
    right=False)`` would assign it to the previous domain.  ``0.001`` is
    small enough not to perturb training and large enough to clear
    float-precision and the boundary tie.
    """

    def _reseed_gate_for_envs(self, env_ids: Tensor) -> None:
        """Re-arm ``next_gate_idx`` for the given envs based on their current phase.

        Picks the smallest active gate whose ``gate_phi >= phase``;  if no
        upcoming gate this period, arms gate 0 of the next period.  If the
        trajectory has no gates at all, sets ``-1``.
        """
        if env_ids.numel() == 0:
            return
        traj_idx = self._get_global_indices()[env_ids]
        phi = self.phase[env_ids]
        gates = self._gate_phi_table[traj_idx]                # [n, MAX_GATES]
        gate_active = self._gate_active_table[traj_idx]       # [n, MAX_GATES]
        upcoming = gate_active & (gates >= phi.unsqueeze(1))
        has_upcoming = upcoming.any(dim=1)
        first_upcoming_idx = upcoming.to(torch.long).argmax(dim=1)
        init_idx = torch.where(
            has_upcoming, first_upcoming_idx, torch.zeros_like(first_upcoming_idx)
        )
        num_gates = self._num_gates_per_traj[traj_idx]
        init_idx = torch.where(num_gates > 0, init_idx, -torch.ones_like(init_idx))
        self.next_gate_idx[env_ids] = init_idx
        self._refresh_gate_rel_phi(env_ids)

    def _advance_gate_for_envs(self, env_ids: Tensor) -> None:
        """Advance ``next_gate_idx`` after a fire / expiry.

        Wraps to gate 0 for periodic; sets to -1 once the last gate of an
        episodic trajectory has been processed.
        """
        if env_ids.numel() == 0:
            return
        traj_idx = self._get_global_indices()[env_ids]
        cur_idx = self.next_gate_idx[env_ids]
        new_idx = cur_idx + 1
        num_gates = self._num_gates_per_traj[traj_idx]
        traj_type = self.data["traj_type"][traj_idx]

        last = new_idx >= num_gates
        is_episodic = traj_type == _EPISODIC_INT

        new_idx = torch.where(
            last & ~is_episodic, torch.zeros_like(new_idx), new_idx
        )
        new_idx = torch.where(
            last & is_episodic, -torch.ones_like(new_idx), new_idx
        )
        self.next_gate_idx[env_ids] = new_idx
        self._refresh_gate_rel_phi(env_ids)

    def advance_phase_for_traj(
        self,
        phase: Tensor,
        traj_idx: Tensor,
        step_dt: float,
    ) -> tuple[Tensor, Tensor]:
        """Advance per-row phase by ``step_dt`` using each row's trajectory rule.

        Pure (side-effect-free) version of the per-trajectory-type
        advancement logic in :meth:`update_phase`.  Called by
        ``update_phase`` for ``self.phase`` and by the cross-fade
        machinery for ``CrossfadeState.old_phase`` so the old
        trajectory keeps its own clock running during a transition
        whose target is perpetual (otherwise the blend can't progress).

        Half/full periodic wrap mod 1.0.  Episodic clamps to ``[0, 1]``.
        Perpetual snaps to 0 (unchanged).

        Args:
            phase: ``(K,)`` current phase per row.
            traj_idx: ``(K,)`` trajectory index per row.
            step_dt: sim step duration (seconds).

        Returns:
            ``(new_phase, phi_movement)`` — both ``(K,)``.  ``phi_movement``
            is the wrap-aware signed advancement (positive for periodic
            including across a wrap, can be negative for the single
            perpetual-snap step where ``new_phase=0`` from a non-zero
            ``prev_phase``).  Caller decides how to interpret negatives.
        """
        total = self.data["total_time"][traj_idx]
        tt = self.data["traj_type"][traj_idx]

        delta = step_dt / total
        new_phase = phase + delta

        is_periodic = (tt == _HALF_PERIODIC_INT) | (tt == _FULL_PERIODIC_INT)
        is_episodic = tt == _EPISODIC_INT
        is_perpetual = tt == _PERPETUAL_INT

        new_phase = torch.where(is_periodic, new_phase % 1.0, new_phase)
        new_phase = torch.where(
            is_episodic, torch.clamp(new_phase, 0.0, 1.0), new_phase
        )
        new_phase = torch.where(
            is_perpetual, torch.zeros_like(new_phase), new_phase
        )

        phi_movement = new_phase - phase
        phi_movement = torch.where(
            is_periodic & (new_phase < phase),
            phi_movement + 1.0,
            phi_movement,
        )
        return new_phase, phi_movement

    def update_phase(
        self, step_dt: float, env_ids: Tensor | None = None
    ) -> None:
        """Advance ``self.phase`` by ``step_dt / total_time[traj_idx]``.

        Half/full periodic phases wrap mod 1.0.  Episodic phases clamp at
        1.0.  Perpetual phases stay at 0.

        ``gate_rel_phi`` advances by the same delta but does *not* wrap or
        clamp — it tracks signed phi-distance from the currently armed
        gate monotonically.  The contact-gate logic in the command term
        reads ``gate_rel_phi`` rather than recomputing ``phase - gate_phi``,
        which means no wrap-advance is needed for phi=1.0 gates: a missed
        wrap-gate contact just becomes a (very) late-side fire on the
        next contact event.

        Args:
            step_dt: Sim step duration (seconds).
            env_ids: Optional subset; if ``None``, advances all envs.
        """
        traj_idx_full = self._get_global_indices()
        n_total = traj_idx_full.shape[0]
        self._ensure_phase_state(n_total)

        if env_ids is None:
            sel = torch.arange(n_total, device=self.device)
        else:
            sel = env_ids
        if sel.numel() == 0:
            return

        traj_idx = traj_idx_full[sel]
        prev_phase = self.phase[sel].clone()
        new_phase, phi_movement = self.advance_phase_for_traj(
            prev_phase, traj_idx, step_dt
        )

        self.phase[sel] = new_phase

        # Surface the wrap-aware per-env delta for downstream consumers
        # (e.g. ``CrossfadeState.step``).  Envs not in ``sel`` keep their
        # previous value zeroed out below to avoid drift.
        self.last_phase_delta[:] = 0.0
        self.last_phase_delta[sel] = phi_movement
        gate_armed = self.next_gate_idx[sel] >= 0
        phi_movement = torch.where(gate_armed, phi_movement, torch.zeros_like(phi_movement))
        self.gate_rel_phi[sel] = self.gate_rel_phi[sel] + phi_movement
        # Snapshot the post-update, pre-snap value for diagnostic logging.
        self.gate_rel_phi_pre_snap[sel] = self.gate_rel_phi[sel]

    def reset_phase(self, env_ids: Tensor, randomize: bool = True) -> None:
        """Reset phase for the given envs.

        Args:
            env_ids: Env indices to reset.
            randomize: If True, sample uniform phase ∈ [0, 1).  Otherwise
                set to 0.
        """
        if env_ids.numel() == 0:
            return
        n_total = self._get_global_indices().shape[0]
        self._ensure_phase_state(n_total)

        if randomize:
            new_phase = torch.rand(env_ids.shape[0], device=self.device)
        else:
            new_phase = torch.zeros(env_ids.shape[0], device=self.device)
        self.phase[env_ids] = new_phase
        self._reseed_gate_for_envs(env_ids)

    def set_phase(self, phase: Tensor, env_ids: Tensor) -> None:
        """Explicitly set phase for the given envs.

        Used by ``reset_on_reference`` to align the phase with a sampled
        reference pose.

        Args:
            phase: ``[len(env_ids)]`` phase values in [0, 1].
            env_ids: Env indices to set.
        """
        if env_ids.numel() == 0:
            return
        n_total = self._get_global_indices().shape[0]
        self._ensure_phase_state(n_total)
        self.phase[env_ids] = phase
        self._reseed_gate_for_envs(env_ids)

    def snap_phase_to_new_domain(self, env_ids: Tensor) -> None:
        """Early-contact snap.  Phase currently in old domain (before gate);
        jump *forward* to ``(gate_phi + _EPS_PHI) % 1.0`` — start of the
        new domain — and advance ``next_gate_idx``.
        """
        if env_ids.numel() == 0:
            return
        traj_idx = self._get_global_indices()[env_ids]
        gate_idx = self.next_gate_idx[env_ids]
        gate_phi = self._gate_phi_table[traj_idx, gate_idx]
        self.phase[env_ids] = (gate_phi + self._EPS_PHI) % 1.0
        self._advance_gate_for_envs(env_ids)

    def snap_phase_to_start_of_current_domain(self, env_ids: Tensor) -> None:
        """Late-contact snap (hold-off).  Phase already past the gate (in
        the new domain); pull *backward* a small amount to
        ``(gate_phi + _EPS_PHI) % 1.0`` — start of the same new domain —
        and advance ``next_gate_idx``.

        Same numerical target as :meth:`snap_phase_to_new_domain` but the
        domain-membership intent (already inside vs. crossing into) is
        different at the call site.
        """
        if env_ids.numel() == 0:
            return
        traj_idx = self._get_global_indices()[env_ids]
        gate_idx = self.next_gate_idx[env_ids]
        gate_phi = self._gate_phi_table[traj_idx, gate_idx]
        self.phase[env_ids] = (gate_phi + self._EPS_PHI) % 1.0
        self._advance_gate_for_envs(env_ids)

    def snap_phase_to_end_of_previous_domain(self, env_ids: Tensor) -> None:
        """Hold-on-late-contact snap.  Phase has just crossed the gate
        boundary into the new domain; pull *backward* to
        ``(gate_phi - _EPS_PHI) % 1.0`` — end of the old (previous) domain.
        Does NOT advance ``next_gate_idx`` — we're still waiting for this
        gate's contact event.
        """
        if env_ids.numel() == 0:
            return
        traj_idx = self._get_global_indices()[env_ids]
        gate_idx = self.next_gate_idx[env_ids]
        gate_phi = self._gate_phi_table[traj_idx, gate_idx]
        self.phase[env_ids] = (gate_phi - self._EPS_PHI) % 1.0
        self._refresh_gate_rel_phi(env_ids)

    # ------------------------------------------------------------------
    # Ref frame map building (needs stored metadata)
    # ------------------------------------------------------------------

    def build_ref_frame_map(self, ref_frames: list[str]) -> None:
        """Pre-compute the reference frame lookup table.

        Call this after construction if you need ``get_ref_frames_in_use``
        to work correctly.  It maps each (trajectory, domain) to the
        appropriate index in ``ref_frames``.

        Args:
            ref_frames: List of reference frame names.
        """
        T = self.num_trajectories
        D = self.max_expanded_domains
        table = torch.zeros(T, D, dtype=torch.long, device=self.device)

        for ti in range(T):
            nd = self.data["num_original_domains"][ti].item()
            ed = self.data["expanded_domains"][ti].item()
            is_half = self.data["traj_type"][ti].item() == _HALF_PERIODIC_INT

            for di in range(nd):
                frame = self._ref_frame_names_per_domain[ti][di]
                if frame in ref_frames:
                    table[ti, di] = ref_frames.index(frame)
                else:
                    raise ValueError(
                        f"Bezier frame '{frame}' not found in ref_frames: {ref_frames}"
                    )

            if is_half:
                for di in range(nd):
                    frame = self._ref_frame_names_per_domain[ti][di]
                    reflected = frame.replace("right", "TEMP").replace("left", "right").replace("TEMP", "left")
                    if reflected in ref_frames:
                        table[ti, nd + di] = ref_frames.index(reflected)
                    else:
                        raise ValueError(
                            f"Reflected frame '{reflected}' not found in ref_frames: {ref_frames}"
                        )

        self._ref_frame_domain_map = table
        self._ref_frame_map = table
        self._ref_frame_key = tuple(ref_frames)
