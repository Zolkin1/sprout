# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for MultiSkillManager.

Cross-validates batched results against individual TrajectoryManager outputs
to ensure correctness of the batched tensor implementation.
"""

import math

import pytest
import torch

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.multiskill_manager import (
    MultiSkillManager,
    ConditionerData,
)
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.trajectory_manager import (
    TrajectoryManager,
    TrajectoryType,
)
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.library_manager import (
    LibraryManager,
)
from conftest import MERGED_LIBRARY_DIR, STANDING_YAML, WALKING_YAML, RUNNING_YAML, DEVICE, TEST_DATA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubSkillOwner:
    """Minimal owner exposing ``skill_id`` and ``_skill_list`` for the
    manager's owner-based skill filter.  Tests that build a manager
    directly (without a trajectory cmd) attach one of these so
    ``_select_trajectories`` can run.  ``ref_poses`` is only needed when a
    stair-dim lookup is registered (it anchors the span query).
    """

    def __init__(
        self,
        skill_id: torch.Tensor,
        skill_list: list[str],
        ref_poses: torch.Tensor | None = None,
    ):
        self.skill_id = skill_id
        self._skill_list = skill_list
        if ref_poses is not None:
            self.ref_poses = ref_poses


def _attach_stub_owner(
    msm: MultiSkillManager,
    skill_id_per_env: torch.Tensor,
    ref_poses: torch.Tensor | None = None,
) -> None:
    """Register a stub skill owner whose ``_skill_list`` matches the
    manager's loaded skills in declaration order."""
    skill_list = sorted(
        msm.skill_name_to_idx.keys(),
        key=lambda n: msm.skill_name_to_idx[n],
    )
    msm.set_skill_owner(_StubSkillOwner(skill_id_per_env, skill_list, ref_poses))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def multiskill_single_skill() -> MultiSkillManager:
    """MultiSkillManager with all trajectories as a single 'default' skill."""
    return MultiSkillManager(
        path=MERGED_LIBRARY_DIR,
        device=DEVICE,
    )


@pytest.fixture(scope="module")
def reference_managers() -> list[TrajectoryManager]:
    """Load each trajectory YAML individually for cross-validation."""
    from pathlib import Path
    yamls = sorted(Path(MERGED_LIBRARY_DIR).glob("*.yaml"))
    managers = []
    for y in yamls:
        mgr = TrajectoryManager(str(y), None, DEVICE)
        managers.append(mgr)
    # Sort by first conditioner value (same as MultiSkillManager/LibraryManager)
    managers.sort(key=lambda m: m.traj_data.conditioner[0])
    return managers


@pytest.fixture(scope="module")
def reference_library() -> LibraryManager:
    """LibraryManager for comparison."""
    return LibraryManager(MERGED_LIBRARY_DIR, None, DEVICE)


# ---------------------------------------------------------------------------
# Construction & Loading
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_loads_all_trajectories(self, multiskill_single_skill, reference_managers):
        """Should load the same number of trajectories as individual loading."""
        assert multiskill_single_skill.num_trajectories == len(reference_managers)

    def test_single_skill_created(self, multiskill_single_skill):
        """With one folder, should have exactly one skill."""
        assert len(multiskill_single_skill.skills) == 1
        assert multiskill_single_skill.skills[0].name == "default"

    def test_tensor_shapes(self, multiskill_single_skill):
        """Global tensors should have correct shapes."""
        msm = multiskill_single_skill
        T = msm.num_trajectories
        D = msm.max_expanded_domains
        P = msm.num_pos_outputs
        V = msm.num_vel_outputs
        K = msm.spline_order

        assert msm.data["coeffs_pos"].shape == (T, D, P, K + 1)
        assert msm.data["coeffs_vel"].shape == (T, D, V, K + 1)
        assert msm.data["domain_times"].shape == (T, D)
        assert msm.data["domain_boundaries"].shape == (T, D + 1)
        assert msm.data["total_time"].shape == (T,)
        assert msm.data["expanded_domains"].shape == (T,)
        assert msm.data["traj_type"].shape == (T,)
        assert msm.data["skill_idx"].shape == (T,)

    def test_output_counts_match_reference(self, multiskill_single_skill, reference_managers):
        """Pos/vel output counts should match reference trajectories."""
        ref = reference_managers[0]
        assert multiskill_single_skill.num_pos_outputs == ref.traj_data.num_pos_outputs
        assert multiskill_single_skill.num_vel_outputs == ref.traj_data.num_vel_outputs

    def test_output_names_match_reference(self, multiskill_single_skill, reference_managers):
        """Output names should match reference trajectories."""
        ref = reference_managers[0]
        assert multiskill_single_skill.pos_output_names == ref.traj_data.pos_output_names
        assert multiskill_single_skill.vel_output_names == ref.traj_data.vel_output_names

    def test_domain_boundaries_padded(self, multiskill_single_skill):
        """Padded boundary slots should equal the last real boundary."""
        msm = multiskill_single_skill
        for ti in range(msm.num_trajectories):
            ed = msm.data["expanded_domains"][ti].item()
            last_boundary = msm.data["domain_boundaries"][ti, ed].item()
            for di in range(ed + 1, msm.max_expanded_domains + 1):
                assert msm.data["domain_boundaries"][ti, di].item() == last_boundary

    def test_total_time_correct(self, multiskill_single_skill, reference_managers):
        """Total time should match reference managers."""
        msm = multiskill_single_skill
        for ti, ref in enumerate(reference_managers):
            if ref.traj_data.trajectory_type == TrajectoryType.HALF_PERIODIC:
                expected = ref.traj_data.total_time * 2
            else:
                expected = ref.traj_data.total_time
            actual = msm.data["total_time"][ti].item()
            assert actual == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# Conditioner Parsing
# ---------------------------------------------------------------------------

class TestConditionerParsing:
    def test_old_format_list(self):
        """Old-format [min, max] should parse to midpoint vel_x."""
        cond = MultiSkillManager._parse_conditioner([0.8, 1.2], "test")
        assert cond.vel_x == pytest.approx(1.0)
        assert cond.vel_y == 0.0
        assert cond.terrain == "flat"

    def test_new_format_dict(self):
        """New-format dict should parse all fields."""
        raw = {
            "vel_x": 1.5, "vel_y": 0.1, "vel_yaw": 0.3,
            "terrain": "stairs", "terrain_height": 0.15,
        }
        cond = MultiSkillManager._parse_conditioner(raw, "test")
        assert cond.vel_x == 1.5
        assert cond.vel_y == 0.1
        assert cond.vel_yaw == 0.3
        assert cond.terrain == "stairs"
        assert cond.terrain_height == 0.15
        assert cond.terrain_width == 0.0  # default

    def test_to_continuous_tensor(self):
        """to_continuous_tensor should return 6 floats."""
        cond = ConditionerData(vel_x=1.0, vel_y=2.0, vel_yaw=3.0)
        vals = cond.to_continuous_tensor()
        assert len(vals) == 6
        assert vals[0] == 1.0
        assert vals[1] == 2.0
        assert vals[2] == 3.0

    def test_fixed_terrain_dim_bounds(self):
        """A single-number terrain dim parses to a degenerate range lo == hi."""
        cond = MultiSkillManager._parse_conditioner(
            {"terrain": "stairs", "terrain_height": 0.135, "terrain_length": 0.233},
            "stairs",
        )
        assert cond.terrain_height_bounds == (0.135, 0.135)
        assert cond.terrain_length_bounds == (0.233, 0.233)
        assert cond.terrain_height == pytest.approx(0.135)

    def test_open_range_terrain_dim(self):
        """A ``{min: v}`` range (no max) -> (v, +inf); representative = min."""
        cond = MultiSkillManager._parse_conditioner(
            {"terrain": "stairs", "terrain_width": {"min": 1.0}}, "stairs",
        )
        lo, hi = cond.terrain_width_bounds
        assert lo == pytest.approx(1.0)
        assert math.isinf(hi)
        assert cond.terrain_width == pytest.approx(1.0)

    def test_closed_range_terrain_dim(self):
        """A ``{min, max}`` range keeps both bounds; representative = midpoint."""
        cond = MultiSkillManager._parse_conditioner(
            {"terrain": "stairs", "terrain_width": {"min": 1.0, "max": 2.0}}, "stairs",
        )
        assert cond.terrain_width_bounds == (1.0, 2.0)
        assert cond.terrain_width == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Trajectory Selection
# ---------------------------------------------------------------------------

class TestTrajectorySelection:
    def test_nearest_neighbor_exact(self, multiskill_single_skill):
        """Exact match on conditioning should select that trajectory."""
        msm = multiskill_single_skill
        skill = msm.skills[0]

        # Single-skill manager: every env's skill_id = 0.
        _attach_stub_owner(msm, torch.zeros(1, dtype=torch.long, device=DEVICE))

        # Use the first trajectory's conditioning vector (pad to 6 dims)
        cond_vec = skill.conditioning_tensor[0:1]  # [1, C]
        global_idx = msm._select_trajectories(cond_vec)
        assert global_idx.item() == skill.traj_indices[0]

    def test_nearest_neighbor_batch(self, multiskill_single_skill):
        """Batch selection should return valid indices."""
        msm = multiskill_single_skill

        N = 10
        _attach_stub_owner(msm, torch.zeros(N, dtype=torch.long, device=DEVICE))
        cond = torch.randn(N, 6)
        global_idx = msm._select_trajectories(cond)
        assert global_idx.shape == (N,)
        assert (global_idx >= 0).all()
        assert (global_idx < msm.num_trajectories).all()

    def test_forced_traj_latch_overrides_selection(self, multiskill_single_skill):
        """An active forced-traj latch wins over the velocity argmin; an
        inactive row falls through to the normal nearest pick."""
        from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.skill_state import (
            ForcedTrajLatch,
        )

        msm = multiskill_single_skill
        N = 3
        _attach_stub_owner(msm, torch.zeros(N, dtype=torch.long, device=DEVICE))

        # Baseline selection from a fixed conditioner.
        cond = torch.zeros(N, 6, device=DEVICE)
        baseline = msm._select_trajectories(cond).clone()

        # Latch env 0 and env 2 to a specific (different) trajectory; leave
        # env 1 unlatched.
        forced_idx = (baseline[0].item() + 1) % msm.num_trajectories
        latch = ForcedTrajLatch(N, device=DEVICE)
        latch.set(torch.tensor([0, 2], device=DEVICE),
                  torch.tensor([forced_idx, forced_idx], device=DEVICE))
        msm.set_forced_traj_latch(latch)
        try:
            out = msm._select_trajectories(cond)
            assert out[0].item() == forced_idx
            assert out[2].item() == forced_idx
            # Unlatched row unchanged.
            assert out[1].item() == baseline[1].item()
        finally:
            # Don't leak the latch into other tests sharing the fixture.
            msm.set_forced_traj_latch(None)


# ---------------------------------------------------------------------------
# Cross-Validation: Output
# ---------------------------------------------------------------------------

class TestBatchedOutput:
    """Cross-validate batched MultiSkillManager output against individual TrajectoryManagers."""

    @pytest.mark.parametrize("traj_idx", [0, 5, 10, 18])
    def test_get_output_matches_reference(
        self, multiskill_single_skill, reference_managers, traj_idx
    ):
        """Batched output should match per-trajectory TrajectoryManager output."""
        if traj_idx >= len(reference_managers):
            pytest.skip("Not enough trajectories")

        msm = multiskill_single_skill
        ref = reference_managers[traj_idx]
        N = 8

        # Test times spanning the trajectory
        total_t = msm.data["total_time"][traj_idx].item()
        t = torch.linspace(0.001, total_t * 0.99, N)
        phase = t / total_t

        # Set all envs to this trajectory
        msm.set_trajectory_indices(torch.full((N,), traj_idx, dtype=torch.long))

        pos_msm, vel_msm = msm.get_output(phase)
        pos_ref, vel_ref = ref.get_output(t)

        assert torch.allclose(pos_msm, pos_ref, atol=1e-4, rtol=1e-4), (
            f"Traj {traj_idx} pos mismatch: max diff = {(pos_msm - pos_ref).abs().max()}"
        )
        assert torch.allclose(vel_msm, vel_ref, atol=1e-4, rtol=1e-4), (
            f"Traj {traj_idx} vel mismatch: max diff = {(vel_msm - vel_ref).abs().max()}"
        )

    def test_multi_traj_dispatch(self, multiskill_single_skill, reference_managers):
        """With mixed trajectory assignments, each env should match its reference."""
        msm = multiskill_single_skill
        num_traj = len(reference_managers)

        # 4 envs: first 2 use traj 0, last 2 use last traj
        idx_a, idx_b = 0, num_traj - 1
        indices = torch.tensor([idx_a, idx_a, idx_b, idx_b], dtype=torch.long)
        msm.set_trajectory_indices(indices)

        t = torch.tensor([0.05, 0.1, 0.05, 0.1])
        total_a = msm.data["total_time"][idx_a].item()
        total_b = msm.data["total_time"][idx_b].item()
        phase = torch.tensor([t[0] / total_a, t[1] / total_a, t[2] / total_b, t[3] / total_b])

        pos_msm, vel_msm = msm.get_output(phase)

        pos_a, vel_a = reference_managers[idx_a].get_output(t[:2])
        pos_b, vel_b = reference_managers[idx_b].get_output(t[2:])

        assert torch.allclose(pos_msm[:2], pos_a, atol=1e-4)
        assert torch.allclose(vel_msm[:2], vel_a, atol=1e-4)
        assert torch.allclose(pos_msm[2:], pos_b, atol=1e-4)
        assert torch.allclose(vel_msm[2:], vel_b, atol=1e-4)

    def test_output_at_t_zero(self, multiskill_single_skill, reference_managers):
        """Output at phase=0 should match reference for all trajectories."""
        msm = multiskill_single_skill
        T = msm.num_trajectories
        # Slightly off zero to avoid boundary; same value works as both phase
        # and time since reference managers use whatever value as time directly.
        t = torch.full((T,), 0.001)
        totals = msm.data["total_time"][:T]
        phase = t / totals
        indices = torch.arange(T, dtype=torch.long)
        msm.set_trajectory_indices(indices)

        pos_msm, vel_msm = msm.get_output(phase)

        for ti in range(T):
            pos_ref, vel_ref = reference_managers[ti].get_output(t[ti:ti+1])
            assert torch.allclose(pos_msm[ti:ti+1], pos_ref, atol=1e-4), (
                f"Traj {ti} phase=0 pos mismatch"
            )
            assert torch.allclose(vel_msm[ti:ti+1], vel_ref, atol=1e-4), (
                f"Traj {ti} phase=0 vel mismatch"
            )


# ---------------------------------------------------------------------------
# Cross-Validation: Phasing Variable
# ---------------------------------------------------------------------------

class TestPhase:
    """Phase state lives directly on the manager — these tests pin the
    set / read round-trip.  Earlier ``get_phasing_var(t)`` tests are gone:
    after the time→phase refactor, ``manager.phase`` *is* the phasing
    variable, so there is no transformation left to cross-validate."""

    def test_set_phase_round_trip(self, multiskill_single_skill):
        """``set_phase`` writes; ``manager.phase`` reads back the same value."""
        msm = multiskill_single_skill
        N = 4
        msm.set_trajectory_indices(torch.zeros(N, dtype=torch.long, device=DEVICE))
        msm._ensure_phase_state(N)
        env_ids = torch.arange(N, device=DEVICE)
        target = torch.tensor([0.0, 0.25, 0.5, 0.99], device=DEVICE)
        msm.set_phase(target, env_ids)
        assert torch.allclose(msm.phase, target, atol=1e-7)

    def test_phase_in_unit_interval_after_update(self, multiskill_single_skill):
        """``update_phase`` keeps phase in [0, 1] for all trajectory types."""
        msm = multiskill_single_skill
        N = 20
        msm.set_trajectory_indices(torch.zeros(N, dtype=torch.long, device=DEVICE))
        msm._ensure_phase_state(N)
        # Seed near the wrap so periodic trajectories actually wrap.
        msm.phase[:] = torch.linspace(0.9, 0.999, N, device=DEVICE)
        # update_phase needs an env with step_dt; reuse the stub via msm.env
        # if present, otherwise skip.
        if msm.env is None or not hasattr(msm.env, "step_dt"):
            pytest.skip("Manager has no env; cannot exercise update_phase")
        msm.update_phase(msm.env.step_dt)
        assert (msm.phase >= 0).all()
        assert (msm.phase <= 1).all()


# ---------------------------------------------------------------------------
# Cross-Validation: Domains
# ---------------------------------------------------------------------------

class TestDomains:
    @pytest.mark.parametrize("traj_idx", [0, 5, 10, 18])
    def test_domains_match_reference(
        self, multiskill_single_skill, reference_managers, traj_idx
    ):
        """Domain indices should match per-trajectory result."""
        if traj_idx >= len(reference_managers):
            pytest.skip("Not enough trajectories")

        msm = multiskill_single_skill
        ref = reference_managers[traj_idx]
        N = 8

        total_t = msm.data["total_time"][traj_idx].item()
        t = torch.linspace(0.001, total_t * 0.99, N)
        phase = t / total_t

        msm.set_trajectory_indices(torch.full((N,), traj_idx, dtype=torch.long))
        dom_msm = msm.get_current_domains(phase)
        dom_ref = ref.get_current_domains(t)

        assert torch.equal(dom_msm, dom_ref), (
            f"Traj {traj_idx} domain mismatch:\nMSM: {dom_msm}\nRef: {dom_ref}"
        )

    @pytest.mark.parametrize("traj_idx", [0, 5, 10, 18])
    def test_domain_times_match_reference(
        self, multiskill_single_skill, reference_managers, traj_idx
    ):
        """Domain durations should match per-trajectory result."""
        if traj_idx >= len(reference_managers):
            pytest.skip("Not enough trajectories")

        msm = multiskill_single_skill
        ref = reference_managers[traj_idx]
        N = 8

        total_t = msm.data["total_time"][traj_idx].item()
        t = torch.linspace(0.001, total_t * 0.99, N)
        phase = t / total_t

        msm.set_trajectory_indices(torch.full((N,), traj_idx, dtype=torch.long))
        dt_msm = msm.get_domain_times(phase)
        dt_ref = ref.get_domain_times(t)

        assert torch.allclose(dt_msm, dt_ref, atol=1e-6), (
            f"Traj {traj_idx} domain time mismatch"
        )

    def test_padded_domains_never_selected(self, multiskill_single_skill):
        """searchsorted should never land in a padded domain slot."""
        msm = multiskill_single_skill
        T = msm.num_trajectories
        N = T
        phase = torch.rand(N)
        indices = torch.arange(T, dtype=torch.long)
        msm.set_trajectory_indices(indices)

        dom = msm.get_current_domains(phase)
        max_valid = msm.data["expanded_domains"][indices] - 1
        assert (dom <= max_valid).all()


# ---------------------------------------------------------------------------
# Cross-Validation: Contact State
# ---------------------------------------------------------------------------

class TestContactState:
    @pytest.mark.parametrize("traj_idx", [0, 5, 10])
    def test_contact_state_matches_reference(
        self, multiskill_single_skill, reference_managers, traj_idx
    ):
        """Contact states should match per-trajectory result."""
        if traj_idx >= len(reference_managers):
            pytest.skip("Not enough trajectories")

        msm = multiskill_single_skill
        ref = reference_managers[traj_idx]
        N = 8
        contact_frames = ["right_ankle_roll_link", "left_ankle_roll_link"]

        total_t = msm.data["total_time"][traj_idx].item()
        t = torch.linspace(0.001, total_t * 0.99, N)
        phase = t / total_t

        msm.set_trajectory_indices(torch.full((N,), traj_idx, dtype=torch.long))
        cs_msm = msm.get_contact_state(phase, contact_frames)
        cs_ref = ref.get_contact_state(t, contact_frames)

        assert torch.equal(cs_msm, cs_ref), (
            f"Traj {traj_idx} contact mismatch:\nMSM: {cs_msm}\nRef: {cs_ref}"
        )


# ---------------------------------------------------------------------------
# Output Ordering
# ---------------------------------------------------------------------------

class TestOrderOutputs:
    def test_order_outputs_reversed(self, reference_managers):
        """Reversing output order should still produce correct results."""
        msm = MultiSkillManager(
            path=MERGED_LIBRARY_DIR,
            device=DEVICE,
        )
        ref = reference_managers[0]

        # Get output in original order
        N = 4
        total_t = msm.data["total_time"][0].item()
        t = torch.tensor([0.01, 0.05, 0.1, 0.15])
        phase = t / total_t
        msm.set_trajectory_indices(torch.zeros(N, dtype=torch.long))
        pos_orig, vel_orig = msm.get_output(phase)

        # Reverse output names
        rev_pos = list(reversed(msm.pos_output_names))
        rev_vel = list(reversed(msm.vel_output_names))
        msm.order_outputs(rev_pos, rev_vel)

        pos_rev, vel_rev = msm.get_output(phase)

        # Reversed output should be the original flipped
        assert torch.allclose(pos_rev, pos_orig.flip(1), atol=1e-6)
        assert torch.allclose(vel_rev, vel_orig.flip(1), atol=1e-6)


# ---------------------------------------------------------------------------
# Placeholder Stubs
# ---------------------------------------------------------------------------

class TestPlaceholders:
    def test_terrain_relative_identity(self, multiskill_single_skill):
        """transform_terrain_relative should return input unchanged."""
        msm = multiskill_single_skill
        positions = torch.randn(4, 3)
        result = msm.transform_terrain_relative(positions)
        assert torch.equal(result, positions)

# ---------------------------------------------------------------------------
# Properties & Interface
# ---------------------------------------------------------------------------

class TestInterface:
    def test_get_num_pos_outputs(self, multiskill_single_skill):
        assert multiskill_single_skill.get_num_pos_outputs() == multiskill_single_skill.num_pos_outputs

    def test_get_num_vel_outputs(self, multiskill_single_skill):
        assert multiskill_single_skill.get_num_vel_outputs() == multiskill_single_skill.num_vel_outputs

    def test_get_pos_output_names(self, multiskill_single_skill):
        assert multiskill_single_skill.get_pos_output_names == multiskill_single_skill.pos_output_names

    def test_get_vel_output_names(self, multiskill_single_skill):
        assert multiskill_single_skill.get_vel_output_names == multiskill_single_skill.vel_output_names

    def test_get_reference_frames(self, multiskill_single_skill):
        assert isinstance(multiskill_single_skill.get_reference_frames(), list)

    def test_invalidate_cache(self, multiskill_single_skill):
        """Cache invalidation should clear the valid flag."""
        msm = multiskill_single_skill
        msm.set_trajectory_indices(torch.zeros(4, dtype=torch.long))
        msm.invalidate_cache()
        assert msm._cache_valid is False


# ---------------------------------------------------------------------------
# Multi-Skill Loading (using separate folders)
# ---------------------------------------------------------------------------

class TestMultiSkill:
    @pytest.fixture()
    def multi_skill_dir(self, tmp_path):
        """Create a directory with two skill subfolders, each symlinking the merged library."""
        import shutil
        from pathlib import Path

        skill_a = tmp_path / "skill_a"
        skill_b = tmp_path / "skill_b"

        # Copy YAMLs into two separate subfolders
        shutil.copytree(MERGED_LIBRARY_DIR, str(skill_a))
        shutil.copytree(MERGED_LIBRARY_DIR, str(skill_b))

        return tmp_path

    def test_two_skills_discovered(self, multi_skill_dir):
        """Subdirectories should be discovered as separate skills."""
        msm = MultiSkillManager(path=str(multi_skill_dir), device=DEVICE)
        # Each subfolder has 19 trajectories
        assert msm.num_trajectories == 38
        assert len(msm.skills) == 2
        assert msm.skills[0].num_trajectories == 19
        assert msm.skills[1].num_trajectories == 19

    def test_skill_indices_distinct(self, multi_skill_dir):
        """Different skills should have non-overlapping trajectory indices."""
        msm = MultiSkillManager(path=str(multi_skill_dir), device=DEVICE)
        indices_a = set(msm.skills[0].traj_indices)
        indices_b = set(msm.skills[1].traj_indices)
        assert len(indices_a & indices_b) == 0

    def test_skill_selection_routes_by_velocity(self, multi_skill_dir):
        """Global selection should pick the trajectory closest to the commanded velocity."""
        msm = MultiSkillManager(path=str(multi_skill_dir), device=DEVICE)

        # Pick a skill_id that admits all trajectories: the skill whose
        # bucket contains the lowest-vel trajectory for the zero case,
        # and the skill containing the high-vel trajectory for the high
        # case.  In practice both skills' buckets should be queried in
        # turn — but the existing test only validates "different command
        # → different traj", which holds as long as both envs use the
        # same skill_id for each call.  Use skill 0 throughout (the
        # cdist filter will mask out skill-1 trajectories) and verify
        # the test's invariant within that skill.
        _attach_stub_owner(msm, torch.zeros(5, dtype=torch.long, device=DEVICE))

        # Zero velocity should select the trajectory with lowest vel_x
        # within the env's skill.
        cond_zero = torch.zeros(5, 6)
        idx_zero = msm._select_trajectories(cond_zero)
        assert idx_zero.shape == (5,)
        # All envs should get the same trajectory (all have same conditioning).
        assert (idx_zero == idx_zero[0]).all()

        # High velocity should select a different trajectory than zero
        # (still within skill 0).
        cond_high = torch.zeros(5, 6)
        cond_high[:, 0] = 100.0  # very high vel_x
        idx_high = msm._select_trajectories(cond_high)
        assert (idx_high == idx_high[0]).all()
        # Different commanded velocity → different traj within the skill.
        assert idx_high[0].item() != idx_zero[0].item()

    def test_single_folder_fallback(self):
        """A flat folder with YAMLs (no subfolders) should create a 'default' skill."""
        msm = MultiSkillManager(path=MERGED_LIBRARY_DIR, device=DEVICE)
        assert len(msm.skills) == 1
        assert msm.skills[0].name == "default"
        assert msm.num_trajectories == 19


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_empty_folder_raises(self, tmp_path):
        """Empty folder with no YAMLs and no subfolders should raise."""
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="No skill subfolders"):
            MultiSkillManager(path=str(empty), device=DEVICE)

    def test_missing_folder_raises(self):
        """Non-existent folder should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            MultiSkillManager(path="/nonexistent/path", device=DEVICE)

    def test_no_env_cache_raises(self, multiskill_single_skill):
        """_ensure_cache without env or set_trajectory_indices should raise."""
        msm = multiskill_single_skill
        msm.invalidate_cache()
        with pytest.raises(RuntimeError, match="Cannot auto-populate"):
            msm._ensure_cache()


# ---------------------------------------------------------------------------
# Contact-gate state machinery: phase, next_gate_idx, gate_rel_phi
# ---------------------------------------------------------------------------


class _StubEnv:
    """Minimal env stub exposing ``step_dt`` for the snap operations."""

    def __init__(self, step_dt: float = 0.02):
        self.step_dt = step_dt


def _find_traj_of_type(msm: MultiSkillManager, type_int: int) -> int:
    """Return the index of the first trajectory of the given type, or -1."""
    for ti in range(msm.num_trajectories):
        if msm.data["traj_type"][ti].item() == type_int:
            return ti
    return -1


class TestGateState:
    """Direct tests for gate-state primitives.

    These exercise ``_refresh_gate_rel_phi``, ``update_phase``,
    ``_advance_gate_for_envs``, and the three snap methods by manipulating
    per-env phase state by hand — no IsaacLab env required, just a
    ``_StubEnv`` exposing ``step_dt`` for the snap path.
    """

    @pytest.fixture
    def msm(self) -> MultiSkillManager:
        """Fresh MSM with a stub env so ``_eps_phi`` works for snap tests."""
        return MultiSkillManager(
            path=MERGED_LIBRARY_DIR,
            device=DEVICE,
            env=_StubEnv(step_dt=0.02),
        )

    @pytest.fixture
    def half_periodic_idx(self, msm) -> int:
        ti = _find_traj_of_type(msm, 0)  # _HALF_PERIODIC_INT
        if ti < 0:
            pytest.skip("No half-periodic trajectory in test data")
        return ti

    # --- _refresh_gate_rel_phi -------------------------------------------

    def test_refresh_with_active_gate(self, msm, half_periodic_idx):
        """gate_rel_phi = phase - gate_phi[armed] for active gates."""
        ti = half_periodic_idx
        N = 4
        msm.set_trajectory_indices(torch.full((N,), ti, dtype=torch.long, device=DEVICE))
        msm._ensure_phase_state(N)

        # Half-periodic gates are at phi=0.5 (gate 0) and phi=1.0 (gate 1).
        msm.phase[:] = torch.tensor([0.30, 0.50, 0.70, 0.95], device=DEVICE)
        msm.next_gate_idx[:] = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=DEVICE)
        msm.gate_rel_phi[:] = -99.0  # poison to ensure refresh writes

        msm._refresh_gate_rel_phi(torch.arange(N, device=DEVICE))

        expected = torch.tensor(
            [0.30 - 0.50, 0.50 - 0.50, 0.70 - 1.00, 0.95 - 1.00],
            device=DEVICE,
        )
        assert torch.allclose(msm.gate_rel_phi, expected, atol=1e-6)

    def test_refresh_with_inactive_gate_writes_zero(self, msm, half_periodic_idx):
        """gate_rel_phi = 0 when next_gate_idx == -1, regardless of phase."""
        ti = half_periodic_idx
        N = 3
        msm.set_trajectory_indices(torch.full((N,), ti, dtype=torch.long, device=DEVICE))
        msm._ensure_phase_state(N)

        msm.phase[:] = torch.tensor([0.10, 0.50, 0.90], device=DEVICE)
        msm.next_gate_idx[:] = torch.tensor([-1, -1, -1], dtype=torch.long, device=DEVICE)
        msm.gate_rel_phi[:] = 99.0  # poison

        msm._refresh_gate_rel_phi(torch.arange(N, device=DEVICE))

        assert torch.allclose(msm.gate_rel_phi, torch.zeros(N, device=DEVICE))

    def test_refresh_subset_only(self, msm, half_periodic_idx):
        """Refresh leaves non-targeted envs untouched."""
        ti = half_periodic_idx
        N = 3
        msm.set_trajectory_indices(torch.full((N,), ti, dtype=torch.long, device=DEVICE))
        msm._ensure_phase_state(N)

        msm.phase[:] = 0.3
        msm.next_gate_idx[:] = 0
        msm.gate_rel_phi[:] = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)

        msm._refresh_gate_rel_phi(torch.tensor([1], device=DEVICE))

        # Only env 1 was refreshed; envs 0 and 2 keep poisoned values.
        assert msm.gate_rel_phi[0].item() == pytest.approx(1.0)
        assert msm.gate_rel_phi[1].item() == pytest.approx(0.3 - 0.5, abs=1e-6)
        assert msm.gate_rel_phi[2].item() == pytest.approx(3.0)

    # --- update_phase: gate_rel_phi accumulation -------------------------

    def test_update_phase_no_wrap(self, msm, half_periodic_idx):
        """Without a wrap, gate_rel_phi advances by step_dt / total."""
        ti = half_periodic_idx
        N = 1
        msm.set_trajectory_indices(torch.tensor([ti], dtype=torch.long, device=DEVICE))
        msm._ensure_phase_state(N)

        total = msm.data["total_time"][ti].item()
        step_dt = 0.02
        delta = step_dt / total

        msm.phase[:] = 0.30
        msm.next_gate_idx[:] = 0
        msm.gate_rel_phi[:] = 0.30 - 0.50

        msm.update_phase(step_dt)

        assert msm.phase[0].item() == pytest.approx(0.30 + delta, abs=1e-6)
        assert msm.gate_rel_phi[0].item() == pytest.approx((0.30 - 0.50) + delta, abs=1e-6)

    def test_update_phase_does_not_drift_when_gate_disarmed(self, msm, half_periodic_idx):
        """``gate_rel_phi`` must not accumulate while ``next_gate_idx == -1``.

        Without this guard, an env that ran on a no-gate trajectory and
        later transitioned to a walking trajectory would see ``gate_rel_phi``
        ramp linearly forever (no anchor) — visible in domain-info plots
        as a monotonic line with no resets.
        """
        ti = half_periodic_idx
        N = 1
        msm.set_trajectory_indices(torch.tensor([ti], dtype=torch.long, device=DEVICE))
        msm._ensure_phase_state(N)

        msm.phase[:] = 0.30
        msm.next_gate_idx[:] = -1  # disarmed
        msm.gate_rel_phi[:] = 0.0

        msm.update_phase(0.02)

        # Phase still advances (the trajectory clock is independent).
        assert msm.phase[0].item() > 0.30
        # gate_rel_phi must stay put while the gate is disarmed.
        assert msm.gate_rel_phi[0].item() == pytest.approx(0.0, abs=1e-7)

    def test_update_phase_wrap_unwraps_gate_rel_phi(self, msm, half_periodic_idx):
        """Across a phase wrap, gate_rel_phi advances by the un-wrapped delta."""
        ti = half_periodic_idx
        N = 1
        msm.set_trajectory_indices(torch.tensor([ti], dtype=torch.long, device=DEVICE))
        msm._ensure_phase_state(N)

        total = msm.data["total_time"][ti].item()
        step_dt = 0.02
        delta = step_dt / total

        # Set up just before the wrap with gate 1 (phi=1.0) armed.
        prev_phase = 1.0 - 0.5 * delta  # advancing will cross 1.0
        msm.phase[:] = prev_phase
        msm.next_gate_idx[:] = 1
        msm.gate_rel_phi[:] = prev_phase - 1.0  # small negative

        msm.update_phase(step_dt)

        # phase wrapped to a small positive
        assert msm.phase[0].item() < prev_phase
        # gate_rel_phi advanced by exactly delta (no wrap effect)
        expected = (prev_phase - 1.0) + delta
        assert msm.gate_rel_phi[0].item() == pytest.approx(expected, abs=1e-6)
        # The wrap moved us from before the gate to past it.
        assert msm.gate_rel_phi[0].item() > 0.0

    def test_update_phase_does_not_advance_next_gate_idx(self, msm, half_periodic_idx):
        """update_phase must not call _advance_gate_for_envs (auto-advance gone)."""
        ti = half_periodic_idx
        N = 1
        msm.set_trajectory_indices(torch.tensor([ti], dtype=torch.long, device=DEVICE))
        msm._ensure_phase_state(N)

        # Phase set up to wrap with gate 1 (phi=1.0) armed — historically
        # would have triggered wrap-advance. Must NOT advance now.
        total = msm.data["total_time"][ti].item()
        step_dt = 0.02
        delta = step_dt / total

        msm.phase[:] = 1.0 - 0.5 * delta
        msm.next_gate_idx[:] = 1
        msm._refresh_gate_rel_phi(torch.tensor([0], device=DEVICE))

        msm.update_phase(step_dt)

        assert msm.next_gate_idx[0].item() == 1, (
            "update_phase should no longer auto-advance the gate index — "
            "gate hand-off now happens via late-fire on contact."
        )

    # --- _advance_gate_for_envs ------------------------------------------

    def test_advance_half_periodic_0_to_1(self, msm, half_periodic_idx):
        """Half-periodic gate 0 → 1; gate_rel_phi recomputed against phi=1.0."""
        ti = half_periodic_idx
        N = 1
        msm.set_trajectory_indices(torch.tensor([ti], dtype=torch.long, device=DEVICE))
        msm._ensure_phase_state(N)

        msm.phase[:] = 0.60  # past gate 0
        msm.next_gate_idx[:] = 0
        msm.gate_rel_phi[:] = 0.10

        msm._advance_gate_for_envs(torch.tensor([0], device=DEVICE))

        assert msm.next_gate_idx[0].item() == 1
        # gate_rel_phi = phase - gate_phi[1] = 0.6 - 1.0 = -0.4
        assert msm.gate_rel_phi[0].item() == pytest.approx(-0.40, abs=1e-6)

    def test_advance_half_periodic_1_to_0(self, msm, half_periodic_idx):
        """Half-periodic gate 1 → 0 (wraps); gate_rel_phi against phi=0.5."""
        ti = half_periodic_idx
        N = 1
        msm.set_trajectory_indices(torch.tensor([ti], dtype=torch.long, device=DEVICE))
        msm._ensure_phase_state(N)

        msm.phase[:] = 0.05  # post-wrap phase
        msm.next_gate_idx[:] = 1

        msm._advance_gate_for_envs(torch.tensor([0], device=DEVICE))

        assert msm.next_gate_idx[0].item() == 0
        assert msm.gate_rel_phi[0].item() == pytest.approx(0.05 - 0.50, abs=1e-6)

    def test_advance_episodic_to_neg_one_zeros_gate_rel_phi(self, msm):
        """Episodic last gate fire → gate_idx=-1, gate_rel_phi=0."""
        ti = _find_traj_of_type(msm, 2)  # _EPISODIC_INT
        if ti < 0:
            pytest.skip("No episodic trajectory in test data")
        N = 1
        msm.set_trajectory_indices(torch.tensor([ti], dtype=torch.long, device=DEVICE))
        msm._ensure_phase_state(N)

        # Episodic has one gate at phi=1.0; advancing past it sets gate=-1.
        msm.phase[:] = 1.0
        msm.next_gate_idx[:] = 0
        msm.gate_rel_phi[:] = 99.0  # poison

        msm._advance_gate_for_envs(torch.tensor([0], device=DEVICE))

        assert msm.next_gate_idx[0].item() == -1
        assert msm.gate_rel_phi[0].item() == 0.0

    # --- snap operations -------------------------------------------------

    def test_snap_to_new_domain(self, msm, half_periodic_idx):
        """Early-fire snap: phase → gate_phi+eps, gate advances, gate_rel_phi consistent."""
        ti = half_periodic_idx
        N = 1
        msm.set_trajectory_indices(torch.tensor([ti], dtype=torch.long, device=DEVICE))
        msm._ensure_phase_state(N)

        msm.phase[:] = 0.46  # in early window of gate 0 (phi=0.5)
        msm.next_gate_idx[:] = 0

        eps = 0.001

        msm.snap_phase_to_new_domain(torch.tensor([0], device=DEVICE))

        assert msm.next_gate_idx[0].item() == 1
        assert msm.phase[0].item() == pytest.approx(0.50 + eps, abs=1e-6)
        # gate_rel_phi = (0.5 + eps) - 1.0
        assert msm.gate_rel_phi[0].item() == pytest.approx(-0.50 + eps, abs=1e-6)

    def test_snap_to_start_of_current_domain(self, msm, half_periodic_idx):
        """Late-fire snap: same numerical target as new-domain snap."""
        ti = half_periodic_idx
        N = 1
        msm.set_trajectory_indices(torch.tensor([ti], dtype=torch.long, device=DEVICE))
        msm._ensure_phase_state(N)

        msm.phase[:] = 0.62  # past gate 0
        msm.next_gate_idx[:] = 0

        eps = 0.001

        msm.snap_phase_to_start_of_current_domain(torch.tensor([0], device=DEVICE))

        assert msm.next_gate_idx[0].item() == 1
        assert msm.phase[0].item() == pytest.approx(0.50 + eps, abs=1e-6)
        assert msm.gate_rel_phi[0].item() == pytest.approx(-0.50 + eps, abs=1e-6)

    def test_snap_to_end_of_previous_domain(self, msm, half_periodic_idx):
        """Hold-on snap: phase pulls back, gate idx unchanged, gate_rel_phi=-eps."""
        ti = half_periodic_idx
        N = 1
        msm.set_trajectory_indices(torch.tensor([ti], dtype=torch.long, device=DEVICE))
        msm._ensure_phase_state(N)

        msm.phase[:] = 0.55
        msm.next_gate_idx[:] = 0

        eps = 0.001

        msm.snap_phase_to_end_of_previous_domain(torch.tensor([0], device=DEVICE))

        # Gate not advanced — still waiting for the contact.
        assert msm.next_gate_idx[0].item() == 0
        assert msm.phase[0].item() == pytest.approx(0.50 - eps, abs=1e-6)
        # gate_rel_phi = (0.5 - eps) - 0.5 = -eps
        assert msm.gate_rel_phi[0].item() == pytest.approx(-eps, abs=1e-6)

    def test_snap_phi_1_gate_wraps_to_eps(self, msm, half_periodic_idx):
        """Snap on phi=1.0 gate wraps phase to eps (mod 1.0)."""
        ti = half_periodic_idx
        N = 1
        msm.set_trajectory_indices(torch.tensor([ti], dtype=torch.long, device=DEVICE))
        msm._ensure_phase_state(N)

        msm.phase[:] = 0.97  # in early window of gate 1 (phi=1.0)
        msm.next_gate_idx[:] = 1

        eps = 0.001

        msm.snap_phase_to_new_domain(torch.tensor([0], device=DEVICE))

        assert msm.next_gate_idx[0].item() == 0
        assert msm.phase[0].item() == pytest.approx(eps, abs=1e-6)
        # gate_rel_phi = eps - 0.5
        assert msm.gate_rel_phi[0].item() == pytest.approx(eps - 0.50, abs=1e-6)

    def test_snap_lands_in_intended_domain(self, msm, half_periodic_idx):
        """``_eps_phi``-driven snaps must land phase in the domain promised
        by each snap's docstring.

        ``snap_phase_to_new_domain`` and
        ``snap_phase_to_start_of_current_domain`` both set
        ``phase = gate_phi + eps``; ``get_current_domains`` should report
        the **new** domain (gate_idx + 1 within original domains).
        ``snap_phase_to_end_of_previous_domain`` sets ``phase = gate_phi -
        eps``; ``get_current_domains`` should report the **previous**
        domain.  Sanity-checks that ``eps_phi = 0.001`` is large enough to
        clear ``searchsorted(..., right=False)``'s tie boundary.
        """
        ti = half_periodic_idx
        N = 1
        msm.set_trajectory_indices(torch.tensor([ti], dtype=torch.long, device=DEVICE))
        msm._ensure_phase_state(N)

        # Gate 0 of half-periodic sits at phi=0.5; before the snap we're in
        # the old (first) domain.
        msm.phase[:] = 0.46
        msm.next_gate_idx[:] = 0
        old_domain = msm.get_current_domains(msm.phase).clone()

        msm.snap_phase_to_new_domain(torch.tensor([0], device=DEVICE))
        post = msm.get_current_domains(msm.phase)
        assert post.item() != old_domain.item(), (
            f"snap_phase_to_new_domain left phase ({msm.phase.item()}) in the "
            f"old domain ({old_domain.item()})."
        )

        # Reset and exercise the late-fire snap (same target).
        msm.phase[:] = 0.62
        msm.next_gate_idx[:] = 0
        msm.snap_phase_to_start_of_current_domain(torch.tensor([0], device=DEVICE))
        post = msm.get_current_domains(msm.phase)
        assert post.item() == old_domain.item() + 1, (
            f"snap_phase_to_start_of_current_domain landed in domain "
            f"{post.item()}; expected {old_domain.item() + 1}."
        )

        # Reset and exercise the hold-on snap (lands in OLD domain).
        msm.phase[:] = 0.55
        msm.next_gate_idx[:] = 0
        msm.snap_phase_to_end_of_previous_domain(torch.tensor([0], device=DEVICE))
        post = msm.get_current_domains(msm.phase)
        assert post.item() == old_domain.item(), (
            f"snap_phase_to_end_of_previous_domain landed in domain "
            f"{post.item()}; expected {old_domain.item()}."
        )



# ---------------------------------------------------------------------------
# Terrain-dim filter
# ---------------------------------------------------------------------------

# Stair geometry grid baked into the synthetic stair skill below, as
# (terrain_height, terrain_length) = (step rise, step depth).
_DIMS_A = (0.10, 0.20)
_DIMS_B = (0.14, 0.24)
# terrain_width written into every synthetic stair conditioner.
_TRAJ_WIDTH = 0.8

_NAN = float("nan")


def _dims3(hl: tuple[float, float], width: float = _NAN) -> list[float]:
    """Terrain-dim row ``[width, height, length]`` for a (height, length)
    pair — conditioner column ordering, width unconstrained by default."""
    return [width, hl[0], hl[1]]


@pytest.fixture(scope="module")
def multiskill_with_stairs(tmp_path_factory) -> MultiSkillManager:
    """Two-skill manager: flat 'walk' + 'stairs' with two stair geometries.

    Built from the walking test YAML with the conditioner rewritten to the
    new dict format.  The stairs skill covers two velocities per geometry so
    the tests can check velocity-nearest within a dim-matched subset.
    """
    import yaml

    root = tmp_path_factory.mktemp("terrain_dim_library")
    with open(WALKING_YAML, "r") as f:
        base = yaml.safe_load(f)

    walk_dir = root / "walk"
    stairs_dir = root / "stairs"
    walk_dir.mkdir()
    stairs_dir.mkdir()

    def _write(folder, name, conditioner):
        data = dict(base)
        data["conditioner"] = conditioner
        with open(folder / name, "w") as f:
            yaml.safe_dump(data, f)

    _write(walk_dir, "walk_50.yaml", {"terrain": "flat", "vel_x": 0.5, "vel_y": 0.0, "vel_yaw": 0.0})
    _write(walk_dir, "walk_100.yaml", {"terrain": "flat", "vel_x": 1.0, "vel_y": 0.0, "vel_yaw": 0.0})
    for (h, l) in (_DIMS_A, _DIMS_B):
        for vx in (0.4, 0.8):
            _write(
                stairs_dir,
                f"stairs_h{h}_l{l}_v{vx}.yaml",
                {
                    "terrain": "stairs",
                    "terrain_height": h,
                    "terrain_length": l,
                    "terrain_width": _TRAJ_WIDTH,
                    "vel_x": vx,
                    "vel_y": 0.0,
                    "vel_yaw": 0.0,
                },
            )

    return MultiSkillManager(path=str(root), device=DEVICE)


@pytest.fixture(scope="module")
def multiskill_range_stairs(tmp_path_factory) -> MultiSkillManager:
    """Manager whose stair skill declares terrain dims as ``{min, max}`` ranges.

    ``terrain_height`` is a closed range (a *don't-care* dim: the trajectory
    works for any step rise in ``[0.10, 0.14]``), ``terrain_width`` an open
    range (``>= 1.0``), and ``terrain_length`` a fixed value.  Exercises the
    range-aware terrain-dim filter + bounds tensors.
    """
    import yaml

    root = tmp_path_factory.mktemp("range_stair_library")
    with open(WALKING_YAML, "r") as f:
        base = yaml.safe_load(f)
    walk_dir = root / "walk"
    stairs_dir = root / "stairs"
    walk_dir.mkdir()
    stairs_dir.mkdir()

    def _write(folder, name, conditioner):
        data = dict(base)
        data["conditioner"] = conditioner
        with open(folder / name, "w") as f:
            yaml.safe_dump(data, f)

    _write(walk_dir, "walk.yaml", {"terrain": "flat", "vel_x": 0.5, "vel_y": 0.0, "vel_yaw": 0.0})
    for vx in (0.4, 0.8):
        _write(stairs_dir, f"stairs_v{vx}.yaml", {
            "terrain": "stairs",
            "terrain_height": {"min": 0.10, "max": 0.14},
            "terrain_length": 0.20,
            "terrain_width": {"min": 1.0},
            "vel_x": vx, "vel_y": 0.0, "vel_yaw": 0.0,
        })

    return MultiSkillManager(path=str(root), device=DEVICE)


def _make_dim_lookup(dims_per_env: torch.Tensor, valid_per_env: torch.Tensor):
    """Fake ``terrain.terrain_dims_at`` returning fixed per-env values."""

    def lookup(xy_w: torch.Tensor):
        assert xy_w.shape[-1] == 2
        return dims_per_env, valid_per_env

    return lookup


class TestTerrainDimRanges:
    """Range-valued terrain dims: bounds tensors + range-containment filter."""

    def _stair_block(self, msm: MultiSkillManager) -> slice:
        idx = msm.skill_name_to_idx["stairs"]
        off = msm._skill_traj_offset[idx]
        cnt = msm.skills[idx].num_trajectories
        return slice(off, off + cnt)

    def test_bounds_tensors_reflect_ranges(self, multiskill_range_stairs):
        msm = multiskill_range_stairs
        sl = self._stair_block(msm)
        lo, hi = msm._terrain_dim_lo[sl], msm._terrain_dim_hi[sl]
        cnt = lo.shape[0]
        # width: open range [1.0, +inf); height: [0.10, 0.14]; length fixed 0.20.
        assert torch.allclose(lo[:, 0], torch.full((cnt,), 1.0), atol=1e-6)
        assert torch.isinf(hi[:, 0]).all()
        assert torch.allclose(lo[:, 1], torch.full((cnt,), 0.10), atol=1e-6)
        assert torch.allclose(hi[:, 1], torch.full((cnt,), 0.14), atol=1e-6)
        assert torch.allclose(lo[:, 2], torch.full((cnt,), 0.20), atol=1e-6)
        assert torch.allclose(hi[:, 2], torch.full((cnt,), 0.20), atol=1e-6)

    def test_range_dim_matches_in_range_feature(self, multiskill_range_stairs):
        """A range height accepts any in-range feature height, rejects outside."""
        msm = multiskill_range_stairs
        stairs_idx = msm.skill_name_to_idx["stairs"]
        sl = self._stair_block(msm)
        skill_ids = torch.full((3,), stairs_idx, dtype=torch.long)
        # feature (height, length) below / inside / above the [0.10, 0.14] range.
        dims = torch.tensor(
            [_dims3((0.08, 0.20)), _dims3((0.12, 0.20)), _dims3((0.20, 0.20))],
            dtype=torch.float,
        )
        valid = torch.tensor([True, True, True])
        msm.set_terrain_dim_lookup(_make_dim_lookup(dims, valid))
        try:
            mask = msm._terrain_dim_allowed(torch.zeros(3, 2), skill_ids)
            assert not mask[0, sl].any(), "feature height below the range must be excluded"
            assert mask[1, sl].all(), "in-range feature height must match every stair traj"
            assert not mask[2, sl].any(), "feature height above the range must be excluded"
        finally:
            msm.set_terrain_dim_lookup(None)


class TestTerrainDimFilter:
    def _skill_indices(self, msm: MultiSkillManager) -> tuple[int, int]:
        return msm.skill_name_to_idx["stairs"], msm.skill_name_to_idx["walk"]

    def _traj_hl(self, msm: MultiSkillManager) -> torch.Tensor:
        """Per-trajectory (terrain_height, terrain_length) columns."""
        return msm._global_conditioning[:, 4:6]

    def test_skill_has_terrain(self, multiskill_with_stairs):
        msm = multiskill_with_stairs
        stairs_idx, walk_idx = self._skill_indices(msm)
        assert msm._skill_has_terrain[stairs_idx].item() is True
        assert msm._skill_has_terrain[walk_idx].item() is False

    def test_dim_mask_filters_terrain_trajectories(self, multiskill_with_stairs):
        """Terrain-skill envs only pass trajectories matching their feature
        dims; flat envs and invalid (feature-less) envs pass everything."""
        msm = multiskill_with_stairs
        stairs_idx, walk_idx = self._skill_indices(msm)

        # env 0: stairs on geometry A; env 1: walk; env 2: stairs, no feature.
        skill_ids = torch.tensor([stairs_idx, walk_idx, stairs_idx], dtype=torch.long)
        dims = torch.tensor([_dims3(_DIMS_A)] * 3, dtype=torch.float)
        valid = torch.tensor([True, True, False])
        msm.set_terrain_dim_lookup(_make_dim_lookup(dims, valid))
        try:
            anchor = torch.zeros(3, 2)
            mask = msm._terrain_dim_allowed(anchor, skill_ids)
            traj_hl = self._traj_hl(msm)
            expected_env0 = (
                (traj_hl[:, 0] - _DIMS_A[0]).abs().le(0.005)
                & (traj_hl[:, 1] - _DIMS_A[1]).abs().le(0.010)
            )
            assert torch.equal(mask[0], expected_env0)
            assert mask[1].all(), "flat-skill env must pass every trajectory"
            assert mask[2].all(), "feature-less env must pass every trajectory"
        finally:
            msm.set_terrain_dim_lookup(None)

    def test_nan_dims_are_unconstrained_but_real_dims_match(self, multiskill_with_stairs):
        """A NaN width never excludes; a non-NaN width participates in the
        match like any other dimension."""
        msm = multiskill_with_stairs
        stairs_idx, _ = self._skill_indices(msm)
        skill_ids = torch.full((3,), stairs_idx, dtype=torch.long)
        dims = torch.tensor(
            [
                _dims3(_DIMS_A),                    # NaN width: unconstrained
                _dims3(_DIMS_A, width=_TRAJ_WIDTH),  # matching width
                _dims3(_DIMS_A, width=0.5),          # mismatched width
            ],
            dtype=torch.float,
        )
        valid = torch.tensor([True, True, True])
        msm.set_terrain_dim_lookup(_make_dim_lookup(dims, valid))
        try:
            mask = msm._terrain_dim_allowed(torch.zeros(3, 2), skill_ids)
            assert torch.equal(mask[0], mask[1]), (
                "NaN width and exactly-matching width must pass the same set"
            )
            assert mask[0].any()
            assert not mask[2].any(), "mismatched width must exclude all"
        finally:
            msm.set_terrain_dim_lookup(None)

    def test_selection_respects_feature_dims(self, multiskill_with_stairs):
        """Two stair envs with the same velocity but different feature dims
        must select trajectories of their own geometry."""
        msm = multiskill_with_stairs
        stairs_idx, _ = self._skill_indices(msm)
        N = 2
        skill_ids = torch.full((N,), stairs_idx, dtype=torch.long)
        ref_poses = torch.zeros(N, 7)
        _attach_stub_owner(msm, skill_ids, ref_poses=ref_poses)

        dims = torch.tensor([_dims3(_DIMS_A), _dims3(_DIMS_B)], dtype=torch.float)
        valid = torch.tensor([True, True])
        msm.set_terrain_dim_lookup(_make_dim_lookup(dims, valid))
        try:
            cond = torch.zeros(N, 6)
            cond[:, 0] = 0.8  # both envs command the same vel_x
            selected = msm._select_trajectories(cond)
            sel_hl = self._traj_hl(msm)[selected]
            assert torch.allclose(sel_hl[0], torch.tensor(_DIMS_A), atol=1e-4)
            assert torch.allclose(sel_hl[1], torch.tensor(_DIMS_B), atol=1e-4)
            # And velocity-nearest within the matched geometry.
            sel_vels = msm._global_conditioning[selected, 0]
            assert torch.allclose(sel_vels, torch.full((N,), 0.8), atol=1e-4)
        finally:
            msm.set_terrain_dim_lookup(None)

    def test_unmatched_dims_fall_back_to_skill_only(self, multiskill_with_stairs):
        """A feature geometry missing from the library must not produce a
        wrong-skill pick — selection falls back to the skill mask."""
        msm = multiskill_with_stairs
        stairs_idx, _ = self._skill_indices(msm)
        skill_ids = torch.tensor([stairs_idx], dtype=torch.long)
        _attach_stub_owner(msm, skill_ids, ref_poses=torch.zeros(1, 7))

        dims = torch.tensor([_dims3((0.99, 0.99))], dtype=torch.float)  # not in library
        msm.set_terrain_dim_lookup(_make_dim_lookup(dims, torch.tensor([True])))
        try:
            cond = torch.zeros(1, 6)
            cond[0, 0] = 0.4
            selected = msm._select_trajectories(cond)
            assert msm._traj_skill_idx[selected].item() == stairs_idx
        finally:
            msm.set_terrain_dim_lookup(None)

    def test_no_lookup_means_no_dim_filter(self, multiskill_with_stairs):
        """Without a registered lookup the dim path must not run (and must
        not touch ``ref_poses``, which offline owners may not define)."""
        msm = multiskill_with_stairs
        stairs_idx, _ = self._skill_indices(msm)
        skill_ids = torch.tensor([stairs_idx], dtype=torch.long)
        _attach_stub_owner(msm, skill_ids)  # no ref_poses on purpose

        assert msm._terrain_dims_at is None
        cond = torch.zeros(1, 6)
        cond[0, 0] = 0.8
        selected = msm._select_trajectories(cond)
        # Velocity-nearest within the skill, regardless of geometry.
        assert msm._traj_skill_idx[selected].item() == stairs_idx
        assert torch.allclose(
            msm._global_conditioning[selected, 0], torch.tensor([0.8]), atol=1e-4
        )

    def test_nearest_traj_in_skill_uses_anchor(self, multiskill_with_stairs):
        """``nearest_traj_in_skill`` applies the dim filter with an explicit
        anchor (the approach-planner path)."""
        msm = multiskill_with_stairs
        stairs_idx, _ = self._skill_indices(msm)
        N = 1
        _attach_stub_owner(
            msm, torch.full((N,), stairs_idx, dtype=torch.long),
            ref_poses=torch.zeros(N, 7),
        )

        class _FakeCondTerm:
            command = torch.tensor([[0.4, 0.0, 0.0]])

        class _FakeCmdMgr:
            def get_term(self, name):
                return _FakeCondTerm()

        class _FakeEnv:
            command_manager = _FakeCmdMgr()

        msm.env = _FakeEnv()
        dims = torch.tensor([_dims3(_DIMS_B)], dtype=torch.float)
        msm.set_terrain_dim_lookup(_make_dim_lookup(dims, torch.tensor([True])))
        try:
            owner_ids = torch.tensor([stairs_idx], dtype=torch.long)
            env_ids = torch.tensor([0], dtype=torch.long)
            sel = msm.nearest_traj_in_skill(owner_ids, env_ids, anchor_xy=torch.zeros(1, 2))
            sel_hl = self._traj_hl(msm)[sel]
            assert torch.allclose(sel_hl[0], torch.tensor(_DIMS_B), atol=1e-4)
            assert torch.allclose(
                msm._global_conditioning[sel, 0], torch.tensor([0.4]), atol=1e-4
            )
        finally:
            msm.set_terrain_dim_lookup(None)
            msm.env = None


# ---------------------------------------------------------------------------
# Per-skill segmented selection equivalence
# ---------------------------------------------------------------------------

def _make_indexed_dim_lookup(dims_table: torch.Tensor, valid_table: torch.Tensor):
    """Position-keyed fake ``terrain.terrain_dims_at``.

    Encodes each env's id in its anchor x coordinate (set ``ref_poses[i, 0]
    == i``) so the lookup returns the correct per-env dims whether it's called
    with the full batch (legacy path) or a per-skill row subset (the segmented
    path).  A plain fixed-size lookup can't serve both, since the segmented
    path queries only the rows of one skill at a time.
    """

    def lookup(xy_w: torch.Tensor):
        idx = xy_w[:, 0].round().long()
        return dims_table[idx], valid_table[idx]

    return lookup


class TestSelectionEquivalence:
    """``_select_within_skills`` must match the legacy full-library path.

    The per-skill segmented selection (used by ``_select_trajectories`` and
    ``nearest_traj_in_skill``) must return byte-identical global indices to
    ``_select_trajectories_reference``.  Identical indices are what keep the
    manager's ``_traj_changed`` signal — and therefore the contact-gate
    re-arm / grace-timer logic in the command term — behaving exactly as
    before the optimization.
    """

    def _idx(self, msm: MultiSkillManager) -> tuple[int, int]:
        return msm.skill_name_to_idx["stairs"], msm.skill_name_to_idx["walk"]

    def _assert_equiv(self, msm, cond, env_ids=None):
        new = msm._select_trajectories(cond, env_ids)
        ref = msm._select_trajectories_reference(cond, env_ids)
        assert torch.equal(new, ref), (
            f"selection mismatch:\n new={new.tolist()}\n ref={ref.tolist()}"
        )

    def test_equiv_flat_no_lookup(self, multiskill_with_stairs):
        """No terrain lookup: pure velocity-nearest within each env's skill."""
        msm = multiskill_with_stairs
        stairs_idx, walk_idx = self._idx(msm)
        msm.set_terrain_dim_lookup(None)
        torch.manual_seed(0)
        N = 50
        pick = torch.randint(0, 2, (N,))
        skill_ids = torch.where(
            pick.bool(),
            torch.full((N,), walk_idx),
            torch.full((N,), stairs_idx),
        ).long()
        _attach_stub_owner(msm, skill_ids, ref_poses=torch.zeros(N, 7))
        cond = torch.randn(N, 6)
        self._assert_equiv(msm, cond)

    def test_equiv_terrain_mixed_valid(self, multiskill_with_stairs):
        """Terrain lookup with stairs-on-feature, walk, and feature-less envs."""
        msm = multiskill_with_stairs
        stairs_idx, walk_idx = self._idx(msm)
        skill_ids = torch.tensor(
            [stairs_idx, walk_idx, stairs_idx, stairs_idx]
        ).long()
        N = skill_ids.numel()
        ref_poses = torch.zeros(N, 7)
        ref_poses[:, 0] = torch.arange(N, dtype=torch.float)
        _attach_stub_owner(msm, skill_ids, ref_poses=ref_poses)

        dims = torch.tensor(
            [_dims3(_DIMS_A), _dims3(_DIMS_A), _dims3(_DIMS_A), _dims3(_DIMS_B)],
            dtype=torch.float,
        )
        valid = torch.tensor([True, True, False, True])  # env 2 = no feature
        msm.set_terrain_dim_lookup(_make_indexed_dim_lookup(dims, valid))
        try:
            cond = torch.zeros(N, 6)
            cond[:, 0] = torch.tensor([0.8, 0.7, 0.4, 0.5])
            self._assert_equiv(msm, cond)
        finally:
            msm.set_terrain_dim_lookup(None)

    def test_equiv_empty_match_fallback(self, multiskill_with_stairs):
        """A commanded geometry absent from the library must fall back to
        skill-only in both paths (identically)."""
        msm = multiskill_with_stairs
        stairs_idx, walk_idx = self._idx(msm)
        skill_ids = torch.tensor([stairs_idx, stairs_idx, walk_idx]).long()
        N = skill_ids.numel()
        ref_poses = torch.zeros(N, 7)
        ref_poses[:, 0] = torch.arange(N, dtype=torch.float)
        _attach_stub_owner(msm, skill_ids, ref_poses=ref_poses)

        dims = torch.tensor(
            [_dims3((0.99, 0.99)), _dims3(_DIMS_B), _dims3(_DIMS_A)],
            dtype=torch.float,
        )  # env 0 geometry not in library
        valid = torch.tensor([True, True, True])
        msm.set_terrain_dim_lookup(_make_indexed_dim_lookup(dims, valid))
        try:
            cond = torch.zeros(N, 6)
            cond[:, 0] = torch.tensor([0.4, 0.8, 0.6])
            self._assert_equiv(msm, cond)
        finally:
            msm.set_terrain_dim_lookup(None)

    def test_equiv_with_forced_latch(self, multiskill_with_stairs):
        """The forced-traj latch overlay is applied identically by both paths."""
        from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.skill_state import (
            ForcedTrajLatch,
        )

        msm = multiskill_with_stairs
        stairs_idx, walk_idx = self._idx(msm)
        msm.set_terrain_dim_lookup(None)
        N = 6
        skill_ids = torch.tensor(
            [stairs_idx, walk_idx, stairs_idx, walk_idx, stairs_idx, walk_idx]
        ).long()
        _attach_stub_owner(msm, skill_ids, ref_poses=torch.zeros(N, 7))

        latch = ForcedTrajLatch(N, device=DEVICE)
        latch.set(
            torch.tensor([0, 3], device=DEVICE),
            torch.tensor([2, 1], device=DEVICE),
        )
        msm.set_forced_traj_latch(latch)
        try:
            torch.manual_seed(2)
            cond = torch.randn(N, 6)
            self._assert_equiv(msm, cond)
        finally:
            msm.set_forced_traj_latch(None)

    def test_equiv_env_ids_subset(self, multiskill_with_stairs):
        """The env-subset selection path (e.g. reset_on_reference) matches too."""
        msm = multiskill_with_stairs
        stairs_idx, walk_idx = self._idx(msm)
        N = 8
        torch.manual_seed(1)
        pick = torch.randint(0, 2, (N,))
        skill_ids = torch.where(
            pick.bool(),
            torch.full((N,), walk_idx),
            torch.full((N,), stairs_idx),
        ).long()
        ref_poses = torch.zeros(N, 7)
        ref_poses[:, 0] = torch.arange(N, dtype=torch.float)
        _attach_stub_owner(msm, skill_ids, ref_poses=ref_poses)

        dims = torch.stack(
            [torch.tensor(_dims3(_DIMS_A), dtype=torch.float) for _ in range(N)]
        )
        valid = torch.ones(N, dtype=torch.bool)
        msm.set_terrain_dim_lookup(_make_indexed_dim_lookup(dims, valid))
        try:
            cond_full = torch.randn(N, 6)
            env_ids = torch.tensor([1, 3, 5, 7])
            self._assert_equiv(msm, cond_full[env_ids], env_ids)
        finally:
            msm.set_terrain_dim_lookup(None)


# ---------------------------------------------------------------------------
# Merged-tensor startup cache
# ---------------------------------------------------------------------------

class TestTensorCache:
    """The startup cache must round-trip to byte-identical state and rebuild
    whenever the library's YAMLs change."""

    def _build_small_library(self, root) -> "Path":
        """Copy the merged test library into a single 'walk' skill folder."""
        import shutil
        from pathlib import Path

        root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(MERGED_LIBRARY_DIR, str(root / "walk"))
        return Path(root)

    def test_cache_roundtrip_matches_fresh(self, tmp_path, monkeypatch):
        """A cache-loaded manager equals a freshly-built one, tensor for tensor."""
        monkeypatch.setenv("ROBOT_RL_TRAJ_CACHE_DIR", str(tmp_path / "cache"))
        lib = self._build_small_library(tmp_path / "lib")

        fresh = MultiSkillManager(path=str(lib), device=DEVICE)   # miss -> builds + writes
        cached = MultiSkillManager(path=str(lib), device=DEVICE)  # hit  -> restores

        assert set(fresh.data.keys()) == set(cached.data.keys())
        for k in fresh.data.keys():
            assert torch.equal(fresh.data[k], cached.data[k]), f"data[{k}] mismatch"

        assert fresh.num_trajectories == cached.num_trajectories
        assert fresh.max_expanded_domains == cached.max_expanded_domains
        assert fresh.trajectory_names == cached.trajectory_names
        assert fresh.skill_labels == cached.skill_labels
        assert fresh.pos_output_names == cached.pos_output_names
        assert fresh.vel_output_names == cached.vel_output_names
        assert fresh.spline_order == cached.spline_order
        assert fresh.ref_frames == cached.ref_frames

        # Derived state rebuilt from the restored tensors must also match.
        assert torch.equal(fresh._global_conditioning, cached._global_conditioning)
        assert torch.equal(fresh._cumsum_T, cached._cumsum_T)
        assert torch.equal(fresh._total_original_T, cached._total_original_T)
        assert [s.name for s in fresh.skills] == [s.name for s in cached.skills]
        assert fresh._skill_traj_offset == cached._skill_traj_offset
        assert fresh._skill_traj_count == cached._skill_traj_count
        assert fresh._contact_bodies_per_domain == cached._contact_bodies_per_domain
        assert torch.equal(fresh._gate_phi_table, cached._gate_phi_table)
        assert torch.equal(fresh._num_gates_per_traj, cached._num_gates_per_traj)

    def test_second_build_uses_cache_not_yaml(self, tmp_path, monkeypatch):
        """On a cache hit the YAML build path must not run at all."""
        monkeypatch.setenv("ROBOT_RL_TRAJ_CACHE_DIR", str(tmp_path / "cache"))
        lib = self._build_small_library(tmp_path / "lib")
        first = MultiSkillManager(path=str(lib), device=DEVICE)

        def _boom(self, *args, **kwargs):
            raise AssertionError("_build_from_yaml ran on a cache hit")

        monkeypatch.setattr(MultiSkillManager, "_build_from_yaml", _boom)
        cached = MultiSkillManager(path=str(lib), device=DEVICE)
        assert cached.num_trajectories == first.num_trajectories

    def test_cache_invalidated_on_yaml_change(self, tmp_path, monkeypatch):
        """Editing a YAML must invalidate the cache (no stale data served)."""
        import yaml

        monkeypatch.setenv("ROBOT_RL_TRAJ_CACHE_DIR", str(tmp_path / "cache"))
        walk = tmp_path / "lib" / "walk"
        walk.mkdir(parents=True)
        with open(WALKING_YAML, "r") as f:
            base = yaml.safe_load(f)

        base["conditioner"] = {"terrain": "flat", "vel_x": 0.5, "vel_y": 0.0, "vel_yaw": 0.0}
        with open(walk / "w.yaml", "w") as f:
            yaml.safe_dump(base, f)
        first = MultiSkillManager(path=str(tmp_path / "lib"), device=DEVICE)
        assert torch.allclose(
            first._global_conditioning[0, 0], torch.tensor(0.5), atol=1e-6
        )

        # Different value AND different serialized length -> new size/mtime ->
        # new key, robust even on coarse-mtime filesystems.
        base["conditioner"]["vel_x"] = 1.25
        with open(walk / "w.yaml", "w") as f:
            yaml.safe_dump(base, f)
        second = MultiSkillManager(path=str(tmp_path / "lib"), device=DEVICE)
        assert torch.allclose(
            second._global_conditioning[0, 0], torch.tensor(1.25), atol=1e-6
        )

    def test_use_cache_false_writes_nothing(self, tmp_path, monkeypatch):
        """``use_cache=False`` neither reads nor writes the cache."""
        cache = tmp_path / "cache"
        monkeypatch.setenv("ROBOT_RL_TRAJ_CACHE_DIR", str(cache))
        lib = self._build_small_library(tmp_path / "lib")
        MultiSkillManager(path=str(lib), device=DEVICE, use_cache=False)
        assert not cache.exists() or not list(cache.glob("*.pt"))
