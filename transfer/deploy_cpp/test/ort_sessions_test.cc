// Copyright 2026 Zachary Olkin. All rights reserved.

// The ONNX session layer against the REAL policy pair.
//
// Needs both artifacts and (for the placement check) a GPU, so it skips cleanly without
// them -- the same idiom test_hardware_offline.py uses. Point it at a run directory with
// G1_POLICY_DIR; the default is the prefetched pair the obelisk stack uses.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <sys/stat.h>

#include "g1_deploy/fm_session.h"
#include "g1_deploy/gen_tracker_core.h"
#include "g1_deploy/tracker_session.h"

namespace g1_deploy {
namespace {

bool Exists(const std::string& p) {
  struct stat st;
  return ::stat(p.c_str(), &st) == 0;
}

std::string PolicyDir() {
  if (const char* env = std::getenv("G1_POLICY_DIR")) return env;
  return G1_DEPLOY_POLICY_DIR;
}

std::vector<Provider> RequestedProviders() {
  // CPU by default so the test is runnable anywhere; the GPU path is opt-in, matching the
  // deploy runtime's own rule that nothing silently starts using an accelerator.
  const char* env = std::getenv("G1_ORT_PROVIDER");
  const std::string name = env ? env : "cpu";
  return {ProviderFromName(name)};
}

class RealArtifacts : public ::testing::Test {
 protected:
  void SetUp() override {
    tracker_path = PolicyDir() + "/tracker.onnx";
    fm_path = PolicyDir() + "/fm.onnx";
    if (!Exists(tracker_path)) {
      GTEST_SKIP() << "no tracker.onnx under " << PolicyDir()
                   << "; set G1_POLICY_DIR or run scripts/prefetch_policy.py";
    }
  }
  std::string tracker_path;
  std::string fm_path;
};

TEST_F(RealArtifacts, TrackerLoadsAndItsGraphAgreesWithItsMetadata) {
  TrackerSession trk(tracker_path, RequestedProviders(), 2);
  const PolicyParams& p = trk.params();
  // The constructor already refuses a graph whose input width disagrees with num_obs; this
  // records the numbers so a failure says which artifact was loaded.
  EXPECT_EQ(p.num_obs, p.TotalObsWidth());
  EXPECT_GT(p.num_actions, 0);
  EXPECT_EQ(static_cast<int>(p.joint_names_isaac.size()), p.num_actions);
  std::cout << "    [tracker] num_obs=" << p.num_obs << " num_actions=" << p.num_actions
            << " dt=" << p.dt << " terms=" << p.obs_terms.size() << "\n";
}

TEST_F(RealArtifacts, TrackerInferenceIsFiniteAndDeterministic) {
  TrackerSession trk(tracker_path, RequestedProviders(), 2);
  const int n_obs = trk.params().num_obs;
  const int n_act = trk.params().num_actions;
  std::vector<float> obs(static_cast<size_t>(n_obs), 0.0f);
  std::vector<float> a1(static_cast<size_t>(n_act), 0.0f);
  std::vector<float> a2(static_cast<size_t>(n_act), 0.0f);

  trk.Infer(obs.data(), a1.data());
  trk.Infer(obs.data(), a2.data());
  for (int i = 0; i < n_act; ++i) {
    EXPECT_TRUE(std::isfinite(a1[i])) << "action " << i << " is not finite";
    // Same input, same output: the tracker is a plain MLP with no state of its own. If this
    // ever fails, something is carrying state across calls that should not be.
    EXPECT_FLOAT_EQ(a1[i], a2[i]) << "action " << i << " is not deterministic";
  }
}

TEST_F(RealArtifacts, GeneratorMetadataAgreesWithTheTrackersWindow) {
  if (!Exists(fm_path)) GTEST_SKIP() << "no fm.onnx under " << PolicyDir();
  TrackerSession trk(tracker_path, RequestedProviders(), 2);
  FmSession fm(fm_path, RequestedProviders(), 2, std::make_shared<ZeroNoise>());

  ASSERT_TRUE(trk.params().reference_window.has_value());
  const auto& rw = *trk.params().reference_window;
  // The generator's output IS this observation. Same blocks, same order.
  AssertWindowBlocksAgree(rw.blocks, fm.meta().window_blocks);

  int node_width = 0;
  for (const auto& b : rw.blocks) node_width += b.second;
  EXPECT_EQ(fm.meta().d, node_width) << "node width disagrees between the two artifacts";

  // F >= max_offset + K, the horizon rule, on the real numbers.
  int max_offset = 0;
  for (int o : rw.offsets) max_offset = std::max(max_offset, o);
  ASSERT_TRUE(trk.params().fm_update_every.has_value());
  const int k = *trk.params().fm_update_every;
  EXPECT_GE(fm.meta().f, max_offset + k)
      << "F=" << fm.meta().f << " cannot cover offset " << max_offset << " at K=" << k;
  std::cout << "    [pair] N=" << fm.meta().n << " F=" << fm.meta().f << " D=" << fm.meta().d
            << " K=" << k << " max_offset=" << max_offset
            << " extero=" << fm.meta().extero_source << "(" << fm.meta().extero_dim() << ")"
            << " history_null=" << fm.meta().has_history_null << "\n";
}

TEST_F(RealArtifacts, GeneratorProducesAFiniteWindowAndIsDeterministicUnderZeroNoise) {
  if (!Exists(fm_path)) GTEST_SKIP() << "no fm.onnx under " << PolicyDir();
  FmSession fm(fm_path, RequestedProviders(), 2, std::make_shared<ZeroNoise>());

  FmConditioning cond;
  cond.extero.assign(static_cast<size_t>(fm.meta().extero_dim()), 0.0);
  cond.future_cmd.assign(static_cast<size_t>(fm.meta().f) * fm.meta().cmd_dim, 0.0);
  fm.SetConditioning(cond);

  FmRequest req;
  req.target_ctrl_step = 7;
  req.history_nodes.assign(static_cast<size_t>(fm.meta().n) * fm.meta().d, 0.0);
  req.history_cmd.assign(static_cast<size_t>(fm.meta().n) * fm.meta().cmd_dim, 0.0);
  req.history_null = true;

  std::vector<double> a(static_cast<size_t>(fm.meta().f) * fm.meta().d, 0.0);
  std::vector<double> b(a.size(), 0.0);
  int64_t target = 0;

  fm.Submit(req);
  ASSERT_TRUE(fm.TryCollect(a.data(), &target));
  EXPECT_EQ(target, 7) << "the plan must carry the boundary it was asked for";
  EXPECT_FALSE(fm.TryCollect(b.data(), &target)) << "a collected plan must not be handed out twice";

  fm.Submit(req);
  ASSERT_TRUE(fm.TryCollect(b.data(), &target));
  for (size_t i = 0; i < a.size(); ++i) {
    ASSERT_TRUE(std::isfinite(a[i])) << "node element " << i << " is not finite";
    // Zero noise means one fixed trajectory per conditioning, which is what makes a re-plan
    // discontinuity attributable to the conditioning rather than to a fresh draw.
    EXPECT_DOUBLE_EQ(a[i], b[i]) << "element " << i << " differs under zero noise";
  }
  std::cout << "    [fm] infer " << fm.last_infer_ms() << " ms\n";
}

// ---------------------------------------------------------------------------------------
// CUDA Graph capture
// ---------------------------------------------------------------------------------------
// CUDA-only by construction, so these skip unless the suite was pointed at a GPU with
// G1_ORT_PROVIDER=cuda. What they check is the one thing that fails SILENTLY: a captured
// graph holds device pointers, so if the per-call host-to-device copy is missed the replay
// returns the FIRST call's answer forever, bit-identical, with no error. That is
// indistinguishable from a healthy generator whose conditioning happens not to have moved --
// which on a walking robot is exactly the situation.

/// A conditioning set that depends on `seed`, so two of them must give different plans.
FmConditioning MakeConditioning(const FmMeta& meta, int seed) {
  FmConditioning cond;
  cond.extero.assign(static_cast<size_t>(meta.extero_dim()), 0.0);
  cond.future_cmd.assign(static_cast<size_t>(meta.f) * meta.cmd_dim, 0.0);
  for (size_t i = 0; i < cond.extero.size(); ++i) {
    cond.extero[i] = 0.05 * static_cast<double>((i * 7 + seed * 13) % 20) - 0.5;
  }
  for (size_t i = 0; i < cond.future_cmd.size(); ++i) {
    cond.future_cmd[i] = 0.1 * static_cast<double>((i + seed * 3) % 5);
  }
  return cond;
}

/// One plan for `seed`'s conditioning.
std::vector<double> PlanFor(FmSession& fm, int seed) {
  fm.SetConditioning(MakeConditioning(fm.meta(), seed));
  FmRequest req;
  req.target_ctrl_step = 1;
  req.history_nodes.assign(static_cast<size_t>(fm.meta().n) * fm.meta().d, 0.0);
  req.history_cmd.assign(static_cast<size_t>(fm.meta().n) * fm.meta().cmd_dim, 0.0);
  req.history_null = true;
  for (size_t i = 0; i < req.history_nodes.size(); ++i) {
    req.history_nodes[i] = 0.01 * static_cast<double>((i + seed) % 11);
  }
  std::vector<double> out(static_cast<size_t>(fm.meta().f) * fm.meta().d, 0.0);
  int64_t target = 0;
  fm.Submit(req);
  EXPECT_TRUE(fm.TryCollect(out.data(), &target));
  return out;
}

double MaxAbsDiff(const std::vector<double>& a, const std::vector<double>& b) {
  double worst = 0.0;
  for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::fabs(a[i] - b[i]));
  return worst;
}

TEST_F(RealArtifacts, CudaGraphReplayTracksItsInputRatherThanRepeatingTheFirstPlan) {
  if (!Exists(fm_path)) GTEST_SKIP() << "no fm.onnx under " << PolicyDir();
  const std::vector<Provider> provs = RequestedProviders();
  if (provs.empty() || provs.front() != Provider::kCuda) {
    GTEST_SKIP() << "capture is a CUDA feature; run with G1_ORT_PROVIDER=cuda";
  }
  FmSession plain(fm_path, provs, 2, std::make_shared<ZeroNoise>(), /*cuda_graph=*/false);
  FmSession captured(fm_path, provs, 2, std::make_shared<ZeroNoise>(), /*cuda_graph=*/true);
  captured.model().WarmUpAndCheckPlacement();   // this is what records the graph
  ASSERT_TRUE(captured.model().cuda_graph());

  const std::vector<double> want_a = PlanFor(plain, 1);
  const std::vector<double> want_b = PlanFor(plain, 2);
  // The premise: the two conditionings really do give different plans, so "replayed the old
  // answer" and "computed the new one" are distinguishable at all.
  ASSERT_GT(MaxAbsDiff(want_a, want_b), 1e-3) << "the two conditionings must differ";

  const std::vector<double> got_a = PlanFor(captured, 1);
  const std::vector<double> got_b = PlanFor(captured, 2);
  // Replay must be EXACT against the plain session: same kernels in the same order, only
  // submitted differently. Anything else means capture changed the arithmetic.
  EXPECT_EQ(MaxAbsDiff(got_a, want_a), 0.0) << "capture must not change the result";
  EXPECT_EQ(MaxAbsDiff(got_b, want_b), 0.0)
      << "the replay returned something other than the plan for its own conditioning";
  // The footgun, stated positively: got_b must NOT be the first plan.
  EXPECT_GT(MaxAbsDiff(got_b, want_a), 1e-3)
      << "the replay reproduced the FIRST plan, so the host-to-device copy is not reaching the "
         "captured device buffers -- every later plan would be stale and nothing would error";
  std::cout << "    [fm] plain " << plain.last_infer_ms() << " ms, captured "
            << captured.last_infer_ms() << " ms\n";
}

TEST_F(RealArtifacts, CudaGraphCaptureIsRefusedOnTheCpu) {
  if (!Exists(fm_path)) GTEST_SKIP() << "no fm.onnx under " << PolicyDir();
  // Capture without a CUDA provider is not a degraded mode, it is a contradiction, and saying
  // so at load is the difference between a clear error and a session that quietly never
  // captured.
  EXPECT_THROW(FmSession(fm_path, {Provider::kCpu}, 2, std::make_shared<ZeroNoise>(),
                         /*cuda_graph=*/true),
               std::runtime_error);
}

TEST_F(RealArtifacts, TheRealGeneratorDrivesTheRealScheduleEndToEnd) {
  // The whole stack below the ROS node: the scheduler, the history projection, the
  // re-anchor and a real fm.onnx, stepped for several cadence periods.
  if (!Exists(fm_path)) GTEST_SKIP() << "no fm.onnx under " << PolicyDir();
  TrackerSession trk(tracker_path, RequestedProviders(), 2);
  auto fm = std::make_shared<FmSession>(fm_path, RequestedProviders(), 2,
                                        std::make_shared<ZeroNoise>());

  const auto& rw = *trk.params().reference_window;
  FmConditioning cond;
  cond.extero.assign(static_cast<size_t>(fm->meta().extero_dim()), 0.0);
  cond.future_cmd.assign(static_cast<size_t>(fm->meta().f) * fm->meta().cmd_dim, 0.0);
  fm->SetConditioning(cond);

  CoreConfig cfg;
  cfg.num_joints = fm->meta().num_joints;
  cfg.num_ee_frames = 0;
  for (const auto& b : rw.blocks) cfg.blocks.push_back(BlockFromName(b.first));
  cfg.offsets = rw.offsets;
  cfg.history_length = fm->meta().n;
  cfg.cmd_dim = fm->meta().cmd_dim;
  cfg.schedule.replan_period = *trk.params().fm_update_every;
  cfg.schedule.horizon = fm->meta().f;
  cfg.schedule.max_offset = *std::max_element(rw.offsets.begin(), rw.offsets.end());
  cfg.schedule.lead_ticks = 3;

  GenTrackerCore core(cfg, fm);
  CoreState state;
  state.command = std::vector<double>(static_cast<size_t>(cfg.cmd_dim), 0.0);
  std::vector<double> window(static_cast<size_t>(cfg.offsets.size()) * fm->meta().d, 0.0);

  int adoptions = 0;
  for (int i = 0; i < 3 * cfg.schedule.replan_period; ++i) {
    // Walk forward a little each tick so the re-anchor has something real to correct.
    state.root_pos.x() += 0.01;
    const TickOutcome outcome = core.Step(state, window.data());
    if (outcome == TickOutcome::kAdoptedOnTime) ++adoptions;
    for (double v : window) ASSERT_TRUE(std::isfinite(v)) << "tick " << i;
  }
  EXPECT_GE(adoptions, 3) << "the schedule should have adopted a plan per cadence period";
  EXPECT_EQ(core.scheduler().late_events(), 0u)
      << "a synchronous sampler always meets its deadline, so nothing should be late";
  EXPECT_EQ(core.scheduler().offsets_clamped(), 0u);
}

}  // namespace
}  // namespace g1_deploy
