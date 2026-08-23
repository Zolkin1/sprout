// Copyright 2026 Zachary Olkin. All rights reserved.

// Parity: g1_deploy::Window (C++) vs transfer/deploy/window_np.py, through the recorded
// fixtures. Tolerances come from the manifest.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "fixture.h"
#include "g1_deploy/quat_math.h"
#include "g1_deploy/window.h"

namespace g1_deploy {
namespace {

using test::Blob;
using test::Case;
using test::Fixtures;

const Fixtures& Loaded() {
  static const Fixtures fixtures = Fixtures::Load(Fixtures::DefaultDir());
  return fixtures;
}

std::vector<const Case*> Cases(const std::string& fn) {
  auto cases = Loaded().ForFunction(fn);
  EXPECT_FALSE(cases.empty()) << "no fixture case for '" << fn << "'; regenerate the fixtures";
  return cases;
}

/// Worst elementwise difference between a computed buffer and the oracle's whole blob.
double WholeError(const std::vector<double>& got, const Blob& want) {
  double worst = 0.0;
  for (size_t i = 0; i < got.size(); ++i) worst = std::max(worst, std::abs(got[i] - want.data[i]));
  return worst;
}

void ReportMargin(const Case& c, double worst, size_t elems) {
  EXPECT_LE(worst, c.atol) << c.name << " (" << c.fn << "): differs by " << worst
                           << ", tolerance " << c.atol;
  std::cout << "    [margin] " << c.name << " worst=" << worst << " atol=" << c.atol
            << " elems=" << elems << "\n";
}

QuatXyzw QuatAt(const Blob& b, int i) {
  const double* r = b.row(i);
  return QuatXyzw(r[0], r[1], r[2], r[3]);
}

Vec3 Vec3At(const Blob& b, int i) {
  const double* r = b.row(i);
  return Vec3(r[0], r[1], r[2]);
}

/// The block selection a case was generated with, as the C++ enum.
std::vector<ReferenceBlock> BlocksOf(const Case& c) {
  auto it = c.string_lists.find("blocks");
  EXPECT_NE(it, c.string_lists.end()) << c.name << ": fixture carries no block list";
  std::vector<ReferenceBlock> out;
  for (const std::string& name : it->second) out.push_back(BlockFromName(name));
  return out;
}

TEST(WindowParity, MatrixToRot6d) {
  for (const Case* c : Cases("matrix_to_rot6d")) {
    const Blob& in = c->In("matrix");
    const Blob& want = c->Out("result");
    std::vector<double> got(want.data.size());
    for (int i = 0; i < in.rows(); ++i) {
      const double* r = in.row(i);
      Mat3 m;
      m << r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8];
      MatrixToRot6d(m, got.data() + static_cast<size_t>(i) * 6);
    }
    ReportMargin(*c, WholeError(got, want), got.size());
  }
}

TEST(WindowParity, QuatToRot6d) {
  for (const Case* c : Cases("quat_to_rot6d")) {
    const Blob& in = c->In("q");
    const Blob& want = c->Out("result");
    std::vector<double> got(want.data.size());
    for (int i = 0; i < in.rows(); ++i) {
      QuatToRot6d(QuatAt(in, i), got.data() + static_cast<size_t>(i) * 6);
    }
    ReportMargin(*c, WholeError(got, want), got.size());
  }
}

TEST(WindowParity, Rot6dToMatrix) {
  for (const Case* c : Cases("rot6d_to_matrix")) {
    const Blob& in = c->In("d6");
    const Blob& want = c->Out("result");
    std::vector<double> got(want.data.size());
    for (int i = 0; i < in.rows(); ++i) {
      const Mat3 m = Rot6dToMatrix(in.row(i));
      double* dst = got.data() + static_cast<size_t>(i) * 9;
      for (int r = 0; r < 3; ++r) {
        for (int col = 0; col < 3; ++col) dst[r * 3 + col] = m(r, col);
      }
    }
    ReportMargin(*c, WholeError(got, want), got.size());
  }
}

TEST(WindowParity, Rot6dToQuat) {
  for (const Case* c : Cases("rot6d_to_quat")) {
    const Blob& in = c->In("d6");
    const Blob& want = c->Out("result");
    // Compared as a rotation: Rot6dToQuat runs through QuatFromMatrix's four-candidate
    // argmax, so a pivot tie can legitimately return the negation.
    double worst = 0.0;
    for (int i = 0; i < in.rows(); ++i) {
      const QuatXyzw q = Rot6dToQuat(in.row(i));
      const double* w = want.row(i);
      double same = 0.0, flipped = 0.0;
      for (int k = 0; k < 4; ++k) {
        same = std::max(same, std::abs(q[k] - w[k]));
        flipped = std::max(flipped, std::abs(q[k] + w[k]));
      }
      worst = std::max(worst, std::min(same, flipped));
    }
    ReportMargin(*c, worst, want.data.size());
  }
}

TEST(WindowParity, WorldToWindow) {
  for (const Case* c : Cases("world_to_window")) {
    const auto blocks = BlocksOf(*c);
    const int nj = c->ints.at("num_joints");
    const int nee = c->ints.at("num_ee_frames");
    const int num_nodes = c->ints.at("num_nodes");

    const Blob& ref_pos = c->In("ref_pos_w");
    const Blob& ref_quat = c->In("ref_quat_w");
    const Blob& ref_lin = c->In("ref_lin_w");
    const Blob& ref_ang = c->In("ref_ang_w");
    const Blob& jpos = c->In("joint_pos");
    const Blob& jvel = c->In("joint_vel");
    const Blob& ee = c->In("ee_pos_w");

    std::vector<WorldNode> nodes(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
      nodes[i].ref_pos_w = Vec3At(ref_pos, i);
      nodes[i].ref_quat_w = QuatAt(ref_quat, i);
      nodes[i].ref_lin_w = Vec3At(ref_lin, i);
      nodes[i].ref_ang_w = Vec3At(ref_ang, i);
      nodes[i].joint_pos.assign(jpos.row(i), jpos.row(i) + nj);
      nodes[i].joint_vel.assign(jvel.row(i), jvel.row(i) + nj);
      for (int e = 0; e < nee; ++e) {
        const double* p = ee.row(i) + 3 * e;
        nodes[i].ee_pos_w.emplace_back(p[0], p[1], p[2]);
      }
    }

    const Blob& want = c->Out("window");
    ASSERT_EQ(want.stride(), WindowDim(nj, nee, blocks))
        << c->name << ": node width disagrees with the port's layout";
    std::vector<double> got(want.data.size());
    WorldToWindow(Vec3At(c->In("root_pos_w"), 0), QuatAt(c->In("root_quat_w"), 0), nodes, nj,
                  nee, blocks, got.data());
    ReportMargin(*c, WholeError(got, want), got.size());
  }
}

TEST(WindowParity, ReanchorWindow) {
  for (const Case* c : Cases("reanchor_window")) {
    const auto blocks = BlocksOf(*c);
    const int nj = c->ints.at("num_joints");
    const int nee = c->ints.at("num_ee_frames");
    const Blob& in = c->In("window");
    const Blob& want = c->Out("window");
    std::vector<double> got(want.data.size());
    ReanchorWindow(in.data.data(), in.rows(), Vec3At(c->In("old_pos_w"), 0),
                   QuatAt(c->In("old_quat_w"), 0), Vec3At(c->In("new_pos_w"), 0),
                   QuatAt(c->In("new_quat_w"), 0), nj, nee, blocks, got.data());
    ReportMargin(*c, WholeError(got, want), got.size());
  }
}

// Properties that hold without the oracle.

TEST(Window, LayoutMatchesTheLiveArtifact) {
  // The deployed pair: 29 joints, no ee frames, base_pose + base_vel + joint_pos. If this
  // ever stops being 44 the tracker and the generator have stopped agreeing on what a node
  // IS, which no shape check downstream would catch in time.
  const std::vector<ReferenceBlock> slim = {ReferenceBlock::kBasePose, ReferenceBlock::kBaseVel,
                                            ReferenceBlock::kJointPos};
  EXPECT_EQ(WindowDim(29, 0, slim), 44);
  const auto slices = WindowBlockSlices(29, 0, slim);
  ASSERT_EQ(slices.size(), 3u);
  EXPECT_EQ(slices[0].offset, 0);
  EXPECT_EQ(slices[1].offset, 9);
  EXPECT_EQ(slices[2].offset, 15);
}

TEST(Window, BlocksAreCanonicallyOrderedRegardlessOfRequestOrder) {
  // The order a caller asks in must not become the packing order: the generator's output IS
  // the tracker's observation, and a permuted layout reads velocities as positions without
  // changing any width.
  const std::vector<ReferenceBlock> shuffled = {
      ReferenceBlock::kJointPos, ReferenceBlock::kBasePose, ReferenceBlock::kBaseVel};
  const auto resolved = ResolveReferenceBlocks(shuffled);
  ASSERT_EQ(resolved.size(), 3u);
  EXPECT_EQ(resolved[0], ReferenceBlock::kBasePose);
  EXPECT_EQ(resolved[1], ReferenceBlock::kBaseVel);
  EXPECT_EQ(resolved[2], ReferenceBlock::kJointPos);
}

TEST(Window, EmptySelectionThrows) {
  EXPECT_THROW(ResolveReferenceBlocks({}), std::runtime_error);
}

TEST(Window, ReanchorOntoTheSameAnchorIsIdentity) {
  // Applied unconditionally at every K including 1, so it has to be an identity there to
  // ~2e-15 -- otherwise every step would inject a small error into the served reference.
  const std::vector<ReferenceBlock> slim = {ReferenceBlock::kBasePose, ReferenceBlock::kBaseVel,
                                            ReferenceBlock::kJointPos};
  const int nj = 29, nodes = 5, dim = WindowDim(nj, 0, slim);
  std::vector<double> in(static_cast<size_t>(nodes) * dim), out(in.size());
  for (size_t i = 0; i < in.size(); ++i) in[i] = 0.01 * static_cast<double>(i % 37) - 0.2;
  const Vec3 p(1.5, -2.0, 0.9);
  const QuatXyzw q = QuatFromAxisAngle(Vec3(0.1, -0.3, 0.7));
  ReanchorWindow(in.data(), nodes, p, q, p, q, nj, 0, slim, out.data());
  double worst = 0.0;
  for (size_t i = 0; i < in.size(); ++i) worst = std::max(worst, std::abs(in[i] - out[i]));
  EXPECT_LT(worst, 2e-15) << "same-anchor re-anchor drifted by " << worst;
}

TEST(Window, ReanchorSupportsInPlaceAliasing) {
  // The serving path re-anchors into the buffer it reads from. Getting this wrong would
  // corrupt base_pose using an already-overwritten position.
  const std::vector<ReferenceBlock> slim = {ReferenceBlock::kBasePose, ReferenceBlock::kBaseVel,
                                            ReferenceBlock::kJointPos};
  const int nj = 29, nodes = 4, dim = WindowDim(nj, 0, slim);
  std::vector<double> a(static_cast<size_t>(nodes) * dim), b;
  for (size_t i = 0; i < a.size(); ++i) a[i] = 0.03 * static_cast<double>(i % 23) - 0.3;
  b = a;
  const Vec3 op(0.2, 0.1, 0.0), np(0.7, -0.4, 0.05);
  const QuatXyzw oq = QuatFromAxisAngle(Vec3(0.0, 0.0, 0.3));
  const QuatXyzw nq = QuatFromAxisAngle(Vec3(0.0, 0.0, -0.2));
  std::vector<double> separate(a.size());
  ReanchorWindow(a.data(), nodes, op, oq, np, nq, nj, 0, slim, separate.data());
  ReanchorWindow(b.data(), nodes, op, oq, np, nq, nj, 0, slim, b.data());
  double worst = 0.0;
  for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::abs(separate[i] - b[i]));
  EXPECT_LT(worst, 1e-15) << "in-place re-anchor differs from out-of-place by " << worst;
}

// ---------------------------------------------------------------------------------------
// Which serve arm the artifact asks for
// ---------------------------------------------------------------------------------------

TEST(Window, ThePlanConventionSelectsThePlanArm) {
  EXPECT_EQ(AnchorFromConvention("pose_in_plan_yaw_frame_pos_ref_minus_plan_ori_rot6d"),
            WindowAnchor::kPlan);
}

TEST(Window, TheRobotNowConventionIsRefusedBecauseItNeedsOdometry) {
  // Not "unsupported" -- REFUSED. The maths for that arm is right here in ReanchorWindow and
  // is tested above; what this runtime does not have is a base position it can trust, and
  // serving robot_now means re-anchoring on one every control step. The two conventions have
  // identical widths and identical blocks, so accepting it would load clean and steer wrong.
  //
  // The message has to say WHICH arm and WHY, because "convention mismatch" sends the reader
  // to look for a shape bug that does not exist.
  try {
    AnchorFromConvention("pose_in_anchor_yaw_frame_pos_ref_minus_actual_ori_rot6d");
    FAIL() << "the robot_now convention was accepted; this runtime has no odometry";
  } catch (const std::runtime_error& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("robot_now"), std::string::npos) << msg;
    EXPECT_NE(msg.find("odometry"), std::string::npos) << msg;
    EXPECT_NE(msg.find("plan"), std::string::npos) << msg;
  }
}

TEST(Window, AnUnknownConventionThrowsAndNamesWhatItKnows) {
  try {
    AnchorFromConvention("pose_in_some_frame_nobody_trained");
    FAIL() << "an unrecognised convention was accepted";
  } catch (const std::runtime_error& e) {
    const std::string msg = e.what();
    // Echo the offending string AND the alternatives: a bare "unknown convention" leaves the
    // reader unable to tell a typo from a genuinely new arm.
    EXPECT_NE(msg.find("pose_in_some_frame_nobody_trained"), std::string::npos) << msg;
    EXPECT_NE(msg.find("pose_in_plan_yaw_frame"), std::string::npos) << msg;
  }
}

TEST(Window, ConventionForAnchorRoundTripsThePlanArm) {
  // The half that is served must survive the round trip, so a diagnostic can print the arm
  // it resolved and mean the same string the artifact carried.
  EXPECT_EQ(AnchorFromConvention(ConventionForAnchor(WindowAnchor::kPlan)),
            WindowAnchor::kPlan);
  EXPECT_STREQ(ConventionForAnchor(WindowAnchor::kRobotNow),
               "pose_in_anchor_yaw_frame_pos_ref_minus_actual_ori_rot6d");
}

}  // namespace
}  // namespace g1_deploy
