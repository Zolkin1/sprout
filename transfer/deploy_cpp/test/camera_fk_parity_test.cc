// Copyright 2026 Zachary Olkin. All rights reserved.

// Parity: the C++ camera pose vs the python oracle it was derived from.
//
// The camera pose is the one input to the generator where a mistake is INVISIBLE. A wrong
// depth value is a wrong number; a wrong pose is a perfectly plausible point cloud sitting
// somewhere the terrain is not, and the model then confidently plans against it. So this is
// held to the oracle rather than to a self-consistency property.
//
// The chain is C++ -> python pinocchio (these fixtures) -> dataset._camera_pose_yaw_frame
// (the training-side function). The second link is asserted in robot_rl's own suite, where
// isaaclab exists; it agreed exactly -- 0.000000 m and 4e-13 of rotation.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "fixture.h"
#include "g1_deploy/camera_fk.h"

namespace g1_deploy {
namespace {

using test::Blob;
using test::Case;
using test::Fixtures;

const Fixtures& Loaded() {
  static const Fixtures fixtures = Fixtures::Load(DefaultFixtureDir());
  return fixtures;
}

/// The URDF the fixtures were generated against, resolved off the fixture directory so the
/// two cannot drift onto different models.
std::string ModelPath(const Case& c) {
  const std::string rel = c.strings.at("urdf");
  const std::string dir = DefaultFixtureDir();
  // fixtures live at <repo>/transfer/deploy/tests/fixtures/parity
  return dir + "/../../../../../" + rel;
}

TEST(CameraFkParity, MatchesThePythonPose) {
  auto cases = Loaded().ForFunction("camera_pose_yaw_frame");
  ASSERT_FALSE(cases.empty()) << "no camera_pose_yaw_frame fixture; regenerate the fixtures";

  for (const Case* c : cases) {
    const Blob& joints = c->In("joint_pos");
    const Blob& quats = c->In("base_quat_xyzw");
    const Blob& want_pos = c->Out("cam_pos");
    const Blob& want_rot = c->Out("cam_rot");

    std::vector<std::string> names = c->string_lists.at("joint_names");
    std::vector<std::string> mounts = c->string_lists.at("mount_bodies");
    // The contract the fixture used: the mounts AND the nominal offsets, both read from the
    // fixture's own attrs so the test cannot quietly describe a different camera pair than the
    // oracle did. That claim used to be false -- the offsets were restated here as literals,
    // and when the mount poses moved from the placeholder pair to the as-built CAD ones this
    // test and its fixture went on agreeing with each other about a robot that ships nowhere.
    // A parity test that pins its own inputs is not a parity test.
    const std::vector<double>& off_flat = c->double_lists.at("offset_pos");
    const std::vector<double>& quat_flat = c->double_lists.at("offset_quat_xyzw");
    ASSERT_EQ(off_flat.size(), mounts.size() * 3);
    ASSERT_EQ(quat_flat.size(), mounts.size() * 4);
    std::vector<DepthCameraSpec> cams(mounts.size());
    for (size_t i = 0; i < cams.size(); ++i) {
      cams[i].sensor = "cam" + std::to_string(i);
      cams[i].mount_body = mounts[i];
      for (size_t k = 0; k < 3; ++k) cams[i].offset_pos[k] = off_flat[i * 3 + k];
      for (size_t k = 0; k < 4; ++k) cams[i].offset_quat_xyzw[k] = quat_flat[i * 4 + k];
    }

    CameraPoseFk fk(ModelPath(*c), names, cams);
    ASSERT_EQ(fk.num_cameras(), cams.size());

    double worst_pos = 0.0;
    double worst_rot = 0.0;
    std::vector<std::array<double, 3>> pos;
    std::vector<std::array<double, 9>> rot;
    for (int r = 0; r < joints.rows(); ++r) {
      const std::vector<double> q(joints.row(r), joints.row(r) + joints.stride());
      std::array<double, 4> base{quats.row(r)[0], quats.row(r)[1], quats.row(r)[2],
                                 quats.row(r)[3]};
      fk.Compute(q, base, &pos, &rot);
      for (size_t cam = 0; cam < cams.size(); ++cam) {
        for (size_t i = 0; i < 3; ++i) {
          worst_pos = std::max(worst_pos,
                               std::abs(pos[cam][i] - want_pos.row(r)[cam * 3 + i]));
        }
        for (size_t i = 0; i < 9; ++i) {
          worst_rot = std::max(worst_rot,
                               std::abs(rot[cam][i] - want_rot.row(r)[cam * 9 + i]));
        }
      }
    }
    EXPECT_LE(worst_pos, c->atol) << "position: worst " << worst_pos << " m";
    EXPECT_LE(worst_rot, c->atol) << "rotation: worst " << worst_rot;
    std::cout << "    [margin] camera pose worst " << worst_pos << " m / " << worst_rot
              << " rot over " << joints.rows() << " configurations\n";
  }
}

TEST(CameraFk, TheHeadingDividesOut) {
  // THE deployability property, asserted directly rather than inferred from the fixtures: the
  // same posture at two different headings must give the same camera pose. That is what lets
  // this run with no odometry and survive a drifting yaw estimate -- and it is the half a
  // reimplementation is most likely to get wrong, because leaving psi in looks correct in
  // every plot drawn in the world frame.
  auto cases = Loaded().ForFunction("camera_pose_yaw_frame");
  ASSERT_FALSE(cases.empty());
  const Case& c = *cases.front();
  std::vector<std::string> names = c.string_lists.at("joint_names");
  // From the fixture, like the parity case above: the yaw property holds for ANY offset, so
  // the literal was harmless -- but a stale literal beside a live one is how the pair drifts.
  std::vector<DepthCameraSpec> cams(1);
  cams[0].mount_body = c.string_lists.at("mount_bodies").front();
  const std::vector<double>& off0 = c.double_lists.at("offset_pos");
  const std::vector<double>& quat0 = c.double_lists.at("offset_quat_xyzw");
  ASSERT_GE(off0.size(), 3u);
  ASSERT_GE(quat0.size(), 4u);
  for (size_t k = 0; k < 3; ++k) cams[0].offset_pos[k] = off0[k];
  for (size_t k = 0; k < 4; ++k) cams[0].offset_quat_xyzw[k] = quat0[k];
  CameraPoseFk fk(ModelPath(c), names, cams);

  std::vector<double> q(names.size(), 0.0);
  for (size_t i = 0; i < q.size(); ++i) q[i] = 0.1 * static_cast<double>(i % 5) - 0.2;
  std::vector<std::array<double, 3>> p0, p1;
  std::vector<std::array<double, 9>> r0, r1;
  // Pure yaw, two very different headings, identical tilt.
  fk.Compute(q, {0.0, 0.0, std::sin(0.3), std::cos(0.3)}, &p0, &r0);
  fk.Compute(q, {0.0, 0.0, std::sin(-1.1), std::cos(-1.1)}, &p1, &r1);
  for (size_t i = 0; i < 3; ++i) EXPECT_NEAR(p0[0][i], p1[0][i], 1e-12);
  for (size_t i = 0; i < 9; ++i) EXPECT_NEAR(r0[0][i], r1[0][i], 1e-12);
}

TEST(CameraFk, RefusesAModelThatIsNotThisRobot) {
  auto cases = Loaded().ForFunction("camera_pose_yaw_frame");
  ASSERT_FALSE(cases.empty());
  const Case& c = *cases.front();
  std::vector<DepthCameraSpec> cams(1);
  cams[0].mount_body = "torso_link";
  // A joint the model does not have: these must be the same robot, and a silent mismatch
  // would write angles into the wrong slots and produce a plausible, wrong pose.
  EXPECT_THROW(CameraPoseFk(ModelPath(c), {"not_a_joint"}, cams), std::runtime_error);
  // A mount body the model does not define.
  std::vector<DepthCameraSpec> bad(1);
  bad[0].mount_body = "not_a_link";
  EXPECT_THROW(CameraPoseFk(ModelPath(c), c.string_lists.at("joint_names"), bad),
               std::runtime_error);
}

}  // namespace
}  // namespace g1_deploy
