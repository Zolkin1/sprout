// Copyright 2026 Zachary Olkin. All rights reserved.

// Semantics of the generator's conditioning history, and the forward projection the async
// generator depends on. No fixtures and no oracle: these are structural properties.

#include <gtest/gtest.h>

#include <vector>

#include "g1_deploy/node_history.h"

namespace g1_deploy {
namespace {

constexpr int kMaxlen = 4;  // N for the live generator
constexpr int kNodeDim = 3;
constexpr int kCmdDim = 2;

/// A recognisable node row, so a misplaced row is obvious in a failure message.
std::vector<double> Node(double tag) { return {tag, tag + 0.1, tag + 0.2}; }
std::vector<double> Cmd(double tag) { return {tag, -tag}; }

std::vector<double> StackNodes(const NodeHistory& h) {
  std::vector<double> out(static_cast<size_t>(h.maxlen()) * h.node_dim());
  h.StackPaddedNodes(out.data());
  return out;
}

TEST(NodeHistory, EmptyStackThrows) {
  NodeHistory h(kMaxlen, kNodeDim, kCmdDim);
  std::vector<double> out(kMaxlen * kNodeDim);
  EXPECT_THROW(h.StackPaddedNodes(out.data()), std::runtime_error);
}

TEST(NodeHistory, UnderfillIsFrontPaddedWithTheOldestAndNewestIsLast) {
  NodeHistory h(kMaxlen, kNodeDim, kCmdDim);
  h.Append(Node(1).data(), Cmd(1).data());
  h.Append(Node(2).data(), Cmd(2).data());
  const auto s = StackNodes(h);
  // Expect [1, 1, 1, 2]: pad by repeating the OLDEST, newest stays at the last row so the
  // caller's anchor is always the live state.
  EXPECT_EQ(s[0 * kNodeDim], 1.0);
  EXPECT_EQ(s[1 * kNodeDim], 1.0);
  EXPECT_EQ(s[2 * kNodeDim], 1.0);
  EXPECT_EQ(s[3 * kNodeDim], 2.0);
}

TEST(NodeHistory, EvictsOldestOnceFull) {
  NodeHistory h(kMaxlen, kNodeDim, kCmdDim);
  for (int i = 1; i <= 6; ++i) h.Append(Node(i).data(), Cmd(i).data());
  const auto s = StackNodes(h);
  for (int j = 0; j < kMaxlen; ++j) EXPECT_EQ(s[j * kNodeDim], 3.0 + j) << "row " << j;
}

TEST(NodeHistory, ValidOnlyAfterMaxlenUninterruptedSteps) {
  NodeHistory h(kMaxlen, kNodeDim, kCmdDim);
  for (int i = 0; i < kMaxlen - 1; ++i) {
    h.Append(Node(i).data(), Cmd(i).data());
    EXPECT_TRUE(h.NullMask()) << "incomplete window must report history_null";
  }
  h.Append(Node(9).data(), Cmd(9).data());
  EXPECT_FALSE(h.NullMask());

  // A reset does not clear the rows, only their trustworthiness -- matching the python.
  h.Invalidate();
  EXPECT_TRUE(h.NullMask());
  for (int i = 0; i < kMaxlen - 1; ++i) {
    h.Append(Node(i).data(), Cmd(i).data());
    EXPECT_TRUE(h.NullMask());
  }
  h.Append(Node(9).data(), Cmd(9).data());
  EXPECT_FALSE(h.NullMask());
}

TEST(NodeHistory, FillWarmStartsEverySlotAndMarksValid) {
  NodeHistory h(kMaxlen, kNodeDim, kCmdDim);
  h.Fill(Node(7).data(), Cmd(7).data());
  EXPECT_FALSE(h.NullMask());
  const auto s = StackNodes(h);
  for (int j = 0; j < kMaxlen; ++j) EXPECT_EQ(s[j * kNodeDim], 7.0);
}

// ---------------------------------------------------------------------------------------
// The projection. This is the assertion the async generator schedule rests on: a plan
// requested `lookahead` ticks before its cadence boundary is conditioned on the history as
// it WILL be at that boundary, so the generator sees exactly what the synchronous runtime
// would have handed it. If this ever stops holding, the async path silently conditions the
// generator on something training never produced.
// ---------------------------------------------------------------------------------------

TEST(NodeHistory, ProjectionEqualsActuallyAppending) {
  for (int prior = 0; prior <= 6; ++prior) {
    for (int lookahead = 0; lookahead <= kMaxlen + 1; ++lookahead) {
      NodeHistory h(kMaxlen, kNodeDim, kCmdDim);
      for (int i = 1; i <= prior; ++i) h.Append(Node(i).data(), Cmd(i).data());
      if (prior == 0 && lookahead == 0) continue;  // empty: covered by EmptyStackThrows

      std::vector<std::vector<double>> future_nodes, future_cmds;
      for (int i = 0; i < lookahead; ++i) {
        future_nodes.push_back(Node(100 + i));
        future_cmds.push_back(Cmd(100 + i));
      }
      std::vector<const double*> node_ptrs, cmd_ptrs;
      for (int i = 0; i < lookahead; ++i) {
        node_ptrs.push_back(future_nodes[i].data());
        cmd_ptrs.push_back(future_cmds[i].data());
      }

      std::vector<double> projected(static_cast<size_t>(kMaxlen) * kNodeDim);
      std::vector<double> projected_cmd(static_cast<size_t>(kMaxlen) * kCmdDim);
      h.StackPaddedNodesProjected(node_ptrs, projected.data());
      h.StackPaddedCmdsProjected(cmd_ptrs, projected_cmd.data());
      const bool projected_null = h.NullMaskAfter(lookahead);

      // The same history, actually advanced.
      NodeHistory advanced(kMaxlen, kNodeDim, kCmdDim);
      for (int i = 1; i <= prior; ++i) advanced.Append(Node(i).data(), Cmd(i).data());
      for (int i = 0; i < lookahead; ++i) {
        advanced.Append(future_nodes[i].data(), future_cmds[i].data());
      }
      std::vector<double> actual(static_cast<size_t>(kMaxlen) * kNodeDim);
      std::vector<double> actual_cmd(static_cast<size_t>(kMaxlen) * kCmdDim);
      advanced.StackPaddedNodes(actual.data());
      advanced.StackPaddedCmds(actual_cmd.data());

      EXPECT_EQ(projected, actual)
          << "node projection differs at prior=" << prior << " lookahead=" << lookahead;
      EXPECT_EQ(projected_cmd, actual_cmd)
          << "cmd projection differs at prior=" << prior << " lookahead=" << lookahead;
      EXPECT_EQ(projected_null, advanced.NullMask())
          << "history_null projection differs at prior=" << prior
          << " lookahead=" << lookahead;
    }
  }
}

TEST(NodeHistory, ProjectionWithNoLookaheadIsTheOrdinaryStack) {
  // lookahead 0 is the synchronous case, and it must not take a different code path.
  NodeHistory h(kMaxlen, kNodeDim, kCmdDim);
  for (int i = 1; i <= 3; ++i) h.Append(Node(i).data(), Cmd(i).data());
  std::vector<double> plain(static_cast<size_t>(kMaxlen) * kNodeDim);
  std::vector<double> projected(plain.size());
  h.StackPaddedNodes(plain.data());
  h.StackPaddedNodesProjected({}, projected.data());
  EXPECT_EQ(plain, projected);
}

TEST(NodeHistory, ProjectionDoesNotMutate) {
  // The request is built on the control thread mid-tick; a projection that advanced the real
  // history would corrupt the next tick's conditioning.
  NodeHistory h(kMaxlen, kNodeDim, kCmdDim);
  for (int i = 1; i <= 3; ++i) h.Append(Node(i).data(), Cmd(i).data());
  const auto before = StackNodes(h);
  const bool null_before = h.NullMask();

  const auto future = Node(42);
  std::vector<double> out(static_cast<size_t>(kMaxlen) * kNodeDim);
  h.StackPaddedNodesProjected({future.data()}, out.data());

  EXPECT_EQ(StackNodes(h), before);
  EXPECT_EQ(h.NullMask(), null_before);
  EXPECT_EQ(h.size(), 3);
}

}  // namespace
}  // namespace g1_deploy
