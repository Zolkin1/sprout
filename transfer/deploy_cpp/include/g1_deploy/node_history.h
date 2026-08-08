// Copyright 2026 Zachary Olkin. All rights reserved.
#pragma once

// Port of transfer/deploy/history.py (BatchedHistory), specialised to the deploy case.
//
// The python class is batched because it is shared with the Isaac side; deploy always runs
// num_envs = 1, so the batching is dropped here and the per-env mask becomes a bool.
//
// This holds the generator's autoregressive conditioning: each row is the reference-window
// node this runtime SERVED at that step, raw, in its plan's own frame -- not a measured
// state and never re-anchored. That detail is what makes the forward projection below
// possible at all.
//
// Storage is a preallocated ring: no allocation after construction, since this is touched
// every control tick.

#include <vector>

namespace g1_deploy {

/// Fixed-length rolling history of served nodes and commands.
class NodeHistory {
 public:
  /// Args:
  ///   maxlen: N, the generator's history length (from fm.onnx metadata).
  ///   node_dim: D, the width of one window node.
  ///   cmd_dim: width of one velocity command.
  NodeHistory(int maxlen, int node_dim, int cmd_dim);

  /// Append one step. `node` is node_dim wide, `cmd` is cmd_dim wide.
  void Append(const double* node, const double* cmd);

  /// Restart the validity counter, for a reset. Does NOT clear the stored rows -- matching
  /// the python, where the content stays and only the counter says it is untrustworthy.
  void Invalidate();

  /// Warm-start: fill every slot with one step and mark the window valid.
  ///
  /// The legacy path only. With the learned null-history embedding the generator runs from
  /// step 0 instead, because a static repeated window never occurs in training (real
  /// history always has motion between its steps).
  void Fill(const double* node, const double* cmd);

  /// True once there are maxlen uninterrupted steps since the last reset.
  bool Valid() const;

  /// True where the window is NOT yet complete -- the generator's `history_null` input.
  ///
  /// Training nulled a fraction of its samples, so the learned null_history embedding covers
  /// exactly this case and the generator can be sampled from step 0 with no warmup and no
  /// fabricated history.
  bool NullMask() const { return !Valid(); }

  /// `history_null` as it WILL be after `lookahead` further appends.
  ///
  /// Used by the async generator: a plan requested `lookahead` ticks before its cadence
  /// boundary must be conditioned exactly as the synchronous runtime would have conditioned
  /// it AT the boundary, and that includes whether the window is complete by then.
  bool NullMaskAfter(int lookahead) const;

  /// Stack the node rows into `out` (maxlen x node_dim, row-major), front-padded by
  /// repeating the OLDEST row when the buffer is not yet full.
  ///
  /// Only the global underfill (the first steps of a run) is padded; staleness after a reset
  /// is not repaired, because those are exactly the windows NullMask flags, whose content the
  /// model replaces with the learned embedding. The newest step stays at the last row, so the
  /// caller's anchor is always the live state.
  ///
  /// Throws std::runtime_error if nothing has been appended yet.
  void StackPaddedNodes(double* out) const;

  /// As StackPaddedNodes, for the command rows (maxlen x cmd_dim).
  void StackPaddedCmds(double* out) const;

  /// The node stack as it WILL be after appending `extra` rows -- the forward projection
  /// that makes the async generator exact.
  ///
  /// The rows a plan needs for its boundary are the nodes served on the ticks between the
  /// request and that boundary. Those are already determined: they are the live plan's nodes
  /// at the cursor and after it, and this runtime serves them raw, in the plan's own frame.
  /// So they can be handed over at request time and the generator sees precisely the history
  /// the synchronous code would have built. Eviction and padding follow Append exactly.
  ///
  /// Args:
  ///   extra: rows to append conceptually, oldest first, each node_dim wide.
  void StackPaddedNodesProjected(const std::vector<const double*>& extra, double* out) const;

  /// As StackPaddedNodesProjected, for commands. The future command is not knowable, so the
  /// caller holds the one at request time -- which is what the FM's own future-command
  /// horizon already does.
  void StackPaddedCmdsProjected(const std::vector<const double*>& extra, double* out) const;

  int maxlen() const { return maxlen_; }
  int size() const { return size_; }
  int node_dim() const { return node_dim_; }
  int cmd_dim() const { return cmd_dim_; }

 private:
  /// Pointer to logical row `i` (0 = oldest) of the node ring, or nullptr if out of range.
  const double* NodeRow(int i) const;
  const double* CmdRow(int i) const;
  void StackPadded(bool nodes, const std::vector<const double*>& extra, double* out) const;

  int maxlen_;
  int node_dim_;
  int cmd_dim_;
  int size_ = 0;   //: rows currently held, <= maxlen_
  int head_ = 0;   //: ring index of the oldest row
  long long steps_since_reset_ = 0;
  std::vector<double> nodes_;
  std::vector<double> cmds_;
};

}  // namespace g1_deploy
