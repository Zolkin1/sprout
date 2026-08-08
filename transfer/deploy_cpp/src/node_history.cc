// Copyright 2026 Zachary Olkin. All rights reserved.

#include "g1_deploy/node_history.h"

#include <algorithm>
#include <stdexcept>

namespace g1_deploy {

NodeHistory::NodeHistory(int maxlen, int node_dim, int cmd_dim)
    : maxlen_(maxlen), node_dim_(node_dim), cmd_dim_(cmd_dim) {
  if (maxlen <= 0) throw std::runtime_error("NodeHistory needs maxlen >= 1");
  nodes_.assign(static_cast<size_t>(maxlen) * node_dim, 0.0);
  cmds_.assign(static_cast<size_t>(maxlen) * cmd_dim, 0.0);
}

const double* NodeHistory::NodeRow(int i) const {
  if (i < 0 || i >= size_) return nullptr;
  const int slot = (head_ + i) % maxlen_;
  return nodes_.data() + static_cast<size_t>(slot) * node_dim_;
}

const double* NodeHistory::CmdRow(int i) const {
  if (i < 0 || i >= size_) return nullptr;
  const int slot = (head_ + i) % maxlen_;
  return cmds_.data() + static_cast<size_t>(slot) * cmd_dim_;
}

void NodeHistory::Append(const double* node, const double* cmd) {
  const int slot = (head_ + size_) % maxlen_;
  std::copy(node, node + node_dim_, nodes_.data() + static_cast<size_t>(slot) * node_dim_);
  std::copy(cmd, cmd + cmd_dim_, cmds_.data() + static_cast<size_t>(slot) * cmd_dim_);
  if (size_ < maxlen_) {
    ++size_;
  } else {
    head_ = (head_ + 1) % maxlen_;  // full: the oldest row is evicted, as deque(maxlen) does
  }
  ++steps_since_reset_;
}

void NodeHistory::Invalidate() { steps_since_reset_ = 0; }

void NodeHistory::Fill(const double* node, const double* cmd) {
  head_ = 0;
  size_ = maxlen_;
  for (int i = 0; i < maxlen_; ++i) {
    std::copy(node, node + node_dim_, nodes_.data() + static_cast<size_t>(i) * node_dim_);
    std::copy(cmd, cmd + cmd_dim_, cmds_.data() + static_cast<size_t>(i) * cmd_dim_);
  }
  steps_since_reset_ = maxlen_;
}

bool NodeHistory::Valid() const {
  if (size_ < maxlen_) return false;
  return steps_since_reset_ >= maxlen_;
}

bool NodeHistory::NullMaskAfter(int lookahead) const {
  if (lookahead < 0) throw std::runtime_error("NullMaskAfter needs a non-negative lookahead");
  // Exactly what Valid() would report after `lookahead` more Appends: the ring fills and the
  // counter advances by one per append, and neither can exceed maxlen in a way that matters.
  const int size_then = std::min(size_ + lookahead, maxlen_);
  const long long since_then = steps_since_reset_ + lookahead;
  return !(size_then >= maxlen_ && since_then >= maxlen_);
}

void NodeHistory::StackPadded(bool nodes, const std::vector<const double*>& extra,
                              double* out) const {
  const int dim = nodes ? node_dim_ : cmd_dim_;
  if (size_ == 0 && extra.empty()) {
    throw std::runtime_error(
        "StackPadded on an empty history: append at least one step first.");
  }

  // The logical sequence after the projected appends, oldest first: the rows held now, then
  // `extra`. Only the last maxlen survive, because Append evicts from the front.
  const int total = size_ + static_cast<int>(extra.size());
  const int kept = std::min(total, maxlen_);
  const int first = total - kept;  // index into the logical sequence of the oldest survivor

  auto logical_row = [&](int idx) -> const double* {
    if (idx < size_) return nodes ? NodeRow(idx) : CmdRow(idx);
    return extra[static_cast<size_t>(idx - size_)];
  };

  // Front-pad by repeating the OLDEST survivor when the window is still underfull, so the
  // newest step always lands on the last row and the caller's anchor is the live state.
  const int pad = maxlen_ - kept;
  const double* oldest = logical_row(first);
  for (int j = 0; j < pad; ++j) {
    std::copy(oldest, oldest + dim, out + static_cast<size_t>(j) * dim);
  }
  for (int j = 0; j < kept; ++j) {
    const double* src = logical_row(first + j);
    std::copy(src, src + dim, out + static_cast<size_t>(pad + j) * dim);
  }
}

void NodeHistory::StackPaddedNodes(double* out) const { StackPadded(true, {}, out); }

void NodeHistory::StackPaddedCmds(double* out) const { StackPadded(false, {}, out); }

void NodeHistory::StackPaddedNodesProjected(const std::vector<const double*>& extra,
                                            double* out) const {
  StackPadded(true, extra, out);
}

void NodeHistory::StackPaddedCmdsProjected(const std::vector<const double*>& extra,
                                           double* out) const {
  StackPadded(false, extra, out);
}

}  // namespace g1_deploy
