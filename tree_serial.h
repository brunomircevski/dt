#pragma once

#include "tree_base.h"

// Single-threaded tree. Ignores thread-count options.
class TreeSerial : public TreeBase {
public:
  void fit(const Dataset &dataset,
           const TrainingOptions &options = TrainingOptions{}) override;

protected:
  SplitSearchResult findBestSplitAtNode(
      const std::vector<std::size_t> &rowIndices) const override;
};
