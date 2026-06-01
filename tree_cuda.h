#pragma once

#include "tree_base.h"

// GPU backend — not implemented yet. Same layout as TreeSerial / TreeParallel.
class TreeCuda : public TreeBase {
public:
  void fit(const Dataset &dataset,
           const TrainingOptions &options = TrainingOptions{}) override;

protected:
  SplitSearchResult findBestSplitAtNode(
      const std::vector<std::size_t> &rowIndices) const override;
};
