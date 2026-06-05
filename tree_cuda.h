#pragma once

#include "tree_base.h"

#include <cstddef>
#include <vector>

// GPU backend: CPU builds the tree recursively; split search at each node runs on GPU.
class TreeCuda : public TreeBase {
public:
  ~TreeCuda() override;

  void fit(const Dataset &dataset,
           const Options &options = Options{}) override;

protected:
  SplitSearchResult findBestSplitAtNode(
      const std::vector<std::size_t> &rowIndices) const override;

private:
  struct CudaState;

  void releaseCudaState() noexcept;

  mutable CudaState *cuda_ = nullptr;
};
