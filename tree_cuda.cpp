#include "tree_cuda.h"

#include <stdexcept>

void TreeCuda::fit(const Dataset &, const TrainingOptions &) {
  throw std::runtime_error(
      "TreeCuda is not implemented yet; use TreeSerial or TreeParallel.");
}

TreeCuda::SplitSearchResult TreeCuda::findBestSplitAtNode(
    const std::vector<std::size_t> &) const {
  throw std::runtime_error(
      "TreeCuda is not implemented yet; use TreeSerial or TreeParallel.");
}
