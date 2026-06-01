#include "tree_serial.h"

#include <utility>

TreeSerial::SplitSearchResult TreeSerial::findBestSplitAtNode(
    const std::vector<std::size_t> &rowIndices) const {
  // Serial means "try one feature after another on this thread".
  // The math for scoring a feature lives in TreeBase so Serial and Parallel
  // learn the same tree when they use the same options.
  const std::size_t featureCount = dataset_->featureNames.size();
  if (featureCount == 0) {
    return {};
  }

  std::vector<SplitSearchResult> featureResults;
  featureResults.reserve(featureCount);

  for (std::size_t featureIndex = 0; featureIndex < featureCount; ++featureIndex) {
    // Evaluate this feature's possible thresholds, then save its best split.
    featureResults.push_back(evaluateFeatureSplit(rowIndices, featureIndex));
  }

  // Pick the best split among all features. This is shared with TreeParallel.
  return reduceBestSplitSearch(std::move(featureResults), rowIndices);
}

void TreeSerial::fit(const Dataset &dataset, const TrainingOptions &options) {
  // Shared setup: store the dataset, copy options, and build fast label lookup.
  prepareFit(dataset, options);

  // At the root, every training row is available for splitting.
  const std::vector<std::size_t> rowIndices = makeRootRowIndices();
  const BuildTimePoint buildStart = startBuildTimer();

  // Build the whole tree with normal recursive calls: left subtree, then right.
  root_ = buildNode(rowIndices, 0);
  finishBuildTimer(buildStart);

  // Shared finish step: run the selected pruning method, if any.
  finalizeFit(options);
}
