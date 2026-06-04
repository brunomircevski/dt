#include "tree_parallel.h"

#include "node.h"

#include <algorithm>
#include <future>
#include <utility>

void TreeParallel::setupParallelExecutors() {
  // Two thread pools:
  // - featureExecutor: score each feature at the same node in parallel
  // - nodeExecutor: build one child subtree while the current thread builds the
  // other
  const std::size_t featureThreadCount =
      static_cast<std::size_t>(std::max(options_.maxFeatureThreadCount, 1));
  const std::size_t nodeThreadCount =
      static_cast<std::size_t>(std::max(options_.maxNodeThreadCount, 1));
  fitContext_->featureExecutor =
      std::make_unique<TaskExecutor>(featureThreadCount);
  fitContext_->nodeExecutor = std::make_unique<TaskExecutor>(nodeThreadCount);
  // Leave at least one worker free so a thread waiting on future.get() does not
  // deadlock.
  fitContext_->maxNodeTasks = nodeThreadCount > 1 ? nodeThreadCount - 1 : 0;
  fitContext_->activeNodeTasks.store(0, std::memory_order_relaxed);
}

bool TreeParallel::shouldParallelizeAttributes(std::size_t featureCount) const {
  return featureCount >= options_.minFeaturesToParallelize;
}

bool TreeParallel::shouldParallelizeNodes(std::size_t rowCount) const {
  return rowCount >= options_.minRowsToParallelize;
}

bool TreeParallel::tryStartNodeTask() const {
  if (fitContext_->maxNodeTasks == 0) {
    return false;
  }
  std::size_t current =
      fitContext_->activeNodeTasks.load(std::memory_order_relaxed);
  while (current < fitContext_->maxNodeTasks) {
    if (fitContext_->activeNodeTasks.compare_exchange_weak(
            current, current + 1, std::memory_order_relaxed,
            std::memory_order_relaxed)) {
      return true;
    }
  }
  return false;
}

void TreeParallel::finishNodeTask() const {
  fitContext_->activeNodeTasks.fetch_sub(1, std::memory_order_relaxed);
}

std::unique_ptr<Node>
TreeParallel::buildNodeParallel(const std::vector<std::size_t> &rowIndices,
                                int depth) const {
  NodeExpandResult expanded = expandOneNode(rowIndices, depth);
  if (expanded.node->isLeaf) {
    return std::move(expanded.node);
  }

  if (!shouldParallelizeNodes(rowIndices.size()) || !tryStartNodeTask()) {
    expanded.node->leftChild =
        buildNodeParallel(expanded.partitions.leftRows, depth + 1);
    expanded.node->rightChild =
        buildNodeParallel(expanded.partitions.rightRows, depth + 1);
    return std::move(expanded.node);
  }

  auto leftRows = std::move(expanded.partitions.leftRows);
  auto rightRows = std::move(expanded.partitions.rightRows);

  std::future<std::unique_ptr<Node>> leftJob =
      fitContext_->nodeExecutor->submit(
          [this, rows = std::move(leftRows), depth]() {
            std::unique_ptr<Node> child = buildNodeParallel(rows, depth + 1);
            finishNodeTask();
            return child;
          });
          
  expanded.node->rightChild = buildNodeParallel(rightRows, depth + 1);
  expanded.node->leftChild = leftJob.get(); // pauses until thread is finished
  return std::move(expanded.node);
}

TreeParallel::SplitSearchResult TreeParallel::findBestSplitAtNode(
    const std::vector<std::size_t> &rowIndices) const {
  // At one node, every feature can be tested independently.
  const std::size_t featureCount = dataset_->featureNames.size();
  if (featureCount == 0) {
    return {};
  }

  std::vector<SplitSearchResult> bestSplitPerFeature;
  bestSplitPerFeature.reserve(featureCount);

  if (shouldParallelizeAttributes(featureCount)) {
    const auto sharedRows =
        std::make_shared<std::vector<std::size_t>>(rowIndices);
    std::vector<std::future<SplitSearchResult>> featureJobs(featureCount);
    for (std::size_t featureIndex = 0; featureIndex < featureCount;
         ++featureIndex) {
      featureJobs[featureIndex] = fitContext_->featureExecutor->submit(
          [this, sharedRows, featureIndex]() {
            return evaluateFeatureSplit(*sharedRows, featureIndex);
          });
    }

    for (std::size_t featureIndex = 0; featureIndex < featureCount;
         ++featureIndex) {
      bestSplitPerFeature.push_back(featureJobs[featureIndex].get());
    }
  } else {
    for (std::size_t featureIndex = 0; featureIndex < featureCount;
         ++featureIndex) {
      bestSplitPerFeature.push_back(
          evaluateFeatureSplit(rowIndices, featureIndex));
    }
  }

  return reduceBestSplitSearch(std::move(bestSplitPerFeature), rowIndices);
}

void TreeParallel::fit(const Dataset &dataset, const TrainingOptions &options) {
  prepareFit(dataset, options);

  fitContext_ = std::make_unique<FitContext>();
  setupParallelExecutors();

  const std::vector<std::size_t> rowIndices = makeRootRowIndices();
  const BuildTimePoint buildStart = startBuildTimer();

  root_ = buildNodeParallel(rowIndices, 0);

  finishBuildTimer(buildStart);
  fitContext_.reset();

  finalizeFit(options);
}
