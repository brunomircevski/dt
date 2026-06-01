#include "tree_parallel.h"

#include "node.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <utility>

void TreeParallel::setupParallelExecutors() {
  // TreeParallel owns two small thread pools:
  // - attributeExecutor: tries different features at the same node in parallel
  // - nodeExecutor: builds left and right subtrees in parallel
  const std::size_t threadCount =
      static_cast<std::size_t>(std::max(options_.maxThreadCount, 1));
  fitContext_->attributeExecutor = std::make_unique<TaskExecutor>(threadCount);
  fitContext_->nodeExecutor = std::make_unique<TaskExecutor>(threadCount);
}

void TreeParallel::growTreeWithNodeParallelism(
    const std::vector<std::size_t> &rowIndices) {
  // The root is also built as a job. A promise/future pair gives the final
  // root node back to fit() after all async work is done.
  auto rootPromise = std::make_shared<std::promise<std::unique_ptr<Node>>>();
  std::future<std::unique_ptr<Node>> rootFuture = rootPromise->get_future();
  auto onComplete =
      wrapNodeOnComplete([rootPromise](std::unique_ptr<Node> node) mutable {
        rootPromise->set_value(std::move(node));
      });

  const RowIndexList rootRows =
      std::make_shared<std::vector<std::size_t>>(rowIndices);

  // pendingNodeJobs counts unfinished node jobs. waitForAllNodeJobs() sleeps
  // until this counter reaches zero.
  fitContext_->pendingNodeJobs.fetch_add(1, std::memory_order_relaxed);
  fitContext_->nodeExecutor->submit(
      [this, rootRows, onComplete = std::move(onComplete)]() mutable {
        expandNodeAsync({rootRows, 0, std::move(onComplete)});
      });

  waitForAllNodeJobs();
  root_ = rootFuture.get();
}

bool TreeParallel::shouldParallelizeAttributes(std::size_t featureCount) const {
  // For very few features, thread overhead can cost more than it saves.
  return featureCount >= options_.minFeaturesToParallelize;
}

bool TreeParallel::shouldParallelizeNodes(std::size_t rowCount) const {
  // Small subtrees are cheaper to build directly on the current thread.
  return rowCount >= options_.minRowsToParallelize;
}

void TreeParallel::notifyNodeJobFinished() const {
  // Called once per finished node job. The last job wakes the waiting fit().
  if (fitContext_->pendingNodeJobs.fetch_sub(1, std::memory_order_acq_rel) ==
      1) {
    std::lock_guard<std::mutex> lock(fitContext_->completionMutex);
    fitContext_->completionCv.notify_all();
  }
}

void TreeParallel::waitForAllNodeJobs() const {
  std::unique_lock<std::mutex> lock(fitContext_->completionMutex);
  fitContext_->completionCv.wait(lock, [this]() {
    return fitContext_->pendingNodeJobs.load(std::memory_order_acquire) == 0;
  });
}

std::function<void(std::unique_ptr<Node>)> TreeParallel::wrapNodeOnComplete(
    std::function<void(std::unique_ptr<Node>)> userCallback) const {
  // Every node job reports its result through a callback. This wrapper makes
  // sure the pending-job counter is updated even if the callback throws.
  return [this, userCallback = std::move(userCallback)](
             std::unique_ptr<Node> node) mutable {
    try {
      userCallback(std::move(node));
    } catch (...) {
      notifyNodeJobFinished();
      throw;
    }
    notifyNodeJobFinished();
  };
}

TreeParallel::SplitSearchResult TreeParallel::findBestSplitAtNode(
    const std::vector<std::size_t> &rowIndices) const {
  // At one node, every feature can be tested independently. That makes split
  // search a good place to use threads without changing the tree algorithm.
  const std::size_t featureCount = dataset_->featureNames.size();
  if (featureCount == 0) {
    return {};
  }

  std::vector<SplitSearchResult> featureResults;
  featureResults.reserve(featureCount);

  if (shouldParallelizeAttributes(featureCount)) {
    // Share the row list with all feature jobs. The rows are read-only here.
    const RowIndexList rows =
        std::make_shared<std::vector<std::size_t>>(rowIndices);
    std::vector<std::future<SplitSearchResult>> futures(featureCount);
    for (std::size_t featureIndex = 0; featureIndex < featureCount;
         ++featureIndex) {
      // One job = find the best threshold for one feature.
      futures[featureIndex] =
          fitContext_->attributeExecutor->submit([this, rows, featureIndex]() {
            return evaluateFeatureSplit(*rows, featureIndex);
          });
    }

    for (std::size_t featureIndex = 0; featureIndex < featureCount;
         ++featureIndex) {
      // get() waits for that feature job and returns its best split.
      featureResults.push_back(futures[featureIndex].get());
    }
  } else {
    // If the node has only a few features, do the same work without threads.
    for (std::size_t featureIndex = 0; featureIndex < featureCount;
         ++featureIndex) {
      featureResults.push_back(evaluateFeatureSplit(rowIndices, featureIndex));
    }
  }

  // Choosing the winning split is shared with TreeSerial.
  return reduceBestSplitSearch(std::move(featureResults), rowIndices);
}

void TreeParallel::scheduleAsyncChildren(
    std::unique_ptr<Node> parent, PartitionedRows partitions, int depth,
    std::function<void(std::unique_ptr<Node>)> onComplete) const {
  // This parent is not complete until both children have been built.
  // shared_ptr keeps this temporary state alive while child jobs run.
  auto parentHolder =
      std::make_shared<std::unique_ptr<Node>>(std::move(parent));
  auto pendingChildren = std::make_shared<std::atomic<int>>(2);
  auto childMutex = std::make_shared<std::mutex>();

  const auto attachChild = [parentHolder, pendingChildren, childMutex,
                            onComplete](bool isLeft) {
    return [parentHolder, pendingChildren, childMutex, onComplete,
            isLeft](std::unique_ptr<Node> child) {
      {
        std::lock_guard<std::mutex> lock(*childMutex);
        if (isLeft) {
          (*parentHolder)->leftChild = std::move(child);
        } else {
          (*parentHolder)->rightChild = std::move(child);
        }
      }

          // The second finished child completes the parent.
      if (pendingChildren->fetch_sub(1, std::memory_order_acq_rel) == 1) {
        onComplete(std::move(*parentHolder));
      }
    };
  };

  const auto submitChild = [&](std::vector<std::size_t> childRows,
                               bool isLeft) {
    const RowIndexList childRowsShared =
        std::make_shared<std::vector<std::size_t>>(std::move(childRows));

    // Each child subtree becomes a separate node job.
    fitContext_->pendingNodeJobs.fetch_add(1, std::memory_order_relaxed);
    fitContext_->nodeExecutor->submit([this, childRowsShared, depth, isLeft,
                                       childOnComplete = wrapNodeOnComplete(
                                           attachChild(isLeft))]() mutable {
      expandNodeAsync({childRowsShared, depth + 1, std::move(childOnComplete)});
    });
  };

  submitChild(std::move(partitions.leftRows), true);
  submitChild(std::move(partitions.rightRows), false);
}

void TreeParallel::expandNodeAsync(NodeBuildJob job) const {
  const RowIndexList &rowIndices = job.rowIndices;
  const int depth = job.depth;
  std::function<void(std::unique_ptr<Node>)> onComplete =
      std::move(job.onComplete);

  if (!shouldParallelizeNodes(rowIndices->size())) {
    // For small subtrees, use the shared recursive builder. It is simpler and
    // avoids spending thread time on tiny jobs.
    onComplete(buildNode(*rowIndices, depth));
    return;
  }

  // Build just this node first. If it splits, its children can run separately.
  NodeExpandResult expanded = expandOneNode(*rowIndices, depth);
  if (expanded.node->isLeaf) {
    onComplete(std::move(expanded.node));
    return;
  }

  scheduleAsyncChildren(std::move(expanded.node),
                        std::move(expanded.partitions), depth,
                        std::move(onComplete));
}

void TreeParallel::fit(const Dataset &dataset, const TrainingOptions &options) {
  // Shared setup is the same as TreeSerial: store data, options, and labels.
  prepareFit(dataset, options);

  // Parallel-only setup: create executors and track async node jobs.
  fitContext_ = std::make_unique<FitContext>();
  setupParallelExecutors();

  const std::vector<std::size_t> rowIndices = makeRootRowIndices();
  const BuildTimePoint buildStart = startBuildTimer();

  // This backend always starts from the node-parallel path. Small subtrees may
  // still fall back to TreeBase::buildNode when that is cheaper.
  growTreeWithNodeParallelism(rowIndices);

  finishBuildTimer(buildStart);
  // Destroy thread pools before pruning. Pruning is currently single-threaded.
  fitContext_.reset();

  finalizeFit(options);
}
