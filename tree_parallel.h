#pragma once

#include "tree_base.h"
#include "task_executor.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>

// Multi-threaded tree. Uses both node-level and attribute-level parallelism.
class TreeParallel : public TreeBase {
public:
  void fit(const Dataset &dataset,
           const TrainingOptions &options = TrainingOptions{}) override;

  SplitSearchResult findBestSplitAtNode(
      const std::vector<std::size_t> &rowIndices) const override;

private:
  struct FitContext {
    std::unique_ptr<TaskExecutor> attributeExecutor;
    std::unique_ptr<TaskExecutor> nodeExecutor;
    std::atomic<std::size_t> pendingNodeJobs{0};
    std::mutex completionMutex;
    std::condition_variable completionCv;
  };

  using RowIndexList = std::shared_ptr<const std::vector<std::size_t>>;

  struct NodeBuildJob {
    RowIndexList rowIndices;
    int depth = 0;
    std::function<void(std::unique_ptr<Node>)> onComplete;
  };

  mutable std::unique_ptr<FitContext> fitContext_;

  void setupParallelExecutors();
  void growTreeWithNodeParallelism(const std::vector<std::size_t> &rowIndices);

  bool shouldParallelizeAttributes(std::size_t featureCount) const;
  bool shouldParallelizeNodes(std::size_t rowCount) const;
  void waitForAllNodeJobs() const;
  void notifyNodeJobFinished() const;
  std::function<void(std::unique_ptr<Node>)>
  wrapNodeOnComplete(std::function<void(std::unique_ptr<Node>)> userCallback) const;

  void scheduleAsyncChildren(std::unique_ptr<Node> parent,
                             PartitionedRows partitions, int depth,
                             std::function<void(std::unique_ptr<Node>)> onComplete) const;
  void expandNodeAsync(NodeBuildJob job) const;
};
