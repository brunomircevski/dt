#pragma once

#include "tree_base.h"
#include "task_executor.h"

#include <atomic>
#include <memory>

// Multi-threaded tree: feature jobs search splits; node jobs build child subtrees.
class TreeParallel : public TreeBase {
public:
  void fit(const Dataset &dataset,
           const TrainingOptions &options = TrainingOptions{}) override;

  SplitSearchResult findBestSplitAtNode(
      const std::vector<std::size_t> &rowIndices) const override;

private:
  struct FitContext {
    std::unique_ptr<TaskExecutor> featureExecutor;
    std::unique_ptr<TaskExecutor> nodeExecutor;
    std::atomic<std::size_t> activeNodeTasks{0};
    std::size_t maxNodeTasks = 0;
  };

  mutable std::unique_ptr<FitContext> fitContext_;

  void setupParallelExecutors();
  bool shouldParallelizeAttributes(std::size_t featureCount) const;
  bool shouldParallelizeNodes(std::size_t rowCount) const;
  bool tryStartNodeTask() const;
  void finishNodeTask() const;

  std::unique_ptr<Node> buildNodeParallel(const std::vector<std::size_t> &rowIndices,
                                          int depth) const;
};
