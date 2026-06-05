#include "pruning/pruning.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

namespace {

double clampConfidenceFactor(double value) {
  return std::max(0.0, std::min(0.5, value));
}

double inverseStandardNormalCdf(double probability) {
  const double a1 = -39.6968302866538;
  const double a2 = 220.946098424521;
  const double a3 = -275.928510446969;
  const double a4 = 138.357751867269;
  const double a5 = -30.6647980661472;
  const double a6 = 2.50662827745924;

  const double b1 = -54.4760987982241;
  const double b2 = 161.585836858041;
  const double b3 = -155.698979859887;
  const double b4 = 66.8013118877197;
  const double b5 = -13.2806815528857;

  const double c1 = -0.00778489400243029;
  const double c2 = -0.322396458041136;
  const double c3 = -2.40075827716184;
  const double c4 = -2.54973253934373;
  const double c5 = 4.37466414146497;
  const double c6 = 2.93816398269878;

  const double d1 = 0.00778469570904146;
  const double d2 = 0.32246712907004;
  const double d3 = 2.445134137143;
  const double d4 = 3.75440866190742;

  const double pLow = 0.02425;
  const double pHigh = 1.0 - pLow;

  if (probability <= 0.0) {
    return -std::numeric_limits<double>::infinity();
  }
  if (probability >= 1.0) {
    return std::numeric_limits<double>::infinity();
  }

  if (probability < pLow) {
    const double q = std::sqrt(-2.0 * std::log(probability));
    return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
           ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
  }

  if (probability <= pHigh) {
    const double q = probability - 0.5;
    const double r = q * q;
    return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
           (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
  }

  const double q = std::sqrt(-2.0 * std::log(1.0 - probability));
  return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
          ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
}

double addErrs(double sampleCount, double observedErrors, double confidenceFactor) {
  if (confidenceFactor > 0.5) {
    return 0.0;
  }

  if (observedErrors < 1.0) {
    const double base =
        sampleCount * (1.0 - std::pow(confidenceFactor, 1.0 / sampleCount));
    if (observedErrors == 0.0) {
      return base;
    }
    return base + observedErrors * (addErrs(sampleCount, 1.0, confidenceFactor) - base);
  }

  if (observedErrors + 0.5 >= sampleCount) {
    return std::max(sampleCount - observedErrors, 0.0);
  }

  const double z = inverseStandardNormalCdf(1.0 - confidenceFactor);
  const double errorRate = (observedErrors + 0.5) / sampleCount;
  const double zSquared = z * z;
  const double upperBoundRate =
      (errorRate + zSquared / (2.0 * sampleCount) +
       z * std::sqrt((errorRate / sampleCount) -
                     (errorRate * errorRate / sampleCount) +
                     (zSquared / (4.0 * sampleCount * sampleCount)))) /
      (1.0 + zSquared / sampleCount);

  return std::max(upperBoundRate * sampleCount - observedErrors, 0.0);
}

double estimatePessimisticLeafErrorCount(std::size_t observedErrors,
                                         std::size_t sampleCount,
                                         double confidenceFactor) {
  if (sampleCount == 0) {
    return 0.0;
  }

  const double clampedFactor = clampConfidenceFactor(confidenceFactor);
  const double n = static_cast<double>(sampleCount);
  const double errors = static_cast<double>(observedErrors);
  return errors + addErrs(n, errors, clampedFactor);
}

std::vector<std::size_t> allRowIndices(const Dataset &dataset) {
  std::vector<std::size_t> rowIndices(dataset.samples.size());
  for (std::size_t index = 0; index < rowIndices.size(); ++index) {
    rowIndices[index] = index;
  }
  return rowIndices;
}

} // namespace

void pruneTree(TreeBase &tree, const Options &options) {
  tree.pruneTimeSeconds_ = 0.0;

  if (!tree.root_ || options.pruningMode == PruningMode::None) {
    return;
  }

  if (options.pruningMode == PruningMode::PessimisticError) {
    const auto pruneStart = std::chrono::steady_clock::now();
    const double confidenceFactor = options.pruningConfidenceFactor;
    const std::vector<std::size_t> rowIndices = allRowIndices(*tree.dataset_);

    std::function<double(const Node *, const std::vector<std::size_t> &)>
        estimateSubtreeErrors;
    estimateSubtreeErrors = [&tree, confidenceFactor, &estimateSubtreeErrors](
                                const Node *subtree,
                                const std::vector<std::size_t> &indices) -> double {
      if (!subtree) {
        return 0.0;
      }

      if (subtree->isLeaf) {
        std::size_t observedErrors = 0;
        for (std::size_t rowIndex : indices) {
          if (tree.dataset_->samples[rowIndex].label != subtree->leafLabel) {
            ++observedErrors;
          }
        }
        return estimatePessimisticLeafErrorCount(observedErrors, indices.size(),
                                                 confidenceFactor);
      }

      const TreeBase::PartitionedRows parts =
          tree.partitionRows(indices, subtree->featureIndex, subtree->threshold);

      return estimateSubtreeErrors(subtree->leftChild.get(), parts.leftRows) +
             estimateSubtreeErrors(subtree->rightChild.get(), parts.rightRows);
    };

    std::function<void(std::unique_ptr<Node> &, const std::vector<std::size_t> &)>
        pruneNode = [&](std::unique_ptr<Node> &node,
                        const std::vector<std::size_t> &indices) {
          if (!node || node->isLeaf) {
            return;
          }

          const TreeBase::PartitionedRows partitions =
              tree.partitionRows(indices, node->featureIndex, node->threshold);

          pruneNode(node->leftChild, partitions.leftRows);
          pruneNode(node->rightChild, partitions.rightRows);

          const std::string majorityLabel = tree.getMajorityLabel(indices);
          std::size_t leafCorrect = 0;
          for (std::size_t rowIndex : indices) {
            if (tree.dataset_->samples[rowIndex].label == majorityLabel) {
              ++leafCorrect;
            }
          }

          const std::size_t observedLeafErrors = indices.size() - leafCorrect;
          const double estimatedLeafErrors = estimatePessimisticLeafErrorCount(
              observedLeafErrors, indices.size(), confidenceFactor);
          const double estimatedSubtreeErrors =
              estimateSubtreeErrors(node.get(), indices);

          if (estimatedLeafErrors <= estimatedSubtreeErrors + tree.options_.epsilon) {
            node = Node::createLeaf(majorityLabel, indices.size());
          }
        };

    pruneNode(tree.root_, rowIndices);

    tree.pruneTimeSeconds_ = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - pruneStart)
                                 .count();
    return;
  }

  if (options.pruningMode == PruningMode::CostComplexity) {
    const auto pruneStart = std::chrono::steady_clock::now();
    const double ccpAlpha = options.ccpAlpha;
    const std::vector<std::size_t> rowIndices = allRowIndices(*tree.dataset_);

    std::function<void(std::unique_ptr<Node> &, const std::vector<std::size_t> &)>
        pruneNode = [&](std::unique_ptr<Node> &node,
                        const std::vector<std::size_t> &indices) {
          if (!node || node->isLeaf) {
            return;
          }

          const TreeBase::PartitionedRows partitions =
              tree.partitionRows(indices, node->featureIndex, node->threshold);

          pruneNode(node->leftChild, partitions.leftRows);
          pruneNode(node->rightChild, partitions.rightRows);

          if (node->isLeaf) {
            return;
          }

          const std::string majorityLabel = tree.getMajorityLabel(indices);
          std::size_t errorsAsLeaf = 0;
          for (std::size_t rowIndex : indices) {
            if (tree.dataset_->samples[rowIndex].label != majorityLabel) {
              ++errorsAsLeaf;
            }
          }

          const std::size_t errorsInSubtree =
              indices.size() - tree.countCorrectPredictions(node.get(), indices);
          const std::size_t leavesInSubtree = tree.countLeafNodes(node.get());

          const double costAsLeaf =
              static_cast<double>(errorsAsLeaf) + ccpAlpha;
          const double costAsSubtree =
              static_cast<double>(errorsInSubtree) +
              ccpAlpha * static_cast<double>(leavesInSubtree);

          if (costAsLeaf <= costAsSubtree + tree.options_.epsilon) {
            node = Node::createLeaf(majorityLabel, indices.size());
          }
        };

    pruneNode(tree.root_, rowIndices);

    tree.pruneTimeSeconds_ = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - pruneStart)
                                 .count();
  }
}
