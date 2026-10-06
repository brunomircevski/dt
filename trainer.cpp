#include "trainer.h"

#include "cpu_builder.h"
#include "gpu_builder.h"
#include "pruning.h"
#include "split_rules.h"
#include "thread_pool.h"
#include "timing.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <thread>

namespace {

unsigned resolveThreads(const Options &options) {
  if (options.threads > 0) {
    return static_cast<unsigned>(options.threads);
  }
  return std::max(1u, std::thread::hardware_concurrency());
}

// Grow the full (unpruned) tree with the configured backend.
// `distinctValues` (optional) receives every feature's sorted distinct values.
std::unique_ptr<Node> growTree(const Dataset &train, const Options &options,
                               ThreadPool *pool, TrainTimings &timings,
                               std::vector<std::vector<float>> *distinctValues = nullptr) {
  if (train.classCount() == 0 || train.rowCount == 0) {
    throw std::runtime_error("Cannot train on an empty dataset.");
  }
  const dt::EntryCodec codec = makeEntryCodec(train.classCount(), train.rowCount);
  const SplitRules rules(options, train.classCount(), train.rowCount);
  std::unique_ptr<Node> root;

  if (options.backend == Backend::Cuda) {
    GpuTimings gpuTimings;
    root = gpuGrowTree(train, rules, codec, options, *pool, gpuTimings, distinctValues);
    timings.gpuSetupSeconds += gpuTimings.setupSeconds;
    timings.buildSeconds += gpuTimings.buildSeconds;
    return root;
  }

  // Presort every feature once, then grow from the sorted columns.
  const std::size_t entryCount = train.featureCount() * train.rowCount;
  std::unique_ptr<dt::Entry[]> entries(new dt::Entry[entryCount]);
  std::unique_ptr<std::uint8_t[]> goesLeft(new std::uint8_t[train.rowCount]);
  {
    ScopedTimer timer(timings.presortSeconds);
    presortColumns(train, codec, entries.get(), pool);
    if (distinctValues) {
      distinctValues->assign(train.featureCount(), {});
      auto collect = [&](std::size_t feature) {
        (*distinctValues)[feature] =
            distinctSortedValues(entries.get() + feature * train.rowCount, train.rowCount);
      };
      if (pool) {
        pool->parallelFor(train.featureCount(), collect);
      } else {
        for (std::size_t feature = 0; feature < train.featureCount(); ++feature) {
          collect(feature);
        }
      }
    }
  }
  {
    ScopedTimer timer(timings.buildSeconds);
    const Columns columns{entries.get(), train.rowCount, train.featureCount()};
    CpuTreeBuilder builder(rules, codec, train.featureCount(), goesLeft.get(), pool, options);
    builder.build(columns, 0, static_cast<std::uint32_t>(train.rowCount), 0, root);
    if (pool) {
      pool->waitIdle();
    }
  }
  return root;
}

// Breiman's choice of alpha: grow a tree on each of K folds' complement,
// measure every fold tree's test error along the main tree's pruning sequence
// (at the geometric midpoints of consecutive alphas), and take the simplest
// tree within one standard error of the best (1-SE rule).
double crossValidateAlpha(const Dataset &train, Node *mainRoot, const Options &options,
                          ThreadPool *pool, bool verbose) {
  const std::vector<double> alphas = cartPruningSequence(mainRoot, train.rowCount);
  std::vector<double> betas(alphas.size());
  for (std::size_t k = 0; k < alphas.size(); ++k) {
    // The last subtree (the root alone) is evaluated at any alpha above the
    // last one; a finite value keeps it inside the half-open intervals below.
    betas[k] = k + 1 < alphas.size() ? std::sqrt(alphas[k] * alphas[k + 1])
                                     : std::numeric_limits<double>::max();
  }

  const int folds = options.ccpFolds;
  std::vector<std::uint32_t> order(train.rowCount);
  std::iota(order.begin(), order.end(), 0u);
  std::mt19937_64 random(2024);
  std::shuffle(order.begin(), order.end(), random);

  std::vector<double> errors(betas.size() + 1, 0.0); // difference array over k
  Options foldOptions = options;
  foldOptions.cartPrune = false;
  for (int fold = 0; fold < folds; ++fold) {
    std::vector<std::uint32_t> fitRows;
    std::vector<std::uint32_t> testRows;
    for (std::size_t index = 0; index < order.size(); ++index) {
      (static_cast<int>(index % folds) == fold ? testRows : fitRows).push_back(order[index]);
    }
    std::sort(fitRows.begin(), fitRows.end());
    const Dataset foldData = selectRows(train, fitRows);
    TrainTimings ignored;
    std::unique_ptr<Node> foldRoot = growTree(foldData, foldOptions, pool, ignored);
    cartPruningSequence(foldRoot.get(), foldData.rowCount); // collapse alphas

    // For a test row, the fold tree pruned at beta predicts with the first
    // node on the row's path whose collapse alpha is <= beta. Along the path
    // the running minimum of collapse alphas only decreases, so each path
    // node owns one interval of beta.
    for (std::uint32_t row : testRows) {
      const std::uint16_t truth = train.labels[row];
      double upper = INFINITY;
      for (const Node *node = foldRoot.get();;) {
        const double lower = node->isLeaf() ? -INFINITY : std::min(upper, node->errors);
        if (lower < upper && node->label != truth) {
          // betas in [lower, upper) misclassify this row.
          const auto first = std::lower_bound(betas.begin(), betas.end(), lower);
          const auto last = std::lower_bound(betas.begin(), betas.end(), upper);
          errors[static_cast<std::size_t>(first - betas.begin())] += 1;
          errors[static_cast<std::size_t>(last - betas.begin())] -= 1;
        }
        if (node->isLeaf()) {
          break;
        }
        upper = lower;
        node = train.value(row, static_cast<std::size_t>(node->feature)) <= node->threshold
                   ? node->left.get()
                   : node->right.get();
      }
    }
  }

  const double n = static_cast<double>(train.rowCount);
  std::vector<double> risk(betas.size());
  double running = 0.0;
  std::size_t bestIndex = 0;
  for (std::size_t k = 0; k < betas.size(); ++k) {
    running += errors[k];
    risk[k] = running / n;
    if (risk[k] < risk[bestIndex]) {
      bestIndex = k;
    }
  }
  const double standardError = std::sqrt(risk[bestIndex] * (1.0 - risk[bestIndex]) / n);
  std::size_t chosen = bestIndex;
  for (std::size_t k = bestIndex; k < betas.size(); ++k) {
    if (risk[k] <= risk[bestIndex] + standardError) {
      chosen = k;
    }
  }
  if (verbose) {
    std::cout << "  CV over " << alphas.size() << " subtrees: best alpha "
              << alphas[bestIndex] << " (cv error " << risk[bestIndex] << "), 1-SE choice "
              << alphas[chosen] << " (cv error " << risk[chosen] << ")\n";
  }
  return alphas[chosen];
}

} // namespace

DecisionTree trainTree(const Dataset &train, const Options &options, TrainTimings &timings,
                       bool verbose) {
  validateOptions(options);
  const unsigned threads = resolveThreads(options);
  std::unique_ptr<ThreadPool> pool;
  if (options.backend != Backend::Serial && threads > 1) {
    pool = std::make_unique<ThreadPool>(threads - 1);
  } else if (options.backend == Backend::Cuda) {
    pool = std::make_unique<ThreadPool>(1);
  }

  DecisionTree tree;
  tree.featureNames = train.featureNames;
  tree.classNames = train.classNames;
  std::vector<std::vector<float>> distinctValues;
  const bool needDistinct = options.algorithm == Algorithm::C45;
  if (options.loadTreePath.empty()) {
    tree.root = growTree(train, options, pool.get(), timings,
                         needDistinct ? &distinctValues : nullptr);
  } else {
    std::ifstream input(options.loadTreePath);
    if (!input) {
      throw std::runtime_error("Could not read " + options.loadTreePath);
    }
    tree.root = readTree(input, train);
  }

  ScopedTimer timer(timings.pruneSeconds);
  if (options.algorithm == Algorithm::C45) {
    c45CollapseUselessSplits(tree.root.get());
    c45UseTrainingValueThresholds(tree.root.get(), train, distinctValues);
    if (options.c45Prune) {
      if (verbose) {
        std::cout << "  unpruned: " << tree.nodeCount() << " nodes\n";
      }
      c45PessimisticPrune(tree.root, train, options.c45ConfidenceFactor,
                          options.c45SubtreeRaising, pool.get());
    }
  } else if (options.cartPrune) {
    if (verbose) {
      std::cout << "  unpruned: " << tree.nodeCount() << " nodes\n";
    }
    double alpha = options.ccpAlpha;
    if (options.ccpFolds > 1) {
      alpha = crossValidateAlpha(train, tree.root.get(), options, pool.get(), verbose);
    }
    cartCostComplexityPrune(tree.root.get(), alpha, train.rowCount);
  }
  return tree;
}
