#include "app/trainer.h"

#include "algo/c45_pruning.h"
#include "algo/cart_pruning.h"
#include "algo/split_rules.h"
#include "build/grower.h"
#include "core/timing.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace dt {

namespace {

// Breiman's choice of alpha: grow a tree on each of K folds' complement,
// measure every fold tree's test error along the main tree's pruning sequence
// (at the geometric midpoints of consecutive alphas), and take the simplest
// tree within one standard error of the best (1-SE rule).
double crossValidateAlpha(const Dataset &train, const Tree &mainTree, const Options &options,
                          Grower &grower, ThreadPool *pool, bool verbose) {
  const std::vector<double> alphas = cartPruningSequence(mainTree).alphas;
  std::vector<double> betas(alphas.size());
  for (std::size_t k = 0; k < alphas.size(); ++k) {
    // The last subtree (the root alone) is evaluated at any alpha above the
    // last one; a finite value keeps it inside the half-open intervals below.
    betas[k] = k + 1 < alphas.size() ? std::sqrt(alphas[k] * alphas[k + 1])
                                     : std::numeric_limits<double>::max();
  }

  const int folds = options.cart.folds;
  std::vector<std::uint32_t> order(train.rowCount);
  std::iota(order.begin(), order.end(), 0u);
  std::mt19937_64 random(2024);
  std::shuffle(order.begin(), order.end(), random);

  std::vector<int> foldOf(train.rowCount);
  for (std::size_t index = 0; index < order.size(); ++index) {
    foldOf[order[index]] = static_cast<int>(index % folds);
  }

  std::vector<double> errors(betas.size() + 1, 0.0); // difference array over k
  for (int fold = 0; fold < folds; ++fold) {
    std::vector<std::uint32_t> fitRows; // increasing, as grow() wants them
    std::vector<std::uint32_t> testRows;
    for (std::uint32_t row = 0; row < train.rowCount; ++row) {
      (foldOf[row] == fold ? testRows : fitRows).push_back(row);
    }
    const SplitRules rules(options, train.classCount(), fitRows.size());
    GrowTimings ignored;
    const Tree foldTree = grower.grow(rules, fitRows, ignored);
    const std::vector<double> collapse = cartPruningSequence(foldTree).collapseAlpha;

    // For a test row, the fold tree pruned at beta predicts with the first
    // node on the row's path whose collapse alpha is <= beta. Along the path
    // the running minimum of collapse alphas only decreases, so each path
    // node owns one interval of beta. Chunks of rows run in parallel, each
    // with its own difference array.
    const std::size_t chunks = pool ? 4 * (pool->workerCount() + 1) : 1;
    std::vector<std::vector<double>> chunkErrors(chunks);
    parallelFor(pool, chunks, [&](std::size_t chunk) {
      std::vector<double> &local = chunkErrors[chunk];
      local.assign(errors.size(), 0.0);
      const std::size_t end = testRows.size() * (chunk + 1) / chunks;
      for (std::size_t index = testRows.size() * chunk / chunks; index < end; ++index) {
        const std::uint32_t row = testRows[index];
        const std::uint16_t truth = train.labels[row];
        double upper = INFINITY;
        for (std::uint32_t id = 0;;) {
          const Node &node = foldTree.nodes[id];
          const double lower = node.isLeaf() ? -INFINITY : std::min(upper, collapse[id]);
          if (lower < upper && node.label != truth) {
            // betas in [lower, upper) misclassify this row.
            const auto first = std::lower_bound(betas.begin(), betas.end(), lower);
            const auto last = std::lower_bound(betas.begin(), betas.end(), upper);
            local[static_cast<std::size_t>(first - betas.begin())] += 1;
            local[static_cast<std::size_t>(last - betas.begin())] -= 1;
          }
          if (node.isLeaf()) {
            break;
          }
          upper = lower;
          id = train.value(row, static_cast<std::size_t>(node.feature)) <= node.threshold
                   ? node.left
                   : node.right;
        }
      }
    });
    for (const std::vector<double> &local : chunkErrors) {
      for (std::size_t k = 0; k < errors.size(); ++k) {
        errors[k] += local[k];
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

Tree trainTree(const Dataset &train, const Options &options, ThreadPool *pool,
               TrainTimings &timings, bool verbose) {
  validateOptions(options);
  if (train.classCount() == 0 || train.rowCount == 0) {
    throw std::runtime_error("Cannot train on an empty dataset.");
  }
  ThreadPool *buildPool = options.backend == Backend::Serial ? nullptr : pool;
  const bool crossValidate = options.algorithm == Algorithm::Cart &&
                             options.cart.pruning == CartPruning::CrossValidation;

  std::unique_ptr<Grower> grower;
  if (options.backend == Backend::Cuda) {
    if (!pool) {
      throw std::logic_error("The Cuda backend needs a thread pool");
    }
    grower = makeGpuGrower(train, options, *pool, crossValidate, timings.gpuSetupSeconds);
  } else {
    grower = makeCpuGrower(train, options, buildPool, crossValidate);
  }

  const SplitRules rules(options, train.classCount(), train.rowCount);
  GrowTimings growTimings;
  std::vector<float> sortedValues; // C4.5's thresholds need them
  Tree tree = grower->grow(rules, {}, growTimings,
                           options.algorithm == Algorithm::C45 ? &sortedValues : nullptr);
  timings.presortSeconds += growTimings.prepareSeconds;
  timings.buildSeconds += growTimings.buildSeconds;
  if (verbose && (options.algorithm == Algorithm::C45 ? options.c45.prune
                                                      : options.cart.pruning != CartPruning::None)) {
    std::cout << "  unpruned: " << tree.nodeCount() << " nodes\n";
  }

  if (options.algorithm == Algorithm::C45) {
    ScopedTimer timer(timings.pruneSeconds);
    c45CollapseUselessSplits(tree);
    c45UseTrainingValueThresholds(tree, sortedValues, train.rowCount);
    if (options.c45.prune) {
      c45PessimisticPrune(tree, train, options.c45.confidence, buildPool);
    }
    return tree;
  }

  double alpha = options.cart.alpha;
  if (crossValidate) {
    ScopedTimer timer(timings.cvSeconds);
    alpha = crossValidateAlpha(train, tree, options, *grower, buildPool, verbose);
  }
  if (options.cart.pruning != CartPruning::None) {
    ScopedTimer timer(timings.pruneSeconds);
    cartCostComplexityPrune(tree, alpha);
  }
  return tree;
}

} // namespace dt
